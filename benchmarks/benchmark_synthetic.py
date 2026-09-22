"""
benchmarks/benchmark_synthetic.py  v2.0 -- omics-regime edition

BENCHMARK SPEC IS LOCKED AT MODULE LEVEL -- see BENCHMARK_SPEC below.
All design decisions (scenarios, metrics, seeds, statistical tests) are
declared before any results are computed.  Modifying BENCHMARK_SPEC after
reviewing results constitutes p-hacking and invalidates the benchmark.

Data regime targeted
--------------------
Real-world processed omics data: small-n / large-p, strong within-pathway
feature correlation, very sparse signal, and clinical-scale class imbalance.

    n  =  50-200  samples   (clinical cohort)
    p  = 500-10000 features (post-QC genomics / proteomics)
    n_informative = 5-20    (0.1-2 % of features)
    within-block correlation = 0.70-0.92 (gene pathways, protein families)
    class imbalance up to 85/15 (rare-disease case/control)

Data generation
---------------
Features are drawn block-by-block from multivariate Gaussians.  Each block
of `block_size` features shares within-block Pearson correlation `rho`,
simulating a co-expressed gene set or protein complex.  `n_informative`
blocks each have one "lead" feature that receives a +/-signal_strength mean
shift depending on class label.  All other features are pure noise.

The ground truth is therefore the set of lead-feature indices:
    {0, block_size, 2*block_size, ..., (n_informative-1)*block_size}

Statistical testing
-------------------
Primary test : Friedman (non-parametric, all methods simultaneously).
Post-hoc     : Pairwise Wilcoxon signed-rank (paired by scenarioxseed)
               vs SelectOmics as reference, with Holm-Bonferroni correction.
Effect size  : Cliff's delta (negligible <0.147, small <0.33,
               medium <0.474, large >=0.474).

Design splits
-------------
DEVELOPMENT_SCENARIOS (11) : used freely during iteration / debugging.
HOLDOUT_SCENARIOS      (2) : run exactly once for final reporting.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

# Sibling module: importable whether this file is run directly or imported.
_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

from _parallel import (                     # noqa: E402
    describe_plan,
    plan_parallelism,
    run_tasks,
)

# -- Optional dependencies ------------------------------------------------------

try:
    from boruta import BorutaPy
    _BORUTA_AVAILABLE = True
except ImportError:
    _BORUTA_AVAILABLE = False

try:
    import shap as _shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

# -- SelectOmics import ---------------------------------------------------------
# File lives at  package/benchmarks/benchmark_synthetic.py
# parents[1]  =  package/   ->  'from SelectOmics import ...' resolves correctly
_PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))

try:
    from SelectOmics import SelectOmicsConfig, SelectOmicsPipeline
    _SO_AVAILABLE = True
except ImportError:
    _SO_AVAILABLE = False
    warnings.warn("SelectOmics not importable -- SelectOmics method skipped.", stacklevel=1)


def _so_default(field_name: str):
    """
    The shipping default for a SelectOmicsConfig field.

    Used so the harness never hardcodes a default that can drift out of step
    with the package. If a benchmark is meant to characterise what users get,
    it has to read what users get.
    """
    return SelectOmicsConfig.__dataclass_fields__[field_name].default

warnings.filterwarnings("ignore")
for _lg in ("SelectOmics", "shap", "xgboost", "sklearn"):
    logging.getLogger(_lg).setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# -- Optimised SelectOmics config (produced by SelectOmics_Config_Optimization.ipynb) --
# File is written to opt_results/ in the same directory as this script.
_OPT_CONFIG_PATH: Path = (
    Path(__file__).resolve().parent / "opt_results" / "selectomics_recommended_config.json"
)


def _load_optimized_params() -> Optional[Dict]:
    """
    Load the recommended SelectOmics parameters from the optimisation notebook
    output (``opt_results/selectomics_recommended_config.json``).

    Returns None if the file does not exist (before the optimisation has been
    run) -- callers fall back to sensible defaults.
    """
    if not _SO_AVAILABLE or not _OPT_CONFIG_PATH.exists():
        return None
    try:
        with open(_OPT_CONFIG_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        params = data.get("params")
        if params:
            logger.info(
                "Loaded optimised SelectOmics config from %s "
                "(composite=%.4f  F1=%.4f)",
                _OPT_CONFIG_PATH,
                data.get("metrics", {}).get("composite", float("nan")),
                data.get("metrics", {}).get("mean_f1",   float("nan")),
            )
        return params
    except Exception as exc:
        logger.warning("Failed to load optimised SO config from %s: %s", _OPT_CONFIG_PATH, exc)
        return None


# ==============================================================================
# 1.  LOCKED BENCHMARK SPECIFICATION
# ==============================================================================

BENCHMARK_SPEC: Dict = {
    "schema_version":        "2.0",
    "locked":                True,
    # Primary metric for each evaluation axis -- declared before any run.
    # Do not switch to a secondary metric because the primary is unfavourable.
    "primary_metrics": {
        "ground_truth_recovery":  "f1",
        "stability":              "kuncheva_stability",
        "downstream_performance": "test_auc",
    },
    "secondary_metrics": [
        "precision", "recall", "jaccard",
        "n_selected", "reduction_ratio", "mean_feature_corr",
    ],
    # Fixed seeds -- do not change after any run has been inspected.
    "seeds":               [0, 1, 2, 3, 4],
    "test_size":           0.25,
    "statistical_test":    "friedman_then_wilcoxon_signed_rank_paired",
    "correction":          "holm_bonferroni",
    "effect_size_metric":  "cliffs_delta",
    "alpha":               0.05,
    "data_regime":         "small_n_large_p_omics_n50-200_p500-10000",
    "development_scenarios": [
        "omics_standard", "omics_tiny_n", "omics_high_dim",
        "omics_imbalanced", "omics_genomics", "adversarial_easy",
        # Added in 0.7.0, declared before the 0.7.0 run was made. The five
        # below were specified from the design gaps in the original six, not
        # from any 0.7.0 result: no null control, no interaction signal, no
        # multiclass, and two factors (correlation, imbalance) that could not
        # be separated from dataset shape.
        "omics_lowcorr", "omics_imbalanced_std", "null_control",
        "omics_interaction", "omics_multiclass",
    ],
    "holdout_scenarios": [
        "omics_ultra_sparse", "adversarial_large_n",
    ],
    "integrity_note": (
        "Scenarios, metrics, seeds, and tests are locked before the first run. "
        "Do not add scenarios, change seeds, or switch primary metrics after "
        "inspecting any results -- doing so constitutes p-hacking."
    ),
}


# ==============================================================================
# 2.  SCENARIO DEFINITIONS
# ==============================================================================

@dataclass
class OmicsScenario:
    """
    Specification for one synthetic omics-like dataset.

    Parameters
    ----------
    name : str
    n_samples : int
    n_features : int
        Total feature count.  Must be divisible by block_size; if not,
        the last partial block is generated with fewer features.
    n_informative : int
        Number of blocks with a discriminative lead feature.
        Must satisfy  n_informative <= n_features // block_size.
    block_size : int
        Features per correlation block (simulates a gene pathway).
    within_block_corr : float
        Pearson correlation between features in the same block [0, 1).
    class_imbalance : float
        Fraction of the *positive* (minority) class.  0.5 = balanced.
    signal_strength : float
        Mean shift +/-signal_strength applied to lead features.
        With unit-variance features, Cohen's d = 2 * signal_strength.
    n_classes : int
        Number of outcome classes.  2 is binary; >2 generates a multiclass
        problem in which each informative block lead is a marker for one
        class (block b marks class b % n_classes).  Every AUC in the harness
        becomes macro one-vs-rest, matching what the pipeline reports.
    signal_mode : str
        'additive'    -- lead features carry a marginal mean shift.  A sparse
                         linear model is the correct model for this data.
        'interaction' -- the outcome depends on products of lead-feature
                         PAIRS, optionally on top of a main effect.  Requires
                         an even n_informative.
    interaction_ratio : float
        Interaction mode only.  Fraction of the latent signal carried by the
        pair products, the remainder being an ordinary main effect.

        ``1.0`` is a PURE interaction, and is a ceiling result rather than a
        comparison: if no single feature's distribution differs by class, no
        method that scores features one at a time can find it, and measurement
        confirms every method in this suite scores at or near zero recovery
        regardless of n, p or signal strength.  A scenario on which everything
        ties tells you nothing about the methods.

        Values below 1.0 give a MIXED signal, which is both the more realistic
        model of biology (epistasis on top of main effects) and the one that
        discriminates: the main effect makes the features findable at all,
        while the interaction rewards a model that can use it.
    split : str
        'development' or 'holdout'.
    description : str
    """
    name:              str
    n_samples:         int
    n_features:        int
    n_informative:     int
    block_size:        int
    within_block_corr: float
    class_imbalance:   float
    signal_strength:   float
    split:             str
    description:       str
    n_classes:         int = 2
    signal_mode:       str = "additive"
    interaction_ratio: float = 1.0

    def __post_init__(self) -> None:
        if self.signal_mode not in ("additive", "interaction"):
            raise ValueError(
                f"Scenario '{self.name}': signal_mode must be 'additive' or "
                f"'interaction', got {self.signal_mode!r}."
            )
        if not 0.0 <= self.interaction_ratio <= 1.0:
            raise ValueError(
                f"Scenario '{self.name}': interaction_ratio must be in "
                f"[0, 1], got {self.interaction_ratio}."
            )
        if self.signal_mode == "interaction" and self.n_classes > 2:
            raise ValueError(
                f"Scenario '{self.name}': interaction signal is defined for "
                f"binary outcomes only (the pair coupling has a single sign)."
            )
        if self.signal_mode == "interaction" and self.n_informative % 2:
            raise ValueError(
                f"Scenario '{self.name}': interaction signal is carried by "
                f"PAIRS of lead features, so n_informative must be even, "
                f"got {self.n_informative}."
            )
        if self.n_classes < 2:
            raise ValueError(
                f"Scenario '{self.name}': n_classes must be >= 2, "
                f"got {self.n_classes}."
            )
        if self.n_classes > 2 and self.class_imbalance != 0.5:
            raise ValueError(
                f"Scenario '{self.name}': class_imbalance applies to the "
                f"binary case only; multiclass scenarios are generated "
                f"balanced. Set class_imbalance=0.5."
            )
        if self.n_informative < 0:
            raise ValueError(
                f"Scenario '{self.name}': n_informative must be >= 0, "
                f"got {self.n_informative}."
            )
        n_blocks = self.n_features // self.block_size
        if self.n_informative > n_blocks:
            raise ValueError(
                f"Scenario '{self.name}': n_informative={self.n_informative} "
                f"exceeds n_blocks={n_blocks} "
                f"(n_features={self.n_features}, block_size={self.block_size})."
            )
        if self.n_classes > 2 and self.n_samples < 2 * self.n_classes:
            raise ValueError(
                f"Scenario '{self.name}': n_samples={self.n_samples} is too "
                f"few for {self.n_classes} classes."
            )
        n_minority = max(2, int(self.n_samples * self.class_imbalance))
        if n_minority < 2:
            raise ValueError(
                f"Scenario '{self.name}': fewer than 2 minority samples "
                f"(n={self.n_samples}, imbalance={self.class_imbalance})."
            )

    @property
    def signal_density(self) -> float:
        """Fraction of features that are truly informative (lead features only)."""
        return self.n_informative / self.n_features

    @property
    def true_feature_indices(self) -> Set[int]:
        """Ground-truth indices: first feature of each informative block."""
        return {b * self.block_size for b in range(self.n_informative)}


# -- Development scenarios (11) --------------------------------------------------

SCENARIOS: List[OmicsScenario] = [
    # 1. Standard transcriptomics cohort
    OmicsScenario(
        name="omics_standard",
        n_samples=100, n_features=2000, n_informative=10,
        block_size=20, within_block_corr=0.70,
        class_imbalance=0.50, signal_strength=1.0,
        split="development",
        description="n=100 * p=2000 * 10 informative (0.5%) * rho=0.70 * balanced",
    ),
    # 2. Very small cohort (rare disease / pilot study)
    OmicsScenario(
        name="omics_tiny_n",
        n_samples=50, n_features=500, n_informative=10,
        block_size=20, within_block_corr=0.70,
        class_imbalance=0.50, signal_strength=1.2,
        split="development",
        description="n=50 * p=500 * 10 informative * rho=0.70 * balanced",
    ),
    # 3. Proteomics / high-dim omics (ultra-sparse signal)
    OmicsScenario(
        name="omics_high_dim",
        n_samples=100, n_features=5000, n_informative=5,
        block_size=20, within_block_corr=0.80,
        class_imbalance=0.50, signal_strength=1.0,
        split="development",
        description="n=100 * p=5000 * 5 informative (0.1%) * rho=0.80 * balanced",
    ),
    # 4. Case/control with class imbalance
    OmicsScenario(
        name="omics_imbalanced",
        n_samples=200, n_features=1000, n_informative=15,
        block_size=20, within_block_corr=0.70,
        class_imbalance=0.15, signal_strength=1.0,
        split="development",
        description="n=200 * p=1000 * 15 informative * rho=0.70 * 85/15",
    ),
    # 5. Genomics scale (WES / methylation panel)
    OmicsScenario(
        name="omics_genomics",
        n_samples=200, n_features=10000, n_informative=10,
        block_size=50, within_block_corr=0.70,
        class_imbalance=0.50, signal_strength=1.0,
        split="development",
        description="n=200 * p=10000 * 10 informative (0.1%) * rho=0.70 * balanced",
    ),
    # 6. Adversarial: large n, low p, weak correlation -- LASSO should win
    OmicsScenario(
        name="adversarial_easy",
        n_samples=500, n_features=200, n_informative=20,
        block_size=10, within_block_corr=0.10,
        class_imbalance=0.50, signal_strength=2.0,
        split="development",
        description="n=500 * p=200 * 20 informative * rho=0.10 * strong signal -- LASSO regime",
    ),
    # -- Added in 0.7.0 -----------------------------------------------------
    #
    # The original six vary five factors at once between the omics scenarios
    # and adversarial_easy (rho 0.70 vs 0.10, n/p 0.05 vs 2.5, block 20 vs 10,
    # signal 1.0 vs 2.0, density 0.005 vs 0.10), so "SelectOmics loses on
    # adversarial" could not be attributed to any one of them. The first two
    # below change ONE factor from omics_standard. The last three cover
    # regimes the suite could not express at all.

    # 7. omics_standard with the adversarial correlation, nothing else changed.
    #    Isolates block correlation, which is exactly what Step 1's class-aware
    #    filter acts on.
    OmicsScenario(
        name="omics_lowcorr",
        n_samples=100, n_features=2000, n_informative=10,
        block_size=20, within_block_corr=0.10,
        class_imbalance=0.50, signal_strength=1.0,
        split="development",
        description="omics_standard with rho=0.10 -- isolates correlation from n/p and signal",
    ),
    # 8. omics_standard with imbalance, nothing else changed. The existing
    #    omics_imbalanced also changes n, p and n_informative, so it cannot
    #    separate imbalance from shape.
    OmicsScenario(
        name="omics_imbalanced_std",
        n_samples=100, n_features=2000, n_informative=10,
        block_size=20, within_block_corr=0.70,
        class_imbalance=0.15, signal_strength=1.0,
        split="development",
        description="omics_standard at 85/15 -- isolates imbalance from shape",
    ),
    # 9. Null control: no informative features at all. Any selection is a false
    #    positive. F1 is undefined here by construction, so this scenario is
    #    read on n_selected. Measured at n=60/p=400 before being added:
    #    SelectOmics 1-15 features, RFECV 5-20, LASSO 25-33, ElasticNet 52-60,
    #    RF_Importance 142-150. The spread is the point.
    OmicsScenario(
        name="null_control",
        n_samples=100, n_features=2000, n_informative=0,
        block_size=20, within_block_corr=0.70,
        class_imbalance=0.50, signal_strength=0.0,
        split="development",
        description="n=100 * p=2000 * NO signal -- every selected feature is a false positive",
    ),
    # 10. Interaction signal. Lead features predict only in pairs and have
    #     ~zero marginal association, so the rest of the suite's implicit
    #     assumption (that a sparse linear model is the right model) is false
    #     here. This is where a tree-based consensus method should win, and
    #     the suite previously had no way to show it.
    OmicsScenario(
        name="omics_interaction",
        n_samples=200, n_features=2000, n_informative=10,
        block_size=20, within_block_corr=0.70,
        class_imbalance=0.50, signal_strength=1.5,
        split="development", signal_mode="interaction", interaction_ratio=0.6,
        description="n=200 * p=2000 * 10 informative in 5 pairs * 60% of signal from interaction",
    ),
    # 11. Multiclass. Every other scenario is binary, yet the flagship examples
    #     (GBM, OV) are multiclass and the whole metric stack is macro-OVR.
    #     Each informative block marks one class, as subtype markers do.
    OmicsScenario(
        name="omics_multiclass",
        n_samples=200, n_features=2000, n_informative=12,
        block_size=20, within_block_corr=0.70,
        class_imbalance=0.50, signal_strength=1.2,
        split="development", n_classes=4,
        description="n=200 * p=2000 * 12 informative * 4 balanced classes * macro-OVR",
    ),

    # -- Holdout scenarios (2) -- run ONCE only ---------------------------------
    # 7. Extreme high-dim: small n, p=10000 (0.1 % signal density)
    OmicsScenario(
        name="omics_ultra_sparse",
        n_samples=100, n_features=10000, n_informative=10,
        block_size=50, within_block_corr=0.75,
        class_imbalance=0.50, signal_strength=1.0,
        split="holdout",
        description="n=100 * p=10000 * 10 informative (0.1%) * rho=0.75 * HOLDOUT",
    ),
    # 8. Adversarial: large n, moderate p, near-independent features
    OmicsScenario(
        name="adversarial_large_n",
        n_samples=800, n_features=300, n_informative=30,
        block_size=10, within_block_corr=0.10,
        class_imbalance=0.50, signal_strength=1.5,
        split="holdout",
        description="n=800 * p=300 * 30 informative * rho=0.10 * HOLDOUT -- large-n regime",
    ),
]

DEVELOPMENT_SCENARIOS = [s for s in SCENARIOS if s.split == "development"]
HOLDOUT_SCENARIOS     = [s for s in SCENARIOS if s.split == "holdout"]


# ==============================================================================
# 3.  DATA GENERATION
# ==============================================================================

def make_omics_dataset(
    scenario: OmicsScenario,
    seed: int,
) -> Tuple[pd.DataFrame, np.ndarray, Set[int]]:
    """
    Generate a synthetic omics-like binary classification dataset.

    Features are organised into correlated blocks (gene-pathway structure).
    Each block is drawn from an equicorrelation multivariate Gaussian.

    How the signal is attached depends on ``scenario.signal_mode``:

    ``additive`` (default)
        Labels are drawn first, then lead features (index 0 within each
        informative block) receive a mean shift.  Binary scenarios shift by
        +/-signal_strength; multiclass scenarios give block b a shift for its
        preferred class ``b % n_classes`` and an offsetting negative shift
        elsewhere, so every feature is a marker for exactly one class and its
        overall mean stays at zero.

    ``interaction``
        Features are drawn first with NO marginal signal, then the label is
        derived from products of consecutive lead-feature PAIRS.  Because
        E[x_a * x_b] is symmetric about zero, each lead feature on its own has
        approximately zero marginal association with the outcome: only the
        pair predicts.  This is the regime a method scoring features one at a
        time cannot solve, and it is deliberately the opposite of the
        linear-additive assumption the rest of the suite is built on.

    ``n_informative = 0`` is the null control: labels are independent of every
    feature, so any selected feature is a false positive.

    Returns
    -------
    X : pd.DataFrame  shape (n_samples, n_features)
        Column names: 'feat_0000', 'feat_0001', ...
    y : np.ndarray    shape (n_samples,)  dtype int
        Values in ``range(scenario.n_classes)``.
    true_indices : set of int
        Indices of the ground-truth informative (lead) features.
        These are exactly  {0, block_size, 2*block_size, ...}, and empty for
        the null control.
    """
    rng = np.random.RandomState(seed)
    n, p   = scenario.n_samples, scenario.n_features
    bs     = scenario.block_size
    rho    = scenario.within_block_corr

    n_classes = scenario.n_classes

    # -- Class labels ----------------------------------------------------------
    # Skipped for interaction mode, where the label is derived from the
    # features rather than the other way round.
    if scenario.signal_mode == "additive":
        if n_classes == 2:
            n_pos = max(2, int(n * scenario.class_imbalance))
            n_neg = n - n_pos
            y = np.array([0] * n_neg + [1] * n_pos, dtype=int)
        else:
            # Balanced multiclass: the remainder is spread over the first
            # classes so the counts differ by at most one.
            base, extra = divmod(n, n_classes)
            y = np.array(
                [c for c in range(n_classes) for _ in range(base + (c < extra))],
                dtype=int,
            )
        rng.shuffle(y)

    # -- Block covariance for full blocks --------------------------------------
    # Equicorrelation matrix: off-diagonal = rho, diagonal = 1.
    # Always PSD for rho in [0, 1).
    Sigma_full = np.full((bs, bs), rho)
    np.fill_diagonal(Sigma_full, 1.0)

    # -- Generate features block by block (avoids pxp matrix for large p) -----
    n_full_blocks = p // bs
    remainder     = p % bs

    blocks: List[np.ndarray] = []
    for _ in range(n_full_blocks):
        blocks.append(rng.multivariate_normal(np.zeros(bs), Sigma_full, size=n))

    if remainder > 0:
        Sigma_rem = np.full((remainder, remainder), rho)
        np.fill_diagonal(Sigma_rem, 1.0)
        blocks.append(rng.multivariate_normal(np.zeros(remainder), Sigma_rem, size=n))

    X = np.hstack(blocks)   # (n, p)  -- may be slightly > p if remainder branch unused

    # -- Attach the signal -----------------------------------------------------
    if scenario.signal_mode == "interaction":
        # Labels are drawn first, then each informative PAIR is coupled to
        # them. Coupling per pair rather than pooling every pair into one
        # latent score is deliberate: a pooled score divides the signal by
        # n_informative, so each feature's marginal association collapses as
        # the informative set grows (measured: |r| ~= 0.52 at 2 informative
        # features, 0.16 at 6), and recovery becomes impossible for reasons
        # that have nothing to do with the interaction. Coupling per pair
        # keeps each pair's signal at full strength whatever n_informative is.
        n_pos = max(2, int(n * scenario.class_imbalance))
        y = np.array([0] * (n - n_pos) + [1] * n_pos, dtype=int)
        rng.shuffle(y)
        s = (2 * y - 1).astype(float)                 # -1 / +1 per sample
        r = scenario.interaction_ratio
        sig = scenario.signal_strength

        for k in range(scenario.n_informative // 2):
            a, b_ = (2 * k) * bs, (2 * k + 1) * bs
            xa, xb = X[:, a].copy(), X[:, b_].copy()

            # Interaction component: bend xb so that sign(xa * xb) tracks the
            # class. Neither marginal mean moves, because |xb| is multiplied
            # by a sign that is symmetric over the sample.
            coupled = s * np.sign(xa) * np.abs(xb)
            xb = r * coupled + (1.0 - r) * xb

            # Main-effect component: an ordinary mean shift on both members,
            # scaled by whatever share of the signal is not in the interaction.
            shift = (1.0 - r) * sig * s
            X[:, a]  = xa + shift
            X[:, b_] = xb + shift
    else:
        # Lead feature of block b is at column index b * bs.
        for b in range(scenario.n_informative):
            lead_idx = b * bs
            if n_classes == 2:
                # +s for class 1, -s for class 0: separation 2s.
                shift = (2 * y - 1).astype(float) * scenario.signal_strength
            else:
                # Block b marks class b % n_classes. The off-class shift
                # offsets the on-class one so the feature mean stays at zero
                # and no class is identifiable from the overall level.
                preferred = b % n_classes
                on = scenario.signal_strength
                off = -on / (n_classes - 1)
                shift = np.where(y == preferred, on, off).astype(float)
            X[:, lead_idx] += shift

    # -- Trim to exactly p columns (handles remainder edge case) --------------
    X = X[:, :p]

    cols = [f"feat_{i:04d}" for i in range(p)]
    return pd.DataFrame(X, columns=cols), y, scenario.true_feature_indices


# ==============================================================================
# 4.  EVALUATION UTILITIES
# ==============================================================================

def ground_truth_metrics(
    selected: Set[int],
    true_informative: Set[int],
) -> Dict[str, float]:
    """
    Precision, Recall, F1, Jaccard relative to the known informative set.

    All metrics return 0.0 when `selected` is empty.

    **Null control.**  When ``true_informative`` is empty there are no true
    positives to find, so precision, recall, F1 and Jaccard are all undefined
    and returned as NaN rather than 0.0.  Returning 0.0 would be actively
    misleading: it scores "selected nothing", which is the correct answer, the
    same as "selected 150 noise features".  On those scenarios the metric that
    carries the information is ``n_selected``, which counts false positives
    directly, and pandas skips the NaNs when aggregating everything else.
    """
    if not true_informative:
        nan = float("nan")
        return dict(precision=nan, recall=nan, f1=nan, jaccard=nan)
    if not selected:
        return dict(precision=0.0, recall=0.0, f1=0.0, jaccard=0.0)
    tp        = len(selected & true_informative)
    precision = tp / len(selected)
    recall    = tp / len(true_informative) if true_informative else 0.0
    denom_f1  = precision + recall
    f1        = 2.0 * precision * recall / denom_f1 if denom_f1 > 0.0 else 0.0
    jaccard   = tp / len(selected | true_informative)
    return dict(precision=precision, recall=recall, f1=f1, jaccard=jaccard)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

# Metrics that count things. Averaging them across seeds or folds produces a
# float, but "10.9 features" is not a quantity anyone can act on, so these are
# rounded to whole numbers wherever a table is produced for a human to read.
# The per-trial raw records keep their exact integer values either way.
_COUNT_METRICS: Set[str] = {"n_selected", "n_features", "n_true_informative"}


def _round_counts(frame: "pd.DataFrame") -> "pd.DataFrame":
    """
    Round count-valued columns of a summary frame to whole numbers.

    Applies to any column whose name is a count metric, with or without an
    aggregation suffix (``n_selected``, ``n_selected_mean``, ...).  The standard
    deviation of a count is left alone: it is a dispersion, not a count.

    Parameters
    ----------
    frame : pd.DataFrame
        Aggregated summary table.  Not mutated.

    Returns
    -------
    pd.DataFrame
        Copy with count columns rounded and stored as nullable integers.
    """
    out = frame.copy()
    for col in out.columns:
        base = str(col)
        if base.endswith("_std"):
            continue
        stem = base.rsplit("_", 1)[0] if base.endswith(("_mean", "_median")) else base
        if stem in _COUNT_METRICS or base in _COUNT_METRICS:
            out[col] = out[col].round(0).astype("Int64")
    return out


def kuncheva_stability(feature_sets: List[Set[int]], n_features: int) -> float:
    """
    Kuncheva Stability Index (Kuncheva 2007).

    KI in [-1, 1]; values near 1 indicate perfect reproducibility.

    Returns nan when the index is mathematically undefined: fewer than two sets
    to compare, or every set empty or full. The correction term is k^2/p and the
    denominator k - k^2/p, so k = 0 and k = p both collapse it.

    A nan here is usually a signal about the *selector*, not about stability.
    A method that returns all p features on every seed (which is what an L1
    model does when regularisation zeroes every coefficient, since
    ``SelectFromModel(threshold='mean')`` then has a zero threshold that every
    feature clears) produces a full set each time and therefore an undefined
    index. The reason is logged so the nan is traceable to its cause rather
    than appearing as an unexplained hole in the results table; inspect
    ``n_selected`` alongside it.
    """
    r = len(feature_sets)
    if r < 2:
        logger.debug(
            "Kuncheva index undefined: need at least 2 feature sets, got %d.", r
        )
        return float("nan")

    ki_sum, n_pairs = 0.0, 0
    n_empty = n_full = 0
    for i in range(r):
        for j in range(i + 1, r):
            si, sj = feature_sets[i], feature_sets[j]
            k = (len(si) + len(sj)) / 2.0   # mean set size for this pair
            if k < 1.0:
                n_empty += 1
                continue                      # degenerate pair -- skip
            if k >= n_features:
                n_full += 1
                continue                      # degenerate pair -- skip
            correction = k ** 2 / n_features
            denom      = k - correction
            if abs(denom) < 1e-12:
                continue
            ki_sum += (len(si & sj) - correction) / denom
            n_pairs += 1

    if n_pairs > 0:
        return ki_sum / n_pairs

    if n_full:
        logger.warning(
            "Kuncheva index undefined: all %d comparable pairs selected the "
            "full feature set (p=%d). The selector reduced nothing, so "
            "stability cannot be measured.",
            n_full, n_features,
        )
    elif n_empty:
        logger.warning(
            "Kuncheva index undefined: all %d comparable pairs selected fewer "
            "than one feature on average. The selector returned nothing.",
            n_empty,
        )
    return float("nan")


def _auc_scoring(y: np.ndarray) -> str:
    """
    sklearn scoring string for AUC, matching the number of classes present.

    'roc_auc' raises on multiclass targets, so anything that scores by AUC
    inside a selector has to ask first. Macro one-vs-rest is what the
    SelectOmics pipeline reports, so the comparators use the same.
    """
    return "roc_auc" if len(np.unique(y)) <= 2 else "roc_auc_ovr"


def _macro_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Binary AUC, or macro one-vs-rest when there are more than two classes."""
    classes = np.unique(y_true)
    if len(classes) <= 2:
        col = proba[:, 1] if proba.ndim > 1 and proba.shape[1] > 1 else proba.ravel()
        return float(roc_auc_score(y_true, col))
    return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))


def downstream_auc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    selected: Set[int],
    seed: int = 0,
) -> float:
    """
    Fit a fresh XGBClassifier on the selected columns and return test AUC.

    Macro one-vs-rest when the target has more than two classes, matching the
    metric the pipeline itself reports.

    Returns nan for empty selection or when y_test contains only one class.
    """
    if not selected:
        return float("nan")
    if len(np.unique(y_test)) < 2:
        return float("nan")

    idx = sorted(selected)
    # The shared downstream evaluator, applied identically to every method's
    # selected columns. It is the yardstick rather than a competitor, so it
    # advantages nobody -- but it is left at XGBoost defaults for the same
    # reason the selectors are, so no part of the comparison rests on a
    # hand-picked setting. It was max_depth=3, learning_rate=0.1 against
    # defaults of 6 and 0.3.
    clf = XGBClassifier(
        random_state=seed, verbosity=0, eval_metric="logloss",
        **_xgb_kwargs(),
    )
    try:
        clf.fit(X_train[:, idx], y_train)
        proba = clf.predict_proba(X_test[:, idx])
        return _macro_auc(y_test, proba)
    except Exception:
        return float("nan")


def mean_pairwise_corr(X: np.ndarray, selected: Set[int]) -> float:
    """
    Mean absolute pairwise Pearson correlation among selected features.
    Returns 0.0 for fewer than 2 selected features or constant columns.
    """
    if len(selected) < 2:
        return 0.0
    sub  = X[:, sorted(selected)]
    corr = np.corrcoef(sub, rowvar=False)
    corr = np.nan_to_num(np.abs(corr), nan=0.0)
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    vals = corr[mask]
    return float(np.mean(vals)) if vals.size > 0 else 0.0


# ==============================================================================
# 5.  STATISTICAL TESTS
# ==============================================================================

def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cliff's delta effect size.  Positive = a tends to be larger than b.

    Interpretation thresholds (absolute value):
        < 0.147  negligible
        < 0.330  small
        < 0.474  medium
        >= 0.474 large
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    dominance = sum(
        (1 if ai > bj else -1 if ai < bj else 0)
        for ai in a for bj in b
    )
    return dominance / (a.size * b.size)


def _holm_bonferroni(p_values: List[float]) -> List[float]:
    """
    Holm-Bonferroni step-down correction.

    NaN values are preserved.  Valid p-values are corrected using the
    number of valid (non-NaN) tests.
    """
    adjusted = list(p_values)
    valid    = [(i, p) for i, p in enumerate(p_values) if not np.isnan(p)]
    if not valid:
        return adjusted
    valid_sorted = sorted(valid, key=lambda x: x[1])
    n_valid      = len(valid_sorted)
    running_max  = 0.0
    for rank, (orig_idx, p) in enumerate(valid_sorted):
        adj         = min(1.0, (n_valid - rank) * p)
        running_max = max(running_max, adj)   # enforce monotonicity
        adjusted[orig_idx] = running_max
    return adjusted


def _effect_label(delta: float) -> str:
    d = abs(delta)
    if d < 0.147:  return "negligible"
    if d < 0.330:  return "small"
    if d < 0.474:  return "medium"
    return "large"


def friedman_test(df: pd.DataFrame, metric: str) -> Dict:
    """
    Friedman test across all methods for one metric.

    Observations are (scenario x seed) pairs -- equal sample sizes are
    enforced by trimming to the minimum count across methods.

    Returns dict with statistic, p_value, n_methods, n_obs.
    """
    methods = sorted(df["method"].unique())
    groups  = [
        df[df["method"] == m][metric].dropna().values
        for m in methods
    ]
    min_len = min(len(g) for g in groups)
    if min_len < 3:
        return dict(statistic=float("nan"), p_value=float("nan"),
                    n_methods=len(methods), n_obs=min_len,
                    note="insufficient observations")
    groups  = [g[:min_len] for g in groups]
    stat, p = friedmanchisquare(*groups)
    return dict(statistic=round(stat, 4), p_value=round(p, 6),
                n_methods=len(methods), n_obs=min_len)


def pairwise_vs_reference(
    df: pd.DataFrame,
    metric: str,
    reference: str = "SelectOmics",
    alpha: float = 0.05,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """
    Holm-Bonferroni corrected pairwise Wilcoxon signed-rank tests comparing
    each method against `reference`, paired by (scenario, seed).

    Returns a DataFrame with columns:
        method, n_pairs, W_statistic, p_raw, p_corrected,
        significant, cliff_delta, effect_size, favours
    """
    if reference not in df["method"].values:
        return pd.DataFrame(
            columns=["method", "n_pairs", "W_statistic", "p_raw",
                     "p_corrected", "significant", "cliff_delta",
                     "effect_size", "favours"]
        )

    # Pivot: rows = (scenario, seed), columns = method
    pivot = df.pivot_table(
        index=["scenario", "seed"],
        columns="method",
        values=metric,
        aggfunc="mean",
    )

    methods  = [m for m in pivot.columns if m != reference]
    raw_p    = []
    records  = []

    for method in methods:
        paired = pivot[[reference, method]].dropna()
        n      = len(paired)
        if n < 4:
            raw_p.append(float("nan"))
            records.append(dict(method=method, n_pairs=n,
                                W_statistic=float("nan"), p_raw=float("nan"),
                                cliff_delta=float("nan")))
            continue

        ref_vals    = paired[reference].values
        method_vals = paired[method].values

        try:
            stat, p = wilcoxon(method_vals, ref_vals, alternative="two-sided")
        except ValueError:
            stat, p = float("nan"), float("nan")

        cd = cliffs_delta(method_vals, ref_vals)
        raw_p.append(p)
        records.append(dict(method=method, n_pairs=n,
                            W_statistic=round(stat, 3) if not np.isnan(stat) else float("nan"),
                            p_raw=round(p, 6) if not np.isnan(p) else float("nan"),
                            cliff_delta=round(cd, 4) if not np.isnan(cd) else float("nan")))

    corrected = _holm_bonferroni(raw_p)

    for i, rec in enumerate(records):
        rec["p_corrected"] = round(corrected[i], 6) if not np.isnan(corrected[i]) else float("nan")
        rec["significant"] = bool(corrected[i] < alpha) if not np.isnan(corrected[i]) else False
        cd = rec["cliff_delta"]
        if np.isnan(cd):
            rec["effect_size"] = "unknown"
            rec["favours"]     = "unknown"
        else:
            rec["effect_size"] = _effect_label(cd)
            # Positive delta = method > reference.
            # For higher-is-better metrics: positive delta favours method.
            # For lower-is-better metrics (time): positive delta favours reference.
            if higher_is_better:
                rec["favours"] = reference if cd < -0.05 else ("method" if cd > 0.05 else "neither")
            else:
                rec["favours"] = reference if cd > 0.05 else ("method" if cd < -0.05 else "neither")

    return pd.DataFrame(records)


# ==============================================================================
# 6.  FEATURE SELECTOR CLASSES
# ==============================================================================

class FeatureSelector:
    """Abstract base.  All subclasses must implement select()."""
    name: str = "base"

    def select(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        scenario: OmicsScenario,
        seed: int,
    ) -> Set[int]:
        raise NotImplementedError


class VarianceCorrelationBaseline(FeatureSelector):
    """
    Baseline: drop features below median variance, then greedily remove
    features with pairwise |r| > 0.90 (keep higher-variance partner).
    """
    name = "VarCorr_Baseline"

    def select(self, X_train, y_train, scenario, seed):
        variances = np.var(X_train, axis=0)
        kept_idx  = np.where(variances > np.median(variances))[0]
        if len(kept_idx) < 2:
            return set(map(int, kept_idx))

        sub  = X_train[:, kept_idx]
        corr = np.abs(np.corrcoef(sub, rowvar=False))
        np.fill_diagonal(corr, 0.0)

        # Greedy vectorised elimination (O(k) outer loop, O(k) numpy inner op):
        # Process features high-variance -> low-variance.  For each surviving
        # feature i, mark all correlated survivors (|r| > 0.90) as dropped.
        # Because i always has the highest remaining variance, we always keep i.
        order = np.argsort(-variances[kept_idx])   # high-var first
        alive = np.ones(len(kept_idx), dtype=bool)

        for i in order:
            if not alive[i]:
                continue
            to_drop = (corr[i] > 0.90) & alive
            to_drop[i] = False          # never self-drop
            alive[to_drop] = False

        return {int(kept_idx[k]) for k in np.where(alive)[0]}


# Comparators run at their library defaults. This benchmark compares what each
# method does out of the box, so nothing here is tuned for these scenarios --
# the same reason SelectOmics is run at its shipped defaults rather than at
# anything the configuration search found.
#
# C = 1.0 is sklearn's LogisticRegression default. The previous 0.1 was a
# deviation from it, and a consequential one: at 0.1 an L1 fit drives EVERY
# coefficient to exactly zero in this regime (measured: 0 non-zero of 2000 at
# n=75, and likewise at p=5000 and p=1000).
#
# max_iter is raised from sklearn's 100 because saga does not converge on this
# data at the default, and a non-converged fit tests nothing. That is a
# convergence requirement, not a tuning choice.
_COMPARATOR_C: float = 1.0
_COMPARATOR_MAX_ITER: int = 2000


def _select_from_sparse_lr(X_train, y_train, seed: int, label: str,
                           **penalty_kwargs) -> Set[int]:
    """
    Shared body for the LASSO and ElasticNet comparators.

    Fits at the library default C and selects with SelectFromModel's default
    'mean' threshold, with one guard: when every coefficient is exactly zero
    the selection is EMPTY, not the full input.

    That guard fixes a real defect rather than tuning anything.
    SelectFromModel keeps ``importance >= threshold``; with all-zero
    coefficients the mean threshold is 0.0 and ``0 >= 0`` admits every feature,
    so the selector silently returned its entire input -- 2000 of 2000 at F1
    0.010, with an undefined Kuncheva index because k = p collapses its
    denominator. "Selected everything" is an artefact of comparing against a
    zero mean, not a selection any method would report.
    """
    est = LogisticRegression(solver="saga", C=_COMPARATOR_C,
                             max_iter=_COMPARATOR_MAX_ITER,
                             random_state=seed, **penalty_kwargs)
    est.fit(X_train, y_train)

    if np.abs(est.coef_).max() == 0:
        logger.warning(
            "%s: every coefficient is zero at C=%.3g. Returning an empty "
            "selection rather than the full input.", label, _COMPARATOR_C,
        )
        return set()

    sfm = SelectFromModel(est, threshold="mean", prefit=True)
    return {int(i) for i in np.where(sfm.get_support())[0]}


class LASSOSelector(FeatureSelector):
    """SelectFromModel -- L1-penalised Logistic Regression, threshold='mean'."""
    name = "LASSO"

    def select(self, X_train, y_train, scenario, seed):
        return _select_from_sparse_lr(X_train, y_train, seed, "LASSO",
                                      penalty="l1")


class ElasticNetSelector(FeatureSelector):
    """SelectFromModel -- ElasticNet-penalised Logistic Regression (l1_ratio=0.5)."""
    name = "ElasticNet"

    def select(self, X_train, y_train, scenario, seed):
        return _select_from_sparse_lr(X_train, y_train, seed, "ElasticNet",
                                      penalty="elasticnet", l1_ratio=0.5)


class RFECVSelector(FeatureSelector):
    """
    RFECV with a single XGBClassifier; CV=3 folds.

    step is set dynamically so that elimination takes ~20 rounds regardless
    of p -- keeping runtime tractable for p=5 000 and p=10 000 scenarios.
    """
    name = "RFECV"

    def select(self, X_train, y_train, scenario, seed):
        n_features = X_train.shape[1]
        # FEASIBILITY, not tuning. RFECV's default step of 1 needs one
        # elimination round per feature: 10,000 rounds times the CV folds on
        # omics_genomics, which is not runnable. cv is 3 rather than the
        # default 5 for the same reason. Both are disclosed in BENCHMARKS.md.
        step    = max(1, n_features // 20)
        min_sel = max(5, n_features // 100)   # floor; no library default exists

        # Estimator left at XGBoost defaults (n_estimators=100, max_depth=6).
        est   = XGBClassifier(random_state=seed, verbosity=0,
                              eval_metric="logloss", **_xgb_kwargs())
        cv    = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        rfecv = RFECV(estimator=est, cv=cv,
                      step=step,
                      min_features_to_select=min_sel,
                      scoring=_auc_scoring(y_train), n_jobs=_N_JOBS)
        rfecv.fit(X_train, y_train)
        return {int(i) for i in np.where(rfecv.support_)[0]}


# ------------------------------------------------------------------------------
# Compute budget for every estimator in this module.
#
# These were pinned to n_jobs=1 for reproducibility.  Measured against
# scikit-learn 1.7 / xgboost 3.3, thread count is not what reproducibility
# depends on: RandomForest, RFECV, and permutation_importance each seed from
# random_state rather than from how the work is divided across threads, and
# return bit-identical results at any n_jobs.
#
# XGBoost carries one condition.  Under tree_method='hist' it is thread-invariant
# ONLY while subsample == 1.0; row subsampling draws in a thread-order-dependent
# way, so the same seed yields different trees at different thread counts
# (measured drift ~2e-2 in predicted probabilities).  No estimator here sets
# subsample, so the harness is reproducible -- but if you add subsample, switch
# to tree_method='exact' at the same time, which is unconditionally
# thread-invariant and is what the SelectOmics package itself pins.
#
# GPU is the standing exception: device='cuda' uses a different split-finding
# implementation and shifts feature importances by ~1e-2 regardless of threads,
# so it is opt-in and must not be used for runs compared against published
# numbers.
# ------------------------------------------------------------------------------

_N_JOBS: int = 1
_DEVICE: str = "cpu"


def set_compute(n_jobs: int = 1, use_gpu: bool = False) -> None:
    """
    Set threads and device for every estimator built in this module.

    Parameters
    ----------
    n_jobs : int
        Threads per estimator. -1 uses all cores. Does not affect results.
    use_gpu : bool
        Route XGBoost through CUDA. **Changes results slightly.** Off by default.
    """
    global _N_JOBS, _DEVICE
    _N_JOBS = int(n_jobs)
    _DEVICE = "cuda" if use_gpu else "cpu"
    if use_gpu:
        print("WARNING: GPU enabled for XGBoost. Results will differ slightly "
              "from the CPU numbers in BENCHMARKS.md; do not mix the two.")


def _xgb_kwargs() -> dict:
    """Shared XGBoost keyword arguments carrying the current compute budget."""
    return {"n_jobs": _N_JOBS, "device": _DEVICE}


class RandomForestSelector(FeatureSelector):
    """SelectFromModel -- RandomForest with mean importance threshold."""
    name = "RF_Importance"

    def select(self, X_train, y_train, scenario, seed):
        # n_estimators left at sklearn's default of 100. It was 200, i.e.
        # tuned upward, which is as much a distortion as tuning downward.
        rf  = RandomForestClassifier(random_state=seed, n_jobs=_N_JOBS)
        sfm = SelectFromModel(rf, threshold="mean")
        sfm.fit(X_train, y_train)
        return {int(i) for i in np.where(sfm.get_support())[0]}


class BorutaSelector(FeatureSelector):
    """Boruta shadow-feature wrapper.  Skipped if boruta not installed."""
    name = "Boruta"

    def select(self, X_train, y_train, scenario, seed):
        if not _BORUTA_AVAILABLE:
            return set()
        rf     = RandomForestClassifier(n_estimators=100, random_state=seed,
                                        n_jobs=_N_JOBS)
        boruta = BorutaPy(estimator=rf, n_estimators="auto",
                          max_iter=50, random_state=seed, verbose=0)
        boruta.fit(X_train, y_train)
        return {int(i) for i in np.where(boruta.support_)[0]}


class SHAPSelector(FeatureSelector):
    """
    XGBClassifier + TreeExplainer; select features with mean|SHAP| >= global mean.
    Skipped if shap not installed.
    """
    name = "SHAP"

    def select(self, X_train, y_train, scenario, seed):
        if not _SHAP_AVAILABLE:
            return set()
        # XGBoost defaults (n_estimators=100, max_depth=6). max_depth was 3.
        clf = XGBClassifier(random_state=seed, verbosity=0,
                            eval_metric="logloss", **_xgb_kwargs())
        clf.fit(X_train, y_train)
        explainer   = _shap.TreeExplainer(clf)
        sv          = explainer.shap_values(X_train)
        importance  = (np.mean([np.abs(s).mean(0) for s in sv], 0)
                       if isinstance(sv, list) else np.abs(sv).mean(0))
        if not np.any(importance > 0):
            # Same trap as PermImportance and SelectFromModel: with every
            # attribution at zero the mean threshold is 0.0 and `>= 0` admits
            # the entire input. It did not fire in the current run, but the
            # guard costs nothing and the failure is silent when it does.
            logger.warning(
                "SHAP: all attributions are zero. Returning an empty "
                "selection rather than the full input."
            )
            return set()
        return {int(i) for i in np.where(importance >= np.mean(importance))[0]}


class PermutationImportanceSelector(FeatureSelector):
    """
    XGBClassifier + sklearn permutation_importance (5 repeats);
    select features with mean importance >= mean of positive values.
    """
    name = "PermImportance"

    def select(self, X_train, y_train, scenario, seed):
        # XGBoost defaults (n_estimators=100, max_depth=6). max_depth was 3.
        clf = XGBClassifier(random_state=seed, verbosity=0,
                            eval_metric="logloss", **_xgb_kwargs())
        clf.fit(X_train, y_train)
        # n_repeats=5 is sklearn's default for permutation_importance.
        res   = permutation_importance(clf, X_train, y_train, n_repeats=5,
                                       scoring=_auc_scoring(y_train),
                                       random_state=seed,
                                       n_jobs=_N_JOBS)
        imp = res.importances_mean
        pos = imp[imp > 0]
        if pos.size == 0:
            # Nothing hurt performance when shuffled, so the method found no
            # signal. The old code set the threshold to 0.0 and kept
            # `imp >= 0`, which admits every feature with exactly-zero
            # importance -- i.e. every feature the model never used. Measured
            # on this benchmark: 2000 of 2000 and 10000 of 10000 selected, at
            # precision 0.005 and 0.001 and Kuncheva -0.67 and -0.44.
            #
            # This is the same shape of defect as a percentile or mean
            # threshold collapsing to zero: a degenerate threshold that admits
            # everything is not a selection.
            logger.warning(
                "PermImportance: no feature had positive permutation "
                "importance. Returning an empty selection rather than the "
                "full input."
            )
            return set()
        return {int(i) for i in np.where(imp >= float(np.mean(pos)))[0]}


class SelectOmicsMultiStepSelector:
    """
    Runs the SelectOmics pipeline in multiple independent configurations per
    (scenario, seed), producing one benchmark record per configuration.

    Two classes of comparison
    -------------------------
    Cumulative configs  -- run on ALL scenarios.  Each starts from raw data
    and applies an increasing prefix of the pipeline.  Because all methods
    in a comparison group start from the same raw input, these are fair:

        SO_S1->2      steps 1+2   (filter + regularise)      vs LASSO, ElasticNet
        SelectOmics   steps 1+2+3 (+ RFECV wrapper)          vs RFECV; end-to-end

    There is no separate SO_S1->2->3 arm: with three selection steps, the full
    pipeline IS the 1+2+3 prefix. Carrying both ran the identical config twice
    and reported the tautology as a measurement.

    Isolated / ablation configs  -- run on ABLATION_SCENARIOS only.
    Each enables exactly one step on the raw feature matrix.  Useful as a
    diagnostic: do SO's individual techniques outperform their standalone
    equivalents?  Note that isolated SO steps were not designed to be used this
    way -- losses here do not imply inferiority of the full pipeline.

        SO_S1_only    only step 1 (raw -> filter)        vs VarCorr_Baseline
        SO_S2_only    only step 2 (raw -> regularise)    vs LASSO, ElasticNet
        SO_S3_only    only step 3 (raw -> RFECV)         vs RFECV
    
    Each config is an independent pipeline run with its own ``time_seconds``.

    Config priority
    ---------------
    If ``opt_results/selectomics_recommended_config.json`` exists (produced by
    *SelectOmics_Config_Optimization.ipynb*) its parameters are loaded
    automatically.  Otherwise sensible defaults are used.
    """

    # (label, enable_step1, enable_step2, enable_step3)
    CUMULATIVE_CONFIGS: List[Tuple] = [
        ("SO_S1->2",   True,  True,  False),
        ("SelectOmics", True, True,  True),
    ]
    ISOLATED_CONFIGS: List[Tuple] = [
        ("SO_S1_only",  True,  False, False),
        ("SO_S2_only",  False, True,  False),
        ("SO_S3_only",  False, False, True),
    ]
    # Scenarios the isolated ablation runs on.
    #
    # omics_tiny_n is the small-n / high-p case (n=50, p=500) and is the one
    # place SelectOmics beats the overall winner outright: on ground-truth
    # recovery it holds F1 0.324 where ElasticNet collapses to 0.039.  That
    # makes it the most informative scenario in the set for this package, so it
    # is deliberately kept in the ablation despite the Step-3 caveat below.
    #
    # Step-3 caveat.  Step 3 needs MIN_FEATURES_FOR_RFECV features to run.
    # That gate was 30 when the counts below were measured, which is why
    # Step 3 skipped almost everywhere; it is 10 from 0.7.0, derived from
    # the point at which the step starts beating a pass-through.
    # Measured survivors after Steps 1+2, seed 0:
    #     omics_tiny_n       p=500     10   Step 3 skipped
    #     adversarial_easy   p=200     15   Step 3 skipped
    #     omics_imbalanced   p=1000    22   Step 3 skipped
    #     omics_standard     p=2000    27   Step 3 skipped
    #     omics_high_dim     p=5000    19   Step 3 skipped
    #     omics_genomics     p=10000   95   Step 3 runs
    #
    # In both ablation scenarios Step 3 self-skips, so SelectOmics returns
    # exactly what SO_S1->2 returned and their difference is zero by
    # construction rather than by measurement.  Do not read that as "Step 3
    # contributes nothing": rows where a requested step did not run carry a
    # 'steps_skipped' value, so filter on it before quoting any Step-3 delta.
    # The measurable Step-3 contribution lives in the CUMULATIVE configs on
    # omics_genomics, where 95 features reach Step 3 and it cuts them to 57.
    #
    # BENCHMARK_SPEC is untouched by any of this, so the locked
    # pre-registration is unaffected.
    ABLATION_SCENARIOS: Set[str]  = {"omics_tiny_n", "adversarial_easy"}
    CUMULATIVE_NAMES:   List[str] = ["SO_S1->2", "SelectOmics"]
    ISOLATED_NAMES:     List[str] = ["SO_S1_only", "SO_S2_only", "SO_S3_only"]
    ALL_STEP_NAMES:     List[str] = CUMULATIVE_NAMES + ISOLATED_NAMES
    name = "SelectOmics_MultiStep"   # registry key; not used as a record label

    def __init__(self, fast_mode: bool = True) -> None:
        self.fast_mode   = fast_mode
        self._opt_params = _load_optimized_params()

    # -- Config builder --------------------------------------------------------

    def _build_config(
        self,
        data_path:    str,
        out_dir:      str,
        seed:         int,
        enable_step1: bool = True,
        enable_step2: bool = True,
        enable_step3: bool = True,
    ) -> "SelectOmicsConfig":
        """Construct SelectOmicsConfig from optimised params (or defaults)."""
        p        = self._opt_params or {}
        # Full mode measures the SHIPPING model count, not a reduced one.
        #
        # This used to be 5 in full mode against a shipping default of 10, so
        # the published figures described a weaker configuration than users
        # get -- and by the parameter that matters most: the configuration
        # search measured n_consensus_models at Spearman r = 0.835, p < 0.0001,
        # monotonic across 3, 5 and 10 (composite 0.641, 0.687, 0.706). A
        # benchmark that understates the product on its most influential
        # setting is not characterising the product.
        #
        # Fast mode keeps a reduced count deliberately: it exists for
        # iterating on evaluation code, and its numbers are not published.
        n_models = int(p.get("n_consensus_models",
                             3 if self.fast_mode
                             else _so_default("n_consensus_models")))
        qt_iter  = int(p.get("quick_tune_iterations", 10 if self.fast_mode else 30))
        return SelectOmicsConfig(
            data_path                        = data_path,
            target_column                    = "target",
            algorithm                        = "XGB",
            n_consensus_models               = n_models,
            random_seed                      = int(seed),
            output_dir                       = out_dir,
            test_size                        = 0.20,
            verbose                          = False,
            create_visualizations            = False,
            save_intermediate_results        = False,
            enable_step_evaluations          = False,
            enable_final_test_evaluation     = False,
            enable_step1                     = enable_step1,
            enable_step2                     = enable_step2,
            enable_step3                     = enable_step3,
            quick_tune_iterations            = qt_iter,
            # Fall back to the SHIPPING default for anything the caller did
            # not set, read from the dataclass rather than repeated here.
            #
            # These were hardcoded, and drifted: the literals said
            # min_consensus=None and min_features_floor=10 while the package
            # had moved to 0.4 and 15. A benchmark run would then have
            # characterised a configuration nobody ships, which is the one
            # thing this harness must never do. Reading the dataclass makes
            # that class of drift impossible.
            min_consensus          = p.get("min_consensus", _so_default("min_consensus")),
            min_features_floor     = int(p.get("min_features_floor", _so_default("min_features_floor"))),
            stability_threshold    = p.get("stability_threshold", _so_default("stability_threshold")),
            step2_target_retention = float(p.get("step2_target_retention", _so_default("step2_target_retention"))),
            step3_target_retention = float(p.get("step3_target_retention", _so_default("step3_target_retention"))),
            step2_role             = p.get("step2_role", _so_default("step2_role")),
            # Match the harness budget so the pipeline does not oversubscribe
            # when several benchmark workers run at once.
            n_jobs                           = _N_JOBS,
            use_gpu                          = (_DEVICE == "cuda"),
        )

    # -- Single-config runner --------------------------------------------------

    def run_config(
        self,
        X_train:       np.ndarray,
        y_train:       np.ndarray,
        scenario:      OmicsScenario,
        seed:          int,
        feature_names: List[str],
        enable_step1:  bool = True,
        enable_step2:  bool = True,
        enable_step3:  bool = True,
    ) -> Tuple[Set[int], List[str]]:
        """
        Run one pipeline configuration; return selected indices and skipped steps.

        ``get_selected_features(step=None)`` returns the output of the last
        *enabled* step, matching the configuration's intent.

        The second return value names steps that were requested but did not
        actually run. The pipeline skips Step 3 when fewer than 30 features
        reach it, which is sensible for an
        analysis but corrupts an ablation: ``SelectOmics`` then returns exactly
        what ``SO_S1->2`` returned, and the comparison reads as "Step 3
        contributed nothing" when Step 3 never ran at all. Callers must record
        this so a degenerate ablation is visible in the results rather than
        indistinguishable from a measured null result.

        Returns
        -------
        (selected, skipped)
            ``selected`` is the set of chosen feature indices; ``skipped`` lists
            the labels of requested-but-skipped steps, e.g. ``['step3']``.
        """
        if not _SO_AVAILABLE:
            return set(), []

        name_to_idx = {name: i for i, name in enumerate(feature_names)}
        df = pd.DataFrame(X_train, columns=feature_names)
        df["target"] = y_train

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.csv")
            out_dir   = os.path.join(tmpdir, "so_out")
            df.to_csv(data_path, index=False)

            config   = self._build_config(
                data_path, out_dir, seed,
                enable_step1=enable_step1, enable_step2=enable_step2,
                enable_step3=enable_step3,
            )
            pipeline = SelectOmicsPipeline(config)
            try:
                pipeline.run(validate=False)
                cols = pipeline.get_selected_features(step=None)
                selected = {name_to_idx[c] for c in cols if c in name_to_idx}

                # Step 3 self-skips below its minimum feature count. Record
                # it, so an ablation arm that never ran is visible as such
                # rather than reading as 'this step contributed nothing'.
                skipped = [
                    key for key, requested in (("step3", enable_step3),
                                               )
                    if requested and pipeline.results.get(key, {}).get("skipped")
                ]
                if skipped:
                    logger.warning(
                        "SO config on %s seed=%d: %s requested but skipped "
                        "(too few features reached it). This configuration is "
                        "therefore identical to the one without %s, so any "
                        "ablation difference is zero by construction, not by "
                        "measurement.",
                        scenario.name, seed, "/".join(skipped),
                        "/".join(skipped),
                    )
                return selected, skipped

            except Exception as exc:
                logger.warning(
                    "SO config (s1=%s s2=%s s3=%s s4=%s) failed [%s seed=%d]: %s",
                    enable_step1, enable_step2, enable_step3,
                    scenario.name, seed, exc,
                )
                return set(), []


# ==============================================================================
# 7.  BENCHMARK RUNNER
# ==============================================================================

class SyntheticBenchmark:
    """
    Orchestrates the full synthetic benchmark.

    Parameters
    ----------
    output_dir : str or Path
        Directory for benchmark_raw.csv and benchmark_summary.xlsx.
    fast_mode : bool
        Passed to SelectOmicsMultiStepSelector.  Reduces n_consensus_models
        to 3 and quick_tune_iterations to 10, and sets default n_seeds to 3
        (vs 5 for full mode).  All four SO steps are always enabled.

        Fast mode is for iterating on evaluation code, not for publishable
        figures: full mode uses the shipping n_consensus_models so the
        benchmark characterises the configuration users actually receive.
    skip_selectomics : bool
        Run comparators only -- useful when iterating on evaluation code.
    n_seeds : int or None
        Independent seeds per (scenario, method).
        Defaults to 3 (fast_mode) or 5 (full mode).
    """

    def __init__(self,
                 output_dir: str = "results",
                 fast_mode: bool = True,
                 skip_selectomics: bool = False,
                 n_seeds: Optional[int] = None) -> None:
        self.output_dir       = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fast_mode        = fast_mode
        self.skip_selectomics = skip_selectomics
        self.n_seeds          = n_seeds if n_seeds is not None else (3 if fast_mode else 5)
        self._methods         = self._build_methods()
        self._records: List[Dict] = []

    # -- Method registry -------------------------------------------------------

    def _build_methods(self) -> List:
        methods: List = [
            VarianceCorrelationBaseline(),
            LASSOSelector(),
            ElasticNetSelector(),
            RFECVSelector(),
            RandomForestSelector(),
            PermutationImportanceSelector(),
        ]
        if _SHAP_AVAILABLE:
            methods.append(SHAPSelector())
        if _BORUTA_AVAILABLE:
            methods.append(BorutaSelector())
        if not self.skip_selectomics and _SO_AVAILABLE:
            # Multi-step: runs pipeline once per seed, exposes SO_Step1 through
            # SelectOmics (full) -- each step compared to its standalone equivalent.
            methods.append(SelectOmicsMultiStepSelector(fast_mode=self.fast_mode))
        return methods

    @property
    def method_names(self) -> List[str]:
        names: List[str] = []
        for m in self._methods:
            if isinstance(m, SelectOmicsMultiStepSelector):
                names.extend(m.ALL_STEP_NAMES)
            else:
                names.append(m.name)
        return names

    # -- Main entry point ------------------------------------------------------

    def run(self, scenarios: Optional[List[OmicsScenario]] = None) -> pd.DataFrame:
        """
        Run all (scenario, method, seed) trials.

        Standard methods contribute one record per (scenario, method, seed).
        ``SelectOmicsMultiStepSelector`` contributes four records per seed
        (SO_Step1, SO_Step2, SO_Step3, SelectOmics) from a single pipeline run.

        Saves raw results to output_dir/benchmark_raw.csv.
        Returns the flat results DataFrame.
        """
        scenarios = scenarios or DEVELOPMENT_SCENARIOS
        seeds     = BENCHMARK_SPEC["seeds"][:self.n_seeds]

        n_std = sum(1 for m in self._methods
                    if not isinstance(m, SelectOmicsMultiStepSelector))
        _mss  = next((m for m in self._methods
                      if isinstance(m, SelectOmicsMultiStepSelector)), None)
        n_cumul   = len(_mss.CUMULATIVE_NAMES)   if _mss else 0
        n_iso     = len(_mss.ISOLATED_NAMES)     if _mss else 0
        n_ablation_sc = sum(1 for sc in scenarios
                            if _mss and sc.name in _mss.ABLATION_SCENARIOS)
        total = (
            len(scenarios)  * (n_std + n_cumul) * len(seeds)
            + n_ablation_sc *  n_iso             * len(seeds)
        )

        print(
            f"Benchmark v{BENCHMARK_SPEC['schema_version']}  |  "
            f"{len(scenarios)} scenarios x "
            f"{n_std + n_cumul} methods"
            + (f" (+ {n_iso} ablation on {n_ablation_sc} fast scenarios)"
               if n_iso and n_ablation_sc else "")
            + f" x {len(seeds)} seeds = {total} trials\n"
        )

        for sc in scenarios:
            print(f"-- {sc.name}  [{sc.description}]")

            for method in self._methods:
                if isinstance(method, SelectOmicsMultiStepSelector):
                    self._run_multistep_method(sc, method, seeds)
                else:
                    self._run_standard_method(sc, method, seeds)

            print()

        df = self._to_dataframe()
        df.to_csv(self.output_dir / "benchmark_raw.csv", index=False)
        print(f"Saved -> {self.output_dir / 'benchmark_raw.csv'}")
        return df

    # -- Per-method runner helpers ---------------------------------------------

    def _run_standard_method(
        self,
        sc:     OmicsScenario,
        method: "FeatureSelector",
        seeds:  List[int],
    ) -> None:
        """Run a single-output selector across all seeds; back-fill KI."""
        seed_sets: List[Set[int]] = []
        n_appended_before = len(self._records)

        for seed in seeds:
            t0     = time.perf_counter()
            record = self._run_one(sc, method, seed)
            record["time_seconds"] = round(time.perf_counter() - t0, 2)
            self._records.append(record)
            seed_sets.append(record["_selected_set"])

            print(
                f"   {method.name:22s}  seed={seed}  "
                f"n_sel={record['n_selected']:4d}  "
                f"F1={record['f1']:.3f}  "
                f"AUC={record['test_auc']:.3f}  "
                f"({record['time_seconds']:.1f}s)"
            )

        ki         = kuncheva_stability(seed_sets, sc.n_features)
        n_appended = len(self._records) - n_appended_before
        for rec in self._records[-n_appended:]:
            rec["kuncheva_stability"] = round(ki, 4)

    def _run_multistep_method(
        self,
        sc:    OmicsScenario,
        mss:   SelectOmicsMultiStepSelector,
        seeds: List[int],
    ) -> None:
        """
        Run each SelectOmics config independently per seed.

        Cumulative configs run on every scenario.  Isolated / ablation configs
        run only on scenarios in ``mss.ABLATION_SCENARIOS``.  Each config is a
        separate pipeline run with its own ``time_seconds``.  KI is back-filled
        per label after all seeds for that scenario complete.
        """
        is_ablation = sc.name in mss.ABLATION_SCENARIOS
        configs     = list(mss.CUMULATIVE_CONFIGS) + (
                      list(mss.ISOLATED_CONFIGS) if is_ablation else [])
        labels      = [c[0] for c in configs]

        step_seed_sets: Dict[str, List[Set[int]]] = {lb: [] for lb in labels}
        n_appended_before = len(self._records)

        for seed in seeds:
            step_records = self._run_multistep_one(sc, mss, seed, configs)

            for label, rec in step_records.items():
                self._records.append(rec)
                step_seed_sets[label].append(rec["_selected_set"])
                iso_tag = "  [ablation]" if label in mss.ISOLATED_NAMES else ""
                print(
                    f"   {label:22s}  seed={seed}  "
                    f"n_sel={rec['n_selected']:4d}  "
                    f"F1={rec['f1']:.3f}  "
                    f"AUC={rec['test_auc']:.3f}  "
                    f"({rec['time_seconds']:.1f}s)"
                    f"{iso_tag}"
                )

        # Back-fill KI per label over records added in this call
        for label in labels:
            ki = kuncheva_stability(step_seed_sets[label], sc.n_features)
            for rec in self._records[n_appended_before:]:
                if rec["method"] == label:
                    rec["kuncheva_stability"] = round(ki, 4)

    def _run_multistep_one(
        self,
        sc:      OmicsScenario,
        mss:     SelectOmicsMultiStepSelector,
        seed:    int,
        configs: List[Tuple],
    ) -> Dict[str, Dict]:
        """
        Generate data once; run each pipeline config; return
        ``{ label -> record_dict }`` with ``time_seconds`` per config.
        """
        data_seed = seed * 97 + 13
        X, y, true_idx = make_omics_dataset(sc, seed=data_seed)

        X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
            X.values, y,
            test_size    = BENCHMARK_SPEC["test_size"],
            stratify     = y,
            random_state = seed,
        )
        scaler = MinMaxScaler()
        X_tr   = scaler.fit_transform(X_tr_raw)
        X_te   = scaler.transform(X_te_raw)
        feature_names = list(X.columns)

        step_records: Dict[str, Dict] = {}
        for label, s1, s2, s3 in configs:
            t0 = time.perf_counter()
            try:
                selected, skipped_steps = mss.run_config(
                    X_tr_raw, y_tr, sc, seed, feature_names,
                    enable_step1=s1, enable_step2=s2,
                    enable_step3=s3,
                )
            except Exception as exc:
                logger.warning("SO %s failed [%s seed=%d]: %s",
                               label, sc.name, seed, exc)
                selected, skipped_steps = set(), []
            elapsed    = round(time.perf_counter() - t0, 2)
            gt         = ground_truth_metrics(selected, true_idx)
            auc        = downstream_auc(X_tr, y_tr, X_te, y_te, selected, seed=seed)
            redundancy = mean_pairwise_corr(X_tr, selected)

            step_records[label] = {
                "scenario":           sc.name,
                "split":              sc.split,
                "n_samples":          sc.n_samples,
                "n_features":         sc.n_features,
                "n_informative":      sc.n_informative,
                "signal_density":     round(sc.signal_density, 4),
                "within_block_corr":  sc.within_block_corr,
                "class_imbalance":    sc.class_imbalance,
                "signal_strength":    sc.signal_strength,
                "method":             label,
                "seed":               seed,
                "precision":          round(gt["precision"], 4),
                "recall":             round(gt["recall"],    4),
                "f1":                 round(gt["f1"],        4),
                "jaccard":            round(gt["jaccard"],   4),
                "test_auc":           round(auc, 4) if not np.isnan(auc) else float("nan"),
                "n_selected":         len(selected),
                "reduction_ratio":    round(1.0 - len(selected) / sc.n_features, 4),
                "mean_feature_corr":  round(redundancy, 4),
                # Steps requested by this ablation config that did not actually
                # run because too few features reached them. A non-empty value
                # means this row duplicates the config one step below it, so
                # the ablation difference is zero by construction rather than
                # by measurement. Filter on it before interpreting step deltas.
                "steps_skipped":      ",".join(skipped_steps),
                "time_seconds":       elapsed,
                "kuncheva_stability": float("nan"),   # back-filled after all seeds
                # See the note on the standard record: which features, not how
                # many. Required to check whether an ablation step changes the
                # panel's composition rather than only its size.
                "selected_indices":   ",".join(str(i) for i in sorted(selected)),
                "_selected_set":      selected,
            }

        return step_records

    def run_holdout(self) -> pd.DataFrame:
        """
        Run the two holdout scenarios exactly once.

        WARNING:  Call this ONLY when all development decisions are final.
        Results are saved to output_dir/holdout_raw.csv.
        """
        print("=" * 60)
        print("HOLDOUT EVALUATION -- run once only")
        print("=" * 60)
        df = self.run(scenarios=HOLDOUT_SCENARIOS)
        df.to_csv(self.output_dir / "holdout_raw.csv", index=False)
        return df

    # -- Single-trial logic ----------------------------------------------------

    def _run_one(self, sc: OmicsScenario, method: FeatureSelector,
                 seed: int) -> Dict:
        # Use a distinct data-generation seed for each (scenario x seed) pair
        data_seed = seed * 97 + 13   # arbitrary, reproducible offset
        X, y, true_idx = make_omics_dataset(sc, seed=data_seed)

        X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
            X.values, y,
            test_size=BENCHMARK_SPEC["test_size"],
            stratify=y,
            random_state=seed,
        )

        # Scale for comparators and for the shared downstream AUC evaluation
        scaler = MinMaxScaler()
        X_tr   = scaler.fit_transform(X_tr_raw)
        X_te   = scaler.transform(X_te_raw)

        # -- Selection ---------------------------------------------------------
        try:
            selected = method.select(X_tr, y_tr, sc, seed)
        except Exception as exc:
            logger.warning("%s failed [%s seed=%d]: %s",
                           method.name, sc.name, seed, exc)
            selected = set()

        # -- Evaluation --------------------------------------------------------
        gt         = ground_truth_metrics(selected, true_idx)
        auc        = downstream_auc(X_tr, y_tr, X_te, y_te, selected, seed=seed)
        redundancy = mean_pairwise_corr(X_tr, selected)

        return {
            "scenario":           sc.name,
            "split":              sc.split,
            "n_samples":          sc.n_samples,
            "n_features":         sc.n_features,
            "n_informative":      sc.n_informative,
            "signal_density":     round(sc.signal_density, 4),
            "within_block_corr":  sc.within_block_corr,
            "class_imbalance":    sc.class_imbalance,
            "signal_strength":    sc.signal_strength,
            "method":             method.name,
            "seed":               seed,
            # Ground-truth recovery
            "precision":          round(gt["precision"], 4),
            "recall":             round(gt["recall"],    4),
            "f1":                 round(gt["f1"],        4),
            "jaccard":            round(gt["jaccard"],   4),
            # Downstream performance
            "test_auc":           round(auc, 4),
            # Efficiency
            "n_selected":         len(selected),
            "reduction_ratio":    round(1.0 - len(selected) / sc.n_features, 4),
            # Present on every record so the column survives _to_dataframe.
            # Only the SelectOmics ablation configs can populate it.
            "steps_skipped":      "",
            "mean_feature_corr":  round(redundancy, 4),
            # Stability -- back-filled after all seeds for this method
            "kuncheva_stability": float("nan"),
            # WHICH features, not merely how many.  Two runs returning twelve
            # features each look identical in every other column while sharing
            # none of the same twelve, and that distinction is the whole point
            # of a reproducibility claim.  The Kuncheva index summarises it to
            # one number; this column is what lets anyone recompute that, ask
            # which specific features recur, or check a panel against biology.
            "selected_indices":   ",".join(str(i) for i in sorted(selected)),
            # Internal (not written to CSV)
            "_selected_set":      selected,
        }

    # -- Output helpers --------------------------------------------------------

    def _to_dataframe(self) -> pd.DataFrame:
        # Take the union of keys rather than record 0's keys alone: the
        # multi-step ablation records carry fields the standard records do not,
        # and keying off the first record silently dropped them (standard
        # methods are registered first, so record 0 is always a standard one).
        export = []
        for record in self._records:
            for key in record:
                if not key.startswith("_") and key not in export:
                    export.append(key)
        return pd.DataFrame(
            [{k: r.get(k, "") for k in export} for r in self._records]
        )

    def summarise(self) -> pd.DataFrame:
        """Mean +/- std per (scenario, method); saves benchmark_summary.xlsx."""
        if not self._records:
            raise RuntimeError("No results -- call run() first.")
        df   = self._to_dataframe()
        cols = ["precision", "recall", "f1", "jaccard", "test_auc",
                "n_selected", "reduction_ratio", "mean_feature_corr",
                "kuncheva_stability", "time_seconds"]
        agg  = df.groupby(["scenario", "method"])[cols].agg(["mean", "std"])
        agg.columns = ["_".join(c) for c in agg.columns]
        agg  = agg.reset_index()
        agg  = _round_counts(agg)
        out  = self.output_dir / "benchmark_summary.xlsx"
        agg.to_excel(out, index=False)
        print(f"Summary -> {out}")
        return agg

    def rank_methods(self, metric: str = "f1") -> pd.DataFrame:
        """Pivot: scenario x method for mean metric; MEAN row appended."""
        if not self._records:
            raise RuntimeError("No results -- call run() first.")
        df    = self._to_dataframe()
        pivot = (df.groupby(["scenario", "method"])[metric]
                   .mean().unstack("method").round(3))
        pivot.loc["MEAN"] = pivot.mean()
        if metric in _COUNT_METRICS:
            pivot = pivot.round(0).astype("Int64")
        return pivot

    def load_results(self, path: Optional[str] = None) -> pd.DataFrame:
        """Load saved raw CSV without re-running."""
        csv = Path(path) if path else (self.output_dir / "benchmark_raw.csv")
        df  = pd.read_csv(csv)
        self._records = df.to_dict(orient="records")
        for r in self._records:
            r["_selected_set"] = set()
        print(f"Loaded {len(df)} rows from {csv}")
        return df


# ==============================================================================
# 9.  PARALLEL DRIVER
# ==============================================================================

def _scenario_task(
    scenario_name: str,
    output_dir:    str,
    fast_mode:     bool,
    skip_selectomics: bool,
    n_seeds:       Optional[int],
    n_jobs:        int,
    use_gpu:       bool,
) -> List[Dict]:
    """
    Run every method on one scenario and return its records.

    Executed in a worker process, so it takes only picklable arguments and
    rebuilds the scenario and method registry locally rather than receiving
    live objects.

    Scenario is the unit of parallelism because the Kuncheva stability index is
    computed across seeds *within* a (scenario, method) pair. Splitting any
    finer would break that back-fill; splitting by scenario leaves it intact,
    so results are identical to a sequential run.
    """
    set_compute(n_jobs, use_gpu)

    scenario = next((s for s in SCENARIOS if s.name == scenario_name), None)
    if scenario is None:
        raise ValueError(f"Unknown scenario: {scenario_name!r}")

    bench = SyntheticBenchmark(
        output_dir=output_dir,
        fast_mode=fast_mode,
        skip_selectomics=skip_selectomics,
        n_seeds=n_seeds,
    )
    bench.run(scenarios=[scenario])

    # Selected-index sets are large and only needed for the KI that run()
    # already back-filled, so drop them rather than pickle them back.
    records = []
    for rec in bench._records:
        rec = dict(rec)
        rec.pop("_selected_set", None)
        records.append(rec)
    return records


def run_parallel(
    scenarios:        Optional[List["OmicsScenario"]] = None,
    output_dir:       str  = "results",
    fast_mode:        bool = True,
    skip_selectomics: bool = False,
    n_seeds:          Optional[int] = None,
    workers:          Optional[int] = None,
    n_jobs:           Optional[int] = None,
    use_gpu:          bool = False,
    verbose:          bool = True,
) -> pd.DataFrame:
    """
    Run the synthetic benchmark with one worker process per scenario.

    Parameters
    ----------
    scenarios : list of OmicsScenario or None
        Defaults to ``DEVELOPMENT_SCENARIOS``.
    output_dir : str
        Destination for benchmark_raw.csv and the per-scenario working dirs.
    fast_mode, skip_selectomics, n_seeds
        As on ``SyntheticBenchmark``.
    workers : int or None
        Concurrent scenarios. ``None`` splits the machine sensibly, ``1`` runs
        in-process and is identical to calling ``SyntheticBenchmark.run``.
    n_jobs : int or None
        Threads per estimator. ``None`` divides cores among workers.
    use_gpu : bool
        Opt-in CUDA for XGBoost. **Changes results**; see ``set_compute``.

    Returns
    -------
    pd.DataFrame
        All records, also written to ``output_dir/benchmark_raw.csv``.
    """
    scenarios = scenarios or DEVELOPMENT_SCENARIOS
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    resolved_workers, resolved_jobs = plan_parallelism(workers, n_jobs)
    # More workers than scenarios would just leave processes idle; re-plan so
    # the freed cores go to the estimators instead.
    if resolved_workers > len(scenarios):
        resolved_workers, resolved_jobs = plan_parallelism(len(scenarios), n_jobs)

    if verbose:
        print(f"Synthetic benchmark: {len(scenarios)} scenarios")
        print(describe_plan(resolved_workers, resolved_jobs, label="  "))
        print()

    tasks = [
        (sc.name, str(out / f"_scenario_{sc.name}"), fast_mode,
         skip_selectomics, n_seeds, resolved_jobs, use_gpu)
        for sc in scenarios
    ]

    def _announce(records: List[Dict]) -> None:
        if verbose and records:
            print(f"  finished {records[0]['scenario']}: {len(records)} records")

    all_records: List[Dict] = []
    for records in run_tasks(
        _scenario_task, tasks,
        workers=resolved_workers, n_jobs=resolved_jobs,
        on_result=_announce,
    ):
        all_records.extend(records)

    df = pd.DataFrame(all_records)
    raw_path = out / "benchmark_raw.csv"
    df.to_csv(raw_path, index=False)
    if verbose:
        print(f"\nSaved -> {raw_path}  ({len(df)} rows)")
    return df


def main(argv: Optional[List[str]] = None) -> int:
    """
    Run the synthetic benchmark from the shell.

    Examples
    --------
        # Development scenarios, filling the machine:
        python benchmark_synthetic.py --workers -1

        # Full protocol including held-out scenarios, 5 seeds:
        python benchmark_synthetic.py --full --seeds 5 --workers 6

        # Comparators only, to iterate on evaluation code:
        python benchmark_synthetic.py --skip-selectomics
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="benchmark_synthetic",
        description="Benchmark SelectOmics against standard feature selectors "
                    "on synthetic omics-shaped data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output", "-o", default="results",
                        help="Output directory (default: results).")
    parser.add_argument("--full", action="store_true",
                        help="Run every scenario, not just the development set.")
    parser.add_argument("--scenarios", nargs="+", default=None, metavar="NAME",
                        help="Explicit scenario names to run.")
    parser.add_argument("--seeds", type=int, default=None,
                        help="Seeds per (scenario, method). Default 3 fast / 5 full.")
    parser.add_argument("--fast", dest="fast_mode", action="store_true", default=True,
                        help="Fast mode (default): fewer consensus models.")
    parser.add_argument("--thorough", dest="fast_mode", action="store_false",
                        help="Full consensus counts and tuning budget.")
    parser.add_argument("--skip-selectomics", action="store_true",
                        help="Run the comparator methods only.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Concurrent scenarios. -1 fills the machine, "
                             "1 runs in-process. Default: half the cores.")
    parser.add_argument("--n-jobs", type=int, default=None, dest="n_jobs",
                        help="Threads per estimator. Default: cores divided "
                             "among workers. Does not affect results.")
    parser.add_argument("--use-gpu", action="store_true",
                        help="Route XGBoost through CUDA. CHANGES RESULTS "
                             "slightly; do not mix with CPU-generated numbers.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.scenarios:
        chosen = [s for s in SCENARIOS if s.name in set(args.scenarios)]
        missing = set(args.scenarios) - {s.name for s in chosen}
        if missing:
            print(f"ERROR: unknown scenario(s): {sorted(missing)}", file=sys.stderr)
            print(f"Available: {[s.name for s in SCENARIOS]}", file=sys.stderr)
            return 1
    elif args.full:
        chosen = list(SCENARIOS)
    else:
        chosen = list(DEVELOPMENT_SCENARIOS)

    t0 = time.perf_counter()
    df = run_parallel(
        scenarios=chosen,
        output_dir=args.output,
        fast_mode=args.fast_mode,
        skip_selectomics=args.skip_selectomics,
        n_seeds=args.seeds,
        workers=args.workers,
        n_jobs=args.n_jobs,
        use_gpu=args.use_gpu,
        verbose=not args.quiet,
    )
    elapsed = time.perf_counter() - t0

    if not df.empty and "f1" in df.columns:
        pivot = (df.groupby(["scenario", "method"])["f1"]
                   .mean().unstack("method").round(3))
        pivot.loc["MEAN"] = pivot.mean()
        print(f"\n{'=' * 72}")
        print("Mean F1 by scenario and method")
        print(pivot.to_string())
        print(f"{'=' * 72}")

        # A scenario with no informative features has no true positives, so F1
        # is undefined rather than zero and prints as NaN. Say so, or the row
        # reads as a crash.
        null_rows = [s for s in pivot.index
                     if s != "MEAN" and pivot.loc[s].isna().all()]
        if null_rows:
            print(
                "NOTE: " + ", ".join(null_rows) + " has no informative "
                "features, so F1 is undefined (NaN) by construction, not by "
                "failure."
            )
            print("      Read those on n_selected: every feature selected "
                  "there is a false positive.")
            counts = (df[df.scenario.isin(null_rows)]
                        .groupby("method")["n_selected"].mean()
                        .sort_values().round(1))
            print("      false positives by method: "
                  + ", ".join(f"{m} {v:g}" for m, v in counts.items()))
            print(f"{'=' * 72}")

    print(f"Wall clock: {elapsed / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
