"""
SelectOmics/selection/step3_wrapper.py

Step 3: Wrappers (wrapper-based selection methods) via RFECV and Stability Selection.

Two independent stages are run in parallel and their kept sets are
intersected:

Stage 1 - RFECV consensus: n_models independent RFECV runs vote on
  feature retention. A binary search over the full [0, 1] percentile
  domain finds the min_features_to_select that targets
  step3_target_retention, converging in O(log n) evaluations vs O(n) for
  an exhaustive grid.  The full voting pass is run once at the best
  percentile.

Stage 2 - Stability selection: STABILITY_N_SUBSAMPLES stratified
  subsamples (50% without replacement) are drawn, each used to fit one
  algorithm-specific model. A feature passes if its normalized importance
  exceeds the subsample mean on >= stability_threshold fraction of valid
  subsamples. Reference: Meinshausen & Buhlmann (2010),
  doi:10.1111/j.1467-9868.2010.00740.x

Design decisions
----------------
- SVM uses LinearSVC (L1 penalty) as a surrogate wrappers/stability
  estimator. LinearSVC produces sparse, directly interpretable coef_
  values and avoids the O(n^3) training cost and lack of coef_ of
  RBF SVC. The RBF SVC is still used for all classification/evaluation.
- Binary search for Wrappers percentile: RFECV is the most expensive
  per-evaluation operation in the pipeline; each call fits n_models
  RFECV instances each with their own internal CV.  Binary search
  converges in O(log n) vs O(n) for grid search.
- RFECV internal CV uses min(3, cv.n_splits) folds for speed.  The outer
  CV object (passed by the pipeline) is used only for step evaluations.
- RFECV seed scheme: base_seed + i*400.  Stability model seed:
  base_seed + (i % n_models)*400 + 500.  The +500 offset separates
  stability models from RFECV models; cycling (i % n_models) limits the
  model variety to n_models distinct seeds while distributing them across
  50 subsamples.
- Warning accumulation: per-iteration exceptions are counted and reported
  once at the end rather than flooding output.
- Step 3 is skipped entirely (pass-through) if X_input.shape[1] < 30.
  RFECV is not effective at very low feature counts and the internal
  CV splits may fail.

"""

from __future__ import annotations

import logging
import warnings as _warnings
from pathlib import Path

logger = logging.getLogger(__name__)
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone as _clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

from ..config import SelectOmicsConfig
from ..models.base import build_consensus_pipeline, xgb_compute_kwargs
from ..models.evaluation import evaluate_models_collection
from ..utils.helpers import (
    save_step_summary,
    _binary_search_percentile,
    relax_consensus_intersection,
)
from ..data.loaders import save_from_config
from .step0_reference import _print_adequacy_warning


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Minimum number of features required to enter RFECV.
#
# 30 is empirical and deliberately conservative. Lowering it to 10 was tried
# and reverted; the record matters because the obvious measurement is
# misleading.
#
# Run Step 3 on a panel that is HALF NOISE (5 informative among p_in) and it
# looks like the gate should be about 10 -- it beats passing the panel through
# from p_in 10 upward, at precision 1.000 throughout:
#
#     p_in     6      8     10     12     15     30
#     keep-all F1  0.909  0.769  0.667  0.588  0.500  0.286
#     Step 3 F1    0.333  0.571  0.750  0.750  1.000  1.000
#
# But that is not the panel Step 2 hands over. Step 2's output is nearly pure
# signal (precision 0.970 across the 5-seed benchmark, 1.000 on most runs), so
# there is no noise in it for Step 3 to remove and elimination can only cut
# true features. Measured end to end at a gate of 10:
#
#     scenario          Step 2 -> Step 3     F1
#     omics_standard    10 -> 1              1.000 -> 0.182
#     omics_genomics    10 -> 1              1.000 -> 0.182
#     omics_genomics    10 -> 2              1.000 -> 0.333
#
# Step 2 delivered a perfect panel and Step 3 destroyed it, at 60s per run
# against 0.1s when skipped.
#
# The size of the incoming panel is therefore the wrong signal; what matters is
# how much noise it still contains, and that is not observable at runtime. The
# high gate stands in for it: a panel that is still wide has probably not been
# filtered to purity yet. Step 3 remains valuable standalone (enable_step1 and
# enable_step2 off, where it receives raw features) and after a deliberately
# permissive Step 2 -- measured on omics_multiclass, 58 features in, 12 out, at
# precision 1.000 and recall 1.000.
MIN_FEATURES_FOR_RFECV: int = 30

# RFECV elimination step size.
#
# RFECV refits the estimator once per elimination round across every internal
# CV fold, so step=1 makes the round count scale with the feature count and the
# total cost roughly quadratic. A run on 545 features cost 34,771 CPU-seconds
# during development and had to be killed.
#
# Measured, 10 informative features, one RFECV fit:
#
#     p_in    step=1              5% step              10% step
#       50    F1 0.833   2.4s    F1 0.833   1.1s      F1 0.800   0.5s
#      100    F1 0.947   4.6s    F1 0.800   1.0s      F1 0.900   0.6s
#      200    F1 0.727  10.6s    F1 0.700   1.1s      F1 0.667   0.6s
#      400    F1 0.643  24.1s    F1 0.600   1.3s      F1 0.222   0.7s
#
# A flat 10% collapses at 400 features (returns 80, F1 0.222): too coarse to
# locate a small panel. A flat 5% is affordable everywhere but gives up F1 in
# the 100-feature range, where step=1 is still cheap at under five seconds.
#
# So: exact below the threshold, proportional above it. The in-pipeline gate
# (MIN_FEATURES_FOR_RFECV = 30) and the standalone use both sit near or inside
# the exact region for ordinary panels, so their behaviour is unchanged; the
# proportional branch exists for the wide panels that standalone use and a
# deliberately permissive Step 2 can hand over, where step=1 is not affordable
# at any accuracy.
_RFECV_EXACT_BELOW: int = 100
_RFECV_STEP_FRACTION: float = 0.05


def _rfecv_step(n_features: int) -> int:
    """Elimination step: exact for narrow inputs, proportional for wide ones."""
    if n_features <= _RFECV_EXACT_BELOW:
        return 1
    return max(1, int(n_features * _RFECV_STEP_FRACTION))


# Number of stratified subsamples for stability selection.  50 is the
# standard lower bound recommended by Shah & Samworth (2013),
# doi:10.1111/j.1467-9868.2012.01043.x
STABILITY_N_SUBSAMPLES: int = 50

# ---------------------------------------------------------------------------
# Seed scheme
# ---------------------------------------------------------------------------

# RFECV models: base_seed + i * _RFECV_SEED_STEP
# Stability models: base_seed + (i % n_models) * _RFECV_SEED_STEP + _STABILITY_SEED_OFFSET
# Stability subsamples: base_seed + i * _SUBSAMPLE_SEED_STEP
# The offsets keep RFECV and stability seeds non-overlapping for typical
# values of n_models and n_subsamples.
_RFECV_SEED_STEP:        int = 400
_STABILITY_SEED_OFFSET:  int = 500
_SUBSAMPLE_SEED_STEP:    int = 500

# Step 3 consensus evaluation models use the same spacing as steps 1 and 2.
_EVAL_SEED_STEP: int = 100


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_step3_wrapper(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    config: SelectOmicsConfig,
    cv: StratifiedKFold,
    tuned_pipelines: Dict[str, Any],
    n_classes: int,
    class_names: List[str],
    ref_result_step0: Optional[Dict[str, Any]] = None,
    adequacy: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Apply Wrappers (RFECV and stability selection) consensus wrapper.

    Stage 1 runs n_models RFECV instances with a binary-search-optimised
    min_features_to_select targeting 60% retention, then takes a consensus
    vote.  Stage 2 runs STABILITY_N_SUBSAMPLES stratified subsamples, fits
    one model per subsample, and selects features whose normalized importance
    exceeds the subsample mean on at least stability_threshold fraction of
    valid subsamples.  The final kept set is the intersection of both stages,
    with union and top-k fallbacks.

    Step 3 is skipped (pass-through) when:
      - ``config.enable_step3`` is False, or
      - ``X_train.shape[1] < MIN_FEATURES_FOR_RFECV`` (30).

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix from Step 2 output.
    X_test : pd.DataFrame
        Test feature matrix aligned with X_train columns.
    y_train : np.ndarray
        Encoded integer class labels, shape (n_samples,).
    config : SelectOmicsConfig
        Pipeline configuration object.
    cv : StratifiedKFold
        Stratified CV splitter shared across all steps.
    tuned_pipelines : dict
        Mapping from algorithm key to fitted sklearn Pipeline.
    n_classes : int
        Number of distinct target classes.
    class_names : list of str
        Human-readable class label strings.
    ref_result_step0 : dict, optional
        Reference CV result from Step 0 for comparison.
    adequacy : dict, optional
        Output of ``assess_sample_size_adequacy`` for warnings.
    output_dir : Path, optional
        Artefact output directory; derived from config if None.

    Returns
    -------
    dict with keys:
        X_train_rfecv : pd.DataFrame
            Training matrix after Step 3 selection.
        X_test_rfecv : pd.DataFrame
            Test matrix with the same column subset.
        dropped_features : list of str
            Features removed by the Wrappers/stability consensus.
        rfecv_selected : np.ndarray of bool
            Feature kept mask from the RFECV voting pass.
        stability_selected : np.ndarray of bool
            Feature kept mask from stability selection.
        rfecv_vote_counts : np.ndarray of int
            Per-feature vote counts from the RFECV stage.
        stability_freq : np.ndarray of float or None
            Per-feature selection frequencies from stability selection.
            None when all subsamples failed.
        best_rfecv_percentile : float
            RFECV percentile from binary search.
        warnings : dict
            Accumulated warning counts:
            'rfecv_failed', 'stability_failed',
            'stability_skipped_class', 'stability_all_failed'.
        consensus_result : dict or None
            CV evaluation result for the selected feature set.
        n_features_in : int
        n_features_out : int
        skipped : bool
            True when Step 3 was skipped due to disabled flag or low
            feature count.
    """
    algorithm = config.algorithm

    if output_dir is None:
        output_dir = Path(config.output_dir)

    # Inherit GPU device from the tuned XGB pipeline when available.
    _device: str = 'cpu'
    if algorithm == 'XGB' and tuned_pipelines:
        try:
            _device = tuned_pipelines['XGB'].named_steps['clf'].get_params().get('device', 'cpu')
        except Exception:
            _device = 'cpu'

    from ..models.base import effective_n_jobs, resolve_n_jobs
    _n_jobs = effective_n_jobs(resolve_n_jobs(config), *X_train.shape)

    logger.info("STEP 3: WRAPPERS (RFECV/STABILITY) - %s", algorithm)

    # ------------------------------------------------------------------
    # Disabled branch
    # ------------------------------------------------------------------
    if not config.enable_step3:
        logger.info("Step 3 disabled - using input data unchanged")
        logger.info("  Features: %d", X_train.shape[1])
        return _passthrough(X_train, X_test, ref_result_step0, skipped=True)

    # ------------------------------------------------------------------
    # Low-feature-count safety check
    # ------------------------------------------------------------------
    if X_train.shape[1] < MIN_FEATURES_FOR_RFECV:
        logger.warning(
            "Only %d features remaining, below the %d needed. A panel this "
            "narrow has already been filtered to near-purity, so recursive "
            "elimination would cut true features rather than noise. "
            "Skipping Step 3 (see MIN_FEATURES_FOR_RFECV).",
            X_train.shape[1], MIN_FEATURES_FOR_RFECV,
        )
        return _passthrough(X_train, X_test, ref_result_step0, skipped=True)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    n_models = config.n_consensus_models
    min_consensus = config.min_consensus
    stability_threshold = config.stability_threshold
    rfecv_target_retention = config.step3_target_retention
    base_seed = config.random_seed

    # The per-stage view is taken at unanimity, matching where the graduated
    # intersection across both stages begins.
    required_votes = n_models
    # Absolute safety net, not a retention target; see config.min_features_floor.
    min_features = min(config.min_features_floor, X_train.shape[1])

    logger.info("Using %s with %d consensus model(s)", algorithm, n_models)
    logger.info(
        "Stability threshold: %s (subsamples: %d)",
        stability_threshold, STABILITY_N_SUBSAMPLES,
    )
    logger.info(
        "Starting at unanimity (%d/%d), relaxing until %d features are "
        "reached%s",
        n_models, n_models, min_features,
        "" if min_consensus is None
        else f", but not below {min_consensus:.0%} agreement",
    )
    logger.info("Input features: %d", X_train.shape[1])

    # Accumulates importances from all models in both stages for fallback.
    all_importances_combined: List[np.ndarray] = []

    # Warning accumulator: counted per occurrence, printed once at end.
    warnings: Dict[str, int] = {
        "rfecv_failed": 0,
        "stability_failed": 0,
        "stability_skipped_class": 0,
        "stability_all_failed": 0,
    }

    # ------------------------------------------------------------------
    # Stage 1: RFECV consensus
    # ------------------------------------------------------------------
    rfecv_selected, rfecv_vote_counts, rfecv_importances, best_rfecv_percentile = (
        _run_rfecv_stage(
            X_train=X_train,
            y_train=y_train,
            algorithm=algorithm,
            n_models=n_models,
            base_seed=base_seed,
            target_retention=rfecv_target_retention,
            required_votes=required_votes,
            cv=cv,
            warnings=warnings,
            verbose=config.verbose,
            device=_device,
            n_jobs=_n_jobs,
        )
    )
    all_importances_combined.extend(rfecv_importances)

    # ------------------------------------------------------------------
    # Stage 2: Stability selection
    # ------------------------------------------------------------------
    stability_selected, stability_freq, stability_importances = _run_stability_stage(
        X_train=X_train,
        y_train=y_train,
        algorithm=algorithm,
        n_models=n_models,
        base_seed=base_seed,
        stability_threshold=stability_threshold,
        min_features=min_features,
        warnings=warnings,
        verbose=config.verbose,
        device=_device,
        n_jobs=_n_jobs,
    )
    all_importances_combined.extend(stability_importances)

    # ------------------------------------------------------------------
    # Consensus intersection with fallback hierarchy
    # ------------------------------------------------------------------
    logger.info("CONSENSUS: RFECV AND STABILITY INTERSECTION")
    logger.info("  RFECV selected:     %d features", rfecv_selected.sum())
    logger.info("  Stability selected: %d features", stability_selected.sum())

    # Put stability on the same vote scale as RFECV so one relaxation ladder
    # governs both. Stability records the fraction of 50 subsamples that chose
    # a feature; multiplying by n_models expresses that as "how many of
    # n_models would this have convinced", which is what the RFECV votes
    # already are. Relaxing the requirement then relaxes model agreement and
    # subsample agreement together, which is the linkage the two thresholds
    # are meant to have.
    if stability_freq is not None:
        freq = np.asarray(stability_freq, dtype=float)
        # Below stability_threshold the Meinshausen-Buhlmann guarantee does not
        # hold, so those features cast no stability vote at any relaxation
        # level. Above it the vote count scales with selection frequency, so
        # relaxing the requirement relaxes subsample agreement in step with
        # model agreement. Without this floor stability_threshold would be
        # vestigial: it would set a mask used only for reporting while the
        # actual selection ran off the raw frequency.
        eligible = freq >= stability_threshold
        stability_votes = np.where(
            eligible, np.floor(freq * n_models), 0
        ).astype(int)
    else:
        # Stability stage failed entirely; it abstains rather than vetoes.
        stability_votes = np.full(X_train.shape[1], n_models, dtype=int)

    consensus = relax_consensus_intersection(
        vote_arrays=[rfecv_vote_counts, stability_votes],
        n_models=n_models,
        min_features=min_features,
        mean_importance=(np.mean(all_importances_combined, axis=0)
                         if all_importances_combined else None),
        min_consensus=min_consensus,
        allow_union=getattr(config, 'allow_union_rung', False),
    )
    final_selected = consensus.mask

    if consensus.floor_used < consensus.floor_requested:
        logger.info(
            "  Minimum feature count capped at %d (asked for %d): nothing "
            "more was selected by any model in both stages.",
            consensus.floor_used, consensus.floor_requested,
        )
    if consensus.how == 'consensus':
        logger.info(
            "  Unanimous: %d features agreed by all %d models in both stages "
            "(agreement: %s).",
            final_selected.sum(), n_models, consensus.label,
        )
    elif consensus.how == 'relaxed':
        logger.info(
            "  Relaxed from %d/%d to %d/%d votes per stage to reach %d "
            "features, agreement %.0f%% (%s). The intersection rule still "
            "holds at that level.",
            n_models, n_models, consensus.votes_required, n_models,
            final_selected.sum(), consensus.agreement * 100, consensus.label,
        )
    elif consensus.how == 'consensus_limited':
        logger.warning(
            "  min_consensus stopped the relaxation at %d/%d votes (%s) "
            "before the %d-feature minimum was reached: only %d features "
            "survive at the agreement you set as acceptable.",
            consensus.votes_required, n_models, consensus.label,
            consensus.floor_requested, final_selected.sum(),
        )
    else:
        logger.warning(
            "  No feature was chosen by any model in both stages. Falling back "
            "to the top %d by mean importance within the voted pool; this "
            "result is ranked, not agreed.", consensus.floor_used,
        )

    X_rfecv = X_train.loc[:, final_selected]
    X_test_rfecv = X_test.loc[:, final_selected]
    dropped_features = X_train.columns[~final_selected].tolist()

    logger.info(
        "Step 3 final: %d features retained, %d dropped",
        X_rfecv.shape[1], len(dropped_features),
    )

    # ------------------------------------------------------------------
    # Print accumulated warnings
    # ------------------------------------------------------------------
    _print_warnings(warnings)

    # ------------------------------------------------------------------
    # Save intermediate results
    # ------------------------------------------------------------------
    n_valid_stability = len(stability_importances)

    if config.save_intermediate_results:
        output_dir.mkdir(parents=True, exist_ok=True)

        save_from_config(
            pd.DataFrame({"dropped_by_step3": dropped_features}),
            output_dir / "step3_dropped_features",
            config,
        )
        save_from_config(
            X_rfecv,
            output_dir / "step3_X_wrappers_final",
            config,
        )

        # Per-feature diagnostics: useful for downstream biological
        # interpretation of which features passed each filter.
        if n_valid_stability > 0 and stability_freq is not None:
            diag_df = pd.DataFrame({
                "feature": X_train.columns,
                "rfecv_votes": rfecv_vote_counts,
                "rfecv_vote_fraction": rfecv_vote_counts / n_models,
                "stability_freq": stability_freq,
                "rfecv_selected": rfecv_selected,
                "stability_selected": stability_selected,
                "final_selected": final_selected,
            })
            save_from_config(
                diag_df,
                output_dir / "step3_selection_diagnostics",
                config,
            )

        # Record the selection facts now; see the note in step2 on why this
        # cannot wait for the optional evaluation.
        save_step_summary("step3", {
            "algorithm": algorithm,
            "n_consensus_models": n_models,
            "input_features": X_train.shape[1],
            "output_features": X_rfecv.shape[1],
            "features_removed": X_train.shape[1] - X_rfecv.shape[1],
            "votes_required": consensus.votes_required,
            "agreement": consensus.agreement,
            "agreement_label": consensus.label,
            "consensus_outcome": consensus.how,
            "consensus_floor_used": consensus.floor_used,
            "consensus_floor_requested": consensus.floor_requested,
        }, output_dir)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    logger.info("Step 3 Summary:")
    logger.info("  Input features:       %d", X_train.shape[1])
    logger.info("  RFECV selected:       %d", rfecv_selected.sum())
    logger.info("  Stability selected:   %d", stability_selected.sum())
    logger.info("  Final (intersection): %d", X_rfecv.shape[1])
    logger.info("  Features removed:     %d", X_train.shape[1] - X_rfecv.shape[1])

    # ------------------------------------------------------------------
    # Step evaluation
    # ------------------------------------------------------------------
    consensus_result = None

    if config.enable_step_evaluations:
        consensus_result = _evaluate_step3(
            X_rfecv=X_rfecv,
            X_train_original=X_train,
            y_train=y_train,
            config=config,
            cv=cv,
            tuned_pipelines=tuned_pipelines,
            n_classes=n_classes,
            class_names=class_names,
            ref_result_step0=ref_result_step0,
            adequacy=adequacy,
            output_dir=output_dir,
            base_seed=base_seed,
            n_models=n_models,
            algorithm=algorithm,
            consensus_summary={
                "votes_required": consensus.votes_required,
                "agreement": consensus.agreement,
                "agreement_label": consensus.label,
                "consensus_outcome": consensus.how,
                "consensus_floor_used": consensus.floor_used,
                "consensus_floor_requested": consensus.floor_requested,
            },
        )
    else:
        logger.info("Step 3 evaluation skipped.")

    return {
        "X_train_rfecv": X_rfecv,
        "X_test_rfecv": X_test_rfecv,
        "dropped_features": dropped_features,
        "rfecv_selected": rfecv_selected,
        "stability_selected": stability_selected,
        "rfecv_vote_counts": rfecv_vote_counts,
        "stability_freq": stability_freq,
        "best_rfecv_percentile": best_rfecv_percentile,
        "warnings": warnings,
        "consensus_result": consensus_result,
        "n_features_in": X_train.shape[1],
        "n_features_out": X_rfecv.shape[1],
        "votes_required": consensus.votes_required,
        "agreement": consensus.agreement,
        "agreement_label": consensus.label,
        "consensus_outcome": consensus.how,
        "consensus_floor_used": consensus.floor_used,
        "consensus_floor_requested": consensus.floor_requested,
        "n_models": n_models,
        "skipped": False,
    }


# ---------------------------------------------------------------------------
# Private helpers: pass-through for disabled/skipped branch
# ---------------------------------------------------------------------------

def _passthrough(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    ref_result_step0: Optional[Dict[str, Any]],
    skipped: bool,
) -> Dict[str, Any]:
    """Return an unchanged pass-through result dict."""
    n = X_train.shape[1]
    return {
        "X_train_rfecv": X_train.copy(),
        "X_test_rfecv": X_test.copy(),
        "dropped_features": [],
        "rfecv_selected": np.ones(n, dtype=bool),
        "stability_selected": np.ones(n, dtype=bool),
        "rfecv_vote_counts": np.zeros(n, dtype=int),
        "stability_freq": None,
        "best_rfecv_percentile": 0.0,
        "warnings": {
            "rfecv_failed": 0,
            "stability_failed": 0,
            "stability_skipped_class": 0,
            "stability_all_failed": 0,
        },
        "consensus_result": ref_result_step0,
        "n_features_in": n,
        "n_features_out": n,
        "skipped": skipped,
        # Present even when skipped, so callers can read these keys off any
        # step without guarding. A skipped step decided nothing, which is
        # what 'not run' records.
        "votes_required": 0,
        "agreement": 0.0,
        "agreement_label": "not run",
        "consensus_outcome": "skipped" if skipped else "disabled",
        "n_models": 0,
        # Added with the consensus work and missed here, which left these two
        # readable on a normal run and absent on a skipped one -- exactly the
        # guarding the comment above says callers should not need.
        "consensus_floor_used": 0,
        "consensus_floor_requested": 0,
    }


# ---------------------------------------------------------------------------
# Private helpers: RFECV estimator construction
# ---------------------------------------------------------------------------

def _build_rfecv_estimator(
    algorithm: str,
    seed: int,
    device: str = 'cpu',
    n_jobs: int = -1,
) -> Tuple[Pipeline, Any]:
    """
    Build an algorithm-specific estimator and importance getter for RFECV.

    For SVM, a LinearSVC (L1 penalty) surrogate is used. L1 LinearSVC
    produces sparse coef_ values directly usable for recursive elimination.
    RBF SVC lacks coef_ and would require permutation importance, which is
    prohibitively expensive inside RFECV. The surrogate is used only within
    this step; all classification tasks use the tuned RBF SVC pipeline.

    Parameters
    ----------
    algorithm : str
    seed : int
    device : str
        XGBoost compute device ('cpu' or 'cuda'). Ignored for non-XGB algorithms.

    Returns
    -------
    estimator : Pipeline
    importance_getter : str or callable
        Passed directly to ``sklearn.feature_selection.RFECV``.
    """
    if algorithm == "LR":
        estimator = Pipeline([
            ("scaler", MinMaxScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, random_state=seed, solver="lbfgs"
            )),
        ])
        importance_getter = (
            lambda est: np.abs(est.named_steps["clf"].coef_).max(axis=0)
        )

    elif algorithm == "XGB":
        if not _XGB_AVAILABLE:
            raise ImportError("xgboost is required for algorithm='XGB'")
        estimator = Pipeline([
            ("scaler", MinMaxScaler()),
            ("clf", XGBClassifier(
                random_state=seed,
                verbosity=0,
                eval_metric='mlogloss',
                **xgb_compute_kwargs(n_jobs, device),
            )),
        ])
        importance_getter = "named_steps.clf.feature_importances_"

    elif algorithm == "RF":
        estimator = Pipeline([
            ("scaler", MinMaxScaler()),
            ("clf", RandomForestClassifier(
                random_state=seed, n_jobs=n_jobs, n_estimators=50
            )),
        ])
        importance_getter = "named_steps.clf.feature_importances_"

    elif algorithm == "SVM":
        # L1 LinearSVC surrogate. probability=True is not required because
        # this estimator is used only for coef_-based importance ranking,
        # never for predict_proba. dual='auto' selects the solver correctly
        # based on n_samples vs n_features relationship.
        estimator = Pipeline([
            ("scaler", MinMaxScaler()),
            ("clf", LinearSVC(
                penalty="l1", dual="auto", C=0.1,
                max_iter=2000, random_state=seed
            )),
        ])

        def importance_getter(est, X=None, y=None):
            clf = (
                est.named_steps["clf"]
                if hasattr(est, "named_steps")
                else est
            )
            # coef_ shape: (1, n_features) binary or
            # (n_classes, n_features) multiclass OVR.
            return np.abs(clf.coef_).max(axis=0)

    else:
        raise ValueError(
            f"Unsupported algorithm '{algorithm}'. "
            f"Expected one of 'LR', 'XGB', 'RF', 'SVM'."
        )

    return estimator, importance_getter


def _subsample_rows(
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int,
    fraction: float = 0.8,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Draw a stratified subsample of rows, without replacement.

    Used to give each consensus replicate different data so that varying the
    seed produces a genuinely different model.  Sampling without replacement
    matters wherever the consumer runs its own cross-validation: duplicated
    rows would land in both train and test folds.

    Falls back to the full data when stratification is impossible (a class with
    too few members), since a slightly less diverse replicate is better than a
    failed one.

    Parameters
    ----------
    X : pd.DataFrame
    y : np.ndarray
    seed : int
    fraction : float
        Portion of rows to keep.

    Returns
    -------
    (X_sub, y_sub)
    """
    n = len(X)
    n_keep = max(int(round(n * fraction)), len(np.unique(y)) * 2)
    if n_keep >= n:
        return X, y
    try:
        splitter = StratifiedShuffleSplit(
            n_splits=1, train_size=n_keep, random_state=seed)
        idx, _ = next(splitter.split(X, y))
    except ValueError:
        return X, y
    return X.iloc[idx], y[idx]


def _rfecv_scoring(algorithm: str, n_classes: int) -> str:
    """
    Return an RFECV scoring string the algorithm's estimator can actually serve.

    The SVM path uses a LinearSVC surrogate, which exposes ``decision_function``
    but not ``predict_proba``.  sklearn's ``roc_auc_ovr`` scorer requires
    ``predict_proba`` and raises ``AttributeError`` on every fit for that
    estimator, which the caller's try/except would silently swallow -- leaving
    the RFECV stage with an all-features-selected default and a misleading
    failure count.  Pick a scorer each estimator supports instead:

    - binary, any algorithm: ``roc_auc`` (accepts decision_function or proba)
    - multiclass, proba-capable algorithms: ``roc_auc_ovr``
    - multiclass, LinearSVC surrogate: ``balanced_accuracy`` (label-based, so
      it needs only ``predict``)

    Parameters
    ----------
    algorithm : str
        One of 'LR', 'XGB', 'RF', 'SVM'.
    n_classes : int
        Number of classes in the target.

    Returns
    -------
    str
        A scoring name accepted by ``sklearn.feature_selection.RFECV``.
    """
    if n_classes == 2:
        return 'roc_auc'
    if algorithm == 'SVM':
        return 'balanced_accuracy'
    return 'roc_auc_ovr'


def _extract_importance_step3(
    model: Pipeline,
    algorithm: str,
) -> np.ndarray:
    """
    Return a normalized [0, 1] importance array from a fitted Step 3 pipeline.

    For SVM, extracts coef_ from the LinearSVC surrogate (not compute_svm_afi,
    which is for the RBF SVC used in evaluation).

    Parameters
    ----------
    model : fitted sklearn Pipeline with named step 'clf'.
    algorithm : str

    Returns
    -------
    np.ndarray, shape (n_features,), values in [0, 1].
    """
    clf = model.named_steps["clf"]

    if algorithm == "LR":
        imp = np.abs(clf.coef_).max(axis=0)
    elif algorithm in ("XGB", "RF"):
        imp = clf.feature_importances_
    elif algorithm == "SVM":
        # coef_ from LinearSVC surrogate.
        imp = np.abs(clf.coef_).max(axis=0)
    else:
        imp = np.zeros(model.named_steps["scaler"].n_features_in_)

    if imp.max() > 0:
        imp = imp / imp.max()
    return imp


# ---------------------------------------------------------------------------
# Private helpers: Stage 1 - RFECV
# ---------------------------------------------------------------------------

def _run_rfecv_stage(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    algorithm: str,
    n_models: int,
    base_seed: int,
    target_retention: float,
    required_votes: int,
    cv: StratifiedKFold,
    warnings: Dict[str, int],
    verbose: bool,
    device: str = 'cpu',
    n_jobs: int = -1,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], float]:
    """
    Run binary-search percentile optimisation then full RFECV voting pass.

    Binary search targets ``target_retention`` feature retention.
    Seeds: base_seed + i*_RFECV_SEED_STEP for model i.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : np.ndarray
    algorithm : str
    n_models : int
    base_seed : int
    target_retention : float
        Fraction of the input to aim to keep; the search runs over the full
        [0, 1] percentile domain to reach it.
    required_votes : int
    cv : StratifiedKFold (used only to determine n_splits for internal CV).
    warnings : dict - mutated in place.
    verbose : bool

    Returns
    -------
    rfecv_selected : np.ndarray of bool
    rfecv_vote_counts : np.ndarray of int
    rfecv_importances : list of np.ndarray
    best_rfecv_percentile : float
    """
    logger.info("STAGE 1: RFECV (Wrappers)")
    logger.info("Optimizing RFECV percentile threshold (binary search)...")

    n_features = X_train.shape[1]
    # Binary search requires evaluate_fn to be DECREASING (higher percentile
    # -> fewer features kept -> more features removed).  Return REMOVED count.
    target_features_removed = n_features * (1.0 - target_retention)

    # Internal RFECV CV: 3 folds or fewer if outer CV uses fewer.
    internal_cv_splits = min(3, cv.n_splits)

    # Scoring must be answerable by the estimator this algorithm supplies;
    # see _rfecv_scoring for the SVM/LinearSVC constraint.
    scoring = _rfecv_scoring(algorithm, int(len(np.unique(y_train))))
    logger.info("RFECV scoring metric: %s", scoring)

    def rfecv_evaluate_fn(percentile: float) -> float:
        """Fit n_models RFECV instances and return mean n_features REMOVED."""
        counts = []
        for i in range(n_models):
            seed = base_seed + i * _RFECV_SEED_STEP
            estimator, importance_getter = _build_rfecv_estimator(
                algorithm, seed, device=device, n_jobs=n_jobs)
            rfecv_cv = StratifiedKFold(
                n_splits=internal_cv_splits,
                shuffle=True,
                random_state=seed,
            )
            rfecv = RFECV(
                estimator=estimator,
                step=_rfecv_step(n_features),
                cv=rfecv_cv,
                scoring=scoring,
                n_jobs=1,
                min_features_to_select=max(5, int(n_features * percentile)),
                importance_getter=importance_getter,
            )
            try:
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore", ConvergenceWarning)
                    rfecv.fit(X_train, y_train)
                counts.append(n_features - rfecv.n_features_)  # REMOVED: DECREASING
            except Exception as e:
                warnings["rfecv_failed"] += 1
                if verbose:
                    logger.debug(
                        "  RFECV probe failed (model %d): %s: %s",
                        i + 1, type(e).__name__, e,
                    )
                # Fallback: approximate removed count (n_features - min_features_to_select)
                counts.append(int(n_features * (1.0 - percentile)))
        return float(np.mean(counts))

    best_percentile, best_avg_removed, n_evals = _binary_search_percentile(
        evaluate_fn=rfecv_evaluate_fn,
        lo=0.0,
        hi=1.0,
        target=target_features_removed,
        tolerance=0.01,
        max_iter=10,
        label="Step 3 RFECV retention",
    )

    logger.info(
        "Best RFECV percentile: %.4f (found in %d evaluations vs 11 for grid search)",
        best_percentile, n_evals,
    )
    logger.info(
        "Expected features removed: %.0f (retained: ~%.0f)",
        best_avg_removed, n_features - best_avg_removed,
    )

    # ------------------------------------------------------------------
    # Voting pass at best percentile
    # ------------------------------------------------------------------
    logger.info("Applying RFECV with consensus voting...")

    rfecv_vote_counts = np.zeros(n_features, dtype=int)
    rfecv_importances: List[np.ndarray] = []

    for i in range(n_models):
        seed = base_seed + i * _RFECV_SEED_STEP
        estimator, importance_getter = _build_rfecv_estimator(
            algorithm, seed, device=device, n_jobs=n_jobs)
        rfecv_cv = StratifiedKFold(
            n_splits=internal_cv_splits,
            shuffle=True,
            random_state=seed,
        )
        rfecv = RFECV(
            estimator=estimator,
            step=_rfecv_step(n_features),
            cv=rfecv_cv,
            scoring=scoring,
            n_jobs=1,
            min_features_to_select=max(5, int(n_features * best_percentile)),
            importance_getter=importance_getter,
        )
        # Subsample the rows so seeds produce genuinely different RFECV runs.
        # Without this the estimators are deterministic given their data for LR
        # and XGB, every model votes identically, and the vote level has no
        # effect (measured: max pairwise importance difference exactly 0.0
        # across five seeds).
        #
        # Sampling is WITHOUT replacement here, unlike Step 2's bootstrap:
        # RFECV runs its own internal cross-validation, and duplicated rows
        # would appear in both its train and test folds, leaking and biasing
        # the scores it selects on.
        if n_models > 1:
            X_fit, y_fit = _subsample_rows(X_train, y_train, seed)
        else:
            X_fit, y_fit = X_train, y_train

        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", ConvergenceWarning)
                rfecv.fit(X_fit, y_fit)
            rfecv_vote_counts += rfecv.support_.astype(int)

            # Re-fit a plain model on full X_train for importance extraction.
            # rfecv.estimator_ holds the refitted estimator on the selected
            # subset; we need importances aligned to the full feature space.
            # clone() is required to prevent aliasing: sharing the clf object
            # directly would mutate rfecv.estimator_ when the plain model is
            # re-fitted on the full feature space.
            plain_model = Pipeline([
                ("scaler", MinMaxScaler()),
                ("clf", _clone(rfecv.estimator_.named_steps["clf"])),
            ])
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", ConvergenceWarning)
                plain_model.fit(X_train, y_train)
            rfecv_importances.append(
                _extract_importance_step3(plain_model, algorithm)
            )

            if verbose:
                logger.debug("  Model %d: %d features selected", i + 1, rfecv.n_features_)

        except Exception as e:
            warnings["rfecv_failed"] += 1
            if verbose:
                logger.debug(
                    "  RFECV voting fit failed (model %d): %s: %s",
                    i + 1, type(e).__name__, e,
                )
            # On failure, all features are treated as selected so that
            # consensus counting remains valid.
            rfecv_vote_counts += np.ones(n_features, dtype=int)
            rfecv_importances.append(
                np.full(n_features, 1.0 / n_features)
            )

    rfecv_selected = rfecv_vote_counts >= required_votes
    logger.info(
        "RFECV consensus selected: %d / %d features",
        rfecv_selected.sum(), n_features,
    )

    return rfecv_selected, rfecv_vote_counts, rfecv_importances, best_percentile


# ---------------------------------------------------------------------------
# Private helpers: Stage 2 - Stability selection
# ---------------------------------------------------------------------------

def _run_stability_stage(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    algorithm: str,
    n_models: int,
    base_seed: int,
    stability_threshold: float,
    min_features: int,
    warnings: Dict[str, int],
    verbose: bool = False,
    device: str = 'cpu',
    n_jobs: int = -1,
) -> Tuple[np.ndarray, Optional[np.ndarray], List[np.ndarray]]:
    """
    Run stability selection over STABILITY_N_SUBSAMPLES stratified subsamples.

    Subsample seed: base_seed + i*500.
    Model seed: base_seed + (i % n_models)*400 + 500.
    The +500 offset separates stability models from RFECV models.
    Cycling (i % n_models) across 50 subsamples limits model variety
    to n_models distinct seeds while distributing uniformly.

    A feature is considered "selected" on a subsample if its normalized
    importance exceeds the subsample mean.  Final selection requires
    selection_freq >= stability_threshold across all valid subsamples.

    Reference: Meinshausen N, Buhlmann P (2010). Stability selection.
    J R Stat Soc B 72:417-473. doi:10.1111/j.1467-9868.2010.00740.x

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : np.ndarray
    algorithm : str
    n_models : int
    base_seed : int
    stability_threshold : float
    min_features : int
        Used to determine minimum subsample size.
    warnings : dict - mutated in place.

    Returns
    -------
    stability_selected : np.ndarray of bool
    stability_freq : np.ndarray of float or None
        None when all subsamples failed.
    stability_importances : list of np.ndarray
    """
    logger.info("STAGE 2: STABILITY SELECTION")
    logger.info("Subsamples: %d, threshold: %s", STABILITY_N_SUBSAMPLES, stability_threshold)

    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]
    n_classes_local = int(len(np.unique(y_train)))

    # StratifiedShuffleSplit requires BOTH sides of the split to hold at least
    # one sample per class, so the complement must keep >= n_classes samples
    # (and the subsample itself must too).  Capping only at n_samples - 1 leaves
    # a 1-sample complement whenever min_features approaches n_samples -- which
    # happens routinely in the p >> n regime this package targets, because
    # min_features is 5% of p -- and StratifiedShuffleSplit then raises an
    # uncaught ValueError that aborts the whole pipeline.
    max_subsample = n_samples - n_classes_local
    subsample_size = min(max(int(n_samples * 0.5), min_features + 1), max_subsample)
    subsample_size = max(subsample_size, n_classes_local)

    if subsample_size < n_classes_local or subsample_size >= n_samples:
        warnings["stability_all_failed"] += 1
        logger.warning(
            "Stability selection needs at least %d samples per split half but "
            "only %d training samples are available. Skipping the stability "
            "stage and retaining all features for it.",
            n_classes_local * 2, n_samples,
        )
        return np.ones(n_features, dtype=bool), None, []

    logger.info(
        "Subsample size: %d of %d training samples", subsample_size, n_samples,
    )

    selection_counts = np.zeros(n_features, dtype=int)
    stability_importances: List[np.ndarray] = []

    for i in range(STABILITY_N_SUBSAMPLES):
        if verbose and i % 10 == 0:
            logger.debug("  Stability subsample %d/%d...", i + 1, STABILITY_N_SUBSAMPLES)

        # Subsample seed: unique per subsample iteration.
        subsample_seed = base_seed + i * _SUBSAMPLE_SEED_STEP

        sss = StratifiedShuffleSplit(
            n_splits=1,
            train_size=subsample_size,
            random_state=subsample_seed,
        )
        try:
            sub_idx, _ = next(sss.split(X_train, y_train))
        except ValueError as exc:
            # Splitting is deterministic in its constraints, so a failure here
            # will recur for every remaining subsample.  Stop rather than
            # repeat the same error 50 times.
            warnings["stability_all_failed"] += 1
            logger.warning(
                "Stability subsampling is not possible for this class "
                "distribution (%s). Retaining all features for this stage.",
                exc,
            )
            return np.ones(n_features, dtype=bool), None, stability_importances

        X_sub = X_train.iloc[sub_idx]
        y_sub = y_train[sub_idx]

        # Skip subsamples where any class is absent; per-class metrics
        # would be undefined and model fitting may fail.
        if len(np.unique(y_sub)) < len(np.unique(y_train)):
            warnings["stability_skipped_class"] += 1
            continue

        # Model seed cycles across n_models; offset separates from RFECV seeds.
        model_seed = base_seed + (i % n_models) * _RFECV_SEED_STEP + _STABILITY_SEED_OFFSET
        estimator, _ = _build_rfecv_estimator(
            algorithm, model_seed, device=device, n_jobs=n_jobs)

        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", ConvergenceWarning)
                estimator.fit(X_sub, y_sub)
            imp = _extract_importance_step3(estimator, algorithm)
            stability_importances.append(imp)

            # Feature is "selected" on this subsample if importance > mean.
            # Using the subsample mean as threshold is scale-invariant and
            # adapts to the data without a fixed cutoff.
            threshold = imp.mean()
            selection_counts += (imp > threshold).astype(int)

        except Exception as e:
            warnings["stability_failed"] += 1
            if verbose:
                logger.debug(
                    "  Stability fit failed (subsample %d): %s: %s",
                    i + 1, type(e).__name__, e,
                )
            continue

    # ------------------------------------------------------------------
    # Compute selection frequencies and apply threshold
    # ------------------------------------------------------------------
    n_valid = len(stability_importances)

    if n_valid == 0:
        warnings["stability_all_failed"] += 1
        # Conservative fallback: retain all features so the consensus
        # intersection degenerates to the RFECV result only.
        stability_selected = np.ones(n_features, dtype=bool)
        stability_freq = None
        logger.warning(
            "All stability subsamples failed. Retaining all features for this stage."
        )
    else:
        stability_freq = selection_counts / n_valid
        stability_selected = stability_freq >= stability_threshold
        logger.info("Valid subsamples: %d / %d", n_valid, STABILITY_N_SUBSAMPLES)
        logger.info(
            "Stability selected: %d / %d features",
            stability_selected.sum(), n_features,
        )
        logger.info(
            "Selection frequency range: [%.3f, %.3f]",
            stability_freq.min(), stability_freq.max(),
        )

    return stability_selected, stability_freq, stability_importances


# ---------------------------------------------------------------------------
# Private helpers: warning printer
# ---------------------------------------------------------------------------

def _print_warnings(warnings: Dict[str, int]) -> None:
    """Log accumulated Step 3 warnings if any occurred."""
    total = sum(warnings.values())
    if total == 0:
        return

    logger.warning("STEP 3 WARNINGS (accumulated):")
    if warnings["rfecv_failed"] > 0:
        logger.warning(
            "    RFECV failures: %d (affected models defaulted to full feature set)",
            warnings['rfecv_failed'],
        )
    if warnings["stability_skipped_class"] > 0:
        logger.warning(
            "    Stability subsamples skipped (missing class): %d",
            warnings['stability_skipped_class'],
        )
    if warnings["stability_failed"] > 0:
        logger.warning(
            "    Stability model fit failures: %d",
            warnings['stability_failed'],
        )
    if warnings["stability_all_failed"] > 0:
        logger.warning(
            "    All stability subsamples failed: stability stage "
            "defaulted to retaining all features"
        )


# ---------------------------------------------------------------------------
# Private helpers: evaluation block
# ---------------------------------------------------------------------------

def _evaluate_step3(
    X_rfecv: pd.DataFrame,
    X_train_original: pd.DataFrame,
    y_train: np.ndarray,
    config: SelectOmicsConfig,
    cv: StratifiedKFold,
    tuned_pipelines: Dict[str, Any],
    n_classes: int,
    class_names: List[str],
    ref_result_step0: Optional[Dict[str, Any]],
    adequacy: Optional[Dict[str, Any]],
    output_dir: Path,
    base_seed: int,
    n_models: int,
    algorithm: str,
    consensus_summary: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Run the Step 3 consensus evaluation and comparison against Step 0.

    Builds n_models consensus pipelines (seeds base_seed + i*100), evaluates
    each on X_rfecv with bootstrap resampling, prints comparison against
    Step 0 reference.

    Parameters
    ----------
    X_rfecv : pd.DataFrame
    X_train_original : pd.DataFrame
    y_train : np.ndarray
    config : SelectOmicsConfig
    cv : StratifiedKFold
    tuned_pipelines : dict
    n_classes : int
    class_names : list of str
    ref_result_step0 : dict or None
    adequacy : dict or None
    output_dir : Path
    base_seed : int
    n_models : int
    algorithm : str

    Returns
    -------
    dict or None
    """
    logger.info("STEP 3 EVALUATION")

    if adequacy is not None:
        _print_adequacy_warning(
            step_name="Step 3: Wrappers",
            n_features_current=X_rfecv.shape[1],
            adequacy=adequacy,
        )

    step3_models: Dict[str, Any] = {}

    if n_models == 1:
        step3_models[f"Wrappers-{algorithm}"] = build_consensus_pipeline(
            algorithm=algorithm,
            seed=base_seed,
            tuned_pipelines=tuned_pipelines,
        )
    else:
        for i in range(n_models):
            seed = base_seed + i * _EVAL_SEED_STEP
            step3_models[f"Wrappers-{algorithm}-{i+1}"] = build_consensus_pipeline(
                algorithm=algorithm,
                seed=seed,
                tuned_pipelines=tuned_pipelines,
            )

    step3_evaluation = evaluate_models_collection(
        pipelines=step3_models,
        X=X_rfecv,
        y=y_train,
        cv=cv,
        n_classes=n_classes,
        classes=np.arange(n_classes),
        config=config,
        step_name=f"Step 3: Wrappers - All {algorithm} Models",
        use_bootstrap=True,
    )

    if n_models == 1:
        consensus_result = list(step3_evaluation["results"].values())[0]
    else:
        best_name = max(
            step3_evaluation["results"],
            key=lambda k: step3_evaluation["results"][k]["mean_auc"],
        )
        consensus_result = step3_evaluation["results"][best_name]

    logger.info("Step 3 Individual Model Performance:")
    for name, result in step3_evaluation["results"].items():
        logger.info(
            "  %s: AUC = %.4f +/- %.4f",
            name, result['mean_auc'], result['std_auc'],
        )
    logger.info(
        "  Feature count: %d (reduced from %d)",
        X_rfecv.shape[1], X_train_original.shape[1],
    )

    logger.info("STEP 3: CONSENSUS vs REFERENCE COMPARISON")

    if ref_result_step0 is not None:
        logger.info(
            "  Reference %s: %.4f +/- %.4f",
            algorithm, ref_result_step0['mean_auc'], ref_result_step0['std_auc'],
        )
    else:
        logger.warning("  No Step 0 reference provided")

    logger.info(
        "  Consensus-%s: %.4f +/- %.4f",
        algorithm, consensus_result['mean_auc'], consensus_result['std_auc'],
    )

    if config.create_visualizations:
        _plot_step3(
            step3_evaluation=step3_evaluation,
            ref_result_step0=ref_result_step0,
            y_train=y_train,
            n_classes=n_classes,
            algorithm=algorithm,
            config=config,
            output_dir=output_dir,
        )

    if config.save_intermediate_results:
        step_summary = {
            "algorithm": algorithm,
            "n_consensus_models": n_models,
            "input_features": X_train_original.shape[1],
            "output_features": X_rfecv.shape[1],
            "features_removed": X_train_original.shape[1] - X_rfecv.shape[1],
            # How strong a claim this panel is: votes_required, agreement
            # and agreement_label belong in the saved record, not just the log.
            **(consensus_summary or {}),
        }
        for name, result in step3_evaluation["results"].items():
            clean_name = name.lower().replace("-", "_")
            step_summary[f"{clean_name}_auc"] = result["mean_auc"]
            step_summary[f"{clean_name}_std"] = result["std_auc"]

        output_dir.mkdir(parents=True, exist_ok=True)
        save_step_summary("step3", step_summary, output_dir)

    logger.info("Step 3 evaluation complete.")
    return consensus_result


def _plot_step3(
    step3_evaluation: Dict[str, Any],
    ref_result_step0: Optional[Dict[str, Any]],
    y_train: np.ndarray,
    n_classes: int,
    algorithm: str,
    config: SelectOmicsConfig,
    output_dir: Path,
) -> None:
    """
    Generate Step 3 diagnostic plots: AUC box plots and ROC curves.

    Parameters
    ----------
    step3_evaluation : dict
    ref_result_step0 : dict or None
    y_train : np.ndarray
    n_classes : int
    algorithm : str
    config : SelectOmicsConfig
    output_dir : Path
    """
    from ..evaluation.visualization import plot_auc_boxplots, plot_roc_curves

    output_dir.mkdir(parents=True, exist_ok=True)

    comparison: Dict[str, Any] = {}
    if ref_result_step0 is not None:
        comparison[f"Ref-{algorithm}"] = ref_result_step0

    best_name = max(
        step3_evaluation["results"],
        key=lambda k: step3_evaluation["results"][k]["mean_auc"],
    )
    comparison[f"Consensus-{algorithm}"] = step3_evaluation["results"][best_name]

    fold_aucs = {name: result["fold_aucs"] for name, result in comparison.items()}

    plot_auc_boxplots(
        step_name="Step 3: Wrappers Consensus vs Reference",
        aucs_dict=fold_aucs,
        config=config,
        save_path=output_dir / "step3_consensus_vs_reference_boxplots",
    )

    plot_roc_curves(
        models_dict=comparison,
        y_true=y_train,
        n_classes=n_classes,
        classes=np.arange(n_classes),
        config=config,
        title="Step 3: Wrappers Consensus vs Reference ROC",
        save_path=output_dir / "step3_consensus_vs_reference_roc",
    )
