"""
SelectOmics/selection/step2_regularization.py

Step 2: Model-based feature filtering via parallel L1 and L2 regularization.

Runs two parallel regularization stages (L1-style and L2-style) using the
active algorithm, then applies independent binary searches to find the
best l1_percentile and l2_percentile (targeting ~40% feature drop each).
A full n_models voting pass is then run at the best thresholds.  The final
kept set is the intersection of the two voting masks, with the same
fallback hierarchy used in Step 1.

Design decisions
----------------
- Parallel L1/L2 stages rather than sequential: running both simultaneously
  and taking their consensus intersection is more principled than applying
  one filter then the other, as neither stage's threshold has been
  calibrated against the output of the other.
- Algorithm-specific regularization interpretation:
    LR  - L1 penalty (saga, sparsity) vs L2 penalty (standard ridge)
    XGB - alpha-dominant (L1-style) vs lambda-dominant (L2-style)
    RF  - sparse tree settings (few estimators, larger leaves) vs
          conservative settings (more estimators, smaller leaves)
    SVM - high regularization (low C) vs low regularization (high C)
- The 7x7 grid search uses one model per stage per grid point for
  efficiency.  The seed is fixed (config.random_seed) for the grid pass
  so results are reproducible.  The full n_models voting pass runs only
  once at the best thresholds, with seeds config.random_seed + i*200 for
  L1 and + i*200 + 100 for L2 to keep stages distinct.
- Importance extraction: LR uses max absolute coefficient across classes,
  XGB/RF use feature_importances_, SVM uses compute_svm_afi (weighted
  variance of support vectors scaled by gamma).
- Thresholding is by importance RANK, capped at the number of features the
  model gave a non-zero importance to.  Comparing against a percentile of the
  importance distribution does not work here: tree importance is exactly zero
  for every feature never split on, and that tie mass is 98-99.8% of the array
  on this package's target data, so the percentile is 0.0 across the whole
  configured search range and the knob does nothing.  See _rank_keep_mask.
- Panel size is set by config.min_features_floor. The graduated relaxation
  starts at unanimity and lowers the vote requirement until the floor is met,
  so the floor is what to change if Step 2 is too aggressive; the vote level
  it settled at is reported as 'votes_required' / 'agreement_label' and says
  how much agreement the data actually supported. config.min_consensus is the
  brake on that descent.
- When Step 2 is disabled (enable_step2=False) X_train and X_test are
  returned unchanged with all dropped lists empty.
  
"""

from __future__ import annotations

import logging
import warnings as _warnings
from pathlib import Path

logger = logging.getLogger(__name__)
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

from ..config import SelectOmicsConfig
from ..models.base import build_consensus_pipeline, xgb_compute_kwargs
from ..utils.helpers import (
    _binary_search_percentile,
    relax_consensus_intersection,
)

# ---------------------------------------------------------------------------
# Seed scheme
# ---------------------------------------------------------------------------

# L1 voting pass: base_seed + i * _L1_SEED_STEP
# L2 voting pass: base_seed + i * _L1_SEED_STEP + _L2_SEED_OFFSET
# The offset keeps the two stages' models on distinct seed sequences.
_L1_SEED_STEP:   int = 200
_L2_SEED_OFFSET: int = 100

# Step 2 consensus evaluation models reuse Step 1's spacing (100) for
# consistency with the other step evaluations.
_EVAL_SEED_STEP: int = 100
from ..models.evaluation import evaluate_models_collection
from ..utils.helpers import save_step_summary
from ..data.loaders import save_from_config
from .step0_reference import _print_adequacy_warning


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_step2_regularization(
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
    Apply parallel L1 and L2 regularization-based feature selection.

    Runs independent binary searches over l1_percentile and l2_percentile
    to find the thresholds that drop ~40% of features in each stage.
    A full n_models consensus voting pass is then run at the best thresholds.
    The final kept feature set is the intersection of the L1 and L2 voting
    masks, with union and top-k fallbacks if the intersection is too small.

    When ``config.enable_step2`` is False the function returns X_train and
    X_test unchanged.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix from Step 1 (or original if Step 1 disabled).
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
        Reference CV result from Step 0 for consensus vs reference comparison.
    adequacy : dict, optional
        Output of ``assess_sample_size_adequacy`` for warnings.
    output_dir : Path, optional
        Artefact output directory; derived from ``config.output_dir`` if None.

    Returns
    -------
    dict with keys:
        X_train_reg : pd.DataFrame
            Training matrix after regularization filtering.
        X_test_reg : pd.DataFrame
            Test matrix with the same column subset.
        dropped_features : list of str
            Features removed by the consensus of L1 and L2 stages.
        best_l1_percentile : float
            Optimal L1 stage percentile from the grid search.
        best_l2_percentile : float
            Optimal L2 stage percentile from the grid search.
        l1_selected : np.ndarray of bool
            Feature kept mask from the L1 voting pass.
        l2_selected : np.ndarray of bool
            Feature kept mask from the L2 voting pass.
        consensus_result : dict or None
            CV evaluation result for the regularized feature set.
        n_features_in : int
            Number of features entering Step 2.
        n_features_out : int
            Number of features after Step 2.
    """
    algorithm = config.algorithm

    if output_dir is None:
        output_dir = Path(config.output_dir)

    # Inherit GPU device from the tuned XGB pipeline when available,
    # mirroring the pattern used in pipeline.py and step3_wrapper.py.
    _colsample: float = float(getattr(config, 'step2_stage_colsample',
                                      _COLSAMPLE_BYTREE))
    _device: str = 'cpu'
    if algorithm == 'XGB' and tuned_pipelines:
        try:
            _device = tuned_pipelines['XGB'].named_steps['clf'].get_params().get('device', 'cpu')
        except Exception:
            _device = 'cpu'

    from ..models.base import effective_n_jobs, resolve_n_jobs
    _n_jobs = effective_n_jobs(resolve_n_jobs(config), *X_train.shape)

    logger.info("STEP 2: MODEL-BASED FILTERS (REGULARIZATION) - %s", algorithm)

    # ------------------------------------------------------------------
    # Disabled branch
    # ------------------------------------------------------------------
    if not config.enable_step2:
        logger.info("Step 2 disabled - using input data unchanged")
        logger.info("  Features: %d", X_train.shape[1])
        return {
            "X_train_reg": X_train.copy(),
            "X_test_reg": X_test.copy(),
            "dropped_features": [],
            "best_l1_percentile": 0.0,
            "best_l2_percentile": 0.0,
            "l1_selected": np.ones(X_train.shape[1], dtype=bool),
            "l2_selected": np.ones(X_train.shape[1], dtype=bool),
            "votes_required": 0,
            "agreement": 0.0,
            "agreement_label": "not run",
            "consensus_outcome": "disabled",
            # Present so the disabled branch returns the same keys as the
            # normal one. Without them a caller reading these worked on an
            # ordinary run and raised KeyError only when Step 2 was turned
            # off, which is the hardest version of that bug to notice. No
            # consensus runs here, so both are zero, matching the neutral
            # values the rest of this branch uses.
            "consensus_floor_used": 0,
            "consensus_floor_requested": 0,
            "n_models": config.n_consensus_models,
            "consensus_result": ref_result_step0,
            "n_features_in": X_train.shape[1],
            "n_features_out": X_train.shape[1],
        }

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    n_models = config.n_consensus_models
    min_consensus = config.min_consensus
    base_seed = config.random_seed

    # Step 2 as a prefilter rather than the final word.
    #
    # The floor is set from Step 3's entry gate rather than from
    # min_features_floor, because the point is to guarantee Step 3 engages with
    # room to eliminate. The agreement brake comes off, since a candidate set
    # is allowed to be over-inclusive: Step 3 is what makes it precise.
    _role = getattr(config, "step2_role", "terminal")
    if _role == "prefilter":
        from .step3_wrapper import MIN_FEATURES_FOR_RFECV
        floor = max(config.min_features_floor,
                    _PREFILTER_GATE_MULTIPLE * MIN_FEATURES_FOR_RFECV)
        min_consensus = None
        logger.info(
            "Step 2 role: prefilter. Targeting >=%d features (%dx the Step 3 "
            "gate of %d) and ignoring min_consensus, so Step 3 receives a "
            "candidate set it can actually eliminate from.",
            floor, _PREFILTER_GATE_MULTIPLE, MIN_FEATURES_FOR_RFECV,
        )
    else:
        floor = config.min_features_floor

    logger.info("Using %s with %d consensus model(s)", algorithm, n_models)
    logger.info(
        "Starting at unanimity (%d/%d), relaxing until %d features are "
        "reached%s",
        n_models, n_models, min(floor, X_train.shape[1]),
        "" if min_consensus is None
        else f", but not below {min_consensus:.0%} agreement",
    )
    logger.info("Input features: %d", X_train.shape[1])

    # Absolute safety net, not a retention target; see config.min_features_floor.
    # In prefilter role this is raised to Step 3's gate; see above.
    min_features = min(floor, X_train.shape[1])

    # Accumulates importances across all models in both stages.
    # Used only if both intersection and union fall below min_features.
    all_importances_combined: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Binary search: find l1_percentile and l2_percentile independently.
    # Target: retain ~60% of features (drop ~40%).
    # Independent binary searches replace the 7x7 grid search.
    # ------------------------------------------------------------------
    # Binary search requires evaluate_fn to be DECREASING (higher percentile
    # -> fewer features kept).  Return KEPT count, not DROP count.
    #
    # The search runs over the full [0, 1] percentile domain.  It used to be
    # bounded by config.l1/l2_percentile_range, which asked the caller to
    # constrain a solver's domain for a quantity they never see, and could
    # silently put the retention they DID ask for out of reach.
    # step2_target_retention is a fraction of the REACHABLE pool, not of the
    # input width.
    #
    # It used to be `X_train.shape[1] * retention`, which is unreachable in this
    # package's target regime and therefore inert. _rank_keep_mask caps the
    # selection at the number of features with non-zero importance, because
    # selecting past that means selecting features the model never used. For a
    # tree on wide data that cap is around 1% of the input: measured, one XGB
    # stage fit used 21 of 2000 features and 30 of 10000. So a request to keep
    # 60% of 2000 clamped to 21, and so did a request for 30% or 90% -- every
    # setting above roughly 1% produced the identical result.
    #
    # Probing the pool at percentile 0.0 costs one extra stage fit per stage and
    # makes the knob live across its whole documented range.
    def l1_evaluate(p: float) -> float:
        mask = _estimate_drop_mask(X_train, y_train, algorithm, "l1", p, base_seed,
                                   device=_device, n_jobs=_n_jobs,
                                   colsample=_colsample)
        return float(X_train.shape[1] - mask.sum())  # KEPT count: DECREASING

    def l2_evaluate(p: float) -> float:
        mask = _estimate_drop_mask(X_train, y_train, algorithm, "l2", p, base_seed,
                                   device=_device, n_jobs=_n_jobs,
                                   colsample=_colsample)
        return float(X_train.shape[1] - mask.sum())  # KEPT count: DECREASING

    # Percentile 0.0 asks for everything the model used, so this measures the
    # reachable pool. One fit per stage.
    l1_pool = l1_evaluate(0.0)
    l2_pool = l2_evaluate(0.0)
    l1_target = max(1.0, l1_pool * config.step2_target_retention)
    l2_target = max(1.0, l2_pool * config.step2_target_retention)
    logger.info(
        "  Reachable pool: L1 %.0f, L2 %.0f of %d features "
        "(what the stage models actually used)",
        l1_pool, l2_pool, X_train.shape[1],
    )
    if max(l1_pool, l2_pool) < 0.02 * X_train.shape[1]:
        logger.info(
            "  Pool is under 2%% of input; raise config.step2_stage_colsample "
            "(currently %.2f) to widen it.",
            getattr(config, "step2_stage_colsample", _COLSAMPLE_BYTREE),
        )

    logger.info(
        "Binary search over L1 percentile (target: keep ~%.0f features)...",
        l1_target,
    )
    best_l1_percentile, _, n_l1_evals = _binary_search_percentile(
        evaluate_fn=l1_evaluate,
        lo=0.0,
        hi=1.0,
        target=l1_target,
        tolerance=0.01,
        max_iter=10,
        label="Step 2 L1 retention",
    )

    logger.info(
        "Binary search over L2 percentile (target: keep ~%.0f features)...",
        l2_target,
    )
    best_l2_percentile, _, n_l2_evals = _binary_search_percentile(
        evaluate_fn=l2_evaluate,
        lo=0.0,
        hi=1.0,
        target=l2_target,
        tolerance=0.01,
        max_iter=10,
        label="Step 2 L2 retention",
    )

    logger.info("  Best L1 percentile: %.4f (%d evals)", best_l1_percentile, n_l1_evals)
    logger.info("  Best L2 percentile: %.4f (%d evals)", best_l2_percentile, n_l2_evals)

    # ------------------------------------------------------------------
    # Voting pass: full n_models consensus at best thresholds.
    # L1 seeds: base_seed + i*200 + 0
    # L2 seeds: base_seed + i*200 + 100  (offset keeps stages distinct)
    # ------------------------------------------------------------------
    l1_selected, l1_importances, l1_votes = _run_voting_pass(
        X_train=X_train,
        y_train=y_train,
        algorithm=algorithm,
        stage_key="l1",
        best_percentile=best_l1_percentile,
        n_models=n_models,
        base_seed=base_seed,
        seed_offset=0,
        stage_label=_stage_label(algorithm, "l1"),
        device=_device,
        n_jobs=_n_jobs,
        colsample=_colsample,
    )
    all_importances_combined.extend(l1_importances)

    l2_selected, l2_importances, l2_votes = _run_voting_pass(
        X_train=X_train,
        y_train=y_train,
        algorithm=algorithm,
        stage_key="l2",
        best_percentile=best_l2_percentile,
        n_models=n_models,
        base_seed=base_seed,
        seed_offset=_L2_SEED_OFFSET,
        stage_label=_stage_label(algorithm, "l2"),
        device=_device,
        n_jobs=_n_jobs,
        colsample=_colsample,
    )
    all_importances_combined.extend(l2_importances)

    # ------------------------------------------------------------------
    # Consensus intersection, relaxed only as far as necessary
    #
    # The intersection rule is never abandoned.  If unanimity leaves too few
    # features the vote requirement is lowered one step at a time and the
    # intersection recomputed, so the result always means "every stage agreed,
    # at this level of agreement".  The previous behaviour fell straight from
    # intersection to union, which for two stages is the loosest setting that
    # exists -- accepting anything either stage proposed.
    # ------------------------------------------------------------------
    logger.info("Consensus intersection (L1 AND L2):")
    logger.info("  L1 selected:  %d features", l1_selected.sum())
    logger.info("  L2 selected:  %d features", l2_selected.sum())

    consensus = relax_consensus_intersection(
        vote_arrays=[l1_votes, l2_votes],
        n_models=n_models,
        min_features=min_features,
        mean_importance=(np.mean(all_importances_combined, axis=0)
                         if all_importances_combined else None),
        min_consensus=min_consensus,
        allow_union=getattr(config, 'allow_union_rung', False),
    )
    final_selected = consensus.mask
    votes_required = consensus.votes_required
    how = consensus.how

    if consensus.floor_used < consensus.floor_requested:
        logger.info(
            "  Minimum feature count capped at %d (asked for %d): only %d "
            "features were selected by any model in both stages, so a larger "
            "floor could only be met by padding with features nothing chose.",
            consensus.floor_used, consensus.floor_requested,
            consensus.floor_used,
        )

    if how == 'consensus':
        logger.info(
            "  Unanimous: %d features agreed by all %d models in both stages "
            "(agreement: %s).",
            final_selected.sum(), n_models, consensus.label,
        )
    elif how == 'relaxed':
        logger.info(
            "  Relaxed from %d/%d to %d/%d votes per stage to reach the "
            "%d-feature minimum. %d features, agreement %.0f%% (%s). The "
            "intersection rule still holds at that level.",
            n_models, n_models, votes_required, n_models,
            consensus.floor_used, final_selected.sum(),
            consensus.agreement * 100, consensus.label,
        )
    elif how == 'consensus_limited':
        logger.warning(
            "  min_consensus stopped the relaxation at %d/%d votes (%s) "
            "before the %d-feature minimum was reached: only %d features "
            "survive at the agreement you set as acceptable. Lower "
            "min_consensus for a larger panel, or accept this one.",
            votes_required, n_models, consensus.label,
            consensus.floor_requested, final_selected.sum(),
        )
    else:
        logger.warning(
            "  No feature was selected by any model in both stages. Falling "
            "back to the top %d by mean importance within the voted pool; "
            "treat this result as ranked, not agreed.",
            consensus.floor_used,
        )

    X_reg = X_train.loc[:, final_selected]
    X_test_reg = X_test.loc[:, final_selected]
    dropped_features = X_train.columns[~final_selected].tolist()

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    logger.info("Step 2 Summary:")
    logger.info("  Input features:  %d", X_train.shape[1])
    logger.info("  After consensus: %d", X_reg.shape[1])
    logger.info("  Total removed:   %d", X_train.shape[1] - X_reg.shape[1])

    # ------------------------------------------------------------------
    # Save intermediate results
    # ------------------------------------------------------------------
    if config.save_intermediate_results:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_from_config(
            pd.DataFrame({"dropped_by_regularization": dropped_features}),
            output_dir / "step2_dropped_features",
            config,
        )
        save_from_config(
            X_reg,
            output_dir / "step2_X_reg_clean",
            config,
        )
        # Record the selection facts now, before the optional evaluation.
        #
        # The fuller summary is written inside _evaluate_step2, which only
        # runs when enable_step_evaluations is True. Agreement is a property
        # of the SELECTION, not of the evaluation, so leaving it there meant a
        # run with evaluations disabled -- a common speed setting, and the one
        # the benchmark harness uses -- persisted no record of how much the
        # models agreed. The evaluation overwrites this file with the same
        # fields plus AUCs when it does run.
        save_step_summary("step2", {
            "algorithm": algorithm,
            "n_consensus_models": n_models,
            "input_features": X_train.shape[1],
            "output_features": X_reg.shape[1],
            "features_removed": X_train.shape[1] - X_reg.shape[1],
            "votes_required": votes_required,
            "agreement": consensus.agreement,
            "agreement_label": consensus.label,
            "consensus_outcome": how,
            "consensus_floor_used": consensus.floor_used,
            "consensus_floor_requested": consensus.floor_requested,
        }, output_dir)

    # ------------------------------------------------------------------
    # Step evaluation
    # ------------------------------------------------------------------
    consensus_result = None

    if config.enable_step_evaluations:
        consensus_result = _evaluate_step2(
            X_reg=X_reg,
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
                "votes_required": votes_required,
                "agreement": consensus.agreement,
                "agreement_label": consensus.label,
                "consensus_outcome": how,
                "consensus_floor_used": consensus.floor_used,
                "consensus_floor_requested": consensus.floor_requested,
            },
        )
    else:
        logger.info("Step 2 evaluation skipped.")

    logger.info("Ready for Step 3.")

    return {
        "X_train_reg": X_reg,
        "X_test_reg": X_test_reg,
        "dropped_features": dropped_features,
        "best_l1_percentile": best_l1_percentile,
        "best_l2_percentile": best_l2_percentile,
        "l1_selected": l1_selected,
        "l2_selected": l2_selected,
        # How much agreement the data actually supported. Because relaxation
        # always starts at unanimity, votes_required is the STRONGEST level at
        # which this panel holds, not an artefact of a starting point. 0 means
        # the rank-average fallback ran. Worth reading alongside the feature
        # list: a panel that needed 4/10 is a weaker claim than one at 10/10.
        "votes_required": votes_required,
        "agreement": consensus.agreement,
        "agreement_label": consensus.label,
        "consensus_outcome": how,
        "consensus_floor_used": consensus.floor_used,
        "consensus_floor_requested": consensus.floor_requested,
        "n_models": n_models,
        "consensus_result": consensus_result,
        "n_features_in": X_train.shape[1],
        "n_features_out": X_reg.shape[1],
    }


# ---------------------------------------------------------------------------
# Private helpers: stage model construction and importance extraction
# ---------------------------------------------------------------------------

# Inverse-regularisation ladder for the L1 stage.
#
# The original fixed C=0.1 fully sparsifies in the small-n / large-p regime
# this package targets: measured at n=75, p=2000, LogisticRegression(penalty=
# 'l1', solver='saga', C=0.1) returned zero non-zero coefficients out of 2000.
# Every importance was then 0, no feature could clear the vote threshold, and
# the L1 stage contributed an empty set on every run -- which forced the
# intersection empty and the fallback to fire unconditionally.
#
# Escalating only on total collapse keeps the intended strong regularisation
# wherever it produces a usable model, and costs one extra fit only when it
# does not.
_L1_C_LADDER: Tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)

# Fraction of features each XGB stage tree may split on.
#
# With all p features on offer, boosting converges onto the same handful in
# every tree, and the importance array Step 2 votes on is nearly empty: 14 to
# 18 non-zero of 1000 to 10000 across the synthetic development scenarios.
# Restricting each tree to a random 30% forces usage to spread, widening the
# pool to 20 to 30 and giving the consensus more to agree about.
#
# Ground-truth recovery at the default floor improves on omics_imbalanced
# (F1 0.571 -> 0.750) and is unchanged on omics_standard, omics_high_dim and
# omics_genomics; precision stays at 1.000 in all four, so the extra features
# recovered are real ones. It is also marginally FASTER, since each tree
# evaluates fewer candidate splits.
#
# 0.1 widens the pool further (34 to 64) but overshoots: recovery falls on
# three of the four scenarios. More trees do nothing at all -- 300 estimators
# reproduced the 100-estimator result exactly, at twice the cost.
# Default only. The live value comes from config.step2_stage_colsample and
# is threaded through _build_stage_model; this is the fallback for callers
# that build a stage model without a config.
# Prefilter aims at this multiple of Step 3's entry gate. Twice the gate
# clears it with margin, so Step 3 both runs and has features to remove
# rather than landing exactly on the threshold.
_PREFILTER_GATE_MULTIPLE: int = 2

_COLSAMPLE_BYTREE: float = 0.3


def _fit_stage_model(
    algorithm: str,
    stage: str,
    seed: int,
    X_fit,
    y_fit,
    colsample: float = _COLSAMPLE_BYTREE,
    device: str = 'cpu',
    n_jobs: int = -1,
) -> Pipeline:
    """
    Fit one stage model, escalating L1 regularisation if it collapses entirely.

    For every algorithm except the LR L1 stage this is a single fit.  For that
    one case, a model whose coefficients are all exactly zero carries no
    information at all, so C is raised one rung and the fit retried.

    Parameters
    ----------
    algorithm, stage, seed, device, n_jobs
        As for ``_build_stage_model``.
    X_fit, y_fit
        Training data for this replicate, already resampled by the caller.

    Returns
    -------
    Pipeline
        Fitted pipeline.  When every rung collapses, the last one is returned
        and the caller sees all-zero importances, which the consensus handles.
    """
    escalates = (algorithm == 'LR' and stage == 'l1')
    ladder = _L1_C_LADDER if escalates else (None,)

    model = None
    for c_value in ladder:
        model = _build_stage_model(algorithm, stage, seed, colsample=colsample,
                                   device=device, n_jobs=n_jobs)
        if c_value is not None:
            model.named_steps['clf'].set_params(C=c_value)
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(X_fit, y_fit)
        if not escalates:
            return model
        coef = getattr(model.named_steps['clf'], 'coef_', None)
        if coef is not None and np.abs(coef).max() > 0:
            if c_value != _L1_C_LADDER[0]:
                logger.debug(
                    "  L1 stage fully sparsified at C=%.3g; using C=%.3g.",
                    _L1_C_LADDER[0], c_value,
                )
            return model

    logger.warning(
        "  L1 stage produced no non-zero coefficients at any C in %s. "
        "This stage contributes nothing to the consensus for this data.",
        list(_L1_C_LADDER),
    )
    return model


def _build_stage_model(
    algorithm: str,
    stage: str,
    seed: int,
    colsample: float = _COLSAMPLE_BYTREE,
    device: str = 'cpu',
    n_jobs: int = -1,
) -> Pipeline:
    """
    Construct a sklearn Pipeline for one regularization stage.

    Parameters
    ----------
    algorithm : str
        Active algorithm key ('LR', 'XGB', 'RF', 'SVM').
    stage : str
        'l1' for sparsity-inducing settings, 'l2' for smoothing settings.
    seed : int
        Random seed for the classifier.
    device : str
        XGBoost compute device ('cpu' or 'cuda'). Ignored for non-XGB algorithms.
    n_jobs : int
        Threads per estimator; -1 uses all cores. Thread count does not
        change results, only wall-clock time.

    Returns
    -------
    sklearn.pipeline.Pipeline

    Raises
    ------
    ValueError
        If algorithm is not one of 'LR', 'XGB', 'RF', 'SVM'.
    ImportError
        If algorithm is 'XGB' and xgboost is not installed.
    """
    if algorithm == "LR":
        clf = (
            LogisticRegression(
                penalty="l1",
                solver="saga",
                C=0.1,
                max_iter=1000,
                random_state=seed,
            )
            if stage == "l1"
            else LogisticRegression(
                penalty="l2",
                C=0.1,
                max_iter=1000,
                random_state=seed,
            )
        )

    elif algorithm == "XGB":
        if not _XGB_AVAILABLE:
            raise ImportError("xgboost is required for algorithm='XGB'")
        # Penalty strength 1.0, not 10.0, keeping the alpha/lambda asymmetry
        # that distinguishes the two stages.
        #
        # At 10.0 the trees split on so few features that most importances are
        # exactly zero, and the vote pool is too small for unanimity to be
        # reachable at all: measured at 5 consensus models, ZERO features
        # cleared 5/5 on omics_high_dim and omics_imbalanced, so a
        # unanimity could never bind there. At 1.0 those
        # become 2 and 3 features respectively, and omics_genomics goes from 3
        # to 7.
        #
        # Ground-truth recovery improves on three of four development
        # scenarios (standard 0.727 -> 0.818, imbalanced 0.786 -> 0.867,
        # high_dim 0.571 -> 0.625); it falls on omics_genomics, from a perfect
        # 1.000 to 0.870. Dropping the penalty further to 0.0 is clearly worse
        # everywhere, with precision collapsing to 0.29-0.48, so this is not a
        # "less regularisation is better" trend to keep following.
        _L1_PENALTY, _L2_PENALTY = 1.0, 0.1
        clf = (
            XGBClassifier(
                reg_alpha=_L1_PENALTY,
                reg_lambda=_L2_PENALTY,
                colsample_bytree=colsample,
                random_state=seed,
                verbosity=0,
                eval_metric='mlogloss',
                **xgb_compute_kwargs(n_jobs, device),
            )
            if stage == "l1"
            else XGBClassifier(
                reg_alpha=_L2_PENALTY,
                reg_lambda=_L1_PENALTY,
                colsample_bytree=colsample,
                random_state=seed,
                verbosity=0,
                eval_metric='mlogloss',
                **xgb_compute_kwargs(n_jobs, device),
            )
        )

    elif algorithm == "RF":
        clf = (
            RandomForestClassifier(
                n_estimators=50,
                max_features="sqrt",
                min_samples_leaf=3,
                random_state=seed,
                n_jobs=n_jobs,
            )
            if stage == "l1"
            else RandomForestClassifier(
                n_estimators=100,
                max_features="log2",
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=n_jobs,
            )
        )

    elif algorithm == "SVM":
        # LinearSVC surrogate for coef_-based importance: same pattern as Step 3.
        # L1-like: high regularization (low C); L2-like: lower regularization (high C).
        clf = (
            LinearSVC(
                penalty="l1", dual="auto", C=0.1, max_iter=2000, random_state=seed
            )
            if stage == "l1"
            else LinearSVC(
                penalty="l2", dual="auto", C=1.0, max_iter=2000, random_state=seed
            )
        )

    else:
        raise ValueError(
            f"Unsupported algorithm '{algorithm}'. "
            f"Expected one of 'LR', 'XGB', 'RF', 'SVM'."
        )

    return Pipeline([("scaler", MinMaxScaler()), ("clf", clf)])


def _rank_keep_mask(imp: np.ndarray, percentile: float) -> np.ndarray:
    """
    Keep the top ``1 - percentile`` fraction by importance RANK, never
    extending the selection into the tie-at-zero region.

    The previous rule was ``imp > np.percentile(imp, percentile * 100)``, which
    on this package's target data does nothing at all.  Tree importance is
    exactly 0.0 for every feature never used in a split, and that zero mass is
    overwhelming: measured on the synthetic development scenarios, a single XGB
    stage fit left 1988 of 2000 importances at zero (99.4%), 4981 of 5000
    (99.6%), and 9983 of 10000 (99.8%).  ``np.percentile`` therefore returns
    0.0 at every point in the default search range of (0.0, 0.6), the test
    collapses to ``imp > 0``, and the kept count is identical at every
    percentile -- 12, 12, 12, 12, 12, 12, 12 across the grid on
    omics_standard.  The binary search over ``l1_percentile_range`` was
    optimising a constant function, and the requested 60% retention was never
    delivered: the stage kept 0.6% of its input instead of 60%.

    Ranking makes the knob real wherever the model gives a broad signal (LR
    with an L2 penalty has no exact zeros, so every percentile bites).  The
    cap at the non-zero count keeps it honest wherever it does not: a feature
    the model never split on carries no evidence for keeping it, and ranking
    past that point would order features by array index rather than by merit.
    Where the cap binds the behaviour matches the old rule exactly, so this is
    a strict improvement rather than a change of regime.

    Parameters
    ----------
    imp : np.ndarray, shape (n_features,)
        Non-negative importance; higher is better.
    percentile : float in [0, 1]
        Fraction of features to DROP.  0.0 keeps everything the model used.

    Returns
    -------
    np.ndarray of bool
        True where the feature is KEPT.
    """
    n_features = len(imp)
    n_nonzero = int(np.count_nonzero(imp))
    # Round before ceiling: (1.0 - 0.85) * 1000 evaluates to 150.00000000000003,
    # which would keep 151 features for a request of exactly 150.
    n_keep = min(int(np.ceil(round((1.0 - percentile) * n_features, 9))), n_nonzero)

    # The search domain now includes percentile 1.0, which asks for nothing at
    # all. Keep the single best feature instead: an empty stage forces the
    # intersection empty and fires the terminal rank-average fallback, turning
    # one extreme probe into a pipeline-wide loss of the consensus rule. An
    # all-zero importance array is different -- there is genuinely nothing to
    # keep, and the consensus is right to say so.
    if n_nonzero > 0:
        n_keep = max(1, n_keep)

    mask = np.zeros(n_features, dtype=bool)
    if n_keep > 0:
        # Stable sort so ties break by ascending index, deterministically.
        mask[np.argsort(-np.asarray(imp, dtype=float), kind='stable')[:n_keep]] = True
    return mask


def _extract_importance(
    model: Pipeline,
    algorithm: str,
) -> np.ndarray:
    """
    Return a normalized [0, 1] importance array from a fitted pipeline.

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
        # coef_ from LinearSVC surrogate: max absolute coefficient across classes.
        imp = np.abs(clf.coef_).max(axis=0)
    else:
        imp = np.zeros(model.named_steps["scaler"].n_features_in_)

    if imp.max() > 0:
        imp = imp / imp.max()
    return imp


def _estimate_drop_mask(
    X: pd.DataFrame,
    y: np.ndarray,
    algorithm: str,
    stage: str,
    percentile: float,
    seed: int,
    device: str = 'cpu',
    n_jobs: int = -1,
    colsample: float = _COLSAMPLE_BYTREE,
) -> np.ndarray:
    """
    Fit one stage model and return a boolean mask of features to DROP.

    Used during the grid search (one model per evaluation for efficiency).

    Parameters
    ----------
    X : pd.DataFrame
    y : np.ndarray
    algorithm : str
    stage : str
    percentile : float
        Fraction of features to drop, applied by rank; see _rank_keep_mask.
    seed : int
    device : str
        XGBoost compute device ('cpu' or 'cuda').

    Returns
    -------
    np.ndarray of bool, shape (X.shape[1],)
        True where the feature is DROPPED.
    """
    model = _fit_stage_model(algorithm, stage, seed, X, y,
                             colsample=colsample, device=device, n_jobs=n_jobs)
    imp = _extract_importance(model, algorithm)
    return ~_rank_keep_mask(imp, percentile)


def _run_voting_pass(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    algorithm: str,
    stage_key: str,
    best_percentile: float,
    n_models: int,
    base_seed: int,
    seed_offset: int,
    stage_label: str,
    device: str = 'cpu',
    n_jobs: int = -1,
    colsample: float = _COLSAMPLE_BYTREE,
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """
    Run the full n_models consensus voting pass at the best percentile.

    Seed scheme: base_seed + i*200 + seed_offset.
    L1 uses seed_offset=0, L2 uses seed_offset=100, so the two stages
    and their within-stage replicates are all distinct.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : np.ndarray
    algorithm : str
    stage_key : str
        'l1' or 'l2'.
    best_percentile : float
    n_models : int
    base_seed : int
    seed_offset : int
    stage_label : str
        Human-readable label for print output.
    device : str
        XGBoost compute device ('cpu' or 'cuda').

    Returns
    -------
    selected_mask : np.ndarray of bool, shape (X_train.shape[1],)
    importances_list : list of np.ndarray
    """
    logger.info(
        "%s voting pass (percentile=%.4f, %d model(s)):",
        stage_label, best_percentile, n_models,
    )

    vote_counts = np.zeros(X_train.shape[1], dtype=int)
    importances_list: List[np.ndarray] = []
    pool_sizes: List[int] = []
    n_samples = len(X_train)

    for i in range(n_models):
        seed = base_seed + i * _L1_SEED_STEP + seed_offset
        model = _build_stage_model(algorithm, stage_key, seed,
                                   colsample=colsample, device=device,
                                   n_jobs=n_jobs)

        # Bootstrap the rows so that varying the seed produces genuinely
        # different models.  Without this the stage estimators are
        # deterministic given their data for LR and for XGB (which is built
        # here without subsample/colsample and runs with tree_method='exact'),
        # so all n_models fits are bit-identical, every vote is unanimous by
        # construction, and the vote level carries no information at all.
        # Measured before this change: max pairwise importance difference
        # across five seeds was exactly 0.0 for both LR and XGB.
        #
        # Resampling is applied per model rather than per algorithm so all four
        # algorithms get diversity from the same mechanism; relying on each
        # estimator to expose a stochastic hyperparameter left LR and SVM
        # effectively deterministic.
        if n_models > 1:
            try:
                boot_idx = resample(
                    np.arange(n_samples), n_samples=n_samples, replace=True,
                    stratify=y_train, random_state=seed,
                )
            except ValueError:
                # Stratification fails when a class has a single member.
                boot_idx = resample(
                    np.arange(n_samples), n_samples=n_samples, replace=True,
                    random_state=seed,
                )
            X_fit = X_train.iloc[boot_idx]
            y_fit = y_train[boot_idx]
        else:
            X_fit, y_fit = X_train, y_train

        model = _fit_stage_model(algorithm, stage_key, seed, X_fit, y_fit,
                                 colsample=colsample,
                                 device=device, n_jobs=n_jobs)
        imp = _extract_importance(model, algorithm)
        importances_list.append(imp)
        vote_counts += _rank_keep_mask(imp, best_percentile).astype(int)
        pool_sizes.append(int(np.count_nonzero(imp)))

    # Reported for the stage log only. The panel is decided by the graduated
    # intersection across BOTH stages, which starts at unanimity; this is the
    # per-stage view at that same level.
    required_votes = n_models
    selected_mask = vote_counts >= required_votes

    # When the model gave a usable importance to only a handful of features,
    # the percentile never binds and the pool alone decides what can be voted
    # on.  Say so: it is the difference between "the threshold chose these"
    # and "these were all there was to choose from".
    mean_pool = float(np.mean(pool_sizes)) if pool_sizes else 0.0
    requested = int(np.ceil((1.0 - best_percentile) * X_train.shape[1]))
    if mean_pool < requested:
        logger.info(
            "  Percentile saturated: the models gave non-zero importance to "
            "%.0f features on average, fewer than the %d the percentile asked "
            "for, so every feature the models used was eligible to vote.",
            mean_pool, requested,
        )

    logger.info(
        "  Stage selected: %d / %d features (threshold: %d votes)",
        selected_mask.sum(), X_train.shape[1], required_votes,
    )

    return selected_mask, importances_list, vote_counts


def _stage_label(algorithm: str, stage: str) -> str:
    """Return a human-readable stage label for print output."""
    labels = {
        "LR": {
            "l1": "Stage 1: L1 regularization (Lasso-style)",
            "l2": "Stage 2: L2 regularization (Ridge-style)",
        },
        "XGB": {
            "l1": "Stage 1: L1 regularization (alpha-dominant)",
            "l2": "Stage 2: L2 regularization (lambda-dominant)",
        },
        "RF": {
            "l1": "Stage 1: Feature importance (sparse tree settings)",
            "l2": "Stage 2: Feature importance (conservative tree settings)",
        },
        "SVM": {
            "l1": "Stage 1: High regularization (low C)",
            "l2": "Stage 2: Low regularization (high C)",
        },
    }
    return labels.get(algorithm, {}).get(stage, f"{algorithm} {stage}")


# ---------------------------------------------------------------------------
# Private helpers: evaluation block
# ---------------------------------------------------------------------------

def _evaluate_step2(
    X_reg: pd.DataFrame,
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
    Run the Step 2 consensus evaluation and comparison against Step 0.

    Builds n_models consensus pipelines with seeds base_seed + i*100,
    evaluates each on X_reg with bootstrap resampling, prints the
    comparison against the Step 0 reference.

    Parameters
    ----------
    X_reg : pd.DataFrame
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
    logger.info("STEP 2 EVALUATION")

    if adequacy is not None:
        _print_adequacy_warning(
            step_name="Step 2: Regularization",
            n_features_current=X_reg.shape[1],
            adequacy=adequacy,
        )

    step2_models: Dict[str, Any] = {}

    if n_models == 1:
        step2_models[f"Reg-{algorithm}"] = build_consensus_pipeline(
            algorithm=algorithm,
            seed=base_seed,
            tuned_pipelines=tuned_pipelines,
        )
    else:
        for i in range(n_models):
            seed = base_seed + i * _EVAL_SEED_STEP
            step2_models[f"Reg-{algorithm}-{i+1}"] = build_consensus_pipeline(
                algorithm=algorithm,
                seed=seed,
                tuned_pipelines=tuned_pipelines,
            )

    step2_evaluation = evaluate_models_collection(
        pipelines=step2_models,
        X=X_reg,
        y=y_train,
        cv=cv,
        n_classes=n_classes,
        classes=np.arange(n_classes),
        config=config,
        step_name=f"Step 2: Regularization - All {algorithm} Models",
        use_bootstrap=True,
    )

    if n_models == 1:
        consensus_result = list(step2_evaluation["results"].values())[0]
    else:
        best_name = max(
            step2_evaluation["results"],
            key=lambda k: step2_evaluation["results"][k]["mean_auc"],
        )
        consensus_result = step2_evaluation["results"][best_name]

    logger.info("Step 2 Individual Model Performance:")
    for name, result in step2_evaluation["results"].items():
        logger.info(
            "  %s: AUC = %.4f +/- %.4f",
            name, result['mean_auc'], result['std_auc'],
        )
    logger.info(
        "  Feature count: %d (reduced from %d)",
        X_reg.shape[1], X_train_original.shape[1],
    )

    logger.info("STEP 2: CONSENSUS vs REFERENCE COMPARISON")

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
        _plot_step2(
            step2_evaluation=step2_evaluation,
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
            "output_features": X_reg.shape[1],
            "features_removed": X_train_original.shape[1] - X_reg.shape[1],
            # How strong a claim this panel is: votes_required, agreement
            # and agreement_label belong in the saved record, not just the log.
            **(consensus_summary or {}),
        }
        for name, result in step2_evaluation["results"].items():
            clean_name = name.lower().replace("-", "_")
            step_summary[f"{clean_name}_auc"] = result["mean_auc"]
            step_summary[f"{clean_name}_std"] = result["std_auc"]

        output_dir.mkdir(parents=True, exist_ok=True)
        save_step_summary("step2", step_summary, output_dir)

    logger.info("Step 2 evaluation complete.")
    return consensus_result


def _plot_step2(
    step2_evaluation: Dict[str, Any],
    ref_result_step0: Optional[Dict[str, Any]],
    y_train: np.ndarray,
    n_classes: int,
    algorithm: str,
    config: SelectOmicsConfig,
    output_dir: Path,
) -> None:
    """
    Generate Step 2 diagnostic plots: AUC box plots and ROC curves.

    Parameters
    ----------
    step2_evaluation : dict
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
        step2_evaluation["results"],
        key=lambda k: step2_evaluation["results"][k]["mean_auc"],
    )
    comparison[f"Consensus-{algorithm}"] = step2_evaluation["results"][best_name]

    fold_aucs = {name: result["fold_aucs"] for name, result in comparison.items()}

    plot_auc_boxplots(
        step_name="Step 2: Regularization Consensus vs Reference",
        aucs_dict=fold_aucs,
        config=config,
        save_path=output_dir / "step2_consensus_vs_reference_boxplots",
    )

    plot_roc_curves(
        models_dict=comparison,
        y_true=y_train,
        n_classes=n_classes,
        classes=np.arange(n_classes),
        config=config,
        title="Step 2: Regularization Consensus vs Reference ROC",
        save_path=output_dir / "step2_consensus_vs_reference_roc",
    )
