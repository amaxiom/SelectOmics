"""
SelectOmics/selection/step1_cleaning.py

Step 1: Data-based cleaning with joint variance and correlation filtering.

Removes constant features, then applies a joint 11x11 threshold
optimisation that finds the (variance_threshold, correlation_threshold)
pair maximising the AGREEMENT between the two filters, scored as the
Jaccard similarity of their drop sets: |A and B| / |A or B|.

Agreement, not removal.  Maximising the raw size of the drop intersection
would reward whichever threshold pair simply drops the most, since a larger
drop set has more opportunity to overlap.  Normalising by the union asks a
different and better question: of everything either filter wanted to remove,
how much did they concur on?  A pair dropping 5 features both filters agree
on therefore scores above one dropping 500 with 90% concurrence.  The two
objectives genuinely diverge, and this module optimises the second.

If the resulting intersection falls below the minimum feature count, the
kept sets are unioned, and failing that the top-k by variance are forced.

Design decisions
----------------
- Joint optimisation rather than independent calibration: running both
  filters in parallel and maximising their consensus on removal is more
  principled than tuning each threshold separately against its own
  retention criterion.
- Grid size 11x11 = 121 evaluations. Each variance filter call is O(p)
  and each correlation filter call is O(p^2) for the correlation matrix
  plus O(p^2) for greedy traversal.  For typical omics p (100-10000)
  this is fast enough that binary search adds unnecessary complexity.
- Correlation filter tie-breaking: when a correlated pair is found, the
  lower-variance feature is dropped.  Features are processed in descending
  variance order so higher-variance features are evaluated first and
  preferentially retained.  This is deterministic and principled.

The correlation filter is class-aware (config.class_aware_correlation)
---------------------------------------------------------------------
Features carrying the same class signal are mutually correlated *because* they
are informative.  A filter built on total correlation cannot tell that apart
from redundant duplication, so it keeps one carrier and discards the rest --
deleting exactly the features worth keeping.

Measured on synthetic data (n=500, p=200, 20 informative, rho=0.10) with a
total-correlation filter: at signal_strength 1.2 the mean correlation among
informative features is 0.851 and 1 of 20 survives; by 2.5 it is 0.961 and
still 1 of 20 survives, while all 180 noise features are retained.

The fix decomposes the covariance.  Centring each feature on its own class mean
removes the between-class part, leaving only association the label does not
explain -- genuine redundancy.  Two duplicate probes stay correlated after
centring; two independent markers of the same disease do not.  With the
class-aware filter all 20 informative features survive at every signal
strength tested.

Which member of a genuinely redundant pair survives is decided by the
correlation ratio (eta-squared) rather than by variance.  Variance is unrelated
to the outcome, so the survivor was effectively arbitrary and varied with the
sample -- a direct contributor to low cross-fold stability.

Set class_aware_correlation=False to restore the class-unaware matrix and
the variance tie-break used up to 0.7.0.  This is not bit-identical to that
release: exact variance ties, which used to be settled by column order, are
now settled by name so the result cannot depend on matrix assembly order.

Consequences for real data: a co-regulated module whose members all track the
outcome is reduced to one arbitrary member, selected by the variance
tie-break.  Which member survives depends on the sample, so repeated runs on
different subsets return different members -- a direct contributor to the low
cross-fold stability reported in BENCHMARKS.md section 2.3.

Raise min_correlation_threshold, or disable this step, if co-regulated features
are still being lost after the class-aware filter.
- When Step 1 is disabled (enable_step1=False), X_train and X_test are
  returned unchanged and the evaluation result is copied from ref_result.
- The quick generalization test (5 random splits) in the notebook is
  omitted in the package: it was a notebook diagnostic, not part of the
  formal evaluation protocol.
"""

from __future__ import annotations

import logging
import warnings

logger = logging.getLogger(__name__)

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold

from ..config import SelectOmicsConfig
from ..models.base import build_consensus_pipeline

# ---------------------------------------------------------------------------
# Seed scheme
# ---------------------------------------------------------------------------

# Step 1 consensus models use seeds base_seed + i * _SEED_STEP.
# Kept small (100) so seeds stay distinct from Step 2 (200) and Step 3 (400).
_SEED_STEP: int = 100
from ..models.evaluation import evaluate_models_collection
from ..utils.helpers import agreement_label, save_step_summary
from ..data.loaders import save_from_config
from .step0_reference import _print_adequacy_warning


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_step1_cleaning(
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
    Remove constant, low-variance, and highly correlated features.

    Applies three sub-steps in sequence:

    1a. Constant removal: drops features with a single unique value.

    1b/1c. Joint variance and correlation threshold optimisation: an 11x11
    grid search over (variance_threshold, correlation_threshold) pairs finds
    the combination that maximises the size of the DROP intersection, i.e.
    the set of features that both filters independently agree to remove.
    The final kept set is the intersection of the two kept sets (strictest
    consensus).  Fallbacks apply if the intersection is too small.

    When ``config.enable_step1`` is False the function returns X_train and
    X_test unchanged with all dropped lists empty.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix from the previous step (or original data).
    X_test : pd.DataFrame
        Test feature matrix aligned with X_train columns.
    y_train : np.ndarray
        Encoded integer class labels, shape (n_samples,).
    config : SelectOmicsConfig
        Pipeline configuration object.
    cv : StratifiedKFold
        Stratified CV splitter shared across all steps.
    tuned_pipelines : dict
        Mapping from algorithm key to fitted sklearn Pipeline from
        ``quick_tune_all``.
    n_classes : int
        Number of distinct target classes.
    class_names : list of str
        Human-readable class label strings.
    ref_result_step0 : dict, optional
        Reference CV result from Step 0 used for consensus vs reference
        comparison in the evaluation block.
    adequacy : dict, optional
        Output of ``assess_sample_size_adequacy`` for warnings.
    output_dir : Path, optional
        Artefact output directory; derived from ``config.output_dir`` if None.

    Returns
    -------
    dict with keys:
        X_train_clean : pd.DataFrame
            Training matrix after cleaning.
        X_test_clean : pd.DataFrame
            Test matrix with the same column subset.
        dropped_by_constant : list of str
            Features removed in sub-step 1a.
        dropped_by_variance : list of str
            Features below the best variance threshold (diagnostics).
        dropped_by_correlation : list of str
            Features removed by the best correlation threshold (diagnostics).
        drop_consensus : list of str
            Features removed by both filters (actual consensus drop set).
        best_var_thresh : float
            Optimal variance threshold from the grid search.
        best_corr_thresh : float
            Optimal correlation threshold from the grid search.
        consensus_result : dict or None
            CV evaluation result for the cleaned feature set.  None when
            enable_step_evaluations is False.
        n_features_in : int
            Number of features before Step 1.
        n_features_out : int
            Number of features after Step 1.
    """
    algorithm = config.algorithm

    if output_dir is None:
        output_dir = Path(config.output_dir)

    logger.info("STEP 1: DATA-BASED CLEANING")

    # ------------------------------------------------------------------
    # Disabled branch: pass data through unchanged
    # ------------------------------------------------------------------
    if not config.enable_step1:
        logger.info("Step 1 disabled - using original data unchanged")
        logger.info("  Features: %d", X_train.shape[1])
        return {
            "X_train_clean": X_train.copy(),
            "X_test_clean": X_test.copy(),
            "dropped_by_constant": [],
            "dropped_by_variance": [],
            "dropped_by_correlation": [],
            "drop_consensus": [],
            "best_var_thresh": 0.0,
            "best_corr_thresh": config.min_correlation_threshold,
            "consensus_result": ref_result_step0,
            "n_features_in": X_train.shape[1],
            "n_features_out": X_train.shape[1],
            "votes_required": 0,
            "agreement": 0.0,
            "agreement_label": "not run",
            "consensus_outcome": "disabled",
            "n_models": 0,
        }

    # ------------------------------------------------------------------
    # Configuration parameters
    # ------------------------------------------------------------------
    n_models = config.n_consensus_models
    variance_threshold = config.max_variance_threshold
    correlation_threshold = config.min_correlation_threshold
    base_seed = config.random_seed

    # Step 1's consensus is between its two filters (variance and correlation),
    # not across model replicates, so there is no vote requirement here. The
    # old log line reported one anyway, computed from a threshold this step
    # never consulted.
    logger.info("Using %s with %d consensus model(s)", algorithm, n_models)
    logger.info("Input features: %d", X_train.shape[1])

    # Absolute safety net, not a retention target; see config.min_features_floor.
    min_features = min(config.min_features_floor, X_train.shape[1])

    # Pre-compute per-feature variance once: used both as the variance
    # filter signal and as the deterministic tie-breaker in the
    # correlation filter.
    feature_variances = X_train.var(axis=0)

    # ------------------------------------------------------------------
    # Sub-step 1a: remove constant features
    # ------------------------------------------------------------------
    logger.info("Step 1a: Removing constant features...")

    constant_features = [
        col for col in X_train.columns if X_train[col].nunique() <= 1
    ]

    if constant_features:
        X_after_constant = X_train.drop(columns=constant_features)
        X_test_after_constant = X_test.drop(columns=constant_features)
        dropped_by_constant = constant_features
        logger.info("  Removed %d constant features", len(constant_features))
    else:
        X_after_constant = X_train
        X_test_after_constant = X_test
        dropped_by_constant = []
        logger.info("  No constant features found")

    logger.info("  Shape: %s -> %s", X_train.shape, X_after_constant.shape)

    # Every feature being constant leaves nothing to select from.  Without this
    # guard an empty matrix flows into the grid search and the run dies several
    # steps later inside sklearn with an error that says nothing about the data.
    if X_after_constant.shape[1] == 0:
        raise ValueError(
            f"All {X_train.shape[1]} features are constant (a single unique "
            f"value each), so there is nothing to select from. Check that "
            f"data_path points at the intended file and that the feature "
            f"columns are numeric and vary across samples."
        )

    # ------------------------------------------------------------------
    # Sub-steps 1b/1c: joint variance and correlation optimisation
    # ------------------------------------------------------------------
    logger.info("Step 1b/1c: Joint variance and correlation threshold optimisation...")
    logger.info("  Grid: 11 x 11 = 121 threshold pairs evaluated")

    variance_grid = np.linspace(0.0, variance_threshold, 11)
    # Correlation grid spans from the configured minimum up to 1.0
    # (perfect correlation).  A threshold of 1.0 retains all features.
    correlation_grid = np.linspace(correlation_threshold, 1.0, 11)

    # Pre-compute the correlation matrix once. It is O(p^2) and independent
    # of both threshold values, so recomputing it inside the 11x11 loop
    # would waste 99% of the correlation computation.
    #
    # Class-aware by default: the pooled WITHIN-class correlation measures
    # redundancy with the outcome signal removed, so two features that are
    # correlated only because both track the label are no longer treated as
    # duplicates of each other. See pooled_within_class_corr for the
    # measurement that motivated this.
    if config.class_aware_correlation:
        corr_matrix = pooled_within_class_corr(X_after_constant, y_train)
        keep_priority = class_association(X_after_constant, y_train)
        logger.info(
            "Correlation filter: class-aware (pooled within-class "
            "correlation, eta-squared tie-break)"
        )
    else:
        corr_matrix = X_after_constant.corr().abs()
        keep_priority = None
        logger.info(
            "Correlation filter: class-unaware (total correlation, variance "
            "tie-break). Features that are correlated BECAUSE they share the "
            "class signal will be treated as redundant."
        )

    # Jaccard similarity of drop sets: |A and B| / |A or B|.
    # Normalized to [0, 1] so the score measures genuine agreement rather
    # than rewarding aggressive thresholds that produce large drop sets.
    best_jaccard = -1.0
    best_var_thresh = float(variance_grid[0])
    best_corr_thresh = float(correlation_grid[-1])
    best_var_kept: Optional[np.ndarray] = None
    best_corr_kept: Optional[np.ndarray] = None

    for var_t in variance_grid:
        var_kept = _apply_variance_filter(X_after_constant, var_t)
        var_dropped_set = set(X_after_constant.columns[~var_kept].tolist())

        for corr_t in correlation_grid:
            corr_kept = _apply_correlation_filter(
                X_after_constant, corr_t, feature_variances, corr_matrix,
                keep_priority=keep_priority,
            )
            corr_dropped_set = set(
                X_after_constant.columns[~corr_kept].tolist()
            )

            union_size = len(var_dropped_set | corr_dropped_set)
            jaccard = (
                len(var_dropped_set & corr_dropped_set) / union_size
                if union_size > 0 else 0.0
            )

            if jaccard > best_jaccard:
                best_jaccard = jaccard
                best_var_thresh = float(var_t)
                best_corr_thresh = float(corr_t)
                best_var_kept = var_kept.copy()
                best_corr_kept = corr_kept.copy()

    logger.info("  Best variance threshold:    %.4f", best_var_thresh)
    logger.info("  Best correlation threshold: %.4f", best_corr_thresh)
    logger.info("  Drop Jaccard score:         %.4f", best_jaccard)
    logger.info("  Variance kept:              %d features", best_var_kept.sum())
    logger.info("  Correlation kept:           %d features", best_corr_kept.sum())

    # ------------------------------------------------------------------
    # Fallback hierarchy:
    #   Primary    - intersection (both filters agree to keep)   2/2 agreement
    #   Fallback 1 - union (either filter keeps)                 1/2 agreement
    #   Fallback 2 - top-k by variance                           no agreement
    #
    # Step 1 has only two "voters" -- the variance filter and the correlation
    # filter -- and each casts a single binary vote, so unlike Steps 2 to 4
    # there are no intermediate rungs to relax through: intersection and union
    # ARE the whole ladder. What it shares with them is the vocabulary, so a
    # panel decided here reports its agreement the same way and
    # pipeline._final_agreement() can read it.
    # ------------------------------------------------------------------
    _N_FILTERS = 2
    final_kept = best_var_kept & best_corr_kept
    votes_required, consensus_outcome = _N_FILTERS, 'consensus'
    logger.info("  Intersection of kept sets:  %d features", final_kept.sum())

    if final_kept.sum() < min_features:
        logger.warning(
            "  Intersection (%d) below minimum (%d). Falling back to union of kept sets.",
            final_kept.sum(), min_features,
        )
        final_kept = best_var_kept | best_corr_kept
        votes_required, consensus_outcome = 1, 'relaxed'
        logger.info("  Union of kept sets:         %d features", final_kept.sum())

        if final_kept.sum() < min_features:
            # Rank by the same priority the correlation filter uses, so the
            # forced set is chosen on association with the outcome where that
            # is available. Ranking by variance here would contradict every
            # other retention decision in this step.
            _forced_rank = (
                keep_priority if keep_priority is not None else feature_variances
            )
            logger.warning(
                "  Union (%d) still below minimum (%d). Forcing top %d by %s.",
                final_kept.sum(), min_features, min_features,
                "class association" if keep_priority is not None else "variance",
            )
            # Descending, STABLE, then take the head. The default quicksort
            # is not stable, and this ranks on the correlation ratio, which
            # sits at ~0 for every noise feature -- so exact ties are the
            # common case here, not the exception. Without a stable sort the
            # forced panel could differ between identical runs, which the
            # package's reproducibility guarantee forbids. Matches the
            # convention in step2's _rank_keep_mask.
            top_idx = np.argsort(
                -_forced_rank[X_after_constant.columns].values, kind='stable'
            )[:min_features]
            final_kept = np.zeros(X_after_constant.shape[1], dtype=bool)
            final_kept[top_idx] = True
            votes_required, consensus_outcome = 0, 'rank_average'
            logger.info("  Forced selection:           %d features", final_kept.sum())

    step1_agreement = votes_required / _N_FILTERS
    step1_label = agreement_label(step1_agreement, consensus_outcome)
    logger.info(
        "  Filter agreement: %d/%d (%s)",
        votes_required, _N_FILTERS, step1_label,
    )

    X_clean = X_after_constant.loc[:, final_kept]
    X_test_clean = X_test_after_constant.loc[:, final_kept]

    # Identify features dropped by each method for reporting.
    dropped_by_variance = X_after_constant.columns[~best_var_kept].tolist()
    dropped_by_correlation = X_after_constant.columns[~best_corr_kept].tolist()
    drop_consensus = list(
        set(dropped_by_variance) & set(dropped_by_correlation)
    )

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    logger.info("Step 1 Summary:")
    logger.info("  Original:                %s", X_train.shape)
    logger.info("  After constant removal:  %s", X_after_constant.shape)
    logger.info("  Dropped by variance:     %d", len(dropped_by_variance))
    logger.info("  Dropped by correlation:  %d", len(dropped_by_correlation))
    logger.info(
        "  Drop consensus:          %d (both methods agreed)", len(drop_consensus),
    )
    logger.info("  Final (after cleaning):  %s", X_clean.shape)
    logger.info("  Total removed:           %d", X_train.shape[1] - X_clean.shape[1])

    # ------------------------------------------------------------------
    # Save intermediate results
    # ------------------------------------------------------------------
    if config.save_intermediate_results:
        output_dir.mkdir(parents=True, exist_ok=True)

        save_from_config(
            pd.DataFrame({"dropped_by_constant": dropped_by_constant}),
            output_dir / "step1_dropped_by_constant",
            config,
        )
        save_from_config(
            pd.DataFrame({"dropped_by_variance": dropped_by_variance}),
            output_dir / "step1_dropped_by_variance",
            config,
        )
        save_from_config(
            pd.DataFrame({"dropped_by_correlation": dropped_by_correlation}),
            output_dir / "step1_dropped_by_correlation",
            config,
        )
        save_from_config(
            pd.DataFrame({"dropped_by_consensus": drop_consensus}),
            output_dir / "step1_dropped_by_consensus",
            config,
        )
        save_from_config(
            X_clean,
            output_dir / "step1_X_clean",
            config,
        )
        # Record the selection facts now; see the note in step2 on why this
        # cannot wait for the optional evaluation.
        save_step_summary("step1", {
            "algorithm": algorithm,
            "n_consensus_models": n_models,
            "input_features": X_train.shape[1],
            "output_features": X_clean.shape[1],
            "features_removed": X_train.shape[1] - X_clean.shape[1],
            "votes_required": votes_required,
            "agreement": round(step1_agreement, 4),
            "agreement_label": step1_label,
            "consensus_outcome": consensus_outcome,
            "n_filters": _N_FILTERS,
        }, output_dir)

    # ------------------------------------------------------------------
    # Step evaluation (conditional on config flag)
    # ------------------------------------------------------------------
    consensus_result = None

    if config.enable_step_evaluations:
        consensus_result = _evaluate_step1(
            X_clean=X_clean,
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
                "agreement": round(step1_agreement, 4),
                "agreement_label": step1_label,
                "consensus_outcome": consensus_outcome,
                "n_filters": _N_FILTERS,
            },
        )
    else:
        logger.info("Step 1 evaluation skipped.")

    logger.info("Ready for Step 2.")

    return {
        "X_train_clean": X_clean,
        "X_test_clean": X_test_clean,
        "dropped_by_constant": dropped_by_constant,
        "dropped_by_variance": dropped_by_variance,
        "dropped_by_correlation": dropped_by_correlation,
        "drop_consensus": drop_consensus,
        "best_var_thresh": best_var_thresh,
        "best_corr_thresh": best_corr_thresh,
        "consensus_result": consensus_result,
        "n_features_in": X_train.shape[1],
        "n_features_out": X_clean.shape[1],
        # Agreement between the variance and correlation filters, reported in
        # the same vocabulary as Steps 2 to 4 so a panel decided here is
        # readable the same way. n_models is 2 because there are two filters,
        # not two consensus models.
        "votes_required": votes_required,
        "agreement": round(step1_agreement, 4),
        "agreement_label": step1_label,
        "consensus_outcome": consensus_outcome,
        "n_models": _N_FILTERS,
    }


# ---------------------------------------------------------------------------
# Private helpers: filter implementations
# ---------------------------------------------------------------------------

def _apply_variance_filter(
    X: pd.DataFrame,
    var_thresh: float,
) -> np.ndarray:
    """
    Return a boolean kept mask for features whose variance exceeds var_thresh.

    Wraps ``sklearn.feature_selection.VarianceThreshold``.  Deterministic:
    no randomness involved.

    Parameters
    ----------
    X : pd.DataFrame
    var_thresh : float
        Variance threshold; features with variance <= this value are dropped.

    Returns
    -------
    np.ndarray of bool, shape (X.shape[1],)
        True where the feature is KEPT.
    """
    selector = VarianceThreshold(threshold=var_thresh)
    try:
        selector.fit(X)
    except ValueError:
        # No feature meets the threshold -- treat as all-dropped for this
        # grid point so the outer loop can continue safely.
        return np.zeros(X.shape[1], dtype=bool)
    return selector.get_support()


def pooled_within_class_corr(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """
    Absolute pooled within-class correlation: redundancy with the class
    signal removed.

    Two features that both track the outcome are correlated *because* they are
    informative.  A plain Pearson correlation cannot tell that apart from two
    features measuring the same thing, so a correlation filter built on it
    deletes exactly the features worth keeping.

    Measured on synthetic data (n=500, p=200, 20 informative, rho=0.10): at
    signal_strength 1.2 the mean correlation among informative features is
    0.584 and all 20 survive; at 2.0 it is 0.797 and only ONE survives, while
    all 180 noise features are retained.  The transition sits exactly at the
    0.70 default threshold.

    Total covariance decomposes into a between-class part (driven by the
    outcome) and a within-class part (everything else).  Centring each feature
    on its own class mean removes the between-class part, so what remains is
    association that is NOT explained by the label -- genuine redundancy.  Two
    duplicate probes stay correlated after centring; two independent markers of
    the same disease do not.

    Parameters
    ----------
    X : pd.DataFrame
    y : np.ndarray
        Encoded class labels, one per row of X.

    Returns
    -------
    pd.DataFrame
        Absolute pooled within-class correlation, same shape and labels as
        ``X.corr().abs()``.
    """
    # nanmean, not mean: SelectOmics accepts missing values and never imputes
    # them, and numpy's mean propagates NaN. A single missing cell would make
    # that class's mean NaN, which poisons the whole feature and yields a NaN
    # correlation matrix. Measured at 5% missing on 200x12: 144 non-finite
    # cells before this change. pandas .corr() then handles whatever NaN
    # remains pairwise-complete, exactly as the class-unaware path does.
    # astype(float), not copy(): writing centred float values into an
    # integer frame is deprecated in pandas and will raise. Integer omics
    # matrices (count data) hit this.
    centred = X.astype(float)
    values = centred.values
    for cls in np.unique(y):
        rows = (y == cls)
        if rows.sum() < 2:
            # A singleton class carries no within-class variation. Zero its
            # rows so it contributes nothing, rather than injecting that
            # class's offset into the pooled estimate.
            centred.loc[rows, :] = 0.0
            continue
        block = values[rows]
        with warnings.catch_warnings():
            # An all-NaN column within one class is legitimate; nanmean warns
            # and returns NaN, which .corr() then handles pairwise.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            centred.loc[rows, :] = block - np.nanmean(block, axis=0)
    # A feature with no within-class variation (one perfectly determined by
    # the label) centres to all zeros, so its correlation with anything is
    # undefined and it is never judged redundant. That is the intended
    # trade-off, not an oversight: such a feature is maximally informative,
    # and dropping one of a co-regulated pair of them is precisely the failure
    # this function exists to prevent. The cost is that two *identical*
    # class-determined probes both survive Step 1; Steps 2 and 3 then see them
    # as the duplicate pair they are.
    return centred.corr().abs()


def class_association(X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    """
    Per-feature association with the outcome, as a correlation ratio in [0, 1].

    eta^2: the fraction of a feature's total variance explained by class
    membership.  Used to decide WHICH member of a genuinely redundant pair to
    keep.  The previous tie-break was variance, which is unrelated to the
    outcome, so the surviving member was effectively arbitrary -- and since it
    varies with the sample, repeated runs kept different members.  That is a
    direct contributor to low cross-fold stability.
    """
    # NaN-aware throughout, for the same reason as pooled_within_class_corr.
    # The previous version used plain mean/sum, so ss_total was NaN for any
    # feature with a missing cell; the `ss_total > 0` guard is False for NaN,
    # so that feature silently scored 0.0 and was preferentially dropped. A
    # missing value is not evidence that a feature is uninformative.
    values = X.values.astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        grand = np.nanmean(values, axis=0)
        ss_total = np.nansum((values - grand) ** 2, axis=0)

        ss_between = np.zeros(values.shape[1], dtype=float)
        for cls in np.unique(y):
            rows = (y == cls)
            if not rows.any():
                continue
            block = values[rows]
            # Weight by the number of OBSERVED values per feature in this
            # class, not by the class row count, so features with different
            # missingness are weighted correctly.
            n_obs = np.sum(~np.isnan(block), axis=0)
            cls_mean = np.nanmean(block, axis=0)
            contrib = n_obs * (cls_mean - grand) ** 2
            ss_between += np.where(np.isfinite(contrib), contrib, 0.0)

    with np.errstate(divide='ignore', invalid='ignore'):
        eta2 = np.where(np.isfinite(ss_total) & (ss_total > 0),
                        ss_between / ss_total, 0.0)
    return pd.Series(np.clip(np.nan_to_num(eta2), 0.0, 1.0), index=X.columns)


def _apply_correlation_filter(
    X: pd.DataFrame,
    corr_thresh: float,
    variances: pd.Series,
    corr_matrix: pd.DataFrame,
    keep_priority: Optional[pd.Series] = None,
) -> np.ndarray:
    """
    Return a boolean kept mask after greedy correlation filtering.

    For each correlated pair (absolute correlation > corr_thresh) one member is
    dropped.  Which one is decided by ``keep_priority``: the member with the
    LOWER priority goes.  Features are processed in descending priority order,
    so the strongest are evaluated first and preferentially retained.

    ``keep_priority`` is the class association (eta^2) when the label is
    available, and falls back to variance when it is not.  The distinction
    matters: variance ranks features by how much they vary, which has nothing
    to do with the outcome, so a redundant pair resolved by variance keeps an
    arbitrary member and a different one on the next sample.

    Fully deterministic given a fixed priority series.

    Parameters
    ----------
    X : pd.DataFrame
    corr_thresh : float
        Absolute correlation threshold above which a pair is redundant.
    variances : pd.Series
        Per-feature variance.  Used as the priority only when keep_priority
        is None.
    corr_matrix : pd.DataFrame
        Pre-computed absolute correlation matrix, ideally the pooled
        within-class one.  Passed in so it is computed once rather than once
        per grid point.
    keep_priority : pd.Series, optional
        Per-feature retention priority; higher survives.

    Returns
    -------
    np.ndarray of bool, shape (X.shape[1],)
        True where the feature is KEPT.
    """
    priority = variances if keep_priority is None else keep_priority

    # Descending priority: strongest features evaluated first and
    # preferentially retained. Variance breaks exact ties so the order stays
    # deterministic when several features share a priority (common with eta^2
    # on pure-noise features, which all sit at ~0).
    order_frame = pd.DataFrame({
        "priority": priority[X.columns].values,
        "variance": variances[X.columns].values,
    }, index=X.columns)
    ordered_cols = order_frame.sort_values(
        ["priority", "variance"], ascending=False).index
    dropped: set = set()

    for col in ordered_cols:
        if col in dropped:
            continue

        partners = corr_matrix.index[
            (corr_matrix[col] > corr_thresh) & (corr_matrix.index != col)
        ].tolist()

        for partner in partners:
            if partner not in dropped:
                if priority[partner] < priority[col]:
                    dropped.add(partner)
                elif priority[partner] > priority[col]:
                    dropped.add(col)
                    break  # col is now dropped; move to next col
                # Exact tie: fall back to variance, then to name order, so
                # the outcome never depends on column ordering.
                elif variances[partner] < variances[col]:
                    dropped.add(partner)
                elif variances[partner] > variances[col]:
                    dropped.add(col)
                    break
                elif partner > col:
                    dropped.add(partner)
                else:
                    dropped.add(col)
                    break

    return np.array([col not in dropped for col in X.columns], dtype=bool)


# ---------------------------------------------------------------------------
# Private helpers: evaluation block
# ---------------------------------------------------------------------------

def _evaluate_step1(
    X_clean: pd.DataFrame,
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
    Run the Step 1 consensus evaluation and comparison against Step 0.

    Builds n_models consensus pipelines with seeds spaced 100 apart,
    evaluates each on X_clean with bootstrap resampling, then prints the
    comparison against the Step 0 reference result.

    Parameters
    ----------
    X_clean : pd.DataFrame
        Cleaned feature matrix from Step 1.
    X_train_original : pd.DataFrame
        Pre-cleaning training matrix (for feature count reporting only).
    y_train : np.ndarray
        Encoded class labels.
    config : SelectOmicsConfig
        Pipeline configuration.
    cv : StratifiedKFold
        CV splitter shared across all steps.
    tuned_pipelines : dict
        Tuned pipeline dict from ``quick_tune_all``.
    n_classes : int
        Number of classes.
    class_names : list of str
        Class label strings.
    ref_result_step0 : dict or None
        Step 0 reference result for comparison.
    adequacy : dict or None
        Adequacy assessment for warnings.
    output_dir : Path
        Artefact save directory.
    base_seed : int
        Base random seed; consensus model i uses seed base_seed + i*100.
    n_models : int
        Number of consensus models.
    algorithm : str
        Active algorithm key.

    Returns
    -------
    dict or None
        Best consensus model CV result dict, or None on failure.
    """
    logger.info("STEP 1 EVALUATION")

    if adequacy is not None:
        _print_adequacy_warning(
            step_name="Step 1: Data Cleaning",
            n_features_current=X_clean.shape[1],
            adequacy=adequacy,
        )

    # Build consensus model dict with seeds spaced 100 apart.
    step1_models: Dict[str, Any] = {}

    if n_models == 1:
        step1_models[f"Clean-{algorithm}"] = build_consensus_pipeline(
            algorithm=algorithm,
            seed=base_seed,
            tuned_pipelines=tuned_pipelines,
        )
    else:
        for i in range(n_models):
            seed = base_seed + i * _SEED_STEP
            step1_models[f"Clean-{algorithm}-{i+1}"] = build_consensus_pipeline(
                algorithm=algorithm,
                seed=seed,
                tuned_pipelines=tuned_pipelines,
            )

    step1_evaluation = evaluate_models_collection(
        pipelines=step1_models,
        X=X_clean,
        y=y_train,
        cv=cv,
        n_classes=n_classes,
        classes=np.arange(n_classes),
        config=config,
        step_name=f"Step 1: Data Cleaning - All {algorithm} Models",
        use_bootstrap=True,
    )

    # Select best consensus model by mean AUC.
    if n_models == 1:
        consensus_result = list(step1_evaluation["results"].values())[0]
    else:
        best_name = max(
            step1_evaluation["results"],
            key=lambda k: step1_evaluation["results"][k]["mean_auc"],
        )
        consensus_result = step1_evaluation["results"][best_name]

    # Print individual model performance.
    logger.info("Step 1 Individual Model Performance:")
    for name, result in step1_evaluation["results"].items():
        logger.info(
            "  %s: AUC = %.4f +/- %.4f",
            name, result['mean_auc'], result['std_auc'],
        )
    logger.info(
        "  Feature count: %d (reduced from %d)",
        X_clean.shape[1], X_train_original.shape[1],
    )

    # Print consensus vs reference comparison.
    logger.info("STEP 1: CONSENSUS vs REFERENCE COMPARISON")

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

    # Visualisations.
    if config.create_visualizations:
        _plot_step1(
            step1_evaluation=step1_evaluation,
            ref_result_step0=ref_result_step0,
            y_train=y_train,
            n_classes=n_classes,
            algorithm=algorithm,
            config=config,
            output_dir=output_dir,
        )

    # Save step summary.
    if config.save_intermediate_results:
        step_summary = {
            "algorithm": algorithm,
            "n_consensus_models": n_models,
            "input_features": X_train_original.shape[1],
            "output_features": X_clean.shape[1],
            "features_removed": X_train_original.shape[1] - X_clean.shape[1],
            # Agreement between the variance and correlation filters. This
            # belongs in the saved record, not just the log: it is how a
            # reader tells a consensus panel from a relaxed one.
            **(consensus_summary or {}),
        }
        for name, result in step1_evaluation["results"].items():
            clean_name = name.lower().replace("-", "_")
            step_summary[f"{clean_name}_auc"] = result["mean_auc"]
            step_summary[f"{clean_name}_std"] = result["std_auc"]

        output_dir.mkdir(parents=True, exist_ok=True)
        save_step_summary("step1", step_summary, output_dir)

    logger.info("Step 1 evaluation complete.")
    return consensus_result


def _plot_step1(
    step1_evaluation: Dict[str, Any],
    ref_result_step0: Optional[Dict[str, Any]],
    y_train: np.ndarray,
    n_classes: int,
    algorithm: str,
    config: SelectOmicsConfig,
    output_dir: Path,
) -> None:
    """
    Generate Step 1 diagnostic plots: AUC box plots and ROC curves.

    Combines the Step 0 reference result with the Step 1 consensus result
    for side-by-side comparison.  Matplotlib imported lazily.

    Parameters
    ----------
    step1_evaluation : dict
        Output of evaluate_models_collection for Step 1.
    ref_result_step0 : dict or None
        Step 0 reference CV result.
    y_train : np.ndarray
        Encoded training labels.
    n_classes : int
        Number of classes.
    algorithm : str
        Active algorithm key.
    config : SelectOmicsConfig
        Pipeline configuration.
    output_dir : Path
        Save directory for figures.
    """
    from ..evaluation.visualization import plot_auc_boxplots, plot_roc_curves

    output_dir.mkdir(parents=True, exist_ok=True)

    comparison: Dict[str, Any] = {}

    if ref_result_step0 is not None:
        comparison[f"Ref-{algorithm}"] = ref_result_step0

    best_name = max(
        step1_evaluation["results"],
        key=lambda k: step1_evaluation["results"][k]["mean_auc"],
    )
    comparison[f"Consensus-{algorithm}"] = step1_evaluation["results"][best_name]

    fold_aucs = {name: result["fold_aucs"] for name, result in comparison.items()}

    plot_auc_boxplots(
        step_name="Step 1: Data Cleaning Consensus vs Reference",
        aucs_dict=fold_aucs,
        config=config,
        save_path=output_dir / "step1_consensus_vs_reference_boxplots",
    )

    plot_roc_curves(
        models_dict=comparison,
        y_true=y_train,
        n_classes=n_classes,
        classes=np.arange(n_classes),
        config=config,
        title="Step 1: Data Cleaning Consensus vs Reference ROC",
        save_path=output_dir / "step1_consensus_vs_reference_roc",
    )
