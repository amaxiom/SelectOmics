"""
SelectOmics/utils/helpers.py

Stateless utility functions shared across pipeline modules.

No globals, no module-level side effects, no imports from other
SelectOmics sub-packages.  All state is received through explicit
parameters.
"""

from __future__ import annotations

import logging
import math
import os
import warnings

logger = logging.getLogger(__name__)
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# ensure_binary_proba
# ---------------------------------------------------------------------------

def ensure_binary_proba(y_proba: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Ensure a probability array has shape (n_samples, n_classes).

    sklearn estimators fitted on binary problems may return a 1-D array
    of positive-class probabilities or a (n_samples, 1) column vector.
    This function expands both forms to a full (n_samples, 2) matrix so
    that downstream code can always index by class.

    For multiclass problems (n_classes > 2) the array is returned
    unchanged provided it is already 2-D.

    Parameters
    ----------
    y_proba : np.ndarray
        Raw probability output from a classifier.  May be 1-D or 2-D.
    n_classes : int
        Number of classes in the problem.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, n_classes).
    """
    y_proba = np.asarray(y_proba, dtype=float)

    if y_proba.ndim == 1:
        y_proba = y_proba.reshape(-1, 1)

    if n_classes == 2 and y_proba.shape[1] == 1:
        p = y_proba[:, 0:1]
        y_proba = np.hstack([1.0 - p, p])

    return y_proba


# ---------------------------------------------------------------------------
# compute_macro_auc_ovr
# ---------------------------------------------------------------------------

def compute_macro_auc_ovr(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
) -> float:
    """
    Compute macro-averaged one-vs-rest AUC robust to missing classes in folds.

    In stratified cross-validation a minority class may be absent from a
    small fold.  sklearn's roc_auc_score raises ValueError in that case.
    This function computes per-class binary AUC only for classes present
    in y_true and averages the results, returning 0.5 (chance level) when
    no valid class AUC can be computed.

    Parameters
    ----------
    y_true : np.ndarray
        Integer class labels, shape (n_samples,).
    y_proba : np.ndarray
        Probability matrix, shape (n_samples, n_classes).  1-D inputs are
        expanded via ensure_binary_proba before scoring.
    classes : np.ndarray
        Array of all class indices known to the pipeline (e.g.
        np.arange(n_classes)).  Used to determine n_classes for
        ensure_binary_proba and to validate column indexing.

    Returns
    -------
    float
        Macro-averaged AUC in [0, 1].  Returns 0.5 if no valid score
        can be computed.
    """
    y_true = np.asarray(y_true)
    n_classes = len(classes)

    y_proba = ensure_binary_proba(y_proba, n_classes)

    # Accept pandas DataFrames from some sklearn versions.
    if hasattr(y_proba, 'values'):
        y_proba = y_proba.values

    present_classes = np.unique(y_true)
    auc_scores = []

    for class_idx in present_classes:
        y_binary = (y_true == class_idx).astype(int)
        # Skip if only one label is present in this slice.
        if len(np.unique(y_binary)) < 2:
            continue
        try:
            auc = roc_auc_score(y_binary, y_proba[:, class_idx])
            auc_scores.append(auc)
        except Exception:
            continue

    return float(np.mean(auc_scores)) if auc_scores else 0.5


# calculate_required_votes(n_models, consensus_threshold) was removed with the
# parameter it served. Every step now starts at unanimity, so the required vote
# count is simply n_models, and relax_consensus_intersection owns the descent
# from there.


# ---------------------------------------------------------------------------
# relax_consensus_intersection
# ---------------------------------------------------------------------------

class ConsensusResult(NamedTuple):
    """Outcome of a graduated consensus selection.

    Attributes
    ----------
    mask : np.ndarray of bool
        Selected features.
    votes_required : int
        Vote level the result was taken at; 0 means rank-averaging was used.
        Because the search always begins at unanimity, this is the STRONGEST
        agreement level at which the returned panel holds, not an artefact of
        where the search happened to start.
    how : str
        'consensus', 'relaxed', 'union', 'consensus_limited', or
        'rank_average'.
        'consensus_limited' means ``min_consensus`` bound before the feature
        floor was reached: the caller asked for more features than survive at
        the agreement level they said was acceptable.
    floor_used : int
        Minimum feature count actually enforced, after capping.
    floor_requested : int
        Minimum originally asked for.  Lower than ``floor_used`` never happens;
        higher means the request exceeded what the votes could support.
    agreement : float
        ``votes_required / n_models``, in [0, 1]. 0.0 for rank-averaging.
    label : str
        Plain-language reading of ``agreement``; see ``agreement_label``.
    """
    mask: np.ndarray
    votes_required: int
    how: str
    floor_used: int
    floor_requested: int
    agreement: float = 0.0
    label: str = 'none'


# Bands for reporting how much model agreement a result actually carries.
#
# The pipeline cannot be asked for an agreement level and a panel size at once:
# they trade against each other and the data decides where the trade lands.
# The caller states the panel size they need, and this reports what that cost
# in agreement. Reporting it in words rather than as a bare ratio is the
# difference between a number the reader has to interpret and a claim they can
# act on.
_AGREEMENT_BANDS = (
    (1.00, 'unanimous'),
    (0.80, 'strong'),
    (0.60, 'moderate'),
    (0.40, 'weak'),
    (0.00, 'minimal'),
)


def agreement_label(agreement: float, how: str = 'consensus') -> str:
    """
    Describe an agreement fraction in words.

    Parameters
    ----------
    agreement : float in [0, 1]
        Fraction of models that had to agree for the panel to survive.
    how : str
        Consensus outcome. ``'rank_average'`` overrides the bands, since a
        ranked result carries no agreement at all.

    Returns
    -------
    str
        One of 'unanimous', 'strong', 'moderate', 'weak', 'minimal', or
        'none (ranked, not agreed)'.
    """
    if how == 'rank_average':
        return 'none (ranked, not agreed)'
    for cutoff, name in _AGREEMENT_BANDS:
        if agreement >= cutoff:
            return name
    return 'minimal'


def relax_consensus_intersection(
    vote_arrays: list,
    n_models: int,
    min_features: int,
    mean_importance: Optional[np.ndarray] = None,
    min_consensus: Optional[float] = None,
    allow_union: bool = False,
) -> "ConsensusResult":
    """
    Select features by intersecting per-stage consensus votes, relaxing the
    vote requirement only as far as needed to reach ``min_features``.

    The rule being enforced is: a feature survives when every stage agrees, and
    each stage agrees when at least ``required`` of its ``n_models`` replicates
    voted for it.  The search ALWAYS begins at unanimity and lowers the
    requirement one vote at a time, stopping at the first level that yields
    enough features.

    Why there is no starting-point parameter
    ----------------------------------------
    There used to be one: ``consensus_threshold``.  It was measured to be
    inert.  Because the search stops at whichever level first satisfies the
    feature floor, and that landing point is a property of the data and the
    floor, starting lower simply skipped rungs on the way to the same answer.
    Across ``omics_standard`` and ``omics_genomics`` at two seeds, thresholds
    of 0.4, 0.6, 0.8 and 1.0 returned not merely the same feature COUNT but the
    identical feature SET in every case.

    Worse, it corrupted the diagnostic.  Relaxation only ever walks downward,
    so a search starting at 4/10 never tested whether 6/10 would also have
    worked, and reported 4 for a panel that in fact held at 6.  Starting at
    unanimity makes ``votes_required`` the strongest level the panel survives,
    which is the only reading that supports a reproducibility claim.

    What ``min_consensus`` does instead
    -----------------------------------
    A floor on agreement, rather than a starting point.  Relaxation stops if
    the next rung would fall below it, even when the feature floor has not been
    met.  This lets a caller say "I would rather have four features at 8/10
    than ten features at 1/10", which was previously inexpressible.  Setting it
    to 1.0 means unanimity or nothing.

    When both bind, the result is the smaller panel, flagged
    ``'consensus_limited'`` so the caller knows their two requirements
    conflicted and which one gave way.

    Rank-averaging is the terminal fallback, reachable only when no feature was
    selected by any replicate in EVERY stage.  It ranks within the union of
    features that received at least one vote somewhere, and steps outside that
    pool only if it is itself too small.  Measured, that restriction changes
    nothing today -- mean importance is exactly zero for a feature no model
    ever used, so ranking already confined itself to the voted pool in 0 of
    161.5 cases on average -- but it makes the guarantee structural rather than
    a happy accident of how tree importances behave.

    Parameters
    ----------
    vote_arrays : list of np.ndarray
        One integer vote-count array per stage, each of shape (n_features,).
    n_models : int
        Replicates per stage; the maximum possible vote count.
    min_features : int
        Minimum acceptable number of surviving features.
    mean_importance : np.ndarray, optional
        Per-feature importance used for the terminal rank-average fallback.
        When omitted, the loosest vote level is returned instead.
    min_consensus : float in (0, 1], optional
        Lowest acceptable agreement fraction.  ``None`` means relax as far as
        the feature floor requires.
    allow_union : bool
        Add a UNION ladder below the intersection ladder.  Off by default.

        The intersection is enforced at every rung of the normal ladder: even
        at one vote a feature must have been kept by some replicate in EVERY
        stage.  That is what caps the result -- the ceiling is the size of the
        intersection at one vote, and no floor can exceed it.  With this set,
        a ladder over ``max`` votes across stages ("kept by r replicates in AT
        LEAST ONE stage") is tried when the intersection cannot reach the
        floor, descending from unanimity as before.  The ceiling rises to the
        union of everything voted for anywhere.

    Returns
    -------
    ConsensusResult
        Carries the mask, the vote level reached, how it was reached, the
        floors, and the agreement fraction with its plain-language label.
    """
    n_features = len(vote_arrays[0])
    # Always start at unanimity: the strongest claim, weakened only as forced.
    start = max(1, int(n_models))

    # Lowest rung the caller will accept, if they set one.
    stop_at = 1
    if min_consensus is not None:
        stop_at = max(1, min(start, int(math.ceil(n_models * min_consensus))))

    # The most the intersection rule can ever return is what survives a single
    # vote in every stage.  Asking for more than that guarantees the ladder is
    # exhausted and rank-averaging pads the result with features no model
    # selected -- and since tree importances tie at exactly zero for every
    # unused feature, that padding is effectively index order.  Measured at
    # n=75, p=2000 with XGBoost: the floor asked for 100 features while only 26
    # were reachable, so 74 were being chosen arbitrarily.  Cap the request at
    # what the votes can support.
    ceiling_mask = np.ones(n_features, dtype=bool)
    for votes in vote_arrays:
        ceiling_mask &= (votes >= 1)
    ceiling = int(ceiling_mask.sum())

    # With a union ladder available the reachable set is larger, and the floor
    # must be clamped to THAT ceiling instead -- otherwise floor_used stays
    # pinned to the intersection ceiling and the union rungs are never reached.
    if allow_union:
        union_mask = np.zeros(n_features, dtype=bool)
        for votes in vote_arrays:
            union_mask |= (votes >= 1)
        ceiling = int(union_mask.sum())

    floor_requested = int(min_features)
    floor_used = max(1, min(floor_requested, ceiling)) if ceiling > 0 else 0

    def _result(mask, required, how):
        agreement = (required / n_models) if n_models else 0.0
        return ConsensusResult(mask, required, how, floor_used,
                               floor_requested, round(agreement, 4),
                               agreement_label(agreement, how))

    if floor_used > 0:
        last_mask, last_required = None, start
        for required in range(start, stop_at - 1, -1):
            mask = np.ones(n_features, dtype=bool)
            for votes in vote_arrays:
                mask &= (votes >= required)
            last_mask, last_required = mask, required
            if mask.sum() >= floor_used:
                how = 'consensus' if required == start else 'relaxed'
                return _result(mask, required, how)

        # The agreement floor bound before the feature floor was reached: the
        # caller asked for more features than survive at the level of agreement
        # they said was acceptable. Honour the agreement floor -- it is the
        # stronger claim -- and say plainly that the two requirements conflict.
        # Union ladder: same graduated descent, but a feature needs its
        # votes in at least one stage rather than in all of them. Only reached
        # when the intersection could not fill the floor.
        if allow_union:
            for required in range(start, stop_at - 1, -1):
                mask = np.zeros(n_features, dtype=bool)
                for votes in vote_arrays:
                    mask |= (votes >= required)
                if mask.sum() >= floor_used:
                    return _result(mask, required, 'union')

        if last_mask is not None and last_mask.any():
            return _result(last_mask, last_required, 'consensus_limited')

    # No feature was selected by any replicate in every stage. Ranking is all
    # that is left, and the caller is told so.
    if mean_importance is not None:
        importance = np.asarray(mean_importance, dtype=float)
        keep = min(n_features, max(1, floor_requested))

        # Rank WITHIN the voted pool: a feature nothing selected anywhere is
        # not evidence, and ordering by an importance of exactly zero is
        # ordering by array index. Step outside only if the pool cannot fill
        # the request.
        voted = np.zeros(n_features, dtype=bool)
        for votes in vote_arrays:
            voted |= (votes >= 1)
        pool = np.flatnonzero(voted)
        if len(pool) >= keep:
            order = pool[np.argsort(-importance[pool], kind='stable')]
        else:
            order = np.argsort(-importance, kind='stable')

        mask = np.zeros(n_features, dtype=bool)
        mask[order[:keep]] = True
        return ConsensusResult(mask, 0, 'rank_average', keep,
                               floor_requested, 0.0,
                               agreement_label(0.0, 'rank_average'))

    return ConsensusResult(ceiling_mask, 1, 'relaxed', ceiling,
                           floor_requested, round(1 / n_models, 4)
                           if n_models else 0.0,
                           agreement_label((1 / n_models) if n_models else 0.0))


# ---------------------------------------------------------------------------
# compute_svm_afi
# ---------------------------------------------------------------------------

def compute_svm_afi(clf) -> np.ndarray:
    """
    Extract normalised absolute feature importance from a fitted LinearSVC.

    The decision function of a LinearSVC is a linear combination of
    features weighted by coef_.  For multiclass problems coef_ has shape
    (n_classes, n_features); the per-feature importance is taken as the
    maximum absolute coefficient across classes, then normalised to [0, 1].

    This function is retained for legacy compatibility with any pipeline
    component that called the original notebook helper of the same name.
    Step 3 uses a LinearSVC surrogate (not RBF SVC) for RFECV importance
    extraction and calls this function via its importance getter.

    Parameters
    ----------
    clf : LinearSVC
        A fitted sklearn LinearSVC instance.  Must have a coef_ attribute.

    Returns
    -------
    np.ndarray
        Normalised importance vector of shape (n_features,), values in [0, 1].

    Raises
    ------
    AttributeError
        If clf does not have a coef_ attribute (i.e. is not fitted or is
        not a linear model).
    """
    coef = np.asarray(clf.coef_)

    # coef_ is (1, n_features) for binary and (n_classes, n_features) for
    # multiclass.  Take max absolute value across the class axis.
    importance = np.abs(coef).max(axis=0)

    # Normalise to [0, 1].  Guard against the degenerate case where all
    # coefficients are zero (e.g. the model failed to converge).
    importance_range = importance.max() - importance.min()
    if importance_range > 0:
        importance = (importance - importance.min()) / importance_range
    else:
        importance = np.zeros_like(importance)

    return importance


# ---------------------------------------------------------------------------
# _binary_search_percentile
# ---------------------------------------------------------------------------

def _binary_search_percentile(
    evaluate_fn: Callable[[float], float],
    lo: float,
    hi: float,
    target: float,
    tolerance: float = 0.01,
    max_iter: int = 10,
    label: str = "binary search",
) -> Tuple[float, float, int]:
    """
    Binary search for the percentile threshold whose average selected feature
    count is closest to a target.

    Exploits the monotonic non-increasing relationship between percentile
    threshold and retained feature count: a higher threshold drops more
    features.  Bisection converges in O(log n) evaluate_fn calls rather
    than exhaustive grid search.

    Parameters
    ----------
    evaluate_fn : callable
        Maps a float percentile in [lo, hi] to a float average feature
        count.  Each call typically fits n_models models, so minimising
        the call count is important.
    lo : float
        Lower bound of the search range (inclusive).
    hi : float
        Upper bound of the search range (inclusive).
    target : float
        Desired average feature count.
    tolerance : float
        Convergence threshold on interval width.  Search halts when
        hi - lo < tolerance.  Default 0.01 provides percentile precision
        adequate for all pipeline steps.
    max_iter : int
        Maximum bisection steps after the two anchor evaluations.  Total
        evaluate_fn calls <= max_iter + 2.
    label : str
        Name of the caller, used in the unreachable-target warning.

    Returns
    -------
    best_percentile : float
        Percentile whose evaluate_fn output is closest to target.
    best_avg_features : float
        Average feature count at best_percentile.
    n_evaluations : int
        Total number of evaluate_fn calls made.

    Notes
    -----
    Both anchor points (lo and hi) are always evaluated so the best
    tracker is valid from the first iteration regardless of whether the
    target lies inside the range.
    """
    lo_features = evaluate_fn(lo)
    hi_features = evaluate_fn(hi)
    n_evaluations = 2

    # Initialise best from the closer anchor.
    if abs(lo_features - target) <= abs(hi_features - target):
        best_percentile, best_avg_features = lo, lo_features
    else:
        best_percentile, best_avg_features = hi, hi_features

    for _ in range(max_iter):
        if hi - lo < tolerance:
            break

        mid = (lo + hi) / 2.0
        mid_features = evaluate_fn(mid)
        n_evaluations += 1

        if abs(mid_features - target) < abs(best_avg_features - target):
            best_percentile = mid
            best_avg_features = mid_features

        # Bisect toward the side that might contain the target.
        # Tracking endpoint values handles out-of-range targets correctly.
        # IMPORTANT: the out-of-range branches must only fire when BOTH
        # endpoints are on the same side of the target.  Checking only
        # lo_features > target (without verifying hi_features > target) can
        # move lo past the target when the target is already in [hi, lo],
        # permanently excluding the optimal interval from the search.
        if lo_features > target and hi_features > target:
            # Both endpoints above target; raise lower bound toward hi.
            lo = mid
            lo_features = mid_features
        elif hi_features < target and lo_features < target:
            # Both endpoints below target; lower upper bound toward lo.
            hi = mid
            hi_features = mid_features
        else:
            # Target is between lo and hi. Standard bisection.
            # evaluate_fn is assumed DECREASING (larger input -> smaller output).
            if mid_features > target:
                lo = mid
                lo_features = mid_features
            else:
                hi = mid
                hi_features = mid_features

    # Say so when the target was never reachable.
    #
    # The search silently returns its closest endpoint when the target lies
    # outside what evaluate_fn can produce, which is how an inert percentile
    # went unnoticed: Step 2's threshold returned an identical feature count at
    # every point in its range, the search dutifully reported an "optimum" to
    # four decimal places, and a configuration search then tuned that number.
    # A caller asking for 60% retention and receiving 0.6% deserves to be told.
    if target > 0 and abs(best_avg_features - target) > max(1.0, 0.1 * target):
        warnings.warn(
            f"{label} could not reach its target of "
            f"{target:.0f} features; the closest achievable was "
            f"{best_avg_features:.0f} at percentile {best_percentile:.4f}. "
            f"The threshold is saturated -- the models did not distinguish "
            f"enough features for the requested retention to be possible, so "
            f"the retention setting is not what determined this result.",
            UserWarning,
            stacklevel=2,
        )

    return best_percentile, best_avg_features, n_evaluations


# ---------------------------------------------------------------------------
# print_adequacy_warning  (canonical)
# ---------------------------------------------------------------------------

def print_adequacy_warning(
    step_name: str,
    n_features_current: int,
    adequacy: Optional[Dict],
) -> None:
    """
    Print contextual sample-size adequacy warnings for a pipeline step.

    Always prints when a non-adequate condition is detected.  Callers
    should decide whether to invoke this function at all based on their
    own verbose/config logic; this function never suppresses warnings
    silently.

    Parameters
    ----------
    step_name : str
    n_features_current : int
        Feature count at the current step, used to recompute n/p.
    adequacy : dict or None
        Output of assess_sample_size_adequacy.  If None, returns immediately.
    """
    if adequacy is None:
        return
    if adequacy.get('overall_severity') == 'adequate':
        return

    n_samples = adequacy.get('n_samples', 0)
    current_n_per_p = n_samples / max(n_features_current, 1)

    active_warnings = []

    if adequacy.get('absolute_n_flag') != 'adequate':
        active_warnings.append(
            f"n={n_samples} is small: {step_name} performance estimates "
            f"carry high uncertainty."
        )

    if current_n_per_p < 0.1:
        active_warnings.append(
            f"n/p = {current_n_per_p:.4f} at {step_name} "
            f"(n={n_samples}, p={n_features_current}): severe "
            f"underdetermination persists."
        )
    elif current_n_per_p < 1.0:
        active_warnings.append(
            f"n/p = {current_n_per_p:.4f} at {step_name}: "
            f"high-dimensional regime, interpret results cautiously."
        )

    if adequacy.get('per_class_flag') != 'adequate':
        active_warnings.append(
            f"Small class sizes detected: per-class metrics at "
            f"{step_name} are unreliable."
        )

    if adequacy.get('cv_reliability_flag') != 'adequate':
        active_warnings.append(
            f"CV fold reliability is {adequacy.get('cv_reliability_flag')}: "
            f"fold-level AUC estimates at {step_name} may be unstable."
        )

    if active_warnings:
        logger.warning("  ADEQUACY WARNINGS (%s):", step_name)
        for warning in active_warnings:
            logger.warning("    %s", warning)


# ---------------------------------------------------------------------------
# _warn_step_adequacy  (deprecated alias for backward compatibility)
# ---------------------------------------------------------------------------

def _warn_step_adequacy(
    step_name: str,
    n_features_current: int,
    adequacy: Dict,
    verbose: bool = True,
) -> None:
    """Deprecated alias for print_adequacy_warning. Use print_adequacy_warning instead."""
    if verbose:
        print_adequacy_warning(step_name, n_features_current, adequacy)

# ---------------------------------------------------------------------------
# save_step_summary
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# detect_gpu
# ---------------------------------------------------------------------------

def detect_gpu() -> bool:
    """
    Return True if an XGBoost-compatible GPU is available.

    Attempts to fit a trivial XGBClassifier with device='cuda'.  Returns
    False on any exception (missing CUDA, driver mismatch, xgboost < 2.0).
    """
    try:
        from xgboost import XGBClassifier
        m = XGBClassifier(n_estimators=1, device='cuda', verbosity=0)
        m.fit(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0, 1]))
        return True
    except Exception:
        return False


def detect_cpu_count() -> int:
    """
    Return the number of cores available to this process, defaulting to 1.

    Mirrors the graceful degradation used for GPU detection: an environment
    that will not report a core count (some restricted containers) yields 1
    rather than an error, so the pipeline runs serially instead of failing.

    Returns
    -------
    int
        Usable core count, at least 1.
    """
    try:
        # os.process_cpu_count (3.13+) respects CPU affinity masks, which
        # os.cpu_count does not; cgroup-limited containers report the host
        # count from the latter and would oversubscribe badly.
        count = getattr(os, 'process_cpu_count', None)
        if count is not None:
            n = count()
        else:
            n = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') \
                else os.cpu_count()
        return max(1, int(n or 1))
    except Exception:
        return 1


def save_step_summary(
    step_name: str,
    results: Dict[str, Any],
    save_dir: Path,
) -> None:
    """
    Persist a plain-text summary of a pipeline step's key metrics.

    Writes one key-value pair per line to
    ``{save_dir}/{step_name}_summary.txt``.  The file is created or
    overwritten on each call.  All values are converted to strings via
    ``str()``, so the function is safe for any JSON-serialisable type.

    Parameters
    ----------
    step_name : str
        Short identifier used as the filename stem, e.g. 'step1'.
    results : dict
        Key-value pairs to record.  Typical keys include
        'algorithm', 'n_features_in', 'n_features_out', 'mean_auc'.
    save_dir : Path
        Destination directory.  Must exist before calling this function.

    Returns
    -------
    None
    """
    summary_path = Path(save_dir) / f"{step_name}_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as fh:
        fh.write(f"Step: {step_name}\n")
        fh.write(f"Timestamp: {datetime.now().isoformat()}\n")
        fh.write("=" * 50 + "\n")
        for key, value in results.items():
            fh.write(f"{key}: {value}\n")
