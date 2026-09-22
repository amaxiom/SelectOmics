"""
SelectOmics/utils/diagnostics.py

Sample size adequacy assessment for the SelectOmics pipeline.

Assesses four signals before pipeline execution and returns a structured
dict stored on the pipeline object as self.sample_adequacy.  The result
is passed explicitly to each step function and to _warn_step_adequacy;
no global state is used.

References
----------
Varoquaux, G. (2018). Cross-validation failure: Small sample sizes lead to
large error bars. NeuroImage, 180, 68-77.
doi:10.1016/j.neuroimage.2017.06.061

Figueroa, R. L., Zeng-Treitler, Q., Kandula, S., & Ngo, L. H. (2012).
Predicting sample size required for classification performance.
BMC Medical Informatics and Decision Making, 12(1), 8.
doi:10.1186/1472-6947-12-8

Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy
estimation and model selection. IJCAI, 14(2), 1137-1145.
doi:10.5555/1643031.1643047

Also provides assess_feature_redundancy, which measures how much correlated
redundancy a feature matrix carries.  Step 1's correlation filter can only
remove features that have a correlated partner, so this measurement predicts
whether Step 1 will do anything at all -- and therefore whether its documented
recovery-versus-stability trade-off applies to a given dataset.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import logging

from SelectOmics.config import SelectOmicsConfig

logger = logging.getLogger(__name__)

# Flag severity ordering used for overall_severity computation.
_FLAG_ORDER: Dict[str, int] = {'critical': 2, 'caution': 1, 'adequate': 0}
_FLAG_LABEL: Dict[str, str] = {
    'adequate': 'OK',
    'caution':  'CAUTION',
    'critical': 'CRITICAL',
}


def assess_feature_redundancy(
    X: pd.DataFrame,
    correlation_threshold: float = 0.70,
    max_features: int = 500,
    random_state: int = 42,
) -> Dict:
    """
    Measure how much correlated redundancy the feature matrix carries.

    Step 1's correlation filter drops one feature from each pair whose absolute
    correlation exceeds ``correlation_threshold``.  A feature with no such
    partner can never be removed by it.  The fraction of features that do have a
    partner therefore bounds what Step 1 can achieve, and predicts whether
    Step 1 will materially change the run.

    .. warning::

       This measures **total** Pearson correlation, and is therefore an upper
       bound rather than a prediction whenever
       ``config.class_aware_correlation`` is on (the default).  That filter
       judges redundancy on the pooled *within-class* correlation, which is
       strictly weaker: any pair correlated only because both track the label
       falls below the threshold once the class means are removed.  Read the
       number here as "at most this much of the matrix is reachable".

    This matters because Step 1 is not a free improvement.  Measured on
    synthetic omics data (``benchmarks/step1_threshold_sweep.py``), where the
    filter fires it removes 73-92% of features, raising ground-truth recovery
    by roughly 0.15-0.27 F1 while lowering cross-seed stability by 0.5-0.8
    Kuncheva.  Where it does not fire it changes nothing but still costs
    runtime.  Both outcomes are worth knowing in advance.

    A random subset of features is used when ``X`` is wide, because the full
    correlation matrix is O(p^2) and p is routinely 10,000 or more here.  The
    subset is drawn with a fixed seed so repeated calls agree.

    Limitation
    ----------
    This measures correlation without reference to the class label, so it
    cannot distinguish redundant duplication from features that are correlated
    *because they carry the same signal*.  Step 1 removes both alike: where the
    correlation induced among informative features exceeds the threshold, it
    keeps one carrier and discards the rest.  A "high" reading here therefore
    means Step 1 will act decisively, not that acting is desirable.  See
    step1_cleaning.py's KNOWN LIMITATION note and BENCHMARKS.md section 4.2.

    Parameters
    ----------
    X : pd.DataFrame
        Numeric feature matrix, samples in rows.
    correlation_threshold : float
        Absolute correlation above which two features count as redundant.
        Pass ``config.min_correlation_threshold`` to match what Step 1 will use.
    max_features : int
        Cap on the number of features sampled for the correlation matrix.
    random_state : int
        Seed for the feature subsample.

    Returns
    -------
    dict with keys:
        redundant_feature_fraction : float
            Fraction of sampled features having at least one partner above the
            threshold.  This is the quantity that predicts Step 1's reach.
        redundant_pair_fraction : float
            Fraction of sampled feature PAIRS above the threshold.
        median_abs_corr, p95_abs_corr : float
        n_features_sampled, n_features_total : int
        redundancy_level : str
            'high', 'moderate', or 'low'.
        step1_will_act : bool
            Whether Step 1 is expected to remove a material number of features.
        rationale : str
            Human-readable explanation, suitable for logging or printing.
    """
    X_num = X.select_dtypes(include=[np.number])
    n_total = X_num.shape[1]

    if n_total < 2:
        return {
            'redundant_feature_fraction': 0.0,
            'redundant_pair_fraction': 0.0,
            'median_abs_corr': float('nan'),
            'p95_abs_corr': float('nan'),
            'n_features_sampled': n_total,
            'n_features_total': n_total,
            'redundancy_level': 'low',
            'step1_will_act': False,
            'rationale': (
                f"Only {n_total} numeric feature(s); correlation filtering is "
                f"not meaningful."
            ),
        }

    # Subsample columns when the matrix is wide: the correlation matrix is
    # O(p^2) and p here is routinely five figures.
    if n_total > max_features:
        rng = np.random.RandomState(random_state)
        cols = rng.choice(n_total, size=max_features, replace=False)
        X_num = X_num.iloc[:, sorted(cols)]
    n_sampled = X_num.shape[1]

    corr = X_num.corr().abs().to_numpy()
    # Off-diagonal entries only; the diagonal is 1 by construction.
    iu = np.triu_indices(n_sampled, k=1)
    pair_corrs = corr[iu]
    pair_corrs = pair_corrs[~np.isnan(pair_corrs)]

    if pair_corrs.size == 0:
        redundant_pair_fraction = 0.0
        median_abs = p95_abs = float('nan')
    else:
        redundant_pair_fraction = float(
            (pair_corrs > correlation_threshold).mean()
        )
        median_abs = float(np.median(pair_corrs))
        p95_abs = float(np.percentile(pair_corrs, 95))

    # Per feature: does it have at least one partner above the threshold?
    above = corr > correlation_threshold
    np.fill_diagonal(above, False)
    has_partner = above.any(axis=1)
    # A constant feature yields NaN correlations and no partner; that is
    # correct here, since Step 1 removes it as constant rather than as
    # correlated.
    redundant_feature_fraction = float(np.nanmean(has_partner))

    if redundant_feature_fraction >= 0.30:
        level = 'high'
        will_act = True
    elif redundant_feature_fraction >= 0.05:
        level = 'moderate'
        will_act = True
    else:
        level = 'low'
        will_act = False

    sampled_note = (
        f" (estimated from a random {n_sampled} of {n_total} features)"
        if n_sampled < n_total else ""
    )

    if level == 'high':
        rationale = (
            f"{redundant_feature_fraction:.0%} of features have at least one "
            f"partner correlated above {correlation_threshold:g}{sampled_note}. "
            f"Step 1 will remove a large share of the matrix. Expect better "
            f"ground-truth recovery and a shorter feature list, at a real cost "
            f"in run-to-run stability."
        )
    elif level == 'moderate':
        rationale = (
            f"{redundant_feature_fraction:.0%} of features have a correlated "
            f"partner above {correlation_threshold:g}{sampled_note}. Step 1 "
            f"will remove some features but will not dominate the result."
        )
    else:
        rationale = (
            f"Only {redundant_feature_fraction:.1%} of features have a partner "
            f"above {correlation_threshold:g}{sampled_note}. Step 1 has almost "
            f"nothing to remove and will cost runtime without changing the "
            f"outcome; consider enable_step1=False."
        )

    return {
        'redundant_feature_fraction': redundant_feature_fraction,
        'redundant_pair_fraction': redundant_pair_fraction,
        'median_abs_corr': median_abs,
        'p95_abs_corr': p95_abs,
        'n_features_sampled': n_sampled,
        'n_features_total': n_total,
        'redundancy_level': level,
        'step1_will_act': will_act,
        'rationale': rationale,
    }


def assess_sample_size_adequacy(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cv: StratifiedKFold,
    config: SelectOmicsConfig,
) -> Dict:
    """
    Assess sample size adequacy across four signals before pipeline execution.

    Signals assessed
    ----------------
    1. Absolute sample size: general adequacy of n for ML estimation.
    2. Samples-to-features ratio (n/p): risk of overfitting during selection.
    3. Samples per class: per-class adequacy for stratified evaluation.
    4. CV fold reliability: whether min class size supports the configured
       number of CV folds without degenerate splits.

    Thresholds are drawn from published guidance for high-dimensional
    small-sample settings:

    - n/p thresholds: Varoquaux (2018)
      doi:10.1016/j.neuroimage.2017.06.061
    - Per-class minimums: Figueroa et al. (2012)
      doi:10.1186/1472-6947-12-8
    - CV fold reliability: Kohavi (1995)
      doi:10.5555/1643031.1643047

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : np.ndarray
        Integer-encoded training labels.
    cv : StratifiedKFold
        The cross-validation splitter configured for this run.
        cv.n_splits is used for the CV reliability signal.
    config : SelectOmicsConfig
        Pipeline configuration.  config.verbose controls whether the
        diagnostic report is printed.

    Returns
    -------
    dict
        Keys:
        n_samples, n_features, n_classes, samples_per_class,
        min_class_size, n_per_p, n_folds,
        absolute_n_flag, n_per_p_flag, per_class_flag, cv_reliability_flag,
        overall_severity, warnings, recommendations.

        Flag values are 'adequate', 'caution', or 'critical'.
        overall_severity is the worst flag across all four signals.
    """
    n_samples: int = X_train.shape[0]
    n_features: int = X_train.shape[1]
    n_folds: int = cv.n_splits

    classes, class_counts = np.unique(y_train, return_counts=True)
    n_classes: int = len(classes)
    samples_per_class: Dict[int, int] = dict(
        zip(classes.tolist(), class_counts.tolist())
    )
    min_class_size: int = int(class_counts.min())
    n_per_p: float = n_samples / max(n_features, 1)

    warnings_list: List[str] = []
    recommendations_list: List[str] = []

    # ------------------------------------------------------------------
    # Signal 1: Absolute sample size
    # ------------------------------------------------------------------
    if n_samples < 30:
        abs_flag = 'critical'
        warnings_list.append(
            f"CRITICAL: n={n_samples} is very small. Performance estimates "
            f"are unlikely to be reliable regardless of validation strategy."
        )
        recommendations_list.append(
            "Consider acquiring more samples before drawing conclusions. "
            "Treat all AUC estimates as highly uncertain."
        )
    elif n_samples < 100:
        abs_flag = 'caution'
        warnings_list.append(
            f"CAUTION: n={n_samples} is small for machine learning. "
            f"Confidence intervals will be wide and estimates may be unstable."
        )
        recommendations_list.append(
            "Interpret AUC estimates with wide uncertainty. Bootstrap 95% CI "
            "width is the most informative reliability indicator here."
        )
    else:
        abs_flag = 'adequate'

    # ------------------------------------------------------------------
    # Signal 2: Samples-to-features ratio (n/p)
    # Varoquaux (2018) identifies n/p < 0.1 as high risk.
    # ------------------------------------------------------------------
    if n_per_p < 0.1:
        np_flag = 'critical'
        warnings_list.append(
            f"CRITICAL: n/p ratio = {n_per_p:.4f} (n={n_samples}, "
            f"p={n_features}). Severe underdetermination. Feature selection "
            f"results are highly sensitive to sample variation."
        )
        recommendations_list.append(
            "Results should be validated on an independent cohort before use. "
            "Consider more aggressive pre-filtering to reduce p before Step 1."
        )
    elif n_per_p < 1.0:
        np_flag = 'caution'
        warnings_list.append(
            f"CAUTION: n/p ratio = {n_per_p:.4f} (n={n_samples}, "
            f"p={n_features}). High-dimensional setting. Selected features "
            f"may not generalise without external validation."
        )
        recommendations_list.append(
            "External validation on an independent cohort is strongly advised."
        )
    else:
        np_flag = 'adequate'

    # ------------------------------------------------------------------
    # Signal 3: Samples per class
    # Figueroa et al. (2012): floor of 10 per class for stratified CV;
    # fewer than 20 per class produces unstable per-class estimates.
    # ------------------------------------------------------------------
    critical_classes = {c: n for c, n in samples_per_class.items() if n < 10}
    caution_classes = {c: n for c, n in samples_per_class.items()
                       if 10 <= n < 20}

    if critical_classes:
        class_flag = 'critical'
        warnings_list.append(
            f"CRITICAL: Classes with fewer than 10 samples: "
            f"{critical_classes}. Stratified splits will be degenerate or "
            f"impossible for these classes."
        )
        recommendations_list.append(
            "Consider merging rare classes, oversampling, or removing them "
            "from analysis. Evaluation metrics for these classes are not "
            "interpretable."
        )
    elif caution_classes:
        class_flag = 'caution'
        warnings_list.append(
            f"CAUTION: Classes with fewer than 20 samples: "
            f"{caution_classes}. Per-class performance estimates will be "
            f"unstable."
        )
        recommendations_list.append(
            "Interpret per-class metrics cautiously. Macro-averaged AUC "
            "will be dominated by uncertainty in small classes."
        )
    else:
        class_flag = 'adequate'

    # ------------------------------------------------------------------
    # Signal 4: CV fold reliability
    # Kohavi (1995): each fold must contain at least one sample per class.
    # Practical minimum: min_class_size >= n_folds * 2 for stable estimates.
    # ------------------------------------------------------------------
    if min_class_size < n_folds:
        cv_flag = 'critical'
        warnings_list.append(
            f"CRITICAL: Minimum class size ({min_class_size}) is less than "
            f"n_folds ({n_folds}). Stratified CV will fail or produce "
            f"degenerate folds. CV folds will be automatically reduced."
        )
        recommendations_list.append(
            f"Reduce cv_splits in config to at most {min_class_size} "
            f"to avoid degenerate splits."
        )
    elif min_class_size < n_folds * 2:
        cv_flag = 'caution'
        warnings_list.append(
            f"CAUTION: Minimum class size ({min_class_size}) is less than "
            f"2 * n_folds ({n_folds * 2}). Some CV folds will have only one "
            f"sample from the smallest class, making fold-level estimates "
            f"unreliable."
        )
        recommendations_list.append(
            f"Consider reducing cv_splits to "
            f"{max(2, min_class_size // 2)} for more stable estimates."
        )
    else:
        cv_flag = 'adequate'

    # ------------------------------------------------------------------
    # Overall severity: worst flag across all four signals
    # ------------------------------------------------------------------
    all_flags = [abs_flag, np_flag, class_flag, cv_flag]
    overall_severity: str = max(all_flags, key=lambda f: _FLAG_ORDER[f])

    # ------------------------------------------------------------------
    # Print report (respects config.verbose)
    # Data-quality warnings are always printed regardless of verbose so
    # that critical issues are never silently suppressed.
    # ------------------------------------------------------------------
    has_warnings = overall_severity != 'adequate'

    if config.verbose or has_warnings:
        logger.info("SELECTOMICS SAMPLE SIZE ADEQUACY DIAGNOSTIC")
        logger.info("Dataset summary:")
        logger.info("  Samples (n):          %d", n_samples)
        logger.info("  Features (p):         %d", n_features)
        logger.info("  Classes:              %d", n_classes)
        logger.info("  n/p ratio:            %.4f", n_per_p)
        logger.info("  CV folds:             %d", n_folds)
        logger.info("Samples per class:")
        for cls, cnt in samples_per_class.items():
            logger.info("  Class %s: %d samples", cls, cnt)

        logger.info("Adequacy assessment:")
        logger.info("  Absolute sample size:      [%s]", _FLAG_LABEL[abs_flag])
        logger.info("  Samples-to-features (n/p): [%s]", _FLAG_LABEL[np_flag])
        logger.info("  Samples per class:         [%s]", _FLAG_LABEL[class_flag])
        logger.info("  CV fold reliability:       [%s]", _FLAG_LABEL[cv_flag])
        logger.info("  Overall severity: %s", overall_severity.upper())

        if warnings_list:
            logger.warning("Warnings:")
            for w in warnings_list:
                logger.warning("  %s", w)

        if recommendations_list:
            logger.info("Recommendations:")
            for r in recommendations_list:
                logger.info("  %s", r)

        if overall_severity == 'adequate':
            logger.info("  Sample size is adequate for reliable feature selection.")

    # ------------------------------------------------------------------
    # Return structured result
    # ------------------------------------------------------------------
    return {
        'n_samples':            n_samples,
        'n_features':           n_features,
        'n_classes':            n_classes,
        'samples_per_class':    samples_per_class,
        'min_class_size':       min_class_size,
        'n_per_p':              float(n_per_p),
        'n_folds':              n_folds,
        'absolute_n_flag':      abs_flag,
        'n_per_p_flag':         np_flag,
        'per_class_flag':       class_flag,
        'cv_reliability_flag':  cv_flag,
        'overall_severity':     overall_severity,
        'warnings':             warnings_list,
        'recommendations':      recommendations_list,
    }