"""
SelectOmics/evaluation/validation.py

Comprehensive feature-set validation for SelectOmics.

FeatureSetValidator runs three independent validation protocols on each
candidate feature set and compares them across pipeline steps:

1. Stratified k-fold cross-validation (3 model seeds for robustness).
2. Leave-one-out or stratified shuffle-split (automatic for n > 50).
3. Bootstrap out-of-bag validation (n_bootstrap iterations).

compare_feature_sets produces the pipeline-level comparison DataFrame
used to select the recommended feature set for downstream analysis.  The
adequacy flag combines empirical bootstrap CI width with the upfront
sample-size severity from diagnostics.assess_sample_size_adequacy.

All global state from the notebook (SELECTOMICS_CONFIG, n_classes,
SELECTOMICS_SAMPLE_ADEQUACY, global step variables) is replaced by
explicit constructor and method parameters.  No module-level execution.

Source: notebook Cell 16.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    LeaveOneOut,
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_score,
)
from sklearn.utils import resample

from SelectOmics.config import SelectOmicsConfig
from SelectOmics.evaluation.metrics import _safe_roc_auc
from SelectOmics.models.base import (
    create_lr_pipeline_consensus,
    create_rf_pipeline_consensus,
    create_svm_pipeline_consensus,
    create_xgb_pipeline_consensus,
)


# ---------------------------------------------------------------------------
# Adequacy flag helpers
# ---------------------------------------------------------------------------

_FLAG_ORDER: Dict[str, int] = {'adequate': 0, 'caution': 1, 'critical': 2}
_FLAG_LABELS: List[str]     = ['adequate', 'caution', 'critical']


# A validation protocol that loses most of its iterations still returns a
# number, and that number is reported with the same authority as a complete
# one. Both loops already count their failures; neither surfaced the count
# above DEBUG, so a CI computed from 5 surviving iterations of 100 looked
# identical to one computed from 100. Warn below this success rate.
_MIN_SUCCESS_RATE: float = 0.5

# Smallest AUC difference allowed to justify a larger feature panel.
#
# Variance-based tolerances collapse to zero when the metric saturates, which
# is the normal case at n << p: every panel scores 1.0 and the standard error
# is 0.0000. Without a floor the parsimony rule silently reverts to argmax
# exactly where it is needed most. Half a point of AUC is below anything that
# changes a downstream decision.
MIN_SCORE_TOLERANCE: float = 0.005

# Largest validation gain over the least-selected panel that is taken at face
# value.
#
# Validation cross-validates each panel on the same samples its features were
# selected from, so a selected panel's score is optimistic, and the more
# aggressive the selection, the more optimistic. The least-selected panel
# (Step 0, every feature) carries no selection at all, which makes it the one
# unbiased reference in the comparison. Measured with nested CV: on GBM, OV
# and LGG miRNA and a 5000-feature synthetic set, 20 outer folds in all, the
# recommended panel's validation score exceeded the reference's by at most
# 0.021. On a null control, where selection can only fit noise, it exceeded it
# by 0.23 to 0.37 while its held-out AUC was 0.46. A gain above this is flagged
# rather than trusted: from the training split alone it cannot be told apart
# from selection fitting noise.
MAX_PLAUSIBLE_SELECTION_GAIN: float = 0.05


def _warn_if_degraded(protocol: str, name: str, ok: int, attempted: int) -> None:
    """Warn when a validation protocol completed too few of its iterations."""
    if attempted <= 0 or ok >= attempted * _MIN_SUCCESS_RATE:
        return
    logger.warning(
        "%s validation for '%s' completed only %d of %d iterations (%.0f%%). "
        "The reported AUC and confidence interval rest on that many samples, "
        "not on %d. Treat them as provisional.",
        protocol, name, ok, attempted, 100.0 * ok / attempted, attempted,
    )


def _ci_width_flag(width: float) -> str:
    """Classify a bootstrap CI width as adequate / caution / critical."""
    if width > 0.3:
        return 'critical'
    if width > 0.2:
        return 'caution'
    return 'adequate'


def _worse_flag(a: str, b: str) -> str:
    """Return the more severe of two flag strings."""
    return _FLAG_LABELS[max(_FLAG_ORDER[a], _FLAG_ORDER[b])]


# ---------------------------------------------------------------------------
# FeatureSetValidator
# ---------------------------------------------------------------------------

class FeatureSetValidator:
    """
    Comprehensive validation of candidate feature sets.

    Parameters
    ----------
    config : SelectOmicsConfig
        Pipeline configuration.  Uses algorithm, random_seed, verbose.
    n_classes : int
        Number of classes in the classification problem.
    tuned_pipelines : dict
        Output of models.base.quick_tune_all.  Used to instantiate
        consensus pipelines that carry tuned hyperparameters.
    n_bootstrap : int
        Number of bootstrap iterations for bootstrap_validation.
    """

    def __init__(
        self,
        config: SelectOmicsConfig,
        n_classes: int,
        tuned_pipelines: Dict[str, Any],
        n_bootstrap: Optional[int] = None,
    ) -> None:
        self.config          = config
        self.n_classes       = n_classes
        self.tuned_pipelines = tuned_pipelines
        self.n_bootstrap     = n_bootstrap if n_bootstrap is not None else getattr(config, 'n_bootstrap', 100)
        self.results: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Model factory
    # ------------------------------------------------------------------

    def _create_model(self, seed: int) -> Any:
        """
        Build a consensus pipeline for the configured algorithm.

        Parameters
        ----------
        seed : int
            Random seed passed to the pipeline factory.

        Returns
        -------
        Unfitted sklearn Pipeline.
        """
        alg     = self.config.algorithm
        tuned   = self.tuned_pipelines[alg]
        factory = {
            'LR':  create_lr_pipeline_consensus,
            'XGB': create_xgb_pipeline_consensus,
            'RF':  create_rf_pipeline_consensus,
            'SVM': create_svm_pipeline_consensus,
        }[alg]
        return factory(seed=seed, **{f'tuned_{alg.lower()}': tuned})

    # ------------------------------------------------------------------
    # Stratified CV
    # ------------------------------------------------------------------

    def stratified_cv_validation(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_set_name: str,
        cv_folds: int = 10,
    ) -> Dict[str, Any]:
        """
        Stratified k-fold cross-validation repeated with 3 model seeds.

        The fold count is clamped to the minimum class size.  Three model
        seeds are used so that performance estimates are not artefacts of
        a single random initialisation.

        Parameters
        ----------
        X : pd.DataFrame
        y : np.ndarray
            Integer-encoded labels.
        feature_set_name : str
            Used in verbose output.
        cv_folds : int
            Target number of folds; clamped to min class size.

        Returns
        -------
        dict
            mean_auc, std_auc, all_scores (list), fold_details (list),
            cv_folds (int).
        """
        if self.config.verbose:
            logger.debug("1. STRATIFIED CROSS-VALIDATION: %s", feature_set_name)

        min_class  = int(np.bincount(y, minlength=self.n_classes).min())
        actual_cv  = min(cv_folds, max(2, min_class))
        cv_obj     = StratifiedKFold(
            n_splits=actual_cv,
            shuffle=True,
            random_state=self.config.random_seed,
        )
        seed       = self.config.random_seed

        if self.config.verbose:
            logger.debug("Features: %d, Samples: %d", X.shape[1], X.shape[0])
            logger.debug("CV Folds: %d (limited by min class size: %d)", actual_cv, min_class)

        all_scores: List[float] = []
        fold_details: List[Dict[str, Any]] = []

        for run in range(3):
            model_seed = seed + run * 100
            model      = self._create_model(model_seed)
            scores     = cross_val_score(
                model, X, y,
                cv=cv_obj,
                scoring='roc_auc_ovr',
                n_jobs=1,
            )
            all_scores.extend(scores.tolist())

            # Detailed fold breakdown on first seed only.
            if run == 0:
                for fold_idx, (tr_idx, va_idx) in enumerate(cv_obj.split(X, y)):
                    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
                    y_tr, y_va = y[tr_idx], y[va_idx]

                    # Same guard as leave_one_out_validation and
                    # bootstrap_validation. A training split missing a class
                    # yields a predict_proba matrix narrower than n_classes,
                    # and roc_auc_score rejects it with "Number of classes in
                    # y_true not equal to the number of columns in y_score".
                    # The other two protocols skip such an iteration; this one
                    # raised that ValueError out of the entire validation,
                    # losing every feature set's result and not only this
                    # fold's diagnostic row.
                    #
                    # Reachable whenever the training split holds a class with
                    # one member, since actual_cv floors at 2 and a 1-member
                    # class cannot be in both training splits.
                    if len(np.unique(y_tr)) < self.n_classes:
                        continue

                    m = clone(self._create_model(model_seed))
                    m.fit(X_tr, y_tr)
                    proba = m.predict_proba(X_va)
                    fold_details.append({
                        'fold':           fold_idx + 1,
                        'train_size':     len(tr_idx),
                        'val_size':       len(va_idx),
                        'val_auc':        _safe_roc_auc(y_va, proba, self.n_classes),
                        'val_class_dist': np.bincount(y_va, minlength=self.n_classes).tolist(),
                    })

        # cross_val_score returns nan for a fold it could not score rather
        # than raising, so those nans flow into the mean below and turn the
        # whole feature set's cv_auc into nan. The aggregation is left as it
        # is, because a mean over folds that could not be scored would be a
        # different quantity reported under the same name. What was missing
        # is the same courtesy the other two protocols get from
        # _warn_if_degraded: saying how many iterations were actually used.
        _n_ok = int(np.sum(~np.isnan(all_scores))) if all_scores else 0
        _warn_if_degraded('Cross-validation', feature_set_name,
                          _n_ok, len(all_scores))

        result = {
            # Guarded like std_auc below: an empty score list means every
            # fold failed, and np.mean([]) reaches nan via a RuntimeWarning
            # rather than by intent.
            'mean_auc':    float(np.mean(all_scores)) if all_scores else float('nan'),
            'std_auc':     float(np.std(all_scores, ddof=1)) if len(all_scores) > 1 else 0.0,
            'all_scores':  all_scores,
            'fold_details': fold_details,
            'cv_folds':    actual_cv,
        }

        if self.config.verbose:
            logger.debug("CV AUC: %.4f +/- %.4f", result['mean_auc'], result['std_auc'])
            # Guarded for the same reason mean_auc above is: all_scores can be
            # empty when every fold failed, and min(()) raises ValueError.
            if all_scores:
                logger.debug("Score range: [%.4f, %.4f]",
                             min(all_scores), max(all_scores))

        return result

    # ------------------------------------------------------------------
    # LOO / stratified shuffle
    # ------------------------------------------------------------------

    def leave_one_out_validation(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_set_name: str,
    ) -> Dict[str, Any]:
        """
        Leave-one-out or stratified shuffle-split validation.

        Full LOO is used when n <= 50.  For larger datasets a stratified
        shuffle-split with 30 iterations is used to keep runtime tractable.
        Per-class prediction confidence is tracked for diagnostics.

        Parameters
        ----------
        X : pd.DataFrame
        y : np.ndarray
        feature_set_name : str

        Returns
        -------
        dict
            mean_auc, std_auc, all_scores, failed_predictions,
            class_predictions, total_tests.
        """
        if self.config.verbose:
            logger.debug("2. LEAVE-ONE-OUT VALIDATION: %s", feature_set_name)

        seed = self.config.random_seed

        if X.shape[0] > 50:
            # test_size must be large enough so that sklearn's proportional
            # allocation gives at least 1 slot to the rarest class.
            # For a class with count k out of n samples, the minimum test size
            # that guarantees at least 1 test sample from that class is:
            #     ceil(n / k)
            # The original formula max(3, min(5, n//10)) is used as a lower
            # bound; we override upward when needed and cap at n//5 (20%).
            _class_counts    = np.bincount(y.astype(int))
            _min_class_count = int(_class_counts.min())
            _min_for_rarest  = int(np.ceil(X.shape[0] / max(_min_class_count, 1)))
            _baseline        = max(3, min(5, X.shape[0] // 10))
            test_size        = max(_baseline, _min_for_rarest, self.n_classes)
            test_size        = min(test_size, X.shape[0] // 5)   # cap at 20 %
            test_size        = max(test_size, 3)                  # absolute floor
            splitter   = StratifiedShuffleSplit(
                n_splits=30,
                test_size=test_size,
                random_state=seed,
            )
            cv_splits  = list(splitter.split(X, y))
            if self.config.verbose:
                logger.debug(
                    "Using stratified shuffle (%d samples/split) on %d samples "
                    "(min_class=%d, n_classes=%d)",
                    test_size, X.shape[0], _min_class_count, self.n_classes,
                )
        else:
            cv_splits = list(LeaveOneOut().split(X, y))
            if self.config.verbose:
                logger.debug("Full LOO on %d samples", X.shape[0])

        loo_scores: List[float] = []
        failed = 0
        class_preds: Dict[int, List[float]] = {c: [] for c in range(self.n_classes)}

        for i, (tr_idx, te_idx) in enumerate(cv_splits):
            if self.config.verbose and i % 10 == 0:
                logger.debug("  Progress: %d/%d", i + 1, len(cv_splits))

            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            if len(np.unique(y_tr)) < self.n_classes:
                failed += 1
                continue

            try:
                m = self._create_model(seed + i)
                m.fit(X_tr, y_tr)
                proba = m.predict_proba(X_te)

                if len(te_idx) == 1:
                    # Single-sample LOO: use probability of true class.
                    # Guard against nan probabilities (degenerate model)
                    # for consistency with the multi-sample branch below.
                    true_cls = int(y_te[0])
                    prob_val = float(proba[0, true_cls])
                    if not np.isnan(prob_val):
                        class_preds[true_cls].append(prob_val)
                        loo_scores.append(prob_val)
                else:
                    auc = _safe_roc_auc(y_te, proba, self.n_classes)
                    if not np.isnan(auc):
                        loo_scores.append(float(auc))
                    for j, tc in enumerate(y_te):
                        class_preds[int(tc)].append(float(proba[j, int(tc)]))
            except Exception:
                failed += 1

        # score_type documents what the scores represent:
        # 'confidence' when full LOO is used (n <= 50): each score is the
        # predicted probability for the true class of a single held-out sample.
        # 'auc' when stratified shuffle-split is used (n > 50).
        score_type = 'auc' if X.shape[0] > 50 else 'confidence'
        result = {
            'mean_auc':           float(np.mean(loo_scores)) if loo_scores else 0.0,
            'std_auc':            float(np.std(loo_scores, ddof=1)) if len(loo_scores) > 1 else 0.0,
            'all_scores':         loo_scores,
            'score_type':         score_type,
            'failed_predictions': failed,
            'class_predictions':  class_preds,
            'total_tests':        len(cv_splits),
        }

        _warn_if_degraded('Leave-one-out', feature_set_name,
                          len(cv_splits) - failed, len(cv_splits))

        if self.config.verbose:
            logger.debug("LOO AUC: %.4f +/- %.4f", result['mean_auc'], result['std_auc'])
            logger.debug("Failed: %d/%d", failed, len(cv_splits))
            for cls, conf in class_preds.items():
                if conf:
                    logger.debug(
                        "Class %s avg confidence: %.4f +/- %.4f (n=%d)",
                        cls, np.mean(conf), np.std(conf), len(conf),
                    )

        return result

    # ------------------------------------------------------------------
    # Bootstrap OOB
    # ------------------------------------------------------------------

    def bootstrap_validation(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_set_name: str,
    ) -> Dict[str, Any]:
        """
        Bootstrap out-of-bag validation.

        Each iteration draws a bootstrap sample with replacement.  The
        OOB set (samples not drawn) forms the evaluation set.  Iterations
        where all samples of any class are in the bootstrap sample (empty
        OOB for that class) are skipped and counted as failed.

        Parameters
        ----------
        X : pd.DataFrame
        y : np.ndarray
        feature_set_name : str

        Returns
        -------
        dict
            mean_auc, std_auc, all_scores, details, successful_iterations,
            confidence_interval ([2.5th, 97.5th percentile]).
        """
        if self.config.verbose:
            logger.debug("3. BOOTSTRAP VALIDATION: %s", feature_set_name)
            logger.debug("Running %d bootstrap iterations...", self.n_bootstrap)

        seed = self.config.random_seed
        scores: List[float] = []
        details: List[Dict[str, Any]] = []

        for i in range(self.n_bootstrap):
            if self.config.verbose and i % 20 == 0:
                logger.debug("  Progress: %d/%d", i + 1, self.n_bootstrap)

            try:
                boot_idx = resample(
                    np.arange(len(X)),
                    n_samples=len(X),
                    replace=True,
                    stratify=y,
                    random_state=seed + i,
                )
            except Exception:
                # stratify can fail if a class has only 1 sample;
                # fall back to unstratified resampling.
                boot_idx = resample(
                    np.arange(len(X)),
                    n_samples=len(X),
                    replace=True,
                    random_state=seed + i,
                )
            # Boolean mask rather than a per-element membership test: the
            # comprehension form rebuilt the whole set once per sample, making
            # OOB extraction O(n^2) per iteration.
            in_bag = np.zeros(len(X), dtype=bool)
            in_bag[np.asarray(boot_idx)] = True
            oob_idx = np.flatnonzero(~in_bag)

            if len(oob_idx) == 0:
                continue

            X_boot, y_boot = X.iloc[boot_idx], y[boot_idx]
            X_oob,  y_oob  = X.iloc[oob_idx],  y[oob_idx]

            if len(np.unique(y_boot)) < self.n_classes:
                continue

            try:
                m = self._create_model(seed + i)
                m.fit(X_boot, y_boot)
                oob_proba = m.predict_proba(X_oob)
                auc = _safe_roc_auc(y_oob, oob_proba, self.n_classes)

                if not np.isnan(auc):
                    scores.append(float(auc))
                    details.append({
                        'iteration':      i + 1,
                        'boot_size':      len(boot_idx),
                        'oob_size':       len(oob_idx),
                        'oob_auc':        float(auc),
                        'boot_class_dist': np.bincount(y_boot, minlength=self.n_classes).tolist(),
                        'oob_class_dist':  np.bincount(y_oob,  minlength=self.n_classes).tolist(),
                    })
            except Exception as exc:
                # An iteration that cannot be fitted or scored is skipped, and
                # _warn_if_degraded below reports how many were lost. It cannot
                # report why, because this used to discard the exception, which
                # left a user told their CI rests on 12 of 100 iterations with
                # no way to find out what went wrong. DEBUG keeps it silent on
                # a normal run and available when diagnosing one.
                logger.debug("Bootstrap iteration %d failed: %s: %s",
                             i + 1, type(exc).__name__, exc)

        ci = (
            list(np.percentile(scores, [2.5, 97.5]).tolist())
            if scores else [0.0, 0.0]
        )
        result = {
            'mean_auc':             float(np.mean(scores)) if scores else 0.0,
            'std_auc':              float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            'all_scores':           scores,
            'details':              details,
            'successful_iterations': len(scores),
            'confidence_interval':  ci,
        }

        _warn_if_degraded('Bootstrap', feature_set_name,
                          len(scores), self.n_bootstrap)

        if self.config.verbose:
            logger.debug("Bootstrap AUC: %.4f +/- %.4f", result['mean_auc'], result['std_auc'])
            logger.debug("95%% CI: [%.4f, %.4f]", ci[0], ci[1])
            logger.debug("Successful iterations: %d/%d", len(scores), self.n_bootstrap)

        return result

    # ------------------------------------------------------------------
    # validate_feature_set
    # ------------------------------------------------------------------

    def validate_feature_set(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_set_name: str,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Run all three validation protocols on one feature set.

        Parameters
        ----------
        X : pd.DataFrame
        y : np.ndarray
        feature_set_name : str
        save_results : bool
            If True, stores the result under self.results[feature_set_name].

        Returns
        -------
        dict
            feature_set_name, n_features, n_samples, algorithm,
            stratified_cv, leave_one_out, bootstrap.
        """
        if self.config.verbose:
            logger.debug("VALIDATING FEATURE SET: %s", feature_set_name)
            logger.debug("Features: %d, Samples: %d", X.shape[1], X.shape[0])
            class_dist = dict(zip(*np.unique(y, return_counts=True)))
            logger.debug("Class distribution: %s", class_dist)

        cv_res   = self.stratified_cv_validation(X, y, feature_set_name)
        loo_res  = self.leave_one_out_validation(X, y, feature_set_name)
        boot_res = self.bootstrap_validation(X, y, feature_set_name)

        # Consistency and stability assessment.
        all_means      = [cv_res['mean_auc'], loo_res['mean_auc'], boot_res['mean_auc']]
        # ddof=0 on purpose, unlike the ddof=1 used for every std_auc above.
        # These three numbers are the complete set of protocols being compared,
        # not a sample drawn from a population, so the spread of exactly these
        # three is what is wanted. Changing this to ddof=1 for consistency with
        # the other standard deviations would be wrong.
        consistency_sd = float(np.std(all_means))
        cv_stability   = (cv_res['std_auc'] / cv_res['mean_auc']
                          if cv_res['mean_auc'] > 0 else float('inf'))
        overall_perf   = float(np.mean(all_means))

        if self.config.verbose:
            logger.debug("VALIDATION SUMMARY")
            logger.debug("Stratified CV: %.4f +/- %.4f", cv_res['mean_auc'], cv_res['std_auc'])
            logger.debug("Leave-One-Out: %.4f +/- %.4f", loo_res['mean_auc'], loo_res['std_auc'])
            logger.debug("Bootstrap:     %.4f +/- %.4f", boot_res['mean_auc'], boot_res['std_auc'])
            logger.debug(
                "95%% CI:        [%.4f, %.4f]",
                boot_res['confidence_interval'][0], boot_res['confidence_interval'][1],
            )
            logger.debug("Consistency SD: %.4f", consistency_sd)
            logger.debug("CV stability (std/mean): %.4f", cv_stability)
            logger.debug("Average performance: %.4f", overall_perf)

        validation_results = {
            'feature_set_name': feature_set_name,
            'n_features':       X.shape[1],
            'n_samples':        X.shape[0],
            'algorithm':        self.config.algorithm,
            'stratified_cv':    cv_res,
            'leave_one_out':    loo_res,
            'bootstrap':        boot_res,
        }

        if save_results:
            self.results[feature_set_name] = validation_results

        return validation_results

    # ------------------------------------------------------------------
    # compare_feature_sets
    # ------------------------------------------------------------------

    def compare_feature_sets(
        self,
        feature_sets: Dict[str, pd.DataFrame],
        y: np.ndarray,
        upfront_severity: str = 'adequate',
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """
        Validate each feature set and return a comparison DataFrame.

        Adequacy flags combine empirical bootstrap CI width with the
        upfront sample-size severity from diagnostics.assess_sample_size_adequacy.
        The overall flag is the worse of the two signals.

        Parameters
        ----------
        feature_sets : dict
            Keys: feature set names.  Values: training-only DataFrames.
            The held-out test set must not be passed here; all three
            validation protocols (CV, LOO, bootstrap) use only training
            samples to avoid test-set leakage.
        y : np.ndarray
            Training labels aligned to the training DataFrames.
        upfront_severity : str
            Overall severity from assess_sample_size_adequacy.
            One of 'adequate', 'caution', 'critical'.

        Returns
        -------
        comparison_df : pd.DataFrame
            One row per feature set with cv_auc, loo_auc, bootstrap_auc,
            CI bounds, adequacy flags.
        all_results : dict
            Per-feature-set validate_feature_set output dicts.
        """
        all_results: Dict[str, Dict[str, Any]] = {}
        for name, X_fs in feature_sets.items():
            res = self.validate_feature_set(X_fs, y, name, save_results=False)
            all_results[name] = res

        # Build comparison DataFrame.
        rows: Dict[str, Dict[str, Any]] = {}
        for name, res in all_results.items():
            rows[name] = {
                'n_features':        res['n_features'],
                # Needed to turn cv_std into a standard error in
                # build_recommendation's one-standard-error rule.
                'cv_folds':          res['stratified_cv'].get('cv_folds', 0),
                'cv_auc':            res['stratified_cv']['mean_auc'],
                'cv_std':            res['stratified_cv']['std_auc'],
                'loo_auc':           res['leave_one_out']['mean_auc'],
                'loo_std':           res['leave_one_out']['std_auc'],
                'bootstrap_auc':     res['bootstrap']['mean_auc'],
                'bootstrap_std':     res['bootstrap']['std_auc'],
                'bootstrap_ci_lower': res['bootstrap']['confidence_interval'][0],
                'bootstrap_ci_upper': res['bootstrap']['confidence_interval'][1],
            }

        comparison_df = pd.DataFrame(rows).T

        # Adequacy flags.
        comparison_df['bootstrap_ci_width'] = (
            comparison_df['bootstrap_ci_upper'] - comparison_df['bootstrap_ci_lower']
        )
        comparison_df['ci_width_flag'] = comparison_df['bootstrap_ci_width'].apply(
            _ci_width_flag
        )
        comparison_df['overall_adequacy'] = comparison_df['ci_width_flag'].apply(
            lambda f: _worse_flag(f, upfront_severity)
        )

        if self.config.verbose:
            logger.debug("COMPARISON SUMMARY:\n%s", comparison_df.round(4).to_string())

        return comparison_df, all_results


# ---------------------------------------------------------------------------
# Standalone recommendation builder
# ---------------------------------------------------------------------------

def weighted_validation_score(comparison_df: pd.DataFrame) -> pd.Series:
    """
    The composite score the recommendation ranks panels on.

    0.5 x stratified CV + 0.25 x leave-one-out + 0.25 x bootstrap. Exposed so
    anything that reports a step's validation score (nested CV does, per
    outer fold) shows the number the choice was actually made on.
    """
    return (
        0.5 * comparison_df['cv_auc']
        + 0.25 * comparison_df['loo_auc']
        + 0.25 * comparison_df['bootstrap_auc']
    )


def build_recommendation(
    comparison_df: pd.DataFrame,
    step_data: Dict[str, Dict[str, Any]],
    X_test: pd.DataFrame,
    *,
    one_se_tolerance: float = 1.0,
    holdout_scores: Optional[pd.DataFrame] = None,
) -> Optional[Dict[str, Any]]:
    """
    Select the best feature set and assemble the recommendation dict.

    The best feature set is chosen by a weighted composite AUC:
    0.5 x CV + 0.25 x LOO + 0.25 x bootstrap.  This down-weights the
    high-variance LOO and bootstrap estimates while still requiring them
    to corroborate the CV result.

    Quality thresholds: >0.8 weighted AUC with CV stability <0.15 is
    'RECOMMENDED'; >0.7 is 'CAUTION'; otherwise 'NOT RECOMMENDED'.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output of FeatureSetValidator.compare_feature_sets.
    step_data : dict
        Keys match feature set names in comparison_df.  Each value is a
        dict with keys: step_name, X_train (pd.DataFrame), eval_result
        (cv_evaluate_model output dict).
    X_test : pd.DataFrame
        Full test set.  Columns are subset to match the selected feature set.
        Labels are never read here, so the test split plays no part in the
        choice.
    holdout_scores : pd.DataFrame or None
        Per-step scores from folds held out of the training split, indexed
        like comparison_df with columns holdout_auc, holdout_std and
        holdout_folds (SelectOmicsPipeline.rank_steps_on_holdout output).
        When given, the ranking, the tolerance band and the quality label come
        from these instead of from validation on the training split, which
        shares its samples with selection. The selection-optimism guard is
        then off: a gain measured on rows that did not select the panel is
        signal, not optimism.

    Returns
    -------
    dict or None
        Recommendation dict with keys:
        step_id, step_name, X_train, X_test, result, cv_auc, cv_std,
        weighted_auc, n_features, cv_stability, quality, reason.
        Returns None if the best feature set cannot be resolved.
    """
    weighted = weighted_validation_score(comparison_df)

    # What the panels are ranked on. Validation on the training split scores
    # each panel on the samples its features were selected from, so it is
    # optimistic, and most optimistic for the most aggressive step. Held-out
    # folds carry no such bias, so they are used whenever supplied.
    scores = weighted
    ranking_basis = 'validation on the training split'
    ranked_on_holdout = False
    if holdout_scores is not None and 'holdout_auc' in holdout_scores:
        _h = holdout_scores['holdout_auc'].reindex(comparison_df.index)
        if _h.notna().any():
            scores = _h
            ranking_basis = 'held-out folds from the training split'
            ranked_on_holdout = True

    # Guard against an all-NaN ranking (e.g. when every validation method
    # returned NaN because all CV folds had a single class).
    if scores.isna().all():
        return None

    # One-standard-error rule: among the panels whose score is statistically
    # indistinguishable from the best, take the SMALLEST.
    #
    # Ranking on weighted AUC alone picks the least selective step, because a
    # larger panel gives the classifier more to work with and usually scores a
    # shade higher. Measured on synthetic data where the true features are
    # known, plain argmax replaced an 18-feature panel at F1 0.714 with a
    # 620-feature one at F1 0.032, and an 11-feature panel at F1 0.957 with a
    # 62-feature one at F1 0.324. Across 12 runs it never improved on the last
    # step and halved mean F1 twice, because AUC and selection quality pull in
    # opposite directions.
    #
    # The tolerance is one standard error of the best panel's fold scores,
    # which is the usual formulation of this rule. Inside that band the AUC
    # difference is noise, and paying hundreds of features for noise is the
    # failure above.
    _best_raw = scores.idxmax(skipna=True)
    _best_score = float(scores.loc[_best_raw])

    # The tolerance is the WIDEST defensible uncertainty estimate available,
    # floored at MIN_SCORE_TOLERANCE.
    #
    # The floor is what makes this work at n << p. CV AUC saturates there:
    # measured on omics_standard, every panel from 18 features to the full
    # 2000 scored cv_auc 1.0000 with cv_std 0.0000, so a variance-derived band
    # had zero width and the rule collapsed back to argmax. Step 1 then won by
    # 0.0003 -- noise -- and took a 620-feature panel at F1 0.032 over an
    # 18-feature one at F1 0.714.
    #
    # An AUC difference below the floor cannot justify hundreds of extra
    # features, whatever the variance estimates say.
    def _spread(col: str) -> float:
        if col not in comparison_df.columns:
            return 0.0
        v = comparison_df.loc[_best_raw, col]
        return float(v) if pd.notna(v) else 0.0

    _se_cv = _spread('cv_std')
    _folds = _spread('cv_folds')
    if _folds > 0:
        _se_cv /= np.sqrt(_folds)

    # bootstrap_std is already the spread of the resampled estimate, so it is
    # a standard error as it stands; the CI half-width is a second read on it.
    _se_boot = _spread('bootstrap_std')
    _ci_half = 0.0
    if {'bootstrap_ci_lower', 'bootstrap_ci_upper'} <= set(comparison_df.columns):
        _lo, _hi = _spread('bootstrap_ci_lower'), _spread('bootstrap_ci_upper')
        if _hi > _lo:
            _ci_half = (_hi - _lo) / 2.0 / 1.96

    _se = max(_se_cv, _se_boot, _ci_half, MIN_SCORE_TOLERANCE)

    # A held-out ranking carries its own spread: the standard error of the
    # fold means is what one standard error means for these scores.
    if ranked_on_holdout:
        _hsd = float(holdout_scores.loc[_best_raw, 'holdout_std'] or 0.0)
        _hn = float(holdout_scores.loc[_best_raw].get('holdout_folds', 0) or 0)
        _se = max(_hsd / np.sqrt(_hn) if _hn > 0 else _hsd,
                  MIN_SCORE_TOLERANCE)

    _se *= float(one_se_tolerance)

    _within = [
        i for i in comparison_df.index
        if pd.notna(scores.loc[i]) and scores.loc[i] >= _best_score - _se
    ]
    if _within:
        # Smallest panel in the band; higher score breaks a size tie.
        best_id = min(
            _within,
            key=lambda i: (int(comparison_df.loc[i, 'n_features']),
                           -float(scores.loc[i])),
        )
    else:
        best_id = _best_raw

    _parsimony = (best_id != _best_raw)
    best_auc     = float(comparison_df.loc[best_id, 'cv_auc'])
    best_std     = float(comparison_df.loc[best_id, 'cv_std'])
    best_weighted = float(weighted.loc[best_id])
    best_ranked  = float(scores.loc[best_id])
    cv_stab      = best_std / best_auc if best_auc > 0 else float('inf')

    # The label reads the score the choice was made on, so under a held-out
    # ranking it describes performance on rows that did not select the panel.
    # That is what demotes a null-control panel, which validates at 0.84 on
    # the training split and scores 0.46 held out.
    if best_ranked > 0.8 and cv_stab < 0.15:
        quality = 'RECOMMENDED'
        reason  = f'{best_id} is suitable for downstream tasks'
    elif best_ranked > 0.7:
        quality = 'CAUTION'
        reason  = f'{best_id} shows promise but monitor carefully'
    else:
        quality = 'NOT RECOMMENDED'
        reason  = 'Consider using fewer selection steps or more data'

    # Say so when parsimony, not raw score, decided this. Otherwise a reader
    # comparing the verdict against the comparison table sees a panel that did
    # not top it and has no way to know that was deliberate.
    if _parsimony:
        reason += (
            f' (chosen over {_best_raw} on the one-standard-error rule: '
            f'{int(comparison_df.loc[best_id, "n_features"])} features vs '
            f'{int(comparison_df.loc[_best_raw, "n_features"])}, and their '
            f'scores differ by less than one standard error)'
        )

    # Selection-optimism guard. The quality thresholds above are applied to a
    # score that selection has already inflated, so on its own the label can
    # pass noise: a null control scored 0.84 here and was labelled CAUTION or
    # better. The least-selected panel is the unbiased reference. When the
    # recommended panel beats it by more than selection plausibly delivers,
    # downgrade and say why. This changes the label only, never the choice.
    ref_id = comparison_df['n_features'].idxmax()
    ref_score = scores.loc[ref_id]
    selection_gain = (best_ranked - float(ref_score)
                      if pd.notna(ref_score) else float('nan'))
    # Never applied to a held-out ranking: there the gain was measured on rows
    # that selected neither panel, so a large one is signal, and flagging it
    # would be a false alarm.
    optimism_suspected = bool(
        not ranked_on_holdout
        and best_id != ref_id
        and pd.notna(selection_gain)
        and selection_gain > MAX_PLAUSIBLE_SELECTION_GAIN
    )
    if optimism_suspected:
        if quality == 'RECOMMENDED':
            quality = 'CAUTION'
        note = (
            f'{best_id} scores {selection_gain:.3f} above {ref_id}, the '
            f'least-selected panel, on the same samples its features were '
            f'selected from. A gain this large is more often selection fitting '
            f'noise than signal the full panel missed. Confirm it with '
            f'enable_nested_cv=True or the held-out test before relying on it'
        )
        # The generic quality text ('suitable for downstream tasks', 'shows
        # promise') would contradict the note, so it is replaced; a
        # parsimony explanation is kept.
        parsimony_note = reason[reason.index(' (chosen over'):] \
            if _parsimony else ''
        if quality == 'NOT RECOMMENDED':
            reason = f'{reason}. {note}'
        else:
            reason = note + parsimony_note

    if best_id not in step_data:
        return None

    sd = step_data[best_id]
    if sd.get('X_train') is None or sd.get('eval_result') is None:
        return None

    X_tr    = sd['X_train']
    X_te    = X_test[X_tr.columns]
    n_feat  = int(comparison_df.loc[best_id, 'n_features'])

    return {
        'step_id':      best_id,
        'step_name':    sd['step_name'],
        'X_train':      X_tr,
        'X_test':       X_te,
        'result':       sd['eval_result'],
        'cv_auc':       best_auc,
        'cv_std':       best_std,
        'weighted_auc': best_weighted,
        'n_features':   n_feat,
        'cv_stability': cv_stab,
        'quality':      quality,
        'reason':       reason,
        # How far the recommended panel's score exceeds the least-selected
        # panel's, and whether that exceeded MAX_PLAUSIBLE_SELECTION_GAIN.
        'reference_id':                 ref_id,
        'selection_gain':               selection_gain,
        'selection_optimism_suspected': optimism_suspected,
        # What the panels were ranked on, and the winner's score on it.
        'ranking_basis':                ranking_basis,
        'ranking_score':                best_ranked,
        'holdout_auc':                  (float(scores.loc[best_id])
                                         if ranked_on_holdout
                                         else float('nan')),
    }
