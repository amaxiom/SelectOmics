"""
SelectOmics/evaluation/metrics.py

Performance metric computation for the SelectOmics pipeline.

Functions cover per-class detailed metrics (sensitivity, specificity,
optimal Youden-J threshold, 95%/99% specificity thresholds), summary
classification tables, and the comprehensive metrics dict saved at the
end of the pipeline.

All global state from the notebook (n_classes, class_names, y_train,
y_test, SELECTOMICS_CONFIG) is replaced by explicit parameters.
No module-level execution.

Source: notebook Cells 9, 17, and 18.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)


def _safe_roc_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_classes: int,
) -> float:
    """
    Compute ROC AUC safely for both binary and multiclass problems.

    Parameters
    ----------
    y_true : np.ndarray
        Integer-encoded ground truth labels.
    y_proba : np.ndarray, shape (n_samples, n_classes)
        Predicted probabilities.
    n_classes : int
        Total number of classes.

    Returns
    -------
    float
        AUC score, or np.nan if fewer than 2 classes are present in y_true.
    """
    present = np.unique(y_true)
    if len(present) < 2:
        return float('nan')
    y_proba = np.asarray(y_proba)
    if n_classes == 2:
        # sklearn's roc_auc_score (all versions) expects a 1-D positive-class
        # probability for binary problems.  Extract column 1 when a full
        # (n_samples, 2) probability matrix is supplied.
        if y_proba.ndim == 2:
            y_proba = y_proba[:, 1]
        return float(roc_auc_score(y_true, y_proba))
    # Multiclass: OvR macro-averaging requires ALL classes to be present in
    # y_true.  If any class is absent from this fold (e.g. tiny minority class
    # in a small test split), sklearn raises ValueError.  Return nan instead so
    # the caller can skip the fold gracefully rather than catching an exception.
    if len(present) < n_classes:
        return float('nan')
    return float(roc_auc_score(
        y_true, y_proba, multi_class='ovr', average='macro'
    ))


def binarize_labels(
    y: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """
    Convert integer-encoded labels to one-hot binary matrix.

    sklearn's label_binarize returns a single column for binary problems;
    this function always returns shape (n_samples, n_classes).

    Parameters
    ----------
    y : np.ndarray
        Integer-encoded labels.
    n_classes : int
        Total number of classes.

    Returns
    -------
    np.ndarray, shape (n_samples, n_classes)
    """
    raw = label_binarize(y, classes=list(range(n_classes)))
    if raw.shape[1] == 1:
        # Binary case: expand to two-column format.
        expanded = np.zeros((len(y), n_classes))
        for idx, label in enumerate(y):
            expanded[idx, label] = 1
        return expanded
    return raw


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    n_classes: int,
) -> pd.DataFrame:
    """
    Compute per-class sensitivity, specificity, AUC, and decision thresholds.

    For each class three threshold operating points are reported:

    - Default (0.5 cutoff): sensitivity and specificity at the argmax prediction.
    - Optimal (Youden's J): the threshold maximising TPR - FPR.
    - High-specificity: sensitivity achievable at 95% and 99% specificity.

    Parameters
    ----------
    y_true : np.ndarray
        Integer-encoded ground truth.
    y_pred : np.ndarray
        Predicted class labels.
    y_proba : np.ndarray, shape (n_samples, n_classes)
        Predicted probabilities.
    class_names : list of str
        Class labels in label-encoder order.
    n_classes : int
        Total number of classes.

    Returns
    -------
    pd.DataFrame
        One row per class with columns:
        Class, AUC, Default_Sens, Default_Spec,
        Opt_Threshold, Opt_Sens, Opt_Spec,
        Thresh_95Spec, Sens_95Spec,
        Thresh_99Spec, Sens_99Spec.
    """
    y_binary = binarize_labels(y_true, n_classes)
    mcm = multilabel_confusion_matrix(
        y_true, y_pred, labels=list(range(n_classes))
    )

    rows: List[Dict[str, Any]] = []

    for i, class_name in enumerate(class_names):
        tn, fp, fn, tp = mcm[i].ravel()
        sensitivity  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Defaults for absent classes.
        class_auc = float('nan')
        opt_thr = opt_sens = opt_spec = float('nan')
        thr_95 = sens_95 = thr_99 = sens_99 = float('nan')

        if i < y_binary.shape[1] and y_binary[:, i].sum() > 0:
            class_auc = float(roc_auc_score(y_binary[:, i], y_proba[:, i]))
            fpr, tpr, thresholds = roc_curve(y_binary[:, i], y_proba[:, i])

            # Youden's J optimal threshold.
            j = tpr - fpr
            opt_idx = int(np.argmax(j))
            opt_thr  = float(thresholds[opt_idx])
            opt_sens = float(tpr[opt_idx])
            opt_spec = float(1.0 - fpr[opt_idx])

            # Threshold at 95% specificity (FPR <= 0.05).
            thr_95, sens_95 = _threshold_at_specificity(fpr, tpr, thresholds, target_fpr=0.05)

            # Threshold at 99% specificity (FPR <= 0.01).
            thr_99, sens_99 = _threshold_at_specificity(fpr, tpr, thresholds, target_fpr=0.01)

        rows.append({
            'Class':        class_name,
            'AUC':          class_auc,
            'Default_Sens': float(sensitivity),
            'Default_Spec': float(specificity),
            'Opt_Threshold': opt_thr,
            'Opt_Sens':      opt_sens,
            'Opt_Spec':      opt_spec,
            'Thresh_95Spec': thr_95,
            'Sens_95Spec':   sens_95,
            'Thresh_99Spec': thr_99,
            'Sens_99Spec':   sens_99,
        })

    return pd.DataFrame(rows)


def _threshold_at_specificity(
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
    target_fpr: float,
) -> tuple:
    """
    Find the threshold and sensitivity at or above a target specificity.

    Selects the highest-TPR operating point where FPR <= target_fpr
    (i.e. specificity >= 1 - target_fpr).  Using argmin(|fpr - target|)
    can overshoot into FPR > target_fpr, reporting sensitivity at a lower
    specificity than requested.

    Parameters
    ----------
    fpr, tpr, thresholds : np.ndarray
        Outputs of sklearn.metrics.roc_curve.
    target_fpr : float
        Target false-positive rate (1 - specificity).

    Returns
    -------
    threshold : float
    sensitivity : float
    Both are np.nan if the target specificity is not achievable.
    """
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) > 0:
        best = valid[int(np.argmax(tpr[valid]))]
        if best < len(thresholds):
            return float(thresholds[best]), float(tpr[best])
    return float('nan'), float('nan')


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    model_name: str = '',
) -> pd.DataFrame:
    """
    Produce a classification report as a pandas DataFrame.

    Parameters
    ----------
    y_true : np.ndarray
        Integer-encoded ground truth.
    y_pred : np.ndarray
        Predicted labels.
    class_names : list of str
        Class labels in label-encoder order.
    model_name : str
        Used as the index name on the returned DataFrame.

    Returns
    -------
    pd.DataFrame
        Rows are classes plus macro/weighted averages.
    """
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    df = pd.DataFrame(report).T
    df.index.name = model_name
    return df


def build_comprehensive_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    n_classes: int,
    training_result: Dict[str, Any],
    step_label: str,
    algorithm: str,
    n_models: int,
    achieved_agreement: float,
    agreement_label: str,
    original_n_features: int,
    final_n_features: int,
) -> Dict[str, Any]:
    """
    Assemble a flat metrics dict suitable for JSON serialisation.

    Captures all aggregate and per-class metrics reported in the notebook
    Cell 18 comprehensive metrics block.

    Parameters
    ----------
    y_test : np.ndarray
        Integer-encoded test labels.
    y_pred : np.ndarray
        Predicted test labels.
    y_proba : np.ndarray, shape (n_test, n_classes)
        Predicted test probabilities.
    class_names : list of str
        Class labels in encoder order.
    n_classes : int
        Total number of classes.
    training_result : dict
        cv_evaluate_model output for the final training step.
        Must contain 'mean_auc' and 'std_auc'.
    step_label : str
        Human-readable name of the final feature selection step.
    algorithm : str
        Algorithm identifier.
    n_models : int
        Number of consensus models used.
    achieved_agreement : float in [0, 1]
        Fraction of models that had to agree for the final panel to survive.
        This is an OUTCOME, not a setting: the caller states the panel size
        they need via min_features_floor and this records what that cost in
        agreement. 0.0 means the rank-average fallback ran.
    agreement_label : str
        Plain-language reading of achieved_agreement, e.g. 'strong'.
    original_n_features : int
        Feature count before any selection.
    final_n_features : int
        Feature count after final selection step.

    Returns
    -------
    dict
        Flat key-value dict.  Per-class keys follow the pattern
        '{class_name}_{metric}'.  Suitable for json.dump.
    """
    test_auc     = _safe_roc_auc(y_test, y_proba, n_classes)
    train_auc    = training_result['mean_auc']
    train_std    = training_result['std_auc']
    gap          = train_auc - test_auc

    overall_acc  = float(accuracy_score(y_test, y_pred))
    bal_acc      = float(balanced_accuracy_score(y_test, y_pred))

    prec, rec, f1, supp = precision_recall_fscore_support(
        y_test, y_pred, average=None,
        labels=list(range(n_classes)), zero_division=0,
    )
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro', zero_division=0,
    )
    w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0,
    )

    detailed = compute_per_class_metrics(y_test, y_pred, y_proba, class_names, n_classes)

    reduction_pct = float(
        (1.0 - final_n_features / original_n_features) * 100
    ) if original_n_features > 0 else 0.0

    metrics: Dict[str, Any] = {
        'algorithm':              algorithm,
        'n_consensus_models':     int(n_models),
        'achieved_agreement':     float(achieved_agreement),
        'agreement_label':        str(agreement_label),
        'final_step':             step_label,
        'final_features':         int(final_n_features),
        'original_features':      int(original_n_features),
        'feature_reduction_pct':  reduction_pct,
        # Guard nan values: json.dump raises ValueError on float('nan').
        'training_cv_auc_mean':   float(train_auc) if not np.isnan(train_auc) else None,
        'training_cv_auc_std':    float(train_std) if not np.isnan(train_std) else None,
        'test_auc_macro':         float(test_auc) if not np.isnan(test_auc) else None,
        'generalization_gap':     float(gap) if not np.isnan(gap) else None,
        'overall_accuracy':       overall_acc,
        'balanced_accuracy':      bal_acc,
        'macro_precision':        float(macro_prec),
        'macro_recall':           float(macro_rec),
        'macro_f1':               float(macro_f1),
        'weighted_precision':     float(w_prec),
        'weighted_recall':        float(w_rec),
        'weighted_f1':            float(w_f1),
    }

    # Per-class metrics.
    for i, class_name in enumerate(class_names):
        row = detailed.iloc[i]
        metrics[f'{class_name}_precision']    = float(prec[i])
        metrics[f'{class_name}_recall']       = float(rec[i])
        metrics[f'{class_name}_f1']           = float(f1[i])
        metrics[f'{class_name}_support']      = int(supp[i])
        metrics[f'{class_name}_auc']          = (
            float(row['AUC']) if not np.isnan(row['AUC']) else None
        )
        metrics[f'{class_name}_sensitivity']  = float(row['Default_Sens'])
        metrics[f'{class_name}_specificity']  = float(row['Default_Spec'])

        for col, key in [
            ('Opt_Threshold',   'optimal_threshold'),
            ('Opt_Sens',        'optimal_sensitivity'),
            ('Opt_Spec',        'optimal_specificity'),
            ('Thresh_95Spec',   'threshold_95spec'),
            ('Sens_95Spec',     'sensitivity_95spec'),
            ('Thresh_99Spec',   'threshold_99spec'),
            ('Sens_99Spec',     'sensitivity_99spec'),
        ]:
            val = row[col]
            metrics[f'{class_name}_{key}'] = (
                float(val) if not np.isnan(val) else None
            )

    return metrics


def save_comprehensive_metrics(
    metrics: Dict[str, Any],
    output_dir: Path,
    algorithm: str,
) -> Path:
    """
    Serialise the comprehensive metrics dict to a JSON file.

    Parameters
    ----------
    metrics : dict
        Output of build_comprehensive_metrics.
    output_dir : Path
        Destination directory.  Created if absent.
    algorithm : str
        Used in the filename.

    Returns
    -------
    Path
        Path to the written JSON file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / f"final_{algorithm}_comprehensive_metrics.json"
    with open(dest, 'w') as fh:
        json.dump(metrics, fh, indent=2)
    return dest


def summarise_generalization(
    training_auc: float,
    test_auc: float,
    verbose: bool = True,
) -> str:
    """
    Classify the generalization gap and return a text label.

    Parameters
    ----------
    training_auc : float
    test_auc : float
    verbose : bool
        If True, prints the assessment.

    Returns
    -------
    str
        One of: 'excellent', 'good', 'moderate', 'poor'.
    """
    gap = abs(training_auc - test_auc)
    if gap < 0.05:
        label = 'excellent'
    elif gap < 0.10:
        label = 'good'
    elif gap < 0.15:
        label = 'moderate'
    else:
        label = 'poor'

    # A nan gap fails every comparison above and lands on 'poor', which reads
    # as "generalises badly" when in fact nothing was measured. The label set
    # is part of this function's contract and is left alone; the silence is
    # not, because 'poor' is the one label a caller acts on.
    if np.isnan(gap):
        logger.warning(
            "  Generalization gap could not be computed (training AUC %s, "
            "test AUC %s). The '%s' label below is the fallback for an "
            "uncomputable gap, not a measurement: treat this run as having "
            "no held-out result.",
            training_auc, test_auc, label,
        )
    elif verbose:
        logger.info("  Generalization gap: %+.4f (%s)", training_auc - test_auc, label)

    return label


def build_pipeline_summary_df(
    step_results: List[Dict[str, Any]],
    original_n_features: int,
    algorithm: str,
) -> pd.DataFrame:
    """
    Build the pipeline-level summary DataFrame shown in compare_feature_sets.

    Parameters
    ----------
    step_results : list of dict
        Each dict has keys: step_name, n_features, eval_result.
        eval_result is a cv_evaluate_model output dict or None.
    original_n_features : int
        Feature count at Step 0 (reference).
    algorithm : str
        Algorithm identifier, used for the AUC column name.

    Returns
    -------
    pd.DataFrame
        Columns: Step, Features, Features_Removed, {algorithm}_CV_AUC,
        Cumulative_Reduction, Reduction_Percentage.
    """
    records = []
    prev_count = original_n_features

    for sr in step_results:
        n_feat  = sr['n_features']
        removed = prev_count - n_feat
        auc     = sr['eval_result']['mean_auc'] if sr['eval_result'] else float('nan')
        records.append({
            'Step':             sr['step_name'],
            'Features':         n_feat,
            'Features_Removed': removed,
            f'{algorithm}_CV_AUC': auc,
        })
        prev_count = n_feat

    df = pd.DataFrame(records)
    df['Cumulative_Reduction']  = original_n_features - df['Features']
    df['Reduction_Percentage']  = 100.0 * df['Cumulative_Reduction'] / original_n_features
    return df
