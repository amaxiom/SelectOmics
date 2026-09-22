"""
SelectOmics/evaluation/visualization.py

Matplotlib figure generation for the SelectOmics pipeline.

All functions accept explicit parameters and a SelectOmicsConfig object
for plot format and verbosity.  No global state is read.  The module-
level colormap reference from the notebook (tab10 = plt.get_cmap(...))
is moved inside each function that needs it.

_save_figure handles format detection, extension replacement, and
format-specific savefig kwargs.  Callers supply a stem path; the
correct extension is applied automatically based on config.plot_format.

Source: notebook Cells 9, 16, 17, and 18.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import label_binarize

from SelectOmics.config import SelectOmicsConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plot format kwargs
# ---------------------------------------------------------------------------

_PLOT_FORMAT_KWARGS: Dict[str, Dict[str, Any]] = {
    'png':  {'dpi': 300, 'bbox_inches': 'tight'},
    'svg':  {'dpi': 300, 'bbox_inches': 'tight'},
    'pdf':  {'dpi': 300, 'bbox_inches': 'tight'},
    'tiff': {'dpi': 300, 'bbox_inches': 'tight'},
    # JPEG quality is passed through Pillow via pil_kwargs.  The older
    # savefig(quality=...) argument was removed in Matplotlib 3.3, so passing
    # it directly raises TypeError on modern Matplotlib and breaks every save
    # when plot_format='jpeg'.
    'jpeg': {'dpi': 300, 'bbox_inches': 'tight', 'pil_kwargs': {'quality': 95}},
}


# ---------------------------------------------------------------------------
# Figure typography
# ---------------------------------------------------------------------------
# Larger-than-matplotlib-default font sizes so that figures remain legible when
# embedded and downscaled in documents and slides.  Applied at the start of
# each plotting function rather than at import time, so importing the package
# does not silently mutate a caller's global matplotlib state.

_FONT_SIZES: Dict[str, float] = {
    'font.size':        13.0,
    'axes.titlesize':   16.0,
    'axes.labelsize':   14.0,
    'xtick.labelsize':  12.0,
    'ytick.labelsize':  12.0,
    'legend.fontsize':  12.0,
    'figure.titlesize': 18.0,
}


def _with_plot_style(fn):
    """
    Run a plotting function with the SelectOmics fonts, then restore them.

    plt.rcParams.update() is a permanent, process-global mutation. Applying it
    per plotting function rather than at import stopped a bare `import
    SelectOmics` from changing a caller's matplotlib, but it still left every
    subsequent plot the caller drew in their own session using this package's
    font sizes, with nothing to signal why. rc_context scopes the change to
    the call, which is what the comment above intended.

    Font sizes are baked into Text artists when they are created, so figures
    built inside the context keep their sizing even if they are saved or shown
    after it exits.

    Figures are closed on the way out as well. Each plotting function closes
    its own figure on the happy path, but those calls sit in the function body,
    not in a finally, so anything raised in between (matplotlib rejects all-NaN
    inputs, for one) left the figure open. Every one of these functions returns
    None and hands back no figure, so closing whatever the call left behind is
    always safe, and it keeps a repeated failure in a notebook session from
    accumulating figures until matplotlib starts warning about the count.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        pre_existing = set(plt.get_fignums())
        try:
            with plt.rc_context(_FONT_SIZES):
                return fn(*args, **kwargs)
        finally:
            for num in set(plt.get_fignums()) - pre_existing:
                plt.close(num)
    return wrapper


# Figures built from an explicit GridSpec (or carrying a colorbar) are not
# compatible with tight_layout: Matplotlib emits a UserWarning on every save and
# discards the GridSpec's wspace.  Those figures set their own spacing, and the
# outer margins are trimmed at save time by bbox_inches='tight' in
# _PLOT_FORMAT_KWARGS, so tight_layout has nothing left to contribute.


def _save_figure(
    save_path: Optional[Path],
    config: SelectOmicsConfig,
) -> None:
    """
    Save the current matplotlib figure using the format in config.plot_format.

    The file extension of save_path is replaced with the configured format.
    If plot_format is unrecognised, PNG is used as a fallback.

    Parameters
    ----------
    save_path : Path or None
        Destination path.  If None the figure is not saved.
    config : SelectOmicsConfig
    """
    if save_path is None:
        return

    fmt = config.plot_format.lower().strip()
    if fmt not in _PLOT_FORMAT_KWARGS:
        fmt = 'png'

    dest = Path(save_path).with_suffix(f'.{fmt}')
    dest.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(dest, **_PLOT_FORMAT_KWARGS[fmt])


# ---------------------------------------------------------------------------
# ROC curve plot
# ---------------------------------------------------------------------------

@_with_plot_style
def plot_roc_curves(
    models_dict: Dict[str, Dict[str, Any]],
    y_true: np.ndarray,
    n_classes: int,
    classes: np.ndarray,
    config: SelectOmicsConfig,
    title: str = 'ROC Curves',
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot OOF ROC curves for a collection of evaluated models.

    For binary problems the positive-class probability column is used.
    For multiclass, macro-averaged ROC is computed by linear interpolation
    across all classes that appear in y_true.

    A scatter marker is placed at 95% specificity (FPR = 0.05) on each curve.

    Parameters
    ----------
    models_dict : dict
        Keyed by model name; values are cv_evaluate_model result dicts
        (must contain 'oof_proba').
    y_true : np.ndarray
        Integer-encoded labels (training set, aligned to oof_proba).
    n_classes : int
    classes : np.ndarray
        Integer class indices.
    config : SelectOmicsConfig
    title : str
    save_path : Path or None
        Stem path; extension is replaced by config.plot_format.
    """
    from SelectOmics.utils.helpers import compute_macro_auc_ovr, ensure_binary_proba

    plt.figure(figsize=(6, 5))
    _n_models = max(len(models_dict) - 1, 1)

    for idx, (name, result) in enumerate(models_dict.items()):
        _color = plt.cm.viridis(0.1 + 0.8 * idx / _n_models)
        y_proba = ensure_binary_proba(result['oof_proba'], n_classes)
        auc = compute_macro_auc_ovr(y_true, y_proba, classes)

        if n_classes == 2:
            fpr, tpr, _ = roc_curve((y_true == 1).astype(int), y_proba[:, 1])
        else:
            y_bin = label_binarize(y_true, classes=list(range(n_classes)))
            all_fpr = np.linspace(0, 1, 100)
            mean_tpr = np.zeros_like(all_fpr)
            _n_valid_cls = 0
            for c in range(n_classes):
                if y_bin[:, c].sum() > 0:
                    c_fpr, c_tpr, _ = roc_curve(y_bin[:, c], y_proba[:, c])
                    mean_tpr += np.interp(
                        all_fpr,
                        np.concatenate([[0], c_fpr, [1]]),
                        np.concatenate([[0], c_tpr, [1]]),
                    )
                    _n_valid_cls += 1
            mean_tpr /= max(_n_valid_cls, 1)
            mean_tpr[0] = 0.0
            fpr, tpr = all_fpr, mean_tpr

        idx_95 = int(np.argmin(np.abs(fpr - 0.05)))
        plt.plot(fpr, tpr, color=_color, label=f'{name} (AUC={auc:.3f})')
        plt.scatter([fpr[idx_95]], [tpr[idx_95]], marker='o', s=100,
                    color=_color, zorder=5)

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{title}\n(markers at 95% specificity)')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.tight_layout()
    _save_figure(save_path, config)
    plt.close()


# ---------------------------------------------------------------------------
# AUC box plots
# ---------------------------------------------------------------------------

@_with_plot_style
def plot_auc_boxplots(
    step_name: str,
    aucs_dict: Dict[str, List[float]],
    config: SelectOmicsConfig,
    save_path: Optional[Path] = None,
) -> None:
    """
    Box plots of per-fold AUC distributions across models.

    Parameters
    ----------
    step_name : str
        Figure title.
    aucs_dict : dict
        Keys are model names; values are lists of per-fold AUC scores.
    config : SelectOmicsConfig
    save_path : Path or None
    """
    plt.figure(figsize=(6, 5))
    data   = list(aucs_dict.values())
    labels = list(aucs_dict.keys())
    # Matplotlib 3.9 renamed the boxplot 'labels' parameter to 'tick_labels'
    # and will drop 'labels' in 3.11.  Prefer the new name, falling back to
    # the old one on Matplotlib < 3.9.
    try:
        bp = plt.boxplot(
            data, tick_labels=labels, patch_artist=True, showmeans=True
        )
    except TypeError:
        bp = plt.boxplot(
            data, labels=labels, patch_artist=True, showmeans=True
        )

    for patch   in bp['boxes']:
        patch.set_facecolor('white')
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
    for whisker in bp['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1.2)
    for cap     in bp['caps']:
        cap.set_color('black')
        cap.set_linewidth(1.2)
    for median  in bp['medians']:
        median.set_color(plt.cm.viridis(0.5))
        median.set_linewidth(1.5)
    for flier   in bp['fliers']:
        flier.set_markeredgecolor('black')
        flier.set_markerfacecolor('white')

    plt.ylabel('Macro ROC-AUC (OVR)')
    plt.title(step_name)
    plt.grid(True, alpha=0.3)
    if len(labels) > 5:
        plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    _save_figure(save_path, config)
    plt.close()


# ---------------------------------------------------------------------------
# Pipeline summary (feature count + CV AUC progression)
# ---------------------------------------------------------------------------

@_with_plot_style
def plot_pipeline_summary(
    summary_df: pd.DataFrame,
    algorithm: str,
    config: SelectOmicsConfig,
    save_path: Optional[Path] = None,
) -> None:
    """
    Two-panel plot: feature count progression with CV AUC overlay (line and bar).

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output of metrics.build_pipeline_summary_df.
        Must contain columns Step, Features, {algorithm}_CV_AUC.
    algorithm : str
    config : SelectOmicsConfig
    save_path : Path or None
    """
    primary = plt.cm.viridis(0.5)
    auc_col = f'{algorithm}_CV_AUC'

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Line plot with twin axes.
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    l1 = ax1.plot(summary_df['Step'], summary_df['Features'], 'o-',
                  color=primary, linewidth=2, markersize=8, label='Feature Count')
    l2 = ax1_twin.plot(summary_df['Step'], summary_df[auc_col], 's-',
                       color='black', linewidth=2, markersize=6, label=f'{algorithm} CV AUC')
    ax1.set_xlabel('Pipeline Step')
    ax1.set_ylabel('Number of Features', color=primary)
    ax1.tick_params(axis='y', labelcolor=primary)
    ax1_twin.set_ylabel('ROC-AUC Score', color='black')
    ax1_twin.tick_params(axis='y', labelcolor='black')
    ax1_twin.set_ylim(0, 1.1)
    ax1.set_title(f'{algorithm} Feature Count vs CV Performance')
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    lines = l1 + l2
    ax1.legend(lines, [l.get_label() for l in lines], loc='center right')

    # Panel 2: Bar chart with CV AUC overlay.
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    ax2.bar(summary_df['Step'], summary_df['Features'],
            alpha=0.7, color=primary, label='Features')
    ax2_twin.plot(summary_df['Step'], summary_df[auc_col], 'ko-',
                  label=f'{algorithm} CV AUC', markersize=8, linewidth=2)
    ax2.set_xlabel('Pipeline Step')
    ax2.set_ylabel('Number of Features', color=primary)
    ax2.tick_params(axis='y', labelcolor=primary)
    ax2_twin.set_ylabel('ROC-AUC Score', color='black')
    ax2_twin.set_ylim(0, 1.1)
    ax2.set_title(f'{algorithm} Features vs Performance Trade-off')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    _save_figure(save_path, config)
    plt.close()


# ---------------------------------------------------------------------------
# Learning curve + training vs test ROC
# ---------------------------------------------------------------------------

@_with_plot_style
def plot_learning_curve_and_roc(
    final_model: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    test_proba: np.ndarray,
    test_auc: float,
    training_result: Dict[str, Any],
    algorithm: str,
    n_classes: int,
    config: SelectOmicsConfig,
    save_path: Optional[Path] = None,
) -> None:
    """
    Two-panel figure: learning curve (left) and training vs test ROC (right).

    Parameters
    ----------
    final_model : fitted Pipeline
        The model trained on the final feature set.
    X_train : pd.DataFrame
        Final training features.
    y_train : np.ndarray
        Training labels.
    y_test : np.ndarray
        Test labels.
    test_proba : np.ndarray, shape (n_test, n_classes)
        Test set predicted probabilities.
    test_auc : float
        Macro OVR AUC on the test set.
    training_result : dict
        cv_evaluate_model output for the final training step.
    algorithm : str
    n_classes : int
    config : SelectOmicsConfig
    save_path : Path or None
    """
    _train_color = plt.cm.viridis(0.2)
    _test_color  = plt.cm.viridis(0.7)
    fig = plt.figure(figsize=(12, 5))
    gs  = fig.add_gridspec(1, 2, wspace=0.3)

    # ---- Panel 1: Learning curve ----
    ax1 = fig.add_subplot(gs[0, 0])
    cv_folds = max(2, min(5, int(np.bincount(y_train, minlength=n_classes).min())))
    train_sizes, train_sc, test_sc = learning_curve(
        final_model, X_train, y_train,
        cv=cv_folds,
        n_jobs=1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='roc_auc_ovr',
        random_state=config.random_seed,
    )
    tr_mean = train_sc.mean(axis=1)
    tr_std  = train_sc.std(axis=1)
    cv_mean = test_sc.mean(axis=1)
    cv_std  = test_sc.std(axis=1)

    # sklearn scores a fold it could not fit as nan rather than raising, so a
    # training size whose subsample held a single class arrives here as nan
    # and is drawn as a gap. That is the right thing to draw, but it left the
    # curve starting partway along its own x axis with nothing to say why.
    #
    # The smallest sizes are the ones at risk: 10% of a small training split
    # is a handful of rows, and LogisticRegression and SVC both refuse to fit
    # a single class. Values and figure are unchanged; only the silence is.
    _lost = int(np.isnan(cv_mean).sum())
    if _lost:
        _sizes = [int(t) for t, m in zip(train_sizes, cv_mean) if np.isnan(m)]
        logger.warning(
            "  Learning curve: %d of %d training sizes could not be scored "
            "(%s examples). At these sizes a cross-validation subsample can "
            "contain a single class, which %s cannot fit, so those points are "
            "absent from the figure rather than zero.",
            _lost, len(cv_mean), _sizes, algorithm,
        )

    ax1.fill_between(train_sizes, tr_mean - tr_std, tr_mean + tr_std,
                     alpha=0.1, color=_train_color)
    ax1.fill_between(train_sizes, cv_mean - cv_std, cv_mean + cv_std,
                     alpha=0.1, color=_test_color)
    ax1.plot(train_sizes, tr_mean, 'o-', color=_train_color,
             linewidth=2, label='Training score')
    ax1.plot(train_sizes, cv_mean, 'o-', color=_test_color,
             linewidth=2, label='Cross-validation score')
    ax1.set_xlabel('Training Examples', fontsize=12)
    ax1.set_ylabel('ROC AUC Score', fontsize=12)
    ax1.set_title(f'Learning Curve\n{algorithm} ({X_train.shape[1]} features)', fontsize=12)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # ---- Panel 2: Training (OOF) vs Test ROC ----
    ax2 = fig.add_subplot(gs[0, 1])
    training_cv_auc = training_result['mean_auc']

    # Training OOF ROC.
    if 'oof_proba' in training_result:
        oof = training_result['oof_proba']
        if n_classes == 2:
            fpr_tr, tpr_tr, _ = roc_curve((y_train == 1).astype(int), oof[:, 1])
        else:
            y_bin_tr = label_binarize(y_train, classes=list(range(n_classes)))
            all_fpr = np.linspace(0, 1, 100)
            mean_tpr = np.zeros_like(all_fpr)
            _n_valid_cls = 0
            for c in range(n_classes):
                if y_bin_tr[:, c].sum() > 0:
                    cf, ct, _ = roc_curve(y_bin_tr[:, c], oof[:, c])
                    mean_tpr += np.interp(all_fpr,
                                          np.concatenate([[0], cf, [1]]),
                                          np.concatenate([[0], ct, [1]]))
                    _n_valid_cls += 1
            mean_tpr /= max(_n_valid_cls, 1)
            mean_tpr[0] = 0.0
            fpr_tr, tpr_tr = all_fpr, mean_tpr
        ax2.plot(fpr_tr, tpr_tr, color=_train_color, linewidth=2,
                 label=f'Training CV (AUC={training_cv_auc:.3f})')

    # Test ROC.
    if n_classes == 2:
        fpr_te, tpr_te, _ = roc_curve((y_test == 1).astype(int), test_proba[:, 1])
    else:
        y_bin_te = label_binarize(y_test, classes=list(range(n_classes)))
        all_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(all_fpr)
        _n_valid_cls = 0
        for c in range(n_classes):
            if y_bin_te[:, c].sum() > 0:
                cf, ct, _ = roc_curve(y_bin_te[:, c], test_proba[:, c])
                mean_tpr += np.interp(all_fpr,
                                      np.concatenate([[0], cf, [1]]),
                                      np.concatenate([[0], ct, [1]]))
                _n_valid_cls += 1
        mean_tpr /= max(_n_valid_cls, 1)
        mean_tpr[0] = 0.0
        fpr_te, tpr_te = all_fpr, mean_tpr

    idx_95 = int(np.argmin(np.abs(fpr_te - 0.05)))
    ax2.plot(fpr_te, tpr_te, color=_test_color, linewidth=2,
             label=f'Test (AUC={test_auc:.3f})')
    ax2.scatter([fpr_te[idx_95]], [tpr_te[idx_95]], s=100, color=_test_color,
                zorder=5, alpha=0.8, marker='o', label='95% Specificity')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax2.set_xlabel('False Positive Rate', fontsize=11)
    ax2.set_ylabel('True Positive Rate', fontsize=11)
    ax2.set_title(f'ROC Curve: Training vs Test\n{algorithm} Model', fontsize=12)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])

    # tight_layout omitted: see the note beside _apply_plot_style.
    _save_figure(save_path, config)
    plt.close()


# ---------------------------------------------------------------------------
# Per-class ROC + confusion matrix
# ---------------------------------------------------------------------------

@_with_plot_style
def plot_per_class_roc_and_confusion(
    y_test: np.ndarray,
    y_test_binary: np.ndarray,
    test_proba: np.ndarray,
    cm_df: pd.DataFrame,
    class_names: List[str],
    algorithm: str,
    config: SelectOmicsConfig,
    save_path: Optional[Path] = None,
) -> None:
    """
    Two-panel figure: per-class ROC curves (left) and confusion matrix (right).

    Parameters
    ----------
    y_test : np.ndarray
        Integer-encoded test labels.
    y_test_binary : np.ndarray, shape (n_test, n_classes)
        One-hot binarized labels.  Use metrics.binarize_labels.
    test_proba : np.ndarray, shape (n_test, n_classes)
    cm_df : pd.DataFrame
        Confusion matrix with class_names as index and columns.
    class_names : list of str
    algorithm : str
    config : SelectOmicsConfig
    save_path : Path or None
    """
    try:
        import seaborn as sns
    except ImportError:
        sns = None

    _n_cls      = max(len(class_names), 1)
    _cls_colors = plt.cm.viridis(np.linspace(0.1, 0.9, _n_cls))
    fig = plt.figure(figsize=(12, 5))
    gs  = fig.add_gridspec(1, 2, wspace=0.3)

    # Per-class ROC.
    ax_roc = fig.add_subplot(gs[0, 0])
    for i, class_name in enumerate(class_names):
        if y_test_binary[:, i].sum() > 0:
            fpr, tpr, _ = roc_curve(y_test_binary[:, i], test_proba[:, i])
            auc = float(roc_auc_score(y_test_binary[:, i], test_proba[:, i]))
            ax_roc.plot(fpr, tpr, linewidth=2,
                        label=f'{class_name} (AUC={auc:.3f})',
                        color=_cls_colors[i])
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax_roc.set_xlabel('False Positive Rate', fontsize=12)
    ax_roc.set_ylabel('True Positive Rate', fontsize=12)
    ax_roc.set_title(f'Per-Class ROC Curves\n{algorithm} Model', fontsize=12)
    ax_roc.legend(loc='lower right')
    ax_roc.grid(True, alpha=0.3)

    # Confusion matrix.
    ax_cm = fig.add_subplot(gs[0, 1])
    if sns is not None:
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='viridis', ax=ax_cm,
                    cbar_kws={'label': 'Count'},
                    linewidths=1, linecolor='gray',
                    square=True, vmin=0)
    else:
        # Fallback if seaborn is unavailable.
        ax_cm.imshow(cm_df.values, cmap='viridis', aspect='auto')
        for r in range(cm_df.shape[0]):
            for c in range(cm_df.shape[1]):
                ax_cm.text(c, r, str(cm_df.values[r, c]),
                           ha='center', va='center')
        ax_cm.set_xticks(range(len(class_names)))
        ax_cm.set_yticks(range(len(class_names)))
        ax_cm.set_xticklabels(class_names)
        ax_cm.set_yticklabels(class_names)

    ax_cm.set_xlabel('Predicted Class', fontsize=12)
    ax_cm.set_ylabel('True Class', fontsize=12)
    ax_cm.set_title(
        f'Confusion Matrix - Test Set\n{algorithm} Model (n={len(y_test)} samples)',
        fontsize=12, pad=15,
    )
    ax_cm.set_xticklabels(class_names, rotation=45, ha='right')
    ax_cm.set_yticklabels(class_names, rotation=0)

    # tight_layout omitted: see the note beside _apply_plot_style.
    _save_figure(save_path, config)
    plt.close()


# ---------------------------------------------------------------------------
# Validation comparison (Cell 16)
# ---------------------------------------------------------------------------

@_with_plot_style
def plot_validation_comparison(
    all_results: Dict[str, Dict[str, Any]],
    config: SelectOmicsConfig,
    save_path: Optional[Path] = None,
) -> None:
    """
    Four-panel validation comparison figure (Cell 16 compare_feature_sets).

    Panel 1: Grouped bar chart comparing CV, LOO, and Bootstrap AUC.
    Panel 2: Scatter of feature count vs CV AUC.
    Panel 3: Bootstrap AUC with 95% CI error bars.
    Panel 4: Stability (CV coefficient of variation) per feature set.

    Parameters
    ----------
    all_results : dict
        Keys: feature set names.
        Values: validate_feature_set result dicts (must contain
        'stratified_cv', 'leave_one_out', 'bootstrap', 'n_features').
    config : SelectOmicsConfig
    save_path : Path or None
    """
    feature_names = list(all_results.keys())
    n_sets = len(feature_names)
    x = np.arange(n_sets)
    width = 0.25
    _method_colors = plt.cm.viridis(np.linspace(0.2, 0.8, 3))

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Panel 1: Grouped bar by method.
    for i, method in enumerate(['CV', 'LOO', 'Bootstrap']):
        if method == 'CV':
            means = [r['stratified_cv']['mean_auc'] for r in all_results.values()]
            stds  = [r['stratified_cv']['std_auc']  for r in all_results.values()]
        elif method == 'LOO':
            means = [r['leave_one_out']['mean_auc'] for r in all_results.values()]
            stds  = [r['leave_one_out']['std_auc']  for r in all_results.values()]
        else:
            means = [r['bootstrap']['mean_auc'] for r in all_results.values()]
            stds  = [r['bootstrap']['std_auc']  for r in all_results.values()]
        axes[0, 0].bar(x + i * width, means, width, yerr=stds,
                       label=method, alpha=0.8, color=_method_colors[i])
    axes[0, 0].set_xlabel('Feature Sets')
    axes[0, 0].set_ylabel('AUC')
    axes[0, 0].set_title('Validation Method Comparison')
    axes[0, 0].set_xticks(x + width)
    axes[0, 0].set_xticklabels(feature_names, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Panel 2: Feature count vs CV AUC scatter.
    n_features = [r['n_features']              for r in all_results.values()]
    cv_aucs    = [r['stratified_cv']['mean_auc'] for r in all_results.values()]
    axes[0, 1].scatter(n_features, cv_aucs, s=100, alpha=0.7, color=plt.cm.viridis(0.5))
    for i, name in enumerate(feature_names):
        axes[0, 1].annotate(name, (n_features[i], cv_aucs[i]),
                            xytext=(5, 5), textcoords='offset points')
    axes[0, 1].set_xlabel('Number of Features')
    axes[0, 1].set_ylabel('CV AUC')
    axes[0, 1].set_title('Features vs Performance')
    axes[0, 1].grid(True, alpha=0.3)

    # Panel 3: Bootstrap CI.
    ci_lower = [r['bootstrap']['confidence_interval'][0] for r in all_results.values()]
    ci_upper = [r['bootstrap']['confidence_interval'][1] for r in all_results.values()]
    boot_means = [r['bootstrap']['mean_auc'] for r in all_results.values()]
    boot_means_arr = np.array(boot_means)
    axes[1, 0].bar(
        np.arange(n_sets), boot_means,
        yerr=[boot_means_arr - np.array(ci_lower),
              np.array(ci_upper) - boot_means_arr],
        alpha=0.7, capsize=5, color=plt.cm.viridis(0.5),
    )
    axes[1, 0].set_xticks(np.arange(n_sets))
    axes[1, 0].set_xticklabels(feature_names, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Bootstrap AUC (95% CI)')
    axes[1, 0].set_title('Bootstrap Confidence Intervals')
    axes[1, 0].set_ylim(0, 1.1)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Panel 4: Stability (CV std / mean AUC -- coefficient of variation).
    # Guard against division by zero when mean_auc is 0.
    stability = [
        (r['stratified_cv']['std_auc'] / r['stratified_cv']['mean_auc']
         if r['stratified_cv']['mean_auc'] > 0 else 0.0)
        for r in all_results.values()
    ]
    axes[1, 1].bar(feature_names, stability, alpha=0.7, color=plt.cm.viridis(0.5))
    axes[1, 1].set_ylabel('CV Coefficient of Variation')
    axes[1, 1].set_title('Stability (lower = more stable)')
    axes[1, 1].set_xticks(range(len(feature_names)))
    axes[1, 1].set_xticklabels(feature_names, rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    _save_figure(save_path, config)
    plt.close()
