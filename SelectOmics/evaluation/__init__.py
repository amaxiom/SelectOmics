"""
SelectOmics/evaluation/__init__.py

Public interface for the SelectOmics evaluation layer.

Exposes metric computation, visualisation, and feature-set validation
functions used by the pipeline and available to users who need direct
access to individual components.
"""

from .metrics import (
    _safe_roc_auc,  # noqa: F401  re-exported for internal callers
    binarize_labels,
    build_comprehensive_metrics,
    build_pipeline_summary_df,
    compute_per_class_metrics,
    generate_classification_report,
    save_comprehensive_metrics,
    summarise_generalization,
)
from .validation import (
    FeatureSetValidator,
    build_recommendation,
)
from .visualization import (
    _save_figure,  # noqa: F401  re-exported for internal callers
    plot_auc_boxplots,
    plot_learning_curve_and_roc,
    plot_per_class_roc_and_confusion,
    plot_pipeline_summary,
    plot_roc_curves,
    plot_validation_comparison,
)

__all__ = [
    # metrics
    "binarize_labels",
    "build_comprehensive_metrics",
    "build_pipeline_summary_df",
    "compute_per_class_metrics",
    "generate_classification_report",
    "save_comprehensive_metrics",
    "summarise_generalization",
    # validation
    "FeatureSetValidator",
    "build_recommendation",
    # visualization
    "plot_auc_boxplots",
    "plot_learning_curve_and_roc",
    "plot_per_class_roc_and_confusion",
    "plot_pipeline_summary",
    "plot_roc_curves",
    "plot_validation_comparison",
]
