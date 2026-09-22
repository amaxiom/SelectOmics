"""
SelectOmics/utils/__init__.py

Public interface for the SelectOmics utility layer.

Exposes state management classes, shared computation helpers, and the
sample size adequacy diagnostic used throughout the pipeline.
"""

from .diagnostics import assess_feature_redundancy, assess_sample_size_adequacy
from .helpers import (
    _binary_search_percentile,
    _warn_step_adequacy,
    agreement_label,
    compute_macro_auc_ovr,
    compute_svm_afi,
    ensure_binary_proba,
    print_adequacy_warning,
    save_step_summary,
)
from .state import DataStateTracker, PipelineState

__all__ = [
    # diagnostics
    "assess_feature_redundancy",
    "assess_sample_size_adequacy",
    # helpers
    "_binary_search_percentile",
    "_warn_step_adequacy",
    "agreement_label",
    "compute_macro_auc_ovr",
    "compute_svm_afi",
    "ensure_binary_proba",
    "print_adequacy_warning",
    "save_step_summary",
    # state
    "DataStateTracker",
    "PipelineState",
]
