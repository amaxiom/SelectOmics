"""
SelectOmics/models/__init__.py

Public interface for the SelectOmics model layer.

Exposes pipeline factory functions and evaluation utilities used by the
selection steps and available to users who need direct access to
individual components.
"""

from .base import (
    build_consensus_pipeline,
    build_consensus_pipelines,
    create_lr_pipeline_consensus,
    create_rf_pipeline_consensus,
    create_svm_pipeline_consensus,
    create_xgb_pipeline_consensus,
    quick_tune_algorithm,
    quick_tune_all,
)
from .evaluation import (
    cv_evaluate_model,
    evaluate_models_collection,
)

__all__ = [
    # base
    "build_consensus_pipeline",
    "build_consensus_pipelines",
    "create_lr_pipeline_consensus",
    "create_rf_pipeline_consensus",
    "create_svm_pipeline_consensus",
    "create_xgb_pipeline_consensus",
    "quick_tune_algorithm",
    "quick_tune_all",
    # evaluation
    "cv_evaluate_model",
    "evaluate_models_collection",
]
