"""
SelectOmics/data/__init__.py

Public interface for the SelectOmics data layer.

Exposes data loading and preprocessing functions used by the pipeline
and available to users who need direct access to individual components.
"""

from .loaders import (
    load_omics_data,
    save_from_config,
    save_results_multiformat,
)
from .preprocessing import (
    create_train_test_split,
    handle_nans,
    prepare_data,
)

__all__ = [
    # loaders
    "load_omics_data",
    "save_from_config",
    "save_results_multiformat",
    # preprocessing
    "create_train_test_split",
    "handle_nans",
    "prepare_data",
]
