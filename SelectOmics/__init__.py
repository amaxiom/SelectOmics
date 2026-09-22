"""
SelectOmics/__init__.py

Public API for the SelectOmics package.

Exposes the pipeline entry point, configuration dataclass, key utilities,
and package-level constants.  All imports are lazy-guarded so that
import errors in optional sub-modules produce clear messages rather than
silent AttributeErrors at call time.

Usage
-----
    import SelectOmics as selectomics

    config = {
        'data_path': 'data.csv',
        'target_column': 'Class',
    }
    pipeline = selectomics.SelectOmicsPipeline(config)
    results  = pipeline.run()
"""

from __future__ import annotations

import logging
from typing import Optional, Union

logging.getLogger(__name__).addHandler(logging.NullHandler())


def enable_logging(
    level: Union[int, str] = logging.INFO,
    handler: Optional[logging.Handler] = None,
    fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
) -> None:
    """
    Configure the SelectOmics logger for console output.

    By default the SelectOmics logger has a NullHandler so that libraries
    which import SelectOmics are not polluted with unexpected console output.
    Call this function once in your application or notebook to activate
    human-readable logging.

    Parameters
    ----------
    level : int or str
        Logging level.  Accepts integer constants (e.g. ``logging.DEBUG``,
        ``logging.INFO``) or string names (e.g. ``'DEBUG'``, ``'WARNING'``).
        Defaults to ``logging.INFO``.
    handler : logging.Handler or None
        Handler to attach.  Defaults to a StreamHandler writing to stderr.
    fmt : str
        Log message format string passed to logging.Formatter.

    Examples
    --------
    >>> import SelectOmics
    >>> SelectOmics.enable_logging()                    # INFO to stderr
    >>> SelectOmics.enable_logging(logging.DEBUG)       # verbose debug output
    >>> SelectOmics.enable_logging('WARNING')           # string form also accepted
    """
    root_logger = logging.getLogger(__name__)
    root_logger.setLevel(level)

    if handler is None:
        handler = logging.StreamHandler()

    handler.setLevel(level)
    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)

    # On repeated calls, update the existing handler of the same type rather
    # than silently leaving its stale level in place.  Without this, calling
    # enable_logging('DEBUG') after enable_logging('WARNING') would update the
    # root-logger level but leave the attached StreamHandler at WARNING,
    # continuing to suppress DEBUG/INFO messages.
    existing = next(
        (h for h in root_logger.handlers if isinstance(h, type(handler))),
        None,
    )
    if existing is not None:
        existing.setLevel(level)
        existing.setFormatter(formatter)
    else:
        root_logger.addHandler(handler)

# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------

__version__: str = '0.7.1'
__author__:  str = 'Amanda S Barnard'

# ---------------------------------------------------------------------------
# Package-level constants
# ---------------------------------------------------------------------------

ALGORITHMS: list = ['LR', 'XGB', 'RF', 'SVM']

SUPPORTED_INPUT_FORMATS: list = [
    'auto', 'csv', 'tsv', 'excel', 'parquet', 'hdf5', 'feather', 'json',
]

SUPPORTED_OUTPUT_FORMATS: list = [
    'csv', 'tsv', 'excel', 'parquet', 'hdf5', 'feather', 'json',
]

SUPPORTED_PLOT_FORMATS: list = [
    'png', 'svg', 'pdf', 'tiff', 'jpeg',
]

# ---------------------------------------------------------------------------
# Core exports
# ---------------------------------------------------------------------------

from SelectOmics.config import SelectOmicsConfig
from SelectOmics.pipeline import SelectOmicsPipeline

# ---------------------------------------------------------------------------
# Utility exports
# ---------------------------------------------------------------------------

from SelectOmics.data.loaders import load_omics_data, save_results_multiformat
from SelectOmics.evaluation.validation import FeatureSetValidator
from SelectOmics.utils.diagnostics import (
    assess_feature_redundancy,
    assess_sample_size_adequacy,
)

# ---------------------------------------------------------------------------
# __all__ -- controls 'from SelectOmics import *'
# ---------------------------------------------------------------------------

__all__: list = [
    # Metadata
    '__version__',
    '__author__',
    # Constants
    'ALGORITHMS',
    'SUPPORTED_INPUT_FORMATS',
    'SUPPORTED_OUTPUT_FORMATS',
    'SUPPORTED_PLOT_FORMATS',
    # Core
    'SelectOmicsConfig',
    'SelectOmicsPipeline',
    # Logging helper
    'enable_logging',
    # Utilities
    'load_omics_data',
    'save_results_multiformat',
    'FeatureSetValidator',
    'assess_feature_redundancy',
    'assess_sample_size_adequacy',
]