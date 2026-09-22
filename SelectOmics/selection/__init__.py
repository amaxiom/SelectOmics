"""
SelectOmics/selection/__init__.py

Public API for the SelectOmics selection layer.

Exports the five pipeline step entry points in execution order:

    run_step0_reference      - baseline reference model (no feature removal)
    run_step1_cleaning       - data-based cleaning: variance/correlation consensus
    run_step2_regularization - embedded selection: parallel L1/L2 consensus
    run_step3_wrapper        - Wrappers: RFECV/stability selection consensus
    
All step functions share a common call signature:

    run_stepN(
        X_train, X_test, y_train, config, cv,
        tuned_pipelines, n_classes, class_names,
        ref_result_step0=None, adequacy=None, output_dir=None
    ) -> dict

Downstream consumers (pipeline.py, notebooks) should import from this
package rather than from the individual step modules directly, so that
module renames and refactors remain transparent.
"""

from .step0_reference import run_step0_reference
from .step1_cleaning import run_step1_cleaning
from .step2_regularization import run_step2_regularization
from .step3_wrapper import run_step3_wrapper

__all__ = [
    "run_step0_reference",
    "run_step1_cleaning",
    "run_step2_regularization",
    "run_step3_wrapper",
]