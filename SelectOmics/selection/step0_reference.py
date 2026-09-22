"""
SelectOmics/selection/step0_reference.py

Step 0: Reference model evaluation.

Evaluates a single tuned reference model on X_train using stratified
cross-validation without bootstrap resampling. The returned result dict
is the baseline against which Steps 1-4 compare their consensus models.
It must be cached and passed explicitly to each subsequent step; it is
never re-computed mid-pipeline.

Design decisions
----------------
- A single model (n_models=1) is always used for the reference, regardless
  of n_consensus_models, so that the baseline reflects a fair single-model
  estimate and is not inflated by ensemble averaging.
- use_bootstrap=False is enforced for the reference so that the CV folds
  are not resampled; the reference must reflect the unmodified training
  distribution.
- No fallback to previous state is needed here: Step 0 always runs on
  X_train directly.
- Visualisations and summary saving are conditional on config flags and
  are skipped entirely when the corresponding flags are False, keeping the
  function side-effect-free in default-off configurations.

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..config import SelectOmicsConfig
from ..models.evaluation import evaluate_models_collection
from ..utils.helpers import save_step_summary, print_adequacy_warning

logger = logging.getLogger(__name__)

# Backward-compat alias: step2 and step3 import _print_adequacy_warning from here.
_print_adequacy_warning = print_adequacy_warning


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_step0_reference(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    config: SelectOmicsConfig,
    cv: StratifiedKFold,
    tuned_pipelines: Dict[str, Any],
    n_classes: int,
    class_names: List[str],
    adequacy: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Evaluate a single tuned reference model on the full training set.

    This establishes the baseline cross-validation AUC against which every
    subsequent feature-selection step is compared. It must be called once
    before Steps 1-4 and its return value passed to each step as
    ``ref_result_step0``.

    Parameters
    ----------
    X_train : pd.DataFrame
        Full training feature matrix, shape (n_samples, n_features).
    y_train : np.ndarray
        Encoded integer class labels, shape (n_samples,).
    config : SelectOmicsConfig
        Pipeline configuration object.
    cv : StratifiedKFold
        Stratified cross-validation splitter. The same object must be
        reused across all steps to ensure comparable fold assignments.
    tuned_pipelines : dict
        Mapping from algorithm key to fitted sklearn Pipeline returned by
        ``quick_tune_all``.  Expected keys: 'LR', 'XGB', 'RF', 'SVM'.
        Only the key matching ``config.algorithm`` is used.
    n_classes : int
        Number of distinct target classes.
    class_names : list of str
        Human-readable class label strings aligned with encoded integers.
    adequacy : dict, optional
        Output of ``assess_sample_size_adequacy``.  When provided, the
        overall severity level is printed as a warning if non-adequate.
    output_dir : Path, optional
        Destination directory for saved artefacts.  Derived from
        ``config.output_dir`` when None.

    Returns
    -------
    dict with keys:
        reference_result : dict
            Single-model CV result with keys 'mean_auc', 'std_auc',
            'fold_aucs', 'oof_proba', 'oof_pred'.  This is the value
            that Steps 1-4 receive as ``ref_result_step0``.
        reference_evaluation : dict
            Full output of ``evaluate_models_collection`` (includes
            'results', 'summary').
        algorithm : str
            Algorithm key used (mirrors config.algorithm).
        n_features : int
            Number of features in X_train.
        model_name : str
            Key under which the reference model was registered
            (e.g. 'Ref-XGB').

    Raises
    ------
    ValueError
        If ``config.algorithm`` is not present in ``tuned_pipelines``.
    """
    algorithm = config.algorithm
    n_models = config.n_consensus_models

    if output_dir is None:
        output_dir = Path(config.output_dir)

    logger.info("STEP 0: REFERENCE MODELS - %s ALGORITHM", algorithm)
    logger.info("Using %s with %d consensus model(s)", algorithm, n_models)

    # ------------------------------------------------------------------
    # Validate algorithm presence in tuned_pipelines
    # ------------------------------------------------------------------
    if algorithm not in tuned_pipelines:
        raise ValueError(
            f"Algorithm '{algorithm}' not found in tuned_pipelines. "
            f"Available keys: {list(tuned_pipelines.keys())}"
        )

    # ------------------------------------------------------------------
    # Build single-model reference dict
    #
    # The reference is always a single tuned pipeline regardless of
    # n_consensus_models.  This ensures the baseline reflects one model's
    # cross-validated performance, not an ensemble.
    # ------------------------------------------------------------------
    model_name = f"Ref-{algorithm}"
    reference_models = {model_name: tuned_pipelines[algorithm]}

    # ------------------------------------------------------------------
    # Print adequacy warnings if a non-adequate assessment was supplied
    # ------------------------------------------------------------------
    if adequacy is not None:
        print_adequacy_warning(
            step_name="Step 0: Reference Models",
            n_features_current=X_train.shape[1],
            adequacy=adequacy,
        )

    # ------------------------------------------------------------------
    # Evaluate reference model via cross-validation
    #
    # use_bootstrap=False: the reference must reflect unmodified training
    # folds.  Bootstrap diversity is reserved for the consensus steps.
    # ------------------------------------------------------------------
    logger.info("Evaluating reference model on training set (no bootstrap)...")

    reference_evaluation = evaluate_models_collection(
        pipelines=reference_models,
        X=X_train,
        y=y_train,
        cv=cv,
        n_classes=n_classes,
        classes=np.arange(n_classes),
        config=config,
        step_name="Step 0: Reference Models",
        use_bootstrap=False,
    )

    reference_result = reference_evaluation["results"][model_name]

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    logger.info("Reference model performance summary:")
    logger.info(
        "  %s: AUC = %.4f +/- %.4f",
        model_name,
        reference_result['mean_auc'],
        reference_result['std_auc'],
    )
    logger.info("  Feature count: %d", X_train.shape[1])

    # ------------------------------------------------------------------
    # Visualisations (conditional on config flag)
    # ------------------------------------------------------------------
    if config.create_visualizations:
        _plot_step0(
            reference_evaluation=reference_evaluation,
            y_train=y_train,
            n_classes=n_classes,
            config=config,
            output_dir=output_dir,
        )

    # ------------------------------------------------------------------
    # Save step summary (conditional on config flag)
    # ------------------------------------------------------------------
    summary_data = {
        "algorithm": algorithm,
        "n_consensus_models": n_models,
        "total_features": X_train.shape[1],
        f"{model_name.lower().replace('-', '_')}_mean_auc": reference_result["mean_auc"],
        f"{model_name.lower().replace('-', '_')}_std_auc": reference_result["std_auc"],
    }

    if config.save_intermediate_results:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_step_summary("step0", summary_data, output_dir)

    logger.info("Step 0 complete. Proceeding with %s-specific pipeline.", algorithm)

    return {
        "reference_result": reference_result,
        "reference_evaluation": reference_evaluation,
        "algorithm": algorithm,
        "n_features": X_train.shape[1],
        "model_name": model_name,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _plot_step0(
    reference_evaluation: Dict[str, Any],
    y_train: np.ndarray,
    n_classes: int,
    config: SelectOmicsConfig,
    output_dir: Path,
) -> None:
    """
    Generate Step 0 diagnostic plots.

    Generates an AUC box plot and ROC curve for the single reference
    model.  Both are saved using the format and path specified by config.
    Matplotlib is imported here to avoid backend side effects at module
    level; the caller is responsible for setting the backend before
    importing this module if a non-interactive environment is required.

    Parameters
    ----------
    reference_evaluation : dict
        Output of evaluate_models_collection for Step 0.
    y_train : np.ndarray
        Encoded training labels.
    n_classes : int
        Number of classes.
    config : SelectOmicsConfig
        Pipeline configuration.
    output_dir : Path
        Directory for saved figures.
    """
    # Import visualization lazily to allow callers to set backend before import.
    from ..evaluation.visualization import (
        plot_auc_boxplots,
        plot_roc_curves,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    fold_aucs = {
        name: result["fold_aucs"]
        for name, result in reference_evaluation["results"].items()
    }

    plot_auc_boxplots(
        step_name="Step 0: Reference Models",
        aucs_dict=fold_aucs,
        config=config,
        save_path=output_dir / "step0_reference_boxplots",
    )

    plot_roc_curves(
        models_dict=reference_evaluation["results"],
        y_true=y_train,
        n_classes=n_classes,
        classes=np.arange(n_classes),
        config=config,
        title="Step 0: Reference Models ROC",
        save_path=output_dir / "step0_reference_roc",
    )
