"""
SelectOmics/models/evaluation.py

Cross-validation evaluation and step-level comparison utilities.

Provides two primary functions:

- cv_evaluate_model: out-of-fold cross-validation with optional
  per-fold bootstrap resampling for consensus diversity.

- evaluate_models_collection: runs cv_evaluate_model over a dict of
  named pipelines and aggregates fold AUCs and OOF predictions.

All global state from the notebook (CLASSES, n_classes, SELECTOMICS_CONFIG)
is replaced by explicit parameters.  No module-level execution.

"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

from SelectOmics.config import SelectOmicsConfig
from SelectOmics.utils.helpers import compute_macro_auc_ovr, ensure_binary_proba


# ---------------------------------------------------------------------------
# cv_evaluate_model
# ---------------------------------------------------------------------------

# Bootstrap seeds pack (model_seed, fold_idx) into one integer. The radix
# exceeds any real fold count, so the packing is injective; the modulus keeps
# the result inside the range scikit-learn accepts for random_state.
_FOLD_SEED_RADIX: int = 1000
_SEED_MODULUS: int = 2 ** 31 - 1


def cv_evaluate_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    cv: StratifiedKFold,
    n_classes: int,
    classes: np.ndarray,
    use_bootstrap: bool = True,
) -> Dict[str, Any]:
    """
    Out-of-fold cross-validation with optional per-fold bootstrap resampling.

    When use_bootstrap=True, each fold's training split is bootstrap-
    resampled before fitting.  The bootstrap seed is derived from the
    model's random_state plus the fold index so that every (model, fold)
    pair sees a unique sample.  This is the mechanism that creates genuine
    diversity across consensus models: identical hyperparameters but
    different training subsets.

    When use_bootstrap=False (reference model evaluation), the original
    training split is used without resampling.

    Parameters
    ----------
    pipeline : Pipeline
        Unfitted sklearn Pipeline with a 'clf' step.
    X : pd.DataFrame
        Feature matrix (training set only).
    y : np.ndarray
        Integer-encoded labels aligned to X.
    cv : StratifiedKFold
        Cross-validator.  Must have been configured for the training set.
    n_classes : int
        Total number of classes in the problem.  Used to pre-allocate the
        OOF probability matrix.
    classes : np.ndarray
        Integer class indices, shape (n_classes,).  Passed to
        compute_macro_auc_ovr.
    use_bootstrap : bool
        If True, bootstrap-resample each fold's training split.

    Returns
    -------
    dict
        oof_proba : np.ndarray, shape (n_samples, n_classes)
            Out-of-fold predicted probabilities.
        oof_pred : np.ndarray, shape (n_samples,)
            Argmax of oof_proba.
        fold_aucs : list of float
            Per-fold macro OVR AUC.
        mean_auc : float
        std_auc : float
    """
    n_samples = X.shape[0]
    oof_proba = np.zeros((n_samples, n_classes), dtype=float)
    fold_aucs: List[float] = []

    # Extract the model's random_state for reproducible per-(model, fold)
    # bootstrap seeds.  Fall back to 0 if unavailable.
    model_seed: int = 0
    if hasattr(pipeline, 'named_steps') and 'clf' in pipeline.named_steps:
        clf = pipeline.named_steps['clf']
        if hasattr(clf, 'random_state') and clf.random_state is not None:
            model_seed = int(clf.random_state)

    X_arr = X  # DataFrame: iloc slicing used throughout

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_arr, y)):
        X_fold_train = X_arr.iloc[train_idx]
        X_fold_test  = X_arr.iloc[test_idx]
        y_fold_train = y[train_idx]
        y_fold_test  = y[test_idx]

        if use_bootstrap:
            # Unique seed per (model, fold) pair ensures no two models share
            # the same bootstrap sample across any fold.
            #
            # This must be an injective encoding of the pair, not a sum of
            # strides. The previous form, model_seed + fold_idx * 1000, aliased:
            # consensus model seeds are spaced 100 apart, so model i+10 at fold f
            # landed on exactly the seed of model i at fold f+1. At the shipped
            # n_consensus_models=20 preset that collapsed 100 (model, fold) pairs
            # onto 60 distinct seeds, making models 10-19 bootstrap duplicates of
            # models 0-9 and breaking the independence that consensus voting
            # assumes. Mixed-radix packing cannot alias, because fold_idx is
            # always far below the 1000 radix.
            boot_seed = (model_seed * _FOLD_SEED_RADIX + fold_idx) % _SEED_MODULUS
            boot_idx = resample(
                np.arange(len(X_fold_train)),
                n_samples=len(X_fold_train),
                replace=True,
                stratify=y_fold_train,
                random_state=boot_seed,
            )
            X_fold_train = X_fold_train.iloc[boot_idx]
            y_fold_train = y_fold_train[boot_idx]

        model = clone(pipeline)
        model.fit(X_fold_train, y_fold_train)

        # Check the underlying clf, not the Pipeline wrapper -- sklearn's
        # Pipeline always exposes predict_proba as a method even when the
        # wrapped estimator doesn't support it, so hasattr(Pipeline, ...)
        # is always True and the fallback branch would never be reached.
        _clf = model.named_steps.get('clf')
        if _clf is not None and hasattr(_clf, 'predict_proba'):
            y_proba_fold = model.predict_proba(X_fold_test)
        else:
            # Hard-prediction fallback: one-hot encode predictions.
            y_pred_fold = model.predict(X_fold_test)
            y_proba_fold = np.zeros((len(y_fold_test), n_classes))
            y_proba_fold[np.arange(len(y_fold_test)), y_pred_fold] = 1.0

        y_proba_fold = ensure_binary_proba(y_proba_fold, n_classes)

        # predict_proba returns one column per class the estimator SAW,
        # ordered by classes_, not one column per class in the problem. When
        # a fold's training split lacks a class the matrix is narrower than
        # n_classes and its columns no longer line up with the class indices
        # every consumer here assumes: the assignment below raises a numpy
        # shape-mismatch naming neither the fold nor the class, and
        # compute_macro_auc_ovr would read the wrong column.
        #
        # Scatter by classes_ so column j always means class j. A class the
        # model never saw gets probability 0, which is what it predicted. The
        # branch is a no-op whenever all classes are present, which is the
        # normal case, so nothing about an ordinary run changes.
        if y_proba_fold.shape[1] != n_classes:
            _fitted = getattr(_clf, 'classes_', None)
            if _fitted is not None and len(_fitted) == y_proba_fold.shape[1]:
                _wide = np.zeros((y_proba_fold.shape[0], n_classes), dtype=float)
                _wide[:, np.asarray(_fitted, dtype=int)] = y_proba_fold
                y_proba_fold = _wide

        oof_proba[test_idx] = y_proba_fold[:, :n_classes]
        fold_auc = compute_macro_auc_ovr(y_fold_test, y_proba_fold, classes)
        fold_aucs.append(fold_auc)

    oof_pred = np.argmax(oof_proba, axis=1)

    return {
        'oof_proba': oof_proba,
        'oof_pred':  oof_pred,
        'fold_aucs': fold_aucs,
        # Guarded like std_auc immediately below. Every fold failing leaves
        # fold_aucs empty, and np.mean([]) yields nan via a RuntimeWarning
        # rather than deliberately. Same value, stated on purpose.
        'mean_auc':  float(np.mean(fold_aucs)) if fold_aucs else float('nan'),
        'std_auc':   float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# evaluate_models_collection
# ---------------------------------------------------------------------------

def evaluate_models_collection(
    pipelines: Dict[str, Pipeline],
    X: pd.DataFrame,
    y: np.ndarray,
    cv: StratifiedKFold,
    n_classes: int,
    classes: np.ndarray,
    config: SelectOmicsConfig,
    step_name: str = '',
    use_bootstrap: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a collection of named pipelines via out-of-fold CV.

    Iterates over pipelines, calls cv_evaluate_model for each, and
    aggregates results.  Returns a structured dict containing per-model
    results, per-model fold AUC lists, and a top-level summary.

    Parameters
    ----------
    pipelines : dict
        Keys: model name strings.  Values: unfitted Pipeline instances.
    X : pd.DataFrame
        Training feature matrix.
    y : np.ndarray
        Integer-encoded training labels.
    cv : StratifiedKFold
        Cross-validator configured for the training set.
    n_classes : int
        Number of classes.
    classes : np.ndarray
        Integer class indices.
    config : SelectOmicsConfig
        Pipeline configuration.  Uses config.verbose.
    step_name : str
        Label used in verbose output.
    use_bootstrap : bool
        Passed to cv_evaluate_model for each pipeline.

    Returns
    -------
    dict
        results : dict
            Per-model cv_evaluate_model output, keyed by model name.
        summary : dict
            model_names, mean_aucs, std_aucs, feature_count.
    """
    results: Dict[str, Any] = {}

    for name, pipeline in pipelines.items():
        if config.verbose:
            mode = 'with bootstrap' if use_bootstrap else 'without bootstrap'
            logger.debug("  Evaluating %s (%s)...", name, mode)

        results[name] = cv_evaluate_model(
            pipeline=pipeline,
            X=X,
            y=y,
            cv=cv,
            n_classes=n_classes,
            classes=classes,
            use_bootstrap=use_bootstrap,
        )

    model_names = list(results.keys())
    mean_aucs   = [results[n]['mean_auc'] for n in model_names]
    std_aucs    = [results[n]['std_auc']  for n in model_names]

    if config.verbose:
        logger.debug("%s Performance Summary:", step_name)
        for name, mu, sd in zip(model_names, mean_aucs, std_aucs):
            logger.debug("  %s: AUC = %.4f +/- %.4f", name, mu, sd)
        logger.debug("  Feature count: %d", X.shape[1])

    return {
        'results': results,
        'summary': {
            'model_names':   model_names,
            'mean_aucs':     mean_aucs,
            'std_aucs':      std_aucs,
            'feature_count': X.shape[1],
        },
    }
