"""
SelectOmics/models/base.py

Model pipeline factories and hyperparameter tuning for SelectOmics.

Provides two layers of pipeline creation:

1. Quick-tune layer: RandomizedSearchCV over a fixed search space for each
   algorithm. Run once at pipeline start on the full training set. Returns
   four fitted pipelines (one per algorithm), of which only the configured
   algorithm is tuned; the others receive sensible fixed defaults.

2. Consensus layer: factory functions that construct a new pipeline
   instance from tuned hyperparameters, varying only the random seed.
   Seed variation drives bootstrap diversity in cv_evaluate_model; all
   other hyperparameters are fixed to the quick-tune result so that
   consensus models are genuinely comparable.

No global state. All configuration is passed as SelectOmicsConfig or as
explicit parameters. Tuned pipelines are passed to the consensus factories
rather than read from a module-level variable.

"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

import logging

from SelectOmics.config import SelectOmicsConfig
from SelectOmics.utils.helpers import detect_cpu_count

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thread budget
# ---------------------------------------------------------------------------
# Every estimator takes its thread count from config.n_jobs through the helpers
# below, so one setting controls the whole pipeline.
#
# Thread count must never change results.  RandomForest, RFECV, and
# permutation_importance satisfy that already: each seeds from random_state
# rather than from how the work is divided, and returns bit-identical output at
# any n_jobs.  XGBoost does NOT satisfy it under its default tree_method once
# subsample < 1, which is why xgb_compute_kwargs pins 'exact'; see that
# function for the measurements.

_DEFAULT_N_JOBS: int = -1

# Matrices smaller than this (n_samples * n_features) are faster single-threaded:
# thread start-up costs more than the work saved.  Measured on a 24-core box,
# RandomForest at n=60/p=80 is ~1.4x SLOWER with all cores, while at p=2000 and
# above it is 2-8x faster.  Only applied to estimators whose output does not
# depend on thread count, so switching on size stays reproducible.
_SMALL_MATRIX_CELLS: int = 50_000


def xgb_compute_kwargs(n_jobs: int, device: str = 'cpu') -> dict:
    """
    Return the XGBoost arguments that control how a fit is computed.

    ``tree_method='exact'`` is pinned deliberately.  XGBoost's default ('auto',
    which resolves to 'hist') is **not** thread-invariant once ``subsample < 1``,
    and the tuned search space always samples subsample in [0.6, 1.0).  Under
    'hist' the same seed therefore yields different trees at different thread
    counts, which would quietly break the reproducibility this package
    promises.  'exact' consumes its RNG independently of the thread split.

    Pinning it also happens to be much faster in the regime SelectOmics targets,
    because 'hist' pays a binning cost that only amortises over large n.
    Measured per fit at n_jobs=-1: n=150/p=2000, 0.065s exact vs 0.459s hist;
    n=200/p=10000, 0.454s vs 4.630s.  'exact' also parallelises better here
    (11.5x across 24 cores, against 2.6x for 'hist').

    CUDA is the exception: the GPU implementation supports only 'hist', and
    already perturbs results relative to CPU, so use_gpu is documented as
    trading reproducibility for speed.

    Parameters
    ----------
    n_jobs : int
        Threads for this estimator.
    device : str
        'cpu' or 'cuda'.

    Returns
    -------
    dict
        Keyword arguments to splat into an XGBClassifier constructor.
    """
    return {
        'n_jobs': n_jobs,
        'device': device,
        'tree_method': 'hist' if device == 'cuda' else 'exact',
    }


def effective_n_jobs(n_jobs: int, n_samples: int, n_features: int) -> int:
    """
    Resolve a thread request against the machine and the size of the work.

    Degrades to serial rather than failing, in two situations:

    - the environment reports a single usable core (a cgroup-limited container,
      a restricted sandbox), so there is nothing to parallelise across;
    - the matrix is too small for threading to pay for itself, where a full
      thread pool measurably loses to a single thread.

    Both are safe because this is only applied to estimators whose output does
    not depend on thread count, so the choice never changes results.

    Parameters
    ----------
    n_jobs : int
        Requested thread count; -1 means all available cores.
    n_samples, n_features : int
        Shape of the matrix about to be fitted.

    Returns
    -------
    int
        The thread count to actually use, always >= 1.
    """
    if n_jobs == 1:
        return 1
    if n_samples * n_features < _SMALL_MATRIX_CELLS:
        return 1
    if detect_cpu_count() <= 1:
        return 1
    return n_jobs


def resolve_n_jobs(config: Optional[SelectOmicsConfig] = None) -> int:
    """
    Return the thread count estimators should use.

    Parameters
    ----------
    config : SelectOmicsConfig or None
        Configuration to read ``n_jobs`` from.  ``None`` falls back to the
        package default, which keeps the standalone factory functions usable
        without a config object.

    Returns
    -------
    int
        ``-1`` for all cores, otherwise a positive thread cap.
    """
    if config is None:
        return _DEFAULT_N_JOBS
    return int(getattr(config, 'n_jobs', _DEFAULT_N_JOBS))


# ---------------------------------------------------------------------------
# Internal default pipeline builders
# Used when an algorithm is not being tuned (non-selected algorithms).
# ---------------------------------------------------------------------------

def _default_lr(seed: int) -> Pipeline:
    # No n_jobs: scikit-learn 1.8 made it a no-op for LogisticRegression
    # (it only ever drove the one-vs-rest outer loop) and 1.10 removes it.
    # The solver already threads through BLAS.
    return Pipeline([
        ('scaler', MinMaxScaler()),
        ('clf', LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=seed,
        )),
    ])


def _default_xgb(seed: int, device: str = 'cpu', n_jobs: int = -1) -> Pipeline:
    return Pipeline([
        ('scaler', MinMaxScaler()),
        ('clf', XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=seed,
            verbosity=0,
            eval_metric='mlogloss',
            **xgb_compute_kwargs(n_jobs, device),
        )),
    ])


def _default_rf(seed: int, n_jobs: int = -1) -> Pipeline:
    return Pipeline([
        ('scaler', MinMaxScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            class_weight='balanced',
            random_state=seed,
            n_jobs=n_jobs,
        )),
    ])


def _default_svm(seed: int) -> Pipeline:
    return Pipeline([
        ('scaler', MinMaxScaler()),
        ('clf', SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=seed,
        )),
    ])


# ---------------------------------------------------------------------------
# quick_tune_algorithm
# ---------------------------------------------------------------------------

def quick_tune_algorithm(
    algorithm: str,
    X: pd.DataFrame,
    y: np.ndarray,
    config: SelectOmicsConfig,
    device: str = 'cpu',
) -> Pipeline:
    """
    Run RandomizedSearchCV for one algorithm and return the best pipeline.

    Search spaces and base estimators match the notebook Cell 10
    implementation exactly.  n_jobs=1 is enforced throughout to prevent
    memory errors on Windows and to ensure deterministic behaviour.

    Parameters
    ----------
    algorithm : str
        One of 'LR', 'XGB', 'RF', 'SVM'.
    X : pd.DataFrame
        Training feature matrix.
    y : np.ndarray
        Integer-encoded training labels.
    config : SelectOmicsConfig
        Pipeline configuration.  Uses random_seed and quick_tune_iterations.

    Returns
    -------
    Pipeline
        Best estimator from RandomizedSearchCV, already fitted on X and y.

    Raises
    ------
    ValueError
        If algorithm is not one of the four supported values.
    """
    seed = config.random_seed
    n_iter = config.quick_tune_iterations
    n_jobs = effective_n_jobs(resolve_n_jobs(config), *X.shape)

    # Determine a conservative CV fold count for the tuning pass.
    # Use len(np.unique(y)) as minlength to guard against the (unlikely) case
    # where a class label is absent from y, which would cause np.bincount to
    # return a truncated array and silently produce the wrong minimum count.
    _n_classes_local = int(len(np.unique(y)))
    min_class = int(np.bincount(y, minlength=_n_classes_local).min())
    tune_cv_splits = min(5, max(2, min_class))
    tune_cv = StratifiedKFold(
        n_splits=tune_cv_splits,
        shuffle=True,
        random_state=seed,
    )

    if algorithm == 'LR':
        base_model = Pipeline([
            ('scaler', MinMaxScaler()),
            ('clf', LogisticRegression(
                max_iter=1000,
                random_state=seed,
            )),
        ])
        param_dist = {
            'clf__C':       loguniform(0.01, 100),
            'clf__penalty': ['l2'],
            'clf__solver':  ['lbfgs', 'saga'],
        }

    elif algorithm == 'XGB':
        base_model = Pipeline([
            ('scaler', MinMaxScaler()),
            ('clf', XGBClassifier(
                random_state=seed,
                verbosity=0,
                eval_metric='mlogloss',
                **xgb_compute_kwargs(n_jobs, device),
            )),
        ])
        param_dist = {
            'clf__n_estimators':    randint(50, 300),
            'clf__max_depth':       randint(3, 8),
            'clf__learning_rate':   loguniform(0.01, 0.3),
            'clf__subsample':       uniform(0.6, 0.4),
            'clf__colsample_bytree': uniform(0.6, 0.4),
            'clf__reg_alpha':       loguniform(0.01, 10),
            'clf__reg_lambda':      loguniform(0.01, 10),
        }

    elif algorithm == 'RF':
        base_model = Pipeline([
            ('scaler', MinMaxScaler()),
            ('clf', RandomForestClassifier(
                random_state=seed,
                n_jobs=n_jobs,
                class_weight='balanced',
            )),
        ])
        param_dist = {
            'clf__n_estimators':    randint(50, 300),
            'clf__max_depth':       [None] + list(randint(3, 20).rvs(10,
                                    random_state=seed)),
            'clf__min_samples_split': randint(2, 10),
            'clf__min_samples_leaf':  randint(1, 5),
            'clf__max_features':    ['sqrt', 'log2', None],
        }

    elif algorithm == 'SVM':
        base_model = Pipeline([
            ('scaler', MinMaxScaler()),
            ('clf', SVC(
                kernel='rbf',
                probability=True,
                class_weight='balanced',
                random_state=seed,
            )),
        ])
        param_dist = {
            'clf__C':     loguniform(0.1, 100),
            'clf__gamma': loguniform(1e-4, 1e-1),
        }

    else:
        raise ValueError(
            f"Unsupported algorithm '{algorithm}'. "
            f"Choose from: 'LR', 'XGB', 'RF', 'SVM'."
        )

    _scorer = 'roc_auc' if _n_classes_local == 2 else 'roc_auc_ovr'
    # The search itself stays serial on purpose.  Its n_jobs spawns worker
    # processes, each holding a full copy of X; at omics feature counts that is
    # the memory blow-up the original notebook hit on Windows.  Parallelism is
    # taken inside each estimator instead, where it costs no extra copies.
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=tune_cv,
        scoring=_scorer,
        n_jobs=1,
        random_state=seed,
        verbose=0,
    )
    search.fit(X, y)

    if config.verbose:
        logger.debug("  %s best CV score: %.4f", algorithm, search.best_score_)
        logger.debug("  %s best params: %s", algorithm, search.best_params_)

    return search.best_estimator_


# ---------------------------------------------------------------------------
# quick_tune_all
# ---------------------------------------------------------------------------

def quick_tune_all(
    X: pd.DataFrame,
    y: np.ndarray,
    config: SelectOmicsConfig,
    device: str = 'cpu',
) -> Dict[str, Pipeline]:
    """
    Tune the configured algorithm and build default pipelines for the rest.

    Only the algorithm specified in config.algorithm is passed through
    RandomizedSearchCV.  The other three algorithms receive fixed-default
    pipelines so that the pipeline can still evaluate all four as reference
    models in Step 0 without incurring the cost of four full searches.

    Parameters
    ----------
    X : pd.DataFrame
        Training feature matrix.
    y : np.ndarray
        Integer-encoded training labels.
    config : SelectOmicsConfig
        Pipeline configuration.

    Returns
    -------
    dict
        Keys: 'LR', 'XGB', 'RF', 'SVM'.
        Values: fitted Pipeline instances.
    """
    seed = config.random_seed
    algorithm = config.algorithm
    n_jobs = effective_n_jobs(resolve_n_jobs(config), *X.shape)

    if config.verbose:
        logger.debug("Quick tuning %s (n_jobs=%d)...", algorithm, n_jobs)

    pipelines: Dict[str, Pipeline] = {
        'LR':  _default_lr(seed),
        'XGB': _default_xgb(seed, device=device, n_jobs=n_jobs),
        'RF':  _default_rf(seed, n_jobs=n_jobs),
        'SVM': _default_svm(seed),
    }

    # Replace the selected algorithm with a tuned version.
    pipelines[algorithm] = quick_tune_algorithm(algorithm, X, y, config, device=device)

    return pipelines


# ---------------------------------------------------------------------------
# Consensus pipeline factories
# ---------------------------------------------------------------------------
# Each factory accepts the tuned pipeline for its algorithm so it can
# extract the best hyperparameters found during quick_tune_all.  Only
# the random seed is varied; all other parameters are locked to the
# tuned values to ensure consensus models are genuinely comparable.
# ---------------------------------------------------------------------------

def create_lr_pipeline_consensus(
    seed: int,
    tuned_lr: Pipeline,
) -> Pipeline:
    """
    Build an LR consensus pipeline seeded with seed.

    Hyperparameters (C) are taken from tuned_lr.  Diversity across
    consensus models comes from bootstrap resampling in cv_evaluate_model,
    not from hyperparameter variation.

    Parameters
    ----------
    seed : int
        Random seed for LogisticRegression.
    tuned_lr : Pipeline
        Fitted LR pipeline returned by quick_tune_all.

    Returns
    -------
    Pipeline
        Unfitted Pipeline ready for cloning and fitting.
    """
    C = tuned_lr.named_steps['clf'].C
    return Pipeline([
        ('scaler', MinMaxScaler()),
        ('clf', LogisticRegression(
            max_iter=1000,
            C=C,
            random_state=seed,
        )),
    ])


def create_xgb_pipeline_consensus(
    seed: int,
    tuned_xgb: Pipeline,
) -> Pipeline:
    """
    Build an XGB consensus pipeline seeded with seed.

    Parameters
    ----------
    seed : int
        Random seed for XGBClassifier.
    tuned_xgb : Pipeline
        Fitted XGB pipeline returned by quick_tune_all.

    Returns
    -------
    Pipeline
        Unfitted Pipeline ready for cloning and fitting.
    """
    p = tuned_xgb.named_steps['clf'].get_params()
    _device = p.get('device', 'cpu')
    # n_jobs and device are inherited from the tuned pipeline, which took them
    # from config.  That keeps the tuned estimator the single source of truth
    # and means consensus models never quietly run on a different budget than
    # the model whose hyperparameters they carry.
    return Pipeline([
        ('scaler', MinMaxScaler()),
        ('clf', XGBClassifier(
            n_estimators=p.get('n_estimators', 100),
            max_depth=p.get('max_depth', 5),
            learning_rate=p.get('learning_rate', 0.1),
            subsample=p.get('subsample', 0.8),
            colsample_bytree=p.get('colsample_bytree', 0.8),
            reg_alpha=p.get('reg_alpha', 0.1),
            reg_lambda=p.get('reg_lambda', 1.0),
            random_state=seed,
            verbosity=0,
            eval_metric='mlogloss',
            **xgb_compute_kwargs(p.get('n_jobs', -1), _device),
        )),
    ])


def create_rf_pipeline_consensus(
    seed: int,
    tuned_rf: Pipeline,
) -> Pipeline:
    """
    Build an RF consensus pipeline seeded with seed.

    Parameters
    ----------
    seed : int
        Random seed for RandomForestClassifier.
    tuned_rf : Pipeline
        Fitted RF pipeline returned by quick_tune_all.

    Returns
    -------
    Pipeline
        Unfitted Pipeline ready for cloning and fitting.
    """
    p = tuned_rf.named_steps['clf'].get_params()
    return Pipeline([
        ('scaler', MinMaxScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=p.get('n_estimators', 100),
            max_depth=p.get('max_depth', None),
            min_samples_split=p.get('min_samples_split', 2),
            min_samples_leaf=p.get('min_samples_leaf', 1),
            max_features=p.get('max_features', 'sqrt'),
            class_weight='balanced',
            random_state=seed,
            n_jobs=p.get('n_jobs', -1),
        )),
    ])


def create_svm_pipeline_consensus(
    seed: int,
    tuned_svm: Pipeline,
) -> Pipeline:
    """
    Build an SVM consensus pipeline seeded with seed.

    Parameters
    ----------
    seed : int
        Random seed for SVC.
    tuned_svm : Pipeline
        Fitted SVM pipeline returned by quick_tune_all.

    Returns
    -------
    Pipeline
        Unfitted Pipeline ready for cloning and fitting.
    """
    p = tuned_svm.named_steps['clf'].get_params()
    return Pipeline([
        ('scaler', MinMaxScaler()),
        ('clf', SVC(
            kernel='rbf',
            C=p.get('C', 1.0),
            gamma=p.get('gamma', 'scale'),
            probability=True,
            class_weight='balanced',
            random_state=seed,
        )),
    ])


# ---------------------------------------------------------------------------
# build_consensus_pipeline  (single pipeline, one seed)
# ---------------------------------------------------------------------------

def build_consensus_pipeline(
    algorithm: str,
    seed: int,
    tuned_pipelines: Dict[str, Pipeline],
) -> Pipeline:
    """
    Build one consensus pipeline for algorithm seeded with seed.

    Parameters
    ----------
    algorithm : str
        One of 'LR', 'XGB', 'RF', 'SVM'.
    seed : int
        Random seed for the estimator.
    tuned_pipelines : dict
        Output of quick_tune_all.

    Returns
    -------
    Pipeline
        Unfitted Pipeline ready for cloning and fitting.
    """
    factories = {
        'LR':  create_lr_pipeline_consensus,
        'XGB': create_xgb_pipeline_consensus,
        'RF':  create_rf_pipeline_consensus,
        'SVM': create_svm_pipeline_consensus,
    }
    factory = factories[algorithm]
    tuned = tuned_pipelines[algorithm]
    return factory(seed=seed, **{f"tuned_{algorithm.lower()}": tuned})


# ---------------------------------------------------------------------------
# build_consensus_pipelines
# ---------------------------------------------------------------------------

def build_consensus_pipelines(
    algorithm: str,
    n_models: int,
    tuned_pipelines: Dict[str, Pipeline],
    base_seed: int,
) -> Dict[str, Pipeline]:
    """
    Build n_models consensus pipelines for the given algorithm.

    Seeds are spaced 100 apart starting from base_seed to ensure
    statistically independent random states across models.

    Parameters
    ----------
    algorithm : str
        One of 'LR', 'XGB', 'RF', 'SVM'.
    n_models : int
        Number of consensus models to create.
    tuned_pipelines : dict
        Output of quick_tune_all.
    base_seed : int
        Starting seed; model i uses base_seed + i * 100.

    Returns
    -------
    dict
        Keys: '{algorithm}-{i}' (1-indexed).
        Values: unfitted Pipeline instances.
    """
    factories = {
        'LR':  create_lr_pipeline_consensus,
        'XGB': create_xgb_pipeline_consensus,
        'RF':  create_rf_pipeline_consensus,
        'SVM': create_svm_pipeline_consensus,
    }
    factory = factories[algorithm]
    tuned = tuned_pipelines[algorithm]

    return {
        f"{algorithm}-{i + 1}": factory(seed=base_seed + i * 100, **{
            f"tuned_{algorithm.lower()}": tuned
        })
        for i in range(n_models)
    }
