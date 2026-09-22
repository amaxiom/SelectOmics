"""
SelectOmics/data/preprocessing.py

Data preparation functions for the SelectOmics pipeline.

Converts a raw loaded DataFrame into train/test splits with encoded
labels and a configured StratifiedKFold object.  All state produced
here is returned explicitly; no globals are written.

"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from SelectOmics.config import SelectOmicsConfig

logger = logging.getLogger(__name__)


def prepare_data(
    df: pd.DataFrame,
    target_column: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract the feature matrix and target vector from a loaded DataFrame.

    Drops the target column from the feature set and retains only numeric
    columns.  Index order is sorted for reproducibility.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset as returned by load_omics_data.
    target_column : str
        Name of the column containing class labels.

    Returns
    -------
    X : pd.DataFrame
        Numeric feature matrix with the target column excluded.
    y : pd.Series
        Target column values, index-aligned to X.

    Raises
    ------
    ValueError
        If target_column is not present in df, or if no numeric feature
        columns remain after dtype filtering.
    """
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    # Separate target before any column filtering so the target column
    # is never accidentally retained as a feature.
    y = df[target_column].copy()

    X = df.drop(columns=[target_column])
    X = X.select_dtypes(include=[np.number]).copy()

    if X.empty:
        raise ValueError(
            "No numeric feature columns remain after dropping the target "
            f"column '{target_column}' and non-numeric columns."
        )

    # Sort index for deterministic ordering across runs.
    X = X.sort_index()
    y = pd.Series(y, index=X.index)

    return X, y


def handle_nans(
    X: pd.DataFrame,
    algorithm: str,
) -> Tuple[pd.DataFrame, str]:
    """
    Inspect the feature matrix for NaN values and adjust the algorithm
    if necessary.

    Logistic Regression and SVM cannot handle NaN values natively.
    When NaNs are detected and the configured algorithm is LR or SVM,
    the algorithm is automatically switched to XGB, which handles NaNs
    natively via its tree-splitting logic.  Random Forest and XGB are
    left unchanged.

    NaN values are never imputed; they are preserved so that tree-based
    algorithms can use the missingness pattern as a signal.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix, which may contain NaN values.
    algorithm : str
        Algorithm identifier from config: 'LR', 'XGB', 'RF', or 'SVM'.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix, unchanged (NaNs are not removed or imputed).
    algorithm : str
        Possibly updated algorithm identifier.  Returns 'XGB' if the
        original algorithm could not handle NaNs and NaNs were detected.
    """
    n_nans = int(X.isna().sum().sum())

    if n_nans == 0:
        return X, algorithm

    total_values = X.shape[0] * X.shape[1]
    nan_pct = (n_nans / total_values) * 100

    # Always warn about NaNs regardless of verbose setting, as this
    # directly affects algorithm selection and results.
    logger.warning(
        "Dataset contains %d NaN values (%.2f%% of data).", n_nans, nan_pct
    )

    if algorithm in ('LR', 'SVM'):
        logger.warning(
            "  Algorithm '%s' cannot handle NaN values. Automatically switching to XGB.",
            algorithm,
        )
        algorithm = 'XGB'
    else:
        logger.info(
            "  Algorithm '%s' handles NaN values natively. NaN values will be preserved.",
            algorithm,
        )

    return X, algorithm


def make_cv_splitter(
    y_train: np.ndarray,
    n_classes: int,
    config: SelectOmicsConfig,
) -> StratifiedKFold:
    """
    Build the StratifiedKFold used for every CV evaluation on a training set.

    An explicit config.cv_splits is clamped to [min_cv_splits,
    max_cv_splits]. Otherwise the fold count follows the minority class,
    within the same bounds, so no fold is left without a member of it.

    Shared by the main split and by each outer fold of nested CV. The inner
    loop of nested CV must build its folds exactly the way run() does, or it
    estimates a procedure that differs from the one the user ran.
    """
    if config.cv_splits is not None:
        n_splits = int(np.clip(
            config.cv_splits,
            config.min_cv_splits,
            config.max_cv_splits,
        ))
    else:
        min_class_count = int(
            np.bincount(np.asarray(y_train), minlength=n_classes).min()
        )
        n_splits = min(
            config.max_cv_splits,
            max(config.min_cv_splits, min_class_count),
        )
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=config.random_seed,
    )


def create_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    config: SelectOmicsConfig,
) -> Tuple[
    pd.DataFrame,   # X_train
    pd.DataFrame,   # X_test
    np.ndarray,     # y_train  (integer-encoded)
    np.ndarray,     # y_test   (integer-encoded)
    np.ndarray,     # classes  (np.arange(n_classes))
    List[str],      # class_names
    LabelEncoder,   # label_encoder
    StratifiedKFold,
]:
    """
    Encode labels, split into train and test sets, and create a
    StratifiedKFold cross-validator sized to the training set.

    Label encoding converts arbitrary class labels (strings, integers)
    to a contiguous integer range [0, n_classes - 1] required by
    sklearn and XGBoost.  The fitted LabelEncoder is returned so the
    pipeline can map predictions back to original class names.

    The number of CV folds is determined as follows:
    - If config.cv_splits is set explicitly, that value is used
      (clamped to [min_cv_splits, max_cv_splits]).
    - Otherwise it is derived from the minority class count in the
      training set: min(max_cv_splits, max(min_cv_splits,
      min_class_count_train)).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix returned by prepare_data or handle_nans.
    y : pd.Series
        Target vector, index-aligned to X.
    config : SelectOmicsConfig
        Pipeline configuration.  Relevant attributes: test_size,
        random_seed, cv_splits, min_cv_splits, max_cv_splits, verbose.

    Returns
    -------
    X_train : pd.DataFrame
    X_test : pd.DataFrame
    y_train : np.ndarray
        Integer-encoded training labels.
    y_test : np.ndarray
        Integer-encoded test labels.
    classes : np.ndarray
        Integer class indices, shape (n_classes,).
    class_names : list of str
        Original class label strings in label-encoder order.
    label_encoder : LabelEncoder
        Fitted encoder for inverse-transforming predictions.
    cv : StratifiedKFold
        Cross-validator configured for the training set.

    Raises
    ------
    ValueError
        If the target vector contains fewer than 2 unique classes, or if a
        class has only one member, which makes a stratified split impossible.
    """
    # Encode labels to contiguous integers.
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    unique_classes = label_encoder.classes_
    n_classes = len(unique_classes)

    if n_classes < 2:
        raise ValueError(
            f"Target column must contain at least 2 classes; "
            f"found {n_classes}."
        )

    # A stratified split needs two members of every class. Without this check
    # the failure surfaces as scikit-learn's message about the minimum number
    # of groups, which names neither the column nor the likely cause. The
    # usual cause is a continuous target: every value is distinct, so label
    # encoding turns n samples into n singleton classes.
    _counts = np.bincount(y_encoded, minlength=n_classes)
    _singletons = np.flatnonzero(_counts < 2)
    if _singletons.size:
        _names = [str(unique_classes[i]) for i in _singletons[:5]]
        _more = "" if _singletons.size <= 5 else f" (and {_singletons.size - 5} more)"
        _continuous = (
            " The target has almost as many distinct values as samples, which "
            "is what a continuous target looks like after label encoding. "
            "SelectOmics classifies; bin a continuous target into classes "
            "before passing it."
            if n_classes > 0.5 * len(y_encoded) else ""
        )
        raise ValueError(
            f"Target column has {_singletons.size} class(es) with fewer than 2 "
            f"samples: {_names}{_more}. A stratified train/test split needs at "
            f"least 2 of every class. Merge or drop the rare class(es), or "
            f"collect more of them.{_continuous}"
        )

    class_names: List[str] = [str(c) for c in unique_classes]
    classes: np.ndarray = np.arange(n_classes)

    # Stratified train/test split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        stratify=y_encoded,
        test_size=config.test_size,
        random_state=config.random_seed,
    )

    train_class_counts = np.bincount(y_train, minlength=n_classes)
    min_class_count_train = int(train_class_counts.min())
    cv = make_cv_splitter(y_train, n_classes, config)
    n_splits = cv.n_splits

    # A rare class can round to zero test samples. Nothing downstream can
    # recover from that: the held-out macro OVR AUC is undefined when a class
    # is absent from y_true, so it comes back nan and the generalisation gap
    # built on it is meaningless. It was previously visible only as a line in
    # the verbose split summary, which is off by default, so a run could
    # report an unmeasurable held-out score without ever saying why.
    # Warned rather than raised: the run is still informative, and refusing it
    # would remove a capability over a condition the caller may accept.
    _test_counts = np.bincount(y_test, minlength=n_classes)
    _absent = np.flatnonzero(_test_counts == 0)
    if _absent.size:
        logger.warning(
            "  %d class(es) absent from the held-out test set: %s. The "
            "held-out AUC cannot be computed for a multiclass problem when a "
            "class is missing from it, so it will be reported as nan and the "
            "generalisation gap will not be meaningful. Increase test_size, "
            "or merge the rare class(es).",
            int(_absent.size),
            [str(unique_classes[i]) for i in _absent],
        )

    if config.verbose:
        train_dist = dict(zip(class_names, train_class_counts.tolist()))
        test_class_counts = np.bincount(y_test, minlength=n_classes)
        test_dist = dict(zip(class_names, test_class_counts.tolist()))

        logger.info("Data split summary:")
        logger.info("  Training set: %s", X_train.shape)
        logger.info("  Test set:     %s", X_test.shape)
        logger.info("  Train class distribution: %s", train_dist)
        logger.info("  Test class distribution:  %s", test_dist)
        logger.info("Dataset summary:")
        logger.info("  Features: %d", X.shape[1])
        logger.info(
            "  Samples:  %d (train: %d, test: %d)",
            X.shape[0], len(X_train), len(X_test),
        )
        logger.info("  Classes: %s (n=%d)", class_names, n_classes)
        logger.info("  Min class count (train): %d", min_class_count_train)
        logger.info("  CV splits: %d", n_splits)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        classes,
        class_names,
        label_encoder,
        cv,
    )