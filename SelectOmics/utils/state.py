"""
SelectOmics/utils/state.py

Class-based pipeline state tracking and reference model cache.

Replaces the global-variable pattern used in the notebook
(X_clean, X_rfecv_final, REFERENCE_CACHE, etc.) with two explicit
objects held on the SelectOmicsPipeline instance:

    self.state_tracker  -- DataStateTracker
    (reference caching is handled inside DataStateTracker)

No module-level mutable state.  All state is instance-owned, so
multiple pipeline instances are fully isolated.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# PipelineState -- immutable snapshot of data and evaluations after one step
# ---------------------------------------------------------------------------

@dataclass
class PipelineState:
    """
    Snapshot of the feature matrices and evaluation results produced by
    one pipeline step.

    Parameters
    ----------
    step_name : str
        Human-readable identifier, e.g. 'Step 1 (Data Cleaning)'.
    X_train : pd.DataFrame
        Training feature matrix after this step.
    X_test : pd.DataFrame
        Test feature matrix after this step (column-aligned with X_train).
    eval_lr : dict or None
        Cross-validation result dict for Logistic Regression, or None if
        the step did not evaluate LR.
    eval_xgb : dict or None
        Cross-validation result dict for XGBoost.
    eval_rf : dict or None
        Cross-validation result dict for Random Forest.
    eval_svm : dict or None
        Cross-validation result dict for SVM.
    """

    step_name: str
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    eval_lr: Optional[Dict] = field(default=None)
    eval_xgb: Optional[Dict] = field(default=None)
    eval_rf: Optional[Dict] = field(default=None)
    eval_svm: Optional[Dict] = field(default=None)


# ---------------------------------------------------------------------------
# DataStateTracker -- ordered history of PipelineState snapshots
# ---------------------------------------------------------------------------

class DataStateTracker:
    """
    Ordered, append-only log of pipeline step outputs.

    The tracker stores one PipelineState per step in insertion order.
    get_latest() returns the most recently added snapshot; get_step()
    retrieves a snapshot by step name for targeted lookup.

    A reference-model cache is embedded here so that Step 0 evaluations
    can be retrieved by downstream steps without recomputation.  The
    cache is keyed on a SHA-256 hash of the sorted feature column names
    concatenated with the algorithm identifier, making it deterministic
    and process-safe.

    Usage
    -----
    Instantiate once on SelectOmicsPipeline and pass self.state_tracker
    to each run_step* function:

        self.state_tracker = DataStateTracker()

        # After Step 1 completes:
        self.state_tracker.update(
            step_name='Step 1 (Data Cleaning)',
            X_train=X_clean,
            X_test=X_test_clean,
            eval_lr=eval_step1_lr,
            eval_xgb=eval_step1_xgb,
            eval_rf=eval_step1_rf,
            eval_svm=eval_step1_svm,
        )

        # Retrieve latest state in the next step:
        state = self.state_tracker.get_latest()
        X_input = state.X_train
    """

    def __init__(self) -> None:
        # Ordered list of PipelineState snapshots; append-only.
        self._history: List[PipelineState] = []

        # Reference model cache.
        # Key: deterministic hex string (see _reference_key).
        # Value: result dict returned by run_reference_evaluation.
        self._reference_cache: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # State history interface
    # ------------------------------------------------------------------

    def update(
        self,
        step_name: str,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        eval_lr: Optional[Dict] = None,
        eval_xgb: Optional[Dict] = None,
        eval_rf: Optional[Dict] = None,
        eval_svm: Optional[Dict] = None,
    ) -> None:
        """
        Append a new PipelineState snapshot to the history.

        Parameters
        ----------
        step_name : str
            Human-readable step identifier.
        X_train : pd.DataFrame
            Training feature matrix produced by the step.
        X_test : pd.DataFrame
            Test feature matrix produced by the step.
        eval_lr : dict or None
            LR evaluation result, or None.
        eval_xgb : dict or None
            XGBoost evaluation result, or None.
        eval_rf : dict or None
            Random Forest evaluation result, or None.
        eval_svm : dict or None
            SVM evaluation result, or None.
        """
        self._history.append(
            PipelineState(
                step_name=step_name,
                X_train=X_train,
                X_test=X_test,
                eval_lr=eval_lr,
                eval_xgb=eval_xgb,
                eval_rf=eval_rf,
                eval_svm=eval_svm,
            )
        )

    def get_latest(self) -> PipelineState:
        """
        Return the most recently appended PipelineState.

        Returns
        -------
        PipelineState
            The last snapshot in the history.

        Raises
        ------
        RuntimeError
            If no state has been recorded yet.
        """
        if not self._history:
            raise RuntimeError(
                "DataStateTracker has no recorded state. "
                "Call update() after at least one pipeline step."
            )
        return self._history[-1]

    def get_step(self, step_name: str) -> Optional[PipelineState]:
        """
        Return the PipelineState recorded under a specific step name.

        If the same step name appears more than once (e.g. due to a
        retry), the most recent matching entry is returned.

        Parameters
        ----------
        step_name : str
            Exact step name as passed to update().

        Returns
        -------
        PipelineState or None
            The matching snapshot, or None if not found.
        """
        # Iterate in reverse so the most recent match is returned first.
        for state in reversed(self._history):
            if state.step_name == step_name:
                return state
        return None

    def clear(self) -> None:
        """
        Remove all recorded states and cached reference evaluations.

        Intended for pipeline resets and unit tests; not called during
        normal pipeline execution.
        """
        self._history.clear()
        self._reference_cache.clear()

    # ------------------------------------------------------------------
    # Reference model cache interface
    # ------------------------------------------------------------------

    @staticmethod
    def _reference_key(X: pd.DataFrame, algorithm: str) -> str:
        """
        Compute a deterministic cache key from the feature column names
        and the algorithm identifier.

        SHA-256 over the UTF-8 encoding of the pipe-joined sorted column
        names and algorithm string is used in place of Python's built-in
        hash(), which is randomised per process via PYTHONHASHSEED.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix whose column names define the feature set.
        algorithm : str
            Algorithm identifier, e.g. 'XGB'.

        Returns
        -------
        str
            A 16-character hex digest suitable for use as a dict key.
        """
        columns_str = '|'.join(sorted(X.columns.tolist()))
        raw = f"{algorithm}::{columns_str}".encode('utf-8')
        return hashlib.sha256(raw).hexdigest()[:16]

    def cache_reference(
        self, X: pd.DataFrame, algorithm: str, result: Dict
    ) -> str:
        """
        Store a reference model evaluation result in the cache.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix used for the reference evaluation.
        algorithm : str
            Algorithm identifier.
        result : dict
            Result dict returned by run_reference_evaluation.

        Returns
        -------
        str
            The cache key under which the result was stored.
        """
        key = self._reference_key(X, algorithm)
        self._reference_cache[key] = result
        return key

    def get_cached_reference(
        self, X: pd.DataFrame, algorithm: str
    ) -> Optional[Dict]:
        """
        Retrieve a cached reference evaluation result, if present.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix that was used for the reference evaluation.
        algorithm : str
            Algorithm identifier.

        Returns
        -------
        dict or None
            The cached result, or None if no entry exists for this
            feature set and algorithm combination.
        """
        key = self._reference_key(X, algorithm)
        return self._reference_cache.get(key)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_steps(self) -> int:
        """Number of pipeline states recorded so far."""
        return len(self._history)

    @property
    def step_names(self) -> List[str]:
        """Ordered list of all recorded step names."""
        return [s.step_name for s in self._history]