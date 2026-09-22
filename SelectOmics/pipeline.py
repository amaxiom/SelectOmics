"""
SelectOmics/pipeline.py

Main orchestration class for the SelectOmics feature selection pipeline.

Converts the notebook's global-variable-driven execution into a stateful
object with an explicit, reproducible execution order.  All inter-step
state is held on the instance; no module-level mutable state is used.

Usage
-----
>>> config = {
...     'data_path': 'data.csv',
...     'target_column': 'Class',
...     'algorithm': 'XGB',
... }
>>> pipeline = SelectOmicsPipeline(config)
>>> results = pipeline.run()
"""

from __future__ import annotations

import gc
import hashlib
import logging
import pickle

logger = logging.getLogger(__name__)
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from .config import SelectOmicsConfig
from .data.loaders import load_omics_data, save_from_config
from .data.preprocessing import (
    create_train_test_split,
    handle_nans,
    make_cv_splitter,
    prepare_data,
)
from .evaluation.metrics import (
    _safe_roc_auc,
    build_comprehensive_metrics,
    build_pipeline_summary_df,
    binarize_labels,
    save_comprehensive_metrics,
    summarise_generalization,
)
from .evaluation.validation import (
    FeatureSetValidator,
    build_recommendation,
    weighted_validation_score,
)
from .evaluation.visualization import (
    plot_learning_curve_and_roc,
    plot_per_class_roc_and_confusion,
    plot_pipeline_summary,
    plot_validation_comparison,
)
from .models.base import quick_tune_all
from .utils.helpers import detect_gpu
from .selection.step0_reference import run_step0_reference
from .selection.step1_cleaning import run_step1_cleaning
from .selection.step2_regularization import run_step2_regularization
from .selection.step3_wrapper import run_step3_wrapper
from .utils.diagnostics import assess_sample_size_adequacy
from .utils.state import DataStateTracker


# ---------------------------------------------------------------------------
# Internal step-name constants (used as state_tracker keys and in results)
# ---------------------------------------------------------------------------

_STEP0_NAME = 'Step 0 (Reference)'
_STEP1_NAME = 'Step 1 (Data Cleaning)'
_STEP2_NAME = 'Step 2 (Regularization)'
_STEP3_NAME = 'Step 3 (Wrappers)'

# Maps public step key to state_tracker step_name.
_STEP_KEY_TO_NAME: Dict[str, str] = {
    'step0': _STEP0_NAME,
    'step1': _STEP1_NAME,
    'step2': _STEP2_NAME,
    'step3': _STEP3_NAME,
}
_STEP_NAME_TO_KEY: Dict[str, str] = {v: k for k, v in _STEP_KEY_TO_NAME.items()}


# ---------------------------------------------------------------------------
# SelectOmicsPipeline
# ---------------------------------------------------------------------------

class SelectOmicsPipeline:
    """
    End-to-end feature selection pipeline for high-dimensional omics data.

    Implements a three-stage reduction strategy against a reference:

    - Step 0: Reference model evaluation (baseline AUC).
    - Step 1: Data-based cleaning (constant, low-variance, correlated removal).
    - Step 2: Model-based regularization (parallel L1 and L2 consensus).
    - Step 3: Wrappers (RFECV with stability selection).

    Parameters
    ----------
    config : dict or SelectOmicsConfig
        Pipeline configuration.  A plain dictionary is converted to
        SelectOmicsConfig via SelectOmicsConfig.from_dict.  Required keys
        are 'data_path' and 'target_column'.

    Attributes
    ----------
    config : SelectOmicsConfig
        Validated configuration instance.
    results : dict
        Keyed by 'step0' through 'step3'; each value is the return dict
        from the corresponding run_step* function.
    sample_adequacy : dict or None
        Output of assess_sample_size_adequacy, populated by run().
    """

    def __init__(
        self,
        config: Union[Dict[str, Any], SelectOmicsConfig],
        on_step_complete: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        """
        Initialise the pipeline from a configuration dictionary or object.

        Parameters
        ----------
        config : dict or SelectOmicsConfig
            Pipeline configuration.  Unknown dict keys are silently ignored.
        on_step_complete : callable or None
            Optional callback invoked at the end of each completed step.
            Signature: ``callback(step_key: str, result: dict) -> None``.
            ``step_key`` is one of 'step0' through 'step3'.
            ``result`` is the return dict from the corresponding run_step*
            function.  Exceptions raised inside the callback are caught and
            logged as warnings so they never abort the pipeline.
        """
        if isinstance(config, dict):
            self.config: SelectOmicsConfig = SelectOmicsConfig.from_dict(config)
        elif isinstance(config, SelectOmicsConfig):
            self.config = config
        else:
            raise TypeError(
                f"config must be a dict or SelectOmicsConfig, "
                f"got {type(config).__name__}."
            )

        # Optional progress callback.
        self._on_step_complete: Optional[Callable[[str, Dict[str, Any]], None]] = on_step_complete

        # Ensure the output directory exists.
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # State tracker holds PipelineState snapshots for each completed step.
        self.state_tracker: DataStateTracker = DataStateTracker()

        # Per-step result dicts.  Populated by run_step* methods.
        self.results: Dict[str, Any] = {}

        # Sample adequacy assessment.  Populated by _assess_sample_size().
        self.sample_adequacy: Optional[Dict[str, Any]] = None

        # Internal data attributes.  Populated by load_data().
        self._df:               Optional[pd.DataFrame]   = None
        self._X_train:          Optional[pd.DataFrame]   = None
        self._X_test:           Optional[pd.DataFrame]   = None
        self._y_train:          Optional[np.ndarray]     = None
        self._y_test:           Optional[np.ndarray]     = None
        self._cv:               Optional[StratifiedKFold] = None
        self._n_classes:        int                       = 0
        self._class_names:      List[str]                 = []
        self._label_encoder:    Optional[LabelEncoder]   = None
        self._tuned_pipelines:  Optional[Dict[str, Any]] = None
        self._original_n_features: int                   = 0

        # Cached reference result (Step 0 single-model CV dict).
        # Passed to Steps 1-3 as ref_result_step0.
        self._ref_result_step0: Optional[Dict[str, Any]] = None

        # Final test evaluation result.
        self._final_test_eval: Optional[Dict[str, Any]] = None

        # Validation results (FeatureSetValidator output).
        self._validation_comparison: Optional[pd.DataFrame]  = None
        self._recommendation: Optional[Dict[str, Any]] = None
        self._holdout_ranking: Optional[pd.DataFrame] = None
        self._validation_all_results: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self) -> 'SelectOmicsPipeline':
        """
        Load, preprocess, and split the dataset; tune the primary model.

        Execution order:
        1. Load raw data from config.data_path via load_omics_data.
        2. Extract feature matrix and target vector via prepare_data.
        3. Handle NaN values; may auto-switch algorithm to XGB.
        4. Stratified train/test split and StratifiedKFold creation.
        5. Quick hyperparameter tune of the primary algorithm.

        Returns
        -------
        SelectOmicsPipeline
            Self (fluent interface).

        Raises
        ------
        FileNotFoundError
            If data_path does not exist.
        ValueError
            If target_column is absent or fewer than 2 classes are present.
        """
        if self.config.verbose:
            logger.info("Loading data from: %s", self.config.data_path)

        self._df = load_omics_data(self.config)

        X_raw, y_raw = prepare_data(self._df, self.config.target_column)

        # NaN detection may switch algorithm to XGB.
        X_raw, algorithm = handle_nans(X_raw, self.config.algorithm)
        if algorithm != self.config.algorithm:
            # Propagate switch so all downstream steps see the new algorithm.
            self.config.algorithm = algorithm

        (
            X_train, X_test, y_train, y_test,
            _classes, class_names, label_encoder, cv,
        ) = create_train_test_split(X_raw, y_raw, self.config)

        self._adopt_split(X_train, X_test, y_train, y_test,
                          class_names, label_encoder, cv)

        # Free raw DataFrame to reclaim memory.
        del self._df
        self._df = None
        gc.collect()

        return self

    def _adopt_split(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
        y_test: Optional[np.ndarray],
        class_names: List[str],
        label_encoder: Optional[LabelEncoder],
        cv: StratifiedKFold,
        device: Optional[str] = None,
    ) -> None:
        """
        Install a train/test split and tune the models on its training half.

        load_data() calls this with the split it makes from the source file.
        Nested CV calls it with an outer fold, so each inner pipeline starts
        from exactly the state load_data() would have produced and runs the
        same code from there.

        y_test may be None: nothing before the final test evaluation reads
        it, and an inner nested-CV pipeline is never given the outer fold's
        labels, so they cannot influence the choice being tested.
        """
        self._X_train, self._X_test = X_train, X_test
        self._y_train, self._y_test = y_train, y_test
        self._class_names = class_names
        self._label_encoder = label_encoder
        self._cv = cv
        self._n_classes = len(class_names)
        self._original_n_features = X_train.shape[1]

        if self.config.verbose:
            logger.info(
                "Dataset: %d train / %d test samples, %d features, %d classes.",
                X_train.shape[0], X_test.shape[0],
                self._original_n_features, self._n_classes,
            )
            logger.info("Tuning %s pipeline ...", self.config.algorithm)

        if device is None:
            device = 'cpu'
            if self.config.use_gpu:
                device = 'cuda' if detect_gpu() else 'cpu'
                if self.config.verbose and device == 'cuda':
                    logger.info("  GPU detected: XGBoost will use CUDA acceleration.")
                elif self.config.verbose:
                    logger.info("  No GPU detected: XGBoost will use CPU.")

        self._tuned_pipelines = quick_tune_all(
            X_train, y_train, self.config, device=device
        )

    # ------------------------------------------------------------------
    # Step 0: Reference evaluation
    # ------------------------------------------------------------------

    def run_step0_reference(self) -> 'SelectOmicsPipeline':
        """
        Evaluate a single reference model on the full training feature set.

        Establishes the baseline cross-validation AUC.  The result is
        cached in the state tracker and stored as self._ref_result_step0
        for use by Steps 1-3.

        Returns
        -------
        SelectOmicsPipeline
            Self (fluent interface).

        Raises
        ------
        RuntimeError
            If load_data() has not been called.
        """
        self._require_data()

        if self.config.verbose:
            logger.info("STEP 0: Reference Evaluation")

        step0_result = run_step0_reference(
            X_train=self._X_train,
            y_train=self._y_train,
            config=self.config,
            cv=self._cv,
            tuned_pipelines=self._tuned_pipelines,
            n_classes=self._n_classes,
            class_names=self._class_names,
            adequacy=self.sample_adequacy,
            output_dir=self.config.output_dir,
        )

        self.results['step0'] = step0_result

        # Cache the single-model CV result used as reference by later steps.
        self._ref_result_step0 = step0_result['reference_result']
        self.state_tracker.cache_reference(
            self._X_train, self.config.algorithm, step0_result
        )

        # Record state: step0 does not change the feature set.
        self.state_tracker.update(
            step_name=_STEP0_NAME,
            X_train=self._X_train.copy(),
            X_test=self._X_test.copy(),
        )

        self._fire_callback('step0', step0_result)
        return self

    # ------------------------------------------------------------------
    # Step 1: Data-based cleaning
    # ------------------------------------------------------------------

    def run_step1_cleaning(self) -> 'SelectOmicsPipeline':
        """
        Remove constant, low-variance, and highly correlated features.

        Uses the current training/test matrices from the state tracker.
        When config.enable_step1 is False, a pass-through result is stored.

        Returns
        -------
        SelectOmicsPipeline
            Self (fluent interface).
        """
        self._require_step0()
        state = self.state_tracker.get_latest()

        if self.config.verbose:
            logger.info("STEP 1: Data-Based Cleaning")

        step1_result = run_step1_cleaning(
            X_train=state.X_train,
            X_test=state.X_test,
            y_train=self._y_train,
            config=self.config,
            cv=self._cv,
            tuned_pipelines=self._tuned_pipelines,
            n_classes=self._n_classes,
            class_names=self._class_names,
            ref_result_step0=self._ref_result_step0,
            adequacy=self.sample_adequacy,
            output_dir=self.config.output_dir,
        )

        self.results['step1'] = step1_result

        self.state_tracker.update(
            step_name=_STEP1_NAME,
            X_train=step1_result['X_train_clean'],
            X_test=step1_result['X_test_clean'],
        )

        self._fire_callback('step1', step1_result)
        gc.collect()
        return self

    # ------------------------------------------------------------------
    # Step 2: Regularization-based filtering
    # ------------------------------------------------------------------

    def run_step2_regularization(self) -> 'SelectOmicsPipeline':
        """
        Apply parallel L1 and L2 regularization consensus feature selection.

        Uses the feature matrix from the most recent completed step.
        When config.enable_step2 is False, a pass-through result is stored.

        Returns
        -------
        SelectOmicsPipeline
            Self (fluent interface).
        """
        self._require_step0()
        state = self.state_tracker.get_latest()

        if self.config.verbose:
            logger.info("STEP 2: Regularization-Based Filtering")

        step2_result = run_step2_regularization(
            X_train=state.X_train,
            X_test=state.X_test,
            y_train=self._y_train,
            config=self.config,
            cv=self._cv,
            tuned_pipelines=self._tuned_pipelines,
            n_classes=self._n_classes,
            class_names=self._class_names,
            ref_result_step0=self._ref_result_step0,
            adequacy=self.sample_adequacy,
            output_dir=self.config.output_dir,
        )

        self.results['step2'] = step2_result

        self.state_tracker.update(
            step_name=_STEP2_NAME,
            X_train=step2_result['X_train_reg'],
            X_test=step2_result['X_test_reg'],
        )

        self._fire_callback('step2', step2_result)
        gc.collect()
        return self

    # ------------------------------------------------------------------
    # Step 3: Wrappers (RFECV and stability selection)
    # ------------------------------------------------------------------

    def run_step3_wrapper(self) -> 'SelectOmicsPipeline':
        """
        Apply Wrappers (RFECV with stability selection) consensus wrapper.

        Uses the feature matrix from the most recent completed step.
        Skipped automatically when fewer than MIN_FEATURES_FOR_RFECV (30)
        features remain, or when config.enable_step3 is False.

        Returns
        -------
        SelectOmicsPipeline
            Self (fluent interface).
        """
        self._require_step0()
        state = self.state_tracker.get_latest()

        if self.config.verbose:
            logger.info("STEP 3: Wrappers (RFECV + Stability Selection)")

        step3_result = run_step3_wrapper(
            X_train=state.X_train,
            X_test=state.X_test,
            y_train=self._y_train,
            config=self.config,
            cv=self._cv,
            tuned_pipelines=self._tuned_pipelines,
            n_classes=self._n_classes,
            class_names=self._class_names,
            ref_result_step0=self._ref_result_step0,
            adequacy=self.sample_adequacy,
            output_dir=self.config.output_dir,
        )

        self.results['step3'] = step3_result

        self.state_tracker.update(
            step_name=_STEP3_NAME,
            X_train=step3_result['X_train_rfecv'],
            X_test=step3_result['X_test_rfecv'],
        )

        self._fire_callback('step3', step3_result)
        gc.collect()
        return self

    # ------------------------------------------------------------------
    # Feature set validation
    # ------------------------------------------------------------------

    def validate_features(self) -> 'SelectOmicsPipeline':
        """
        Run multi-method validation across all completed pipeline steps.

        Instantiates a FeatureSetValidator and calls compare_feature_sets
        with the feature matrices from each recorded step.  Results are
        stored in self._validation_comparison and self._validation_all_results.

        Validation uses only the training split so that the held-out test
        set remains unseen by all cross-validation protocols.

        Returns
        -------
        SelectOmicsPipeline
            Self (fluent interface).

        Notes
        -----
        Requires at least one step beyond load_data to have completed.
        """
        self._require_step0()

        if self.config.verbose:
            logger.info("FEATURE SET VALIDATION")

        # Collect feature sets from each completed step.
        # Only the training split is used -- the test set must not be visible
        # to the cross-validation protocols inside FeatureSetValidator.
        feature_sets: Dict[str, pd.DataFrame] = {}
        for key, step_name in _STEP_KEY_TO_NAME.items():
            state = self.state_tracker.get_step(step_name)
            if state is None:
                continue
            feature_sets[step_name] = state.X_train

        if not feature_sets:
            return self

        y_combined = self._y_train

        upfront_severity: str = 'adequate'
        if self.sample_adequacy is not None:
            upfront_severity = self.sample_adequacy.get('overall_severity', 'adequate')

        validator = FeatureSetValidator(
            config=self.config,
            n_classes=self._n_classes,
            tuned_pipelines=self._tuned_pipelines,
        )

        comparison_df, all_results = validator.compare_feature_sets(
            feature_sets=feature_sets,
            y=y_combined,
            upfront_severity=upfront_severity,
        )

        self._validation_comparison = comparison_df
        self._validation_all_results = all_results

        # Turn the comparison into a verdict. The comparison alone says how
        # each step's panel scored; it does not say which to use, and a caller
        # reading only get_selected_features() would take the LAST step
        # regardless of whether an earlier one validated better.
        step_data: Dict[str, Dict[str, Any]] = {}
        for _name in _STEP_KEY_TO_NAME.values():
            _state = self.state_tracker.get_step(_name)
            if _state is None or _name not in feature_sets:
                continue
            # build_recommendation refuses a winner it cannot describe, so
            # every step needs an eval_result. _panel_cv_result reads Step 0's
            # 'reference_result' as well as the selection steps'
            # 'consensus_result' (reading only the latter made the whole
            # recommendation None whenever the reference panel won), and falls
            # back to this comparison's stratified CV. Without that fallback,
            # availability depended on which step won: with
            # enable_step_evaluations off only Step 0 carries an evaluation.
            step_data[_name] = {
                'step_name':   _name,
                'X_train':     _state.X_train,
                'eval_result': self._panel_cv_result(_name),
            }

        # Rank the steps on rows that did not select them, when asked. The
        # comparison above cannot: it scores every panel on the samples its
        # features came from.
        self._holdout_ranking = None
        if self.config.enable_holdout_ranking:
            try:
                self._holdout_ranking = self.rank_steps_on_holdout()
            except Exception as exc:
                # Advisory, like the recommendation itself: fall back to the
                # training-split ranking rather than losing the run.
                logger.warning(
                    "  Held-out step ranking failed (%s). Ranking on "
                    "validation from the training split instead.", exc,
                )

        self._recommendation = None
        if self._X_test is not None and step_data:
            try:
                self._recommendation = build_recommendation(
                    comparison_df, step_data, self._X_test,
                    holdout_scores=self._holdout_ranking)
            except Exception as exc:
                # Advisory: a run that produced panels is still usable without
                # a verdict, so this must not take the run down.
                logger.warning("  Could not build a recommendation: %s", exc)

        if self._recommendation is not None:
            logger.info(
                "  Recommended feature set: %s (%d features, %s %.4f) -- %s",
                self._recommendation['step_id'],
                self._recommendation['n_features'],
                'held-out AUC' if self._holdout_ranking is not None
                else 'weighted AUC',
                self._recommendation['ranking_score'],
                self._recommendation['quality'],
            )
            # Warned rather than logged at INFO: this is the case where the
            # number above is least trustworthy, and a quiet run would be the
            # only thing the user sees.
            if self._recommendation.get('selection_optimism_suspected'):
                logger.warning("  %s", self._recommendation['reason'])
        else:
            logger.info(
                "  No recommendation could be built: the validation results "
                "did not support choosing between the steps."
            )

        # Persist comparison to disk.
        if self.config.save_intermediate_results:
            out_path = self.config.output_dir / 'feature_set_validation'
            save_from_config(comparison_df, out_path, self.config)

        # Visualise validation results.
        if self.config.create_visualizations and all_results:
            plot_validation_comparison(
                all_results=all_results,
                config=self.config,
                save_path=self.config.output_dir / 'validation_comparison',
            )

        gc.collect()
        return self

    # ------------------------------------------------------------------
    # Full pipeline execution
    # ------------------------------------------------------------------

    def run(self, validate: bool = True, resume: bool = False) -> Dict[str, Any]:
        """
        Execute the complete pipeline from data loading to final evaluation.

        Calls, in order: load_data, _assess_sample_size, run_step0_reference,
        run_step1_cleaning, run_step2_regularization, run_step3_wrapper,
        and optionally validate_features and
        _run_final_test_evaluation.  Each step is conditional on the
        corresponding enable flag in config.

        Parameters
        ----------
        validate : bool
            If True, run validate_features and _run_final_test_evaluation
            after the selection steps.  Note this gates BOTH: with
            validate=False the final test evaluation is skipped even when
            config.enable_final_test_evaluation is True, and no comprehensive
            metrics JSON is written.  The skip is logged.
        resume : bool
            If True, attempt to load a checkpoint and skip already-completed
            steps.  Requires save_intermediate_results=True on the previous run.

        Returns
        -------
        dict
            self.results, extended with keys:
            'sample_adequacy'    -- assess_sample_size_adequacy output.
            'validation'         -- comparison_df and all_results (if validate).
            'recommendation'     -- the panel validation chose (if validate).
            'holdout_ranking'    -- per-step scores on rows held out of the
                                    training split (if enable_holdout_ranking).
            'final_test'         -- held-out metrics for the recommended panel,
                                    or the last step's when there is no
                                    recommendation (if validate).
            'nested_cv'          -- run_nested_cv output (if enable_nested_cv).
        """
        if self.config.verbose:
            self._print_config()

        _STEP_ORDER = ['step0', 'step1', 'step2', 'step3']

        last_completed: Optional[str] = None
        if resume:
            last_completed = self._load_checkpoint()

        if last_completed is not None and last_completed not in _STEP_ORDER:
            # The checkpoint names a step this build does not run. The version
            # stamp cannot catch every case of this: Step 4 was removed without
            # a version bump, so a 0.7.0 checkpoint can still name 'step4'.
            # Restarting is the only honest option; resuming would silently
            # skip steps that no longer exist.
            logger.warning(
                "  Checkpoint records step '%s', which this build does not "
                "run (known steps: %s). Starting from the beginning.",
                last_completed, ', '.join(_STEP_ORDER),
            )
            last_completed = None

        def _skip(step_key: str) -> bool:
            """Return True if this step was already completed in a resumed checkpoint."""
            if last_completed is None:
                return False
            return _STEP_ORDER.index(step_key) <= _STEP_ORDER.index(last_completed)

        # Data layer.
        if not _skip('step0'):
            self.load_data()
            self._assess_sample_size()

        self._run_selection_steps(_skip)

        # Evaluation and validation (always re-run, not checkpointed).
        if validate:
            self.validate_features()
            if self.config.enable_final_test_evaluation:
                self._run_final_test_evaluation()
        elif self.config.enable_final_test_evaluation:
            # Say so rather than silently ignoring the flag. validate=False
            # skips the final test evaluation by design (see the docstring),
            # but a caller who set enable_final_test_evaluation=True and gets
            # no metrics JSON has no way to tell that from a failure.
            logger.info(
                "Final test evaluation skipped: enable_final_test_evaluation "
                "is True but run(validate=False) was called, and the test "
                "evaluation runs only under validate=True. Call "
                "run(validate=True) if you want it."
            )

        # Aggregate top-level metadata into results dict.
        self.results['sample_adequacy'] = self.sample_adequacy

        if self._validation_comparison is not None:
            self.results['validation'] = {
                'comparison_df': self._validation_comparison,
                'all_results': self._validation_all_results,
            }

        if self._recommendation is not None:
            self.results['recommendation'] = self._recommendation

        if self._holdout_ranking is not None:
            self.results['holdout_ranking'] = self._holdout_ranking

        # Nested CV is opt-in and expensive: outer_cv_splits full inner
        # pipelines. It was reachable only by calling run_nested_cv() by hand,
        # so a user who set the flag and called run() got nothing and had no
        # way to tell that from a run where it had executed.
        if self.config.enable_nested_cv:
            logger.info(
                "Nested CV enabled: running %d outer folds. This refits the "
                "inner pipeline once per fold.",
                self.config.outer_cv_splits,
            )
            try:
                self.results['nested_cv'] = self.run_nested_cv()
            except Exception as exc:
                # An unbiased generalisation estimate is a bonus on top of a
                # completed run, not a precondition for one.
                logger.warning("Nested CV failed: %s", exc)

        if self._final_test_eval is not None:
            self.results['final_test'] = self._final_test_eval

        self._print_summary()

        return self.results

    def _run_selection_steps(
        self, skip: Callable[[str], bool] = lambda _key: False
    ) -> None:
        """
        Run Steps 0 to 3 in order, honouring the enable flags.

        The one definition of the selection sequence. run() calls it with a
        resume filter; each outer fold of nested CV calls it plainly, which is
        what guarantees nested CV estimates the procedure run() performs
        rather than an approximation of it.
        """
        if not skip('step0'):
            self.run_step0_reference()
            self._save_checkpoint('step0')

        if self.config.enable_step1 and not skip('step1'):
            self.run_step1_cleaning()
            self._save_checkpoint('step1')

        if self.config.enable_step2 and not skip('step2'):
            self.run_step2_regularization()
            self._save_checkpoint('step2')

        if self.config.enable_step3 and not skip('step3'):
            self.run_step3_wrapper()
            self._save_checkpoint('step3')

    def rank_steps_on_holdout(self) -> pd.DataFrame:
        """
        Score every step's panel on rows that did not select it.

        Folds are carved from the TRAINING split alone. Each fold reruns the
        configured selection steps on its own training portion and scores
        every step's panel on the rows held out from that portion, so no panel
        is ever scored on samples that chose its features. The pipeline's
        held-out test split is not touched and stays independent of the
        choice, which is what separates this from ranking on the test set.

        This is what the recommendation ranks on when enable_holdout_ranking
        is set. Validation on the training split cannot do that job: it shares
        its samples with selection, so each panel is flattered and the most
        aggressive step is flattered most. On a null control the selected
        panel validated at 0.84 and scored 0.46 on rows it had not seen.

        What is ranked is the STEP, not the exact panel. Each fold selects its
        own features, so a score says how that step performs on this data; the
        panel delivered is still the one the full training split produced.

        Returns
        -------
        pd.DataFrame
            Indexed by step name, with columns holdout_auc, holdout_std,
            holdout_folds and holdout_n_features.

        Raises
        ------
        RuntimeError
            If load_data() and Step 0 have not run.
        """
        self._require_step0()

        n_splits = int(self.config.holdout_ranking_splits)
        min_class = int(np.bincount(self._y_train,
                                    minlength=self._n_classes).min())
        if min_class < n_splits:
            # Stratification needs a member of every class in every fold.
            n_splits = max(2, min_class)
            logger.warning(
                "  Held-out ranking: smallest class has %d samples, so %d "
                "folds are used instead of %d.",
                min_class, n_splits, self.config.holdout_ranking_splits,
            )

        folds = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                random_state=self.config.random_seed)
        device = 'cpu'
        if self.config.use_gpu:
            device = 'cuda' if detect_gpu() else 'cpu'

        logger.info(
            "  Ranking steps on %d folds held out of the training split "
            "(this reruns the selection steps once per fold).", n_splits,
        )

        scores: Dict[str, List[float]] = {}
        sizes: Dict[str, List[int]] = {}

        for i, (tr_idx, ho_idx) in enumerate(
            folds.split(self._X_train, self._y_train), start=1
        ):
            X_tr, X_ho = self._X_train.iloc[tr_idx], self._X_train.iloc[ho_idx]
            y_tr, y_ho = self._y_train[tr_idx], self._y_train[ho_idx]

            inner_config = SelectOmicsConfig.from_dict({
                **self.config.to_dict(),
                'output_dir': str(self.config.output_dir
                                  / f'holdout_rank_fold_{i}'),
                'save_intermediate_results': False,
                'create_visualizations': False,
                'verbose': False,
                'enable_final_test_evaluation': False,
                'enable_nested_cv': False,
                'enable_holdout_ranking': False,
            })
            inner = SelectOmicsPipeline(inner_config)
            # y_ho is withheld: the inner run needs the held-out features to
            # subset them alongside its own, never their labels.
            inner._adopt_split(
                X_tr, X_ho, y_tr, None, self._class_names, self._label_encoder,
                make_cv_splitter(y_tr, self._n_classes, inner_config),
                device=device,
            )
            inner._assess_sample_size(advise=False)
            inner._run_selection_steps()

            # A step that skipped hands its input on unchanged, so identical
            # panels are fitted once.
            fitted: Dict[Tuple[str, ...], float] = {}
            for step_name in _STEP_KEY_TO_NAME.values():
                state = inner.state_tracker.get_step(step_name)
                if state is None:
                    continue
                cols = tuple(state.X_train.columns)
                if cols not in fitted:
                    model = inner._fit_final_model(state.X_train, y_tr)
                    fitted[cols] = float(_safe_roc_auc(
                        y_ho, model.predict_proba(state.X_test),
                        self._n_classes))
                scores.setdefault(step_name, []).append(fitted[cols])
                sizes.setdefault(step_name, []).append(len(cols))

            del inner
            gc.collect()
            try:
                inner_config.output_dir.rmdir()
            except OSError:
                pass

        rows: List[Dict[str, Any]] = []
        for step_name, values in scores.items():
            v = np.asarray(values, dtype=float)
            ok = int(np.isfinite(v).sum())
            rows.append({
                'step':               step_name,
                'holdout_auc':        float(np.nanmean(v)) if ok else float('nan'),
                'holdout_std':        float(np.nanstd(v, ddof=1)) if ok > 1 else 0.0,
                'holdout_folds':      ok,
                'holdout_n_features': float(np.mean(sizes[step_name])),
            })
        ranking = pd.DataFrame(rows).set_index('step')

        for step_name, row in ranking.iterrows():
            logger.info(
                "    %-26s held-out AUC %.4f +/- %.4f (%.0f features)",
                step_name, row['holdout_auc'], row['holdout_std'],
                row['holdout_n_features'],
            )
        return ranking

    def run_nested_cv(self) -> Dict[str, Any]:
        """
        Estimate how well the whole procedure generalises, step choice included.

        The outer loop is a StratifiedKFold with config.outer_cv_splits folds.
        Each outer fold runs the procedure run() performs, on the outer
        training portion alone: every enabled selection step (Step 3
        included), the validation comparison, and the recommendation. Every
        step's panel is then refitted on that portion and scored on the outer
        test fold, which nothing in the inner run has seen. The inner run is
        handed the outer fold's features, which the steps subset alongside
        the training matrix, but never its labels.

        The headline estimate is the RECOMMENDED panel's outer AUC, because
        that is the panel a user is told to take. The inner loop used to stop
        at Step 2 and score that panel whatever the configuration, so it
        estimated a procedure nobody ran and could not show whether the
        recommendation's choice holds up on unseen data. The per-step scores
        answer that directly: ``mean_regret`` is how far the recommended
        panel fell short of whichever step scored best on each outer fold.
        Zero means the recommendation never cost anything.

        Costs roughly outer_cv_splits full validated runs.

        Returns
        -------
        dict with keys:
            fold_aucs              : list of float, the recommended panel's
                                     AUC on each outer fold (NaN where none
                                     could be computed).
            mean_auc, std_auc      : float, over the finite fold_aucs.
            fold_features          : list of list of str, the recommended
                                     panel on each fold.
            fold_recommended_steps : list of str, the step recommended on
                                     each fold.
            feature_stability      : pd.DataFrame, the fraction of folds whose
                                     recommended panel contained each feature.
            fold_details           : pd.DataFrame, one row per fold and step:
                                     n_features, the inner validation score
                                     the choice was made on, the outer AUC,
                                     and whether the step was the recommended
                                     one and the last one.
            step_summary           : pd.DataFrame, per step: mean outer AUC,
                                     mean panel size, times recommended.
            last_step_mean_auc     : float, mean outer AUC of the last step's
                                     panel.
            best_step_mean_auc     : float, mean over folds of the best step's
                                     outer AUC. Picked with hindsight, so an
                                     upper bound rather than an estimate.
            mean_regret            : float, mean over folds of the best step's
                                     outer AUC minus the recommended panel's.
        """
        logger.info("NESTED CV (%d outer folds)", self.config.outer_cv_splits)

        # Load full data (no split yet).
        df = load_omics_data(self.config)
        X_all, y_raw = prepare_data(df, self.config.target_column)
        del df
        X_all, algorithm = handle_nans(X_all, self.config.algorithm)
        if algorithm != self.config.algorithm:
            self.config.algorithm = algorithm

        le = LabelEncoder()
        y_all = le.fit_transform(y_raw)
        class_names = [str(c) for c in le.classes_]
        n_classes = len(class_names)

        outer_cv = StratifiedKFold(
            n_splits=self.config.outer_cv_splits,
            shuffle=True,
            random_state=self.config.random_seed,
        )

        device = 'cpu'
        if self.config.use_gpu:
            device = 'cuda' if detect_gpu() else 'cpu'

        rows: List[Dict[str, Any]] = []
        fold_aucs: List[float] = []
        fold_features: List[List[str]] = []
        fold_steps: List[str] = []

        for fold_idx, (tr_idx, te_idx) in enumerate(outer_cv.split(X_all, y_all)):
            fold = fold_idx + 1
            logger.info("--- Outer fold %d/%d ---", fold, self.config.outer_cv_splits)
            X_tr, X_te = X_all.iloc[tr_idx], X_all.iloc[te_idx]
            y_tr, y_te = y_all[tr_idx], y_all[te_idx]

            inner_config = SelectOmicsConfig.from_dict({
                **self.config.to_dict(),
                'output_dir': str(self.config.output_dir / f'nested_cv_fold_{fold}'),
                # The inner run needs its panels and its recommendation, and
                # nothing it would write to disk.
                'save_intermediate_results': False,
                'create_visualizations': False,
                'verbose': False,
                'enable_final_test_evaluation': False,
                'enable_nested_cv': False,
            })
            inner = SelectOmicsPipeline(inner_config)
            inner._adopt_split(
                X_tr, X_te, y_tr, None, class_names, le,
                make_cv_splitter(y_tr, n_classes, inner_config),
                device=device,
            )
            inner._assess_sample_size(advise=False)
            inner._run_selection_steps()
            inner.validate_features()

            last_name = inner.state_tracker.get_latest().step_name
            rec = inner._recommendation
            if rec is None:
                logger.warning(
                    "  Outer fold %d: no recommendation could be built, so "
                    "the last step's panel stands in for it.", fold,
                )
            rec_name = rec['step_name'] if rec is not None else last_name

            inner_scores = (
                weighted_validation_score(inner._validation_comparison)
                if inner._validation_comparison is not None
                else pd.Series(dtype=float)
            )

            # A step that skipped passes its input through unchanged, so
            # identical panels are fitted once.
            scored: Dict[Tuple[str, ...], float] = {}
            for step_name in _STEP_KEY_TO_NAME.values():
                state = inner.state_tracker.get_step(step_name)
                if state is None:
                    continue
                cols = tuple(state.X_train.columns)
                if cols not in scored:
                    model = inner._fit_final_model(state.X_train, y_tr)
                    scored[cols] = float(_safe_roc_auc(
                        y_te, model.predict_proba(state.X_test), n_classes))
                rows.append({
                    'fold':        fold,
                    'step':        step_name,
                    'n_features':  len(cols),
                    'inner_score': float(inner_scores.get(step_name, np.nan)),
                    'outer_auc':   scored[cols],
                    'recommended': step_name == rec_name,
                    'last':        step_name == last_name,
                })

            rec_state = inner.state_tracker.get_step(rec_name)
            rec_auc = scored[tuple(rec_state.X_train.columns)]
            fold_aucs.append(rec_auc)
            fold_features.append(rec_state.X_train.columns.tolist())
            fold_steps.append(rec_name)
            logger.info(
                "  Outer fold %d AUC: %.4f (%s, %d features)",
                fold, rec_auc, rec_name, rec_state.X_train.shape[1],
            )

            del inner
            gc.collect()
            # With saving and plots off the inner run writes nothing, but its
            # constructor still creates the directory. Remove it when empty
            # rather than leave a nested_cv_fold_* shell per fold behind.
            try:
                inner_config.output_dir.rmdir()
            except OSError:
                pass

        details = pd.DataFrame(rows)
        aucs = np.asarray(fold_aucs, dtype=float)
        n_ok = int(np.isfinite(aucs).sum())
        if n_ok < len(aucs):
            # Reported as missing rather than scored 0.0: an AUC of zero is a
            # perfectly inverted classifier, and substituting it dragged the
            # mean toward a failure that never happened.
            logger.warning(
                "Nested CV: %d of %d outer folds produced no AUC and are left "
                "out of the estimate.", len(aucs) - n_ok, len(aucs),
            )
        mean_auc = float(np.nanmean(aucs)) if n_ok else float('nan')
        std_auc = float(np.nanstd(aucs, ddof=1)) if n_ok > 1 else 0.0

        best_per_fold = details.groupby('fold')['outer_auc'].max().to_numpy()
        last_per_fold = (details.loc[details['last']]
                         .sort_values('fold')['outer_auc'].to_numpy())
        regret = best_per_fold - aucs

        def _nanmean(v: np.ndarray) -> float:
            v = np.asarray(v, dtype=float)
            return float(np.nanmean(v)) if np.isfinite(v).any() else float('nan')

        step_summary = (
            details.groupby('step', sort=False)
            .agg(mean_outer_auc=('outer_auc', 'mean'),
                 mean_n_features=('n_features', 'mean'),
                 times_recommended=('recommended', 'sum'))
            .reset_index()
        )

        # Feature stability across the recommended panels. Sets, because a
        # list membership test per feature is quadratic in panel size and a
        # recommended Step 0 panel holds every feature.
        fold_sets = [set(ff) for ff in fold_features]
        stability = {
            f: sum(f in s for s in fold_sets) / len(fold_sets)
            for f in X_all.columns
        }
        stability_df = pd.DataFrame([
            {'feature': f, 'selection_frequency': v}
            for f, v in sorted(stability.items(), key=lambda x: -x[1])
        ])

        results = {
            'fold_aucs':              fold_aucs,
            'mean_auc':               mean_auc,
            'std_auc':                std_auc,
            'fold_features':          fold_features,
            'fold_recommended_steps': fold_steps,
            'feature_stability':      stability_df,
            'fold_details':           details,
            'step_summary':           step_summary,
            'last_step_mean_auc':     _nanmean(last_per_fold),
            'best_step_mean_auc':     _nanmean(best_per_fold),
            'mean_regret':            _nanmean(regret),
        }

        logger.info(
            "Nested CV complete: AUC = %.4f +/- %.4f for the recommended "
            "panel (last step %.4f; best step per fold with hindsight %.4f; "
            "mean regret %.4f)",
            results['mean_auc'], results['std_auc'],
            results['last_step_mean_auc'], results['best_step_mean_auc'],
            results['mean_regret'],
        )
        return results

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_selected_features(
        self, step: Optional[str] = None
    ) -> List[str]:
        """
        Return the feature names retained after a given pipeline step.

        Parameters
        ----------
        step : str or None
            One of 'step0', 'step1', 'step2', 'step3'.
            None returns the features from the most recently completed step.

        Returns
        -------
        list of str
            Column names of the feature matrix at the requested step.

        Raises
        ------
        ValueError
            If the requested step has not been run or the key is invalid.
        """
        if step is None:
            return self.state_tracker.get_latest().X_train.columns.tolist()

        if step not in _STEP_KEY_TO_NAME:
            raise ValueError(
                f"step must be one of {sorted(_STEP_KEY_TO_NAME)}, got '{step}'."
            )

        state = self.state_tracker.get_step(_STEP_KEY_TO_NAME[step])
        if state is None:
            raise ValueError(
                f"Step '{step}' has not been run.  Call run() or the "
                f"corresponding run_step* method first."
            )
        return state.X_train.columns.tolist()

    def get_recommended_features(self) -> List[str]:
        """
        Return the feature panel the validation actually recommends.

        get_selected_features() returns the LAST completed step, which is the
        right default for a caller who wants the end of the pipeline. It is not
        always the best panel: a later step can validate worse than an earlier
        one, and Step 3 in particular can prune a clean candidate set past the
        point where it helps. This returns whichever step won the weighted
        comparison instead.

        Returns
        -------
        list of str
            Column names of the recommended step's feature matrix.

        Raises
        ------
        RuntimeError
            If no recommendation exists. That happens when validate_features()
            has not run, or when the validation results could not support a
            choice (every protocol returned NaN, for instance). It does not
            depend on enable_step_evaluations: the recommendation rests on the
            validation comparison, which covers every step either way.
        """
        if self._recommendation is None:
            raise RuntimeError(
                "No recommendation is available. Call run(validate=True), or "
                "use get_selected_features() for the last completed step."
            )
        return list(self._recommendation['X_train'].columns)

    def get_feature_provenance(self) -> pd.DataFrame:
        """
        Return a per-feature provenance DataFrame showing survival at each step.

        Columns: feature, survived_step1, survived_step2,
                 survived_step3, n_steps_survived, first_dropped_at.

        Only covers steps that have been run. Features dropped before the
        pipeline started are not present in any step's column set.

        Returns
        -------
        pd.DataFrame, sorted by n_steps_survived descending then feature name.

        Raises
        ------
        RuntimeError
            If no selection steps have been run yet.
        """
        if 'step0' not in self.results:
            raise RuntimeError(
                "No steps have been run. Call run() or run_step0_reference() first."
            )

        # All features entering the pipeline (from Step 0 / original data).
        all_features = self._X_train.columns.tolist()
        records: Dict[str, Dict[str, Any]] = {f: {} for f in all_features}

        step_map = [
            ('step1', _STEP1_NAME, 'survived_step1'),
            ('step2', _STEP2_NAME, 'survived_step2'),
            ('step3', _STEP3_NAME, 'survived_step3'),
        ]

        for key, step_name, col in step_map:
            if key not in self.results:
                continue
            state = self.state_tracker.get_step(step_name)
            if state is None:
                continue
            survivors = set(state.X_train.columns.tolist())
            for feat in all_features:
                records[feat][col] = feat in survivors

        rows = []
        for feat, data in records.items():
            survived_cols = [col for _, _, col in step_map if col in data]
            n_survived = sum(data.get(col, True) for col in survived_cols)
            # Determine first step where feature was dropped.
            first_dropped = None
            for key, step_name, col in step_map:
                if col in data and not data[col]:
                    first_dropped = key
                    break
            row = {'feature': feat}
            row.update(data)
            row['n_steps_survived'] = n_survived
            row['first_dropped_at'] = first_dropped
            rows.append(row)

        df = pd.DataFrame(rows)
        # Sort: features that survived most steps first, then alphabetically.
        df = df.sort_values(
            ['n_steps_survived', 'feature'],
            ascending=[False, True],
        ).reset_index(drop=True)

        return df

    def save_results(
        self, output_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Persist pipeline outputs to disk.

        Saves:
        - config.json: serialised pipeline configuration.
        - selected_features: the last step's feature list, in
          config.output_format.
        - recommended_features: the recommended panel, when validation
          produced a recommendation.
        - final_<ALGO>_comprehensive_metrics.json: test-set metrics for the
          recommended panel (if the final test evaluation ran).  The algorithm
          prefix is the one actually used, so a NaN-triggered switch to XGB is
          visible in the filename.

        Parameters
        ----------
        output_dir : str or Path or None
            Override for the output directory.  Defaults to config.output_dir.
        """
        out = Path(output_dir) if output_dir is not None else self.config.output_dir
        out.mkdir(parents=True, exist_ok=True)

        # Save configuration.
        self.config.to_json(out / 'config.json')

        # Save final feature list.
        try:
            final_features = self.get_selected_features()
            feat_df = pd.DataFrame({'feature': final_features})
            save_from_config(feat_df, out / 'selected_features', self.config)
        except RuntimeError:
            pass  # No steps have completed yet.

        # Written alongside selected_features rather than in its place: that
        # file has always meant the last step's panel, and the two differ
        # exactly when a later step pruned past the point where it helped.
        # Without this file, a run that had said "use Step 2" left on disk
        # only the panel it had advised against.
        if self._recommendation is not None:
            rec_df = pd.DataFrame({'feature': self.get_recommended_features()})
            save_from_config(rec_df, out / 'recommended_features', self.config)

        # Save comprehensive metrics if the final test evaluation ran.
        if self._final_test_eval is not None:
            metrics = self._final_test_eval.get('metrics')
            if metrics is not None:
                save_comprehensive_metrics(
                    metrics=metrics,
                    output_dir=out,
                    algorithm=self.config.algorithm,
                )

        if self.config.verbose:
            logger.info("Results saved to: %s", out)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assess_sample_size(self, advise: bool = True) -> None:
        """
        Assess sample size adequacy on the training set.

        Stores the result dict in self.sample_adequacy.  Called once by
        run() before Step 0.

        advise=False skips the Step 1 redundancy advisory. Nested CV passes
        it for each outer fold: the advisory changes nothing about the run,
        and the user has already seen it once for the same data.
        """
        self._require_data()
        self.sample_adequacy = assess_sample_size_adequacy(
            X_train=self._X_train,
            y_train=self._y_train,
            cv=self._cv,
            config=self.config,
        )

        if self.config.verbose and self.sample_adequacy:
            severity = self.sample_adequacy.get('overall_severity', 'adequate')
            if severity != 'adequate':
                logger.warning(
                    "SAMPLE ADEQUACY WARNING (%s): %s",
                    severity.upper(),
                    '; '.join(self.sample_adequacy.get('warnings', [])),
                )

        if advise:
            self._assess_step1_fit()
        else:
            self.feature_redundancy = None

    def _assess_step1_fit(self) -> None:
        """
        Warn when Step 1 has nothing to remove from this data.

        Step 1's correlation filter only removes features that have a
        correlated partner.  With little redundancy present it costs runtime
        and changes nothing, which is invisible from the output alone: the run
        simply looks like it worked.  Measuring it up front and saying so is
        cheaper than a user wondering why Step 1 dropped two features out of
        ten thousand.

        Advisory only.  The configuration is never modified from here -- a run
        whose behaviour depended on an automatic switch would be harder to
        reproduce than one that did what it was told.

        Result is stored on ``self.feature_redundancy`` for callers who want it.
        """
        self.feature_redundancy = None
        if not self.config.enable_step1:
            return
        try:
            from .utils.diagnostics import assess_feature_redundancy
            self.feature_redundancy = assess_feature_redundancy(
                self._X_train,
                correlation_threshold=self.config.min_correlation_threshold,
            )
        except Exception as exc:
            logger.debug("Feature redundancy assessment skipped: %s", exc)
            return

        if not self.feature_redundancy['step1_will_act']:
            logger.warning(
                "Step 1 is enabled but this data carries little correlated "
                "redundancy: %s Step 1 will still run.",
                self.feature_redundancy['rationale'],
            )
        elif self.config.verbose:
            logger.info(
                "Feature redundancy: %s", self.feature_redundancy['rationale']
            )

    def _checkpoint_path(self) -> Path:
        """Return the path for the pipeline checkpoint file."""
        return self.config.output_dir / '.selectomics_checkpoint.pkl'

    def _source_data_fingerprint(self) -> Optional[str]:
        """
        Fingerprint the source data file, or None if it cannot be read.

        SHA-256 over the file contents, matching the convention in
        DataStateTracker._reference_key, which uses it in place of Python's
        hash() because that is randomised per process by PYTHONHASHSEED.

        Read in chunks so a large omics matrix is not held in memory twice.
        """
        try:
            path = Path(self.config.data_path)
            h = hashlib.sha256()
            h.update(str(self.config.target_column).encode('utf-8'))
            with open(path, 'rb') as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b''):
                    h.update(chunk)
            return h.hexdigest()[:16]
        except Exception as exc:
            logger.debug("Could not fingerprint source data: %s", exc)
            return None

    def _save_checkpoint(self, step_key: str) -> None:
        """
        Persist pipeline state to disk after completing a step.

        Only active when config.save_intermediate_results is True.
        The checkpoint includes results, state_tracker, and cached references.
        """
        if not self.config.save_intermediate_results:
            return
        from . import __version__ as _pkg_version   # local: avoids a cycle
        checkpoint = {
            # Stamped so a checkpoint written by a different build is never
            # silently resumed. Selection semantics change between versions
            # (0.6.1 to 0.7.0 alters results on every algorithm), so mixing
            # steps computed under two builds produces a result that matches
            # neither and says nothing about either.
            'selectomics_version': _pkg_version,
            # Resuming past step0 skips load_data entirely, so the restored
            # data comes from this file and is never re-read from disk. Without
            # this stamp, editing the source and resuming silently produced
            # results computed on the old data while appearing to describe the
            # new file. That is the same failure the version stamp above
            # guards against, from the other direction.
            'data_fingerprint':    self._source_data_fingerprint(),
            # Stamped for the same reason as the two above. A checkpoint
            # carries the tuned pipelines and every completed step's result,
            # all of them produced by one learner. Resuming under another
            # delivers the first learner's panel while the final metrics and
            # their filename record the second, so the output describes
            # neither. The algorithm here is the effective one, after any
            # NaN-driven switch to XGB, because that is what actually ran.
            'algorithm':           self.config.algorithm,
            'step_key':            step_key,
            'results':             self.results,
            'state_tracker':       self.state_tracker,
            '_ref_result_step0':   self._ref_result_step0,
            '_X_train':            self._X_train,
            '_X_test':             self._X_test,
            '_y_train':            self._y_train,
            '_y_test':             self._y_test,
            '_n_classes':          self._n_classes,
            '_class_names':        self._class_names,
            '_label_encoder':      self._label_encoder,
            '_cv':                 self._cv,
            '_tuned_pipelines':    self._tuned_pipelines,
            '_original_n_features': self._original_n_features,
            'sample_adequacy':     self.sample_adequacy,
        }
        try:
            with open(self._checkpoint_path(), 'wb') as fh:
                pickle.dump(checkpoint, fh, protocol=4)
        except Exception as exc:
            logger.warning("  Checkpoint save failed: %s", exc)

    def _load_checkpoint(self) -> Optional[str]:
        """
        Load a previously saved checkpoint if one exists.

        Returns the step_key of the last completed step, or None if no
        checkpoint exists or loading fails.

        After loading, the pipeline instance is ready to resume from the
        step after the returned step_key.
        """
        cp_path = self._checkpoint_path()
        if not cp_path.exists():
            return None
        try:
            with open(cp_path, 'rb') as fh:
                checkpoint = pickle.load(fh)

            from . import __version__ as _pkg_version
            written_by = checkpoint.get('selectomics_version')
            if written_by != _pkg_version:
                logger.warning(
                    "  Checkpoint was written by SelectOmics %s but this is "
                    "%s. Resuming would combine steps computed under two "
                    "different builds, and selection semantics change between "
                    "versions, so the result would describe neither. Starting "
                    "fresh. Delete %s to silence this.",
                    written_by or "an unversioned build", _pkg_version,
                    cp_path.name,
                )
                return None

            stored_fp = checkpoint.get('data_fingerprint')
            current_fp = self._source_data_fingerprint()
            if (stored_fp is not None and current_fp is not None
                    and stored_fp != current_fp):
                logger.warning(
                    "  Checkpoint was written for different source data than "
                    "'%s' holds now. Resuming would report results computed on "
                    "the previous data as though they described the current "
                    "file. Starting fresh. Delete %s to silence this.",
                    self.config.data_path, cp_path.name,
                )
                return None

            stored_algo = checkpoint.get('algorithm')
            if stored_algo is not None and stored_algo != self.config.algorithm:
                logger.warning(
                    "  Checkpoint was written for algorithm %s but this run "
                    "is configured for %s. Resuming would deliver the panel "
                    "%s selected while the metrics recorded %s, so the result "
                    "would describe neither. Starting fresh. Delete %s to "
                    "silence this.",
                    stored_algo, self.config.algorithm,
                    stored_algo, self.config.algorithm, cp_path.name,
                )
                return None

            self.results             = checkpoint['results']
            self.state_tracker       = checkpoint['state_tracker']
            self._ref_result_step0   = checkpoint['_ref_result_step0']
            self._X_train            = checkpoint['_X_train']
            self._X_test             = checkpoint['_X_test']
            self._y_train            = checkpoint['_y_train']
            self._y_test             = checkpoint['_y_test']
            self._n_classes          = checkpoint['_n_classes']
            self._class_names        = checkpoint['_class_names']
            self._label_encoder      = checkpoint['_label_encoder']
            self._cv                 = checkpoint['_cv']
            self._tuned_pipelines    = checkpoint['_tuned_pipelines']
            self._original_n_features = checkpoint['_original_n_features']
            self.sample_adequacy     = checkpoint['sample_adequacy']
            last_step = checkpoint['step_key']
            logger.info("  Checkpoint restored: resuming after '%s'.", last_step)
            return last_step
        except Exception as exc:
            logger.warning("  Checkpoint load failed (%s). Starting fresh.", exc)
            return None

    def _run_final_test_evaluation(self) -> None:
        """
        Evaluate the recommended feature set on the held-out test set.

        Scores the panel get_recommended_features() returns, falling back to
        the last completed step when there is no recommendation. The
        recommendation is chosen on the training split alone, so this still
        spends the held-out set exactly once. Scoring the last step instead,
        as this used to, reported metrics for a panel the user had just been
        told not to use whenever the two differed.

        Clones the tuned pipeline for the configured algorithm, refits on
        the chosen training panel, predicts on the test set, and assembles
        comprehensive metrics.  Saves metrics and plots when the
        corresponding config flags are True.

        Results are stored in self._final_test_eval, including 'step_name'
        and 'panel' ('recommended' or 'last step') so the evaluated panel is
        never ambiguous.
        """
        self._require_step0()

        rec = self._recommendation
        if rec is not None:
            X_train_final = rec['X_train']
            X_test_final  = rec['X_test']
            step_label    = rec['step_name']
            panel         = 'recommended'
        else:
            final_state   = self.state_tracker.get_latest()
            X_train_final = final_state.X_train
            X_test_final  = final_state.X_test
            step_label    = final_state.step_name
            panel         = 'last step'

        alg = self.config.algorithm

        # The training-split CV result for the SAME panel, so the reported
        # generalisation gap compares like with like. This used to iterate
        # reversed(['step3', 'step2', 'step1', 'step0']), which runs Step 0
        # first, and stop at the first match: every gap was measured from the
        # full-feature reference AUC, whichever panel was being tested.
        training_result = self._panel_cv_result(step_label)
        if training_result is None:
            training_result = {'mean_auc': float('nan'), 'std_auc': float('nan')}

        agreement, agreement_label = self._final_agreement(
            upto=_STEP_NAME_TO_KEY.get(step_label))

        if self.config.verbose:
            logger.info("FINAL TEST EVALUATION")
            logger.info(
                "Fitting %s on the %s panel, %s: %d features (%d samples) ...",
                alg, panel, step_label,
                X_train_final.shape[1], X_train_final.shape[0],
            )

        final_model = self._fit_final_model(X_train_final, self._y_train)

        test_proba: np.ndarray = final_model.predict_proba(X_test_final)
        test_pred:  np.ndarray = final_model.predict(X_test_final)

        test_auc = _safe_roc_auc(self._y_test, test_proba, self._n_classes)

        # _safe_roc_auc returns nan so that a fold loop can skip a fold. This
        # is not a fold loop: there is nothing to skip, and the value becomes
        # the headline held-out number. Say so rather than publishing a bare
        # nan, and name the cause, which is always a class the test split
        # does not contain.
        if np.isnan(test_auc):
            _present = set(np.unique(self._y_test).tolist())
            _missing = [self._class_names[i] for i in range(self._n_classes)
                        if i not in _present]
            logger.warning(
                "  Held-out test AUC is undefined for this split: %s. "
                "A macro one-vs-rest AUC needs every class present in the "
                "test set. The panel is unchanged and its other test metrics "
                "are valid, but this run has no held-out AUC and no "
                "meaningful generalisation gap.",
                f"class(es) {_missing} absent from the test set" if _missing
                else "fewer than 2 classes present in the test set",
            )

        metrics = build_comprehensive_metrics(
            y_test=self._y_test,
            y_pred=test_pred,
            y_proba=test_proba,
            class_names=self._class_names,
            n_classes=self._n_classes,
            training_result=training_result,
            step_label=step_label,
            algorithm=alg,
            n_models=self.config.n_consensus_models,
            achieved_agreement=agreement,
            agreement_label=agreement_label,
            original_n_features=self._original_n_features,
            final_n_features=X_train_final.shape[1],
        )

        gen_label = summarise_generalization(
            training_auc=training_result['mean_auc'],
            test_auc=test_auc,
            verbose=self.config.verbose,
        )

        self._final_test_eval = {
            'metrics':      metrics,
            'test_proba':   test_proba,
            'test_pred':    test_pred,
            'test_auc':     test_auc,
            'gen_label':    gen_label,
            'final_model':  final_model,
            'step_name':    step_label,
            'panel':        panel,
            'n_features':   int(X_train_final.shape[1]),
        }

        if self.config.save_intermediate_results:
            save_comprehensive_metrics(
                metrics=metrics,
                output_dir=self.config.output_dir,
                algorithm=alg,
            )

        if self.config.create_visualizations:
            # Learning curve and train vs test ROC.
            plot_learning_curve_and_roc(
                final_model=final_model,
                X_train=X_train_final,
                y_train=self._y_train,
                y_test=self._y_test,
                test_proba=test_proba,
                test_auc=test_auc,
                training_result=training_result,
                algorithm=alg,
                n_classes=self._n_classes,
                config=self.config,
                save_path=self.config.output_dir / 'learning_curve_roc',
            )

            # Per-class ROC and confusion matrix.
            y_test_binary = binarize_labels(self._y_test, self._n_classes)
            cm = pd.DataFrame(
                confusion_matrix(self._y_test, test_pred),
                index=self._class_names,
                columns=self._class_names,
            )
            plot_per_class_roc_and_confusion(
                y_test=self._y_test,
                y_test_binary=y_test_binary,
                test_proba=test_proba,
                cm_df=cm,
                class_names=self._class_names,
                algorithm=alg,
                config=self.config,
                save_path=self.config.output_dir / 'roc_confusion',
            )

    def _fit_final_model(self, X_train: pd.DataFrame, y_train: np.ndarray):
        """
        Fit the delivered model on a panel: the tuned pipeline, refitted,
        and calibrated when config.calibrate_probabilities is set.

        Shared by the final test evaluation and by nested CV's outer-fold
        scoring, so both measure the same model.
        """
        model = clone(self._tuned_pipelines[self.config.algorithm])
        model.fit(X_train, y_train)

        # Optional isotonic calibration. CalibratedClassifierCV cross-fits
        # internally, so no separate held-out set is needed.
        if getattr(self.config, 'calibrate_probabilities', False):
            min_class = int(np.bincount(y_train, minlength=self._n_classes).min())
            cal_cv = min(3, max(2, min_class))
            model = CalibratedClassifierCV(model, method='isotonic', cv=cal_cv)
            model.fit(X_train, y_train)
        return model

    def _panel_cv_result(
        self, step_name: str, use_validation: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        The training-split CV result that describes one step's panel.

        A step that skipped or was disabled passes its input through
        unchanged, and the step modules file Step 0's reference evaluation as
        its consensus_result. That result describes the full feature set, not
        the panel, so reading it directly put Step 0's AUC against a skipped
        Step 3: in the summary table, in the recommendation's result, and as
        the training AUC the held-out gap was measured from. This walks back
        to the step that actually decided the panel instead.

        When that step carries no evaluation of its own
        (enable_step_evaluations=False), use_validation falls back to the
        validation comparison's stratified CV of the same panel.
        """
        names = list(_STEP_KEY_TO_NAME.values())
        if step_name not in names:
            return None
        for name in reversed(names[:names.index(step_name) + 1]):
            r = self.results.get(_STEP_NAME_TO_KEY[name])
            # Absent: run() never called the step, so its panel is the
            # previous one's. Skipped or disabled: it passed that panel on.
            if not r or r.get('consensus_outcome') in ('skipped', 'disabled'):
                continue
            cr = r.get('consensus_result') or r.get('reference_result')
            if cr is not None and 'mean_auc' in cr:
                return cr
            break
        if use_validation:
            return ((self._validation_all_results or {})
                    .get(step_name, {}).get('stratified_cv'))
        return None

    def _final_agreement(self, upto: Optional[str] = None) -> Tuple[float, str]:
        """
        How much model agreement the delivered panel actually carries.

        Reads the last selection step that ran a consensus, at or before
        ``upto`` (a step key) when given, so a recommended Step 2 panel reports
        Step 2's agreement rather than a later step's. This is an outcome,
        not a setting: the caller states the panel size they need via
        ``min_features_floor``, and the relaxation reports what that cost in
        agreement. Because relaxation always begins at unanimity, the level
        recorded is the strongest at which the panel holds.

        Returns
        -------
        (float, str)
            Agreement fraction in [0, 1] and its plain-language label.
            (0.0, 'not run') when no step recorded a consensus.
        """
        keys = ['step3', 'step2', 'step1']
        if upto is not None:
            keys = [k for k in keys if k <= upto]
        # Step 1 is included: when Steps 2 and 3 are disabled or pass
        # through, its filter agreement is what decided the panel.
        for key in keys:
            result = self.results.get(key)
            if not result:
                continue
            outcome = result.get('consensus_outcome')
            # 'disabled' and 'skipped' mean this step did not decide the
            # panel, so keep walking back to the one that did.
            if outcome in (None, 'disabled', 'skipped'):
                continue
            return (float(result.get('agreement', 0.0)),
                    str(result.get('agreement_label', 'unknown')))
        return 0.0, 'not run'

    def _print_config(self) -> None:
        """Log a formatted summary of the pipeline configuration."""
        logger.info("SelectOmics Pipeline Configuration")
        logger.info("  Data path:          %s", self.config.data_path)
        logger.info("  Target column:      %s", self.config.target_column)
        logger.info("  Algorithm:          %s", self.config.algorithm)
        logger.info("  Consensus models:   %d", self.config.n_consensus_models)
        logger.info("  Min features:       %d", self.config.min_features_floor)
        logger.info(
            "  Min consensus:      %s",
            'none (relax as needed)' if self.config.min_consensus is None
            else f"{self.config.min_consensus:.0%}",
        )
        logger.info("  Random seed:        %d", self.config.random_seed)
        logger.info("  Test size:          %s", self.config.test_size)
        logger.info("  Output dir:         %s", self.config.output_dir)
        logger.info("  Output format:      %s", self.config.output_format)
        logger.info("  Plot format:        %s", self.config.plot_format)
        logger.info(
            "  Steps enabled:      %s%s%s",
            'S1 ' if self.config.enable_step1 else '-- ',
            'S2 ' if self.config.enable_step2 else '-- ',
            'S3 ' if self.config.enable_step3 else '-- ',
        )

    def _print_summary(self) -> None:
        """Print the pipeline-level feature count and AUC summary table."""
        step_records: List[Dict[str, Any]] = []

        # Reference.
        if 'step0' in self.results:
            r0 = self.results['step0']
            step_records.append({
                'step_name':   _STEP0_NAME,
                'n_features':  r0.get('n_features', self._original_n_features),
                'eval_result': r0.get('reference_result'),
            })

        # Steps 1-3.
        for key, step_name in [
            ('step1', _STEP1_NAME),
            ('step2', _STEP2_NAME),
            ('step3', _STEP3_NAME),
        ]:
            if key not in self.results:
                continue
            state = self.state_tracker.get_step(step_name)
            if state is None:
                continue
            # Not r['consensus_result']: a skipped step files Step 0's
            # reference evaluation there, which put the full-feature AUC on
            # the row of a panel that is not the full feature set.
            eval_r = self._panel_cv_result(step_name, use_validation=False)
            step_records.append({
                'step_name':   step_name,
                'n_features':  state.X_train.shape[1],
                'eval_result': eval_r,
            })

        if not step_records:
            return

        summary_df = build_pipeline_summary_df(
            step_results=step_records,
            original_n_features=self._original_n_features,
            algorithm=self.config.algorithm,
        )

        if self.config.verbose:
            logger.info("PIPELINE SUMMARY")
            logger.info("%s", summary_df.to_string(index=False))

            # The summary lists every step; it does not say which to use. A
            # reader who stops here would take the last row, and the last row
            # is not always the panel that validated best.
            if self._recommendation is not None:
                rec = self._recommendation
                logger.info(
                    "RECOMMENDED: %s -- %d features, weighted AUC %.4f, "
                    "quality %s (%s)",
                    rec['step_id'], rec['n_features'], rec['weighted_auc'],
                    rec['quality'], rec['reason'],
                )
                _last = summary_df.iloc[-1].get('Step', '')
                if _last and str(rec['step_id']) not in str(_last):
                    logger.info(
                        "  Note: this is NOT the last step. "
                        "get_selected_features() returns the last step's "
                        "panel; get_recommended_features() returns this one."
                    )

        # Save and plot unconditionally (gated on their own flags).
        if self.config.save_intermediate_results:
            save_from_config(
                summary_df,
                self.config.output_dir / 'pipeline_summary',
                self.config,
            )

        if self.config.create_visualizations:
            plot_pipeline_summary(
                summary_df=summary_df,
                algorithm=self.config.algorithm,
                config=self.config,
                save_path=self.config.output_dir / 'pipeline_summary',
            )

        # Feature provenance table.
        try:
            prov_df = self.get_feature_provenance()
            if self.config.save_intermediate_results:
                save_from_config(
                    prov_df,
                    self.config.output_dir / 'feature_provenance',
                    self.config,
                )
            if self.config.verbose:
                logger.info("Feature provenance (top 10):")
                logger.info("%s", prov_df.head(10).to_string(index=False))
        except RuntimeError:
            pass  # No steps have completed yet.

    def _fire_callback(self, step_key: str, result: Dict[str, Any]) -> None:
        """
        Invoke the on_step_complete callback safely.

        Catches all exceptions so that a misbehaving callback never aborts
        the pipeline.

        Parameters
        ----------
        step_key : str
            Step identifier, e.g. 'step0'.
        result : dict
            The return dict from the corresponding run_step* function.
        """
        if self._on_step_complete is None:
            return
        try:
            self._on_step_complete(step_key, result)
        except Exception as exc:
            logger.warning(
                "on_step_complete callback raised an exception for '%s': %s",
                step_key, exc,
            )

    # ------------------------------------------------------------------
    # Internal guards
    # ------------------------------------------------------------------

    def _require_data(self) -> None:
        """Raise RuntimeError if load_data() has not been called."""
        if self._X_train is None:
            raise RuntimeError(
                "Data not loaded.  Call load_data() before this method."
            )

    def _require_step0(self) -> None:
        """Raise RuntimeError if run_step0_reference() has not been called."""
        self._require_data()
        if 'step0' not in self.results:
            raise RuntimeError(
                "Step 0 has not been run.  Call run_step0_reference() "
                "before this method."
            )
