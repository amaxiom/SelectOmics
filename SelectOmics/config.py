"""
SelectOmics/config.py

Configuration dataclass for the SelectOmics pipeline.
All defaults match the SELECTOMICS_CONFIG dictionary in the notebook (Cell 3)
and the cv split logic in Cell 5.

Standard library only -- no third-party imports.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union


# ---------------------------------------------------------------------------
# Valid option sets used in __post_init__ validation.
# ---------------------------------------------------------------------------

_VALID_ALGORITHMS: frozenset = frozenset({'LR', 'XGB', 'RF', 'SVM'})
_VALID_INPUT_FORMATS: frozenset = frozenset(
    {'auto', 'csv', 'tsv', 'excel', 'parquet', 'hdf5', 'feather', 'json'}
)
_VALID_OUTPUT_FORMATS: frozenset = frozenset(
    {'csv', 'tsv', 'excel', 'parquet', 'hdf5', 'feather', 'json'}
)
_VALID_PLOT_FORMATS: frozenset = frozenset({'png', 'svg', 'pdf', 'tiff', 'jpeg'})
_VALID_EXCEL_ENGINES: frozenset = frozenset({'openpyxl', 'xlsxwriter'})


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class SelectOmicsConfig:
    """
    Pipeline configuration for SelectOmics.

    Required fields
    ---------------
    data_path : str or Path
        Path to the input data file.
    target_column : str
        Name of the column containing class labels.

    All other fields are optional and default to the values used in the
    SelectOmics notebook (SELECTOMICS_CONFIG, Cell 3; cv split logic, Cell 5).
    """

    # ------------------------------------------------------------------
    # Required -- no defaults; must be supplied by the caller.
    # ------------------------------------------------------------------
    data_path: Union[str, Path]
    target_column: str

    # ------------------------------------------------------------------
    # Schema versioning
    # ------------------------------------------------------------------
    # Used for forward/backward compatibility when loading serialised configs.
    # Migration logic in from_dict() maps older schema versions to current fields.
    schema_version: str = "1.0"

    # ------------------------------------------------------------------
    # Data input settings
    # ------------------------------------------------------------------
    # Auto-detection inspects the file extension; specify explicitly when
    # the extension is ambiguous or missing.
    input_format: str = 'auto'

    # HDF5 dataset key (ignored for all other formats).
    hdf5_key: str = 'data'

    # Excel sheet to read: integer index (0-based) or sheet name string.
    # Ignored for all non-Excel formats.
    excel_sheet_name: Union[str, int] = 0

    # ------------------------------------------------------------------
    # Output settings
    # ------------------------------------------------------------------
    output_dir: Union[str, Path] = 'selectomics_results'
    output_format: str = 'csv'

    # Engine used when output_format == 'excel'.
    # openpyxl supports .xlsx; xlsxwriter is an alternative with richer
    # formatting but does not support reading.
    excel_output_engine: str = 'openpyxl'

    # Save per-step feature matrices and evaluation summaries to disk.
    save_intermediate_results: bool = True

    # Generate and save all diagnostic plots.
    create_visualizations: bool = True

    # Raster/vector format for all saved figures.
    # jpeg uses quality=95; all others use dpi=300.
    plot_format: str = 'png'

    # Print step-by-step progress and summary tables.
    verbose: bool = False

    # ------------------------------------------------------------------
    # Core algorithm settings
    # ------------------------------------------------------------------
    # Primary classification algorithm used throughout the pipeline.
    # XGB is automatically selected when the dataset contains missing values.
    algorithm: str = 'XGB'

    # Number of bootstrap-seeded consensus models per selection step.
    # Higher counts improve stability at the cost of runtime.
    n_consensus_models: int = 10

    # Reproducibility seed for all stochastic operations.
    random_seed: int = 42

    # ------------------------------------------------------------------
    # Train / test split
    # ------------------------------------------------------------------
    # Fraction of samples reserved for the held-out test set.
    # Matches the hardcoded value in notebook Cell 5.
    test_size: float = 0.25

    # ------------------------------------------------------------------
    # Cross-validation settings
    # ------------------------------------------------------------------
    # Number of stratified folds.  None triggers automatic selection:
    #   cv_splits = min(max_cv_splits, max(min_cv_splits, min_class_count))
    # where min_class_count is the size of the smallest class in X_train.
    cv_splits: Optional[int] = None
    min_cv_splits: int = 3
    max_cv_splits: int = 20

    # ------------------------------------------------------------------
    # Hyperparameter tuning
    # ------------------------------------------------------------------
    # RandomizedSearchCV iteration budget for the initial quick-tune pass.
    quick_tune_iterations: int = 50

    # Attempt GPU acceleration for XGBoost when True.
    # Requires CUDA-capable GPU and xgboost >= 2.0.  Falls back to CPU
    # silently when the hardware or driver is unavailable.
    use_gpu: bool = True

    # ------------------------------------------------------------------
    # Parallelism
    # ------------------------------------------------------------------
    # Threads each estimator may use.  -1 uses every core, 1 is strictly
    # single-threaded, and any positive integer sets an explicit cap.
    #
    # This is safe to change: XGBoost ('hist'), RandomForest,
    # permutation_importance, and RFECV all produce bit-identical output
    # regardless of thread count, because each is seeded independently of how
    # the work is divided.  Reproducibility comes from random_seed, not from
    # serial execution.  (Moving XGBoost to the GPU via use_gpu is a different
    # matter and does perturb results slightly; see that field.)
    #
    # Set to 1 when running several pipelines concurrently yourself, so the
    # processes do not oversubscribe the machine and slow each other down.
    n_jobs: int = -1

    # ------------------------------------------------------------------
    # Step enable flags
    # ------------------------------------------------------------------
    enable_step1: bool = True   # Data-based cleaning (variance/correlation)
    enable_step2: bool = True   # Model-based filters (L1/L2 regularization)
    enable_step3: bool = True   # Wrappers (RFECV + stability)

    # ------------------------------------------------------------------
    # Evaluation panel flags
    # ------------------------------------------------------------------
    enable_step_evaluations: bool = True         # Per-step consensus model report
    enable_final_test_evaluation: bool = True      # Held-out test set metrics

    # ------------------------------------------------------------------
    # Feature selection -- Step 1 thresholds
    # ------------------------------------------------------------------
    # Joint variance/correlation grid search upper bounds.
    # A 11x11 grid is swept from 0 to these maxima; the (variance, correlation)
    # pair that maximises the drop-intersection size is selected.
    max_variance_threshold: float = 0.30
    min_correlation_threshold: float = 0.70   # note: semantically a MAX threshold

    # Should Step 1's correlation filter know about the outcome?
    #
    # True (default) uses the pooled WITHIN-class correlation to decide what is
    # redundant, and the correlation ratio (eta-squared) to decide which member
    # of a redundant pair to keep.
    #
    # The class-unaware filter cannot distinguish two probes measuring the same
    # thing from two independent markers of the same disease: both are
    # correlated, and it deletes one either way. That is not a corner case in
    # omics, where co-regulation is the norm. Measured on synthetic data
    # (n=500, p=200, 20 informative, rho=0.10): at signal_strength 1.2 the mean
    # correlation among informative features is 0.584 and all 20 survive; at
    # 2.0 it is 0.797 and exactly ONE survives while all 180 noise features are
    # retained. The collapse happens at the 0.70 default threshold.
    #
    # Centring each feature on its class mean removes the between-class
    # covariance, leaving only association the label does not explain. Genuine
    # duplicates stay correlated; co-regulated markers do not.
    #
    # Set False to restore the class-UNAWARE correlation matrix and the
    # variance tie-break used up to 0.7.0. Note this is not bit-identical
    # to that release: where two features had exactly equal variance the
    # old code let column order decide, and the survivor is now settled by
    # name so the result cannot depend on how the matrix was assembled.
    # Only exact ties differ, and only in which of two equivalent features
    # survives.
    class_aware_correlation: bool = True

    # ------------------------------------------------------------------
    # Feature selection -- minimum retained features
    # ------------------------------------------------------------------
    # Absolute floor on the number of features any selection step may return.
    # A safety net against collapsing to nothing, not a retention target: each
    # step already has its own retention target and percentile search for that.
    #
    # This is the setting that determines panel size in practice. Every step
    # starts at unanimity and relaxes its consensus requirement one vote at a
    # time until the floor is met, so the floor decides how many features come
    # out and the vote level reached (reported as 'votes_required' and
    # 'agreement_label') records how much agreement the data supported at that
    # size. min_consensus is the brake on that descent.
    #
    # It was previously max(10, 5% of the step's input), which scaled the floor
    # with the input of a step whose purpose is aggressive reduction -- 100
    # features at p=2000, 500 at p=10000. Because the consensus can only ever
    # return features some model actually selected, such a floor is unreachable
    # and forces the vote requirement all the way down to 1.
    #
    # Measured with everything else held fixed at the configuration-search
    # winner, over all six development scenarios at five seeds, so the floor
    # is the only thing varying:
    #
    #     floor  5:  composite 0.716  F1 0.624  KI 0.490   162 s
    #     floor  8:  composite 0.784  F1 0.714  KI 0.614   160 s
    #     floor 10:  composite 0.808  F1 0.745  KI 0.660   159 s
    #     floor 15:  composite 0.820  F1 0.760  KI 0.683   157 s   <- default
    #     floor 20:  composite 0.819  F1 0.763  KI 0.674   327 s
    #     floor 25:  composite 0.820  F1 0.764  KI 0.672   389 s
    #
    # Monotonic to 15, then flat: 15, 20 and 25 differ by under 0.001. What
    # separates them is runtime. Floors of 20 and above pass enough features
    # to wake Step 3, which is by far the most expensive step (RFECV
    # eliminates one feature at a time, inside a binary search, across every
    # consensus model), so they cost two to two and a half times as much for
    # no measurable gain.
    #
    # The improvement from 10 to 15 comes almost entirely from one scenario,
    # omics_imbalanced (F1 0.818 -> 0.918); the other five tie exactly and
    # none is materially harmed.
    #
    # LOWER IT when you expect a compact signal: a floor above the true number
    # of informative features has to pad the panel.
    #
    # A floor larger than the consensus ceiling cannot be honoured. It is
    # capped at the number of features some model actually selected, and the
    # step logs the capping when it happens.
    #
    # Earlier releases used max(10, 5% of the step's input), which scaled with
    # the input of a step whose purpose is aggressive reduction (500 features
    # at p=10000). Such a floor is unreachable, and demanding it drove the vote
    # requirement to its minimum on every run.
    min_features_floor: int = 15

    # ------------------------------------------------------------------
    # Feature selection -- minimum acceptable agreement
    # ------------------------------------------------------------------
    # Lowest fraction of models that may be required to agree before a result
    # is rejected as too weak.
    #
    # Every step starts at unanimity and lowers the vote requirement one step
    # at a time until the feature floor is met. This is the brake on that
    # descent:
    #   min_consensus = 1.0   unanimity or nothing
    #   min_consensus = 0.8   strong agreement or nothing
    #   min_consensus = 0.4   the default; stop before agreement gets thin
    #   min_consensus = None  whatever it takes to reach the floor
    #
    # 0.4 rather than None, and rather than anything stricter, is measured.
    # Across two search phases the response has a clear interior optimum:
    #
    #   Phase 1 (205 configs, 2 scenarios, 3 seeds), composite by setting:
    #     None 0.751 | 0.4 0.784 | 0.6 0.743 | 0.8 0.599 | 1.0 0.503
    #   Phase 2 (30 configs, 3 scenarios, 5 seeds), mean F1 by setting:
    #     None 0.475 | 0.4 0.563 | 0.6 0.495
    #
    # In Phase 2 the 0.4 setting wins on every scenario individually, not
    # merely on the mean: adversarial_easy 0.066 vs 0.059, omics_standard
    # 0.938 vs 0.836, omics_tiny_n 0.684 vs 0.530. It was the strongest
    # parameter in Phase 1 (Spearman r = -0.746, p < 0.0001) and the second
    # strongest in Phase 2 (r = -0.422, p = 0.020).
    #
    # Read the direction carefully: this is STRICTER than the old behaviour,
    # not looser. With no floor the relaxation can bottom out at one vote in
    # ten, and the measurement says that costs both accuracy and
    # reproducibility. Requiring unanimity is worse still -- F1 0.340 and
    # Kuncheva 0.127 at min_consensus 1.0 -- because almost nothing survives.
    # Enforcing agreement helps; demanding all of it does not.
    #
    # When this and min_features_floor conflict, agreement wins and the step
    # reports 'consensus_limited' with both numbers, so the conflict is
    # visible rather than silently resolved.
    #
    # This REPLACES consensus_threshold, which set where the descent STARTED
    # and was measured to be inert: because the search stops at whichever level
    # first satisfies the floor, starting lower only skipped rungs on the way
    # to the same answer. On omics_standard and omics_genomics at two seeds,
    # thresholds of 0.4, 0.6, 0.8 and 1.0 returned the identical feature SET,
    # not merely the same count. It also corrupted the diagnostic: relaxation
    # walks only downward, so starting at 4/10 never tested 6/10 and reported
    # 4 for a panel that held at 6.
    min_consensus: Optional[float] = 0.4

    # ------------------------------------------------------------------
    # Feature selection -- Step 2 retention target
    # ------------------------------------------------------------------
    # Fraction of Step 2 input features each regularization stage targets for
    # retention.  Binary search converges to the importance rank threshold that
    # produces this many features per stage; the L1 and L2 stages are then
    # intersected, so the step as a whole returns fewer than this.
    #
    # Whether the target is reachable depends on the model: it is capped at the
    # number of features the stage model gave a non-zero importance to, which
    # for tree algorithms on wide data is far below any sane target (see
    # _rank_keep_mask).  Step 2 logs when the target is saturated that way.
    # Fraction of the REACHABLE pool that Step 2 keeps per stage, where the
    # pool is the set of features the stage models actually used (non-zero
    # importance).  It was previously a fraction of ALL features, which is
    # unreachable on wide data and made the knob inert; see step2_regularization.
    #
    # Default 1.0 = keep the whole pool. That is what the inert version
    # effectively did, so this preserves measured behaviour rather than
    # silently tightening it. Lower it to cut harder within the pool.
    # Measured on omics_standard seed 0: 0.4 -> 6 features (F1 0.750),
    # 0.8 -> 7 (0.824), 1.0 -> 8 (0.889).  Must be in (0, 1].
    step2_target_retention: float = 1.0

    # Column-sampling rate for Step 2's XGB stage models.
    #
    # This is what sets the CEILING on Step 2's output, and it was not
    # previously reachable from config. A tree assigns non-zero importance
    # only to features it actually split on, and _rank_keep_mask refuses to
    # select beyond that set, so the size of that set is the most Step 2 can
    # ever return. Measured at 0.3 on the synthetic scenarios, one XGB stage
    # fit used 21 of 2000 features (1.1%) and 30 of 10000 (0.3%); L1 logistic
    # regression on the same data used 66 and 293.
    #
    # LOWER it to widen the pool. This is counter-intuitive and was measured
    # rather than assumed: at colsample 1.0 every tree sees every feature and
    # keeps splitting on the same strongest few, so FEWER distinct features are
    # ever used. A low rate forces each tree onto a different random subset, so
    # more features get touched across the ensemble.
    #
    #     usable pool, one XGB stage fit, seed 0
    #     colsample          0.1    0.3    0.6    1.0
    #     omics_standard      61     21     21     20   (p=2000)
    #     omics_genomics      70     30     25     23   (p=10000)
    #
    # Widening is not free: at 0.1 the extra features are largely noise, and
    # measured Step 2 precision fell from 1.000 to 0.857. The 0.3 default sits
    # near the knee. step2_target_retention then decides where within the pool
    # to cut.  No effect for LR/RF/SVM stages.  Must be in (0, 1].
    step2_stage_colsample: float = 0.3

    # What job Step 2 is doing, which decides how hard it should cut.
    #
    # Step 2's optimal aggression depends on whether it is the LAST step. It
    # currently always behaves as though it is, because nothing tells it
    # otherwise, and its output is consequently a narrow high-precision panel
    # that leaves Step 3 nothing to eliminate (and below the Step 3 gate, so
    # Step 3 skips).
    #
    #   'terminal'  -- Step 2 optimises the panel it delivers. The default, and
    #                  measured best on 4 of 6 scenarios.
    #   'prefilter' -- Step 2 delivers CANDIDATES, deliberately overshooting the
    #                  Step 3 gate so Step 3 engages and does the precision
    #                  work. Relaxes the agreement brake to get there.
    #
    # Measured, the closest available approximation to 'prefilter' against the
    # default across six scenarios: it wins outright on omics_multiclass
    # (F1 1.000 against 0.691, precision and recall both perfect), ties on
    # omics_high_dim, and loses the other four, mean 0.725 against 0.803. It is
    # therefore an option rather than a default.
    #
    # 'prefilter' with enable_step3=False is rejected: that combination hands
    # back an unfiltered candidate set with nothing to prune it.
    step2_role: str = "terminal"

    # Add a UNION rung below the intersection ladder in Steps 2 and 3.
    #
    # The intersection across stages is enforced at EVERY rung of the normal
    # ladder: even at one vote, a feature must have been kept by some replicate
    # in every stage. That is what sets the ceiling, and no floor can exceed
    # it -- which is why asking Step 2 for 60 features returned 60 in only 5 of
    # 50 measured runs. With this on, a ladder over "kept by r replicates in AT
    # LEAST ONE stage" is tried when the intersection cannot reach the floor.
    #
    # This is the sensitivity switch: measured over 10 signal-bearing
    # scenarios at 5 seeds it moves recall 0.844 to 0.897 and precision 0.902
    # to 0.726, for four more features and 1.8x the runtime, and returns 17.0
    # false positives on a null control against 9.4.
    #
    # Off by default because the trade runs against this package's own regime.
    # It pays where recall is genuinely short, which is the lower-p end
    # (p=200: recall 0.62 to 0.80 at precision 1.000; p=1000: 0.80 to 0.947
    # for 9 precision points). It is a loss where p is large and signal sparse
    # (p=5000, 5 informative: precision 0.752 to 0.247; p=10 000: 1.000 to
    # 0.655 for no recall at all, recall already being 1.000).
    #
    # Never combine it with step2_role='prefilter': that pairing measured
    # precision 0.318 and F1 0.411, the worst arm tested. See BENCHMARKS.md
    # section 7.3 and the user guide's sensitivity section.
    allow_union_rung: bool = False

    # ------------------------------------------------------------------
    # Feature selection -- Step 3 stability selection
    # ------------------------------------------------------------------
    # Minimum proportion of bootstrap subsamples in which a feature must
    # be selected to survive stability filtering.
    #
    # None (the default) uses 0.6, the permissive end of the 0.6-0.9 range
    # Meinshausen & Buhlmann (2010) give theoretical guarantees over. Below 0.6
    # the bound is no stronger than a single model; at 0.9 the filter demands
    # near-unanimity across 50 subsamples, which measured far stricter than
    # unanimity across 5 or 10 models -- on omics_tiny_n every threshold at or
    # above 0.7 selected nothing at all.
    #
    # This is a floor, not a target: Step 3 maps stability frequency onto the
    # consensus vote scale, so features above this floor still face the full
    # vote requirement and relax with it. Set a float to raise the floor.
    # Reference: doi:10.1111/j.1467-9868.2010.00740.x
    stability_threshold: Optional[float] = None

    # ------------------------------------------------------------------
    # Feature selection -- Step 3 retention target
    # ------------------------------------------------------------------
    # Fraction of Step 3 input features to target for retention via Wrappers.
    # Binary search converges to the min_features_to_select value that
    # produces this many features on average.  0.6 retains 60% of the
    # features entering Step 3, which is the notebook default.
    step3_target_retention: float = 0.6


    # ------------------------------------------------------------------
    # Validation -- bootstrap iterations
    # ------------------------------------------------------------------
    # Number of bootstrap iterations for FeatureSetValidator.
    # Higher values reduce variance in the OOB AUC estimate and widen
    # the confidence interval, at the cost of proportionally more compute.
    n_bootstrap: int = 100

    # ------------------------------------------------------------------
    # Final model calibration
    # ------------------------------------------------------------------
    # Wrap the final model in CalibratedClassifierCV (isotonic regression)
    # to improve probability calibration. Adds one inner CV pass on the
    # training set. Recommended when predicted probabilities (not just
    # rank order) are used downstream.
    calibrate_probabilities: bool = False

    # ------------------------------------------------------------------
    # Nested cross-validation
    # ------------------------------------------------------------------
    # When True, run() finishes with nested cross-validation, in addition to
    # the usual single train/test split. Each of outer_cv_splits stratified
    # folds reruns the whole procedure on its training portion (every enabled
    # step, validation, and the recommendation) and scores every step's panel
    # on the held-out fold. The headline is the recommended panel's held-out
    # AUC; the per-step scores show whether the recommendation chose well.
    # NOTE: adds roughly outer_cv_splits full validated runs to the runtime.
    enable_nested_cv: bool = False
    outer_cv_splits: int = 5

    # ------------------------------------------------------------------
    # Held-out step ranking
    # ------------------------------------------------------------------
    # Which step's panel the recommendation names is decided, by default, by
    # validation on the training split. That validation scores each panel on
    # the same samples its features were selected from, so every panel is
    # flattered and the most aggressive step is flattered most: on a null
    # control the selected panel validated at 0.84 and scored 0.46 held out.
    #
    # With this True, the ranking instead comes from holdout_ranking_splits
    # folds carved from the TRAINING split: each fold reruns the selection
    # steps on its own training portion and scores every step's panel on the
    # rows held out from it, so no panel is scored on the samples that chose
    # it. The pipeline's test split is never touched and stays independent.
    #
    # Off by default because it costs holdout_ranking_splits extra runs of the
    # selection steps. What it ranks is the step, not the exact panel: each
    # fold selects its own features, and the panel delivered is still the one
    # the full training split produced.
    #
    # Off by default on evidence, not caution. Measured on seven datasets it
    # agreed with the training-split ranking on four and changed the choice on
    # three, each time for the worse: 122 features instead of 11 and 77
    # instead of 3 on synthetic sets where the true features are known (their
    # ground-truth F1 fell from 0.99 to 0.18 and from 0.73 to 0.24), and 592
    # noise features on a null control. At n << p, AUC barely separates panel
    # sizes, so the ranking turns on noise and noise favours the larger panel.
    # Its real benefit is an honest quality label on data with no signal, and
    # MAX_PLAUSIBLE_SELECTION_GAIN reaches a warning there for free. Switch
    # this on when predictive AUC is the deliverable rather than a short panel.
    #
    # Five folds, not three. Each fold trains on 80% of the training split
    # rather than 67%, which matters because a step behaves differently on
    # less data: measured on a synthetic rescue case, Step 3 returned 1.3
    # features per fold at three folds against the 3 it delivered from the
    # full split, so the ranking judged a more aggressive step than the one
    # being recommended. Five folds also give a less noisy standard error,
    # and that error sets the band inside which the smaller panel wins.
    enable_holdout_ranking: bool = False
    holdout_ranking_splits: int = 5

    # ------------------------------------------------------------------
    # Post-init validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        # Coerce path types to Path objects for consistent downstream use.
        self.data_path = Path(self.data_path)
        self.output_dir = Path(self.output_dir)

        # --- algorithm ---
        if self.algorithm not in _VALID_ALGORITHMS:
            raise ValueError(
                f"algorithm must be one of {sorted(_VALID_ALGORITHMS)}, "
                f"got '{self.algorithm}'."
            )

        # --- input_format ---
        if self.input_format not in _VALID_INPUT_FORMATS:
            raise ValueError(
                f"input_format must be one of {sorted(_VALID_INPUT_FORMATS)}, "
                f"got '{self.input_format}'."
            )

        # --- output_format ---
        if self.output_format not in _VALID_OUTPUT_FORMATS:
            raise ValueError(
                f"output_format must be one of {sorted(_VALID_OUTPUT_FORMATS)}, "
                f"got '{self.output_format}'."
            )

        # --- plot_format ---
        if self.plot_format not in _VALID_PLOT_FORMATS:
            raise ValueError(
                f"plot_format must be one of {sorted(_VALID_PLOT_FORMATS)}, "
                f"got '{self.plot_format}'."
            )

        # --- excel_output_engine ---
        if self.excel_output_engine not in _VALID_EXCEL_ENGINES:
            raise ValueError(
                f"excel_output_engine must be one of {sorted(_VALID_EXCEL_ENGINES)}, "
                f"got '{self.excel_output_engine}'."
            )

        # --- test_size ---
        if not 0.0 < self.test_size < 1.0:
            raise ValueError(
                f"test_size must be in (0, 1), got {self.test_size}."
            )

        # --- random_seed ---
        # scikit-learn requires random_state in [0, 2**32). Every seed the
        # package derives is an offset from this one, so an out-of-range value
        # here is accepted at construction and then fails deep inside a run
        # that may already have taken hours, with an sklearn
        # InvalidParameterError that does not name this field.
        if not 0 <= self.random_seed < 2 ** 32:
            raise ValueError(
                f"random_seed must be in [0, 2**32), got {self.random_seed}. "
                f"scikit-learn rejects random_state outside that range."
            )

        # --- n_consensus_models ---
        if self.n_consensus_models < 1:
            raise ValueError(
                f"n_consensus_models must be >= 1, got {self.n_consensus_models}."
            )

        # --- min_consensus ---
        if self.min_consensus is not None and not 0.0 < self.min_consensus <= 1.0:
            raise ValueError(
                f"min_consensus must be in (0, 1] or None, got {self.min_consensus}."
            )

        # --- stability_threshold ---
        # Derived as a FLOOR at the permissive end of the Meinshausen &
        # Buhlmann range, not as a formula mapping consensus onto stability.
        #
        # Mapping them directly -- clip(consensus, 0.6, 0.9) -- looks
        # principled but lands badly at unanimity, which clips to 0.9, the
        # strict end. Measured there, Step 3 returned 2 features where 0.6
        # returned 4, and on omics_tiny_n every threshold at or above 0.7
        # selected nothing at all. Unanimity across 50 subsamples is a far
        # stronger demand than unanimity across 5 or 10 models, so the two
        # thresholds do not belong on a shared scale.
        #
        # The linkage that does hold is enforced in Step 3, where stability
        # frequency is mapped onto the vote scale and relaxes in step with the
        # model votes. This field sets the floor below which a feature casts no
        # stability vote at any relaxation level.
        if self.stability_threshold is None:
            self.stability_threshold = 0.6
        elif not 0.0 < self.stability_threshold <= 1.0:
            raise ValueError(
                f"stability_threshold must be in (0, 1], got {self.stability_threshold}."
            )

        # --- allow_union_rung ---
        if not isinstance(self.allow_union_rung, bool):
            raise ValueError(
                f"allow_union_rung must be a bool, got "
                f"{type(self.allow_union_rung).__name__}."
            )

        # --- step2_role ---
        if self.step2_role not in ("terminal", "prefilter"):
            raise ValueError(
                f"step2_role must be 'terminal' or 'prefilter', got "
                f"{self.step2_role!r}."
            )
        if self.step2_role == "prefilter" and not self.enable_step3:
            raise ValueError(
                "step2_role='prefilter' requires enable_step3=True: Step 2 "
                "would hand back a deliberately over-inclusive candidate set "
                "with no later step to prune it."
            )

        # --- step2_stage_colsample ---
        if not 0.0 < self.step2_stage_colsample <= 1.0:
            raise ValueError(
                f"step2_stage_colsample must be in (0, 1], got "
                f"{self.step2_stage_colsample}."
            )

        # --- step2_target_retention ---
        # Upper bound is inclusive. It used to be a fraction of ALL features,
        # where 1.0 meant "keep everything" and was therefore meaningless. It is
        # now a fraction of the REACHABLE pool (the features the stage models
        # actually used), so 1.0 means "keep the whole pool", which is the most
        # permissive Step 2 can be and a setting a caller may legitimately want.
        if not 0.0 < self.step2_target_retention <= 1.0:
            raise ValueError(
                f"step2_target_retention must be in (0, 1], got {self.step2_target_retention}."
            )

        # --- step3_target_retention ---
        if not 0.0 < self.step3_target_retention < 1.0:
            raise ValueError(
                f"step3_target_retention must be in (0, 1), got {self.step3_target_retention}."
            )

        # --- cv_splits ---
        if self.cv_splits is not None and self.cv_splits < 2:
            raise ValueError(
                f"cv_splits must be >= 2 when set explicitly, got {self.cv_splits}."
            )

        # --- min / max cv splits ---
        if self.min_cv_splits < 2:
            raise ValueError(
                f"min_cv_splits must be >= 2, got {self.min_cv_splits}."
            )
        if self.max_cv_splits < self.min_cv_splits:
            raise ValueError(
                f"max_cv_splits ({self.max_cv_splits}) must be >= "
                f"min_cv_splits ({self.min_cv_splits})."
            )

        # --- threshold positivity ---
        if self.max_variance_threshold <= 0.0:
            raise ValueError(
                f"max_variance_threshold must be > 0, got {self.max_variance_threshold}."
            )
        if not 0.0 < self.min_correlation_threshold < 1.0:
            raise ValueError(
                f"min_correlation_threshold must be in (0, 1), "
                f"got {self.min_correlation_threshold}."
            )

        # --- quick_tune_iterations ---
        if self.quick_tune_iterations < 1:
            raise ValueError(
                f"quick_tune_iterations must be >= 1, got {self.quick_tune_iterations}."
            )

        # --- n_jobs ---
        # -1 means "all cores"; 0 is meaningless to every consumer of this value.
        if self.n_jobs == 0 or self.n_jobs < -1:
            raise ValueError(
                f"n_jobs must be -1 (all cores) or a positive integer, "
                f"got {self.n_jobs}."
            )

        # --- min_features_floor ---
        if self.min_features_floor < 1:
            raise ValueError(
                f"min_features_floor must be >= 1, got {self.min_features_floor}."
            )

        # --- n_bootstrap ---
        if self.n_bootstrap < 10:
            raise ValueError(
                f"n_bootstrap must be >= 10, got {self.n_bootstrap}."
            )

        # --- holdout_ranking_splits ---
        if self.holdout_ranking_splits < 2:
            raise ValueError(
                f"holdout_ranking_splits must be >= 2, got "
                f"{self.holdout_ranking_splits}."
            )

        # --- outer_cv_splits ---
        if self.outer_cv_splits < 2:
            raise ValueError(
                f"outer_cv_splits must be >= 2, got {self.outer_cv_splits}."
            )

        # --- schema_version ---
        _known_versions = {"1.0"}
        if self.schema_version not in _known_versions:
            warnings.warn(
                f"SelectOmicsConfig: unknown schema_version '{self.schema_version}'. "
                f"Expected one of {sorted(_known_versions)}. Proceeding with caution.",
                UserWarning,
                stacklevel=3,
            )

    # ------------------------------------------------------------------
    # Preset constructors
    # ------------------------------------------------------------------

    @classmethod
    def quick(
        cls,
        data_path: Union[str, Path],
        target_column: str,
        **kwargs,
    ) -> "SelectOmicsConfig":
        """
        Fast preset: fewer consensus models, shorter tuning, Step 3 off.

        Suitable for rapid iteration and exploratory analysis. Steps 1 and 2
        are enabled; Step 3 (Wrappers) is skipped to reduce
        wall-clock time substantially.

        Parameters
        ----------
        data_path : str or Path
        target_column : str
        **kwargs
            Any SelectOmicsConfig field can be overridden.
        """
        params = {
            'data_path': data_path,
            'target_column': target_column,
            'n_consensus_models': 5,
            'quick_tune_iterations': 10,
            'enable_step3': False,
            'n_bootstrap': 10,
        }
        params.update(kwargs)
        return cls.from_dict(params)

    @classmethod
    def standard(
        cls,
        data_path: Union[str, Path],
        target_column: str,
        **kwargs,
    ) -> "SelectOmicsConfig":
        """
        Balanced preset: all steps enabled with moderate consensus.

        Uses the notebook default consensus model count (n=10) and all
        three selection steps.  Suitable for most production analyses
        where runtime is acceptable.

        Parameters
        ----------
        data_path : str or Path
        target_column : str
        **kwargs
            Any SelectOmicsConfig field can be overridden.
        """
        params = {
            'data_path': data_path,
            'target_column': target_column,
            'n_consensus_models': 10,
            'quick_tune_iterations': 25,
            'enable_step3': True,
            'n_bootstrap': 50,
        }
        params.update(kwargs)
        return cls.from_dict(params)

    @classmethod
    def thorough(
        cls,
        data_path: Union[str, Path],
        target_column: str,
        **kwargs,
    ) -> "SelectOmicsConfig":
        """
        High-stability preset: more consensus models and bootstrap iterations.

        Uses more consensus models (n=20) and tuning iterations for
        maximum stability. Intended for final publication-quality analyses
        where runtime is not a constraint.

        Parameters
        ----------
        data_path : str or Path
        target_column : str
        **kwargs
            Any SelectOmicsConfig field can be overridden.
        """
        params = {
            'data_path': data_path,
            'target_column': target_column,
            'n_consensus_models': 20,
            'quick_tune_iterations': 50,
            'enable_step3': True,
            'n_bootstrap': 100,
        }
        params.update(kwargs)
        return cls.from_dict(params)

    @classmethod
    def omics(
        cls,
        data_path: Union[str, Path],
        target_column: str,
        **kwargs,
    ) -> "SelectOmicsConfig":
        """
        Omics-tuned preset for the small-n / large-p regime (n=50-200, p=500-10000).

        Use this when your data matches the target regime -- a clinical-cohort or
        pilot-study sample count with post-QC genomics / proteomics
        dimensionality.  On large-n or low-dimensional data the conservative
        defaults (``standard`` / ``thorough``) remain more appropriate.

        All three selection steps are enabled.  Any field can be overridden
        via kwargs.

        .. warning::

           **These values are provisional and need re-deriving.**  The
           four-phase search that produced them
           (``benchmarks/SelectOmics_Config_Optimization.ipynb``, winning
           config ``h_071``) tuned ten parameters, and five of them have since
           been shown to have no effect on the result:

           - the four ``*_percentile_range`` bounds, which bounded a search
             over an importance percentile that is inert on tree models
             (98 to 99.8% of tree importances are exactly zero, so
             ``np.percentile`` returned 0.0 across the whole range);
           - ``consensus_threshold``, which set where the graduated relaxation
             STARTED.  Because the relaxation stops at whichever level first
             satisfies the feature floor, starting lower only skipped rungs on
             the way to the same answer.

           The search's own Phase-1 sensitivity had already ranked the four
           percentile bounds last, at Spearman r between -0.038 and 0.045 with
           p from 0.52 to 0.98.  ``consensus_threshold`` measured r = -0.837
           there, but that was under the pre-relaxation code; re-measured after
           the relaxation was introduced it falls to r = -0.013 (p = 0.86), and
           thresholds of 0.4 through 1.0 return the identical feature SET.

           So half the search budget went on parameters with no effect, and the
           dominant parameter -- ``min_features_floor`` -- was fixed at its
           default throughout and never searched at all.  Re-run the search
           with ``benchmarks/config_search.py`` before relying on these values.

        Parameters
        ----------
        data_path : str or Path
        target_column : str
        **kwargs
            Any SelectOmicsConfig field can be overridden.
        """
        params = {
            'data_path': data_path,
            'target_column': target_column,
            # n_consensus_models is deliberately NOT overridden. The old
            # search pinned it at 5, below the dataclass default of 10, which
            # made this preset weaker than the plain defaults on the one
            # parameter the re-run search ranks highest: Spearman r = 0.835,
            # p < 0.0001, monotonic across 3, 5 and 10. A preset aimed at the
            # hardest regime should not be the least stable configuration on
            # offer, so it now inherits 10.
            'quick_tune_iterations': 20,
            'enable_step3': True,
            'n_bootstrap': 100,
            # consensus_threshold 0.442 used to sit here. It is gone rather
            # than translated: it was a starting point for the relaxation, and
            # min_consensus is a floor on it, so there is no value of one that
            # means the same as a value of the other.
            'stability_threshold': 0.259,
            'step3_target_retention': 0.354,
        }
        params.update(kwargs)
        return cls.from_dict(params)

    @classmethod
    def suggest(
        cls,
        data_path: Union[str, Path],
        target_column: str,
        **overrides,
    ) -> 'SelectOmicsConfig':
        """
        Inspect the dataset and recommend a starting configuration.

        Reads the data file (any format ``load_omics_data`` accepts) to
        determine sample count, feature count, missing value prevalence, and
        class balance, then selects appropriate defaults and logs a
        human-readable rationale.

        The returned config is a good starting point; review the printed
        rationale and adjust fields as needed before calling pipeline.run().

        Parameters
        ----------
        data_path : str or Path
            Path to the input data file (any format supported by load_omics_data).
        target_column : str
            Name of the target column.
        **overrides : any SelectOmicsConfig field to override the suggestion.

        Returns
        -------
        SelectOmicsConfig
        """
        import logging
        _suggest_logger = logging.getLogger(__name__)

        data_path = Path(data_path)
        params: dict = {
            'data_path': str(data_path),
            'target_column': target_column,
        }
        notes: list = []

        # --- Load data header ---
        try:
            # Reuse the package loader so suggest() accepts every format the
            # pipeline itself accepts.  Reading by extension here would leave
            # feather, HDF5, and JSON files falling through to read_csv and
            # landing in the "could not inspect data" branch below.
            from SelectOmics.data.loaders import load_omics_data
            df_full = load_omics_data(
                cls(data_path=data_path, target_column=target_column)
            )

            n_samples = len(df_full)
            n_features = df_full.shape[1] - 1  # subtract target column

            if target_column not in df_full.columns:
                raise ValueError(
                    f"target_column '{target_column}' not found in {data_path.name}. "
                    f"Available columns: {df_full.columns.tolist()[:10]}"
                )

            y = df_full[target_column]
            n_classes = y.nunique()
            class_counts = y.value_counts()
            min_class = int(class_counts.min())
            imbalance_ratio = float(class_counts.max() / class_counts.min())

            X = df_full.drop(columns=[target_column])
            missing_frac = float(X.isnull().mean().mean())

        except Exception as exc:
            notes.append(
                f"Could not inspect data ({exc}). "
                "Defaulting to 'standard' preset -- adjust manually."
            )
            params.update(overrides)
            cfg = cls.from_dict(params)
            _suggest_logger.info("SelectOmics config suggestion (data unreadable):")
            for note in notes:
                _suggest_logger.warning("  %s", note)
            return cfg

        notes.append(
            f"Dataset: {n_samples} samples, {n_features} features, "
            f"{n_classes} class(es)"
        )

        # --- Algorithm ---
        params['algorithm'] = 'XGB'
        if missing_frac > 0:
            notes.append(
                f"Missing values detected ({missing_frac:.1%} of feature cells). "
                "Algorithm set to XGB (handles NaN natively)."
            )
        else:
            notes.append("No missing values detected. Algorithm: XGB (default; change to RF/LR/SVM if preferred).")

        # --- n_consensus_models based on sample size ---
        if n_samples < 50:
            params['n_consensus_models'] = 1
            params['enable_step3'] = False
            params['enable_nested_cv'] = False
            params['n_bootstrap'] = 30
            notes.append(
                f"Very small dataset (n={n_samples}). "
                "Using 1 consensus model; Step 3 (Wrappers) disabled; n_bootstrap=30. "
                "Results will have high variance -- interpret with caution."
            )
        elif n_samples < 150:
            params['n_consensus_models'] = 3
            params['n_bootstrap'] = 50
            notes.append(
                f"Small dataset (n={n_samples}). "
                "Using 3 consensus models; n_bootstrap=50."
            )
        elif n_samples < 500:
            params['n_consensus_models'] = 5
            params['n_bootstrap'] = 100
            notes.append(
                f"Moderate dataset (n={n_samples}). "
                "Using 5 consensus models (standard preset)."
            )
        else:
            params['n_consensus_models'] = 10
            params['n_bootstrap'] = 200
            notes.append(
                f"Large dataset (n={n_samples}). "
                "Using 10 consensus models (thorough preset)."
            )

        # --- Feature count ---
        if n_features > 10_000:
            params['quick_tune_iterations'] = 30
            notes.append(
                f"High-dimensional dataset (p={n_features}). "
                "quick_tune_iterations reduced to 30; Step 3 (Wrappers) may be slow -- "
                "consider setting enable_step3=False for a first pass."
            )
        elif n_features > 1_000:
            params['quick_tune_iterations'] = 50
            notes.append(
                f"Moderately high-dimensional dataset (p={n_features}). "
                "quick_tune_iterations=50."
            )
        elif n_features < 30:
            params['enable_step3'] = False
            notes.append(
                f"Low feature count (p={n_features}). "
                "Step 3 (Wrappers) requires >=30 features and will be disabled."
            )
        else:
            params['quick_tune_iterations'] = 50

        # --- Step 1: does this data have redundancy for it to remove? ---
        #
        # Step 1's correlation filter can only drop a feature that has a
        # correlated partner.  Measuring how many do predicts whether Step 1
        # will change the run at all, and lets the trade-off be stated up front
        # rather than discovered afterwards.  Recommendation only: the caller
        # decides, and enable_step1 is never set silently from this.
        try:
            from SelectOmics.utils.diagnostics import assess_feature_redundancy
            _corr_thresh = float(
                overrides.get('min_correlation_threshold', 0.70)
            )
            redundancy = assess_feature_redundancy(
                X, correlation_threshold=_corr_thresh
            )
            notes.append(redundancy['rationale'])

            if not redundancy['step1_will_act']:
                notes.append(
                    "Recommendation: enable_step1=False. It is left ON in this "
                    "suggestion because disabling a step changes what the "
                    "pipeline means; set it yourself if you want the runtime "
                    "back."
                )
            elif redundancy['redundancy_level'] == 'high':
                notes.append(
                    "Step 1 will drive most of the reduction here. On synthetic "
                    "omics data at this redundancy level it raised ground-truth "
                    "recovery by about 0.15-0.27 F1 and lowered cross-seed "
                    "stability by about 0.5-0.8 Kuncheva. If reproducible "
                    "feature lists matter more to you than the shortest list, "
                    "compare a run with enable_step1=False before deciding."
                )
        except Exception as exc:
            notes.append(
                f"Could not assess feature redundancy ({exc}); Step 1 guidance "
                f"unavailable."
            )

        # --- Class balance ---
        if imbalance_ratio > 5:
            notes.append(
                f"Severe class imbalance detected (ratio {imbalance_ratio:.1f}:1). "
                "Consider oversampling (SMOTE) before running the pipeline, "
                "or using algorithm='RF' which supports class_weight='balanced'."
            )
        elif imbalance_ratio > 2:
            notes.append(
                f"Moderate class imbalance (ratio {imbalance_ratio:.1f}:1). "
                "AUC is a good metric here (used by SelectOmics by default)."
            )

        # --- Minimum class size (affects CV splits) ---
        if min_class < 10:
            notes.append(
                f"Smallest class has only {min_class} sample(s). "
                f"CV fold count will be clamped to {min_class}. "
                "Consider whether this class has enough data for reliable classification."
            )

        # --- Multiclass ---
        if n_classes > 2:
            notes.append(
                f"Multiclass problem ({n_classes} classes). "
                "AUC computed as macro-averaged OVR. "
                "Ensure all classes have adequate sample sizes."
            )

        # --- Sample-to-feature ratio ---
        n_per_p = n_samples / max(n_features, 1)
        if n_per_p < 0.1:
            notes.append(
                f"Extremely high-dimensional regime (n/p={n_per_p:.3f}). "
                "Feature selection is critical. All three selection steps "
                "strongly recommended."
            )
        elif n_per_p < 1.0:
            notes.append(
                f"High-dimensional regime (n/p={n_per_p:.2f}). "
                "Feature selection pipeline is well-motivated."
            )

        # Apply any caller overrides last.
        params.update(overrides)

        cfg = cls.from_dict(params)

        # Print the suggestion rationale.
        _suggest_logger.info("=" * 60)
        _suggest_logger.info("SelectOmics Config Suggestion for: %s", data_path.name)
        _suggest_logger.info("=" * 60)
        for note in notes:
            _suggest_logger.info("  %s", note)
        _suggest_logger.info("")
        _suggest_logger.info("Suggested config fields:")
        # Show suggested non-default fields
        suggested_fields = {
            k: v for k, v in cfg.to_dict().items()
            if k in params and k not in ('data_path', 'target_column', 'schema_version')
        }
        for k, v in suggested_fields.items():
            _suggest_logger.info("  %s = %r", k, v)
        _suggest_logger.info("")
        _suggest_logger.info(
            "Review these settings and adjust before running. "
            "Save with: config.to_yaml('my_config.yaml')"
        )
        _suggest_logger.info("=" * 60)

        return cfg

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a plain dictionary representation of the configuration.

        Path objects are converted to strings so the result is
        JSON-serialisable without a custom encoder.
        """
        d = asdict(self)
        # asdict recurses into dataclasses and named-tuples; Path objects
        # are not handled automatically and must be converted explicitly.
        d['data_path'] = str(d['data_path'])
        d['output_dir'] = str(d['output_dir'])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SelectOmicsConfig":
        """Construct a SelectOmicsConfig from a plain dictionary.

        Unknown keys are silently ignored so that serialised configs from
        older pipeline versions remain loadable.  Schema migration is
        applied before construction when schema_version is present.
        """
        d = dict(d)  # Shallow copy; do not mutate the caller's dict.

        # ------------------------------------------------------------------
        # Schema migration
        # ------------------------------------------------------------------
        schema_version = d.get('schema_version', '1.0')
        # No migrations needed for version 1.0 -> 1.0 (current).
        # Future versions add migration blocks here:
        #   if schema_version == '0.9':
        #       d['new_field'] = d.pop('old_field', default)
        #       d['schema_version'] = '1.0'
        _ = schema_version  # suppress "unused variable" lint warning

        # Collect only the field names recognised by this dataclass version.
        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(d.keys()) - known
        if unknown:
            warnings.warn(
                f"SelectOmicsConfig: unrecognised config key(s) {sorted(unknown)} will be "
                f"ignored. Check for typos -- misspelled keys silently revert to defaults.",
                UserWarning,
                stacklevel=2,
            )
        filtered = {k: v for k, v in d.items() if k in known}

        return cls(**filtered)

    @staticmethod
    def _anchor_paths(d: dict, config_path: Union[str, Path]) -> dict:
        """
        Resolve relative ``data_path`` / ``output_dir`` against the config file.

        A config that stores a relative path is portable, but only if loading it
        anchors that path somewhere predictable.  Anchoring on the process CWD
        means the same file works or fails depending on which directory the
        caller happened to be in; anchoring on the config file itself means a
        config and its data can be moved together and keep working.

        Absolute paths are left alone, so configs written by earlier versions
        load exactly as before.

        Parameters
        ----------
        d : dict
            Raw mapping parsed from the config file.  Not mutated.
        config_path : str or Path
            Path of the file ``d`` was read from.

        Returns
        -------
        dict
            Copy of ``d`` with relative path fields made absolute.
        """
        d = dict(d)
        base = Path(config_path).resolve().parent
        for field_name in ('data_path', 'output_dir'):
            value = d.get(field_name)
            if value is None:
                continue
            candidate = Path(value)
            if not candidate.is_absolute():
                d[field_name] = str((base / candidate).resolve())
        return d

    def to_json(self, path: Union[str, Path]) -> None:
        """Serialise the configuration to a JSON file.

        Parameters
        ----------
        path : str or Path
            Destination file path.  The file is created or overwritten.
        """
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(self.to_dict(), fh, indent=4)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "SelectOmicsConfig":
        """Load a SelectOmicsConfig from a JSON file written by to_json.

        A relative ``data_path`` or ``output_dir`` is resolved against the
        directory holding the config file, not the current working directory,
        so a config and its data can be moved together and keep working.
        Absolute paths are used as-is.

        Parameters
        ----------
        path : str or Path
            Path to the JSON file.

        Returns
        -------
        SelectOmicsConfig
            Reconstructed configuration instance.
        """
        with open(path, 'r', encoding='utf-8') as fh:
            d = json.load(fh)
        return cls.from_dict(cls._anchor_paths(d, path))

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Serialise the configuration to a YAML file.

        Requires PyYAML (``pip install pyyaml``).  Tuple fields are
        converted to lists for YAML compatibility; from_yaml restores them.

        Parameters
        ----------
        path : str or Path
            Destination file path.  The file is created or overwritten.

        Raises
        ------
        ImportError
            If PyYAML is not installed.
        """
        try:
            import yaml  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "to_yaml requires PyYAML: pip install pyyaml"
            ) from exc

        with open(path, 'w', encoding='utf-8') as fh:
            yaml.dump(self.to_dict(), fh, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "SelectOmicsConfig":
        """Load a SelectOmicsConfig from a YAML file written by to_yaml.

        Requires PyYAML (``pip install pyyaml``).  Tuple fields are
        restored from lists automatically via from_dict.

        A relative ``data_path`` or ``output_dir`` is resolved against the
        directory holding the config file, as in from_json.

        Parameters
        ----------
        path : str or Path
            Path to the YAML file.

        Returns
        -------
        SelectOmicsConfig

        Raises
        ------
        ImportError
            If PyYAML is not installed.
        """
        try:
            import yaml  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "from_yaml requires PyYAML: pip install pyyaml"
            ) from exc

        with open(path, 'r', encoding='utf-8') as fh:
            d = yaml.safe_load(fh)
        return cls.from_dict(cls._anchor_paths(d, path))

    @classmethod
    def template_yaml(cls, path: Union[str, Path, None] = None) -> str:
        """
        Generate a fully-commented YAML configuration template.

        Writes every available configuration field with its default value,
        valid range, and a human-readable description.  Use this as a
        starting point for manual configuration:

            SelectOmicsConfig.template_yaml('my_config.yaml')
            # Edit my_config.yaml, then:
            config = SelectOmicsConfig.from_yaml('my_config.yaml')

        Parameters
        ----------
        path : str or Path or None
            File path to write the template to.  If None, returns the
            YAML string without writing to disk.

        Returns
        -------
        str
            The YAML template string.
        """
        template = """\
# SelectOmics Configuration Template
# Generated by SelectOmicsConfig.template_yaml()
# Schema version: 1.0
#
# Instructions:
#   1. Fill in data_path and target_column (required)
#   2. Choose a preset level: quick / standard / thorough
#      OR manually adjust individual fields below
#   3. Load with: SelectOmicsConfig.from_yaml('this_file.yaml')
#
# TIP: Run SelectOmicsConfig.suggest('data.csv', 'Target') first to get
#      dataset-specific recommendations.

schema_version: "1.0"

# ===========================================================================
# REQUIRED -- must be set for every run
# ===========================================================================

# Path to the input data file.
# Supported formats: csv, tsv, excel (.xlsx/.xls), parquet, feather, hdf5, json
data_path: "path/to/your/data.csv"

# Name of the column in the file that contains class labels.
target_column: "Class"

# ===========================================================================
# ALGORITHM
# ===========================================================================

# Primary classification algorithm used throughout the pipeline.
# Options: XGB (XGBoost), RF (Random Forest), LR (Logistic Regression), SVM
# XGB is automatically selected when the dataset contains missing values.
# Recommendation: XGB for tabular omics; RF as a robust alternative.
algorithm: "XGB"

# Number of bootstrap-seeded consensus models per selection step.
# Higher counts improve stability but increase runtime proportionally.
# quick=5, standard=10, thorough=20
n_consensus_models: 10

# Threads each estimator may use. -1 uses every core; 1 forces serial.
# Does NOT change results: the pipeline pins settings that are thread-invariant,
# so a run at n_jobs=-1 selects exactly the features a run at n_jobs=1 does.
# Falls back to serial automatically when only one core is available, or when
# the matrix is too small for threading to pay for itself.
# Set to 1 when launching several pipelines yourself, so they do not compete.
n_jobs: -1

# Attempt GPU acceleration for XGBoost (requires CUDA-capable GPU and xgboost>=2.0).
# Falls back to CPU silently when hardware/driver is unavailable.
# NOTE: unlike n_jobs, the GPU DOES shift results slightly (it uses a different
# split-finding implementation). Leave false for runs you intend to compare
# against CPU-generated numbers.
use_gpu: true

# ===========================================================================
# REPRODUCIBILITY
# ===========================================================================

# Master random seed for all stochastic operations.
# Change this to verify result stability across seeds.
random_seed: 42

# ===========================================================================
# DATA SPLIT
# ===========================================================================

# Fraction of samples reserved for the held-out test set.
# Recommended range: 0.15-0.30. Smaller test sets for very small n.
test_size: 0.25

# ===========================================================================
# CROSS-VALIDATION
# ===========================================================================

# Number of stratified CV folds. Leave as null for automatic selection:
#   cv_splits = min(max_cv_splits, max(min_cv_splits, min_class_count))
# Set explicitly to override automatic selection (must be >= 2).
cv_splits: null

min_cv_splits: 3    # Minimum CV folds (must be >= 2)
max_cv_splits: 20   # Maximum CV folds

# ===========================================================================
# HYPERPARAMETER TUNING
# ===========================================================================

# RandomizedSearchCV iteration budget for the initial quick-tune pass.
# Higher values find better hyperparameters but take longer.
# quick=10, standard=25, thorough=50
quick_tune_iterations: 50

# ===========================================================================
# PIPELINE STEPS
# ===========================================================================
# Each step can be independently enabled or disabled.
# Disabling a step passes data through unchanged.

enable_step1: true   # Data cleaning: removes constant, low-variance, correlated features
enable_step2: true   # Regularization: parallel L1/L2 consensus filtering
enable_step3: true   # Wrappers: RFECV + stability selection (requires >=30 features)

# Judge redundancy on the pooled WITHIN-class correlation rather than total
# correlation. Two features correlated only because both track the label are
# then not treated as duplicates of each other, and the survivor of a genuinely
# redundant pair is chosen by association with the outcome rather than by
# variance. Set false for the class-unaware filter used up to 0.7.0.
class_aware_correlation: true

# ===========================================================================
# MINIMUM AGREEMENT
# ===========================================================================

# Lowest agreement level you will accept in the delivered panel.
#
# Relaxation ALWAYS starts at unanimity and steps down one vote at a time
# until min_features_floor is met. This is the level below which it will not
# step, even if that leaves a smaller panel than you asked for.
#
# It is a FLOOR, not a target: you specify the panel size you need and the
# pipeline reports the agreement it achieved. Leave blank to remove the brake
# entirely; 1.0 means unanimity or nothing.
min_consensus: 0.4

# ===========================================================================
# STABILITY SELECTION (Step 3)
# ===========================================================================

# Minimum fraction of bootstrap subsamples where a feature must be selected
# to survive stability selection. Range: (0, 1].
# Meinshausen & Buhlmann (2010) recommend 0.6-0.9 for theoretical guarantees.
# Values below 0.5 provide no stronger guarantee than a single model.
stability_threshold: 0.6

# ===========================================================================
# RETENTION TARGETS
# ===========================================================================
# How much of each step's input it aims to keep. These are the knobs for
# "too aggressive" or "too conservative", together with min_features_floor.
# A step cannot retain more than its models found usable, so the target is
# an upper bound rather than a promise; each step logs when it saturates.

# Fraction of Step 2 input features each regularization stage TARGETS.
# The L1 and L2 stages are intersected, so Step 2 as a whole returns fewer.
step2_target_retention: 1.0

# Column-sampling rate for Step 2's XGB stage models. Sets the CEILING on
# what Step 2 can return: a tree only assigns importance to features it
# splits on, and Step 2 never selects beyond that set. Raise toward 1.0 to
# widen the pool; this is the only knob that makes Step 2 less aggressive.
step2_stage_colsample: 0.3

# What job Step 2 is doing: 'terminal' (it delivers the final panel) or
# 'prefilter' (it delivers candidates for Step 3 to prune, overshooting the
# Step 3 gate so that step actually engages). 'prefilter' requires
# enable_step3: true.
step2_role: terminal

# Add a union rung below the intersection ladder when the intersection cannot
# reach min_features_floor. Raises the ceiling on how many features a step can
# return. Off by default.
allow_union_rung: false

# Fraction of Step 3 input features to TARGET for retention (via Wrappers binary search).
# 0.6 = keep 60% of features entering Step 3.
step3_target_retention: 0.6


# ===========================================================================
# STEP 1 THRESHOLDS
# ===========================================================================

# Upper bound for the variance threshold grid search.
# Features with variance <= threshold are dropped. Grid: [0, max_variance_threshold].
max_variance_threshold: 0.30

# Upper bound for the correlation threshold grid search (semantically a MAX threshold).
# Correlated feature pairs with correlation >= threshold: lower-variance one dropped.
min_correlation_threshold: 0.70

# ===========================================================================
# EVALUATION
# ===========================================================================

enable_step_evaluations: true         # Per-step CV evaluation and comparison
enable_final_test_evaluation: true     # Held-out test set metrics

# Number of bootstrap iterations for FeatureSetValidator.
# Higher reduces variance of OOB AUC estimate. quick=10, standard=50, thorough=100.
n_bootstrap: 100

# Wrap the final model in CalibratedClassifierCV (isotonic regression).
# Improves probability calibration. Adds one inner CV pass.
# Recommended for thorough/publication analyses.
calibrate_probabilities: false

# ===========================================================================
# NESTED CROSS-VALIDATION
# ===========================================================================

# Use nested CV for an unbiased generalization estimate of the whole procedure.
# Each outer fold reruns selection, validation and the recommendation on its
# training portion and scores every step's panel on the held-out fold, so it
# also shows whether the recommended step was the right choice.
# Adds roughly outer_cv_splits full runs. Recommended for publication.
enable_nested_cv: false

# Number of outer CV folds (must be >= 2).
outer_cv_splits: 5

# Rank the steps on folds held out from the training split, rather than on
# validation that shares its samples with selection. Costs one extra run of the
# selection steps per fold. The test split is never touched either way.
enable_holdout_ranking: false

# Number of ranking folds taken from the training split (must be >= 2).
holdout_ranking_splits: 5

# ===========================================================================
# OUTPUT
# ===========================================================================

output_dir: "selectomics_results"   # Directory for all output files
output_format: "csv"                # csv | tsv | excel | parquet | feather | json
plot_format: "png"                  # png | svg | pdf | tiff | jpeg
save_intermediate_results: true     # Save per-step feature matrices and summaries
create_visualizations: true         # Generate and save all diagnostic plots
verbose: false                      # Set true to see DEBUG-level log detail

# ===========================================================================
# INPUT FORMAT (usually auto-detected from file extension)
# ===========================================================================

input_format: "auto"        # auto | csv | tsv | excel | parquet | hdf5 | feather | json
hdf5_key: "data"            # HDF5 dataset key (ignored for other formats)
excel_sheet_name: 0         # Excel sheet index or name (0 = first sheet)
excel_output_engine: "openpyxl"  # openpyxl | xlsxwriter
"""
        if path is not None:
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(template)
        return template

    # ------------------------------------------------------------------
    # Human-readable representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a compact, eval-able representation of the configuration.

        Only fields that differ from their defaults are shown, keeping the
        repr concise for interactive use.  The full dict is always available
        via ``config.to_dict()``.
        """
        diffs = self.diff_from_defaults()
        if diffs:
            pairs = ", ".join(f"{k}={v!r}" for k, v in diffs.items())
            return (
                f"SelectOmicsConfig(data_path={self.data_path!r}, "
                f"target_column={self.target_column!r}, {pairs})"
            )
        return (
            f"SelectOmicsConfig(data_path={self.data_path!r}, "
            f"target_column={self.target_column!r})"
        )

    def diff_from_defaults(self) -> dict:
        """Return a dict of fields whose values differ from the class defaults.

        Compares against a reference instance constructed with only
        ``data_path`` and ``target_column`` so that ``__post_init__``
        coercions (e.g. Path conversion) are applied to both sides before
        comparison.  ``data_path``, ``target_column``, and ``schema_version``
        are always excluded from the result.

        Returns
        -------
        dict
            Keys are field names; values are the current (non-default) values.

        Examples
        --------
        >>> cfg = SelectOmicsConfig('data.csv', 'Label', algorithm='RF')
        >>> cfg.diff_from_defaults()
        {'algorithm': 'RF'}
        """
        # Build a reference config with only the required fields.
        # This ensures __post_init__ coercions apply symmetrically.
        _reference = type(self)(
            data_path=self.data_path,
            target_column=self.target_column,
        )
        _exclude = {'data_path', 'target_column', 'schema_version'}
        diffs = {}
        for f in self.__dataclass_fields__.values():
            if f.name in _exclude:
                continue
            current = getattr(self, f.name)
            default = getattr(_reference, f.name)
            if current != default:
                diffs[f.name] = current
        return diffs
