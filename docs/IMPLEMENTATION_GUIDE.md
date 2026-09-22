# SelectOmics Implementation Guide

This guide documents the internal engineering of the SelectOmics pipeline for developers and advanced users. It is intended as a reference for the mechanisms that make the pipeline reproducible, leak-free, memory-bounded, and robust to failure, rather than as a user tutorial. Every claim here is sourced from the current package code. File paths are given relative to the package root (`SelectOmics/`).

The pipeline reduces a high-dimensional omics feature matrix through five stages held on a single stateful `SelectOmicsPipeline` object:

- Step 0: reference model evaluation (baseline cross-validation AUC).
- Step 1: data-based cleaning (constant, low-variance, correlated feature removal).
- Step 2: model-based regularization (parallel L1 and L2 consensus).
- Step 3: wrappers (RFECV with stability selection).

A single stratified train/test split is made up front; the test set is untouched until the optional final evaluation. All inter-step state lives on the instance, so no module-level mutable state exists and multiple pipelines are fully isolated.

---

## Table of Contents

1. [Reproducibility](#1-reproducibility)
2. [Data Leakage Prevention](#2-data-leakage-prevention)
3. [Memory Management](#3-memory-management)
4. [Parallel Processing](#4-parallel-processing)
5. [GPU Support](#5-gpu-support)
6. [Algorithm Fallbacks](#6-algorithm-fallbacks)
7. [Checkpoint / Resume System](#7-checkpoint--resume-system)
8. [Config Validation](#8-config-validation)
9. [Pipeline State Machine](#9-pipeline-state-machine)
10. [Error Handling and Logging](#10-error-handling-and-logging)

---

## 1. Reproducibility

### 1.1 The `random_seed` parameter

`config.random_seed` (default `42`) is the single master seed for all stochastic operations. It is stored once on `SelectOmicsConfig` and passed explicitly to every function that needs it. No module-level random state is ever modified: the pipeline never seeds `numpy.random` or Python's `random` global state, so importing or running SelectOmics has no side effect on the surrounding process.

### 1.2 Train/test split and CV seeding

`create_train_test_split` (`data/preprocessing.py`) calls `sklearn.model_selection.train_test_split` with `random_state=config.random_seed` and stratification on the encoded labels. The shared `StratifiedKFold` object is constructed with the same seed:

```python
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.random_seed)
```

The fold count `n_splits` is deterministic. If `config.cv_splits` is set it is clamped to `[min_cv_splits, max_cv_splits]`; otherwise it is derived from the minority class count of the training set as `min(max_cv_splits, max(min_cv_splits, min_class_count_train))`. This one `cv` object is threaded through every step (Step 0 reference, all step evaluations), so fold assignments are identical everywhere.

### 1.3 Hyperparameter tuning seeding

`quick_tune_algorithm` (`models/base.py`) builds a `RandomizedSearchCV` with `random_state=config.random_seed`, `n_iter=config.quick_tune_iterations`, and its own tuning `StratifiedKFold` (`n_splits = min(5, max(2, min_class))`, `random_state=config.random_seed`). `n_jobs=1` is enforced throughout so results do not depend on thread scheduling. The scorer is `'roc_auc'` for binary problems and `'roc_auc_ovr'` for multiclass. Only the algorithm named in `config.algorithm` is tuned; the other three receive fixed-default pipelines from `_default_lr / _default_xgb / _default_rf / _default_svm`.

### 1.4 Consensus and per-stage seed offset strategy

Each stage seeds its models with `base_seed + i * STEP` offsets chosen so that no two models across the pipeline collide for typical `n_models`. The schemes are enforced directly inside each step module (the constants named below are module-level):

| Step | Context | Seed formula |
|------|---------|--------------|
| Step 1 | Consensus eval models | `base_seed + i * 100` (`_SEED_STEP`) |
| Step 2 | L1 voting-pass models | `base_seed + i * 200` (`_L1_SEED_STEP`) |
| Step 2 | L2 voting-pass models | `base_seed + i * 200 + 100` (`+ _L2_SEED_OFFSET`) |
| Step 2 | Consensus eval models | `base_seed + i * 100` (`_EVAL_SEED_STEP`) |
| Step 2 | Grid/binary-search stage probes | `base_seed` (fixed) |
| Step 3 | RFECV models (probe and voting) | `base_seed + i * 400` (`_RFECV_SEED_STEP`) |
| Step 3 | Stability model seed | `base_seed + (i % n_models) * 400 + 500` (`+ _STABILITY_SEED_OFFSET`) |
| Step 3 | Stability subsample split | `base_seed + i * 500` (`_SUBSAMPLE_SEED_STEP`) |
| Step 3 | Consensus eval models | `base_seed + i * 100` (`_EVAL_SEED_STEP`) |

The general-purpose helper `build_consensus_pipelines` in `models/base.py` documents a `base_seed + i * 100` spacing for its own API, but each step builds its consensus/stage models directly with the offsets above. In all cases the numpy/random/Python global state is left untouched.

### 1.5 Bootstrap sampling seeding in `cv_evaluate_model`

`cv_evaluate_model` (`models/evaluation.py`) seeds each fold's bootstrap resample uniquely per `(model, fold)` pair:

```python
boot_seed = model_seed + fold_idx * 1000
boot_idx = resample(..., replace=True, stratify=y_fold_train, random_state=boot_seed)
```

`model_seed` is read from the pipeline's classifier `random_state` (falling back to `0` if unavailable). This guarantees that model 1 and model 2 see different bootstrap samples within the same fold, while the samples are identical across independent runs. Bootstrap is enabled for all consensus step evaluations and disabled for the Step 0 reference (`use_bootstrap=False`), so the reference reflects the unmodified training distribution.

### 1.6 RFECV internal CV seeding

Each RFECV instance in Step 3 constructs its own internal `StratifiedKFold` with `random_state=seed` where `seed = base_seed + i * 400`:

```python
rfecv_cv = StratifiedKFold(n_splits=internal_cv_splits, shuffle=True, random_state=seed)
rfecv = RFECV(estimator=estimator, step=1, cv=rfecv_cv, scoring="roc_auc_ovr", n_jobs=1, ...)
```

The internal CV uses `min(3, cv.n_splits)` folds for speed, deliberately not the outer pipeline CV (which is reserved for step evaluations).

### 1.7 Binary search determinism

`_binary_search_percentile` (`utils/helpers.py`) always evaluates the two anchor points `lo` and `hi` first, then bisects, tracking the closest result seen. The evaluation order is fixed (`lo`, `hi`, then successive midpoints), with `tolerance=0.01` and `max_iter=10`, so total `evaluate_fn` calls are at most 12. Because each `evaluate_fn` is itself deterministic (fixed seeds), the search result is deterministic for a given `(lo, hi, target)`.

### 1.8 Reference-model cache key

The reference cache in `DataStateTracker` (`utils/state.py`) is keyed on a SHA-256 digest, not Python's process-randomised `hash()`:

```python
columns_str = '|'.join(sorted(X.columns.tolist()))
raw = f"{algorithm}::{columns_str}".encode('utf-8')
return hashlib.sha256(raw).hexdigest()[:16]
```

The key is the first 16 hex characters of the SHA-256 of the pipe-joined sorted feature names concatenated with the algorithm string, making it deterministic and stable across processes.

### 1.9 Summary guarantee

Given the same `random_seed`, data file, and package versions, every feature set selected, every AUC value reported, and every saved artefact is identical across independent `pipeline.run()` invocations.

---

## 2. Data Leakage Prevention

### 2.1 Train/test split timing

The split is the first operation after loading. In `load_data` (`pipeline.py`):

1. `load_omics_data` reads the file.
2. `prepare_data` extracts the numeric feature matrix and target vector.
3. `handle_nans` inspects NaNs and may switch the algorithm (Section 6).
4. `create_train_test_split` immediately produces `X_train`, `X_test`, `y_train`, `y_test`, the label encoder, and the shared `cv`.
5. `quick_tune_all` tunes on the training set only.

The test set is held in `self._X_test` / `self._y_test` and passed to no selection step. Only `_run_final_test_evaluation` reads it.

### 2.2 Scaling fit/transform discipline

There is no standalone scaling or imputation stage that could leak statistics. `MinMaxScaler` is the first step (`('scaler', MinMaxScaler())`) of every sklearn `Pipeline` in the package: quick-tune pipelines, all consensus factories, stage models, RFECV estimators, stability models, and RFECV estimators. Because sklearn calls `scaler.fit_transform` on the training fold and `scaler.transform` on the held-out fold or test set, the scaler is always fitted on training data only.

### 2.3 Step 1 statistics computed on train only

Step 1 (`selection/step1_cleaning.py`) computes all filtering statistics on `X_train` after constant removal:

- `feature_variances = X_train.var(axis=0)` (also the correlation tie-breaker).
- `corr_matrix = X_after_constant.corr().abs()`, computed once.
- `VarianceThreshold.fit(X)` in `_apply_variance_filter`, on train only.

The best-threshold pair is found by an 11x11 grid maximising the Jaccard overlap of the two filters' drop sets. The resulting boolean mask is applied to both matrices by column subsetting (`X_after_constant.loc[:, final_kept]` and the matching test slice); the test set never influences threshold selection.

### 2.4 Step 2 importance computed on train only

Step 2 (`selection/step2_regularization.py`) fits every stage model (`_estimate_drop_mask`, `_run_voting_pass`) on `X_train` / `y_train`. Both binary searches and both voting passes operate entirely within training. `X_test_reg` is produced only by column subsetting at the end.

### 2.5 Step 3 RFECV and stability on train only

Step 3 (`selection/step3_wrapper.py`) fits all RFECV instances on `X_train`; RFECV internal CV splits `X_train` only. Stability selection draws `StratifiedShuffleSplit` subsamples from `X_train` only. `X_test_rfecv` is produced by column subsetting.

### 2.6 Step evaluations use CV within train only

Every per-step evaluation (`_evaluate_step1` through `_evaluate_step3`) calls `evaluate_models_collection` or `cv_evaluate_model` on the reduced training matrix with the shared `cv`. No step-level CV protocol sees the held-out test set.

### 2.7 Feature-set validation uses train only

`validate_features` (`pipeline.py`) passes only `state.X_train` for each recorded step to `FeatureSetValidator.compare_feature_sets`. All three validation protocols in `evaluation/validation.py` (stratified CV, LOO / stratified shuffle-split, bootstrap OOB) operate on the training split. The docstring states this explicitly: validation uses only the training split so the held-out test set stays unseen by all cross-validation protocols.

### 2.8 Final test evaluation

`_run_final_test_evaluation` (`pipeline.py`) scores the recommended panel, or the last completed step's when there is no recommendation. `_fit_final_model` clones the tuned pipeline for the active algorithm, refits it on that panel, and optionally wraps it in an isotonic `CalibratedClassifierCV` (Section 9.4); the model then predicts on the matching test columns. The recommendation is made from the training split alone, so the test set is still spent exactly once. The training AUC the generalisation gap is measured from comes from `_panel_cv_result` for the same panel. The model never saw test labels during any selection step or validation protocol.

`run_nested_cv` follows the same discipline per outer fold. Each fold builds an inner `SelectOmicsPipeline` through `_adopt_split`, handing it the outer training portion and the outer test fold's features but not its labels, then runs `_run_selection_steps` (the same sequence `run()` uses), `validate_features`, and the recommendation. Every step's panel is refitted with `_fit_final_model` and scored on the outer fold.

---

## 3. Memory Management

### 3.1 Progressive feature narrowing

Each step receives only the matrix produced by the previous step. `DataStateTracker.get_latest()` returns the most recent `PipelineState`, whose `X_train` already has the reduced column set. The full original `self._X_train` is retained only for provenance reporting (`get_feature_provenance`), not fed to steps.

### 3.2 Raw DataFrame disposal after loading

Once the split and quick-tune complete in `load_data`, the raw loaded DataFrame is freed:

```python
del self._df
self._df = None
gc.collect()
```

This releases the largest single object in the load phase before any selection work begins.

### 3.3 `gc.collect()` between steps

Each of `run_step1_cleaning`, `run_step2_regularization`, and `run_step3_wrapper` ends with `gc.collect()`. `load_data` and `validate_features` also collect at the end.

### 3.4 Intermediate matrices written to disk

When `save_intermediate_results=True`, each step writes its reduced feature matrix and diagnostics to `output_dir` in `config.output_format` (for example `step1_X_clean`, `step2_X_reg_clean`, `step3_X_wrappers_final`) rather than accumulating every intermediate in memory.

### 3.5 Consensus models fitted sequentially

Consensus and stage models are constructed, fitted, evaluated, and discarded one at a time. `evaluate_models_collection` iterates a `pipelines` dict and calls `cv_evaluate_model`, which `clone`s and fits inside the fold loop. At no point do all `n_consensus_models` fitted objects coexist. Runtime scales roughly linearly with `n_consensus_models`; peak memory is bounded by one model plus the current matrix.

### 3.6 Checkpoint size

`_save_checkpoint` pickles the full-feature `self._X_train` and `self._X_test` (Section 7), so checkpoint file size is proportional to the original matrices, not the step-reduced ones. Large datasets produce large checkpoints.

---

## 4. Parallel Processing

### 4.1 Single-threaded by design

The pipeline is single-threaded at the Python level. `n_jobs=1` is enforced across sklearn estimators, cross-validators, and search objects. This is intentional: it removes non-determinism from OS thread scheduling and avoids the memory-duplication failures common when sklearn forks worker processes on Windows.

Concrete enforcement points:

- `RandomizedSearchCV(n_jobs=1)` in `quick_tune_algorithm` (`models/base.py`).
- `LogisticRegression(n_jobs=1)` and `RandomForestClassifier(n_jobs=1)` in every default, consensus, and stage builder.
- `RFECV(n_jobs=1)` in `_run_rfecv_stage` (`selection/step3_wrapper.py`).
- `cross_val_score(n_jobs=1)` in `FeatureSetValidator.stratified_cv_validation` (`evaluation/validation.py`).

### 4.2 XGBoost internal threading

XGBoost is the one exception, and only on GPU. Its `n_jobs` is set as:

```python
n_jobs=1 if device == 'cpu' else -1
```

On CPU (`device='cpu'`) it stays single-threaded for determinism. On GPU (`device='cuda'`) it receives `n_jobs=-1` so the CUDA execution can use available resources. This pattern appears in `_default_xgb`, `quick_tune_algorithm`, `create_xgb_pipeline_consensus`, Step 2's `_build_stage_model`, and Step 3's `_build_rfecv_estimator`. XGBoost's internal parallelism is separate from Python's threading and does not affect reproducibility provided the same GPU and driver are used across runs.

### 4.3 RFECV `n_jobs`

`RFECV` is hardcoded to `n_jobs=1` in `_run_rfecv_stage`; there is no config key to change it. Enabling parallel RFECV would require verifying fork-safety of the estimator on all target platforms.

### 4.4 Consensus fitting is sequential

All `n_consensus_models` models are fitted in a plain Python `for` loop with no cross-model parallelism. Because runtime scales with `n_consensus_models`, presets trade stability for time: `quick` uses 5 models, `standard` 10, `thorough` 20.

---

## 5. GPU Support

### 5.1 Detection and activation

GPU support applies to XGBoost only. `use_gpu` (default `True`) gates detection. In `load_data`:

```python
_xgb_device = 'cpu'
if self.config.use_gpu:
    _xgb_device = 'cuda' if detect_gpu() else 'cpu'
```

`detect_gpu` (`utils/helpers.py`) probes by fitting a trivial `XGBClassifier(n_estimators=1, device='cuda')` on a 2x2 array. Any exception (no CUDA, driver mismatch, `xgboost < 2.0`) is caught and it returns `False`, so the pipeline falls back to CPU transparently.

### 5.2 Device propagation

The resolved device string is passed as `device=_xgb_device` into `quick_tune_all`, which threads it into the XGB default builder and `quick_tune_algorithm`. Downstream steps do not re-detect; they read the device back from the tuned pipeline:

```python
_device = tuned_pipelines['XGB'].named_steps['clf'].get_params().get('device', 'cpu')
```

This inheritance pattern appears in `run_step2_regularization` and `run_step3_wrapper`, each guarded by a try/except that defaults to `'cpu'`. `run_nested_cv` calls `detect_gpu` once and reuses the result across outer folds.

### 5.3 Enabling GPU

Leave `use_gpu=True` (the default). With a CUDA-capable GPU and `xgboost >= 2.0`, CUDA is used automatically with no further configuration. An incompatible driver is handled by `detect_gpu` returning `False`, falling back to CPU.

### 5.4 Non-XGB algorithms

LR, RF, and SVM have no GPU path. When the active algorithm is not XGB, no device parameter is passed to their constructors and the device-inheritance blocks are skipped (they are guarded by `algorithm == 'XGB'`).

---

## 6. Algorithm Fallbacks

### 6.1 NaN detection and algorithm switch

`handle_nans` (`data/preprocessing.py`) counts NaNs once (`X.isna().sum().sum()`). If zero, it returns immediately. If NaNs are present it always logs a warning (regardless of `verbose`). When the configured algorithm is `'LR'` or `'SVM'` (which cannot handle NaN natively) it switches to `'XGB'`:

```python
if algorithm in ('LR', 'SVM'):
    logger.warning("Algorithm '%s' cannot handle NaN values. Automatically switching to XGB.", algorithm)
    algorithm = 'XGB'
```

`load_data` propagates the switch to `self.config.algorithm`, so tuning, Steps 0-4, validation, and final evaluation all use the updated algorithm. For `'RF'` and `'XGB'`, NaNs are preserved (never imputed) so tree splits can exploit the missingness pattern; an informational message is logged for RF.

### 6.2 SVM surrogate for importance

The RBF `SVC` has no `coef_`, so Steps 2 and 3 use a `LinearSVC` surrogate for importance and recursive elimination: Step 2 `_build_stage_model` (L1 penalty low-C vs L2 penalty), Step 3 `_build_rfecv_estimator` (L1 `LinearSVC`, `dual='auto'`). The RBF `SVC` is still used for all AUC scoring, including every Step 0 through Step 3 evaluation. Importance is extracted as the max absolute coefficient across classes.

### 6.3 Step 3 RFECV failure fallback

When an RFECV fit fails (probe or voting pass), `warnings["rfecv_failed"]` is incremented and all features are treated as selected for that model (`rfecv_vote_counts += np.ones(...)`, importance `np.full(n_features, 1.0/n_features)`), a conservative choice preventing a single failure from unfairly excluding features.

### 6.4 Step 3 stability fallbacks

Subsamples missing a class increment `stability_skipped_class` and are skipped; per-subsample fit failures increment `stability_failed`. If all `STABILITY_N_SUBSAMPLES` (50) subsamples fail, `stability_all_failed` is set and `stability_selected` defaults to all-True, so the consensus intersection degenerates to the RFECV result. All counts are logged once at step end via `_print_warnings`.

### 6.5 Skip thresholds

Step 3 is skipped (pass-through, `skipped=True`) when `X_train.shape[1] < MIN_FEATURES_FOR_RFECV` (30). The threshold stands in for a quantity that is not observable at runtime: how much noise the incoming panel still contains. Step 2's output is nearly pure signal, so running elimination on it cuts true features rather than noise. Lowering the gate to 10 was tried and reverted after measurement; see the constant's comment in `selection/step3_wrapper.py`.

Steps 1 and 2 have no auto-skip: they run whenever enabled. Every step passes through when its enable flag is False.

### 6.6 Consensus intersection and relaxation (Steps 1 to 3)

The floor throughout is `min(config.min_features_floor, n_features_in)`, capped further at the consensus ceiling (see METHODS_GUIDE, Minimum-feature floor).

**Steps 2 and 3** share `relax_consensus_intersection` in `utils/helpers.py`. It intersects one integer vote array per stage, starting at `required = n_models` (unanimity) and lowering `required` by one until the intersection reaches the floor, stopping early if the next rung would fall below `config.min_consensus`. Outcomes are `consensus`, `relaxed`, `consensus_limited`, or `rank_average`; each carries `votes_required`, `agreement` and `agreement_label`.

Rank-averaging is reachable only when the ceiling is zero, that is when no feature was selected by any replicate in every stage. It ranks within the union of features that received a vote somewhere, stepping outside that pool only if it cannot fill the request.

**Step 1** cannot use the shared helper: its two voters are the variance and correlation filters, each casting a single binary vote, so there are no intermediate rungs. Its ladder is intersection (2/2) then union (1/2) then top-k by variance (0/2). It reports the same keys so `pipeline._final_agreement()` can read it when Steps 2 and 3 are disabled or pass through.

---

## 7. Checkpoint / Resume System

### 7.1 What is checkpointed

`_save_checkpoint` (`pipeline.py`) pickles a dict with these keys:

| Key | Content |
|-----|---------|
| `step_key` | Last completed step (`'step0'` through `'step3'`) |
| `results` | `self.results` (all completed step result dicts) |
| `state_tracker` | Full `DataStateTracker`, including all `PipelineState` history and the reference cache |
| `_ref_result_step0` | Step 0 single-model CV reference result |
| `_X_train`, `_X_test` | Full-feature train and test matrices |
| `_y_train`, `_y_test` | Encoded labels |
| `_n_classes`, `_class_names` | Class count and names |
| `_label_encoder` | Fitted `LabelEncoder` |
| `_cv` | Shared `StratifiedKFold` |
| `_tuned_pipelines` | Tuned pipelines from `quick_tune_all` |
| `_original_n_features` | Original feature count |
| `sample_adequacy` | Adequacy assessment dict |

The dump uses pickle protocol 4 (`pickle.dump(checkpoint, fh, protocol=4)`).

### 7.2 Location

The checkpoint is written to `{config.output_dir}/.selectomics_checkpoint.pkl` (a dot-prefixed file), returned by `_checkpoint_path`.

### 7.3 When it is written

A checkpoint is written after each successfully completed selection step, only when `config.save_intermediate_results=True`:

```python
self.run_step0_reference(); self._save_checkpoint('step0')
# ... run_step1_cleaning(); self._save_checkpoint('step1') ... etc.
```

A step exception propagates before its checkpoint write, so a checkpoint reflects only steps that finished. Enabled/disabled step flags gate whether a step (and thus its checkpoint) runs at all.

### 7.4 How `resume=True` works

`run(resume=True)` calls `_load_checkpoint` first. If the file exists, it unpickles it, restores every instance attribute, and returns the last completed `step_key`. A `_skip` closure then decides which steps to bypass:

```python
_STEP_ORDER = ['step0', 'step1', 'step2', 'step3']

def _skip(step_key: str) -> bool:
    if last_completed is None:
        return False
    return _STEP_ORDER.index(step_key) <= _STEP_ORDER.index(last_completed)
```

Steps at or before `last_completed` are skipped; later steps run normally. `load_data` and `_assess_sample_size` are inside the `if not _skip('step0')` block, so on a successful resume they are skipped too (the loaded checkpoint already carries the split, tuned pipelines, and adequacy). The validation phase (`validate_features`, `_run_final_test_evaluation`) is never checkpointed and always re-runs after the selection steps.

### 7.5 Limitations

- Checkpoints are not cross-version safe: the pickle depends on the exact Python, scikit-learn, and XGBoost versions used to write it.
- With `save_intermediate_results=False`, no checkpoint is written and `resume=True` has no effect (no file to load).
- Both save and load are wrapped in try/except. A failed save logs a warning and does not interrupt the step; a failed load logs a warning and the pipeline starts fresh (`_load_checkpoint` returns `None`).

---

## 8. Config Validation

### 8.1 `__post_init__` validation

`SelectOmicsConfig` is a dataclass; `__post_init__` (`config.py`) runs immediately after construction. It coerces `data_path` and `output_dir` to `pathlib.Path`, then validates every field, raising `ValueError` on any violation:

| Field | Constraint |
|-------|-----------|
| `algorithm` | In `{'LR', 'XGB', 'RF', 'SVM'}` (`_VALID_ALGORITHMS`) |
| `input_format` | In `{'auto', 'csv', 'tsv', 'excel', 'parquet', 'hdf5', 'feather', 'json'}` |
| `output_format` | In `{'csv', 'tsv', 'excel', 'parquet', 'hdf5', 'feather', 'json'}` |
| `plot_format` | In `{'png', 'svg', 'pdf', 'tiff', 'jpeg'}` |
| `excel_output_engine` | In `{'openpyxl', 'xlsxwriter'}` |
| `test_size` | `0.0 < test_size < 1.0` |
| `n_consensus_models` | `>= 1` |
| `min_consensus` | `None`, or `0.0 < x <= 1.0` |
| `stability_threshold` | `0.0 < x <= 1.0` |
| `step2_target_retention` | `0.0 < x < 1.0` |
| `step3_target_retention` | `0.0 < x < 1.0` |
| `min_features_floor` | `>= 1` |
| `cv_splits` | If not `None`, `>= 2` |
| `min_cv_splits` | `>= 2` |
| `max_cv_splits` | `>= min_cv_splits` |
| `max_variance_threshold` | `> 0.0` |
| `min_correlation_threshold` | `0.0 < x < 1.0` |
| `quick_tune_iterations` | `>= 1` |
| `n_bootstrap` | `>= 10` |
| `outer_cv_splits` | `>= 2` |
| `schema_version` | Unknown versions emit a `UserWarning` (do not raise) |

The current known schema version is `'1.0'`. Note the defaults `min_cv_splits=3` and `max_cv_splits=20`.

The four `*_percentile_range` fields were removed. They bounded the domain of an internal bisection solver whose target is a retention fraction the caller sets separately, so they asked the user to constrain a solver for a quantity they never see, and the default upper bound of `0.6` silently capped retention at 40% (making any smaller `stepN_target_retention` unreachable). Searches now run over the full `[0, 1]` domain. With no tuple fields left in the dataclass, `to_dict` and `from_dict` no longer need list/tuple coercion.

### 8.2 `from_dict`

`from_dict` shallow-copies the input, applies schema migration (a no-op for version 1.0, with a documented extension point), then filters to recognised dataclass fields. Unknown keys are dropped with a `UserWarning` naming them (protecting against silent typos), which is also how a config still setting a removed percentile-range field announces itself.

`from_json` and `from_yaml` both route through `from_dict`, so they inherit both behaviours. `SelectOmicsPipeline.__init__` also routes a plain dict config through `from_dict`.

### 8.3 Preset constructors

Five classmethod presets simplify configuration:

- `quick(data_path, target_column, **kwargs)`: `n_consensus_models=5`, `quick_tune_iterations=10`, `enable_step3=False`, `n_bootstrap=10`.
- `standard(...)`: `n_consensus_models=10`, `quick_tune_iterations=25`, all steps enabled, `n_bootstrap=50`.
- `thorough(...)`: `n_consensus_models=20`, `quick_tune_iterations=50`, all steps enabled, `n_bootstrap=100`.
- `omics(...)`: tuned for the small-n / large-p regime; all steps enabled, `quick_tune_iterations=20`, `n_bootstrap=100`, plus the surviving thresholds from the configuration search (`stability_threshold=0.259`, `step3_target_retention=0.354`). **Provisional.** Five of the search's ten dimensions have since been shown to have no effect: the four removed `*_percentile_range` bounds, which it ranked last by influence (Spearman `r` between `-0.038` and `0.045`, p from 0.52 to 0.98), and `consensus_threshold`, also removed. Its `consensus_threshold=0.442` is not translated to `min_consensus`, since a starting point and a floor are different quantities. The dominant parameter, `min_features_floor`, was fixed at its default throughout and never searched. Re-derive with `benchmarks/config_search.py`.

  Its `n_consensus_models=5` override has been dropped, so the preset now inherits the default of 10. The re-run search ranks that parameter highest (Spearman `r = 0.835, p < 0.0001`, monotonic across 3, 5 and 10), and a preset aimed at the hardest regime should not have been the least stable configuration on offer.
- `suggest(data_path, target_column, **overrides)`: inspects the data file (sample count, feature count, missing-value fraction, class balance, minimum class size, n/p ratio) and selects defaults, logging a rationale for each decision. If the file cannot be read it falls back to `standard` defaults with a warning.

All presets forward `**kwargs` overrides (applied last) and build through `from_dict`. Serialisation helpers `to_dict`, `to_json`, `to_yaml`, `template_yaml`, `diff_from_defaults`, and a concise `__repr__` (showing only non-default fields) are also provided.

---

## 9. Pipeline State Machine

### 9.1 Execution order

`pipeline.run()` executes:

```
load_data()                                              [skipped on resume past step0]
_assess_sample_size()                                    [skipped on resume past step0]
run_step0_reference()      -> _save_checkpoint('step0')
run_step1_cleaning()       -> _save_checkpoint('step1')   [if enable_step1]
run_step2_regularization() -> _save_checkpoint('step2')   [if enable_step2]
run_step3_wrapper()        -> _save_checkpoint('step3')   [if enable_step3]
validate_features()                                       [if validate=True]
_run_final_test_evaluation()                              [if validate and enable_final_test_evaluation]
```

`rank_steps_on_holdout()` runs inside `validate_features` when `enable_holdout_ranking=True`, before `build_recommendation`, and its DataFrame is passed to that function as `holdout_scores`. Each fold comes from `StratifiedKFold` over `_X_train` alone, lowered to the smallest class size when necessary; the fold's inner pipeline is built with `_adopt_split` (training rows, the fold's held-out features, and no labels for them) and runs `_run_selection_steps`. Panels identical across steps are fitted once. A failure is logged and the ranking falls back to the training-split comparison.

`run_nested_cv()` runs after the steps above when `enable_nested_cv=True`, and can also be called on its own. It does not replace the single split: it adds `outer_cv_splits` folds, each rerunning the full configured procedure, Step 3 included, on its training portion.

### 9.2 State tracker updates

`DataStateTracker` (`utils/state.py`) holds an ordered, append-only list of immutable `PipelineState` snapshots (`step_name`, `X_train`, `X_test`, optional per-algorithm eval dicts). After each step, the pipeline appends a snapshot:

| Step method | X_train recorded | X_test recorded |
|-------------|------------------|-----------------|
| `run_step0_reference` | `self._X_train.copy()` | `self._X_test.copy()` |
| `run_step1_cleaning` | `step1_result['X_train_clean']` | `step1_result['X_test_clean']` |
| `run_step2_regularization` | `step2_result['X_train_reg']` | `step2_result['X_test_reg']` |
| `run_step3_wrapper` | `step3_result['X_train_rfecv']` | `step3_result['X_test_rfecv']` |

Each subsequent step reads `state = self.state_tracker.get_latest()` and consumes `state.X_train` / `state.X_test`. This is how the progressively reduced matrices flow through the pipeline without any global variables. `get_step(name)` returns the most recent snapshot with a given name; `get_latest()` raises `RuntimeError` if nothing is recorded; `clear()` resets both history and the reference cache.

### 9.3 Disabled and skipped steps pass through

A disabled step (`enable_stepN=False`) returns its input matrices unchanged, and the pipeline still records a `PipelineState` for it, so downstream steps see the same feature set. Pass-through return keys:

- Step 1: `X_train_clean = X_train.copy()`, all dropped lists empty.
- Step 2: `X_train_reg = X_train.copy()`, masks all-True.
- Step 3: `_passthrough` with `skipped=True`, `rfecv_selected` and `stability_selected` all-True.

Step 3 also auto-skips when fewer than 30 features remain (Section 6.5).

### 9.4 Final test evaluation

`_run_final_test_evaluation` picks the training CV result to report by searching completed steps from last to first:

```python
for key in reversed(['step3', 'step2', 'step1', 'step0']):
    if key in self.results:
        cr = self.results[key].get('consensus_result') or self.results[key].get('reference_result')
        if cr is not None and 'mean_auc' in cr:
            training_result = cr
            break
```

The model itself is always `clone(self._tuned_pipelines[alg])` refitted on `X_train_final` (the latest state). This fallback chain only chooses which training AUC to compare against test; it does not change the model or feature set. When `config.calibrate_probabilities` is True, the refitted model is wrapped in `CalibratedClassifierCV(method='isotonic', cv=min(3, max(2, min_class)))` and refitted before test prediction. The step then builds `comprehensive_metrics`, computes a generalization label, and (per config flags) saves metrics and plots.

### 9.5 `on_step_complete` callback

An optional `on_step_complete` callback passed to `SelectOmicsPipeline.__init__` is invoked via `_fire_callback` after each step, with signature `callback(step_key: str, result: dict) -> None`. Exceptions raised inside it are caught and logged as warnings, never aborting the pipeline (Section 10.6).

---

## 10. Error Handling and Logging

### 10.1 Logging setup

Every module uses `logger = logging.getLogger(__name__)`. The package `__init__` attaches a `NullHandler` to the top-level `SelectOmics` logger at import:

```python
logging.getLogger(__name__).addHandler(logging.NullHandler())
```

so importing SelectOmics produces no console output. `enable_logging(level=logging.INFO, handler=None, fmt=...)` activates console logging: it sets the logger level, defaults to a `StreamHandler`, and on repeated calls updates the existing handler's level and formatter in place (so `enable_logging('DEBUG')` after `enable_logging('WARNING')` actually raises the effective verbosity rather than leaving a stale handler level). `config.verbose` additionally gates many `logger.info`/`logger.debug` calls, but `logger.warning` calls (NaN switches, adequacy warnings, accumulated step warnings) always fire.

### 10.2 Step-level warning accumulation

Step 3 accumulates warning counts in a dict and reports once, rather than logging per occurrence. It tracks `rfecv_failed`, `stability_failed`, `stability_skipped_class`, and `stability_all_failed`, printed by `_print_warnings`. The dict is returned in the step result, reachable at `pipeline.results['step3']['warnings']`. Adequacy warnings are surfaced per step via `print_adequacy_warning`, which only prints when the assessment is non-adequate.

### 10.3 Visualization exception handling

Steps 0 to 3 call their plotting helpers only when `config.create_visualizations=True` and do not wrap them in a local try/except, so a matplotlib-level failure inside a step propagates rather than being swallowed.

### 10.4 Guard methods

Two private guards enforce ordering:

- `_require_data()` raises `RuntimeError` if `self._X_train is None` (i.e. `load_data` was never called).
- `_require_step0()` calls `_require_data()` then raises `RuntimeError` if `'step0'` is not in `self.results`.

They are called at the top of every step method and every accessor that depends on pipeline state. `get_feature_provenance` raises `RuntimeError` if Step 0 has not run.

### 10.5 Checkpoint failure isolation

`_save_checkpoint` and `_load_checkpoint` both wrap their pickle I/O in try/except. A save failure logs `"Checkpoint save failed: %s"` and lets the step continue; a load failure logs `"Checkpoint load failed (%s). Starting fresh."` and returns `None`, so the pipeline runs from scratch (Section 7.5).

### 10.6 Callback isolation

`_fire_callback` catches all exceptions from the user-supplied `on_step_complete`:

```python
try:
    self._on_step_complete(step_key, result)
except Exception as exc:
    logger.warning("on_step_complete callback raised an exception for '%s': %s", step_key, exc)
```

A misbehaving callback never corrupts pipeline state or aborts the run.
