# SelectOmics User Guide

**SelectOmics finds the right features in small-n, large-p data, not just fewer of them.**

LASSO will give you a short list. On the benchmark suite it returns 38 features, and about half of them are noise. SelectOmics returns 8, and 97% of them are real. It gets there by making every feature survive three independent selection criteria, data-based cleaning, regularisation and wrapper methods (RFECV with stability selection), each decided by a vote across many independently seeded models. It then re-validates every step's panel and recommends the one to use. The held-out test set is touched once, at the end, on that panel, so the performance you report comes from samples nothing in the pipeline has seen.

This guide covers version 0.7.1, a pre-release. Author: Amanda S Barnard. Python 3.9 or later is required.

### Is it right for your data?

Use it when a short, defensible panel is the deliverable. The evidence, from `benchmarks/BENCHMARKS.md`:

- **Precision 0.970**, against 0.501 for LASSO, 0.555 for SHAP and 0.232 for ElasticNet.
- **It does not invent signal.** On a null control with no informative features, it returns 5.8 false positives against LASSO's 53.6 and ElasticNet's 180.
- **The lead grows with p**: F1 0.915 at p of 5 000 or more, against 0.553 for the next-best method.
- **On real data it concedes almost nothing**: 18 features from 34 021 on TCGA lower-grade glioma at AUC 0.9885, against the best method's 0.9891.

Do not use it when sensitivity is the objective (recall 0.730, against LASSO's 0.893), when runtime is binding (about 100 s per trial against seconds for LASSO), or when n is large and the signal dense, where a default ElasticNet beats it. And do not expect it to be the most stable method on real data: it leads on stability on synthetic data, but on real TCGA data ElasticNet and LASSO return more consistent panels on every layer measured. The advantage that survives real data is parsimony.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Input Data](#input-data)
4. [Configuration Reference](#configuration-reference)
5. [Choosing an Algorithm](#choosing-an-algorithm)
6. [Choosing Key Parameters](#choosing-key-parameters)
7. [Configuration Presets](#configuration-presets)
8. [Auto-Suggested Configuration](#auto-suggested-configuration)
9. [Serialising and Loading Configurations](#serialising-and-loading-configurations)
10. [Running the Pipeline](#running-the-pipeline)
11. [CLI Usage](#cli-usage)
12. [Worked Examples](#worked-examples)

---

## Installation

SelectOmics is a pre-release and is not yet on PyPI. Install it from GitHub.

### From GitHub

```bash
pip install git+https://github.com/amaxiom/SelectOmics.git
```

### From a Clone (Development Install)

Use a clone if you plan to run the examples or the test suite:

```bash
git clone https://github.com/amaxiom/SelectOmics.git
cd SelectOmics
pip install -e ".[dev,viz]"
```

The examples are Jupyter notebooks that download their data on first run, so they also need Jupyter and a network connection. The package itself needs neither.

### Python Version Requirements

Python 3.9 or later is required. Tested on Python 3.9 through 3.13.

### Core Dependencies

The base install pulls in numpy, pandas, scikit-learn (>= 1.4), scipy, xgboost (>= 2.0), matplotlib, openpyxl, pyarrow, and pyyaml. These are sufficient for the full pipeline including CSV, TSV, Excel, Parquet, Feather, and JSON I/O, and for the YAML config files used by the CLI.

### Optional Dependency Groups

Install optional groups by appending the group name in brackets:

```bash
pip install "SelectOmics[viz] @ git+https://github.com/amaxiom/SelectOmics.git"   # from GitHub
pip install -e ".[viz]"                                                           # from a clone
```

| Group | Adds | When you need it |
|-------|------|------------------|
| `[viz]` | `seaborn >= 0.11.0` | Enhanced confusion matrix heatmaps. Without seaborn the confusion matrix falls back to a plain `imshow` rendering. |
| `[io]` | `tables >= 3.7.0`, `fastparquet >= 0.8.0` | Required to read or write HDF5 files (`tables`) and for the `fastparquet` engine with Parquet files. `pyarrow` (included in the base install) already handles Parquet. |
| `[dev]` | `pytest >= 7.0.0`, `pytest-cov >= 4.0.0` | Running the test suite and measuring code coverage. Not needed for analysis work. |

Groups can be combined:

```bash
pip install "SelectOmics[viz,io] @ git+https://github.com/amaxiom/SelectOmics.git"
```

---

## Quick Start

The minimal working example below loads data, constructs a configuration, runs the full pipeline, and retrieves results. Only `data_path` and `target_column` are required.

```python
import SelectOmics as selectomics

# Activate logging so you can see pipeline progress (optional).
selectomics.enable_logging()

# Minimal configuration dict: only data_path and target_column are required.
config = {
    "data_path":     "data/omics.csv",
    "target_column": "Class",
}

# Instantiate the pipeline.
pipeline = selectomics.SelectOmicsPipeline(config)

# Run all three selection steps plus validation and final test evaluation.
results = pipeline.run()

# The panel validation recommends. Usually the last step's, and an earlier
# step's when a later one pruned past the point where it helped.
features = pipeline.get_recommended_features()
print(f"Recommended {len(features)} features: {features[:5]} ...")

# The last step's panel, whatever validation said.
last_step = pipeline.get_selected_features()

# Save config.json, selected_features, recommended_features, and metrics.
pipeline.save_results()
```

With default settings, all output files are written to the `selectomics_results/` directory.

---

## Input Data

### Supported File Formats

The input format is auto-detected from the file extension when `input_format` is `"auto"` (the default).

| Format | Extensions auto-detected | `input_format` value |
|--------|--------------------------|----------------------|
| CSV | `.csv` | `csv` |
| TSV | `.tsv`, `.txt` | `tsv` |
| Excel | `.xlsx`, `.xls` | `excel` |
| Parquet | `.parquet`, `.pq` | `parquet` |
| HDF5 | `.h5`, `.hdf5` | `hdf5` |
| Feather | `.feather` | `feather` |
| JSON | `.json` | `json` |

Supported output formats (for tabular result files) are `csv`, `tsv`, `excel`, `parquet`, `hdf5`, `feather`, and `json`. Supported plot formats are `png`, `svg`, `pdf`, `tiff`, and `jpeg`.

### Format Auto-Detection vs Explicit Specification

When `input_format` is `"auto"`, the format is inferred from the extension. Set `input_format` explicitly when the file extension is non-standard or missing:

```python
config = {
    "data_path":     "data/omics_data",   # no extension
    "target_column": "Class",
    "input_format":  "csv",               # must specify explicitly
}
```

Format-specific settings:

- `excel_sheet_name` (default `0`) selects the sheet to read by 0-based index or name; ignored for non-Excel formats.
- `hdf5_key` (default `"data"`) is the dataset key within an HDF5 file; ignored for other formats.

### Required Data Structure

- **Rows** are samples (observations, patients, specimens).
- **Columns** are features plus one target column.
- The target column contains class labels (strings or integers). Binary and multiclass problems are both supported; at least two classes are required.
- There is no required column ordering; the target column can appear anywhere.

During preparation, the target column is separated first, then only **numeric** feature columns are retained (non-numeric columns are dropped automatically). The feature index is sorted for reproducibility.

Example (CSV):

```
SampleID,GeneA,GeneB,GeneC,Class
S001,1.23,0.45,2.11,Control
S002,0.89,1.67,0.34,Case
S003,2.01,0.22,1.88,Case
```

A non-numeric identifier column such as `SampleID` is dropped automatically because it is not numeric. Numeric identifier columns, however, would be treated as features, so remove them before running.

### Specifying the Target Column

Set `target_column` to the exact column header in the file:

```python
config = {
    "data_path":     "proteomics.csv",
    "target_column": "Diagnosis",   # must match the column header exactly
}
```

A `ValueError` is raised if the target column is not present, or if fewer than two classes are found.

### Missing Value Handling

SelectOmics detects missing values (`NaN`) during data loading and never imputes them:

- If the configured `algorithm` is `LR` or `SVM` and NaNs are present, the algorithm is automatically switched to `XGB`, which handles NaN natively via its tree-splitting logic. A warning is logged whenever this switch occurs (regardless of `verbose`).
- `XGB` and `RF` preserve NaN values unchanged, so the missingness pattern itself can serve as a signal.
- No imputation is performed. If you prefer LR or SVM on data with missing values, impute your data before passing it to SelectOmics.

---

## Configuration Reference

Parameters can be passed as a plain Python dictionary to `SelectOmicsPipeline`, or as a `SelectOmicsConfig` dataclass. The two approaches are equivalent:

```python
# Dict form (converted internally via SelectOmicsConfig.from_dict).
pipeline = selectomics.SelectOmicsPipeline({"data_path": "data.csv", "target_column": "Class"})

# Dataclass form.
from SelectOmics import SelectOmicsConfig
cfg = SelectOmicsConfig(data_path="data.csv", target_column="Class", algorithm="RF")
pipeline = selectomics.SelectOmicsPipeline(cfg)
```

Unknown dictionary keys are silently ignored with a `UserWarning`. Check for typos, as misspelled keys revert to their defaults.

The table below lists every configuration field with its default and description.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_path` | **required** | Path to the input data file (str or Path). |
| `target_column` | **required** | Name of the column containing class labels. |
| `schema_version` | `"1.0"` | Config schema version, used for forward/backward compatibility when loading serialised configs. |
| `input_format` | `"auto"` | Input file format: `auto`, `csv`, `tsv`, `excel`, `parquet`, `hdf5`, `feather`, `json`. `"auto"` infers from the file extension. |
| `hdf5_key` | `"data"` | Dataset key within an HDF5 file. Ignored for other formats. |
| `excel_sheet_name` | `0` | Excel sheet to read: 0-based index or sheet name. Ignored for non-Excel formats. |
| `output_dir` | `"selectomics_results"` | Directory for all output files. Created automatically if it does not exist. |
| `output_format` | `"csv"` | Format for tabular output files: `csv`, `tsv`, `excel`, `parquet`, `hdf5`, `feather`, `json`. |
| `excel_output_engine` | `"openpyxl"` | Pandas Excel write engine: `openpyxl` or `xlsxwriter`. Used only when `output_format="excel"`. |
| `save_intermediate_results` | `True` | Save per-step feature matrices, summaries, and the resume checkpoint. |
| `create_visualizations` | `True` | Generate and save all diagnostic plots. |
| `plot_format` | `"png"` | Format for saved figures: `png`, `svg`, `pdf`, `tiff`, `jpeg`. `jpeg` uses quality 95; all others use 300 DPI. |
| `verbose` | `False` | Print step-by-step progress and summary tables through the logger. |
| `algorithm` | `"XGB"` | Primary classifier used throughout: `LR`, `XGB`, `RF`, `SVM`. Auto-switched to `XGB` if NaNs are present and `algorithm` is `LR` or `SVM`. |
| `n_consensus_models` | `10` | Number of bootstrap-seeded consensus models per selection step. Higher counts improve stability at proportionally higher runtime. Must be `>= 1`. |
| `random_seed` | `42` | Master seed for all stochastic operations (split, CV shuffles, model init, bootstrap sampling). |
| `test_size` | `0.25` | Fraction of samples reserved for the held-out, stratified test set. Must be in `(0.0, 1.0)`. |
| `cv_splits` | `None` | Number of stratified CV folds. `None` triggers automatic selection (see below). If set explicitly, must be `>= 2` and is clamped to `[min_cv_splits, max_cv_splits]`. |
| `min_cv_splits` | `3` | Lower bound for automatic CV fold selection. Must be `>= 2`. |
| `max_cv_splits` | `20` | Upper bound for automatic CV fold selection. Must be `>= min_cv_splits`. |
| `quick_tune_iterations` | `50` | `RandomizedSearchCV` iteration budget for the initial quick-tune pass. Must be `>= 1`. |
| `use_gpu` | `True` | Attempt CUDA acceleration for XGBoost (needs a CUDA GPU and xgboost >= 2.0). Falls back to CPU silently. No effect for LR/RF/SVM. |
| `enable_step1` | `True` | Run Step 1 (data-based cleaning: variance/correlation). |
| `enable_step2` | `True` | Run Step 2 (model-based filters: L1/L2 regularization). |
| `enable_step3` | `True` | Run Step 3 (wrappers: RFECV + stability selection). |
| `enable_step_evaluations` | `True` | Generate per-step consensus model CV reports and comparison tables. |
| `enable_final_test_evaluation` | `True` | Evaluate the final feature set on the held-out test set with ROC and confusion plots. |
| `max_variance_threshold` | `0.30` | Upper bound for the Step 1 variance-threshold grid search. Features with variance at or below the selected threshold are dropped. Must be `> 0`. |
| `min_correlation_threshold` | `0.70` | Upper bound for the Step 1 correlation-threshold grid search (semantically a max). For a correlated pair at or above the selected threshold, the lower-variance feature is dropped. Must be in `(0.0, 1.0)`. |
| `class_aware_correlation` | `True` | Use the pooled within-class correlation matrix, prioritising features by the correlation ratio eta-squared, instead of total Pearson correlation prioritised by variance. Centring on class means strips the between-class covariance, so co-regulated markers stop looking redundant while genuine duplicates still do. On the synthetic case built to expose this, retention of 20 informative features went from 1/20 to 20/20. |
| `min_consensus` | `0.4` | Lowest acceptable agreement. Relaxation always starts at unanimity and stops here rather than descending further to reach `min_features_floor`. `None` removes the brake; `1.0` means unanimity or nothing. The default is measured, with an interior optimum. Must be in `(0.0, 1.0]` or `None`. |
| `stability_threshold` | `None` (resolves to `0.6`) | Floor on the fraction of Step 3 bootstrap subsamples in which a feature must be selected before it casts any stability vote. Meinshausen and Buhlmann (2010) recommend 0.6 to 0.9; `None` takes the permissive end of that range. It is a floor, not a target: features above it still face the full vote requirement and relax with it. Must be in `(0.0, 1.0]` or `None`. |
| `min_features_floor` | `15` | Absolute floor on what any step may return. Each step relaxes its consensus requirement until this is met, so together with `min_consensus` this is what sets panel size. Capped at the consensus ceiling. Must be `>= 1`. |
| `step2_target_retention` | `1.0` | Fraction of the **reachable pool** each regularization stage keeps, where the pool is the features the stage models actually used. `1.0` keeps the whole pool. The L1 and L2 stages are intersected, so the step returns fewer. Must be in `(0.0, 1.0]`. |
| `step2_stage_colsample` | `0.3` | Column-sampling rate for Step 2's XGB stage models, which sets the size of that pool. **Lower widens it.** No effect for LR/RF/SVM. Must be in `(0.0, 1.0]`. |
| `step2_role` | `"terminal"` | What Step 2 is for. `"terminal"` makes its own final cut and hands Step 3 a panel that is already nearly pure signal. `"prefilter"` deliberately keeps more and leaves the real cut to Step 3; it reaches the same recall as `allow_union_rung` but takes roughly nine times the runtime, and the two must not be combined. Must be `"terminal"` or `"prefilter"`. |
| `allow_union_rung` | `False` | Add a union rung beneath the intersection ladder in Steps 2 and 3, trading precision for recall. See the sensitivity section below for the measured cost. |
| `step3_target_retention` | `0.6` | Fraction of Step 3 input features to target for retention (binary search over RFECV `min_features_to_select`). Must be in `(0.0, 1.0)`. |
| `n_bootstrap` | `100` | Number of bootstrap iterations for `FeatureSetValidator`. Higher values reduce variance in the out-of-bag AUC estimate. Must be `>= 10`. |
| `calibrate_probabilities` | `False` | Wrap the final model in `CalibratedClassifierCV` (isotonic regression) before the test evaluation. Adds one inner CV pass. Recommended when predicted probabilities (not just rank order) are used downstream. |
| `enable_nested_cv` | `False` | When `True`, `run()` finishes with nested CV (also callable directly as `run_nested_cv()`): each outer fold reruns every enabled step, validation and the recommendation, then scores every step's panel on the held-out fold. Adds roughly `outer_cv_splits` full runs. |
| `outer_cv_splits` | `5` | Number of outer folds for nested CV. Must be `>= 2`. |
| `enable_holdout_ranking` | `False` | When `True`, the recommendation ranks the steps on folds held out of the **training** split instead of on validation that shares its samples with selection. Costs one extra run of the selection steps per fold. The test split is untouched either way. Off by default for a measured reason: an AUC ranking at n << p turns on noise and favours larger panels, so on seven datasets it changed three choices and each was worse on ground truth (`BENCHMARKS.md`, section 8). Switch it on when predictive AUC on new samples is the deliverable rather than a short panel. |
| `holdout_ranking_splits` | `5` | Number of ranking folds taken from the training split. Must be `>= 2`, and is lowered to the smallest class size when that is smaller. Fewer folds train each fold on less data, which makes the later steps look more aggressive than they are in the real run. |
| `n_jobs` | `-1` | Threads each estimator may use. `-1` uses every core, `1` forces serial, any positive integer sets an explicit cap. Does not affect results. Must be `-1` or `>= 1`. |
| `use_gpu` | `True` | Attempt CUDA acceleration for XGBoost. Falls back to CPU silently when unavailable. Unlike `n_jobs`, this **does** shift results slightly. |

### Parallelism and Graceful Degradation

SelectOmics uses every available core by default and needs nothing installed to
do so. It degrades rather than failing when the machine cannot cooperate:

| Situation | Behaviour |
|---|---|
| No CUDA GPU, or a driver mismatch | XGBoost runs on CPU, silently |
| Only one usable core reported | Runs serially |
| Matrix too small for threading to pay off | Fitted on a single thread |

**Thread count does not change results.** A run at `n_jobs=-1` selects exactly
the features a run at `n_jobs=1` selects, so `n_jobs` is purely a
speed-versus-courtesy dial. Set it to `1` when you are launching several
pipelines concurrently yourself, so they do not compete for the same cores.

This guarantee required pinning XGBoost to `tree_method='exact'`. Its default
(`hist`) draws subsample rows in a thread-order-dependent way, so the same seed
produced different trees at different thread counts. The pinned method is also
several times faster in the small-n regime this package targets.

`use_gpu` is the one setting that trades reproducibility for speed: the CUDA
implementation finds splits differently and shifts results slightly. Leave it
off for runs you intend to compare against CPU-generated numbers.

### Automatic CV Fold Selection

When `cv_splits` is `None`, the number of stratified folds is derived from the smallest class in the training set:

```
cv_splits = min(max_cv_splits, max(min_cv_splits, min_class_count_train))
```

with `min_cv_splits=3` and `max_cv_splits=20` by default. Setting `cv_splits` explicitly overrides this (still clamped to the min/max bounds).

---

## Choosing an Algorithm

SelectOmics supports four classifiers. `XGB` is the default.

### XGBoost (`XGB`), the default

Best for:

- Mixed data and non-linear feature interactions.
- Datasets with missing values: XGB handles `NaN` natively and is auto-selected whenever NaNs are present and the configured algorithm cannot cope.
- Medium-sized datasets (roughly 50 to a few thousand samples).

Considerations: more hyperparameters than RF; slightly longer tuning time. Can use GPU acceleration via `use_gpu`.

### Random Forest (`RF`)

Best for:

- Robustness to outliers and feature scaling (RF is scale-invariant).
- Preserving NaN handling without imputation (like XGB).
- Meaningful feature importance with minimal tuning.

Limitations: can overfit on very small datasets; importance can be biased toward high-cardinality features.

### Logistic Regression (`LR`)

Best for:

- Approximately linearly separable problems.
- Coefficient interpretability: L1/L2 coefficients map directly to feature weights.
- Larger sample sizes (n > 500).

Limitations: assumes linear relationships; cannot handle NaN (auto-switched to XGB when NaNs are present).

### Support Vector Machine (`SVM`)

Best for:

- Small-to-medium datasets.
- High-dimensional spaces where the number of features is large relative to sample count.

Limitations: probability calibration is approximate; scales poorly to large datasets; cannot handle NaN (auto-switched to XGB when NaNs are present).

### Automatic Fallback

When missing values are detected and the configured algorithm is `LR` or `SVM`, the algorithm is automatically switched to `XGB` and a warning is logged. `XGB` and `RF` are left unchanged because they preserve NaN natively.

---

## Choosing Key Parameters

### `n_consensus_models`

Controls the trade-off between selection stability and runtime. Each consensus model uses a distinct seed derived from `random_seed`; more models reduce the variance of consensus voting.

| Dataset size | Recommended | Notes |
|--------------|-------------|-------|
| Very small (n < 50) | 1 | Only one model is reliable at this scale. |
| Small (50 <= n < 150) | 3 to 5 | `suggest()` auto-sets 3. |
| Moderate (150 <= n < 500) | 5 to 10 | `suggest()` auto-sets 5. |
| Large (n >= 500) | 10 to 20 | `suggest()` auto-sets 10. |

### `min_consensus`

**How this works.** Every selection step starts by requiring *unanimity*: a feature survives only if every model, in every stage, voted for it. If that leaves fewer features than `min_features_floor`, the requirement drops one vote at a time until the floor is met. So the panel size is what you ask for, and the agreement level is what the data gives you in return.

`min_consensus` is the brake on that descent. Relaxation stops there even if the feature floor has not been reached.

- **`0.4` (default)**: stop before agreement gets thin.
- `0.8`: strong agreement or nothing.
- `1.0`: unanimity or nothing.
- `None`: no brake; relax as far as `min_features_floor` requires.

**The default is measured, and the response has an interior optimum.** Composite objective by setting, over 205 configurations on two scenarios:

| setting | `None` | **0.4** | 0.6 | 0.8 | 1.0 |
|---|---|---|---|---|---|
| composite | 0.751 | **0.784** | 0.743 | 0.599 | 0.503 |

A second phase on three scenarios confirmed it, with 0.4 winning on **every scenario individually** rather than only on the mean.

Read the direction carefully: `0.4` is **stricter** than no brake, not looser. Without it the relaxation can bottom out at one vote in ten, and that costs both accuracy and reproducibility. But demanding unanimity is worse still (F1 0.340, Kuncheva 0.127 at `1.0`), because almost nothing survives. Enforcing agreement helps; demanding all of it does not.

Use it when a defensible panel matters more than a full one. Setting `min_consensus=1.0` with `min_features_floor=10` says "give me ten features, but only if all models agree on them; otherwise give me the ones they do agree on and tell me." That case reports `consensus_limited` with both numbers, so the conflict is visible rather than silently resolved.

You cannot specify both the panel size and the agreement level and expect both: they trade against each other and the data decides where the trade lands. That is why there is no parameter for "how conservative should the pipeline be", see the reporting section below, which tells you what you actually got.

> **Replaced `consensus_threshold`.** That parameter set where the relaxation *started*, and was measured to be inert: because the descent stops at whichever level first meets the feature floor, starting lower only skipped rungs on the way to the same answer. On two scenarios at two seeds, thresholds of 0.4, 0.6, 0.8 and 1.0 returned the *identical feature set*. It also made the reported agreement level depend on the starting point rather than on the data. There is no translation between the two settings, since a starting point and a floor are different things.

### Reading how conservative your result was

Because you no longer *ask* for an agreement level, the pipeline *reports* one. Steps 2, 3 and 4 each return:

- `votes_required`: how many models had to agree. Since relaxation always starts at unanimity, this is the **strongest** level at which your panel holds.
- `agreement`: that as a fraction.
- `agreement_label`: the same in words.

| achieved | label | how to read it |
|---|---|---|
| 100% | `unanimous` | every model agreed on every feature |
| 80 to 99% | `strong` | a solid claim |
| 60 to 79% | `moderate` | defensible, worth stating alongside the panel |
| 40 to 59% | `weak` | the models disagree substantially |
| below 40% | `minimal` | the panel size was reached by relaxing almost all the way |
| fallback | `none (ranked, not agreed)` | no consensus existed; features are ranked by importance only |

```python
step2 = pipeline.results['step2']
print(f"{step2['n_features_out']} features at "
      f"{step2['votes_required']}/{step2['n_models']} agreement "
      f"({step2['agreement_label']})")
```

The final metrics JSON carries `achieved_agreement` and `agreement_label` for the delivered panel.

**Expect this to vary by algorithm.** On the same data at 5 models, Random Forest typically settles at `strong` while XGBoost and SVM reach only `minimal`, because RF gives non-zero importance to far more features (247 to 696 against XGBoost's 22) and so has a much wider pool to agree about. A `minimal` label is not necessarily a bad result, but it is a weaker claim, and it belongs in any write-up alongside the feature list.

### `test_size`

Standard choices are 0.2 to 0.25 (default `0.25`). The split is stratified, so class proportions are preserved.

- Smaller datasets (n < 100): reduce to 0.15 or 0.10 to leave more data for training and CV.
- Larger datasets (n > 1000): 0.2 gives a large enough absolute test set.

### Retention Targets (`step2_target_retention`, `step3_target_retention`)

These are the knobs for "too aggressive" or "too conservative". Each sets the fraction of a step's input that the step aims to keep, and a binary search over the full percentile domain converges to the threshold that delivers it.

- Step 2 default `1.0` keeps the whole reachable pool per regularization stage. The L1 and L2 stages are then intersected, so Step 2 as a whole returns fewer.

  **Step 2's target is a fraction of the reachable pool, not of the input.** A tree assigns non-zero importance only to features it splits on, and the step never selects past that set, because doing so would mean selecting features no model used. On wide data that set is around 1% of the input: measured, one XGB stage fit used 21 of 2000 features and 30 of 10 000. Until 0.7.0 this field was a fraction of the *input* width, so any value above roughly 1% asked for more features than existed and clamped to the same answer. Settings of 0.3, 0.6 and 0.9 produced identical output on 8 of 10 measured scenario-seed cells. It is now a fraction of the pool and live across its whole range.

  The pool size itself is set by `step2_stage_colsample`, and lowering that widens it. That is measured, not assumed: at colsample 1.0 every tree sees every feature and keeps splitting on the same strongest few, so fewer distinct features are used overall.

  | colsample | 0.1 | 0.3 | 0.6 | 1.0 |
  |---|---|---|---|---|
  | pool, `omics_standard` (p=2000) | 61 | 21 | 21 | 20 |
  | pool, `omics_genomics` (p=10 000) | 70 | 30 | 25 | 23 |

  Widening is not free: at 0.1 the extra features were largely noise and Step 2 precision fell from 1.000 to 0.857. The default sits near the knee.

  **Neither of these is the main control on panel size.** Measured across 5 scenarios at 2 seeds, `min_consensus` changed the result in 10 of 10 cells, `min_features_floor` in 7 of 10, and `step2_target_retention` in 2 of 10. If a run is too aggressive or too conservative, reach for `min_consensus` first.
- Step 3 default `0.6` retains about 60% of features entering Step 3. Use higher (0.8 to 0.9) when Step 2 already reduced heavily; lower (0.3 to 0.5) for a large cut.

A target is an upper bound, not a promise. A step cannot retain more features than its models could distinguish between, and for tree algorithms on wide data that ceiling is far below any sane target: a single XGBoost fit on 2000 features typically gives non-zero importance to about 20 of them, so a 60% target saturates at roughly 1%. When that happens the step logs it and a `UserWarning` names the step, the target and the closest achievable value. Read that warning as "the retention setting is not what determined this result" and reach for `min_features_floor` instead.

**Removed in this version.** `l1_percentile_range`, `l2_percentile_range` and `rfecv_percentile_range` no longer exist. They bounded the search domain rather than expressing anything about your data or your result, and they could silently put the retention you asked for out of reach: the default upper bound of `0.6` capped retention at 40%, so any smaller target was unreachable. The configuration search that tuned them ranked them last by influence, at Spearman `r` between `-0.038` and `0.045` with p-values from 0.52 to 0.98. Searches now run over the full `[0, 1]` domain. A config that still sets one of these fields loads with the usual unknown-key warning.

### `min_features_floor`

The absolute floor on what any step may return, and **in practice the setting that determines panel size**. Because each step relaxes its consensus requirement one vote at a time until the floor is met, the floor is what to change when the pipeline returns too few features. `consensus_threshold` sets where that relaxation starts, not where it stops. The level actually reached is reported as `votes_required` and tells you how much agreement the data supported at that panel size.

Measured with every other setting held fixed, over all six development scenarios at five seeds, so the floor is the only thing varying:

| `min_features_floor` | composite | F1 | Cross-seed Kuncheva | runtime |
|---|---|---|---|---|
| 5 | 0.716 | 0.624 | 0.490 | 162 s |
| 8 | 0.784 | 0.714 | 0.614 | 160 s |
| 10 | 0.808 | 0.745 | 0.660 | 159 s |
| **15 (default)** | **0.820** | 0.760 | **0.683** | **157 s** |
| 20 | 0.819 | 0.763 | 0.674 | 327 s |
| 25 | 0.820 | 0.764 | 0.672 | 389 s |

Monotonic to 15, then flat: 15, 20 and 25 differ by less than 0.001. What separates them is cost: floors of 20 and above pass enough features to wake Step 3, the most expensive step, for no measurable gain.

The improvement from 10 to 15 comes almost entirely from `omics_imbalanced` (F1 0.818 → 0.918). The other five scenarios tie exactly.

**Lower it when you expect a compact signal**, since a floor above the true number of informative features has to pad the panel. If your panels are the wrong size, this and `min_consensus` are the two controls; `min_consensus` is the one that also changes how strong a claim the panel carries.

**Raising it above 15 buys nothing but runtime.** Floors of 20 and 25 score identically to 15 while costing two to two and a half times as much, because they pass enough features to wake Step 3.

A floor larger than the consensus ceiling cannot be honoured: it is capped at the number of features some model actually selected, and the step logs the capping when it happens. This is why a floor of 30 can still return 12 features.

### `stability_threshold`

Governs the Step 3 stability selection filter: a feature must appear in at least this fraction of bootstrap subsamples to survive.

- `0.9 to 1.0`: very strict; few but highly reproducible features.
- `0.6 to 0.8`: recommended by Meinshausen and Buhlmann (2010) for theoretical control of expected false positives.
- Below 0.5: no stronger guarantee than a single model run.

### If you need sensitivity (`allow_union_rung`)

SelectOmics is built to return few features and be right about them: measured
mean precision 0.970 at recall 0.730 (`benchmarks/BENCHMARKS.md`, section 2.1).
If missing a true feature costs you more than carrying a false one, one flag
trades in the other direction.

`allow_union_rung=True` adds a union rung beneath the intersection ladder in
Steps 2 and 3, so the ceiling becomes "kept by some replicate in at least one
stage" rather than "in every stage". Measured over 10 signal-bearing scenarios
at 5 seeds:

| Setting | F1 | Precision | Recall | Features | Runtime |
|---|---:|---:|---:|---:|---:|
| default | 0.849 | 0.902 | 0.844 | 10.8 | 145 s |
| `allow_union_rung=True` | 0.755 | 0.726 | **0.897** | 15.4 | 257 s |

```python
config = SelectOmicsConfig(
    data_path="data/omics.csv",
    target_column="Class",
    allow_union_rung=True,      # recall over precision
)
```

Three things to know before using it.

**Do not combine it with `step2_role="prefilter"`.** That pairing was measured
at precision 0.318 and F1 0.411, the worst of the four arms tested. `prefilter`
on its own reaches the same recall as the union rung but takes nine times the
runtime to do it.

**It is a loss in the sparse high-p regime**, which is the regime this package
targets. Precision fell from 0.752 to 0.247 at p=5 000 with 5 informative
features, and from 1.000 to 0.655 at p=10 000, the latter buying no recall at
all because recall was already 1.000. It paid off at p=200 (recall 0.62 to
0.80 at unchanged precision 1.000) and at p=1 000 (0.80 to 0.947 for 9
precision points). Use it when recall is genuinely short, not by default.

**It is less disciplined about noise.** On a null control with no informative
features it returned 17.0 false positives against 9.4 for the default.

Everything else is unchanged: the same evaluations run, the same
recommendation is made, and the held-out test is spent the same way. Section
7.3 of `BENCHMARKS.md` carries the per-scenario numbers.

---

## Configuration Presets

`SelectOmicsConfig` provides four preset classmethods that set sensible parameter groups for different time/quality trade-offs. Any field can be overridden with keyword arguments. All four presets are also available on the CLI via `--preset {quick,standard,thorough,omics}`.

| Preset | `n_consensus_models` | `quick_tune_iterations` | `n_bootstrap` | Steps | Notable overrides |
|--------|----------------------|-------------------------|---------------|-------|-------------------|
| `quick` | 5 | 10 | 10 | 1 and 2 only (3 off) | Fast exploratory pass. |
| `standard` | 10 | 25 | 50 | all three | Balanced default. |
| `thorough` | 20 | 50 | 100 | all three | High-stability, publication-quality. |
| `omics` | 5 | 20 | 100 | all three | Benchmark-tuned thresholds for small-n/large-p (see below). |

```python
from SelectOmics import SelectOmicsConfig

cfg = SelectOmicsConfig.quick("data.csv", "Class")
cfg = SelectOmicsConfig.standard("data.csv", "Class")
cfg = SelectOmicsConfig.thorough("data.csv", "Class")
cfg = SelectOmicsConfig.omics("data.csv", "Class")

# Override any field on a preset:
cfg = SelectOmicsConfig.quick("data.csv", "Class", algorithm="RF", verbose=True)
```

### The `omics` Preset

The `omics` preset carries thresholds found by a four-phase configuration search (winning config `h_071`) for the small-n / large-p regime (n roughly 50 to 200, p roughly 500 to 10,000). In addition to the counts above it sets:

- `stability_threshold = 0.259`
- `step3_target_retention = 0.354`

Prefer `standard` or `thorough` for large-n or low-dimensional data.

**These values are provisional and need re-deriving.** The search that produced them tuned ten parameters, five of which have since been shown to have no effect: the four `*_percentile_range` bounds, and `consensus_threshold`. A sixth, `step4_target_retention`, belonged to a step that no longer exists. Most of the search budget therefore went on noise, and the remaining values were fitted around it.

Worse, the parameter that actually dominates the result, `min_features_floor`, was fixed at its default throughout and never searched at all.

The preset's `consensus_threshold = 0.442` is **not** carried over as a `min_consensus` value. The two are different quantities: one was a starting point for the relaxation, the other is a floor on it, so no value of one means the same as a value of the other.

Re-run the search with `benchmarks/config_search.py` before relying on these numbers.

---

## Auto-Suggested Configuration

`SelectOmicsConfig.suggest(data_path, target_column, **overrides)` inspects the dataset (shape, missingness, class balance) and returns a config with a printed rationale. Use it as a starting point when you are unsure which settings to use, then review and adjust before running.

```python
from SelectOmics import SelectOmicsConfig
import SelectOmics
SelectOmics.enable_logging()   # so the rationale is printed

cfg = SelectOmicsConfig.suggest("data.csv", "Class")

# Override individual fields after suggestion:
cfg = SelectOmicsConfig.suggest("data.csv", "Class", algorithm="RF", output_dir="my_results")
```

What `suggest()` adjusts automatically:

- `algorithm`: set to `XGB`, with a note stating whether missing values were detected.
- `n_consensus_models` and `n_bootstrap`: scaled to sample size (1 model / 30 bootstraps for n < 50; 3 / 50 for n < 150; 5 / 100 for n < 500; 10 / 200 for n >= 500).
- `enable_step3`: disabled for n < 50, and when fewer than 30 features remain (it requires >= 30 features).
- `quick_tune_iterations`: reduced to 30 for p > 10,000.
- Warnings for severe class imbalance, very small minimum class sizes, multiclass problems, and extreme n/p ratios.

If the data cannot be inspected, `suggest()` falls back to the standard defaults and logs a note. Any caller overrides are applied last.

Save the suggestion to YAML for review and editing:

```python
cfg = SelectOmicsConfig.suggest("data.csv", "Class")
cfg.to_yaml("my_config.yaml")
# Edit my_config.yaml, then:
cfg = SelectOmicsConfig.from_yaml("my_config.yaml")
```

---

## Serialising and Loading Configurations

Configurations can be saved and reloaded in JSON or YAML.

```python
from SelectOmics import SelectOmicsConfig

cfg = SelectOmicsConfig("data.csv", "Class", algorithm="RF", n_consensus_models=15)

# JSON round-trip.
cfg.to_json("config.json")
cfg2 = SelectOmicsConfig.from_json("config.json")

# YAML round-trip (requires PyYAML: pip install pyyaml).
cfg.to_yaml("config.yaml")
cfg3 = SelectOmicsConfig.from_yaml("config.yaml")
```

Tuple fields are stored as lists in JSON/YAML and restored to tuples automatically on load. Unknown keys are ignored with a `UserWarning`, so configs from older versions remain loadable.

### Commented Template

`template_yaml(path=None)` generates a fully-commented YAML template listing every field with its default, valid range, and a description. Writes to disk when a path is given, and always returns the template string.

```python
# Write a commented template to disk.
SelectOmicsConfig.template_yaml("template.yaml")

# Or get the string without writing.
text = SelectOmicsConfig.template_yaml()

# Edit template.yaml, then load it:
cfg = SelectOmicsConfig.from_yaml("template.yaml")
```

### Inspecting Non-Default Fields

`diff_from_defaults()` returns a dict of fields that differ from the defaults (excluding `data_path`, `target_column`, and `schema_version`). The `repr()` of a config shows only these non-default fields.

```python
cfg = SelectOmicsConfig("data.csv", "Class", algorithm="RF")
print(cfg.diff_from_defaults())   # {'algorithm': 'RF'}
print(repr(cfg))                  # compact repr showing only non-default fields
```

---

## Running the Pipeline

### Full Run

```python
import SelectOmics as selectomics

pipeline = selectomics.SelectOmicsPipeline({
    "data_path":     "data.csv",
    "target_column": "Class",
})
results = pipeline.run()
```

`run(validate=True, resume=False)` executes, in order: `load_data()`, sample-size assessment, `run_step0_reference()`, `run_step1_cleaning()`, `run_step2_regularization()`, `run_step3_wrapper()`, and (when `validate=True`) `validate_features()` and the final held-out test evaluation. Each selection step is conditional on its `enable_stepN` flag. `run()` returns `self.results`.

```python
results = pipeline.run(validate=False)   # skip validation and test evaluation
```

The constructor accepts either a dict or a `SelectOmicsConfig` instance. It also accepts an optional `on_step_complete` callback invoked at the end of each completed step with `(step_key, result_dict)`; exceptions raised inside the callback are caught and logged so they never abort the pipeline.

### Resume from Checkpoint

When `save_intermediate_results=True` (the default), a checkpoint file is written to `output_dir/.selectomics_checkpoint.pkl` after each completed step. If a run is interrupted, resume without rerunning completed steps:

```python
pipeline = selectomics.SelectOmicsPipeline(config)
results = pipeline.run(resume=True)
```

The checkpoint is stamped with the package version, a fingerprint of the source data, and the algorithm. A resume is refused, with a message naming which stamp failed, when any of the three has changed since the checkpoint was written; the run then starts fresh. Each stamp exists for the same reason: a checkpoint carries the tuned pipelines and every completed step's result, so resuming across a change would deliver one run's panel under another run's description.

### Stepwise / Fluent Execution

Every step method returns `self`, enabling method chaining:

```python
pipeline = selectomics.SelectOmicsPipeline(config)

(
    pipeline
    .load_data()
    .run_step0_reference()
    .run_step1_cleaning()
    .run_step2_regularization()
    .run_step3_wrapper()
    .validate_features()
)
```

Methods can also be called individually for fine-grained inspection between steps:

```python
pipeline = selectomics.SelectOmicsPipeline(config)
pipeline.load_data()
pipeline.run_step0_reference()
pipeline.run_step1_cleaning()

# How many features remain after Step 1?
print("After Step 1:", len(pipeline.get_selected_features("step1")), "features")

pipeline.run_step2_regularization()
pipeline.run_step3_wrapper()
pipeline.validate_features()
```

`load_data()` must be called before Step 0, and `run_step0_reference()` before Steps 1 to 3; calling out of order raises `RuntimeError`. Step 3 is skipped automatically when fewer than 30 features remain.

### Public Methods

| Method | Description |
|--------|-------------|
| `load_data()` | Load and preprocess data, handle NaN (may auto-switch algorithm), split train/test, build the CV object, and quick-tune the primary model. Returns `self`. |
| `run_step0_reference()` | Evaluate a single reference model on the full feature set to establish baseline CV AUC. Returns `self`. |
| `run_step1_cleaning()` | Remove constant, low-variance, and highly correlated features. Returns `self`. |
| `run_step2_regularization()` | Apply parallel L1/L2 consensus filtering. Returns `self`. |
| `run_step3_wrapper()` | Apply RFECV with stability selection. Returns `self`. |
| `validate_features()` | Run multi-method validation across all completed steps (training split only). Returns `self`. |
| `run(validate=True, resume=False)` | Execute the full pipeline end to end. Returns the results dict. |
| `run_nested_cv()` | Estimate how the whole procedure generalises, step choice included. Each outer fold reruns the configured steps, validation and the recommendation on its training portion and scores every step's panel on the held-out fold. Returns `mean_auc` / `std_auc` for the recommended panel, plus `fold_details`, `step_summary`, `last_step_mean_auc`, `best_step_mean_auc` and `mean_regret`. |
| `get_selected_features(step=None)` | Return the feature names retained at a given step, or the latest step if `step` is `None`. |
| `get_recommended_features()` | Return the panel the validation recommends, which can be an earlier step than the last. Requires `run(validate=True)`. |
| `rank_steps_on_holdout()` | Score every step's panel on folds held out of the training split, so no panel is scored on the samples that selected it. Returns a DataFrame of `holdout_auc`, `holdout_std`, `holdout_folds` and `holdout_n_features` per step. Used automatically when `enable_holdout_ranking=True`. |
| `get_feature_provenance()` | Return a per-feature survival table across all completed steps. |
| `save_results(output_dir=None)` | Save the config, the last step's feature list (`selected_features`), the recommended panel (`recommended_features`, when there is one), and metrics to disk. |

### `get_selected_features(step=None)`

Returns a list of feature column names. Valid step keys are `"step0"` through `"step3"`. `None` returns the most recently completed step. Raises `ValueError` for an unrun or invalid step.

```python
features   = pipeline.get_selected_features()          # latest step
after_step1 = pipeline.get_selected_features("step1")
after_step3 = pipeline.get_selected_features("step3")
```

### `get_feature_provenance()`

Returns a DataFrame showing whether each feature survived each completed step, with columns `feature`, `survived_step1`, `survived_step2`, `survived_step3`, `n_steps_survived`, and `first_dropped_at`. Rows are sorted by `n_steps_survived` descending, then by feature name. `first_dropped_at` is the step key where a feature was first removed, or `None` if it survived all completed steps. Raises `RuntimeError` if no steps have been run.

```python
prov = pipeline.get_feature_provenance()
print(prov.head(10))
```

### `save_results(output_dir=None)`

Persists outputs to `output_dir` (defaults to `config.output_dir`):

1. `config.json`: the serialised pipeline configuration.
2. `recommended_features.<ext>`: the panel the pipeline evaluation recommends, written whenever validation produced a recommendation. **This is the deliverable.**
3. `selected_features.<ext>`: a one-column DataFrame (`feature`) with the last step's feature names, in the format given by `output_format`. Identical to the recommended panel unless a later step pruned past the point where it helped.
4. `final_<ALGO>_comprehensive_metrics.json`: full test-set metrics for the recommended panel, only if the final test evaluation ran. `<ALGO>` is the algorithm actually used, so a NaN-triggered switch to XGB is visible in the filename.

```python
pipeline.save_results()                          # uses config.output_dir
pipeline.save_results(output_dir="custom_dir")   # override directory
```

### Enabling Logging

By default the SelectOmics logger has a `NullHandler`, so importing the package produces no console output. Call `enable_logging(level)` once to activate console logging:

```python
import SelectOmics
import logging

SelectOmics.enable_logging()               # INFO to stderr
SelectOmics.enable_logging(logging.DEBUG)  # verbose
SelectOmics.enable_logging("WARNING")      # string form accepted

# Custom handler, e.g. file logging.
handler = logging.FileHandler("selectomics.log")
SelectOmics.enable_logging(handler=handler)
```

`verbose=True` in the config causes the pipeline to emit extra `logger.info`/`logger.debug` calls; those still require `enable_logging()` to reach the console.

---

## Worked Examples

`examples/` carries twelve notebooks on two TCGA cohorts (glioblastoma and
ovarian carcinoma), four omic layers each plus the merged matrix, with an index
in `examples/README.md`. Every notebook enables all evaluations and delivers
the recommended panel. Each was executed end to end before shipping and then
cleared, so what you read is what ran.

| Notebook | Runs in | Read it for |
|---|---|---|
| `GS-GBM/GBM_Quickstart.ipynb` | 4 min | The shortest path from a CSV to a panel |
| `GS-OV/OV_SelectOmics_Showcase.ipynb` | 20 min | Every evaluation module, baselines at matched AUC, step ablation, nested CV, checkpoint resume |
| Ten per-layer notebooks | 18 to 102 min | One layer each, identical structure, so layers and cohorts compare directly |

The data is not committed. `examples/mlomics_data.py` downloads it from a
pinned, checksummed MLOmics revision on first use, so a clone stays small and
the numbers below are reproducible from the published source.

Four things the per-layer runs demonstrate that are hard to see from a single
dataset:

- **The recommendation overrides the last step when a later step prunes too
  far.** On OV miRNA, Step 3 cuts Step 2's 115 features to 10 and the
  recommendation keeps the 115. The run logs that the delivered panel is not
  the last step's.
- **Step 3 usually does not run.** It skips when fewer than 30 features reach
  it, which is seven of the ten per-layer notebooks. OV Methy is the near-miss
  worth studying: Step 2 returns 29, one short of the gate, so the most
  expensive step is skipped on a single feature's margin.
- **The held-out test catches optimism that no training-split statistic can.**
  OV CNV validated well and scored 0.584 on unseen samples, a gap of 0.226
  labelled `poor`. Read the gap, not just the AUC.
- **Runtime follows Step 3, not the feature count.** OV CNV is the slowest at
  102 minutes because Step 2 hands Step 3 seventy features, while the
  34 000-feature merged matrices finish sooner because Step 2 cuts them below
  the gate and Step 3 never runs at all.

---

## CLI Usage

After installation, the `selectomics` command is available on the PATH.

```
selectomics [--verbose] [--version] <command> [options]
```

| Command | Description |
|---------|-------------|
| `suggest` | Inspect a dataset and print or save a recommended configuration. |
| `template` | Generate a fully-commented YAML configuration template. |
| `run` | Execute the SelectOmics pipeline. |

Global flags: `--verbose` / `-v` enables DEBUG-level logging; `--version` prints the version and exits.

`--set KEY=VALUE` overrides are auto-typed: `true`/`false` become booleans, numeric strings become int or float, everything else stays a string. `--set` is repeatable.

### `selectomics suggest`

```
selectomics suggest <data> --target <column> [--output <file.yaml>] [--set KEY=VALUE ...] [--verbose]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `data` | yes | Path to the input data file. |
| `--target` / `-t` | yes | Target column name. |
| `--output` / `-o` | no | Write the suggested config to this YAML file. If omitted, key fields are printed to stdout. |
| `--set KEY=VALUE` | no | Override a suggested field. Repeatable. |

Examples:

```bash
# Print suggested config fields to stdout.
selectomics suggest data.csv --target Disease

# Save suggested config to a YAML file.
selectomics suggest data.csv --target Disease --output config.yaml

# Override the algorithm suggestion.
selectomics suggest data.csv --target Disease --output config.yaml --set algorithm=RF
```

### `selectomics template`

```
selectomics template [--output <file.yaml>]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--output` / `-o` | no | Write the template to this file. If omitted, prints to stdout. |

Examples:

```bash
# Print template to stdout.
selectomics template

# Write to a file for editing.
selectomics template --output my_config.yaml
```

### `selectomics run`

```
selectomics run [<data>] [--target <column>] [--preset <preset>] [--config <file>]
                [--output <dir>] [--resume] [--set KEY=VALUE ...] [--verbose]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `data` | conditional | Path to the input data file. Required unless `--config` is provided. |
| `--target` / `-t` | conditional | Target column name. Required unless `--config` is provided. |
| `--preset` / `-p` | no | Starting preset: `quick`, `standard` (default), `thorough`, or `omics`. |
| `--config` / `-c` | no | Path to a YAML or JSON config file. Overrides `--preset` and the positional data argument. |
| `--output` / `-o` | no | Override `output_dir` in the configuration. |
| `--resume` | no | Resume from an existing checkpoint in `output_dir`. |
| `--set KEY=VALUE` | no | Override any config field. Repeatable and auto-typed. |

Examples:

```bash
# Run with the standard preset (default).
selectomics run data.csv --target Disease

# Run with the quick preset and a custom output directory.
selectomics run data.csv --target Disease --preset quick --output results/

# Run with the omics preset (benchmark-tuned small-n/large-p thresholds).
selectomics run data.csv --target Disease --preset omics

# Run with the thorough preset and a specific algorithm.
selectomics run data.csv --target Disease --preset thorough --set algorithm=RF

# Run from a saved YAML or JSON config.
selectomics run --config my_config.yaml

# Run from a config file, overriding the output directory.
selectomics run --config my_config.yaml --output new_results/

# Resume a previously interrupted run.
selectomics run --config my_config.yaml --resume

# Override multiple fields at once.
selectomics run data.csv --target Class \
    --set n_consensus_models=5 \
    --set min_consensus=0.8 \
    --set enable_step3=false
```

After a run completes, the CLI saves results and prints a summary:

```
==================================================
SelectOmics complete.
  Selected features: 42
  Results saved to:  selectomics_results
  Final test AUC:    0.9123
==================================================
```
