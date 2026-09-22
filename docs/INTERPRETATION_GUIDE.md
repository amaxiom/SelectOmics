# SelectOmics Output Interpretation Guide

This guide explains how to read and interpret every output file, plot, and metric produced by the SelectOmics pipeline. Its focus is interpretation: what each figure panel shows, what each column and JSON key means, and how to diagnose a run from its outputs.

Every claim about column names, thresholds, and computation methods is grounded directly in the current source code (`evaluation/visualization.py`, `evaluation/metrics.py`, `evaluation/validation.py`, `pipeline.py`, `selection/step3_wrapper.py`, and `utils/diagnostics.py`).

Concrete numbers quoted throughout come from a run of the bundled GBM miRNA example (`examples/GS-GBM/GBM_SelectOmics_miRNA.ipynb`): a 5-class glioblastoma miRNA dataset, algorithm `XGB`, 5 consensus models. The tables and figures reproduced below are from an older run of that notebook that reduced 325 features to 47, kept here because they illustrate the output *formats*. The current notebook reduces 286 features to 8 (286 to 157 to 110 to 8) and delivers Step 3's panel at a held-out AUC of 0.819. Re-running it regenerates the directory these examples describe; `examples/README.md` lists what every notebook currently produces.

The input is 286 rather than 325 features because the examples now fetch their data from a pinned MLOmics revision rather than from copies committed here; see `examples/README.md`.

The pipeline runs four stages, each of which appears in the outputs by name:

- Step 0 (Reference): baseline model on all features.
- Step 1 (Data Cleaning): remove constant, low-variance, and highly correlated features.
- Step 2 (Regularization): parallel L1 and L2 consensus filtering.
- Step 3 (Wrappers): RFECV with stability selection.

File extensions depend on config: tables are written in `config.output_format` (CSV by default; Parquet or JSON also supported), and figures in `config.plot_format` (PNG by default). Throughout this guide, files are referred to with a `.*` extension to mean "in the configured format". The example outputs use `.csv` and `.png`.

---

## Table of Contents

1. [Understanding the Pipeline Summary](#1-understanding-the-pipeline-summary)
2. [Understanding the Selected Features](#2-understanding-the-selected-features)
3. [Comprehensive Metrics](#3-comprehensive-metrics)
4. [ROC Curves and Confusion Matrix](#4-roc-curves-and-confusion-matrix)
5. [Learning Curve and Train vs Test ROC](#5-learning-curve-and-train-vs-test-roc)
6. [Validation Comparison](#6-validation-comparison)
7. [Step-by-Step Diagnostic Plots](#7-stepbystep-diagnostic-plots)
8. [Sample Size Warnings](#8-sample-size-warnings)
9. [Warning Flags in Results](#9-warning-flags-in-results)
10. [Downstream Use of Selected Features](#10-downstream-use-of-selected-features)
11. [Common Failure Modes and Diagnosis](#11-common-failure-modes-and-diagnosis)

---

## 1. Understanding the Pipeline Summary

### What `pipeline_summary.csv` contains

`pipeline_summary.*` is a table built by `metrics.build_pipeline_summary_df()` and written by `pipeline._print_summary()`. It has one row per completed step, from Step 0 (Reference) through Step 3 (Wrappers), with these columns:

| Column | Meaning |
|--------|---------|
| `Step` | Step name: "Step 0 (Reference)", "Step 1 (Data Cleaning)", "Step 2 (Regularization)", "Step 3 (Wrappers)" |
| `Features` | Number of features retained after this step |
| `Features_Removed` | Features dropped relative to the previous step (Step 0 is 0) |
| `<ALGO>_CV_AUC` | Cross-validated macro OVR AUC at this step. The column name embeds the algorithm, so for XGB it is `XGB_CV_AUC`; for LR it is `LR_CV_AUC`, and so on. This is the consensus CV AUC for Steps 1 to 3 and the reference CV AUC for Step 0 |
| `Cumulative_Reduction` | Total features removed relative to Step 0 (`original_n_features - Features`) |
| `Reduction_Percentage` | `100 * Cumulative_Reduction / original_n_features` |

The real example `pipeline_summary.csv`:

```csv
Step,Features,Features_Removed,XGB_CV_AUC,Cumulative_Reduction,Reduction_Percentage
Step 0 (Reference),325,0,0.8572822869905817,0,0.0
Step 1 (Data Cleaning),193,132,0.8430239388291835,132,40.61538461538461
Step 2 (Regularization),61,132,0.857510658292657,264,81.23076923076923
Step 3 (Wrappers),47,14,0.8697470309013825,278,85.53846153846153
```

### The `pipeline_summary.png` figure

Produced by `visualization.plot_pipeline_summary()`. It has two panels:

- **Left panel**: a line chart on twin axes. The viridis line with circle markers is the feature count (left y-axis); the black line with square markers is `<ALGO>_CV_AUC` on the right y-axis (fixed 0 to 1.1). Read this as the trade-off curve: features fall step by step while the AUC line tracks whether that reduction cost anything.
- **Right panel**: a bar chart of the feature count per step (viridis bars, left axis) with the CV AUC overlaid as a black line with circle markers (right axis, 0 to 1.1). Same information, emphasising the magnitude of each reduction.

### How to read the reduction pattern

Read top to bottom. Step 0 is the baseline: how many features entered and what a single tuned model scored on all of them. Each later row shows how many survived and whether AUC moved.

In the example, features fell from 325 to 47 (an 85.5% reduction) while CV AUC actually rose from 0.857 to 0.870. This is the ideal pattern: uninformative and noisy features are being discarded while predictive signal is preserved or concentrated.

| Pattern | Diagnosis | Action |
|---------|-----------|--------|
| AUC flat or rising, large reduction | Healthy. Uninformative features removed | Proceed |
| AUC rises then drops sharply at one step | That step is too aggressive | Relax that step's threshold, or inspect its diagnostic plots (Section 7) |
| AUC falls continuously as features drop | Multiple steps too aggressive | Disable a step or relax thresholds |
| AUC flat but almost no reduction | Selection too conservative | Tighten thresholds or enable more steps |
| AUC high but nearly all features kept | Redundancy not addressed | Confirm Step 1 correlation filtering is enabled |

A single-step drop of more than roughly 0.05 in `<ALGO>_CV_AUC` is a caution signal that the step removed features the model needed. Section 7 (step diagnostic plots) and Section 6 (validation comparison) let you confirm this and, if needed, fall back to an earlier step's feature set.

---

## 2. Understanding the Selected Features

### What `selected_features.csv` contains

`selected_features.*` is a single-column table with header `feature`, written by `pipeline.save_results()`. It lists the feature names that survived the final completed step. Internally this is `get_selected_features()`, which returns `state_tracker.get_latest().X_train.columns.tolist()`, i.e. the columns of the training matrix at the most recently completed step.

The example `selected_features.csv` lists 47 miRNA features (`hsa.mir.320a_miRNA`, `hsa.mir.3934_miRNA`, and so on), matching the Step 3 count in the pipeline summary.

### What `recommended_features.csv` contains

When validation ran, `save_results()` also writes `recommended_features.*`: the panel of the step the recommendation chose (see "The recommendation" in Section 6). It is usually the same list as `selected_features.*`. It differs when a later step pruned past the point where it helped, and then it is the one to use: the final test metrics describe this panel, not the last step's.

### Retrieving features programmatically

```python
selected = pipeline.get_selected_features()          # final step feature list
step2_features = pipeline.get_selected_features(step='step2')  # features after Step 2
recommended = pipeline.get_recommended_features()    # the panel validation chose
```

`get_selected_features(step=...)` accepts `'step0'` through `'step3'` and returns the columns retained at that step. Passing `None` (the default) returns the latest step. Requesting a step that was not run raises `ValueError`.

### How many features to expect

| Outcome | Interpretation |
|---------|---------------|
| Very few (e.g. < 10) | Aggressive selection. Confirm AUC did not collapse in the pipeline summary. Can be correct for concentrated signal |
| Moderate (e.g. 20 to 100) | Typical for omics data. The example landed at 47 |
| Many (e.g. hundreds retained from thousands) | Conservative selection or genuinely diffuse signal. Consider tightening thresholds or enabling more steps |
| Same count as Step 0 | All steps disabled, or every step passed features through. Check the `skipped` flags (Section 9) |

---

## 3. Comprehensive Metrics

### What `final_<ALGO>_comprehensive_metrics.json` contains

Built by `metrics.build_comprehensive_metrics()` and written by `save_comprehensive_metrics()` as `final_<ALGO>_comprehensive_metrics.json` (the example file is `final_XGB_comprehensive_metrics.json`). It is a flat JSON object and is the authoritative record of held-out test-set performance.

Top-level keys (with example values):

| Key | Meaning | Example |
|-----|---------|---------|
| `algorithm` | Algorithm used | `"XGB"` |
| `n_consensus_models` | Consensus models configured | `5` |
| `achieved_agreement` | Fraction of models that had to agree for the delivered panel. An outcome, not a setting. | `0.6` |
| `agreement_label` | The same in words | `"moderate"` |
| `final_step` | Last step that set the feature list | `"Step 3 (Wrappers)"` |
| `final_features` | Feature count after selection | `47` |
| `original_features` | Feature count before selection | `325` |
| `feature_reduction_pct` | Percent of features removed | `85.54` |
| `training_cv_auc_mean` | Mean CV AUC of the final training step | `0.8573` |
| `training_cv_auc_std` | Std of that CV AUC | `0.0377` |
| `test_auc_macro` | Macro OVR AUC on the test set | `0.8627` |
| `generalization_gap` | `training_cv_auc_mean - test_auc_macro` | `-0.0054` |
| `overall_accuracy` | Accuracy on test set | `0.5738` |
| `balanced_accuracy` | Accuracy averaged per class | `0.4938` |
| `macro_precision` / `macro_recall` / `macro_f1` | Unweighted means across classes | `0.6745` / `0.4938` / `0.51` |
| `weighted_precision` / `weighted_recall` / `weighted_f1` | Class-size-weighted means | `0.6282` / `0.5738` / `0.5585` |

Any metric that is undefined (for example an AUC when a class is absent from the test split) is written as JSON `null` rather than `NaN`, so the file always parses.

Per-class keys follow the pattern `<class>_<metric>`. In the example, classes are integer-encoded (`0`, `1`, `2`, `3`, `4`), so keys look like `2_auc`, `3_sensitivity`, etc.

| Suffix | Meaning |
|--------|---------|
| `_precision`, `_recall`, `_f1` | Per-class (OvR) precision, recall, F1 |
| `_support` | Number of test samples in this class |
| `_auc` | Per-class OvR AUC on the test set |
| `_sensitivity`, `_specificity` | TPR and TNR at the default (argmax) decision |
| `_optimal_threshold` | Threshold maximising Youden's J (`TPR - FPR`) |
| `_optimal_sensitivity`, `_optimal_specificity` | Sensitivity and specificity at that optimal threshold |
| `_threshold_95spec`, `_sensitivity_95spec` | Threshold achieving specificity >= 95% (FPR <= 0.05) and the sensitivity attainable there |
| `_threshold_99spec`, `_sensitivity_99spec` | Same at specificity >= 99% (FPR <= 0.01) |

### Macro AUC versus per-class AUC

`test_auc_macro` is the macro-averaged one-vs-rest AUC (`roc_auc_score(..., multi_class='ovr', average='macro')`): each class becomes a binary problem and the AUC values are averaged without weighting by class size. Always read it alongside the per-class `<class>_auc` values, because a strong macro figure can mask a weak class. In the example, `test_auc_macro = 0.863`, but per-class AUCs range from `1_auc = 0.798` up to `3_auc = 0.961`.

### The threshold story matters as much as the AUC

AUC measures ranking, not the quality of the default decision boundary. Class 3 in the example illustrates this vividly:

```json
"3_auc": 0.9607,
"3_recall": 0.2,               (sensitivity at the default threshold)
"3_sensitivity_95spec": 0.8,
"3_optimal_threshold": 0.1459,
"3_optimal_sensitivity": 1.0
```

The model ranks class 3 almost perfectly (AUC 0.96), yet at the default threshold it catches only 20% of class-3 cases. Lowering the threshold to the Youden-optimal 0.146 recovers 100% sensitivity. When you see high AUC but low default sensitivity, the fix is threshold adjustment, not more data or a different model. Note also `3_support = 5`: with only 5 test samples, these per-class figures are inherently unstable (see Section 8).

The `_sensitivity_95spec` and `_sensitivity_99spec` values (from `metrics._threshold_at_specificity()`) report the best sensitivity achievable while holding specificity at or above 95% or 99%. These matter in screening and clinical settings where false positives are costly.

### Which summary metric to prioritise

| Metric | Prioritise when |
|--------|-----------------|
| Precision | False positives are costly (e.g. unnecessary treatment) |
| Recall / sensitivity | False negatives are costly (e.g. missed diagnosis) |
| F1 | You want one balanced summary |
| Balanced accuracy | Classes are imbalanced and you want equal weight per class |

The example shows why balanced accuracy (0.494) sits well below overall accuracy (0.574): the model does well on the larger, easier classes and poorly on the small, hard ones, and balanced accuracy exposes that.

### Confidence intervals

Bootstrap 95% confidence intervals on AUC are not stored in this JSON; they live in `feature_set_validation.*` and the `validation_comparison.*` figure (Section 6). CI width is the single most useful reliability indicator on small datasets.

---

## 4. ROC Curves and Confusion Matrix

### File: `roc_confusion.png`

Produced by `visualization.plot_per_class_roc_and_confusion()`. Two panels.

### Left panel: per-class ROC curves (OvR)

One curve per class, each a binary ROC where that class is the positive and all others are negative, scored by the class's predicted-probability column from `test_proba`. Curves are coloured on a viridis gradient.

- x-axis: False Positive Rate (`1 - specificity`).
- y-axis: True Positive Rate (sensitivity).
- The dashed diagonal is random (AUC 0.5).
- The legend shows each class name and its per-class AUC.

A curve hugging the upper-left discriminates that class well; a curve near the diagonal cannot separate it. The per-class AUCs here match the `<class>_auc` keys in the metrics JSON.

### Right panel: confusion matrix

A viridis heatmap with counts annotated in each cell. Rows are the true class, columns are the predicted class; the title reports the total test sample count. The diagonal is correct predictions; every off-diagonal cell is an error. Each row sums to that class's test support.

Reading it:

- **Good**: large diagonal, near-zero off-diagonal.
- **Systematic confusion**: a large off-diagonal cell relative to that row's diagonal means two classes are being conflated. Asymmetry (class A predicted as B more often than B as A) can reflect a real biological overlap the feature set does not resolve.
- **Collapse to a majority class**: an entire predicted column dominates while others stay near zero.
- **Class imbalance**: a small-support class (e.g. support 5) has a small row total, so a single error is a large percentage. Do not compare raw counts across classes with very different support; use the per-class recall (`<class>_sensitivity`) or a row-normalised view, and lean on `balanced_accuracy` as the summary.

---

## 5. Learning Curve and Train vs Test ROC

### File: `learning_curve_roc.png`

Produced by `visualization.plot_learning_curve_and_roc()`. Two panels.

### Left panel: learning curve

- x-axis: number of training examples, from 10% to 100% of the training set.
- y-axis: macro OVR ROC AUC.
- Training-score line (dark viridis) versus cross-validation-score line (mid viridis), each with a shaded +/- 1 standard-deviation band across folds.
- Fold count for the curve is `max(2, min(5, smallest_class_count))`.

Diagnosis:

| Pattern | Diagnosis | Action |
|---------|-----------|--------|
| Both curves low and flat | Underfitting | More expressive model, more features, or check for label noise |
| Training high, CV low, wide persistent gap | Overfitting | Fewer features, more regularisation, or more data |
| Both converge high | Good fit | None needed |
| CV curve still rising at the right edge | More data would help | Collect more samples if feasible |
| CV curve plateaued well below training | Overfitting unlikely to resolve with data | Reconsider model complexity or the selection strategy |

### Right panel: training (OOF) versus test ROC

- Training-CV curve (dark viridis): macro ROC from out-of-fold predictions on the training set. This is already cross-validated, so it is penalised for in-sample overfitting.
- Test curve (mid viridis): macro ROC on the held-out test set.
- A circle marker sits on the test curve at the 95% specificity operating point (FPR = 0.05).
- The legend reports each curve's AUC.

Close, overlapping curves indicate good generalisation. A large gap, even though the training curve is cross-validated, suggests the feature set or model was tuned too closely to the training distribution (repeated CV during selection can add a mild optimistic bias).

### The generalization gap label

The `generalization_gap` key equals `training_cv_auc_mean - test_auc_macro`. `metrics.summarise_generalization()` classifies its absolute size:

| Gap (absolute) | Label |
|----------------|-------|
| < 0.05 | excellent |
| 0.05 to 0.10 | good |
| 0.10 to 0.15 | moderate |
| > 0.15 | poor |

A small gap is good, and a small negative gap (the test set scoring slightly higher than training CV) is also good. The example gap is `-0.0054`, which is excellent: the test AUC (0.863) actually edged out the training CV AUC (0.857).

---

## 6. Validation Comparison

### Files: `feature_set_validation.csv` and `validation_comparison.png`

`feature_set_validation.*` is the DataFrame from `FeatureSetValidator.compare_feature_sets()`, one row per pipeline step's feature set. `validation_comparison.*` is the four-panel figure from `visualization.plot_validation_comparison()`. All validation runs on the **training split only**; the held-out test set is never exposed here.

### The three validation protocols

`FeatureSetValidator` scores each feature set with three independent protocols:

1. **Stratified k-fold CV** (`stratified_cv_validation`): k-fold stratified CV repeated with 3 model seeds (`random_seed`, `+100`, `+200`), scoring `roc_auc_ovr`. Folds are clamped to `min(cv_folds, max(2, min_class_size))`. This is the most reliable estimate and carries weight 0.5 in the recommendation.
2. **Leave-one-out or stratified shuffle-split** (`leave_one_out_validation`): full LOO when `n_samples <= 50`; otherwise 30 stratified shuffle-split iterations sized to guarantee the rarest class appears in each test split. Weight 0.25. Important: the scores mean different things in the two branches. With shuffle-split (`n > 50`) they are AUCs (`score_type = 'auc'`); with full LOO they are per-sample true-class probabilities (`score_type = 'confidence'`). When the branch cannot produce usable scores, the `loo_auc` column can be `0.0` (as it is throughout the example, where every `loo_auc` is `0.0`). Treat `loo_auc = 0` as "not informative for this run", not as a genuinely zero AUC.
3. **Bootstrap out-of-bag** (`bootstrap_validation`): `n_bootstrap` (default 100) stratified bootstrap resamples; each iteration scores on its out-of-bag samples. The 95% CI is the 2.5th and 97.5th percentiles of the successful iterations. Weight 0.25.

### The comparison columns

| Column | Meaning |
|--------|---------|
| `n_features` | Feature count for this step's set |
| `cv_auc` | Stratified CV mean AUC |
| `cv_std` | Stratified CV AUC std |
| `loo_auc` | LOO / shuffle-split mean score (may be 0 when uninformative) |
| `loo_std` | LOO / shuffle-split std |
| `bootstrap_auc` | Bootstrap mean OOB AUC |
| `bootstrap_std` | Bootstrap AUC std |
| `bootstrap_ci_lower` | 2.5th percentile of the bootstrap AUC distribution |
| `bootstrap_ci_upper` | 97.5th percentile |
| `bootstrap_ci_width` | `ci_upper - ci_lower` |
| `ci_width_flag` | CI-width adequacy: `adequate`, `caution`, or `critical` |
| `overall_adequacy` | The worse of `ci_width_flag` and the upfront sample-size severity from Section 8 |

Real example `feature_set_validation.csv` (one row per step, top row is the 325-feature reference set, bottom row the 61-feature Step 2 set):

```csv
n_features,cv_auc,cv_std,loo_auc,loo_std,bootstrap_auc,bootstrap_std,bootstrap_ci_lower,bootstrap_ci_upper,bootstrap_ci_width,ci_width_flag,overall_adequacy
325.0,0.8556,0.0501,0.0,0.0,0.8333,0.0329,0.7707,0.8981,0.1274,adequate,caution
61.0,0.8864,0.0542,0.0,0.0,0.8601,0.0281,0.8077,0.9173,0.1096,adequate,caution
```

Here every `ci_width_flag` is `adequate` (all widths below 0.20), while `overall_adequacy` is `caution`: the upfront sample-size assessment flagged caution (small n), and `overall_adequacy` reports the worse of the two signals. Note also that the 61-feature set has both a higher CV AUC (0.886 against 0.856) and a narrower bootstrap CI (0.110 against 0.127) than the full 325-feature reference: reduction improved stability here rather than costing it.

### CI-width adequacy thresholds

From `validation._ci_width_flag()`:

| CI width | Flag |
|----------|------|
| <= 0.20 | adequate |
| 0.20 to 0.30 | caution |
| > 0.30 | critical |

Wide CIs mean the AUC estimate is unstable across bootstrap samples, almost always because n is small. More data is the primary remedy.

### The recommendation

`validation.build_recommendation()` picks the best feature set by a weighted composite AUC:

```python
weighted_auc = 0.5 * cv_auc + 0.25 * loo_auc + 0.25 * bootstrap_auc
```

It does not simply take the highest `weighted_auc`. It applies the one-standard-error rule: every set whose composite lies within one standard error of the best is treated as tied, and the **smallest** of those wins. The standard error is the widest of the best set's CV standard error, bootstrap standard deviation and bootstrap CI half-width, floored at `MIN_SCORE_TOLERANCE = 0.005`. The floor matters at n << p, where CV AUC often saturates at 1.0 with zero spread: without it the band has zero width and the rule collapses back to plain argmax, which on synthetic benchmarks traded an 18-feature panel at ground-truth F1 0.714 for a 620-feature one at F1 0.032 over an AUC difference of 0.0003. When parsimony rather than the raw score decided, the `reason` string says so.

Quality is then labelled from that composite and CV stability (`cv_std / cv_auc`):

| Condition | Quality |
|-----------|---------|
| `weighted_auc > 0.8` and `cv_stability < 0.15` | RECOMMENDED |
| `weighted_auc > 0.7` (and not RECOMMENDED) | CAUTION |
| otherwise | NOT RECOMMENDED |

**The selection-optimism guard.** Every panel is validated on the same samples its features were selected from, so a selected panel's score is optimistic, and more so the harder it was selected. Step 0 carries no selection and is the one unbiased reference. When the recommended panel scores more than `MAX_PLAUSIBLE_SELECTION_GAIN = 0.05` above the least-selected panel, a RECOMMENDED label is lowered to CAUTION, `selection_optimism_suspected` is set, and the reason says so. It changes the label, never the choice. The threshold is measured: across four datasets with real signal (20 nested-CV folds) the gain was at most 0.021, while on a null control it was 0.23 to 0.37, with a held-out AUC of 0.46 for a panel that validated at 0.84. `selection_gain` reports the gap either way.

Because `loo_auc` is 0 in the example, the composite is driven by CV (weight 0.5) and bootstrap (weight 0.25); the 21-feature set wins on both.

**Where the recommendation is used.** `run(validate=True)` stores it as `results['recommendation']`, and `get_recommended_features()` returns its panel. `save_results()` writes it to `recommended_features.*` next to `selected_features.*`, which remains the last step's panel. The final test evaluation scores the recommended panel (its result records `panel` and `step_name`), and nested CV reports the recommended panel's held-out AUC as its headline. So you can run Steps 2 and 3 in series and still end up with Step 2's panel whenever Step 3 does not validate better.

**Ranking on held-out folds.** `enable_holdout_ranking=True` changes what the choice is made from. Instead of the validation above, each of `holdout_ranking_splits` folds carved from the **training** split reruns the selection steps on its own training portion and scores every step's panel on the rows held out from it, so no panel is scored on the samples that selected it. The ranking, the one-standard-error band and the quality label then all come from those scores, and `recommendation['ranking_basis']` says so. The selection-optimism guard switches off, because a gain measured on rows that selected neither panel is signal rather than optimism. The held-out **test** split is never involved, so the final test stays independent. What this ranks is the step rather than the exact panel: each fold selects its own features, and the panel delivered is still the one the full training split produced. It costs one extra run of the selection steps per fold.

Leave it off unless predictive AUC is what you are delivering. Measured on seven datasets, it agreed with the default ranking on four and changed the choice on three, and each change was for the worse: two panels an order of magnitude larger whose ground-truth F1 was a fifth to a third of the panel they replaced, and 592 noise features on a null control, all for AUC differences of 0.002 to 0.025. At n << p a panel carrying the true features plus a hundred noise ones predicts about as well as the true ones alone, so an AUC ranking turns on noise, and noise favours the larger panel. What it does improve is the quality label on data with no signal, and the selection-optimism guard above reaches a warning there from the training split alone. See `benchmarks/BENCHMARKS.md`, section 8.

**Checking the choice.** The recommendation is made on the training split, which is also the data every panel was selected on. `enable_nested_cv=True` tests whether the choice holds up on unseen data: each outer fold reruns selection, validation and the recommendation, then scores every step's panel on the held-out fold. `results['nested_cv']['mean_regret']` is how far the recommended panel fell short of the step that did best on each fold with hindsight, and `fold_details` shows every fold and step.

### Reading the `validation_comparison.png` figure

Four panels:

- **Panel 1 (top-left)**: grouped bars of CV, LOO, and Bootstrap AUC per feature set, with error bars. The set with consistently tall bars across all three methods is the most robust.
- **Panel 2 (top-right)**: scatter of feature count versus CV AUC, each point annotated with its step name. Upper-left (high AUC, few features) is the efficiency sweet spot.
- **Panel 3 (bottom-left)**: bootstrap mean AUC bars with 95% CI error bars. Narrow bars mean stable estimates.
- **Panel 4 (bottom-right)**: CV coefficient of variation (`cv_std / cv_auc`) per set. Lower is more stable; above about 0.15 is less reliable.

### When an earlier step wins

If Step 3 has the highest weighted AUC, the reduction found the most informative subset (the expected outcome, and what happens in the example). If an earlier step (Step 1 or 2) validates better, later steps may have removed useful features; use that step's set via `get_selected_features(step='step2')`, or relax the later thresholds. If Step 0 (all features) beats every selected set, the signal is diffuse and selection is counterproductive for this data and algorithm.

---

## 7. Step-by-Step Diagnostic Plots

Each selection step writes two diagnostic figures comparing that step's consensus model(s) against the Step 0 reference. Step 0 writes its own reference plots. Filenames:

- `step0_reference_boxplots.*` and `step0_reference_roc.*`
- `step<N>_consensus_vs_reference_boxplots.*` and `step<N>_consensus_vs_reference_roc.*` for N = 1 to 3

### The boxplot figure (`visualization.plot_auc_boxplots`)

Box plots of per-fold AUC distributions, comparing the reference model (all Step 0 features) against the consensus model(s) on the reduced set. Each box shows the median (viridis line), the interquartile range (white box, black edge), whiskers at 1.5x IQR, outlier fliers, and the mean (triangle marker, `showmeans=True`).

- Overlapping boxes with similar medians: the reduced set performs as well as the reference. Reduction was safe.
- Consensus box clearly above reference: the reduced set is more effective (a good outcome).
- Consensus box clearly below with no overlap: this step dropped features the model needed. Relax its threshold.
- Very wide consensus IQR: unstable across folds, often from too few features or too little data.

### The ROC figure (`visualization.plot_roc_curves`)

Macro OVR ROC curves (from out-of-fold predictions) for the reference and consensus models overlaid, each with a marker at the 95% specificity point (FPR = 0.05) and its AUC in the legend. If the consensus curve tracks the reference closely, the step's selection was sound; if it falls below, especially near the upper-left (high-specificity) region, the step removed features critical for confident positive prediction.

### The `step<N>_summary.txt` files

Each step writes a plain `key: value` summary. The Step 3 summary carries the most diagnostic detail:

```
Step: step3
algorithm: XGB
n_features_in: 61
n_features_out: 47
votes_required: 3
n_models: 5
agreement: 0.6
agreement_label: moderate
consensus_outcome: relaxed
consensus_mean_auc: 0.8697470309013825
consensus_std_auc: 0.028445416066101525
```

`votes_required` is the strongest agreement level at which the panel still met `min_features_floor`; `consensus_outcome` records how it got there. Every step reports the reference AUC and per-consensus-model AUCs alongside these (`ref_xgb_mean_auc`, `clean_xgb_1_auc`, `wrappers_xgb_1_auc`, and so on).

### Diagnosing divergence

If a step's consensus AUC diverges from the reference by more than roughly 0.05:

1. Check `Features_Removed` for that step in `pipeline_summary.*`; large removals are the usual cause.
2. Check the sample-size diagnostics (Section 8); small n makes the consensus sensitive to which features survive.
3. Raise `min_features_floor`. This, not any agreement setting, is what determines panel size: each step relaxes its vote requirement until the floor is met. Check `agreement_label` afterwards to see what the larger panel cost you in agreement.

---

## 8. Sample Size Warnings

### When and how they are generated

`diagnostics.assess_sample_size_adequacy()` runs once on the training set before Step 0. The result is stored on `pipeline.sample_adequacy` and in `results['sample_adequacy']`. Warnings are always logged at WARNING level when severity is non-adequate, regardless of `verbose`. These signals are advisory: they never halt the pipeline.

The returned dict includes `n_samples`, `n_features`, `n_classes`, `samples_per_class`, `min_class_size`, `n_per_p`, `n_folds`, the four per-signal flags (`absolute_n_flag`, `n_per_p_flag`, `per_class_flag`, `cv_reliability_flag`), `overall_severity` (the worst of the four), and lists `warnings` and `recommendations`.

### Signal 1: absolute sample count (n)

| n | Flag |
|---|------|
| < 30 | critical |
| 30 to 99 | caution |
| >= 100 | adequate |

Basis: Varoquaux (2018), doi:10.1016/j.neuroimage.2017.06.061. At critical n, treat all AUCs as placeholders and do not draw biological conclusions without external validation. At caution n, the bootstrap CI width (Section 6) is your best reliability indicator.

### Signal 2: samples-to-features ratio (n/p)

Computed as `n_samples / n_features` on the current training matrix.

| n/p | Flag |
|-----|------|
| < 0.1 | critical |
| 0.1 to 1.0 | caution |
| >= 1.0 | adequate |

Basis: Varoquaux (2018). Critical n/p means severe underdetermination: results are highly sensitive to sampling; independent-cohort validation is mandatory, and heavier pre-filtering before Step 1 helps. Because the assessment is a function of feature count, the n/p picture improves as later steps shrink p, even though the formal one-time flag is set on the initial matrix.

### Signal 3: samples per class

| Smallest class count | Flag |
|----------------------|------|
| any class < 10 | critical |
| any class 10 to 19 | caution |
| all classes >= 20 | adequate |

Basis: Figueroa et al. (2012), doi:10.1186/1472-6947-12-8. Critical per-class counts make per-class metrics and confusion-matrix rows uninterpretable for the affected classes (recall the example's class 3 with support 5). Consider merging rare classes, removing them, or careful oversampling.

### Signal 4: CV fold reliability

With `min_class_size` the smallest class count and `n_folds` the configured folds:

| Condition | Flag |
|-----------|------|
| `min_class_size < n_folds` | critical |
| `min_class_size < 2 * n_folds` | caution |
| `min_class_size >= 2 * n_folds` | adequate |

Basis: Kohavi (1995), doi:10.5555/1643031.1643047. The pipeline auto-reduces the fold count when a class is too small, so a critical flag means the CV actually running has fewer folds than configured and is higher-variance.

### Proceed or collect more data

Proceed despite warnings when the work is exploratory (a hypothesis list, not a validated biomarker), when n/p is critical but n exceeds 100, or when the bootstrap CI width is below 0.20 despite small n.

Collect more data when n < 30 and the goal is a validated classifier, when any class has fewer than 10 samples, when the bootstrap CI width exceeds 0.30, or when the learning curve (Section 5) is still rising at maximum training size.

---

## 9. Warning Flags in Results

### Where warnings surface

After `pipeline.run()`:

1. **`results['sample_adequacy']`**: the full adequacy dict, including `overall_severity` and the `warnings` list (Section 8).
2. **`results['step<N>']['skipped']`**: a boolean set when a step was skipped, either because it was disabled or because too few features reached it.
3. **`results['step<N>']['consensus_outcome']`**: how the panel was reached (`consensus`, `relaxed`, `consensus_limited`, or `rank_average`). Anything other than the first two deserves a look.
4. **Log output**: critical adequacy warnings are logged at WARNING level regardless of `verbose`.

### Checking warnings systematically

```python
results = pipeline.run()

adequacy = results.get('sample_adequacy', {})
print("Overall severity:", adequacy.get('overall_severity'))
for w in adequacy.get('warnings', []):
    print(" -", w)

for key in ('step1', 'step2', 'step3'):
    step = results.get(key, {})
    if step.get('skipped'):
        print(f"{key} was skipped: features passed through unchanged")
        continue
    print(f"{key}: {step.get('n_features_out')} features, "
          f"agreement {step.get('agreement_label')} "
          f"({step.get('consensus_outcome')})")
```

### What each flag means

- **`skipped = True`**: Step 3 is skipped (pass-through) when disabled or when fewer than `MIN_FEATURES_FOR_RFECV = 30` features remain; Steps 1 and 2 skip only when disabled. A skipped step passes its input through unchanged, so its row in `pipeline_summary.*` repeats the previous feature count.
- **`consensus_outcome = 'consensus_limited'`**: the descent from unanimity was stopped by `min_consensus` before `min_features_floor` was reached. The panel is smaller than you asked for but better supported than a relaxed one would be. Lower `min_consensus` if you want the larger panel.
- **`consensus_outcome = 'rank_average'`**: no feature was selected by any replicate in every stage, so there was no consensus evidence to aggregate and the step fell back to rank-averaging within the voted pool. Treat that panel as provisional and check the step's diagnostic plots.

---

## 10. Downstream Use of Selected Features

### Extracting the subset

```python
# From a completed pipeline object: the panel validation recommends
selected = pipeline.get_recommended_features()
X_train_sel = X_train_original[selected]
X_test_sel  = X_test_original[selected]

# From the saved file, in a fresh session
import pandas as pd
selected = pd.read_csv("output/recommended_features.csv")["feature"].tolist()
X_new = new_dataset[selected]

# The last step's panel, or the panel at a specific step
last_step = pipeline.get_selected_features()
step2_features = pipeline.get_selected_features(step='step2')
```

Prefer the recommended panel. It is the one the held-out metrics describe, and it
differs from the last step's exactly when a later step removed features the model
needed. `get_recommended_features()` needs `run(validate=True)`; without validation
there is no recommendation and `get_selected_features()` is the only choice.

### Checking how strongly the panel is supported

```python
step3 = pipeline.results['step3']
print(step3['votes_required'], "of", step3['n_models'], "models")
print(step3['agreement_label'], step3['consensus_outcome'])
```

The panel is the set of features that cleared `votes_required` in every stage of the
last step that ran. There is no per-feature importance score to rank within it: the
pipeline delivers a set, not an ordering. If you want an ordering, run SHAP or
permutation importance yourself on a model fitted to the selected panel.

### Feature provenance

`pipeline.get_feature_provenance()` (also written as `feature_provenance.*`) shows which steps each feature survived:

| Column | Meaning |
|--------|---------|
| `feature` | Feature name |
| `survived_step1` ... `survived_step3` | Boolean per step (only for steps that ran) |
| `n_steps_survived` | Count of steps survived |
| `first_dropped_at` | Step key where it was first dropped, blank if it survived to the end |

The table is sorted by `n_steps_survived` descending, then feature name. Example header and top rows:

```csv
feature,survived_step1,survived_step2,survived_step3,n_steps_survived,first_dropped_at
hsa.let.7f.1_miRNA,True,True,True,3,
hsa.mir.113b_miRNA,True,True,True,3,
```

Features with the maximum `n_steps_survived` and a blank `first_dropped_at` passed every filter and are the highest-confidence candidates for follow-up.

### Why consensus selection suits biological interpretation

- **Fewer spurious hits**: a feature important in one model but not others is likely noise; consensus removes most of them.
- **Correlated groups**: pathway co-members can each earn partial credit from the consensus rather than one being kept arbitrarily and the rest discarded, making the list more pathway-coherent.
- **Provenance-aware pathway analysis**: check whether pathway members were dropped at Step 1 (correlation filter) versus retained through Step 3; the pathway may still be represented by one member.
- **Report the agreement level with the panel**: a panel that held at unanimous or strong agreement is a stronger claim than one that needed relaxing to weak, even at the same size.

### Manuscript reporting checklist

- [ ] Dataset: n samples, p initial features, class count, train/test split and stratification.
- [ ] Algorithm and reason for choice.
- [ ] `n_consensus_models`, `min_features_floor`, and any `min_consensus`.
- [ ] The **achieved** agreement for the final panel (`agreement_label` and `votes_required` / `n_models`). This is the strongest level at which the panel holds and belongs next to the feature list in any write-up.
- [ ] Steps enabled and any non-default thresholds.
- [ ] Step 0 reference AUC (mean +/- std).
- [ ] Feature count and CV AUC per step (the `pipeline_summary.*` table).
- [ ] Final feature count and test macro OVR AUC.
- [ ] Bootstrap 95% CI on the validation AUC and its width flag.
- [ ] `generalization_gap` and its label (excellent / good / moderate / poor).
- [ ] Sample-size adequacy flags and `overall_severity`.
- [ ] `consensus_outcome` for the final step.
- [ ] External-validation status.
- [ ] Random seed, and SelectOmics / Python / scikit-learn versions.

---

## 11. Common Failure Modes and Diagnosis

### Too few features selected (or an empty list)

**Symptom**: far fewer features than expected, or `selected_features.*` is nearly empty.

**Causes and fixes**:
- `min_features_floor` too low. This is the main control on panel size, since every step relaxes its vote requirement until the floor is met. Raising it is the direct fix; `agreement_label` then tells you what the larger panel cost.
- `min_consensus` set too high. If `consensus_outcome` reads `consensus_limited`, the descent was stopped by your agreement floor before the feature floor was reached: the pipeline is telling you those two requirements conflict on this data. Lower `min_consensus`, or accept the smaller, better-supported panel.
- The floor was capped. If the log reports the minimum feature count being capped, only that many features were selected by any model in every stage; a larger panel would have to be padded with features nothing chose.
- Step 1 correlation threshold too tight, removing most of a correlated omics block. Relax `correlation_threshold`.

### AUC collapses at Step 3

**Symptom**: `<ALGO>_CV_AUC` drops sharply (> 0.05 to 0.10) at Step 3 in `pipeline_summary.*`.

**Causes**: too few features entering the step, or RFECV eliminating important features when the estimator or CV is noisy.

**Fixes**: raise `min_features_floor` so the step stops relaxing sooner; disable Step 3 (`enable_step3=False`) and rely on Steps 1 + 2; or inspect `feature_set_validation.*` and adopt an earlier step's set via `get_selected_features(step=...)`. Confirm the divergence in the step's `consensus_vs_reference` plots (Section 7).

### A step was skipped

**Symptom**: a step's `skipped` flag is `True` and its feature count equals the previous step.

**Cause**: Step 3 skips below 30 features (`MIN_FEATURES_FOR_RFECV`), and any step skips when its enable flag is off. This is expected behaviour, not an error: Steps 1 and 2 have already filtered the panel to near-purity, so recursive elimination would remove true features rather than noise. Measured, a panel of 10 that Step 2 had selected at F1 1.000 was cut by Step 3 to a single feature.

### Algorithm silently switched to XGB

**Symptom**: `algorithm` in the outputs is `XGB` when you configured LR, RF, or SVM.

**Cause**: `handle_nans()` in `load_data()` switches to XGB when the data contains NaNs, because XGB handles missing values natively. Impute the NaNs beforehand if you need a different algorithm.

### Wide confidence intervals

**Symptom**: `bootstrap_ci_width > 0.20` (caution) or `> 0.30` (critical) in `feature_set_validation.*`.

**Cause**: small n (the usual reason), severe class imbalance (minority class sometimes absent from the OOB set), or high n/p. **Action**: collect more data; consider reducing to binary classification if a minority class is tiny; report the CI width explicitly; or use `pipeline.run_nested_cv()` for a more conservative generalisation estimate.

### GPU not used

**Symptom**: XGBoost runs on CPU despite `use_gpu=True`.

**Cause**: no CUDA device detected (`detect_gpu()` returns False) or an XGBoost build without CUDA support. The pipeline falls back to CPU and logs the decision.

### Resume did nothing

**Symptom**: `run(resume=True)` re-runs everything from scratch.

**Cause**: the previous run did not write a checkpoint. Checkpointing requires `save_intermediate_results=True`; without it, `.selectomics_checkpoint.pkl` is never created. A corrupt or stale checkpoint is logged and ignored, and the run starts fresh.

### Selected features differ between runs with the same seed

**Symptom**: identical config and `random_seed`, different `selected_features.*`.

**Likely causes**: non-deterministic multi-threaded XGBoost (set `n_jobs=1`); floating-point ordering effects that flip features sitting exactly on the percentile threshold; a changed input row order (which changes stratified splits even at a fixed seed); or a stale resume checkpoint (delete `.selectomics_checkpoint.pkl`). Compare the Step 2 stage percentiles across runs: if they differ, a binary search diverged; if they match but the feature list differs, the cause is boundary features sitting near the threshold.

---

*For configuration parameters see the `SelectOmicsConfig` class; for per-step mechanics see the module docstrings in `selection/step<N>_*.py`.*
