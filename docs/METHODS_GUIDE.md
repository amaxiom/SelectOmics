# SelectOmics Methods Guide

This document provides a technically precise description of every algorithmic component in the SelectOmics feature selection pipeline. All claims are grounded in the package source code; function and parameter references correspond directly to the implementations in `SelectOmics/selection/`, `SelectOmics/models/`, `SelectOmics/evaluation/`, and `SelectOmics/utils/`. Where a numeric constant or search space is stated, it is the value found in the current source.

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Step 0: Reference Evaluation](#step-0-reference-evaluation)
3. [Step 1: Data Cleaning](#step-1-data-cleaning)
4. [Step 2: Regularisation](#step-2-regularisation)
5. [Step 3: Wrappers](#step-3-wrappers)
6. [Consensus Mechanics (Cross-Cutting)](#consensus-mechanics-cross-cutting)
7. [Model Details](#model-details)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Sample Size Diagnostics](#sample-size-diagnostics)

---

## Pipeline Overview

SelectOmics applies three sequential feature selection steps, each narrowing the feature set passed to the next, preceded by a reference-evaluation step (Step 0) that establishes a performance baseline. The pipeline supports four classification algorithms, selected via `config.algorithm`: logistic regression (`'LR'`), gradient-boosted trees (`'XGB'`), random forest (`'RF'`), and support vector machine (`'SVM'`).

```
X_train (full)
    |
    v
Step 0: Reference evaluation (single tuned model, no selection; establishes AUC baseline)
    |
    v
Step 1: Data cleaning (constant removal, joint variance and correlation filter)
    |
    v
Step 2: Regularisation (parallel L1 and L2 importance passes, intersection)
    |
    v
Step 3: Wrappers (RFECV consensus AND stability selection, intersection)
    |
    v
Final feature set
```

### Shared design invariants

- **One shared cross-validator.** A single `StratifiedKFold` object is constructed once during data preparation and reused across every step. Its fold count follows `cv_splits = min(max_cv_splits, max(min_cv_splits, min_class_count))` when `cv_splits` is left unset (defaults: `min_cv_splits=3`, `max_cv_splits=20`). Reusing the identical splitter guarantees comparable fold assignments across all evaluations.
- **Single tuning pass.** The active algorithm is tuned once by `quick_tune_all` (via `RandomizedSearchCV`) before Step 0. The three non-selected algorithms receive fixed-default pipelines so they can serve only as reference comparators; they are never tuned.
- **Baseline caching.** The Step 0 reference result is computed once and passed explicitly to every later step as `ref_result_step0`. It is never recomputed mid-pipeline.
- **Step toggles and pass-through.** Each of Steps 1 to 3 can be disabled (`enable_step1` through `enable_step3`). A disabled step returns `X_train`/`X_test` unchanged, empty drop lists, and copies the reference result forward as its `consensus_result`. Step 3 additionally self-skips when the incoming feature count is too low (see that step).
- **Leakage-safe pipelines.** Every model pipeline is `MinMaxScaler` followed by a classifier. The scaler is fit inside each training fold, so no test-fold information leaks into scaling.
- **GPU inheritance.** When `use_gpu` is set and `detect_gpu()` confirms a CUDA-capable device, the tuned XGB pipeline carries `device='cuda'`. Steps 2 and 3 read this device off the tuned XGB pipeline and propagate it to their internal XGB estimators.

---

## Step 0: Reference Evaluation

**Source:** `selection/step0_reference.py`

### Purpose

Step 0 establishes the classification AUC baseline on the full, unfiltered training set. This baseline is the value against which Steps 1 to 3 compare their consensus performance.

### Method

1. The active algorithm's tuned pipeline is retrieved from `tuned_pipelines[config.algorithm]` and registered under the name `Ref-<algorithm>` (for example `Ref-XGB`). A `ValueError` is raised if the algorithm is absent from `tuned_pipelines`.
2. **A single model is always used** (`n_models = 1`) regardless of `config.n_consensus_models`, so the baseline reflects one model's cross-validated performance rather than an ensemble average.
3. Evaluation runs through `evaluate_models_collection` with **`use_bootstrap=False`**, so folds are not resampled and the reference reflects the unmodified training distribution.
4. The per-fold metric is macro one-vs-rest AUC (`compute_macro_auc_ovr`); an out-of-fold (OOF) probability matrix is assembled across folds. The returned `reference_result` dict carries `mean_auc`, `std_auc`, `fold_aucs`, `oof_proba`, and `oof_pred`.

If an `adequacy` assessment is supplied, sample-size warnings are printed before evaluation. Diagnostic box plots and ROC curves are produced only when `config.create_visualizations` is set; a plain-text summary is saved only when `config.save_intermediate_results` is set.

### Return

`{reference_result, reference_evaluation, algorithm, n_features, model_name}`. `reference_result` is the cached baseline passed downstream.

---

## Step 1: Data Cleaning

**Source:** `selection/step1_cleaning.py`

Step 1 removes constant features, then jointly optimises a variance threshold and a correlation threshold by maximising the agreement between the two filters on which features to drop. When `enable_step1` is False the data passes through unchanged.

### Sub-step 1a: Constant removal

Any feature with `nunique() <= 1` is dropped. This runs before the grid search so the correlation matrix is computed on a non-degenerate matrix.

### Sub-steps 1b/1c: Joint variance and correlation optimisation

A full **11 x 11 = 121-point grid** is evaluated:

- Variance grid: `np.linspace(0.0, max_variance_threshold, 11)` (default `max_variance_threshold = 0.30`).
- Correlation grid: `np.linspace(min_correlation_threshold, 1.0, 11)` (default `min_correlation_threshold = 0.70`; a threshold of 1.0 keeps all features).

For each grid point:

- **Variance filter** (`_apply_variance_filter`): wraps `sklearn.feature_selection.VarianceThreshold`; features with variance `<= threshold` are dropped. If no feature meets the threshold, the point is treated as all-dropped.
- **Correlation filter** (`_apply_correlation_filter`): greedy elimination in **descending priority order**. The absolute correlation matrix is computed once (outside the loop) for efficiency. For any pair correlating above `threshold`, the lower-priority member is dropped; processing high-priority features first preferentially retains them.

  Which matrix and which priority depends on `class_aware_correlation` (default `True`):

  | | matrix | priority | tie-breaks |
  |---|---|---|---|
  | `True` | pooled within-class (`pooled_within_class_corr`) | correlation ratio eta-squared (`class_association`) | variance, then name |
  | `False` | total absolute Pearson | variance | name |

  **Why class-aware is the default.** Features carrying the same class signal are mutually correlated *because* they are informative. Total correlation cannot distinguish that from redundant duplication, so it keeps one carrier and discards the rest. Centring each feature on its own class mean removes the between-class covariance, leaving only association the label does not explain: genuine duplicates stay correlated after centring, co-regulated markers do not. On the synthetic case built to expose this, retention of 20 informative features went from 1/20 to 20/20.

  One consequence worth stating: a feature with no within-class variation, one perfectly determined by the label, centres to all zeros and so is never judged redundant. That is the intended trade-off rather than an oversight, since such a feature is maximally informative. Two identical class-determined probes therefore both survive Step 1.

  Deterministic either way, and independent of column order.

The objective is the **Jaccard similarity of the two drop sets**: `|A and B| / |A or B|`, where A and B are the variance and correlation drop sets. Normalising by the union prevents aggressive thresholds (which produce large drop sets) from being rewarded for size alone; the score measures genuine agreement. The grid point with the highest Jaccard score is retained, along with both kept masks.

### Final selection and fallback hierarchy

The kept set is the **intersection** of the two best kept masks (strictest consensus). If this falls below `min_features_floor`:

1. Fall back to the **union** of the two kept masks (1/2 filter agreement).
2. If still below the floor, force the **top-k features by variance** (k = `min_features_floor`).

Step 1 has only two voters, the variance filter and the correlation filter, and each casts a single binary vote. Unlike Steps 2 and 3 there are therefore no intermediate rungs to relax through: intersection and union are the entire ladder. It reports its outcome in the same vocabulary regardless, as `votes_required` out of `n_models = 2` with an `agreement_label`, so a panel decided here carries the same record as one decided later.

Features dropped by each filter are reported for diagnostics; `drop_consensus` is the set-intersection of the two drop lists.

### Consensus evaluation

If `enable_step_evaluations` is set, `n_models` consensus pipelines are built with seeds `base_seed + i * 100` and evaluated on the cleaned matrix with **`use_bootstrap=True`**. The best model by `mean_auc` becomes `consensus_result`, which is printed against the Step 0 reference.

---

## Step 2: Regularisation

**Source:** `selection/step2_regularization.py`

Step 2 runs two parallel regularisation stages, an L1-style (sparsity-inducing) pass and an L2-style (smoothing) pass, and intersects their consensus selections. When `enable_step2` is False the data passes through unchanged.

### Stage models (per algorithm)

Each stage builds a `MinMaxScaler + classifier` pipeline (`_build_stage_model`):

| Algorithm | L1-style stage | L2-style stage |
|-----------|----------------|----------------|
| LR  | `LogisticRegression(penalty='l1', solver='saga', C=0.1, max_iter=1000)` | `LogisticRegression(penalty='l2', C=0.1, max_iter=1000)` |
| XGB | `XGBClassifier(reg_alpha=10.0, reg_lambda=0.1)` (alpha-dominant) | `XGBClassifier(reg_alpha=0.1, reg_lambda=10.0)` (lambda-dominant) |
| RF  | `RandomForestClassifier(n_estimators=50, max_features='sqrt', min_samples_leaf=3)` | `RandomForestClassifier(n_estimators=100, max_features='log2', min_samples_leaf=2)` |
| SVM | `LinearSVC(penalty='l1', dual='auto', C=0.1, max_iter=2000)` | `LinearSVC(penalty='l2', dual='auto', C=1.0, max_iter=2000)` |

The SVM stage uses a `LinearSVC` surrogate for `coef_`-based importance (the RBF SVC lacks usable `coef_`); the tuned RBF SVC is still used for all evaluation.

### Importance extraction (`_extract_importance`)

- **LR, SVM:** `np.abs(clf.coef_).max(axis=0)` (max absolute coefficient across classes).
- **XGB, RF:** `clf.feature_importances_`.

Importances are normalised to `[0, 1]` by dividing by their maximum (when positive).

### Threshold selection (binary search)

Each stage's percentile is found by an independent binary search (`_binary_search_percentile`) rather than a grid, over the full `[0, 1]` domain with `tolerance=0.01` and `max_iter=10`, so at most 12 evaluations per stage. The search targets `step2_target_retention` (default 1.0, a fraction of the **reachable pool** rather than of the input width), so `evaluate_fn` returns the **kept count** (decreasing in the percentile) via `_estimate_drop_mask`, which fits one stage model and applies `_rank_keep_mask`. The pool is the set of features the stage models actually used, and its size is governed by `step2_stage_colsample`; lowering that widens it.

Selection is by importance **rank**, capped at the number of features the model gave a non-zero importance to. Comparing values against a percentile of the importance distribution does not work here: tree importance is exactly zero for every feature never used in a split, and that tie mass is 98 to 99.8% of the array on this package's target data, so `np.percentile` returns 0.0 across the whole search range and the kept count is identical at every point. The rank cap is what keeps this honest at the other end: a feature the model never split on carries no evidence for keeping it, so the selection never extends into the tied-zero region even when the retention target asks for more. When the cap binds, the step logs that the percentile is saturated and `_binary_search_percentile` raises a `UserWarning` naming the target and the closest achievable value.

### Voting pass (`_run_voting_pass`)

At the best percentile, a full `n_models` voting pass is run per stage. Seeds are:

- **L1 stage:** `base_seed + i * 200`
- **L2 stage:** `base_seed + i * 200 + 100` (the +100 offset keeps stages on distinct seed sequences)

Each model fits, extracts importance, and votes for features whose importance exceeds the per-model percentile cut. A feature is selected in a stage when its vote count `>= required_votes` (see [Consensus Mechanics](#consensus-mechanics-cross-cutting)).

### Final selection and fallback hierarchy

Final kept set = **L1 mask AND L2 mask**, taken at the strongest vote level that yields `min_features_floor` features. The requirement starts at unanimity and drops one vote at a time; see Consensus Mechanics. There is no union rung by default: with `n_models` replicates per stage there are `n_models` levels between unanimity and the loosest setting, and a direct fall to union would discard all of them. `allow_union_rung=True` adds one beneath the ladder, trading precision for recall; section 7.3 of `BENCHMARKS.md` carries the cost.

Rank-averaging over the stage models' mean importance is the terminal fallback, reachable only when no feature was selected by any replicate in both stages.

### Consensus evaluation

Identical structure to Step 1: `n_models` consensus pipelines with seeds `base_seed + i * 100`, evaluated with bootstrap, best-by-`mean_auc` becomes `consensus_result`.

---

## Step 3: Wrappers

**Source:** `selection/step3_wrapper.py`

Step 3 runs two independent wrapper-based stages, RFECV consensus and stability selection, and intersects their kept sets. It is skipped (pass-through) when `enable_step3` is False **or when the incoming feature count is below `MIN_FEATURES_FOR_RFECV = 30`**.

The gate stands in for a quantity that cannot be observed at runtime: how much noise the incoming panel still holds. Recursive elimination is only useful on a panel that still contains features worth eliminating, and Step 2's output does not -- its precision is 0.970 across the benchmark. Lowering the gate to 10 was tried and reverted: on three measured runs Step 2 delivered a panel of 10 features at F1 1.000 and Step 3 cut it to one or two, dropping F1 to 0.182. A wide incoming panel is the available proxy for an unfiltered one. The step remains valuable standalone, and after a deliberately permissive Step 2.

### Stage 1: RFECV consensus (`_run_rfecv_stage`)

**Estimators** (`_build_rfecv_estimator`, all wrapped in `MinMaxScaler + clf`):

- **LR:** `LogisticRegression(solver='lbfgs', max_iter=1000)`; importance getter = max absolute `coef_` across classes.
- **XGB:** `XGBClassifier` defaults; importance getter = `feature_importances_`.
- **RF:** `RandomForestClassifier(n_estimators=50)`; importance getter = `feature_importances_`.
- **SVM:** `LinearSVC(penalty='l1', dual='auto', C=0.1, max_iter=2000)` surrogate; importance getter = max absolute `coef_` across classes.

**RFECV configuration:** `step=1`, `scoring='roc_auc_ovr'`, internal CV of `min(3, cv.n_splits)` folds (a fresh `StratifiedKFold` seeded per model), and `min_features_to_select = max(5, int(n_features * percentile))`.

**Percentile search:** binary search over the full `[0, 1]` percentile domain, targeting `step3_target_retention` (default 0.6). Because RFECV is the most expensive per-evaluation operation, `evaluate_fn` returns the **removed count** (decreasing in percentile). Probe failures increment `warnings['rfecv_failed']` and fall back to an approximate removed count.

**Voting pass:** at the best percentile, `n_models` RFECV instances are fit with seeds `base_seed + i * 400`. Each contributes votes via `rfecv.support_`. For importance accumulation, the refitted `rfecv.estimator_`'s classifier is **cloned** (to avoid aliasing) into a fresh pipeline, refit on the full feature space, and its normalised importance recorded. On a fit failure, all features are treated as selected (vote) and a uniform importance vector is recorded. A feature is RFECV-selected when its vote count `>= required_votes`.

### Stage 2: Stability selection (`_run_stability_stage`)

Follows Meinshausen and Buhlmann (2010). `STABILITY_N_SUBSAMPLES = 50` (the lower bound recommended by Shah and Samworth, 2013).

- **Subsampling:** each subsample is a 50% stratified draw **without replacement** via `StratifiedShuffleSplit(n_splits=1, train_size=subsample_size)`, where `subsample_size = min(max(int(0.5 * n_samples), min_features + 1), n_samples - 1)`. Subsample seed = `base_seed + i * 500`.
- **Class guard:** subsamples missing any class are skipped (`warnings['stability_skipped_class']`).
- **Model seed:** `base_seed + (i % n_models) * 400 + 500`. The +500 offset separates stability models from RFECV models; cycling `(i % n_models)` limits model variety to `n_models` distinct seeds while distributing them across the 50 subsamples. The same algorithm-specific estimators as Stage 1 are used.
- **Selection rule:** on each valid subsample, a feature is "selected" if its normalised importance exceeds the **subsample mean** (scale-invariant, no fixed cutoff). `stability_freq = selection_count / n_valid_subsamples`. A feature passes when `stability_freq >= stability_threshold` (declared default `None`, resolving to 0.6).
- **All-fail fallback:** if every subsample fails, `warnings['stability_all_failed']` is set and the stage retains all features (stability mask all-True), so the intersection degenerates to the RFECV result alone.

### Final selection and fallback hierarchy

Final kept set = **RFECV mask AND stability mask**, taken at the strongest vote level that yields `min_features_floor` features, by the same graduated relaxation used in Step 2. Stability frequencies are first mapped onto the vote scale (`floor(freq * n_models)` for features clearing `stability_threshold`, zero otherwise) so the stability stage relaxes in step with the model votes rather than acting as a fixed veto.

Rank-averaging over mean importance is the terminal fallback.

Per-feature diagnostics (RFECV votes and fractions, stability frequencies, per-stage and final masks) are saved when `save_intermediate_results` is set and at least one stability subsample succeeded. Accumulated warnings are logged once at the end.

### Consensus evaluation

Same structure as Steps 1 and 2: seeds `base_seed + i * 100`, bootstrap evaluation, best-by-`mean_auc`.

---

## Consensus Mechanics (Cross-Cutting)

These mechanisms are shared by the voting and evaluation logic in every step.

### Graduated relaxation from unanimity

Each step pairs two independent stages (Step 2: L1 and L2; Step 3: RFECV and stability selection) and intersects them. A feature survives when **every** stage agrees, and a stage agrees when at least `required` of its `n_models` replicates voted for it.

`required` **always starts at `n_models`** (unanimity) and is lowered one vote at a time until the intersection yields `min_features_floor` features, stopping early if the next rung would fall below `min_consensus`. Implemented in `relax_consensus_intersection`.

Three consequences worth stating explicitly:

**The panel size is the input; the agreement level is the output.** They trade against each other and the data decides where the trade lands, so the caller specifies one and observes the other. This is why there is no parameter for how conservative to be.

**`votes_required` is the strongest level the panel holds at.** Because the descent always begins at unanimity, the reported level is a property of the data rather than of a setting. An earlier design let the caller choose the starting point via `consensus_threshold`; since relaxation only walks downward, a search starting at 4/10 never tested whether 6/10 would also have worked, and reported 4 for a panel that held at 6. That parameter has been removed. It was also inert: measured across two scenarios at two seeds, starting points of 0.4, 0.6, 0.8 and 1.0 produced the identical feature set.

**The floor is capped at what the votes support.** The most the intersection can ever return is what survives a single vote in every stage. Asking for more than that would guarantee padding with features no model selected, so the request is capped at that ceiling and the capping is logged.

Relaxing gradually rather than falling from intersection straight to union also preserves information: with two stages, union *is* the vote requirement at its floor, so a direct fallback jumps from the strictest setting to the loosest in one step and discards every level between. At `n_models = 10` there are ten rungs; that scheme used two.

### Terminal fallback

Rank-averaging is reachable only when no feature was selected by any replicate in **every** stage, so there is no consensus evidence to aggregate at all. It ranks within the union of features that received a vote somewhere, stepping outside that pool only if it cannot fill the request.

Two alternatives were measured and rejected. Inserting a union rung between intersection and ranking ties rank-averaging in 30 of 31 forced cases (mean F1 0.2300 against 0.2276), because mean importance is exactly zero for a feature no model used, so ranking already confines itself to the voted pool. Firing the fallback earlier, on the grounds that weak agreement is untrustworthy, is unsupported: at equal panel size, voting matched or beat ranking at **every** agreement level from 0.1 to 1.0, and was more reproducible across seeds from 0.1 to 0.8.

### Reported agreement

Every step returns `votes_required`, `agreement` (`votes_required / n_models`), `agreement_label`, and `consensus_outcome` (`consensus`, `relaxed`, `consensus_limited`, or `rank_average`). Labels band the fraction as unanimous (100%), strong (80 to 99%), moderate (60 to 79%), weak (40 to 59%) or minimal (below 40%).

The achievable level is strongly algorithm-dependent, because it depends on how many features the models give non-zero importance to at all. On `omics_standard` at 5 models, Random Forest settles at strong (4/5) with pools of 247 to 696 features, while XGBoost and SVM reach only minimal (1/5) with pools of about 22.

### Bootstrap diversity in evaluation (`cv_evaluate_model`)

When `use_bootstrap=True`, each fold's **training** split is resampled with replacement, stratified by the fold labels, with `random_state = model_seed + fold_idx * 1000` (where `model_seed` is read from the classifier's `random_state`). The **test** split is never resampled. This is the mechanism that makes consensus models genuinely diverse: identical locked hyperparameters, different bootstrap training subsets. The reference model (Step 0) uses `use_bootstrap=False`.

Each fold clones the pipeline, fits it, and produces class probabilities (`predict_proba`, checked on the underlying `clf`, with a one-hot hard-prediction fallback). Probabilities are normalised to `(n_samples, n_classes)` via `ensure_binary_proba`, assembled into an OOF matrix, and scored per fold with `compute_macro_auc_ovr`. `std_auc` uses `ddof=1` (0.0 for a single fold).

### Minimum-feature floor

Every step enforces `min(config.min_features_floor, n_features_in)` as the floor that the relaxation descends towards.

This replaced `max(10, int(0.05 * n_features_in))`, which scaled the floor with the *input* of a step whose purpose is aggressive reduction: 100 features at p=2000, 500 at p=10000. Since the consensus can only ever return features that some model actually selected, such a floor is usually unreachable, and demanding it forced the vote requirement all the way down to 1 on every run.

The floor is additionally capped at the consensus ceiling, the number of features surviving a single vote in every stage. Asking for more than that could only be satisfied by padding with features nothing chose, so the request is capped and the capping is logged as `floor_used` below `floor_requested`.

### Seed spacing summary

| Context | Seed formula |
|---------|--------------|
| Step 1 consensus eval | `base_seed + i * 100` |
| Step 2 L1 voting | `base_seed + i * 200` |
| Step 2 L2 voting | `base_seed + i * 200 + 100` |
| Step 2/3 consensus eval | `base_seed + i * 100` |
| Step 3 RFECV | `base_seed + i * 400` |
| Step 3 stability subsample | `base_seed + i * 500` |
| Step 3 stability model | `base_seed + (i % n_models) * 400 + 500` |
| Consensus factory (`build_consensus_pipelines`) | `base_seed + i * 100` |

---

## Model Details

**Source:** `models/base.py`, `models/evaluation.py`

### Pipeline structure

Every model, for tuning, consensus, stage work, and evaluation, is `Pipeline([('scaler', MinMaxScaler()), ('clf', ...)])`. The scaler is fit per training fold, preventing leakage.

### Tuning (`quick_tune_algorithm`, `quick_tune_all`)

Only `config.algorithm` is tuned; the other three algorithms get fixed defaults (`_default_lr/_xgb/_rf/_svm`) for reference. Tuning uses `RandomizedSearchCV` with:

- `n_iter = quick_tune_iterations`
- `scoring = 'roc_auc'` (binary) or `'roc_auc_ovr'` (multiclass)
- CV = `StratifiedKFold(n_splits = min(5, max(2, min_class_count)))`
- `n_jobs = 1` (deterministic, avoids Windows memory issues)

**Search spaces:**

| Algorithm | Search space |
|-----------|--------------|
| LR  | `C ~ loguniform(0.01, 100)`; `penalty ∈ {'l2'}`; `solver ∈ {'lbfgs', 'saga'}` |
| XGB | `n_estimators ~ randint(50, 300)`; `max_depth ~ randint(3, 8)`; `learning_rate ~ loguniform(0.01, 0.3)`; `subsample ~ uniform(0.6, 1.0)`; `colsample_bytree ~ uniform(0.6, 1.0)`; `reg_alpha ~ loguniform(0.01, 10)`; `reg_lambda ~ loguniform(0.01, 10)` |
| RF  | `n_estimators ~ randint(50, 300)`; `max_depth ∈ {None} or randint(3, 20) draws`; `min_samples_split ~ randint(2, 10)`; `min_samples_leaf ~ randint(1, 5)`; `max_features ∈ {'sqrt', 'log2', None}`; `class_weight='balanced'` |
| SVM | `C ~ loguniform(0.1, 100)`; `gamma ~ loguniform(1e-4, 1e-1)`; `kernel='rbf'`; `probability=True`; `class_weight='balanced'` |

(`uniform(0.6, 0.4)` in scipy notation spans `[0.6, 1.0]`.) GPU detection (`detect_gpu`) probes a trivial `XGBClassifier(device='cuda')` fit; on success XGB tuning and downstream XGB models use `device='cuda'`.

### Consensus pipeline factories

`create_lr/xgb/rf/svm_pipeline_consensus` read the tuned hyperparameters off the tuned pipeline and vary **only** `random_state`. All other hyperparameters are locked so consensus models are comparable; their diversity comes entirely from bootstrap resampling.

- **LR:** locks `C`.
- **XGB:** locks `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, and inherits `device`.
- **RF:** locks `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`; forces `class_weight='balanced'`.
- **SVM:** locks `C`, `gamma`; forces `kernel='rbf'`, `probability=True`, `class_weight='balanced'`.

### Importance extraction per role

| Role | LR / SVM | XGB / RF |
|------|----------|----------|
| Step 2 stage (`_extract_importance`) | max abs `coef_` across classes | `feature_importances_` |
| Step 3 stage (`_extract_importance_step3`) | max abs `coef_` (LinearSVC surrogate for SVM) | `feature_importances_` |

`compute_svm_afi` (in `utils/helpers.py`) provides a min-max normalised `coef_`-based importance for the LinearSVC surrogate, retained for compatibility with legacy importance getters.

---

## Evaluation Metrics

**Source:** `evaluation/metrics.py`, `utils/helpers.py`

### Fold AUC (`compute_macro_auc_ovr`)

Custom macro OVR AUC that is robust to missing classes in a fold: it computes per-class binary AUC only for classes present in `y_true`, skips any slice with a single label, and returns **0.5** (chance) when no class is scorable. This avoids the `ValueError` sklearn raises when a minority class is absent from a small fold.

`_safe_roc_auc` (used for the final test-set AUC) returns `nan` when fewer than two classes are present or, in multiclass, when any class is absent, and extracts the positive-class column for binary problems.

### Per-class metrics (`compute_per_class_metrics`)

For each class, three operating points are reported from `multilabel_confusion_matrix` and `roc_curve`:

- **Default:** sensitivity and specificity at the argmax prediction.
- **Optimal (Youden's J):** the threshold maximising `TPR - FPR`, with its sensitivity and specificity.
- **High-specificity:** sensitivity achievable at 95% and 99% specificity. `_threshold_at_specificity` selects the **maximum-TPR** operating point among those with `FPR <= target` (0.05 or 0.01), so the reported specificity never falls below the requested level. Values are `nan` when the target specificity is unachievable.

### Comprehensive metrics (`build_comprehensive_metrics`)

Assembles a flat, JSON-serialisable dict: test macro AUC, training CV AUC mean and std, generalization gap (`train - test`), overall and balanced accuracy, macro and weighted precision/recall/F1 (`precision_recall_fscore_support` with `zero_division=0`), feature counts and reduction percentage, plus all per-class metrics keyed `{class_name}_{metric}`. `nan` values are stored as `None` to keep `json.dump` from failing.

### Generalization gap labels (`summarise_generalization`)

Based on `abs(train_auc - test_auc)`: `excellent < 0.05`, `good < 0.10`, `moderate < 0.15`, else `poor`.

### Pipeline summary (`build_pipeline_summary_df`)

Per-step table of feature counts, features removed, per-step CV AUC, cumulative reduction, and reduction percentage relative to the Step 0 feature count.

---

## Sample Size Diagnostics

**Source:** `utils/diagnostics.py`, `utils/helpers.py`

`assess_sample_size_adequacy` evaluates four signals before the pipeline runs, each flagged `adequate`, `caution`, or `critical`:

1. **Absolute sample size (n):** `critical` if `n < 30`, `caution` if `n < 100`, else `adequate`.
2. **Samples-to-features ratio (n/p):** `critical` if `n/p < 0.1`, `caution` if `n/p < 1.0`, else `adequate` (Varoquaux 2018, doi:10.1016/j.neuroimage.2017.06.061).
3. **Samples per class:** `critical` if any class has `< 10` samples, `caution` if any class has `10 to 20` (10 inclusive, 20 exclusive), else `adequate` (Figueroa et al. 2012, doi:10.1186/1472-6947-12-8).
4. **CV fold reliability:** `critical` if `min_class_size < n_folds`, `caution` if `min_class_size < 2 * n_folds`, else `adequate` (Kohavi 1995).

**Overall severity** is the worst flag across all four signals. Each signal carries human-readable warnings and recommendations. The diagnostic report is logged when `config.verbose` is set or whenever any signal is non-adequate (critical issues are never silently suppressed).

Each step re-checks adequacy through `print_adequacy_warning`, recomputing **n/p at the current feature count** (feature selection raises n/p as p falls) and re-emitting the absolute-n, per-class, and CV-reliability warnings in the step's context.

---

## References

- Varoquaux, G. (2018). Cross-validation failure: small sample sizes lead to large error bars. *NeuroImage*, 180, 68 to 77. doi:10.1016/j.neuroimage.2017.06.061
- Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. *IJCAI*, 14(2), 1137 to 1145.
- Figueroa, R. L., Zeng-Treitler, Q., Kandula, S., and Ngo, L. H. (2012). Predicting sample size required for classification performance. *BMC Medical Informatics and Decision Making*, 12(1), 8. doi:10.1186/1472-6947-12-8
- Meinshausen, N., and Buhlmann, P. (2010). Stability selection. *Journal of the Royal Statistical Society: Series B*, 72(4), 417 to 473. doi:10.1111/j.1467-9868.2010.00740.x
- Shah, R. D., and Samworth, R. J. (2013). Variable selection with error control: another look at stability selection. *Journal of the Royal Statistical Society: Series B*, 75(1), 55 to 80. doi:10.1111/j.1467-9868.2012.01043.x
- Lundberg, S. M., and Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30. arXiv:1705.07874
