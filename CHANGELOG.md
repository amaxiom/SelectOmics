# Changelog

## 0.7.1 (2026-09-23), pre-release

### Added: what a beta tester needs

The README is rebuilt as the GitHub landing page for a pre-release: a notice
saying what testers are being asked to try, installation from GitHub rather
than PyPI (where the package is not yet published), and, before anything else,
whether the package suits your data, with the evidence and the limits side by
side. The user guide, in both its Markdown and PDF forms, carries the same
section.

`CONTRIBUTING.md`, issue templates and a pull request template are new, so the
notice's promise of templates under **Issues** is true. The bug template asks
first for class counts, because six of the twelve defects in this release's bug
sweep were a class missing from some split, and every one was diagnosed from
them. `CITATION.cff` is new, and is what GitHub reads to offer **Cite this
repository**.

`CODE_OF_CONDUCT.md` is new, rewritten for a single-maintainer project so that
it describes what will actually happen rather than a review process that does
not exist here. The package classifier moves from `Development Status ::
3 - Alpha` to `4 - Beta`, which is what the pre-release notice asks testers to
treat it as.

### Fixed: the README's chained example redid all the work it claimed to save

It ended `.run()  # completes remaining steps and validation`. `run()` without
`resume=True` reloads the data and runs every step from the beginning, so the
example doubled its runtime while telling the reader it was finishing a partial
run. The README now chains through `validate_features()` and says plainly what
`run()` does.

Also corrected there: the README's Benchmarks section still said results were
being re-measured and were not quoted, weeks after they had been; and it said
missing values switch every algorithm to XGBoost, when only LR and SVM switch.

### Changed: the stability claim is qualified where it leads

`BENCHMARKS.md` opened by calling SelectOmics the most reproducible method
tested. That holds on synthetic data and not on real data, where ElasticNet and
LASSO are more stable on every LGG layer, which the same document reports in
section 5.3. The headline now says so, so it cannot be quoted without the
qualifier.

### Changed: the example data is fetched, not committed

`examples/` and `benchmarks/` carried about 400 MB of MLOmics CSVs. They are now
downloaded on first use by `examples/mlomics_data.py` and cached under
`examples/.mlomics_cache/`, which is ignored. The repository went from 422 MB
across 217 tracked files to **12.8 MB across 205**.

Size was the smaller reason. The published numbers were reproducible only from
files that existed nowhere but this repository, and shipping them re-distributed
third-party CC-BY data under this package's Apache-2.0 licence. Fetching fixes
both.

Three properties make the fetch trustworthy rather than merely convenient:

- **The revision is pinned** to an immutable dataset commit, not `main`, so an
  upstream release cannot silently change what the notebooks produce.
- **Every file is checksummed.** A SHA-256 is recorded for each of the fifteen
  files and verified after download and on every cache hit.
- **The variant is part of the path.** MLOmics publishes `Original`, `Aligned`
  and `Top`; these use `Aligned`. They are not interchangeable: on GBM microRNA,
  `Original` shares only 143 of its features with `Aligned` and the shared
  values differ by up to 7.45. Pointing the URLs at the wrong one would change
  the result while still appearing to work.

`benchmark_lgg.py` builds its merged matrix from the same pinned revision, so
the benchmark and the examples cannot drift apart.

### Changed: every example re-run against the fetched data

The fetched data is not quite what was shipped. Upstream is a strict subset:
the labels are byte-identical, the shared values are identical on all eight
layers, and the committed copies carried between 1 and 39 extra features per
layer. Most are all-zero constants; a few are low-variance near-duplicates that
upstream appears to have de-duplicated.

Those extras still changed the results, which is worth understanding. Tuning
runs before Step 0, necessarily, since Step 0 is the all-features reference, so
the randomised search scores its candidates on the unfiltered matrix. The same
seeded candidates are drawn either way, but a different one wins at 325 columns
than at 286: on GBM microRNA, depth 17 and `max_features=None` became depth 6
and `sqrt`. **Constant columns that the pipeline itself discards in Step 1a
still determine which model is tuned, and so which panel is delivered.** Two
labs with the same assay, one of which dropped empty probes first, will get
different panels. Moving constant removal ahead of tuning would fix it and
would change every number this package has published, so it is recorded here
rather than done.

All twelve notebooks were re-executed end to end, zero failures, and every
figure in `user_guide/` regenerated from the new quickstart run. What changed:

- **The recommended step is the same in all twelve.** Panel sizes moved and
  AUCs drifted by at most 0.013 in both directions, but the pipeline chose the
  same step every time, under genuinely different tuned hyperparameters.
- **Runtimes fell everywhere**, sometimes sharply: GBM microRNA 247 to 26
  minutes, OV copy-number 194 to 102, GBM copy-number 69 to 18.
- **Step 3 now skips in seven of the ten per-layer notebooks**, up from five,
  making the benchmark's "the shipped pipeline is effectively two steps" point
  visible in the examples.
- **One documented lesson changed character rather than value.** Two notebooks
  used to show the recommendation overriding the last step; only OV microRNA
  still does. On OV methylation Step 2 now returns 29 features, one short of
  Step 3's gate of 30, so Step 3 skips and there is nothing to override. It is
  now documented as what it became: the near-miss showing that whether the most
  expensive step runs at all can turn on a single feature.
- **The two other lessons survive.** OV copy-number still validates well and
  scores 0.584 held out, a gap of 0.226 labelled `poor`. Merging still
  concentrates on mRNA, now 24 of 25 features for GBM and 15 of 15 for OV.

The LGG benchmark was re-run on the same pinned revision, 400 trials over
5 layers by 8 methods by 10 folds, 8.8 hours. **Every conclusion in section 5
survived.** SelectOmics still leads on none of the layers for AUC, still
returns the shortest working panel on all five (18 features from 34 021 at
0.9885 against a best of 0.9891), and is still beaten on stability by
ElasticNet and LASSO on every layer. The layer where it places second on AUC
moved from CNV to the merged matrix, which is the case the package targets.
Its cost per fold on the single layers roughly doubled, which the example
notebooks did not do on the same data change; the cause is not established and
section 5.4 says so rather than guessing.

The data behind the previously published numbers is an older MLOmics release
that no longer exists upstream. It is kept outside the repository in
`archive/pre-fetch-data-2026-09-21/`, along with the August LGG benchmark
results, so those numbers remain reproducible.

### Fixed: a bug sweep across every algorithm and target shape

Ran the full pipeline over `{LR, XGB, RF, SVM}` by `{binary, multiclass}` with
visualisations on, plus a continuous target, then probed the resume path. All
eight classification cells passed. Eleven defects came out of it, all of them
silent or cryptic rather than loud. No configuration, capability or output
changes on a run that was already working.

**Resuming under a different algorithm mixed two learners.** The checkpoint is
stamped with the package version and a source-data fingerprint, and a mismatch
on either is refused because "mixing steps computed under two builds produces a
result that matches neither". The algorithm was never stamped. Measured: an RF
run wrote a checkpoint, a resume configured for XGB restored it, and the run
finished delivering RF's panel with `step0` still recording `Ref-RF` while the
metrics recorded `algorithm: XGB` in a file named
`final_XGB_comprehensive_metrics.json`. Every feature in that panel was chosen
by Random Forest. The algorithm is now stamped and a mismatch refused, in the
style of the two existing guards.

**A class absent from a split was silent in five places.** It is one failure
mode wearing different clothes:

- `cv_evaluate_model` assumed `predict_proba` returns one column per class. It
  returns one per class the estimator *saw*, so a fold whose training split
  lacked a class raised a numpy shape mismatch naming neither fold nor class.
  Columns are now scattered by `classes_`, a no-op when all classes are
  present.
- `stratified_cv_validation` **raised** where `leave_one_out_validation` and
  `bootstrap_validation` both skip, losing every feature set's result rather
  than one diagnostic row. It now uses the guard its two siblings already had.
- A class the stratified split leaves out of the test set entirely was visible
  only in the verbose split summary, which is off by default. It is now warned
  at the split, where the cause is.
- The resulting held-out AUC is `nan` and was published as the headline number
  in silence. Now warned, naming the absent classes.
- `summarise_generalization` maps a `nan` gap to `poor`, which reads as
  "generalises badly" when nothing was measured. The label set is part of its
  contract and is unchanged; the silence is not.

**A continuous target** returned scikit-learn's message about the minimum
number of groups, naming neither the column nor the cause. It now raises a
`ValueError` naming the singleton classes and saying that a continuous target
looks like this after label encoding, and that SelectOmics classifies.

**The learning curve lost its left end without saying so.** It asks for ten
training sizes from 10% with five folds, and at the smallest sizes a subsample
can hold one class, which LR and SVC refuse to fit. scikit-learn scores those
folds `nan`, the mean propagates it, and matplotlib draws a gap. The figure is
unchanged, since `nanmean` would plot a point averaged over fewer folds than
its neighbours; a warning now names the sizes lost and the algorithm that
refused them. This gave `evaluation/visualization.py` a module logger, the only
module in the package without one.

**Also:** the cross-validation protocol never reported degraded folds, though
this module has `_warn_if_degraded` for exactly that; its verbose score-range
line called `min()` on a list the line above it allows to be empty; and the
checkpoint fingerprint guard was a 116-character line with a run of spaces
where a continuation had been lost.

### Changed: the typeset user guide rewritten for 0.7.1

`user_guide/` described version 0.6.0 and had not been rebuilt since. Roughly
half of it documented things that no longer exist: a fourth selection step
(SHAP and permutation explainers), `consensus_threshold`, the four
`*_percentile_range` tuples, `enable_final_importance_analysis`, and a
`get_feature_importance()` method. It also predated everything 0.7.x added, so
it never mentioned the recommendation, the one-standard-error rule, the
selection-optimism guard, the graduated relaxation from unanimity,
`class_aware_correlation`, `step2_role`, `min_consensus`,
`min_features_floor`, held-out step ranking, or the rewritten nested CV.

All seven source files were rewritten against the current code and the PDF
rebuilt: 68 pages, no unresolved references, no overfull lines. The shipped
figures went with it. They came from a 0.6.0 run and included a Step 4 box plot
for a step that no longer exists and a `feature_importance` panel whose
plotting function was removed with it; every figure is now from a current
quickstart run (GBM microRNA, RF, seed 42, 325 features to 29 at a held-out AUC
of 0.856), which the caption names so it can be reproduced.

New chapters cover the pipeline evaluation and the recommendation, choosing key
parameters, worked examples, and benchmarks. `part3_steps34_models_metrics.tex`
is now `part3_step3_models_metrics.tex`.

### Fixed: documentation that had drifted from the source

Found by checking every documented symbol against the code it describes.

- `class_aware_correlation` and `step2_role` were absent from the user guide's
  configuration table, and `allow_union_rung` appeared only in prose.
- The metrics file was documented as `comprehensive_metrics.json` in the README,
  the user guide and `save_results()`'s own docstring. It has always been
  written as `final_<ALGO>_comprehensive_metrics.json`.
- `save_results()`'s output list omitted `recommended_features`, the file the
  recommendation exists to produce, while the method table above it named it.
- `stability_threshold` was documented as defaulting to `0.6`. Its declared
  default is `None`, which resolves to 0.6; a caller reading the value back off
  the config sees `None`.
- Three Step 4 references outlived the step: two in the methods guide, one in
  `SelectOmicsConfig.quick()`'s docstring, which still said "steps 3 and 4 off".
- The methods guide gave `step2_target_retention` a default of 0.6. It became
  1.0 in 0.7.0 when the field changed from a fraction of the input width to a
  fraction of the reachable pool.
- `__author__` said "SelectOmics Development Team" where `pyproject.toml` says
  Amanda S Barnard.
- The README had no link to any of the guides, so a reader arriving from the
  package index had no route to them.

### Added: `.gitignore`

Covering build products, coverage, notebook checkpoints, pipeline output
directories, and the 148 MB of `*_input.csv` files the example notebooks cache
from data that is already tracked. The benchmark result directories are
deliberately not ignored: `BENCHMARKS.md` cites their raw CSVs as the evidence
behind every number on the page.

### Fixed: a corrupt parquet or feather file reported pyarrow's error, not ours

Every reader failure is wrapped in a RuntimeError naming the path and the
format attempted, which is what the docstring promised. It did not hold for
parquet or feather: pyarrow raises ArrowInvalid, which subclasses ValueError,
and ValueError took a pass-through branch meant for the loader's own refusals.
A corrupt parquet surfaced pyarrow's message with no mention of the file or the
format, while the same corruption in HDF5 or Excel was named properly.

The unsupported-format check now happens before the read, so the pass-through
is no longer needed and reader errors are wrapped whatever their type.
FileNotFoundError still passes through untouched, and the loader's own
ValueErrors (undetectable extension, unsupported format, empty frame) are
unchanged.

### Changed: every example notebook rebuilt around the recommendation

The twelve notebooks in `examples/` now enable every evaluation and deliver the
panel the pipeline recommends rather than the last step's. Ten of them run one
layer each, two cohorts by four omic layers plus the merged matrix, with
identical structure so layers and cohorts compare directly. Three are new
(`GBM_SelectOmics_Methy`, `GBM_SelectOmics_CNV`, `OV_SelectOmics_CNV`), and
`examples/README.md` indexes all of them with what each produced.

Every notebook was executed end to end before shipping, with zero errors and
every expected figure present, then cleared. Two runs show the recommendation
overriding the last step: OV Methy, where Step 3 cut 35 features to 3, and OV
miRNA, where it cut 121 to 10. One shows the held-out test doing what no
training-split statistic can: OV CNV validated at 0.81 and scored 0.585 unseen.

Nested CV is off in the layer notebooks. Measured on GBM miRNA it took 282
minutes against 247 for the run alone, because each outer fold reruns selection
and all three validation protocols; the showcase demonstrates it instead.

### Documented: how to run for sensitivity rather than precision

`allow_union_rung=True` is the one measured lever that trades precision for
recall (0.902 to 0.726 for 0.844 to 0.897). The user guide now carries a
section on when it pays, `BENCHMARKS.md` section 7.3 gains recall figures and a
per-scenario breakdown, and the field's own comment records the numbers. No
preset ships for it: the regime where it wins is not the one the package
targets, and a name like "sensitive" would invite its use where it costs a
third of the precision for no recall at all.

### Removed: dead and stale code

`SelectOmicsSelector` in `benchmarks/benchmark_synthetic.py` was never
instantiated: the live arm is `SelectOmicsMultiStepSelector`, and the only
other mention was an `isinstance` check that could not be true. Both are gone,
82 lines of them, and the benchmark's method list is unchanged.

`pypi_staging/` held the 0.6.1 wheel and sdist plus copies of the README,
changelog and pyproject that had drifted from the ones above them. Removed,
along with a stale `coverage.json`, two `.bak` benchmark result files, and the
`.pytest_cache` and `SelectOmics.egg-info` build artefacts. `MANIFEST.in` no
longer prunes a directory that does not exist.

The 0.7.0 tree is archived at
`SelectOmics/archive/SelectOmics-0.7.0-2026-09-15.zip`, code, docs, notebooks
and benchmark results, with its own manifest listing what was left out.

### Fixed: the recommendation was computed and then ignored

`build_recommendation` was public, tested, and called from nowhere, so a run
produced the evidence for choosing between steps and never used it. `run()`
now stores it as `results['recommendation']`, `get_recommended_features()`
returns its panel, and `save_results()` writes `recommended_features.*` beside
`selected_features.*`, which remains the last step's panel. The CLI names both.

### Fixed: the held-out test scored the last step, not the recommended panel

Whenever the two differed, the reported test metrics described the panel the
user had just been told not to use. `_run_final_test_evaluation` now scores the
recommended panel and records which one it scored (`panel`, `step_name`,
`n_features`).

### Fixed: the generalisation gap was always measured from Step 0

The training AUC it compares against was chosen by iterating
`reversed(['step3', 'step2', 'step1', 'step0'])`, which visits Step 0 first,
and stopping at the first match. Every gap was therefore measured from the
full-feature reference, whichever panel was tested. `_panel_cv_result` now
returns the CV result for the panel being scored.

### Fixed: a skipped step was credited with the reference AUC

A skipped or disabled step files Step 0's evaluation as its `consensus_result`.
Read directly, that put the full-feature AUC on the row of a panel that is not
the full feature set, in the pipeline summary and in the recommendation. Step 3
skips in most default runs, so this was the common case.

### Changed: nested CV measures the procedure that actually runs

`run_nested_cv` forced `enable_step3=False` and scored Step 2's panel whatever
the configuration, so it estimated a procedure nobody ran, and it could not
show whether the recommendation's choice held up. With `step2_role='prefilter'`,
which requires Step 3, building the inner config raised and nested CV failed
outright. Each outer fold now runs the same steps, validation and
recommendation as `run()`, and scores every step's panel on the held-out fold.
`fold_aucs` and `mean_auc` describe the recommended panel; `fold_details`,
`step_summary`, `last_step_mean_auc`, `best_step_mean_auc` and `mean_regret`
are new. A fold that produces no AUC is reported as missing rather than scored
0.0.

### New: `enable_holdout_ranking`

Ranks the steps on `holdout_ranking_splits` folds (default 5) held out of the
training split rather than on validation that shares its samples with
selection. Off by default; costs one extra run of the selection steps per
fold. The test split is not involved, so the final test stays independent.

Measured on seven datasets it agreed with the training-split ranking on four
and changed the choice on three, and each change was for the worse: 122
features instead of 11 and 77 instead of 3 where the true features are known,
and 592 noise features on a null control, for AUC differences of 0.002 to
0.025. At n << p an AUC ranking turns on noise and noise favours the larger
panel. Its real benefit is the quality label on data with no signal, which the
selection-optimism guard also reaches from the training split at no cost, so
the flag stays off unless predictive AUC is the deliverable. What it ranks is
the step, not the exact panel: each fold selects its own features.

### New: selection-optimism guard on the quality label

Validation scores each panel on the samples its features came from, so the
label could pass noise: on a null control the recommended panel validated at
0.84 and scored 0.46 on unseen rows. When a panel's score exceeds the
least-selected panel's by more than `MAX_PLAUSIBLE_SELECTION_GAIN = 0.05`, a
RECOMMENDED label is lowered to CAUTION and the reason says why. Measured
across four datasets with signal the gain reached 0.021; on the null control,
0.23 to 0.37. The guard changes the label, never the choice, and is off when
the ranking already comes from held-out folds.

### Fixed: bootstrap seeds aliased across consensus models

`model_seed * 1000 + fold_idx` collided at the shipped preset, so models 10 to
19 repeated models 0 to 9 and consensus measured agreement between duplicates.

### Fixed: resume never re-read the source file

A checkpoint now carries a SHA-256 fingerprint of the data. Editing the input
and resuming reported the old results as the new file's.

## 0.7.0

### Removed: Step 4 (Explainers)

The pipeline is now three selection steps against a Step 0 reference, not four.
`selection/step4_agnostic.py`, `pipeline.run_step4_agnostic()`,
`pipeline.get_feature_importance()`, the `feature_importance` figure, and the
`enable_step4` / `step4_target_retention` config fields are gone. `shap` is no
longer a dependency.

The measurement that decided it: in the ablation arms, `SelectOmics` and
`SO_S1->2->3` returned **identical feature sets** on every scenario and every
seed. Step 4's contribution to the delivered panel was exactly 0.000. It was
either reducing a panel too small to rank meaningfully, or explaining one it
did not change.

The step also pulled the package toward a purpose it does not have. SelectOmics
exists to return the right features reproducibly; attribution is a separate
question, and SHAP or permutation importance can be run on the delivered panel
by anyone who wants a ranking. Shipping a half-committed explainer inside a
selection pipeline made both jobs worse.

Consequences for callers:

- `get_selected_features(step=...)` accepts `'step0'` through `'step3'`.
- `feature_provenance` has `survived_step1` through `survived_step3`.
- Checkpoints written by 0.6.x are not resumable; they are rejected on a
  version mismatch rather than silently mixing builds.
- A config that still sets `enable_step4` or `step4_target_retention` loads
  with the usual unknown-key warning and ignores the field.

### Changed: benchmark comparators run at library defaults

Every comparator (LASSO, ElasticNet, RFECV, RF importance, permutation
importance, SHAP) now uses its library's default hyperparameters, and the LGG
harness uses a plain `SelectOmicsConfig(...)` rather than the tuned `omics`
preset. SelectOmics is not tuned to these scenarios, so nothing else is either.
This compares defaults against defaults, which is the comparison the page
claims to make. Published figures from before this change are not comparable
and `benchmarks/BENCHMARKS.md` is marked accordingly pending a re-run.

### Fixed: Step 1's correlation filter is now class-aware

The long-documented limitation is implemented. The filter used total Pearson
correlation, which cannot distinguish two probes measuring the same thing from
two independent markers of the same disease: both are correlated, and it
deleted one either way. In omics, where co-regulation is the norm, that removes
exactly the features worth keeping.

Redundancy is now judged on the **pooled within-class correlation**. Centring
each feature on its own class mean removes the between-class covariance,
leaving only association the label does not explain. Genuine duplicates stay
correlated after centring; co-regulated markers do not. The survivor of a
redundant pair is chosen by correlation ratio (eta-squared) rather than by
variance, which was unrelated to the outcome and therefore effectively
arbitrary.

On the synthetic case built to expose this, with 20 informative features:

| signal strength | mean \|r\| among informative | old: kept | new: kept |
|---|---|---|---|
| 1.2 | 0.851 | 1/20 | **20/20** |
| 2.0 | 0.941 | 1/20 | **20/20** |
| 2.5 | 0.961 | 1/20 | **20/20** |

End to end on `adversarial_easy`, the scenario where the pipeline collapsed:
F1 0.052 to **0.740**, precision 0.05 to **1.000**, cross-seed Jaccard 0.037 to
**0.436**.

**Be careful reading the average.** Across the six development scenarios the
mean F1 goes 0.766 to 0.880, but that gain is almost entirely the one scenario
this was built to fix. Excluding it, mean F1 is 0.909 against 0.908 -- flat --
and cross-seed Jaccard 0.753 against 0.774. Two scenarios get slightly worse
(`omics_standard` -0.039, `omics_imbalanced` -0.050), both trading recall for
precision that was already 1.000.

The honest summary is: it eliminates a catastrophic failure mode and is
otherwise neutral. That is reason enough, but it is not a broad accuracy gain
and should not be presented as one.

Set `class_aware_correlation=False` to restore the previous behaviour.

### `min_consensus` defaults to 0.4, from a re-run configuration search

The four-phase search was re-run over a corrected parameter space (five inert
dimensions removed, `min_consensus` and `min_features_floor` added, Phase 1
widened from one scenario to two so it can actually rank).

`min_consensus` is the strongest parameter in the space, and the response has a
clear interior optimum:

| setting | `None` | **0.4** | 0.6 | 0.8 | 1.0 |
|---|---|---|---|---|---|
| Phase 1 composite | 0.751 | **0.784** | 0.743 | 0.599 | 0.503 |
| Phase 2 mean F1 | 0.475 | **0.563** | 0.495 | -- | -- |

Phase 1: 205 configs, 2 scenarios, 3 seeds, `r = -0.746, p < 0.0001`.
Phase 2: 30 configs, 3 scenarios, 5 seeds, `r = -0.422, p = 0.020`, with 0.4
winning on **every scenario individually** (adversarial_easy 0.066 vs 0.059,
omics_standard 0.938 vs 0.836, omics_tiny_n 0.684 vs 0.530).

The direction matters: 0.4 is **stricter** than the previous `None`, not
looser. Without a brake the relaxation can bottom out at one vote in ten, and
that costs both accuracy and reproducibility. But unanimity is worse still
(F1 0.340, Kuncheva 0.127), because almost nothing survives. Enforcing
agreement helps; demanding all of it does not.

### Changed: `omics()` no longer pins `n_consensus_models` below the default

The preset set 5 while the dataclass default is 10, so a preset aimed at the
hardest regime -- small n, large p, where stability matters most -- was the
*least* stable configuration on offer. The override is dropped and it now
inherits 10.

The re-run configuration search ranks this parameter highest (Spearman
`r = 0.835, p < 0.0001` in Phase 2, monotonic across 3, 5 and 10 at composite
0.641, 0.687, 0.706). The 5 came from the earlier search, half of whose
dimensions have since been shown inert.

This roughly doubles the per-run cost of the `omics` preset. Pass
`n_consensus_models=5` explicitly to restore the old behaviour.

### Changed: the synthetic benchmark measures the shipping model count

Full mode used 5 consensus models against a shipping default of 10, so the
published figures described a weaker configuration than users receive, on the
parameter that most affects the result. Full mode now reads the dataclass
default. Fast mode still uses 3: it exists for iterating on evaluation code
and its numbers are not published.

### Fixed: `run(resume=True)` could silently mix two builds

Checkpoints carried no version stamp, so a checkpoint written by 0.6.1 would
be restored by 0.7.0 without complaint. Since selection semantics change
between these versions, the resumed run computed its early steps under one
rule set and its later steps under another, producing a result that describes
neither, with nothing to detect it.

Checkpoints now record the writing version. A checkpoint from a different
build (or from any pre-0.7.0 build, which carries no stamp at all) is refused
with a warning naming both versions, and the run starts fresh. Delete
`.selectomics_checkpoint.pkl` to silence it.

### `min_features_floor` defaults to 15

Settled by an isolated sweep: every other setting held at the search winner,
only the floor varying, over all six development scenarios at five seeds.

| floor | composite | F1 | Kuncheva | runtime |
|---|---|---|---|---|
| 5 | 0.716 | 0.624 | 0.490 | 162 s |
| 8 | 0.784 | 0.714 | 0.614 | 160 s |
| 10 | 0.808 | 0.745 | 0.660 | 159 s |
| **15** | **0.820** | 0.760 | **0.683** | **157 s** |
| 20 | 0.819 | 0.763 | 0.674 | 327 s |
| 25 | 0.820 | 0.764 | 0.672 | 389 s |

Monotonic to 15, then flat within 0.001. Floors of 20 and above pass enough
features to wake Step 3, the most expensive step, costing two to two and a half
times the runtime for no measurable gain. The improvement from 10 to 15 comes
almost entirely from `omics_imbalanced` (F1 0.818 to 0.918); the other five
scenarios tie exactly.

The isolation matters. Across three earlier evaluation sets this parameter read
as dominant, then insignificant, then dominant again, because each reading came
from a different scenario mix with the floor confounded against everything else
in the shortlist. Only the sweep above varies it alone.

`n_consensus_models` is confirmed at 10 (`r = 0.835, p < 0.0001` in Phase 2,
monotonic across 3, 5, 10).

### Removed: `consensus_threshold`. Added: `min_consensus`

`consensus_threshold` set where the graduated relaxation **started**. Because
the relaxation stops at whichever vote level first satisfies
`min_features_floor`, and that landing point is a property of the data and the
floor, starting lower only skipped rungs on the way to the same answer.

Measured on `omics_standard` and `omics_genomics` at two seeds, thresholds of
0.4, 0.6, 0.8 and 1.0 returned not merely the same feature *count* but the
**identical feature set** in all four groups. Its influence on the
configuration objective fell from Spearman `r = -0.837` (measured before the
relaxation existed) to `r = -0.013, p = 0.86`.

It also corrupted the diagnostic it fed. Relaxation walks only downward, so a
search starting at 4/10 never tested whether 6/10 would have worked, and
reported `votes_required = 4` for a panel that in fact held at 6.

**Relaxation now always starts at unanimity.** There is no parameter, because
there is only one sensible place to begin: the strongest claim, weakened only
as far as forced. `votes_required` is consequently the *strongest* level at
which the returned panel holds.

**`min_consensus`** (default `None`) replaces it as a **floor** on agreement
rather than a starting point. Relaxation stops there even if the feature floor
has not been met, so "I would rather have 4 features at 8/10 than 10 features
at 1/10" is now expressible; `min_consensus=1.0` means unanimity or nothing.
When it and `min_features_floor` conflict, agreement wins and the step reports
`consensus_limited` with both numbers.

Unlike the parameter it replaces, this one demonstrably works: across all four
algorithms at two seeds, settings of `None`, 0.6 and 1.0 produced 2 to 3
distinct feature sets per run.

There is no translation between the two. A starting point and a floor are
different things, so `SelectOmicsConfig.omics()` drops its `0.442` rather than
converting it.

### New: every result reports how much agreement it carries

Removing `consensus_threshold` leaves no way to *ask* for conservatism, so the
pipeline instead *reports* it. Steps 2, 3 and 4 return `agreement` (the
fraction of models that had to agree) and `agreement_label`:

| achieved | label |
|---|---|
| 100% | unanimous |
| 80 to 99% | strong |
| 60 to 79% | moderate |
| 40 to 59% | weak |
| below 40% | minimal |
| rank fallback | none (ranked, not agreed) |

The final metrics record carries `achieved_agreement` and `agreement_label` in
place of the old `consensus_threshold` field. A run now reports "10 features at
6/10 agreement (moderate)" rather than a bare count.

This is worth reading per algorithm: on `omics_standard` at 5 models, RF
settles at **strong** (4/5) while XGB and SVM reach only **minimal** (1/5),
because RF gives non-zero importance to 247 to 696 features against XGB's 22.

### Changed: the rank fallback stays inside the voted pool

Rank-averaging fires only when no feature was selected by any replicate in
every stage. It now ranks within the union of features that received a vote
somewhere, stepping outside only if that pool cannot fill the request.

Measured, this changes nothing today: mean importance is exactly zero for a
feature no model ever used, so ranking already confined itself to the voted
pool in **0 of 161.5** features on average. The change makes that guarantee
structural rather than an accident of how tree importances behave.

Two alternatives were tested and rejected. A union rung between intersection
and ranking ties rank-averaging 30 times in 31 (mean F1 0.2300 vs 0.2276), and
firing the fallback earlier on weak agreement is unjustified: at equal panel
size, voting beat ranking or tied it at **every** agreement level from 0.1 to
1.0, and was also more reproducible from 0.1 to 0.8.

Minor rather than patch: four public config fields are removed. Selection
results change on every algorithm, so figures generated with 0.6.1 must be
regenerated before being compared against 0.7.0 output.

Summary of what moves the numbers:

- Step 2's importance threshold was inert and is now rank-based. This affected
  7 of 8 algorithm-stage combinations; only ridge LR, whose coefficients have
  no exact zeros, had a working threshold before.
- Step 2's XGB stages use `colsample_bytree=0.3`, widening the vote pool.
- Step 4 no longer cuts panels too small to rank, and enforces its floor.
- `min_features_floor` default 5 to 10, which sets panel size in practice.
- Consensus relaxation now always starts at unanimity, so `votes_required` is
  the strongest level a panel holds at rather than an artefact of a setting.

Two config fields are removed (`consensus_threshold`, and the four
`*_percentile_range` bounds) and three added (`min_consensus`,
`step2_target_retention`). A config setting a removed field loads and warns.

### Changed: `min_features_floor` default raised from 5 to 10

This is the setting that actually determines panel size. Every step relaxes its
consensus requirement one vote at a time until the floor is met, so the floor
decides how many features come out; `consensus_threshold` sets where that
relaxation starts, not where it stops.

Measured end to end over five synthetic omics scenarios at two seeds:

| floor | F1 | Precision | Recall | Cross-seed Jaccard | Step 3 runs |
|---|---|---|---|---|---|
| 5 | 0.705 | 0.967 | 0.583 | 0.414 | 0/10 |
| **10** | **0.777** | 0.780 | 0.843 | **0.553** | 0/10 |
| 20 | 0.613 | 0.711 | 0.693 | 0.195 | 4/10 |
| 30 | 0.690 | 0.786 | 0.703 | 0.255 | 5/10 |

10 is better on ground-truth recovery *and* on cross-seed reproducibility, so
this is not the usual precision-for-recall trade. Precision is what pays,
falling from 0.967 to 0.780.

Lower it to 5 when the expected signal is compact: a floor above the true
number of informative features has to pad the panel, and on the one
development scenario with 5 informative features floor 10 drops F1 from 0.864
to 0.486. Three of the five scenarios carry 10 or more, which is part of why 10
wins on the mean.

**Raising it to force Step 3 to run is counterproductive.** Floors of 20 and 30
are the only settings that supply the 30 features Step 3 requires, and at
exactly those settings the pipeline is worse than at floor 5 on both metrics.
On synthetic data Step 3 is not a dormant capability waiting for the right
setting.

### Removed: the four percentile search ranges

`l1_percentile_range`, `l2_percentile_range`, `rfecv_percentile_range` and
`shap_pi_percentile_range` are gone. They were the search *domain* of an
internal bisection solver whose *target* is a retention fraction the caller
already sets separately, so they asked users to constrain a solver for a
quantity they never see.

Three concrete harms:

- **They silently contradicted the target.** The default upper bound of `0.6`
  keeps `ceil(0.4 x n)` at its extreme, so any `stepN_target_retention` below
  40% was unreachable and the search returned its nearest endpoint without
  complaint. The test suite had already accommodated this: the Step 4
  retention test set 0.3 and asserted only `n_out <= n_in * 0.6`, annotated
  `# Very loose check`.
- **The asymmetry was backwards.** Step 2's retention target was hardcoded at
  0.6 and not exposed at all, while two of its solver bounds were public.
- **They absorbed configuration-search budget for nothing.** The search behind
  `SelectOmicsConfig.omics()` ranked all four last by influence, at Spearman
  `r` of 0.045, -0.038, -0.002 and 0.002, with p-values from 0.52 to 0.98,
  against `consensus_threshold` at `r = -0.837`. It nonetheless reported
  optima to three decimal places, and those numbers shipped in the preset.

Replaced by **`step2_target_retention`** (default 0.6), exposing what Step 2
previously hardcoded and making all three reducing steps symmetric. Searches
now run over the full `[0, 1]` domain, which costs one extra bisection step,
well inside the existing `max_iter=10`.

Nothing is deprecated-and-ignored: a config still setting a removed field gets
the standard unknown-key `UserWarning`. The example configs under `examples/`
have been migrated.

`SelectOmicsConfig.omics()` keeps its remaining values but now carries a
warning that they need re-deriving, since a search that spent part of its
budget on inert parameters is not a clean read on the ones that mattered.

### New: the binary search says when it cannot reach its target

`_binary_search_percentile` returned its closest endpoint when the target lay
outside what `evaluate_fn` could produce, which is how an inert percentile went
unnoticed for so long: Step 2's threshold returned an identical feature count
at every point in its range, the search dutifully reported an "optimum" to four
decimal places, and a configuration search then tuned that number.

It now emits a `UserWarning` naming the caller, the target and the closest
achievable value. A caller asking for 60% retention and receiving 0.6% is told
that the retention setting is not what determined the result.

### Fixed: `NameError` when Step 3's stability stage failed entirely

`stability_votes = np.full(n_features, ...)` referenced a name that does not
exist in that scope, so the total-failure branch raised instead of abstaining.

### Fixed: Step 2's percentile threshold did nothing

Step 2 kept a feature when `imp > np.percentile(imp, percentile * 100)`. Tree
importance is exactly `0.0` for every feature never used in a split, and on
this package's target data that tie mass is overwhelming: a single XGB stage
fit left 1988 of 2000 importances at zero (99.4%), 4981 of 5000 (99.6%) and
9983 of 10000 (99.8%). `np.percentile` therefore returned `0.0` at every point
in the default `(0.0, 0.6)` search range, the test collapsed to `imp > 0`, and
the kept count was **identical at every percentile** (12, 12, 12, 12, 12, 12,
12 across the grid on `omics_standard`).

Two consequences: the binary search over `l1_percentile_range` and
`l2_percentile_range` was optimising a constant function, and a stage asked to
retain 60% retained 0.6% instead.

Thresholding is now by importance rank, capped at the number of features the
model gave a non-zero importance to. The cap matters: ranking past it would
order features by array index rather than by merit, since a feature the model
never split on carries no evidence either way. Where the cap binds, behaviour
matches the old rule exactly.

`step2_regularization` now logs when the percentile is saturated by the pool,
which is the difference between "the threshold chose these" and "these were
all there was to choose from".

### Changed: Step 2's XGB stages use `colsample_bytree=0.3`

With every feature on offer, boosting converges onto the same handful in every
tree and the importance array Step 2 votes on is nearly empty (14 to 18
non-zero of 1000 to 10000). Restricting each tree to a random 30% spreads
usage and widens the pool to 20 to 30.

Ground-truth recovery improves on `omics_imbalanced` (F1 0.571 to 0.750) and is
unchanged on `omics_standard`, `omics_high_dim` and `omics_genomics`. Precision
stays at 1.000 in all four, so the features recovered are real ones. It is also
marginally faster. Neither `colsample_bytree=0.1` (recovery falls on three of
four) nor 300 estimators (bit-identical at twice the cost) was worth keeping.

### Fixed: Step 4 cut panels it could not meaningfully rank

Step 4 is asked to keep `step4_target_retention` of its input and never to go
below `min_features_floor`. Those coexist only while
`ceil(n_in * retention) >= floor`. Below that crossover the output size was
pinned to the floor regardless of what SHAP and permutation importance found:
with the defaults, an 8-feature input always produced exactly 4 features.

Measured over the synthetic development scenarios, that cost **0.14 F1 and
0.16 Kuncheva** against simply passing the 8 through. The panels Step 4
produced were smaller *and* less accurate *and* less reproducible than its own
input.

Step 4 now reduces only above the crossover. Below it the step still runs in
**explain-only** mode.

### Fixed: Step 4's minimum-feature floor was not enforced

The floor appeared only in a fallback that fired when *nothing* cleared the
vote threshold, so one surviving feature produced a panel of one while zero
surviving produced a panel of `min_features_floor`. On `adversarial_easy` this
returned 4 features from an 11-feature input against a floor of 5.

Step 4 now uses the same graduated relaxation as Steps 1 to 3
(`relax_consensus_intersection`), and reports `votes_required` and
`consensus_outcome` alongside them.

### New: Step 4 explains without reducing

Step 4 is the only step that attributes importance to individual features. A
panel handed on by Steps 2 and 3 arrives with no per-feature account of why its
members were chosen; Step 4 supplies it. Earlier versions skipped the step
outright below five features, discarding that attribution exactly when the
panel was small enough for a reader to want it feature by feature.

Step 4 now produces a real explanation whenever it has at least two features to
attribute between (`_MIN_FEATURES_TO_EXPLAIN`, was `_MIN_FEATURES = 5`).
`importance_df` carries real SHAP values, real permutation importances and real
consensus vote counts; the votes are reported as diagnostics, naming the
features that *would* have been dropped, without acting on them.

Two new result keys distinguish the cases:

| `skipped` | `reduced` | Meaning |
|---|---|---|
| `True` | `False` | No explanation produced; importance columns are filler. |
| `False` | `False` | Explained intact; importance columns are real. |
| `False` | `True` | Explained and reduced. |

Skipping the binary search in explain-only mode also removes most of Step 4's
runtime on small panels.

### Fixed: off-by-one in rank thresholds

`(1.0 - 0.85) * 1000` evaluates to `150.00000000000003`, so a request for
exactly 150 features kept 151. Both Step 2 and Step 4 now round before
ceiling.

## 0.6.1

Bug-fix and performance release. The public API is unchanged apart from one new
config field (`n_jobs`).

**Results change relative to 0.6.0** in three cases, each because 0.6.0 was
computing the wrong thing:

- `algorithm='SVM'`: Step 3 was silently a no-op.
- Step 4 on data with mixed feature scales: SHAP was computed in the wrong space.
- `algorithm='XGB'` generally: XGBoost is now pinned to `tree_method='exact'`
  (see Parallelism below). Benchmark figures generated with 0.6.0 should be
  regenerated before being compared against 0.6.1 output.

### Known limitation, newly characterised: Step 1 can remove the signal

This behaviour is not new in 0.6.1, but it was not previously documented and it
affects how results should be read.

Step 1's correlation filter is not class-aware. Features that carry the same
outcome signal are mutually correlated *because* they are informative, and the
filter cannot tell that apart from redundant duplication. Where that induced
correlation exceeds `min_correlation_threshold`, Step 1 keeps one carrier and
discards the others.

Measured on synthetic data (n=500, p=200, 20 informative, rho=0.10): at
`signal_strength` 1.2 the mean correlation among informative features is 0.584
and all 20 survive; at 2.0 it is 0.797 and **one** survives, while all 180 noise
features are retained. The transition sits at the 0.70 threshold.

For real data this matters most for co-regulated modules whose members all track
the outcome: Step 1 reduces such a module to one arbitrary member, chosen by the
variance tie-break, and a different sample yields a different member. This is a
direct contributor to the low cross-fold stability reported in `BENCHMARKS.md`.

Mitigations available today: raise `min_correlation_threshold`, or set
`enable_step1=False`, when the features of interest are expected to be
co-regulated. `SelectOmicsConfig.suggest()` now reports the redundancy level of
your data, though it cannot distinguish signal-induced correlation from genuine
redundancy. A class-aware filter would fix this properly and has not been
implemented.

### New: data-shape guidance

- **`assess_feature_redundancy()`** measures the fraction of features carrying a
  correlated partner above the threshold, which bounds what Step 1 can remove.
  Exported from the package root.
- **`SelectOmicsConfig.suggest()`** reports the expected Step 1 trade-off for
  your data before you run, and **`SelectOmicsPipeline`** warns when Step 1 is
  enabled on data with nothing to remove. Both are advisory: neither modifies
  the configuration, because a run whose behaviour depends on an automatic
  switch is harder to reproduce than one that did what it was told.

### Parallelism

The pipeline now uses all available cores by default and falls back gracefully
when it cannot.

- **New `n_jobs` config field, defaulting to `-1`.** Every estimator takes its
  thread count from it. Previously every estimator was pinned to `n_jobs=1`.
- **Thread count provably does not change results.** XGBoost is pinned to
  `tree_method='exact'` to make that true: its default (`hist`) draws subsample
  rows in a thread-order-dependent way, so the same seed produced different
  trees at different thread counts. `exact` is also several times faster in the
  small-n regime this package targets, and parallelises far better (measured
  per fit at n=200/p=10000: 0.454 s exact against 4.630 s hist, scaling 11.5x
  across 24 cores against 2.6x).
- **Graceful degradation, matching the existing GPU behaviour.** A machine
  reporting one usable core runs serially. A matrix too small for threading to
  pay for itself is fitted on one thread, because a full pool measurably loses
  there. Neither is an error, and neither changes the result.
- **Benchmarks parallelised.** `benchmark_lgg.py` and `benchmark_synthetic.py`
  gained `--workers` and `--n-jobs`, a shared `_parallel` module, and
  command-line entry points. Work is scheduled at (method, fold) granularity for
  LGG and per scenario for synthetic, the latter chosen so the Kuncheva
  stability back-fill across seeds stays intact. If a process pool cannot be
  created the run completes sequentially with a warning rather than aborting.
- **GPU remains opt-in and is documented as results-affecting**, since the CUDA
  split-finding implementation differs from the CPU one.

### Correctness

- **Step 3 (Wrappers) did nothing for `algorithm='SVM'`.** RFECV was scored
  with `roc_auc_ovr`, which requires `predict_proba`. The SVM path uses a
  LinearSVC surrogate, which exposes only `decision_function`, so every RFECV
  fit raised `AttributeError`. The step caught the exception, counted a
  failure, and fell back to "all features selected", meaning the RFECV stage
  contributed nothing while appearing to run. The scorer is now chosen from
  what the estimator can actually serve.

- **Step 4 (Explainers) computed SHAP in the wrong feature space.**
  `TreeExplainer` was given the bare classifier, which is fitted on
  `MinMaxScaler` output, together with *unscaled* rows. Every sample took the
  wrong path through the trees. On data with mixed feature scales this changed
  both the attribution magnitudes and the resulting feature ranking. Rows are
  now pushed through the pipeline's transformers first.

- **Step 3 stability selection crashed in the p >> n regime.** The subsample
  was capped at `n_samples - 1`, leaving a one-sample complement whenever
  `min_features` (5% of p) approached the training-set size.
  `StratifiedShuffleSplit` then raised an uncaught `ValueError` that aborted
  the run. The complement is now kept at least `n_classes` samples wide, and
  an impossible class distribution degrades to a warning instead of a crash.

- **`excel_output_engine` and `hdf5_key` were ignored.** Every save site passed
  only `output_format`, discarding the other two output settings. All sites now
  route through `save_from_config`.

- **A global `--verbose` before the subcommand was discarded.** The subparsers
  redeclared the flag with a `store_true` default, which overwrote the value
  set on the main parser. `selectomics -v run ...` now behaves the same as
  `selectomics run ... -v`.

### Robustness

- An all-constant feature matrix now fails in Step 1 with a message naming the
  cause, instead of passing a zero-column matrix downstream to die inside
  sklearn several steps later.
- `SelectOmicsConfig.suggest` reads through `load_omics_data`, so it now
  accepts feather, HDF5, and JSON inputs rather than falling back to its
  "could not inspect data" branch for them.
- Bootstrap out-of-bag extraction in `FeatureSetValidator` no longer rebuilds a
  membership set once per sample (it was O(n^2) per iteration).

### Packaging

- Added the missing `LICENSE` file that the README and package metadata both
  referenced.
- Dependency floors corrected to match what the code actually uses:
  `xgboost>=2.0` (the `device=` argument), `scikit-learn>=1.4` (RandomForest
  accepting NaN, which `handle_nans` relies on), `python>=3.9`.
- `pyyaml` promoted from undeclared to a hard dependency: the CLI advertises
  YAML configs in its own help text.

### Housekeeping

- All source files are ASCII. Non-ASCII typography rendered as mojibake in
  Windows consoles, corrupting user-facing warning text.
- Removed the `tight_layout` calls on GridSpec figures, which warned on every
  save and discarded the configured panel spacing.
- Removed unused imports and dead locals across the package.
- Added `tests/test_regressions.py` covering each fix above.

## 0.6.0

Initial packaged release of the SelectOmics pipeline.
