# SelectOmics

[![Status](https://img.shields.io/badge/status-pre--release-orange)](#installation)
[![Coverage](https://img.shields.io/badge/branch%20coverage-96%25-brightgreen)](#testing)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://github.com/amaxiom/SelectOmics)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

**Finds the right features in small-n, large-p data, not just fewer of them.**

**A reproducible, leakage-proof feature selector for omics classification.**

> **Pre-release.** Version 0.7.1 is complete and tested (807 tests, 96%
> branch coverage) but has not yet been used by anyone outside its author. If
> you are reading this because you were asked to try it, that is what you are
> being asked to try: whether it works on your data, and whether the
> documentation tells you what you need. Please open an issue for anything that
> surprises you, including anything the guides fail to explain. There are
> templates under **Issues**, and `CONTRIBUTING.md` says what makes a report easy
> to act on.

LASSO will give you a short list. On our benchmark it returns 38 features, and
about half of them are noise. SelectOmics returns 8, and 97% of them are real.

It gets there by making every feature survive three independent selection
criteria and a vote across many independently seeded models, then re-validating
each step's panel and recommending the one to use. The held-out test set is
touched once, at the end, on that panel, so the number you report is one that
nothing in the pipeline has seen.

---

## Is it right for your data?

**Use it when a short, defensible panel is the deliverable.** Measured on a
pre-registered synthetic benchmark and on real TCGA data, against LASSO,
ElasticNet, RFECV, SHAP, RF importance and permutation importance at their
library defaults:

- **Precision 0.970**, against 0.501 for LASSO, 0.555 for SHAP and 0.232 for
  ElasticNet. When it returns a feature, that feature is almost always real.
- **It does not invent signal.** On a null control with no informative features
  at all, it returns 5.8 false positives. LASSO returns 53.6, ElasticNet 180.
- **The lead grows with p**, which is the regime it is built for: F1 0.915 at
  p of 5 000 or more, against 0.553 for the next-best method.
- **On real data it concedes almost nothing.** 18 features from 34 021 on TCGA
  lower-grade glioma, at AUC 0.9885 against the best method's 0.9891: a panel
  31 times shorter than LASSO's for 0.0006 AUC.

**Do not use it when:**

- **Sensitivity is the objective.** Recall is 0.730 against LASSO's 0.893. It
  will miss true features. `allow_union_rung=True` buys recall back and pays for
  it in precision.
- **Runtime is binding.** About 100 s per trial against seconds for LASSO, and
  close to two hours per cross-validation fold on 34 000 features.
- **n is large and the signal dense.** There a default ElasticNet beats it.

**And do not expect** it to be the most stable method on real data. It leads on
stability on synthetic data, but on real TCGA data ElasticNet and LASSO return
more consistent panels on every layer measured. Its advantage that survives
contact with real data is parsimony, not stability.

Every figure above comes from [`benchmarks/BENCHMARKS.md`](benchmarks/BENCHMARKS.md),
which also reports where the method loses.

---

## How it works

A reference evaluation fixes the baseline, three selection steps run in order,
and a mandatory evaluation decides which of their panels to deliver.

- **Step 0, Reference.** No selection. One tuned model cross-validated on every
  feature, the baseline every later step is compared against.
- **Step 1, Data cleaning.** Removes constant, low-variance and redundant
  features, by consensus between a variance filter and a class-aware
  correlation filter.
- **Step 2, Regularisation.** Parallel L1-style and L2-style models; a feature
  must survive both.
- **Step 3, Wrappers.** Recursive feature elimination intersected with
  stability selection. Skipped when fewer than 30 features reach it, which on
  wide data is most of the time.
- **Pipeline evaluation.** Every step's panel is re-validated three ways on the
  training split and one is recommended. It can be an earlier step than the
  last: when a later step prunes past the point where it helped, the
  recommendation keeps the panel before it.

Within each step no single model decides anything. Every step starts by
requiring unanimity across its consensus models and relaxes only as far as it
must, then reports the agreement it actually achieved. You specify the panel
size you need; the pipeline tells you how strong a claim that panel carries.

Four classifiers are supported throughout: `XGB` (the default), `RF`, `LR` and
`SVM`. Re-running with a different one is a cheap robustness check.

---

## Installation

Requires Python 3.9 or later.

```bash
pip install git+https://github.com/amaxiom/SelectOmics.git
```

To work from a clone, which is what you want if you plan to run the examples or
the test suite:

```bash
git clone https://github.com/amaxiom/SelectOmics.git
cd SelectOmics
pip install -e ".[dev,viz]"
```

`[viz]` adds seaborn for the confusion-matrix heatmaps, `[io]` adds PyTables and
fastparquet for HDF5 input, and `[dev]` adds pytest and pytest-cov. Extras also
work without a clone:

```bash
pip install "SelectOmics[viz] @ git+https://github.com/amaxiom/SelectOmics.git"
```

The examples are Jupyter notebooks and fetch their data on first run, so they
need Jupyter and a network connection. The package itself needs neither.

---

## Quick start

```python
import SelectOmics as selectomics

selectomics.enable_logging()          # progress is reported through logging

pipeline = selectomics.SelectOmicsPipeline({
    "data_path":     "data/omics.csv",
    "target_column": "Class",
})
results = pipeline.run()

# The panel to use. Usually the last step's, and an earlier step's when a
# later one pruned past the point where it helped.
features = pipeline.get_recommended_features()
print(f"Recommended {len(features)} features.")

pipeline.save_results()               # config, both panels, metrics, figures
```

Use `get_recommended_features()`, not `get_selected_features()`. The second
returns whichever step ran last, which is right most of the time and wrong
exactly when it matters.

**Your data** is one table: samples as rows, features as columns, one column of
class labels. CSV, TSV, Excel, Parquet, HDF5, Feather and JSON are read, detected
from the extension. Non-numeric columns such as sample IDs are dropped, but a
*numeric* ID column would be treated as a feature, so remove it first. The target
must be categorical; SelectOmics classifies, and a continuous target is refused
with a message saying so.

**Expect warnings.** Before selection starts, the pipeline checks sample size
against four thresholds from the literature, and data in the regime this package
targets will trip at least one. They qualify the result rather than invalidate
it, and the run continues: they mean confidence intervals will be wide, not that
something is broken.

If you are unsure what settings suit your data, let it look first:

```python
cfg = selectomics.SelectOmicsConfig.suggest("data/omics.csv", "Class")
```

The same run from a terminal:

```bash
selectomics run data/omics.csv --target Class --output results/
```

---

## Examples

Twelve notebooks on two TCGA cohorts, glioblastoma and ovarian carcinoma, with
four omic layers each plus the merged matrix, in
[`examples/`](examples/README.md). Every one runs the full pipeline with all
evaluations on and delivers the recommended panel. All were executed end to end
before release, then cleared.

| Start here | Runs in | Shows |
|---|---|---|
| `examples/GS-GBM/GBM_Quickstart.ipynb` | 4 min | A CSV to a panel, shortest path |
| `examples/GS-OV/OV_SelectOmics_Showcase.ipynb` | 20 min | Every evaluation module, baselines, ablation, nested CV |

The data is not committed. `examples/mlomics_data.py` downloads it on first use
from a pinned, checksummed [MLOmics](https://huggingface.co/datasets/AIBIC/MLOmics)
revision (CC-BY-4.0), so a clone is small and every published number is
reproducible from the public source.

Two lessons worth reading before you run your own data. On OV miRNA, Step 3 cuts
Step 2's 115 features to 10 and the recommendation keeps the 115: enabling every
step costs you nothing, because the evaluation catches a step that over-prunes.
And on OV copy-number, the panel validates well and scores 0.584 on held-out
samples, a gap labelled `poor`: read the gap, not only the AUC.

---

## Configuration

Parameters go in a plain dictionary or a `SelectOmicsConfig`. Only
`data_path` and `target_column` are required. The ones you are most likely to
change:

| Parameter | Default | What it controls |
|---|---|---|
| `algorithm` | `"XGB"` | `LR`, `XGB`, `RF` or `SVM`. LR and SVM switch to XGB automatically when the data has missing values |
| `min_features_floor` | `15` | The smallest panel any step may return. In practice the control on panel size |
| `min_consensus` | `0.4` | How thin model agreement may get in pursuit of that floor |
| `n_consensus_models` | `10` | Models per consensus vote. More is more stable and slower |
| `test_size` | `0.25` | The held-out fraction, touched once at the end |
| `allow_union_rung` | `False` | Trade precision for recall. Read the user guide first |
| `enable_nested_cv` | `False` | Estimate how the whole procedure generalises, step choice included |
| `random_seed` | `42` | Every stochastic operation derives from it |

Start from a preset and override what you need:

```python
cfg = selectomics.SelectOmicsConfig.standard("data/omics.csv", "Class", algorithm="RF")
```

| Preset | Intended use |
|---|---|
| `quick` | Rapid iteration; Step 3 off |
| `standard` | Most analyses |
| `thorough` | Publication runs, where runtime is no object |
| `omics` | Small-n, large-p. Its tuned values are provisional; see the user guide |

The [user guide](docs/USER_GUIDE.md) documents every parameter, with the
measurements behind each default.

---

## Running step by step

Every step is a public method returning `self`:

```python
pipeline = selectomics.SelectOmicsPipeline(config)
(pipeline
    .load_data()
    .run_step0_reference()
    .run_step1_cleaning()
    .run_step2_regularization()
    .run_step3_wrapper()
    .validate_features())

after_step2 = pipeline.get_selected_features("step2")
```

`run()` is not a way to finish a partly stepped pipeline: without
`resume=True` it reloads the data and runs every step again from the start.

---

## Reproducibility

Every stochastic operation derives from `random_seed`, no global random state is
modified, and **thread count does not change results**: a run on every core
selects exactly the features a serial run selects. `use_gpu=True` is the one
exception, because CUDA finds splits differently, so leave it off for runs you
will compare against CPU-generated numbers.

One sensitivity is worth knowing. Hyperparameters are tuned on the full feature
matrix before Step 1 removes anything, so columns that Step 1 later discards,
even all-zero ones, can change which model is tuned and therefore which panel is
delivered. Two runs on the same assay, one with empty probes dropped beforehand,
can return different panels. Filter your input the same way each time.

---

## Testing

```bash
pip install -e ".[dev]"
pytest
```

807 tests at 96% branch coverage. The full suite fits real models on
synthetic data, so allow several minutes.

---

## Documentation

| Document | Covers |
|---|---|
| [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) | Installation, configuration, every parameter, the CLI |
| [`user_guide/SelectOmics_User_Guide.pdf`](user_guide/SelectOmics_User_Guide.pdf) | The same ground in one typeset document, with figures |
| [`docs/METHODS_GUIDE.md`](docs/METHODS_GUIDE.md) | The algorithm behind each step, with source references |
| [`docs/INTERPRETATION_GUIDE.md`](docs/INTERPRETATION_GUIDE.md) | Every output file, figure panel and metric, read line by line |
| [`docs/IMPLEMENTATION_GUIDE.md`](docs/IMPLEMENTATION_GUIDE.md) | Module layout and internals, for contributors |
| [`benchmarks/BENCHMARKS.md`](benchmarks/BENCHMARKS.md) | What was measured, against what, and where it loses |
| [`examples/README.md`](examples/README.md) | Twelve executed notebooks on two TCGA cohorts |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | How to report a problem usefully, and the house rules |
| [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) | What is expected of everyone taking part, and how to report a problem |
| [`CITATION.cff`](CITATION.cff) | Citation metadata, read by GitHub and by reference managers |
| [`CHANGELOG.md`](CHANGELOG.md) | Release history |

---

## Requirements

- Python 3.9 or later
- numpy, pandas, scikit-learn 1.4 or later, scipy, xgboost 2.0 or later,
  matplotlib, openpyxl, pyarrow, pyyaml
- Optional: seaborn (`[viz]`), PyTables and fastparquet (`[io]`)

---

## Repository layout

```
SelectOmics/
├── SelectOmics/      # the package
├── benchmarks/       # benchmark harnesses, their results, and BENCHMARKS.md
├── docs/             # user, methods, interpretation and implementation guides
├── examples/         # twelve executed notebooks, and the MLOmics data fetcher
├── tests/            # the test suite
├── user_guide/       # the typeset user guide, PDF and LaTeX source
├── pyproject.toml
├── CHANGELOG.md
└── LICENSE
```

---

## Citation

If you use SelectOmics in published work, please cite the software. There is no
paper yet; this section will be updated when there is one.

```bibtex
@software{barnard2026selectomics,
  author  = {Barnard, Amanda S.},
  title   = {{SelectOmics}: reproducible, leakage-proof feature selection for
             small-n, large-p classification},
  year    = {2026},
  version = {0.7.1},
  url     = {https://github.com/amaxiom/SelectOmics},
  note    = {Python package, Apache-2.0 licence}
}
```

`CITATION.cff` carries the same metadata in machine-readable form, which is what
GitHub reads to offer **Cite this repository** in the sidebar.

If you run the examples, please also cite the data:

> MLOmics: Cancer Multi-Omics Database for Machine Learning. *Scientific Data*
> (2025). doi:10.1038/s41597-025-05235-x

---

## License

Apache-2.0. See [`LICENSE`](LICENSE).
