# SelectOmics examples

Twelve notebooks on two TCGA cohorts from the MLOmics benchmark
(`MLOmics_Dataset_Reference.md`): glioblastoma (GBM, 244 samples, 5 subtypes)
and ovarian carcinoma (OV, 284 samples, 4 subtypes), each with four omic layers
and the merged matrix.

Every notebook runs the full pipeline with all evaluations switched on and
reports the panel the pipeline **recommends** rather than whichever step
happened to run last. All of them were executed end to end before shipping and
then cleared, so the cells you see are the cells that ran.

## Where to start

| Notebook | Runs in | Read it for |
|---|---|---|
| [GS-GBM/GBM_Quickstart.ipynb](GS-GBM/GBM_Quickstart.ipynb) | 4 min | The shortest path from a CSV to a panel |
| [GS-OV/OV_SelectOmics_Showcase.ipynb](GS-OV/OV_SelectOmics_Showcase.ipynb) | 20 min | Every evaluation module, baselines, step ablation, nested CV, resume |

Start with the quickstart. The showcase is the reference for what the package
reports and how to read it.

## One layer at a time

Ten notebooks with identical structure, so layers and cohorts can be compared
directly. Each: data, `suggest()` and config, run, each step against the Step 0
reference, the three validation protocols, the recommendation, the held-out
test on the recommended panel, provenance, every figure checked, and the saved
outputs.

| Notebook | Features | Trajectory | Recommended | Panel | Test AUC | Gap | Runs in |
|---|---:|---|---|---:|---:|---|---:|
| [GBM miRNA](GS-GBM/GBM_SelectOmics_miRNA.ipynb) | 286 | 286 > 157 > 110 > 8 | Step 3 | 8 | 0.819 | excellent | 26 min |
| [GBM Methy](GS-GBM/GBM_SelectOmics_Methy.ipynb) | 11 189 | 11189 > 2100 > 19 | Step 2 | 19 | 0.838 | good | 29 min |
| [GBM mRNA](GS-GBM/GBM_SelectOmics_mRNA.ipynb) | 11 343 | 11343 > 6967 > 18 | Step 2 | 18 | 0.922 | good | 21 min |
| [GBM CNV](GS-GBM/GBM_SelectOmics_CNV.ipynb) | 11 203 | 11203 > 16 | Step 1 | 16 | 0.783 | excellent | 18 min |
| [GBM merged](GS-GBM/GBM_SelectOmics.ipynb) | 34 021 | 34021 > 7116 > 25 | Step 2 | 25 | 0.926 | good | 89 min |
| [OV miRNA](GS-OV/OV_SelectOmics_miRNA.ipynb) | 286 | 286 > 126 > 115 > 10 | **Step 2** | 115 | 0.913 | excellent | 37 min |
| [OV Methy](GS-OV/OV_SelectOmics_Methy.ipynb) | 11 189 | 11189 > 1670 > 29 | Step 2 | 29 | 0.902 | excellent | 49 min |
| [OV mRNA](GS-OV/OV_SelectOmics_mRNA.ipynb) | 11 343 | 11343 > 6689 > 20 | Step 2 | 20 | 0.977 | excellent | 43 min |
| [OV CNV](GS-OV/OV_SelectOmics_CNV.ipynb) | 11 203 | 11203 > 442 > 70 > 6 | Step 3 | 6 | 0.584 | **poor** | 102 min |
| [OV merged](GS-OV/OV_SelectOmics.ipynb) | 34 021 | 34021 > 6717 > 15 | Step 2 | 15 | 0.977 | excellent | 82 min |

All ten run XGBoost at 5 consensus models, 10 tuning iterations and 30
bootstrap iterations, seed 42. Runtime is the run cell on a 24-core desktop.
A trajectory that stops before Step 3 means Step 3 skipped, which it does in
seven of these ten when Step 2 leaves fewer than 30 features.

## Five things these runs show

**The recommendation overrides the last step, and it matters.** On OV miRNA,
Step 3 cuts Step 2's 115 features to 10; the recommendation keeps the 115, and
the run logs that the delivered panel is not the last step's. This is the
reason to enable every step: a later step that prunes past the point where it
helps does not cost you the result.

**Step 3 mostly does not run at all, and that is deliberate.** It skips
whenever fewer than 30 features reach it, which happens in seven of these ten
notebooks, because Step 2 on wide data cuts well below that. OV Methy is the
instructive near-miss: Step 2 returns 29 features, one short of the gate, so
Step 3 skips. The same layer previously arrived at 35, Step 3 fired, and it cut
them to 3. Whether the most expensive step runs at all can turn on a single
feature, which is the strongest argument for reading the trajectory rather than
assuming the pipeline did what you configured.

**The held-out test catches what validation cannot.** OV CNV validated well
and scored 0.584 on unseen samples, a gap of 0.226 labelled `poor`. The panel
should not be used. The selection-optimism guard stays quiet there and is right
to: it compares the panel against the unselected Step 0 panel, and on that layer
Step 0 is optimistic too. Optimism shared by every panel is what the held-out
test exists to expose, and no training-split statistic can substitute for it.

**Merging concentrates on one layer.** Given all 34 000 features, both cohorts
select almost entirely mRNA: 24 of 25 features for GBM, 15 of 15 for OV. The
per-layer notebooks are the control that makes this readable, and OV mRNA alone
(0.977) scores exactly as well as OV merged (0.977) from 20 features rather
than 15 out of 34 021.

**Runtime is set by Step 3, not by the feature count.** The slowest notebook
is OV CNV at 102 minutes, because Step 2 hands Step 3 seventy features and
Step 3 refits once per feature it eliminates, per consensus model, per inner
fold, per probe of its retention search. The 34 000-feature merged matrices
finish sooner than that, because Step 2 cuts them below the gate and Step 3
never runs. Switching `algorithm` to `"RF"` cuts it further still: the
quickstart runs the GBM miRNA layer in 4 minutes.

## Data

**Nothing in this directory ships with the data.** `mlomics_data.py` downloads
it on first use from MLOmics, caches it under `.mlomics_cache/`, and every
notebook builds its own model-ready CSV from that cache. A first run needs a
network connection and pulls about 250 MB; later runs read the cache.

```python
from mlomics_data import build_input, build_merged

df = build_input("GBM", "miRNA")   # samples x features, plus 'Label'
df = build_merged("OV")            # all four layers joined, plus 'Label'
```

Three properties are worth knowing, because they are what make a re-run
trustworthy rather than merely convenient:

**The revision is pinned.** URLs name an immutable dataset commit, not `main`,
so an upstream release cannot silently change what these notebooks produce.
Bumping it is a deliberate edit to `mlomics_data.py`.

**Every file is checksummed.** A SHA-256 is recorded for each of the fifteen
files and verified after download and on every cache hit. A truncated or
substituted file is refused rather than analysed.

**The variant is part of the path, not a parameter.** MLOmics publishes
`Original`, `Aligned` and `Top` for each layer; these notebooks use `Aligned`,
which is sample-aligned across layers so the multi-omic examples line up.
The variants are not interchangeable: on GBM microRNA, `Original` shares only
143 of its features with `Aligned` and their values differ on the shared block
by up to 7.45. Pointing the URLs at the wrong variant would change the result
while still appearing to work.

Layer files arrive as features by samples, so `build_input` transposes them,
adds a layer suffix to each feature name, and attaches the labels by row order.
The suffix is what makes the per-layer breakdown in the merged notebooks
possible.

The data is published by the MLOmics authors under CC-BY-4.0. If you use it,
cite their paper as well as this package:

> MLOmics: Cancer Multi-Omics Database for Machine Learning.
> *Scientific Data* (2025). doi:10.1038/s41597-025-05235-x

## Reproducing

Run any notebook top to bottom. The first one to run downloads the data it
needs; the rest read the shared cache. Nothing depends on another notebook, and
each writes its own `results_<COHORT>_<LAYER>/` directory containing the
figures, the per-step tables, `selected_features.csv` (the last step) and
`recommended_features.csv` (the panel to use).
