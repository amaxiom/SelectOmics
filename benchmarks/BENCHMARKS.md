# SelectOmics - Benchmark Results

*Synthetic-benchmark suite, omics-regime edition (schema v2.0), SelectOmics 0.7.0.
Generated from `benchmark_synthetic.py` under the pre-registered protocol in
`BENCHMARK_SPEC`. Every number on this page comes from a single run of that
protocol: 13 scenarios, 5 seeds, 615 trials, 92.7 minutes wall clock.*

*The version stamp is the version the SYNTHETIC runs were made under, and is
left as recorded rather than advanced with the package. 0.7.1 changed no
selection behaviour: it bumped the version, removed dead code, and fixed the
error wrapping for corrupt Parquet and Feather files. The numbers below
therefore still describe the shipped selector.*

*Section 5, the real-data LGG results, was re-run on 2026-09-21 under 0.7.1
against the pinned MLOmics revision that the examples now fetch. The synthetic
sections were not re-run: they are generated from a known model rather than
from MLOmics, so the data move does not touch them.*

---

## TL;DR

**SelectOmics returns by far the shortest and cleanest feature panel of any
method tested, and it does so at competitive downstream AUC. It is the most
reproducible on synthetic data but not on real data. It is not the most
sensitive method, and it is slow.**

- **Ground-truth recovery: first, and not narrowly.** Mean F1 **0.804** against
  0.577 for the next-best (SHAP) and 0.548 for LASSO. Wins 6 of 10
  signal-bearing development scenarios.
- **Precision is the shape of the win: 0.970.** Against 0.555 (SHAP), 0.501
  (LASSO), 0.232 (ElasticNet). When SelectOmics returns a feature, it is almost
  always a real one.
- **Recall is the price: 0.730.** LASSO reaches 0.893 and ElasticNet 1.000. It
  finds fewer of the true features and almost nothing false.
- **Most reproducible on synthetic data: Kuncheva 0.690** against 0.420 for
  the next-best. The same features come back across seeds. **This does not
  transfer to real data**: on TCGA lower-grade glioma, ElasticNet and LASSO
  return more consistent panels on every layer (section 5.3). Quote it with
  that qualifier or not at all.
- **Most parsimonious quality method: 8.2 features** at a reduction ratio of
  0.991, against 33 (SHAP), 38 (LASSO) and 126 (ElasticNet).
- **It does not invent signal.** On a null control with no informative features
  at all, it returns **5.8** false positives against 53.6 (LASSO), 66.8 (SHAP),
  180.0 (ElasticNet) and 531.8 (RF importance).
- **It holds up on the held-out target regime.** On `omics_ultra_sparse`
  (n=100, p=10 000, 0.1 % signal), run once: **F1 0.826** against 0.617 (SHAP)
  and 0.194 (LASSO).
- **It loses where it says it will.** On the two adversarial scenarios, which
  are large-n, low-p, dense-signal regimes it does not target, ElasticNet takes
  both (1.000 against 0.600 and 0.858).
- **Recall can be bought back, at a price.** `allow_union_rung=True` moves
  recall 0.844 to 0.897 and precision 0.902 to 0.726, and is a loss in the
  sparse high-p regime (section 7.3).
- **The recommendation generalises.** Under nested CV on 3 real cohorts and
  3 synthetic sets, the recommended panel was 14x to 540x smaller than the
  full feature set at a held-out AUC within 0.011 of it (section 8).
- **It is slow: 96.9 s per trial** against 0.8 s (SHAP) and 3.5 s (LASSO).

> **One line:** choose SelectOmics when a short, defensible, reproducible panel
> is the deliverable. Do not choose it when sensitivity is the objective, when
> the data is large-n and low-dimensional, or when runtime is binding.

---

## 1. What was measured

Three pre-declared primary axes, locked in `BENCHMARK_SPEC` before the first
run. Changing them after seeing results would be p-hacking.

| Axis | Metric | Question |
|------|--------|----------|
| Ground-truth recovery | **F1** (plus precision / recall / Jaccard) | Did it find the features actually built into the data? |
| Stability | **Kuncheva Index**, in [-1, 1] | Does it pick the same features across seeds? |
| Downstream performance | **test AUC** | Do the selected features predict on a held-out split, using a fresh XGBoost model? |

Secondary: `n_selected`, `reduction_ratio`, `mean_feature_corr`, wall-clock
seconds. Statistics: **Friedman** omnibus, then pairwise **Wilcoxon
signed-rank** against SelectOmics, paired by scenario x seed, **Holm-Bonferroni**
corrected, with **Cliff's delta** for effect size. alpha = 0.05.

**Every method runs at its own library defaults.** SelectOmics is not tuned to
these scenarios, so nothing else is either. This is a comparison of defaults,
which is the comparison the page claims to make. Earlier editions gave some
comparators tuned operating points and are not comparable to this one.

### Scenarios

| Split | Scenario | n | p | informative | rho | Notes |
|-------|----------|---|---|-------------|-----|-------|
| dev | `omics_standard` | 100 | 2 000 | 10 (0.5 %) | 0.70 | balanced base case |
| dev | `omics_tiny_n` | 50 | 500 | 10 (2 %) | 0.70 | smallest sample |
| dev | `omics_high_dim` | 100 | 5 000 | 5 (0.1 %) | 0.80 | very sparse signal |
| dev | `omics_imbalanced` | 200 | 1 000 | 15 (1.5 %) | 0.70 | 85/15 class split |
| dev | `omics_genomics` | 200 | 10 000 | 10 (0.1 %) | 0.70 | genomics-scale p |
| dev | `adversarial_easy` | 500 | 200 | 20 (10 %) | 0.10 | **LASSO regime, SO not expected to win** |
| dev | `omics_lowcorr` | 100 | 2 000 | 10 (0.5 %) | 0.10 | `omics_standard` at low correlation |
| dev | `omics_imbalanced_std` | 100 | 2 000 | 10 (0.5 %) | 0.70 | `omics_standard` at 85/15 |
| dev | `null_control` | 100 | 2 000 | **0** | 0.70 | no signal; read on `n_selected` |
| dev | `omics_interaction` | 200 | 2 000 | 10 in 5 pairs | 0.70 | 60 % of signal in interactions |
| dev | `omics_multiclass` | 200 | 2 000 | 12 (0.6 %) | 0.70 | 4 balanced classes, macro-OVR |
| **holdout** | `omics_ultra_sparse` | 100 | 10 000 | 10 (0.1 %) | 0.75 | run once, SO target regime |
| **holdout** | `adversarial_large_n` | 800 | 300 | 30 (10 %) | 0.10 | run once, large-n LASSO regime |

The adversarial scenarios are included on purpose: they test whether the
benchmark is honest, not whether SelectOmics wins. Holdout scenarios were set
aside before any development result was inspected and run exactly once.

### Scenarios added in 0.7.0

Five were added **before** the 0.7.0 run and declared in `BENCHMARK_SPEC` at the
same time, from design gaps in the original six rather than from any result:

- **Five factors moved at once** between the omics scenarios and
  `adversarial_easy` (rho 0.70 against 0.10, n/p 0.05 against 2.5, block 20
  against 10, signal 1.0 against 2.0, density 0.5 % against 10 %), so
  "SelectOmics loses on adversarial" could not be attributed to any one of
  them. `omics_lowcorr` and `omics_imbalanced_std` each change exactly one
  factor from `omics_standard`. **This paid off: see §2.1.**
- **No null control.** Nothing measured what a method does when there is
  nothing to find.
- **Every scenario was linear-additive**, the model a sparse linear method
  assumes, so the suite could not show a method winning on non-linear
  structure. A *pure* interaction was tried first and rejected: if no single
  feature's distribution differs by class, no method that scores features one
  at a time can find it, and every method scored at or near zero regardless of
  n, p or signal strength. A scenario everything ties on measures nothing. The
  shipped scenario mixes a main effect with the interaction.
- **Every scenario was binary**, while the flagship examples are multiclass and
  the whole metric stack is macro one-vs-rest.

The original eight scenarios are byte-for-byte unchanged: their generated
matrices were checksummed before and after the generator was extended.

---

## 2. Method comparison, development scenarios

Means over the 10 signal-bearing development scenarios x 5 seeds.
`null_control` is excluded here because F1 is undefined with no true positives;
it has its own section (§2.5).

### 2.1 Ground-truth recovery (F1), *primary*

| Method | F1 | sd | Precision | Recall | Kuncheva | test AUC | n selected | s/trial |
|--------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| **SelectOmics** | **0.804** | 0.162 | **0.970** | 0.730 | **0.690** | 0.987 | **8.2** | 96.9 |
| SHAP | 0.577 | 0.228 | 0.555 | 0.816 | 0.391 | 0.991 | 33.2 | 0.8 |
| LASSO | 0.548 | 0.261 | 0.501 | 0.893 | 0.404 | 0.990 | 37.7 | 3.5 |
| RFECV | 0.423 | 0.202 | 0.428 | 0.747 | 0.420 | 0.990 | 36.6 | 11.1 |
| ElasticNet | 0.302 | 0.307 | 0.232 | **1.000** | 0.227 | 0.991 | 126.2 | 3.1 |
| PermImportance | 0.190 | 0.151 | 0.770 | 0.113 | 0.150 | 0.882 | 1.2 | 16.3 |
| RF_Importance | 0.152 | 0.287 | 0.127 | 0.989 | 0.119 | 0.990 | 459.8 | 0.4 |
| VarCorr_Baseline | 0.051 | 0.097 | 0.029 | 0.723 | 0.010 | 0.977 | 1333.9 | 0.9 |

Per scenario:

| Scenario | SelectOmics | SHAP | LASSO | RFECV | ElasticNet | winner |
|---|---:|---:|---:|---:|---:|---|
| `omics_genomics` | **0.979** | 0.653 | 0.128 | 0.182 | 0.068 | SelectOmics |
| `omics_interaction` | **0.957** | 0.634 | 0.393 | 0.627 | 0.130 | SelectOmics |
| `omics_high_dim` | **0.941** | 0.388 | 0.200 | 0.175 | 0.060 | SelectOmics |
| `omics_lowcorr` | **0.885** | 0.668 | 0.645 | 0.587 | 0.201 | SelectOmics |
| `omics_standard` | **0.858** | 0.717 | 0.664 | 0.560 | 0.180 | SelectOmics |
| `omics_multiclass` | **0.826** | 0.120 | 0.407 | 0.196 | 0.082 | SelectOmics |
| `omics_imbalanced` | 0.806 | 0.662 | **0.883** | 0.555 | 0.358 | LASSO |
| `omics_tiny_n` | 0.623 | 0.563 | 0.744 | 0.560 | **0.746** | ElasticNet |
| `adversarial_easy` | 0.600 | 0.727 | 0.888 | 0.336 | **1.000** | ElasticNet |
| `omics_imbalanced_std` | 0.566 | **0.636** | 0.531 | 0.453 | 0.193 | SHAP |

**Six wins of ten, and the margins are large where it wins.** On the three
sparsest, widest scenarios (`omics_genomics`, `omics_high_dim`,
`omics_ultra_sparse` in holdout) it is between 0.29 and 0.74 F1 ahead of every
comparator. That is the regime the package targets.

**The de-confounding worked, and gave a clean answer.** `omics_lowcorr` is
`omics_standard` with block correlation dropped from 0.70 to 0.10 and nothing
else changed. SelectOmics **wins** it (0.885). On `adversarial_easy`, which is
also rho = 0.10 but additionally large-n, low-p and dense-signal, it loses
badly (0.600 against 1.000). **Low correlation is therefore not what beats it.**
The n/p and signal-density regime is. That question was unanswerable with the
original six scenarios, where the two factors always moved together.

**`omics_multiclass` is the widest margin in the table**: 0.826 against SHAP's
0.120. Multiclass is the flagship use case (the bundled GBM and OV examples are
4- and 5-class) and had no synthetic coverage before 0.7.0.

### 2.2 Downstream AUC, *primary*

Every method except PermImportance lands between 0.977 and 0.991, with
SelectOmics at 0.987. **This axis does not separate the methods.** With signal
this concentrated, any reasonable subset predicts well, and a method returning
460 features scores the same as one returning 8. Read AUC as a floor check that
selection did not destroy predictive content, not as a ranking.

### 2.3 Stability (Kuncheva), *primary*

**SelectOmics 0.690**, then RFECV 0.420, LASSO 0.404, SHAP 0.391, ElasticNet
0.227. A 64 % lead over the next-best.

This is the axis that most directly supports the package's stated purpose. The
same panel comes back when the seed changes, which is what makes a result
citable rather than an artefact of one fit.

### 2.4 Parsimony

Mean 8.2 features at reduction ratio 0.991, the shortest of any method that is
not degenerate. PermImportance returns 1.2 features, but at F1 0.190: it is
short because it finds almost nothing, not because it is selective.

| Scenario | p | SelectOmics | SHAP | LASSO | ElasticNet | RF_Importance |
|---|---:|---:|---:|---:|---:|---:|
| `omics_genomics` | 10 000 | **10** | 21 | 147 | 283 | 1105 |
| `omics_high_dim` | 5 000 | **5** | 19 | 46 | 162 | 608 |
| `omics_multiclass` | 2 000 | **15** | 191 | 48 | 280 | 738 |
| `omics_standard` | 2 000 | **8** | 14 | 20 | 101 | 434 |

### 2.5 Null control: does the method invent signal?

`null_control` has **no informative features**. Every selected feature is a
false positive. F1 is undefined and reported as NaN by design; the metric here
is `n_selected`.

| Method | mean | min | max |
|---|---:|---:|---:|
| PermImportance | **0.0** | 0 | 0 |
| **SelectOmics** | **5.8** | 1 | 15 |
| LASSO | 53.6 | 45 | 59 |
| SHAP | 66.8 | 61 | 77 |
| RFECV | 68.0 | 20 | 100 |
| ElasticNet | 180.0 | 173 | 184 |
| RF_Importance | 531.8 | 519 | 556 |
| VarCorr_Baseline | 1000.0 | 1000 | 1000 |

Second only to PermImportance, which returns nothing anywhere. LASSO returns 54
features from pure noise; ElasticNet returns 180.

**One caveat worth stating.** The range is 1 to 15, and 15 is exactly
`min_features_floor`. On some seeds the floor forces a full panel out of noise.
The consensus ceiling usually caps the result well below the floor, but not
always, and a run on genuinely signal-free data can return a full-size panel
with an agreement label attached. Read `consensus_outcome` and the Step 0
reference AUC before trusting any panel.

### 2.6 Statistical tests

**Friedman** across the 8 core methods, paired by scenario x seed
(n = 50 blocks): chi-squared = 217.1, p = 2.8e-43.

Pairwise **Wilcoxon signed-rank** against SelectOmics, Holm-Bonferroni corrected:

| Comparison | p (Holm) | Cliff's delta | Verdict |
|---|---:|---:|---|
| vs VarCorr_Baseline | < 1e-5 | +1.000 | SelectOmics better |
| vs PermImportance | < 1e-5 | +0.986 | SelectOmics better |
| vs RFECV | < 1e-5 | +0.840 | SelectOmics better |
| vs RF_Importance | < 1e-5 | +0.817 | SelectOmics better |
| vs ElasticNet | < 1e-4 | +0.742 | SelectOmics better |
| vs SHAP | 0.00004 | +0.589 | SelectOmics better |
| vs LASSO | 0.00007 | +0.580 | SelectOmics better |

Every comparison survives correction, including against LASSO and SHAP.

---

## 3. Holdout evaluation (run once)

Set aside before any development result was inspected, run exactly once.

| Scenario | SelectOmics | SHAP | LASSO | RFECV | ElasticNet | RF_Imp |
|---|---:|---:|---:|---:|---:|---:|
| `omics_ultra_sparse` (n=100, p=10 000, 0.1 %) | **0.826** | 0.617 | 0.194 | 0.156 | 0.081 | 0.032 |
| `adversarial_large_n` (n=800, p=300, 10 %) | 0.858 | 0.863 | 0.912 | 0.438 | **1.000** | 1.000 |

Panel sizes: on `omics_ultra_sparse`, SelectOmics returns **7** features against
17 (SHAP), 94 (LASSO), 238 (ElasticNet) and 586 (RF importance).

**`omics_ultra_sparse` is the result that matters.** It is the target regime,
it was run once under a locked protocol, and SelectOmics wins it by 0.21 F1
over the next-best while returning a panel a third the size.

**`adversarial_large_n` behaves exactly as predicted.** The protocol declared in
advance that SelectOmics was not expected to win the adversarial scenarios, and
it did not. Predicting your own loss and then losing is what makes the win on
the other holdout credible.

---

## 4. The regime the package actually claims

Sections 2 and 3 average over everything, including two adversarial scenarios
the package explicitly does not target. Cutting to the claimed regime, small n
and large p, sharpens the result considerably.

### 4.1 Target regime (n <= 200, p >= 500, n/p <= 0.5)

10 scenarios, 400 trials.

| Method | F1 | Precision | Recall | Kuncheva | n selected |
|--------|-----:|-----:|-----:|-----:|-----:|
| **SelectOmics** | **0.827** | **0.968** | 0.759 | **0.719** | **8.1** |
| SHAP | 0.566 | 0.504 | 0.833 | 0.381 | 33.6 |
| LASSO | 0.479 | 0.412 | 0.913 | 0.336 | 45.5 |
| RFECV | 0.405 | 0.352 | 0.812 | 0.482 | 46.1 |
| ElasticNet | 0.210 | 0.136 | **1.000** | 0.131 | 148.0 |
| PermImportance | 0.182 | 0.750 | 0.107 | 0.142 | 1.0 |
| RF_Importance | 0.055 | 0.029 | 0.986 | 0.022 | 516.5 |

**+0.261 F1 over the next-best.** Every pairwise Wilcoxon is p < 1e-6 after Holm
correction, with Cliff's delta from +0.754 (SHAP) to +1.000 (RF importance).
A delta of +0.757 against LASSO means SelectOmics wins roughly 88 % of all
paired comparisons.

### 4.2 Very large p (p >= 5000), the sharpest cut

3 scenarios, 120 trials: `omics_high_dim`, `omics_genomics`,
`omics_ultra_sparse`.

| Method | F1 | Precision | Recall | Kuncheva | n selected |
|--------|-----:|-----:|-----:|-----:|-----:|
| **SelectOmics** | **0.915** | **0.970** | 0.880 | **0.838** | **7.4** |
| SHAP | 0.553 | 0.411 | 0.913 | 0.371 | 18.8 |
| PermImportance | 0.250 | 0.867 | 0.153 | 0.167 | 1.1 |
| LASSO | 0.174 | 0.096 | 1.000 | 0.098 | 95.6 |
| RFECV | 0.171 | 0.094 | 0.940 | 0.740 | 83.3 |
| ElasticNet | 0.070 | 0.036 | 1.000 | 0.037 | 227.6 |
| RF_Importance | 0.022 | 0.011 | 0.980 | 0.010 | 766.5 |

Per scenario:

| p | Scenario | SelectOmics | SHAP | LASSO | ElasticNet |
|---:|---|---:|---:|---:|---:|
| 5 000 | `omics_high_dim` | **0.941** | 0.388 | 0.200 | 0.060 |
| 10 000 | `omics_genomics` | **0.979** | 0.653 | 0.128 | 0.068 |
| 10 000 | `omics_ultra_sparse` *(holdout)* | **0.826** | 0.617 | 0.194 | 0.081 |

**The advantage grows with p**: 0.804 across all scenarios, 0.827 in the target
regime, 0.915 at p >= 5000. Five times LASSO's score at p = 10 000.

The precision column explains it. At p = 10 000, LASSO returns 96 features at
precision 0.096, so roughly 9 of its 96 are real. ElasticNet returns 228 at
precision 0.036. SelectOmics returns 7 at precision 0.970 with recall 0.880.

### 4.3 Outside the regime (n > p)

| Method | F1 | Precision | Recall |
|--------|-----:|-----:|-----:|
| ElasticNet | **1.000** | 1.000 | 1.000 |
| RF_Importance | 0.997 | 1.000 | 0.995 |
| LASSO | 0.900 | 1.000 | 0.820 |
| SHAP | 0.795 | 1.000 | 0.705 |
| **SelectOmics** | 0.729 | 1.000 | 0.592 |

Fifth of eight. Recall falls to 0.592 while precision stays at 1.000: the same
conservatism that wins at p = 10 000 costs it here, where there is plenty of
data and nothing to be cautious about.

**This is the shape of an honest result.** The package wins where it claims to,
by margins that grow with the thing it was built for, and loses where it says
it will.

---

## 5. Real data: TCGA lower-grade glioma

Synthetic data is generated from a known model, so ground-truth recovery is
measurable but the model is ours. The LGG benchmark removes that: real
expression, methylation, miRNA and CNV matrices, 10 folds
(5-fold x 2 repeats), same eight methods at library defaults. There is no
ground truth, so F1 against a true feature set is unavailable and the axes are
downstream AUC, stability across folds, and panel size.

400 trials: 5 layers x 8 methods x 10 folds, complete.

*Re-run on 2026-09-21 against the pinned MLOmics revision the examples now
fetch, replacing an earlier run on copies committed to this repository. The
labels are byte-identical and the shared values identical; upstream carries a
few features fewer per layer, mostly all-zero constants. Every conclusion below
survived the change. What moved is recorded in each subsection.*

### 5.1 Downstream AUC: nobody separates

| Layer | p | best method | best AUC | SelectOmics AUC | SO rank |
|---|---:|---|---:|---:|---:|
| mRNA | 11 343 | VarCorr_Baseline | 0.9885 | 0.9866 | 7 / 8 |
| Methy | 11 189 | ElasticNet | 0.5404 | 0.5081 | 7 / 8 |
| miRNA | 286 | SHAP | 0.9805 | 0.9714 | 7 / 8 |
| CNV | 11 203 | ElasticNet | 0.9834 | 0.9714 | 7 / 8 |
| merged | 34 021 | VarCorr_Baseline | 0.9891 | **0.9885** | **2 / 8** |

**SelectOmics does not lead on AUC on any layer.** But the spread is tiny: on
mRNA the top seven methods lie between 0.9866 and 0.9885, and `VarCorr_Baseline`
takes the layer while selecting **5 525** features. On this data, doing almost
nothing predicts as well as anything. AUC is a floor check here, not a ranking.

**Methylation defeats every method.** Best AUC 0.540, essentially chance. No
method finds signal, so the layer discriminates nothing, and the ranking within
it is noise.

*Changed on re-run:* the layer where SelectOmics places second moved from CNV
to merged, which is the layer the package is actually built for. On merged it
now concedes 0.0006 AUC to a baseline that selects 11 199 features.

### 5.2 Parsimony: the clear win

| Layer | p | SelectOmics | SHAP | LASSO | ElasticNet | RF_Imp | VarCorr |
|---|---:|---:|---:|---:|---:|---:|---:|
| mRNA | 11 343 | **18** | 108 | 408 | 718 | 894 | 5 525 |
| Methy | 11 189 | **16** | 561 | 233 | 567 | 2 288 | 5 407 |
| miRNA | 286 | **20** | 45 | 42 | 62 | 88 | 132 |
| CNV | 11 203 | **19** | 120 | 293 | 830 | 680 | 303 |
| merged | 34 021 | **18** | 90 | 559 | 1 049 | 615 | 11 199 |

**18 features from 34 021** on the integrated matrix, at AUC 0.9885 against the
best 0.9891. A 0.0006 AUC concession for a panel 31 times shorter than LASSO's.
This is the result the package exists to produce, and real data confirms it.

The shortest panel on every layer, with one caveat that matters for reading the
table: `PermImportance` returns a single feature on miRNA and CNV, at AUC 0.746
and 0.806. It is shorter and it is useless. SelectOmics is the most parsimonious
method that still works, which is the claim, and the AUC column is what
distinguishes the two cases.

### 5.3 Stability: the synthetic result does not transfer

| Layer | SelectOmics | ElasticNet | LASSO | VarCorr | RFECV |
|---|---:|---:|---:|---:|---:|
| mRNA | 0.552 | 0.640 | **0.646** | 0.714 | 0.265 |
| Methy | 0.072 | **0.509** | 0.439 | 0.959 | 0.048 |
| miRNA | 0.458 | **0.685** | 0.631 | 0.884 | 0.497 |
| CNV | 0.229 | **0.659** | 0.542 | 0.478 | 0.357 |
| merged | 0.541 | 0.651 | **0.638** | 0.959 | 0.720 |

**This contradicts the synthetic finding and is the most important caveat on
this page.** On synthetic data SelectOmics led stability outright (Kuncheva
0.690 against 0.420 next-best). On real data it is mid-table, beaten by
ElasticNet and LASSO on every layer.

`VarCorr_Baseline` scores highest of all, which shows what the metric rewards:
selecting nearly everything is perfectly reproducible and useless. Read the
stability column alongside `n_selected`, never alone.

The honest reading is that the synthetic stability advantage is a property of
data generated from a sparse known model, and does not survive contact with
real correlated omics. Parsimony does survive; stability does not.

*Unchanged on re-run:* still beaten by ElasticNet and LASSO on all five layers.
Methylation got worse, 0.118 to 0.072, on a layer where no method finds signal
and the features being compared are therefore arbitrary.

### 5.4 Cost on real data

Mean seconds per fold:

| Layer | SelectOmics | LASSO | ElasticNet | SHAP | RF_Imp |
|---|---:|---:|---:|---:|---:|
| mRNA | 1 340 | 75 | 79 | 7.9 | 0.2 |
| CNV | 1 360 | 69 | 54 | 15.0 | 0.2 |
| Methy | 2 473 | 71 | 67 | 26.8 | 0.3 |
| miRNA | 132 | 0.8 | 0.6 | 0.2 | 0.1 |
| merged | **6 828** | 239 | 247 | 31.8 | 0.4 |

Nearly two hours per fold on the 34k integrated matrix, 29x LASSO and 17 000x
RF importance.

*Changed on re-run, and not in the direction the rest of this page moved.* The
single layers roughly doubled: mRNA 574 to 1 340, Methy 1 034 to 2 473, CNV 910
to 1 360, while merged fell slightly. The example notebooks got markedly faster
on the same data change, so this is not a general speed-up or slow-down.

**The cause is not established.** Step 3's entry gate was the obvious
candidate, since a layer whose Step 2 output lands near 30 features can flip
between skipping the most expensive step and running it. The panel sizes argue
against it: they barely moved (17 to 18, 16 to 16, 24 to 20, 20 to 19, 18 to
18), which is not what a step appearing or disappearing looks like. The
benchmark does not log step activity, so it could not be checked directly.

The comparison also carries a confound the rest of this page does not: the
earlier timings were measured weeks earlier under an older package version,
whereas every other table here was re-run head to head. Treat the doubling as
measured but unexplained, and the ratios against other methods, which were
measured in the same run, as the reliable part.

---

## 6. Step decomposition and ablation

Cumulative prefixes, each starting from the raw feature matrix, plus isolated
single steps on the two ablation scenarios.

| Arm | F1 | Precision | Recall | Kuncheva | n | s/trial |
|---|---:|---:|---:|---:|---:|---:|
| **SelectOmics** (1+2+3) | **0.804** | **0.970** | 0.730 | **0.690** | 8.2 | 96.9 |
| `SO_S1->2` | 0.793 | 0.953 | 0.735 | 0.673 | 9.4 | 94.5 |
| `SO_S2_only` | 0.613 | 1.000 | 0.445 | 0.425 | 6.6 | 5.1 |
| `SO_S3_only` | 0.413 | 0.767 | 0.400 | 0.177 | 9.4 | 262.5 |
| `SO_S1_only` | 0.160 | 0.087 | **1.000** | 0.044 | 167.7 | 7.1 |

There is no `SO_S1->2->3` arm: with three selection steps the full pipeline
*is* the 1+2+3 prefix, and carrying both ran the identical configuration twice.

**Step 2 does the work.** Alone it reaches F1 0.613 at precision 1.000. Step 1
alone is close to useless (F1 0.160, precision 0.087) but is not pointless: it
cuts p by an order of magnitude so Step 2 runs on a tractable matrix, and its
recall of 1.000 means it discards essentially no true features.

**Step 3 fired in 1 of 55 runs.** It self-skips below
`MIN_FEATURES_FOR_RFECV = 30`, and Steps 1+2 deliver a mean of 8 features, so
the gate almost never opens. The +0.011 F1 that `SelectOmics` shows over
`SO_S1->2` comes essentially from that single firing, on `omics_multiclass`,
where Step 3 took 70 features to 9 and F1 from 0.293 to 0.857.

**So the shipped pipeline is effectively Step 1 to Step 2**, with Step 3 as a
standalone tool (`SO_S3_only` at F1 0.413, comfortably ahead of Step 1 alone)
and as a rarely-triggered safety net for unusually wide Step 2 output. That is
an honest description of what runs, and section 7 records the attempts to
change it.

---

## 7. Architecture experiments, all negative

Three attempts to make Step 3 a routine part of the pipeline rather than a rare
event. All are implemented and available; none is the default, because all made
results worse. They are recorded so the next person does not repeat them.

**The constraint behind all three** is the consensus ceiling. Steps 2 and 3
intersect two stages at every rung of the relaxation ladder: even at one vote a
feature must have been kept by some replicate in **every** stage. The ceiling is
therefore the size of that one-vote intersection, and `min_features_floor` is
explicitly clamped to it. Asking a step for more features cannot exceed what the
votes support.

### 7.1 Lowering the Step 3 entry gate (tried, reverted)

Measured on panels that are half noise, the gate looks like it should sit near
10 rather than 30: Step 3 beats a pass-through from `p_in` 10 upward, at
precision 1.000 throughout.

But that is not the panel Step 2 delivers. Step 2 output is nearly pure signal
(precision 0.970), so elimination has no noise to remove and can only cut true
features. Measured end to end at a gate of 10:

| Scenario | Step 2 to Step 3 | F1 | time |
|---|---|---|---|
| `omics_standard` | 10 to 1 | 1.000 to 0.182 | 0.1 s to 57 s |
| `omics_genomics` | 10 to 1 | 1.000 to 0.182 | 0.1 s to 62 s |
| `omics_genomics` | 10 to 2 | 1.000 to 0.333 | 0.1 s to 61 s |

Step 2 delivered perfect panels and Step 3 destroyed them. The gate stays at 30:
it is a proxy for "has this panel already been filtered to purity?", which is
not otherwise observable at runtime.

### 7.2 `step2_role="prefilter"` (available, not default)

Declares Step 2 a candidate generator, targeting twice the Step 3 gate and
dropping the agreement brake so Step 3 engages. 11 scenarios x 5 seeds:

| Arm | F1 | Precision | Step 3 fired | s/trial |
|---|---:|---:|---:|---:|
| `terminal` (default) | **0.849** | **0.902** | 0/50 | **145** |
| `prefilter` | 0.785 | 0.760 | 11/50 | 1 255 |

**When Step 3 did fire it worked exactly as intended**: across those 11 runs it
took Step 2 candidate sets from F1 0.336 to **0.818 at precision 0.983**. The
cascade is real.

**It just rarely fires.** Prefilter asked for 60 features and reached them in
**5 of 50** runs, because the ceiling ignores the floor. In the other 39 it
degenerated into "Step 2 with the brake off", which was better on some scenarios
(`omics_tiny_n` +0.111, `adversarial_easy` +0.168) and much worse on others
(`omics_high_dim` -0.379).

### 7.3 `allow_union_rung=True` (available, not default)

Adds a union ladder below the intersection ladder, raising the ceiling from
"kept by someone in every stage" to "kept by someone in any stage". This is the
only lever that moves the ceiling, and it works:

| Arm | after Step 2 | Step 3 fired | F1 | Precision | Recall | null-control FPs |
|---|---:|---:|---:|---:|---:|---:|
| `terminal` | 10.8 | 0/50 | **0.849** | **0.902** | 0.844 | 9.4 |
| `prefilter` | 24.7 | 11/50 | 0.785 | 0.760 | **0.901** | **4.2** |
| `terminal +union` | 17.4 | 5/50 | 0.755 | 0.726 | 0.897 | 17.0 |
| `prefilter +union` | **98.3** | **45/50** | 0.411 | 0.318 | 0.860 | 24.4 |

`prefilter +union` finally delivers the cascade: about 98 candidates, with
Step 3 firing in 45 of 50 runs. **And it is the worst arm in the table.** Step 3
prunes 98 to 46 at precision 0.318, so roughly 31 of the 46 are noise. Recall
stays high at 0.860, meaning the true features are in there, buried.

**Conclusion.** The intersection is not an implementation artefact to be
relaxed. It is what produces precision 0.970, and precision is what the entire
advantage rests on. Every route to a more permissive Step 2 costs more than it
returns as a *default*, because the candidate set a permissive Step 2 produces
is too dilute for Step 3 to repair.

#### As a deliberate sensitivity setting

`terminal +union` is the one arm worth offering to a user who would rather
carry a false feature than miss a true one. It buys 5 recall points for 18
precision points, four extra features and 1.8x the runtime. `prefilter` reaches
the same recall at nine times the runtime, and the two together are the worst
arm in the table, so the sensitivity setting is the union rung alone.

Where the trade lands is not uniform, and it runs against this package's own
target regime:

| Scenario | n / p | Precision | Recall |
|---|---|---|---|
| `adversarial_easy` | 500 / 200 | 1.000 -> 1.000 | 0.62 -> **0.80** |
| `omics_imbalanced` | 200 / 1 000 | 1.000 -> 0.912 | 0.80 -> **0.947** |
| `omics_lowcorr` | 100 / 2 000 | 0.980 -> 0.900 | 0.84 -> 0.90 |
| `omics_imbalanced_std` | 100 / 2 000 | 0.938 -> 0.878 | 0.72 -> 0.74 |
| `omics_standard` | 100 / 2 000 | 0.980 -> **0.652** | 0.92 -> 0.98 |
| `omics_interaction` | 200 / 2 000 | 0.870 -> **0.517** | 0.90 -> 0.92 |
| `omics_high_dim` | 100 / 5 000 | 0.752 -> **0.247** | 0.96 -> 1.00 |
| `omics_genomics` | 200 / 10 000 | 1.000 -> **0.655** | 1.00 -> 1.00 |

`omics_multiclass` and `omics_tiny_n` are unchanged by it.

It is a clear win at p in the hundreds and a clear loss at p in the thousands.
On `omics_genomics` it costs a third of the precision and buys nothing at all,
because recall was already 1.000. The pattern is that the union rung pays only
where recall is genuinely short, and recall is rarely short in the sparse
high-p regime: what is scarce there is precision.

For comparison, the sensitivity-first baselines in §2.1 reach recall 0.893
(LASSO) at precision 0.501 and recall 1.000 (ElasticNet) at precision 0.232, so
even in this mode the panel is markedly cleaner than theirs. Read that across
tables rather than as a paired result: the arms here run a plain
`SelectOmicsConfig` and score the default at recall 0.844, where §2.1's
SelectOmics arm scores 0.730.

**No preset ships for this.** A preset is regime advice, and the regime where
this setting wins is not the one the package targets; naming it would invite
its use exactly where it does the most damage. It is one documented flag.

---

## 8. Does the recommendation generalise?

The recommendation picks one step's panel from validation on the training
split, which is also where every panel's features were selected. So the choice
is scored on the data it was made from. `nested_recommendation.py` tests it
with 5-fold nested CV: each outer fold reruns selection, validation and the
recommendation on its training portion only, then scores every step's panel on
the held-out fold. 7 datasets, 35 outer folds.

| Dataset | Recommended | Panel | Held-out AUC | All features | Best step, hindsight | Selection optimism | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| GBM miRNA (325 f, 5 classes) | Step 3 x5 | 18.6 | 0.837 | 0.848 | 0.858 | +0.016 | |
| OV miRNA (321 f, 4 classes) | Step 3 x5 | 22.2 | 0.877 | 0.888 | 0.898 | +0.020 | |
| LGG miRNA (328 f, 3 classes) | Step 3 x5 | 22.6 | 0.979 | 0.976 | 0.982 | -0.002 | |
| `omics_multiclass`, prefilter | Step 3 x5 | 11.8 | 0.985 | 0.985 | 0.990 | +0.003 | 0.991 |
| `omics_standard`, prefilter, 15 models | Step 3 x5 | 5.8 | 0.996 | 0.996 | 1.000 | +0.003 | 0.731 |
| `omics_high_dim`, terminal | Step 2 x5 | 9.2 | 0.996 | 0.996 | 0.998 | +0.006 | 0.720 |
| `null_control`, terminal | Step 2 x5 | 20.8 | **0.460** | 0.568 | 0.612 | **+0.402** | |

*Selection optimism* is the recommended panel's validation score minus its
held-out AUC, less the same gap for Step 0, which involves no selection. *Best
step, hindsight* takes whichever step scored highest on each held-out fold, so
it is an upper bound rather than an achievable result. F1 is ground truth for
the recommended panel; every panel with all features scores 0.002 to 0.012.

**On every dataset with signal the recommendation holds up.** Its panel is 14x
to 540x smaller than the full feature set, and its held-out AUC is within
0.011 of using every feature on all six (-0.011, -0.011, +0.003, 0.000, 0.000,
0.000). Where ground truth is known, the recommended panel had the best F1 of
any step on every fold. Optimism is at most 0.02.

**Validation cannot rank the panels; parsimony decides.** Within a fold, the
validation scores of the four panels do not track their held-out AUCs: the mean
Spearman correlation per dataset runs from -0.86 to +0.16 and is never
meaningfully positive. The panels validate within about
0.01 of each other, and the one-standard-error rule takes the smallest. It took
the last step on all 35 folds, and on this evidence that was right. Choosing
instead from held-out scores with the same rule would have picked the same step
on every dataset with signal, so that was not built.

**The rescue case costs nothing measurable.** On `omics_standard` under
prefilter, Step 3 cut 61 to 94 candidates to 5 to 7 features at precision 1.000
on every fold, keeping only 58% of the true features. Held-out AUC did not move
(0.996), because the signal is redundant: any half of the true set predicts as
well as all of it. The panel is incomplete, not wrong.

**The null control is the failure, and it is now guarded.** With no signal to
find, Step 2 fitted noise, validated at 0.84, and scored 0.46 held out, an
optimism of 0.40 against 0.02 at most with real signal. The quality label had
passed such a panel as CAUTION. `build_recommendation` now compares the
recommended panel with the least-selected one, which carries no selection, and
lowers the label when the gain exceeds `MAX_PLAUSIBLE_SELECTION_GAIN = 0.05`
(largest gain with real signal: 0.021; smallest on the null: 0.229). It changes
the label and never the choice.

---

### Ranking the steps on held-out folds

`enable_holdout_ranking` replaces the ranking above with one computed on folds
carved from the training split: each fold reruns the selection steps on its own
training portion and scores every step's panel on the rows held out from it, so
no panel is scored on the samples that selected it. The test split is untouched
either way. Run on the same seven datasets, at the default five folds:

| Dataset | Held-out ranking | Training-split ranking | Nested-CV reference | Ground truth |
|---|---|---|---|---|
| GBM miRNA | Step 3, 29 f | Step 3, 29 f | Step 3 | |
| LGG miRNA | Step 3, 24 f | Step 3, 24 f | Step 3 | |
| OV miRNA | Step 3, 20 f | Step 3, 20 f | Step 3 | |
| `omics_high_dim` | Step 2, 9 f | Step 2, 9 f | Step 2 | Step 2, F1 0.72 |
| `omics_multiclass` | **Step 2, 122 f** | Step 3, 11 f | Step 3 | Step 3 F1 **0.99** vs Step 2 0.18 |
| `omics_standard` rescue | **Step 2, 77 f** | Step 3, 3 f | Step 3 | Step 3 F1 **0.73** vs Step 2 0.24 |
| `null_control` | **Step 1, 592 f** | Step 2, 17 f | Step 1 | no signal to find |

Panel sizes are what each run delivered. The ground-truth column is the F1 of
those steps' panels as measured in the nested-CV runs above, on the same
datasets at the same settings, since these runs do not recompute it.

**It never improved a choice.** The two rankings agree on four datasets. On the
three where they differ, the held-out choice is worse or pointless: two panels
an order of magnitude larger whose ground-truth F1 is a fifth to a third of the
panel it replaced, and 592 noise features on the null control. The AUC
differences driving those switches were 0.002 to 0.025, against fold-to-fold
standard deviations of 0.008 to 0.22. On `omics_multiclass` Step 3 missed the
tolerance band by 0.002.

**The cause is structural.** At n << p, AUC barely separates panel sizes: a
panel carrying the true features plus a hundred noise ones predicts about as
well as the true ones alone, so the ranking turns on noise, and noise favours
the larger panel because it is closer to the full reference model. Three folds
instead of five moved which dataset it got wrong (OV instead of
`omics_multiclass`) without changing the pattern.

**What it does fix is the label, and the guard fixes that more cheaply.** On the
null control the held-out ranking reports 0.64 and labels the panel NOT
RECOMMENDED, where validation reports 0.81. The selection-optimism guard reaches
a warning from the training split alone, at no extra runtime, which is why it is
the default protection and this flag is off.

**When to switch it on.** When predictive AUC on new samples is the deliverable
rather than a short panel, or when you want an honest step comparison for its
own sake. It costs one extra run of the selection steps per fold.

---

## 9. Computational cost

Mean seconds per synthetic trial:

| Method | s/trial | vs SHAP | F1 | F1 per second |
|---|---:|---:|---:|---:|
| RF_Importance | 0.4 | 0.5x | 0.152 | 0.401 |
| SHAP | 0.8 | 1x | 0.577 | 0.737 |
| ElasticNet | 3.1 | 4x | 0.302 | 0.098 |
| LASSO | 3.5 | 5x | 0.548 | 0.155 |
| RFECV | 11.1 | 14x | 0.423 | 0.038 |
| PermImportance | 16.3 | 21x | 0.190 | 0.012 |
| **SelectOmics** | **96.9** | **124x** | **0.804** | **0.008** |

**On F1 per second SelectOmics is last by two orders of magnitude.** 124 times
the runtime of SHAP for +0.227 F1. That is a bad trade if F1 is the only
objective. It is a reasonable one if the deliverable is a panel of 8 features at
precision 0.970 that will reproduce, which nothing else here provides at any
price. On a single dataset the absolute cost is minutes.

On real data the cost is worse: **7 034 s per fold** on the 34k integrated LGG
matrix, 29 times LASSO.

---

## 10. Honest summary: when to use SelectOmics

**Use it when:**

- p is large and n is small. The advantage grows with p: F1 0.827 in the target
  regime, **0.915 at p >= 5000**, against 0.553 for the next-best.
- A short panel is the deliverable. 8 features from 2 000 on synthetic data, 18
  from 34 021 on real integrated data, at competitive AUC.
- False positives are expensive. Precision 0.970, and 5.8 false positives from
  pure noise against LASSO 53.6 and ElasticNet 180.0.
- The panel must be defensible. Every run reports the agreement level it
  achieved rather than one you asked for.

**Do not use it when:**

- n > p, or the signal is dense. It ranks fifth of eight there; a default
  ElasticNet scores 1.000 against its 0.729.
- Sensitivity is the objective. Recall 0.730 against LASSO 0.893 and ElasticNet
  1.000. It will miss true features.
- Runtime is binding. 124x SHAP on synthetic data, 29x LASSO on real integrated
  data.
- Reproducibility on real data is the requirement. Section 5.3 is the caveat:
  the synthetic stability lead does **not** transfer to LGG, where ElasticNet
  and LASSO are more stable on every layer.

**The two results that most constrain the claims:**

1. **Stability does not transfer.** Kuncheva 0.690 and first place on synthetic
   data; mid-table on real data. Parsimony transfers, stability does not.
2. **Step 3 fires in 1 of 55 runs.** The shipped pipeline is effectively two
   steps. Section 7 documents three attempts to change that, all negative.

---

## 11. Reproducing

```bash
# from package/benchmarks/
python benchmark_synthetic.py --full --seeds 5 --workers 8 -o results_full_5seed
python benchmark_lgg.py --layers mRNA Methy miRNA CNV merged --folds 5 --repeats 2 --workers 6
python role_comparison.py --workers 6
python nested_recommendation.py --dataset gbm_mirna   # one per dataset, then:
python nested_recommendation.py --report
python holdout_ranking_check.py --dataset gbm_mirna   # one per dataset, then:
python holdout_ranking_check.py --report
```

Both harnesses checkpoint per trial and resume: re-running skips work already
recorded, so an interrupted run costs at most one trial. A failing trial is
logged and skipped rather than ending the run; pass `on_error="raise"` to
`run_tasks` if you would rather it stop.

Raw outputs: `results_full_5seed/benchmark_raw.csv` (615 rows),
`results_lgg/benchmark_lgg_10fold.csv` (400 rows),
`results_role_comparison.csv` (220 rows).
