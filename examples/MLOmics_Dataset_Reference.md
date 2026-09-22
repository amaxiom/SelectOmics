# MLOmics Dataset Reference

## Overview

**MLOmics** (Cancer Multi-Omics Database for Machine Learning) is a curated benchmark
resource providing harmonised multi-omics data for 32 TCGA cancer types, designed
specifically to support reproducible machine-learning research in oncology.

| Field | Detail |
|---|---|
| Full name | Cancer Multi-Omics Database for Machine Learning |
| Publication | *Scientific Data* 2025, DOI: [10.1038/s41597-025-05235-x](https://doi.org/10.1038/s41597-025-05235-x) |
| GitHub | <https://github.com/chenzRG/Cancer-Multi-Omics-Benchmark> |
| HuggingFace | <https://huggingface.co/datasets/AIBIC/MLOmics> |
| Samples | 8,314 across 32 cancer types |
| Omic layers | mRNA expression, DNA methylation (Methy), miRNA expression, copy-number variation (CNV) |

---

## Raw File Format

Each omic layer is stored as a separate CSV with the following orientation:

```
             TCGA.02.0001.01   TCGA.02.0003.01   ...   (patients as columns)
hsa.mir.232       0.000000         0.000000
hsa.mir.551b     -1.347401        -0.205540
...
(features as rows)
```

**Rows = features** (gene names, miRNA IDs, CpG probe IDs, etc.)  
**Columns = patients** (TCGA sample barcodes)

The companion label file is a plain `(n_samples × 1)` CSV whose rows align
positionally with the patient columns of the omic files:

```
Label
0
1
2
...
```

### Preparing data for machine learning

Before modelling, each omic file must be **transposed** so that rows become
patients and columns become features, the standard layout expected by scikit-learn
and SelectOmics. The notebooks in this directory perform this step automatically
in the *Data Preparation* cell:

```python
df = pd.read_csv(path, index_col=0).T   # T flips features×patients → patients×features
df.columns = [f"{c}_{layer}" for c in df.columns]   # add layer suffix, e.g. APOD_mRNA
```

All four layers are then merged horizontally (inner join on patient index) and the
label vector is attached as a final column.

---

## Gold-Standard (GS) Subtype Datasets

The GS datasets provide **biologically validated subtype labels** derived from
landmark TCGA classification studies. Numeric labels are assigned as shown below.

Source file in repository: `GS-Subtype labels.xlsx`

---

### GBM: Glioblastoma Multiforme
**244 samples · 5 classes · 34,068 features (all layers)**

Subtypes follow the **Verhaak et al. (2010)** transcriptional classification
(*Cancer Cell*, 17:98–110).

| Label | Subtype | Notes |
|---|---|---|
| 0 | Classical | High *EGFR* amplification; *CDKN2A* deletion |
| 1 | Proneural | *PDGFRA* amplification; *IDH1* mutation; younger patients |
| 2 | Mesenchymal | *NF1* loss; high immune infiltration; poorest prognosis |
| 3 | G-CIMP | Glioma CpG Island Methylator Phenotype; *IDH1*-mutant; best prognosis |
| 4 | Neural | Normal neural gene expression; less distinct |

Class distribution: 0=61, 1=46, 2=74, 3=20 (**smallest**), 4=43

**Modelling note:** Use `cv_splits=5` to avoid empty validation folds for class 3
(only ~15 training samples after an 75/25 split).

---

### LGG: Lower-Grade Glioma
**247 samples · 3 classes · 34,069 features (all layers)**

Subtypes defined by **IDH mutation status** and **1p/19q co-deletion** -
the two most clinically actionable molecular features in LGG (TCGA, *NEJM* 2015).

| Label | Subtype | Histology | Prognosis |
|---|---|---|---|
| 0 | IDHmut-non-codel | Astrocytoma | Intermediate |
| 1 | IDHwt | Most aggressive LGG; similar to GBM | Poor |
| 2 | IDHmut-codel | Oligodendroglioma | Best |

Class distribution: 0=76, 1=125 (**largest**), 2=46 (**smallest**)

---

### OV: Ovarian Serous Cystadenocarcinoma
**284 samples · 4 classes · 34,061 features (all layers)**

Subtypes from the **TCGA (2011)** ovarian cancer study (*Nature* 474:609–615),
defined by gene-expression clustering.

| Label | Subtype | Characteristics |
|---|---|---|
| 0 | Immunoreactive | High T-cell infiltration; *CXCL10/11* signature; best survival |
| 1 | Differentiated | Low-grade markers; *MUC16* (CA-125) high |
| 2 | Proliferative | High proliferation signatures (*MCM2*, *PCNA*) |
| 3 | Mesenchymal | Stromal/desmoplastic; worst survival |

Class distribution: 0=64, 1=79, 2=67, 3=74 (well balanced, ~1.2:1 ratio)

---

### BRCA: Breast Invasive Carcinoma *(for reference)*
**5 classes**

| Label | Subtype |
|---|---|
| 0 | LumA |
| 1 | Her2 |
| 2 | LumB |
| 3 | Normal |
| 4 | Basal |

---

### COAD: Colon Adenocarcinoma *(for reference)*
**3 classes**

| Label | Subtype |
|---|---|
| 0 | GI.CIN |
| 1 | GI.MSI |
| 2 | GI.GS |

---

## Feature Counts per Omic Layer

Counts are after MLOmics alignment (consistent across all GS datasets).

| Layer | Features | Description |
|---|---|---|
| mRNA | ~11,343–11,346 | RNA-seq gene expression (log-normalised) |
| Methy | ~11,189–11,192 | Illumina 450K DNA methylation β-values |
| miRNA | 286–325 | miRNA expression |
| CNV | ~11,203–11,205 | GISTIC2 copy-number segment values |
| **All (merged)** | **~34,061–34,068** | Horizontal concatenation of all four layers |

**Missing data:** The Methy layer contains ~0.17–0.18 % NaN values in all three
GS datasets. Use an algorithm that handles NaN natively (e.g. XGBoost) or
impute before training.

---

## Selecting a Single Omic Layer

Each notebook includes a `SELECTED_LAYER` variable immediately after the data
preparation cell. Set it to filter to one layer before running the pipeline:

```python
SELECTED_LAYER = None      # use all layers (default)
# SELECTED_LAYER = 'mRNA'
# SELECTED_LAYER = 'Methy'
# SELECTED_LAYER = 'miRNA'
# SELECTED_LAYER = 'CNV'
```

When `SELECTED_LAYER` is set, the notebook saves a filtered CSV and redirects
`MERGED_PATH` / `DATA_PATH` so that all downstream cells (config, pipeline,
results) operate on the single-layer data without any further changes.

---

## Citation

If you use MLOmics data in your work, please cite:

> Chen Z, *et al.* (2025). Cancer Multi-Omics Benchmark for Machine Learning.
> *Scientific Data*, **12**, XXXX. https://doi.org/10.1038/s41597-025-05235-x

---

*Last updated: 2026-05-18*
