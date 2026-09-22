"""
SelectOmics -- Real-World Benchmark on TCGA Lower-Grade Glioma (LGG)
==================================================================

A *real-data* companion to ``benchmark_synthetic.py``.  Where the synthetic
suite measures ground-truth recovery (it knows which features are informative),
real omics data has **no ground-truth feature set** -- so the evaluation axes
change:

    1. Downstream predictive performance  -- macro one-vs-rest AUC, macro-F1,
       balanced accuracy of a fixed XGBoost classifier trained on the selected
       features (selection refit INSIDE each CV fold -> no leakage).
    2. Stability                          -- Kuncheva Index across CV folds
       (the headline differentiator: do we pick the same features each split?).
    3. Parsimony / redundancy             -- n_selected, reduction ratio, mean
       pairwise |r| among selected features.
    4. (Notebook) biological soft-truth   -- do selected genes recover known
       glioma drivers (IDH1/2, ATRX, TP53, CIC, FUBP1, EGFR, CDKN2A, 1p/19q)?

Dataset: ``LGG_merged.csv`` -- 247 samples x 34,069 features across four omic
layers (mRNA / Methy / miRNA / CNV), 3-class molecular subtype (0/1/2).
Each layer is benchmarked separately (columns carry a ``_<layer>`` suffix).

Design notes
------------
* **Multiclass throughout.**  Competitor scoring is macro-OVR (the synthetic
  harness is binary); LASSO/ElasticNet use multinomial logistic regression.
* **Leakage-free.**  Per fold we fit median imputation on the *train* split
  only, run selection on train only, and score on the held-out test split.
* **Fair input.**  Every method receives the same per-fold median-imputed
  matrix (Methy has ~0.17% NaN; sklearn methods cannot ingest NaN, and imputing
  for all keeps the comparison apples-to-apples).  Features are *not*
  re-standardised -- the layers are pre-normalised upstream and forcing unit
  variance would neutralise the variance-based steps (SelectOmics Step 1,
  VarCorr baseline).
* **Checkpoint-safe.**  Each (layer, method, fold) row is appended immediately;
  re-running resumes from the saved CSV.
* Reuses the validated statistics from ``benchmark_synthetic`` by mapping
  ``scenario -> layer`` and ``seed -> fold index``.
"""

from __future__ import annotations

import os
import sys
import time
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

warnings.filterwarnings("ignore")

# Make sibling modules importable when run from anywhere.
_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse validated, label-agnostic helpers from the synthetic harness.
from benchmark_synthetic import (          # noqa: E402,F401
    kuncheva_stability,
    friedman_test,          # re-exported: the notebook imports it from here
    pairwise_vs_reference,  # re-exported: the notebook imports it from here
)
from _parallel import (                    # noqa: E402
    describe_plan,
    plan_parallelism,
    run_tasks,
    worker_context,
)

try:                                        # optional deps
    import shap as _shap
    _SHAP_AVAILABLE = True
except Exception:
    _SHAP_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except Exception:
    _XGB_AVAILABLE = False

try:
    from SelectOmics import SelectOmicsConfig, SelectOmicsPipeline
    _SO_AVAILABLE = True
except Exception:
    _SO_AVAILABLE = False

import logging
logger = logging.getLogger("benchmark_lgg")

LAYERS: List[str] = ["mRNA", "Methy", "miRNA", "CNV"]
# Tokens that mean "all layers together in one integrated matrix" (34k features),
# as opposed to LAYERS which benchmarks the four layers separately.
MERGED_TOKENS = {"merged", "all", "all_layers", "combined", "integrated"}
MERGED_LABEL = "merged"
TARGET_COL = "Label"


def _is_merged(layer) -> bool:
    return layer is None or (isinstance(layer, str) and layer.lower() in MERGED_TOKENS)


# ==============================================================================
# 1.  DATA
# ==============================================================================

def load_merged(path: str | Path = None) -> pd.DataFrame:
    """Load the merged LGG multi-omics matrix (samples x features + Label)."""
    path = Path(path) if path else _HERE / "LGG_merged.csv"
    if not path.exists():
        # The MLOmics data is not committed. Build the merged matrix from the
        # pinned dataset revision, the same source and revision the example
        # notebooks use, so the benchmark and the examples cannot drift apart.
        _examples = _HERE.parent / "examples"
        if str(_examples) not in sys.path:
            sys.path.insert(0, str(_examples))
        try:
            from mlomics_data import merged_csv
        except ImportError as exc:
            raise FileNotFoundError(
                f"{path} not found, and examples/mlomics_data.py is not "
                f"importable to build it ({exc})."
            ) from exc
        path = merged_csv("LGG", dest=path)
    return pd.read_csv(path)


def get_layer(df: pd.DataFrame, layer: Optional[str]):
    """
    Return (X_values, y, feature_names) for one omic layer.

    ``layer=None`` or any of ``MERGED_TOKENS`` ('merged', 'all', ...) uses ALL
    layers together (the full 34k-feature integrated matrix).  Otherwise columns
    are matched by the ``_<layer>`` suffix.
    """
    if _is_merged(layer):
        feat_cols = [c for c in df.columns if c != TARGET_COL]
    else:
        feat_cols = [c for c in df.columns if c.endswith(f"_{layer}")]
        if not feat_cols:
            raise ValueError(
                f"No columns with suffix '_{layer}'. Valid layers: {LAYERS}, "
                f"or one of {sorted(MERGED_TOKENS)} for all layers together."
            )
    X = df[feat_cols].to_numpy(dtype=float)
    y = df[TARGET_COL].to_numpy()
    return X, y, feat_cols


# ==============================================================================
# 2.  MULTICLASS-AWARE FEATURE SELECTORS
#     Each .select(X_train, y_train, seed) returns a set of column indices.
# ==============================================================================

# ------------------------------------------------------------------------------
# Thread budget for every estimator built in this module.
#
# The original pin to n_jobs=1 was justified as "bit-reproducible".  Measured
# against scikit-learn 1.7 / xgboost 3.3, thread count is not what
# reproducibility rests on here: RandomForest, RFECV, and permutation_importance
# each seed from random_state rather than from how the work is split, and return
# bit-identical output at any n_jobs.
#
# XGBoost needs one condition attached.  Under tree_method='hist' it is
# thread-invariant ONLY while subsample == 1.0; row subsampling draws in a
# thread-order-dependent way, and the same seed then yields different trees at
# different thread counts (measured drift ~2e-2 in predicted probabilities).
# _xgb() below never sets subsample, so the benchmark stays reproducible -- but
# do not add subsample here without also switching to tree_method='exact', which
# is unconditionally thread-invariant.
#
# GPU is the standing exception: device='cuda' uses a different split-finding
# implementation and shifts results regardless of threads.  It is opt-in via
# set_compute(use_gpu=True) and must not be enabled for runs meant to be
# compared against published numbers.
# ------------------------------------------------------------------------------

_N_JOBS: int = 1
_DEVICE: str = "cpu"


def set_compute(n_jobs: int = 1, use_gpu: bool = False) -> None:
    """
    Set the thread count and device used by every estimator in this module.

    Parameters
    ----------
    n_jobs : int
        Threads per estimator. -1 uses all cores. Does not affect results.
    use_gpu : bool
        Route XGBoost through CUDA. **Changes results slightly** and so
        invalidates comparison with CPU-generated benchmark numbers. Off by
        default; enable only for exploratory timing runs.
    """
    global _N_JOBS, _DEVICE
    _N_JOBS = int(n_jobs)
    _DEVICE = "cuda" if use_gpu else "cpu"
    if use_gpu:
        logger.warning(
            "GPU enabled for XGBoost. Results will differ slightly from the "
            "CPU numbers reported in BENCHMARKS.md; do not mix the two."
        )


def _xgb(seed: int, **kw):
    """
    XGBClassifier at library defaults.

    n_estimators, max_depth and learning_rate are no longer pinned. They were
    100 / 3 / 0.1 against XGBoost defaults of 100 / 6 / 0.3, so two of the
    three were tuned away from the shipped configuration. This benchmark
    compares what each method does out of the box, so nothing is set here
    beyond reproducibility (random_state) and the compute budget.
    """
    return XGBClassifier(
        random_state=seed, verbosity=0, eval_metric="mlogloss",
        tree_method="hist", n_jobs=_N_JOBS, device=_DEVICE, **kw,
    )


# Penalised-LR comparators run at sklearn's default C of 1.0. They were pinned
# at 0.1, which in the small-n / large-p regime drives every coefficient to
# exactly zero; SelectFromModel(threshold='mean') then compares against a
# threshold of 0.0 and returns the ENTIRE input, which is an artefact rather
# than a selection. max_iter stays raised because saga does not converge at
# sklearn's default of 100 on this data, and a non-converged fit tests nothing.
_LR_MAX_ITER: int = 3000


def _select_from_penalised_lr(X, y, seed: int, label: str, **penalty_kwargs):
    """Fit at the library default C; select nothing if it fully sparsifies."""
    est = LogisticRegression(solver="saga", max_iter=_LR_MAX_ITER,
                             random_state=seed, **penalty_kwargs)
    est.fit(X, y)
    if np.abs(est.coef_).max() == 0:
        logger.warning(
            "%s: every coefficient is zero. Returning an empty selection "
            "rather than the full input.", label,
        )
        return set()
    sfm = SelectFromModel(est, threshold="mean", prefit=True)
    return {int(i) for i in np.where(sfm.get_support())[0]}


class VarCorrBaseline:
    name = "VarCorr_Baseline"

    def select(self, X, y, seed):
        var = np.var(X, axis=0)
        kept = np.where(var > np.median(var))[0]
        if len(kept) < 2:
            return set(map(int, kept))
        corr = np.abs(np.corrcoef(X[:, kept], rowvar=False))
        np.fill_diagonal(corr, 0.0)
        order = np.argsort(-var[kept])
        alive = np.ones(len(kept), dtype=bool)
        for i in order:
            if not alive[i]:
                continue
            drop = (corr[i] > 0.90) & alive
            drop[i] = False
            alive[drop] = False
        return {int(kept[k]) for k in np.where(alive)[0]}


class LASSOSelector:
    """Multinomial L1 logistic regression + SelectFromModel(threshold='mean')."""
    name = "LASSO"

    def select(self, X, y, seed):
        return _select_from_penalised_lr(X, y, seed, "LASSO", penalty="l1")


class ElasticNetSelector:
    """Multinomial elastic-net logistic regression (l1_ratio=0.5)."""
    name = "ElasticNet"

    def select(self, X, y, seed):
        return _select_from_penalised_lr(X, y, seed, "ElasticNet",
                                         penalty="elasticnet", l1_ratio=0.5)


class RFECVSelector:
    """RFECV with an XGB estimator; macro-OVR AUC scoring, ~20 elimination rounds."""
    name = "RFECV"

    def select(self, X, y, seed):
        p = X.shape[1]
        step = max(1, p // 20)
        min_sel = max(5, p // 100)
        est = _xgb(seed)   # XGBoost defaults
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        rfecv = RFECV(estimator=est, cv=cv, step=step,
                      min_features_to_select=min_sel, scoring="roc_auc_ovr",
                      n_jobs=_N_JOBS)
        rfecv.fit(X, y)
        return {int(i) for i in np.where(rfecv.support_)[0]}


class RFImportanceSelector:
    name = "RF_Importance"

    def select(self, X, y, seed):
        rf = RandomForestClassifier(random_state=seed, n_jobs=_N_JOBS)
        sfm = SelectFromModel(rf, threshold="mean").fit(X, y)
        return {int(i) for i in np.where(sfm.get_support())[0]}


class PermImportanceSelector:
    """XGB + permutation importance (macro-OVR AUC); keep above mean of positives."""
    name = "PermImportance"

    def select(self, X, y, seed):
        clf = _xgb(seed).fit(X, y)
        res = permutation_importance(clf, X, y, n_repeats=5,
                                     scoring="roc_auc_ovr",
                                     random_state=seed, n_jobs=_N_JOBS)
        imp = res.importances_mean
        pos = imp[imp > 0]
        thr = float(np.mean(pos)) if pos.size else 0.0
        return {int(i) for i in np.where(imp >= thr)[0]}


class SHAPSelector:
    """XGB + TreeExplainer; keep features with mean|SHAP| >= global mean.

    Multiclass SHAP returns one matrix per class (list) or a 3-D array;
    importance is averaged over classes and samples.
    """
    name = "SHAP"

    def select(self, X, y, seed):
        if not _SHAP_AVAILABLE:
            return set()
        clf = _xgb(seed).fit(X, y)
        sv = _shap.TreeExplainer(clf).shap_values(X)
        if isinstance(sv, list):                      # list of (n, p) per class
            imp = np.mean([np.abs(s).mean(0) for s in sv], axis=0)
        else:
            a = np.abs(np.asarray(sv))
            imp = a.mean(axis=(0, 2)) if a.ndim == 3 else a.mean(axis=0)
        return {int(i) for i in np.where(imp >= np.mean(imp))[0]}


class SelectOmicsMethod:
    """
    The SelectOmics pipeline using the benchmark-tuned ``omics`` preset
    (config ``h_071``).  Writes the train fold to a temp CSV, runs the full
    pipeline, and maps selected column names back to indices.

    ``so_fast=True`` disables Step 3 (RFECV wrapper) for a quicker pass.
    """
    name = "SelectOmics"

    def __init__(self, so_fast: bool = False):
        self.so_fast = so_fast

    def select(self, X, y, seed, feature_names: Optional[List[str]] = None):
        if not _SO_AVAILABLE:
            return set()
        p = X.shape[1]
        if feature_names is None:
            feature_names = [f"feat_{i:05d}" for i in range(p)]
        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y
        with tempfile.TemporaryDirectory() as tmp:
            data_path = os.path.join(tmp, "data.csv")
            out_dir = os.path.join(tmp, "so_out")
            df.to_csv(data_path, index=False)
            # PLAIN DEFAULTS, not the omics() preset.
            #
            # omics() carries thresholds derived from a configuration search
            # (stability_threshold, step3 retention). Running SelectOmics
            # tuned against comparators at their library defaults is the
            # asymmetry this benchmark exists to avoid, and it is the same
            # distortion that the comparators' own non-default settings
            # introduced in the other direction.
            #
            # Every method here now runs as its library ships it. If a
            # "best achievable performance" comparison is wanted later, that
            # is a separate study with a tuning budget for every method, not a
            # variation folded into this one.
            cfg = SelectOmicsConfig(
                data_path=data_path, target_column="target",
                random_seed=int(seed), output_dir=out_dir,
                test_size=0.20, verbose=False,
                create_visualizations=False, save_intermediate_results=False,
                enable_step_evaluations=False,
                enable_final_test_evaluation=False,
                enable_step3=(not self.so_fast),
                # Match the surrounding harness so the pipeline does not
                # oversubscribe when several benchmark workers run at once.
                n_jobs=_N_JOBS,
                use_gpu=(_DEVICE == "cuda"),
            )
            try:
                pipe = SelectOmicsPipeline(cfg)
                pipe.run(validate=False)
                cols = pipe.get_selected_features(step=None)
            except Exception as exc:
                logger.warning("SelectOmics failed (seed=%d): %s", seed, exc)
                return set()
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        return {name_to_idx[c] for c in cols if c in name_to_idx}


def default_methods(so_fast: bool = False) -> List:
    methods = [
        VarCorrBaseline(), LASSOSelector(), ElasticNetSelector(),
        RFECVSelector(), RFImportanceSelector(), PermImportanceSelector(),
    ]
    if _SHAP_AVAILABLE:
        methods.append(SHAPSelector())
    if _SO_AVAILABLE:
        methods.append(SelectOmicsMethod(so_fast=so_fast))
    return methods


# ==============================================================================
# 3.  DOWNSTREAM EVALUATION (multiclass, leakage-free)
# ==============================================================================

def downstream_metrics(X_tr, y_tr, X_te, y_te, selected: Set[int],
                       seed: int = 0) -> Dict[str, float]:
    """Train a fixed XGB on selected columns; score macro-OVR AUC / F1 / bal-acc."""
    out = dict(auc_ovr=np.nan, f1_macro=np.nan, balanced_acc=np.nan)
    if not selected:
        return out
    idx = sorted(selected)
    classes = np.unique(y_tr)
    try:
        clf = _xgb(seed)
        clf.fit(X_tr[:, idx], y_tr)
        proba = clf.predict_proba(X_te[:, idx])
        pred = clf.predict(X_te[:, idx])
        # Prediction-based metrics are always defined.
        out["f1_macro"] = float(f1_score(y_te, pred, average="macro"))
        out["balanced_acc"] = float(balanced_accuracy_score(y_te, pred))
        # macro-OVR AUC: pass labels=classes so proba columns align even if a
        # class is rare in the test fold (computed unconditionally; only a
        # truly-absent class would raise, which we then leave as NaN).
        try:
            out["auc_ovr"] = float(roc_auc_score(
                y_te, proba, multi_class="ovr", average="macro", labels=classes))
        except ValueError as exc:
            logger.warning("auc_ovr undefined this fold (class absent?): %s", exc)
    except Exception as exc:
        logger.warning("downstream eval failed: %s", exc)
    return out


# ==============================================================================
# 4.  CV RUNNER  (checkpoint-safe)
# ==============================================================================

def _done_set(df: pd.DataFrame) -> set:
    if df.empty:
        return set()
    return {(r["layer"], r["method"], int(r["fold"])) for _, r in df.iterrows()}


def _redundancy(X: np.ndarray, selected: Set[int], cap: int = 2000, seed: int = 0) -> float:
    """
    Mean |pairwise Pearson r| among selected features.

    A full kxk correlation matrix is O(k^2) in memory -- on the merged 34k-feature
    layer a low-selectivity method can select nearly everything, which would try
    to allocate a ~9 GB matrix. When the selection exceeds ``cap`` features we
    estimate the mean on a seeded random subsample of ``cap`` of them (a robust
    estimator of the mean pairwise correlation), bounding memory at cap^2.
    Returns identical values to a full computation whenever k <= cap.
    """
    idx = sorted(selected)
    if len(idx) < 2:
        return 0.0
    if len(idx) > cap:
        rng = np.random.RandomState(seed)
        idx = sorted(rng.choice(idx, size=cap, replace=False))
    corr = np.abs(np.corrcoef(X[:, idx], rowvar=False))
    corr[~np.isfinite(corr)] = 0.0
    iu = np.triu_indices_from(corr, k=1)
    vals = corr[iu]
    return float(np.mean(vals)) if vals.size else 0.0


def _run_one_trial(
    layer_label: str,
    method_name: str,
    fold: int,
    tr: np.ndarray,
    te: np.ndarray,
    fold_seed: int,
    n_features: int,
) -> Dict:
    """
    Run one (method, fold) trial and return its result row.

    Executed in a worker process when ``workers > 1``, so it must be
    importable at module level and take only picklable arguments. The feature
    matrix is not an argument: it is broadcast to each worker once at start-up
    and read back through ``worker_context()``, which keeps a 34k-feature
    matrix from being pickled once per task.

    Imputation is refit here rather than once per fold so that each trial is
    self-contained. The cost is negligible next to selection, and it is what
    lets folds and methods be scheduled independently.
    """
    ctx = worker_context()
    X, y = ctx["X"], ctx["y"]
    feat_names, methods = ctx["feat_names"], ctx["methods"]
    set_compute(ctx["n_jobs"], ctx["use_gpu"])

    imp = SimpleImputer(strategy="median").fit(X[tr])
    X_tr, X_te = imp.transform(X[tr]), imp.transform(X[te])
    y_tr, y_te = y[tr], y[te]

    method = methods[method_name]
    t0 = time.perf_counter()
    try:
        if method_name == "SelectOmics":
            sel = method.select(X_tr, y_tr, fold_seed, feature_names=feat_names)
        else:
            sel = method.select(X_tr, y_tr, fold_seed)
    except Exception as exc:
        logger.warning("%s failed [%s fold=%d]: %s",
                       method_name, layer_label, fold, exc)
        sel = set()
    elapsed = round(time.perf_counter() - t0, 1)

    met = downstream_metrics(X_tr, y_tr, X_te, y_te, sel, seed=fold_seed)
    return {
        "layer": layer_label, "method": method_name, "fold": fold,
        "scenario": layer_label, "seed": fold,   # for reused stats fns
        "n_features": n_features, "n_selected": len(sel),
        "reduction_ratio": round(1 - len(sel) / n_features, 4) if n_features else np.nan,
        "redundancy": round(_redundancy(X_tr, sel, seed=fold_seed), 4),
        "auc_ovr": round(met["auc_ovr"], 4) if met["auc_ovr"] == met["auc_ovr"] else np.nan,
        "f1_macro": round(met["f1_macro"], 4) if met["f1_macro"] == met["f1_macro"] else np.nan,
        "balanced_acc": round(met["balanced_acc"], 4) if met["balanced_acc"] == met["balanced_acc"] else np.nan,
        "time_seconds": elapsed,
        "selected_indices": ",".join(map(str, sorted(sel))),
    }


def run_layer(
    df_merged: pd.DataFrame,
    layer: str,
    methods: List,
    save_path: Path,
    n_splits: int = 5,
    n_repeats: int = 2,
    base_seed: int = 0,
    verbose: bool = True,
    workers: Optional[int] = 1,
    n_jobs: Optional[int] = None,
    use_gpu: bool = False,
) -> pd.DataFrame:
    """
    Benchmark every method on one layer with RepeatedStratifiedKFold.

    Per fold: median-impute (train-fit) -> select on train -> score on test.
    Appends one row per (method, fold) to ``save_path``.

    Features are **not** re-standardised: the LGG layers are already normalised
    upstream (per-layer), and forcing unit variance would neutralise the
    variance signal that SelectOmics' Step 1 and the VarCorr baseline rely on --
    biasing the comparison. Imputation is needed only for the methylation layer
    (~0.17% NaN), which sklearn methods cannot ingest; it is applied to every
    method identically to keep the comparison fair.

    Parallelism
    -----------
    ``workers`` sets how many (method, fold) trials run concurrently; ``n_jobs``
    sets threads per estimator inside each. Leave ``n_jobs`` as None to have the
    cores split evenly across workers. ``workers=1`` (the default) keeps the
    original in-process behaviour for notebooks and debugging. Results are
    identical either way; only ``use_gpu=True`` changes them.

    Trials are scheduled at (method, fold) granularity rather than per layer
    because method costs differ by orders of magnitude -- SelectOmics and RFECV
    dominate VarCorr -- and fine granularity is what keeps every core busy to
    the end of the run.
    """
    X, y, feat_names = get_layer(df_merged, layer)
    # Stamp merged/None runs with a real label so summarise()'s groupby (which
    # drops NaN groups) keeps them and the 'layer' column is human-readable.
    layer_label = MERGED_LABEL if _is_merged(layer) else layer
    p = X.shape[1]

    resolved_workers, resolved_jobs = plan_parallelism(workers, n_jobs)
    set_compute(resolved_jobs, use_gpu)

    if verbose:
        print(f"\n-- Layer {layer_label}  [n={X.shape[0]} * p={p} * "
              f"{n_splits}x{n_repeats} folds]")
        print(f"   {describe_plan(resolved_workers, resolved_jobs)}")

    done = _done_set(pd.read_csv(save_path)) if save_path.exists() else set()
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                  random_state=base_seed)

    tasks = []
    for fold, (tr, te) in enumerate(rkf.split(X, y)):
        fold_seed = base_seed * 1000 + fold
        for m in methods:
            if (layer_label, m.name, fold) in done:
                continue
            tasks.append((layer_label, m.name, fold, tr, te, fold_seed, p))

    if not tasks:
        if verbose:
            print("   all trials already present in the checkpoint; nothing to do")
        return pd.read_csv(save_path)

    # The parent process is the only writer, so incremental appends stay
    # uncorrupted no matter how many workers are producing rows.
    def _on_result(row: Dict) -> None:
        _append_row(save_path, row)
        if verbose:
            print(f"   {row['method']:18s} fold={row['fold']}  "
                  f"n_sel={row['n_selected']:5d}  AUC={row['auc_ovr']}  "
                  f"F1={row['f1_macro']}  ({row['time_seconds']:.0f}s)")

    run_tasks(
        _run_one_trial,
        tasks,
        workers=resolved_workers,
        n_jobs=resolved_jobs,
        context={
            "X": X, "y": y, "feat_names": feat_names,
            "methods": {m.name: m for m in methods},
            "n_jobs": resolved_jobs, "use_gpu": use_gpu,
        },
        on_result=_on_result,
    )

    return pd.read_csv(save_path)


def run_all_layers(
    df_merged: pd.DataFrame,
    methods: List,
    save_path: Path,
    layers: Optional[List[str]] = None,
    n_splits: int = 5,
    n_repeats: int = 2,
    base_seed: int = 0,
    verbose: bool = True,
    workers: Optional[int] = None,
    n_jobs: Optional[int] = None,
    use_gpu: bool = False,
) -> pd.DataFrame:
    """
    Benchmark every layer in turn, parallelising trials within each.

    Layers are run one after another rather than concurrently: each holds its
    own copy of the feature matrix in every worker, and the merged layer alone
    is ~67 MB, so running four at once would multiply peak memory for no gain
    over the per-trial parallelism already used inside a layer.

    Parameters
    ----------
    layers : list of str or None
        Which layers to run. Defaults to ``LAYERS``. Pass ``["merged"]`` for
        the integrated matrix.
    workers, n_jobs, use_gpu
        See ``run_layer``.

    Returns
    -------
    pd.DataFrame
        Every row accumulated in ``save_path``.
    """
    layers = layers if layers is not None else LAYERS
    for layer in layers:
        run_layer(
            df_merged, layer, methods, save_path,
            n_splits=n_splits, n_repeats=n_repeats, base_seed=base_seed,
            verbose=verbose, workers=workers, n_jobs=n_jobs, use_gpu=use_gpu,
        )
    return pd.read_csv(save_path)


def _append_row(path: Path, row: dict) -> None:
    header = not path.exists()
    pd.DataFrame([row]).to_csv(path, mode="a", header=header, index=False)


# ==============================================================================
# 5.  AGGREGATION  (incl. cross-fold stability)
# ==============================================================================

def summarise(df: pd.DataFrame, warn_incomplete: bool = True) -> pd.DataFrame:
    """
    Per (layer, method): mean downstream metrics + cross-fold Kuncheva KI.

    Warns when any (layer, method) is missing folds. A partial checkpoint
    averages over whatever rows exist and produces a confident-looking number
    with no indication that it rests on half the data, which is exactly how an
    aborted run went unnoticed: a CSV holding 282 of 320 trials listed every
    method and every fold index SOMEWHERE, and only a per-cell fold count
    exposed that one layer had stopped at fold 5.
    """
    if warn_incomplete and not df.empty:
        counts = df.groupby(["layer", "method"])["fold"].nunique()
        expected = int(counts.max())
        short = counts[counts < expected]
        if len(short):
            logger.warning(
                "Incomplete results: %d of %d (layer, method) cells have fewer "
                "than %d folds. Means below are computed over the folds that "
                "are present. Re-run to fill the gaps (existing rows resume).",
                len(short), len(counts), expected,
            )
            for (layer, method), n in short.items():
                logger.warning("    %-8s %-18s %d/%d folds",
                               layer, method, int(n), expected)
    rows = []
    for (layer, method), g in df.groupby(["layer", "method"]):
        p = int(g["n_features"].iloc[0])
        sets = [set(int(i) for i in str(s).split(",") if i != "")
                for s in g["selected_indices"]]
        ki = kuncheva_stability(sets, p)
        rows.append({
            "layer": layer, "method": method,
            "auc_ovr": round(g["auc_ovr"].mean(), 4),
            "f1_macro": round(g["f1_macro"].mean(), 4),
            "balanced_acc": round(g["balanced_acc"].mean(), 4),
            "kuncheva": round(ki, 4) if ki == ki else np.nan,
            # A feature count is a count. Reporting a mean of 10.9 invites the
            # reader to picture nine tenths of a gene; round to a whole number.
            "n_selected": int(round(g["n_selected"].mean())),
            "reduction_ratio": round(g["reduction_ratio"].mean(), 4),
            "redundancy": round(g["redundancy"].mean(), 4),
            "time_s": round(g["time_seconds"].mean(), 1),
        })
    return (pd.DataFrame(rows)
            .sort_values(["layer", "auc_ovr"], ascending=[True, False])
            .reset_index(drop=True))


# ==============================================================================
# 6.  COMMAND-LINE ENTRY POINT
# ==============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    """
    Run the LGG benchmark from the shell.

    Examples
    --------
        # All four layers, filling the machine, resumable:
        python benchmark_lgg.py --workers -1

        # One layer, 8 concurrent trials, 3 threads each:
        python benchmark_lgg.py --layers mRNA --workers 8 --n-jobs 3

        # Quick pass with Step 3 of SelectOmics disabled:
        python benchmark_lgg.py --layers miRNA --so-fast --folds 3 --repeats 1
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="benchmark_lgg",
        description="Benchmark SelectOmics against six standard methods on TCGA LGG.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", default=None,
                        help="Path to LGG_merged.csv (default: alongside this script).")
    parser.add_argument("--layers", nargs="+", default=None,
                        metavar="LAYER",
                        help=f"Layers to run. Choices: {LAYERS} or 'merged'. "
                             "Default: all four separately.")
    parser.add_argument("--output", "-o", default="results_lgg/benchmark_lgg_raw.csv",
                        help="Result CSV. Existing rows are resumed, not repeated.")
    parser.add_argument("--folds", type=int, default=5, help="CV folds (default 5).")
    parser.add_argument("--repeats", type=int, default=2,
                        help="CV repeats (default 2).")
    parser.add_argument("--seed", type=int, default=0, help="Base seed (default 0).")
    parser.add_argument("--workers", type=int, default=None,
                        help="Concurrent trials. -1 fills the machine, 1 runs "
                             "in-process. Default: half the cores.")
    parser.add_argument("--n-jobs", type=int, default=None, dest="n_jobs",
                        help="Threads per estimator. Default: cores divided "
                             "among workers. Does not affect results.")
    parser.add_argument("--use-gpu", action="store_true",
                        help="Route XGBoost through CUDA. CHANGES RESULTS "
                             "slightly; do not mix with CPU-generated numbers.")
    parser.add_argument("--so-fast", action="store_true",
                        help="Disable SelectOmics Step 3 for a quicker pass.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-trial output.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    save_path = Path(args.output)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading LGG matrix ...")
    df = load_merged(args.data)
    methods = default_methods(so_fast=args.so_fast)
    print(f"Methods: {', '.join(m.name for m in methods)}")

    t0 = time.perf_counter()
    raw = run_all_layers(
        df, methods, save_path,
        layers=args.layers,
        n_splits=args.folds, n_repeats=args.repeats, base_seed=args.seed,
        verbose=not args.quiet,
        workers=args.workers, n_jobs=args.n_jobs, use_gpu=args.use_gpu,
    )
    elapsed = time.perf_counter() - t0

    summary = summarise(raw)
    summary_path = save_path.with_name(save_path.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"\n{'=' * 72}")
    print(summary.to_string(index=False))
    print(f"{'=' * 72}")
    print(f"Wall clock: {elapsed / 60:.1f} min")
    print(f"Raw rows  : {save_path}")
    print(f"Summary   : {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
