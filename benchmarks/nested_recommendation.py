"""
Does the recommendation generalise?

The recommendation picks one step's panel from validation on the training
split, which is the same data every panel was selected on. So it is a choice
scored on the data it was made with. Two ways that could go wrong:

1. Optimism. The recommended panel's validation score overstates how it does
   on unseen samples, because selection already fitted that data.

2. A biased choice. Heavier selection fits the training split harder, so the
   most aggressively selected panel could win validation because it overfits
   most, not because it generalises best.

Nested CV tests both directly. Each outer fold reruns the full procedure
(selection, validation, recommendation) on the outer training portion only,
then scores every step's panel on the held-out fold. That is what
run_nested_cv() does as of this version, so this harness calls it unchanged.

Reported per outer fold and step: panel size, the inner validation score the
choice was made on, the held-out AUC, and whether the step was the recommended
and the last one. Synthetic datasets add ground-truth F1 for every panel.

Usage:
    python nested_recommendation.py --dataset gbm_mirna
    python nested_recommendation.py --report
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

from SelectOmics import SelectOmicsConfig, SelectOmicsPipeline   # noqa: E402

_EX = _HERE.parent / "examples"

# Real layers are multiclass miRNA where validation AUC does not saturate, so
# AUC can separate the panels; Step 3 fires on all three under defaults.
# Synthetic cases carry ground truth: the rescue case where Step 3 over-prunes
# (omics_standard, prefilter, 15 models), the case where the cascade wins
# (omics_multiclass, prefilter), a typical n << p case where Step 3 does not
# fire (omics_high_dim), and a null control, where every held-out AUC should
# sit near 0.5 and any validation score above that is pure optimism.
DATASETS = {
    "gbm_mirna": dict(kind="csv", path=_EX / "GS-GBM" / "GBM_miRNA_input.csv",
                      target="Label", algorithm="RF", n_models=5,
                      role="terminal", n_bootstrap=30),
    "ov_mirna": dict(kind="csv", path=_EX / "GS-OV" / "OV_miRNA_input.csv",
                     target="Label", algorithm="RF", n_models=5,
                     role="terminal", n_bootstrap=30),
    "lgg_mirna": dict(kind="lgg", layer="miRNA", target="Label",
                      algorithm="RF", n_models=5, role="terminal",
                      n_bootstrap=30),
    "standard_prefilter": dict(kind="synthetic", scenario="omics_standard",
                               algorithm="XGB", n_models=15,
                               role="prefilter", n_bootstrap=20),
    "multiclass_prefilter": dict(kind="synthetic",
                                 scenario="omics_multiclass",
                                 algorithm="XGB", n_models=5,
                                 role="prefilter", n_bootstrap=20),
    "high_dim_terminal": dict(kind="synthetic", scenario="omics_high_dim",
                              algorithm="XGB", n_models=5, role="terminal",
                              n_bootstrap=20),
    "null_terminal": dict(kind="synthetic", scenario="null_control",
                          algorithm="XGB", n_models=5, role="terminal",
                          n_bootstrap=20),
}

STEP_NAMES = ["Step 0 (Reference)", "Step 1 (Data Cleaning)",
              "Step 2 (Regularization)", "Step 3 (Wrappers)"]
OUTER_FOLDS = 5
SEED = 0


def _materialise(spec, tmp: Path):
    """Write the dataset to CSV; return (path, target, true_idx or None)."""
    if spec["kind"] == "csv":
        return spec["path"], spec["target"], None
    if spec["kind"] == "lgg":
        import benchmark_lgg as bl
        df = bl.load_merged()
        cols = [c for c in df.columns if c.endswith(f"_{spec['layer']}")]
        out = df[cols + [spec["target"]]]
        path = tmp / "lgg_layer.csv"
        out.to_csv(path, index=False)
        return path, spec["target"], None
    import benchmark_synthetic as bs
    sc = next(s for s in bs.SCENARIOS if s.name == spec["scenario"])
    X, y, true_idx = bs.make_omics_dataset(sc, seed=SEED)
    df = X.copy()
    df["Class"] = y
    path = tmp / "synthetic.csv"
    df.to_csv(path, index=False)
    return path, "Class", set(true_idx)


def _capture_inner_panels(store):
    """
    Record every step's panel inside each nested fold, for ground-truth F1.

    run_nested_cv reports panel sizes per step but only the recommended
    panel's feature names. An inner fold pipeline is the one built without
    test labels, which is how this tells it apart from any other.
    """
    real = SelectOmicsPipeline._run_selection_steps

    def spy(self, *args, **kwargs):
        real(self, *args, **kwargs)
        if self._y_test is None:
            store.append({
                name: list(st.X_train.columns)
                for name in STEP_NAMES
                if (st := self.state_tracker.get_step(name)) is not None
            })

    SelectOmicsPipeline._run_selection_steps = spy


def run(name: str, out: Path, n_jobs: int) -> int:
    spec = DATASETS[name]
    tmp = Path(tempfile.mkdtemp())
    path, target, true_idx = _materialise(spec, tmp)

    panels: list = []
    _capture_inner_panels(panels)

    cfg = SelectOmicsConfig(
        data_path=str(path), target_column=target,
        algorithm=spec["algorithm"], output_dir=str(tmp / "o"),
        n_consensus_models=spec["n_models"], quick_tune_iterations=10,
        n_bootstrap=spec["n_bootstrap"], verbose=False,
        create_visualizations=False, save_intermediate_results=False,
        enable_step3=True, step2_role=spec["role"], allow_union_rung=False,
        enable_step_evaluations=True, enable_final_test_evaluation=False,
        enable_nested_cv=True, outer_cv_splits=OUTER_FOLDS, n_jobs=n_jobs,
    )
    t0 = time.perf_counter()
    nested = SelectOmicsPipeline(cfg).run_nested_cv()
    seconds = time.perf_counter() - t0

    d = nested["fold_details"].copy()
    d.insert(0, "dataset", name)
    if true_idx is not None:
        import benchmark_synthetic as bs
        f1, prec, rec = [], [], []
        for _, row in d.iterrows():
            cols = panels[int(row["fold"]) - 1][row["step"]]
            g = bs.ground_truth_metrics(
                {int(c.split("_")[1]) for c in cols}, true_idx)
            f1.append(g["f1"])
            prec.append(g["precision"])
            rec.append(g["recall"])
        d["gt_f1"], d["gt_precision"], d["gt_recall"] = f1, prec, rec
    d["seconds_total"] = round(seconds, 1)
    d.to_csv(out, index=False)

    print(f"{name}: {seconds / 60:.1f} min")
    print(f"  recommended  {nested['mean_auc']:.4f} +/- {nested['std_auc']:.4f}")
    print(f"  last step    {nested['last_step_mean_auc']:.4f}")
    print(f"  best (oracle){nested['best_step_mean_auc']:.4f}")
    print(f"  mean regret  {nested['mean_regret']:.4f}")
    print(f"  recommended  {nested['fold_recommended_steps']}")
    print(f"wrote {out}")
    return 0


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _spearman(a, b):
    a, b = pd.Series(a).rank(), pd.Series(b).rank()
    if a.nunique() < 2 or b.nunique() < 2:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def report() -> int:
    files = sorted(_HERE.glob("results_nested_*.csv"))
    if not files:
        print("no results_nested_*.csv files")
        return 1
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    rows = []
    for (ds, fold), g in d.groupby(["dataset", "fold"], sort=False):
        rec = g[g["recommended"]].iloc[0]
        last = g[g["last"]].iloc[0]
        s0 = g[g["step"] == "Step 0 (Reference)"].iloc[0]
        # Distinct panels only: a skipped step duplicates its input, and
        # counting it twice would inflate the rank agreement.
        distinct = g.drop_duplicates(subset=["n_features", "outer_auc",
                                             "inner_score"])
        r = {
            "dataset": ds, "fold": fold,
            "rec_step": rec["step"].split(" (")[0], "rec_n": rec["n_features"],
            "rec_auc": rec["outer_auc"], "last_auc": last["outer_auc"],
            "step0_auc": s0["outer_auc"], "best_auc": g["outer_auc"].max(),
            "rec_inner": rec["inner_score"],
            "step0_inner": s0["inner_score"],
            "rank_rho": _spearman(distinct["inner_score"],
                                  distinct["outer_auc"]),
            "n_distinct": len(distinct),
        }
        if "gt_f1" in g and g["gt_f1"].notna().any():
            r.update(rec_f1=rec["gt_f1"], last_f1=last["gt_f1"],
                     best_f1=g["gt_f1"].max())
        rows.append(r)
    f = pd.DataFrame(rows)
    f["regret"] = f["best_auc"] - f["rec_auc"]
    f["optimism"] = f["rec_inner"] - f["rec_auc"]
    f["step0_optimism"] = f["step0_inner"] - f["step0_auc"]

    pd.set_option("display.width", 200)
    print("Per fold")
    print(f.round(4).to_string(index=False))

    agg = f.groupby("dataset", sort=False).agg(
        folds=("fold", "count"),
        rec=("rec_step", lambda s: ", ".join(
            f"{k}x{v}" for k, v in s.value_counts().items())),
        rec_n=("rec_n", "mean"),
        rec_auc=("rec_auc", "mean"), last_auc=("last_auc", "mean"),
        step0_auc=("step0_auc", "mean"), best_auc=("best_auc", "mean"),
        regret=("regret", "mean"), max_regret=("regret", "max"),
        optimism=("optimism", "mean"),
        step0_optimism=("step0_optimism", "mean"),
        rank_rho=("rank_rho", "mean"),
    )
    if "rec_f1" in f:
        agg = agg.join(f.groupby("dataset", sort=False)[
            ["rec_f1", "last_f1", "best_f1"]].mean())
    print("\nPer dataset (means over outer folds)")
    print(agg.round(4).to_string())

    print("\nAll folds")
    print(f"  folds                        {len(f)}")
    print(f"  recommended = best (<=0.005) {(f['regret'] <= 0.005).sum()}")
    print(f"  mean regret                  {f['regret'].mean():.4f}")
    print(f"  max regret                   {f['regret'].max():.4f}")
    print(f"  rec - last AUC, mean         {(f['rec_auc'] - f['last_auc']).mean():+.4f}")
    print(f"  rec - step0 AUC, mean        {(f['rec_auc'] - f['step0_auc']).mean():+.4f}")
    print(f"  optimism rec / step0         {f['optimism'].mean():+.4f} / "
          f"{f['step0_optimism'].mean():+.4f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(DATASETS))
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=3)
    args = ap.parse_args()
    if args.report:
        return report()
    if not args.dataset:
        ap.error("--dataset or --report is required")
    out = _HERE / f"results_nested_{args.dataset}.csv"
    return run(args.dataset, out, args.n_jobs)


if __name__ == "__main__":
    raise SystemExit(main())
