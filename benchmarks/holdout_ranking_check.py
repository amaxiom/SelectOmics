"""
What does ranking the steps on held-out folds change?

enable_holdout_ranking moves the recommendation off validation computed on the
training split, which scores every panel on the samples its features were
selected from, and onto folds held out of that split. This runs one pipeline
per dataset with the flag on and reports, for the same run:

  holdout_step   the step the held-out ranking names
  default_step   the step the training-split validation would have named,
                 recomputed from the same comparison table
  nested_step    for reference, the step that scored best under nested CV in
                 nested_recommendation.py, with the same parsimony rule

Datasets and settings come from nested_recommendation.py, so the arms are
comparable with the numbers in BENCHMARKS.md section 8.

Usage:
    python holdout_ranking_check.py --dataset gbm_mirna
    python holdout_ranking_check.py --report
"""
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

from SelectOmics import SelectOmicsConfig, SelectOmicsPipeline     # noqa: E402
from SelectOmics.evaluation.validation import (                    # noqa: E402
    build_recommendation,
)
import nested_recommendation as nr                                 # noqa: E402

FIELDS = ["dataset", "holdout_step", "holdout_n", "holdout_auc",
          "default_step", "default_n", "default_weighted",
          "quality", "changed", "seconds", "error"]


def _default_choice(pipeline):
    """What the training-split ranking alone would have named."""
    step_data = {
        name: {"step_name": name, "X_train": st.X_train,
               "eval_result": pipeline._panel_cv_result(name)}
        for name in ("Step 0 (Reference)", "Step 1 (Data Cleaning)",
                     "Step 2 (Regularization)", "Step 3 (Wrappers)")
        if (st := pipeline.state_tracker.get_step(name)) is not None
    }
    return build_recommendation(pipeline._validation_comparison, step_data,
                                pipeline._X_test)


def run(name: str, out: Path, n_jobs: int) -> int:
    spec = nr.DATASETS[name]
    tmp = Path(tempfile.mkdtemp())
    path, target, _ = nr._materialise(spec, tmp)

    cfg = SelectOmicsConfig(
        data_path=str(path), target_column=target,
        algorithm=spec["algorithm"], output_dir=str(tmp / "o"),
        n_consensus_models=spec["n_models"], quick_tune_iterations=10,
        n_bootstrap=spec["n_bootstrap"], verbose=False,
        create_visualizations=False, save_intermediate_results=False,
        enable_step3=True, step2_role=spec["role"], allow_union_rung=False,
        enable_step_evaluations=True, enable_final_test_evaluation=False,
        enable_holdout_ranking=True, n_jobs=n_jobs,
    )
    row = dict.fromkeys(FIELDS, "")
    row["dataset"] = name
    t0 = time.perf_counter()
    p = SelectOmicsPipeline(cfg)
    res = p.run(validate=True)

    rec = res["recommendation"]
    default = _default_choice(p)
    row.update(
        holdout_step=rec["step_id"], holdout_n=rec["n_features"],
        holdout_auc=round(float(rec["ranking_score"]), 4),
        default_step=default["step_id"], default_n=default["n_features"],
        default_weighted=round(float(default["weighted_auc"]), 4),
        quality=rec["quality"],
        changed=rec["step_id"] != default["step_id"],
        seconds=round(time.perf_counter() - t0, 1),
    )
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerow(row)
    print(f"{name}: held-out picks {row['holdout_step']} "
          f"({row['holdout_n']}f, AUC {row['holdout_auc']}), "
          f"default picks {row['default_step']} ({row['default_n']}f), "
          f"changed={row['changed']}, quality={row['quality']} "
          f"({row['seconds']}s)")
    print(res["holdout_ranking"].round(4).to_string())
    return 0


def report() -> int:
    files = sorted(_HERE.glob("results_holdout_ranking_*.csv"))
    if not files:
        print("no results_holdout_ranking_*.csv files")
        return 1
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # The nested-CV ranking, for reference: best mean held-out AUC per step,
    # then the smallest panel within one standard error of it.
    nested = {}
    for f in sorted(_HERE.glob("results_nested_*.csv")):
        n = pd.read_csv(f)
        s = n.groupby("step", sort=False).agg(
            k=("n_features", "mean"), auc=("outer_auc", "mean"),
            sd=("outer_auc", "std"))
        best = s.auc.idxmax()
        se = max(s.loc[best, "sd"] / np.sqrt(n.fold.nunique()), 0.005)
        nested[n.dataset[0]] = s[s.auc >= s.loc[best, "auc"] - se].k.idxmin()
    d["nested_step"] = d.dataset.map(nested)

    pd.set_option("display.width", 200)
    print(d.to_string(index=False))
    print(f"\nchanged the choice: {int(d['changed'].sum())} of {len(d)}")
    agree = (d.holdout_step == d.nested_step).sum()
    print(f"held-out choice matches the nested-CV ranking: {agree} of "
          f"{d.nested_step.notna().sum()}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(nr.DATASETS))
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=3)
    args = ap.parse_args()
    if args.report:
        return report()
    if not args.dataset:
        ap.error("--dataset or --report is required")
    return run(args.dataset,
               _HERE / f"results_holdout_ranking_{args.dataset}.csv",
               args.n_jobs)


if __name__ == "__main__":
    raise SystemExit(main())
