"""
step2_role: 'terminal' vs 'prefilter', across the development suite.

Terminal is the shipping default: Step 2 optimises the panel it delivers.
Prefilter declares Step 2 a candidate generator, overshooting Step 3's entry
gate so Step 3 engages and does the precision work.

Writes each row as it completes and skips rows already present, so the run is
resumable and an interruption costs at most one trial.

    python role_comparison.py --workers 6
    python role_comparison.py --report-only
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                              # noqa: E402
from sklearn.model_selection import train_test_split            # noqa: E402
import benchmark_synthetic as bs                                # noqa: E402
from SelectOmics import SelectOmicsConfig, SelectOmicsPipeline  # noqa: E402

# (step2_role, allow_union_rung) -- the union rung is the lever that
# raises the ceiling, so it is tested both alone and paired with prefilter.
ARMS = (("terminal", False), ("terminal", True),
        ("prefilter", False), ("prefilter", True))
SEEDS = (0, 1, 2, 3, 4)
FIELDS = ["scenario", "seed", "role", "n_after_s1", "n_after_s2", "s3_ran",
          "n_final", "s2_f1", "f1", "precision", "recall", "test_auc",
          "union", "s2_outcome",
          "seconds", "error"]


def _one(task):
    name, seed, role, union, n_jobs = task
    t0 = time.perf_counter()
    row = dict.fromkeys(FIELDS, "")
    row.update(scenario=name, seed=seed, role=role, union=union)
    try:
        sc = next(s for s in bs.SCENARIOS if s.name == name)
        X, y, ti = bs.make_omics_dataset(sc, seed=seed)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=bs.BENCHMARK_SPEC["test_size"],
            stratify=y, random_state=seed)
        tmp = Path(tempfile.mkdtemp()); csv_path = tmp / "d.csv"
        df = Xtr.copy(); df["Class"] = ytr; df.to_csv(csv_path, index=False)
        p = SelectOmicsPipeline(SelectOmicsConfig(
            data_path=str(csv_path), target_column="Class", algorithm="XGB",
            output_dir=str(tmp / "o"), n_consensus_models=5,
            quick_tune_iterations=10, n_bootstrap=10, verbose=False,
            create_visualizations=False, save_intermediate_results=False,
            enable_step_evaluations=False, enable_final_test_evaluation=False,
            test_size=0.2, step2_role=role, allow_union_rung=union,
            n_jobs=n_jobs))
        p.load_data(); p.run_step0_reference()
        p.run_step1_cleaning()
        row["n_after_s1"] = len(p.get_selected_features("step1"))
        p.run_step2_regularization()
        s2_cols = p.get_selected_features("step2")
        row["n_after_s2"] = len(s2_cols)
        row["s2_f1"] = round(bs.ground_truth_metrics(
            {int(c.split("_")[1]) for c in s2_cols}, ti)["f1"], 4)
        row["s2_outcome"] = p.results["step2"].get("consensus_outcome", "")
        p.run_step3_wrapper()
        cols = p.get_selected_features("step3")
        row["s3_ran"] = not p.results["step3"].get("skipped", True)
        row["n_final"] = len(cols)
        idx = {int(c.split("_")[1]) for c in cols}
        g = bs.ground_truth_metrics(idx, ti)
        row.update(f1=round(g["f1"], 4), precision=round(g["precision"], 4),
                   recall=round(g["recall"], 4))
        auc = bs.downstream_auc(Xtr.values, ytr, Xte.values, yte, idx, seed)
        row["test_auc"] = "" if np.isnan(auc) else round(auc, 4)
    except Exception:
        row["error"] = traceback.format_exc(limit=2).replace("\n", " | ")[:300]
    row["seconds"] = round(time.perf_counter() - t0, 1)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(_HERE / "results_role_comparison.csv"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)

    done = set()
    if out.exists():
        with open(out, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                done.add((r["scenario"], int(r["seed"]), r["role"],
                          r.get("union") in ("True", True)))
    if args.report_only:
        print(f"{len(done)} rows recorded in {out}")
        return 0

    n_jobs = max(1, (os.cpu_count() or 8) // max(1, args.workers))
    tasks = [(s.name, seed, role, union, n_jobs)
             for s in bs.DEVELOPMENT_SCENARIOS
             for seed in SEEDS for role, union in ARMS
             if (s.name, seed, role, union) not in done]
    print(f"{len(done)} already done, {len(tasks)} to run, "
          f"{args.workers} workers x n_jobs={n_jobs}", flush=True)
    if not tasks:
        return 0

    write_header = not out.exists()
    fh = open(out, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if write_header:
        w.writeheader(); fh.flush()

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_one, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            w.writerow(row); fh.flush()          # durable per trial
            tag = "ERROR" if row["error"] else (
                f"S2={row['n_after_s2']} S3={'Y' if row['s3_ran'] else 'n'} "
                f"-> {row['n_final']} F1={row['f1']}")
            print(f"[{i}/{len(tasks)}] {row['scenario']:22} seed={row['seed']} "
                  f"{row['role']:10} union={str(row['union']):5} {tag}  "
                  f"({row['seconds']}s)", flush=True)
    fh.close()
    print(f"\nComplete in {(time.perf_counter()-t0)/3600:.2f} h -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
