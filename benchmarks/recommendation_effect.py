"""
Does the recommendation change how SelectOmics performs?

Every benchmark on disk recorded the LAST step's panel, because the
recommendation was computed and then never used. With it wired into run(), the
panel a user is told to take can differ from the one the pipeline ends on, so
the measured performance may differ too.

Two things are being tested.

1. Rescue. Step 3 returns only true features whenever it fires -- precision was
   1.000 in 19 of 21 firing runs -- but it can return far too few of them. On
   omics_standard with prefilter it cut 55 candidates to 3 and F1 fell from
   0.930 to 0.473. If the recommendation works, it should name Step 2 there and
   hand back the better panel.

2. Inflation. The recommendation ranks on weighted AUC and applies NO penalty
   for panel size, while Step 0 holds every feature. A larger panel often
   scores marginally better AUC, so the recommendation could systematically
   prefer Step 0 and make the panels far worse on ground-truth F1. That would
   be a defect in the ranking, and this is the experiment that would show it.

Reported per run: the last step's F1, the recommended step's F1, which step was
recommended, and both panel sizes.

Usage:
    python recommendation_effect.py --out results_recommendation_effect.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

import benchmark_synthetic as bs                                   # noqa: E402
from SelectOmics import SelectOmicsConfig, SelectOmicsPipeline     # noqa: E402

# omics_standard   : Step 3 over-prunes under prefilter; the rescue case.
# omics_multiclass : the cascade genuinely wins here.
# omics_high_dim   : prefilter lost badly on the old last-step measurement.
import os as _os

# Overridable so the same harness can run a narrow diagnostic or a broad sweep.
_ALL = ("omics_standard", "omics_multiclass", "omics_high_dim",
        "omics_genomics", "omics_lowcorr", "omics_imbalanced",
        "omics_imbalanced_std", "omics_tiny_n", "omics_interaction",
        "adversarial_easy", "null_control")
SCENARIOS = tuple(_os.environ.get("RE_SCENARIOS", "").split(","))     if _os.environ.get("RE_SCENARIOS") else     ("omics_standard", "omics_multiclass", "omics_high_dim")
if SCENARIOS == ("ALL",):
    SCENARIOS = _ALL
SEEDS = tuple(int(s) for s in _os.environ.get("RE_SEEDS", "0,1").split(","))
ROLES = tuple(_os.environ.get("RE_ROLES", "terminal,prefilter").split(","))

FIELDS = ["scenario", "seed", "role", "n_after_s2", "s3_ran",
          "last_step", "last_n", "last_f1", "last_precision", "last_recall",
          "rec_step", "rec_n", "rec_f1", "rec_precision", "rec_recall",
          "rec_quality", "rec_weighted_auc", "rescued", "seconds", "error"]


def _idx(cols):
    """Ground-truth indices from generated column names like 'feature_12'."""
    return {int(c.split("_")[1]) for c in cols}


def _one(name, seed, role):
    t0 = time.perf_counter()
    row = dict.fromkeys(FIELDS, "")
    row.update(scenario=name, seed=seed, role=role)
    try:
        sc = next(s for s in bs.SCENARIOS if s.name == name)
        X, y, ti = bs.make_omics_dataset(sc, seed=seed)
        Xtr, _, ytr, _ = train_test_split(
            X, y, test_size=bs.BENCHMARK_SPEC["test_size"], stratify=y,
            random_state=seed)
        tmp = Path(tempfile.mkdtemp())
        csv_path = tmp / "d.csv"
        df = Xtr.copy()
        df["Class"] = ytr
        df.to_csv(csv_path, index=False)

        p = SelectOmicsPipeline(SelectOmicsConfig(
            data_path=str(csv_path), target_column="Class", algorithm="XGB",
            output_dir=str(tmp / "o"), n_consensus_models=5,
            quick_tune_iterations=10, n_bootstrap=20, verbose=False,
            create_visualizations=False, save_intermediate_results=False,
            enable_step3=True, step2_role=role, allow_union_rung=False,
            # The whole point: these produce the recommendation.
            enable_step_evaluations=True, enable_final_test_evaluation=False,
            test_size=0.2, n_jobs=1))
        res = p.run(validate=True)

        row["n_after_s2"] = len(p.get_selected_features("step2"))
        row["s3_ran"] = not res.get("step3", {}).get("skipped", True)

        last = p.get_selected_features()
        g = bs.ground_truth_metrics(_idx(last), ti)
        row.update(last_step="last", last_n=len(last),
                   last_f1=round(g["f1"], 4),
                   last_precision=round(g["precision"], 4),
                   last_recall=round(g["recall"], 4))

        rec = res.get("recommendation")
        if rec is None:
            row["rec_step"] = "none"
        else:
            rcols = list(rec["X_train"].columns)
            gr = bs.ground_truth_metrics(_idx(rcols), ti)
            row.update(rec_step=rec["step_id"], rec_n=len(rcols),
                       rec_f1=round(gr["f1"], 4),
                       rec_precision=round(gr["precision"], 4),
                       rec_recall=round(gr["recall"], 4),
                       rec_quality=rec["quality"],
                       rec_weighted_auc=round(rec["weighted_auc"], 4))
            row["rescued"] = bool(gr["f1"] > g["f1"] + 1e-9)
    except Exception:
        row["error"] = traceback.format_exc(limit=2).replace("\n", " | ")[:300]
    row["seconds"] = round(time.perf_counter() - t0, 1)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",
                    default=str(_HERE / "results_recommendation_effect.csv"))
    args = ap.parse_args()

    tasks = [(s, sd, r) for s in SCENARIOS for sd in SEEDS for r in ROLES]
    print(f"{len(tasks)} runs", flush=True)

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for i, (name, seed, role) in enumerate(tasks, 1):
            row = _one(name, seed, role)
            w.writerow(row)
            fh.flush()
            print(f"[{i:>2}/{len(tasks)}] {name:<18} seed {seed} {role:<10} "
                  f"last {row['last_n']}f F1={row['last_f1']} -> "
                  f"rec {row['rec_step']} {row['rec_n']}f F1={row['rec_f1']} "
                  f"rescued={row['rescued']} ({row['seconds']}s)"
                  + (f"  ERROR {row['error'][:60]}" if row["error"] else ""),
                  flush=True)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
