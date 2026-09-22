"""
Can Step 3 be made to fire without the union rung's precision collapse?

Step 3 engages only when Step 2 hands it at least MIN_FEATURES_FOR_RFECV (30)
features. Step 2's output is capped by the consensus ceiling: the count of
features holding at least one vote in EVERY stage. The floor is clamped to that
ceiling, so step2_role='prefilter' can ask for 60 and still deliver 10.

Measured so far (results_role_comparison.csv, n_consensus_models=5):

    terminal            Step 3 fires   0%   mean F1 0.849
    prefilter                         29%            0.785
    terminal + union                  13%            0.755
    prefilter + union                 91%            0.411   <- precision 0.318

The union rung raises the ceiling by replacing the intersection with a union,
and precision collapses. This asks whether the model count raises the same
ceiling without that cost: every additional replicate is another chance for a
feature to clear the one-vote bar in each stage, so the intersection itself
should widen.

That lever was untestable until recently. Bootstrap seeds were derived as
model_seed + fold_idx * 1000 against model seeds spaced 100 apart, so model
i+10 reused the seed of model i at the next fold. Past ten models the extra
replicates were bootstrap duplicates and cast no independent votes, leaving the
ceiling flat. That aliasing is fixed, so raising the count now adds real votes.

Usage:
    python gate_experiment.py --out results_gate_experiment.csv
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

import benchmark_synthetic as bs                     # noqa: E402
from SelectOmics import SelectOmicsConfig, SelectOmicsPipeline   # noqa: E402


# One scenario where prefilter already wins, one where it loses badly, so a
# lever that only helps the easy case is visible as such.
SCENARIOS = ("omics_multiclass", "omics_standard")
SEEDS = (0, 1, 2)

# (label, step2_role, n_consensus_models, colsample)
# colsample=None leaves the configured default alone.
ARMS = (
    ("terminal_n5",        "terminal",   5,  None),
    ("prefilter_n5",       "prefilter",  5,  None),
    ("prefilter_n15",      "prefilter", 15,  None),
    ("prefilter_n30",      "prefilter", 30,  None),
    ("prefilter_n15_cs10", "prefilter", 15,  0.10),
)

FIELDS = ["scenario", "seed", "arm", "role", "n_models", "colsample",
          "n_after_s1", "n_after_s2", "ceiling_used", "s2_outcome",
          "s3_ran", "n_final", "s2_f1", "f1", "precision", "recall",
          "seconds", "error"]


def _one(name, seed, arm, role, n_models, colsample):
    t0 = time.perf_counter()
    row = dict.fromkeys(FIELDS, "")
    row.update(scenario=name, seed=seed, arm=arm, role=role,
               n_models=n_models, colsample="" if colsample is None else colsample)
    try:
        sc = next(s for s in bs.SCENARIOS if s.name == name)
        X, y, ti = bs.make_omics_dataset(sc, seed=seed)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=bs.BENCHMARK_SPEC["test_size"],
            stratify=y, random_state=seed)
        tmp = Path(tempfile.mkdtemp())
        csv_path = tmp / "d.csv"
        df = Xtr.copy()
        df["Class"] = ytr
        df.to_csv(csv_path, index=False)

        kw = dict(
            data_path=str(csv_path), target_column="Class", algorithm="XGB",
            output_dir=str(tmp / "o"), n_consensus_models=n_models,
            quick_tune_iterations=10, n_bootstrap=10, verbose=False,
            create_visualizations=False, save_intermediate_results=False,
            enable_step_evaluations=False, enable_final_test_evaluation=False,
            test_size=0.2, step2_role=role, allow_union_rung=False, n_jobs=1,
        )
        if colsample is not None:
            kw["step2_stage_colsample"] = colsample

        p = SelectOmicsPipeline(SelectOmicsConfig(**kw))
        p.load_data()
        p.run_step0_reference()
        p.run_step1_cleaning()
        row["n_after_s1"] = len(p.get_selected_features("step1"))

        p.run_step2_regularization()
        s2_cols = p.get_selected_features("step2")
        row["n_after_s2"] = len(s2_cols)
        # The ceiling actually reached, which is what gates Step 3.
        row["ceiling_used"] = p.results["step2"].get("consensus_floor_used", "")
        row["s2_outcome"] = p.results["step2"].get("consensus_outcome", "")
        row["s2_f1"] = round(bs.ground_truth_metrics(
            {int(c.split("_")[1]) for c in s2_cols}, ti)["f1"], 4)

        p.run_step3_wrapper()
        cols = p.get_selected_features("step3")
        row["s3_ran"] = not p.results["step3"].get("skipped", True)
        row["n_final"] = len(cols)
        g = bs.ground_truth_metrics({int(c.split("_")[1]) for c in cols}, ti)
        row.update(f1=round(g["f1"], 4), precision=round(g["precision"], 4),
                   recall=round(g["recall"], 4))
    except Exception:
        row["error"] = traceback.format_exc(limit=2).replace("\n", " | ")[:300]
    row["seconds"] = round(time.perf_counter() - t0, 1)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(_HERE / "results_gate_experiment.csv"))
    args = ap.parse_args()

    tasks = [(sc, sd, *arm) for sc in SCENARIOS for sd in SEEDS for arm in ARMS]
    print(f"{len(tasks)} runs: {len(SCENARIOS)} scenarios x {len(SEEDS)} seeds "
          f"x {len(ARMS)} arms", flush=True)

    out = Path(args.out)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for i, (name, seed, arm, role, n_models, cs) in enumerate(tasks, 1):
            row = _one(name, seed, arm, role, n_models, cs)
            w.writerow(row)
            fh.flush()
            print(f"[{i:>2}/{len(tasks)}] {name:<18} seed {seed} {arm:<20} "
                  f"S2={row['n_after_s2']:<5} S3ran={row['s3_ran']!s:<6} "
                  f"F1={row['f1']} ({row['seconds']}s)"
                  + (f"  ERROR {row['error'][:60]}" if row["error"] else ""),
                  flush=True)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
