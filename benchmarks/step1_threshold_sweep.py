"""
Where does Step 1 start helping, and where does it destroy the result?

The 0.6.1 ablation showed Step 1 (the joint variance / correlation grid search)
behaving in opposite directions in two regimes:

    adversarial_easy  n=500  p=200   F1 0.536 -> 0.053 when Step 1 is added
    omics_tiny_n      n=50   p=500   F1 0.286 -> 0.380 (helps recovery)
                                     KI 0.705 -> 0.116 (destroys stability)

Two points is not enough to site a cutoff, and a guardrail that fires at the
wrong feature count is worse than none: it would disable a step users need.
This sweep measures the crossover across omics-realistic shapes so the
recommendation in SelectOmicsConfig.suggest() rests on a curve rather than a
guess.

For each shape it runs two configs on identical data:

    S2_only   Step 2 alone on the raw matrix
    S1->2     Step 1 then Step 2

and reports the difference in ground-truth recovery (F1) and cross-seed
stability (Kuncheva Index). A positive delta means Step 1 earned its place.

Usage
-----
    python step1_threshold_sweep.py                  # full grid
    python step1_threshold_sweep.py --seeds 3        # fewer seeds
    python step1_threshold_sweep.py --max-p 5000     # skip the slowest row
    python step1_threshold_sweep.py --workers 8      # cap parallelism

Results are written to results_step1_sweep/ and the run is resumable: a shape
already present in the raw CSV is skipped.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import benchmark_synthetic as bs
from benchmark_synthetic import (
    OmicsScenario,
    ground_truth_metrics,
    kuncheva_stability,
    make_omics_dataset,
)

logger = logging.getLogger("step1_sweep")

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
# Spans the low-dimensional regime where Step 1 was catastrophic through to
# genomics scale. n/p is the quantity the recommendation is likely to key on,
# so the grid varies it deliberately rather than sweeping p alone.
#
#   (n, p, label)
SHAPES: List[Tuple[int, int, str]] = [
    (500, 200,   "adversarial control"),     # n/p 2.50  Step 1 known harmful
    (200, 500,   "low-dim panel"),           # n/p 0.40
    (50,  500,   "pilot study"),             # n/p 0.10  = omics_tiny_n
    (100, 1000,  "targeted panel"),          # n/p 0.10
    (100, 2000,  "small omics"),             # n/p 0.05
    (200, 5000,  "proteomics scale"),        # n/p 0.04
    (100, 5000,  "omics, small cohort"),     # n/p 0.02
    (200, 10000, "genomics scale"),          # n/p 0.02
    (100, 10000, "genomics, small cohort"),  # n/p 0.01
]

# Correlation axis.
#
# The shape grid above holds within_block_corr at 0.70 throughout, so it cannot
# explain the one case where Step 1 was catastrophic: adversarial_easy, which
# differs in shape AND in correlation (rho=0.10). Step 1's second filter drops
# one of every correlated pair, so rho is exactly the knob that should govern
# whether it helps or harms. Sweep it directly at three shapes.
RHO_VALUES: List[float] = [0.10, 0.30, 0.50, 0.70, 0.90]
RHO_SHAPES: List[Tuple[int, int, str]] = [
    (500, 200,  "adversarial shape"),
    (100, 2000, "small omics"),
    (100, 5000, "omics, small cohort"),
]

CONFIGS = [
    # label,    step1, step2
    ("S2_only",  False, True),
    ("S1->2",    True,  True),
]

OUT_DIR = Path(__file__).resolve().parent / "results_step1_sweep"
RAW_CSV = OUT_DIR / "step1_sweep_raw.csv"
RHO_CSV = OUT_DIR / "step1_rho_raw.csv"


def _make_scenario(n: int, p: int, rho: float = 0.70) -> OmicsScenario:
    """Build a scenario matching the benchmark's own generative settings."""
    # 2% informative, matching omics_tiny_n's density, floored so the smallest
    # shapes still carry a signal a method could in principle find.
    n_informative = max(5, min(p // 50, p // 20))
    return OmicsScenario(
        name=f"n{n}_p{p}_rho{rho:g}",
        n_samples=n,
        n_features=p,
        n_informative=n_informative,
        block_size=20,
        within_block_corr=rho,
        class_imbalance=0.50,
        signal_strength=1.2,
        split="sweep",
        description=f"n={n} p={p} rho={rho:g} sweep point",
    )


def _one_trial(args) -> Dict:
    """Run one (shape, rho, config, seed). Executed in a worker process."""
    n, p, label, cfg_label, s1, s2, seed, rho = args
    warnings.filterwarnings("ignore")
    logging.getLogger("SelectOmics").setLevel(logging.ERROR)
    bs.set_compute(n_jobs=1, use_gpu=False)

    sc = _make_scenario(n, p, rho)
    X, y, true_idx = make_omics_dataset(sc, seed=seed * 97 + 13)

    from sklearn.model_selection import train_test_split
    X_tr, _, y_tr, _ = train_test_split(
        X.values, y, test_size=bs.BENCHMARK_SPEC["test_size"],
        stratify=y, random_state=seed)

    mss = bs.SelectOmicsMultiStepSelector(fast_mode=True)
    t0 = time.perf_counter()
    selected, skipped = mss.run_config(
        X_tr, y_tr, sc, seed, list(X.columns),
        enable_step1=s1, enable_step2=s2,
        enable_step3=False,
    )
    elapsed = time.perf_counter() - t0

    # run_config returns column indices, not names.
    sel_idx = {int(i) for i in selected}
    gt = ground_truth_metrics(sel_idx, set(true_idx))

    return {
        "n": n, "p": p, "n_over_p": round(n / p, 4), "regime": label,
        "rho": rho, "config": cfg_label, "seed": seed,
        "n_selected": len(sel_idx),
        "f1": round(gt["f1"], 4),
        "precision": round(gt["precision"], 4),
        "recall": round(gt["recall"], 4),
        "time_seconds": round(elapsed, 1),
        "selected_indices": ",".join(str(i) for i in sorted(sel_idx)),
        "n_informative": len(true_idx),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3,
                    help="Seeds per (shape, config). Default 3.")
    ap.add_argument("--max-p", type=int, default=None,
                    help="Skip shapes with more features than this.")
    ap.add_argument("--workers", type=int, default=-1,
                    help="Parallel workers; -1 uses every core. Default -1.")
    ap.add_argument("--fresh", action="store_true",
                    help="Ignore any existing raw CSV and rerun everything.")
    ap.add_argument("--mode", choices=("shape", "rho"), default="shape",
                    help="'shape' sweeps n and p at rho=0.70; 'rho' sweeps "
                         "within-block correlation at three fixed shapes.")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_csv = RAW_CSV if args.mode == "shape" else RHO_CSV
    if args.mode == "shape":
        grid = [(n, p, label, 0.70)
                for n, p, label in SHAPES
                if args.max_p is None or p <= args.max_p]
    else:
        grid = [(n, p, label, rho)
                for n, p, label in RHO_SHAPES
                for rho in RHO_VALUES
                if args.max_p is None or p <= args.max_p]

    done = set()
    existing = pd.DataFrame()
    if raw_csv.exists() and not args.fresh:
        existing = pd.read_csv(raw_csv)
        done = {(r.n, r.p, r.config, r.seed, getattr(r, "rho", 0.70))
                for r in existing.itertuples()}
        logger.info("Resuming: %d trials already recorded.", len(done))

    tasks = []
    for n, p, label, rho in grid:
        for cfg_label, s1, s2 in CONFIGS:
            for seed in range(args.seeds):
                if (n, p, cfg_label, seed, rho) in done:
                    continue
                tasks.append((n, p, label, cfg_label, s1, s2, seed, rho))

    if not tasks:
        logger.info("Nothing to do; all trials present.")
    else:
        n_workers = os.cpu_count() if args.workers == -1 else max(1, args.workers)
        # Longest trials first so the tail does not stall a free core.
        tasks.sort(key=lambda t: -t[1])
        logger.info("%d trials over %d grid points on %d workers (mode=%s)",
                    len(tasks), len(grid), n_workers, args.mode)

        records = list(existing.to_dict("records")) if len(existing) else []
        t_start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_one_trial, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    rec = fut.result()
                except Exception as exc:
                    t = futures[fut]
                    logger.warning("  FAILED n=%d p=%d %s seed=%d: %s",
                                   t[0], t[1], t[3], t[6], exc)
                    continue
                records.append(rec)
                logger.info("  [%d/%d] n=%-4d p=%-6d rho=%.2f %-8s seed=%d  "
                            "F1=%.3f  n_sel=%-5d (%.0fs)",
                            i, len(tasks), rec["n"], rec["p"], rec["rho"],
                            rec["config"], rec["seed"], rec["f1"],
                            rec["n_selected"], rec["time_seconds"])
                # Checkpoint after every trial: the p=10000 rows are expensive.
                pd.DataFrame(records).to_csv(raw_csv, index=False)
        logger.info("Wall clock: %.1f min", (time.perf_counter() - t_start) / 60)

    raw = pd.read_csv(raw_csv)
    summary = summarise(raw, by_rho=(args.mode == "rho"))
    out = OUT_DIR / (f"step1_{args.mode}_summary.csv")
    summary.to_csv(out, index=False)
    logger.info("\n%s", summary.to_string(index=False))
    logger.info("\nSummary -> %s", out)
    return 0


def summarise(raw: pd.DataFrame, by_rho: bool = False) -> pd.DataFrame:
    """Per grid point: does Step 1 help or hurt recovery and stability?"""
    keys = ["n", "p", "rho"] if by_rho else ["n", "p"]
    if "rho" not in raw.columns:
        raw = raw.assign(rho=0.70)
    rows = []
    for key, g in raw.groupby(keys):
        n, p = (key[0], key[1]) if isinstance(key, tuple) else (key, None)
        rec = {"n": n, "p": p, "n_over_p": round(n / p, 4),
               "regime": g["regime"].iloc[0]}
        if by_rho:
            rec["rho"] = key[2]
        for cfg in ("S2_only", "S1->2"):
            sub = g[g["config"] == cfg]
            if sub.empty:
                continue
            sets = [set(int(i) for i in str(s).split(",") if i != "")
                    for s in sub["selected_indices"]]
            ki = kuncheva_stability(sets, int(p))
            rec[f"{cfg}_f1"] = round(sub["f1"].mean(), 3)
            rec[f"{cfg}_KI"] = round(ki, 3) if ki == ki else np.nan
            rec[f"{cfg}_n"] = int(round(sub["n_selected"].mean()))
            rec[f"{cfg}_secs"] = round(sub["time_seconds"].mean(), 1)
        if "S1->2_f1" in rec and "S2_only_f1" in rec:
            rec["step1_f1_delta"] = round(rec["S1->2_f1"] - rec["S2_only_f1"], 3)
            if rec.get("S1->2_KI") == rec.get("S1->2_KI") and \
               rec.get("S2_only_KI") == rec.get("S2_only_KI"):
                rec["step1_KI_delta"] = round(rec["S1->2_KI"] - rec["S2_only_KI"], 3)
            rec["step1_verdict"] = (
                "helps" if rec["step1_f1_delta"] > 0.02 else
                "hurts" if rec["step1_f1_delta"] < -0.02 else "neutral")
        rows.append(rec)
    df = pd.DataFrame(rows)
    sort_by = ["p", "rho"] if by_rho else ["n_over_p"]
    return df.sort_values(sort_by).reset_index(drop=True)


if __name__ == "__main__":
    sys.exit(main())
