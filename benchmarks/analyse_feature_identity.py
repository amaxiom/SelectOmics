"""
Which features recur, not merely how many.

Two runs returning twelve features each look identical in every summary column
while sharing none of the same twelve. A reproducibility claim lives or dies on
that distinction, and the Kuncheva index compresses it to a single number. This
script reports the underlying set behaviour directly:

  core          features selected by EVERY seed
  core_frac     core as a fraction of the mean panel size
  mean_jaccard  average pairwise overlap between seed pairs
  ever          features selected by at least one seed
  churn         ever / mean size -- how many distinct features the method
                cycles through to produce a panel of its typical size

A method with core_frac near 1.0 returns essentially the same panel every time.
One with high churn and low core_frac returns a panel of stable SIZE and
unstable CONTENT, which is the failure mode that a feature-count table cannot
show.

Usage
-----
    python analyse_feature_identity.py                        # synthetic
    python analyse_feature_identity.py --csv path/to/raw.csv  # any raw file
    python analyse_feature_identity.py --group layer method   # LGG-style
"""
from __future__ import annotations

import argparse
import ast
import sys
from itertools import combinations
from pathlib import Path
from typing import List, Optional, Set

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))


def parse_indices(raw) -> Set[int]:
    """Parse a selected_indices cell. Handles list-repr and comma-separated."""
    if not isinstance(raw, str) or not raw.strip():
        return set()
    text = raw.strip()
    if text.startswith("["):
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return set()
        if isinstance(value, (int, np.integer)):
            return {int(value)}
        return {int(v) for v in value}
    return {int(tok) for tok in text.split(",") if tok.strip()}


def summarise(df: pd.DataFrame, group: List[str], replicate: str) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group):
        sets = [s for s in g.sort_values(replicate)["_fset"] if s]
        if len(sets) < 2:
            continue
        core = set.intersection(*sets)
        ever = set.union(*sets)
        sizes = [len(s) for s in sets]
        mean_size = float(np.mean(sizes))

        jaccards = [
            len(a & b) / len(a | b)
            for a, b in combinations(sets, 2) if (a | b)
        ]

        record = dict(zip(group, key if isinstance(key, tuple) else (key,)))
        record.update({
            "n_replicates": len(sets),
            "mean_size": round(mean_size, 1),
            "core": len(core),
            "core_frac": round(len(core) / mean_size, 3) if mean_size else 0.0,
            "mean_jaccard": round(float(np.mean(jaccards)), 3) if jaccards else np.nan,
            "ever": len(ever),
            "churn": round(len(ever) / mean_size, 1) if mean_size else np.nan,
        })
        rows.append(record)
    return pd.DataFrame(rows)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(HERE / "results" / "benchmark_raw.csv"),
                    help="Raw benchmark CSV containing a selected_indices column.")
    ap.add_argument("--group", nargs="+", default=None,
                    help="Columns identifying one method-condition. "
                         "Default: scenario+method, or layer+method for LGG.")
    ap.add_argument("--replicate", default=None,
                    help="Column varying within a group. Default: seed, or fold.")
    ap.add_argument("--out", default=None, help="Write the table to this CSV.")
    args = ap.parse_args(argv)

    path = Path(args.csv)
    if not path.exists():
        print(f"No such file: {path}")
        return 1
    df = pd.read_csv(path)

    if "selected_indices" not in df.columns:
        print(f"{path.name} has no selected_indices column. Feature identity "
              f"was not recorded for this run; re-run the benchmark.")
        return 1

    group = args.group or (["layer", "method"] if "layer" in df.columns
                           else ["scenario", "method"])
    replicate = args.replicate or ("fold" if "fold" in df.columns else "seed")
    missing = [c for c in group + [replicate] if c not in df.columns]
    if missing:
        print(f"Missing column(s): {missing}. Available: {list(df.columns)}")
        return 1

    df["_fset"] = df["selected_indices"].apply(parse_indices)
    table = summarise(df, group, replicate)
    if table.empty:
        print("Nothing to summarise: fewer than two replicates per group.")
        return 1

    pd.set_option("display.width", 200)

    print("=" * 92)
    print(f"Feature identity across {replicate}s -- {path.name}")
    print("=" * 92)
    print("core = selected by EVERY replicate; churn = distinct features used "
          "per panel-worth\n")

    method_col = "method"
    overall = (table.groupby(method_col)
                    .agg(mean_size=("mean_size", "mean"),
                         core=("core", "mean"),
                         core_frac=("core_frac", "mean"),
                         mean_jaccard=("mean_jaccard", "mean"),
                         churn=("churn", "mean"))
                    .round(3)
                    .sort_values("core_frac", ascending=False))
    print(overall.to_string())

    print(f"\n{'-' * 92}\nPer condition\n{'-' * 92}")
    print(table.sort_values(group).to_string(index=False))

    if args.out:
        table.to_csv(args.out, index=False)
        print(f"\nWritten -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
