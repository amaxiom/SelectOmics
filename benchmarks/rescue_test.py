"""
The case the recommendation exists for: Step 3 fires and over-prunes.

At n_consensus_models=15 with prefilter, Step 2 clears Step 3's gate and Step 3
cuts ~55 candidates to ~3. Its precision stays 1.000 -- every feature it keeps
is real -- but recall collapses and F1 fell 0.930 to 0.473 in the gate
experiment. Step 2's panel is the better one, and the recommendation should say
so. Nothing has tested that.
"""
import sys, tempfile, time, csv, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, ".."); sys.path.insert(0, ".")
from sklearn.model_selection import train_test_split
import benchmark_synthetic as bs
from SelectOmics import SelectOmicsConfig, SelectOmicsPipeline

rows = []
for seed in (0, 1, 2):
    t0 = time.perf_counter()
    sc = next(s for s in bs.SCENARIOS if s.name == "omics_standard")
    X, y, ti = bs.make_omics_dataset(sc, seed=seed)
    Xtr, _, ytr, _ = train_test_split(
        X, y, test_size=bs.BENCHMARK_SPEC["test_size"], stratify=y,
        random_state=seed)
    tmp = Path(tempfile.mkdtemp()); csvp = tmp / "d.csv"
    df = Xtr.copy(); df["Class"] = ytr; df.to_csv(csvp, index=False)

    p = SelectOmicsPipeline(SelectOmicsConfig(
        data_path=str(csvp), target_column="Class", algorithm="XGB",
        output_dir=str(tmp / "o"), n_consensus_models=15,
        quick_tune_iterations=10, n_bootstrap=20, verbose=False,
        create_visualizations=False, save_intermediate_results=False,
        enable_step3=True, step2_role="prefilter",
        enable_step_evaluations=True, enable_final_test_evaluation=False,
        test_size=0.2, n_jobs=1))
    res = p.run(validate=True)

    def f1(cols):
        return bs.ground_truth_metrics({int(c.split("_")[1]) for c in cols}, ti)

    s2 = f1(p.get_selected_features("step2"))
    last_cols = p.get_selected_features()
    last = f1(last_cols)
    rec = res.get("recommendation")
    rcols = list(rec["X_train"].columns) if rec else []
    r = f1(rcols) if rcols else {"f1": float("nan"), "recall": float("nan")}

    row = dict(seed=seed,
               s3_ran=not res.get("step3", {}).get("skipped", True),
               n_s2=len(p.get_selected_features("step2")), s2_f1=round(s2["f1"], 4),
               n_last=len(last_cols), last_f1=round(last["f1"], 4),
               last_recall=round(last["recall"], 4),
               rec_step=(rec or {}).get("step_id", "none"), n_rec=len(rcols),
               rec_f1=round(r["f1"], 4),
               rescued=bool(r["f1"] > last["f1"] + 1e-9),
               seconds=round(time.perf_counter() - t0, 1))
    rows.append(row)
    print(f"seed {seed}: S3ran={row['s3_ran']} S2={row['n_s2']}f F1={row['s2_f1']} | "
          f"last={row['n_last']}f F1={row['last_f1']} | rec={row['rec_step']} "
          f"{row['n_rec']}f F1={row['rec_f1']} | RESCUED={row['rescued']} "
          f"({row['seconds']}s)", flush=True)

with open("results_rescue_test.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("\nrescued:", sum(r["rescued"] for r in rows), "of", len(rows))
