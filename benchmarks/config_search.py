"""
Four-phase configuration search for SelectOmics, over parameters that work.

This replaces the search in ``SelectOmics_Config_Optimization.ipynb``, which
produced the values in ``SelectOmicsConfig.omics()``. Those values need
re-deriving, for two reasons.

Note on names: the dimension names below are the *old* search's, kept so the
historical account stays checkable against the notebook that produced them.
``shap_upper`` and ``step4_target_retention`` belonged to the explainer step,
which has since been removed from the pipeline entirely; the search space in
this module no longer contains them.

Four of its ten dimensions did nothing
--------------------------------------
``l1_upper``, ``l2_upper``, ``rfecv_upper`` and ``shap_upper`` set the upper
bound of a binary search over an importance percentile. That percentile was
inert: tree importance is exactly 0.0 for every feature never used in a split,
which is 98 to 99.8% of features on omics-shaped data, so ``np.percentile``
returned 0.0 across the whole configured range and the kept count was identical
at every setting.

The search measured this itself. Phase-1 sensitivity ranked all four last, at
Spearman r of 0.045, -0.038, -0.002 and 0.002 with p-values from 0.52 to 0.98,
against ``consensus_threshold`` at r = -0.837. Two of them sat at r = +-0.002,
which is what a parameter with no effect looks like. It nonetheless reported
optima for them to three decimal places, and those values shipped.

Forty per cent of a 4500-evaluation budget went on noise, and the remaining
dimensions were fitted around that noise. That is why every tuned value needs
re-deriving, not just the four that were dropped.

The dominant parameter was not in the space at all
--------------------------------------------------
``min_features_floor`` is what actually determines panel size: every step
relaxes its consensus requirement one vote at a time until the floor is met, so
the floor decides how many features come out and ``consensus_threshold`` only
decides where the relaxation starts. Measured end to end, moving it from 5 to
10 changed ground-truth F1 from 0.705 to 0.777 and cross-seed Jaccard from
0.414 to 0.553. It was fixed at its default throughout the original search.

Both it and ``step2_target_retention`` (which exposes a value Step 2 previously
hardcoded) are in the space here.

Usage
-----
    python config_search.py --workers 4
    python config_search.py --workers 4 --phase1-trials 120   # cheaper
    python config_search.py --report-only                     # read results

The search is resumable: every evaluation is appended to its phase CSV as it
completes, and a re-run skips work already recorded. Interrupt it freely.

Set ``--workers`` to leave cores free if a benchmark is running: each worker
runs one pipeline, and the pipeline's own thread budget is divided among them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmark_synthetic import (  # noqa: E402
    DEVELOPMENT_SCENARIOS, SCENARIOS, OmicsScenario,
    make_omics_dataset, ground_truth_metrics, downstream_auc,
    kuncheva_stability, _SO_AVAILABLE,
)

if not _SO_AVAILABLE:
    raise ImportError("SelectOmics is required for the configuration search.")

from SelectOmics import SelectOmicsConfig, SelectOmicsPipeline  # noqa: E402


# ===========================================================================
# Parameter space
# ===========================================================================

# Eight live dimensions, against the original ten of which four were inert.
#
# min_features_floor is a 'choice' rather than a uniform range because the
# measured response is not monotonic: 10 beat both 5 and 20, and 20 and 30
# (the only settings that let Step 3 run at all) were worse than 5. A
# low-discrepancy sample over a continuous range would waste evaluations in
# the 20-to-30 region already known to be bad.
#
# consensus_threshold is NOT here. It set where the graduated relaxation
# started, and the relaxation stops at whichever level first satisfies the
# feature floor, so starting lower only skipped rungs on the way to the same
# answer. Measured across omics_standard and omics_genomics at two seeds,
# thresholds of 0.4, 0.6, 0.8 and 1.0 returned the identical feature SET.
# It has been removed from the package; min_consensus replaces it as a FLOOR
# on agreement rather than a starting point, and is searched here.
PARAM_SPACE: Dict[str, tuple] = {
    'min_consensus':          ('choice',  [0, 40, 60, 80, 100]),   # percent; 0 = None
    'stability_threshold':    ('uniform', 0.20, 0.80),
    'min_features_floor':     ('choice',  [5, 8, 10, 12, 15, 20]),
    'n_consensus_models':     ('choice',  [3, 5, 10]),
    'quick_tune_iterations':  ('choice',  [10, 20, 30]),
    'step2_target_retention': ('uniform', 0.30, 1.00),
    # Sets the size of the pool step2_target_retention then cuts within.
    # Measured inverse: LOWER colsample means MORE distinct features are
    # used across the ensemble, so a low value WIDENS the pool.
    'step2_stage_colsample':  ('choice',  [0.1, 0.3, 0.6, 1.0]),
    'step3_target_retention': ('uniform', 0.30, 0.70),
}


def _min_consensus(params: dict):
    """Percent-encoded so the Halton sampler can treat it as a choice."""
    pct = int(params.get('min_consensus', 0))
    return None if pct <= 0 else pct / 100.0

def _so_default(field: str):
    """Read a live default off the dataclass, so this file cannot drift."""
    from dataclasses import MISSING
    f = SelectOmicsConfig.__dataclass_fields__[field]
    if f.default is not MISSING:
        return f.default
    return f.default_factory()          # pragma: no cover - none currently


def _shipping_default() -> dict:
    """
    The arm every candidate must beat: the configuration a user gets by
    writing SelectOmicsConfig(data_path=..., target_column=...).

    Hand-copying these was already wrong once -- it carried min_features_floor
    10 against a shipped 15, and min_consensus 0 (None) against a shipped 0.4,
    so the 'default' arm was not the default and the search was measuring its
    candidates against a config nobody runs. Read them off the dataclass.
    """
    floor = _so_default('min_features_floor')
    mc = _so_default('min_consensus')
    stab = _so_default('stability_threshold')
    return {
        # min_consensus is percent-encoded here; 0 means None.
        'min_consensus': 0 if mc is None else int(round(mc * 100)),
        # stability_threshold defaults to None and __post_init__ resolves it.
        'stability_threshold': 0.6 if stab is None else float(stab),
        'min_features_floor': int(floor),
        'n_consensus_models': int(_so_default('n_consensus_models')),
        # The one deliberate deviation: PARAM_SPACE samples tuning iterations
        # from [10, 20, 30], so the shipped 50 is outside the search space
        # entirely. Using the cheapest rung keeps the baseline comparable with
        # the candidates instead of giving it a budget none of them can have.
        'quick_tune_iterations': 10,
        'step2_target_retention': float(_so_default('step2_target_retention')),
        'step2_stage_colsample': float(_so_default('step2_stage_colsample')),
        'step3_target_retention': float(_so_default('step3_target_retention')),
    }


BASELINE_CONFIGS: Dict[str, dict] = {
    # The shipping defaults. The search must beat this to justify a preset.
    'default': _shipping_default(),
    # The surviving values from the previous search (h_071), carried forward
    # so the new run can be compared against what currently ships. Its four
    # percentile bounds and its consensus_threshold are simply gone.
    'h_071_survivors': {
        'min_consensus': 0, 'stability_threshold': 0.259,
        'min_features_floor': 5, 'n_consensus_models': 5,
        'quick_tune_iterations': 20, 'step2_target_retention': 0.6, 'step2_stage_colsample': 0.3,
        'step3_target_retention': 0.354,
    },
    # h_071's thresholds at the new floor default, isolating whether its
    # advantage survives once the dominant parameter is set sensibly.
    'h_071_at_floor_10': {
        'min_consensus': 0, 'stability_threshold': 0.259,
        'min_features_floor': 10, 'n_consensus_models': 5,
        'quick_tune_iterations': 20, 'step2_target_retention': 0.6, 'step2_stage_colsample': 0.3,
        'step3_target_retention': 0.354,
    },
    # Unanimity or nothing. This is the configuration users reach for when
    # they want a defensible panel rather than a large one, and it is only
    # expressible now that min_consensus binds.
    'strict': {
        'min_consensus': 100, 'stability_threshold': 0.8,
        'min_features_floor': 5, 'n_consensus_models': 10,
        'quick_tune_iterations': 20, 'step2_target_retention': 0.4, 'step2_stage_colsample': 0.3,
        'step3_target_retention': 0.4,
    },
    'lenient': {
        'min_consensus': 0, 'stability_threshold': 0.4,
        'min_features_floor': 15, 'n_consensus_models': 5,
        'quick_tune_iterations': 10, 'step2_target_retention': 0.75, 'step2_stage_colsample': 0.3,
        'step3_target_retention': 0.7,
    },
}

_HALTON_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]


def _halton_val(i: int, base: int) -> float:
    f, r = 1.0, 0.0
    k = i + 1
    while k > 0:
        f /= base
        r += f * (k % base)
        k //= base
    return r


def sample_params_halton(idx: int) -> dict:
    """Sample one config from the Halton low-discrepancy sequence."""
    p = {}
    for d, (key, spec) in enumerate(PARAM_SPACE.items()):
        x = _halton_val(idx, _HALTON_PRIMES[d])
        if spec[0] == 'uniform':
            p[key] = round(float(spec[1] + x * (spec[2] - spec[1])), 3)
        else:
            choices = spec[1]
            p[key] = int(choices[min(int(x * len(choices)), len(choices) - 1)])
    return p


def generate_neighbours(params: dict, n: int, rng: np.random.RandomState,
                        scale: float = 0.12) -> List[dict]:
    """Random perturbations of params, clipped to PARAM_SPACE bounds."""
    result = []
    for _ in range(n):
        new_p = {}
        for key, spec in PARAM_SPACE.items():
            v = params[key]
            if spec[0] == 'uniform':
                lo, hi = spec[1], spec[2]
                new_p[key] = round(
                    float(np.clip(v + rng.normal(0, scale * (hi - lo)), lo, hi)), 3)
            else:
                choices = spec[1]
                new_p[key] = int(rng.choice(choices)) if rng.random() < 0.25 else int(v)
        result.append(new_p)
    return result


# ===========================================================================
# Trial execution
# ===========================================================================

def build_so_config(params: dict, data_path: str, out_dir: str, seed: int,
                    n_jobs: int) -> SelectOmicsConfig:
    """Construct a SelectOmicsConfig from a search parameter dict."""
    return SelectOmicsConfig(
        data_path=data_path,
        target_column='target',
        algorithm='XGB',
        n_consensus_models=int(params['n_consensus_models']),
        random_seed=int(seed),
        output_dir=out_dir,
        test_size=0.20,
        verbose=False,
        create_visualizations=False,
        save_intermediate_results=False,
        enable_step_evaluations=False,
        enable_final_test_evaluation=False,
        enable_step3=True,
        n_jobs=n_jobs,
        quick_tune_iterations=int(params['quick_tune_iterations']),
        min_consensus=_min_consensus(params),
        stability_threshold=float(params['stability_threshold']),
        min_features_floor=int(params['min_features_floor']),
        step2_target_retention=float(params['step2_target_retention']),
        step2_stage_colsample=float(params['step2_stage_colsample']),
        step3_target_retention=float(params['step3_target_retention']),
    )


def run_so_trial(params: dict, scenario: OmicsScenario, seed: int,
                 config_name: str, n_jobs: int) -> dict:
    """Run SelectOmics on one (config, scenario, seed)."""
    data_seed = seed * 97 + 13
    X, y, true_idx = make_omics_dataset(scenario, seed=data_seed)

    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X.values, y, test_size=0.25, stratify=y, random_state=seed
    )
    scaler = MinMaxScaler()
    X_tr = scaler.fit_transform(X_tr_raw)
    X_te = scaler.transform(X_te_raw)

    feature_names = list(X.columns)
    df_tr = pd.DataFrame(X_tr_raw, columns=feature_names)
    df_tr['target'] = y_tr

    t0 = time.perf_counter()
    selected: set = set()
    status = 'ok'
    step_state = ''

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'data.csv')
            out_dir = os.path.join(tmpdir, 'so_out')
            df_tr.to_csv(csv_path, index=False)

            cfg = build_so_config(params, csv_path, out_dir, seed, n_jobs)
            pipeline = SelectOmicsPipeline(cfg)
            pipeline.run(validate=False)

            final_cols = pipeline.get_selected_features()
            name_to_idx = {n: i for i, n in enumerate(feature_names)}
            selected = {name_to_idx[c] for c in final_cols if c in name_to_idx}

            # Record whether the later steps did anything. A config that scores
            # well only because Steps 3 and 4 never ran is a different finding
            # from one that scores well with them active.
            s3 = pipeline.results.get('step3', {})
            step_state = (
                f"s3={'run' if not s3.get('skipped', True) else 'skip'}"
            )
    except Exception as exc:
        status = f'{type(exc).__name__}: {str(exc)[:80]}'

    elapsed = round(time.perf_counter() - t0, 1)
    gt = ground_truth_metrics(selected, true_idx)
    auc = downstream_auc(X_tr, y_tr, X_te, y_te, selected, seed=seed)

    return {
        'config': config_name,
        'scenario': scenario.name,
        'n_samples': scenario.n_samples,
        'n_features': scenario.n_features,
        'n_informative': scenario.n_informative,
        'seed': seed,
        'f1': round(gt['f1'], 4),
        'precision': round(gt['precision'], 4),
        'recall': round(gt['recall'], 4),
        'jaccard': round(gt['jaccard'], 4),
        'test_auc': round(float(auc), 4) if not np.isnan(auc) else np.nan,
        'n_selected': len(selected),
        'elapsed': elapsed,
        'status': status,
        'step_state': step_state,
        'kuncheva_stability': np.nan,
        'selected_indices': json.dumps(sorted(selected)),
        **{k: params.get(k, np.nan) for k in PARAM_SPACE},
    }


def _worker(task):
    params, scenario, seed, config_name, n_jobs = task
    return run_so_trial(params, scenario, seed, config_name, n_jobs)


# ===========================================================================
# Checkpointed, parallel scan
# ===========================================================================

def _done_key(config: str, scenario: str, seed: int) -> str:
    return f'{config}||{scenario}||{seed}'


def _build_done_set(df: pd.DataFrame) -> set:
    if df.empty:
        return set()
    return {_done_key(c, s, i) for c, s, i
            in zip(df['config'], df['scenario'], df['seed'])}


def recompute_ki(csv_path: Path) -> None:
    """Fill kuncheva_stability for every (config, scenario) with >= 2 seeds."""
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    for (cfg, scen), grp in df.groupby(['config', 'scenario']):
        sets = []
        for raw in grp['selected_indices']:
            try:
                sets.append(set(json.loads(raw)))
            except Exception:
                pass
        if len(sets) < 2:
            continue
        n_feat = int(grp['n_features'].iloc[0])
        ki = kuncheva_stability(sets, n_feat)
        df.loc[(df['config'] == cfg) & (df['scenario'] == scen),
               'kuncheva_stability'] = (
            round(float(ki), 4) if not np.isnan(ki) else np.nan)
    df.to_csv(csv_path, index=False)


def run_scan(trial_list: Sequence[Tuple[str, dict]],
             scenarios: Sequence[OmicsScenario],
             seeds: Sequence[int],
             save_path: Path,
             phase_label: str,
             workers: int,
             n_jobs: int) -> pd.DataFrame:
    """Run all (config, scenario, seed) combinations, resumable and parallel."""
    existing = pd.read_csv(save_path) if save_path.exists() else pd.DataFrame()
    done = _build_done_set(existing)

    tasks = [
        (params, scenario, seed, name, n_jobs)
        for name, params in trial_list
        for scenario in scenarios
        for seed in seeds
        if not _done_key(name, scenario.name, seed) in done
    ]
    n_total = len(trial_list) * len(scenarios) * len(seeds)
    n_skip = n_total - len(tasks)

    print(f'\n[{phase_label}] {n_total:,} evals, {n_skip:,} already done, '
          f'{len(tasks):,} to run on {workers} worker(s)', flush=True)
    if not tasks:
        recompute_ki(save_path)
        return pd.read_csv(save_path)

    t0 = time.time()
    n_err = 0
    write_header = not save_path.exists()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            pd.DataFrame([rec]).to_csv(save_path, mode='a',
                                       header=write_header, index=False)
            write_header = False
            if rec['status'] != 'ok':
                n_err += 1
            rate = (time.time() - t0) / i
            eta = rate * (len(tasks) - i) / 60.0
            flag = '' if rec['status'] == 'ok' else '  !! ' + rec['status'][:40]
            print(f'  [{i:4d}/{len(tasks)}] {rec["config"][:20]:20s} '
                  f'{rec["scenario"]:18s} seed={rec["seed"]} '
                  f'F1={rec["f1"]:.3f} n={rec["n_selected"]:4d} '
                  f'{rec["step_state"]:22s} ({rec["elapsed"]:.0f}s) '
                  f'ETA {eta:.0f}m{flag}', flush=True)

    recompute_ki(save_path)
    print(f'[{phase_label}] complete in {(time.time() - t0) / 60:.1f} min, '
          f'{n_err} errors', flush=True)
    return pd.read_csv(save_path)


# Objective weights.
#
# Rebalanced from the original 0.50 / 0.30 / 0.20. What SelectOmics has to
# deliver is the RIGHT features (F1 against ground truth) and the SAME features
# across seeds (Kuncheva); reduction is not in the objective at all and should
# not be. Those two are now weighted equally.
#
# AUC is kept only as a guard against a config that scores well on both while
# destroying predictive signal. It earns a small weight because it is nearly
# saturated in this regime -- 0.98 to 0.99 across almost every config in the
# previous search -- so it discriminates very little and mostly adds a constant.
W_F1, W_KI, W_AUC = 0.45, 0.45, 0.10


def aggregate_configs(df: pd.DataFrame, w_f1=W_F1, w_ki=W_KI,
                      w_auc=W_AUC) -> pd.DataFrame:
    """Composite objective over ground truth, stability, and predictive AUC."""
    rows = []
    for config_name, grp in df.groupby('config'):
        mean_f1 = float(grp['f1'].mean())
        mean_auc = float(grp['test_auc'].mean())
        mean_ki = float(grp['kuncheva_stability'].mean())
        composite = w_f1 * mean_f1 + w_ki * (mean_ki + 1.0) / 2.0 + w_auc * mean_auc
        rows.append({
            'config': config_name,
            'mean_f1': round(mean_f1, 4),
            'mean_auc': round(mean_auc, 4),
            'mean_ki': round(mean_ki, 4),
            'n_sel_mean': round(float(grp['n_selected'].mean()), 1),
            'elapsed_mean': round(float(grp['elapsed'].mean()), 1),
            'composite': round(composite, 4),
            **{k: grp[k].iloc[0] for k in PARAM_SPACE if k in grp.columns},
        })
    return (pd.DataFrame(rows).sort_values('composite', ascending=False)
              .reset_index(drop=True))


def degeneracy(agg: pd.DataFrame) -> dict:
    """
    How much of the search space actually resolved into distinct outcomes?

    The first run of this search produced 205 configs and only 16 distinct
    composite values, with 12 configs tied for first spanning the ENTIRE range
    of five of the eight parameters. A phase that cannot rank cannot shortlist,
    and anything built on that shortlist inherits an arbitrary slice of a tie.
    This makes the problem visible instead of letting it pass as a ranking.
    """
    n = len(agg)
    distinct = agg['composite'].nunique()
    top = agg[agg['composite'] == agg['composite'].max()]
    return {
        'n_configs': n,
        'distinct_outcomes': distinct,
        'resolution': round(distinct / n, 3) if n else 0.0,
        'tied_at_top': len(top),
        'largest_tie': int(agg['composite'].value_counts().max()) if n else 0,
    }


def shortlist(agg: pd.DataFrame, n_top: int) -> List[str]:
    """
    Take the best n_top configs, spreading the choice across ties.

    ``agg.head(n)`` is wrong when the leading configs are tied: it returns
    whatever pandas happened to sort first, which within a 12-way tie is an
    arbitrary slice and silently discards the parameter diversity that made
    the tie interesting. Here, tied groups are entered in turn and members are
    drawn to maximise how many distinct values of the discrete parameters make
    it through, so the next phase re-tests genuinely different configurations.
    """
    picked: List[str] = []
    discrete = [k for k, spec in PARAM_SPACE.items() if spec[0] == 'choice']

    for _, group in agg.groupby('composite', sort=False):
        if len(picked) >= n_top:
            break
        room = n_top - len(picked)
        if len(group) <= room:
            picked.extend(group['config'].tolist())
            continue

        # Greedy spread: repeatedly take the row adding the most unseen
        # discrete-parameter values, breaking ties by original order.
        seen = {k: set() for k in discrete}
        remaining = group.copy()
        while room > 0 and not remaining.empty:
            gain = remaining.apply(
                lambda r: sum(r[k] not in seen[k] for k in discrete
                              if k in remaining.columns),
                axis=1,
            )
            idx = gain.idxmax()
            row = remaining.loc[idx]
            picked.append(row['config'])
            for k in discrete:
                if k in remaining.columns:
                    seen[k].add(row[k])
            remaining = remaining.drop(index=idx)
            room -= 1

    return picked[:n_top]


def sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Spearman correlation of each parameter with the composite objective.

    Parameters with no variation at this phase are reported as 'not varied'
    rather than dropped. The funnel can eliminate a parameter's variation
    entirely -- Phase 3's shortlist was 15 configs all sharing one
    min_consensus value, because every Phase 2 leader had it -- and silently
    omitting such a parameter reads as "measured and unimportant" when the
    truth is "not measurable here". A later phase confirming nothing about a
    parameter is a fact the reader needs.
    """
    from scipy.stats import spearmanr
    agg = aggregate_configs(df)
    rows = []
    for key in PARAM_SPACE:
        if key not in agg.columns:
            continue
        if agg[key].nunique() < 2:
            rows.append({'parameter': key, 'spearman_r': np.nan,
                         'p_value': np.nan, 'abs_r': -1.0,
                         'note': f'not varied (all {agg[key].iloc[0]})'})
            continue
        r, p = spearmanr(agg[key], agg['composite'])
        rows.append({'parameter': key, 'spearman_r': round(float(r), 3),
                     'p_value': round(float(p), 4),
                     'abs_r': round(abs(float(r)), 3), 'note': ''})
    return (pd.DataFrame(rows).sort_values('abs_r', ascending=False)
              .reset_index(drop=True))


# ===========================================================================
# Main
# ===========================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workers', type=int, default=4,
                    help='Parallel pipelines. Leave cores free if a benchmark '
                         'is running concurrently. Default 4.')
    ap.add_argument('--total-cores', type=int, default=os.cpu_count() or 8,
                    help='Cores to divide among workers for each pipeline.')
    ap.add_argument('--phase1-trials', type=int, default=200)
    ap.add_argument('--phase2-top', type=int, default=30)
    ap.add_argument('--phase3-top', type=int, default=15)
    ap.add_argument('--phase4-top', type=int, default=5)
    ap.add_argument('--phase4-perturb', type=int, default=20)
    ap.add_argument('--out', default=str(_HERE / 'opt_results_0_7_0'))
    ap.add_argument('--seed', type=int, default=99)
    ap.add_argument('--report-only', action='store_true',
                    help='Summarise existing CSVs without running anything.')
    ap.add_argument('--phases', default='1234',
                    help='Which phases to run, e.g. "12" or "4".')
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_jobs = max(1, args.total_cores // max(1, args.workers))

    seeds_fast = [0, 1, 2]
    seeds_full = [0, 1, 2, 3, 4]

    # Phase 1 runs on TWO scenarios, not one.
    #
    # The first run of this search used omics_tiny_n alone and could not rank:
    # 205 configs collapsed to 16 distinct outcomes, and the 12 tied for first
    # differed across the whole range of five parameters. omics_tiny_n is also
    # the scenario that most disagrees with the others about the dominant
    # parameter -- it prefers min_features_floor 5 where the five-scenario
    # sweep prefers 10 -- so shortlisting on it alone selects for overfitting
    # to a single small dataset.
    #
    # omics_imbalanced is added because it is cheap (p=1000) and differs on
    # exactly the axis that matters: 15 informative features against
    # omics_tiny_n's 10, so a floor that suits one does not automatically suit
    # the other and the parameter has to earn its ranking.
    p1_scen = [s for s in SCENARIOS
               if s.name in ('omics_tiny_n', 'omics_imbalanced')]
    p2_scen = [s for s in SCENARIOS
               if s.name in ('omics_tiny_n', 'omics_standard', 'adversarial_easy')]
    p34_scen = list(DEVELOPMENT_SCENARIOS)

    csvs = {n: out_dir / f'phase{n}_raw.csv' for n in (1, 2, 3, 4)}

    if args.report_only:
        return _report(csvs, out_dir)

    print(f'Output      : {out_dir}')
    print(f'Workers     : {args.workers} x {n_jobs} threads '
          f'(of {args.total_cores} cores)')
    print(f'Dimensions  : {len(PARAM_SPACE)} -- {", ".join(PARAM_SPACE)}')

    # -- Phase 1: Halton scan on the cheapest scenario --------------------
    if '1' in args.phases:
        trials = [(name, p) for name, p in BASELINE_CONFIGS.items()]
        trials += [(f'h_{i:03d}', sample_params_halton(i))
                   for i in range(args.phase1_trials)]
        df1 = run_scan(trials, p1_scen, seeds_fast, csvs[1], 'phase1',
                       args.workers, n_jobs)
        print('\nPhase 1 sensitivity (Spearman r vs composite):')
        print(sensitivity(df1).to_string(index=False))

    # -- Phase 2: top configs on three scenarios --------------------------
    if '2' in args.phases:
        agg1 = aggregate_configs(pd.read_csv(csvs[1]))
        _report_degeneracy(agg1, 'phase1')
        top = shortlist(agg1, args.phase2_top)
        trials = _rebuild(pd.read_csv(csvs[1]), top)
        run_scan(trials, p2_scen, seeds_full, csvs[2], 'phase2',
                 args.workers, n_jobs)

    # -- Phase 3: top configs on all development scenarios ----------------
    if '3' in args.phases:
        agg2 = aggregate_configs(pd.read_csv(csvs[2]))
        _report_degeneracy(agg2, 'phase2')
        top = shortlist(agg2, args.phase3_top)
        trials = _rebuild(pd.read_csv(csvs[2]), top)
        run_scan(trials, p34_scen, seeds_full, csvs[3], 'phase3',
                 args.workers, n_jobs)

    # -- Phase 4: neighbourhood refinement --------------------------------
    if '4' in args.phases:
        agg3 = aggregate_configs(pd.read_csv(csvs[3]))
        _report_degeneracy(agg3, 'phase3')
        top = shortlist(agg3, args.phase4_top)
        base = _rebuild(pd.read_csv(csvs[3]), top)
        rng = np.random.RandomState(args.seed + 42)
        trials = list(base)
        for name, params in base:
            for j, nb in enumerate(generate_neighbours(
                    params, args.phase4_perturb, rng)):
                trials.append((f'{name}_n{j:02d}', nb))
        run_scan(trials, p34_scen, seeds_full, csvs[4], 'phase4',
                 args.workers, n_jobs)

    return _report(csvs, out_dir)


def _report_degeneracy(agg: pd.DataFrame, label: str) -> None:
    """Warn loudly when a phase produced a ranking it cannot actually support."""
    d = degeneracy(agg)
    print(f"\n[{label}] resolution: {d['distinct_outcomes']} distinct outcomes "
          f"from {d['n_configs']} configs ({d['resolution']:.1%}); "
          f"{d['tied_at_top']} tied at the top, largest tie "
          f"{d['largest_tie']}")

    # A parameter the shortlist collapsed to one value cannot be validated by
    # this phase, however many evaluations it runs.
    frozen = [k for k in PARAM_SPACE
              if k in agg.columns and agg[k].nunique() < 2]
    if frozen:
        print(f"  NOTE: {label} cannot validate "
              f"{', '.join(frozen)} -- the shortlist has a single value for "
              f"each, so this phase confirms nothing about them. Their support "
              f"comes from whichever earlier phase last varied them.")
    if d['resolution'] < 0.25 or d['tied_at_top'] > 3:
        print(f"  WARNING: {label} is largely degenerate on these scenarios. "
              f"The shortlist is being spread across ties rather than taken "
              f"in sort order, but a phase this flat gives weak evidence. "
              f"Treat the winner as provisional and prefer the deeper phases.")


def _rebuild(df: pd.DataFrame, names: Sequence[str]) -> List[Tuple[str, dict]]:
    """Recover parameter dicts for named configs from a results frame."""
    out = []
    for name in names:
        grp = df[df['config'] == name]
        if grp.empty:
            continue
        row = grp.iloc[0]
        params = {}
        for k, spec in PARAM_SPACE.items():
            params[k] = (int(row[k]) if spec[0] == 'choice' else float(row[k]))
        out.append((name, params))
    return out


def _report(csvs: Dict[int, Path], out_dir: Path) -> int:
    frames = [pd.read_csv(p) for p in csvs.values() if p.exists()]
    if not frames:
        print('No results yet.')
        return 1

    pd.set_option('display.width', 220)
    latest = max((n for n, p in csvs.items() if p.exists()))
    df = pd.read_csv(csvs[latest])
    agg = aggregate_configs(df)

    print(f'\n{"=" * 100}')
    print(f'Configuration search result -- deepest completed phase: {latest}')
    print('=' * 100)
    print(agg.head(12).to_string(index=False))

    _report_degeneracy(agg, f'phase{latest}')

    print(f'\n{"-" * 100}\nSensitivity at this phase\n{"-" * 100}')
    print(sensitivity(df).to_string(index=False))

    best = agg.iloc[0]
    tied = agg[agg['composite'] == best['composite']]

    print(f'\n{"-" * 100}\nRecommended for SelectOmicsConfig.omics()\n{"-" * 100}')
    if len(tied) > 1:
        print(f"  NOTE: {len(tied)} configs tie for first. The values below are "
              f"one member of that tie.\n  Parameters that VARY across the tied "
              f"set are undetermined by this evidence:")
        for k in PARAM_SPACE:
            if k in tied.columns and tied[k].nunique() > 1:
                print(f"    {k}: {sorted(tied[k].unique().tolist())}")
        print()

    for k in PARAM_SPACE:
        if k not in best:
            continue
        v = best[k]
        if k == 'min_consensus':
            pct = int(v)
            print(f"    'min_consensus': "
                  f"{'None' if pct <= 0 else round(pct / 100.0, 2)},")
        elif PARAM_SPACE[k][0] == 'choice':
            print(f"    {k!r}: {int(v)},")
        else:
            print(f"    {k!r}: {round(float(v), 3)},")
    print(f"\n  composite {best['composite']}  F1 {best['mean_f1']}  "
          f"KI {best['mean_ki']}  AUC {best['mean_auc']}  "
          f"n_selected {best['n_sel_mean']}")

    for name in ('default', 'h_071_survivors', 'h_071_at_floor_10'):
        row = agg[agg['config'] == name]
        if not row.empty:
            r = row.iloc[0]
            print(f"  {name:20s} composite {r['composite']}  F1 {r['mean_f1']}  "
                  f"KI {r['mean_ki']}  n {r['n_sel_mean']}")

    if 'step_state' in df.columns:
        print(f'\n{"-" * 100}\nDid the later steps actually run?\n{"-" * 100}')
        print(df['step_state'].value_counts().to_string())

    agg.to_csv(out_dir / 'summary.csv', index=False)
    print(f'\nWritten -> {out_dir / "summary.csv"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
