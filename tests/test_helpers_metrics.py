"""
Helper and metric edge cases.

These are the small functions every step calls, where the interesting
behaviour is what happens on degenerate input: a fold containing one class, a
probability matrix with a column missing, an AUC that cannot be computed. Each
returns NaN or a fallback rather than raising, which is correct but means a
regression here is silent.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from SelectOmics.evaluation.metrics import (
    binarize_labels,
    build_comprehensive_metrics,
    build_pipeline_summary_df,
)
from SelectOmics.models.base import build_consensus_pipelines, resolve_n_jobs
from SelectOmics.utils.helpers import (
    _binary_search_percentile,
    compute_macro_auc_ovr,
    compute_svm_afi,
    detect_cpu_count,
    detect_gpu,
    ensure_binary_proba,
    print_adequacy_warning,
    save_step_summary,
)


# ---------------------------------------------------------------------------
# AUC helpers
# ---------------------------------------------------------------------------

def test_macro_auc_on_a_single_class_returns_chance():
    """
    With one class present, AUC is undefined and the function returns 0.5 by
    documented design, not NaN.

    Worth knowing when reading aggregates: 0.5 is a real number, so a fold that
    could not be scored is averaged in as chance rather than excluded the way a
    NaN would be. A run whose folds mostly degenerate therefore reports a mean
    AUC pulled toward 0.5 rather than reporting that it could not be measured.
    """
    y = np.zeros(20, dtype=int)
    proba = np.column_stack([np.ones(20) * 0.6, np.ones(20) * 0.4])
    assert compute_macro_auc_ovr(y, proba, classes=np.arange(2)) == 0.5


def test_macro_auc_with_a_class_absent_from_the_fold():
    """
    A stratified fold can still lose a rare class. The macro average must be
    taken over the classes present, not fail.
    """
    y = np.array([0] * 10 + [1] * 10)          # class 2 declared but absent
    proba = np.random.RandomState(0).dirichlet(np.ones(3), size=20)
    val = compute_macro_auc_ovr(y, proba, classes=np.arange(3))
    assert np.isnan(val) or 0.0 <= val <= 1.0


def test_macro_auc_accepts_a_dataframe():
    """predict_proba output is sometimes wrapped; .values must be taken."""
    y = np.array([0, 0, 1, 1, 0, 1])
    proba = pd.DataFrame({0: [.9, .8, .2, .1, .7, .3],
                          1: [.1, .2, .8, .9, .3, .7]})
    val = compute_macro_auc_ovr(y, proba, classes=np.arange(2))
    assert 0.0 <= val <= 1.0


def test_ensure_binary_proba_shapes():
    """A 1-D positive-class vector and a 2-column matrix must both work."""
    one_d = np.array([0.2, 0.8, 0.5])
    two_d = np.column_stack([1 - one_d, one_d])
    assert ensure_binary_proba(one_d, 2).shape[0] == 3
    assert ensure_binary_proba(two_d, 2).shape[0] == 3


def test_binarize_labels():
    y = np.array([0, 1, 2, 1])
    b = binarize_labels(y, 3)
    assert b.shape == (4, 3)
    assert (b.sum(axis=1) == 1).all()


# ---------------------------------------------------------------------------
# SVM surrogate importance
# ---------------------------------------------------------------------------

def test_compute_svm_afi_is_normalised():
    from sklearn.svm import LinearSVC
    rng = np.random.RandomState(0)
    X = rng.randn(40, 5)
    y = (X[:, 0] > 0).astype(int)
    clf = LinearSVC(dual="auto", max_iter=2000).fit(X, y)
    afi = compute_svm_afi(clf)
    assert afi.shape == (5,)
    assert np.all(afi >= 0) and afi.max() <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Binary search
# ---------------------------------------------------------------------------

def test_binary_search_hits_a_reachable_target():
    """A monotone decreasing function should be solved close to the target."""
    def f(p):
        return 100.0 * (1.0 - p)
    best_p, best_v, n = _binary_search_percentile(f, 0.0, 1.0, target=40.0)
    assert abs(best_v - 40.0) <= 5.0
    assert n >= 2


def test_binary_search_warns_when_the_target_is_unreachable():
    """
    Saturation is the normal case on tree importance, so the caller must be
    told the retention setting is not what determined the result.
    """
    def saturated(p):
        return 10.0                      # never reaches 500 whatever p is
    with pytest.warns(UserWarning, match="could not reach"):
        best_p, best_v, _ = _binary_search_percentile(
            saturated, 0.0, 1.0, target=500.0, label="test retention")
    assert best_v == 10.0


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

def test_detect_cpu_count_is_sane():
    n = detect_cpu_count()
    assert isinstance(n, int) and n >= 1


def test_detect_gpu_returns_a_bool():
    """Must answer, not raise, on a machine with no CUDA."""
    assert isinstance(detect_gpu(), bool)


@pytest.mark.parametrize("n_jobs", [None, 1, 2, -1])
def test_resolve_n_jobs(n_jobs):
    from SelectOmics.config import SelectOmicsConfig
    cfg = SelectOmicsConfig(data_path="u.csv", target_column="Class",
                            n_jobs=n_jobs) if n_jobs is not None else None
    got = resolve_n_jobs(cfg)
    assert isinstance(got, int)
    assert got >= 1 or got == -1


# ---------------------------------------------------------------------------
# Adequacy warnings
# ---------------------------------------------------------------------------

def test_print_adequacy_warning_is_silent_when_adequate(caplog):
    caplog.set_level(logging.WARNING)
    print_adequacy_warning("Step 2", 50,
                           {"overall_severity": "adequate", "warnings": []})
    assert "Step 2" not in caplog.text


def test_print_adequacy_warning_speaks_up_when_critical(caplog):
    caplog.set_level(logging.WARNING)
    print_adequacy_warning("Step 2", 50, {
        "overall_severity": "critical",
        "warnings": ["CRITICAL: far too few samples"],
    })
    assert caplog.text.strip(), "a critical assessment must be surfaced"


def test_print_adequacy_warning_tolerates_none():
    print_adequacy_warning("Step 2", 50, None)


# ---------------------------------------------------------------------------
# Summary building
# ---------------------------------------------------------------------------

def test_build_pipeline_summary_df():
    # Each dict carries eval_result (a cv_evaluate_model output), not
    # consensus_result.
    steps = [
        {"step_name": "Step 0 (Reference)", "n_features": 100,
         "eval_result": {"mean_auc": 0.80, "std_auc": 0.02}},
        {"step_name": "Step 1 (Data Cleaning)", "n_features": 40,
         "eval_result": {"mean_auc": 0.83, "std_auc": 0.02}},
    ]
    df = build_pipeline_summary_df(steps, original_n_features=100,
                                   algorithm="XGB")
    assert "Step" in df.columns and "Features" in df.columns
    assert "XGB_CV_AUC" in df.columns
    assert df["Reduction_Percentage"].iloc[-1] == pytest.approx(60.0)


def test_save_step_summary_writes(tmp_path):
    """Signature is (step_name, results, save_dir)."""
    save_step_summary("step2", {"Step": "step2", "n_features_out": 21},
                      tmp_path)
    written = list(tmp_path.glob("step2*"))
    assert written and written[0].stat().st_size > 0


def test_build_comprehensive_metrics():
    rng = np.random.RandomState(0)
    y = np.array([0, 1] * 15)
    proba = rng.dirichlet(np.ones(2), size=30)
    pred = proba.argmax(axis=1)
    m = build_comprehensive_metrics(
        y_test=y, y_pred=pred, y_proba=proba, class_names=["A", "B"],
        n_classes=2, training_result={"mean_auc": 0.8, "std_auc": 0.03},
        step_label="Step 3 (Wrappers)", algorithm="XGB", n_models=5,
        achieved_agreement=0.6, agreement_label="moderate",
        original_n_features=100, final_n_features=12)
    assert m["algorithm"] == "XGB"
    assert m["final_features"] == 12
    assert m["original_features"] == 100
    assert m["achieved_agreement"] == pytest.approx(0.6)
    assert m["feature_reduction_pct"] == pytest.approx(88.0)


# ---------------------------------------------------------------------------
# Consensus pipeline factory
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("algorithm", ["LR", "RF"])
def test_build_consensus_pipelines(algorithm):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    clf = (LogisticRegression() if algorithm == "LR"
           else RandomForestClassifier(n_estimators=5))
    tuned = {algorithm: Pipeline([("scaler", MinMaxScaler()), ("clf", clf)])}
    models = build_consensus_pipelines(algorithm, 3, tuned, base_seed=0)
    assert len(models) == 3
    seeds = {m.named_steps["clf"].get_params().get("random_state")
             for m in models.values()}
    assert len(seeds) == 3, "consensus models must not share a seed"
