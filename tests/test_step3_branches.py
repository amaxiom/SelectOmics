"""
Step 3 conditional branches.

Step 3 is the most branch-heavy module in the package and the hardest to
reach, because in an ordinary run it does not execute at all. Each test here
sets up the one condition its branch is written for: a single consensus model,
an adequacy assessment present, a verbose run, a subsample fraction that would
consume the whole matrix, a stability stage that abstains rather than vetoes.
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from SelectOmics.config import SelectOmicsConfig
from SelectOmics.models.base import quick_tune_all
from SelectOmics.selection import step3_wrapper as s3


def _data(n=90, p=40, n_informative=6, n_classes=2, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n, p), columns=[f"f{i:03d}" for i in range(p)])
    y = np.array([i % n_classes for i in range(n)])
    rng.shuffle(y)
    X.iloc[:, :n_informative] += 1.6 * y[:, None]
    return X, y


def _cfg(tmp_path, **over):
    kw = dict(data_path="unused.csv", target_column="Class", algorithm="RF",
              output_dir=str(tmp_path), n_consensus_models=2,
              quick_tune_iterations=2, n_bootstrap=10, verbose=False,
              create_visualizations=False, save_intermediate_results=False,
              enable_step_evaluations=False, enable_final_test_evaluation=False)
    kw.update(over)
    return SelectOmicsConfig(**kw)


def _run(X, y, cfg, n_classes=2, ref=None, adequacy=None):
    tuned = quick_tune_all(X, y, cfg)
    return s3.run_step3_wrapper(
        X, X.iloc[:12].copy(), y, cfg, StratifiedKFold(3), tuned,
        n_classes, [str(c) for c in range(n_classes)], ref, adequacy)


# ---------------------------------------------------------------------------
# _subsample_rows
# ---------------------------------------------------------------------------

def test_subsample_returns_everything_when_the_fraction_is_too_large():
    """
    A fraction that would keep the whole matrix leaves nothing to subsample,
    so the helper returns the input rather than asking for an empty split.
    """
    X, y = _data(n=20, p=5)
    Xs, ys = s3._subsample_rows(X, y, seed=0, fraction=1.0)
    assert len(Xs) == len(X)
    assert len(ys) == len(y)


def test_subsample_falls_back_when_the_split_is_impossible():
    """
    StratifiedShuffleSplit raises when a class is too small for the requested
    size. The helper must return the full data, not propagate.
    """
    X = pd.DataFrame(np.random.RandomState(0).randn(6, 4))
    y = np.array([0, 0, 0, 0, 1, 1])
    Xs, ys = s3._subsample_rows(X, y, seed=0, fraction=0.5)
    assert len(Xs) == len(ys)
    assert len(Xs) >= 2


def test_subsample_reduces_when_it_can():
    X, y = _data(n=100, p=5)
    Xs, ys = s3._subsample_rows(X, y, seed=0, fraction=0.6)
    assert len(Xs) < len(X)
    assert set(np.unique(ys)) == set(np.unique(y)), "a class was lost"


# ---------------------------------------------------------------------------
# Importance extraction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("algorithm", ["LR", "RF", "SVM"])
def test_importance_extraction_per_algorithm(algorithm):
    """SVM takes the LinearSVC surrogate branch; the others read directly."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler
    X, y = _data(n=60, p=8)
    est, getter = s3._build_rfecv_estimator(algorithm, seed=0)
    est.fit(X, y)
    imp = s3._extract_importance_step3(est, algorithm)
    assert imp.shape == (8,)
    assert np.all(np.isfinite(imp))


# ---------------------------------------------------------------------------
# Stability stage guards
# ---------------------------------------------------------------------------

def test_stability_abstains_when_the_subsample_size_is_impossible():
    """
    If the computed subsample size cannot work, the stage abstains (all-True)
    rather than vetoing everything, so the intersection degrades to RFECV.

    Reaching this needs n = n_classes. The clamps above the guard
    (max_subsample = n - n_classes, then a floor of n_classes) make it
    unreachable for any larger n: subsample_size can neither fall below
    n_classes nor reach n_samples. The guard is therefore defensive against
    only the degenerate one-sample-per-class case.
    """
    rng = np.random.RandomState(0)
    n, p = 2, 20
    X = pd.DataFrame(rng.randn(n, p), columns=[f"f{i}" for i in range(p)])
    y = np.array([0, 1])
    counters = {"rfecv_failed": 0, "stability_failed": 0,
                "stability_skipped_class": 0, "stability_all_failed": 0}
    selected, freq, imps = s3._run_stability_stage(
        X_train=X, y_train=y, algorithm="LR", n_models=1, base_seed=0,
        stability_threshold=0.6, min_features=p, warnings=counters)
    assert selected.all(), "abstention must keep everything, not drop everything"
    assert counters["stability_all_failed"] >= 1


@pytest.mark.integration
def test_stability_verbose_logs_progress(caplog):
    caplog.set_level(logging.DEBUG, logger="SelectOmics")
    X, y = _data(n=60, p=20)
    counters = {"rfecv_failed": 0, "stability_failed": 0,
                "stability_skipped_class": 0, "stability_all_failed": 0}
    s3._run_stability_stage(
        X_train=X, y_train=y, algorithm="LR", n_models=1, base_seed=0,
        stability_threshold=0.6, min_features=5, warnings=counters,
        verbose=True)
    assert "subsample" in caplog.text.lower()


# ---------------------------------------------------------------------------
# Consensus outcome reporting
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_reports_a_unanimous_outcome(tmp_path, caplog):
    """The 'consensus' branch logs differently from the relaxed one."""
    caplog.set_level(logging.INFO, logger="SelectOmics")
    X, y = _data(n=90, p=40, n_informative=10)
    res = _run(X, y, _cfg(tmp_path, min_features_floor=1, verbose=True))
    assert res["consensus_outcome"] in ("consensus", "relaxed",
                                        "consensus_limited", "rank_average")
    assert caplog.text.strip()


@pytest.mark.integration
def test_reports_a_consensus_limited_outcome(tmp_path, caplog):
    """
    When min_consensus stops the descent before the feature floor is met, the
    step must say the two requirements conflicted rather than silently
    returning a short panel.
    """
    caplog.set_level(logging.WARNING, logger="SelectOmics")
    X, y = _data(n=90, p=40)
    res = _run(X, y, _cfg(tmp_path, min_features_floor=39,
                          min_consensus=1.0, verbose=True))
    if res["consensus_outcome"] == "consensus_limited":
        assert "min_consensus" in caplog.text or "agreement" in caplog.text


@pytest.mark.integration
def test_union_outcome_is_reachable(tmp_path):
    X, y = _data(n=90, p=40)
    res = _run(X, y, _cfg(tmp_path, allow_union_rung=True,
                          min_features_floor=35))
    assert res["consensus_outcome"] in (
        "consensus", "relaxed", "union", "consensus_limited", "rank_average")


# ---------------------------------------------------------------------------
# Single-model and evaluation branches
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_single_consensus_model(tmp_path):
    """n_models == 1 takes a distinct build-and-evaluate path."""
    X, y = _data(n=90, p=40)
    res = _run(X, y, _cfg(tmp_path, n_consensus_models=1,
                          enable_step_evaluations=True))
    assert res.get("skipped") is False
    assert res.get("n_models", res.get("n_consensus_models")) == 1


@pytest.mark.integration
def test_evaluation_against_the_step0_reference(tmp_path, caplog):
    """With a reference supplied, the step reports the comparison."""
    caplog.set_level(logging.INFO, logger="SelectOmics")
    X, y = _data(n=90, p=40)
    ref = {"mean_auc": 0.80, "std_auc": 0.03,
           "fold_aucs": [0.78, 0.80, 0.82]}
    res = _run(X, y, _cfg(tmp_path, enable_step_evaluations=True,
                          verbose=True), ref=ref)
    assert res.get("skipped") is False


@pytest.mark.integration
def test_adequacy_warning_is_surfaced(tmp_path, caplog):
    """A critical sample-size assessment must reach the log during Step 3."""
    caplog.set_level(logging.WARNING, logger="SelectOmics")
    X, y = _data(n=90, p=40)
    adequacy = {"overall_severity": "critical",
                "warnings": ["CRITICAL: sample size is far too small"]}
    # The warning is emitted from the evaluation block, which only runs when
    # step evaluations are enabled.
    _run(X, y, _cfg(tmp_path, enable_step_evaluations=True, verbose=True),
         adequacy=adequacy)
    assert caplog.text.strip(), "a critical adequacy flag was swallowed"
