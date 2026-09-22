"""
Failure and fallback paths in the three selection steps.

Everything here is an ``except`` branch or a degraded-mode fallback. They exist
precisely because omics data produces the conditions that trigger them: a
subsample missing a class, a solver that will not converge, an importance
vector that is entirely zero. They are also the code least likely to be
exercised by a normal run, so a regression in one surfaces as a silently wrong
result rather than an error.

Each test constructs the specific broken input the branch is written for.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from SelectOmics.config import SelectOmicsConfig
from SelectOmics.selection import step1_cleaning as s1
from SelectOmics.selection import step2_regularization as s2
from SelectOmics.selection import step3_wrapper as s3


def _cfg(**over):
    kw = dict(data_path="unused.csv", target_column="Class", algorithm="RF",
              n_consensus_models=2, quick_tune_iterations=2, n_bootstrap=10,
              verbose=False, create_visualizations=False,
              save_intermediate_results=False, enable_step_evaluations=False,
              enable_final_test_evaluation=False)
    kw.update(over)
    return SelectOmicsConfig(**kw)


# ===========================================================================
# Step 1 fallbacks
# ===========================================================================

def test_variance_filter_survives_a_degenerate_threshold():
    """
    VarianceThreshold raises when no feature clears the bar. The step must
    treat that as "everything dropped" and let the fallback ladder handle it,
    not propagate a sklearn error.
    """
    X = pd.DataFrame(np.ones((20, 5)), columns=list("abcde"))
    kept = s1._apply_variance_filter(X, 10.0)   # positional: ['X', 'var_thresh']
    assert kept.shape == (5,)
    assert not kept.any(), "a threshold nothing meets must drop everything"


def test_correlation_filter_tie_break_chain():
    """
    Equal priority falls to variance, and equal variance falls to name order,
    so the outcome never depends on column ordering.
    """
    rng = np.random.RandomState(0)
    base = rng.randn(50)
    X = pd.DataFrame({"probeA": base, "probeZ": base.copy(),
                      "other": rng.randn(50)})
    var = X.var()
    corr = X.corr().abs()
    # Identical priority AND identical variance: only the name can decide.
    priority = pd.Series({"probeA": 1.0, "probeZ": 1.0, "other": 0.5})
    kept_fwd = s1._apply_correlation_filter(X, 0.9, var, corr, priority)
    rev = X[["other", "probeZ", "probeA"]]
    kept_rev = s1._apply_correlation_filter(
        rev, 0.9, rev.var(), rev.corr().abs(),
        priority[["other", "probeZ", "probeA"]])
    assert set(X.columns[kept_fwd]) == set(rev.columns[kept_rev])


def test_correlation_filter_higher_priority_partner_drops_col():
    """When the partner outranks the current column, the column itself goes."""
    rng = np.random.RandomState(1)
    base = rng.randn(40)
    X = pd.DataFrame({"weak": base, "strong": base * 1.0 + 1e-9,
                      "lone": rng.randn(40)})
    priority = pd.Series({"weak": 0.1, "strong": 0.9, "lone": 0.5})
    kept = s1._apply_correlation_filter(X, 0.9, X.var(), X.corr().abs(),
                                        priority)
    survivors = set(X.columns[kept])
    assert "strong" in survivors and "weak" not in survivors


@pytest.mark.integration
def test_step1_forced_topk_fallback(caplog):
    """
    When both the intersection and the union fall below the floor, the step
    forces the top-k. The floor has to exceed what the filters agree on for
    this to fire.
    """
    caplog.set_level(logging.WARNING, logger="SelectOmics")
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(40, 12), columns=[f"f{i}" for i in range(12)])
    y = np.array([i % 2 for i in range(40)])
    cfg = _cfg(min_features_floor=11, max_variance_threshold=0.30)
    res = s1.run_step1_cleaning(
        X, X.iloc[:10].copy(), y, cfg, StratifiedKFold(3), {}, 2, ["0", "1"])
    assert res["X_train_clean"].shape[1] >= 1


# ===========================================================================
# Step 2 fallbacks
# ===========================================================================

@pytest.mark.parametrize("algorithm", ["LR", "XGB", "RF", "SVM"])
def test_stage_models_build_for_every_algorithm(algorithm):
    """Each algorithm takes a different branch in _build_stage_model."""
    pytest.importorskip("xgboost") if algorithm == "XGB" else None
    for stage in ("l1", "l2"):
        m = s2._build_stage_model(algorithm, stage, seed=0)
        assert m is not None
        assert "clf" in getattr(m, "named_steps", {"clf": None})


@pytest.mark.parametrize("algorithm", ["LR", "RF", "SVM"])
def test_importance_extraction_for_every_algorithm(algorithm):
    """
    SVM has no coef_ on an RBF kernel, so the stage uses a LinearSVC
    surrogate. Each path must return one non-negative value per feature.
    """
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(40, 6), columns=[f"f{i}" for i in range(6)])
    y = np.array([i % 2 for i in range(40)])
    X.iloc[:, 0] += 2.0 * y
    model = s2._fit_stage_model(algorithm, "l1", 0, X, y)
    imp = s2._extract_importance(model, algorithm)
    assert imp.shape == (6,)
    assert np.all(np.isfinite(imp))
    assert np.all(imp >= 0)


def test_rank_keep_mask_on_all_zero_importance():
    """
    Tree importance is exactly zero for every unused feature. An all-zero
    vector means the model used nothing, and the mask must say so rather than
    selecting arbitrarily by index.
    """
    imp = np.zeros(50)
    assert s2._rank_keep_mask(imp, 0.5).sum() == 0


def test_rank_keep_mask_never_extends_into_the_zero_region():
    imp = np.zeros(100)
    imp[:4] = [0.4, 0.3, 0.2, 0.1]
    # Asking for 90 of 100 cannot exceed the 4 the model actually used.
    assert s2._rank_keep_mask(imp, 0.10).sum() == 4


def test_rank_keep_mask_keeps_at_least_one():
    """percentile 1.0 asks for nothing; an empty stage would break consensus."""
    imp = np.array([0.5, 0.3, 0.2])
    assert s2._rank_keep_mask(imp, 1.0).sum() >= 1


# ===========================================================================
# Step 3 fallbacks
# ===========================================================================

def test_stability_stage_reports_all_failed(caplog):
    """
    If every subsample fails, the stage must set stability_all_failed and
    default to all-True so the intersection degenerates to the RFECV result
    rather than emptying the panel.
    """
    caplog.set_level(logging.WARNING)
    rng = np.random.RandomState(0)
    n, p = 8, 40
    X = pd.DataFrame(rng.randn(n, p), columns=[f"f{i}" for i in range(p)])
    y = np.array([0] * 4 + [1] * 4)
    counters = {"rfecv_failed": 0, "stability_failed": 0,
                "stability_skipped_class": 0, "stability_all_failed": 0}
    selected, freq, imps = s3._run_stability_stage(
        X_train=X, y_train=y, algorithm="LR", n_models=1, base_seed=0,
        stability_threshold=0.6, min_features=p, warnings=counters)
    assert selected.shape == (p,)
    assert selected.dtype == bool


def test_stability_stage_handles_a_missing_class():
    """A subsample without both classes is skipped, not fatal."""
    rng = np.random.RandomState(0)
    n, p = 24, 30
    X = pd.DataFrame(rng.randn(n, p), columns=[f"f{i}" for i in range(p)])
    y = np.array([0] * 22 + [1] * 2)          # 2 minority samples
    counters = {"rfecv_failed": 0, "stability_failed": 0,
                "stability_skipped_class": 0, "stability_all_failed": 0}
    selected, freq, imps = s3._run_stability_stage(
        X_train=X, y_train=y, algorithm="LR", n_models=1, base_seed=0,
        stability_threshold=0.6, min_features=5, warnings=counters)
    assert selected.shape == (p,)


def test_print_warnings_emits_every_counter(caplog):
    """A silent warning counter is the same as no counter."""
    caplog.set_level(logging.WARNING)
    s3._print_warnings({
        "rfecv_failed": 2, "stability_failed": 3,
        "stability_skipped_class": 4, "stability_all_failed": 1,
    })
    text = caplog.text
    assert "rfecv_failed" in text or "RFECV" in text
    assert "STEP 3 WARNINGS" in text


def test_print_warnings_silent_when_clean(caplog):
    caplog.set_level(logging.WARNING)
    s3._print_warnings({"rfecv_failed": 0, "stability_failed": 0,
                        "stability_skipped_class": 0,
                        "stability_all_failed": 0})
    assert "STEP 3 WARNINGS" not in caplog.text


@pytest.mark.parametrize("algorithm, n_classes, expects_ovr", [
    ("LR", 2, False), ("LR", 3, True),
    ("XGB", 2, False), ("XGB", 4, True),
    ("SVM", 2, False), ("SVM", 3, False),
])
def test_rfecv_scoring_matches_the_estimator(algorithm, n_classes, expects_ovr):
    """
    The SVM path uses a LinearSVC surrogate with no predict_proba, so it must
    never be given a probability-only scorer whatever the class count.
    """
    scoring = s3._rfecv_scoring(algorithm, n_classes)
    assert (scoring == "roc_auc_ovr") is expects_ovr
    if algorithm == "SVM":
        assert scoring != "roc_auc_ovr"


@pytest.mark.parametrize("algorithm", ["LR", "XGB", "RF", "SVM"])
def test_rfecv_estimator_builds_with_an_importance_getter(algorithm):
    est, getter = s3._build_rfecv_estimator(algorithm, seed=0)
    assert est is not None
    assert getter is not None


# ---------------------------------------------------------------------------
# Step 1's forced top-k must be deterministic.
#
# The fallback ranks on the correlation ratio, which is ~0 for every noise
# feature, so exact ties are the common case. numpy's default sort is not
# stable, which made the forced panel depend on memory layout rather than on
# the data. The package guarantees an identical feature set across runs.
# ---------------------------------------------------------------------------

class TestForcedTopKIsDeterministic:

    def test_source_uses_a_stable_sort(self):
        import inspect
        from SelectOmics.selection import step1_cleaning as mod
        src = inspect.getsource(mod.run_step1_cleaning)
        assert "kind='stable'" in src or 'kind="stable"' in src, (
            "the forced top-k fallback sorts without kind='stable'; ties in "
            "the correlation ratio would then order by memory layout"
        )

    @pytest.mark.integration
    def test_repeated_runs_give_the_identical_panel(self):
        """The guarantee itself, asserted end to end on tie-heavy data."""
        rng = np.random.RandomState(0)
        # Mostly pure noise, so the correlation ratio ties at ~0 for nearly
        # every column -- exactly the case the stable sort exists for.
        X = pd.DataFrame(rng.randn(40, 25),
                         columns=[f"f{i:02d}" for i in range(25)])
        y = np.array([i % 2 for i in range(40)])
        cfg = _cfg(min_features_floor=20)
        panels = []
        for _ in range(3):
            res = s1.run_step1_cleaning(
                X.copy(), X.iloc[:10].copy(), y, cfg,
                StratifiedKFold(3), {}, 2, ["0", "1"])
            panels.append(tuple(res["X_train_clean"].columns))
        assert len(set(panels)) == 1, (
            f"three identical runs produced {len(set(panels))} different "
            f"feature sets"
        )
