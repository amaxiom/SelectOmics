"""
End-to-end coverage of the 0.7.0 configuration flags.

Each of these was added during the 0.7.0 work and each changes selection
behaviour, but they were only ever exercised through ad-hoc benchmark scripts.
Unit tests pin their *definitions*; these run them through the steps they
actually modify, so a regression shows up as a failing test rather than as a
silently different feature panel.

Covered here:
  class_aware_correlation   Step 1: which correlation matrix and tie-break
  step2_role                Step 2: terminal panel or candidate set
  allow_union_rung          Steps 2 and 3: the ceiling on what a step returns
  step2_stage_colsample     Step 2: the size of the reachable pool
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
from SelectOmics.selection.step1_cleaning import run_step1_cleaning
from SelectOmics.selection.step2_regularization import run_step2_regularization
from SelectOmics.models.base import quick_tune_all


def _data(n=80, p=40, n_informative=8, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n, p), columns=[f"f{i:03d}" for i in range(p)])
    y = np.array([i % 2 for i in range(n)])
    rng.shuffle(y)
    X.iloc[:, :n_informative] += 1.6 * y[:, None]
    return X, y


def _cfg(**over):
    kw = dict(data_path="unused.csv", target_column="Class", algorithm="RF",
              n_consensus_models=2, quick_tune_iterations=2, n_bootstrap=10,
              verbose=False, create_visualizations=False,
              save_intermediate_results=False, enable_step_evaluations=False,
              enable_final_test_evaluation=False)
    kw.update(over)
    return SelectOmicsConfig(**kw)


def _step1(X, y, cfg):
    return run_step1_cleaning(X, X.iloc[:12].copy(), y, cfg,
                              StratifiedKFold(3), {}, 2, ["0", "1"])


def _step2(X, y, cfg):
    tuned = quick_tune_all(X, y, cfg)
    return run_step2_regularization(X, X.iloc[:12].copy(), y, cfg,
                                    StratifiedKFold(3), tuned, 2, ["0", "1"])


# ===========================================================================
# class_aware_correlation
# ===========================================================================

@pytest.mark.integration
@pytest.mark.parametrize("class_aware", [True, False])
def test_step1_runs_under_both_correlation_modes(class_aware):
    X, y = _data()
    res = _step1(X, y, _cfg(class_aware_correlation=class_aware))
    assert res["X_train_clean"].shape[1] >= 1
    assert set(res["X_train_clean"].columns) <= set(X.columns)


@pytest.mark.integration
def test_class_aware_logs_which_matrix_it_used(caplog):
    """
    Which correlation matrix produced a panel is not recoverable from the
    output, so the log is the only record. A reader comparing two runs needs
    it.
    """
    caplog.set_level(logging.INFO, logger="SelectOmics")
    X, y = _data()
    _step1(X, y, _cfg(class_aware_correlation=True))
    assert "class-aware" in caplog.text.lower()


@pytest.mark.integration
def test_class_aware_keeps_co_regulated_markers():
    """
    The whole point of the class-aware filter: features correlated only
    because both track the label must survive, where a total-correlation
    filter would keep one and discard the rest.
    """
    rng = np.random.RandomState(0)
    n = 80
    y = np.array([i % 2 for i in range(n)])
    rng.shuffle(y)
    # Six co-regulated markers: highly correlated with each other, but only
    # because each independently tracks the class.
    markers = np.column_stack([1.8 * y + rng.randn(n) * 0.35 for _ in range(6)])
    noise = rng.randn(n, 24)
    X = pd.DataFrame(np.hstack([markers, noise]),
                     columns=[f"m{i}" for i in range(6)]
                             + [f"n{i}" for i in range(24)])
    aware = _step1(X, y, _cfg(class_aware_correlation=True,
                              min_correlation_threshold=0.70))
    kept = [c for c in aware["X_train_clean"].columns if c.startswith("m")]
    assert len(kept) >= 2, (
        "the class-aware filter collapsed a co-regulated module to one member, "
        "which is the behaviour it exists to prevent"
    )


# ===========================================================================
# step2_role
# ===========================================================================

@pytest.mark.integration
def test_step2_terminal_is_the_default_behaviour():
    X, y = _data()
    res = _step2(X, y, _cfg())
    assert res["X_train_reg"].shape[1] >= 1


@pytest.mark.integration
def test_step2_prefilter_targets_the_step3_gate(caplog):
    """
    Prefilter must aim above MIN_FEATURES_FOR_RFECV and say so, since the
    whole point is that Step 3 subsequently engages.
    """
    from SelectOmics.selection.step3_wrapper import MIN_FEATURES_FOR_RFECV
    caplog.set_level(logging.INFO, logger="SelectOmics")
    X, y = _data(n=80, p=60)
    res = _step2(X, y, _cfg(step2_role="prefilter", enable_step3=True))
    assert res["X_train_reg"].shape[1] >= 1
    assert "prefilter" in caplog.text.lower()
    assert str(MIN_FEATURES_FOR_RFECV) in caplog.text


# ===========================================================================
# allow_union_rung
# ===========================================================================

@pytest.mark.integration
@pytest.mark.parametrize("union", [False, True])
def test_step2_runs_under_both_ceiling_modes(union):
    X, y = _data()
    res = _step2(X, y, _cfg(allow_union_rung=union))
    assert res["X_train_reg"].shape[1] >= 1
    assert res["consensus_outcome"] in (
        "consensus", "relaxed", "union", "consensus_limited", "rank_average")


@pytest.mark.integration
def test_union_rung_never_returns_fewer_than_intersection():
    """
    The union rung raises the ceiling, so it can only ever return at least as
    many features as the intersection did on the same data.
    """
    X, y = _data(n=80, p=50)
    tight = _step2(X, y, _cfg(allow_union_rung=False, min_features_floor=40))
    loose = _step2(X, y, _cfg(allow_union_rung=True, min_features_floor=40))
    assert (loose["X_train_reg"].shape[1]
            >= tight["X_train_reg"].shape[1])


# ===========================================================================
# step2_stage_colsample
# ===========================================================================

@pytest.mark.integration
@pytest.mark.parametrize("colsample", [0.1, 1.0])
def test_step2_stage_colsample_is_accepted(colsample):
    """Only XGB reads it, but no algorithm may break on it."""
    X, y = _data()
    res = _step2(X, y, _cfg(step2_stage_colsample=colsample))
    assert res["X_train_reg"].shape[1] >= 1


# ===========================================================================
# Reporting under verbose
# ===========================================================================

@pytest.mark.integration
def test_step2_verbose_reports_its_agreement(caplog):
    """
    The agreement level is the claim the panel carries, so a verbose run must
    state it rather than leaving the reader to infer it from the count.
    """
    caplog.set_level(logging.INFO, logger="SelectOmics")
    X, y = _data()
    res = _step2(X, y, _cfg(verbose=True))
    text = caplog.text.lower()
    assert "unanimity" in text or "agreement" in text or "votes" in text
    assert res["agreement_label"]
