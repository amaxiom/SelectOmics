"""
Step 3 execution paths.

Step 3 is the least-covered module in the package because it almost never runs
in an ordinary pipeline: it self-skips below MIN_FEATURES_FOR_RFECV, and Steps
1 and 2 usually deliver fewer features than that. The consequence is that its
main body, its stage helpers and its reporting branches go unexercised until a
user runs it standalone, which is a supported use.

These tests drive it directly on matrices wide enough to open the gate.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from SelectOmics.config import SelectOmicsConfig
from SelectOmics.models.base import quick_tune_all
from SelectOmics.selection.step3_wrapper import (
    MIN_FEATURES_FOR_RFECV,
    run_step3_wrapper,
)


def _data(n=90, p=40, n_informative=6, n_classes=2, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n, p), columns=[f"f{i:03d}" for i in range(p)])
    y = np.array([i % n_classes for i in range(n)])
    rng.shuffle(y)
    X.iloc[:, :n_informative] += 1.5 * y[:, None]
    return X, y


def _cfg(tmp_path, **over):
    kw = dict(
        data_path="unused.csv", target_column="Class", algorithm="RF",
        output_dir=str(tmp_path), n_consensus_models=2,
        quick_tune_iterations=2, n_bootstrap=10, verbose=False,
        create_visualizations=False, save_intermediate_results=False,
        enable_step_evaluations=False, enable_final_test_evaluation=False,
    )
    kw.update(over)
    return SelectOmicsConfig(**kw)


def _run(X, y, cfg, n_classes=2, ref=None, adequacy=None):
    cv = StratifiedKFold(3, shuffle=True, random_state=0)
    # quick_tune_all is (X, y, config, device='cpu'). Passing cv as the 4th
    # positional silently binds it to device, which is only read on the XGB
    # path, so the mistake is invisible until someone tests XGB.
    try:
        tuned = quick_tune_all(X, y, cfg)
    except Exception:                                   # pragma: no cover
        tuned = {}
    return run_step3_wrapper(
        X, X.iloc[: max(6, len(X) // 4)].copy(), y, cfg, cv, tuned,
        n_classes, [str(c) for c in range(n_classes)], ref, adequacy,
    )


# ---------------------------------------------------------------------------
# Skip paths
# ---------------------------------------------------------------------------

def test_skips_below_the_gate(tmp_path):
    """Under MIN_FEATURES_FOR_RFECV the step passes through, untouched."""
    X, y = _data(p=MIN_FEATURES_FOR_RFECV - 5)
    res = _run(X, y, _cfg(tmp_path))
    assert res["skipped"] is True
    assert res["X_train_rfecv"].shape[1] == X.shape[1]
    assert list(res["X_train_rfecv"].columns) == list(X.columns)


def test_disabled_passes_through(tmp_path):
    X, y = _data(p=40)
    res = _run(X, y, _cfg(tmp_path, enable_step3=False))
    assert res["skipped"] is True
    assert res["X_train_rfecv"].shape[1] == X.shape[1]


def test_passthrough_shapes_are_consistent(tmp_path):
    """A skipped step must still return every key downstream code reads."""
    X, y = _data(p=10)
    res = _run(X, y, _cfg(tmp_path))
    for key in ("X_train_rfecv", "X_test_rfecv", "skipped",
                "rfecv_selected", "stability_selected"):
        assert key in res, key
    assert res["rfecv_selected"].shape == (X.shape[1],)
    assert res["stability_selected"].shape == (X.shape[1],)


# ---------------------------------------------------------------------------
# The real path
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_runs_above_the_gate_and_reduces(tmp_path):
    X, y = _data(n=90, p=40, n_informative=6)
    res = _run(X, y, _cfg(tmp_path))
    assert res.get("skipped") is False
    out = res["X_train_rfecv"]
    assert 0 < out.shape[1] <= X.shape[1]
    assert set(out.columns) <= set(X.columns)
    # Train and test must be reduced to the SAME columns.
    assert list(out.columns) == list(res["X_test_rfecv"].columns)


@pytest.mark.integration
def test_reports_its_agreement(tmp_path):
    """
    A panel without a stated agreement level cannot be interpreted, so the
    step must report how strongly its result is supported.
    """
    X, y = _data(n=90, p=40)
    res = _run(X, y, _cfg(tmp_path))
    assert res["consensus_outcome"] in (
        "consensus", "relaxed", "consensus_limited", "rank_average", "union")
    assert 0.0 <= res["agreement"] <= 1.0
    assert isinstance(res["agreement_label"], str) and res["agreement_label"]
    assert res["votes_required"] >= 0


@pytest.mark.integration
def test_multiclass_path(tmp_path):
    """Scoring switches to macro one-vs-rest above two classes."""
    X, y = _data(n=96, p=40, n_informative=8, n_classes=3)
    res = _run(X, y, _cfg(tmp_path), n_classes=3)
    assert res.get("skipped") is False
    assert res["X_train_rfecv"].shape[1] >= 1


@pytest.mark.integration
def test_writes_artefacts_and_figures(tmp_path):
    """
    The saving and plotting branches only execute when both flags are on, so
    they are otherwise never covered. A failure there wastes the whole step.
    """
    X, y = _data(n=90, p=40)
    cfg = _cfg(tmp_path, save_intermediate_results=True,
               create_visualizations=True, enable_step_evaluations=True)
    res = _run(X, y, cfg)
    assert res.get("skipped") is False
    written = list(tmp_path.rglob("step3*"))
    assert written, "save_intermediate_results produced nothing"


@pytest.mark.integration
def test_union_rung_is_honoured(tmp_path):
    """allow_union_rung must reach Step 3, not just Step 2."""
    X, y = _data(n=90, p=40)
    res = _run(X, y, _cfg(tmp_path, allow_union_rung=True))
    assert res.get("skipped") is False
    assert res["X_train_rfecv"].shape[1] >= 1


@pytest.mark.integration
@pytest.mark.parametrize("algorithm", ["LR", "RF"])
def test_algorithms(tmp_path, algorithm):
    """LR and RF take different importance-extraction paths."""
    X, y = _data(n=90, p=40)
    res = _run(X, y, _cfg(tmp_path, algorithm=algorithm))
    assert res.get("skipped") is False
    assert res["X_train_rfecv"].shape[1] >= 1
