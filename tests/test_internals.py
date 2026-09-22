"""
Utility internals: diagnostics bands, consensus edges, state tracking, metrics.

These are the small modules the pipeline leans on constantly. Their happy paths
are covered incidentally by every integration test; their *bands* and *edges*
are not. A severity threshold that never fires in a test is a threshold nobody
knows still works, and these decide what warnings a user sees about whether
their sample size can support the result at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from SelectOmics.config import SelectOmicsConfig
from SelectOmics.utils.diagnostics import assess_sample_size_adequacy
from SelectOmics.utils.helpers import (
    ConsensusResult,
    agreement_label,
    relax_consensus_intersection,
)
from SelectOmics.utils.state import DataStateTracker


# ---------------------------------------------------------------------------
# Sample-size adequacy: every band of every signal
# ---------------------------------------------------------------------------

def _assess(n, p, n_classes=2, minority=None, cv_splits=5):
    """
    assess_sample_size_adequacy takes (X, y, cv, config) and returns a FLAT
    dict of *_flag keys, not nested per-signal dicts.
    """
    from sklearn.model_selection import StratifiedKFold
    y = np.array([i % n_classes for i in range(n)])
    if minority is not None:
        y = np.array([1] * minority + [0] * (n - minority))
    X = pd.DataFrame(np.zeros((n, p)), columns=[f"f{i}" for i in range(p)])
    cfg = SelectOmicsConfig(data_path="u.csv", target_column="Class")
    return assess_sample_size_adequacy(
        X, y, StratifiedKFold(cv_splits), cfg)


@pytest.mark.parametrize("n, expected", [
    (15, "critical"),    # far below any usable size
    (40, "caution"),
    (400, "adequate"),
])
def test_absolute_sample_count_bands(n, expected):
    a = _assess(n, 50)
    assert a["absolute_n_flag"] == expected


@pytest.mark.parametrize("n, p, expected", [
    (30, 5000, "critical"),   # n/p vanishingly small
    (200, 2000, "caution"),
    (500, 50, "adequate"),
])
def test_samples_to_features_bands(n, p, expected):
    a = _assess(n, p)
    assert a["n_per_p_flag"] == expected


@pytest.mark.parametrize("minority, expected", [
    (3, "critical"),
    (12, "caution"),
    (100, "adequate"),
])
def test_samples_per_class_bands(minority, expected):
    a = _assess(220, 50, minority=minority)
    assert a["per_class_flag"] == expected


@pytest.mark.parametrize("n, cv_splits, expected", [
    # The bands are on min_class_size against n_folds, not on n:
    #   < n_folds       -> critical (folds would be degenerate)
    #   < 2 * n_folds   -> caution  (some folds hold one minority sample)
    #   otherwise       -> adequate
    (12, 10, "critical"),    # min class 6  < 10
    (30, 10, "caution"),     # min class 15 < 20
    (60, 10, "adequate"),    # min class 30 >= 20
    (600, 5, "adequate"),
])
def test_cv_fold_reliability_bands(n, cv_splits, expected):
    a = _assess(n, 40, cv_splits=cv_splits)
    assert a["cv_reliability_flag"] == expected, (
        f"min_class_size={a['min_class_size']} n_folds={a['n_folds']}")


def test_overall_severity_is_the_worst_signal():
    """One critical signal must not be averaged away by three adequate ones."""
    a = _assess(500, 50, minority=3)
    assert a["overall_severity"] == "critical"


def test_fully_adequate_data_produces_no_warnings():
    a = _assess(800, 40)
    assert a["overall_severity"] == "adequate"
    assert not a["warnings"]


def test_reference_cache_round_trips():
    """The reference result is cached per (algorithm, feature set)."""
    t = DataStateTracker()
    X = pd.DataFrame(np.zeros((4, 3)), columns=list("abc"))
    payload = {"mean_auc": 0.9}
    t.cache_reference(X, "XGB", payload)
    assert t.get_cached_reference(X, "XGB") == payload
    assert t.get_cached_reference(X, "RF") is None
    assert t.get_cached_reference(X[["a"]], "XGB") is None


# ---------------------------------------------------------------------------
# agreement_label bands
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("frac, expected", [
    (1.00, "unanimous"), (0.90, "strong"), (0.80, "strong"),
    (0.70, "moderate"), (0.60, "moderate"),
    (0.50, "weak"), (0.40, "weak"),
    (0.20, "minimal"), (0.00, "minimal"),
])
def test_agreement_label_bands(frac, expected):
    assert agreement_label(frac) == expected


def test_rank_average_overrides_the_bands():
    """A ranked result carries no agreement, whatever the fraction says."""
    assert "ranked" in agreement_label(0.9, how="rank_average")


# ---------------------------------------------------------------------------
# relax_consensus_intersection edges
# ---------------------------------------------------------------------------

def test_unanimity_when_the_floor_is_already_met():
    votes = np.array([5, 5, 5, 5, 0, 0])
    r = relax_consensus_intersection([votes, votes], n_models=5, min_features=3)
    assert r.how == "consensus"
    assert r.votes_required == 5
    assert r.mask.sum() == 4


def test_relaxes_only_as_far_as_needed():
    a = np.array([5, 4, 3, 2, 1, 0])
    b = np.array([5, 4, 3, 2, 1, 0])
    r = relax_consensus_intersection([a, b], n_models=5, min_features=3)
    assert r.how == "relaxed"
    assert r.votes_required == 3, "descended further than the floor required"


def test_min_consensus_stops_the_descent():
    a = np.array([5, 1, 1, 1, 1, 1])
    r = relax_consensus_intersection([a, a], n_models=5, min_features=6,
                                     min_consensus=0.8)
    assert r.how == "consensus_limited"
    assert r.votes_required >= 4


def test_floor_is_capped_at_the_ceiling():
    """Asking for more than the votes support must not pad with unvoted features."""
    a = np.array([5, 5, 0, 0, 0, 0])
    r = relax_consensus_intersection([a, a], n_models=5, min_features=6,
                                     mean_importance=np.arange(6)[::-1] * 1.0)
    assert r.floor_used <= 2
    assert r.floor_requested == 6


def test_rank_average_is_the_terminal_fallback():
    """No feature voted in every stage leaves ranking as the only option."""
    a = np.array([5, 5, 0, 0])
    b = np.array([0, 0, 5, 5])          # disjoint: intersection is empty
    r = relax_consensus_intersection([a, b], n_models=5, min_features=2,
                                     mean_importance=np.array([4.0, 3, 2, 1]))
    assert r.how == "rank_average"
    assert r.votes_required == 0
    assert r.mask.sum() == 2


def test_rank_average_stays_inside_the_voted_pool():
    """Ranking must not reach for features no model selected anywhere."""
    a = np.array([5, 0, 0, 0])
    b = np.array([0, 5, 0, 0])
    r = relax_consensus_intersection([a, b], n_models=5, min_features=2,
                                     mean_importance=np.array([1.0, 1.0, 9.0, 9.0]))
    chosen = set(np.flatnonzero(r.mask))
    assert chosen <= {0, 1}, "ranked outside the voted pool despite votes existing"


def test_consensus_result_carries_its_label():
    a = np.array([5, 5, 5])
    r = relax_consensus_intersection([a, a], n_models=5, min_features=2)
    assert isinstance(r, ConsensusResult)
    assert r.label == "unanimous"
    assert r.agreement == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# DataStateTracker
# ---------------------------------------------------------------------------

def _add(tracker, name, p):
    """The tracker exposes update(step_name, X_train, X_test), not add(state)."""
    X = pd.DataFrame(np.zeros((4, p)), columns=[f"f{i}" for i in range(p)])
    tracker.update(name, X, X.copy())


def test_tracker_records_and_retrieves():
    t = DataStateTracker()
    _add(t, "Step 0", 10)
    _add(t, "Step 1", 6)
    assert t.get_latest().X_train.shape[1] == 6
    assert t.get_step("Step 0").X_train.shape[1] == 10
    assert t.n_steps == 2
    assert list(t.step_names) == ["Step 0", "Step 1"]


def test_get_latest_on_empty_tracker_raises():
    """Returning None here would fail confusingly several steps later."""
    with pytest.raises(RuntimeError):
        DataStateTracker().get_latest()


def test_get_step_unknown_returns_none():
    t = DataStateTracker()
    _add(t, "Step 0", 4)
    assert t.get_step("Step 9") is None


def test_clear_resets_everything():
    t = DataStateTracker()
    _add(t, "Step 0", 4)
    t.clear()
    assert t.n_steps == 0
    with pytest.raises(RuntimeError):
        t.get_latest()


def test_get_step_returns_the_most_recent_of_a_repeated_name():
    t = DataStateTracker()
    _add(t, "Step 1", 10)
    _add(t, "Step 1", 3)
    assert t.get_step("Step 1").X_train.shape[1] == 3
