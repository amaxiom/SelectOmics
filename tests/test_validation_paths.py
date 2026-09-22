"""
Feature-set validation and the recommendation it produces.

``build_recommendation`` decides which step's feature set the pipeline puts
forward and what confidence label it carries, and was entirely uncovered. The
three validation protocols were covered only on their happy path, so their
degenerate branches (a fold with one class, an empty feature set, a failed fit)
went untested even though those are the cases small-n omics data produces.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from SelectOmics.config import SelectOmicsConfig
from SelectOmics.evaluation.validation import (
    FeatureSetValidator,
    _ci_width_flag,
    _worse_flag,
    build_recommendation,
)


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    n, p = 60, 12
    X = pd.DataFrame(rng.randn(n, p), columns=[f"f{i:02d}" for i in range(p)])
    y = np.array([i % 2 for i in range(n)])
    X.iloc[:, :4] += 1.8 * y[:, None]
    return X, y


@pytest.fixture
def validator(data):
    """
    _create_model looks the algorithm up in tuned_pipelines, so an empty dict
    raises KeyError. The validator needs the real tuned pipeline the pipeline
    would have handed it.
    """
    from SelectOmics.models.base import quick_tune_all
    X, y = data
    cfg = SelectOmicsConfig(data_path="u.csv", target_column="Class",
                            algorithm="LR", n_bootstrap=10, random_seed=0,
                            quick_tune_iterations=2)
    # (X, y, config, device) -- there is no cv parameter.
    tuned = quick_tune_all(X, y, cfg)
    return FeatureSetValidator(cfg, n_classes=2, tuned_pipelines=tuned,
                               n_bootstrap=10)


# ---------------------------------------------------------------------------
# Flag helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("width, expected", [
    (0.05, "adequate"), (0.15, "adequate"),
    (0.25, "caution"), (0.45, "critical"),
])
def test_ci_width_flag_bands(width, expected):
    assert _ci_width_flag(width) == expected


def test_worse_flag_takes_the_worse_of_two():
    assert _worse_flag("adequate", "critical") == "critical"
    assert _worse_flag("caution", "adequate") == "caution"
    assert _worse_flag("adequate", "adequate") == "adequate"


# ---------------------------------------------------------------------------
# The three protocols
# ---------------------------------------------------------------------------

def test_stratified_cv(validator, data):
    X, y = data
    r = validator.stratified_cv_validation(X, y, "all")
    assert 0.0 <= r["mean_auc"] <= 1.0
    assert r["std_auc"] >= 0.0


def test_leave_one_out(validator, data):
    X, y = data
    r = validator.leave_one_out_validation(X, y, "all")
    assert "mean_auc" in r and "std_auc" in r


def test_bootstrap_reports_a_confidence_interval(validator, data):
    X, y = data
    r = validator.bootstrap_validation(X, y, "all")
    lo, hi = r["confidence_interval"]
    assert lo <= r["mean_auc"] <= hi, "mean must lie inside its own CI"
    assert hi >= lo


def test_validate_feature_set_returns_all_three(validator, data):
    X, y = data
    r = validator.validate_feature_set(X, y, "all")
    for key in ("stratified_cv", "leave_one_out", "bootstrap", "n_features"):
        assert key in r, key
    assert r["n_features"] == X.shape[1]


def test_single_feature_set_still_validates(validator, data):
    """One column is a legitimate outcome of an aggressive pipeline."""
    X, y = data
    r = validator.validate_feature_set(X.iloc[:, :1], y, "one")
    assert r["n_features"] == 1


def test_compare_feature_sets(validator, data):
    """Returns (comparison_df, all_results), not a bare frame."""
    X, y = data
    sets = {"all": X, "top4": X.iloc[:, :4]}
    df, all_results = validator.compare_feature_sets(sets, y)
    assert len(df) == 2
    assert "n_features" in df.columns
    assert set(all_results) == {"all", "top4"}
    # The per-set dicts must be the shape build_recommendation and the
    # comparison figure both consume.
    for r in all_results.values():
        assert {"stratified_cv", "leave_one_out", "bootstrap"} <= set(r)


def test_compare_carries_the_upfront_severity(validator, data):
    """
    overall_adequacy is the worse of the CI width and the upfront sample-size
    severity, so a critical upfront flag must not be lost.
    """
    X, y = data
    df, _ = validator.compare_feature_sets({"all": X}, y,
                                           upfront_severity="critical")
    assert "overall_adequacy" in df.columns
    assert (df["overall_adequacy"] == "critical").all(), (
        "a critical upfront sample-size flag must not be downgraded by a "
        "narrow CI"
    )


# ---------------------------------------------------------------------------
# build_recommendation
# ---------------------------------------------------------------------------

def _comparison(rows):
    """
    build_recommendation reads comparison_df.loc[best_id, ...] and
    step_data[best_id], so the frame must be INDEXED by the step key. A
    feature_set column with a default integer index silently yields None.
    """
    df = pd.DataFrame(rows).set_index("feature_set")
    df.index.name = None
    return df


def _step_data(names, X):
    return {
        n: {"step_name": n, "X_train": X,
            "eval_result": {"mean_auc": 0.9, "std_auc": 0.02,
                            "fold_aucs": [0.88, 0.9, 0.92]}}
        for n in names
    }


def test_build_recommendation_picks_the_best_weighted_auc(data):
    X, _ = data
    df = _comparison([
        {"feature_set": "weak", "n_features": 12, "cv_auc": 0.70,
         "cv_std": 0.05, "loo_auc": 0.68, "bootstrap_auc": 0.66},
        {"feature_set": "strong", "n_features": 4, "cv_auc": 0.93,
         "cv_std": 0.02, "loo_auc": 0.91, "bootstrap_auc": 0.90},
    ])
    rec = build_recommendation(df, _step_data(["weak", "strong"], X), X)
    assert rec is not None
    assert rec["step_name"] == "strong"
    assert rec["cv_auc"] == pytest.approx(0.93)


def test_build_recommendation_labels_confidence(data):
    """A weak result must be labelled, not silently recommended."""
    X, _ = data
    weak = _comparison([
        {"feature_set": "poor", "n_features": 12, "cv_auc": 0.55,
         "cv_std": 0.30, "loo_auc": 0.52, "bootstrap_auc": 0.50},
    ])
    rec = build_recommendation(weak, _step_data(["poor"], X), X)
    assert rec is not None
    label = " ".join(str(v) for v in rec.values()).upper()
    assert "NOT RECOMMENDED" in label or "CAUTION" in label


def test_build_recommendation_subsets_the_test_set(data):
    """X_test must be cut to the recommended feature set, not passed whole."""
    X, _ = data
    df = _comparison([
        {"feature_set": "top4", "n_features": 4, "cv_auc": 0.92,
         "cv_std": 0.02, "loo_auc": 0.90, "bootstrap_auc": 0.89},
    ])
    step = {"top4": {"step_name": "top4", "X_train": X.iloc[:, :4],
                     "eval_result": {"mean_auc": 0.92, "std_auc": 0.02,
                                     "fold_aucs": [0.9, 0.93]}}}
    rec = build_recommendation(df, step, X)
    assert rec is not None
    assert list(rec["X_test"].columns) == list(X.columns[:4])


def test_build_recommendation_on_empty_comparison(data):
    """Nothing to recommend must return None, not raise."""
    X, _ = data
    empty = pd.DataFrame(columns=["feature_set", "n_features", "cv_auc",
                                  "cv_std", "loo_auc", "bootstrap_auc"])
    assert build_recommendation(empty, {}, X) is None


# ---------------------------------------------------------------------------
# Degraded-run warnings
#
# Both protocols already counted their failures, but neither surfaced the
# count above DEBUG: a bootstrap CI computed from 5 surviving iterations of
# 100 was reported with the same authority as one from 100.
# ---------------------------------------------------------------------------

class TestDegradedValidationWarning:

    def test_warns_below_the_success_threshold(self, caplog):
        import logging
        from SelectOmics.evaluation.validation import _warn_if_degraded
        caplog.set_level(logging.WARNING)
        _warn_if_degraded("Bootstrap", "all features", 5, 100)
        assert "5 of 100" in caplog.text
        assert "provisional" in caplog.text

    def test_silent_when_mostly_successful(self, caplog):
        import logging
        from SelectOmics.evaluation.validation import _warn_if_degraded
        caplog.set_level(logging.WARNING)
        _warn_if_degraded("Bootstrap", "all features", 90, 100)
        assert caplog.text == ""

    def test_no_division_by_zero(self, caplog):
        import logging
        from SelectOmics.evaluation.validation import _warn_if_degraded
        caplog.set_level(logging.WARNING)
        _warn_if_degraded("LOO", "all features", 0, 0)
        assert caplog.text == ""

    def test_bootstrap_still_reports_its_iteration_count(self, validator, data):
        """The count must remain in the result, not only in the log."""
        X, y = data
        r = validator.bootstrap_validation(X, y, "all")
        assert "successful_iterations" in r
        assert r["successful_iterations"] <= validator.n_bootstrap
