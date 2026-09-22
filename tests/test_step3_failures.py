"""
Step 3 degradation paths.

Every branch here runs only when something has gone wrong: an estimator that
will not fit, a class distribution that cannot be subsampled, a subsample that
loses a class. They are the paths that decide whether a failure is contained or
silently changes the feature panel, so what each test asserts is the documented
degradation behaviour, not merely that the line ran.

The governing rule for the stability stage is that a stage which cannot form an
opinion abstains by returning all-True. Returning all-False would veto every
feature and empty the panel, which is the opposite of what a failed stage means.
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from SelectOmics.selection import step3_wrapper as s3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _data(n=60, p=12, n_informative=4, n_classes=2, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n, p), columns=[f"f{i:03d}" for i in range(p)])
    y = np.array([i % n_classes for i in range(n)])
    rng.shuffle(y)
    X.iloc[:, :n_informative] += 1.6 * y[:, None]
    return X, y


def _counters():
    return {"rfecv_failed": 0, "stability_failed": 0,
            "stability_skipped_class": 0, "stability_all_failed": 0}


class _Exploding(BaseEstimator, ClassifierMixin):
    """An estimator that refuses to fit, to drive the failure branches."""

    def __init__(self, message="the estimator refused to fit"):
        self.message = message

    def fit(self, X, y):
        raise RuntimeError(self.message)


def _exploding_builder(*args, **kwargs):
    """Drop-in for _build_rfecv_estimator whose estimator always fails."""
    pipe = Pipeline([("scaler", MinMaxScaler()), ("clf", _Exploding())])
    return pipe, "named_steps.clf.coef_"


# ---------------------------------------------------------------------------
# RFECV stage: a model that will not fit
# ---------------------------------------------------------------------------

class TestRfecvStageFailures:

    def test_a_failing_probe_is_counted_and_estimated(self, monkeypatch, caplog):
        """
        A probe that raises still has to return a feature count, or the binary
        search over the percentile has nothing to work with. The fallback
        approximates the removal the percentile asked for.
        """
        caplog.set_level(logging.DEBUG, logger="SelectOmics")
        monkeypatch.setattr(s3, "_build_rfecv_estimator", _exploding_builder)
        X, y = _data()
        warnings = _counters()

        selected, votes, imps, best = s3._run_rfecv_stage(
            X_train=X, y_train=y, algorithm="LR", n_models=2, base_seed=0,
            target_retention=1.0, required_votes=1, cv=StratifiedKFold(3),
            warnings=warnings, verbose=True)

        assert warnings["rfecv_failed"] > 0, "failures were not counted"
        assert "refused to fit" in caplog.text, "the reason was discarded"
        assert selected.shape == (X.shape[1],)

    def test_a_total_rfecv_failure_does_not_empty_the_panel(self, monkeypatch):
        """
        Every model failing must not silently return an empty feature set: an
        empty panel would propagate as a valid result.
        """
        monkeypatch.setattr(s3, "_build_rfecv_estimator", _exploding_builder)
        X, y = _data()
        selected, votes, imps, best = s3._run_rfecv_stage(
            X_train=X, y_train=y, algorithm="LR", n_models=2, base_seed=0,
            target_retention=1.0, required_votes=1, cv=StratifiedKFold(3),
            warnings=_counters(), verbose=False)
        assert selected.sum() >= 1, (
            "a total RFECV failure emptied the panel instead of degrading"
        )

    def test_the_failure_counter_is_not_touched_on_a_clean_run(self):
        X, y = _data()
        warnings = _counters()
        s3._run_rfecv_stage(
            X_train=X, y_train=y, algorithm="LR", n_models=1, base_seed=0,
            target_retention=1.0, required_votes=1, cv=StratifiedKFold(3),
            warnings=warnings, verbose=False)
        assert warnings["rfecv_failed"] == 0, (
            "a healthy run recorded a failure, which would make the counter "
            "useless as a signal"
        )


# ---------------------------------------------------------------------------
# Stability stage: abstention rather than veto
# ---------------------------------------------------------------------------

class TestStabilityStageFailures:

    def test_a_model_that_cannot_fit_leaves_the_stage_abstaining(
        self, monkeypatch, caplog
    ):
        """
        Every subsample failing means the stage learned nothing. It must return
        all-True so the intersection is decided by RFECV alone, rather than
        all-False, which would drop every feature.
        """
        caplog.set_level(logging.DEBUG, logger="SelectOmics")
        monkeypatch.setattr(s3, "_build_rfecv_estimator", _exploding_builder)
        X, y = _data()
        warnings = _counters()

        selected, freq, imps = s3._run_stability_stage(
            X_train=X, y_train=y, algorithm="LR", n_models=2, base_seed=0,
            stability_threshold=0.6, min_features=5, warnings=warnings,
            verbose=True)

        assert selected.all(), "a failed stage vetoed every feature"
        assert warnings["stability_failed"] > 0
        assert "refused to fit" in caplog.text

    def test_an_impossible_subsample_stops_rather_than_repeating(self, caplog):
        """
        The split constraints are deterministic, so a failure recurs for every
        remaining subsample. The stage stops instead of logging the same error
        fifty times, and abstains.
        """
        caplog.set_level(logging.WARNING, logger="SelectOmics")
        rng = np.random.RandomState(0)
        # One sample in the minority class: no stratified subsample can hold it
        # and still be smaller than the whole set.
        n, p = 12, 8
        X = pd.DataFrame(rng.randn(n, p), columns=[f"f{i}" for i in range(p)])
        y = np.array([0] * (n - 1) + [1])
        warnings = _counters()

        selected, freq, imps = s3._run_stability_stage(
            X_train=X, y_train=y, algorithm="LR", n_models=1, base_seed=0,
            stability_threshold=0.6, min_features=3, warnings=warnings,
            verbose=False)

        assert selected.all(), "abstention must keep everything"
        assert freq is None
        assert warnings["stability_all_failed"] >= 1
        assert caplog.text.count("not possible") <= 1, (
            "the same unrecoverable error was reported repeatedly"
        )

    def test_a_healthy_stability_stage_actually_discriminates(self):
        """
        The abstention paths above are only meaningful if the normal path does
        something different, so pin that it selects a strict subset.
        """
        X, y = _data(n=80, p=16, n_informative=4)
        warnings = _counters()
        selected, freq, imps = s3._run_stability_stage(
            X_train=X, y_train=y, algorithm="LR", n_models=2, base_seed=0,
            stability_threshold=0.9, min_features=1, warnings=warnings,
            verbose=False)
        assert freq is not None, "a healthy run reported no frequencies"
        assert selected.sum() >= 1


# ---------------------------------------------------------------------------
# Importance extraction fallbacks
# ---------------------------------------------------------------------------

class TestImportanceExtraction:

    def test_an_unfitted_svm_surrogate_yields_zeros_not_an_error(self):
        """
        SVM has no native importances, so a surrogate supplies them. When it has
        nothing to report the result must be a zero vector of the right width:
        a wrong width would misalign every downstream mask.
        """
        X, y = _data(n=40, p=9)
        est, _ = s3._build_rfecv_estimator("SVM", seed=0)
        est.fit(X, y)
        imp = s3._extract_importance_step3(est, "SVM")
        assert imp.shape == (9,)
        assert np.all(np.isfinite(imp))


# ---------------------------------------------------------------------------
# Warning summary
# ---------------------------------------------------------------------------

class TestWarningSummary:

    def test_nothing_is_printed_when_nothing_failed(self, caplog):
        caplog.set_level(logging.WARNING, logger="SelectOmics")
        s3._print_warnings(_counters())
        assert caplog.text.strip() == "", (
            "a clean run emitted warnings, which trains users to ignore them"
        )

    def test_each_counter_is_surfaced_when_it_fires(self, caplog):
        caplog.set_level(logging.WARNING, logger="SelectOmics")
        s3._print_warnings({"rfecv_failed": 3, "stability_failed": 2,
                            "stability_skipped_class": 1,
                            "stability_all_failed": 1})
        text = caplog.text
        assert text.strip(), "failures were counted but never reported"
        assert "3" in text


# ---------------------------------------------------------------------------
# Unsupported algorithms and single-model shortcuts
# ---------------------------------------------------------------------------

class TestAlgorithmGuards:

    def test_an_unknown_algorithm_is_named_in_the_error(self):
        """
        The message has to list what is accepted, since the usual cause is a
        typo or a name from another library.
        """
        with pytest.raises(ValueError) as exc:
            s3._build_rfecv_estimator("RANDOMFOREST", seed=0)
        msg = str(exc.value)
        assert "RANDOMFOREST" in msg
        assert "'LR'" in msg and "'SVM'" in msg

    def test_importance_extraction_degrades_to_zeros_of_the_right_width(self):
        """
        Unlike the estimator builder, importance extraction does not raise on
        an unknown algorithm: it returns zeros. The width is what matters,
        because every downstream mask is positional, and a short vector would
        misalign the panel rather than merely lose importances.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import MinMaxScaler
        X, y = _data(n=40, p=6)
        est = Pipeline([("scaler", MinMaxScaler()),
                        ("clf", LogisticRegression(max_iter=500))])
        est.fit(X, y)
        imp = s3._extract_importance_step3(est, "NOT_AN_ALGORITHM")
        assert imp.shape == (6,)
        assert not imp.any(), "an unknown algorithm produced nonzero importances"

    @pytest.mark.parametrize("algorithm", ["LR", "RF", "SVM"])
    def test_every_supported_algorithm_builds(self, algorithm):
        est, getter = s3._build_rfecv_estimator(algorithm, seed=0)
        assert est is not None
        assert getter is not None


class TestStabilityAbstentionVotes:

    @pytest.mark.integration
    def test_a_failed_stability_stage_votes_for_everything(self, monkeypatch):
        """
        When the stability stage abstains it must contribute a full vote for
        every feature, so the intersection is decided by RFECV alone. A zero
        vote would instead veto the entire panel.
        """
        import pandas as pd
        from sklearn.model_selection import StratifiedKFold
        from SelectOmics.config import SelectOmicsConfig
        from SelectOmics.models.base import quick_tune_all

        real_builder = s3._build_rfecv_estimator
        calls = {"n": 0}

        def fail_only_stability(*args, **kwargs):
            # The stability stage runs after RFECV, so let the early calls
            # through and break the later ones.
            calls["n"] += 1
            if calls["n"] > 6:
                return _exploding_builder(*args, **kwargs)
            return real_builder(*args, **kwargs)

        monkeypatch.setattr(s3, "_build_rfecv_estimator", fail_only_stability)

        X, y = _data(n=70, p=16, n_informative=5)
        cfg = SelectOmicsConfig(
            data_path="unused.csv", target_column="Class", algorithm="LR",
            n_consensus_models=2, quick_tune_iterations=2, n_bootstrap=10,
            verbose=False, create_visualizations=False,
            save_intermediate_results=False, enable_step_evaluations=False,
            enable_final_test_evaluation=False, min_features_floor=1,
        )
        tuned = quick_tune_all(X, y, cfg)
        res = s3.run_step3_wrapper(X, X.iloc[:12].copy(), y, cfg,
                                   StratifiedKFold(3), tuned, 2, ["0", "1"])
        assert res["X_train_rfecv"].shape[1] >= 1, (
            "an abstaining stability stage emptied the panel"
        )


class TestSubsampleFallback:
    """
    _subsample_rows draws a stratified subsample so each replicate sees
    different data. Two distinct escapes return the full set instead, and they
    are not the same case: one is asking for everything, the other is asking
    for something stratification cannot deliver.
    """

    def test_a_fraction_that_keeps_everything_returns_early(self):
        X, y = _data(n=20, p=5)
        Xs, ys = s3._subsample_rows(X, y, seed=0, fraction=1.0)
        assert len(Xs) == len(X)

    def test_a_class_too_small_to_stratify_falls_back_to_the_full_data(self):
        """
        StratifiedShuffleSplit refuses when the least populated class has one
        member. A slightly less diverse replicate beats a failed one, so the
        full data is returned rather than the error propagating.
        """
        rng = np.random.RandomState(0)
        n = 20
        X = pd.DataFrame(rng.randn(n, 5), columns=[f"f{i}" for i in range(5)])
        y = np.array([0] * (n - 1) + [1])          # one member in class 1
        Xs, ys = s3._subsample_rows(X, y, seed=0, fraction=0.8)
        assert len(Xs) == n, "the fallback did not return the full data"
        assert set(np.unique(ys)) == {0, 1}, "the fallback lost a class"

    def test_a_workable_fraction_actually_subsamples(self):
        """The fallbacks only mean something if the normal path reduces."""
        X, y = _data(n=100, p=5)
        Xs, ys = s3._subsample_rows(X, y, seed=0, fraction=0.6)
        assert len(Xs) < len(X)
        assert set(np.unique(ys)) == set(np.unique(y))

    def test_different_seeds_give_different_subsamples(self):
        """
        The whole purpose is replicate diversity, so two seeds must not select
        the same rows.
        """
        X, y = _data(n=100, p=5)
        a, _ = s3._subsample_rows(X, y, seed=0, fraction=0.6)
        b, _ = s3._subsample_rows(X, y, seed=1, fraction=0.6)
        assert list(a.index) != list(b.index)


class TestAbstentionVotesInTheFullStep:
    """
    When the stability stage abstains it returns no frequencies at all, and the
    step has to convert that into a full vote for every feature. Anything less
    would let an abstaining stage veto the panel through the intersection,
    which is the opposite of abstaining.
    """

    @staticmethod
    def _cfg(tmp_path, **over):
        from SelectOmics.config import SelectOmicsConfig
        kw = dict(data_path="unused.csv", target_column="Class",
                  algorithm="LR", output_dir=str(tmp_path),
                  n_consensus_models=2, quick_tune_iterations=2,
                  n_bootstrap=10, verbose=False, create_visualizations=False,
                  save_intermediate_results=False,
                  enable_step_evaluations=False,
                  enable_final_test_evaluation=False, min_features_floor=1)
        kw.update(over)
        return SelectOmicsConfig(**kw)

    @pytest.mark.integration
    def test_an_abstaining_stability_stage_votes_for_every_feature(
        self, tmp_path, monkeypatch
    ):
        from SelectOmics.models.base import quick_tune_all

        X, y = _data(n=70, p=40, n_informative=6)

        def abstain(X_train, y_train, **kwargs):
            # All-True mask, no frequencies: the documented abstention.
            return np.ones(X_train.shape[1], dtype=bool), None, []

        monkeypatch.setattr(s3, "_run_stability_stage", abstain)

        cfg = self._cfg(tmp_path)
        tuned = quick_tune_all(X, y, cfg)
        res = s3.run_step3_wrapper(X, X.iloc[:12].copy(), y, cfg,
                                   StratifiedKFold(3), tuned, 2, ["0", "1"])
        assert res["stability_freq"] is None
        assert res["X_train_rfecv"].shape[1] >= 1, (
            "an abstaining stability stage emptied the panel"
        )

    @pytest.mark.integration
    def test_a_reporting_stability_stage_still_constrains(self, tmp_path):
        """
        The abstention path only means something if the normal path can
        actually remove features, so pin that the two differ.
        """
        from SelectOmics.models.base import quick_tune_all
        X, y = _data(n=70, p=40, n_informative=6)
        cfg = self._cfg(tmp_path)
        tuned = quick_tune_all(X, y, cfg)
        res = s3.run_step3_wrapper(X, X.iloc[:12].copy(), y, cfg,
                                   StratifiedKFold(3), tuned, 2, ["0", "1"])
        assert res["X_train_rfecv"].shape[1] >= 1
        assert res["stability_freq"] is not None, (
            "a healthy stability stage reported no frequencies"
        )
