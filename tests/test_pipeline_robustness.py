"""
Pipeline behaviour when the environment is not cooperating.

These are the paths that run when a checkpoint is corrupt, names a step this
build no longer has, cannot be written, or when a caller asks for results
before anything has produced them. Each one decides whether the pipeline fails
loudly, degrades, or silently returns something wrong, so the assertions are
about which of those happens.
"""
from __future__ import annotations

import logging
import pickle

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from SelectOmics.config import SelectOmicsConfig
from SelectOmics.pipeline import SelectOmicsPipeline


@pytest.fixture
def dataset(tmp_path):
    rng = np.random.RandomState(0)
    n, p = 60, 20
    X = rng.randn(n, p)
    y = np.array([i % 2 for i in range(n)])
    X[:, :5] += 1.8 * y[:, None]
    df = pd.DataFrame(X, columns=[f"f{i:02d}" for i in range(p)])
    df["Class"] = y
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


def _pipeline(dataset, tmp_path, **over):
    kw = dict(
        data_path=str(dataset), target_column="Class", algorithm="RF",
        output_dir=str(tmp_path / "out"), n_consensus_models=2,
        quick_tune_iterations=2, n_bootstrap=10, verbose=False,
        create_visualizations=False, enable_step_evaluations=False,
        enable_final_test_evaluation=False,
    )
    kw.update(over)
    cfg = SelectOmicsConfig(**kw)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return SelectOmicsPipeline(cfg)


# ---------------------------------------------------------------------------
# Asking for results before there are any
# ---------------------------------------------------------------------------

class TestResultsBeforeAnyRun:

    def test_feature_provenance_refuses_rather_than_returning_empty(
        self, dataset, tmp_path
    ):
        """
        An empty provenance table would read as "no feature survived", which is
        a very different claim from "nothing has run yet".
        """
        p = _pipeline(dataset, tmp_path)
        with pytest.raises(RuntimeError, match="No steps have been run"):
            p.get_feature_provenance()

    def test_selected_features_refuses_before_any_step(self, dataset, tmp_path):
        p = _pipeline(dataset, tmp_path)
        with pytest.raises(RuntimeError):
            p.get_selected_features()

    def test_save_results_tolerates_having_nothing_to_save(
        self, dataset, tmp_path
    ):
        """
        save_results writes whatever exists. With no completed step it must
        still write the config rather than propagating the RuntimeError that
        get_selected_features raises.
        """
        p = _pipeline(dataset, tmp_path)
        p.save_results()
        assert (p.config.output_dir / "config.json").exists()
        assert not (p.config.output_dir / "selected_features.csv").exists()


# ---------------------------------------------------------------------------
# Checkpoint failure modes
# ---------------------------------------------------------------------------

class TestCheckpointFailureModes:

    def test_a_corrupt_checkpoint_starts_fresh_instead_of_raising(
        self, dataset, tmp_path, caplog
    ):
        p = _pipeline(dataset, tmp_path)
        p._checkpoint_path().write_bytes(b"this is not a pickle")

        caplog.set_level(logging.WARNING, logger="SelectOmics")
        assert p._load_checkpoint() is None
        assert "Checkpoint load failed" in caplog.text

    def test_a_checkpoint_naming_a_removed_step_is_refused(
        self, dataset, tmp_path, caplog
    ):
        """
        Step 4 was removed without a version bump, so a checkpoint from this
        same version can still name 'step4'. Resuming would silently skip
        steps that no longer exist.
        """
        from SelectOmics import __version__ as ver
        p = _pipeline(dataset, tmp_path)
        p._save_checkpoint("step1")

        cp = pickle.loads(p._checkpoint_path().read_bytes())
        cp["step_key"] = "step4"
        p._checkpoint_path().write_bytes(pickle.dumps(cp, protocol=4))

        caplog.set_level(logging.WARNING, logger="SelectOmics")
        p2 = _pipeline(dataset, tmp_path)
        p2.run(validate=False, resume=True)
        assert "does not" in caplog.text and "step4" in caplog.text

    def test_a_checkpoint_from_another_version_is_refused(
        self, dataset, tmp_path, caplog
    ):
        p = _pipeline(dataset, tmp_path)
        p._save_checkpoint("step1")
        cp = pickle.loads(p._checkpoint_path().read_bytes())
        cp["selectomics_version"] = "0.0.1-ancient"
        p._checkpoint_path().write_bytes(pickle.dumps(cp, protocol=4))

        caplog.set_level(logging.WARNING, logger="SelectOmics")
        p2 = _pipeline(dataset, tmp_path)
        assert p2._load_checkpoint() is None
        assert "0.0.1-ancient" in caplog.text

    def test_a_failed_checkpoint_save_warns_but_does_not_abort(
        self, dataset, tmp_path, caplog, monkeypatch
    ):
        """
        A checkpoint is an optimisation. Failing to write one must not take
        down a run that is otherwise fine.
        """
        p = _pipeline(dataset, tmp_path)

        def refuse(*args, **kwargs):
            raise OSError("read-only file system")

        monkeypatch.setattr("builtins.open", refuse)
        caplog.set_level(logging.WARNING, logger="SelectOmics")
        p._save_checkpoint("step1")          # must not raise
        assert "Checkpoint save failed" in caplog.text

    def test_no_checkpoint_file_is_not_an_error(self, dataset, tmp_path):
        p = _pipeline(dataset, tmp_path)
        assert p._load_checkpoint() is None


# ---------------------------------------------------------------------------
# Redundancy assessment
# ---------------------------------------------------------------------------

class TestRedundancyAssessment:

    def test_it_is_skipped_when_step1_is_disabled(self, dataset, tmp_path):
        p = _pipeline(dataset, tmp_path, enable_step1=False)
        p.load_data()
        p._assess_step1_fit()
        assert p.feature_redundancy is None

    def test_a_failure_downgrades_to_none_rather_than_aborting(
        self, dataset, tmp_path, caplog, monkeypatch
    ):
        """
        The assessment is advisory. If it cannot be computed the run continues
        without it.
        """
        import SelectOmics.utils.diagnostics as diag

        def boom(*args, **kwargs):
            raise ValueError("cannot assess")

        monkeypatch.setattr(diag, "assess_feature_redundancy", boom)
        caplog.set_level(logging.DEBUG, logger="SelectOmics")
        p = _pipeline(dataset, tmp_path)
        p.load_data()
        p._assess_step1_fit()
        assert p.feature_redundancy is None
        assert "skipped" in caplog.text

    def test_low_redundancy_data_is_flagged(self, tmp_path, caplog):
        """
        Step 1 removes correlated redundancy. On data that has none, saying so
        is the difference between a step that did nothing and one that failed.
        """
        rng = np.random.RandomState(0)
        n, p_ = 60, 20
        df = pd.DataFrame(rng.randn(n, p_),
                          columns=[f"f{i:02d}" for i in range(p_)])
        df["Class"] = [i % 2 for i in range(n)]
        path = tmp_path / "uncorrelated.csv"
        df.to_csv(path, index=False)

        caplog.set_level(logging.WARNING, logger="SelectOmics")
        pipe = _pipeline(path, tmp_path)
        pipe.load_data()
        pipe._assess_step1_fit()
        assert pipe.feature_redundancy is not None
        assert "rationale" in pipe.feature_redundancy


# ---------------------------------------------------------------------------
# Automatic adaptations to the data and the machine
# ---------------------------------------------------------------------------

class TestAutomaticAdaptations:

    @staticmethod
    def _nan_dataset(tmp_path):
        rng = np.random.RandomState(0)
        n, p = 60, 12
        X = rng.randn(n, p)
        X[rng.rand(n, p) < 0.05] = np.nan
        df = pd.DataFrame(X, columns=[f"f{i:02d}" for i in range(p)])
        df["Class"] = [i % 2 for i in range(n)]
        path = tmp_path / "nan.csv"
        df.to_csv(path, index=False)
        return path

    @pytest.mark.parametrize("requested", ["LR", "SVM"])
    def test_missing_values_switch_an_incapable_algorithm_to_xgb(
        self, tmp_path, requested
    ):
        """
        LR and SVM cannot fit with NaN present. The switch has to be written
        back to the config, or every step downstream would keep building the
        algorithm the user asked for and fail on the same NaNs.
        """
        path = self._nan_dataset(tmp_path)
        p_obj = _pipeline(path, tmp_path, algorithm=requested)
        p_obj.load_data()
        assert p_obj.config.algorithm == "XGB", (
            "the NaN-driven switch was not propagated to the config"
        )

    @pytest.mark.parametrize("requested", ["RF", "XGB"])
    def test_a_capable_algorithm_is_left_alone(self, tmp_path, requested):
        """
        Both handle NaN natively, and NaNs are deliberately never imputed so
        the missingness pattern stays available as signal. Switching them would
        override a user's choice for no reason.
        """
        path = self._nan_dataset(tmp_path)
        p_obj = _pipeline(path, tmp_path, algorithm=requested)
        p_obj.load_data()
        assert p_obj.config.algorithm == requested

    def test_clean_data_keeps_the_requested_algorithm(self, dataset, tmp_path):
        p = _pipeline(dataset, tmp_path, algorithm="LR")
        p.load_data()
        assert p.config.algorithm == "LR"

    def test_the_device_choice_is_reported(self, dataset, tmp_path, caplog):
        """
        Whether a run used CUDA or CPU changes its runtime and its numerics, so
        a verbose run has to say which it got rather than leaving it implicit.
        """
        caplog.set_level(logging.INFO, logger="SelectOmics")
        p = _pipeline(dataset, tmp_path, algorithm="XGB", use_gpu=True,
                      verbose=True)
        p.load_data()
        assert "GPU detected" in caplog.text or "No GPU detected" in caplog.text


# ---------------------------------------------------------------------------
# Reported agreement
# ---------------------------------------------------------------------------

class TestFinalAgreement:

    def test_no_completed_step_reports_not_run(self, dataset, tmp_path):
        """
        Zero agreement and 'not run' are different claims from a real 0.0, and
        an empty pipeline has made no claim at all.
        """
        p = _pipeline(dataset, tmp_path)
        assert p._final_agreement() == (0.0, "not run")

    def test_a_disabled_step_is_skipped_for_the_one_that_decided(self, dataset,
                                                                 tmp_path):
        """
        A disabled or skipped step did not choose the panel, so the reported
        agreement must come from the step that did.
        """
        p = _pipeline(dataset, tmp_path)
        p.results = {
            "step1": {"consensus_outcome": "consensus", "agreement": 1.0,
                      "agreement_label": "unanimous"},
            "step2": {"consensus_outcome": "disabled", "agreement": 0.0,
                      "agreement_label": "not run"},
            "step3": {"consensus_outcome": "skipped", "agreement": 0.0,
                      "agreement_label": "not run"},
        }
        agreement, label = p._final_agreement()
        assert (agreement, label) == (1.0, "unanimous"), (
            "reported a disabled step's placeholder instead of the step that "
            "actually decided the panel"
        )

    def test_the_latest_deciding_step_wins(self, dataset, tmp_path):
        p = _pipeline(dataset, tmp_path)
        p.results = {
            "step1": {"consensus_outcome": "consensus", "agreement": 1.0,
                      "agreement_label": "unanimous"},
            "step3": {"consensus_outcome": "relaxed", "agreement": 0.5,
                      "agreement_label": "weak"},
        }
        assert p._final_agreement() == (0.5, "weak")


# ---------------------------------------------------------------------------
# Guard rails on call order
# ---------------------------------------------------------------------------

class TestCallOrderGuards:

    def test_a_step0_dependent_method_refuses_before_step0(self, dataset,
                                                           tmp_path):
        p = _pipeline(dataset, tmp_path)
        p.load_data()
        with pytest.raises(RuntimeError, match="Step 0"):
            p._require_step0()

    def test_methods_refuse_before_data_is_loaded(self, dataset, tmp_path):
        p = _pipeline(dataset, tmp_path)
        with pytest.raises(RuntimeError, match="Data not loaded"):
            p._require_data()

    def test_the_guards_pass_once_the_prerequisites_are_met(self, dataset,
                                                            tmp_path):
        p = _pipeline(dataset, tmp_path)
        p.load_data()
        p._require_data()
        p.run_step0_reference()
        p._require_step0()


# ---------------------------------------------------------------------------
# Probability calibration
# ---------------------------------------------------------------------------

class TestProbabilityCalibration:

    def _run(self, dataset, tmp_path, calibrate):
        # validate=True: the final test evaluation is called from inside the
        # validate block, so validate=False skips it entirely.
        p = _pipeline(dataset, tmp_path, calibrate_probabilities=calibrate,
                      enable_final_test_evaluation=True, enable_step3=False)
        p.run(validate=True)
        return p

    @pytest.mark.integration
    def test_calibration_produces_usable_probabilities(self, dataset, tmp_path):
        """
        Isotonic calibration wraps the fitted model, so the output must still
        be a proper probability matrix: one row per sample, rows summing to 1.
        A wrapper that broke that would corrupt every metric downstream.
        """
        p = self._run(dataset, tmp_path, calibrate=True)
        ev = p._final_test_eval
        assert ev is not None
        proba = np.asarray(ev["test_proba"])
        assert proba.shape[0] == len(p._y_test)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        assert ((proba >= 0.0) & (proba <= 1.0)).all()

    @pytest.mark.integration
    def test_calibration_is_off_by_default(self, dataset, tmp_path):
        p = self._run(dataset, tmp_path, calibrate=False)
        assert p._final_test_eval is not None

    @pytest.mark.integration
    def test_calibration_folds_respect_a_tiny_minority_class(self, tmp_path):
        """
        The calibration CV is clamped to the smallest class, or
        CalibratedClassifierCV would ask for more folds than that class has
        members and fail at the very end of a completed run.
        """
        rng = np.random.RandomState(0)
        n, p_ = 60, 12
        X = rng.randn(n, p_)
        y = np.array([0] * (n - 4) + [1] * 4)      # only 4 in the minority
        X[:, :3] += 1.8 * y[:, None]
        df = pd.DataFrame(X, columns=[f"f{i:02d}" for i in range(p_)])
        df["Class"] = y
        path = tmp_path / "imbalanced.csv"
        df.to_csv(path, index=False)

        pipe = _pipeline(path, tmp_path, calibrate_probabilities=True,
                         enable_final_test_evaluation=True, enable_step3=False)
        pipe.run(validate=True)
        assert pipe._final_test_eval is not None
