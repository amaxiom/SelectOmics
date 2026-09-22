"""
Pipeline integration tests for SelectOmics.

Default run (pytest -m 'not slow'):
  - Construction and config validation tests (instantaneous)
  - Steps 0, 1, 2 on tiny synthetic data (marked 'integration', ~10-30 s)

Slow run (pytest -m slow):
  - Full pipeline including RFECV and stability selection (minutes)

Run integration tests:
    pytest -m integration

Run everything including slow:
    pytest -m 'integration or slow'
"""
import pytest

import SelectOmics as selectomics
from SelectOmics.config import SelectOmicsConfig


# ---------------------------------------------------------------------------
# Construction tests (no ML, instantaneous)
# ---------------------------------------------------------------------------

class TestPipelineConstruction:
    def test_accepts_dict(self, fast_config):
        p = selectomics.SelectOmicsPipeline(fast_config)
        assert isinstance(p.config, SelectOmicsConfig)

    def test_accepts_config_object(self, fast_config):
        cfg = SelectOmicsConfig.from_dict(fast_config)
        p = selectomics.SelectOmicsPipeline(cfg)
        assert p.config is cfg

    def test_rejects_invalid_type(self):
        with pytest.raises(TypeError):
            selectomics.SelectOmicsPipeline("not_a_config")

    def test_invalid_config_raises_on_construction(self, fast_config):
        fast_config["algorithm"] = "INVALID"
        with pytest.raises(ValueError, match="algorithm"):
            selectomics.SelectOmicsPipeline(fast_config)

    def test_results_initially_empty(self, fast_config):
        p = selectomics.SelectOmicsPipeline(fast_config)
        assert p.results == {}

    def test_output_dir_created(self, fast_config, tmp_path):
        p = selectomics.SelectOmicsPipeline(fast_config)
        assert p.config.output_dir.exists()


# ---------------------------------------------------------------------------
# Step-level integration tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestStep0Reference:
    def test_step0_returns_auc(self, fast_config):
        p = selectomics.SelectOmicsPipeline(fast_config)
        p.load_data()
        p.run_step0_reference()
        r = p.results["step0"]
        assert "reference_evaluation" in r
        summary = r["reference_evaluation"]["summary"]
        mean_auc = summary["mean_aucs"][0]
        assert 0.0 <= mean_auc <= 1.0

    def test_step0_n_features(self, fast_config):
        p = selectomics.SelectOmicsPipeline(fast_config)
        p.load_data()
        p.run_step0_reference()
        r = p.results["step0"]
        assert r["n_features"] == 30


@pytest.mark.integration
class TestStep1Cleaning:
    def test_step1_reduces_features(self, fast_config):
        p = selectomics.SelectOmicsPipeline(fast_config)
        p.load_data()
        p.run_step0_reference()
        p.run_step1_cleaning()
        r = p.results["step1"]
        assert r["n_features_out"] <= r["n_features_in"]
        assert r["n_features_out"] >= 1

    def test_step1_output_matches_next_input(self, fast_config):
        p = selectomics.SelectOmicsPipeline(fast_config)
        p.load_data()
        p.run_step0_reference()
        p.run_step1_cleaning()
        r = p.results["step1"]
        assert r["X_train_clean"].shape[1] == r["n_features_out"]

    def test_step1_disabled_passes_all_features_through(self, fast_config, tmp_path):
        fast_config["enable_step1"] = False
        fast_config["output_dir"] = str(tmp_path / "r2")
        p = selectomics.SelectOmicsPipeline(fast_config)
        p.load_data()
        p.run_step0_reference()
        p.run_step1_cleaning()
        r = p.results["step1"]
        assert r["n_features_in"] == r["n_features_out"]


@pytest.mark.integration
class TestStep2Regularization:
    def test_step2_reduces_features(self, fast_config):
        p = selectomics.SelectOmicsPipeline(fast_config)
        p.load_data()
        p.run_step0_reference()
        p.run_step1_cleaning()
        p.run_step2_regularization()
        r = p.results["step2"]
        assert r["n_features_out"] <= r["n_features_in"]
        assert r["n_features_out"] >= 1

    def test_step2_best_percentiles_in_range(self, fast_config):
        p = selectomics.SelectOmicsPipeline(fast_config)
        p.load_data()
        p.run_step0_reference()
        p.run_step1_cleaning()
        p.run_step2_regularization()
        r = p.results["step2"]
        assert 0.0 <= r["best_l1_percentile"] <= 1.0
        assert 0.0 <= r["best_l2_percentile"] <= 1.0


# ---------------------------------------------------------------------------
# Full pipeline (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestFullPipeline:
    def _full_config(self, fast_config, tmp_path):
        cfg = dict(fast_config)
        cfg["enable_step3"] = True
        cfg["output_dir"] = str(tmp_path / "full_results")
        return cfg

    def test_full_pipeline_runs(self, fast_config, tmp_path):
        cfg = self._full_config(fast_config, tmp_path)
        p = selectomics.SelectOmicsPipeline(cfg)
        results = p.run()
        assert results is not None

    def test_full_pipeline_features_decrease_monotonically(self, fast_config, tmp_path):
        cfg = self._full_config(fast_config, tmp_path)
        p = selectomics.SelectOmicsPipeline(cfg)
        p.run()
        steps = ["step1", "step2", "step3"]
        prev = 30
        for step in steps:
            r = p.results.get(step, {})
            if r.get("skipped"):
                continue
            n_out = r["n_features_out"]
            assert n_out <= prev, f"{step}: {n_out} > previous {prev}"
            prev = n_out

    def test_get_selected_features_returns_list(self, fast_config, tmp_path):
        cfg = self._full_config(fast_config, tmp_path)
        p = selectomics.SelectOmicsPipeline(cfg)
        p.run()
        features = p.get_selected_features()
        assert isinstance(features, list)
        assert len(features) >= 1
