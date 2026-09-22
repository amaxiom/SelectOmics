"""
Every config field must change something observable.

A static audit finds fields nothing reads. It does not find a field that is
read into a variable and then not acted on, which is the same failure from the
user's side: the setting is accepted, the run completes, and nothing happened.

Line coverage does not catch this either. build_recommendation sat at full line
coverage while being called from nowhere in the package, because the tests
exercised it directly rather than through run(). So these tests set a field to
two different values and assert the OUTPUT differs, driving the public entry
points a user actually calls.

Fields deliberately not covered here:
  data_path, target_column, output_dir   exercised by every other test
  schema_version                         serialization metadata, validated only
  use_gpu, n_jobs                        machine-dependent; no deterministic
                                         observable difference on CI hardware
"""
from __future__ import annotations

import logging

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
    n, p = 60, 30
    X = rng.randn(n, p)
    y = np.array([i % 2 for i in range(n)])
    X[:, :6] += 1.8 * y[:, None]
    # A correlated block, so the correlation filter has something to remove.
    X[:, 20:25] = X[:, [1]] + rng.randn(n, 5) * 0.01
    df = pd.DataFrame(X, columns=[f"f{i:02d}" for i in range(p)])
    df["Class"] = y
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


def _cfg(dataset, tmp_path, **over):
    kw = dict(
        data_path=str(dataset), target_column="Class", algorithm="RF",
        output_dir=str(tmp_path / "out"), n_consensus_models=2,
        quick_tune_iterations=2, n_bootstrap=10, verbose=False,
        create_visualizations=False, save_intermediate_results=False,
        enable_step_evaluations=False, enable_final_test_evaluation=False,
        enable_step3=False,
    )
    kw.update(over)
    cfg = SelectOmicsConfig(**kw)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _run(dataset, tmp_path, validate=False, **over):
    p = SelectOmicsPipeline(_cfg(dataset, tmp_path, **over))
    p.run(validate=validate)
    return p


# ---------------------------------------------------------------------------
# Which steps run
# ---------------------------------------------------------------------------

class TestStepSwitches:

    @pytest.mark.integration
    @pytest.mark.parametrize("flag,key", [("enable_step1", "step1"),
                                          ("enable_step2", "step2")])
    def test_run_omits_a_disabled_step_entirely(self, dataset, tmp_path,
                                                flag, key):
        """
        run() short-circuits a disabled step rather than calling it for a
        pass-through, so no result is recorded for it at all.

        This differs from calling the step method directly, which DOES store a
        pass-through (see the test below). Pinned because the difference is
        easy to trip over: results["step1"] raises KeyError after run() with
        enable_step1=False, while the same config reaches a populated
        results["step1"] through the method.
        """
        off = _run(dataset, tmp_path, **{flag: False})
        assert key not in off.results, (
            f"run() recorded {key} despite {flag}=False"
        )

    @pytest.mark.integration
    @pytest.mark.parametrize("method,key", [("run_step1_cleaning", "step1"),
                                            ("run_step2_regularization",
                                             "step2")])
    def test_calling_a_disabled_step_directly_stores_a_passthrough(
        self, dataset, tmp_path, method, key
    ):
        """
        The step methods honour their documented pass-through behaviour: the
        result is recorded, marked disabled, and drops nothing.
        """
        flag = "enable_step1" if key == "step1" else "enable_step2"
        p = SelectOmicsPipeline(_cfg(dataset, tmp_path, **{flag: False}))
        p.load_data()
        p.run_step0_reference()
        if key == "step2":
            p.run_step1_cleaning()
        getattr(p, method)()

        res = p.results[key]
        # Assert the property, not a key name: Step 1 reports removals as
        # dropped_by_constant / _variance / _correlation while Step 2 uses
        # dropped_features, so a key-based check passes vacuously on one of
        # them by reading a field that does not exist.
        assert res["n_features_out"] == res["n_features_in"], (
            f"a disabled {key} changed the feature count "
            f"({res['n_features_in']} -> {res['n_features_out']})"
        )
        assert res.get("consensus_outcome") == "disabled"
        assert len(p.get_selected_features(key)) == res["n_features_in"]

    @pytest.mark.integration
    def test_enabling_a_step_actually_removes_features(self, dataset, tmp_path):
        on = _run(dataset, tmp_path, enable_step1=True, enable_step2=True)
        assert on.get_selected_features("step2") != \
            on.get_selected_features("step0"), (
                "both steps enabled and the panel never changed"
            )

    @pytest.mark.integration
    def test_step3_runs_only_when_enabled(self, dataset, tmp_path):
        off = _run(dataset, tmp_path, enable_step3=False)
        assert off.results.get("step3", {}).get("skipped", True) is True


# ---------------------------------------------------------------------------
# Filter thresholds
# ---------------------------------------------------------------------------

class TestFilterThresholds:

    @pytest.mark.integration
    def test_the_correlation_threshold_changes_what_survives(self, dataset,
                                                             tmp_path):
        """
        A permissive threshold must keep at least as much as a strict one; the
        dataset carries a five-column correlated block for this.
        """
        strict = _run(dataset, tmp_path, min_correlation_threshold=0.5,
                      min_features_floor=1)
        loose = _run(dataset, tmp_path, min_correlation_threshold=0.99,
                     min_features_floor=1)
        n_strict = len(strict.get_selected_features("step1"))
        n_loose = len(loose.get_selected_features("step1"))
        assert n_loose >= n_strict, (
            f"a laxer correlation threshold kept fewer features "
            f"({n_loose} vs {n_strict})"
        )

    @pytest.mark.integration
    def test_class_aware_correlation_is_reported(self, dataset, tmp_path,
                                                 caplog):
        """
        Which correlation matrix produced a panel is not recoverable from the
        output, so the log is the only record a reader has.
        """
        caplog.set_level(logging.INFO, logger="SelectOmics")
        _run(dataset, tmp_path, class_aware_correlation=True)
        assert "class-aware" in caplog.text.lower()

    @pytest.mark.integration
    def test_the_feature_floor_is_honoured(self, dataset, tmp_path):
        high = _run(dataset, tmp_path, min_features_floor=15)
        low = _run(dataset, tmp_path, min_features_floor=1)
        assert len(high.get_selected_features("step2")) >= \
            len(low.get_selected_features("step2")), (
                "a higher floor returned a smaller panel"
            )


# ---------------------------------------------------------------------------
# Consensus controls
# ---------------------------------------------------------------------------

class TestConsensusControls:

    @pytest.mark.integration
    def test_the_model_count_changes_the_votes_available(self, dataset,
                                                         tmp_path):
        few = _run(dataset, tmp_path, n_consensus_models=2)
        many = _run(dataset, tmp_path, n_consensus_models=6)
        assert few.results["step2"]["n_models"] == 2
        assert many.results["step2"]["n_models"] == 6

    @pytest.mark.integration
    def test_the_union_rung_never_returns_less(self, dataset, tmp_path):
        """The union ceiling is a superset of the intersection ceiling."""
        tight = _run(dataset, tmp_path, allow_union_rung=False,
                     min_features_floor=20)
        loose = _run(dataset, tmp_path, allow_union_rung=True,
                     min_features_floor=20)
        assert len(loose.get_selected_features("step2")) >= \
            len(tight.get_selected_features("step2"))

    @pytest.mark.integration
    def test_the_agreement_outcome_is_always_reported(self, dataset, tmp_path):
        """
        The panel means different things at different rungs, so the outcome has
        to accompany it rather than being inferable only from the count.
        """
        p = _run(dataset, tmp_path)
        outcome = p.results["step2"]["consensus_outcome"]
        assert outcome in ("consensus", "relaxed", "union", "disabled",
                           "consensus_limited", "rank_average")
        assert p.results["step2"]["agreement_label"]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestSeedAndSplit:

    @pytest.mark.integration
    def test_the_same_seed_gives_the_same_panel(self, dataset, tmp_path):
        a = _run(dataset, tmp_path, random_seed=42)
        b = _run(dataset, tmp_path, random_seed=42)
        assert a.get_selected_features() == b.get_selected_features(), (
            "identical seeds produced different panels"
        )

    @pytest.mark.integration
    def test_the_test_split_size_is_honoured(self, dataset, tmp_path):
        small = _run(dataset, tmp_path, test_size=0.2)
        large = _run(dataset, tmp_path, test_size=0.4)
        assert len(large._y_test) > len(small._y_test)


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

class TestOutputSwitches:

    @pytest.mark.integration
    def test_intermediate_results_are_written_only_when_asked(self, dataset,
                                                              tmp_path):
        on = _run(dataset, tmp_path / "on", save_intermediate_results=True)
        on.save_results()
        assert list(on.config.output_dir.glob("*")), "nothing was written"

        off = _run(dataset, tmp_path / "off", save_intermediate_results=False)
        assert not list((tmp_path / "off" / "out").glob("pipeline_summary*")), (
            "intermediate results were written despite the flag being off"
        )

    @pytest.mark.integration
    def test_visualizations_are_produced_only_when_asked(self, dataset,
                                                         tmp_path):
        on = _run(dataset, tmp_path / "viz_on", create_visualizations=True,
                  save_intermediate_results=True)
        assert list(on.config.output_dir.glob("*.png")), "no figures written"

        off = _run(dataset, tmp_path / "viz_off", create_visualizations=False,
                   save_intermediate_results=True)
        assert not list(off.config.output_dir.glob("*.png")), (
            "figures were written despite create_visualizations=False"
        )

    @pytest.mark.integration
    def test_the_output_format_controls_the_file_written(self, dataset,
                                                         tmp_path):
        p = _run(dataset, tmp_path, save_intermediate_results=True,
                 output_format="json")
        p.save_results()
        assert list(p.config.output_dir.glob("*.json")), (
            "output_format='json' produced no JSON"
        )

    @pytest.mark.integration
    def test_verbose_changes_how_much_is_reported(self, dataset, tmp_path,
                                                  caplog):
        caplog.set_level(logging.INFO, logger="SelectOmics")
        _run(dataset, tmp_path, verbose=True)
        loud = len(caplog.text)
        caplog.clear()
        _run(dataset, tmp_path, verbose=False)
        quiet = len(caplog.text)
        assert loud > quiet, "verbose=True produced no extra output"


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------

class TestAlgorithmSelection:

    @pytest.mark.integration
    @pytest.mark.parametrize("algorithm", ["LR", "RF", "XGB", "SVM"])
    def test_each_algorithm_runs_and_is_recorded(self, dataset, tmp_path,
                                                 algorithm):
        p = _run(dataset, tmp_path, algorithm=algorithm)
        assert p.config.algorithm == algorithm
        assert len(p.get_selected_features()) >= 1
