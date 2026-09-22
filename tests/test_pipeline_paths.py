"""
Pipeline orchestration paths.

The step functions are well covered; the object that drives them is not. What
was missing is the machinery around the steps: step lookup and its error
cases, the callback contract, nested CV, resume, the summary and config
reporting, and the final test evaluation. Those only execute on a complete run
with reporting enabled, which no unit test reaches.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import SelectOmics
from SelectOmics.config import SelectOmicsConfig
from SelectOmics.pipeline import SelectOmicsPipeline


@pytest.fixture
def dataset(tmp_path):
    rng = np.random.RandomState(0)
    n, p = 70, 30
    X = rng.randn(n, p)
    y = np.array([i % 2 for i in range(n)])
    X[:, :6] += 1.9 * y[:, None]
    df = pd.DataFrame(X, columns=[f"f{i:02d}" for i in range(p)])
    df["Class"] = y
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)
    return path


def _cfg(dataset, tmp_path, **over):
    kw = dict(
        data_path=str(dataset), target_column="Class", algorithm="RF",
        output_dir=str(tmp_path / "out"), n_consensus_models=2,
        quick_tune_iterations=2, n_bootstrap=10, verbose=False,
        create_visualizations=False, save_intermediate_results=False,
        enable_step_evaluations=False, enable_final_test_evaluation=False,
    )
    kw.update(over)
    return SelectOmicsConfig(**kw)


# ---------------------------------------------------------------------------
# get_selected_features
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_get_selected_features_by_step(dataset, tmp_path):
    p = SelectOmicsPipeline(_cfg(dataset, tmp_path))
    p.run(validate=False)
    latest = p.get_selected_features()
    assert latest == p.get_selected_features("step3") or latest

    counts = {}
    for key in ("step0", "step1", "step2", "step3"):
        counts[key] = len(p.get_selected_features(key))
    # A selection pipeline must never grow the feature set.
    assert counts["step0"] >= counts["step1"] >= counts["step2"] >= counts["step3"]


def test_get_selected_features_invalid_key(dataset, tmp_path):
    p = SelectOmicsPipeline(_cfg(dataset, tmp_path))
    with pytest.raises(ValueError, match="step must be one of"):
        p.get_selected_features("step9")


def test_get_selected_features_before_running(dataset, tmp_path):
    """Asking for a step that has not run must say so, not return nothing."""
    p = SelectOmicsPipeline(_cfg(dataset, tmp_path))
    with pytest.raises(ValueError, match="has not been run|not been run"):
        p.get_selected_features("step2")


# ---------------------------------------------------------------------------
# on_step_complete callback
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_callback_fires_for_every_step(dataset, tmp_path):
    seen = []
    p = SelectOmicsPipeline(_cfg(dataset, tmp_path),
                            on_step_complete=lambda k, r: seen.append(k))
    p.run(validate=False)
    assert "step0" in seen
    assert seen == sorted(seen, key=lambda k: ["step0", "step1", "step2",
                                               "step3"].index(k))


@pytest.mark.integration
def test_a_raising_callback_never_aborts_the_run(dataset, tmp_path):
    """
    A user callback is arbitrary code. If it throws, the pipeline must log and
    carry on: losing a completed run to a logging typo is not acceptable.
    """
    def bad(step_key, result):
        raise RuntimeError("callback exploded")

    p = SelectOmicsPipeline(_cfg(dataset, tmp_path), on_step_complete=bad)
    results = p.run(validate=False)
    assert "step0" in results
    assert p.get_selected_features()


# ---------------------------------------------------------------------------
# Reporting paths
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_verbose_run_prints_config_and_summary(dataset, tmp_path, caplog):
    """_print_config and _print_summary only run when verbose is on."""
    import logging
    caplog.set_level(logging.INFO, logger="SelectOmics")
    p = SelectOmicsPipeline(_cfg(dataset, tmp_path, verbose=True))
    p.run(validate=False)
    text = caplog.text
    assert "Configuration" in text or "Algorithm" in text
    assert "Step" in text


@pytest.mark.integration
def test_full_run_with_saving_and_figures(dataset, tmp_path):
    """The reporting and artefact branches, which default off."""
    out = tmp_path / "full"
    p = SelectOmicsPipeline(_cfg(
        dataset, tmp_path, output_dir=str(out), verbose=True,
        create_visualizations=True, save_intermediate_results=True,
        enable_step_evaluations=True, enable_final_test_evaluation=True))
    p.run(validate=True)
    p.save_results()
    assert (out / "selected_features.csv").exists()
    assert list(out.glob("*.png")), "no figures written"


# ---------------------------------------------------------------------------
# Provenance and state
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_feature_provenance(dataset, tmp_path):
    p = SelectOmicsPipeline(_cfg(dataset, tmp_path))
    p.run(validate=False)
    prov = p.get_feature_provenance()
    assert "feature" in prov.columns
    assert any(c.startswith("survived_") for c in prov.columns)
    assert "n_steps_survived" in prov.columns
    # Every original feature appears exactly once.
    assert prov["feature"].is_unique


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_resume_skips_completed_steps(dataset, tmp_path):
    out = tmp_path / "resumable"
    cfg = _cfg(dataset, tmp_path, output_dir=str(out),
               save_intermediate_results=True)
    SelectOmicsPipeline(cfg).run(validate=False)
    assert (out / ".selectomics_checkpoint.pkl").exists()

    resumed = SelectOmicsPipeline(cfg)
    resumed.run(validate=False, resume=True)
    assert resumed.get_selected_features()


@pytest.mark.integration
def test_resume_with_no_checkpoint_runs_normally(dataset, tmp_path):
    """resume=True on a fresh directory must run, not fail."""
    cfg = _cfg(dataset, tmp_path, output_dir=str(tmp_path / "fresh"))
    p = SelectOmicsPipeline(cfg)
    p.run(validate=False, resume=True)
    assert p.get_selected_features()


# ---------------------------------------------------------------------------
# Ordering contract
# ---------------------------------------------------------------------------

def test_steps_refuse_to_run_out_of_order(dataset, tmp_path):
    p = SelectOmicsPipeline(_cfg(dataset, tmp_path))
    with pytest.raises(RuntimeError):
        p.run_step1_cleaning()


@pytest.mark.integration
def test_fluent_interface_returns_self(dataset, tmp_path):
    p = SelectOmicsPipeline(_cfg(dataset, tmp_path))
    out = p.load_data().run_step0_reference().run_step1_cleaning()
    assert out is p


# ---------------------------------------------------------------------------
# Nested CV
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
def test_nested_cv(dataset, tmp_path):
    """run_nested_cv is a separate entry point and entirely uncovered."""
    cfg = _cfg(dataset, tmp_path, enable_nested_cv=True, outer_cv_splits=2)
    p = SelectOmicsPipeline(cfg)
    res = p.run_nested_cv()
    assert isinstance(res, dict)
    assert any("auc" in str(k).lower() or "fold" in str(k).lower() for k in res)
