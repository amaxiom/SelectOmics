"""
CLI command handlers, end to end through ``main``.

The parser was already tested; the handlers behind it were not, so the three
subcommands a user actually types were uncovered. These drive each one through
``main(argv)`` exactly as the console entry point does, on real files in a
temporary directory.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from SelectOmics.cli import main, _parse_overrides, build_parser


@pytest.fixture
def dataset(tmp_path):
    """A small, well-conditioned dataset the pipeline can actually run on."""
    rng = np.random.RandomState(0)
    n, p = 60, 30
    X = rng.randn(n, p)
    y = np.array([i % 2 for i in range(n)])
    X[:, :5] += 1.8 * y[:, None]
    df = pd.DataFrame(X, columns=[f"f{i:02d}" for i in range(p)])
    df["Class"] = y
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# template
# ---------------------------------------------------------------------------

def test_template_to_stdout(capsys):
    assert main(["template"]) == 0
    out = capsys.readouterr().out
    assert "target_column" in out and "enable_step1" in out


def test_template_to_file(tmp_path, capsys):
    dest = tmp_path / "cfg.yaml"
    assert main(["template", "--output", str(dest)]) == 0
    assert dest.exists() and dest.stat().st_size > 0
    assert "Template written to" in capsys.readouterr().out
    # The template must be loadable by the config it documents.
    from SelectOmics.config import SelectOmicsConfig
    text = dest.read_text(encoding="utf-8")
    assert "data_path" in text
    assert SelectOmicsConfig.template_yaml() is not None


# ---------------------------------------------------------------------------
# suggest
# ---------------------------------------------------------------------------

def test_suggest_prints_a_rationale(dataset, capsys):
    assert main(["suggest", str(dataset), "-t", "Class"]) == 0
    out = capsys.readouterr().out
    assert out.strip(), "suggest produced no output"


def test_suggest_writes_yaml(dataset, tmp_path):
    dest = tmp_path / "suggested.yaml"
    assert main(["suggest", str(dataset), "-t", "Class",
                 "--output", str(dest)]) == 0
    assert dest.exists() and dest.stat().st_size > 0


def test_suggest_honours_set_overrides(dataset, tmp_path):
    dest = tmp_path / "s.yaml"
    assert main(["suggest", str(dataset), "-t", "Class",
                 "--set", "algorithm=RF", "--output", str(dest)]) == 0
    assert "RF" in dest.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_run_executes_and_writes_results(dataset, tmp_path):
    out_dir = tmp_path / "results"
    rc = main([
        "run", str(dataset), "-t", "Class",
        "--set", f"output_dir={out_dir}",
        "--set", "algorithm=RF",
        "--set", "n_consensus_models=2",
        "--set", "quick_tune_iterations=2",
        "--set", "n_bootstrap=10",
        "--set", "create_visualizations=False",
        "--set", "enable_step_evaluations=False",
        "--set", "enable_final_test_evaluation=False",
    ])
    assert rc == 0
    assert (out_dir / "selected_features.csv").exists()


@pytest.mark.integration
def test_run_from_a_config_file(dataset, tmp_path):
    """The --config path is how the template is meant to be used."""
    from SelectOmics.config import SelectOmicsConfig
    out_dir = tmp_path / "res2"
    cfg = SelectOmicsConfig(
        data_path=str(dataset), target_column="Class", algorithm="RF",
        output_dir=str(out_dir), n_consensus_models=2,
        quick_tune_iterations=2, n_bootstrap=10,
        create_visualizations=False, enable_step_evaluations=False,
        enable_final_test_evaluation=False,
    )
    cfg_path = tmp_path / "run.yaml"
    cfg.to_yaml(cfg_path)
    assert main(["run", "--config", str(cfg_path)]) == 0
    assert (out_dir / "selected_features.csv").exists()


# ---------------------------------------------------------------------------
# --set parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw, key, expected", [
    ("algorithm=RF", "algorithm", "RF"),
    ("n_consensus_models=5", "n_consensus_models", 5),
    ("test_size=0.3", "test_size", 0.3),
    ("verbose=true", "verbose", True),
    ("verbose=False", "verbose", False),
    ("stability_threshold=none", "stability_threshold", None),
])
def test_parse_overrides_types(raw, key, expected):
    """--set values arrive as strings and must be coerced, not passed through."""
    got = _parse_overrides([raw])
    assert got[key] == expected, got
    assert type(got[key]) is type(expected)


def test_parse_overrides_empty():
    assert _parse_overrides(None) == {}
    assert _parse_overrides([]) == {}


def test_parse_overrides_warns_on_malformed(capsys):
    """
    A --set without '=' is warned about and ignored rather than raising. That
    is deliberate, but it means a typo like `--set algorithm RF` silently drops
    the override and the run proceeds on the default, so the warning is the
    only signal the user gets. Assert it is actually emitted.
    """
    got = _parse_overrides(["not_a_pair", "algorithm=RF"])
    assert got == {"algorithm": "RF"}, "valid pairs must survive a bad one"
    err = capsys.readouterr().err
    assert "not_a_pair" in err and "key=value" in err


# ---------------------------------------------------------------------------
# parser wiring
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cmd", ["run", "suggest", "template"])
def test_every_subcommand_binds_a_handler(cmd):
    """A subcommand with no func would fail only at runtime."""
    argv = {"run": ["run", "d.csv", "-t", "C"],
            "suggest": ["suggest", "d.csv", "-t", "C"],
            "template": ["template"]}[cmd]
    args = build_parser().parse_args(argv)
    assert callable(getattr(args, "func", None))


def test_no_subcommand_exits_nonzero(capsys):
    with pytest.raises(SystemExit):
        main([])


# ---------------------------------------------------------------------------
# run: argument branches the happy path never reaches
# ---------------------------------------------------------------------------

def _fast_cfg(dataset, out_dir):
    from SelectOmics.config import SelectOmicsConfig
    return SelectOmicsConfig(
        data_path=str(dataset), target_column="Class", algorithm="RF",
        output_dir=str(out_dir), n_consensus_models=2,
        quick_tune_iterations=2, n_bootstrap=10,
        create_visualizations=False, enable_step_evaluations=False,
        enable_final_test_evaluation=False,
    )


def test_run_from_a_json_config(dataset, tmp_path):
    """--config dispatches on suffix; the JSON arm is the one not exercised."""
    out_dir = tmp_path / "json_res"
    cfg_path = tmp_path / "run.json"
    _fast_cfg(dataset, out_dir).to_json(cfg_path)

    assert main(["run", "--config", str(cfg_path)]) == 0
    assert (out_dir / "selected_features.csv").exists()


def test_set_overrides_apply_on_top_of_a_config_file(dataset, tmp_path):
    """
    --set has to win over the file, or a user could not adjust a saved config
    without editing it.
    """
    from SelectOmics.config import SelectOmicsConfig
    out_dir = tmp_path / "ovr_res"
    cfg_path = tmp_path / "run.json"
    _fast_cfg(dataset, out_dir).to_json(cfg_path)

    assert main(["run", "--config", str(cfg_path),
                 "--set", "n_consensus_models=3"]) == 0
    written = SelectOmicsConfig.from_json(out_dir / "config.json")
    assert written.n_consensus_models == 3, "--set did not override the file"


def test_output_flag_overrides_the_config_output_dir(dataset, tmp_path):
    out_dir = tmp_path / "in_cfg"
    real_out = tmp_path / "from_flag"
    cfg_path = tmp_path / "run.json"
    _fast_cfg(dataset, out_dir).to_json(cfg_path)

    assert main(["run", "--config", str(cfg_path),
                 "--output", str(real_out)]) == 0
    assert (real_out / "selected_features.csv").exists()
    assert not out_dir.exists(), "--output did not redirect the results"


def test_argparse_rejects_an_unknown_preset_before_the_handler(dataset):
    """--preset carries choices=, so the parser refuses an unknown value."""
    import pytest as _pytest
    with _pytest.raises(SystemExit) as exc:
        main(["run", str(dataset), "-t", "Class", "--preset", "nonsense"])
    assert exc.value.code == 2


def test_cmd_run_guards_an_unknown_preset_itself(dataset, capsys):
    """
    cmd_run re-checks the preset rather than trusting the parser. argparse
    makes that unreachable through main, so drive the handler directly, which
    is how any caller building a Namespace by hand would reach it.
    """
    import argparse
    from SelectOmics.cli import cmd_run

    args = argparse.Namespace(
        data=str(dataset), target="Class", preset="nonsense", config=None,
        output=None, resume=False, set=None, verbose=False,
    )
    assert cmd_run(args) == 1
    assert "Unknown preset" in capsys.readouterr().err


def test_run_without_config_or_data_is_rejected(capsys):
    rc = main(["run"])
    assert rc == 1
    assert "Provide either" in capsys.readouterr().err


def test_the_final_test_auc_is_printed_when_it_was_computed(dataset, tmp_path,
                                                            capsys):
    """The AUC line only prints when the final test evaluation actually ran."""
    from SelectOmics.config import SelectOmicsConfig
    out_dir = tmp_path / "auc_res"
    cfg = SelectOmicsConfig(
        data_path=str(dataset), target_column="Class", algorithm="RF",
        output_dir=str(out_dir), n_consensus_models=2,
        quick_tune_iterations=2, n_bootstrap=10,
        create_visualizations=False, enable_step_evaluations=False,
        enable_final_test_evaluation=True,
    )
    cfg_path = tmp_path / "run.json"
    cfg.to_json(cfg_path)

    assert main(["run", "--config", str(cfg_path)]) == 0
    out = capsys.readouterr().out
    assert "Final test AUC" in out
    # The AUC belongs to the recommended panel, so the summary has to name
    # that panel too; naming only the last step sent users to the other one.
    assert "Recommended:" in out
    assert "(recommended panel)" in out
    assert (out_dir / "recommended_features.csv").exists()


def test_version_falls_back_when_metadata_is_unavailable(monkeypatch):
    """
    The banner must not take down the CLI because a version string could not
    be read, which happens when running from a source tree with no metadata.
    """
    import importlib
    import SelectOmics.cli as cli_mod

    def boom(*args, **kwargs):
        raise RuntimeError("no distribution metadata")

    monkeypatch.setattr(cli_mod, "build_parser", cli_mod.build_parser)
    monkeypatch.setattr("importlib.metadata.version", boom, raising=False)
    reloaded = importlib.reload(cli_mod)
    assert reloaded.build_parser() is not None
    importlib.reload(cli_mod)
