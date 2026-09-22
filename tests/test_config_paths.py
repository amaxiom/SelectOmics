"""
Config validation guards, presets, and the suggest() decision tree.

Every field validated in ``__post_init__`` exists because an invalid value
would otherwise fail somewhere far downstream, mid-run, with an unrelated
message. A guard that is never tested is a guard nobody knows still fires, so
these assert each one rejects what it claims to.

``suggest()`` is a decision tree over dataset shape whose branches were almost
entirely uncovered: it picks consensus counts, tuning budgets and step toggles
from n, p and class balance, and a user who follows its advice deserves to know
it was exercised.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from SelectOmics.config import SelectOmicsConfig


def _cfg(**over):
    kw = dict(data_path="unused.csv", target_column="Class")
    kw.update(over)
    return SelectOmicsConfig(**kw)


# ---------------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field, bad", [
    ("input_format", "invented"),
    ("output_format", "invented"),
    ("plot_format", "invented"),
    ("excel_output_engine", "invented"),
    ("algorithm", "invented"),
])
def test_enum_fields_reject_unknown_values(field, bad):
    with pytest.raises(ValueError, match=field):
        _cfg(**{field: bad})


@pytest.mark.parametrize("field, bad", [
    ("cv_splits", 1),
    ("min_cv_splits", 1),
    ("max_cv_splits", 1),
    ("quick_tune_iterations", 0),
    ("min_features_floor", 0),
    ("n_bootstrap", 5),
    ("outer_cv_splits", 1),
    ("n_consensus_models", 0),
])
def test_integer_bounds_are_enforced(field, bad):
    with pytest.raises(ValueError, match=field):
        _cfg(**{field: bad})


@pytest.mark.parametrize("field, bad", [
    ("max_variance_threshold", 0.0),
    ("max_variance_threshold", -0.1),
    ("test_size", 0.0),
    ("test_size", 1.0),
    ("min_correlation_threshold", 0.0),
    ("min_correlation_threshold", 1.0),
    ("stability_threshold", 0.0),
    ("stability_threshold", 1.5),
    ("step2_target_retention", 0.0),
    ("step3_target_retention", 0.0),
    ("step2_stage_colsample", 0.0),
    ("min_consensus", 0.0),
    ("min_consensus", 1.5),
])
def test_fraction_bounds_are_enforced(field, bad):
    with pytest.raises(ValueError, match=field):
        _cfg(**{field: bad})


def test_allow_union_rung_must_be_bool():
    with pytest.raises(ValueError, match="allow_union_rung"):
        _cfg(allow_union_rung="yes")


def test_min_cv_splits_must_not_exceed_max():
    with pytest.raises(ValueError):
        _cfg(min_cv_splits=10, max_cv_splits=3)


def test_valid_edge_values_are_accepted():
    """The bounds are inclusive where the docs say they are."""
    c = _cfg(min_consensus=1.0, step2_target_retention=1.0,
             step2_stage_colsample=1.0, stability_threshold=1.0,
             n_bootstrap=10, min_features_floor=1, cv_splits=2)
    assert c.min_consensus == 1.0
    assert c.n_bootstrap == 10


def test_min_consensus_accepts_none():
    """None means 'relax as far as the feature floor requires'."""
    assert _cfg(min_consensus=None).min_consensus is None


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("preset", ["quick", "standard", "thorough", "omics"])
def test_presets_build_and_validate(preset):
    cfg = getattr(SelectOmicsConfig, preset)("d.csv", "Class")
    assert cfg.data_path is not None
    assert cfg.target_column == "Class"
    assert cfg.n_consensus_models >= 1


@pytest.mark.parametrize("preset", ["quick", "standard", "thorough", "omics"])
def test_presets_accept_overrides(preset):
    cfg = getattr(SelectOmicsConfig, preset)("d.csv", "Class",
                                             algorithm="RF", verbose=True)
    assert cfg.algorithm == "RF" and cfg.verbose is True


def test_quick_preset_disables_step3():
    """quick is documented as steps 1 and 2 only."""
    assert SelectOmicsConfig.quick("d.csv", "Class").enable_step3 is False


# ---------------------------------------------------------------------------
# suggest(): the decision tree over dataset shape
# ---------------------------------------------------------------------------

def _write(tmp_path, n, p, imbalance=0.5, name="d.csv"):
    rng = np.random.RandomState(0)
    n_pos = max(2, int(n * imbalance))
    y = np.array([1] * n_pos + [0] * (n - n_pos))
    rng.shuffle(y)
    df = pd.DataFrame(rng.randn(n, p), columns=[f"f{i}" for i in range(p)])
    df["Class"] = y
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


@pytest.mark.parametrize("n, p", [
    (25, 40),     # tiny n: 1 consensus model, step 3 off
    (60, 200),    # small
    (150, 800),   # moderate
    (600, 3000),  # large n
])
def test_suggest_covers_the_sample_size_branches(tmp_path, n, p):
    cfg = SelectOmicsConfig.suggest(str(_write(tmp_path, n, p)), "Class")
    assert cfg.n_consensus_models >= 1
    assert cfg.n_bootstrap >= 10
    if n < 50:
        assert cfg.enable_step3 is False, "step 3 should be off at tiny n"


def test_suggest_disables_step3_on_narrow_data(tmp_path):
    """Below the RFECV gate there is nothing for step 3 to eliminate."""
    cfg = SelectOmicsConfig.suggest(str(_write(tmp_path, 120, 12)), "Class")
    assert cfg.enable_step3 is False


def test_suggest_logs_its_rationale(tmp_path, caplog):
    """
    suggest() emits its reasoning through a logger, not print, so a caller
    that has not configured logging sees nothing. The CLI wires that up; a
    library caller must opt in.
    """
    import logging
    # Scoped to the package logger, matching the rest of the suite. An
    # unscoped set_level only raises the root logger, which is not enough
    # when the package logger itself is filtering at a higher level.
    caplog.set_level(logging.INFO, logger="SelectOmics")
    SelectOmicsConfig.suggest(str(_write(tmp_path, 200, 500, imbalance=0.08)),
                              "Class")
    assert "Config Suggestion" in caplog.text
    assert "Suggested config fields" in caplog.text


def test_suggest_accepts_overrides(tmp_path):
    cfg = SelectOmicsConfig.suggest(str(_write(tmp_path, 100, 300)), "Class",
                                    algorithm="RF")
    assert cfg.algorithm == "RF"


# ---------------------------------------------------------------------------
# Round-tripping
# ---------------------------------------------------------------------------

def test_yaml_round_trip(tmp_path):
    cfg = _cfg(algorithm="RF", n_consensus_models=7, min_consensus=None)
    path = tmp_path / "c.yaml"
    cfg.to_yaml(path)
    back = SelectOmicsConfig.from_yaml(path)
    assert back.algorithm == "RF"
    assert back.n_consensus_models == 7
    assert back.min_consensus is None


def test_dict_round_trip():
    cfg = _cfg(algorithm="SVM", test_size=0.3)
    back = SelectOmicsConfig.from_dict(cfg.to_dict())
    assert back.algorithm == "SVM"
    assert back.test_size == pytest.approx(0.3)


def test_unknown_keys_warn_but_load():
    """A config written by an older version must not become unloadable."""
    with pytest.warns(UserWarning):
        cfg = SelectOmicsConfig.from_dict({
            "data_path": "d.csv", "target_column": "Class",
            "enable_step4": True, "consensus_threshold": 0.5,
        })
    assert not hasattr(cfg, "enable_step4")


def test_template_yaml_is_valid_yaml():
    import yaml
    text = SelectOmicsConfig.template_yaml()
    parsed = yaml.safe_load(text)
    assert isinstance(parsed, dict)
    assert "target_column" in parsed
