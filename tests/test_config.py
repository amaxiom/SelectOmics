"""
Unit tests for SelectOmics/config.py.

Covers:
- Valid construction with required fields only
- Validation errors for every guarded field
- from_dict / to_dict roundtrip
- from_json / to_json roundtrip
- Unknown-key warning in from_dict
"""
import json
import warnings

import pytest

from SelectOmics.config import SelectOmicsConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal(**overrides) -> SelectOmicsConfig:
    defaults = dict(data_path="data.csv", target_column="Class")
    defaults.update(overrides)
    return SelectOmicsConfig(**defaults)


# ---------------------------------------------------------------------------
# Valid construction
# ---------------------------------------------------------------------------

def test_minimal_construction():
    cfg = _minimal()
    assert cfg.algorithm == "XGB"
    assert cfg.n_consensus_models == 10
    # 1.0 = keep the whole reachable pool. The field is a fraction of the
    # features the stage models actually used, not of the input width, so
    # 1.0 is a meaningful upper bound rather than a no-op.
    assert cfg.step2_target_retention == 1.0
    assert cfg.step3_target_retention == 0.6


def test_min_features_floor_default_is_fifteen():
    """
    Measured with everything else fixed, over six scenarios at five seeds, so
    only the floor varied. Composite is monotonic to 15 then flat:
    5 -> 0.716, 8 -> 0.784, 10 -> 0.808, 15 -> 0.820, 20 -> 0.819, 25 -> 0.820.

    15 rather than 20 or 25 because those pass enough features to wake Step 3,
    the most expensive step, costing 2 to 2.5x the runtime (157s vs 327s and
    389s) for no measurable gain.

    Changing this default should follow a re-measurement, not a hunch: earlier
    confounded readings put this parameter's importance anywhere from dominant
    to insignificant depending on the scenario mix.
    """
    assert _minimal().min_features_floor == 15


def test_path_coerced_to_path_object():
    from pathlib import Path
    cfg = _minimal()
    assert isinstance(cfg.data_path, Path)
    assert isinstance(cfg.output_dir, Path)


# ---------------------------------------------------------------------------
# algorithm
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("algo", ["LR", "XGB", "RF", "SVM"])
def test_valid_algorithms(algo):
    _minimal(algorithm=algo)


def test_invalid_algorithm():
    with pytest.raises(ValueError, match="algorithm"):
        _minimal(algorithm="GBM")


# ---------------------------------------------------------------------------
# test_size
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_invalid_test_size(bad):
    with pytest.raises(ValueError, match="test_size"):
        _minimal(test_size=bad)


# ---------------------------------------------------------------------------
# n_consensus_models
# ---------------------------------------------------------------------------

def test_invalid_n_consensus_models():
    with pytest.raises(ValueError, match="n_consensus_models"):
        _minimal(n_consensus_models=0)


# ---------------------------------------------------------------------------
# min_consensus (replaced consensus_threshold)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0.0, 1.1, -0.5])
def test_invalid_min_consensus(bad):
    with pytest.raises(ValueError, match="min_consensus"):
        _minimal(min_consensus=bad)


def test_min_consensus_defaults_to_point_four():
    """
    Not arbitrary. The response has a clear interior optimum, measured over
    two search phases: composite None 0.751 / 0.4 0.784 / 0.6 0.743 /
    0.8 0.599 / 1.0 0.503 in Phase 1, and in Phase 2 the 0.4 setting wins on
    every scenario individually. Changing this default should follow a
    re-measurement, not a hunch.
    """
    assert _minimal().min_consensus == 0.4


def test_min_consensus_none_is_still_expressible():
    """None disables the floor entirely; it is a valid setting, not absent."""
    assert _minimal(min_consensus=None).min_consensus is None


def test_min_consensus_accepts_valid_fractions():
    for good in (0.4, 0.6, 0.8, 1.0):
        assert _minimal(min_consensus=good).min_consensus == good


def test_consensus_threshold_field_is_gone():
    """
    It set where the graduated relaxation STARTED, and the relaxation stops
    at whichever level first satisfies the feature floor, so starting lower
    only skipped rungs on the way to the same answer. Measured across two
    scenarios and two seeds, thresholds of 0.4 to 1.0 returned the identical
    feature SET. Removed rather than deprecated.
    """
    assert "consensus_threshold" not in SelectOmicsConfig.__dataclass_fields__

    d = {"data_path": "x.csv", "target_column": "y",
         "consensus_threshold": 0.5}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SelectOmicsConfig.from_dict(d)
    assert any("consensus_threshold" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# stability_threshold
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0.0, 1.1])
def test_invalid_stability_threshold(bad):
    with pytest.raises(ValueError, match="stability_threshold"):
        _minimal(stability_threshold=bad)


# ---------------------------------------------------------------------------
# step3_target_retention
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("good", [0.1, 0.6, 0.9])
def test_valid_step3_target_retention(good):
    _minimal(step3_target_retention=good)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_invalid_step3_target_retention(bad):
    with pytest.raises(ValueError, match="step3_target_retention"):
        _minimal(step3_target_retention=bad)


# ---------------------------------------------------------------------------
# percentile search ranges were removed; retention targets replaced them
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field", [
    "l1_percentile_range",
    "l2_percentile_range",
    "rfecv_percentile_range",
])
def test_percentile_range_fields_are_gone(field):
    """
    These bounded a binary search over a quantity that did nothing: the
    config search's own sensitivity analysis ranked all four last, at
    Spearman r between -0.038 and 0.045 with p from 0.52 to 0.98. They are
    not deprecated-and-ignored, they are removed, so a config still setting
    one gets the unknown-key warning rather than silent no-op behaviour.
    """
    assert field not in SelectOmicsConfig.__dataclass_fields__

    d = {"data_path": "x.csv", "target_column": "y", field: (0.0, 0.5)}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SelectOmicsConfig.from_dict(d)
    assert any(field in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# format fields
# ---------------------------------------------------------------------------

def test_invalid_plot_format():
    with pytest.raises(ValueError, match="plot_format"):
        _minimal(plot_format="bmp")


def test_invalid_output_format():
    with pytest.raises(ValueError, match="output_format"):
        _minimal(output_format="xml")


# ---------------------------------------------------------------------------
# from_dict / to_dict roundtrip
# ---------------------------------------------------------------------------

def test_to_dict_from_dict_roundtrip():
    cfg = _minimal(algorithm="RF", n_consensus_models=3)
    d = cfg.to_dict()
    cfg2 = SelectOmicsConfig.from_dict(d)
    assert cfg2.algorithm == "RF"
    assert cfg2.n_consensus_models == 3


def test_retention_targets_survive_a_yaml_roundtrip():
    """
    The tuple fields that needed list/tuple coercion are gone; the three
    retention targets are plain floats and must round-trip unchanged.
    """
    cfg = _minimal(step2_target_retention=0.45,
                   step3_target_retention=0.35,
                   )
    cfg2 = SelectOmicsConfig.from_dict(cfg.to_dict())
    assert cfg2.step2_target_retention == pytest.approx(0.45)
    assert cfg2.step3_target_retention == pytest.approx(0.35)


def test_from_dict_unknown_keys_warns():
    d = {"data_path": "x.csv", "target_column": "y", "typo_key": 99}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SelectOmicsConfig.from_dict(d)
    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("typo_key" in m for m in messages)


def test_from_dict_unknown_keys_does_not_raise():
    d = {"data_path": "x.csv", "target_column": "y", "unknown": "value"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = SelectOmicsConfig.from_dict(d)
    assert cfg.target_column == "y"


# ---------------------------------------------------------------------------
# to_json / from_json roundtrip
# ---------------------------------------------------------------------------

def test_json_roundtrip(tmp_path):
    cfg = _minimal(algorithm="SVM")
    json_path = tmp_path / "config.json"
    cfg.to_json(json_path)

    cfg2 = SelectOmicsConfig.from_json(json_path)
    assert cfg2.algorithm == "SVM"


def test_json_is_valid_json(tmp_path):
    cfg = _minimal()
    json_path = tmp_path / "config.json"
    cfg.to_json(json_path)
    with open(json_path) as fh:
        d = json.load(fh)
    assert "algorithm" in d
    assert "step3_target_retention" in d


# ---------------------------------------------------------------------------
# Preset constructors (Phase 1B)
# ---------------------------------------------------------------------------

class TestPresets:
    """Tests for SelectOmicsConfig preset class methods."""

    def test_quick_preset_returns_config(self):
        cfg = SelectOmicsConfig.quick("data.csv", "Class")
        assert isinstance(cfg, SelectOmicsConfig)

    def test_quick_preset_steps_3_4_disabled(self):
        cfg = SelectOmicsConfig.quick("data.csv", "Class")
        assert cfg.enable_step3 is False

    def test_quick_preset_fewer_models(self):
        cfg = SelectOmicsConfig.quick("data.csv", "Class")
        assert cfg.n_consensus_models == 5

    def test_quick_preset_kwargs_override(self):
        cfg = SelectOmicsConfig.quick("data.csv", "Class", algorithm="RF")
        assert cfg.algorithm == "RF"

    def test_standard_preset_returns_config(self):
        cfg = SelectOmicsConfig.standard("data.csv", "Class")
        assert isinstance(cfg, SelectOmicsConfig)

    def test_standard_preset_all_steps_enabled(self):
        cfg = SelectOmicsConfig.standard("data.csv", "Class")
        assert cfg.enable_step3 is True

    def test_standard_preset_ten_models(self):
        cfg = SelectOmicsConfig.standard("data.csv", "Class")
        assert cfg.n_consensus_models == 10

    def test_thorough_preset_returns_config(self):
        cfg = SelectOmicsConfig.thorough("data.csv", "Class")
        assert isinstance(cfg, SelectOmicsConfig)

    def test_thorough_preset_more_models(self):
        cfg = SelectOmicsConfig.thorough("data.csv", "Class")
        assert cfg.n_consensus_models == 20

    def test_thorough_preset_kwargs_override(self):
        cfg = SelectOmicsConfig.thorough("data.csv", "Class", n_consensus_models=5)
        assert cfg.n_consensus_models == 5

    def test_omics_preset_returns_config(self):
        cfg = SelectOmicsConfig.omics("data.csv", "Class")
        assert isinstance(cfg, SelectOmicsConfig)

    def test_omics_preset_all_steps_enabled(self):
        cfg = SelectOmicsConfig.omics("data.csv", "Class")
        assert cfg.enable_step3 is True

    def test_omics_preset_tuned_thresholds(self):
        """
        Omics preset carries the surviving benchmark-tuned values (h_071).
        Its consensus_threshold of 0.442 is deliberately NOT translated into
        min_consensus: one was a starting point for the relaxation and the
        other is a floor on it, so no value of one means the same as a value
        of the other.
        """
        cfg = SelectOmicsConfig.omics("data.csv", "Class")
        # n_consensus_models is no longer overridden: the old search pinned it
        # at 5, below the default of 10, making this preset weaker than the
        # plain defaults on the parameter the re-run search ranks highest.
        assert cfg.n_consensus_models == 10
        assert cfg.stability_threshold == 0.259
        # Not overridden: the preset inherits the measured default, and the
        # search that produced 0.4 targeted exactly this regime.
        assert cfg.min_consensus == 0.4

    def test_omics_preset_kwargs_override(self):
        cfg = SelectOmicsConfig.omics("data.csv", "Class", min_consensus=0.8)
        assert cfg.min_consensus == 0.8

    def test_presets_build_without_removed_fields(self):
        """
        Every preset must construct cleanly now the percentile ranges are
        gone -- from_dict warns on unknown keys, so a stale preset would be
        noisy rather than merely wrong.
        """
        for factory in (SelectOmicsConfig.quick, SelectOmicsConfig.standard,
                        SelectOmicsConfig.thorough, SelectOmicsConfig.omics):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                cfg = factory("data.csv", "Class")
            assert not [w for w in caught if "unrecognised" in str(w.message)], (
                f"{factory.__name__} passes a field the dataclass no longer has"
            )
            assert 0.0 < cfg.step2_target_retention <= 1.0

    def test_preset_schema_version(self):
        """Presets produced by from_dict must carry schema_version."""
        cfg = SelectOmicsConfig.quick("data.csv", "Class")
        assert cfg.schema_version == "1.0"


# ---------------------------------------------------------------------------
# schema_version (Phase 1D)
# ---------------------------------------------------------------------------

class TestSchemaVersion:
    """Tests for schema_version field and migration."""

    def test_default_schema_version(self):
        cfg = _minimal()
        assert cfg.schema_version == "1.0"

    def test_schema_version_in_to_dict(self):
        cfg = _minimal()
        d = cfg.to_dict()
        assert "schema_version" in d
        assert d["schema_version"] == "1.0"

    def test_schema_version_round_trips_json(self, tmp_path):
        cfg = _minimal()
        path = tmp_path / "cfg.json"
        cfg.to_json(path)
        cfg2 = SelectOmicsConfig.from_json(path)
        assert cfg2.schema_version == "1.0"

    def test_unknown_schema_version_warns(self):
        d = {"data_path": "x.csv", "target_column": "y", "schema_version": "99.0"}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            SelectOmicsConfig.from_dict(d)
        messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        assert any("schema_version" in m for m in messages)


# ---------------------------------------------------------------------------
# diff_from_defaults and __repr__ (Phase 3B / 3C)
# ---------------------------------------------------------------------------

class TestDiffAndRepr:
    """Tests for diff_from_defaults() and __repr__."""

    def test_diff_empty_for_defaults(self):
        cfg = _minimal()
        diff = cfg.diff_from_defaults()
        # data_path, target_column, schema_version excluded; rest are defaults.
        assert diff == {}

    def test_diff_detects_changed_algorithm(self):
        cfg = _minimal(algorithm="RF")
        diff = cfg.diff_from_defaults()
        assert "algorithm" in diff
        assert diff["algorithm"] == "RF"

    def test_diff_excludes_required_fields(self):
        cfg = _minimal(algorithm="SVM")
        diff = cfg.diff_from_defaults()
        assert "data_path" not in diff
        assert "target_column" not in diff

    def test_repr_contains_data_path(self):
        cfg = _minimal()
        r = repr(cfg)
        assert "data_path" in r

    def test_repr_shows_non_default(self):
        cfg = _minimal(algorithm="LR")
        r = repr(cfg)
        assert "algorithm='LR'" in r

    def test_repr_is_str(self):
        cfg = _minimal()
        assert isinstance(repr(cfg), str)


# ---------------------------------------------------------------------------
# New features: suggest(), template_yaml(), CLI helpers
# ---------------------------------------------------------------------------

class TestConfigDiscovery:
    """Tests for suggest(), template_yaml(), and CLI helpers."""

    def test_template_yaml_returns_string(self):
        template = SelectOmicsConfig.template_yaml()
        assert isinstance(template, str)
        assert 'data_path' in template
        assert 'target_column' in template
        assert 'algorithm' in template
        assert 'n_consensus_models' in template

    def test_template_yaml_writes_file(self, tmp_path):
        out = tmp_path / 'template.yaml'
        SelectOmicsConfig.template_yaml(path=out)
        assert out.exists()
        content = out.read_text()
        assert 'schema_version' in content

    def test_template_yaml_parseable(self, tmp_path):
        """Template with required fields filled should load as a valid config."""
        pytest.importorskip('yaml')
        import yaml
        template = SelectOmicsConfig.template_yaml()
        d = yaml.safe_load(template)
        # Fill required path fields with valid temp values.
        d['data_path'] = str(tmp_path / 'data.csv')
        d['target_column'] = 'Class'
        # cv_splits: null loads as None -- ensure from_dict handles it.
        cfg = SelectOmicsConfig.from_dict(d)
        assert cfg.algorithm == 'XGB'

    def test_suggest_small_dataset(self, binary_dataset):
        """suggest() should return a valid config for a real dataset."""
        import SelectOmics
        SelectOmics.enable_logging('WARNING')  # suppress INFO during tests
        cfg = SelectOmicsConfig.suggest(binary_dataset, 'Class')
        assert isinstance(cfg, SelectOmicsConfig)
        assert cfg.algorithm in ('XGB', 'RF', 'LR', 'SVM')
        assert cfg.n_consensus_models >= 1

    def test_suggest_with_override(self, binary_dataset):
        """suggest() should respect caller overrides."""
        import SelectOmics
        SelectOmics.enable_logging('WARNING')
        cfg = SelectOmicsConfig.suggest(binary_dataset, 'Class', algorithm='RF')
        assert cfg.algorithm == 'RF'

    def test_suggest_invalid_target(self, binary_dataset):
        """suggest() with a missing target column returns a fallback config and warns."""
        import SelectOmics
        SelectOmics.enable_logging('WARNING')
        # suggest() catches data-loading errors internally and returns a fallback config.
        cfg = SelectOmicsConfig.suggest(binary_dataset, 'NonexistentColumn')
        # The fallback config should still be a valid SelectOmicsConfig instance.
        assert isinstance(cfg, SelectOmicsConfig)
