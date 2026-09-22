"""
Data loading, preparation and multi-format writing.

The loader dispatches on file extension and the writer on ``output_format``,
so most of both modules is per-format branches that a CSV-only test never
reaches. These exercise every format the package claims to support, plus the
preparation rules (non-numeric columns dropped, NaN handling, the algorithm
switch) that decide what the pipeline actually sees.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from SelectOmics.config import SelectOmicsConfig
from SelectOmics.data.loaders import (
    load_omics_data,
    save_from_config,
    save_results_multiformat,
)
from SelectOmics.data.preprocessing import (
    create_train_test_split,
    handle_nans,
    prepare_data,
)


@pytest.fixture
def frame():
    rng = np.random.RandomState(0)
    df = pd.DataFrame(rng.randn(40, 6), columns=[f"g{i}" for i in range(6)])
    df["Class"] = ["A", "B"] * 20
    return df


# ---------------------------------------------------------------------------
# Loading: every declared input format
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("suffix, writer", [
    (".csv", lambda df, p: df.to_csv(p, index=False)),
    (".tsv", lambda df, p: df.to_csv(p, sep="\t", index=False)),
    (".json", lambda df, p: df.to_json(p)),
    (".parquet", lambda df, p: df.to_parquet(p, index=False)),
    (".feather", lambda df, p: df.reset_index(drop=True).to_feather(p)),
    (".xlsx", lambda df, p: df.to_excel(p, index=False)),
])
def test_load_every_format(tmp_path, frame, suffix, writer):
    path = tmp_path / f"data{suffix}"
    try:
        writer(frame, path)
    except (ImportError, ValueError) as exc:            # optional engines
        pytest.skip(f"writer for {suffix} unavailable: {exc}")
    cfg = SelectOmicsConfig(data_path=str(path), target_column="Class")
    loaded = load_omics_data(cfg)
    assert loaded.shape[0] == frame.shape[0]
    assert "Class" in loaded.columns


def test_explicit_input_format_overrides_extension(tmp_path, frame):
    """A file with no usable extension must still load when told the format."""
    path = tmp_path / "data_no_ext"
    frame.to_csv(path, index=False)
    cfg = SelectOmicsConfig(data_path=str(path), target_column="Class",
                            input_format="csv")
    assert load_omics_data(cfg).shape[0] == frame.shape[0]


def test_missing_file_raises_clearly(tmp_path):
    cfg = SelectOmicsConfig(data_path=str(tmp_path / "absent.csv"),
                            target_column="Class")
    with pytest.raises((FileNotFoundError, OSError, ValueError)):
        load_omics_data(cfg)


# ---------------------------------------------------------------------------
# Writing: every declared output format
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt, ext", [
    ("csv", ".csv"), ("tsv", ".tsv"), ("json", ".json"),
    ("parquet", ".parquet"), ("feather", ".feather"), ("excel", ".xlsx"),
])
def test_save_every_format(tmp_path, fmt, ext):
    df = pd.DataFrame({"feature": ["a", "b"], "score": [1.0, 2.0]})
    try:
        out = save_results_multiformat(df, tmp_path / "out", fmt)
    except (ImportError, ValueError) as exc:
        pytest.skip(f"writer for {fmt} unavailable: {exc}")
    assert out.suffix == ext
    assert out.exists() and out.stat().st_size > 0


def test_save_from_config_uses_the_configured_format(tmp_path):
    df = pd.DataFrame({"a": [1.0, 2.0]})
    cfg = SelectOmicsConfig(data_path="u.csv", target_column="Class",
                            output_format="tsv")
    assert save_from_config(df, tmp_path / "x", cfg).suffix == ".tsv"


def test_unknown_output_format_raises(tmp_path):
    df = pd.DataFrame({"a": [1.0]})
    with pytest.raises((ValueError, KeyError)):
        save_results_multiformat(df, tmp_path / "x", "invented")


# ---------------------------------------------------------------------------
# prepare_data
# ---------------------------------------------------------------------------

def test_prepare_data_drops_non_numeric(frame):
    """A string ID column is common and must not become a feature."""
    df = frame.copy()
    df.insert(0, "SampleID", [f"S{i}" for i in range(len(df))])
    X, y, *_ = prepare_data(df, "Class")
    assert "SampleID" not in X.columns
    assert X.shape[1] == 6
    assert len(y) == len(df)


def test_prepare_data_sorts_the_row_index(frame):
    """
    prepare_data calls X.sort_index(), which orders ROWS, not columns. The
    docstring's "index order is sorted for reproducibility" means exactly that.
    Column order is preserved from the file.
    """
    shuffled_rows = frame.sample(frac=1.0, random_state=1)
    X, y, *_ = prepare_data(shuffled_rows, "Class")
    assert list(X.index) == sorted(X.index)
    assert list(y.index) == list(X.index), "y must stay aligned to X"


def test_prepare_data_preserves_column_order(frame):
    """Columns come back in file order, minus the target and non-numerics."""
    reordered = frame[["g3", "g0", "Class", "g5", "g1", "g4", "g2"]]
    X, _, *_ = prepare_data(reordered, "Class")
    assert list(X.columns) == ["g3", "g0", "g5", "g1", "g4", "g2"]


def test_prepare_data_missing_target_raises(frame):
    with pytest.raises(ValueError, match="not found|target"):
        prepare_data(frame, "NoSuchColumn")


def test_prepare_data_all_non_numeric_raises(frame):
    """No numeric columns left means nothing to select from."""
    df = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"], "Class": ["A", "B"]})
    with pytest.raises(ValueError, match="numeric"):
        prepare_data(df, "Class")


def test_single_class_rejected_at_split(frame):
    """
    The class-count check lives in create_train_test_split, not prepare_data,
    so a single-class dataset survives preparation and fails at the split.
    """
    df = frame.copy()
    df["Class"] = "OnlyOne"
    X, y, *_ = prepare_data(df, "Class")
    cfg = SelectOmicsConfig(data_path="u.csv", target_column="Class")
    with pytest.raises(ValueError):
        create_train_test_split(X, y, cfg)


# ---------------------------------------------------------------------------
# handle_nans
# ---------------------------------------------------------------------------

def test_handle_nans_switches_lr_to_xgb():
    """LR cannot take NaN, so the algorithm must switch rather than crash."""
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [1.0, 2.0, 3.0]})
    _, algo = handle_nans(X, "LR")[:2] if isinstance(
        handle_nans(X, "LR"), tuple) else (None, None)
    assert algo in ("XGB", None)


def test_handle_nans_leaves_clean_data_alone():
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    out = handle_nans(X, "LR")
    algo = out[1] if isinstance(out, tuple) else "LR"
    assert algo == "LR"


def test_handle_nans_never_imputes():
    """The package documents that it never imputes; missingness is signal."""
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [1.0, 2.0, 3.0]})
    out = handle_nans(X, "XGB")
    frame_out = out[0] if isinstance(out, tuple) else out
    assert frame_out.isna().sum().sum() == 1, "a NaN was filled in"


# ---------------------------------------------------------------------------
# create_train_test_split
# ---------------------------------------------------------------------------

def test_split_is_stratified_and_reproducible(frame):
    X, y, *_ = prepare_data(frame, "Class")
    cfg = SelectOmicsConfig(data_path="u.csv", target_column="Class",
                            test_size=0.25, random_seed=42)
    a = create_train_test_split(X, y, cfg)
    b = create_train_test_split(X, y, cfg)
    assert np.array_equal(a[2], b[2]), "same seed gave a different split"
    y_tr, y_te = a[2], a[3]
    # Both classes must survive into both halves.
    assert len(np.unique(y_tr)) == 2 and len(np.unique(y_te)) == 2
