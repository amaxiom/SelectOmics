"""
Every input and output format the loader claims to support.

The format table is a promise: config.input_format accepts seven names and the
extension map auto-detects eight suffixes. Only CSV was ever exercised, so the
rest of the table was documentation rather than tested behaviour. These round
trip each one and check the failure messages name what a user has to change.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from SelectOmics.config import SelectOmicsConfig
from SelectOmics.data.loaders import (
    _INPUT_EXTENSION_MAP,
    load_omics_data,
    save_results_multiformat,
)


@pytest.fixture
def frame():
    rng = np.random.RandomState(0)
    df = pd.DataFrame(rng.randn(20, 4),
                      columns=[f"f{i}" for i in range(4)])
    df["Class"] = [i % 2 for i in range(20)]
    return df


def _cfg(path, **over):
    kw = dict(data_path=str(path), target_column="Class")
    kw.update(over)
    return SelectOmicsConfig(**kw)


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------

class TestFormatRoundTrips:

    def test_csv(self, frame, tmp_path):
        p = tmp_path / "d.csv"
        frame.to_csv(p, index=False)
        assert load_omics_data(_cfg(p)).shape == frame.shape

    def test_tsv(self, frame, tmp_path):
        p = tmp_path / "d.tsv"
        frame.to_csv(p, sep="\t", index=False)
        assert load_omics_data(_cfg(p)).shape == frame.shape

    def test_parquet(self, frame, tmp_path):
        p = tmp_path / "d.parquet"
        frame.to_parquet(p)
        assert load_omics_data(_cfg(p)).shape == frame.shape

    def test_feather(self, frame, tmp_path):
        p = tmp_path / "d.feather"
        frame.to_feather(p)
        assert load_omics_data(_cfg(p)).shape == frame.shape

    def test_json(self, frame, tmp_path):
        p = tmp_path / "d.json"
        frame.to_json(p, orient="records", indent=2)
        assert load_omics_data(_cfg(p)).shape == frame.shape

    def test_excel_with_the_sheet_reported(self, frame, tmp_path, caplog):
        caplog.set_level(logging.INFO, logger="SelectOmics")
        p = tmp_path / "d.xlsx"
        frame.to_excel(p, index=False, sheet_name="measurements")
        cfg = _cfg(p, excel_sheet_name="measurements", verbose=True)
        assert load_omics_data(cfg).shape == frame.shape
        assert "measurements" in caplog.text, (
            "the sheet actually read was not reported, so a run against the "
            "wrong sheet would look identical in the log"
        )

    def test_excel_by_sheet_index(self, frame, tmp_path, caplog):
        caplog.set_level(logging.INFO, logger="SelectOmics")
        p = tmp_path / "d.xlsx"
        frame.to_excel(p, index=False)
        cfg = _cfg(p, excel_sheet_name=0, verbose=True)
        assert load_omics_data(cfg).shape == frame.shape
        assert "index 0" in caplog.text

    def test_hdf5_with_the_key_reported(self, frame, tmp_path, caplog):
        caplog.set_level(logging.INFO, logger="SelectOmics")
        p = tmp_path / "d.h5"
        frame.to_hdf(p, key="measurements", mode="w")
        cfg = _cfg(p, hdf5_key="measurements", verbose=True)
        assert load_omics_data(cfg).shape == frame.shape
        assert "measurements" in caplog.text


# ---------------------------------------------------------------------------
# Failure messages
# ---------------------------------------------------------------------------

class TestLoaderFailures:

    def test_a_missing_file_is_named(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_omics_data(_cfg(tmp_path / "absent.csv"))

    def test_an_unrecognised_extension_lists_what_is_supported(self, frame,
                                                               tmp_path):
        """
        A user who names a file .dat needs to know which suffixes work and that
        input_format can override the guess.
        """
        p = tmp_path / "d.dat"
        frame.to_csv(p, index=False)
        with pytest.raises(ValueError) as exc:
            load_omics_data(_cfg(p))
        msg = str(exc.value)
        assert "auto-detect" in msg
        assert "input_format" in msg
        for ext in sorted(_INPUT_EXTENSION_MAP)[:3]:
            assert ext in msg

    def test_an_unsupported_format_is_rejected_at_config_time(self, frame,
                                                              tmp_path):
        """
        The config validates input_format on construction, so a bad name fails
        before any file is opened. That makes the loader's own check defensive
        rather than the one users hit.
        """
        p = tmp_path / "d.csv"
        frame.to_csv(p, index=False)
        with pytest.raises(ValueError, match="input_format must be one of"):
            _cfg(p, input_format="hieroglyphics")

    def test_the_loader_still_guards_an_unsupported_format_itself(self, frame,
                                                                 tmp_path):
        """
        Reached only by setting the field after construction, which is how any
        caller bypassing validation would arrive here.
        """
        p = tmp_path / "d.csv"
        frame.to_csv(p, index=False)
        cfg = _cfg(p)
        object.__setattr__(cfg, "input_format", "hieroglyphics")
        with pytest.raises(ValueError, match="Unsupported input format"):
            load_omics_data(cfg)

    def test_an_empty_file_is_rejected_rather_than_returned(self, tmp_path):
        """
        An empty frame would flow downstream and fail somewhere far less
        informative than the load.
        """
        p = tmp_path / "empty.csv"
        p.write_text("f0,f1,Class\n", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            load_omics_data(_cfg(p))

    def test_a_corrupt_file_is_wrapped_with_its_format(self, tmp_path):
        """
        The RuntimeError names the format attempted, since the usual cause is a
        file whose contents do not match its extension.
        """
        p = tmp_path / "d.h5"
        p.write_bytes(b"definitely not hdf5")
        with pytest.raises(RuntimeError, match="HDF5"):
            load_omics_data(_cfg(p))

    @pytest.mark.parametrize("suffix,label", [(".parquet", "PARQUET"),
                                              (".feather", "FEATHER")])
    def test_a_corrupt_arrow_file_is_wrapped_like_any_other(self, tmp_path,
                                                            suffix, label):
        """
        pyarrow raises ArrowInvalid, which subclasses ValueError. While
        ValueError passed straight through, a corrupt parquet or feather file
        surfaced pyarrow's own message naming neither the path nor the format,
        where the same corruption in HDF5 or Excel was named properly.
        """
        p = tmp_path / f"d{suffix}"
        p.write_bytes(b"definitely not columnar")
        with pytest.raises(RuntimeError) as exc:
            load_omics_data(_cfg(p))
        assert label in str(exc.value)
        assert str(p) in str(exc.value)

    def test_a_missing_file_is_not_reclassified(self, tmp_path):
        """
        FileNotFoundError passes through untouched: wrapping it in RuntimeError
        would lose the type callers catch on, and it says the one thing the
        wrapped message could not add.
        """
        with pytest.raises(FileNotFoundError):
            load_omics_data(_cfg(tmp_path / "nope.parquet"))
# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

class TestSaveFormats:

    @pytest.mark.parametrize("fmt,suffix", [
        ("csv", ".csv"), ("tsv", ".tsv"), ("parquet", ".parquet"),
        ("feather", ".feather"), ("json", ".json"), ("excel", ".xlsx"),
    ])
    def test_each_output_format_writes_a_readable_file(self, frame, tmp_path,
                                                       fmt, suffix):
        out = tmp_path / "results"
        save_results_multiformat(frame, out, output_format=fmt)
        written = out.with_suffix(suffix)
        assert written.exists() and written.stat().st_size > 0

    def test_hdf5_output(self, frame, tmp_path):
        out = tmp_path / "results"
        save_results_multiformat(frame, out, output_format="hdf5",
                                 hdf5_key="results")
        assert out.with_suffix(".h5").exists()

    def test_an_unknown_output_format_is_rejected_before_writing(self, frame,
                                                                 tmp_path):
        """
        Rejected up front, and the message lists what is available rather than
        leaving the caller to guess.
        """
        out = tmp_path / "results"
        with pytest.raises(ValueError) as exc:
            save_results_multiformat(frame, out, output_format="nonsense")
        msg = str(exc.value)
        assert "Unsupported output format" in msg
        assert "csv" in msg and "parquet" in msg


class TestSaveFailures:

    def test_a_write_failure_is_wrapped_with_the_format(self, frame, tmp_path):
        """
        A path that cannot be written must fail with a message naming the file
        and the format, not with a bare pandas error from three layers down.
        """
        blocked = tmp_path / "results.csv"
        blocked.mkdir()                      # a directory where a file must go
        with pytest.raises(RuntimeError) as exc:
            save_results_multiformat(frame, tmp_path / "results",
                                     output_format="csv")
        msg = str(exc.value)
        assert "CSV" in msg
        assert "results" in msg
