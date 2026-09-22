"""
SelectOmics/data/loaders.py

Multi-format data I/O for the SelectOmics pipeline.

Supports CSV, TSV, Excel, Parquet, HDF5, Feather, and JSON for both
input and output.  All functions accept SelectOmicsConfig directly and
read format-specific settings from config attributes rather than
duplicating defaults.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import pandas as pd

from SelectOmics.config import SelectOmicsConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extension maps
# ---------------------------------------------------------------------------

_INPUT_EXTENSION_MAP: dict = {
    '.csv':     'csv',
    '.tsv':     'tsv',
    '.txt':     'tsv',
    '.xlsx':    'excel',
    '.xls':     'excel',
    '.parquet': 'parquet',
    '.pq':      'parquet',
    '.h5':      'hdf5',
    '.hdf5':    'hdf5',
    '.feather': 'feather',
    '.json':    'json',
}

_OUTPUT_EXTENSION_MAP: dict = {
    'csv':     '.csv',
    'tsv':     '.tsv',
    'excel':   '.xlsx',
    'parquet': '.parquet',
    'hdf5':    '.h5',
    'feather': '.feather',
    'json':    '.json',
}


# ---------------------------------------------------------------------------
# load_omics_data
# ---------------------------------------------------------------------------

# What this loader can read. Separate from _OUTPUT_EXTENSION_MAP, which says
# what save_from_config can write: the two happen to coincide today, and
# reading one from the other would hide it the day they diverge.
_SUPPORTED_INPUT_FORMATS: frozenset = frozenset(
    {'csv', 'tsv', 'excel', 'parquet', 'hdf5', 'feather', 'json'}
)


def load_omics_data(config: SelectOmicsConfig) -> pd.DataFrame:
    """
    Load an omics dataset from disk in any supported format.

    Format is determined by config.input_format.  When set to 'auto',
    the format is inferred from the file extension.  Format-specific
    settings (excel_sheet_name, hdf5_key) are read from config
    attributes.

    Parameters
    ----------
    config : SelectOmicsConfig
        Pipeline configuration.  Relevant attributes:
        data_path, input_format, excel_sheet_name, hdf5_key, verbose.

    Returns
    -------
    pd.DataFrame
        Loaded dataset with original column names and dtypes preserved.

    Raises
    ------
    FileNotFoundError
        If config.data_path does not exist.
    ValueError
        If the format cannot be determined from the extension (auto mode),
        if the format is unsupported, or if the loaded DataFrame is empty.
        These are raised before or after the read, never by a reader.
    RuntimeError
        If reading the file fails, whatever the reader raised. The message
        names the path and the format attempted.
    """
    data_path = Path(config.data_path)
    input_format = config.input_format.lower()

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Resolve format from extension when auto-detection is requested.
    if input_format == 'auto':
        extension = data_path.suffix.lower()
        input_format = _INPUT_EXTENSION_MAP.get(extension)
        if input_format is None:
            raise ValueError(
                f"Cannot auto-detect format from extension '{extension}'. "
                f"Supported extensions: {sorted(_INPUT_EXTENSION_MAP.keys())}. "
                f"Set config.input_format explicitly to override."
            )
        if config.verbose:
            logger.info("  Auto-detected input format: %s (from %s)", input_format.upper(), extension)

    if input_format not in _SUPPORTED_INPUT_FORMATS:
        # Checked before the read, not inside it: the handler below wraps
        # everything a reader raises, and this error is ours rather than a
        # reader's.
        raise ValueError(
            f"Unsupported input format: '{input_format}'. "
            f"Supported: csv, tsv, excel, parquet, hdf5, feather, json."
        )

    if config.verbose:
        logger.info("  Loading data from: %s", data_path)
        logger.info("  Format: %s", input_format.upper())

    try:
        if input_format == 'csv':
            df = pd.read_csv(data_path)

        elif input_format == 'tsv':
            df = pd.read_csv(data_path, sep='\t')

        elif input_format == 'excel':
            df = pd.read_excel(
                data_path,
                sheet_name=config.excel_sheet_name,
                engine='openpyxl',
            )
            if config.verbose:
                sheet_label = (
                    config.excel_sheet_name
                    if isinstance(config.excel_sheet_name, str)
                    else f"index {config.excel_sheet_name}"
                )
                logger.info("  Sheet: %s", sheet_label)

        elif input_format == 'parquet':
            df = pd.read_parquet(data_path)

        elif input_format == 'hdf5':
            df = pd.read_hdf(data_path, key=config.hdf5_key)
            if config.verbose:
                logger.info("  HDF5 key: %s", config.hdf5_key)

        elif input_format == 'feather':
            df = pd.read_feather(data_path)

        else:   # 'json'
            df = pd.read_json(data_path, orient='records')

    except FileNotFoundError:
        raise
    except Exception as exc:
        # Every reader failure is wrapped, including the ValueError subclasses.
        # pyarrow raises ArrowInvalid, which IS a ValueError, so a pass-through
        # for ValueError meant a corrupt parquet or feather file surfaced
        # pyarrow's own message with no mention of the path or the format
        # attempted, while the same corruption in an HDF5 or Excel file was
        # named properly. The usual cause is contents that do not match the
        # extension, and that is what the wrapped message says.
        raise RuntimeError(
            f"Error loading '{data_path}' as {input_format.upper()}: {exc}"
        ) from exc

    if df.empty:
        raise ValueError(
            f"Loaded DataFrame from '{data_path}' is empty."
        )

    if config.verbose:
        logger.info("  Data loaded: %d rows x %d columns", df.shape[0], df.shape[1])

    return df


# ---------------------------------------------------------------------------
# save_results_multiformat
# ---------------------------------------------------------------------------

def save_results_multiformat(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    output_format: str,
    excel_engine: str = 'openpyxl',
    hdf5_key: str = 'data',
) -> Path:
    """
    Save a DataFrame to disk in the specified format.

    The file extension of output_path is replaced to match output_format,
    so callers may pass a stem-only path and receive the correct filename
    back via the return value.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    output_path : str or Path
        Destination path.  The extension is replaced based on output_format.
    output_format : str
        One of: 'csv', 'tsv', 'excel', 'parquet', 'hdf5', 'feather', 'json'.
    excel_engine : str
        Pandas Excel write engine.  'openpyxl' (default) or 'xlsxwriter'.
        Used only when output_format == 'excel'.
    hdf5_key : str
        Dataset key within the HDF5 file.  Default 'data'.
        Used only when output_format == 'hdf5'.

    Returns
    -------
    Path
        Absolute path to the saved file.

    Raises
    ------
    ValueError
        If output_format is not a supported format string.
    RuntimeError
        If pandas raises an unexpected error while writing the file.
    """
    output_format = output_format.lower()

    if output_format not in _OUTPUT_EXTENSION_MAP:
        raise ValueError(
            f"Unsupported output format: '{output_format}'. "
            f"Supported: {sorted(_OUTPUT_EXTENSION_MAP.keys())}."
        )

    output_path = Path(output_path).with_suffix(_OUTPUT_EXTENSION_MAP[output_format])

    # Ensure parent directory exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if output_format == 'csv':
            df.to_csv(output_path, index=False)

        elif output_format == 'tsv':
            df.to_csv(output_path, sep='\t', index=False)

        elif output_format == 'excel':
            df.to_excel(output_path, index=False, engine=excel_engine)

        elif output_format == 'parquet':
            df.to_parquet(output_path, index=False)

        elif output_format == 'hdf5':
            df.to_hdf(output_path, key=hdf5_key, mode='w')

        elif output_format == 'feather':
            df.to_feather(output_path)

        elif output_format == 'json':
            df.to_json(output_path, orient='records', indent=2)

    except Exception as exc:
        raise RuntimeError(
            f"Error saving results to '{output_path}' "
            f"as {output_format.upper()}: {exc}"
        ) from exc

    return output_path


# ---------------------------------------------------------------------------
# save_from_config
# ---------------------------------------------------------------------------

def save_from_config(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    config,
) -> Path:
    """
    Save a DataFrame using every output setting the config carries.

    Call sites that pass only ``config.output_format`` to
    ``save_results_multiformat`` silently discard ``excel_output_engine`` and
    ``hdf5_key``, so those two config fields have no effect on the files the
    pipeline writes.  This wrapper forwards all three.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    output_path : str or Path
        Destination stem; the extension is set from the output format.
    config : SelectOmicsConfig or dict
        Pipeline configuration.  Some callers hold config as a plain dict,
        so both attribute and key access are supported.

    Returns
    -------
    Path
        Absolute path to the saved file.
    """
    def _get(name: str, default):
        if isinstance(config, dict):
            return config.get(name, default)
        return getattr(config, name, default)

    return save_results_multiformat(
        df,
        output_path,
        output_format=_get('output_format', 'csv'),
        excel_engine=_get('excel_output_engine', 'openpyxl'),
        hdf5_key=_get('hdf5_key', 'data'),
    )