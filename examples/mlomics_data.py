"""
Fetch the MLOmics benchmark data that the examples and benchmarks run on.

The data is not committed to this repository. It is published by the MLOmics
authors under CC-BY-4.0 and is downloaded on first use, cached beside this
module, and reused thereafter. Committing it would add roughly 400 MB of
third-party data to every clone and would re-distribute it under this
repository's licence rather than its own.

Reproducibility
---------------
Every URL is pinned to an immutable dataset revision, not to ``main``. A
dataset that moves under a pinned revision cannot change what these notebooks
produce. Each file's SHA-256 is recorded in ``_MANIFEST`` and checked after
download, so a truncated or substituted file is refused rather than silently
analysed.

Which variant, and why it matters
---------------------------------
MLOmics publishes three variants of each layer: ``Original``, ``Aligned`` and
``Top``. These modules use ``Aligned``, the sample-aligned matrices, because
the multi-omic examples need the same patients across all four layers.

The variants are not interchangeable. Measured on GBM microRNA, ``Original``
shares only 143 of its features with ``Aligned`` and the values differ on the
shared block by up to 7.45. Pointing these URLs at ``Original`` would change
the science while still appearing to work, which is why the variant is part of
the pinned path rather than a parameter.

Citation
--------
MLOmics: Cancer Multi-Omics Database for Machine Learning.
*Scientific Data* (2025). doi:10.1038/s41597-025-05235-x
https://huggingface.co/datasets/AIBIC/MLOmics

Usage
-----
    from mlomics_data import build_input, build_merged

    df = build_input("GBM", "miRNA")   # samples x features, plus 'Label'
    df = build_merged("GBM")           # all four layers joined, plus 'Label'
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Pinned source
# ---------------------------------------------------------------------------

REPO = "AIBIC/MLOmics"

# An immutable commit, not a branch. Bump this deliberately, and re-run
# everything that depends on it: the tuning pass sees the full feature matrix
# before Step 1 removes anything, so even a change in constant columns can
# move the selected panel.
REVISION = "4bfee25b63781c5cce1acaf64f8cbf3d51f02b12"

VARIANT = "Aligned"

_BASE = (f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}"
         f"/Main_Dataset/Classification_datasets")

COHORTS: Tuple[str, ...] = ("GBM", "OV", "LGG")
LAYERS: Tuple[str, ...] = ("miRNA", "mRNA", "Methy", "CNV")

CACHE_DIR = Path(__file__).resolve().parent / ".mlomics_cache"

# filename -> (sha256, size_bytes), recorded from the pinned revision and
# checked after every download. A mismatch means the file was corrupted or
# substituted, because a pinned revision cannot legitimately change.
_MANIFEST: Dict[str, Tuple[str, int]] = {
    'GBM_CNV_aligned.csv':
        ('74d3595b234e6f5d97a6d03907e653a849f6139d3dd2f6b0190d159190ad7d54', 19656115),
    'GBM_Methy_aligned.csv':
        ('d263b19d4ce1498e7b6820ef1efa590356bc658550c275d5e08ce9dc09cfa7d9', 30814632),
    'GBM_label_num.csv':
        ('0744feaa57579f5b73dfccd7e4f06eaf33b5c095970f134d22c10c25473b555c', 494),
    'GBM_mRNA_aligned.csv':
        ('102daf01134a1dbf6a412c813893d448d33292b843560a330caa681f61f5ea1d', 32188488),
    'GBM_miRNA_aligned.csv':
        ('cce56145a079253719dc26285b83689c8b6ce3b7128f40e8c597e430d3979e53', 726061),
    'LGG_CNV_aligned.csv':
        ('5d9a34635b7b733e2b73f9ddaafa3c4c26d10fa94c104b9e859c33c3bbc7f9bb', 19962530),
    'LGG_Methy_aligned.csv':
        ('037b39068187d8a44a20c90a1fce6eeff765bc94c2854fc231f94283d4dbbd75', 31183915),
    'LGG_label_num.csv':
        ('22a4f5b40cb25f720c37edff459ae5ad9646199de9281ca5d36bcf938b4220eb', 500),
    'LGG_mRNA_aligned.csv':
        ('7f017223dcd2434f25d946626c57ce04278f1a7c88a4cbbdb06480614a876428', 27381290),
    'LGG_miRNA_aligned.csv':
        ('27d5c9507936f06f70ec1c78ed439ab867a4046fa9979875fc1086e321c12ca3', 619679),
    'OV_CNV_aligned.csv':
        ('d1014a73a0b5606e0346b6a57498821f26170b3fe478a4745c1d8dd2562e0588', 23296174),
    'OV_Methy_aligned.csv':
        ('e442da556fdf8f71a9c622dd2fd4aae39557c288d03f12e111ae280646a7ca6c', 36030176),
    'OV_label_num.csv':
        ('5f82ef2f1494a5bda3401dc8a6ed19d0555416dafbe3e31b592c9278f1f2e1ed', 574),
    'OV_mRNA_aligned.csv':
        ('e9e4fe5e30f437c66dac03ae597354d3c085713d11c5c954aa768d8cbdd0be77', 31406867),
    'OV_miRNA_aligned.csv':
        ('5fb26e1db5c02679af13f9983e182983e54c695a95a999298e2cf08594389a2e', 704106),
}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def layer_url(cohort: str, layer: str) -> str:
    """URL of one omic layer, pinned to REVISION and the Aligned variant."""
    _check(cohort, layer)
    return f"{_BASE}/GS-{cohort}/{VARIANT}/{cohort}_{layer}_aligned.csv"


def labels_url(cohort: str) -> str:
    """URL of one cohort's label vector."""
    _check(cohort)
    return f"{_BASE}/GS-{cohort}/{VARIANT}/{cohort}_label_num.csv"


def _check(cohort: str, layer: str = None) -> None:
    if cohort not in COHORTS:
        raise ValueError(f"Unknown cohort {cohort!r}. Choose from {COHORTS}.")
    if layer is not None and layer not in LAYERS:
        raise ValueError(f"Unknown layer {layer!r}. Choose from {LAYERS}.")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, verbose: bool = True) -> Path:
    """
    Download ``url`` to ``dest`` unless it is already there and intact.

    Writes to a temporary file and moves it into place, so an interrupted
    download never leaves a half-file that looks cached.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = _MANIFEST.get(dest.name)

    if dest.exists():
        if expected is None or _sha256(dest) == expected[0]:
            return dest
        if verbose:
            print(f"  cached {dest.name} failed its checksum; re-downloading")
        dest.unlink()

    if verbose:
        print(f"  downloading {dest.name} ...", end="", flush=True)
    # mkstemp hands back an OPEN descriptor. Leaving it open blocks the
    # rename below on Windows, so close it before writing through our own
    # handle.
    _fd, _tmp_name = tempfile.mkstemp(dir=str(dest.parent), suffix=".part")
    os.close(_fd)
    tmp = Path(_tmp_name)
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as fh:
            shutil.copyfileobj(r, fh)
    except urllib.error.URLError as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download {url}\n"
            f"  {exc}\n"
            f"  The examples need the MLOmics data, which is not committed to "
            f"this repository. Check your network, or download the file by "
            f"hand and place it at {dest}."
        ) from exc

    if expected is not None:
        got = _sha256(tmp)
        if got != expected[0]:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for {dest.name}.\n"
                f"  expected {expected[0]}\n"
                f"  got      {got}\n"
                f"  The pinned revision should be immutable, so this means the "
                f"download was corrupted or intercepted. Nothing was cached."
            )
    tmp.replace(dest)
    if verbose:
        print(f" {dest.stat().st_size / 1e6:.1f} MB")
    return dest


def layer_path(cohort: str, layer: str, verbose: bool = True) -> Path:
    """Local path to one layer, downloading it if needed."""
    dest = CACHE_DIR / f"{cohort}_{layer}_aligned.csv"
    return fetch(layer_url(cohort, layer), dest, verbose=verbose)


def labels_path(cohort: str, verbose: bool = True) -> Path:
    """Local path to one cohort's labels, downloading if needed."""
    dest = CACHE_DIR / f"{cohort}_label_num.csv"
    return fetch(labels_url(cohort), dest, verbose=verbose)


# ---------------------------------------------------------------------------
# Model-ready frames
# ---------------------------------------------------------------------------

TARGET = "Label"


def load_layer(cohort: str, layer: str, verbose: bool = True) -> pd.DataFrame:
    """One layer as published: rows are features, columns are patients."""
    return pd.read_csv(layer_path(cohort, layer, verbose), index_col=0)


def build_input(cohort: str, layer: str, verbose: bool = True) -> pd.DataFrame:
    """
    One layer as SelectOmics wants it: rows are samples, columns are features.

    Feature names carry a layer suffix so a merged matrix can be broken back
    down by layer. The label column is attached last.
    """
    raw = load_layer(cohort, layer, verbose)
    df = raw.T.reset_index(drop=True)
    df.columns = [f"{c}_{layer}" for c in df.columns]
    labels = pd.read_csv(labels_path(cohort, verbose))
    if len(labels) != len(df):
        raise RuntimeError(
            f"{cohort} {layer}: {len(df)} samples but {len(labels)} labels. "
            f"The published label vector aligns positionally with the patient "
            f"columns, so a length mismatch means the two files disagree."
        )
    df[TARGET] = labels.iloc[:, 0].values
    return df


def build_merged(cohort: str, verbose: bool = True) -> pd.DataFrame:
    """
    All four layers of one cohort, joined on sample, plus the label column.

    The layers are sample-aligned upstream, so this is a positional
    concatenation; the row count is checked against the labels once.
    """
    frames = []
    for layer in LAYERS:
        raw = load_layer(cohort, layer, verbose)
        part = raw.T.reset_index(drop=True)
        part.columns = [f"{c}_{layer}" for c in part.columns]
        frames.append(part)

    widths = {len(f) for f in frames}
    if len(widths) != 1:
        raise RuntimeError(
            f"{cohort}: layers disagree on sample count ({sorted(widths)}). "
            f"The Aligned variant should be sample-aligned across layers."
        )

    df = pd.concat(frames, axis=1)
    labels = pd.read_csv(labels_path(cohort, verbose))
    if len(labels) != len(df):
        raise RuntimeError(
            f"{cohort} merged: {len(df)} samples but {len(labels)} labels."
        )
    df[TARGET] = labels.iloc[:, 0].values
    return df


def merged_csv(cohort: str, dest: Path = None, verbose: bool = True) -> Path:
    """
    Write the merged matrix to CSV and return its path, building it if absent.

    Used by the benchmark harnesses, which take a file path rather than a
    frame.
    """
    dest = Path(dest) if dest else CACHE_DIR / f"{cohort}_merged.csv"
    if dest.exists():
        return dest
    if verbose:
        print(f"  building {dest.name} from the four {cohort} layers ...")
    build_merged(cohort, verbose).to_csv(dest, index=False)
    return dest


if __name__ == "__main__":
    import sys

    which = sys.argv[1:] or list(COHORTS)
    for c in which:
        print(f"{c}:")
        for l in LAYERS:
            p = layer_path(c, l)
            print(f"    {p.name}: {p.stat().st_size / 1e6:.1f} MB")
        labels_path(c)
