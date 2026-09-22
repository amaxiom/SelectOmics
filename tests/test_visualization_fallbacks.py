"""
Plotting fallbacks for environments the developer's machine does not have.

Each branch here exists for a setup this test machine is not in: seaborn not
installed, an older matplotlib whose boxplot still takes `labels`, a plot format
that got past validation. They are invisible in an ordinary run and would only
ever fail on a user's machine, so they are driven here by removing the thing
they compensate for.
"""
from __future__ import annotations

import builtins

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from SelectOmics.config import SelectOmicsConfig
from SelectOmics.evaluation.visualization import (
    plot_auc_boxplots,
    plot_per_class_roc_and_confusion,
)


@pytest.fixture
def cfg(tmp_path):
    return SelectOmicsConfig(data_path="unused.csv", target_column="Class",
                             output_dir=str(tmp_path), create_visualizations=True)


def _written(path):
    hits = list(path.parent.glob(path.stem + ".*"))
    return bool(hits) and all(h.stat().st_size > 0 for h in hits)


def _confusion_inputs(n=40, n_classes=2, seed=0):
    rng = np.random.RandomState(seed)
    y = np.array([i % n_classes for i in range(n)])
    proba = rng.rand(n, n_classes)
    proba = proba / proba.sum(axis=1, keepdims=True)
    y_binary = np.zeros((n, n_classes), dtype=int)
    y_binary[np.arange(n), y] = 1
    names = [str(c) for c in range(n_classes)]
    # Square and sized to the class count, not hardcoded to binary.
    counts = rng.randint(1, 15, size=(n_classes, n_classes))
    cm = pd.DataFrame(counts, index=names, columns=names)
    return y, y_binary, proba, cm, names


# ---------------------------------------------------------------------------
# seaborn absent
# ---------------------------------------------------------------------------

class TestWithoutSeaborn:

    @staticmethod
    def _hide_seaborn(monkeypatch):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "seaborn" or name.startswith("seaborn."):
                raise ImportError("seaborn is not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

    def test_the_confusion_matrix_still_renders(self, cfg, tmp_path,
                                                monkeypatch):
        """
        seaborn is an optional extra. Without it the heatmap is drawn with
        imshow instead, and the figure must still be written.
        """
        self._hide_seaborn(monkeypatch)
        y, y_bin, proba, cm, names = _confusion_inputs()
        out = tmp_path / "cm_noseaborn.png"
        plot_per_class_roc_and_confusion(y, y_bin, proba, cm, names, "RF",
                                         cfg, out)
        assert _written(out)

    def test_the_fallback_labels_every_class(self, cfg, tmp_path, monkeypatch):
        """
        The imshow path sets its own ticks. Getting that wrong would silently
        mislabel the axes of a confusion matrix, which is worse than not
        drawing one.
        """
        self._hide_seaborn(monkeypatch)
        y, y_bin, proba, cm, names = _confusion_inputs(n=45, n_classes=3)
        out = tmp_path / "cm3_noseaborn.png"
        plot_per_class_roc_and_confusion(y, y_bin, proba, cm, names, "RF",
                                         cfg, out)
        assert _written(out)

    def test_the_seaborn_path_still_works_when_it_is_present(self, cfg,
                                                             tmp_path):
        """The fallback must be a fallback, not the only path that works."""
        y, y_bin, proba, cm, names = _confusion_inputs()
        out = tmp_path / "cm_seaborn.png"
        plot_per_class_roc_and_confusion(y, y_bin, proba, cm, names, "RF",
                                         cfg, out)
        assert _written(out)


# ---------------------------------------------------------------------------
# Older matplotlib
# ---------------------------------------------------------------------------

class TestOlderMatplotlib:

    def test_boxplot_falls_back_to_the_legacy_keyword(self, cfg, tmp_path,
                                                      monkeypatch):
        """
        matplotlib renamed boxplot's `labels` to `tick_labels`. On an older
        install the modern call raises TypeError and the legacy one is used.
        """
        real_boxplot = plt.boxplot
        calls = {"n": 0}

        def picky_boxplot(*args, **kwargs):
            if "tick_labels" in kwargs:
                calls["n"] += 1
                raise TypeError("unexpected keyword argument 'tick_labels'")
            return real_boxplot(*args, **kwargs)

        monkeypatch.setattr(plt, "boxplot", picky_boxplot)
        out = tmp_path / "box_legacy.png"
        plot_auc_boxplots("Step 2", {"Ref-RF": [0.81, 0.83, 0.79],
                                     "Consensus-1": [0.86, 0.88, 0.84]},
                          cfg, out)
        assert calls["n"] == 1, "the modern keyword was never attempted"
        assert _written(out)


# ---------------------------------------------------------------------------
# Plot format
# ---------------------------------------------------------------------------

class TestPlotFormat:

    def test_an_unvalidated_format_falls_back_to_png(self, cfg, tmp_path):
        """
        The config validates plot_format, so this guard is reached only when a
        caller sets the field directly. Falling back beats raising while a
        figure is already open.
        """
        object.__setattr__(cfg, "plot_format", "not-a-format")
        out = tmp_path / "fallback"
        plot_auc_boxplots("Step 2", {"Ref-RF": [0.81, 0.83, 0.79]}, cfg, out)
        assert (tmp_path / "fallback.png").exists()

    def test_the_config_rejects_a_bad_format_up_front(self, tmp_path):
        with pytest.raises(ValueError, match="plot_format"):
            SelectOmicsConfig(data_path="unused.csv", target_column="Class",
                              output_dir=str(tmp_path),
                              plot_format="not-a-format")

    @pytest.mark.parametrize("fmt", ["png", "pdf", "svg"])
    def test_each_valid_format_is_written_with_its_suffix(self, tmp_path, fmt):
        cfg = SelectOmicsConfig(data_path="unused.csv", target_column="Class",
                                output_dir=str(tmp_path), plot_format=fmt)
        out = tmp_path / f"box_{fmt}"
        plot_auc_boxplots("Step 2", {"Ref-RF": [0.81, 0.83, 0.79]}, cfg, out)
        assert (tmp_path / f"box_{fmt}.{fmt}").exists()
