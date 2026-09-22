"""
The advice SelectOmicsConfig.suggest() gives about a dataset.

suggest() inspects a file and returns a config plus a list of notes explaining
what it saw and why it chose what it chose. Those notes are the package's only
proactive guidance, and each one fires on a specific property of the data, so a
note that never fires is advice a user silently never receives. Each test here
builds a dataset with exactly the property one note is written for.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from SelectOmics.config import SelectOmicsConfig


def _write(tmp_path, n=60, p=20, n_classes=2, imbalance=None, missing=0.0,
           name="d.csv", seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)

    if imbalance is not None:
        n_minor = max(1, int(round(n * imbalance)))
        y = np.array([1] * n_minor + [0] * (n - n_minor))
    else:
        y = np.array([i % n_classes for i in range(n)])

    df = pd.DataFrame(X, columns=[f"f{i:05d}" for i in range(p)])
    if missing > 0:
        mask = rng.rand(n, p) < missing
        df = df.mask(mask)
    df["Class"] = y
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def _notes(path, caplog):
    """suggest() reports through the logger, so capture what it said."""
    caplog.clear()
    caplog.set_level(logging.INFO, logger="SelectOmics")
    cfg = SelectOmicsConfig.suggest(str(path), "Class")
    return cfg, caplog.text


# ---------------------------------------------------------------------------
# Data properties that change the recommendation
# ---------------------------------------------------------------------------

class TestSuggestNotes:

    def test_missing_values_select_an_algorithm_that_handles_them(
        self, tmp_path, caplog
    ):
        path = _write(tmp_path, missing=0.10)
        cfg, text = _notes(path, caplog)
        assert cfg.algorithm == "XGB"
        assert "Missing values" in text
        assert "NaN" in text

    def test_a_clean_dataset_says_nothing_about_missing_values(
        self, tmp_path, caplog
    ):
        """A note that fires on every dataset carries no information."""
        path = _write(tmp_path, missing=0.0)
        _, text = _notes(path, caplog)
        assert "Missing values detected" not in text

    def test_severe_imbalance_suggests_a_remedy(self, tmp_path, caplog):
        path = _write(tmp_path, n=100, imbalance=0.05)
        _, text = _notes(path, caplog)
        assert "imbalance" in text.lower()
        assert "SMOTE" in text or "class_weight" in text

    def test_moderate_imbalance_is_distinguished_from_severe(self, tmp_path,
                                                             caplog):
        """
        A 3:1 split needs a different message from a 19:1 one, or the advice is
        the same regardless of what was found.
        """
        path = _write(tmp_path, n=100, imbalance=0.25)
        _, text = _notes(path, caplog)
        assert "Moderate class imbalance" in text
        assert "SMOTE" not in text

    def test_a_tiny_minority_class_warns_about_cv_folds(self, tmp_path, caplog):
        path = _write(tmp_path, n=100, imbalance=0.06)
        _, text = _notes(path, caplog)
        assert "Smallest class" in text
        assert "fold" in text.lower()

    def test_multiclass_explains_how_auc_is_computed(self, tmp_path, caplog):
        path = _write(tmp_path, n=90, n_classes=3)
        cfg, text = _notes(path, caplog)
        assert "Multiclass" in text
        assert "OVR" in text or "macro" in text

    def test_a_binary_problem_does_not_mention_multiclass(self, tmp_path,
                                                          caplog):
        path = _write(tmp_path, n=60, n_classes=2)
        _, text = _notes(path, caplog)
        assert "Multiclass problem" not in text

    def test_an_extreme_dimensional_regime_is_called_out(self, tmp_path,
                                                         caplog):
        """n/p below 0.1 is the regime the package is built for."""
        path = _write(tmp_path, n=30, p=400)
        _, text = _notes(path, caplog)
        assert "high-dimensional regime" in text.lower()
        assert "n/p" in text

    def test_the_advice_does_not_reference_a_step_that_was_removed(
        self, tmp_path, caplog
    ):
        """
        Step 4 was removed. Advice telling a user to run "all 4 steps" would
        name something the package no longer has.
        """
        path = _write(tmp_path, n=30, p=400)
        _, text = _notes(path, caplog)
        assert "4 steps" not in text
        assert "step4" not in text.lower()

    @pytest.mark.slow
    def test_a_very_wide_dataset_reduces_the_tuning_budget(self, tmp_path,
                                                           caplog):
        """
        Past 10,000 features the tuning budget is cut, because the default
        would dominate the runtime.
        """
        path = _write(tmp_path, n=20, p=10_001, name="wide.csv")
        cfg, text = _notes(path, caplog)
        assert cfg.quick_tune_iterations == 30
        assert "High-dimensional dataset" in text


# ---------------------------------------------------------------------------
# The redundancy assessment inside suggest()
# ---------------------------------------------------------------------------

class TestSuggestRedundancy:

    def test_a_failed_assessment_is_reported_not_swallowed(self, tmp_path,
                                                           caplog, monkeypatch):
        """
        The redundancy read is advisory. If it cannot be computed, suggest()
        still returns a config and says the guidance is unavailable.
        """
        import SelectOmics.utils.diagnostics as diag

        def boom(*args, **kwargs):
            raise ValueError("cannot assess redundancy")

        monkeypatch.setattr(diag, "assess_feature_redundancy", boom)
        path = _write(tmp_path)
        cfg, text = _notes(path, caplog)
        assert isinstance(cfg, SelectOmicsConfig)
        assert "unavailable" in text

    def test_highly_redundant_data_gets_the_step1_tradeoff(self, tmp_path,
                                                           caplog):
        """
        On redundant data Step 1 does most of the work but costs cross-seed
        stability. A user choosing between a short list and a reproducible one
        needs that stated.
        """
        rng = np.random.RandomState(0)
        n, n_blocks = 60, 6
        base = rng.randn(n, n_blocks)
        # Each block repeated: high pairwise correlation throughout.
        cols = np.hstack([np.repeat(base[:, [i]], 5, axis=1)
                          + rng.randn(n, 5) * 0.01
                          for i in range(n_blocks)])
        df = pd.DataFrame(cols, columns=[f"f{i:03d}" for i in range(cols.shape[1])])
        df["Class"] = [i % 2 for i in range(n)]
        path = tmp_path / "redundant.csv"
        df.to_csv(path, index=False)

        _, text = _notes(path, caplog)
        assert "Step 1" in text
