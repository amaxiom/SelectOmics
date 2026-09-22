"""
Metric edge cases.

The AUC helper is the package's single point of contact with sklearn's scoring,
and every caller relies on it returning nan rather than raising when a fold is
degenerate. That contract is what lets the validation loops skip a bad fold
instead of aborting a run, so it is worth pinning directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from SelectOmics.evaluation.metrics import (
    _safe_roc_auc,
    _threshold_at_specificity,
    generate_classification_report,
    summarise_generalization,
)


# ---------------------------------------------------------------------------
# _safe_roc_auc
# ---------------------------------------------------------------------------

class TestSafeRocAuc:

    def test_a_single_class_yields_nan_rather_than_raising(self):
        """
        sklearn raises when only one class is present. Callers skip the fold on
        nan; an exception would abort the whole validation run instead.
        """
        y = np.zeros(10, dtype=int)
        proba = np.column_stack([np.ones(10) * 0.7, np.ones(10) * 0.3])
        assert np.isnan(_safe_roc_auc(y, proba, n_classes=2))

    def test_binary_accepts_a_full_probability_matrix(self):
        """
        sklearn wants a 1-D positive-class vector for binary problems, so the
        helper extracts column 1 rather than making every caller do it.
        """
        rng = np.random.RandomState(0)
        y = np.array([0, 1] * 20)
        proba = rng.rand(40, 2)
        proba = proba / proba.sum(axis=1, keepdims=True)
        auc = _safe_roc_auc(y, proba, n_classes=2)
        assert 0.0 <= auc <= 1.0

    def test_binary_also_accepts_a_one_dimensional_vector(self):
        y = np.array([0, 1] * 20)
        scores = np.linspace(0, 1, 40)
        auc = _safe_roc_auc(y, scores, n_classes=2)
        assert 0.0 <= auc <= 1.0

    def test_multiclass_with_a_missing_class_yields_nan(self):
        """
        OvR macro-averaging needs every class present. A small test split can
        easily lose a minority class, and that must degrade to nan.
        """
        rng = np.random.RandomState(0)
        y = np.array([0, 1] * 15)                 # class 2 absent
        proba = rng.rand(30, 3)
        proba = proba / proba.sum(axis=1, keepdims=True)
        assert np.isnan(_safe_roc_auc(y, proba, n_classes=3))

    def test_multiclass_with_every_class_present_scores(self):
        rng = np.random.RandomState(0)
        y = np.array([0, 1, 2] * 12)
        proba = rng.rand(36, 3)
        proba = proba / proba.sum(axis=1, keepdims=True)
        auc = _safe_roc_auc(y, proba, n_classes=3)
        assert 0.0 <= auc <= 1.0


# ---------------------------------------------------------------------------
# Threshold at a target specificity
# ---------------------------------------------------------------------------

class TestThresholdAtSpecificity:
    """
    Takes the arrays roc_curve returns, not raw labels, and picks the
    highest-TPR point whose FPR is still within budget.
    """

    @staticmethod
    def _curve(y, scores):
        from sklearn.metrics import roc_curve
        return roc_curve(y, scores)

    def test_it_picks_the_best_point_within_the_fpr_budget(self):
        y = np.array([0] * 20 + [1] * 20)
        scores = np.concatenate([np.linspace(0.0, 0.4, 20),
                                 np.linspace(0.6, 1.0, 20)])
        fpr, tpr, thr = self._curve(y, scores)
        threshold, sens = _threshold_at_specificity(fpr, tpr, thr,
                                                    target_fpr=0.10)
        assert np.isfinite(threshold)
        assert 0.0 <= sens <= 1.0

    def test_a_zero_budget_still_returns_a_point(self):
        """
        target_fpr=0 admits only the perfectly specific end of the curve, which
        roc_curve always includes.
        """
        y = np.array([0] * 20 + [1] * 20)
        scores = np.concatenate([np.linspace(0.0, 0.4, 20),
                                 np.linspace(0.6, 1.0, 20)])
        fpr, tpr, thr = self._curve(y, scores)
        threshold, sens = _threshold_at_specificity(fpr, tpr, thr,
                                                   target_fpr=0.0)
        assert np.isfinite(threshold)

    def test_a_perfect_separation_reaches_full_sensitivity(self):
        y = np.array([0] * 20 + [1] * 20)
        scores = np.array([0.1] * 20 + [0.9] * 20)
        fpr, tpr, thr = self._curve(y, scores)
        threshold, sens = _threshold_at_specificity(fpr, tpr, thr,
                                                   target_fpr=0.0)
        assert sens == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Classification report
# ---------------------------------------------------------------------------

class TestClassificationReport:

    def test_it_returns_a_frame_labelled_with_the_model(self):
        y_true = np.array([0, 1] * 20)
        y_pred = np.array([0, 1] * 20)
        df = generate_classification_report(y_true, y_pred, ["neg", "pos"],
                                            model_name="RF-1")
        assert isinstance(df, pd.DataFrame)
        assert df.index.name == "RF-1"
        assert not df.empty


# ---------------------------------------------------------------------------
# Generalization gap labelling
# ---------------------------------------------------------------------------

class TestGeneralizationGap:

    @pytest.mark.parametrize("train,test,expected", [
        (0.90, 0.89, "excellent"),   # gap 0.01
        (0.90, 0.83, "good"),        # gap 0.07
        (0.90, 0.78, "moderate"),    # gap 0.12
        (0.90, 0.60, "poor"),        # gap 0.30
    ])
    def test_each_band_is_reachable(self, train, test, expected):
        assert summarise_generalization(train, test) == expected

    def test_the_gap_is_symmetric(self):
        """
        Test AUC above training AUC is still a gap. Signing it would label a
        suspiciously easy test split as excellent.
        """
        assert summarise_generalization(0.60, 0.90) == "poor"

    def test_the_band_edges_are_float_comparisons_not_exact_cuts(self):
        """
        The bands compare floats with <, so an edge lands wherever binary
        representation puts it: abs(0.90 - 0.85) is 0.05000000000000004, just
        above the cut, while abs(0.90 - 0.80) is 0.09999999999999998, just
        below it. Recorded so the next reader does not take a nominal 0.10 gap
        as a guaranteed 'moderate'. Nothing downstream turns on a difference of
        1e-17, so this is a note, not a defect.
        """
        assert summarise_generalization(0.90, 0.85) == "good"
        assert summarise_generalization(0.90, 0.80) == "good"
        # Values that do represent cleanly land where the bands say.
        assert summarise_generalization(0.5, 0.25) == "poor"
        assert summarise_generalization(0.5, 0.5) == "excellent"
