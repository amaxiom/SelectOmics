"""
Unit tests for SelectOmics/utils/helpers.py.

All tests are pure computation -- no ML fitting, sub-millisecond each.
"""
import numpy as np
import pytest

from SelectOmics.utils.helpers import (
    _binary_search_percentile,
    agreement_label,
    compute_svm_afi,
    ensure_binary_proba,
)


# ---------------------------------------------------------------------------
# agreement_label
#
# Replaces calculate_required_votes, which computed
# ceil(n_models * consensus_threshold). Both that function and the parameter
# it served are gone: relaxation now always starts at unanimity, so the
# required vote count is simply n_models and the interesting quantity is what
# level the descent ACHIEVED.
# ---------------------------------------------------------------------------

class TestAgreementLabel:
    def test_unanimity(self):
        assert agreement_label(1.0) == 'unanimous'

    def test_band_edges_are_inclusive_below(self):
        assert agreement_label(0.80) == 'strong'
        assert agreement_label(0.60) == 'moderate'
        assert agreement_label(0.40) == 'weak'

    def test_just_under_an_edge_drops_a_band(self):
        assert agreement_label(0.79) == 'moderate'
        assert agreement_label(0.59) == 'weak'
        assert agreement_label(0.39) == 'minimal'

    def test_rank_average_overrides_the_bands(self):
        assert agreement_label(1.0, how='rank_average').startswith('none')

    def test_zero(self):
        assert agreement_label(0.0) == 'minimal'


# ---------------------------------------------------------------------------
# ensure_binary_proba
# ---------------------------------------------------------------------------

class TestEnsureBinaryProba:
    def test_1d_expands_to_2_columns(self):
        p = np.array([0.2, 0.8, 0.5])
        out = ensure_binary_proba(p, n_classes=2)
        assert out.shape == (3, 2)
        np.testing.assert_allclose(out[:, 0], 1 - p)
        np.testing.assert_allclose(out[:, 1], p)

    def test_2d_single_col_expands(self):
        p = np.array([[0.3], [0.7]])
        out = ensure_binary_proba(p, n_classes=2)
        assert out.shape == (2, 2)

    def test_2d_binary_unchanged(self):
        p = np.array([[0.3, 0.7], [0.6, 0.4]])
        out = ensure_binary_proba(p, n_classes=2)
        assert out.shape == (2, 2)
        np.testing.assert_array_equal(out, p)

    def test_multiclass_unchanged(self):
        p = np.array([[0.1, 0.7, 0.2], [0.3, 0.4, 0.3]])
        out = ensure_binary_proba(p, n_classes=3)
        assert out.shape == (2, 3)
        np.testing.assert_array_equal(out, p)

    def test_row_sums_to_one_after_expansion(self):
        p = np.array([0.0, 0.5, 1.0])
        out = ensure_binary_proba(p, n_classes=2)
        np.testing.assert_allclose(out.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# compute_svm_afi
# ---------------------------------------------------------------------------

class TestComputeSvmAfi:
    def _make_linear_svc(self, coef: np.ndarray):
        from unittest.mock import MagicMock
        clf = MagicMock()
        clf.coef_ = coef
        return clf

    def test_binary_output_range(self):
        coef = np.array([[0.1, -0.5, 0.0, 0.9]])
        clf = self._make_linear_svc(coef)
        imp = compute_svm_afi(clf)
        assert imp.shape == (4,)
        assert imp.min() == pytest.approx(0.0)
        assert imp.max() == pytest.approx(1.0)

    def test_multiclass_takes_max_abs(self):
        coef = np.array([
            [0.1, 0.0, -0.8],
            [-0.5, 0.2,  0.3],
        ])
        clf = self._make_linear_svc(coef)
        imp = compute_svm_afi(clf)
        # max abs across classes: [0.5, 0.2, 0.8] -> min-max normalised: (x-0.2)/0.6 -> [0.5, 0.0, 1.0]
        assert imp.shape == (3,)
        assert imp.max() == pytest.approx(1.0)
        assert imp.min() == pytest.approx(0.0)

    def test_all_zero_coef_returns_zeros(self):
        coef = np.zeros((1, 5))
        clf = self._make_linear_svc(coef)
        imp = compute_svm_afi(clf)
        np.testing.assert_array_equal(imp, 0.0)


# ---------------------------------------------------------------------------
# _binary_search_percentile
# ---------------------------------------------------------------------------

class TestBinarySearchPercentile:
    def _monotone_fn(self, slope=-50.0, intercept=50.0):
        """Returns a callable: features = slope * percentile + intercept."""
        def fn(p):
            return slope * p + intercept
        return fn

    def test_target_inside_range(self):
        fn = self._monotone_fn(slope=-40, intercept=40)
        # target=20 -> percentile=0.5
        best_p, best_f, n_evals = _binary_search_percentile(fn, 0.0, 1.0, target=20)
        assert abs(best_f - 20) <= 1.0
        assert 0.0 <= best_p <= 1.0

    def test_target_above_range_picks_lo(self):
        fn = self._monotone_fn(slope=-10, intercept=10)
        # fn(0.0)=10, fn(1.0)=0 -> target=100 is above range, best is lo
        best_p, best_f, _ = _binary_search_percentile(fn, 0.0, 1.0, target=100)
        assert best_p == pytest.approx(0.0)

    def test_target_below_range_picks_hi(self):
        fn = self._monotone_fn(slope=-10, intercept=10)
        # fn(1.0)=0 -> target=-5 is below range, best is hi
        best_p, best_f, _ = _binary_search_percentile(fn, 0.0, 1.0, target=-5)
        assert best_p == pytest.approx(1.0)

    def test_n_evaluations_bounded(self):
        fn = self._monotone_fn()
        _, _, n_evals = _binary_search_percentile(fn, 0.0, 1.0, target=25, max_iter=5)
        assert n_evals <= 7  # 2 anchors + max_iter

    def test_convergence_within_tolerance(self):
        fn = self._monotone_fn(slope=-100, intercept=100)
        # target = 50 -> p = 0.5 exactly
        best_p, best_f, _ = _binary_search_percentile(
            fn, 0.0, 1.0, target=50, tolerance=0.001, max_iter=20
        )
        assert abs(best_p - 0.5) < 0.01

    def test_call_count_memoised_externally(self):
        """Verify evaluate_fn is not called more than max_iter+2 times."""
        call_count = {"n": 0}

        def counting_fn(p):
            call_count["n"] += 1
            return 100 - 50 * p

        _binary_search_percentile(counting_fn, 0.0, 1.0, target=50, max_iter=8)
        assert call_count["n"] <= 10
