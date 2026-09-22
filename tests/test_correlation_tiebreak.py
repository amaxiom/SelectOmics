"""
Which member of a correlated pair survives.

_apply_correlation_filter drops one member of every redundant pair, and the
choice runs down a ladder: class association first, then variance, then column
name. The name rung exists so the outcome never depends on the order columns
happen to arrive in, which is what makes the same data give the same panel
twice.

The distinction matters beyond determinism. Resolving a pair by variance keeps
whichever member happens to vary more, which has nothing to do with the
outcome, so a different sample can keep a different member. Resolving it by
class association keeps the one that tracks the label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from SelectOmics.selection.step1_cleaning import _apply_correlation_filter


def _inputs(X):
    """Absolute correlation matrix and variances, as the filter expects."""
    return X.corr().abs(), X.var(axis=0)


def _kept(X, priority=None, thresh=0.9):
    corr, var = _inputs(X)
    mask = _apply_correlation_filter(X, thresh, var, corr, priority)
    return set(X.columns[mask])


# ---------------------------------------------------------------------------
# The class-association rung
# ---------------------------------------------------------------------------

class TestPriorityDecidesFirst:

    @staticmethod
    def _pair(seed=0, n=60):
        """Two correlated columns; 'strong' tracks the label more closely."""
        rng = np.random.RandomState(seed)
        y = np.array([i % 2 for i in range(n)])
        strong = 2.0 * y + rng.randn(n) * 0.2
        weak = strong + rng.randn(n) * 0.25      # correlated, noisier
        return pd.DataFrame({"strong": strong, "weak": weak}), y

    def test_the_higher_priority_member_survives(self):
        X, y = self._pair()
        priority = pd.Series({"strong": 0.9, "weak": 0.2})
        assert _kept(X, priority) == {"strong"}

    def test_the_choice_reverses_with_the_priority(self):
        """
        Proves the priority is actually consulted rather than the column order
        deciding and the priority merely agreeing with it.
        """
        X, y = self._pair()
        priority = pd.Series({"strong": 0.1, "weak": 0.8})
        assert _kept(X, priority) == {"weak"}


# ---------------------------------------------------------------------------
# The variance rung
# ---------------------------------------------------------------------------

class TestVarianceBreaksPriorityTies:

    @staticmethod
    def _scaled_pair(n=60, seed=0):
        """
        Identical up to scale: class association is scale-invariant, so their
        priority ties exactly and only variance separates them.
        """
        rng = np.random.RandomState(seed)
        y = np.array([i % 2 for i in range(n)])
        base = 2.0 * y + rng.randn(n) * 0.3
        return pd.DataFrame({"small": base, "large": base * 5.0}), y

    def test_the_higher_variance_member_survives_an_exact_tie(self):
        X, y = self._scaled_pair()
        priority = pd.Series({"small": 0.5, "large": 0.5})   # exact tie
        assert _kept(X, priority) == {"large"}

    def test_the_choice_follows_variance_when_it_is_reversed(self):
        X, y = self._scaled_pair()
        X = X.rename(columns={"small": "large", "large": "small"})
        priority = pd.Series({"small": 0.5, "large": 0.5})
        # 'small' now holds the 5x-scaled column, so it is the one kept.
        assert _kept(X, priority) == {"small"}


# ---------------------------------------------------------------------------
# The name rung
# ---------------------------------------------------------------------------

class TestNameBreaksRemainingTies:

    @staticmethod
    def _identical_pair(n=60, seed=0):
        rng = np.random.RandomState(seed)
        y = np.array([i % 2 for i in range(n)])
        base = 2.0 * y + rng.randn(n) * 0.3
        return pd.DataFrame({"aaa": base.copy(), "bbb": base.copy()}), y

    def test_an_exact_tie_in_both_falls_back_to_the_name(self):
        """
        Identical columns tie on priority and on variance. Something still has
        to decide, and the name is the only input left that does not depend on
        column ordering.
        """
        X, y = self._identical_pair()
        priority = pd.Series({"aaa": 0.5, "bbb": 0.5})
        kept = _kept(X, priority)
        assert len(kept) == 1

    def test_column_order_does_not_change_the_survivor(self):
        """
        The property the name rung exists for: reversing the column order must
        not change which feature is kept.
        """
        X, y = self._identical_pair()
        priority = pd.Series({"aaa": 0.5, "bbb": 0.5})
        forward = _kept(X, priority)
        reversed_ = _kept(X[["bbb", "aaa"]], priority)
        assert forward == reversed_, (
            f"column order changed the survivor: {forward} vs {reversed_}"
        )

    def test_repeated_runs_agree(self):
        X, y = self._identical_pair()
        priority = pd.Series({"aaa": 0.5, "bbb": 0.5})
        assert len({frozenset(_kept(X, priority)) for _ in range(5)}) == 1


# ---------------------------------------------------------------------------
# Behaviour with no priority supplied
# ---------------------------------------------------------------------------

class TestWithoutPriority:

    def test_variance_is_used_when_no_priority_is_given(self):
        """
        Falling back to variance is documented as the weaker choice, but it
        still has to be deterministic.
        """
        rng = np.random.RandomState(0)
        n = 60
        base = rng.randn(n)
        X = pd.DataFrame({"small": base, "large": base * 4.0})
        assert _kept(X) == {"large"}

    def test_uncorrelated_columns_are_all_kept(self):
        rng = np.random.RandomState(0)
        X = pd.DataFrame(rng.randn(60, 5),
                         columns=[f"f{i}" for i in range(5)])
        assert _kept(X) == set(X.columns)

    def test_a_chain_of_correlated_features_collapses_to_one(self):
        """
        Three mutually correlated columns carry one signal, and the greedy pass
        must not leave two of them behind.
        """
        rng = np.random.RandomState(0)
        n = 60
        base = rng.randn(n)
        X = pd.DataFrame({
            "a": base,
            "b": base + rng.randn(n) * 0.01,
            "c": base + rng.randn(n) * 0.01,
        })
        assert len(_kept(X)) == 1
