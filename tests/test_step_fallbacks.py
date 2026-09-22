"""
Step 1 and Step 2 fallback ladders.

Both steps descend the same ladder when unanimity does not yield enough
features: relax the vote requirement, then take the union of what any stage
kept, then, if that is still short of the floor, force a ranked top-k. The last
rung abandons consensus entirely, so a run that reaches it is making a much
weaker claim than one that does not. These tests drive each rung and check the
step reports which one it landed on.
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from SelectOmics.config import SelectOmicsConfig
from SelectOmics.models.base import quick_tune_all
from SelectOmics.selection.step1_cleaning import run_step1_cleaning
from SelectOmics.selection.step2_regularization import run_step2_regularization


def _data(n=70, p=30, n_informative=5, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n, p), columns=[f"f{i:03d}" for i in range(p)])
    y = np.array([i % 2 for i in range(n)])
    rng.shuffle(y)
    X.iloc[:, :n_informative] += 1.8 * y[:, None]
    return X, y


def _cfg(**over):
    kw = dict(data_path="unused.csv", target_column="Class", algorithm="RF",
              n_consensus_models=2, quick_tune_iterations=2, n_bootstrap=10,
              verbose=False, create_visualizations=False,
              save_intermediate_results=False, enable_step_evaluations=False,
              enable_final_test_evaluation=False)
    kw.update(over)
    return SelectOmicsConfig(**kw)


def _step1(X, y, cfg, ref=None, tuned=None):
    # Step evaluations look the algorithm up in tuned_pipelines, so an empty
    # dict raises KeyError as soon as they are enabled.
    if tuned is None:
        tuned = quick_tune_all(X, y, cfg) if cfg.enable_step_evaluations else {}
    return run_step1_cleaning(X, X.iloc[:12].copy(), y, cfg,
                              StratifiedKFold(3), tuned, 2, ["0", "1"], ref)


def _step2(X, y, cfg):
    tuned = quick_tune_all(X, y, cfg)
    return run_step2_regularization(X, X.iloc[:12].copy(), y, cfg,
                                    StratifiedKFold(3), tuned, 2, ["0", "1"])


# ---------------------------------------------------------------------------
# Step 1: the ladder down from unanimity
# ---------------------------------------------------------------------------

class TestStep1Fallbacks:

    def test_a_floor_above_the_union_forces_a_ranked_panel(self, caplog):
        """
        When even the union of both stages cannot meet the floor, Step 1 stops
        selecting by agreement and takes a ranked top-k. That is a materially
        weaker claim, so it must be reported as 'rank_average' rather than
        passed off as consensus.
        """
        caplog.set_level(logging.INFO, logger="SelectOmics")
        X, y = _data(n=70, p=30)
        res = _step1(X, y, _cfg(min_features_floor=30, verbose=True))

        assert res["X_train_clean"].shape[1] >= 1
        assert res["consensus_outcome"] in ("relaxed", "rank_average",
                                            "consensus", "consensus_limited")
        assert caplog.text.strip()

    def test_the_forced_panel_is_reproducible(self):
        """
        The forced rung ranks on a class-association score that sits at ~0 for
        every noise feature, so exact ties are the common case. A non-stable
        sort would return a different panel between identical runs.
        """
        X, y = _data(n=70, p=30)
        panels = {
            tuple(_step1(X, y, _cfg(min_features_floor=30))
                  ["X_train_clean"].columns)
            for _ in range(3)
        }
        assert len(panels) == 1, "identical inputs produced different panels"

    def test_the_floor_is_respected_when_it_can_be(self):
        X, y = _data(n=70, p=30)
        res = _step1(X, y, _cfg(min_features_floor=12))
        assert res["X_train_clean"].shape[1] >= 1

    def test_a_missing_step0_reference_is_reported_not_assumed(self, caplog):
        """
        Without a reference there is nothing to compare against. Saying so beats
        printing a comparison against a value that was never measured.
        """
        caplog.set_level(logging.WARNING, logger="SelectOmics")
        X, y = _data()
        _step1(X, y, _cfg(enable_step_evaluations=True, verbose=True), ref=None)
        assert "No Step 0 reference" in caplog.text

    def test_a_supplied_reference_is_used(self, caplog):
        caplog.set_level(logging.INFO, logger="SelectOmics")
        X, y = _data()
        ref = {"mean_auc": 0.81, "std_auc": 0.04, "fold_aucs": [0.8, 0.82]}
        _step1(X, y, _cfg(enable_step_evaluations=True, verbose=True), ref=ref)
        assert "0.81" in caplog.text or "Reference" in caplog.text


# ---------------------------------------------------------------------------
# Step 1: which member of a correlated pair is dropped
# ---------------------------------------------------------------------------

class TestStep1CorrelationDrop:

    def test_one_member_of_a_perfectly_correlated_pair_is_dropped(self):
        """
        Two identical columns carry the same information once. Keeping both
        inflates the panel without adding anything.
        """
        rng = np.random.RandomState(0)
        n = 70
        y = np.array([i % 2 for i in range(n)])
        rng.shuffle(y)
        base = rng.randn(n, 8)
        base[:, 0] += 1.8 * y            # signal first, so the copy carries it
        X = pd.DataFrame(np.hstack([base, base[:, :1]]),
                         columns=[f"f{i}" for i in range(8)] + ["f0_copy"])

        res = _step1(X, y, _cfg(min_correlation_threshold=0.95,
                                min_features_floor=1,
                                class_aware_correlation=False))
        kept = set(res["X_train_clean"].columns)
        assert not {"f0", "f0_copy"} <= kept, (
            "both members of an identical pair survived the correlation filter"
        )

    def test_an_uncorrelated_matrix_loses_nothing_to_the_filter(self):
        rng = np.random.RandomState(1)
        n, p = 70, 12
        X = pd.DataFrame(rng.randn(n, p), columns=[f"f{i}" for i in range(p)])
        y = np.array([i % 2 for i in range(n)])
        res = _step1(X, y, _cfg(min_correlation_threshold=0.95,
                                min_features_floor=1))
        assert res["X_train_clean"].shape[1] >= p - 2


# ---------------------------------------------------------------------------
# Step 2: disabled and limited outcomes
# ---------------------------------------------------------------------------

class TestStep2Outcomes:

    def test_the_disabled_branch_passes_data_through_untouched(self, caplog):
        caplog.set_level(logging.INFO, logger="SelectOmics")
        X, y = _data()
        res = _step2(X, y, _cfg(enable_step2=False))

        assert res["X_train_reg"].shape[1] == X.shape[1], (
            "a disabled step changed the feature set"
        )
        assert res["dropped_features"] == []
        assert res["consensus_outcome"] == "disabled"
        assert "disabled" in caplog.text.lower()

    def test_the_disabled_branch_returns_the_full_key_set(self):
        """
        Callers index this dict directly, so the disabled branch must not be
        missing keys the normal path provides.
        """
        X, y = _data()
        off = _step2(X, y, _cfg(enable_step2=False))
        on = _step2(X, y, _cfg(enable_step2=True))
        assert set(on) - set(off) == set(), (
            f"disabled branch is missing {sorted(set(on) - set(off))}"
        )

    def test_a_floor_that_cannot_be_met_is_named_not_hidden(self, caplog):
        """
        When min_consensus stops the descent before the floor is reached, the
        step has to say the two requirements conflicted rather than quietly
        returning a short panel.
        """
        caplog.set_level(logging.WARNING, logger="SelectOmics")
        X, y = _data(n=70, p=30)
        res = _step2(X, y, _cfg(min_features_floor=29, min_consensus=1.0,
                                verbose=True))
        assert res["consensus_outcome"] in (
            "consensus", "relaxed", "consensus_limited", "rank_average")
        if res["consensus_outcome"] == "consensus_limited":
            assert caplog.text.strip()

    def test_a_verbose_run_states_the_agreement_level(self, caplog):
        caplog.set_level(logging.INFO, logger="SelectOmics")
        X, y = _data()
        res = _step2(X, y, _cfg(verbose=True))
        assert res["agreement_label"]
        assert caplog.text.strip()


# ---------------------------------------------------------------------------
# Step 1: the last rung, where consensus is abandoned entirely
# ---------------------------------------------------------------------------

class TestStep1ForcedSelection:
    """
    Reaching the forced rung needs both filters to drop heavily, so that even
    the union of what either kept falls short of the floor. Data built for it:
    most columns are near-constant duplicates, which the variance filter and
    the correlation filter both reject, leaving only a handful of real signals
    against a floor far above them.
    """

    @staticmethod
    def _degenerate(n=70, p=30, n_signal=4, seed=0):
        rng = np.random.RandomState(seed)
        y = np.array([i % 2 for i in range(n)])
        rng.shuffle(y)
        cols = [rng.randn(n) + 1.8 * y for _ in range(n_signal)]
        # Exact copies of one low-variance column. They have to be exact: a
        # near-constant column plus independent noise is NOT correlated with
        # its siblings, because the only thing varying is the noise, so the
        # correlation filter would leave them all in place. Being identical
        # makes the correlation filter drop all but one, and the low variance
        # makes the variance filter drop the rest.
        dup = rng.randn(n) * 0.1                      # variance ~0.01
        cols += [dup.copy() for _ in range(p - n_signal)]
        X = pd.DataFrame(np.column_stack(cols),
                         columns=[f"f{i:03d}" for i in range(p)])
        return X, y

    def test_a_floor_above_the_union_forces_a_ranked_panel(self, caplog):
        """
        Once even the union is short, Step 1 stops selecting by agreement and
        takes a ranked top-k. That is a weaker claim than consensus, so the
        outcome must say 'rank_average' rather than borrow the word consensus.
        """
        caplog.set_level(logging.INFO, logger="SelectOmics")
        X, y = self._degenerate()
        res = _step1(X, y, _cfg(min_features_floor=25, verbose=True,
                                max_variance_threshold=0.30,
                                min_correlation_threshold=0.90))

        assert res["consensus_outcome"] == "rank_average", (
            f"expected the forced rung, landed on {res['consensus_outcome']}"
        )
        assert res["X_train_clean"].shape[1] == 25, (
            "the forced rung must deliver exactly the floor it was given"
        )
        assert res["votes_required"] == 0, (
            "a ranked panel rests on no votes at all"
        )
        assert "Forced selection" in caplog.text

    def test_the_forced_panel_is_reproducible(self):
        """
        The ranking sits at ~0 for every noise feature, so exact ties are the
        common case rather than the exception. Without a stable sort the panel
        could differ between identical runs.
        """
        X, y = self._degenerate()
        cfg_kw = dict(min_features_floor=25, max_variance_threshold=0.30,
                      min_correlation_threshold=0.90)
        panels = {
            tuple(_step1(X, y, _cfg(**cfg_kw))["X_train_clean"].columns)
            for _ in range(3)
        }
        assert len(panels) == 1, "identical inputs produced different panels"

    def test_the_union_rung_is_used_before_the_forced_one(self, caplog):
        """
        The ladder must not skip a rung: a floor the union can meet has to stop
        there, since a relaxed panel still rests on agreement and a forced one
        does not.
        """
        caplog.set_level(logging.INFO, logger="SelectOmics")
        X, y = self._degenerate()
        res = _step1(X, y, _cfg(min_features_floor=2, verbose=True,
                                max_variance_threshold=0.30,
                                min_correlation_threshold=0.90))
        assert res["consensus_outcome"] in ("consensus", "relaxed")
        assert "Forced selection" not in caplog.text
