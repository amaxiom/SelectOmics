"""
Step 2 internals: stage construction, importance extraction, voting.

The regularization step builds one model per stage, reads an importance vector
out of whatever estimator that produced, and turns the two stages' votes into a
panel. Each of those has a branch for an input the happy path never supplies,
and getting them wrong misaligns the vote arrays rather than raising, so the
assertions here are about shape and alignment as much as values.
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
from SelectOmics.selection import step2_regularization as s2


def _data(n=70, p=20, n_informative=5, seed=0):
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


def _step2(X, y, cfg, ref=None):
    tuned = quick_tune_all(X, y, cfg)
    return s2.run_step2_regularization(X, X.iloc[:12].copy(), y, cfg,
                                       StratifiedKFold(3), tuned, 2,
                                       ["0", "1"], ref)


# ---------------------------------------------------------------------------
# Stage model construction
# ---------------------------------------------------------------------------

class TestStageModelConstruction:

    @pytest.mark.parametrize("algorithm", ["LR", "RF", "SVM"])
    def test_every_supported_algorithm_builds_a_stage_model(self, algorithm):
        model = s2._build_stage_model(algorithm, stage="l2", seed=0)
        assert model is not None
        assert "clf" in model.named_steps

    def test_an_unknown_algorithm_names_what_is_accepted(self):
        with pytest.raises(ValueError) as exc:
            s2._build_stage_model("RANDOMFOREST", stage="l2", seed=0)
        msg = str(exc.value)
        assert "RANDOMFOREST" in msg
        assert "'LR'" in msg and "'SVM'" in msg


# ---------------------------------------------------------------------------
# Importance extraction
# ---------------------------------------------------------------------------

class TestImportanceExtraction:

    @pytest.mark.parametrize("algorithm", ["LR", "RF", "SVM"])
    def test_each_algorithm_yields_one_importance_per_feature(self, algorithm):
        X, y = _data(n=50, p=9)
        model = s2._build_stage_model(algorithm, stage="l2", seed=0)
        model.fit(X, y)
        imp = s2._extract_importance(model, algorithm)
        assert imp.shape == (9,)
        assert np.all(np.isfinite(imp))
        assert imp.max() <= 1.0 + 1e-9, "importances are meant to be normalized"

    def test_an_unknown_algorithm_yields_zeros_of_the_right_width(self):
        """
        Degrades rather than raising, and the width is the part that matters:
        the vote arrays are positional, so a short vector would shift every
        feature's votes onto its neighbour instead of merely losing them.
        """
        X, y = _data(n=50, p=9)
        model = s2._build_stage_model("LR", stage="l2", seed=0)
        model.fit(X, y)
        imp = s2._extract_importance(model, "NOT_AN_ALGORITHM")
        assert imp.shape == (9,)
        assert not imp.any()


# ---------------------------------------------------------------------------
# Voting pass
# ---------------------------------------------------------------------------

class TestVotingPass:
    """
    Returns (selected, importances, votes) in that order, and takes a
    stage_key of 'l1' or 'l2' with a seed_offset that keeps the two stages'
    seeds from colliding.
    """

    @staticmethod
    def _run(X, y, n_models, stage_key="l1", seed_offset=0):
        return s2._run_voting_pass(
            X_train=X, y_train=y, algorithm="LR", stage_key=stage_key,
            best_percentile=0.5, n_models=n_models, base_seed=0,
            seed_offset=seed_offset, stage_label="L1")

    def test_an_unstratifiable_bootstrap_falls_back(self):
        """
        Stratified resampling raises when a class has a single member. The
        voting pass falls back to plain resampling rather than losing the
        model's vote entirely.
        """
        rng = np.random.RandomState(0)
        n, p = 24, 8
        X = pd.DataFrame(rng.randn(n, p), columns=[f"f{i}" for i in range(p)])
        y = np.array([0] * (n - 1) + [1])
        selected, importances, votes = self._run(X, y, n_models=2)
        assert votes.shape == (p,)
        assert selected.shape == (p,)
        assert len(importances) == 2

    def test_a_single_model_pass_casts_at_most_one_vote(self):
        """With one model there is no diversity to create, so no resampling."""
        X, y = _data(n=50, p=8)
        selected, importances, votes = self._run(X, y, n_models=1)
        assert votes.max() <= 1
        assert len(importances) == 1

    def test_votes_never_exceed_the_model_count(self):
        X, y = _data(n=60, p=10)
        selected, importances, votes = self._run(X, y, n_models=3)
        assert votes.max() <= 3, "a model voted more than once"
        assert len(importances) == 3
        assert all(i.shape == (10,) for i in importances), (
            "an importance vector was the wrong width, which would shift every "
            "feature's votes onto its neighbour"
        )

    def test_the_two_stages_do_not_share_seeds(self):
        """
        seed_offset separates L1 from L2. Sharing seeds would make the two
        stages fit identical models, and their agreement would be trivial.
        """
        X, y = _data(n=60, p=10)
        _, l1_imps, _ = self._run(X, y, n_models=2, stage_key="l1",
                                  seed_offset=0)
        _, l2_imps, _ = self._run(X, y, n_models=2, stage_key="l2",
                                  seed_offset=100)
        assert not np.array_equal(l1_imps[0], l2_imps[0]), (
            "both stages produced identical importances"
        )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

class TestStep2Reporting:

    @pytest.mark.integration
    def test_a_missing_step0_reference_is_reported(self, caplog):
        """
        Without a reference there is nothing to compare the step against, and
        saying so beats printing a comparison to a value never measured.
        """
        caplog.set_level(logging.WARNING, logger="SelectOmics")
        X, y = _data()
        _step2(X, y, _cfg(enable_step_evaluations=True, verbose=True), ref=None)
        assert "No Step 0 reference" in caplog.text

    @pytest.mark.integration
    def test_a_supplied_reference_is_reported(self, caplog):
        caplog.set_level(logging.INFO, logger="SelectOmics")
        X, y = _data()
        ref = {"mean_auc": 0.82, "std_auc": 0.03, "fold_aucs": [0.81, 0.83]}
        _step2(X, y, _cfg(enable_step_evaluations=True, verbose=True), ref=ref)
        assert "Reference" in caplog.text

    @pytest.mark.integration
    def test_a_capped_floor_explains_why_it_was_capped(self, caplog):
        """
        A floor above what any model selected cannot be met without padding the
        panel with features nothing chose. The step says so rather than
        silently returning fewer features than asked for.
        """
        caplog.set_level(logging.INFO, logger="SelectOmics")
        X, y = _data(n=70, p=20)
        res = _step2(X, y, _cfg(min_features_floor=20, verbose=True))
        assert res["consensus_floor_used"] <= res["consensus_floor_requested"]
        if res["consensus_floor_used"] < res["consensus_floor_requested"]:
            assert "capped" in caplog.text or "padding" in caplog.text
