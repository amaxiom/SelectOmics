"""
Helper edge cases and environment probes.

These functions sit under everything else: the AUC helper every fold calls, the
label attached to a consensus level, and the hardware probes that decide
whether a run uses a GPU or how many cores it claims. They are written to
degrade rather than raise, because a failure here would abort a run over
something incidental, so each test drives a failure and checks the degraded
value.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import pytest

from SelectOmics.utils.helpers import (
    agreement_label,
    compute_macro_auc_ovr,
    detect_cpu_count,
    detect_gpu,
    ensure_binary_proba,
    print_adequacy_warning,
    relax_consensus_intersection,
    _warn_step_adequacy,
)


# ---------------------------------------------------------------------------
# compute_macro_auc_ovr
# ---------------------------------------------------------------------------

class TestMacroAucOvr:

    def test_a_dataframe_of_probabilities_is_accepted(self):
        """Some sklearn versions hand back a DataFrame rather than an array."""
        rng = np.random.RandomState(0)
        y = np.array([0, 1] * 20)
        proba = pd.DataFrame(rng.rand(40, 2), columns=["a", "b"])
        auc = compute_macro_auc_ovr(y, proba, classes=np.array([0, 1]))
        assert 0.0 <= auc <= 1.0

    def test_a_single_class_falls_back_to_chance(self):
        """
        With one class present no per-class slice has two labels, so nothing is
        scorable. 0.5 records "no information", which is what a caller
        averaging over folds needs.
        """
        y = np.zeros(20, dtype=int)
        proba = np.column_stack([np.ones(20) * 0.6, np.ones(20) * 0.4])
        assert compute_macro_auc_ovr(y, proba, classes=np.array([0, 1])) == 0.5

    def test_an_unscorable_class_is_skipped_not_fatal(self):
        """
        A probability column that makes roc_auc_score raise must cost that one
        class, not the whole score.
        """
        y = np.array([0, 1, 2] * 10)
        proba = np.full((30, 3), np.nan)
        proba[:, 0] = np.linspace(0, 1, 30)
        auc = compute_macro_auc_ovr(y, proba, classes=np.array([0, 1, 2]))
        assert 0.0 <= auc <= 1.0

    def test_a_scorable_multiclass_case_beats_chance(self):
        y = np.array([0, 1, 2] * 10)
        proba = np.zeros((30, 3))
        proba[np.arange(30), y] = 1.0
        auc = compute_macro_auc_ovr(y, proba, classes=np.array([0, 1, 2]))
        assert auc == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ensure_binary_proba
# ---------------------------------------------------------------------------

class TestEnsureBinaryProba:

    def test_a_one_dimensional_score_becomes_two_columns(self):
        out = ensure_binary_proba(np.linspace(0, 1, 10), n_classes=2)
        assert out.shape == (10, 2)
        assert np.allclose(out.sum(axis=1), 1.0)

    def test_an_already_two_column_matrix_is_left_alone(self):
        proba = np.column_stack([np.linspace(1, 0, 10), np.linspace(0, 1, 10)])
        out = ensure_binary_proba(proba, n_classes=2)
        assert out.shape == (10, 2)


# ---------------------------------------------------------------------------
# agreement_label
# ---------------------------------------------------------------------------

class TestAgreementLabel:

    @pytest.mark.parametrize("agreement,expected", [
        (1.00, "unanimous"),
        (0.85, "strong"),
        (0.65, "moderate"),
        (0.45, "weak"),
        (0.10, "minimal"),
        (0.00, "minimal"),
    ])
    def test_each_band(self, agreement, expected):
        assert agreement_label(agreement) == expected

    def test_a_ranked_outcome_says_it_was_not_agreed(self):
        """
        A ranked panel carries no agreement at all. Labelling it with a band
        would overstate what the run established.
        """
        assert agreement_label(0.0, how="rank_average") == \
            "none (ranked, not agreed)"
        assert agreement_label(1.0, how="rank_average") == \
            "none (ranked, not agreed)"

    def test_a_negative_agreement_still_lands_in_a_band(self):
        """Defensive: the loop must not fall through to an undefined label."""
        assert agreement_label(-0.5) == "minimal"


# ---------------------------------------------------------------------------
# relax_consensus_intersection: the ceiling rung
# ---------------------------------------------------------------------------

class TestConsensusCeiling:
    """
    The second argument is n_models, not n_features. Each vote array holds a
    per-feature COUNT of how many of the n_models replicates voted for it
    within one stage, and a feature survives only where every stage agrees.
    """

    def test_unanimity_is_reported_as_consensus(self):
        """Landing on the first rung means the panel held at full agreement."""
        votes = [np.array([2, 2, 2, 0, 0]), np.array([2, 2, 2, 0, 0])]
        res = relax_consensus_intersection(votes, n_models=2, min_features=3)
        assert res.mask.sum() == 3
        assert res.how == "consensus"
        assert res.votes_required == 2

    def test_falling_short_at_unanimity_relaxes_one_vote(self):
        """
        Only two features are unanimous, so a floor of three forces the search
        down a rung, and that has to be reported as relaxed rather than passed
        off as full agreement.
        """
        votes = [np.array([2, 2, 1, 1, 0]), np.array([2, 2, 2, 1, 0])]
        res = relax_consensus_intersection(votes, n_models=2, min_features=3)
        assert res.how == "relaxed"
        assert res.votes_required < 2
        assert res.mask.sum() >= 3

    def test_a_floor_above_the_ceiling_cannot_invent_features(self):
        """
        Nothing can be selected that some stage gave no vote at all. A floor
        above that one-vote ceiling is unmeetable, and the result is capped
        there rather than padded.
        """
        votes = [np.array([2, 2, 0, 0, 0, 0, 0, 0, 0, 0]),
                 np.array([2, 1, 0, 0, 0, 0, 0, 0, 0, 0])]
        res = relax_consensus_intersection(votes, n_models=2, min_features=9)
        assert res.mask.sum() <= 2, (
            "selected a feature that a stage never voted for"
        )

    def test_the_union_rung_can_exceed_the_intersection_ceiling(self):
        """
        allow_union replaces the intersection ceiling with the union one, so it
        can only ever return at least as much as the intersection did.
        """
        votes = [np.array([2, 2, 0, 0, 0]), np.array([0, 0, 2, 2, 0])]
        tight = relax_consensus_intersection(votes, n_models=2, min_features=4)
        loose = relax_consensus_intersection(votes, n_models=2, min_features=4,
                                             allow_union=True)
        assert loose.mask.sum() >= tight.mask.sum()


# ---------------------------------------------------------------------------
# Hardware probes
# ---------------------------------------------------------------------------

class TestHardwareProbes:

    def test_gpu_detection_answers_rather_than_raising(self):
        """
        Runs on machines with no CUDA, an old xgboost, or no xgboost at all.
        Any of those must be a False, not an exception.
        """
        assert detect_gpu() in (True, False)

    def test_cpu_count_is_at_least_one(self):
        assert detect_cpu_count() >= 1

    def test_cpu_count_falls_back_when_the_modern_api_is_absent(self,
                                                                monkeypatch):
        """
        os.process_cpu_count arrived in 3.13. On older interpreters the
        affinity mask is consulted instead, and that path is unreachable here
        without removing the newer function.
        """
        monkeypatch.delattr(os, "process_cpu_count", raising=False)
        assert detect_cpu_count() >= 1

    def test_cpu_count_degrades_to_one_when_the_environment_refuses(
        self, monkeypatch
    ):
        """
        Some restricted containers will not report a core count. Running
        serially beats failing to start.
        """
        def boom():
            raise OSError("no core count available")

        monkeypatch.setattr(os, "process_cpu_count", boom, raising=False)
        monkeypatch.setattr(os, "cpu_count", boom)
        monkeypatch.delattr(os, "sched_getaffinity", raising=False)
        assert detect_cpu_count() == 1


# ---------------------------------------------------------------------------
# Sample-size adequacy warnings
# ---------------------------------------------------------------------------

class TestAdequacyWarnings:

    def test_severe_underdetermination_is_reported(self, caplog):
        caplog.set_level(logging.WARNING, logger="SelectOmics")
        adequacy = {"n_samples": 20, "per_class_flag": "critical",
                    "overall_severity": "critical", "warnings": []}
        print_adequacy_warning("Step 1", n_features_current=5000,
                               adequacy=adequacy)
        assert "n/p" in caplog.text

    def test_the_high_dimensional_band_is_distinct(self, caplog):
        caplog.set_level(logging.WARNING, logger="SelectOmics")
        adequacy = {"n_samples": 80, "per_class_flag": "adequate",
                    "overall_severity": "moderate", "warnings": []}
        print_adequacy_warning("Step 2", n_features_current=120,
                               adequacy=adequacy)
        assert "high-dimensional" in caplog.text

    def test_no_adequacy_information_is_not_an_error(self):
        print_adequacy_warning("Step 1", 100, None)

    def test_the_deprecated_alias_still_forwards(self, caplog):
        caplog.set_level(logging.WARNING, logger="SelectOmics")
        adequacy = {"n_samples": 20, "per_class_flag": "critical",
                    "overall_severity": "critical", "warnings": []}
        _warn_step_adequacy("Step 1", 5000, adequacy, verbose=True)
        assert caplog.text.strip()

    def test_the_deprecated_alias_stays_silent_when_not_verbose(self, caplog):
        caplog.set_level(logging.WARNING, logger="SelectOmics")
        adequacy = {"n_samples": 20, "per_class_flag": "critical",
                    "overall_severity": "critical", "warnings": []}
        _warn_step_adequacy("Step 1", 5000, adequacy, verbose=False)
        assert caplog.text.strip() == ""


# ---------------------------------------------------------------------------
# enable_logging
# ---------------------------------------------------------------------------

class TestEnableLogging:
    """
    Public API that mutates the package logger. The conftest fixture restores
    it after each test, so these can configure it freely.
    """

    def test_a_new_handler_type_is_attached(self):
        import logging
        import SelectOmics

        pkg = logging.getLogger("SelectOmics")
        before = len(pkg.handlers)
        SelectOmics.enable_logging(logging.INFO,
                                   handler=logging.FileHandler(
                                       __import__("tempfile").mkstemp()[1]))
        assert len(pkg.handlers) > before, "the handler was never attached"

    def test_repeating_a_call_updates_rather_than_stacking(self):
        """
        Calling twice must not attach a second handler of the same type, or
        every message would be emitted twice per call.
        """
        import logging
        import SelectOmics

        pkg = logging.getLogger("SelectOmics")
        SelectOmics.enable_logging("WARNING")
        after_first = len(pkg.handlers)
        SelectOmics.enable_logging("DEBUG")
        assert len(pkg.handlers) == after_first, "a duplicate handler stacked"

    def test_the_level_actually_changes_on_a_repeat_call(self):
        """
        The second call has to update the existing handler's level too, or
        enable_logging('DEBUG') after enable_logging('WARNING') would raise the
        logger level while the handler kept filtering the messages out.
        """
        import logging
        import SelectOmics

        SelectOmics.enable_logging("WARNING")
        SelectOmics.enable_logging("DEBUG")
        pkg = logging.getLogger("SelectOmics")
        assert pkg.level == logging.DEBUG
        # The package attaches a NullHandler at import whose level is NOTSET,
        # so only the handler enable_logging manages is checked here.
        managed = [h for h in pkg.handlers
                   if not isinstance(h, logging.NullHandler)]
        assert managed, "enable_logging attached no handler"
        assert all(h.level == logging.DEBUG for h in managed), (
            "the logger level was raised but its handler kept filtering at the "
            "old level, so the extra messages would never be emitted"
        )
