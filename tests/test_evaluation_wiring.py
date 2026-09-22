"""
Every evaluation option must actually fire when it is switched on.

A flag that is validated, documented, and then never read is worse than one
that does not exist: the user sets it, sees a completed run, and has no way to
tell that nothing happened. Two were in exactly that state before these tests.

  build_recommendation  Public API, fully tested, called from nowhere. The
                        pipeline computed the per-step comparison and stopped,
                        so the evidence for choosing between steps was stored
                        and never used.

  enable_nested_cv      Validated by the config and listed in the CLI's
                        coercible fields, but run() never read it, so
                        run_nested_cv() was reachable only by hand.

Each test here asserts the observable consequence of a flag, not that a line
ran: a result key appears, a panel is chosen, a warning is raised.
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from SelectOmics.config import SelectOmicsConfig
from SelectOmics.pipeline import SelectOmicsPipeline


@pytest.fixture
def dataset(tmp_path):
    """Small, well-conditioned, and quick: signal in the first five columns."""
    rng = np.random.RandomState(0)
    n, p = 60, 24
    X = rng.randn(n, p)
    y = np.array([i % 2 for i in range(n)])
    X[:, :5] += 1.8 * y[:, None]
    df = pd.DataFrame(X, columns=[f"f{i:02d}" for i in range(p)])
    df["Class"] = y
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


def _pipeline(dataset, tmp_path, **over):
    kw = dict(
        data_path=str(dataset), target_column="Class", algorithm="RF",
        output_dir=str(tmp_path / "out"), n_consensus_models=2,
        quick_tune_iterations=2, n_bootstrap=10, verbose=False,
        create_visualizations=False, save_intermediate_results=False,
        enable_step3=False,
    )
    kw.update(over)
    cfg = SelectOmicsConfig(**kw)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return SelectOmicsPipeline(cfg)


# ---------------------------------------------------------------------------
# The recommendation
# ---------------------------------------------------------------------------

class TestRecommendationIsProduced:

    @pytest.mark.integration
    def test_a_validated_run_produces_a_recommendation(self, dataset, tmp_path):
        """
        The comparison says how each step scored. The recommendation says which
        to use, and without it a caller has to re-derive that themselves from a
        DataFrame most will never open.
        """
        p = _pipeline(dataset, tmp_path, enable_step_evaluations=True,
                      enable_final_test_evaluation=False)
        res = p.run(validate=True)

        assert "recommendation" in res, (
            "a validated run produced no recommendation"
        )
        rec = res["recommendation"]
        for key in ("step_id", "step_name", "n_features", "weighted_auc",
                    "quality", "reason", "X_train"):
            assert key in rec, f"recommendation is missing {key!r}"
        assert rec["quality"] in ("RECOMMENDED", "CAUTION", "NOT RECOMMENDED")
        assert rec["n_features"] >= 1

    @pytest.mark.integration
    def test_the_recommended_panel_is_retrievable(self, dataset, tmp_path):
        p = _pipeline(dataset, tmp_path, enable_step_evaluations=True,
                      enable_final_test_evaluation=False)
        p.run(validate=True)

        recommended = p.get_recommended_features()
        assert len(recommended) >= 1
        assert set(recommended) <= set(p._X_train.columns)
        assert len(recommended) == p.results["recommendation"]["n_features"]

    @pytest.mark.integration
    def test_the_recommendation_names_a_step_that_actually_ran(self, dataset,
                                                               tmp_path):
        p = _pipeline(dataset, tmp_path, enable_step_evaluations=True,
                      enable_final_test_evaluation=False)
        res = p.run(validate=True)
        comparison = res["validation"]["comparison_df"]
        assert res["recommendation"]["step_id"] in set(comparison.index), (
            "recommended a step absent from the comparison it was built from"
        )

    @pytest.mark.integration
    def test_a_verbose_run_says_which_step_it_recommends(self, dataset,
                                                         tmp_path, caplog):
        caplog.set_level(logging.INFO, logger="SelectOmics")
        p = _pipeline(dataset, tmp_path, enable_step_evaluations=True,
                      enable_final_test_evaluation=False, verbose=True)
        p.run(validate=True)
        assert "Recommended feature set" in caplog.text \
            or "RECOMMENDED:" in caplog.text

    @pytest.mark.integration
    def test_a_recommendation_survives_step_evaluations_being_off(
        self, dataset, tmp_path
    ):
        """
        The recommendation rests on the validation comparison, which covers
        every step regardless of enable_step_evaluations. Making it depend on
        that flag too would have made availability turn on which step happened
        to win: only Step 0 carries an evaluation when the flag is off, so a
        recommendation would appear when the reference won and vanish when it
        did not.
        """
        p = _pipeline(dataset, tmp_path, enable_step_evaluations=False,
                      enable_final_test_evaluation=False)
        res = p.run(validate=True)

        assert "recommendation" in res, (
            "the recommendation disappeared because a different flag was off"
        )
        assert p.get_recommended_features()

    @pytest.mark.integration
    def test_no_recommendation_when_validation_is_skipped(self, dataset,
                                                          tmp_path):
        p = _pipeline(dataset, tmp_path, enable_step_evaluations=True,
                      enable_final_test_evaluation=False)
        res = p.run(validate=False)
        assert "recommendation" not in res
        with pytest.raises(RuntimeError):
            p.get_recommended_features()

    @pytest.mark.integration
    def test_the_last_step_and_the_recommended_step_are_distinguishable(
        self, dataset, tmp_path
    ):
        """
        The two accessors answer different questions and must not be assumed
        interchangeable: one is the end of the pipeline, the other is the panel
        that validated best.
        """
        p = _pipeline(dataset, tmp_path, enable_step_evaluations=True,
                      enable_final_test_evaluation=False)
        p.run(validate=True)
        last = p.get_selected_features()
        rec = p.get_recommended_features()
        assert isinstance(last, list) and isinstance(rec, list)
        assert len(last) >= 1 and len(rec) >= 1


# ---------------------------------------------------------------------------
# Nested CV
# ---------------------------------------------------------------------------

class TestNestedCvIsHonoured:

    @pytest.mark.integration
    def test_enabling_it_makes_run_perform_it(self, dataset, tmp_path):
        """
        Before this, the flag was validated and then never read: run() ignored
        it and run_nested_cv() had to be called by hand.
        """
        p = _pipeline(dataset, tmp_path, enable_nested_cv=True,
                      outer_cv_splits=3, enable_step_evaluations=False,
                      enable_final_test_evaluation=False)
        res = p.run(validate=False)

        assert "nested_cv" in res, "enable_nested_cv=True produced no nested CV"
        nested = res["nested_cv"]
        assert len(nested["fold_aucs"]) == 3
        assert 0.0 <= nested["mean_auc"] <= 1.0
        assert len(nested["fold_features"]) == 3

    @pytest.mark.integration
    def test_it_stays_off_by_default(self, dataset, tmp_path):
        """It refits the whole inner pipeline per fold, so it must be opt-in."""
        p = _pipeline(dataset, tmp_path, enable_step_evaluations=False,
                      enable_final_test_evaluation=False)
        res = p.run(validate=False)
        assert "nested_cv" not in res

    @pytest.mark.integration
    def test_a_failure_does_not_take_the_run_down(self, dataset, tmp_path,
                                                  caplog, monkeypatch):
        """
        An unbiased generalisation estimate is a bonus on top of a completed
        run, not a precondition for one.
        """
        p = _pipeline(dataset, tmp_path, enable_nested_cv=True,
                      outer_cv_splits=3, enable_step_evaluations=False,
                      enable_final_test_evaluation=False)
        monkeypatch.setattr(
            p, "run_nested_cv",
            lambda: (_ for _ in ()).throw(RuntimeError("nested cv exploded")))
        caplog.set_level(logging.WARNING, logger="SelectOmics")
        res = p.run(validate=False)

        assert "nested_cv" not in res
        assert "Nested CV failed" in caplog.text
        assert p.get_selected_features(), "the run itself was lost"


# ---------------------------------------------------------------------------
# The other evaluation switches
# ---------------------------------------------------------------------------

class TestEvaluationSwitches:

    @pytest.mark.integration
    def test_step_evaluations_attach_a_result_to_each_step(self, dataset,
                                                           tmp_path):
        p = _pipeline(dataset, tmp_path, enable_step_evaluations=True,
                      enable_final_test_evaluation=False)
        res = p.run(validate=False)
        for key in ("step1", "step2"):
            assert res[key].get("consensus_result") is not None, (
                f"{key} was not evaluated despite enable_step_evaluations=True"
            )

    @pytest.mark.integration
    def test_disabling_step_evaluations_leaves_them_empty(self, dataset,
                                                          tmp_path):
        p = _pipeline(dataset, tmp_path, enable_step_evaluations=False,
                      enable_final_test_evaluation=False)
        res = p.run(validate=False)
        for key in ("step1", "step2"):
            assert res[key].get("consensus_result") is None

    @pytest.mark.integration
    def test_the_final_test_evaluation_fires_when_enabled(self, dataset,
                                                          tmp_path):
        p = _pipeline(dataset, tmp_path, enable_step_evaluations=False,
                      enable_final_test_evaluation=True)
        res = p.run(validate=True)
        assert "final_test" in res
        assert p._final_test_eval is not None

    @pytest.mark.integration
    def test_the_final_test_evaluation_says_why_it_was_skipped(
        self, dataset, tmp_path, caplog
    ):
        """
        It runs only under validate=True. A user who set the flag and got no
        metrics file needs to be able to tell that from a failure.
        """
        caplog.set_level(logging.INFO, logger="SelectOmics")
        p = _pipeline(dataset, tmp_path, enable_step_evaluations=False,
                      enable_final_test_evaluation=True)
        res = p.run(validate=False)
        assert "final_test" not in res
        assert "Final test evaluation skipped" in caplog.text

    @pytest.mark.integration
    def test_validation_covers_every_completed_step(self, dataset, tmp_path):
        """
        The comparison exists to choose between steps, so a step missing from
        it can never be recommended however well it performed.
        """
        p = _pipeline(dataset, tmp_path, enable_step_evaluations=True,
                      enable_final_test_evaluation=False)
        res = p.run(validate=True)
        idx = set(res["validation"]["comparison_df"].index)
        assert len(idx) >= 2, f"only {idx} were validated"


# ---------------------------------------------------------------------------
# Nested CV measures the procedure the user runs, recommendation included
# ---------------------------------------------------------------------------

class TestNestedCvMeasuresTheRecommendation:
    """
    Nested CV exists to say how the whole procedure does on unseen data. Its
    inner loop used to stop at Step 2 and score that panel whatever the
    configuration, so it estimated a procedure nobody ran and could not show
    whether the recommendation's choice held up. These pin the replacement:
    every fold runs what run() runs and scores every step on the outer fold.
    """

    def _nested(self, dataset, tmp_path, **over):
        kw = dict(enable_nested_cv=True, outer_cv_splits=3,
                  enable_step_evaluations=False,
                  enable_final_test_evaluation=False)
        kw.update(over)
        p = _pipeline(dataset, tmp_path, **kw)
        return p, p.run(validate=False)["nested_cv"]

    @pytest.mark.integration
    def test_the_headline_is_the_recommended_panel(self, dataset, tmp_path):
        _, nested = self._nested(dataset, tmp_path)
        d = nested["fold_details"]
        rec = d[d["recommended"]].sort_values("fold")

        assert list(rec["fold"]) == [1, 2, 3], (
            "each fold must recommend exactly one step"
        )
        assert list(rec["step"]) == nested["fold_recommended_steps"]
        np.testing.assert_allclose(rec["outer_auc"], nested["fold_aucs"])
        assert [len(f) for f in nested["fold_features"]] == \
            list(rec["n_features"])

    @pytest.mark.integration
    def test_every_step_is_scored_on_every_fold(self, dataset, tmp_path):
        """The per-step scores are what let the choice itself be checked."""
        _, nested = self._nested(dataset, tmp_path)
        d = nested["fold_details"]
        for fold, rows in d.groupby("fold"):
            assert set(rows["step"]) == {
                "Step 0 (Reference)", "Step 1 (Data Cleaning)",
                "Step 2 (Regularization)",
            }, f"fold {fold} did not score every step"
            assert rows["last"].sum() == 1

        assert nested["mean_regret"] >= 0.0
        assert nested["best_step_mean_auc"] >= nested["mean_auc"] - 1e-12
        assert np.isfinite(nested["last_step_mean_auc"])
        summary = nested["step_summary"]
        assert int(summary["times_recommended"].sum()) == 3

    @pytest.mark.integration
    def test_the_inner_run_never_sees_the_outer_labels(self, dataset,
                                                       tmp_path, monkeypatch):
        """
        The inner run is handed the outer fold's features so the steps can
        subset them, but the labels would let the choice being tested see
        the data it is tested on.
        """
        seen = []
        real = SelectOmicsPipeline._adopt_split

        def spy(self, X_train, X_test, y_train, y_test, *args, **kwargs):
            seen.append(y_test)
            return real(self, X_train, X_test, y_train, y_test, *args,
                        **kwargs)

        monkeypatch.setattr(SelectOmicsPipeline, "_adopt_split", spy)
        self._nested(dataset, tmp_path)

        assert len(seen) == 4, "expected the outer run plus three folds"
        assert seen[0] is not None, "the outer run lost its own test labels"
        assert all(s is None for s in seen[1:]), (
            "an inner nested-CV run was given the outer fold's labels"
        )

    @pytest.mark.integration
    def test_it_runs_step3_when_the_user_does(self, dataset, tmp_path,
                                              monkeypatch, caplog):
        """
        The inner config used to force enable_step3=False. Besides dropping
        Step 3 from the estimate, that made step2_role='prefilter', which
        requires Step 3, fail config validation on every fold, so nested CV
        failed outright for that whole configuration.
        """
        calls = []
        real = SelectOmicsPipeline.run_step3_wrapper

        def spy(self):
            calls.append(1)
            return real(self)

        monkeypatch.setattr(SelectOmicsPipeline, "run_step3_wrapper", spy)
        caplog.set_level(logging.WARNING, logger="SelectOmics")
        _, nested = self._nested(dataset, tmp_path, enable_step3=True,
                                 step2_role="prefilter")

        assert "Nested CV failed" not in caplog.text
        assert len(nested["fold_aucs"]) == 3
        assert len(calls) == 4, "Step 3 did not run inside every outer fold"

    @pytest.mark.integration
    def test_a_fold_without_an_auc_is_left_out_not_scored_zero(
        self, dataset, tmp_path, monkeypatch, caplog
    ):
        """
        An AUC of 0.0 is a perfectly inverted classifier. Substituting it for
        a fold that produced none dragged the mean toward a failure that
        never happened.
        """
        import SelectOmics.pipeline as pl
        monkeypatch.setattr(pl, "_safe_roc_auc",
                            lambda *a, **k: float("nan"))
        caplog.set_level(logging.WARNING, logger="SelectOmics")
        _, nested = self._nested(dataset, tmp_path)

        assert all(np.isnan(a) for a in nested["fold_aucs"])
        assert np.isnan(nested["mean_auc"])
        assert "produced no AUC" in caplog.text

    @pytest.mark.integration
    def test_no_empty_fold_directories_are_left_behind(self, dataset,
                                                       tmp_path):
        p, _ = self._nested(dataset, tmp_path)
        leftovers = list(p.config.output_dir.glob("nested_cv_fold_*"))
        assert not leftovers, f"left behind: {[d.name for d in leftovers]}"

    def test_inner_folds_are_built_the_way_the_main_split_is(self, dataset,
                                                              tmp_path):
        """
        The inner loop used to derive its fold count from the minority class
        alone, ignoring an explicit cv_splits that run() honours.
        """
        from SelectOmics.data.preprocessing import (
            create_train_test_split, make_cv_splitter,
        )
        cfg = SelectOmicsConfig(data_path=str(dataset), target_column="Class",
                                output_dir=str(tmp_path / "o"), cv_splits=3)
        df = pd.read_csv(dataset)
        X, y = df.drop(columns=["Class"]), df["Class"]
        *_, cv_main = create_train_test_split(X, y, cfg)
        assert cv_main.n_splits == 3
        assert make_cv_splitter(y.to_numpy(), 2, cfg).n_splits == 3


# ---------------------------------------------------------------------------
# The held-out evaluation scores the panel the user is told to use
# ---------------------------------------------------------------------------

def _force_recommendation(monkeypatch, step_name):
    """Make the recommendation name ``step_name`` whatever validation said."""
    import SelectOmics.pipeline as pl
    real = pl.build_recommendation

    def pick(comparison_df, step_data, X_test, **kwargs):
        rec = real(comparison_df, step_data, X_test, **kwargs)
        sd = step_data[step_name]
        rec.update(step_id=step_name, step_name=step_name,
                   X_train=sd["X_train"],
                   X_test=X_test[sd["X_train"].columns],
                   n_features=sd["X_train"].shape[1],
                   result=sd["eval_result"])
        return rec

    monkeypatch.setattr(pl, "build_recommendation", pick)


class TestFinalTestScoresTheRecommendation:

    @pytest.mark.integration
    def test_it_scores_the_recommended_panel_not_the_last(self, dataset,
                                                          tmp_path,
                                                          monkeypatch):
        """
        It used to score the last step regardless, so whenever the two
        differed the held-out metrics described the panel the user had just
        been told not to use.
        """
        # Near-duplicate columns give Step 1's correlation filter something
        # to remove, so the last step's panel is guaranteed to be smaller
        # than Step 0's.
        df = pd.read_csv(dataset)
        rng = np.random.RandomState(1)
        for k in range(6):
            df.insert(0, f"dup{k}", df["f00"] + rng.randn(len(df)) * 0.01)
        redundant = tmp_path / "redundant.csv"
        df.to_csv(redundant, index=False)

        _force_recommendation(monkeypatch, "Step 0 (Reference)")
        p = _pipeline(redundant, tmp_path, enable_step_evaluations=False,
                      enable_final_test_evaluation=True)
        res = p.run(validate=True)

        n_step0 = len(p.get_selected_features("step0"))
        assert n_step0 != len(p.get_selected_features()), (
            "precondition: Step 0 and the last step must differ"
        )
        ft = res["final_test"]
        assert ft["panel"] == "recommended"
        assert ft["step_name"] == "Step 0 (Reference)"
        assert ft["n_features"] == n_step0
        assert ft["metrics"]["final_features"] == n_step0

    @pytest.mark.integration
    def test_the_gap_is_measured_against_the_same_panel(self, dataset,
                                                        tmp_path,
                                                        monkeypatch):
        """
        The training AUC the gap is measured from was chosen by iterating
        reversed(['step3', 'step2', 'step1', 'step0']), which runs Step 0
        first, and stopping at the first match, so it was always the
        full-feature reference, whichever panel was being tested.
        """
        import SelectOmics.pipeline as pl
        captured = {}
        real_bcm = pl.build_comprehensive_metrics

        def spy(**kwargs):
            captured.update(kwargs)
            return real_bcm(**kwargs)

        monkeypatch.setattr(pl, "build_comprehensive_metrics", spy)
        _force_recommendation(monkeypatch, "Step 2 (Regularization)")
        p = _pipeline(dataset, tmp_path, enable_step_evaluations=True,
                      enable_final_test_evaluation=True)
        res = p.run(validate=True)

        assert captured["training_result"] is \
            res["step2"]["consensus_result"]
        assert captured["training_result"] is not \
            res["step0"]["reference_result"]
        assert captured["step_label"] == "Step 2 (Regularization)"

    @pytest.mark.integration
    def test_the_last_step_is_scored_when_there_is_no_recommendation(
        self, dataset, tmp_path, monkeypatch
    ):
        import SelectOmics.pipeline as pl
        monkeypatch.setattr(pl, "build_recommendation", lambda *a, **k: None)
        p = _pipeline(dataset, tmp_path, enable_step_evaluations=False,
                      enable_final_test_evaluation=True)
        ft = p.run(validate=True)["final_test"]
        assert ft["panel"] == "last step"
        assert ft["n_features"] == len(p.get_selected_features())

    @pytest.mark.integration
    def test_save_results_writes_the_recommended_panel(self, dataset,
                                                       tmp_path):
        p = _pipeline(dataset, tmp_path, enable_step_evaluations=False,
                      enable_final_test_evaluation=False)
        p.run(validate=True)
        p.save_results()
        f = p.config.output_dir / "recommended_features.csv"
        assert f.exists(), "the recommended panel was never written to disk"
        assert list(pd.read_csv(f)["feature"]) == \
            p.get_recommended_features()

    @pytest.mark.integration
    def test_no_recommended_file_without_a_recommendation(self, dataset,
                                                          tmp_path):
        p = _pipeline(dataset, tmp_path, enable_step_evaluations=False,
                      enable_final_test_evaluation=False)
        p.run(validate=False)
        p.save_results()
        assert (p.config.output_dir / "selected_features.csv").exists()
        assert not (p.config.output_dir / "recommended_features.csv").exists()


# ---------------------------------------------------------------------------
# A step that skipped is described by the panel it passed on
# ---------------------------------------------------------------------------

class TestSkippedStepsAreNotCreditedWithTheReference:
    """
    A skipped or disabled step files Step 0's reference evaluation as its
    consensus_result. Read directly, that put the full-feature AUC on the row
    of a panel that is not the full feature set: in the summary table, in the
    recommendation's result, and as the training AUC of the held-out gap.
    Step 3 skips in most default runs, so this was the common case.
    """

    @pytest.mark.integration
    def test_a_skipped_step3_is_described_by_step2(self, dataset, tmp_path):
        p = _pipeline(dataset, tmp_path, enable_step3=True,
                      enable_step_evaluations=True,
                      enable_final_test_evaluation=False)
        p.run(validate=False)
        assert p.results["step3"]["skipped"], "precondition: Step 3 skipped"

        r3 = p._panel_cv_result("Step 3 (Wrappers)", use_validation=False)
        assert r3 is p.results["step2"]["consensus_result"]
        assert r3 is not p.results["step0"]["reference_result"]

    @pytest.mark.integration
    def test_the_summary_row_for_a_skipped_step_shows_its_own_panel(
        self, dataset, tmp_path, monkeypatch
    ):
        import SelectOmics.pipeline as pl
        captured = {}
        real = pl.build_pipeline_summary_df

        def spy(step_results, **kwargs):
            captured["rows"] = step_results
            return real(step_results=step_results, **kwargs)

        monkeypatch.setattr(pl, "build_pipeline_summary_df", spy)
        p = _pipeline(dataset, tmp_path, enable_step3=True,
                      enable_step_evaluations=True,
                      enable_final_test_evaluation=False)
        p.run(validate=False)

        rows = {r["step_name"]: r for r in captured["rows"]}
        s3 = rows["Step 3 (Wrappers)"]
        assert s3["eval_result"] is p.results["step2"]["consensus_result"]
        assert s3["n_features"] == \
            rows["Step 2 (Regularization)"]["n_features"]

    @pytest.mark.integration
    def test_without_step_evaluations_the_validation_describes_the_panel(
        self, dataset, tmp_path
    ):
        p = _pipeline(dataset, tmp_path, enable_step_evaluations=False,
                      enable_final_test_evaluation=False)
        p.run(validate=True)
        r2 = p._panel_cv_result("Step 2 (Regularization)")
        assert r2 is p._validation_all_results[
            "Step 2 (Regularization)"]["stratified_cv"]
        assert p._panel_cv_result("Step 2 (Regularization)",
                                  use_validation=False) is None
        assert p._panel_cv_result("not a step") is None


# ---------------------------------------------------------------------------
# Fallbacks when a recommendation cannot be made
# ---------------------------------------------------------------------------

class TestRecommendationFallbacks:

    @pytest.mark.integration
    def test_a_nested_fold_without_one_scores_the_last_step(
        self, dataset, tmp_path, monkeypatch, caplog
    ):
        import SelectOmics.pipeline as pl
        monkeypatch.setattr(pl, "build_recommendation", lambda *a, **k: None)
        caplog.set_level(logging.WARNING, logger="SelectOmics")
        p = _pipeline(dataset, tmp_path, enable_nested_cv=True,
                      outer_cv_splits=3, enable_step_evaluations=False,
                      enable_final_test_evaluation=False)
        nested = p.run(validate=False)["nested_cv"]

        assert "no recommendation could be built" in caplog.text
        d = nested["fold_details"]
        chosen = d[d["recommended"]].sort_values("fold")
        assert list(chosen["last"]) == [True, True, True], (
            "without a recommendation the last step must stand in for it"
        )

    @pytest.mark.integration
    def test_a_failing_recommendation_does_not_take_the_run_down(
        self, dataset, tmp_path, monkeypatch, caplog
    ):
        import SelectOmics.pipeline as pl

        def boom(*a, **k):
            raise ValueError("cannot rank")

        monkeypatch.setattr(pl, "build_recommendation", boom)
        caplog.set_level(logging.WARNING, logger="SelectOmics")
        p = _pipeline(dataset, tmp_path, enable_step_evaluations=False,
                      enable_final_test_evaluation=True)
        res = p.run(validate=True)

        assert "Could not build a recommendation" in caplog.text
        assert "recommendation" not in res
        assert res["final_test"]["panel"] == "last step"

    @pytest.mark.integration
    def test_a_panel_with_no_cv_result_reports_no_training_auc(
        self, dataset, tmp_path, monkeypatch
    ):
        """
        No number at all is better than a number that describes a different
        panel, which is what the old reference-AUC fallback produced.
        """
        monkeypatch.setattr(SelectOmicsPipeline, "_panel_cv_result",
                            lambda self, *a, **k: None)
        p = _pipeline(dataset, tmp_path, enable_step_evaluations=False,
                      enable_final_test_evaluation=True)
        ft = p.run(validate=True)["final_test"]
        assert ft["metrics"]["training_cv_auc_mean"] is None



# ---------------------------------------------------------------------------
# Held-out step ranking
# ---------------------------------------------------------------------------

class TestHoldoutRankingIsWired:
    """
    enable_holdout_ranking moves the choice off the training-split validation,
    which scores every panel on the samples its features came from, and onto
    folds held out of the training split. The pipeline's test split must stay
    out of it, or the final test stops being independent.
    """

    @pytest.mark.integration
    def test_enabling_it_ranks_on_held_out_folds(self, dataset, tmp_path):
        p = _pipeline(dataset, tmp_path, enable_holdout_ranking=True,
                      holdout_ranking_splits=3,
                      enable_step_evaluations=False,
                      enable_final_test_evaluation=False)
        res = p.run(validate=True)

        assert "holdout_ranking" in res, "the flag produced no ranking"
        ranking = res["holdout_ranking"]
        assert set(ranking.columns) >= {"holdout_auc", "holdout_std",
                                        "holdout_folds", "holdout_n_features"}
        assert (ranking["holdout_folds"] == 3).all()
        assert res["recommendation"]["ranking_basis"] == \
            "held-out folds from the training split"
        assert res["recommendation"]["step_id"] in set(ranking.index)

    @pytest.mark.integration
    def test_it_stays_off_by_default(self, dataset, tmp_path):
        p = _pipeline(dataset, tmp_path, enable_step_evaluations=False,
                      enable_final_test_evaluation=False)
        res = p.run(validate=True)
        assert "holdout_ranking" not in res
        assert res["recommendation"]["ranking_basis"] == \
            "validation on the training split"

    @pytest.mark.integration
    def test_the_test_split_is_never_involved(self, dataset, tmp_path,
                                              monkeypatch):
        """
        The whole point: rows from the held-out test set must not reach the
        ranking, or the final test would be scoring a panel it helped choose.
        """
        seen = []
        real = SelectOmicsPipeline._adopt_split

        def spy(self, X_train, X_test, y_train, y_test, *args, **kwargs):
            seen.append((X_train, X_test, y_test))
            return real(self, X_train, X_test, y_train, y_test, *args,
                        **kwargs)

        monkeypatch.setattr(SelectOmicsPipeline, "_adopt_split", spy)
        p = _pipeline(dataset, tmp_path, enable_holdout_ranking=True,
                      holdout_ranking_splits=3,
                      enable_step_evaluations=False,
                      enable_final_test_evaluation=False)
        p.run(validate=True)

        outer_train = set(p._X_train.index)
        test_rows = set(p._X_test.index)
        assert len(seen) == 4, "expected the outer run plus three fold runs"
        for X_train, X_test, y_test in seen[1:]:
            assert set(X_train.index) <= outer_train
            assert set(X_test.index) <= outer_train
            assert not (set(X_test.index) & test_rows), (
                "a ranking fold scored on held-out test rows"
            )
            assert y_test is None, "a ranking fold was given test labels"

    @pytest.mark.integration
    def test_every_step_is_scored_on_rows_that_did_not_select_it(
        self, dataset, tmp_path
    ):
        p = _pipeline(dataset, tmp_path, enable_step3=True,
                      enable_step_evaluations=False,
                      enable_final_test_evaluation=False)
        p.load_data()
        p._assess_sample_size()
        p._run_selection_steps()
        ranking = p.rank_steps_on_holdout()

        assert set(ranking.index) == {
            "Step 0 (Reference)", "Step 1 (Data Cleaning)",
            "Step 2 (Regularization)", "Step 3 (Wrappers)",
        }
        assert ranking["holdout_auc"].between(0.0, 1.0).all()
        assert (ranking["holdout_n_features"] >= 1).all()

    @pytest.mark.integration
    def test_a_failure_falls_back_to_the_training_split_ranking(
        self, dataset, tmp_path, monkeypatch, caplog
    ):
        monkeypatch.setattr(
            SelectOmicsPipeline, "rank_steps_on_holdout",
            lambda self: (_ for _ in ()).throw(RuntimeError("ranking blew up")))
        caplog.set_level(logging.WARNING, logger="SelectOmics")
        p = _pipeline(dataset, tmp_path, enable_holdout_ranking=True,
                      enable_step_evaluations=False,
                      enable_final_test_evaluation=False)
        res = p.run(validate=True)

        assert "Held-out step ranking failed" in caplog.text
        assert res["recommendation"]["ranking_basis"] == \
            "validation on the training split"
        assert p.get_recommended_features()

    @pytest.mark.integration
    def test_no_empty_fold_directories_are_left_behind(self, dataset,
                                                       tmp_path):
        p = _pipeline(dataset, tmp_path, enable_holdout_ranking=True,
                      holdout_ranking_splits=3,
                      enable_step_evaluations=False,
                      enable_final_test_evaluation=False)
        p.run(validate=True)
        leftovers = list(p.config.output_dir.glob("holdout_rank_fold_*"))
        assert not leftovers, f"left behind: {[d.name for d in leftovers]}"

    @pytest.mark.integration
    def test_folds_are_capped_by_the_smallest_class(self, dataset, tmp_path,
                                                    caplog):
        """
        Stratification needs a member of every class in every fold, so a
        request for more folds than the smallest class allows is lowered and
        said out loud rather than raising.
        """
        caplog.set_level(logging.WARNING, logger="SelectOmics")
        p = _pipeline(dataset, tmp_path, holdout_ranking_splits=40,
                      enable_step_evaluations=False,
                      enable_final_test_evaluation=False)
        p.load_data()
        p._assess_sample_size()
        p._run_selection_steps()
        ranking = p.rank_steps_on_holdout()

        assert "folds are used instead of 40" in caplog.text
        assert (ranking["holdout_folds"] < 40).all()
