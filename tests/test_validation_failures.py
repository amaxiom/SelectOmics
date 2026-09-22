"""
Validation degradation paths and the final recommendation.

The three validation protocols each skip work they cannot do rather than
aborting: a fold missing a class, a bootstrap draw with no out-of-bag samples,
a model that will not fit. build_recommendation then reduces the whole
comparison to one verdict, and it has to refuse rather than guess when the
inputs cannot support one.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from SelectOmics.config import SelectOmicsConfig
from SelectOmics.evaluation.validation import (
    FeatureSetValidator,
    build_recommendation,
)
from SelectOmics.models.base import quick_tune_all


def _data(n=40, p=5, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n, p), columns=[f"f{i}" for i in range(p)])
    y = np.array([i % 2 for i in range(n)])
    X.iloc[:, 0] += 1.5 * y
    return X, y


def _validator(tmp_path, X, y, n_bootstrap=10, verbose=False, tuned=True):
    cfg = SelectOmicsConfig(data_path="unused.csv", target_column="Class",
                            output_dir=str(tmp_path), algorithm="RF",
                            quick_tune_iterations=2, verbose=verbose)
    pipes = quick_tune_all(X, y, cfg) if tuned else {}
    return FeatureSetValidator(cfg, n_classes=2, tuned_pipelines=pipes,
                               n_bootstrap=n_bootstrap)


# ---------------------------------------------------------------------------
# Leave-one-out
# ---------------------------------------------------------------------------

class TestLeaveOneOut:

    @pytest.mark.integration
    def test_a_verbose_run_reports_which_scheme_it_used(self, tmp_path, caplog):
        """
        LOO is swapped for a stratified shuffle above a sample threshold. The
        two give different numbers, so a reader needs to know which ran.
        """
        caplog.set_level(logging.DEBUG, logger="SelectOmics")
        X, y = _data(n=24)
        v = _validator(tmp_path, X, y, verbose=True)
        v.leave_one_out_validation(X, y, "probe")
        assert "LOO" in caplog.text or "stratified shuffle" in caplog.text

    @pytest.mark.integration
    def test_folds_that_lose_a_class_are_skipped_not_scored(self, tmp_path):
        """
        A training split missing a class cannot fit a two-class model. Those
        folds are counted as failures rather than contributing a bogus score.
        """
        rng = np.random.RandomState(0)
        n = 24
        X = pd.DataFrame(rng.randn(n, 4), columns=[f"f{i}" for i in range(4)])
        # Two minority samples: leaving either out strands the training split.
        y = np.array([0] * (n - 2) + [1, 1])
        v = _validator(tmp_path, X, y)
        res = v.leave_one_out_validation(X, y, "probe")
        assert res["total_tests"] >= 1
        assert 0.0 <= res["mean_auc"] <= 1.0 or np.isnan(res["mean_auc"])

    @pytest.mark.integration
    def test_a_model_that_will_not_fit_is_counted_as_a_failure(self, tmp_path,
                                                               caplog):
        X, y = _data(n=24)
        v = _validator(tmp_path, X, y)
        v._create_model = lambda seed: (_ for _ in ()).throw(
            RuntimeError("refused"))
        caplog.set_level(logging.WARNING, logger="SelectOmics")
        res = v.leave_one_out_validation(X, y, "probe")
        assert res["mean_auc"] == 0.0 or np.isnan(res["mean_auc"])
        assert caplog.text.strip(), "a fully failed protocol said nothing"


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

class TestBootstrapDegradation:

    @pytest.mark.integration
    def test_an_unstratifiable_draw_falls_back_to_plain_resampling(self,
                                                                   tmp_path):
        """
        Stratified resampling raises when a class has a single sample. Falling
        back to unstratified beats losing the iteration.
        """
        rng = np.random.RandomState(0)
        n = 30
        X = pd.DataFrame(rng.randn(n, 4), columns=[f"f{i}" for i in range(4)])
        y = np.array([0] * (n - 1) + [1])
        v = _validator(tmp_path, X, y, n_bootstrap=5)
        res = v.bootstrap_validation(X, y, "probe")
        assert "successful_iterations" in res
        assert res["successful_iterations"] >= 0

    @pytest.mark.integration
    def test_a_healthy_bootstrap_reports_an_interval(self, tmp_path):
        X, y = _data(n=40)
        v = _validator(tmp_path, X, y, n_bootstrap=10)
        res = v.bootstrap_validation(X, y, "probe")
        lo, hi = res["confidence_interval"]
        assert lo <= hi


# ---------------------------------------------------------------------------
# build_recommendation
# ---------------------------------------------------------------------------

class TestBuildRecommendation:

    @staticmethod
    def _frame(cv, loo, boot, n_features=5, idx="Step 1"):
        return pd.DataFrame(
            {"cv_auc": [cv], "loo_auc": [loo], "bootstrap_auc": [boot],
             "cv_std": [0.02], "n_features": [n_features]},
            index=[idx],
        )

    @staticmethod
    def _step_data(X, idx="Step 1"):
        return {idx: {"X_train": X, "step_name": idx,
                      "eval_result": {"mean_auc": 0.85}}}

    def test_an_all_nan_comparison_refuses_to_recommend(self):
        """
        Every protocol returning NaN means nothing was measured. Picking a
        winner from that would be inventing a result.
        """
        df = self._frame(np.nan, np.nan, np.nan)
        X, _ = _data()
        assert build_recommendation(df, self._step_data(X), X) is None

    def test_a_strong_result_is_recommended(self):
        X, _ = _data()
        df = self._frame(0.90, 0.88, 0.89)
        rec = build_recommendation(df, self._step_data(X), X)
        assert rec is not None
        assert rec["quality"] == "RECOMMENDED"

    def test_a_middling_result_is_flagged_for_caution(self):
        """
        Between the bands the verdict is neither an endorsement nor a
        rejection, and saying so is the point of having three levels.
        """
        X, _ = _data()
        df = self._frame(0.75, 0.74, 0.76)
        rec = build_recommendation(df, self._step_data(X), X)
        assert rec is not None
        assert rec["quality"] == "CAUTION"
        assert "monitor" in rec["reason"].lower()

    def test_a_weak_result_is_not_recommended(self):
        X, _ = _data()
        df = self._frame(0.60, 0.58, 0.61)
        rec = build_recommendation(df, self._step_data(X), X)
        assert rec is not None
        assert rec["quality"] == "NOT RECOMMENDED"

    def test_a_winner_with_no_recorded_data_refuses(self):
        """
        The best row naming a step that was never stored cannot be turned into
        a recommendation carrying a feature matrix.
        """
        X, _ = _data()
        df = self._frame(0.90, 0.88, 0.89, idx="Step 9")
        assert build_recommendation(df, self._step_data(X), X) is None

    def test_a_winner_with_an_empty_record_refuses(self):
        X, _ = _data()
        df = self._frame(0.90, 0.88, 0.89)
        broken = {"Step 1": {"X_train": None, "step_name": "Step 1",
                             "eval_result": None}}
        assert build_recommendation(df, broken, X) is None

    def test_the_best_row_wins_on_the_weighted_score(self):
        """
        The verdict weights CV at 0.5 and the other two at 0.25 each, so the
        winner is not simply the best CV AUC.
        """
        X, _ = _data()
        df = pd.DataFrame(
            {"cv_auc": [0.86, 0.84], "loo_auc": [0.60, 0.88],
             "bootstrap_auc": [0.60, 0.88], "cv_std": [0.02, 0.02],
             "n_features": [5, 5]},
            index=["Step 1", "Step 2"],
        )
        step_data = {
            "Step 1": {"X_train": X, "step_name": "Step 1",
                       "eval_result": {"mean_auc": 0.86}},
            "Step 2": {"X_train": X, "step_name": "Step 2",
                       "eval_result": {"mean_auc": 0.84}},
        }
        rec = build_recommendation(df, step_data, X)
        assert rec is not None
        assert rec["step_id"] == "Step 2", (
            "the higher CV AUC won despite a much weaker weighted score"
        )


# ---------------------------------------------------------------------------
# Bootstrap draws that cannot be scored
# ---------------------------------------------------------------------------

class TestUnscorableBootstrapDraws:
    """
    Two draws are skipped before any model is built: one that leaves no
    out-of-bag samples, and one whose in-bag half lost a class. Both are
    astronomically unlikely by chance, so they are forced here.
    """

    def test_a_draw_with_no_out_of_bag_samples_is_skipped(self, tmp_path,
                                                          monkeypatch):
        """
        A bootstrap that happens to draw every sample leaves nothing to score
        against, so the iteration contributes nothing rather than scoring on
        the training data.
        """
        import SelectOmics.evaluation.validation as val
        X, y = _data(n=30)
        monkeypatch.setattr(val, "resample",
                            lambda arr, **kw: np.arange(len(X)))
        v = _validator(tmp_path, X, y, n_bootstrap=5)
        res = v.bootstrap_validation(X, y, "probe")
        assert res["successful_iterations"] == 0, (
            "an iteration with no out-of-bag samples produced a score"
        )

    def test_a_draw_missing_a_class_is_skipped(self, tmp_path, monkeypatch):
        """
        A single-class in-bag sample cannot fit a two-class model, so the draw
        is skipped rather than allowed to fail inside the fit.
        """
        import SelectOmics.evaluation.validation as val
        X, y = _data(n=30)
        first_class = np.flatnonzero(y == y[0])

        def one_class_draw(arr, **kw):
            return np.resize(first_class, len(X))

        monkeypatch.setattr(val, "resample", one_class_draw)
        v = _validator(tmp_path, X, y, n_bootstrap=5)
        res = v.bootstrap_validation(X, y, "probe")
        assert res["successful_iterations"] == 0


# ---------------------------------------------------------------------------
# Estimators that cannot report probabilities
# ---------------------------------------------------------------------------

class TestHardPredictionFallback:
    """
    cv_evaluate_model prefers predict_proba and falls back to one-hot encoding
    hard predictions when the estimator has none. The check is deliberately
    against the wrapped classifier rather than the Pipeline: sklearn's Pipeline
    always exposes predict_proba as an attribute even when the estimator it
    wraps does not, so hasattr on the Pipeline is always True and this branch
    would never run.
    """

    @staticmethod
    def _pipeline(with_proba):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.svm import LinearSVC
        clf = LogisticRegression(max_iter=500) if with_proba else LinearSVC()
        return Pipeline([("scaler", MinMaxScaler()), ("clf", clf)])

    def test_an_estimator_without_predict_proba_still_scores(self):
        from sklearn.model_selection import StratifiedKFold
        from SelectOmics.models.evaluation import cv_evaluate_model
        X, y = _data(n=60, p=6)
        res = cv_evaluate_model(
            self._pipeline(with_proba=False), X, y, StratifiedKFold(3),
            n_classes=2, classes=np.array([0, 1]), use_bootstrap=False)
        assert 0.0 <= res["mean_auc"] <= 1.0
        assert len(res["fold_aucs"]) == 3

    def test_the_probabilistic_path_still_works(self):
        from sklearn.model_selection import StratifiedKFold
        from SelectOmics.models.evaluation import cv_evaluate_model
        X, y = _data(n=60, p=6)
        res = cv_evaluate_model(
            self._pipeline(with_proba=True), X, y, StratifiedKFold(3),
            n_classes=2, classes=np.array([0, 1]), use_bootstrap=False)
        assert 0.0 <= res["mean_auc"] <= 1.0

    def test_the_one_hot_fallback_produces_valid_rows(self):
        """
        One-hot encoded hard predictions must still be a proper probability
        matrix, or the AUC computed from them is meaningless rather than
        merely coarse.
        """
        from sklearn.model_selection import StratifiedKFold
        from SelectOmics.models.evaluation import cv_evaluate_model
        X, y = _data(n=60, p=6)
        res = cv_evaluate_model(
            self._pipeline(with_proba=False), X, y, StratifiedKFold(3),
            n_classes=2, classes=np.array([0, 1]), use_bootstrap=False)
        proba = np.asarray(res["oof_proba"])
        assert proba.shape == (60, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)


class TestStratifiedResampleFallback:

    def test_a_refused_stratified_draw_falls_back_to_plain_resampling(
        self, tmp_path, monkeypatch
    ):
        """
        Stratified resampling is attempted first and plain resampling is the
        fallback. sklearn does not actually refuse on a single-member class,
        so the refusal is injected here; what matters is that the iteration
        still produces a score rather than being lost.
        """
        import SelectOmics.evaluation.validation as val
        from sklearn.utils import resample as real_resample

        def refuse_stratified(arr, **kw):
            if kw.pop("stratify", None) is not None:
                raise ValueError("stratification is not possible")
            return real_resample(arr, **kw)

        monkeypatch.setattr(val, "resample", refuse_stratified)
        X, y = _data(n=40)
        v = _validator(tmp_path, X, y, n_bootstrap=5)
        res = v.bootstrap_validation(X, y, "probe")
        assert res["successful_iterations"] > 0, (
            "the unstratified fallback produced no usable iterations"
        )


# ---------------------------------------------------------------------------
# The one-standard-error rule
# ---------------------------------------------------------------------------

class TestOneStandardErrorRule:
    """
    Ranking on weighted AUC alone prefers the least selective step, because a
    larger panel usually scores a shade higher. Measured on synthetic data with
    known ground truth, plain argmax swapped an 18-feature panel at F1 0.714
    for a 620-feature one at F1 0.032.

    The rule: among panels whose score is within one standard error of the
    best, take the smallest.
    """

    @staticmethod
    def _frame(rows):
        """rows: {name: (weighted_parts, n_features, cv_std, cv_folds)}."""
        data = {}
        for name, (cv, loo, boot, n_feat, cv_std, folds) in rows.items():
            data[name] = {"cv_auc": cv, "loo_auc": loo, "bootstrap_auc": boot,
                          "cv_std": cv_std, "cv_folds": folds,
                          "n_features": n_feat}
        return pd.DataFrame(data).T

    @staticmethod
    def _step_data(X, names):
        return {n: {"X_train": X, "step_name": n,
                    "eval_result": {"mean_auc": 0.85}} for n in names}

    def test_a_smaller_panel_inside_the_band_wins(self):
        """
        The large panel scores marginally higher, but well inside one standard
        error. Paying 600 features for that difference is the failure this
        rule exists to prevent.
        """
        X, _ = _data()
        df = self._frame({
            "Step 1": (0.905, 0.905, 0.905, 620, 0.06, 5),
            "Step 2": (0.900, 0.900, 0.900, 18,  0.06, 5),
        })
        rec = build_recommendation(df, self._step_data(X, df.index), X)
        assert rec is not None
        assert rec["step_id"] == "Step 2", (
            f"took the 620-feature panel for a 0.005 AUC gain "
            f"(picked {rec['step_id']})"
        )
        assert rec["n_features"] == 18

    def test_a_materially_better_larger_panel_still_wins(self):
        """
        The rule must not simply always pick the smallest: a difference far
        outside the band is real and should be honoured.
        """
        X, _ = _data()
        df = self._frame({
            "Step 1": (0.95, 0.95, 0.95, 400, 0.01, 5),
            "Step 2": (0.60, 0.60, 0.60, 10,  0.01, 5),
        })
        rec = build_recommendation(df, self._step_data(X, df.index), X)
        assert rec["step_id"] == "Step 1"
        assert rec["n_features"] == 400

    def test_the_reason_says_when_parsimony_decided(self):
        """
        A reader comparing the verdict against the comparison table sees a
        panel that did not top it, and needs to know that was deliberate.
        """
        X, _ = _data()
        df = self._frame({
            "Step 1": (0.905, 0.905, 0.905, 620, 0.06, 5),
            "Step 2": (0.900, 0.900, 0.900, 18,  0.06, 5),
        })
        rec = build_recommendation(df, self._step_data(X, df.index), X)
        assert "one-standard-error" in rec["reason"]
        assert "620" in rec["reason"] and "18" in rec["reason"]

    def test_a_zero_tolerance_reverts_to_plain_argmax(self):
        """The rule is tunable, and switching it off restores the old choice."""
        X, _ = _data()
        df = self._frame({
            "Step 1": (0.905, 0.905, 0.905, 620, 0.06, 5),
            "Step 2": (0.900, 0.900, 0.900, 18,  0.06, 5),
        })
        rec = build_recommendation(df, self._step_data(X, df.index), X,
                                   one_se_tolerance=0.0)
        assert rec["step_id"] == "Step 1"

    def test_an_exact_tie_takes_the_smaller_panel(self):
        X, _ = _data()
        df = self._frame({
            "Step 1": (0.90, 0.90, 0.90, 500, 0.02, 5),
            "Step 2": (0.90, 0.90, 0.90, 12,  0.02, 5),
        })
        rec = build_recommendation(df, self._step_data(X, df.index), X)
        assert rec["n_features"] == 12

    def test_a_missing_fold_count_still_produces_a_choice(self):
        """
        cv_folds only arrived with this rule, so a comparison built elsewhere
        may not carry it. The tolerance then falls back to the raw standard
        deviation rather than failing.
        """
        X, _ = _data()
        df = self._frame({
            "Step 1": (0.905, 0.905, 0.905, 620, 0.06, 5),
            "Step 2": (0.900, 0.900, 0.900, 18,  0.06, 5),
        }).drop(columns=["cv_folds"])
        rec = build_recommendation(df, self._step_data(X, df.index), X)
        assert rec is not None
        assert rec["n_features"] == 18

    def test_a_saturated_metric_still_leaves_a_usable_band(self):
        """
        The case that broke the first version of this rule.

        At n << p every panel saturates: cv_auc 1.0000 and cv_std 0.0000 for
        18 features and for the full 2000. A variance-derived tolerance is then
        exactly zero, the band admits only the top row, and the rule silently
        reverts to argmax precisely where it is needed. Step 1 won by 0.0003 --
        noise -- and returned 620 features at F1 0.032 over 18 at F1 0.714.

        MIN_SCORE_TOLERANCE floors the band so this cannot recur.
        """
        from SelectOmics.evaluation.validation import MIN_SCORE_TOLERANCE

        X, _ = _data()
        c = pd.DataFrame({
            "n_features": [2000, 620, 18, 18],
            "cv_auc": [1.0, 1.0, 1.0, 1.0],
            "cv_std": [0.0, 0.0, 0.0, 0.0],          # saturated: no variance
            "cv_folds": [10, 10, 10, 10],
            "loo_auc": [0.9944, 1.0, 1.0, 1.0],
            "bootstrap_auc": [0.9881, 0.9863, 0.9854, 0.9854],
        }, index=["Step 0", "Step 1", "Step 2", "Step 3"])
        step_data = {n: {"X_train": X, "step_name": n,
                         "eval_result": {"mean_auc": 0.99}} for n in c.index}

        rec = build_recommendation(c, step_data, X)
        assert rec is not None
        assert rec["n_features"] == 18, (
            f"a zero-variance band took the {rec['n_features']}-feature panel "
            f"for a 0.0003 AUC difference"
        )
        assert MIN_SCORE_TOLERANCE > 0

    def test_the_floor_does_not_swamp_a_real_difference(self):
        """
        The floor must not make the rule always pick the smallest: a gap far
        wider than MIN_SCORE_TOLERANCE is real and has to be honoured.
        """
        X, _ = _data()
        c = pd.DataFrame({
            "n_features": [400, 10],
            "cv_auc": [0.95, 0.60], "cv_std": [0.0, 0.0], "cv_folds": [10, 10],
            "loo_auc": [0.95, 0.60], "bootstrap_auc": [0.95, 0.60],
        }, index=["Step 1", "Step 2"])
        step_data = {n: {"X_train": X, "step_name": n,
                         "eval_result": {"mean_auc": 0.9}} for n in c.index}
        rec = build_recommendation(c, step_data, X)
        assert rec["n_features"] == 400



class TestSelectionOptimismGuard:
    """
    Validation cross-validates each panel on the samples its features were
    selected from, so a selected panel's score is optimistic. On a null
    control the recommended panel scored 0.84 against 0.54 for the full
    feature set, while its held-out AUC was 0.46, and the quality label passed
    it. The least-selected panel carries no selection, so a gain over it
    beyond MAX_PLAUSIBLE_SELECTION_GAIN is flagged rather than trusted.
    """

    @staticmethod
    def _frame(ref_score, sel_score, sel_std=0.02, n_sel=20):
        return pd.DataFrame({
            "n_features":    [2000, n_sel],
            "cv_auc":        [ref_score, sel_score],
            "cv_std":        [0.05, sel_std],
            "cv_folds":      [5, 5],
            "loo_auc":       [ref_score, sel_score],
            "bootstrap_auc": [ref_score, sel_score],
        }, index=["Step 0", "Step 2"])

    @staticmethod
    def _rec(df):
        X, _ = _data()
        step_data = {n: {"X_train": X, "step_name": n,
                         "eval_result": {"mean_auc": 0.8}} for n in df.index}
        return build_recommendation(df, step_data, X)

    def test_the_null_control_pattern_is_downgraded(self):
        """The measured null: 0.84 selected against 0.54 unselected."""
        rec = self._rec(self._frame(ref_score=0.54, sel_score=0.84))
        assert rec["step_id"] == "Step 2", "the guard must not change the choice"
        assert rec["selection_optimism_suspected"] is True
        assert rec["selection_gain"] == pytest.approx(0.30)
        assert rec["quality"] == "CAUTION", (
            "a panel scoring 0.30 above the unselected reference was passed "
            "as RECOMMENDED"
        )
        assert "enable_nested_cv" in rec["reason"]
        assert "suitable for downstream tasks" not in rec["reason"]

    def test_a_plausible_gain_is_left_alone(self):
        """Real cohorts gained at most 0.021 over the reference."""
        rec = self._rec(self._frame(ref_score=0.87, sel_score=0.89))
        assert rec["selection_optimism_suspected"] is False
        assert rec["quality"] == "RECOMMENDED"
        assert rec["selection_gain"] == pytest.approx(0.02)

    def test_the_threshold_separates_flagged_from_trusted(self):
        from SelectOmics.evaluation.validation import (
            MAX_PLAUSIBLE_SELECTION_GAIN as G,
        )
        under = self._rec(self._frame(ref_score=0.85, sel_score=0.85 + G - 1e-6))
        over = self._rec(self._frame(ref_score=0.85, sel_score=0.85 + G + 1e-6))
        assert under["selection_optimism_suspected"] is False
        assert over["selection_optimism_suspected"] is True

    def test_recommending_the_reference_itself_is_never_flagged(self):
        df = pd.DataFrame({
            "n_features": [2000, 20], "cv_auc": [0.95, 0.60],
            "cv_std": [0.01, 0.01], "cv_folds": [5, 5],
            "loo_auc": [0.95, 0.60], "bootstrap_auc": [0.95, 0.60],
        }, index=["Step 0", "Step 2"])
        rec = self._rec(df)
        assert rec["step_id"] == "Step 0"
        assert rec["selection_gain"] == 0.0
        assert rec["selection_optimism_suspected"] is False

    def test_a_missing_reference_score_disables_the_guard(self):
        df = self._frame(ref_score=0.54, sel_score=0.84)
        df.loc["Step 0", ["cv_auc", "loo_auc", "bootstrap_auc"]] = np.nan
        rec = self._rec(df)
        assert rec["selection_optimism_suspected"] is False
        assert np.isnan(rec["selection_gain"])

    def test_not_recommended_stays_not_recommended_and_explains(self):
        """The guard only ever lowers confidence; it adds the reason."""
        rec = self._rec(self._frame(ref_score=0.50, sel_score=0.66))
        assert rec["quality"] == "NOT RECOMMENDED"
        assert rec["selection_optimism_suspected"] is True
        assert "fewer selection steps" in rec["reason"]
        assert "least-selected panel" in rec["reason"]

    def test_a_parsimony_explanation_survives_the_downgrade(self):
        """
        Three panels: the 1-SE rule picks the smallest over a marginally
        better mid-size one, and both are far above the reference.
        """
        X, _ = _data()
        df = pd.DataFrame({
            "n_features": [2000, 60, 10],
            "cv_auc": [0.55, 0.905, 0.90], "cv_std": [0.05, 0.06, 0.06],
            "cv_folds": [5, 5, 5], "loo_auc": [0.55, 0.905, 0.90],
            "bootstrap_auc": [0.55, 0.905, 0.90],
        }, index=["Step 0", "Step 2", "Step 3"])
        step_data = {n: {"X_train": X, "step_name": n,
                         "eval_result": {"mean_auc": 0.8}} for n in df.index}
        rec = build_recommendation(df, step_data, X)
        assert rec["step_id"] == "Step 3"
        assert rec["selection_optimism_suspected"] is True
        assert "one-standard-error" in rec["reason"]
        assert "least-selected panel" in rec["reason"]



class TestHoldoutRanking:
    """
    Ranking on validation is ranking on the samples that did the selecting.
    Held-out folds from the training split do not have that problem, and when
    they are supplied the ranking, the band and the quality label all come
    from them. The test split is not involved either way.
    """

    @staticmethod
    def _frame():
        # Validation likes the small panel (it was selected on these rows);
        # the big panel is genuinely better on rows that selected neither.
        return pd.DataFrame({
            "n_features": [2000, 20],
            "cv_auc": [0.60, 0.88], "cv_std": [0.02, 0.02],
            "cv_folds": [5, 5],
            "loo_auc": [0.60, 0.88], "bootstrap_auc": [0.60, 0.88],
        }, index=["Step 0", "Step 2"])

    @staticmethod
    def _holdout(step0, step2, std=0.01, folds=3):
        return pd.DataFrame({
            "holdout_auc": [step0, step2],
            "holdout_std": [std, std],
            "holdout_folds": [folds, folds],
            "holdout_n_features": [2000.0, 20.0],
        }, index=["Step 0", "Step 2"])

    def _rec(self, df, holdout=None):
        X, _ = _data()
        step_data = {n: {"X_train": X, "step_name": n,
                         "eval_result": {"mean_auc": 0.8}} for n in df.index}
        return build_recommendation(df, step_data, X, holdout_scores=holdout)

    def test_held_out_scores_decide_the_ranking(self):
        """
        Validation puts Step 2 at 0.88 against 0.60. On rows that selected
        neither, Step 0 is far better, and that is the honest ordering.
        """
        rec = self._rec(self._frame(), self._holdout(0.90, 0.60))
        assert rec["step_id"] == "Step 0"
        assert rec["ranking_basis"] == "held-out folds from the training split"
        assert rec["ranking_score"] == pytest.approx(0.90)
        assert rec["holdout_auc"] == pytest.approx(0.90)

    def test_without_them_the_training_split_still_decides(self):
        rec = self._rec(self._frame())
        assert rec["step_id"] == "Step 2"
        assert rec["ranking_basis"] == "validation on the training split"
        assert np.isnan(rec["holdout_auc"])

    def test_parsimony_still_applies_within_the_held_out_band(self):
        """A near-tie on honest scores still takes the smaller panel."""
        rec = self._rec(self._frame(), self._holdout(0.902, 0.900))
        assert rec["step_id"] == "Step 2"
        assert rec["n_features"] == 20

    def test_the_quality_label_reads_the_held_out_score(self):
        """
        The null-control pattern: validation 0.88, held out 0.46. Labelling
        from validation called that CAUTION; from the held-out score it is
        NOT RECOMMENDED.
        """
        rec = self._rec(self._frame(), self._holdout(0.50, 0.46))
        assert rec["quality"] == "NOT RECOMMENDED"

    def test_the_optimism_guard_is_off_for_held_out_ranking(self):
        """
        A large gain over the unselected panel is optimism when measured on
        the rows that selected it, and signal when measured on rows that did
        not. Flagging the second would be a false alarm.
        """
        rec = self._rec(self._frame(), self._holdout(0.55, 0.85))
        assert rec["step_id"] == "Step 2"
        assert rec["selection_gain"] == pytest.approx(0.30)
        assert rec["selection_optimism_suspected"] is False
        assert rec["quality"] == "RECOMMENDED"

    def test_all_nan_held_out_scores_fall_back_to_validation(self):
        rec = self._rec(self._frame(),
                        self._holdout(float("nan"), float("nan")))
        assert rec["ranking_basis"] == "validation on the training split"
        assert rec["step_id"] == "Step 2"

    def test_the_band_uses_the_standard_error_of_the_fold_means(self):
        """
        Wide fold-to-fold spread means a wide band, so the smaller panel is
        kept; a tight spread makes the same difference decisive.
        """
        wide = self._rec(self._frame(), self._holdout(0.90, 0.86, std=0.12))
        tight = self._rec(self._frame(), self._holdout(0.90, 0.86, std=0.001))
        assert wide["step_id"] == "Step 2"
        assert tight["step_id"] == "Step 0"
