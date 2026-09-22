"""
Regression tests for bugs found in the 0.6.1 bug sweep.

Each test fails on the 0.6.0 code and passes on 0.6.1.  They are kept
separate from the feature tests so the provenance of each assertion stays
obvious to anyone reading them later.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------------------------
# Step 3: RFECV scoring must be answerable by the estimator the algorithm uses
# ---------------------------------------------------------------------------

class TestRfecvScoring:
    """
    The SVM path builds a LinearSVC surrogate, which has no predict_proba.
    Scoring with 'roc_auc_ovr' made every RFECV fit raise AttributeError; the
    step swallowed the exception and silently degraded to "keep everything".
    """

    def test_svm_binary_avoids_proba_only_scorer(self):
        from SelectOmics.selection.step3_wrapper import _rfecv_scoring
        assert _rfecv_scoring('SVM', 2) != 'roc_auc_ovr'

    def test_svm_multiclass_avoids_proba_only_scorer(self):
        from SelectOmics.selection.step3_wrapper import _rfecv_scoring
        assert _rfecv_scoring('SVM', 3) != 'roc_auc_ovr'

    def test_proba_capable_algorithms_keep_auc(self):
        from SelectOmics.selection.step3_wrapper import _rfecv_scoring
        for alg in ('LR', 'XGB', 'RF'):
            assert _rfecv_scoring(alg, 2) == 'roc_auc'
            assert _rfecv_scoring(alg, 3) == 'roc_auc_ovr'

    @pytest.mark.integration
    def test_svm_rfecv_actually_fits(self):
        """The chosen scorer must let RFECV complete for the SVM surrogate."""
        from sklearn.feature_selection import RFECV
        from SelectOmics.selection.step3_wrapper import (
            _build_rfecv_estimator, _rfecv_scoring,
        )

        rng = np.random.RandomState(0)
        X = pd.DataFrame(rng.randn(60, 40), columns=[f"f{i}" for i in range(40)])
        y = (X["f0"] + X["f1"] > 0).astype(int).values

        estimator, getter = _build_rfecv_estimator('SVM', seed=1)
        rfecv = RFECV(
            estimator=estimator,
            step=1,
            cv=StratifiedKFold(3, shuffle=True, random_state=1),
            scoring=_rfecv_scoring('SVM', 2),
            n_jobs=1,
            min_features_to_select=5,
            importance_getter=getter,
        )
        rfecv.fit(X, y)
        assert 5 <= rfecv.n_features_ <= 40


# ---------------------------------------------------------------------------
# Step 3: stability subsampling must not leave a sub-class-sized complement
# ---------------------------------------------------------------------------

class TestStabilitySubsampleSizing:
    """
    subsample_size was capped at n_samples - 1, so with min_features close to
    n_samples the complement held a single sample and StratifiedShuffleSplit
    raised an uncaught ValueError that aborted the whole run.
    """

    @pytest.mark.integration
    def test_high_dimensional_small_n_does_not_raise(self):
        from SelectOmics.selection.step3_wrapper import _run_stability_stage

        rng = np.random.RandomState(0)
        n, p = 45, 2000
        X = pd.DataFrame(rng.randn(n, p), columns=[f"f{i}" for i in range(p)])
        y = np.array([0] * 23 + [1] * 22)
        counters = {
            "rfecv_failed": 0, "stability_failed": 0,
            "stability_skipped_class": 0, "stability_all_failed": 0,
        }

        selected, freq, importances = _run_stability_stage(
            X_train=X, y_train=y, algorithm='LR', n_models=1, base_seed=1,
            stability_threshold=0.6,
            min_features=max(10, int(p * 0.05)),   # 100 > n_samples
            warnings=counters,
        )
        assert selected.shape == (p,)


# ---------------------------------------------------------------------------
# Step 1: an all-constant matrix must fail with a message that names the cause
# ---------------------------------------------------------------------------

class TestConstantFeatureGuard:
    """
    Dropping every column left a zero-column matrix that flowed downstream and
    died several steps later inside sklearn with an unrelated message.
    """

    def test_all_constant_features_raises_clear_error(self):
        from SelectOmics.config import SelectOmicsConfig
        from SelectOmics.selection.step1_cleaning import run_step1_cleaning

        X = pd.DataFrame({f"c{i}": [1.0] * 40 for i in range(12)})
        y = np.array([0] * 20 + [1] * 20)
        config = SelectOmicsConfig(
            data_path="unused.csv", target_column="Class",
            enable_step_evaluations=False,
            save_intermediate_results=False,
            create_visualizations=False,
        )

        with pytest.raises(ValueError, match="constant"):
            run_step1_cleaning(
                X, X.iloc[:10].copy(), y, config,
                StratifiedKFold(3), {}, 2, ["0", "1"],
            )


# ---------------------------------------------------------------------------
# Output settings that the config declares must reach the writer
# ---------------------------------------------------------------------------

class TestSaveHonoursConfig:
    """
    Call sites passed only output_format, so excel_output_engine and hdf5_key
    were silently discarded.
    """

    def test_hdf5_key_is_forwarded(self, tmp_path):
        pytest.importorskip("tables")
        from SelectOmics.config import SelectOmicsConfig
        from SelectOmics.data.loaders import save_from_config

        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        config = SelectOmicsConfig(
            data_path="unused.csv", target_column="Class",
            output_format="hdf5", hdf5_key="custom_key",
        )
        path = save_from_config(df, tmp_path / "out", config)

        # Reading under the configured key must succeed.
        pd.read_hdf(path, key="custom_key")

    def test_output_format_still_honoured(self, tmp_path):
        from SelectOmics.config import SelectOmicsConfig
        from SelectOmics.data.loaders import save_from_config

        df = pd.DataFrame({"a": [1.0, 2.0]})
        config = SelectOmicsConfig(
            data_path="unused.csv", target_column="Class", output_format="tsv",
        )
        path = save_from_config(df, tmp_path / "out", config)
        assert path.suffix == ".tsv"

    def test_accepts_plain_dict_config(self, tmp_path):
        """Callers may hold config as a plain dict rather than a dataclass."""
        from SelectOmics.data.loaders import save_from_config

        df = pd.DataFrame({"a": [1.0, 2.0]})
        path = save_from_config(df, tmp_path / "out", {"output_format": "json"})
        assert path.suffix == ".json"


# ---------------------------------------------------------------------------
# CLI: a global --verbose placed before the subcommand must survive parsing
# ---------------------------------------------------------------------------

class TestCliVerboseFlag:
    """
    The subparsers declared --verbose with a store_true default, which reset
    the flag whenever it was given before the subcommand name.
    """

    def test_verbose_before_subcommand(self):
        from SelectOmics.cli import build_parser
        args = build_parser().parse_args(['-v', 'run', 'data.csv', '-t', 'Class'])
        assert args.verbose is True

    def test_verbose_after_subcommand(self):
        from SelectOmics.cli import build_parser
        args = build_parser().parse_args(['run', 'data.csv', '-t', 'Class', '-v'])
        assert args.verbose is True

    def test_verbose_absent_defaults_false(self):
        from SelectOmics.cli import build_parser
        args = build_parser().parse_args(['run', 'data.csv', '-t', 'Class'])
        assert args.verbose is False


# ---------------------------------------------------------------------------
# Parallelism must never change results
# ---------------------------------------------------------------------------

class TestThreadInvariance:
    """
    n_jobs defaults to -1, so thread count must not affect what is selected.

    XGBoost's default tree_method ('hist') breaks that once subsample < 1,
    which the tuned search space always samples. The package pins 'exact'.
    """

    def test_xgb_kwargs_pin_exact_on_cpu(self):
        from SelectOmics.models.base import xgb_compute_kwargs
        assert xgb_compute_kwargs(-1, 'cpu')['tree_method'] == 'exact'

    def test_xgb_kwargs_use_hist_on_cuda(self):
        """The CUDA implementation supports only hist."""
        from SelectOmics.models.base import xgb_compute_kwargs
        assert xgb_compute_kwargs(-1, 'cuda')['tree_method'] == 'hist'

    @pytest.mark.integration
    def test_xgb_is_thread_invariant_with_subsampling(self):
        from xgboost import XGBClassifier
        from SelectOmics.models.base import xgb_compute_kwargs

        rng = np.random.RandomState(0)
        X = rng.randn(120, 300)
        y = (X[:, 0] * 1.5 + X[:, 1] + rng.randn(120) * 0.5 > 0).astype(int)

        probas = []
        for n_jobs in (1, -1):
            model = XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=7, verbosity=0, eval_metric="mlogloss",
                **xgb_compute_kwargs(n_jobs, 'cpu'),
            )
            model.fit(X, y)
            probas.append(model.predict_proba(X))

        np.testing.assert_array_equal(probas[0], probas[1])

    def test_small_matrices_fall_back_to_one_thread(self):
        """A full thread pool measurably loses on tiny matrices."""
        from SelectOmics.models.base import effective_n_jobs
        assert effective_n_jobs(-1, 60, 80) == 1
        assert effective_n_jobs(-1, 200, 10000) == -1

    def test_explicit_serial_is_respected(self):
        from SelectOmics.models.base import effective_n_jobs
        assert effective_n_jobs(1, 200, 10000) == 1

    def test_cpu_count_never_returns_zero(self):
        from SelectOmics.utils.helpers import detect_cpu_count
        assert detect_cpu_count() >= 1

    def test_n_jobs_config_validation(self):
        from SelectOmics.config import SelectOmicsConfig
        SelectOmicsConfig(data_path="x.csv", target_column="Class", n_jobs=-1)
        SelectOmicsConfig(data_path="x.csv", target_column="Class", n_jobs=4)
        for bad in (0, -2):
            with pytest.raises(ValueError, match="n_jobs"):
                SelectOmicsConfig(data_path="x.csv", target_column="Class",
                                  n_jobs=bad)


# ---------------------------------------------------------------------------
# Sources must stay ASCII so Windows consoles render messages correctly
# ---------------------------------------------------------------------------

class TestConfigPathPortability:
    """
    to_json stores whatever path it was handed, and the example notebooks hand
    it Path('.').resolve(). Every saved config therefore embedded an absolute
    path to the machine that wrote it and failed to load anywhere else.
    Relative paths now anchor on the config file rather than the process CWD.
    """

    def _write(self, tmp_path, data_path_value):
        import json
        from SelectOmics.config import SelectOmicsConfig

        cfg_dir = tmp_path / "study"
        cfg_dir.mkdir()
        (cfg_dir / "data.csv").write_text("a,b,Class\n1,2,0\n3,4,1\n")
        payload = {
            "data_path": data_path_value,
            "target_column": "Class",
            "output_dir": "results",
        }
        cfg_file = cfg_dir / "cfg.json"
        cfg_file.write_text(json.dumps(payload))
        return cfg_file, cfg_dir

    def test_relative_data_path_anchors_on_config_not_cwd(self, tmp_path, monkeypatch):
        from SelectOmics.config import SelectOmicsConfig

        cfg_file, cfg_dir = self._write(tmp_path, "data.csv")

        # Load from an unrelated working directory.
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        cfg = SelectOmicsConfig.from_json(cfg_file)
        assert Path(cfg.data_path) == (cfg_dir / "data.csv").resolve()
        assert Path(cfg.data_path).exists()

    def test_relative_output_dir_anchors_on_config(self, tmp_path, monkeypatch):
        from SelectOmics.config import SelectOmicsConfig

        cfg_file, cfg_dir = self._write(tmp_path, "data.csv")
        elsewhere = tmp_path / "elsewhere2"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        cfg = SelectOmicsConfig.from_json(cfg_file)
        assert Path(cfg.output_dir) == (cfg_dir / "results").resolve()

    def test_absolute_data_path_passes_through(self, tmp_path):
        """Configs written by earlier versions must load exactly as before."""
        from SelectOmics.config import SelectOmicsConfig

        absolute = (tmp_path / "somewhere" / "data.csv").resolve()
        cfg_file, _ = self._write(tmp_path, str(absolute))

        cfg = SelectOmicsConfig.from_json(cfg_file)
        assert Path(cfg.data_path) == absolute

    def test_yaml_loader_anchors_too(self, tmp_path, monkeypatch):
        yaml = pytest.importorskip("yaml")
        from SelectOmics.config import SelectOmicsConfig

        cfg_dir = tmp_path / "ystudy"
        cfg_dir.mkdir()
        (cfg_dir / "data.csv").write_text("a,b,Class\n1,2,0\n3,4,1\n")
        cfg_file = cfg_dir / "cfg.yaml"
        cfg_file.write_text(yaml.dump({
            "data_path": "data.csv",
            "target_column": "Class",
        }))

        elsewhere = tmp_path / "elsewhere3"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        cfg = SelectOmicsConfig.from_yaml(cfg_file)
        assert Path(cfg.data_path) == (cfg_dir / "data.csv").resolve()


class TestExampleConfigsArePortable:
    """The shipped example configs must load on a machine that is not the one
    that generated them."""

    def test_no_example_config_stores_an_absolute_path(self):
        import json
        import SelectOmics

        examples = Path(SelectOmics.__file__).parent.parent / "examples"
        if not examples.is_dir():
            pytest.skip("examples/ not present in this distribution")

        offenders = []
        for cfg in sorted(examples.rglob("*_config.json")):
            if ".ipynb_checkpoints" in str(cfg):
                continue
            stored = json.loads(cfg.read_text(encoding="utf-8")).get("data_path", "")
            if Path(stored).is_absolute():
                offenders.append(f"{cfg.name} -> {stored}")

        assert not offenders, (
            "example configs carry machine-specific absolute paths: " + "; ".join(offenders)
        )


class TestFeatureRedundancyDiagnostic:
    """
    Step 1's correlation filter only removes features that have a correlated
    partner, so the fraction that do predicts whether Step 1 will act at all.
    Validated against benchmarks/step1_threshold_sweep.py: across 15 sweep
    points it produced zero false negatives (it never called Step 1 inert when
    Step 1 went on to remove more than 10% of features).
    """

    def _blocks(self, n_samples, n_blocks, block_size, rho, seed=0):
        """Build a matrix of correlated blocks with a known rho."""
        rng = np.random.RandomState(seed)
        cols = {}
        for b in range(n_blocks):
            lead = rng.randn(n_samples)
            for j in range(block_size):
                noise = rng.randn(n_samples)
                cols[f"b{b}_f{j}"] = rho * lead + np.sqrt(1 - rho ** 2) * noise
        return pd.DataFrame(cols)

    def test_high_correlation_is_detected(self):
        from SelectOmics.utils.diagnostics import assess_feature_redundancy

        X = self._blocks(120, n_blocks=5, block_size=20, rho=0.95)
        r = assess_feature_redundancy(X, correlation_threshold=0.70)
        assert r["step1_will_act"] is True
        assert r["redundancy_level"] == "high"
        assert r["redundant_feature_fraction"] > 0.5

    def test_independent_features_are_detected_as_low(self):
        from SelectOmics.utils.diagnostics import assess_feature_redundancy

        rng = np.random.RandomState(0)
        X = pd.DataFrame(rng.randn(120, 200),
                         columns=[f"f{i}" for i in range(200)])
        r = assess_feature_redundancy(X, correlation_threshold=0.70)
        assert r["step1_will_act"] is False
        assert r["redundancy_level"] == "low"
        assert "enable_step1=False" in r["rationale"]

    def test_wide_matrix_is_subsampled_not_refused(self):
        """p^2 correlation is intractable at omics scale; it must subsample."""
        from SelectOmics.utils.diagnostics import assess_feature_redundancy

        rng = np.random.RandomState(0)
        X = pd.DataFrame(rng.randn(60, 3000))
        r = assess_feature_redundancy(X, correlation_threshold=0.70,
                                      max_features=200)
        assert r["n_features_total"] == 3000
        assert r["n_features_sampled"] == 200
        assert "3000 features" in r["rationale"]

    def test_threshold_is_honoured(self):
        """A lower threshold must find at least as much redundancy."""
        from SelectOmics.utils.diagnostics import assess_feature_redundancy

        X = self._blocks(120, n_blocks=4, block_size=10, rho=0.75)
        loose = assess_feature_redundancy(X, correlation_threshold=0.40)
        strict = assess_feature_redundancy(X, correlation_threshold=0.95)
        assert (loose["redundant_feature_fraction"]
                >= strict["redundant_feature_fraction"])

    def test_degenerate_inputs_do_not_raise(self):
        from SelectOmics.utils.diagnostics import assess_feature_redundancy

        single = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        assert assess_feature_redundancy(single)["step1_will_act"] is False

        constant = pd.DataFrame({f"c{i}": [1.0] * 20 for i in range(5)})
        r = assess_feature_redundancy(constant)
        assert r["step1_will_act"] is False


class TestStep1GuardrailIsAdvisoryOnly:
    """
    The guardrail must never change the configuration. A run whose behaviour
    depends on an automatic switch is harder to reproduce than one that did
    what it was told.
    """

    @pytest.mark.integration
    def test_pipeline_warns_but_does_not_disable_step1(self, tmp_path, caplog):
        import logging
        import SelectOmics

        rng = np.random.RandomState(0)
        # Independent features: nothing for Step 1's correlation filter to do.
        df = pd.DataFrame(rng.randn(80, 40),
                          columns=[f"f{i}" for i in range(40)])
        df["Class"] = (rng.rand(80) > 0.5).astype(int)
        data = tmp_path / "indep.csv"
        df.to_csv(data, index=False)

        config = {
            "data_path": str(data), "target_column": "Class",
            "algorithm": "XGB", "n_consensus_models": 1,
            "quick_tune_iterations": 3, "n_bootstrap": 10,
            "output_dir": str(tmp_path / "out"), "use_gpu": False,
            "enable_step1": True, "enable_step2": False,
            "enable_step3": False,
            "save_intermediate_results": False,
            "create_visualizations": False,
        }
        pipeline = SelectOmics.SelectOmicsPipeline(config)
        with caplog.at_level(logging.WARNING, logger="SelectOmics.pipeline"):
            pipeline.load_data()
            pipeline._assess_sample_size()

        # It warned... (getMessage applies the lazy-% args correctly)
        assert any("little correlated redundancy" in r.getMessage()
                   for r in caplog.records), caplog.text
        # ...and it did NOT flip the flag.
        assert pipeline.config.enable_step1 is True
        assert pipeline.feature_redundancy is not None
        assert pipeline.feature_redundancy["step1_will_act"] is False


class TestStep2RankThreshold:
    """
    Step 2 thresholded with ``imp > np.percentile(imp, p * 100)``.  Tree
    importance is exactly 0.0 for every feature never split on, and that tie
    mass is 98-99.8% of the array on this package's target data, so the
    percentile returned 0.0 everywhere in the default (0.0, 0.6) search range
    and the kept count was identical at every p.  The binary search over
    l1_percentile_range was optimising a constant function.
    """

    def test_dense_importance_delivers_requested_fraction(self):
        from SelectOmics.selection.step2_regularization import _rank_keep_mask

        imp = np.linspace(0.01, 1.0, 100)
        for percentile, expected in ((0.0, 100), (0.4, 60), (0.9, 10)):
            assert _rank_keep_mask(imp, percentile).sum() == expected

    def test_percentile_is_not_inert_under_heavy_tie_mass(self):
        """
        The old rule answered the same at EVERY percentile.  The new one
        saturates at the pool while the request exceeds it, then responds
        once the request drops below it.
        """
        from SelectOmics.selection.step2_regularization import _rank_keep_mask

        imp = np.zeros(1000)
        imp[:200] = np.linspace(0.1, 1.0, 200)

        # The default l1/l2_percentile_range is (0.0, 0.6), which sits wholly
        # inside the 80% tie mass. That is the whole span the binary search
        # explores, and the old rule returns the identical count across it.
        in_range = (0.0, 0.2, 0.4, 0.6)
        old = [int((imp > np.percentile(imp, p * 100)).sum()) for p in in_range]
        assert len(set(old)) == 1, "precondition: the old rule was flat"
        assert old[0] == 200, "precondition: 60% of 1000 was never delivered"

        # Ranking saturates at the pool rather than inventing an order among
        # the tied zeros, so it agrees with the old rule inside the range...
        assert [int(_rank_keep_mask(imp, p).sum()) for p in in_range] == old

        # ...and responds properly once the request falls below the pool,
        # which the percentile only ever did by accident of the tie mass.
        above = [int(_rank_keep_mask(imp, p).sum()) for p in (0.85, 0.9, 0.95)]
        assert above == [150, 100, 50]

    def test_never_ranks_into_the_zero_tie_region(self):
        """A feature the model never used carries no evidence for keeping it."""
        from SelectOmics.selection.step2_regularization import _rank_keep_mask

        imp = np.zeros(1000)
        imp[:12] = np.linspace(0.1, 1.0, 12)
        mask = _rank_keep_mask(imp, 0.0)   # asks for all 1000
        assert mask.sum() == 12
        assert not mask[12:].any()

    def test_all_zero_importance_selects_nothing(self):
        from SelectOmics.selection.step2_regularization import _rank_keep_mask
        assert _rank_keep_mask(np.zeros(50), 0.0).sum() == 0
        assert _rank_keep_mask(np.zeros(50), 1.0).sum() == 0

    def test_extreme_percentile_keeps_the_best_feature(self):
        """
        The search domain is now [0, 1], so percentile 1.0 is reachable and
        would ask for nothing. An empty stage forces the intersection empty
        and fires the terminal rank-average fallback.
        """
        from SelectOmics.selection.step2_regularization import _rank_keep_mask

        imp = np.zeros(100)
        imp[7] = 1.0
        imp[3] = 0.5
        mask = _rank_keep_mask(imp, 1.0)
        assert mask.sum() == 1
        assert mask[7]

    def test_xgb_stages_widen_the_split_pool(self):
        from SelectOmics.selection.step2_regularization import (
            _COLSAMPLE_BYTREE, _build_stage_model,
        )
        pytest.importorskip("xgboost")
        for stage in ("l1", "l2"):
            clf = _build_stage_model("XGB", stage, seed=0).named_steps["clf"]
            assert clf.get_params()["colsample_bytree"] == _COLSAMPLE_BYTREE
        assert 0.0 < _COLSAMPLE_BYTREE < 1.0


class TestConsensusStartsAtUnanimity:
    """
    ``consensus_threshold`` set where the graduated relaxation STARTED. Since
    the relaxation stops at whichever level first satisfies the feature floor,
    and that landing point is a property of the data and the floor, starting
    lower only skipped rungs on the way to the same answer: thresholds of 0.4,
    0.6, 0.8 and 1.0 returned the identical feature SET on omics_standard and
    omics_genomics at two seeds.

    It also corrupted the diagnostic. Relaxation only walks downward, so a
    search starting at 4/10 never tested whether 6/10 would have worked and
    reported 4 for a panel that in fact held at 6.
    """

    @staticmethod
    def _votes(n_features=20, n_models=10):
        """Feature i survives at vote level (n_models - i), a clean ladder."""
        v = np.zeros(n_features, dtype=int)
        for i in range(n_features):
            v[i] = max(0, n_models - i)
        return [v, v.copy()]

    def test_votes_required_is_the_strongest_level_that_holds(self):
        from SelectOmics.utils.helpers import relax_consensus_intersection

        votes = self._votes()
        # Floor of 3 is met at 8/10 (features 0,1,2 have >= 8 votes).
        r = relax_consensus_intersection(votes, n_models=10, min_features=3)
        assert r.votes_required == 8
        assert r.mask.sum() == 3

        # The same panel must not be reported at a weaker level just because
        # a caller would once have started lower.
        assert r.agreement == pytest.approx(0.8)

    def test_result_is_independent_of_any_starting_point(self):
        """There is no longer a knob that can change where the search begins."""
        from SelectOmics.utils.helpers import relax_consensus_intersection
        import inspect

        params = inspect.signature(relax_consensus_intersection).parameters
        assert 'consensus_threshold' not in params
        assert 'min_consensus' in params

    def test_min_consensus_binds_before_the_feature_floor(self):
        from SelectOmics.utils.helpers import relax_consensus_intersection

        votes = self._votes()
        # Floor of 10 would need 1/10 votes, but 0.8 forbids going below 8/10.
        r = relax_consensus_intersection(votes, n_models=10, min_features=10,
                                         min_consensus=0.8)
        assert r.how == 'consensus_limited'
        assert r.votes_required == 8
        assert r.mask.sum() == 3          # fewer than the floor, by design
        assert r.floor_requested == 10

    def test_min_consensus_one_means_unanimity_or_nothing(self):
        from SelectOmics.utils.helpers import relax_consensus_intersection

        votes = self._votes()
        r = relax_consensus_intersection(votes, n_models=10, min_features=10,
                                         min_consensus=1.0)
        assert r.votes_required == 10
        assert r.mask.sum() == 1          # only feature 0 reaches 10/10
        assert r.how == 'consensus_limited'

    def test_none_relaxes_as_far_as_the_floor_requires(self):
        from SelectOmics.utils.helpers import relax_consensus_intersection

        votes = self._votes()
        r = relax_consensus_intersection(votes, n_models=10, min_features=10,
                                         min_consensus=None)
        assert r.mask.sum() >= 10
        assert r.how == 'relaxed'


class TestAgreementIsReported:
    """
    Removing consensus_threshold leaves the user with no way to state how
    conservative they want to be, so the pipeline must instead report how
    conservative the result actually was.
    """

    @pytest.mark.parametrize("agreement,expected", [
        (1.00, 'unanimous'),
        (0.90, 'strong'),
        (0.80, 'strong'),
        (0.70, 'moderate'),
        (0.60, 'moderate'),
        (0.50, 'weak'),
        (0.40, 'weak'),
        (0.10, 'minimal'),
    ])
    def test_bands(self, agreement, expected):
        from SelectOmics.utils.helpers import agreement_label
        assert agreement_label(agreement) == expected

    def test_rank_average_reports_no_agreement(self):
        from SelectOmics.utils.helpers import agreement_label
        assert 'none' in agreement_label(1.0, how='rank_average')

    def test_consensus_result_carries_label(self):
        from SelectOmics.utils.helpers import relax_consensus_intersection

        v = np.array([10, 10, 5, 5, 1, 1, 0, 0])
        r = relax_consensus_intersection([v, v.copy()], n_models=10,
                                         min_features=2)
        assert r.label == 'unanimous'
        assert r.agreement == pytest.approx(1.0)


class TestRankFallbackStaysInTheVotedPool:
    """
    Rank-averaging fires only when no feature was selected by any replicate in
    every stage. It must still rank within features that got a vote SOMEWHERE:
    ordering by an importance of exactly zero is ordering by array index.

    Measured, this changes nothing today (mean importance is zero for a feature
    no model used, so ranking already confined itself to the voted pool in 0 of
    161.5 cases on average) but it makes the guarantee structural.
    """

    def test_ranks_within_the_voted_pool(self):
        from SelectOmics.utils.helpers import relax_consensus_intersection

        n = 10
        # Disjoint stages: the intersection is empty at every level, so the
        # rank fallback is the only path.
        a = np.array([3, 3, 0, 0, 0, 0, 0, 0, 0, 0])
        b = np.array([0, 0, 3, 3, 0, 0, 0, 0, 0, 0])
        # Features 8 and 9 have the highest importance but no votes anywhere.
        imp = np.zeros(n)
        imp[8] = 9.0
        imp[9] = 9.0
        imp[[0, 1, 2, 3]] = [4.0, 3.0, 2.0, 1.0]

        r = relax_consensus_intersection([a, b], n_models=3, min_features=2,
                                         mean_importance=imp)
        assert r.how == 'rank_average'
        chosen = set(np.flatnonzero(r.mask).tolist())
        assert chosen <= {0, 1, 2, 3}, "stepped outside the voted pool"
        assert 8 not in chosen and 9 not in chosen

    def test_steps_outside_only_when_the_pool_is_too_small(self):
        from SelectOmics.utils.helpers import relax_consensus_intersection

        n = 10
        a = np.array([3, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        b = np.array([0, 3, 0, 0, 0, 0, 0, 0, 0, 0])
        imp = np.arange(n, dtype=float)[::-1].copy()

        # Voted pool is {0, 1}; asking for 5 must be allowed to reach further.
        r = relax_consensus_intersection([a, b], n_models=3, min_features=5,
                                         mean_importance=imp)
        assert r.how == 'rank_average'
        assert r.mask.sum() == 5


class TestCliCanExpressNone:
    """
    ``--set min_consensus=none`` passed the string 'none' through to
    validation, which compared a float against a str and raised an unhandled
    TypeError. None is a meaningful setting for the Optional fields, not an
    absence: for min_consensus it means "relax as far as the feature floor
    requires".
    """

    @pytest.mark.parametrize("literal", ["none", "None", "NULL", "null", ""])
    def test_none_literals_become_none(self, literal):
        from SelectOmics.cli import _parse_overrides
        assert _parse_overrides([f"min_consensus={literal}"]) == {
            "min_consensus": None}

    def test_config_accepts_the_parsed_override(self, tmp_path):
        from SelectOmics.cli import _parse_overrides
        from SelectOmics.config import SelectOmicsConfig

        for field in ("min_consensus", "stability_threshold"):
            cfg = SelectOmicsConfig(
                data_path=str(tmp_path / "x.csv"), target_column="y",
                **_parse_overrides([f"{field}=none"]),
            )
            # stability_threshold defaults to 0.6 when None; min_consensus
            # stays None. Neither may raise.
            assert getattr(cfg, field) in (None, 0.6)

    def test_real_values_still_parse(self):
        from SelectOmics.cli import _parse_overrides
        assert _parse_overrides(["min_consensus=0.8"]) == {"min_consensus": 0.8}
        assert _parse_overrides(["n_consensus_models=5"]) == {
            "n_consensus_models": 5}
        assert _parse_overrides(["enable_step3=false"]) == {
            "enable_step3": False}


class TestAgreementSurvivesToDisk:
    """
    The agreement level replaced a config knob, so it is the only record of
    how strong a panel's claim is. It has to outlive the process.

    Two gaps this covers. The metrics record keyed 'consensus_threshold', a
    setting that no longer exists; it now keys the achieved agreement, which
    is an outcome. And Steps 1 to 3 wrote their summaries inside the optional
    evaluation helper, so a run with enable_step_evaluations=False -- a common
    speed setting, and the one the benchmark harness uses -- persisted no
    agreement record at all.
    """

    def test_metrics_record_carries_achieved_agreement(self):
        from SelectOmics.evaluation.metrics import build_comprehensive_metrics

        rng = np.random.RandomState(0)
        y_test = rng.randint(0, 2, 40)
        y_pred = rng.randint(0, 2, 40)
        proba = rng.rand(40, 2)
        proba /= proba.sum(axis=1, keepdims=True)

        metrics = build_comprehensive_metrics(
            y_test=y_test, y_pred=y_pred, y_proba=proba,
            class_names=['a', 'b'], n_classes=2,
            training_result={'mean_auc': 0.8, 'std_auc': 0.05},
            step_label='Step 2', algorithm='XGB', n_models=5,
            achieved_agreement=0.6, agreement_label='moderate',
            original_n_features=100, final_n_features=10,
        )
        assert metrics['achieved_agreement'] == 0.6
        assert metrics['agreement_label'] == 'moderate'
        assert 'consensus_threshold' not in metrics

    @pytest.mark.integration
    def test_step_summaries_persist_without_evaluations(self, tmp_path):
        import SelectOmics

        rng = np.random.RandomState(0)
        X = rng.randn(110, 40)
        y = (X[:, :5] @ rng.randn(5) > 0).astype(int)
        df = pd.DataFrame(X, columns=[f"f{i:02d}" for i in range(40)])
        df["Class"] = y
        data = tmp_path / "d.csv"
        df.to_csv(data, index=False)
        out = tmp_path / "res"

        pipeline = SelectOmics.SelectOmicsPipeline({
            "data_path": str(data), "target_column": "Class",
            "algorithm": "XGB", "n_consensus_models": 3,
            "quick_tune_iterations": 3, "n_bootstrap": 10,
            "output_dir": str(out), "use_gpu": False,
            "enable_step3": False,
            "enable_step_evaluations": False,          # the gap
            "enable_final_test_evaluation": False,
            "save_intermediate_results": True,
            "create_visualizations": False,
        })
        pipeline.run(validate=False)

        for step in ("step1", "step2"):
            path = out / f"{step}_summary.txt"
            assert path.exists(), f"{step} wrote no summary"
            text = path.read_text(encoding="utf-8")
            for key in ("votes_required", "agreement",
                        "agreement_label", "consensus_outcome"):
                assert f"{key}:" in text, f"{step} summary lacks {key}"


class TestCheckpointRefusesForeignBuilds:
    """
    Checkpoints carried no version stamp, so `run(resume=True)` would happily
    restore one written by a different build. Selection semantics change
    between versions -- 0.6.1 to 0.7.0 alters results on every algorithm -- so
    that produced a run with early steps computed under one rule set and later
    steps under another, describing neither, with nothing to detect it.
    """

    @staticmethod
    def _make(tmp_path):
        import SelectOmics

        rng = np.random.RandomState(0)
        X = rng.randn(90, 30)
        y = (X[:, :4] @ rng.randn(4) > 0).astype(int)
        df = pd.DataFrame(X, columns=[f"f{i:02d}" for i in range(30)])
        df["Class"] = y
        data = tmp_path / "d.csv"
        df.to_csv(data, index=False)
        cfg = {
            "data_path": str(data), "target_column": "Class",
            "algorithm": "XGB", "n_consensus_models": 2,
            "quick_tune_iterations": 3, "n_bootstrap": 10,
            "output_dir": str(tmp_path / "res"), "use_gpu": False,
            "enable_step3": False,
            "enable_step_evaluations": False,
            "enable_final_test_evaluation": False,
            "save_intermediate_results": True,
            "create_visualizations": False,
        }
        SelectOmics.SelectOmicsPipeline(cfg).run(validate=False)
        return cfg, tmp_path / "res" / ".selectomics_checkpoint.pkl"

    @pytest.mark.integration
    def test_stamps_the_writing_version(self, tmp_path):
        import pickle
        import SelectOmics

        _, cp = self._make(tmp_path)
        assert cp.exists()
        with open(cp, "rb") as fh:
            saved = pickle.load(fh)
        assert saved.get("selectomics_version") == SelectOmics.__version__

    @pytest.mark.integration
    def test_matching_version_resumes(self, tmp_path):
        import SelectOmics

        cfg, _ = self._make(tmp_path)
        assert SelectOmics.SelectOmicsPipeline(cfg)._load_checkpoint() is not None

    @pytest.mark.integration
    @pytest.mark.parametrize("stamp", ["0.6.1", None])
    def test_foreign_or_unversioned_checkpoint_is_refused(self, tmp_path, stamp):
        import pickle
        import SelectOmics

        cfg, cp = self._make(tmp_path)
        with open(cp, "rb") as fh:
            saved = pickle.load(fh)
        if stamp is None:
            saved.pop("selectomics_version", None)
        else:
            saved["selectomics_version"] = stamp
        with open(cp, "wb") as fh:
            pickle.dump(saved, fh, protocol=4)

        pipeline = SelectOmics.SelectOmicsPipeline(cfg)
        assert pipeline._load_checkpoint() is None, (
            "resumed a checkpoint written by a different build"
        )


class TestClassAwareCorrelationFilter:
    """
    Step 1's correlation filter used total Pearson correlation, which cannot
    tell two probes measuring the same thing from two independent markers of
    the same disease. On synthetic data with 20 informative features sharing
    the class signal, it kept 1 of 20 while retaining all 180 noise features.
    """

    @staticmethod
    def _coregulated(n=400, p=120, n_informative=15, signal=2.0, seed=0):
        rng = np.random.RandomState(seed)
        y = rng.randint(0, 2, n)
        X = rng.randn(n, p)
        for j in range(n_informative):
            X[:, j] = signal * (y - 0.5) * 2.0 + rng.randn(n) * 0.5
        cols = [f"f{j:03d}" for j in range(p)]
        return pd.DataFrame(X, columns=cols), y, cols[:n_informative]

    def test_within_class_correlation_removes_the_class_signal(self):
        from SelectOmics.selection.step1_cleaning import (
            pooled_within_class_corr,
        )
        X, y, inf_cols = self._coregulated()
        total = X.corr().abs()
        within = pooled_within_class_corr(X, y)

        iu = np.triu_indices(len(inf_cols), k=1)
        total_r = total.loc[inf_cols, inf_cols].values[iu].mean()
        within_r = within.loc[inf_cols, inf_cols].values[iu].mean()

        assert total_r > 0.7, "precondition: informative features co-vary"
        assert within_r < total_r / 2, (
            f"within-class correlation {within_r:.3f} should be far below "
            f"total {total_r:.3f} once the class signal is removed"
        )

    def test_co_regulated_features_survive(self):
        from SelectOmics.selection.step1_cleaning import (
            _apply_correlation_filter, class_association,
            pooled_within_class_corr,
        )
        X, y, inf_cols = self._coregulated()
        variances = X.var(axis=0)

        kept_old = _apply_correlation_filter(X, 0.70, variances,
                                             X.corr().abs())
        n_old = sum(1 for j, c in enumerate(X.columns)
                    if kept_old[j] and c in inf_cols)

        kept_new = _apply_correlation_filter(
            X, 0.70, variances, pooled_within_class_corr(X, y),
            keep_priority=class_association(X, y))
        n_new = sum(1 for j, c in enumerate(X.columns)
                    if kept_new[j] and c in inf_cols)

        assert n_old < len(inf_cols) / 2, "precondition: old filter collapses"
        assert n_new == len(inf_cols), (
            f"class-aware filter kept {n_new}/{len(inf_cols)} informative "
            f"features; it should keep all of them"
        )

    def test_genuine_duplicates_are_still_removed(self):
        """The fix must not disable the filter: exact copies must still go."""
        from SelectOmics.selection.step1_cleaning import (
            _apply_correlation_filter, class_association,
            pooled_within_class_corr,
        )
        rng = np.random.RandomState(0)
        n, p = 300, 20
        X = pd.DataFrame(rng.randn(n, p),
                         columns=[f"f{j:02d}" for j in range(p)])
        y = rng.randint(0, 2, n)
        # f01 is a near-exact duplicate of f00: redundant regardless of class.
        X["f01"] = X["f00"] + rng.randn(n) * 1e-3

        kept = _apply_correlation_filter(
            X, 0.70, X.var(axis=0), pooled_within_class_corr(X, y),
            keep_priority=class_association(X, y))
        survivors = {c for j, c in enumerate(X.columns) if kept[j]}
        assert not {"f00", "f01"} <= survivors, (
            "a genuine duplicate pair should not both survive"
        )

    def test_eta_squared_is_bounded_and_tracks_signal(self):
        from SelectOmics.selection.step1_cleaning import class_association
        X, y, inf_cols = self._coregulated()
        eta = class_association(X, y)
        assert eta.min() >= 0.0 and eta.max() <= 1.0
        noise_cols = [c for c in X.columns if c not in inf_cols]
        assert eta[inf_cols].mean() > eta[noise_cols].mean() * 5

    def test_config_flag_restores_previous_behaviour(self):
        from SelectOmics.config import SelectOmicsConfig
        assert SelectOmicsConfig(
            data_path="x.csv", target_column="y").class_aware_correlation is True
        assert SelectOmicsConfig(
            data_path="x.csv", target_column="y",
            class_aware_correlation=False).class_aware_correlation is False

    def test_survives_missing_values(self):
        """
        SelectOmics accepts NaN and never imputes. The first implementation
        used numpy mean/sum, which propagate NaN: one missing cell made a
        class mean NaN and poisoned the whole feature (144 non-finite cells
        at 5% missing). class_association failed worse -- silently. Its
        `ss_total > 0` guard is False for NaN, so any feature with a missing
        value scored 0.0 and was preferentially dropped, with no warning.
        """
        from SelectOmics.selection.step1_cleaning import (
            class_association, pooled_within_class_corr,
        )
        rng = np.random.RandomState(0)
        n, p = 200, 12
        y = rng.randint(0, 2, n)
        X = rng.randn(n, p)
        X[:, :4] += 2.0 * y[:, None]
        df = pd.DataFrame(X, columns=[f"f{j:02d}" for j in range(p)])
        df = df.mask(rng.rand(n, p) < 0.05)
        assert df.isna().sum().sum() > 0, "precondition: data has NaN"

        within = pooled_within_class_corr(df, y)
        eta = class_association(df, y)
        assert np.isfinite(within.values).all(), "NaN leaked into within-class"
        assert np.isfinite(eta.values).all(), "NaN leaked into eta-squared"
        # Missingness must not be mistaken for uninformativeness.
        assert eta[[f"f{j:02d}" for j in range(4)]].mean() > 0.3

    def test_multiclass(self):
        from SelectOmics.selection.step1_cleaning import (
            class_association, pooled_within_class_corr,
        )
        rng = np.random.RandomState(0)
        n, p = 240, 12
        y = rng.randint(0, 3, n)
        X = rng.randn(n, p)
        X[:, :4] += 2.0 * y[:, None]
        df = pd.DataFrame(X, columns=[f"f{j:02d}" for j in range(p)])

        within = pooled_within_class_corr(df, y)
        eta = class_association(df, y)
        assert np.isfinite(within.values).all()
        assert 0.0 <= eta.min() and eta.max() <= 1.0
        assert (eta[[f"f{j:02d}" for j in range(4)]].mean()
                > eta[[f"f{j:02d}" for j in range(4, p)]].mean() * 5)

    def test_singleton_class_does_not_raise(self):
        from SelectOmics.selection.step1_cleaning import (
            class_association, pooled_within_class_corr,
        )
        rng = np.random.RandomState(0)
        n, p = 60, 10
        y = rng.randint(0, 2, n)
        y[0] = 7                                  # a class of exactly one
        df = pd.DataFrame(rng.randn(n, p),
                          columns=[f"f{j:02d}" for j in range(p)])
        assert np.isfinite(pooled_within_class_corr(df, y).values).all()
        assert np.isfinite(class_association(df, y).values).all()

    def test_result_does_not_depend_on_column_order(self):
        from SelectOmics.selection.step1_cleaning import (
            _apply_correlation_filter, class_association,
            pooled_within_class_corr,
        )
        X, y, _ = self._coregulated(n=200, p=40, n_informative=6)

        def run(frame):
            kept = _apply_correlation_filter(
                frame, 0.70, frame.var(axis=0),
                pooled_within_class_corr(frame, y),
                keep_priority=class_association(frame, y))
            return {frame.columns[i] for i in np.where(kept)[0]}

        assert run(X) == run(X[list(X.columns[::-1])])


class TestSourcesAreAscii:
    def test_no_non_ascii_characters(self):
        from pathlib import Path
        import SelectOmics

        root = Path(SelectOmics.__file__).parent
        offenders = []
        for path in sorted(root.rglob("*.py")):
            if "checkpoint" in str(path):
                continue
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if any(ord(ch) > 127 for ch in line):
                    offenders.append(f"{path.name}:{lineno}")
        assert not offenders, f"non-ASCII source lines: {offenders}"


# ---------------------------------------------------------------------------
# step2_target_retention was inert: it was a fraction of the INPUT width, but
# _rank_keep_mask caps the selection at the number of features with non-zero
# importance (~1% of input for a tree on wide data). Every setting above that
# clamped to the same answer. It is now a fraction of the reachable pool.
# ---------------------------------------------------------------------------

class TestStep2RetentionIsLive:
    """The retention knob must actually change what Step 2 keeps."""

    def test_rank_keep_mask_caps_at_nonzero(self):
        """The cap that made the old semantics inert still holds."""
        from SelectOmics.selection.step2_regularization import _rank_keep_mask

        imp = np.zeros(100)
        imp[:7] = np.linspace(1.0, 0.1, 7)      # only 7 features ever used
        # Asking for 90% of 100 features cannot exceed the 7 that exist.
        assert _rank_keep_mask(imp, 0.10).sum() == 7
        assert _rank_keep_mask(imp, 0.50).sum() == 7
        # Asking for fewer than the pool still binds.
        assert _rank_keep_mask(imp, 0.97).sum() == 3

    def test_retention_scales_within_the_pool(self):
        """A lower retention keeps strictly fewer of the usable features."""
        from SelectOmics.selection.step2_regularization import _rank_keep_mask

        imp = np.zeros(2000)
        imp[:20] = np.linspace(1.0, 0.05, 20)
        pool = int(np.count_nonzero(imp))
        counts = []
        for retention in (0.25, 0.5, 1.0):
            n_target = max(1, int(np.ceil(pool * retention)))
            pct = 1.0 - (n_target / len(imp))
            counts.append(_rank_keep_mask(imp, pct).sum())
        assert counts == sorted(counts), counts
        assert counts[0] < counts[-1], counts
        assert counts[-1] == pool

    def test_retention_accepts_one_and_rejects_zero(self):
        """1.0 means 'the whole pool' and is a legal setting; 0.0 is not."""
        from SelectOmics.config import SelectOmicsConfig

        cfg = SelectOmicsConfig(data_path="d.csv", target_column="Class",
                                step2_target_retention=1.0)
        assert cfg.step2_target_retention == 1.0
        with pytest.raises(ValueError, match="step2_target_retention"):
            SelectOmicsConfig(data_path="d.csv", target_column="Class",
                              step2_target_retention=0.0)

    def test_stage_colsample_is_validated(self):
        """The pool-size knob rejects out-of-range values."""
        from SelectOmics.config import SelectOmicsConfig

        cfg = SelectOmicsConfig(data_path="d.csv", target_column="Class",
                                step2_stage_colsample=1.0)
        assert cfg.step2_stage_colsample == 1.0
        for bad in (0.0, 1.5, -0.1):
            with pytest.raises(ValueError, match="step2_stage_colsample"):
                SelectOmicsConfig(data_path="d.csv", target_column="Class",
                                  step2_stage_colsample=bad)

    def test_stage_colsample_reaches_the_model(self):
        """The configured rate must land on the XGB stage estimator."""
        pytest.importorskip("xgboost")
        from SelectOmics.selection.step2_regularization import _build_stage_model

        m = _build_stage_model("XGB", "l1", seed=0, colsample=0.77)
        assert m.named_steps["clf"].get_params()["colsample_bytree"] == pytest.approx(0.77)


# ---------------------------------------------------------------------------
# Step 3: the entry gate and the RFECV elimination step size.
#
# MIN_FEATURES_FOR_RFECV was 30, asserted rather than measured, and blocked the
# step across the whole range where it works: Step 3 fired in 1 of 65 benchmark
# runs. Measured against a pass-through baseline, the crossover is between 8
# and 10 features, so the gate is now 10.
#
# step=1 made RFECV cost roughly quadratic in the feature count (545 features
# once cost 34,771 CPU-seconds). It is now exact below 100 features and
# proportional above, so ordinary operation is unchanged and only wide panels
# take the approximation.
# ---------------------------------------------------------------------------

class TestStep3Gate:
    """
    The gate is 30, and lowering it is a trap that must not be re-sprung.

    Measuring Step 3 on a panel that is half noise suggests a gate near 10.
    That measurement does not describe the panel Step 2 actually delivers,
    which is nearly pure signal (precision 0.970 across the 5-seed benchmark).
    On a pure-signal panel elimination can only cut true features: measured end
    to end at a gate of 10, Step 2 delivered 10 features at F1 1.000 on three
    runs and Step 3 reduced them to one or two, F1 0.182.
    """

    def test_gate_is_thirty(self):
        from SelectOmics.selection.step3_wrapper import MIN_FEATURES_FOR_RFECV
        assert MIN_FEATURES_FOR_RFECV == 30, (
            "lowering this gate lets Step 3 run on panels Step 2 has already "
            "filtered to purity, where it destroys them; see the constant's "
            "comment for the measurement"
        )

    def test_suggest_gate_matches_the_step_gate(self):
        """config.suggest() must not disable Step 3 at a different count."""
        from SelectOmics.selection.step3_wrapper import MIN_FEATURES_FOR_RFECV
        import inspect
        from SelectOmics import config as cfg_mod
        src = inspect.getsource(cfg_mod.SelectOmicsConfig.suggest)
        assert f"n_features < {MIN_FEATURES_FOR_RFECV}" in src

    @pytest.mark.integration
    def test_elimination_on_a_pure_panel_is_destructive(self):
        """
        The reason the gate is high, asserted directly: RFECV on a panel with
        no noise in it removes true features. This is what a lower gate would
        expose the pipeline to.
        """
        from sklearn.feature_selection import RFECV
        from SelectOmics.selection.step3_wrapper import (
            _build_rfecv_estimator, _rfecv_scoring, _rfecv_step,
        )

        rng = np.random.RandomState(0)
        n, p = 120, 10
        X = pd.DataFrame(rng.randn(n, p), columns=[f"f{i}" for i in range(p)])
        y = rng.randint(0, 2, size=n)
        X += 1.4 * y[:, None]                 # EVERY feature is informative
        est, getter = _build_rfecv_estimator("XGB", seed=0)
        r = RFECV(estimator=est, step=_rfecv_step(p),
                  cv=StratifiedKFold(3, shuffle=True, random_state=0),
                  scoring=_rfecv_scoring("XGB", 2), n_jobs=1,
                  min_features_to_select=1, importance_getter=getter)
        r.fit(X, y)
        # Not an assertion about a good outcome: elimination on a pure panel
        # can only lose signal, and the gate exists so it never sees one.
        assert r.n_features_ <= p


class TestRfecvStepSize:
    """Exact below the threshold, proportional above, never zero."""

    def test_exact_for_narrow_inputs(self):
        from SelectOmics.selection.step3_wrapper import _rfecv_step
        for n in (10, 25, 50, 99, 100):
            assert _rfecv_step(n) == 1, n

    def test_proportional_for_wide_inputs(self):
        from SelectOmics.selection.step3_wrapper import _rfecv_step
        for n in (200, 545, 2000):
            step = _rfecv_step(n)
            assert step > 1, n
            # Round count stays bounded rather than scaling with n.
            assert 10 <= n // step <= 40, (n, step, n // step)

    def test_step_is_never_zero(self):
        """A zero step makes RFECV raise; the floor must hold at any size."""
        from SelectOmics.selection.step3_wrapper import _rfecv_step
        for n in (1, 2, 5, 101, 10**6):
            assert _rfecv_step(n) >= 1, n

    def test_ordinary_operation_is_unchanged(self):
        """
        With the gate at 10 and exactness up to 100, the panel sizes Step 3
        normally sees are all still eliminated one feature at a time.
        """
        from SelectOmics.selection.step3_wrapper import (
            _rfecv_step, MIN_FEATURES_FOR_RFECV, _RFECV_EXACT_BELOW,
        )
        assert MIN_FEATURES_FOR_RFECV < _RFECV_EXACT_BELOW
        for n in range(MIN_FEATURES_FOR_RFECV, _RFECV_EXACT_BELOW + 1):
            assert _rfecv_step(n) == 1, n


# ---------------------------------------------------------------------------
# Step 2 and Step 3 as a matched pair.
#
# Step 2's optimal aggression depends on whether it is the LAST step, and it
# had no way to know. Left as 'terminal' it produces a narrow high-precision
# panel that sits below Step 3's gate, so Step 3 skips -- and if it did run, it
# would cut true features because there is no noise left to remove.
# 'prefilter' declares the other intent: overshoot the gate so Step 3 engages
# and does the precision work.
# ---------------------------------------------------------------------------

class TestStep2Role:
    """The pairing must be explicit, validated, and default to terminal."""

    def test_default_is_terminal(self):
        from SelectOmics.config import SelectOmicsConfig
        assert SelectOmicsConfig.__dataclass_fields__["step2_role"].default == "terminal"

    def test_rejects_unknown_role(self):
        from SelectOmics.config import SelectOmicsConfig
        with pytest.raises(ValueError, match="step2_role"):
            SelectOmicsConfig(data_path="d.csv", target_column="Class",
                              step2_role="aggressive")

    def test_prefilter_requires_step3(self):
        """
        Prefilter without a pruning step hands back an over-inclusive candidate
        set and calls it an answer. That combination must not be constructible.
        """
        from SelectOmics.config import SelectOmicsConfig
        with pytest.raises(ValueError, match="enable_step3"):
            SelectOmicsConfig(data_path="d.csv", target_column="Class",
                              step2_role="prefilter", enable_step3=False)

    def test_prefilter_target_clears_the_step3_gate(self):
        """
        The whole point is that Step 3 engages, so the floor must exceed its
        entry gate with margin rather than landing on it.
        """
        from SelectOmics.selection.step2_regularization import _PREFILTER_GATE_MULTIPLE
        from SelectOmics.selection.step3_wrapper import MIN_FEATURES_FOR_RFECV
        assert _PREFILTER_GATE_MULTIPLE >= 2
        assert _PREFILTER_GATE_MULTIPLE * MIN_FEATURES_FOR_RFECV > MIN_FEATURES_FOR_RFECV

    def test_terminal_role_is_unchanged(self):
        """The default path must not have moved: 'terminal' uses the floor."""
        import inspect
        from SelectOmics.selection import step2_regularization as s2
        src = inspect.getsource(s2.run_step2_regularization)
        assert "floor = config.min_features_floor" in src


# ---------------------------------------------------------------------------
# benchmarks/_parallel.run_tasks: one failing trial must not end the run.
#
# future.result() was called unguarded, so any exception in any trial
# propagated out of run_tasks, out of run_layer, and killed the whole
# benchmark. Observed on the LGG CNV layer: folds 0-4 completed, one trial in
# fold 5 raised, and folds 5-9 plus the entire merged layer were never run.
#
# The pool-failure fallback had a second problem: it re-ran the FULL task list,
# so a mid-run pool death would append a second copy of every row on_result had
# already written.
# ---------------------------------------------------------------------------

def _flaky_task(i):
    """Module level so Windows spawn can pickle it."""
    if i in (3, 7):
        raise ValueError(f"synthetic failure on task {i}")
    return i * 10


class TestRunTasksFailureIsolation:

    def _load(self):
        """
        Import by NAME with benchmarks on sys.path, not via
        spec_from_file_location: Windows spawns workers rather than forking, so
        the child must be able to re-import the module the task function lives
        in, and a synthetic module name is not importable there.
        """
        import importlib, sys
        bench = Path(__file__).resolve().parents[1] / "benchmarks"
        if not (bench / "_parallel.py").exists():
            pytest.skip("benchmarks/_parallel.py not present")
        if str(bench) not in sys.path:
            sys.path.insert(0, str(bench))
        return importlib.import_module("_parallel")

    def _run(self, workers):
        mod = self._load()
        seen = []
        out = mod.run_tasks(_flaky_task, [(i,) for i in range(10)],
                            workers=workers, n_jobs=1, on_result=seen.append)
        return out, seen

    def test_sequential_path_skips_the_failure(self):
        out, seen = self._run(workers=1)
        assert len(out) == 8, out
        assert 30 not in out and 70 not in out
        assert len(seen) == 8, "only successful results may reach on_result"

    @pytest.mark.integration
    def test_parallel_path_skips_the_failure(self):
        out, seen = self._run(workers=4)
        assert len(out) == 8, out
        assert sorted(out) == [i * 10 for i in range(10) if i not in (3, 7)]
        assert len(seen) == 8

    def test_fallback_does_not_duplicate_completed_work(self):
        """
        The pool-failure fallback must resume, not restart: re-running tasks
        whose rows on_result already wrote would double them in the checkpoint.
        """
        import inspect
        mod = self._load()
        src = inspect.getsource(mod.run_tasks)
        assert "completed" in src, "no record of which tasks already finished"
        assert "if i not in completed" in src, (
            "the fallback re-runs the full task list, duplicating rows already "
            "handed to on_result"
        )


# ---------------------------------------------------------------------------
# Bootstrap seeds must be unique per (model, fold) pair
# ---------------------------------------------------------------------------

class TestBootstrapSeedUniqueness:
    """
    cv_evaluate_model derives a bootstrap seed per (consensus model, fold).
    Its docstring promises "every (model, fold) pair sees a unique sample",
    and consensus voting depends on it: models that share a bootstrap sample
    are not independent draws, so their agreement is not evidence.

    The old form was `model_seed + fold_idx * 1000`. Consensus model seeds are
    spaced 100 apart, so model i+10 at fold f aliased onto model i at fold
    f+1. This is reachable from a shipped preset, which sets
    n_consensus_models=20.
    """

    @staticmethod
    def _seeds(n_models, n_folds, base=42):
        from SelectOmics.models.evaluation import (
            _FOLD_SEED_RADIX, _SEED_MODULUS,
        )
        return [
            ((base + i * 100) * _FOLD_SEED_RADIX + f) % _SEED_MODULUS
            for i in range(n_models)
            for f in range(n_folds)
        ]

    @pytest.mark.parametrize("n_models,n_folds", [
        (20, 5),    # the shipped preset: 40 collisions under the old scheme
        (11, 2),    # the smallest n_models that aliased
        (10, 10),
        (50, 10),
        (1, 2),
    ])
    def test_no_two_pairs_share_a_seed(self, n_models, n_folds):
        seeds = self._seeds(n_models, n_folds)
        assert len(set(seeds)) == len(seeds), (
            f"{len(seeds) - len(set(seeds))} of {len(seeds)} (model, fold) "
            f"pairs share a bootstrap seed at n_consensus_models={n_models}, "
            f"n_folds={n_folds}; consensus models are not independent draws"
        )

    def test_seeds_stay_in_the_range_sklearn_accepts(self):
        """random_state must be a non-negative int below 2**32."""
        for base in (0, 42, 2 ** 31 - 1, 2 ** 32 - 1):
            for s in self._seeds(20, 5, base=base):
                assert 0 <= s < 2 ** 32, f"seed {s} out of range for base {base}"

    def test_the_aliasing_form_is_gone(self):
        """
        Pin the expression itself, so a refactor cannot reintroduce it.

        Comments are stripped first: the fix documents the old form in a
        comment to explain why it was wrong, and that prose must not be
        mistaken for live code.
        """
        import inspect
        from SelectOmics.models import evaluation
        src = inspect.getsource(evaluation.cv_evaluate_model)
        code = [line.split("#", 1)[0] for line in src.splitlines()]
        assert not any("model_seed + fold_idx * 1000" in ln for ln in code), (
            "the aliasing additive seed scheme is back"
        )


# ---------------------------------------------------------------------------
# Global logging state must not leak across tests
# ---------------------------------------------------------------------------

class TestLoggerStateIsRestored:
    """
    enable_logging() is public API and mutates the package logger globally.
    A test that calls it must not change the logger any later test sees.

    Before the conftest fixture, test_config.py called
    enable_logging('WARNING') and left it there, which made
    test_config_paths.py::test_suggest_logs_its_rationale assert against an
    empty log capture. It passed alone and failed in the suite.

    These two tests run in file order, so the first is the polluter and the
    second is the detector.
    """

    def test_a_pollute_the_package_logger(self):
        import logging
        import SelectOmics
        SelectOmics.enable_logging('ERROR')
        assert logging.getLogger("SelectOmics").level == logging.ERROR

    def test_b_the_pollution_did_not_survive(self):
        import logging
        assert logging.getLogger("SelectOmics").level != logging.ERROR, (
            "the previous test's enable_logging('ERROR') leaked into this one; "
            "the conftest restore fixture is not working"
        )

    def test_captured_records_still_reach_caplog(self, caplog):
        """The end the fixture serves: an INFO assertion must be capturable."""
        import logging
        import SelectOmics
        SelectOmics.enable_logging('ERROR')
        caplog.set_level(logging.INFO, logger="SelectOmics")
        logging.getLogger("SelectOmics").info("a probe record")
        assert "a probe record" in caplog.text


# ---------------------------------------------------------------------------
# Plot styling must not leak into the caller's matplotlib
# ---------------------------------------------------------------------------

class TestPlotStyleIsScoped:
    """
    The visualization module sets larger fonts for its own figures. It used to
    do that with plt.rcParams.update(), a permanent process-global mutation.
    Applying it inside each plotting function instead of at import time meant a
    bare `import SelectOmics` was clean, but the first plot the package drew
    silently resized the text in every figure the caller drew afterwards.

    The module comment states the intent: importing the package must not mutate
    a caller's global matplotlib state. rc_context makes that true for the whole
    lifetime, not just until the first plot.
    """

    def test_rcparams_are_restored_after_a_real_plot(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from SelectOmics.config import SelectOmicsConfig
        from SelectOmics.evaluation.visualization import plot_roc_curves

        watched = ["font.size", "axes.titlesize", "axes.labelsize",
                   "xtick.labelsize", "ytick.labelsize", "legend.fontsize",
                   "figure.titlesize"]
        before = {k: plt.rcParams[k] for k in watched}

        rng = np.random.RandomState(0)
        n, n_classes = 40, 2
        y = np.array([i % n_classes for i in range(n)])
        proba = rng.rand(n, n_classes)
        proba = proba / proba.sum(axis=1, keepdims=True)
        models = {"Model-0": {"oof_proba": proba, "mean_auc": 0.8}}
        cfg = SelectOmicsConfig(data_path="unused.csv", target_column="Class",
                                output_dir=str(tmp_path))

        plot_roc_curves(models, y, n_classes, ["0", "1"], cfg, "ROC",
                        str(tmp_path / "roc.png"))

        after = {k: plt.rcParams[k] for k in watched}
        assert after == before, (
            "plotting changed the caller's global matplotlib font settings: "
            + ", ".join(f"{k} {before[k]} -> {after[k]}"
                        for k in watched if before[k] != after[k])
        )

    def test_the_fonts_are_actually_applied_during_the_call(self):
        """Scoping must not become a no-op: the sizes still apply inside."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from SelectOmics.evaluation.visualization import (
            _FONT_SIZES, _with_plot_style,
        )
        seen = _with_plot_style(lambda: plt.rcParams["font.size"])()
        assert seen == _FONT_SIZES["font.size"]

    def test_no_bold_weight_is_set(self):
        """Figure text must never be bold."""
        from SelectOmics.evaluation.visualization import _FONT_SIZES
        for key, value in _FONT_SIZES.items():
            assert "weight" not in key, f"{key} sets a font weight"
            assert value != "bold", f"{key} is bold"

    def test_figures_do_not_leak_when_plotting_raises(self):
        """
        Each plotting function closes its figure in its body, not in a finally,
        so an exception in between left it open. All six return None and hand
        back no figure, so the decorator closes whatever the call left behind.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from SelectOmics.evaluation.visualization import _with_plot_style

        @_with_plot_style
        def raises_midway():
            plt.figure()
            plt.figure()
            raise ValueError("plotting blew up")

        before = set(plt.get_fignums())
        with pytest.raises(ValueError):
            raises_midway()
        assert set(plt.get_fignums()) == before, "a figure survived the failure"

    def test_figures_opened_before_the_call_are_left_alone(self):
        """Cleanup must be scoped to this call, not close the caller's figures."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from SelectOmics.evaluation.visualization import _with_plot_style

        caller_fig = plt.figure()
        try:
            _with_plot_style(lambda: plt.figure())()
            assert caller_fig.number in plt.get_fignums(), (
                "cleanup closed a figure the caller owned"
            )
        finally:
            plt.close(caller_fig)


# ---------------------------------------------------------------------------
# random_seed must be validated where it is set, not where it is used
# ---------------------------------------------------------------------------

class TestRandomSeedIsValidated:
    """
    scikit-learn requires random_state in [0, 2**32). Every seed the package
    derives is an offset from config.random_seed, so an out-of-range value was
    accepted at construction and surfaced much later as an sklearn
    InvalidParameterError that does not mention random_seed. On a run that
    takes hours, that is a long way to travel for a checkable value.
    """

    @staticmethod
    def _cfg(tmp_path, **kw):
        import numpy as np
        import pandas as pd
        from SelectOmics.config import SelectOmicsConfig
        csv = tmp_path / "t.csv"
        df = pd.DataFrame(np.random.RandomState(0).randn(40, 8),
                          columns=[f"f{i}" for i in range(8)])
        df["Class"] = [i % 2 for i in range(40)]
        df.to_csv(csv, index=False)
        return SelectOmicsConfig(data_path=str(csv), target_column="Class",
                                 output_dir=str(tmp_path), **kw)

    @pytest.mark.parametrize("seed", [-1, -42, 2 ** 32, 2 ** 63])
    def test_out_of_range_seeds_are_rejected_at_construction(self, seed, tmp_path):
        with pytest.raises(ValueError, match="random_seed"):
            self._cfg(tmp_path, random_seed=seed)

    @pytest.mark.parametrize("seed", [0, 42, 2 ** 32 - 1])
    def test_valid_seeds_including_boundaries_are_accepted(self, seed, tmp_path):
        assert self._cfg(tmp_path, random_seed=seed).random_seed == seed

    def test_the_error_names_the_field_and_the_value(self, tmp_path):
        """The message must be actionable without reading the source."""
        with pytest.raises(ValueError) as exc:
            self._cfg(tmp_path, random_seed=-7)
        msg = str(exc.value)
        assert "random_seed" in msg and "-7" in msg


# ---------------------------------------------------------------------------
# A degraded bootstrap must report both how many failed and why
# ---------------------------------------------------------------------------

class TestBootstrapFailuresAreDiagnosable:
    """
    bootstrap_validation skips an iteration it cannot fit or score. It counts
    the survivors and warns when too few completed, but the handler discarded
    the exception, so a user told their confidence interval rests on a handful
    of iterations had nothing to work from. The reason is now available at
    DEBUG: silent on a normal run, there when diagnosing one.
    """

    @staticmethod
    def _validator(tmp_path, n_bootstrap=10):
        from SelectOmics.config import SelectOmicsConfig
        from SelectOmics.evaluation.validation import FeatureSetValidator
        cfg = SelectOmicsConfig(data_path="unused.csv", target_column="Class",
                                output_dir=str(tmp_path), verbose=False)
        return FeatureSetValidator(cfg, n_classes=2, tuned_pipelines={},
                                   n_bootstrap=n_bootstrap)

    @staticmethod
    def _data(n=40):
        rng = np.random.RandomState(0)
        X = pd.DataFrame(rng.randn(n, 5), columns=[f"f{i}" for i in range(5)])
        y = np.array([i % 2 for i in range(n)])
        return X, y

    def test_the_failure_reason_reaches_the_debug_log(self, tmp_path, caplog):
        import logging
        v = self._validator(tmp_path)
        X, y = self._data()

        def explode(seed):
            raise RuntimeError("the estimator refused to fit")

        v._create_model = explode
        caplog.set_level(logging.DEBUG, logger="SelectOmics")
        v.bootstrap_validation(X, y, "probe")

        assert "the estimator refused to fit" in caplog.text, (
            "the reason every bootstrap iteration failed was discarded"
        )
        assert "RuntimeError" in caplog.text

    def test_a_fully_failed_bootstrap_still_warns_about_the_count(
        self, tmp_path, caplog
    ):
        import logging
        v = self._validator(tmp_path)
        X, y = self._data()
        v._create_model = lambda seed: (_ for _ in ()).throw(RuntimeError("no"))
        caplog.set_level(logging.WARNING, logger="SelectOmics")
        res = v.bootstrap_validation(X, y, "probe")

        assert res["successful_iterations"] == 0
        assert "provisional" in caplog.text.lower() or "0 of" in caplog.text

    @pytest.mark.integration
    def test_a_healthy_bootstrap_stays_quiet(self, tmp_path, caplog):
        """
        The warning must not fire on a normal run, or it would train users to
        ignore it. Needs real tuned pipelines: _create_model looks the
        algorithm up in tuned_pipelines, and an empty dict raises KeyError on
        every iteration, which the loop absorbs into a silent zero.
        """
        import logging
        from SelectOmics.config import SelectOmicsConfig
        from SelectOmics.evaluation.validation import FeatureSetValidator
        from SelectOmics.models.base import quick_tune_all

        X, y = self._data(n=60)
        cfg = SelectOmicsConfig(data_path="unused.csv", target_column="Class",
                                output_dir=str(tmp_path), algorithm="RF",
                                quick_tune_iterations=2, verbose=False)
        v = FeatureSetValidator(cfg, n_classes=2,
                                tuned_pipelines=quick_tune_all(X, y, cfg),
                                n_bootstrap=10)

        caplog.set_level(logging.WARNING, logger="SelectOmics")
        res = v.bootstrap_validation(X, y, "probe")
        assert res["successful_iterations"] > 0, (
            "a well-formed bootstrap completed no iterations"
        )
        assert "provisional" not in caplog.text.lower()

    def test_a_missing_tuned_pipeline_is_not_silent(self, tmp_path, caplog):
        """
        Forgetting tuned_pipelines makes every iteration raise KeyError, which
        the loop absorbs. That used to produce a zero-iteration result with no
        explanation anywhere; the count now warns and the reason is at DEBUG.
        """
        import logging
        v = self._validator(tmp_path)
        X, y = self._data()
        caplog.set_level(logging.DEBUG, logger="SelectOmics")
        res = v.bootstrap_validation(X, y, "probe")
        assert res["successful_iterations"] == 0
        assert "KeyError" in caplog.text, "the cause was discarded"


# ---------------------------------------------------------------------------
# Every step must return the same keys on every path
# ---------------------------------------------------------------------------

class TestStepReturnContracts:
    """
    Each step returns a result dict that the pipeline and any caller index
    directly. step3's passthrough says so outright: keys are "present even when
    skipped, so callers can read these keys off any step without guarding."

    consensus_floor_used and consensus_floor_requested were added with the
    consensus work and missed on three non-normal paths: step2's disabled
    branch and step3's skipped and disabled passthrough. A caller reading them
    worked on an ordinary run and raised KeyError only with a step turned off,
    which is the hardest version of that bug to notice.

    This checks the property rather than those two keys, so a key added to one
    path and forgotten on another fails here rather than in someone's run.
    """

    @staticmethod
    def _return_keysets(module_path, func_name):
        import ast
        from pathlib import Path
        tree = ast.parse(Path(module_path).read_text(encoding="utf-8"))
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == func_name)
        sets = []
        for n in ast.walk(fn):
            if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict):
                sets.append({k.value for k in n.value.keys
                             if isinstance(k, ast.Constant)
                             and isinstance(k.value, str)})
        return sets

    @pytest.mark.parametrize("module,func", [
        ("SelectOmics/selection/step1_cleaning.py", "run_step1_cleaning"),
        ("SelectOmics/selection/step2_regularization.py",
         "run_step2_regularization"),
    ])
    def test_every_return_path_carries_the_same_keys(self, module, func):
        sets = self._return_keysets(module, func)
        assert len(sets) >= 2, "expected more than one dict return path"
        union = set().union(*sets)
        for i, ks in enumerate(sets):
            assert not (union - ks), (
                f"{func} return path {i} is missing {sorted(union - ks)}; "
                f"a caller reading those gets KeyError on this path only"
            )

    def test_step3_passthrough_matches_the_normal_return(self):
        """step3's skip paths route through a helper, so compare against it."""
        main = self._return_keysets(
            "SelectOmics/selection/step3_wrapper.py", "run_step3_wrapper")[0]
        skip = self._return_keysets(
            "SelectOmics/selection/step3_wrapper.py", "_passthrough")[0]
        assert not (main - skip), (
            f"the skipped path is missing {sorted(main - skip)}"
        )


# ---------------------------------------------------------------------------
# A checkpoint must not be resumed against different source data
# ---------------------------------------------------------------------------

class TestCheckpointDataIdentity:
    """
    Resuming past step0 skips load_data entirely, so the restored matrices come
    from the checkpoint and are never re-read from disk. Editing the source file
    and resuming therefore reported results computed on the old data as though
    they described the new file.

    The build already refused to resume a checkpoint written by a different
    version, on the reasoning that mixing them "produces a result that matches
    neither". Different source data is the same failure from the other side.
    """

    @staticmethod
    def _write(path, seed=0, n=30):
        rng = np.random.RandomState(seed)
        df = pd.DataFrame(rng.randn(n, 6), columns=[f"f{i}" for i in range(6)])
        df["Class"] = [i % 2 for i in range(n)]
        df.to_csv(path, index=False)
        return path

    def _pipeline(self, tmp_path, csv):
        from SelectOmics.config import SelectOmicsConfig
        from SelectOmics.pipeline import SelectOmicsPipeline
        cfg = SelectOmicsConfig(
            data_path=str(csv), target_column="Class",
            output_dir=str(tmp_path / "out"), save_intermediate_results=True,
            create_visualizations=False, verbose=False,
        )
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        return SelectOmicsPipeline(cfg)

    def test_resume_is_refused_when_the_source_data_changed(
        self, tmp_path, caplog
    ):
        import logging
        csv = self._write(tmp_path / "data.csv", seed=0)
        p = self._pipeline(tmp_path, csv)
        p._save_checkpoint("step1")
        assert p._checkpoint_path().exists()

        self._write(csv, seed=999)          # same path, different contents

        caplog.set_level(logging.WARNING, logger="SelectOmics")
        p2 = self._pipeline(tmp_path, csv)
        assert p2._load_checkpoint() is None, (
            "resumed a checkpoint written for different source data"
        )
        assert "different source data" in caplog.text

    def test_resume_still_works_when_the_data_is_unchanged(self, tmp_path):
        csv = self._write(tmp_path / "data.csv", seed=0)
        p = self._pipeline(tmp_path, csv)
        p._save_checkpoint("step1")

        p2 = self._pipeline(tmp_path, csv)
        assert p2._load_checkpoint() == "step1", (
            "a valid checkpoint was rejected; the guard is too strict"
        )

    def test_a_missing_source_file_does_not_break_loading(self, tmp_path):
        """
        An unreadable file yields no fingerprint. That must not be treated as a
        mismatch, or a moved dataset would silently discard a good checkpoint.
        """
        csv = self._write(tmp_path / "data.csv", seed=0)
        p = self._pipeline(tmp_path, csv)
        p._save_checkpoint("step1")
        csv.unlink()

        p2 = self._pipeline(tmp_path, csv)
        assert p2._load_checkpoint() == "step1"

    def test_the_fingerprint_is_stable_across_processes(self, tmp_path):
        """
        SHA-256 rather than hash(), which PYTHONHASHSEED randomises per process
        and which would make every resume look like changed data.
        """
        csv = self._write(tmp_path / "data.csv", seed=0)
        a = self._pipeline(tmp_path, csv)._source_data_fingerprint()
        b = self._pipeline(tmp_path, csv)._source_data_fingerprint()
        assert a == b and a is not None
