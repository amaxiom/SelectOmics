"""
Smoke tests for every figure the pipeline produces.

Plotting code is easy to leave untested and easy to break: it runs only when
``create_visualizations`` is on, it is the last thing to execute in a step, and
a failure there wastes the whole run that preceded it. These tests do not check
that the figures look right, which no automated test can. They check that each
function accepts the shapes the pipeline actually hands it, writes a non-empty
file, and closes its figures, for binary and multiclass alike.

Matplotlib runs on the Agg backend, so nothing opens a window.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from SelectOmics.config import SelectOmicsConfig
from SelectOmics.evaluation.visualization import (
    plot_auc_boxplots,
    plot_learning_curve_and_roc,
    plot_per_class_roc_and_confusion,
    plot_pipeline_summary,
    plot_roc_curves,
    plot_validation_comparison,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg(tmp_path):
    return SelectOmicsConfig(
        data_path="unused.csv", target_column="Class",
        output_dir=str(tmp_path), create_visualizations=True,
    )


def _proba(n, k, seed=0):
    """Row-stochastic probability matrix, as predict_proba returns."""
    rng = np.random.RandomState(seed)
    p = rng.rand(n, k) + 0.01
    return p / p.sum(axis=1, keepdims=True)


def _labels(n, k, seed=0):
    """Every class present at least twice, which the plots assume."""
    base = np.repeat(np.arange(k), 2)
    rest = np.random.RandomState(seed).randint(0, k, max(0, n - len(base)))
    return np.concatenate([base, rest])[:n]


def _written(path):
    return path.exists() and path.stat().st_size > 0


@pytest.fixture(autouse=True)
def _no_leaked_figures():
    """A plot helper that forgets to close leaks memory across a long run."""
    plt.close("all")
    yield
    assert plt.get_fignums() == [], "figure left open"
    plt.close("all")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_classes", [2, 4])
def test_plot_roc_curves(cfg, tmp_path, n_classes):
    n = 40
    y = _labels(n, n_classes)
    models = {
        f"Model-{i}": {"oof_proba": _proba(n, n_classes, seed=i),
                       "mean_auc": 0.8 + 0.01 * i}
        for i in range(2)
    }
    out = tmp_path / f"roc_{n_classes}.png"
    plot_roc_curves(models, y, n_classes, [str(c) for c in range(n_classes)],
                    cfg, "ROC", out)
    assert _written(out)


def test_plot_auc_boxplots(cfg, tmp_path):
    aucs = {"Ref-XGB": [0.81, 0.83, 0.79, 0.85],
            "Consensus-1": [0.86, 0.88, 0.84, 0.87]}
    out = tmp_path / "box.png"
    plot_auc_boxplots("Step 2", aucs, cfg, out)
    assert _written(out)


def test_plot_auc_boxplots_single_model(cfg, tmp_path):
    """Step 0 has exactly one model; a single box must not break the layout."""
    out = tmp_path / "box1.png"
    plot_auc_boxplots("Step 0", {"Ref-XGB": [0.8, 0.82, 0.78]}, cfg, out)
    assert _written(out)


def test_plot_pipeline_summary(cfg, tmp_path):
    df = pd.DataFrame({
        "Step": ["Step 0 (Reference)", "Step 1 (Data Cleaning)",
                 "Step 2 (Regularization)", "Step 3 (Wrappers)"],
        "Features": [2000, 532, 21, 12],
        "XGB_CV_AUC": [0.857, 0.843, 0.858, 0.870],
    })
    out = tmp_path / "summary.png"
    plot_pipeline_summary(df, "XGB", cfg, out)
    assert _written(out)


@pytest.mark.parametrize("n_classes", [2, 3])
def test_plot_per_class_roc_and_confusion(cfg, tmp_path, n_classes):
    from SelectOmics.evaluation.metrics import binarize_labels
    n = 36
    y = _labels(n, n_classes)
    proba = _proba(n, n_classes)
    names = [f"C{i}" for i in range(n_classes)]
    cm = pd.DataFrame(
        np.random.RandomState(0).randint(0, 8, (n_classes, n_classes)),
        index=names, columns=names,
    )
    out = tmp_path / f"roc_cm_{n_classes}.png"
    plot_per_class_roc_and_confusion(
        y, binarize_labels(y, n_classes), proba, cm, names, "XGB", cfg, out)
    assert _written(out)


def test_plot_validation_comparison(cfg, tmp_path):
    def one(n_feat, auc):
        """Shape returned by FeatureSetValidator.validate_feature_set."""
        return {
            "n_features": n_feat,
            "stratified_cv": {"mean_auc": auc, "std_auc": 0.03},
            "leave_one_out": {"mean_auc": auc - 0.01, "std_auc": 0.02},
            "bootstrap": {"mean_auc": auc - 0.02, "std_auc": 0.025,
                          "confidence_interval": (auc - 0.09, auc + 0.05)},
        }
    results = {"All features": one(2000, 0.856), "Step 2": one(61, 0.886)}
    out = tmp_path / "valcmp.png"
    plot_validation_comparison(results, cfg, out)
    assert _written(out)


def test_plot_validation_comparison_zero_auc(cfg, tmp_path):
    """Panel 4 divides by mean_auc; a degenerate 0.0 must not raise."""
    zero = {
        "n_features": 10,
        "stratified_cv": {"mean_auc": 0.0, "std_auc": 0.0},
        "leave_one_out": {"mean_auc": 0.0, "std_auc": 0.0},
        "bootstrap": {"mean_auc": 0.0, "std_auc": 0.0,
                      "confidence_interval": (0.0, 0.0)},
    }
    out = tmp_path / "valcmp_zero.png"
    plot_validation_comparison({"degenerate": zero}, cfg, out)
    assert _written(out)


@pytest.mark.integration
@pytest.mark.parametrize("n_classes", [2, 3])
def test_plot_learning_curve_and_roc(cfg, tmp_path, n_classes):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.linear_model import LogisticRegression

    rng = np.random.RandomState(0)
    n_tr, n_te, p = 60, 24, 8
    X_tr = pd.DataFrame(rng.randn(n_tr, p), columns=[f"f{i}" for i in range(p)])
    y_tr = _labels(n_tr, n_classes)
    y_te = _labels(n_te, n_classes, seed=1)
    model = Pipeline([("s", MinMaxScaler()),
                      ("clf", LogisticRegression(max_iter=200))]).fit(X_tr, y_tr)
    training = {"mean_auc": 0.84, "std_auc": 0.03,
                "oof_proba": _proba(n_tr, n_classes)}
    out = tmp_path / f"lc_{n_classes}.png"
    plot_learning_curve_and_roc(
        model, X_tr, y_tr, y_te, _proba(n_te, n_classes, seed=2),
        0.83, training, "LR", n_classes, cfg, out)
    assert _written(out)


def test_save_path_none_does_not_write(cfg):
    """save_path=None must still run cleanly and close its figure."""
    plot_auc_boxplots("Step 2", {"A": [0.8, 0.9]}, cfg, None)


@pytest.mark.parametrize("fmt", ["png", "svg", "pdf"])
def test_configured_plot_format_is_honoured(tmp_path, fmt):
    """config.plot_format decides the extension, not the caller's suffix."""
    c = SelectOmicsConfig(data_path="u.csv", target_column="Class",
                          output_dir=str(tmp_path), plot_format=fmt)
    out = tmp_path / f"fmt_{fmt}"
    plot_auc_boxplots("Step 2", {"A": [0.8, 0.9, 0.85]}, c, out)
    written = list(tmp_path.glob(f"fmt_{fmt}*"))
    assert written, f"nothing written for plot_format={fmt}"
    assert written[0].stat().st_size > 0
