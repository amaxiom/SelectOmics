"""
Shared fixtures for the SelectOmics test suite.
"""
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Synthetic dataset fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def binary_dataset(tmp_path_factory):
    """80-sample, 30-feature binary classification CSV."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=80,
        n_features=30,
        n_informative=8,
        n_redundant=5,
        n_classes=2,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(30)])
    df["Class"] = y

    out_dir = tmp_path_factory.mktemp("data")
    csv_path = out_dir / "binary.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def fast_config(binary_dataset, tmp_path):
    """Minimal config that runs quickly: 1 model, no plots, no intermediate saves."""
    return {
        "data_path": binary_dataset,
        "target_column": "Class",
        "algorithm": "XGB",
        "n_consensus_models": 1,
        "n_bootstrap": 10,
        "random_seed": 42,
        "output_dir": str(tmp_path / "results"),
        "verbose": False,
        "save_intermediate_results": False,
        "create_visualizations": False,
        "quick_tune_iterations": 5,
        "enable_step3": False,
    }


# ---------------------------------------------------------------------------
# Global logging state must not leak between tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _restore_package_logger():
    """
    Snapshot and restore the SelectOmics logger around every test.

    enable_logging() is public API: it sets the level on the package logger
    and attaches a handler, and those changes are process-global and
    permanent. A test that calls it to quiet output leaves every later test
    running against a different logger configuration than it would see alone.

    That produced a real, deterministic failure rather than a flake.
    test_config.py calls enable_logging('WARNING') and never restores it,
    which pinned the package logger at WARNING; test_suggest_logs_its_rationale
    then raised only the *root* level via caplog, so its INFO records were
    filtered at the package logger before they could propagate, and it
    asserted against an empty capture. Ordering made it deterministic:
    test_config collects before test_config_paths.

    Restoring here fixes the whole class, not just that one pair, so a future
    test calling enable_logging cannot silently break a distant assertion.
    """
    import logging

    pkg = logging.getLogger("SelectOmics")
    saved_level = pkg.level
    saved_handlers = list(pkg.handlers)
    saved_propagate = pkg.propagate
    try:
        yield
    finally:
        pkg.setLevel(saved_level)
        pkg.handlers[:] = saved_handlers
        pkg.propagate = saved_propagate
