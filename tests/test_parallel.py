"""
Tests for the benchmark parallel-execution helpers.

These cover the scheduling arithmetic and the graceful fallback to sequential
execution, which is the behaviour a restricted environment depends on.
"""
import sys
from pathlib import Path

import pytest

# The benchmarks directory is not part of the installed package.
_BENCH = Path(__file__).resolve().parent.parent / "benchmarks"
if _BENCH.exists() and str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

pytest.importorskip("_parallel", reason="benchmarks/ not present in this install")

from _parallel import (  # noqa: E402
    apply_thread_limits,
    cpu_count,
    describe_plan,
    plan_parallelism,
    run_tasks,
)


# Must be importable at module level: the process pool spawns on Windows.
def _double(x):
    return x * 2


def _boom(x):
    raise ValueError(f"task failed for {x}")


class TestPlanParallelism:
    def test_explicit_workers_splits_cores(self):
        assert plan_parallelism(workers=4, total_cores=24) == (4, 6)

    def test_serial_gets_every_core_for_threads(self):
        assert plan_parallelism(workers=1, total_cores=24) == (1, 24)

    def test_explicit_n_jobs_derives_workers(self):
        assert plan_parallelism(n_jobs=2, total_cores=24) == (12, 2)

    def test_both_explicit_are_honoured(self):
        assert plan_parallelism(workers=3, n_jobs=5, total_cores=24) == (3, 5)

    def test_neither_given_fills_the_machine(self):
        workers, jobs = plan_parallelism(total_cores=24)
        assert workers * jobs <= 24
        assert workers > 1 and jobs >= 1

    def test_single_core_machine_degrades_to_serial(self):
        workers, jobs = plan_parallelism(total_cores=1)
        assert workers == 1 and jobs == 1

    def test_more_workers_than_cores_still_valid(self):
        workers, jobs = plan_parallelism(workers=64, total_cores=4)
        assert workers == 64 and jobs >= 1

    def test_minus_one_workers_means_auto(self):
        workers, jobs = plan_parallelism(workers=-1, total_cores=8)
        assert workers >= 1 and jobs >= 1


class TestRunTasks:
    def test_sequential_path(self):
        results = run_tasks(_double, [(1,), (2,), (3,)], workers=1, n_jobs=1)
        assert sorted(results) == [2, 4, 6]

    def test_empty_task_list(self):
        assert run_tasks(_double, [], workers=4, n_jobs=1) == []

    def test_on_result_callback_fires_once_per_task(self):
        seen = []
        run_tasks(_double, [(1,), (2,)], workers=1, n_jobs=1,
                  on_result=seen.append)
        assert sorted(seen) == [2, 4]

    @pytest.mark.slow
    def test_parallel_matches_sequential(self):
        args = [(i,) for i in range(8)]
        serial = sorted(run_tasks(_double, args, workers=1, n_jobs=1))
        parallel = sorted(run_tasks(_double, args, workers=2, n_jobs=1))
        assert serial == parallel

    @pytest.mark.slow
    def test_task_exception_propagates_when_asked(self):
        """
        A real failure must surface, not be silently swallowed by the pool.

        This is on_error='raise'. The default is 'skip', because a benchmark
        that checkpoints per trial should not lose every remaining trial to one
        failure -- but 'skip' logs at ERROR level, so neither mode is silent.
        """
        with pytest.raises(ValueError, match="task failed"):
            run_tasks(_boom, [(1,)], workers=2, n_jobs=1, on_error="raise")

    @pytest.mark.slow
    def test_default_skips_and_completes_the_rest(self):
        """One failing task must not cost the others."""
        out = run_tasks(_boom, [(1,)], workers=2, n_jobs=1)
        assert out == []

    def test_invalid_on_error_is_rejected(self):
        with pytest.raises(ValueError, match="on_error"):
            run_tasks(_double, [(1,)], workers=1, n_jobs=1, on_error="ignore")

    def test_falls_back_to_sequential_when_pool_unavailable(self, monkeypatch):
        """
        A machine that cannot host a process pool must still finish the work.

        This is the same contract as falling back from GPU to CPU: slower, not
        different, and never an error.
        """
        import _parallel

        def _refuse(*args, **kwargs):
            raise OSError("process pools are not permitted here")

        monkeypatch.setattr(_parallel, "ProcessPoolExecutor", _refuse)

        results = run_tasks(_double, [(1,), (2,), (3,)], workers=4, n_jobs=1)
        assert sorted(results) == [2, 4, 6]


class TestThreadLimits:
    def test_apply_thread_limits_sets_env(self, monkeypatch):
        import os
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        apply_thread_limits(3)
        assert os.environ["OMP_NUM_THREADS"] == "3"

    def test_thread_limits_floor_at_one(self, monkeypatch):
        import os
        apply_thread_limits(0)
        assert os.environ["OMP_NUM_THREADS"] == "1"


class TestDescribePlan:
    def test_mentions_sequential_when_single_worker(self):
        assert "sequential" in describe_plan(1, 8)

    def test_mentions_worker_count_when_parallel(self):
        assert "4 worker processes" in describe_plan(4, 6)

    def test_cpu_count_is_positive(self):
        assert cpu_count() >= 1
