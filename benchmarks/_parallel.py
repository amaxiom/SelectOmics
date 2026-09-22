"""
benchmarks/_parallel.py

Shared parallel-execution helpers for the SelectOmics benchmark harnesses.

Both harnesses originally pinned every estimator to ``n_jobs=1`` on the grounds
that this made runs bit-reproducible. Measured against scikit-learn 1.7 and
xgboost 3.3, that is not what ``n_jobs`` controls: XGBoost (``tree_method``
'hist'), RandomForest, ``permutation_importance``, and RFECV all return
bit-identical output at ``n_jobs=1`` and ``n_jobs=-1``, because each is seeded
independently of how the work is split across threads. What *does* change
results is moving XGBoost to the GPU (``device='cuda'``), which uses a
different split-finding implementation and drifts on the order of 1e-2 in
feature importances. GPU is therefore never enabled implicitly.

Two independent axes of parallelism are available:

- **Threads inside each estimator** (``n_jobs``). Cheap, reproducible, roughly
  3x on a many-core machine.
- **Processes across work units** (folds, scenarios, methods). Scales further
  and is reproducible by construction, since each unit is self-contained.

They multiply, so oversubscription is the thing to avoid: with ``W`` worker
processes each estimator should get about ``cores / W`` threads.
``plan_parallelism`` works that split out.

Reproducibility contract
------------------------
Every result produced through this module is identical to the sequential
``workers=1, n_jobs=1`` result, with one exception that must be requested
explicitly: ``use_gpu=True``.

That contract is what makes the fallback safe. If a process pool cannot be
created -- restricted sandbox, container without the needed syscalls, a spawn
context that cannot re-import the parent module -- ``run_tasks`` logs a warning
and finishes the work sequentially instead of failing, exactly as the pipeline
falls back from GPU to CPU when CUDA is unavailable. The run gets slower; it
does not get different.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Everything a process pool can raise on a machine that cannot host one.
# PermissionError and OSError cover sandboxes and containers that forbid the
# needed syscalls; ImportError covers a child that cannot re-import the parent
# module; BrokenProcessPool covers a worker dying during start-up.
_POOL_FAILURES = (
    BrokenProcessPool,
    ImportError,
    OSError,          # covers PermissionError and BlockingIOError
    RuntimeError,     # raised when spawning is not permitted
    AttributeError,   # unpicklable task function in a spawn context
)

# Environment variables that cap thread pools inside native libraries. Each
# worker process sets these before importing numpy so that BLAS does not spawn
# a full set of threads per worker on top of the estimator's own pool.
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def cpu_count() -> int:
    """Return the usable core count, falling back to 1 when undetectable."""
    return os.cpu_count() or 1


def plan_parallelism(
    workers: Optional[int] = None,
    n_jobs: Optional[int] = None,
    total_cores: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Split the available cores between worker processes and estimator threads.

    Parameters
    ----------
    workers : int or None
        Number of worker processes. ``None`` or ``-1`` requests one per core,
        capped so that each still gets at least one thread. ``1`` runs
        everything in the calling process, which is the sequential path.
    n_jobs : int or None
        Threads per estimator. ``None`` divides the remaining cores evenly
        across workers. ``-1`` gives every worker all cores, which is only
        sensible when ``workers=1``.
    total_cores : int or None
        Override the detected core count. Intended for tests.

    Returns
    -------
    (workers, n_jobs)
        Both resolved to concrete positive integers.

    Examples
    --------
    >>> plan_parallelism(workers=4, n_jobs=None, total_cores=24)
    (4, 6)
    >>> plan_parallelism(workers=1, n_jobs=None, total_cores=24)
    (1, 24)
    >>> plan_parallelism(workers=None, n_jobs=2, total_cores=24)
    (12, 2)
    """
    cores = total_cores if total_cores is not None else cpu_count()

    if workers is not None and workers not in (-1, 0):
        resolved_workers = max(1, int(workers))
        if n_jobs is None:
            resolved_jobs = max(1, cores // resolved_workers)
        else:
            resolved_jobs = cores if n_jobs in (-1, 0) else max(1, int(n_jobs))
        return resolved_workers, resolved_jobs

    # workers unspecified: choose it from n_jobs so the product fills the machine.
    if n_jobs is not None and n_jobs not in (-1, 0):
        resolved_jobs = max(1, int(n_jobs))
        resolved_workers = max(1, cores // resolved_jobs)
        return resolved_workers, resolved_jobs

    # Neither given: prefer processes over threads. Process-level parallelism
    # scales better here because the per-unit work is long and independent,
    # while thread scaling inside a single small fit saturates early.
    resolved_workers = max(1, cores // 2)
    resolved_jobs = max(1, cores // resolved_workers)
    return resolved_workers, resolved_jobs


def apply_thread_limits(n_threads: int) -> None:
    """
    Cap native thread pools for the current process.

    Must run before numpy is imported to take full effect, which is why it is
    called from the pool initializer rather than at task time.
    """
    value = str(max(1, int(n_threads)))
    for var in _THREAD_ENV_VARS:
        os.environ[var] = value


_WORKER_CONTEXT: Dict[str, Any] = {}


def _initializer(n_threads: int, context: Optional[Dict[str, Any]]) -> None:
    """Pool initializer: set thread caps once, then stash shared read-only data."""
    apply_thread_limits(n_threads)
    if context:
        _WORKER_CONTEXT.update(context)


def worker_context() -> Dict[str, Any]:
    """Return the read-only payload handed to this worker at start-up.

    Tasks read their large inputs from here rather than receiving them as
    arguments, so a feature matrix is pickled once per worker instead of once
    per task.
    """
    return _WORKER_CONTEXT


def run_tasks(
    task_fn: Callable[..., Any],
    task_args: Iterable[Tuple],
    workers: int,
    n_jobs: int,
    context: Optional[Dict[str, Any]] = None,
    on_result: Optional[Callable[[Any], None]] = None,
    on_error: str = "skip",
) -> List[Any]:
    """
    Execute independent tasks, in parallel when asked to.

    Results are returned in completion order, not submission order. Every
    caller here aggregates by explicit keys, so ordering carries no meaning;
    if you add a caller that depends on order, sort the results yourself.

    ``workers=1`` runs everything in the calling process with no pool at all.
    That path exists so notebooks, debuggers, and reproducibility checks can
    bypass multiprocessing entirely.

    Parameters
    ----------
    task_fn : callable
        Executed once per entry in ``task_args``. Must be importable at module
        level: Windows spawns workers rather than forking, so closures and
        locally-defined functions cannot be sent to them.
    task_args : iterable of tuple
        Positional arguments for each call.
    workers : int
        Worker process count. 1 means run inline.
    n_jobs : int
        Thread cap applied inside each worker.
    context : dict or None
        Read-only payload broadcast to every worker once, retrievable with
        ``worker_context()``. Use it for feature matrices.
    on_result : callable or None
        Invoked in the parent process as each result arrives. This is where
        incremental checkpoint writes belong: doing them here keeps a single
        writer and avoids interleaved appends corrupting the CSV.
    on_error : {'skip', 'raise'}
        What a failing task does to the run.

        ``'raise'`` re-raises, so a real failure surfaces immediately. Correct
        for short jobs and for tests, where a swallowed exception hides a bug.

        ``'skip'`` (default) logs the failure at ERROR level and continues with
        the remaining tasks. Correct for benchmarks, which run for hours and
        checkpoint per trial: aborting discards every remaining trial and every
        later layer for the sake of one. Observed on the LGG CNV layer, where a
        single trial in fold 5 raised and took folds 5-9 plus the entire merged
        layer with it. Skipped tasks are simply absent from the checkpoint, so
        the caller's resume logic retries them on the next run.

        Neither mode is silent: 'skip' always logs what failed and why.

    Returns
    -------
    list
        Every task's return value.
    """
    if on_error not in ("skip", "raise"):
        raise ValueError(
            f"on_error must be 'skip' or 'raise', got {on_error!r}."
        )
    task_args = list(task_args)
    if not task_args:
        return []

    if workers <= 1:
        return _run_sequential(task_fn, task_args, n_jobs, context,
                               on_result, on_error)

    # Which submissions produced a result, so a pool-level failure can resume
    # rather than re-running work already written by on_result.
    completed: set = set()
    results = []
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initializer,
            initargs=(n_jobs, context),
        ) as pool:
            futures = {pool.submit(task_fn, *args): i
                       for i, args in enumerate(task_args)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                except _POOL_FAILURES:
                    # The pool itself is gone; the outer handler deals with it.
                    raise
                except Exception as exc:                     # noqa: BLE001
                    if on_error == "raise":
                        raise
                    logger.error(
                        "Task %d of %d failed and was skipped (%s: %s). The run "
                        "continues; re-run to retry it.",
                        idx + 1, len(task_args), type(exc).__name__, exc,
                    )
                    continue
                completed.add(idx)
                if on_result is not None:
                    on_result(result)
                results.append(result)
        return results

    except _POOL_FAILURES as exc:
        # Process pools are not available everywhere: locked-down sandboxes,
        # some container and HPC configurations, and environments where the
        # spawned child cannot re-import the parent module all fail here.  None
        # of that should end a long benchmark run, and the sequential path
        # produces identical results, so degrade rather than abort -- the same
        # contract as falling back from GPU to CPU.
        logger.warning(
            "Parallel execution unavailable (%s: %s). Falling back to "
            "sequential execution with n_jobs=%s; results are unchanged, only "
            "slower.",
            type(exc).__name__, exc, n_jobs,
        )
        # Only what has not already been handed to on_result. Re-running the
        # full list would append a second copy of every row already written.
        remaining = [a for i, a in enumerate(task_args) if i not in completed]
        return results + _run_sequential(
            task_fn, remaining, n_jobs, context, on_result, on_error)


def _run_sequential(
    task_fn: Callable[..., Any],
    task_args: List[Tuple],
    n_jobs: int,
    context: Optional[Dict[str, Any]],
    on_result: Optional[Callable[[Any], None]],
    on_error: str = "skip",
) -> List[Any]:
    """Run every task in the calling process. Identical results, no pool."""
    apply_thread_limits(n_jobs)
    if context:
        _WORKER_CONTEXT.update(context)
    results = []
    for i, args in enumerate(task_args):
        try:
            result = task_fn(*args)
        except Exception as exc:                             # noqa: BLE001
            if on_error == "raise":
                raise
            logger.error(
                "Task %d of %d failed and was skipped (%s: %s). The run "
                "continues; re-run to retry it.",
                i + 1, len(task_args), type(exc).__name__, exc,
            )
            continue
        if on_result is not None:
            on_result(result)
        results.append(result)
    return results


def describe_plan(workers: int, n_jobs: int, label: str = "") -> str:
    """Return a one-line human-readable summary of the resolved plan."""
    cores = cpu_count()
    mode = "sequential" if workers == 1 else f"{workers} worker processes"
    return (
        f"{label}parallelism: {mode} x {n_jobs} thread(s) per estimator "
        f"= {workers * n_jobs} of {cores} cores"
    )
