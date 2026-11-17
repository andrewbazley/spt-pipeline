"""Shared efficiency helpers for the ergem fork: worker-count resolution,
progress reporting (tqdm with a plain fallback), and parallel map over
processes (CPU-bound movie ingest) or threads (I/O-bound file reads and
subprocess launches).

Design goal: this package is meant for large datasets (1000s of movies), so
every stage should parallelize and report progress at a high level without
per-file spam.
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterable, List, Optional, Sequence

DEFAULT_N_JOBS = 8


def resolve_n_jobs(requested: Optional[int]) -> int:
    """Clamp requested workers to [1, cpu_count]; default to DEFAULT_N_JOBS."""
    cpu = os.cpu_count() or 1
    if requested is None:
        requested = DEFAULT_N_JOBS
    if requested <= 0:
        return max(1, cpu)
    return max(1, min(int(requested), cpu))


def _get_tqdm():
    try:
        from tqdm.auto import tqdm  # type: ignore
        return tqdm
    except Exception:
        return None


class _FallbackBar:
    """Minimal progress reporter used when tqdm is unavailable."""

    def __init__(self, total: Optional[int], desc: str):
        self.total = total or 0
        self.desc = desc
        self.n = 0
        self._step = max(1, (self.total // 20) or 1)

    def update(self, k: int = 1):
        self.n += k
        if self.total and (self.n % self._step == 0 or self.n == self.total):
            pct = 100.0 * self.n / self.total
            print(f"  {self.desc}: {self.n}/{self.total} ({pct:4.0f}%)", file=sys.stderr, flush=True)

    def close(self):
        if self.total:
            print(f"  {self.desc}: done ({self.n}/{self.total})", file=sys.stderr, flush=True)


def make_bar(total: Optional[int], desc: str, enable: bool = True):
    if not enable:
        return _FallbackBar(0, desc)
    tqdm = _get_tqdm()
    if tqdm is not None:
        return tqdm(total=total, desc=desc, unit="movie", dynamic_ncols=True, leave=True)
    return _FallbackBar(total, desc)


def pmap(worker: Callable[[Any], Any], jobs: Sequence[Any], n_jobs: int,
         desc: str = "processing", mode: str = "process",
         progress: bool = True) -> List[Any]:
    """Parallel map with a progress bar. ``worker`` must be a top-level callable
    (picklable) for ``mode='process'``. Returns results in completion order.
    """
    jobs = list(jobs)
    n = len(jobs)
    n_jobs = resolve_n_jobs(n_jobs)
    if n == 0:
        return []
    if n_jobs == 1 or n == 1:
        bar = make_bar(n, desc, progress)
        out = []
        for j in jobs:
            out.append(worker(j))
            bar.update(1)
        bar.close()
        return out
    Executor = ProcessPoolExecutor if mode == "process" else ThreadPoolExecutor
    kwargs = {}
    if mode == "process":
        import multiprocessing as mp
        kwargs["mp_context"] = mp.get_context("spawn")
    results: List[Any] = []
    bar = make_bar(n, desc, progress)
    with Executor(max_workers=n_jobs, **kwargs) as ex:
        futs = [ex.submit(worker, j) for j in jobs]
        for fut in as_completed(futs):
            results.append(fut.result())
            bar.update(1)
    bar.close()
    return results
