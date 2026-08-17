"""The pool that renders per-step figures.

Matplotlib builds each figure in Python, so the GIL serialises a thread pool; worker processes
each hold their own interpreter and overlap for real.

Callers must be spawn-safe: an importable callable, picklable arguments, a
``if __name__ == "__main__":`` guard, and no mutation of what was submitted.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import Executor, ProcessPoolExecutor


def _init_worker() -> None:
    # torch sizes its intra-op pool from the machine's core count, so N workers each grab N_cores
    # threads and thrash.
    import ctypes
    import signal

    import torch

    torch.set_num_threads(1)
    # A renderer must not outlive the process it renders for. The pool's shutdown does not always
    # run: a hard per-scenario deadline exits from a C thread, and a kill or the OOM killer runs
    # nothing at all -- after which the child blocks on its call queue forever, reparented to init,
    # still holding its RSS. PR_SET_PDEATHSIG fires on the death of the thread that created the
    # child, and ProcessPoolExecutor spawns from whichever thread called submit, so this is only
    # equivalent to process death while that is the main thread.
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


def render_pool(workers: int = 1) -> Executor:
    """A pool of ``workers`` renderers; anything below 1 is one worker.

    Spawn, not fork: the caller may hold a CUDA context. Workers start on the first submit, so a
    run that draws nothing spawns nothing. Each worker costs ~780 MB of RSS.
    """
    return ProcessPoolExecutor(
        max_workers=max(1, workers),
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
    )
