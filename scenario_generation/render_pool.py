"""The pool that renders per-step figures.

Matplotlib builds each figure in Python, so the GIL serialises a thread pool; worker processes
each hold their own interpreter and overlap for real.

Callers must be spawn-safe: an importable callable, picklable arguments, a
``if __name__ == "__main__":`` guard, and no mutation of what was submitted.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from typing import Protocol

# Undrained-future cap shared by every caller that streams render_pool frames straight into an
# ffmpeg pipe instead of writing PNGs (see drain_oldest_frame): a ProcessPoolExecutor buffers
# each completed result on the parent side whether or not .result() has been called, so an
# unbounded backlog holds one RGBA frame (~4 MB at 1000x1000) in memory per unconsumed step.
MAX_INFLIGHT_FRAMES = 8


class _FrameWriter(Protocol):
    def write_frame(self, frame_bytes: bytes, width: int, height: int, pix_fmt: str = ...) -> None: ...


def drain_oldest_frame(
    pending: list[Future], writer: _FrameWriter, cap: int = MAX_INFLIGHT_FRAMES
) -> None:
    """Write the oldest buffered frame into ``writer`` once more than ``cap`` are unconsumed.

    Call this after every ``pending.append(pool.submit(...))`` in a streaming step loop, so the
    backlog is drained in step order as it grows instead of only after the whole rollout ends.
    """
    if len(pending) > cap:
        writer.write_frame(*pending.pop(0).result())


def _pin_worker_threads() -> None:
    # torch sizes its intra-op pool from the machine's core count, so N workers each grab N_cores
    # threads and thrash.
    import torch

    torch.set_num_threads(1)


def render_pool(workers: int = 1) -> Executor:
    """A pool of ``workers`` renderers; anything below 1 is one worker.

    Spawn, not fork: the caller may hold a CUDA context. Workers start on the first submit, so a
    run that draws nothing spawns nothing. Each worker costs ~780 MB of RSS.
    """
    return ProcessPoolExecutor(
        max_workers=max(1, workers),
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_pin_worker_threads,
    )
