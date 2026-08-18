"""Work-distribution helpers for closed-loop evaluation.

Two strategies. Static sharding partitions the job list up front and is reproducible: a given
rank always gets the same jobs, so a run can be repeated exactly. Claiming hands work out as
ranks become free, which matters when job durations vary enough that a static split leaves one
rank still working while the others idle -- at the cost of a run that no longer assigns the same
job to the same rank twice.
"""

from __future__ import annotations

import os
from pathlib import Path


def shard_items(items: list, rank: int, world_size: int) -> list:
    """Round-robin assignment: rank ``r`` gets indices r, r+world_size, ..."""
    if world_size <= 1:
        return items
    return [items[i] for i in range(rank, len(items), world_size)]


def claim(claim_dir: Path, index: int) -> bool:
    """True iff this process won the race for job ``index``.

    ``O_EXCL`` on a shared filesystem is the whole mechanism: exactly one creator succeeds, so
    no second channel has to exist for a parent to hand work out. It is crash-safe by
    construction -- a worker that dies holds no lock anyone is waiting on -- and the flip side
    is that its unfinished job stays claimed, so a crash costs that job rather than the run.
    """
    path = Path(claim_dir) / f"{index:06d}"
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    except FileNotFoundError:
        # Create the directory only when it is actually missing. Every rank is offered every
        # job, so a mkdir on the steady path would be one wasted metadata round trip per rank
        # per job -- on the shared filesystem this mechanism is built for, against a directory
        # every rank is writing to at once.
        Path(claim_dir).mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return False
    os.write(fd, f"{os.getpid()}\n".encode())
    os.close(fd)
    return True
