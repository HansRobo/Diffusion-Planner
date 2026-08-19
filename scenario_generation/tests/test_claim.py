"""``claim`` is the only thing keeping two workers off the same job, so race it."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from scenario_generation.closed_loop_ddp import claim

WORKERS = 8


def _claim_one(args: tuple[Path, int]) -> bool:
    claim_dir, index = args
    return claim(claim_dir, index)


def _claim_all(args: tuple[Path, int]) -> list[int]:
    claim_dir, n_jobs = args
    return [i for i in range(n_jobs) if claim(claim_dir, i)]


def test_one_index_is_won_by_exactly_one_process(tmp_path):
    claim_dir = tmp_path / "claims"
    with ProcessPoolExecutor(WORKERS) as pool:
        won = list(pool.map(_claim_one, [(claim_dir, 0)] * WORKERS))
    assert sum(won) == 1


def test_the_directory_is_created_by_whoever_gets_there_first(tmp_path):
    # No parent creates it: the workers race on a path whose directory does not exist yet, and
    # the loser of the mkdir must not mistake it for a lost claim.
    claim_dir = tmp_path / "missing" / "claims"
    assert not claim_dir.exists()
    with ProcessPoolExecutor(WORKERS) as pool:
        won = list(pool.map(_claim_one, [(claim_dir, 0)] * WORKERS))
    assert sum(won) == 1


def test_every_job_is_claimed_once_when_all_workers_walk_the_whole_list(tmp_path):
    claim_dir = tmp_path / "claims"
    n_jobs = 64
    with ProcessPoolExecutor(WORKERS) as pool:
        shares = list(pool.map(_claim_all, [(claim_dir, n_jobs)] * WORKERS))
    claimed = [index for share in shares for index in share]
    assert sorted(claimed) == list(range(n_jobs))
