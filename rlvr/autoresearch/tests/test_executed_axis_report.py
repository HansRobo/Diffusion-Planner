"""The executed-axis reader is the instrument a multi-day anchor run is judged on, and its
chain grouping has one non-obvious rule: the frame number is only unique within a *segment*
of a log, so a key built from the parent dir alone splices two segments into one chain.  These
pin that, the stride rule, and the sign convention that makes delta / t / z agree.
"""

import math
from pathlib import Path

import numpy as np

from rlvr.autoresearch.tools.executed_axis_report import (
    _chain_diffs,
    chain_runs,
    epoch_rows,
    paired,
)


def _paths(log: str, segment: str, frames):
    return [f"/data/{log}/{log}_{segment}_{f:08d}.npz" for f in frames]


def test_chain_runs_keeps_consecutive_frames_and_drops_short_and_strided():
    # 5 consecutive, a stride jump, then 3 consecutive (too short to difference).
    paths = np.asarray(_paths("09-42-22", "00000000", [10, 11, 12, 13, 14, 34, 35, 36]))
    runs = chain_runs(paths)
    assert len(runs) == 1
    assert runs[0].tolist() == [0, 1, 2, 3, 4]


def test_chain_runs_does_not_splice_two_segments_sharing_frame_numbers():
    # The same frame numbers recur across segments of one log (1,739 times in the real
    # corpus).  A dirname-only key would sort them interleaved and emit garbage chains.
    paths = np.asarray(
        _paths("09-42-22", "00000000", [10, 11, 12, 13])
        + _paths("09-42-22", "00000001", [10, 11, 12, 13])
    )
    runs = chain_runs(paths)
    assert sorted(run.tolist() for run in runs) == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_chain_runs_separates_logs():
    paths = np.asarray(
        _paths("09-42-22", "00000000", [1, 2, 3, 4])
        + _paths("13-49-19", "00000000", [1, 2, 3, 4])
    )
    assert len(chain_runs(paths)) == 2


def test_chain_diffs_are_differenced_within_chains_only():
    paths = np.asarray(_paths("a", "0", [1, 2, 3, 4]) + _paths("a", "0", [40, 41, 42, 43]))
    runs = chain_runs(paths)
    values = np.asarray([0.0, 1.0, 2.0, 3.0, 100.0, 101.0, 102.0, 103.0])
    # Six steps of 1.0, and never the 97.0 jump between the two chains.
    d1 = _chain_diffs(values, runs, 1)
    assert d1.size == 6
    assert np.allclose(d1, 1.0)
    # A constant-slope ramp has no jerk.
    assert np.allclose(_chain_diffs(values, runs, 3), 0.0)


def test_chain_jitter_recovers_a_halved_deviation():
    frames = list(range(200))
    paths = np.asarray(_paths("a", "0", frames))
    runs = chain_runs(paths)
    rng = np.random.default_rng(0)
    human = np.cumsum(rng.normal(0, 4e-4, len(frames)))
    deviation = rng.normal(0, 3e-3, len(frames))
    rms = lambda x: float(np.sqrt((x**2).mean()))
    base = rms(_chain_diffs(human + deviation, runs, 1))
    cand = rms(_chain_diffs(human + 0.5 * deviation, runs, 1))
    floor = rms(_chain_diffs(human, runs, 1))
    # Halving a white deviation on top of a smooth human halves the d1 excess over it.
    closed = (base - cand) / (base - floor)
    assert 0.45 < closed < 0.55


def test_paired_signs_agree_with_the_direction_of_the_move():
    down = np.asarray([-1.0, -1.0, -1.0, -1.0, 1.0])
    mean, t, z = paired(down)
    assert mean < 0 and t < 0 and z < 0
    mean, t, z = paired(-down)
    assert mean > 0 and t > 0 and z > 0
    # 4 of 5 moved one way: |1 - 5/2| / sqrt(5/4).
    assert math.isclose(abs(z), 1.5 / math.sqrt(5 / 4.0), rel_tol=1e-9)


def test_an_arm_resumed_into_a_second_run_still_baselines_on_the_first(tmp_path):
    """A run names its directory after its start, and a resume makes a second one.

    Reading the arm has to see through both, earliest first, or the baseline
    silently becomes the checkpoint the run resumed from.
    """

    for stamp, epochs in (("20260101-000000_dual", (0, 2)), ("20260102-000000_dual", (3,))):
        for epoch in epochs:
            rows = tmp_path / stamp / f"eval_epoch_{epoch:03d}"
            rows.mkdir(parents=True)
            (rows / "scenes.json").write_text("[]")
    found = epoch_rows(str(tmp_path))
    assert [Path(p).parent.name for p in found] == [
        "eval_epoch_000",
        "eval_epoch_002",
        "eval_epoch_003",
    ]
    # A single run directory is the same read, one level up.
    assert len(epoch_rows(str(tmp_path / "20260101-000000_dual"))) == 2
