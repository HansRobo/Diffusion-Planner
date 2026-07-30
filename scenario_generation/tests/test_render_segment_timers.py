"""``render_segment``'s stage timers.

The stage breakdown is the only way to answer "which part of a closed-loop run took the time".
``run_segments_batched`` has had it since the mining pipeline was instrumented; ``render_segment``
-- the path every closed-loop *evaluation* actually takes -- collected a few blocks into a Timers
it then threw away, so a slow run gave no clue where the time went.

These tests pin the contract:
- the CALLER's Timers is the one that gets filled (so a driver can report a whole-run breakdown),
- every stage of the step loop lands in a named block, i.e. the instrumentation has no blind spot,
- and passing no Timers leaves the returned metrics exactly as they were.
"""

import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scenario_generation.perf_timer import Timers
from scenario_generation.reproducer_rollout import render_segment
from scenario_generation.route_timeline import RouteTimeline

N_FRAMES = 12
STEP_M = 1.0  # ~10 m/s along +x: moves, so the run does not terminate as "stuck"
N_NEIGHBORS = 8
FUTURE_LEN = 8
EGO_SHAPE = np.array([4.76, 2.0, 1.5], dtype=np.float32)

# Every stage of the step loop, excluding ``draw`` (rendering is off in these runs) and
# ``cursor_step`` / ``timeline_*`` (nested inside ``input_build`` and the timeline, i.e. a
# breakdown OF a block below rather than a block of its own).
TOP_LEVEL_STAGES = {
    "input_build",
    "to_torch",
    "model_forward",
    "score",
    "advance",
    "rollout_jsonl",
    "finalize",
}


def _frame_npz(tmp_path, i: int):
    """One recorded frame with every key the reproducer consumes."""
    past = np.zeros((31, 3), dtype=np.float32)
    past[:, 0] = (np.arange(31) - 30) * STEP_M
    p = tmp_path / f"route_{i:010d}.npz"
    np.savez_compressed(
        p,
        ego_agent_past=past,
        ego_current_state=np.zeros(10, dtype=np.float32),
        neighbor_agents_past=np.zeros((N_NEIGHBORS, 31, 11), dtype=np.float32),
        lanes=np.zeros((4, 20, 33), dtype=np.float32),
        lanes_speed_limit=np.full((4, 1), 10.0, dtype=np.float32),
        lanes_has_speed_limit=np.ones((4, 1), dtype=bool),
        route_lanes=np.zeros((2, 20, 33), dtype=np.float32),
        route_lanes_speed_limit=np.full((2, 1), 10.0, dtype=np.float32),
        route_lanes_has_speed_limit=np.ones((2, 1), dtype=bool),
        polygons=np.zeros((2, 40, 3), dtype=np.float32),
        line_strings=np.zeros((6, 20, 4), dtype=np.float32),
        static_objects=np.zeros((5, 10), dtype=np.float32),
        ego_shape=EGO_SHAPE,
        turn_indicators=np.zeros(31, dtype=np.int64),
        goal_pose=np.array([float(N_FRAMES * STEP_M), 0.0, 0.0], dtype=np.float32),
    )
    sidecar = {
        "timestamp": float(i) * 0.1,
        "x": float(i * STEP_M),
        "y": 0.0,
        "z": 0.0,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }
    (tmp_path / f"route_{i:010d}.json").write_text(json.dumps(sidecar))
    return p


@pytest.fixture
def route(tmp_path):
    return RouteTimeline([_frame_npz(tmp_path, i) for i in range(N_FRAMES)])


@pytest.fixture
def model_pair():
    """A model that returns a straight-ahead plan, so the rollout advances deterministically."""

    def model(data):
        n = data["ego_agent_past"].shape[0]
        pred = torch.zeros((n, 1, FUTURE_LEN, 4), dtype=torch.float32)
        # (x, y, cos, sin) in the ego frame: a metre per step straight ahead.
        pred[..., 0] = torch.arange(1, FUTURE_LEN + 1, dtype=torch.float32) * STEP_M
        pred[..., 2] = 1.0
        return None, {
            "prediction": pred,
            "turn_indicator_logit": torch.zeros((n, 4), dtype=torch.float32),
        }

    model_args = SimpleNamespace(
        predicted_neighbor_num=N_NEIGHBORS,
        future_len=FUTURE_LEN,
        observation_normalizer=lambda d: d,
    )
    return model, model_args


def _run(route, model_pair, out_dir, **kw):
    model, model_args = model_pair
    return render_segment(
        model,
        model_args,
        route,
        0,
        len(route),
        out_dir,
        device="cpu",
        tracker_mode="perfect",
        # Recorded neighbours: the sim tracker needs sidecar neighbor_ids the fixture has no
        # reason to carry, and neither mode changes which stages the step loop runs.
        neighbor_history_mode="recorded",
        draw_every=None,
        max_steps=N_FRAMES,
        **kw,
    )


def test_caller_timers_are_the_ones_filled(route, model_pair, tmp_path):
    """A caller that passes its own Timers gets this segment's stages in it."""
    timers = Timers()
    _run(route, model_pair, tmp_path / "out", timers=timers)

    recorded = set(timers.as_dict())
    assert TOP_LEVEL_STAGES <= recorded, f"missing stages: {sorted(TOP_LEVEL_STAGES - recorded)}"
    # The nested breakdown comes along too, which is what makes ``input_build`` decomposable.
    # (``timeline_load_*`` lands in the same Timers only when the DRIVER hands it to
    # RouteTimeline as well, which closed_loop_eval / closed_loop_evaluation both do.)
    assert "cursor_step" in recorded
    assert all(d["calls"] > 0 and d["total_s"] >= 0.0 for d in timers.as_dict().values())


def test_stage_timers_cover_the_run(route, model_pair, tmp_path):
    """The timed blocks account for essentially the whole segment.

    This is the claim the instrumentation has to make: not that anything got faster, but that
    the reported stages leave no unexplained remainder for a slow run to hide in. Only the
    top-level blocks are summed -- ``cursor_step``/``timeline_*`` are nested and would double
    count. The bound is loose because the fixture's per-step work is tiny, so process-level
    noise (import-time lazy work, GC) is a large share of a short run.
    """
    timers = Timers()
    t0 = time.perf_counter()
    _run(route, model_pair, tmp_path / "out", timers=timers)
    elapsed = time.perf_counter() - t0

    d = timers.as_dict()
    covered = sum(d[name]["total_s"] for name in TOP_LEVEL_STAGES)
    assert covered / elapsed > 0.5, f"coverage {covered / elapsed:.1%} of {elapsed:.3f}s: {d}"


def test_timers_argument_does_not_change_the_result(route, model_pair, tmp_path):
    """Instrumentation only: the metrics are identical with and without a caller's Timers."""
    without = _run(route, model_pair, tmp_path / "a")
    with_timers = _run(route, model_pair, tmp_path / "b", timers=Timers())

    assert json.dumps(without, sort_keys=True, default=float) == json.dumps(
        with_timers, sort_keys=True, default=float
    )
