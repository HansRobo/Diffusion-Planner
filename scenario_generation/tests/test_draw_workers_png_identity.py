"""``render_segment(draw_workers=N)`` must render byte-identical PNGs to the in-loop path.

The neighbours carry hex track UUIDs because that is the case that broke: the colour is looked
up from the id, so a seed-dependent hash gave each worker its own palette (see
``test_stable_palette_index.py``). Numeric ids would pass even with that bug present.
"""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scenario_generation.reproducer_rollout import render_segment
from scenario_generation.route_timeline import RouteTimeline

N_FRAMES = 10
STEP_M = 1.0  # ~10 m/s along +x: the ego moves, so the run does not terminate as "stuck"
N_NEIGHBORS = 4
FUTURE_LEN = 8
EGO_SHAPE = np.array([4.76, 2.0, 1.5], dtype=np.float32)
# hex, i.e. no all-digit suffix -- the hashed branch of the palette lookup
TRACK_UUIDS = [f"{0xA3F91C2E + i * 0x1111:08x}" for i in range(N_NEIGHBORS)]


def _neighbors() -> np.ndarray:
    """Neighbour block with non-zero pose/velocity, so the agents survive extraction."""
    nb = np.zeros((N_NEIGHBORS, 31, 11), dtype=np.float32)
    for j in range(N_NEIGHBORS):
        nb[j, :, 0] = 6.0 + 4.0 * j  # x, spread along the road
        nb[j, :, 1] = -3.0 if j % 2 else 3.0  # y, both sides
        nb[j, :, 2] = 1.0  # cos(heading)
        nb[j, :, 4] = 5.0  # vx -- part of the "is this slot populated" check
        nb[j, :, 6] = 2.0  # width
        nb[j, :, 7] = 4.5  # length
        nb[j, :, 8] = 1.0  # is_vehicle
    return nb


def _frame_npz(tmp_path, i: int):
    past = np.zeros((31, 3), dtype=np.float32)
    past[:, 0] = (np.arange(31) - 30) * STEP_M
    lanes = np.zeros((4, 20, 33), dtype=np.float32)
    lanes[:, :, 0] = np.arange(20, dtype=np.float32) * 2.0 - 20.0  # something to draw
    p = tmp_path / f"route_{i:010d}.npz"
    np.savez_compressed(
        p,
        ego_agent_past=past,
        ego_current_state=np.zeros(10, dtype=np.float32),
        neighbor_agents_past=_neighbors(),
        lanes=lanes,
        lanes_speed_limit=np.full((4, 1), 10.0, dtype=np.float32),
        lanes_has_speed_limit=np.ones((4, 1), dtype=bool),
        route_lanes=lanes[:2].copy(),
        route_lanes_speed_limit=np.full((2, 1), 10.0, dtype=np.float32),
        route_lanes_has_speed_limit=np.ones((2, 1), dtype=bool),
        polygons=np.zeros((2, 40, 3), dtype=np.float32),
        line_strings=np.zeros((6, 20, 4), dtype=np.float32),
        static_objects=np.zeros((5, 10), dtype=np.float32),
        ego_shape=EGO_SHAPE,
        turn_indicators=np.zeros(31, dtype=np.int64),
        goal_pose=np.array([float(N_FRAMES * STEP_M), 0.0, 0.0], dtype=np.float32),
    )
    (tmp_path / f"route_{i:010d}.json").write_text(
        json.dumps(
            {
                "timestamp": float(i) * 0.1,
                "x": float(i * STEP_M),
                "y": 0.0,
                "z": 0.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
                "neighbor_ids": TRACK_UUIDS,
            }
        )
    )
    return p


@pytest.fixture
def route(tmp_path):
    return RouteTimeline([_frame_npz(tmp_path, i) for i in range(N_FRAMES)])


@pytest.fixture
def model_pair():
    """A model returning a straight-ahead plan, so the rollout advances deterministically."""

    def model(data):
        n = data["ego_agent_past"].shape[0]
        pred = torch.zeros((n, 1, FUTURE_LEN, 4), dtype=torch.float32)
        pred[..., 0] = torch.arange(1, FUTURE_LEN + 1, dtype=torch.float32) * STEP_M
        pred[..., 2] = 1.0
        return None, {
            "prediction": pred,
            "turn_indicator_logit": torch.zeros((n, 4), dtype=torch.float32),
        }

    return model, SimpleNamespace(
        predicted_neighbor_num=N_NEIGHBORS,
        future_len=FUTURE_LEN,
        observation_normalizer=lambda d: d,
    )


def _render(route, model_pair, out_dir, draw_workers: int) -> dict[str, str]:
    model, model_args = model_pair
    render_segment(
        model,
        model_args,
        route,
        0,
        len(route),
        out_dir,
        device="cpu",
        tracker_mode="perfect",
        # recorded neighbours + no interpolation: both would otherwise vary what is drawn
        # for reasons unrelated to draw_workers
        neighbor_history_mode="recorded",
        interpolate=False,
        color_by_uuid=True,
        draw_every=1,
        max_steps=N_FRAMES,
        draw_workers=draw_workers,
    )
    pngs = sorted(out_dir.glob("*.png"))
    assert pngs, "the rollout drew no PNGs, so this proves nothing"
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in pngs}


def test_in_loop_rendering_is_reproducible(route, model_pair, tmp_path):
    """Baseline: two in-loop runs already agree, so a later mismatch means the pool."""
    a = _render(route, model_pair, tmp_path / "a", draw_workers=0)
    b = _render(route, model_pair, tmp_path / "b", draw_workers=0)
    assert a == b


@pytest.mark.parametrize("workers", [1, 2, 3])
def test_worker_processes_render_identical_bytes(route, model_pair, tmp_path, workers):
    """``workers=3`` has more frames than workers, so a per-worker difference cannot hide in
    a single process."""
    in_loop = _render(route, model_pair, tmp_path / "in_loop", draw_workers=0)
    pooled = _render(route, model_pair, tmp_path / f"pool{workers}", draw_workers=workers)
    assert pooled == in_loop


def test_no_pool_is_started_when_nothing_is_drawn(route, model_pair, tmp_path, monkeypatch):
    """``draw_workers`` set + ``draw_every=None`` must not spawn anything: every non-final
    epoch of a training run takes this path, so the entry-point default of 4 rides on it."""
    import scenario_generation.reproducer_rollout as rr

    started = []
    real = rr._DrawDispatcher._ensure_pool

    def spy(self):
        started.append(1)
        return real(self)

    monkeypatch.setattr(rr._DrawDispatcher, "_ensure_pool", spy)

    model, model_args = model_pair
    rr.render_segment(
        model,
        model_args,
        route,
        0,
        len(route),
        tmp_path / "nomedia",
        device="cpu",
        tracker_mode="perfect",
        neighbor_history_mode="recorded",
        interpolate=False,
        draw_every=None,
        max_steps=N_FRAMES,
        draw_workers=4,
    )
    assert not started, "a worker pool was started for a segment that draws nothing"
    assert not list((tmp_path / "nomedia").glob("*.png"))
