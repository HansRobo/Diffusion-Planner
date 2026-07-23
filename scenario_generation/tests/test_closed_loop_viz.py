"""Tests for closed-loop viz helpers: event onsets, hollow ego box, metric map."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scenario_generation.closed_loop_metric_map import (
    _subsample_arrows,
    collect_route_map_polylines,
    write_route_metric_map,
)
from scenario_generation.reproducer_rollout import _event_count, _event_onsets
from scenario_generation.visualize import draw_agent_box


class _FakeTimeline:
    """Minimal timeline: poses + npz(lanes/line_strings/route_lanes) per frame."""

    def __init__(self, n: int = 10, yaw: float = 0.0):
        self.poses = np.zeros((n, 3), dtype=np.float64)
        self.poses[:, 0] = np.arange(n, dtype=np.float64) * 5.0  # advance +x
        self.poses[:, 2] = yaw
        lane = np.zeros((1, 20, 33), dtype=np.float32)
        lane[0, :, 0] = np.linspace(0, 30, 20)
        lane[0, :, 1] = 0.0
        lane[0, :, 4] = 0.0
        lane[0, :, 5] = 1.75
        lane[0, :, 6] = 0.0
        lane[0, :, 7] = -1.75
        ls = np.zeros((1, 10, 4), dtype=np.float32)
        ls[0, :, 0] = np.linspace(0, 30, 10)
        ls[0, :, 1] = 4.0
        ls[0, :, 3] = 1.0
        route = np.zeros((1, 20, 12), dtype=np.float32)
        route[0, :, 0] = np.linspace(0, 30, 20)
        self._npz = {
            "lanes": lane,
            "line_strings": ls,
            "route_lanes": route,
        }

    def __len__(self) -> int:
        return len(self.poses)

    def npz(self, idx: int) -> dict:
        return self._npz


def test_event_onsets_match_event_count():
    mask = np.array(
        [0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        dtype=bool,
    )
    onsets = _event_onsets(mask)
    assert onsets == [1, 6, 10]
    assert _event_count(mask) == len(onsets)


def test_event_onsets_debounce_short_gap():
    mask = np.array([1, 1, 0, 1, 1, 0, 0, 0, 1], dtype=bool)
    onsets = _event_onsets(mask)
    assert onsets == [0, 8]
    assert _event_count(mask) == 2


def test_draw_agent_box_hollow():
    fig, ax = plt.subplots()
    draw_agent_box(ax, 0.0, 0.0, 0.0, 4.5, 1.8, "#3366cc", filled=False, wheelbase=2.7)
    patch = ax.patches[-1]
    assert patch.get_facecolor()[3] == 0.0 or patch.get_facecolor()[:3] == (0.0, 0.0, 0.0)
    assert patch.get_edgecolor()[2] > 0.5
    plt.close(fig)


def test_collect_route_map_polylines_world_frame():
    tl = _FakeTimeline(n=10, yaw=0.0)
    polys = collect_route_map_polylines(tl, 0, 10, stride=3)
    assert len(polys["lane_centerlines"]) > 0
    assert len(polys["lane_borders"]) > 0
    assert len(polys["road_borders"]) > 0
    assert len(polys["route_lanes"]) > 0
    c0 = polys["lane_centerlines"][0]
    assert c0[:, 0].min() >= -0.1
    assert c0[:, 0].max() <= 30.1
    last = polys["lane_centerlines"][-1]
    assert last[:, 0].max() > 40.0


def test_subsample_arrows_every_5_frames():
    # 40 ticks; stride 5 -> 0,5,...,40 = 9 arrows including end.
    xy = np.column_stack([np.linspace(0, 40, 41), np.zeros(41)])
    sub, ang = _subsample_arrows(xy, stride=5)
    assert len(sub) == 9
    assert len(ang) == 9
    # Heading along +X -> plotly angle 90 (arrow points right).
    assert abs(ang[0] - 90.0) < 1e-6


def test_write_route_metric_map_with_map(tmp_path: Path):
    n = 40
    live = np.column_stack([np.linspace(0, 20, n), np.zeros(n)])
    gt = live.copy()
    clearances = np.full(n, 5.0, dtype=np.float32)
    collisions = np.zeros(n, dtype=bool)
    collisions[10] = True
    collisions[11] = True
    clearances[10:12] = 0.0
    clearances[20] = 0.3
    rb = np.full(n, 2.0, dtype=np.float32)
    rb[25] = -0.05
    red = np.zeros(n, dtype=bool)
    red[30] = True
    accels = np.zeros(n, dtype=np.float32)
    accels[5] = -4.0
    accels[6] = -4.0

    tl = _FakeTimeline(n=10)
    out = write_route_metric_map(
        out_path=tmp_path / "route_metrics_map.png",
        live_xy=live,
        gt_xy=gt,
        clearances=clearances,
        collisions=collisions,
        rb_dists=rb,
        red_light=red,
        accels=accels,
        near_miss_thresh=0.5,
        strong_brake_mps2=-2.5,
        snap_xy=[[15.0, 0.0]],
        title="test_route",
        metrics={
            "object": {"collision_count": 1, "miss_count": 1},
            "road_border": {"collision_count": 1, "miss_count": 0},
            "red_light_violation": {"count": 1},
            "strong_brake": {"count": 1},
            "reproducer": {"snap_count": 1},
        },
        tl=tl,
        start=0,
        end=10,
        margin_m=10.0,
    )
    assert out.is_file()
    assert out.stat().st_size > 0
    assert out.with_suffix(".html").is_file()
