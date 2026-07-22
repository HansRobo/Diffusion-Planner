"""Unit tests for closed-loop step scorers and nested aggregate."""

from __future__ import annotations

import numpy as np

from scenario_generation.closed_loop_eval import aggregate, metrics_for_json
from scenario_generation.metrics import StepMetricContext
from scenario_generation.metrics.red_light import score_red_light_step
from scenario_generation.metrics.road_border import score_road_border_step
from scenario_generation.reproducer_rollout import _clearance_stats, _event_count


def _ctx(np_dict: dict, *, speed: float = 2.0) -> StepMetricContext:
    return StepMetricContext(
        np_dict=np_dict,
        device="cpu",
        k=0,
        ego_speed_mps=speed,
        live_pose=np.zeros(3, dtype=np.float64),
        ego_hist=np.zeros((31, 3), dtype=np.float64),
    )


def test_clearance_stats_p5():
    stats = _clearance_stats(np.arange(1, 101, dtype=np.float32))
    assert stats["clearance_min_m"] == 1.0
    assert abs(stats["clearance_mean_m"] - 50.5) < 1e-5
    assert abs(stats["clearance_p5_m"] - 5.95) < 0.1
    assert "_tdigest" in stats


def test_ego_traj_ego_frame_encodes_speed():
    ctx = _ctx({}, speed=2.0)
    traj1 = ctx.ego_traj_ego_frame(1)
    assert traj1.shape == (1, 1, 4)
    assert float(traj1[0, 0, 0]) == 0.0
    traj2 = ctx.ego_traj_ego_frame(2)
    assert traj2.shape == (1, 2, 4)
    assert float(traj2[0, 1, 0]) == 0.0
    assert float(traj2[0, 0, 0]) < 0.0


def test_red_light_step_detects_nearby_red():
    rl = np.zeros((1, 4, 33), dtype=np.float32)
    rl[0, 0, 0] = 1.0
    rl[0, 0, 2] = 1.0  # dx
    rl[0, 0, 10] = 1.0  # red
    assert score_red_light_step(_ctx({"route_lanes": rl}, speed=2.0))["red_light_violation"]
    assert not score_red_light_step(_ctx({"route_lanes": rl}, speed=0.1))["red_light_violation"]


def test_red_light_step_ignores_white():
    rl = np.zeros((1, 2, 33), dtype=np.float32)
    rl[0, 0, 0] = 1.0
    rl[0, 0, 2] = 1.0
    rl[0, 0, 11] = 1.0  # white
    assert not score_red_light_step(_ctx({"route_lanes": rl}, speed=2.0))["red_light_violation"]


def test_road_border_step_no_borders_is_inf():
    ls = np.zeros((2, 4, 4), dtype=np.float32)
    out = score_road_border_step(
        _ctx({"line_strings": ls, "ego_shape": np.array([2.7, 4.0, 2.0], dtype=np.float32)}),
        near_miss_thresh_m=0.3,
    )
    assert out["rb_dist_m"] == float("inf") or out["rb_dist_m"] > 1e3
    assert out["rb_collision"] is False
    assert out["rb_miss"] is False


def test_road_border_step_near_border_counts_miss():
    ls = np.zeros((1, 4, 4), dtype=np.float32)
    ls[0, 0, :2] = (-2.0, 0.05)
    ls[0, 1, :2] = (2.0, 0.05)
    ls[0, 0, 3] = 1.0
    ls[0, 1, 3] = 1.0
    out = score_road_border_step(
        _ctx({"line_strings": ls, "ego_shape": np.array([2.7, 4.0, 2.0], dtype=np.float32)}),
        near_miss_thresh_m=0.3,
    )
    assert np.isfinite(out["rb_dist_m"])
    assert out["rb_dist_m"] < 0.3
    assert out["rb_miss"] is True


def test_strong_brake_step_and_mask():
    from scenario_generation.metrics.strong_brake import score_strong_brake_step, strong_brake_mask

    assert score_strong_brake_step(ego_accel_mps2=-4.0, thresh_mps2=-3.0)["strong_brake"]
    assert not score_strong_brake_step(ego_accel_mps2=-2.0, thresh_mps2=-3.0)["strong_brake"]
    mask = strong_brake_mask(
        np.array([0.0, -4.0, -3.0, -2.9], dtype=np.float32), thresh_mps2=-3.0
    )
    assert mask.tolist() == [False, True, True, False]


def test_aggregate_nested_and_strips_private_series():
    from scenario_generation.metrics.tdigest import TDIGEST_KEY, tdigest_dict_from_values

    obj_vals = np.array([0.1, 0.5, 1.0, 2.0], dtype=np.float32)
    rb_vals = np.array([0.2, 0.8, 1.0, 2.0], dtype=np.float32)
    rows = [
        {
            "route": "a",
            "n_steps_run": 4,
            "terminated": "goal",
            "object": {
                "near_miss_thresh_m": 0.5,
                "collision_steps": 1,
                "collision_count": 1,
                "miss_steps": 2,
                "miss_count": 1,
                "clearance_min_m": 0.1,
                "clearance_mean_m": float(obj_vals.mean()),
                "clearance_p5_m": float(np.percentile(obj_vals, 5)),
                TDIGEST_KEY: tdigest_dict_from_values(obj_vals),
            },
            "road_border": {
                "near_miss_thresh_m": 0.5,
                "collision_steps": 0,
                "collision_count": 0,
                "miss_steps": 1,
                "miss_count": 1,
                "clearance_min_m": 0.2,
                "clearance_mean_m": float(rb_vals.mean()),
                "clearance_p5_m": float(np.percentile(rb_vals, 5)),
                TDIGEST_KEY: tdigest_dict_from_values(rb_vals),
            },
            "red_light_violation": {"steps": 1, "count": 1},
            "strong_brake": {"thresh_mps2": -3.0, "steps": 0, "count": 0},
            "reproducer": {
                "expand_count": 1,
                "snap_count": 0,
                "normal_steps": 3,
                "repeat_steps": 1,
            },
        }
    ]
    summary = aggregate(rows, near_miss_thresh=0.5, strong_brake_mps2=-3.0)
    assert summary["object"]["collision_steps"] == 1
    assert summary["object"]["miss_count"] == 1
    assert abs(summary["object"]["clearance_min_m"] - 0.1) < 1e-6
    assert abs(summary["object"]["clearance_mean_m"] - float(obj_vals.mean())) < 1e-5
    assert np.isfinite(summary["object"]["clearance_p5_m"])
    assert summary["road_border"]["miss_steps"] == 1
    assert summary["red_light_violation"]["count"] == 1
    assert summary["reproducer"]["expand_count"] == 1
    assert abs(summary["reproducer"]["repeat_step_rate"] - 0.25) < 1e-9
    assert TDIGEST_KEY not in metrics_for_json(rows[0])["object"]


def test_event_count_debounce_unchanged():
    assert _event_count(np.array([1, 0, 0, 0, 1], dtype=bool)) == 2
