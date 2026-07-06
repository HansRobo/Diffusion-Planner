"""Instrumented closed-loop segment rollout with per-step group metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from scenario_generation.metrics.centerline_deviation import lateral_offset_from_route_lanes
from scenario_generation.metrics.safety_clearance import score_safety_step
from scenario_generation.metrics.turn_indicator import score_turn_indicator_step
from scenario_generation.perf_timer import Timers
from scenario_generation.reproducer_rollout import (
    DT,
    _advance_step,
    _pre_step,
    _seed_state,
    _to_torch_batch,
)
from scenario_generation.route_timeline import RouteTimeline


@dataclass
class InstrumentedSegmentResult:
    base_metrics: dict
    centerline_offsets: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    turn_matches: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    neighbor_violations: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    rb_violations: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))


def _percentile(arr: np.ndarray, q: float) -> float:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("inf")
    return float(np.percentile(finite, q))


@torch.no_grad()
def run_segment_instrumented(
    model,
    model_args,
    tl: RouteTimeline,
    start: int,
    end: int,
    metric_group: str,
    device: str = "cuda",
    *,
    near_miss_thresh: float = 0.3,
    search_radius: float = 1.5,
    warmup_steps: int = 0,
    replan_interval: int = 40,
) -> InstrumentedSegmentResult:
    """Closed-loop segment with per-step metrics for scenario-group evaluation."""
    cap = max(3 * (end - start), end - start)
    timers = Timers()
    s = _seed_state(
        tl,
        start,
        end,
        search_radius,
        warmup_steps,
        near_miss_thresh,
        goal_reach_m=5.0,
        max_stuck_steps=0,
        timers=timers,
        max_steps=cap,
        neighbor_history_mode="recorded",
    )

    centerline: list[float] = []
    turn_matches: list[bool] = []
    neighbor_viol: list[bool] = []
    rb_viol: list[bool] = []
    clearances: list[float] = []
    collisions: list[bool] = []
    plan_world = None

    while not s.done:
        pre = _pre_step(s)
        if pre is None:
            break
        np_dict, neighbors_live, idx, _slot_uuids, _wbu = pre
        offset = s.k % replan_interval
        override = None
        if plan_world is None or offset == 0:
            data = _to_torch_batch([np_dict], model_args, device)
            _, outputs = model(data)
            pred = outputs["prediction"][0, 0].cpu().numpy()
            from scenario_generation.reproducer_rollout import _ego_pred_to_world

            plan_world = _ego_pred_to_world(
                pred[:, :2], pred[:, 2:4], s.live_pose[0], s.live_pose[1], s.live_pose[2]
            )
            pred_cur = pred
            ti = score_turn_indicator_step(outputs, metric_group)
            if ti["turn_match"] is not None:
                turn_matches.append(bool(ti["turn_match"]))
        else:
            from scenario_generation.reproducer_rollout import _world_plan_to_ego

            off = min(offset, len(plan_world[0]) - 1)
            tx, ty, th = (
                float(plan_world[0][off, 0]),
                float(plan_world[0][off, 1]),
                float(plan_world[1][off]),
            )
            spd = float(np.hypot(tx - s.live_pose[0], ty - s.live_pose[1]) / DT)
            override = (np.array([tx, ty, th], dtype=np.float64), spd)
            pred_cur = _world_plan_to_ego(
                plan_world[0][off:],
                plan_world[1][off:],
                s.live_pose[0],
                s.live_pose[1],
                s.live_pose[2],
            )

        if metric_group in ("straight", "curve"):
            centerline.append(lateral_offset_from_route_lanes(np_dict["route_lanes"]))

        safety = score_safety_step(
            neighbors_live,
            s.ego_shape,
            np_dict,
            device,
            margin_m=near_miss_thresh,
        )
        clearances.append(float(safety["clearance_m"]))
        collisions.append(bool(safety["collision"]))
        neighbor_viol.append(bool(safety["neighbor_violation"]))
        rb_viol.append(bool(safety["rb_violation"]))

        _advance_step(s, pred_cur, idx, device, timers, override=override)
        if s.n_snaps > 0 and override is not None:
            plan_world = None

    cl_arr = np.array(centerline, dtype=np.float32)
    tm_arr = np.array(turn_matches, dtype=bool)
    nv_arr = np.array(neighbor_viol, dtype=bool)
    rb_arr = np.array(rb_viol, dtype=bool)
    cl_all = np.array(clearances, dtype=np.float32)

    base = {
        "segment": [int(s.start), int(s.end)],
        "n_steps_run": int(s.k),
        "terminated": s.terminated,
        "min_clearance": float(cl_all[np.isfinite(cl_all)].min()) if cl_all.size else float("inf"),
        "mean_clearance": float(cl_all[np.isfinite(cl_all)].mean())
        if cl_all.size
        else float("inf"),
        "n_collision_steps": int(sum(collisions)),
        "n_near_miss_steps": int(np.sum(cl_all <= near_miss_thresh)),
        "n_snaps": int(s.n_snaps),
    }

    base.update(
        {
            "centerline_mean_m": float(cl_arr[np.isfinite(cl_arr)].mean()) if cl_arr.size else None,
            "centerline_p95_m": _percentile(cl_arr, 95) if cl_arr.size else None,
            "centerline_max_m": float(cl_arr[np.isfinite(cl_arr)].max()) if cl_arr.size else None,
            "turn_match_rate": float(tm_arr.mean()) if tm_arr.size else None,
            "neighbor_violation_steps": int(nv_arr.sum()),
            "rb_violation_steps": int(rb_arr.sum()),
            "collision_steps": int(sum(collisions)),
            "min_clearance_m": base["min_clearance"],
            "min_rb_dist_m": None,
        }
    )
    return InstrumentedSegmentResult(
        base_metrics=base,
        centerline_offsets=cl_arr,
        turn_matches=tm_arr,
        neighbor_violations=nv_arr,
        rb_violations=rb_arr,
    )
