#!/usr/bin/env python3
"""Analyze HDP/Autoware timing and control signals in MCAP recordings.

The script intentionally streams only planning, control, localization, and
vehicle-status topics.  It does not modify the input bags and does not decode
camera, lidar, or object payloads.

Run with the pure-Python readers, for example:

  uv run --no-project --with mcap --with rosbags \
    python ros_scripts/analyze_hdp_rosbags.py \
      /mnt/nvme/wangbin/data/exp-rosbags/20260714 \
      --out-dir /tmp/hdp_rosbag_analysis_20260714
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from mcap.reader import make_reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

TOPICS: dict[str, tuple[str, str]] = {
    # HDP generator and its diagnostics.
    "hdp_trajectory": (
        "/planning/trajectory_generator/neural_network_based_planner/"
        "diffusion_planner_node/output/trajectory",
        "trajectory",
    ),
    "hdp_inference_ms": (
        "/planning/trajectory_generator/neural_network_based_planner/"
        "diffusion_planner_node/debug/inference_time_ms",
        "float",
    ),
    "hdp_processing_ms": (
        "/planning/trajectory_generator/neural_network_based_planner/"
        "diffusion_planner_node/debug/processing_time_ms",
        "float_stamped",
    ),
    "hdp_processing_tree": (
        "/planning/trajectory_generator/neural_network_based_planner/"
        "diffusion_planner_node/debug/processing_time_detail_ms",
        "processing_tree",
    ),
    "hdp_candidates": (
        "/planning/generator/diffusion_planner/candidate_trajectories",
        "candidate",
    ),
    "hdp_modified_candidates": (
        "/planning/generator/diffusion_planner/modified_candidate_trajectories",
        "candidate",
    ),
    # Older HDP output name retained for compatibility with some deployments.
    "hdp_legacy_trajectory": ("/planning/diffusion_planner/trajectory", "trajectory"),
    # Planning outputs downstream of the generator.
    "planning_trajectory": ("/planning/trajectory", "trajectory"),
    "planning_validated": ("/planning/trajectory_validated", "trajectory"),
    "planning_validated_published": (
        "/planning/trajectory_validated/debug/published_time",
        "published_time",
    ),
    # Controller and vehicle command paths.
    "controller_cmd": ("/control/trajectory_follower/control_cmd", "control"),
    "command_control_cmd": ("/control/command/control_cmd", "control"),
    "sub_command_control_cmd": ("/sub/control/command/control_cmd", "control"),
    "actuation_cmd": ("/control/command/actuation_cmd", "none"),
    "sub_actuation_cmd": ("/control/command/sub_actuation_cmd", "none"),
    "control_published": (
        "/control/trajectory_follower/control_cmd/debug/published_time",
        "published_time",
    ),
    # Explicit latency monitors.
    "control_component_latency": ("/control/control_component_latency", "float_stamped"),
    "control_latency_ms": (
        "/system/pipeline_latency_monitor/debug/control_latency_ms",
        "float_stamped",
    ),
    "planning_latency_ms": (
        "/system/pipeline_latency_monitor/debug/planning_latency_ms",
        "float_stamped",
    ),
    "pipeline_total_latency_ms": (
        "/system/pipeline_latency_monitor/debug/pipeline_total_latency_ms",
        "float_stamped",
    ),
    "pipeline_output_total_latency_ms": (
        "/system/pipeline_latency_monitor/output/total_latency_ms",
        "float_stamped",
    ),
    "planning_component_latency": (
        "/planning/trajectory_generator/trajectory_adapter_node/debug/planning_component_latency_s",
        "float_stamped",
    ),
    # State and feedback.
    "localization": ("/localization/kinematic_state", "odometry"),
    "acceleration": ("/localization/acceleration", "none"),
    "steering_status": ("/vehicle/status/steering_status", "steering"),
    "raw_steering_status": ("/vehicle/status/raw_steering_status", "steering"),
    "velocity_status": ("/vehicle/status/velocity_status", "velocity"),
}

TOPIC_TO_LABEL = {topic: label for label, (topic, _) in TOPICS.items()}
KIND_BY_LABEL = {label: kind for label, (_, kind) in TOPICS.items()}
TARGET_TOPICS = list(dict.fromkeys(TOPIC_TO_LABEL))


def stamp_ns(value: Any) -> int | None:
    """Convert a ROS builtin_interfaces/Time-like value to nanoseconds."""

    if value is None:
        return None
    sec = getattr(value, "sec", None)
    nanosec = getattr(value, "nanosec", None)
    if sec is None or nanosec is None:
        return None
    return int(sec) * 1_000_000_000 + int(nanosec)


def quat_yaw(q: Any) -> float:
    """Return yaw from a geometry_msgs Quaternion."""

    x = float(q.x)
    y = float(q.y)
    z = float(q.z)
    w = float(q.w)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def finite(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def qsummary(values: Any, scale: float = 1.0) -> dict[str, float | int | None]:
    arr = finite(values) * scale
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p01": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    qs = np.percentile(arr, [1, 5, 50, 95, 99])
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p01": float(qs[0]),
        "p05": float(qs[1]),
        "p50": float(qs[2]),
        "p95": float(qs[3]),
        "p99": float(qs[4]),
        "max": float(np.max(arr)),
    }


def timing_summary(records: list[dict[str, Any]], time_key: str = "log_ns") -> dict[str, Any]:
    times = np.array(
        sorted(int(r[time_key]) for r in records if r.get(time_key) is not None), dtype=np.int64
    )
    result: dict[str, Any] = {"count": int(times.size), "time_key": time_key}
    if times.size < 2:
        result.update(
            {
                "duration_s": 0.0,
                "rate_hz": None,
                "median_period_s": None,
                "dt": qsummary([]),
                "gap_count": 0,
                "lost_cycles": 0,
                "max_gap_s": None,
            }
        )
        return result
    dt = np.diff(times).astype(float) * 1e-9
    dt = dt[dt >= 0]
    period = float(np.median(dt)) if dt.size else float("nan")
    mean_period = float(np.mean(dt)) if dt.size else float("nan")
    duration_s = float((times[-1] - times[0]) * 1e-9)
    average_rate = float((times.size - 1) / duration_s) if duration_s > 0 else None
    threshold = max(1.5 * period, 0.02) if np.isfinite(period) and period > 0 else 0.2
    gaps = dt[dt > threshold]
    coefficient_of_variation = mean_period and float(np.std(dt) / mean_period)
    lost = (
        int(np.sum(np.maximum(np.rint(dt / period).astype(int) - 1, 0)))
        if period > 0 and coefficient_of_variation < 0.25
        else None
    )
    result.update(
        {
            "start_ns": int(times[0]),
            "end_ns": int(times[-1]),
            "duration_s": duration_s,
            "rate_hz": average_rate,
            "median_rate_hz": float(1.0 / period) if period > 0 else None,
            "mean_period_s": mean_period,
            "period_coefficient_of_variation": coefficient_of_variation,
            "median_period_s": period,
            "dt": qsummary(dt),
            "gap_threshold_s": float(threshold),
            "gap_count": int(gaps.size),
            "lost_cycles_estimate": lost,
            "max_gap_s": float(np.max(dt)) if dt.size else None,
        }
    )
    return result


def nearest_indices(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Nearest-neighbor indices into sorted ref timestamps."""

    pos = np.searchsorted(ref, query)
    pos = np.clip(pos, 0, len(ref) - 1)
    left = np.maximum(pos - 1, 0)
    use_left = np.abs(ref[left] - query) < np.abs(ref[pos] - query)
    return np.where(use_left, left, pos)


def sequence_metrics(records: list[dict[str, Any]], value_key: str) -> dict[str, Any]:
    if len(records) < 2:
        return {
            "count": len(records),
            "value": qsummary([]),
            "step": qsummary([]),
            "rate": qsummary([]),
            "sign_changes": 0,
        }
    records = sorted(records, key=lambda r: int(r["log_ns"]))
    t = np.array([int(r["log_ns"]) for r in records], dtype=float) * 1e-9
    v = np.array([float(r[value_key]) for r in records], dtype=float)
    dt = np.diff(t)
    dv = np.diff(v)
    rate = np.divide(dv, dt, out=np.full_like(dv, np.nan), where=dt > 0)
    nonzero = np.abs(v) > 0.005
    sign_changes = int(np.sum((v[:-1] * v[1:] < 0) & nonzero[:-1] & nonzero[1:]))
    return {
        "count": len(records),
        "value": qsummary(v),
        "step": qsummary(dv),
        "abs_step": qsummary(np.abs(dv)),
        "rate": qsummary(rate),
        "abs_rate": qsummary(np.abs(rate)),
        "sign_changes_above_0p005_rad": sign_changes,
    }


def decode_trajectory(obj: Any, log_ns: int) -> dict[str, Any]:
    points = list(getattr(obj, "points", []))
    n = len(points)
    time_s = np.empty(n, dtype=float)
    xy = np.empty((n, 2), dtype=float)
    yaw = np.empty(n, dtype=float)
    velocity = np.empty(n, dtype=float)
    acceleration = np.empty(n, dtype=float)
    steer = np.empty(n, dtype=float)
    heading_rate = np.empty(n, dtype=float)
    for i, point in enumerate(points):
        time_s[i] = (stamp_ns(point.time_from_start) or 0) * 1e-9
        xy[i] = [float(point.pose.position.x), float(point.pose.position.y)]
        yaw[i] = quat_yaw(point.pose.orientation)
        velocity[i] = float(point.longitudinal_velocity_mps)
        acceleration[i] = float(point.acceleration_mps2)
        steer[i] = float(point.front_wheel_angle_rad)
        heading_rate[i] = float(point.heading_rate_rps)
    return {
        "log_ns": int(log_ns),
        "header_ns": stamp_ns(getattr(obj, "header", None).stamp),
        "n_points": n,
        "time_s": time_s,
        "xy": xy,
        "yaw": yaw,
        "velocity": velocity,
        "acceleration": acceleration,
        "steer": steer,
        "heading_rate": heading_rate,
    }


def decode_message(obj: Any, kind: str, log_ns: int) -> dict[str, Any]:
    rec: dict[str, Any] = {"log_ns": int(log_ns)}
    if kind == "trajectory":
        return decode_trajectory(obj, log_ns)
    if kind == "float":
        rec["value"] = float(obj.data)
    elif kind == "float_stamped":
        rec["stamp_ns"] = stamp_ns(getattr(obj, "stamp", None))
        rec["value"] = float(obj.data)
    elif kind == "processing_tree":
        rec["nodes"] = [
            {
                "id": int(n.id),
                "name": str(n.name),
                "processing_time": float(n.processing_time),
                "parent_id": int(n.parent_id),
                "comment": str(n.comment),
            }
            for n in getattr(obj, "nodes", [])
        ]
    elif kind == "candidate":
        candidates = list(getattr(obj, "candidate_trajectories", []))
        rec["n_candidates"] = len(candidates)
        rec["candidate_points"] = [len(getattr(c, "points", [])) for c in candidates]
        rec["header_ns"] = (
            stamp_ns(getattr(candidates[0], "header", None).stamp) if candidates else None
        )
    elif kind == "published_time":
        rec["header_ns"] = stamp_ns(getattr(obj, "header", None).stamp)
        rec["published_ns"] = stamp_ns(getattr(obj, "published_stamp", None))
    elif kind == "control":
        rec["stamp_ns"] = stamp_ns(getattr(obj, "stamp", None))
        lateral = getattr(obj, "lateral", None)
        longitudinal = getattr(obj, "longitudinal", None)
        rec["lateral_stamp_ns"] = stamp_ns(getattr(lateral, "stamp", None))
        rec["longitudinal_stamp_ns"] = stamp_ns(getattr(longitudinal, "stamp", None))
        rec["steer"] = float(getattr(lateral, "steering_tire_angle", np.nan))
        rec["steer_rate"] = float(getattr(lateral, "steering_tire_rotation_rate", np.nan))
        rec["velocity"] = float(getattr(longitudinal, "velocity", np.nan))
        rec["acceleration"] = float(getattr(longitudinal, "acceleration", np.nan))
    elif kind == "steering":
        rec["stamp_ns"] = stamp_ns(getattr(obj, "stamp", None))
        rec["steer"] = float(obj.steering_tire_angle)
    elif kind == "velocity":
        rec["stamp_ns"] = stamp_ns(getattr(obj, "header", None).stamp)
        rec["velocity"] = float(obj.longitudinal_velocity)
        rec["lateral_velocity"] = float(obj.lateral_velocity)
        rec["heading_rate"] = float(obj.heading_rate)
    elif kind == "odometry":
        rec["stamp_ns"] = stamp_ns(getattr(obj, "header", None).stamp)
        pose = obj.pose.pose
        rec["x"] = float(pose.position.x)
        rec["y"] = float(pose.position.y)
        rec["yaw"] = quat_yaw(pose.orientation)
        rec["velocity"] = float(obj.twist.twist.linear.x)
        rec["lateral_velocity"] = float(obj.twist.twist.linear.y)
        rec["heading_rate"] = float(obj.twist.twist.angular.z)
    return rec


def type_store_for(reader_summary: Any, available: set[str]) -> tuple[Any, dict[int, str]]:
    ts = get_typestore(Stores.ROS2_HUMBLE)
    schema_names: dict[int, str] = {}
    registered: set[str] = set()
    for channel in reader_summary.channels.values():
        if channel.topic not in available:
            continue
        if KIND_BY_LABEL[TOPIC_TO_LABEL[channel.topic]] == "none":
            continue
        schema = reader_summary.schemas.get(channel.schema_id)
        if schema is None:
            continue
        schema_names[channel.schema_id] = schema.name
        if schema.name in registered or schema.encoding != "ros2msg":
            continue
        try:
            ts.register(get_types_from_msg(schema.data.decode(), schema.name))
            registered.add(schema.name)
        except Exception as exc:  # pragma: no cover - depends on external schema set
            print(f"warning: could not register {schema.name}: {exc}", file=sys.stderr)
    return ts, schema_names


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def find_topic_stats(records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for label, recs in records.items():
        entry: dict[str, Any] = {"topic": TOPICS[label][0], "timing": timing_summary(recs)}
        for time_key in ("stamp_ns", "header_ns", "published_ns"):
            if sum(r.get(time_key) is not None for r in recs) >= 2:
                entry[f"{time_key}_timing"] = timing_summary(recs, time_key)
        if label in {"control_component_latency", "planning_component_latency"}:
            vals = [r["value"] for r in recs if "value" in r]
            entry["value_seconds"] = qsummary(vals)
            entry["value_milliseconds_assumed"] = qsummary(vals, scale=1000.0)
        elif label == "hdp_inference_ms":
            entry["inference_ms"] = qsummary([r["value"] for r in recs if "value" in r])
        elif label == "hdp_processing_ms":
            entry["processing_ms"] = qsummary([r["value"] for r in recs if "value" in r])
        elif label.endswith("_ms"):
            vals = [r["value"] for r in recs if "value" in r]
            entry["value"] = qsummary(vals)
        elif label in {"controller_cmd", "command_control_cmd", "sub_command_control_cmd"}:
            entry["steering"] = sequence_metrics(recs, "steer")
            entry["acceleration"] = qsummary([r["acceleration"] for r in recs])
            entry["velocity_command"] = qsummary([r["velocity"] for r in recs])
        elif label in {"steering_status", "raw_steering_status"}:
            entry["steering"] = sequence_metrics(recs, "steer")
        elif label == "velocity_status":
            entry["velocity"] = qsummary([r["velocity"] for r in recs])
            entry["heading_rate"] = qsummary([r["heading_rate"] for r in recs])
        elif label == "hdp_processing_tree":
            by_name: dict[str, list[float]] = defaultdict(list)
            for rec in recs:
                for node in rec.get("nodes", []):
                    by_name[node["name"]].append(node["processing_time"])
            entry["nodes_ms"] = {name: qsummary(vals) for name, vals in sorted(by_name.items())}
        elif label == "hdp_trajectory":
            entry["n_points"] = qsummary([r["n_points"] for r in recs])
        elif label in {"hdp_candidates", "hdp_modified_candidates"}:
            entry["n_candidates"] = qsummary([r["n_candidates"] for r in recs])
        stats[label] = entry
    return stats


def planner_latency(records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    traj = sorted(records.get("hdp_trajectory", []), key=lambda r: r["log_ns"])
    if not traj:
        traj = sorted(records.get("hdp_legacy_trajectory", []), key=lambda r: r["log_ns"])
    result: dict[str, Any] = {
        "trajectory_topic_used": "hdp_trajectory"
        if records.get("hdp_trajectory")
        else "hdp_legacy_trajectory"
    }
    lat = [
        (int(r["log_ns"]) - int(r["header_ns"])) * 1e-6
        for r in traj
        if r.get("header_ns") is not None
    ]
    result["trajectory_record_minus_header_ms"] = qsummary(lat)
    result["reported_processing_ms"] = qsummary(
        [r["value"] for r in records.get("hdp_processing_ms", []) if "value" in r]
    )
    result["reported_inference_ms"] = qsummary(
        [r["value"] for r in records.get("hdp_inference_ms", []) if "value" in r]
    )
    # The debug stamp is the planner-cycle timestamp. Match it to the
    # nearest trajectory header timestamp, then expose cycle-to-output delay.
    proc = [r for r in records.get("hdp_processing_ms", []) if r.get("stamp_ns") is not None]
    if proc and traj:
        output_pairs = [
            (r["header_ns"], r["log_ns"]) for r in traj if r.get("header_ns") is not None
        ]
        if output_pairs:
            out_t = np.array([pair[0] for pair in output_pairs], dtype=np.int64)
            out_log_t = np.array([pair[1] for pair in output_pairs], dtype=np.int64)
            p_t = np.array([r["stamp_ns"] for r in proc], dtype=np.int64)
            idx = nearest_indices(p_t, out_t)
            cycle_to_output = (out_log_t[idx] - p_t).astype(float) * 1e-6
            cycle_to_output = cycle_to_output[np.abs(cycle_to_output) < 1000.0]
            result["processing_stamp_to_nearest_output_ms"] = qsummary(cycle_to_output)
        else:
            result["processing_stamp_to_nearest_output_ms"] = qsummary([])
    else:
        result["processing_stamp_to_nearest_output_ms"] = qsummary([])
    return result


def trajectory_stability(records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    traj = sorted(records.get("hdp_trajectory", []), key=lambda r: r["log_ns"])
    if not traj:
        traj = sorted(records.get("hdp_legacy_trajectory", []), key=lambda r: r["log_ns"])
    result: dict[str, Any] = {"trajectory_count": len(traj)}
    if not traj:
        return result
    valid = [r for r in traj if r.get("n_points", 0) > 0]
    first_steer = [r["steer"][0] for r in valid]
    first_accel = [r["acceleration"][0] for r in valid]
    first_velocity = [r["velocity"][0] for r in valid]
    first_yaw = [r["yaw"][0] for r in valid]
    result["first_point_steering_rad"] = qsummary(first_steer)
    result["first_point_acceleration_mps2"] = qsummary(first_accel)
    result["first_point_velocity_mps"] = qsummary(first_velocity)

    # Changes in the first predicted action between replans are the most
    # direct indicator of a noisy receding-horizon policy.
    if len(valid) > 1:
        t = np.array([r["log_ns"] for r in valid], dtype=float) * 1e-9
        steer = np.array([r["steer"][0] for r in valid], dtype=float)
        accel = np.array([r["acceleration"][0] for r in valid], dtype=float)
        dt = np.diff(t)
        result["replan_first_steering_step_rad"] = qsummary(np.diff(steer))
        result["replan_first_steering_rate_radps"] = qsummary(
            np.divide(np.diff(steer), dt, out=np.full_like(dt, np.nan), where=dt > 0)
        )
        result["replan_first_acceleration_step_mps2"] = qsummary(np.diff(accel))
        result["replan_first_heading_step_rad"] = qsummary(
            np.arctan2(np.sin(np.diff(first_yaw)), np.cos(np.diff(first_yaw)))
        )
        result["replan_first_steering_sequence"] = sequence_metrics(
            [{"log_ns": r["log_ns"], "steer": r["steer"][0]} for r in valid], "steer"
        )

        horizon = min(10, min(r["n_points"] for r in valid))
        delta_xy: dict[str, Any] = {}
        delta_yaw: dict[str, Any] = {}
        for k in range(horizon):
            xy = np.array([r["xy"][k] for r in valid], dtype=float)
            yaw = np.array([r["yaw"][k] for r in valid], dtype=float)
            delta_xy[f"h{k}"] = qsummary(np.linalg.norm(np.diff(xy, axis=0), axis=1))
            delta_yaw[f"h{k}"] = qsummary(np.arctan2(np.sin(np.diff(yaw)), np.cos(np.diff(yaw))))
        result["replan_same_horizon_xy_delta_m"] = delta_xy
        result["replan_same_horizon_yaw_delta_rad"] = delta_yaw

    # Smoothness inside each emitted trajectory.
    internal_steer_steps: list[float] = []
    internal_accel_steps: list[float] = []
    internal_xy_speeds: list[float] = []
    for r in valid:
        if r["n_points"] < 2:
            continue
        dt = np.diff(r["time_s"])
        internal_steer_steps.extend(np.diff(r["steer"]).tolist())
        internal_accel_steps.extend(np.diff(r["acceleration"]).tolist())
        internal_xy_speeds.extend((np.linalg.norm(np.diff(r["xy"], axis=0), axis=1) / dt).tolist())
    result["within_trajectory_steering_step_rad"] = qsummary(internal_steer_steps)
    result["within_trajectory_acceleration_step_mps2"] = qsummary(internal_accel_steps)
    result["within_trajectory_xy_speed_mps"] = qsummary(internal_xy_speeds)

    # Express the first point relative to the nearest localization pose at
    # the trajectory header time.  This checks for a bad frame/timestamp
    # contract independently of controller behavior.
    loc = sorted(records.get("localization", []), key=lambda r: r.get("stamp_ns", r["log_ns"]))
    loc = [r for r in loc if r.get("stamp_ns") is not None]
    if loc:
        lt = np.array([r["stamp_ns"] for r in loc], dtype=np.int64)
        longitudinal: list[float] = []
        lateral: list[float] = []
        heading_error: list[float] = []
        for r in valid:
            if r.get("header_ns") is None:
                continue
            j = int(nearest_indices(np.array([r["header_ns"]], dtype=np.int64), lt)[0])
            e = loc[j]
            dx = float(r["xy"][0, 0] - e["x"])
            dy = float(r["xy"][0, 1] - e["y"])
            cy, sy = math.cos(e["yaw"]), math.sin(e["yaw"])
            longitudinal.append(cy * dx + sy * dy)
            lateral.append(-sy * dx + cy * dy)
            heading_error.append(
                math.atan2(math.sin(r["yaw"][0] - e["yaw"]), math.cos(r["yaw"][0] - e["yaw"]))
            )
        result["first_point_relative_longitudinal_m"] = qsummary(longitudinal)
        result["first_point_relative_lateral_m"] = qsummary(lateral)
        result["first_point_heading_error_rad"] = qsummary(heading_error)
    return result


def command_actual_lag(records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    commands = records.get("command_control_cmd", []) or records.get("controller_cmd", [])
    actual = records.get("steering_status", [])
    commands = sorted(
        [r for r in commands if r.get("steer") is not None],
        key=lambda r: r.get("stamp_ns", r["log_ns"]),
    )
    actual = sorted(
        [r for r in actual if r.get("steer") is not None],
        key=lambda r: r.get("stamp_ns", r["log_ns"]),
    )
    if len(commands) < 20 or len(actual) < 100:
        return {
            "command_count": len(commands),
            "actual_count": len(actual),
            "status": "insufficient_data",
        }
    ct = np.array([r.get("stamp_ns") or r["log_ns"] for r in commands], dtype=float) * 1e-9
    cv = np.array([r["steer"] for r in commands], dtype=float)
    at = np.array([r.get("stamp_ns") or r["log_ns"] for r in actual], dtype=float) * 1e-9
    av = np.array([r["steer"] for r in actual], dtype=float)
    valid_command = np.isfinite(ct) & np.isfinite(cv)
    valid_actual = np.isfinite(at) & np.isfinite(av)
    ct, cv = ct[valid_command], cv[valid_command]
    at, av = at[valid_actual], av[valid_actual]
    if np.std(cv) < 1e-4 or np.std(av) < 1e-4:
        return {
            "command_count": len(cv),
            "actual_count": len(av),
            "status": "insufficient_excitation",
        }
    candidates = np.arange(0.0, 0.501, 0.005)
    correlations: list[float] = []
    rmses: list[float] = []
    biases: list[float] = []
    for lag in candidates:
        keep = (ct + lag >= at[0]) & (ct + lag <= at[-1])
        if np.sum(keep) < 20:
            correlations.append(np.nan)
            rmses.append(np.nan)
            biases.append(np.nan)
            continue
        pred = np.interp(ct[keep] + lag, at, av)
        correlations.append(float(np.corrcoef(cv[keep], pred)[0, 1]))
        residual = pred - cv[keep]
        rmses.append(float(np.sqrt(np.mean(residual * residual))))
        biases.append(float(np.mean(residual)))
    corr = np.asarray(correlations)
    if not np.any(np.isfinite(corr)):
        return {"command_count": len(cv), "actual_count": len(av), "status": "no_overlap"}
    best = int(np.nanargmax(corr))
    lag = float(candidates[best])
    return {
        "status": "ok",
        "command_count": int(len(cv)),
        "actual_count": int(len(av)),
        "best_positive_lag_s": lag,
        "best_lag_ms": lag * 1000.0,
        "correlation": float(corr[best]),
        "rms_actual_minus_command_rad": float(rmses[best]),
        "bias_actual_minus_command_rad": float(biases[best]),
        "interpretation": "positive lag means vehicle steering follows command after that delay",
    }


def write_topic_csv(path: Path, stats: dict[str, Any]) -> None:
    fields = [
        "label",
        "topic",
        "count",
        "rate_hz",
        "median_period_s",
        "p01_dt_s",
        "p50_dt_s",
        "p95_dt_s",
        "p99_dt_s",
        "max_gap_s",
        "gap_count",
        "lost_cycles_estimate",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for label, entry in stats.items():
            timing = entry["timing"]
            dt = timing.get("dt", {})
            writer.writerow(
                {
                    "label": label,
                    "topic": entry["topic"],
                    "count": timing.get("count"),
                    "rate_hz": timing.get("rate_hz"),
                    "median_period_s": timing.get("median_period_s"),
                    "p01_dt_s": dt.get("p01"),
                    "p50_dt_s": dt.get("p50"),
                    "p95_dt_s": dt.get("p95"),
                    "p99_dt_s": dt.get("p99"),
                    "max_gap_s": timing.get("max_gap_s"),
                    "gap_count": timing.get("gap_count"),
                    "lost_cycles_estimate": timing.get("lost_cycles_estimate"),
                }
            )


def write_signal_csv(path: Path, records: dict[str, list[dict[str, Any]]]) -> None:
    with path.open("w", newline="") as f:
        fields = [
            "source",
            "log_time_s",
            "stamp_time_s",
            "header_time_s",
            "published_time_s",
            "steering_rad",
            "velocity_mps",
            "acceleration_mps2",
            "planner_first_steering_rad",
            "planner_first_acceleration_mps2",
            "planner_first_velocity_mps",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        traj = sorted(
            records.get("hdp_trajectory", []) or records.get("hdp_legacy_trajectory", []),
            key=lambda r: r["log_ns"],
        )
        for r in traj:
            writer.writerow(
                {
                    "source": "hdp_trajectory",
                    "log_time_s": r["log_ns"] * 1e-9,
                    "header_time_s": (r.get("header_ns") or 0) * 1e-9,
                    "planner_first_steering_rad": r["steer"][0] if r["n_points"] else None,
                    "planner_first_acceleration_mps2": r["acceleration"][0]
                    if r["n_points"]
                    else None,
                    "planner_first_velocity_mps": r["velocity"][0] if r["n_points"] else None,
                }
            )
        for label in ("command_control_cmd", "controller_cmd", "steering_status"):
            for r in sorted(records.get(label, []), key=lambda x: x["log_ns"]):
                writer.writerow(
                    {
                        "source": label,
                        "log_time_s": r["log_ns"] * 1e-9,
                        "stamp_time_s": (r.get("stamp_ns") or 0) * 1e-9,
                        "steering_rad": r.get("steer"),
                        "velocity_mps": r.get("velocity"),
                        "acceleration_mps2": r.get("acceleration"),
                    }
                )


def write_plots(out_dir: Path, records: dict[str, list[dict[str, Any]]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional visualization dependency
        print(f"warning: plotting unavailable: {exc}", file=sys.stderr)
        return
    t0_candidates = [r["log_ns"] for recs in records.values() for r in recs if "log_ns" in r]
    if not t0_candidates:
        return
    t0 = min(t0_candidates) * 1e-9
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for label, name, key, scale in [
        ("hdp_trajectory", "HDP first-point steering", "steer", 1.0),
        ("command_control_cmd", "command steering", "steer", 1.0),
        ("steering_status", "measured steering", "steer", 1.0),
    ]:
        rs = records.get(label, [])
        if not rs:
            continue
        if label == "hdp_trajectory":
            x = np.array([r["log_ns"] for r in rs]) * 1e-9 - t0
            y = np.array([r["steer"][0] if r["n_points"] else np.nan for r in rs]) * scale
        else:
            x = np.array([r["log_ns"] for r in rs]) * 1e-9 - t0
            y = np.array([r.get(key, np.nan) for r in rs]) * scale
        axes[0].plot(x, y, ".", ms=1.5, label=name)
    axes[0].set_ylabel("steering [rad]")
    axes[0].legend(loc="upper right", ncol=3)
    for label, name, key in [
        ("hdp_processing_ms", "HDP processing", "value"),
        ("hdp_inference_ms", "HDP inference", "value"),
        ("planning_latency_ms", "planning pipeline", "value"),
        ("control_latency_ms", "control pipeline", "value"),
    ]:
        rs = records.get(label, [])
        if not rs:
            continue
        x = np.array([r["log_ns"] for r in rs]) * 1e-9 - t0
        y = np.array([r.get(key, np.nan) for r in rs])
        axes[1].plot(x, y, ".", ms=1.5, label=name)
    axes[1].set_ylabel("latency [ms]")
    axes[1].legend(loc="upper right", ncol=4)
    for label, name in [
        ("hdp_trajectory", "HDP output"),
        ("command_control_cmd", "control command"),
        ("localization", "localization"),
    ]:
        rs = records.get(label, [])
        if not rs:
            continue
        x = np.array([r["log_ns"] for r in rs]) * 1e-9 - t0
        dt = np.diff(x)
        axes[2].plot(x[1:], dt * 1000.0, ".", ms=1.5, label=name)
    axes[2].set_ylabel("inter-arrival [ms]")
    axes[2].set_xlabel("recorded time from corpus start [s]")
    axes[2].legend(loc="upper right", ncol=3)
    axes[2].set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_dir / "hdp_timing_and_control.png", dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/hdp_rosbag_analysis"))
    parser.add_argument("--limit", type=int, default=0, help="only process the first N MCAP files")
    args = parser.parse_args()
    files = sorted(args.bag_dir.glob("*.mcap"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no .mcap files found in {args.bag_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records: dict[str, list[dict[str, Any]]] = {label: [] for label in TOPICS}
    bag_rows: list[dict[str, Any]] = []
    decode_errors: list[dict[str, str]] = []

    for file_index, path in enumerate(files, start=1):
        print(f"[{file_index}/{len(files)}] {path.name}", flush=True)
        with path.open("rb") as stream:
            reader = make_reader(stream)
            summary = reader.get_summary()
            topic_channels = {c.topic: c for c in summary.channels.values()}
            available_topics = set(TARGET_TOPICS) & set(topic_channels)
            ts, schema_names = type_store_for(summary, available_topics)
            counts = summary.statistics.channel_message_counts if summary.statistics else {}
            target_counts = {
                topic: int(counts.get(topic_channels[topic].id, 0))
                for topic in sorted(available_topics)
            }
            bag_rows.append(
                {
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "start_ns": int(summary.statistics.message_start_time)
                    if summary.statistics
                    else None,
                    "end_ns": int(summary.statistics.message_end_time)
                    if summary.statistics
                    else None,
                    "duration_s": (
                        (
                            summary.statistics.message_end_time
                            - summary.statistics.message_start_time
                        )
                        * 1e-9
                        if summary.statistics
                        else None
                    ),
                    "target_counts": target_counts,
                }
            )
            for _, channel, message in reader.iter_messages(topics=sorted(available_topics)):
                label = TOPIC_TO_LABEL[channel.topic]
                kind = KIND_BY_LABEL[label]
                if kind == "none":
                    records[label].append({"log_ns": int(message.log_time)})
                    continue
                schema_name = schema_names.get(channel.schema_id)
                if schema_name is None:
                    decode_errors.append(
                        {"file": path.name, "topic": channel.topic, "error": "missing schema"}
                    )
                    continue
                try:
                    obj = ts.deserialize_cdr(message.data, schema_name)
                    records[label].append(decode_message(obj, kind, int(message.log_time)))
                except Exception as exc:
                    decode_errors.append(
                        {"file": path.name, "topic": channel.topic, "error": repr(exc)}
                    )

    topic_stats = find_topic_stats({k: v for k, v in records.items() if v})
    report = {
        "bag_dir": str(args.bag_dir),
        "files_processed": len(files),
        "input_bytes": int(sum(r["bytes"] for r in bag_rows)),
        "bags": bag_rows,
        "topics": topic_stats,
        "planner_latency": planner_latency(records),
        "trajectory_stability": trajectory_stability(records),
        "control_actual_lag": command_actual_lag(records),
        "decode_error_count": len(decode_errors),
        "decode_errors_sample": decode_errors[:20],
        "notes": [
            "Rates and gaps use MCAP log_time, i.e. recorder arrival time.",
            "ROS header/stamp fields are retained separately for latency calculations.",
            "control_component_latency and planning_component_latency are reported in raw seconds and assumed milliseconds.",
            "A positive command/feedback lag means measured steering follows the command later.",
        ],
    }
    (args.out_dir / "summary.json").write_text(json.dumps(json_ready(report), indent=2))
    write_topic_csv(args.out_dir / "topic_stats.csv", topic_stats)
    write_signal_csv(args.out_dir / "signals.csv", records)
    write_plots(args.out_dir, records)
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "files": len(files),
                "input_gb": sum(r["bytes"] for r in bag_rows) / 1e9,
                "decode_errors": len(decode_errors),
                "topic_counts": {label: len(recs) for label, recs in records.items() if recs},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
