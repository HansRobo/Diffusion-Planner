#!/usr/bin/env python3
"""Controller-in-the-loop comparison for HDP trajectory timing contracts.

This tool replays the trajectory updates from a real MCAP at their recorded
arrival times.  Each trajectory is expressed relative to the recorded ego pose
at its header stamp and then re-centered on the simulated ego pose at the same
time.  This preserves the ego-frame equivariance of the planner while allowing
the lateral vehicle state to evolve in closed loop.

The three built-in variants isolate the issue under investigation:

* temporal_no_anchor: current HDP output (+0.1 s is the first point)
* temporal_anchor: measured ego state at t=0 followed by the HDP future
* spatial_anchor: the same anchored trajectory consumed by spatial projection

The controller is a linear surrogate fitted to the MPC diagnostics in the bag;
the vehicle is a delayed first-order steering bicycle model.  Therefore this is
a fast screening test, not a replacement for the real Autoware MPC integration
test.

Example:

  uv run --no-project --with mcap --with rosbags --with numpy --with scipy \
    python ros_scripts/replay_hdp_trajectory_control.py BAG.mcap
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from mcap.reader import make_reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore
from scipy.interpolate import CubicSpline

TRAJECTORY_TOPIC = "/planning/trajectory"
ODOMETRY_TOPIC = "/localization/kinematic_state"
DIAGNOSTIC_TOPIC = "/control/trajectory_follower/lateral/diagnostic"


def stamp_ns(value: Any) -> int:
    return int(value.sec) * 1_000_000_000 + int(value.nanosec)


def quat_yaw(q: Any) -> float:
    return math.atan2(
        2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y)),
        1.0 - 2.0 * (float(q.y) ** 2 + float(q.z) ** 2),
    )


def wrap(angle: float | np.ndarray) -> float | np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


@dataclass
class TrajectoryRecord:
    log_s: float
    header_s: float
    time_s: np.ndarray
    xy: np.ndarray
    yaw: np.ndarray
    velocity: np.ndarray


@dataclass
class StateSeries:
    time_s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    velocity: np.ndarray


@dataclass
class ReferencePath:
    header_s: float
    time_s: np.ndarray
    xy: np.ndarray
    yaw: np.ndarray
    velocity: np.ndarray
    arc_s: np.ndarray
    curvature_s: np.ndarray
    curvature: np.ndarray


def read_bag(path: Path) -> tuple[list[TrajectoryRecord], StateSeries, np.ndarray, np.ndarray]:
    topics = {TRAJECTORY_TOPIC, ODOMETRY_TOPIC, DIAGNOSTIC_TOPIC}
    trajectories: list[TrajectoryRecord] = []
    odometry: list[tuple[float, float, float, float, float]] = []
    diagnostic_time: list[float] = []
    diagnostics: list[np.ndarray] = []

    with path.open("rb") as stream:
        reader = make_reader(stream)
        summary = reader.get_summary()
        typestore = get_typestore(Stores.ROS2_HUMBLE)
        schema_names: dict[int, str] = {}
        registered: set[str] = set()
        for channel in summary.channels.values():
            if channel.topic not in topics:
                continue
            schema = summary.schemas[channel.schema_id]
            schema_names[channel.schema_id] = schema.name
            if schema.name in registered:
                continue
            # Standard Humble messages are already present in the typestore.
            with contextlib.suppress(Exception):
                typestore.register(get_types_from_msg(schema.data.decode(), schema.name))
            registered.add(schema.name)

        for _, channel, message in reader.iter_messages(topics=sorted(topics)):
            obj = typestore.deserialize_cdr(message.data, schema_names[channel.schema_id])
            if channel.topic == TRAJECTORY_TOPIC:
                points = list(obj.points)
                if len(points) < 3:
                    continue
                times = np.asarray([stamp_ns(p.time_from_start) * 1.0e-9 for p in points])
                xy = np.asarray(
                    [[float(p.pose.position.x), float(p.pose.position.y)] for p in points]
                )
                yaw = np.asarray([quat_yaw(p.pose.orientation) for p in points])
                velocity = np.asarray([float(p.longitudinal_velocity_mps) for p in points])
                trajectories.append(
                    TrajectoryRecord(
                        log_s=message.log_time * 1.0e-9,
                        header_s=stamp_ns(obj.header.stamp) * 1.0e-9,
                        time_s=times,
                        xy=xy,
                        yaw=yaw,
                        velocity=velocity,
                    )
                )
            elif channel.topic == ODOMETRY_TOPIC:
                pose = obj.pose.pose
                odometry.append(
                    (
                        stamp_ns(obj.header.stamp) * 1.0e-9,
                        float(pose.position.x),
                        float(pose.position.y),
                        quat_yaw(pose.orientation),
                        float(obj.twist.twist.linear.x),
                    )
                )
            else:
                data = np.asarray(obj.data, dtype=float)
                if data.size >= 21:
                    diagnostic_time.append(message.log_time * 1.0e-9)
                    diagnostics.append(data)

    trajectories.sort(key=lambda item: item.log_s)
    odometry.sort(key=lambda item: item[0])
    odom = np.asarray(odometry, dtype=float)
    if not trajectories or odom.shape[0] < 2 or not diagnostics:
        raise RuntimeError("bag does not contain enough trajectory, odometry, and MPC diagnostics")
    return (
        trajectories,
        StateSeries(odom[:, 0], odom[:, 1], odom[:, 2], np.unwrap(odom[:, 3]), odom[:, 4]),
        np.asarray(diagnostic_time),
        np.asarray(diagnostics),
    )


def interpolate_state(series: StateSeries, time_s: float) -> np.ndarray:
    return np.asarray(
        [
            np.interp(time_s, series.time_s, series.x),
            np.interp(time_s, series.time_s, series.y),
            np.interp(time_s, series.time_s, series.yaw),
            np.interp(time_s, series.time_s, series.velocity),
        ]
    )


def interpolate_history(times: list[float], states: list[np.ndarray], query_s: float) -> np.ndarray:
    if query_s <= times[0]:
        return states[0].copy()
    if query_s >= times[-1]:
        return states[-1].copy()
    values = np.asarray(states)
    t = np.asarray(times)
    return np.asarray(
        [
            np.interp(query_s, t, values[:, 0]),
            np.interp(query_s, t, values[:, 1]),
            np.interp(query_s, t, np.unwrap(values[:, 2])),
        ]
    )


def transform_trajectory(
    trajectory: TrajectoryRecord, recorded_ego: np.ndarray, simulated_ego: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    c0, s0 = math.cos(recorded_ego[2]), math.sin(recorded_ego[2])
    c1, s1 = math.cos(simulated_ego[2]), math.sin(simulated_ego[2])
    delta = trajectory.xy - recorded_ego[:2]
    local = np.column_stack(
        (c0 * delta[:, 0] + s0 * delta[:, 1], -s0 * delta[:, 0] + c0 * delta[:, 1])
    )
    xy = (
        np.column_stack((c1 * local[:, 0] - s1 * local[:, 1], s1 * local[:, 0] + c1 * local[:, 1]))
        + simulated_ego[:2]
    )
    yaw = np.unwrap(trajectory.yaw + simulated_ego[2] - recorded_ego[2])
    return xy, yaw


def circle_curvature(xy: np.ndarray, smoothing_points: int) -> np.ndarray:
    count = xy.shape[0]
    result = np.zeros(count)
    if count < 3:
        return result
    width = min(smoothing_points, (count - 1) // 2)
    for i in range(width, count - width):
        p1, p2, p3 = xy[i - width], xy[i], xy[i + width]
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)
        denominator = a * b * c
        if denominator > 1.0e-9:
            cross = np.cross(p2 - p1, p3 - p1)
            result[i] = 2.0 * float(cross) / denominator
    result[:width] = result[width]
    result[count - width :] = result[count - width - 1]
    return result


def make_reference(
    trajectory: TrajectoryRecord,
    recorded_ego: np.ndarray,
    simulated_ego: np.ndarray,
    anchored: bool,
    spatial_step_m: float,
    curvature_smoothing_points: int,
) -> ReferencePath:
    xy, yaw = transform_trajectory(trajectory, recorded_ego, simulated_ego)
    times = trajectory.time_s.copy()
    velocity = trajectory.velocity.copy()
    if anchored:
        xy = np.vstack((simulated_ego[:2], xy))
        yaw = np.r_[simulated_ego[2], yaw]
        times = np.r_[0.0, times]
        velocity = np.r_[recorded_ego[3], velocity]

    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    keep = np.r_[True, np.diff(arc) > 1.0e-3]
    unique_s = arc[keep]
    unique_xy = xy[keep]
    if unique_s.size < 3 or unique_s[-1] < 2.0 * spatial_step_m:
        curvature_s = np.asarray([0.0, max(float(unique_s[-1]), spatial_step_m)])
        curvature = np.zeros(2)
    else:
        curvature_s = np.arange(0.0, unique_s[-1], spatial_step_m)
        if curvature_s[-1] < unique_s[-1] - 1.0e-6:
            curvature_s = np.r_[curvature_s, unique_s[-1]]
        spline_x = CubicSpline(unique_s, unique_xy[:, 0])
        spline_y = CubicSpline(unique_s, unique_xy[:, 1])
        spatial_xy = np.column_stack((spline_x(curvature_s), spline_y(curvature_s)))
        curvature = circle_curvature(spatial_xy, curvature_smoothing_points)
    return ReferencePath(trajectory.header_s, times, xy, yaw, velocity, arc, curvature_s, curvature)


def project_to_path(path: ReferencePath, state: np.ndarray) -> tuple[float, np.ndarray, float]:
    starts = path.xy[:-1]
    segments = path.xy[1:] - starts
    length_sq = np.sum(segments * segments, axis=1)
    ratios = np.divide(
        np.sum((state[:2] - starts) * segments, axis=1),
        length_sq,
        out=np.zeros_like(length_sq),
        where=length_sq > 1.0e-9,
    )
    ratios = np.clip(ratios, 0.0, 1.0)
    projections = starts + ratios[:, None] * segments
    index = int(np.argmin(np.sum((projections - state[:2]) ** 2, axis=1)))
    segment_length = math.sqrt(max(float(length_sq[index]), 0.0))
    arc = float(path.arc_s[index] + ratios[index] * segment_length)
    yaw = math.atan2(segments[index, 1], segments[index, 0])
    return arc, projections[index], yaw


def reference_at(
    path: ReferencePath,
    state: np.ndarray,
    now_s: float,
    mode: str,
    input_delay_s: float,
) -> tuple[np.ndarray, float, float, float]:
    if mode == "temporal":
        relative_time = float(np.clip(now_s - path.header_s, path.time_s[0], path.time_s[-1]))
        x = np.interp(relative_time, path.time_s, path.xy[:, 0])
        y = np.interp(relative_time, path.time_s, path.xy[:, 1])
        yaw = np.interp(relative_time, path.time_s, np.unwrap(path.yaw))
        velocity = np.interp(relative_time, path.time_s, path.velocity)
        target_time = min(relative_time + input_delay_s, path.time_s[-1])
        target_arc = np.interp(target_time, path.time_s, path.arc_s)
    else:
        nearest_arc, nearest_xy, yaw = project_to_path(path, state)
        x, y = nearest_xy
        velocity = np.interp(nearest_arc, path.arc_s, path.velocity)
        target_arc = min(nearest_arc + abs(float(velocity)) * input_delay_s, path.arc_s[-1])
    curvature = np.interp(target_arc, path.curvature_s, path.curvature)
    return np.asarray([x, y, yaw]), float(velocity), float(curvature), float(target_arc)


def fit_mpc_surrogate(diagnostics: np.ndarray) -> np.ndarray:
    # target Uex(0) from [feed-forward, lateral error, yaw error, measured steer, bias]
    valid = np.all(np.isfinite(diagnostics[:, [1, 2, 4, 5, 8, 10]]), axis=1) & (
        np.abs(diagnostics[:, 10]) > 0.5
    )
    data = diagnostics[valid]
    x = np.column_stack((data[:, 2], data[:, 5], data[:, 8], data[:, 4], np.ones(data.shape[0])))
    y = data[:, 1]
    ridge = 1.0e-5 * np.eye(x.shape[1])
    ridge[-1, -1] = 0.0
    return np.linalg.solve(x.T @ x + ridge, x.T @ y)


def summarize(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"rms": None, "p95": None, "p99": None, "max": None}
    return {
        "rms": float(np.sqrt(np.mean(values * values))),
        "p95": float(np.percentile(np.abs(values), 95)),
        "p99": float(np.percentile(np.abs(values), 99)),
        "max": float(np.max(np.abs(values))),
    }


def run_variant(
    trajectories: list[TrajectoryRecord],
    recorded: StateSeries,
    controller_coefficients: np.ndarray,
    mode: str,
    anchored: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    start_s = max(trajectories[0].log_s, recorded.time_s[0])
    end_s = min(trajectories[-1].log_s, recorded.time_s[-1])
    initial = interpolate_state(recorded, start_s)
    state = initial[:3].copy()
    steer = 0.0
    command = 0.0
    command_queue: list[tuple[float, float]] = [(start_s, 0.0)]
    history_time = [start_s]
    history_state = [state.copy()]
    plan_index = -1
    path: ReferencePath | None = None
    next_time = start_s

    command_values: list[float] = []
    steer_values: list[float] = []
    lateral_errors: list[float] = []
    yaw_errors: list[float] = []
    speeds: list[float] = []
    times: list[float] = []

    while next_time <= end_s:
        while (
            plan_index + 1 < len(trajectories) and trajectories[plan_index + 1].log_s <= next_time
        ):
            plan_index += 1
            trajectory = trajectories[plan_index]
            recorded_ego = interpolate_state(recorded, trajectory.header_s)
            simulated_header = interpolate_history(history_time, history_state, trajectory.header_s)
            simulated_ego = np.r_[simulated_header, recorded_ego[3]]
            path = make_reference(
                trajectory,
                recorded_ego,
                simulated_ego,
                anchored,
                args.spatial_step_m,
                args.curvature_smoothing_points,
            )

        velocity = float(np.interp(next_time, recorded.time_s, recorded.velocity))
        if path is not None:
            reference, _, curvature, _ = reference_at(
                path, state, next_time, mode, args.input_delay_s
            )
            dx = state[0] - reference[0]
            dy = state[1] - reference[1]
            lateral_error = -math.sin(reference[2]) * dx + math.cos(reference[2]) * dy
            yaw_error = float(wrap(state[2] - reference[2]))
            feed_forward = math.atan(args.wheelbase_m * curvature)
            features = np.asarray([feed_forward, lateral_error, yaw_error, steer, 1.0])
            raw_command = float(controller_coefficients @ features)
            raw_command = float(np.clip(raw_command, -args.steer_limit_rad, args.steer_limit_rad))
            max_step = args.steer_rate_limit_rad_s * args.control_period_s
            raw_command = float(np.clip(raw_command, command - max_step, command + max_step))
            alpha = 1.0 - math.exp(-2.0 * math.pi * args.command_lpf_hz * args.control_period_s)
            command += alpha * (raw_command - command)
        else:
            lateral_error = 0.0
            yaw_error = 0.0

        command_queue.append((next_time, command))
        delayed_time = next_time - args.steer_delay_s
        delayed_command = command_queue[0][1]
        for queued_time, queued_command in reversed(command_queue):
            if queued_time <= delayed_time:
                delayed_command = queued_command
                break
        while len(command_queue) > 2 and command_queue[1][0] < delayed_time - 1.0:
            command_queue.pop(0)
        steer += args.control_period_s * (delayed_command - steer) / args.steer_time_constant_s
        state[0] += args.control_period_s * velocity * math.cos(state[2])
        state[1] += args.control_period_s * velocity * math.sin(state[2])
        state[2] = float(
            wrap(state[2] + args.control_period_s * velocity * math.tan(steer) / args.wheelbase_m)
        )

        times.append(next_time)
        command_values.append(command)
        steer_values.append(steer)
        lateral_errors.append(lateral_error)
        yaw_errors.append(yaw_error)
        speeds.append(velocity)
        history_time.append(next_time + args.control_period_s)
        history_state.append(state.copy())
        next_time += args.control_period_s

    time = np.asarray(times)
    command_array = np.asarray(command_values)
    steer_array = np.asarray(steer_values)
    lateral_array = np.asarray(lateral_errors)
    yaw_array = np.asarray(yaw_errors)
    speed_array = np.asarray(speeds)
    valid = (time > start_s + args.warmup_s) & (np.abs(speed_array) > 0.5)
    command_rate = np.gradient(command_array, args.control_period_s)
    command_acceleration = np.gradient(command_rate, args.control_period_s)
    lateral_acceleration = speed_array**2 * np.tan(steer_array) / args.wheelbase_m
    lateral_jerk = np.gradient(lateral_acceleration, args.control_period_s)
    return {
        "mode": mode,
        "anchored": anchored,
        "samples": int(np.sum(valid)),
        "duration_s": float(time[-1] - time[0]),
        "command_rad": summarize(command_array[valid]),
        "command_rate_rad_s": summarize(command_rate[valid]),
        "command_acceleration_rad_s2": summarize(command_acceleration[valid]),
        "steer_rad": summarize(steer_array[valid]),
        "lateral_error_m": summarize(lateral_array[valid]),
        "yaw_error_rad": summarize(yaw_array[valid]),
        "lateral_jerk_m_s3": summarize(lateral_jerk[valid]),
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bag", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--control-period-s", type=float, default=0.03)
    p.add_argument("--input-delay-s", type=float, default=0.24)
    p.add_argument("--wheelbase-m", type=float, default=2.79)
    p.add_argument("--steer-delay-s", type=float, default=0.10)
    p.add_argument("--steer-time-constant-s", type=float, default=0.10)
    p.add_argument("--steer-limit-rad", type=float, default=0.60)
    p.add_argument("--steer-rate-limit-rad-s", type=float, default=1.0)
    p.add_argument("--command-lpf-hz", type=float, default=3.0)
    p.add_argument("--spatial-step-m", type=float, default=0.1)
    p.add_argument("--curvature-smoothing-points", type=int, default=15)
    p.add_argument("--warmup-s", type=float, default=2.0)
    return p


def main() -> int:
    args = parser().parse_args()
    trajectories, recorded, _, diagnostics = read_bag(args.bag)
    coefficients = fit_mpc_surrogate(diagnostics)
    variants = {
        "temporal_no_anchor": run_variant(
            trajectories, recorded, coefficients, "temporal", False, args
        ),
        "temporal_anchor": run_variant(
            trajectories, recorded, coefficients, "temporal", True, args
        ),
        "spatial_anchor": run_variant(trajectories, recorded, coefficients, "spatial", True, args),
    }
    report = {
        "bag": str(args.bag),
        "trajectory_count": len(trajectories),
        "controller_surrogate_coefficients": {
            name: float(value)
            for name, value in zip(
                ["feed_forward", "lateral_error", "yaw_error", "measured_steer", "bias"],
                coefficients,
                strict=False,
            )
        },
        "vehicle_model": {
            "wheelbase_m": args.wheelbase_m,
            "steer_delay_s": args.steer_delay_s,
            "steer_time_constant_s": args.steer_time_constant_s,
        },
        "variants": variants,
        "limitations": [
            "MPC commands are approximated by a linear surrogate fitted to bag diagnostics.",
            "Recorded trajectories are ego-frame re-centered, but perception and model inference are not rerun.",
            "Use the real MPC plus autoware_simple_planning_simulator before selecting a vehicle default.",
        ],
    }
    output = json.dumps(report, indent=2, allow_nan=False)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
