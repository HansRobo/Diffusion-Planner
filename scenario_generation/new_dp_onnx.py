"""Compatibility adapter from the legacy planner tensors to the new DP sampler ONNX.

The new model split several packed legacy tensors into dedicated inputs.  This
module deliberately contains only that boundary conversion; it does not import
the new model implementation.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch


NEW_DP_INPUT_NAMES = (
    "ego_agent_past", "neighbor_agents_past", "agent_shape", "agent_label",
    "lanes", "lane_types", "lanes_speed_limit", "lane_traffic_light_past",
    "lane_traffic_light_future", "route_lanes", "route_lane_types",
    "route_lanes_speed_limit", "route_traffic_light_past",
    "route_traffic_light_future", "intersection_area", "stop_lines",
    "road_borders", "goal_pose", "ego_shape", "turn_indicators",
)


class IdentityNormalizer:
    """Normalizer contract used by the old evaluator before this adapter runs."""

    def __call__(self, data):
        return data


def compatibility_model_args():
    """Minimal old-evaluator configuration for the fixed new-DP ONNX schema."""
    return SimpleNamespace(
        observation_normalizer=IdentityNormalizer(),
        predicted_neighbor_num=320,
        future_len=80,
        new_dp_onnx=True,
    )


def _numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _batched(data: dict, key: str) -> np.ndarray:
    value = _numpy(data[key])
    # Inputs reaching the model from the old evaluators always have a batch
    # dimension.  Direct NPZ callers do not, so add it based on known fields.
    unbatched_ndim = {
        "ego_agent_past": 2, "neighbor_agents_past": 3, "lanes": 3,
        "route_lanes": 3, "polygons": 3, "line_strings": 3,
        "goal_pose": 1, "ego_shape": 1, "turn_indicators": 1,
        "lanes_speed_limit": 2, "route_lanes_speed_limit": 2,
    }[key]
    return value[None] if value.ndim == unbatched_ndim else value


def _pose4(value: np.ndarray) -> np.ndarray:
    if value.shape[-1] == 4:
        return value.astype(np.float32, copy=False)
    heading = value[..., 2]
    return np.concatenate(
        [value[..., :2], np.cos(heading)[..., None], np.sin(heading)[..., None]],
        axis=-1,
    ).astype(np.float32)


def _ego_history(data: dict) -> np.ndarray:
    pose = _pose4(_batched(data, "ego_agent_past"))
    out = np.zeros((*pose.shape[:-1], 6), dtype=np.float32)
    out[..., :4] = pose
    # Old NPZ has kinematics only for the current frame.  Finite differences
    # reconstruct history more faithfully than broadcasting the current speed.
    delta = np.diff(pose[..., :2], axis=-2, prepend=pose[..., :1, :2])
    out[..., 4] = np.linalg.norm(delta, axis=-1) / 0.1
    yaw = np.unwrap(np.arctan2(pose[..., 3], pose[..., 2]), axis=-1)
    out[..., 5] = np.diff(yaw, axis=-1, prepend=yaw[..., :1]) / 0.1
    if "ego_current_state" in data:
        state = _numpy(data["ego_current_state"])
        if state.ndim == 1:
            state = state[None]
        out[:, -1, 4] = np.linalg.norm(state[:, 4:6], axis=-1)
        out[:, -1, 5] = state[:, 9]
    return out


def _lane_parts(packed: np.ndarray):
    geometry = packed[..., [0, 1, 4, 5, 6, 7]].astype(np.float32)
    # Boundary type is constant along a segment; max is robust to padded points.
    lane_types = packed[..., 13:33].max(axis=-2).astype(np.float32)
    light5 = packed[..., 8:13].max(axis=-2).astype(np.float32)
    light6 = np.zeros((*light5.shape[:-1], 6), dtype=np.float32)
    light6[..., :3] = light5[..., :3]
    light6[..., 4] = np.maximum(light5[..., 3], light5[..., 4])
    # Legacy data has one current light state, not a temporal signal history.
    past = np.repeat(light6[..., None, :], 31, axis=-2)
    future = np.repeat(light6[..., None, :], 80, axis=-2)
    return geometry, lane_types, past, future


def _split_lines(lines: np.ndarray, flag_index: int, count: int, points: int) -> np.ndarray:
    batch = lines.shape[0]
    out = np.zeros((batch, count, points, 2), dtype=np.float32)
    for b in range(batch):
        selected = lines[b, np.any(lines[b, :, :, flag_index] > 0.5, axis=-1), :, :2]
        n = min(count, selected.shape[0])
        p = min(points, selected.shape[1] if n else 0)
        if n and p:
            out[b, :n, :p] = selected[:n, :p]
    return out


def legacy_to_new_dp(data: dict) -> dict[str, np.ndarray]:
    """Convert batched legacy evaluator tensors into raw new-DP inputs."""
    neighbors = _batched(data, "neighbor_agents_past").astype(np.float32)
    lanes = _batched(data, "lanes").astype(np.float32)
    routes = _batched(data, "route_lanes").astype(np.float32)
    lane_geom, lane_types, lane_tl_past, lane_tl_future = _lane_parts(lanes)
    route_geom, route_types, route_tl_past, route_tl_future = _lane_parts(routes)
    polygons = _batched(data, "polygons").astype(np.float32)
    lines = _batched(data, "line_strings").astype(np.float32)
    goal = _pose4(_batched(data, "goal_pose"))

    result = {
        "ego_agent_past": _ego_history(data),
        "neighbor_agents_past": neighbors[..., :4],
        "agent_shape": neighbors[:, :, -1, 6:8],
        "agent_label": neighbors[:, :, -1, 8:11],
        "lanes": lane_geom,
        "lane_types": lane_types,
        "lanes_speed_limit": _batched(data, "lanes_speed_limit").astype(np.float32),
        "lane_traffic_light_past": lane_tl_past,
        "lane_traffic_light_future": lane_tl_future,
        "route_lanes": route_geom,
        "route_lane_types": route_types,
        "route_lanes_speed_limit": _batched(data, "route_lanes_speed_limit").astype(np.float32),
        "route_traffic_light_past": route_tl_past,
        "route_traffic_light_future": route_tl_future,
        "intersection_area": polygons[..., :2],
        "stop_lines": _split_lines(lines, 2, 30, 2),
        "road_borders": _split_lines(lines, 3, 30, 20),
        "goal_pose": goal,
        "ego_shape": _batched(data, "ego_shape").astype(np.float32),
        "turn_indicators": _batched(data, "turn_indicators").astype(np.float32),
    }
    return result


def normalize_new_dp(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Apply the fixed normalization used by the new DP training package."""
    result = {key: np.array(value, copy=True, dtype=np.float32) for key, value in data.items()}
    for key in ("ego_agent_past", "neighbor_agents_past", "goal_pose"):
        result[key][..., :2] /= 50.0
    for key in ("lanes", "route_lanes", "intersection_area", "stop_lines", "road_borders"):
        result[key] /= 50.0
    for key in ("lanes_speed_limit", "route_lanes_speed_limit"):
        result[key] /= 15.0
    for key in ("agent_shape", "ego_shape"):
        result[key] /= 10.0
    return result


def denormalize_trajectory(trajectory: np.ndarray) -> np.ndarray:
    result = np.array(trajectory, copy=True)
    result[..., :2] *= 50.0
    yaw = result[..., 2:4]
    result[..., 2:4] = yaw / np.maximum(np.linalg.norm(yaw, axis=-1, keepdims=True), 1e-6)
    return result
