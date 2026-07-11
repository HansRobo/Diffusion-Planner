"""Centerline lateral offset from route_lanes in the current ego frame."""

from __future__ import annotations

import numpy as np


def lateral_offset_from_route_lanes(route_lanes: np.ndarray) -> float:
    """Min distance from ego origin to valid route centerline segments (m).

      At each closed-loop step the observation is re-centered so the live ego sits
    at the origin. Segment distance avoids a sampling-density-dependent positive bias.
    """
    lanes = np.asarray(route_lanes, dtype=np.float32)
    if lanes.ndim < 3 or lanes.shape[-1] < 2:
        return float("inf")
    xy = lanes[..., :2]
    feature_width = min(lanes.shape[-1], 8)
    valid = np.any(np.abs(lanes[..., :feature_width]) > 1e-6, axis=-1)
    seg_valid = valid[..., :-1] & valid[..., 1:]
    start = xy[..., :-1, :][seg_valid]
    end = xy[..., 1:, :][seg_valid]
    distances: list[np.ndarray] = []
    if start.size:
        delta = end - start
        denom = np.sum(delta * delta, axis=-1)
        projection = np.divide(
            -np.sum(start * delta, axis=-1),
            denom,
            out=np.zeros_like(denom),
            where=denom > 1e-12,
        )
        closest = start + np.clip(projection, 0.0, 1.0)[:, None] * delta
        distances.append(np.linalg.norm(closest, axis=-1))

    isolated = xy[valid]
    if isolated.size:
        distances.append(np.linalg.norm(isolated, axis=-1))
    if not distances:
        return float("inf")
    return float(min(float(distance.min()) for distance in distances if distance.size))


def lateral_offset_series(route_lanes_list: list[np.ndarray]) -> np.ndarray:
    return np.array(
        [lateral_offset_from_route_lanes(rl) for rl in route_lanes_list], dtype=np.float32
    )
