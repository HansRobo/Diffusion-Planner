"""Nearest centerline *point* distance from route_lanes in the current ego frame.

This is **not** a lateral (perpendicular-to-polyline) offset. At each closed-loop
step the observation is re-centered so the live ego sits at the origin; the
metric is the Euclidean distance from ``(0, 0)`` to the nearest valid
``route_lanes`` centerline sample (xy). Sparse samples or points far ahead on
the route can therefore inflate the value even when the ego is on-center.
"""

from __future__ import annotations

import numpy as np


def _valid_centerline_points(route_lanes: np.ndarray) -> np.ndarray:
    valid = np.abs(route_lanes[..., :2]).sum(axis=-1) > 1e-6
    pts = route_lanes[..., :2][valid]
    if pts.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return pts.reshape(-1, 2).astype(np.float32)


def nearest_centerline_point_dist_m(route_lanes: np.ndarray) -> float:
    """Min Euclidean distance (m) from ego origin to any route_lanes centerline point.

    Args:
        route_lanes: Route lane tensor in the live-ego frame; uses channels ``[..., :2]``.

    Returns:
        Distance in meters, or ``+inf`` when no valid centerline points exist.
    """
    pts = _valid_centerline_points(np.asarray(route_lanes))
    if pts.shape[0] == 0:
        return float("inf")
    return float(np.linalg.norm(pts, axis=-1).min())


def nearest_centerline_point_dist_series(route_lanes_list: list[np.ndarray]) -> np.ndarray:
    """Per-frame :func:`nearest_centerline_point_dist_m` over a list of route_lanes."""
    return np.array(
        [nearest_centerline_point_dist_m(rl) for rl in route_lanes_list], dtype=np.float32
    )
