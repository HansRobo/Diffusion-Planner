import json
import math
from pathlib import Path

import numpy as np

_FULL_TRACK_MATCH_TOLERANCE_M = 0.02
_FULL_TRACK_MATCH_MARGIN_M = 0.04
_FULL_TRACK_MIN_MOTION_M = 0.05
_ALIGNED_NEIGHBOR_FUTURE_VERSION = 3


def _next_indexed_npz_path(source_path: str) -> Path | None:
    path = Path(source_path)
    prefix, separator, index_text = path.stem.rpartition("_")
    if not separator or len(index_text) != 8 or not index_text.isdigit():
        return None
    return path.with_name(f"{prefix}_{int(index_text) + 1:08d}{path.suffix}")


def _sidecar_planar_pose(npz_path: Path) -> tuple[float, float, float, int] | None:
    sidecar_path = npz_path.with_suffix(".json")
    if not sidecar_path.is_file():
        return None
    with sidecar_path.open(encoding="utf-8") as sidecar_file:
        sidecar = json.load(sidecar_file)
    required = ("x", "y", "qw", "qx", "qy", "qz", "timestamp")
    if any(key not in sidecar for key in required):
        return None
    qw, qx, qy, qz = (float(sidecar[key]) for key in ("qw", "qx", "qy", "qz"))
    yaw = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    return float(sidecar["x"]), float(sidecar["y"]), yaw, int(sidecar["timestamp"])


def _transform_xy(xy: np.ndarray, pose: tuple[float, float, float]) -> np.ndarray:
    x, y, yaw = pose
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    transformed = np.empty_like(xy, dtype=np.float64)
    transformed[..., 0] = x + cos_yaw * xy[..., 0] - sin_yaw * xy[..., 1]
    transformed[..., 1] = y + sin_yaw * xy[..., 0] + cos_yaw * xy[..., 1]
    return transformed


def _verified_full_track_shift_mask(
    data: dict,
    source_path: str,
    candidates: np.ndarray,
) -> np.ndarray:
    """Verify a retained legacy seed against the next logged scene."""
    needs_shift = np.zeros_like(candidates)
    source = Path(source_path)
    next_path = _next_indexed_npz_path(source_path)
    if next_path is None or not next_path.is_file():
        return needs_shift

    source_pose = _sidecar_planar_pose(source)
    next_pose = _sidecar_planar_pose(next_path)
    if source_pose is None or next_pose is None:
        return needs_shift
    delta_seconds = (next_pose[3] - source_pose[3]) * 1e-9
    if not 0.05 <= delta_seconds <= 0.15:
        return needs_shift

    with np.load(next_path, allow_pickle=False) as next_archive:
        if "neighbor_agents_past" not in next_archive:
            return needs_shift
        next_past = next_archive["neighbor_agents_past"]
    if next_past.ndim != 3 or next_past.shape[-1] < 8:
        return needs_shift
    next_valid = np.any(next_past[:, -1, :8] != 0, axis=-1)
    if not np.any(next_valid):
        return needs_shift

    future = data["neighbor_agents_future"]
    candidate_indices = np.flatnonzero(candidates)
    candidate_points = _transform_xy(future[candidate_indices, :2, :2], source_pose[:3])
    next_points = _transform_xy(next_past[next_valid, -1, :2], next_pose[:3])
    distance = np.linalg.norm(
        candidate_points[:, :, None, :] - next_points[None, None, :, :], axis=-1
    )
    distance_from_first = distance[:, 0].min(axis=-1)
    distance_from_second = distance[:, 1].min(axis=-1)
    first_motion = np.linalg.norm(candidate_points[:, 1] - candidate_points[:, 0], axis=-1)
    verified = (
        (first_motion > _FULL_TRACK_MIN_MOTION_M)
        & (distance_from_second <= _FULL_TRACK_MATCH_TOLERANCE_M)
        & (distance_from_first - distance_from_second >= _FULL_TRACK_MATCH_MARGIN_M)
    )
    needs_shift[candidate_indices[verified]] = True
    return needs_shift


def align_legacy_neighbor_futures_on_load(
    data: dict,
    atol: float = 1e-4,
    source_path: str | None = None,
    data_version: int | None = None,
) -> None:
    """Fix legacy neighbor timing in memory without writing shared data.

    Short tracks are unambiguous because the old converter's current-frame seed
    leaves zero padding at the tail. A full track can also retain that seed when
    the neighbor disappears exactly at the horizon boundary. Those cases are
    shifted only after matching their second point to the next logged scene.
    """
    if data_version is not None and data_version >= _ALIGNED_NEIGHBOR_FUTURE_VERSION:
        return
    if "neighbor_agents_future" not in data or "neighbor_agents_past" not in data:
        return
    future = data["neighbor_agents_future"]
    past = data["neighbor_agents_past"]
    if future.ndim != 3 or past.ndim != 3 or future.shape[0] != past.shape[0]:
        return
    if future.shape[1] < 2:
        return

    valid = np.any(future != 0, axis=-1)
    valid_count = valid.sum(axis=-1)
    current_valid = np.any(past[:, -1, :8] != 0, axis=-1)
    first_matches_current = np.max(np.abs(future[:, 0, :2] - past[:, -1, :2]), axis=-1) <= atol
    needs_shift = (
        (valid_count > 0)
        & (valid_count < future.shape[1])
        & current_valid
        & valid[:, 0]
        & first_matches_current
    )
    if source_path is not None:
        full_track_candidates = (
            (valid_count == future.shape[1]) & current_valid & valid[:, 0] & first_matches_current
        )
        if np.any(full_track_candidates):
            needs_shift |= _verified_full_track_shift_mask(data, source_path, full_track_candidates)
    if not np.any(needs_shift):
        return

    aligned = future.copy()
    aligned[needs_shift, :-1] = future[needs_shift, 1:]
    aligned[needs_shift, -1] = 0
    data["neighbor_agents_future"] = aligned
