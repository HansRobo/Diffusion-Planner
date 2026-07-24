"""Unit tests for the standalone scene_features library.

Synthetic tensors only (no model, no data files); a minimal NPZ is written to a
tmp path for the end-to-end entry point. Verifies the numeric features, the
path-relevant interaction flags, and the label taxonomy.
"""

from __future__ import annotations

import numpy as np
import pytest

from rlvr.autoresearch import scene_features as sf


# --------------------------------------------------------------------------- #
# helpers to build synthetic scenes
# --------------------------------------------------------------------------- #
def _straight_ego_future(n: int = 80, step: float = 0.5) -> np.ndarray:
    """Ego drives straight along +x (ego frame): [x, y, yaw]."""
    fut = np.zeros((n, 3), np.float32)
    fut[:, 0] = np.arange(n, dtype=np.float32) * step
    return fut


def _turning_ego_future(total_yaw_deg: float, n: int = 80) -> np.ndarray:
    """Ego future with a smooth heading ramp to total_yaw_deg (signed)."""
    fut = np.zeros((n, 3), np.float32)
    yaw = np.linspace(0.0, np.radians(total_yaw_deg), n).astype(np.float32)
    fut[:, 2] = yaw
    # integrate a unit-ish speed path so x/y move (not required for yaw metric)
    dx = np.cos(yaw) * 0.5
    dy = np.sin(yaw) * 0.5
    fut[:, 0] = np.cumsum(dx)
    fut[:, 1] = np.cumsum(dy)
    return fut


def _neighbor_row(x, y, *, is_veh=0, is_ped=0, is_bike=0):
    """One neighbor feature row (11 cols): [x,y,cos,sin,vx,vy,w,l,veh,ped,bike]."""
    return [x, y, 1.0, 0.0, 0.0, 0.0, 0.6, 1.0, is_veh, is_ped, is_bike]


def _neighbors(rows, n_slots=8, t=2):
    """Pack neighbor rows into (n_slots, t, 11); unused slots stay zero (padding)."""
    nap = np.zeros((n_slots, t, 11), np.float32)
    for i, r in enumerate(rows):
        nap[i, :, :] = np.array(r, np.float32)
    return nap


def _ego_state(speed_ms: float) -> np.ndarray:
    s = np.zeros(10, np.float32)
    s[2] = 1.0  # cos heading
    s[4] = speed_ms  # vx
    return s


# --------------------------------------------------------------------------- #
# geometry reuse
# --------------------------------------------------------------------------- #
def test_min_dist_to_polyline_straight():
    poly = np.array([[0.0, 0.0], [40.0, 0.0]], np.float32)  # along +x
    pts = np.array([[5.0, 3.0], [10.0, -1.0], [20.0, 0.0]], np.float32)
    d = sf.min_dist_to_polyline(pts, poly)
    assert np.allclose(d, [3.0, 1.0, 0.0], atol=1e-4)


def test_batch_min_dist_matches_canonical_per_scene():
    """The batched GPU helper must equal the canonical per-scene reuse (the basis
    for GPU==CPU identity in the extractor)."""
    import torch

    rng = np.random.default_rng(0)
    B, Q, P = 5, 7, 80
    dists_ref = np.zeros((B, Q), np.float32)
    pts = np.zeros((B, Q, 2), np.float32)
    p1 = np.zeros((B, P - 1, 2), np.float32)
    p2 = np.zeros((B, P - 1, 2), np.float32)
    for b in range(B):
        path = np.cumsum(rng.normal(0, 1, size=(P, 2)).astype(np.float32), axis=0)
        points = rng.normal(0, 5, size=(Q, 2)).astype(np.float32)
        pts[b] = points
        p1[b], p2[b] = path[:-1], path[1:]
        dists_ref[b] = sf.min_dist_to_polyline(points, path)  # canonical per-scene
    out = sf.batch_min_dist_to_paths(
        torch.from_numpy(pts), torch.from_numpy(p1), torch.from_numpy(p2)
    ).numpy()
    assert np.allclose(out, dists_ref, atol=1e-4)


def test_min_dist_to_polyline_empty_and_degenerate():
    poly = np.array([[0.0, 0.0], [10.0, 0.0]], np.float32)
    assert sf.min_dist_to_polyline(np.zeros((0, 2), np.float32), poly).shape == (0,)
    # degenerate polyline (single vertex) -> point-to-point
    d = sf.min_dist_to_polyline(
        np.array([[3.0, 4.0]], np.float32), np.array([[0.0, 0.0]], np.float32)
    )
    assert np.isclose(d[0], 5.0, atol=1e-4)


# --------------------------------------------------------------------------- #
# scalar features / label bins
# --------------------------------------------------------------------------- #
def test_speed_bin_boundaries():
    bins = (0.5, 3.0, 8.0)
    assert sf.speed_bin(0.0, bins) == "stopped"
    assert sf.speed_bin(2.9, bins) == "low"
    assert sf.speed_bin(5.0, bins) == "mid"
    assert sf.speed_bin(12.0, bins) == "high"


def test_maneuver_sign():
    assert sf.maneuver(5.0, 15.0) == "straight"
    assert sf.maneuver(40.0, 15.0) == "turn-L"
    assert sf.maneuver(-40.0, 15.0) == "turn-R"


def test_signed_total_yaw_matches_maneuver():
    assert sf.signed_total_yaw_deg(_turning_ego_future(60.0)) > 15.0
    assert sf.signed_total_yaw_deg(_turning_ego_future(-60.0)) < -15.0
    assert abs(sf.signed_total_yaw_deg(_straight_ego_future())) < 1.0


def test_interaction_priority_ped_first():
    # a lead vehicle AND a relevant pedestrian both present -> has-ped wins
    assert (
        sf.interaction(
            n_interacting=10, has_lead=True, has_pedestrian=True, static_count=3, dense_count=4
        )
        == "has-ped"
    )
    assert (
        sf.interaction(
            n_interacting=1, has_lead=True, has_pedestrian=False, static_count=0, dense_count=4
        )
        == "has-lead"
    )
    assert (
        sf.interaction(
            n_interacting=5, has_lead=False, has_pedestrian=False, static_count=0, dense_count=4
        )
        == "dense"
    )
    assert (
        sf.interaction(
            n_interacting=0, has_lead=False, has_pedestrian=False, static_count=2, dense_count=4
        )
        == "has-static"
    )
    assert (
        sf.interaction(
            n_interacting=0, has_lead=False, has_pedestrian=False, static_count=0, dense_count=4
        )
        == "none"
    )


# --------------------------------------------------------------------------- #
# path-relevant neighbor flags — the fix
# --------------------------------------------------------------------------- #
def test_pedestrian_near_path_flagged():
    ego_future = _straight_ego_future()  # along +x
    nap = _neighbors([_neighbor_row(10.0, 1.0, is_ped=1)])  # 1 m off the path
    out = sf.neighbor_stats(
        nap,
        ego_future,
        radius_m=30.0,
        lead_range_m=40.0,
        lead_half_width_m=2.0,
        interaction_corridor_m=8.0,
        ped_corridor_m=6.0,
    )
    assert out["has_pedestrian"] is True
    assert out["nearest_ped_path_m"] < 6.0


def test_pedestrian_far_from_path_not_flagged():
    ego_future = _straight_ego_future()
    nap = _neighbors([_neighbor_row(10.0, 25.0, is_ped=1)])  # 25 m off the path
    out = sf.neighbor_stats(
        nap,
        ego_future,
        radius_m=30.0,
        lead_range_m=40.0,
        lead_half_width_m=2.0,
        interaction_corridor_m=8.0,
        ped_corridor_m=6.0,
    )
    assert out["has_pedestrian"] is False
    assert out["nearest_ped_path_m"] > 6.0  # present but far -> recorded, not flagged


def test_lead_vehicle_front_cone():
    ego_future = _straight_ego_future()
    nap = _neighbors([_neighbor_row(15.0, 0.0, is_veh=1)])  # directly ahead
    out = sf.neighbor_stats(
        nap,
        ego_future,
        radius_m=30.0,
        lead_range_m=40.0,
        lead_half_width_m=2.0,
        interaction_corridor_m=8.0,
        ped_corridor_m=6.0,
    )
    assert out["has_lead"] is True
    assert out["nearest_ped_path_m"] is None  # no pedestrians present


def test_no_active_neighbors():
    out = sf.neighbor_stats(
        np.zeros((8, 2, 11), np.float32),
        _straight_ego_future(),
        radius_m=30.0,
        lead_range_m=40.0,
        lead_half_width_m=2.0,
        interaction_corridor_m=8.0,
        ped_corridor_m=6.0,
    )
    assert out["neighbor_count"] == 0
    assert out["n_interacting"] == 0
    assert out["has_lead"] is False
    assert out["has_pedestrian"] is False


# --------------------------------------------------------------------------- #
# end-to-end entry point
# --------------------------------------------------------------------------- #
def _write_npz(path, ego_future, nap, speed_ms=5.0):
    np.savez(
        path,
        ego_current_state=_ego_state(speed_ms),
        ego_agent_future=ego_future.astype(np.float32),
        neighbor_agents_past=nap.astype(np.float32),
        static_objects=np.zeros((5, 10), np.float32),
        goal_pose=np.array([20.0, 0.0, 0.0], np.float32),
        ego_shape=np.array([4.76, 7.24, 2.29], np.float32),
    )


def test_extract_scene_features_end_to_end(tmp_path):
    p = tmp_path / "scene_00000042.npz"
    _write_npz(
        p, _turning_ego_future(50.0), _neighbors([_neighbor_row(8.0, 1.5, is_ped=1)]), speed_ms=5.0
    )
    row = sf.extract_scene_features(str(p), with_sidecar=False)
    # every declared column present
    assert set(row) == set(sf.FEATURE_COLUMNS)
    assert row["bag"] == "scene"
    assert row["frame"] == 42
    assert row["speed_bin"] == "mid"
    assert row["maneuver"] == "turn-L"
    assert row["interaction"] == "has-ped"
    assert row["label"] == "mid|turn-L|has-ped"
    assert row["has_pedestrian"] is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
