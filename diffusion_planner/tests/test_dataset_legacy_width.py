"""DiffusionPlannerData / DiffusionPlannerPairData must transparently widen real, on-disk
11-col neighbor_agents_past (the actual production corpus format today -- the external C++
generator that would emit 12-col data hasn't been updated) to the 12-col width the model
now requires, so training keeps working on existing datasets without regenerating them."""

import json

import numpy as np
import pytest

from diffusion_planner.utils.dataset import (
    DiffusionPlannerData,
    DiffusionPlannerPairData,
    _match_neighbor_past_width,
)

_KEYS_SHAPES = {
    "ego_agent_past": (31, 4),
    "ego_current_state": (10,),
    "ego_agent_future": (80, 4),
    "neighbor_agents_future": (32, 80, 4),
    "static_objects": (5, 10),
    "lanes": (1, 20, 33),
    "lanes_speed_limit": (1, 1),
    "lanes_has_speed_limit": (1, 1),
    "route_lanes": (1, 20, 33),
    "route_lanes_speed_limit": (1, 1),
    "route_lanes_has_speed_limit": (1, 1),
    "goal_pose": (4,),
    "turn_indicators": (31,),
    "ego_shape": (3,),
}


def _write_scene_npz(path, neighbor_cols: int):
    data = {k: np.zeros(shape, dtype=np.float32) for k, shape in _KEYS_SHAPES.items()}
    data["lanes_has_speed_limit"] = data["lanes_has_speed_limit"].astype(bool)
    data["route_lanes_has_speed_limit"] = data["route_lanes_has_speed_limit"].astype(bool)
    data["turn_indicators"] = data["turn_indicators"].astype(np.int64)
    data["neighbor_agents_past"] = np.ones((32, 31, neighbor_cols), dtype=np.float32)
    data["version"] = np.array(1, dtype=np.int64)
    np.savez(path, **data)


def test_match_neighbor_past_width_widens_legacy_11col():
    legacy = np.ones((32, 31, 11), dtype=np.float32)
    widened = _match_neighbor_past_width(legacy)
    assert widened.shape == (32, 31, 12)
    assert np.array_equal(widened[..., :11], legacy)
    assert np.all(widened[..., 11] == 0.0)


def test_match_neighbor_past_width_is_noop_at_12():
    x = np.ones((32, 31, 12), dtype=np.float32)
    assert _match_neighbor_past_width(x) is x


def test_match_neighbor_past_width_rejects_unexpected_width():
    with pytest.raises(ValueError):
        _match_neighbor_past_width(np.zeros((32, 31, 9), dtype=np.float32))


def test_dataset_loads_real_legacy_11col_npz_as_12col(tmp_path):
    npz_path = tmp_path / "scene_0.npz"
    _write_scene_npz(npz_path, neighbor_cols=11)
    data_list = tmp_path / "list.json"
    data_list.write_text(json.dumps([str(npz_path)]))

    ds = DiffusionPlannerData(str(data_list))
    sample = ds[0]
    assert sample["neighbor_agents_past"].shape == (32, 31, 12)
    assert np.all(sample["neighbor_agents_past"][..., 11] == 0.0)


def test_dataset_passes_through_already_12col_npz_unchanged(tmp_path):
    npz_path = tmp_path / "scene_0.npz"
    _write_scene_npz(npz_path, neighbor_cols=12)
    data_list = tmp_path / "list.json"
    data_list.write_text(json.dumps([str(npz_path)]))

    ds = DiffusionPlannerData(str(data_list))
    sample = ds[0]
    assert sample["neighbor_agents_past"].shape == (32, 31, 12)


def test_pair_dataset_widens_both_frames(tmp_path):
    path_a = tmp_path / "scene_0.npz"
    path_b = tmp_path / "scene_1.npz"
    _write_scene_npz(path_a, neighbor_cols=11)
    _write_scene_npz(path_b, neighbor_cols=11)
    data_list = tmp_path / "list.json"
    data_list.write_text(json.dumps([str(path_a), str(path_b)]))

    ds = DiffusionPlannerPairData(str(data_list), expected_gap=None)
    assert len(ds) >= 1
    pair = ds[0]
    assert pair["current"]["neighbor_agents_past"].shape[-1] == 12
    assert pair["next"]["neighbor_agents_past"].shape[-1] == 12
