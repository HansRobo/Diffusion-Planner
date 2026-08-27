"""Placed obstacles must carry a real one-hot agent type into the model input.

Unknown is the 4th one-hot column (index 11), not an all-zero one-hot: the encoder reads
cols 8:12 with argmax, so an all-zero vector would tie-break to *vehicle* -- the placement
would silently become the thing the user was trying not to place.
"""

import torch

from scenario_generation.tools.scene_branch_editor import (
    _AGENT_TYPE_ONE_HOT_COL,
    _inject_obstacles_into_tensors,
)


class _Obstacle:
    """The duck-typed subset of ObstaclePlacement that the injector reads."""

    is_moving = False
    history_steps = 30

    def __init__(self, agent_type: str) -> None:
        self.x, self.y, self.yaw_rad = 3.0, 4.0, 0.0
        self.width, self.length = 2.0, 4.5
        self.agent_type = agent_type


def _inject(agent_type: str, width: int) -> torch.Tensor:
    data = {"neighbor_agents_past": torch.zeros(1, 5, 31, width)}
    out = _inject_obstacles_into_tensors(data, [_Obstacle(agent_type)], torch.device("cpu"))
    return out["neighbor_agents_past"]


def test_column_map_covers_every_offered_type():
    assert _AGENT_TYPE_ONE_HOT_COL == {"vehicle": 8, "pedestrian": 9, "bicycle": 10, "unknown": 11}


def test_each_type_sets_exactly_its_own_column():
    for agent_type, col in _AGENT_TYPE_ONE_HOT_COL.items():
        nap = _inject(agent_type, 12)
        one_hot = nap[0, 0, -1, 8:12]
        assert one_hot.sum() == 1.0, f"{agent_type}: {one_hot.tolist()}"
        assert one_hot[col - 8] == 1.0, f"{agent_type}: {one_hot.tolist()}"


def test_unknown_widens_a_legacy_11_column_tensor():
    """Recorded NPZs are 11 wide and have no Unknown column, so the injector must make one
    rather than indexing off the end."""
    nap = _inject("unknown", 11)
    assert nap.shape[-1] == 12
    assert nap[0, 0, -1, 8:12].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_known_types_leave_a_legacy_tensor_at_its_original_width():
    for agent_type in ("vehicle", "pedestrian", "bicycle"):
        assert _inject(agent_type, 11).shape[-1] == 11, agent_type


def test_unrecognised_type_still_falls_back_to_vehicle():
    nap = _inject("motorcycle", 12)
    assert nap[0, 0, -1, 8:12].tolist() == [1.0, 0.0, 0.0, 0.0]
