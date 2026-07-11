import math

import pytest
import torch

from planner_metrics.config import RewardConfig
from planner_metrics.subscores import (
    compute_ego_neighbor_signed_clearance,
    compute_road_border_penalty,
    compute_ttc_score_batch,
)


def _square_trajectory(center_x: float, center_y: float, steps: int) -> torch.Tensor:
    trajectory = torch.zeros(1, steps, 4)
    trajectory[..., 0] = center_x
    trajectory[..., 1] = center_y
    trajectory[..., 2] = 1.0
    return trajectory


def test_signed_obb_clearance_uses_euclidean_corner_gap():
    ego = _square_trajectory(0.0, 0.0, 1)
    neighbor = _square_trajectory(3.0, 4.0, 1)
    clearance = compute_ego_neighbor_signed_clearance(
        ego,
        torch.tensor([0.0, 2.0, 2.0]),
        neighbor,
        torch.tensor([[2.0, 2.0]]),
        torch.ones(1, 1, dtype=torch.bool),
    )

    assert clearance.shape == (1, 1, 1)
    assert clearance.item() == pytest.approx(math.sqrt(5.0), abs=1e-4)


def test_road_border_penalty_reports_perimeter_to_segment_distance():
    ego = _square_trajectory(0.0, 0.0, 4)
    line_strings = torch.zeros(1, 1, 2, 4)
    line_strings[0, 0, :, 0] = torch.tensor([-10.0, 10.0])
    line_strings[0, 0, :, 1] = 2.5
    line_strings[0, 0, :, 3] = 1.0

    gate, _, _, crossing_steps, _, per_timestep = compute_road_border_penalty(
        ego,
        torch.tensor([0.0, 2.0, 2.0]),
        {"line_strings": line_strings},
        RewardConfig(),
    )

    torch.testing.assert_close(gate, torch.ones(1))
    assert crossing_steps == [None]
    torch.testing.assert_close(per_timestep, torch.full((1, 4), 1.5), atol=1e-5, rtol=0)


def test_ttc_marks_the_full_horizon_before_a_future_collision():
    steps = 80
    ego = _square_trajectory(1.0, 0.0, steps)
    neighbor = _square_trajectory(100.0, 0.0, steps)
    neighbor[0, 30, :2] = torch.tensor([1.0, 0.0])

    ttc = compute_ttc_score_batch(
        ego,
        torch.tensor([0.0, 2.0, 2.0]),
        neighbor,
        torch.tensor([[2.0, 2.0]]),
        torch.ones(1, steps, dtype=torch.bool),
    )

    assert ttc["first_collision_steps"] == [30]
    assert ttc["first_unsafe_steps"] == [20]
    assert ttc["unsafe_at_t"][0, 20:31].all()
    assert not ttc["unsafe_at_t"][0, :20].any()
    assert ttc["score"].item() == pytest.approx(69.0 / 80.0, abs=1e-6)
