"""Tests for planner tensor normalization."""

from __future__ import annotations

import unittest

import torch

from diffusion_planner.models.normalizer import PlannerNormalizer


class PlannerNormalizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.normalizer = PlannerNormalizer(
            position_scale=10.0,
            speed_scale=5.0,
            vehicle_shape_scale=2.0,
        )

    def test_trajectory_round_trip(self) -> None:
        trajectory = torch.tensor([[[[20.0, -10.0, 1.0, 0.0]]]])

        normalized = self.normalizer.normalize_trajectory(trajectory)
        restored = self.normalizer.denormalize_trajectory(normalized)

        torch.testing.assert_close(
            normalized, torch.tensor([[[[2.0, -1.0, 1.0, 0.0]]]])
        )
        torch.testing.assert_close(restored, trajectory)

    def test_input_normalization_is_non_destructive(self) -> None:
        input_data = {
            "neighbor_agents_past": torch.tensor([[[[10.0, 20.0, 1.0, 0.0]]]]),
            "lanes": torch.full((1, 1, 1, 6), 10.0),
            "route_lanes": torch.full((1, 1, 1, 6), 10.0),
            "intersection_area": torch.full((1, 1, 1, 2), 10.0),
            "stop_lines": torch.full((1, 1, 1, 2), 10.0),
            "road_borders": torch.full((1, 1, 1, 2), 10.0),
            "goal_pose": torch.tensor([[20.0, -10.0, 1.0, 0.0]]),
            "lanes_speed_limit": torch.tensor([[[10.0]]]),
            "route_lanes_speed_limit": torch.tensor([[[10.0]]]),
            "agent_shape": torch.tensor([[[2.0, 4.0]]]),
            "ego_shape": torch.tensor([[4.0, 6.0, 2.0]]),
            "agent_label": torch.ones(1, 1, 3),
        }

        normalized = self.normalizer.normalize_input(input_data)

        self.assertEqual(input_data["goal_pose"][0, 0], 20.0)
        torch.testing.assert_close(
            normalized["goal_pose"], torch.tensor([[2.0, -1.0, 1.0, 0.0]])
        )
        torch.testing.assert_close(
            normalized["lanes_speed_limit"], torch.tensor([[[2.0]]])
        )
        torch.testing.assert_close(
            normalized["ego_shape"], torch.tensor([[2.0, 3.0, 1.0]])
        )
        self.assertIs(normalized["agent_label"], input_data["agent_label"])

    def test_normalizes_yaw_vectors(self) -> None:
        trajectory = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])

        normalized = self.normalizer.normalize_yaw_vector(trajectory)

        torch.testing.assert_close(
            normalized, torch.tensor([[[[1.0, 2.0, 0.6, 0.8]]]])
        )


if __name__ == "__main__":
    unittest.main()
