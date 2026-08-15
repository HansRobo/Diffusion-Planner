"""Tests for the conditional flow-matching planner."""

from __future__ import annotations

import unittest

import torch

from diffusion_planner.data.dimensions import (
    EGO_HISTORY_LENGTH,
    INTERSECTION_AREA_LENGTH,
    LANE_LENGTH,
    ROAD_BORDER_LENGTH,
    STOP_LINE_LENGTH,
    TRAFFIC_LIGHT_FUTURE_LENGTH,
    TRAFFIC_LIGHT_PAST_LENGTH,
    TRAJECTORY_LENGTH,
)
from diffusion_planner.models.diffusion_planner import DiffusionPlanner
from diffusion_planner.models.flow_matching import sample_time


def make_input_data() -> dict[str, torch.Tensor]:
    batch = 1
    neighbors = 2
    data = {
        "ego_agent_past": torch.zeros(batch, EGO_HISTORY_LENGTH, 6),
        "neighbor_agents_past": torch.zeros(
            batch, neighbors, EGO_HISTORY_LENGTH, 4
        ),
        "agent_shape": torch.zeros(batch, neighbors, 2),
        "agent_label": torch.zeros(batch, neighbors, 3),
        "lanes": torch.zeros(batch, 2, LANE_LENGTH, 6),
        "lane_types": torch.zeros(batch, 2, 20),
        "lanes_speed_limit": torch.zeros(batch, 2, 1),
        "lane_traffic_light_past": torch.zeros(
            batch, 2, TRAFFIC_LIGHT_PAST_LENGTH, 6
        ),
        "lane_traffic_light_future": torch.zeros(
            batch, 2, TRAFFIC_LIGHT_FUTURE_LENGTH, 6
        ),
        "route_lanes": torch.zeros(batch, 1, LANE_LENGTH, 6),
        "route_lane_types": torch.zeros(batch, 1, 20),
        "route_lanes_speed_limit": torch.zeros(batch, 1, 1),
        "route_traffic_light_past": torch.zeros(
            batch, 1, TRAFFIC_LIGHT_PAST_LENGTH, 6
        ),
        "route_traffic_light_future": torch.zeros(
            batch, 1, TRAFFIC_LIGHT_FUTURE_LENGTH, 6
        ),
        "intersection_area": torch.zeros(
            batch, 1, INTERSECTION_AREA_LENGTH, 2
        ),
        "stop_lines": torch.zeros(batch, 1, STOP_LINE_LENGTH, 2),
        "road_borders": torch.zeros(batch, 1, ROAD_BORDER_LENGTH, 2),
        "goal_pose": torch.tensor([[10.0, 0.0, 1.0, 0.0]]),
        "ego_shape": torch.tensor([[3.8, 4.9, 1.9]]),
        "ego_agent_future": torch.zeros(batch, TRAJECTORY_LENGTH, 6),
        "neighbor_agents_future": torch.zeros(
            batch, neighbors, TRAJECTORY_LENGTH, 4
        ),
    }
    data["ego_agent_past"][..., 2] = 1.0
    data["neighbor_agents_past"][:, 0, :, 2] = 1.0
    data["agent_shape"][:, 0] = torch.tensor([2.0, 4.5])
    data["agent_label"][:, 0, 0] = 1.0
    data["ego_agent_future"][..., 2] = 1.0
    data["neighbor_agents_future"][:, 0, :, 2] = 1.0
    return data


class DiffusionPlannerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = DiffusionPlanner(
            hidden_dim=16,
            num_heads=4,
            scene_fusion_depth=1,
            element_encoder_depth=1,
            decoder_depth=1,
            trajectory_encoder_depth=1,
            feedforward_dim=32,
            embed_dim=8,
        )
        self.input_data = make_input_data()

    def test_compute_loss(self) -> None:
        loss = self.model.compute_loss(self.input_data)

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_logistic_normal_time_is_inside_unit_interval(self) -> None:
        time = sample_time(
            128,
            torch.device("cpu"),
            torch.float32,
            self.model.time_mean,
            self.model.time_std,
        )

        self.assertTrue(torch.all(time > 0))
        self.assertTrue(torch.all(time < 1))

    def test_sample_encodes_scene_once_and_masks_missing_agents(self) -> None:
        call_count = 0
        decoder_call_count = 0

        def count_scene_calls(
            _module: torch.nn.Module,
            _args: tuple[dict[str, torch.Tensor]],
            _output: tuple[torch.Tensor, torch.Tensor],
        ) -> None:
            nonlocal call_count
            call_count += 1

        def count_decoder_calls(
            _module: torch.nn.Module,
            _args: tuple[torch.Tensor, ...],
            _output: torch.Tensor,
        ) -> None:
            nonlocal decoder_call_count
            decoder_call_count += 1

        handle = self.model.scene_encoder.register_forward_hook(count_scene_calls)
        decoder_handle = self.model.trajectory_decoder.register_forward_hook(
            count_decoder_calls
        )
        trajectories = self.model.sample(self.input_data, num_steps=2)
        handle.remove()
        decoder_handle.remove()

        self.assertEqual(trajectories.shape, (1, 3, TRAJECTORY_LENGTH, 4))
        torch.testing.assert_close(trajectories[:, 2], torch.zeros_like(trajectories[:, 2]))
        self.assertEqual(call_count, 1)
        self.assertEqual(decoder_call_count, 3)


if __name__ == "__main__":
    unittest.main()
