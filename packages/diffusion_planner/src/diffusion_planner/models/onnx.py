"""ONNX boundaries for scene encoding and iterative trajectory decoding."""

from __future__ import annotations

import torch
from torch import nn

from .decoder import TrajectoryDecoder
from .diffusion_planner import DiffusionPlanner
from .encoder import SceneEncoder

SCENE_INPUT_NAMES = (
    "ego_agent_past",
    "neighbor_agents_past",
    "agent_shape",
    "agent_label",
    "lanes",
    "lane_types",
    "lanes_speed_limit",
    "lane_traffic_light_past",
    "lane_traffic_light_future",
    "route_lanes",
    "route_lane_types",
    "route_lanes_speed_limit",
    "route_traffic_light_past",
    "route_traffic_light_future",
    "intersection_area",
    "stop_lines",
    "road_borders",
    "goal_pose",
    "ego_shape",
)


class SceneEncoderOnnxWrapper(nn.Module):
    """Expose scene encoding through positional tensor inputs for ONNX runtimes."""

    def __init__(self, scene_encoder: SceneEncoder) -> None:
        super().__init__()
        self.scene_encoder = scene_encoder

    def forward(
        self,
        ego_agent_past: torch.Tensor,
        neighbor_agents_past: torch.Tensor,
        agent_shape: torch.Tensor,
        agent_label: torch.Tensor,
        lanes: torch.Tensor,
        lane_types: torch.Tensor,
        lanes_speed_limit: torch.Tensor,
        lane_traffic_light_past: torch.Tensor,
        lane_traffic_light_future: torch.Tensor,
        route_lanes: torch.Tensor,
        route_lane_types: torch.Tensor,
        route_lanes_speed_limit: torch.Tensor,
        route_traffic_light_past: torch.Tensor,
        route_traffic_light_future: torch.Tensor,
        intersection_area: torch.Tensor,
        stop_lines: torch.Tensor,
        road_borders: torch.Tensor,
        goal_pose: torch.Tensor,
        ego_shape: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return scene tokens, scene mask, current agent poses, and agent mask."""
        input_data: dict[str, torch.Tensor] = dict(
            zip(
                SCENE_INPUT_NAMES,
                (
                    ego_agent_past,
                    neighbor_agents_past,
                    agent_shape,
                    agent_label,
                    lanes,
                    lane_types,
                    lanes_speed_limit,
                    lane_traffic_light_past,
                    lane_traffic_light_future,
                    route_lanes,
                    route_lane_types,
                    route_lanes_speed_limit,
                    route_traffic_light_past,
                    route_traffic_light_future,
                    intersection_area,
                    stop_lines,
                    road_borders,
                    goal_pose,
                    ego_shape,
                ),
                strict=True,
            )
        )
        scene, scene_mask = self.scene_encoder(input_data)
        ego_pose = ego_agent_past[:, -1, :4].unsqueeze(1)
        neighbor_pose = neighbor_agents_past[:, :, -1, :4]
        agent_pose = torch.cat((ego_pose, neighbor_pose), dim=1)
        neighbor_mask = neighbor_agents_past.abs().sum(dim=(-2, -1)) == 0
        ego_mask = torch.zeros_like(neighbor_mask[:, :1])
        agent_mask = torch.cat((ego_mask, neighbor_mask), dim=1)
        return scene, scene_mask, agent_pose, agent_mask


class TrajectoryDecoderOnnxWrapper(nn.Module):
    """Expose one x0-prediction call independently from the sampling loop."""

    def __init__(self, trajectory_decoder: TrajectoryDecoder) -> None:
        super().__init__()
        self.trajectory_decoder = trajectory_decoder

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        scene: torch.Tensor,
        scene_mask: torch.Tensor,
        agent_pose: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        """Predict normalized clean trajectories with shape `(B, A, T, 4)`."""
        return self.trajectory_decoder(x, x_mask, scene, scene_mask, agent_pose, time)


class DiffusionPlannerSamplerOnnxWrapper(nn.Module):
    """Expose fixed 10-step Heun sampling as one ONNX graph."""

    def __init__(self, planner: DiffusionPlanner) -> None:
        super().__init__()
        self.planner = planner

    def forward(
        self,
        initial_noise: torch.Tensor,
        ego_agent_past: torch.Tensor,
        neighbor_agents_past: torch.Tensor,
        agent_shape: torch.Tensor,
        agent_label: torch.Tensor,
        lanes: torch.Tensor,
        lane_types: torch.Tensor,
        lanes_speed_limit: torch.Tensor,
        lane_traffic_light_past: torch.Tensor,
        lane_traffic_light_future: torch.Tensor,
        route_lanes: torch.Tensor,
        route_lane_types: torch.Tensor,
        route_lanes_speed_limit: torch.Tensor,
        route_traffic_light_past: torch.Tensor,
        route_traffic_light_future: torch.Tensor,
        intersection_area: torch.Tensor,
        stop_lines: torch.Tensor,
        road_borders: torch.Tensor,
        goal_pose: torch.Tensor,
        ego_shape: torch.Tensor,
        turn_indicators: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate trajectories and turn-indicator logits."""
        input_data: dict[str, torch.Tensor] = dict(
            zip(
                SCENE_INPUT_NAMES,
                (
                    ego_agent_past,
                    neighbor_agents_past,
                    agent_shape,
                    agent_label,
                    lanes,
                    lane_types,
                    lanes_speed_limit,
                    lane_traffic_light_past,
                    lane_traffic_light_future,
                    route_lanes,
                    route_lane_types,
                    route_lanes_speed_limit,
                    route_traffic_light_past,
                    route_traffic_light_future,
                    intersection_area,
                    stop_lines,
                    road_borders,
                    goal_pose,
                    ego_shape,
                ),
                strict=True,
            )
        )
        input_data["turn_indicators"] = turn_indicators
        return self.planner.sample(
            input_data, initial_noise, num_steps=10, time_epsilon=1e-5
        )
