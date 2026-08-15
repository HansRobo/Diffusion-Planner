"""Physical-scale normalization for planner tensors."""

from __future__ import annotations

import torch
from torch import nn


class PlannerNormalizer(nn.Module):
    """Normalize continuous planner features without changing padding zeros."""

    position_scale: torch.Tensor
    speed_scale: torch.Tensor
    vehicle_shape_scale: torch.Tensor

    def __init__(
        self,
        position_scale: float = 50.0,
        speed_scale: float = 15.0,
        vehicle_shape_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.register_buffer("position_scale", torch.tensor(position_scale))
        self.register_buffer("speed_scale", torch.tensor(speed_scale))
        self.register_buffer(
            "vehicle_shape_scale", torch.tensor(vehicle_shape_scale)
        )

    def normalize_input(
        self, input_data: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Return a shallow-copied input map with continuous features normalized."""
        normalized = input_data.copy()
        if "ego_agent_past" in input_data:
            normalized["ego_agent_past"] = self._normalize_pose(
                input_data["ego_agent_past"]
            )
        normalized["neighbor_agents_past"] = self._normalize_pose(
            input_data["neighbor_agents_past"]
        )
        normalized["lanes"] = input_data["lanes"] / self.position_scale
        normalized["route_lanes"] = input_data["route_lanes"] / self.position_scale
        normalized["intersection_area"] = (
            input_data["intersection_area"] / self.position_scale
        )
        normalized["stop_lines"] = input_data["stop_lines"] / self.position_scale
        normalized["road_borders"] = (
            input_data["road_borders"] / self.position_scale
        )
        normalized["goal_pose"] = self._normalize_pose(input_data["goal_pose"])
        normalized["lanes_speed_limit"] = (
            input_data["lanes_speed_limit"] / self.speed_scale
        )
        normalized["route_lanes_speed_limit"] = (
            input_data["route_lanes_speed_limit"] / self.speed_scale
        )
        normalized["agent_shape"] = (
            input_data["agent_shape"] / self.vehicle_shape_scale
        )
        normalized["ego_shape"] = input_data["ego_shape"] / self.vehicle_shape_scale
        return normalized

    def normalize_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Normalize `(B, A, T, 4)` trajectories while preserving yaw vectors."""
        return self._normalize_pose(trajectory)

    def denormalize_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Restore meter units for `(B, A, T, 4)` trajectories."""
        return torch.cat(
            (trajectory[..., :2] * self.position_scale, trajectory[..., 2:]),
            dim=-1,
        )

    @staticmethod
    def normalize_yaw_vector(trajectory: torch.Tensor) -> torch.Tensor:
        """Project trajectory `(cos_yaw, sin_yaw)` pairs onto the unit circle."""
        yaw = trajectory[..., 2:4]
        norm = torch.linalg.vector_norm(yaw, dim=-1, keepdim=True).clamp_min(1e-6)
        normalized_yaw = yaw / norm
        return torch.cat((trajectory[..., :2], normalized_yaw), dim=-1)

    def _normalize_pose(self, values: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (values[..., :2] / self.position_scale, values[..., 2:]), dim=-1
        )
