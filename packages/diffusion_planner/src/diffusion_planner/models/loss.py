"""Diffusion-planner-specific training loss construction."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

from diffusion_planner.data.dimensions import TRAJECTORY_DIM

from .flow_matching import compute_x0_flow_matching_loss, x0_velocity_error

PlannerModel = Callable[
    [torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor],
    torch.Tensor,
]


def trajectory_error_in_target_frame(
    error: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Rotate global xy error into target longitudinal/lateral coordinates."""
    position_error = error[..., :2]
    target_cos = target[..., 2]
    target_sin = target[..., 3]
    longitudinal = (
        position_error[..., 0] * target_cos + position_error[..., 1] * target_sin
    )
    lateral = -position_error[..., 0] * target_sin + position_error[..., 1] * target_cos
    return torch.cat(
        (longitudinal.unsqueeze(-1), lateral.unsqueeze(-1), error[..., 2:]),
        dim=-1,
    )


def trajectory_huber_loss(
    x_prediction: torch.Tensor,
    target: torch.Tensor,
    time: torch.Tensor,
    time_epsilon: float,
    ego_loss_weight: float = 1.0,
    neighbor_loss_weight: float = 1.0,
) -> torch.Tensor:
    """Apply agent-weighted Huber loss after target-frame position rotation."""
    target_frame_error = trajectory_error_in_target_frame(x_prediction - target, target)
    target_frame_error = x0_velocity_error(target_frame_error, time, time_epsilon)
    elementwise_loss = F.huber_loss(
        target_frame_error,
        torch.zeros_like(target_frame_error),
        reduction="none",
    )
    agent_weights = torch.cat(
        (
            elementwise_loss.new_full((1,), ego_loss_weight),
            elementwise_loss.new_full(
                (elementwise_loss.shape[1] - 1,), neighbor_loss_weight
            ),
        )
    )
    return elementwise_loss * agent_weights.view(1, -1, 1, 1)


def create_target_trajectory(
    input_data: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Combine labels into `(B, A, T, 4)` ego and neighbor trajectories."""
    ego_future = input_data["ego_agent_future"][..., :TRAJECTORY_DIM].unsqueeze(1)
    return torch.cat((ego_future, input_data["neighbor_agents_future"]), dim=1)


def compute_diffusion_planner_loss(
    model: PlannerModel,
    input_data: dict[str, torch.Tensor],
    *,
    time_mean: float,
    time_std: float,
    time_epsilon: float,
    noise_scale: float,
    ego_loss_weight: float = 1.0,
    neighbor_loss_weight: float = 1.0,
) -> torch.Tensor:
    """Compute masked x0 flow-matching loss for one planner batch."""
    target = create_target_trajectory(input_data)
    training_mask = (torch.count_nonzero(target, dim=-1) == 0).any(dim=-1)
    return compute_x0_flow_matching_loss(
        x0_model=lambda state, time: model(state, training_mask, input_data, time),
        loss_function=lambda x_prediction, clean_target, time: trajectory_huber_loss(
            x_prediction,
            clean_target,
            time,
            time_epsilon,
            ego_loss_weight,
            neighbor_loss_weight,
        ),
        target=target,
        mask=training_mask,
        time_mean=time_mean,
        time_std=time_std,
        noise_scale=noise_scale,
    )
