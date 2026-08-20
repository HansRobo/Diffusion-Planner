"""Conditional flow-matching planner model."""

from __future__ import annotations

import torch
from torch import nn

from diffusion_planner.data.dimensions import TRAJECTORY_DIM

from .decoder import TrajectoryDecoder
from .encoder import SceneEncoder
from .flow_matching import sample


class DiffusionPlanner(nn.Module):
    """Predict joint ego and neighbor trajectories with conditional flow matching."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        scene_fusion_depth: int = 4,
        element_encoder_depth: int = 2,
        decoder_depth: int = 6,
        trajectory_encoder_depth: int = 2,
        feedforward_dim: int = 1024,
        embed_dim: int = 128,
        drop_path_rate: float = 0.0,
        dropout: float = 0.0,
        velocity_threshold: float = 0.1,
        goal_max_distance: float = 2.0,
    ) -> None:
        super().__init__()
        self.scene_encoder = SceneEncoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            fusion_depth=scene_fusion_depth,
            encoder_depth=element_encoder_depth,
            drop_path_rate=drop_path_rate,
            dropout=dropout,
            embed_dim=embed_dim,
            velocity_threshold=velocity_threshold,
            goal_max_distance=goal_max_distance,
        )
        self.trajectory_decoder = TrajectoryDecoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            depth=decoder_depth,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            trajectory_encoder_depth=trajectory_encoder_depth,
        )

    @staticmethod
    def create_agent_pose(
        input_data: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Create current `[x, y, cos_yaw, sin_yaw]` poses with shape `(B, A, 4)`."""
        ego_pose = input_data["ego_agent_past"][:, -1, :TRAJECTORY_DIM].unsqueeze(1)
        neighbor_pose = input_data["neighbor_agents_past"][:, :, -1, :TRAJECTORY_DIM]
        return torch.cat((ego_pose, neighbor_pose), dim=1)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        input_data: dict[str, torch.Tensor],
        time: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the clean trajectory at a flow state.

        Args:
            x: Normalized flow-state trajectories with shape `(B, A, T, 4)`.
            x_mask: Invalid-agent mask with shape `(B, A)`.
            input_data: Batched planner input tensors, including traffic-light future.
            time: Flow times with shape `(B,)` or `(B, 1)`.

        Returns:
            Predicted normalized clean trajectories with shape `(B, A, T, 4)`.
        """
        scene, scene_mask = self.scene_encoder(input_data)
        agent_pose = self.create_agent_pose(input_data)
        return self.trajectory_decoder(x, x_mask, scene, scene_mask, agent_pose, time)

    @torch.no_grad()
    def sample(
        self,
        input_data: dict[str, torch.Tensor],
        initial_noise: torch.Tensor,
        num_steps: int = 20,
        time_epsilon: float = 1e-5,
    ) -> torch.Tensor:
        """Generate normalized `(B, A, T, 4)` trajectories with Heun integration.

        `input_data` must already contain training ground-truth or inference-time
        heuristic traffic-light future tensors. `initial_noise` has shape
        `(B, A, T, 4)` and completely determines the initial flow state.
        """
        scene, scene_mask = self.scene_encoder(input_data)
        agent_pose = self.create_agent_pose(input_data)
        neighbor_mask = (
            torch.count_nonzero(input_data["neighbor_agents_past"], dim=(-2, -1)) == 0
        )
        ego_mask = torch.zeros(
            neighbor_mask.shape[0],
            1,
            dtype=torch.bool,
            device=neighbor_mask.device,
        )
        agent_mask = torch.cat((ego_mask, neighbor_mask), dim=1)
        trajectory = sample(
            x0_model=lambda state, time: self.trajectory_decoder(
                state, agent_mask, scene, scene_mask, agent_pose, time
            ),
            initial_state=initial_noise,
            num_steps=num_steps,
            epsilon=time_epsilon,
            project_state=lambda state: state.masked_fill(
                agent_mask[:, :, None, None], 0.0
            ),
        )
        yaw = trajectory[..., 2:4]
        yaw = yaw / torch.linalg.vector_norm(yaw, dim=-1, keepdim=True).clamp_min(1e-6)
        trajectory = torch.cat((trajectory[..., :2], yaw), dim=-1)
        return trajectory.masked_fill(agent_mask[:, :, None, None], 0.0)
