"""Conditional flow-matching planner model."""

from __future__ import annotations

import torch
from torch import nn

from diffusion_planner.data.dimensions import TRAJECTORY_DIM, TRAJECTORY_LENGTH

from .decoder import TrajectoryDecoder
from .encoder import SceneEncoder
from .flow_matching import compute_x0_flow_matching_loss, sample
from .normalizer import PlannerNormalizer


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
        time_mean: float = -0.4,
        time_std: float = 1.0,
        time_epsilon: float = 1e-5,
        noise_scale: float = 1.0,
        position_scale: float = 50.0,
        speed_scale: float = 15.0,
        vehicle_shape_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.time_mean = time_mean
        self.time_std = time_std
        self.time_epsilon = time_epsilon
        self.noise_scale = noise_scale
        self.normalizer = PlannerNormalizer(
            position_scale=position_scale,
            speed_scale=speed_scale,
            vehicle_shape_scale=vehicle_shape_scale,
        )
        self.scene_encoder = SceneEncoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            fusion_depth=scene_fusion_depth,
            encoder_depth=element_encoder_depth,
            drop_path_rate=drop_path_rate,
            dropout=dropout,
            embed_dim=embed_dim,
            velocity_threshold=velocity_threshold,
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
        neighbor_pose = input_data["neighbor_agents_past"][
            :, :, -1, :TRAJECTORY_DIM
        ]
        return torch.cat((ego_pose, neighbor_pose), dim=1)

    @staticmethod
    def create_target_trajectory(
        input_data: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Combine labels into `(B, A, T, 4)` ego and neighbor trajectories."""
        ego_future = input_data["ego_agent_future"][..., :TRAJECTORY_DIM].unsqueeze(1)
        return torch.cat((ego_future, input_data["neighbor_agents_future"]), dim=1)

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
        normalized_input = self.normalizer.normalize_input(input_data)
        scene, scene_mask = self.scene_encoder(normalized_input)
        agent_pose = self.create_agent_pose(normalized_input)
        return self.trajectory_decoder(
            x, x_mask, scene, scene_mask, agent_pose, time
        )

    def compute_loss(self, input_data: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute masked conditional flow-matching loss for one batch."""
        target = self.create_target_trajectory(input_data)
        training_mask = (torch.count_nonzero(target, dim=-1) == 0).any(dim=-1)
        target = self.normalizer.normalize_trajectory(target)
        return compute_x0_flow_matching_loss(
            x0_model=lambda state, time: self(
                state, training_mask, input_data, time
            ),
            loss_function=lambda error: error.square(),
            target=target,
            mask=training_mask,
            time_mean=self.time_mean,
            time_std=self.time_std,
            time_epsilon=self.time_epsilon,
            noise_scale=self.noise_scale,
        )

    @torch.no_grad()
    def sample(
        self,
        input_data: dict[str, torch.Tensor],
        num_steps: int = 20,
    ) -> torch.Tensor:
        """Generate physical-unit `(B, A, T, 4)` trajectories with Heun integration.

        `input_data` must already contain training ground-truth or inference-time
        heuristic traffic-light future tensors.
        """
        normalized_input = self.normalizer.normalize_input(input_data)
        scene, scene_mask = self.scene_encoder(normalized_input)
        agent_pose = self.create_agent_pose(normalized_input)
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
        batch, agents = agent_mask.shape
        initial_state = self.noise_scale * torch.randn(
            batch,
            agents,
            TRAJECTORY_LENGTH,
            TRAJECTORY_DIM,
            device=scene.device,
            dtype=scene.dtype,
        )
        normalized_trajectory = sample(
            x0_model=lambda state, time: self.trajectory_decoder(
                state, agent_mask, scene, scene_mask, agent_pose, time
            ),
            initial_state=initial_state,
            num_steps=num_steps,
            epsilon=self.time_epsilon,
            project_state=lambda state: state.masked_fill(
                agent_mask[:, :, None, None], 0.0
            ),
        )
        trajectory = self.normalizer.denormalize_trajectory(normalized_trajectory)
        trajectory = self.normalizer.normalize_yaw_vector(trajectory)
        return trajectory.masked_fill(agent_mask[:, :, None, None], 0.0)
