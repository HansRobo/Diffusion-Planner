"""Flow-matching trajectory decoder with scene cross-attention."""

from __future__ import annotations

import math

import torch
from timm.layers.mlp import Mlp
from timm.models.mlp_mixer import MixerBlock
from torch import nn

from diffusion_planner.data.dimensions import TRAJECTORY_DIM, TRAJECTORY_LENGTH


class SinusoidalTimeEmbedding(nn.Module):
    """Embed scalar flow times with sinusoidal features and an MLP."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.frequency_dim = hidden_dim
        self.mlp = Mlp(
            in_features=hidden_dim,
            hidden_features=hidden_dim * 4,
            out_features=hidden_dim,
            act_layer=nn.SiLU,
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """Encode flow times `(B,)` or `(B, 1)` into `(B, H)`."""
        time = time.reshape(-1)
        half_dim = self.frequency_dim // 2
        frequencies = torch.exp(
            -math.log(10_000)
            * torch.arange(half_dim, device=time.device, dtype=time.dtype)
            / max(half_dim - 1, 1)
        )
        angles = time[:, None] * frequencies[None]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if embedding.shape[-1] < self.frequency_dim:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return self.mlp(embedding)


class AdaptiveLayerNorm(nn.Module):
    """Apply LayerNorm modulated by a flow-time embedding."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.modulation = nn.Linear(hidden_dim, hidden_dim * 2)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, values: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Modulate values `(..., H)` using time features `(B, H)`."""
        scale, shift = self.modulation(time).chunk(2, dim=-1)
        while scale.ndim < values.ndim:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        return self.norm(values) * (1 + scale) + shift


class TrajectoryEncoder(nn.Module):
    """Encode one complete trajectory into one token per agent."""

    def __init__(
        self,
        hidden_dim: int,
        depth: int,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.trajectory_len = TRAJECTORY_LENGTH
        self.input_projection = nn.Linear(TRAJECTORY_DIM, hidden_dim)
        self.blocks = nn.ModuleList(
            MixerBlock(hidden_dim, TRAJECTORY_LENGTH, drop_path=drop_path_rate)
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, trajectories: torch.Tensor) -> torch.Tensor:
        """Encode trajectories `(B, A, T, 4)` into agent tokens `(B, A, H)`.

        Args:
            trajectories: Ego and neighbor trajectories with shape `(B, A, T, 4)`.

        Returns:
            One token per agent with shape `(B, A, H)`.
        """
        batch, agents = trajectories.shape[:2]
        features = self.input_projection(trajectories)
        features = features.reshape(batch * agents, self.trajectory_len, -1)
        for block in self.blocks:
            features = block(features)
        features = self.norm(features).mean(dim=1)
        return features.reshape(batch, agents, -1)


class TrajectoryDecoderBlock(nn.Module):
    """Fuse independent agent trajectory tokens with scene memory."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.cross_norm = AdaptiveLayerNorm(hidden_dim)
        self.feedforward_norm = AdaptiveLayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.scene_norm = nn.LayerNorm(hidden_dim)
        self.feedforward = Mlp(
            in_features=hidden_dim,
            hidden_features=feedforward_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        scene: torch.Tensor,
        time: torch.Tensor,
        scene_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decode masked agent tokens using masked scene memory and flow time."""
        query = self.cross_norm(x, time)
        memory = self.scene_norm(scene)
        cross = self.cross_attention(
            query,
            memory,
            memory,
            key_padding_mask=scene_mask,
            need_weights=False,
        )[0]
        x = x + self.dropout(cross)
        x = x + self.dropout(self.feedforward(self.feedforward_norm(x, time)))
        return x.masked_fill(x_mask.unsqueeze(-1), 0.0)


class TrajectoryDecoder(nn.Module):
    """Decode one token per agent into clean trajectory predictions."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        depth: int = 6,
        feedforward_dim: int = 1024,
        dropout: float = 0.0,
        trajectory_encoder_depth: int = 2,
    ) -> None:
        super().__init__()
        self.trajectory_len = TRAJECTORY_LENGTH
        self.trajectory_encoder = TrajectoryEncoder(
            hidden_dim,
            trajectory_encoder_depth,
            dropout,
        )
        self.ego_embedding = nn.Parameter(torch.empty(hidden_dim))
        self.neighbor_embedding = nn.Parameter(torch.empty(hidden_dim))
        self.agent_pose_embedding = nn.Linear(TRAJECTORY_DIM, hidden_dim)
        self.time_embedding = SinusoidalTimeEmbedding(hidden_dim)
        self.blocks = nn.ModuleList(
            TrajectoryDecoderBlock(hidden_dim, num_heads, feedforward_dim, dropout)
            for _ in range(depth)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(
            hidden_dim, TRAJECTORY_LENGTH * TRAJECTORY_DIM
        )
        nn.init.normal_(self.ego_embedding, std=0.02)
        nn.init.normal_(self.neighbor_embedding, std=0.02)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        scene: torch.Tensor,
        scene_mask: torch.Tensor,
        agent_pos: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        """Predict clean trajectories from a noisy flow state.

        Args:
            x: Ego and neighbor trajectories with shape `(B, A, T, 4)`, where
                each state is `[x, y, cos_yaw, sin_yaw]`, agent zero is ego,
                and `T` equals `TRAJECTORY_LENGTH`. Each complete trajectory
                is encoded into one agent token before attention.
            x_mask: Invalid-agent mask with shape `(B, A)`.
            scene: Scene tokens with shape `(B, S, H)`.
            scene_mask: Invalid-scene-token mask with shape `(B, S)`.
            agent_pos: Normalized current poses with shape `(B, A, 4)`, where
                each pose is `[x, y, cos_yaw, sin_yaw]` and is aligned with `x`.
            time: Flow times with shape `(B,)` or `(B, 1)`.

        Returns:
            Predicted clean trajectories with shape `(B, A, T, 4)`.
        """
        batch, agents, _, _ = x.shape
        features = self.trajectory_encoder(x)

        agent_embedding = torch.cat(
            (
                self.ego_embedding.unsqueeze(0),
                self.neighbor_embedding.expand(agents - 1, -1),
            ),
            dim=0,
        )
        features = (
            features + agent_embedding[None] + self.agent_pose_embedding(agent_pos)
        )
        features = features.masked_fill(x_mask.unsqueeze(-1), 0.0)

        time_features = self.time_embedding(time.to(dtype=features.dtype))
        for block in self.blocks:
            features = block(features, x_mask, scene, time_features, scene_mask)
        output = self.output_projection(self.output_norm(features))
        output = output.reshape(batch, agents, self.trajectory_len, TRAJECTORY_DIM)
        return output.masked_fill(x_mask[:, :, None, None], 0.0)
