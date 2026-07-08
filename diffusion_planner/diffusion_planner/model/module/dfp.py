import math

import torch
import torch.nn as nn
from timm.models.layers import Mlp

from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


_VP_SDE = VPSDE_linear()


def vp_alpha_sigma(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    alpha, sigma = _VP_SDE.marginal_prob(torch.ones_like(t), t)
    return alpha, sigma


def normalize_ego_trajectory(state_normalizer, x: torch.Tensor) -> torch.Tensor:
    mean = state_normalizer.mean.to(x.device, dtype=x.dtype)[0]
    std = state_normalizer.std.to(x.device, dtype=x.dtype)[0]
    return (x - mean) / std


def inverse_normalize_ego_trajectory(state_normalizer, x: torch.Tensor) -> torch.Tensor:
    mean = state_normalizer.mean.to(x.device, dtype=x.dtype)[0]
    std = state_normalizer.std.to(x.device, dtype=x.dtype)[0]
    return x * std + mean


def _agent_stats_view(state_normalizer, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [1, P, ..., 4] stats for tensors shaped [B, P, ..., 4]."""
    if x.dim() < 3:
        raise ValueError(f"agent trajectory tensor must be at least 3D, got {tuple(x.shape)}")
    P = x.shape[1]
    mean = state_normalizer.mean.to(x.device, dtype=x.dtype)[:P, 0, :]
    std = state_normalizer.std.to(x.device, dtype=x.dtype)[:P, 0, :]
    view_shape = [1, P] + [1] * (x.dim() - 3) + [4]
    return mean.reshape(view_shape), std.reshape(view_shape)


def normalize_trajectory_by_agent(state_normalizer, x: torch.Tensor) -> torch.Tensor:
    mean, std = _agent_stats_view(state_normalizer, x)
    return (x - mean) / std


def inverse_normalize_trajectory_by_agent(state_normalizer, x: torch.Tensor) -> torch.Tensor:
    mean, std = _agent_stats_view(state_normalizer, x)
    return x * std + mean


def normalize_neighbor_trajectory(state_normalizer, x: torch.Tensor) -> torch.Tensor:
    mean = state_normalizer.mean.to(x.device, dtype=x.dtype)[1, 0]
    std = state_normalizer.std.to(x.device, dtype=x.dtype)[1, 0]
    return (x - mean) / std


def inverse_normalize_neighbor_trajectory(state_normalizer, x: torch.Tensor) -> torch.Tensor:
    mean = state_normalizer.mean.to(x.device, dtype=x.dtype)[1, 0]
    std = state_normalizer.std.to(x.device, dtype=x.dtype)[1, 0]
    return x * std + mean


def _modulate_tokenwise(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale) + shift


class DFPDiTBlock(nn.Module):
    """Chunk-token DiT block with per-token timestep conditioning."""

    def __init__(self, dim=256, heads=8, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp1 = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.norm3 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm4 = nn.LayerNorm(dim)
        self.mlp2 = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x, cross_c, y, attn_mask, cross_attn_mask):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(y).chunk(6, dim=-1)
        )

        modulated_x = _modulate_tokenwise(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.attn(
            modulated_x,
            modulated_x,
            modulated_x,
            key_padding_mask=attn_mask,
            need_weights=False,
        )[0]

        modulated_x = _modulate_tokenwise(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp1(modulated_x)

        x = x + self.cross_attn(
            self.norm3(x),
            cross_c,
            cross_c,
            key_padding_mask=cross_attn_mask,
            need_weights=False,
        )[0]
        x = x + self.mlp2(self.norm4(x))
        return x


class DFPFinalLayer(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4, bias=True),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, output_size, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, y):
        shift, scale = self.adaLN_modulation(y).chunk(2, dim=-1)
        x = _modulate_tokenwise(self.norm_final(x), shift, scale)
        return self.proj(x)


class DFPDiT(nn.Module):
    """Diffusion Forcing Planner chunk-token decoder.

    This is an additive branch on top of the original planner encoder. Tokens are trajectory
    chunks: one ego-history chunk, one repeated-current chunk, then future chunks.
    """

    def __init__(
        self,
        num_chunks: int,
        chunk_len: int,
        depth: int,
        hidden_dim: int = 256,
        heads: int = 8,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.num_chunks = num_chunks
        self.chunk_len = chunk_len
        self.output_dim = chunk_len * 4
        self.preproj = Mlp(
            in_features=self.output_dim,
            hidden_features=512,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.chunk_pos_embed = nn.Parameter(torch.zeros(1, num_chunks, hidden_dim))
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.blocks = nn.ModuleList(
            [DFPDiTBlock(hidden_dim, heads, dropout, mlp_ratio) for _ in range(depth)]
        )
        self.final_layer = DFPFinalLayer(hidden_dim, self.output_dim)
        nn.init.normal_(self.chunk_pos_embed, std=0.02)
        nn.init.constant_(self.final_layer.proj[-1].weight, 0.0)
        nn.init.constant_(self.final_layer.proj[-1].bias, 0.0)

    def forward(self, chunks, t, cross_c):
        """
        Args:
            chunks: [B, N, L, 4]
            t: [B, N]
            cross_c: [B, M, D]
        Returns:
            x0 prediction: [B, N, L, 4]
        """
        B, N, L, D = chunks.shape
        assert N == self.num_chunks, f"{N=} expected {self.num_chunks}"
        assert L == self.chunk_len, f"{L=} expected {self.chunk_len}"
        assert D == 4

        x = chunks.reshape(B, N, L * D)
        x = self.preproj(x) + self.chunk_pos_embed[:, :N]
        y = self.t_embedder(t.reshape(B * N)).reshape(B, N, -1)

        attn_mask = torch.zeros((B, N), dtype=torch.bool, device=chunks.device)
        cross_attn_mask = torch.all(cross_c == 0, dim=-1)
        all_masked = torch.all(cross_attn_mask, dim=1)
        if torch.any(all_masked):
            cross_attn_mask = cross_attn_mask.clone()
            cross_attn_mask[all_masked, 0] = False

        for block in self.blocks:
            x = block(x, cross_c, y, attn_mask, cross_attn_mask)

        x = self.final_layer(x, y)
        return x.reshape(B, N, L, D)
