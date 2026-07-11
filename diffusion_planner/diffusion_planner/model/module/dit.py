import math

import torch
import torch.nn as nn
from timm.layers import Mlp

from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class TimestepEmbedder(nn.Module):
    """Embed scalar continuous diffusion times with sinusoidal features."""

    def __init__(self, hidden_dim: int, frequency_dim: int = 256):
        super().__init__()
        self.frequency_dim = frequency_dim
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10_000):
        half = dim // 2
        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t.float().unsqueeze(-1) * frequencies
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[..., :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor):
        return self.mlp(self.timestep_embedding(t, self.frequency_dim))


class DiTBlock(nn.Module):
    """DiT block with temporal self-attention and scene cross-attention."""

    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=dropout,
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim, bias=True))

    def forward(self, x, cross_c, y, cross_attn_mask):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mca,
            scale_mca,
            gate_mca,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(y).chunk(9, dim=-1)

        modulated_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = (
            x
            + gate_msa[:, None]
            * self.attn(
                modulated_x,
                modulated_x,
                modulated_x,
                need_weights=False,
            )[0]
        )
        x = (
            x
            + gate_mca[:, None]
            * self.cross_attn(
                modulate(self.norm2(x), shift_mca, scale_mca),
                cross_c,
                cross_c,
                key_padding_mask=cross_attn_mask,
                need_weights=False,
            )[0]
        )
        return x + gate_mlp[:, None] * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))


class FinalLayer(nn.Module):
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
        return self.proj(modulate(self.norm_final(x), shift, scale))


class DiT(nn.Module):
    """Ego-only HDP decoder with one action token per future timestep."""

    def __init__(
        self,
        depth,
        output_dim,
        hidden_dim=192,
        heads=6,
        dropout=0.1,
        mlp_ratio=4.0,
        model_type="x_start",
        sde=None,
        future_len=80,
    ):
        super().__init__()
        if model_type not in {"x_start", "noise", "score", "v", "flow_matching"}:
            raise ValueError(f"Unsupported model_type={model_type!r}")
        if output_dim != 4:
            raise ValueError(f"Temporal HDP DiT requires output_dim=4, got {output_dim}")

        self._model_type = model_type
        self._sde = sde if sde is not None else VPSDE_linear()
        self.future_len = future_len
        self.action_in_proj = Mlp(
            in_features=4,
            hidden_features=512,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.action_pos_emb = nn.Parameter(torch.zeros(1, future_len, hidden_dim))
        self.ego_velocity_proj = nn.Linear(2, hidden_dim)
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, heads, dropout, mlp_ratio) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_dim, output_dim)

    @property
    def model_type(self):
        return self._model_type

    def forward(self, x, t, cross_c, ego_current_velocity):
        """
        Args:
            x: ``[B, 1, T, 4]`` or ``[B, T, 4]`` noised ego action trajectory.
            t: ``[B]`` continuous diffusion time, one scalar per scene.
            cross_c: ``[B, N, H]`` scene tokens.
            ego_current_velocity: ``[B, 2]`` normalized current ego vx/vy.
        """
        if x.dim() == 4:
            if x.shape[1] != 1:
                raise ValueError(f"HDP decoder is ego-only, got action shape {tuple(x.shape)}")
            x = x[:, 0]
        if x.dim() != 3 or x.shape[1:] != (self.future_len, 4):
            raise ValueError(f"Expected ego actions [B,{self.future_len},4], got {tuple(x.shape)}")
        B, T, _ = x.shape

        if t.shape != (B,):
            raise ValueError(f"Expected one diffusion time per scene [B], got {tuple(t.shape)}")

        condition = self.t_embedder(t)
        x = (
            self.action_in_proj(x)
            + self.action_pos_emb
            + self.ego_velocity_proj(ego_current_velocity)[:, None, :]
        )
        cross_attn_mask = torch.all(cross_c == 0, dim=-1)
        for block in self.blocks:
            x = block(x, cross_c, condition, cross_attn_mask)
        x = self.final_layer(x, condition)
        if self._model_type == "score":
            x = x / (self._sde.marginal_prob_std(t).unsqueeze(-1) + 1e-6)
        return x[:, None]
