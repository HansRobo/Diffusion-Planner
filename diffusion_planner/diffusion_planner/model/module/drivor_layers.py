"""Transformer primitives of the DrivoR predictor head.

Ported from ``DrivoR/navsim/agents/drivoR/transformer_decoder.py`` and
``DrivoR/navsim/agents/drivoR/layers/utils/mlp.py``.  The block structure,
residual order and LayerScale/DropPath placement are kept value-for-value with
the reference implementation so a Diffusion-Planner run and a DrivoR run share
the same head arithmetic; only the scene-token producer differs.
"""

from typing import Optional, Type

import torch
import torch.nn as nn
from timm.layers import DropPath
from timm.models.layers import Mlp

try:  # timm >= 0.9 exposes LayerScale; older releases do not.
    from timm.layers import LayerScale
except ImportError:  # pragma: no cover - depends on the installed timm

    class LayerScale(nn.Module):
        def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False):
            super().__init__()
            self.inplace = inplace
            self.gamma = nn.Parameter(init_values * torch.ones(dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.mul_(self.gamma) if self.inplace else x * self.gamma


class MLP(nn.Module):
    """DrivoR's trajectory/scorer MLP (Linear-LN-ReLU x2 then Linear)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.mlp(inputs)


class Attention(nn.Module):
    """Self- or cross-attention wrapper around ``nn.MultiheadAttention``."""

    def __init__(self, dim: int, num_heads: int = 8, proj_drop: float = 0.0) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.proj_drop = nn.Dropout(proj_drop)
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=0.0,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None,
            vdim=None,
            batch_first=True,
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if kv is None:
            x = self.attn(query=q, key=q, value=q, need_weights=False)[0]
        else:
            x = self.attn(
                query=q,
                key=kv,
                value=kv,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )[0]
        return self.proj_drop(x)


class Block(nn.Module):
    """Self-attention -> cross-attention -> FFN, pre-norm with residuals."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        scale_mlp_norm: bool = False,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        mlp_layer: Type[nn.Module] = Mlp,
    ):
        super().__init__()

        self.self_attn_norm = norm_layer(dim)
        self.self_attn = Attention(dim, num_heads=num_heads, proj_drop=proj_drop)
        self.self_attn_ls = (
            LayerScale(dim, init_values=init_values) if (init_values > 0) else nn.Identity()
        )
        self.self_attn_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.cross_attn_norm_kv = norm_layer(dim)
        self.cross_attn_norm_q = norm_layer(dim)
        self.cross_attn = Attention(dim, num_heads=num_heads, proj_drop=proj_drop)
        self.cross_attn_ls = (
            LayerScale(dim, init_values=init_values) if (init_values > 0) else nn.Identity()
        )
        self.cross_attn_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.mlp_norm = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            norm_layer=norm_layer if scale_mlp_norm else None,
            bias=proj_bias,
            drop=proj_drop,
        )
        self.mlp_ls = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.mlp_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        x_cross: torch.Tensor,
        x_cross_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.self_attn_drop_path(self.self_attn_ls(self.self_attn(self.self_attn_norm(x))))
        x = x + self.cross_attn_drop_path(
            self.cross_attn_ls(
                self.cross_attn(
                    self.cross_attn_norm_q(x),
                    self.cross_attn_norm_kv(x_cross),
                    key_padding_mask=x_cross_padding_mask,
                )
            )
        )
        x = x + self.mlp_drop_path(self.mlp_ls(self.mlp(self.mlp_norm(x))))
        return x


class TransformerDecoder(nn.Module):
    """Iterative proposal refinement; returns every intermediate stage."""

    def __init__(self, num_layers: int, d_model: int, num_heads: int, ls_values: float,
                 proj_drop: float, drop_path: float, return_intermediate: bool = True):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                Block(
                    dim=d_model,
                    num_heads=num_heads,
                    init_values=ls_values,
                    proj_drop=proj_drop,
                    drop_path=drop_path,
                )
                for _ in range(num_layers)
            ]
        )
        self.return_intermediate = return_intermediate

    def forward(
        self,
        x: torch.Tensor,
        x_cross: torch.Tensor,
        x_cross_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        intermediate = []
        for layer in self.layers:
            x = layer(x, x_cross, x_cross_padding_mask)
            if self.return_intermediate:
                intermediate.append(x)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return x


__all__ = [
    "MLP",
    "Attention",
    "Block",
    "LayerScale",
    "TransformerDecoder",
]
