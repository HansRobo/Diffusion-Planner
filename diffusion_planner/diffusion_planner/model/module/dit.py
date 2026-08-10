import math

import torch
import torch.nn as nn
from timm.models.layers import Mlp


def modulate(x, shift, scale):
    x = x * (1 + scale) + shift
    return x


def broadcast_conditioning(chunks):
    """Give adaLN chunks a token axis when the conditioning is per-sample.

    Real-time chunking conditions on a per-agent, per-timestep diffusion time, so the
    chunks already carry the token axis. With ``disable_real_time_chunking`` the
    conditioning is one scalar timestep per sample, shaped (B, hidden), and has to be
    broadcast across tokens.
    """
    return tuple(c.unsqueeze(1) for c in chunks) if chunks[0].dim() == 2 else chunks


def conditioning_chunks(projected, count):
    """Split adaLN output and give it a token axis if the conditioning is per-sample.

    The split axis is spelled out rather than written as -1: a negative axis is traced
    into the exported ONNX verbatim, which would change the graph bytes even where the
    operation is identical.
    """
    return broadcast_conditioning(projected.chunk(count, dim=projected.dim() - 1))


class TimestepEmbedder(nn.Module):
    """Embed a scalar diffusion timestep, sinusoidal features into a 2-layer MLP.

    Used only when ``disable_real_time_chunking`` is set; real-time chunking instead
    feeds a whole (B, P, T) grid of timesteps through an Mlp.
    """

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
        """Sinusoidal embeddings for a 1-D tensor of (possibly fractional) timesteps."""
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
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning for ego and Cross-Attention.
    """

    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp1 = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.norm3 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm4 = nn.LayerNorm(dim)

        self.mlp2 = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
        )

    def forward(self, x, cross_c, y, attn_mask, cross_attn_mask):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = conditioning_chunks(
            self.adaLN_modulation(y), 6
        )

        modulated_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = (
            x
            + gate_msa
            * self.attn(
                modulated_x,
                modulated_x,
                modulated_x,
                key_padding_mask=attn_mask,
                need_weights=False,
            )[0]
        )

        modulated_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp1(modulated_x)

        # key_padding_mask=None is what MultiheadAttention defaults to, so the traced
        # graph is unchanged while the flag is off.
        x = (
            x
            + self.cross_attn(
                self.norm3(x),
                cross_c,
                cross_c,
                key_padding_mask=cross_attn_mask,
                need_weights=False,
            )[0]
        )
        x = x + self.mlp2(self.norm4(x))

        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

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
        B, P, _ = x.shape

        shift, scale = conditioning_chunks(self.adaLN_modulation(y), 2)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.proj(x)
        return x


class DiT(nn.Module):
    def __init__(
        self,
        depth,
        output_dim,
        hidden_dim=192,
        heads=6,
        dropout=0.1,
        mlp_ratio=4.0,
        T=81,
        D=4,
        use_cross_attn_mask=False,
        scalar_time=False,
    ):
        super().__init__()

        self._T = T
        self._D = D
        self._use_cross_attn_mask = use_cross_attn_mask
        self._scalar_time = scalar_time

        self.agent_embedding = nn.Embedding(2, hidden_dim)
        self.preproj = Mlp(
            in_features=T * D,
            hidden_features=512,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        # Real-time chunking conditions on a timestep per agent per horizon step, so the
        # embedder consumes the whole (B, P, T) grid. Without it there is one scalar
        # timestep per sample and the usual sinusoidal embedder applies.
        self.t_embedder = (
            TimestepEmbedder(hidden_dim)
            if scalar_time
            else Mlp(
                in_features=T,
                hidden_features=512,
                out_features=hidden_dim,
                act_layer=nn.GELU,
                drop=0.0,
            )
        )
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, heads, dropout, mlp_ratio) for i in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_dim, output_dim)

    def forward(self, x, t, cross_c, neighbor_current_mask):
        """
        Forward pass of DiT.

        With real-time chunking (the default):
            x: (B, P, T, D)   -> returned in the same layout
            t: (B, P, T, 1)   -> a timestep per agent per horizon step
        With ``scalar_time`` (``disable_real_time_chunking``):
            x: (B, P, T * D)  -> returned in the same layout
            t: (B,)           -> one timestep per sample
        cross_c: (B, N, D)      -> Cross-Attention context
        """
        if self._scalar_time:
            assert x.dim() == 3, f"{x.dim()=}"
            assert t.dim() == 1, f"{t.dim()=}"
            B, P, _ = x.shape
            T = D = None
        else:
            assert x.dim() == 4, f"{x.dim()=}"
            assert t.dim() == 4, f"{t.dim()=}"
            assert x.shape[2] == t.shape[2], f"{x.shape[2]=} {t.shape[2]=}"
            B, P, T, D = x.shape
            x = x.reshape(B, P, T * D)  # (B, P, T*D)
            t = t.reshape(B, P, T)  # (B, P, T)

        x = self.preproj(x)  # (B, P, hidden_dim)
        t = self.t_embedder(t)  # (B, P, hidden_dim) or (B, hidden_dim)

        x_embedding = torch.cat(
            [
                self.agent_embedding.weight[0][None, :],
                self.agent_embedding.weight[1][None, :].expand(P - 1, -1),
            ],
            dim=0,
        )  # (P, hidden_dim)
        x_embedding = x_embedding[None, :, :].expand(B, -1, -1)  # (B, P, hidden_dim)
        x = x + x_embedding

        ego_mask = torch.zeros((B, 1), dtype=torch.bool, device=x.device)
        attn_mask = torch.cat([ego_mask, neighbor_current_mask], dim=1)
        cross_attn_mask = torch.all(cross_c == 0, dim=-1) if self._use_cross_attn_mask else None

        for block in self.blocks:
            x = block(x, cross_c, t, attn_mask, cross_attn_mask)

        x = self.final_layer(x, t)  # (B, P, output_dim)
        if self._scalar_time:
            return x
        return x.reshape(B, P, T, D)
