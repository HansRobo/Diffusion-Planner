import torch
import torch.nn as nn
from timm.models.layers import Mlp


def modulate(x, shift, scale):
    x = x * (1 + scale) + shift
    return x


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning and Cross-Attention.

    The block holds no attention between agents because the network denoises the ego
    trajectory only: neighbors reach the ego prediction through the scene memory, which
    carries their observed histories, and never through a generated future.
    """

    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp1 = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim, bias=True))
        self.norm3 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm4 = nn.LayerNorm(dim)

        self.mlp2 = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
        )

    def forward(self, x, cross_c, y):
        """
        Args:
            x: (B, 1, dim) ego trajectory token.
            cross_c: (B, N, dim) scene memory.
            y: (B, 1, dim) diffusion-time embedding.
        """
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=2)

        modulated_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp1(modulated_x)

        x = x + self.cross_attn(self.norm3(x), cross_c, cross_c, need_weights=False)[0]
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
        shift, scale = self.adaLN_modulation(y).chunk(2, dim=2)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.proj(x)
        return x


class DiT(nn.Module):
    """Denoising network for the ego trajectory.

    The network has no agent dimension: it embeds one trajectory - the ego one - into a
    single token and conditions it on the scene memory through cross-attention. Neighbor
    futures are neither an input nor an output.
    """

    def __init__(
        self,
        depth,
        output_dim,
        hidden_dim=192,
        heads=6,
        dropout=0.1,
        mlp_ratio=4.0,
    ):
        super().__init__()

        T = 81
        D = 4
        self.preproj = Mlp(
            in_features=T * D,
            hidden_features=512,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.t_embedder = Mlp(
            in_features=T,
            hidden_features=512,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, heads, dropout, mlp_ratio) for i in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_dim, output_dim)

    def forward(self, x, t, cross_c):
        """
        Forward pass of DiT.

        Args:
            x: (B, T, D) noised ego trajectory.
            t: (B, T, 1) diffusion time per trajectory step.
            cross_c: (B, N, D) Cross-Attention context.

        Returns:
            (B, T, D) denoised ego trajectory.
        """
        assert x.dim() == 3, f"{x.dim()=}"
        assert t.dim() == 3, f"{t.dim()=}"
        assert x.shape[1] == t.shape[1], f"{x.shape[1]=} {t.shape[1]=}"
        B, T, D = x.shape

        x = x.reshape(B, 1, T * D)  # (B, 1, T*D)
        t = t.reshape(B, 1, T)  # (B, 1, T)

        x = self.preproj(x)  # (B, 1, hidden_dim)
        t = self.t_embedder(t)  # (B, 1, hidden_dim)

        for block in self.blocks:
            x = block(x, cross_c, t)

        x = self.final_layer(x, t)  # (B, 1, output_dim)
        x = x.reshape(B, T, D)
        return x
