"""Turn-indicator prediction from frozen scene tokens and indicator history."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from diffusion_planner.data.dimensions import TURN_INDICATOR_HISTORY_LENGTH

from .encoder import OneHotSequenceEncoder


class TurnIndicatorDecoder(nn.Module):
    """Predict the next indicator state from a scene and its recent history."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        history_encoder_depth: int = 2,
        embed_dim: int = 128,
        drop_path_rate: float = 0.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.history_encoder = OneHotSequenceEncoder(
            sequence_len=TURN_INDICATOR_HISTORY_LENGTH,
            num_classes=4,
            hidden_dim=hidden_dim,
            depth=history_encoder_depth,
            drop_path_rate=drop_path_rate,
            embed_dim=embed_dim,
        )
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 3)

    def forward(
        self,
        scene: torch.Tensor,
        scene_mask: torch.Tensor,
        turn_indicators: torch.Tensor,
    ) -> torch.Tensor:
        """Return DISABLE/LEFT/RIGHT logits.

        Args:
            scene: Frozen scene tokens with shape ``(B, N, H)``.
            scene_mask: Invalid scene-token mask with shape ``(B, N)``.
            turn_indicators: Raw report history with shape ``(B, 31)``. Values
                0, 1, 2, and 3 represent missing, disabled, left, and right.

        Returns:
            Next-state logits with shape ``(B, 3)``.
        """
        history = turn_indicators.to(torch.long).clamp(0, 3)
        history_one_hot = F.one_hot(history, num_classes=4).to(scene.dtype)
        history_token = self.history_encoder(history_one_hot).unsqueeze(1)
        attended, _ = self.cross_attention(
            history_token,
            scene,
            scene,
            key_padding_mask=scene_mask,
            need_weights=False,
        )
        token = history_token + attended
        token = token + self.mlp(self.norm(token))
        return self.classifier(self.output_norm(token[:, 0]))
