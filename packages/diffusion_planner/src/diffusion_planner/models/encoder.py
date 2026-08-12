import torch
from timm.models.mlp_mixer import MixerBlock
from torch import nn

CLASS_TYPE_EGO = 0
CLASS_TYPE_NEIGHBOR = 1
CLASS_TYPE_STATIC = 2
CLASS_TYPE_LANE = 3
CLASS_TYPE_ROUTE = 4
CLASS_TYPE_POLYGON = 5
CLASS_TYPE_LINE_STRING = 6
CLASS_TYPE_GOAL_POSE = 7
CLASS_TYPE_EGO_SHAPE = 8
CLASS_TYPE_TURN_INDICATOR = 9
CLASS_TYPE_NUM = 10


class LineEncoder(nn.Module):
    """Encode polyline elements (e.g. lanes, route segments) into per-element features.

    Each polyline is a fixed-length sequence of 2D points with a type one-hot
    appended to every point. The point coordinates are embedded by a linear stem,
    mixed across tokens by MLP-Mixer blocks, and average-pooled into a single
    feature per polyline. The type one-hot is injected additively via a separate
    embedding instead of being fed through the stem/Mixer.
    """

    def __init__(
        self,
        line_len: int,
        class_type: int,
        drop_path_rate: float,
        hidden_dim: int,
        depth: int,
        num_types: int,
        embed_dim: int = 128,
    ) -> None:
        """Encode polyline elements for initialization.

        Args:
            line_len: Number of points per polyline (token count for the Mixer).
            class_type: Class type id of this element (one of the CLASS_TYPE_* constants).
            drop_path_rate: Stochastic depth rate for the Mixer blocks.
            hidden_dim: Output feature dimension of the head.
            depth: Number of MixerBlock layers.
            num_types: Size of the per-point type one-hot in the input.
            embed_dim: Internal embedding dimension.
        """
        super().__init__()
        self._class_type = class_type

        self.type_emb = nn.Linear(num_types, embed_dim)

        self.stem = nn.Linear(2, embed_dim)  # x, y
        self.blocks = nn.ModuleList(
            [
                MixerBlock(embed_dim, line_len, drop_path=drop_path_rate)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of polylines.

        Args:
            x: Tensor of shape (B, P, V, D) where B is the batch size, P is the
                number of polylines, V is the number of points per polyline
                (== line_len), and D = 2 + num_types with each point laid out as
                (x, y, type_one_hot...).

        Returns:
            A tuple of:
                - feat: Tensor of shape (B, P, hidden_dim). Per-polyline features;
                  rows for invalid (all-zero) polylines are zeroed out.
                - mask_p: Bool tensor of shape (B, P). True where the polyline is
                  invalid (all elements zero).
        """
        b, p, v, _ = x.shape

        # Mask before splitting: an element is empty only if every coord/type is 0.
        mask_p = torch.sum(torch.ne(x, 0), dim=(-2, -1)) == 0
        valid_indices = ~mask_p.view(-1)

        # Split coordinates and type one-hot. The type is constant within an element,
        # so a single representative point is enough.
        coords = x[..., :2]  # (b, p, v, 2)
        type_one_hot = x[:, :, 0, 2:]  # (b, p, num_types)

        feat = coords.view(b * p, v, -1)  # (b * p, v, 2): x, y

        feat = self.stem(feat)  # (b * p, v, embed_dim)
        for block in self.blocks:
            feat = block(feat)
        feat = self.norm(feat)

        # global average pooling over tokens
        feat = torch.mean(feat, dim=1)

        # Inject type information via embedding instead of through the stem/Mixer.
        feat = feat + self.type_emb(type_one_hot.view(b * p, -1))

        feat = self.head(feat)

        # Apply mask to zero out invalid positions
        feat = feat * valid_indices.float().unsqueeze(-1)

        return feat.view(b, p, -1), mask_p.reshape(b, -1)


class LaneEncoder(nn.Module):
    """Encode lane polylines with attributes (speed limit, traffic lights, line types)."""

    def __init__(
        self,
        lane_len,
        class_type,
        drop_path_rate,
        hidden_dim,
        depth,
        embed_dim=128,
    ):
        super().__init__()

        self._lane_len = lane_len
        self._class_type = class_type

        self.speed_limit_emb = nn.Linear(1, embed_dim)
        self.unknown_speed_emb = nn.Embedding(1, embed_dim)
        self.attribute_emb = nn.Linear(
            5 + 2 * 10, embed_dim
        )  # traffic_light and line type

        self.stem = nn.Linear(8, embed_dim)
        self.blocks = nn.ModuleList(
            [
                MixerBlock(embed_dim, lane_len, drop_path=drop_path_rate)
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, hidden_dim)

    def forward(self, x, speed_limit, has_speed_limit):
        """Encode lane polylines.

        Args:
            x: Tensor of shape (B, P, V, D) with coords (x, y, x_left-x, y_left-y,
                x_right-x, y_right-y) plus traffic (5) and line_type (2*10).
            speed_limit: Tensor of shape (B, P, 1).
            has_speed_limit: Bool tensor of shape (B, P, 1).
        """
        attribute = x[:, :, 0, 8:]
        x = x[..., :8]

        b, p, v, _ = x.shape
        mask_p = torch.sum(torch.ne(x[..., :8], 0), dim=(-2, -1)) == 0
        valid_indices = ~mask_p.view(-1)

        x = x.view(b * p, v, -1)

        # Use torch.where instead of indexing to maintain fixed size
        x = torch.where(valid_indices.view(-1, 1, 1), x, torch.zeros_like(x))

        x = self.stem(x)  # (b * p, v, embed_dim)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # global average pooling over tokens
        x = torch.mean(x, dim=1)

        # Reshape speed_limit and traffic to match flattened dimensions
        speed_limit = speed_limit.view(b * p, 1)
        has_speed_limit = has_speed_limit.view(b * p, 1)
        attribute = attribute.view(b * p, -1)

        # Create embeddings for all positions
        speed_limit_emb = self.speed_limit_emb(speed_limit)
        unknown_speed_emb = self.unknown_speed_emb(
            torch.zeros(b * p, dtype=torch.long, device=x.device)
        )
        speed_limit_embedding = torch.where(
            has_speed_limit, speed_limit_emb, unknown_speed_emb
        )

        # Process traffic lights for all positions
        traffic_light_embedding = self.attribute_emb(attribute)

        x = x + speed_limit_embedding + traffic_light_embedding
        x = self.head(x)

        # Apply mask to zero out invalid positions
        x = x * valid_indices.float().unsqueeze(-1)

        return x.view(b, p, -1), mask_p.reshape(b, -1)
