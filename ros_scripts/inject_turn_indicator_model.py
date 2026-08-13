"""Overwrite a Diffusion Planner's turn-indicator output with a separately trained head.

Loads a full checkpoint (``best_model.pth``) plus a standalone turn-indicator checkpoint
(``best_model_turn_indicator.pth``, as produced by
``ros_scripts/extract_turn_indicator_model.py``), swaps the planner's turn-indicator
predictor for the standalone one, and exports the usual one-step
``diffusion_planner.onnx``.

The ONNX schema is unchanged: the graph still returns ``prediction`` and
``turn_indicator_logit``. The ``turn_indicator_logit`` value is simply *overwritten* with
the injected head's output, so the ROS node / TensorRT engine needs no modification.

The turn-indicator network definition is copied verbatim into this script (rather than
imported from ``diffusion_planner.model.module.turn_indicator``) so the injected head stays
pinned to the architecture the standalone checkpoint was trained with, independent of later
changes in the main tree.

Usage:
    uv run python ros_scripts/inject_turn_indicator_model.py \
        best_model.pth best_model_turn_indicator.pth -o diffusion_planner.onnx
"""

import argparse
from pathlib import Path

import torch
from diffusion_planner.dimensions import *
from diffusion_planner.utils.config import Config
from diffusion_planner.utils.onnx_export import (
    FULL_INPUT_NAMES,
    FULL_OUTPUT_NAMES,
    FullONNXWrapper,
    build_dummy_inputs,
    export_onnx,
    load_model,
    onnx_export_backends,
)
from timm.layers import DropPath
from timm.models.layers import Mlp
from torch import nn
from torch.nn import functional as F

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.mha.set_fastpath_enabled(False)


# ---------------------------------------------------------------------------
# Building blocks (copied from diffusion_planner/model/module/{mixer,encoder}.py)
# ---------------------------------------------------------------------------


class MixerBlock(nn.Module):
    def __init__(self, tokens_mlp_dim, channels_mlp_dim, drop_path_rate):
        super().__init__()

        self.norm1 = nn.LayerNorm(channels_mlp_dim)
        self.channels_mlp = Mlp(
            in_features=channels_mlp_dim,
            hidden_features=channels_mlp_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )
        self.norm2 = nn.LayerNorm(channels_mlp_dim)
        self.tokens_mlp = Mlp(
            in_features=tokens_mlp_dim,
            hidden_features=tokens_mlp_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )

    def forward(self, x):
        y = self.norm1(x)
        y = y.permute(0, 2, 1)
        y = self.tokens_mlp(y)
        y = y.permute(0, 2, 1)
        x = x + y
        y = self.norm2(x)
        return x + self.channels_mlp(y)


class LaneEncoder(nn.Module):
    """Copied from the main encoder. Only its features / mask are used here; the
    positional output is built in this network's own vocabulary by ``make_lane_pos``.
    """

    def __init__(
        self,
        lane_len,
        class_type,
        drop_path_rate,
        hidden_dim,
        depth,
        tokens_mlp_dim=64,
        channels_mlp_dim=128,
    ):
        super().__init__()

        self._lane_len = lane_len
        self._class_type = class_type

        self.speed_limit_emb = nn.Linear(1, channels_mlp_dim)
        self.unknown_speed_emb = nn.Embedding(1, channels_mlp_dim)
        self.attribute_emb = nn.Linear(5 + 2 * 10, channels_mlp_dim)  # traffic_light and line type

        self.channel_pre_project = Mlp(
            in_features=8,
            hidden_features=channels_mlp_dim,
            out_features=channels_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.token_pre_project = Mlp(
            in_features=lane_len,
            hidden_features=tokens_mlp_dim,
            out_features=tokens_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

        self.blocks = nn.ModuleList(
            [MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate) for i in range(depth)]
        )

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(
            in_features=channels_mlp_dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )

    def forward(self, x, speed_limit, has_speed_limit):
        """
        x: B, P, V, D (x, y, x'-x, y'-y, x_left-x, y_left-y, x_right-x, y_right-y, traffic(5) + line_type(2 * 10))
        speed_limit: B, P, 1
        has_speed_limit: B, P, 1
        """
        attribute = x[:, :, 0, 8:]
        x = x[..., :8]

        pos = x[:, :, int(self._lane_len / 2), :4].clone()  # x, y, x'-x, y'-y
        heading = torch.atan2(pos[..., 3], pos[..., 2])
        pos = torch.stack(
            [pos[..., 0], pos[..., 1], torch.cos(heading), torch.sin(heading)], dim=-1
        )
        pos = add_class_type(pos, self._class_type)

        B, P, V, _ = x.shape
        mask_p = torch.sum(torch.ne(x[..., :8], 0), dim=(-2, -1)) == 0
        valid_indices = ~mask_p.view(-1)

        x = x.view(B * P, V, -1)

        # Use torch.where instead of indexing to maintain fixed size
        x = torch.where(valid_indices.view(-1, 1, 1), x, torch.zeros_like(x))

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)

        x = torch.mean(x, dim=1)

        # Reshape speed_limit and traffic to match flattened dimensions
        speed_limit = speed_limit.view(B * P, 1)
        has_speed_limit = has_speed_limit.view(B * P, 1)
        attribute = attribute.view(B * P, -1)

        # Create embeddings for all positions
        speed_limit_emb = self.speed_limit_emb(speed_limit)
        unknown_speed_emb = self.unknown_speed_emb(
            torch.zeros(B * P, dtype=torch.long, device=x.device)
        )
        speed_limit_embedding = torch.where(has_speed_limit, speed_limit_emb, unknown_speed_emb)

        # Process traffic lights for all positions
        traffic_light_embedding = self.attribute_emb(attribute)

        x = x + speed_limit_embedding + traffic_light_embedding
        x = self.emb_project(self.norm(x))

        # Apply mask to zero out invalid positions
        x = x * valid_indices.float().unsqueeze(-1)

        return x.view(B, P, -1), mask_p.reshape(B, -1), pos.view(B, P, -1)


# ---------------------------------------------------------------------------
# Turn-indicator network (copied from diffusion_planner/model/module/turn_indicator.py)
# ---------------------------------------------------------------------------

# This network keeps its own, deliberately small, token-type vocabulary so that
# it stays independent from the main diffusion Encoder (which owns a different,
# larger CLASS_TYPE_* set). Keeping them separate means changes here never shift
# the main encoder's positional-embedding dimensions or invalidate its
# checkpoints.
CLASS_TYPE_TURN_INDICATOR_HISTORY = 0
CLASS_TYPE_LANE = 1
CLASS_TYPE_ROUTE = 2
CLASS_TYPE_NUM = 3

# Raw turn-indicator states stored in ``turn_indicators`` (none / disable /
# enable-left / enable-right). Note this is distinct from
# ``TURN_INDICATOR_OUTPUT_DIM``, which additionally carries the "keep" class the
# network predicts.
TURN_INDICATOR_HISTORY_NUM_CLASSES = 4


def add_class_type(x, class_type):
    """
    Add class type to the input tensor.
    Args:
        x: Tensor of shape (B, T, D=4) where D=4 represents (x, y, cos, sin)
        class_type: Class type to add (int)
    Returns:
        x: Tensor with class type added at the end
    """
    B, T, D = x.shape
    assert D == 4, "Input tensor must have 4 features (x, y, cos, sin)"
    class_type_tensor = F.one_hot(
        torch.full((B, T), class_type, device=x.device, dtype=torch.long),
        num_classes=CLASS_TYPE_NUM,
    ).to(dtype=x.dtype)
    return torch.cat([x, class_type_tensor], dim=-1)


def make_baselink_pose(batch_size: int, device: torch.device):
    return torch.cat(
        [
            torch.zeros((batch_size, 1, 2), device=device),  # x, y
            torch.ones((batch_size, 1, 1), device=device),  # cos(yaw)
            torch.zeros((batch_size, 1, 1), device=device),  # sin(yaw)
        ],
        dim=-1,
    )  # (B, 1, 4)


def make_lane_pos(x: torch.Tensor, lane_len: int, class_type: int):
    """Build a per-element positional token from raw lane/route geometry.

    Mirrors ``LaneEncoder``'s positional logic but emits it in *this* network's
    token vocabulary, so the positions of the reused ``LaneEncoder`` (which are
    encoded in the main encoder's vocabulary) can be discarded.

    Args:
        x: (B, P, V, D) raw lane tensor. D layout starts with (x, y, dx, dy).
        lane_len: Number of points per element (V).
        class_type: This network's class-type id for the element.
    Returns:
        (B, P, 4 + CLASS_TYPE_NUM) positional tokens.
    """
    pos = x[:, :, int(lane_len / 2), :4].clone()  # x, y, dx, dy
    heading = torch.atan2(pos[..., 3], pos[..., 2])
    pos = torch.stack([pos[..., 0], pos[..., 1], torch.cos(heading), torch.sin(heading)], dim=-1)
    return add_class_type(pos, class_type)


class TrajectoryEncoder(nn.Module):
    """Encode an ego trajectory (B, T, D) into a single cross-attention query.

    Reuses the shared ``MixerBlock`` (repo convention), but operates at
    ``hidden_dim`` width throughout: a linear pose embedding, ``depth`` mixer
    blocks, a final norm and global average pooling over time.
    """

    def __init__(
        self,
        time_len: int,
        pose_dim: int,
        hidden_dim: int,
        drop_path_rate: float = 0.0,
        depth: int = 3,
    ):
        super(TrajectoryEncoder, self).__init__()

        self.input_projection = nn.Linear(pose_dim, hidden_dim)
        # Token mixing runs on the time axis (tokens_mlp_dim = time_len); channel
        # mixing runs at hidden_dim, so no width bottleneck / re-projection.
        self.blocks = nn.ModuleList(
            [MixerBlock(time_len, hidden_dim, drop_path_rate) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, D) tensor of trajectory data, where D is the feature dimension.
        Returns:
            x: (B, 1, hidden_dim) query token.
        """
        x = self.input_projection(x)  # (B, T, hidden_dim)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # global average pooling over time
        return torch.mean(x, dim=1, keepdim=True)  # (B, 1, hidden_dim)


class TurnIndicatorHistoryEncoder(nn.Module):
    """Encode the past turn-indicator states into a single fusion token."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int = TURN_INDICATOR_HISTORY_NUM_CLASSES,
    ):
        super(TurnIndicatorHistoryEncoder, self).__init__()
        self.num_classes = num_classes
        self.mlp = Mlp(
            in_features=input_dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T) tensor of turn indicator history, where each element is an
               integer representing the turn indicator class.
        Returns:
            x: (B, 1, hidden_dim) encoded turn indicator history token.
            mask: (B, 1) boolean mask (never masked).
            pos: (B, 1, 4 + CLASS_TYPE_NUM) positional token.
        """
        B = x.shape[0]

        x = F.one_hot(x, num_classes=self.num_classes).float()  # (B, T, num_classes)
        x = x.view(B, -1)  # (B, T * num_classes)
        x = self.mlp(x)  # (B, hidden_dim)
        x = x.unsqueeze(1)  # (B, 1, hidden_dim)

        mask = torch.zeros((B, 1), device=x.device, dtype=torch.bool)
        pos = add_class_type(make_baselink_pose(B, x.device), CLASS_TYPE_TURN_INDICATOR_HISTORY)

        return x, mask, pos


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention + MLP block (query attends to key/value)."""

    def __init__(self, dim, heads, dropout):
        super().__init__()
        mlp_ratio = 4.0

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)

        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=dropout
        )

    def forward(self, q, kv, kv_mask):
        kv = self.norm_kv(kv)
        q = q + self.drop_path(
            self.attn(self.norm_q(q), kv, kv, key_padding_mask=kv_mask, need_weights=False)[0]
        )
        q = q + self.drop_path(self.mlp(self.norm2(q)))
        return q


class CrossFusionEncoder(nn.Module):
    """Stack of cross-attention blocks fusing key/value context into the query."""

    def __init__(self, hidden_dim, num_heads, drop_path_rate, depth):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(hidden_dim, num_heads, dropout=drop_path_rate)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, q, kv, kv_mask):
        for b in self.blocks:
            q = b(q, kv, kv_mask)
        return self.norm(q)


class TurnIndicatorNetwork(nn.Module):
    """Standalone network predicting the turn-indicator logit.

    Independent from the diffusion Encoder/Decoder. The (future) ego trajectory
    is encoded into a query token which cross-attends to key/value tokens built
    from the past turn-indicator states, the lane map, and the route; the fused
    query then predicts a logit over ``TURN_INDICATOR_OUTPUT_DIM`` classes.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mixer_depth: int,
        fusion_depth: int,
        drop_path_rate: float = 0.0,
    ):
        super(TurnIndicatorNetwork, self).__init__()

        self.trajectory_encoder = TrajectoryEncoder(
            time_len=OUTPUT_T,
            pose_dim=POSE_DIM,
            hidden_dim=hidden_dim,
            drop_path_rate=drop_path_rate,
            depth=mixer_depth,
        )
        self.turn_indicator_history_encoder = TurnIndicatorHistoryEncoder(
            input_dim=INPUT_T * TURN_INDICATOR_HISTORY_NUM_CLASSES,
            hidden_dim=hidden_dim,
            num_classes=TURN_INDICATOR_HISTORY_NUM_CLASSES,
        )
        # Reused for feature/mask extraction only; their positional outputs are
        # recomputed locally (make_lane_pos) so the class_type here just tags the
        # discarded internal positions.
        self.lane_encoder = LaneEncoder(
            POINTS_PER_LANELET,
            class_type=CLASS_TYPE_LANE,
            drop_path_rate=drop_path_rate,
            hidden_dim=hidden_dim,
            depth=mixer_depth,
            tokens_mlp_dim=32,
            channels_mlp_dim=64,
        )
        self.route_encoder = LaneEncoder(
            POINTS_PER_LANELET,
            class_type=CLASS_TYPE_ROUTE,
            drop_path_rate=drop_path_rate,
            hidden_dim=hidden_dim,
            depth=mixer_depth,
            tokens_mlp_dim=32,
            channels_mlp_dim=64,
        )

        # position embedding for key/value tokens encodes x, y, cos, sin, type
        self.pos_emb = nn.Linear(4 + CLASS_TYPE_NUM, hidden_dim)

        self.fusion = CrossFusionEncoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            drop_path_rate=drop_path_rate,
            depth=fusion_depth,
        )

        self.head = nn.Linear(hidden_dim, TURN_INDICATOR_OUTPUT_DIM)

    def _encode_query(self, ego_trajectory: torch.Tensor) -> torch.Tensor:
        """Encode the ego trajectory into the cross-attention query.

        Args:
            ego_trajectory: (B, T, D) ego trajectory to condition on.
        Returns:
            query: (B, 1, hidden) query token.
        """
        return self.trajectory_encoder(ego_trajectory)

    def _encode_key_value(
        self, inputs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode the key/value context tokens from history, lanes and route.

        Args:
            inputs: Dict containing lanes / route / turn_indicators tensors.
        Returns:
            kv: (B, N, hidden) key/value tokens.
            kv_mask: (B, N) boolean padding mask (True = masked out).
            pos: (B, N, 4 + CLASS_TYPE_NUM) positional tokens.
        """
        lanes = inputs["lanes"]
        route = inputs["route_lanes"]

        # Drop the current step so the history matches INPUT_T.
        turn_indicator_history = inputs["turn_indicators"][:, :-1].long()  # (B, INPUT_T)

        enc_hist, mask_hist, pos_hist = self.turn_indicator_history_encoder(turn_indicator_history)
        enc_lane, mask_lane, _ = self.lane_encoder(
            lanes, inputs["lanes_speed_limit"], inputs["lanes_has_speed_limit"]
        )
        enc_route, mask_route, _ = self.route_encoder(
            route, inputs["route_lanes_speed_limit"], inputs["route_lanes_has_speed_limit"]
        )

        # LaneEncoder emits positions in the main encoder's vocabulary, so build
        # them locally in this network's vocabulary instead.
        pos_lane = make_lane_pos(lanes, POINTS_PER_LANELET, CLASS_TYPE_LANE)
        pos_route = make_lane_pos(route, POINTS_PER_LANELET, CLASS_TYPE_ROUTE)

        kv = torch.cat([enc_hist, enc_lane, enc_route], dim=1)  # (B, N, hidden)
        kv_mask = torch.cat([mask_hist, mask_lane, mask_route], dim=1)  # (B, N)
        pos = torch.cat([pos_hist, pos_lane, pos_route], dim=1)  # (B, N, 4 + CLASS_TYPE_NUM)

        return kv, kv_mask, pos

    def _add_positional_embedding(
        self, kv: torch.Tensor, kv_mask: torch.Tensor, pos: torch.Tensor
    ) -> torch.Tensor:
        """Add positional embeddings to key/value tokens, zeroed on masked ones.

        Args:
            kv: (B, N, hidden) key/value tokens.
            kv_mask: (B, N) boolean padding mask (True = masked out).
            pos: (B, N, 4 + CLASS_TYPE_NUM) positional tokens.
        Returns:
            kv: (B, N, hidden) key/value tokens with positional embeddings added.
        """
        B, N, _ = kv.shape

        pos_emb = self.pos_emb(pos.view(B * N, -1))
        pos_emb = torch.where(
            (~kv_mask.view(-1)).unsqueeze(-1),
            pos_emb,
            torch.zeros_like(pos_emb),
        )
        return kv + pos_emb.view(B, N, -1)

    def forward(self, ego_trajectory: torch.Tensor, inputs: dict[str, torch.Tensor]):
        """
        Args:
            ego_trajectory: (B, T, D) ego trajectory to condition on.
            inputs: Dict containing lanes / route / turn_indicators tensors.
        Returns:
            turn_indicator_logit: (B, TURN_INDICATOR_OUTPUT_DIM)
        """
        query = self._encode_query(ego_trajectory)  # (B, 1, hidden)

        kv, kv_mask, pos = self._encode_key_value(inputs)
        kv = self._add_positional_embedding(kv, kv_mask, pos)

        # The history token is always valid, so no key/value row is fully masked
        # (which would make attention produce NaNs).
        fused = self.fusion(query, kv, kv_mask)  # (B, 1, hidden)

        return self.head(fused.squeeze(1))


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------


def build_turn_indicator_network(config: Config) -> TurnIndicatorNetwork:
    """Instantiate the head with the same halved hyperparameters the Decoder uses."""
    return TurnIndicatorNetwork(
        hidden_dim=config.hidden_dim // 2,
        num_heads=config.num_heads // 2,
        mixer_depth=config.encoder_mixer_depth // 2,
        fusion_depth=config.encoder_fusion_depth // 2,
        drop_path_rate=config.encoder_drop_path_rate,
    ).eval()


def load_turn_indicator_state_dict(path: Path, use_ema: bool) -> dict[str, torch.Tensor]:
    """Read a standalone turn-indicator checkpoint, rooted at TurnIndicatorNetwork."""
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        key = "ema_state_dict" if use_ema else "model"
        if key not in ckpt:
            raise SystemExit(f"'{key}' not found in {path} (keys: {sorted(ckpt)})")
        state_dict = ckpt[key]
        print(f"Turn-indicator checkpoint: {path} (key '{key}', epoch={ckpt.get('epoch')})")
    else:
        state_dict = ckpt  # bare state_dict
        print(f"Turn-indicator checkpoint: {path} (bare state_dict)")
    return {k[len("module.") :] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def overwrite_turn_indicator(model, network: TurnIndicatorNetwork) -> None:
    """Make the planner's ``turn_indicator_logit`` come from ``network``.

    The injected head replaces the decoder's predictor module, and the decoder's forward
    is wrapped so the value published under ``turn_indicator_logit`` is overwritten with
    the injected head's output. The output schema is left untouched.
    """
    decoder = model.decoder
    decoder.independent_turn_indicator_predictor = network

    original_forward = decoder.forward

    def forward(encoding, inputs):
        outputs = original_forward(encoding, inputs)
        outputs["turn_indicator_logit"] = outputs["independent_turn_indicator_logit"]
        return outputs

    decoder.forward = forward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="full Diffusion Planner .pth")
    parser.add_argument(
        "turn_indicator_checkpoint", type=Path, help="standalone turn-indicator .pth"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output ONNX path (default: diffusion_planner.onnx next to the checkpoint)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="args.json for the full model (default: next to the checkpoint)",
    )
    parser.add_argument(
        "--use-ema", action="store_true", help="use EMA weights from both checkpoints"
    )
    parser.add_argument("--use-simplify", action="store_true", help="run onnxsim on the output")
    parser.add_argument("--opset-version", type=int, default=20)
    parser.add_argument(
        "--external-data", action="store_true", help="store weights as separate files"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = args.config or args.checkpoint.parent / "args.json"
    if not config_path.exists():
        raise SystemExit(f"config not found: {config_path} (pass --config)")
    output_path = args.output or args.checkpoint.parent / "diffusion_planner.onnx"

    torch.manual_seed(42)

    model = load_model(str(config_path), str(args.checkpoint), args.use_ema)

    network = build_turn_indicator_network(Config(str(config_path)))
    state_dict = load_turn_indicator_state_dict(args.turn_indicator_checkpoint, args.use_ema)
    network.load_state_dict(state_dict)
    overwrite_turn_indicator(model, network)
    print(f"Overwrote turn_indicator_logit with the injected head ({len(state_dict)} tensors)")

    wrapper = FullONNXWrapper(model).eval()
    with onnx_export_backends():
        export_onnx(
            wrapper,
            build_dummy_inputs(),
            FULL_INPUT_NAMES,
            FULL_OUTPUT_NAMES,
            output_path,
            args.use_simplify,
            args.opset_version,
            args.external_data,
        )

    print(f"\nWrote {output_path} (outputs: {FULL_OUTPUT_NAMES})")


if __name__ == "__main__":
    main()
