"""Encoder that consumes rendered BEV images instead of the raw vector scene.

Each sample carries ``NUM_SCALES`` multi-channel BEV rasters (see
:mod:`diffusion_planner.utils.render_bev`).  One trunk, shared across every scale, turns each
raster into a small grid of tokens; those become the tokens handed to the decoder, tagged with
a learned cell position and a learned per-scale embedding.  ``config.image_backbone`` picks the
trunk, and every choice yields the same token count, so the decoder never notices the swap.

A raster cannot carry everything the planner needs, so a few scalar tokens are appended,
reusing the same small encoders as the vector pipeline:

* ego motion -- the history channel is a binary polyline with no time markers, so it gives
  the path shape and the mean speed over the window, but not the instantaneous velocity or
  the acceleration.  Only those four numbers are passed; the pose is always the origin, and
  the ego footprint is already drawn as a box on ``CH_EGO``.
* turn indicators -- no pixel representation at all.

The goal pose is deliberately NOT among them.  It is the route destination, typically several
hundred metres out, so as a scalar it is a distraction far more often than it is guidance; the
raster draws it on ``CH_GOAL_POSE`` on the rare frames where it actually falls inside a view,
and the route channel carries the intent that matters within the horizon.
"""

import timm
import torch
import torch.nn as nn
from torchvision.models import resnet18

from diffusion_planner.dimensions import INPUT_T
from diffusion_planner.model.module.encoder import FloatsEncoder, FusionEncoder
from diffusion_planner.utils.render_bev import NUM_CHANNELS, NUM_SCALES

# (vx, vy, ax, ay) inside ego_current_state; the remaining fields are either identically
# constant in the ego frame (pose) or already visible in the raster.
EGO_MOTION_SLICE = slice(4, 8)

RESNET_FEATURE_DIM = 512
RESNET_STRIDE = 32  # total downsampling factor of resnet18 up to layer4
UINT8_SCALE = 255.0

# Side of the token neighbourhood the merger folds into one token.  A ViT/16 leaves a 14x14
# grid per 224px view against the ResNet's 7x7, so merging 2x2 restores the ResNet's token
# count and every trunk hands the decoder exactly the same number of tokens.
PIXEL_MERGE_SIZE = 2

# Width of the per-pixel MLP that folds the semantic planes down to the 3 channels a pretrained
# trunk expects.  Parameters are not what this costs: at 1x1 it holds barely a thousand of
# them, but it runs at the full raster resolution, so its activations are what the backward
# pass has to keep.  Measured at batch 32, every doubling of this width adds ~0.4 GiB of peak
# memory and ~0.2 ms per sample, which is why it is not simply set generously wide.
CHANNEL_ADAPTER_HIDDEN = 32

# timm entry point of every pretrained ViT trunk that can stand in for the ResNet.  Both are
# patch-16, so they land on the same 14x14 grid and the merger below restores the ResNet's
# token count; DINOv2 is patch-14 and would not, which is why it is absent.
TIMM_MODEL_OF_BACKBONE = {
    "dinov3_small": "vit_small_patch16_dinov3.lvd1689m",
    "dinov3_base": "vit_base_patch16_dinov3.lvd1689m",
}


class BevBackbone(nn.Module):
    """ResNet18 trunk adapted to ``NUM_CHANNELS`` semantic input planes.

    ImageNet weights are not loaded: the input planes are binary semantic masks whose
    statistics have nothing in common with natural images, and the first convolution would
    have to be reshaped anyway.
    """

    def __init__(self, in_channels: int, image_size: int, out_dim: int):
        super().__init__()
        assert image_size % RESNET_STRIDE == 0, (
            f"bev_image_size={image_size} must be a multiple of {RESNET_STRIDE}"
        )
        self.grid_size = image_size // RESNET_STRIDE
        self.tokens_per_image = self.grid_size * self.grid_size

        trunk = resnet18(weights=None)
        trunk.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.stem = nn.Sequential(trunk.conv1, trunk.bn1, trunk.relu, trunk.maxpool)
        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3
        self.layer4 = trunk.layer4
        self.proj = nn.Conv2d(RESNET_FEATURE_DIM, out_dim, kernel_size=1)

    def forward(self, images):
        """images: (N, C, H, W) float -> (N, tokens_per_image, out_dim)."""
        x = self.stem(images)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class ChannelAdapter(nn.Module):
    """Learned per-pixel MLP folding the semantic planes into the 3 channels a trunk expects.

    Staying at 3 channels keeps the pretrained patch embedding usable instead of discarding it
    for a randomly initialised ``NUM_CHANNELS`` convolution, and a learned fold beats a fixed
    colour palette: which planes deserve to survive the squeeze is exactly the kind of question
    the data can answer and a hand-picked palette cannot.

    It is always trained, including when the trunk is frozen -- which is why the frozen trunk
    is not run under ``no_grad``: gradients have to reach back through it to here.  The 1x1
    convolutions make this an MLP applied independently to every pixel, so it re-weights planes
    but never mixes neighbours; all spatial work stays in the trunk.

    No fixed input normalisation follows it.  The pretrained mean/std is an affine map, which
    this MLP's last layer can represent and absorb on its own.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, images):
        """(N, NUM_CHANNELS, H, W) in [0, 1] -> (N, 3, H, W) in the trunk's input space."""
        return self.mlp(images)


class PixelMerger(nn.Module):
    """Qwen-VL's patch merger: LayerNorm, 2x2 concat, Linear, GELU, Linear.

    Folding a 2x2 neighbourhood with a learned projection rather than with pooling keeps the
    detail the four tokens disagree on, and brings the ViT grid back to the token count the
    decoder already saw from the ResNet.
    """

    def __init__(self, in_dim: int, grid_size: int, merge_size: int, out_dim: int):
        super().__init__()
        assert grid_size % merge_size == 0, (
            f"token grid {grid_size} is not divisible by merge size {merge_size}"
        )
        self.merge_size = merge_size
        self.merged_grid = grid_size // merge_size
        merged_dim = in_dim * merge_size * merge_size

        self.norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(merged_dim, merged_dim),
            nn.GELU(),
            nn.Linear(merged_dim, out_dim),
        )

    def forward(self, tokens):
        """(N, grid_size ** 2, in_dim) -> (N, merged_grid ** 2, out_dim)."""
        num_images = tokens.shape[0]
        merge = self.merge_size
        x = self.norm(tokens)
        x = x.view(num_images, self.merged_grid, merge, self.merged_grid, merge, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(num_images, self.merged_grid**2, -1)
        return self.mlp(x)


class ViTBackbone(nn.Module):
    """A pretrained timm ViT trunk plus the 2x2 merger, standing in for :class:`BevBackbone`.

    ``freeze`` leaves the pretrained weights exactly as they came and holds the trunk in eval
    mode, so its features stay deterministic.  Gradients still flow *through* it, because the
    channel adapter in front of it always trains; only the trunk's own weights are spared a
    gradient and an optimizer slot.
    """

    def __init__(self, backbone: str, image_size: int, out_dim: int, freeze: bool):
        super().__init__()
        assert backbone in TIMM_MODEL_OF_BACKBONE, f"unknown image_backbone {backbone}"

        self.freeze = freeze
        self.trunk = timm.create_model(
            TIMM_MODEL_OF_BACKBONE[backbone],
            pretrained=True,
            num_classes=0,
            global_pool="",
            img_size=image_size,
            in_chans=3,
            # DINOv3 defaults to accepting any resolution, which makes it rebuild its rotary
            # position embedding from the input's shape on every call.  That traces into ONNX as
            # Shape-driven branches whose outputs have no static rank, and TensorRT refuses to
            # build the graph.  The raster size is fixed at bev_image_size, so the embedding is a
            # constant: pinning the resolution caches it once, leaves the outputs bit-identical,
            # and lets the graph through.
            dynamic_img_size=False,
        )
        # Registers and a class token sit in front of the patch tokens in ``forward_features``;
        # they carry no position, so the merger must never see them.
        self.num_prefix_tokens = self.trunk.num_prefix_tokens

        patch_size = self.trunk.patch_embed.patch_size[0]
        assert image_size % patch_size == 0, (
            f"bev_image_size={image_size} must be a multiple of patch size {patch_size}"
        )
        grid_size = image_size // patch_size

        self.to_pixels = ChannelAdapter(
            in_channels=NUM_CHANNELS,
            hidden_channels=CHANNEL_ADAPTER_HIDDEN,
            out_channels=self.trunk.patch_embed.proj.in_channels,
        )

        self.merger = PixelMerger(
            in_dim=self.trunk.embed_dim,
            grid_size=grid_size,
            merge_size=PIXEL_MERGE_SIZE,
            out_dim=out_dim,
        )
        self.tokens_per_image = self.merger.merged_grid**2

        if freeze:
            self.trunk.requires_grad_(False)
            self.trunk.eval()

    def train(self, mode: bool = True):
        """Keep a frozen trunk in eval mode however the enclosing model is switched."""
        super().train(mode)
        if self.freeze:
            self.trunk.eval()
        return self

    def forward(self, images):
        """images: (N, NUM_CHANNELS, H, W) float in [0, 1] -> (N, tokens_per_image, out_dim)."""
        pixels = self.to_pixels(images)
        features = self.trunk.forward_features(pixels)
        return self.merger(features[:, self.num_prefix_tokens :, :])


def build_bev_backbone(config):
    """Pick the raster trunk matching ``config.image_backbone``."""
    if config.image_backbone == "resnet18":
        assert not config.freeze_image_backbone, (
            "the ResNet is trained from scratch, so freezing it would leave it random"
        )
        return BevBackbone(
            in_channels=NUM_CHANNELS,
            image_size=config.bev_image_size,
            out_dim=config.hidden_dim,
        )
    return ViTBackbone(
        backbone=config.image_backbone,
        image_size=config.bev_image_size,
        out_dim=config.hidden_dim,
        freeze=config.freeze_image_backbone,
    )


class ImageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_dim = config.hidden_dim
        self.use_turn_indicators = config.use_turn_indicators

        self.image_size = config.bev_image_size
        self.backbone = build_bev_backbone(config)
        self.tokens_per_image = self.backbone.tokens_per_image
        self.image_token_num = NUM_SCALES * self.tokens_per_image

        # One embedding per feature-map cell, shared by both scales, plus one per scale so
        # the decoder can tell the near view from the far view.
        self.cell_embedding = nn.Parameter(
            torch.randn(1, self.tokens_per_image, config.hidden_dim) * 0.02
        )
        self.scale_embedding = nn.Parameter(torch.randn(NUM_SCALES, 1, config.hidden_dim) * 0.02)

        # Scene state that has no pixel representation.
        self.ego_motion_encoder = FloatsEncoder(
            num_float=EGO_MOTION_SLICE.stop - EGO_MOTION_SLICE.start,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
        )
        self.turn_indicator_encoder = FloatsEncoder(
            num_float=INPUT_T,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
        )
        self.scalar_token_num = 2
        self.token_num = self.image_token_num + self.scalar_token_num

        self.fusion = FusionEncoder(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            drop_path_rate=config.encoder_drop_path_rate,
            depth=config.encoder_fusion_depth,
        )

        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.ego_motion_encoder.apply(_basic_init)
        self.turn_indicator_encoder.apply(_basic_init)
        self.fusion.apply(_basic_init)

    def encode_images(self, images):
        """images: (B, S, C, H, W) uint8 or float -> (B, S * tokens_per_image, hidden_dim)."""
        B, S = images.shape[:2]
        if images.dtype == torch.uint8:
            images = images.float() / UINT8_SCALE
        flat = images.reshape(B * S, *images.shape[2:])

        tokens = self.backbone(flat)  # (B * S, tokens_per_image, hidden_dim)
        tokens = tokens + self.cell_embedding
        tokens = tokens.view(B, S, self.tokens_per_image, self.hidden_dim)
        tokens = tokens + self.scale_embedding
        return tokens.reshape(B, S * self.tokens_per_image, self.hidden_dim)

    def forward(self, inputs):
        images = inputs["bev_image"]  # (B, S, C, H, W)
        image_tokens = self.encode_images(images)

        ego_motion = inputs["ego_current_state"][:, EGO_MOTION_SLICE]  # (B, D=4)

        turn_indicator = inputs["turn_indicators"][:, :-1].float()  # (B, T)
        if not self.use_turn_indicators:
            turn_indicator = torch.zeros_like(turn_indicator)

        encoding_ego_motion, _, _ = self.ego_motion_encoder(ego_motion)
        encoding_turn_indicator, _, _ = self.turn_indicator_encoder(turn_indicator)

        encoding_input = torch.cat(
            [
                image_tokens,
                encoding_ego_motion,
                encoding_turn_indicator,
            ],
            dim=1,
        )

        # Every token is always present: the rasters are dense and the scalars always exist.
        encoding_mask = torch.zeros(
            (encoding_input.shape[0], self.token_num),
            dtype=torch.bool,
            device=encoding_input.device,
        )

        return self.fusion(encoding_input, encoding_mask)
