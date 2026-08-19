"""Encoder that consumes rendered BEV images instead of the raw vector scene.

Each sample carries ``NUM_SCALES`` multi-channel BEV rasters (see
:mod:`diffusion_planner.utils.render_bev`).  One ResNet, shared across every scale, turns
each raster into a spatial feature map; the cells of that map become the tokens handed to
the decoder, tagged with a learned cell position and a learned per-scale embedding.

A raster cannot carry everything the planner needs: ego speed and acceleration, the ego
footprint, the turn indicator history and the goal pose (which usually lies hundreds of
metres away, far outside even the wide view) have no pixels.  Those are appended as
separate scalar tokens, reusing the same small encoders as the vector pipeline.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18

from diffusion_planner.dimensions import INPUT_T
from diffusion_planner.model.module.encoder import (
    FloatsEncoder,
    FusionEncoder,
    GoalPoseEncoder,
)
from diffusion_planner.utils.render_bev import NUM_CHANNELS, NUM_SCALES

RESNET_FEATURE_DIM = 512
RESNET_STRIDE = 32  # total downsampling factor of resnet18 up to layer4
UINT8_SCALE = 255.0


class BevBackbone(nn.Module):
    """ResNet18 trunk adapted to ``NUM_CHANNELS`` semantic input planes.

    ImageNet weights are not loaded: the input planes are binary semantic masks whose
    statistics have nothing in common with natural images, and the first convolution would
    have to be reshaped anyway.
    """

    def __init__(self, in_channels: int, out_dim: int):
        super().__init__()
        trunk = resnet18(weights=None)
        trunk.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.stem = nn.Sequential(trunk.conv1, trunk.bn1, trunk.relu, trunk.maxpool)
        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3
        self.layer4 = trunk.layer4
        self.proj = nn.Conv2d(RESNET_FEATURE_DIM, out_dim, kernel_size=1)

    def forward(self, images):
        """images: (N, C, H, W) float -> (N, out_dim, H / 32, W / 32)."""
        x = self.stem(images)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.proj(x)


class ImageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_dim = config.hidden_dim
        self.use_turn_indicators = config.use_turn_indicators

        grid_size = config.bev_image_size // RESNET_STRIDE
        if grid_size * RESNET_STRIDE != config.bev_image_size:
            raise ValueError(
                f"bev_image_size={config.bev_image_size} must be a multiple of {RESNET_STRIDE}"
            )
        self.image_size = config.bev_image_size
        self.grid_size = grid_size
        self.tokens_per_image = grid_size * grid_size
        self.image_token_num = NUM_SCALES * self.tokens_per_image

        self.backbone = BevBackbone(in_channels=NUM_CHANNELS, out_dim=config.hidden_dim)

        # One embedding per feature-map cell, shared by both scales, plus one per scale so
        # the decoder can tell the near view from the far view.
        self.cell_embedding = nn.Parameter(
            torch.randn(1, self.tokens_per_image, config.hidden_dim) * 0.02
        )
        self.scale_embedding = nn.Parameter(torch.randn(NUM_SCALES, 1, config.hidden_dim) * 0.02)

        # Scene state that has no pixel representation.
        self.goal_pose_encoder = GoalPoseEncoder(
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
        )
        self.ego_state_encoder = FloatsEncoder(
            num_float=config.ego_current_state_dim,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
        )
        self.ego_shape_encoder = FloatsEncoder(
            num_float=3,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
        )
        self.turn_indicator_encoder = FloatsEncoder(
            num_float=INPUT_T,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
        )
        self.scalar_token_num = 4
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

        self.goal_pose_encoder.apply(_basic_init)
        self.ego_state_encoder.apply(_basic_init)
        self.ego_shape_encoder.apply(_basic_init)
        self.turn_indicator_encoder.apply(_basic_init)
        self.fusion.apply(_basic_init)

    def encode_images(self, images):
        """images: (B, S, C, H, W) uint8 or float -> (B, S * tokens_per_image, hidden_dim)."""
        B, S = images.shape[:2]
        if images.dtype == torch.uint8:
            images = images.float() / UINT8_SCALE
        flat = images.reshape(B * S, *images.shape[2:])

        features = self.backbone(flat)  # (B * S, hidden_dim, G, G)
        tokens = features.flatten(2).transpose(1, 2)  # (B * S, G * G, hidden_dim)
        tokens = tokens + self.cell_embedding
        tokens = tokens.view(B, S, self.tokens_per_image, self.hidden_dim)
        tokens = tokens + self.scale_embedding
        return tokens.reshape(B, S * self.tokens_per_image, self.hidden_dim)

    def forward(self, inputs):
        images = inputs["bev_image"]  # (B, S, C, H, W)
        image_tokens = self.encode_images(images)

        goal_pose = inputs["goal_pose"]  # (B, D=4)
        ego_state = inputs["ego_current_state"]  # (B, D)
        ego_shape = inputs["ego_shape"]  # (B, D=3)

        turn_indicator = inputs["turn_indicators"][:, :-1].float()  # (B, T)
        if not self.use_turn_indicators:
            turn_indicator = torch.zeros_like(turn_indicator)

        encoding_goal_pose, _, _ = self.goal_pose_encoder(goal_pose)
        encoding_ego_state, _, _ = self.ego_state_encoder(ego_state)
        encoding_ego_shape, _, _ = self.ego_shape_encoder(ego_shape)
        encoding_turn_indicator, _, _ = self.turn_indicator_encoder(turn_indicator)

        encoding_input = torch.cat(
            [
                image_tokens,
                encoding_goal_pose,
                encoding_ego_state,
                encoding_ego_shape,
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
