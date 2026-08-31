import torch.nn as nn

from diffusion_planner.model.module.decoder import Decoder
from diffusion_planner.model.module.encoder import Encoder
from diffusion_planner.model.module.image_encoder import ImageEncoder


def build_encoder(config):
    """Pick the scene encoder matching ``config.input_type``.

    Configs saved before image input existed carry no ``input_type``; they are vector runs.
    """
    input_type = getattr(config, "input_type", "vector")
    if input_type == "image":
        return ImageEncoder(config)
    if input_type == "vector":
        return Encoder(config)
    raise ValueError(f"Unknown input_type {input_type}")


class Diffusion_Planner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = build_encoder(config)
        self.decoder = Decoder(config)

    @property
    def sde(self):
        return self.decoder.sde

    def forward(self, inputs):
        encoder_outputs = self.encoder(inputs)
        decoder_outputs = self.decoder(encoder_outputs, inputs)

        return encoder_outputs, decoder_outputs
