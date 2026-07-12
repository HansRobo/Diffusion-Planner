import torch.nn as nn

from diffusion_planner.model.module.decoder import Decoder
from diffusion_planner.model.module.encoder import Encoder


class Diffusion_Planner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    @property
    def sde(self):
        return self.decoder.sde

    def forward(self, inputs):
        # RL group sampling and multi-sample validation share scene observations across
        # candidates. Reusing a detached encoding avoids repeating the expensive scene encoder.
        encoder_outputs = inputs.get("_cached_encoding")
        if encoder_outputs is None:
            encoder_outputs = self.encoder(inputs)
            repeat_interleave = int(inputs.get("_encoder_repeat_interleave", 1))
            if repeat_interleave < 1:
                raise ValueError("_encoder_repeat_interleave must be >= 1")
            if repeat_interleave > 1:
                encoder_outputs = encoder_outputs.repeat_interleave(repeat_interleave, dim=0)
        decoder_outputs = self.decoder(encoder_outputs, inputs)

        return encoder_outputs, decoder_outputs
