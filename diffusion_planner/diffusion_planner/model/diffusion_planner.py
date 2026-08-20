import torch.nn as nn

from diffusion_planner.model.module.decoder import Decoder
from diffusion_planner.model.module.drivor_decoder import DrivoRDecoder
from diffusion_planner.model.module.encoder import Encoder

PREDICTOR_HEADS = ("diffusion", "drivor")


class Diffusion_Planner(nn.Module):
    def __init__(self, config):
        super().__init__()
        head = getattr(config, "predictor_head", "diffusion")
        if head not in PREDICTOR_HEADS:
            raise ValueError(f"predictor_head must be one of {PREDICTOR_HEADS}, got {head!r}")
        self.predictor_head = head

        self.encoder = Encoder(config)
        self.decoder = DrivoRDecoder(config) if head == "drivor" else Decoder(config)

    @property
    def sde(self):
        # The DrivoR head is not a diffusion model; nothing that reads ``sde``
        # (the DPM solver, the noise schedule) applies to it.
        if self.predictor_head != "diffusion":
            return None
        return self.decoder.sde

    def forward(self, inputs):
        if self.predictor_head == "drivor":
            encoder_outputs, encoding_mask = self.encoder(inputs, return_mask=True)
            decoder_outputs = self.decoder(encoder_outputs, encoding_mask)
            return encoder_outputs, decoder_outputs

        encoder_outputs = self.encoder(inputs)
        decoder_outputs = self.decoder(encoder_outputs, inputs)

        return encoder_outputs, decoder_outputs
