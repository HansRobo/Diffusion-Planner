import json

import torch

from diffusion_planner.utils.normalizer import (
    ControlNormalizer,
    ObservationNormalizer,
    StateNormalizer,
)


def model_flag(config, name: str) -> bool:
    """Read an optional model-behaviour flag, defaulting to off.

    ``Config`` only sets the keys its ``args.json`` actually contains, so a checkpoint
    trained before a flag existed has no such attribute. A plain ``config.<name>`` would
    therefore break ONNX re-export for every checkpoint predating the flag; this returns
    False for them, which is the pre-flag behaviour by construction.

    Works for both ``Config`` (attributes set from JSON) and ``TrainConfig`` (dataclass).
    """
    return bool(getattr(config, name, False))


class Config:
    def __init__(self, args_file, guidance_fn=None):
        with open(args_file, "r") as f:
            args_dict = json.load(f)

        for key, value in args_dict.items():
            setattr(self, key, value)

        self.state_normalizer = StateNormalizer(
            self.state_normalizer["mean"], self.state_normalizer["std"]
        )
        self.observation_normalizer = ObservationNormalizer(
            {
                k: {"mean": torch.as_tensor(v["mean"]), "std": torch.as_tensor(v["std"])}
                for k, v in self.observation_normalizer.items()
            }
        )
        self.control_normalizer = ControlNormalizer(
            self.control_normalizer["mean"], self.control_normalizer["std"]
        )
        self.neighbor_control_normalizer = ControlNormalizer(
            self.neighbor_control_normalizer["mean"], self.neighbor_control_normalizer["std"]
        )

        self.guidance_fn = guidance_fn
