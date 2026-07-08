import json

import torch

from diffusion_planner.utils.hdp_compat import require_velocity_normalizer
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer


class Config:
    def __init__(self, args_file, guidance_fn=None):
        with open(args_file, "r") as f:
            args_dict = json.load(f)

        for key, value in args_dict.items():
            setattr(self, key, value)
        defaults = {
            "use_velocity_representation": False,
            "planning_hybrid_loss": 0.0,
            "hybrid_loss_window": 10,
            "diffusion_supervision_type": getattr(self, "diffusion_model_type", "x_start"),
            "diffusion_time_sample_method": "uniform",
            "diffusion_sample_steps": 10,
            "official_reward_normalize": "group",
            "official_reward_beta": 1.0,
            "rl_noise_scale": 0.5,
        }
        for key, value in defaults.items():
            if not hasattr(self, key):
                setattr(self, key, value)
        state_normalizer = getattr(self, "state_normalizer", None)
        if not isinstance(state_normalizer, dict):
            raise RuntimeError("args.json/state_normalizer is required to load Diffusion Planner.")
        if self.use_velocity_representation:
            require_velocity_normalizer(state_normalizer, "args.json/state_normalizer")
        self.state_normalizer = StateNormalizer(
            state_normalizer["mean"],
            state_normalizer["std"],
            state_normalizer.get("ego_velocity_mean"),
            state_normalizer.get("ego_velocity_std"),
        )
        self.observation_normalizer = ObservationNormalizer(
            {
                k: {"mean": torch.as_tensor(v["mean"]), "std": torch.as_tensor(v["std"])}
                for k, v in self.observation_normalizer.items()
            }
        )

        self.guidance_fn = guidance_fn

        # Default guidance scale; overridable without reloading the model.
        if not hasattr(self, "guidance_scale"):
            self.guidance_scale = 0.5
