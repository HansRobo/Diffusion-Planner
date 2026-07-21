import json

import torch

from diffusion_planner.utils.hdp_compat import require_velocity_normalizer
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer


class Config:
    def __init__(self, args_file):
        with open(args_file, "r") as f:
            args_dict = json.load(f)

        for key, value in args_dict.items():
            setattr(self, key, value)
        defaults = {
            "use_velocity_representation": False,
            "diffusion_model_type": "x_start",
            "planning_hybrid_loss": 0.0,
            "hybrid_loss_window": 10,
            "diffusion_supervision_type": getattr(self, "diffusion_model_type", "x_start"),
            "diffusion_time_sample_method": "uniform",
            "diffusion_sample_steps": 6,
            "num_generations": 8,
            "rl_reward_normalize": getattr(self, "official_reward_normalize", "group"),
            "rl_reward_beta": getattr(self, "official_reward_beta", 0.5),
            "rl_noise_scale": 1.5,
            "rl_eval_noise_scale": 0.5,
            "rl_eval_num_generations": 32,
            # Preserve inference behavior for pre-window-contract checkpoints
            # that do not have this explicit field. New runs serialize 21.
            "ego_history_frames": 6,
        }
        for key, value in defaults.items():
            if not hasattr(self, key):
                setattr(self, key, value)
        tokenization = getattr(self, "decoder_tokenization", None)
        if tokenization != "temporal":
            raise RuntimeError(
                "This HDP branch requires a temporal-token decoder checkpoint; "
                f"args.json has decoder_tokenization={tokenization!r}."
            )
        if not self.use_velocity_representation:
            raise RuntimeError("This HDP branch requires a velocity-representation checkpoint.")
        if self.diffusion_model_type != "x_start" or self.diffusion_supervision_type != "x_start":
            raise RuntimeError("This HDP branch requires x_start prediction and supervision.")
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
