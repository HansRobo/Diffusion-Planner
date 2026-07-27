from copy import copy

import torch

from diffusion_planner.utils.train_utils import openjson


class StateNormalizer:
    def __init__(self, mean, std, ego_velocity_mean=None, ego_velocity_std=None):
        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)
        self._validate_stats("state", self.mean, self.std)
        if (ego_velocity_mean is None) != (ego_velocity_std is None):
            raise ValueError("ego_velocity_mean and ego_velocity_std must be provided together")
        self.ego_velocity_mean = (
            torch.as_tensor(ego_velocity_mean) if ego_velocity_mean is not None else None
        )
        self.ego_velocity_std = (
            torch.as_tensor(ego_velocity_std) if ego_velocity_std is not None else None
        )
        if self.ego_velocity_mean is not None:
            self._validate_stats("ego_velocity", self.ego_velocity_mean, self.ego_velocity_std)

    @staticmethod
    def _validate_stats(name, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.shape != std.shape:
            raise ValueError(
                f"{name} normalization mean/std shapes differ: "
                f"{tuple(mean.shape)} vs {tuple(std.shape)}"
            )
        mean_float = mean.detach().to(dtype=torch.float64)
        std_float = std.detach().to(dtype=torch.float64)
        if not torch.isfinite(mean_float).all():
            raise ValueError(f"{name} normalization mean contains non-finite values")
        if not torch.isfinite(std_float).all() or (std_float <= 0).any():
            raise ValueError(f"{name} normalization std must contain only finite positive values")

    @classmethod
    def from_json(cls, args):
        data = openjson(args.normalization_file_path)
        if "ego" not in data:
            raise KeyError(
                "'ego' is required in normalization_file_path for waypoint trajectory scaling"
            )
        if getattr(args, "use_velocity_representation", False) and "ego_velocity" not in data:
            raise KeyError(
                "'ego_velocity' is required in normalization_file_path when "
                f"use_velocity_representation={getattr(args, 'use_velocity_representation', False)}"
            )
        mean = [[data["ego"]["mean"]]] + [[data["neighbor"]["mean"]]] * args.predicted_neighbor_num
        std = [[data["ego"]["std"]]] + [[data["neighbor"]["std"]]] * args.predicted_neighbor_num
        ego_velocity = data.get("ego_velocity")
        return cls(
            mean,
            std,
            ego_velocity["mean"] if ego_velocity is not None else None,
            ego_velocity["std"] if ego_velocity is not None else None,
        )

    def _cached_stats(self, name, mean, std, device, dtype=None):
        # Per-device cache: without it every call issues two host->device copies in the
        # training hot loop. getattr-lazy so unpickled instances from old checkpoints work.
        cache = getattr(self, "_device_cache", None)
        if cache is None:
            cache = {}
            self._device_cache = cache
        key = (name, str(device), dtype)
        if key not in cache:
            cache[key] = (
                mean.to(device=device, dtype=dtype),
                std.to(device=device, dtype=dtype),
            )
        return cache[key]

    def _mean_std_on(self, device, dtype=None):
        return self._cached_stats("state", self.mean, self.std, device, dtype)

    def ego_velocity_stats_on(self, device, dtype=None):
        if self.ego_velocity_mean is None or self.ego_velocity_std is None:
            raise RuntimeError(
                "ego_velocity normalization stats are required for HDP velocity mode"
            )
        return self._cached_stats(
            "ego_velocity",
            self.ego_velocity_mean,
            self.ego_velocity_std,
            device,
            dtype,
        )

    def __call__(self, data):
        mean, std = self._mean_std_on(data.device, data.dtype)
        return (data - mean) / std

    def inverse(self, data):
        mean, std = self._mean_std_on(data.device, data.dtype)
        return data * std + mean

    def to_dict(self):
        result = {
            "mean": self.mean.detach().cpu().numpy().tolist(),
            "std": self.std.detach().cpu().numpy().tolist(),
        }
        if self.ego_velocity_mean is not None and self.ego_velocity_std is not None:
            result["ego_velocity_mean"] = self.ego_velocity_mean.detach().cpu().numpy().tolist()
            result["ego_velocity_std"] = self.ego_velocity_std.detach().cpu().numpy().tolist()
        return result


class ObservationNormalizer:
    def __init__(self, normalization_dict):
        self._normalization_dict = {}
        for key, stats in normalization_dict.items():
            if "mean" not in stats or "std" not in stats:
                raise ValueError(f"Normalization entry {key!r} needs mean and std")
            mean = torch.as_tensor(stats["mean"])
            std = torch.as_tensor(stats["std"])
            StateNormalizer._validate_stats(key, mean, std)
            self._normalization_dict[key] = {"mean": mean, "std": std}

    @classmethod
    def from_json(cls, args):
        path = args if isinstance(args, str) else args.normalization_file_path

        data = openjson(path)
        ndt = {}
        for k, v in data.items():
            if k not in ["ego", "ego_velocity", "neighbor"]:
                ndt[k] = {
                    "mean": torch.tensor(v["mean"], dtype=torch.float32),
                    "std": torch.tensor(v["std"], dtype=torch.float32),
                }
        return cls(ndt)

    def stats(self, key):
        """Return the (mean, std) tensors used to normalize input `key`."""
        entry = self._normalization_dict[key]
        return entry["mean"], entry["std"]

    def _stats_on(self, key, device):
        # Per-device cache: without it every __call__/inverse issues ~2 host->device
        # copies per input key per training step. getattr-lazy for unpickle safety.
        cache = getattr(self, "_device_cache", None)
        if cache is None:
            cache = {}
            self._device_cache = cache
        cache_key = (key, str(device))
        if cache_key not in cache:
            v = self._normalization_dict[key]
            cache[cache_key] = (v["mean"].to(device), v["std"].to(device))
        return cache[cache_key]

    def __call__(self, data):
        norm_data = copy(data)
        for k in self._normalization_dict:
            if k not in data:  # Check if key `k` exists in `data`
                continue
            mean, std = self._stats_on(k, data[k].device)
            mask = torch.all(data[k] == 0, dim=-1, keepdim=True)
            norm_data[k] = (data[k] - mean) / std
            norm_data[k].masked_fill_(mask, 0)
        return norm_data

    def inverse(self, data):
        norm_data = copy(data)
        for k in self._normalization_dict:
            if k not in data:  # Check if key `k` exists in `data`
                continue
            mean, std = self._stats_on(k, data[k].device)
            mask = torch.all(data[k] == 0, dim=-1, keepdim=True)
            norm_data[k] = data[k] * std + mean
            norm_data[k].masked_fill_(mask, 0)
        return norm_data

    def to_dict(self):
        return {
            k: {kk: vv.detach().cpu().numpy().tolist() for kk, vv in v.items()}
            for k, v in self._normalization_dict.items()
        }
