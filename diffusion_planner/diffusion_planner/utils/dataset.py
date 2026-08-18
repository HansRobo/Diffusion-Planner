import os

import numpy as np
from torch.utils.data import Dataset

from diffusion_planner.utils.train_utils import openjson
from planner_metrics.temporal_stability import consecutive_frame_pairs

_NEIGHBOR_PAST_COLS = 12


def _match_neighbor_past_width(neighbor_agents_past: np.ndarray) -> np.ndarray:
    """Widen a legacy 11-col ``neighbor_agents_past`` array to 12 (append a zero Unknown
    column). Real NPZ corpora written before the Unknown-class change are 11-wide; a missing
    4th one-hot column is exactly "no Unknown label", which is correct for that legacy data.
    """
    cols = neighbor_agents_past.shape[-1]
    if cols == _NEIGHBOR_PAST_COLS:
        return neighbor_agents_past
    if cols == _NEIGHBOR_PAST_COLS - 1:
        pad = np.zeros(neighbor_agents_past.shape[:-1] + (1,), dtype=neighbor_agents_past.dtype)
        return np.concatenate([neighbor_agents_past, pad], axis=-1)
    raise ValueError(
        f"neighbor_agents_past has unexpected width {cols} "
        f"(expected {_NEIGHBOR_PAST_COLS - 1} or {_NEIGHBOR_PAST_COLS})"
    )


class DiffusionPlannerData(Dataset):
    def __init__(self, data_list):
        if isinstance(data_list, (str, bytes, os.PathLike)):
            self.data_list = openjson(data_list)
        else:
            self.data_list = list(data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx], allow_pickle=True)
        data = dict(data)  # npz to dict
        data.pop("version", None)
        if "neighbor_agents_past" in data:
            data["neighbor_agents_past"] = _match_neighbor_past_width(data["neighbor_agents_past"])
        return data


class DiffusionPlannerPairData(Dataset):
    def __init__(self, data_list, expected_gap: int | None = None):
        paths = openjson(data_list)
        expected_gap = expected_gap or None
        self.pairs = list(consecutive_frame_pairs(paths, expected_gap=expected_gap))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        _, path_a, _, path_b, gap = self.pairs[idx]
        data_a = dict(np.load(path_a, allow_pickle=True))
        data_b = dict(np.load(path_b, allow_pickle=True))
        for data in (data_a, data_b):
            if "neighbor_agents_past" in data:
                data["neighbor_agents_past"] = _match_neighbor_past_width(
                    data["neighbor_agents_past"]
                )
        return {
            "current": data_a,
            "next": data_b,
            "frame_gap": np.array(gap, dtype=np.int64),
        }
