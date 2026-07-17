import numpy as np
from torch.utils.data import Dataset

from diffusion_planner.utils.train_utils import openjson
from diffusion_planner.utils.neighbor_future_alignment import (
    align_neighbor_future_numpy,
)
from planner_metrics.temporal_stability import consecutive_frame_pairs


class DiffusionPlannerData(Dataset):
    def __init__(self, data_list):
        self.data_list = openjson(data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx], allow_pickle=True)
        data = dict(data)  # npz to dict
        data.pop("version", None)
        if "neighbor_agents_future" in data:
            # Keep ordinary DP/SFT and RL loaders on the HDP-compatible
            # temporal convention. Set DP_NEIGHBOR_FUTURE_OFFSET=0 when a
            # regenerated archive is already future-only.
            data["neighbor_agents_future"] = align_neighbor_future_numpy(
                data["neighbor_agents_future"]
            )
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
        return {
            "current": data_a,
            "next": data_b,
            "frame_gap": np.array(gap, dtype=np.int64),
        }
