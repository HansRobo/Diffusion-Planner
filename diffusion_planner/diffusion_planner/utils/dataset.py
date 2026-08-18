import os

import numpy as np
from torch.utils.data import Dataset

from diffusion_planner.utils.train_utils import openjson
from planner_metrics.temporal_stability import consecutive_frame_pairs


class DiffusionPlannerData(Dataset):
    def __init__(self, data_list, with_index: bool = False):
        """
        Args:
            data_list: JSON path list, or an iterable of NPZ paths.
            with_index: also return each sample's index under ``"sample_index"``, so a
                caller holding this dataset can map a batch element back to its NPZ path
                (used by the training loop to name the files behind a non-finite step).
                Off by default because every other consumer feeds the returned dict
                straight to the model and an unexpected key would ride along.
        """
        if isinstance(data_list, (str, bytes, os.PathLike)):
            self.data_list = openjson(data_list)
        else:
            self.data_list = list(data_list)
        self.with_index = with_index

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx], allow_pickle=True)
        data = dict(data)  # npz to dict
        data.pop("version", None)
        if self.with_index:
            data["sample_index"] = np.array(idx, dtype=np.int64)
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
