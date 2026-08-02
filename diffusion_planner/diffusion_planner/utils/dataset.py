import os

import numpy as np
from torch.utils.data import Dataset

from diffusion_planner.utils.train_utils import openjson
from planner_metrics.temporal_stability import consecutive_frame_pairs


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
        data["neighbor_agents_future"] = _neighbor_future_to_cos_sin(
            data.get("neighbor_agents_future")
        )
        return data


def _neighbor_future_to_cos_sin(nf):
    """Normalize neighbor_agents_future to 4 channels (x, y, cos, sin) at load time.

    Some converter versions store it as 3 channels (x, y, heading), others as 4
    (x, y, cos, sin). ``train_epoch`` applies ``heading_to_cos_sin`` AFTER the
    DataLoader collate, so a batch mixing 3- and 4-channel datasets fails to
    collate ("Trying to resize storage that is not resizable"). Converting here —
    before collation — lets differently-generated datasets (e.g. odaiba 4ch +
    mini 3ch) be combined. Padded (all-zero) rows stay all-zero so downstream
    masking is unchanged; already-4-channel data is returned untouched. The
    result matches what train_epoch would have produced for 3-channel data, so
    single-dataset behavior is preserved.
    """
    if nf is None:
        return nf
    nf = np.asarray(nf, dtype=np.float32)
    if nf.shape[-1] != 3:
        return nf
    valid = np.any(nf != 0, axis=-1, keepdims=True)
    cs = np.concatenate([nf[..., :2], np.cos(nf[..., 2:3]), np.sin(nf[..., 2:3])], axis=-1)
    return np.where(valid, cs, 0.0).astype(np.float32)


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
