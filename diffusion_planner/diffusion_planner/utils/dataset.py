import os

import numpy as np
from torch.utils.data import Dataset

from diffusion_planner.utils.render_bev import VIEW_EXTENTS_M, render_sample
from diffusion_planner.utils.train_utils import openjson
from planner_metrics.temporal_stability import consecutive_frame_pairs


def bev_render_settings(config):
    """Return ``(render_bev_image, bev_image_size)`` for a model config.

    Configs saved before image input existed carry neither field, so they fall back to the
    vector pipeline instead of failing to load.
    """
    input_type = getattr(config, "input_type", "vector")
    return input_type == "image", getattr(config, "bev_image_size", 0)


class DiffusionPlannerData(Dataset):
    """Samples of the vector scene, optionally with the BEV rasters rendered alongside.

    Rasterisation runs here, in the DataLoader worker, so it parallelises with the training
    step -- but that also places it before the on-GPU augmentation, which is why image mode
    requires augmentation to be disabled.
    """

    def __init__(self, data_list, render_bev_image, bev_image_size):
        if isinstance(data_list, (str, bytes, os.PathLike)):
            self.data_list = openjson(data_list)
        else:
            self.data_list = list(data_list)
        self.render_bev_image = render_bev_image
        self.bev_image_size = bev_image_size

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx], allow_pickle=True)
        data = dict(data)  # npz to dict
        data.pop("version", None)
        if self.render_bev_image:
            data["bev_image"] = render_sample(data, self.bev_image_size, VIEW_EXTENTS_M)
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
