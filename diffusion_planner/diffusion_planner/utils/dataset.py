import os

import numpy as np
import torch
from torch.utils.data import Dataset

from diffusion_planner.utils.render_bev import VIEW_EXTENTS_M, render_sample
from diffusion_planner.utils.train_utils import heading_to_cos_sin, openjson
from planner_metrics.temporal_stability import consecutive_frame_pairs


def bev_render_settings(config):
    """Return ``(render_bev_image, bev_image_size)`` for a model config.

    Configs saved before image input existed carry neither field, so they fall back to the
    vector pipeline instead of failing to load.
    """
    input_type = getattr(config, "input_type", "vector")
    return input_type == "image", getattr(config, "bev_image_size", 0)


def worker_init(worker_id: int):
    """DataLoader ``worker_init_fn``: keep each worker single-threaded.

    Perturbation and rasterisation both run in the workers now, so ``num_workers`` processes
    each doing intra-op threading would oversubscribe the machine; the tensors involved are far
    too small for the extra threads to pay for themselves anyway.
    """
    torch.set_num_threads(1)


def augment_sample(data: dict, aug) -> dict:
    """Perturb one sample's ego pose and re-express the whole scene around it.

    ``aug`` is a ``StatePerturbation`` (either flavour) built on the CPU device.  It works on
    batched tensors, so the sample is given a batch dimension of one for the call and stripped
    of it again afterwards.  The ego past and the goal pose are converted to cos/sin form here
    because the perturbation expects that layout; the conversion is idempotent, so the training
    loop repeating it later is a no-op.
    """
    batch = {key: torch.from_numpy(np.asarray(value)).unsqueeze(0) for key, value in data.items()}
    batch["ego_agent_past"] = heading_to_cos_sin(batch["ego_agent_past"])
    batch["goal_pose"] = heading_to_cos_sin(batch["goal_pose"])

    batch, ego_future, neighbors_future = aug(
        batch, batch["ego_agent_future"], batch["neighbor_agents_future"]
    )
    batch["ego_agent_future"] = ego_future
    batch["neighbor_agents_future"] = neighbors_future

    return {key: value.squeeze(0).numpy() for key, value in batch.items()}


class DiffusionPlannerData(Dataset):
    """Samples of the vector scene, optionally perturbed and rasterised into BEV images.

    Both steps run here, in the DataLoader worker, so they parallelise with the training step --
    and, more importantly, in the right order: the perturbation rewrites the ego frame, so the
    rasters have to be drawn after it or they would describe the recorded ego pose instead of
    the perturbed one.  That is why augmentation moved out of the training loop and in here.

    ``aug`` is the perturbation (a CPU-device ``StatePerturbation``), or None to hand the
    recorded scene through untouched -- validation and every offline consumer pass None.
    """

    def __init__(self, data_list, render_bev_image, bev_image_size, aug):
        if isinstance(data_list, (str, bytes, os.PathLike)):
            self.data_list = openjson(data_list)
        else:
            self.data_list = list(data_list)
        self.render_bev_image = render_bev_image
        self.bev_image_size = bev_image_size
        self.aug = aug

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx], allow_pickle=True)
        data = dict(data)  # npz to dict
        data.pop("version", None)
        if self.aug is not None:
            data = augment_sample(data, self.aug)
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
