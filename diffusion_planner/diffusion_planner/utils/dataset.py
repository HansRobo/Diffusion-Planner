import numpy as np
from torch.utils.data import Dataset, DistributedSampler, Sampler

from diffusion_planner.utils.legacy_neighbor_alignment import (
    align_legacy_neighbor_futures_on_load,
)
from diffusion_planner.utils.train_utils import openjson


class DiffusionPlannerData(Dataset):
    def __init__(
        self,
        data_list,
        align_legacy_neighbor_futures: bool = False,
        extra_data_list=None,
        extra_data_repeat: int = 0,
        include_neighbor_futures: bool = True,
    ):
        self.data_list = openjson(data_list)
        if extra_data_repeat < 0:
            raise ValueError("extra_data_repeat must be >= 0")
        if extra_data_repeat > 0:
            if not extra_data_list:
                raise ValueError("extra_data_list is required when extra_data_repeat > 0")
            extra_lists = [extra_data_list] if isinstance(extra_data_list, str) else extra_data_list
            extra_paths = []
            for list_path in extra_lists:
                extra_paths.extend(openjson(list_path))
            # List multiplication copies references, not path strings or NPZ contents. This keeps
            # weighting in memory and avoids writing a multi-hundred-MB combined JSON manifest.
            self.data_list.extend(extra_paths * extra_data_repeat)
        self.align_legacy_neighbor_futures = align_legacy_neighbor_futures
        self.include_neighbor_futures = include_neighbor_futures

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        path = self.data_list[idx]
        align_neighbor_futures = (
            self.align_legacy_neighbor_futures and self.include_neighbor_futures
        )
        with np.load(path, allow_pickle=False) as archive:
            data_version = (
                int(np.asarray(archive["version"]).reshape(-1)[0])
                if align_neighbor_futures and "version" in archive.files and archive["version"].size
                else None
            )
            data = {
                key: archive[key]
                for key in archive.files
                if key != "version"
                and (self.include_neighbor_futures or key != "neighbor_agents_future")
            }
        if "ego_shape" in data and data["ego_shape"].shape != (3,):
            raise ValueError(
                f"{path}: ego_shape with shape {data['ego_shape'].shape}, expected (3,)"
            )
        if align_neighbor_futures:
            align_legacy_neighbor_futures_on_load(
                data,
                source_path=path,
                data_version=data_version,
            )
        for key, value in data.items():
            if (
                isinstance(value, np.ndarray)
                and np.issubdtype(value.dtype, np.unsignedinteger)
                and value.dtype != np.uint8
            ):
                data[key] = value.astype(np.int64, copy=False)
        return data

    def subsample(self, step: int) -> None:
        if step < 1:
            raise ValueError("subsample step must be >= 1")
        self.data_list = self.data_list[::step]


class DistributedEvalSampler(Sampler[int]):
    """Shard evaluation indices without DistributedSampler's duplicate padding."""

    def __init__(self, dataset: Dataset, num_replicas: int, rank: int):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        remaining = len(self.dataset) - self.rank
        return max(0, (remaining + self.num_replicas - 1) // self.num_replicas)


class BatchAlignedDistributedSampler(DistributedSampler):
    """Pad a shuffled distributed epoch to complete global batches.

    ``DistributedSampler`` only aligns the number of samples across ranks. If that
    per-rank count is not divisible by the local batch size, ``drop_last=True``
    silently discards samples and ``drop_last=False`` creates a second compiled
    shape. This sampler adds the minimum shuffled-prefix padding needed for every
    rank to receive complete local batches. Every source index is therefore used
    at least once, while at most one global batch minus one is repeated.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int,
        rank: int,
        local_batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
    ):
        if local_batch_size <= 0:
            raise ValueError("local_batch_size must be positive")
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )
        global_batch_size = num_replicas * local_batch_size
        global_batches = (len(dataset) + global_batch_size - 1) // global_batch_size
        self.num_samples = global_batches * local_batch_size
        self.total_size = self.num_samples * num_replicas
        self.padding_size = self.total_size - len(dataset)
