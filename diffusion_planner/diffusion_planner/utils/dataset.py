import numpy as np
from torch.utils.data import Dataset, Sampler

from diffusion_planner.dimensions import (
    TRAFFIC_LIGHT,
    TRAFFIC_LIGHT_NO_TRAFFIC_LIGHT,
    TRAFFIC_LIGHT_ONE_HOT_DIM,
)
from diffusion_planner.utils.train_utils import openjson


def align_legacy_neighbor_futures_on_load(data: dict, atol: float = 1e-4) -> None:
    """Fix legacy short-track timing in memory without writing the shared NPZ."""
    if "neighbor_agents_future" not in data or "neighbor_agents_past" not in data:
        return
    future = data["neighbor_agents_future"]
    past = data["neighbor_agents_past"]
    if future.ndim != 3 or past.ndim != 3 or future.shape[0] != past.shape[0]:
        return
    if future.shape[1] < 2:
        return

    valid = np.any(future != 0, axis=-1)
    valid_count = valid.sum(axis=-1)
    short_track = (valid_count > 0) & (valid_count < future.shape[1])
    current_valid = np.any(past[:, -1, :8] != 0, axis=-1)
    first_matches_current = np.max(np.abs(future[:, 0, :2] - past[:, -1, :2]), axis=-1) <= atol
    needs_shift = short_track & current_valid & valid[:, 0] & first_matches_current
    if not np.any(needs_shift):
        return

    aligned = future.copy()
    aligned[needs_shift, :-1] = future[needs_shift, 1:]
    aligned[needs_shift, -1] = 0
    data["neighbor_agents_future"] = aligned


class DiffusionPlannerData(Dataset):
    def __init__(
        self,
        data_list,
        align_legacy_neighbor_futures: bool = False,
        extra_data_list=None,
        extra_data_repeat: int = 0,
        extra_data_mask_traffic_lights: bool = False,
        include_neighbor_futures: bool = True,
    ):
        self.data_list = openjson(data_list)
        base_data_count = len(self.data_list)
        self._source_index_stride = 1
        self._traffic_light_mask_start = None
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
            if extra_data_mask_traffic_lights:
                self._traffic_light_mask_start = base_data_count
        else:
            extra_paths = []
        self.align_legacy_neighbor_futures = align_legacy_neighbor_futures
        self.include_neighbor_futures = include_neighbor_futures

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        path = self.data_list[idx]
        with np.load(path, allow_pickle=False) as archive:
            data = {
                key: archive[key]
                for key in archive.files
                if key != "version"
                and (self.include_neighbor_futures or key != "neighbor_agents_future")
            }
        normalized_idx = idx if idx >= 0 else len(self.data_list) + idx
        source_idx = normalized_idx * self._source_index_stride
        if (
            self._traffic_light_mask_start is not None
            and source_idx >= self._traffic_light_mask_start
        ):
            self._mask_traffic_lights(data)
        if self.align_legacy_neighbor_futures:
            align_legacy_neighbor_futures_on_load(data)
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
        self._source_index_stride *= step

    @staticmethod
    def _mask_traffic_lights(data: dict) -> None:
        """Hide lane traffic-light state for selected right-turn samples in worker memory."""
        for key in ("lanes", "route_lanes"):
            if key not in data:
                continue
            lanes = data[key].copy()
            valid = np.any(np.abs(lanes[..., :TRAFFIC_LIGHT]) > 0, axis=-1)
            lanes[..., TRAFFIC_LIGHT : TRAFFIC_LIGHT + TRAFFIC_LIGHT_ONE_HOT_DIM] = 0.0
            lanes[..., TRAFFIC_LIGHT_NO_TRAFFIC_LIGHT] = valid.astype(lanes.dtype)
            data[key] = lanes


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
