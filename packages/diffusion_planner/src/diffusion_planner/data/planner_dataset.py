"""Torch dataset turning a frame-index Parquet file into diffusion planner samples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

VEHICLE_COLUMNS = ("base_link_to_front", "vehicle_length", "vehicle_width")
REQUIRED_INDEX_COLUMNS = frozenset({"bag_path", "map_path", "frame_time_ns", *VEHICLE_COLUMNS})


def _data_tools() -> Any:
    """Import the native preprocessing module with an actionable error message."""
    try:
        import diffusion_planner_data_tools as dpt
    except ImportError as error:
        raise RuntimeError(
            "diffusion_planner_data_tools is unavailable. Build the ROS 2 workspace and "
            "source ros2_ws/install/setup.bash before training."
        ) from error
    return dpt


@dataclass(frozen=True)
class VehicleParameters:
    """Primitive parameters accepted by ``diffusion_planner_data_tools.VehicleSpec``."""

    base_link_to_front: float
    vehicle_length: float
    vehicle_width: float


class PlannerDataset(Dataset):
    """Build model inputs and labels on the fly from a frame-index Parquet file.

    The Parquet file only stores which frame of which bag is usable; the tensors themselves
    are produced per item by the same C++ preprocessing that runs at inference time. The
    native bag/map caches are not picklable and are therefore created lazily inside each
    worker process, so every worker keeps its own readers warm.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        *,
        reader_capacity: int = 4,
        map_capacity: int = 2,
        traffic_light_timeout_s: float = 0.2,
        neighbor_observation_timeout_s: float = 0.3,
    ) -> None:
        """Load the frame index and remember how frames should be built.

        Args:
            parquet_path: Frame index written by ``create_parquet_rosbag_from_label.py``, which
                writes one file per split and stamps the ego dimensions into every row.
            reader_capacity: Bags kept open per worker; one worker reads few bags at a time.
            map_capacity: Parsed lanelet2 maps kept per worker; maps are expensive to hold.
            traffic_light_timeout_s: Age after which a traffic signal is treated as unknown.
            neighbor_observation_timeout_s: Age after which a tracked object is dropped.
        """
        self._path = Path(parquet_path).expanduser().resolve()
        if not self._path.is_file():
            raise FileNotFoundError(f"Parquet file not found: {self._path}")

        table = pq.read_table(self._path)
        missing = sorted(REQUIRED_INDEX_COLUMNS.difference(table.column_names))
        if missing:
            raise ValueError(f"Missing required Parquet columns: {', '.join(missing)}")

        if table.num_rows == 0:
            raise ValueError(f"Frame index is empty: {self._path}")

        self._bag_paths = _column(table, "bag_path")
        self._map_paths = _column(table, "map_path")
        self._frame_times_ns = _column(table, "frame_time_ns").astype(np.int64, copy=False)

        # Ego dimensions travel with the index, so training needs no separate vehicle config.
        self._vehicles = np.stack(
            [_column(table, name).astype(np.float64, copy=False) for name in VEHICLE_COLUMNS],
            axis=1,
        )

        self._reader_capacity = reader_capacity
        self._map_capacity = map_capacity
        self._traffic_light_timeout_s = traffic_light_timeout_s
        self._neighbor_observation_timeout_s = neighbor_observation_timeout_s
        self._cache: Any = None
        self._specs: dict[tuple[float, float, float], Any] = {}

    def __len__(self) -> int:
        return len(self._frame_times_ns)

    def source(self, index: int) -> tuple[str, int]:
        """Return the bag path and frame time behind one item, for error reporting."""
        return str(self._bag_paths[index]), int(self._frame_times_ns[index])

    def vehicle(self, index: int) -> VehicleParameters:
        """Return the ego dimensions stamped into one row of the index."""
        return VehicleParameters(*self._vehicles[index].tolist())

    def __getitem__(self, index: int) -> dict[str, torch.Tensor] | None:
        """Return one frame as tensors, or None when the frame cannot be built."""
        cache, spec = self._cache_and_spec(tuple(self._vehicles[index].tolist()))
        frame = cache.create_frame_data(
            bag_path=str(self._bag_paths[index]),
            map_path=str(self._map_paths[index]),
            frame_time_ns=int(self._frame_times_ns[index]),
            vehicle_spec=spec,
            traffic_light_timeout_s=self._traffic_light_timeout_s,
            neighbor_observation_timeout_s=self._neighbor_observation_timeout_s,
        )
        if frame is None:
            return None
        return {key: torch.from_numpy(value) for key, value in frame.items()}

    def _cache_and_spec(self, vehicle: tuple[float, float, float]) -> tuple[Any, Any]:
        """Return this process' frame cache and this row's vehicle spec, made on demand.

        An index holds a handful of distinct vehicles at most, so the specs are cached by
        their dimensions instead of being rebuilt for every frame.
        """
        dpt = _data_tools()
        if self._cache is None:
            self._cache = dpt.FrameDataCache(
                reader_capacity=self._reader_capacity,
                map_capacity=self._map_capacity,
            )
        spec = self._specs.get(vehicle)
        if spec is None:
            base_link_to_front, vehicle_length, vehicle_width = vehicle
            spec = dpt.VehicleSpec(
                base_link_to_front=base_link_to_front,
                vehicle_length=vehicle_length,
                vehicle_width=vehicle_width,
            )
            self._specs[vehicle] = spec
        return self._cache, spec

    def __getstate__(self) -> dict[str, Any]:
        """Drop the native handles so the dataset can be sent to worker processes."""
        return {**self.__dict__, "_cache": None, "_specs": {}}


def _column(table: Any, name: str, purpose: str = "this dataset") -> NDArray[Any]:
    """Return one Parquet column as a numpy array."""
    if name not in table.column_names:
        raise ValueError(f"Parquet file has no '{name}' column, which {purpose} requires")
    return np.asarray(table[name].combine_chunks().to_numpy(zero_copy_only=False))


def collate_frames(batch: list[dict[str, torch.Tensor] | None]) -> dict[str, torch.Tensor] | None:
    """Stack frames into a batch, dropping the ones that could not be built.

    A frame is dropped when the bag turns out to be unreadable at that timestamp, which the
    index scan cannot rule out. The batch is therefore occasionally smaller than requested,
    and None when every frame in it failed.
    """
    frames = [frame for frame in batch if frame is not None]
    if not frames:
        return None
    return {key: torch.stack([frame[key] for frame in frames]) for key in frames[0]}


def build_dataloader(
    dataset: PlannerDataset,
    *,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 8,
    **kwargs: Any,
) -> DataLoader:
    """Wrap the dataset in a DataLoader with the frame-aware collate function."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_frames,
        persistent_workers=num_workers > 0,
        **kwargs,
    )
