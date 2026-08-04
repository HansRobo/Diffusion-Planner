"""Parquet frame-index loading independent of the Streamlit UI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from numpy.typing import NDArray

REQUIRED_COLUMNS = frozenset({"bag_path", "map_path", "frame_time_ns"})
STAT_COLUMNS = (
    "ego_speed_mps",
    "ego_yaw_rate_rps",
    "turn_indicator",
    "num_objects",
    "traffic_signal_fresh",
)


@dataclass(frozen=True)
class FrameIndexRow:
    """One resolved row in a frame-index Parquet file."""

    index: int
    bag_path: str
    map_path: str
    frame_time_ns: int
    stats: dict[str, object]


@dataclass(frozen=True)
class FrameIndex:
    """In-memory columns needed to browse a frame-index Parquet file."""

    path: Path
    bag_paths: NDArray[np.str_]
    map_paths: NDArray[np.str_]
    frame_times_ns: NDArray[np.int64]
    stats: dict[str, NDArray[np.generic]]

    def __len__(self) -> int:
        return len(self.frame_times_ns)

    @property
    def bags(self) -> tuple[str, ...]:
        """Return unique bag paths in first-occurrence order."""
        return tuple(dict.fromkeys(self.bag_paths.tolist()))

    def indices_for_bag(self, bag_path: str | None) -> NDArray[np.int64]:
        """Return absolute row indices, optionally filtered by bag path."""
        if bag_path is None:
            return np.arange(len(self), dtype=np.int64)
        return np.flatnonzero(self.bag_paths == bag_path).astype(np.int64, copy=False)

    def row(self, index: int) -> FrameIndexRow:
        """Return one row by its absolute index."""
        if not 0 <= index < len(self):
            raise IndexError(f"Frame index {index} is out of range for {len(self)} rows")
        return FrameIndexRow(
            index=index,
            bag_path=str(self.bag_paths[index]),
            map_path=str(self.map_paths[index]),
            frame_time_ns=int(self.frame_times_ns[index]),
            stats={key: values[index].item() for key, values in self.stats.items()},
        )


def load_frame_index(path: str | Path) -> FrameIndex:
    """Load and validate an index created by ``scripts/create_parquet.py``."""
    parquet_path = Path(path).expanduser().resolve()
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    parquet_file = pq.ParquetFile(parquet_path)
    column_names = set(parquet_file.schema_arrow.names)
    missing = sorted(REQUIRED_COLUMNS.difference(column_names))
    if missing:
        raise ValueError(f"Missing required Parquet columns: {', '.join(missing)}")

    selected_columns = [*sorted(REQUIRED_COLUMNS), *(c for c in STAT_COLUMNS if c in column_names)]
    table = parquet_file.read(columns=selected_columns)
    if table.num_rows == 0:
        raise ValueError(f"Parquet frame index is empty: {parquet_path}")

    def column(name: str) -> NDArray[np.generic]:
        return np.asarray(table[name].combine_chunks().to_numpy(zero_copy_only=False))

    return FrameIndex(
        path=parquet_path,
        bag_paths=column("bag_path").astype(np.str_),
        map_paths=column("map_path").astype(np.str_),
        frame_times_ns=column("frame_time_ns").astype(np.int64, copy=False),
        stats={name: column(name) for name in STAT_COLUMNS if name in table.column_names},
    )
