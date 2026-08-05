"""Create the frame index of a rosbags_from_label download as one parquet file."""

from __future__ import annotations

import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import diffusion_planner_data_tools as dpt
import hydra
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import DictConfig
from tqdm import tqdm

from diffusion_planner.data import VEHICLE_COLUMNS, VehicleParameters, build_vehicles

SPLITS = ("train", "valid", "auto")


def scan_bag(bag_path: Path, map_path: Path, frame_interval_s: float) -> pa.Table:
    """Scan one rosbag and return its frame index as a pyarrow Table."""
    index = dpt.scan_bag_index(str(bag_path), frame_interval_s=frame_interval_s)
    num_frames = len(index["frame_time_ns"])
    inconsistent_columns = [name for name, values in index.items() if len(values) != num_frames]
    if inconsistent_columns:
        raise RuntimeError(
            f"scan returned inconsistent column lengths for {bag_path}: "
            f"{', '.join(inconsistent_columns)}"
        )
    frame_times = index["frame_time_ns"]
    if num_frames > 1 and not (frame_times[1:] > frame_times[:-1]).all():
        raise RuntimeError(f"scan returned non-increasing or duplicate frame times for {bag_path}")
    return pa.table(
        {
            "bag_path": pa.array([str(bag_path)] * num_frames, pa.string()),
            "map_path": pa.array([str(map_path)] * num_frames, pa.string()),
            **index,
        }
    )


def discover_bags(root: Path, split: str) -> list[dict[str, str]]:
    """Collect every rosbag of one split under root with the map it was recorded on.

    Every bag directory is identified by its log_file_info.json rather than by a fixed
    directory depth, so root may be the whole download directory, a single project, a single
    dataset, a single split, a single date, or one bag directory.
    """
    entries = []
    for info_path in sorted(root.rglob("log_file_info.json")):
        bag_path = info_path.parent
        if not (bag_path / "metadata.yaml").is_file():
            continue
        # Layout: <project>/<dataset>/<split>/<date>/<time>, so the split and the dataset
        # are read off the path regardless of where root sits in that hierarchy.
        if bag_path.parents[1].name != split:
            continue
        info = json.loads(info_path.read_text())
        # The map version differs from bag to bag even inside one dataset, so it must be
        # resolved per bag instead of being passed in once on the command line.
        map_version = info["area_map_version_id"]
        dataset_dir = bag_path.parents[2]
        map_path = dataset_dir / "map" / map_version / "lanelet2_map.osm"
        if not map_path.is_file():
            raise FileNotFoundError(f"map not found for {bag_path}: {map_path}")
        entries.append(
            {
                "bag_path": str(bag_path),
                "map_path": str(map_path),
                "project_id": info["project_id"],
                "area_map_id": info["area_map_id"],
                "area_map_version_id": map_version,
                "split": split,
            }
        )
    return entries


def scan_entry(
    entry: dict[str, str], frame_interval_s: float, vehicle: VehicleParameters
) -> pa.Table | None:
    """Scan one bag and tag its frames with the dataset metadata; None if it fails."""
    try:
        table = scan_bag(Path(entry["bag_path"]), Path(entry["map_path"]), frame_interval_s)
    except Exception as error:  # a single unreadable bag must not abort the whole run
        print(f"{entry['bag_path']}: skipped ({error})", file=sys.stderr)
        return None
    num_rows = table.num_rows
    for name in ("project_id", "area_map_id", "area_map_version_id", "split"):
        table = table.append_column(name, pa.array([entry[name]] * num_rows, pa.string()))
    # The ego dimensions are baked in so that the index alone is enough to train from.
    for name in VEHICLE_COLUMNS:
        value = getattr(vehicle, name)
        table = table.append_column(name, pa.array([value] * num_rows, pa.float64()))
    return table


def _worker(packed: tuple[dict[str, str], float, VehicleParameters]) -> pa.Table | None:
    entry, frame_interval_s, vehicle = packed
    return scan_entry(entry, frame_interval_s, vehicle)


@hydra.main(
    version_base=None, config_path="../../configs", config_name="dataset/create_dataset_index"
)
def main(config: DictConfig) -> None:
    """Scan every bag of one split and write their frames to a single parquet file."""
    if config.split not in SPLITS:
        raise ValueError(f"split must be one of {', '.join(SPLITS)}: {config.split}")
    if not math.isfinite(config.frame_interval) or config.frame_interval <= 0.0:
        raise ValueError(f"frame_interval must be finite and positive: {config.frame_interval}")
    if config.jobs < 1:
        raise ValueError(f"jobs must be at least 1: {config.jobs}")

    root = Path(config.root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")
    output = Path(config.output).expanduser().resolve()
    vehicles = build_vehicles(config.vehicles)

    entries = discover_bags(root, config.split)
    if config.limit is not None:
        entries = entries[: config.limit]
    if not entries:
        raise FileNotFoundError(
            f"no {config.split} rosbags found under {root} (set split= to pick another split)"
        )
    # Failing here costs seconds; failing after a day-long scan does not.
    unknown = sorted({entry["project_id"] for entry in entries} - set(vehicles))
    if unknown:
        raise ValueError(f"no vehicle configured for project(s): {', '.join(unknown)}")
    print(f"found {len(entries)} {config.split} rosbags")

    output.parent.mkdir(parents=True, exist_ok=True)

    tables = []
    failed = 0
    packed = [(entry, config.frame_interval, vehicles[entry["project_id"]]) for entry in entries]
    with ProcessPoolExecutor(max_workers=config.jobs) as executor:
        progress = tqdm(executor.map(_worker, packed), total=len(packed), unit="bag", smoothing=0.0)
        for table in progress:
            if table is None:
                failed += 1
                continue
            tables.append(table)
            progress.set_postfix(frames=sum(t.num_rows for t in tables), failed=failed)

    if not tables:
        raise RuntimeError("no bag produced any valid frame")

    combined = pa.concat_tables(tables)
    pq.write_table(combined, output)
    print(
        f"wrote {output}: {combined.num_rows} valid frames "
        f"from {len(tables)} bag(s), {failed} skipped"
    )


if __name__ == "__main__":
    main()
