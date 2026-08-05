"""Create the frame index of a rosbags_from_label download as one parquet file."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import diffusion_planner_data_tools as dpt
import hydra
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import DictConfig
from tqdm import tqdm

from diffusion_planner.data import VEHICLE_COLUMNS, VehicleParameters

SPLITS = ("train", "valid", "auto")


def build_vehicles(config: DictConfig) -> dict[str, VehicleParameters]:
    """Instantiate the project id to vehicle mapping stamped into the index.

    Config keys are not necessarily strings, so they are converted to the project ids that
    the scanned bags report.
    """
    return {str(project): hydra.utils.instantiate(node) for project, node in config.items()}


def build_indexer_param(
    frame_interval_s: float,
    min_travel_distance: float,
    config: Mapping[str, float],
) -> dpt.IndexerParam:
    """Build the native frame-grid and dropout configuration."""
    if not math.isfinite(frame_interval_s) or frame_interval_s <= 0.0:
        raise ValueError(f"frame_interval must be finite and positive: {frame_interval_s}")
    if not math.isfinite(min_travel_distance) or min_travel_distance < 0.0:
        raise ValueError(
            f"min_travel_distance must be finite and non-negative: {min_travel_distance}"
        )

    topic_drop_thresholds = dpt.TopicDropThresholds()
    for topic, limit in config.items():
        name = str(topic)
        if not hasattr(topic_drop_thresholds, name):
            raise ValueError(f"unknown topic in topic_drop_thresholds: {topic}")
        setattr(topic_drop_thresholds, name, float(limit))

    param = dpt.IndexerParam()
    param.time_step_s = frame_interval_s
    param.min_travel_distance = min_travel_distance
    param.topic_drop_thresholds = topic_drop_thresholds
    return param


def scan_bag(
    bag_path: Path,
    map_path: Path,
    indexer_param: dpt.IndexerParam,
) -> tuple[pa.Table, list[str], dict[str, int]]:
    """Scan one rosbag and return its frame index, its dropouts, and its frame counts."""
    index, warnings, stats = dpt.scan_bag_index(str(bag_path), param=indexer_param)
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
    table = pa.table({
        "bag_path": pa.array([str(bag_path)] * num_frames, pa.string()),
        "map_path": pa.array([str(map_path)] * num_frames, pa.string()),
        **index,
    })
    return table, warnings, stats


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
        entries.append({
            "bag_path": str(bag_path),
            "map_path": str(map_path),
            "project_id": info["project_id"],
            "area_map_id": info["area_map_id"],
            "area_map_version_id": map_version,
            "split": split,
        })
    return entries


def scan_entry(
    entry: dict[str, str],
    vehicle: VehicleParameters,
    indexer_param: dpt.IndexerParam,
) -> tuple[pa.Table, list[str], dict[str, int]] | None:
    """Scan one bag and tag its frames with the dataset metadata; None if it fails."""
    try:
        table, warnings, stats = scan_bag(
            Path(entry["bag_path"]),
            Path(entry["map_path"]),
            indexer_param,
        )
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
    return table, warnings, stats


def _worker(
    packed: tuple[dict[str, str], VehicleParameters, float, float, dict[str, float]],
) -> tuple[pa.Table, list[str], dict[str, int]] | None:
    entry, vehicle, frame_interval_s, min_travel_distance, topic_drop_thresholds = packed
    indexer_param = build_indexer_param(
        frame_interval_s, min_travel_distance, topic_drop_thresholds
    )
    return scan_entry(entry, vehicle, indexer_param)


@hydra.main(
    version_base=None, config_path="../../configs", config_name="dataset/create_dataset_index"
)
def main(config: DictConfig) -> None:
    """Scan every bag of one split and write their frames to a single parquet file."""
    if config.split not in SPLITS:
        raise ValueError(f"split must be one of {', '.join(SPLITS)}: {config.split}")
    if config.jobs < 1:
        raise ValueError(f"jobs must be at least 1: {config.jobs}")

    root = Path(config.root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")
    output = Path(config.output).expanduser().resolve()
    vehicles = build_vehicles(config.vehicles)
    frame_interval_s = float(config.frame_interval)
    min_travel_distance = float(config.min_travel_distance)
    topic_drop_thresholds = {
        str(topic): float(limit) for topic, limit in config.topic_drop_thresholds.items()
    }

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
    empty = 0
    skipped = 0
    dropouts = 0
    usable = 0
    warning_log_path = output.parent / datetime.now().strftime("%Y-%m-%d-%H-%M-%S.log")
    packed = [
        (
            entry,
            vehicles[entry["project_id"]],
            frame_interval_s,
            min_travel_distance,
            topic_drop_thresholds,
        )
        for entry in entries
    ]
    with (
        warning_log_path.open("w", encoding="utf-8") as warning_log,
        ProcessPoolExecutor(max_workers=config.jobs) as executor,
    ):
        progress = tqdm(
            zip(entries, executor.map(_worker, packed), strict=True),
            total=len(packed),
            unit="bag",
            smoothing=0.0,
        )
        for entry, result in progress:
            if result is None:
                failed += 1
                continue
            table, warnings, stats = result
            usable += stats["usable_frames"]
            for warning in warnings:
                warning_log.write(f"{entry['bag_path']}: {warning}\n")
                warning_log.flush()
            if stats["skipped"]:
                skipped += 1
                continue
            dropouts += len(warnings)
            if table.num_rows == 0:
                # Every frame was rejected, most often by the dropout thresholds.
                empty += 1
                continue
            tables.append(table)
            progress.set_postfix(frames=sum(t.num_rows for t in tables), empty=empty, failed=failed)

    if not tables:
        raise RuntimeError("no bag produced any valid frame")

    combined = pa.concat_tables(tables)
    pq.write_table(combined, output)
    # `usable` is what the same bags would have yielded without the dropout checks, so the
    # ratio is exactly what the thresholds cost.
    retained = 100.0 * combined.num_rows / usable if usable else 0.0
    print(
        f"wrote {output}: {combined.num_rows} frames from {len(tables)} bag(s), "
        f"{dropouts} dropout(s), {skipped} short-travel bag(s) skipped, "
        f"{empty} bag(s) with no usable frame, {failed} unreadable"
    )
    print(
        f"kept {retained:.1f}% of the {usable} frames these bags could have yielded; "
        f"{usable - combined.num_rows} lost to topic dropouts"
    )
    if dropouts:
        print(f"wrote dropout warnings to {warning_log_path}")


if __name__ == "__main__":
    main()
