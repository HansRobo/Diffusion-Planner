"""Build one current-schema HDF5 frame per matrix entry directly from ROSBAG."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, REPO_ROOT.as_posix())

import ml_planner_data as dpt  # noqa: E402

hdf5plugin.register(filters="zstd")
_WORKER_CACHE: dpt.FrameDataCache | None = None
_BAG_PATHS_BY_NAME: dict[tuple[Path, str], Path] = {}


def samples(matrix: Path) -> list[tuple[str, Path]]:
    data = json.loads(matrix.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [("frames", Path(path).resolve()) for path in data]
    return [(metric, Path(path).resolve()) for metric, paths in data.items() for path in paths]


def sidecar(npz: Path) -> dict[str, object]:
    return json.loads(npz.with_suffix(".json").read_text(encoding="utf-8"))


def resolve_bag(npz: Path, dataset_root_name: str, rosbag_root: Path) -> Path:
    parts = npz.parts
    marker = parts.index(dataset_root_name)
    bag = rosbag_root / Path(*parts[marker + 1 :]).parent.parent
    if (bag / "log_file_info.json").is_file():
        return bag

    # Some portable packages retain a project-id level that is absent from the
    # evaluator NPZ hierarchy. Bag directory names are UUID-derived and unique.
    key = (rosbag_root, bag.name)
    if key not in _BAG_PATHS_BY_NAME:
        matches = [
            path.parent
            for path in rosbag_root.rglob("log_file_info.json")
            if path.parent.name == bag.name
        ]
        if len(matches) != 1:
            raise FileNotFoundError(f"ROSBAG metadata not uniquely resolved for {npz}: {matches}")
        _BAG_PATHS_BY_NAME[key] = matches[0]
    return _BAG_PATHS_BY_NAME[key]


def resolve_map(bag: Path, map_version_id: str) -> Path:
    """Find the packaged map without assuming one fixed bag hierarchy depth."""
    for parent in bag.parents:
        candidate = parent / "map" / map_version_id / "lanelet2_map.osm"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"map {map_version_id} not found in any ancestor of ROSBAG: {bag}")


def chunk_paths(bag: Path) -> dict[int, Path]:
    result = {}
    for path in (*bag.glob("*.db3"), *bag.glob("*.mcap")):
        match = re.search(r"_(\d+)\.(?:db3|mcap)$", path.name)
        if match:
            result[int(match.group(1))] = path
    return result


def route_chunk_indices(chunks: dict[int, Path]) -> set[int]:
    """Find chunks containing a serialized route message without decoding it."""
    result = set()
    for index, path in chunks.items():
        if path.suffix != ".db3":
            continue
        with sqlite3.connect(path) as database:
            found = database.execute(
                """SELECT 1 FROM topics AS t JOIN messages AS m ON m.topic_id = t.id
                   WHERE t.name = '/planning/mission_planning/route' LIMIT 1"""
            ).fetchone()
        if found is not None:
            result.add(index)
    return result


def start_timestamp_ns(bag: Path) -> int:
    info = json.loads((bag / "log_file_info.json").read_text(encoding="utf-8"))
    if "start_timestamp" in info:
        value = str(info["start_timestamp"])
        return int(
            datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
            .replace(tzinfo=timezone.utc)
            .timestamp()
            * 1e9
        )
    metadata = yaml.safe_load((bag / "metadata.yaml").read_text(encoding="utf-8"))
    return int(metadata["rosbag2_bagfile_information"]["starting_time"]["nanoseconds_since_epoch"])


def stage_frame_bag(bag: Path, timestamp: int, root: Path) -> Path:
    """Stage route chunk plus only chunks needed around one requested frame."""
    chunks = chunk_paths(bag)
    target_index = int((timestamp - start_timestamp_ns(bag)) // 60_000_000_000)
    needed_indices = {target_index - 1, target_index, target_index + 1}
    needed_indices.update(route_chunk_indices(chunks))
    selected = [chunks[index] for index in sorted(needed_indices) if index in chunks]
    if target_index not in chunks:
        raise FileNotFoundError(f"target chunk {target_index} not found in {bag}")

    staged = root / bag.name
    staged.mkdir(parents=True, exist_ok=True)
    (staged / "log_file_info.json").symlink_to(bag / "log_file_info.json")
    metadata_source = (bag / "metadata.yaml").read_text(encoding="utf-8")
    before, after = metadata_source.split("  relative_file_paths:\n", 1)
    _, files_section = after.split("  files:", 1)
    metadata = before + "  relative_file_paths:\n"
    metadata += "".join(f"  - {path.name}\n" for path in selected)
    (staged / "metadata.yaml").write_text(metadata + "  files:" + files_section, encoding="utf-8")
    for path in selected:
        (staged / path.name).symlink_to(path)
    return staged


def write_frame(
    output: Path,
    frame: dict[str, np.ndarray],
    npz: Path,
    bag: Path,
    timestamp: int,
    available_future_steps: int,
) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as h5:
        h5.attrs.update(
            format="diffusion_planner_frame_dataset",
            format_version=4,
            source_bag_path=bag.as_posix(),
            source_npz_path=npz.as_posix(),
            num_frames=1,
            frame_interval_s=0.1,
            available_future_steps=available_future_steps,
        )
        frames = h5.create_group("frames")
        for key, value in frame.items():
            array = np.asarray(value, dtype=np.float32)
            frames.create_dataset(
                key,
                data=array[None],
                chunks=(1, *array.shape),
                shuffle=True,
                **hdf5plugin.Zstd(),
            )
        metadata = h5.create_group("metadata")
        speed = float(frame["ego_agent_past"][-1, 4])
        yaw_rate = float(frame["ego_agent_past"][-1, 5])
        turn = int(frame["turn_indicators"][-1])
        num_objects = int(
            np.count_nonzero(np.abs(frame["neighbor_agents_past"]).sum(axis=(-2, -1)))
        )
        metadata.create_dataset("frame_time_ns", data=[timestamp])
        metadata.create_dataset("ego_speed_mps", data=[speed])
        metadata.create_dataset("ego_yaw_rate_rps", data=[yaw_rate])
        metadata.create_dataset("turn_indicator", data=[turn], dtype="u1")
        metadata.create_dataset("num_objects", data=[num_objects], dtype="i4")
    return {
        "h5_path": output.resolve().as_posix(),
        "frame_index": 0,
        "frame_time_ns": timestamp,
        "ego_speed_mps": speed,
        "ego_yaw_rate_rps": yaw_rate,
        "turn_indicator": turn,
        "num_objects": num_objects,
        "available_future_steps": available_future_steps,
        "source_npz_path": npz.as_posix(),
    }


def extend_future(frame: dict[str, np.ndarray], steps: int) -> dict[str, np.ndarray]:
    """Extend a short, valid future horizon to the fixed 80-step H5 schema."""
    if steps == 80:
        return frame
    future_axes = {
        "ego_agent_future": 0,
        "neighbor_agents_future": 1,
        "turn_indicators_future": 0,
        "lane_traffic_light_future": 1,
        "route_traffic_light_future": 1,
    }
    result = dict(frame)
    for key, axis in future_axes.items():
        values = result[key]
        if values.shape[axis] != steps:
            raise ValueError(f"{key} has {values.shape[axis]} future steps, expected {steps}")
        padding = [(0, 0)] * values.ndim
        padding[axis] = (0, 80 - steps)
        result[key] = np.pad(values, padding, mode="edge")
    return result


def init_worker() -> None:
    global _WORKER_CACHE
    _WORKER_CACHE = dpt.FrameDataCache()


def convert_one(
    task: tuple[int, str, str, str, str, str, str, int, float, float],
) -> tuple[int, dict[str, object]]:
    (
        index,
        metric,
        npz_text,
        bag_text,
        staged_text,
        map_text,
        output_text,
        timestamp,
        traffic_timeout,
        neighbor_timeout,
    ) = task
    if _WORKER_CACHE is None:
        raise RuntimeError("worker cache is not initialized")
    frame = None
    available_future_steps = 80
    for steps in (80, 50, 30):
        frame = _WORKER_CACHE.create_frame_data(
            staged_text,
            map_text,
            timestamp,
            dpt.VehicleSpec(5.71111, 7.2369, 2.42741),
            traffic_timeout,
            steps,
            neighbor_timeout,
        )
        if frame is not None:
            available_future_steps = steps
            break
    if frame is None:
        raise RuntimeError(f"ROSBAG frame could not be created: {npz_text}")
    npz = Path(npz_text)
    bag = Path(bag_text)
    output = Path(output_text) / metric / f"{npz.stem}.h5"
    row = write_frame(
        output,
        extend_future(
            {str(key): np.asarray(value) for key, value in frame.items()},
            available_future_steps,
        ),
        npz,
        bag,
        timestamp,
        available_future_steps,
    )
    row["matrix_group"] = metric
    return index, row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("rosbag_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--dataset-root-name", default="20260814_basic_dataset")
    parser.add_argument("--traffic-light-timeout", type=float, default=0.2)
    parser.add_argument("--neighbor-timeout", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()

    selected = samples(args.matrix)
    rosbag_root = args.rosbag_root.resolve()
    output_root = args.output_root.resolve()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    with tempfile.TemporaryDirectory(prefix="matrix_rosbag_frame_") as temporary:
        tasks = []
        for index, (metric, npz) in enumerate(selected):
            info = sidecar(npz)
            timestamp = int(info["timestamp"])
            bag = resolve_bag(npz, args.dataset_root_name, rosbag_root)
            staged = stage_frame_bag(bag, timestamp, Path(temporary) / str(index))
            map_path = resolve_map(bag, str(info["map_version_id"]))
            tasks.append(
                (
                    index,
                    metric,
                    npz.as_posix(),
                    bag.as_posix(),
                    staged.as_posix(),
                    map_path.as_posix(),
                    output_root.as_posix(),
                    timestamp,
                    args.traffic_light_timeout,
                    args.neighbor_timeout,
                )
            )
            print(f"[{index + 1}/{len(selected)}] prepared {npz.name}", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as executor:
            results = list(executor.map(convert_one, tasks))
    rows = []
    for index, row in sorted(results):
        row["matrix_group"] = selected[index][0]
        rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows), output_root / "index.parquet")
    print(f"Wrote {len(rows)} HDF5 frames from ROSBAG data to {output_root}")


if __name__ == "__main__":
    main()
