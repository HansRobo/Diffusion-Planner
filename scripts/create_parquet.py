from __future__ import annotations

import argparse
from pathlib import Path

import diffusion_planner_data_tools as dpt
import pyarrow as pa
import pyarrow.parquet as pq


def scan_bag(bag_path: Path, map_path: Path, frame_interval_s: float) -> pa.Table:
    """Scan one rosbag and return its frame index as a pyarrow Table."""
    index = dpt.scan_bag_index(str(bag_path), frame_interval_s=frame_interval_s)
    num_frames = len(index["frame_time_ns"])
    return pa.table(
        {
            "bag_path": pa.array([str(bag_path)] * num_frames, pa.string()),
            "map_path": pa.array([str(map_path)] * num_frames, pa.string()),
            **index,
        }
    )


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Create a parquet file containing the frame index of one or more rosbags"
    )
    parser.add_argument("bags", nargs="+", type=Path, help="rosbag directories")
    parser.add_argument(
        "--map", type=Path, required=True, help="lanelet2 map (.osm) used by all given bags"
    )
    parser.add_argument("--output", type=Path, required=True, help="output parquet path")
    parser.add_argument(
        "--frame-interval",
        type=float,
        default=0.1,
        help="spacing between generated frames [s]; changes only the index sample density, "
        "not the model's internal 0.1 s grid (default: %(default)s)",
    )
    args = parser.parse_args()

    map_path = args.map.resolve()
    if not map_path.is_file():
        parser.error(f"map file not found: {map_path}")
    for bag_path in args.bags:
        if not (bag_path / "metadata.yaml").is_file():
            parser.error(f"not a rosbag directory (no metadata.yaml): {bag_path}")

    tables = []
    for bag_path in args.bags:
        table = scan_bag(bag_path.resolve(), map_path, args.frame_interval)
        print(f"{bag_path}: {table.num_rows} valid frames")
        tables.append(table)

    combined = pa.concat_tables(tables)
    pq.write_table(combined, args.output)
    print(f"wrote {args.output}: {combined.num_rows} valid frames from {len(args.bags)} bag(s)")


if __name__ == "__main__":
    main()
