from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import argcomplete
import pyarrow as pa
import pyarrow.parquet as pq
from create_parquet import positive_finite_float, scan_bag  # noqa: E402
from tqdm import tqdm

SPLITS = ("train", "valid", "auto")


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


def scan_entry(entry: dict[str, str], frame_interval_s: float) -> pa.Table | None:
    """Scan one bag and tag its frames with the dataset metadata; None if it fails."""
    try:
        table = scan_bag(Path(entry["bag_path"]), Path(entry["map_path"]), frame_interval_s)
    except Exception as error:  # a single unreadable bag must not abort the whole run
        print(f"{entry['bag_path']}: skipped ({error})", file=sys.stderr)
        return None
    num_rows = table.num_rows
    for name in ("project_id", "area_map_id", "area_map_version_id", "split"):
        table = table.append_column(name, pa.array([entry[name]] * num_rows, pa.string()))
    return table


def _worker(packed: tuple[dict[str, str], float]) -> pa.Table | None:
    entry, frame_interval_s = packed
    return scan_entry(entry, frame_interval_s)


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Create one parquet file with the frame index of every rosbag of one "
        "split under a rosbags_from_label style directory"
    )
    parser.add_argument("root", type=Path, help="rosbags_from_label root directory")
    parser.add_argument("--output", type=Path, required=True, help="output parquet path")
    parser.add_argument(
        "--split",
        choices=SPLITS,
        default="train",
        help="split directory to scan (default: %(default)s)",
    )
    parser.add_argument(
        "--frame-interval",
        type=positive_finite_float,
        default=0.1,
        help="spacing between generated frames [s] (default: %(default)s)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=32,
        help="number of bags scanned in parallel; the scan is disk bound (default: %(default)s)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="scan only the first N bags (for a trial run)"
    )
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"not a directory: {root}")

    entries = discover_bags(root, args.split)
    if args.limit is not None:
        entries = entries[: args.limit]
    if not entries:
        parser.error(
            f"no {args.split} rosbags found under {root} (use --split to pick another split)"
        )
    print(f"found {len(entries)} {args.split} rosbags")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    tables = []
    failed = 0
    packed = [(entry, args.frame_interval) for entry in entries]
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
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
    pq.write_table(combined, args.output)
    print(
        f"wrote {args.output}: {combined.num_rows} valid frames "
        f"from {len(tables)} bag(s), {failed} skipped"
    )


if __name__ == "__main__":
    main()
