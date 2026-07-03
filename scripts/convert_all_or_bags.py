#!/usr/bin/env python3
"""Batch-convert all ORScene override bags on NAS to npz.

For each bag listed in `data/bag_map_versions.json` (built by an earlier task
by reading `/map/vector_map` out of every bag once):

1. Look up (or extract, if not already cached) a lanelet2 `.osm` map for that
   bag's `(area, map_version)` pair. Extraction is delegated to
   `ros_scripts/extract_map_from_bag.py` as a subprocess -- that script
   already handles two non-obvious correctness issues (MGRS coordinate-shift
   for round-tripping a possibly-negative local frame, and a dangling cached
   lanelet member that makes `lanelet2.io.write()` silently drop lanelets).
   Do NOT reimplement map extraction inline here -- see that script's
   docstring for why a naive `lanelet2.io.loadRobust` + `write` does not work.
2. Run `ros_scripts/parse_rosbag.py` as a subprocess against the bag + the
   matched `.osm`, producing one `.npz` (+ companion `.json`) per valid frame
   under `data/or_scene_npz/<area>/<category>/<scene>/`.

Maps are cached on disk keyed by `(area, version)`, so bags sharing a map
version (there are only 5 distinct versions across all 75 bags) reuse the
same extracted `.osm` -- extraction only runs once per version, not once per
bag.

## Environment (IMPORTANT -- do not use `uv run python` for this script)

This script and the subprocesses it launches need ROS2 packages
(`rosbag2_py`, `lanelet2`, `autoware_lanelet2_extension_python`,
`autoware_map_msgs`, ...) that live in the *system* Python, not in this
repo's `uv` venv. Run it with system `python3` after sourcing the ROS2
workspace:

    source /home/chenglin/workspace/at-team-tools/lin/diffusion-planner-risk-scene/external/pilot-auto.x2/install/setup.bash
    cd /home/chenglin/workspace/Diffusion-Planner
    python3 scripts/convert_all_or_bags.py

(No PYTHONPATH needed on the invoking command itself -- this script does not
import `diffusion_planner`/`diffusion_planner_ros` directly. It sets the
correct PYTHONPATH internally for the `parse_rosbag.py` subprocess it
launches, since that subprocess needs both those packages' outer directories
on its path -- see `convert_bag()`.)

Subprocess calls below hardcode the literal `python3` command (never
`sys.executable`) so that this still works correctly even if the top-level
script is accidentally invoked through `uv run python` -- `sys.executable`
would then resolve to the uv venv's interpreter, which cannot import ROS2
packages.

Usage:

    # Smoke test: convert just 1-2 bags before committing to the full batch.
    python3 scripts/convert_all_or_bags.py --max-bags 1

    # Full batch (all 75 bags). Takes hours; run in the background.
    python3 scripts/convert_all_or_bags.py
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

ORSCENE_ROOT = Path("/mnt/nas/private_workspace/chenglin/ORScene_bags")
BAG_MAP_VERSIONS_PATH = REPO_ROOT / "data" / "bag_map_versions.json"
OUTPUT_ROOT = REPO_ROOT / "data" / "or_scene_npz"
MAP_CACHE_DIR = Path("/home/chenglin/autoware_map/extracted")
EXTRACT_MAP_SCRIPT = REPO_ROOT / "ros_scripts" / "extract_map_from_bag.py"
PARSE_SCRIPT = REPO_ROOT / "ros_scripts" / "parse_rosbag.py"

MIN_FREE_GB_DEFAULT = 50.0


def load_bag_map_versions(path: Path) -> dict[str, tuple[str, str]]:
    """Load {bag_path: (area, version)} from the pre-built mapping file.

    This mapping was built once (by reading `/map/vector_map` from each of
    the 75 bags) and is the authoritative list of which bags to convert --
    it is exactly the 75 real scene bags (verified 1:1 against a filesystem
    walk of ORSCENE_ROOT), so we iterate it directly instead of re-walking
    NAS and guessing which `*.db3` directories are "real" bags vs. chunks.
    """
    with open(path) as f:
        raw = json.load(f)
    return {k: (v[0], v[1]) for k, v in raw.items()}


def bag_output_dir(bag_path: Path, area: str) -> tuple[Path, str]:
    """Return (save_dir, bag_stem) for a bag, matching the required layout
    `data/or_scene_npz/<area>/<category>/<scene>/*.npz`.

    `bag_map_versions.json` keys are always
    `ORSCENE_ROOT/<area>/<category>/<scene>/<uuid>/<bag_name>.db3`, and every
    scene directory contains exactly one bag (verified across all 75
    entries), so `<area>/<category>/<scene>` is a safe, collision-free save
    directory without an extra per-bag subdirectory.
    """
    rel = bag_path.relative_to(ORSCENE_ROOT)
    parts = rel.parts
    if len(parts) != 5:
        raise ValueError(f"Unexpected bag path depth ({len(parts)} parts, expected 5): {bag_path}")
    path_area, category, scene, _uuid, bag_dirname = parts
    if path_area != area:
        print(f"  WARNING: area mismatch for {bag_path}: path says {path_area!r}, json says {area!r}")
    save_dir = OUTPUT_ROOT / area / category / scene
    bag_stem = Path(bag_dirname).stem  # strips ".db3"
    return save_dir, bag_stem


def existing_npz_count(save_dir: Path, bag_stem: str) -> int:
    if not save_dir.exists():
        return 0
    return len(list(save_dir.glob(f"{bag_stem}_*.npz")))


def get_or_create_osm(area: str, version: str, bag_path: Path, cache_dir: Path) -> Path:
    """Get the cached `.osm` for (area, version), extracting it from `bag_path`
    via `ros_scripts/extract_map_from_bag.py` if not already cached."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    osm_path = cache_dir / f"{area}_v{version}.osm"
    if osm_path.exists() and osm_path.stat().st_size > 0:
        return osm_path

    cmd = [
        "python3", str(EXTRACT_MAP_SCRIPT),
        str(bag_path),
        "--output", str(osm_path),
    ]
    print(f"  Extracting map for {area} v{version} from {bag_path.name} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not osm_path.exists() or osm_path.stat().st_size == 0:
        # Clean up a partial/empty file so a later retry doesn't treat it as cached.
        if osm_path.exists():
            osm_path.unlink()
        raise RuntimeError(
            f"map extraction failed for {area} v{version} (bag={bag_path}):\n"
            f"{result.stderr[-2000:]}"
        )
    print(f"  Cached map: {osm_path}")
    return osm_path


def convert_bag(bag_path: Path, osm_path: Path, save_dir: Path) -> tuple[int, str | None]:
    """Run parse_rosbag.py and return (npz_count_produced_by_this_bag, error_or_None)."""
    save_dir.mkdir(parents=True, exist_ok=True)
    bag_stem = bag_path.stem

    before = existing_npz_count(save_dir, bag_stem)

    cmd = [
        "python3", str(PARSE_SCRIPT),
        str(bag_path), str(osm_path), str(save_dir),
        "--step", "1", "--min_frames", "0",
    ]
    env = os.environ.copy()
    # `diffusion_planner.dimensions` and `diffusion_planner_ros.*` are only
    # importable if their *outer* directories are on PYTHONPATH (they are
    # normally picked up via editable-install in the uv venv, which is not
    # what parse_rosbag.py runs under here -- it needs system python3 for
    # ROS2 packages instead). REPO_ROOT alone is not enough.
    dp_path = str(REPO_ROOT / "diffusion_planner")
    dp_ros_path = str(REPO_ROOT / "diffusion_planner_ros")
    env["PYTHONPATH"] = f"{dp_path}:{dp_ros_path}:{env.get('PYTHONPATH', '')}"

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    after = existing_npz_count(save_dir, bag_stem)
    produced = after - before

    if result.returncode != 0:
        return produced, result.stderr[-2000:]
    return produced, None


def iter_bags(bag_map: dict[str, tuple[str, str]]):
    for bag_path_str in sorted(bag_map.keys()):
        area, version = bag_map[bag_path_str]
        yield Path(bag_path_str), area, version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-bags", type=int, default=999, help="Stop after converting this many bags (for smoke testing).")
    parser.add_argument("--no-skip-existing", action="store_true", help="Reconvert bags even if npz already exist for them.")
    parser.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB_DEFAULT, help="Abort if free space on the output disk drops below this.")
    parser.add_argument("--bag-map-versions", type=Path, default=BAG_MAP_VERSIONS_PATH)
    parser.add_argument("--map-cache-dir", type=Path, default=MAP_CACHE_DIR)
    args = parser.parse_args()

    if not args.bag_map_versions.exists():
        print(f"ERROR: {args.bag_map_versions} not found.")
        sys.exit(1)

    bag_map = load_bag_map_versions(args.bag_map_versions)
    print(f"Loaded {len(bag_map)} bags from {args.bag_map_versions}")

    n_processed = 0
    n_skipped = 0
    n_failed = 0
    total_npz = 0
    t0 = time.monotonic()

    for bag_path, area, version in iter_bags(bag_map):
        if n_processed + n_skipped + n_failed >= args.max_bags:
            print(f"\nReached --max-bags={args.max_bags}, stopping.")
            break

        idx = n_processed + n_skipped + n_failed
        print(f"[{idx}] {area}/v{version}: {bag_path}")

        if not bag_path.exists():
            print("  SKIP: bag path does not exist on NAS (unmounted or moved?)")
            n_failed += 1
            continue

        try:
            save_dir, bag_stem = bag_output_dir(bag_path, area)
        except ValueError as e:
            print(f"  FAILED: {e}")
            n_failed += 1
            continue

        if not args.no_skip_existing:
            existing = existing_npz_count(save_dir, bag_stem)
            if existing > 0:
                print(f"  SKIP (already converted: {existing} npz in {save_dir})")
                total_npz += existing
                n_skipped += 1
                continue

        stat = shutil.disk_usage(REPO_ROOT)
        free_gb = stat.free / (1024**3)
        if free_gb < args.min_free_gb:
            print(f"ERROR: only {free_gb:.1f}GB free (< --min-free-gb={args.min_free_gb}), stopping.")
            break

        try:
            osm_path = get_or_create_osm(area, version, bag_path, args.map_cache_dir)
            n_npz, err = convert_bag(bag_path, osm_path, save_dir)
            if err is not None:
                print(f"  FAILED (parse_rosbag.py exited nonzero): {err}")
                n_failed += 1
                continue
            print(f"  -> {n_npz} npz in {save_dir}")
            total_npz += n_npz
            n_processed += 1
        except Exception as e:  # noqa: BLE001 - keep the batch going on any single-bag failure
            print(f"  FAILED: {e}")
            n_failed += 1

    elapsed_min = (time.monotonic() - t0) / 60.0
    print(
        f"\nDone in {elapsed_min:.1f} min: "
        f"{n_processed} converted, {n_skipped} skipped, {n_failed} failed, "
        f"{total_npz} total npz."
    )


if __name__ == "__main__":
    main()
