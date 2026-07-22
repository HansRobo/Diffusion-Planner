"""Match test frames to training data on the same lanelet, fetch and transform trajectories."""

import argparse
import fcntl
import hashlib
import math
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from human_match_prototype.build_lanelet_index import load_lanelet_index
from human_match_prototype.coord_transform import (
    WorldPose,
    pose_from_sidecar,
    transform_trajectory,
)
from human_match_prototype.metrics import multi_human_metrics


@dataclass
class MatchResult:
    n_matched: int
    human_futures: list[np.ndarray] = field(default_factory=list)
    match_metadata: list[dict] = field(default_factory=list)


def build_lanelet_lookup(index: list[dict]) -> dict[int, list[dict]]:
    lookup: dict[int, list[dict]] = defaultdict(list)
    for entry in index:
        lookup[entry["lanelet_id"]].append(entry)
    return dict(lookup)


def _heading_diff_deg(a_deg: float, b_deg: float) -> float:
    d = a_deg - b_deg
    d = (d + 180.0) % 360.0 - 180.0
    return abs(d)


def _bag_prefix(npz_path: str) -> str:
    """Extract bag/sequence prefix from NPZ path (everything before the last _NNNNN.npz)."""
    return npz_path.rsplit("_", 1)[0]


def _deduplicate_by_drive(entries: list[dict]) -> list[dict]:
    """Keep one representative frame per driving sequence (bag prefix)."""
    bags: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        bags[_bag_prefix(e["npz_path"])].append(e)
    result = []
    for bag_entries in bags.values():
        bag_entries.sort(key=lambda e: e.get("timestamp", 0))
        result.append(bag_entries[len(bag_entries) // 2])
    return result


def find_matches(
    test_sidecar: dict,
    lanelet_lookup: dict[int, list[dict]],
    test_lanelet_id: int,
    max_heading_diff_deg: float = 45.0,
    max_distance_m: float = 30.0,
    max_per_lanelet: int = 200,
    seed_key: str = "",
) -> list[dict]:
    candidates = lanelet_lookup.get(test_lanelet_id, [])
    if not candidates:
        return []

    tx, ty = test_sidecar["x"], test_sidecar["y"]
    th = test_sidecar["heading_deg"]

    filtered = []
    for c in candidates:
        if _heading_diff_deg(c["heading_deg"], th) > max_heading_diff_deg:
            continue
        dist = math.sqrt((c["x"] - tx) ** 2 + (c["y"] - ty) ** 2)
        if dist > max_distance_m:
            continue
        filtered.append(c)

    filtered = _deduplicate_by_drive(filtered)

    if len(filtered) > max_per_lanelet:
        seed = int(hashlib.md5(seed_key.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(filtered), size=max_per_lanelet, replace=False)
        filtered = [filtered[i] for i in sorted(idx)]

    return filtered


def fetch_npzs(npz_paths: list[str], dest: Path, host: str, bwlimit: int = 10000) -> None:
    """Fetch NPZ files (+ JSON sidecars) via rsync with flock serialization."""
    to_fetch = []
    for p in npz_paths:
        local = dest / p.lstrip("/")
        if not local.exists():
            to_fetch.append(p)
            sidecar = p[:-4] + ".json"
            to_fetch.append(sidecar)
    if not to_fetch:
        return

    listfile = dest / ".rsync_fetch_list.txt"
    listfile.write_text("\n".join(p.lstrip("/") for p in to_fetch) + "\n")

    lockfile = dest / ".rsync.lock"
    with open(lockfile, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            cmd = [
                "rsync",
                "-a",
                "--info=progress2",
                f"--bwlimit={bwlimit}",
                f"--files-from={listfile}",
                f"{host}:/",
                str(dest),
            ]
            subprocess.run(cmd, check=True)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def run_multi_human(
    scores_csv: str,
    index_path: str,
    map_path: str,
    output_csv: str,
    host: str = "sakurab",
    bwlimit: int = 10000,
    max_heading_diff_deg: float = 45.0,
    max_distance_m: float = 30.0,
    max_per_lanelet: int = 200,
    fetch_dest: str = "data/human_match/multi_human_mirror",
) -> None:
    import lanelet2
    from autoware_lanelet2_extension_python.projection import MGRSProjector

    from diffusion_planner.util_scripts.search_scenes import read_sidecar

    index, _ = load_lanelet_index(index_path)
    lookup = build_lanelet_lookup(index)

    projection = MGRSProjector(lanelet2.io.Origin(0.0, 0.0))
    lanelet_map = lanelet2.io.load(map_path, projection)
    ll_layer = lanelet_map.laneletLayer

    df = pd.read_csv(scores_csv)
    dest = Path(fetch_dest)
    dest.mkdir(parents=True, exist_ok=True)

    multi_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Multi-human matching"):
        npz_path = row["npz_path"]
        test_sidecar = read_sidecar(npz_path)
        if test_sidecar is None:
            multi_rows.append(multi_human_metrics([], np.empty((0, 80, 3)), np.empty((80, 3))))
            continue

        point = lanelet2.core.BasicPoint2d(test_sidecar["x"], test_sidecar["y"])
        nearest = lanelet2.geometry.findNearest(ll_layer, point, 1)
        if not nearest:
            multi_rows.append(multi_human_metrics([], np.empty((0, 80, 3)), np.empty((80, 3))))
            continue
        test_ll_id = nearest[0][1].id

        matches = find_matches(
            test_sidecar,
            lookup,
            test_ll_id,
            max_heading_diff_deg,
            max_distance_m,
            max_per_lanelet,
            seed_key=npz_path,
        )

        if not matches:
            multi_rows.append(multi_human_metrics([], np.empty((0, 80, 3)), np.empty((80, 3))))
            continue

        fetch_npzs([m["npz_path"] for m in matches], dest, host, bwlimit)

        to_pose = pose_from_sidecar(test_sidecar)
        human_futures = []
        for m in matches:
            local_npz = dest / m["npz_path"].lstrip("/")
            if not local_npz.exists():
                continue
            try:
                data = np.load(str(local_npz))
                traj = data["ego_agent_future"][:, :3]  # (80, 3)
            except Exception:
                continue
            match_sidecar = read_sidecar(str(local_npz))
            if match_sidecar is None:
                match_sidecar = m
            from_pose = pose_from_sidecar(match_sidecar)
            transformed = transform_trajectory(traj, from_pose, to_pose)
            human_futures.append(transformed)

        test_human = np.load(npz_path)["ego_agent_future"][:, :3]
        # Re-sample from ONNX to get DP samples for dp/human coverage metrics
        from human_match_prototype.sampler import TrajectorySampler

        if not hasattr(run_multi_human, "_sampler"):
            _model = Path("/opt/autoware/mlmodels/diffusion_planner_for_x2")
            run_multi_human._sampler = TrajectorySampler(
                str(_model / "args.json"), str(_model / "diffusion_planner.onnx")
            )
        sr = run_multi_human._sampler.sample(npz_path, num_samples=64, seed=0)
        multi_rows.append(multi_human_metrics(human_futures, sr.ego_samples, test_human))

    multi_df = pd.DataFrame(multi_rows)
    result = pd.concat([df.reset_index(drop=True), multi_df.reset_index(drop=True)], axis=1)
    result.to_csv(output_csv, index=False)
    print(f"Wrote {len(result)} rows to {output_csv}")


def parse_args_cli():
    p = argparse.ArgumentParser(description="Run multi-human matching on scored frames.")
    p.add_argument("--scores_csv", required=True)
    p.add_argument("--index", required=True, help="Lanelet index .parquet")
    p.add_argument("--map", required=True, help="Lanelet2 .osm map")
    p.add_argument("--output", required=True)
    p.add_argument("--host", default="sakurab")
    p.add_argument("--bwlimit", type=int, default=10000)
    p.add_argument("--max_per_lanelet", type=int, default=200)
    p.add_argument("--fetch_dest", default="data/human_match/multi_human_mirror")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args_cli()
    run_multi_human(
        args.scores_csv,
        args.index,
        args.map,
        args.output,
        host=args.host,
        bwlimit=args.bwlimit,
        max_per_lanelet=args.max_per_lanelet,
        fetch_dest=args.fetch_dest,
    )
