"""Fetch a per-cluster sample of training NPZs from the training server.

One batched rsync connection (never run find/glob on the server), then a
local symlink farm so run_all.py's category column picks up the cluster id
from the parent directory name.

Usage:
  SSH_AUTH_SOCK=/run/user/1000/keyring/.ssh .venv/bin/python -m \
    human_match_prototype.fetch_cluster_samples \
    --cluster_json /home/chenglin/workspace/cluster_report_output/cluster_sft_k20.json \
    --dest data/human_match/cluster_takanawa \
    [--per_cluster 50] [--seed 0] [--clusters 2] [--host sakurab]
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

SAFETY_FACTOR = 5
FALLBACK_NPZ_BYTES = 1_000_000
DEFAULT_SIZE_REF_DIR = Path("data/or_scene_npz")


def select_paths(clusters, per_cluster, seed, limit_clusters=None):
    """Deterministic per-cluster sample; clusters ordered by numeric id."""
    names = sorted(clusters, key=lambda k: int(k.replace("cluster_id", "")))
    if limit_clusters is not None:
        names = names[:limit_clusters]
    out = {}
    for name in names:
        paths = clusters[name]
        rng = np.random.default_rng(seed + int(name.replace("cluster_id", "")))
        k = min(per_cluster, len(paths))
        idx = rng.choice(len(paths), size=k, replace=False)
        out[name] = [paths[i] for i in sorted(idx)]
    return out


def estimate_avg_npz_bytes(ref_dir: Path) -> int:
    files = list(ref_dir.rglob("*.npz"))[:200] if ref_dir.exists() else []
    if not files:
        return FALLBACK_NPZ_BYTES
    return int(sum(f.stat().st_size for f in files) / len(files))


def check_capacity(est_bytes: int, free_bytes: int, factor: int = SAFETY_FACTOR) -> None:
    if free_bytes < factor * est_bytes:
        raise RuntimeError(
            f"Insufficient disk space: free {free_bytes / 1e9:.2f} GB < "
            f"{factor}x estimated transfer {est_bytes / 1e9:.2f} GB"
        )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cluster_json", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--per_cluster", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clusters", type=int, default=None, help="limit to first N clusters")
    p.add_argument("--host", default="sakurab")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.cluster_json) as f:
        clusters = json.load(f)
    selection = select_paths(clusters, args.per_cluster, args.seed, args.clusters)
    all_paths = [p for ps in selection.values() for p in ps]

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    est = len(all_paths) * estimate_avg_npz_bytes(DEFAULT_SIZE_REF_DIR)
    free = shutil.disk_usage(dest).free
    print(
        f"planned: {len(all_paths)} files, estimated {est / 1e9:.2f} GB, free {free / 1e9:.2f} GB"
    )
    check_capacity(est, free)

    mirror = dest / "mirror"
    mirror.mkdir(exist_ok=True)
    listfile = dest / "rsync_files.txt"
    listfile.write_text("\n".join(p.lstrip("/") for p in all_paths) + "\n")
    cmd = [
        "rsync",
        "-a",
        "--info=progress2",
        f"--files-from={listfile}",
        f"{args.host}:/",
        str(mirror),
    ]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    collected = []
    for name, paths in selection.items():
        cdir = dest / name
        cdir.mkdir(exist_ok=True)
        missing = 0
        for i, p in enumerate(paths):
            local = mirror / p.lstrip("/")
            if not local.exists():
                missing += 1
                print(f"MISSING after rsync: {p}")
                continue
            link = cdir / f"{i:04d}_{Path(p).name}"
            if not link.exists():
                link.symlink_to(local.resolve())
            collected.append(str(link))
        print(f"{name}: {len(paths) - missing}/{len(paths)} files")

    out_list = dest / "path_list.json"
    with out_list.open("w") as f:
        json.dump(collected, f, indent=1)
    print(f"wrote {len(collected)} paths to {out_list}")


if __name__ == "__main__":
    main()
