"""Recursively glob ``*.npz`` files and write a path-list JSON.

Two output formats (see ``diffusion_planner/utils/path_list.py``):

  * legacy per-frame files (no ``frame_indices`` key) are written as plain path strings;
  * packed sequence files (with ``frame_indices``) are written as manifest objects
    ``{"path", "num_frames"}`` so the loader knows how many samples each file yields.

The format is decided per file, so a mixed root produces a mixed list that the loader handles
transparently. ``valid`` is intentionally left out here (loader treats "no valid" as "all
frames"); apply frame-level filtering as a separate pass that adds a ``valid`` list.
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir_list", type=Path, nargs="+")
    parser.add_argument("--save_path", type=Path, default=None)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="processes used to read each npz's frame count (packed detection)",
    )
    return parser.parse_args()


def probe_entry(npz_file: Path):
    """Return a path-list entry for one npz: a manifest dict if packed, else a plain string."""
    path = str(npz_file)
    with np.load(npz_file, allow_pickle=True) as npz:
        if "frame_indices" in npz.files:
            return {"path": path, "num_frames": int(len(npz["frame_indices"]))}
    return path


if __name__ == "__main__":
    args = parse_args()
    root_dir_list = args.root_dir_list
    save_path = args.save_path
    if save_path is None:
        save_path = root_dir_list[0].parent / "path_list.json"

    log = open(save_path.with_suffix(".log"), "w")

    npz_paths = []
    for root_dir in root_dir_list:
        root_dir = root_dir.resolve()
        assert root_dir.is_absolute(), f"{root_dir} is not an absolute path."
        assert root_dir.exists(), f"{root_dir} does not exist."
        assert root_dir.is_dir(), f"{root_dir} is not a directory."

        found = sorted(root_dir.rglob("*.npz"))
        print(f"Found {len(found)} npz files in {root_dir}.")
        log.write(f"Found {len(found)} npz files in {root_dir}.\n")
        npz_paths.extend(found)

    print(f"Found {len(npz_paths)} npz files in total. Probing frame counts...")
    log.write(f"Found {len(npz_paths)} npz files in total.\n")

    with Pool(args.num_workers) as pool:
        all_list = list(tqdm(pool.imap(probe_entry, npz_paths, chunksize=16), total=len(npz_paths)))

    num_packed = sum(1 for e in all_list if isinstance(e, dict))
    num_frames = sum(e["num_frames"] for e in all_list if isinstance(e, dict))
    num_frames += sum(1 for e in all_list if isinstance(e, str))
    msg = f"Packed files: {num_packed}/{len(all_list)}; total samples (frames): {num_frames}."
    print(msg)
    log.write(msg + "\n")

    with open(save_path, "w") as f:
        json.dump(all_list, f, indent=4)

    print(f"Saved path list to {save_path}")
