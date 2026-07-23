"""Pre-filter a scene list by the converter's per-frame skip flag (standalone pre-pass).

When a corpus is generated with the cpp converter's ``--write_skipped_npz=1`` (so the
closed-loop reproducer gets a gap-free timeline), every 10 Hz frame is written,
including the ones the production filter would normally drop (stopped at a red/yellow
light, no future progress, GT collision, off-lane, stale data). Each frame's JSON
sidecar carries ``is_skipped``. TRAINING must not learn from those frames.

The training/eval loaders (``DiffusionPlannerData``) do not filter at load time. The
training and validation entrypoints prepare a run-local cache automatically; this
script is useful when a filtered list is needed by another tool.

It uses the same shared helper as the training entrypoints. Filtering is parallelized
over sidecar reads and the output is written atomically. Input and output are both a
flat JSON list of npz paths (the format ``DiffusionPlannerData`` consumes, e.g.
``path_list_valid.json``).

Backward-compatible: a frame with no resolvable sidecar (older corpora) is treated as NOT
skipped, so existing lists pass through unchanged.

Example::

    python ros_scripts/filter_scene_list.py \
        --scenes train_all.json --sidecar_root /path/to/npz_dir \
        --out train_all_noskip.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from diffusion_planner.utils.scene_skip import filter_manifest_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scenes",
        type=Path,
        required=True,
        help="input scene list (.json: flat list of npz paths)",
    )
    p.add_argument("--out", type=Path, required=True, help="output filtered scene list")
    p.add_argument(
        "--sidecar_root",
        type=Path,
        default=None,
        help="root of pose/skip JSON sidecars if not next to the NPZ (e.g. the "
        "pre-padding conversion tree when the padded NPZs dropped their sidecars)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=32,
        help="parallel sidecar readers (default: 32)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scenes_path = args.scenes
    out_path = args.out
    sidecar_root = args.sidecar_root

    stats = filter_manifest_file(
        scenes_path,
        out_path,
        sidecar_root=sidecar_root,
        workers=args.workers,
        label=str(scenes_path),
    )
    print(
        f"wrote {stats['kept_count']}/{stats['source_count']} scenes "
        f"(dropped {stats['dropped_count']}) -> {out_path}"
    )


if __name__ == "__main__":
    main()
