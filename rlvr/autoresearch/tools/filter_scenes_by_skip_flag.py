"""Filter a training scene list by the converter's per-frame skip flag (CLI pre-pass).

When a corpus is generated with the cpp converter's ``--write_skipped_npz=1`` (so the
closed-loop reproducer gets a gap-free timeline), every 10 Hz frame is written,
including the ones the production filter would normally drop (stopped at a red light,
no future progress, ...). Each frame's JSON sidecar carries ``is_skipped``. TRAINING
must not learn from those frames.

Most training/eval paths now drop them automatically (the loader and scene-list
intakes call ``diffusion_planner.utils.scene_skip.filter_scene_list`` by default). This
tool stays as an explicit pre-pass for cases where you want a pre-filtered list on disk
(e.g. handing a list to an external script). It is a thin wrapper over the same shared
helper, so the filtering logic lives in exactly one place.

Backward-compatible: a sidecar without the field (older corpora) is treated as NOT
skipped, so existing lists pass through unchanged.

Example::

    python -m rlvr.autoresearch.tools.filter_scenes_by_skip_flag \
        --scenes train_all.json --sidecar_root /path/to/npz_dir \
        --out train_all_noskip.json
"""

from __future__ import annotations

import argparse
import functools
import json
import os
from multiprocessing import Pool
from pathlib import Path

# Re-exported so existing importers keep working; the implementation lives in the
# shared, dependency-light helper at the lowest package layer.
from diffusion_planner.utils.scene_skip import filter_scene_list, is_skipped  # noqa: F401


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scenes", type=Path, required=True, help="input scene list (.json of npz paths)"
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
        default=min(os.cpu_count() or 1, 64),
        help="parallel sidecar readers for large lists (default: min(cpu_count, 64))",
    )
    p.add_argument(
        "--chunksize",
        type=int,
        default=256,
        help="multiprocessing chunk size for sidecar reads",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scenes = json.loads(args.scenes.read_text())
    if not isinstance(scenes, list):
        raise ValueError(f"{args.scenes} is not a JSON list of npz paths")
    if args.workers <= 1:
        kept = filter_scene_list(scenes, sidecar_root=args.sidecar_root, label=str(args.scenes))
    else:
        # The shared helper remains the source of truth for one path.  This
        # parallel intake is needed for the 5M-frame corpus: a sequential
        # sidecar open would turn a pre-pass into an hours-long bottleneck.
        checker = functools.partial(is_skipped, sidecar_root=args.sidecar_root)
        # Sidecars live on RDMA storage. A bounded process pool overlaps the
        # filesystem latency without the 64-worker kernel-I/O pile-up seen on
        # the first attempt; callers should normally use 8--16 workers.
        with Pool(processes=args.workers) as pool:
            flags = list(pool.imap(checker, scenes, chunksize=max(1, args.chunksize)))
        kept = [scene for scene, skipped in zip(scenes, flags) if not skipped]
        dropped = sum(bool(flag) for flag in flags)
        print(
            f"[skip-filter] {args.scenes}: kept {len(kept)}/{len(scenes)} "
            f"(dropped {dropped} skip_for_training; workers={args.workers})"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(kept))
    print(f"wrote {len(kept)} scenes -> {args.out}")


if __name__ == "__main__":
    main()
