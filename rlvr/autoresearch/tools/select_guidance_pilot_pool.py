#!/usr/bin/env python3
"""Build a reproducible real-scene pool for guided-exploration audits.

The pool is stratified by *observed base-policy behavior*, not hand-picked
filenames.  It is intended for PlannerRFT-style response curves and OBB/GIF
visualizations before any guided sampler is allowed into full AWR mining.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _terminal(row: dict) -> bool:
    return any(
        float(row.get(key, 0.0)) > 0.5
        for key in ("det_collision", "det_rb_crossing", "det_kinematic_violation")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes_json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quota", type=int, default=24)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if args.quota < 1:
        raise ValueError("--quota must be positive")

    rows = json.loads(args.scenes_json.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"expected a non-empty scene-row list: {args.scenes_json}")

    safe = [row for row in rows if not _terminal(row)]
    random_pool = [
        row
        for row in safe
        if float(row.get("det_reward", 0.0)) >= 0.90
        and float(row.get("all_zero_reward_group", 0.0)) < 0.5
    ]
    rng = random.Random(args.seed)
    rng.shuffle(random_pool)

    strata = {
        "all_zero_or_tied": sorted(
            [row for row in rows if float(row.get("all_zero_reward_group", 0.0)) > 0.5],
            key=lambda row: (float(row.get("det_reward", 0.0)), row["scene_path"]),
        ),
        "collision": sorted(
            [row for row in rows if float(row.get("det_collision", 0.0)) > 0.5],
            key=lambda row: row["scene_path"],
        ),
        "road_border": sorted(
            [row for row in rows if float(row.get("det_rb_crossing", 0.0)) > 0.5],
            key=lambda row: (float(row.get("det_reward", 0.0)), row["scene_path"]),
        ),
        "kinematic": sorted(
            [row for row in rows if float(row.get("det_kinematic_violation", 0.0)) > 0.5],
            key=lambda row: (float(row.get("det_reward", 0.0)), row["scene_path"]),
        ),
        "low_reward_nonterminal": sorted(
            safe,
            key=lambda row: (float(row.get("det_reward", 0.0)), row["scene_path"]),
        ),
        "native_mode_collapse": sorted(
            safe,
            key=lambda row: (
                float(row.get("candidate_endpoint_spread_max", float("inf"))),
                float(row.get("candidate_reward_std", float("inf"))),
                row["scene_path"],
            ),
        ),
        "native_high_diversity": sorted(
            safe,
            key=lambda row: (
                -float(row.get("candidate_endpoint_spread_max", 0.0)),
                row["scene_path"],
            ),
        ),
        "ordinary_random": random_pool,
    }

    selected: list[dict] = []
    seen: set[str] = set()
    counts: dict[str, int] = {}
    for stratum, candidates in strata.items():
        taken = 0
        for row in candidates:
            path = str(row["scene_path"])
            if path in seen:
                continue
            seen.add(path)
            selected.append({"stratum": stratum, **row})
            taken += 1
            if taken >= args.quota:
                break
        counts[stratum] = taken

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps([row["scene_path"] for row in selected], indent=2) + "\n")
    manifest = output.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "source": str(args.scenes_json.resolve()),
                "source_sha256": _sha256(args.scenes_json),
                "seed": args.seed,
                "quota_per_stratum": args.quota,
                "source_scene_count": len(rows),
                "selected_scene_count": len(selected),
                "stratum_counts": counts,
                "rows": selected,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"output": str(output), "manifest": str(manifest), "counts": counts}))


if __name__ == "__main__":
    main()
