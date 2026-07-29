#!/usr/bin/env python3
"""Select a reproducible hard-case pool from an AWR evaluation manifest.

This mirrors the useful part of the original HDP/PlannerRFT protocol: score a
large pool with the frozen base planner, then train on a controlled mixture of
low-score and safety-risk scenes.  It intentionally writes only scene paths;
the source evaluation JSON remains the audit record.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def risk(row: dict) -> float:
    reward = float(row.get("det_reward", 0.0))
    return (
        100.0 * float(row.get("det_collision", 0.0))
        + 35.0 * float(row.get("det_rb_crossing", 0.0))
        + 25.0 * float(row.get("det_lane_crossing", 0.0))
        + 35.0 * float(row.get("det_kinematic_violation", 0.0))
        + max(0.0, 50.0 - reward)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes_json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument(
        "--random_count",
        type=int,
        default=0,
        help="append this many reproducibly sampled non-hard scenes after the ranked hard pool",
    )
    parser.add_argument(
        "--safety_only",
        action="store_true",
        help="keep only scenes whose frozen-base row has a collision or road-border crossing",
    )
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    rows = json.loads(args.scenes_json.read_text())
    if not isinstance(rows, list):
        raise ValueError(f"expected a scene-row list: {args.scenes_json}")
    all_rows = list(rows)
    if args.safety_only:
        rows = [
            row
            for row in rows
            if float(row.get("det_collision", 0.0)) > 0.5
            or float(row.get("det_rb_crossing", 0.0)) > 0.5
        ]
        if not rows:
            raise ValueError("--safety_only removed every scene")
    ranked = sorted(rows, key=risk, reverse=True)
    hard_rows = ranked[: max(1, args.count)]
    chosen = [str(row["scene_path"]) for row in hard_rows]
    if args.random_count > 0:
        hard_paths = {str(row["scene_path"]) for row in hard_rows}
        remaining = [
            row for row in all_rows if str(row["scene_path"]) not in hard_paths
        ]
        rng = random.Random(args.seed)
        take = min(int(args.random_count), len(remaining))
        chosen.extend(str(row["scene_path"]) for row in rng.sample(remaining, take))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(chosen, indent=2) + "\n")
    print(f"selected {len(chosen)} / {len(rows)} hard scenes -> {args.output}")


if __name__ == "__main__":
    main()
