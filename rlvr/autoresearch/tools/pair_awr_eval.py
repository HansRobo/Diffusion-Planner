#!/usr/bin/env python3
"""Create a scene-paired baseline versus AWR comparison artifact."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


METRICS = [
    "det_reward",
    "det_collision",
    "det_rb_crossing",
    "det_lane_crossing",
    "det_kinematic_violation",
    "det_off_road_fraction",
    "det_progress",
    "det_safety",
    "det_centerline",
    "det_smoothness",
    "configured_safe",
    "path_len",
    "ade",
    "fde",
]

METRIC_DIRECTIONS = {
    "det_reward": 1,
    "det_collision": -1,
    "det_rb_crossing": -1,
    "det_lane_crossing": -1,
    "det_kinematic_violation": -1,
    "det_off_road_fraction": -1,
    "det_progress": 1,
    "det_safety": 1,
    "det_centerline": 1,
    "det_smoothness": 1,
    "configured_safe": 1,
    # Path length alone is neither good nor bad; do not manufacture a
    # better-fraction for it without route context.
    "path_len": 0,
    "ade": -1,
    "fde": -1,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--awr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and "rows" in payload:
        return payload["rows"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"unsupported evaluation artifact: {path}")


def _finite_pair(before: dict[str, Any], after: dict[str, Any], metric: str) -> tuple[float, float] | None:
    """Return a numeric pair only when both scene-level values are finite."""
    try:
        before_value = float(before[metric])
        after_value = float(after[metric])
    except (KeyError, TypeError, ValueError):
        return None
    if not (math.isfinite(before_value) and math.isfinite(after_value)):
        return None
    return before_value, after_value


def main() -> None:
    args = _parse_args()
    baseline_rows = {row["scene_path"]: row for row in _load(args.baseline)}
    awr_rows = {row["scene_path"]: row for row in _load(args.awr)}
    common = sorted(set(baseline_rows) & set(awr_rows))
    if not common:
        raise ValueError("baseline and AWR evaluation sets have no common scene paths")
    paired: list[dict[str, Any]] = []
    for path in common:
        before = baseline_rows[path]
        after = awr_rows[path]
        row: dict[str, Any] = {"scene_path": path}
        for metric in METRICS:
            if metric in before and metric in after:
                row[f"baseline_{metric}"] = before[metric]
                row[f"awr_{metric}"] = after[metric]
                try:
                    row[f"delta_{metric}"] = float(after[metric]) - float(before[metric])
                except (TypeError, ValueError):
                    pass
        # Expert path length is scene context, not a policy metric.  Preserve
        # it once so contraction can be separated into stopped/low-speed and
        # genuinely moving scenes instead of being interpreted from one
        # corpus-wide path-length average.
        try:
            expert_path_len = float(before["gt_path_len"])
            if math.isfinite(expert_path_len):
                row["expert_gt_path_len"] = expert_path_len
        except (KeyError, TypeError, ValueError):
            pass
        # Do not turn a missing/non-finite value into a false comparison.  The
        # named fractions below must use the same finite paired denominator as
        # the metric-specific fractions and confidence intervals.
        for output_name, metric, direction in (
            ("reward_improved", "det_reward", 1),
            ("collision_improved", "det_collision", -1),
            ("rb_improved", "det_rb_crossing", -1),
            ("ade_improved", "ade", -1),
        ):
            values = _finite_pair(before, after, metric)
            if values is not None:
                before_value, after_value = values
                row[output_name] = float((after_value - before_value) * direction > 0.0)
        paired.append(row)

    summary: dict[str, Any] = {
        "scene_count": len(paired),
        "baseline_scene_count": len(baseline_rows),
        "awr_scene_count": len(awr_rows),
    }
    for metric in METRICS:
        deltas = [
            float(row[f"delta_{metric}"])
            for row in paired
            if f"delta_{metric}" in row and math.isfinite(float(row[f"delta_{metric}"]))
        ]
        if deltas:
            summary[f"mean_delta_{metric}"] = float(np.mean(deltas))
            summary[f"median_delta_{metric}"] = float(np.median(deltas))
            direction = int(METRIC_DIRECTIONS.get(metric, 0))
            if direction:
                summary[f"awr_better_fraction_{metric}"] = float(
                    np.mean(np.asarray(deltas) * direction > 0.0)
                )
    for name in ("reward_improved", "collision_improved", "rb_improved", "ade_improved"):
        values = [float(row[name]) for row in paired if name in row]
        if values:
            metric_name = name.removesuffix("_improved")
            summary[f"fraction_{name}"] = float(np.mean(values))
            # Keep the denominator and the number of positive transitions
            # explicit.  The old ``count_*_improved`` name stored the finite
            # pair denominator, which was numerically correct but easy to
            # misread as the number of improved scenes.
            summary[f"finite_pair_count_{metric_name}"] = len(values)
            summary[f"improved_scene_count_{metric_name}"] = int(sum(values))

    path_strata: list[dict[str, Any]] = []
    for label, lower, upper in (
        ("le_1m", -math.inf, 1.0),
        ("1_to_10m", 1.0, 10.0),
        ("10_to_30m", 10.0, 30.0),
        ("30_to_60m", 30.0, 60.0),
        ("gt_60m", 60.0, math.inf),
    ):
        selected = [
            row
            for row in paired
            if "expert_gt_path_len" in row
            and lower < float(row["expert_gt_path_len"]) <= upper
        ]
        if not selected:
            continue
        item: dict[str, Any] = {
            "stratum": label,
            "lower_exclusive_m": None if not math.isfinite(lower) else lower,
            "upper_inclusive_m": None if not math.isfinite(upper) else upper,
            "scene_count": len(selected),
        }
        for metric in (
            "det_reward",
            "path_len",
            "ade",
            "fde",
            "det_progress",
            "det_collision",
            "det_rb_crossing",
            "det_kinematic_violation",
        ):
            deltas = [
                float(row[f"delta_{metric}"])
                for row in selected
                if f"delta_{metric}" in row
                and math.isfinite(float(row[f"delta_{metric}"]))
            ]
            if deltas:
                item[f"mean_delta_{metric}"] = float(np.mean(deltas))
                item[f"median_delta_{metric}"] = float(np.median(deltas))
        path_strata.append(item)
    summary["expert_path_strata"] = path_strata

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "rows": paired}, indent=2) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
