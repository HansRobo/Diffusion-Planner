"""Aggregate per-segment metric rows into summary tables."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


def aggregate_segment_rows(rows: list[dict]) -> dict:
    """Build per metric_group and per area_name summaries from segment rows."""
    by_group: dict[str, list[dict]] = defaultdict(list)
    by_area: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_group[row["metric_group"]].append(row)
        by_area[row["area_name"]].append(row)

    def _agg(items: list[dict]) -> dict:
        if not items:
            return {"n_segments": 0}
        n = len(items)

        def _mean(key: str) -> float | None:
            vals = [r[key] for r in items if r.get(key) is not None and r[key] == r[key]]
            return float(sum(vals) / len(vals)) if vals else None

        def _sum(key: str) -> int:
            return int(sum(r.get(key, 0) or 0 for r in items))

        return {
            "n_segments": n,
            "centerline_mean_m": _mean("centerline_mean_m"),
            "centerline_p95_m": _mean("centerline_p95_m"),
            "turn_match_rate": _mean("turn_match_rate"),
            "neighbor_violation_steps": _sum("neighbor_violation_steps"),
            "rb_violation_steps": _sum("rb_violation_steps"),
            "collision_steps": _sum("collision_steps"),
            "min_clearance_m": min(
                (r["min_clearance_m"] for r in items if r.get("min_clearance_m") is not None),
                default=None,
            ),
        }

    return {
        "by_metric_group": {k: _agg(v) for k, v in sorted(by_group.items())},
        "by_area_name": {k: _agg(v) for k, v in sorted(by_area.items())},
        "n_segments_total": len(rows),
    }


RESULTS_TABLE_COLUMNS = [
    "metric_group",
    "area_name",
    "sequence",
    "bag",
    "span_index",
    "list_start_idx",
    "list_end_idx",
    "segment",
    "n_steps_run",
    "terminated",
    "centerline_mean_m",
    "centerline_p95_m",
    "centerline_max_m",
    "turn_match_rate",
    "neighbor_violation_steps",
    "rb_violation_steps",
    "collision_steps",
    "n_collision_steps",
    "n_near_miss_steps",
    "min_clearance",
    "min_clearance_m",
    "mean_clearance",
    "min_rb_dist_m",
    "n_snaps",
    "video_path",
]


def write_results_table(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    extra = {k for row in rows for k in row} - set(RESULTS_TABLE_COLUMNS)
    fieldnames = RESULTS_TABLE_COLUMNS + sorted(extra)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_metrics_summary(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
