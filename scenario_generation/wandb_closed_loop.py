"""Build wandb log payloads for grouped closed-loop validation."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import wandb

from scenario_generation.artifact_names import artifact_component

RESULTS_TABLE_COLUMNS = [
    "metric_group",
    "area_name",
    "bag",
    "span_index",
    "segment",
    "n_steps_run",
    "episode_reached",
    "episode_completed",
    "episode_progress_fraction",
    "centerline_mean_m",
    "centerline_p95_m",
    "turn_match_rate",
    "collision_steps",
    "n_collision_steps",
    "n_near_miss_steps",
    "min_clearance_m",
    "mean_clearance",
    "video_path",
]


def build_grouped_closed_loop_wandb_log(summary: dict, *, max_videos: int = 24) -> dict:
    """Scalars, per-area/per-group aggregates, results table, and episode videos."""
    log: dict = {"closed_loop/mode": "grouped"}
    log["closed_loop/grouped/n_episodes"] = int(summary.get("n_episodes", 0))
    log["closed_loop/grouped/n_episodes_expected"] = int(summary.get("n_episodes_expected", 0))
    log["closed_loop/grouped/episode_report_fraction"] = float(
        summary.get("episode_report_fraction", 0.0)
    )
    log["closed_loop/grouped/n_bags"] = int(summary.get("n_bags", 0))
    log["closed_loop/grouped/n_bags_expected"] = int(summary.get("n_bags_expected", 0))
    log["closed_loop/grouped/bag_report_fraction"] = float(summary.get("bag_report_fraction", 0.0))
    log["closed_loop/grouped/elapsed_sec"] = float(summary.get("elapsed_sec", 0.0))

    agg = summary.get("grouped_summary") or {}
    for group, stats in (agg.get("by_metric_group") or {}).items():
        safe_group = artifact_component(group)
        for key, val in stats.items():
            if not _wandb_scalar(val):
                continue
            log[f"closed_loop/grouped/metric_group/{safe_group}/{key}"] = val

    for area, stats in (agg.get("by_area_name") or {}).items():
        safe_area = artifact_component(area)
        for key, val in stats.items():
            if not _wandb_scalar(val):
                continue
            log[f"closed_loop/grouped/area/{safe_area}/{key}"] = val

    rows = summary.get("segments") or []
    if rows:
        df = pd.DataFrame(rows)
        cols = [c for c in RESULTS_TABLE_COLUMNS if c in df.columns]
        extra = [c for c in df.columns if c not in cols and c != "labeled_ranges"]
        log["closed_loop/grouped/results_table"] = wandb.Table(dataframe=df[cols + extra])

    video_mp4s = summary.get("video_mp4s") or []
    log["closed_loop/grouped/n_videos"] = len(video_mp4s)
    log["closed_loop/grouped/n_videos_logged"] = min(len(video_mp4s), max_videos)
    for mp4 in video_mp4s[:max_videos]:
        mp4_path = Path(mp4)
        key = _video_wandb_key(mp4_path)
        log[key] = wandb.Video(str(mp4_path), format="mp4")

    return log


def build_full_closed_loop_wandb_log(summary: dict) -> dict:
    """Legacy full-route closed-loop wandb payload."""
    scalar_keys = [
        "collision_segment_rate",
        "collision_step_rate",
        "near_miss_segment_rate",
        "near_miss_step_rate",
        "global_min_clearance",
        "mean_segment_min_clearance",
        "mean_segment_mean_clearance",
        "total_collision_steps",
        "total_near_miss_steps",
        "total_snaps",
        "total_steps",
    ]
    log = {"closed_loop/mode": "full"}
    for key in scalar_keys:
        val = summary.get(key)
        if _wandb_scalar(val):
            log[f"closed_loop/{key}"] = val
    for mp4 in summary.get("video_mp4s") or []:
        log[f"closed_loop/video/{Path(mp4).stem}"] = wandb.Video(str(mp4), format="mp4")
    return log


def _wandb_scalar(val) -> bool:
    if val is None:
        return False
    if isinstance(val, (int, bool)):
        return True
    if isinstance(val, float):
        return math.isfinite(val)
    return False


def _video_wandb_key(mp4_path: Path) -> str:
    parts = mp4_path.parts
    if "videos" in parts:
        idx = parts.index("videos")
        rel = "/".join(parts[idx + 1 :])
        return f"closed_loop/grouped/video/{rel.replace('/', '_')}"
    return f"closed_loop/grouped/video/{mp4_path.stem}"
