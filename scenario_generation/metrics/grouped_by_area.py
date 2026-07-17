"""Grouped-by-area post-processing plugin (no rollout logic).

Consumes per-step records from an instrumented bag rollout and produces
episode-level metrics and videos using scenario classification JSON spans.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from diffusion_planner.utils.scene_skip import is_skipped

from scenario_generation.closed_loop_eval import build_mp4
from scenario_generation.closed_loop_types import BagRolloutResult, StepRecord
from scenario_generation.route_timeline import RouteTimeline
from scenario_generation.scenario_classification import idx_in_labeled_ranges


def _percentile(arr: np.ndarray, q: float) -> float:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("inf")
    return float(np.percentile(finite, q))


def aggregate_episode_metrics(
    steps: list[StepRecord],
    area_name: str,
    metric_group: str,
    route_key: str,
    bag_name: str,
    tl: RouteTimeline,
    *,
    labeled_ranges: list[list[int]],
    video_start_idx: int,
    video_end_idx: int,
    span_index: int,
    near_miss_thresh: float = 0.3,
) -> dict | None:
    """Aggregate metrics on labeled, non-skipped frames only."""
    area_steps = [
        st
        for st in steps
        if st.area == area_name
        and idx_in_labeled_ranges(st.rec_idx, labeled_ranges)
        and not is_skipped(tl.npz_paths[st.rec_idx])
    ]
    if not area_steps:
        return None

    f_start = int(tl.frame_indices[video_start_idx])
    f_end = int(tl.frame_indices[min(video_end_idx - 1, len(tl.frame_indices) - 1)])

    cl = np.array(
        [st.centerline_m for st in area_steps if st.centerline_m is not None],
        dtype=np.float32,
    )
    tm = np.array(
        [st.turn_match for st in area_steps if st.turn_match is not None],
        dtype=bool,
    )
    clearances = np.array([st.clearance_m for st in area_steps], dtype=np.float32)
    rb_dists = np.array([st.rb_dist_m for st in area_steps], dtype=np.float32)
    collisions = [st.collision for st in area_steps]

    return {
        "metric_group": metric_group,
        "area_name": area_name,
        "sequence": route_key,
        "bag": bag_name,
        "span_index": span_index,
        "list_start_idx": video_start_idx,
        "list_end_idx": video_end_idx,
        "labeled_ranges": labeled_ranges,
        "segment": f"[{f_start},{f_end}]",
        "n_steps_run": len(area_steps),
        "terminated": "area_span",
        "min_clearance": float(clearances[np.isfinite(clearances)].min())
        if clearances.size
        else float("inf"),
        "mean_clearance": float(clearances[np.isfinite(clearances)].mean())
        if clearances.size
        else float("inf"),
        "n_collision_steps": int(sum(collisions)),
        "n_near_miss_steps": int(np.sum(clearances <= near_miss_thresh)),
        "n_snaps": 0,
        "centerline_mean_m": float(cl[np.isfinite(cl)].mean()) if cl.size else None,
        "centerline_p95_m": _percentile(cl, 95) if cl.size else None,
        "centerline_max_m": float(cl[np.isfinite(cl)].max()) if cl.size else None,
        "turn_match_rate": float(tm.mean()) if tm.size else None,
        "neighbor_violation_steps": int(sum(st.neighbor_violation for st in area_steps)),
        "rb_violation_steps": int(sum(st.rb_violation for st in area_steps)),
        "collision_steps": int(sum(collisions)),
        "min_clearance_m": float(clearances[np.isfinite(clearances)].min())
        if clearances.size
        else float("inf"),
        "min_rb_dist_m": float(rb_dists[np.isfinite(rb_dists)].min())
        if rb_dists.size and np.isfinite(rb_dists).any()
        else None,
        "video_path": "",
    }


def build_episode_video(
    rollout: BagRolloutResult,
    out_mp4: Path,
    fps: float,
    *,
    video_start_idx: int,
    video_end_idx: int,
) -> bool:
    """Copy PNGs for a continuous video span (includes unlabeled stopping gaps)."""
    if rollout.png_dir is None:
        return False
    video_steps = [
        st
        for st in rollout.steps
        if st.png_path is not None and video_start_idx <= st.rec_idx < video_end_idx
    ]
    if not video_steps:
        return False

    staging = out_mp4.parent / f"_staging_{out_mp4.stem}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    for st in sorted(video_steps, key=lambda x: x.k):
        if st.png_path and st.png_path.is_file():
            shutil.copy2(st.png_path, staging / f"{st.png_path.name}")

    if not any(staging.glob("*.png")):
        shutil.rmtree(staging, ignore_errors=True)
        return False

    build_mp4(staging, out_mp4, fps)
    shutil.rmtree(staging, ignore_errors=True)
    return True
