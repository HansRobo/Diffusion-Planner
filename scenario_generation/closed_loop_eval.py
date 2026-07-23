"""Low-level closed-loop rollout helpers.

``run_closed_loop_eval`` is the tier4-main entry point used by ``train.py``.
New code should use :class:`scenario_generation.closed_loop_evaluation.FullRouteClosedLoopEvaluation`.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from scenario_generation.route_timeline import group_routes


def enumerate_routes(npz_root: Path) -> dict[str, list[Path]]:
    """Group all .npz under ``npz_root`` into routes (bag-prefix groups)."""
    paths = sorted(Path(npz_root).rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz under {npz_root}")
    return group_routes(paths)


def aggregate(rows: list[dict], near_miss_thresh: float) -> dict:
    """Aggregate per-segment metric rows into a single closed-loop summary."""
    n_seg = len(rows)
    total_steps = sum(r["n_steps_run"] for r in rows)
    total_collision_steps = sum(r["n_collision_steps"] for r in rows)
    total_near_miss_steps = sum(r["n_near_miss_steps"] for r in rows)
    total_snaps = sum(r["n_snaps"] for r in rows)

    n_seg_collision = sum(1 for r in rows if r["n_collision_steps"] > 0)
    n_seg_near_miss = sum(1 for r in rows if r["n_near_miss_steps"] > 0)
    n_seg_diverged = sum(1 for r in rows if r["terminated"] == "diverged")

    finite_min_cl = [r["min_clearance"] for r in rows if np.isfinite(r["min_clearance"])]
    finite_mean_cl = [r["mean_clearance"] for r in rows if np.isfinite(r["mean_clearance"])]

    term_counts: dict[str, int] = {}
    for r in rows:
        term_counts[r["terminated"]] = term_counts.get(r["terminated"], 0) + 1

    return {
        "near_miss_thresh": near_miss_thresh,
        "n_segments": n_seg,
        "total_steps": total_steps,
        "n_segments_with_collision": n_seg_collision,
        "collision_segment_rate": n_seg_collision / n_seg if n_seg else 0.0,
        "total_collision_steps": total_collision_steps,
        "collision_step_rate": total_collision_steps / total_steps if total_steps else 0.0,
        "n_segments_with_near_miss": n_seg_near_miss,
        "near_miss_segment_rate": n_seg_near_miss / n_seg if n_seg else 0.0,
        "total_near_miss_steps": total_near_miss_steps,
        "near_miss_step_rate": total_near_miss_steps / total_steps if total_steps else 0.0,
        "global_min_clearance": float(min(finite_min_cl)) if finite_min_cl else float("inf"),
        "mean_segment_min_clearance": float(np.mean(finite_min_cl))
        if finite_min_cl
        else float("inf"),
        "mean_segment_mean_clearance": float(np.mean(finite_mean_cl))
        if finite_mean_cl
        else float("inf"),
        "total_snaps": total_snaps,
        "n_segments_diverged": n_seg_diverged,
        "diverged_segment_rate": n_seg_diverged / n_seg if n_seg else 0.0,
        "terminated_counts": term_counts,
    }


def build_mp4(png_dir: Path, mp4_path: Path, fps: float) -> None:
    """Encode the PNG sequence in ``png_dir`` to an MP4."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-pattern_type",
            "glob",
            "-i",
            str(png_dir / "*.png"),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            str(mp4_path),
        ],
        check=True,
    )


def run_closed_loop_eval(
    model,
    model_args,
    npz_root,
    out_dir,
    *,
    seg_len: int,
    device: str,
    near_miss_thresh: float,
    search_radius: float,
    warmup_steps: int,
    unstick_after: int,
    unstick_advance_m: float,
    fps: float,
    draw_every: int,
    neighbor_history_mode: str,
    replan_interval: int = 10,
    unstick_radius_mult: float = 10.0,
    unstick_teleport_after: int = 300,
    tracker_mode: str = "mpc",
    verbose: bool = True,
    ddp_rank: int = 0,
    ddp_world_size: int = 1,
) -> dict:
    """tier4-main entry point; delegates to :class:`FullRouteClosedLoopEvaluation`."""
    from scenario_generation.closed_loop_evaluation import (
        ClosedLoopEvalConfig,
        FullRouteClosedLoopEvaluation,
        RolloutParams,
    )

    evaluator = FullRouteClosedLoopEvaluation(
        model,
        model_args,
        ClosedLoopEvalConfig(
            out_dir=Path(out_dir),
            params=RolloutParams(
                device=device,
                near_miss_thresh=near_miss_thresh,
                search_radius=search_radius,
                warmup_steps=warmup_steps,
                unstick_after=unstick_after,
                unstick_advance_m=unstick_advance_m,
                unstick_radius_mult=unstick_radius_mult,
                unstick_teleport_after=unstick_teleport_after,
                draw_every=draw_every,
                replan_interval=replan_interval,
                tracker_mode=tracker_mode,
                neighbor_history_mode=neighbor_history_mode,
            ),
            fps=fps,
            verbose=verbose,
        ),
        npz_root,
        seg_len=seg_len,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
    )
    if ddp_world_size > 1:
        return evaluator.run_distributed()
    return evaluator.run()
