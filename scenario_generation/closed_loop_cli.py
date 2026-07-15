"""CLI helpers for full-route closed-loop validation."""

from __future__ import annotations

from pathlib import Path


def print_full_route_summary(summary: dict, near_miss_thresh: float, out_dir: Path) -> None:
    n_seg = summary["n_segments"]
    print(f"\n=== closed-loop validation: {n_seg} segments in {summary['elapsed_sec']:.1f}s ===")
    print(
        f"collision: {summary['n_segments_with_collision']}/{n_seg} segments "
        f"(rate {summary['collision_segment_rate']:.4f}), "
        f"{summary['total_collision_steps']} steps (rate {summary['collision_step_rate']:.6f})"
    )
    print(
        f"near-miss (<= {near_miss_thresh} m): "
        f"{summary['n_segments_with_near_miss']}/{n_seg} segments "
        f"(rate {summary['near_miss_segment_rate']:.4f}), {summary['total_near_miss_steps']} steps"
    )
    print(
        f"global_min_clearance={summary['global_min_clearance']:.3f} m  "
        f"mean_segment_min_clearance={summary['mean_segment_min_clearance']:.3f} m  "
        f"mean_segment_mean_clearance={summary['mean_segment_mean_clearance']:.3f} m"
    )
    print(f"total_snaps={summary['total_snaps']}  terminated={summary['terminated_counts']}")
    print(f"videos: per-segment <route>_<start>_<end>.mp4 in {out_dir}")
