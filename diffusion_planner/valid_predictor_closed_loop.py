"""Closed-loop validation of a Diffusion-Planner checkpoint with the PerfectTracker.

Open-loop counterpart: ``valid_predictor.py``. Uses
:class:`scenario_generation.closed_loop_evaluation.FullRouteClosedLoopEvaluation`.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from scenario_generation.closed_loop_evaluation import (
    ClosedLoopEvalConfig,
    FullRouteClosedLoopEvaluation,
    RolloutParams,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="checkpoint .pth; args.json must sit next to it (e.g. epoch0001/best_model.pth)",
    )
    p.add_argument(
        "--npz_root",
        type=Path,
        required=True,
        help="dir tree of route NPZ frames (recursively globbed, grouped into routes).",
    )
    p.add_argument("--seg_len", type=int, default=6000, help="frames per segment (~60s @10Hz)")
    p.add_argument("--device", type=str, default="cuda", help="'cuda' or 'cpu'")
    p.add_argument("--near_miss_thresh", type=float, default=0.5, help="near-miss clearance (m)")
    p.add_argument(
        "--search_radius", type=float, default=1.5, help="PerceptionReproducer cursor search (m)"
    )
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--unstick_after", type=int, default=300)
    p.add_argument("--unstick_advance_m", type=float, default=2.5)
    p.add_argument("--unstick_radius_mult", type=float, default=3.0)
    p.add_argument("--unstick_teleport_after", type=int, default=300)
    p.add_argument("--fps", type=int, default=10, help="output video frame rate (10 = realtime)")
    p.add_argument("--replan_interval", type=int, default=4)
    p.add_argument("--draw_every", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.model_path.parent / "closed_loop" / datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"device: {args.device} | model: {args.model_path} | out: {out_dir}")

    config = ClosedLoopEvalConfig(
        out_dir=out_dir,
        params=RolloutParams(
            device=args.device,
            near_miss_thresh=args.near_miss_thresh,
            search_radius=args.search_radius,
            warmup_steps=args.warmup_steps,
            unstick_after=args.unstick_after,
            unstick_advance_m=args.unstick_advance_m,
            unstick_radius_mult=args.unstick_radius_mult,
            unstick_teleport_after=args.unstick_teleport_after,
            draw_every=args.draw_every,
            replan_interval=args.replan_interval,
            tracker_mode="perfect",
            neighbor_history_mode="recorded",
        ),
        fps=float(args.fps),
        verbose=True,
    )
    evaluator = FullRouteClosedLoopEvaluation.from_checkpoint(
        args.model_path,
        args.npz_root,
        config,
        seg_len=args.seg_len,
    )
    summary = evaluator.run()
    summary["model_path"] = str(args.model_path)


if __name__ == "__main__":
    main()
