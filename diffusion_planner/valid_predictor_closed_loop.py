"""Closed-loop validation of a Diffusion-Planner checkpoint with the PerfectTracker.

Open-loop counterpart: ``valid_predictor.py``. Uses
:class:`scenario_generation.closed_loop_evaluation.FullRouteClosedLoopEvaluation`.

A *route* = one bag-prefix group of consecutive 10 Hz NPZ frames (``RouteTimeline``);
each route is sliced into ``--seg_len`` segments and rolled out with ``render_segment``.
Per-segment metrics are streamed to ``segments.jsonl`` and aggregated into ``summary.json``.

Only ``--model_path`` and ``--npz_root`` are required; outputs land under
``<model_path dir>/closed_loop/<timestamp>/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scenario_generation.closed_loop_evaluation import (
    ClosedLoopEvalConfig,
    FullRouteClosedLoopEvaluation,
    RolloutParams,
    resolve_default_out_dir,
)


def parse_args() -> argparse.Namespace:
    # Only the checkpoint and the NPZ dir are required; everything else is a tunable
    # knob with the closed-loop mining default. Outputs land next to the checkpoint.
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
        help="dir tree of route NPZ frames (recursively globbed, grouped into routes). "
        "Pose JSON sidecars are read from next to each .npz, falling back to this same tree.",
    )
    # --- tunable knobs (default to the closed-loop mining config) ---
    p.add_argument("--seg_len", type=int, default=6000, help="frames per segment (~60s @10Hz)")
    p.add_argument("--device", type=str, default="cuda", help="'cuda' or 'cpu'")
    p.add_argument("--near_miss_thresh", type=float, default=0.5, help="near-miss clearance (m)")
    p.add_argument(
        "--search_radius", type=float, default=1.5, help="PerceptionReproducer cursor search (m)"
    )
    p.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="steps driven by the recorded GT pose before handing control to the model",
    )
    p.add_argument(
        "--unstick_after",
        type=int,
        default=300,
        help="snap the ego to the GT pose ahead after this many no-progress steps (0=off)",
    )
    p.add_argument(
        "--unstick_advance_m", type=float, default=2.5, help="how far ahead to snap when unsticking"
    )
    p.add_argument(
        "--unstick_radius_mult",
        type=float,
        default=3.0,
        help="when stuck, first widen the cursor search_radius to this x nominal so it reaches "
        "frames further ahead (model proceeds on its own); restored to nominal once the ego moves. "
        "<=1 disables this gentle stage (teleport straight away at --unstick_after)",
    )
    p.add_argument(
        "--unstick_teleport_after",
        type=int,
        default=300,
        help="if still stuck this many steps AFTER the radius was widened, fall back to the hard "
        "teleport onto the GT pose ahead (last resort)",
    )
    p.add_argument("--fps", type=int, default=10, help="output video frame rate (10 = realtime)")
    p.add_argument(
        "--replan_interval",
        type=int,
        default=4,
        help="re-run the model every N steps (1 = every step). Between inferences the cached plan "
        "is executed, re-expressed in the current ego frame each step; the ego still steps at 10Hz",
    )
    p.add_argument(
        "--draw_every",
        type=int,
        default=8,
        help="render a PNG only every N steps (1 = every step). PNG rendering (matplotlib) is the "
        "dominant cost; this throttles it without touching the rollout. Frames are encoded at --fps "
        "regardless, so the video also plays N x faster (shorter). For real-time use --fps 10/N",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = resolve_default_out_dir(args.model_path)
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
