"""Shared CLI argument helpers and evaluator factory for closed-loop validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from scenario_generation.closed_loop_evaluation import (
    ClosedLoopEvalConfig,
    FullRouteClosedLoopEvaluation,
    RolloutParams,
    resolve_default_out_dir,
)
from scenario_generation.grouped_closed_loop_eval import (
    GroupedClassificationParams,
    GroupedMultiDateClosedLoopEvaluation,
)


def add_rollout_args(parser: argparse.ArgumentParser) -> None:
    """Rollout knobs shared by full-route and grouped-by-area validation."""
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="checkpoint .pth; args.json must sit next to it",
    )
    parser.add_argument(
        "--npz_root",
        type=Path,
        required=True,
        help="dir tree of route NPZ frames (recursively globbed into routes). "
        "Pose JSON sidecars are read from next to each .npz, falling back to this same tree.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="'cuda' or 'cpu'")
    parser.add_argument(
        "--near_miss_thresh",
        "--margin_m",
        type=float,
        default=0.5,
        help="clearance threshold (m) for near-miss / safety violation counting",
    )
    parser.add_argument(
        "--search_radius",
        type=float,
        default=1.5,
        help="PerceptionReproducer cursor search radius (m)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="steps driven by recorded GT pose before model control",
    )
    parser.add_argument(
        "--unstick_after",
        type=int,
        default=300,
        help="snap ego to GT pose after this many no-progress steps (0=off)",
    )
    parser.add_argument(
        "--unstick_advance_m",
        type=float,
        default=2.5,
        help="how far ahead to snap when unsticking (m)",
    )
    parser.add_argument(
        "--unstick_radius_mult",
        type=float,
        default=3.0,
        help="widen cursor search radius by this factor before hard teleport",
    )
    parser.add_argument(
        "--unstick_teleport_after",
        type=int,
        default=300,
        help="steps after radius widen before hard teleport onto GT pose ahead",
    )
    parser.add_argument("--fps", type=int, default=10, help="output video frame rate")
    parser.add_argument(
        "--draw_every",
        type=int,
        default=8,
        help="render PNG every N steps (matplotlib cost throttle)",
    )
    parser.add_argument(
        "--replan_interval",
        type=int,
        default=10,
        help="re-run the model every N closed-loop steps (1=every step)",
    )
    parser.add_argument(
        "--tracker_mode",
        choices=("mpc", "perfect"),
        default="perfect",
        help="ego advance: mpc=bicycle MPC tracker; perfect=place ego on predicted polyline",
    )
    parser.add_argument(
        "--abort_deviation_m",
        type=float,
        default=50.0,
        help="early-abort a segment (terminated='diverged') once the live ego strays this far "
        "(m) from the recorded GT pose for --abort_after consecutive steps (0=disabled). Set "
        "well above unstick_advance_m: unstick tries to save a merely-stuck rollout, abort "
        "gives up on one that won't recover.",
    )
    parser.add_argument(
        "--abort_after",
        type=int,
        default=30,
        help="consecutive steps of GT deviation (see --abort_deviation_m) before aborting",
    )
    parser.add_argument(
        "--abort_max_snaps",
        type=int,
        default=0,
        help="early-abort a segment once its unstick teleport has fired this many times "
        "(0=disabled) — repeated snapping is itself a sign of a bad rollout",
    )
    parser.add_argument(
        "--drop_objects",
        action="store_true",
        help="empty-world ablation: run with NO other traffic (dynamic neighbors + static "
        "objects zeroed each step); map/route kept. collision/near-miss go to 0 by design.",
    )


def add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--out_dir",
        "--output_dir",
        type=Path,
        default=None,
        help="output directory (full: default <model_dir>/closed_loop/<ts>; grouped: required)",
    )


def add_full_route_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--seg_len",
        type=int,
        default=6000,
        help="frames per route segment in full-route mode (~60s @10Hz)",
    )
    add_output_args(parser)


def add_grouped_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--classification_json_root",
        type=Path,
        default=None,
        help="Root of scenario_classification_json/ ({root}/{project}/{map}/{date}.json).",
    )
    parser.add_argument(
        "--classification_json",
        type=Path,
        default=None,
        help="Optional legacy single {date}.json override.",
    )
    parser.add_argument(
        "--areas",
        nargs="*",
        default=None,
        help="optional subset of area names in the classification JSON",
    )


def rollout_params_from_args(args: argparse.Namespace) -> RolloutParams:
    return RolloutParams(
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
        tracker_mode=args.tracker_mode,
        abort_deviation_m=args.abort_deviation_m,
        abort_after=args.abort_after,
        abort_max_snaps=args.abort_max_snaps,
        drop_objects=args.drop_objects,
    )


def create_evaluator_from_args(args: argparse.Namespace):
    """Return the closed-loop evaluator for ``--mode`` (subclass selection only)."""
    params = rollout_params_from_args(args)
    if args.mode == "grouped":
        if args.out_dir is None:
            raise SystemExit("grouped mode requires --out_dir")
        return GroupedMultiDateClosedLoopEvaluation.from_checkpoint(
            args.model_path,
            args.out_dir,
            GroupedClassificationParams(
                npz_root=Path(args.npz_root),
                classification_json_root=args.classification_json_root,
                classification_json=args.classification_json,
                dataset_name=getattr(args, "scenario_dataset_name", None),
                areas=args.areas,
            ),
            params=params,
            fps=float(args.fps),
            verbose=True,
            require_jobs=True,
        )

    out_dir = resolve_default_out_dir(args.model_path, args.out_dir)
    print(f"device: {args.device} | model: {args.model_path} | out: {out_dir}")
    config = ClosedLoopEvalConfig(
        out_dir=out_dir,
        params=params,
        fps=float(args.fps),
        verbose=True,
    )
    return FullRouteClosedLoopEvaluation.from_checkpoint(
        args.model_path,
        args.npz_root,
        config,
        seg_len=args.seg_len,
    )
