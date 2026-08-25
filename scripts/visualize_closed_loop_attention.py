"""Render Fusion/Decoder-rollout attention over a SELF-DRIVEN closed-loop rollout.

The existing ``visualize_neighbor_attention(_video)``/``visualize_all_token_attention
(_video)``/``visualize_attention_rollout(_video)`` tools all replay recorded NPZ log
entries open-loop: the "ego" position across frames is just what actually happened in
the log, and the model never controls anything. This tool instead drives the ego through
``scenario_generation.reproducer_rollout_token_importance.run_segment`` (the same
closed-loop Perception-Reproducer step machinery ``reproducer_rollout.py`` uses for
closed-loop training/validation) — the model's own predicted trajectory advances the ego
one step at a time (perfect tracking), recorded neighbors are replayed keyed on the live
ego pose, and THIS tool captures + renders attention at every step of that self-driven
rollout instead of open-loop log replay.

Fusion/Decoder attention capture reuses the existing, unmodified hooks
(``token_analysis_common.patch_fusion`` / ``visualize_attention_rollout.patch_decoder``);
per-step rendering reuses the existing, unmodified ``draw_report`` functions from
``visualize_neighbor_attention.py`` / ``visualize_all_token_attention.py``; per-step
scoring reuses ``all_token_attention`` / ``token_records`` / ``attention_rollout``. Only
the closed-loop plumbing (running the rollout, wiring the callback, assembling the video)
is new.

Ground-truth futures do not exist here (the ego is self-driven) — the reproducer's
per-step model-input dict simply has no ``ego_agent_future``/``neighbor_agents_future``
keys, and the open-loop drawing code (``diffusion_planner/utils/visualize_input.py``)
already guards those as optional, so nothing is drawn for them; no code changes needed.

Example::

  uv run python scripts/visualize_closed_loop_attention.py \
    --model_path /path/to/best_model.pth \
    --npz_root /path/to/npz_root \
    --attention_mode rollout --chunk_len 200 --device cuda \
    --out_dir /tmp/closed_loop_attention
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scenario_generation.closed_loop_eval import build_mp4
from scenario_generation.reproducer_rollout_token_importance import run_segment
from scenario_generation.route_timeline import RouteTimeline, group_routes
from scenario_generation.simulate import load_model
from token_analysis_common import find_fusion, patch_fusion
from visualize_all_token_attention import (
    all_token_attention,
    class_summary,
    draw_report as draw_all_report,
    token_records as all_token_records,
)
from visualize_attention_rollout import attention_rollout, patch_decoder
from visualize_neighbor_attention import (
    NEIGHBOR_OFFSET,
    draw_report as draw_neighbor_report,
    ego_neighbor_attention,
    token_records as neighbor_token_records,
)

try:
    from diffusion_planner.dimensions import MAX_NUM_NEIGHBORS
except ImportError:  # pragma: no cover - fallback if the constant moves
    MAX_NUM_NEIGHBORS = 320

ATTENTION_MODES = ("neighbor", "all_token", "rollout")


def _unbatch(np_dict: dict) -> dict:
    """Drop the leading batch dim of 1 -> plain per-scene numpy dict.

    Matches the layout ``token_records``/``token_layout``/``movement_and_turn`` expect
    (the same layout as one raw ``DiffusionPlannerData`` sample), whereas the reproducer's
    ``np_dict`` (from ``build_input_np``) is always ``[1, ...]`` batched.
    """
    return {k: np.asarray(v)[0] for k, v in np_dict.items()}


def _closed_loop_batch(np_dict: dict) -> dict:
    """Wrap an already-batched ``[1, ...]`` numpy dict as torch tensors for ``draw_report``.

    Unlike ``visualize_neighbor_attention.sample_to_batch`` (which adds the batch dim to
    an unbatched sample), the reproducer's ``np_dict`` already has it — no unsqueeze here.
    """
    return {k: torch.as_tensor(v) for k, v in np_dict.items() if isinstance(v, np.ndarray)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", required=True, help="checkpoint .pth (args.json alongside it)")
    p.add_argument("--npz_root", required=True, help="dir tree of route NPZ frames")
    p.add_argument("--sidecar_root", default=None, help="pose JSON sidecar tree, if separate")
    p.add_argument("--route", default=None, help="route key to roll out (default: first route)")
    p.add_argument("--start", type=int, default=None, help="segment start frame (default: 0)")
    p.add_argument("--end", type=int, default=None, help="segment end frame (default: route end)")
    p.add_argument(
        "--chunk_len",
        type=int,
        default=None,
        help="if set (and --start/--end unset), roll out only the first --chunk_len frames",
    )
    p.add_argument(
        "--attention_mode",
        choices=ATTENTION_MODES,
        default="all_token",
        help="'neighbor': ego-query attention on neighbor tokens only; 'all_token': ego-query "
        "attention on every valid Fusion token; 'rollout': Decoder-to-input attention rollout "
        "(needs a real diffusion decode each step, slower)",
    )
    p.add_argument("--layer", default="mean", help="'mean', 'last', or a zero-based Fusion layer")
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--view_range", type=float, default=80.0)
    p.add_argument("--colormap", default="plasma")
    p.add_argument("--marker_size_min", type=float, default=25.0)
    p.add_argument("--marker_size_max", type=float, default=700.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--search_radius", type=float, default=1.5)
    p.add_argument("--near_miss_thresh", type=float, default=0.5)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--unstick_after", type=int, default=300)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--video_width", type=int, default=1920)
    p.add_argument("--video_height", type=int, default=1080)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--output_name", default="closed_loop_attention")
    p.add_argument("--keep_frames", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.colormap not in __import__("matplotlib").colormaps:
        raise ValueError(f"unknown Matplotlib colormap: {args.colormap}")

    out_dir = Path(args.out_dir).resolve()
    frames_dir = out_dir / f"{args.output_name}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(Path(args.npz_root).rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no .npz under {args.npz_root}")
    routes = group_routes(paths)
    route_key = args.route or sorted(routes)[0]
    if route_key not in routes:
        raise KeyError(f"route {route_key!r} not found; available: {sorted(routes)[:5]}...")
    tl = RouteTimeline(routes[route_key], sidecar_dir=args.sidecar_root)

    start = args.start if args.start is not None else 0
    end = args.end if args.end is not None else (start + args.chunk_len if args.chunk_len else len(tl))
    end = min(end, len(tl))
    print(f"route={route_key} segment=[{start},{end}) ({end - start} frames)", flush=True)

    model, model_args = load_model(Path(args.model_path), args.device)
    fusion_store: list[dict] = []
    decoder_store: list[dict] = []
    patch_fusion(find_fusion(model.encoder), fusion_store)
    if args.attention_mode == "rollout":
        patch_decoder(model.decoder, decoder_store)

    jsonl_path = out_dir / f"{args.output_name}.jsonl"
    jsonl_file = jsonl_path.open("w")
    global_vmax = 0.0  # first pass would require buffering every frame; instead rescale
    # per-frame like the open-loop tools' non-video (single-scene) mode does, since a
    # closed-loop rollout's attention distribution isn't known ahead of time without a
    # second pass. Document this as a known difference from the open-loop *_video tools,
    # which use one global scale computed in a first analysis pass.

    def on_step(k: int, np_dict: dict, outputs: dict) -> None:
        sample = _unbatch(np_dict)
        batch = _closed_loop_batch(np_dict)
        ego_pred = outputs["prediction"][0, 0].cpu().numpy()
        frame_path = frames_dir / f"frame_{k:06d}.png"
        row: dict = {"step": k}

        if args.attention_mode == "neighbor":
            scores = ego_neighbor_attention(fusion_store, args.layer)[0].cpu().numpy()
            layer_scores = np.stack(
                [
                    record["weights"][0, 0, NEIGHBOR_OFFSET : NEIGHBOR_OFFSET + MAX_NUM_NEIGHBORS]
                    .cpu()
                    .numpy()
                    for record in fusion_store
                ]
            )
            records = neighbor_token_records(sample, scores, layer_scores)
            draw_neighbor_report(
                batch,
                records,
                k,
                f"{route_key}[{start}:{end}]",
                0.0,
                0.0,
                args.layer,
                args.top_k,
                args.view_range,
                args.colormap,
                args.marker_size_min,
                args.marker_size_max,
                frame_path,
                title_prefix="Closed-Loop Neighbor Attention",
                sample=sample,
                ego_pred=ego_pred,
            )
            row["valid_neighbor_count"] = len(records)
            row["top_tokens"] = records[: args.top_k]
        else:
            if args.attention_mode == "rollout":
                scores, valid = attention_rollout(fusion_store, decoder_store)
                scores = scores.cpu().numpy()
                valid = valid.cpu().numpy()
                title_prefix = "Closed-Loop Decoder-to-Input Attention Rollout"
                attention_label = "Decoder-to-input rollout"
            else:
                scores = all_token_attention(fusion_store, args.layer)[0].cpu().numpy()
                valid = (~fusion_store[0]["mask"][0]).cpu().numpy()
                title_prefix = "Closed-Loop All-Token Attention"
                attention_label = "Fusion attention from ego query"
            records = all_token_records(sample, scores, valid)
            draw_all_report(
                batch,
                records,
                k,
                f"{route_key}[{start}:{end}]",
                0.0,
                0.0,
                args.layer,
                args.top_k,
                args.view_range,
                args.colormap,
                args.marker_size_min,
                args.marker_size_max,
                frame_path,
                attention_label=attention_label,
                title_prefix=title_prefix,
                sample=sample,
                ego_pred=ego_pred,
            )
            row["valid_token_count"] = len(records)
            row["class_summary"] = class_summary(records)
            row["top_tokens"] = records[: args.top_k]

        jsonl_file.write(json.dumps(row, default=float) + "\n")
        fusion_store.clear()
        decoder_store.clear()

    result = run_segment(
        model,
        model_args,
        tl,
        start,
        end,
        device=args.device,
        near_miss_thresh=args.near_miss_thresh,
        search_radius=args.search_radius,
        warmup_steps=args.warmup_steps,
        unstick_after=args.unstick_after,
        step_callback=on_step,
    )
    jsonl_file.close()

    print(f"rollout finished: {result}", flush=True)
    out_mp4 = out_dir / f"{args.output_name}.mp4"
    build_mp4(frames_dir, out_mp4, args.fps)
    print(f"wrote {out_mp4}")
    print(f"wrote {jsonl_path}")
    with (out_dir / f"{args.output_name}_summary.json").open("w") as f:
        json.dump(
            {
                "route": route_key,
                "segment": [start, end],
                "attention_mode": args.attention_mode,
                "rollout_metrics": result,
            },
            f,
            indent=2,
        )
    if not args.keep_frames:
        for png in frames_dir.glob("*.png"):
            png.unlink()
        frames_dir.rmdir()


if __name__ == "__main__":
    main()
