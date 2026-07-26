#!/usr/bin/env python3
"""Fast paired evaluation for the full filtered original-DP validation set.

The training runner's monitor evaluation is intentionally small.  This tool
does the deployment-relevant check on every filtered validation scene using a
zero-noise, K=1 original-DP rollout, packed across eight independent GPU
workers.  It also supports ``--aggregate`` to combine the rank JSON files
without putting Python objects through NCCL.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlvr.awr import (
    AWRRolloutConfig,
    _slice_scene_data,
    rollout_and_score_scene_batch,
)
from rlvr.reward import RewardConfig
from rlvr.train_awr import (
    _compile_planner_modules,
    _configure_cuda_runtime,
    _iter_prefetched_eval_batches,
    _json_safe,
    _load_config,
    _load_scene_batch,
    _move_prefetched_eval_data,
    _trajectory_metrics,
    load_original_dp_checkpoint,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--args_path", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=192)
    parser.add_argument("--scene_load_workers", type=int, default=2)
    parser.add_argument("--prefetch_batches", type=int, default=0)
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=10,
        help="deployment solver steps; 10 is the original DP checkpoint default (training uses HDP's 5)",
    )
    parser.add_argument("--amp_dtype", choices=["off", "bf16", "fp16"], default="bf16")
    parser.add_argument(
        "--use_policy_state",
        action="store_true",
        help="evaluate checkpoint['model']; default evaluates deployable ema_state_dict",
    )
    parser.add_argument("--disable_compile", action="store_true")
    parser.add_argument("--compile_mode", default="default")
    parser.add_argument("--aggregate", action="store_true")
    return parser.parse_args()


def _summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
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
        "path_len",
        "ade",
        "fde",
    ]
    result: dict[str, float] = {"scene_count": float(len(rows))}
    for key in keys:
        values = [
            float(row[key])
            for row in rows
            if key in row and math.isfinite(float(row[key]))
        ]
        if values:
            result[f"mean_{key}"] = float(np.mean(values))
            result[f"p50_{key}"] = float(np.median(values))
    return result


def _configured_safe(reward: Any, config: RewardConfig) -> bool:
    return bool(
        reward.collision_step is None
        and (not config.rb_gate_enabled or not reward.rb_crossing)
        and (not config.lane_gate_enabled or not reward.lane_crossing)
        and (
            not config.static_collision_enabled
            or not config.sc_gate_enabled
            or not reward.static_crossing
        )
        and not reward.kinematic_violated
        and reward.red_light >= -0.5
    )


def _evaluate_rank(args: argparse.Namespace) -> None:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    _configure_cuda_runtime(device)
    random.seed(3407 + rank)
    np.random.seed(3407 + rank)
    torch.manual_seed(3407 + rank)

    raw_config, _, reward_config = _load_config(args.config)
    model, model_args = load_original_dp_checkpoint(
        args.model_path,
        device,
        args.args_path,
        use_ema=not args.use_policy_state,
    )
    model.eval()
    if not args.disable_compile and device.type == "cuda":
        _compile_planner_modules(
            model,
            {
                "enabled": True,
                "backend": "inductor",
                "mode": args.compile_mode,
                "fullgraph": False,
                "dynamic": False,
                "encoder": False,
                "decoder": True,
            },
            f"eval_{args.label}",
        )

    paths = json.loads(args.scenes.read_text())
    if not isinstance(paths, list) or not paths:
        raise ValueError(f"empty scene list: {args.scenes}")
    local_paths = paths[rank::world_size]
    # The zero-noise K=1 member is the deploy-time original DP policy.  No
    # HDP augmentation is used at evaluation: augmentation belongs to the
    # rollout distribution used to form AWR targets, not the policy being
    # compared on the full validation set.
    rollout_config = AWRRolloutConfig(
        n_trajectories=1,
        sample_steps=max(2, int(args.sample_steps)),
        noise_scale_range=(0.0, 0.0),
        beta=1.0,
        weight_clip=1e9,
        normalize_weights=False,
        min_group_std=1e-6,
        deterministic_first=True,
        hdp_trajectory_augmentation=False,
        inference_amp_dtype=args.amp_dtype,
    )

    batch_size = max(1, int(args.batch_size))
    rows: list[dict[str, Any]] = []
    prepared_batches = _iter_prefetched_eval_batches(
        local_paths,
        batch_size,
        max(1, int(args.scene_load_workers)),
        max(0, int(args.prefetch_batches)),
    )
    for start, chunk, model_chunk, prefetched_data in prepared_batches:
        real_count = len(chunk)
        if prefetched_data is None:
            data = _load_scene_batch(
                model_chunk,
                device,
                workers=max(1, int(args.scene_load_workers)),
            )
        else:
            data = _move_prefetched_eval_data(prefetched_data, device)
        rollouts = rollout_and_score_scene_batch(
            model, model_args, data, rollout_config, reward_config, device
        )
        for offset in range(real_count):
            scene_path = local_paths[start + offset]
            scene_data = _slice_scene_data(data, offset)
            rollout = rollouts[offset]
            reward = rollout.rewards[0]
            metrics = _trajectory_metrics(rollout.trajectories[0], scene_data)
            rows.append(
                {
                    "scene_index": start + offset,
                    "global_scene_index": rank + (start + offset) * world_size,
                    "scene_path": scene_path,
                    "det_reward": float(reward.total),
                    "det_collision": float(reward.collision_step is not None),
                    "det_rb_crossing": float(reward.rb_crossing),
                    "det_lane_crossing": float(reward.lane_crossing),
                    "det_kinematic_violation": float(reward.kinematic_violated),
                    "det_off_road_fraction": float(reward.off_road_fraction),
                    "det_progress": float(reward.progress),
                    "det_safety": float(reward.safety),
                    "det_centerline": float(reward.centerline),
                    "det_smoothness": float(reward.smoothness),
                    "configured_safe": float(_configured_safe(reward, reward_config)),
                    **metrics,
                }
            )
        del data, rollouts
        if device.type == "cuda" and (start // batch_size) % 16 == 0:
            torch.cuda.synchronize(device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rank_path = args.output_dir / f"rank_{rank:02d}.json"
    rank_path.write_text(
        json.dumps(
            {
                "label": args.label,
                "rank": rank,
                "world_size": world_size,
                "scenes": str(args.scenes),
                "config": str(args.config),
                "model": str(args.model_path),
                "args": str(args.args_path),
                "model_size_bytes": args.model_path.stat().st_size,
                "model_mtime_ns": args.model_path.stat().st_mtime_ns,
                "sample_steps": int(args.sample_steps),
                "candidate_count": 1,
                "noise_scale": 0.0,
                "checkpoint_payload": (
                    "model" if args.use_policy_state else "ema_state_dict"
                ),
                "amp_dtype": args.amp_dtype,
                "batch_size": batch_size,
                "scene_load_workers": max(1, int(args.scene_load_workers)),
                "prefetch_batches": max(0, int(args.prefetch_batches)),
                "scene_count": len(rows),
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"{args.label}: rank {rank} wrote {len(rows)} rows to {rank_path}", flush=True)


def _aggregate(args: argparse.Namespace) -> None:
    rank_files = sorted(args.output_dir.glob("rank_*.json"))
    if not rank_files:
        raise FileNotFoundError(f"no rank_*.json in {args.output_dir}")
    rows: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    for path in rank_files:
        payload = json.loads(path.read_text())
        payloads.append(payload)
        rows.extend(payload.get("rows", []))

    world_sizes = {int(payload.get("world_size", -1)) for payload in payloads}
    if len(world_sizes) != 1 or next(iter(world_sizes)) <= 0:
        raise RuntimeError(f"inconsistent/invalid world_size across rank outputs: {world_sizes}")
    world_size = next(iter(world_sizes))
    ranks = [int(payload.get("rank", -1)) for payload in payloads]
    if len(rank_files) != world_size or sorted(ranks) != list(range(world_size)):
        raise RuntimeError(
            f"incomplete rank set: files={len(rank_files)} world_size={world_size} ranks={sorted(ranks)}"
        )

    metadata_keys = (
        "label",
        "scenes",
        "config",
        "model",
        "args",
        "model_size_bytes",
        "model_mtime_ns",
        "sample_steps",
        "candidate_count",
        "noise_scale",
        "checkpoint_payload",
        "amp_dtype",
        "batch_size",
        "scene_load_workers",
        "prefetch_batches",
    )
    metadata: dict[str, Any] = {}
    for key in metadata_keys:
        values = {str(payload.get(key, "")) for payload in payloads}
        if len(values) != 1:
            raise RuntimeError(f"mixed {key} values across rank outputs: {sorted(values)}")
        metadata[key] = next(iter(values))

    expected_paths = json.loads(args.scenes.read_text())
    if not isinstance(expected_paths, list) or not expected_paths:
        raise ValueError(f"empty/invalid expected scene list: {args.scenes}")
    actual_paths = [str(row.get("scene_path", "")) for row in rows]
    expected_counts = Counter(map(str, expected_paths))
    actual_counts = Counter(actual_paths)
    if actual_counts != expected_counts:
        missing = list((expected_counts - actual_counts).elements())[:10]
        extra = list((actual_counts - expected_counts).elements())[:10]
        raise RuntimeError(
            "rank outputs do not exactly cover the requested scene multiset: "
            f"expected={len(expected_paths)} actual={len(actual_paths)} "
            f"missing_sample={missing} extra_sample={extra}"
        )
    rows.sort(key=lambda row: row.get("scene_path", ""))
    summary = _summary(rows)
    summary["mean_configured_safe"] = float(
        np.mean([float(row["configured_safe"]) for row in rows])
    ) if rows else float("nan")
    payload = {**metadata, "summary": summary, "rows": rows}
    (args.output_dir / "scenes.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2) + "\n"
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "rank_files": [str(path) for path in rank_files],
                "world_size": world_size,
                "scene_contract_verified": True,
                "expected_scene_count": len(expected_paths),
                "actual_scene_count": len(rows),
                **metadata,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(_json_safe(summary), sort_keys=True))


def main() -> None:
    args = _parse_args()
    if args.aggregate:
        _aggregate(args)
    else:
        _evaluate_rank(args)


if __name__ == "__main__":
    main()
