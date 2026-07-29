#!/usr/bin/env python3
"""Evaluate an HDP/R2LPL-style safe override gate on paired scene rows.

The base rows identify scenes where the original DP closed-loop rollout has a
configured safety event.  Only those scenes are re-mined with a K-candidate
AWR policy.  The gate replaces the base trajectory only when at least one
candidate passes the configured hard safety gates; the highest-reward safe
candidate is selected.  Non-event scenes are copied from the base rows.

This is intentionally an offline validation tool.  The base event label uses
the recorded neighbor future, so a deployment gate must replace it with an
online risk predictor or the same semi-closed-loop reproducer used by R2LPL.
The tool nevertheless measures the exact upper bound of the safe-repair
mechanism without silently claiming that an oracle label is deployable.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from planner_metrics.config import RewardConfig
from rlvr.awr import (
    AWRRolloutConfig,
    load_original_dp_checkpoint,
    load_scene,
    rollout_and_score_scene,
)
from rlvr.train_awr import _trajectory_metrics


def _finite(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


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


def _load_reward_config(path: Path, profile: str | None) -> RewardConfig:
    raw = json.loads(path.read_text())
    raw = dict(raw.get("reward", raw.get("reward_config", raw)))
    fields = {field.name for field in dataclasses.fields(RewardConfig)}
    config = RewardConfig(**{key: value for key, value in raw.items() if key in fields})
    if profile is not None:
        config.reward_profile = profile
    return config


def _event(row: dict[str, Any]) -> bool:
    return any(
        _finite(row.get(key)) > 0.5
        for key in ("det_collision", "det_rb_crossing", "det_kinematic_violation")
    )


def _row_from_candidate(
    base: dict[str, Any],
    scene_path: str,
    trajectory: torch.Tensor,
    reward: Any,
    data: dict[str, torch.Tensor],
    safe_count: int,
    candidate_count: int,
    best_safe_rank: int,
) -> dict[str, Any]:
    metrics = _trajectory_metrics(trajectory, data)
    return {
        "scene_index": base.get("scene_index"),
        "scene_path": scene_path,
        "source": "awr_safe_candidate",
        "base_event": True,
        "selected_candidate_rank": int(best_safe_rank),
        "candidate_count": int(candidate_count),
        "safe_candidate_count": int(safe_count),
        "det_reward": float(reward.total),
        "det_collision": float(reward.collision_step is not None),
        "det_rb_crossing": float(bool(reward.rb_crossing)),
        "det_kinematic_violation": float(bool(reward.kinematic_violated)),
        "det_lane_crossing": float(bool(reward.lane_crossing)),
        "det_safety": float(reward.safety),
        "det_progress": float(reward.progress),
        "det_centerline": float(reward.centerline),
        "det_smoothness": float(reward.smoothness),
        **metrics,
    }


def _summary(rows: list[dict[str, Any]], replaced: int, base_events: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "scene_count": len(rows),
        "base_event_count": int(base_events),
        "replacement_count": int(replaced),
        "replacement_rate_among_base_events": float(replaced / max(base_events, 1)),
        "gate_type": "offline_base_event_then_best_safe_candidate",
    }
    keys = [
        "det_reward",
        "det_collision",
        "det_rb_crossing",
        "det_kinematic_violation",
        "det_lane_crossing",
        "det_safety",
        "det_progress",
        "det_centerline",
        "det_smoothness",
        "path_len",
        "ade",
        "fde",
    ]
    for key in keys:
        values = [_finite(row.get(key)) for row in rows]
        values = [value for value in values if math.isfinite(value)]
        if values:
            result[f"mean_{key}"] = float(np.mean(values))
            result[f"p50_{key}"] = float(np.median(values))
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_scenes", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--args_path", type=Path, required=True)
    parser.add_argument("--reward_config", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--sample_steps", type=int, default=5)
    parser.add_argument("--noise_min", type=float, default=0.5)
    parser.add_argument("--noise_max", type=float, default=0.5)
    parser.add_argument("--reward_profile", default="hdp_pdm")
    parser.add_argument("--seed", type=int, default=3408)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    base_rows = json.loads(args.base_scenes.read_text())
    if not isinstance(base_rows, list):
        raise ValueError(f"expected a list of scene rows: {args.base_scenes}")
    reward_config = _load_reward_config(args.reward_config, args.reward_profile)
    model, model_args = load_original_dp_checkpoint(
        args.model_path, device, args.args_path, use_ema=True
    )
    rollout_config = AWRRolloutConfig(
        n_trajectories=args.K,
        sample_steps=args.sample_steps,
        noise_scale_range=(args.noise_min, args.noise_max),
        deterministic_first=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_path = {str(row["scene_path"]): row for row in base_rows}
    output_rows: list[dict[str, Any]] = []
    replacement_rows: list[dict[str, Any]] = []
    base_event_count = sum(_event(row) for row in base_rows)
    replaced = 0

    for index, base in enumerate(base_rows):
        scene_path = str(base["scene_path"])
        if not _event(base):
            copied = dict(base)
            copied["source"] = "original_dp_base"
            copied["base_event"] = False
            output_rows.append(copied)
            continue
        print(f"[{index + 1}/{len(base_rows)}] remine {Path(scene_path).name}", flush=True)
        data = load_scene(scene_path, device)
        rollout = rollout_and_score_scene(
            model, model_args, data, rollout_config, reward_config, device
        )
        safe_indices = [
            index
            for index, reward in enumerate(rollout.rewards)
            if _configured_safe(reward, reward_config)
        ]
        if safe_indices:
            best_index = max(
                safe_indices, key=lambda candidate: float(rollout.rewards[candidate].total)
            )
            best_reward = rollout.rewards[best_index]
            # Ranks are one-indexed in the visualization and paper narrative.
            selected = _row_from_candidate(
                base,
                scene_path,
                rollout.trajectories[best_index],
                best_reward,
                data,
                len(safe_indices),
                len(rollout.rewards),
                best_index + 1,
            )
            selected["best_safe_reward"] = float(best_reward.total)
            replacement_rows.append(
                {
                    "base": base,
                    "selected": selected,
                    "safe_indices": safe_indices,
                    "trajectories": rollout.trajectories.detach().cpu().numpy(),
                    "rewards": [float(reward.total) for reward in rollout.rewards],
                    "selected_index": int(best_index),
                    "scene_path": scene_path,
                }
            )
            output_rows.append(selected)
            replaced += 1
        else:
            copied = dict(base)
            copied["source"] = "original_dp_base_no_safe_repair"
            copied["base_event"] = True
            copied["candidate_count"] = len(rollout.rewards)
            copied["safe_candidate_count"] = 0
            output_rows.append(copied)

    output_rows.sort(key=lambda row: int(row.get("scene_index", 0)))
    summary = _summary(output_rows, replaced, base_event_count)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "scenes.json").write_text(
        json.dumps(output_rows, indent=2, sort_keys=True) + "\n"
    )
    # Save all repaired groups, but only for the base-event subset.  These
    # compact NPZs are consumed by scene visualizers and make the gate result
    # independently auditable without rerunning the model.
    for index, item in enumerate(replacement_rows):
        np.savez_compressed(
            args.output_dir / f"repair_{index:04d}.npz",
            trajectories=item["trajectories"],
            rewards=np.asarray(item["rewards"], dtype=np.float32),
            selected_index=np.asarray(item["selected_index"], dtype=np.int64),
            safe_indices=np.asarray(item["safe_indices"], dtype=np.int64),
        )
    manifest = [
        {
            "scene_path": item["scene_path"],
            "repair_npz": str(args.output_dir / f"repair_{index:04d}.npz"),
            "selected_index": item["selected_index"],
            "safe_indices": item["safe_indices"],
        }
        for index, item in enumerate(replacement_rows)
    ]
    (args.output_dir / "repairs.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
