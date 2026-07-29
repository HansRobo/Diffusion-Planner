#!/usr/bin/env python3
"""Pairwise audit of a replay cache under short and full reward horizons.

The formal Original-DP planner predicts 80 steps (8 s), while an HDP/NAVSIM
reward recipe may score only 40 steps (4 s).  This tool re-scores the *same*
cached K trajectories against the same NPZ scene at both horizons.  It focuses
on groups that actually contain a positive-advantage AWR target, so it answers
the decision-relevant question: does a target selected at 4 s remain better
and feasible when its complete 8 s regression target is inspected?

It is an audit only.  It never edits the replay cache or training config.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from preference_optimization.utils import load_npz_data
from rlvr.autoresearch.tools.build_positive_anchor_replay_overlay import (
    compute_positive_anchor_weights,
)
from rlvr.awr import reward_compatible_data
from rlvr.reward import RewardConfig, compute_reward_batch


def _committed_count(rank_dir: Path, allow_partial_prefix: bool) -> int:
    manifest = rank_dir / "manifest.json"
    if manifest.is_file():
        return int(json.loads(manifest.read_text())["scene_count"])
    if not allow_partial_prefix:
        raise FileNotFoundError(
            f"{manifest}; pass --allow-partial-prefix only for an active strict writer"
        )
    with (rank_dir / "scene_paths.jsonl").open() as stream:
        return sum(1 for line in stream if line.strip())


def _load_paths(rank_dir: Path, count: int) -> list[str]:
    result: list[str] = []
    with (rank_dir / "scene_paths.jsonl").open() as stream:
        for line in stream:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                value = value.get("path") or value.get("npz_path")
            if not isinstance(value, str) or not value:
                raise ValueError(f"invalid scene path row in {rank_dir}: {line[:200]}")
            result.append(value)
            if len(result) == count:
                break
    if len(result) != count:
        raise RuntimeError(f"{rank_dir}: loaded {len(result)} paths, expected {count}")
    return result


def _reward_config(effective: dict[str, Any], horizon: int) -> RewardConfig:
    names = {item.name for item in fields(RewardConfig)}
    payload = {
        key: value
        for key, value in effective.get("reward", {}).items()
        if key in names
    }
    payload["reward_horizon_steps"] = int(horizon)
    return RewardConfig(**payload)


def _hard_reason(reward: Any) -> tuple[str, ...]:
    reasons: list[str] = []
    if reward.collision_step is not None:
        reasons.append("collision")
    if bool(reward.rb_crossing):
        reasons.append("road_border")
    if bool(reward.kinematic_violated):
        reasons.append("kinematic")
    if float(reward.red_light) < -0.5:
        reasons.append("red_light")
    return tuple(reasons)


def _dataset_bucket(scene_path: str) -> str:
    parts = set(Path(scene_path).parts)
    for name in ("x2", "x2_dev", "xx1_real_train", "xx1_psim", "j6_real_train"):
        if name in parts:
            return name
    return "other"


def _sample_active_rows(
    replay_root: Path,
    *,
    sample_groups: int,
    seed: int,
    allow_partial_prefix: bool,
    beta: float,
    margin: float,
) -> tuple[list[dict[str, Any]], int]:
    """Uniformly sample positive-overlay groups, balanced over replay ranks."""

    rank_dirs = sorted(path for path in replay_root.glob("rank_*") if path.is_dir())
    if not rank_dirs:
        raise FileNotFoundError(f"no rank directories under {replay_root}")
    rng = np.random.default_rng(seed)
    quota = int(np.ceil(sample_groups / len(rank_dirs)))
    rows: list[dict[str, Any]] = []
    active_total = 0
    for rank_dir in rank_dirs:
        count = _committed_count(rank_dir, allow_partial_prefix)
        rewards = np.load(rank_dir / "rewards.npy", mmap_mode="r")
        active_parts: list[np.ndarray] = []
        for lower in range(0, count, 65536):
            upper = min(count, lower + 65536)
            reward = np.asarray(rewards[lower:upper], dtype=np.float64)
            weights = compute_positive_anchor_weights(
                reward,
                beta=beta,
                margin=margin,
                behavior_anchor_weight=0.0,
                unsafe_behavior_anchor_weight=0.0,
            )
            local = np.flatnonzero(weights.sum(axis=1) > 0.0)
            if local.size:
                active_parts.append(local + lower)
        active = (
            np.concatenate(active_parts)
            if active_parts
            else np.empty(0, dtype=np.int64)
        )
        active_total += int(active.size)
        take = min(quota, int(active.size))
        if take == 0:
            continue
        chosen = np.sort(rng.choice(active, size=take, replace=False))
        paths = _load_paths(rank_dir, count)
        rank = int(rank_dir.name.split("_")[-1])
        for index in chosen:
            rows.append(
                {
                    "rank": rank,
                    "local_index": int(index),
                    "scene_path": paths[int(index)],
                }
            )
    if len(rows) > sample_groups:
        chosen = np.sort(rng.choice(len(rows), size=sample_groups, replace=False))
        rows = [rows[int(index)] for index in chosen]
    return rows, active_total


def audit(
    replay_root: Path,
    *,
    device: torch.device,
    sample_groups: int,
    seed: int,
    allow_partial_prefix: bool,
    short_horizon: int,
    full_horizon: int,
    beta: float,
    margin: float,
    top_examples: int,
) -> dict[str, Any]:
    replay_root = replay_root.expanduser().resolve(strict=True)
    effective = json.loads((replay_root.parent / "effective_config.json").read_text())
    short_config = _reward_config(effective, short_horizon)
    full_config = _reward_config(effective, full_horizon)
    selected, active_total = _sample_active_rows(
        replay_root,
        sample_groups=sample_groups,
        seed=seed,
        allow_partial_prefix=allow_partial_prefix,
        beta=beta,
        margin=margin,
    )
    if not selected:
        raise RuntimeError("no active positive-advantage groups available to audit")

    rank_arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    aggregates: dict[str, float] = {}
    category_aggregates: dict[str, dict[str, float]] = {}
    examples: list[dict[str, Any]] = []
    rescore_mismatches: list[dict[str, Any]] = []
    maximum_short_rescore_error = 0.0

    def add(name: str, value: float | int | bool) -> None:
        aggregates[name] = aggregates.get(name, 0.0) + float(value)

    def category_add(category: str, name: str, value: float | int | bool) -> None:
        bucket = category_aggregates.setdefault(category, {})
        bucket[name] = bucket.get(name, 0.0) + float(value)

    for ordinal, row in enumerate(selected, start=1):
        rank = int(row["rank"])
        index = int(row["local_index"])
        if rank not in rank_arrays:
            rank_dir = replay_root / f"rank_{rank:04d}"
            rank_arrays[rank] = (
                np.load(rank_dir / "trajectories.npy", mmap_mode="r"),
                np.load(rank_dir / "rewards.npy", mmap_mode="r"),
            )
        trajectories_mm, rewards_mm = rank_arrays[rank]
        category = _dataset_bucket(row["scene_path"])
        trajectories = torch.from_numpy(
            np.array(trajectories_mm[index], dtype=np.float32, copy=True)
        ).to(device)
        stored_short = np.asarray(rewards_mm[index], dtype=np.float64)
        data = load_npz_data(row["scene_path"], device)
        reward_data = reward_compatible_data(data)
        short_rewards = compute_reward_batch(trajectories, reward_data, short_config)
        full_rewards = compute_reward_batch(trajectories, reward_data, full_config)
        short = np.asarray([float(item.total) for item in short_rewards], dtype=np.float64)
        full = np.asarray([float(item.total) for item in full_rewards], dtype=np.float64)
        short_positive = short > short[0] + margin
        full_positive = full > full[0] + margin
        short_positive[0] = False
        full_positive[0] = False
        selected_slots = np.flatnonzero(short_positive)
        if not selected_slots.size:
            # The stored cache was selected before float32 serialization. Keep
            # this visible rather than silently changing the sample population.
            add("lost_short_positive_after_rescore", 1)
        retained_slots = selected_slots[full_positive[selected_slots]]
        selected_full_advantages = full[selected_slots] - full[0]
        newly_hard_slots: list[int] = []
        det_full_reasons = set(_hard_reason(full_rewards[0]))
        for slot in selected_slots:
            if not _hard_reason(short_rewards[int(slot)]) and _hard_reason(
                full_rewards[int(slot)]
            ):
                newly_hard_slots.append(int(slot))
            short_reasons = set(_hard_reason(short_rewards[int(slot)]))
            full_reasons = set(_hard_reason(full_rewards[int(slot)]))
            for reason in ("collision", "road_border", "kinematic", "red_light"):
                newly_present = reason not in short_reasons and reason in full_reasons
                add(
                    f"short_positive_slot_new_{reason}",
                    newly_present,
                )
                add(
                    f"short_positive_slot_unique_new_{reason}",
                    newly_present and reason not in det_full_reasons,
                )

        short_best = int(np.argmax(short))
        full_best = int(np.argmax(full))
        maximum_short_rescore_error = max(
            maximum_short_rescore_error,
            float(np.abs(short - stored_short).max()),
        )
        rescore_abs_error = np.abs(short - stored_short)
        rescore_mismatches.append(
            {
                **row,
                "mean_abs_error": float(rescore_abs_error.mean()),
                "max_abs_error": float(rescore_abs_error.max()),
                "stored_short_rewards": stored_short.tolist(),
                "recomputed_short_rewards": short.tolist(),
            }
        )
        add("groups", 1)
        add("stored_short_abs_error", float(np.abs(short - stored_short).mean()))
        add("short_positive_slots", int(selected_slots.size))
        add("full_retained_positive_slots", int(retained_slots.size))
        add(
            "full_nonworse_short_positive_slots",
            int((selected_full_advantages >= -1e-8).sum()),
        )
        add(
            "full_strictly_better_short_positive_slots",
            int((selected_full_advantages > 1e-8).sum()),
        )
        add(
            "short_positive_slot_full_advantage",
            float(selected_full_advantages.sum()),
        )
        add("all_short_positive_lost", bool(selected_slots.size and not retained_slots.size))
        add("best_index_changed", short_best != full_best)
        add(
            "short_best_became_hard",
            not _hard_reason(short_rewards[short_best])
            and bool(_hard_reason(full_rewards[short_best])),
        )
        short_best_reasons = set(_hard_reason(short_rewards[short_best]))
        full_best_reasons = set(_hard_reason(full_rewards[short_best]))
        for reason in ("collision", "road_border", "kinematic", "red_light"):
            newly_present = (
                reason not in short_best_reasons and reason in full_best_reasons
            )
            add(
                f"short_best_new_{reason}",
                newly_present,
            )
            add(
                f"short_best_unique_new_{reason}",
                newly_present and reason not in det_full_reasons,
            )
        add("selected_slots_newly_hard", len(newly_hard_slots))
        add("det_newly_hard", not _hard_reason(short_rewards[0]) and bool(_hard_reason(full_rewards[0])))
        add("short_best_full_advantage", float(full[short_best] - full[0]))
        add("short_best_full_nonworse", bool(full[short_best] >= full[0] - 1e-8))
        add("short_best_full_strictly_better", bool(full[short_best] > full[0] + 1e-8))
        add(
            "short_best_full_above_margin",
            bool(full[short_best] > full[0] + margin),
        )
        add("full_best_advantage", float(full[full_best] - full[0]))
        add("short_best_reward", float(short[short_best]))
        add("short_best_full_reward", float(full[short_best]))
        add("det_short_reward", float(short[0]))
        add("det_full_reward", float(full[0]))
        category_add(category, "groups", 1)
        category_add(category, "short_positive_slots", int(selected_slots.size))
        category_add(category, "retained_slots", int(retained_slots.size))
        category_add(
            category,
            "nonworse_slots",
            int((selected_full_advantages >= -1e-8).sum()),
        )
        category_add(category, "newly_hard_slots", len(newly_hard_slots))
        category_add(
            category,
            "short_best_nonworse",
            bool(full[short_best] >= full[0] - 1e-8),
        )
        category_add(
            category,
            "short_best_full_advantage",
            float(full[short_best] - full[0]),
        )
        category_add(
            category,
            "det_newly_hard",
            not _hard_reason(short_rewards[0]) and bool(_hard_reason(full_rewards[0])),
        )

        severity = float(
            (full[0] - full[short_best])
            + 1.0 * bool(_hard_reason(full_rewards[short_best]))
            + 0.25 * len(newly_hard_slots)
        )
        examples.append(
            {
                **row,
                "severity": severity,
                "short_best_slot": short_best,
                "full_best_slot": full_best,
                "short_positive_slots": selected_slots.tolist(),
                "full_retained_positive_slots": retained_slots.tolist(),
                "newly_hard_short_positive_slots": newly_hard_slots,
                "short_rewards": short.tolist(),
                "full_rewards": full.tolist(),
                "full_hard_reasons": [list(_hard_reason(item)) for item in full_rewards],
            }
        )
        if ordinal % 64 == 0:
            print(f"rescored {ordinal}/{len(selected)} active groups", flush=True)

    groups = aggregates["groups"]
    short_positive_slots = aggregates["short_positive_slots"]
    examples.sort(key=lambda item: item["severity"], reverse=True)
    rescore_mismatches.sort(key=lambda item: item["max_abs_error"], reverse=True)
    return {
        "contract": {
            "evidence": "same cached K trajectories and same scene rescored at paired horizons",
            "population": "groups active under the formal positive-advantage 4 s overlay",
            "not_evidence_of": "post-update checkpoint performance",
            "numeric_protocol": (
                "CUDA high-matmul TF32 matches formal mining when --device is cuda; "
                "CPU is suitable for aggregate horizon analysis but is not bitwise "
                "equivalent near TF32-sensitive reward geometry"
            ),
        },
        "replay_root": str(replay_root),
        "parameters": {
            "seed": seed,
            "sample_groups": sample_groups,
            "short_horizon_steps": short_horizon,
            "full_horizon_steps": full_horizon,
            "beta": beta,
            "positive_advantage_margin": margin,
            "device": str(device),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_tf32": (
                bool(torch.backends.cuda.matmul.allow_tf32)
                if device.type == "cuda"
                else False
            ),
        },
        "active_groups_in_committed_prefix": active_total,
        "audited_groups": int(groups),
        "raw_counts": {
            "short_positive_slots": int(short_positive_slots),
            "full_retained_positive_slots": int(
                aggregates["full_retained_positive_slots"]
            ),
            "full_nonworse_short_positive_slots": int(
                aggregates["full_nonworse_short_positive_slots"]
            ),
            "full_strictly_better_short_positive_slots": int(
                aggregates["full_strictly_better_short_positive_slots"]
            ),
            "all_short_positive_lost_groups": int(
                aggregates["all_short_positive_lost"]
            ),
            "best_candidate_index_changed_groups": int(
                aggregates["best_index_changed"]
            ),
            "short_best_became_hard_groups": int(
                aggregates["short_best_became_hard"]
            ),
            "short_positive_slots_newly_hard": int(
                aggregates["selected_slots_newly_hard"]
            ),
            "deterministic_newly_hard_groups": int(
                aggregates["det_newly_hard"]
            ),
        },
        "stored_vs_recomputed_short_mean_abs_error": aggregates[
            "stored_short_abs_error"
        ]
        / groups,
        "stored_vs_recomputed_short_max_abs_error": maximum_short_rescore_error,
        "lost_short_positive_after_rescore_group_fraction": aggregates.get(
            "lost_short_positive_after_rescore", 0.0
        )
        / groups,
        "positive_slot_retention_fraction_at_full_horizon": (
            aggregates["full_retained_positive_slots"] / short_positive_slots
            if short_positive_slots
            else 0.0
        ),
        "short_positive_slot_nonworse_fraction_at_full_horizon": (
            aggregates["full_nonworse_short_positive_slots"] / short_positive_slots
            if short_positive_slots
            else 0.0
        ),
        "short_positive_slot_strictly_better_fraction_at_full_horizon": (
            aggregates["full_strictly_better_short_positive_slots"]
            / short_positive_slots
            if short_positive_slots
            else 0.0
        ),
        "mean_short_positive_slot_full_horizon_advantage": (
            aggregates["short_positive_slot_full_advantage"] / short_positive_slots
            if short_positive_slots
            else 0.0
        ),
        "all_short_positive_lost_group_fraction": aggregates[
            "all_short_positive_lost"
        ]
        / groups,
        "best_candidate_index_changed_fraction": aggregates["best_index_changed"]
        / groups,
        "short_best_became_hard_fraction": aggregates["short_best_became_hard"]
        / groups,
        "short_best_new_event_fraction": {
            reason: aggregates.get(f"short_best_new_{reason}", 0.0) / groups
            for reason in ("collision", "road_border", "kinematic", "red_light")
        },
        "short_best_unique_new_event_vs_deterministic_fraction": {
            reason: aggregates.get(f"short_best_unique_new_{reason}", 0.0) / groups
            for reason in ("collision", "road_border", "kinematic", "red_light")
        },
        "short_positive_slot_new_hard_fraction": (
            aggregates["selected_slots_newly_hard"] / short_positive_slots
            if short_positive_slots
            else 0.0
        ),
        "short_positive_slot_new_event_fraction": {
            reason: (
                aggregates.get(f"short_positive_slot_new_{reason}", 0.0)
                / short_positive_slots
                if short_positive_slots
                else 0.0
            )
            for reason in ("collision", "road_border", "kinematic", "red_light")
        },
        "short_positive_slot_unique_new_event_vs_deterministic_fraction": {
            reason: (
                aggregates.get(f"short_positive_slot_unique_new_{reason}", 0.0)
                / short_positive_slots
                if short_positive_slots
                else 0.0
            )
            for reason in ("collision", "road_border", "kinematic", "red_light")
        },
        "deterministic_new_hard_fraction": aggregates["det_newly_hard"] / groups,
        "mean_short_best_full_horizon_advantage": aggregates[
            "short_best_full_advantage"
        ]
        / groups,
        "short_best_nonworse_fraction_at_full_horizon": aggregates[
            "short_best_full_nonworse"
        ]
        / groups,
        "short_best_strictly_better_fraction_at_full_horizon": aggregates[
            "short_best_full_strictly_better"
        ]
        / groups,
        "short_best_above_margin_fraction_at_full_horizon": aggregates[
            "short_best_full_above_margin"
        ]
        / groups,
        "mean_full_best_full_horizon_advantage": aggregates["full_best_advantage"]
        / groups,
        "mean_short_best_reward": aggregates["short_best_reward"] / groups,
        "mean_short_best_full_reward": aggregates["short_best_full_reward"] / groups,
        "mean_det_short_reward": aggregates["det_short_reward"] / groups,
        "mean_det_full_reward": aggregates["det_full_reward"] / groups,
        "by_dataset": {
            category: {
                "groups": int(values["groups"]),
                "positive_slot_retention_fraction_at_full_horizon": (
                    values["retained_slots"] / values["short_positive_slots"]
                    if values["short_positive_slots"]
                    else 0.0
                ),
                "short_positive_slot_nonworse_fraction_at_full_horizon": (
                    values["nonworse_slots"] / values["short_positive_slots"]
                    if values["short_positive_slots"]
                    else 0.0
                ),
                "short_positive_slot_new_hard_fraction": (
                    values["newly_hard_slots"] / values["short_positive_slots"]
                    if values["short_positive_slots"]
                    else 0.0
                ),
                "short_best_nonworse_fraction_at_full_horizon": (
                    values["short_best_nonworse"] / values["groups"]
                ),
                "mean_short_best_full_horizon_advantage": (
                    values["short_best_full_advantage"] / values["groups"]
                ),
                "deterministic_new_hard_fraction": (
                    values["det_newly_hard"] / values["groups"]
                ),
            }
            for category, values in sorted(category_aggregates.items())
        },
        "top_short_rescore_mismatch_examples": rescore_mismatches[:top_examples],
        "top_horizon_mismatch_examples": examples[:top_examples],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-groups", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--allow-partial-prefix", action="store_true")
    parser.add_argument("--short-horizon", type=int, default=40)
    parser.add_argument("--full-horizon", type=int, default=80)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--margin", type=float, default=0.01)
    parser.add_argument("--top-examples", type=int, default=32)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Reward rescore device. Use cuda for bitwise agreement with the formal TF32 miner.",
    )
    args = parser.parse_args()
    if args.sample_groups < 1:
        parser.error("--sample-groups must be positive")
    if not 0 < args.short_horizon < args.full_horizon:
        parser.error("require 0 < short horizon < full horizon")
    torch.set_num_threads(max(1, int(args.torch_threads)))
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("--device cuda requested but CUDA is unavailable")
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    summary = audit(
        args.replay_root,
        device=device,
        sample_groups=int(args.sample_groups),
        seed=int(args.seed),
        allow_partial_prefix=bool(args.allow_partial_prefix),
        short_horizon=int(args.short_horizon),
        full_horizon=int(args.full_horizon),
        beta=float(args.beta),
        margin=float(args.margin),
        top_examples=int(args.top_examples),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    hidden = {"top_horizon_mismatch_examples", "top_short_rescore_mismatch_examples"}
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key not in hidden},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
