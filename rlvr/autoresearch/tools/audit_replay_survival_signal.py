#!/usr/bin/env python3
"""Audit whether PlannerRFT-style survival reward revives all-zero AWR groups.

The tool samples only groups whose stored gate rewards are all zero, reloads
the exact scene through the canonical T4 loader, and re-scores the same cached
K trajectories under gate and survival shaping.  It is read-only and never
changes a replay cache or training configuration.

Besides signal recovery, the audit measures progress and path-length changes
of the survival winner relative to candidate 0.  This makes the main failure
mode visible: a reward can delay failure by learning to stop rather than by
finding a genuinely better maneuver.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from preference_optimization.utils import load_npz_data
from rlvr.autoresearch.tools.audit_replay_reward_horizon import (
    _committed_count,
    _dataset_bucket,
    _hard_reason,
    _reward_config,
)
from rlvr.awr import reward_compatible_data
from rlvr.reward import compute_reward_batch, compute_subscores_batch


def _select_constrained_terminal_repair(
    *,
    terminal_steps: np.ndarray,
    survival_total: np.ndarray,
    progress: np.ndarray,
    path_length: np.ndarray,
    hard_reasons: list[set[str]],
    min_delay_steps: int,
    progress_tolerance: float,
    path_length_tolerance_m: float,
    require_no_new_hard_reason: bool,
) -> int | None:
    """Select a delayed-failure candidate without rewarding stopping or event swaps.

    Candidate zero is the deployed behavior.  This helper intentionally
    requires strict terminal-time improvement; ties cannot become training
    targets merely because a secondary quality score changed.
    """

    if terminal_steps.ndim != 1:
        raise ValueError("terminal_steps must be one-dimensional")
    size = int(terminal_steps.shape[0])
    if not all(len(value) == size for value in (survival_total, progress, path_length)):
        raise ValueError("constrained survival arrays must have equal length")
    if len(hard_reasons) != size:
        raise ValueError("hard_reasons must have one set per candidate")
    if size < 1:
        return None

    baseline_reasons = hard_reasons[0]
    eligible: list[int] = []
    for index in range(1, size):
        if int(terminal_steps[index]) < int(terminal_steps[0]) + int(min_delay_steps):
            continue
        if float(progress[index]) < float(progress[0]) - float(progress_tolerance):
            continue
        if float(path_length[index]) < float(path_length[0]) - float(
            path_length_tolerance_m
        ):
            continue
        if require_no_new_hard_reason and not hard_reasons[index].issubset(
            baseline_reasons
        ):
            continue
        eligible.append(index)
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda index: (
            int(terminal_steps[index]),
            float(survival_total[index]),
            float(progress[index]),
            -index,
        ),
    )


def _selected_paths(rank_dir: Path, count: int, indices: np.ndarray) -> dict[int, str]:
    wanted = {int(value) for value in indices.tolist()}
    result: dict[int, str] = {}
    with (rank_dir / "scene_paths.jsonl").open() as stream:
        committed = -1
        for line in stream:
            if not line.strip():
                continue
            committed += 1
            if committed >= count:
                break
            if committed not in wanted:
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                value = value.get("path") or value.get("npz_path")
            if not isinstance(value, str) or not value:
                raise ValueError(f"invalid scene path at {rank_dir}:{committed}")
            result[committed] = value
            if len(result) == len(wanted):
                break
    missing = wanted.difference(result)
    if missing:
        raise RuntimeError(f"{rank_dir}: missing {len(missing)} selected paths")
    return result


def _sample_all_zero_rows(
    replay_root: Path,
    *,
    sample_groups: int,
    seed: int,
    allow_partial_prefix: bool,
) -> tuple[list[dict[str, Any]], int, int]:
    rank_dirs = sorted(path for path in replay_root.glob("rank_*") if path.is_dir())
    if not rank_dirs:
        raise FileNotFoundError(f"no rank directories under {replay_root}")
    rng = np.random.default_rng(seed)
    quota = int(np.ceil(sample_groups / len(rank_dirs)))
    rows: list[dict[str, Any]] = []
    all_zero_total = 0
    committed_total = 0
    for rank_dir in rank_dirs:
        count = _committed_count(rank_dir, allow_partial_prefix)
        committed_total += count
        rewards = np.load(rank_dir / "rewards.npy", mmap_mode="r")
        zero_parts: list[np.ndarray] = []
        for lower in range(0, count, 65536):
            upper = min(count, lower + 65536)
            reward = np.asarray(rewards[lower:upper], dtype=np.float32)
            local = np.flatnonzero(np.max(np.abs(reward), axis=1) <= 1e-8)
            if local.size:
                zero_parts.append(local + lower)
        all_zero = (
            np.concatenate(zero_parts)
            if zero_parts
            else np.empty(0, dtype=np.int64)
        )
        all_zero_total += int(all_zero.size)
        take = min(quota, int(all_zero.size))
        if not take:
            continue
        chosen = np.sort(rng.choice(all_zero, size=take, replace=False))
        paths = _selected_paths(rank_dir, count, chosen)
        rank = int(rank_dir.name.split("_")[-1])
        rows.extend(
            {
                "rank": rank,
                "local_index": int(index),
                "scene_path": paths[int(index)],
            }
            for index in chosen
        )
    if len(rows) > sample_groups:
        keep = np.sort(rng.choice(len(rows), size=sample_groups, replace=False))
        rows = [rows[int(index)] for index in keep]
    return rows, all_zero_total, committed_total


def _path_length(trajectories: torch.Tensor, horizon: int) -> np.ndarray:
    xy = trajectories[:, :horizon, :2].cpu().numpy().astype(np.float64)
    if xy.shape[1] < 2:
        return np.zeros(xy.shape[0], dtype=np.float64)
    return np.linalg.norm(np.diff(xy, axis=1), axis=-1).sum(axis=1)


def audit(
    replay_root: Path,
    *,
    sample_groups: int,
    seed: int,
    allow_partial_prefix: bool,
    horizon: int,
    top_examples: int,
    min_delay_steps: int,
    progress_tolerance: float,
    path_length_tolerance_m: float,
    require_no_new_hard_reason: bool,
) -> dict[str, Any]:
    replay_root = replay_root.expanduser().resolve(strict=True)
    effective = json.loads((replay_root.parent / "effective_config.json").read_text())
    gate_config = _reward_config(effective, horizon)
    survival_config = _reward_config(effective, horizon)
    gate_config.reward_mode = "gate"
    survival_config.reward_mode = "survival"
    rows, all_zero_total, committed_total = _sample_all_zero_rows(
        replay_root,
        sample_groups=sample_groups,
        seed=seed,
        allow_partial_prefix=allow_partial_prefix,
    )
    if not rows:
        raise RuntimeError("no all-zero groups available to audit")

    rank_trajectories: dict[int, np.ndarray] = {}
    sums: dict[str, float] = {}
    by_dataset: dict[str, dict[str, float]] = {}
    examples: list[dict[str, Any]] = []

    def add(name: str, value: float | int | bool) -> None:
        sums[name] = sums.get(name, 0.0) + float(value)

    def bucket_add(bucket: str, name: str, value: float | int | bool) -> None:
        values = by_dataset.setdefault(bucket, {})
        values[name] = values.get(name, 0.0) + float(value)

    for ordinal, row in enumerate(rows, start=1):
        rank = int(row["rank"])
        index = int(row["local_index"])
        if rank not in rank_trajectories:
            rank_trajectories[rank] = np.load(
                replay_root / f"rank_{rank:04d}" / "trajectories.npy",
                mmap_mode="r",
            )
        trajectories = torch.from_numpy(
            np.array(rank_trajectories[rank][index], dtype=np.float32, copy=True)
        )
        data = load_npz_data(row["scene_path"], torch.device("cpu"))
        reward_data = reward_compatible_data(data)
        gate = compute_reward_batch(trajectories, reward_data, gate_config)
        survival = compute_reward_batch(trajectories, reward_data, survival_config)
        subscores = compute_subscores_batch(
            trajectories[:, :horizon], reward_data, gate_config
        )
        gate_total = np.asarray([float(item.total) for item in gate], dtype=np.float64)
        survival_total = np.asarray(
            [float(item.total) for item in survival], dtype=np.float64
        )
        gate_error = float(np.max(np.abs(gate_total)))
        if gate_error > 1e-6:
            raise RuntimeError(
                f"stored all-zero group re-scored non-zero at rank={rank}, "
                f"index={index}: max={gate_error}"
            )
        winner = int(np.argmax(survival_total))
        spread = float(np.max(survival_total) - np.min(survival_total))
        active = spread > 1e-6
        path_length = _path_length(trajectories, horizon)
        path_delta = float(path_length[winner] - path_length[0])
        progress_delta = float(survival[winner].progress - survival[0].progress)
        winner_gain = float(survival_total[winner] - survival_total[0])
        winner_reasons = _hard_reason(survival[winner])
        det_reasons = _hard_reason(survival[0])
        # Faithful terminal-prefix signal: only the number of valid steps
        # before the first collision/road-border event contributes.  Unlike
        # the repository's historical survival mode, this cannot be improved
        # by accumulating progress after the trajectory has already failed.
        terminal_steps = np.full(len(survival), horizon, dtype=np.int64)
        unsupported_kinematic_only = np.zeros(len(survival), dtype=bool)
        for slot in range(len(survival)):
            collision_step = subscores["collision_step"][slot]
            rb_step = subscores["rb_crossing_steps"][slot]
            if collision_step is not None:
                terminal_steps[slot] = min(terminal_steps[slot], int(collision_step))
            if rb_step is not None:
                terminal_steps[slot] = min(terminal_steps[slot], int(rb_step))
            unsupported_kinematic_only[slot] = bool(
                float(subscores["kinematic_gate"][slot]) < 0.5
                and collision_step is None
                and rb_step is None
            )
        terminal_reward = terminal_steps.astype(np.float64) / float(horizon)
        terminal_winner = int(np.argmax(terminal_reward))
        terminal_spread = float(terminal_reward.max() - terminal_reward.min())
        terminal_active = terminal_spread > 1e-8
        terminal_delay_steps = int(terminal_steps[terminal_winner] - terminal_steps[0])
        terminal_path_delta = float(
            path_length[terminal_winner] - path_length[0]
        )
        terminal_progress_delta = float(
            survival[terminal_winner].progress - survival[0].progress
        )
        candidate_reasons = [set(_hard_reason(item)) for item in survival]
        candidate_progress = np.asarray(
            [float(item.progress) for item in survival], dtype=np.float64
        )
        constrained_winner = _select_constrained_terminal_repair(
            terminal_steps=terminal_steps,
            survival_total=survival_total,
            progress=candidate_progress,
            path_length=path_length,
            hard_reasons=candidate_reasons,
            min_delay_steps=min_delay_steps,
            progress_tolerance=progress_tolerance,
            path_length_tolerance_m=path_length_tolerance_m,
            require_no_new_hard_reason=require_no_new_hard_reason,
        )
        constrained_active = constrained_winner is not None
        constrained_index = 0 if constrained_winner is None else constrained_winner
        constrained_delay_steps = int(
            terminal_steps[constrained_index] - terminal_steps[0]
        )
        constrained_path_delta = float(
            path_length[constrained_index] - path_length[0]
        )
        constrained_progress_delta = float(
            candidate_progress[constrained_index] - candidate_progress[0]
        )
        bucket = _dataset_bucket(row["scene_path"])

        add("groups", 1)
        add("active", active)
        add("winner_is_candidate0", winner == 0)
        add("winner_gain", winner_gain)
        add("spread", spread)
        add("winner_path_delta", path_delta)
        add("winner_progress_delta", progress_delta)
        add("winner_path_shorter_0p5m", path_delta < -0.5)
        add("winner_path_shorter_1m", path_delta < -1.0)
        add("winner_progress_lower_0p02", progress_delta < -0.02)
        add("winner_collision", "collision" in winner_reasons)
        add("winner_road_border", "road_border" in winner_reasons)
        add("winner_kinematic", "kinematic" in winner_reasons)
        add("terminal_active", terminal_active)
        add("terminal_winner_is_candidate0", terminal_winner == 0)
        add("terminal_delay_steps", terminal_delay_steps)
        add("terminal_spread", terminal_spread)
        add("terminal_winner_path_delta", terminal_path_delta)
        add("terminal_winner_progress_delta", terminal_progress_delta)
        add("terminal_winner_path_shorter_1m", terminal_path_delta < -1.0)
        add("unsupported_kinematic_only_slots", unsupported_kinematic_only.sum())
        add("constrained_active", constrained_active)
        if constrained_active:
            add("constrained_delay_steps_active", constrained_delay_steps)
            add("constrained_path_delta_active", constrained_path_delta)
            add("constrained_progress_delta_active", constrained_progress_delta)
        for name, value in (
            ("groups", 1),
            ("active", active),
            ("winner_gain", winner_gain),
            ("winner_path_delta", path_delta),
            ("winner_progress_delta", progress_delta),
            ("winner_path_shorter_1m", path_delta < -1.0),
            ("terminal_active", terminal_active),
            ("terminal_delay_steps", terminal_delay_steps),
            ("terminal_winner_path_delta", terminal_path_delta),
            ("terminal_winner_progress_delta", terminal_progress_delta),
        ):
            bucket_add(bucket, name, value)

        stop_risk = max(-path_delta, 0.0) + 4.0 * max(-progress_delta, 0.0)
        usefulness = winner_gain + 0.2 * max(path_delta, 0.0) + max(progress_delta, 0.0)
        examples.append(
            {
                **row,
                "dataset": bucket,
                "survival_winner": winner,
                "survival_rewards": survival_total.tolist(),
                "survival_spread": spread,
                "winner_advantage_over_candidate0": winner_gain,
                "winner_path_length_delta_m": path_delta,
                "winner_progress_delta": progress_delta,
                "winner_reasons": list(winner_reasons),
                "candidate0_reasons": list(det_reasons),
                "terminal_prefix_rewards": terminal_reward.tolist(),
                "terminal_steps": terminal_steps.tolist(),
                "terminal_prefix_winner": terminal_winner,
                "terminal_delay_over_candidate0_steps": terminal_delay_steps,
                "terminal_winner_path_length_delta_m": terminal_path_delta,
                "terminal_winner_progress_delta": terminal_progress_delta,
                "kinematic_only_slots_unsupported_by_terminal_time": np.flatnonzero(
                    unsupported_kinematic_only
                ).tolist(),
                "constrained_terminal_winner": constrained_winner,
                "constrained_terminal_delay_steps": constrained_delay_steps,
                "constrained_winner_path_length_delta_m": constrained_path_delta,
                "constrained_winner_progress_delta": constrained_progress_delta,
                "stop_risk_severity": stop_risk,
                "usefulness_score": usefulness,
            }
        )
        if ordinal % 64 == 0:
            print(f"rescored {ordinal}/{len(rows)} all-zero groups", flush=True)

    groups = sums["groups"]
    active_groups = sums["active"]
    useful = sorted(examples, key=lambda item: item["usefulness_score"], reverse=True)
    stop_risk = sorted(examples, key=lambda item: item["stop_risk_severity"], reverse=True)
    return {
        "contract": {
            "evidence": "same cached K trajectories in stored all-zero gate groups rescored with survival shaping",
            "not_evidence_of": "post-update checkpoint improvement",
            "risk_checked": "survival winner can gain by shortening/stopping rather than a better maneuver",
        },
        "replay_root": str(replay_root),
        "parameters": {
            "sample_groups": sample_groups,
            "seed": seed,
            "reward_horizon_steps": horizon,
            "constrained_min_delay_steps": min_delay_steps,
            "constrained_progress_tolerance": progress_tolerance,
            "constrained_path_length_tolerance_m": path_length_tolerance_m,
            "constrained_require_no_new_hard_reason": require_no_new_hard_reason,
        },
        "committed_prefix_groups": committed_total,
        "all_zero_groups_in_committed_prefix": all_zero_total,
        "all_zero_group_fraction": all_zero_total / committed_total,
        "audited_groups": int(groups),
        "legacy_full_quality_survival": {
            "active_group_fraction": active_groups / groups,
            "candidate0_is_winner_fraction": sums["winner_is_candidate0"] / groups,
            "mean_winner_advantage": sums["winner_gain"] / groups,
            "mean_spread": sums["spread"] / groups,
            "mean_winner_path_length_delta_m": sums["winner_path_delta"] / groups,
            "mean_winner_progress_delta": sums["winner_progress_delta"] / groups,
            "winner_path_shorter_0p5m_fraction": sums["winner_path_shorter_0p5m"] / groups,
            "winner_path_shorter_1m_fraction": sums["winner_path_shorter_1m"] / groups,
            "winner_progress_lower_0p02_fraction": sums["winner_progress_lower_0p02"] / groups,
        },
        "faithful_terminal_prefix_survival": {
            "active_group_fraction": sums["terminal_active"] / groups,
            "candidate0_is_winner_fraction": sums["terminal_winner_is_candidate0"] / groups,
            "mean_delay_over_candidate0_steps": sums["terminal_delay_steps"] / groups,
            "mean_spread": sums["terminal_spread"] / groups,
            "mean_winner_path_length_delta_m": sums["terminal_winner_path_delta"] / groups,
            "mean_winner_progress_delta": sums["terminal_winner_progress_delta"] / groups,
            "winner_path_shorter_1m_fraction": sums["terminal_winner_path_shorter_1m"] / groups,
            "kinematic_only_slot_fraction_not_ranked": sums["unsupported_kinematic_only_slots"]
            / (groups * len(survival)),
        },
        "constrained_terminal_repair": {
            "active_group_fraction_of_all_zero": sums["constrained_active"] / groups,
            "estimated_fraction_of_committed_corpus": (
                all_zero_total / committed_total
            )
            * (sums["constrained_active"] / groups),
            "mean_delay_steps_when_active": (
                sums.get("constrained_delay_steps_active", 0.0)
                / max(sums["constrained_active"], 1.0)
            ),
            "mean_path_length_delta_m_when_active": (
                sums.get("constrained_path_delta_active", 0.0)
                / max(sums["constrained_active"], 1.0)
            ),
            "mean_progress_delta_when_active": (
                sums.get("constrained_progress_delta_active", 0.0)
                / max(sums["constrained_active"], 1.0)
            ),
        },
        "legacy_winner_event_fraction": {
            "collision": sums["winner_collision"] / groups,
            "road_border": sums["winner_road_border"] / groups,
            "kinematic": sums["winner_kinematic"] / groups,
        },
        "by_dataset": {
            bucket: {
                "groups": int(values["groups"]),
                "legacy_survival_active_group_fraction": values["active"] / values["groups"],
                "legacy_mean_survival_winner_advantage": values["winner_gain"] / values["groups"],
                "legacy_mean_winner_path_length_delta_m": values["winner_path_delta"] / values["groups"],
                "legacy_mean_winner_progress_delta": values["winner_progress_delta"] / values["groups"],
                "winner_path_shorter_1m_fraction": values["winner_path_shorter_1m"] / values["groups"],
                "terminal_prefix_active_group_fraction": values["terminal_active"] / values["groups"],
                "terminal_prefix_mean_delay_steps": values["terminal_delay_steps"] / values["groups"],
                "terminal_prefix_mean_winner_path_delta_m": values["terminal_winner_path_delta"] / values["groups"],
                "terminal_prefix_mean_winner_progress_delta": values["terminal_winner_progress_delta"] / values["groups"],
            }
            for bucket, values in sorted(by_dataset.items())
        },
        "top_useful_examples": useful[:top_examples],
        "top_stop_risk_examples": stop_risk[:top_examples],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-groups", type=int, default=512)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--allow-partial-prefix", action="store_true")
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--top-examples", type=int, default=32)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--min-delay-steps", type=int, default=1)
    parser.add_argument("--progress-tolerance", type=float, default=0.02)
    parser.add_argument("--path-length-tolerance-m", type=float, default=0.2)
    parser.add_argument("--allow-hard-reason-substitution", action="store_true")
    args = parser.parse_args()
    if args.sample_groups < 1 or args.horizon < 1 or args.min_delay_steps < 1:
        parser.error("sample-groups, horizon and min-delay-steps must be positive")
    if args.progress_tolerance < 0 or args.path_length_tolerance_m < 0:
        parser.error("survival tolerances must be non-negative")
    torch.set_num_threads(max(1, int(args.torch_threads)))
    summary = audit(
        args.replay_root,
        sample_groups=int(args.sample_groups),
        seed=int(args.seed),
        allow_partial_prefix=bool(args.allow_partial_prefix),
        horizon=int(args.horizon),
        top_examples=int(args.top_examples),
        min_delay_steps=int(args.min_delay_steps),
        progress_tolerance=float(args.progress_tolerance),
        path_length_tolerance_m=float(args.path_length_tolerance_m),
        require_no_new_hard_reason=not bool(args.allow_hard_reason_substitution),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    hidden = {"top_useful_examples", "top_stop_risk_examples"}
    print(json.dumps({k: v for k, v in summary.items() if k not in hidden}, indent=2))


if __name__ == "__main__":
    main()
