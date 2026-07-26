#!/usr/bin/env python3
"""Audit the geometry of the exact AWR + expert regression targets.

This is a pre-update data audit, not checkpoint-performance evidence.  It can
read a strictly closed replay cache or a writer-committed prefix while mining
is active.  In the latter case the number of complete JSONL path rows is the
read boundary; no unwritten memmap tail is inspected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rlvr.autoresearch.tools.build_positive_anchor_replay_overlay import (
    compute_positive_anchor_weights,
)
from rlvr.autoresearch.tools.select_replay_visualization_scenes import (
    _endpoint_route_coordinates,
)


def _path_length(trajectories: np.ndarray) -> np.ndarray:
    """Origin-to-first plus inter-waypoint xy arc length."""

    origin_shape = (*trajectories.shape[:-2], 1, 2)
    xy = trajectories[..., :2]
    steps = np.diff(
        np.concatenate([np.zeros(origin_shape, dtype=xy.dtype), xy], axis=-2),
        axis=-2,
    )
    return np.linalg.norm(steps, axis=-1).sum(axis=-1)


def _rank_count(rank_dir: Path, allow_partial_prefix: bool) -> tuple[int, dict[str, Any] | None]:
    manifest_path = rank_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        return int(manifest["scene_count"]), manifest
    if not allow_partial_prefix:
        raise FileNotFoundError(
            f"{manifest_path}; pass --allow-partial-prefix only for an active strict writer"
        )
    paths = rank_dir / "scene_paths.jsonl"
    if not paths.is_file():
        raise FileNotFoundError(paths)
    with paths.open() as stream:
        count = sum(1 for line in stream if line.strip())
    return count, None


def analyze(
    replay_root: Path,
    *,
    allow_partial_prefix: bool,
    recompute_positive_weights: bool,
    beta: float,
    margin: float,
    behavior_anchor_weight: float,
    unsafe_behavior_anchor_weight: float,
    expert_weight: float,
    chunk_size: int = 4096,
    mode_longitudinal_threshold_m: float = 0.5,
    mode_lateral_threshold_m: float = 0.25,
    max_groups_per_rank: int = 0,
) -> dict[str, Any]:
    replay_root = replay_root.expanduser().resolve(strict=True)
    rank_dirs = sorted(path for path in replay_root.glob("rank_*" ) if path.is_dir())
    if not rank_dirs:
        raise FileNotFoundError(f"no rank directories under {replay_root}")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if expert_weight < 0.0:
        raise ValueError("expert_weight must be non-negative")
    if mode_longitudinal_threshold_m <= 0.0:
        raise ValueError("mode_longitudinal_threshold_m must be positive")
    if mode_lateral_threshold_m <= 0.0:
        raise ValueError("mode_lateral_threshold_m must be positive")
    if max_groups_per_rank < 0:
        raise ValueError("max_groups_per_rank must be non-negative")

    effective_path = replay_root.parent / "effective_config.json"
    effective = json.loads(effective_path.read_text()) if effective_path.is_file() else {}
    if effective and not bool(effective.get("awr", {}).get("deterministic_first", False)):
        raise RuntimeError("target-geometry audit requires deterministic candidate zero")
    configured_reward_horizon = int(
        effective.get("reward", {}).get("reward_horizon_steps", 0) or 0
    )

    sums: dict[str, float] = {}
    mode_sums: dict[str, dict[str, float]] = {
        name: {}
        for name in (
            "faster_only",
            "slower_only",
            "left_only",
            "right_only",
            "faster_left",
            "faster_right",
            "slower_left",
            "slower_right",
            "near_anchor",
        )
    }

    def add(name: str, value: float | np.number) -> None:
        sums[name] = sums.get(name, 0.0) + float(value)

    def add_mode(mode: str, name: str, value: float | np.number) -> None:
        bucket = mode_sums[mode]
        bucket[name] = bucket.get(name, 0.0) + float(value)

    rank_rows: list[dict[str, Any]] = []
    all_closed = True
    group_size: int | None = None
    future_len: int | None = None
    for rank_dir in rank_dirs:
        committed_count, manifest = _rank_count(rank_dir, allow_partial_prefix)
        count = (
            min(committed_count, int(max_groups_per_rank))
            if max_groups_per_rank > 0
            else committed_count
        )
        all_closed = all_closed and manifest is not None
        arrays = manifest.get("arrays", {}) if manifest else {}
        reward_name = str(arrays.get("rewards", "rewards.npy"))
        weight_name = str(arrays.get("weights", "weights.npy"))
        trajectory_name = str(arrays.get("trajectories", "trajectories.npy"))
        rewards = np.load(rank_dir / reward_name, mmap_mode="r")
        stored_weights = np.load(rank_dir / weight_name, mmap_mode="r")
        trajectories = np.load(rank_dir / trajectory_name, mmap_mode="r")
        expert_trajectories = np.load(rank_dir / "expert_trajectories.npy", mmap_mode="r")
        expert_safe = np.load(rank_dir / "expert_safe.npy", mmap_mode="r")
        if rewards.ndim != 2 or trajectories.ndim != 4:
            raise ValueError(f"invalid replay arrays in {rank_dir}")
        if trajectories.shape[:2] != rewards.shape:
            raise ValueError(f"trajectory/reward shape mismatch in {rank_dir}")
        if stored_weights.shape != rewards.shape:
            raise ValueError(f"weight/reward shape mismatch in {rank_dir}")
        if expert_trajectories.shape[0] < count or expert_safe.shape[0] < count:
            raise ValueError(f"expert sidecar is shorter than prefix in {rank_dir}")
        if count > rewards.shape[0]:
            raise ValueError(f"path count exceeds replay capacity in {rank_dir}")
        if group_size is None:
            group_size = int(rewards.shape[1])
            future_len = int(trajectories.shape[2])
        if int(rewards.shape[1]) != group_size or int(trajectories.shape[2]) != future_len:
            raise RuntimeError("mixed replay group/horizon shapes")
        reward_horizon = (
            min(configured_reward_horizon, future_len)
            if configured_reward_horizon > 0
            else future_len
        )
        if reward_horizon < 1:
            raise ValueError("effective reward horizon must contain at least one waypoint")

        for lower in range(0, count, chunk_size):
            upper = min(count, lower + chunk_size)
            scene_count = upper - lower
            reward = np.asarray(rewards[lower:upper], dtype=np.float64)
            if not np.isfinite(reward).all():
                raise ValueError(f"non-finite reward in committed prefix {rank_dir}")
            if recompute_positive_weights:
                weights = compute_positive_anchor_weights(
                    reward,
                    beta=beta,
                    margin=margin,
                    behavior_anchor_weight=behavior_anchor_weight,
                    unsafe_behavior_anchor_weight=unsafe_behavior_anchor_weight,
                ).astype(np.float64)
            else:
                weights = np.asarray(stored_weights[lower:upper], dtype=np.float64)
            trajectory = np.asarray(trajectories[lower:upper], dtype=np.float64)
            expert = np.asarray(expert_trajectories[lower:upper], dtype=np.float64)
            safe = np.asarray(expert_safe[lower:upper], dtype=np.float64) > 0.5

            candidate_weight = weights.sum(axis=1)
            expert_weights = float(expert_weight) * safe.astype(np.float64)
            total_weight = candidate_weight + expert_weights
            active = candidate_weight > 0.0
            any_target = total_weight > 0.0
            ordinary_expert_only = (~active) & safe
            candidate_count = (weights > 0.0).sum(axis=1)
            positive_mask = weights > 0.0
            multi_target = candidate_count >= 2
            endpoint_x = trajectory[:, :, -1, 0]
            endpoint_y = trajectory[:, :, -1, 1]
            route_s, route_d = _endpoint_route_coordinates(trajectory)
            positive_x_min = np.where(positive_mask, endpoint_x, np.inf).min(axis=1)
            positive_x_max = np.where(positive_mask, endpoint_x, -np.inf).max(axis=1)
            positive_y_min = np.where(positive_mask, endpoint_y, np.inf).min(axis=1)
            positive_y_max = np.where(positive_mask, endpoint_y, -np.inf).max(axis=1)
            positive_x_range = np.zeros(scene_count, dtype=np.float64)
            positive_y_range = np.zeros(scene_count, dtype=np.float64)
            positive_x_range[multi_target] = (
                positive_x_max[multi_target] - positive_x_min[multi_target]
            )
            positive_y_range[multi_target] = (
                positive_y_max[multi_target] - positive_y_min[multi_target]
            )
            positive_endpoint_diag = np.hypot(positive_x_range, positive_y_range)
            positive_s_min = np.where(positive_mask, route_s, np.inf).min(axis=1)
            positive_s_max = np.where(positive_mask, route_s, -np.inf).max(axis=1)
            positive_d_min = np.where(positive_mask, route_d, np.inf).min(axis=1)
            positive_d_max = np.where(positive_mask, route_d, -np.inf).max(axis=1)
            positive_s_range = np.zeros(scene_count, dtype=np.float64)
            positive_d_range = np.zeros(scene_count, dtype=np.float64)
            positive_s_range[multi_target] = (
                positive_s_max[multi_target] - positive_s_min[multi_target]
            )
            positive_d_range[multi_target] = (
                positive_d_max[multi_target] - positive_d_min[multi_target]
            )
            all_zero = np.max(np.abs(reward), axis=1) <= 1e-8
            reward_tied = np.std(reward, axis=1) <= 1e-6
            best_index = np.argmax(reward, axis=1)
            hard_recovery = (reward[:, 0] <= 1e-8) & (
                reward[np.arange(scene_count), best_index] >= 0.9
            )
            best_trajectory = trajectory[np.arange(scene_count), best_index]
            best_vs_det_xy = np.linalg.norm(
                best_trajectory[..., :2] - trajectory[:, 0, :, :2], axis=-1
            )
            best_vs_det_mean = best_vs_det_xy.mean(axis=1)
            best_vs_det_max = best_vs_det_xy.max(axis=1)
            best_vs_det_endpoint = best_vs_det_xy[:, -1]

            path_lengths = _path_length(trajectory)
            expert_lengths = _path_length(expert)
            det_lengths = path_lengths[:, 0]
            weighted_candidate_length = np.divide(
                (weights * path_lengths).sum(axis=1),
                candidate_weight,
                out=np.zeros(scene_count, dtype=np.float64),
                where=active,
            )
            combined_length = np.divide(
                (weights * path_lengths).sum(axis=1) + expert_weights * expert_lengths,
                total_weight,
                out=np.zeros(scene_count, dtype=np.float64),
                where=any_target,
            )
            weighted_candidate_reward = np.divide(
                (weights * reward).sum(axis=1),
                candidate_weight,
                out=np.zeros(scene_count, dtype=np.float64),
                where=active,
            )
            weighted_endpoint_x = np.divide(
                (weights * trajectory[:, :, -1, 0]).sum(axis=1),
                candidate_weight,
                out=np.zeros(scene_count, dtype=np.float64),
                where=active,
            )
            weighted_endpoint_y = np.divide(
                (weights * trajectory[:, :, -1, 1]).sum(axis=1),
                candidate_weight,
                out=np.zeros(scene_count, dtype=np.float64),
                where=active,
            )
            weighted_route_s = np.divide(
                (weights * route_s).sum(axis=1),
                candidate_weight,
                out=np.zeros(scene_count, dtype=np.float64),
                where=active,
            )
            weighted_route_d = np.divide(
                (weights * route_d).sum(axis=1),
                candidate_weight,
                out=np.zeros(scene_count, dtype=np.float64),
                where=active,
            )
            route_s_delta = weighted_route_s - route_s[:, 0]
            route_d_delta = weighted_route_d - route_d[:, 0]
            longitudinal_state = np.zeros(scene_count, dtype=np.int8)
            lateral_state = np.zeros(scene_count, dtype=np.int8)
            longitudinal_state[
                route_s_delta >= float(mode_longitudinal_threshold_m)
            ] = 1
            longitudinal_state[
                route_s_delta <= -float(mode_longitudinal_threshold_m)
            ] = -1
            lateral_state[route_d_delta >= float(mode_lateral_threshold_m)] = 1
            lateral_state[route_d_delta <= -float(mode_lateral_threshold_m)] = -1
            mode_masks = {
                "faster_only": active & (longitudinal_state == 1) & (lateral_state == 0),
                "slower_only": active & (longitudinal_state == -1) & (lateral_state == 0),
                "left_only": active & (longitudinal_state == 0) & (lateral_state == 1),
                "right_only": active & (longitudinal_state == 0) & (lateral_state == -1),
                "faster_left": active & (longitudinal_state == 1) & (lateral_state == 1),
                "faster_right": active & (longitudinal_state == 1) & (lateral_state == -1),
                "slower_left": active & (longitudinal_state == -1) & (lateral_state == 1),
                "slower_right": active & (longitudinal_state == -1) & (lateral_state == -1),
                "near_anchor": active & (longitudinal_state == 0) & (lateral_state == 0),
            }
            target_reward_gain = weighted_candidate_reward - reward[:, 0]
            for mode, mask in mode_masks.items():
                add_mode(mode, "groups", mask.sum())
                add_mode(mode, "reward_gain", target_reward_gain[mask].sum())
                add_mode(mode, "candidate_count", candidate_count[mask].sum())
                add_mode(mode, "route_s_delta", route_s_delta[mask].sum())
                add_mode(mode, "route_d_delta", route_d_delta[mask].sum())
                add_mode(mode, "path_length_delta", (
                    weighted_candidate_length[mask] - det_lengths[mask]
                ).sum())
            expert_share = np.divide(
                expert_weights,
                total_weight,
                out=np.zeros(scene_count, dtype=np.float64),
                where=any_target,
            )

            # The Original DP predicts 80 waypoints while the HDP-derived PDM
            # reward can be configured for only the first 40.  Quantify the
            # exact positive-AWR target displacement on both sides of that
            # boundary; otherwise a candidate selected for its 4 s reward may
            # silently contribute an unscored 4--8 s regression tail.
            candidate_vs_det_xy = np.linalg.norm(
                trajectory[..., :2] - trajectory[:, :1, :, :2], axis=-1
            )
            weighted_candidate_displacement = np.divide(
                (weights[..., None] * candidate_vs_det_xy).sum(axis=1),
                candidate_weight[:, None],
                out=np.zeros((scene_count, future_len), dtype=np.float64),
                where=active[:, None],
            )
            scored_mean_displacement = weighted_candidate_displacement[
                :, :reward_horizon
            ].mean(axis=1)
            if reward_horizon < future_len:
                tail_mean_displacement = weighted_candidate_displacement[
                    :, reward_horizon:
                ].mean(axis=1)
                first_unscored_displacement = weighted_candidate_displacement[
                    :, reward_horizon
                ]
            else:
                tail_mean_displacement = np.zeros(scene_count, dtype=np.float64)
                first_unscored_displacement = np.zeros(scene_count, dtype=np.float64)

            add("groups", scene_count)
            add("active_groups", active.sum())
            add("ordinary_expert_only_groups", ordinary_expert_only.sum())
            add("any_target_groups", any_target.sum())
            add("expert_safe_groups", safe.sum())
            add("all_zero_groups", all_zero.sum())
            add("all_zero_expert_safe_groups", (all_zero & safe).sum())
            add("all_zero_without_safe_expert_groups", (all_zero & ~safe).sum())
            add("reward_tied_groups", reward_tied.sum())
            add("reward_tied_expert_safe_groups", (reward_tied & safe).sum())
            add("hard_recovery_groups", hard_recovery.sum())
            add(
                "hard_recovery_tiny_1cm_groups",
                (hard_recovery & (best_vs_det_max <= 0.01)).sum(),
            )
            add(
                "hard_recovery_tiny_5cm_groups",
                (hard_recovery & (best_vs_det_max <= 0.05)).sum(),
            )
            add("hard_recovery_mean_displacement", best_vs_det_mean[hard_recovery].sum())
            add("hard_recovery_max_displacement", best_vs_det_max[hard_recovery].sum())
            add(
                "hard_recovery_endpoint_displacement",
                best_vs_det_endpoint[hard_recovery].sum(),
            )
            add("candidate_active_targets", candidate_count.sum())
            add("multi_target_groups", multi_target.sum())
            add(
                "positive_endpoint_separation_1m_groups",
                (multi_target & (positive_endpoint_diag >= 1.0)).sum(),
            )
            add(
                "positive_endpoint_separation_2m_groups",
                (multi_target & (positive_endpoint_diag >= 2.0)).sum(),
            )
            add(
                "positive_lateral_separation_1m_groups",
                (multi_target & (positive_d_range >= 1.0)).sum(),
            )
            add(
                "positive_longitudinal_separation_2m_groups",
                (multi_target & (positive_s_range >= 2.0)).sum(),
            )
            add("multi_target_route_s_range", positive_s_range[multi_target].sum())
            add("multi_target_route_d_range", positive_d_range[multi_target].sum())
            add("candidate_weight", candidate_weight.sum())
            add("expert_weight", expert_weights.sum())
            add("total_weight", total_weight.sum())
            add("expert_share", expert_share.sum())
            add("active_det_reward", reward[active, 0].sum())
            add("active_target_reward", weighted_candidate_reward[active].sum())
            add("active_det_length", det_lengths[active].sum())
            add("active_candidate_length", weighted_candidate_length[active].sum())
            add("active_combined_length", combined_length[active].sum())
            add("active_det_endpoint_x", trajectory[active, 0, -1, 0].sum())
            add("active_target_endpoint_x", weighted_endpoint_x[active].sum())
            add("active_det_abs_endpoint_y", np.abs(trajectory[active, 0, -1, 1]).sum())
            add("active_target_abs_endpoint_y", np.abs(weighted_endpoint_y[active]).sum())
            add(
                "active_candidate_scored_mean_displacement",
                scored_mean_displacement[active].sum(),
            )
            add(
                "active_candidate_tail_mean_displacement",
                tail_mean_displacement[active].sum(),
            )
            add(
                "active_candidate_reward_boundary_displacement",
                weighted_candidate_displacement[active, reward_horizon - 1].sum(),
            )
            add(
                "active_candidate_first_unscored_displacement",
                first_unscored_displacement[active].sum(),
            )
            add(
                "active_candidate_endpoint_displacement",
                weighted_candidate_displacement[active, -1].sum(),
            )
            add("ordinary_det_length", det_lengths[ordinary_expert_only].sum())
            add("ordinary_expert_length", expert_lengths[ordinary_expert_only].sum())

        rank_rows.append(
            {
                "rank": int(manifest.get("rank", len(rank_rows))) if manifest else len(rank_rows),
                "scene_count": count,
                "committed_scene_count": committed_count,
                "strict_manifest": manifest is not None,
            }
        )

    groups = sums["groups"]
    active_groups = sums["active_groups"]
    ordinary_groups = sums["ordinary_expert_only_groups"]
    any_target_groups = sums["any_target_groups"]
    total_weight = sums["total_weight"]
    hard_recovery_groups = sums["hard_recovery_groups"]
    multi_target_groups = sums["multi_target_groups"]

    def ratio(numerator: str, denominator: float) -> float:
        return sums[numerator] / denominator if denominator else 0.0

    return {
        "contract": {
            "evidence": "reward-weighted candidate/expert target geometry before optimizer update",
            "not_evidence_of": "post-training checkpoint improvement",
            "candidate_zero": "zero-noise unguided deployment behavior",
            "path_length": "origin-to-first plus inter-waypoint xy arc length",
            "weight_source": (
                "recomputed positive-advantage overlay"
                if recompute_positive_weights
                else "stored replay weights"
            ),
        },
        "replay_root": str(replay_root),
        "source_complete": all_closed,
        "analysis_truncated": bool(max_groups_per_rank > 0),
        "rank_counts": rank_rows,
        "prefix_groups": int(groups),
        "group_size": int(group_size or 0),
        "future_len": int(future_len or 0),
        "reward_horizon_steps": int(
            min(configured_reward_horizon, int(future_len or 0))
            if configured_reward_horizon > 0
            else int(future_len or 0)
        ),
        "unscored_tail_steps": int(
            max(
                0,
                int(future_len or 0)
                - (
                    min(configured_reward_horizon, int(future_len or 0))
                    if configured_reward_horizon > 0
                    else int(future_len or 0)
                ),
            )
        ),
        "parameters": {
            "beta": float(beta) if recompute_positive_weights else None,
            "margin": float(margin) if recompute_positive_weights else None,
            "behavior_anchor_weight": (
                float(behavior_anchor_weight) if recompute_positive_weights else None
            ),
            "unsafe_behavior_anchor_weight": (
                float(unsafe_behavior_anchor_weight) if recompute_positive_weights else None
            ),
            "expert_weight": float(expert_weight),
            "mode_longitudinal_threshold_m": float(
                mode_longitudinal_threshold_m
            ),
            "mode_lateral_threshold_m": float(mode_lateral_threshold_m),
            "max_groups_per_rank": int(max_groups_per_rank),
        },
        "active_target_mode_distribution": {
            mode: {
                "groups": int(values.get("groups", 0.0)),
                "fraction_of_active_groups": (
                    values.get("groups", 0.0) / active_groups
                    if active_groups
                    else 0.0
                ),
                "mean_reward_gain": (
                    values.get("reward_gain", 0.0) / values.get("groups", 0.0)
                    if values.get("groups", 0.0)
                    else 0.0
                ),
                "mean_positive_candidate_count": (
                    values.get("candidate_count", 0.0) / values.get("groups", 0.0)
                    if values.get("groups", 0.0)
                    else 0.0
                ),
                "mean_route_s_delta_m": (
                    values.get("route_s_delta", 0.0) / values.get("groups", 0.0)
                    if values.get("groups", 0.0)
                    else 0.0
                ),
                "mean_route_d_delta_m": (
                    values.get("route_d_delta", 0.0) / values.get("groups", 0.0)
                    if values.get("groups", 0.0)
                    else 0.0
                ),
                "mean_path_length_delta_m": (
                    values.get("path_length_delta", 0.0) / values.get("groups", 0.0)
                    if values.get("groups", 0.0)
                    else 0.0
                ),
            }
            for mode, values in mode_sums.items()
        },
        "active_group_fraction": active_groups / groups,
        "mean_active_candidate_count": ratio("candidate_active_targets", active_groups),
        "positive_multitarget_fraction_active": ratio(
            "multi_target_groups", active_groups
        ),
        "positive_endpoint_separation_1m_fraction_active": ratio(
            "positive_endpoint_separation_1m_groups", active_groups
        ),
        "positive_endpoint_separation_2m_fraction_active": ratio(
            "positive_endpoint_separation_2m_groups", active_groups
        ),
        "positive_lateral_separation_1m_fraction_active": ratio(
            "positive_lateral_separation_1m_groups", active_groups
        ),
        "positive_longitudinal_separation_2m_fraction_active": ratio(
            "positive_longitudinal_separation_2m_groups", active_groups
        ),
        "positive_multitarget_mean_route_s_range_m": ratio(
            "multi_target_route_s_range", multi_target_groups
        ),
        "positive_multitarget_mean_route_d_range_m": ratio(
            "multi_target_route_d_range", multi_target_groups
        ),
        "expert_safe_fraction": ratio("expert_safe_groups", groups),
        "all_zero_group_fraction": ratio("all_zero_groups", groups),
        "all_zero_expert_safe_fraction": ratio(
            "all_zero_expert_safe_groups", sums["all_zero_groups"]
        ),
        "no_current_candidate_or_safe_expert_fraction": ratio(
            "all_zero_without_safe_expert_groups", groups
        ),
        "reward_tied_group_fraction": ratio("reward_tied_groups", groups),
        "reward_tied_expert_safe_fraction": ratio(
            "reward_tied_expert_safe_groups", sums["reward_tied_groups"]
        ),
        "deterministic_hard_recovery_fraction": ratio("hard_recovery_groups", groups),
        "hard_recovery_tiny_1cm_fraction": ratio(
            "hard_recovery_tiny_1cm_groups", hard_recovery_groups
        ),
        "hard_recovery_tiny_5cm_fraction": ratio(
            "hard_recovery_tiny_5cm_groups", hard_recovery_groups
        ),
        "hard_recovery_tiny_5cm_group_fraction": ratio(
            "hard_recovery_tiny_5cm_groups", groups
        ),
        "hard_recovery_mean_path_displacement_m": ratio(
            "hard_recovery_mean_displacement", hard_recovery_groups
        ),
        "hard_recovery_mean_max_displacement_m": ratio(
            "hard_recovery_max_displacement", hard_recovery_groups
        ),
        "hard_recovery_mean_endpoint_displacement_m": ratio(
            "hard_recovery_endpoint_displacement", hard_recovery_groups
        ),
        "ordinary_expert_only_group_fraction": ordinary_groups / groups,
        "global_expert_weight_share": ratio("expert_weight", total_weight),
        "mean_per_scene_expert_share_when_any_target": ratio(
            "expert_share", any_target_groups
        ),
        "effective_mean_weight_per_target_slot": total_weight
        / (groups * (int(group_size or 0) + 1)),
        "active_candidate_target_reward_gain": ratio(
            "active_target_reward", active_groups
        )
        - ratio("active_det_reward", active_groups),
        "active_candidate_path_length_delta_m": ratio(
            "active_candidate_length", active_groups
        )
        - ratio("active_det_length", active_groups),
        "active_combined_path_length_delta_m": ratio(
            "active_combined_length", active_groups
        )
        - ratio("active_det_length", active_groups),
        "active_candidate_endpoint_x_delta_m": ratio(
            "active_target_endpoint_x", active_groups
        )
        - ratio("active_det_endpoint_x", active_groups),
        "active_candidate_abs_endpoint_y_delta_m": ratio(
            "active_target_abs_endpoint_y", active_groups
        )
        - ratio("active_det_abs_endpoint_y", active_groups),
        "active_candidate_mean_xy_displacement_scored_m": ratio(
            "active_candidate_scored_mean_displacement", active_groups
        ),
        "active_candidate_mean_xy_displacement_unscored_tail_m": ratio(
            "active_candidate_tail_mean_displacement", active_groups
        ),
        "active_candidate_unscored_to_scored_displacement_ratio": (
            ratio("active_candidate_tail_mean_displacement", active_groups)
            / ratio("active_candidate_scored_mean_displacement", active_groups)
            if ratio("active_candidate_scored_mean_displacement", active_groups) > 0.0
            else 0.0
        ),
        "active_candidate_reward_boundary_displacement_m": ratio(
            "active_candidate_reward_boundary_displacement", active_groups
        ),
        "active_candidate_first_unscored_displacement_m": ratio(
            "active_candidate_first_unscored_displacement", active_groups
        ),
        "active_candidate_endpoint_displacement_m": ratio(
            "active_candidate_endpoint_displacement", active_groups
        ),
        "ordinary_expert_vs_det_path_length_delta_m": ratio(
            "ordinary_expert_length", ordinary_groups
        )
        - ratio("ordinary_det_length", ordinary_groups),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial-prefix", action="store_true")
    parser.add_argument("--recompute-positive-weights", action="store_true")
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--margin", type=float, default=0.01)
    parser.add_argument("--behavior-anchor-weight", type=float, default=0.0)
    parser.add_argument("--unsafe-behavior-anchor-weight", type=float, default=0.0)
    parser.add_argument("--expert-weight", type=float, default=0.4)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--mode-longitudinal-threshold-m", type=float, default=0.5)
    parser.add_argument("--mode-lateral-threshold-m", type=float, default=0.25)
    parser.add_argument(
        "--max-groups-per-rank",
        type=int,
        default=0,
        help="Analyze only this committed prefix per rank; zero reads every group.",
    )
    args = parser.parse_args()
    summary = analyze(
        args.replay_root,
        allow_partial_prefix=bool(args.allow_partial_prefix),
        recompute_positive_weights=bool(args.recompute_positive_weights),
        beta=float(args.beta),
        margin=float(args.margin),
        behavior_anchor_weight=float(args.behavior_anchor_weight),
        unsafe_behavior_anchor_weight=float(args.unsafe_behavior_anchor_weight),
        expert_weight=float(args.expert_weight),
        chunk_size=int(args.chunk_size),
        mode_longitudinal_threshold_m=float(args.mode_longitudinal_threshold_m),
        mode_lateral_threshold_m=float(args.mode_lateral_threshold_m),
        max_groups_per_rank=int(args.max_groups_per_rank),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
