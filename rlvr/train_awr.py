#!/usr/bin/env python3
"""Train and evaluate AWR on the original four-channel Diffusion Planner.

The runner is intentionally independent of the HDP and GRPO training paths.
It consumes the v5.0 original-DP checkpoint, samples unguided DP diffusion
plans, scores them with the repository's OBB-aware T4 reward, and performs a
positive advantage-weighted diffusion regression update.

Examples:

    # The v5.0 release zip contains a .param.json instead of args.json.
    python -m rlvr.train_awr \
      --weights_zip ~/v5.0-20260715T090010Z-1-001.zip \
      --train_npz_list /mnt/storage_rdma/diffusion_planner/dataset/20260623_full_sequence/path_list_train_sft.json \
      --valid_npz_list /mnt/storage_rdma/diffusion_planner/dataset/20260623_full_sequence/path_list_valid_sft_balanced.json \
      --config rlvr/configs/awr_original_dp_t4.json \
      --device cuda --exp_name original_dp_awr_t4

For a fast CPU plumbing smoke test, override ``max_train_scenes`` and
``max_valid_scenes`` on the command line.  The checkpoint is still validated
against the original DP architecture before any training starts.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import random
import shutil
import time
import zipfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, fields
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.utils.neighbor_future_alignment import (
    align_neighbor_future_numpy,
    get_neighbor_future_offset,
)
from diffusion_planner.utils.scene_skip import filter_scene_list
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

from planner_metrics.config import RewardConfig
from preference_optimization.utils import (
    X2_LEGACY_EGO_WIDTH_M,
    XX1_LEGACY_EGO_WIDTH_M,
    apply_npz_heading_transforms,
    load_npz_data,
)
from rlvr.awr import (
    AWRRollout,
    AWRRolloutConfig,
    breakdown_metrics,
    load_original_dp_checkpoint,
    load_scene,
    reward_compatible_data,
    rollout_and_score_scene,
    rollout_and_score_scene_batch,
    rollout_to_json,
    save_checkpoint_pair,
    update_ema,
)
from rlvr.grpo_loss import compute_batched_trajectory_losses
from rlvr.replay_context_codec import CompressedReplayContextReader
from rlvr.reward import compute_reward_batch


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    # bool subclasses int in Python. Preserve manifest/config semantics rather
    # than silently serializing audit flags as 0/1.
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _json_safe(value.item())
        return _json_safe(value.detach().cpu().tolist())
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n")


def _write_json_atomic(path: Path, value: Any) -> None:
    """Publish a manifest only after its complete contents reach one file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(_json_safe(value), sort_keys=True) + "\n")


# Percentiles of binary rates and counts add dozens of charts while conveying
# almost no information (for example collision P50 is normally exactly zero).
# Keep full mean/P10/P50/P90 summaries in the local JSON audit trail, but show
# P50/P90 in W&B only where the distribution shape is operationally useful.
_EVAL_WANDB_PERCENTILE_BASES = {
    "det_reward",
    "best_group_reward",
    "ade",
    "fde",
    "candidate_endpoint_spread_mean",
    "candidate_pairwise_ade",
}

# The audit JSON keeps the original scalar representation.  The W&B dashboard
# deliberately does not expose that machine-oriented representation: a rate
# such as 0.0016 is too easy to read as "zero" on a human-facing chart.
_WANDB_RATE_BASES = {
    "det_collision": "collision_rate",
    "det_rb_crossing": "road_border_crossing_rate",
    "det_kinematic_violation": "kinematic_failure_rate",
    "det_lane_crossing": "lane_crossing_diagnostic_rate",
    "det_off_road_fraction": "polygon_off_road_rate",
    "safe_candidate_fraction": "safe_candidate_rate",
    "candidate_collision_fraction": "candidate_collision_rate",
    "candidate_rb_crossing_fraction": "candidate_road_border_crossing_rate",
    "candidate_lane_crossing_fraction": "candidate_lane_crossing_rate",
    "candidate_kinematic_violation_fraction": "candidate_kinematic_failure_rate",
    "det_zero_reward": "deterministic_zero_reward_rate",
    "zero_reward_candidate_fraction": "zero_reward_candidate_rate",
    "zero_reward_weight_share": "zero_reward_weight_share",
    "all_zero_reward_group": "all_zero_reward_group_rate",
    "dropped_all_zero_group": "dropped_all_zero_group_rate",
    "all_equal_reward_group": "all_equal_reward_group_rate",
    "det_zero_collision": "zero_reward_collision_rate",
    "det_zero_rb_crossing": "zero_reward_road_border_rate",
    "det_zero_kinematic": "zero_reward_kinematic_rate",
    "det_zero_without_collision_rb_kinematic": "zero_reward_unexplained_rate",
    "valid_group": "valid_group_rate",
    "expert_anchor_used": "expert_anchor_usage_rate",
    "replay_sampled_with_replacement": "replay_sampled_with_replacement_rate",
    "replay_decoder_context_cached": "replay_decoder_context_cache_hit_rate",
    "replay_compressed_context_cached": "compressed_context_cache_hit_rate",
    "replay_context_materialized_group_fraction": "context_materialized_group_rate",
    "shared_encoder_cache_hit": "shared_encoder_cache_hit_rate",
    "replay_update": "replay_update_rate",
    "skipped": "skipped_scene_rate",
    "active_target_fraction": "optimized_candidate_rate",
    "effective_sample_size_fraction": "effective_weight_coverage",
    "has_positive_candidate": "repair_target_group_rate",
    "behavior_anchor_active": "behavior_anchor_active_rate",
    "first_waypoint_invalid_candidate_fraction": "first_waypoint_invalid_candidate_rate",
    "first_waypoint_gate_enabled": "first_waypoint_gate_enabled_rate",
    "first_waypoint_gate_low_speed": "first_waypoint_low_speed_rate",
    "first_waypoint_gate_rejected_fraction": "first_waypoint_rejection_rate",
    "first_waypoint_backward_fraction": "first_waypoint_backward_rate",
    "stop_turn_scene": "stop_turn_scene_rate",
    "stop_turn_first_waypoint_gate_rejected_fraction": "stop_turn_rejection_rate",
}

_WANDB_REWARD_BASES = {
    "det_reward": "reward",
    "best_group_reward": "best_candidate_reward",
    "best_reward": "best_candidate_reward",
    "mean_group_reward": "candidate_reward",
    "best_safe_reward": "best_safe_candidate_reward",
    "best_vs_det_reward_gain": "reward_gain",
}

# These are signed reward components, not the bounded HDP-PDM total reward.
# The dashboard exposes their positive penalty magnitude instead.
_WANDB_SIGNED_PENALTY_BASES = {
    "det_safety": "safety_penalty_magnitude",
    "det_centerline": "centerline_penalty_magnitude",
    "det_smoothness": "smoothness_penalty_magnitude",
}

# Names used by the human-facing dashboard.  These are deliberately short and
# carry the unit in the name.  The original flat names remain only in local
# audit JSON, not in the W&B primary view.
_WANDB_PRIMARY_UNIT_BASES = {
    "ade": "ade_m",
    "fde": "fde_m",
    "path_len": "path_length_m",
    "gt_path_len": "expert_path_length_m",
    "candidate_endpoint_spread_mean": "endpoint_spread_mean_m",
    "candidate_endpoint_spread_max": "endpoint_spread_max_m",
    "candidate_pairwise_ade": "pairwise_trajectory_ade_m",
    "candidate_temporal_spread": "temporal_spread_m",
    "first_waypoint_gate_rejected_count": "first_waypoint_rejected_count",
    "first_waypoint_invalid_candidate_count": "first_waypoint_invalid_candidate_count",
    "first_waypoint_displacement_p95_m": "first_waypoint_displacement_p95_m",
    "first_waypoint_lateral_abs_p95_m": "first_waypoint_lateral_abs_p95_m",
    "first_waypoint_path_tangent_abs_deg_p95": "first_waypoint_path_tangent_abs_deg_p95",
    "stop_turn_first_waypoint_displacement_p95_m": "stop_turn_first_waypoint_displacement_p95_m",
    "stop_turn_first_waypoint_path_tangent_abs_deg_p95": "stop_turn_first_waypoint_path_tangent_abs_deg_p95",
    "det_progress": "progress_score",
    "candidate_count": "candidate_count",
    "safe_candidate_count": "safe_candidate_count",
    "reward_unique_count": "unique_reward_count",
    "candidate_reward_std": "candidate_reward_std",
    "reward_std": "candidate_reward_std",
    "candidate_collision_count": "candidate_collision_count",
    "candidate_rb_crossing_count": "candidate_road_border_crossing_count",
    "candidate_kinematic_violation_count": "candidate_kinematic_failure_count",
    "weight_sum": "weight_sum",
    "optimizer_step": "optimizer_step",
    "gpu_peak_allocated_gib_rank0": "gpu_peak_allocated_gib",
    "gpu_peak_reserved_gib_rank0": "gpu_peak_reserved_gib",
    "cycle": "cycle",
    "rollout_refresh": "rollout_refresh",
    "replay_training": "replay_training",
    "fallback_nonfinite_update": "nonfinite_update_fallback",
    "elapsed_sec": "elapsed_sec",
    "scenes_per_sec": "scenes_per_sec",
    "optimizer_steps": "optimizer_steps",
    "learning_rate": "learning_rate",
    "loss": "loss",
    "grad_norm": "grad_norm",
    "effective_sample_size": "effective_sample_size",
    "active_weight_count": "weighted_candidate_count",
    "top1_weight_share": "top1_weight_share",
    "positive_candidate_count": "positive_candidate_count",
    "behavior_anchor_weight": "behavior_anchor_weight",
    "active_target_count": "optimized_candidate_count",
    "full_target_count": "available_candidate_count",
    "computed_target_count": "decoder_candidate_count",
    "policy_transaction_candidate": "proposal_evaluated_0_or_1",
    "policy_transaction_accepted": "proposal_selected_as_best_0_or_1",
    "policy_transaction_rolled_back": "proposal_rolled_back_0_or_1",
    "policy_training_continued_from_proposal": "proposal_used_as_next_start_0_or_1",
    "optimizer_state_reset_after_policy_commit": "optimizer_state_reset_0_or_1",
    "policy_proposal_relative_l2": "proposal_relative_l2",
    "policy_accepted_relative_l2": "committed_update_relative_l2",
}

# Transaction quantities that need percentage scaling but are neither event
# rates nor generic reward summaries.  Keep their units explicit: a selector
# delta is a percentage-point change, while the EMA commit is a percentage.
_WANDB_POLICY_TRANSACTION_PERCENT_BASES = {
    "ema_policy_commit_rate": "ema_commit_rate_pct",
    "train_selection_reward_before_epoch": "best_reward_before_epoch_pct",
    "train_selection_reward_delta_vs_best": "reward_delta_vs_best_pp",
}

# Candidate-group geometry is meaningful only when K > 1.  It belongs to the
# replay/mining view, not to the deterministic K=1 deployment validation.  A
# dedicated set keeps the dashboard grouping and the K=1 suppression contract
# in one place.
_WANDB_MULTIMODALITY_BASES = {
    "candidate_count",
    "candidate_endpoint_spread_mean",
    "candidate_endpoint_spread_max",
    "candidate_pairwise_ade",
    "candidate_temporal_spread",
    "candidate_reward_std",
    "reward_std",
    "reward_unique_count",
}

# Only these delta aliases are emitted.  A positive value always means that
# the current policy improved over the source DP.
_WANDB_IMPROVEMENT_BASES = {
    "det_reward": ("reward", "pp"),
    "best_group_reward": ("best_candidate_reward", "pp"),
    "ade": ("ade", "m"),
    "fde": ("fde", "m"),
    "det_collision": ("collision_rate", "pp"),
    "det_rb_crossing": ("road_border_crossing_rate", "pp"),
    "det_kinematic_violation": ("kinematic_failure_rate", "pp"),
    "det_lane_crossing": ("lane_crossing_diagnostic_rate", "pp"),
    "det_off_road_fraction": ("polygon_off_road_rate", "pp"),
    "det_zero_reward": ("deterministic_zero_reward_rate", "pp"),
    "all_zero_reward_group": ("all_zero_reward_group_rate", "pp"),
    "det_safety": ("safety_penalty_magnitude", "score"),
    "det_centerline": ("centerline_penalty_magnitude", "score"),
    "det_smoothness": ("smoothness_penalty_magnitude", "score"),
}

_WANDB_READABLE_PREFIXES = {
    "train",
    "train_selector_ema",
    "eval",
    "baseline",
    "full_eval",
    "full_baseline",
    "hard_eval",
    "hard_baseline",
}


def _wandb_primary_metric_name(key: str) -> tuple[str | None, str | None]:
    """Return (statistic, human-facing name) for one summary key.

    ``mean_mean_group_reward`` is a real key in the audit summary: the first
    ``mean`` is the summary statistic and the second belongs to the original
    metric name.  Normalising it here prevents ugly names such as
    ``mean_mean_candidate_reward_pct`` in the dashboard.
    """

    statistic, base = _split_summary_stat(key)
    if base == "mean_group_reward":
        base = "group_reward"
    if base in _WANDB_RATE_BASES:
        name = f"{_WANDB_RATE_BASES[base]}_pct"
    elif base in _WANDB_REWARD_BASES:
        name = f"{_WANDB_REWARD_BASES[base]}_pct"
    elif base in _WANDB_SIGNED_PENALTY_BASES:
        name = _WANDB_SIGNED_PENALTY_BASES[base]
    elif base in _WANDB_POLICY_TRANSACTION_PERCENT_BASES:
        name = _WANDB_POLICY_TRANSACTION_PERCENT_BASES[base]
    elif base in _WANDB_PRIMARY_UNIT_BASES:
        name = _WANDB_PRIMARY_UNIT_BASES[base]
    elif base == "group_reward":
        name = "candidate_reward_pct"
    elif base == "scene_count":
        name = "scene_count"
    else:
        return statistic, None
    if statistic in {"p50", "p90"}:
        name = f"{name}_{statistic}"
    return statistic, name


def _wandb_primary_section(prefix: str, key: str) -> str:
    """Choose a small human-facing section for one primary metric."""

    _, base = _split_summary_stat(key)
    if base == "mean_group_reward":
        base = "group_reward"
    if prefix == "train":
        if base in {
            "policy_transaction_candidate",
            "policy_transaction_accepted",
            "policy_transaction_rolled_back",
            "policy_training_continued_from_proposal",
            "optimizer_state_reset_after_policy_commit",
            "policy_proposal_relative_l2",
            "policy_accepted_relative_l2",
            "ema_policy_commit_rate",
            "train_selection_reward_before_epoch",
            "train_selection_reward_delta_vs_best",
        }:
            return "policy_transaction"
        if base in {
            "loss",
            "grad_norm",
            "learning_rate",
            "optimizer_step",
            "optimizer_steps",
            "fallback_nonfinite_update",
        }:
            return "optimization"
        if base in {
            "elapsed_sec",
            "scenes_per_sec",
            "gpu_peak_allocated_gib_rank0",
            "gpu_peak_reserved_gib_rank0",
            "scene_count",
            "cycle",
            "rollout_refresh",
            "replay_training",
            "replay_compressed_context_cached",
            "replay_context_materialized_group_fraction",
        }:
            return "system"
        if base in _WANDB_MULTIMODALITY_BASES:
            return "multimodality"
        return "awr_signal"
    if base in {
        "det_reward",
        "best_group_reward",
        "best_reward",
        "group_reward",
        "best_safe_reward",
        "best_vs_det_reward_gain",
        "det_zero_reward",
        "zero_reward_candidate_fraction",
        "zero_reward_weight_share",
        "all_zero_reward_group",
        "all_equal_reward_group",
        "det_zero_collision",
        "det_zero_rb_crossing",
        "det_zero_kinematic",
        "det_zero_without_collision_rb_kinematic",
    }:
        return "reward"
    if base in {
        "det_collision",
        "det_rb_crossing",
        "det_kinematic_violation",
        "safe_candidate_count",
        "safe_candidate_fraction",
        "candidate_collision_count",
        "candidate_collision_fraction",
        "candidate_rb_crossing_count",
        "candidate_rb_crossing_fraction",
        "candidate_kinematic_violation_count",
        "candidate_kinematic_violation_fraction",
    }:
        return "safety"
    if base in {
        "det_lane_crossing",
        "candidate_lane_crossing_fraction",
        "det_off_road_fraction",
    }:
        return "road"
    if base in {"ade", "fde", "gt_path_len"}:
        return "imitation"
    if base in _WANDB_MULTIMODALITY_BASES:
        return "multimodality"
    if base in {
        "first_waypoint_gate_enabled",
        "first_waypoint_gate_low_speed",
        "first_waypoint_gate_rejected_fraction",
        "first_waypoint_gate_rejected_count",
        "first_waypoint_invalid_candidate_fraction",
        "first_waypoint_invalid_candidate_count",
        "first_waypoint_displacement_p95_m",
        "first_waypoint_lateral_abs_p95_m",
        "first_waypoint_path_tangent_abs_deg_p95",
        "first_waypoint_backward_fraction",
        "stop_turn_scene",
        "stop_turn_first_waypoint_displacement_p95_m",
        "stop_turn_first_waypoint_path_tangent_abs_deg_p95",
        "stop_turn_first_waypoint_gate_rejected_fraction",
    }:
        return "startup_continuity"
    if base in {
        "det_progress",
        "det_safety",
        "det_centerline",
        "det_smoothness",
        "path_len",
    }:
        return "driving_quality"
    if base == "scene_count":
        return "status"
    return "diagnostics"


def _split_summary_stat(key: str) -> tuple[str | None, str]:
    for statistic in ("mean", "p10", "p50", "p90"):
        prefix = f"{statistic}_"
        if key.startswith(prefix):
            return statistic, key[len(prefix) :]
    return None, key


def _wandb_readable_aliases(
    prefix: str,
    key: str,
    value: float,
) -> dict[str, float]:
    """Return the human-facing W&B primary metric for one scalar.

    The aliases intentionally do not replace the audit names.  Rates and
    bounded rewards are percentages, trajectory errors are metres, and signed
    reward subcomponents are positive penalty magnitudes.  This keeps the
    primary dashboard readable without changing the numerical contract of the
    local summaries.
    """

    if prefix not in _WANDB_READABLE_PREFIXES:
        return {}
    statistic, name = _wandb_primary_metric_name(key)
    if statistic == "p10":
        return {}
    _, base = _split_summary_stat(key)
    if (
        prefix in {
            "train_selector_ema",
            "eval",
            "baseline",
            "full_eval",
            "full_baseline",
            "hard_eval",
            "hard_baseline",
        }
        and statistic in {"p50", "p90"}
        and base not in _EVAL_WANDB_PERCENTILE_BASES
    ):
        return {}
    if name is None:
        return {}
    # In all-stochastic group-relative mining, the historical internal key
    # ``det_reward`` is reward of group member zero, not deployment K=1
    # reward. Evaluation and train-selector prefixes do use a deterministic
    # candidate, so only the train-facing label changes.
    if prefix == "train" and base == "det_reward":
        name = "candidate0_reward_pct"
    elif prefix == "train" and base == "det_zero_reward":
        name = "candidate0_zero_reward_rate_pct"
    if base == "mean_group_reward":
        base = "group_reward"
    if (
        base == "group_reward"
        or base in _WANDB_RATE_BASES
        or base in _WANDB_REWARD_BASES
        or base in _WANDB_POLICY_TRANSACTION_PERCENT_BASES
    ):
        value = float(value) * 100.0
    elif base in _WANDB_SIGNED_PENALTY_BASES:
        value = max(0.0, -float(value))
    section = _wandb_primary_section(prefix, key)
    aliases = {f"{prefix}_primary/{section}/{name}": float(value)}
    return aliases


def _wandb_improvement_aliases(
    prefix: str,
    key: str,
    value: float,
    baseline: float,
) -> dict[str, float]:
    """Return deltas where positive is always better than source DP."""

    if prefix not in {"train_selector_ema", "eval", "full_eval", "hard_eval"}:
        return {}
    statistic, base = _split_summary_stat(key)
    if statistic in {"p10", "p50", "p90"} or base not in _WANDB_IMPROVEMENT_BASES:
        return {}
    name, unit = _WANDB_IMPROVEMENT_BASES[base]
    if base in _WANDB_REWARD_BASES:
        improvement = (float(value) - float(baseline)) * 100.0
    elif base in _WANDB_RATE_BASES or base in {"ade", "fde"}:
        # Rates and imitation errors are lower-is-better.
        improvement = float(baseline) - float(value)
        if unit == "pp":
            improvement *= 100.0
    else:
        # The source values are signed penalties.  Moving toward zero is an
        # improvement, hence baseline - current.
        improvement = float(baseline) - float(value)
    section = _wandb_primary_section(prefix, key)
    return {
        f"{prefix}_delta_primary/{section}/improvement_{name}_{unit}": improvement
    }


def _wandb_log_documentation(run: Any) -> None:
    """Publish a compact metric contract once at run start."""

    try:
        import wandb

        table = wandb.Table(
            columns=["metric", "meaning", "unit", "better", "role"],
            data=[
                [
                    "eval_primary/reward/reward_pct",
                    "HDP-PDM aggregate reward of the deterministic deployed plan",
                    "% of reward scale",
                    "higher",
                    "primary objective",
                ],
                [
                    "eval_delta_primary/reward/improvement_reward_pp",
                    "change in reward versus source original DP",
                    "percentage points",
                    "positive = better",
                    "replacement evidence",
                ],
                [
                    "eval_primary/safety/collision_rate_pct",
                    "scenes with an ego-neighbor collision",
                    "% of scenes",
                    "lower",
                    "hard safety outcome",
                ],
                [
                    "eval_primary/safety/road_border_crossing_rate_pct",
                    "scenes where ego OBB crosses the road-border margin",
                    "% of scenes",
                    "lower",
                    "hard safety outcome",
                ],
                [
                    "eval_primary/safety/kinematic_failure_rate_pct",
                    "scenes violating the configured kinematic-rate limits",
                    "% of scenes",
                    "lower",
                    "hard feasibility outcome",
                ],
                [
                    "eval_primary/road/lane_crossing_diagnostic_rate_pct",
                    "lane-crossing diagnostic; not a hard gate in this run",
                    "% of scenes",
                    "context only",
                    "soft road diagnostic",
                ],
                [
                    "eval_primary/imitation/ade_m / fde_m",
                    "open-loop distance to the logged expert trajectory",
                    "metres",
                    "lower",
                    "imitation diagnostic",
                ],
                [
                    "eval_primary/reward/all_zero_reward_group_rate_pct",
                    "groups where every sampled plan receives zero reward",
                    "% of groups",
                    "lower",
                    "AWR learning-signal health",
                ],
                [
                    "train_primary/multimodality/endpoint_spread_mean_m",
                    "mean endpoint separation among the replay candidate group",
                    "metres",
                    "diagnostic; not maximize blindly",
                    "multimodality",
                ],
                [
                    "train_primary/multimodality/pairwise_trajectory_ade_m",
                    "mean pairwise distance between replay candidate trajectories",
                    "metres",
                    "diagnostic; inspect with safety",
                    "multimodality",
                ],
                [
                    "train_primary/optimization/loss",
                    "advantage-weighted diffusion regression loss",
                    "model loss",
                    "lower is not sufficient alone",
                    "optimization",
                ],
                [
                    "train_primary/policy_transaction/proposal_selected_as_best_0_or_1",
                    "whether this epoch's committed 5% EMA proposal beat the historical best on the fixed train selector",
                    "binary (1 selected as best, 0 not selected)",
                    "1",
                    "checkpoint selection",
                ],
                [
                    "train_primary/policy_transaction/proposal_rolled_back_0_or_1",
                    "whether live policy and EMA were restored to best_train.pth after selector rejection",
                    "binary (1 rolled back, 0 retained)",
                    "context only",
                    "training continuity",
                ],
                [
                    "train_primary/policy_transaction/proposal_used_as_next_start_0_or_1",
                    "whether the committed proposal becomes the starting policy for the next replay epoch, independently of best-checkpoint selection",
                    "binary (1 continued, 0 not continued)",
                    "1 for cumulative replay",
                    "training continuity",
                ],
                [
                    "train_primary/policy_transaction/reward_delta_vs_best_pp",
                    "fixed DP10 train-selector reward of the proposal minus the previous best",
                    "percentage points",
                    "positive = selected as best",
                    "checkpoint selection",
                ],
                [
                    "train_primary/awr_signal/candidate0_reward_pct",
                    "reward of group member zero; deterministic only when deterministic_first is enabled",
                    "% of reward scale",
                    "context only",
                    "AWR learning-signal health",
                ],
                [
                    "train_selector_ema_primary/reward/reward_pct",
                    "deterministic K=1 reward of deployable EMA on a fixed train subset",
                    "% of reward scale",
                    "higher",
                    "checkpoint selection",
                ],
                [
                    "train_primary/awr_signal/effective_sample_size",
                    "effective number of active trajectories per scene-group after AWR weighting",
                    "trajectory-equivalents / group",
                    "interpret with active count",
                    "AWR learning-signal health",
                ],
                [
                    "train_primary/awr_signal/effective_weight_coverage_pct",
                    "ESS divided by active weighted trajectories; low means one target dominates",
                    "% of active trajectories",
                    "higher means less concentrated",
                    "AWR learning-signal health",
                ],
                [
                    "train_primary/awr_signal/zero_reward_weight_share_pct",
                    "fraction of total AWR target weight assigned to zero-reward trajectories",
                    "% of AWR weight",
                    "lower; interpret with hard-case coverage",
                    "AWR learning-signal health",
                ],
                [
                    "train_primary/awr_signal/repair_target_group_rate_pct",
                    "scene-groups containing at least one sampled plan above Rdet + margin",
                    "% of scene-groups",
                    "coverage diagnostic",
                    "AWR learning-signal health",
                ],
                [
                    "train_primary/system/context_materialized_group_rate_pct",
                    "fraction of replay groups whose immutable decoder context was read because they carry non-zero target weight",
                    "% of replay groups",
                    "lower means less mathematically dead replay I/O",
                    "RL replay throughput",
                ],
            ],
        )
        # ``documentation/*`` uses the real epoch as its W&B step metric just
        # like every other clean-dashboard namespace.  Include epoch=0 so the
        # guide is not left on an undefined/custom-step row.
        run.log({"epoch": 0, "documentation/metric_guide": table})
        run.summary["documentation/dashboard_schema"] = "primary_v2"
        run.summary["documentation/x_axis"] = "epoch"
        run.summary["documentation/checkpoint_selection"] = (
            "train-set mean deterministic reward; validation is report-only"
        )
        run.summary["documentation/positive_delta_rule"] = (
            "*_delta_primary/* is positive when current policy improves source DP"
        )
    except Exception as error:  # documentation must never stop training
        print(f"W&B documentation table skipped: {error}")


def _wandb_scene_visual_payload(
    rows: list[dict[str, Any]] | None,
    baseline_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build data-backed W&B distributions and a small scene drill-down table.

    The full JSON scene audit remains on disk.  W&B gets a deliberately small
    but useful view: actual distributions and representative scenes selected
    by reward, imitation error, safety events, and policy change.  No synthetic
    trajectories or fake examples are introduced.
    """

    if not rows:
        return {}
    try:
        import wandb
    except ImportError:
        return {}

    valid_rows = [row for row in rows if isinstance(row, dict)]
    if not valid_rows:
        return {}
    baseline_by_scene = {
        str(row.get("scene_path")): row
        for row in (baseline_rows or [])
        if isinstance(row, dict) and row.get("scene_path") is not None
    }

    def score(row: dict[str, Any], key: str, default: float = 0.0) -> float:
        try:
            value = float(row.get(key, default))
            return value if math.isfinite(value) else default
        except (TypeError, ValueError):
            return default

    def scene_key(row: dict[str, Any]) -> str:
        return str(row.get("scene_path", row.get("scene_index", "")))

    chosen: list[tuple[str, dict[str, Any]]] = []
    chosen_keys: set[str] = set()

    def add(label: str, row: dict[str, Any]) -> None:
        key = scene_key(row)
        if key in chosen_keys:
            return
        chosen_keys.add(key)
        chosen.append((label, row))

    for row in sorted(valid_rows, key=lambda item: score(item, "det_reward"))[:8]:
        add("lowest_reward", row)
    for row in sorted(valid_rows, key=lambda item: score(item, "ade"), reverse=True)[:8]:
        add("highest_ADE", row)
    for row in sorted(valid_rows, key=lambda item: score(item, "fde"), reverse=True)[:8]:
        add("highest_FDE", row)
    safety_rows = [
        row
        for row in valid_rows
        if any(
            score(row, key) > 0.0
            for key in ("det_collision", "det_rb_crossing", "det_kinematic_violation")
        )
    ]
    for row in sorted(
        safety_rows,
        key=lambda item: (
            score(item, "det_collision"),
            score(item, "det_rb_crossing"),
            score(item, "det_kinematic_violation"),
        ),
        reverse=True,
    )[:8]:
        add("safety_event", row)
    if baseline_by_scene:

        def reward_change(row: dict[str, Any]) -> float:
            base = baseline_by_scene.get(scene_key(row), {})
            return score(row, "det_reward") - score(base, "det_reward")

        for row in sorted(valid_rows, key=reward_change, reverse=True)[:8]:
            add("largest_reward_gain", row)
        for row in sorted(valid_rows, key=reward_change)[:8]:
            add("largest_reward_regression", row)

    columns = [
        "selection",
        "scene_index",
        "scene_path",
        "reward_pct",
        "source_reward_pct",
        "reward_change_pp",
        "ade_m",
        "fde_m",
        "collision_pct",
        "road_border_crossing_pct",
        "kinematic_failure_pct",
        "lane_crossing_diagnostic_pct",
        "endpoint_spread_mean_m",
        "pairwise_trajectory_ade_m",
        "all_zero_reward_group",
    ]
    table_rows = []
    for label, row in chosen:
        base = baseline_by_scene.get(scene_key(row), {})
        table_rows.append(
            [
                label,
                int(score(row, "scene_index", -1)),
                scene_key(row),
                score(row, "det_reward") * 100.0,
                score(base, "det_reward") * 100.0 if base else None,
                (score(row, "det_reward") - score(base, "det_reward")) * 100.0
                if base
                else None,
                score(row, "ade"),
                score(row, "fde"),
                score(row, "det_collision") * 100.0,
                score(row, "det_rb_crossing") * 100.0,
                score(row, "det_kinematic_violation") * 100.0,
                score(row, "det_lane_crossing") * 100.0,
                score(row, "candidate_endpoint_spread_mean"),
                score(row, "candidate_pairwise_ade"),
                score(row, "all_zero_reward_group"),
            ]
        )
    payload: dict[str, Any] = {
        "visuals/scene_examples": wandb.Table(columns=columns, data=table_rows)
    }

    histogram_specs = {
        "reward_pct": ("det_reward", 100.0),
        "ade_m": ("ade", 1.0),
        "fde_m": ("fde", 1.0),
    }
    if any(score(row, "candidate_count", 1.0) > 1.0 for row in valid_rows):
        histogram_specs.update(
            {
                "endpoint_spread_mean_m": (
                    "candidate_endpoint_spread_mean",
                    1.0,
                ),
                "pairwise_trajectory_ade_m": (
                    "candidate_pairwise_ade",
                    1.0,
                ),
            }
        )
    for name, (key, scale) in histogram_specs.items():
        values = [score(row, key) * scale for row in valid_rows]
        if values:
            counts, edges = np.histogram(values, bins=32)
            payload[f"visuals/distribution/{name}"] = wandb.Histogram(
                np_histogram=(counts, edges)
            )
    return payload


def _init_wandb(
    args: argparse.Namespace,
    run_dir: Path,
    effective_config: dict[str, Any],
):
    """Start one rank-zero W&B run only when explicitly requested."""

    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("--wandb was requested but the wandb package is unavailable") from error
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or run_dir.name,
        id=args.wandb_run_id,
        resume="allow" if args.wandb_run_id else None,
        mode=args.wandb_mode,
        dir=str(run_dir),
        tags=args.wandb_tags or None,
        config=_json_safe(effective_config),
    )
    run.define_metric("epoch")
    # Custom step metrics keep train/eval/full-validation curves aligned on
    # the actual epoch rather than W&B's internal row counter.  The latter is
    # not a training epoch because baseline, train, eval and optional hard
    # validation records are logged at different times.
    run.define_metric("visuals/*", step_metric="epoch")
    run.define_metric("documentation/*", step_metric="epoch")
    primary_prefixes = (
        "train",
        "train_selector_ema",
        "eval",
        "baseline",
        "full_eval",
        "full_baseline",
        "hard_eval",
        "hard_baseline",
    )
    for prefix in primary_prefixes:
        run.define_metric(f"{prefix}_primary/*", step_metric="epoch")
    for prefix in ("train_selector_ema", "eval", "full_eval", "hard_eval"):
        run.define_metric(f"{prefix}_delta_primary/*", step_metric="epoch")
    namespaces = {
        "system/*",
        "best/*",
        *{
            f"{prefix}_primary/*"
            for prefix in (
                "train",
                "train_selector_ema",
                "eval",
                "baseline",
                "full_eval",
                "full_baseline",
                "hard_eval",
                "hard_baseline",
            )
        },
        *{
            f"{prefix}_delta_primary/*"
            for prefix in ("train_selector_ema", "eval", "full_eval", "hard_eval")
        },
    }
    for namespace in sorted(namespaces):
        run.define_metric(namespace, step_metric="epoch")
    _wandb_log_documentation(run)
    return run


def _wandb_summary_payload(
    prefix: str,
    summary: dict[str, Any],
    baseline: dict[str, Any] | None = None,
) -> dict[str, float | int]:
    """Build one W&B payload without committing it.

    Keeping payload construction separate lets train and eval be committed as
    one row at the same epoch.  This prevents the W&B internal ``_step`` from
    interleaving train/eval rows and makes panels that use ``epoch`` reliable.
    """

    payload: dict[str, float | int] = {}
    deterministic_k1 = False
    candidate_count = summary.get("mean_candidate_count")
    if prefix != "train" and isinstance(candidate_count, (int, float)):
        deterministic_k1 = float(candidate_count) <= 1.0 + 1e-9
    for key, value in summary.items():
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        _, base = _split_summary_stat(key)
        if deterministic_k1 and base in _WANDB_MULTIMODALITY_BASES:
            # A K=1 deployment evaluation has zero spread by definition.  Do
            # not manufacture a dashboard full of flat-zero "multimodality"
            # curves; the corresponding K=10 replay metrics remain visible.
            continue
        aliases = _wandb_readable_aliases(prefix, key, float(value))
        if aliases:
            payload.update(aliases)
        if baseline is not None and key in baseline:
            base = baseline[key]
            if isinstance(base, (int, float)) and math.isfinite(float(base)):
                payload.update(
                    _wandb_improvement_aliases(
                        prefix, key, float(value), float(base)
                    )
                )
    return payload


def _wandb_log_summary(
    run: Any,
    prefix: str,
    summary: dict[str, Any],
    epoch: int,
    baseline: dict[str, Any] | None = None,
    rows: list[dict[str, Any]] | None = None,
    baseline_rows: list[dict[str, Any]] | None = None,
    step: int | None = None,
) -> None:
    if run is None:
        return
    payload = {"epoch": int(epoch)}
    payload.update(_wandb_summary_payload(prefix, summary, baseline))
    if rows is not None:
        payload.update(_wandb_scene_visual_payload(rows, baseline_rows))
    if step is None:
        run.log(payload)
    else:
        run.log(payload, step=int(step))


def _wandb_log_epoch_summaries(
    run: Any,
    epoch: int,
    train_summary: dict[str, Any],
    eval_summary: dict[str, Any],
    baseline: dict[str, Any] | None = None,
    eval_rows: list[dict[str, Any]] | None = None,
    baseline_rows: list[dict[str, Any]] | None = None,
) -> None:
    """Commit train and eval metrics in one monotonic W&B history row."""

    if run is None:
        return
    payload: dict[str, float | int] = {"epoch": int(epoch)}
    payload.update(_wandb_summary_payload("train", train_summary))
    payload.update(_wandb_summary_payload("eval", eval_summary, baseline))
    if eval_rows is not None:
        payload.update(_wandb_scene_visual_payload(eval_rows, baseline_rows))
    # Let W&B assign one monotonic internal row while every metric family uses
    # the explicit ``epoch`` field as its chart x-axis.  Passing ``step=epoch``
    # would make baseline/full-eval side logs collide with the combined row.
    run.log(payload)


def _find_inherited_refresh_summaries(
    source_run: Path, inherited_epoch: int
) -> tuple[Path, Path] | None:
    """Locate split-process refresh summaries across local/global numbering.

    Cycle 2 was mined as a local epoch 1 before global numbering was made
    explicit. Later standalone refresh runs write their actual global epoch
    (21, 31, ...). Prefer that unambiguous path while retaining the already
    completed cycle-2 artifact as a read-only fallback.
    """

    candidates = [int(inherited_epoch)]
    if int(inherited_epoch) != 1:
        candidates.append(1)
    for epoch in candidates:
        train = source_run / f"train_epoch_{epoch:03d}" / "summary.json"
        evaluation = source_run / f"eval_epoch_{epoch:03d}" / "summary.json"
        if train.is_file() and evaluation.is_file():
            return train, evaluation
    return None


def _resolve_inherited_refresh_source_run(replay_root: Path) -> Path:
    """Resolve a replay overlay back to the immutable mine run it derives from.

    A replay-only process normally receives ``.../overlay/replay_buffer``.
    The refresh summaries live beside the source replay, not beside that
    lightweight weight overlay.  Following the overlay's committed provenance
    keeps the W&B timeline complete without changing replay data or training.
    Direct source replays remain supported for older runs and unit tests.
    """

    replay_root = Path(replay_root).expanduser().resolve()
    direct_run = replay_root.parent
    candidates: list[Path] = [direct_run]

    overlay_manifest = direct_run / "overlay_manifest.json"
    if overlay_manifest.is_file():
        payload = json.loads(overlay_manifest.read_text())
        source_replay = payload.get("source_replay")
        if isinstance(source_replay, str) and source_replay:
            candidates.append(Path(source_replay).expanduser().resolve().parent)

    provenance = direct_run / "provenance.json"
    if provenance.is_file():
        payload = json.loads(provenance.read_text())
        source_replay = payload.get("replay_weight_overlay_source")
        if isinstance(source_replay, str) and source_replay:
            candidates.append(Path(source_replay).expanduser().resolve().parent)

    for candidate in candidates:
        if any(candidate.glob("train_epoch_*/summary.json")):
            return candidate
    return direct_run


def _load_inherited_refresh_baseline(
    source_run: Path,
    *,
    current_checkpoint: Path,
    current_args: Path,
    valid_paths: list[str],
    train_selector_paths: list[str],
    rollout_config: AWRRolloutConfig,
    reward_config: RewardConfig,
    eval_k: int,
    eval_sample_steps: int,
    verify_checkpoint_hash: bool,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    """Load the exact pre-refresh evaluations for a split replay process.

    A mine-only process and its replay continuation start from the same
    behavior/EMA checkpoint.  Re-running the all-valid K-sample baseline and
    fixed train selector in the second process is therefore redundant.  This
    loader is deliberately stricter than a summary copy: checkpoint, model
    args, reward/AWR config, loader semantics, and every scene path must agree.
    The caller performs the expensive checkpoint hash only on rank zero, then
    broadcasts success before other ranks consume the same immutable JSON.
    """

    source_run = Path(source_run).expanduser().resolve()
    provenance_path = source_run / "provenance.json"
    effective_path = source_run / "effective_config.json"
    final_summary_path = source_run / "final_summary.json"
    for required in (provenance_path, effective_path, final_summary_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    provenance = json.loads(provenance_path.read_text())
    effective = json.loads(effective_path.read_text())
    if provenance.get("resume_replay_root") is not None:
        raise RuntimeError("inherited baseline source is not a mine-only run")
    source_start = int(provenance.get("start_epoch", -1))
    source_epochs = int(effective.get("training", {}).get("train_epochs", -1))
    interval = int(
        effective.get("training", {}).get("hdp_rollout_interval", 0)
    )
    if (
        source_start <= 0
        or source_start != source_epochs
        or interval <= 0
        or (source_start - 1) % interval != 0
    ):
        raise RuntimeError(
            "inherited baseline source is not one completed rollout-only "
            f"refresh: start={source_start}, epochs={source_epochs}, "
            f"interval={interval}"
        )
    if bool(provenance.get("use_policy_state", False)):
        raise RuntimeError(
            "inherited baseline source evaluated live policy rather than the "
            "deployable behavior/EMA payload"
        )
    if int(provenance.get("neighbor_future_alignment_offset", -1)) != int(
        get_neighbor_future_offset()
    ):
        raise RuntimeError("inherited baseline neighbor-future contract differs")
    if not math.isclose(
        float(provenance.get("x2_legacy_ego_width_m", float("nan"))),
        2.29156,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError("inherited baseline X2 geometry contract differs")

    source_checkpoint = Path(provenance["staged_model"]).expanduser().resolve()
    source_args = Path(provenance["staged_args"]).expanduser().resolve()
    if not source_checkpoint.is_file():
        raise FileNotFoundError(source_checkpoint)
    if not source_args.is_file():
        raise FileNotFoundError(source_args)
    if _file_sha256(source_args) != _file_sha256(current_args):
        raise RuntimeError("inherited baseline model args differ")
    if verify_checkpoint_hash and _file_sha256(source_checkpoint) != _file_sha256(
        current_checkpoint
    ):
        raise RuntimeError("inherited baseline checkpoint differs")

    source_awr = dict(effective.get("awr", {}))
    current_awr = _json_safe(asdict(rollout_config))
    if source_awr != current_awr:
        mismatches = sorted(
            key
            for key in set(source_awr) | set(current_awr)
            if source_awr.get(key) != current_awr.get(key)
        )
        raise RuntimeError(
            f"inherited baseline AWR/evaluation config differs: {mismatches}"
        )
    source_reward = dict(effective.get("reward", {}))
    # These two fields document the resolved HDP/PDM adapter but do not alter
    # RewardConfig or evaluation. Compare the executable dataclass contract.
    source_reward.pop("effective_quality_weights", None)
    source_reward.pop("hdp_multi_weight_fields_active", None)
    current_reward = _json_safe(asdict(reward_config))
    if source_reward != current_reward:
        mismatches = sorted(
            key
            for key in set(source_reward) | set(current_reward)
            if source_reward.get(key) != current_reward.get(key)
        )
        raise RuntimeError(
            f"inherited baseline reward config differs: {mismatches}"
        )
    if int(effective.get("training", {}).get("eval_k", -1)) != int(eval_k):
        raise RuntimeError("inherited baseline evaluation K differs")
    source_eval_steps = int(
        effective.get("training", {}).get(
            "eval_sample_steps", source_awr.get("sample_steps", -1)
        )
    )
    if source_eval_steps != int(eval_sample_steps):
        raise RuntimeError("inherited baseline evaluation sample steps differ")

    eval_summary_path = source_run / "eval_epoch_000" / "summary.json"
    eval_rows_path = source_run / "eval_epoch_000" / "scenes.json"
    selector_summary_path = (
        source_run / "train_selector_epoch_000" / "summary.json"
    )
    selector_rows_path = source_run / "train_selector_epoch_000" / "scenes.json"
    for required in (
        eval_summary_path,
        eval_rows_path,
        selector_summary_path,
        selector_rows_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    eval_summary = json.loads(eval_summary_path.read_text())
    eval_rows = json.loads(eval_rows_path.read_text())
    selector_summary = json.loads(selector_summary_path.read_text())
    selector_rows = json.loads(selector_rows_path.read_text())
    if [str(row.get("scene_path")) for row in eval_rows] != list(valid_paths):
        raise RuntimeError("inherited baseline valid scene order differs")
    if [str(row.get("scene_path")) for row in selector_rows] != list(
        train_selector_paths
    ):
        raise RuntimeError("inherited baseline train-selector order differs")
    if int(eval_summary.get("scene_count", -1)) != len(valid_paths):
        raise RuntimeError("inherited baseline valid summary count differs")
    if int(selector_summary.get("scene_count", -1)) != len(
        train_selector_paths
    ):
        raise RuntimeError("inherited baseline train-selector count differs")
    return eval_summary, eval_rows, selector_summary, selector_rows


def _wandb_log_inherited_refresh_epoch(
    run: Any,
    args: argparse.Namespace,
    run_dir: Path,
    baseline: dict[str, Any],
) -> None:
    """Carry the completed refresh-only epoch into a resumed W&B run.

    A split-process HDP cycle mines one local epoch in the source run, then
    resumes replay at the next *global* epoch.  Map that local source epoch to
    ``start_epoch - 1`` so cycle 2 appears as epochs 11--20 (rather than
    overwriting another 1--10 range) while the replay cache remains the source
    of truth.
    """

    if run is None or args.resume_replay_root is None or int(args.start_epoch) <= 1:
        return
    inherited_epoch = int(args.start_epoch) - 1
    source_run = _resolve_inherited_refresh_source_run(
        args.resume_replay_root
    )
    source_summaries = _find_inherited_refresh_summaries(
        source_run, inherited_epoch
    )
    if source_summaries is None:
        print(
            "W&B resume: refresh summaries not found for "
            f"global epoch {inherited_epoch} or legacy local epoch 1 under "
            f"{source_run}"
        )
        return
    source_train, source_eval = source_summaries
    try:
        train_summary = json.loads(source_train.read_text())
        eval_summary = json.loads(source_eval.read_text())
        if not isinstance(train_summary, dict) or not isinstance(eval_summary, dict):
            raise TypeError("refresh summary JSON must contain objects")
        source_train_epoch = train_summary.get("epoch", 1)
        source_eval_epoch = eval_summary.get("epoch", 1)
        inherited_train_summary = {
            **train_summary,
            "epoch": inherited_epoch,
            "source_local_epoch": source_train_epoch,
        }
        # A faithful refresh is collect-only: neither live policy nor
        # deployable EMA changes.  Mine processes intentionally evaluate only
        # a tiny smoke subset, whereas the replay process has just evaluated
        # the same untouched policy on the complete validation set.  Reuse
        # that exact full baseline for the refresh epoch instead of presenting
        # an 8-scene smoke metric as if it were comparable to later epochs.
        inherited_eval_summary = {
            **baseline,
            "epoch": inherited_epoch,
            "source_local_epoch": source_eval_epoch,
            "source_refresh_eval_scene_count": eval_summary.get(
                "scene_count", 0
            ),
            "policy_unchanged_during_refresh": 1.0,
        }
        _write_json(
            run_dir / f"inherited_epoch_{inherited_epoch:03d}_train_summary.json",
            inherited_train_summary,
        )
        _write_json(
            run_dir / f"inherited_epoch_{inherited_epoch:03d}_eval_summary.json",
            inherited_eval_summary,
        )
        # Keep the local JSONL audit timeline contiguous as well as the W&B
        # timeline. Without these records, local plotters would omit the
        # rollout-only global epoch immediately preceding replay.
        _append_jsonl(
            run_dir / "metrics.jsonl",
            {"phase": "train", "inherited": True, **inherited_train_summary},
        )
        _append_jsonl(
            run_dir / "metrics.jsonl",
            {
                "phase": "eval",
                "inherited": True,
                "validation_scope": (
                    "full_baseline_reused_policy_unchanged_during_refresh"
                ),
                **inherited_eval_summary,
            },
        )
        _wandb_log_epoch_summaries(
            run,
            epoch=inherited_epoch,
            train_summary=train_summary,
            eval_summary=inherited_eval_summary,
            baseline=baseline,
        )
        run.summary["resume/inherited_refresh_epoch"] = inherited_epoch
        run.summary["resume/source_run_dir"] = str(source_run)
        run.summary["resume/source_refresh_cache"] = str(args.resume_replay_root)
        print(
            f"W&B resume: mapped local refresh epoch 1 to global epoch "
            f"{inherited_epoch} from {source_run}"
        )
    except Exception as error:
        print(f"W&B resume: failed to inherit epoch-1 summaries: {error}")


def _copy_json_config(config_path: Path, run_dir: Path, raw_config: dict[str, Any]) -> None:
    _write_json(run_dir / "awr_config.json", raw_config)
    shutil.copy2(config_path, run_dir / "awr_config_source.json")


def _prepare_checkpoint(
    model_path: Path | None,
    weights_zip: Path | None,
    run_dir: Path,
    args_path: Path | None = None,
) -> tuple[Path, Path]:
    """Stage a checkpoint and its JSON args without modifying the source zip."""

    stage = run_dir / "base_model"
    stage.mkdir(parents=True, exist_ok=True)
    if weights_zip is not None:
        if not weights_zip.exists():
            raise FileNotFoundError(weights_zip)
        with zipfile.ZipFile(weights_zip) as archive:
            pth_members = [m for m in archive.namelist() if m.endswith("diffusion_planner.pth")]
            param_members = [m for m in archive.namelist() if m.endswith("diffusion_planner.param.json")]
            if len(pth_members) != 1 or len(param_members) != 1:
                raise RuntimeError(
                    f"expected one original DP .pth and .param.json in {weights_zip}; "
                    f"found pth={pth_members}, param={param_members}"
                )
            staged_model = stage / "original_dp_v5.pth"
            staged_args = stage / "args.json"
            with archive.open(pth_members[0]) as src, staged_model.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            with archive.open(param_members[0]) as src, staged_args.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            return staged_model, staged_args

    if model_path is None:
        raise ValueError("provide either --model_path or --weights_zip")
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    args_candidates: list[Path] = []
    if args_path is not None:
        args_candidates.append(args_path)
    args_candidates.extend(
        [model_path.parent / "args.json", model_path.parent / "model_args.json"]
    )
    args_candidates.extend(sorted(model_path.parent.glob("*.param.json")))
    args_source = next((p for p in args_candidates if p.exists()), None)
    if args_source is None:
        raise FileNotFoundError(f"no args.json or *.param.json beside {model_path}")
    staged_model = stage / model_path.name
    staged_args = stage / "args.json"
    shutil.copy2(model_path, staged_model)
    shutil.copy2(args_source, staged_args)
    return staged_model, staged_args


def _restore_behavior_ema(model: nn.Module, checkpoint_path: Path) -> bool:
    """Restore the frozen behavior EMA from an AWR continuation checkpoint.

    ``--use_policy_state`` loads the checkpoint's ``model`` state into the
    trainable policy.  A faithful continuation also needs the separately
    saved ``ema_state_dict`` for the behavior/reference policy used by the
    next HDP replay refresh.  Original v5 checkpoints do not contain a
    separate policy state, so this is deliberately a no-op for them.
    """

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "ema_state_dict" not in checkpoint:
        return False
    state = {
        key.replace("module.", "").replace("._orig_mod.", ".").replace("_orig_mod.", ""): value
        for key, value in checkpoint["ema_state_dict"].items()
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "continuation checkpoint EMA is not architecture-compatible: "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
    return True


def _restore_optimizer_state(
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path,
    *,
    learning_rate: float,
) -> dict[str, Any]:
    """Restore Adam moments while keeping this run's explicit learning rate.

    ``Optimizer.load_state_dict`` also restores param-group hyperparameters.
    A new cycle may intentionally choose another LR after inspecting the prior
    cycle, so restore the accumulated moments/steps but re-apply the effective
    LR from the current command.  Parameter-group order is validated by
    PyTorch and any incompatibility fails closed before training starts.
    """

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "optimizer_state_dict" not in checkpoint:
        raise RuntimeError(
            "--resume_optimizer_state requested, but checkpoint has no "
            f"optimizer_state_dict: {checkpoint_path}"
        )
    state = checkpoint["optimizer_state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint optimizer_state_dict is not a mapping")
    optimizer.load_state_dict(state)
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)
    return {
        "restored": True,
        "source_epoch": int(checkpoint.get("epoch", -1)),
        "source_optimizer_class": checkpoint.get("optimizer_class"),
        "state_parameter_count": int(len(optimizer.state)),
        "effective_learning_rate": float(learning_rate),
    }


def _load_config(path: Path) -> tuple[dict[str, Any], AWRRolloutConfig, RewardConfig]:
    raw = json.loads(path.read_text())
    awr_raw = dict(raw.get("awr", raw))
    noise_range = tuple(float(x) for x in awr_raw.get("noise_scale_range", (0.5, 2.0)))
    rollout = AWRRolloutConfig(
        n_trajectories=int(awr_raw.get("n_trajectories", awr_raw.get("K", 8))),
        sample_steps=int(awr_raw.get("sample_steps", 5)),
        noise_scale_range=noise_range,
        beta=float(awr_raw.get("beta", 0.75)),
        weight_clip=float(awr_raw.get("weight_clip", 20.0)),
        normalize_weights=bool(awr_raw.get("normalize_weights", True)),
        min_group_std=float(awr_raw.get("min_group_std", 1e-5)),
        safe_only=bool(awr_raw.get("safe_only", False)),
        structured_exploration=bool(awr_raw.get("structured_exploration", False)),
        deterministic_first=bool(awr_raw.get("deterministic_first", True)),
        hdp_trajectory_augmentation=bool(
            awr_raw.get("hdp_trajectory_augmentation", False)
        ),
        hdp_trajectory_augmentation_std=float(
            awr_raw.get("hdp_trajectory_augmentation_std", 0.5)
        ),
        hdp_trajectory_augmentation_ramp_steps=int(
            awr_raw.get("hdp_trajectory_augmentation_ramp_steps", 0)
        ),
        hdp_trajectory_augmentation_heading_mode=str(
            awr_raw.get("hdp_trajectory_augmentation_heading_mode", "preserve")
        ),
        hdp_trajectory_augmentation_skip_low_speed=bool(
            awr_raw.get("hdp_trajectory_augmentation_skip_low_speed", True)
        ),
        hdp_trajectory_augmentation_low_speed_threshold=float(
            awr_raw.get("hdp_trajectory_augmentation_low_speed_threshold", 2.0)
        ),
        hdp_trajectory_augmentation_schedule=str(
            awr_raw.get(
                "hdp_trajectory_augmentation_schedule", "first_five_epochs"
            )
        ),
        positive_advantage_only=bool(awr_raw.get("positive_advantage_only", False)),
        positive_advantage_margin=float(
            awr_raw.get("positive_advantage_margin", 0.0)
        ),
        behavior_anchor_weight=float(awr_raw.get("behavior_anchor_weight", 1.0)),
        unsafe_behavior_anchor_weight=float(
            awr_raw.get("unsafe_behavior_anchor_weight", 1.0)
        ),
        drop_all_zero_groups=bool(awr_raw.get("drop_all_zero_groups", False)),
        plannerrft_guided_exploration=bool(
            awr_raw.get("plannerrft_guided_exploration", False)
        ),
        plannerrft_trigger_best_reward_lt=float(
            awr_raw.get("plannerrft_trigger_best_reward_lt", 0.9)
        ),
        plannerrft_guided_candidates=int(
            awr_raw.get(
                "plannerrft_guided_candidates",
                awr_raw.get("n_trajectories", awr_raw.get("K", 8)),
            )
        ),
        plannerrft_guidance_bucket_size=int(
            awr_raw.get("plannerrft_guidance_bucket_size", 32)
        ),
        plannerrft_eta_scheme=str(
            awr_raw.get("plannerrft_eta_scheme", "stratified_beta")
        ),
        plannerrft_beta_concentration=float(
            awr_raw.get("plannerrft_beta_concentration", 1.0 + math.log(2.0))
        ),
        plannerrft_reference_mode=str(
            awr_raw.get("plannerrft_reference_mode", "zero_noise")
        ),
        plannerrft_reference_noise_scale=float(
            awr_raw.get("plannerrft_reference_noise_scale", 0.5)
        ),
        plannerrft_longitudinal_mode=str(
            awr_raw.get("plannerrft_longitudinal_mode", "candidate_stretch")
        ),
        plannerrft_lambda_lat=float(
            awr_raw.get("plannerrft_lambda_lat", 1.0)
        ),
        plannerrft_lambda_lon=float(
            awr_raw.get("plannerrft_lambda_lon", 0.25)
        ),
        plannerrft_guidance_scale=float(
            awr_raw.get("plannerrft_guidance_scale", 2.0)
        ),
        plannerrft_ramp_steps=int(
            awr_raw.get("plannerrft_ramp_steps", 20)
        ),
        original_dp_first_waypoint_gate_enabled=bool(
            awr_raw.get("original_dp_first_waypoint_gate_enabled", True)
        ),
        original_dp_first_waypoint_gate_speed_mps=float(
            awr_raw.get("original_dp_first_waypoint_gate_speed_mps", 1.0)
        ),
        original_dp_first_waypoint_gate_max_step_m=float(
            awr_raw.get("original_dp_first_waypoint_gate_max_step_m", 0.25)
        ),
        original_dp_first_waypoint_gate_max_lateral_m=float(
            awr_raw.get("original_dp_first_waypoint_gate_max_lateral_m", 0.20)
        ),
        original_dp_first_waypoint_gate_max_backward_m=float(
            awr_raw.get("original_dp_first_waypoint_gate_max_backward_m", 0.05)
        ),
        original_dp_first_waypoint_gate_max_tangent_deg=float(
            awr_raw.get("original_dp_first_waypoint_gate_max_tangent_deg", 75.0)
        ),
        original_dp_first_waypoint_gate_max_implied_steer_rad=float(
            awr_raw.get(
                "original_dp_first_waypoint_gate_max_implied_steer_rad", 0.64
            )
        ),
        original_dp_first_waypoint_gate_tangent_min_step_m=float(
            awr_raw.get("original_dp_first_waypoint_gate_tangent_min_step_m", 0.005)
        ),
        original_dp_stop_turn_speed_mps=float(
            awr_raw.get("original_dp_stop_turn_speed_mps", 1.0)
        ),
    )
    reward_raw = dict(raw.get("reward", raw.get("reward_config", {})))
    reward_fields = {field.name for field in fields(RewardConfig)}
    reward_kwargs = {key: value for key, value in reward_raw.items() if key in reward_fields}
    reward = RewardConfig(**reward_kwargs)
    return raw, rollout, reward


def _validate_replay_source_contract(
    replay_root: Path,
    rollout_config: AWRRolloutConfig,
    reward_config: RewardConfig,
    *,
    allow_missing_defaults: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Fail closed when a frozen cache was mined by another AWR/reward setup.

    Replay arrays already contain trajectories, rewards and final weights.
    Changing beta, positive-only semantics, augmentation, heading handling or
    reward config on the resume command does not retroactively change those
    arrays.  Treating such a cache as compatible is a silent method change.
    """

    source_run = Path(replay_root).resolve().parent
    config_path = source_run / "effective_config.json"
    provenance_path = source_run / "provenance.json"
    if not config_path.exists() or not provenance_path.exists():
        raise RuntimeError(
            "replay cache has no source effective_config/provenance contract: "
            f"{replay_root}"
        )
    source = json.loads(config_path.read_text())
    expected = {
        "awr": _json_safe(asdict(rollout_config)),
        "reward": _json_safe(asdict(reward_config)),
    }
    mismatches: list[str] = []
    allowed_missing = allow_missing_defaults or {}
    for section, expected_values in expected.items():
        source_values = source.get(section)
        if not isinstance(source_values, dict):
            mismatches.append(f"missing source section {section}")
            continue
        for key, expected_value in expected_values.items():
            if key not in source_values:
                section_defaults = allowed_missing.get(section, {})
                if (
                    key in section_defaults
                    and expected_value == section_defaults[key]
                ):
                    continue
                mismatches.append(f"{section}.{key}: missing in source")
            elif source_values[key] != expected_value:
                mismatches.append(
                    f"{section}.{key}: source={source_values[key]!r}, "
                    f"resume={expected_value!r}"
                )
    provenance = json.loads(provenance_path.read_text())
    source_offset = provenance.get("neighbor_future_alignment_offset")
    expected_offset = int(get_neighbor_future_offset())
    if source_offset != expected_offset:
        mismatches.append(
            "neighbor_future_alignment_offset: "
            f"source={source_offset!r}, resume={expected_offset!r}"
        )
    if mismatches:
        preview = "\n  - ".join(mismatches[:20])
        suffix = (
            f"\n  ... and {len(mismatches) - 20} more"
            if len(mismatches) > 20
            else ""
        )
        raise RuntimeError(
            "frozen replay cache contract mismatch; re-mine the cache instead "
            f"of silently changing its method:\n  - {preview}{suffix}"
        )


def _choose_paths(
    path_list: Path,
    limit: int,
    seed: int,
    label: str,
    skip_filtered_scenes: bool = False,
    sidecar_root: Path | None = None,
) -> list[str]:
    paths = json.loads(path_list.read_text())
    if not isinstance(paths, list) or not paths:
        raise ValueError(f"{label} list is empty or not a JSON list: {path_list}")
    source_count = len(paths)
    if skip_filtered_scenes:
        paths = filter_scene_list(
            paths,
            sidecar_root=sidecar_root,
            enabled=True,
            label=f"{label}:{path_list.name}",
        )
        if not paths:
            raise ValueError(f"{label} is empty after is_skipped filtering: {path_list}")
        print(f"{label}: skip-filter kept {len(paths)} / {source_count}")
    if limit <= 0 or limit >= len(paths):
        selected = list(paths)
    else:
        rng = random.Random(seed)
        selected = [paths[i] for i in rng.sample(range(len(paths)), limit)]
    print(f"{label}: selected {len(selected)} / {len(paths)} scenes")
    return [str(path) for path in selected]


def _ordered_paths_sha256(paths: list[str]) -> str:
    """Hash an ordered scene list with an unambiguous newline delimiter."""

    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_expected_paths_sha256(
    label: str,
    paths: list[str],
    expected: str | None,
) -> str:
    """Fail before evaluation/training when an ordered scene contract drifts."""

    actual = _ordered_paths_sha256(paths)
    if expected is None:
        return actual
    expected = str(expected).strip().lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError(f"{label} expected SHA-256 is malformed: {expected!r}")
    if actual != expected:
        raise RuntimeError(
            f"{label} ordered scene SHA-256 differs: actual={actual}, "
            f"expected={expected}, count={len(paths)}"
        )
    return actual


def _scene_selection_manifest(paths: list[str]) -> Any:
    """Keep full-corpus provenance small without losing reproducibility."""

    if len(paths) <= 100_000:
        return paths
    return {
        "count": len(paths),
        "sha256_newline_joined_paths": _ordered_paths_sha256(paths),
        "first_paths": paths[:3],
        "last_paths": paths[-3:],
        "note": "Full list omitted from the run directory; source JSON path is recorded in provenance.json.",
    }


def _file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 without loading a checkpoint into RAM."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_encoder_sha256(
    checkpoint_path: Path,
    *,
    use_policy_state: bool,
) -> str:
    """Hash the exact encoder payload selected by the original-DP loader."""

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint is not a mapping: {checkpoint_path}")
    if use_policy_state and isinstance(checkpoint.get("model"), dict):
        state = checkpoint["model"]
    elif isinstance(checkpoint.get("ema_state_dict"), dict):
        state = checkpoint["ema_state_dict"]
    elif isinstance(checkpoint.get("model"), dict):
        state = checkpoint["model"]
    else:
        state = checkpoint
    encoder: dict[str, torch.Tensor] = {}
    for raw_key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        key = str(raw_key).replace("module.", "").replace(
            "._orig_mod.", "."
        ).replace("_orig_mod.", "")
        if key.startswith("encoder."):
            encoder[key] = value.detach().cpu().contiguous()
    if not encoder:
        raise RuntimeError(f"checkpoint contains no encoder tensors: {checkpoint_path}")
    digest = hashlib.sha256()
    for key in sorted(encoder):
        value = encoder[key]
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _is_mine_only_refresh_run(
    *,
    start_epoch: int,
    epochs: int,
    rollout_interval: int,
    faithful_disk_replay: bool,
    resumed_replay: bool,
) -> bool:
    """Whether one update-free refresh can reuse its starting evaluation.

    A faithful HDP refresh epoch only samples a frozen behavior policy and
    writes its replay buffer; it performs no optimizer or EMA update.  If the
    process contains exactly that one epoch, evaluating the unchanged model a
    second time after cache close is redundant.  Both evaluators preserve the
    training RNG transactionally, so reuse is numerically equivalent as well
    as substantially faster.
    """

    interval = int(rollout_interval)
    start = int(start_epoch)
    return bool(
        faithful_disk_replay
        and not resumed_replay
        and interval > 0
        and start == int(epochs)
        and (start - 1) % interval == 0
    )


def _configure_trainable_decoder(
    model: nn.Module, trainable_scope: str = "dit"
) -> list[nn.Parameter]:
    """Freeze the scene encoder and select a conservative DP denoiser scope.

    ``dit`` matches HDP's decoder-only fine-tuning.  ``last_block`` and
    ``output`` are useful stability ablations for the smaller original-DP
    checkpoint: they let AWR change the denoising output without rewriting
    every attention block, which is important when exploration targets are
    noisy or the local reward is only a proxy for full PDM.
    """

    planner = model.module if hasattr(model, "module") else model
    for parameter in planner.parameters():
        parameter.requires_grad_(False)
    scope = str(trainable_scope).lower()
    if scope == "dit":
        modules = [planner.decoder.dit]
    elif scope == "last_block":
        modules = [planner.decoder.dit.blocks[-1], planner.decoder.dit.final_layer]
    elif scope == "output":
        modules = [planner.decoder.dit.final_layer]
    else:
        raise ValueError(
            f"unknown trainable_scope={trainable_scope!r}; expected dit, last_block, or output"
        )
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    # No turn-indicator loss is used by AWR; keeping this head frozen avoids a
    # side objective whose labels are unrelated to the reward-weighted target.
    planner.decoder.turn_indicator_predictor.requires_grad_(False)
    planner.encoder.eval()
    planner.decoder.turn_indicator_predictor.eval()
    return [parameter for parameter in planner.parameters() if parameter.requires_grad]


def _make_optimizer(parameters: list[nn.Parameter], config: dict[str, Any], device: torch.device):
    weight_decay = float(config.get("weight_decay", 1e-4))
    decay, no_decay = [], []
    for parameter in parameters:
        if parameter.ndim <= 1:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs = {"lr": float(config.get("learning_rate", 1e-5)), "betas": (0.9, 0.999)}
    if device.type == "cuda" and bool(config.get("fused_adamw", True)):
        try:
            optimizer = AdamW(groups, fused=True, **kwargs)
            print("Optimizer: fused AdamW")
            return optimizer
        except (TypeError, RuntimeError) as error:
            print(f"Fused AdamW unavailable ({error}); using standard AdamW")
    return AdamW(groups, **kwargs)


@torch.no_grad()
def _commit_epoch_ema_policy_update(
    model: nn.Module,
    behavior_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    decay: float,
) -> dict[str, float]:
    """Accept one conservative Original-DP proposal and reset optimization.

    The HDP paper reports ``EMA=0.05`` but does not define whether that number
    is an update rate, a decay, or an epoch-boundary transaction.  Its released
    NAVSIM configuration disables EMA and the corresponding initialization is
    commented out.  This helper therefore implements the explicit
    Original-DP/T4 interpretation evaluated here: form
    ``accepted = decay * old + (1 - decay) * live``, copy the accepted policy
    back to the live model, and clear Adam state before the next proposal.
    It is a conservative accepted-policy interpolation, not a claim of
    line-for-line public-HDP reproduction.  Parameter objects and compiled
    wrappers are preserved for DDP safety.
    """

    decay = float(decay)
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"ema_decay must be in [0, 1), got {decay}")
    live = model.module if hasattr(model, "module") else model
    old = behavior_model.module if hasattr(behavior_model, "module") else behavior_model
    old_parameters = dict(old.named_parameters())
    delta_sq: torch.Tensor | None = None
    reference_sq: torch.Tensor | None = None
    for name, parameter in live.named_parameters():
        if not parameter.requires_grad:
            continue
        old_parameter = old_parameters[name]
        current_delta = (
            parameter.detach().float() - old_parameter.detach().float()
        ).square().sum()
        current_reference = old_parameter.detach().float().square().sum()
        delta_sq = current_delta if delta_sq is None else delta_sq + current_delta
        reference_sq = (
            current_reference
            if reference_sq is None
            else reference_sq + current_reference
        )
    if delta_sq is None or reference_sq is None:
        raise RuntimeError("epoch EMA commit found no trainable policy parameters")
    proposal_relative_l2 = torch.sqrt(delta_sq / reference_sq.clamp_min(1e-12))

    update_ema(behavior_model, model, decay)
    # load_state_dict copies into the existing tensors. DDP reducers and the
    # optimizer therefore retain valid parameter identities.
    live.load_state_dict(old.state_dict(), strict=True)
    optimizer.state.clear()
    optimizer.zero_grad(set_to_none=True)
    commit_rate = 1.0 - decay
    return {
        "policy_proposal_relative_l2": float(proposal_relative_l2.item()),
        "policy_accepted_relative_l2": float(
            proposal_relative_l2.item() * commit_rate
        ),
        "ema_policy_commit_rate": float(commit_rate),
        "optimizer_state_reset_after_policy_commit": 1.0,
    }


@torch.no_grad()
def _restore_best_policy_transaction(
    model: nn.Module,
    behavior_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Restore both replay policies to one selected deployable EMA payload.

    A rejected epoch is still saved for diagnosis, but it must not silently
    become the starting point of the next full replay pass.  Checkpoints store
    portable keys with nested ``torch.compile`` wrappers removed; active
    training modules may still contain those wrappers.  Translate by canonical
    key rather than unwrapping/recompiling the hot graph.
    """

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("ema_state_dict"), dict
    ):
        raise RuntimeError(
            "best-policy rollback requires an ema_state_dict checkpoint: "
            f"{checkpoint_path}"
        )

    def canonical(key: str) -> str:
        return (
            str(key)
            .replace("module.", "")
            .replace("._orig_mod.", ".")
            .replace("_orig_mod.", "")
        )

    portable = {
        canonical(key): value
        for key, value in checkpoint["ema_state_dict"].items()
    }

    def restore(module: nn.Module) -> None:
        inner = module.module if hasattr(module, "module") else module
        active_state = inner.state_dict()
        translated: dict[str, torch.Tensor] = {}
        missing: list[str] = []
        for active_key in active_state:
            portable_key = canonical(active_key)
            value = portable.get(portable_key)
            if value is None:
                missing.append(portable_key)
            else:
                translated[active_key] = value
        active_canonical = {canonical(key) for key in active_state}
        unexpected = sorted(set(portable) - active_canonical)
        if missing or unexpected:
            raise RuntimeError(
                "best-policy rollback checkpoint is architecture-incompatible: "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
        inner.load_state_dict(translated, strict=True)

    restore(model)
    restore(behavior_model)
    optimizer.state.clear()
    optimizer.zero_grad(set_to_none=True)
    return {
        "restored_checkpoint": str(checkpoint_path.resolve()),
        "restored_epoch": int(checkpoint.get("epoch", -1)),
        "optimizer_state_reset": True,
    }


def _resolve_amp_dtype(requested: str, device: torch.device) -> str:
    requested = str(requested).lower()
    if requested in {"off", "none", "false"} or device.type != "cuda":
        return "off"
    if requested in {"fp16", "float16", "half"}:
        return "fp16"
    if requested in {"bf16", "bfloat16"}:
        return "bf16"
    if requested != "auto":
        raise ValueError(f"unsupported amp_dtype={requested}")
    capability = torch.cuda.get_device_capability(device)
    return "bf16" if capability[0] >= 8 else "fp16"


def _configure_cuda_runtime(device: torch.device) -> dict[str, Any]:
    """Enable the safe, high-throughput CUDA switches used by T4 training.

    The AWR loop has fixed tensor shapes (80 future steps and 321 agents), so
    the H100 can use TF32 for fp32 matmuls, bf16 autocast for the denoiser, and
    the compiler's shape-specialised kernels.  These flags do not alter the
    model architecture or the reward calculation.
    """

    if device.type != "cuda":
        return {"cuda": False, "tf32": False, "cudnn_benchmark": False}

    # The non-login experiment shell does not source the cluster CUDA setup,
    # while TorchInductor invokes nvcc by name when compiling a CUDA kernel.
    # Make the installed toolkit discoverable inside the training process so
    # --compile is reproducible from both an interactive shell and torchrun.
    cuda_bin = Path("/usr/local/cuda/bin")
    if cuda_bin.is_dir():
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        cuda_bin_str = str(cuda_bin)
        if cuda_bin_str not in path_entries:
            os.environ["PATH"] = os.pathsep.join(
                [cuda_bin_str, *[entry for entry in path_entries if entry]]
            )

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # The planner contains MultiheadAttention.  Keep the fused SDPA kernels
    # enabled; PyTorch will select the kernel supported by the active dtype and
    # sequence length.
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    # Scene-wise AWR always reuses the same static geometry shapes.
    torch.backends.cudnn.benchmark = True
    return {
        "cuda": True,
        "device": str(device),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "nvcc": shutil.which("nvcc"),
        "flash_sdp": bool(torch.backends.cuda.flash_sdp_enabled())
        if hasattr(torch.backends.cuda, "flash_sdp_enabled")
        else None,
        "mem_efficient_sdp": bool(torch.backends.cuda.mem_efficient_sdp_enabled())
        if hasattr(torch.backends.cuda, "mem_efficient_sdp_enabled")
        else None,
    }


def _compile_planner_modules(
    model: nn.Module,
    compile_config: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    """Compile fixed-shape encoder/DiT modules, leaving control flow eager.

    Compiling the whole planner would capture dictionaries, reward-side
    metadata, and the DPM solver's Python control flow.  Compiling only the
    tensor-heavy encoder and denoiser gives the useful Inductor/Triton speedup
    while preserving the original DP inference path and checkpoint format.
    """

    result: dict[str, Any] = {"label": label, "requested": bool(compile_config.get("enabled", True))}
    if not result["requested"]:
        result["enabled"] = False
        result["reason"] = "disabled_by_config"
        return result
    if not hasattr(torch, "compile"):
        result["enabled"] = False
        result["reason"] = "torch.compile_unavailable"
        return result

    planner = model.module if hasattr(model, "module") else model
    mode = str(compile_config.get("mode", "max-autotune"))
    backend = str(compile_config.get("backend", "inductor"))
    fullgraph = bool(compile_config.get("fullgraph", False))
    dynamic = bool(compile_config.get("dynamic", False))
    result.update({"enabled": True, "mode": mode, "backend": backend, "fullgraph": fullgraph, "dynamic": dynamic})

    targets: list[tuple[str, nn.Module, Any, str]] = []
    if bool(compile_config.get("encoder", True)):
        targets.append(("encoder", planner.encoder, planner, "encoder"))
    if bool(compile_config.get("decoder", True)):
        targets.append(("decoder_dit", planner.decoder.dit, planner.decoder, "dit"))

    modules: dict[str, Any] = {}
    for name, module, parent, attr in targets:
        if hasattr(module, "_orig_mod"):
            modules[name] = {"compiled": True, "already_compiled": True}
            continue
        try:
            compiled = torch.compile(
                module,
                backend=backend,
                mode=mode,
                fullgraph=fullgraph,
                dynamic=dynamic,
            )
            setattr(parent, attr, compiled)
            modules[name] = {"compiled": True, "already_compiled": False}
        except Exception as error:  # pragma: no cover - depends on local compiler
            modules[name] = {
                "compiled": False,
                "error": f"{type(error).__name__}: {error}",
            }
    result["modules"] = modules
    return result


def _unwrap_compiled_modules(model: nn.Module) -> None:
    """Replace OptimizedModule wrappers with their original modules in-place."""

    planner = model.module if hasattr(model, "module") else model
    for parent, attr in ((planner, "encoder"), (planner.decoder, "dit")):
        module = getattr(parent, attr, None)
        original = getattr(module, "_orig_mod", None)
        if original is not None:
            setattr(parent, attr, original)


def _warmup_compiled_models(
    policy_model: nn.Module,
    behavior_model: nn.Module,
    model_args,
    scene_path: str | list[str],
    rollout_config: AWRRolloutConfig,
    reward_config: RewardConfig,
    device: torch.device,
    exploration_reference_model: nn.Module | None = None,
    exploration_reference_model_args: Any | None = None,
) -> dict[str, Any]:
    """Trigger the production-sized native graph and one guided bucket.

    The deployment evaluator, train selector, native rollout and sparse
    PlannerRFT branch use different static batch shapes.  Warming only one
    validation scene lets the cheap monitor consume Inductor's recompile
    budget before the long rollout starts.  A packed scene list therefore
    warms the native production shape first; guidance is then warmed on one
    scene, which rounds to exactly one configured guidance bucket.  All
    outputs are discarded and every RNG stream is restored below.
    """

    scene_paths = (
        [str(scene_path)]
        if isinstance(scene_path, (str, Path))
        else [str(path) for path in scene_path]
    )
    if not scene_paths:
        raise ValueError("compile warmup requires at least one scene")
    bulk_data = (
        load_scene(scene_paths[0], device)
        if len(scene_paths) == 1
        else _load_scene_batch(scene_paths, device, workers=1)
    )
    single_data = (
        bulk_data
        if len(scene_paths) == 1
        else load_scene(scene_paths[0], device)
    )
    # Compilation warmup is an implementation detail, not exploration.  Keep
    # it from changing rollout candidates, replay sampling or diffusion noise
    # relative to an eager run with the same seed.
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.get_rng_state()
    cuda_rng_state = (
        torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    )
    start = time.perf_counter()
    try:
        native_warmup_config = copy.copy(rollout_config)
        guidance_enabled = bool(
            native_warmup_config.plannerrft_guided_exploration
            and exploration_reference_model is not None
        )
        # The packed call must compile the high-frequency native B*K graph,
        # not a worst-case all-guided B*K + B*K graph that production never
        # executes because the Lt90 trigger is sparse.
        native_warmup_config.plannerrft_guided_exploration = False
        guided_warmup_config = copy.copy(rollout_config)
        if guidance_enabled:
            # Force one bucketed guided call so the first measured training
            # batch does not pay a hidden reference/guidance compile. RNG is
            # restored below, and warmup outputs never enter replay.
            guided_warmup_config.plannerrft_trigger_best_reward_lt = 1.000001
        # Warm both copies: the behavior copy is used for evaluation and the
        # policy copy is used by the AWR loss.  Keeping these calls here makes
        # the first reported metric a post-compile metric rather than a compile
        # plus inference measurement.
        def warm_native(model_to_warm: nn.Module) -> None:
            kwargs = {
                "exploration_reference_model": exploration_reference_model,
                "exploration_reference_model_args": (
                    exploration_reference_model_args
                ),
            }
            if len(scene_paths) == 1:
                rollout_and_score_scene(
                    model_to_warm,
                    model_args,
                    bulk_data,
                    native_warmup_config,
                    reward_config,
                    device,
                    **kwargs,
                )
            else:
                rollout_and_score_scene_batch(
                    model_to_warm,
                    model_args,
                    bulk_data,
                    native_warmup_config,
                    reward_config,
                    device,
                    **kwargs,
                )

        warm_native(behavior_model)
        warm_native(policy_model)
        if guidance_enabled:
            rollout_and_score_scene(
                behavior_model,
                model_args,
                single_data,
                guided_warmup_config,
                reward_config,
                device,
                exploration_reference_model=exploration_reference_model,
                exploration_reference_model_args=exploration_reference_model_args,
            )
        warm_native(behavior_model)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        return {
            "success": True,
            "elapsed_sec": elapsed,
            "native_scene_batch": len(scene_paths),
            "guided_bucket_warmed": guidance_enabled,
        }
    except Exception as error:  # pragma: no cover - compiler/backend dependent
        _unwrap_compiled_modules(policy_model)
        _unwrap_compiled_modules(behavior_model)
        if exploration_reference_model is not None:
            _unwrap_compiled_modules(exploration_reference_model)
        return {"success": False, "elapsed_sec": time.perf_counter() - start, "error": f"{type(error).__name__}: {error}"}
    finally:
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.set_rng_state(torch_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, device)


def _trajectory_metrics(trajectory: torch.Tensor, data: dict[str, torch.Tensor]) -> dict[str, float]:
    trajectory = trajectory.detach().float()
    xy = trajectory[:, :2]
    path_len = float(torch.diff(xy, dim=0).norm(dim=-1).sum().item())
    result = {"path_len": path_len}
    gt = data.get("ego_agent_future")
    if gt is None:
        return result
    if gt.dim() == 3:
        gt = gt[0]
    gt_xy = gt[:, :2].float()
    valid = gt_xy.abs().sum(dim=-1) > 0.1
    if not bool(valid.any()):
        return result
    n = min(int(valid.sum().item()), trajectory.shape[0])
    gt_xy = gt_xy[valid][:n]
    pred_xy = xy[:n]
    result["ade"] = float((pred_xy - gt_xy).norm(dim=-1).mean().item())
    result["fde"] = float((pred_xy[-1] - gt_xy[-1]).norm().item())
    result["gt_path_len"] = float(torch.diff(gt_xy, dim=0).norm(dim=-1).sum().item()) if n > 1 else 0.0
    return result


def _candidate_diversity_metrics(
    trajectories: torch.Tensor, rewards: list[Any]
) -> dict[str, float]:
    """Measure useful K-sample spread; this is observation-only."""

    xy = trajectories.detach().float()[..., :2]
    K = int(xy.shape[0])
    totals = np.asarray([float(reward.total) for reward in rewards], dtype=np.float64)
    result = {
        "candidate_reward_std": float(np.nanstd(totals)),
        "best_vs_det_reward_gain": float(np.nanmax(totals) - totals[0]),
    }
    if K < 2:
        result.update(
            {
                "candidate_endpoint_spread_mean": 0.0,
                "candidate_endpoint_spread_max": 0.0,
                "candidate_pairwise_ade": 0.0,
                "candidate_temporal_spread": 0.0,
            }
        )
        return result
    pair = torch.triu_indices(K, K, offset=1, device=xy.device)
    pairwise_t = (xy[pair[0]] - xy[pair[1]]).norm(dim=-1)
    endpoint_pairwise = pairwise_t[:, -1]
    centroid = xy.mean(dim=0, keepdim=True)
    result.update(
        {
            "candidate_endpoint_spread_mean": float(endpoint_pairwise.mean().item()),
            "candidate_endpoint_spread_max": float(endpoint_pairwise.max().item()),
            "candidate_pairwise_ade": float(pairwise_t.mean(dim=-1).mean().item()),
            "candidate_temporal_spread": float(
                (xy - centroid).norm(dim=-1).mean().item()
            ),
        }
    )
    return result


@torch.no_grad()
def _slice_eval_scene_data(
    data: dict[str, torch.Tensor], index: int
) -> dict[str, torch.Tensor]:
    """Keep one scene's loader batch dimension for per-scene metrics."""

    batch_size = int(data["ego_current_state"].shape[0])
    return {
        key: value[index : index + 1]
        if isinstance(value, torch.Tensor)
        and value.dim() > 0
        and value.shape[0] == batch_size
        else value
        for key, value in data.items()
    }


def _iter_prefetched_eval_batches(
    scene_paths: list[str],
    batch_size: int,
    scene_load_workers: int,
    prefetch_batches: int,
):
    """Overlap full-scene NPZ loading with the preceding CUDA evaluation.

    The background worker only reads and merges CPU tensors.  CUDA transfer,
    heading conversion, model inference, reward calculation and row emission
    remain ordered on the caller thread.  With depth zero this yields the
    historical direct-loader path; positive depth changes scheduling only.
    """

    eval_batch_size = max(1, int(batch_size))
    starts = iter(range(0, len(scene_paths), eval_batch_size))
    cpu = torch.device("cpu")
    exhausted = object()

    def batch_spec(batch_start: int):
        batch_paths = scene_paths[batch_start : batch_start + eval_batch_size]
        real_batch_count = len(batch_paths)
        model_batch_paths = batch_paths
        if (
            real_batch_count > 0
            and len(scene_paths) >= eval_batch_size
            and real_batch_count < eval_batch_size
        ):
            pad_count = eval_batch_size - real_batch_count
            model_batch_paths = batch_paths + [
                batch_paths[index % real_batch_count]
                for index in range(pad_count)
            ]
        return batch_start, batch_paths, model_batch_paths

    def prepare_next():
        try:
            batch_start = next(starts)
        except StopIteration:
            return exhausted
        spec = batch_spec(batch_start)
        data = _load_scene_batch(
            spec[2],
            cpu,
            workers=max(1, int(scene_load_workers)),
            defer_full_scene_heading_transforms=True,
        )
        return (*spec, data)

    depth = max(0, int(prefetch_batches))
    if depth == 0:
        for batch_start in starts:
            yield (*batch_spec(batch_start), None)
        return

    # A single coordinator preserves generator and NPZ order.  Depth two is
    # sufficient for one active read plus one queued read without multiplying
    # the already-large full-scene host-memory footprint.
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = deque(executor.submit(prepare_next) for _ in range(depth))
        while pending:
            prepared = pending.popleft().result()
            if prepared is exhausted:
                break
            pending.append(executor.submit(prepare_next))
            yield prepared


def _move_prefetched_eval_data(
    data: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    """Move one deferred-heading evaluation batch to its CUDA device."""

    moved = {
        key: value.to(device=device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in data.items()
    }
    apply_npz_heading_transforms(moved)
    return moved


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    model_args,
    scene_paths: list[str],
    rollout_config: AWRRolloutConfig,
    reward_config: RewardConfig,
    device: torch.device,
    output_dir: Path | None = None,
    epoch: int = 0,
    save_rollouts: int = 0,
    scene_batch_size: int = 1,
    scene_load_workers: int = 1,
    scene_prefetch_batches: int = 0,
    log_progress: bool = False,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Evaluate deterministic DP plus a fresh candidate group.

    Evaluation keeps exactly the same per-scene reward and metric definitions,
    but packs scenes into the same decoder batch used by full-corpus training.
    The old scene-at-a-time evaluator made a 2,048-scene validation set spend
    thousands of launches in Python and left seven GPUs idle behind rank 0.
    """

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    eval_batch_size = max(1, int(scene_batch_size))
    total_batches = max(1, math.ceil(len(scene_paths) / eval_batch_size))
    progress_interval = max(1, total_batches // 10)
    prepared_batches = _iter_prefetched_eval_batches(
        scene_paths,
        eval_batch_size,
        scene_load_workers,
        scene_prefetch_batches,
    )
    for batch_start, batch_paths, model_batch_paths, prefetched_data in prepared_batches:
        real_batch_count = len(batch_paths)
        if prefetched_data is None:
            data = _load_scene_batch(
                model_batch_paths,
                device,
                workers=max(1, int(scene_load_workers)),
            )
        else:
            data = _move_prefetched_eval_data(prefetched_data, device)
        if len(model_batch_paths) == 1:
            batch_rollouts = [
                rollout_and_score_scene(
                    model, model_args, data, rollout_config, reward_config, device
                )
            ]
        else:
            batch_rollouts = rollout_and_score_scene_batch(
                model, model_args, data, rollout_config, reward_config, device
            )
        for offset, (scene_path, rollout) in enumerate(
            zip(batch_paths, batch_rollouts[:real_batch_count])
        ):
            index = batch_start + offset
            scene_data = _slice_eval_scene_data(data, offset)
            det = rollout.rewards[0]
            best = max(rollout.rewards, key=lambda reward: reward.total)

            def configured_safe(reward: Any) -> bool:
                return bool(
                    reward.collision_step is None
                    and (not reward_config.rb_gate_enabled or not reward.rb_crossing)
                    and (not reward_config.lane_gate_enabled or not reward.lane_crossing)
                    and (
                        not reward_config.static_collision_enabled
                        or not reward_config.sc_gate_enabled
                        or not reward.static_crossing
                    )
                    and not reward.kinematic_violated
                    and reward.red_light >= -0.5
                )

            safe_rewards = [
                reward for reward in rollout.rewards if configured_safe(reward)
            ]
            best_safe = (
                max(safe_rewards, key=lambda reward: reward.total)
                if safe_rewards
                else None
            )
            traj_metrics = _trajectory_metrics(
                rollout.trajectories[0], scene_data
            )
            diversity_metrics = _candidate_diversity_metrics(
                rollout.trajectories, rollout.rewards
            )
            first_waypoint_metrics = {
                key: value
                for key, value in rollout.diagnostics.items()
                if key.startswith("first_waypoint_")
                or key.startswith("stop_turn_")
            }
            candidate_count = max(1, len(rollout.rewards))
            candidate_totals = np.asarray(
                [float(reward.total) for reward in rollout.rewards], dtype=np.float64
            )
            finite_candidate_totals = candidate_totals[
                np.isfinite(candidate_totals)
            ]
            zero_tol = 1e-8
            row: dict[str, Any] = {
                "scene_index": index,
                "scene_path": scene_path,
                "det_reward": det.total,
                "best_group_reward": best.total,
                "mean_group_reward": float(
                    np.mean([reward.total for reward in rollout.rewards])
                ),
                "det_collision": float(det.collision_step is not None),
                "det_rb_crossing": float(det.rb_crossing),
                "det_lane_crossing": float(det.lane_crossing),
                "det_kinematic_violation": float(det.kinematic_violated),
                "det_off_road_fraction": det.off_road_fraction,
                "det_progress": det.progress,
                "det_safety": det.safety,
                "det_centerline": det.centerline,
                "det_smoothness": det.smoothness,
                "candidate_count": float(len(rollout.rewards)),
                "safe_candidate_count": float(len(safe_rewards)),
                "candidate_collision_count": float(
                    sum(
                        reward.collision_step is not None
                        for reward in rollout.rewards
                    )
                ),
                "candidate_rb_crossing_count": float(
                    sum(bool(reward.rb_crossing) for reward in rollout.rewards)
                ),
                "candidate_kinematic_violation_count": float(
                    sum(
                        bool(reward.kinematic_violated)
                        for reward in rollout.rewards
                    )
                ),
                "safe_candidate_fraction": float(len(safe_rewards) / candidate_count),
                "candidate_collision_fraction": float(
                    sum(reward.collision_step is not None for reward in rollout.rewards)
                    / candidate_count
                ),
                "candidate_rb_crossing_fraction": float(
                    sum(bool(reward.rb_crossing) for reward in rollout.rewards)
                    / candidate_count
                ),
                "candidate_lane_crossing_fraction": float(
                    sum(bool(reward.lane_crossing) for reward in rollout.rewards)
                    / candidate_count
                ),
                "candidate_kinematic_violation_fraction": float(
                    sum(bool(reward.kinematic_violated) for reward in rollout.rewards)
                    / candidate_count
                ),
                "det_zero_reward": float(abs(float(det.total)) <= zero_tol),
                "zero_reward_candidate_fraction": float(
                    np.mean(np.abs(finite_candidate_totals) <= zero_tol)
                )
                if finite_candidate_totals.size
                else float("nan"),
                "all_zero_reward_group": float(
                    bool(finite_candidate_totals.size)
                    and bool(np.all(np.abs(finite_candidate_totals) <= zero_tol))
                ),
                "all_equal_reward_group": float(
                    bool(finite_candidate_totals.size)
                    and float(
                        np.max(finite_candidate_totals)
                        - np.min(finite_candidate_totals)
                    )
                    <= zero_tol
                ),
                "reward_unique_count": float(
                    np.unique(np.round(finite_candidate_totals, decimals=8)).size
                ),
                "det_zero_collision": float(
                    abs(float(det.total)) <= zero_tol
                    and det.collision_step is not None
                ),
                "det_zero_rb_crossing": float(
                    abs(float(det.total)) <= zero_tol and bool(det.rb_crossing)
                ),
                "det_zero_kinematic": float(
                    abs(float(det.total)) <= zero_tol
                    and bool(det.kinematic_violated)
                ),
                "det_zero_without_collision_rb_kinematic": float(
                    abs(float(det.total)) <= zero_tol
                    and det.collision_step is None
                    and not bool(det.rb_crossing)
                    and not bool(det.kinematic_violated)
                ),
                "best_safe_reward": (
                    float(best_safe.total)
                    if best_safe is not None
                    else float("nan")
                ),
                **traj_metrics,
                **diversity_metrics,
                **first_waypoint_metrics,
            }
            rows.append(row)
            if output_dir is not None and index < save_rollouts:
                stem = f"scene_{index:04d}"
                np.savez_compressed(
                    output_dir / f"{stem}.npz",
                    trajectories=rollout.trajectories.detach().cpu().numpy(),
                    noise_scales=rollout.noise_scales.detach().cpu().numpy(),
                    weights=rollout.weights.detach().cpu().numpy(),
                )
                _write_json(
                    output_dir / f"{stem}.json",
                    rollout_to_json(rollout, scene_path),
                )

        completed_batches = batch_start // eval_batch_size + 1
        if bool(log_progress) and (
            completed_batches == 1
            or completed_batches == total_batches
            or completed_batches % progress_interval == 0
        ):
            print(
                f"eval epoch {int(epoch)} rank-0 shard "
                f"{min(batch_start + real_batch_count, len(scene_paths))}/"
                f"{len(scene_paths)} scenes "
                f"({100.0 * completed_batches / total_batches:.1f}%)",
                flush=True,
            )

    if not rows:
        return {"scene_count": 0.0}, rows
    keys = [
        "det_reward",
        "best_group_reward",
        "mean_group_reward",
        "det_collision",
        "det_rb_crossing",
        "det_lane_crossing",
        "det_kinematic_violation",
        "det_off_road_fraction",
        "det_progress",
        "det_safety",
        "det_centerline",
        "det_smoothness",
        "candidate_count",
        "safe_candidate_count",
        "candidate_collision_count",
        "candidate_rb_crossing_count",
        "candidate_kinematic_violation_count",
        "safe_candidate_fraction",
        "candidate_collision_fraction",
        "candidate_rb_crossing_fraction",
        "candidate_lane_crossing_fraction",
        "candidate_kinematic_violation_fraction",
        "det_zero_reward",
        "zero_reward_candidate_fraction",
        "all_zero_reward_group",
        "all_equal_reward_group",
        "reward_unique_count",
        "det_zero_collision",
        "det_zero_rb_crossing",
        "det_zero_kinematic",
        "det_zero_without_collision_rb_kinematic",
        "candidate_reward_std",
        "best_vs_det_reward_gain",
        "candidate_endpoint_spread_mean",
        "candidate_endpoint_spread_max",
        "candidate_pairwise_ade",
        "candidate_temporal_spread",
        "best_safe_reward",
        "path_len",
        "ade",
        "fde",
        "first_waypoint_gate_enabled",
        "first_waypoint_gate_low_speed",
        "first_waypoint_gate_rejected_fraction",
        "first_waypoint_gate_rejected_count",
        "first_waypoint_displacement_p95_m",
        "first_waypoint_lateral_abs_p95_m",
        "first_waypoint_path_tangent_abs_deg_p95",
        "first_waypoint_backward_fraction",
        "stop_turn_scene",
        "stop_turn_first_waypoint_displacement_p95_m",
        "stop_turn_first_waypoint_path_tangent_abs_deg_p95",
        "stop_turn_first_waypoint_gate_rejected_fraction",
    ]
    summary: dict[str, float] = {"scene_count": float(len(rows))}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
        if values:
            summary[f"mean_{key}"] = float(np.mean(values))
            summary[f"p10_{key}"] = float(np.percentile(values, 10))
            summary[f"p50_{key}"] = float(np.median(values))
            summary[f"p90_{key}"] = float(np.percentile(values, 90))
    return summary, rows


_EVAL_SUMMARY_KEYS = (
    "det_reward",
    "best_group_reward",
    "mean_group_reward",
    "det_collision",
    "det_rb_crossing",
    "det_lane_crossing",
    "det_kinematic_violation",
    "det_off_road_fraction",
    "det_progress",
    "det_safety",
    "det_centerline",
    "det_smoothness",
    "candidate_count",
    "safe_candidate_count",
    "candidate_collision_count",
    "candidate_rb_crossing_count",
    "candidate_kinematic_violation_count",
    "safe_candidate_fraction",
    "candidate_collision_fraction",
    "candidate_rb_crossing_fraction",
    "candidate_lane_crossing_fraction",
    "candidate_kinematic_violation_fraction",
    "det_zero_reward",
    "zero_reward_candidate_fraction",
    "all_zero_reward_group",
    "all_equal_reward_group",
    "reward_unique_count",
    "det_zero_collision",
    "det_zero_rb_crossing",
    "det_zero_kinematic",
    "det_zero_without_collision_rb_kinematic",
    "candidate_reward_std",
    "best_vs_det_reward_gain",
    "candidate_endpoint_spread_mean",
    "candidate_endpoint_spread_max",
    "candidate_pairwise_ade",
    "candidate_temporal_spread",
    "best_safe_reward",
    "path_len",
    "ade",
    "fde",
    "first_waypoint_gate_enabled",
    "first_waypoint_gate_low_speed",
    "first_waypoint_gate_rejected_fraction",
    "first_waypoint_gate_rejected_count",
    "first_waypoint_displacement_p95_m",
    "first_waypoint_lateral_abs_p95_m",
    "first_waypoint_path_tangent_abs_deg_p95",
    "first_waypoint_backward_fraction",
    "stop_turn_scene",
    "stop_turn_first_waypoint_displacement_p95_m",
    "stop_turn_first_waypoint_path_tangent_abs_deg_p95",
    "stop_turn_first_waypoint_gate_rejected_fraction",
)


def _summarize_eval_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Summarize an already-evaluated, fixed scene subset."""

    summary: dict[str, float] = {"scene_count": float(len(rows))}
    for key in _EVAL_SUMMARY_KEYS:
        values = [
            float(row[key])
            for row in rows
            if key in row and math.isfinite(float(row[key]))
        ]
        if values:
            summary[f"mean_{key}"] = float(np.mean(values))
            summary[f"p10_{key}"] = float(np.percentile(values, 10))
            summary[f"p50_{key}"] = float(np.median(values))
            summary[f"p90_{key}"] = float(np.percentile(values, 90))
    return summary


_TRAIN_PERCENTILE_KEYS = frozenset(
    {
        "loss",
        "grad_norm",
        "det_reward",
        "best_reward",
        "reward_std",
        "effective_sample_size",
        "top1_weight_share",
        "best_vs_det_reward_gain",
    }
)


def _compact_numeric_row_summary(
    rows: list[dict[str, Any]], *, include_percentile_values: bool = False
) -> dict[str, Any]:
    """Reduce a rank-local rollout row list before distributed collection.

    A full refresh produces roughly 680k Python dictionaries per rank.  Sending
    all of them through ``gather_object`` costs minutes of CPU serialization and
    tens of GiB even though the exact per-scene trajectories/rewards/weights are
    already durably stored in the replay memmaps.  This compact representation
    preserves the scalar means used by the W&B/JSON summary. Replay epochs may
    additionally retain only the eight small scalar vectors needed to reproduce
    their historical p50/p90 fields exactly, without serializing every metric
    dictionary.
    """

    scene_count = 0
    numeric: dict[str, list[float | int]] = {}
    percentile_values: dict[str, list[float]] = {}
    for row in rows:
        scene_count += int(row.get("scene_count", 1))
        for key, value in row.items():
            if key in {"scene_count", "epoch"} or not isinstance(
                value, (int, float)
            ):
                continue
            number = float(value)
            if not math.isfinite(number):
                continue
            slot = numeric.setdefault(key, [0, 0.0])
            slot[0] = int(slot[0]) + 1
            slot[1] = float(slot[1]) + number
            if include_percentile_values and key in _TRAIN_PERCENTILE_KEYS:
                percentile_values.setdefault(key, []).append(number)
    return {
        "row_count": len(rows),
        "scene_count": scene_count,
        "numeric": {
            key: {"count": int(value[0]), "sum": float(value[1])}
            for key, value in numeric.items()
        },
        "percentile_values": percentile_values,
    }


def _merge_compact_numeric_row_summaries(
    summaries: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, float]]:
    """Merge rank-local compact summaries on rank zero."""

    scene_count = sum(int(summary.get("scene_count", 0)) for summary in summaries)
    row_count = sum(int(summary.get("row_count", 0)) for summary in summaries)
    counts: dict[str, int] = {}
    sums: dict[str, float] = {}
    percentile_values: dict[str, list[float]] = {}
    for summary in summaries:
        for key, stats in summary.get("numeric", {}).items():
            counts[key] = counts.get(key, 0) + int(stats["count"])
            sums[key] = sums.get(key, 0.0) + float(stats["sum"])
        for key, values in summary.get("percentile_values", {}).items():
            percentile_values.setdefault(key, []).extend(float(value) for value in values)
    merged: dict[str, Any] = {
        "scene_count": scene_count,
        "row_count": row_count,
        "scene_rows_materialized": 0,
        "compact_distributed_summary": 1,
        "numeric_percentiles_omitted": int(not bool(percentile_values)),
    }
    for key in sorted(sums):
        if counts[key] > 0:
            merged[f"mean_{key}"] = sums[key] / counts[key]
    for key, values in sorted(percentile_values.items()):
        if values:
            merged[f"p50_{key}"] = float(np.median(values))
            merged[f"p90_{key}"] = float(np.percentile(values, 90))
    return merged, sums


@contextlib.contextmanager
def _preserve_training_rng(device: torch.device):
    """Keep validation sampling from changing subsequent training noise.

    K-sample evaluation is intentionally stochastic, but validation size and
    cadence are observability choices.  Without this transaction, changing
    either one advances Python/NumPy/Torch RNG and silently changes the next
    replay epoch's diffusion noise.  Restore only RNG state; model state and
    evaluation outputs are unaffected.
    """

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = (
        torch.cuda.get_rng_state(device)
        if device.type == "cuda" and torch.cuda.is_available()
        else None
    )
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)


def _distributed_evaluate_model_impl(
    model: nn.Module,
    model_args,
    scene_paths: list[str],
    rollout_config: AWRRolloutConfig,
    reward_config: RewardConfig,
    device: torch.device,
    output_dir: Path | None,
    epoch: int,
    save_rollouts: int,
    scene_batch_size: int,
    scene_load_workers: int,
    scene_prefetch_batches: int,
    distributed: bool,
    rank: int,
    world_size: int,
    object_group: Any,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Evaluate a fixed scene list across all ranks and reduce only JSON rows.

    The training job is already launched with eight DDP workers.  Validation
    must use those workers too: each rank owns a deterministic strided slice,
    runs the normal batched evaluator locally, and sends only per-scene scalar
    diagnostics to rank zero through the CPU/Gloo side group.  No model,
    scene tensor, or candidate trajectory is transferred between ranks.

    Non-main ranks return empty results after the gather; callers perform file
    writes and W&B logging only on rank zero.  ``scene_index`` is restored to
    the original global ordering before aggregation so hard-set mining remains
    reproducible and independent of world size.
    """

    if not distributed:
        return evaluate_model(
            model,
            model_args,
            scene_paths,
            rollout_config,
            reward_config,
            device,
            output_dir,
            epoch=epoch,
            save_rollouts=save_rollouts,
            scene_batch_size=scene_batch_size,
            scene_load_workers=scene_load_workers,
            scene_prefetch_batches=scene_prefetch_batches,
        )

    local_indices = list(range(rank, len(scene_paths), world_size))
    local_paths = [scene_paths[index] for index in local_indices]
    local_output_dir = (
        output_dir / f"rank_{rank:04d}" if output_dir is not None else None
    )
    _, local_rows = evaluate_model(
        model,
        model_args,
        local_paths,
        rollout_config,
        reward_config,
        device,
        local_output_dir,
        epoch=epoch,
        save_rollouts=save_rollouts,
        scene_batch_size=scene_batch_size,
        scene_load_workers=scene_load_workers,
        scene_prefetch_batches=scene_prefetch_batches,
        log_progress=rank == 0,
    )
    for local_offset, row in enumerate(local_rows):
        row["scene_index"] = local_indices[local_offset]

    gathered: list[dict[str, Any] | None] | None = (
        [None] * world_size if rank == 0 else None
    )
    if object_group is None:
        raise RuntimeError("distributed evaluation requires the CPU/Gloo object group")
    dist.gather_object(
        local_rows,
        gathered,
        dst=0,
        group=object_group,
    )
    if rank != 0:
        return {}, []

    assert gathered is not None
    rows = [
        row
        for rank_rows in gathered
        if rank_rows
        for row in rank_rows
    ]
    rows.sort(key=lambda row: int(row.get("scene_index", 0)))
    return _summarize_eval_rows(rows), rows


def _distributed_evaluate_model(
    model: nn.Module,
    model_args,
    scene_paths: list[str],
    rollout_config: AWRRolloutConfig,
    reward_config: RewardConfig,
    device: torch.device,
    output_dir: Path | None,
    epoch: int,
    save_rollouts: int,
    scene_batch_size: int,
    scene_load_workers: int,
    scene_prefetch_batches: int,
    distributed: bool,
    rank: int,
    world_size: int,
    object_group: Any,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Evaluate without coupling monitoring cadence to optimization RNG."""

    with _preserve_training_rng(device):
        return _distributed_evaluate_model_impl(
            model,
            model_args,
            scene_paths,
            rollout_config,
            reward_config,
            device,
            output_dir,
            epoch,
            save_rollouts,
            scene_batch_size,
            scene_load_workers,
            scene_prefetch_batches,
            distributed,
            rank,
            world_size,
            object_group,
        )


def _select_hard_eval_rows(
    rows: list[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    """Select a fixed baseline-mined hard set without checkpoint cherry-picking."""

    if count <= 0 or count >= len(rows):
        return list(rows)

    def priority(row: dict[str, Any]) -> tuple[int, float, float]:
        failure = bool(
            float(row.get("det_collision", 0.0)) > 0.5
            or float(row.get("det_rb_crossing", 0.0)) > 0.5
            or float(row.get("det_kinematic_violation", 0.0)) > 0.5
            or float(row.get("all_zero_reward_group", 0.0)) > 0.5
        )
        reward = float(row.get("det_reward", 0.0))
        ade = float(row.get("ade", 0.0))
        return (0 if failure else 1, reward, -ade)

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in sorted(rows, key=priority):
        path = str(row.get("scene_path", ""))
        if path in seen:
            continue
        seen.add(path)
        selected.append(row)
        if len(selected) >= count:
            break
    return selected


def _save_rollout(output_dir: Path, scene_index: int, scene_path: str, rollout: AWRRollout) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"scene_{scene_index:04d}"
    np.savez_compressed(
        output_dir / f"{stem}.npz",
        trajectories=rollout.trajectories.detach().cpu().numpy(),
        noise_scales=rollout.noise_scales.detach().cpu().numpy(),
        weights=rollout.weights.detach().cpu().numpy(),
    )
    _write_json(output_dir / f"{stem}.json", rollout_to_json(rollout, scene_path))


def _rollout_to_cpu(rollout: AWRRollout) -> AWRRollout:
    """Detach one rollout so it can live in the HDP-style replay buffer."""

    return AWRRollout(
        trajectories=rollout.trajectories.detach().cpu(),
        noise_scales=rollout.noise_scales.detach().cpu(),
        rewards=rollout.rewards,
        weights=rollout.weights.detach().cpu(),
        reward_data={
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
            for key, value in rollout.reward_data.items()
        },
        scene_encoding=(
            rollout.scene_encoding.detach().cpu()
            if rollout.scene_encoding is not None
            else None
        ),
        diagnostics=dict(rollout.diagnostics),
    )


def _rollout_to_device(rollout: AWRRollout, device: torch.device) -> AWRRollout:
    """Move a replay record back to the rank-local device for regression."""

    return AWRRollout(
        trajectories=rollout.trajectories.to(device=device),
        noise_scales=rollout.noise_scales.to(device=device),
        rewards=rollout.rewards,
        weights=rollout.weights.to(device=device),
        reward_data={
            key: value.to(device=device) if isinstance(value, torch.Tensor) else value
            for key, value in rollout.reward_data.items()
        },
        scene_encoding=(
            rollout.scene_encoding.to(device=device)
            if rollout.scene_encoding is not None
            else None
        ),
        diagnostics=dict(rollout.diagnostics),
    )


def _merge_scene_data(datas: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Concatenate loader outputs into one scene-batched observation."""

    if not datas:
        raise ValueError("cannot merge an empty scene batch")
    if len(datas) == 1:
        return datas[0]
    merged: dict[str, torch.Tensor] = {}
    for key in datas[0]:
        values = [data[key] for data in datas]
        first = values[0]
        if (
            isinstance(first, torch.Tensor)
            and first.dim() > 0
            and first.shape[0] == 1
            and all(isinstance(value, torch.Tensor) and value.shape == first.shape for value in values)
        ):
            merged[key] = torch.cat(values, dim=0)
        else:
            merged[key] = first
    return merged


_REPLAY_CONTEXT_KEYS = (
    "ego_current_state",
    "neighbor_agents_past",
    "neighbor_agents_future",
)
_REPLAY_DECODER_CONTEXT_SCHEMA = "canonical_loader_aligned_v1"
_REPLAY_EXPERT_SCHEMA = "logged_ego_future_hard_gate_v1"
_REPLAY_EXPERT_TRAJECTORY_KEY = "_replay_expert_trajectory"
_REPLAY_EXPERT_REWARD_KEY = "_replay_expert_reward"
_REPLAY_EXPERT_SAFE_KEY = "_replay_expert_safe"
_REPLAY_EXPERT_KEYS = (
    _REPLAY_EXPERT_TRAJECTORY_KEY,
    _REPLAY_EXPERT_REWARD_KEY,
    _REPLAY_EXPERT_SAFE_KEY,
)


def _shared_decoder_context_contract(
    root: Path,
    *,
    rank: int,
    current_args: Path,
    order_epoch: int,
) -> dict[str, Any]:
    """Validate one immutable rank-local decoder-context generation.

    Decoder context is policy-independent but is *not* representation-
    independent: its neighbor time alignment, X2 geometry correction, model
    cardinalities, and scene order must match the consuming run.  This
    contract deliberately fails closed before any new cache symlink is made.
    """

    root = Path(root).expanduser().resolve()
    source_run = root.parent
    provenance_path = source_run / "provenance.json"
    if not provenance_path.is_file():
        raise RuntimeError(
            "shared replay context has no source provenance: "
            f"{provenance_path}"
        )
    source_provenance = json.loads(provenance_path.read_text())
    source_args = Path(source_provenance["staged_args"]).expanduser().resolve()
    if not source_args.is_file():
        raise FileNotFoundError(source_args)
    current_args = Path(current_args).expanduser().resolve()
    source_args_sha = _file_sha256(source_args)
    current_args_sha = _file_sha256(current_args)
    if source_args_sha != current_args_sha:
        raise RuntimeError(
            "shared replay context was produced with different model args/"
            f"normalization: source={source_args_sha}, current={current_args_sha}"
        )
    source_offset = int(
        source_provenance.get("neighbor_future_alignment_offset", -1)
    )
    current_offset = int(get_neighbor_future_offset())
    if source_offset != current_offset:
        raise RuntimeError(
            "shared replay context neighbor-future alignment differs: "
            f"source={source_offset}, current={current_offset}"
        )
    source_width = float(
        source_provenance.get("x2_legacy_ego_width_m", float("nan"))
    )
    source_xx1_width = float(
        source_provenance.get("xx1_legacy_ego_width_m", float("nan"))
    )
    if not math.isclose(
        source_width, X2_LEGACY_EGO_WIDTH_M, rel_tol=0.0, abs_tol=1e-9
    ) or not math.isclose(
        source_xx1_width, XX1_LEGACY_EGO_WIDTH_M, rel_tol=0.0, abs_tol=1e-9
    ):
        raise RuntimeError(
            "shared replay context does not carry the legacy body-width "
            f"contract: x2={source_width} (want {X2_LEGACY_EGO_WIDTH_M}), "
            f"xx1={source_xx1_width} (want {XX1_LEGACY_EGO_WIDTH_M})"
        )

    rank_dir = root / f"rank_{int(rank):04d}"
    manifest_path = rank_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if not bool(manifest.get("decoder_context_cached", False)):
        raise RuntimeError(
            f"shared rank {rank} manifest has no cached decoder context"
        )
    if manifest.get("decoder_context_schema") != _REPLAY_DECODER_CONTEXT_SCHEMA:
        raise RuntimeError(
            "shared replay context has an unsafe/unknown alignment schema: "
            f"{manifest.get('decoder_context_schema')!r}"
        )
    scene_count = int(manifest.get("scene_count", -1))
    if scene_count <= 0:
        raise RuntimeError(
            f"shared replay context rank {rank} has invalid scene count "
            f"{scene_count}"
        )
    paths_name = manifest.get("paths")
    if not paths_name:
        raise RuntimeError(f"shared replay context rank {rank} has no path index")
    paths_path = (rank_dir / str(paths_name)).resolve()
    if not paths_path.is_file():
        raise FileNotFoundError(paths_path)

    arrays = manifest.get("arrays", {})
    manifest_shapes = manifest.get("decoder_context_shapes", {})
    context_paths: dict[str, str] = {}
    context_shapes: dict[str, list[int]] = {}
    for key in _REPLAY_CONTEXT_KEYS:
        array_name = f"context_{key}"
        filename = arrays.get(array_name)
        if not filename:
            raise RuntimeError(
                f"shared replay context rank {rank} is missing {array_name}"
            )
        array_path = (rank_dir / str(filename)).resolve()
        if not array_path.is_file():
            raise FileNotFoundError(array_path)
        array = np.load(array_path, mmap_mode="r")
        if array.dtype != np.float32:
            raise RuntimeError(
                f"shared replay context {key} must be float32, got {array.dtype}"
            )
        if int(array.shape[0]) != scene_count:
            raise RuntimeError(
                f"shared replay context {key} count differs from manifest: "
                f"{array.shape[0]} != {scene_count}"
            )
        shape = [int(value) for value in array.shape[1:]]
        if [int(value) for value in manifest_shapes.get(key, [])] != shape:
            raise RuntimeError(
                f"shared replay context {key} shape differs from manifest: "
                f"{shape} != {manifest_shapes.get(key)}"
            )
        context_paths[key] = str(array_path)
        context_shapes[key] = shape

    return {
        "enabled": True,
        "root": str(root),
        "rank": int(rank),
        "source_provenance": str(provenance_path),
        "source_args": str(source_args),
        "source_args_sha256": source_args_sha,
        "current_args_sha256": current_args_sha,
        "neighbor_future_alignment_offset": current_offset,
        "x2_legacy_ego_width_m": source_width,
        "decoder_context_schema": _REPLAY_DECODER_CONTEXT_SCHEMA,
        "scene_count": scene_count,
        "paths_path": str(paths_path),
        "context_paths": context_paths,
        "context_shapes": context_shapes,
        "order_epoch": int(order_epoch),
    }


def _shared_expert_anchor_contract(
    root: Path,
    *,
    rank: int,
    current_args: Path,
    current_train_npz_list: Path,
    current_skip_filtered_scenes: bool,
    current_reward_config: RewardConfig,
    order_epoch: int,
) -> dict[str, Any]:
    """Validate one immutable policy-independent expert-anchor sidecar.

    Logged ego future, its hard-safe bit, and its configured reward do not
    depend on the policy being refreshed.  They *do* depend on loader
    alignment, X2 geometry, trajectory horizon, scene order and the complete
    reward contract.  Bind all of those before a later mining cycle is allowed
    to reuse the sidecar; any mismatch fails before rollout starts.
    """

    root = Path(root).expanduser().resolve()
    source_run = root.parent
    provenance_path = source_run / "provenance.json"
    effective_config_path = source_run / "effective_config.json"
    if not provenance_path.is_file():
        raise RuntimeError(
            "shared expert anchor has no source provenance: "
            f"{provenance_path}"
        )
    if not effective_config_path.is_file():
        raise RuntimeError(
            "shared expert anchor has no source effective config: "
            f"{effective_config_path}"
        )
    source_provenance = json.loads(provenance_path.read_text())
    source_effective = json.loads(effective_config_path.read_text())

    source_args = Path(source_provenance["staged_args"]).expanduser().resolve()
    current_args = Path(current_args).expanduser().resolve()
    if not source_args.is_file():
        raise FileNotFoundError(source_args)
    source_args_sha = _file_sha256(source_args)
    current_args_sha = _file_sha256(current_args)
    if source_args_sha != current_args_sha:
        raise RuntimeError(
            "shared expert anchor was produced with different model args/"
            f"normalization: source={source_args_sha}, current={current_args_sha}"
        )

    source_train_list = Path(
        source_provenance.get("train_npz_list", "")
    ).expanduser().resolve()
    current_train_list = Path(current_train_npz_list).expanduser().resolve()
    if source_train_list != current_train_list:
        raise RuntimeError(
            "shared expert anchor was produced from a different train list: "
            f"source={source_train_list}, current={current_train_list}"
        )
    source_skip = bool(source_provenance.get("skip_filtered_scenes", True))
    if source_skip != bool(current_skip_filtered_scenes):
        raise RuntimeError(
            "shared expert anchor skip-filter contract differs: "
            f"source={source_skip}, current={bool(current_skip_filtered_scenes)}"
        )
    source_offset = int(
        source_provenance.get("neighbor_future_alignment_offset", -1)
    )
    current_offset = int(get_neighbor_future_offset())
    if source_offset != current_offset:
        raise RuntimeError(
            "shared expert anchor neighbor-future alignment differs: "
            f"source={source_offset}, current={current_offset}"
        )
    source_width = float(
        source_provenance.get("x2_legacy_ego_width_m", float("nan"))
    )
    source_xx1_width = float(
        source_provenance.get("xx1_legacy_ego_width_m", float("nan"))
    )
    if not math.isclose(
        source_width, X2_LEGACY_EGO_WIDTH_M, rel_tol=0.0, abs_tol=1e-9
    ) or not math.isclose(
        source_xx1_width, XX1_LEGACY_EGO_WIDTH_M, rel_tol=0.0, abs_tol=1e-9
    ):
        raise RuntimeError(
            "shared expert anchor does not carry the legacy body-width "
            f"contract: x2={source_width} (want {X2_LEGACY_EGO_WIDTH_M}), "
            f"xx1={source_xx1_width} (want {XX1_LEGACY_EGO_WIDTH_M})"
        )

    current_reward = asdict(current_reward_config)
    source_reward_raw = source_effective.get("reward", {})
    source_reward = {
        key: source_reward_raw.get(key, object()) for key in current_reward
    }
    # JSON-normalise tuples and scalar subclasses before exact comparison.
    try:
        source_reward_json = json.dumps(source_reward, sort_keys=True)
    except TypeError as error:
        raise RuntimeError(
            "shared expert anchor source reward config is incomplete"
        ) from error
    current_reward_json = json.dumps(current_reward, sort_keys=True)
    if source_reward_json != current_reward_json:
        differing = sorted(
            key
            for key, value in current_reward.items()
            if source_reward_raw.get(key, object()) != value
        )
        raise RuntimeError(
            "shared expert anchor reward contract differs: "
            f"fields={differing}"
        )

    rank_dir = root / f"rank_{int(rank):04d}"
    replay_manifest_path = rank_dir / "manifest.json"
    expert_manifest_path = rank_dir / "expert_anchor_manifest.json"
    if not replay_manifest_path.is_file():
        raise FileNotFoundError(replay_manifest_path)
    if not expert_manifest_path.is_file():
        raise FileNotFoundError(expert_manifest_path)
    replay_manifest = json.loads(replay_manifest_path.read_text())
    expert_manifest = json.loads(expert_manifest_path.read_text())
    if expert_manifest.get("schema") != _REPLAY_EXPERT_SCHEMA:
        raise RuntimeError(
            "shared expert anchor has an unsafe/unknown schema: "
            f"{expert_manifest.get('schema')!r}"
        )
    scene_count = int(replay_manifest.get("scene_count", -1))
    if scene_count <= 0 or int(expert_manifest.get("scene_count", -2)) != scene_count:
        raise RuntimeError(
            "shared expert anchor scene count differs from replay manifest: "
            f"expert={expert_manifest.get('scene_count')}, replay={scene_count}"
        )
    current_args_data = json.loads(current_args.read_text())
    future_len = int(current_args_data["future_len"])
    if int(expert_manifest.get("future_len", -1)) != future_len:
        raise RuntimeError(
            "shared expert anchor future length differs: "
            f"source={expert_manifest.get('future_len')}, current={future_len}"
        )
    replay_paths_name = replay_manifest.get("paths")
    expert_paths_name = expert_manifest.get("paths")
    if not replay_paths_name or expert_paths_name != replay_paths_name:
        raise RuntimeError(
            "shared expert anchor path index differs from replay manifest"
        )
    paths_path = (rank_dir / str(replay_paths_name)).resolve()
    if not paths_path.is_file():
        raise FileNotFoundError(paths_path)

    expected_names = {
        "expert_trajectories",
        "expert_rewards",
        "expert_safe",
    }
    names = expert_manifest.get("arrays", {})
    if set(names) != expected_names:
        raise RuntimeError(
            "shared expert anchor sidecar is incomplete: "
            f"{sorted(names)}"
        )
    expected_shapes = {
        "expert_trajectories": (scene_count, future_len, 4),
        "expert_rewards": (scene_count,),
        "expert_safe": (scene_count,),
    }
    array_paths: dict[str, str] = {}
    array_shapes: dict[str, list[int]] = {}
    for name in sorted(expected_names):
        path = (rank_dir / str(names[name])).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        array = np.load(path, mmap_mode="r")
        expected_shape = expected_shapes[name]
        if array.dtype != np.float32 or tuple(array.shape) != expected_shape:
            raise RuntimeError(
                f"shared expert anchor {name} must be float32 {expected_shape}, "
                f"got dtype={array.dtype}, shape={tuple(array.shape)}"
            )
        probe = np.asarray(sorted({0, scene_count - 1}), dtype=np.int64)
        if not np.isfinite(np.asarray(array[probe])).all():
            raise RuntimeError(
                f"shared expert anchor {name} contains non-finite probe values"
            )
        if name == "expert_safe":
            safe = np.asarray(array)
            if not bool(np.logical_or(safe == 0.0, safe == 1.0).all()):
                raise RuntimeError(
                    "shared expert anchor safe flags must be exactly 0/1"
                )
        array_paths[name] = str(path)
        array_shapes[name] = [int(value) for value in array.shape]

    return {
        "enabled": True,
        "root": str(root),
        "rank": int(rank),
        "source_provenance": str(provenance_path),
        "source_effective_config": str(effective_config_path),
        "source_args": str(source_args),
        "source_args_sha256": source_args_sha,
        "current_args_sha256": current_args_sha,
        "train_npz_list": str(current_train_list),
        "skip_filtered_scenes": bool(current_skip_filtered_scenes),
        "neighbor_future_alignment_offset": current_offset,
        "x2_legacy_ego_width_m": source_width,
        "reward_config_sha256": hashlib.sha256(
            current_reward_json.encode("utf-8")
        ).hexdigest(),
        "expert_anchor_schema": _REPLAY_EXPERT_SCHEMA,
        "scene_count": scene_count,
        "future_len": future_len,
        "paths_path": str(paths_path),
        "array_paths": array_paths,
        "array_shapes": array_shapes,
        "order_epoch": int(order_epoch),
    }


def _compact_replay_decoder_context(
    data: dict[str, torch.Tensor],
    predicted_neighbor_num: int,
) -> dict[str, torch.Tensor]:
    """Keep only byte-relevant original-DP decoder context for frozen replay.

    The behavior encoder is already stored separately.  With expert/neighbor
    auxiliary losses disabled, the diffusion regression still needs the
    current ego state, current states of the predicted neighbors, and their
    future tokens used to construct the joint noisy trajectory.  The source
    NPZ carries 320 agents and 31 history frames, while this checkpoint uses
    only the first ``predicted_neighbor_num`` agents and the final history
    frame.  ``load_npz_data`` has already applied the configured legacy +1
    future alignment at the canonical loader boundary, so persist that tensor
    directly.  Applying alignment again here would silently train on t=+2.
    """

    missing = [key for key in _REPLAY_CONTEXT_KEYS if key not in data]
    if missing:
        raise KeyError(f"replay decoder context is missing {missing}")
    for key in _REPLAY_CONTEXT_KEYS:
        if not isinstance(data[key], torch.Tensor):
            raise TypeError(f"replay decoder context {key} is not a tensor")
    neighbor_count = max(0, int(predicted_neighbor_num))
    return {
        "ego_current_state": data["ego_current_state"].detach(),
        "neighbor_agents_past": data["neighbor_agents_past"][
            :, :neighbor_count, -1:, :
        ].detach(),
        "neighbor_agents_future": data["neighbor_agents_future"][
            :, :neighbor_count
        ].detach(),
    }


def _load_scene_batch(
    scene_paths: list[str],
    device: torch.device,
    workers: int = 1,
    replay_context_only: bool = False,
    defer_full_scene_heading_transforms: bool = False,
) -> dict[str, torch.Tensor]:
    """Load a scene batch without serial GPU copies for every NPZ file.

    The original scene-wise path moves each tensor to CUDA while parsing each
    NPZ and then concatenates 192 one-scene CUDA tensors.  Full-corpus AWR is
    dominated by that Python/NPZ path between denoiser calls.  Parse a small
    batch concurrently on CPU, concatenate once, and perform one device copy
    per field.  The tensors and their ordering are unchanged; this is only an
    I/O/layout optimization.
    """

    if not scene_paths:
        raise ValueError("cannot load an empty scene batch")

    def load_one(
        path: str,
        target_device: torch.device,
        *,
        defer_heading_transforms: bool = False,
    ):
        if not replay_context_only:
            if defer_heading_transforms:
                return load_npz_data(
                    path,
                    target_device,
                    defer_heading_transforms=True,
                )
            return load_scene(path, target_device)

        # Frozen replay records already contain the exact behavior-policy
        # encoder output. The diffusion regression only consumes these three
        # observation tensors. Reading maps, lanes and 31-frame histories
        # from every compressed NPZ is otherwise pure I/O/allocation overhead.
        with np.load(str(path)) as loaded:
            required = {
                "ego_current_state",
                "neighbor_agents_past",
                "neighbor_agents_future",
            }
            missing = sorted(required.difference(loaded.files))
            if missing:
                raise KeyError(f"replay scene {path} is missing {missing}")
            ego = np.array(loaded["ego_current_state"], copy=True)
            # Preserve the canonical loss' ``[..., -1, :4]`` access while
            # avoiding the 30 unused historical frames.
            neighbor_current = np.array(
                loaded["neighbor_agents_past"][:, -1:, :], copy=True
            )
            neighbor_future = align_neighbor_future_numpy(
                np.array(loaded["neighbor_agents_future"], copy=True)
            )
        return {
            "ego_current_state": torch.from_numpy(ego)
            .unsqueeze(0)
            .to(target_device),
            "neighbor_agents_past": torch.from_numpy(neighbor_current)
            .unsqueeze(0)
            .to(target_device),
            "neighbor_agents_future": torch.from_numpy(neighbor_future)
            .unsqueeze(0)
            .to(target_device),
        }

    if len(scene_paths) == 1:
        return _merge_scene_data([load_one(scene_paths[0], device)])

    # Parse every packed multi-scene batch on CPU even with one reader, merge
    # it there, then copy one tensor per field to CUDA.  Device-local heading
    # trigonometry is deliberately deferred until after that transfer: doing
    # it on CPU differs from the historical CUDA loader by up to one float32
    # ULP.  This preserves bitwise model inputs while removing hundreds of
    # tiny per-scene transfers.
    cpu = torch.device("cpu")
    worker_count = max(1, min(int(workers), len(scene_paths)))
    if worker_count == 1:
        cpu_datas = [
            load_one(path, cpu, defer_heading_transforms=True)
            for path in scene_paths
        ]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            cpu_datas = list(
                executor.map(
                    lambda path: load_one(
                        path, cpu, defer_heading_transforms=True
                    ),
                    scene_paths,
                )
            )
    merged = _merge_scene_data(cpu_datas)
    if device.type != "cpu":
        merged = {
            key: value.to(device=device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in merged.items()
        }
    if not replay_context_only and not defer_full_scene_heading_transforms:
        apply_npz_heading_transforms(merged)
    return merged


def _effective_eval_scene_load_workers(train_config: dict[str, Any]) -> int:
    """Resolve full-scene evaluation I/O independently from replay I/O.

    Frozen replay opens only three arrays and benefits from a few concurrent
    readers.  Evaluation parses every model/reward field from each compressed
    NPZ; on the production validation store that path is fastest with one
    reader per rank.  Keeping separate knobs avoids making one phase slower to
    accelerate the other and has no effect on scene ordering or tensor values.
    """

    return max(
        1,
        int(
            train_config.get(
                "eval_scene_load_workers",
                train_config.get("scene_load_workers", 1),
            )
        ),
    )


class _DiskReplayWriter:
    """Write an HDP-compatible replay epoch as rank-local NVMe memmaps.

    The released HDP agent keeps ``(token, rollout, gt, reward)`` tuples in
    RAM, while its encoder features are supplied by an offline cache.  A full
    T4 corpus is too large for the former representation, so this class keeps
    the same tuple contents in contiguous rank-local ``.npy`` arrays.  The
    rollout targets, group weights, rewards, and *frozen behavior encoder*
    are written once during the rollout epoch and are sampled with
    replacement during replay epochs.  No policy-dependent quantity is
    regenerated while replaying.

    Memmaps are intentionally uncompressed: the rollout pass is compute
    bound, and sequential writes plus direct random batch reads are much
    faster than millions of compressed NPZ files.  The arrays are float32 at
    rest, matching the loss/reference cache numerics; BF16 is still used for
    model forwards.
    """

    def __init__(
        self,
        root: Path,
        rank: int,
        expected_count: int,
        group_size: int,
        future_len: int,
        cache_decoder_context: bool = False,
        cache_expert_anchor: bool = False,
        shared_scene_encoding_path: Path | None = None,
        shared_decoder_context_paths: dict[str, Path] | None = None,
        expected_scene_paths: list[str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.rank = int(rank)
        self.rank_dir = self.root / f"rank_{self.rank:04d}"
        self.rank_dir.mkdir(parents=True, exist_ok=True)
        # A refresh overwrites the rank-local arrays. Revoke the previous
        # generation before touching any memmap so an interrupted epoch 11+
        # cannot expose partially-new arrays through a stale manifest.
        (self.rank_dir / "manifest.json").unlink(missing_ok=True)
        for stale_tmp in self.rank_dir.glob(".manifest.json.tmp.*"):
            stale_tmp.unlink(missing_ok=True)
        self.expected_count = int(expected_count)
        self.group_size = int(group_size)
        self.future_len = int(future_len)
        self.cache_decoder_context = bool(cache_decoder_context)
        self.cache_expert_anchor = bool(cache_expert_anchor)
        self.shared_scene_encoding_path = (
            Path(shared_scene_encoding_path).resolve()
            if shared_scene_encoding_path is not None
            else None
        )
        self.shared_scene_encoding_source_count: int | None = None
        self.shared_decoder_context_paths = {
            str(key): Path(path).resolve()
            for key, path in (shared_decoder_context_paths or {}).items()
        }
        self.shared_decoder_context_source_counts: dict[str, int] = {}
        if self.shared_decoder_context_paths and not self.cache_decoder_context:
            raise ValueError(
                "shared decoder context requires cache_decoder_context=true"
            )
        if self.shared_decoder_context_paths and set(
            self.shared_decoder_context_paths
        ) != set(_REPLAY_CONTEXT_KEYS):
            raise ValueError(
                "shared decoder context must contain exactly "
                f"{list(_REPLAY_CONTEXT_KEYS)}"
            )
        self.expected_scene_paths = (
            [str(path) for path in expected_scene_paths]
            if expected_scene_paths is not None
            else None
        )
        self.position = 0
        self._encoding_shape: tuple[int, ...] | None = None
        self._context_shapes: dict[str, tuple[int, ...]] = {}
        self._arrays: dict[str, np.memmap] = {}
        self._expert_arrays: dict[str, np.memmap] = {}
        self._paths_path = self.rank_dir / "scene_paths.jsonl"
        self._expected_paths_path = self.rank_dir / "expected_scene_paths.jsonl"
        # A refresh reuses the rank directory.  Truncate the path index before
        # appending this epoch; otherwise a second refresh would look like a
        # larger buffer than the newly-created memmaps.
        self._paths_path.write_text("")
        # The sidecar manifest is the commit marker.  Removing it before
        # creating a new generation prevents a stale complete sidecar from
        # being paired with partially overwritten arrays after interruption.
        (self.rank_dir / "expert_anchor_manifest.json").unlink(missing_ok=True)

        if self.expected_count <= 0:
            raise ValueError("a replay writer needs at least one scene")
        if self.group_size <= 0 or self.future_len <= 0:
            raise ValueError(
                f"invalid replay shape group_size={self.group_size}, "
                f"future_len={self.future_len}"
            )
        if (
            self.expected_scene_paths is not None
            and len(self.expected_scene_paths) != self.expected_count
        ):
            raise ValueError(
                "expected replay path count differs from writer capacity: "
                f"{len(self.expected_scene_paths)} != {self.expected_count}"
            )
        if self.expected_scene_paths is not None:
            with self._expected_paths_path.open("w") as handle:
                for path in self.expected_scene_paths:
                    handle.write(json.dumps(path) + "\n")
        else:
            self._expected_paths_path.unlink(missing_ok=True)
        _write_json_atomic(
            self.rank_dir / "writer_contract.json",
            {
                "version": 1,
                "expected_count": self.expected_count,
                "expected_paths": (
                    self._expected_paths_path.name
                    if self.expected_scene_paths is not None
                    else None
                ),
                "decoder_context_cached": self.cache_decoder_context,
                "decoder_context_schema": (
                    _REPLAY_DECODER_CONTEXT_SCHEMA
                    if self.cache_decoder_context
                    else None
                ),
                "neighbor_future_alignment_offset": get_neighbor_future_offset(),
                "shared_scene_encoding_path": (
                    str(self.shared_scene_encoding_path)
                    if self.shared_scene_encoding_path is not None
                    else None
                ),
                "shared_decoder_context_paths": {
                    key: str(path)
                    for key, path in self.shared_decoder_context_paths.items()
                },
                "expert_anchor_cached": self.cache_expert_anchor,
                "expert_anchor_schema": (
                    _REPLAY_EXPERT_SCHEMA if self.cache_expert_anchor else None
                ),
            },
        )
        self._create_array(
            "trajectories",
            (self.expected_count, self.group_size, self.future_len, 4),
        )
        self._create_array("noise_scales", (self.expected_count, self.group_size))
        self._create_array("weights", (self.expected_count, self.group_size))
        self._create_array("rewards", (self.expected_count, self.group_size))
        if self.cache_expert_anchor:
            self._create_expert_array(
                "expert_trajectories",
                (self.expected_count, self.future_len, 4),
            )
            self._create_expert_array("expert_rewards", (self.expected_count,))
            self._create_expert_array("expert_safe", (self.expected_count,))
        if self.shared_scene_encoding_path is not None:
            shared_encoding = np.load(
                self.shared_scene_encoding_path, mmap_mode="r"
            )
            self.shared_scene_encoding_source_count = int(
                shared_encoding.shape[0]
            )
            if self.shared_scene_encoding_source_count < self.expected_count:
                raise RuntimeError(
                    "shared scene-encoding is shorter than replay order: "
                    f"{self.shared_scene_encoding_source_count} < "
                    f"{self.expected_count}"
                )
            if shared_encoding.dtype != np.float32:
                raise RuntimeError(
                    "shared original-DP scene encoding must be float32, got "
                    f"{shared_encoding.dtype}"
                )
            self._encoding_shape = tuple(
                int(x) for x in shared_encoding.shape[1:]
            )
            link = self.rank_dir / "scene_encoding.npy"
            link.unlink(missing_ok=True)
            link.symlink_to(self.shared_scene_encoding_path)
            self._arrays["scene_encoding"] = shared_encoding
        for key, source_path in self.shared_decoder_context_paths.items():
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            shared_context = np.load(source_path, mmap_mode="r")
            source_count = int(shared_context.shape[0])
            if source_count < self.expected_count:
                raise RuntimeError(
                    f"shared decoder context {key} is shorter than replay order: "
                    f"{source_count} < {self.expected_count}"
                )
            self.shared_decoder_context_source_counts[key] = source_count
            if shared_context.dtype != np.float32:
                raise RuntimeError(
                    f"shared decoder context {key} must be float32, got "
                    f"{shared_context.dtype}"
                )
            shape = tuple(int(x) for x in shared_context.shape[1:])
            self._context_shapes[key] = shape
            array_name = f"context_{key}"
            link = self.rank_dir / f"{array_name}.npy"
            link.unlink(missing_ok=True)
            link.symlink_to(source_path)
            self._arrays[array_name] = shared_context

    def _create_array(self, name: str, shape: tuple[int, ...]) -> None:
        path = self.rank_dir / f"{name}.npy"
        self._arrays[name] = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.float32,
            shape=shape,
        )

    def _create_expert_array(self, name: str, shape: tuple[int, ...]) -> None:
        path = self.rank_dir / f"{name}.npy"
        self._expert_arrays[name] = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.float32,
            shape=shape,
        )

    @staticmethod
    def _to_numpy(value: torch.Tensor, dtype: torch.dtype = torch.float32) -> np.ndarray:
        return (
            value.detach()
            .to(device="cpu", dtype=dtype)
            .contiguous()
            .numpy()
        )

    def append(
        self,
        scene_paths: list[str],
        rollout: AWRRollout,
        group_size: int,
    ) -> None:
        paths = [str(path) for path in scene_paths]
        batch_size = len(paths)
        if batch_size <= 0:
            return
        if int(group_size) != self.group_size:
            raise ValueError(
                f"replay group-size changed: expected {self.group_size}, got {group_size}"
            )
        expected_candidates = batch_size * self.group_size
        if rollout.trajectories.shape[0] != expected_candidates:
            raise ValueError(
                "combined rollout does not preserve scene-major groups: "
                f"paths={batch_size}, K={self.group_size}, "
                f"trajectories={tuple(rollout.trajectories.shape)}"
            )
        if self.position + batch_size > self.expected_count:
            raise ValueError("replay writer received more scenes than its manifest")
        if self.expected_scene_paths is not None:
            expected_paths = self.expected_scene_paths[
                self.position : self.position + batch_size
            ]
            if paths != expected_paths:
                mismatch = next(
                    index
                    for index, (actual, expected) in enumerate(
                        zip(paths, expected_paths)
                    )
                    if actual != expected
                )
                raise RuntimeError(
                    "replay append is not a canonical path prefix: "
                    f"rank={self.rank}, position={self.position + mismatch}, "
                    f"actual={paths[mismatch]!r}, expected={expected_paths[mismatch]!r}"
                )

        trajectories = self._to_numpy(rollout.trajectories).reshape(
            batch_size, self.group_size, self.future_len, 4
        )
        noise_scales = self._to_numpy(rollout.noise_scales).reshape(
            batch_size, self.group_size
        )
        weights = self._to_numpy(rollout.weights).reshape(batch_size, self.group_size)
        if rollout.rewards and len(rollout.rewards) == expected_candidates:
            rewards = np.asarray(
                [float(reward.total) for reward in rollout.rewards], dtype=np.float32
            ).reshape(batch_size, self.group_size)
        else:
            rewards = np.full(
                (batch_size, self.group_size), np.nan, dtype=np.float32
            )

        expert_values: dict[str, np.ndarray] = {}
        if self.cache_expert_anchor:
            missing_expert = [
                key for key in _REPLAY_EXPERT_KEYS if key not in rollout.reward_data
            ]
            if missing_expert:
                raise ValueError(
                    "replay expert sidecar disappeared before disk write: "
                    f"{missing_expert}"
                )
            expert_trajectory = rollout.reward_data[_REPLAY_EXPERT_TRAJECTORY_KEY]
            expert_reward = rollout.reward_data[_REPLAY_EXPERT_REWARD_KEY]
            expert_safe = rollout.reward_data[_REPLAY_EXPERT_SAFE_KEY]
            if not all(
                isinstance(value, torch.Tensor)
                for value in (expert_trajectory, expert_reward, expert_safe)
            ):
                raise TypeError("replay expert sidecar values must be tensors")
            if tuple(expert_trajectory.shape) != (
                batch_size,
                self.future_len,
                4,
            ):
                raise ValueError(
                    "replay expert trajectory shape mismatch: "
                    f"expected={(batch_size, self.future_len, 4)}, "
                    f"got={tuple(expert_trajectory.shape)}"
                )
            if tuple(expert_reward.shape) != (batch_size,):
                raise ValueError(
                    "replay expert reward shape mismatch: "
                    f"expected={(batch_size,)}, got={tuple(expert_reward.shape)}"
                )
            if tuple(expert_safe.shape) != (batch_size,):
                raise ValueError(
                    "replay expert safe shape mismatch: "
                    f"expected={(batch_size,)}, got={tuple(expert_safe.shape)}"
                )
            expert_values = {
                "expert_trajectories": self._to_numpy(expert_trajectory),
                "expert_rewards": self._to_numpy(expert_reward),
                "expert_safe": self._to_numpy(expert_safe),
            }

        encoding = rollout.scene_encoding
        if encoding is not None:
            if self.shared_scene_encoding_path is None:
                encoding_np = self._to_numpy(encoding)
            if int(encoding.shape[0]) != batch_size:
                raise ValueError(
                    "replay behavior encoding is not scene-batched: "
                    f"expected B={batch_size}, got {tuple(encoding.shape)}"
                )
            encoding_shape = tuple(int(x) for x in encoding.shape[1:])
            if self._encoding_shape is None:
                self._encoding_shape = encoding_shape
                self._create_array(
                    "scene_encoding", (self.expected_count, *self._encoding_shape)
                )
            elif encoding_shape != self._encoding_shape:
                raise ValueError(
                    "replay behavior encoding shape changed: "
                    f"expected {self._encoding_shape}, got {encoding_shape}"
                )
        elif self._encoding_shape is not None:
            raise ValueError("replay behavior encoding disappeared mid-epoch")

        context_values: dict[str, np.ndarray] = {}
        if self.cache_decoder_context:
            missing_context = [
                key for key in _REPLAY_CONTEXT_KEYS if key not in rollout.reward_data
            ]
            if missing_context:
                raise ValueError(
                    "replay decoder context disappeared before disk write: "
                    f"{missing_context}"
                )
            for key in _REPLAY_CONTEXT_KEYS:
                value = rollout.reward_data[key]
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"replay decoder context {key} is not a tensor")
                if value.shape[0] != batch_size:
                    raise ValueError(
                        f"replay decoder context {key} expected B={batch_size}, "
                        f"got {tuple(value.shape)}"
                    )
                shape = tuple(int(x) for x in value.shape[1:])
                array_name = f"context_{key}"
                if key not in self._context_shapes:
                    self._context_shapes[key] = shape
                    self._create_array(
                        array_name,
                        (self.expected_count, *shape),
                    )
                elif shape != self._context_shapes[key]:
                    raise ValueError(
                        f"replay decoder context {key} shape changed: "
                        f"expected {self._context_shapes[key]}, got {shape}"
                    )
                if key not in self.shared_decoder_context_paths:
                    context_values[array_name] = self._to_numpy(value)

        start = self.position
        end = start + batch_size
        self._arrays["trajectories"][start:end] = trajectories
        self._arrays["noise_scales"][start:end] = noise_scales
        self._arrays["weights"][start:end] = weights
        self._arrays["rewards"][start:end] = rewards
        if encoding is not None and self.shared_scene_encoding_path is None:
            self._arrays["scene_encoding"][start:end] = encoding_np
        for array_name, value_np in context_values.items():
            self._arrays[array_name][start:end] = value_np
        for array_name, value_np in expert_values.items():
            self._expert_arrays[array_name][start:end] = value_np
        self.position = end
        with self._paths_path.open("a") as handle:
            for path in paths:
                handle.write(json.dumps(path) + "\n")

    def close(self) -> Path:
        if self.position != self.expected_count:
            raise RuntimeError(
                "faithful replay rollout was incomplete: "
                f"rank={self.rank}, wrote={self.position}, "
                f"expected={self.expected_count}"
            )
        for array in self._arrays.values():
            array.flush()
        for array in self._expert_arrays.values():
            array.flush()
        manifest = {
            "version": 3 if self.cache_decoder_context else 1,
            "rank": self.rank,
            "scene_count": self.position,
            "group_size": self.group_size,
            "future_len": self.future_len,
            "encoding_shape": list(self._encoding_shape)
            if self._encoding_shape is not None
            else None,
            "shared_scene_encoding_path": (
                str(self.shared_scene_encoding_path)
                if self.shared_scene_encoding_path is not None
                else None
            ),
            "shared_scene_encoding_source_count": (
                getattr(self, "shared_scene_encoding_source_count", None)
            ),
            "shared_scene_encoding_prefix_count": (
                self.position
                if self.shared_scene_encoding_path is not None
                else None
            ),
            "shared_decoder_context_paths": {
                key: str(path)
                for key, path in self.shared_decoder_context_paths.items()
            },
            "shared_decoder_context_source_counts": dict(
                self.shared_decoder_context_source_counts
            ),
            "shared_decoder_context_prefix_count": (
                self.position if self.shared_decoder_context_paths else None
            ),
            "decoder_context_cached": bool(self.cache_decoder_context),
            "decoder_context_schema": (
                _REPLAY_DECODER_CONTEXT_SCHEMA
                if self.cache_decoder_context
                else None
            ),
            "neighbor_future_alignment_offset": get_neighbor_future_offset(),
            "decoder_context_shapes": {
                key: list(shape) for key, shape in self._context_shapes.items()
            },
            "arrays": {
                name: str(path.name)
                for name, path in {
                    **{key: self.rank_dir / f"{key}.npy" for key in self._arrays}
                }.items()
            },
            "paths": self._paths_path.name,
            "expected_paths": (
                self._expected_paths_path.name
                if self.expected_scene_paths is not None
                else None
            ),
        }
        manifest_path = self.rank_dir / "manifest.json"
        _write_json_atomic(manifest_path, manifest)
        if self.cache_expert_anchor:
            _write_json_atomic(
                self.rank_dir / "expert_anchor_manifest.json",
                {
                    "version": 1,
                    "schema": _REPLAY_EXPERT_SCHEMA,
                    "scene_count": self.position,
                    "future_len": self.future_len,
                    "arrays": {
                        name: f"{name}.npy" for name in self._expert_arrays
                    },
                    "paths": self._paths_path.name,
                },
            )
        return manifest_path


class _DiskReplayReader:
    """Read the frozen HDP replay epoch with group sampling semantics."""

    def __init__(
        self,
        root: Path,
        rank: int,
        *,
        positive_advantage_only: bool = False,
        deterministic_first: bool = False,
        compressed_context_root: Path | None = None,
        compressed_context_workers: int = 1,
        context_active_weight_only: bool = False,
    ) -> None:
        self.rank = int(rank)
        self.positive_advantage_only = bool(positive_advantage_only)
        self.deterministic_first = bool(deterministic_first)
        self.context_active_weight_only = bool(context_active_weight_only)
        self.rank_dir = Path(root) / f"rank_{self.rank:04d}"
        manifest_path = self.rank_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        self.manifest = json.loads(manifest_path.read_text())
        self.scene_count = int(self.manifest["scene_count"])
        self.group_size = int(self.manifest["group_size"])
        self.future_len = int(self.manifest["future_len"])
        self.trajectories = np.load(
            self.rank_dir / self.manifest["arrays"]["trajectories"], mmap_mode="r"
        )
        self.noise_scales = np.load(
            self.rank_dir / self.manifest["arrays"]["noise_scales"], mmap_mode="r"
        )
        self.weights = np.load(
            self.rank_dir / self.manifest["arrays"]["weights"], mmap_mode="r"
        )
        self.rewards = np.load(
            self.rank_dir / self.manifest["arrays"]["rewards"], mmap_mode="r"
        )
        expected_paths_sha256 = str(
            self.manifest.get("decoder_context_attachment", {}).get(
                "expected_paths_sha256", ""
            )
        )
        self.compressed_context = (
            CompressedReplayContextReader(
                compressed_context_root,
                self.rank,
                expected_scene_count=self.scene_count,
                expected_paths_sha256=expected_paths_sha256,
                workers=compressed_context_workers,
            )
            if compressed_context_root is not None
            else None
        )
        if self.compressed_context is not None:
            compressed_shapes = {
                str(field["name"]): [
                    int(value) for value in field.get("row_shape", [])
                ]
                for field in self.compressed_context.fields
            }
            expected_compressed_shapes = {
                "scene_encoding": [
                    int(value) for value in self.manifest.get("encoding_shape", [])
                ],
                "ego_current_state": [
                    int(value)
                    for value in self.manifest.get(
                        "decoder_context_shapes", {}
                    ).get("ego_current_state", [])
                ],
                "neighbor_agents_past": [
                    int(value)
                    for value in self.manifest.get(
                        "decoder_context_shapes", {}
                    ).get("neighbor_agents_past", [])
                ],
                "neighbor_agents_future": [
                    int(value)
                    for value in self.manifest.get(
                        "decoder_context_shapes", {}
                    ).get("neighbor_agents_future", [])
                ],
                "expert_trajectories": [self.future_len, 4],
                "expert_rewards": [],
                "expert_safe": [],
            }
            if compressed_shapes != expected_compressed_shapes:
                raise RuntimeError(
                    "compressed replay context/model shapes differ: "
                    f"compressed={compressed_shapes}, "
                    f"expected={expected_compressed_shapes}"
                )
        encoding_name = self.manifest["arrays"].get("scene_encoding")
        self.scene_encoding = (
            np.load(self.rank_dir / encoding_name, mmap_mode="r")
            if encoding_name and self.compressed_context is None
            else None
        )
        context_array_names = {
            key: f"context_{key}" for key in _REPLAY_CONTEXT_KEYS
        }
        present_context = {
            key: name
            for key, name in context_array_names.items()
            if name in self.manifest["arrays"]
        }
        if present_context and self.manifest.get(
            "decoder_context_schema"
        ) != _REPLAY_DECODER_CONTEXT_SCHEMA:
            raise RuntimeError(
                "replay decoder context uses an unsafe/unknown alignment schema: "
                f"{self.manifest.get('decoder_context_schema')!r}; re-mine it"
            )
        if present_context and len(present_context) != len(context_array_names):
            raise RuntimeError(
                "replay manifest has an incomplete decoder context: "
                f"present={sorted(present_context)}"
            )
        self.decoder_context = (
            {
                key: np.load(
                    self.rank_dir / self.manifest["arrays"][name], mmap_mode="r"
                )
                for key, name in present_context.items()
            }
            if self.compressed_context is None
            else {}
        )
        expert_manifest_path = self.rank_dir / "expert_anchor_manifest.json"
        self.expert_anchor: dict[str, np.ndarray] = {}
        self.expert_anchor_scalars: dict[str, np.ndarray] = {}
        if expert_manifest_path.is_file():
            expert_manifest = json.loads(expert_manifest_path.read_text())
            if expert_manifest.get("schema") != _REPLAY_EXPERT_SCHEMA:
                raise RuntimeError(
                    "replay expert sidecar uses an unsafe/unknown schema: "
                    f"{expert_manifest.get('schema')!r}"
                )
            if int(expert_manifest.get("scene_count", -1)) != self.scene_count:
                raise RuntimeError(
                    "replay expert sidecar count differs from replay manifest: "
                    f"{expert_manifest.get('scene_count')} != {self.scene_count}"
                )
            if int(expert_manifest.get("future_len", -1)) != self.future_len:
                raise RuntimeError(
                    "replay expert sidecar future length differs from replay: "
                    f"{expert_manifest.get('future_len')} != {self.future_len}"
                )
            array_names = expert_manifest.get("arrays", {})
            expected_expert_arrays = {
                "expert_trajectories",
                "expert_rewards",
                "expert_safe",
            }
            if set(array_names) != expected_expert_arrays:
                raise RuntimeError(
                    "replay expert sidecar arrays are incomplete: "
                    f"{sorted(array_names)}"
                )
            self.expert_anchor = (
                {
                    name: np.load(
                        self.rank_dir / str(array_names[name]), mmap_mode="r"
                    )
                    for name in sorted(expected_expert_arrays)
                }
                if self.compressed_context is None
                else {}
            )
            if self.compressed_context is not None:
                # Keep the two tiny scalar sidecars mmap-backed.  Active-only
                # context decoding may skip the large row for a zero-gradient
                # scene, but diagnostics still report the exact pre-gate
                # expert safety/reward population.
                self.expert_anchor_scalars = {
                    name: np.load(
                        self.rank_dir / str(array_names[name]), mmap_mode="r"
                    )
                    for name in ("expert_rewards", "expert_safe")
                }
            expected_shapes = {
                "expert_trajectories": (
                    self.scene_count,
                    self.future_len,
                    4,
                ),
                "expert_rewards": (self.scene_count,),
                "expert_safe": (self.scene_count,),
            }
            if self.compressed_context is None:
                for name, expected_shape in expected_shapes.items():
                    array = self.expert_anchor[name]
                    if array.dtype != np.float32 or tuple(array.shape) != expected_shape:
                        raise RuntimeError(
                            f"invalid replay expert sidecar {name}: "
                            f"dtype={array.dtype}, shape={tuple(array.shape)}, "
                            f"expected float32 {expected_shape}"
                        )
        with (self.rank_dir / self.manifest["paths"]).open() as handle:
            self.scene_paths = [json.loads(line) for line in handle]
        if len(self.scene_paths) != self.scene_count:
            raise RuntimeError(
                f"replay path count mismatch: {len(self.scene_paths)} != {self.scene_count}"
            )

    def __len__(self) -> int:
        return self.scene_count

    def close(self) -> None:
        """Release old generation mmaps before a refresh truncates them."""

        for array in (
            self.trajectories,
            self.noise_scales,
            self.weights,
            self.rewards,
            self.scene_encoding,
            *self.decoder_context.values(),
            *self.expert_anchor.values(),
            *self.expert_anchor_scalars.values(),
        ):
            mmap = getattr(array, "_mmap", None) if array is not None else None
            if mmap is not None:
                mmap.close()
        if self.compressed_context is not None:
            self.compressed_context.close()

    @property
    def has_cached_scene_encoding(self) -> bool:
        return self.scene_encoding is not None or self.compressed_context is not None

    @property
    def has_cached_decoder_context(self) -> bool:
        return bool(self.decoder_context) or self.compressed_context is not None

    @property
    def has_cached_expert_anchor(self) -> bool:
        return bool(self.expert_anchor) or self.compressed_context is not None

    def batch_from_indices(
        self,
        indices: np.ndarray | list[int],
        *,
        sampled_with_replacement: bool = True,
        priority_best_reward_lt: float | None = None,
    ):
        """Materialize one replay batch from buffer indices."""

        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or indices.size <= 0:
            raise ValueError("replay indices must be a non-empty 1-D array")
        batch_size = int(indices.size)

        # Random AWR sampling is intentionally with replacement, but issuing
        # 192 large memmap reads in random physical order wastes NVMe queue
        # locality. Read the exact same indices in stable sorted order, then
        # restore the original RNG order before constructing tensors. This is
        # byte-exact, including when one index appears more than once.
        read_order = np.argsort(indices, kind="stable")
        sorted_indices = indices[read_order]
        restore_order = np.empty_like(read_order)
        restore_order[read_order] = np.arange(batch_size, dtype=read_order.dtype)

        def materialize(array: np.ndarray, *, copy: bool = True) -> np.ndarray:
            sorted_values = np.array(
                array[sorted_indices], dtype=np.float32, copy=copy
            )
            return sorted_values[restore_order]

        trajectories = torch.from_numpy(
            materialize(self.trajectories)
        ).reshape(batch_size * self.group_size, self.future_len, 4)
        noise_scales = torch.from_numpy(
            materialize(self.noise_scales)
        ).reshape(batch_size * self.group_size)
        weight_array = materialize(self.weights)
        weights = torch.from_numpy(weight_array).reshape(batch_size * self.group_size)
        context_mask = (
            (
                ~np.isfinite(weight_array).all(axis=1)
                | (weight_array > 0.0).any(axis=1)
            )
            if self.context_active_weight_only
            else np.ones(batch_size, dtype=bool)
        )
        compressed: dict[str, np.ndarray] | None = None
        if self.compressed_context is not None:
            active_positions = np.flatnonzero(context_mask)
            if active_positions.size:
                active_indices = indices[active_positions]
                active_read_order = np.argsort(active_indices, kind="stable")
                active_restore_order = np.empty_like(active_read_order)
                active_restore_order[active_read_order] = np.arange(
                    active_read_order.size, dtype=active_read_order.dtype
                )
                active_compressed = self.compressed_context.materialize(
                    active_indices[active_read_order]
                )
            else:
                active_restore_order = np.empty(0, dtype=np.int64)
                active_compressed = {}
            compressed = {}
            for field in self.compressed_context.fields:
                name = str(field["name"])
                shape = tuple(int(value) for value in field["row_shape"])
                values = np.zeros((batch_size, *shape), dtype=np.float32)
                if active_positions.size:
                    values[active_positions] = active_compressed[name][
                        active_restore_order
                    ]
                compressed[name] = values
            for name, array in self.expert_anchor_scalars.items():
                compressed[name] = materialize(array)

        def compressed_materialize(name: str) -> np.ndarray:
            if compressed is None:
                raise RuntimeError("compressed replay context is not active")
            return compressed[name]
        if compressed is not None:
            scene_encoding = torch.from_numpy(
                compressed_materialize("scene_encoding")
            )
        elif self.scene_encoding is not None:
            scene_encoding = torch.from_numpy(materialize(self.scene_encoding))
        else:
            scene_encoding = None
        if compressed is not None:
            replay_context = {
                key: torch.from_numpy(compressed_materialize(key))
                for key in _REPLAY_CONTEXT_KEYS
            }
        else:
            replay_context = {
                key: torch.from_numpy(materialize(array))
                for key, array in self.decoder_context.items()
            }
        if compressed is not None:
            replay_context.update(
                {
                    _REPLAY_EXPERT_TRAJECTORY_KEY: torch.from_numpy(
                        compressed_materialize("expert_trajectories")
                    ),
                    _REPLAY_EXPERT_REWARD_KEY: torch.from_numpy(
                        compressed_materialize("expert_rewards")
                    ),
                    _REPLAY_EXPERT_SAFE_KEY: torch.from_numpy(
                        compressed_materialize("expert_safe")
                    ),
                }
            )
        elif self.expert_anchor:
            replay_context.update(
                {
                    _REPLAY_EXPERT_TRAJECTORY_KEY: torch.from_numpy(
                        materialize(self.expert_anchor["expert_trajectories"])
                    ),
                    _REPLAY_EXPERT_REWARD_KEY: torch.from_numpy(
                        materialize(self.expert_anchor["expert_rewards"])
                    ),
                    _REPLAY_EXPERT_SAFE_KEY: torch.from_numpy(
                        materialize(self.expert_anchor["expert_safe"])
                    ),
                }
            )
        reward_array = materialize(self.rewards)
        finite_reward = np.isfinite(reward_array)
        zero_reward = finite_reward & (np.abs(reward_array) <= 1e-8)
        finite_counts = finite_reward.sum(axis=1)
        row_ranges = np.where(
            finite_counts > 0,
            np.nanmax(reward_array, axis=1) - np.nanmin(reward_array, axis=1),
            np.nan,
        )
        positive_weight = weight_array > 0.0
        active_weight_counts = positive_weight.sum(axis=1)
        weight_sums = weight_array.sum(axis=1)
        weight_square_sums = np.square(weight_array).sum(axis=1)
        ess = np.divide(
            np.square(weight_sums),
            weight_square_sums,
            out=np.zeros_like(weight_sums),
            where=weight_square_sums > 0.0,
        )
        ess_fraction = np.divide(
            ess,
            active_weight_counts,
            out=np.zeros_like(ess),
            where=active_weight_counts > 0,
        )
        top1_share = np.divide(
            weight_array.max(axis=1),
            weight_sums,
            out=np.zeros_like(weight_sums),
            where=weight_sums > 0.0,
        )
        priority_enabled = priority_best_reward_lt is not None
        if priority_enabled:
            priority_mask = (
                np.isfinite(np.nanmax(reward_array, axis=1))
                & (np.nanmax(reward_array, axis=1) < float(priority_best_reward_lt))
                & (weight_sums > 0.0)
            )
            priority_fraction = float(np.mean(priority_mask))
        else:
            priority_fraction = 0.0
        diagnostics = {
            # Per-scene K, matching on-policy/evaluation semantics.  The
            # enclosing row already carries scene_count=batch_size.
            "candidate_count": float(self.group_size),
            "valid_group": float(bool(np.isfinite(weights.numpy()).all())),
            "det_reward": float(np.nanmean(reward_array[:, 0]))
            if finite_reward[:, 0].any()
            else float("nan"),
            "best_reward": float(np.nanmean(np.nanmax(reward_array, axis=1)))
            if finite_reward.any()
            else float("nan"),
            "replay_sampled_with_replacement": float(sampled_with_replacement),
            "replay_priority_enabled": float(priority_enabled),
            "replay_priority_fraction": priority_fraction,
            "replay_priority_best_reward_lt": (
                float(priority_best_reward_lt) if priority_enabled else 0.0
            ),
            "det_zero_reward": float(np.mean(zero_reward[:, 0])),
            "zero_reward_candidate_fraction": float(
                zero_reward.sum() / max(1, finite_reward.sum())
            ),
            "zero_reward_weight_share": float(
                np.mean(
                    np.divide(
                        (weight_array * zero_reward).sum(axis=1),
                        weight_sums,
                        out=np.zeros_like(weight_sums),
                        where=weight_sums > 0.0,
                    )
                )
            ),
            "all_zero_reward_group": float(
                np.mean((finite_counts > 0) & (zero_reward.sum(axis=1) == finite_counts))
            ),
            "all_equal_reward_group": float(np.nanmean(row_ranges <= 1e-8)),
            "reward_std": float(np.nanmean(np.nanstd(reward_array, axis=1))),
            "best_vs_det_reward_gain": float(
                np.nanmean(np.nanmax(reward_array, axis=1) - reward_array[:, 0])
            ),
            "effective_sample_size": float(np.mean(ess)),
            "active_weight_count": float(np.mean(active_weight_counts)),
            "effective_sample_size_fraction": float(np.mean(ess_fraction)),
            "top1_weight_share": float(np.mean(top1_share)),
            "candidate0_deterministic": float(self.deterministic_first),
            "replay_decoder_context_cached": float(
                self.has_cached_decoder_context
            ),
            "replay_expert_anchor_cached": float(
                self.has_cached_expert_anchor
            ),
            "replay_compressed_context_cached": float(
                self.compressed_context is not None
            ),
            "replay_context_materialized_group_fraction": float(
                np.mean(context_mask)
            ),
        }
        if self.positive_advantage_only:
            positive_candidate_counts = positive_weight[:, 1:].sum(axis=1)
            diagnostics.update(
                {
                    "positive_candidate_count": float(
                        np.mean(positive_candidate_counts)
                    ),
                    "has_positive_candidate": float(
                        np.mean(positive_candidate_counts > 0)
                    ),
                    "behavior_anchor_active": float(
                        np.mean(positive_weight[:, 0])
                    ),
                }
            )
        return [self.scene_paths[int(index)] for index in indices], AWRRollout(
            trajectories=trajectories,
            noise_scales=noise_scales,
            rewards=[],
            weights=weights,
            reward_data=replay_context,
            scene_encoding=scene_encoding,
            diagnostics=diagnostics,
        )

    def iter_random_batches(
        self,
        batch_size: int,
        num_batches: int,
        seed: int,
    ):
        """Yield B scene groups, sampled with replacement like ReplayBuffer.get."""

        if self.scene_count <= 0:
            return
        batch_size = max(1, int(batch_size))
        num_batches = max(0, int(num_batches))
        rng = np.random.default_rng(int(seed))
        for _ in range(num_batches):
            indices = rng.integers(0, self.scene_count, size=batch_size, dtype=np.int64)
            yield self.batch_from_indices(indices)

    def iter_reward_stratified_batches(
        self,
        batch_size: int,
        num_batches: int,
        seed: int,
        *,
        priority_best_reward_lt: float,
        priority_fraction: float,
    ):
        """Yield a fixed hard/ordinary mixture from one immutable cache.

        This is an explicit R2LPL-style mistake-priority ablation, not the
        released HDP replay sampler.  A hard group has a finite final best
        reward below ``priority_best_reward_lt`` and at least one non-zero AWR
        target.  The latter prevents intentionally dropped all-zero groups
        from consuming the hard quota.  Both pools are sampled with
        replacement; every batch receives the requested deterministic count.
        """

        if self.scene_count <= 0:
            return
        batch_size = max(1, int(batch_size))
        num_batches = max(0, int(num_batches))
        threshold = float(priority_best_reward_lt)
        fraction = float(priority_fraction)
        if not math.isfinite(threshold):
            raise ValueError("replay priority reward threshold must be finite")
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(
                "replay priority fraction must lie in [0,1], got "
                f"{fraction}"
            )

        reward_array = np.asarray(self.rewards)
        finite_reward = np.isfinite(reward_array)
        finite_counts = finite_reward.sum(axis=1)
        best_reward = np.max(
            np.where(finite_reward, reward_array, -np.inf), axis=1
        )
        active_group = np.asarray(self.weights).sum(axis=1) > 0.0
        hard_mask = (
            (finite_counts > 0)
            & active_group
            & (best_reward < threshold)
        )
        hard_indices = np.flatnonzero(hard_mask)
        ordinary_indices = np.flatnonzero(~hard_mask)
        hard_count = int(round(batch_size * fraction))
        ordinary_count = batch_size - hard_count
        if hard_count > 0 and hard_indices.size == 0:
            raise RuntimeError(
                "reward-stratified replay requested hard scenes but the hard "
                "pool is empty"
            )
        if ordinary_count > 0 and ordinary_indices.size == 0:
            raise RuntimeError(
                "reward-stratified replay requested ordinary scenes but the "
                "ordinary pool is empty"
            )

        rng = np.random.default_rng(int(seed))
        for _ in range(num_batches):
            pieces = []
            if hard_count:
                pieces.append(
                    rng.choice(hard_indices, size=hard_count, replace=True)
                )
            if ordinary_count:
                pieces.append(
                    rng.choice(
                        ordinary_indices, size=ordinary_count, replace=True
                    )
                )
            indices = np.concatenate(pieces).astype(np.int64, copy=False)
            rng.shuffle(indices)
            yield self.batch_from_indices(
                indices,
                sampled_with_replacement=True,
                priority_best_reward_lt=threshold,
            )


def _can_materialize_only_nonzero_replay_groups(
    train_config: dict[str, Any],
) -> bool:
    """Whether zero-weight groups are guaranteed to have zero replay gradient."""

    if float(train_config.get("bc_weight", 0.0)) > 0.0:
        return False
    expert_weight = float(train_config.get("expert_anchor_weight", 0.0))
    if expert_weight > 0.0 and not bool(
        train_config.get("expert_anchor_active_groups_only", False)
    ):
        return False
    return True


def _iter_prefetched_replay_batches(
    replay_iterator,
    scene_load_workers: int,
    prefetch_batches: int,
    replay_context_only: bool,
):
    """Prepare replay memmaps and decoder context while CUDA trains the prior batch.

    One background coordinator consumes the replay iterator in exactly the
    same order. New caches materialize all required tensors directly from
    contiguous memmaps; legacy caches invoke the existing CPU NPZ loader.
    All CUDA copies and every model/optimizer operation stay on the main rank
    thread. Thus prefetch changes scheduling only, not RNG, sample order,
    tensor values, or update order.
    """

    cpu = torch.device("cpu")

    def prepare_item(scenes, rollout):
        cached_expert = {
            key: rollout.reward_data[key]
            for key in _REPLAY_EXPERT_KEYS
            if key in rollout.reward_data
        }
        if cached_expert and len(cached_expert) != len(_REPLAY_EXPERT_KEYS):
            raise RuntimeError(
                "cached replay expert sidecar is incomplete: "
                f"{sorted(cached_expert)}"
            )
        cached = bool(
            float(
                rollout.diagnostics.get(
                    "replay_decoder_context_cached", 0.0
                )
            )
            > 0.5
        )
        if cached:
            missing = [
                key for key in _REPLAY_CONTEXT_KEYS if key not in rollout.reward_data
            ]
            if missing:
                raise RuntimeError(
                    f"cached replay decoder context is incomplete: {missing}"
                )
            data = rollout.reward_data
            # Context is passed through ``data_override`` below.  Remove the
            # duplicate reference from the rollout so _rollout_to_device does
            # not perform a second host-to-device copy of the same tensors.
            rollout_without_context = copy.copy(rollout)
            rollout_without_context.reward_data = {}
            return scenes, rollout_without_context, data
        data = _load_scene_batch(
            scenes,
            cpu,
            workers=max(1, int(scene_load_workers)),
            replay_context_only=replay_context_only,
        )
        data.update(cached_expert)
        if cached_expert:
            rollout_without_expert = copy.copy(rollout)
            rollout_without_expert.reward_data = {
                key: value
                for key, value in rollout.reward_data.items()
                if key not in _REPLAY_EXPERT_KEYS
            }
            rollout = rollout_without_expert
        return scenes, rollout, data

    depth = max(0, int(prefetch_batches))
    if depth == 0:
        for scenes, rollout in replay_iterator:
            yield prepare_item(scenes, rollout)
        return

    source = iter(replay_iterator)
    exhausted = object()

    def prepare_next():
        try:
            scenes, rollout = next(source)
        except StopIteration:
            return exhausted
        return prepare_item(scenes, rollout)

    # A single coordinator keeps Python generator access strictly serial.
    # Its scene loader may still use the measured-optimal inner worker count.
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = deque(executor.submit(prepare_next) for _ in range(depth))
        while pending:
            prepared = pending.popleft().result()
            if prepared is exhausted:
                break
            pending.append(executor.submit(prepare_next))
            yield prepared


def _iter_prefetched_scene_batches(
    local_order: list[str],
    batch_starts,
    batch_size: int,
    scene_load_workers: int,
    prefetch_batches: int,
    shared_scene_encoding: np.ndarray | None = None,
    shared_expert_anchor: dict[str, np.ndarray] | None = None,
):
    """Read the next rollout batch on CPU while CUDA samples the current one.

    The coordinator is single-threaded, so it consumes ``batch_starts`` in
    exactly the original order.  Inner NPZ loading may retain its configured
    thread count, but all RNG, CUDA copies, model calls, reward calculation and
    replay writes remain on the main rank thread.  A loader exception is
    returned to that thread so the existing packed-batch isolation path still
    retries every scene individually instead of losing a whole batch.
    """

    depth = max(0, int(prefetch_batches))
    source = iter(batch_starts)
    if (
        depth == 0
        and shared_scene_encoding is None
        and not shared_expert_anchor
    ):
        for batch_start in source:
            yield batch_start, None, None, None
        return

    cpu = torch.device("cpu")
    exhausted = object()

    def prepare_next():
        try:
            batch_start = next(source)
        except StopIteration:
            return exhausted
        scene_paths = local_order[batch_start : batch_start + batch_size]
        try:
            data = _load_scene_batch(
                scene_paths,
                cpu,
                workers=max(1, int(scene_load_workers)),
                defer_full_scene_heading_transforms=True,
            )
            if shared_scene_encoding is not None:
                end = batch_start + len(scene_paths)
                encoding = np.array(
                    shared_scene_encoding[batch_start:end],
                    dtype=np.float32,
                    copy=True,
                )
                if int(encoding.shape[0]) != len(scene_paths):
                    raise RuntimeError(
                        "shared encoding batch is shorter than scene batch: "
                        f"{encoding.shape[0]} != {len(scene_paths)}"
                    )
                data["_cached_encoding"] = torch.from_numpy(encoding)
            if shared_expert_anchor:
                expected_names = {
                    "expert_trajectories": _REPLAY_EXPERT_TRAJECTORY_KEY,
                    "expert_rewards": _REPLAY_EXPERT_REWARD_KEY,
                    "expert_safe": _REPLAY_EXPERT_SAFE_KEY,
                }
                if set(shared_expert_anchor) != set(expected_names):
                    raise RuntimeError(
                        "shared expert-anchor arrays are incomplete: "
                        f"{sorted(shared_expert_anchor)}"
                    )
                end = batch_start + len(scene_paths)
                for name, data_key in expected_names.items():
                    values = np.array(
                        shared_expert_anchor[name][batch_start:end],
                        dtype=np.float32,
                        copy=True,
                    )
                    if int(values.shape[0]) != len(scene_paths):
                        raise RuntimeError(
                            "shared expert-anchor batch is shorter than scene "
                            f"batch for {name}: {values.shape[0]} != "
                            f"{len(scene_paths)}"
                        )
                    data[data_key] = torch.from_numpy(values)
            return batch_start, scene_paths, data, None
        except Exception as error:
            return batch_start, scene_paths, None, error

    if depth == 0:
        while True:
            prepared = prepare_next()
            if prepared is exhausted:
                break
            yield prepared
        return

    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = deque(executor.submit(prepare_next) for _ in range(depth))
        while pending:
            prepared = pending.popleft().result()
            if prepared is exhausted:
                break
            pending.append(executor.submit(prepare_next))
            yield prepared


def _salvage_partial_replay_manifest(root: Path, rank: int) -> dict[str, Any]:
    """Recover a fully-written prefix from an interrupted rollout cache.

    The disk writer preallocates every array to the padded rank-local corpus
    size.  If one packed scene batch fails, ``close()`` intentionally refuses
    to publish a manifest, but every earlier prefix is still complete and
    auditable through ``scene_paths.jsonl``.  This recovery path publishes
    only that common prefix; it never exposes the zero-filled tail and never
    mutates the large memmaps.
    """

    rank_dir = Path(root) / f"rank_{int(rank):04d}"
    paths_path = rank_dir / "scene_paths.jsonl"
    if not paths_path.exists():
        raise FileNotFoundError(paths_path)
    with paths_path.open() as handle:
        scene_count = sum(1 for line in handle if line.strip())
    if scene_count <= 0:
        raise RuntimeError(f"partial replay rank {rank} contains no scene paths")

    writer_contract_path = rank_dir / "writer_contract.json"
    if not writer_contract_path.is_file():
        raise RuntimeError(
            "partial replay predates the canonical-order writer contract; "
            "refusing unsafe count-only salvage"
        )
    writer_contract = json.loads(writer_contract_path.read_text())
    expected_paths_name = writer_contract.get("expected_paths")
    if not expected_paths_name:
        raise RuntimeError("partial replay has no canonical expected-path index")
    expected_paths_path = rank_dir / str(expected_paths_name)
    if not expected_paths_path.is_file():
        raise FileNotFoundError(expected_paths_path)
    with paths_path.open() as actual_handle, expected_paths_path.open() as expected_handle:
        for index in range(scene_count):
            actual_line = actual_handle.readline()
            expected_line = expected_handle.readline()
            if not actual_line or not expected_line:
                raise RuntimeError(
                    f"partial replay path index ended before scene {index}"
                )
            actual_path = json.loads(actual_line)
            expected_path = json.loads(expected_line)
            if actual_path != expected_path:
                raise RuntimeError(
                    "partial replay is not a canonical prefix: "
                    f"rank={rank}, index={index}, actual={actual_path!r}, "
                    f"expected={expected_path!r}"
                )

    arrays: dict[str, np.memmap] = {}
    required_names = [
        "trajectories",
        "noise_scales",
        "weights",
        "rewards",
        "scene_encoding",
    ]
    context_names = [f"context_{key}" for key in _REPLAY_CONTEXT_KEYS]
    present_context = [
        name for name in context_names if (rank_dir / f"{name}.npy").exists()
    ]
    if present_context and len(present_context) != len(context_names):
        raise RuntimeError(
            "partial replay has an incomplete decoder context: "
            f"present={present_context}"
        )
    context_build_manifest: dict[str, Any] | None = None
    if present_context and writer_contract.get(
        "decoder_context_schema"
    ) != _REPLAY_DECODER_CONTEXT_SCHEMA:
        # Cycle-1 built the policy-independent compact decoder context in
        # parallel with candidate mining.  That sidecar intentionally does
        # not mutate the live writer contract before strict replay close.
        # Accept it for prefix salvage only when its own atomic completion
        # manifest proves the exact full rank order, geometry and +1 legacy
        # neighbour alignment.  Merely finding context .npy files is never
        # enough to bypass the schema guard.
        context_build_path = rank_dir / "context_build_manifest.json"
        if not context_build_path.is_file():
            raise RuntimeError(
                "partial replay decoder context has an unsafe/unknown "
                "alignment schema"
            )
        context_build_manifest = json.loads(context_build_path.read_text())
        expected_context_arrays = {
            key: f"context_{key}.npy" for key in _REPLAY_CONTEXT_KEYS
        }
        if (
            context_build_manifest.get("schema")
            != _REPLAY_DECODER_CONTEXT_SCHEMA
            or int(context_build_manifest.get("rank", -1)) != int(rank)
            or int(context_build_manifest.get("scene_count", -1))
            != int(writer_contract.get("expected_count", -1))
            or int(
                context_build_manifest.get(
                    "neighbor_future_alignment_offset", -1
                )
            )
            != int(get_neighbor_future_offset())
            or float(
                context_build_manifest.get("x2_legacy_ego_width_m", float("nan"))
            )
            != 2.29156
            or context_build_manifest.get("expected_paths_sha256")
            != _file_sha256(expected_paths_path)
            or context_build_manifest.get("arrays") != expected_context_arrays
        ):
            raise RuntimeError(
                "partial replay independently-built decoder context failed "
                "its canonical alignment contract"
            )
    for name in [*required_names, *present_context]:
        array_path = rank_dir / f"{name}.npy"
        if not array_path.exists():
            raise FileNotFoundError(array_path)
        arrays[name] = np.load(array_path, mmap_mode="r")

    if context_build_manifest is not None:
        context_shapes = context_build_manifest.get("shapes", {})
        for key in _REPLAY_CONTEXT_KEYS:
            name = f"context_{key}"
            if [int(value) for value in arrays[name].shape] != [
                int(value) for value in context_shapes.get(key, [])
            ]:
                raise RuntimeError(
                    f"partial replay independently-built context {key} "
                    "shape differs from its completion manifest"
                )

    capacity = int(arrays["trajectories"].shape[0])
    if scene_count > capacity:
        raise RuntimeError(
            f"partial replay rank {rank} path count {scene_count} exceeds "
            f"array capacity {capacity}"
        )
    if any(int(array.shape[0]) != capacity for array in arrays.values()):
        shapes = {name: tuple(array.shape) for name, array in arrays.items()}
        raise RuntimeError(f"partial replay rank {rank} array capacities differ: {shapes}")

    trajectories = arrays["trajectories"]
    if trajectories.ndim != 4 or trajectories.shape[-1] != 4:
        raise RuntimeError(
            f"partial replay rank {rank} has invalid trajectory shape "
            f"{tuple(trajectories.shape)}"
        )
    group_size = int(trajectories.shape[1])
    future_len = int(trajectories.shape[2])
    expected_group_shapes = {
        "noise_scales": (capacity, group_size),
        "weights": (capacity, group_size),
        "rewards": (capacity, group_size),
    }
    for name, expected_shape in expected_group_shapes.items():
        if tuple(arrays[name].shape) != expected_shape:
            raise RuntimeError(
                f"partial replay rank {rank} {name} shape "
                f"{tuple(arrays[name].shape)} != {expected_shape}"
            )

    # Probe both ends of the published prefix.  This is cheap even for the
    # 300-GB encoding memmap and catches truncated/zero-tail publication.
    probe = np.asarray(sorted({0, scene_count - 1}), dtype=np.int64)
    for name, array in arrays.items():
        values = np.asarray(array[probe])
        if not np.isfinite(values).all():
            raise RuntimeError(
                f"partial replay rank {rank} has non-finite {name} values "
                "inside the recoverable prefix"
            )
    probe_weights = np.asarray(arrays["weights"][probe])
    probe_rewards = np.asarray(arrays["rewards"][probe])
    has_weight = (probe_weights > 0).any(axis=1)
    intentional_all_zero = (
        np.isfinite(probe_rewards).all(axis=1)
        & (np.abs(probe_rewards) <= 1e-8).all(axis=1)
    )
    if not bool((has_weight | intentional_all_zero).all()):
        raise RuntimeError(
            f"partial replay rank {rank} has an unexplained all-zero "
            "AWR-weight probe"
        )

    manifest = {
        "version": 3 if present_context else 1,
        "rank": int(rank),
        "scene_count": int(scene_count),
        "group_size": group_size,
        "future_len": future_len,
        "encoding_shape": list(arrays["scene_encoding"].shape[1:]),
        "decoder_context_cached": bool(present_context),
        "decoder_context_schema": (
            _REPLAY_DECODER_CONTEXT_SCHEMA if present_context else None
        ),
        "decoder_context_shapes": {
            key: list(arrays[f"context_{key}"].shape[1:])
            for key in _REPLAY_CONTEXT_KEYS
            if f"context_{key}" in arrays
        },
        "arrays": {name: f"{name}.npy" for name in arrays},
        "paths": paths_path.name,
        "expected_paths": expected_paths_path.name,
        "salvaged_partial_prefix": True,
        "array_capacity": capacity,
        "discarded_tail_slots": capacity - scene_count,
        "shared_scene_encoding_path": writer_contract.get(
            "shared_scene_encoding_path"
        ),
        "shared_decoder_context_paths": writer_contract.get(
            "shared_decoder_context_paths", {}
        ),
    }
    _write_json_atomic(rank_dir / "manifest.json", manifest)
    return manifest


def _is_complete_mining_rank_cache(
    root: Path,
    rank: int,
    expected_count: int,
    group_size: int,
    future_len: int,
    expected_scene_paths: list[str],
) -> bool:
    """Return whether one rank is safe to reuse during mining recovery.

    A mining restart must never trust a preallocated array merely because it
    exists: the old process may have been killed before publishing its commit
    marker.  This check accepts only a strict manifest whose path index is the
    exact deterministic order for this run and whose referenced arrays have
    the expected scene-major shapes.  It intentionally does not read the
    multi-terabyte payload; ``_DiskReplayReader`` performs its normal finite
    value probes after the final all-rank barrier.
    """

    try:
        rank_dir = Path(root) / f"rank_{int(rank):04d}"
        manifest_path = rank_dir / "manifest.json"
        writer_contract_path = rank_dir / "writer_contract.json"
        if not manifest_path.is_file() or not writer_contract_path.is_file():
            return False
        manifest = json.loads(manifest_path.read_text())
        writer = json.loads(writer_contract_path.read_text())
        if (
            int(manifest.get("version", 0)) < 3
            or int(manifest.get("rank", -1)) != int(rank)
            or int(manifest.get("scene_count", -1)) != int(expected_count)
            or int(manifest.get("group_size", -1)) != int(group_size)
            or int(manifest.get("future_len", -1)) != int(future_len)
            or int(writer.get("expected_count", -1)) != int(expected_count)
            or not bool(manifest.get("decoder_context_cached", False))
            or manifest.get("decoder_context_schema")
            != _REPLAY_DECODER_CONTEXT_SCHEMA
            or not bool(writer.get("expert_anchor_cached", False))
        ):
            return False

        paths_name = manifest.get("paths")
        expected_paths_name = writer.get("expected_paths")
        if not paths_name or not expected_paths_name:
            return False
        paths_path = rank_dir / str(paths_name)
        expected_paths_path = rank_dir / str(expected_paths_name)
        if not paths_path.is_file() or not expected_paths_path.is_file():
            return False
        if len(expected_scene_paths) != int(expected_count):
            return False
        # Compare the committed path index directly with the deterministic
        # order already held by this rank.  This catches reuse of a cache from
        # another seed, dataset, or shuffle epoch without hashing 680k Python
        # strings into a second large object.
        with paths_path.open() as actual_handle, expected_paths_path.open() as expected_handle:
            for expected in expected_scene_paths:
                actual_line = actual_handle.readline()
                canonical_line = expected_handle.readline()
                if not actual_line or not canonical_line:
                    return False
                if json.loads(actual_line) != expected:
                    return False
                if json.loads(canonical_line) != expected:
                    return False
            if any(line.strip() for line in actual_handle):
                return False
            if any(line.strip() for line in expected_handle):
                return False

        arrays = dict(manifest.get("arrays", {}))
        required_arrays = [
            "trajectories",
            "noise_scales",
            "weights",
            "rewards",
            "scene_encoding",
            *(f"context_{key}" for key in _REPLAY_CONTEXT_KEYS),
        ]
        for name in required_arrays:
            filename = arrays.get(name)
            if not filename:
                return False
            path = rank_dir / str(filename)
            if not path.is_file():
                return False
        expert_manifest_path = rank_dir / "expert_anchor_manifest.json"
        if not expert_manifest_path.is_file():
            return False
        expert_manifest = json.loads(expert_manifest_path.read_text())
        if (
            expert_manifest.get("schema") != _REPLAY_EXPERT_SCHEMA
            or int(expert_manifest.get("scene_count", -1)) != int(expected_count)
        ):
            return False
        expert_arrays = dict(expert_manifest.get("arrays", {}))
        for name in ("expert_trajectories", "expert_rewards", "expert_safe"):
            filename = expert_arrays.get(name)
            if not filename or not (rank_dir / str(filename)).is_file():
                return False

        loaded: dict[str, np.ndarray] = {}
        for name in required_arrays:
            loaded[name] = np.load(rank_dir / str(arrays[name]), mmap_mode="r")
        trajectories = loaded["trajectories"]
        if (
            trajectories.ndim != 4
            or int(trajectories.shape[0]) != int(expected_count)
            or int(trajectories.shape[1]) != int(group_size)
            or int(trajectories.shape[2]) != int(future_len)
            or int(trajectories.shape[3]) != 4
        ):
            return False
        for name in ("noise_scales", "weights", "rewards"):
            if tuple(loaded[name].shape) != (int(expected_count), int(group_size)):
                return False
        for name in ["scene_encoding", *(f"context_{key}" for key in _REPLAY_CONTEXT_KEYS)]:
            if loaded[name].ndim < 1 or int(loaded[name].shape[0]) < int(expected_count):
                return False
        expected_expert_shapes = {
            "expert_trajectories": (int(expected_count), int(future_len), 4),
            "expert_rewards": (int(expected_count),),
            "expert_safe": (int(expected_count),),
        }
        for name, shape in expected_expert_shapes.items():
            array = np.load(rank_dir / str(expert_arrays[name]), mmap_mode="r")
            if tuple(array.shape) != shape:
                return False
        return True
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return False


def _combine_rollouts(rollouts: list[AWRRollout], data: dict[str, torch.Tensor]) -> AWRRollout:
    """Pack per-scene groups for one batched diffusion-regression forward."""

    if not rollouts:
        raise ValueError("cannot combine an empty rollout list")
    if len(rollouts) == 1:
        return rollouts[0]
    diagnostics: dict[str, float] = {}
    keys = set().union(*(rollout.diagnostics for rollout in rollouts))
    for key in keys:
        values = [rollout.diagnostics[key] for rollout in rollouts if key in rollout.diagnostics]
        if values and all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
            diagnostics[key] = float(np.mean([float(value) for value in values]))
    # Diagnostics are per scene. Keep K rather than reporting B*K for a packed
    # rollout; the enclosing row separately records scene_count=B.
    diagnostics["candidate_count"] = float(
        np.mean([rollout.trajectories.shape[0] for rollout in rollouts])
    )
    diagnostics["valid_group"] = float(
        all(float(rollout.diagnostics.get("valid_group", 0.0)) > 0.5 for rollout in rollouts)
    )
    return AWRRollout(
        trajectories=torch.cat([rollout.trajectories for rollout in rollouts], dim=0),
        noise_scales=torch.cat([rollout.noise_scales for rollout in rollouts], dim=0),
        rewards=[reward for rollout in rollouts for reward in rollout.rewards],
        weights=torch.cat([rollout.weights for rollout in rollouts], dim=0),
        reward_data=data,
        scene_encoding=torch.cat([rollout.scene_encoding for rollout in rollouts], dim=0),
        diagnostics=diagnostics,
    )


def _expert_reward_is_hard_safe(reward: Any, reward_config: RewardConfig) -> bool:
    """Apply the exact configured hard gates to one logged-expert future."""

    return bool(
        reward.collision_step is None
        and (not reward_config.rb_gate_enabled or not reward.rb_crossing)
        and (not reward_config.lane_gate_enabled or not reward.lane_crossing)
        and (
            not reward_config.static_collision_enabled
            or not reward_config.sc_gate_enabled
            or not reward.static_crossing
        )
        and reward.red_light >= -0.5
        and not reward.kinematic_violated
    )


@torch.no_grad()
def _score_expert_anchor_batch(
    data: dict[str, torch.Tensor],
    future_len: int,
    reward_config: RewardConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Build and independently hard-score one logged target per scene.

    Reward geometry cannot be broadcast across scenes because map, actors,
    wheelbase, and masks differ.  The model sampling/denoising path remains
    fully batched; this optional retention audit performs one additional
    reward call per scene and is disabled in the faithful HDP experiment.
    """

    future = data.get("ego_agent_future")
    if not isinstance(future, torch.Tensor):
        return None
    expert = heading_to_cos_sin(future)[:, : int(future_len), :4]
    batch_size = int(expert.shape[0])
    rewards = torch.empty(batch_size, device=expert.device, dtype=torch.float32)
    safe = torch.zeros(batch_size, device=expert.device, dtype=torch.bool)
    for scene_index in range(batch_size):
        scene_data = _slice_eval_scene_data(data, scene_index)
        scene_reward = compute_reward_batch(
            expert[scene_index : scene_index + 1],
            reward_compatible_data(scene_data),
            reward_config,
        )[0]
        rewards[scene_index] = float(scene_reward.total)
        safe[scene_index] = _expert_reward_is_hard_safe(
            scene_reward, reward_config
        )
    return expert, rewards, safe


def _inject_expert_anchor_batch(
    target_trajectories: torch.Tensor,
    target_weights: torch.Tensor,
    expert_trajectories: torch.Tensor,
    expert_safe: torch.Tensor,
    *,
    group_size: int,
    expert_weight: float,
    replace_worst: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Insert safe experts scene-major without mixing different groups."""

    batch_size = int(expert_trajectories.shape[0])
    group_size = int(group_size)
    expected = batch_size * group_size
    if target_trajectories.shape[0] != expected or target_weights.numel() != expected:
        raise ValueError(
            "expert-anchor group shape mismatch: "
            f"B={batch_size}, K={group_size}, "
            f"targets={tuple(target_trajectories.shape)}, weights={tuple(target_weights.shape)}"
        )
    if expert_safe.shape != (batch_size,):
        raise ValueError(
            f"expert safe mask must be [{batch_size}], got {tuple(expert_safe.shape)}"
        )
    trajectories = target_trajectories.reshape(
        batch_size, group_size, *target_trajectories.shape[1:]
    )
    weights = target_weights.reshape(batch_size, group_size)
    safe = expert_safe.to(device=weights.device, dtype=torch.bool)
    experts = expert_trajectories.to(device=trajectories.device, dtype=trajectories.dtype)
    if replace_worst:
        trajectories = trajectories.clone()
        weights = weights.clone()
        rows = torch.arange(batch_size, device=weights.device)[safe]
        if rows.numel() > 0:
            columns = torch.argmin(weights, dim=1)[safe]
            trajectories[rows, columns] = experts[safe]
            weights[rows, columns] = float(expert_weight)
    else:
        # Keep a fixed K+1 shape for compiled batched training.  Unsafe logged
        # targets occupy a zero-weight slot and therefore cannot override a
        # configured hard gate or change the weighted objective.
        anchor_weights = torch.where(
            safe,
            torch.full_like(safe, float(expert_weight), dtype=weights.dtype),
            torch.zeros_like(safe, dtype=weights.dtype),
        )
        trajectories = torch.cat([trajectories, experts[:, None]], dim=1)
        weights = torch.cat([weights, anchor_weights[:, None]], dim=1)
    return trajectories.flatten(0, 1), weights.flatten()


def _build_awr_target_loss_horizons(
    target_weights: torch.Tensor,
    *,
    batch_size: int,
    group_size: int,
    future_len: int,
    candidate_horizon: int,
    deterministic_first: bool,
    expert_safe: torch.Tensor | None,
    expert_replace_worst: bool,
) -> torch.Tensor:
    """Assign scorer-supported horizons while retaining full-horizon anchors."""

    expected = int(batch_size) * int(group_size)
    if target_weights.numel() != expected:
        raise ValueError(
            "candidate loss-horizon group shape mismatch: "
            f"targets={target_weights.numel()}, B={batch_size}, K={group_size}"
        )
    future_len = int(future_len)
    candidate_horizon = int(candidate_horizon)
    if not 1 <= candidate_horizon <= future_len:
        raise ValueError(
            "candidate loss horizon must lie inside the model future: "
            f"horizon={candidate_horizon}, T={future_len}"
        )
    horizons = torch.full(
        (int(batch_size), int(group_size)),
        candidate_horizon,
        dtype=torch.long,
        device=target_weights.device,
    )
    if deterministic_first:
        horizons[:, 0] = future_len
    if expert_safe is None:
        return horizons.flatten()
    if expert_safe.shape != (int(batch_size),):
        raise ValueError(
            "expert safe mask must match the loss-horizon scene batch: "
            f"expected={(int(batch_size),)}, got={tuple(expert_safe.shape)}"
        )
    safe = expert_safe.to(device=target_weights.device, dtype=torch.bool)
    if expert_replace_worst:
        horizons = horizons.clone()
        rows = torch.arange(int(batch_size), device=target_weights.device)[safe]
        if rows.numel() > 0:
            columns = torch.argmin(
                target_weights.reshape(int(batch_size), int(group_size)), dim=1
            )[rows]
            horizons[rows, columns] = future_len
        return horizons.flatten()
    # The expert injection keeps a fixed K+1 graph shape; an unsafe expert is
    # zero-weight but still occupies its final slot.  Giving every such slot
    # the full horizon keeps the horizon vector exactly shape-aligned.
    return torch.cat(
        [
            horizons,
            torch.full(
                (int(batch_size), 1),
                future_len,
                dtype=torch.long,
                device=target_weights.device,
            ),
        ],
        dim=1,
    ).flatten()


def _mask_expert_anchor_to_active_awr_groups(
    expert_safe: torch.Tensor,
    candidate_weights: torch.Tensor,
    *,
    batch_size: int,
    group_size: int,
) -> torch.Tensor:
    """Keep expert retention only where the same scene has an AWR signal."""

    if expert_safe.shape != (int(batch_size),):
        raise ValueError(
            "expert safe mask must match the AWR scene batch: "
            f"expected={(int(batch_size),)}, got={tuple(expert_safe.shape)}"
        )
    if candidate_weights.numel() != int(batch_size) * int(group_size):
        raise ValueError(
            "candidate weights do not match the AWR group shape: "
            f"weights={candidate_weights.numel()}, B={batch_size}, K={group_size}"
        )
    active = candidate_weights.reshape(int(batch_size), int(group_size)).gt(0).any(
        dim=1
    )
    return expert_safe.to(device=active.device, dtype=torch.bool) & active


def _train_scene(
    model: nn.Module,
    behavior_model: nn.Module,
    model_args,
    optimizer,
    scene_path: str | list[str],
    rollout_config: AWRRolloutConfig,
    reward_config: RewardConfig,
    train_config: dict[str, Any],
    device: torch.device,
    amp_dtype: str,
    grad_scaler: Any | None = None,
    rollout_override: AWRRollout | None = None,
    replay_update: bool = False,
    optimizer_step: bool = True,
    zero_grad: bool = True,
    sync_gradients: bool = True,
    gradient_scale: float = 1.0,
    collect_only: bool = False,
    scene_load_workers: int = 1,
    data_override: dict[str, torch.Tensor] | None = None,
    exploration_reference_model: nn.Module | None = None,
    exploration_reference_model_args: Any | None = None,
) -> tuple[dict[str, Any], AWRollout | None]:
    scene_paths = [scene_path] if isinstance(scene_path, str) else list(scene_path)
    if not scene_paths:
        raise ValueError("scene_path batch is empty")
    expert_anchor_weight = float(train_config.get("expert_anchor_weight", 0.0))
    # Persisting the policy-independent expert sidecar is a replay-cache
    # contract, not necessarily a request to use that anchor in this call's
    # objective.  Production mining normally has expert_anchor_weight > 0,
    # but an in-place tail recovery must be able to reconstruct a cache whose
    # writer contract requires the sidecar even if a recovery config omitted
    # or overrode the training weight.  This flag only scores/caches the
    # logged future during collect_only; it never injects an expert target
    # when expert_anchor_weight == 0.
    cache_replay_expert_anchor = bool(
        train_config.get("cache_replay_expert_anchor", False)
    )
    replay_context_only = bool(
        replay_update
        and rollout_override is not None
        and rollout_override.scene_encoding is not None
        and expert_anchor_weight <= 0.0
    )
    if data_override is None:
        data = _load_scene_batch(
            scene_paths,
            device,
            workers=scene_load_workers,
            replay_context_only=replay_context_only,
        )
    else:
        data = {
            key: value.to(device=device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in data_override.items()
        }
        # Rollout prefetch deliberately leaves raw heading angles on CPU so
        # cosine/sine is evaluated on the destination CUDA device, matching
        # the historical non-prefetched loader bit-for-bit.  The helper is
        # idempotent for already-converted 4-channel tensors and a no-op for
        # compact replay context.
        apply_npz_heading_transforms(data)
    scoring_reward_config = reward_config
    # In the HDP/PDM profile lane robustness is the centerline-based reward
    # term.  The separate lane-departure polygon diagnostic is not a reward
    # or a hard gate in the faithful full-corpus config (all lane penalty
    # scales and lane_gate_enabled are off).  It is nevertheless expensive:
    # it evaluates ego perimeter containment against every nearby lane for
    # every one of K sampled trajectories.  Skip only that unused diagnostic
    # during rollout mining; the exact expert lane-change mask inside the HDP
    # lane reward remains active, and all validation/evaluation keeps the full
    # metric config.  Thus AWR weights and their hard semantics are unchanged.
    if (
        collect_only
        and bool(train_config.get("skip_unused_hdp_diagnostics", False))
        and reward_config.reward_profile in {"hdp_pdm", "hdp_multi"}
        and not reward_config.lane_gate_enabled
        and float(reward_config.lane_near_scale) == 0.0
        and float(reward_config.lane_wide_scale) == 0.0
        and float(reward_config.lane_cont_scale) == 0.0
    ):
        scoring_reward_config = copy.copy(reward_config)
        scoring_reward_config.enable_lane_departure = False
    if rollout_override is None:
        if len(scene_paths) == 1:
            rollout = rollout_and_score_scene(
                behavior_model,
                model_args,
                data,
                rollout_config,
                scoring_reward_config,
                device,
                exploration_reference_model=exploration_reference_model,
                exploration_reference_model_args=exploration_reference_model_args,
            )
        else:
            rollout = _combine_rollouts(
                rollout_and_score_scene_batch(
                    behavior_model,
                    model_args,
                    data,
                    rollout_config,
                    scoring_reward_config,
                    device,
                    exploration_reference_model=exploration_reference_model,
                    exploration_reference_model_args=exploration_reference_model_args,
                ),
                data,
            )
    else:
        rollout = _rollout_to_device(rollout_override, device)
    result: dict[str, Any] = {
        "scene_path": scene_path,
        "scene_count": len(scene_paths),
        **rollout.diagnostics,
    }
    if replay_update:
        result["replay_update"] = 1.0
    expert_anchor_batch: torch.Tensor | None = None
    expert_anchor_safe: torch.Tensor | None = None
    if expert_anchor_weight > 0.0 or (
        collect_only and cache_replay_expert_anchor
    ):
        cached_expert_keys = [key for key in _REPLAY_EXPERT_KEYS if key in data]
        if cached_expert_keys and len(cached_expert_keys) != len(
            _REPLAY_EXPERT_KEYS
        ):
            raise RuntimeError(
                "cached replay expert sidecar is incomplete: "
                f"{sorted(cached_expert_keys)}"
            )
        if cached_expert_keys:
            expert_anchor_batch = data[_REPLAY_EXPERT_TRAJECTORY_KEY]
            expert_rewards = data[_REPLAY_EXPERT_REWARD_KEY].float()
            expert_anchor_safe = data[_REPLAY_EXPERT_SAFE_KEY] > 0.5
            expected_expert_shape = (
                len(scene_paths),
                int(model_args.future_len),
                4,
            )
            if tuple(expert_anchor_batch.shape) != expected_expert_shape:
                raise RuntimeError(
                    "cached replay expert trajectory shape mismatch: "
                    f"{tuple(expert_anchor_batch.shape)} != "
                    f"{expected_expert_shape}"
                )
            if tuple(expert_rewards.shape) != (len(scene_paths),):
                raise RuntimeError(
                    "cached replay expert reward shape mismatch: "
                    f"{tuple(expert_rewards.shape)}"
                )
            if tuple(expert_anchor_safe.shape) != (len(scene_paths),):
                raise RuntimeError(
                    "cached replay expert safe shape mismatch: "
                    f"{tuple(expert_anchor_safe.shape)}"
                )
            scored_experts = (
                expert_anchor_batch,
                expert_rewards,
                expert_anchor_safe,
            )
            result["expert_anchor_cache_hit"] = 1.0
        else:
            scored_experts = _score_expert_anchor_batch(
                data, int(model_args.future_len), reward_config
            )
            result["expert_anchor_cache_hit"] = 0.0
        if scored_experts is not None:
            expert_anchor_batch, expert_rewards, expert_anchor_safe = scored_experts
            safe_count = int(expert_anchor_safe.sum().item())
            result["expert_anchor_count"] = float(safe_count)
            result["expert_anchor_used"] = safe_count / max(1, len(scene_paths))
            if safe_count:
                result["expert_anchor_reward"] = float(
                    expert_rewards[expert_anchor_safe].mean().item()
                )
        else:
            result["expert_anchor_used"] = 0.0
    else:
        result["expert_anchor_used"] = 0.0

    if collect_only:
        # Faithful HDP phase: one epoch only mines a frozen-behavior replay
        # buffer; the following epochs train on those fixed groups. Keep
        # this before optimizer/gradient construction so the rollout epoch is
        # genuinely update-free rather than mixing on-policy and replay loss.
        if bool(train_config.get("cache_replay_decoder_context", False)):
            # Replace the scoring-oriented reward view with the exact compact
            # decoder context.  It is consumed only by the disk writer and
            # avoids reopening the compressed source NPZ in every replay
            # epoch.  Candidate trajectories/rewards/weights are untouched.
            rollout.reward_data = _compact_replay_decoder_context(
                data,
                int(model_args.predicted_neighbor_num),
            )
        if (
            expert_anchor_batch is not None
            and expert_anchor_safe is not None
            and (expert_anchor_weight > 0.0 or cache_replay_expert_anchor)
        ):
            # These policy-independent values were already constructed and
            # hard-gated above.  Persist them once with the mined cache so
            # every replay epoch consumes byte-identical expert targets
            # without reopening full NPZ geometry or recomputing reward.
            rollout.reward_data[_REPLAY_EXPERT_TRAJECTORY_KEY] = (
                expert_anchor_batch.detach()
            )
            rollout.reward_data[_REPLAY_EXPERT_REWARD_KEY] = (
                expert_rewards.detach()
            )
            rollout.reward_data[_REPLAY_EXPERT_SAFE_KEY] = (
                expert_anchor_safe.detach().to(dtype=torch.float32)
            )
        result.update(
            {
                "skipped": 0.0,
                "loss": 0.0,
                "grad_norm": 0.0,
                "weight_sum": float(rollout.weights.sum().item()),
                "optimizer_step": 0.0,
                "rollout_only": 1.0,
                "shared_encoder_cache_hit": float(
                    isinstance(data.get("_cached_encoding"), torch.Tensor)
                ),
            }
        )
        return result, rollout

    if (
        expert_anchor_safe is not None
        and bool(train_config.get("expert_anchor_active_groups_only", False))
    ):
        original_safe_count = int(expert_anchor_safe.sum().item())
        expert_anchor_safe = _mask_expert_anchor_to_active_awr_groups(
            expert_anchor_safe,
            rollout.weights,
            batch_size=len(scene_paths),
            group_size=int(rollout_config.n_trajectories),
        )
        active_safe_count = int(expert_anchor_safe.sum().item())
        result["expert_anchor_safe_before_active_group_gate"] = float(
            original_safe_count
        )
        result["expert_anchor_active_group_count"] = float(active_safe_count)
        result["expert_anchor_active_group_gate"] = 1.0

    if (
        float(rollout.diagnostics.get("valid_group", 0.0)) < 0.5
        and not bool(
            expert_anchor_safe is not None and expert_anchor_safe.any().item()
        )
    ):
        # This is only reachable for a group with no finite reward at all.
        # A tied finite group remains a valid/provenance-complete row even when
        # the explicit HDP-compatible drop flag gives it all-zero weights, so
        # it does not enter this malformed-scene fallback.
        if hasattr(model, "no_sync"):
            target_trajectories = rollout.trajectories
            target_weights = torch.zeros_like(rollout.weights)
            target_weights[0] = 1.0
            result["fallback_nonfinite_update"] = 1.0
        else:
            result["skipped"] = 1.0
            result["skip_reason"] = "zero_or_nonfinite_group_std"
            return result, rollout
    else:
        target_trajectories = rollout.trajectories
        target_weights = rollout.weights

    planner = model.module if hasattr(model, "module") else model
    planner.train()
    planner.encoder.eval()
    planner.decoder.turn_indicator_predictor.eval()
    if zero_grad:
        optimizer.zero_grad(set_to_none=True)
    P = 1 + int(model_args.predicted_neighbor_num)
    T = int(model_args.future_len)
    candidate_loss_horizon = int(
        train_config.get("awr_candidate_loss_horizon", 0)
    )
    if candidate_loss_horizon == 0:
        candidate_loss_horizon = T
    if not 1 <= candidate_loss_horizon <= T:
        raise ValueError(
            "awr_candidate_loss_horizon must be 0 or lie inside the model "
            f"future: horizon={candidate_loss_horizon}, T={T}"
        )
    initial_group_size = int(rollout_config.n_trajectories)
    replace_expert_worst = bool(
        train_config.get("expert_anchor_replace_worst", False)
    )
    target_loss_horizons = _build_awr_target_loss_horizons(
        target_weights,
        batch_size=len(scene_paths),
        group_size=initial_group_size,
        future_len=T,
        candidate_horizon=candidate_loss_horizon,
        deterministic_first=bool(rollout_config.deterministic_first),
        expert_safe=(
            expert_anchor_safe
            if expert_anchor_batch is not None and expert_anchor_safe is not None
            else None
        ),
        expert_replace_worst=replace_expert_worst,
    )
    n_steps = max(1, int(train_config.get("diffusion_k_steps", 4)))
    t_min, t_max = (float(x) for x in train_config.get("diffusion_t_range", (0.05, 0.95)))
    total_loss = 0.0
    if expert_anchor_batch is not None and expert_anchor_safe is not None:
        target_trajectories, target_weights = _inject_expert_anchor_batch(
            target_trajectories,
            target_weights,
            expert_anchor_batch,
            expert_anchor_safe,
            group_size=int(rollout_config.n_trajectories),
            expert_weight=expert_anchor_weight,
            replace_worst=replace_expert_worst,
        )
    bc_weight = float(train_config.get("bc_weight", 0.0))
    if bc_weight > 0.0 and target_weights.numel() > 0:
        # HDP's BC term is a retention prior against exploration collapse.  In
        # the original DP path the deterministic zero-noise sample is the
        # frozen behavior/reference trajectory, so increasing its weight is
        # the exact same denoising regression without a second graph shape.
        target_weights = target_weights.clone()
        effective_group_size = int(target_weights.numel()) // len(scene_paths)
        target_weights[::effective_group_size] = (
            target_weights[::effective_group_size] + bc_weight
        )
        target_loss_horizons[::effective_group_size] = T
        result["bc_weight"] = bc_weight

    result["candidate_loss_horizon_steps"] = float(candidate_loss_horizon)
    result["full_loss_horizon_target_fraction"] = float(
        (target_loss_horizons == T).float().mean().item()
    )

    # Positive-only AWR intentionally gives most K members zero weight.  A
    # decoder forward for those targets is mathematically dead work: the DiT
    # has no cross-sample state, and their losses are multiplied by zero.
    # Compact only the model input while retaining the original B*K
    # denominator below.  Full-sized noise/t tensors are still sampled before
    # indexing so the active targets receive the same RNG draws as the
    # uncompressed objective.
    full_target_count = int(target_trajectories.shape[0])
    loss_data = data
    loss_encoding = rollout.scene_encoding
    model_indices: torch.Tensor | None = None
    active_mask = target_weights > 0
    active_count = int(active_mask.sum().item())
    if 0 < active_count < full_target_count:
        active_indices = torch.nonzero(active_mask, as_tuple=False).flatten()
        # Static-shape Inductor is substantially faster and more reliable than
        # dynamic compilation under DDP.  Round the sparse active set up to a
        # small fixed bucket with zero-weight duplicates.  The extra rows have
        # exactly zero contribution, while the real targets keep their
        # original ordering, noise, diffusion time, weight and B*K denominator.
        # A second bucket is compiled only for an unusually dense batch.
        bucket_size = max(0, int(train_config.get("active_target_bucket_size", 0)))
        model_indices = active_indices
        if bucket_size > 0:
            model_target_count = min(
                full_target_count,
                int(math.ceil(active_count / bucket_size) * bucket_size),
            )
            pad_count = model_target_count - active_count
            if pad_count > 0:
                model_indices = torch.cat(
                    [active_indices, active_indices[:1].expand(pad_count)], dim=0
                )
        effective_group_size = full_target_count // len(scene_paths)
        if effective_group_size * len(scene_paths) != full_target_count:
            raise ValueError(
                "AWR target count is not divisible by scene batch: "
                f"targets={full_target_count}, scenes={len(scene_paths)}"
            )
        target_scene_indices = torch.div(
            model_indices, effective_group_size, rounding_mode="floor"
        )
        batch_size = len(scene_paths)
        loss_data = {
            key: value.index_select(0, target_scene_indices)
            if isinstance(value, torch.Tensor)
            and value.dim() > 0
            and value.shape[0] == batch_size
            else value
            for key, value in data.items()
        }
        if (
            isinstance(loss_encoding, torch.Tensor)
            and loss_encoding.shape[0] == batch_size
        ):
            loss_encoding = loss_encoding.index_select(0, target_scene_indices)
        target_trajectories = target_trajectories.index_select(0, model_indices)
        target_loss_horizons = target_loss_horizons.index_select(
            0, model_indices
        )
        active_weights = target_weights.index_select(0, active_indices)
        if model_indices.numel() > active_indices.numel():
            active_weights = torch.cat(
                [
                    active_weights,
                    torch.zeros(
                        model_indices.numel() - active_indices.numel(),
                        device=active_weights.device,
                        dtype=active_weights.dtype,
                    ),
                ],
                dim=0,
            )
        target_weights = active_weights
    result["active_target_count"] = float(active_count)
    result["full_target_count"] = float(full_target_count)
    result["active_target_fraction"] = float(
        active_count / max(1, full_target_count)
    )
    result["computed_target_count"] = float(target_trajectories.shape[0])
    for step_index in range(n_steps):
        # HDP samples each replay action with its own diffusion perturbation
        # and time.  Sharing one noise/t across a whole reward group is still
        # an unbiased Monte-Carlo estimate, but creates an avoidable strongly
        # correlated gradient and is not the released implementation.
        target_count = int(target_trajectories.shape[0])
        full_noise = torch.randn(full_target_count, P, T, 4, device=device)
        noise = (
            full_noise.index_select(0, model_indices)
            if model_indices is not None
            else full_noise
        )
        t_schedule = train_config.get("diffusion_t_schedule")
        if isinstance(t_schedule, (list, tuple)) and len(t_schedule) >= n_steps:
            # HDP's five-step DDIM/GRPO chain is represented by the same
            # descending log-SNR points used by the original DP solver.  A
            # fixed schedule trains every denoising decision instead of
            # repeatedly perturbing only the final low-noise output.
            t_value = float(t_schedule[step_index])
            full_t = torch.full(
                (full_target_count,), min(max(t_value, t_min), t_max), device=device
            )
        else:
            full_t = (
                torch.rand(full_target_count, device=device) * (t_max - t_min)
                + t_min
            )
        t = (
            full_t.index_select(0, model_indices)
            if model_indices is not None
            else full_t
        )
        per_trajectory_loss = compute_batched_trajectory_losses(
            # Keep the DDP wrapper on the gradient path.  Calling
            # ``model.module`` here silently bypasses DDP's gradient
            # all-reduce, leaving every rank with a different policy while
            # rank 0 saves only its local update.
            model,
            loss_data,
            target_trajectories,
            model_args,
            noise,
            t,
            device,
            neighbor_loss_weight=float(train_config.get("neighbor_loss_weight", 0.0)),
            cached_encoding=loss_encoding,
            amp_dtype=amp_dtype,
            hdp_hybrid_loss_weight=float(train_config.get("hdp_hybrid_loss_weight", 0.0)),
            hdp_hybrid_waypoint_weight=float(train_config.get("hdp_hybrid_waypoint_weight", 0.1)),
            hdp_hybrid_window=int(train_config.get("hdp_hybrid_window", 10)),
            use_prefix_mask=bool(train_config.get("awr_use_prefix_mask", True)),
            awr_loss_type=str(train_config.get("awr_loss_type", "dp_geometry")),
            ego_loss_horizons=target_loss_horizons,
        )
        local_loss_finite = bool(torch.isfinite(per_trajectory_loss).all())
        global_loss_finite = local_loss_finite
        if dist.is_available() and dist.is_initialized():
            finite_flag = torch.tensor(
                [1 if local_loss_finite else 0],
                dtype=torch.int32,
                device=device,
            )
            dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
            global_loss_finite = bool(finite_flag.item())
        if not global_loss_finite:
            optimizer.zero_grad(set_to_none=True)
            result["skipped"] = 1.0
            result["skip_reason"] = (
                "nonfinite_diffusion_loss"
                if not local_loss_finite
                else "peer_nonfinite_diffusion_loss"
            )
            return result, rollout
        weights = target_weights.to(device=device, dtype=per_trajectory_loss.dtype)
        if str(train_config.get("awr_loss_reduction", "weighted_mean")).lower() in {
            "hdp_mean",
            "mean",
        }:
            # This is the reduction used by the released HDP code:
            # mean(exp(group-normalised reward) * per-sample diffusion loss).
            # Keep it opt-in because the original DP path normally normalises
            # by the finite candidate weight sum to make scenes with different
            # invalid-candidate counts comparable.
            weighted_loss = (per_trajectory_loss * weights).sum() / max(
                1, full_target_count
            )
        else:
            weighted_loss = (per_trajectory_loss * weights).sum() / weights.sum().clamp_min(1e-6)
        # HDP discounts denoising decisions from the fixed reference chain.
        # The original DP runner samples continuous t, so map t≈1 (first,
        # noisiest step) to s=0 and t≈0 (last, cleanest step) to the final
        # denoising index.  Apply this per candidate because the faithful HDP
        # path samples an independent t for every replay action.
        denoise_gamma = float(train_config.get("denoising_discount", 1.0))
        denoise_steps = max(1, int(train_config.get("denoising_steps", 5)))
        denoise_index = torch.round(
            (1.0 - t.detach()) * max(denoise_steps - 1, 0)
        ).clamp_min(0.0)
        denoise_weights = torch.pow(
            torch.as_tensor(denoise_gamma, device=device, dtype=per_trajectory_loss.dtype),
            denoise_index,
        )
        if str(train_config.get("awr_loss_reduction", "weighted_mean")).lower() in {
            "hdp_mean",
            "mean",
        }:
            weighted_loss = (
                per_trajectory_loss * weights * denoise_weights
            ).sum() / max(1, full_target_count)
        else:
            weighted_loss = (per_trajectory_loss * weights * denoise_weights).sum() / weights.sum().clamp_min(1e-6)
        # Scale by the accumulation window so the eventual update is the
        # mean scene loss, not a learning-rate multiplier tied to the window.
        loss_for_backward = weighted_loss / n_steps * float(gradient_scale)
        # The K denoising samples for one scene contribute to one optimizer
        # step.  Accumulate the first K-1 backward passes locally and let DDP
        # all-reduce only the final pass; this is mathematically equivalent
        # to synchronizing every pass but avoids K redundant full-model
        # collectives on the 8-GPU path.
        should_sync = bool(sync_gradients) and step_index == n_steps - 1
        sync_context = (
            model.no_sync()
            if hasattr(model, "no_sync") and not should_sync
            else contextlib.nullcontext()
        )
        with sync_context:
            if grad_scaler is not None:
                grad_scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()
        total_loss += float(weighted_loss.detach().item()) / n_steps

    if not optimizer_step:
        # Keep these gradients for the next scene in the accumulation window.
        # Clipping each scene separately would not be equivalent to a batch.
        result.update(
            {
                "skipped": 0.0,
                "loss": total_loss,
                "grad_norm": 0.0,
                "weight_sum": float(target_weights.sum().item()),
                "optimizer_step": 0.0,
            }
        )
        return result, rollout

    if grad_scaler is not None:
        grad_scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in planner.parameters() if parameter.requires_grad],
        float(train_config.get("grad_clip_norm", 1.0)),
    )
    grad_norm_value = float(
        grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
    )
    # A scene-wise AWR update can become numerically unstable after the policy
    # has drifted away from the behavior model.  Never let a non-finite update
    # contaminate the EMA behavior model or the next scene.
    if not math.isfinite(grad_norm_value):
        optimizer.zero_grad(set_to_none=True)
        if grad_scaler is not None:
            grad_scaler.update()
        result.update(
            {
                "skipped": 1.0,
                "skip_reason": "nonfinite_grad_norm",
                "loss": total_loss,
                "grad_norm": grad_norm_value,
                "weight_sum": float(target_weights.sum().item()),
                "optimizer_step": 1.0,
            }
        )
        return result, rollout
    if grad_scaler is not None:
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        optimizer.step()
    result.update(
        {
            "skipped": 0.0,
            "loss": total_loss,
            "grad_norm": grad_norm_value,
            "weight_sum": float(target_weights.sum().item()),
            "optimizer_step": 1.0,
        }
    )
    return result, rollout


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model_path", type=Path)
    source.add_argument("--weights_zip", type=Path)
    parser.add_argument(
        "--args_path",
        type=Path,
        default=None,
        help="optional model args JSON, useful for an AWR epoch checkpoint",
    )
    parser.add_argument(
        "--use_policy_state",
        action="store_true",
        help="load an AWR checkpoint's live policy state instead of its deployable EMA",
    )
    parser.add_argument(
        "--resume_optimizer_state",
        action="store_true",
        help=(
            "restore AdamW moments/steps from the checkpoint while preserving "
            "this run's configured learning rate; requires --use_policy_state"
        ),
    )
    parser.add_argument("--train_npz_list", type=Path, required=True)
    parser.add_argument(
        "--extra_train_set_list",
        type=Path,
        action="append",
        default=None,
        help=(
            "additional scene list oversampled into the training set; repeatable. "
            "Each list is appended --extra_train_set_repeat times after the fixed "
            "train-selector is drawn, so the selection yardstick stays comparable "
            "across cycles. Incompatible with --shared_replay_* reuse because the "
            "canonical scene order changes"
        ),
    )
    parser.add_argument(
        "--extra_train_set_repeat",
        type=int,
        default=10,
        help=(
            "how many times each --extra_train_set_list is repeated into the "
            "training set (0 disables the oversampling)"
        ),
    )
    parser.add_argument("--valid_npz_list", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/awr_original_dp"))
    parser.add_argument("--exp_name", type=str, default="original_dp_awr_t4")
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="log rank-zero training/evaluation diagnostics to Weights & Biases",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=os.environ.get("WANDB_PROJECT", "original-dp-awr"),
    )
    parser.add_argument("--wandb_entity", type=str, default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_run_id", type=str, default=os.environ.get("WANDB_RUN_ID"))
    parser.add_argument(
        "--wandb_mode",
        choices=["online", "offline", "disabled"],
        default=os.environ.get("WANDB_MODE", "online"),
    )
    parser.add_argument("--wandb_tags", nargs="*", default=None)
    parser.add_argument(
        "--resume_replay_root",
        type=Path,
        default=None,
        help=(
            "reuse an existing rank-local disk replay directory instead of "
            "re-mining epoch 1"
        ),
    )
    parser.add_argument(
        "--resume_mining_cache",
        type=Path,
        default=None,
        help=(
            "reuse committed rank-local replay manifests during a mining-only "
            "epoch and remine only incomplete ranks; intended for crash recovery"
        ),
    )
    parser.add_argument(
        "--compressed_replay_context_root",
        type=Path,
        default=None,
        help=(
            "strict lossless row-addressable sidecar for cached scene encoding, "
            "decoder context and expert anchors; requires --resume_replay_root"
        ),
    )
    parser.add_argument(
        "--inherit_refresh_baseline_root",
        type=Path,
        default=None,
        help=(
            "completed mine-only run whose exact EMA validation/train-selector "
            "baseline is reused after strict checkpoint/config/scene-order "
            "verification"
        ),
    )
    parser.add_argument(
        "--shared_replay_encoding_root",
        type=Path,
        default=None,
        help=(
            "immutable completed replay root whose rank-local scene_encoding.npy "
            "and scene order are reused by rollout/replay; avoids re-encoding "
            "and re-writing the frozen original-DP encoder"
        ),
    )
    parser.add_argument(
        "--shared_replay_context_root",
        type=Path,
        default=None,
        help=(
            "immutable completed replay root whose canonical aligned decoder "
            "context arrays are linked into a new refresh cache"
        ),
    )
    parser.add_argument(
        "--shared_replay_expert_anchor_root",
        type=Path,
        default=None,
        help=(
            "immutable completed replay root whose policy-independent logged "
            "expert trajectories/rewards/hard-safe flags are reused while "
            "mining a new policy refresh"
        ),
    )
    parser.add_argument(
        "--shared_replay_order_epoch",
        type=int,
        default=1,
        help=(
            "shuffle epoch that defines the shared rank-local scene order "
            "(Cycle-1/2 formal caches use 1)"
        ),
    )
    parser.add_argument(
        "--salvage_partial_replay",
        action="store_true",
        help=(
            "publish the fully-written scene_paths prefix of an interrupted "
            "preallocated disk replay cache"
        ),
    )
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=1,
        help="first HDP lifecycle epoch to execute (use 2 when resuming epoch-1 replay)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max_train_scenes", type=int, default=None)
    parser.add_argument("--max_valid_scenes", type=int, default=None)
    parser.add_argument(
        "--train_selector_scenes",
        type=int,
        default=None,
        help="fixed unbiased train-set K=1 EMA selector size",
    )
    parser.add_argument(
        "--expected_train_selector_sha256",
        type=str,
        default=None,
        help="fail before evaluation/training if the ordered train selector differs",
    )
    parser.add_argument(
        "--expected_valid_sha256",
        type=str,
        default=None,
        help="fail before evaluation/training if the ordered validation set differs",
    )
    parser.add_argument(
        "--full_valid_interval",
        type=int,
        default=0,
        help=(
            "run full validation every N epochs (and optional hard validation "
            "when --hard_valid_scenes>0); 0 disables the extra evaluation"
        ),
    )
    parser.add_argument(
        "--hard_valid_scenes",
        type=int,
        default=8192,
        help=(
            "number of baseline-low-reward scenes in the fixed hard validation set; "
            "0 disables separate hard validation"
        ),
    )
    parser.add_argument(
        "--skip_filtered_scenes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="drop sidecar frames with is_skipped=true before sampling/training (default: on)",
    )
    parser.add_argument(
        "--sidecar_root",
        type=Path,
        default=None,
        help="optional sidecar root when <npz>.json is not next to each NPZ",
    )
    parser.add_argument(
        "--skip_filter_manifest",
        type=Path,
        default=None,
        help="record an explicit pre-filter manifest when the input lists were filtered offline",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--K", type=int, default=None, help="override the AWR/eval group size")
    parser.add_argument(
        "--noise_scale_range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="override the initial-DP-noise scale range for rollout groups",
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=None,
        help="override the stochastic original-DP rollout solver steps (HDP uses 5)",
    )
    parser.add_argument(
        "--awr_beta",
        type=float,
        default=None,
        help="override the group-normalised AWR inverse-temperature",
    )
    parser.add_argument(
        "--awr_weight_clip",
        type=float,
        default=None,
        help="override the positive AWR weight clip",
    )
    parser.add_argument(
        "--plannerrft_reference_mode",
        choices=["zero_noise", "stochastic"],
        default=None,
        help="frozen-reference latent used only for guided rollout candidates",
    )
    parser.add_argument(
        "--plannerrft_reference_noise_scale",
        type=float,
        default=None,
        help="latent temperature for a stochastic frozen reference",
    )
    parser.add_argument(
        "--plannerrft_longitudinal_mode",
        choices=["candidate_stretch", "reference_speed_push"],
        default=None,
        help="longitudinal energy used only during PlannerRFT candidate generation",
    )
    parser.add_argument(
        "--plannerrft_lambda_lat",
        type=float,
        default=None,
        help="maximum lateral guidance target in metres for candidate generation",
    )
    parser.add_argument(
        "--deterministic_first",
        action="store_true",
        help="prepend the zero-noise behavior trajectory as candidate 0",
    )
    parser.add_argument(
        "--hdp_trajectory_augmentation",
        action="store_true",
        help="apply HDP's per-trajectory N(0,0.5m) longitudinal/lateral rollout augmentation",
    )
    parser.add_argument("--hdp_trajectory_augmentation_std", type=float, default=None)
    parser.add_argument(
        "--hdp_trajectory_augmentation_ramp_steps", type=int, default=None
    )
    parser.add_argument(
        "--hdp_trajectory_augmentation_heading_mode",
        choices=["preserve", "velocity_offset", "tangent"],
        default=None,
    )
    parser.add_argument(
        "--hdp_trajectory_augmentation_schedule",
        choices=["first_five_epochs", "first_refresh", "every_refresh", "never"],
        default=None,
    )
    parser.add_argument(
        "--positive_advantage_only",
        action="store_true",
        help="weight only sampled candidates that beat deterministic behavior",
    )
    parser.add_argument("--positive_advantage_margin", type=float, default=None)
    parser.add_argument("--behavior_anchor_weight", type=float, default=None)
    parser.add_argument("--unsafe_behavior_anchor_weight", type=float, default=None)
    parser.add_argument(
        "--drop_all_zero_groups",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--diffusion_k_steps", type=int, default=None)
    parser.add_argument("--diffusion_t_range", type=float, nargs=2, default=None)
    parser.add_argument("--neighbor_loss_weight", type=float, default=None)
    parser.add_argument("--hdp_hybrid_loss_weight", type=float, default=None)
    parser.add_argument(
        "--awr_loss_type",
        choices=["dp_geometry", "plain_mse"],
        default=None,
        help="diffusion regression metric used for AWR targets",
    )
    parser.add_argument(
        "--active_target_bucket_size",
        type=int,
        default=None,
        help="round sparse nonzero AWR targets up to this static compile batch size; 0 disables padding",
    )
    parser.add_argument(
        "--trainable_scope",
        choices=["dit", "last_block", "output"],
        default=None,
        help="decoder scope to update; output/last_block are conservative AWR ablations",
    )
    parser.add_argument("--eval_k", type=int, default=None)
    parser.add_argument(
        "--eval_sample_steps",
        type=int,
        default=None,
        help=(
            "diffusion steps used only for checkpoint selection/evaluation; "
            "defaults to the rollout sampler steps"
        ),
    )
    parser.add_argument(
        "--rollback_rejected_epoch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "after train-selector rejection, restore live/EMA to best_train "
            "before the next replay epoch"
        ),
    )
    parser.add_argument("--save_rollouts", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--ema_decay", type=float, default=None)
    parser.add_argument("--bc_weight", type=float, default=None)
    parser.add_argument("--denoising_discount", type=float, default=None)
    parser.add_argument("--denoising_steps", type=int, default=None)
    parser.add_argument("--replay_buffer_size", type=int, default=None)
    parser.add_argument("--replay_updates_per_epoch", type=int, default=None)
    parser.add_argument(
        "--replay_storage",
        choices=["memory", "disk"],
        default=None,
        help=(
            "replay backend; disk is the full-corpus HDP backend and keeps "
            "frozen targets/encodings in NVMe memmaps"
        ),
    )
    parser.add_argument(
        "--replay_sampling",
        choices=["with_replacement", "shuffle", "reward_stratified"],
        default=None,
        help=(
            "sampling rule for disk replay (HDP uses with_replacement; "
            "reward_stratified is an explicit hard-case ablation)"
        ),
    )
    parser.add_argument(
        "--replay_priority_best_reward_lt",
        type=float,
        default=None,
        help="hard-pool threshold used only by reward_stratified replay",
    )
    parser.add_argument(
        "--replay_priority_fraction",
        type=float,
        default=None,
        help="fraction of each reward_stratified replay batch drawn from the hard pool",
    )
    parser.add_argument(
        "--hdp_rollout_interval",
        type=int,
        default=None,
        help=(
            "refresh/collect-only interval for the HDP replay cycle; 0 uses "
            "streaming on-policy AWR so every full-corpus scene is trained"
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_scenes",
        type=int,
        default=None,
        help=(
            "mean this many scene groups before one optimizer step; this is "
            "the HDP-style large-batch stability knob"
        ),
    )
    parser.add_argument(
        "--scene_batch_size",
        type=int,
        default=None,
        help="pack this many scenes into one decoder/reward-regression batch",
    )
    parser.add_argument(
        "--rollout_scene_batch_size",
        type=int,
        default=None,
        help=(
            "pack this many scenes only during collect-only rollout; replay "
            "keeps scene_batch_size so optimizer semantics do not change"
        ),
    )
    parser.add_argument(
        "--benchmark_rollout_batches_per_rank",
        type=int,
        default=None,
        help=(
            "benchmark-only cap on collect-only rollout batches per rank; "
            "0/default preserves the full corpus and formal runs must leave "
            "this disabled"
        ),
    )
    parser.add_argument(
        "--eval_scene_batch_size",
        type=int,
        default=None,
        help=(
            "pack this many scenes during no-gradient evaluation; defaults to "
            "scene_batch_size so existing runs remain unchanged"
        ),
    )
    parser.add_argument(
        "--scene_load_workers",
        type=int,
        default=None,
        help="CPU threads used to parse one packed scene batch",
    )
    parser.add_argument(
        "--eval_scene_load_workers",
        type=int,
        default=None,
        help=(
            "CPU readers used only for full-scene validation; defaults to "
            "scene_load_workers for backward compatibility"
        ),
    )
    parser.add_argument(
        "--eval_prefetch_batches",
        type=int,
        default=None,
        help=(
            "prefetch this many full-scene validation batches on CPU while "
            "the previous batch runs on CUDA; 0 preserves synchronous loading"
        ),
    )
    parser.add_argument("--rollout_scene_load_workers", type=int, default=None)
    parser.add_argument(
        "--rollout_prefetch_batches",
        type=int,
        default=None,
        help=(
            "prefetch this many full-scene rollout batches on CPU while the "
            "previous batch runs on CUDA; 0 preserves synchronous loading"
        ),
    )
    parser.add_argument("--replay_prefetch_batches", type=int, default=None)
    parser.add_argument(
        "--skip_unused_hdp_diagnostics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "during HDP/PDM rollout mining, skip lane-departure diagnostics "
            "when they are not a configured reward/gate; evaluation remains full"
        ),
    )
    parser.add_argument(
        "--ema_per_epoch",
        action="store_true",
        help="keep the behavior/reference policy fixed during an epoch and refresh it once at epoch end",
    )
    parser.add_argument(
        "--ema_commit_live_policy",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "after an epoch-boundary EMA update, copy the accepted EMA back "
            "to live policy and clear optimizer state (explicit conservative "
            "accepted-policy interpolation)"
        ),
    )
    parser.add_argument("--expert_anchor_weight", type=float, default=None)
    parser.add_argument(
        "--awr_candidate_loss_horizon",
        type=int,
        default=None,
        help=(
            "score-supported ego-loss horizon for sampled AWR candidates; "
            "0 keeps the full model horizon, while safe expert and "
            "deterministic retention targets always keep the full horizon"
        ),
    )
    parser.add_argument(
        "--expert_anchor_replace_worst",
        action="store_true",
        help="replace the lowest-weight candidate with a safe logged-expert anchor (keeps K fixed)",
    )
    parser.add_argument(
        "--expert_anchor_active_groups_only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "inject a safe logged-expert anchor only in scenes that already "
            "contain a non-zero AWR candidate target"
        ),
    )
    parser.add_argument(
        "--reward_mode",
        choices=["survival", "gate"],
        default=None,
        help="override reward aggregation for an ablation",
    )
    parser.add_argument("--reward_profile", type=str, default=None)
    parser.add_argument("--reward_horizon_steps", type=int, default=None)
    parser.add_argument("--pdm_comfort_scale", type=float, default=None)
    parser.add_argument("--hdp_risk_weight", type=float, default=None)
    parser.add_argument("--hdp_follow_weight", type=float, default=None)
    parser.add_argument("--hdp_lane_weight", type=float, default=None)
    parser.add_argument("--hdp_risk_use_clearance", action="store_true")
    parser.add_argument(
        "--safe_only",
        action="store_true",
        help=(
            "weight only candidates that pass the configured hard safety gates; "
            "lane departure is not hard unless lane_gate_enabled is true"
        ),
    )
    parser.add_argument(
        "--structured_exploration",
        action="store_true",
        help="add HDP-inspired smooth yield/hold and lateral-offset proposal candidates",
    )
    parser.add_argument(
        "--amp_dtype",
        type=str,
        choices=["auto", "off", "bf16", "fp16"],
        default=None,
        help="override acceleration.amp_dtype (H100 auto-selects bf16)",
    )
    parser.add_argument(
        "--disable_compile",
        action="store_true",
        help="disable torch.compile for a clean eager comparison",
    )
    parser.add_argument(
        "--enable_compile",
        action="store_true",
        help="force torch.compile even when the source config leaves it disabled",
    )
    parser.add_argument("--compile_mode", type=str, default=None)
    parser.add_argument(
        "--compile_encoder",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="compile the scene encoder too (decoder-only is faster to warm up)",
    )
    parser.add_argument(
        "--compile_decoder",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="compile the diffusion decoder/DiT module",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="expect torchrun; multi-GPU mode is also enabled automatically when WORLD_SIZE>1",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.resume_optimizer_state and not args.use_policy_state:
        raise ValueError(
            "--resume_optimizer_state requires --use_policy_state so optimizer "
            "moments remain paired with the live policy parameters"
        )
    if args.inherit_refresh_baseline_root is not None:
        if args.resume_replay_root is None:
            raise ValueError(
                "--inherit_refresh_baseline_root requires --resume_replay_root"
            )
        inherited_root = args.inherit_refresh_baseline_root.expanduser().resolve()
        replay_owner = args.resume_replay_root.expanduser().resolve().parent
        if inherited_root != replay_owner:
            raise ValueError(
                "inherited baseline must belong to the resumed replay cache: "
                f"{inherited_root} != {replay_owner}"
            )
    if (
        args.compressed_replay_context_root is not None
        and args.resume_replay_root is None
    ):
        raise ValueError(
            "--compressed_replay_context_root requires --resume_replay_root"
        )
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size_env > 1
    if args.distributed and not distributed:
        raise RuntimeError("--distributed requires torchrun with WORLD_SIZE>1")
    rank = 0
    world_size = 1
    object_group = None
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed AWR requires CUDA/NCCL")
        # A full-corpus mining rank flushes hundreds of gigabytes of memmaps
        # before the strict replay barrier.  The default process-group
        # timeout is only about ten minutes and can abort healthy, slower
        # ranks during that final NVMe flush.  Keep the timeout configurable,
        # but make the production-safe default long enough for a real cache
        # commit; this does not mask failures because strict manifests and
        # the subsequent all-rank reader validation remain mandatory.
        ddp_timeout_minutes = int(
            os.environ.get("DP_DDP_TIMEOUT_MINUTES", "180")
        )
        if ddp_timeout_minutes <= 0:
            raise ValueError("DP_DDP_TIMEOUT_MINUTES must be positive")
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=ddp_timeout_minutes),
        )
        # ``gather_object`` with a NCCL group stages serialized Python objects
        # on the current CUDA device.  On this PyTorch/CUDA combination its
        # size exchange can occasionally be interpreted as an absurd tensor
        # allocation (>1 EB) even though rows contain only JSON scalars.  Use
        # a CPU Gloo side group for metrics/checkpoint metadata; gradients and
        # model synchronization remain NCCL.
        object_group = dist.new_group(backend="gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    elif args.device.startswith("cuda"):
        device = torch.device(args.device)
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"requested {args.device}, but CUDA is unavailable; pass --device cpu for a smoke test"
            )
        if device.index is None:
            device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
    else:
        device = torch.device(args.device)
    is_main = rank == 0

    runtime_acceleration = _configure_cuda_runtime(device)

    # Keep scene ordering identical across ranks, but give each rank an
    # independent diffusion/noise stream. DDP averages the resulting AWR
    # gradients into one policy update.
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + rank)

    raw_config, rollout_config, reward_config = _load_config(args.config)
    train_config = dict(raw_config.get("training", raw_config))
    acceleration_config = dict(raw_config.get("acceleration", {}))
    plannerrft_config = dict(raw_config.get("plannerrft", {}))
    if args.amp_dtype is not None:
        acceleration_config["amp_dtype"] = args.amp_dtype
    if args.disable_compile:
        acceleration_config["compile"] = False
    if args.enable_compile:
        acceleration_config["compile"] = True
    if args.compile_mode is not None:
        acceleration_config["mode"] = args.compile_mode
    if args.compile_encoder is not None:
        acceleration_config["encoder"] = bool(args.compile_encoder)
    if args.compile_decoder is not None:
        acceleration_config["decoder"] = bool(args.compile_decoder)
    if args.epochs is not None:
        train_config["train_epochs"] = args.epochs
    # Keep the canonical effective-config artifact aligned with the values
    # that `_choose_paths` and the rollout writer actually consume below.
    # These limits used to be applied only at the call site, leaving stale
    # JSON defaults under `training` even though the separate validation
    # contract recorded the correct CLI value.  That was harmless to
    # execution but misleading in W&B/config audits.
    if args.max_train_scenes is not None:
        if args.max_train_scenes < 0:
            raise ValueError("max_train_scenes must be non-negative")
        train_config["max_train_scenes"] = args.max_train_scenes
    if args.max_valid_scenes is not None:
        if args.max_valid_scenes < 0:
            raise ValueError("max_valid_scenes must be non-negative")
        train_config["max_valid_scenes"] = args.max_valid_scenes
    if args.save_rollouts is not None:
        if args.save_rollouts < 0:
            raise ValueError("save_rollouts must be non-negative")
        train_config["save_rollouts"] = args.save_rollouts
    if args.train_selector_scenes is not None:
        if args.train_selector_scenes < 0:
            raise ValueError("train_selector_scenes must be non-negative")
        train_config["train_selector_scenes"] = args.train_selector_scenes
    if args.eval_k is not None:
        if args.eval_k <= 0:
            raise ValueError("eval_k must be positive")
        train_config["eval_k"] = args.eval_k
    if args.eval_sample_steps is not None:
        if args.eval_sample_steps <= 0:
            raise ValueError("eval_sample_steps must be positive")
        train_config["eval_sample_steps"] = args.eval_sample_steps
    if args.rollback_rejected_epoch is not None:
        train_config["rollback_rejected_epoch"] = bool(
            args.rollback_rejected_epoch
        )
    if args.diffusion_k_steps is not None:
        train_config["diffusion_k_steps"] = args.diffusion_k_steps
    if args.diffusion_t_range is not None:
        train_config["diffusion_t_range"] = list(args.diffusion_t_range)
    if args.neighbor_loss_weight is not None:
        train_config["neighbor_loss_weight"] = args.neighbor_loss_weight
    if args.hdp_hybrid_loss_weight is not None:
        train_config["hdp_hybrid_loss_weight"] = args.hdp_hybrid_loss_weight
    if args.awr_loss_type is not None:
        train_config["awr_loss_type"] = args.awr_loss_type
    if args.active_target_bucket_size is not None:
        if args.active_target_bucket_size < 0:
            raise ValueError("active_target_bucket_size must be non-negative")
        train_config["active_target_bucket_size"] = args.active_target_bucket_size
    if args.trainable_scope is not None:
        train_config["trainable_scope"] = args.trainable_scope
    if args.learning_rate is not None:
        train_config["learning_rate"] = args.learning_rate
    if args.ema_decay is not None:
        train_config["ema_decay"] = args.ema_decay
    if args.bc_weight is not None:
        train_config["bc_weight"] = args.bc_weight
    if args.denoising_discount is not None:
        train_config["denoising_discount"] = args.denoising_discount
    if args.denoising_steps is not None:
        train_config["denoising_steps"] = args.denoising_steps
    if args.replay_buffer_size is not None:
        train_config["replay_buffer_size"] = args.replay_buffer_size
    if args.replay_updates_per_epoch is not None:
        train_config["replay_updates_per_epoch"] = args.replay_updates_per_epoch
    if args.replay_storage is not None:
        train_config["replay_storage"] = args.replay_storage
    if args.replay_sampling is not None:
        train_config["replay_sampling"] = args.replay_sampling
    if args.replay_priority_best_reward_lt is not None:
        train_config["replay_priority_best_reward_lt"] = (
            args.replay_priority_best_reward_lt
        )
    if args.replay_priority_fraction is not None:
        train_config["replay_priority_fraction"] = args.replay_priority_fraction
    if args.hdp_rollout_interval is not None:
        train_config["hdp_rollout_interval"] = args.hdp_rollout_interval
    if args.gradient_accumulation_scenes is not None:
        train_config["gradient_accumulation_scenes"] = args.gradient_accumulation_scenes
    if args.scene_batch_size is not None:
        train_config["scene_batch_size"] = args.scene_batch_size
    if args.rollout_scene_batch_size is not None:
        train_config["rollout_scene_batch_size"] = args.rollout_scene_batch_size
    if args.benchmark_rollout_batches_per_rank is not None:
        if args.benchmark_rollout_batches_per_rank < 0:
            raise ValueError(
                "benchmark_rollout_batches_per_rank must be non-negative"
            )
        train_config["benchmark_rollout_batches_per_rank"] = int(
            args.benchmark_rollout_batches_per_rank
        )
    if args.eval_scene_batch_size is not None:
        train_config["eval_scene_batch_size"] = args.eval_scene_batch_size
    if args.scene_load_workers is not None:
        train_config["scene_load_workers"] = args.scene_load_workers
    if args.eval_scene_load_workers is not None:
        train_config["eval_scene_load_workers"] = args.eval_scene_load_workers
    if args.eval_prefetch_batches is not None:
        if args.eval_prefetch_batches < 0:
            raise ValueError("eval_prefetch_batches must be non-negative")
        train_config["eval_prefetch_batches"] = args.eval_prefetch_batches
    if args.rollout_scene_load_workers is not None:
        train_config["rollout_scene_load_workers"] = args.rollout_scene_load_workers
    if args.rollout_prefetch_batches is not None:
        if args.rollout_prefetch_batches < 0:
            raise ValueError("rollout_prefetch_batches must be non-negative")
        train_config["rollout_prefetch_batches"] = args.rollout_prefetch_batches
    if args.replay_prefetch_batches is not None:
        train_config["replay_prefetch_batches"] = args.replay_prefetch_batches
    if args.skip_unused_hdp_diagnostics is not None:
        train_config["skip_unused_hdp_diagnostics"] = bool(
            args.skip_unused_hdp_diagnostics
        )
    if args.ema_per_epoch:
        train_config["ema_per_epoch"] = True
    if args.ema_commit_live_policy is not None:
        train_config["ema_commit_live_policy"] = bool(
            args.ema_commit_live_policy
        )
    if args.expert_anchor_weight is not None:
        train_config["expert_anchor_weight"] = args.expert_anchor_weight
    if args.awr_candidate_loss_horizon is not None:
        if args.awr_candidate_loss_horizon < 0:
            raise ValueError("awr_candidate_loss_horizon must be non-negative")
        train_config["awr_candidate_loss_horizon"] = int(
            args.awr_candidate_loss_horizon
        )
    if args.expert_anchor_replace_worst:
        train_config["expert_anchor_replace_worst"] = True
    if args.expert_anchor_active_groups_only is not None:
        train_config["expert_anchor_active_groups_only"] = bool(
            args.expert_anchor_active_groups_only
        )
    if args.reward_mode is not None:
        reward_config.reward_mode = args.reward_mode
    if args.reward_profile is not None:
        reward_config.reward_profile = args.reward_profile
    if args.reward_horizon_steps is not None:
        reward_config.reward_horizon_steps = args.reward_horizon_steps
    if args.pdm_comfort_scale is not None:
        reward_config.pdm_comfort_scale = args.pdm_comfort_scale
    if args.hdp_risk_weight is not None:
        reward_config.hdp_risk_weight = args.hdp_risk_weight
    if args.hdp_follow_weight is not None:
        reward_config.hdp_follow_weight = args.hdp_follow_weight
    if args.hdp_lane_weight is not None:
        reward_config.hdp_lane_weight = args.hdp_lane_weight
    if args.hdp_risk_use_clearance:
        reward_config.hdp_risk_use_clearance = True
    if args.K is not None:
        rollout_config.n_trajectories = args.K
    if args.noise_scale_range is not None:
        rollout_config.noise_scale_range = tuple(float(x) for x in args.noise_scale_range)
    if args.sample_steps is not None:
        rollout_config.sample_steps = args.sample_steps
    if args.awr_beta is not None:
        rollout_config.beta = args.awr_beta
    if args.awr_weight_clip is not None:
        rollout_config.weight_clip = args.awr_weight_clip
    if args.plannerrft_reference_mode is not None:
        rollout_config.plannerrft_reference_mode = args.plannerrft_reference_mode
    if args.plannerrft_reference_noise_scale is not None:
        rollout_config.plannerrft_reference_noise_scale = (
            args.plannerrft_reference_noise_scale
        )
    if args.plannerrft_longitudinal_mode is not None:
        rollout_config.plannerrft_longitudinal_mode = (
            args.plannerrft_longitudinal_mode
        )
    if args.plannerrft_lambda_lat is not None:
        if args.plannerrft_lambda_lat <= 0.0:
            raise ValueError("plannerrft_lambda_lat must be positive")
        rollout_config.plannerrft_lambda_lat = float(args.plannerrft_lambda_lat)
    if args.deterministic_first:
        rollout_config.deterministic_first = True
    if args.hdp_trajectory_augmentation:
        rollout_config.hdp_trajectory_augmentation = True
    if args.hdp_trajectory_augmentation_std is not None:
        rollout_config.hdp_trajectory_augmentation_std = args.hdp_trajectory_augmentation_std
    if args.hdp_trajectory_augmentation_ramp_steps is not None:
        rollout_config.hdp_trajectory_augmentation_ramp_steps = (
            args.hdp_trajectory_augmentation_ramp_steps
        )
    if args.hdp_trajectory_augmentation_heading_mode is not None:
        rollout_config.hdp_trajectory_augmentation_heading_mode = (
            args.hdp_trajectory_augmentation_heading_mode
        )
    if args.hdp_trajectory_augmentation_schedule is not None:
        rollout_config.hdp_trajectory_augmentation_schedule = (
            args.hdp_trajectory_augmentation_schedule
        )
    if args.positive_advantage_only:
        rollout_config.positive_advantage_only = True
    if args.positive_advantage_margin is not None:
        rollout_config.positive_advantage_margin = args.positive_advantage_margin
    if args.behavior_anchor_weight is not None:
        rollout_config.behavior_anchor_weight = args.behavior_anchor_weight
    if args.unsafe_behavior_anchor_weight is not None:
        rollout_config.unsafe_behavior_anchor_weight = (
            args.unsafe_behavior_anchor_weight
        )
    if args.drop_all_zero_groups is not None:
        rollout_config.drop_all_zero_groups = args.drop_all_zero_groups
    if args.safe_only:
        rollout_config.safe_only = True
    if args.structured_exploration:
        rollout_config.structured_exploration = True
    if rollout_config.positive_advantage_only and not rollout_config.deterministic_first:
        raise ValueError(
            "positive_advantage_only requires deterministic_first=true so "
            "candidate 0 is the deploy-time behavior anchor"
        )
    if rollout_config.behavior_anchor_weight < 0.0:
        raise ValueError("behavior_anchor_weight must be non-negative")
    if rollout_config.unsafe_behavior_anchor_weight < 0.0:
        raise ValueError("unsafe_behavior_anchor_weight must be non-negative")
    if rollout_config.hdp_trajectory_augmentation_ramp_steps < 0:
        raise ValueError(
            "hdp_trajectory_augmentation_ramp_steps must be non-negative"
        )
    if rollout_config.hdp_trajectory_augmentation_heading_mode not in {
        "preserve",
        "velocity_offset",
        "tangent",
    }:
        raise ValueError(
            "invalid hdp_trajectory_augmentation_heading_mode: "
            f"{rollout_config.hdp_trajectory_augmentation_heading_mode}"
        )
    if rollout_config.plannerrft_guided_exploration:
        if rollout_config.hdp_trajectory_augmentation:
            raise ValueError(
                "PlannerRFT guided enrichment replaces HDP output-space "
                "augmentation; enabling both would be an unaudited method"
            )
        # Do not reject positive-anchor semantics at process startup.  A
        # replay-only run may intentionally reweight an already immutable
        # PlannerRFT cache whose deterministic candidate zero was preserved.
        # The actual rollout path retains its stricter combination guard, so a
        # future refresh cannot silently change the mined-cache contract.
        if rollout_config.plannerrft_eta_scheme not in {
            "iid_beta",
            "stratified_beta",
        }:
            raise ValueError(
                "plannerrft_eta_scheme must be iid_beta or stratified_beta"
            )
        if rollout_config.plannerrft_reference_mode not in {
            "zero_noise",
            "stochastic",
        }:
            raise ValueError("invalid plannerrft_reference_mode")
        if (
            rollout_config.plannerrft_reference_mode == "stochastic"
            and rollout_config.plannerrft_reference_noise_scale <= 0.0
        ):
            raise ValueError(
                "stochastic PlannerRFT reference requires positive noise scale"
            )
        if rollout_config.plannerrft_longitudinal_mode not in {
            "candidate_stretch",
            "reference_speed_push",
        }:
            raise ValueError("invalid plannerrft_longitudinal_mode")
        if rollout_config.plannerrft_lambda_lat <= 0.0:
            raise ValueError("plannerrft_lambda_lat must be positive")
        if rollout_config.plannerrft_guided_candidates < 1:
            raise ValueError("plannerrft_guided_candidates must be >= 1")
        if rollout_config.plannerrft_guidance_bucket_size < 1:
            raise ValueError("plannerrft_guidance_bucket_size must be >= 1")
        if not 0.0 < rollout_config.plannerrft_trigger_best_reward_lt <= 1.0:
            raise ValueError(
                "plannerrft_trigger_best_reward_lt must lie in (0,1]"
            )
        for required_key in ("reference_model_path", "reference_args_path"):
            if not plannerrft_config.get(required_key):
                raise ValueError(
                    "PlannerRFT exploration requires config section "
                    f"plannerrft.{required_key}"
                )
    save_rollouts = int(
        args.save_rollouts if args.save_rollouts is not None else train_config.get("save_rollouts", 8)
    )
    if is_main:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = args.output_dir / f"{stamp}_{args.exp_name}"
        run_dir.mkdir(parents=True, exist_ok=False)
        _copy_json_config(args.config, run_dir, raw_config)
        # ``awr_config.json`` is the immutable source file.  Keep a second
        # artifact with every command-line override applied; otherwise a
        # full-corpus run can appear to have the small smoke-test batch/K/lr
        # even though the effective process used different values.
        effective_config = dict(raw_config)
        effective_config["acceleration"] = dict(acceleration_config)
        effective_config["training"] = dict(train_config)
        effective_config["awr"] = asdict(rollout_config)
        effective_config["reward"] = asdict(reward_config)
        effective_config["plannerrft"] = dict(plannerrft_config)
        effective_config["validation"] = {
            "monitor_scene_limit": int(
                args.max_valid_scenes
                if args.max_valid_scenes is not None
                else train_config.get("max_valid_scenes", 0)
            ),
            "full_valid_interval": int(args.full_valid_interval),
            "hard_valid_scenes": max(0, int(args.hard_valid_scenes)),
            "hard_selection": "baseline_failure_first_then_low_det_reward",
            "valid_selection": (
                "all_scenes_when_limit_zero"
                if args.max_valid_scenes == 0
                else "configured_monitor_subset"
            ),
            "checkpoint_selection": "fixed_train_selector_ema_mean_det_reward",
            "primary_eval_policy_state": "deployable_ema_state_dict",
            "validation_role": "report_only",
            "parallelism": {
                "world_size": int(world_size),
                "scene_shard": "rank::world_size" if distributed else "single_process",
                "row_reduce": "cpu_gloo_gather_object" if distributed else "local",
            },
        }
        effective_config["continuation"] = {
            "load_live_policy_state": bool(args.use_policy_state),
            "resume_optimizer_state": bool(args.resume_optimizer_state),
            "resume_mining_cache": (
                str(args.resume_mining_cache.expanduser().resolve())
                if args.resume_mining_cache is not None
                else None
            ),
            "optimizer_learning_rate_source": "current_effective_config",
            "inherit_refresh_baseline_root": (
                str(args.inherit_refresh_baseline_root.expanduser().resolve())
                if args.inherit_refresh_baseline_root is not None
                else None
            ),
            "shared_replay_encoding_root": (
                str(args.shared_replay_encoding_root.expanduser().resolve())
                if args.shared_replay_encoding_root is not None
                else None
            ),
            "compressed_replay_context_root": (
                str(
                    args.compressed_replay_context_root
                    .expanduser()
                    .resolve()
                )
                if args.compressed_replay_context_root is not None
                else None
            ),
            "shared_replay_context_root": (
                str(args.shared_replay_context_root.expanduser().resolve())
                if args.shared_replay_context_root is not None
                else None
            ),
            "shared_replay_expert_anchor_root": (
                str(
                    args.shared_replay_expert_anchor_root
                    .expanduser()
                    .resolve()
                )
                if args.shared_replay_expert_anchor_root is not None
                else None
            ),
            "shared_replay_order_epoch": int(args.shared_replay_order_epoch),
        }
        if reward_config.reward_profile == "hdp_pdm":
            # The HDP/PDM adapter uses the released scorer's fixed 5/5/2/2
            # quality composition.  The RewardConfig hdp_* fields belong to
            # the paper's separate risk/follow/lane profile and are inactive
            # here; record the effective objective explicitly for W&B readers.
            effective_config["reward"]["effective_quality_weights"] = {
                "ttc": 5.0,
                "ego_progress": 5.0,
                "lane_keeping": 2.0,
                "comfort": 2.0,
                "normalizer": 14.0,
            }
            effective_config["reward"]["hdp_multi_weight_fields_active"] = False
        _write_json(run_dir / "effective_config.json", effective_config)
        staged_model, staged_args = _prepare_checkpoint(
            args.model_path, args.weights_zip, run_dir, args.args_path
        )
        shutil.copy2(staged_args, run_dir / "model_args.json")
        staged_model_sha256 = _file_sha256(staged_model)
        _write_json(
            run_dir / "provenance.json",
            {
                "model_source": str(args.model_path or args.weights_zip),
                "staged_model": str(staged_model),
                "staged_model_sha256": staged_model_sha256,
                "staged_args": str(staged_args),
                "architecture": "original Diffusion Planner, x_start, 4-channel ego/neighbor state",
                "use_policy_state": bool(args.use_policy_state),
                "resume_optimizer_state": bool(args.resume_optimizer_state),
                "neighbor_future_alignment_offset": get_neighbor_future_offset(),
                "neighbor_future_alignment_env": "DP_NEIGHBOR_FUTURE_OFFSET",
                "x2_legacy_ego_width_m": X2_LEGACY_EGO_WIDTH_M,
                "xx1_legacy_ego_width_m": XX1_LEGACY_EGO_WIDTH_M,
                "train_npz_list": str(args.train_npz_list),
                "extra_train_set_lists": [
                    str(Path(p).expanduser().resolve())
                    for p in (args.extra_train_set_list or [])
                ],
                "extra_train_set_repeat": max(0, int(args.extra_train_set_repeat)),
                "valid_npz_list": str(args.valid_npz_list),
                "skip_filtered_scenes": bool(args.skip_filtered_scenes),
                "sidecar_root": str(args.sidecar_root) if args.sidecar_root else None,
                "skip_filter_manifest": str(args.skip_filter_manifest) if args.skip_filter_manifest else None,
                "full_valid_interval": int(args.full_valid_interval),
                "hard_valid_scenes": max(0, int(args.hard_valid_scenes)),
                "device": str(device),
                "seed": args.seed,
                "distributed": bool(distributed),
                "world_size": world_size,
                "rank": rank,
                "awr_beta": rollout_config.beta,
                "awr_weight_clip": rollout_config.weight_clip,
                "plannerrft_guided_exploration": bool(
                    rollout_config.plannerrft_guided_exploration
                ),
                "plannerrft_reference_model": (
                    str(
                        Path(plannerrft_config["reference_model_path"])
                        .expanduser()
                        .resolve()
                    )
                    if rollout_config.plannerrft_guided_exploration
                    else None
                ),
                "plannerrft_reference_args": (
                    str(
                        Path(plannerrft_config["reference_args_path"])
                        .expanduser()
                        .resolve()
                    )
                    if rollout_config.plannerrft_guided_exploration
                    else None
                ),
                "resume_replay_root": str(args.resume_replay_root)
                if args.resume_replay_root
                else None,
                "resume_mining_cache": (
                    str(args.resume_mining_cache.expanduser().resolve())
                    if args.resume_mining_cache is not None
                    else None
                ),
                "compressed_replay_context_root": (
                    str(
                        args.compressed_replay_context_root
                        .expanduser()
                        .resolve()
                    )
                    if args.compressed_replay_context_root is not None
                    else None
                ),
                "inherit_refresh_baseline_root": (
                    str(args.inherit_refresh_baseline_root.expanduser().resolve())
                    if args.inherit_refresh_baseline_root is not None
                    else None
                ),
                "shared_replay_encoding_root": (
                    str(args.shared_replay_encoding_root.expanduser().resolve())
                    if args.shared_replay_encoding_root is not None
                    else None
                ),
                "shared_replay_context_root": (
                    str(args.shared_replay_context_root.expanduser().resolve())
                    if args.shared_replay_context_root is not None
                    else None
                ),
                "shared_replay_expert_anchor_root": (
                    str(
                        args.shared_replay_expert_anchor_root
                        .expanduser()
                        .resolve()
                    )
                    if args.shared_replay_expert_anchor_root is not None
                    else None
                ),
                "shared_replay_order_epoch": int(args.shared_replay_order_epoch),
                "salvage_partial_replay": bool(args.salvage_partial_replay),
                "start_epoch": int(args.start_epoch),
            },
        )
        run_metadata = [str(run_dir), str(staged_model), str(staged_args)]
    else:
        run_dir = None
        staged_model = None
        staged_args = None
        run_metadata = ["", "", ""]
    if distributed:
        dist.broadcast_object_list(run_metadata, src=0)
        run_dir = Path(run_metadata[0])
        staged_model = Path(run_metadata[1])
        staged_args = Path(run_metadata[2])
        dist.barrier()
    assert run_dir is not None and staged_model is not None and staged_args is not None

    shared_encoding_root = (
        args.shared_replay_encoding_root.expanduser().resolve()
        if args.shared_replay_encoding_root is not None
        else None
    )
    shared_encoding_contract: dict[str, Any] = {
        "enabled": False,
    }
    if shared_encoding_root is not None:
        source_run = shared_encoding_root.parent
        provenance_path = source_run / "provenance.json"
        if not provenance_path.is_file():
            raise RuntimeError(
                "shared replay encoding has no source provenance: "
                f"{provenance_path}"
            )
        source_provenance = json.loads(provenance_path.read_text())
        source_checkpoint = Path(source_provenance["staged_model"]).resolve()
        if not source_checkpoint.is_file():
            raise FileNotFoundError(source_checkpoint)
        source_args = Path(source_provenance["staged_args"]).resolve()
        if not source_args.is_file():
            raise FileNotFoundError(source_args)
        source_args_sha = _file_sha256(source_args)
        current_args_sha = _file_sha256(staged_args)
        if source_args_sha != current_args_sha:
            raise RuntimeError(
                "shared replay encoding was produced with different model args/"
                f"normalization: source={source_args_sha}, current={current_args_sha}"
            )
        if int(source_provenance.get("neighbor_future_alignment_offset", -1)) != int(
            get_neighbor_future_offset()
        ):
            raise RuntimeError(
                "shared encoding neighbor-future contract differs from this run"
            )
        source_encoder_sha = _checkpoint_encoder_sha256(
            source_checkpoint,
            use_policy_state=bool(source_provenance.get("use_policy_state", False)),
        )
        current_encoder_sha = _checkpoint_encoder_sha256(
            staged_model,
            use_policy_state=bool(args.use_policy_state),
        )
        if source_encoder_sha != current_encoder_sha:
            raise RuntimeError(
                "shared replay encoding was produced by a different encoder: "
                f"source={source_encoder_sha}, current={current_encoder_sha}"
            )
        shared_rank_dir = shared_encoding_root / f"rank_{rank:04d}"
        shared_manifest_path = shared_rank_dir / "manifest.json"
        if not shared_manifest_path.is_file():
            raise FileNotFoundError(shared_manifest_path)
        shared_manifest = json.loads(shared_manifest_path.read_text())
        shared_encoding_name = shared_manifest.get("arrays", {}).get(
            "scene_encoding"
        )
        if not shared_encoding_name:
            raise RuntimeError(
                f"shared rank {rank} manifest contains no scene_encoding"
            )
        shared_encoding_path = (
            shared_rank_dir / str(shared_encoding_name)
        ).resolve()
        if not shared_encoding_path.is_file():
            raise FileNotFoundError(shared_encoding_path)
        shared_encoding_contract = {
            "enabled": True,
            "root": str(shared_encoding_root),
            "rank": int(rank),
            "source_checkpoint": str(source_checkpoint),
            "source_args": str(source_args),
            "source_args_sha256": source_args_sha,
            "current_args_sha256": current_args_sha,
            "source_encoder_sha256": source_encoder_sha,
            "current_encoder_sha256": current_encoder_sha,
            "scene_count": int(shared_manifest["scene_count"]),
            "encoding_shape": list(shared_manifest.get("encoding_shape") or []),
            "encoding_path": str(shared_encoding_path),
            "order_epoch": int(args.shared_replay_order_epoch),
        }
        if is_main:
            _write_json(
                run_dir / "shared_replay_encoding_contract.json",
                shared_encoding_contract,
            )

    shared_context_root = (
        args.shared_replay_context_root.expanduser().resolve()
        if args.shared_replay_context_root is not None
        else None
    )
    shared_context_contract: dict[str, Any] = {"enabled": False}
    if shared_context_root is not None:
        if not bool(train_config.get("cache_replay_decoder_context", False)):
            raise ValueError(
                "--shared_replay_context_root requires "
                "training.cache_replay_decoder_context=true"
            )
        shared_context_contract = _shared_decoder_context_contract(
            shared_context_root,
            rank=rank,
            current_args=staged_args,
            order_epoch=int(args.shared_replay_order_epoch),
        )
        if is_main:
            _write_json(
                run_dir / "shared_replay_context_contract.json",
                shared_context_contract,
            )

    shared_expert_anchor_root = (
        args.shared_replay_expert_anchor_root.expanduser().resolve()
        if args.shared_replay_expert_anchor_root is not None
        else None
    )
    shared_expert_anchor_contract: dict[str, Any] = {"enabled": False}
    if shared_expert_anchor_root is not None:
        if float(train_config.get("expert_anchor_weight", 0.0)) <= 0.0:
            raise ValueError(
                "--shared_replay_expert_anchor_root requires "
                "training.expert_anchor_weight > 0"
            )
        shared_expert_anchor_contract = _shared_expert_anchor_contract(
            shared_expert_anchor_root,
            rank=rank,
            current_args=staged_args,
            current_train_npz_list=args.train_npz_list,
            current_skip_filtered_scenes=bool(args.skip_filtered_scenes),
            current_reward_config=reward_config,
            order_epoch=int(args.shared_replay_order_epoch),
        )
        if is_main:
            _write_json(
                run_dir / "shared_replay_expert_anchor_contract.json",
                shared_expert_anchor_contract,
            )

    if is_main:
        print(f"Run directory: {run_dir}")
        print(f"Device: {device}; rank={rank}/{world_size}")
    model, model_args = load_original_dp_checkpoint(
        staged_model, device, staged_args, use_ema=not args.use_policy_state
    )
    exploration_reference_model: nn.Module | None = None
    exploration_reference_model_args = None
    exploration_reference_contract: dict[str, Any] = {"enabled": False}
    if rollout_config.plannerrft_guided_exploration:
        planned_start_epoch = max(1, int(args.start_epoch))
        planned_end_epoch = max(
            0, int(train_config.get("train_epochs", 1))
        )
        planned_rollout_interval = max(
            0, int(train_config.get("hdp_rollout_interval", 0))
        )
        reference_required_on_cuda = bool(
            planned_rollout_interval <= 0
            or any(
                (epoch - 1) % planned_rollout_interval == 0
                for epoch in range(
                    planned_start_epoch, planned_end_epoch + 1
                )
            )
        )
        reference_model_path = (
            Path(plannerrft_config["reference_model_path"])
            .expanduser()
            .resolve()
        )
        reference_args_path = (
            Path(plannerrft_config["reference_args_path"])
            .expanduser()
            .resolve()
        )
        if not reference_model_path.is_file():
            raise FileNotFoundError(reference_model_path)
        if not reference_args_path.is_file():
            raise FileNotFoundError(reference_args_path)
        exploration_reference_contract = {
            "enabled": True,
            "reference_model_path": str(reference_model_path),
            "reference_model_sha256": _file_sha256(reference_model_path),
            "reference_args_path": str(reference_args_path),
            "reference_args_sha256": _file_sha256(reference_args_path),
            "use_ema": True,
            "frozen": True,
            "deployment_guidance": False,
            "trigger_best_reward_lt": float(
                rollout_config.plannerrft_trigger_best_reward_lt
            ),
            "eta_scheme": rollout_config.plannerrft_eta_scheme,
            "reference_mode": rollout_config.plannerrft_reference_mode,
            "reference_noise_scale": float(
                rollout_config.plannerrft_reference_noise_scale
            ),
            "longitudinal_mode": rollout_config.plannerrft_longitudinal_mode,
            "lambda_lat_m": float(rollout_config.plannerrft_lambda_lat),
            "lambda_lon_fraction": float(rollout_config.plannerrft_lambda_lon),
            "guidance_scale": float(rollout_config.plannerrft_guidance_scale),
            "ramp_steps": int(rollout_config.plannerrft_ramp_steps),
            "native_then_guided_topk_union": True,
            "loaded_on_cuda": reference_required_on_cuda,
            "planned_rollout_epochs": [
                epoch
                for epoch in range(
                    planned_start_epoch, planned_end_epoch + 1
                )
                if planned_rollout_interval <= 0
                or (epoch - 1) % planned_rollout_interval == 0
            ],
        }
        if reference_required_on_cuda:
            (
                exploration_reference_model,
                exploration_reference_model_args,
            ) = load_original_dp_checkpoint(
                reference_model_path,
                device,
                reference_args_path,
                use_ema=True,
            )
            if (
                int(exploration_reference_model_args.future_len)
                != int(model_args.future_len)
                or int(
                    exploration_reference_model_args.predicted_neighbor_num
                )
                != int(model_args.predicted_neighbor_num)
            ):
                raise RuntimeError(
                    "PlannerRFT frozen-reference architecture differs from "
                    "policy: reference future/neighbors="
                    f"{exploration_reference_model_args.future_len}/"
                    f"{exploration_reference_model_args.predicted_neighbor_num}, "
                    f"policy={model_args.future_len}/"
                    f"{model_args.predicted_neighbor_num}"
                )
            for parameter in exploration_reference_model.parameters():
                parameter.requires_grad_(False)
            exploration_reference_model.eval()
        else:
            exploration_reference_contract["cuda_skip_reason"] = (
                "replay-only epoch range has no rollout refresh"
            )
        if is_main:
            _write_json(
                run_dir / "plannerrft_reference_contract.json",
                exploration_reference_contract,
            )
    initial_checkpoint_payload_request = (
        "model" if args.use_policy_state else "ema_state_dict_if_available"
    )
    trainable_scope = str(train_config.get("trainable_scope", "dit"))
    trainable = _configure_trainable_decoder(model, trainable_scope)
    behavior_model = copy.deepcopy(model).to(device)
    restored_behavior_ema = False
    if args.use_policy_state:
        restored_behavior_ema = _restore_behavior_ema(behavior_model, staged_model)
    for parameter in behavior_model.parameters():
        parameter.requires_grad_(False)
    behavior_model.eval()
    requested_amp = str(acceleration_config.get("amp_dtype", train_config.get("amp_dtype", "auto")))
    amp_dtype = _resolve_amp_dtype(requested_amp, device)
    # Use the same autocast mode for frozen behavior rollouts and for the
    # baseline/post-training evaluator.  Without this, training claimed bf16
    # while the expensive diffusion sampler still ran entirely in fp32.
    rollout_config.inference_amp_dtype = amp_dtype
    # ``effective_config.json`` is created before the model is loaded so the
    # launcher can record all CLI overrides early.  The selected autocast
    # dtype is only knowable after device capability resolution; refresh this
    # field here so the audit artifact describes the actual rollout numerics,
    # not the dataclass default ("off").
    if is_main:
        effective_config["awr"]["inference_amp_dtype"] = amp_dtype
        _write_json(run_dir / "effective_config.json", effective_config)
    if is_main:
        print(
            f"Original DP validated: agents={model_args.agent_num}, predicted_neighbors={model_args.predicted_neighbor_num}, "
            f"future={model_args.future_len}, DPM steps={model.decoder._sample_steps}"
        )
        print(
            "Original-DP first-waypoint safeguard: "
            f"enabled={rollout_config.original_dp_first_waypoint_gate_enabled}; "
            f"speed<{rollout_config.original_dp_first_waypoint_gate_speed_mps:.2f}m/s; "
            f"max_step={rollout_config.original_dp_first_waypoint_gate_max_step_m:.2f}m; "
            f"max_lateral={rollout_config.original_dp_first_waypoint_gate_max_lateral_m:.2f}m; "
            f"max_tangent={rollout_config.original_dp_first_waypoint_gate_max_tangent_deg:.1f}deg; "
            f"max_implied_steer={rollout_config.original_dp_first_waypoint_gate_max_implied_steer_rad:.3f}rad; "
            f"tangent_floor={rollout_config.original_dp_first_waypoint_gate_tangent_min_step_m*1000:.0f}mm"
        )
        print(
            "HDP output-space augmentation low-speed policy: "
            f"skip={rollout_config.hdp_trajectory_augmentation_skip_low_speed}; "
            f"threshold={rollout_config.hdp_trajectory_augmentation_low_speed_threshold:.2f}m/s"
        )
        print(
            f"Trainable decoder parameters: {sum(p.numel() for p in trainable):,}; scope={trainable_scope}; "
            f"amp={amp_dtype}; tf32={runtime_acceleration.get('tf32_matmul', False)}; "
            f"initial_checkpoint_payload={initial_checkpoint_payload_request}; "
            f"behavior_init=copy_of_loaded_checkpoint; "
            f"separate_behavior_ema_restore={restored_behavior_ema}"
        )
    optimizer = _make_optimizer(trainable, train_config, device)
    optimizer_resume_info: dict[str, Any] = {
        "requested": bool(args.resume_optimizer_state),
        "restored": False,
        "source_epoch": None,
        "source_optimizer_class": None,
        "state_parameter_count": 0,
        "effective_learning_rate": float(train_config.get("learning_rate", 1e-5)),
    }
    if args.resume_optimizer_state:
        optimizer_resume_info.update(
            _restore_optimizer_state(
                optimizer,
                staged_model,
                learning_rate=float(train_config.get("learning_rate", 1e-5)),
            )
        )
    if is_main:
        _write_json(run_dir / "optimizer_resume.json", optimizer_resume_info)
        print("Optimizer continuation:", json.dumps(optimizer_resume_info, sort_keys=True))

    # Compile after checkpoint loading and optimizer construction so the
    # checkpoint remains a normal DP state dict and optimizer parameters are
    # the original tensors.  Only tensor-heavy submodules are compiled.
    compile_enabled = bool(acceleration_config.get("compile", True)) and device.type == "cuda"
    compile_config = dict(acceleration_config)
    compile_config["enabled"] = compile_enabled
    compile_policy = _compile_planner_modules(model, compile_config, "policy")
    compile_behavior = _compile_planner_modules(behavior_model, compile_config, "behavior")
    compile_reference = (
        _compile_planner_modules(
            exploration_reference_model,
            compile_config,
            "plannerrft_frozen_reference",
        )
        if exploration_reference_model is not None
        else {"enabled": False}
    )
    if distributed:
        # Compile before wrapping so the checkpoint/state-dict format remains
        # the ordinary DP format. The optimizer still owns the same parameter
        # objects that DDP wraps, so all-reduced gradients update one policy.
        model = DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    acceleration_info: dict[str, Any] = {
        "runtime": runtime_acceleration,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "amp_requested": requested_amp,
        "amp_selected": amp_dtype,
        "compile": {
            "policy": compile_policy,
            "behavior": compile_behavior,
            "plannerrft_frozen_reference": compile_reference,
        },
        "plannerrft_reference": exploration_reference_contract,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "distributed": distributed,
        "rank": rank,
        "world_size": world_size,
    }
    if is_main:
        _write_json(run_dir / "acceleration.json", acceleration_info)
        print("Acceleration:", json.dumps(_json_safe(acceleration_info), sort_keys=True))

    train_limit = int(train_config.get("max_train_scenes", 0))
    valid_limit = int(train_config.get("max_valid_scenes", 0))
    if args.max_train_scenes is not None:
        train_limit = args.max_train_scenes
    if args.max_valid_scenes is not None:
        valid_limit = args.max_valid_scenes
    train_paths = _choose_paths(
        args.train_npz_list,
        train_limit,
        args.seed,
        "train",
        skip_filtered_scenes=bool(args.skip_filtered_scenes),
        sidecar_root=args.sidecar_root,
    )
    valid_paths = _choose_paths(
        args.valid_npz_list,
        valid_limit,
        args.seed + 1,
        "valid",
        skip_filtered_scenes=bool(args.skip_filtered_scenes),
        sidecar_root=args.sidecar_root,
    )
    train_selector_count = max(
        0, int(train_config.get("train_selector_scenes", 0))
    )
    if train_selector_count > 0:
        selector_rng = random.Random(args.seed + 91_337)
        selector_indices = selector_rng.sample(
            range(len(train_paths)), min(train_selector_count, len(train_paths))
        )
        train_selector_paths = [train_paths[index] for index in selector_indices]
    else:
        train_selector_paths = []
    train_selector_sha256 = _validate_expected_paths_sha256(
        "train selector",
        train_selector_paths,
        args.expected_train_selector_sha256,
    )
    # Oversampled extras are appended *after* the selector draw above so the
    # fixed train-selector — the yardstick every cycle's commit decision is
    # measured against — stays byte-identical to earlier cycles.  Growing
    # ``train_paths`` changes the canonical per-rank scene order, which the
    # shared-encoding prefix contract (see the shared_scene_paths check below)
    # rejects outright, so refuse that combination up front instead of failing
    # deep inside epoch 0 with an opaque prefix mismatch.
    extra_train_manifest: list[dict[str, Any]] = []
    extra_train_repeat = max(0, int(args.extra_train_set_repeat))
    extra_train_lists = list(args.extra_train_set_list or [])
    if extra_train_lists and extra_train_repeat > 0:
        # Reusing a shared cache is correct only when that cache was mined with
        # exactly the same oversampling — same lists, same repeat — because the
        # canonical scene order has to match the cache's prefix contract.  Check
        # the cache's own provenance rather than refusing outright: forcing a
        # fresh full-corpus mine every cycle would roughly double the campaign's
        # wall clock for no benefit.
        resolved_extras = [
            str(Path(p).expanduser().resolve()) for p in extra_train_lists
        ]
        for shared_flag, shared_value in (
            ("--shared_replay_encoding_root", args.shared_replay_encoding_root),
            ("--shared_replay_context_root", args.shared_replay_context_root),
            (
                "--shared_replay_expert_anchor_root",
                args.shared_replay_expert_anchor_root,
            ),
        ):
            if shared_value is None:
                continue
            provenance_path = (
                Path(shared_value).expanduser().resolve().parent / "provenance.json"
            )
            if not provenance_path.is_file():
                raise ValueError(
                    f"{shared_flag} points at a cache without provenance.json "
                    f"({provenance_path}); cannot prove it was mined with the "
                    "same --extra_train_set_list oversampling, so refusing to "
                    "reuse it. Mine a fresh cache instead."
                )
            provenance = json.loads(provenance_path.read_text())
            cached_lists = [
                str(Path(p).expanduser().resolve())
                for p in (provenance.get("extra_train_set_lists") or [])
            ]
            cached_repeat = int(provenance.get("extra_train_set_repeat", 0) or 0)
            if cached_lists != resolved_extras or cached_repeat != extra_train_repeat:
                raise ValueError(
                    f"{shared_flag} was mined with a different training corpus: "
                    f"cache has lists={cached_lists} repeat={cached_repeat}, this "
                    f"run has lists={resolved_extras} repeat={extra_train_repeat}. "
                    "The canonical scene order would differ and the cache's prefix "
                    "contract would reject it. Mine a fresh cache instead."
                )
        for extra_list in extra_train_lists:
            extra_paths = _choose_paths(
                extra_list,
                0,
                args.seed,
                f"extra_train:{Path(extra_list).name}",
                skip_filtered_scenes=bool(args.skip_filtered_scenes),
                sidecar_root=args.sidecar_root,
            )
            train_paths.extend(extra_paths * extra_train_repeat)
            extra_train_manifest.append(
                {
                    "list": str(Path(extra_list).expanduser().resolve()),
                    "scenes": len(extra_paths),
                    "repeat": extra_train_repeat,
                    "appended": len(extra_paths) * extra_train_repeat,
                    "sha256": _ordered_paths_sha256(extra_paths),
                }
            )
        appended_total = sum(row["appended"] for row in extra_train_manifest)
        print(
            f"extra_train: appended {appended_total} entries from "
            f"{len(extra_train_manifest)} list(s) at repeat={extra_train_repeat}; "
            f"train set is now {len(train_paths)} scenes "
            "(train selector unchanged)"
        )
    elif extra_train_lists:
        print(
            f"extra_train: {len(extra_train_lists)} list(s) ignored because "
            "--extra_train_set_repeat is 0"
        )
    valid_sha256 = _validate_expected_paths_sha256(
        "validation",
        valid_paths,
        args.expected_valid_sha256,
    )
    requested_full_valid_interval = max(0, int(args.full_valid_interval))
    full_valid_interval = requested_full_valid_interval
    hard_valid_scene_count = max(0, int(args.hard_valid_scenes))
    full_valid_paths: list[str] = []
    if full_valid_interval > 0:
        full_valid_paths = _choose_paths(
            args.valid_npz_list,
            0,
            args.seed + 1,
            "full_valid",
            skip_filtered_scenes=bool(args.skip_filtered_scenes),
            sidecar_root=args.sidecar_root,
        )
    # ``max_valid_scenes=0`` already makes the ordinary per-epoch evaluation
    # cover the complete validation manifest.  Running ``full_eval`` over the
    # identical ordered list would execute the same K-way inference twice and
    # cannot add selection or reporting evidence.  Deduplicate it here rather
    # than relying on every launch script to notice the overlap.  Keep the
    # separate pass when a hard set must be mined from its rows.
    full_valid_deduplicated = bool(
        full_valid_interval > 0
        and hard_valid_scene_count == 0
        and valid_paths == full_valid_paths
    )
    if full_valid_deduplicated:
        full_valid_interval = 0
        full_valid_paths = []
    if is_main:
        _write_json(
            run_dir / "scene_selection.json",
            {
                "train": _scene_selection_manifest(train_paths),
                "valid": _scene_selection_manifest(valid_paths),
                "train_selector": _scene_selection_manifest(train_selector_paths)
                if train_selector_paths
                else None,
                "train_selector_count": len(train_selector_paths),
                "train_selector_sha256_newline_joined_paths": (
                    train_selector_sha256
                    if train_selector_paths
                    else None
                ),
                "valid_count": len(valid_paths),
                "valid_sha256_newline_joined_paths": valid_sha256,
                "expected_train_selector_sha256": args.expected_train_selector_sha256,
                "expected_valid_sha256": args.expected_valid_sha256,
                "full_valid": _scene_selection_manifest(full_valid_paths)
                if full_valid_paths
                else None,
                "full_valid_interval_requested": requested_full_valid_interval,
                "full_valid_interval": full_valid_interval,
                "full_valid_deduplicated_into_eval": full_valid_deduplicated,
                "hard_valid_scenes": hard_valid_scene_count,
                "skip_filtered_scenes": bool(args.skip_filtered_scenes),
                "sidecar_root": str(args.sidecar_root) if args.sidecar_root else None,
                "skip_filter_manifest": str(args.skip_filter_manifest) if args.skip_filter_manifest else None,
            },
        )

    wandb_run = None
    if is_main:
        wandb_run = _init_wandb(args, run_dir, effective_config)
        if wandb_run is not None:
            wandb_run.summary["system/optimizer_state_restore_requested"] = int(
                bool(optimizer_resume_info["requested"])
            )
            wandb_run.summary["system/optimizer_state_restored"] = int(
                bool(optimizer_resume_info["restored"])
            )
            wandb_run.summary["system/optimizer_state_parameter_count"] = int(
                optimizer_resume_info["state_parameter_count"]
            )

    if compile_enabled and valid_paths:
        # Compile the high-frequency native graph before the small validation
        # and K=1 selector shapes can exhaust static recompile slots.  Bound
        # the warmup by the replay batch as well as the rollout batch so a
        # large mining-only pack cannot create an avoidable startup OOM.
        warmup_scene_count = max(
            1,
            min(
                len(train_paths),
                int(
                    train_config.get(
                        "rollout_scene_batch_size",
                        train_config.get("scene_batch_size", 1),
                    )
                ),
                int(train_config.get("scene_batch_size", 1)),
            ),
        )
        warmup_scene_paths: str | list[str] = (
            train_paths[:warmup_scene_count]
            if train_paths
            else valid_paths[0]
        )
        warmup = _warmup_compiled_models(
            model,
            behavior_model,
            model_args,
            warmup_scene_paths,
            rollout_config,
            reward_config,
            device,
            exploration_reference_model=exploration_reference_model,
            exploration_reference_model_args=exploration_reference_model_args,
        )
        if is_main:
            acceleration_info["compile_warmup"] = warmup
            _write_json(run_dir / "acceleration.json", acceleration_info)
            print("Compile warmup:", json.dumps(_json_safe(warmup), sort_keys=True))

    eval_deterministic_first = bool(
        train_config.get("eval_deterministic_first", True)
    )
    # Mining benefits from the cheaper PlannerRFT five-step sampler, while a
    # replacement checkpoint must be selected with the exact solver used by
    # the deployed Original-DP policy.  Keep those protocols independently
    # configurable: changing the rollout sampler must not silently change the
    # meaning of the fixed train selector or full validation.
    eval_sample_steps = int(
        train_config.get("eval_sample_steps", rollout_config.sample_steps)
    )
    if eval_sample_steps <= 0:
        raise ValueError("training.eval_sample_steps must be positive")
    eval_rollout_config = AWRRolloutConfig(
        n_trajectories=int(train_config.get("eval_k", rollout_config.n_trajectories)),
        sample_steps=eval_sample_steps,
        noise_scale_range=rollout_config.noise_scale_range,
        beta=rollout_config.beta,
        weight_clip=rollout_config.weight_clip,
        normalize_weights=rollout_config.normalize_weights,
        min_group_std=rollout_config.min_group_std,
        safe_only=rollout_config.safe_only,
        structured_exploration=rollout_config.structured_exploration,
        # Evaluation candidate 0 is always the deploy-time zero-noise DP
        # policy.  Training may use the faithful HDP all-stochastic group,
        # but logging a random candidate as "det" would make baseline/AWR
        # comparisons irreproducible.
        deterministic_first=eval_deterministic_first
        or rollout_config.deterministic_first,
        # Augmentation is a training-rollout exploration device in HDP, not
        # part of the deployed/evaluation policy.  Keeping it off here makes
        # baseline and post-training metrics directly comparable.
        hdp_trajectory_augmentation=False,
        hdp_trajectory_augmentation_std=rollout_config.hdp_trajectory_augmentation_std,
        hdp_trajectory_augmentation_schedule="never",
        positive_advantage_only=rollout_config.positive_advantage_only,
        positive_advantage_margin=rollout_config.positive_advantage_margin,
        behavior_anchor_weight=rollout_config.behavior_anchor_weight,
        unsafe_behavior_anchor_weight=rollout_config.unsafe_behavior_anchor_weight,
        drop_all_zero_groups=rollout_config.drop_all_zero_groups,
        inference_amp_dtype=rollout_config.inference_amp_dtype,
    )
    # Evaluation is inference-only and can usually use the larger rollout
    # packing budget. Keep the default tied to the historical training batch
    # so this optimization is opt-in and prior experiment numerics are
    # unchanged unless ``--eval_scene_batch_size`` is supplied.
    eval_scene_batch_size = max(
        1,
        int(
            train_config.get(
                "eval_scene_batch_size",
                train_config.get("scene_batch_size", 1),
            )
        ),
    )
    eval_scene_load_workers = _effective_eval_scene_load_workers(train_config)
    eval_prefetch_batches = max(
        0, int(train_config.get("eval_prefetch_batches", 0))
    )
    eval_dir = run_dir / "eval_epoch_000"
    inherited_baseline: tuple[
        dict[str, Any],
        list[dict[str, Any]],
        dict[str, Any],
        list[dict[str, Any]],
    ] | None = None
    if args.inherit_refresh_baseline_root is not None:
        inherited_error: list[str | None] = [None]
        if is_main:
            try:
                inherited_baseline = _load_inherited_refresh_baseline(
                    args.inherit_refresh_baseline_root,
                    current_checkpoint=staged_model,
                    current_args=staged_args,
                    valid_paths=valid_paths,
                    train_selector_paths=train_selector_paths,
                    rollout_config=rollout_config,
                    reward_config=reward_config,
                    eval_k=int(
                        train_config.get(
                            "eval_k", rollout_config.n_trajectories
                        )
                    ),
                    eval_sample_steps=eval_sample_steps,
                    verify_checkpoint_hash=True,
                )
            except Exception as error:
                inherited_error[0] = f"{type(error).__name__}: {error}"
        if distributed:
            dist.broadcast_object_list(
                inherited_error,
                src=0,
                group=object_group,
            )
        if inherited_error[0] is not None:
            raise RuntimeError(
                "inherited refresh baseline verification failed: "
                f"{inherited_error[0]}"
            )
        if not is_main:
            inherited_baseline = _load_inherited_refresh_baseline(
                args.inherit_refresh_baseline_root,
                current_checkpoint=staged_model,
                current_args=staged_args,
                valid_paths=valid_paths,
                train_selector_paths=train_selector_paths,
                rollout_config=rollout_config,
                reward_config=reward_config,
                eval_k=int(
                    train_config.get("eval_k", rollout_config.n_trajectories)
                ),
                eval_sample_steps=eval_sample_steps,
                verify_checkpoint_hash=False,
            )
        assert inherited_baseline is not None
        (
            base_summary,
            base_rows,
            inherited_selector_summary,
            inherited_selector_rows,
        ) = inherited_baseline
        if is_main:
            print(
                "Reusing hash/config/path-verified refresh baseline from "
                f"{args.inherit_refresh_baseline_root.expanduser().resolve()}"
            )
            inherited_contract = {
                "verified": True,
                "source_run": str(
                    args.inherit_refresh_baseline_root.expanduser().resolve()
                ),
                "checkpoint_sha256": staged_model_sha256,
                "model_args_sha256": _file_sha256(staged_args),
                "neighbor_future_alignment_offset": get_neighbor_future_offset(),
                "x2_legacy_ego_width_m": X2_LEGACY_EGO_WIDTH_M,
                "xx1_legacy_ego_width_m": XX1_LEGACY_EGO_WIDTH_M,
                "valid_scene_count": len(valid_paths),
                "train_selector_scene_count": len(train_selector_paths),
                "valid_scene_order_verified": True,
                "train_selector_scene_order_verified": True,
                "awr_config_verified": True,
                "reward_config_verified": True,
            }
            _write_json(
                run_dir / "inherited_refresh_baseline.json",
                inherited_contract,
            )
            if wandb_run is not None:
                wandb_run.summary["resume/inherited_refresh_baseline"] = 1
                wandb_run.summary["resume/inherited_refresh_baseline_source"] = (
                    inherited_contract["source_run"]
                )
    else:
        inherited_selector_summary = {}
        inherited_selector_rows = []
        if is_main:
            print(
                "Evaluating the untouched original-DP EMA checkpoint on all "
                "validation GPUs..."
            )
        base_summary, base_rows = _distributed_evaluate_model(
            behavior_model,
            model_args,
            valid_paths,
            eval_rollout_config,
            reward_config,
            device,
            eval_dir / "rollouts",
            epoch=0,
            save_rollouts=save_rollouts,
            scene_batch_size=eval_scene_batch_size,
            scene_load_workers=eval_scene_load_workers,
            scene_prefetch_batches=eval_prefetch_batches,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            object_group=object_group,
        )
    if is_main:
        _write_json(eval_dir / "summary.json", base_summary)
        _write_json(eval_dir / "scenes.json", base_rows)
        _append_jsonl(run_dir / "metrics.jsonl", {"phase": "eval", "epoch": 0, **base_summary})
        _wandb_log_summary(
            wandb_run,
            "eval",
            base_summary,
            epoch=0,
            rows=base_rows,
        )
        if wandb_run is not None:
            for metric_path, value in _wandb_summary_payload(
                "baseline", base_summary
            ).items():
                wandb_run.summary[metric_path] = float(value)
        print("Baseline:", json.dumps(_json_safe(base_summary), sort_keys=True))
    if distributed:
        dist.barrier()
    # Checkpoint selection uses a fixed, unbiased train-set monitor and the
    # deployable EMA state.  Candidate-0 reward from a freshly sampled replay
    # group is not a valid selector: it changes with rollout RNG and was fully
    # stochastic in the original HDP-style run.
    train_selector_config = copy.copy(eval_rollout_config)
    train_selector_config.n_trajectories = 1
    train_selector_base_summary: dict[str, Any] = {}
    train_selector_base_rows: list[dict[str, Any]] = []
    if train_selector_paths:
        selector_dir = run_dir / "train_selector_epoch_000"
        if inherited_baseline is not None:
            train_selector_base_summary = inherited_selector_summary
            train_selector_base_rows = inherited_selector_rows
        else:
            if is_main:
                print(
                    "Evaluating the untouched original-DP EMA on the fixed "
                    f"train selector ({len(train_selector_paths):,} scenes)..."
                )
            train_selector_base_summary, train_selector_base_rows = (
                _distributed_evaluate_model(
                    behavior_model,
                    model_args,
                    train_selector_paths,
                    train_selector_config,
                    reward_config,
                    device,
                    selector_dir / "rollouts",
                    epoch=0,
                    save_rollouts=0,
                    scene_batch_size=eval_scene_batch_size,
                    scene_load_workers=eval_scene_load_workers,
                    scene_prefetch_batches=eval_prefetch_batches,
                    distributed=distributed,
                    rank=rank,
                    world_size=world_size,
                    object_group=object_group,
                )
            )
        if is_main:
            _write_json(selector_dir / "summary.json", train_selector_base_summary)
            _write_json(selector_dir / "scenes.json", train_selector_base_rows)
            _append_jsonl(
                run_dir / "metrics.jsonl",
                {
                    "phase": "train_selector_ema",
                    "epoch": 0,
                    **train_selector_base_summary,
                },
            )
            _wandb_log_summary(
                wandb_run,
                "train_selector_ema",
                train_selector_base_summary,
                epoch=0,
                rows=train_selector_base_rows,
            )
        if distributed:
            dist.barrier()
    full_base_summary: dict[str, Any] = {}
    hard_base_summary: dict[str, Any] = {}
    hard_valid_paths: list[str] = []
    if full_valid_interval > 0:
        full_base_dir = run_dir / "full_eval_epoch_000"
        if is_main:
            print(
                "Evaluating the untouched original-DP checkpoint on full valid "
                f"({len(full_valid_paths):,} scenes)"
                + (
                    " for fixed hard-set mining..."
                    if hard_valid_scene_count > 0
                    else "..."
                )
            )
        full_base_summary, full_base_rows = _distributed_evaluate_model(
            behavior_model,
            model_args,
            full_valid_paths,
            eval_rollout_config,
            reward_config,
            device,
            full_base_dir / "rollouts",
            epoch=0,
            save_rollouts=0,
            scene_batch_size=eval_scene_batch_size,
            scene_load_workers=eval_scene_load_workers,
            scene_prefetch_batches=eval_prefetch_batches,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            object_group=object_group,
        )
        if is_main:
            _write_json(full_base_dir / "summary.json", full_base_summary)
            _write_json(full_base_dir / "scenes.json", full_base_rows)
            _append_jsonl(
                run_dir / "metrics.jsonl",
                {"phase": "full_baseline", "epoch": 0, **full_base_summary},
            )
            _wandb_log_summary(
                wandb_run,
                "full_baseline",
                full_base_summary,
                epoch=0,
                rows=full_base_rows,
            )
            if hard_valid_scene_count > 0:
                hard_base_rows = _select_hard_eval_rows(
                    full_base_rows, hard_valid_scene_count
                )
                hard_valid_paths = [str(row["scene_path"]) for row in hard_base_rows]
                hard_base_summary = _summarize_eval_rows(hard_base_rows)
                hard_base_dir = run_dir / "hard_eval_epoch_000"
                _write_json(hard_base_dir / "summary.json", hard_base_summary)
                _write_json(hard_base_dir / "scenes.json", hard_base_rows)
                _write_json(run_dir / "hard_valid_list.json", hard_valid_paths)
                _append_jsonl(
                    run_dir / "metrics.jsonl",
                    {"phase": "hard_baseline", "epoch": 0, **hard_base_summary},
                )
                _wandb_log_summary(
                    wandb_run,
                    "hard_baseline",
                    hard_base_summary,
                    epoch=0,
                    rows=hard_base_rows,
                )
                print(
                    "Hard validation set fixed from baseline: "
                    f"{len(hard_valid_paths):,} scenes"
                )
            else:
                print("Separate hard validation disabled; full validation covers all scenes")
        if distributed:
            hard_paths_payload: list[Any] = [hard_valid_paths if is_main else None]
            dist.broadcast_object_list(
                hard_paths_payload,
                src=0,
                group=object_group,
            )
            hard_valid_paths = list(hard_paths_payload[0] or [])
            dist.barrier()
    if is_main:
        _wandb_log_inherited_refresh_epoch(
            wandb_run,
            args,
            run_dir,
            base_summary,
        )

    epochs = max(0, int(train_config.get("train_epochs", 1)))
    start_epoch = max(1, int(args.start_epoch))
    if start_epoch > epochs:
        raise ValueError(
            f"start_epoch={start_epoch} exceeds configured epochs={epochs}"
        )
    ema_decay = float(train_config.get("ema_decay", 0.995))
    ema_per_epoch = bool(train_config.get("ema_per_epoch", False))
    ema_commit_live_policy = bool(
        train_config.get("ema_commit_live_policy", False)
    )
    if ema_commit_live_policy and not ema_per_epoch:
        raise ValueError(
            "ema_commit_live_policy requires ema_per_epoch=true; an "
            "epoch-boundary accepted-policy commit cannot run after "
            "minibatch EMA updates"
        )
    grad_scaler = None
    if amp_dtype == "fp16" and device.type == "cuda":
        grad_scaler = torch.amp.GradScaler("cuda", enabled=True)
    replay_capacity = max(0, int(train_config.get("replay_buffer_size", 0)))
    replay_updates_per_epoch = int(train_config.get("replay_updates_per_epoch", 0))
    # In the faithful disk backend, ``-1`` means one replay DataLoader-sized
    # batch per rank-local mined-scene batch. This matches the released HDP
    # pattern: mine the buffer once, then sample a fresh group on each train
    # batch from that frozen buffer.
    if replay_updates_per_epoch < -1:
        replay_updates_per_epoch = -1
    hdp_rollout_interval = max(0, int(train_config.get("hdp_rollout_interval", 0)))
    replay_storage = str(train_config.get("replay_storage", "memory")).lower()
    replay_sampling = str(
        train_config.get("replay_sampling", "with_replacement")
    ).lower()
    faithful_disk_replay = replay_storage == "disk" and hdp_rollout_interval > 0
    if faithful_disk_replay and replay_sampling not in {
        "with_replacement",
        "shuffle",
        "reward_stratified",
    }:
        raise ValueError(
            "replay_sampling must be with_replacement, shuffle, or "
            "reward_stratified, got "
            f"{replay_sampling!r}"
        )
    replay_priority_best_reward_lt = float(
        train_config.get("replay_priority_best_reward_lt", 0.925)
    )
    replay_priority_fraction = float(
        train_config.get("replay_priority_fraction", 0.5)
    )
    if replay_sampling == "reward_stratified":
        if not math.isfinite(replay_priority_best_reward_lt):
            raise ValueError("replay_priority_best_reward_lt must be finite")
        if not 0.0 <= replay_priority_fraction <= 1.0:
            raise ValueError(
                "replay_priority_fraction must lie in [0,1], got "
                f"{replay_priority_fraction}"
            )
    resume_mining_root = (
        args.resume_mining_cache.expanduser().resolve()
        if args.resume_mining_cache is not None
        else None
    )
    if resume_mining_root is not None:
        if args.resume_replay_root is not None:
            raise ValueError(
                "--resume_mining_cache cannot be combined with "
                "--resume_replay_root"
            )
        if not faithful_disk_replay:
            raise ValueError(
                "--resume_mining_cache requires the faithful disk replay backend"
            )
        if start_epoch != 1 or epochs != 1:
            raise ValueError(
                "--resume_mining_cache is only valid for one-epoch mining "
                f"recovery (start_epoch={start_epoch}, epochs={epochs})"
            )
    reuse_baseline_after_mining = _is_mine_only_refresh_run(
        start_epoch=start_epoch,
        epochs=epochs,
        rollout_interval=hdp_rollout_interval,
        faithful_disk_replay=faithful_disk_replay,
        resumed_replay=args.resume_replay_root is not None,
    )
    gradient_accumulation_scenes = max(
        1, int(train_config.get("gradient_accumulation_scenes", 1))
    )
    scene_batch_size = max(1, int(train_config.get("scene_batch_size", 1)))
    rollout_scene_batch_size = max(
        1, int(train_config.get("rollout_scene_batch_size", scene_batch_size))
    )
    rollout_scene_load_workers = max(
        1,
        int(
            train_config.get(
                "rollout_scene_load_workers",
                train_config.get("scene_load_workers", 1),
            )
        ),
    )
    rollout_prefetch_batches = max(
        0, int(train_config.get("rollout_prefetch_batches", 0))
    )
    benchmark_rollout_batches_per_rank = max(
        0, int(train_config.get("benchmark_rollout_batches_per_rank", 0))
    )
    if benchmark_rollout_batches_per_rank > 0 and not faithful_disk_replay:
        raise ValueError(
            "benchmark_rollout_batches_per_rank requires collect-only disk replay"
        )
    replay_buffer = deque(maxlen=replay_capacity or None)
    # A resumed run reads the completed source cache for the first replay
    # cycle, then owns a new cache directory for the next periodic refresh.
    # Keeping these roots separate prevents epoch-11 refresh from truncating
    # the cache that was used to resume epoch 2.
    resume_replay_root = (
        args.resume_replay_root.expanduser().resolve()
        if args.resume_replay_root is not None
        else None
    )
    # Mining recovery points at the interrupted cache.  Committed ranks are
    # reused read-only and only incomplete rank directories are rewritten by
    # the strict writer below; ordinary runs still own a fresh local cache.
    replay_root = resume_mining_root or (run_dir / "replay_buffer")
    replay_source_root = resume_replay_root or replay_root
    shared_scene_encoding: np.memmap | None = None
    shared_scene_paths: list[str] = []
    if bool(shared_encoding_contract.get("enabled", False)):
        shared_scene_encoding = np.load(
            Path(shared_encoding_contract["encoding_path"]), mmap_mode="r"
        )
        shared_rank_dir = shared_encoding_root / f"rank_{rank:04d}"
        shared_manifest = json.loads(
            (shared_rank_dir / "manifest.json").read_text()
        )
        with (shared_rank_dir / str(shared_manifest["paths"])).open() as handle:
            shared_scene_paths = [json.loads(line) for line in handle]
        if len(shared_scene_paths) != int(shared_scene_encoding.shape[0]):
            raise RuntimeError(
                "shared scene path/encoding counts differ: "
                f"{len(shared_scene_paths)} != {shared_scene_encoding.shape[0]}"
            )
    shared_decoder_context_paths: dict[str, Path] = {}
    shared_context_scene_paths: list[str] = []
    if bool(shared_context_contract.get("enabled", False)):
        shared_decoder_context_paths = {
            str(key): Path(path).resolve()
            for key, path in shared_context_contract["context_paths"].items()
        }
        with Path(shared_context_contract["paths_path"]).open() as handle:
            shared_context_scene_paths = [
                json.loads(line) for line in handle if line.strip()
            ]
        if len(shared_context_scene_paths) != int(
            shared_context_contract["scene_count"]
        ):
            raise RuntimeError(
                "shared decoder-context path count differs from manifest: "
                f"{len(shared_context_scene_paths)} != "
                f"{shared_context_contract['scene_count']}"
            )
    shared_expert_anchor: dict[str, np.ndarray] = {}
    shared_expert_scene_paths: list[str] = []
    if bool(shared_expert_anchor_contract.get("enabled", False)):
        shared_expert_anchor = {
            str(name): np.load(Path(path), mmap_mode="r")
            for name, path in shared_expert_anchor_contract[
                "array_paths"
            ].items()
        }
        expert_paths_path = Path(
            shared_expert_anchor_contract["paths_path"]
        ).resolve()
        encoding_paths_path = (
            Path(shared_encoding_root)
            / f"rank_{rank:04d}"
            / str(shared_manifest["paths"])
        ).resolve() if shared_scene_paths else None
        if (
            shared_scene_paths
            and encoding_paths_path == expert_paths_path
        ):
            # The formal refresh reuses encoding and expert data from the
            # same generation.  Share the already parsed ~120-MB path list
            # rather than materialising a second Python string copy/rank.
            shared_expert_scene_paths = shared_scene_paths
        else:
            with expert_paths_path.open() as handle:
                shared_expert_scene_paths = [
                    json.loads(line) for line in handle if line.strip()
                ]
        if len(shared_expert_scene_paths) != int(
            shared_expert_anchor_contract["scene_count"]
        ):
            raise RuntimeError(
                "shared expert-anchor path count differs from manifest: "
                f"{len(shared_expert_scene_paths)} != "
                f"{shared_expert_anchor_contract['scene_count']}"
            )
    disk_replay_reader: _DiskReplayReader | None = None
    if args.resume_replay_root is not None and not faithful_disk_replay:
        raise ValueError("--resume_replay_root requires the faithful disk replay backend")
    if args.resume_replay_root is not None and start_epoch <= 1:
        raise ValueError("resumed epoch-1 replay must start at epoch 2 or later")
    if faithful_disk_replay:
        if args.resume_replay_root is not None:
            if not replay_source_root.exists():
                raise FileNotFoundError(replay_source_root)
            _validate_replay_source_contract(
                replay_source_root,
                rollout_config,
                reward_config,
                allow_missing_defaults={
                    "awr": {
                        # These fields were added after the first replay
                        # generations.  Missing means the historical sampler
                        # behavior; accepting only these exact defaults keeps
                        # strict validation for every existing method choice.
                        "hdp_trajectory_augmentation_skip_low_speed": True,
                        "hdp_trajectory_augmentation_low_speed_threshold": 2.0,
                        "original_dp_first_waypoint_gate_enabled": True,
                        "original_dp_first_waypoint_gate_speed_mps": 1.0,
                        "original_dp_first_waypoint_gate_max_step_m": 0.25,
                        "original_dp_first_waypoint_gate_max_lateral_m": 0.20,
                        "original_dp_first_waypoint_gate_max_backward_m": 0.05,
                        "original_dp_first_waypoint_gate_max_tangent_deg": 75.0,
                        "original_dp_stop_turn_speed_mps": 1.0,
                    }
                },
            )
            if args.salvage_partial_replay:
                local_manifest = _salvage_partial_replay_manifest(
                    replay_source_root, rank
                )
            else:
                manifest_path = (
                    replay_source_root / f"rank_{rank:04d}" / "manifest.json"
                )
                if not manifest_path.exists():
                    raise FileNotFoundError(
                        f"{manifest_path}; pass --salvage_partial_replay to recover "
                        "a complete prefix"
                    )
                local_manifest = json.loads(manifest_path.read_text())
            if distributed:
                dist.barrier()
            disk_replay_reader = _DiskReplayReader(
                replay_source_root,
                rank,
                positive_advantage_only=rollout_config.positive_advantage_only,
                deterministic_first=rollout_config.deterministic_first,
                compressed_context_root=args.compressed_replay_context_root,
                compressed_context_workers=max(
                    1, int(train_config.get("scene_load_workers", 1))
                ),
                context_active_weight_only=(
                    args.compressed_replay_context_root is not None
                    and _can_materialize_only_nonzero_replay_groups(train_config)
                ),
            )
            local_count = torch.tensor(
                [len(disk_replay_reader)], device=device, dtype=torch.int64
            )
            if distributed:
                gathered_counts = [torch.zeros_like(local_count) for _ in range(world_size)]
                dist.all_gather(gathered_counts, local_count)
                replay_rank_counts = [int(value.item()) for value in gathered_counts]
            else:
                replay_rank_counts = [int(local_count.item())]
            if is_main:
                _write_json(
                    run_dir / "replay_resume.json",
                    {
                        "source_replay_root": str(replay_source_root),
                        "compressed_replay_context_root": (
                            str(
                                args.compressed_replay_context_root
                                .expanduser()
                                .resolve()
                            )
                            if args.compressed_replay_context_root is not None
                            else None
                        ),
                        "lossless_compressed_context": bool(
                            args.compressed_replay_context_root is not None
                        ),
                        "compressed_context_workers": max(
                            1, int(train_config.get("scene_load_workers", 1))
                        ),
                        "context_active_weight_only": bool(
                            args.compressed_replay_context_root is not None
                            and _can_materialize_only_nonzero_replay_groups(
                                train_config
                            )
                        ),
                        "salvaged_partial_replay": bool(args.salvage_partial_replay),
                        "start_epoch": start_epoch,
                        "rank_scene_counts": replay_rank_counts,
                        "total_scene_groups": int(sum(replay_rank_counts)),
                        "group_size": int(local_manifest["group_size"]),
                        "scene_batch_size": scene_batch_size,
                        "sampling": replay_sampling,
                    },
                )
                print(
                    "Resuming frozen replay: "
                    f"{sum(replay_rank_counts):,} groups across {len(replay_rank_counts)} "
                    f"ranks; counts={replay_rank_counts}"
                )
        else:
            # A mining-recovery run intentionally owns an existing
            # interrupted cache.  Never delete it here; the rank-local
            # completeness check below decides which directories may be
            # reused and the writer only rewrites incomplete ranks.
            if resume_mining_root is None:
                if is_main and replay_root.exists():
                    shutil.rmtree(replay_root)
            if distributed:
                dist.barrier()
    if resume_mining_root is not None and not replay_root.exists():
        raise FileNotFoundError(
            f"mining recovery cache does not exist: {replay_root}"
        )

    # Checkpoint selection is deliberately train-set based.  Validation is a
    # reporting signal only; it must not choose a replacement epoch.  When a
    # faithful resume starts after a completed refresh epoch. The freshly
    # evaluated starting EMA remains eligible under that preceding global epoch.
    best_train_reward = float(
        train_selector_base_summary.get("mean_det_reward", -math.inf)
    )
    baseline_selection_epoch = (
        start_epoch - 1 if args.resume_replay_root is not None else 0
    )
    best_train_epoch = baseline_selection_epoch
    if train_selector_paths and math.isfinite(best_train_reward) and is_main:
        save_checkpoint_pair(
            model,
            behavior_model,
            run_dir / "best_train.pth",
            baseline_selection_epoch,
            {
                "train_selector_ema": train_selector_base_summary,
                "eval": base_summary,
            },
            optimizer=optimizer,
        )
        if wandb_run is not None:
            wandb_run.summary["best/train_selection_epoch"] = int(
                baseline_selection_epoch
            )
            wandb_run.summary["best/train_selection_reward_pct"] = float(
                best_train_reward * 100.0
            )
    # Report-only convenience summaries.  These never participate in
    # checkpoint selection, which remains train-set based.
    best_eval_report: dict[str, tuple[float, int] | None] = {
        "reward_pct": None,
        "ade_m": None,
        "fde_m": None,
        "collision_rate_pct": None,
        "road_border_crossing_rate_pct": None,
        "kinematic_failure_rate_pct": None,
    }
    if (
        not train_selector_paths
        and args.resume_replay_root is not None
        and int(args.start_epoch) > 1
    ):
        source_train_summary = (
            args.resume_replay_root.expanduser().resolve().parent
            / "train_epoch_001"
            / "summary.json"
        )
        try:
            inherited_train = json.loads(source_train_summary.read_text())
            inherited_reward = float(inherited_train.get("mean_det_reward", -math.inf))
            if math.isfinite(inherited_reward):
                best_train_reward = inherited_reward
                best_train_epoch = 1
                if is_main:
                    source_epoch_one = source_train_summary.parent.parent / "epoch_001.pth"
                    if source_epoch_one.exists():
                        shutil.copy2(source_epoch_one, run_dir / "best_train.pth")
                    if wandb_run is not None:
                        wandb_run.summary["best/train_selection_epoch"] = 1
                        wandb_run.summary["best/train_selection_reward_pct"] = float(
                            inherited_reward * 100.0
                        )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            if is_main:
                print(f"Train-set checkpoint selector: epoch-1 seed unavailable: {error}")
    # A continuation may start from an already selected ``best_train.pth``
    # without reusing the previous replay cache (epoch 11 is a fresh HDP
    # refresh).  Seed the selector from that checkpoint so a merely adequate
    # new epoch cannot overwrite the best result from epochs 1--10.
    if (
        not train_selector_paths
        and best_train_reward == -math.inf
        and args.use_policy_state
    ):
        try:
            starting_checkpoint = torch.load(
                staged_model, map_location="cpu", weights_only=False
            )
            starting_train = (
                starting_checkpoint.get("metrics", {}).get("train", {})
                if isinstance(starting_checkpoint, dict)
                else {}
            )
            starting_reward = float(
                starting_train.get("mean_det_reward", -math.inf)
            )
            starting_epoch = int(starting_checkpoint.get("epoch", 0))
            if math.isfinite(starting_reward):
                best_train_reward = starting_reward
                best_train_epoch = starting_epoch
                if is_main:
                    shutil.copy2(staged_model, run_dir / "best_train.pth")
                    if wandb_run is not None:
                        wandb_run.summary["best/train_selection_epoch"] = int(
                            starting_epoch
                        )
                        wandb_run.summary["best/train_selection_reward_pct"] = float(
                            starting_reward * 100.0
                        )
        except (OSError, TypeError, ValueError, KeyError, RuntimeError) as error:
            if is_main:
                print(f"Train-set checkpoint selector: starting checkpoint seed unavailable: {error}")

    # Freeze the incumbent after every supported initialization path.  The
    # final audit can then distinguish a genuinely improved replay epoch from
    # a run that correctly retained its starting checkpoint.
    starting_train_selection_epoch = best_train_epoch
    starting_train_selection_reward = best_train_reward
    accepted_policy_transactions = 0
    rejected_policy_transactions = 0

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        hdp_rollout_phase = (
            hdp_rollout_interval > 0
            and (epoch - 1) % hdp_rollout_interval == 0
        )
        active_scene_batch_size = (
            rollout_scene_batch_size if hdp_rollout_phase else scene_batch_size
        )
        if hdp_rollout_phase:
            # Start a fresh on-policy mining cycle, as in HDP's periodic
            # replay-buffer refresh.  Training epochs below consume these
            # frozen groups and do not re-rollout the scenes.
            replay_buffer.clear()
            order = list(train_paths)
            rollout_order_epoch = (
                int(args.shared_replay_order_epoch)
                if shared_scene_encoding is not None
                else epoch
            )
            random.Random(args.seed + rollout_order_epoch).shuffle(order)
        elif hdp_rollout_interval > 0:
            order = []
        else:
            # Preserve the original online AWR behavior when the HDP replay
            # schedule is not requested by the config.
            order = list(train_paths)
            random.Random(args.seed + epoch).shuffle(order)
        if distributed:
            # DDP requires every rank to execute the same number of backward
            # passes. Pad only the final few slots; every real scene remains
            # present exactly once in the epoch.
            pad_multiple = world_size * active_scene_batch_size
            remainder = len(order) % pad_multiple
            if remainder:
                order.extend(order[: pad_multiple - remainder])
            local_order = order[rank::world_size]
        else:
            local_order = order
        if hdp_rollout_phase and benchmark_rollout_batches_per_rank > 0:
            # This explicit benchmark-only path executes the real mining,
            # scoring and strict writer stack on a canonical rank-local
            # prefix.  Formal configs leave the cap at zero and therefore
            # still require the complete corpus.
            local_order = local_order[
                : benchmark_rollout_batches_per_rank * active_scene_batch_size
            ]
        if hdp_rollout_phase and shared_scene_encoding is not None:
            # The immutable source may have been padded for a larger rollout
            # pack (Cycle 2 used 640 scenes/rank, PlannerRFT uses 192).  The
            # unpadded shuffle and all real scenes are identical; only the
            # final repeated prefix is longer.  Reuse is safe exactly when the
            # current canonical order is a complete prefix of the source.
            shared_prefix_matches = (
                len(shared_scene_paths) >= len(local_order)
                and local_order == shared_scene_paths[: len(local_order)]
            )
            if not shared_prefix_matches:
                mismatch = next(
                    (
                        index
                        for index, (current, shared) in enumerate(
                            zip(local_order, shared_scene_paths)
                        )
                        if current != shared
                    ),
                    min(len(local_order), len(shared_scene_paths)),
                )
                raise RuntimeError(
                    "shared scene encoding order mismatch: "
                    f"rank={rank}, index={mismatch}, "
                    f"current_count={len(local_order)}, "
                    f"shared_count={len(shared_scene_paths)}"
                )
            # Keep the canonical scene order/index while assigning a fresh,
            # reproducible candidate-noise stream to every refresh cycle.
            rollout_rng_seed = args.seed + epoch * 1_000_003 + rank
            random.seed(rollout_rng_seed)
            np.random.seed(rollout_rng_seed % (2**32))
            torch.manual_seed(rollout_rng_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(rollout_rng_seed)
        if hdp_rollout_phase and shared_context_scene_paths:
            shared_context_prefix_matches = (
                len(shared_context_scene_paths) >= len(local_order)
                and local_order
                == shared_context_scene_paths[: len(local_order)]
            )
            if not shared_context_prefix_matches:
                mismatch = next(
                    (
                        index
                        for index, (current, shared) in enumerate(
                            zip(local_order, shared_context_scene_paths)
                        )
                        if current != shared
                    ),
                    min(len(local_order), len(shared_context_scene_paths)),
                )
                raise RuntimeError(
                    "shared decoder-context order mismatch: "
                    f"rank={rank}, index={mismatch}, "
                    f"current_count={len(local_order)}, "
                    f"shared_count={len(shared_context_scene_paths)}"
                )
        if hdp_rollout_phase and shared_expert_scene_paths:
            shared_expert_prefix_matches = (
                len(shared_expert_scene_paths) >= len(local_order)
                and local_order
                == shared_expert_scene_paths[: len(local_order)]
            )
            if not shared_expert_prefix_matches:
                mismatch = next(
                    (
                        index
                        for index, (current, shared) in enumerate(
                            zip(local_order, shared_expert_scene_paths)
                        )
                        if current != shared
                    ),
                    min(len(local_order), len(shared_expert_scene_paths)),
                )
                raise RuntimeError(
                    "shared expert-anchor order mismatch: "
                    f"rank={rank}, index={mismatch}, "
                    f"current_count={len(local_order)}, "
                    f"shared_count={len(shared_expert_scene_paths)}"
                )
        reuse_mining_rank = False
        if resume_mining_root is not None and hdp_rollout_phase:
            reuse_mining_rank = _is_complete_mining_rank_cache(
                replay_root,
                rank=rank,
                expected_count=len(local_order),
                group_size=int(rollout_config.n_trajectories),
                future_len=int(model_args.future_len),
                expected_scene_paths=local_order,
            )
            print(
                f"Mining-cache recovery rank {rank}: "
                f"{'reusing committed cache' if reuse_mining_rank else 'remine rank'}",
                flush=True,
            )
            if reuse_mining_rank:
                # The rank must still reach the final DDP barrier, but it
                # must not execute rollout forwards or backward collectives.
                local_order = []
        disk_replay_writer: _DiskReplayWriter | None = None
        if faithful_disk_replay and hdp_rollout_phase and not reuse_mining_rank:
            if disk_replay_reader is not None:
                disk_replay_reader.close()
                disk_replay_reader = None
            disk_replay_writer = _DiskReplayWriter(
                replay_root,
                rank=rank,
                expected_count=len(local_order),
                group_size=int(rollout_config.n_trajectories),
                future_len=int(model_args.future_len),
                cache_decoder_context=bool(
                    train_config.get("cache_replay_decoder_context", False)
                ),
                cache_expert_anchor=(
                    float(train_config.get("expert_anchor_weight", 0.0)) > 0.0
                ),
                shared_scene_encoding_path=(
                    Path(shared_encoding_contract["encoding_path"])
                    if bool(shared_encoding_contract.get("enabled", False))
                    else None
                ),
                shared_decoder_context_paths=(
                    shared_decoder_context_paths or None
                ),
                expected_scene_paths=local_order,
            )
        # The released HDP condition is global epoch<=5.  T4 also supports an
        # explicit every-refresh schedule because one frozen replay cycle is
        # ten epochs long and unaugmented later-cycle K bundles can collapse.
        epoch_rollout_config = rollout_config
        if hdp_rollout_interval > 0 and hdp_rollout_phase:
            epoch_rollout_config = copy.copy(rollout_config)
            schedule = str(
                rollout_config.hdp_trajectory_augmentation_schedule
            ).lower()
            if schedule == "first_five_epochs":
                enabled = epoch <= 5
            elif schedule == "first_refresh":
                enabled = epoch == start_epoch
            elif schedule == "every_refresh":
                enabled = True
            elif schedule == "never":
                enabled = False
            else:
                raise ValueError(
                    "unknown hdp_trajectory_augmentation_schedule="
                    f"{schedule!r}"
                )
            epoch_rollout_config.hdp_trajectory_augmentation = bool(
                rollout_config.hdp_trajectory_augmentation and enabled
            )
        train_rows: list[dict[str, Any]] = []
        if active_scene_batch_size > 1:
            batch_starts = range(0, len(local_order), active_scene_batch_size)
        else:
            batch_starts = range(len(local_order))
        active_rollout_prefetch = (
            rollout_prefetch_batches
            if faithful_disk_replay
            and hdp_rollout_phase
            and active_scene_batch_size > 1
            else 0
        )
        prepared_scene_batches = _iter_prefetched_scene_batches(
            local_order,
            batch_starts,
            active_scene_batch_size,
            rollout_scene_load_workers,
            active_rollout_prefetch,
            shared_scene_encoding=(
                shared_scene_encoding if hdp_rollout_phase else None
            ),
            shared_expert_anchor=(
                shared_expert_anchor if hdp_rollout_phase else None
            ),
        )
        for (
            batch_start,
            prefetched_scene_paths,
            prefetched_scene_data,
            prefetch_error,
        ) in prepared_scene_batches:
            if active_scene_batch_size > 1:
                scene_paths = (
                    prefetched_scene_paths
                    if prefetched_scene_paths is not None
                    else local_order[
                        batch_start : batch_start + active_scene_batch_size
                    ]
                )
                local_index = batch_start // active_scene_batch_size
                scene_path: str | list[str] = scene_paths
                scene_index = rank + local_index * world_size * scene_batch_size
                accumulation_count = 1
                is_accumulation_end = True
                accumulation_start = local_index
            else:
                scene_path = local_order[batch_start]
                scene_paths = [scene_path]
                local_index = batch_start
                scene_index = rank + local_index * world_size
                accumulation_start = (
                    local_index // gradient_accumulation_scenes
                ) * gradient_accumulation_scenes
                accumulation_end = min(
                    accumulation_start + gradient_accumulation_scenes,
                    len(local_order),
                )
                accumulation_count = max(1, accumulation_end - accumulation_start)
                is_accumulation_end = local_index + 1 == accumulation_end
            rollout: AWRRollout | None = None
            try:
                if prefetch_error is not None:
                    raise prefetch_error
                row, rollout = _train_scene(
                    model,
                    behavior_model,
                    model_args,
                    optimizer,
                    scene_path,
                    epoch_rollout_config,
                    reward_config,
                    train_config,
                    device,
                    amp_dtype,
                    grad_scaler,
                    optimizer_step=is_accumulation_end,
                    zero_grad=local_index == accumulation_start,
                    sync_gradients=is_accumulation_end,
                    gradient_scale=1.0 / accumulation_count,
                    collect_only=hdp_rollout_phase,
                    scene_load_workers=(
                        rollout_scene_load_workers
                        if hdp_rollout_phase
                        else int(train_config.get("scene_load_workers", 1))
                    ),
                    data_override=prefetched_scene_data,
                    exploration_reference_model=exploration_reference_model,
                    exploration_reference_model_args=exploration_reference_model_args,
                )
                if is_main and rollout is not None and len(scene_paths) == 1 and scene_index < save_rollouts:
                    _save_rollout(run_dir / f"train_epoch_{epoch:03d}" / "rollouts", scene_index, scene_path, rollout)
            except Exception as error:
                # Never silently turn a CUDA OOM into a partially trained
                # checkpoint.  Scene-level data issues may be skipped, but a
                # memory failure means the configured full-corpus batch is
                # not valid and must be fixed before trusting the run.
                if isinstance(error, torch.cuda.OutOfMemoryError):
                    raise
                if (
                    faithful_disk_replay
                    and hdp_rollout_phase
                    and disk_replay_writer is not None
                    and len(scene_paths) > 1
                ):
                    # Isolate malformed NPZs without discarding the other 191
                    # valid scene groups in a packed rollout batch.  Successful
                    # single-scene retries are written immediately.  A truly
                    # bad scene still leaves the strict writer incomplete, so
                    # close() remains fail-closed and the valid prefix can be
                    # salvaged explicitly on resume.
                    print(
                        f"Packed rollout batch failed; retrying {len(scene_paths)} "
                        f"scenes individually: {type(error).__name__}: {error}"
                    )
                    for fallback_path in scene_paths:
                        try:
                            fallback_row, fallback_rollout = _train_scene(
                                model,
                                behavior_model,
                                model_args,
                                optimizer,
                                fallback_path,
                                epoch_rollout_config,
                                reward_config,
                                train_config,
                                device,
                                amp_dtype,
                                grad_scaler,
                                collect_only=True,
                                scene_load_workers=1,
                                exploration_reference_model=exploration_reference_model,
                                exploration_reference_model_args=exploration_reference_model_args,
                            )
                            train_rows.append(fallback_row)
                            disk_replay_writer.append(
                                [fallback_path],
                                fallback_rollout,
                                group_size=int(rollout_config.n_trajectories),
                            )
                        except Exception as fallback_error:
                            if isinstance(
                                fallback_error, torch.cuda.OutOfMemoryError
                            ):
                                raise
                            train_rows.append(
                                {
                                    "scene_path": fallback_path,
                                    "scene_count": 1,
                                    "skipped": 1.0,
                                    "skip_reason": (
                                        f"{type(fallback_error).__name__}: "
                                        f"{fallback_error}"
                                    ),
                                }
                            )
                            print(
                                f"Scene failed after isolated retry "
                                f"({fallback_path}): {fallback_error}"
                            )
                            # Do not append any later scene into this failed
                            # scene's positional slot.  A strict canonical
                            # prefix is the only partial cache that can be
                            # salvaged without path/encoding misalignment.
                            raise RuntimeError(
                                "rollout mining stopped at the first "
                                f"unrecoverable scene: {fallback_path}"
                            ) from fallback_error
                    continue
                row = {"scene_path": scene_path, "skipped": 1.0, "skip_reason": f"{type(error).__name__}: {error}"}
                print(f"Scene failed ({scene_path}): {error}")
            train_rows.append(row)
            if (
                faithful_disk_replay
                and hdp_rollout_phase
                and disk_replay_writer is not None
                and rollout is not None
            ):
                # Store the entire scene-major group, including the frozen
                # behavior encoder.  This is the original HDP cache contract:
                # replay does not re-rollout or re-encode with the changing
                # policy.  A failed scene is intentionally not silently
                # replaced by a bad target; close() will fail closed below.
                disk_replay_writer.append(
                    scene_paths,
                    rollout,
                    group_size=int(rollout_config.n_trajectories),
                )
            if (
                not faithful_disk_replay
                and
                replay_capacity > 0
                and rollout is not None
                and float(rollout.weights.sum().item()) > 0.0
                and float(rollout.diagnostics.get("valid_group", 0.0)) > 0.5
                and len(scene_paths) == 1
            ):
                # Keep targets/encodings on host memory; replay must not pin
                # one scene's CUDA graph or grow rank-local VRAM over an
                # 8-GPU epoch.
                replay_buffer.append((str(scene_path), _rollout_to_cpu(rollout)))
            if is_main and (local_index + 1) % max(1, int(train_config.get("log_interval", 10))) == 0:
                print(
                    f"epoch {epoch} rank0 scene {local_index + 1}/"
                    f"{max(1, len(local_order) // active_scene_batch_size)} "
                    f"loss={row.get('loss', float('nan')):.5f} "
                    f"Rdet={row.get('det_reward', float('nan')):+.2f} "
                    f"Rbest={row.get('best_reward', float('nan')):+.2f} "
                    f"valid={row.get('valid_group', 0):.0f}"
                )
            if (
                not hdp_rollout_phase
                and optimizer_step
                and not ema_per_epoch
            ):
                update_ema(behavior_model, model, ema_decay)

        if faithful_disk_replay and hdp_rollout_phase:
            if disk_replay_writer is None:
                if not (resume_mining_root is not None and reuse_mining_rank):
                    raise RuntimeError(
                        "missing faithful replay writer in rollout epoch"
                    )
                print(
                    f"HDP replay refresh epoch {epoch}: rank {rank} "
                    "reused its committed mining cache",
                    flush=True,
                )
            else:
                manifest_path = disk_replay_writer.close()
                if is_main:
                    print(
                        f"HDP replay refresh epoch {epoch}: "
                        f"{disk_replay_writer.position:,} frozen groups/rank "
                        f"written to {manifest_path.parent}"
                    )
            if distributed:
                dist.barrier()
            disk_replay_reader = _DiskReplayReader(
                replay_root,
                rank,
                positive_advantage_only=rollout_config.positive_advantage_only,
                deterministic_first=rollout_config.deterministic_first,
                compressed_context_root=args.compressed_replay_context_root,
                compressed_context_workers=max(
                    1, int(train_config.get("scene_load_workers", 1))
                ),
                context_active_weight_only=(
                    args.compressed_replay_context_root is not None
                    and _can_materialize_only_nonzero_replay_groups(train_config)
                ),
            )
            if distributed:
                replay_size = torch.tensor(
                    [len(disk_replay_reader)], device=device, dtype=torch.int64
                )
                replay_min = replay_size.clone()
                replay_max = replay_size.clone()
                dist.all_reduce(replay_min, op=dist.ReduceOp.MIN)
                dist.all_reduce(replay_max, op=dist.ReduceOp.MAX)
                if int(replay_min.item()) != int(replay_max.item()):
                    raise RuntimeError(
                        "DDP ranks wrote different replay sizes: "
                        f"min={int(replay_min.item())}, max={int(replay_max.item())}"
                    )

        # HDP trains reward-weighted samples from a replay buffer after the
        # rollout pass.  The disk backend below follows the reference
        # ``ReplayBuffer.get(B)`` contract: each replay batch draws B groups
        # with replacement, then performs one reward-weighted diffusion
        # update.  The legacy memory backend remains available for small
        # ablations.
        replay_rows: list[dict[str, Any]] = []
        if faithful_disk_replay:
            if not hdp_rollout_phase:
                if disk_replay_reader is None:
                    raise RuntimeError(
                        "faithful replay training started without a replay cache"
                    )
                if replay_updates_per_epoch < 0:
                    # One replay DataLoader-sized update per rank-local
                    # buffer population, as in the reference trainer.  The
                    # groups *inside* each update are sampled with
                    # replacement, so this is not an accidental second
                    # deterministic pass over the mining order.
                    current_replay_updates = max(
                        1,
                        math.ceil(len(disk_replay_reader) / scene_batch_size),
                    )
                else:
                    current_replay_updates = max(0, replay_updates_per_epoch)
                if distributed:
                    # Salvaged prefixes may differ by a few failed packed
                    # batches.  DDP must still execute the same number of
                    # backwards on every rank.  With replacement, using the
                    # largest rank-local pass preserves all available update
                    # budget while smaller ranks simply resample a little
                    # more often.
                    update_count = torch.tensor(
                        [current_replay_updates], device=device, dtype=torch.int64
                    )
                    dist.all_reduce(update_count, op=dist.ReduceOp.MAX)
                    current_replay_updates = int(update_count.item())
                if replay_sampling == "with_replacement":
                    replay_iterator = disk_replay_reader.iter_random_batches(
                        scene_batch_size,
                        current_replay_updates,
                        seed=args.seed + epoch * 1000003 + rank,
                    )
                elif replay_sampling == "reward_stratified":
                    replay_iterator = (
                        disk_replay_reader.iter_reward_stratified_batches(
                            scene_batch_size,
                            current_replay_updates,
                            seed=args.seed + epoch * 1000003 + rank,
                            priority_best_reward_lt=(
                                replay_priority_best_reward_lt
                            ),
                            priority_fraction=replay_priority_fraction,
                        )
                    )
                else:
                    # A deterministic shuffled pass is useful for IO
                    # benchmarks.  The faithful experiment uses the branch
                    # above; keep this option explicit rather than silently
                    # changing HDP's random-choice distribution.
                    permutation = np.random.default_rng(
                        args.seed + epoch * 1000003 + rank
                    ).permutation(len(disk_replay_reader))

                    def _iter_shuffled_replay():
                        if len(permutation) == 0:
                            return
                        for update_index in range(current_replay_updates):
                            start = update_index * scene_batch_size
                            positions = (
                                np.arange(start, start + scene_batch_size)
                                % len(permutation)
                            )
                            indices = permutation[positions]
                            yield disk_replay_reader.batch_from_indices(
                                indices, sampled_with_replacement=False
                            )

                    replay_iterator = _iter_shuffled_replay()
                replay_context_only = bool(
                    disk_replay_reader.has_cached_scene_encoding
                    and (
                        float(train_config.get("expert_anchor_weight", 0.0))
                        <= 0.0
                        or disk_replay_reader.has_cached_expert_anchor
                    )
                )
                prepared_replay_iterator = _iter_prefetched_replay_batches(
                    replay_iterator,
                    scene_load_workers=int(
                        train_config.get("scene_load_workers", 1)
                    ),
                    prefetch_batches=int(
                        train_config.get("replay_prefetch_batches", 0)
                    ),
                    replay_context_only=replay_context_only,
                )
                # Keep the effective HDP replay batch while allowing a
                # smaller CUDA microbatch.  This is required when the rollout
                # job also carried a frozen guidance model: B=192 targets can
                # exceed 80 GiB during the backward graph even though mining
                # fits comfortably.  Scaling each microbatch loss and stepping
                # EMA only at the accumulation boundary is mathematically the
                # same mean objective and preserves one optimizer/EMA update
                # per configured effective batch.
                replay_accumulation_steps = max(
                    1,
                    math.ceil(
                        gradient_accumulation_scenes / scene_batch_size
                    ),
                )
                for replay_index, (
                    replay_scenes,
                    replay_rollout,
                    replay_data,
                ) in enumerate(prepared_replay_iterator):
                    replay_window_start = (
                        replay_index // replay_accumulation_steps
                    ) * replay_accumulation_steps
                    replay_window_end = min(
                        replay_window_start + replay_accumulation_steps,
                        current_replay_updates,
                    )
                    replay_window_count = max(
                        1, replay_window_end - replay_window_start
                    )
                    replay_is_window_start = replay_index == replay_window_start
                    replay_is_window_end = replay_index + 1 == replay_window_end
                    replay_row, _ = _train_scene(
                        model,
                        behavior_model,
                        model_args,
                        optimizer,
                        replay_scenes,
                        rollout_config,
                        reward_config,
                        train_config,
                        device,
                        amp_dtype,
                        grad_scaler,
                        rollout_override=replay_rollout,
                        replay_update=True,
                        optimizer_step=replay_is_window_end,
                        zero_grad=replay_is_window_start,
                        sync_gradients=replay_is_window_end,
                        gradient_scale=1.0 / replay_window_count,
                        scene_load_workers=int(
                            train_config.get("scene_load_workers", 1)
                        ),
                        data_override=replay_data,
                    )
                    replay_rows.append(replay_row)
                    if replay_is_window_end and not ema_per_epoch:
                        # Conventional minibatch EMA ablation. The explicit
                        # conservative policy-interpolation path instead uses
                        # one boundary commit below.
                        update_ema(behavior_model, model, ema_decay)
                    if is_main and (
                        replay_index == 0
                        or (replay_index + 1)
                        % max(1, int(train_config.get("log_interval", 64)))
                        == 0
                    ):
                        print(
                            f"epoch {epoch} replay {replay_index + 1}/"
                            f"{current_replay_updates} "
                            f"loss={replay_row.get('loss', float('nan')):.5f} "
                            f"grad={replay_row.get('grad_norm', float('nan')):.3f}"
                        )
        else:
            if hdp_rollout_phase:
                current_replay_updates = 0
            elif replay_updates_per_epoch < 0:
                current_replay_updates = len(replay_buffer)
                if distributed:
                    replay_count_tensor = torch.tensor(
                        [current_replay_updates], device=device, dtype=torch.int64
                    )
                    dist.all_reduce(replay_count_tensor, op=dist.ReduceOp.MIN)
                    current_replay_updates = int(replay_count_tensor.item())
            else:
                current_replay_updates = replay_updates_per_epoch
            replay_ready = bool(replay_buffer) and current_replay_updates > 0
            if distributed:
                ready_tensor = torch.tensor(
                    [1 if replay_ready else 0], device=device, dtype=torch.int64
                )
                dist.all_reduce(ready_tensor, op=dist.ReduceOp.MIN)
                replay_ready = bool(ready_tensor.item())
            if replay_ready:
                for replay_index in range(current_replay_updates):
                    buffer_index = (
                        epoch * current_replay_updates + replay_index + rank
                    ) % len(replay_buffer)
                    replay_scene, replay_rollout = replay_buffer[buffer_index]
                    replay_start = (
                        replay_index // gradient_accumulation_scenes
                    ) * gradient_accumulation_scenes
                    replay_end = min(
                        replay_start + gradient_accumulation_scenes,
                        current_replay_updates,
                    )
                    replay_count = max(1, replay_end - replay_start)
                    replay_is_end = replay_index + 1 == replay_end
                    replay_row, _ = _train_scene(
                        model,
                        behavior_model,
                        model_args,
                        optimizer,
                        replay_scene,
                        rollout_config,
                        reward_config,
                        train_config,
                        device,
                        amp_dtype,
                        grad_scaler,
                        rollout_override=replay_rollout,
                        replay_update=True,
                        optimizer_step=replay_is_end,
                        zero_grad=replay_index == replay_start,
                        sync_gradients=replay_is_end,
                        gradient_scale=1.0 / replay_count,
                        scene_load_workers=int(
                            train_config.get("scene_load_workers", 1)
                        ),
                    )
                    replay_rows.append(replay_row)
                    if (
                        replay_is_end
                        and not ema_per_epoch
                    ):
                        update_ema(behavior_model, model, ema_decay)
        train_rows.extend(replay_rows)

        epoch_policy_commit: dict[str, float] = {}
        if ema_per_epoch:
            if ema_commit_live_policy:
                epoch_policy_commit = _commit_epoch_ema_policy_update(
                    model,
                    behavior_model,
                    optimizer,
                    decay=ema_decay,
                )
            else:
                update_ema(behavior_model, model, ema_decay)

        compact_train_summaries: list[dict[str, Any]] | None = None
        compact_numeric_sums: dict[str, float] = {}
        compact_mine_summary = bool(
            reuse_baseline_after_mining and hdp_rollout_phase
        )
        # A replay epoch has already completed every optimizer/EMA operation.
        # Only its scalar report remains, so avoid gathering thousands of
        # Python dictionaries and preserve the historical percentiles through
        # eight small numeric vectors instead. This changes reporting cost only.
        compact_replay_summary = bool(
            faithful_disk_replay and not hdp_rollout_phase
        )
        compact_train_summary = compact_mine_summary or compact_replay_summary
        if distributed:
            if compact_train_summary:
                # A mine-only refresh has no optimizer update and its exact
                # per-scene facts already live in the strict replay memmaps.
                # Gathering 5.4M Python dicts only to compute scalar means is a
                # reporting bottleneck, not part of the learning objective.
                # Replay rows likewise have no downstream consumer after their
                # exact mean/p50/p90 report has been reduced.
                local_summary = _compact_numeric_row_summary(
                    train_rows,
                    include_percentile_values=compact_replay_summary,
                )
                gathered_summaries: list[dict[str, Any] | None] | None = (
                    [None] * world_size if is_main else None
                )
                dist.gather_object(
                    local_summary,
                    gathered_summaries,
                    dst=0,
                    group=object_group,
                )
                train_rows.clear()
                if is_main:
                    assert gathered_summaries is not None
                    compact_train_summaries = [
                        summary for summary in gathered_summaries if summary is not None
                    ]
            else:
                gathered_rows: list[list[dict[str, Any]] | None] | None = (
                    [None] * world_size if is_main else None
                )
                dist.gather_object(train_rows, gathered_rows, dst=0, group=object_group)
                if is_main:
                    assert gathered_rows is not None
                    train_rows = [
                        row
                        for rank_rows in gathered_rows
                        if rank_rows
                        for row in rank_rows
                    ]
        elif compact_train_summary:
            compact_train_summaries = [
                _compact_numeric_row_summary(
                    train_rows,
                    include_percentile_values=compact_replay_summary,
                )
            ]
            train_rows.clear()

        eval_dir = run_dir / f"eval_epoch_{epoch:03d}"
        # ``best_train.pth`` is selected by the fixed train-set score of this
        # EMA and downstream checkpoint loading uses ``ema_state_dict`` by
        # default. Validate that same deployable policy, not the transient live
        # optimizer state. This keeps W&B/full-valid evidence aligned with the
        # single checkpoint that can actually replace the source model.
        eval_policy = behavior_model
        # A standalone faithful refresh never updates either policy. Reuse the
        # exact starting rows instead of spending another full valid + selector
        # pass on an identical checkpoint. All other runs still evaluate every
        # epoch across all ranks.
        if reuse_baseline_after_mining and hdp_rollout_phase:
            eval_summary = dict(base_summary)
            eval_rows = list(base_rows)
            train_selector_summary = dict(train_selector_base_summary)
            train_selector_rows = list(train_selector_base_rows)
            if is_main:
                print(
                    "Mine-only refresh left the policy unchanged; reusing the "
                    "starting validation and train-selector results."
                )
        else:
            # Every rank participates in validation.  The policy is already
            # synchronized by DDP; only scalar per-scene rows are gathered after
            # inference, so the seven non-main GPUs are never left idle behind a
            # long full-corpus evaluation.
            eval_summary, eval_rows = _distributed_evaluate_model(
                eval_policy,
                model_args,
                valid_paths,
                eval_rollout_config,
                reward_config,
                device,
                eval_dir / "rollouts",
                epoch=epoch,
                save_rollouts=save_rollouts,
                scene_batch_size=eval_scene_batch_size,
                scene_load_workers=eval_scene_load_workers,
                scene_prefetch_batches=eval_prefetch_batches,
                distributed=distributed,
                rank=rank,
                world_size=world_size,
                object_group=object_group,
            )

            train_selector_summary = {}
            train_selector_rows = []
            if train_selector_paths:
                selector_dir = run_dir / f"train_selector_epoch_{epoch:03d}"
                train_selector_summary, train_selector_rows = (
                    _distributed_evaluate_model(
                        behavior_model,
                        model_args,
                        train_selector_paths,
                        train_selector_config,
                        reward_config,
                        device,
                        selector_dir / "rollouts",
                        epoch=epoch,
                        save_rollouts=0,
                        scene_batch_size=eval_scene_batch_size,
                        scene_load_workers=eval_scene_load_workers,
                        scene_prefetch_batches=eval_prefetch_batches,
                        distributed=distributed,
                        rank=rank,
                        world_size=world_size,
                        object_group=object_group,
                    )
                )

        rollback_epoch_to_best = False
        if is_main:
            if compact_train_summaries is not None:
                train_summary, compact_numeric_sums = (
                    _merge_compact_numeric_row_summaries(compact_train_summaries)
                )
                if compact_mine_summary:
                    train_summary["scene_rows_source"] = str(replay_root)
                else:
                    train_summary["scene_rows_source"] = (
                        "omitted_after_exact_numeric_reduction"
                    )
            else:
                train_summary = {
                    "scene_count": int(
                        sum(int(row.get("scene_count", 1)) for row in train_rows)
                    )
                }
                numeric_keys = sorted(
                    {
                        key
                        for row in train_rows
                        for key, value in row.items()
                        if key not in {"scene_count", "epoch"}
                        and isinstance(value, (int, float))
                    }
                )
                for key in numeric_keys:
                    values = [
                        float(row[key])
                        for row in train_rows
                        if key in row
                        and isinstance(row[key], (int, float))
                        and math.isfinite(float(row[key]))
                    ]
                    if values:
                        train_summary[f"mean_{key}"] = float(np.mean(values))
                        if key in {
                            "loss",
                            "grad_norm",
                            "det_reward",
                            "best_reward",
                            "reward_std",
                            "effective_sample_size",
                            "top1_weight_share",
                            "best_vs_det_reward_gain",
                        }:
                            train_summary[f"p50_{key}"] = float(np.median(values))
                            train_summary[f"p90_{key}"] = float(
                                np.percentile(values, 90)
                            )
            train_summary.update(
                {
                    "epoch": epoch,
                    "cycle": 1 + (epoch - 1) // max(1, hdp_rollout_interval),
                    "rollout_refresh": float(hdp_rollout_phase),
                    "replay_training": float(not hdp_rollout_phase),
                }
            )
            train_summary.update(epoch_policy_commit)
            train_summary["elapsed_sec"] = time.time() - epoch_start
            train_summary["scenes_per_sec"] = (
                float(train_summary["scene_count"])
                / max(float(train_summary["elapsed_sec"]), 1e-6)
            )
            train_summary["optimizer_steps"] = int(
                (
                    compact_numeric_sums.get("optimizer_step", 0.0)
                    if compact_train_summaries is not None
                    else sum(
                        float(row.get("optimizer_step", 0.0))
                        for row in train_rows
                    )
                )
                / max(1, world_size)
            )
            train_summary["learning_rate"] = float(optimizer.param_groups[0]["lr"])
            if device.type == "cuda":
                train_summary["gpu_peak_allocated_gib_rank0"] = float(
                    torch.cuda.max_memory_allocated(device) / (1024**3)
                )
                train_summary["gpu_peak_reserved_gib_rank0"] = float(
                    torch.cuda.max_memory_reserved(device) / (1024**3)
                )
            selector_source = (
                train_selector_summary if train_selector_paths else train_summary
            )
            current_train_reward = float(
                selector_source.get("mean_det_reward", -math.inf)
            )
            best_reward_before_epoch = float(best_train_reward)
            train_selector_update = current_train_reward > best_train_reward
            policy_transaction_candidate = bool(
                not hdp_rollout_phase
                and ema_per_epoch
                and ema_commit_live_policy
                and train_selector_paths
            )
            rollback_epoch_to_best = bool(
                policy_transaction_candidate
                and bool(train_config.get("rollback_rejected_epoch", False))
                and not train_selector_update
            )
            train_summary["policy_transaction_candidate"] = float(
                policy_transaction_candidate
            )
            train_summary["policy_transaction_accepted"] = float(
                policy_transaction_candidate and train_selector_update
            )
            train_summary["policy_transaction_rolled_back"] = float(
                rollback_epoch_to_best
            )
            train_summary["policy_training_continued_from_proposal"] = float(
                policy_transaction_candidate and not rollback_epoch_to_best
            )
            train_summary["train_selection_reward_before_epoch"] = (
                best_reward_before_epoch
            )
            train_summary["train_selection_reward_delta_vs_best"] = (
                current_train_reward - best_reward_before_epoch
            )
            if policy_transaction_candidate:
                if train_selector_update:
                    accepted_policy_transactions += 1
                else:
                    rejected_policy_transactions += 1
            if train_selector_update:
                best_train_reward = current_train_reward
                best_train_epoch = epoch
                if wandb_run is not None:
                    wandb_run.summary["best/train_selection_epoch"] = int(
                        best_train_epoch
                    )
                    wandb_run.summary["best/train_selection_reward_pct"] = float(
                        best_train_reward * 100.0
                    )
            train_dir = run_dir / f"train_epoch_{epoch:03d}"
            _write_json(train_dir / "summary.json", train_summary)
            # Mine-only per-scene trajectories/rewards/weights are the replay
            # cache itself. Replay epochs have already reduced their exact
            # numeric summary. Writing either full dict list adds no model or
            # selection evidence and causes multi-minute close overhead.
            _write_json(
                train_dir / "scenes.json",
                [] if compact_train_summaries is not None else train_rows,
            )
            _append_jsonl(run_dir / "metrics.jsonl", {"phase": "train", **train_summary})

            # Validation and train selection both describe the deployable EMA
            # stored in the checkpoint. The transient live optimizer state is
            # intentionally not presented as replacement-model performance.
            _write_json(eval_dir / "summary.json", eval_summary)
            _write_json(eval_dir / "scenes.json", eval_rows)
            _append_jsonl(run_dir / "metrics.jsonl", {"phase": "eval", "epoch": epoch, **eval_summary})
            if train_selector_paths:
                selector_dir = run_dir / f"train_selector_epoch_{epoch:03d}"
                _write_json(selector_dir / "summary.json", train_selector_summary)
                _write_json(selector_dir / "scenes.json", train_selector_rows)
                _append_jsonl(
                    run_dir / "metrics.jsonl",
                    {
                        "phase": "train_selector_ema",
                        "epoch": epoch,
                        **train_selector_summary,
                    },
                )
            _wandb_log_epoch_summaries(
                wandb_run,
                epoch=epoch,
                train_summary=train_summary,
                eval_summary=eval_summary,
                baseline=base_summary,
                eval_rows=eval_rows,
                baseline_rows=base_rows,
            )
            if train_selector_paths:
                _wandb_log_summary(
                    wandb_run,
                    "train_selector_ema",
                    train_selector_summary,
                    epoch=epoch,
                    baseline=train_selector_base_summary,
                    rows=train_selector_rows,
                    baseline_rows=train_selector_base_rows,
                )
            if wandb_run is not None:
                report_specs = (
                    ("reward_pct", "mean_det_reward", 100.0, True),
                    ("ade_m", "mean_ade", 1.0, False),
                    ("fde_m", "mean_fde", 1.0, False),
                    ("collision_rate_pct", "mean_det_collision", 100.0, False),
                    (
                        "road_border_crossing_rate_pct",
                        "mean_det_rb_crossing",
                        100.0,
                        False,
                    ),
                    (
                        "kinematic_failure_rate_pct",
                        "mean_det_kinematic_violation",
                        100.0,
                        False,
                    ),
                )
                for name, source_key, scale, higher_is_better in report_specs:
                    value = eval_summary.get(source_key)
                    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                        continue
                    candidate = float(value) * scale
                    previous = best_eval_report[name]
                    if previous is None or (
                        candidate > previous[0]
                        if higher_is_better
                        else candidate < previous[0]
                    ):
                        best_eval_report[name] = (candidate, epoch)
                        wandb_run.summary[f"best/eval_{name}"] = candidate
                        wandb_run.summary[f"best/eval_{name}_epoch"] = int(epoch)
            save_checkpoint_pair(
                model,
                behavior_model,
                run_dir / f"epoch_{epoch:03d}.pth",
                epoch,
                {
                    "train": train_summary,
                    "train_selector_ema": train_selector_summary,
                    "eval": eval_summary,
                },
                optimizer=optimizer,
            )
            if train_selector_update:
                shutil.copy2(
                    run_dir / f"epoch_{epoch:03d}.pth",
                    run_dir / "best_train.pth",
                )
            print(
                f"Epoch {epoch} done in {train_summary['elapsed_sec']:.1f}s: "
                f"train loss={train_summary.get('mean_loss', float('nan')):.5f}, "
                f"train-selector EMA reward={current_train_reward:+.3f}, "
                f"eval det reward={eval_summary.get('mean_det_reward', float('nan')):+.3f}, "
                f"eval collision={eval_summary.get('mean_det_collision', float('nan')):.3f}"
            )

        # Checkpoint writing is rank-zero-only.  Synchronize before all ranks
        # enter the next distributed full/hard evaluation so a fast rank
        # cannot advance to a different collective order.
        if distributed:
            rollback_tensor = torch.tensor(
                [1 if rollback_epoch_to_best else 0],
                device=device,
                dtype=torch.int64,
            )
            dist.broadcast(rollback_tensor, src=0)
            rollback_epoch_to_best = bool(rollback_tensor.item())
            dist.barrier()
        if full_valid_interval > 0 and epoch % full_valid_interval == 0:
            full_eval_dir = run_dir / f"full_eval_epoch_{epoch:03d}"
            if is_main:
                print(
                    f"Running scheduled full validation at epoch {epoch}: "
                    f"{len(full_valid_paths):,} scenes on {world_size} GPUs"
                )
            full_eval_summary, full_eval_rows = _distributed_evaluate_model(
                eval_policy,
                model_args,
                full_valid_paths,
                eval_rollout_config,
                reward_config,
                device,
                full_eval_dir / "rollouts",
                epoch=epoch,
                save_rollouts=0,
                scene_batch_size=eval_scene_batch_size,
                scene_load_workers=eval_scene_load_workers,
                scene_prefetch_batches=eval_prefetch_batches,
                distributed=distributed,
                rank=rank,
                world_size=world_size,
                object_group=object_group,
            )
            if is_main:
                _write_json(full_eval_dir / "summary.json", full_eval_summary)
                _write_json(full_eval_dir / "scenes.json", full_eval_rows)
                _append_jsonl(
                    run_dir / "metrics.jsonl",
                    {"phase": "full_eval", "epoch": epoch, **full_eval_summary},
                )
                _wandb_log_summary(
                    wandb_run,
                    "full_eval",
                    full_eval_summary,
                    epoch=epoch,
                    baseline=full_base_summary,
                    rows=full_eval_rows,
                    baseline_rows=full_base_rows,
                )
            if distributed:
                dist.barrier()

            if hard_valid_paths:
                hard_eval_dir = run_dir / f"hard_eval_epoch_{epoch:03d}"
                hard_eval_summary, hard_eval_rows = _distributed_evaluate_model(
                    eval_policy,
                    model_args,
                    hard_valid_paths,
                    eval_rollout_config,
                    reward_config,
                    device,
                    hard_eval_dir / "rollouts",
                    epoch=epoch,
                    save_rollouts=0,
                    scene_batch_size=eval_scene_batch_size,
                    scene_load_workers=eval_scene_load_workers,
                    scene_prefetch_batches=eval_prefetch_batches,
                    distributed=distributed,
                    rank=rank,
                    world_size=world_size,
                    object_group=object_group,
                )
                if is_main:
                    _write_json(hard_eval_dir / "summary.json", hard_eval_summary)
                    _write_json(hard_eval_dir / "scenes.json", hard_eval_rows)
                    _append_jsonl(
                        run_dir / "metrics.jsonl",
                        {"phase": "hard_eval", "epoch": epoch, **hard_eval_summary},
                    )
                    _wandb_log_summary(
                        wandb_run,
                        "hard_eval",
                        hard_eval_summary,
                        epoch=epoch,
                        baseline=hard_base_summary,
                        rows=hard_eval_rows,
                        baseline_rows=hard_base_rows,
                    )
                    print(
                        f"Full validation epoch {epoch}: "
                        f"det reward={full_eval_summary.get('mean_det_reward', float('nan')):+.4f}; "
                        f"hard reward={hard_eval_summary.get('mean_det_reward', float('nan')):+.4f}"
                    )
                if distributed:
                    dist.barrier()
            elif is_main:
                print(
                    f"Full validation epoch {epoch}: "
                    f"det reward={full_eval_summary.get('mean_det_reward', float('nan')):+.4f}"
                )
        if distributed:
            dist.barrier()
        if rollback_epoch_to_best:
            rollback_info = _restore_best_policy_transaction(
                model,
                behavior_model,
                optimizer,
                run_dir / "best_train.pth",
            )
            if is_main:
                rollback_info.update(
                    {
                        "rejected_epoch": int(epoch),
                        "selector_reward": float(current_train_reward),
                        "best_reward": float(best_train_reward),
                        "reason": "fixed_train_selector_mean_det_reward_not_improved",
                    }
                )
                _write_json(
                    run_dir / f"policy_transaction_epoch_{epoch:03d}.json",
                    rollback_info,
                )
                print(
                    f"Epoch {epoch} policy transaction rejected; restored "
                    f"best epoch {best_train_epoch} before the next replay pass"
                )
        if distributed:
            dist.barrier()

    if is_main:
        best_train_path = run_dir / "best_train.pth"
        best_train_sha256 = (
            _file_sha256(best_train_path) if best_train_path.is_file() else None
        )
        _write_json(
            run_dir / "final_summary.json",
            {
                "baseline": base_summary,
                "start_epoch": start_epoch,
                "epochs": epochs,
                "resumed_replay": args.resume_replay_root is not None,
                "resumed_mining_cache": args.resume_mining_cache is not None,
                "inherited_refresh_baseline": (
                    str(args.inherit_refresh_baseline_root.expanduser().resolve())
                    if args.inherit_refresh_baseline_root is not None
                    else None
                ),
                "checkpoint_selection": (
                    "fixed_train_selector_ema_mean_det_reward"
                    if train_selector_paths
                    else "legacy_replay_candidate0_mean_reward"
                ),
                "primary_eval_policy_state": "deployable_ema_state_dict",
                "optimizer_continuation": optimizer_resume_info,
                "checkpoint_optimizer_state_saved": True,
                "best_train_epoch": best_train_epoch,
                "best_train_reward": best_train_reward,
                "best_train_checkpoint_sha256": best_train_sha256,
                "starting_train_selection_epoch": starting_train_selection_epoch,
                "starting_train_selection_reward": starting_train_selection_reward,
                "best_train_reward_delta_from_start": (
                    best_train_reward - starting_train_selection_reward
                    if math.isfinite(best_train_reward)
                    and math.isfinite(starting_train_selection_reward)
                    else None
                ),
                "best_train_is_starting_incumbent": (
                    best_train_epoch == starting_train_selection_epoch
                ),
                "rollback_rejected_epoch": bool(
                    train_config.get("rollback_rejected_epoch", False)
                ),
                "accepted_policy_transactions": int(
                    accepted_policy_transactions
                ),
                "rejected_policy_transactions": int(
                    rejected_policy_transactions
                ),
                "validation_role": "report_only",
            },
        )
        print(f"AWR run complete: {run_dir}")
        if wandb_run is not None:
            wandb_run.finish()
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
