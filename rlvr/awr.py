"""Advantage-weighted regression for the original Diffusion Planner.

This module deliberately contains only the AWR path used for the original
four-channel x-start DP model.  It does not import the HDP/velocity-latent
training code and it does not use GRPO guidance to manufacture targets.

The loop is:

1. sample a group of unguided DP plans from one scene;
2. score the plans with the rule-based, OBB-aware T4 reward;
3. convert group-normalised rewards to positive AWR weights;
4. regress the diffusion denoiser toward the sampled plans with those
   weights.

Groups may either retain a deterministic (zero-temperature) first member or,
for the faithful released-HDP setting, sample every member at the configured
non-zero diffusion temperature.  Members differ through their initial
diffusion noise; no hand-crafted guidance trajectory is injected.
"""

from __future__ import annotations

import contextlib
import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.utils.config import Config
from preference_optimization.utils import load_npz_data
from rlvr.reward import (
    RewardBreakdown,
    RewardConfig,
    compute_reward_batch,
    reward_breakdown_to_json_dict,
)


@dataclass
class AWRRolloutConfig:
    """Sampling and weighting settings for one AWR scene group."""

    n_trajectories: int = 8
    # HDP/PlannerRFT uses a short stochastic DDIM rollout for exploration.
    # The original DP implementation exposes the same x-start solver with a
    # configurable number of function evaluations; five is the closest
    # sampler available without changing the checkpoint's architecture.
    sample_steps: int = 5
    noise_scale_range: tuple[float, float] = (0.5, 2.0)
    beta: float = 0.75
    weight_clip: float = 20.0
    normalize_weights: bool = True
    min_group_std: float = 1e-5
    safe_only: bool = False
    structured_exploration: bool = False
    # HDP samples every member of a group from the same non-zero diffusion
    # temperature.  Keep the deterministic first member as the historical
    # original-DP default, but allow a faithful HDP-style group when this is
    # disabled.
    deterministic_first: bool = True
    # Released HDP perturbs every rollout waypoint batch during the first
    # rollout epochs: one Gaussian along-track and one Gaussian lateral offset
    # per trajectory, both with std=0.5 m.  It leaves heading unchanged.  The
    # original DP adaptation keeps this opt-in because it changes the sampled
    # behavior distribution, but enabling it is the closest reproduction of
    # HDP's ``augment_trajectory_batch`` path.
    hdp_trajectory_augmentation: bool = False
    hdp_trajectory_augmentation_std: float = 0.5
    # Conservative original-DP adaptation of HDP's group AWR: retain the
    # deterministic behavior target and only regress toward sampled members
    # whose reward is strictly above that behavior reward.
    positive_advantage_only: bool = False
    # Ignore numerically tiny reward wins.  A positive-only AWR update with a
    # zero margin can turn ordinary sampler noise on otherwise-good scenes
    # into a global drift signal; a small margin keeps the full-data run
    # focused on meaningful repairs while retaining true safety recoveries.
    positive_advantage_margin: float = 0.0
    # Inference uses the same bf16 autocast path as the HDP decoder rollout
    # when the caller enables it.  Keeping this on the rollout config makes
    # baseline and post-training evaluation use identical numerics.
    inference_amp_dtype: str = "off"


@dataclass
class AWRRollout:
    """A sampled and scored group, retained for training and visualization."""

    trajectories: torch.Tensor  # [K, T, 4], raw ego-frame coordinates
    noise_scales: torch.Tensor  # [K]
    rewards: list[RewardBreakdown]
    weights: torch.Tensor  # [K], positive, zero for invalid groups
    reward_data: dict[str, torch.Tensor]
    # The normal on-policy rollout always carries a detached behavior
    # encoding.  A disk replay reader may omit it for an old/incomplete
    # cache, in which case the loss path recomputes the encoder.
    scene_encoding: torch.Tensor | None  # [B, N, D], detached behavior encoding
    diagnostics: dict[str, float]


def clone_tensor_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Clone tensor values without disturbing non-tensor scene metadata."""

    return {
        key: value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
        for key, value in data.items()
    }


def reward_compatible_data(data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return a reward view of a raw DP scene.

    The current original-DP T4 NPZ files retain the historical
    ``neighbor_agents_future[..., 3] = yaw`` representation.  The neutral
    planner-metrics reward intentionally consumes the newer four-channel
    representation.  Convert only this view; the raw scene passed to the DP
    loss remains unchanged.
    """

    # Reward code is read-only.  The previous implementation cloned every
    # tensor (including maps and all neighbor histories) for every scene just
    # to convert the legacy 3-channel neighbor future.  Keep the scene view
    # shallow and allocate only the converted field when it is actually
    # needed; this preserves the raw DP input and the exact reward values.
    out = dict(data)
    future = out.get("neighbor_agents_future")
    if isinstance(future, torch.Tensor) and future.shape[-1] == 3:
        out["neighbor_agents_future"] = heading_to_cos_sin(future)
    return out


def expand_scene_batch(data: dict[str, Any], batch_size: int) -> dict[str, Any]:
    """Expand a B=1 scene to a candidate batch without copying large geometry."""

    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == 1:
            out[key] = value.expand(batch_size, *value.shape[1:]).contiguous()
        else:
            out[key] = value
    return out


def _model_without_wrappers(model: nn.Module) -> nn.Module:
    """Unwrap common DDP/PEFT containers when accessing the DP modules."""

    inner = model.module if hasattr(model, "module") else model
    if hasattr(inner, "base_model") and hasattr(inner.base_model, "model"):
        inner = inner.base_model.model
    return inner


def _inference_autocast(device: torch.device, amp_dtype: str):
    """Return the rollout autocast context used by the accelerated runner."""

    dtype_name = str(amp_dtype).lower()
    if device.type != "cuda":
        return contextlib.nullcontext()
    if dtype_name in {"bf16", "bfloat16"}:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if dtype_name in {"fp16", "float16", "half"}:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


@torch.no_grad()
def encode_scene(
    model: nn.Module,
    model_args: Config,
    data: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Encode one raw scene once and return a detached scene representation."""

    planner = _model_without_wrappers(model)
    norm = model_args.observation_normalizer(clone_tensor_dict(data))
    return planner.encoder(norm).detach()


@torch.no_grad()
def sample_unguided_dp_group(
    model: nn.Module,
    model_args: Config,
    data: dict[str, torch.Tensor],
    config: AWRRolloutConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample K original-DP trajectories with one cached scene encoding.

    Returns:
        trajectories: ``[K, future_len, 4]`` in raw ego-frame coordinates;
        noise_scales: ``[K]`` (the first value is zero);
        scene_encoding: detached ``[1, encoder_tokens, hidden_dim]`` tensor.
    """

    if config.n_trajectories < 1:
        raise ValueError("AWR requires n_trajectories >= 1")
    if config.noise_scale_range[0] < 0 or config.noise_scale_range[1] < config.noise_scale_range[0]:
        raise ValueError(f"invalid noise_scale_range={config.noise_scale_range}")

    planner = _model_without_wrappers(model)
    previous_training = planner.training
    previous_guidance_fn = planner.decoder._guidance_fn
    previous_guidance_scale = planner.decoder._guidance_scale
    previous_sample_steps = planner.decoder._sample_steps
    planner.eval()
    # AWR target diversity comes from the DP diffusion initial state only.
    # In particular, no centerline/road-border/anchor guidance is injected.
    planner.decoder._guidance_fn = None

    K = int(config.n_trajectories)
    P = 1 + int(model_args.predicted_neighbor_num)
    future_len = int(model_args.future_len)
    if config.deterministic_first:
        noise_scales = torch.zeros(K, device=device, dtype=torch.float32)
        if K > 1:
            noise_scales[1:] = torch.empty(K - 1, device=device).uniform_(*config.noise_scale_range)
    else:
        # This is the sampling scheme in the released HDP agent: every
        # candidate starts from Gaussian noise scaled by one fixed sampler
        # temperature.  A degenerate zero-noise anchor would make the group
        # less representative of the previous policy distribution.
        noise_scales = torch.empty(K, device=device, dtype=torch.float32).uniform_(*config.noise_scale_range)

    try:
        planner.decoder._sample_steps = max(2, int(config.sample_steps))
        norm_single = model_args.observation_normalizer(clone_tensor_dict(data))
        with _inference_autocast(device, config.inference_amp_dtype):
            scene_encoding = planner.encoder(norm_single).detach()
            norm_batch = expand_scene_batch(norm_single, K)
            initial_noise = torch.randn(K, P, future_len + 1, 4, device=device)
            norm_batch["sampled_trajectories"] = (
                noise_scales[:, None, None, None] * initial_noise
            )
            norm_batch["_cached_encoding"] = scene_encoding.expand(
                K, *scene_encoding.shape[1:]
            ).contiguous()
            norm_batch["_skip_turn_indicator"] = True
            _, outputs = planner(norm_batch)
            trajectories = outputs["prediction"][:, 0].float().detach()
    finally:
        planner.decoder._guidance_fn = previous_guidance_fn
        planner.decoder._guidance_scale = previous_guidance_scale
        planner.decoder._sample_steps = previous_sample_steps
        planner.train(previous_training)

    if trajectories.shape != (K, future_len, 4):
        raise RuntimeError(
            f"original DP returned unexpected ego shape {tuple(trajectories.shape)}; "
            f"expected {(K, future_len, 4)}"
        )
    if config.structured_exploration and K >= 4:
        # HDP/PlannerRFT shows that unconstrained Gaussian noise often gives
        # jitter rather than a new maneuver. Keep candidate 0 untouched, and
        # add deterministic, auditable proposal modes around it: smooth
        # longitudinal time-warping (yield/hold) and route-frame lateral
        # offsets. They are still scored by the same OBB reward and can be
        # rejected by safe_only; no guidance gradient is injected into DP.
        base = trajectories[0]

        def time_warp(path: torch.Tensor, alpha: float) -> torch.Tensor:
            grid = torch.linspace(
                0.0, float(future_len - 1) * alpha, future_len, device=path.device
            )
            lo = grid.floor().long().clamp(max=future_len - 1)
            hi = (lo + 1).clamp(max=future_len - 1)
            frac = (grid - lo.float()).unsqueeze(-1)
            xy = torch.lerp(path[lo, :2], path[hi, :2], frac)
            heading = torch.lerp(path[lo, 2:4], path[hi, 2:4], frac)
            heading = heading / heading.norm(dim=-1, keepdim=True).clamp_min(1e-5)
            return torch.cat([xy, heading], dim=-1)

        def lateral_offset(path: torch.Tensor, offset: float) -> torch.Tensor:
            tangent = path[1:, :2] - path[:-1, :2]
            tangent = torch.cat([tangent[:1], tangent], dim=0)
            tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(1e-5)
            normal = torch.stack([-tangent[:, 1], tangent[:, 0]], dim=-1)
            ramp = torch.linspace(0.0, 1.0, future_len, device=path.device).pow(3.0)
            xy = path[:, :2] + normal * (float(offset) * ramp[:, None])
            return torch.cat([xy, path[:, 2:4]], dim=-1)

        structured = [
            time_warp(base, 0.88),
            time_warp(base, 0.72),
            lateral_offset(base, -0.65),
            lateral_offset(base, 0.65),
            lateral_offset(time_warp(base, 0.88), -0.45),
            lateral_offset(time_warp(base, 0.88), 0.45),
            time_warp(base, 0.58),
        ]
        for slot, candidate in enumerate(structured, start=1):
            if slot >= K:
                break
            trajectories[slot] = candidate
    if config.hdp_trajectory_augmentation:
        # Match HDP-navsim/scoring.py exactly: independent scalar a/b per
        # candidate, transformed in the candidate's own heading frame.  Do
        # this after optional structured proposals so the generated target is
        # the trajectory that was actually scored and replayed.
        std = float(config.hdp_trajectory_augmentation_std)
        if std > 0.0:
            a = torch.randn(K, 1, device=trajectories.device) * std
            b = torch.randn(K, 1, device=trajectories.device) * std
            if config.deterministic_first:
                # The adaptation's candidate 0 is the deploy-time original-DP
                # behavior anchor.  HDP itself has no zero-noise member, but
                # retaining this member is necessary when we explicitly use
                # positive-advantage filtering on the original DP model.
                a[0] = 0.0
                b[0] = 0.0
            cos_yaw = trajectories[..., 2]
            sin_yaw = trajectories[..., 3]
            x_new = trajectories[..., 0] + a * cos_yaw - b * sin_yaw
            y_new = trajectories[..., 1] + a * sin_yaw + b * cos_yaw
            trajectories = torch.stack(
                (x_new, y_new, trajectories[..., 2], trajectories[..., 3]), dim=-1
            )
    return trajectories, noise_scales, scene_encoding


def _repeat_scene_batch(data: dict[str, Any], repeats: int) -> dict[str, Any]:
    """Repeat every scene-batched tensor ``repeats`` times in scene order."""

    if repeats < 1:
        raise ValueError(f"repeats must be positive, got {repeats}")
    batch_size = int(data["ego_current_state"].shape[0])
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == batch_size:
            out[key] = value.repeat_interleave(repeats, dim=0).contiguous()
        else:
            out[key] = value
    return out


@torch.no_grad()
def sample_unguided_dp_group_batch(
    model: nn.Module,
    model_args: Config,
    data: dict[str, torch.Tensor],
    config: AWRRolloutConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample K candidates for several scenes in one decoder forward.

    The original scene-wise API is intentionally kept for small diagnostic
    tools.  Full-corpus post-training uses this batched variant: the encoder
    runs once per scene and the decoder evaluates ``B*K`` candidates together,
    which is the important throughput difference for millions of NPZ frames.

    Returns:
        trajectories: ``[B, K, future_len, 4]``;
        noise_scales: ``[B, K]``;
        scene_encoding: ``[B, encoder_tokens, hidden_dim]``.
    """

    if config.n_trajectories < 1:
        raise ValueError("AWR requires n_trajectories >= 1")
    if config.noise_scale_range[0] < 0 or config.noise_scale_range[1] < config.noise_scale_range[0]:
        raise ValueError(f"invalid noise_scale_range={config.noise_scale_range}")

    batch_size = int(data["ego_current_state"].shape[0])
    K = int(config.n_trajectories)
    P = 1 + int(model_args.predicted_neighbor_num)
    future_len = int(model_args.future_len)
    planner = _model_without_wrappers(model)
    previous_training = planner.training
    previous_guidance_fn = planner.decoder._guidance_fn
    previous_guidance_scale = planner.decoder._guidance_scale
    previous_sample_steps = planner.decoder._sample_steps
    planner.eval()
    planner.decoder._guidance_fn = None

    if config.deterministic_first:
        noise_scales = torch.zeros(batch_size, K, device=device, dtype=torch.float32)
        if K > 1:
            noise_scales[:, 1:] = torch.empty(
                batch_size, K - 1, device=device, dtype=torch.float32
            ).uniform_(*config.noise_scale_range)
    else:
        noise_scales = torch.empty(
            batch_size, K, device=device, dtype=torch.float32
        ).uniform_(*config.noise_scale_range)

    try:
        planner.decoder._sample_steps = max(2, int(config.sample_steps))
        norm_scene = model_args.observation_normalizer(clone_tensor_dict(data))
        with _inference_autocast(device, config.inference_amp_dtype):
            scene_encoding = planner.encoder(norm_scene).detach()
            norm_candidates = _repeat_scene_batch(norm_scene, K)
            initial_noise = torch.randn(batch_size * K, P, future_len + 1, 4, device=device)
            norm_candidates["sampled_trajectories"] = (
                noise_scales.reshape(batch_size * K, 1, 1, 1) * initial_noise
            )
            norm_candidates["_cached_encoding"] = scene_encoding.repeat_interleave(
                K, dim=0
            ).contiguous()
            norm_candidates["_skip_turn_indicator"] = True
            _, outputs = planner(norm_candidates)
            trajectories = outputs["prediction"][:, 0].float().detach().reshape(
                batch_size, K, future_len, 4
            )
    finally:
        planner.decoder._guidance_fn = previous_guidance_fn
        planner.decoder._guidance_scale = previous_guidance_scale
        planner.decoder._sample_steps = previous_sample_steps
        planner.train(previous_training)

    if trajectories.shape != (batch_size, K, future_len, 4):
        raise RuntimeError(
            f"original DP returned unexpected batched ego shape {tuple(trajectories.shape)}; "
            f"expected {(batch_size, K, future_len, 4)}"
        )

    if config.hdp_trajectory_augmentation:
        std = float(config.hdp_trajectory_augmentation_std)
        if std > 0.0:
            a = torch.randn(batch_size, K, 1, device=trajectories.device) * std
            b = torch.randn(batch_size, K, 1, device=trajectories.device) * std
            if config.deterministic_first:
                a[:, 0] = 0.0
                b[:, 0] = 0.0
            cos_yaw = trajectories[..., 2]
            sin_yaw = trajectories[..., 3]
            x_new = trajectories[..., 0] + a * cos_yaw - b * sin_yaw
            y_new = trajectories[..., 1] + a * sin_yaw + b * cos_yaw
            trajectories = torch.stack(
                (x_new, y_new, trajectories[..., 2], trajectories[..., 3]), dim=-1
            )
    return trajectories, noise_scales, scene_encoding


def _slice_scene_data(data: dict[str, torch.Tensor], index: int) -> dict[str, torch.Tensor]:
    """Take one scene from a batch while retaining the loader's batch dim."""

    batch_size = int(data["ego_current_state"].shape[0])
    return {
        key: value[index : index + 1] if isinstance(value, torch.Tensor)
        and value.dim() > 0 and value.shape[0] == batch_size else value
        for key, value in data.items()
    }


def _add_rollout_reward_diagnostics(
    diagnostics: dict[str, float], rewards: list[RewardBreakdown]
) -> None:
    """Expose reward health without changing sampling, weights, or loss."""

    if not rewards:
        return
    det = rewards[0]
    best = max(rewards, key=lambda reward: reward.total)
    totals = np.asarray([float(reward.total) for reward in rewards], dtype=np.float64)
    finite_totals = totals[np.isfinite(totals)]
    zero_tol = 1e-8
    scalar_fields = (
        "total",
        "safety",
        "progress",
        "smoothness",
        "feasibility",
        "centerline",
        "red_light",
        "off_road_fraction",
        "rb_min_dist",
        "lane_near_frac",
        "lane_wide_frac",
        "sc_min_dist",
    )
    for field in scalar_fields:
        values = np.asarray([float(getattr(reward, field)) for reward in rewards])
        finite = values[np.isfinite(values)]
        diagnostics[f"det_{field}"] = float(getattr(det, field))
        diagnostics[f"best_{field}"] = float(getattr(best, field))
        if finite.size:
            diagnostics[f"candidate_mean_{field}"] = float(finite.mean())
    candidate_count = float(len(rewards))
    diagnostics.update(
        {
            "candidate_collision_rate": sum(
                reward.collision_step is not None for reward in rewards
            )
            / candidate_count,
            "candidate_rb_crossing_rate": sum(
                bool(reward.rb_crossing) for reward in rewards
            )
            / candidate_count,
            "candidate_lane_crossing_rate": sum(
                bool(reward.lane_crossing) for reward in rewards
            )
            / candidate_count,
            "candidate_kinematic_violation_rate": sum(
                bool(reward.kinematic_violated) for reward in rewards
            )
            / candidate_count,
            "det_lane_crossing": float(det.lane_crossing),
            "best_vs_det_reward_gain": float(best.total - det.total),
            "det_zero_reward": float(abs(float(det.total)) <= zero_tol),
            "zero_reward_candidate_fraction": float(
                np.mean(np.abs(finite_totals) <= zero_tol)
            )
            if finite_totals.size
            else float("nan"),
            "all_zero_reward_group": float(
                bool(finite_totals.size)
                and bool(np.all(np.abs(finite_totals) <= zero_tol))
            ),
            "all_equal_reward_group": float(
                bool(finite_totals.size)
                and float(np.max(finite_totals) - np.min(finite_totals)) <= zero_tol
            ),
            "reward_unique_count": float(
                np.unique(np.round(finite_totals, decimals=8)).size
            ),
            "det_zero_collision": float(
                abs(float(det.total)) <= zero_tol and det.collision_step is not None
            ),
            "det_zero_rb_crossing": float(
                abs(float(det.total)) <= zero_tol and bool(det.rb_crossing)
            ),
            "det_zero_kinematic": float(
                abs(float(det.total)) <= zero_tol and bool(det.kinematic_violated)
            ),
            "det_zero_without_collision_rb_kinematic": float(
                abs(float(det.total)) <= zero_tol
                and det.collision_step is None
                and not bool(det.rb_crossing)
                and not bool(det.kinematic_violated)
            ),
        }
    )


@torch.no_grad()
def rollout_and_score_scene_batch(
    behavior_model: nn.Module,
    model_args: Config,
    data: dict[str, torch.Tensor],
    rollout_config: AWRRolloutConfig,
    reward_config: RewardConfig,
    device: torch.device,
) -> list[AWRRollout]:
    """Generate one batched decoder pass, then score each scene independently."""

    trajectories, noise_scales, scene_encoding = sample_unguided_dp_group_batch(
        behavior_model, model_args, data, rollout_config, device
    )
    result: list[AWRRollout] = []
    for scene_index in range(trajectories.shape[0]):
        scene_data = _slice_scene_data(data, scene_index)
        scene_trajectories = trajectories[scene_index]
        # The reward adapter clones legacy neighbor futures once per scene so
        # the planner's raw loss input stays untouched.  Reuse that one view
        # for scoring and replay; cloning it twice was pure CPU/GPU overhead
        # in the full-corpus batch loop.
        scene_reward_data = reward_compatible_data(scene_data)
        scene_rewards = compute_reward_batch(
            scene_trajectories, scene_reward_data, reward_config
        )
        weights_cpu, diagnostics = compute_awr_weights(
            scene_rewards,
            beta=rollout_config.beta,
            weight_clip=rollout_config.weight_clip,
            normalize=rollout_config.normalize_weights,
            min_group_std=rollout_config.min_group_std,
            safe_only=rollout_config.safe_only,
            reward_config=reward_config,
            positive_advantage_only=rollout_config.positive_advantage_only,
            positive_advantage_margin=rollout_config.positive_advantage_margin,
        )
        diagnostics = dict(diagnostics)
        diagnostics["candidate_count"] = float(scene_trajectories.shape[0])
        diagnostics["det_reward"] = float(scene_rewards[0].total)
        diagnostics["best_reward"] = float(max(r.total for r in scene_rewards))
        diagnostics["det_collision"] = float(scene_rewards[0].collision_step is not None)
        diagnostics["det_rb_crossing"] = float(scene_rewards[0].rb_crossing)
        diagnostics["det_kinematic_violation"] = float(scene_rewards[0].kinematic_violated)
        _add_rollout_reward_diagnostics(diagnostics, scene_rewards)
        result.append(
            AWRRollout(
                trajectories=scene_trajectories,
                noise_scales=noise_scales[scene_index],
                rewards=scene_rewards,
                weights=weights_cpu.to(device=device, dtype=torch.float32),
                reward_data=scene_reward_data,
                scene_encoding=scene_encoding[scene_index : scene_index + 1],
                diagnostics=diagnostics,
            )
        )
    return result


def compute_awr_weights(
    rewards: list[RewardBreakdown],
    beta: float = 0.75,
    weight_clip: float = 20.0,
    normalize: bool = True,
    min_group_std: float = 1e-5,
    safe_only: bool = False,
    reward_config: RewardConfig | None = None,
    positive_advantage_only: bool = False,
    positive_advantage_margin: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Convert one scene's rewards to stable positive AWR weights.

    AWR uses ``w_i = exp(beta * (R_i - mean(R)) / (std(R) + eps))`` with the
    sample standard deviation used by the released HDP agent.  A finite tied
    group therefore receives unit weights: it has no within-group ranking
    signal, but it still retains the sampled behavior distribution.  This
    differs from the paper prose that says identical-reward groups are
    discarded; the caller may filter such groups separately for that
    ablation.  Optional finite-weight mean normalisation preserves the
    diffusion-loss scale across scenes.
    """

    totals = np.asarray([float(r.total) for r in rewards], dtype=np.float64)
    finite = np.isfinite(totals)
    original_finite = finite.copy()
    # ``lane_crossing`` is deliberately not an unconditional safety failure.
    # The metric is computed against the union of local lane polygons and a
    # normal lane change crosses a *shared* lane boundary.  HDP/PDM treats
    # drivable-area / collision / direction as safety semantics; lane keeping
    # is a quality diagnostic (and its official PDM weight is zero).  Only add
    # it to the AWR hard filter when the active reward config explicitly turns
    # on ``lane_gate_enabled``.  The same rule is used for road-border and
    # stopped-neighbor gates so ``safe_only`` means "the configured terminal
    # gates", not "every diagnostic that happened to be logged".
    if reward_config is None:
        reward_config = RewardConfig()
    safe = np.asarray(
        [
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
            for reward in rewards
        ],
        dtype=bool,
    )
    safe_candidate_count = int((finite & safe).sum())
    if safe_only and safe_candidate_count > 0:
        finite &= safe
    if totals.size == 0:
        return torch.zeros(0), {
            "reward_mean": float("nan"),
            "reward_std": float("nan"),
            "reward_max": float("nan"),
            "valid_group": 0.0,
            "weight_mean": 0.0,
            "weight_max": 0.0,
            "effective_sample_size": 0.0,
            "top1_weight_share": 0.0,
            "safe_candidate_count": float(safe_candidate_count),
            "safe_only": float(safe_only),
            "positive_advantage_only": float(positive_advantage_only),
            "positive_advantage_margin": float(positive_advantage_margin),
            "positive_candidate_count": 0.0,
        }

    if positive_advantage_only and bool(original_finite[0]):
        # Candidate zero is the deterministic behavior trajectory whenever
        # deterministic_first=True. Lower-return stochastic samples are kept
        # for diagnostics but receive no regression weight. If no sampled
        # target is better, retain only the behavior anchor.
        positive = original_finite & (
            totals > totals[0] + max(float(positive_advantage_margin), 1e-8)
        )
        if bool(positive.any()):
            finite &= positive
            finite[0] = True
        else:
            finite[:] = False
            finite[0] = True

    finite_totals = totals[finite]
    mean = float(finite_totals.mean()) if finite_totals.size else float("nan")
    # ``torch.std`` in the released HDP agent uses its default unbiased
    # estimator (correction=1).  Match that denominator instead of NumPy's
    # population standard deviation so the exp(group-normalised reward)
    # weights are numerically the same for K=10.
    std = (
        float(finite_totals.std(ddof=1))
        if finite_totals.size > 1
        else (0.0 if finite_totals.size == 1 else float("nan"))
    )
    # Match the released HDP implementation for tied groups.  HDP does not
    # discard a group merely because its empirical standard deviation is
    # zero; it divides by ``std + 1e-6``, which gives every finite candidate
    # weight one when all rewards tie.  Dropping these groups (or selecting a
    # single arbitrary candidate) changes the behavior distribution and is
    # especially harmful for the original DP, where many easy scenes have no
    # reward ranking at the sampled temperature.
    valid_group = bool(finite_totals.size > 0)

    weights = np.zeros_like(totals, dtype=np.float32)
    if valid_group:
        if finite_totals.size == 1:
            weights[finite] = 1.0
        else:
            z = (totals - mean) / (std + max(float(min_group_std), 1e-8))
            log_weights = np.clip(float(beta) * z, -30.0, math.log(max(weight_clip, 1.0)))
            weights[finite] = np.exp(log_weights[finite]).astype(np.float32)
        if positive_advantage_only:
            # Unit retention weight for the behavior target; any explicit BC
            # weight is added by the trainer after this function returns.
            weights[0] = 1.0
        if normalize:
            weights[finite] /= max(float(weights[finite].mean()), 1e-8)

    finite_weights = weights[weights > 0]
    weight_sum = float(finite_weights.sum())
    ess = (weight_sum * weight_sum / float((finite_weights**2).sum())) if weight_sum else 0.0
    top_share = float(finite_weights.max() / weight_sum) if weight_sum else 0.0
    diagnostics = {
        "reward_mean": mean,
        "reward_std": std,
        "reward_max": float(np.max(finite_totals)) if finite_totals.size else float("nan"),
        "valid_group": float(valid_group),
        "weight_mean": float(finite_weights.mean()) if finite_weights.size else 0.0,
        "weight_max": float(finite_weights.max()) if finite_weights.size else 0.0,
        "weight_min": float(finite_weights.min()) if finite_weights.size else 0.0,
        "effective_sample_size": ess,
        "top1_weight_share": top_share,
        "safe_candidate_count": float(safe_candidate_count),
        "safe_only": float(safe_only),
        "positive_advantage_only": float(positive_advantage_only),
        "positive_advantage_margin": float(positive_advantage_margin),
        "positive_candidate_count": float(
            (
                original_finite
                & (
                    totals
                    > totals[0] + max(float(positive_advantage_margin), 1e-8)
                )
            ).sum()
            if bool(original_finite[0])
            else 0
        ),
    }
    return torch.from_numpy(weights), diagnostics


@torch.no_grad()
def rollout_and_score_scene(
    behavior_model: nn.Module,
    model_args: Config,
    data: dict[str, torch.Tensor],
    rollout_config: AWRRolloutConfig,
    reward_config: RewardConfig,
    device: torch.device,
) -> AWRRollout:
    """Sample, OBB-score and weight one raw T4 scene."""

    trajectories, noise_scales, scene_encoding = sample_unguided_dp_group(
        behavior_model, model_args, data, rollout_config, device
    )
    reward_data = reward_compatible_data(data)
    rewards = compute_reward_batch(trajectories, reward_data, reward_config)
    weights_cpu, diagnostics = compute_awr_weights(
        rewards,
        beta=rollout_config.beta,
        weight_clip=rollout_config.weight_clip,
        normalize=rollout_config.normalize_weights,
        min_group_std=rollout_config.min_group_std,
        safe_only=rollout_config.safe_only,
        reward_config=reward_config,
        positive_advantage_only=rollout_config.positive_advantage_only,
        positive_advantage_margin=rollout_config.positive_advantage_margin,
    )
    weights = weights_cpu.to(device=device, dtype=torch.float32)
    diagnostics = dict(diagnostics)
    diagnostics["candidate_count"] = float(trajectories.shape[0])
    diagnostics["det_reward"] = float(rewards[0].total)
    diagnostics["best_reward"] = float(max(r.total for r in rewards))
    diagnostics["det_collision"] = float(rewards[0].collision_step is not None)
    diagnostics["det_rb_crossing"] = float(rewards[0].rb_crossing)
    diagnostics["det_kinematic_violation"] = float(rewards[0].kinematic_violated)
    _add_rollout_reward_diagnostics(diagnostics, rewards)
    return AWRRollout(
        trajectories=trajectories,
        noise_scales=noise_scales,
        rewards=rewards,
        weights=weights,
        reward_data=reward_data,
        scene_encoding=scene_encoding,
        diagnostics=diagnostics,
    )


def breakdown_metrics(reward: RewardBreakdown, prefix: str = "") -> dict[str, Any]:
    """Flatten a reward breakdown for JSONL/HTML consumers."""

    values = reward_breakdown_to_json_dict(reward)
    return {f"{prefix}{key}": value for key, value in values.items()}


def rollout_to_json(rollout: AWRRollout, scene_path: str | Path | None = None) -> dict[str, Any]:
    """Serialize the useful, human-auditable part of a rollout."""

    result: dict[str, Any] = {
        "scene_path": str(scene_path) if scene_path is not None else None,
        "diagnostics": rollout.diagnostics,
        "noise_scales": rollout.noise_scales.detach().cpu().tolist(),
        "rewards": [reward_breakdown_to_json_dict(r) for r in rollout.rewards],
        "weights": rollout.weights.detach().cpu().tolist(),
    }
    return result


def save_checkpoint_pair(
    model: nn.Module,
    ema_model: nn.Module,
    output_path: str | Path,
    epoch: int,
    metrics: dict[str, Any] | None = None,
) -> None:
    """Save CPU state dicts in the same shape as the original DP checkpoints."""

    def cpu_state(module: nn.Module) -> dict[str, torch.Tensor]:
        # torch.compile wraps modules in OptimizedModule and prefixes its
        # state-dict keys with ``_orig_mod``.  Strip that wrapper at save time
        # so the artifact can be loaded by the uncompiled original DP model.
        inner = module.module if hasattr(module, "module") else module
        while hasattr(inner, "_orig_mod"):
            inner = inner._orig_mod
        return {
            key.replace("._orig_mod.", ".").replace("_orig_mod.", ""): value.detach().cpu()
            for key, value in inner.state_dict().items()
        }

    payload = {
        "epoch": int(epoch),
        "model": cpu_state(model),
        "ema_state_dict": cpu_state(ema_model),
        "metrics": metrics or {},
    }
    torch.save(payload, str(output_path))


def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    """In-place EMA update, including buffers."""

    decay = float(decay)
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"ema_decay must be in [0, 1), got {decay}")
    with torch.no_grad():
        ema_inner = ema_model.module if hasattr(ema_model, "module") else ema_model
        model_inner = model.module if hasattr(model, "module") else model
        ema_state = ema_inner.state_dict()
        model_state = model_inner.state_dict()
        for key, ema_value in ema_state.items():
            value = model_state[key]
            if torch.is_floating_point(ema_value):
                ema_value.mul_(decay).add_(value, alpha=1.0 - decay)
            else:
                ema_value.copy_(value)


def dataclass_dict(value: Any) -> dict[str, Any]:
    """JSON-friendly conversion for the small AWR dataclasses."""

    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(f"expected dataclass, got {type(value)}")


def load_original_dp_checkpoint(
    model_path: str | Path,
    device: torch.device,
    args_path: str | Path | None = None,
    use_ema: bool = True,
) -> tuple[Diffusion_Planner, Config]:
    """Load and validate an original 4-channel DP checkpoint.

    The v5.0 release bundle stores its configuration as ``*.param.json`` and
    its state dict under ``ema_state_dict``.  ``args_path`` makes that layout
    usable without changing the source checkpoint.
    """

    model_path = Path(model_path)
    if args_path is None:
        candidates = [model_path.parent / "args.json"] + sorted(model_path.parent.glob("*.param.json"))
        args_path = next((p for p in candidates if p.exists()), None)
    if args_path is None or not Path(args_path).exists():
        raise FileNotFoundError(f"could not find args.json/param.json beside {model_path}")

    model_args = Config(str(args_path))
    if bool(getattr(model_args, "use_velocity_representation", False)):
        raise ValueError(
            "This AWR runner is intentionally restricted to the original DP "
            "x-start 4-channel representation; use_velocity_representation=True"
        )
    if str(getattr(model_args, "diffusion_model_type", "x_start")) != "x_start":
        raise ValueError("original DP AWR requires diffusion_model_type='x_start'")

    model = Diffusion_Planner(model_args)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if use_ema and isinstance(checkpoint, dict) and "ema_state_dict" in checkpoint:
        state = checkpoint["ema_state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint
    state = {
        key.replace("module.", "").replace("._orig_mod.", ".").replace("_orig_mod.", ""): value
        for key, value in state.items()
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint is not architecture-compatible with original DP: "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
    model.to(device)
    return model, model_args


def load_scene(path: str | Path, device: torch.device) -> dict[str, torch.Tensor]:
    """Load a raw NPZ scene using the repository's canonical DP loader."""

    return load_npz_data(str(path), device)
