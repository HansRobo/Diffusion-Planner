"""Advantage-weighted regression for the original Diffusion Planner.

This module deliberately contains only the AWR path used for the original
four-channel x-start DP model.  It does not import the HDP/velocity-latent
training code and it does not use GRPO guidance to manufacture targets.

The default loop is:

1. sample a group of unguided DP plans from one scene;
2. score the plans with the rule-based, OBB-aware T4 reward;
3. convert group-normalised rewards to positive AWR weights;
4. regress the diffusion denoiser toward the sampled plans with those
  weights.

An opt-in PlannerRFT exploration adapter can enrich only low-reward native
groups.  It generates an additional fixed-Beta guided bank around a frozen IL
reference, scores the union, and keeps the top K.  Deployment and evaluation
remain the ordinary unguided DP.  The adapter is deliberately off by default,
so historical replay contracts and the released-HDP reproduction are
unchanged.

Groups may either retain a deterministic (zero-temperature) first member or,
for the faithful released-HDP setting, sample every member at the configured
non-zero diffusion temperature.  Members differ through their initial
diffusion noise; no hand-crafted guidance trajectory is injected.
"""

from __future__ import annotations

import contextlib
import copy
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.utils.config import Config
from torch import nn

from preference_optimization.utils import load_npz_data
from rlvr.posttrain.objectives import resolve as resolve_objective
from rlvr.reward import (
    RewardBreakdown,
    RewardConfig,
    compute_reward_batch,
    reward_breakdown_to_json_dict,
)

# Wall-clock accounting for the rollout stages.  The mine leaves about two
# thirds of every GPU idle, and until now there were no timers anywhere in the
# rollout path -- only a whole-epoch one -- so there was no way to see which
# stage owns that time.  Note that denoising is one batched GPU call while
# scoring and weighting are per-scene Python loops, so a per-stage split is
# exactly what distinguishes "the GPU is too slow" from "the GPU is waiting
# for Python".
#
# These markers are deliberately sync-free: inserting torch.cuda.synchronize()
# to get textbook attribution would itself change the thing being measured.
# Asynchronous GPU work is therefore charged to whichever stage first blocks
# on it, which is the honest answer to "where does the wall clock go".
_STAGE_SECONDS: dict[str, float] = {}


def stage_end(name: str, start: float) -> None:
    """Charge ``time.perf_counter() - start`` to stage ``name``."""
    _STAGE_SECONDS[name] = _STAGE_SECONDS.get(name, 0.0) + (
        time.perf_counter() - start
    )


def drain_stage_seconds() -> dict[str, float]:
    """Return the accumulated per-stage seconds and reset the accumulator."""
    drained = dict(_STAGE_SECONDS)
    _STAGE_SECONDS.clear()
    return drained


@dataclass
class AWRRolloutConfig:
    """Sampling and weighting settings for one AWR scene group."""

    n_trajectories: int = 8
    # Which candidate-weighting objective turns group rewards into per-candidate
    # weights: see rlvr/posttrain/objectives.py for the registry.
    objective: str = "awr"
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
    # HDP adds one constant route-frame offset to every waypoint.  That is not
    # state-continuous for the original DP's absolute x-start trajectory: the
    # first 0.1 s point can jump by ~0.5 m.  A positive value ramps the same
    # sampled offset from zero with a minimum-jerk quintic onset, then updates
    # heading from the resulting path.  Zero preserves the released HDP
    # transform exactly for reproduction/ablation.
    hdp_trajectory_augmentation_ramp_steps: int = 0
    # A time-varying offset does not have HDP's exact heading-preservation
    # symmetry.  ``preserve`` is the released-transform-compatible and robust
    # default. ``velocity_offset`` adds the smooth offset velocity to the
    # model heading without differentiating noisy sampled x/y. ``tangent`` is
    # retained only for diagnostics; direct 0.1 s finite differences amplify
    # stochastic DP position jitter and must not be used for formal mining.
    hdp_trajectory_augmentation_heading_mode: str = "preserve"
    # The released HDP transform is unsafe for a low-speed absolute
    # x-start trajectory unless its offset starts at zero.  Keep the
    # low-speed guard explicit so an old replay contract can distinguish a
    # historical cache from a cache mined with the corrected sampler.
    hdp_trajectory_augmentation_skip_low_speed: bool = True
    hdp_trajectory_augmentation_low_speed_threshold: float = 2.0
    # The released HDP condition is tied to the first five global epochs.
    # Long T4 replay cycles may instead re-enable the same transform whenever
    # a new cache is mined, so later K-sample bundles do not collapse merely
    # because the global epoch is greater than five.
    hdp_trajectory_augmentation_schedule: str = "first_five_epochs"
    # Conservative original-DP adaptation of HDP's group AWR: retain the
    # deterministic behavior target and only regress toward sampled members
    # whose reward is strictly above that behavior reward.
    positive_advantage_only: bool = False
    # Ignore numerically tiny reward wins.  A positive-only AWR update with a
    # zero margin can turn ordinary sampler noise on otherwise-good scenes
    # into a global drift signal; a small margin keeps the full-data run
    # focused on meaningful repairs while retaining true safety recoveries.
    positive_advantage_margin: float = 0.0
    # Candidate zero is a deterministic retention anchor.  It can receive a
    # different weight when its hard-gate reward is zero and a sampled repair
    # exists; all-zero groups can be removed because they contain no ranking
    # direction at all.
    behavior_anchor_weight: float = 1.0
    unsafe_behavior_anchor_weight: float = 1.0
    drop_all_zero_groups: bool = False
    # PlannerRFT-style low-reward candidate enrichment.  Native K is always
    # generated and scored first.  Only groups whose best native reward is
    # below the trigger receive a second guided K bank; the top K of the union
    # become AWR targets.  This guarantees that enrichment cannot lower the
    # final group's best reward (the original native-best trajectory itself
    # may be replaced by a higher-reward guided member) while concentrating
    # extra sampling on the paper's Lt90 regime.  The frozen reference model
    # itself is supplied by the trainer.
    plannerrft_guided_exploration: bool = False
    plannerrft_trigger_best_reward_lt: float = 0.9
    plannerrft_guided_candidates: int = 10
    # Pad the triggered scene subset to a small fixed multiple.  This bounds
    # torch.compile shape variants without guiding all ordinary scenes.
    plannerrft_guidance_bucket_size: int = 32
    plannerrft_eta_scheme: str = "stratified_beta"
    plannerrft_beta_concentration: float = 1.0 + math.log(2.0)
    # Supplementary Algorithm 1 samples one frozen-reference trajectory from
    # Gaussian noise.  The formal Original-DP Cycle 1 instead uses the
    # deployable zero-noise reference.  Keep both as explicit candidate-
    # generation choices so a paired audit can select later refreshes without
    # changing deployment or replay semantics.
    plannerrft_reference_mode: str = "zero_noise"
    plannerrft_reference_noise_scale: float = 0.5
    # The paper's printed longitudinal energy is ambiguous at eta_lon=0.
    # ``candidate_stretch`` is the audited Cycle-1 zero-centred surrogate;
    # ``reference_speed_push`` is an opt-in zero-centred tangent push derived
    # from the frozen reference trajectory.
    plannerrft_longitudinal_mode: str = "candidate_stretch"
    plannerrft_lambda_lat: float = 1.0
    plannerrft_lambda_lon: float = 0.25
    plannerrft_guidance_scale: float = 2.0
    plannerrft_ramp_steps: int = 20
    # Inference uses the same bf16 autocast path as the HDP decoder rollout
    # when the caller enables it.  Keeping this on the rollout config makes
    # baseline and post-training evaluation use identical numerics.
    inference_amp_dtype: str = "off"
    # Original-DP-specific target quality guard.  It is enabled for new
    # mining runs; legacy replay contracts may omit the field because stored
    # trajectories/weights are not retroactively changed.
    # Candidate 0 is always retained as the behavior anchor.  Other low-speed
    # candidates are zero-weighted when their first 0.1 s waypoint is too far,
    # lateral, backward, or nearly perpendicular to the current heading.
    original_dp_first_waypoint_gate_enabled: bool = True
    original_dp_first_waypoint_gate_speed_mps: float = 1.0
    original_dp_first_waypoint_gate_max_step_m: float = 0.25
    original_dp_first_waypoint_gate_max_lateral_m: float = 0.20
    original_dp_first_waypoint_gate_max_backward_m: float = 0.05
    original_dp_first_waypoint_gate_max_tangent_deg: float = 75.0
    # The tangent test only applies once the first step is a real displacement.
    # Below this floor the path direction is the direction of millimeter-level
    # sampler noise: at standstill the model emits x=0 exactly, making the
    # tangent 90 degrees for every candidate and silently discarding the whole
    # low-speed exploration group (audited on the 2026-07-23 Cycle-1 cache:
    # 6.86M of 6.99M rejections had first steps under 5 cm while the expert GT
    # itself failed the unfloored test in 35% of low-speed scenes).
    # 5 mm is genuine sampler/quantisation noise; 1-4 cm is the standstill jitter
    # regime we must still police.  The original 5 cm floor sat *above* the
    # measured stop-turn p95 first step (3.9 cm), so it silenced the tangent test
    # exactly where steering jitter lives and drove the low-speed rejection rate
    # to zero across cycles 1-4.
    original_dp_first_waypoint_gate_tangent_min_step_m: float = 0.005
    # Absolute displacement/lateral limits cannot express steering feasibility:
    # a 1 cm lateral offset over a 4 cm step implies a ~8 cm turn radius, i.e. a
    # steering-lock command, while passing a 20 cm lateral threshold untouched.
    # Gate the implied front-wheel angle instead — delta = atan(wheel_base * k)
    # with k = 2 * |y| / s**2 — which is what the driver feels at the wheel.
    original_dp_first_waypoint_gate_max_implied_steer_rad: float = 0.64
    original_dp_stop_turn_speed_mps: float = 1.0


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


def _apply_hdp_trajectory_augmentation(
    trajectories: torch.Tensor,
    *,
    std: float,
    deterministic_first: bool,
    ramp_steps: int,
    heading_mode: str = "preserve",
    augmentation_enabled: torch.Tensor | bool | None = None,
) -> torch.Tensor:
    """Apply HDP route-frame offsets, optionally with a DP-safe onset."""

    if std <= 0.0:
        return trajectories
    future_len = int(trajectories.shape[-2])
    leading_shape = tuple(int(value) for value in trajectories.shape[:-2])
    if not leading_shape:
        raise ValueError("trajectory augmentation requires a candidate dimension")
    a = torch.randn(*leading_shape, 1, device=trajectories.device) * float(std)
    b = torch.randn(*leading_shape, 1, device=trajectories.device) * float(std)
    if deterministic_first:
        # Candidate is the penultimate dimension of a/b: [K,1] or [B,K,1].
        a.select(dim=-2, index=0).zero_()
        b.select(dim=-2, index=0).zero_()

    if augmentation_enabled is not None:
        enabled = torch.as_tensor(
            augmentation_enabled, device=trajectories.device, dtype=torch.bool
        )
        if enabled.numel() == 1:
            enabled = enabled.reshape(*([1] * len(leading_shape)))
        elif tuple(enabled.shape) == leading_shape:
            pass
        elif (
            len(leading_shape) >= 2
            and tuple(enabled.shape) == leading_shape[:-1]
        ):
            # A scene-level mask is supplied for [B,K,T,4] candidates.
            enabled = enabled.unsqueeze(-1)
        else:
            raise ValueError(
                "augmentation_enabled must be scalar, scene-shaped, or "
                f"candidate-shaped; got {tuple(enabled.shape)} for leading "
                f"trajectory shape {leading_shape}"
            )
        a = torch.where(enabled[..., None], a, torch.zeros_like(a))
        b = torch.where(enabled[..., None], b, torch.zeros_like(b))

    ramp_steps = max(0, int(ramp_steps))
    if ramp_steps > 0:
        u = (
            torch.arange(
                1,
                future_len + 1,
                device=trajectories.device,
                dtype=trajectories.dtype,
            )
            / float(ramp_steps)
        ).clamp(0.0, 1.0)
        # Minimum-jerk interpolation: zero velocity/acceleration at both ends.
        ramp = u.pow(3) * (10.0 - 15.0 * u + 6.0 * u.pow(2))
        ramp = ramp.view(*([1] * len(leading_shape)), future_len)
        a = a * ramp
        b = b * ramp

    cos_yaw = trajectories[..., 2]
    sin_yaw = trajectories[..., 3]
    x_new = trajectories[..., 0] + a * cos_yaw - b * sin_yaw
    y_new = trajectories[..., 1] + a * sin_yaw + b * cos_yaw
    heading = trajectories[..., 2:4]
    heading_mode = str(heading_mode).lower()
    if heading_mode not in {"preserve", "velocity_offset", "tangent"}:
        raise ValueError(
            "hdp trajectory augmentation heading mode must be preserve, "
            f"velocity_offset, or tangent; got {heading_mode!r}"
        )
    if ramp_steps > 0 and heading_mode == "tangent":
        xy_new = torch.stack([x_new, y_new], dim=-1)
        origin = torch.zeros_like(xy_new[..., :1, :])
        tangent = torch.diff(torch.cat([origin, xy_new], dim=-2), dim=-2)
        tangent_norm = tangent.norm(dim=-1, keepdim=True)
        tangent_heading = tangent / tangent_norm.clamp_min(1e-5)
        heading = torch.where(tangent_norm > 1e-4, tangent_heading, heading)
    elif ramp_steps > 0 and heading_mode == "velocity_offset":
        # Do not infer yaw from a one-frame difference of stochastic x/y:
        # sub-centimetre sample jitter becomes large yaw-rate noise at 10 Hz.
        # Instead, retain the model's smooth heading, estimate only its scalar
        # step length from x/y, and add the analytically smooth offset step.
        xy = trajectories[..., :2]
        xy_new = torch.stack([x_new, y_new], dim=-1)
        origin = torch.zeros_like(xy[..., :1, :])
        base_step = torch.diff(torch.cat([origin, xy], dim=-2), dim=-2)
        base_step_length = base_step.norm(dim=-1, keepdim=True)
        heading_unit = heading / heading.norm(dim=-1, keepdim=True).clamp_min(1e-5)
        offset = xy_new - xy
        offset_step = torch.diff(
            torch.cat([torch.zeros_like(offset[..., :1, :]), offset], dim=-2),
            dim=-2,
        )
        augmented_step = heading_unit * base_step_length + offset_step
        augmented_norm = augmented_step.norm(dim=-1, keepdim=True)
        augmented_heading = augmented_step / augmented_norm.clamp_min(1e-5)
        heading = torch.where(augmented_norm > 1e-4, augmented_heading, heading_unit)
    result = torch.cat([torch.stack([x_new, y_new], dim=-1), heading], dim=-1)
    if deterministic_first:
        # Recomputing tangent heading for the ramped proposals must never
        # touch the deploy-time behavior anchor, even when the model's own
        # heading is not exactly equal to the finite-difference path tangent.
        result.select(dim=-3, index=0).copy_(
            trajectories.select(dim=-3, index=0)
        )
    return result


def _scene_speed_mps(
    data: dict[str, torch.Tensor], *, device: torch.device
) -> torch.Tensor | None:
    """Return absolute ego speed for one scene or a scene batch.

    The T4 Original-DP loader stores longitudinal/lateral velocity at
    ``ego_current_state[..., 4:6]``.  The sampler only needs the longitudinal
    value used by the existing SFT augmentation guard; keeping this helper
    shape-preserving avoids copying the full scene.
    """

    state = data.get("ego_current_state")
    if not isinstance(state, torch.Tensor) or state.shape[-1] <= 4:
        return None
    return state[..., 4].detach().to(device=device, dtype=torch.float32).abs()


def _augmentation_enabled_for_scene(
    data: dict[str, torch.Tensor],
    *,
    skip_low_speed: bool,
    low_speed_threshold: float,
    device: torch.device,
) -> torch.Tensor | bool:
    """Resolve the scene-level HDP offset mask without touching scene data."""

    if not skip_low_speed:
        return True
    speed = _scene_speed_mps(data, device=device)
    if speed is None:
        return True
    return speed >= float(low_speed_threshold)


def _stop_turn_scene(
    data: dict[str, torch.Tensor], *, speed_threshold: float
) -> bool:
    """Classify the diagnostic slice ``stopped/slow + left/right command``."""

    speed = _scene_speed_mps(data, device=torch.device("cpu"))
    if speed is None or float(speed.reshape(-1)[0].item()) >= float(speed_threshold):
        return False
    indicators = data.get("turn_indicators")
    if not isinstance(indicators, torch.Tensor) or indicators.numel() == 0:
        return False
    values = indicators.detach().reshape(-1)
    return int(values[-1].item()) in {2, 3}


def _wheel_base_m(
    data: dict[str, torch.Tensor], *, default: float = 2.79
) -> float:
    """Ego wheel base from ``ego_shape`` = (wheel_base, length, width)."""

    shape = data.get("ego_shape")
    if not isinstance(shape, torch.Tensor) or shape.numel() < 1:
        return float(default)
    value = float(shape.detach().reshape(-1)[0].item())
    return value if math.isfinite(value) and value > 0.0 else float(default)


def _implied_first_step_steer_rad(
    first_xy: torch.Tensor,
    step_norm: torch.Tensor,
    data: dict[str, torch.Tensor],
    *,
    min_step_m: float,
) -> torch.Tensor:
    """Front-wheel angle implied by the first step's lateral offset.

    A first step of arc length ``s`` that ends ``y`` off the current heading
    sits on a circle of curvature ``k = 2 * |y| / s**2``; the bicycle model
    turns that into ``delta = atan(wheel_base * k)``.  This is scale-aware in
    the way absolute displacement/lateral thresholds are not: at standstill a
    1 cm lateral offset over a 4 cm step is a steering-lock command even though
    both magnitudes look negligible on their own.

    Steps shorter than ``min_step_m`` are pure numerical noise — their geometry
    is meaningless, so they report ``0`` and are left to the caller's own
    measurability mask rather than being flagged.
    """

    safe_step = step_norm.clamp_min(float(min_step_m))
    curvature = 2.0 * first_xy[..., 1].abs() / safe_step.pow(2)
    steer = torch.atan(_wheel_base_m(data) * curvature)
    return torch.where(
        step_norm >= float(min_step_m), steer, torch.zeros_like(steer)
    )


def _first_waypoint_gate_and_diagnostics(
    trajectories: torch.Tensor,
    data: dict[str, torch.Tensor],
    config: AWRRolloutConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Audit/gate the first 0.1 s of Original-DP candidate trajectories.

    The gate is intentionally local to the first waypoint and only active for
    low-speed scenes.  It is not a lane-keeping or turn classifier.  A normal
    lane change later in the horizon is untouched.  Candidate 0 is the exact
    behavior anchor and is never rejected, so enabling this guard cannot erase
    the incumbent policy from a group.
    """

    if trajectories.ndim != 3 or trajectories.shape[-1] != 4:
        raise ValueError(
            "first-waypoint gate expects [K,T,4], got "
            f"{tuple(trajectories.shape)}"
        )
    count = int(trajectories.shape[0])
    valid = torch.ones(count, device=trajectories.device, dtype=torch.bool)
    first_xy = trajectories[:, 0, :2].float()
    step_norm = first_xy.norm(dim=-1)
    lateral_abs = first_xy[:, 1].abs()
    backward = first_xy[:, 0] < -float(
        config.original_dp_first_waypoint_gate_max_backward_m
    )
    tangent_abs_deg = torch.rad2deg(torch.atan2(first_xy[:, 1].abs(), first_xy[:, 0]))
    implied_steer_rad = _implied_first_step_steer_rad(
        first_xy,
        step_norm,
        data,
        min_step_m=float(
            config.original_dp_first_waypoint_gate_tangent_min_step_m
        ),
    )
    speed = _scene_speed_mps(data, device=trajectories.device)
    speed_value = float(speed.reshape(-1)[0].item()) if speed is not None else float("nan")
    low_speed = (
        math.isfinite(speed_value)
        and speed_value < float(config.original_dp_first_waypoint_gate_speed_mps)
    )
    gate_enabled = bool(config.original_dp_first_waypoint_gate_enabled)
    rejected = torch.zeros_like(valid)
    if gate_enabled and low_speed and count > 1:
        tangent_measurable = step_norm >= float(
            config.original_dp_first_waypoint_gate_tangent_min_step_m
        )
        rejected[1:] = (
            (step_norm[1:] > float(config.original_dp_first_waypoint_gate_max_step_m))
            | (
                lateral_abs[1:]
                > float(config.original_dp_first_waypoint_gate_max_lateral_m)
            )
            | backward[1:]
            | (
                tangent_measurable[1:]
                & (
                    tangent_abs_deg[1:]
                    > float(config.original_dp_first_waypoint_gate_max_tangent_deg)
                )
            )
            | (
                tangent_measurable[1:]
                & (
                    implied_steer_rad[1:]
                    > float(
                        config.original_dp_first_waypoint_gate_max_implied_steer_rad
                    )
                )
            )
        )
        valid &= ~rejected

    stop_turn = _stop_turn_scene(
        data, speed_threshold=float(config.original_dp_stop_turn_speed_mps)
    )
    finite = torch.isfinite(step_norm) & torch.isfinite(tangent_abs_deg)
    finite_steps = step_norm[finite].detach().cpu().numpy()
    finite_tangent = tangent_abs_deg[finite].detach().cpu().numpy()
    diagnostics = {
        "first_waypoint_gate_enabled": float(gate_enabled),
        "first_waypoint_gate_low_speed": float(low_speed),
        "first_waypoint_gate_rejected_fraction": float(rejected.float().mean().item()),
        "first_waypoint_gate_rejected_count": float(rejected.sum().item()),
        "first_waypoint_displacement_p95_m": (
            float(np.percentile(finite_steps, 95)) if finite_steps.size else float("nan")
        ),
        "first_waypoint_lateral_abs_p95_m": (
            float(np.percentile(lateral_abs[finite].detach().cpu().numpy(), 95))
            if bool(finite.any())
            else float("nan")
        ),
        "first_waypoint_implied_steer_p95_rad": (
            float(np.percentile(implied_steer_rad[finite].detach().cpu().numpy(), 95))
            if bool(finite.any())
            else float("nan")
        ),
        "first_waypoint_path_tangent_abs_deg_p95": (
            float(np.percentile(finite_tangent, 95))
            if finite_tangent.size
            else float("nan")
        ),
        "first_waypoint_backward_fraction": float(backward.float().mean().item()),
        "stop_turn_scene": float(stop_turn),
        "stop_turn_first_waypoint_displacement_p95_m": (
            float(np.percentile(finite_steps, 95))
            if stop_turn and finite_steps.size
            else float("nan")
        ),
        "stop_turn_first_waypoint_path_tangent_abs_deg_p95": (
            float(np.percentile(finite_tangent, 95))
            if stop_turn and finite_tangent.size
            else float("nan")
        ),
        "stop_turn_first_waypoint_gate_rejected_fraction": (
            float(rejected.float().mean().item()) if stop_turn else float("nan")
        ),
        # The headline jitter metric: implied steering angle at standstill is
        # what the driver feels at the wheel.
        "stop_turn_first_waypoint_implied_steer_p95_rad": (
            float(np.percentile(implied_steer_rad[finite].detach().cpu().numpy(), 95))
            if stop_turn and bool(finite.any())
            else float("nan")
        ),
    }
    return valid, diagnostics


def _first_waypoint_gate_and_diagnostics_batch(
    trajectories: torch.Tensor,
    data: dict[str, torch.Tensor],
    config: AWRRolloutConfig,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    """Vectorized first-waypoint audit for the full AWR scene batch.

    Full-corpus mining processes millions of groups.  The scene-wise helper is
    intentionally easy to test, but calling it in a Python loop would force a
    CUDA-to-CPU synchronization and a NumPy percentile for every scene.  This
    batch path keeps all geometry on device and transfers only the tiny
    per-scene diagnostic vectors once.
    """

    if trajectories.ndim != 4 or trajectories.shape[-1] != 4:
        raise ValueError(
            "batched first-waypoint gate expects [B,K,T,4], got "
            f"{tuple(trajectories.shape)}"
        )
    batch_size, count = int(trajectories.shape[0]), int(trajectories.shape[1])
    first_xy = trajectories[:, :, 0, :2].float()
    step_norm = first_xy.norm(dim=-1)
    lateral_abs = first_xy[:, :, 1].abs()
    backward = first_xy[:, :, 0] < -float(
        config.original_dp_first_waypoint_gate_max_backward_m
    )
    tangent_abs_deg = torch.rad2deg(
        torch.atan2(first_xy[:, :, 1].abs(), first_xy[:, :, 0])
    )
    implied_steer_rad = _implied_first_step_steer_rad(
        first_xy,
        step_norm,
        data,
        min_step_m=float(
            config.original_dp_first_waypoint_gate_tangent_min_step_m
        ),
    )
    speed = _scene_speed_mps(data, device=trajectories.device)
    if speed is None:
        low_speed = torch.zeros(
            batch_size, device=trajectories.device, dtype=torch.bool
        )
    else:
        low_speed = speed.reshape(batch_size) < float(
            config.original_dp_first_waypoint_gate_speed_mps
        )
    rejected = torch.zeros(
        (batch_size, count), device=trajectories.device, dtype=torch.bool
    )
    if bool(config.original_dp_first_waypoint_gate_enabled) and count > 1:
        tangent_measurable = step_norm >= float(
            config.original_dp_first_waypoint_gate_tangent_min_step_m
        )
        rejected[:, 1:] = low_speed[:, None] & (
            (step_norm[:, 1:] > float(config.original_dp_first_waypoint_gate_max_step_m))
            | (
                lateral_abs[:, 1:]
                > float(config.original_dp_first_waypoint_gate_max_lateral_m)
            )
            | backward[:, 1:]
            | (
                tangent_measurable[:, 1:]
                & (
                    tangent_abs_deg[:, 1:]
                    > float(config.original_dp_first_waypoint_gate_max_tangent_deg)
                )
            )
            | (
                tangent_measurable[:, 1:]
                & (
                    implied_steer_rad[:, 1:]
                    > float(
                        config.original_dp_first_waypoint_gate_max_implied_steer_rad
                    )
                )
            )
        )
    valid = ~rejected

    indicators = data.get("turn_indicators")
    if isinstance(indicators, torch.Tensor) and indicators.numel() > 0:
        current_turn = indicators.reshape(batch_size, -1)[:, -1].long()
        stop_turn = low_speed & ((current_turn == 2) | (current_turn == 3))
    else:
        stop_turn = torch.zeros_like(low_speed)

    finite = torch.isfinite(step_norm) & torch.isfinite(tangent_abs_deg)
    finite_step = torch.where(finite, step_norm, torch.zeros_like(step_norm))
    finite_lateral = torch.where(
        finite, lateral_abs, torch.zeros_like(lateral_abs)
    )
    finite_tangent = torch.where(
        finite, tangent_abs_deg, torch.zeros_like(tangent_abs_deg)
    )
    finite_steer = torch.where(
        finite, implied_steer_rad, torch.zeros_like(implied_steer_rad)
    )
    steer_p95 = torch.quantile(finite_steer, 0.95, dim=-1).detach().cpu().tolist()
    step_p95 = torch.quantile(finite_step, 0.95, dim=-1).detach().cpu().tolist()
    lateral_p95 = (
        torch.quantile(finite_lateral, 0.95, dim=-1).detach().cpu().tolist()
    )
    tangent_p95 = (
        torch.quantile(finite_tangent, 0.95, dim=-1).detach().cpu().tolist()
    )
    rejected_fraction = rejected.float().mean(dim=-1).detach().cpu().tolist()
    rejected_count = rejected.sum(dim=-1).detach().cpu().tolist()
    backward_fraction = backward.float().mean(dim=-1).detach().cpu().tolist()
    low_speed_cpu = low_speed.detach().cpu().tolist()
    stop_turn_cpu = stop_turn.detach().cpu().tolist()
    diagnostics: list[dict[str, float]] = []
    for index in range(batch_size):
        is_stop_turn = bool(stop_turn_cpu[index])
        diagnostics.append(
            {
                "first_waypoint_gate_enabled": float(
                    bool(config.original_dp_first_waypoint_gate_enabled)
                ),
                "first_waypoint_gate_low_speed": float(low_speed_cpu[index]),
                "first_waypoint_gate_rejected_fraction": float(
                    rejected_fraction[index]
                ),
                "first_waypoint_gate_rejected_count": float(rejected_count[index]),
                "first_waypoint_displacement_p95_m": float(step_p95[index]),
                "first_waypoint_lateral_abs_p95_m": float(lateral_p95[index]),
                "first_waypoint_implied_steer_p95_rad": float(steer_p95[index]),
                "first_waypoint_path_tangent_abs_deg_p95": float(
                    tangent_p95[index]
                ),
                "first_waypoint_backward_fraction": float(
                    backward_fraction[index]
                ),
                "stop_turn_scene": float(is_stop_turn),
                "stop_turn_first_waypoint_displacement_p95_m": (
                    float(step_p95[index]) if is_stop_turn else float("nan")
                ),
                "stop_turn_first_waypoint_path_tangent_abs_deg_p95": (
                    float(tangent_p95[index]) if is_stop_turn else float("nan")
                ),
                "stop_turn_first_waypoint_gate_rejected_fraction": (
                    float(rejected_fraction[index])
                    if is_stop_turn
                    else float("nan")
                ),
                "stop_turn_first_waypoint_implied_steer_p95_rad": (
                    float(steer_p95[index]) if is_stop_turn else float("nan")
                ),
            }
        )
    return valid, diagnostics


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
        # ``_cached_encoding`` is already a detached, normalized encoder
        # output.  Do not clone it with the raw observation dictionary: a
        # full rollout batch contains hundreds of MiB of encoding tokens and
        # the decoder will create the one required K-way expansion below.
        cached_encoding = data.get("_cached_encoding")
        norm_single = model_args.observation_normalizer(
            {key: value for key, value in data.items() if key != "_cached_encoding"}
        )
        with _inference_autocast(device, config.inference_amp_dtype):
            if cached_encoding is None:
                scene_encoding = planner.encoder(norm_single).detach()
            else:
                if not isinstance(cached_encoding, torch.Tensor):
                    raise TypeError("_cached_encoding must be a tensor")
                if cached_encoding.dim() != 3 or cached_encoding.shape[0] != 1:
                    raise ValueError(
                        "single-scene cached encoding must have shape [1,L,D], "
                        f"got {tuple(cached_encoding.shape)}"
                    )
                if cached_encoding.device != device:
                    raise ValueError(
                        "cached encoding was not moved with its scene batch: "
                        f"{cached_encoding.device} != {device}"
                    )
                scene_encoding = cached_encoding.detach()
            norm_batch = expand_scene_batch(
                _decoder_inference_scene_inputs(
                    norm_single, int(model_args.predicted_neighbor_num)
                ),
                K,
            )
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
        # Independent scalar a/b per candidate, transformed in the
        # candidate's own heading frame. Candidate 0 remains the exact
        # deploy-time behavior anchor. A zero ramp reproduces HDP's constant
        # offset; the original-DP profile uses a state-continuous onset.
        trajectories = _apply_hdp_trajectory_augmentation(
            trajectories,
            std=float(config.hdp_trajectory_augmentation_std),
            deterministic_first=bool(config.deterministic_first),
            ramp_steps=int(config.hdp_trajectory_augmentation_ramp_steps),
            heading_mode=config.hdp_trajectory_augmentation_heading_mode,
            augmentation_enabled=_augmentation_enabled_for_scene(
                data,
                skip_low_speed=bool(
                    config.hdp_trajectory_augmentation_skip_low_speed
                ),
                low_speed_threshold=float(
                    config.hdp_trajectory_augmentation_low_speed_threshold
                ),
                device=device,
            ),
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


def _decoder_inference_scene_inputs(
    normalized_scene: dict[str, Any],
    predicted_neighbor_num: int,
) -> dict[str, torch.Tensor]:
    """Keep only tensors consumed by unguided cached-encoder inference."""

    required = ("ego_current_state", "neighbor_agents_past", "delay")
    missing = [key for key in required if key not in normalized_scene]
    if missing:
        raise KeyError(f"decoder inference scene is missing {missing}")
    ego = normalized_scene["ego_current_state"]
    neighbors = normalized_scene["neighbor_agents_past"]
    delay = normalized_scene["delay"]
    if not all(isinstance(value, torch.Tensor) for value in (ego, neighbors, delay)):
        raise TypeError("decoder inference ego, neighbors and delay must be tensors")
    if ego.dim() < 2 or neighbors.dim() != 4:
        raise ValueError(
            "unexpected decoder current-state shapes: "
            f"ego={tuple(ego.shape)}, neighbors={tuple(neighbors.shape)}"
        )
    neighbor_count = int(predicted_neighbor_num)
    if neighbors.shape[1] < neighbor_count:
        raise ValueError(
            "decoder scene has fewer neighbors than the model predicts: "
            f"{neighbors.shape[1]} < {neighbor_count}"
        )
    # Decoder._prepare_current_states reads exactly these four channels and
    # only the final history frame. Guidance is explicitly disabled above,
    # and the turn-indicator head is skipped below, so map/route/encoder-only
    # tensors must not be expanded K times.
    return {
        "ego_current_state": ego[..., :4],
        "neighbor_agents_past": neighbors[:, :neighbor_count, -1:, :4],
        "delay": delay,
    }


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
        cached_encoding = data.get("_cached_encoding")
        norm_scene = model_args.observation_normalizer(
            {key: value for key, value in data.items() if key != "_cached_encoding"}
        )
        with _inference_autocast(device, config.inference_amp_dtype):
            if cached_encoding is None:
                scene_encoding = planner.encoder(norm_scene).detach()
            else:
                if not isinstance(cached_encoding, torch.Tensor):
                    raise TypeError("_cached_encoding must be a tensor")
                if (
                    cached_encoding.dim() != 3
                    or cached_encoding.shape[0] != batch_size
                ):
                    raise ValueError(
                        "batched cached encoding must have shape [B,L,D], "
                        f"B={batch_size}, got {tuple(cached_encoding.shape)}"
                    )
                if cached_encoding.device != device:
                    raise ValueError(
                        "cached encoding was not moved with its scene batch: "
                        f"{cached_encoding.device} != {device}"
                    )
                scene_encoding = cached_encoding.detach()
            norm_candidates = _repeat_scene_batch(
                _decoder_inference_scene_inputs(
                    norm_scene, int(model_args.predicted_neighbor_num)
                ),
                K,
            )
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
        trajectories = _apply_hdp_trajectory_augmentation(
            trajectories,
            std=float(config.hdp_trajectory_augmentation_std),
            deterministic_first=bool(config.deterministic_first),
            ramp_steps=int(config.hdp_trajectory_augmentation_ramp_steps),
            heading_mode=config.hdp_trajectory_augmentation_heading_mode,
            augmentation_enabled=_augmentation_enabled_for_scene(
                data,
                skip_low_speed=bool(
                    config.hdp_trajectory_augmentation_skip_low_speed
                ),
                low_speed_threshold=float(
                    config.hdp_trajectory_augmentation_low_speed_threshold
                ),
                device=device,
            ),
        )
    return trajectories, noise_scales, scene_encoding


def _index_scene_batch(
    data: dict[str, Any], indices: torch.Tensor
) -> dict[str, Any]:
    """Select arbitrary scene rows while preserving non-batched metadata."""

    batch_size = int(data["ego_current_state"].shape[0])
    return {
        key: value.index_select(0, indices)
        if isinstance(value, torch.Tensor)
        and value.dim() > 0
        and value.shape[0] == batch_size
        else value
        for key, value in data.items()
    }


def _sample_fixed_beta_eta_batch(
    batch_size: int,
    group_size: int,
    *,
    concentration: float,
    scheme: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw PlannerRFT fixed-Beta commands, with optional CDF stratification.

    PlannerRFT's fixed-explorer ablation uses the non-learned distribution at
    zero initialization, but the paper does not publish its concentration
    parameterization.  This repository's exploration head maps zero logits to
    ``alpha=beta=softplus(0)+1``; the fixed AWR sampler deliberately reuses
    that local prior.  ``iid_beta`` draws independently from it.
    ``stratified_beta`` keeps one draw in every equal-probability CDF stratum
    per axis, then independently permutes each scene's lateral/longitudinal
    values.  The latter is a local variance-reduction adaptation: it retains
    the fixed-Beta support while avoiding K=10 groups that accidentally miss
    an entire yield/accelerate/left/right region.
    """

    if batch_size < 1 or group_size < 1:
        raise ValueError(
            f"invalid fixed-Beta shape B={batch_size}, K={group_size}"
        )
    if concentration <= 0.0:
        raise ValueError(f"Beta concentration must be positive, got {concentration}")
    if scheme == "iid_beta":
        distribution = torch.distributions.Beta(
            torch.tensor(float(concentration), device=device),
            torch.tensor(float(concentration), device=device),
        )
        lat = distribution.sample((batch_size, group_size))
        lon = distribution.sample((batch_size, group_size))
        return 2.0 * lat - 1.0, 2.0 * lon - 1.0
    if scheme != "stratified_beta":
        raise ValueError(f"unknown PlannerRFT eta scheme {scheme!r}")

    # scipy is already present in the planner environment.  Keep the inverse
    # CDF on CPU: at B=192,K=10 this is tiny relative to one DiT rollout and
    # avoids a coarse lookup-table approximation changing the audited Beta.
    from scipy.special import betaincinv

    q_lat = (
        np.arange(group_size, dtype=np.float64)[None, :]
        + np.random.random((batch_size, group_size))
    ) / float(group_size)
    q_lon = (
        np.arange(group_size, dtype=np.float64)[None, :]
        + np.random.random((batch_size, group_size))
    ) / float(group_size)
    lat = betaincinv(float(concentration), float(concentration), q_lat)
    lon = betaincinv(float(concentration), float(concentration), q_lon)
    for row in range(batch_size):
        np.random.shuffle(lat[row])
        np.random.shuffle(lon[row])
    lat_tensor = torch.as_tensor(lat, dtype=torch.float32, device=device)
    lon_tensor = torch.as_tensor(lon, dtype=torch.float32, device=device)
    return 2.0 * lat_tensor - 1.0, 2.0 * lon_tensor - 1.0


@torch.no_grad()
def sample_plannerrft_guided_dp_group_batch(
    model: nn.Module,
    model_args: Config,
    reference_model: nn.Module,
    reference_model_args: Config,
    data: dict[str, torch.Tensor],
    scene_encoding: torch.Tensor,
    config: AWRRolloutConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate an independent guided candidate bank around a frozen IL prior.

    Returns guided trajectories ``[B,G,T,4]``, noise scales ``[B,G]`` and the
    sampled lateral/longitudinal eta tensors ``[B,G]``.  The original-DP
    lateral target uses the audited 2-second quintic onset, so the first plan
    point starts at the current state instead of inheriting HDP's constant
    action/anchor offset.
    """

    if reference_model is None:
        raise RuntimeError(
            "PlannerRFT guided exploration requires a frozen reference model"
        )
    batch_size = int(data["ego_current_state"].shape[0])
    guided_count = int(config.plannerrft_guided_candidates)
    if guided_count < 1:
        raise ValueError("plannerrft_guided_candidates must be >= 1")
    if scene_encoding.shape[0] != batch_size:
        raise ValueError(
            "guided scene encoding batch mismatch: "
            f"encoding={tuple(scene_encoding.shape)}, B={batch_size}"
        )

    # The global reference is an untouched, deterministic IL planner.  Never
    # reuse the current policy's cached encoding: the encoder weights are part
    # of the reference contract and may differ from a later AWR checkpoint.
    reference_config = copy.copy(config)
    reference_config.n_trajectories = 1
    reference_mode = str(config.plannerrft_reference_mode)
    if reference_mode not in {"zero_noise", "stochastic"}:
        raise ValueError(f"unknown PlannerRFT reference mode {reference_mode!r}")
    reference_config.deterministic_first = reference_mode == "zero_noise"
    if reference_mode == "stochastic":
        reference_noise_scale = float(config.plannerrft_reference_noise_scale)
        if reference_noise_scale <= 0.0:
            raise ValueError(
                "stochastic PlannerRFT reference requires a positive noise scale"
            )
        reference_config.noise_scale_range = (
            reference_noise_scale,
            reference_noise_scale,
        )
    reference_config.structured_exploration = False
    reference_config.hdp_trajectory_augmentation = False
    reference_config.plannerrft_guided_exploration = False
    reference_data = {
        key: value for key, value in data.items() if key != "_cached_encoding"
    }
    reference_trajectories, _, _ = sample_unguided_dp_group_batch(
        reference_model,
        reference_model_args,
        reference_data,
        reference_config,
        device,
    )
    reference = reference_trajectories[:, 0].detach()

    eta_lat, eta_lon = _sample_fixed_beta_eta_batch(
        batch_size,
        guided_count,
        concentration=float(config.plannerrft_beta_concentration),
        scheme=str(config.plannerrft_eta_scheme),
        device=device,
    )
    planner = _model_without_wrappers(model)
    previous_training = planner.training
    previous_guidance_fn = planner.decoder._guidance_fn
    previous_guidance_scale = planner.decoder._guidance_scale
    previous_sample_steps = planner.decoder._sample_steps
    planner.eval()

    noise_min, noise_max = config.noise_scale_range
    noise_scales = torch.empty(
        batch_size, guided_count, device=device, dtype=torch.float32
    ).uniform_(float(noise_min), float(noise_max))
    players = 1 + int(model_args.predicted_neighbor_num)
    future_len = int(model_args.future_len)
    initial_noise = torch.randn(
        batch_size * guided_count,
        players,
        future_len + 1,
        4,
        device=device,
    )

    from rlvr.guidance_batched import build_head_composer, dit_memo

    longitudinal_mode = str(config.plannerrft_longitudinal_mode)
    if longitudinal_mode == "candidate_stretch":
        longitudinal_head = "stretch"
    elif longitudinal_mode == "reference_speed_push":
        longitudinal_head = "reference_speed"
    else:
        raise ValueError(
            f"unknown PlannerRFT longitudinal mode {longitudinal_mode!r}"
        )
    composer = build_head_composer(
        {
            "lateral": eta_lat.reshape(-1),
            longitudinal_head: eta_lon.reshape(-1),
        },
        lambda_lat=float(config.plannerrft_lambda_lat),
        lat_scale=1.0,
        lambda_spd=float(config.plannerrft_lambda_lon),
        lambda_lon=float(config.plannerrft_lambda_lon),
        stretch_scale=1.0,
        guidance_scale=float(config.plannerrft_guidance_scale),
        envelope="v2",
        ramp_steps=int(config.plannerrft_ramp_steps),
        fast=True,
    )
    repeated_reference = reference.repeat_interleave(
        guided_count, dim=0
    ).contiguous()
    # The active lateral+stretch set needs only the physical frozen reference.
    # Avoid inverse-normalising the decoder's compact 4-channel cached context,
    # whose full observation statistics have 10/11 channels.
    composer.set_physical_inputs({"reference_trajectory": repeated_reference})
    try:
        planner.decoder._sample_steps = max(2, int(config.sample_steps))
        planner.decoder._guidance_fn = composer
        planner.decoder._guidance_scale = float(config.plannerrft_guidance_scale)
        norm_scene = model_args.observation_normalizer(
            {key: value for key, value in data.items() if key != "_cached_encoding"}
        )
        norm_candidates = _repeat_scene_batch(
            _decoder_inference_scene_inputs(
                norm_scene, int(model_args.predicted_neighbor_num)
            ),
            guided_count,
        )
        norm_candidates["reference_trajectory"] = repeated_reference
        norm_candidates["sampled_trajectories"] = (
            noise_scales.reshape(-1, 1, 1, 1) * initial_noise
        )
        norm_candidates["_cached_encoding"] = scene_encoding.repeat_interleave(
            guided_count, dim=0
        ).contiguous()
        norm_candidates["_skip_turn_indicator"] = True
        with _inference_autocast(device, config.inference_amp_dtype):
            with dit_memo(planner.decoder):
                _, outputs = planner(norm_candidates)
        trajectories = outputs["prediction"][:, 0].float().detach().reshape(
            batch_size, guided_count, future_len, 4
        )
    finally:
        planner.decoder._guidance_fn = previous_guidance_fn
        planner.decoder._guidance_scale = previous_guidance_scale
        planner.decoder._sample_steps = previous_sample_steps
        planner.train(previous_training)
    return trajectories, noise_scales, eta_lat, eta_lon


def _select_topk_native_guided_union(
    native_trajectories: torch.Tensor,
    native_noise_scales: torch.Tensor,
    native_rewards: list[RewardBreakdown],
    guided_trajectories: torch.Tensor,
    guided_noise_scales: torch.Tensor,
    guided_rewards: list[RewardBreakdown],
    *,
    keep: int,
    preserve_native_index0: bool = False,
    native_valid_mask: torch.Tensor | None = None,
    guided_valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[RewardBreakdown], dict[str, float]]:
    """Reward-rank a native+guided union, preferring native members on ties.

    ``preserve_native_index0`` is the explicit original-DP behavior-anchor
    ablation.  Candidate zero remains first and consumes one slot; the other
    slots are reward-ranked over native[1:] plus guided candidates.  The
    default is false and retains the all-stochastic PlannerRFT/HDP behavior.
    """

    if len(native_rewards) != native_trajectories.shape[0]:
        raise ValueError("native trajectory/reward count mismatch")
    if len(guided_rewards) != guided_trajectories.shape[0]:
        raise ValueError("guided trajectory/reward count mismatch")
    if keep < 1:
        raise ValueError(f"top-K union keep must be positive, got {keep}")
    if native_valid_mask is None:
        native_valid_mask = torch.ones(
            len(native_rewards), dtype=torch.bool, device=native_trajectories.device
        )
    if guided_valid_mask is None:
        guided_valid_mask = torch.ones(
            len(guided_rewards), dtype=torch.bool, device=guided_trajectories.device
        )
    if native_valid_mask.numel() != len(native_rewards):
        raise ValueError("native validity/reward count mismatch")
    if guided_valid_mask.numel() != len(guided_rewards):
        raise ValueError("guided validity/reward count mismatch")
    entries: list[tuple[bool, float, int, int, RewardBreakdown]] = []
    native_start = 1 if preserve_native_index0 else 0
    for index, reward in enumerate(native_rewards[native_start:], start=native_start):
        score = float(reward.total)
        entries.append(
            (
                bool(native_valid_mask[index].item()),
                score if math.isfinite(score) else -math.inf,
                1,
                index,
                reward,
            )
        )
    for index, reward in enumerate(guided_rewards):
        score = float(reward.total)
        entries.append(
            (
                bool(guided_valid_mask[index].item()),
                score if math.isfinite(score) else -math.inf,
                0,
                index,
                reward,
            )
        )
    ranked_keep = keep - 1 if preserve_native_index0 else keep
    # Valid first-waypoint targets always outrank a rejected target.  If a
    # pathological scene has fewer than K valid candidates, fill the group
    # with the rejected tail; its final AWR weight is zero and the shortage is
    # visible in the diagnostics rather than silently changing K.
    selected = sorted(entries, key=lambda row: (row[0], row[1], row[2]), reverse=True)[
        :ranked_keep
    ]
    if preserve_native_index0:
        anchor_reward = native_rewards[0]
        anchor_score = float(anchor_reward.total)
        selected.insert(
            0,
            (
                True,
                anchor_score if math.isfinite(anchor_score) else -math.inf,
                1,
                0,
                anchor_reward,
            ),
        )
    if len(selected) != keep:
        raise RuntimeError(
            f"top-K union selected {len(selected)} candidates, expected {keep}"
        )
    trajectory_bank = torch.cat([native_trajectories, guided_trajectories], dim=0)
    noise_bank = torch.cat([native_noise_scales, guided_noise_scales], dim=0)
    native_count = int(native_trajectories.shape[0])
    indices = torch.as_tensor(
        [
            index if source == 1 else native_count + index
            for _, _, source, index, _ in selected
        ],
        dtype=torch.long,
        device=trajectory_bank.device,
    )
    selected_rewards = [reward for _, _, _, _, reward in selected]
    selected_native = sum(source == 1 for _, _, source, _, _ in selected)
    selected_valid = sum(valid for valid, _, _, _, _ in selected)
    native_scores = [
        float(reward.total)
        for reward, valid in zip(native_rewards, native_valid_mask.tolist())
        if valid and math.isfinite(float(reward.total))
    ]
    guided_scores = [
        float(reward.total)
        for reward, valid in zip(guided_rewards, guided_valid_mask.tolist())
        if valid and math.isfinite(float(reward.total))
    ]
    selected_valid_scores = [
        float(entry[4].total)
        for entry in selected
        if entry[0] and math.isfinite(float(entry[4].total))
    ]
    native_best = max(native_scores) if native_scores else -math.inf
    guided_best = max(guided_scores) if guided_scores else -math.inf
    enriched_best = max(selected_valid_scores) if selected_valid_scores else -math.inf
    if enriched_best + 1e-12 < native_best:
        raise RuntimeError(
            "top-K enrichment lowered the native best reward: "
            f"native={native_best}, enriched={enriched_best}"
        )
    return (
        trajectory_bank.index_select(0, indices),
        noise_bank.index_select(0, indices),
        selected_rewards,
        {
            "guided_exploration_triggered": 1.0,
            "native_candidate_count_generated": float(len(native_rewards)),
            "guided_candidate_count_generated": float(len(guided_rewards)),
            "native_candidate_count_selected": float(selected_native),
            "guided_candidate_count_selected": float(keep - selected_native),
            "first_waypoint_valid_candidate_count_selected": float(selected_valid),
            "first_waypoint_rejected_candidate_count_selected": float(
                keep - selected_valid
            ),
            "behavior_anchor_preserved": float(preserve_native_index0),
            "native_best_reward_before_enrichment": native_best,
            "native_best_reward_before_enrichment_all": max(
                float(reward.total)
                for reward in native_rewards
                if math.isfinite(float(reward.total))
            ),
            "guided_best_reward": guided_best,
            "enriched_best_reward_gain": enriched_best - native_best,
        },
    )


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
    exploration_reference_model: nn.Module | None = None,
    exploration_reference_model_args: Config | None = None,
    expert_trajectories: torch.Tensor | None = None,
    expert_rewards_out: list[RewardBreakdown] | None = None,
) -> list[AWRRollout]:
    """Generate, optionally enrich, then weight each scene's candidate group.

    ``expert_trajectories`` [B, T, C] is an optional extra target per scene --
    the logged expert future -- scored as one additional row of each scene's
    existing reward call, with its breakdown appended to ``expert_rewards_out``
    in scene order and excluded from the candidate group.  A reward call costs
    the same for K rows as for K+1 (measured flat from N=1 to N=80), so this
    replaces a whole second per-scene call with nothing.  The candidate rewards
    are unaffected because the reward reduces over neighbors and timesteps but
    never over the candidate dimension.
    """

    if (expert_trajectories is None) != (expert_rewards_out is None):
        raise ValueError(
            "expert_trajectories and expert_rewards_out must be supplied "
            "together: one without the other would either score an expert "
            "whose breakdown is discarded or return an empty list the caller "
            "would read as 'no expert'"
        )
    if (
        rollout_config.plannerrft_guided_exploration
        and rollout_config.hdp_trajectory_augmentation
    ):
        raise ValueError(
            "PlannerRFT guided enrichment and HDP output-space trajectory "
            "augmentation are alternative exploration mechanisms; do not "
            "silently stack them"
        )
    if (
        rollout_config.plannerrft_guided_exploration
        and rollout_config.positive_advantage_only
    ):
        raise ValueError(
            "PlannerRFT top-union enrichment is group-relative and cannot use "
            "candidate-0 behavior-anchor semantics"
        )

    _stage_start = time.perf_counter()
    trajectories, noise_scales, scene_encoding = sample_unguided_dp_group_batch(
        behavior_model, model_args, data, rollout_config, device
    )
    stage_end("sample_native", _stage_start)
    batch_size = int(trajectories.shape[0])
    reward_data_by_scene: list[dict[str, torch.Tensor]] = []
    rewards_by_scene: list[list[RewardBreakdown]] = []
    _stage_start = time.perf_counter()
    native_valid_batch, first_waypoint_diagnostics_by_scene = (
        _first_waypoint_gate_and_diagnostics_batch(
            trajectories, data, rollout_config
        )
    )
    stage_end("gate_native", _stage_start)
    native_valid_by_scene = [native_valid_batch[index] for index in range(batch_size)]
    if expert_trajectories is not None:
        expected_expert_shape = (batch_size, *tuple(trajectories.shape[2:]))
        if tuple(expert_trajectories.shape) != expected_expert_shape:
            raise ValueError(
                "expert trajectory shape must match one candidate per scene: "
                f"{tuple(expert_trajectories.shape)} != {expected_expert_shape}"
            )
        expert_trajectories = expert_trajectories.to(
            device=trajectories.device, dtype=trajectories.dtype
        )
    _stage_start = time.perf_counter()
    for scene_index in range(batch_size):
        scene_data = _slice_scene_data(data, scene_index)
        scene_reward_data = reward_compatible_data(scene_data)
        reward_data_by_scene.append(scene_reward_data)
        scene_candidates = trajectories[scene_index]
        if expert_trajectories is not None:
            scene_candidates = torch.cat(
                [
                    scene_candidates,
                    expert_trajectories[scene_index : scene_index + 1],
                ],
                dim=0,
            )
        scene_rewards = compute_reward_batch(
            scene_candidates, scene_reward_data, reward_config
        )
        if expert_rewards_out is not None:
            # Trailing row, so every index below still means the same
            # candidate it did before the expert was merged in.
            expert_rewards_out.append(scene_rewards[-1])
            scene_rewards = scene_rewards[:-1]
        rewards_by_scene.append(scene_rewards)
    stage_end("score_native", _stage_start)

    _stage_start = time.perf_counter()
    enrichment_by_scene: dict[int, dict[str, float]] = {}
    if rollout_config.plannerrft_guided_exploration:
        if exploration_reference_model is None or exploration_reference_model_args is None:
            raise RuntimeError(
                "PlannerRFT guided exploration is enabled but the trainer did "
                "not supply a frozen reference model and args"
            )
        threshold = float(rollout_config.plannerrft_trigger_best_reward_lt)
        trigger_indices = [
            index
            for index, rewards in enumerate(rewards_by_scene)
            if (
                not bool(native_valid_by_scene[index].any())
                or max(
                    float(reward.total)
                    for reward, valid in zip(
                        rewards, native_valid_by_scene[index].tolist()
                    )
                    if valid
                )
                < threshold
            )
        ]
        if trigger_indices:
            real_trigger_count = len(trigger_indices)
            bucket_size = max(
                1, int(rollout_config.plannerrft_guidance_bucket_size)
            )
            padded_trigger_count = int(
                math.ceil(real_trigger_count / bucket_size) * bucket_size
            )
            padded_trigger_indices = trigger_indices + [trigger_indices[0]] * (
                padded_trigger_count - real_trigger_count
            )
            index_tensor = torch.as_tensor(
                padded_trigger_indices, dtype=torch.long, device=device
            )
            triggered_data = _index_scene_batch(data, index_tensor)
            triggered_encoding = scene_encoding.index_select(0, index_tensor)
            (
                guided_trajectories,
                guided_noise_scales,
                eta_lat,
                eta_lon,
            ) = sample_plannerrft_guided_dp_group_batch(
                behavior_model,
                model_args,
                exploration_reference_model,
                exploration_reference_model_args,
                triggered_data,
                triggered_encoding,
                rollout_config,
                device,
            )
            guided_valid_batch, guided_first_waypoint_diagnostics_batch = (
                _first_waypoint_gate_and_diagnostics_batch(
                    guided_trajectories, triggered_data, rollout_config
                )
            )
            for guided_row, scene_index in enumerate(trigger_indices):
                scene_data = _slice_scene_data(data, scene_index)
                guided_valid = guided_valid_batch[guided_row]
                guided_first_waypoint_diagnostics = guided_first_waypoint_diagnostics_batch[
                    guided_row
                ]
                guided_rewards = compute_reward_batch(
                    guided_trajectories[guided_row],
                    reward_data_by_scene[scene_index],
                    reward_config,
                )
                (
                    selected_trajectories,
                    selected_noise_scales,
                    selected_rewards,
                    enrichment,
                ) = _select_topk_native_guided_union(
                    trajectories[scene_index],
                    noise_scales[scene_index],
                    rewards_by_scene[scene_index],
                    guided_trajectories[guided_row],
                    guided_noise_scales[guided_row],
                    guided_rewards,
                    keep=int(rollout_config.n_trajectories),
                    preserve_native_index0=bool(
                        rollout_config.deterministic_first
                    ),
                    native_valid_mask=native_valid_by_scene[scene_index],
                    guided_valid_mask=guided_valid,
                )
                trajectories[scene_index] = selected_trajectories
                noise_scales[scene_index] = selected_noise_scales
                rewards_by_scene[scene_index] = selected_rewards
                native_valid_by_scene[scene_index], first_waypoint_diagnostics_by_scene[
                    scene_index
                ] = _first_waypoint_gate_and_diagnostics(
                    selected_trajectories, scene_data, rollout_config
                )
                first_waypoint_diagnostics_by_scene[scene_index].update(
                    {
                        "guided_first_waypoint_gate_rejected_fraction": float(
                            (~guided_valid).float().mean().item()
                        ),
                        **{
                            f"guided_{key}": value
                            for key, value in guided_first_waypoint_diagnostics.items()
                            if key
                            in {
                                "first_waypoint_displacement_p95_m",
                                "first_waypoint_path_tangent_abs_deg_p95",
                                "stop_turn_first_waypoint_displacement_p95_m",
                                "stop_turn_first_waypoint_path_tangent_abs_deg_p95",
                            }
                        },
                    }
                )
                enrichment.update(
                    {
                        "guided_trigger_batch_real_scenes": float(real_trigger_count),
                        "guided_trigger_batch_padded_scenes": float(
                            padded_trigger_count
                        ),
                        "guided_trigger_padding_fraction": float(
                            (padded_trigger_count - real_trigger_count)
                            / padded_trigger_count
                        ),
                        "guided_eta_lat_abs_mean": float(
                            eta_lat[guided_row].abs().mean().item()
                        ),
                        "guided_eta_lon_abs_mean": float(
                            eta_lon[guided_row].abs().mean().item()
                        ),
                        "guided_eta_lat_min": float(eta_lat[guided_row].min().item()),
                        "guided_eta_lat_max": float(eta_lat[guided_row].max().item()),
                        "guided_eta_lon_min": float(eta_lon[guided_row].min().item()),
                        "guided_eta_lon_max": float(eta_lon[guided_row].max().item()),
                    }
                )
                enrichment_by_scene[scene_index] = enrichment
    stage_end("guided", _stage_start)

    _stage_start = time.perf_counter()
    result: list[AWRRollout] = []
    for scene_index in range(batch_size):
        scene_trajectories = trajectories[scene_index]
        scene_reward_data = reward_data_by_scene[scene_index]
        scene_rewards = rewards_by_scene[scene_index]
        weights_cpu, diagnostics = resolve_objective(rollout_config.objective)(
            scene_rewards,
            beta=rollout_config.beta,
            weight_clip=rollout_config.weight_clip,
            normalize=rollout_config.normalize_weights,
            min_group_std=rollout_config.min_group_std,
            safe_only=rollout_config.safe_only,
            reward_config=reward_config,
            positive_advantage_only=rollout_config.positive_advantage_only,
            positive_advantage_margin=rollout_config.positive_advantage_margin,
            behavior_anchor_weight=rollout_config.behavior_anchor_weight,
            unsafe_behavior_anchor_weight=rollout_config.unsafe_behavior_anchor_weight,
            drop_all_zero_groups=rollout_config.drop_all_zero_groups,
            candidate_valid_mask=native_valid_by_scene[scene_index],
        )
        diagnostics = dict(diagnostics)
        diagnostics.update(first_waypoint_diagnostics_by_scene[scene_index])
        diagnostics.update(
            enrichment_by_scene.get(
                scene_index,
                {
                    "guided_exploration_triggered": 0.0,
                    "native_candidate_count_generated": float(
                        rollout_config.n_trajectories
                    ),
                    "guided_candidate_count_generated": 0.0,
                    "native_candidate_count_selected": float(
                        rollout_config.n_trajectories
                    ),
                    "guided_candidate_count_selected": 0.0,
                    "behavior_anchor_preserved": float(
                        rollout_config.deterministic_first
                    ),
                    "native_best_reward_before_enrichment": float(
                        max(reward.total for reward in scene_rewards)
                    ),
                    "guided_best_reward": float("nan"),
                    "enriched_best_reward_gain": 0.0,
                },
            )
        )
        diagnostics["guided_exploration_trigger_threshold"] = float(
            rollout_config.plannerrft_trigger_best_reward_lt
        )
        diagnostics["candidate_count"] = float(scene_trajectories.shape[0])
        diagnostics["candidate0_deterministic"] = float(
            rollout_config.deterministic_first
        )
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
    stage_end("weight", _stage_start)
    return result


def safe_candidate_mask(
    rewards: list[RewardBreakdown], reward_config: RewardConfig | None = None
) -> np.ndarray:
    """The configured terminal gates, not every gate that happens to be logged.

    ``lane_crossing`` is deliberately not an unconditional safety failure: the
    metric is computed against the union of local lane polygons, so a normal
    lane change crosses a *shared* boundary, and its official PDM weight is
    zero.  Road-border and stopped-neighbor gates follow the same rule -- each
    counts only when its own ``*_gate_enabled`` flag is on.
    """

    if reward_config is None:
        reward_config = RewardConfig()
    return np.asarray(
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
    behavior_anchor_weight: float = 1.0,
    unsafe_behavior_anchor_weight: float = 1.0,
    drop_all_zero_groups: bool = False,
    candidate_valid_mask: torch.Tensor | np.ndarray | list[bool] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Convert one scene's rewards to stable positive AWR weights.

    AWR uses ``w_i = exp(beta * (R_i - mean(R)) / (std(R) + eps))`` with the
    sample standard deviation used by HDP.  The HDP paper and local reference
    trainer discard groups whose actions all receive an identical reward,
    because they contain no preference direction.  The default remains
    explicit for old ablations; ``drop_all_zero_groups`` restores that rule
    for the terminal tied stratum used by the formal T4 reward (the full-cache
    audit verifies that its reward-tied groups are exactly the all-zero
    groups).  Optional finite-weight mean normalisation preserves the
    diffusion-loss scale across scenes.
    """

    totals = np.asarray([float(r.total) for r in rewards], dtype=np.float64)
    finite = np.isfinite(totals)
    if candidate_valid_mask is None:
        candidate_valid = np.ones(totals.shape, dtype=bool)
    else:
        candidate_valid = np.asarray(
            candidate_valid_mask.detach().cpu().numpy()
            if isinstance(candidate_valid_mask, torch.Tensor)
            else candidate_valid_mask,
            dtype=bool,
        ).reshape(-1)
        if candidate_valid.size != totals.size:
            raise ValueError(
                "candidate_valid_mask/reward count mismatch: "
                f"{candidate_valid.size} != {totals.size}"
            )
    finite &= candidate_valid
    original_finite = finite.copy()
    invalid_candidate_count = int((~candidate_valid).sum())
    # ``lane_crossing`` is deliberately not an unconditional safety failure.
    # The metric is computed against the union of local lane polygons and a
    # normal lane change crosses a *shared* lane boundary.  HDP/PDM treats
    # drivable-area / collision / direction as safety semantics; lane keeping
    # is a quality diagnostic (and its official PDM weight is zero).  Only add
    # it to the AWR hard filter when the active reward config explicitly turns
    # on ``lane_gate_enabled``.  The same rule is used for road-border and
    # stopped-neighbor gates so ``safe_only`` means "the configured terminal
    # gates", not "every diagnostic that happened to be logged".
    safe = safe_candidate_mask(rewards, reward_config)
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
            "zero_reward_weight_share": 0.0,
            "safe_candidate_count": float(safe_candidate_count),
            "safe_only": float(safe_only),
            "positive_advantage_only": float(positive_advantage_only),
            "positive_advantage_margin": float(positive_advantage_margin),
            "positive_candidate_count": 0.0,
            "first_waypoint_invalid_candidate_count": float(invalid_candidate_count),
            "first_waypoint_invalid_candidate_fraction": float(
                invalid_candidate_count / max(1, totals.size)
            ),
        }

    zero_tol = 1e-8
    all_zero_group = bool(
        original_finite.any()
        and np.all(np.abs(totals[original_finite]) <= zero_tol)
    )
    positive_candidate_count = 0
    selected_anchor_weight = 0.0
    if positive_advantage_only and bool(original_finite[0]):
        # Candidate zero is the deterministic behavior trajectory whenever
        # deterministic_first=True. Lower-return stochastic samples are kept
        # for diagnostics but receive no regression weight. If no sampled
        # target is better, retain only the behavior anchor.
        positive = original_finite & (
            totals > totals[0] + max(float(positive_advantage_margin), 1e-8)
        )
        positive_candidate_count = int(positive.sum())
        selected_anchor_weight = float(
            unsafe_behavior_anchor_weight
            if abs(float(totals[0])) <= zero_tol and positive_candidate_count > 0
            else behavior_anchor_weight
        )
        if selected_anchor_weight < 0.0:
            raise ValueError(
                "behavior anchor weights must be non-negative, got "
                f"{selected_anchor_weight}"
            )
        if bool(positive.any()):
            finite &= positive
            if selected_anchor_weight > 0.0:
                finite[0] = True
        else:
            finite[:] = False
            if selected_anchor_weight > 0.0:
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
    # Compute finite tied groups deterministically.  When the formal
    # HDP-compatible drop flag below is enabled, all-zero terminal ties are
    # subsequently assigned no gradient.  Keeping unit weights is available
    # only as an explicit retention ablation.
    valid_group = bool(finite_totals.size > 0)

    weights = np.zeros_like(totals, dtype=np.float32)
    if valid_group:
        if finite_totals.size == 1:
            weights[finite] = 1.0
        else:
            z = (totals - mean) / (std + max(float(min_group_std), 1e-8))
            log_weights = np.clip(float(beta) * z, -30.0, math.log(max(weight_clip, 1.0)))
            weights[finite] = np.exp(log_weights[finite]).astype(np.float32)
        if positive_advantage_only and bool(original_finite[0]):
            weights[0] = selected_anchor_weight
        if normalize:
            active = weights > 0
            if bool(active.any()):
                weights[active] /= max(float(weights[active].mean()), 1e-8)

    dropped_all_zero_group = bool(drop_all_zero_groups and all_zero_group)
    if dropped_all_zero_group:
        # Reward finiteness still makes this a valid row.  Zeroing the weights
        # marks an intentional no-gradient group instead of corrupt input.
        weights[:] = 0.0

    finite_weights = weights[weights > 0]
    weight_sum = float(finite_weights.sum())
    zero_reward_candidates = original_finite & (np.abs(totals) <= zero_tol)
    zero_reward_weight_share = (
        float(weights[zero_reward_candidates].sum()) / weight_sum
        if weight_sum > 0.0
        else 0.0
    )
    ess = (weight_sum * weight_sum / float((finite_weights**2).sum())) if weight_sum else 0.0
    top_share = float(finite_weights.max() / weight_sum) if weight_sum else 0.0
    active_weight_count = int(finite_weights.size)
    ess_fraction = ess / active_weight_count if active_weight_count else 0.0
    diagnostics = {
        "reward_mean": mean,
        "reward_std": std,
        "reward_max": float(np.max(finite_totals)) if finite_totals.size else float("nan"),
        "valid_group": float(valid_group),
        "weight_mean": float(finite_weights.mean()) if finite_weights.size else 0.0,
        "weight_max": float(finite_weights.max()) if finite_weights.size else 0.0,
        "weight_min": float(finite_weights.min()) if finite_weights.size else 0.0,
        "effective_sample_size": ess,
        # ESS is measured over active trajectories inside this scene group,
        # not over scenes.  The fraction makes beta concentration comparable
        # when positive-only filtering leaves a different number of targets.
        "active_weight_count": float(active_weight_count),
        "effective_sample_size_fraction": float(ess_fraction),
        "top1_weight_share": top_share,
        "zero_reward_weight_share": zero_reward_weight_share,
        "safe_candidate_count": float(safe_candidate_count),
        "first_waypoint_invalid_candidate_count": float(invalid_candidate_count),
        "first_waypoint_invalid_candidate_fraction": float(
            invalid_candidate_count / max(1, totals.size)
        ),
        "safe_only": float(safe_only),
        "positive_advantage_only": float(positive_advantage_only),
        "all_zero_reward_group": float(all_zero_group),
        "dropped_all_zero_group": float(dropped_all_zero_group),
    }
    # These diagnostics only have meaning when candidate 0 is explicitly the
    # deterministic behavior anchor. In all-stochastic group-relative AWR,
    # candidate 0 is just another random plan.
    if positive_advantage_only:
        diagnostics.update(
            {
                "positive_advantage_margin": float(positive_advantage_margin),
                "positive_candidate_count": float(positive_candidate_count),
                "has_positive_candidate": float(positive_candidate_count > 0),
                "behavior_anchor_weight": float(selected_anchor_weight),
                "behavior_anchor_active": float(
                    weights.size > 0 and weights[0] > 0
                ),
            }
        )
    return torch.from_numpy(weights), diagnostics


@torch.no_grad()
def rollout_and_score_scene(
    behavior_model: nn.Module,
    model_args: Config,
    data: dict[str, torch.Tensor],
    rollout_config: AWRRolloutConfig,
    reward_config: RewardConfig,
    device: torch.device,
    exploration_reference_model: nn.Module | None = None,
    exploration_reference_model_args: Config | None = None,
) -> AWRRollout:
    """Sample, OBB-score and weight one raw T4 scene."""

    if rollout_config.plannerrft_guided_exploration:
        return rollout_and_score_scene_batch(
            behavior_model,
            model_args,
            data,
            rollout_config,
            reward_config,
            device,
            exploration_reference_model=exploration_reference_model,
            exploration_reference_model_args=exploration_reference_model_args,
        )[0]

    trajectories, noise_scales, scene_encoding = sample_unguided_dp_group(
        behavior_model, model_args, data, rollout_config, device
    )
    reward_data = reward_compatible_data(data)
    candidate_valid, first_waypoint_diagnostics = (
        _first_waypoint_gate_and_diagnostics(trajectories, data, rollout_config)
    )
    rewards = compute_reward_batch(trajectories, reward_data, reward_config)
    weights_cpu, diagnostics = resolve_objective(rollout_config.objective)(
        rewards,
        beta=rollout_config.beta,
        weight_clip=rollout_config.weight_clip,
        normalize=rollout_config.normalize_weights,
        min_group_std=rollout_config.min_group_std,
        safe_only=rollout_config.safe_only,
        reward_config=reward_config,
        positive_advantage_only=rollout_config.positive_advantage_only,
        positive_advantage_margin=rollout_config.positive_advantage_margin,
        behavior_anchor_weight=rollout_config.behavior_anchor_weight,
        unsafe_behavior_anchor_weight=rollout_config.unsafe_behavior_anchor_weight,
        drop_all_zero_groups=rollout_config.drop_all_zero_groups,
        candidate_valid_mask=candidate_valid,
    )
    weights = weights_cpu.to(device=device, dtype=torch.float32)
    diagnostics = dict(diagnostics)
    diagnostics.update(first_waypoint_diagnostics)
    diagnostics["candidate_count"] = float(trajectories.shape[0])
    diagnostics["candidate0_deterministic"] = float(
        rollout_config.deterministic_first
    )
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
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Save a restartable policy/EMA pair using portable CPU tensors.

    The original-DP loader ignores unknown keys, so adding the optional
    optimizer payload keeps deployment compatibility while allowing a later
    rollout--replay cycle to preserve Adam moments.  Optimizer restoration is
    deliberately opt-in in the trainer: selecting an EMA as a new independent
    starting point and continuing the live policy/optimizer are different
    experimental semantics and must never be conflated silently.
    """

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

    def cpu_tree(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: cpu_tree(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cpu_tree(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cpu_tree(item) for item in value)
        return value

    payload = {
        "epoch": int(epoch),
        "model": cpu_state(model),
        "ema_state_dict": cpu_state(ema_model),
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = cpu_tree(optimizer.state_dict())
        payload["optimizer_class"] = type(optimizer).__name__
    torch.save(payload, str(output_path))


def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    """Update trainable parameters by EMA and copy frozen state exactly.

    Repeatedly applying ``decay * x + (1-decay) * x`` is not an identity in
    float32.  The original-DP AWR job trains only the DiT; blending hundreds
    of frozen encoder tensors for tens of thousands of steps introduced small
    parameter roundoff that amplified into materially different cached scene
    encodings.  EMA is meaningful only for parameters that can change.
    Frozen parameters and all buffers are copied from the live model so their
    intended equality is bit-exact while legitimate buffer updates still
    propagate.
    """

    decay = float(decay)
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"ema_decay must be in [0, 1), got {decay}")
    with torch.no_grad():
        ema_inner = ema_model.module if hasattr(ema_model, "module") else ema_model
        model_inner = model.module if hasattr(model, "module") else model
        ema_state = ema_inner.state_dict()
        model_state = model_inner.state_dict()
        trainable_names = {
            name
            for name, parameter in model_inner.named_parameters()
            if parameter.requires_grad
        }
        for key, ema_value in ema_state.items():
            value = model_state[key]
            if key in trainable_names and torch.is_floating_point(ema_value):
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
