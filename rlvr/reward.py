"""Rule-based trajectory reward for GRPO training.

Computes R = w_safety * S + w_progress * P + w_smooth * M + w_feasibility * F + w_centerline * C
using log-replay data. Reuses ego bbox construction and lane/neighbor penalty
functions from diffusion_planner.loss for proper vehicle-footprint-aware checks.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Issue #130: the raw subscore / geometry / config code now lives in the
# neutral planner_metrics workspace library. This module keeps the reward
# *shaping* (RewardBreakdown, compute_reward_batch, GRPO advantages) and
# re-exports every moved symbol so `from rlvr.reward import ...` keeps working.
# ---------------------------------------------------------------------------
from planner_metrics.aggregate import (  # noqa: F401
    compute_subscores_batch,
    compute_subscores_scene_batch,
)
from planner_metrics.config import RewardConfig  # noqa: F401  (re-export)
from planner_metrics.geometry import *  # noqa: F401,F403
from planner_metrics.subscores import *  # noqa: F401,F403


@dataclass
class RewardBreakdown:
    safety: float
    progress: float
    smoothness: float
    feasibility: float
    centerline: float
    red_light: float
    total: float
    collision_step: int | None
    off_road_fraction: float
    rb_crossing: bool = False
    rb_near_penalty: float = 0.0  # near-zone penalty (frac or survival-style depending on mode)
    rb_wide_penalty: float = 0.0  # wide-zone penalty (frac or survival-style depending on mode)
    rb_min_dist: float = ROAD_BORDER_NO_DATA_DISTANCE_M  # min ego-border distance (m, skip t=0)
    lane_crossing: bool = False
    lane_near_frac: float = 0.0
    lane_wide_frac: float = 0.0
    # Static-collision (stopped-neighbor clearance) diagnostics. Zero/False
    # when static_collision_enabled=False.
    static_crossing: bool = False
    sc_near_penalty: float = 0.0
    sc_wide_penalty: float = 0.0
    sc_cont_penalty: float = 0.0
    sc_min_dist: float = 99.0  # min OBB clearance to any stopped neighbor (t>=1, ego moving)
    sc_n_stopped: int = 0  # how many stopped neighbors were found in the scene
    # Kinematic feasibility violation (yaw rate + bicycle-model curvature).
    # When True, the trajectory is INFEASIBLE and compute_reward_batch floors
    # ``total`` to the offroad floor. Convention matches the other gate
    # booleans on this dataclass (rb_crossing, lane_crossing, static_crossing):
    # True = violation occurred.
    kinematic_violated: bool = False


def _json_safe_value(value):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.floating):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"cannot JSON-serialize non-scalar tensor with shape {tuple(value.shape)}"
            )
        return _json_safe_value(float(value.item()))
    return value


def reward_breakdown_to_json_dict(breakdown: RewardBreakdown) -> dict:
    """Return a standards-compliant JSON dict for a reward breakdown.

    Missing geometric measurements are represented internally as ``inf`` so
    they cannot be confused with a valid large clearance. JSON has no standard
    infinity value, so public serialized diagnostics use ``null`` for those
    fields.
    """
    return {key: _json_safe_value(value) for key, value in asdict(breakdown).items()}


def _slice_reward_horizon(
    data: dict[str, torch.Tensor], horizon: int
) -> dict[str, torch.Tensor]:
    """Slice only time-indexed futures for a short HDP/PDM reward horizon."""

    out = dict(data)
    for key in ("ego_agent_future", "neighbor_agents_future"):
        value = out.get(key)
        if isinstance(value, torch.Tensor) and value.dim() >= 3:
            out[key] = value[..., :horizon, :]
    return out


def _expert_route_progress_ratio(
    ego_trajs: torch.Tensor,
    reference_future: torch.Tensor | None,
) -> torch.Tensor:
    """GPU-vectorized EP proxy: project predicted endpoints onto expert arc.

    HDP defines EP as accumulated route progress divided by expert progress,
    not Euclidean distance to a far-away goal.  The T4 cache has no nuPlan
    route object, but it does retain the logged ego future.  Projecting the
    candidate start/end onto that polyline gives the same proposal-relative
    quantity and avoids rewarding a trajectory that merely drives sideways or
    stops short.  The implementation is batched over AWR candidates.
    """

    N, T, _ = ego_trajs.shape
    device = ego_trajs.device
    if reference_future is None:
        return torch.ones(N, device=device)
    ref = reference_future
    if ref.dim() == 3:
        ref = ref[0]
    ref_xy = ref[..., :2].to(device=device, dtype=ego_trajs.dtype)
    valid = ref_xy.abs().sum(dim=-1) > 0.1
    ref_xy = ref_xy[valid]
    if ref_xy.shape[0] < 2:
        return torch.ones(N, device=device)

    p0 = ref_xy[:-1]
    seg = ref_xy[1:] - p0
    seg_len_sq = (seg * seg).sum(dim=-1).clamp_min(1e-6)
    seg_len = seg_len_sq.sqrt()
    cumulative = torch.cat(
        [torch.zeros(1, device=device, dtype=ego_trajs.dtype), seg_len.cumsum(dim=0)[:-1]]
    )

    def _arc(points: torch.Tensor) -> torch.Tensor:
        # points [N,2], segments [S,2] -> [N,S]
        delta = points[:, None, :] - p0[None, :, :]
        u = (delta * seg[None, :, :]).sum(dim=-1) / seg_len_sq[None, :]
        u = u.clamp(0.0, 1.0)
        projection = p0[None, :, :] + u[..., None] * seg[None, :, :]
        distance_sq = ((points[:, None, :] - projection) ** 2).sum(dim=-1)
        nearest = distance_sq.argmin(dim=1)
        return cumulative[nearest] + u[torch.arange(points.shape[0], device=device), nearest] * seg_len[nearest]

    start_arc = _arc(ego_trajs[:, 0, :2])
    end_arc = _arc(ego_trajs[:, -1, :2])
    raw_progress = (end_arc - start_arc).clamp_min(0.0)
    reference_progress = (cumulative[-1] + seg_len[-1]).clamp_min(1e-3)
    # Match nuPlan's uninformative short-expert branch.
    return torch.where(
        reference_progress > 5.0,
        (raw_progress / reference_progress).clamp(0.0, 1.0),
        torch.ones(N, device=device),
    )


def _low_speed_steer_penalty(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    config: RewardConfig,
) -> torch.Tensor:
    """Bounded [0, 1] penalty on how far past the steering limit the first step is.

    ``delta = atan(wheel_base * 2|y| / s**2)`` is the front-wheel angle implied by
    a first step of arc length ``s`` ending ``y`` off the current heading -- the
    same criterion AWR's first-waypoint gate uses.  Penalise only the *excess*
    over ``low_speed_steer_max_rad``, which is the physical steering limit:

    * Below the limit the command is executable, so a genuine creeping turn of
      radius ``R`` (which implies exactly ``atan(wheel_base / R)``) is not taxed
      for turning correctly.  Ramping from zero instead would cost an 8 m radius
      creep turn 0.048 reward against a typical within-group headroom of 0.013,
      and this corpus oversamples unprotected right turns x10.
    * ``delta`` is an ``atan``, so it is bounded by ``pi/2``; normalising the
      excess by ``pi/2 - limit`` maps the whole infeasible range onto [0, 1]
      without saturating.  That matters because the deployed candidate's median
      low-speed implied steer is 1.4716 rad: under a ramp-from-zero form every
      bad candidate clamps to 1.0, and AWR cannot prefer the least infeasible of
      two -- exactly the discrimination this term exists to create.

    Only active below ``low_speed_steer_speed_mps``; steps shorter than
    ``low_speed_steer_min_step_m`` are sampler noise whose geometry is
    meaningless and score zero.  Returns zeros when the weight is zero or the
    scene carries no ego velocity, so no existing profile changes silently.
    """

    N = ego_trajs.shape[0]
    device = ego_trajs.device
    zero = torch.zeros(N, device=device, dtype=ego_trajs.dtype)
    weight = float(getattr(config, "low_speed_steer_penalty", 0.0))
    if weight <= 0.0 or ego_trajs.shape[1] == 0:
        return zero

    state = data.get("ego_current_state")
    if not isinstance(state, torch.Tensor) or state.shape[-1] <= 4:
        return zero
    speed = float(state.reshape(-1, state.shape[-1])[0, 4].abs().item())
    if not math.isfinite(speed) or speed >= float(config.low_speed_steer_speed_mps):
        return zero

    wheel_base = 2.79
    shape = data.get("ego_shape")
    if isinstance(shape, torch.Tensor) and shape.numel():
        candidate = float(shape.reshape(-1)[0].item())
        if math.isfinite(candidate) and candidate > 0.0:
            wheel_base = candidate

    first_xy = ego_trajs[:, 0, :2].to(dtype=torch.float32)
    step = first_xy.norm(dim=-1)
    min_step = max(float(config.low_speed_steer_min_step_m), 1e-6)
    steer = torch.atan(
        wheel_base * 2.0 * first_xy[:, 1].abs() / step.clamp_min(min_step).pow(2)
    )
    steer = torch.where(step >= min_step, steer, torch.zeros_like(steer))
    limit = min(max(float(config.low_speed_steer_max_rad), 1e-6), math.pi / 2.0 - 1e-6)
    excess = (steer - limit) / (math.pi / 2.0 - limit)
    return (weight * excess.clamp(0.0, 1.0)).to(dtype=ego_trajs.dtype)


def _hdp_multi_reward_components(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    subs: dict,
    config: RewardConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the three bounded reward families used by original HDP-RL.

    The released HDP implementation calls the official PDM scorer, while the
    paper describes its RL signal as risk + car-following + lane robustness.
    T4 NPZ scenes do not contain nuPlan's scorer objects, so this adapter keeps
    the same semantics using the data that is actually present:

    * risk: pessimistic TTC / road-border / OBB safety;
    * follow: leader gap, time gap, speed matching and comfort;
    * lane: route-centerline proximity, masked for logged lane changes.

    Every return value is in [0, 1].  Missing leaders or missing lane geometry
    are neutral (1.0), rather than silently becoming a negative training
    signal.  This is deliberately separate from the legacy signed reward and
    from the PlannerRFT ``hdp_pdm`` profile.
    """

    N, T, _ = ego_trajs.shape
    device = ego_trajs.device

    collision_ok = torch.tensor(
        [1.0 if step is None else 0.0 for step in subs["collision_step"]],
        device=device,
        dtype=ego_trajs.dtype,
    )
    # Road-border proximity is a soft occupancy/risk signal; crossing remains
    # a hard zero.  The near/wide terms are fractions of the horizon.
    rb_score = (
        1.0
        - subs["rb_near_penalty"].to(dtype=ego_trajs.dtype)
        - subs["rb_wide_penalty"].to(dtype=ego_trajs.dtype)
    ).clamp(0.0, 1.0)
    ttc_score = subs["ttc"].to(dtype=ego_trajs.dtype).clamp(0.0, 1.0)
    risk_score = torch.minimum(ttc_score, rb_score)
    if bool(getattr(config, "hdp_risk_use_clearance", False)):
        # ``ttc_min_clearance`` is the minimum signed OBB edge clearance over
        # the short TTC look-ahead for every base timestep.  Unlike the old
        # fraction-of-safe-frames TTC score, this gives HDP's risk reward a
        # useful gradient on near misses before they become collisions.
        clearance = subs.get("ttc_min_clearance")
        if isinstance(clearance, torch.Tensor) and clearance.dim() == 2:
            safe_m = max(float(getattr(config, "hdp_risk_clearance_safe_m", 2.0)), 1e-3)
            clearance_score = (
                clearance.to(dtype=ego_trajs.dtype) / safe_m
            ).clamp(0.0, 1.0).min(dim=-1).values
            risk_score = torch.minimum(risk_score, clearance_score)
    risk_score = risk_score * collision_ok

    # ---- car-following reward -------------------------------------------
    follow_score = torch.ones(N, device=device, dtype=ego_trajs.dtype)
    nf = data.get("neighbor_agents_future")
    if isinstance(nf, torch.Tensor):
        if nf.dim() == 4:
            nf = nf[0]
        if nf.dim() == 3 and nf.shape[-1] >= 4 and nf.shape[0] > 0:
            nf = nf[:, :T, :4]
            valid = nf[..., :2].abs().sum(dim=-1) > 0.1
            # A leader is selected from the current +1 frame, in front of the
            # ego and within a broad lane corridor.  The canonical loader has
            # already applied the legacy +1 neighbor-future correction.
            first = nf[:, 0]
            leader_mask = (
                valid[:, 0]
                & (first[:, 0] > 0.5)
                & (first[:, 1].abs() <= float(config.hdp_leader_lateral_threshold))
            )
            if bool(leader_mask.any()):
                leader_x = first[:, 0].masked_fill(~leader_mask, float("inf"))
                leader_index = int(leader_x.argmin().item())
                leader_valid = valid[leader_index]
                if bool(leader_valid.any()):
                    leader = nf[leader_index]
                    ego_xy = ego_trajs[..., :2]

                    # Speed from the generated path.  The first value uses
                    # the origin-to-waypoint displacement, then finite
                    # differences match the 10 Hz data convention.
                    if T > 1:
                        ego_vel = torch.diff(ego_xy, dim=1) / max(float(config.dt), 1e-3)
                        ego_vel = torch.cat(
                            [ego_xy[:, :1] / max(float(config.dt), 1e-3), ego_vel], dim=1
                        )
                        lead_vel = torch.diff(leader[:, :2], dim=0) / max(float(config.dt), 1e-3)
                        lead_vel = torch.cat(
                            [lead_vel[:1], lead_vel], dim=0
                        )
                    else:
                        ego_vel = ego_xy / max(float(config.dt), 1e-3)
                        lead_vel = torch.zeros_like(leader[:, :2])
                    ego_speed = ego_vel.norm(dim=-1).clamp_min(0.0)
                    lead_speed = lead_vel.norm(dim=-1).unsqueeze(0)
                    gap = leader[None, :, 0] - ego_xy[..., 0]
                    desired_gap = 2.0 + 1.5 * ego_speed
                    desired_time_gap = max(float(config.hdp_desired_time_gap), 0.1)
                    time_gap = gap / ego_speed.clamp_min(1.0)

                    spacing = torch.exp(
                        -torch.abs(gap - desired_gap) / (desired_gap + 2.0)
                    )
                    time_gap_score = torch.exp(
                        -torch.abs(time_gap - desired_time_gap) / 1.5
                    )
                    speed_match = torch.exp(-torch.abs(ego_speed - lead_speed) / 3.0)
                    comfort = torch.exp(
                        subs["comfort"].to(dtype=ego_trajs.dtype).clamp(-50.0, 0.0)
                        / max(float(config.pdm_comfort_scale), 1e-3)
                    )
                    comfort = comfort * torch.exp(
                        subs["feasibility"].to(dtype=ego_trajs.dtype).clamp(-10.0, 0.0)
                    )
                    per_step = torch.stack(
                        [spacing, time_gap_score, speed_match, comfort[:, None].expand(N, T)],
                        dim=-1,
                    ).mean(dim=-1)
                    follow_score = torch.where(
                        leader_valid[None, :], per_step, torch.ones_like(per_step)
                    ).sum(dim=-1) / leader_valid[None, :].to(ego_trajs.dtype).sum().clamp_min(1.0)
                    follow_score = follow_score.clamp(0.0, 1.0)

    # ---- lane robustness reward ----------------------------------------
    lane_score = torch.exp(
        subs["centerline"].to(dtype=ego_trajs.dtype).clamp(-50.0, 0.0)
        / max(float(config.hdp_lane_score_scale), 1e-3)
    ).clamp(0.0, 1.0)

    # Paper intent: do not punish an intentional lane change.  This local
    # primitive detects the logged expert leaving the UNION of nearby lane
    # polygons; shared boundaries between two legal adjacent lanes are removed
    # as internal edges.  It is therefore an off-lane / outer-union mask, not a
    # complete lane-change detector.  A future T4 profile should additionally
    # consume a validated lane-change sidecar (or indicator/route semantics)
    # before claiming full parity with the HDP paper's lane-change mask.
    gt = data.get("ego_agent_future")
    if isinstance(gt, torch.Tensor):
        if gt.dim() == 3:
            gt = gt[0]
        gt = gt[:T]
        if gt.dim() == 2 and gt.shape[-1] >= 3:
            if gt.shape[-1] == 3:
                gt4 = torch.cat([gt[:, :2], torch.cos(gt[:, 2:3]), torch.sin(gt[:, 2:3])], dim=-1)
            else:
                gt4 = gt[:, :4]
            # Only the lane-departure result is needed for the expert off-lane
            # mask.  Calling compute_subscores_batch here used to
            # recompute collision, TTC, border, smoothness, feasibility and
            # red-light metrics for the logged expert on every scene.  That
            # was mathematically redundant and dominated full-corpus HDP
            # rollout time.  Reuse the exact lane geometry primitive instead.
            expert_shape = data.get("ego_shape")
            if isinstance(expert_shape, torch.Tensor):
                if expert_shape.dim() == 2:
                    expert_shape = expert_shape[0]
                expert_lane = compute_lane_departure_penalty(
                    gt4.unsqueeze(0), expert_shape[:3].to(device), data, config=config
                )
                expert_lane_steps = expert_lane[3]
            else:
                expert_lane_steps = [None]
            if isinstance(expert_lane_steps, list) and expert_lane_steps and expert_lane_steps[0] is not None:
                lane_score = torch.ones_like(lane_score)

    return risk_score, follow_score, lane_score


@torch.no_grad()
def compute_reward_batch(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    config: RewardConfig = RewardConfig(),
) -> list[RewardBreakdown]:
    """Compute reward breakdowns for N trajectories in a single batched pass.

    Thin wrapper: the raw subscores (input marshalling + per-subscore
    computation) come from ``compute_subscores_batch``; reward shaping (weights,
    hard gates, survival/gate aggregation) is applied by ``_shape_reward``.
    Output is byte-identical to the pre-split implementation (pinned by the
    golden tests).

    Args:
        ego_trajs: (N, T, 4) x, y, cos_yaw, sin_yaw.
        data: Observation dict from load_npz_data (with batch dim).
        config: RewardConfig with component weights.

    Returns:
        List of N RewardBreakdown instances.
    """
    horizon = int(getattr(config, "reward_horizon_steps", 0))
    if 0 < horizon < ego_trajs.shape[1]:
        reward_trajs = ego_trajs[:, :horizon]
        reward_data = _slice_reward_horizon(data, horizon)
    else:
        reward_trajs = ego_trajs
        reward_data = data
    return _shape_reward(
        compute_subscores_batch(reward_trajs, reward_data, config),
        reward_trajs,
        reward_data,
        config,
    )


@torch.no_grad()
def _shape_reward(
    subs: dict,
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    config: RewardConfig = RewardConfig(),
) -> list[RewardBreakdown]:
    """Apply reward shaping (weights, hard gates, survival/gate aggregation) to the
    raw subscores from ``compute_subscores_batch`` and build the RewardBreakdown
    list. RLVR-specific; the raw subscore computation is neutral (lives in
    ``planner_metrics``)."""
    N, T, _ = ego_trajs.shape
    device = ego_trajs.device

    safety_scores = subs["safety"]
    collision_steps = subs["collision_step"]
    red_light_scores = subs["red_light"]
    progress_scores = subs["progress"]
    smoothness_scores = subs["comfort"]
    centerline_scores = subs["centerline"]
    feasibility_scores = subs["feasibility"]
    off_road_fractions = subs["off_road_fraction"]
    ttc_scores = subs["ttc"]
    kinematic_gate = subs["kinematic_gate"]
    rb_crossing_gate = subs["rb_crossing_gate"]
    rb_near_pen = subs["rb_near_penalty"]
    rb_wide_pen = subs["rb_wide_penalty"]
    rb_cont_penalty = subs["rb_cont_penalty"]
    rb_crossing_steps = subs["rb_crossing_steps"]
    rb_min_dist_t = subs["rb_min_dist"]
    lane_crossing_gate = subs["lane_crossing_gate"]
    lane_near_frac = subs["lane_near_frac"]
    lane_wide_frac = subs["lane_wide_frac"]
    lane_cont_penalty = subs["lane_cont_penalty"]
    lane_crossing_steps = subs["lane_crossing_steps"]
    sc_crossing_gate = subs["sc_crossing_gate"]
    sc_near_pen = subs["sc_near_penalty"]
    sc_wide_pen = subs["sc_wide_penalty"]
    sc_cont_pen = subs["sc_cont_penalty"]
    sc_crossing_steps = subs["sc_crossing_steps"]
    sc_min_dist_scalar = subs["sc_min_dist"]
    sc_n_stopped_scene = int(subs["sc_n_stopped"][0].item()) if N > 0 else 0

    # NAVSIM PDMS-style multiplicative reward aggregation.
    # Safety gates: binary 0/1 multipliers. If any gate is 0, total is 0.
    # This prevents reward hacking (e.g. stopping to avoid offroad penalty)
    # because stopped trajectories get progress=0 → total=0, same as offroad.
    # Only trajectories that drive AND stay on-road get positive reward.

    # Gate 1: No collision (binary: 1 if no collision, 0 if collision)
    has_collision = torch.tensor(
        [1.0 if cs is not None else 0.0 for cs in collision_steps],
        device=device,
    )
    collision_gate = 1.0 - has_collision  # (N,)

    # Gate 2: Drivable area compliance — steep sigmoid
    # Near-binary but allows ranking of partially-offroad trajectories.
    # Hard binary (offroad>0 → 0) was tested in exp028 but gave worse
    # prob offroad (5% vs 0.8% with sigmoid in exp023) because it kills
    # the ranking signal for scenes where ALL trajectories have some offroad.
    # Binary gate: ANY offroad → gate = 0. No partial credit.
    # Polygon drivable_gate removed — road border crossing gate handles offroad detection
    drivable_gate = torch.ones(N, device=device)  # always passes (polygon check disabled)

    # Gate 3: Red light compliance
    has_red_light_violation = (red_light_scores < -0.5).float()
    red_light_gate = 1.0 - has_red_light_violation  # (N,)

    # Multiplicative safety product (hard gates only)
    # TTC is included in quality_score as a soft penalty instead of a gate,
    # because at intersections many good trajectories pass near NPCs.
    # Gate 4: Road border compliance (crossing = instant fail)
    # Road border perimeter check is the primary offroad detection (v4).
    # Lane polygon drivable_gate is kept as a soft penalty only, not a hard gate,
    # since lane polygons can disagree with road borders at intersection corners.
    safety_product = collision_gate * red_light_gate  # (N,)
    if config.rb_gate_enabled:
        safety_product = safety_product * rb_crossing_gate
    if config.lane_gate_enabled:
        safety_product = safety_product * lane_crossing_gate
    if config.static_collision_enabled and config.sc_gate_enabled:
        safety_product = safety_product * sc_crossing_gate

    # Weighted quality metrics (only matter when safety gates pass)
    # Progress is the primary positive signal. Smoothness/centerline are penalties.
    # Safety score includes proximity penalty to NPCs (closer = more negative).
    clamped_progress = progress_scores.clamp(min=0)

    # Progress-related penalties (overprogress, stopped, underprogress) are floors,
    # not progress rewards — they must apply even when w_progress=0. Accumulate
    # them into `progress_penalty` and subtract from quality_score directly,
    # bypassing the w_progress multiplier.
    progress_penalty = torch.zeros(N, device=device)

    # Normalize progress as percentage of GT path length, then apply
    # overprogress/underprogress/stopped penalties.
    # This ensures a 10m path on a 12m GT scene and a 10m path on a 22m GT scene
    # get different progress scores (83% vs 45%).
    if config.enable_overprogress and "ego_agent_future" in data:
        gt_future = data["ego_agent_future"]
        if gt_future.dim() == 3:
            gt_future = gt_future[0]  # (T_gt, 3)
        gt_xy = gt_future[:, :2]
        gt_valid = gt_xy.abs().sum(dim=-1) > 0.1
        # ALWAYS compute model path lengths — they are used for stopped and
        # underprogress penalties which must fire whether or not GT is
        # present (synthetic-data RSFT passes zero GT; without this the
        # penalties silently no-op and the model collapses path).
        model_path_lens = torch.diff(ego_trajs[:, :, :2], dim=1).norm(dim=-1).sum(dim=-1)  # (N,)
        baseline_path_len_scalar = None
        if "baseline_path_len" in data:
            bpl_t = torch.as_tensor(
                data["baseline_path_len"],
                device=device,
                dtype=torch.float32,
            ).reshape(())
            baseline_path_len_scalar = float(bpl_t.clamp(min=1e-3).item())

        if gt_valid.sum() >= 10:
            gt_path_len = torch.diff(gt_xy[gt_valid], dim=0).norm(dim=-1).sum()

            # Normalize progress to [0, 1] as fraction of GT, capped at margin.
            # 100% GT = 1.0 (max), >margin% GT = capped + penalized.
            progress_frac = (clamped_progress / gt_path_len.clamp(min=1e-3)).clamp(
                max=config.overprogress_margin
            )
            clamped_progress = progress_frac * config.progress_norm_scale

            # Compute path ratio for symmetric over/under progress penalties.
            # Both use the same ratio-based method: penalty * |deviation from threshold|.
            path_ratio = model_path_lens / gt_path_len.clamp(min=1e-3)

            # Overprogress: penalize path exceeding margin × GT (ratio-based).
            # NOTE: Changed from meter-based (pre-April 2026) to ratio-based.
            # Old: penalty * relu(path_meters - cap_meters). New: penalty * relu(ratio - margin).
            # Configs must use ratio-scale penalties (e.g. 100.0), not meter-scale (e.g. 0.3).
            # E.g., margin=1.0, penalty=100: at 1.5x GT → 100*(1.5-1.0)=50 penalty.
            overprogress = torch.relu(path_ratio - config.overprogress_margin)
            progress_penalty = progress_penalty + config.overprogress_penalty * overprogress

        # Stopped penalty: fires on any trajectory that barely moves,
        # whenever an anchor scene "should have moved" — GT is the
        # canonical anchor when present, else baseline_path_len
        # (underprogress_reference). Without either, we can't distinguish
        # a legitimate stop (red light) from reward-hacking collapse.
        anchor_len: float | None = None
        if gt_valid.sum() >= 10:
            anchor_len = float(torch.diff(gt_xy[gt_valid], dim=0).norm(dim=-1).sum().item())
        elif baseline_path_len_scalar is not None:
            anchor_len = baseline_path_len_scalar
        if config.stopped_penalty > 0 and anchor_len is not None and anchor_len > 5.0:
            is_stopped = (model_path_lens < 1.0).float()
            progress_penalty = progress_penalty + config.stopped_penalty * is_stopped

        # Underprogress: penalize trajectories shorter than the reference path.
        # When ``underprogress_reference="baseline"`` AND ``data["baseline_path_len"]``
        # is present, ALWAYS fire (even at N=1, and even when GT is absent —
        # the whole point of the baseline anchor is it doesn't depend on the
        # current rollout). The N>1 guard only makes sense for the legacy
        # "det" reference where traj[0] is the reference and ratio ≡ 1.0.
        _have_baseline_ref = (
            config.underprogress_reference == "baseline" and baseline_path_len_scalar is not None
        )
        if config.underprogress_penalty > 0 and (N > 1 or _have_baseline_ref):
            # Reference selection:
            #   "det"      — path of the deterministic traj (traj[0]). Adapts to
            #                current model, but can collapse to short when model
            #                starts producing short det trajs.
            #   "baseline" — baseline LoRA-less det path length, passed via
            #                `data["baseline_path_len"]` (a scalar tensor). Frozen
            #                anchor that doesn't collapse with training.
            if config.underprogress_reference == "baseline" and "baseline_path_len" in data:
                # Accept tensor / numpy scalar / Python float — callers may inject metadata
                # in any of these forms when wiring custom data dicts.
                ref_path_len = torch.as_tensor(
                    data["baseline_path_len"],
                    device=device,
                    dtype=torch.float32,
                )
                if ref_path_len.numel() != 1:
                    raise ValueError(
                        "data['baseline_path_len'] must be a scalar value, got shape "
                        f"{tuple(ref_path_len.shape)}"
                    )
                ref_path_len = ref_path_len.reshape(()).clamp(min=1e-3)
            else:
                ref_path_len = model_path_lens[0].clamp(min=1e-3)
            ratio = model_path_lens / ref_path_len
            underprogress = torch.relu(config.underprogress_threshold - ratio.clamp(max=1.0))
            progress_penalty = progress_penalty + config.underprogress_penalty * underprogress

    # TTC as quality bonus
    ttc_bonus = config.w_safety * (ttc_scores - 0.5) * 2

    # Road border proximity penalties (soft, applied even when on-road)
    # Thresholds configurable via config.rb_near_thresh / rb_wide_thresh
    rb_penalty = (
        config.rb_near_scale * rb_near_pen
        + config.rb_wide_scale * rb_wide_pen
        + config.rb_cont_scale * rb_cont_penalty
    )

    # Lane departure proximity penalties
    lane_penalty = (
        config.lane_near_scale * lane_near_frac
        + config.lane_wide_scale * lane_wide_frac
        + config.lane_cont_scale * lane_cont_penalty
    )

    # Static-collision proximity penalties (stopped neighbors).
    sc_penalty = (
        config.sc_near_scale * sc_near_pen
        + config.sc_wide_scale * sc_wide_pen
        + config.sc_cont_scale * sc_cont_pen
    )

    # Penalty magnitude preserves legacy behavior for configs with w_progress >= 1
    # (historical default range: w_progress ∈ {2.0, 7.0}), where the old code
    # effectively multiplied the penalty by w_progress via the clamped_progress
    # sum. For w_progress < 1 we floor at 1.0 so penalties still fire on
    # CL-only / reward-sculpted configs (w_progress=0 was the original bug).
    penalty_mult = max(float(config.w_progress), 1.0)
    quality_score = (
        config.w_progress * clamped_progress
        + config.w_safety * safety_scores
        + config.w_smooth * smoothness_scores
        + config.w_centerline * centerline_scores
        + ttc_bonus
        - rb_penalty
        - lane_penalty
        - sc_penalty
        - penalty_mult * progress_penalty
    )

    _OFFROAD_FLOOR = -50.0

    if config.reward_profile == "hdp_multi":
        # Original HDP-RL multi-reward profile: the paper reports a bounded
        # weighted sum of risk, car-following and lane robustness.  Unlike the
        # PlannerRFT/PDM profile below, this deliberately does not substitute
        # an EP/Speed weighted average for HDP's follow/lane rewards.
        risk_score, follow_score, lane_score = _hdp_multi_reward_components(
            ego_trajs, data, subs, config
        )
        total_hdp_weight = max(
            float(config.hdp_risk_weight)
            + float(config.hdp_follow_weight)
            + float(config.hdp_lane_weight),
            1e-6,
        )
        totals = (
            float(config.hdp_risk_weight) * risk_score
            + float(config.hdp_follow_weight) * follow_score
            + float(config.hdp_lane_weight) * lane_score
        ) / total_hdp_weight
        # Keep the local hard kinematic guard below.  Collision/DAC are
        # already represented continuously in r_risk, matching the HDP paper
        # multi-reward formulation instead of turning every candidate into a
        # single -50 terminal label.
    elif config.reward_profile == "hdp_pdm":
        # Published HDP/PlannerRFT reward.  The NPZ interface does not carry
        # nuPlan's full route-progress and speed-limit object, so EP/Speed are
        # computed from the local logged horizon and route-lane speed-limit
        # tensor respectively; the terminal Col/DAC checks remain the same
        # OBB road-border checks used by the open-source PDM port.
        reference_path = None
        gt_future = data.get("ego_agent_future")
        if isinstance(gt_future, torch.Tensor):
            if gt_future.dim() == 3:
                gt_future = gt_future[0]
            gt_xy = gt_future[:, :2]
            gt_valid = gt_xy.abs().sum(dim=-1) > 0.1
            if gt_valid.sum() >= 2:
                reference_path = torch.diff(gt_xy[gt_valid], dim=0).norm(dim=-1).sum()
        if reference_path is None or float(reference_path.item()) < 1e-3:
            baseline_ref = data.get("baseline_path_len")
            if baseline_ref is not None:
                reference_path = torch.as_tensor(baseline_ref, device=device, dtype=torch.float32).reshape(())
            else:
                reference_path = torch.tensor(1.0, device=device)

        # Ego progress (EP): use expert-polyline arc progress, which is the
        # local-data equivalent of HDP's route-progress numerator.  The old
        # Euclidean goal-distance proxy can reward a short sideways path or a
        # stop when the goal pose is far outside the cached scene.
        ep_score = _expert_route_progress_ratio(ego_trajs, gt_future)

        # Comfort (C): a continuous proxy for nuPlan's all-constraints comfort
        # indicator.  Smoothness and acceleration feasibility are both <= 0
        # penalties in the neutral metrics layer.
        comfort_score = torch.exp(
            smoothness_scores.clamp(min=-50.0, max=0.0)
            / max(float(config.pdm_comfort_scale), 1e-3)
        )
        comfort_score = comfort_score * torch.exp(feasibility_scores.clamp(min=-10.0, max=0.0))
        comfort_score = comfort_score.clamp(0.0, 1.0)

        # Low-speed steering feasibility.  ``smoothness_scores`` is computed from
        # the whole 8 s horizon and cannot see a standstill first step: the
        # deployed candidate implies a median 1.47 rad front-wheel angle on a
        # third of low-speed scenes and still scores a full comfort slot.  AWR's
        # first-waypoint gate rejects such candidates but is powerless against
        # the anchor, and without a reward term the smoother of two survivors was
        # never preferred -- which is why low-speed behaviour never improved
        # across a campaign while the real vehicle jittered at a stop.
        comfort_score = comfort_score - _low_speed_steer_penalty(
            ego_trajs, data, config
        )
        comfort_score = comfort_score.clamp(0.0, 1.0)

        # HDP's released NAVSIM agent uses the PDM scorer.  Its weighted
        # terms are EP=5, TTC=5, lane-keeping=2 and history comfort=2; the
        # speed-limit term belongs to other PlannerRFT variants and must not
        # silently replace lane keeping in an HDP reproduction.  Reuse the
        # local lane adapter above so the intentional-lane-change exemption is
        # retained.  Missing oncoming-traffic / traffic-light semantics stay
        # neutral, just as the data-availability note in the local PDM port
        # requires.
        _, _, lane_score = _hdp_multi_reward_components(
            ego_trajs, data, subs, config
        )
        driving_direction_score = torch.ones(N, device=device)

        # The bare PDM TTC indicator is dead signal on this corpus: measured over
        # 300 mined scenes x 10 candidates it was exactly 1.0 on 99.67% of
        # candidates and varied within a candidate group on 0.33% of scenes,
        # while collision fired on 0.00%.  Those are the *only* neighbour-aware
        # terms in this profile -- risk and follow are computed and discarded --
        # so AWR was ranking candidates on progress/lane/comfort alone and had no
        # reason to keep clear of other vehicles.  That is the reward-up /
        # collisions-up trade, not an inherent property of the method.
        #
        # ``hdp_risk_use_clearance`` already means "grade near misses before they
        # become collisions"; it just only reached the hdp_multi profile.  Grade
        # the same clearance here and take the min, so a real TTC failure still
        # lands and the clearance term can only ever tighten.  Non-saturated on
        # 26.0% of scenes and discriminative within the candidate group on 25.7%,
        # which moves the selected candidate on 16.3% of scenes.  Deliberately
        # the clearance alone and not the composite ``risk_score``: that also
        # folds in road-border proximity, which this profile already gates
        # separately, and only the clearance version was measured.
        ttc_term = ttc_scores.clamp(0.0, 1.0)
        clearance = subs.get("ttc_min_clearance")
        if bool(getattr(config, "hdp_risk_use_clearance", False)) and isinstance(
            clearance, torch.Tensor
        ) and clearance.dim() == 2:
            safe_m = max(float(getattr(config, "hdp_risk_clearance_safe_m", 2.0)), 1e-3)
            clearance_score = (
                clearance.to(dtype=ttc_term.dtype) / safe_m
            ).clamp(0.0, 1.0).min(dim=-1).values
            ttc_term = torch.minimum(ttc_term, clearance_score)
        pdm_quality = (
            5.0 * ttc_term
            + 5.0 * ep_score
            + 2.0 * lane_score.clamp(0.0, 1.0)
            + 2.0 * comfort_score
        ) / 14.0

        # Col×DAC terminal product.  ``survival`` is an optional
        # PlannerRFT-style hard-scene ablation that adds first-failure-time
        # credit; it is not part of HDP's published max-collision / risk
        # reward.  A road-border cross is DAC=0 regardless of whether the
        # custom profile's optional rb gate is enabled.
        pdm_terminal = collision_gate * rb_crossing_gate * driving_direction_score
        if config.reward_mode == "survival":
            pdm_survival_frac = torch.ones(N, device=device)
            for i in range(N):
                first_terminal = T
                if collision_steps[i] is not None:
                    first_terminal = min(first_terminal, collision_steps[i])
                if rb_crossing_steps[i] is not None:
                    first_terminal = min(first_terminal, rb_crossing_steps[i])
                pdm_survival_frac[i] = max(first_terminal, 1) / max(T, 1)
            totals = pdm_survival_frac * pdm_quality * driving_direction_score
        else:
            totals = pdm_terminal * pdm_quality
    elif config.reward_mode == "survival":
        # PlannerRFT-style survival reward: proportional credit based on how
        # long the trajectory survives before the first terminal event.
        # survival_frac = first_terminal_step / T. A crash at t=60/80 gets 75%
        # of quality_score. This prevents gradient death on hard scenes where
        # all trajectories fail — later crashes still rank higher.
        survival_frac = torch.ones(N, device=device)
        for i in range(N):
            first_terminal = T  # no failure → full survival
            if collision_steps[i] is not None:
                first_terminal = min(first_terminal, collision_steps[i])
            if config.rb_gate_enabled and rb_crossing_steps[i] is not None:
                first_terminal = min(first_terminal, rb_crossing_steps[i])
            # Lane keeping is a diagnostic in the HDP/PDM-aligned profile.  A
            # legal lane change necessarily crosses a shared lane boundary, so
            # it must not shorten survival unless the caller explicitly turns
            # on the lane hard gate.  ``enable_lane_departure`` only controls
            # whether the geometry is measured/logged.
            if (
                config.lane_gate_enabled
                and config.enable_lane_departure
                and lane_crossing_steps[i] is not None
            ):
                first_terminal = min(first_terminal, lane_crossing_steps[i])
            if (
                config.static_collision_enabled
                and config.sc_gate_enabled
                and sc_crossing_steps[i] is not None
            ):
                first_terminal = min(first_terminal, sc_crossing_steps[i])
            survival_frac[i] = max(first_terminal, 1) / T  # at least 1/T to avoid 0

        # Blend: survived portion gets quality, failed portion gets floor.
        # Red light violations still use a hard gate on top of survival —
        # red light doesn't have a per-timestep failure point, so we apply
        # it as a binary multiplier like in gate mode.
        totals = survival_frac * quality_score + (1.0 - survival_frac) * _OFFROAD_FLOOR
        totals = totals * red_light_gate + (1.0 - red_light_gate) * _OFFROAD_FLOOR
    else:
        # Default "gate" mode: binary safety gates × quality.
        # Any terminal event → full floor penalty regardless of when it happens.
        totals = safety_product * quality_score + (1.0 - safety_product) * _OFFROAD_FLOOR

    # Kinematic feasibility hard gate.  The released HDP RL path receives the
    # normalized PDM score in [0, 1] and filters only NaN / out-of-range
    # proposals (``reward > -1``); a failed proposal therefore contributes a
    # zero score, not the signed custom-reward floor used by the legacy DP
    # objective.  Keep that non-negative convention for the HDP/PDM adapter so
    # group z-normalisation has the same semantics as the reference code.
    if config.reward_profile in {"hdp_pdm", "hdp_multi"}:
        # Both published HDP reward families are bounded in [0, 1].  A
        # kinematic failure is therefore a zero-scoring proposal, just like
        # the released HDP replay filter's ``reward > -1`` convention; it
        # must not inherit the legacy -50 custom floor and distort group
        # normalisation.
        totals = totals * kinematic_gate
    else:
        # Custom / historical profiles retain their explicit floor so existing
        # rule-based GRPO experiments remain numerically backward compatible.
        totals = totals * kinematic_gate + (1.0 - kinematic_gate) * _OFFROAD_FLOOR

    # Also compute additive total for backward compat in breakdown
    on_road_factor = 1.0 - off_road_fractions
    adjusted_progress = progress_scores * on_road_factor

    # Materialise every scalar diagnostic with one device-to-host transfer.
    #
    # The rollout miner calls this function once per scene with K candidates.
    # Converting each field with ``float(cuda_tensor[i])`` in the loop below
    # used to introduce O(K * fields) independent CUDA synchronisations per
    # scene.  Packing the already-computed tensors changes no reward math (the
    # same values, in the same dtype, are merely copied together) while
    # reducing that boundary to one synchronisation per scene.  Keep the
    # Python dataclass ABI unchanged for replay writers and diagnostics.
    packed_rows = torch.stack(
        (
            safety_scores,
            adjusted_progress,
            smoothness_scores,
            feasibility_scores,
            centerline_scores,
            red_light_scores,
            totals,
            off_road_fractions,
            rb_crossing_gate,
            rb_near_pen,
            rb_wide_pen,
            rb_min_dist_t,
            lane_crossing_gate,
            lane_near_frac,
            lane_wide_frac,
            sc_crossing_gate,
            sc_near_pen,
            sc_wide_pen,
            sc_cont_pen,
            sc_min_dist_scalar,
            kinematic_gate,
        ),
        dim=1,
    ).detach().cpu().tolist()

    results: list[RewardBreakdown] = []
    for i, row in enumerate(packed_rows):
        (
            safety_value,
            progress_value,
            smoothness_value,
            feasibility_value,
            centerline_value,
            red_light_value,
            total_value,
            off_road_value,
            rb_gate_value,
            rb_near_value,
            rb_wide_value,
            rb_min_value,
            lane_gate_value,
            lane_near_value,
            lane_wide_value,
            sc_gate_value,
            sc_near_value,
            sc_wide_value,
            sc_cont_value,
            sc_min_value,
            kinematic_gate_value,
        ) = row
        results.append(
            RewardBreakdown(
                safety=float(safety_value),
                progress=float(progress_value),
                smoothness=float(smoothness_value),
                feasibility=float(feasibility_value),
                centerline=float(centerline_value),
                red_light=float(red_light_value),
                total=float(total_value),
                collision_step=collision_steps[i],
                # Always 0 for the current polygon-disabled profile; retain
                # it for backward-compatible diagnostics.
                off_road_fraction=float(off_road_value),
                rb_crossing=bool(rb_gate_value < 0.5),
                rb_near_penalty=float(rb_near_value),
                rb_wide_penalty=float(rb_wide_value),
                rb_min_dist=float(rb_min_value),
                lane_crossing=bool(lane_gate_value < 0.5),
                lane_near_frac=float(lane_near_value),
                lane_wide_frac=float(lane_wide_value),
                static_crossing=bool(sc_gate_value < 0.5),
                sc_near_penalty=float(sc_near_value),
                sc_wide_penalty=float(sc_wide_value),
                sc_cont_penalty=float(sc_cont_value),
                sc_min_dist=float(sc_min_value),
                sc_n_stopped=sc_n_stopped_scene,
                kinematic_violated=bool(kinematic_gate_value < 0.5),
            )
        )

    return results


def compute_reward(
    ego_traj: torch.Tensor,
    data: dict[str, torch.Tensor],
    config: RewardConfig = RewardConfig(),
) -> RewardBreakdown:
    """Single-trajectory convenience wrapper around compute_reward_batch."""
    return compute_reward_batch(ego_traj.unsqueeze(0), data, config)[0]


def compute_group_advantages(
    rewards: list[RewardBreakdown],
    epsilon: float = 1e-8,
    mode: str = "normalized",
    fixed_scale: float = 10.0,
) -> np.ndarray:
    """Compute GRPO-style group-relative advantages.

    Args:
        rewards: List of RewardBreakdown for each trajectory in the group.
        epsilon: Small constant for numerical stability.
        mode: Advantage computation mode:
            "normalized": Standard GRPO (mean=0, std=1 per group).
            "vd_grpo": Variance-Decoupled GRPO (center only, fixed scale).
                Preserves absolute magnitude of negative rewards across groups.
            "raw": Centered advantages without std normalization. Uses
                fixed_scale as denominator. If all trajectories in a group
                are bad (e.g., all leave lane), all get negative advantages
                instead of half getting positive weight.
            "positive_only": Like "normalized" but clips negative advantages
                to zero. Only updates on trajectories that are better than
                the group mean.
        fixed_scale: Denominator for vd_grpo and raw modes.

    Returns:
        (G,) array of advantages.
    """
    totals = np.array([r.total for r in rewards])
    mean = totals.mean()

    if mode == "vd_grpo":
        if fixed_scale <= 0.0:
            raise ValueError(f"advantage_fixed_scale must be positive, got {fixed_scale}")
        return (totals - mean) / max(fixed_scale, epsilon)
    elif mode == "normalized":
        std = totals.std()
        if std < epsilon:
            return np.zeros(len(rewards))
        return (totals - mean) / (std + epsilon)
    elif mode == "raw":
        # Centered advantages without per-group std normalization.
        # If all K trajectories are bad, all get negative advantages.
        # This prevents normalized advantages from giving half of an
        # all-bad group positive weight.
        if fixed_scale <= 0.0:
            raise ValueError(f"advantage_fixed_scale must be positive, got {fixed_scale}")
        return (totals - mean) / max(fixed_scale, epsilon)
    elif mode == "absolute":
        # No centering, no normalization. Advantage = total / fixed_scale.
        # Positive reward → positive advantage, negative reward → negative advantage.
        # A group where all trajs score -30 gets ALL negative advantages.
        # Only trajs with positive absolute reward get reinforced.
        if fixed_scale <= 0.0:
            raise ValueError(f"advantage_fixed_scale must be positive, got {fixed_scale}")
        return totals / max(fixed_scale, epsilon)
    elif mode == "softmax":
        # Softmax-weighted advantages. Temperature = fixed_scale.
        # Rank 1 gets disproportionately strong signal (~0.9), others decay sharply.
        # Low temperature (5) = very sharp (rank 1 dominates).
        # High temperature (20) = softer (more spread across top trajs).
        # Centered so mean≈0 for stable GRPO training.
        temp = max(fixed_scale, epsilon)
        logits = totals / temp
        logits = logits - logits.max()  # numerical stability
        exp_logits = np.exp(logits)
        weights = exp_logits / exp_logits.sum()
        # Center and scale: mean=0, max≈1
        advantages = (weights - weights.mean()) / max(weights.max(), epsilon)
        return advantages
    elif mode == "positive_only":
        # Standard normalization but clip negatives to zero.
        # Only reinforces trajectories better than the group mean.
        std = totals.std()
        if std < epsilon:
            return np.zeros(len(rewards))
        advantages = (totals - mean) / (std + epsilon)
        return np.maximum(advantages, 0.0)
    elif mode == "ddv2":
        # DiffusionDriveV2 Inter-Anchor Truncated GRPO (arXiv:2512.07745, Eq. 10):
        # 1. Standard intra-group normalization
        # 2. Clip negative advantages to 0 (only reinforce improvements over group mean)
        # 3. Hard -1 penalty for safety violations (collision, off-road, lane departure)
        # Extension vs paper: paper only penalizes collisions; we also penalize
        # road-border crossings and lane departures. No inter-anchor distinction
        # since we don't use DDV2's multi-anchor GMM architecture.
        std = totals.std()
        if std < epsilon:
            advantages = np.zeros(len(rewards))
        else:
            advantages = (totals - mean) / (std + epsilon)
        # Clip negative to 0
        advantages = np.maximum(advantages, 0.0)
        # Hard -1 for safety violations
        for i, rb in enumerate(rewards):
            if rb.collision_step is not None or rb.rb_crossing or rb.lane_crossing:
                advantages[i] = -1.0
        return advantages
    else:
        raise ValueError(
            f"Unknown advantage mode: {mode!r}. "
            f"Expected 'normalized', 'vd_grpo', 'raw', 'absolute', 'softmax', "
            f"'positive_only', or 'ddv2'."
        )
