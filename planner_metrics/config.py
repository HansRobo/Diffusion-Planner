"""Configuration thresholds for the ``planner_metrics`` subscores."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardConfig:
    w_safety: float = 5.0
    w_progress: float = 2.0
    w_smooth: float = 0.5
    w_feasibility: float = 5.0
    w_centerline: float = 5.0
    # Centerline usage mode:
    #   "baselink" (default): lane_usage = |baselink_lat| / side_hw —
    #       perfectly centered = 0; readings are directly interpretable as
    #       "rear axle is X fraction of the way to the lane edge".
    #   "body" (DEPRECATED): lane_usage = (|baselink_lat| + ego_half_w) /
    #       side_hw. Adds half-vehicle-width to the offset so a centered
    #       wide ego already reads non-zero. Easy to misread as lateral
    #       metres when it is not. Kept for backward compatibility with
    #       configs from before 2026-04-27 — emits a DeprecationWarning.
    centerline_usage_mode: str = "baselink"
    # Centerline time-weight floor. The per-step centerline penalty is averaged
    # with weights torch.linspace(1.0, centerline_time_weight_min, T). The default
    # 0.3 matches the historical behavior (late timesteps count 30% of early).
    # Set to 1.0 for a flat (uniform) time-average — recommended when late-curve
    # lane-following matters as much as early, and the training signal is being
    # compressed by the decay.
    centerline_time_weight_min: float = 0.3
    collision_penalty: float = -10.0
    red_light_penalty: float = -10.0
    max_accel: float = 8.0  # m/s^2
    dt: float = 0.1  # 10 Hz
    # Historical avoidance reward behavior ignores moving collisions where a
    # following vehicle hits the ego from behind. Keep that as the generic
    # default, but allow workflows like R2LPL mining/repair to count them.
    ignore_rear_end_collisions: bool = True

    # Road border penalty scales and thresholds
    rb_near_scale: float = 3.0
    rb_wide_scale: float = 0.2
    rb_cont_scale: float = 0.0  # continuous penalty (0=disabled)
    rb_gate_enabled: bool = True  # if True, rb crossing is a hard safety gate
    rb_penalty_mode: str = (
        "frac"  # "frac" = fraction of timesteps, "survival" = first-violation time-decay
    )
    rb_cross_thresh: float = 0.20  # metres — ego perimeter within this = crossing
    rb_near_thresh: float = 0.45  # metres — near zone boundary (+20cm vs lane)
    rb_wide_thresh: float = 0.60  # metres — wide zone boundary (+20cm vs lane)
    rb_cont_thresh: float = 1.00  # metres — continuous penalty max distance (+20cm vs lane)

    # Lane departure penalty scales and thresholds
    enable_lane_departure: bool = False
    lane_gate_enabled: bool = False  # if True, lane crossing kills reward
    lane_near_scale: float = 3.0
    lane_wide_scale: float = 0.2
    lane_cont_scale: float = 0.0
    lane_cross_thresh: float = (
        0.20  # metres — signed distance threshold for crossing (matches rb_cross_thresh)
    )
    lane_near_thresh: float = 0.25  # metres — near zone boundary
    lane_wide_thresh: float = 0.40  # metres — wide zone boundary
    lane_cont_thresh: float = 0.80  # metres — continuous penalty max distance

    # Static-collision (stopped-neighbor clearance) penalty. Same staged
    # pattern as rb_*: gate + near/wide/cont zones. Off by default — enabling
    # changes training reward math, so set `static_collision_enabled=True` and
    # non-zero scales explicitly.
    static_collision_enabled: bool = False
    sc_gate_enabled: bool = (
        False  # hard terminator if any predicted step overlaps a stopped neighbor
    )
    sc_penalty_mode: str = "frac"  # "frac" or "survival" (matches rb_penalty_mode semantics)
    sc_near_scale: float = 0.0
    sc_wide_scale: float = 0.0
    sc_cont_scale: float = 0.0
    sc_cross_thresh: float = 0.2  # clearance below this (metres) = crossing. 0.2 m matches the "visually touching" threshold observed on the bigcurve resim — below that, SAT signed distance is slightly positive but the boxes are in practice a collision.
    sc_near_thresh: float = 0.4
    sc_wide_thresh: float = 0.7
    sc_cont_thresh: float = 1.0
    sc_neighbor_vel_thresh: float = 0.1  # m/s — |v0| below this counts as stationary
    sc_neighbor_disp_thresh: float = (
        0.5  # m — max displacement across GT future below this counts as stationary
    )
    sc_ego_min_speed: float = (
        1.0  # m/s — timesteps below this are not scored (matches collision suppression)
    )

    # Lateral acceleration penalty
    max_lat_accel: float = 2.0  # m/s^2
    lat_accel_scale: float = 3.0

    # Yaw-rate feasibility gate (absolute cap). Thresholds chosen so GT
    # trajectories pass (GT peaks ≈0.5 rad/s on tight human turns) and only
    # clearly unphysical predictions (e.g. pivot-in-place) fail.
    max_yaw_rate: float = 1.0  # rad/s  (2× GT peak)

    # Bicycle-model kinematic feasibility gate. κ_max = tan(max_steer)/wheelbase.
    # Wheelbase is read from ego_shape[0] per scene; max_steer is configured below.
    # Effective curvature bound: kinematic_margin × tan(max_steer) / wheelbase.
    # Margin absorbs SG finite-differencing noise and tight human driving.
    max_steer: float = 0.64  # rad — bicycle-model steering range
    kinematic_margin: float = 2.5  # multiplier over physical bicycle-model bound

    # Overprogress: cap progress at GT path × margin, penalize excess
    enable_overprogress: bool = False
    overprogress_margin: float = 1.1
    overprogress_penalty: float = 0.3
    stopped_penalty: float = 50.0  # applied in compute_reward_batch progress section

    # Underprogress: penalize trajectories that drive much less than a reference.
    underprogress_penalty: float = (
        0.0  # scale (0=disabled). Penalty = scale * max(0, threshold - ratio)
    )
    underprogress_threshold: float = 0.5  # fire when ratio < threshold
    # Reference for underprogress:
    #   "baseline" — (default) frozen LoRA-less baseline det path, passed via
    #                data["baseline_path_len"]. Anchors ratio regardless of
    #                training drift — recommended.
    #   "det"      — deterministic traj (traj[0]) path length. Adaptive but can
    #                collapse when model output itself collapses (path shrinks
    #                while threshold follows it → penalty never fires).
    underprogress_reference: str = "baseline"

    # Progress normalization scale: when enable_overprogress=True, progress is
    # normalized to [0, 1] as fraction of GT, then multiplied by this scale.
    # 100% GT progress → progress_norm_scale points. Default 20.
    progress_norm_scale: float = 20.0

    # Reward aggregation mode:
    # "gate" (default): binary safety gates × quality. Any terminal event → floor (-50).
    # "survival" (PlannerRFT): proportional credit based on how long the trajectory
    #   survives before the first terminal event. A crash at t=60/80 gets 75% of the
    #   quality score. Prevents gradient death on hard scenes where all trajectories fail.
    reward_mode: str = "gate"

    # Optional RL scoring horizon.  HDP found a moderate 4--6 s horizon more
    # useful than ranking an 8 s proposal with a very delayed failure.  Zero
    # keeps the full trajectory (the historical/default behavior).
    reward_horizon_steps: int = 0

    # ``"hdp_pdm"`` selects the published HDP/PDM-style normalized reward:
    # Col×DAC×(5*TTC + 5*EP + 2*C + 4*Speed)/16.  The default keeps the
    # repository's historical signed custom shaping for backwards parity.
    reward_profile: str = "custom"
    # hdp_exact only: which of the reference implementation's two aggregations
    # composes the vendored reward. "weighted_sum" is the paper-exact form its
    # historical runs used -- collisions reach the total only through the risk
    # term (~10% of the weight mass) because w_safety is zero there.
    # "gated_product" is that codebase's own fix: collision, red light and
    # road border multiply the whole reward, i.e. real hard gates.
    hdp_exact_aggregation: str = "weighted_sum"
    # hdp_exact only: reward-term overrides mirroring the reference codebase's
    # own knobs. Defaults reproduce its historical validated runs; the
    # "border" variant (its first measured CLIMBING configuration) sets
    # w_progress 0, w_road_border 1, red light off, occupancy-border off.
    hdp_exact_w_progress: float = 3.0
    hdp_exact_w_road_border: float = 0.0
    hdp_exact_red_light_constraint: bool = True
    hdp_exact_occupancy_use_road_border: bool = True
    pdm_comfort_scale: float = 10.0
    # Original HDP multi-reward post-training profile.  These are the weights
    # reported in the HDP paper (risk, car-following, lane robustness).  The
    # local NPZ adapter computes the same bounded [0, 1] reward families from
    # OBB/TTC, replayed neighbor futures, and route-lane geometry.
    hdp_risk_weight: float = 1.0
    hdp_follow_weight: float = 3.0
    hdp_lane_weight: float = 2.5
    hdp_lane_score_scale: float = 1.0
    hdp_leader_lateral_threshold: float = 3.0
    hdp_desired_time_gap: float = 1.5
    # Continuous near-miss risk adapter.  HDP defines risk from pessimistic
    # TTC/THW/OCC scores rather than only a binary collision label.  The local
    # NPZ scorer exposes the OBB clearance seen within its TTC horizon; this
    # scale maps a 0 m overlap to 0 and a comfortably separated 2 m clearance
    # to 1 while preserving the existing binary collision gate.
    hdp_risk_use_clearance: bool = False
    hdp_risk_clearance_safe_m: float = 2.0
    # Low-speed steering feasibility.  A first step of arc length ``s`` ending
    # ``y`` off the current heading implies a front-wheel angle
    # ``atan(wheel_base * 2|y| / s**2)``; at standstill a 1 cm offset over a 4 cm
    # step is a steering-lock command.  AWR's first-waypoint gate rejects such
    # candidates but never the deterministic anchor, and nothing rewarded the
    # smoother of two survivors, so low-speed behaviour was effectively untrained
    # while the deployed output itself implied a median 1.47 rad on the third of
    # low-speed scenes where it exceeded the limit.
    #
    # ``low_speed_steer_max_rad`` is the physical steering limit, and only the
    # *excess* over it is penalised, normalised by ``pi/2 - limit`` (the implied
    # angle is an ``atan``, so that is the full infeasible range).  Two measured
    # reasons not to ramp from zero instead: it taxes executable creeping turns
    # (an 8 m radius costs 0.048 against a typical 0.013 within-group headroom,
    # on a corpus that oversamples right turns x10), and it saturated 100% of
    # unexecutable candidates at full penalty -- within-group spread exactly
    # 0.0000 -- so AWR could not prefer the least infeasible of two.  Hinging
    # takes low-speed trainable scenes from 8.5% (term off) to 24.1% versus
    # 17.9% for the ramp, at the same rate of selecting an unexecutable
    # candidate.  Weight 0 disables.  Thresholds mirror
    # rlvr/campaign_contract.py's gate.
    low_speed_steer_penalty: float = 1.0
    low_speed_steer_max_rad: float = 0.64
    low_speed_steer_speed_mps: float = 1.0
    low_speed_steer_min_step_m: float = 0.005


__all__ = [
    "RewardConfig",
]
