"""Verbatim port of the audited original-DP ``hdp_pdm`` training reward.

Source: ``Diffusion-Planner-t4-main`` ``rlvr/reward.py`` (branch
``feature/original-dp-awr-post-training``), the profile behind both audited
positive full-corpus AWR results. The two helper functions are copied verbatim;
the aggregation reproduces the exact formula::

    quality  = (5*TTC + 5*EP + 2*lane + 2*comfort) / 14
    terminal = collision_gate * road_border_crossing_gate      (binary)
    reward   = terminal * quality                              in [0, 1]

The port is possible because both repositories share the same
``planner_metrics`` subscore layer (``subscores.py`` is byte-identical and
``compute_subscores_batch`` exposes the identical 32-key contract), which is
also the layer this repository's EPDMS evaluation uses.

Two documented deltas from the source:

- an optional red-light gate (``rl_pdm_red_light_gate``, default on): the
  source data had no traffic-light semantics ("missing traffic-light semantics
  stay neutral"), while Tier IV scenes do, and the source repository's custom
  profile gates red light whenever the signal exists;
- the scoring horizon comes from the shared ``rl_reward_horizon_steps``
  contract instead of a second horizon field (the audited runs used 40).

The held-out selection reward is deliberately NOT switchable to this profile:
checkpoint selection stays on the frozen native evaluation objective so a
training-reward ablation can never grade itself.
"""

from dataclasses import dataclass

import torch

from planner_metrics.aggregate import compute_subscores_batch
from planner_metrics.config import RewardConfig
from planner_metrics.subscores import compute_lane_departure_penalty


@dataclass(frozen=True)
class PDMPortConfig:
    """Knobs the ported helpers read, at the source repository's audited values."""

    dt: float = 0.1
    pdm_comfort_scale: float = 10.0
    hdp_lane_score_scale: float = 1.0
    hdp_leader_lateral_threshold: float = 3.0
    hdp_desired_time_gap: float = 1.5
    hdp_risk_use_clearance: bool = False
    hdp_risk_clearance_safe_m: float = 2.0


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
        return (
            cumulative[nearest]
            + u[torch.arange(points.shape[0], device=device), nearest] * seg_len[nearest]
        )

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


def _hdp_multi_reward_components(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    subs: dict,
    config: PDMPortConfig,
    metrics_config: RewardConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the three bounded reward families used by original HDP-RL.

    Copied verbatim from the source repository (its docstring retained below);
    only the config split (port knobs vs shared metrics config) differs.

    * risk: pessimistic TTC / road-border / OBB safety;
    * follow: leader gap, time gap, speed matching and comfort;
    * lane: route-centerline proximity, masked for logged lane changes.

    Every return value is in [0, 1].  Missing leaders or missing lane geometry
    are neutral (1.0), rather than silently becoming a negative training
    signal.
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
    if config.hdp_risk_use_clearance:
        clearance = subs.get("ttc_min_clearance")
        if isinstance(clearance, torch.Tensor) and clearance.dim() == 2:
            safe_m = max(float(config.hdp_risk_clearance_safe_m), 1e-3)
            clearance_score = (
                (clearance.to(dtype=ego_trajs.dtype) / safe_m).clamp(0.0, 1.0).min(dim=-1).values
            )
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
            # A leader is selected from the first future frame, in front of the
            # ego and within a broad lane corridor. New-corpus futures already
            # start at t+0.1 s (the source loader applied the legacy +1
            # correction to reach the same convention).
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
                        lead_vel = torch.cat([lead_vel[:1], lead_vel], dim=0)
                    else:
                        ego_vel = ego_xy / max(float(config.dt), 1e-3)
                        lead_vel = torch.zeros_like(leader[:, :2])
                    ego_speed = ego_vel.norm(dim=-1).clamp_min(0.0)
                    lead_speed = lead_vel.norm(dim=-1).unsqueeze(0)
                    gap = leader[None, :, 0] - ego_xy[..., 0]
                    desired_gap = 2.0 + 1.5 * ego_speed
                    desired_time_gap = max(float(config.hdp_desired_time_gap), 0.1)
                    time_gap = gap / ego_speed.clamp_min(1.0)

                    spacing = torch.exp(-torch.abs(gap - desired_gap) / (desired_gap + 2.0))
                    time_gap_score = torch.exp(-torch.abs(time_gap - desired_time_gap) / 1.5)
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

    # Paper intent: do not punish an intentional lane change.  The logged
    # expert leaving the union of nearby lane polygons masks the lane term.
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
            expert_shape = data.get("ego_shape")
            if isinstance(expert_shape, torch.Tensor):
                if expert_shape.dim() == 2:
                    expert_shape = expert_shape[0]
                expert_lane = compute_lane_departure_penalty(
                    gt4.unsqueeze(0),
                    expert_shape[:3].to(device),
                    data,
                    config=metrics_config,
                )
                expert_lane_steps = expert_lane[3]
            else:
                expert_lane_steps = [None]
            if (
                isinstance(expert_lane_steps, list)
                and expert_lane_steps
                and expert_lane_steps[0] is not None
            ):
                lane_score = torch.ones_like(lane_score)

    return risk_score, follow_score, lane_score


@torch.no_grad()
def compute_pdm_port_reward(
    ego_world: torch.Tensor,
    scene_inputs: dict[str, torch.Tensor],
    neighbors_future: torch.Tensor,
    num_scenes: int,
    n: int,
    args,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The audited ``hdp_pdm`` total, with the same interface as the native reward."""
    horizon = int(getattr(args, "rl_reward_horizon_steps", 0) or 0)
    if horizon > 0 and horizon < ego_world.shape[1]:
        ego_world = ego_world[:, :horizon]
        neighbors_future = neighbors_future[:, :, :horizon]
        scene_inputs = dict(scene_inputs)
        scene_inputs["ego_agent_future"] = scene_inputs["ego_agent_future"][:, :horizon]
    if ego_world.shape[0] != num_scenes * n:
        raise ValueError(
            f"pdm-port reward expects {num_scenes * n} candidates, got {ego_world.shape[0]}"
        )
    port_config = PDMPortConfig()
    metrics_config = RewardConfig()
    red_light_gated = bool(getattr(args, "rl_pdm_red_light_gate", True))
    device = ego_world.device
    grouped = ego_world.view(num_scenes, n, ego_world.shape[1], 4)

    scene_keys = (
        "ego_shape",
        "lanes",
        "route_lanes",
        "line_strings",
        "polygons",
        "static_objects",
        "goal_pose",
        "ego_current_state",
        "turn_indicators",
    )
    totals = []
    components: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "reward_pdm_ttc_score",
            "reward_pdm_ep_score",
            "reward_pdm_lane_score",
            "reward_pdm_comfort_score",
            "reward_pdm_terminal_gate",
            "reward_pdm_red_light_gate",
        )
    }
    for scene in range(num_scenes):
        data = {key: scene_inputs[key][scene] for key in scene_keys if key in scene_inputs}
        data["neighbor_agents_future"] = neighbors_future[scene]
        data["ego_agent_future"] = scene_inputs["ego_agent_future"][scene]
        candidates = grouped[scene]
        subs = compute_subscores_batch(candidates, data, metrics_config)

        ttc = subs["ttc"].to(dtype=candidates.dtype).clamp(0.0, 1.0)
        ep = _expert_route_progress_ratio(candidates, data["ego_agent_future"])
        _, _, lane = _hdp_multi_reward_components(
            candidates, data, subs, port_config, metrics_config
        )
        comfort = torch.exp(
            subs["comfort"].to(dtype=candidates.dtype).clamp(-50.0, 0.0)
            / max(port_config.pdm_comfort_scale, 1e-3)
        )
        comfort = (
            comfort * torch.exp(subs["feasibility"].to(dtype=candidates.dtype).clamp(-10.0, 0.0))
        ).clamp(0.0, 1.0)

        quality = (5.0 * ttc + 5.0 * ep + 2.0 * lane.clamp(0.0, 1.0) + 2.0 * comfort) / 14.0
        collision_gate = torch.tensor(
            [1.0 if step is None else 0.0 for step in subs["collision_step"]],
            device=device,
            dtype=candidates.dtype,
        )
        rb_gate = subs["rb_crossing_gate"].to(dtype=candidates.dtype)
        terminal = collision_gate * rb_gate
        red_gate = torch.ones_like(terminal)
        if red_light_gated:
            red_gate = (subs["red_light"].to(dtype=candidates.dtype) >= -0.5).to(candidates.dtype)
            terminal = terminal * red_gate
        totals.append(terminal * quality)
        components["reward_pdm_ttc_score"].append(ttc)
        components["reward_pdm_ep_score"].append(ep)
        components["reward_pdm_lane_score"].append(lane)
        components["reward_pdm_comfort_score"].append(comfort)
        components["reward_pdm_terminal_gate"].append(terminal)
        components["reward_pdm_red_light_gate"].append(red_gate)

    reward = torch.cat(totals, dim=0)
    metrics = {key: torch.cat(values, dim=0).mean() for key, values in components.items()}
    return reward, metrics
