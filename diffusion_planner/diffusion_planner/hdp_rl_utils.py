"""HDP reward-weighted RL-Hybrid building blocks for the diffusion planner.

This module is the RL counterpart of ``train_epoch.py``/``decoder.compute_training_loss``.
The single supported pipeline is:

  1. ``expand_batch``        - replicate every scene in the batch ``N`` times so we can
                               draw a *group* of ``N`` trajectories per scene in a single
                               multi-batch inference pass.
  2. ``sample_group``        - run the model in inference mode with random initial noise to
                               produce ``N`` diverse ego trajectories per scene.
  3. ``compute_hdp_reward`` - score each trajectory with the HDP safety/risk/follow/lane
                               structure, adapted to the signals available in Tier IV NPZs.
  4. ``compute_reward_weighted_loss`` - apply exp(beta * group-normalized reward)
                               to the HDP hybrid diffusion loss.
"""

from dataclasses import dataclass

import torch

from diffusion_planner.dimensions import OUTPUT_T, POSE_DIM
from diffusion_planner.loss import (
    hybrid_loss_components,
    inverse_normalize_ego_velocity,
    normalize_ego_state,
    normalize_ego_velocity,
    sample_diffusion_time,
    waypoints_to_velocity,
)
from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear
from planner_metrics.config import RewardConfig
from planner_metrics.geometry import _point_to_segments_min_dist
from planner_metrics.subscores import (
    compute_ego_neighbor_signed_clearance,
    compute_road_border_penalty,
)

_RL_REWARD_CONFIG = RewardConfig()


@dataclass(frozen=True)
class HDPRewardConfig:
    """Published HDP reward structure plus explicitly assumed shaping thresholds.

    The paper defines the aggregation and qualitative shaping behavior, but does not publish
    the numerical speed-adaptive shaping functions. These defaults are therefore local,
    checkpointed assumptions and must not be described as exact paper hyperparameters.
    """

    dt: float = 0.1
    ttc_critical_s: float = 0.5
    ttc_safe_s: float = 3.0
    thw_critical_s: float = 0.5
    thw_safe_s: float = 2.0
    occupancy_critical_m: float = 0.25
    occupancy_safe_m: float = 2.0
    occupancy_speed_gain_s: float = 0.10
    lane_half_width_m: float = 1.75
    leader_lateral_margin_m: float = 0.75
    rear_end_penalty: float = 0.3


def _hdp_reward_config(args) -> HDPRewardConfig:
    return HDPRewardConfig(
        dt=float(getattr(args, "rl_reward_dt", 0.1)),
        ttc_critical_s=float(getattr(args, "rl_ttc_critical_s", 0.5)),
        ttc_safe_s=float(getattr(args, "rl_ttc_safe_s", 3.0)),
        thw_critical_s=float(getattr(args, "rl_thw_critical_s", 0.5)),
        thw_safe_s=float(getattr(args, "rl_thw_safe_s", 2.0)),
        occupancy_critical_m=float(getattr(args, "rl_occupancy_critical_m", 0.25)),
        occupancy_safe_m=float(getattr(args, "rl_occupancy_safe_m", 2.0)),
        occupancy_speed_gain_s=float(getattr(args, "rl_occupancy_speed_gain_s", 0.10)),
        lane_half_width_m=float(getattr(args, "rl_lane_half_width_m", 1.75)),
        leader_lateral_margin_m=float(getattr(args, "rl_leader_lateral_margin_m", 0.75)),
    )


def _linear_safe_score(
    value: torch.Tensor,
    critical: float | torch.Tensor,
    safe: float | torch.Tensor,
) -> torch.Tensor:
    critical_t = torch.as_tensor(critical, dtype=value.dtype, device=value.device)
    safe_t = torch.as_tensor(safe, dtype=value.dtype, device=value.device)
    return ((value - critical_t) / (safe_t - critical_t).clamp_min(1e-6)).clamp(0.0, 1.0)


def _attenuate_rear_only_risk(
    risk: torch.Tensor,
    collision_rear: torch.Tensor,
    collision_active: torch.Tensor,
    rear_end_penalty: float,
) -> torch.Tensor:
    """Apply the paper's rear-end attenuation to the final conservative risk score."""
    rear_only = collision_rear.bool() & ~collision_active.bool()
    rear_floor = 1.0 - float(rear_end_penalty)
    return torch.where(rear_only, risk.clamp_min(rear_floor), risk)


def _trajectory_speed(xy: torch.Tensor, dt: float, initial_xy: torch.Tensor | None = None):
    if initial_xy is None:
        first = torch.zeros_like(xy[..., :1, :])
    else:
        first = initial_xy[..., None, :]
    step = torch.diff(torch.cat([first, xy], dim=-2), dim=-2)
    return step.norm(dim=-1) / dt


def _scene_neighbors(
    neighbors_future: torch.Tensor,
    neighbor_past: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return valid future poses, shapes, per-step validity and initial positions."""
    T = neighbors_future.shape[1]
    # Padding is explicitly zeroed after heading conversion, while a real pose has a unit
    # (cos, sin) vector even when its center crosses the ego origin.
    pose_valid = neighbors_future[..., 2:4].norm(dim=-1) > 0.5
    slot_valid = pose_valid.any(dim=1)
    future = neighbors_future[slot_valid, :T, :4]
    past = neighbor_past[slot_valid]
    valid = future[..., 2:4].norm(dim=-1) > 0.5
    if future.shape[0] == 0:
        return (
            future,
            future.new_zeros((0, 2)),
            valid,
            future.new_zeros((0, 2)),
            torch.zeros(0, dtype=torch.bool, device=future.device),
        )

    shapes = past[:, -1, [6, 7]].abs()  # width, length
    fallback = shapes.new_tensor([2.0, 4.5])
    shapes = torch.where((shapes > 1e-3).all(dim=-1, keepdim=True), shapes, fallback)
    initial_xy = past[:, -1, :2]
    is_vehicle = past[:, -1, 8:11].argmax(dim=-1) == 0
    return future, shapes, valid, initial_xy, is_vehicle


def _collision_and_leader_terms(
    ego_trajs: torch.Tensor,
    ego_shape: torch.Tensor,
    neighbor_futures: torch.Tensor,
    neighbor_shapes: torch.Tensor,
    neighbor_valid: torch.Tensor,
    neighbor_initial_xy: torch.Tensor,
    neighbor_is_vehicle: torch.Tensor,
    config: HDPRewardConfig,
) -> dict[str, torch.Tensor]:
    """Compute paper safety, TTC, THW and leader-conditioned following terms."""
    C, T, _ = ego_trajs.shape
    device = ego_trajs.device
    ego_xy = ego_trajs[..., :2]
    heading = ego_trajs[..., 2:4]
    heading = heading / heading.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    ego_speed = _trajectory_speed(ego_xy, config.dt)

    if neighbor_futures.shape[0] == 0:
        ones = torch.ones(C, T, device=device)
        return {
            "safety": torch.ones(C, device=device),
            "ttc": ones,
            "thw": ones,
            "follow": torch.ones(C, device=device),
            "leader_fraction": torch.zeros(C, device=device),
            "collision_active": torch.zeros(C, device=device),
            "collision_rear": torch.zeros(C, device=device),
            "stopped_clearance": torch.full((C, T), float("inf"), device=device),
            "stopped_available": torch.tensor(False, device=device),
        }

    clearance = compute_ego_neighbor_signed_clearance(
        ego_trajs,
        ego_shape,
        neighbor_futures,
        neighbor_shapes,
        neighbor_valid,
    )
    collision = clearance < 0.0
    ego_center = ego_xy + 0.5 * ego_shape[0] * heading
    rel_xy = neighbor_futures[None, :, :, :2] - ego_center[:, None, :, :]
    longitudinal = (rel_xy * heading[:, None]).sum(dim=-1)
    lateral = torch.abs(
        rel_xy[..., 0] * -heading[:, None, :, 1] + rel_xy[..., 1] * heading[:, None, :, 0]
    )
    npc_behind = longitudinal < 0.0
    rear_collision = collision & npc_behind
    active_collision = collision & ~npc_behind

    per_step_penalty = torch.where(
        active_collision.any(dim=1),
        torch.ones(C, T, device=device),
        torch.where(
            rear_collision.any(dim=1),
            torch.full((C, T), config.rear_end_penalty, device=device),
            torch.zeros(C, T, device=device),
        ),
    )
    safety = 1.0 - per_step_penalty.max(dim=-1).values

    # Continuous TTC from OBB clearance closing rate. This remains informative for near misses
    # that do not actually overlap within the finite logged horizon.
    clearance_delta = clearance[..., :-1] - clearance[..., 1:]
    closing_rate = clearance_delta / config.dt
    valid_pair = neighbor_valid[:, :-1] & neighbor_valid[:, 1:]
    per_neighbor_ttc = torch.full_like(clearance, config.ttc_safe_s)
    per_neighbor_ttc[..., :-1] = torch.where(
        valid_pair[None] & (closing_rate > 1e-3),
        clearance[..., :-1].clamp_min(0.0) / closing_rate.clamp_min(1e-3),
        per_neighbor_ttc[..., :-1],
    )
    per_neighbor_ttc = torch.where(collision, torch.zeros_like(per_neighbor_ttc), per_neighbor_ttc)
    ttc_value, ttc_neighbor = per_neighbor_ttc.min(dim=1)
    ttc_is_rear = torch.gather(npc_behind, 1, ttc_neighbor[:, None]).squeeze(1)

    ttc_score = _linear_safe_score(
        ttc_value,
        config.ttc_critical_s,
        config.ttc_safe_s,
    )
    ttc_score = torch.where(
        ttc_is_rear,
        1.0 - config.rear_end_penalty * (1.0 - ttc_score),
        ttc_score,
    )

    ego_length = ego_shape[1].abs().clamp_min(1.0)
    ego_width = ego_shape[2].abs().clamp_min(1.0)
    npc_width = neighbor_shapes[:, 0]
    npc_length = neighbor_shapes[:, 1]
    bumper_gap = longitudinal - 0.5 * (ego_length + npc_length[None, :, None])
    lateral_limit = 0.5 * (ego_width + npc_width[None, :, None]) + config.leader_lateral_margin_m
    leader_candidate = (
        neighbor_valid[None]
        & neighbor_is_vehicle[None, :, None]
        & (longitudinal > 0.0)
        & (lateral <= lateral_limit)
    )
    candidate_gap = bumper_gap.clamp_min(0.0).masked_fill(~leader_candidate, float("inf"))
    leader_gap, leader_index = candidate_gap.min(dim=1)
    leader_present = torch.isfinite(leader_gap)

    neighbor_speed = _trajectory_speed(
        neighbor_futures[..., :2],
        config.dt,
        initial_xy=neighbor_initial_xy,
    )
    both_valid_01 = neighbor_valid[:, 0] & neighbor_valid[:, 1]
    future_displacement = (neighbor_futures[..., :2] - neighbor_futures[:, :1, :2]).norm(dim=-1)
    max_displacement = future_displacement.masked_fill(~neighbor_valid, 0.0).max(dim=-1).values
    stopped_mask = (
        both_valid_01
        & (neighbor_speed[:, 1] < _RL_REWARD_CONFIG.sc_neighbor_vel_thresh)
        & (max_displacement < _RL_REWARD_CONFIG.sc_neighbor_disp_thresh)
    )
    stopped_clearance = (
        clearance[:, stopped_mask].min(dim=1).values
        if stopped_mask.any()
        else torch.full((C, T), float("inf"), device=device)
    )
    selected_leader_speed = torch.gather(
        neighbor_speed[None].expand(C, -1, -1),
        1,
        leader_index[:, None, :],
    ).squeeze(1)
    selected_leader_speed = torch.where(leader_present, selected_leader_speed, ego_speed)

    thw = leader_gap / ego_speed.clamp_min(0.1)
    thw_score = _linear_safe_score(thw, config.thw_critical_s, config.thw_safe_s)
    thw_score = torch.where(leader_present & (ego_speed > 0.1), thw_score, 1.0)

    # The four published following aspects. Numerical shaping constants are explicit local
    # assumptions because the paper/release does not provide them.
    ideal_gap_s = 1.0 + ego_speed.clamp(0.0, 20.0) / 20.0
    gap_score = (1.0 - torch.abs(thw - ideal_gap_s) / 1.5).clamp(0.0, 1.0)
    target_spacing = 2.0 + ideal_gap_s * ego_speed
    spacing_tolerance = (5.0 + 0.5 * target_spacing).clamp_min(1.0)
    spacing_score = (1.0 - torch.abs(leader_gap - target_spacing) / spacing_tolerance).clamp(
        0.0, 1.0
    )
    speed_score = (1.0 - torch.abs(ego_speed - selected_leader_speed) / 5.0).clamp(0.0, 1.0)
    acceleration = torch.diff(torch.cat([ego_speed[:, :1], ego_speed], dim=-1), dim=-1) / config.dt
    comfort_score = _linear_safe_score(6.0 - acceleration.abs(), 0.0, 4.0)

    aspect_sum = gap_score + spacing_score + speed_score + comfort_score
    aspect_sum = torch.where(leader_present, aspect_sum, torch.full_like(aspect_sum, 4.0))
    follow = aspect_sum.mean(dim=-1) / 4.0
    return {
        "safety": safety,
        "ttc": ttc_score,
        "thw": thw_score,
        "follow": follow,
        "leader_fraction": leader_present.float().mean(dim=-1),
        "collision_active": active_collision.any(dim=(1, 2)).float(),
        "collision_rear": rear_collision.any(dim=(1, 2)).float(),
        "stopped_clearance": stopped_clearance,
        "stopped_available": stopped_mask.any(),
    }


def _occupancy_score(
    ego_trajs: torch.Tensor,
    ego_shape: torch.Tensor,
    static_objects: torch.Tensor | None,
    line_strings: torch.Tensor | None,
    stopped_clearance: torch.Tensor,
    stopped_available: torch.Tensor,
    config: HDPRewardConfig,
) -> tuple[torch.Tensor, dict[str, bool]]:
    """Fuse available static boxes, stopped agents and road-border clearance."""
    C, T, _ = ego_trajs.shape
    device = ego_trajs.device
    source_clearances: list[torch.Tensor] = []
    sources = {"static": False, "stopped": False, "road_border": False}

    if static_objects is not None:
        valid = (static_objects[..., :2].abs().sum(dim=-1) > 1e-6) | (
            static_objects[..., 4:6].abs().sum(dim=-1) > 1e-6
        )
        objects = static_objects[valid]
        if objects.shape[0] > 0:
            poses = objects[:, None, :4].expand(-1, T, -1).clone()
            heading_norm = poses[..., 2:4].norm(dim=-1, keepdim=True)
            poses[..., 2:4] = torch.where(
                heading_norm > 1e-6,
                poses[..., 2:4] / heading_norm.clamp_min(1e-6),
                poses.new_tensor([1.0, 0.0]),
            )
            shapes = objects[:, [4, 5]].abs()  # width, length
            shapes = torch.where(
                (shapes > 1e-3).all(dim=-1, keepdim=True),
                shapes,
                shapes.new_tensor([1.0, 1.0]),
            )
            object_valid = torch.ones(objects.shape[0], T, dtype=torch.bool, device=device)
            source_clearances.append(
                compute_ego_neighbor_signed_clearance(
                    ego_trajs,
                    ego_shape,
                    poses,
                    shapes,
                    object_valid,
                )
                .min(dim=1)
                .values
            )
            sources["static"] = True

    if bool(stopped_available.item()):
        source_clearances.append(stopped_clearance)
        sources["stopped"] = True

    # Road borders are a fallback for the current corpus, not literal occupancy. Avoid the
    # expensive perimeter-to-segment kernel when a stronger static source already exists.
    if line_strings is not None and not source_clearances:
        road_border = compute_road_border_penalty(
            ego_trajs,
            ego_shape,
            {"line_strings": line_strings},
            _RL_REWARD_CONFIG,
        )[-1]
        if torch.isfinite(road_border).any():
            source_clearances.append(road_border)
            sources["road_border"] = True

    if not source_clearances:
        return torch.ones(C, T, device=device), sources

    clearance = torch.stack(source_clearances, dim=0).amin(dim=0)
    ego_speed = _trajectory_speed(ego_trajs[..., :2], config.dt)
    safe_distance = config.occupancy_safe_m + config.occupancy_speed_gain_s * ego_speed
    critical_distance = config.occupancy_critical_m + config.occupancy_speed_gain_s * ego_speed
    return _linear_safe_score(clearance, critical_distance, safe_distance), sources


def _nearest_lane_distance(points: torch.Tensor, lanes: torch.Tensor) -> torch.Tensor:
    center = lanes[..., :2]
    valid = center.abs().sum(dim=-1) > 1e-6
    valid_pair = valid[..., :-1] & valid[..., 1:]
    seg_start = center[..., :-1, :][valid_pair]
    seg_end = center[..., 1:, :][valid_pair]
    if seg_start.shape[0] == 0:
        lane_points = center[valid]
        if lane_points.shape[0] == 0:
            raise RuntimeError("HDP lane reward requires valid lane centerline points.")
        distance = torch.cdist(points.reshape(-1, 2), lane_points).min(dim=-1).values
    else:
        distance = _point_to_segments_min_dist(points.reshape(-1, 2), seg_start, seg_end)
    return distance.reshape(points.shape[:-1])


def _hdp_lane_score(
    ego_trajs: torch.Tensor,
    expert_future: torch.Tensor,
    lanes: torch.Tensor,
    turn_indicators: torch.Tensor | None,
    config: HDPRewardConfig,
) -> tuple[torch.Tensor, bool, bool]:
    if not (lanes[..., :2].abs().sum(dim=-1) > 1e-6).any():
        return torch.zeros(ego_trajs.shape[0], device=ego_trajs.device), True, False
    pred_distance = _nearest_lane_distance(ego_trajs[..., :2], lanes)
    center_score = (1.0 - pred_distance / config.lane_half_width_m).clamp(0.0, 1.0)

    expert_xy = expert_future[..., :2]
    expert_valid = expert_future[..., 2:4].norm(dim=-1) > 0.5
    if not expert_valid.any():
        return torch.zeros(ego_trajs.shape[0], device=ego_trajs.device), True, False
    expert_distance = _nearest_lane_distance(expert_xy[expert_valid], lanes)
    expert_off_lane = bool(
        (
            (expert_distance.mean() > config.lane_half_width_m)
            | (expert_distance.max() > 2.0 * config.lane_half_width_m)
        ).item()
    )

    indicator_active = bool(
        turn_indicators is not None
        and turn_indicators.numel() > 0
        and int(turn_indicators[-1].item()) in (2, 3)
    )
    # Measure lateral motion in the road frame. Ego-frame endpoint y is not a lateral offset on a
    # curve and used to classify ordinary bends as lane changes. A centerline-to-centerline change
    # has a pronounced distance peak while both ends remain close to some lane centerline. When a
    # turn indicator is active, a one-sided distance increase is sufficient because the target
    # centerline can be absent from the cropped lane tensor.
    edge_window = min(3, expert_distance.numel())
    start_distance = expert_distance[:edge_window].mean()
    end_distance = expert_distance[-edge_window:].mean()
    peak_distance = expert_distance.max()
    starts_on_lane = start_distance < 0.5 * config.lane_half_width_m
    centerline_change = (
        starts_on_lane
        & (end_distance < 0.5 * config.lane_half_width_m)
        & (peak_distance > 0.75 * config.lane_half_width_m)
    )
    indicated_departure = starts_on_lane & (
        peak_distance - start_distance > 0.5 * config.lane_half_width_m
    )
    expert_lane_change = bool((centerline_change | (indicator_active & indicated_departure)).item())
    masked = expert_off_lane or expert_lane_change
    if masked:
        return (
            torch.zeros(ego_trajs.shape[0], device=ego_trajs.device),
            expert_off_lane,
            expert_lane_change,
        )
    return center_score.mean(dim=-1), False, False


@torch.no_grad()
def compute_hdp_reward(
    ego_world: torch.Tensor,
    scene_inputs: dict[str, torch.Tensor],
    neighbors_future: torch.Tensor,
    num_scenes: int,
    n: int,
    args,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the real-vehicle-oriented HDP reward on Tier IV scene tensors."""
    config = _hdp_reward_config(args)
    ego_group = ego_world.reshape(num_scenes, n, ego_world.shape[1], ego_world.shape[2])
    if neighbors_future.shape[0] != num_scenes:
        raise ValueError(
            "HDP reward expects one logged-neighbor tensor per scene, got "
            f"{neighbors_future.shape[0]} tensors for {num_scenes} scenes"
        )
    neighbor_group = neighbors_future

    metric_lists: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "safety",
            "risk",
            "follow",
            "lane",
            "ttc",
            "thw",
            "occupancy",
            "leader_fraction",
            "collision_active",
            "collision_rear",
            "lane_masked",
            "lane_change_masked",
            "occupancy_any_source",
            "occupancy_static_source",
            "occupancy_stopped_source",
            "occupancy_road_border_source",
        )
    }
    rewards = []

    required = ("ego_shape", "neighbor_agents_past", "ego_agent_future", "lanes")
    missing = [key for key in required if key not in scene_inputs]
    if missing:
        raise ValueError(f"HDP reward is missing required scene tensors: {missing}")

    for scene in range(num_scenes):
        ego_shape = scene_inputs["ego_shape"][scene, :3]
        nf, neighbor_shapes, neighbor_valid, neighbor_initial, neighbor_is_vehicle = (
            _scene_neighbors(neighbor_group[scene], scene_inputs["neighbor_agents_past"][scene])
        )
        terms = _collision_and_leader_terms(
            ego_group[scene],
            ego_shape,
            nf,
            neighbor_shapes,
            neighbor_valid,
            neighbor_initial,
            neighbor_is_vehicle,
            config,
        )
        occupancy, occupancy_sources = _occupancy_score(
            ego_group[scene],
            ego_shape,
            scene_inputs.get("static_objects", None)[scene]
            if "static_objects" in scene_inputs
            else None,
            scene_inputs.get("line_strings", None)[scene]
            if "line_strings" in scene_inputs
            else None,
            terms["stopped_clearance"],
            terms["stopped_available"],
            config,
        )
        risk = torch.stack([terms["ttc"], terms["thw"], occupancy], dim=0).amin(dim=(0, 2))
        risk = _attenuate_rear_only_risk(
            risk,
            terms["collision_rear"],
            terms["collision_active"],
            config.rear_end_penalty,
        )
        lane, expert_off_lane, expert_lane_change = _hdp_lane_score(
            ego_group[scene],
            heading_to_cos_sin_if_needed(scene_inputs["ego_agent_future"][scene]),
            scene_inputs["lanes"][scene],
            scene_inputs.get("turn_indicators", None)[scene]
            if "turn_indicators" in scene_inputs
            else None,
            config,
        )
        reward = (
            args.rl_reward_w_risk * risk
            + args.rl_reward_w_follow * terms["follow"]
            + args.rl_reward_w_lane * lane
        )
        rewards.append(reward)
        metric_lists["safety"].append(terms["safety"])
        metric_lists["risk"].append(risk)
        metric_lists["follow"].append(terms["follow"])
        metric_lists["lane"].append(lane)
        metric_lists["ttc"].append(terms["ttc"].amin(dim=-1))
        metric_lists["thw"].append(terms["thw"].amin(dim=-1))
        metric_lists["occupancy"].append(occupancy.amin(dim=-1))
        metric_lists["leader_fraction"].append(terms["leader_fraction"])
        metric_lists["collision_active"].append(terms["collision_active"])
        metric_lists["collision_rear"].append(terms["collision_rear"])
        metric_lists["lane_masked"].append(lane.new_full((n,), float(expert_off_lane)))
        metric_lists["lane_change_masked"].append(lane.new_full((n,), float(expert_lane_change)))
        metric_lists["occupancy_any_source"].append(
            lane.new_full((n,), float(any(occupancy_sources.values())))
        )
        for metric_key, source_key in (
            ("occupancy_static_source", "static"),
            ("occupancy_stopped_source", "stopped"),
            ("occupancy_road_border_source", "road_border"),
        ):
            metric_lists[metric_key].append(
                lane.new_full((n,), float(occupancy_sources[source_key]))
            )

    flattened = {key: torch.cat(values) for key, values in metric_lists.items()}
    metrics = {f"reward_{key}_score": value.mean() for key, value in flattened.items()}
    return torch.cat(rewards), metrics


def heading_to_cos_sin_if_needed(trajectory: torch.Tensor) -> torch.Tensor:
    if trajectory.shape[-1] >= 4:
        return trajectory[..., :4]
    if trajectory.shape[-1] != 3:
        raise ValueError(f"Expected trajectory last dimension 3 or >=4, got {trajectory.shape}")
    padding = trajectory[..., :3].abs().sum(dim=-1) == 0
    converted = torch.cat(
        [trajectory[..., :2], trajectory[..., 2:3].cos(), trajectory[..., 2:3].sin()], dim=-1
    )
    return converted.masked_fill(padding.unsqueeze(-1), 0.0)


def expand_batch(inputs: dict[str, torch.Tensor], n: int) -> dict[str, torch.Tensor]:
    """Replicate every scene ``n`` times along the batch dimension.

    Scene ``b`` ends up occupying rows ``[b * n, (b + 1) * n)`` so a group of ``n`` samples
    for the same scene is contiguous (handy for the group-relative advantage reshape).
    """
    return {key: value.repeat_interleave(n, dim=0) for key, value in inputs.items()}


@torch.no_grad()
def sample_group(
    model,
    norm_inputs: dict[str, torch.Tensor],
    noise_scale: float,
    device: torch.device,
    *,
    scene_norm_inputs: dict[str, torch.Tensor] | None = None,
    group_size: int = 1,
    use_bf16: bool = False,
    sample_steps: int | None = None,
    return_encoding: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
    """Generate one ego trajectory per (already replicated) row via inference sampling.

    Args:
        model: the Diffusion_Planner (or DDP-wrapped) model.
        norm_inputs: observation-normalized inputs, batch dimension already expanded.
        noise_scale: rollout sampling temperature. ``0`` stays deterministic.
        device: target device.

    Returns:
        ego_world: [B*N, T, 4] ego trajectories in the ego-centric world frame
            (x, y, cos_yaw, sin_yaw).
    """
    was_training = model.training
    model.eval()

    B = norm_inputs["ego_current_state"].shape[0]
    net = getattr(model, "module", model)
    action_agent_num = 1 + net.decoder._predicted_neighbor_num
    inference_inputs = dict(norm_inputs)
    # Official HDP-RL samples rollouts with a fixed sampling temperature.
    if noise_scale == 0.0:
        sampled = torch.zeros(B, action_agent_num, OUTPUT_T, POSE_DIM, device=device)
    else:
        sampled = torch.randn(B, action_agent_num, OUTPUT_T, POSE_DIM, device=device) * float(
            noise_scale
        )
    inference_inputs["sampled_trajectories"] = sampled

    cached_encoding = None
    cached_global_route_condition = None
    previous_sample_steps = net.decoder._sample_steps
    if sample_steps is not None:
        net.decoder._sample_steps = int(sample_steps)
    try:
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=use_bf16 and torch.device(device).type == "cuda",
        ):
            if scene_norm_inputs is not None:
                cached_encoding = net.encoder(scene_norm_inputs).repeat_interleave(
                    group_size, dim=0
                )
                cached_global_route_condition = net.decoder.global_route_encoder(
                    scene_norm_inputs["route_lanes"]
                ).repeat_interleave(group_size, dim=0)
                inference_inputs["_cached_global_route_condition"] = (
                    cached_global_route_condition
                )
                outputs = net.decoder(cached_encoding, inference_inputs)
            else:
                _, outputs = model(inference_inputs)
    finally:
        net.decoder._sample_steps = previous_sample_steps
    ego_world = outputs["prediction"][:, 0].detach().float()  # [B*N, T, 4]

    if was_training:
        model.train()
    if return_encoding:
        return ego_world, cached_encoding
    return ego_world


def compute_reward_weights(
    reward: torch.Tensor,
    num_scenes: int,
    n: int,
    normalize: str,
    beta: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if n < 2:
        raise ValueError("HDP-RL requires num_generations >= 2 for group reward normalization.")
    grouped = reward.view(num_scenes, n)
    group_std = grouped.std(dim=1, keepdim=True)
    finite_group = torch.isfinite(grouped).all(dim=1, keepdim=True)
    # This filter is part of the paper's RL procedure, independently of the normalization
    # ablation: a scene with identical candidate rewards contains no action preference signal.
    valid_group = finite_group & torch.isfinite(group_std) & (group_std > eps)
    if normalize == "group":
        mean = grouped.mean(dim=1, keepdim=True)
        # The HDP paper discards scenes whose candidate actions all receive the
        # same reward. Keeping them at exp(0) == 1 silently turns those scenes
        # into unweighted self-distillation and can dominate the useful groups.
        reward_norm = torch.where(
            valid_group,
            (grouped - mean) / (group_std + eps),
            torch.zeros_like(grouped),
        ).reshape(-1)
        valid_sample = valid_group.expand(-1, n).reshape(-1)
    elif normalize == "batch":
        valid_sample = valid_group.expand(-1, n).reshape(-1)
        if valid_sample.any():
            finite_reward = reward[valid_sample]
            std = finite_reward.std()
            if torch.isfinite(std) and std > eps:
                reward_norm = (reward - finite_reward.mean()) / (std + eps)
            else:
                reward_norm = torch.zeros_like(reward)
        else:
            reward_norm = torch.zeros_like(reward)
    elif normalize == "none":
        reward_norm = reward
        valid_sample = valid_group.expand(-1, n).reshape(-1)
    else:
        raise ValueError(f"Unsupported rl_reward_normalize={normalize!r}")
    reward_norm = torch.nan_to_num(reward_norm, nan=0.0, posinf=0.0, neginf=0.0)
    weights = torch.where(
        valid_sample, torch.exp(beta * reward_norm), torch.zeros_like(reward_norm)
    )
    return weights.detach(), valid_sample


def distributed_valid_sample_count(
    valid_sample: torch.Tensor,
    use_ddp: bool,
) -> tuple[torch.Tensor, int]:
    """Return global valid-sample count and DDP world size.

    DDP averages gradients across ranks. Scaling each rank by its local valid count would therefore
    overweight ranks with sparse valid groups. The global count gives the exact global sample mean.
    """
    count = valid_sample.sum().to(dtype=torch.float32)
    world_size = 1
    if use_ddp and torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)
    return count, world_size


def _compute_policy_ego_loss_per_sample(
    model,
    norm_inputs: dict[str, torch.Tensor],
    ego_pseudo_gt: torch.Tensor,
    args,
    cached_encoding: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if not args.use_velocity_representation:
        raise ValueError("HDP RL requires velocity representation")
    if args.diffusion_model_type != "x_start" or args.diffusion_supervision_type != "x_start":
        raise ValueError("HDP RL requires x_start prediction and supervision")
    norm = args.state_normalizer
    ego_target = ego_pseudo_gt.detach()

    B, T, _ = ego_target.shape
    device = ego_pseudo_gt.device

    gt_future = ego_target[:, None]  # [B, 1, T, 4]

    eps = 1e-3
    t = sample_diffusion_time(
        B,
        device,
        eps,
        getattr(args, "diffusion_time_sample_method", "uniform"),
    )
    t_broadcast = t.view(B, 1, 1, 1)
    z = torch.randn_like(gt_future)

    waypoint_gt = normalize_ego_state(gt_future, norm)
    ego_velocity_gt = waypoints_to_velocity(ego_target)
    all_gt = normalize_ego_velocity(ego_velocity_gt, norm)[:, None]

    model_ref = getattr(model, "module", model)
    sde = getattr(model_ref, "sde", None)
    if sde is None:
        sde = VPSDE_linear()
    # Same schedule values as marginal_prob(ones_like(...)) without the full-size ones tensor.
    alpha = sde.marginal_alpha(t_broadcast)
    std = sde.marginal_prob_std(t_broadcast)
    x0_target = all_gt
    xT = alpha * x0_target + std * z
    merged_inputs = {
        **norm_inputs,
        "gt_trajectories": all_gt,
        "turn_indicator_trajectories": waypoint_gt,
        "sampled_trajectories": xT,
        "diffusion_time": t,
        # RL optimizes only the trajectory policy. Avoid an unused turn-head forward and keep
        # that SFT classifier frozen while the DiT policy changes under reward weighting.
        "_skip_turn_indicator_training": True,
    }
    if cached_encoding is not None:
        merged_inputs["_cached_encoding"] = cached_encoding
    # Same amp scoping as decoder.compute_training_loss: forward only, losses stay fp32.
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=getattr(args, "amp_dtype", "off") == "bf16" and device.type == "cuda",
    ):
        _, decoder_output = model(merged_inputs)
    model_output = decoder_output["model_output"].float()  # [B, 1, T, 4]

    pred_x_start = model_output
    ego_diffusion_loss = torch.sum((pred_x_start[:, 0] - x0_target[:, 0]) ** 2, dim=-1)
    ego_pred_velocity = pred_x_start[:, 0]
    ego_pred_velocity_raw = inverse_normalize_ego_velocity(ego_pred_velocity, norm)
    _, ego_waypoint_loss = hybrid_loss_components(
        ego_pred_velocity,
        x0_target[:, 0],
        ego_pred_velocity_raw,
        ego_target,
        W=args.hybrid_loss_window,
    )
    ego_reconstruction = ego_diffusion_loss + args.planning_hybrid_loss * ego_waypoint_loss
    ego_loss_per_sample = ego_reconstruction[:, : args.ego_prediction_horizon].mean(dim=-1)

    return {
        "ego_loss_per_sample": ego_loss_per_sample,
        "ego_hdp_diffusion_loss": ego_diffusion_loss.mean().detach(),
        "ego_hdp_waypoint_loss": ego_waypoint_loss.mean().detach(),
    }


def compute_reward_weighted_loss(
    model,
    norm_inputs: dict[str, torch.Tensor],
    ego_pseudo_gt: torch.Tensor,
    reward: torch.Tensor,
    num_scenes: int,
    n: int,
    args,
    cached_encoding: torch.Tensor | None = None,
    reward_weights: torch.Tensor | None = None,
    valid_sample: torch.Tensor | None = None,
    global_valid_count: torch.Tensor | None = None,
    ddp_world_size: int | None = None,
) -> dict[str, torch.Tensor]:
    if reward_weights is None or valid_sample is None:
        reward_weights, valid_sample = compute_reward_weights(
            reward,
            num_scenes,
            n,
            args.rl_reward_normalize,
            getattr(args, "rl_reward_beta", 1.0),
            args.advantage_eps,
        )
    if global_valid_count is None or ddp_world_size is None:
        global_valid_count, ddp_world_size = distributed_valid_sample_count(
            valid_sample,
            bool(getattr(args, "ddp", False)),
        )
    if not bool(global_valid_count > 0):
        raise ValueError("RL loss requires at least one valid reward group across all DDP ranks")

    loss_terms = _compute_policy_ego_loss_per_sample(
        model,
        norm_inputs,
        ego_pseudo_gt,
        args,
        cached_encoding,
    )
    ego_loss_per_sample = loss_terms["ego_loss_per_sample"]
    valid_weight = valid_sample.to(ego_loss_per_sample.dtype)
    numerator = (reward_weights * valid_weight * ego_loss_per_sample).sum()
    # DDP averages gradients, hence the world-size multiplier converts the per-rank numerator
    # into a true global valid-sample mean after reducer averaging.
    loss = numerator * float(ddp_world_size) / global_valid_count.to(ego_loss_per_sample.dtype)

    return {
        "loss": loss,
        "reward_weighted_loss": loss.detach(),
        "ego_reconstruction_loss": ego_loss_per_sample.mean().detach(),
        "ego_hdp_diffusion_loss": loss_terms["ego_hdp_diffusion_loss"],
        "ego_hdp_waypoint_loss": loss_terms["ego_hdp_waypoint_loss"],
        "reward_weight_mean": reward_weights.mean().detach(),
        "reward_weight_max": reward_weights.max().detach(),
        "reward_weight_min": reward_weights.min().detach(),
        "valid_group_fraction": valid_sample.view(num_scenes, n).any(dim=1).float().mean().detach(),
    }
