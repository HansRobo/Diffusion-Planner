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
    hybrid_waypoint_loss,
    inverse_normalize_ego_velocity,
    normalize_ego_velocity,
    sample_diffusion_time,
    waypoints_to_velocity,
)
from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear
from planner_metrics.config import RewardConfig
from planner_metrics.geometry import _build_ego_bbox_corners, _point_to_segments_dist
from planner_metrics.subscores import (
    compute_ego_neighbor_signed_clearance,
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


def _cross_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _road_border_clearance_exact(
    ego_trajs: torch.Tensor,
    ego_shape: torch.Tensor,
    line_strings: torch.Tensor | None,
) -> torch.Tensor:
    """Exact minimum distance between the ego rectangle and road-border segments."""
    C, T, _ = ego_trajs.shape
    no_data = torch.full(
        (C, T),
        float("inf"),
        dtype=ego_trajs.dtype,
        device=ego_trajs.device,
    )
    if line_strings is None or line_strings.shape[-1] < 4:
        return no_data

    border_xy = line_strings[..., :2]
    # Channel 3 is the explicit road-border validity/type flag. A coordinate at
    # the ego origin is legitimate and must not be confused with zero padding.
    border_point = line_strings[..., 3] > 0.5
    valid_pair = border_point[..., :-1] & border_point[..., 1:]
    if not bool(valid_pair.any()):
        return no_data
    seg_a = border_xy[..., :-1, :][valid_pair]
    seg_b = border_xy[..., 1:, :][valid_pair]

    corners = _build_ego_bbox_corners(ego_trajs, ego_shape).reshape(C * T, 4, 2)
    rect_a = corners
    rect_b = torch.roll(corners, shifts=-1, dims=1)
    query_count = corners.shape[0]
    segment_count = seg_a.shape[0]
    max_pair_elements = 5_000_000
    chunk_size = max(1, max_pair_elements // max(segment_count * 4, 1))
    clearance_chunks = []

    for start in range(0, query_count, chunk_size):
        end = min(start + chunk_size, query_count)
        corner = corners[start:end]
        edge_a = rect_a[start:end]
        edge_b = rect_b[start:end]

        # Rectangle vertices to road-border segments: [Q, 4, E].
        border_vec = seg_b - seg_a
        border_len2 = border_vec.square().sum(dim=-1).clamp_min(1e-12)
        corner_rel = corner[:, :, None, :] - seg_a[None, None, :, :]
        corner_t = (
            (corner_rel * border_vec[None, None]).sum(dim=-1) / border_len2[None, None]
        ).clamp(0.0, 1.0)
        corner_foot = seg_a[None, None] + corner_t[..., None] * border_vec[None, None]
        corner_dist2 = (corner[:, :, None] - corner_foot).square().sum(dim=-1)

        # Border endpoints to rectangle edges: [Q, E, 2 endpoints, 4 edges].
        edge_vec = edge_b - edge_a
        edge_len2 = edge_vec.square().sum(dim=-1).clamp_min(1e-12)
        endpoints = torch.stack((seg_a, seg_b), dim=1)
        endpoint_rel = endpoints[None, :, :, None, :] - edge_a[:, None, None, :, :]
        endpoint_t = (
            (endpoint_rel * edge_vec[:, None, None]).sum(dim=-1) / edge_len2[:, None, None]
        ).clamp(0.0, 1.0)
        endpoint_foot = edge_a[:, None, None] + endpoint_t[..., None] * edge_vec[:, None, None]
        endpoint_dist2 = (endpoints[None, :, :, None] - endpoint_foot).square().sum(dim=-1)

        min_dist2 = torch.minimum(
            corner_dist2.amin(dim=(1, 2)),
            endpoint_dist2.amin(dim=(1, 2, 3)),
        )

        # Endpoint distances alone are non-zero when two segment interiors cross.
        edge_vec_exp = edge_vec[:, :, None, :]
        seg_vec_exp = border_vec[None, None, :, :]
        seg_a_rel = seg_a[None, None] - edge_a[:, :, None]
        seg_b_rel = seg_b[None, None] - edge_a[:, :, None]
        edge_a_rel = edge_a[:, :, None] - seg_a[None, None]
        edge_b_rel = edge_b[:, :, None] - seg_a[None, None]
        denominator = _cross_2d(edge_vec_exp, seg_vec_exp)
        non_parallel = denominator.abs() > 1e-8
        intersects = (
            non_parallel
            & (_cross_2d(edge_vec_exp, seg_a_rel) * _cross_2d(edge_vec_exp, seg_b_rel) <= 0)
            & (_cross_2d(seg_vec_exp, edge_a_rel) * _cross_2d(seg_vec_exp, edge_b_rel) <= 0)
        ).any(dim=(1, 2))

        # A short border segment may lie entirely inside the ego footprint.
        edge_to_endpoint = endpoints[None, :, :, None, :] - edge_a[:, None, None, :, :]
        side = _cross_2d(edge_vec[:, None, None], edge_to_endpoint)
        endpoint_inside = ((side >= -1e-7).all(dim=-1) | (side <= 1e-7).all(dim=-1)).any(dim=(1, 2))
        clearance_chunks.append(
            torch.where(
                intersects | endpoint_inside,
                torch.zeros_like(min_dist2),
                min_dist2.clamp_min(0.0).sqrt(),
            )
        )

    return torch.cat(clearance_chunks).reshape(C, T)


def _trajectory_speed(xy: torch.Tensor, dt: float, initial_xy: torch.Tensor | None = None):
    if initial_xy is None:
        first = torch.zeros_like(xy[..., :1, :])
    else:
        first = initial_xy[..., None, :]
    step = torch.diff(torch.cat([first, xy], dim=-2), dim=-2)
    return step.norm(dim=-1) / dt


def _relative_progress_score(
    ego_trajs: torch.Tensor,
    expert_future: torch.Tensor,
    *,
    min_reference_m: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return signed route progress relative to the expert endpoint, capped for reward use."""
    expert_displacement = expert_future[-1, :2]
    expert_distance = expert_displacement.norm()
    moving_reference = expert_distance >= min_reference_m
    expert_direction = expert_displacement / expert_distance.clamp_min(min_reference_m)
    candidate_progress = (ego_trajs[:, -1, :2] * expert_direction).sum(dim=-1)
    ratio = torch.where(
        moving_reference,
        candidate_progress / expert_distance.clamp_min(min_reference_m),
        torch.ones_like(candidate_progress),
    )
    return ratio, ratio.clamp(0.0, 1.0)


def _safety_gated_progress(
    progress_score: torch.Tensor, safety_score: torch.Tensor
) -> torch.Tensor:
    """Prevent the DP-native progress term from rewarding collision progress."""
    return progress_score * safety_score


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
    valid_indices = slot_valid.nonzero(as_tuple=True)[0]
    future = neighbors_future.index_select(0, valid_indices)[:, :T, :4]
    past = neighbor_past.index_select(0, valid_indices)
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


def _scene_neighbors_batch(
    neighbors_future: torch.Tensor,
    neighbor_past: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack each scene's valid neighbor slots to one batch-local fixed width."""
    B, N, T, _ = neighbors_future.shape
    pose_valid = neighbors_future[..., 2:4].norm(dim=-1) > 0.5
    slot_valid = pose_valid.any(dim=-1)
    max_neighbors = int(slot_valid.sum(dim=1).max().item()) if B > 0 else 0

    slot_index = torch.arange(N, device=neighbors_future.device).expand(B, -1)
    packed_index = torch.where(slot_valid, slot_index, N).sort(dim=1).values[:, :max_neighbors]
    packed_valid = packed_index < N
    safe_index = packed_index.clamp_max(max(N - 1, 0))

    future = torch.gather(
        neighbors_future,
        1,
        safe_index[:, :, None, None].expand(-1, -1, T, neighbors_future.shape[-1]),
    )
    past = torch.gather(
        neighbor_past,
        1,
        safe_index[:, :, None, None].expand(
            -1, -1, neighbor_past.shape[-2], neighbor_past.shape[-1]
        ),
    )
    future = future.masked_fill(~packed_valid[:, :, None, None], 0.0)
    past = past.masked_fill(~packed_valid[:, :, None, None], 0.0)
    valid = (future[..., 2:4].norm(dim=-1) > 0.5) & packed_valid[:, :, None]

    shapes = past[:, :, -1, [6, 7]].abs()
    fallback = shapes.new_tensor([2.0, 4.5])
    shapes = torch.where((shapes > 1e-3).all(dim=-1, keepdim=True), shapes, fallback)
    initial_xy = past[:, :, -1, :2]
    is_vehicle = (past[:, :, -1, 8:11].argmax(dim=-1) == 0) & packed_valid
    return future[..., :4], shapes, valid, initial_xy, is_vehicle


def _collision_and_leader_terms(
    ego_trajs: torch.Tensor,
    ego_shape: torch.Tensor,
    neighbor_futures: torch.Tensor,
    neighbor_shapes: torch.Tensor,
    neighbor_valid: torch.Tensor,
    neighbor_initial_xy: torch.Tensor,
    neighbor_is_vehicle: torch.Tensor,
    config: HDPRewardConfig,
    leader_reference: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute paper safety, TTC, THW and leader-conditioned following terms."""
    C, T, _ = ego_trajs.shape
    device = ego_trajs.device
    ego_xy = ego_trajs[..., :2]
    heading = ego_trajs[..., 2:4]
    heading = heading / heading.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    ego_speed = _trajectory_speed(ego_xy, config.dt)
    acceleration = torch.diff(torch.cat([ego_speed[:, :1], ego_speed], dim=-1), dim=-1) / config.dt
    comfort_score = _linear_safe_score(6.0 - acceleration.abs(), 0.0, 4.0)

    if neighbor_futures.shape[0] == 0:
        ones = torch.ones(C, T, device=device)
        return {
            "safety": torch.ones(C, device=device),
            "ttc": ones,
            "thw": ones,
            "follow": torch.ones(C, device=device),
            "comfort": comfort_score.mean(dim=-1),
            "leader_fraction": torch.zeros(C, device=device),
            "collision_active": torch.zeros(C, device=device),
            "collision_rear": torch.zeros(C, device=device),
            "stopped_clearance": torch.full((C, T), float("inf"), device=device),
            "stopped_is_rear": torch.zeros(C, T, dtype=torch.bool, device=device),
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
    if leader_reference is None:
        leader_candidate = (
            neighbor_valid[None]
            & neighbor_is_vehicle[None, :, None]
            & (longitudinal > 0.0)
            & (lateral <= lateral_limit)
        )
        candidate_gap = bumper_gap.clamp_min(0.0).masked_fill(~leader_candidate, float("inf"))
        leader_gap, leader_index = candidate_gap.min(dim=1)
        leader_present = torch.isfinite(leader_gap)
    else:
        reference = leader_reference[:T]
        reference_heading = reference[..., 2:4]
        reference_heading = reference_heading / reference_heading.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        reference_center = reference[..., :2] + 0.5 * ego_shape[0] * reference_heading
        reference_rel = neighbor_futures[..., :2] - reference_center[None]
        reference_longitudinal = (reference_rel * reference_heading[None]).sum(dim=-1)
        reference_lateral = torch.abs(
            reference_rel[..., 0] * -reference_heading[None, :, 1]
            + reference_rel[..., 1] * reference_heading[None, :, 0]
        )
        reference_candidate = (
            neighbor_valid
            & neighbor_is_vehicle[:, None]
            & (reference_longitudinal > 0.0)
            & (reference_lateral <= lateral_limit[0])
        )
        reference_gap = (
            reference_longitudinal - 0.5 * (ego_length + npc_length[:, None])
        ).clamp_min(0.0)
        reference_gap = reference_gap.masked_fill(~reference_candidate, float("inf"))
        reference_min_gap, reference_index = reference_gap.min(dim=0)
        reference_present = torch.isfinite(reference_min_gap)
        leader_index = reference_index[None].expand(C, -1)
        leader_present = reference_present[None].expand(C, -1)
        leader_gap = torch.gather(bumper_gap, 1, leader_index[:, None]).squeeze(1).clamp_min(0.0)
        leader_gap = torch.where(
            leader_present,
            leader_gap,
            torch.full_like(leader_gap, float("inf")),
        )

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
    stopped_clearance, stopped_index = clearance.masked_fill(
        ~stopped_mask[None, :, None], float("inf")
    ).min(dim=1)
    stopped_is_rear = torch.gather(
        npc_behind,
        1,
        stopped_index[:, None, :],
    ).squeeze(1) & torch.isfinite(stopped_clearance)
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
    aspect_sum = gap_score + spacing_score + speed_score + comfort_score
    aspect_sum = torch.where(leader_present, aspect_sum, torch.full_like(aspect_sum, 4.0))
    follow = aspect_sum.mean(dim=-1) / 4.0
    return {
        "safety": safety,
        "ttc": ttc_score,
        "thw": thw_score,
        "follow": follow,
        "comfort": comfort_score.mean(dim=-1),
        "leader_fraction": leader_present.float().mean(dim=-1),
        "collision_active": active_collision.any(dim=(1, 2)).float(),
        "collision_rear": rear_collision.any(dim=(1, 2)).float(),
        "stopped_clearance": stopped_clearance,
        "stopped_is_rear": stopped_is_rear,
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
    stopped_is_rear: torch.Tensor | None = None,
    *,
    static_available: bool | None = None,
    stopped_available_flag: bool | None = None,
    road_border_available: bool | None = None,
) -> tuple[torch.Tensor, dict[str, bool]]:
    """Fuse static boxes/stopped agents, with road border as a weak fallback."""
    C, T, _ = ego_trajs.shape
    device = ego_trajs.device
    source_clearances: list[torch.Tensor] = []
    source_rear_masks: list[torch.Tensor] = []
    sources = {"static": False, "stopped": False, "road_border": False}

    if static_objects is not None:
        valid = (static_objects[..., :2].abs().sum(dim=-1) > 1e-6) | (
            static_objects[..., 4:6].abs().sum(dim=-1) > 1e-6
        )
        has_static = bool(valid.any()) if static_available is None else static_available
        if has_static:
            objects = static_objects[valid]
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
            source_rear_masks.append(torch.zeros(C, T, dtype=torch.bool, device=device))
            sources["static"] = True

    has_stopped = (
        bool(stopped_available.item()) if stopped_available_flag is None else stopped_available_flag
    )
    if has_stopped:
        source_clearances.append(stopped_clearance)
        source_rear_masks.append(
            stopped_is_rear
            if stopped_is_rear is not None
            else torch.zeros(C, T, dtype=torch.bool, device=device)
        )
        sources["stopped"] = True

    if source_clearances:
        clearance_stack = torch.stack(source_clearances, dim=0)
        clearance, source_index = clearance_stack.min(dim=0)
        winning_rear = torch.gather(
            torch.stack(source_rear_masks, dim=0),
            0,
            source_index.unsqueeze(0),
        ).squeeze(0)
        ego_speed = _trajectory_speed(ego_trajs[..., :2], config.dt)
        safe_distance = config.occupancy_safe_m + config.occupancy_speed_gain_s * ego_speed
        critical_distance = config.occupancy_critical_m + config.occupancy_speed_gain_s * ego_speed
        score = _linear_safe_score(clearance, critical_distance, safe_distance)
        score = torch.where(
            winning_rear,
            1.0 - config.rear_end_penalty * (1.0 - score),
            score,
        )
        return score, sources

    # Road borders are not literal occupancy. Use them only when the scene has at least one
    # real border segment, and score lateral clearance without the longitudinal speed gain used
    # for obstacles ahead. The previous coupling made safe high-speed/right-turn GT trajectories
    # receive zero OCC reward merely for passing near a curb.
    has_road_border = road_border_available
    if has_road_border is None and line_strings is not None and line_strings.shape[-1] >= 4:
        border_point = line_strings[..., 3] > 0.5
        has_road_border = bool((border_point[..., :-1] & border_point[..., 1:]).any().item())
    if has_road_border:
        road_border = _road_border_clearance_exact(
            ego_trajs,
            ego_shape,
            line_strings,
        )
        if road_border_available is not None or torch.isfinite(road_border).any():
            sources["road_border"] = True
            return _linear_safe_score(
                road_border,
                0.0,
                _RL_REWARD_CONFIG.rb_wide_thresh,
            ), sources

    return torch.ones(C, T, device=device), sources


def _nearest_lane_distance(
    points: torch.Tensor,
    lanes: torch.Tensor,
) -> torch.Tensor:
    center = lanes[..., :2]
    valid = center.abs().sum(dim=-1) > 1e-6
    valid_pair = (valid[..., :-1] & valid[..., 1:]).reshape(-1)
    seg_start = center[..., :-1, :].reshape(-1, 2)
    seg_end = center[..., 1:, :].reshape(-1, 2)
    flat_points = points.reshape(-1, 2)
    max_distance_elements = 10_000_000
    chunk_size = max(1, max_distance_elements // seg_start.shape[0])
    chunks = []
    for start in range(0, flat_points.shape[0], chunk_size):
        distance = _point_to_segments_dist(
            flat_points[start : start + chunk_size],
            seg_start,
            seg_end,
        )
        chunks.append(distance.masked_fill(~valid_pair[None], float("inf")).amin(dim=-1))
    distance = torch.cat(chunks)
    return distance.reshape(points.shape[:-1])


def _nearest_lane_distance_batch(
    points: torch.Tensor,
    lanes: torch.Tensor,
) -> torch.Tensor:
    """Scene-aligned point-to-centerline distance without per-scene kernel launches."""
    if points.shape[0] != lanes.shape[0]:
        raise ValueError(
            "Batched lane distance expects matching scene dimensions, got "
            f"{points.shape[0]} and {lanes.shape[0]}"
        )
    batch_size = points.shape[0]
    center = lanes[..., :2]
    point_valid = center.abs().sum(dim=-1) > 1e-6
    segment_valid = (point_valid[..., :-1] & point_valid[..., 1:]).flatten(1)
    segment_start = center[..., :-1, :].flatten(1, 2)
    segment_end = center[..., 1:, :].flatten(1, 2)
    flat_points = points.reshape(batch_size, -1, 2)

    max_distance_elements = 10_000_000
    elements_per_query = max(batch_size * segment_start.shape[1], 1)
    chunk_size = max(1, max_distance_elements // elements_per_query)
    chunks = []
    segment = segment_end - segment_start
    segment_len2 = segment.square().sum(dim=-1).clamp_min(1e-10)
    for start in range(0, flat_points.shape[1], chunk_size):
        query = flat_points[:, start : start + chunk_size]
        relative = query[:, :, None, :] - segment_start[:, None, :, :]
        projection = ((relative * segment[:, None]).sum(dim=-1) / segment_len2[:, None]).clamp(
            0.0, 1.0
        )
        closest = segment_start[:, None] + projection[..., None] * segment[:, None]
        distance = (query[:, :, None] - closest).norm(dim=-1)
        chunks.append(distance.masked_fill(~segment_valid[:, None], float("inf")).amin(dim=-1))
    return torch.cat(chunks, dim=1).reshape(points.shape[:-1])


def _lane_reward_centerlines(
    expert_future: torch.Tensor,
    lanes: torch.Tensor,
    route_lanes: torch.Tensor | None,
    config: HDPRewardConfig,
    *,
    route_aligned: bool | None = None,
) -> tuple[torch.Tensor, bool]:
    """Prefer the navigation route when it agrees with the logged expert trajectory."""
    if route_lanes is None:
        return lanes, False
    if route_aligned is None:
        route_aligned = bool(_route_alignment(expert_future, route_lanes, config).item())
    return (route_lanes, True) if route_aligned else (lanes, False)


def _route_alignment(
    expert_future: torch.Tensor,
    route_lanes: torch.Tensor | None,
    config: HDPRewardConfig,
) -> torch.Tensor:
    if route_lanes is None:
        return torch.zeros((), dtype=torch.bool, device=expert_future.device)
    route_valid = route_lanes[..., :2].abs().sum(dim=-1) > 1e-6
    has_route = (route_valid[..., :-1] & route_valid[..., 1:]).any()
    route_distance = _nearest_lane_distance(expert_future[..., :2], route_lanes)
    return (
        has_route
        & (route_distance.mean() <= config.lane_half_width_m)
        & (route_distance.max() <= 2.0 * config.lane_half_width_m)
    )


def _hdp_lane_score(
    ego_trajs: torch.Tensor,
    expert_future: torch.Tensor,
    lanes: torch.Tensor,
    turn_indicators: torch.Tensor | None,
    config: HDPRewardConfig,
    *,
    lanes_available: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lane_valid = lanes[..., :2].abs().sum(dim=-1) > 1e-6
    has_lanes = (
        bool((lane_valid[..., :-1] & lane_valid[..., 1:]).any())
        if lanes_available is None
        else lanes_available
    )
    if not has_lanes:
        zero = torch.zeros((), dtype=torch.bool, device=ego_trajs.device)
        return torch.zeros(ego_trajs.shape[0], device=ego_trajs.device), ~zero, zero
    pred_distance = _nearest_lane_distance(ego_trajs[..., :2], lanes)
    center_score = (1.0 - pred_distance / config.lane_half_width_m).clamp(0.0, 1.0)

    expert_xy = expert_future[..., :2]
    expert_distance = _nearest_lane_distance(expert_xy, lanes)
    expert_off_lane = (expert_distance.mean() > config.lane_half_width_m) | (
        expert_distance.max() > 2.0 * config.lane_half_width_m
    )

    indicator_active = torch.zeros((), dtype=torch.bool, device=ego_trajs.device)
    if turn_indicators is not None and turn_indicators.numel() > 0:
        indicator_active = (turn_indicators[-1] == 2) | (turn_indicators[-1] == 3)
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
    expert_lane_change = centerline_change | (indicator_active & indicated_departure)
    masked = expert_off_lane | expert_lane_change
    lane_score = torch.where(masked, torch.zeros_like(center_score), center_score).mean(dim=-1)
    return lane_score, expert_off_lane, expert_lane_change


def _batched_lane_and_progress(
    ego_group: torch.Tensor,
    expert_future: torch.Tensor,
    lanes: torch.Tensor,
    route_lanes: torch.Tensor | None,
    turn_indicators: torch.Tensor | None,
    config: HDPRewardConfig,
) -> tuple[torch.Tensor, ...]:
    """Compute route selection, lane masks, and progress for a scene batch."""
    lane_points = lanes[..., :2]
    lane_point_valid = lane_points.abs().sum(dim=-1) > 1e-6
    lane_available = (lane_point_valid[..., :-1] & lane_point_valid[..., 1:]).flatten(1).any(1)
    lane_candidate_distance = _nearest_lane_distance_batch(ego_group[..., :2], lanes)
    lane_expert_distance = _nearest_lane_distance_batch(expert_future[..., :2], lanes)

    if route_lanes is None:
        route_aligned = torch.zeros_like(lane_available)
        candidate_distance = lane_candidate_distance
        expert_distance = lane_expert_distance
    else:
        route_points = route_lanes[..., :2]
        route_point_valid = route_points.abs().sum(dim=-1) > 1e-6
        route_available = (
            (route_point_valid[..., :-1] & route_point_valid[..., 1:]).flatten(1).any(1)
        )
        route_candidate_distance = _nearest_lane_distance_batch(ego_group[..., :2], route_lanes)
        route_expert_distance = _nearest_lane_distance_batch(expert_future[..., :2], route_lanes)
        route_aligned = (
            route_available
            & (route_expert_distance.mean(dim=-1) <= config.lane_half_width_m)
            & (route_expert_distance.amax(dim=-1) <= 2.0 * config.lane_half_width_m)
        )
        candidate_distance = torch.where(
            route_aligned[:, None, None], route_candidate_distance, lane_candidate_distance
        )
        expert_distance = torch.where(
            route_aligned[:, None], route_expert_distance, lane_expert_distance
        )

    centerline_available = lane_available | route_aligned
    expert_off_lane = (expert_distance.mean(dim=-1) > config.lane_half_width_m) | (
        expert_distance.amax(dim=-1) > 2.0 * config.lane_half_width_m
    )
    indicator_active = torch.zeros_like(centerline_available)
    if turn_indicators is not None and turn_indicators.numel() > 0:
        indicator_active = (turn_indicators[:, -1] == 2) | (turn_indicators[:, -1] == 3)
    edge_window = min(3, expert_distance.shape[-1])
    start_distance = expert_distance[:, :edge_window].mean(dim=-1)
    end_distance = expert_distance[:, -edge_window:].mean(dim=-1)
    peak_distance = expert_distance.amax(dim=-1)
    starts_on_lane = start_distance < 0.5 * config.lane_half_width_m
    centerline_change = (
        starts_on_lane
        & (end_distance < 0.5 * config.lane_half_width_m)
        & (peak_distance > 0.75 * config.lane_half_width_m)
    )
    indicated_departure = starts_on_lane & (
        peak_distance - start_distance > 0.5 * config.lane_half_width_m
    )
    expert_lane_change = centerline_change | (indicator_active & indicated_departure)
    expert_off_lane = torch.where(
        centerline_available, expert_off_lane, torch.ones_like(expert_off_lane)
    )
    expert_lane_change = centerline_available & expert_lane_change
    lane_masked = expert_off_lane | expert_lane_change
    center_score = (1.0 - candidate_distance / config.lane_half_width_m).clamp(0.0, 1.0)
    lane_score = torch.where(
        lane_masked[:, None, None], torch.zeros_like(center_score), center_score
    ).mean(dim=-1)

    expert_displacement = expert_future[:, -1, :2]
    expert_distance_m = expert_displacement.norm(dim=-1)
    moving_reference = expert_distance_m >= 1.0
    expert_direction = expert_displacement / expert_distance_m.clamp_min(1.0)[:, None]
    candidate_progress = (ego_group[:, :, -1, :2] * expert_direction[:, None]).sum(dim=-1)
    progress_ratio = torch.where(
        moving_reference[:, None],
        candidate_progress / expert_distance_m.clamp_min(1.0)[:, None],
        torch.ones_like(candidate_progress),
    )
    return (
        lane_score,
        expert_off_lane,
        expert_lane_change,
        route_aligned,
        progress_ratio,
        progress_ratio.clamp(0.0, 1.0),
        lane_available,
    )


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
            "lane_route_source",
            "occupancy_any_source",
            "occupancy_static_source",
            "occupancy_stopped_source",
            "occupancy_road_border_source",
            "risk_zero_fraction",
            "risk_one_fraction",
            "ttc_zero_fraction",
            "thw_zero_fraction",
            "occupancy_zero_fraction",
            "progress",
            "progress_raw",
            "progress_ratio",
            "underprogress_fraction",
            "overprogress_fraction",
            "comfort",
        )
    }
    rewards = []
    required = ("ego_shape", "neighbor_agents_past", "ego_agent_future", "lanes")
    missing = [key for key in required if key not in scene_inputs]
    if missing:
        raise ValueError(f"HDP reward is missing required scene tensors: {missing}")

    expert_futures_tensor = heading_to_cos_sin_if_needed(scene_inputs["ego_agent_future"])
    packed_neighbors = _scene_neighbors_batch(
        neighbor_group,
        scene_inputs["neighbor_agents_past"],
    )

    def collision_terms_for_scene(
        ego_trajectories,
        ego_shape,
        neighbor_futures,
        neighbor_shapes,
        neighbor_valid,
        neighbor_initial,
        neighbor_is_vehicle,
        expert_future,
    ):
        return _collision_and_leader_terms(
            ego_trajectories,
            ego_shape,
            neighbor_futures,
            neighbor_shapes,
            neighbor_valid,
            neighbor_initial,
            neighbor_is_vehicle,
            config,
            leader_reference=expert_future,
        )

    scene_term_tensors = torch.vmap(collision_terms_for_scene)(
        ego_group,
        scene_inputs["ego_shape"][:, :3],
        *packed_neighbors,
        expert_futures_tensor,
    )
    scene_terms = [
        {key: value[scene] for key, value in scene_term_tensors.items()}
        for scene in range(num_scenes)
    ]
    route_lanes_batch = scene_inputs.get("route_lanes")
    (
        lane_scores,
        expert_off_lanes,
        expert_lane_changes,
        route_aligned,
        progress_ratios,
        progress_scores,
        _lane_available,
    ) = _batched_lane_and_progress(
        ego_group,
        expert_futures_tensor,
        scene_inputs["lanes"],
        route_lanes_batch,
        scene_inputs.get("turn_indicators"),
        config,
    )

    device = ego_world.device
    static_objects_batch = scene_inputs.get("static_objects")
    if static_objects_batch is None:
        static_available = torch.zeros(num_scenes, dtype=torch.bool, device=device)
    else:
        static_available = (
            (static_objects_batch[..., :2].abs().sum(dim=-1) > 1e-6)
            | (static_objects_batch[..., 4:6].abs().sum(dim=-1) > 1e-6)
        ).any(dim=-1)
    line_strings_batch = scene_inputs.get("line_strings")
    if line_strings_batch is None or line_strings_batch.shape[-1] < 4:
        road_border_available = torch.zeros(num_scenes, dtype=torch.bool, device=device)
    else:
        border_point = line_strings_batch[..., 3] > 0.5
        road_border_available = (border_point[..., :-1] & border_point[..., 1:]).flatten(1).any(1)
    stopped_available = torch.stack([terms["stopped_available"] for terms in scene_terms])
    availability = (
        torch.stack(
            (
                static_available,
                stopped_available,
                road_border_available,
                route_aligned,
            ),
            dim=1,
        )
        .cpu()
        .tolist()
    )

    for scene in range(num_scenes):
        ego_shape = scene_inputs["ego_shape"][scene, :3]
        terms = scene_terms[scene]
        has_static, has_stopped, has_road_border, use_route = availability[scene]
        occupancy, occupancy_sources = _occupancy_score(
            ego_group[scene],
            ego_shape,
            static_objects_batch[scene] if static_objects_batch is not None else None,
            line_strings_batch[scene] if line_strings_batch is not None else None,
            terms["stopped_clearance"],
            terms["stopped_available"],
            config,
            stopped_is_rear=terms["stopped_is_rear"],
            static_available=has_static,
            stopped_available_flag=has_stopped,
            road_border_available=has_road_border,
        )
        risk = torch.stack([terms["ttc"], terms["thw"], occupancy], dim=0).amin(dim=(0, 2))
        progress_ratio = progress_ratios[scene]
        progress_score = progress_scores[scene]
        progress_reward = _safety_gated_progress(progress_score, terms["safety"])
        lane = lane_scores[scene]
        expert_off_lane = expert_off_lanes[scene]
        expert_lane_change = expert_lane_changes[scene]
        reward = (
            getattr(args, "rl_reward_w_safety", 0.0) * terms["safety"]
            + args.rl_reward_w_risk * risk
            + args.rl_reward_w_follow * terms["follow"]
            + args.rl_reward_w_lane * lane
            + getattr(args, "rl_reward_w_progress", 0.0) * progress_reward
        )
        rewards.append(reward)
        metric_lists["safety"].append(terms["safety"])
        ttc_min = terms["ttc"].amin(dim=-1)
        thw_min = terms["thw"].amin(dim=-1)
        occupancy_min = occupancy.amin(dim=-1)
        metric_lists["risk"].append(risk)
        metric_lists["follow"].append(terms["follow"])
        metric_lists["comfort"].append(terms["comfort"])
        metric_lists["lane"].append(lane)
        metric_lists["progress"].append(progress_reward)
        metric_lists["progress_raw"].append(progress_score)
        metric_lists["progress_ratio"].append(progress_ratio.clamp_max(2.0))
        metric_lists["underprogress_fraction"].append((progress_ratio < 0.8).to(risk.dtype))
        metric_lists["overprogress_fraction"].append((progress_ratio > 1.2).to(risk.dtype))
        metric_lists["ttc"].append(ttc_min)
        metric_lists["thw"].append(thw_min)
        metric_lists["occupancy"].append(occupancy_min)
        metric_lists["risk_zero_fraction"].append((risk <= 1e-6).to(risk.dtype))
        metric_lists["risk_one_fraction"].append((risk >= 1.0 - 1e-6).to(risk.dtype))
        metric_lists["ttc_zero_fraction"].append((ttc_min <= 1e-6).to(risk.dtype))
        metric_lists["thw_zero_fraction"].append((thw_min <= 1e-6).to(risk.dtype))
        metric_lists["occupancy_zero_fraction"].append((occupancy_min <= 1e-6).to(risk.dtype))
        metric_lists["leader_fraction"].append(terms["leader_fraction"])
        metric_lists["collision_active"].append(terms["collision_active"])
        metric_lists["collision_rear"].append(terms["collision_rear"])
        metric_lists["lane_masked"].append(expert_off_lane.to(lane.dtype).expand(n))
        metric_lists["lane_change_masked"].append(expert_lane_change.to(lane.dtype).expand(n))
        metric_lists["lane_route_source"].append(lane.new_full((n,), float(use_route)))
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
    fraction_keys = {
        "risk_zero_fraction",
        "risk_one_fraction",
        "ttc_zero_fraction",
        "thw_zero_fraction",
        "occupancy_zero_fraction",
        "underprogress_fraction",
        "overprogress_fraction",
    }
    metrics = {
        (f"reward_{key}" if key in fraction_keys else f"reward_{key}_score"): value.mean()
        for key, value in flattened.items()
    }
    reward_group = torch.stack(rewards)
    component_groups = {
        key: torch.stack(metric_lists[key])
        for key in ("safety", "risk", "follow", "lane", "progress", "leader_fraction")
    }
    reward_centered = reward_group - reward_group.mean(dim=1, keepdim=True)
    reward_winner = reward_group.argmax(dim=1, keepdim=True)
    for key in ("safety", "risk", "follow", "lane", "progress"):
        component = component_groups[key]
        component_centered = component - component.mean(dim=1, keepdim=True)
        component_denominator = (
            reward_centered.square().sum(dim=1).sqrt()
            * component_centered.square().sum(dim=1).sqrt()
        )
        component_correlation = (reward_centered * component_centered).sum(
            dim=1
        ) / component_denominator.clamp_min(1e-6)
        component_correlation = torch.where(
            component_denominator > 1e-6,
            component_correlation,
            torch.zeros_like(component_correlation),
        )
        metrics[f"reward_{key}_correlation"] = component_correlation.mean()
    for key in ("safety", "risk", "follow", "lane", "progress"):
        component = component_groups[key]
        winner_value = component.gather(1, reward_winner).squeeze(1)
        metrics[f"reward_winner_{key}_advantage"] = (winner_value - component.mean(dim=1)).mean()
        metrics[f"reward_{key}_group_std"] = (
            component.std(dim=1).mean() if n > 1 else component.new_zeros(())
        )
        metrics[f"reward_{key}_group_range"] = (
            component.max(dim=1).values - component.min(dim=1).values
        ).mean()
    leader_group = component_groups["leader_fraction"]
    metrics["reward_leader_group_range"] = (
        leader_group.max(dim=1).values - leader_group.min(dim=1).values
    ).mean()
    return torch.cat(rewards), metrics


def heading_to_cos_sin_if_needed(trajectory: torch.Tensor) -> torch.Tensor:
    if trajectory.shape[-1] >= 4:
        return trajectory[..., :4]
    if trajectory.shape[-1] != 3:
        raise ValueError(f"Expected trajectory last dimension 3 or >=4, got {trajectory.shape}")
    return torch.cat(
        [trajectory[..., :2], trajectory[..., 2:3].cos(), trajectory[..., 2:3].sin()], dim=-1
    )


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
    generator: torch.Generator | None = None,
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
        sampled = torch.randn(
            B,
            action_agent_num,
            OUTPUT_T,
            POSE_DIM,
            device=device,
            generator=generator,
        ) * float(noise_scale)
    inference_inputs["sampled_trajectories"] = sampled
    # RL consumes only trajectories. Avoid pooling the expanded scene encoding and running
    # the frozen auxiliary classifier for every rollout candidate.
    inference_inputs["_skip_turn_indicator"] = True

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
                inference_inputs["_cached_global_route_condition"] = cached_global_route_condition
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
        "sampled_trajectories": xT,
        "diffusion_time": t,
        # RL optimizes only the trajectory policy. Avoid an unused turn-head forward and keep
        # that SFT classifier frozen while the DiT policy changes under reward weighting.
        "_skip_turn_indicator": True,
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
    ego_waypoint_loss = hybrid_waypoint_loss(
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
    expert_norm_inputs: dict[str, torch.Tensor] | None = None,
    expert_ego_gt: torch.Tensor | None = None,
    expert_cached_encoding: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if reward_weights is None or valid_sample is None:
        reward_weights, valid_sample = compute_reward_weights(
            reward,
            num_scenes,
            n,
            args.rl_reward_normalize,
            getattr(args, "rl_reward_beta", 0.5),
            args.advantage_eps,
        )
    if global_valid_count is None or ddp_world_size is None:
        global_valid_count, ddp_world_size = distributed_valid_sample_count(
            valid_sample,
            bool(getattr(args, "ddp", False)),
        )
    has_valid_group = bool(global_valid_count > 0)
    local_valid_count = valid_sample.sum().to(reward_weights.dtype)
    reward_weight_mean = reward_weights.sum() / local_valid_count.clamp_min(1.0)
    reward_weight_max = torch.nan_to_num(
        reward_weights.masked_fill(~valid_sample, -torch.inf).max(), neginf=0.0
    )
    reward_weight_min = torch.nan_to_num(
        reward_weights.masked_fill(~valid_sample, torch.inf).min(), posinf=0.0
    )
    bc_weight = float(getattr(args, "rl_bc_weight", 0.0))
    zero = reward.new_zeros(())
    # With a BC anchor, keep the candidate forward in every step so DDP static_graph sees the
    # same graph even in the rare case where every reward group is constant.
    if has_valid_group or bc_weight > 0.0:
        loss_terms = _compute_policy_ego_loss_per_sample(
            model,
            norm_inputs,
            ego_pseudo_gt,
            args,
            cached_encoding,
        )
        ego_loss_per_sample = loss_terms["ego_loss_per_sample"]
        if has_valid_group:
            valid_weight = valid_sample.to(ego_loss_per_sample.dtype)
            numerator = (reward_weights * valid_weight * ego_loss_per_sample).sum()
            # DDP averages gradients, hence the world-size multiplier converts the per-rank
            # numerator into a true global valid-sample mean after reducer averaging.
            rl_loss = (
                numerator * float(ddp_world_size) / global_valid_count.to(ego_loss_per_sample.dtype)
            )
        else:
            rl_loss = ego_loss_per_sample.sum() * 0.0
        ego_reconstruction_loss = ego_loss_per_sample.mean().detach()
    else:
        loss_terms = {
            "ego_hdp_diffusion_loss": zero,
            "ego_hdp_waypoint_loss": zero,
        }
        rl_loss = zero
        ego_reconstruction_loss = zero

    bc_loss = zero
    if bc_weight > 0.0:
        if expert_norm_inputs is None or expert_ego_gt is None:
            raise ValueError("rl_bc_weight > 0 requires expert observations and trajectories")
        expert_terms = _compute_policy_ego_loss_per_sample(
            model,
            expert_norm_inputs,
            expert_ego_gt,
            args,
            expert_cached_encoding,
        )
        expert_per_sample = expert_terms["ego_loss_per_sample"]
        expert_count = expert_per_sample.new_tensor(float(expert_per_sample.numel()))
        if (
            bool(getattr(args, "ddp", False))
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            torch.distributed.all_reduce(expert_count, op=torch.distributed.ReduceOp.SUM)
        bc_loss = expert_per_sample.sum() * float(ddp_world_size) / expert_count

    loss = rl_loss + bc_weight * bc_loss

    return {
        "loss": loss,
        "rl_loss": rl_loss.detach(),
        "reward_weighted_loss": rl_loss.detach(),
        "bc_loss": bc_loss.detach(),
        "ego_reconstruction_loss": ego_reconstruction_loss,
        "ego_hdp_diffusion_loss": loss_terms["ego_hdp_diffusion_loss"],
        "ego_hdp_waypoint_loss": loss_terms["ego_hdp_waypoint_loss"],
        "reward_weight_mean": reward_weight_mean.detach(),
        "reward_weight_max": reward_weight_max.detach(),
        "reward_weight_min": reward_weight_min.detach(),
        "valid_group_fraction": valid_sample.view(num_scenes, n).any(dim=1).float().mean().detach(),
    }
