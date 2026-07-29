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

import math
from dataclasses import dataclass

import torch

from diffusion_planner.dimensions import OUTPUT_T, POSE_DIM, TRAFFIC_LIGHT_RED
from diffusion_planner.loss import (
    hybrid_waypoint_loss,
    inverse_normalize_ego_velocity,
    normalize_ego_velocity,
    waypoints_to_velocity,
)
from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear
from diffusion_planner.utils.masks import lane_point_padding_mask
from planner_metrics.geometry import _build_ego_bbox_corners, _point_to_segments_dist
from planner_metrics.subscores import (
    compute_ego_neighbor_signed_clearance,
)

_RL_GEOMETRY_PAIR_STEP_BUDGET = 12_000_000
_STOP_LINE_ROUTE_COSINE_MAX = 0.5


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
    stopped_neighbor_vel_thresh: float = 0.1
    stopped_neighbor_disp_thresh: float = 0.5
    lane_half_width_m: float = 1.75
    leader_lateral_margin_m: float = 0.75
    stationary_reference_threshold_m: float = 1.0
    stationary_progress_tolerance_m: float = 2.0
    stationary_progress_mode: str = "distance"
    red_light_lane_tolerance_m: float = 2.0
    rear_end_penalty: float = 0.3
    road_border_critical_m: float = 0.20
    road_border_safe_m: float = 0.60

    def __post_init__(self) -> None:
        finite_fields = (
            "dt",
            "ttc_critical_s",
            "ttc_safe_s",
            "thw_critical_s",
            "thw_safe_s",
            "occupancy_critical_m",
            "occupancy_safe_m",
            "occupancy_speed_gain_s",
            "stopped_neighbor_vel_thresh",
            "stopped_neighbor_disp_thresh",
            "lane_half_width_m",
            "leader_lateral_margin_m",
            "stationary_reference_threshold_m",
            "stationary_progress_tolerance_m",
            "red_light_lane_tolerance_m",
            "rear_end_penalty",
            "road_border_critical_m",
            "road_border_safe_m",
        )
        non_finite = [
            name for name in finite_fields if not math.isfinite(float(getattr(self, name)))
        ]
        if non_finite:
            raise ValueError(f"HDP reward fields must be finite: {non_finite}")
        if self.dt <= 0.0:
            raise ValueError("dt must be > 0")
        if self.ttc_critical_s < 0.0:
            raise ValueError("ttc_critical_s must be >= 0")
        if self.ttc_safe_s <= self.ttc_critical_s:
            raise ValueError("ttc_safe_s must exceed ttc_critical_s")
        if self.thw_critical_s < 0.0:
            raise ValueError("thw_critical_s must be >= 0")
        if self.thw_safe_s <= self.thw_critical_s:
            raise ValueError("thw_safe_s must exceed thw_critical_s")
        if self.occupancy_critical_m < 0.0:
            raise ValueError("occupancy_critical_m must be >= 0")
        if self.occupancy_safe_m <= self.occupancy_critical_m:
            raise ValueError("occupancy_safe_m must exceed occupancy_critical_m")
        if self.occupancy_speed_gain_s < 0.0:
            raise ValueError("occupancy_speed_gain_s must be >= 0")
        if self.lane_half_width_m <= 0.0:
            raise ValueError("lane_half_width_m must be > 0")
        if self.leader_lateral_margin_m < 0.0:
            raise ValueError("leader_lateral_margin_m must be >= 0")
        if self.stationary_reference_threshold_m <= 0.0:
            raise ValueError("stationary_reference_threshold_m must be > 0")
        if self.stationary_progress_tolerance_m <= 0.0:
            raise ValueError("stationary_progress_tolerance_m must be > 0")
        if self.stationary_progress_mode not in {"constant", "distance"}:
            raise ValueError(
                f"Unsupported stationary_progress_mode={self.stationary_progress_mode!r}"
            )
        if self.red_light_lane_tolerance_m <= 0.0:
            raise ValueError("red_light_lane_tolerance_m must be > 0")
        if self.stopped_neighbor_vel_thresh < 0.0:
            raise ValueError("stopped_neighbor_vel_thresh must be >= 0")
        if self.stopped_neighbor_disp_thresh < 0.0:
            raise ValueError("stopped_neighbor_disp_thresh must be >= 0")
        if not 0.0 <= self.rear_end_penalty <= 1.0:
            raise ValueError("rear_end_penalty must be in [0, 1]")
        if self.road_border_critical_m < 0.0:
            raise ValueError("road_border_critical_m must be >= 0")
        if self.road_border_safe_m <= self.road_border_critical_m:
            raise ValueError("road_border_safe_m must exceed road_border_critical_m")


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
        stopped_neighbor_vel_thresh=float(getattr(args, "rl_stopped_neighbor_vel_thresh", 0.1)),
        stopped_neighbor_disp_thresh=float(getattr(args, "rl_stopped_neighbor_disp_thresh", 0.5)),
        lane_half_width_m=float(getattr(args, "rl_lane_half_width_m", 1.75)),
        leader_lateral_margin_m=float(getattr(args, "rl_leader_lateral_margin_m", 0.75)),
        stationary_reference_threshold_m=float(
            getattr(args, "rl_stationary_reference_threshold_m", 1.0)
        ),
        stationary_progress_tolerance_m=float(
            getattr(args, "rl_stationary_progress_tolerance_m", 2.0)
        ),
        stationary_progress_mode=str(getattr(args, "rl_stationary_progress_mode", "distance")),
        red_light_lane_tolerance_m=float(getattr(args, "rl_red_light_lane_tolerance_m", 2.0)),
        road_border_critical_m=float(getattr(args, "rl_road_border_critical_m", 0.20)),
        road_border_safe_m=float(getattr(args, "rl_road_border_safe_m", 0.60)),
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
    *,
    valid_segments_known: bool = False,
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
    if not valid_segments_known and not bool(valid_pair.any()):
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


def _batched_road_border_clearance(
    ego_group: torch.Tensor,
    ego_shapes: torch.Tensor,
    line_strings: torch.Tensor | None,
    road_border_available: torch.Tensor,
) -> torch.Tensor:
    """Compute exact border clearance only for scenes that contain valid border segments.

    Border geometry has scene-dependent segment counts, so it cannot use the common fully
    batched path. The returned ``[B, C, T]`` tensor is ``+inf`` for scenes without a valid
    border, making those scenes neutral when converted to a score.
    """
    batch_size, candidates, time_steps = ego_group.shape[:3]
    clearance = torch.full(
        (batch_size, candidates, time_steps),
        float("inf"),
        dtype=ego_group.dtype,
        device=ego_group.device,
    )
    if line_strings is None or line_strings.shape[-1] < 4 or not bool(road_border_available.any()):
        return clearance
    rows = road_border_available.nonzero(as_tuple=False).flatten().cpu().tolist()
    for scene in rows:
        clearance[scene] = _road_border_clearance_exact(
            ego_group[scene],
            ego_shapes[scene],
            line_strings[scene],
            valid_segments_known=True,
        )
    return clearance


def _trajectory_speed(xy: torch.Tensor, dt: float, initial_xy: torch.Tensor | None = None):
    first = torch.zeros_like(xy[..., :1, :]) if initial_xy is None else initial_xy[..., None, :]
    step = torch.diff(torch.cat([first, xy], dim=-2), dim=-2)
    return step.norm(dim=-1) / dt


def _relative_progress_score(
    ego_trajs: torch.Tensor,
    expert_future: torch.Tensor,
    *,
    min_reference_m: float = 1.0,
    stationary_tolerance_m: float = 2.0,
    stationary_mode: str = "distance",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return signed route progress and a stopped-reference-aware reward score."""
    if min_reference_m <= 0.0:
        raise ValueError("min_reference_m must be > 0")
    if stationary_tolerance_m <= 0.0:
        raise ValueError("stationary_tolerance_m must be > 0")
    if stationary_mode not in {"constant", "distance"}:
        raise ValueError(f"Unsupported stationary_progress_mode={stationary_mode!r}")
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
    if stationary_mode == "constant":
        stationary_score = torch.ones_like(candidate_progress)
    elif stationary_mode == "distance":
        endpoint_error = (ego_trajs[:, -1, :2] - expert_displacement).norm(dim=-1)
        stationary_score = (1.0 - endpoint_error / stationary_tolerance_m).clamp(0.0, 1.0)
    score = torch.where(moving_reference, ratio.clamp(0.0, 1.0), stationary_score)
    return ratio, score


def _path_crosses_segments(
    path_xy: torch.Tensor,
    segment_start: torch.Tensor,
    segment_end: torch.Tensor,
) -> torch.Tensor:
    """Return whether each continuous path crosses any finite line segment."""
    candidate_count = path_xy.shape[0]
    path_segment_count = max(path_xy.shape[1] - 1, 0)
    if path_segment_count == 0 or segment_start.shape[0] == 0:
        return torch.zeros(candidate_count, dtype=torch.bool, device=path_xy.device)

    path_start = path_xy[:, :-1]
    path_vector = path_xy[:, 1:] - path_start
    crossed = torch.zeros(candidate_count, dtype=torch.bool, device=path_xy.device)
    max_pair_elements = 5_000_000
    segment_chunk = max(
        1,
        max_pair_elements // max(candidate_count * path_segment_count, 1),
    )
    for start in range(0, segment_start.shape[0], segment_chunk):
        stop_a = segment_start[start : start + segment_chunk]
        stop_vector = segment_end[start : start + segment_chunk] - stop_a
        path_vector_expanded = path_vector[:, :, None]
        stop_vector_expanded = stop_vector[None, None]
        denominator = _cross_2d(path_vector_expanded, stop_vector_expanded)
        offset = stop_a[None, None] - path_start[:, :, None]
        non_parallel = denominator.abs() > 1e-8
        safe_denominator = torch.where(non_parallel, denominator, torch.ones_like(denominator))
        path_fraction = _cross_2d(offset, stop_vector_expanded) / safe_denominator
        stop_fraction = _cross_2d(offset, path_vector_expanded) / safe_denominator
        # A touch at the current front-bumper pose is not a new violation. A crossing anywhere
        # after that pose, including between sampled waypoints, is detected.
        intersects = (
            non_parallel
            & (path_fraction > 1e-4)
            & (path_fraction <= 1.0 + 1e-6)
            & (stop_fraction >= -1e-6)
            & (stop_fraction <= 1.0 + 1e-6)
        )
        crossed |= intersects.any(dim=(1, 2))
    return crossed


def _front_bumper_path(trajectories: torch.Tensor, ego_shape: torch.Tensor) -> torch.Tensor:
    """Convert rear-axle trajectories to a path beginning at the current front bumper."""
    heading = trajectories[..., 2:4]
    heading = heading / heading.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    front_offset = 0.5 * (ego_shape[0].abs() + ego_shape[1].abs())
    future_front = trajectories[..., :2] + front_offset * heading
    current_front = trajectories.new_zeros((trajectories.shape[0], 1, 2))
    current_front[..., 0] = front_offset
    return torch.cat((current_front, future_front), dim=1)


def _red_light_constraint_score(
    ego_group: torch.Tensor,
    expert_future: torch.Tensor,
    ego_shapes: torch.Tensor,
    route_lanes: torch.Tensor | None,
    line_strings: torch.Tensor | None,
    config: HDPRewardConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Penalize candidate stop-line crossings that the logged expert does not make."""
    batch_size, candidate_count = ego_group.shape[:2]
    scores = ego_group.new_ones((batch_size, candidate_count))
    active = torch.zeros(batch_size, dtype=torch.bool, device=ego_group.device)
    expert_crossing = torch.zeros_like(active)
    if (
        route_lanes is None
        or route_lanes.shape[-1] <= TRAFFIC_LIGHT_RED
        or line_strings is None
        or line_strings.shape[-1] < 4
    ):
        return scores, active, expert_crossing

    route_valid = ~lane_point_padding_mask(route_lanes)
    red_lane = ((route_lanes[..., TRAFFIC_LIGHT_RED] > 0.5) & route_valid).any(dim=-1)
    red_scenes = red_lane.any(dim=-1).nonzero(as_tuple=True)[0].cpu().tolist()
    for scene in red_scenes:
        red_point_mask = red_lane[scene, :, None] & route_valid[scene]
        red_points = route_lanes[scene, ..., :2][red_point_mask]
        red_directions = route_lanes[scene, ..., 2:4][red_point_mask]
        stop_point = line_strings[scene, ..., 2] > 0.5
        stop_line = stop_point.any(dim=-1)
        stop_point = stop_point[stop_line]
        stop_xy = line_strings[scene, stop_line, :, :2]
        if stop_xy.shape[0] == 0:
            continue
        stop_pair = stop_point[..., :-1] & stop_point[..., 1:]

        point_distance = torch.cdist(stop_xy.flatten(0, 1), red_points)
        line_distance = point_distance.reshape(
            stop_xy.shape[0], stop_xy.shape[1], red_points.shape[0]
        ).masked_fill(~stop_point[..., None], float("inf"))
        line_associated = line_distance.amin(dim=(1, 2)) <= config.red_light_lane_tolerance_m
        associated_pair = stop_pair & line_associated[:, None]
        segment_start = stop_xy[..., :-1, :][associated_pair]
        segment_end = stop_xy[..., 1:, :][associated_pair]
        if segment_start.shape[0] > 0:
            midpoint = 0.5 * (segment_start + segment_end)
            nearest_red_point = torch.cdist(midpoint, red_points).argmin(dim=-1)
            route_direction = red_directions[nearest_red_point]
            route_direction_norm = route_direction.norm(dim=-1, keepdim=True)
            route_direction = route_direction / route_direction_norm.clamp_min(1e-6)
            stop_direction = segment_end - segment_start
            stop_direction = stop_direction / stop_direction.norm(dim=-1, keepdim=True).clamp_min(
                1e-6
            )
            direction_compatible = (route_direction_norm.squeeze(-1) <= 1e-6) | (
                (stop_direction * route_direction).sum(dim=-1).abs() <= _STOP_LINE_ROUTE_COSINE_MAX
            )
            segment_start = segment_start[direction_compatible]
            segment_end = segment_end[direction_compatible]

        expert_path = _front_bumper_path(expert_future[scene : scene + 1], ego_shapes[scene])
        scene_expert_crossing = _path_crosses_segments(expert_path, segment_start, segment_end)[0]
        candidate_path = _front_bumper_path(ego_group[scene], ego_shapes[scene])
        candidate_crossing = _path_crosses_segments(candidate_path, segment_start, segment_end)
        scene_active = candidate_crossing.new_tensor(segment_start.shape[0] > 0) & (
            ~scene_expert_crossing
        )
        active[scene] = scene_active
        expert_crossing[scene] = scene_expert_crossing
        scores[scene] = torch.where(
            scene_active,
            (~candidate_crossing).to(scores.dtype),
            torch.ones_like(scores[scene]),
        )
    return scores, active, expert_crossing


def _safety_gated_behavior(
    behavior_score: torch.Tensor, safety_score: torch.Tensor
) -> torch.Tensor:
    """Prevent non-risk behavior rewards from preferring a colliding trajectory."""
    return behavior_score * safety_score


def _apply_behavior_gate(
    behavior_score: torch.Tensor,
    safety_score: torch.Tensor,
    risk_score: torch.Tensor,
    gate: str,
) -> torch.Tensor:
    """Apply the selected real-vehicle reward constraint to behavior terms."""
    if gate == "none":
        return behavior_score
    if gate == "safety":
        return _safety_gated_behavior(behavior_score, safety_score)
    if gate == "risk":
        return behavior_score * risk_score
    raise ValueError(f"Unsupported rl_behavior_gate={gate!r}")


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
    ego_initial_speed: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute paper safety, TTC, THW and leader-conditioned following terms."""
    C, T, _ = ego_trajs.shape
    device = ego_trajs.device
    ego_xy = ego_trajs[..., :2]
    heading = ego_trajs[..., 2:4]
    heading = heading / heading.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    ego_speed = _trajectory_speed(ego_xy, config.dt)
    if ego_initial_speed is None:
        initial_speed = ego_speed[:, :1]
    else:
        initial_speed = ego_initial_speed.reshape(1, 1).expand(C, 1)
    acceleration = torch.diff(torch.cat([initial_speed, ego_speed], dim=-1), dim=-1) / config.dt
    comfort_score = _linear_safe_score(6.0 - acceleration.abs(), 0.0, 4.0)

    if neighbor_futures.shape[0] == 0:
        ones = torch.ones(C, T, device=device)
        return {
            "safety": torch.ones(C, device=device),
            "ttc": ones,
            "thw": ones,
            "follow": torch.ones(C, device=device),
            "comfort": comfort_score.mean(dim=-1),
            "ego_speed": ego_speed,
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
    # Lock the collision direction at first contact for each contiguous event. In a
    # non-reactive replay, a following vehicle can pass through the ego while still
    # overlapping; reclassifying that event halfway through would erase the paper's
    # rear-end attenuation. A later collision after separation starts a new event.
    previous_collision = torch.cat(
        [torch.zeros_like(collision[..., :1]), collision[..., :-1]], dim=-1
    )
    event_start = collision & ~previous_collision
    step_index = torch.arange(T, device=device).view(1, 1, T)
    event_start_index = torch.where(event_start, step_index, -1).cummax(dim=-1).values
    event_start_index = event_start_index.clamp_min(0)
    rear_collision_event = collision & torch.gather(npc_behind, -1, event_start_index)
    rear_collision = collision & rear_collision_event
    active_collision = collision & ~rear_collision_event
    npc_rear_for_risk = npc_behind | rear_collision_event

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
    ttc_is_rear = torch.gather(npc_rear_for_risk, 1, ttc_neighbor[:, None]).squeeze(1)

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
        & (neighbor_speed[:, 1] < config.stopped_neighbor_vel_thresh)
        & (max_displacement < config.stopped_neighbor_disp_thresh)
    )
    stopped_clearance, stopped_index = clearance.masked_fill(
        ~stopped_mask[None, :, None], float("inf")
    ).min(dim=1)
    stopped_is_rear = torch.gather(
        npc_rear_for_risk,
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
    time_headway_score = _linear_safe_score(thw, config.thw_critical_s, config.thw_safe_s)
    # THW is undefined at zero speed. Retain a continuous near-contact signal by
    # falling back to the forward bumper gap, and use the conservative score at
    # nonzero speed as well so stop-and-go leaders cannot look perfectly safe.
    safe_gap = config.occupancy_safe_m + config.occupancy_speed_gain_s * ego_speed
    distance_headway_score = _linear_safe_score(
        leader_gap,
        config.occupancy_critical_m,
        safe_gap,
    )
    thw_score = torch.minimum(time_headway_score, distance_headway_score)
    thw_score = torch.where(leader_present, thw_score, 1.0)

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
        "ego_speed": ego_speed,
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
            valid_segments_known=road_border_available is True,
        )
        if road_border_available is not None or torch.isfinite(road_border).any():
            sources["road_border"] = True
            # Keep the fallback on the same explicit road-border thresholds as the
            # direct reward path.  The previous planner_metrics global (normally
            # equal to 0.60 m) made --rl_road_border_safe_m ineffective whenever
            # occupancy was sourced from a road border.
            return _linear_safe_score(road_border, 0.0, config.road_border_safe_m), sources

    return torch.ones(C, T, device=device), sources


def _batched_occupancy_score(
    ego_group: torch.Tensor,
    ego_shapes: torch.Tensor,
    static_objects: torch.Tensor | None,
    line_strings: torch.Tensor | None,
    stopped_clearance: torch.Tensor,
    stopped_is_rear: torch.Tensor,
    static_available: torch.Tensor,
    stopped_available: torch.Tensor,
    road_border_available: torch.Tensor,
    config: HDPRewardConfig,
    *,
    ego_speed: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch the common stopped-agent path and loop only exact geometry sources.

    Returns occupancy ``[B, C, T]`` and source flags ``[B, 3]`` ordered as
    static/stopped/road-border.
    """
    batch_size, candidates, time_steps = ego_group.shape[:3]
    occupancy = ego_group.new_ones(batch_size, candidates, time_steps)
    source_flags = torch.stack(
        (
            static_available,
            stopped_available,
            (~static_available & ~stopped_available & road_border_available),
        ),
        dim=1,
    )

    stopped_only = ~static_available & stopped_available
    if ego_speed is None:
        ego_speed = _trajectory_speed(ego_group[..., :2], config.dt)
    safe_distance = config.occupancy_safe_m + config.occupancy_speed_gain_s * ego_speed
    critical_distance = config.occupancy_critical_m + config.occupancy_speed_gain_s * ego_speed
    stopped_score = _linear_safe_score(stopped_clearance, critical_distance, safe_distance)
    stopped_score = torch.where(
        stopped_is_rear,
        1.0 - config.rear_end_penalty * (1.0 - stopped_score),
        stopped_score,
    )
    occupancy = torch.where(stopped_only[:, None, None], stopped_score, occupancy)

    # Static OBB clearance and exact road-border distance have scene-dependent geometry sizes.
    # Keep their proven implementations and synchronize only the small exceptional index list.
    exceptional = static_available | (~stopped_available & road_border_available)
    exceptional_rows = (
        torch.cat(
            (
                exceptional.nonzero().flatten()[:, None],
                static_available[exceptional, None],
                stopped_available[exceptional, None],
                road_border_available[exceptional, None],
            ),
            dim=1,
        )
        .cpu()
        .tolist()
    )
    for scene, has_static, has_stopped, has_road_border in exceptional_rows:
        scene_occupancy, _ = _occupancy_score(
            ego_group[scene],
            ego_shapes[scene],
            static_objects[scene] if static_objects is not None else None,
            line_strings[scene] if line_strings is not None else None,
            stopped_clearance[scene],
            stopped_available[scene],
            config,
            stopped_is_rear=stopped_is_rear[scene],
            static_available=bool(has_static),
            stopped_available_flag=bool(has_stopped),
            road_border_available=bool(has_road_border),
        )
        occupancy[scene] = scene_occupancy

    return occupancy, source_flags


def _nearest_lane_distance(
    points: torch.Tensor,
    lanes: torch.Tensor,
) -> torch.Tensor:
    center = lanes[..., :2]
    valid = ~lane_point_padding_mask(lanes)
    valid_pair = (valid[..., :-1] & valid[..., 1:]).reshape(-1)
    seg_start = center[..., :-1, :].reshape(-1, 2)
    seg_end = center[..., 1:, :].reshape(-1, 2)
    flat_points = points.reshape(-1, 2)
    if seg_start.shape[0] == 0:
        return torch.full(points.shape[:-1], float("inf"), dtype=points.dtype, device=points.device)
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
    if batch_size == 0:
        return points.new_empty(points.shape[:-1])
    center = lanes[..., :2]
    point_valid = ~lane_point_padding_mask(lanes)
    segment_valid = (point_valid[..., :-1] & point_valid[..., 1:]).flatten(1)
    segment_start = center[..., :-1, :].flatten(1, 2)
    segment_end = center[..., 1:, :].flatten(1, 2)
    flat_points = points.reshape(batch_size, -1, 2)
    if segment_start.shape[1] == 0:
        return torch.full(points.shape[:-1], float("inf"), dtype=points.dtype, device=points.device)

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
        distance2 = (query[:, :, None] - closest).square().sum(dim=-1)
        min_distance2 = distance2.masked_fill(~segment_valid[:, None], float("inf")).amin(dim=-1)
        chunks.append(min_distance2.sqrt())
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
    route_valid = ~lane_point_padding_mask(route_lanes)
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
    lane_valid = ~lane_point_padding_mask(lanes)
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
    lane_point_valid = ~lane_point_padding_mask(lanes)
    lane_available = (lane_point_valid[..., :-1] & lane_point_valid[..., 1:]).flatten(1).any(1)

    if route_lanes is None:
        route_aligned = torch.zeros_like(lane_available)
        candidate_distance = _nearest_lane_distance_batch(ego_group[..., :2], lanes)
        expert_distance = _nearest_lane_distance_batch(expert_future[..., :2], lanes)
    else:
        route_point_valid = ~lane_point_padding_mask(route_lanes)
        route_available = (
            (route_point_valid[..., :-1] & route_point_valid[..., 1:]).flatten(1).any(1)
        )
        route_expert_distance = _nearest_lane_distance_batch(expert_future[..., :2], route_lanes)
        route_aligned = (
            route_available
            & (route_expert_distance.mean(dim=-1) <= config.lane_half_width_m)
            & (route_expert_distance.amax(dim=-1) <= 2.0 * config.lane_half_width_m)
        )
        candidate_distance = ego_group.new_empty(ego_group.shape[:3])
        candidate_distance[route_aligned] = _nearest_lane_distance_batch(
            ego_group[route_aligned, ..., :2], route_lanes[route_aligned]
        )
        lane_fallback = ~route_aligned
        candidate_distance[lane_fallback] = _nearest_lane_distance_batch(
            ego_group[lane_fallback, ..., :2], lanes[lane_fallback]
        )
        expert_distance = route_expert_distance.new_empty(route_expert_distance.shape)
        expert_distance[route_aligned] = route_expert_distance[route_aligned]
        expert_distance[lane_fallback] = _nearest_lane_distance_batch(
            expert_future[lane_fallback, ..., :2], lanes[lane_fallback]
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
    moving_reference = expert_distance_m >= config.stationary_reference_threshold_m
    expert_direction = (
        expert_displacement
        / expert_distance_m.clamp_min(config.stationary_reference_threshold_m)[:, None]
    )
    candidate_progress = (ego_group[:, :, -1, :2] * expert_direction[:, None]).sum(dim=-1)
    progress_ratio = torch.where(
        moving_reference[:, None],
        candidate_progress
        / expert_distance_m.clamp_min(config.stationary_reference_threshold_m)[:, None],
        torch.ones_like(candidate_progress),
    )
    if config.stationary_progress_mode == "constant":
        stationary_score = torch.ones_like(candidate_progress)
    elif config.stationary_progress_mode == "distance":
        endpoint_error = (ego_group[:, :, -1, :2] - expert_displacement[:, None]).norm(dim=-1)
        stationary_score = (1.0 - endpoint_error / config.stationary_progress_tolerance_m).clamp(
            0.0, 1.0
        )
    progress_score = torch.where(
        moving_reference[:, None], progress_ratio.clamp(0.0, 1.0), stationary_score
    )
    return (
        lane_score,
        expert_off_lane,
        expert_lane_change,
        route_aligned,
        progress_ratio,
        progress_score,
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
    horizon = int(getattr(args, "rl_reward_horizon_steps", 0) or 0)
    if horizon > 0 and horizon < ego_world.shape[1]:
        # Score only the near-term prefix (the original-DP AWR profile scores 4 s of the
        # 8 s plan). All time-indexed inputs are truncated together so collision, TTC,
        # progress and red-light terms stay mutually consistent on the same window.
        ego_world = ego_world[:, :horizon]
        neighbors_future = neighbors_future[:, :, :horizon]
        scene_inputs = dict(scene_inputs)
        scene_inputs["ego_agent_future"] = scene_inputs["ego_agent_future"][:, :horizon]
    ego_group = ego_world.reshape(num_scenes, n, ego_world.shape[1], ego_world.shape[2])
    if neighbors_future.shape[0] != num_scenes:
        raise ValueError(
            "HDP reward expects one logged-neighbor tensor per scene, got "
            f"{neighbors_future.shape[0]} tensors for {num_scenes} scenes"
        )
    neighbor_group = neighbors_future

    required = (
        "ego_current_state",
        "ego_shape",
        "neighbor_agents_past",
        "ego_agent_future",
        "lanes",
    )
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
        ego_initial_speed,
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
            ego_initial_speed=ego_initial_speed,
        )

    pair_steps_per_scene = max(
        ego_group.shape[1] * packed_neighbors[0].shape[1] * ego_group.shape[2],
        1,
    )
    scene_chunk_size = max(
        min(_RL_GEOMETRY_PAIR_STEP_BUDGET // pair_steps_per_scene, num_scenes),
        1,
    )
    scene_term_tensors = torch.vmap(
        collision_terms_for_scene,
        chunk_size=scene_chunk_size,
    )(
        ego_group,
        scene_inputs["ego_shape"][:, :3],
        *packed_neighbors,
        scene_inputs["ego_current_state"][:, 4:6].norm(dim=-1),
        expert_futures_tensor,
    )
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
    use_road_border_occupancy = bool(getattr(args, "rl_occupancy_use_road_border", True))
    road_border_weight = float(getattr(args, "rl_reward_w_road_border", 0.0))
    # Geometry validity is independent from the occupancy fallback switch. The
    # direct road-border reward must remain usable when occupancy is ablated, but
    # that switch must still prevent borders from becoming an implicit occupancy
    # source.
    use_road_border_geometry = use_road_border_occupancy or road_border_weight > 0.0
    if (
        not use_road_border_geometry
        or line_strings_batch is None
        or line_strings_batch.shape[-1] < 4
    ):
        line_border_available = torch.zeros(num_scenes, dtype=torch.bool, device=device)
    else:
        border_point = line_strings_batch[..., 3] > 0.5
        line_border_available = (border_point[..., :-1] & border_point[..., 1:]).flatten(1).any(1)
    occupancy_road_border_available = line_border_available & use_road_border_occupancy
    stopped_available = scene_term_tensors["stopped_available"]
    occupancy, occupancy_sources = _batched_occupancy_score(
        ego_group,
        scene_inputs["ego_shape"][:, :3],
        static_objects_batch,
        line_strings_batch,
        scene_term_tensors["stopped_clearance"],
        scene_term_tensors["stopped_is_rear"],
        static_available,
        stopped_available,
        occupancy_road_border_available,
        config,
        ego_speed=scene_term_tensors["ego_speed"],
    )
    collision_safety = scene_term_tensors["safety"]
    if road_border_weight > 0.0:
        road_border_clearance = _batched_road_border_clearance(
            ego_group,
            scene_inputs["ego_shape"][:, :3],
            line_strings_batch,
            line_border_available,
        )
        road_border_score_t = _linear_safe_score(
            road_border_clearance,
            config.road_border_critical_m,
            config.road_border_safe_m,
        )
        road_border_score_t = torch.where(
            line_border_available[:, None, None],
            road_border_score_t,
            torch.ones_like(road_border_score_t),
        )
        road_border_score = road_border_score_t.amin(dim=-1)
        # Below the configured clearance margin, behavior rewards must not compensate
        # for a trajectory that is already unsafe. Nearer-but-safe candidates still
        # receive a continuous score through road_border_score.
        road_border_gate = (road_border_score > 0.0).to(collision_safety.dtype)
    else:
        road_border_score_t = torch.ones_like(occupancy)
        road_border_score = torch.ones_like(collision_safety)
        road_border_gate = torch.ones_like(collision_safety)
    if bool(getattr(args, "rl_red_light_constraint", True)):
        red_light_score, red_light_active, red_light_expert_crossing = _red_light_constraint_score(
            ego_group,
            expert_futures_tensor,
            scene_inputs["ego_shape"][:, :3],
            route_lanes_batch,
            line_strings_batch,
            config,
        )
    else:
        red_light_score = torch.ones_like(collision_safety)
        red_light_active = torch.zeros(num_scenes, dtype=torch.bool, device=device)
        red_light_expert_crossing = torch.zeros_like(red_light_active)
    safety = torch.minimum(collision_safety, red_light_score)
    if road_border_weight > 0.0:
        safety = torch.minimum(safety, road_border_gate)
    risk = torch.stack(
        (scene_term_tensors["ttc"], scene_term_tensors["thw"], occupancy), dim=0
    ).amin(dim=(0, 3))
    risk = torch.minimum(risk, red_light_score)
    behavior_gate_name = getattr(args, "rl_behavior_gate", "safety")
    behavior_gate = _apply_behavior_gate(torch.ones_like(safety), safety, risk, behavior_gate_name)
    progress_reward = progress_scores * behavior_gate
    behavior_reward = (
        args.rl_reward_w_follow * scene_term_tensors["follow"]
        + args.rl_reward_w_lane * lane_scores
        + getattr(args, "rl_reward_w_progress", 0.0) * progress_scores
    )
    aggregation = getattr(args, "rl_reward_aggregation", "weighted_sum")
    if aggregation == "gated_product":
        # PDM-style bounded objective (the aggregation behind the audited original-DP
        # AWR gains): continuous multiplicative safety gates times a normalized quality
        # mix, so no behavior term can buy back an unsafe candidate and the reward stays
        # in [0, 1]. The safety weight is unused here because safety *is* the gate.
        quality_weight_sum = (
            args.rl_reward_w_risk
            + args.rl_reward_w_follow
            + args.rl_reward_w_lane
            + getattr(args, "rl_reward_w_progress", 0.0)
        )
        if quality_weight_sum <= 0.0:
            raise ValueError(
                "rl_reward_aggregation='gated_product' requires a positive "
                "risk/follow/lane/progress weight sum"
            )
        quality = (
            args.rl_reward_w_risk * risk
            + args.rl_reward_w_follow * scene_term_tensors["follow"]
            + args.rl_reward_w_lane * lane_scores
            + getattr(args, "rl_reward_w_progress", 0.0) * progress_scores
        ) / quality_weight_sum
        reward_group = collision_safety * red_light_score * road_border_score * quality
    elif aggregation == "weighted_sum":
        reward_group = (
            getattr(args, "rl_reward_w_safety", 0.0) * safety
            + args.rl_reward_w_risk * risk
            + behavior_reward * behavior_gate
            + road_border_weight * road_border_score
        )
    else:
        raise ValueError(f"Unsupported rl_reward_aggregation={aggregation!r}")
    ttc_min = scene_term_tensors["ttc"].amin(dim=-1)
    thw_min = scene_term_tensors["thw"].amin(dim=-1)
    occupancy_min = occupancy.amin(dim=-1)
    stationary_reference = (
        expert_futures_tensor[:, -1, :2].norm(dim=-1) < config.stationary_reference_threshold_m
    )

    def candidate_flags(value: torch.Tensor) -> torch.Tensor:
        return value.to(lane_scores.dtype)[:, None].expand(-1, n)

    metric_groups = {
        "safety": safety,
        "collision_safety": collision_safety,
        "risk": risk,
        "follow": scene_term_tensors["follow"],
        "lane": lane_scores,
        "ttc": ttc_min,
        "thw": thw_min,
        "occupancy": occupancy_min,
        "road_border": road_border_score,
        "road_border_available": candidate_flags(line_border_available),
        "road_border_violation_fraction": (road_border_score <= 0.0).to(risk.dtype),
        "leader_fraction": scene_term_tensors["leader_fraction"],
        "collision_active": scene_term_tensors["collision_active"],
        "collision_rear": scene_term_tensors["collision_rear"],
        "lane_masked": candidate_flags(expert_off_lanes),
        "lane_change_masked": candidate_flags(expert_lane_changes),
        "lane_route_source": candidate_flags(route_aligned),
        "occupancy_any_source": candidate_flags(occupancy_sources.any(dim=1)),
        "occupancy_static_source": candidate_flags(occupancy_sources[:, 0]),
        "occupancy_stopped_source": candidate_flags(occupancy_sources[:, 1]),
        "occupancy_road_border_source": candidate_flags(occupancy_sources[:, 2]),
        "risk_zero_fraction": (risk <= 1e-6).to(risk.dtype),
        "risk_one_fraction": (risk >= 1.0 - 1e-6).to(risk.dtype),
        "ttc_zero_fraction": (ttc_min <= 1e-6).to(risk.dtype),
        "thw_zero_fraction": (thw_min <= 1e-6).to(risk.dtype),
        "occupancy_zero_fraction": (occupancy_min <= 1e-6).to(risk.dtype),
        "progress": progress_reward,
        "progress_raw": progress_scores,
        "progress_ratio": progress_ratios.clamp(-1.0, 2.0),
        "underprogress_fraction": (progress_ratios < 0.8).to(risk.dtype),
        "overprogress_fraction": (progress_ratios > 1.2).to(risk.dtype),
        "stationary_reference_fraction": candidate_flags(stationary_reference),
        "red_light": red_light_score,
        "red_light_constraint_active_fraction": candidate_flags(red_light_active),
        "red_light_expert_crossing_fraction": candidate_flags(red_light_expert_crossing),
        "red_light_violation_fraction": ((red_light_score < 0.5) & red_light_active[:, None]).to(
            risk.dtype
        ),
        "comfort": scene_term_tensors["comfort"],
        "behavior_gate": behavior_gate,
    }
    flattened = {key: value.reshape(-1) for key, value in metric_groups.items()}
    fraction_keys = {
        "risk_zero_fraction",
        "risk_one_fraction",
        "ttc_zero_fraction",
        "thw_zero_fraction",
        "occupancy_zero_fraction",
        "road_border_violation_fraction",
        "underprogress_fraction",
        "overprogress_fraction",
        "stationary_reference_fraction",
        "red_light_constraint_active_fraction",
        "red_light_expert_crossing_fraction",
        "red_light_violation_fraction",
    }
    metrics = {
        (f"reward_{key}" if key in fraction_keys else f"reward_{key}_score"): value.mean()
        for key, value in flattened.items()
    }
    stationary_candidate_mask = candidate_flags(stationary_reference)
    # Keep these statistics additive. Callers derive conditional means only after their
    # candidate/DDP aggregation; averaging per-batch conditional means would overweight
    # batches that happen to contain very few stopped references.
    metrics["reward_stationary_progress_weighted"] = (
        progress_scores * stationary_candidate_mask
    ).mean()
    component_groups = {
        key: metric_groups[key]
        for key in (
            "safety",
            "risk",
            "follow",
            "lane",
            "progress",
            "road_border",
            "leader_fraction",
        )
    }
    reward_centered = reward_group - reward_group.mean(dim=1, keepdim=True)
    reward_winner = reward_group.argmax(dim=1, keepdim=True)
    for key in ("safety", "risk", "follow", "lane", "progress", "road_border"):
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
    for key in ("safety", "risk", "follow", "lane", "progress", "road_border"):
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
    metrics["reward_stationary_progress_group_range_weighted"] = (
        (progress_scores.max(dim=1).values - progress_scores.min(dim=1).values)
        * stationary_reference
    ).mean()
    metrics["reward_red_light_group_range"] = (
        red_light_score.max(dim=1).values - red_light_score.min(dim=1).values
    ).mean()
    return reward_group.reshape(-1), metrics


def heading_to_cos_sin_if_needed(trajectory: torch.Tensor) -> torch.Tensor:
    if trajectory.shape[-1] >= 4:
        # Return an owning tensor: reward geometry may mask padded poses in place
        # and must never mutate the caller's scene batch or a larger feature view.
        return trajectory[..., :4].clone()
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
    scene_chunk_size: int | None = None,
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
        When ``return_encoding`` is true, the second result is the unexpanded scene encoding.
        Candidate expansion is deferred to differentiable update chunks to minimize peak memory.
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
                scene_count = scene_norm_inputs["ego_current_state"].shape[0]
                if scene_count * group_size != B:
                    raise ValueError(
                        f"Expanded rollout batch {B} does not match {scene_count} scenes * "
                        f"group_size {group_size}"
                    )
                if scene_chunk_size is None:
                    scene_chunk_size = scene_count
                if scene_chunk_size <= 0 or scene_count % scene_chunk_size != 0:
                    raise ValueError(
                        "scene_chunk_size must be a positive divisor of the scene batch, got "
                        f"{scene_chunk_size} for {scene_count} scenes"
                    )

                scene_encoding = net.encoder(scene_norm_inputs)
                scene_route_condition = net.decoder.global_route_encoder(
                    scene_norm_inputs["route_lanes"]
                )
                predictions = []
                for scene_start in range(0, scene_count, scene_chunk_size):
                    scene_stop = scene_start + scene_chunk_size
                    candidate_start = scene_start * group_size
                    candidate_stop = scene_stop * group_size
                    chunk_inputs = {}
                    for key, value in inference_inputs.items():
                        if not torch.is_tensor(value) or value.ndim == 0:
                            chunk_inputs[key] = value
                        elif value.shape[0] == B:
                            chunk_inputs[key] = value[candidate_start:candidate_stop]
                        elif value.shape[0] == scene_count:
                            chunk_inputs[key] = value[scene_start:scene_stop]
                        else:
                            raise ValueError(
                                f"Rollout tensor {key!r} has batch {value.shape[0]}, expected "
                                f"{scene_count} scenes or {B} candidates"
                            )
                    chunk_encoding = scene_encoding[scene_start:scene_stop].repeat_interleave(
                        group_size, dim=0
                    )
                    chunk_inputs["_cached_global_route_condition"] = scene_route_condition[
                        scene_start:scene_stop
                    ].repeat_interleave(group_size, dim=0)
                    chunk_outputs = net.decoder(chunk_encoding, chunk_inputs)
                    predictions.append(chunk_outputs["prediction"])
                prediction = torch.cat(predictions, dim=0)
                cached_encoding = scene_encoding if return_encoding else None
            else:
                _, outputs = model(inference_inputs)
                prediction = outputs["prediction"]
    finally:
        net.decoder._sample_steps = previous_sample_steps
        if was_training:
            model.train()
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("Non-finite trajectory returned by HDP rollout sampler")
    ego_world = prediction[:, 0].detach().float()  # [B*N, T, 4]
    if return_encoding:
        return ego_world, cached_encoding
    return ego_world


@torch.no_grad()
def augment_rollout_candidates(
    ego_world: torch.Tensor,
    num_scenes: int,
    n: int,
    epoch: int,
    args,
    generator=None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The released HDP rollout-candidate augmentation, verbatim.

    Port of ``augment_trajectory_batch`` (HDP-navsim ``agent/dp_vla/scoring.py``),
    applied at the same point in the loop as the release (``dp_vla_rl_agent.py``: right
    after ``generate``, before scoring): every candidate is translated by a route-frame
    offset ``(a, b) ~ N(0, std)`` drawn once per candidate and held constant over all
    timesteps, with the heading channels untouched, for the first
    ``rl_candidate_aug_epochs`` epochs. The perturbed candidates are what the reward
    scores and what AWR regresses onto, exactly as in the release.

    ``rl_candidate_aug_epochs = 0`` disables it; the release's value is 5.

    Know the scale before enabling. This pipeline emits 80 waypoints at 10 Hz and the
    vehicle executes waypoint 1; the release emits 8 at 2 Hz and re-simulates them
    through a controller. The same 0.5 m offset is therefore ~95% of the executed first
    step here against ~20% of the release's first waypoint. See
    docs/hdp_rl_augmentation_multimodality_evidence_20260729.md.
    """
    zero = ego_world.new_zeros(())
    idle_metrics = {"reward_candidate_aug_offset_abs_mean_m": zero}
    aug_epochs = int(getattr(args, "rl_candidate_aug_epochs", 0))
    std = float(getattr(args, "rl_candidate_aug_std", 0.5))
    if aug_epochs <= 0 or std <= 0.0 or int(epoch) >= aug_epochs:
        return ego_world, idle_metrics
    if ego_world.shape[0] != num_scenes * n:
        raise ValueError(
            f"candidate augmentation expects {num_scenes * n} candidates, got {ego_world.shape[0]}"
        )

    total = num_scenes * n
    kwargs = {"device": ego_world.device, "dtype": ego_world.dtype, "generator": generator}
    # Shape (total, 1) broadcasts over the time axis: one constant offset per candidate,
    # which is what makes the released transform a rigid translation.
    a = torch.randn(total, 1, **kwargs) * std
    b = torch.randn(total, 1, **kwargs) * std
    x, y = ego_world[..., 0], ego_world[..., 1]
    cos_yaw, sin_yaw = ego_world[..., 2], ego_world[..., 3]
    augmented = torch.stack(
        (
            x + a * cos_yaw - b * sin_yaw,
            y + a * sin_yaw + b * cos_yaw,
            cos_yaw,
            sin_yaw,
        ),
        dim=-1,
    )
    metrics = {"reward_candidate_aug_offset_abs_mean_m": (a.abs() + b.abs()).mean() * 0.5}
    return augmented, metrics


def compute_reward_weights(
    reward: torch.Tensor,
    num_scenes: int,
    n: int,
    beta: float,
    eps: float,
    candidate_valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if reward.ndim != 1:
        raise ValueError(f"RL reward tensor must be 1-D, got shape {tuple(reward.shape)}")
    if not reward.is_floating_point():
        raise TypeError("RL reward tensor must use a floating-point dtype")
    if num_scenes < 1 or n < 2 or reward.numel() != num_scenes * n:
        raise ValueError(
            "RL reward shape must contain exactly num_scenes complete groups: "
            f"numel={reward.numel()}, num_scenes={num_scenes}, n={n}"
        )
    if not math.isfinite(float(beta)) or beta <= 0.0:
        raise ValueError("RL reward beta must be finite and > 0")
    if not math.isfinite(float(eps)) or eps <= 0.0:
        raise ValueError("RL reward normalization epsilon must be finite and > 0")
    grouped = reward.view(num_scenes, n)
    if candidate_valid_mask is None:
        candidate_keep = torch.ones_like(grouped, dtype=torch.bool)
        group_mean = grouped.mean(dim=1, keepdim=True)
        group_std = grouped.std(dim=1, keepdim=True)
        finite_group = torch.isfinite(grouped).all(dim=1, keepdim=True)
        # A scene whose candidates all score the same carries no action preference signal.
        valid_group = finite_group & torch.isfinite(group_std) & (group_std > eps)
    else:
        if candidate_valid_mask.shape != reward.shape:
            raise ValueError(
                "candidate_valid_mask must match the flattened reward shape "
                f"{tuple(reward.shape)}, got {tuple(candidate_valid_mask.shape)}"
            )
        if candidate_valid_mask.dtype != torch.bool:
            raise TypeError("candidate_valid_mask must be a boolean tensor")
        # Gate-rejected candidates are excluded from the group statistics as well as
        # from the weights; otherwise an infeasible candidate still shifts every other
        # candidate's advantage. A group needs at least two surviving candidates to
        # carry a preference signal.
        candidate_keep = candidate_valid_mask.view(num_scenes, n)
        keep_f = candidate_keep.to(grouped.dtype)
        valid_count = keep_f.sum(dim=1, keepdim=True)
        safe = torch.where(candidate_keep, grouped, torch.zeros_like(grouped))
        group_mean = safe.sum(dim=1, keepdim=True) / valid_count.clamp_min(1.0)
        variance = ((safe - group_mean).square() * keep_f).sum(dim=1, keepdim=True) / (
            valid_count - 1.0
        ).clamp_min(1.0)
        group_std = variance.clamp_min(0.0).sqrt()
        finite_group = (torch.isfinite(grouped) | ~candidate_keep).all(dim=1, keepdim=True)
        valid_group = (
            finite_group & (valid_count >= 2.0) & torch.isfinite(group_std) & (group_std > eps)
        )
    group_valid_sample = (valid_group.expand(-1, n) & candidate_keep).reshape(-1)
    # ap:implementation -- "we apply reward group normalization ... Additionally, we
    # discard samples in which all actions receive identical rewards". Keeping a
    # constant group at exp(0) == 1 silently turns that scene into unweighted
    # self-distillation and can dominate the useful groups.
    reward_norm = torch.where(
        valid_group,
        (grouped - group_mean) / (group_std + eps),
        torch.zeros_like(grouped),
    ).reshape(-1)
    valid_sample = group_valid_sample
    reward_norm = torch.nan_to_num(reward_norm, nan=0.0, posinf=0.0, neginf=0.0)
    # Keep the exponential finite even under an unusually large beta or a malformed
    # reward scale.  The normal configuration is far below this bound; this only
    # prevents an overflow from being silently converted to a zero weight by
    # ``nan_to_num`` downstream.
    if not reward_norm.dtype.is_floating_point:
        raise TypeError("RL reward normalization requires a floating-point reward tensor")
    # Leave a small margin because exp(round(log(max_float))) can still round to
    # infinity at the dtype boundary.
    max_logit = math.log(torch.finfo(reward_norm.dtype).max) - 1.0
    logits = (beta * reward_norm).clamp(min=-max_logit, max=max_logit)
    weights = torch.where(valid_sample, torch.exp(logits), torch.zeros_like(reward_norm))
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
    diffusion_time: torch.Tensor | None = None,
    diffusion_noise: torch.Tensor | None = None,
    loss_horizon: int | None = None,
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
    if (diffusion_time is None) != (diffusion_noise is None):
        raise ValueError("diffusion_time and diffusion_noise must be provided together")
    if diffusion_time is None:
        method = getattr(args, "diffusion_time_sample_method", "uniform")
        if method != "uniform":
            raise ValueError(f"Unsupported diffusion_time_sample_method={method!r}")
        # eq:awr_hybrid takes the expectation over the whole [eps, 1] diffusion-time range.
        t = torch.rand(B, device=device) * (1.0 - eps) + eps
    else:
        t = diffusion_time
    t_broadcast = t.view(B, 1, 1, 1)
    z = torch.randn_like(gt_future) if diffusion_noise is None else diffusion_noise
    if t.shape != (B,) or z.shape != gt_future.shape:
        raise ValueError(
            f"Invalid pre-sampled diffusion tensors: t={tuple(t.shape)}, z={tuple(z.shape)}, "
            f"expected {(B,)} and {tuple(gt_future.shape)}"
        )

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
    horizon = args.ego_prediction_horizon if loss_horizon is None else int(loss_horizon)
    if not 1 <= horizon <= ego_reconstruction.shape[1]:
        raise ValueError(f"loss horizon {horizon} outside [1, {ego_reconstruction.shape[1]}]")
    ego_loss_per_sample = ego_reconstruction[:, :horizon].mean(dim=-1)

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
    include_bc: bool = True,
    policy_diffusion_time: torch.Tensor | None = None,
    policy_diffusion_noise: torch.Tensor | None = None,
    bc_scene_active_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if reward_weights is None or valid_sample is None:
        reward_weights, valid_sample = compute_reward_weights(
            reward,
            num_scenes,
            n,
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
    # A candidate scored on a short reward prefix is regressed on that same prefix:
    # full-horizon regression of prefix-scored candidates is significantly negative
    # (the unscored tail is pure noise imitation).
    candidate_loss_horizon = int(getattr(args, "rl_reward_horizon_steps", 0) or 0)
    candidate_horizon = candidate_loss_horizon if candidate_loss_horizon > 0 else None
    # With a BC anchor, keep the candidate forward in every step so DDP static_graph sees the
    # same graph even in the rare case where every reward group is constant.
    if has_valid_group or bc_weight > 0.0:
        loss_terms = _compute_policy_ego_loss_per_sample(
            model,
            norm_inputs,
            ego_pseudo_gt,
            args,
            cached_encoding,
            policy_diffusion_time,
            policy_diffusion_noise,
            loss_horizon=candidate_horizon,
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
    if bc_weight > 0.0 and include_bc:
        if expert_norm_inputs is None or expert_ego_gt is None:
            raise ValueError("rl_bc_weight > 0 requires expert observations and trajectories")
        # The expert anchor keeps the full supervision horizon on purpose; only reward-
        # scored candidates are restricted to the scored prefix.
        expert_terms = _compute_policy_ego_loss_per_sample(
            model,
            expert_norm_inputs,
            expert_ego_gt,
            args,
            expert_cached_encoding,
        )
        expert_per_sample = expert_terms["ego_loss_per_sample"]
        active_mask = bc_scene_active_mask
        if active_mask is None and bool(getattr(args, "rl_bc_active_groups_only", True)):
            if expert_per_sample.shape[0] != num_scenes:
                raise ValueError(
                    "rl_bc_active_groups_only requires bc_scene_active_mask when the "
                    "expert batch does not match the local candidate scene count"
                )
            active_mask = valid_sample.view(num_scenes, n).any(dim=1)
        if active_mask is not None:
            # Anchoring every scene ("broad SFT") was significantly negative in the
            # original-DP AWR ablations; the retained recipe anchors only scenes that
            # still carry an active reward group.
            if active_mask.shape != expert_per_sample.shape:
                raise ValueError(
                    f"BC active-scene mask shape {tuple(active_mask.shape)} must match "
                    f"expert loss shape {tuple(expert_per_sample.shape)}"
                )
            active_f = active_mask.to(expert_per_sample.dtype)
            expert_per_sample = expert_per_sample * active_f
            expert_count = active_f.sum()
        else:
            expert_count = expert_per_sample.new_tensor(float(expert_per_sample.numel()))
        if (
            bool(getattr(args, "ddp", False))
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            torch.distributed.all_reduce(expert_count, op=torch.distributed.ReduceOp.SUM)
        bc_loss = expert_per_sample.sum() * float(ddp_world_size) / expert_count.clamp_min(1.0)

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
