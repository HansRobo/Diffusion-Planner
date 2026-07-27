import math
from argparse import Namespace

import torch
import torch.nn.functional as F

from diffusion_planner.dimensions import (
    TURN_INDICATOR_OUTPUT_DIM,
    TURN_INDICATOR_OUTPUT_DISABLE,
    TURN_INDICATOR_OUTPUT_ENABLE_LEFT,
    TURN_INDICATOR_OUTPUT_ENABLE_RIGHT,
)
from planner_metrics.collision_geometry import center_rect_to_points

_NEIGHBOR_EVAL_STEPS = [0, 20, 40, 60, 79]

# One-hot agent type occupies columns 8..10 = [vehicle, pedestrian, bicycle]
# (kept consistent with the agent type encoding used by planner metrics).
_TYPE_BASE = 8
_TYPE_VEHICLE = 0
_TYPE_PEDESTRIAN = 1
_TYPE_BICYCLE = 2


def waypoints_to_velocity(waypoints: torch.Tensor) -> torch.Tensor:
    """Convert ego-centric future waypoints to HDP per-step displacement actions."""
    if waypoints.shape[-2] < 1:
        raise ValueError("waypoints_to_velocity expects at least one future timestep")
    origin = torch.zeros_like(waypoints[..., :1, :])
    velocity = torch.diff(torch.cat([origin, waypoints], dim=-2), dim=-2)
    if waypoints.shape[-1] > 2:
        velocity = torch.cat([velocity[..., :2], waypoints[..., 2:]], dim=-1)
    return velocity


def velocity_to_waypoints(velocity: torch.Tensor) -> torch.Tensor:
    """Integrate HDP per-step displacement actions back to ego-centric waypoints."""
    pos = torch.cumsum(velocity[..., :2], dim=-2)
    if velocity.shape[-1] > 2:
        return torch.cat([pos, velocity[..., 2:]], dim=-1)
    return pos


def normalize_ego_state(data: torch.Tensor, normalizer) -> torch.Tensor:
    mean, std = normalizer._mean_std_on(data.device, data.dtype)
    mean = mean[0].reshape(-1)[: data.shape[-1]]
    std = std[0].reshape(-1)[: data.shape[-1]]
    shape = (1,) * (data.ndim - 1) + (data.shape[-1],)
    return (data - mean.reshape(shape)) / std.reshape(shape)


def inverse_normalize_ego_state(data: torch.Tensor, normalizer) -> torch.Tensor:
    mean, std = normalizer._mean_std_on(data.device, data.dtype)
    mean = mean[0].reshape(-1)[: data.shape[-1]]
    std = std[0].reshape(-1)[: data.shape[-1]]
    shape = (1,) * (data.ndim - 1) + (data.shape[-1],)
    return data * std.reshape(shape) + mean.reshape(shape)


def _ego_velocity_stats(data: torch.Tensor, normalizer) -> tuple[torch.Tensor, torch.Tensor]:
    mean, std = normalizer.ego_velocity_stats_on(data.device, data.dtype)
    mean = mean.reshape(-1)[: data.shape[-1]]
    std = std.reshape(-1)[: data.shape[-1]]
    shape = (1,) * (data.ndim - 1) + (data.shape[-1],)
    return mean.reshape(shape), std.reshape(shape)


def normalize_ego_velocity(data: torch.Tensor, normalizer) -> torch.Tensor:
    mean, std = _ego_velocity_stats(data, normalizer)
    return (data - mean) / std


def inverse_normalize_ego_velocity(data: torch.Tensor, normalizer) -> torch.Tensor:
    mean, std = _ego_velocity_stats(data, normalizer)
    return data * std + mean


def sample_diffusion_time(
    batch_size: int,
    device: torch.device,
    eps: float,
    method: str,
) -> torch.Tensor:
    if method == "uniform":
        return torch.rand(batch_size, device=device) * (1.0 - eps) + eps
    raise ValueError(f"Unsupported diffusion_time_sample_method={method!r}")


def _detached_integral(v: torch.Tensor, W: int) -> torch.Tensor:
    """Integrate velocity while limiting waypoint-loss gradients to a recent window."""
    T = v.shape[-2]
    W = max(1, min(int(W), T))

    wpt = torch.cumsum(v, dim=-2)  # [..., T, 2]
    shift = torch.roll(wpt, shifts=W, dims=-2)
    shift[..., :W, :] = 0.0

    return wpt + shift.detach() - shift


def hybrid_waypoint_loss(
    pred_v_raw: torch.Tensor,
    gt_waypoints_raw: torch.Tensor,
    W: int,
) -> torch.Tensor:
    """Waypoint part of HDP hybrid loss; velocity MSE is computed by the caller."""
    pred_pos = _detached_integral(pred_v_raw[..., :2], W)  # [..., T, 2]
    return torch.sum((pred_pos - gt_waypoints_raw[..., :2]) ** 2, dim=-1)  # [..., T]


def make_turn_indicator_gt(
    turn_indicators: torch.Tensor,  # [B, INPUT_T + 1]
) -> torch.Tensor:
    """Map the current raw Autoware report to dense OFF/LEFT/RIGHT labels."""
    if turn_indicators.ndim != 2:
        raise ValueError(f"turn_indicators must be [B, T], got {tuple(turn_indicators.shape)}")
    raw_values = turn_indicators[:, -1]
    raw_labels = raw_values.long()
    invalid = (
        ~torch.isfinite(raw_values)
        | (raw_values != raw_labels.to(dtype=raw_values.dtype))
        | (raw_labels < 1)
        | (raw_labels > 3)
    )
    if invalid.any():
        values = torch.unique(raw_values[invalid]).detach().cpu().tolist()
        raise ValueError(f"Invalid TurnIndicatorsReport values at current frame: {values}")
    return raw_labels - 1


# Opposite driving state per dense class; OFF has no opposite direction.
_TURN_INDICATOR_OPPOSITE = {
    TURN_INDICATOR_OUTPUT_ENABLE_LEFT: TURN_INDICATOR_OUTPUT_ENABLE_RIGHT,
    TURN_INDICATOR_OUTPUT_ENABLE_RIGHT: TURN_INDICATOR_OUTPUT_ENABLE_LEFT,
}


def turn_indicator_geometry_evidence(
    ego_future: torch.Tensor,  # [B, T, >=4] raw ego-frame waypoints (x, y, cos, sin)
    *,
    min_yaw_rad: float,
    full_yaw_rad: float,
    min_lateral_m: float,
    full_lateral_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Which way the expert future actually turns, and how unambiguously.

    Mirrors ``util_scripts/audit_turn_indicator_events.py``: ``y`` and yaw are
    left-positive, yaw is measured from the first future frame, and evidence is the
    horizon maximum of the same-side quantity. Strength ramps 0->1 between the audited
    weak bar (5 deg / 0.5 m) and a strong bar, and is reduced by whatever evidence the
    opposite side shows, so an S-shaped future cannot masquerade as a turn.

    Returns ``(direction, strength)`` with direction in the dense OFF/LEFT/RIGHT space
    and ``strength`` in [0, 1]; direction is OFF exactly where strength is zero.
    """
    if ego_future.ndim != 3 or ego_future.shape[-1] < 4:
        raise ValueError(f"ego_future must be [B, T, >=4], got {tuple(ego_future.shape)}")
    if not 0.0 <= min_yaw_rad < full_yaw_rad:
        raise ValueError(
            f"require 0 <= min_yaw_rad < full_yaw_rad, got {min_yaw_rad}/{full_yaw_rad}"
        )
    if not 0.0 <= min_lateral_m < full_lateral_m:
        raise ValueError(
            f"require 0 <= min_lateral_m < full_lateral_m, got {min_lateral_m}/{full_lateral_m}"
        )

    # Unwrapped heading relative to the first future frame: wrapping the per-step delta
    # keeps a hard turn monotonic instead of folding it back at +-pi.
    yaw = torch.atan2(ego_future[..., 3], ego_future[..., 2])  # [B, T]
    delta = torch.diff(yaw, dim=-1)
    delta = torch.atan2(torch.sin(delta), torch.cos(delta))
    yaw = torch.cat([torch.zeros_like(yaw[:, :1]), torch.cumsum(delta, dim=-1)], dim=-1)
    lateral = ego_future[..., 1]

    def ramp(value: torch.Tensor, low: float, high: float) -> torch.Tensor:
        return ((value - low) / (high - low)).clamp(0.0, 1.0)

    def side_strength(sign: float) -> torch.Tensor:
        return torch.maximum(
            ramp((sign * yaw).amax(dim=-1), min_yaw_rad, full_yaw_rad),
            ramp((sign * lateral).amax(dim=-1), min_lateral_m, full_lateral_m),
        )

    left = side_strength(1.0)
    right = side_strength(-1.0)
    strength = (left - right).abs()
    direction = torch.where(
        strength > 0,
        torch.where(
            left >= right,
            torch.full_like(strength, TURN_INDICATOR_OUTPUT_ENABLE_LEFT),
            torch.full_like(strength, TURN_INDICATOR_OUTPUT_ENABLE_RIGHT),
        ),
        torch.full_like(strength, TURN_INDICATOR_OUTPUT_DISABLE),
    ).long()
    return direction, strength


def turn_indicator_soft_targets(
    target: torch.Tensor,  # [B] dense OFF/LEFT/RIGHT labels
    implied_direction: torch.Tensor,  # [B]
    implied_strength: torch.Tensor,  # [B]
    *,
    smoothing: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Relax OFF labels whose expert future already commits to a turn.

    Human lever timing leads the maneuver by a median 3.5 s (1.6-6.8 s p10-p90), so an
    OFF frame with a clearly turning future is a late annotation rather than a negative.
    At most ``smoothing`` of that sample's target mass moves from OFF onto the implied
    direction. Active labels stay one-hot and mass is never moved between LEFT and
    RIGHT, so the objective can never learn that the two directions are interchangeable.

    Returns ``(soft_target, moved_mass)``; both are plain data with no autograd history.
    """
    if not 0.0 <= smoothing < 1.0:
        raise ValueError(f"turn-indicator smoothing must be in [0, 1), got {smoothing}")
    soft = F.one_hot(target, TURN_INDICATOR_OUTPUT_DIM).to(dtype=implied_strength.dtype)
    moved = torch.zeros_like(implied_strength)
    if smoothing > 0.0:
        eligible = (target == TURN_INDICATOR_OUTPUT_DISABLE) & (
            implied_direction != TURN_INDICATOR_OUTPUT_DISABLE
        )
        moved = smoothing * implied_strength * eligible.to(dtype=implied_strength.dtype)
        soft[:, TURN_INDICATOR_OUTPUT_DISABLE] = soft[:, TURN_INDICATOR_OUTPUT_DISABLE] - moved
        # ``moved`` is zero wherever the implied direction is OFF, so scattering it back
        # into column OFF for those rows is a no-op.
        soft = soft.scatter_add(1, implied_direction.unsqueeze(-1), moved.unsqueeze(-1))
    return soft, moved


def turn_indicator_objective(
    logit: torch.Tensor,  # [B, TURN_INDICATOR_OUTPUT_DIM] fp32 logits
    target: torch.Tensor,  # [B] dense OFF/LEFT/RIGHT labels
    ego_future: torch.Tensor | None,  # [B, T, >=4] raw expert future, or None
    args: Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Per-sample turn-intent loss priced by the deployment cost of each error.

    Cross entropy against the (optionally relaxed) target keeps the head a proper
    scoring rule, because the deployment state machine gates on absolute probabilities
    rather than on the argmax. Added to it is the one cost plain cross entropy gets
    wrong: commanding the opposite direction is not interchangeable with a late or
    leaked signal, so opposite-direction probability mass carries an explicit expected
    cost. Both terms reduce to the legacy objective at their zero defaults.

    Returns the unreduced per-sample loss and detached scalar diagnostics.
    """
    if logit.shape[-1] != TURN_INDICATOR_OUTPUT_DIM:
        raise ValueError(f"turn-indicator logits must be [B, {TURN_INDICATOR_OUTPUT_DIM}]")
    smoothing = float(getattr(args, "turn_indicator_implied_intent_smoothing", 0.0))
    opposite_weight = float(getattr(args, "turn_indicator_opposite_direction_weight", 0.0))
    if opposite_weight < 0.0:
        raise ValueError(
            f"turn_indicator_opposite_direction_weight must be non-negative, got {opposite_weight}"
        )

    if smoothing > 0.0 and ego_future is not None:
        # Evidence comes from the expert future in fp32 regardless of the autocast dtype:
        # a bf16 heading integral would make the ramp thresholds batch-dependent.
        implied_direction, implied_strength = turn_indicator_geometry_evidence(
            ego_future.float(),
            min_yaw_rad=math.radians(
                float(getattr(args, "turn_indicator_implied_intent_min_yaw_deg", 5.0))
            ),
            full_yaw_rad=math.radians(
                float(getattr(args, "turn_indicator_implied_intent_full_yaw_deg", 20.0))
            ),
            min_lateral_m=float(getattr(args, "turn_indicator_implied_intent_min_lateral_m", 0.5)),
            full_lateral_m=float(
                getattr(args, "turn_indicator_implied_intent_full_lateral_m", 2.0)
            ),
        )
    else:
        smoothing = 0.0
        implied_direction = torch.full_like(target, TURN_INDICATOR_OUTPUT_DISABLE)
        implied_strength = torch.zeros(target.shape, dtype=logit.dtype, device=logit.device)

    soft_target, moved = turn_indicator_soft_targets(
        target, implied_direction, implied_strength.to(dtype=logit.dtype), smoothing=smoothing
    )
    loss = -(soft_target * F.log_softmax(logit, dim=-1)).sum(dim=-1)  # [B]

    opposite = torch.full_like(target, TURN_INDICATOR_OUTPUT_DISABLE)
    for active, mirrored in _TURN_INDICATOR_OPPOSITE.items():
        opposite = torch.where(target == active, torch.full_like(opposite, mirrored), opposite)
    active_mask = target != TURN_INDICATOR_OUTPUT_DISABLE
    opposite_probability = F.softmax(logit, dim=-1).gather(1, opposite.unsqueeze(-1)).squeeze(-1)
    opposite_probability = opposite_probability * active_mask.to(dtype=logit.dtype)
    if opposite_weight > 0.0:
        loss = loss + opposite_weight * opposite_probability

    with torch.no_grad():
        diagnostics = {
            "turn_indicator_opposite_probability": _masked_mean(opposite_probability, active_mask),
            "turn_indicator_implied_intent_mass": _masked_mean(
                moved, target == TURN_INDICATOR_OUTPUT_DISABLE
            ),
        }
    return loss, diagnostics


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``values`` over ``mask``, or zero when the mask selects nothing."""
    count = mask.sum()
    total = (values * mask.to(dtype=values.dtype)).sum()
    return total / count.to(dtype=values.dtype) if count > 0 else torch.zeros_like(total)


def loss_func(
    trajectory_pred: torch.Tensor, trajectory_gt: torch.Tensor
) -> dict[str, torch.Tensor]:
    """
    Calculate the loss between predicted and ground truth trajectories.

    Args:
        trajectory_pred (torch.Tensor): Predicted trajectory of shape [..., T, D].
        trajectory_gt (torch.Tensor): Ground truth trajectory of shape [..., T, D].
        where, D=4 (x, y, cos, sin).

    Returns:
        dict[str, torch.Tensor]: A dictionary containing the loss values.
        where, each loss' shape is [..., T].
    """
    result_dict = {}

    ###################
    # Basic L2 Losses #
    ###################
    # simple L2 loss
    result_dict["simple_l2_loss"] = torch.mean((trajectory_pred - trajectory_gt) ** 2, dim=-1)

    # Position loss (x, y coordinates)
    position_pred = trajectory_pred[..., :2]  # [..., T, 2]
    position_gt = trajectory_gt[..., :2]  # [..., T, 2]

    # Calculate L2 distance for each time step
    position_diff = position_pred - position_gt  # [..., T, 2]
    position_error = torch.sum(position_diff**2, dim=-1)  # [..., T]
    result_dict["position_l2_loss"] = position_error

    # Heading loss (cos, sin components)
    cos_sin_pred = trajectory_pred[..., 2:]  # [..., T, 2]
    cos_sin_gt = trajectory_gt[..., 2:]  # [..., T, 2]

    # heading l2 loss
    heading_loss = torch.sum((cos_sin_pred - cos_sin_gt) ** 2, dim=-1)  # [..., T]
    result_dict["heading_l2_loss"] = heading_loss

    ######################
    # Specialized Losses #
    ######################
    # Lateral or longitudinal error (along vehicle direction)
    cos_gt = cos_sin_gt[..., 0]  # [..., T]
    sin_gt = cos_sin_gt[..., 1]  # [..., T]
    lon_diff = +position_diff[..., 0] * cos_gt + position_diff[..., 1] * sin_gt  # [..., T]
    lat_diff = -position_diff[..., 0] * sin_gt + position_diff[..., 1] * cos_gt  # [..., T]
    lat_error = torch.abs(lat_diff)  # [..., T]
    lon_error = torch.abs(lon_diff)  # [..., T]
    result_dict["position_lat_loss"] = lat_error
    result_dict["position_lon_loss"] = lon_error

    # Cosine similarity loss
    cosine_similarity = torch.sum(cos_sin_pred * cos_sin_gt, dim=-1)  # [..., T]
    result_dict["cosine_similarity_loss"] = 1.0 - cosine_similarity  # [..., T]

    return result_dict


def point_to_segment_distance(
    p: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Compute distance from points to line segments.

    Args:
        p: [..., 2] query points.
        a: [..., 2] segment start points.
        b: [..., 2] segment end points.

    Returns:
        dist: [...] non-negative distances.
    """
    ab = b - a
    ap = p - a
    t = (ap * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-8)
    t = t.clamp(0.0, 1.0)
    closest = a + t.unsqueeze(-1) * ab
    return ((p - closest) ** 2).sum(-1).clamp_min(1e-8).sqrt()


def compute_ego_bbox_corners(
    ego_traj: torch.Tensor,
    ego_shape: torch.Tensor,
) -> torch.Tensor:
    """Compute ego bounding box corners from trajectory and vehicle shape.

    Args:
        ego_traj: [B, T, 4] ego trajectory (x, y, cos_heading, sin_heading).
        ego_shape: [B, 3] (wheelbase, length, width).

    Returns:
        corners: [B, T, 4, 2] four corners per timestep.
    """
    B, T, _ = ego_traj.shape
    device = ego_traj.device
    dtype = ego_traj.dtype

    heading = ego_traj[..., 2:]
    heading_unit = heading / torch.linalg.norm(heading, dim=-1, keepdim=True).clamp_min(1e-6)
    ego_xy = ego_traj[..., :2]

    cog_to_rear = 0.5 * ego_shape[:, 0:1].unsqueeze(-1)  # [B, 1, 1]
    ego_center_xy = ego_xy + heading_unit * cog_to_rear

    half_length = (ego_shape[:, 1] / 2.0).unsqueeze(-1).expand(-1, T)
    half_width = (ego_shape[:, 2] / 2.0).unsqueeze(-1).expand(-1, T)
    half_sizes = torch.stack([half_length, half_width], dim=-1)  # [B, T, 2]

    corner_signs = torch.tensor(
        [[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    local_corners = corner_signs[None, None, :, :] * half_sizes[:, :, None, :]
    rot = torch.stack(
        [
            heading_unit[..., 0],
            -heading_unit[..., 1],
            heading_unit[..., 1],
            heading_unit[..., 0],
        ],
        dim=-1,
    ).reshape(B, T, 2, 2)
    rotated_corners = torch.einsum("btij,btkj->btki", rot, local_corners)
    return ego_center_xy[:, :, None, :] + rotated_corners


def compute_ego_edge_points(
    ego_traj: torch.Tensor,
    ego_shape: torch.Tensor,
    n_interp: int,
) -> torch.Tensor:
    """Compute sample points along ego bounding box edges.

    Args:
        ego_traj: [B, T, 4] ego trajectory (x, y, cos_heading, sin_heading).
        ego_shape: [B, 3] (wheelbase, length, width).
        n_interp: number of intermediate points per edge.
            n_interp=0: 4 points (corners only).
            n_interp=1: 8 points (corners + midpoints).

    Returns:
        points: [B, T, 4*(n_interp+1), 2] sampled points.
    """
    corners = compute_ego_bbox_corners(ego_traj, ego_shape)  # [B, T, 4, 2]

    starts = corners  # [B, T, 4, 2]
    ends = torch.roll(corners, -1, dims=2)  # [B, T, 4, 2]

    n_pts = n_interp + 1
    t = torch.linspace(0.0, 1.0, n_pts + 1, device=corners.device)[:-1]
    t = t.reshape(1, 1, 1, n_pts, 1)

    starts = starts.unsqueeze(3)  # [B, T, 4, 1, 2]
    ends = ends.unsqueeze(3)  # [B, T, 4, 1, 2]

    points = starts + t * (ends - starts)  # [B, T, 4, n_pts, 2]
    B, T = points.shape[:2]
    return points.reshape(B, T, 4 * n_pts, 2)


def compute_road_border_penalty(
    ego_edge_points: torch.Tensor,
    line_strings: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Compute road border penalty for ego trajectory.

    Args:
        ego_edge_points: [B, T, K, 2] sample points on ego bbox edges.
        line_strings: [B, N, P, D] denormalized line strings.
        margin: distance threshold (meters).

    Returns:
        penalty: [B, T] non-negative penalty per timestep.
    """
    line_strings_xy = line_strings[..., :2]  # [B, N, P, 2]
    border_point = line_strings[..., 3] > 0.5
    road_border_mask = border_point.any(dim=-1)  # [B, N]

    B, T, K, _ = ego_edge_points.shape
    device = ego_edge_points.device

    # Pre-filter: keep only line strings that are road border in any batch
    any_rb = road_border_mask.any(dim=0)  # [N]
    if not any_rb.any():
        return torch.zeros(B, T, device=device)

    # Segment endpoints: [B, N, S, 2]
    seg_a = line_strings_xy[:, :, :-1, :]
    seg_b = line_strings_xy[:, :, 1:, :]
    S = seg_a.shape[2]

    # Coordinates at the ego origin are valid map points. Channel 3 is the explicit
    # per-point road-border flag, so both endpoints must carry that flag.
    seg_valid = border_point[:, :, :-1] & border_point[:, :, 1:]  # [B, N, S]

    # Pre-filter valid line strings to reduce memory
    valid_ls_indices = any_rb.nonzero(as_tuple=True)[0]  # [M]
    seg_a = seg_a[:, valid_ls_indices]  # [B, M, S, 2]
    seg_b = seg_b[:, valid_ls_indices]  # [B, M, S, 2]
    seg_valid = seg_valid[:, valid_ls_indices]  # [B, M, S]
    M = valid_ls_indices.shape[0]

    # Flatten segments: [B, M*S, 2]
    seg_a_flat = seg_a.reshape(B, M * S, 2)
    seg_b_flat = seg_b.reshape(B, M * S, 2)
    seg_valid_flat = seg_valid.reshape(B, M * S)

    # Compute distances: ego_edge_points [B, T, K, 2] vs segments [B, M*S, 2]
    p = ego_edge_points.reshape(B, T * K, 1, 2)
    a = seg_a_flat[:, None, :, :]  # [B, 1, M*S, 2]
    b = seg_b_flat[:, None, :, :]  # [B, 1, M*S, 2]

    dist = point_to_segment_distance(p, a, b)  # [B, T*K, M*S]

    # Mask invalid segments
    dist = torch.where(
        seg_valid_flat[:, None, :],
        dist,
        torch.full_like(dist, float("inf")),
    )

    # Min over all segments and all edge points per timestep
    min_dist_per_point = dist.min(dim=-1).values  # [B, T*K]
    min_dist_per_point = min_dist_per_point.reshape(B, T, K)
    min_dist = min_dist_per_point.min(dim=-1).values  # [B, T]

    return torch.where(
        torch.isfinite(min_dist),
        F.relu(margin - min_dist),
        torch.zeros_like(min_dist),
    )


def _box_outward_normals(corners: torch.Tensor) -> torch.Tensor:
    """Unit outward edge normals of a convex box. corners: [..., 4, 2] -> [..., 4, 2]."""
    edges = torch.roll(corners, -1, dims=-2) - corners
    normals = torch.stack([-edges[..., 1], edges[..., 0]], dim=-1)
    return normals / torch.linalg.norm(normals, dim=-1, keepdim=True).clamp_min(1e-9)


def sat_penetration_depth(ego_corners: torch.Tensor, nbr_corners: torch.Tensor) -> torch.Tensor:
    """SAT overlap (penetration) depth between an ego box and neighbor boxes.

    Unlike the point-to-segment distance, this registers *any* polygon overlap (including
    shallow / interpenetrating ones the discrete edge-point sampling misses), returning the
    minimum-translation-distance magnitude.

    Args:
        ego_corners: [B, S, 4, 2] ego box corners per (batch, eval step).
        nbr_corners: [B, S, Pn, 4, 2] neighbor box corners.

    Returns:
        depth: [B, S, Pn] >= 0 (0 when the boxes are disjoint).
    """
    B, S, Pn, _, _ = nbr_corners.shape
    ego = ego_corners[:, :, None].expand(B, S, Pn, 4, 2)
    # candidate separating axes: edge normals of both boxes.
    axes = torch.cat([_box_outward_normals(ego), _box_outward_normals(nbr_corners)], dim=-2)
    pe = torch.einsum("bspax,bspcx->bspac", axes, ego)  # [B, S, Pn, 8, 4]
    pn = torch.einsum("bspax,bspcx->bspac", axes, nbr_corners)  # [B, S, Pn, 8, 4]
    overlap = torch.minimum(pe.amax(-1), pn.amax(-1)) - torch.maximum(pe.amin(-1), pn.amin(-1))
    # disjoint <=> some axis has non-positive overlap -> min over axes <= 0 -> relu = 0.
    return overlap.amin(-1).clamp_min(0.0)  # [B, S, Pn]


def compute_neighbor_collision_penalty(
    ego_edge_points: torch.Tensor,
    neighbors_future: torch.Tensor,
    neighbors_future_valid: torch.Tensor,
    neighbor_agents_past: torch.Tensor,
    margin_vehicle: float,
    margin_pedestrian: float,
    margin_bicycle: float,
    eval_steps: "list[int] | None" = None,
) -> torch.Tensor:
    """Compute neighbor collision penalty for ego trajectory.

    The neighbor box is inflated per agent type and the penalty is the SAT penetration depth of
    the ego box into that inflated box. This is equivalent to a ``relu(margin - distance)``
    proximity hinge while a buffer remains, smoothly continuing into a true overlap-depth penalty
    once the boxes actually touch -- and, unlike a point-to-segment distance, it does not depend
    on any discrete edge-point sampling (so it never misses shallow overlaps or a small agent
    sitting inside the ego footprint).

    The inflation amount is set independently per agent type (one-hot type in past cols 8..10);
    all three must be specified explicitly by the caller.

    Args:
        ego_edge_points: [B, T, K, 2] sample points on ego bbox edges (only the 4 corners,
            i.e. every ``K // 4``-th point, are used here).
        neighbors_future: [B, Pn, T, 4] neighbor future trajectories in world frame.
        neighbors_future_valid: [B, Pn, T] validity mask for neighbor timesteps.
        neighbor_agents_past: [B, Pn_max, T_past, D] denormalized neighbor past states.
        margin_vehicle: per-side inflation (meters) for vehicles.
        margin_pedestrian: per-side inflation (meters) for pedestrians.
        margin_bicycle: per-side inflation (meters) for bicycles.

    Returns:
        penalty: [B, T] non-negative penalty per timestep.
    """
    B, T_full, K, _ = ego_edge_points.shape
    Pn = neighbors_future.shape[1]
    device = ego_edge_points.device

    step_list = _NEIGHBOR_EVAL_STEPS if eval_steps is None else eval_steps
    steps = torch.tensor(
        sorted({s for s in step_list if s < T_full}), device=device, dtype=torch.long
    )
    S = steps.shape[0]
    if S == 0:
        return torch.zeros(B, T_full, device=device)

    # Drop neighbor slots that are invalid at every eval step across the whole batch.
    # The penalty is an amax over neighbors and invalid neighbors contribute zero, so
    # removing globally-invalid (padding) slots leaves the result unchanged while
    # shrinking the SAT tensors from the padded Pn (e.g. 320) down to the real count.
    neighbor_agents_past = neighbor_agents_past[:, :Pn]
    keep = neighbors_future_valid[:, :, steps].any(dim=2).any(dim=0)  # [Pn]
    if not bool(keep.any()):
        return torch.zeros(B, T_full, device=device)
    neighbors_future = neighbors_future[:, keep]
    neighbors_future_valid = neighbors_future_valid[:, keep]
    neighbor_agents_past = neighbor_agents_past[:, keep]
    Pn = neighbors_future.shape[1]

    # Ego box corners at eval timesteps (corners are every K // 4-th edge sample point).
    ego_corners = ego_edge_points[:, steps][:, :, :: K // 4, :]  # [B, S, 4, 2]

    # Per-neighbor inflation margin selected by the agent type one-hot (cols 8..10).
    neighbor_sizes = neighbor_agents_past[:, :Pn, -1, :]
    margin_by_type = torch.tensor(
        [margin_vehicle, margin_pedestrian, margin_bicycle],
        device=device,
        dtype=neighbor_sizes.dtype,
    )  # ordered [vehicle, pedestrian, bicycle]
    type_idx = neighbor_sizes[..., _TYPE_BASE : _TYPE_BASE + 3].argmax(dim=-1)  # [B, Pn]
    neighbor_margin = margin_by_type[type_idx]  # [B, Pn]

    # Neighbor sizes from last past timestep, inflated by the per-type margin on each side.
    neighbor_width = (
        torch.clamp(neighbor_sizes[..., 6], min=1e-3) + 2.0 * neighbor_margin
    )  # [B, Pn]
    neighbor_length = (
        torch.clamp(neighbor_sizes[..., 7], min=1e-3) + 2.0 * neighbor_margin
    )  # [B, Pn]

    # Neighbor pose at eval timesteps.
    neighbor_pos = neighbors_future[:, :, steps, :2]  # [B, Pn, S, 2]
    neighbor_cos = neighbors_future[:, :, steps, 2]  # [B, Pn, S]
    neighbor_sin = neighbors_future[:, :, steps, 3]  # [B, Pn, S]
    orientation_norm = torch.sqrt(neighbor_cos**2 + neighbor_sin**2).clamp_min(1e-6)
    neighbor_cos = neighbor_cos / orientation_norm
    neighbor_sin = neighbor_sin / orientation_norm

    # Build the inflated neighbor rect: [B, Pn, S, 6] -> corners [B, Pn, S, 4, 2].
    neighbor_rect = torch.stack(
        [
            neighbor_pos[..., 0],
            neighbor_pos[..., 1],
            neighbor_cos,
            neighbor_sin,
            neighbor_length.unsqueeze(-1).expand(-1, -1, S),
            neighbor_width.unsqueeze(-1).expand(-1, -1, S),
        ],
        dim=-1,
    )
    neighbor_corners = center_rect_to_points(neighbor_rect.reshape(-1, 6)).reshape(B, Pn, S, 4, 2)

    # SAT penetration of the ego box into each inflated neighbor box.
    nbr_corners = neighbor_corners.permute(0, 2, 1, 3, 4)  # [B, S, Pn, 4, 2]
    penetration = sat_penetration_depth(ego_corners, nbr_corners)  # [B, S, Pn]
    valid = neighbors_future_valid[:, :, steps].permute(0, 2, 1)  # [B, S, Pn]
    penetration = torch.where(valid, penetration, torch.zeros_like(penetration))
    penalty_s = penetration.amax(dim=-1)  # worst (deepest) overlap per (B, step)

    # Scatter to full T
    penalty = torch.zeros(B, T_full, device=device)
    penalty[:, steps] = penalty_s

    return penalty
