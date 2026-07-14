import torch
import torch.nn.functional as F

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


# Turn-indicator state loss shaping. Human signal timing already leads the maneuver
# (signal-on precedes the turn by seconds), so the current-frame state is the label —
# no transition classes, no exact-frame pathology. The class weights counter the
# heavy off/unset majority; the onset bonus focuses learning on the second right
# after a real switch, where any frame in the window teaches the correct new state.
_TURN_INDICATOR_CLASS_WEIGHTS = (0.1, 0.1, 1.0, 1.0)  # none, disable, left, right
_TURN_INDICATOR_ONSET_BONUS = 5.0
_TURN_INDICATOR_ONSET_WINDOW = 10  # frames at 10 Hz = 1 s


def make_turn_indicator_gt(
    turn_indicators: torch.Tensor,  # [B, INPUT_T + 1]
) -> torch.Tensor:
    """Return the raw indicator STATE at the current frame as the class label."""
    return turn_indicators[:, -1].long()


def turn_indicator_loss_weights(turn_indicators: torch.Tensor) -> torch.Tensor:
    """Per-sample CE weights: class weight, boosted right after a state switch."""
    if turn_indicators.ndim != 2:
        raise ValueError(f"turn_indicators must be [B, T], got {tuple(turn_indicators.shape)}")
    labels = turn_indicators[:, -1].long()
    class_weights = torch.tensor(
        _TURN_INDICATOR_CLASS_WEIGHTS, dtype=torch.float32, device=turn_indicators.device
    )
    weights = class_weights[labels]
    changes = turn_indicators[:, 1:] != turn_indicators[:, :-1]
    recent_change = changes[:, -_TURN_INDICATOR_ONSET_WINDOW:].any(dim=1)
    return torch.where(recent_change, weights * _TURN_INDICATOR_ONSET_BONUS, weights)


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
