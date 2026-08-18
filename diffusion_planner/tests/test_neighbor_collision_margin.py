"""compute_neighbor_collision_penalty must select margin_unknown (not silently fall back to
margin_vehicle) for a neighbor whose one-hot type is Unknown (col 11), the 4th class added
for the Unknown-class change."""

import torch

from diffusion_planner.loss import compute_ego_edge_points, compute_neighbor_collision_penalty

B, T_full, Pn = 1, 80, 1
EGO_SHAPE = torch.tensor([[2.5, 4.0, 2.0]])  # wheelbase, length, width


def _ego_edge_points():
    ego_traj = torch.zeros(B, T_full, 4)
    ego_traj[..., 2] = 1.0  # cos(heading)=1 -> heading 0, stationary at origin
    return compute_ego_edge_points(ego_traj, EGO_SHAPE, n_interp=0)


def _neighbor_tensors(type_col: int):
    """A tiny neighbor centered at the ego's origin (max overlap) with the given one-hot
    type column set, so the whole overlap depth is driven by the per-type margin inflation."""
    neighbor_agents_past = torch.zeros(B, Pn, 1, 12)
    neighbor_agents_past[:, :, -1, 6] = 0.1  # width
    neighbor_agents_past[:, :, -1, 7] = 0.1  # length
    neighbor_agents_past[:, :, -1, type_col] = 1.0

    neighbors_future = torch.zeros(B, Pn, T_full, 4)
    neighbors_future[..., 2] = 1.0  # cos(heading)=1
    neighbors_future_valid = torch.ones(B, Pn, T_full, dtype=torch.bool)
    return neighbor_agents_past, neighbors_future, neighbors_future_valid


def _penalty(type_col: int, **margins) -> torch.Tensor:
    ego_edge_points = _ego_edge_points()
    neighbor_agents_past, neighbors_future, neighbors_future_valid = _neighbor_tensors(type_col)
    return compute_neighbor_collision_penalty(
        ego_edge_points,
        neighbors_future,
        neighbors_future_valid,
        neighbor_agents_past,
        **margins,
    )


def test_unknown_margin_increases_penalty():
    small = _penalty(
        type_col=11,
        margin_vehicle=0.1,
        margin_pedestrian=0.1,
        margin_bicycle=0.1,
        margin_unknown=0.1,
    )
    large = _penalty(
        type_col=11,
        margin_vehicle=0.1,
        margin_pedestrian=0.1,
        margin_bicycle=0.1,
        margin_unknown=5.0,
    )
    assert torch.all(large >= small)
    assert torch.any(large > small)


def test_unknown_type_uses_margin_unknown_not_margin_vehicle():
    """Swapping only margin_unknown (leaving margin_vehicle untouched) must change the
    penalty for an Unknown-typed neighbor -- proves col 11 -> index 3 -> margin_unknown,
    not an off-by-one fallback onto margin_vehicle (index 0)."""
    baseline = _penalty(
        type_col=11,
        margin_vehicle=0.1,
        margin_pedestrian=0.1,
        margin_bicycle=0.1,
        margin_unknown=0.1,
    )
    bumped = _penalty(
        type_col=11,
        margin_vehicle=0.1,
        margin_pedestrian=0.1,
        margin_bicycle=0.1,
        margin_unknown=3.0,
    )
    assert not torch.allclose(baseline, bumped)


def test_vehicle_and_unknown_margins_are_independent():
    """A Vehicle-typed neighbor must be unaffected by margin_unknown, and an Unknown-typed
    neighbor must be unaffected by margin_vehicle."""
    vehicle_penalty = _penalty(
        type_col=8,
        margin_vehicle=0.1,
        margin_pedestrian=0.1,
        margin_bicycle=0.1,
        margin_unknown=5.0,
    )
    vehicle_penalty_ref = _penalty(
        type_col=8,
        margin_vehicle=0.1,
        margin_pedestrian=0.1,
        margin_bicycle=0.1,
        margin_unknown=0.1,
    )
    assert torch.allclose(vehicle_penalty, vehicle_penalty_ref)

    unknown_penalty = _penalty(
        type_col=11,
        margin_vehicle=5.0,
        margin_pedestrian=0.1,
        margin_bicycle=0.1,
        margin_unknown=0.1,
    )
    unknown_penalty_ref = _penalty(
        type_col=11,
        margin_vehicle=0.1,
        margin_pedestrian=0.1,
        margin_bicycle=0.1,
        margin_unknown=0.1,
    )
    assert torch.allclose(unknown_penalty, unknown_penalty_ref)
