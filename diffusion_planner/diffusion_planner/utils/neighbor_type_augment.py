"""Training-time augmentation that erases the type of some real neighbors.

Neighbor agents carry a one-hot type (vehicle/pedestrian/bicycle) in columns 8..10 of
``neighbor_agents_past`` that is constant across a given neighbor's whole time history
(see ``scenario_generation/tensor_converter.py`` / ``diffusion_planner_ros/utils.py``). In
training data this one-hot is always exactly one of the three known classes -- the model has
never seen anything else.

Real perception sometimes cannot confidently classify an object (Autoware's ``UNKNOWN``
class). Rather than adding a 4th "unknown" class column, "unknown" is represented the same
way a genuinely-unclassifiable object naturally would be: an all-zero type one-hot with an
otherwise valid state (position/heading/velocity/size unchanged). ``NeighborEncoder.type_emb``
(a plain ``nn.Linear(3, ...)``) already maps an all-zero input to its bias term, so this needs
no architecture change -- only training examples that exercise it.

``apply_neighbor_unknown_augment`` randomly zeroes the type one-hot of some real (non-padding)
neighbors, per training step, with a probability that depends on the neighbor's true class and
its distance from ego -- real classifiers fail more often on small/occluded VRU classes, and
lose confidence with range, so a single flat probability would not reflect that.
"""

import torch

_TYPE_BASE = 8  # one-hot [vehicle, pedestrian, bicycle] occupies columns 8..10


def apply_neighbor_unknown_augment(
    neighbor_agents_past: torch.Tensor,
    prob_vehicle: float,
    prob_pedestrian: float,
    prob_bicycle: float,
    distance_scale_max: float,
    distance_scale_range_m: float,
    prob_cap: float,
) -> torch.Tensor:
    """Zero out the type one-hot of a random subset of real neighbors.

    Args:
        neighbor_agents_past: [B, Pn, T, D] with D >= 11, type one-hot at cols 8..10.
        prob_vehicle, prob_pedestrian, prob_bicycle: base drop probability per true class.
        distance_scale_max: probability multiplier reached at/beyond distance_scale_range_m.
        distance_scale_range_m: distance (metres) at which the multiplier saturates.
        prob_cap: hard upper bound on the final per-neighbor probability.

    Returns:
        A new tensor of the same shape with the type one-hot zeroed for the selected neighbors
        (all timesteps). Padding slots (all-zero already) are left untouched.
    """
    device = neighbor_agents_past.device
    type_onehot = neighbor_agents_past[:, :, -1, _TYPE_BASE : _TYPE_BASE + 3]  # [B, Pn, 3]

    # A slot has a real, currently-assigned type iff exactly one of the 3 columns is hot; this
    # also excludes padding slots for free, since padding is all-zero everywhere including type.
    eligible = type_onehot.sum(dim=-1) > 0  # [B, Pn]

    class_idx = type_onehot.argmax(dim=-1)  # [B, Pn], meaningless where not eligible
    prob_base_table = torch.tensor(
        [prob_vehicle, prob_pedestrian, prob_bicycle],
        device=device,
        dtype=neighbor_agents_past.dtype,
    )
    prob_base = prob_base_table[class_idx]  # [B, Pn]

    dist = neighbor_agents_past[:, :, -1, 0:2].norm(dim=-1)  # [B, Pn], ego-centric frame
    scale = 1.0 + (distance_scale_max - 1.0) * (dist / distance_scale_range_m).clamp(0.0, 1.0)
    prob_final = (prob_base * scale).clamp(max=prob_cap)

    draw = torch.rand(type_onehot.shape[:2], device=device)
    drop = eligible & (draw < prob_final)  # [B, Pn]

    neighbor_agents_past = neighbor_agents_past.clone()
    neighbor_agents_past[..., _TYPE_BASE : _TYPE_BASE + 3] = torch.where(
        drop.unsqueeze(-1).unsqueeze(-1),
        torch.zeros_like(neighbor_agents_past[..., _TYPE_BASE : _TYPE_BASE + 3]),
        neighbor_agents_past[..., _TYPE_BASE : _TYPE_BASE + 3],
    )
    return neighbor_agents_past
