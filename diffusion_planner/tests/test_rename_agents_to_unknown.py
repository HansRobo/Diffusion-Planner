"""rename_agents_to_unknown: the training-time "rename to Unknown with probability p"
augmentation (data_augmentation.py). Off by default (prob=0.0).

The tests below the divider cover the optional shaping knobs -- per-class rates,
distance scaling, the cap. They are statistical, so they use enough agents that the
binomial spread is far smaller than the effect being asserted.
"""

from unittest.mock import patch

import torch
from diffusion_planner.grpo_utils import expand_batch
from diffusion_planner.utils.data_augmentation import rename_agents_to_unknown

B, N, T, D = 2, 4, 5, 12


def _make_batch():
    x = torch.zeros(B, N, T, D)
    # Two valid agents per scene (vehicle, pedestrian); the rest are padding.
    x[:, 0, :, 0] = 1.0
    x[:, 0, :, 8] = 1.0  # vehicle
    x[:, 1, :, 1] = 1.0
    x[:, 1, :, 9] = 1.0  # pedestrian
    return x


def test_prob_zero_is_a_true_noop_and_draws_no_random_numbers():
    x = _make_batch()
    original = x.clone()
    with patch("torch.rand") as mock_rand:
        out, renamed, valid = rename_agents_to_unknown(x, 0.0)
    mock_rand.assert_not_called()
    assert out is x
    assert torch.equal(out, original)
    assert not renamed.any()
    # valid_mask is still reported accurately even when the feature is off, so callers can
    # show "0/2 renamed" rather than "0/0" -- the whole point is confirming it's off.
    assert torch.equal(valid, torch.tensor([[True, True, False, False]] * B))


def test_prob_one_renames_all_valid_agents_every_timestep():
    x = _make_batch()
    out, renamed, valid = rename_agents_to_unknown(x, 1.0)

    assert torch.equal(renamed, valid)
    # Valid agents (slots 0, 1) become Unknown at every past timestep.
    for slot in (0, 1):
        assert torch.equal(
            out[:, slot, :, 8:12], torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(B, T, 4)
        )
    # Padding slots (2, 3) are untouched (still all-zero).
    assert torch.all(out[:, 2:, :, :] == 0.0)
    # Kinematic columns (0:8, excluding the type one-hot) are bit-identical.
    original = _make_batch()
    assert torch.equal(out[..., :6], original[..., :6])
    assert torch.equal(out[..., 6:8], original[..., 6:8])


def test_renamed_agent_is_consistent_across_all_timesteps():
    """An agent's class must not flip frame-to-frame -- either all timesteps are Unknown or
    none are, never a mix."""
    torch.manual_seed(0)
    x = _make_batch()
    out, _renamed, _valid = rename_agents_to_unknown(x, 0.5)
    is_unknown_per_t = out[..., 11] == 1.0  # [B, N, T]
    all_t_agree = (is_unknown_per_t.all(dim=-1) == is_unknown_per_t.any(dim=-1)).all()
    assert bool(all_t_agree)


def test_padding_slots_never_renamed():
    torch.manual_seed(1)
    x = _make_batch()
    out, renamed, _valid = rename_agents_to_unknown(x, 1.0)  # even at prob=1.0
    assert torch.all(out[:, 2:, :, 8:12] == 0.0)
    assert not renamed[:, 2:].any()


def test_renamed_mask_matches_which_agents_actually_changed():
    torch.manual_seed(3)
    x = _make_batch()
    out, renamed, _valid = rename_agents_to_unknown(x, 0.5)
    became_unknown = out[:, :, -1, 11] == 1.0  # [B, N]
    assert torch.equal(renamed, became_unknown)


def test_grpo_group_invariant_rename_before_expand():
    """Mirrors _grpo_step's actual call order: rename runs on the per-scene batch BEFORE
    expand_batch, so every one of the n expanded copies of a scene gets an identical rename
    decision -- required for comparable group advantages."""
    torch.manual_seed(2)
    raw = {"neighbor_agents_past": _make_batch()}
    n = 8
    raw["neighbor_agents_past"], _renamed, _valid = rename_agents_to_unknown(
        raw["neighbor_agents_past"], 0.5
    )
    exp = expand_batch(raw, n)
    onehot = exp["neighbor_agents_past"][..., 8:12]  # [B*n, N, T, 4]
    onehot = onehot.view(B, n, N, T, 4)
    for b in range(B):
        first = onehot[b, 0]
        for k in range(1, n):
            assert torch.equal(onehot[b, k], first)


# --------------------------------------------------------------------------------------
# Shaping knobs: per-class rate, distance scaling, cap. Every default is a no-op, so the
# tests above -- which pass only `prob` -- also pin down that the defaults change nothing.
# --------------------------------------------------------------------------------------

BIG_N = 4000


def _make_population(n_per_class: int = BIG_N, distance_m: float = 1.0) -> torch.Tensor:
    """One scene holding n_per_class vehicles, pedestrians and bicycles, all the same
    distance from the ego, so class is the only thing that differs between them."""
    x = torch.zeros(1, 3 * n_per_class, T, D)
    x[:, :, :, :8] = 1.0  # every slot valid
    for cls in range(3):
        lo, hi = cls * n_per_class, (cls + 1) * n_per_class
        x[:, lo:hi, :, 8 + cls] = 1.0
        x[:, lo:hi, -1, 0] = distance_m
    return x


def _rate_by_class(renamed: torch.Tensor, n_per_class: int = BIG_N) -> list[float]:
    flat = renamed[0]
    return [flat[c * n_per_class : (c + 1) * n_per_class].float().mean().item() for c in range(3)]


def test_per_class_overrides_hit_their_own_rate_and_leave_the_others_alone():
    torch.manual_seed(0)
    _, renamed, _ = rename_agents_to_unknown(
        _make_population(),
        0.1,
        prob_vehicle=0.02,
        prob_pedestrian=0.5,
        # bicycle deliberately unset -- it must fall back to `prob`.
    )
    veh, ped, bike = _rate_by_class(renamed)
    assert abs(veh - 0.02) < 0.01
    assert abs(ped - 0.5) < 0.03
    assert abs(bike - 0.1) < 0.02


def test_distance_scaling_raises_the_rate_for_far_agents():
    kwargs = dict(distance_scale_max=3.0, distance_scale_range_m=50.0)
    torch.manual_seed(0)
    _, near, _ = rename_agents_to_unknown(_make_population(distance_m=0.0), 0.1, **kwargs)
    torch.manual_seed(0)
    _, far, _ = rename_agents_to_unknown(_make_population(distance_m=50.0), 0.1, **kwargs)
    near_rate = near.float().mean().item()
    far_rate = far.float().mean().item()
    assert abs(near_rate - 0.1) < 0.01, near_rate  # 1x at the ego
    assert abs(far_rate - 0.3) < 0.02, far_rate  # 3x at the range limit


def test_distance_scaling_saturates_beyond_the_range():
    kwargs = dict(distance_scale_max=3.0, distance_scale_range_m=50.0)
    torch.manual_seed(0)
    _, at_limit, _ = rename_agents_to_unknown(_make_population(distance_m=50.0), 0.1, **kwargs)
    torch.manual_seed(0)
    _, way_out, _ = rename_agents_to_unknown(_make_population(distance_m=500.0), 0.1, **kwargs)
    assert torch.equal(at_limit, way_out)


def test_prob_cap_bounds_the_scaled_probability():
    torch.manual_seed(0)
    _, renamed, _ = rename_agents_to_unknown(
        _make_population(distance_m=50.0),
        0.4,
        distance_scale_max=3.0,
        distance_scale_range_m=50.0,
        prob_cap=0.5,
    )
    # Uncapped this would be 0.4 * 3 = 1.2, i.e. every agent.
    assert abs(renamed.float().mean().item() - 0.5) < 0.02


def test_already_unknown_agents_are_never_reselected():
    """Relabeling an Unknown agent is a no-op that would still inflate the reported count,
    so the rate a run logs would stop meaning what it says."""
    x = torch.zeros(1, 100, T, D)
    x[:, :, :, :8] = 1.0
    x[:, :, :, 11] = 1.0  # everything already Unknown
    _, renamed, valid = rename_agents_to_unknown(x, 1.0)
    assert valid.all()
    assert not renamed.any()


def test_all_class_probabilities_zero_is_still_a_true_noop():
    x = _make_batch()
    with patch("torch.rand") as mock_rand:
        _, renamed, valid = rename_agents_to_unknown(
            x, 0.0, prob_vehicle=0.0, prob_pedestrian=0.0, prob_bicycle=0.0
        )
    mock_rand.assert_not_called()
    assert not renamed.any()
    assert valid.any()


def test_a_single_class_override_is_enough_to_turn_the_augmentation_on():
    """prob stays 0.0, so the old early-return would have skipped this entirely."""
    torch.manual_seed(0)
    _, renamed, _ = rename_agents_to_unknown(_make_population(), 0.0, prob_pedestrian=0.5)
    veh, ped, bike = _rate_by_class(renamed)
    assert veh == 0.0 and bike == 0.0
    assert abs(ped - 0.5) < 0.03
