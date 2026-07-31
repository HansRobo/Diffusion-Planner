"""The reference rule behind ``drop_constant_reward_groups``: a group whose
surviving candidates score within eps of one another (or number fewer than
two) carries no preference direction and must get zero weight -- not
exp(0)=1 -- so constant groups cannot become unweighted self-distillation."""
from types import SimpleNamespace

import pytest
import torch

from rlvr.awr import compute_awr_weights


def _rewards(totals):
    return [
        SimpleNamespace(
            total=t, collision_step=None, rb_crossing=False, lane_crossing=False,
            static_crossing=False, kinematic_violated=False, red_light=0.0,
        )
        for t in totals
    ]


def test_constant_nonzero_group_is_dropped_only_under_the_flag():
    rewards = _rewards([7.5, 7.5, 7.5, 7.5])
    kept, _ = compute_awr_weights(
        rewards, beta=0.5, weight_clip=1e9, normalize=False, min_group_std=1e-6,
    )
    dropped, _ = compute_awr_weights(
        rewards, beta=0.5, weight_clip=1e9, normalize=False, min_group_std=1e-6,
        drop_constant_reward_groups=True,
    )
    assert torch.all(kept == 1.0), "legacy behavior: exp(0)=1 everywhere"
    assert torch.all(dropped == 0.0), "reference behavior: constant group discarded"


def test_single_survivor_group_is_dropped_under_the_flag():
    rewards = _rewards([3.0, 4.0, 5.0])
    mask = [True, False, False]
    dropped, _ = compute_awr_weights(
        rewards, beta=0.5, weight_clip=1e9, normalize=False, min_group_std=1e-6,
        drop_constant_reward_groups=True, candidate_valid_mask=mask,
    )
    assert torch.all(dropped == 0.0)


def test_spread_group_weights_are_unchanged_by_the_flag():
    rewards = _rewards([1.0, 2.0, 3.0, 4.0])
    base, _ = compute_awr_weights(
        rewards, beta=0.5, weight_clip=1e9, normalize=False, min_group_std=1e-6,
    )
    flagged, _ = compute_awr_weights(
        rewards, beta=0.5, weight_clip=1e9, normalize=False, min_group_std=1e-6,
        drop_constant_reward_groups=True,
    )
    assert torch.equal(base, flagged)
    assert float(flagged.max()) > float(flagged.min()) > 0.0


def test_all_zero_group_is_subsumed():
    rewards = _rewards([0.0, 0.0, 0.0])
    dropped, _ = compute_awr_weights(
        rewards, beta=0.5, weight_clip=1e9, normalize=False, min_group_std=1e-6,
        drop_constant_reward_groups=True,
    )
    assert torch.all(dropped == 0.0)
