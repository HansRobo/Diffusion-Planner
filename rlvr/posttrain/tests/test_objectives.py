import numpy as np
import pytest
import torch

from rlvr.posttrain.objectives import available, register, resolve
from rlvr.reward import RewardBreakdown


def _reward(total: float, **overrides) -> RewardBreakdown:
    fields = dict(
        safety=0.0,
        progress=0.0,
        smoothness=0.0,
        feasibility=0.0,
        centerline=0.0,
        red_light=0.0,
        total=total,
        collision_step=None,
        off_road_fraction=0.0,
    )
    fields.update(overrides)
    return RewardBreakdown(**fields)


def test_awr_is_the_registered_default_and_unchanged():
    from rlvr.awr import compute_awr_weights

    assert resolve("awr") is compute_awr_weights
    assert {"awr", "grpo", "dpo"} <= set(available())


def test_unknown_objective_names_its_alternatives():
    with pytest.raises(ValueError, match="grpo"):
        resolve("ppo")


def test_registering_a_new_objective_needs_nothing_else():
    register("constant", lambda rewards, **_: (torch.ones(len(rewards)), {}))
    weights, _ = resolve("constant")([_reward(1.0), _reward(2.0)])
    assert weights.tolist() == [1.0, 1.0]


def test_grpo_keeps_the_sign_of_the_advantage():
    weights, diag = resolve("grpo")([_reward(0.0), _reward(1.0), _reward(2.0)])
    assert weights[0] < 0 < weights[2]
    assert abs(float(weights[1])) < 1e-6
    assert diag["negative_weight_count"] == 1.0


def test_grpo_positive_only_clamps_at_the_group_mean():
    weights, _ = resolve("grpo")(
        [_reward(0.0), _reward(1.0), _reward(2.0)], positive_advantage_only=True
    )
    assert weights[0] == 0.0 and weights[1] == 0.0 and weights[2] > 0.0


def test_grpo_is_zero_without_a_preference_direction():
    weights, diag = resolve("grpo")([_reward(1.0)] * 4)
    assert weights.abs().sum() == 0.0
    assert diag["valid_group"] == 1.0


def test_dpo_weights_only_the_best_and_worst_candidate():
    weights, diag = resolve("dpo")(
        [_reward(0.0), _reward(1.0), _reward(0.5), _reward(2.0)], beta=0.5
    )
    assert diag["dpo_pair_emitted"] == 1.0
    assert weights[3] > 0.0 and weights[0] < 0.0
    assert weights[1] == 0.0 and weights[2] == 0.0
    assert float(weights[3] / -weights[0]) == pytest.approx(2.0)


def test_dpo_emits_nothing_when_every_candidate_ties():
    weights, diag = resolve("dpo")([_reward(1.0)] * 3)
    assert weights.abs().sum() == 0.0
    assert diag["dpo_pair_emitted"] == 0.0


def test_dpo_positive_only_requires_beating_candidate_zero():
    rewards = [_reward(2.0), _reward(1.0), _reward(0.0)]
    weights, diag = resolve("dpo")(rewards, positive_advantage_only=True)
    assert weights.abs().sum() == 0.0 and diag["dpo_pair_emitted"] == 0.0


def test_invalid_candidates_are_excluded_by_the_mask():
    rewards = [_reward(0.0), _reward(9.0), _reward(1.0)]
    weights, diag = resolve("grpo")(
        rewards, candidate_valid_mask=[True, False, True]
    )
    assert weights[1] == 0.0
    assert diag["first_waypoint_invalid_candidate_count"] == 1.0


def test_safe_only_drops_colliding_candidates():
    rewards = [_reward(0.0), _reward(5.0, collision_step=3), _reward(1.0)]
    weights, diag = resolve("grpo")(rewards, safe_only=True)
    assert weights[1] == 0.0
    assert diag["safe_candidate_count"] == 2.0


def test_drop_all_zero_groups_zeroes_the_group():
    weights, diag = resolve("dpo")([_reward(0.0)] * 3, drop_all_zero_groups=True)
    assert weights.abs().sum() == 0.0
    assert diag["dropped_all_zero_group"] == 1.0


@pytest.mark.parametrize("name", ["grpo", "dpo"])
def test_diagnostic_keys_are_identical_across_groups(name):
    """The trainer reduces diagnostics across DDP ranks; a rank-dependent key
    set desynchronises that reduce, so the key set must not depend on inputs."""

    fn = resolve(name)
    groups = [
        [_reward(0.0), _reward(1.0)],
        [_reward(1.0)] * 3,
        [_reward(0.0), _reward(np.nan)],
        [_reward(0.0, collision_step=1), _reward(2.0)],
    ]
    keys = {frozenset(fn(group)[1]) for group in groups}
    assert len(keys) == 1
