"""Regression tests for optional scene-batched AWR expert retention."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from planner_metrics.config import RewardConfig
from rlvr.train_awr import (
    _expert_reward_is_hard_safe,
    _inject_expert_anchor_batch,
)


def _targets() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # B=2, K=3.  Values identify scene/group membership unambiguously.
    targets = torch.tensor([0, 1, 2, 10, 11, 12], dtype=torch.float32).view(6, 1, 1)
    weights = torch.tensor([0.4, 0.1, 0.3, 0.2, 0.6, 0.5])
    experts = torch.tensor([99, 199], dtype=torch.float32).view(2, 1, 1)
    safe = torch.tensor([True, False])
    return targets, weights, experts, safe


def test_replace_worst_is_independent_per_scene_group() -> None:
    targets, weights, experts, safe = _targets()
    output, output_weights = _inject_expert_anchor_batch(
        targets,
        weights,
        experts,
        safe,
        group_size=3,
        expert_weight=0.75,
        replace_worst=True,
    )
    assert output.flatten().tolist() == [0.0, 99.0, 2.0, 10.0, 11.0, 12.0]
    assert output_weights.tolist() == pytest.approx([0.4, 0.75, 0.3, 0.2, 0.6, 0.5])


def test_append_keeps_scene_major_order_and_zero_weights_unsafe_expert() -> None:
    targets, weights, experts, safe = _targets()
    output, output_weights = _inject_expert_anchor_batch(
        targets,
        weights,
        experts,
        safe,
        group_size=3,
        expert_weight=0.75,
        replace_worst=False,
    )
    assert output.flatten().tolist() == [0.0, 1.0, 2.0, 99.0, 10.0, 11.0, 12.0, 199.0]
    assert output_weights.tolist() == pytest.approx(
        [0.4, 0.1, 0.3, 0.75, 0.2, 0.6, 0.5, 0.0]
    )


def test_anchor_shape_mismatch_fails_closed() -> None:
    targets, weights, experts, safe = _targets()
    with pytest.raises(ValueError, match="group shape mismatch"):
        _inject_expert_anchor_batch(
            targets[:-1],
            weights[:-1],
            experts,
            safe,
            group_size=3,
            expert_weight=1.0,
            replace_worst=True,
        )


def test_logged_expert_must_pass_every_enabled_hard_gate() -> None:
    config = RewardConfig(
        rb_gate_enabled=True,
        lane_gate_enabled=True,
        static_collision_enabled=True,
        sc_gate_enabled=True,
    )
    safe = dict(
        collision_step=None,
        rb_crossing=False,
        lane_crossing=False,
        static_crossing=False,
        red_light=0.0,
        kinematic_violated=False,
    )
    assert _expert_reward_is_hard_safe(SimpleNamespace(**safe), config)
    for field, value in (
        ("collision_step", 3),
        ("rb_crossing", True),
        ("lane_crossing", True),
        ("static_crossing", True),
        ("red_light", -1.0),
        ("kinematic_violated", True),
    ):
        failed = dict(safe)
        failed[field] = value
        assert not _expert_reward_is_hard_safe(SimpleNamespace(**failed), config)
