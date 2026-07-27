"""Regression tests for optional scene-batched AWR expert retention."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from planner_metrics.config import RewardConfig
import rlvr.awr as awr_module
from rlvr.awr import AWRRollout, AWRRolloutConfig, rollout_and_score_scene_batch
from rlvr.reward import RewardBreakdown
import rlvr.train_awr as train_awr_module
from rlvr.train_awr import (
    _expert_anchor_from_rewards,
    _expert_reward_is_hard_safe,
    _inject_expert_anchor_batch,
    _train_scene,
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


def test_collect_only_can_persist_expert_sidecar_without_injecting_anchor(
    monkeypatch,
) -> None:
    """The writer contract must not depend on this call's training weight."""

    rollout = AWRRollout(
        trajectories=torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(
            3, 4, 4
        ),
        noise_scales=torch.ones(3),
        rewards=[],
        weights=torch.tensor([0.2, 0.3, 0.5]),
        reward_data={},
        scene_encoding=torch.zeros(1, 2, 3),
        diagnostics={"valid_group": 1.0},
    )
    expert = torch.full((1, 4, 4), 7.0)
    expert_reward = torch.tensor([0.9])
    expert_safe = torch.tensor([True])
    monkeypatch.setattr(
        train_awr_module,
        "rollout_and_score_scene",
        lambda *args, **kwargs: rollout,
    )
    monkeypatch.setattr(
        train_awr_module,
        "_score_expert_anchor_batch",
        lambda *args, **kwargs: (expert, expert_reward, expert_safe),
    )

    _, collected = _train_scene(
        torch.nn.Identity(),
        torch.nn.Identity(),
        SimpleNamespace(future_len=4, predicted_neighbor_num=2),
        None,
        "scene.npz",
        AWRRolloutConfig(n_trajectories=3),
        RewardConfig(),
        {
            "expert_anchor_weight": 0.0,
            "cache_replay_expert_anchor": True,
        },
        torch.device("cpu"),
        "off",
        collect_only=True,
        data_override={"ego_current_state": torch.zeros(1, 10)},
    )

    assert collected is rollout
    assert torch.equal(collected.trajectories, rollout.trajectories)
    assert torch.equal(collected.weights, torch.tensor([0.2, 0.3, 0.5]))
    assert torch.equal(
        collected.reward_data[train_awr_module._REPLAY_EXPERT_TRAJECTORY_KEY],
        expert,
    )
    assert torch.equal(
        collected.reward_data[train_awr_module._REPLAY_EXPERT_REWARD_KEY],
        expert_reward,
    )
    assert torch.equal(
        collected.reward_data[train_awr_module._REPLAY_EXPERT_SAFE_KEY],
        expert_safe.float(),
    )


def _breakdown(total: float, **overrides) -> RewardBreakdown:
    fields = dict(
        safety=1.0,
        progress=1.0,
        smoothness=1.0,
        feasibility=1.0,
        centerline=1.0,
        red_light=0.0,
        total=total,
        collision_step=None,
        off_road_fraction=0.0,
    )
    fields.update(overrides)
    return RewardBreakdown(**fields)


def _stub_batch_rollout(monkeypatch, batch_size: int, group_size: int):
    """Drive rollout_and_score_scene_batch with recorded reward calls.

    Trajectory values are their own identity (scene-major 0..B*K-1, experts
    100*(scene+1)), so an assertion on the recorded rows pins down exactly
    which candidates each reward call saw and in what order.
    """

    trajectories = torch.arange(
        batch_size * group_size, dtype=torch.float32
    ).view(batch_size, group_size, 1, 1)
    monkeypatch.setattr(
        awr_module,
        "sample_unguided_dp_group_batch",
        lambda *args, **kwargs: (
            trajectories.clone(),
            torch.zeros(batch_size, group_size),
            torch.zeros(batch_size, 1, 1),
        ),
    )
    monkeypatch.setattr(
        awr_module,
        "_first_waypoint_gate_and_diagnostics_batch",
        lambda *args, **kwargs: (
            torch.ones(batch_size, group_size, dtype=torch.bool),
            [{} for _ in range(batch_size)],
        ),
    )
    scored_rows: list[list[float]] = []

    def fake_reward(trajs, data, config):
        rows = [float(row.flatten()[0]) for row in trajs]
        scored_rows.append(rows)
        return [_breakdown(value) for value in rows]

    monkeypatch.setattr(awr_module, "compute_reward_batch", fake_reward)
    return trajectories, scored_rows


def test_merged_expert_is_scored_as_a_trailing_row_and_left_out_of_the_group(
    monkeypatch,
) -> None:
    batch_size, group_size = 2, 3
    _, scored_rows = _stub_batch_rollout(monkeypatch, batch_size, group_size)
    experts = torch.tensor([100.0, 200.0]).view(batch_size, 1, 1)
    expert_rewards: list[RewardBreakdown] = []

    rollouts = rollout_and_score_scene_batch(
        torch.nn.Identity(),
        SimpleNamespace(future_len=1),
        {"ego_current_state": torch.zeros(batch_size, 10)},
        AWRRolloutConfig(n_trajectories=group_size),
        RewardConfig(),
        torch.device("cpu"),
        expert_trajectories=experts,
        expert_rewards_out=expert_rewards,
    )

    # One reward call per scene, each carrying K candidates plus the expert
    # last -- not a second call per scene.
    assert scored_rows == [[0.0, 1.0, 2.0, 100.0], [3.0, 4.0, 5.0, 200.0]]
    # The expert is reported separately and never joins the weighted group,
    # which would otherwise hand AWR a candidate the model cannot produce.
    assert [reward.total for reward in expert_rewards] == [100.0, 200.0]
    assert [[r.total for r in rollout.rewards] for rollout in rollouts] == [
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
    ]
    assert all(len(rollout.weights) == group_size for rollout in rollouts)


def test_unmerged_rollout_scores_only_the_candidates(monkeypatch) -> None:
    batch_size, group_size = 2, 3
    _, scored_rows = _stub_batch_rollout(monkeypatch, batch_size, group_size)

    rollouts = rollout_and_score_scene_batch(
        torch.nn.Identity(),
        SimpleNamespace(future_len=1),
        {"ego_current_state": torch.zeros(batch_size, 10)},
        AWRRolloutConfig(n_trajectories=group_size),
        RewardConfig(),
        torch.device("cpu"),
    )

    assert scored_rows == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    assert [[r.total for r in rollout.rewards] for rollout in rollouts] == [
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
    ]


def test_expert_kwargs_must_be_supplied_together(monkeypatch) -> None:
    _stub_batch_rollout(monkeypatch, 2, 3)
    for kwargs in (
        {"expert_trajectories": torch.zeros(2, 1, 1)},
        {"expert_rewards_out": []},
    ):
        with pytest.raises(ValueError, match="must be supplied"):
            rollout_and_score_scene_batch(
                torch.nn.Identity(),
                SimpleNamespace(future_len=1),
                {"ego_current_state": torch.zeros(2, 10)},
                AWRRolloutConfig(n_trajectories=3),
                RewardConfig(),
                torch.device("cpu"),
                **kwargs,
            )


def test_merged_expert_shape_mismatch_fails_closed(monkeypatch) -> None:
    _stub_batch_rollout(monkeypatch, 2, 3)
    with pytest.raises(ValueError, match="one candidate per scene"):
        rollout_and_score_scene_batch(
            torch.nn.Identity(),
            SimpleNamespace(future_len=1),
            {"ego_current_state": torch.zeros(2, 10)},
            AWRRolloutConfig(n_trajectories=3),
            RewardConfig(),
            torch.device("cpu"),
            expert_trajectories=torch.zeros(2, 4, 4),
            expert_rewards_out=[],
        )


def test_expert_gates_are_applied_to_merged_breakdowns() -> None:
    """The merged route must gate identically to the standalone one."""

    config = RewardConfig(rb_gate_enabled=True)
    experts = torch.zeros(3, 1, 1)
    _, rewards, safe = _expert_anchor_from_rewards(
        experts,
        [
            _breakdown(0.8),
            _breakdown(0.5, collision_step=2),
            _breakdown(0.7, rb_crossing=True),
        ],
        config,
    )
    assert rewards.tolist() == pytest.approx([0.8, 0.5, 0.7])
    assert safe.tolist() == [True, False, False]


def test_expert_anchor_requires_one_scored_expert_per_scene() -> None:
    with pytest.raises(ValueError, match="one scored expert per scene"):
        _expert_anchor_from_rewards(
            torch.zeros(3, 1, 1), [_breakdown(0.8)], RewardConfig()
        )
