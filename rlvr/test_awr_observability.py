import pytest

from rlvr.awr import _add_rollout_reward_diagnostics
from rlvr.reward import RewardBreakdown
from rlvr.train_awr import _wandb_metric_path


def _reward(total: float, **kwargs) -> RewardBreakdown:
    return RewardBreakdown(
        safety=0.0,
        progress=0.0,
        smoothness=0.0,
        feasibility=0.0,
        centerline=0.0,
        red_light=0.0,
        total=total,
        collision_step=kwargs.get("collision_step"),
        off_road_fraction=0.0,
        rb_crossing=kwargs.get("rb_crossing", False),
        kinematic_violated=kwargs.get("kinematic_violated", False),
    )


def test_all_zero_reward_group_is_distinguished_from_equal_nonzero_group() -> None:
    zero = {}
    _add_rollout_reward_diagnostics(
        zero, [_reward(0.0, collision_step=3), _reward(0.0)]
    )
    assert zero["all_zero_reward_group"] == 1.0
    assert zero["all_equal_reward_group"] == 1.0
    assert zero["zero_reward_candidate_fraction"] == 1.0
    assert zero["det_zero_collision"] == 1.0

    tied = {}
    _add_rollout_reward_diagnostics(tied, [_reward(0.7), _reward(0.7)])
    assert tied["all_zero_reward_group"] == 0.0
    assert tied["all_equal_reward_group"] == 1.0
    assert tied["zero_reward_candidate_fraction"] == 0.0


def test_reward_group_reports_useful_rank_signal() -> None:
    diagnostics = {}
    _add_rollout_reward_diagnostics(
        diagnostics, [_reward(0.0), _reward(0.4), _reward(0.8)]
    )
    assert diagnostics["all_equal_reward_group"] == 0.0
    assert diagnostics["reward_unique_count"] == 3.0
    assert diagnostics["best_vs_det_reward_gain"] == pytest.approx(0.8)
    assert diagnostics["zero_reward_candidate_fraction"] == pytest.approx(1.0 / 3.0)


def test_eval_wandb_metrics_are_grouped_and_noisy_percentiles_are_hidden() -> None:
    assert (
        _wandb_metric_path("eval", "mean_det_reward")
        == "eval_01_reward/mean_det_reward"
    )
    assert (
        _wandb_metric_path("eval", "mean_det_collision")
        == "eval_02_safety_gates/mean_det_collision"
    )
    assert (
        _wandb_metric_path("eval", "mean_det_lane_crossing")
        == "eval_03_lane_road/mean_det_lane_crossing"
    )
    assert (
        _wandb_metric_path("eval", "p90_fde")
        == "eval_05_imitation/p90_fde"
    )
    assert (
        _wandb_metric_path("eval", "mean_candidate_pairwise_ade")
        == "eval_06_multimodality/mean_candidate_pairwise_ade"
    )
    assert _wandb_metric_path("eval", "p10_det_reward") is None
    assert _wandb_metric_path("eval", "p50_det_collision") is None


def test_train_wandb_metrics_are_grouped_by_purpose() -> None:
    assert (
        _wandb_metric_path("train", "mean_loss")
        == "train_01_optimization/mean_loss"
    )
    assert (
        _wandb_metric_path("train", "mean_effective_sample_size")
        == "train_02_awr_signal/mean_effective_sample_size"
    )
    assert (
        _wandb_metric_path("train", "scenes_per_sec")
        == "train_05_system/scenes_per_sec"
    )
