"""The plan-contract guard: a head whose plan does not match how the rollout executes it must
fail at startup rather than silently change what is measured."""

from types import SimpleNamespace

import pytest

from scenario_generation.reproducer_rollout import _assert_plan_contract


def _drivor(num_poses=40, pose_dt=0.1):
    # SimpleNamespace, NOT MagicMock: a MagicMock auto-creates predictor_head, which would
    # silently take the diffusion branch and make every assertion below vacuous.
    return SimpleNamespace(
        predictor_head="drivor", drivor_num_poses=num_poses, drivor_pose_dt=pose_dt, future_len=80
    )


def test_diffusion_head_reports_future_len():
    assert _assert_plan_contract(SimpleNamespace(future_len=80), 4, "mpc") == 80


def test_drivor_head_reports_its_own_horizon_not_future_len():
    assert _assert_plan_contract(_drivor(), 4, "perfect") == 40


def test_drivor_pose_dt_defaults_to_dt_when_absent():
    args = SimpleNamespace(predictor_head="drivor", drivor_num_poses=40, future_len=80)
    assert _assert_plan_contract(args, 4, "perfect") == 40


def test_pose_dt_other_than_the_sim_step_is_rejected():
    with pytest.raises(ValueError, match="pose_dt"):
        _assert_plan_contract(_drivor(num_poses=8, pose_dt=0.5), 1, "perfect")


@pytest.mark.parametrize("interval", [0, 41])
def test_replan_interval_outside_the_horizon_is_rejected(interval):
    with pytest.raises(ValueError, match="replan_interval"):
        _assert_plan_contract(_drivor(), interval, "perfect")


def test_replan_interval_equal_to_the_horizon_is_allowed():
    assert _assert_plan_contract(_drivor(), 40, "perfect") == 40


def test_mpc_rejects_a_plan_shorter_than_its_preview_window():
    from scenario_generation.mpc_tracker import MPC_HORIZON_STEPS

    with pytest.raises(ValueError, match="mpc"):
        _assert_plan_contract(_drivor(num_poses=MPC_HORIZON_STEPS - 1), 1, "mpc")


def test_perfect_tracker_allows_a_short_plan():
    assert _assert_plan_contract(_drivor(num_poses=8), 1, "perfect") == 8
