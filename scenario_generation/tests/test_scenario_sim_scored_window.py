"""Tests for the window a scenario_sim rollout scores.

Pure Python: no lanelet2 and no map on disk.
"""

from __future__ import annotations

from pathlib import Path


def _traj(n: int, *, goal_d) -> list[dict]:
    """A straight drive at 1 m/s, with ``goal_d`` per tick supplied by the test."""
    return [
        {
            "step": i,
            "x": float(i),
            "y": 0.0,
            "heading": 0.0,
            "speed": 1.0,
            "goal_d": float(goal_d(i)),
        }
        for i in range(n)
    ]


def _row(tmp_path: Path, traj: list[dict]) -> dict:
    from scenario_generation.scenario_sim_rollout import RolloutConfig, _finalize_row

    n = len(traj)
    return _finalize_row(
        tmp_path,
        trajectory_log=traj,
        ego_state=None,
        cfg=RolloutConfig(),
        clearances=[10.0] * n,
        collisions=[False] * n,
        scored_from=0,
        terminated_reason="goal",
        result_kind="Pass",
        coord_err=0.0,
        yaw_err=0.0,
        borders=[],
    )


def test_scored_window_ends_at_the_first_tick_the_ego_is_at_the_goal(tmp_path: Path) -> None:
    # Closes on the goal, reaches it at tick 10, then keeps driving for 20 more ticks.
    row = _row(tmp_path, _traj(30, goal_d=lambda i: max(50.0 - 5.0 * i, 0.5)))

    assert row["scored_stopped_at_goal"] is True
    assert row["scored_until_step"] == 11
    assert row["n_steps_run"] == 11


def test_a_run_that_never_reaches_the_goal_is_scored_whole(tmp_path: Path) -> None:
    row = _row(tmp_path, _traj(30, goal_d=lambda i: 100.0))

    assert row["scored_stopped_at_goal"] is False
    assert row["scored_until_step"] == 30
    assert row["n_steps_run"] == 30


def test_max_speed_is_reported_over_the_whole_drive_not_the_scored_window(
    tmp_path: Path,
) -> None:
    # The pull-away happens after the goal, so a window-only maximum would miss it.
    traj = _traj(20, goal_d=lambda i: 0.5 if i >= 2 else 50.0)
    for e in traj[10:]:
        e["speed"] = 9.0

    row = _row(tmp_path, traj)

    assert row["scored_until_step"] == 3
    assert row["max_speed_mps"] == 9.0
