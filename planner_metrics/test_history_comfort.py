"""The devkit's ``history_comfort`` prefix: the recorded past, not just the rollout.

``navsim_score.py::_history_comfort`` is not navsim's ``COMFORTABLE``. It
finite-differences the ego's *recorded* past, drops its last pose -- the simulated
array's row 0 already is the current state -- prepends that to the simulated
future and requires all six bounds over the concatenation. navsim scores the
rollout alone. These tests pin the difference, including the consequence that
matters for a scorer head: the prefix is shared by every proposal in a scene, so
a rough past gates the whole scene to 0.
"""

from __future__ import annotations

import numpy as np
import torch

from planner_metrics import pdms_navsim as ns
from planner_metrics.pdms_proxy import pdms_proxy

DT = 0.1
HORIZON = 40
HISTORY = 31  # the dataset's ego_agent_past: 3.0 s of past plus the current pose


def _straight_past(speed: float, batch: int = 1) -> np.ndarray:
    """Constant-speed straight past ending exactly at the ego, as in the NPZ."""
    t = (np.arange(HISTORY) - (HISTORY - 1)) * DT  # ..., -0.2, -0.1, 0.0
    past = np.zeros((HISTORY, 4))
    past[:, 0] = speed * t
    past[:, 2] = 1.0
    return np.tile(past, (batch, 1, 1))


def _straight_proposal(speed: float, batch: int = 1) -> np.ndarray:
    t = np.arange(1, HORIZON + 1) * DT
    poses = np.zeros((HORIZON, 4))
    poses[:, 0] = speed * t
    poses[:, 2] = 1.0
    return np.tile(poses, (batch, 1, 1))


def _ego(speed: float, batch: int = 1) -> np.ndarray:
    ego = np.zeros((batch, 10))
    ego[:, 2] = 1.0  # cos(heading)
    ego[:, 4] = speed
    return ego


def test_history_prefix_drops_the_current_pose():
    past = _straight_past(6.0)
    prefix = ns.history_states_from_poses(past, DT)
    assert prefix.shape == (1, HISTORY - 1, ns.STATE_SIZE)
    # It is the finite-difference path applied to the past minus its last row --
    # the past already happened, so there is no tracker to run over it.
    expected = ns.states_from_poses(past[:, :-1], DT)
    np.testing.assert_array_equal(prefix, expected)


def test_history_prefix_is_empty_when_there_is_no_past():
    for rows in (0, 1):
        prefix = ns.history_states_from_poses(np.zeros((2, rows, 4)), DT)
        assert prefix.shape == (2, 0, ns.STATE_SIZE)


def test_comfort_score_with_history_is_the_concatenation():
    speed = 8.0
    poses, past, ego = _straight_proposal(speed), _straight_past(speed), _ego(speed)
    combined = np.concatenate(
        (
            ns.history_states_from_poses(past, DT),
            ns.simulated_states_from_poses(poses, DT, ego, 2.8),
        ),
        axis=1,
    )
    np.testing.assert_array_equal(
        ns.comfort_score(poses, DT, ego, 2.8, history_poses=past),
        ns.comfort_score_from_states(combined, DT),
    )


def test_comfort_score_without_history_scores_the_rollout_alone():
    """Omitting the history must leave navsim's ``COMFORTABLE`` untouched."""
    speed = 8.0
    poses, ego = _straight_proposal(speed), _ego(speed)
    np.testing.assert_array_equal(
        ns.comfort_score(poses, DT, ego, 2.8),
        ns.comfort_score_from_states(ns.simulated_states_from_poses(poses, DT, ego, 2.8), DT),
    )


def test_history_gates_the_scene_for_every_proposal():
    """A rough past zeroes comfort for proposals that are themselves smooth.

    This is the property that makes ``history_comfort`` a per-scene gate rather
    than a per-proposal score: the prefix is identical across the proposal axis,
    so no trajectory can score its way out of it.
    """
    speed = 8.0
    ego = _ego(speed)
    proposals = _straight_proposal(speed, batch=4)

    smooth = np.repeat(_straight_past(speed), 4, axis=0)
    assert ns.comfort_score(proposals, DT, ego, 2.8, history_poses=smooth).tolist() == [
        1.0,
        1.0,
        1.0,
        1.0,
    ]

    # 0.5 m of lateral zig-zag in the past; the last pose is left alone because
    # it is dropped, so this is purely the prefix.
    rough = smooth.copy()
    rough[:, :-1:2, 1] += 0.5
    assert ns.comfort_score(proposals, DT, ego, 2.8, history_poses=rough).tolist() == [
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    # ... while the proposals alone are perfectly comfortable.
    assert ns.comfort_score(proposals, DT, ego, 2.8).tolist() == [1.0, 1.0, 1.0, 1.0]


def test_one_history_broadcasts_over_the_proposal_axis():
    """``[P, T, 4]`` proposals against a single ``[1, H, 4]`` scene history."""
    speed = 6.0
    ego = _ego(speed)
    proposals = _straight_proposal(speed, batch=8)
    past = _straight_past(speed)  # one scene
    broadcast = ns.comfort_score(proposals, DT, ego, 2.8, history_poses=past)
    repeated = ns.comfort_score(proposals, DT, ego, 2.8, history_poses=np.repeat(past, 8, axis=0))
    np.testing.assert_array_equal(broadcast, repeated)


def test_proxy_history_comfort_matches_the_reference():
    """``pdms_proxy(ego_history_poses=...)`` is ``comfort_score(history_poses=...)``."""
    speed = 8.0
    ego = _ego(speed)
    poses = _straight_proposal(speed)
    pred = torch.as_tensor(poses, dtype=torch.float32)  # (x, y, cos, sin)
    gt = pred.clone()

    smooth = _straight_past(speed)
    rough = smooth.copy()
    rough[:, :-1:2, 1] += 0.5

    for history in (None, smooth, rough):
        out = pdms_proxy(pred, gt, ego_current_state=ego, ego_history_poses=history, dt=DT)
        # No wheel_base reaches the proxy, so the reference must use the same
        # pacifica default the rollout falls back to.
        want = ns.comfort_score(poses, DT, ego, None, history_poses=history)
        assert out["history_comfort"].reshape(-1).tolist() == want.reshape(-1).tolist()

    # The three cases must not be degenerate: the prefix has to change the answer.
    assert pdms_proxy(pred, gt, ego_current_state=ego, ego_history_poses=rough, dt=DT)[
        "history_comfort"
    ].reshape(-1).tolist() == [0.0]
