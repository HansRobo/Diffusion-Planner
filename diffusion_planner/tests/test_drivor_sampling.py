"""Tests for the three trajectory samplings the DrivoR head lives between.

Both shipped configurations are covered: the default 40 poses at 0.1 s (which
lands exactly on the dataset's and the scorer's own grid, so every conversion
degenerates to a slice) and DrivoR's upstream 8 poses at 0.5 s (which needs real
interpolation on the way to the scorer).

The up-sampling is checked against a numpy transcription of nuPlan's
``InterpolatedTrajectory`` (scipy ``interp1d`` on x/y, ``np.unwrap`` ->
``interp1d`` -> ``principal_value`` on the heading), which is what navsim's
``transform_trajectory`` + ``get_trajectory_as_array`` pair actually runs -- the
same "compare against the reference, don't assert my own arithmetic" standard the
oracle tests use.
"""

import numpy as np
import pytest
import torch
from scipy.interpolate import interp1d

from diffusion_planner.utils.drivor_sampling import (
    DATASET_POSE_DT,
    expert_future_slice,
    resample_expert_future,
    scoring_horizon_slice,
    upsample_poses,
)

# The shipped configuration: navsim's 4 s horizon at the dataset's own 10 Hz.
NUM_POSES = 40
POSE_DT = 0.1
# DrivoR's upstream configuration, still supported: the same 4 s, 8 coarse poses.
COARSE_POSES = 8  # ``drivoR.yaml``'s ``num_poses``
COARSE_DT = 0.5  # ``t4_training.yaml``'s ``t4_trajectory_dt_s``
SCORING_STEPS = 40  # navsim's ``proposal_sampling.num_poses``
SCORING_DT = 0.1  # navsim's ``proposal_sampling.interval_length``
STORED = 80  # ``ego_agent_future`` rows


def _reference_upsample(
    poses: np.ndarray, num_steps: int, pose_dt: float, step_dt: float
) -> np.ndarray:
    """nuPlan's ``InterpolatedTrajectory`` over ``[ego origin] + poses``."""
    num_poses = poses.shape[0]
    anchor_t = np.arange(num_poses + 1) * pose_dt
    xy = np.concatenate((np.zeros((1, 2)), poses[:, :2]), axis=0)
    heading = np.arctan2(poses[:, 3], poses[:, 2])
    angular = np.unwrap(np.concatenate(([0.0], heading)))

    # ``get_trajectory_as_array`` clips the sample times into the trajectory's
    # own span, which is what keeps the last stamp inside interp1d's range.
    stamps = np.clip(np.arange(1, num_steps + 1) * step_dt, anchor_t[0], anchor_t[-1])
    out_xy = interp1d(anchor_t, xy, axis=0)(stamps)
    # ``AngularInterpolator.interpolate`` -> ``principal_value(min_=-pi)``.
    out_h = ((interp1d(anchor_t, angular, axis=0)(stamps) + np.pi) % (2.0 * np.pi)) - np.pi
    return np.concatenate(
        (out_xy, np.cos(out_h)[:, None], np.sin(out_h)[:, None]), axis=-1
    )


def _random_poses(rng: np.random.Generator, num_poses: int = COARSE_POSES) -> np.ndarray:
    """A plausible 4 s ego trajectory: forward motion with a drifting heading."""
    scale = 0.25 * COARSE_POSES / num_poses
    heading = np.cumsum(rng.normal(scale=scale, size=num_poses))
    step = rng.uniform(2.0, 7.0, size=num_poses) * COARSE_POSES / num_poses
    xy = np.cumsum(np.stack([step * np.cos(heading), step * np.sin(heading)], axis=-1), axis=0)
    return np.concatenate((xy, np.cos(heading)[:, None], np.sin(heading)[:, None]), axis=-1)


# --------------------------------------------------------------------------
# expert target: sub-sampling the stored future
# --------------------------------------------------------------------------
def test_expert_future_slice_is_the_documented_grid():
    # The default emits on the dataset's own grid, so the target is the first 4 s
    # verbatim -- no sub-sampling at all.
    assert expert_future_slice(NUM_POSES, POSE_DT, STORED) == slice(0, 40, 1)
    # DrivoR's coarse grid takes every fifth row, starting at row 4 (t = 0.5 s).
    assert expert_future_slice(COARSE_POSES, COARSE_DT, STORED) == slice(4, 40, 5)


@pytest.mark.parametrize(
    "num_poses, pose_dt", [(NUM_POSES, POSE_DT), (COARSE_POSES, COARSE_DT)]
)
def test_expert_future_slice_lands_on_the_wanted_stamps(num_poses, pose_dt):
    """Row ``i`` is ``t = (i + 1) * 0.1``, so the rows must be the wanted stamps."""
    rows = np.arange(STORED)[expert_future_slice(num_poses, pose_dt, STORED)]
    stamps = (rows + 1) * DATASET_POSE_DT
    np.testing.assert_allclose(stamps, np.arange(1, num_poses + 1) * pose_dt)
    # Both configurations cover navsim's horizon exactly.
    assert stamps[-1] == pytest.approx(SCORING_STEPS * SCORING_DT)


def test_resample_expert_future_takes_stored_rows_verbatim():
    """Sub-sampling, not interpolation: the values must be untouched."""
    future = torch.arange(STORED, dtype=torch.float32)[None, :, None].repeat(2, 1, 4)

    dense = resample_expert_future(future, NUM_POSES, POSE_DT)
    assert dense.shape == (2, NUM_POSES, 4)
    torch.testing.assert_close(dense[0, :, 0], future[0, :NUM_POSES, 0])

    coarse = resample_expert_future(future, COARSE_POSES, COARSE_DT)
    assert coarse.shape == (2, COARSE_POSES, 4)
    expected = torch.tensor([4.0, 9.0, 14.0, 19.0, 24.0, 29.0, 34.0, 39.0])
    torch.testing.assert_close(coarse[0, :, 0], expected)


def test_resample_expert_future_is_a_view():
    future = torch.zeros(2, STORED, 4)
    assert resample_expert_future(future, NUM_POSES, POSE_DT)._base is future
    assert resample_expert_future(future, COARSE_POSES, COARSE_DT)._base is future


@pytest.mark.parametrize("pose_dt", [0.35, 0.15, 0.0])
def test_expert_future_slice_rejects_non_multiples(pose_dt):
    with pytest.raises(ValueError, match="integer multiple"):
        expert_future_slice(COARSE_POSES, pose_dt, STORED)


@pytest.mark.parametrize("num_poses, pose_dt", [(100, POSE_DT), (20, COARSE_DT)])
def test_expert_future_slice_rejects_a_horizon_the_dataset_cannot_cover(num_poses, pose_dt):
    with pytest.raises(ValueError, match="stored rows"):
        expert_future_slice(num_poses, pose_dt, STORED)  # 10 s out of 8 s


def test_scoring_horizon_slice_clips_to_the_scorer_horizon():
    assert scoring_horizon_slice(SCORING_STEPS, STORED) == slice(0, SCORING_STEPS)
    rows = np.arange(STORED)[scoring_horizon_slice(SCORING_STEPS, STORED)]
    stamps = (rows + 1) * DATASET_POSE_DT
    # Same 4 s as the head's own horizon, at the scorer's 0.1 s.
    np.testing.assert_allclose(stamps, np.arange(1, SCORING_STEPS + 1) * SCORING_DT)
    assert stamps[-1] == pytest.approx(NUM_POSES * POSE_DT)


def test_scoring_horizon_slice_rejects_more_steps_than_stored():
    with pytest.raises(ValueError, match="dataset stores"):
        scoring_horizon_slice(STORED + 1, STORED)


# --------------------------------------------------------------------------
# scoring grid: up-sampling the head's poses
# --------------------------------------------------------------------------
def test_upsample_matches_the_nuplan_interpolator():
    rng = np.random.default_rng(0)
    for _ in range(32):
        poses = _random_poses(rng)
        got = upsample_poses(
            torch.as_tensor(poses, dtype=torch.float64), SCORING_STEPS, COARSE_DT, SCORING_DT
        )
        want = _reference_upsample(poses, SCORING_STEPS, COARSE_DT, SCORING_DT)
        np.testing.assert_allclose(got.numpy(), want, atol=1e-12)


def test_upsample_matches_the_reference_across_the_branch_cut():
    """Headings straddling +-pi: unwrap before interpolating or the blend goes
    the long way round and the interpolated pose points backwards."""
    heading = np.array([3.0, 3.10, -3.10, -3.0, -2.9, 3.0, 3.13, -3.13])
    xy = np.cumsum(np.stack([np.cos(heading), np.sin(heading)], axis=-1) * 4.0, axis=0)
    poses = np.concatenate(
        (xy, np.cos(heading)[:, None], np.sin(heading)[:, None]), axis=-1
    )
    got = upsample_poses(
        torch.as_tensor(poses, dtype=torch.float64), SCORING_STEPS, COARSE_DT, SCORING_DT
    )
    want = _reference_upsample(poses, SCORING_STEPS, COARSE_DT, SCORING_DT)
    np.testing.assert_allclose(got.numpy(), want, atol=1e-12)


def test_upsample_is_the_identity_on_the_default_grid():
    """At the shipped density the head already emits the scorer's own stamps, so
    the conversion must not perturb a single pose -- that is what lets the oracle
    short-circuit it entirely."""
    rng = np.random.default_rng(3)
    poses = torch.as_tensor(_random_poses(rng, NUM_POSES), dtype=torch.float64)
    got = upsample_poses(poses, SCORING_STEPS, POSE_DT, SCORING_DT)
    torch.testing.assert_close(got, poses, atol=1e-12, rtol=0.0)


def test_upsample_reproduces_the_input_poses_at_their_own_stamps():
    rng = np.random.default_rng(7)
    poses = torch.as_tensor(_random_poses(rng), dtype=torch.float64)
    dense = upsample_poses(poses, SCORING_STEPS, COARSE_DT, SCORING_DT)
    stride = int(round(COARSE_DT / SCORING_DT))
    torch.testing.assert_close(dense[stride - 1 :: stride], poses)


def test_upsample_anchors_on_the_ego_origin():
    """The first output stamp is 0.1 s into a 0.5 s segment that starts at the
    ego's current pose, i.e. one fifth of the way to the first predicted pose.
    Without that anchor t = 0.1 .. 0.4 would have to be extrapolated."""
    poses = torch.zeros(1, COARSE_POSES, 4, dtype=torch.float64)
    poses[..., 0] = torch.arange(1, COARSE_POSES + 1, dtype=torch.float64) * 5.0
    poses[..., 2] = 1.0
    dense = upsample_poses(poses, SCORING_STEPS, COARSE_DT, SCORING_DT)
    assert dense[0, 0, 0] == pytest.approx(1.0)  # 0.2 * 5.0
    torch.testing.assert_close(
        dense[0, :, 0], torch.arange(1, SCORING_STEPS + 1, dtype=torch.float64)
    )


def test_upsample_keeps_leading_dimensions():
    poses = torch.zeros(3, 6, COARSE_POSES, 4, dtype=torch.float32)
    poses[..., 2] = 1.0
    assert upsample_poses(poses, SCORING_STEPS, COARSE_DT, SCORING_DT).shape == (
        3,
        6,
        SCORING_STEPS,
        4,
    )


def test_upsample_rejects_a_horizon_mismatch():
    poses = torch.zeros(1, COARSE_POSES, 4)
    with pytest.raises(ValueError, match="horizon mismatch"):
        upsample_poses(poses, 80, COARSE_DT, SCORING_DT)  # 8 s of output from 4 s of input


def test_upsample_rejects_three_column_poses():
    with pytest.raises(ValueError, match=r"\(x, y, cos, sin\)"):
        upsample_poses(torch.zeros(1, COARSE_POSES, 3), SCORING_STEPS, COARSE_DT, SCORING_DT)


# --------------------------------------------------------------------------
# the shipped defaults
# --------------------------------------------------------------------------
def test_shipped_defaults_need_no_conversion_anywhere():
    """The point of the 40 @ 0.1 s default: expert target, neighbour futures and
    proposals all live on one grid, so nothing is sub-sampled or interpolated."""
    import dataclasses

    from diffusion_planner.train_config import TrainConfig

    # Read the declared defaults: ``TrainConfig`` has required fields (paths, the
    # experiment name) that a test has no business inventing.
    shipped = {
        field.name: field.default
        for field in dataclasses.fields(TrainConfig)
        if field.default is not dataclasses.MISSING
    }

    horizon = shipped["drivor_num_poses"] * shipped["drivor_pose_dt"]
    assert horizon == pytest.approx(
        shipped["drivor_scoring_num_poses"] * shipped["drivor_oracle_dt"]
    )
    assert horizon == pytest.approx(SCORING_STEPS * SCORING_DT)  # navsim's 4 s
    assert shipped["drivor_pose_dt"] == pytest.approx(DATASET_POSE_DT)
    assert expert_future_slice(
        shipped["drivor_num_poses"], shipped["drivor_pose_dt"], STORED
    ) == slice(0, shipped["drivor_num_poses"], 1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
