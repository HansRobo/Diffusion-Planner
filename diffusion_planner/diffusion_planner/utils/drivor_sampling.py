"""Trajectory sampling between the dataset, the DrivoR head and the PDM scorer.

Three samplings meet in this head:

* the **dataset** stores 80 future ego poses at 0.1 s -- an 8 s horizon.  There is
  no ``t=0`` row: row ``i`` is ``t = (i + 1) * 0.1``, verified against the
  recorded ego speed on moving samples rather than assumed.
* the **head** emits its own trajectory over navsim's 4 s horizon.  The horizon is
  fixed by the scorer; the density inside it is a deployment choice, because a
  one-shot head pays nothing for extra poses.  Default: 40 poses at 0.1 s, i.e.
  the dataset's own rate, so a downstream controller gets 10 Hz waypoints
  directly.  DrivoR upstream instead uses ``num_poses: 8`` (``drivoR.yaml``) at
  ``t4_trajectory_dt_s: 0.5`` (``t4_training.yaml``).
* the **PDM scorer** works at ``num_poses: 40, interval_length: 0.1``
  (``pdm_scoring/default_scoring_parameters.yaml``) -- the same 4 s at 10 Hz.
  DrivoR reaches it via ``transform_trajectory`` + ``get_trajectory_as_array``
  (``score_module/compute_navsim_score.py``), which builds an
  ``InterpolatedTrajectory`` over ``[initial_ego_state] + poses`` and samples it:
  linear in x/y, and unwrap -> linear -> wrap for the heading (nuPlan's
  ``InterpolatedTrajectory`` + ``AngularInterpolator``).  At the default density
  the head is already on this grid, so :func:`upsample_poses` is a no-op; it is
  what makes the coarse 8-pose configuration work unchanged.

This module is the only place that converts between the three, so the pose count
never has to be re-derived at a call site.
"""

from __future__ import annotations

import math

import torch

# ``ego_agent_future`` / ``neighbor_agents_future`` step in the shards.
DATASET_POSE_DT = 0.1

_TOL = 1e-6


def expert_future_slice(
    num_poses: int,
    pose_dt: float,
    dataset_len: int,
    dataset_dt: float = DATASET_POSE_DT,
) -> slice:
    """Dataset rows holding ``t = pose_dt, 2 * pose_dt, ..., num_poses * pose_dt``.

    Row ``i`` of the stored future is ``t = (i + 1) * dataset_dt`` -- there is no
    ``t=0`` row -- so ``t = k * pose_dt`` lives at index
    ``k * pose_dt / dataset_dt - 1``.  With the defaults (8 poses at 0.5 s out of
    80 rows at 0.1 s) that is ``slice(4, 40, 5)`` -> rows 4, 9, ..., 39.

    A ``slice`` rather than an index tensor so the caller gets a view.
    """
    ratio = pose_dt / dataset_dt
    step = int(round(ratio))
    if step < 1 or abs(ratio - step) > _TOL:
        raise ValueError(
            f"output step {pose_dt} s is not an integer multiple of the dataset's "
            f"{dataset_dt} s sampling"
        )
    stop = num_poses * step
    if stop > dataset_len:
        raise ValueError(
            f"{num_poses} poses at {pose_dt} s need {stop} stored rows "
            f"({num_poses * pose_dt} s) but the dataset only has {dataset_len} "
            f"({dataset_len * dataset_dt} s)"
        )
    return slice(step - 1, stop, step)


def resample_expert_future(
    ego_future: torch.Tensor,
    num_poses: int,
    pose_dt: float,
    dataset_dt: float = DATASET_POSE_DT,
) -> torch.Tensor:
    """``[B, 80, D]`` stored expert future -> ``[B, num_poses, D]`` at ``pose_dt``.

    Sub-sampling, not interpolation: every wanted stamp is an exact dataset row
    (see :func:`expert_future_slice`), so there is nothing to interpolate and no
    smoothing to introduce.
    """
    return ego_future[:, expert_future_slice(num_poses, pose_dt, ego_future.shape[1], dataset_dt)]


def scoring_horizon_slice(num_steps: int, stored_len: int) -> slice:
    """The first ``num_steps`` stored rows, i.e. ``t = dt .. num_steps * dt``.

    Used to clip the neighbour futures and the EP reference path to the scorer's
    4 s horizon.  These stay at the dataset's 0.1 s, which *is* the scoring step,
    so clipping is exact where up-sampling the coarse 8 poses would not be.
    """
    if num_steps > stored_len:
        raise ValueError(
            f"the scorer needs {num_steps} steps but the dataset stores {stored_len}"
        )
    return slice(0, num_steps)


def upsample_poses(
    poses: torch.Tensor,
    num_steps: int,
    pose_dt: float,
    step_dt: float,
) -> torch.Tensor:
    """``[..., P, 4]`` poses at ``pose_dt`` -> ``[..., num_steps, 4]`` at ``step_dt``.

    ``transform_trajectory`` + ``get_trajectory_as_array`` in tensor form.  The
    anchor set is ``[ego pose at t=0] + poses``, which in this head's ego-centric
    frame makes the extra anchor the origin with zero heading -- that is what
    lets ``t < pose_dt`` be interpolated instead of extrapolated.  Output stamps
    are ``t = step_dt, ..., num_steps * step_dt``; navsim also emits the ``t=0``
    row, which every consumer here drops.

    x/y are linear.  The heading is unwrapped along the anchor axis *before*
    interpolating -- otherwise a pose crossing the branch cut would be blended
    the long way round -- and then re-exported as (cos, sin), which is already
    2*pi-periodic, so no re-wrap is needed.
    """
    if poses.shape[-1] != 4:
        raise ValueError(f"poses must be (x, y, cos, sin), got {poses.shape[-1]} columns")
    num_poses = poses.shape[-2]
    if abs(num_steps * step_dt - num_poses * pose_dt) > _TOL:
        raise ValueError(
            f"horizon mismatch: {num_poses} x {pose_dt} s = {num_poses * pose_dt} s "
            f"but {num_steps} x {step_dt} s = {num_steps * step_dt} s"
        )
    lead = poses.shape[:-2]

    anchor_xy = torch.cat((poses.new_zeros(lead + (1, 2)), poses[..., :2]), dim=-2)
    heading = torch.atan2(poses[..., 3], poses[..., 2])
    anchor_h = _phase_unwrap(torch.cat((heading.new_zeros(lead + (1,)), heading), dim=-1))

    stamps = torch.arange(1, num_steps + 1, device=poses.device, dtype=poses.dtype) * step_dt
    position = stamps / pose_dt
    index = position.floor().clamp_(0, num_poses - 1).long()  # [num_steps]
    alpha = position - index.to(position.dtype)

    xy0 = anchor_xy.index_select(-2, index)
    xy1 = anchor_xy.index_select(-2, index + 1)
    xy = torch.lerp(xy0, xy1, alpha[:, None])

    h0 = anchor_h.index_select(-1, index)
    h1 = anchor_h.index_select(-1, index + 1)
    interpolated = torch.lerp(h0, h1, alpha)
    return torch.cat((xy, interpolated.cos()[..., None], interpolated.sin()[..., None]), dim=-1)


def _phase_unwrap(headings: torch.Tensor) -> torch.Tensor:
    """``numpy.unwrap`` along the last dim (``AngularInterpolator``'s pre-step)."""
    two_pi = 2.0 * math.pi
    adjustments = torch.zeros_like(headings)
    adjustments[..., 1:] = torch.cumsum(
        torch.round(torch.diff(headings, dim=-1) / two_pi), dim=-1
    )
    return headings - two_pi * adjustments


__all__ = [
    "DATASET_POSE_DT",
    "expert_future_slice",
    "resample_expert_future",
    "scoring_horizon_slice",
    "upsample_poses",
]
