"""Batched GPU port of NAVSIM's ``PDMSimulator`` (LQR tracker + bicycle model).

Semantically identical to :mod:`planner_metrics.pdm_simulator` -- the literal
numpy transliteration of the original -- but restructured so the whole proposal
set rolls out in one pass on the GPU.  ``planner_metrics/test_pdm_simulator.py`` pins the
two implementations against each other.

Three exact algebraic reformulations carry the speed; each is a regrouping of
the same arithmetic, so they agree with the original to fp64 round-off rather
than approximating it:

**Velocity/acceleration fit -> one constant matmul.**  ``A = diag(a) @ L`` with
``a[2i] = cos h_i``, ``a[2i+1] = sin h_i`` and ``L`` a pose-independent
lower-triangular matrix (``L[i,0] = dt``, ``L[i,1..i] = dt**2``).  Then

    (A^T A)_{cd} = sum_i (cos^2 h_i + sin^2 h_i) L_ic L_id = (L^T L)_{cd}

is *pose-independent*, and ``A^T y = L^T s`` with ``s_i = dx_i cos h_i + dy_i
sin h_i``.  The batched ``pinv`` of the original therefore collapses to a single
constant ``W = pinv(L^T L + jerk R^T R) @ L^T`` and the fit becomes ``x = s @
W^T``.  ``W`` is built once with numpy's ``pinv`` on the CPU, the same LAPACK
call the original makes, so the pseudo-inverse itself is bit-identical.

**Curvature fit -> batched Cholesky instead of SVD-pinv.**  ``A = diag(v) @ C``
with the same constant ``C`` (``C == L``), so ``A^T A + Q = C^T diag(v^2) C +
Q``.  ``Q`` is diagonal and strictly positive (``Q[0,0] = 1e-10``, the rest
``1e-2``), which makes the system symmetric positive-definite; on an SPD matrix
``pinv`` and the inverse coincide, so a Cholesky solve replaces the SVD.

**Lateral LQR -> closed form instead of a 10-step matrix product.**  The
per-step transition is ``S_t = I + a_t E01 + b_t E12`` and among ``{E01, E12}``
only ``E01 E12 = E02`` is non-zero, every other product vanishing.  Unrolling
the horizon product gives

    A = I + (sum_t a_t) E01 + (sum_t b_t) E12 + (sum_{i>j} a_i b_j) E02
    B = dt [ sum_k sum_{i>j>k} a_i b_j , sum_k sum_{t>k} b_t , H ]^T
    g = [ sum_k w_k sum_{t>k} a_t , sum_k w_k , 0 ]

so the tracking-horizon loop becomes a handful of cumulative sums.

These regroupings change floating-point summation order, so results differ from
the literal implementation at ~1e-13 -- the same magnitude the original's own
authors accepted when they benchmarked an ``einsum`` expansion of this loop
(see the note in ``batch_lqr.py`` about ``c_einsum`` accumulation order).  The
comfort labels the simulator feeds are 0/1 decisions against bounds no proposal
sits within 1e-13 of, and the tests assert they come out bit-identical.

The 80-step rollout itself is irreducibly sequential.  What is left is per-step
elementwise work on ``[B]`` and ``[B, 10]`` tensors, which is launch-bound
rather than FLOP-bound, so the step body is handed to ``torch.compile`` to be
fused into a couple of Triton kernels.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from planner_metrics.pdm_simulator import (
    INITIAL_CURVATURE_PENALTY,
    STATE_ACC_X,
    STATE_ACC_Y,
    STATE_ANGULAR_ACCELERATION,
    STATE_ANGULAR_VELOCITY,
    STATE_HEADING,
    STATE_SIZE,
    STATE_STEERING_ANGLE,
    STATE_STEERING_RATE,
    STATE_VEL_X,
    STATE_VEL_Y,
    STATE_X,
    STATE_Y,
    WHEEL_BASE_PACIFICA,
    _make_banded_difference_matrix,
)

#: ``BatchLQRTracker`` / ``BatchKinematicBicycleModel`` defaults, verbatim.
Q_LONGITUDINAL = 10.0
R_LONGITUDINAL = 1.0
Q_LATERAL = (1.0, 10.0, 0.0)
R_LATERAL = 1.0
TRACKING_HORIZON = 10
JERK_PENALTY = 1e-4
CURVATURE_RATE_PENALTY = 1e-2
STOPPING_PROPORTIONAL_GAIN = 0.5
STOPPING_VELOCITY = 0.2
MAX_STEERING_ANGLE = np.pi / 3
ACCEL_TIME_CONSTANT = 0.2
STEERING_ANGLE_TIME_CONSTANT = 0.05

_COMPILE_DISABLED = os.environ.get("PDM_SIM_NO_COMPILE", "") not in ("", "0", "false", "False")


# ---------------------------------------------------------------------------
# constants, built once per (num_poses, dt, device, dtype)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Constants:
    fit_weights: torch.Tensor  # [M, M] velocity/accel fit operator W
    profile_matrix: torch.Tensor  # [M, M] C (== L), curvature fit design
    curvature_penalty: torch.Tensor  # [M, M] Q
    ref_velocity_index: torch.Tensor  # [T] gather index for reference velocity
    ref_curvature_index: torch.Tensor  # [T * H] gather index for curvature slices
    horizon_steps: torch.Tensor  # [H] dt * arange(H)
    lon_gain: float  # closed-form longitudinal LQR gain


_CACHE: dict[tuple, _Constants] = {}


def _lower_design_matrix(num: int, dt: float) -> np.ndarray:
    """``L``: ``L[i, 0] = dt``, ``L[i, 1..i] = dt**2``, zero above the diagonal.

    This is simultaneously the velocity fit's per-pose row block (after the
    ``triu`` mask) and the curvature fit's ``A / diag(velocity)``: navsim builds
    the two from different code but they are the same matrix.
    """
    matrix = np.tril(np.full((num, num), dt**2, dtype=np.float64))
    matrix[:, 0] = dt
    return matrix


def _build_constants(num_poses: int, dt: float, device, dtype) -> _Constants:
    matrix = _lower_design_matrix(num_poses, dt)

    # jerk regularizer: R = [0 | banded difference], rows = num_poses - 2.
    banded = _make_banded_difference_matrix(num_poses - 2)
    reg = np.concatenate([np.zeros((len(banded), 1), dtype=np.float64), banded], axis=1)

    normal = matrix.T @ matrix + JERK_PENALTY * (reg.T @ reg)
    fit_weights = np.linalg.pinv(normal) @ matrix.T

    penalty = CURVATURE_RATE_PENALTY * np.eye(num_poses, dtype=np.float64)
    penalty[0, 0] = INITIAL_CURVATURE_PENALTY

    # reference_idx = min(current_index + tracking_horizon, num_poses - 1)
    steps = np.arange(num_poses)
    ref_velocity_index = np.minimum(steps + TRACKING_HORIZON, num_poses - 1)
    # The curvature slice at current_index t is
    #   [kappa_t, ..., kappa_{ref_idx-1}] padded with kappa_{ref_idx},
    # which is exactly kappa[min(t + k, num_poses - 1)] for k in [0, H).
    ref_curvature_index = np.minimum(
        steps[:, None] + np.arange(TRACKING_HORIZON)[None, :], num_poses - 1
    )

    lon_b = TRACKING_HORIZON * dt
    lon_gain = -(lon_b * Q_LONGITUDINAL) / (lon_b * Q_LONGITUDINAL * lon_b + R_LONGITUDINAL)

    as_tensor = lambda array: torch.as_tensor(array, device=device, dtype=dtype)  # noqa: E731
    as_index = lambda array: torch.as_tensor(  # noqa: E731
        np.ascontiguousarray(array.reshape(-1)), device=device, dtype=torch.long
    )
    return _Constants(
        fit_weights=as_tensor(fit_weights),
        profile_matrix=as_tensor(matrix),
        curvature_penalty=as_tensor(penalty),
        ref_velocity_index=as_index(ref_velocity_index),
        ref_curvature_index=as_index(ref_curvature_index),
        horizon_steps=as_tensor(dt * np.arange(TRACKING_HORIZON, dtype=np.float64)),
        lon_gain=float(lon_gain),
    )


def _constants(num_poses: int, dt: float, device, dtype) -> _Constants:
    key = (num_poses, float(dt), str(device), dtype)
    cached = _CACHE.get(key)
    if cached is None:
        cached = _build_constants(num_poses, dt, device, dtype)
        _CACHE[key] = cached
    return cached


# ---------------------------------------------------------------------------
# profile fitting
# ---------------------------------------------------------------------------


def _generate_profile(initial: torch.Tensor, derivatives: torch.Tensor, dt: float) -> torch.Tensor:
    """``batch_lqr_utils._generate_profile_from_initial_condition_and_derivatives``."""
    cumulative = torch.cumsum(derivatives * dt, dim=-1)
    padded = torch.nn.functional.pad(cumulative, (1, 0))
    return initial[..., None] + padded


def fit_profiles(poses: torch.Tensor, dt: float, constants: Optional[_Constants] = None):
    """Velocity and curvature profiles for the reference trajectory.

    ``poses`` is ``[B, M + 1, 3]``; returns ``([B, M], [B, M])``.
    """
    num_poses = poses.shape[1] - 1
    if constants is None:
        constants = _constants(num_poses, dt, poses.device, poses.dtype)

    differences = poses[:, 1:] - poses[:, :-1]
    heading_reference = poses[:, :-1, 2]
    cos_h, sin_h = torch.cos(heading_reference), torch.sin(heading_reference)

    # A^T y collapses onto the heading-projected displacement; see module docstring.
    projected = differences[..., 0] * cos_h + differences[..., 1] * sin_h
    fit = projected @ constants.fit_weights.transpose(0, 1)
    velocity_profile = _generate_profile(fit[:, 0], fit[:, 1:], dt)

    heading_displacements = torch.atan2(
        torch.sin(differences[..., 2]), torch.cos(differences[..., 2])
    )

    design = constants.profile_matrix[None] * velocity_profile[:, :, None]
    design_t = design.transpose(1, 2)
    normal = design_t @ design + constants.curvature_penalty
    rhs = design_t @ heading_displacements[..., None]

    # Q's diagonal is strictly positive, so `normal` is SPD and pinv == inverse.
    factor, _info = torch.linalg.cholesky_ex(normal)
    solution = torch.cholesky_solve(rhs, factor)[..., 0]
    curvature_profile = _generate_profile(solution[:, 0], solution[:, 1:], dt)

    return velocity_profile, curvature_profile


# ---------------------------------------------------------------------------
# one rollout step
# ---------------------------------------------------------------------------


def _step(
    pos_x: torch.Tensor,
    pos_y: torch.Tensor,
    heading: torch.Tensor,
    velocity: torch.Tensor,
    accel: torch.Tensor,
    steering: torch.Tensor,
    yaw_rate: torch.Tensor,
    ref_x: torch.Tensor,
    ref_y: torch.Tensor,
    ref_cos: torch.Tensor,
    ref_sin: torch.Tensor,
    ref_heading: torch.Tensor,
    ref_velocity: torch.Tensor,
    ref_curvature: torch.Tensor,
    horizon_steps: torch.Tensor,
    dt: float,
    wheel_base,
    lon_gain: float,
):
    """``BatchLQRTracker.track_trajectory`` + ``propagate_state`` for one index.

    ``wheel_base`` is the *motion model's* -- a scalar or a ``[B]`` tensor.  The
    tracker's stays at :data:`WHEEL_BASE_PACIFICA`; see the note in
    :func:`simulate_proposals`.
    """
    # -- initial velocity and lateral state -------------------------------
    error_x = pos_x - ref_x
    error_y = pos_y - ref_y
    lateral_error = -error_x * ref_sin + error_y * ref_cos
    heading_error_raw = heading - ref_heading
    heading_error = torch.atan2(torch.sin(heading_error_raw), torch.cos(heading_error_raw))

    # -- longitudinal command ---------------------------------------------
    velocity_error = velocity - ref_velocity
    stopping = (ref_velocity <= STOPPING_VELOCITY) & (velocity <= STOPPING_VELOCITY)
    accel_command = torch.where(
        stopping, -STOPPING_PROPORTIONAL_GAIN * velocity_error, lon_gain * velocity_error
    )

    # -- lateral command: closed form of the tracking-horizon product -----
    profile = velocity[:, None] + accel_command[:, None] * horizon_steps  # [B, H]
    lateral_gain = profile * dt  # a_t
    heading_gain = lateral_gain / WHEEL_BASE_PACIFICA  # b_t, tracker's wheel base
    affine = -profile * ref_curvature * dt  # w_t

    horizon = float(TRACKING_HORIZON)
    sum_a = lateral_gain.sum(-1)
    sum_b = heading_gain.sum(-1)
    # tail_a[j] = sum_{i > j} a_i
    tail_a = sum_a[:, None] - torch.cumsum(lateral_gain, dim=-1)

    cross = heading_gain * tail_a  # b_j * sum_{i>j} a_i
    a_02 = cross.sum(-1)
    # sum_k sum_{j>k} c_j = H * sum_j c_j - sum_k cumsum(c)_k
    b_0 = dt * (horizon * a_02 - torch.cumsum(cross, dim=-1).sum(-1))
    b_1 = dt * (horizon * sum_b - torch.cumsum(heading_gain, dim=-1).sum(-1))
    b_2 = dt * horizon

    g_0 = (affine * tail_a).sum(-1)
    g_1 = affine.sum(-1)

    error_0 = lateral_error + sum_a * heading_error + a_02 * steering + g_0
    error_1 = heading_error + sum_b * steering + g_1
    error_1 = torch.atan2(torch.sin(error_1), torch.cos(error_1))

    q_0, q_1, q_2 = Q_LATERAL
    denominator = b_0 * b_0 * q_0 + b_1 * b_1 * q_1 + b_2 * b_2 * q_2 + R_LATERAL
    numerator = b_0 * q_0 * error_0 + b_1 * q_1 * error_1
    if q_2:  # pragma: no cover - navsim pins q_lateral[LATERAL_STEERING_ANGLE] to 0
        numerator = numerator + b_2 * q_2 * torch.atan2(torch.sin(steering), torch.cos(steering))
    steering_rate_command = torch.where(
        stopping, torch.zeros_like(numerator), -numerator / denominator
    )

    # -- kinematic bicycle model ------------------------------------------
    ideal_steering = dt * steering_rate_command + steering
    updated_accel = dt / (dt + ACCEL_TIME_CONSTANT) * (accel_command - accel) + accel
    updated_steering = (
        dt / (dt + STEERING_ANGLE_TIME_CONSTANT) * (ideal_steering - steering) + steering
    )
    updated_steering_rate = (updated_steering - steering) / dt

    next_x = pos_x + velocity * torch.cos(heading) * dt
    next_y = pos_y + velocity * torch.sin(heading) * dt
    raw_heading = heading + velocity * torch.tan(steering) / wheel_base * dt
    next_heading = torch.remainder(raw_heading + np.pi, 2.0 * np.pi) - np.pi
    next_velocity = velocity + updated_accel * dt
    next_steering = torch.clamp(
        steering + updated_steering_rate * dt, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE
    )
    next_yaw_rate = next_velocity * torch.tan(next_steering) / wheel_base
    yaw_accel = (next_yaw_rate - yaw_rate) / dt

    return (
        next_x,
        next_y,
        next_heading,
        next_velocity,
        updated_accel,
        next_steering,
        next_yaw_rate,
        updated_steering_rate,
        yaw_accel,
    )


_compiled_step = None
_compile_abandoned = False


def _guarded_step(*args):
    """Call the compiled step, degrading to eager the first time Dynamo refuses.

    ``dynamic=False, fullgraph=True`` gives one compiled variant per input shape
    and only eight recompile slots; exhausting them *raises* instead of falling
    back to eager.  Any process that pushes more than eight distinct shapes
    through the simulator -- a whole test session, or a validation pass batched
    differently from training -- would otherwise die on a limit that has nothing
    to do with the numerics.  Eager is the same computation, so the fallback
    costs speed and nothing else.
    """
    global _compile_abandoned
    try:
        return _compiled_step(*args)
    except Exception as exc:  # pragma: no cover - depends on Dynamo state
        _compile_abandoned = True
        warnings.warn(
            f"pdm_simulator_torch: falling back to the eager step ({exc.__class__.__name__}: {exc}). "
            "Numerics are unchanged; the step is ~1.6x slower.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _step(*args)


def _get_step(enable_compile: bool):
    global _compiled_step
    if not enable_compile or _COMPILE_DISABLED or _compile_abandoned:
        return _step
    if _compiled_step is None:
        try:
            _compiled_step = torch.compile(_step, dynamic=False, fullgraph=True)
        except Exception:  # pragma: no cover - inductor unavailable
            _compiled_step = _step
    return _guarded_step


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def simulate_proposals(
    poses: torch.Tensor,
    initial_states: torch.Tensor,
    dt: float,
    wheel_base=WHEEL_BASE_PACIFICA,
    compile: bool = True,
) -> torch.Tensor:
    """Roll the LQR tracker + bicycle model along ``poses``.

    Args:
        poses: ``[B, T, 3]`` reference poses (x, y, heading) -- the *predicted*
            trajectory, without the ego's own pose.  The ``[B, T+1, 3]``
            reference array the tracker sees is built here with navsim's row-0
            duplication (see :func:`pdm_simulator.reference_states_from_poses`).
        initial_states: ``[B, STATE_SIZE]`` navsim-layout ego states to start from.
        dt: ``proposal_sampling.interval_length``; 0.1 s in NAVSIM and in this
            dataset.
        wheel_base: the *motion model's* wheel base -- a float, or a ``[B]``
            tensor for a per-scene vehicle.  The tracker keeps pacifica's
            3.089 m regardless: ``PDMSimulator.simulate_proposals`` assigns
            ``self._motion_model._vehicle`` from the ego state but never touches
            ``self._tracker._wheel_base``, so the original tracks every vehicle
            as if it were a Pacifica.  That asymmetry is reproduced, not fixed.
        compile: fuse the step body with ``torch.compile``.  Disable for tiny
            batches, where compilation costs more than it saves.

    Returns:
        ``[B, T+1, STATE_SIZE]`` simulated states; row 0 is ``initial_states``.
    """
    assert poses.dim() == 3 and poses.shape[-1] == 3, poses.shape
    dtype = poses.dtype
    device = poses.device
    batch, num_poses, _ = poses.shape

    reference = torch.cat((poses[:, :1], poses), dim=1)  # row-0 duplication
    constants = _constants(num_poses, dt, device, dtype)
    velocity_profile, curvature_profile = fit_profiles(reference, dt, constants)

    ref_velocity = velocity_profile.index_select(1, constants.ref_velocity_index)  # [B, T]
    ref_curvature = curvature_profile.index_select(1, constants.ref_curvature_index).reshape(
        batch, num_poses, TRACKING_HORIZON
    )
    ref_pose = reference[:, :num_poses]  # tracker reads proposal_states[:, current_index]
    ref_x, ref_y, ref_heading = ref_pose[..., 0], ref_pose[..., 1], ref_pose[..., 2]
    ref_cos, ref_sin = torch.cos(ref_heading), torch.sin(ref_heading)

    pos_x = initial_states[:, STATE_X]
    pos_y = initial_states[:, STATE_Y]
    heading = initial_states[:, STATE_HEADING]
    velocity = initial_states[:, STATE_VEL_X]
    accel = initial_states[:, STATE_ACC_X]
    steering = initial_states[:, STATE_STEERING_ANGLE]
    yaw_rate = initial_states[:, STATE_ANGULAR_VELOCITY]

    step_fn = _get_step(compile and device.type == "cuda")
    zeros = torch.zeros_like(pos_x)
    rows = [
        torch.stack(
            (
                pos_x,
                pos_y,
                heading,
                velocity,
                initial_states[:, STATE_VEL_Y],
                accel,
                initial_states[:, STATE_ACC_Y],
                steering,
                initial_states[:, STATE_STEERING_RATE],
                yaw_rate,
                initial_states[:, STATE_ANGULAR_ACCELERATION],
            ),
            dim=-1,
        )
    ]

    for index in range(num_poses):
        (
            pos_x,
            pos_y,
            heading,
            velocity,
            accel,
            steering,
            yaw_rate,
            steering_rate,
            yaw_accel,
        ) = step_fn(
            pos_x,
            pos_y,
            heading,
            velocity,
            accel,
            steering,
            yaw_rate,
            ref_x[:, index],
            ref_y[:, index],
            ref_cos[:, index],
            ref_sin[:, index],
            ref_heading[:, index],
            ref_velocity[:, index],
            ref_curvature[:, index],
            constants.horizon_steps,
            dt,
            wheel_base,
            constants.lon_gain,
        )
        # VELOCITY_Y and ACCELERATION_Y are identically zero in every simulated
        # state -- `_update_commands` writes 0.0 into ACCELERATION_Y and
        # `propagate_state` copies it through.  The lateral-accel comfort bound
        # is structurally vacuous in NAVSIM because of this; reproducing it is
        # the whole point of running the simulator.
        rows.append(
            torch.stack(
                (
                    pos_x,
                    pos_y,
                    heading,
                    velocity,
                    zeros,
                    accel,
                    zeros,
                    steering,
                    steering_rate,
                    yaw_rate,
                    yaw_accel,
                ),
                dim=-1,
            )
        )

    return torch.stack(rows, dim=1)


def initial_states_from_ego(ego_current_state: torch.Tensor) -> torch.Tensor:
    """``ego_state_to_state_array`` for Diffusion-Planner's ``ego_current_state``.

    DP stores ``[x, y, cos, sin, vx, vy, ax, ay, steering_angle, yaw_rate]`` in
    the ego frame, covering every column the rollout reads: X, Y, HEADING,
    VELOCITY_X, ACCELERATION_X, STEERING_ANGLE, ANGULAR_VELOCITY.
    ``STEERING_RATE`` and ``ANGULAR_ACCELERATION`` are never read (the tracker's
    command overwrites the steering rate before ``get_state_dot`` sees it), so
    leaving them zero is exact, not an approximation.

    Autoware's ``base_link`` is taken as the rear axle, matching navsim's
    rear-axle state convention.

    ``ego_current_state`` is ``[..., >=10]``; returns ``[..., STATE_SIZE]``.
    """
    ego = ego_current_state
    out = ego.new_zeros(ego.shape[:-1] + (STATE_SIZE,))
    out[..., STATE_X] = ego[..., 0]
    out[..., STATE_Y] = ego[..., 1]
    out[..., STATE_HEADING] = torch.atan2(ego[..., 3], ego[..., 2])
    out[..., STATE_VEL_X] = ego[..., 4]
    out[..., STATE_VEL_Y] = ego[..., 5]
    out[..., STATE_ACC_X] = ego[..., 6]
    out[..., STATE_ACC_Y] = ego[..., 7]
    out[..., STATE_STEERING_ANGLE] = ego[..., 8]
    out[..., STATE_ANGULAR_VELOCITY] = ego[..., 9]
    return out


def initial_states_from_poses(poses: torch.Tensor, dt: float) -> torch.Tensor:
    """Fallback initial state when the ego's own kinematics are unavailable.

    NAVSIM starts the rollout from ``metric_cache.ego_state``.  Callers that do
    not carry it (the standalone ``comfort_score(poses, dt)`` entry point) get
    the next best thing: the trajectory's own first pose, with velocity and
    acceleration from its leading finite differences and the steering angle
    inverted from the initial yaw rate through the bicycle model.  This makes
    the rollout start *on* the reference with a consistent state instead of
    from rest, which would otherwise spend the first second catching up.
    """
    dtype = poses.dtype
    velocity = (poses[:, 1, :2] - poses[:, 0, :2]).norm(dim=-1) / dt
    second = (poses[:, 2, :2] - poses[:, 1, :2]).norm(dim=-1) / dt
    heading = poses[:, 0, 2]
    yaw_rate = (
        torch.atan2(
            torch.sin(poses[:, 1, 2] - poses[:, 0, 2]), torch.cos(poses[:, 1, 2] - poses[:, 0, 2])
        )
        / dt
    )

    out = poses.new_zeros((poses.shape[0], STATE_SIZE))
    out[:, STATE_X] = poses[:, 0, 0]
    out[:, STATE_Y] = poses[:, 0, 1]
    out[:, STATE_HEADING] = heading
    out[:, STATE_VEL_X] = velocity
    out[:, STATE_ACC_X] = (second - velocity) / dt
    out[:, STATE_STEERING_ANGLE] = torch.atan(
        yaw_rate * WHEEL_BASE_PACIFICA / velocity.clamp_min(torch.finfo(dtype).eps)
    ).clamp(-MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)
    out[:, STATE_ANGULAR_VELOCITY] = yaw_rate
    return out


__all__ = [
    "fit_profiles",
    "initial_states_from_ego",
    "initial_states_from_poses",
    "simulate_proposals",
]
