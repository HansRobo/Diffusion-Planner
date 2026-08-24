"""Literal transliteration of NAVSIM's ``PDMSimulator`` closed-loop rollout.

NAVSIM never applies the comfort bounds to a planner's raw waypoints.
``PDMScorer.score_proposals`` is handed ``self._states``, and those states come
out of ``PDMSimulator.simulate_proposals``: a ``BatchLQRTracker`` computing
(acceleration, steering-rate) commands that a ``BatchKinematicBicycleModel``
integrates forward.  Both stages low-pass the reference -- the bicycle model
does so explicitly, with ``accel_time_constant = 0.2 s`` and
``steering_angle_time_constant = 0.05 s`` first-order lags.  Comfort therefore
measures how a tracked vehicle would ride, never the waypoint noise of the
trajectory head.

This module is the reference implementation: a 1:1 numpy transliteration of

    navsim/planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py
    navsim/planning/simulation/planner/pdm_planner/simulation/batch_lqr.py
    navsim/planning/simulation/planner/pdm_planner/simulation/batch_lqr_utils.py
    navsim/planning/simulation/planner/pdm_planner/simulation/batch_kinematic_bicycle.py

kept deliberately literal -- same loops, same ``einsum`` contraction order, same
``pinv`` -- so it can serve as the ground truth the fast GPU port is tested
against.  Speed is not a goal here; ``drivor_simulator`` is the fast path.

Three properties of the original are load-bearing and easy to lose:

* ``ACCELERATION_Y`` is identically zero in every simulated state.
  ``_update_commands`` writes ``0.0`` into it and ``propagate_state`` copies the
  propagating state's ``ACCELERATION_2D`` into the output, so the lateral-accel
  comfort bound is structurally vacuous and magnitude-jerk collapses onto
  ``|lon jerk|``.  A port that finite-differences poses instead gets a live
  lateral bound and fails on noise.
* The scorer reads rear-axle ``ACCELERATION_X/Y`` directly.
  ``state_array_to_center_state_array`` is never called anywhere in the scoring
  path, so there is no rear-axle-to-centre conversion to replicate.
* ``PDMSimulator.simulate_proposals`` overrides ``_motion_model._vehicle`` but
  never ``_tracker._wheel_base``, so the tracker keeps pacifica's 3.089 m even
  when the ego vehicle is something else.  See ``simulate_proposals``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# ``StateIndex`` from navsim/.../utils/pdm_enums.py -- the 11-column ego state.
STATE_X = 0
STATE_Y = 1
STATE_HEADING = 2
STATE_VEL_X = 3
STATE_VEL_Y = 4
STATE_ACC_X = 5
STATE_ACC_Y = 6
STATE_STEERING_ANGLE = 7
STATE_STEERING_RATE = 8
STATE_ANGULAR_VELOCITY = 9
STATE_ANGULAR_ACCELERATION = 10
STATE_SIZE = 11

STATE_SE2 = slice(0, 3)
VELOCITY_2D = slice(3, 5)
ACCELERATION_2D = slice(5, 7)

#: ``LateralStateIndex``.
LATERAL_ERROR = 0
HEADING_ERROR = 1
LATERAL_STEERING_ANGLE = 2
N_LATERAL_STATES = 3

#: ``get_pacifica_parameters().wheel_base``.  The tracker uses this
#: unconditionally -- see the module docstring.
WHEEL_BASE_PACIFICA = 3.089

#: ``batch_lqr_utils.INITIAL_CURVATURE_PENALTY``.
INITIAL_CURVATURE_PENALTY = 1e-10

# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------


def normalize_angle(angle):
    """``pdm_geometry_utils.normalize_angle`` (verbatim)."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def principal_value(angle, min_: float = -np.pi):
    """``nuplan.common.geometry.compute.principal_value`` (verbatim)."""
    return (angle - min_) % (2 * np.pi) + min_


# ---------------------------------------------------------------------------
# batch_lqr_utils.py
# ---------------------------------------------------------------------------


def _batch_matmul(a, b):
    return np.einsum("bij, bjk -> bik", a, b)


def _generate_profile_from_initial_condition_and_derivatives(
    initial_condition: npt.NDArray[np.float64],
    derivatives: npt.NDArray[np.float64],
    discretization_time: float,
) -> npt.NDArray[np.float64]:
    """``batch_lqr_utils._generate_profile_from_initial_condition_and_derivatives``."""
    cumsum = np.cumsum(derivatives * discretization_time, axis=-1)
    return initial_condition[..., None] + np.pad(cumsum, [(0, 0), (1, 0)], mode="constant")


def _get_xy_heading_displacements_from_poses(poses: npt.NDArray[np.float64]):
    """``batch_lqr_utils._get_xy_heading_displacements_from_poses``."""
    assert len(poses.shape) == 3
    assert poses.shape[1] > 1
    assert poses.shape[2] == 3

    pose_differences = np.diff(poses, axis=1)
    xy_displacements = pose_differences[..., :2]
    heading_displacements = normalize_angle(pose_differences[..., 2])
    return xy_displacements, heading_displacements


def _make_banded_difference_matrix(number_rows: int) -> npt.NDArray[np.float64]:
    """``batch_lqr_utils._make_banded_difference_matrix``."""
    banded_matrix = np.zeros((number_rows, number_rows + 1), dtype=np.float64)
    eye = np.eye(number_rows, dtype=np.float64)
    banded_matrix[:, 1:] = eye
    banded_matrix[:, :-1] = -eye
    return banded_matrix


def _fit_initial_velocity_and_acceleration_profile(
    xy_displacements: npt.NDArray[np.float64],
    heading_profile: npt.NDArray[np.float64],
    discretization_time: float,
    jerk_penalty: float,
):
    """``batch_lqr_utils._fit_initial_velocity_and_acceleration_profile``."""
    num_displacements = xy_displacements.shape[1]
    batch_size = heading_profile.shape[0]

    y = xy_displacements.reshape(batch_size, -1)

    headings = np.array(heading_profile, dtype=np.float64)
    A_column = np.zeros(y.shape, dtype=np.float64)
    A_column[:, 0::2] = np.cos(headings)
    A_column[:, 1::2] = np.sin(headings)

    A = np.repeat(A_column[..., None] * discretization_time**2, num_displacements, axis=2)
    A[..., 0] = A_column * discretization_time

    upper_triangle_mask = np.triu(np.ones((num_displacements, num_displacements), dtype=bool), k=1)
    upper_triangle_mask = np.repeat(upper_triangle_mask, 2, axis=0)
    A[:, upper_triangle_mask] = 0.0

    banded_matrix = _make_banded_difference_matrix(num_displacements - 2)
    R = np.block([np.zeros((len(banded_matrix), 1)), banded_matrix])
    R = np.repeat(R[None, ...], batch_size, axis=0)

    A_T, R_T = np.transpose(A, (0, 2, 1)), np.transpose(R, (0, 2, 1))

    intermediate_solution = _batch_matmul(
        np.linalg.pinv(_batch_matmul(A_T, A) + jerk_penalty * _batch_matmul(R_T, R)), A_T
    )
    x = np.einsum("bij, bj -> bi", intermediate_solution, y)

    return x[:, 0], x[:, 1:]


def _fit_initial_curvature_and_curvature_rate_profile(
    heading_displacements: npt.NDArray[np.float64],
    velocity_profile: npt.NDArray[np.float64],
    discretization_time: float,
    curvature_rate_penalty: float,
    initial_curvature_penalty: float = INITIAL_CURVATURE_PENALTY,
):
    """``batch_lqr_utils._fit_initial_curvature_and_curvature_rate_profile``."""
    y = heading_displacements
    batch_dim, dim = y.shape

    A = np.repeat(np.tri(dim, dtype=np.float64)[None, ...], batch_dim, axis=0)
    A[:, :, 0] = velocity_profile * discretization_time

    velocity = velocity_profile * discretization_time**2
    A[:, 1:, 1:] *= velocity[:, None, 1:].transpose(0, 2, 1)

    Q = curvature_rate_penalty * np.eye(dim)
    Q[0, 0] = initial_curvature_penalty

    A_T = A.transpose(0, 2, 1)
    intermediate = _batch_matmul(np.linalg.pinv(_batch_matmul(A_T, A) + Q), A_T)
    x = np.einsum("bij,bj->bi", intermediate, y)

    return x[:, 0], x[:, 1:]


def get_velocity_curvature_profiles_with_derivatives_from_poses(
    discretization_time: float,
    poses: npt.NDArray[np.float64],
    jerk_penalty: float,
    curvature_rate_penalty: float,
):
    """``batch_lqr_utils.get_velocity_curvature_profiles_with_derivatives_from_poses``."""
    xy_displacements, heading_displacements = _get_xy_heading_displacements_from_poses(poses)

    initial_velocity, acceleration_profile = _fit_initial_velocity_and_acceleration_profile(
        xy_displacements=xy_displacements,
        heading_profile=poses[:, :-1, 2],
        discretization_time=discretization_time,
        jerk_penalty=jerk_penalty,
    )

    velocity_profile = _generate_profile_from_initial_condition_and_derivatives(
        initial_condition=initial_velocity,
        derivatives=acceleration_profile,
        discretization_time=discretization_time,
    )

    initial_curvature, curvature_rate_profile = _fit_initial_curvature_and_curvature_rate_profile(
        heading_displacements=heading_displacements,
        velocity_profile=velocity_profile,
        discretization_time=discretization_time,
        curvature_rate_penalty=curvature_rate_penalty,
    )

    curvature_profile = _generate_profile_from_initial_condition_and_derivatives(
        initial_condition=initial_curvature,
        derivatives=curvature_rate_profile,
        discretization_time=discretization_time,
    )

    return velocity_profile, acceleration_profile, curvature_profile, curvature_rate_profile


# ---------------------------------------------------------------------------
# batch_lqr.py
# ---------------------------------------------------------------------------


class BatchLQRTracker:
    """``batch_lqr.BatchLQRTracker`` (literal)."""

    def __init__(
        self,
        q_longitudinal=(10.0,),
        r_longitudinal=(1.0,),
        q_lateral=(1.0, 10.0, 0.0),
        r_lateral=(1.0,),
        discretization_time: float = 0.1,
        tracking_horizon: int = 10,
        jerk_penalty: float = 1e-4,
        curvature_rate_penalty: float = 1e-2,
        stopping_proportional_gain: float = 0.5,
        stopping_velocity: float = 0.2,
        wheel_base: float = WHEEL_BASE_PACIFICA,
    ) -> None:
        self._q_longitudinal = float(q_longitudinal[0])
        self._r_longitudinal = float(r_longitudinal[0])
        self._q_lateral = np.diag(q_lateral)
        self._r_lateral = np.diag(r_lateral)
        self._discretization_time = discretization_time
        self._tracking_horizon = tracking_horizon
        self._wheel_base = wheel_base
        self._jerk_penalty = jerk_penalty
        self._curvature_rate_penalty = curvature_rate_penalty
        self._stopping_proportional_gain = stopping_proportional_gain
        self._stopping_velocity = stopping_velocity

        self._proposal_states = None
        self._velocity_profile = None
        self._curvature_profile = None
        self._initialized = False

    def update(self, proposal_states: npt.NDArray[np.float64]) -> None:
        self._proposal_states = proposal_states
        self._velocity_profile, self._curvature_profile = None, None
        self._initialized = True

    def track_trajectory(
        self, current_index: int, initial_states: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """``track_trajectory``; the two ``SimulationIteration`` objects only ever
        contribute ``current_iteration.index``, so it is passed directly."""
        assert self._initialized

        batch_size = len(initial_states)
        (
            initial_velocity,
            initial_lateral_state_vector,
        ) = self._compute_initial_velocity_and_lateral_state(current_index, initial_states)

        (
            reference_velocities,
            curvature_profiles,
        ) = self._compute_reference_velocity_and_curvature_profile(current_index)

        accel_cmds = np.zeros(batch_size, dtype=np.float64)
        steering_rate_cmds = np.zeros(batch_size, dtype=np.float64)

        should_stop_mask = np.logical_and(
            reference_velocities <= self._stopping_velocity,
            initial_velocity <= self._stopping_velocity,
        )
        stopping_accel_cmd, stopping_steering_rate_cmd = self._stopping_controller(
            initial_velocity[should_stop_mask], reference_velocities[should_stop_mask]
        )
        accel_cmds[should_stop_mask] = stopping_accel_cmd
        steering_rate_cmds[should_stop_mask] = stopping_steering_rate_cmd

        accel_cmds[~should_stop_mask] = self._longitudinal_lqr_controller(
            initial_velocity[~should_stop_mask], reference_velocities[~should_stop_mask]
        )

        velocity_profiles = _generate_profile_from_initial_condition_and_derivatives(
            initial_condition=initial_velocity[~should_stop_mask],
            derivatives=np.repeat(
                accel_cmds[~should_stop_mask, None], self._tracking_horizon, axis=-1
            ),
            discretization_time=self._discretization_time,
        )[:, : self._tracking_horizon]

        steering_rate_cmds[~should_stop_mask] = self._lateral_lqr_controller(
            initial_lateral_state_vector[~should_stop_mask],
            velocity_profiles,
            curvature_profiles[~should_stop_mask],
        )

        # ``DynamicStateIndex``: ACCELERATION_X = 0, STEERING_RATE = 1.
        command_states = np.zeros((batch_size, 2), dtype=np.float64)
        command_states[:, 0] = accel_cmds
        command_states[:, 1] = steering_rate_cmds
        return command_states

    def _compute_initial_velocity_and_lateral_state(self, current_index, initial_values):
        initial_trajectory_values = self._proposal_states[:, current_index]

        x_errors = initial_values[:, STATE_X] - initial_trajectory_values[:, STATE_X]
        y_errors = initial_values[:, STATE_Y] - initial_trajectory_values[:, STATE_Y]
        heading_references = initial_trajectory_values[:, STATE_HEADING]

        lateral_errors = -x_errors * np.sin(heading_references) + y_errors * np.cos(
            heading_references
        )
        heading_errors = normalize_angle(initial_values[:, STATE_HEADING] - heading_references)

        initial_velocities = initial_values[:, STATE_VEL_X]
        initial_lateral_state_vector = np.stack(
            [lateral_errors, heading_errors, initial_values[:, STATE_STEERING_ANGLE]], axis=-1
        )
        return initial_velocities, initial_lateral_state_vector

    def _compute_reference_velocity_and_curvature_profile(self, current_index):
        poses = self._proposal_states[..., STATE_SE2]

        if self._velocity_profile is None or self._curvature_profile is None:
            (
                self._velocity_profile,
                _acceleration_profile,
                self._curvature_profile,
                _curvature_rate_profile,
            ) = get_velocity_curvature_profiles_with_derivatives_from_poses(
                discretization_time=self._discretization_time,
                poses=poses,
                jerk_penalty=self._jerk_penalty,
                curvature_rate_penalty=self._curvature_rate_penalty,
            )

        batch_size, num_poses = self._velocity_profile.shape
        reference_idx = min(current_index + self._tracking_horizon, num_poses - 1)
        reference_velocities = self._velocity_profile[:, reference_idx]

        reference_curvature_profiles = np.zeros(
            (batch_size, self._tracking_horizon), dtype=np.float64
        )
        reference_length = reference_idx - current_index
        reference_curvature_profiles[:, 0:reference_length] = self._curvature_profile[
            :, current_index:reference_idx
        ]
        if reference_length < self._tracking_horizon:
            reference_curvature_profiles[:, reference_length:] = self._curvature_profile[
                :, reference_idx, None
            ]

        return reference_velocities, reference_curvature_profiles

    def _stopping_controller(self, initial_velocities, reference_velocities):
        return -self._stopping_proportional_gain * (initial_velocities - reference_velocities), 0.0

    def _longitudinal_lqr_controller(self, initial_velocities, reference_velocities):
        batch_size = len(initial_velocities)
        A = np.ones(batch_size, dtype=np.float64)
        B = np.zeros(batch_size, dtype=np.float64)
        B.fill(self._tracking_horizon * self._discretization_time)
        g = np.zeros(batch_size, dtype=np.float64)
        return self._solve_one_step_longitudinal_lqr(
            initial_state=initial_velocities, reference_state=reference_velocities, A=A, B=B, g=g
        )

    def _lateral_lqr_controller(
        self, initial_lateral_state_vector, velocity_profile, curvature_profile
    ):
        batch_dim = velocity_profile.shape[0]
        I = np.eye(N_LATERAL_STATES, dtype=np.float64)

        in_matrix = np.zeros((N_LATERAL_STATES, 1), np.float64)
        in_matrix[LATERAL_STEERING_ANGLE] = self._discretization_time

        states_matrix_at_step = np.tile(
            I[None, None, ...], [self._tracking_horizon, batch_dim, 1, 1]
        )
        states_matrix_at_step[:, :, LATERAL_ERROR, HEADING_ERROR] = (
            velocity_profile.T * self._discretization_time
        )
        states_matrix_at_step[:, :, HEADING_ERROR, LATERAL_STEERING_ANGLE] = (
            velocity_profile.T * self._discretization_time / self._wheel_base
        )

        affine_terms = np.zeros(
            (self._tracking_horizon, batch_dim, N_LATERAL_STATES), dtype=np.float64
        )
        affine_terms[:, :, HEADING_ERROR] = (
            -velocity_profile.T * curvature_profile.T * self._discretization_time
        )

        A = np.tile(I[None, ...], [batch_dim, 1, 1])
        B = np.zeros((batch_dim, N_LATERAL_STATES, 1), dtype=np.float64)
        g = np.zeros((batch_dim, N_LATERAL_STATES), dtype=np.float64)

        for state_matrix_at_step, affine_term in zip(states_matrix_at_step, affine_terms):
            A = np.einsum("bij, bjk -> bik", state_matrix_at_step, A)
            B = np.einsum("bij, bjk -> bik", state_matrix_at_step, B) + in_matrix
            g = np.einsum("bij, bj  -> bi", state_matrix_at_step, g) + affine_term

        steering_rate_cmd = self._solve_one_step_lateral_lqr(
            initial_state=initial_lateral_state_vector, A=A, B=B, g=g
        )
        return np.squeeze(steering_rate_cmd, axis=-1)

    def _solve_one_step_longitudinal_lqr(self, initial_state, reference_state, A, B, g):
        state_error_zero_input = A * initial_state + g - reference_state
        inverse = -1 / (B * self._q_longitudinal * B + self._r_longitudinal)
        return inverse * B * self._q_longitudinal * state_error_zero_input

    def _solve_one_step_lateral_lqr(self, initial_state, A, B, g):
        Q, R = self._q_lateral, self._r_lateral
        angle_diff_indices = [HEADING_ERROR, LATERAL_STEERING_ANGLE]
        BT = B.transpose(0, 2, 1)

        state_error_zero_input = np.einsum("bij, bj -> bi", A, initial_state) + g
        angle = state_error_zero_input[..., angle_diff_indices]
        state_error_zero_input[..., angle_diff_indices] = np.arctan2(np.sin(angle), np.cos(angle))

        BT_x_Q = np.einsum("bij, jk -> bik", BT, Q)
        Inv = -1 / (np.einsum("bij, bji -> bi", BT_x_Q, B) + R)
        Tail = np.einsum("bij, bj -> bi", BT_x_Q, state_error_zero_input)
        return Inv * Tail


# ---------------------------------------------------------------------------
# batch_kinematic_bicycle.py
# ---------------------------------------------------------------------------


class BatchKinematicBicycleModel:
    """``batch_kinematic_bicycle.BatchKinematicBicycleModel`` (literal)."""

    def __init__(
        self,
        wheel_base: float = WHEEL_BASE_PACIFICA,
        max_steering_angle: float = np.pi / 3,
        accel_time_constant: float = 0.2,
        steering_angle_time_constant: float = 0.05,
    ) -> None:
        self._wheel_base = wheel_base
        self._max_steering_angle = max_steering_angle
        self._accel_time_constant = accel_time_constant
        self._steering_angle_time_constant = steering_angle_time_constant

    def get_state_dot(self, states):
        state_dots = np.zeros(states.shape, dtype=np.float64)
        longitudinal_speeds = states[:, STATE_VEL_X]

        state_dots[:, STATE_X] = longitudinal_speeds * np.cos(states[:, STATE_HEADING])
        state_dots[:, STATE_Y] = longitudinal_speeds * np.sin(states[:, STATE_HEADING])
        state_dots[:, STATE_HEADING] = (
            longitudinal_speeds * np.tan(states[:, STATE_STEERING_ANGLE]) / self._wheel_base
        )
        state_dots[:, VELOCITY_2D] = states[:, ACCELERATION_2D]
        state_dots[:, ACCELERATION_2D] = 0.0
        state_dots[:, STATE_STEERING_ANGLE] = states[:, STATE_STEERING_RATE]
        return state_dots

    def _update_commands(self, states, command_states, sampling_time: float):
        propagating_state = states.copy()
        dt_control = sampling_time

        accel = states[:, STATE_ACC_X]
        steering_angle = states[:, STATE_STEERING_ANGLE]

        ideal_accel_x = command_states[:, 0]
        ideal_steering_angle = dt_control * command_states[:, 1] + steering_angle

        updated_accel_x = (
            dt_control / (dt_control + self._accel_time_constant) * (ideal_accel_x - accel) + accel
        )
        updated_steering_angle = (
            dt_control
            / (dt_control + self._steering_angle_time_constant)
            * (ideal_steering_angle - steering_angle)
            + steering_angle
        )
        updated_steering_rate = (updated_steering_angle - steering_angle) / dt_control

        propagating_state[:, STATE_ACC_X] = updated_accel_x
        propagating_state[:, STATE_ACC_Y] = 0.0
        propagating_state[:, STATE_STEERING_RATE] = updated_steering_rate
        return propagating_state

    def propagate_state(self, states, command_states, sampling_time: float):
        propagating_state = self._update_commands(states, command_states, sampling_time)
        output_state = states.copy()
        state_dot = self.get_state_dot(propagating_state)

        output_state[:, STATE_X] = states[:, STATE_X] + state_dot[:, STATE_X] * sampling_time
        output_state[:, STATE_Y] = states[:, STATE_Y] + state_dot[:, STATE_Y] * sampling_time
        output_state[:, STATE_HEADING] = principal_value(
            states[:, STATE_HEADING] + state_dot[:, STATE_HEADING] * sampling_time
        )
        output_state[:, STATE_VEL_X] = (
            states[:, STATE_VEL_X] + state_dot[:, STATE_VEL_X] * sampling_time
        )
        output_state[:, STATE_VEL_Y] = 0.0
        output_state[:, STATE_STEERING_ANGLE] = np.clip(
            propagating_state[:, STATE_STEERING_ANGLE]
            + state_dot[:, STATE_STEERING_ANGLE] * sampling_time,
            -self._max_steering_angle,
            self._max_steering_angle,
        )
        output_state[:, STATE_ANGULAR_VELOCITY] = (
            output_state[:, STATE_VEL_X]
            * np.tan(output_state[:, STATE_STEERING_ANGLE])
            / self._wheel_base
        )
        output_state[:, ACCELERATION_2D] = state_dot[:, VELOCITY_2D]
        output_state[:, STATE_ANGULAR_ACCELERATION] = (
            output_state[:, STATE_ANGULAR_VELOCITY] - states[:, STATE_ANGULAR_VELOCITY]
        ) / sampling_time
        output_state[:, STATE_STEERING_RATE] = state_dot[:, STATE_STEERING_ANGLE]
        return output_state


# ---------------------------------------------------------------------------
# pdm_simulator.py
# ---------------------------------------------------------------------------


def reference_states_from_poses(poses: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Build the ``[B, T+1, STATE_SIZE]`` reference array the tracker is handed.

    ``compute_navsim_score`` builds it as ``transform_trajectory`` ->
    ``get_trajectory_as_array``.  ``_get_fixed_timesteps`` starts the
    interpolated trajectory one interval *after* the ego's timestamp, while
    ``get_trajectory_as_array`` asks for ``arange(0, horizon + dt, dt)`` and
    clips those requests into the trajectory's span.  The request at t=0
    therefore clamps up onto the first predicted pose, so row 0 is a duplicate
    of row 1 -- reproduced here rather than substituting the ego's own pose.

    Only ``STATE_SE2`` is ever read off this array; ``transform_trajectory``
    itself notes "velocity and acceleration ignored by LQR + bicycle model" and
    fills them with zeros.  ``poses`` is ``[B, T, 3]`` = (x, y, heading).
    """
    poses = np.asarray(poses, dtype=np.float64)
    batch, horizon, _ = poses.shape
    states = np.zeros((batch, horizon + 1, STATE_SIZE), dtype=np.float64)
    states[:, 1:, STATE_SE2] = poses
    states[:, 0, STATE_SE2] = poses[:, 0]
    return states


def simulate_proposals(
    reference_states: npt.NDArray[np.float64],
    initial_states: npt.NDArray[np.float64],
    discretization_time: float,
    wheel_base: float = WHEEL_BASE_PACIFICA,
) -> npt.NDArray[np.float64]:
    """``PDMSimulator.simulate_proposals``.

    ``reference_states`` is ``[B, T+1, STATE_SIZE]`` (see
    :func:`reference_states_from_poses`), ``initial_states`` is ``[B,
    STATE_SIZE]`` -- one row per proposal, the ego state the rollout starts from.

    ``wheel_base`` reaches the motion model only.  The original assigns
    ``self._motion_model._vehicle = initial_ego_state.car_footprint...`` but
    never touches ``self._tracker._wheel_base``, so the tracker is always
    constructed with pacifica's 3.089 m no matter what vehicle is being
    simulated.  That asymmetry is deliberate here.
    """
    reference_states = np.asarray(reference_states, dtype=np.float64)
    initial_states = np.asarray(initial_states, dtype=np.float64)
    num_poses = reference_states.shape[1] - 1

    tracker = BatchLQRTracker(discretization_time=discretization_time)
    motion_model = BatchKinematicBicycleModel(wheel_base=wheel_base)

    proposal_states = reference_states[:, : num_poses + 1]
    tracker.update(proposal_states)

    simulated_states = np.zeros(proposal_states.shape, dtype=np.float64)
    simulated_states[:, 0] = initial_states

    for time_idx in range(1, num_poses + 1):
        command_states = tracker.track_trajectory(time_idx - 1, simulated_states[:, time_idx - 1])
        simulated_states[:, time_idx] = motion_model.propagate_state(
            states=simulated_states[:, time_idx - 1],
            command_states=command_states,
            sampling_time=discretization_time,
        )

    return simulated_states


def initial_states_from_ego_current_state(
    ego_current_state: npt.NDArray[np.float64], num_proposals: int = 1
) -> npt.NDArray[np.float64]:
    """Map Diffusion-Planner's ``ego_current_state`` onto ``ego_state_to_state_array``.

    DP stores ``[x, y, cos, sin, vx, vy, ax, ay, steering_angle, yaw_rate]`` in
    the ego frame, which covers every column the rollout actually reads: X, Y,
    HEADING, VELOCITY_X, ACCELERATION_X, STEERING_ANGLE and ANGULAR_VELOCITY.
    ``STEERING_RATE`` and ``ANGULAR_ACCELERATION`` are write-only in the
    rollout -- ``_update_commands`` overwrites the steering rate before
    ``get_state_dot`` reads it, and angular acceleration is only ever an output
    -- so leaving them at zero changes nothing.

    ``ACCELERATION_Y`` is kept at the ego's real lateral acceleration for row 0
    exactly as ``ego_state_to_state_array`` does, even though every simulated
    row after it is zero; the savgol edge fit does see that first sample.

    ``ego_current_state`` is ``[..., 10]``; returns ``[..., num_proposals,
    STATE_SIZE]`` with the proposal axis inserted, or ``[..., STATE_SIZE]`` when
    ``num_proposals`` is None.
    """
    ego = np.asarray(ego_current_state, dtype=np.float64)
    lead = ego.shape[:-1]
    out = np.zeros(lead + (STATE_SIZE,), dtype=np.float64)
    out[..., STATE_X] = ego[..., 0]
    out[..., STATE_Y] = ego[..., 1]
    out[..., STATE_HEADING] = np.arctan2(ego[..., 3], ego[..., 2])
    out[..., STATE_VEL_X] = ego[..., 4]
    out[..., STATE_VEL_Y] = ego[..., 5]
    out[..., STATE_ACC_X] = ego[..., 6]
    out[..., STATE_ACC_Y] = ego[..., 7]
    out[..., STATE_STEERING_ANGLE] = ego[..., 8]
    out[..., STATE_ANGULAR_VELOCITY] = ego[..., 9]
    if num_proposals is None:
        return out
    return np.repeat(out[..., None, :], num_proposals, axis=-2)
