"""``pdm_simulator_torch`` (fast, batched) vs ``pdm_simulator`` (literal navsim).

``pdm_simulator`` is a 1:1 transliteration of NAVSIM's ``PDMSimulator`` /
``BatchLQRTracker`` / ``BatchKinematicBicycleModel``: same loops, same
``einsum`` order, same ``pinv``.  The torch port regroups three of those
computations into constant matmuls, a Cholesky solve and a closed-form horizon
product; these tests are what makes "the regrouping is exact" a checked claim
rather than an argument.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from planner_metrics import pdm_simulator as ref
from planner_metrics import pdm_simulator_torch as fast

DT = 0.1
HORIZON = 80


def _random_trajectories(batch: int, horizon: int = HORIZON, seed: int = 0):
    """Plausible ego trajectories: a curving, accelerating path plus pose noise.

    The noise matters -- a clean path would hide exactly the high-frequency
    component the tracker is supposed to filter out.
    """
    rng = np.random.default_rng(seed)
    speed = rng.uniform(0.0, 18.0, size=(batch, 1))
    accel = rng.uniform(-2.0, 2.0, size=(batch, 1))
    curvature = rng.uniform(-0.06, 0.06, size=(batch, 1))

    steps = np.arange(1, horizon + 1)[None, :] * DT
    velocity = np.clip(speed + accel * steps, 0.0, None)
    arclength = np.cumsum(velocity * DT, axis=1)
    heading = curvature * arclength
    x = np.cumsum(velocity * np.cos(heading) * DT, axis=1)
    y = np.cumsum(velocity * np.sin(heading) * DT, axis=1)

    poses = np.stack((x, y, heading), axis=-1)
    poses[..., :2] += rng.normal(0.0, 0.05, size=poses[..., :2].shape)
    poses[..., 2] += rng.normal(0.0, 0.01, size=poses[..., 2].shape)

    ego = np.zeros((batch, 10), dtype=np.float64)
    ego[:, 2] = 1.0  # cos(heading) = 1, heading = 0
    ego[:, 4] = speed[:, 0]
    ego[:, 6] = accel[:, 0]
    ego[:, 8] = rng.uniform(-0.05, 0.05, size=batch)
    ego[:, 9] = velocity[:, 0] * curvature[:, 0]
    return poses, ego


def _run_reference(poses: np.ndarray, ego: np.ndarray) -> np.ndarray:
    reference_states = ref.reference_states_from_poses(poses)
    initial = ref.initial_states_from_ego_current_state(ego, num_proposals=None)
    return ref.simulate_proposals(reference_states, initial, DT)


def _run_fast(poses: np.ndarray, ego: np.ndarray, device: str = "cpu") -> np.ndarray:
    poses_t = torch.as_tensor(poses, dtype=torch.float64, device=device)
    ego_t = torch.as_tensor(ego, dtype=torch.float64, device=device)
    initial = fast.initial_states_from_ego(ego_t)
    out = fast.simulate_proposals(poses_t, initial, DT, compile=False)
    return out.cpu().numpy()


# ---------------------------------------------------------------------------
# the profile fits, isolated
# ---------------------------------------------------------------------------


def test_velocity_fit_matches_batched_pinv():
    """The constant-``W`` collapse reproduces navsim's per-batch ``pinv``."""
    poses, _ = _random_trajectories(16, seed=1)
    reference = ref.reference_states_from_poses(poses)[..., ref.STATE_SE2]

    xy, _heading_diff = ref._get_xy_heading_displacements_from_poses(reference)
    want_v0, want_accel = ref._fit_initial_velocity_and_acceleration_profile(
        xy, reference[:, :-1, 2], DT, fast.JERK_PENALTY
    )

    poses_t = torch.as_tensor(reference, dtype=torch.float64)
    constants = fast._constants(reference.shape[1] - 1, DT, poses_t.device, poses_t.dtype)
    differences = poses_t[:, 1:] - poses_t[:, :-1]
    heading = poses_t[:, :-1, 2]
    projected = differences[..., 0] * torch.cos(heading) + differences[..., 1] * torch.sin(heading)
    got = (projected @ constants.fit_weights.T).numpy()

    assert np.abs(got[:, 0] - want_v0).max() < 1e-9
    assert np.abs(got[:, 1:] - want_accel).max() < 1e-9


def test_profile_fits_match_reference():
    """Velocity and curvature profiles, i.e. everything the tracker plans against."""
    poses, _ = _random_trajectories(16, seed=2)
    reference = ref.reference_states_from_poses(poses)[..., ref.STATE_SE2]

    want_velocity, _, want_curvature, _ = (
        ref.get_velocity_curvature_profiles_with_derivatives_from_poses(
            DT, reference, fast.JERK_PENALTY, fast.CURVATURE_RATE_PENALTY
        )
    )
    got_velocity, got_curvature = fast.fit_profiles(
        torch.as_tensor(reference, dtype=torch.float64), DT
    )

    assert np.abs(got_velocity.numpy() - want_velocity).max() < 1e-9
    # The curvature fit swaps SVD-pinv for a Cholesky solve on the same SPD
    # system; the residual is set by the system's conditioning, not the method.
    assert np.abs(got_curvature.numpy() - want_curvature).max() < 1e-7


# ---------------------------------------------------------------------------
# the full rollout
# ---------------------------------------------------------------------------


def test_rollout_matches_reference():
    poses, ego = _random_trajectories(24, seed=3)
    want = _run_reference(poses, ego)
    got = _run_fast(poses, ego)

    assert got.shape == want.shape == (24, HORIZON + 1, ref.STATE_SIZE)
    # Positions in metres after 8 s of closed-loop tracking.
    assert np.abs(got[..., ref.STATE_X] - want[..., ref.STATE_X]).max() < 1e-8
    assert np.abs(got[..., ref.STATE_Y] - want[..., ref.STATE_Y]).max() < 1e-8
    assert np.abs(got - want).max() < 1e-7


def test_rollout_matches_reference_at_standstill():
    """The stopping controller's branch: both reference and ego below 0.2 m/s."""
    batch = 8
    poses = np.zeros((batch, HORIZON, 3), dtype=np.float64)
    poses[..., 0] = np.linspace(0.0, 0.4, HORIZON)[None, :]  # 0.05 m/s crawl
    ego = np.zeros((batch, 10), dtype=np.float64)
    ego[:, 2] = 1.0
    ego[:, 4] = np.linspace(0.0, 0.15, batch)

    want = _run_reference(poses, ego)
    got = _run_fast(poses, ego)
    assert np.abs(got - want).max() < 1e-9


def test_acceleration_y_is_identically_zero():
    """The property that makes NAVSIM's lateral-accel comfort bound vacuous.

    ``_update_commands`` writes ``0.0`` into ``ACCELERATION_Y`` and
    ``propagate_state`` copies the propagating state's ``ACCELERATION_2D``
    through, so no simulated state after row 0 can carry lateral acceleration.
    """
    poses, ego = _random_trajectories(8, seed=4)
    for states in (_run_reference(poses, ego), _run_fast(poses, ego)):
        assert np.all(states[:, 1:, ref.STATE_ACC_Y] == 0.0)
        assert np.all(states[:, 1:, ref.STATE_VEL_Y] == 0.0)


def test_rollout_starts_at_initial_state():
    poses, ego = _random_trajectories(4, seed=5)
    initial = ref.initial_states_from_ego_current_state(ego, num_proposals=None)
    states = _run_fast(poses, ego)
    assert np.abs(states[:, 0] - initial).max() == 0.0


def test_simulator_low_passes_waypoint_noise():
    """Why comfort needs the simulator at all.

    Finite-differencing raw poses amplifies pose noise by ``1/dt**2`` = 100.
    The tracker's first-order lags (``accel_time_constant = 0.2 s``) are what
    keep a noisy-but-drivable trajectory inside the comfort bounds.
    """
    poses, ego = _random_trajectories(32, seed=6)
    states = _run_fast(poses, ego)

    simulated = np.abs(states[:, 1:, ref.STATE_ACC_X])
    finite_difference = np.abs(np.gradient(np.gradient(poses[..., 0], DT, axis=1), DT, axis=1))
    assert simulated.max() < 10.0
    assert finite_difference.max() > 10.0 * simulated.max()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_matches_cpu():
    poses, ego = _random_trajectories(64, seed=7)
    cpu = _run_fast(poses, ego, device="cpu")
    cuda = _run_fast(poses, ego, device="cuda")
    assert np.abs(cuda - cpu).max() < 1e-9


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_compiled_matches_eager():
    # ``_get_step`` memoises a single ``dynamic=False`` compile of the step body, so
    # every distinct shape the session has already pushed through it has burned one of
    # Dynamo's 8 recompile slots.  Once they are gone ``fullgraph=True`` escalates the
    # next recompile to a hard error, which would make this test fail or pass purely on
    # how much of the suite ran before it.  Reset both caches so it measures numerics.
    torch._dynamo.reset()
    fast._compiled_step = None
    fast._compile_abandoned = False

    poses, ego = _random_trajectories(64, seed=8)
    poses_t = torch.as_tensor(poses, dtype=torch.float64, device="cuda")
    initial = fast.initial_states_from_ego(torch.as_tensor(ego, dtype=torch.float64, device="cuda"))
    eager = fast.simulate_proposals(poses_t, initial, DT, compile=False)
    compiled = fast.simulate_proposals(poses_t, initial, DT, compile=True)
    assert torch.abs(compiled - eager).max().item() < 1e-9
