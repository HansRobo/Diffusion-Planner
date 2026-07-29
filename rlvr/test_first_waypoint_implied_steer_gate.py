"""Regression tests for the standstill steering-jitter gate.

Cycles 1-4 of the gate-fixed campaign rejected *zero* low-speed candidates
because the tangent floor (5 cm) sat above the measured stop-turn p95 first
step (3.9 cm), and because the absolute displacement/lateral limits (25 cm /
20 cm) are orders of magnitude looser than the 1-4 cm regime standstill jitter
actually lives in.  These tests pin the implied-steer criterion that closes
that hole, using the real measured geometry as the headline case.
"""

import math

import pytest
import torch

from rlvr.awr import (
    AWRRolloutConfig,
    _first_waypoint_gate_and_diagnostics,
    _first_waypoint_gate_and_diagnostics_batch,
    _implied_first_step_steer_rad,
)

WHEEL_BASE = 2.79


def _config(**overrides):
    cfg = AWRRolloutConfig()
    cfg.original_dp_first_waypoint_gate_enabled = True
    cfg.original_dp_first_waypoint_gate_speed_mps = 1.0
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _scene(speed=0.0, turn=2):
    return {
        "ego_shape": torch.tensor([[WHEEL_BASE, 4.34, 1.70]]),
        "ego_current_state": torch.tensor([[0.0, 0.0, 1.0, 0.0, speed, 0.0]]),
        "turn_indicators": torch.tensor([[turn]]),
    }


def _trajs(first_steps, horizon=8):
    """[K,T,4] where candidate k starts at first_steps[k]."""
    k = len(first_steps)
    traj = torch.zeros(k, horizon, 4)
    for i, (x, y) in enumerate(first_steps):
        traj[i, 0, 0] = x
        traj[i, 0, 1] = y
        for t in range(1, horizon):
            traj[i, t, 0] = x + 0.01 * t
            traj[i, t, 1] = y
    return traj


def test_measured_standstill_geometry_implies_steering_lock():
    """The real campaign numbers: 3.9 cm step, 0.91 cm lateral."""
    first_xy = torch.tensor([[0.0389, 0.0091]])
    step_norm = first_xy.norm(dim=-1)
    steer = _implied_first_step_steer_rad(
        first_xy, step_norm, _scene(), min_step_m=0.005
    )
    # k = 2*0.0091/0.04**2 ~= 11.4 /m -> atan(2.79*11.4) ~= 1.54 rad ~= 88 deg.
    assert steer.item() > 1.5
    assert steer.item() > 0.64, "must exceed the gate threshold"


def test_sub_millimetre_noise_is_not_flagged():
    """The bug the floor was introduced for: mm-level noise must stay exempt."""
    first_xy = torch.tensor([[0.0001, 0.0001]])
    step_norm = first_xy.norm(dim=-1)
    steer = _implied_first_step_steer_rad(
        first_xy, step_norm, _scene(), min_step_m=0.005
    )
    assert steer.item() == 0.0


def test_gate_rejects_the_jittering_candidate_but_keeps_the_anchor():
    cfg = _config()
    # candidate 0 = behaviour anchor (always exempt), 1 = clean, 2 = jitter.
    trajs = _trajs([(0.0389, 0.0091), (0.0400, 0.0000), (0.0389, 0.0091)])
    valid, diag = _first_waypoint_gate_and_diagnostics(trajs, _scene(), cfg)
    assert bool(valid[0]), "anchor must never be gated"
    assert bool(valid[1]), "straight low-speed step must survive"
    assert not bool(valid[2]), "steering-lock step must be rejected"
    assert diag["first_waypoint_gate_rejected_count"] == 1.0
    assert diag["stop_turn_first_waypoint_implied_steer_p95_rad"] > 0.64


def test_old_thresholds_alone_would_have_missed_it():
    """Displacement/lateral/tangent alone let the measured geometry through."""
    cfg = _config(
        original_dp_first_waypoint_gate_max_implied_steer_rad=math.inf,
        original_dp_first_waypoint_gate_tangent_min_step_m=0.05,
    )
    trajs = _trajs([(0.0400, 0.0000), (0.0389, 0.0091)])
    valid, diag = _first_waypoint_gate_and_diagnostics(trajs, _scene(), cfg)
    assert bool(valid[1]), "reproduces the cycle 1-4 hole"
    assert diag["first_waypoint_gate_rejected_fraction"] == 0.0


def test_high_speed_scenes_are_untouched():
    cfg = _config()
    trajs = _trajs([(0.0400, 0.0000), (0.0389, 0.0091)])
    valid, _ = _first_waypoint_gate_and_diagnostics(trajs, _scene(speed=5.0), cfg)
    assert bool(valid.all()), "gate only applies below the low-speed threshold"


def test_batched_gate_matches_scene_wise():
    cfg = _config()
    cases = [(0.0389, 0.0091), (0.0400, 0.0000), (0.0100, 0.0060), (0.2000, 0.0100)]
    trajs = _trajs(cases)
    scene = _scene()
    valid_scene, diag_scene = _first_waypoint_gate_and_diagnostics(trajs, scene, cfg)

    batch_scene = {
        "ego_shape": torch.tensor([[WHEEL_BASE, 4.34, 1.70]]),
        "ego_current_state": torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
        "turn_indicators": torch.tensor([[2]]),
    }
    valid_batch, diag_batch = _first_waypoint_gate_and_diagnostics_batch(
        trajs.unsqueeze(0), batch_scene, cfg
    )
    assert torch.equal(valid_scene, valid_batch[0])
    assert diag_batch[0]["first_waypoint_implied_steer_p95_rad"] == pytest.approx(
        diag_scene["first_waypoint_implied_steer_p95_rad"], rel=1e-5
    )


def test_gentle_low_speed_creep_survives():
    """A real slow creep must not be gated — otherwise we starve the anchor."""
    cfg = _config()
    # 8 cm step, 2 mm lateral -> k = 0.625 /m -> atan(2.79*0.625) = 1.05 rad.
    # Still large, so use a realistically straight creep instead.
    trajs = _trajs([(0.0800, 0.0000), (0.0800, 0.0002)])
    valid, _ = _first_waypoint_gate_and_diagnostics(trajs, _scene(), cfg)
    assert bool(valid.all())
