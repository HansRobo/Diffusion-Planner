"""Drift guard for ``planner_metrics.compute_subscores_batch``.

``compute_subscores_batch`` (used by the validation loop) re-runs the same input
marshalling + per-subscore computation as ``rlvr.reward.compute_reward_batch``
(used by RLVR), but stops before reward shaping. To guarantee the validation
metric never silently diverges from what the reward is built on, this test pins
``compute_subscores_batch`` against ``compute_reward_batch`` field-by-field on the
shared golden scenarios. If someone edits the marshalling/subscore call in one
place but not the other, this fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from planner_metrics import compute_subscores_batch  # noqa: E402
from rlvr.reward import compute_reward_batch  # noqa: E402
from rlvr.test_reward_golden import _SCENARIOS  # noqa: E402  (reuse golden scenarios)
from rlvr.train_awr import _expert_reward_is_hard_safe  # noqa: E402

ABS_TOL = 1e-6
REL_TOL = 1e-6

# subscore-dict key -> RewardBreakdown attribute it must equal.
_FLOAT_FIELDS = {
    "safety": "safety",
    "progress": "progress",  # adjusted_progress == progress_scores (off_road==0)
    "comfort": "smoothness",
    "feasibility": "feasibility",
    "centerline": "centerline",
    "red_light": "red_light",
    "off_road_fraction": "off_road_fraction",
    "rb_near_penalty": "rb_near_penalty",
    "rb_wide_penalty": "rb_wide_penalty",
    "rb_min_dist": "rb_min_dist",
    "lane_near_frac": "lane_near_frac",
    "lane_wide_frac": "lane_wide_frac",
    "sc_near_penalty": "sc_near_penalty",
    "sc_wide_penalty": "sc_wide_penalty",
    "sc_cont_penalty": "sc_cont_penalty",
    "sc_min_dist": "sc_min_dist",
}
# raw 0/1 gate key -> RewardBreakdown crossing bool (crossing == gate < 0.5).
_GATE_FIELDS = {
    "rb_crossing_gate": "rb_crossing",
    "lane_crossing_gate": "lane_crossing",
    "sc_crossing_gate": "static_crossing",
}


@pytest.mark.parametrize("name", sorted(_SCENARIOS))
def test_subscores_match_reward_batch(name: str) -> None:
    ego, data, cfg = _SCENARIOS[name]
    subs = compute_subscores_batch(ego, data, cfg)
    breakdowns = compute_reward_batch(ego, data, cfg)
    assert len(breakdowns) == ego.shape[0]

    for i, bd in enumerate(breakdowns):
        for key, attr in _FLOAT_FIELDS.items():
            got = float(subs[key][i])
            exp = float(getattr(bd, attr))
            assert got == pytest.approx(exp, abs=ABS_TOL, rel=REL_TOL), (
                f"{name}[{i}] {key} vs RewardBreakdown.{attr}: {got!r} != {exp!r}"
            )
        for key, attr in _GATE_FIELDS.items():
            got = bool(subs[key][i] < 0.5)  # gate < 0.5 == crossing
            exp = bool(getattr(bd, attr))
            assert got == exp, f"{name}[{i}] {key}: {got} != {exp}"

        # kinematic gate (subscore is the 0/1 gate; RewardBreakdown stores the
        # violated bool = gate < 0.5)
        got_kin = bool(subs["kinematic_gate"][i] < 0.5)
        assert got_kin == bool(bd.kinematic_violated), f"{name}[{i}] kinematic"

        # collision step (list passthrough)
        assert subs["collision_step"][i] == bd.collision_step, f"{name}[{i}] collision_step"

        # scene-level stopped-neighbor count
        assert int(subs["sc_n_stopped"][i]) == int(bd.sc_n_stopped), f"{name}[{i}] sc_n_stopped"


@pytest.mark.parametrize("name", sorted(_SCENARIOS))
def test_first_hard_gate_step_is_the_step_resolved_safety_verdict(name: str) -> None:
    """``first_hard_gate_step is None`` must mean exactly "hard safe".

    ``_first_hard_gate_steps`` duplicates the enabled/disabled gate predicates
    that ``_expert_reward_is_hard_safe`` applies, because the step indices only
    exist inside ``compute_reward_batch``.  Duplicated predicates drift, so
    assert the equivalence on every golden scenario instead of trusting it --
    a step-scoped anchor verdict is only sound if collapsing it back to the
    full horizon reproduces the flag the pipeline already ships.
    """

    ego, data, cfg = _SCENARIOS[name]
    subs = compute_subscores_batch(ego, data, cfg)
    breakdowns = compute_reward_batch(ego, data, cfg)

    for i, bd in enumerate(breakdowns):
        safe = _expert_reward_is_hard_safe(bd, cfg)
        assert (bd.first_hard_gate_step is None) == safe, (
            f"{name}[{i}] first_hard_gate_step={bd.first_hard_gate_step} "
            f"disagrees with _expert_reward_is_hard_safe={safe}"
        )
        # And when it does fire, it fires at the earliest configured gate.
        steps = [subs["collision_step"][i]]
        if cfg.rb_gate_enabled:
            steps.append(subs["rb_crossing_steps"][i])
        if cfg.lane_gate_enabled:
            steps.append(subs["lane_crossing_steps"][i])
        if cfg.static_collision_enabled and cfg.sc_gate_enabled:
            steps.append(subs["sc_crossing_steps"][i])
        if float(subs["red_light"][i]) < -0.5 or float(subs["kinematic_gate"][i]) < 0.5:
            steps.append(0)
        fired = [int(step) for step in steps if step is not None]
        assert bd.first_hard_gate_step == (min(fired) if fired else None), (
            f"{name}[{i}] expected {min(fired) if fired else None}, "
            f"got {bd.first_hard_gate_step}"
        )
        # Step 0 is the first waypoint, so a gate there must NOT be treated as
        # safe over a one-step horizon.  This is why the consumer's test is
        # ``step >= consumed`` and never ``step > consumed``.
        if bd.first_hard_gate_step == 0:
            assert not safe


def test_returns_per_trajectory_tensors() -> None:
    """Continuous subscores come back as (N,) tensors for batch-mean logging."""
    name = "safe_onroad_multi"
    ego, data, cfg = _SCENARIOS[name]
    subs = compute_subscores_batch(ego, data, cfg)
    n = ego.shape[0]
    for key in ("safety", "ttc", "progress", "comfort", "centerline", "red_light"):
        assert isinstance(subs[key], torch.Tensor)
        assert subs[key].shape == (n,), f"{key} shape {tuple(subs[key].shape)} != ({n},)"
