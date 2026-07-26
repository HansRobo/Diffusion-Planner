"""Contract tests for the per-cycle non-regression veto.

Two failure modes this pins down, both found on 2026-07-26 before the veto had
ever run for real:

1. The jitter statistic was read from the aggregated ``summary.json``, which does
   not carry the implied-steer diagnostic at all — only the per-scene
   ``scenes.json`` does.  Combined with fail-closed handling, that meant *every*
   cycle would have been vetoed and a 100-epoch campaign would have committed
   nothing.
2. p95 of the implied steer is unusable as the statistic: atan() saturates at
   pi/2 and sub-floor steps report 0, so the distribution is bimodal.  The gate
   is on the *fraction* of stop-turn scenes above an infeasible angle.
"""

import json

from rlvr.autoresearch.tools.enforce_cycle_non_regression import (
    JITTER_SCENE_KEY,
    STOP_TURN_FLAG_KEY,
    evaluate,
    stop_turn_exceed_fraction,
)


def _audit(**overrides):
    def metric(prob, improvement):
        return {"probability_improved": prob, "improvement": improvement}

    metrics = {
        "det_reward": metric(1.0, 0.002),
        "det_collision": metric(0.5, 0.0),
        "det_kinematic_violation": metric(0.5, 0.0),
        "det_safety": metric(0.5, 0.0),
    }
    metrics.update(overrides)
    return {"epochs": {"8": {"metrics": metrics}}}


def _scenes(path, values, stop_turn=True, include_key=True):
    rows = []
    for value in values:
        row = {STOP_TURN_FLAG_KEY: 1.0 if stop_turn else 0.0}
        if include_key:
            row[JITTER_SCENE_KEY] = value
        rows.append(row)
    path.write_text(json.dumps(rows))
    return path


def test_exceed_fraction_counts_only_stop_turn_scenes(tmp_path):
    path = tmp_path / "scenes.json"
    rows = [
        {STOP_TURN_FLAG_KEY: 1.0, JITTER_SCENE_KEY: 1.5},
        {STOP_TURN_FLAG_KEY: 1.0, JITTER_SCENE_KEY: 0.0},
        # Non-stop-turn scenes must not dilute the rate.
        {STOP_TURN_FLAG_KEY: 0.0, JITTER_SCENE_KEY: 1.5},
    ]
    path.write_text(json.dumps(rows))
    fraction, count = stop_turn_exceed_fraction(path)
    assert count == 2
    assert fraction == 0.5


def test_sub_floor_zeros_count_as_feasible(tmp_path):
    """A step under the gate's 5 mm floor reports 0 and cannot move the wheel."""
    path = _scenes(tmp_path / "s.json", [0.0, 0.0, 0.0, 1.5])
    fraction, count = stop_turn_exceed_fraction(path)
    assert count == 4
    assert fraction == 0.25


def test_missing_field_is_reported_as_no_evidence(tmp_path):
    path = _scenes(tmp_path / "s.json", [1.5, 1.5], include_key=False)
    fraction, _ = stop_turn_exceed_fraction(path)
    assert fraction is None


def test_absent_on_both_sides_skips_instead_of_vetoing():
    """The campaign-killer: fail-closed here would veto every single cycle."""
    result = evaluate(_audit(), "8", None, None)
    assert result["verdict"] == "commit"
    row = next(r for r in result["checks"] if r["check"] == "standstill_steer_not_worse")
    assert row["passed"] is True
    assert row["skipped"] is True
    assert "NOT CHECKED" in row["requirement"]


def test_absent_on_one_side_only_vetoes():
    for candidate, baseline in ((None, 0.5), (0.5, None)):
        result = evaluate(_audit(), "8", candidate, baseline)
        assert result["verdict"] == "veto"
        assert "standstill_steer_not_worse" in result["failed_checks"]


def test_jitter_increase_vetoes_even_when_reward_improves():
    result = evaluate(_audit(), "8", 0.70, 0.65)
    assert result["verdict"] == "veto"
    assert result["failed_checks"] == ["standstill_steer_not_worse"]


def test_jitter_within_tolerance_commits():
    result = evaluate(_audit(), "8", 0.6531, 0.6500)
    assert result["verdict"] == "commit"


def test_confident_safety_regression_vetoes_despite_reward_gain():
    """The cycle-4 shape: reward up, collisions confidently worse."""
    result = evaluate(
        _audit(det_collision={"probability_improved": 0.003, "improvement": -0.0002}),
        "8",
        0.65,
        0.65,
    )
    assert result["verdict"] == "veto"
    assert "det_collision_not_worse" in result["failed_checks"]


def test_reward_must_be_a_real_improvement():
    result = evaluate(
        _audit(det_reward={"probability_improved": 0.6, "improvement": 0.0001}),
        "8",
        0.65,
        0.65,
    )
    assert result["verdict"] == "veto"
    assert "det_reward_improved" in result["failed_checks"]
