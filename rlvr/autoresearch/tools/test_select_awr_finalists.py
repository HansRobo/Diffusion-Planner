from __future__ import annotations

from rlvr.autoresearch.tools.select_awr_finalists import select_finalists


def _metric(base: float, candidate: float, ci: tuple[float, float], *, direction: str = "higher"):
    improvement = candidate - base if direction == "higher" else base - candidate
    return {
        "baseline_mean": base,
        "candidate_mean": candidate,
        "improvement": improvement,
        "improvement_ci95": list(ci),
        "relative_delta": (candidate - base) / abs(base) if base else 0.0,
        "paired_scene_count": 100,
        "regressed_scene_fraction": 0.03,
    }


def _epoch(gain: float, ci_low: float, *, ade_delta: float = 0.0, new_collision: bool = False):
    collision = _metric(0.01, 0.02 if new_collision else 0.01, (-0.02, 0.02), direction="lower")
    collision["regressed_scene_fraction"] = 0.01 if new_collision else 0.0
    return {
        "scene_count": 100,
        "metrics": {
            "det_reward": _metric(0.80, 0.80 + gain, (ci_low, gain + 0.01)),
            "det_collision": collision,
            "det_rb_crossing": _metric(0.02, 0.01, (0.0, 0.02), direction="lower"),
            "det_kinematic_violation": _metric(0.0, 0.0, (0.0, 0.0), direction="lower"),
            "ade": _metric(1.0, 1.0 + ade_delta, (-0.5, 0.5), direction="lower"),
            "fde": _metric(2.0, 2.0 + ade_delta, (-0.5, 0.5), direction="lower"),
        },
    }


def test_ranks_by_reward_lcb_not_ade_or_new_event_veto() -> None:
    report = {
        "epochs": {
            "1": _epoch(0.03, 0.02, ade_delta=0.40, new_collision=True),
            "2": _epoch(0.02, 0.01, ade_delta=0.00, new_collision=False),
        }
    }
    result = select_finalists(report, expected_scene_count=100, max_finalists=2)
    assert result["finalists"][0]["epoch"] == 1
    assert result["promotion_recommendation"]["epoch"] == 1
    assert result["finalists"][0]["diagnostics"]["ade"]["relative_delta"] > 0
    assert result["finalists"][0]["diagnostics"]["det_collision"]["new_paired_event_count"] == 1


def test_positive_mean_without_positive_ci_is_not_promoted() -> None:
    report = {"epochs": {"1": _epoch(0.03, -0.001)}}
    result = select_finalists(report, expected_scene_count=100)
    assert result["finalists"][0]["epoch"] == 1
    assert result["promotion_recommendation"] is None


def test_scene_contract_is_required_for_promotion() -> None:
    report = {"epochs": {"1": _epoch(0.03, 0.01)}}
    result = select_finalists(report, expected_scene_count=101)
    assert result["promotion_recommendation"] is None
    assert "scene count" in result["finalists"][0]["promotion_reasons"][0]
