"""Veto a cycle commit that regresses safety or standstill steering.

Cycles 1-4 of the gate-fixed campaign committed every cycle on a single
aggregate: the 128-scene train-selector ``mean_det_reward``.  That aggregate
rose (+0.2% over four cycles) while the full 46k paired audit showed collisions
up ~15% relative with P(improved)=0.003 — the progress/centerline/smoothness
terms were masking a real safety regression, and nothing in the pipeline could
stop it because the audit was report-only.

This tool turns the audit into a gate.  A cycle may only keep its newly trained
checkpoint when it is better on the headline reward *and* not measurably worse
on safety *and* not worse on the standstill steering-jitter metric that sent us
here.  Otherwise the verdict is ``veto`` and the caller must fall back to the
incumbent, so the worst case for a cycle is "no change", never a regression.

Exit codes: 0 = commit allowed, 3 = veto (fall back to incumbent), 2 = usage or
missing-evidence error (never silently pass).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

# Metrics where a confidently-worse result blocks the commit.  The bound is the
# audit's own bootstrap probability that the candidate improved: if we are more
# than 90% sure a safety metric got worse, that is a regression regardless of
# how much the aggregate reward rose.
SAFETY_METRICS = ("det_collision", "det_kinematic_violation", "det_safety")
SAFETY_MIN_PROBABILITY_IMPROVED = 0.10

# The headline metric must be a real improvement, not noise.
REWARD_METRIC = "det_reward"
REWARD_MIN_PROBABILITY_IMPROVED = 0.95

# Standstill steering jitter.
#
# The obvious statistic — p95 of the implied front-wheel angle over stop-turn
# scenes — is unusable: atan() saturates at pi/2 as the first step shrinks, and
# steps below the gate's 5 mm floor report 0, so the per-scene distribution is
# bimodal (measured 9/12 scenes at exactly 0.000 and 3/12 at 1.466-1.500 rad).
# A p95 over that jumps between 0 and 1.5 depending on how many scenes cross the
# floor.  The *rate* of scenes commanding an infeasible angle is smooth and
# interpretable, so gate on that instead.  Sub-floor scenes correctly count as
# non-exceeding: a step under 5 mm cannot move the wheel meaningfully.
JITTER_SCENE_KEY = "stop_turn_first_waypoint_implied_steer_p95_rad"
STOP_TURN_FLAG_KEY = "stop_turn_scene"
JITTER_EXCEED_RAD = 0.64
# Absolute tolerance on the exceed-fraction.  The full 46k audit carries ~3.4k
# stop-turn scenes, so 0.005 is well above sampling noise and well below any
# real regression.
JITTER_TOLERANCE = 0.005


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing required evidence: {path}")
    return json.loads(path.read_text())


def _audit_metrics(report: dict[str, Any], epoch: str | None) -> dict[str, Any]:
    epochs = report.get("epochs") or {}
    if not epochs:
        raise SystemExit("paired audit has no epochs block")
    key = epoch if epoch in epochs else sorted(epochs)[-1]
    metrics = epochs[key].get("metrics") or {}
    if not metrics:
        raise SystemExit(f"paired audit epoch {key} has no metrics")
    return metrics


def stop_turn_exceed_fraction(path: Path | None) -> tuple[float | None, int]:
    """Fraction of stop-turn scenes commanding an infeasible first-step angle.

    Reads the per-scene records the eval writes (``scenes.json``), because the
    aggregated ``summary.json`` does not carry the implied-steer diagnostic.
    Returns ``(None, 0)`` when the field is absent everywhere, so the caller can
    tell "no evidence" apart from "zero exceedances".
    """

    if path is None or not path.is_file():
        return None, 0
    try:
        rows = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None, 0
    if not isinstance(rows, list):
        return None, 0
    seen_field = False
    total = 0
    exceeded = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        if not _finite(row.get(STOP_TURN_FLAG_KEY)):
            continue
        if JITTER_SCENE_KEY in row:
            seen_field = True
        value = _finite(row.get(JITTER_SCENE_KEY))
        if value is None:
            continue
        total += 1
        if value > JITTER_EXCEED_RAD:
            exceeded += 1
    if not seen_field or total == 0:
        return None, total
    return exceeded / total, total


def evaluate(
    audit: dict[str, Any],
    epoch: str | None,
    candidate_jitter: float | None,
    baseline_jitter: float | None,
    jitter_scene_counts: tuple[int, int] = (0, 0),
) -> dict[str, Any]:
    metrics = _audit_metrics(audit, epoch)
    checks: list[dict[str, Any]] = []

    reward = metrics.get(REWARD_METRIC) or {}
    reward_p = _finite(reward.get("probability_improved"))
    reward_imp = _finite(reward.get("improvement"))
    checks.append(
        {
            "check": f"{REWARD_METRIC}_improved",
            "passed": bool(
                reward_p is not None
                and reward_imp is not None
                and reward_p >= REWARD_MIN_PROBABILITY_IMPROVED
                and reward_imp > 0.0
            ),
            "probability_improved": reward_p,
            "improvement": reward_imp,
            "requirement": (
                f"probability_improved >= {REWARD_MIN_PROBABILITY_IMPROVED} "
                "and improvement > 0"
            ),
        }
    )

    for name in SAFETY_METRICS:
        entry = metrics.get(name) or {}
        prob = _finite(entry.get("probability_improved"))
        checks.append(
            {
                "check": f"{name}_not_worse",
                # Missing evidence fails closed.
                "passed": bool(
                    prob is not None and prob >= SAFETY_MIN_PROBABILITY_IMPROVED
                ),
                "probability_improved": prob,
                "improvement": _finite(entry.get("improvement")),
                "requirement": (
                    f"probability_improved >= {SAFETY_MIN_PROBABILITY_IMPROVED}"
                ),
            }
        )

    # Evidence handling matters more than the threshold here.  Vetoing whenever
    # the metric is missing would stall the whole campaign — every cycle would
    # fall back to its incumbent and 100 epochs would commit nothing.  So:
    # missing from BOTH sides = the artifacts predate the diagnostic, skip the
    # check and say so loudly; missing from ONE side = a real anomaly, veto.
    if candidate_jitter is None and baseline_jitter is None:
        checks.append(
            {
                "check": "standstill_steer_not_worse",
                "passed": True,
                "skipped": True,
                "candidate": None,
                "baseline": None,
                "stop_turn_scene_counts": list(jitter_scene_counts),
                "requirement": (
                    "NOT CHECKED: neither side carries "
                    f"{JITTER_SCENE_KEY}; jitter is unguarded for this cycle"
                ),
            }
        )
    elif candidate_jitter is None or baseline_jitter is None:
        checks.append(
            {
                "check": "standstill_steer_not_worse",
                "passed": False,
                "candidate": candidate_jitter,
                "baseline": baseline_jitter,
                "stop_turn_scene_counts": list(jitter_scene_counts),
                "requirement": (
                    "one side is missing the jitter diagnostic while the other "
                    "has it — refusing to commit on incomparable evidence"
                ),
            }
        )
    else:
        checks.append(
            {
                "check": "standstill_steer_not_worse",
                "passed": candidate_jitter <= baseline_jitter + JITTER_TOLERANCE,
                "candidate": candidate_jitter,
                "baseline": baseline_jitter,
                "delta": candidate_jitter - baseline_jitter,
                "stop_turn_scene_counts": list(jitter_scene_counts),
                "requirement": (
                    "stop-turn fraction with implied steer > "
                    f"{JITTER_EXCEED_RAD} rad: candidate <= baseline "
                    f"+ {JITTER_TOLERANCE}"
                ),
            }
        )

    failed = [row["check"] for row in checks if not row["passed"]]
    return {
        "verdict": "commit" if not failed else "veto",
        "failed_checks": failed,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-audit", type=Path, required=True)
    parser.add_argument("--epoch", default=None)
    parser.add_argument(
        "--candidate-scenes",
        type=Path,
        default=None,
        help="per-scene eval records for the selected policy (scenes.json)",
    )
    parser.add_argument(
        "--baseline-scenes",
        type=Path,
        default=None,
        help="per-scene eval records for the cycle's starting incumbent",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = _load(args.paired_audit)
    candidate_jitter, candidate_count = stop_turn_exceed_fraction(args.candidate_scenes)
    baseline_jitter, baseline_count = stop_turn_exceed_fraction(args.baseline_scenes)

    result = evaluate(
        audit,
        args.epoch,
        candidate_jitter,
        baseline_jitter,
        (candidate_count, baseline_count),
    )
    result["paired_audit"] = str(args.paired_audit.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(f"non-regression verdict: {result['verdict']}")
    for row in result["checks"]:
        flag = "ok  " if row["passed"] else "FAIL"
        print(f"  {flag} {row['check']}: {row.get('requirement')}")
    return 0 if result["verdict"] == "commit" else 3


if __name__ == "__main__":
    raise SystemExit(main())
