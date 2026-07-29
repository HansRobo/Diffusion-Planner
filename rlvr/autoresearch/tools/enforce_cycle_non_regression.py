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

# Measured 2026-07-28 on the incumbent vs original DP: det_progress is -8.83e-3
# with probability_improved 0.0000 and a CI95 entirely below zero, while every
# reward axis sits at p~0.93.  Trading progress for smoothness is the only
# statistically decisive thing the campaign has done, the aggregate reward cannot
# see it (ep_score is 5/14 of pdm_quality yet total reward still rises), and a
# sluggish car is exactly what a driver feels.  So it gets its own axis.
#
# Advisory until the in-flight cycle lands: making it blocking mid-cycle would
# change the adjudication of the run that is currently measuring the existing
# settings.  Flip ADVISORY to False once that verdict is recorded.
PROGRESS_METRIC = "det_progress"
PROGRESS_MIN_PROBABILITY_IMPROVED = 0.10
PROGRESS_ADVISORY = True

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
# One-sided significance the regression must ALSO reach before it vetoes.  The
# tolerance above is 0.84 paired stderr on the real 46k audit, so on its own it
# is a coin flip; see jitter_discordance for the measurement.
JITTER_SIGNIFICANCE = 0.05


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

    rows = _stop_turn_exceed_rows(path)
    if rows is None:
        return None, 0
    total = len(rows)
    if total == 0:
        return None, total
    return sum(1 for _, over in rows if over) / total, total


def _stop_turn_exceed_rows(path: Path | None) -> list[tuple[str | None, bool]] | None:
    """Per-scene exceed indicators as (scene_path, exceeded) in file order.

    Split out of ``stop_turn_exceed_fraction`` so the paired discordance test
    below applies byte-identical population filtering.  The key stays optional
    because the fraction never needed one; only pairing does, and pairing drops
    keyless rows itself.  Returns None when the diagnostic is absent everywhere.
    """

    if path is None or not path.is_file():
        return None
    try:
        rows = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(rows, list):
        return None
    seen_field = False
    exceed: list[tuple[str | None, bool]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        # NB: _finite returns the number, and 0.0 is falsy -- so this skips both
        # an absent flag and an explicit 0.0.  Deliberate; do not "clean up".
        if not _finite(row.get(STOP_TURN_FLAG_KEY)):
            continue
        if JITTER_SCENE_KEY in row:
            seen_field = True
        value = _finite(row.get(JITTER_SCENE_KEY))
        if value is None:
            continue
        scene = row.get("scene_path")
        exceed.append(
            (scene if isinstance(scene, str) else None, value > JITTER_EXCEED_RAD)
        )
    if not seen_field:
        return None
    return exceed


def jitter_discordance(
    baseline_path: Path | None, candidate_path: Path | None
) -> tuple[int, int, int] | None:
    """Paired McNemar counts (baseline-only, candidate-only, n) for the jitter axis.

    The axis is a paired binary RATE over the same scenes, so its resolution is
    set by how many scenes disagree, not by sqrt(p(1-p)/n) of either fraction: a
    scene that exceeds on both sides contributes nothing to the difference.
    Measured 2026-07-28 (`<scratchpad>/price_standstill_steer_axis.py`) on two
    evals of IDENTICAL weights, this axis moves by -0.003187 -- 64% of the whole
    0.005 tolerance -- and on the real cycle01 audit the paired stderr is
    0.005965, so the tolerance is only 0.84 stderr and vetoes on a true null
    20% of the time.  Cycle01's own +0.006373 is z=+1.07, i.e. not evidence of
    anything.  Hence the significance requirement in ``evaluate``.
    """

    base_rows = _stop_turn_exceed_rows(baseline_path)
    cand_rows = _stop_turn_exceed_rows(candidate_path)
    if base_rows is None or cand_rows is None:
        return None
    # Pairing needs a stable identity, so keyless rows are dropped here rather
    # than matched by position -- two evals need not enumerate scenes in the
    # same order, and a positional match would silently pair different scenes.
    base = {s: over for s, over in base_rows if s is not None}
    cand = {s: over for s, over in cand_rows if s is not None}
    shared = base.keys() & cand.keys()
    if not shared:
        return None
    baseline_only = sum(1 for s in shared if base[s] and not cand[s])
    candidate_only = sum(1 for s in shared if cand[s] and not base[s])
    return baseline_only, candidate_only, len(shared)


def mcnemar_p_one_sided(pair: tuple[int, int, int] | None) -> float | None:
    """P(this many scenes newly exceed, or more | the policy changed nothing).

    Exact binomial: conditional on the discordant pairs, each is a fair coin
    under the null, so candidate_only ~ Binomial(discordant, 1/2).  Exact rather
    than normal-approximate because the discordant count can be small on a
    narrow corpus, which is exactly when the approximation misleads.
    """

    if pair is None:
        return None
    baseline_only, candidate_only, _ = pair
    discordant = baseline_only + candidate_only
    if discordant <= 0:
        # No scene changed verdict, so there is no evidence of a regression.
        return 1.0
    tail = sum(
        math.comb(discordant, k) for k in range(candidate_only, discordant + 1)
    )
    return tail / (2**discordant)


def evaluate(
    audit: dict[str, Any],
    epoch: str | None,
    candidate_jitter: float | None,
    baseline_jitter: float | None,
    jitter_scene_counts: tuple[int, int] = (0, 0),
    jitter_pair: tuple[int, int, int] | None = None,
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

    progress = metrics.get(PROGRESS_METRIC) or {}
    progress_p = _finite(progress.get("probability_improved"))
    checks.append(
        {
            "check": f"{PROGRESS_METRIC}_not_worse",
            "passed": bool(
                progress_p is not None
                and progress_p >= PROGRESS_MIN_PROBABILITY_IMPROVED
            ),
            "advisory": PROGRESS_ADVISORY,
            "probability_improved": progress_p,
            "improvement": _finite(progress.get("improvement")),
            "requirement": (
                f"probability_improved >= {PROGRESS_MIN_PROBABILITY_IMPROVED}"
                + (" (ADVISORY — reported, does not veto)" if PROGRESS_ADVISORY else "")
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
        # Two conditions, not one.  The absolute tolerance says the regression is
        # big enough to care about; the McNemar test says the evidence supports
        # it at all.  Vetoing on the tolerance alone rejects a true null 20% of
        # the time (see jitter_discordance), and it already threw away cycle01 on
        # a z=+1.07 reading.  A genuine regression is unaffected: it has to clear
        # the tolerance either way, and at 0.006 stderr anything truly large is
        # significant automatically.  Missing pairing evidence fails closed onto
        # the old tolerance-only rule, matching how this file treats absent
        # diagnostics everywhere else.
        over_tolerance = candidate_jitter > baseline_jitter + JITTER_TOLERANCE
        p_one_sided = mcnemar_p_one_sided(jitter_pair)
        significant = p_one_sided is None or p_one_sided < JITTER_SIGNIFICANCE
        checks.append(
            {
                "check": "standstill_steer_not_worse",
                "passed": not (over_tolerance and significant),
                "candidate": candidate_jitter,
                "baseline": baseline_jitter,
                "delta": candidate_jitter - baseline_jitter,
                "over_tolerance": over_tolerance,
                "mcnemar_p_one_sided": p_one_sided,
                "mcnemar_discordant": (
                    None if jitter_pair is None else list(jitter_pair[:2])
                ),
                "stop_turn_scene_counts": list(jitter_scene_counts),
                "requirement": (
                    "stop-turn fraction with implied steer > "
                    f"{JITTER_EXCEED_RAD} rad: vetoes only if candidate > "
                    f"baseline + {JITTER_TOLERANCE} AND one-sided McNemar "
                    f"p < {JITTER_SIGNIFICANCE} (tolerance alone is 0.84 paired "
                    "stderr, so it cannot resolve its own threshold)"
                ),
            }
        )

    failed = [
        row["check"]
        for row in checks
        if not row["passed"] and not row.get("advisory")
    ]
    return {
        "verdict": "commit" if not failed else "veto",
        "failed_checks": failed,
        "advisory_failed_checks": [
            row["check"]
            for row in checks
            if not row["passed"] and row.get("advisory")
        ],
        "checks": checks,
    }


def _resolve_jitter_sources(
    audit: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    """Pick the per-scene records the standstill check should read.

    The reward and safety checks come from the paired audit, which evaluates the
    ALPHA-SCALED checkpoint a cycle actually commits.  The jitter check reads
    ``--candidate-scenes``, and the caller can only point that at train_awr's
    per-epoch eval -- i.e. at the *unscaled* direction, historically a 20x
    larger step than the commit.  Cycle 1 of the jitterfix campaign was vetoed
    on exactly that mismatch: the direction moved the exceed-fraction +0.0055
    against a +0.005 tolerance, while the a0p05 commit under gate was a
    twentieth of that step.

    So prefer the committed policy's own records whenever they carry the
    diagnostic AND a same-evaluator baseline does too -- comparing one
    evaluator's candidate against another's baseline would trade a known step
    mismatch for an unknown evaluator bias.  Either way, record which artifact
    each side came from and whether it is the policy being committed, so a
    verdict can never again read as if it measured the commit when it did not.
    """

    committed = audit.get("candidate")
    committed_path = Path(committed) if isinstance(committed, str) else None
    audit_baseline = audit.get("baseline")
    audit_baseline_path = (
        Path(audit_baseline) if isinstance(audit_baseline, str) else None
    )
    sources: dict[str, Any] = {
        "candidate": args.candidate_scenes,
        "baseline": args.baseline_scenes,
        "committed_policy_scenes": committed_path,
        "candidate_is_committed_policy": False,
        "note": (
            "jitter measured on --candidate-scenes, which is the unscaled "
            "direction rather than the committed policy; the reward and safety "
            "checks measure the commit"
        ),
    }
    if committed_path is None or audit_baseline_path is None:
        return sources
    if stop_turn_exceed_fraction(committed_path)[0] is None:
        return sources
    if stop_turn_exceed_fraction(audit_baseline_path)[0] is None:
        sources["note"] = (
            "the committed policy carries the jitter diagnostic but the audit "
            "baseline does not; regenerate the baseline eval to gate the commit "
            "itself instead of the unscaled direction"
        )
        return sources
    sources.update(
        {
            "candidate": committed_path,
            "baseline": audit_baseline_path,
            "candidate_is_committed_policy": True,
            "note": "jitter measured on the committed policy, same evaluator both sides",
        }
    )
    return sources


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
    jitter_sources = _resolve_jitter_sources(audit, args)
    candidate_jitter, candidate_count = stop_turn_exceed_fraction(
        jitter_sources["candidate"]
    )
    baseline_jitter, baseline_count = stop_turn_exceed_fraction(
        jitter_sources["baseline"]
    )

    result = evaluate(
        audit,
        args.epoch,
        candidate_jitter,
        baseline_jitter,
        (candidate_count, baseline_count),
        jitter_discordance(jitter_sources["baseline"], jitter_sources["candidate"]),
    )
    result["paired_audit"] = str(args.paired_audit.resolve())
    result["jitter_evidence"] = {
        key: (None if value is None else str(value))
        for key, value in jitter_sources.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(f"non-regression verdict: {result['verdict']}")
    for row in result["checks"]:
        flag = "ok  " if row["passed"] else ("warn" if row.get("advisory") else "FAIL")
        print(f"  {flag} {row['check']}: {row.get('requirement')}")
    return 0 if result["verdict"] == "commit" else 3


if __name__ == "__main__":
    raise SystemExit(main())
