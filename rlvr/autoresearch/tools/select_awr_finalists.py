#!/usr/bin/env python3
"""Rank AWR checkpoints by the paper-aligned mean driving score.

HDP's released RL implementation validates with the mean PDM score.  NAVSIM
similarly defines one per-scene PDM score (hard safety multipliers times a
weighted quality score) and averages it over the evaluation set.  We therefore
use ``det_reward`` -- the local PDM-compatible per-scene score -- as the single
checkpoint-selection objective.

ADE/FDE and individual safety/comfort components remain visible diagnostics.
They are intentionally not independent promotion vetoes: doing so would turn a
single driving objective into an undocumented conjunction of unrelated open-
and pseudo-closed-loop metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PRIMARY_METRIC = "det_reward"
HARD_DIAGNOSTICS = ("det_collision", "det_rb_crossing", "det_kinematic_violation")
RETENTION_DIAGNOSTICS = ("ade", "fde")
OTHER_DIAGNOSTICS = ("safe_candidate_count", "smoothness")


def _metric(epoch: dict[str, Any], name: str) -> dict[str, Any] | None:
    value = epoch.get("metrics", {}).get(name)
    return value if isinstance(value, dict) else None


def _ci(epoch: dict[str, Any], name: str) -> list[float]:
    row = _metric(epoch, name)
    values = row.get("improvement_ci95") if row is not None else None
    if not isinstance(values, list) or len(values) != 2:
        return [float("-inf"), float("-inf")]
    return [float(values[0]), float(values[1])]


def _candidate_mean(epoch: dict[str, Any], name: str) -> float | None:
    row = _metric(epoch, name)
    if row is None or row.get("candidate_mean") is None:
        return None
    return float(row["candidate_mean"])


def _baseline_mean(epoch: dict[str, Any], name: str) -> float | None:
    row = _metric(epoch, name)
    if row is None or row.get("baseline_mean") is None:
        return None
    return float(row["baseline_mean"])


def _new_event_count(epoch: dict[str, Any], name: str) -> int | None:
    row = _metric(epoch, name)
    if row is None:
        return None
    fraction = row.get("regressed_scene_fraction")
    if fraction is None:
        return None
    paired = int(row.get("paired_scene_count", epoch.get("scene_count", 0)))
    return int(round(float(fraction) * paired))


def _diagnostic(epoch: dict[str, Any], name: str) -> dict[str, Any] | None:
    row = _metric(epoch, name)
    if row is None:
        return None
    result: dict[str, Any] = {
        "baseline_mean": _baseline_mean(epoch, name),
        "candidate_mean": _candidate_mean(epoch, name),
        "improvement": float(row["improvement"]) if row.get("improvement") is not None else None,
        "improvement_ci95": _ci(epoch, name),
        "relative_delta": float(row["relative_delta"]) if row.get("relative_delta") is not None else None,
    }
    if name in HARD_DIAGNOSTICS:
        result["new_paired_event_count"] = _new_event_count(epoch, name)
    return result


def _summary(epoch_number: int, epoch: dict[str, Any], expected_scene_count: int) -> dict[str, Any]:
    reward = _metric(epoch, PRIMARY_METRIC)
    if reward is None:
        raise ValueError(f"epoch {epoch_number} is missing {PRIMARY_METRIC}")
    scene_count = int(epoch.get("scene_count", 0))
    ci95 = _ci(epoch, PRIMARY_METRIC)
    scene_contract_ok = expected_scene_count <= 0 or scene_count == expected_scene_count
    promotion_reasons: list[str] = []
    if not scene_contract_ok:
        promotion_reasons.append(
            f"scene count {scene_count} does not match required {expected_scene_count}"
        )
    if ci95[0] <= 0.0:
        promotion_reasons.append(
            "paired bootstrap CI95 for mean det_reward improvement does not exclude zero"
        )

    baseline_mean = float(reward["baseline_mean"])
    candidate_mean = float(reward["candidate_mean"])
    diagnostics = {
        name: value
        for name in (*HARD_DIAGNOSTICS, *RETENTION_DIAGNOSTICS, *OTHER_DIAGNOSTICS)
        if (value := _diagnostic(epoch, name)) is not None
    }
    return {
        "epoch": epoch_number,
        "scene_count": scene_count,
        "scene_contract_ok": scene_contract_ok,
        "baseline_dataset_score": 100.0 * baseline_mean,
        "candidate_dataset_score": 100.0 * candidate_mean,
        "score_gain_points": 100.0 * float(reward["improvement"]),
        "score_gain_ci95_points": [100.0 * ci95[0], 100.0 * ci95[1]],
        "reward_improvement": float(reward["improvement"]),
        "reward_ci95": ci95,
        "reward_ci_low": ci95[0],
        "promotion_reasons": promotion_reasons,
        "diagnostics": diagnostics,
    }


def select_finalists(
    report: dict[str, Any], *, expected_scene_count: int = 0, max_finalists: int = 3
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for key, epoch in report.get("epochs", {}).items():
        try:
            epoch_number = int(key)
        except (TypeError, ValueError):
            continue
        if _metric(epoch, PRIMARY_METRIC) is not None:
            rows.append(_summary(epoch_number, epoch, expected_scene_count))
    rows.sort(key=lambda row: row["epoch"])
    if not rows:
        raise RuntimeError("paired report contains no usable epochs")

    # The lower confidence bound is the robust estimate of aggregate gain.
    # Mean gain and the earlier epoch are deterministic tie-breakers.
    ranked = sorted(
        rows,
        key=lambda row: (
            row["scene_contract_ok"],
            row["reward_ci_low"],
            row["reward_improvement"],
            -row["epoch"],
        ),
        reverse=True,
    )
    finalists = []
    for index, row in enumerate(ranked[: max(1, int(max_finalists))]):
        finalists.append(
            {
                "role": "aggregate_score_champion" if index == 0 else "aggregate_score_runner_up",
                **row,
            }
        )

    promotable = [row for row in ranked if not row["promotion_reasons"]]
    promotion = promotable[0] if promotable else None
    return {
        "selection_contract": {
            "name": "HDP_NAVSIM_mean_score",
            "primary_metric": PRIMARY_METRIC,
            "per_scene_score": (
                "no-collision × on-road × kinematic-feasible × "
                "(5×TTC + 5×progress + 2×lane + 2×comfort) / 14"
            ),
            "dataset_score": "100 × arithmetic mean of the per-scene score",
            "checkpoint_ranking": (
                "descending lower bound of the paired bootstrap 95% CI for mean score gain; "
                "mean gain is the tie-breaker"
            ),
            "promotion": (
                "complete evaluation scene contract and paired bootstrap CI95 lower bound > 0"
            ),
            "diagnostics_not_vetoes": [
                *HARD_DIAGNOSTICS,
                *RETENTION_DIAGNOSTICS,
                *OTHER_DIAGNOSTICS,
            ],
            "full_validation_required": True,
            "references": [
                "https://github.com/ziqipang/Hyper-Diffusion-Planner",
                "https://github.com/autonomousvision/navsim/blob/main/docs/metrics.md",
                "https://arxiv.org/abs/2601.12901",
                "https://arxiv.org/abs/2502.19908",
            ],
        },
        "expected_scene_count": int(expected_scene_count),
        "promotion_recommendation": promotion,
        # Compatibility alias for the already-running finalization shell.  It
        # now carries the aggregate-score contract, not the removed strict gate.
        "strict_recommendation": promotion,
        "finalists": finalists,
        "all_epochs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paired_report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-scene-count", type=int, default=0)
    parser.add_argument("--max-finalists", type=int, default=3)
    # Accepted only so old launchers fail safely after this policy change.
    parser.add_argument("--retention-tolerance", type=float, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    report = json.loads(args.paired_report.read_text())
    payload = {
        "source_report": str(args.paired_report.resolve()),
        **select_finalists(
            report,
            expected_scene_count=int(args.expected_scene_count),
            max_finalists=int(args.max_finalists),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "promotion": payload["promotion_recommendation"],
                "finalists": payload["finalists"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
