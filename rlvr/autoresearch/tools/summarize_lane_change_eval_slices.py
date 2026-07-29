#!/usr/bin/env python3
"""Join a lane-transition audit to paired planner evaluation rows.

This is a diagnostic slice report, not a lane-change ground-truth benchmark.
It keeps the independent logged-future detector's labels separate from the
formal reward and reports source-to-checkpoint changes with paired bootstrap
intervals and hard-event transitions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


HIGHER_IS_BETTER = (
    "det_reward",
    "best_group_reward",
    "safe_candidate_count",
    "det_smoothness",
)
LOWER_IS_BETTER = (
    "det_collision",
    "det_rb_crossing",
    "det_kinematic_violation",
    "ade",
    "fde",
)
HARD_METRICS = (
    "det_collision",
    "det_rb_crossing",
    "det_kinematic_violation",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane-change-audit", type=Path, required=True)
    parser.add_argument("--baseline-scenes", type=Path, required=True)
    parser.add_argument("--candidate-scenes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def _moving(row: dict) -> bool:
    return row.get("actual_meta", {}).get("reason") != "stopped"


def _slice_summary(
    paths: list[str], baseline: dict[str, dict], candidate: dict[str, dict], rng, bootstrap: int
) -> dict:
    result: dict = {"matched_scenes": len(paths), "metrics": {}, "hard_transitions": {}}
    for metric in HIGHER_IS_BETTER + LOWER_IS_BETTER:
        valid = [p for p in paths if metric in baseline[p] and metric in candidate[p]]
        if not valid:
            continue
        before = np.asarray([baseline[p][metric] for p in valid], dtype=np.float64)
        after = np.asarray([candidate[p][metric] for p in valid], dtype=np.float64)
        delta = after - before
        if len(delta) == 1:
            lo = hi = float(delta[0])
        else:
            indices = rng.integers(0, len(delta), size=(bootstrap, len(delta)))
            boot_mean = delta[indices].mean(axis=1)
            lo, hi = np.quantile(boot_mean, [0.025, 0.975]).tolist()
        result["metrics"][metric] = {
            "n": len(valid),
            "direction": "higher" if metric in HIGHER_IS_BETTER else "lower",
            "baseline_mean": float(before.mean()),
            "candidate_mean": float(after.mean()),
            "paired_delta_candidate_minus_baseline": float(delta.mean()),
            "paired_bootstrap_ci95": [float(lo), float(hi)],
        }
    for metric in HARD_METRICS:
        valid = [p for p in paths if metric in baseline[p] and metric in candidate[p]]
        result["hard_transitions"][metric] = {
            "n": len(valid),
            "recovered": sum(baseline[p][metric] > 0.5 and candidate[p][metric] <= 0.5 for p in valid),
            "newly_introduced": sum(
                baseline[p][metric] <= 0.5 and candidate[p][metric] > 0.5 for p in valid
            ),
        }
    return result


def main() -> None:
    args = _args()
    audit = json.loads(args.lane_change_audit.read_text())
    baseline = {row["scene_path"]: row for row in json.loads(args.baseline_scenes.read_text())}
    candidate = {row["scene_path"]: row for row in json.loads(args.candidate_scenes.read_text())}
    labels = {
        row["scene"]: row
        for row in audit["rows"]
        if "error" not in row and row["scene"] in baseline and row["scene"] in candidate
    }
    slices = {
        "detector_lane_change": [p for p, row in labels.items() if _moving(row) and row["actual"]],
        "moving_detector_non_change": [
            p for p, row in labels.items() if _moving(row) and not row["actual"]
        ],
        "stopped": [p for p, row in labels.items() if not _moving(row)],
    }
    rng = np.random.default_rng(args.seed)
    payload = {
        "contract": {
            "purpose": "paired current-policy diagnostic; not checkpoint promotion evidence",
            "lane_label": "independent logged-future detector; not perfect ground truth",
            "delta": "candidate minus baseline",
            "bootstrap": "paired scene resampling",
            "seed": args.seed,
            "bootstrap_samples": args.bootstrap,
        },
        "sources": {
            "lane_change_audit": str(args.lane_change_audit),
            "baseline_scenes": str(args.baseline_scenes),
            "candidate_scenes": str(args.candidate_scenes),
        },
        "matched_total": len(labels),
        "slices": {
            name: _slice_summary(paths, baseline, candidate, rng, args.bootstrap)
            for name, paths in slices.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({name: row["matched_scenes"] for name, row in payload["slices"].items()}))


if __name__ == "__main__":
    main()
