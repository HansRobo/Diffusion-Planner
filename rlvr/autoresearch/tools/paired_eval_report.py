#!/usr/bin/env python3
"""Paired, scene-level checkpoint comparison for full-corpus AWR runs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

METRICS = {
    "det_reward": 1,
    "best_group_reward": 1,
    "mean_group_reward": 1,
    "best_safe_reward": 1,
    "safe_candidate_count": 1,
    "det_centerline": 1,
    "det_smoothness": 1,
    "det_progress": 1,
    "det_safety": 1,
    "det_zero_reward": -1,
    # Output distance is essential contraction context, but more distance is
    # not inherently better (for example, a correct stop).  Bootstrap its raw
    # signed change without assigning a policy-quality direction.
    "path_len": 0,
    "ade": -1,
    "fde": -1,
    "det_collision": -1,
    "det_rb_crossing": -1,
    "det_kinematic_violation": -1,
    "det_lane_crossing": -1,
}


def load_rows(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text())
    by_path = {str(row["scene_path"]): row for row in rows}
    if len(by_path) != len(rows):
        raise RuntimeError(f"duplicate scene_path in {path}")
    return by_path


def compare(
    baseline: dict[str, dict],
    candidate: dict[str, dict],
    bootstrap: int,
    seed: int,
    metric_keys: tuple[str, ...] | None = None,
) -> dict:
    if baseline.keys() != candidate.keys():
        missing = sorted(baseline.keys() - candidate.keys())[:3]
        extra = sorted(candidate.keys() - baseline.keys())[:3]
        raise RuntimeError(f"scene mismatch: missing={missing}, extra={extra}")
    paths = sorted(baseline)
    rng = np.random.default_rng(seed)
    result = {"scene_count": len(paths), "bootstrap_samples": bootstrap, "metrics": {}}
    selected_metrics = METRICS if metric_keys is None else {
        key: METRICS[key] for key in metric_keys
    }
    for key, direction in selected_metrics.items():
        # Some scene metrics are undefined by construction (for example,
        # best_safe_reward when no candidate passes every hard gate).  Keep
        # the paired contract but bootstrap only paths with finite values on
        # both sides; one null must not discard other valid metrics/scenes.
        paired = []
        for path in paths:
            if key not in baseline[path] or key not in candidate[path]:
                continue
            try:
                base_value = float(baseline[path][key])
                candidate_value = float(candidate[path][key])
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(base_value) and np.isfinite(candidate_value)):
                continue
            paired.append((base_value, candidate_value))
        if not paired:
            continue

        values = np.asarray(paired, dtype=np.float64)
        base_values = values[:, 0]
        candidate_values = values[:, 1]
        raw_delta = candidate_values - base_values
        improvement = raw_delta * float(direction)
        bootstrap_values = improvement if direction else raw_delta
        boot_means = np.empty(bootstrap, dtype=np.float64)
        chunk = 256
        for start in range(0, bootstrap, chunk):
            count = min(chunk, bootstrap - start)
            indices = rng.integers(0, len(paired), size=(count, len(paired)))
            boot_means[start : start + count] = bootstrap_values[indices].mean(axis=1)

        mean_raw_delta = float(raw_delta.mean())
        metric_result = {
            "direction": (
                "higher" if direction > 0 else "lower" if direction < 0 else "neutral"
            ),
            "paired_scene_count": int(len(paired)),
            "dropped_nonfinite_or_missing": int(len(paths) - len(paired)),
            "baseline_mean": float(base_values.mean()),
            "candidate_mean": float(candidate_values.mean()),
            "raw_delta": mean_raw_delta,
            "relative_delta": (
                mean_raw_delta / float(abs(base_values.mean()))
                if abs(float(base_values.mean())) > 1e-12
                else None
            ),
        }
        if direction:
            metric_result.update(
                {
                    "improvement": float(improvement.mean()),
                    "improvement_ci95": [
                        float(np.quantile(boot_means, 0.025)),
                        float(np.quantile(boot_means, 0.975)),
                    ],
                    "probability_improved": float((boot_means > 0.0).mean()),
                    "improved_scene_fraction": float((improvement > 0.0).mean()),
                    "regressed_scene_fraction": float((improvement < 0.0).mean()),
                }
            )
        else:
            metric_result.update(
                {
                    "raw_delta_ci95": [
                        float(np.quantile(boot_means, 0.025)),
                        float(np.quantile(boot_means, 0.975)),
                    ],
                    "probability_delta_positive": float(
                        (boot_means > 0.0).mean()
                    ),
                    "positive_delta_scene_fraction": float(
                        (raw_delta > 0.0).mean()
                    ),
                    "negative_delta_scene_fraction": float(
                        (raw_delta < 0.0).mean()
                    ),
                }
            )
        result["metrics"][key] = metric_result
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline_path = args.run_dir / "eval_epoch_000" / "scenes.json"
    baseline = load_rows(baseline_path)
    report = {
        "run_dir": str(args.run_dir.resolve()),
        "baseline": str(baseline_path),
        "epochs": {},
    }
    epoch_dirs = sorted(args.run_dir.glob("eval_epoch_*"))
    for epoch_dir in epoch_dirs:
        match = re.fullmatch(r"eval_epoch_(\d+)", epoch_dir.name)
        if not match or int(match.group(1)) == 0:
            continue
        scenes_path = epoch_dir / "scenes.json"
        if not scenes_path.exists():
            continue
        epoch = int(match.group(1))
        report["epochs"][str(epoch)] = compare(
            baseline,
            load_rows(scenes_path),
            max(1, int(args.bootstrap)),
            int(args.seed) + epoch,
        )

    output = args.output or args.run_dir / "paired_eval_summary.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
