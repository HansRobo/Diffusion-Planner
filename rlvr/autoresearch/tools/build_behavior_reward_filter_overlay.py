#!/usr/bin/env python3
"""Filter an immutable AWR replay by deterministic behavior reward.

The candidate bank and its original group-relative AWR weights remain
bit-identical.  Groups whose candidate-0 reward is not below the configured
threshold receive all-zero weights, so replay updates only the selected
pretrained-policy hard set (PlannerRFT's Lt90 data-distribution ablation).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from rlvr.autoresearch.tools.build_positive_anchor_replay_overlay import (
    _sha256,
    _write_json,
)


def build_overlay(
    source_replay: Path,
    output_replay: Path,
    *,
    behavior_reward_lt: float,
) -> dict[str, Any]:
    source_replay = source_replay.expanduser().resolve(strict=True)
    output_replay = output_replay.expanduser().resolve()
    source_run = source_replay.parent
    output_run = output_replay.parent
    if not np.isfinite(behavior_reward_lt):
        raise ValueError("behavior reward threshold must be finite")
    if output_replay.exists():
        raise FileExistsError(output_replay)

    source_config_path = source_run / "effective_config.json"
    source_provenance_path = source_run / "provenance.json"
    if not source_config_path.is_file() or not source_provenance_path.is_file():
        raise FileNotFoundError("source replay lacks effective_config/provenance")
    source_config = json.loads(source_config_path.read_text())
    if not bool(source_config.get("awr", {}).get("deterministic_first", False)):
        raise RuntimeError("behavior-reward filtering requires deterministic_first=true")

    rank_dirs = sorted(source_replay.glob("rank_*"))
    if not rank_dirs:
        raise FileNotFoundError(f"no rank directories under {source_replay}")
    temporary = output_replay.with_name(output_replay.name + ".building")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    rows: list[dict[str, Any]] = []
    totals = {"groups": 0, "eligible_groups": 0, "active_groups": 0, "active_targets": 0}
    try:
        for source_rank in rank_dirs:
            manifest = json.loads((source_rank / "manifest.json").read_text())
            count = int(manifest["scene_count"])
            group_size = int(manifest["group_size"])
            arrays = manifest["arrays"]
            rewards = np.load(source_rank / arrays["rewards"], mmap_mode="r")
            source_weights = np.load(source_rank / arrays["weights"], mmap_mode="r")
            expected = (count, group_size)
            if rewards.shape != expected or source_weights.shape != expected:
                raise ValueError(f"reward/weight shape mismatch in {source_rank}")
            if not np.isfinite(rewards).all() or not np.isfinite(source_weights).all():
                raise ValueError(f"non-finite reward/weight in {source_rank}")
            if (source_weights < 0.0).any():
                raise ValueError(f"negative source AWR weight in {source_rank}")

            eligible = np.asarray(rewards[:, 0] < float(behavior_reward_lt))
            weights = np.asarray(source_weights, dtype=np.float32).copy()
            weights[~eligible] = 0.0
            active = weights.sum(axis=1) > 0.0

            output_rank = temporary / source_rank.name
            output_rank.mkdir()
            for name in arrays.values():
                if name != arrays["weights"]:
                    os.symlink((source_rank / name).resolve(), output_rank / name)
            paths_name = str(manifest.get("paths", "scene_paths.jsonl"))
            if not (output_rank / paths_name).exists():
                os.symlink((source_rank / paths_name).resolve(), output_rank / paths_name)
            expected_name = manifest.get("expected_paths")
            if expected_name and (source_rank / expected_name).exists():
                os.symlink((source_rank / expected_name).resolve(), output_rank / expected_name)
            shutil.copy2(source_rank / "manifest.json", output_rank / "manifest.json")
            output_weights = output_rank / arrays["weights"]
            np.save(output_weights, weights)

            row = {
                "rank": int(manifest["rank"]),
                "scene_count": count,
                "eligible_groups": int(eligible.sum()),
                "active_groups": int(active.sum()),
                "active_targets": int((weights > 0.0).sum()),
                "weights_sha256": _sha256(output_weights),
            }
            rows.append(row)
            totals["groups"] += count
            totals["eligible_groups"] += row["eligible_groups"]
            totals["active_groups"] += row["active_groups"]
            totals["active_targets"] += row["active_targets"]
        os.replace(temporary, output_replay)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    effective = json.loads(source_config_path.read_text())
    effective["replay_group_filter"] = {
        "type": "deterministic_behavior_reward_lt",
        "threshold": float(behavior_reward_lt),
    }
    _write_json(output_run / "effective_config.json", effective)
    provenance = json.loads(source_provenance_path.read_text())
    provenance.update(
        {
            "replay_weight_overlay": "deterministic_behavior_reward_filter_v1",
            "replay_weight_overlay_source": str(source_replay),
        }
    )
    _write_json(output_run / "provenance.json", provenance)
    group_count = totals["groups"]
    payload = {
        "contract": "deterministic_behavior_reward_filter_v1",
        "source_replay": str(source_replay),
        "output_replay": str(output_replay),
        "behavior_reward_lt": float(behavior_reward_lt),
        **totals,
        "eligible_group_fraction": totals["eligible_groups"] / group_count,
        "active_group_fraction": totals["active_groups"] / group_count,
        "bulk_arrays": "absolute read-only symlinks to immutable source",
        "ranks": rows,
    }
    _write_json(output_run / "overlay_manifest.json", payload)

    for path in output_replay.rglob("*"):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode & ~0o222)
    for path in sorted(output_replay.rglob("*"), reverse=True):
        if path.is_dir() and not path.is_symlink():
            path.chmod(path.stat().st_mode & ~0o222)
    output_replay.chmod(output_replay.stat().st_mode & ~0o222)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-replay", type=Path, required=True)
    parser.add_argument("--output-replay", type=Path, required=True)
    parser.add_argument("--behavior-reward-lt", type=float, default=0.9)
    args = parser.parse_args()
    result = build_overlay(
        args.source_replay,
        args.output_replay,
        behavior_reward_lt=args.behavior_reward_lt,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
