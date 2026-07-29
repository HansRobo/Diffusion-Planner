#!/usr/bin/env python3
"""Select one global AWR checkpoint across split training processes.

The formal Original-DP experiment selects checkpoints only by deterministic
K=1 ``mean_det_reward`` on one fixed train-set scene list.  A policy-iteration
cycle may be split across processes, so each process-local ``best_train.pth``
is not necessarily the global incumbent.  This tool verifies the ordered scene
contract, recomputes every candidate mean from per-scene rows, and materializes
the single highest-scoring checkpoint with an auditable manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any


def _sha256_file(path: Path, *, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _ordered_path_sha256(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        scene_path = str(row.get("scene_path", ""))
        if not scene_path:
            raise ValueError("selector row is missing scene_path")
        digest.update(scene_path.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _parse_candidate(value: str) -> dict[str, Any]:
    fields = value.split("|")
    if len(fields) != 5:
        raise argparse.ArgumentTypeError(
            "candidate must be LABEL|EPOCH|CHECKPOINT|SUMMARY_JSON|SCENES_JSON"
        )
    label, epoch, checkpoint, summary, scenes = fields
    try:
        parsed_epoch = int(epoch)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid candidate epoch: {epoch}") from error
    return {
        "label": label,
        "epoch": parsed_epoch,
        "checkpoint": Path(checkpoint).expanduser(),
        "summary": Path(summary).expanduser(),
        "scenes": Path(scenes).expanduser(),
    }


def _audit_candidate(
    spec: dict[str, Any],
    *,
    expected_scene_count: int,
    expected_order_sha256: str | None,
) -> tuple[dict[str, Any], str]:
    checkpoint = spec["checkpoint"].resolve(strict=True)
    summary_path = spec["summary"].resolve(strict=True)
    scenes_path = spec["scenes"].resolve(strict=True)
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise ValueError(f"empty checkpoint: {checkpoint}")

    summary = json.loads(summary_path.read_text())
    rows = json.loads(scenes_path.read_text())
    if not isinstance(summary, dict) or not isinstance(rows, list):
        raise TypeError(f"invalid selector artifacts for {spec['label']}")
    if len(rows) != expected_scene_count:
        raise ValueError(
            f"{spec['label']}: {len(rows)} rows != {expected_scene_count}"
        )
    if int(summary.get("scene_count", -1)) != expected_scene_count:
        raise ValueError(f"{spec['label']}: summary scene_count mismatch")

    order_sha256 = _ordered_path_sha256(rows)
    if expected_order_sha256 is not None and order_sha256 != expected_order_sha256:
        raise ValueError(f"{spec['label']}: ordered selector scene contract differs")

    rewards: list[float] = []
    for index, row in enumerate(rows):
        try:
            reward = float(row["det_reward"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{spec['label']}: invalid det_reward at row {index}"
            ) from error
        if not math.isfinite(reward):
            raise ValueError(
                f"{spec['label']}: non-finite det_reward at row {index}"
            )
        rewards.append(reward)
    recomputed_mean = math.fsum(rewards) / len(rewards)
    reported_mean = float(summary["mean_det_reward"])
    if not math.isclose(recomputed_mean, reported_mean, rel_tol=0.0, abs_tol=2e-12):
        raise ValueError(
            f"{spec['label']}: reported mean {reported_mean} != "
            f"recomputed {recomputed_mean}"
        )

    row = {
        "label": str(spec["label"]),
        "selection_epoch": int(spec["epoch"]),
        "mean_det_reward": recomputed_mean,
        "dataset_score": 100.0 * recomputed_mean,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "summary": str(summary_path),
        "scenes": str(scenes_path),
        "ordered_scene_path_sha256": order_sha256,
        "scene_count": len(rows),
    }
    return row, order_sha256


def select_global_checkpoint(
    candidates: list[dict[str, Any]],
    *,
    expected_scene_count: int,
    output_checkpoint: Path,
    output_manifest: Path,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("at least one --candidate is required")
    audited: list[dict[str, Any]] = []
    order_sha256: str | None = None
    labels: set[str] = set()
    for spec in candidates:
        label = str(spec["label"])
        if label in labels:
            raise ValueError(f"duplicate candidate label: {label}")
        labels.add(label)
        row, candidate_order_sha256 = _audit_candidate(
            spec,
            expected_scene_count=expected_scene_count,
            expected_order_sha256=order_sha256,
        )
        order_sha256 = candidate_order_sha256
        audited.append(row)

    # The earliest epoch is the deterministic tie-breaker.  A later
    # checkpoint must strictly improve the fixed train-set mean to replace it.
    winner = max(
        audited,
        key=lambda row: (float(row["mean_det_reward"]), -int(row["selection_epoch"])),
    )
    baseline = audited[0]
    output_checkpoint = output_checkpoint.expanduser().resolve()
    output_manifest = output_manifest.expanduser().resolve()
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_checkpoint.with_suffix(output_checkpoint.suffix + ".tmp")
    shutil.copy2(winner["checkpoint"], temporary)
    os.replace(temporary, output_checkpoint)
    materialized_sha256 = _sha256_file(output_checkpoint)
    if materialized_sha256 != winner["checkpoint_sha256"]:
        raise RuntimeError("materialized checkpoint hash differs from winner")

    payload = {
        "selection_contract": "fixed_train_selector_K1_mean_det_reward",
        "validation_role": "report_only_not_checkpoint_selection",
        "expected_scene_count": expected_scene_count,
        "ordered_scene_path_sha256": order_sha256,
        "candidate_count": len(audited),
        "baseline": baseline,
        "winner": winner,
        "winner_reward_delta_from_baseline": (
            float(winner["mean_det_reward"]) - float(baseline["mean_det_reward"])
        ),
        "materialized_checkpoint": str(output_checkpoint),
        "materialized_checkpoint_sha256": materialized_sha256,
        "candidates": audited,
    }
    temporary_manifest = output_manifest.with_suffix(output_manifest.suffix + ".tmp")
    temporary_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_manifest, output_manifest)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        type=_parse_candidate,
        required=True,
        help="LABEL|EPOCH|CHECKPOINT|SUMMARY_JSON|SCENES_JSON",
    )
    parser.add_argument("--expected-scene-count", type=int, default=65_536)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.expected_scene_count <= 0:
        parser.error("--expected-scene-count must be positive")
    payload = select_global_checkpoint(
        args.candidate,
        expected_scene_count=args.expected_scene_count,
        output_checkpoint=args.output_checkpoint,
        output_manifest=args.output_manifest,
    )
    print(
        json.dumps(
            {
                "winner": payload["winner"],
                "winner_reward_delta_from_baseline": payload[
                    "winner_reward_delta_from_baseline"
                ],
                "materialized_checkpoint": payload["materialized_checkpoint"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
