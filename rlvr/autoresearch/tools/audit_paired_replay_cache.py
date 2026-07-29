#!/usr/bin/env python3
"""Paired audit of two HDP disk replay caches with identical scene ordering.

The tool is intentionally read-only.  It compares every completed candidate
prefix row against a reference cache before old bulk arrays are rotated away,
and saves compact evidence about reward movement, hard-zero transitions,
weight/rank changes, and a fixed-prefix trajectory-diversity sample.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk_size", type=int, default=8192)
    parser.add_argument("--diversity_groups_per_rank", type=int, default=512)
    return parser.parse_args()


def _completed_rows(rank_dir: Path) -> int:
    path_index = rank_dir / "scene_paths.jsonl"
    if not path_index.is_file():
        raise FileNotFoundError(path_index)
    with path_index.open("rb") as stream:
        return sum(1 for _ in stream)


def _paired_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    count = int(values.size)
    if count == 0:
        raise ValueError("cannot summarize an empty paired vector")
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / math.sqrt(count)) if count > 1 else 0.0
    return {
        "mean_delta": mean,
        "normal_95_ci": [
            mean - 1.959963984540054 * standard_error,
            mean + 1.959963984540054 * standard_error,
        ],
        "median_delta": float(np.median(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "positive_fraction": float(np.mean(values > 1e-8)),
        "negative_fraction": float(np.mean(values < -1e-8)),
        "exact_equal_fraction": float(np.mean(np.abs(values) <= 1e-8)),
    }


def _diversity_summary(root: Path, per_rank: int) -> dict[str, Any]:
    sums = {
        key: 0.0
        for key in (
            "endpoint_pairwise_mean_m",
            "endpoint_pairwise_max_m",
            "pairwise_trajectory_ade_m",
            "temporal_spread_m",
            "first_waypoint_pairwise_mean_m",
            "first_waypoint_pairwise_max_m",
        )
    }
    count = 0
    pair = np.triu_indices(10, 1)
    for rank_dir in sorted(root.glob("rank_*")):
        available = _completed_rows(rank_dir)
        take = min(max(int(per_rank), 0), available)
        if take == 0:
            continue
        trajectories = np.load(rank_dir / "trajectories.npy", mmap_mode="r")
        if trajectories.shape[1] != 10:
            raise ValueError(f"expected K=10 in {rank_dir}, got {trajectories.shape}")
        for start in range(0, take, 64):
            xy = np.asarray(
                trajectories[start : min(take, start + 64), ..., :2],
                dtype=np.float32,
            )
            delta = xy[:, pair[0]] - xy[:, pair[1]]
            distance = np.linalg.norm(delta, axis=-1)
            centroid = xy.mean(axis=1, keepdims=True)
            batch = int(xy.shape[0])
            sums["endpoint_pairwise_mean_m"] += float(
                distance[:, :, -1].mean(axis=1).sum()
            )
            sums["endpoint_pairwise_max_m"] += float(
                distance[:, :, -1].max(axis=1).sum()
            )
            sums["pairwise_trajectory_ade_m"] += float(
                distance.mean(axis=(1, 2)).sum()
            )
            sums["temporal_spread_m"] += float(
                np.linalg.norm(xy - centroid, axis=-1).mean(axis=(1, 2)).sum()
            )
            sums["first_waypoint_pairwise_mean_m"] += float(
                distance[:, :, 0].mean(axis=1).sum()
            )
            sums["first_waypoint_pairwise_max_m"] += float(
                distance[:, :, 0].max(axis=1).sum()
            )
            count += batch
    return {"sampled_groups": count, **{key: value / count for key, value in sums.items()}}


def audit(reference: Path, candidate: Path, chunk_size: int, diversity_per_rank: int) -> dict[str, Any]:
    reference = reference.expanduser().resolve()
    candidate = candidate.expanduser().resolve()
    reference_ranks = sorted(reference.glob("rank_*"))
    candidate_ranks = sorted(candidate.glob("rank_*"))
    if [path.name for path in reference_ranks] != [path.name for path in candidate_ranks]:
        raise ValueError("reference/candidate rank directories differ")
    if len(candidate_ranks) != 8:
        raise ValueError(f"expected 8 ranks, found {len(candidate_ranks)}")

    vectors: dict[str, list[np.ndarray]] = {
        key: [] for key in ("group_mean", "best", "weighted", "candidate0")
    }
    scalar = {
        key: 0.0
        for key in (
            "groups",
            "candidates",
            "path_match",
            "candidate_delta",
            "candidate_better",
            "candidate_worse",
            "candidate_equal",
            "allzero_reference",
            "allzero_candidate",
            "allzero_resolved",
            "allzero_added",
            "zero_candidate_reference",
            "zero_candidate_candidate",
            "candidate_positive_to_zero",
            "candidate_zero_to_positive",
            "best_positive_to_zero",
            "best_zero_to_positive",
            "weight_l1",
            "same_top_weight",
            "reference_reward_sum",
            "candidate_reward_sum",
            "reference_ess",
            "candidate_ess",
            "reference_top1_share",
            "candidate_top1_share",
        )
    }
    rank_counts: dict[str, int] = {}
    chunk_size = max(1, int(chunk_size))

    for candidate_rank in candidate_ranks:
        reference_rank = reference / candidate_rank.name
        count = _completed_rows(candidate_rank)
        reference_count = _completed_rows(reference_rank)
        if reference_count < count:
            raise ValueError(
                f"reference {candidate_rank.name} has {reference_count} rows, candidate has {count}"
            )
        rank_counts[candidate_rank.name] = count
        with (reference_rank / "scene_paths.jsonl").open("rb") as ref_paths, (
            candidate_rank / "scene_paths.jsonl"
        ).open("rb") as candidate_paths:
            for _ in range(count):
                scalar["path_match"] += float(ref_paths.readline() == candidate_paths.readline())

        reference_rewards = np.load(reference_rank / "rewards.npy", mmap_mode="r")
        candidate_rewards = np.load(candidate_rank / "rewards.npy", mmap_mode="r")
        reference_weights = np.load(reference_rank / "weights.npy", mmap_mode="r")
        candidate_weights = np.load(candidate_rank / "weights.npy", mmap_mode="r")
        for start in range(0, count, chunk_size):
            end = min(count, start + chunk_size)
            ref = np.asarray(reference_rewards[start:end], dtype=np.float64)
            cand = np.asarray(candidate_rewards[start:end], dtype=np.float64)
            ref_weight = np.asarray(reference_weights[start:end], dtype=np.float64)
            cand_weight = np.asarray(candidate_weights[start:end], dtype=np.float64)
            if ref.shape != cand.shape or ref.shape[1] != 10:
                raise ValueError(f"invalid reward shape at {candidate_rank}:{start}: {ref.shape}/{cand.shape}")
            if not np.isfinite(ref).all() or not np.isfinite(cand).all():
                raise ValueError(f"non-finite reward at {candidate_rank}:{start}")

            delta = cand - ref
            ref_best = ref.max(axis=1)
            cand_best = cand.max(axis=1)
            ref_weight_sum = ref_weight.sum(axis=1)
            cand_weight_sum = cand_weight.sum(axis=1)
            if np.any(ref_weight_sum <= 0) or np.any(cand_weight_sum <= 0):
                raise ValueError(f"non-positive weight sum at {candidate_rank}:{start}")
            ref_weighted = (ref * ref_weight).sum(axis=1) / ref_weight_sum
            cand_weighted = (cand * cand_weight).sum(axis=1) / cand_weight_sum
            ref_zero = np.all(np.abs(ref) <= 1e-8, axis=1)
            cand_zero = np.all(np.abs(cand) <= 1e-8, axis=1)
            ref_ess = ref_weight_sum**2 / np.square(ref_weight).sum(axis=1)
            cand_ess = cand_weight_sum**2 / np.square(cand_weight).sum(axis=1)

            vectors["group_mean"].append((cand.mean(axis=1) - ref.mean(axis=1)).astype(np.float32))
            vectors["best"].append((cand_best - ref_best).astype(np.float32))
            vectors["weighted"].append((cand_weighted - ref_weighted).astype(np.float32))
            vectors["candidate0"].append(delta[:, 0].astype(np.float32))

            batch = end - start
            scalar["groups"] += batch
            scalar["candidates"] += delta.size
            scalar["candidate_delta"] += float(delta.sum())
            scalar["candidate_better"] += float(np.sum(delta > 1e-8))
            scalar["candidate_worse"] += float(np.sum(delta < -1e-8))
            scalar["candidate_equal"] += float(np.sum(np.abs(delta) <= 1e-8))
            scalar["allzero_reference"] += float(ref_zero.sum())
            scalar["allzero_candidate"] += float(cand_zero.sum())
            scalar["allzero_resolved"] += float(np.sum(ref_zero & ~cand_zero))
            scalar["allzero_added"] += float(np.sum(~ref_zero & cand_zero))
            scalar["zero_candidate_reference"] += float(np.sum(np.abs(ref) <= 1e-8))
            scalar["zero_candidate_candidate"] += float(np.sum(np.abs(cand) <= 1e-8))
            scalar["candidate_positive_to_zero"] += float(np.sum((ref > 1e-8) & (cand <= 1e-8)))
            scalar["candidate_zero_to_positive"] += float(np.sum((ref <= 1e-8) & (cand > 1e-8)))
            scalar["best_positive_to_zero"] += float(np.sum((ref_best > 1e-8) & (cand_best <= 1e-8)))
            scalar["best_zero_to_positive"] += float(np.sum((ref_best <= 1e-8) & (cand_best > 1e-8)))
            scalar["weight_l1"] += float(np.abs(cand_weight - ref_weight).mean(axis=1).sum())
            scalar["same_top_weight"] += float(
                np.sum(np.argmax(ref_weight, axis=1) == np.argmax(cand_weight, axis=1))
            )
            scalar["reference_reward_sum"] += float(ref.sum())
            scalar["candidate_reward_sum"] += float(cand.sum())
            scalar["reference_ess"] += float(ref_ess.sum())
            scalar["candidate_ess"] += float(cand_ess.sum())
            scalar["reference_top1_share"] += float(
                (ref_weight.max(axis=1) / ref_weight_sum).sum()
            )
            scalar["candidate_top1_share"] += float(
                (cand_weight.max(axis=1) / cand_weight_sum).sum()
            )

    groups = int(scalar["groups"])
    candidates = int(scalar["candidates"])
    if int(scalar["path_match"]) != groups:
        raise ValueError(
            f"scene order mismatch: {int(scalar['path_match'])}/{groups} rows match"
        )
    paired = {
        key: _paired_summary(np.concatenate(chunks)) for key, chunks in vectors.items()
    }
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "reference_replay": str(reference),
        "candidate_replay": str(candidate),
        "paired_groups": groups,
        "paired_candidates": candidates,
        "path_match_count": int(scalar["path_match"]),
        "rank_counts": rank_counts,
        "reward_distribution": {
            "reference_mean_candidate_reward": scalar["reference_reward_sum"] / candidates,
            "candidate_mean_candidate_reward": scalar["candidate_reward_sum"] / candidates,
            "mean_candidate_reward_delta": scalar["candidate_delta"] / candidates,
            "candidate_better_fraction": scalar["candidate_better"] / candidates,
            "candidate_worse_fraction": scalar["candidate_worse"] / candidates,
            "candidate_exact_equal_fraction": scalar["candidate_equal"] / candidates,
            "paired_group_mean": paired["group_mean"],
            "paired_group_best": paired["best"],
            "paired_awr_weighted_reward": paired["weighted"],
            "paired_candidate0": paired["candidate0"],
        },
        "hard_zero_transitions": {
            "reference_all_zero_group_fraction": scalar["allzero_reference"] / groups,
            "candidate_all_zero_group_fraction": scalar["allzero_candidate"] / groups,
            "all_zero_groups_resolved": int(scalar["allzero_resolved"]),
            "all_zero_groups_added": int(scalar["allzero_added"]),
            "reference_zero_candidate_fraction": scalar["zero_candidate_reference"] / candidates,
            "candidate_zero_candidate_fraction": scalar["zero_candidate_candidate"] / candidates,
            "candidate_positive_to_zero": int(scalar["candidate_positive_to_zero"]),
            "candidate_zero_to_positive": int(scalar["candidate_zero_to_positive"]),
            "best_positive_to_zero": int(scalar["best_positive_to_zero"]),
            "best_zero_to_positive": int(scalar["best_zero_to_positive"]),
        },
        "weight_distribution": {
            "reference_mean_ess": scalar["reference_ess"] / groups,
            "candidate_mean_ess": scalar["candidate_ess"] / groups,
            "reference_mean_top1_share": scalar["reference_top1_share"] / groups,
            "candidate_mean_top1_share": scalar["candidate_top1_share"] / groups,
            "mean_per_group_candidate_weight_l1": scalar["weight_l1"] / groups,
            "same_top_weight_fraction": scalar["same_top_weight"] / groups,
        },
        "trajectory_diversity": {
            "reference": _diversity_summary(reference, diversity_per_rank),
            "candidate": _diversity_summary(candidate, diversity_per_rank),
        },
        "interpretation_scope": (
            "Paired cache-distribution evidence only. It does not replace K=1 train-selector, "
            "full validation, reactive simulation, or closed-loop evaluation."
        ),
    }


def main() -> None:
    args = _parse_args()
    result = audit(
        args.reference,
        args.candidate,
        args.chunk_size,
        args.diversity_groups_per_rank,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
