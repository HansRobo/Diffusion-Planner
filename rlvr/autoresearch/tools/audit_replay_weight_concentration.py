#!/usr/bin/env python3
"""Audit positive-advantage AWR weight concentration without reading trajectories.

The replay reward memmaps are sufficient to reconstruct the lightweight
positive-only overlay exactly.  This tool compares beta values on the same
committed groups and reports whether a small number of scenes/candidates would
dominate the unnormalised diffusion regression loss.  It is a pre-update data
audit, not evidence that any checkpoint improved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rlvr.autoresearch.tools.build_positive_anchor_replay_overlay import (
    compute_positive_anchor_weights,
)


def _committed_count(rank_dir: Path, allow_partial_prefix: bool) -> tuple[int, bool]:
    manifest_path = rank_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        return int(manifest["scene_count"]), True
    if not allow_partial_prefix:
        raise FileNotFoundError(
            f"{manifest_path}; pass --allow-partial-prefix only for an active writer"
        )
    paths_path = rank_dir / "scene_paths.jsonl"
    if not paths_path.is_file():
        raise FileNotFoundError(paths_path)
    with paths_path.open() as stream:
        return sum(1 for line in stream if line.strip()), False


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {name: 0.0 for name in ("p50", "p90", "p95", "p99", "p999", "max")}
    points = np.quantile(values, [0.5, 0.9, 0.95, 0.99, 0.999])
    return {
        "p50": float(points[0]),
        "p90": float(points[1]),
        "p95": float(points[2]),
        "p99": float(points[3]),
        "p999": float(points[4]),
        "max": float(values.max()),
    }


def _top_group_mass_share(group_mass: np.ndarray, fraction: float) -> float:
    if group_mass.size == 0 or float(group_mass.sum()) <= 0.0:
        return 0.0
    count = max(1, int(np.ceil(group_mass.size * float(fraction))))
    # Partition avoids a full O(N log N) sort for each requested fraction.
    threshold = group_mass.size - count
    heaviest = np.partition(group_mass, threshold)[threshold:]
    return float(heaviest.sum() / group_mass.sum())


def analyze(
    replay_root: Path,
    *,
    betas: tuple[float, ...],
    margin: float,
    behavior_anchor_weight: float,
    unsafe_behavior_anchor_weight: float,
    expert_weight: float,
    allow_partial_prefix: bool,
    chunk_size: int = 65536,
    max_groups_per_rank: int = 0,
) -> dict[str, Any]:
    replay_root = replay_root.expanduser().resolve(strict=True)
    if not betas or any(beta < 0.0 for beta in betas):
        raise ValueError("betas must be a non-empty tuple of non-negative values")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if margin < 0.0:
        raise ValueError("margin must be non-negative")
    if min(behavior_anchor_weight, unsafe_behavior_anchor_weight, expert_weight) < 0.0:
        raise ValueError("anchor weights must be non-negative")
    if max_groups_per_rank < 0:
        raise ValueError("max_groups_per_rank must be non-negative")

    rank_dirs = sorted(path for path in replay_root.glob("rank_*") if path.is_dir())
    if not rank_dirs:
        raise FileNotFoundError(f"no rank directories under {replay_root}")

    accumulators = {
        beta: {
            "group_mass": [],
            "max_weight": [],
            "top1_share": [],
            "ess": [],
            "ess_fraction": [],
            "active_count": [],
            "candidate_mass": 0.0,
            "expert_mass": 0.0,
        }
        for beta in betas
    }
    ranks: list[dict[str, Any]] = []
    all_closed = True
    total_groups = 0
    group_size: int | None = None

    for rank_index, rank_dir in enumerate(rank_dirs):
        committed_count, strict = _committed_count(rank_dir, allow_partial_prefix)
        count = (
            min(committed_count, int(max_groups_per_rank))
            if max_groups_per_rank > 0
            else committed_count
        )
        manifest_path = rank_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if strict else {}
        arrays = manifest.get("arrays", {})
        rewards = np.load(
            rank_dir / str(arrays.get("rewards", "rewards.npy")), mmap_mode="r"
        )
        expert_safe = np.load(rank_dir / "expert_safe.npy", mmap_mode="r")
        if rewards.ndim != 2 or rewards.shape[0] < count:
            raise ValueError(f"invalid/short rewards in {rank_dir}: {rewards.shape}")
        if expert_safe.ndim != 1 or expert_safe.shape[0] < count:
            raise ValueError(f"invalid/short expert_safe in {rank_dir}: {expert_safe.shape}")
        if group_size is None:
            group_size = int(rewards.shape[1])
        elif int(rewards.shape[1]) != group_size:
            raise RuntimeError("mixed replay group sizes")

        for lower in range(0, count, chunk_size):
            upper = min(count, lower + chunk_size)
            reward = np.asarray(rewards[lower:upper], dtype=np.float64)
            if not np.isfinite(reward).all():
                raise ValueError(f"non-finite reward in committed prefix {rank_dir}")
            safe = np.asarray(expert_safe[lower:upper], dtype=np.float64) > 0.5
            expert_mass = float(expert_weight) * float(safe.sum())
            for beta in betas:
                weights = compute_positive_anchor_weights(
                    reward,
                    beta=float(beta),
                    margin=float(margin),
                    behavior_anchor_weight=float(behavior_anchor_weight),
                    unsafe_behavior_anchor_weight=float(
                        unsafe_behavior_anchor_weight
                    ),
                    normalize=False,
                    drop_all_zero_groups=True,
                ).astype(np.float64)
                positive = weights > 0.0
                active_count = positive.sum(axis=1)
                group_mass = weights.sum(axis=1)
                active = group_mass > 0.0
                if active.any():
                    active_weights = weights[active]
                    active_mass = group_mass[active]
                    active_counts = active_count[active]
                    square_mass = np.square(active_weights).sum(axis=1)
                    ess = np.divide(
                        np.square(active_mass),
                        square_mass,
                        out=np.zeros_like(active_mass),
                        where=square_mass > 0.0,
                    )
                    top1 = np.divide(
                        active_weights.max(axis=1),
                        active_mass,
                        out=np.zeros_like(active_mass),
                        where=active_mass > 0.0,
                    )
                    bucket = accumulators[beta]
                    bucket["group_mass"].append(active_mass)
                    bucket["max_weight"].append(active_weights.max(axis=1))
                    bucket["top1_share"].append(top1)
                    bucket["ess"].append(ess)
                    bucket["ess_fraction"].append(ess / active_counts)
                    bucket["active_count"].append(active_counts.astype(np.float64))
                    bucket["candidate_mass"] += float(active_mass.sum())
                accumulators[beta]["expert_mass"] += expert_mass

        total_groups += count
        all_closed = all_closed and strict
        ranks.append(
            {
                "rank": int(manifest.get("rank", rank_index)),
                "scene_count": int(count),
                "committed_scene_count": int(committed_count),
                "strict_manifest": bool(strict),
            }
        )

    results: dict[str, Any] = {}
    for beta in betas:
        bucket = accumulators[beta]

        def joined(name: str) -> np.ndarray:
            chunks = bucket[name]
            return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float64)

        group_mass = joined("group_mass")
        max_weight = joined("max_weight")
        top1 = joined("top1_share")
        ess = joined("ess")
        ess_fraction = joined("ess_fraction")
        active_count = joined("active_count")
        candidate_mass = float(bucket["candidate_mass"])
        expert_mass = float(bucket["expert_mass"])
        total_mass = candidate_mass + expert_mass
        results[f"{beta:g}"] = {
            "active_groups": int(group_mass.size),
            "active_group_fraction": float(group_mass.size / total_groups),
            "mean_active_candidate_count": float(active_count.mean())
            if active_count.size
            else 0.0,
            "candidate_weight_mass": candidate_mass,
            "expert_weight_mass": expert_mass,
            "global_expert_weight_share": float(expert_mass / total_mass)
            if total_mass > 0.0
            else 0.0,
            "group_candidate_weight_sum": _quantiles(group_mass),
            "max_candidate_weight": _quantiles(max_weight),
            "top1_weight_share": {
                **_quantiles(top1),
                "mean": float(top1.mean()) if top1.size else 0.0,
                "fraction_gt_0p50": float(np.mean(top1 > 0.50)) if top1.size else 0.0,
                "fraction_gt_0p80": float(np.mean(top1 > 0.80)) if top1.size else 0.0,
                "fraction_gt_0p95": float(np.mean(top1 > 0.95)) if top1.size else 0.0,
            },
            "effective_sample_size": {
                **_quantiles(ess),
                "mean": float(ess.mean()) if ess.size else 0.0,
            },
            "effective_sample_size_fraction": {
                **_quantiles(ess_fraction),
                "mean": float(ess_fraction.mean()) if ess_fraction.size else 0.0,
            },
            "candidate_mass_share_from_heaviest_active_groups": {
                "top_1pct": _top_group_mass_share(group_mass, 0.01),
                "top_5pct": _top_group_mass_share(group_mass, 0.05),
                "top_10pct": _top_group_mass_share(group_mass, 0.10),
            },
        }

    return {
        "contract": {
            "evidence": "exact positive-overlay weight concentration before optimizer update",
            "not_evidence_of": "checkpoint performance",
            "loss_relevance": "formal replay uses unnormalised per-target weights",
            "candidate_zero": "deterministic deployment behavior",
        },
        "replay_root": str(replay_root),
        "source_complete": bool(all_closed),
        "analysis_truncated": bool(max_groups_per_rank > 0),
        "prefix_groups": int(total_groups),
        "group_size": int(group_size or 0),
        "rank_counts": ranks,
        "parameters": {
            "betas": [float(beta) for beta in betas],
            "margin": float(margin),
            "behavior_anchor_weight": float(behavior_anchor_weight),
            "unsafe_behavior_anchor_weight": float(unsafe_behavior_anchor_weight),
            "expert_weight": float(expert_weight),
            "max_groups_per_rank": int(max_groups_per_rank),
        },
        "beta_results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--betas", type=float, nargs="+", default=(0.25, 0.5, 1.0, 2.0))
    parser.add_argument("--margin", type=float, default=0.01)
    parser.add_argument("--behavior-anchor-weight", type=float, default=0.0)
    parser.add_argument("--unsafe-behavior-anchor-weight", type=float, default=0.0)
    parser.add_argument("--expert-weight", type=float, default=0.4)
    parser.add_argument("--allow-partial-prefix", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--max-groups-per-rank", type=int, default=0)
    args = parser.parse_args()
    payload = analyze(
        args.replay_root,
        betas=tuple(dict.fromkeys(float(beta) for beta in args.betas)),
        margin=float(args.margin),
        behavior_anchor_weight=float(args.behavior_anchor_weight),
        unsafe_behavior_anchor_weight=float(args.unsafe_behavior_anchor_weight),
        expert_weight=float(args.expert_weight),
        allow_partial_prefix=bool(args.allow_partial_prefix),
        chunk_size=int(args.chunk_size),
        max_groups_per_rank=int(args.max_groups_per_rank),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
