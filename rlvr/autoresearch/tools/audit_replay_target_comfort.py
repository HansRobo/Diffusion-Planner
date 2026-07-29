#!/usr/bin/env python3
"""Audit comfort/feasibility of the exact positive AWR replay targets.

The replay reward already contains HDP's comfort term and a kinematic hard
gate, but that does not answer a more diagnostic question: among candidates
that beat deterministic candidate zero, how much comfort margin is retained,
and what target distribution is the regression loss asked to fit?

This tool reads the immutable replay trajectories and the *derived* positive
weights used by training.  It compares, on active groups only:

* candidate zero (zero-noise deployment behaviour),
* the reward-weighted positive candidate distribution,
* the highest-reward positive candidate,
* the safe logged expert anchor, and
* the exact candidate/expert mixture implied by their stored loss weights.

Metrics are recomputed with the production reward primitives.  Both the AWR
candidate-loss prefix and the complete trajectory are reported.  The audit is
sampled systematically over every rank's active rows so it does not need to
stream the full multi-terabyte decoder context.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from planner_metrics.config import RewardConfig
from planner_metrics.subscores import (
    compute_feasibility_score_batch,
    compute_kinematic_gate,
    compute_smoothness_score_batch,
)
from preference_optimization.utils import effective_ego_shape_numpy


def _load_reward_config(path: Path) -> tuple[RewardConfig, dict[str, Any]]:
    effective = json.loads(path.read_text())
    raw = effective.get("reward", {})
    valid = {item.name for item in fields(RewardConfig)}
    kwargs = {key: value for key, value in raw.items() if key in valid}
    return RewardConfig(**kwargs), effective


def _systematic_active_indices(weights: np.ndarray, count: int) -> np.ndarray:
    """Return up to ``count`` rows uniformly spaced in active-row order."""

    active = np.flatnonzero(np.any(np.asarray(weights) > 0.0, axis=1))
    if count <= 0 or active.size <= count:
        return active.astype(np.int64, copy=False)
    positions = np.linspace(0, active.size - 1, num=count, dtype=np.int64)
    return active[positions]


@torch.inference_mode()
def _trajectory_metrics(
    trajectories: np.ndarray,
    *,
    horizon: int,
    config: RewardConfig,
    ego_shapes: np.ndarray,
) -> dict[str, np.ndarray]:
    """Recompute production trajectory-only reward terms on CPU."""

    shape = trajectories.shape
    if len(shape) < 3 or shape[-1] != 4:
        raise ValueError(f"expected [...,T,4] trajectories, got {shape}")
    flat = torch.from_numpy(
        np.ascontiguousarray(trajectories[..., :horizon, :], dtype=np.float32)
    ).reshape(-1, min(horizon, shape[-2]), 4)
    if ego_shapes.shape != (shape[0], 3):
        raise ValueError(
            f"ego shape rows {ego_shapes.shape} do not match trajectory batch {shape}"
        )
    candidates_per_scene = int(np.prod(shape[1:-2], dtype=np.int64))
    flat_shapes = np.repeat(
        np.asarray(ego_shapes, dtype=np.float32), candidates_per_scene, axis=0
    )
    smoothness = compute_smoothness_score_batch(flat, config)
    # The current soft feasibility primitive is trajectory-only.  Pass one
    # valid shape to preserve its public contract.
    feasibility, _ = compute_feasibility_score_batch(
        flat, torch.from_numpy(flat_shapes[0]), {}, config
    )
    # Kinematic curvature depends on wheelbase.  The primitive accepts one
    # vehicle shape per call, so evaluate each distinct wheelbase separately.
    kinematic = torch.empty(flat.shape[0], dtype=torch.float32)
    for wheelbase in np.unique(flat_shapes[:, 0]):
        rows_np = np.flatnonzero(flat_shapes[:, 0] == wheelbase)
        rows = torch.from_numpy(rows_np)
        vehicle_shape = torch.from_numpy(flat_shapes[rows_np[0]])
        kinematic[rows] = compute_kinematic_gate(
            flat.index_select(0, rows), config, vehicle_shape
        )
    comfort = torch.exp(
        smoothness.clamp(min=-50.0, max=0.0)
        / max(float(config.pdm_comfort_scale), 1e-3)
    ) * torch.exp(feasibility.clamp(min=-10.0, max=0.0))
    leading = shape[:-2]
    return {
        "smoothness": smoothness.numpy().reshape(leading),
        "feasibility": feasibility.numpy().reshape(leading),
        "comfort": comfort.clamp(0.0, 1.0).numpy().reshape(leading),
        "kinematic": kinematic.numpy().reshape(leading),
    }


def _selected_scene_paths(rank_dir: Path, indices: np.ndarray) -> list[Path]:
    """Read selected JSONL rows without materialising the complete path list."""

    wanted = {int(index): position for position, index in enumerate(indices.tolist())}
    result: list[Path | None] = [None] * len(indices)
    with (rank_dir / "scene_paths.jsonl").open() as stream:
        for row, line in enumerate(stream):
            position = wanted.get(row)
            if position is not None:
                result[position] = Path(json.loads(line))
                if all(path is not None for path in result):
                    break
    if any(path is None for path in result):
        raise RuntimeError(f"scene path JSONL is shorter than selected rows in {rank_dir}")
    return [path for path in result if path is not None]


def _effective_shapes(paths: list[Path]) -> np.ndarray:
    """Load the exact loader-visible ego shape, cached per recorded drive."""

    # Vehicle parameters are constant within one recording directory.  Cache
    # that immutable metadata so a 16k-scene audit opens hundreds of NPZs, not
    # 16k.  X2 width normalization is delegated to the canonical loader helper.
    cache: dict[Path, np.ndarray] = {}
    rows: list[np.ndarray] = []
    for path in paths:
        key = path.parent
        shape = cache.get(key)
        if shape is None:
            with np.load(path, allow_pickle=False) as loaded:
                raw = loaded["ego_shape"] if "ego_shape" in loaded else None
            shape = effective_ego_shape_numpy(path, raw)
            cache[key] = shape
        rows.append(shape)
    return np.stack(rows).astype(np.float32, copy=False)


def _describe(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "mean": None}
    q = np.quantile(values, [0.01, 0.05, 0.5, 0.95, 0.99])
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p01": float(q[0]),
        "p05": float(q[1]),
        "p50": float(q[2]),
        "p95": float(q[3]),
        "p99": float(q[4]),
        "min": float(values.min()),
        "max": float(values.max()),
        "positive_fraction": float(np.mean(values > 1e-8)),
        "negative_fraction": float(np.mean(values < -1e-8)),
    }


def _weighted(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mass = weights.sum(axis=1)
    return np.divide(
        (values * weights).sum(axis=1),
        mass,
        out=np.zeros(values.shape[0], dtype=np.float64),
        where=mass > 0.0,
    )


def analyze(
    replay_root: Path,
    *,
    config_path: Path,
    expert_weight: float,
    samples_per_rank: int,
    batch_groups: int,
    candidate_horizon: int,
) -> dict[str, Any]:
    replay_root = replay_root.expanduser().resolve(strict=True)
    config, effective = _load_reward_config(config_path.expanduser().resolve(strict=True))
    rank_dirs = sorted(path for path in replay_root.glob("rank_*") if path.is_dir())
    if not rank_dirs:
        raise FileNotFoundError(f"no rank directories below {replay_root}")
    if samples_per_rank < 1 or batch_groups < 1:
        raise ValueError("samples_per_rank and batch_groups must be positive")
    if expert_weight < 0.0:
        raise ValueError("expert_weight must be non-negative")
    collected: dict[str, dict[str, list[np.ndarray]]] = {}
    reward_gain_rows: list[np.ndarray] = []
    rank_rows: list[dict[str, int]] = []
    full_horizon = 0

    for rank_dir in rank_dirs:
        manifest = json.loads((rank_dir / "manifest.json").read_text())
        arrays = manifest.get("arrays", {})
        trajectories = np.load(
            rank_dir / arrays.get("trajectories", "trajectories.npy"), mmap_mode="r"
        )
        rewards = np.load(
            rank_dir / arrays.get("rewards", "rewards.npy"), mmap_mode="r"
        )
        weights = np.load(
            rank_dir / arrays.get("weights", "weights.npy"), mmap_mode="r"
        )
        experts = np.load(rank_dir / "expert_trajectories.npy", mmap_mode="r")
        expert_safe = np.load(rank_dir / "expert_safe.npy", mmap_mode="r")
        scene_count = int(manifest["scene_count"])
        full_horizon = int(trajectories.shape[2])
        indices = _systematic_active_indices(weights[:scene_count], samples_per_rank)
        scene_paths = _selected_scene_paths(rank_dir, indices)
        ego_shapes = _effective_shapes(scene_paths)
        rank_rows.append(
            {
                "rank": int(manifest["rank"]),
                "active_groups": int(np.any(weights[:scene_count] > 0.0, axis=1).sum()),
                "sampled_active_groups": int(indices.size),
            }
        )

        for lower in range(0, indices.size, batch_groups):
            selected = indices[lower : lower + batch_groups]
            selected_shapes = ego_shapes[lower : lower + batch_groups]
            candidate = np.asarray(trajectories[selected], dtype=np.float32)
            reward = np.asarray(rewards[selected], dtype=np.float64)
            weight = np.asarray(weights[selected], dtype=np.float64)
            expert = np.asarray(experts[selected], dtype=np.float32)
            safe = np.asarray(expert_safe[selected], dtype=np.float64) > 0.5
            if not np.all(weight.sum(axis=1) > 0.0):
                raise RuntimeError("systematic sampler returned an inactive replay row")

            candidate_mass = weight.sum(axis=1)
            expert_mass = expert_weight * safe.astype(np.float64)
            total_mass = candidate_mass + expert_mass
            candidate_centroid = np.divide(
                (candidate.astype(np.float64) * weight[..., None, None]).sum(axis=1),
                candidate_mass[:, None, None],
                out=np.zeros_like(expert, dtype=np.float64),
                where=candidate_mass[:, None, None] > 0.0,
            ).astype(np.float32)
            combined_centroid = np.divide(
                candidate_centroid.astype(np.float64) * candidate_mass[:, None, None]
                + expert.astype(np.float64) * expert_mass[:, None, None],
                total_mass[:, None, None],
                out=np.zeros_like(expert, dtype=np.float64),
                where=total_mass[:, None, None] > 0.0,
            ).astype(np.float32)

            positive = weight > 0.0
            masked_reward = np.where(positive, reward, -np.inf)
            best = masked_reward.argmax(axis=1)
            reward_gain_rows.append(
                _weighted(reward, weight) - reward[:, 0]
            )

            for horizon_name, horizon in (
                ("candidate_prefix", min(candidate_horizon, full_horizon)),
                ("full_trajectory", full_horizon),
            ):
                candidate_metrics = _trajectory_metrics(
                    candidate,
                    horizon=horizon,
                    config=config,
                    ego_shapes=selected_shapes,
                )
                expert_metrics = _trajectory_metrics(
                    expert,
                    horizon=horizon,
                    config=config,
                    ego_shapes=selected_shapes,
                )
                candidate_centroid_metrics = _trajectory_metrics(
                    candidate_centroid,
                    horizon=horizon,
                    config=config,
                    ego_shapes=selected_shapes,
                )
                combined_centroid_metrics = _trajectory_metrics(
                    combined_centroid,
                    horizon=horizon,
                    config=config,
                    ego_shapes=selected_shapes,
                )
                bucket = collected.setdefault(horizon_name, {})
                rows = np.arange(selected.size)

                for metric, candidate_value in candidate_metrics.items():
                    det = candidate_value[:, 0].astype(np.float64)
                    candidate_mean = _weighted(candidate_value, weight)
                    best_value = candidate_value[rows, best].astype(np.float64)
                    expert_value = expert_metrics[metric].astype(np.float64)
                    candidate_centroid_value = candidate_centroid_metrics[metric].astype(
                        np.float64
                    )
                    combined_centroid_value = combined_centroid_metrics[metric].astype(
                        np.float64
                    )
                    combined = np.divide(
                        candidate_mean * candidate_mass + expert_value * expert_mass,
                        total_mass,
                        out=np.zeros_like(candidate_mean),
                        where=total_mass > 0.0,
                    )
                    metric_bucket = bucket.setdefault(metric, [])
                    metric_bucket.extend(
                        [
                            candidate_mean - det,
                            best_value - det,
                            np.where(safe, expert_value - det, np.nan),
                            combined - det,
                            candidate_centroid_value - det,
                            combined_centroid_value - det,
                            det,
                            candidate_mean,
                            combined,
                            candidate_centroid_value,
                            combined_centroid_value,
                        ]
                    )

    # Lists were appended in a fixed group of arrays per metric.
    output_horizons: dict[str, Any] = {}
    labels = (
        "positive_candidate_delta_vs_det",
        "best_positive_delta_vs_det",
        "safe_expert_delta_vs_det",
        "combined_target_delta_vs_det",
        "candidate_centroid_delta_vs_det",
        "combined_centroid_delta_vs_det",
        "deterministic_value",
        "positive_candidate_value",
        "combined_target_value",
        "candidate_centroid_value",
        "combined_centroid_value",
    )
    for horizon_name, metric_map in collected.items():
        horizon_out: dict[str, Any] = {}
        for metric, interleaved in metric_map.items():
            metric_out: dict[str, Any] = {}
            for offset, label in enumerate(labels):
                metric_out[label] = _describe(
                    np.concatenate(interleaved[offset::len(labels)])
                )
            horizon_out[metric] = metric_out
        output_horizons[horizon_name] = horizon_out

    sampled = sum(row["sampled_active_groups"] for row in rank_rows)
    return {
        "contract": {
            "population": "stored-weight-positive AWR groups only",
            "candidate_zero": "zero-noise unguided deployment trajectory",
            "metric_direction": "all reported deltas are target minus candidate zero; positive is better",
            "candidate_target": "stored positive AWR weights",
            "combined_target": "positive candidate weights plus safe expert anchor weight",
            "centroid": (
                "raw trajectory-coordinate weighted mean; diagnostic proxy for the plain-MSE "
                "conditional mean, not an assertion that diffusion always emits this path"
            ),
            "not_checkpoint_evidence": True,
        },
        "replay_root": str(replay_root),
        "config_path": str(config_path),
        "reward_profile": effective.get("reward", {}).get("reward_profile"),
        "sampled_active_groups": int(sampled),
        "samples_per_rank": int(samples_per_rank),
        "candidate_horizon_steps": int(min(candidate_horizon, full_horizon)),
        "full_horizon_steps": int(full_horizon),
        "expert_weight": float(expert_weight),
        "ego_shape_source": (
            "scene NPZ, cached per recording directory; canonical legacy X2 width override"
        ),
        "reward_gain_vs_deterministic": _describe(np.concatenate(reward_gain_rows)),
        "horizons": output_horizons,
        "ranks": rank_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expert-weight", type=float, default=0.4)
    parser.add_argument("--samples-per-rank", type=int, default=2048)
    parser.add_argument("--batch-groups", type=int, default=128)
    parser.add_argument("--candidate-horizon", type=int, default=40)
    args = parser.parse_args()
    result = analyze(
        args.replay_root,
        config_path=args.config,
        expert_weight=args.expert_weight,
        samples_per_rank=args.samples_per_rank,
        batch_groups=args.batch_groups,
        candidate_horizon=args.candidate_horizon,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
