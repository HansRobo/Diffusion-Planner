#!/usr/bin/env python3
"""Select reproducible scene groups for formal replay visualizations.

The selector reads only writer-committed groups (``scene_paths.jsonl``) and
never inspects an unwritten memmap tail.  It ranks complementary evidence
types rather than choosing attractive examples by hand:

* deterministic hard failure recovered by another candidate;
* lateral or longitudinal endpoint branching with a positive reward gain;
* largest reward improvement over candidate 0;
* diverse all-zero groups, which expose the exploration failure pool.

The output contains rank-local indices consumed directly by the colleague's
``grpo_viz.py --replay_cache_root ... --replay_rank ... --replay_indices``.
These are exploration/target examples, not proof of a trained checkpoint gain.
"""

from __future__ import annotations

import argparse
import heapq
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Candidate:
    score: float
    rank: int
    index: int
    metrics: dict[str, float | int]


def _push(heap: list, item: Candidate, limit: int) -> None:
    key = (item.score, -item.rank, -item.index, item)
    if len(heap) < limit:
        heapq.heappush(heap, key)
    elif key[:3] > heap[0][:3]:
        heapq.heapreplace(heap, key)


def _committed_count(rank_dir: Path, allow_partial: bool) -> tuple[int, bool]:
    manifest = rank_dir / "manifest.json"
    if manifest.exists():
        return int(json.loads(manifest.read_text())["scene_count"]), True
    if not allow_partial:
        raise RuntimeError(f"{rank_dir} is not strictly closed")
    paths = rank_dir / "scene_paths.jsonl"
    with paths.open("rb") as stream:
        return sum(1 for _ in stream), False


def _scene_paths(rank_dir: Path, indices: set[int]) -> dict[int, str]:
    found: dict[int, str] = {}
    if not indices:
        return found
    with (rank_dir / "scene_paths.jsonl").open() as stream:
        for index, line in enumerate(stream):
            if index not in indices:
                continue
            payload = json.loads(line)
            if isinstance(payload, str):
                found[index] = payload
            elif isinstance(payload, dict):
                found[index] = str(payload.get("path") or payload.get("scene_path"))
            else:
                raise TypeError(f"unsupported scene-path row at {rank_dir}:{index}")
            if len(found) == len(indices):
                break
    if set(found) != indices:
        raise RuntimeError(f"missing selected paths in {rank_dir}: {indices - set(found)}")
    return found


def _endpoint_route_coordinates(trajectories: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project candidate endpoints into candidate-0 path coordinates.

    ``s`` follows the closest point and local tangent of the deterministic
    10 Hz path; ``d`` is signed normal offset.  This prevents a curved road's
    ego-frame y change from being mislabeled as lateral multimodality.
    """

    trajectory = np.asarray(trajectories, dtype=np.float32)
    reference = trajectory[:, 0, :, :2]
    if trajectory.shape[-1] >= 4:
        reference_heading = trajectory[:, 0, :, 2:4]
    else:
        reference_heading = np.zeros_like(reference)
        reference_heading[..., 0] = 1.0
    endpoints = trajectory[:, :, -1, :2]
    tangent = np.zeros_like(reference)
    if reference.shape[1] > 1:
        tangent[:, 0] = reference[:, 1] - reference[:, 0]
        tangent[:, -1] = reference[:, -1] - reference[:, -2]
    if reference.shape[1] > 2:
        tangent[:, 1:-1] = reference[:, 2:] - reference[:, :-2]
    tangent_norm = np.linalg.norm(tangent, axis=-1, keepdims=True)
    heading_norm = np.linalg.norm(reference_heading, axis=-1, keepdims=True)
    heading = np.divide(
        reference_heading,
        heading_norm,
        out=np.zeros_like(reference_heading),
        where=heading_norm > 1e-6,
    )
    heading[..., 0] = np.where(heading_norm[..., 0] > 1e-6, heading[..., 0], 1.0)
    tangent = np.divide(
        tangent,
        tangent_norm,
        out=heading.copy(),
        where=tangent_norm > 1e-6,
    )

    delta_to_reference = endpoints[:, :, None, :] - reference[:, None, :, :]
    nearest = np.square(delta_to_reference).sum(axis=-1).argmin(axis=-1)
    batch = np.arange(len(trajectory))[:, None]
    nearest_reference = reference[batch, nearest]
    nearest_tangent = tangent[batch, nearest]
    endpoint_delta = endpoints - nearest_reference

    origin_step = np.linalg.norm(reference[:, :1], axis=-1)
    later_steps = np.linalg.norm(np.diff(reference, axis=1), axis=-1)
    arc = np.cumsum(np.concatenate([origin_step, later_steps], axis=1), axis=1)
    s = arc[batch, nearest] + np.sum(endpoint_delta * nearest_tangent, axis=-1)
    d = (
        nearest_tangent[..., 0] * endpoint_delta[..., 1]
        - nearest_tangent[..., 1] * endpoint_delta[..., 0]
    )
    return s, d


def select(
    replay_root: Path,
    *,
    per_category: int = 4,
    chunk_size: int = 4096,
    reward_margin: float = 0.01,
    hard_recovery_threshold: float = 0.9,
    allow_partial: bool = False,
) -> dict:
    categories: dict[str, list] = {
        "deterministic_hard_recovery": [],
        "hard_gate_boundary_cliff": [],
        "lateral_multimodality": [],
        "longitudinal_multimodality": [],
        "largest_reward_gain": [],
        "all_zero_diverse_failure": [],
        "wide_endpoint_spread_audit": [],
    }
    rank_counts = []
    group_size = future_len = None

    for rank_dir in sorted(replay_root.glob("rank_[0-9][0-9][0-9][0-9]")):
        rank = int(rank_dir.name.rsplit("_", 1)[1])
        count, strict = _committed_count(rank_dir, allow_partial)
        rewards = np.load(rank_dir / "rewards.npy", mmap_mode="r")
        trajectories = np.load(rank_dir / "trajectories.npy", mmap_mode="r")
        if rewards.shape[0] < count or trajectories.shape[0] < count:
            raise RuntimeError(f"committed boundary exceeds arrays in {rank_dir}")
        if rewards.shape[1] != trajectories.shape[1]:
            raise RuntimeError(f"K mismatch in {rank_dir}")
        group_size = int(rewards.shape[1]) if group_size is None else group_size
        future_len = int(trajectories.shape[2]) if future_len is None else future_len
        rank_counts.append({"rank": rank, "scene_count": count, "strict_manifest": strict})

        for start in range(0, count, chunk_size):
            stop = min(count, start + chunk_size)
            reward = np.asarray(rewards[start:stop], dtype=np.float32)
            trajectory = np.asarray(trajectories[start:stop], dtype=np.float32)
            endpoint = trajectory[:, :, -1, :2]
            route_s, route_d = _endpoint_route_coordinates(trajectory)
            det = reward[:, 0]
            best_index = reward.argmax(axis=1)
            best = reward[np.arange(len(reward)), best_index]
            gain = best - det
            x_range = endpoint[:, :, 0].max(axis=1) - endpoint[:, :, 0].min(axis=1)
            y_range = endpoint[:, :, 1].max(axis=1) - endpoint[:, :, 1].min(axis=1)
            bbox_diag = np.hypot(x_range, y_range)
            high_count = (reward >= hard_recovery_threshold).sum(axis=1)
            all_zero = np.all(np.abs(reward) <= 1e-8, axis=1)
            positive = reward > (det[:, None] + reward_margin)

            for local in range(stop - start):
                index = start + local
                metrics = {
                    "det_reward": float(det[local]),
                    "best_reward": float(best[local]),
                    "reward_gain": float(gain[local]),
                    "best_candidate": int(best_index[local]),
                    "high_reward_candidate_count": int(high_count[local]),
                    "endpoint_x_range_m": float(x_range[local]),
                    "endpoint_y_range_m": float(y_range[local]),
                    "endpoint_bbox_diagonal_m": float(bbox_diag[local]),
                }
                active_endpoint = endpoint[local, positive[local]]
                if len(active_endpoint) >= 2:
                    active_x_range = float(np.ptp(active_endpoint[:, 0]))
                    active_y_range = float(np.ptp(active_endpoint[:, 1]))
                    active_s_range = float(np.ptp(route_s[local, positive[local]]))
                    active_d_range = float(np.ptp(route_d[local, positive[local]]))
                else:
                    active_x_range = active_y_range = 0.0
                    active_s_range = active_d_range = 0.0
                best_endpoint_delta = endpoint[local, best_index[local]] - endpoint[local, 0]
                metrics.update(
                    {
                        "positive_advantage_candidate_count": int(positive[local].sum()),
                        "positive_endpoint_x_range_m": active_x_range,
                        "positive_endpoint_y_range_m": active_y_range,
                        "positive_route_s_range_m": active_s_range,
                        "positive_route_d_range_m": active_d_range,
                        "best_endpoint_delta_s_m": float(
                            route_s[local, best_index[local]] - route_s[local, 0]
                        ),
                        "best_endpoint_delta_d_m": float(
                            route_d[local, best_index[local]] - route_d[local, 0]
                        ),
                        "best_endpoint_delta_x_m": float(best_endpoint_delta[0]),
                        "best_endpoint_delta_y_m": float(best_endpoint_delta[1]),
                        "best_endpoint_displacement_m": float(np.linalg.norm(best_endpoint_delta)),
                    }
                )
                candidate = lambda score: Candidate(float(score), rank, index, metrics)
                is_hard_recovery = (
                    det[local] <= 1e-8 and best[local] >= hard_recovery_threshold
                )
                best_displacement = float(np.linalg.norm(best_endpoint_delta))
                if is_hard_recovery and best_displacement >= 0.1:
                    _push(
                        categories["deterministic_hard_recovery"],
                        candidate(
                            best[local]
                            + gain[local]
                            - 0.01 * max(0.0, float(bbox_diag[local]) - 20.0)
                        ),
                        per_category,
                    )
                if is_hard_recovery and best_displacement <= 0.05:
                    _push(
                        categories["hard_gate_boundary_cliff"],
                        candidate(best[local] + gain[local] - best_displacement),
                        per_category,
                    )
                if (
                    positive[local].sum() >= 2
                    and 1.0 <= active_d_range <= 6.0
                    and active_s_range <= 15.0
                    and best[local] >= 0.8
                ):
                    _push(
                        categories["lateral_multimodality"],
                        candidate(
                            best[local]
                            + 2.0 * gain[local]
                            + 0.1 * min(int(positive[local].sum()), 5)
                            - 0.15 * abs(active_d_range - 3.5)
                            - 0.02 * active_s_range
                        ),
                        per_category,
                    )
                if (
                    positive[local].sum() >= 2
                    and 2.0 <= active_s_range <= 30.0
                    and active_d_range <= 2.0
                    and best[local] >= 0.8
                ):
                    _push(
                        categories["longitudinal_multimodality"],
                        candidate(
                            best[local]
                            + 2.0 * gain[local]
                            + 0.1 * min(int(positive[local].sum()), 5)
                            - 0.03 * abs(active_s_range - 12.0)
                            - 0.1 * active_d_range
                        ),
                        per_category,
                    )
                if gain[local] >= reward_margin:
                    _push(categories["largest_reward_gain"], candidate(gain[local]), per_category)
                if all_zero[local] and bbox_diag[local] >= 1.0:
                    _push(
                        categories["all_zero_diverse_failure"],
                        candidate(bbox_diag[local]),
                        per_category,
                    )
                if y_range[local] > 15.0 or x_range[local] > 40.0:
                    _push(
                        categories["wide_endpoint_spread_audit"],
                        candidate(bbox_diag[local]),
                        per_category,
                    )

    selected: dict[str, list[Candidate]] = {}
    selected_by_rank: dict[int, set[int]] = {}
    for name, heap in categories.items():
        rows = [entry[3] for entry in sorted(heap, reverse=True)]
        selected[name] = rows
        for row in rows:
            selected_by_rank.setdefault(row.rank, set()).add(row.index)

    paths_by_rank = {
        rank: _scene_paths(replay_root / f"rank_{rank:04d}", indices)
        for rank, indices in selected_by_rank.items()
    }
    payload_categories = {}
    for name, rows in selected.items():
        payload_categories[name] = [
            {
                "score": row.score,
                "rank": row.rank,
                "rank_local_index": row.index,
                "scene_path": paths_by_rank[row.rank][row.index],
                "metrics": row.metrics,
            }
            for row in rows
        ]

    return {
        "schema": "formal_replay_visual_selection_v1",
        "replay_root": str(replay_root.resolve()),
        "source_complete": all(row["strict_manifest"] for row in rank_counts),
        "prefix_groups": sum(row["scene_count"] for row in rank_counts),
        "group_size": group_size,
        "future_len": future_len,
        "parameters": {
            "per_category": per_category,
            "reward_margin": reward_margin,
            "hard_recovery_threshold": hard_recovery_threshold,
        },
        "rank_counts": rank_counts,
        "categories": payload_categories,
        "contract": {
            "candidate_zero": "zero-noise unguided deployment behavior",
            "selection_role": "reproducible exploration/target visualization; not checkpoint evidence",
            "endpoint_modes": "geometric diversity only; no semantic maneuver label",
            "positive_multimodality": "spread is measured only among candidates above the AWR reward margin",
            "route_coordinates": "endpoint s/d from closest point and local tangent on candidate-0 10 Hz path",
            "hard_gate_boundary_cliff": "diagnostic only: large reward flip with <=5 cm endpoint displacement",
            "wide_endpoint_spread_audit": "ambiguous diagnostic category; scene geometry must decide useful go/yield diversity versus an outlier",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--reward-margin", type=float, default=0.01)
    parser.add_argument("--hard-recovery-threshold", type=float, default=0.9)
    parser.add_argument("--allow-partial-prefix", action="store_true")
    args = parser.parse_args()
    payload = select(
        args.replay_root,
        per_category=args.per_category,
        chunk_size=args.chunk_size,
        reward_margin=args.reward_margin,
        hard_recovery_threshold=args.hard_recovery_threshold,
        allow_partial=args.allow_partial_prefix,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
