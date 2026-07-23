"""Audit normalization scales on a real NPZ manifest.

This is intentionally a diagnostic tool, not a replacement for the checked-in
normalization contract.  It uses the same padding semantics as the model and
keeps bounded/categorical channels separate from continuous geometry.  The
result records the manifest hash, sample seed, schema, and valid-element counts
so a candidate normalization file can be reproduced before a fresh training
run is started.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_NORMALIZATION = Path(__file__).resolve().parents[1] / "normalization.json"


@dataclass
class _Accumulator:
    count: int
    total: np.ndarray
    total_sq: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    zero_rows: int

    @classmethod
    def create(cls, width: int) -> "_Accumulator":
        return cls(
            count=0,
            total=np.zeros(width, dtype=np.float64),
            total_sq=np.zeros(width, dtype=np.float64),
            minimum=np.full(width, np.inf, dtype=np.float64),
            maximum=np.full(width, -np.inf, dtype=np.float64),
            zero_rows=0,
        )

    def add(self, values: np.ndarray, valid: np.ndarray | None = None) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        else:
            values = values.reshape(-1, values.shape[-1])
        if valid is not None:
            values = values[np.asarray(valid, dtype=bool).reshape(-1)]
        if values.size == 0:
            return
        if values.ndim != 2:
            raise ValueError(f"Expected a two-dimensional feature matrix, got {values.shape}")
        if values.shape[1] != self.total.shape[0]:
            raise ValueError(
                f"Feature width changed from {self.total.shape[0]} to {values.shape[1]}"
            )
        self.count += int(values.shape[0])
        self.total += values.sum(axis=0)
        self.total_sq += np.square(values).sum(axis=0)
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))
        self.zero_rows += int(np.all(values == 0, axis=1).sum())


def _pose_with_heading(pose: np.ndarray) -> np.ndarray:
    if pose.shape[-1] != 3:
        raise ValueError(f"Expected [x, y, heading] input, got {pose.shape}")
    return np.concatenate(
        [pose[..., :2], np.cos(pose[..., 2:3]), np.sin(pose[..., 2:3])], axis=-1
    )


def _update(summary: dict[str, _Accumulator], key: str, values: np.ndarray, valid=None) -> None:
    values = np.asarray(values)
    width = values.shape[-1] if values.ndim else 1
    if key not in summary:
        summary[key] = _Accumulator.create(width)
    summary[key].add(values, valid)


def _worker(
    paths: list[str],
) -> tuple[dict[str, _Accumulator], int, dict[str, int], list[dict[str, str]]]:
    summary: dict[str, _Accumulator] = {}
    errors = 0
    versions: dict[str, int] = {}
    error_examples: list[dict[str, str]] = []
    for path in paths:
        try:
            with np.load(path, allow_pickle=False) as archive:
                required = {
                    "ego_agent_past",
                    "ego_current_state",
                    "ego_agent_future",
                    "neighbor_agents_past",
                    "static_objects",
                    "lanes",
                    "lanes_speed_limit",
                    "lanes_has_speed_limit",
                    "route_lanes",
                    "route_lanes_speed_limit",
                    "route_lanes_has_speed_limit",
                    "polygons",
                    "line_strings",
                    "goal_pose",
                    "ego_shape",
                }
                missing = required.difference(archive.files)
                if missing:
                    raise KeyError(f"missing fields: {sorted(missing)}")

                if "version" in archive.files:
                    version = int(np.asarray(archive["version"]).reshape(-1)[0])
                    versions[str(version)] = versions.get(str(version), 0) + 1

                ego_past = archive["ego_agent_past"]
                ego_future = archive["ego_agent_future"]
                _update(summary, "ego_agent_past_pose", _pose_with_heading(ego_past))
                _update(summary, "ego_agent_future_pose", _pose_with_heading(ego_future))
                velocity = np.diff(
                    np.concatenate(
                        [np.zeros_like(ego_future[:1, :2]), ego_future[:, :2]], axis=0
                    ),
                    axis=0,
                )
                # ``waypoints_to_velocity`` keeps the orientation channels from
                # the corresponding future waypoint; only x/y are differenced.
                velocity_heading = np.cos(ego_future[:, 2:3])
                velocity_yaw = np.sin(ego_future[:, 2:3])
                _update(
                    summary,
                    "ego_velocity",
                    np.concatenate([velocity, velocity_heading, velocity_yaw], axis=-1),
                )
                _update(summary, "ego_current_state", archive["ego_current_state"])
                # The model converts goal heading to cos/sin before applying the
                # observation normalizer, so audit the four-channel contract.
                _update(summary, "goal_pose", _pose_with_heading(archive["goal_pose"]))
                _update(summary, "ego_shape", archive["ego_shape"])

                neighbor = archive["neighbor_agents_past"]
                neighbor_valid = np.any(neighbor[..., :8] != 0, axis=-1)
                _update(summary, "neighbor_agents_past", neighbor, neighbor_valid)
                _update(summary, "neighbor_agent_type", neighbor[..., 8:], neighbor_valid)

                static = archive["static_objects"]
                _update(summary, "static_objects", static, np.any(static != 0, axis=-1))

                lanes = archive["lanes"]
                lane_valid = np.any(lanes[..., :8] != 0, axis=-1)
                _update(summary, "lanes_geometry", lanes[..., :8], lane_valid)
                _update(summary, "lanes_attributes", lanes[..., 8:], lane_valid)
                lane_speed_valid = archive["lanes_has_speed_limit"].reshape(-1).astype(bool)
                _update(summary, "lanes_speed_limit", archive["lanes_speed_limit"], lane_speed_valid)

                route = archive["route_lanes"]
                route_valid = np.any(route[..., :8] != 0, axis=-1)
                _update(summary, "route_lanes_geometry", route[..., :8], route_valid)
                _update(summary, "route_lanes_attributes", route[..., 8:], route_valid)
                route_speed_valid = archive["route_lanes_has_speed_limit"].reshape(-1).astype(bool)
                _update(
                    summary,
                    "route_lanes_speed_limit",
                    archive["route_lanes_speed_limit"],
                    route_speed_valid,
                )

                polygons = archive["polygons"]
                lines = archive["line_strings"]
                _update(summary, "polygons", polygons, np.any(polygons != 0, axis=-1))
                _update(summary, "line_strings", lines, np.any(lines != 0, axis=-1))
        except Exception as error:
            errors += 1
            if len(error_examples) < 5:
                error_examples.append({"path": path, "error": repr(error)})
    return summary, errors, versions, error_examples


def _merge(
    destination: dict[str, _Accumulator],
    source: dict[str, _Accumulator],
) -> None:
    for key, value in source.items():
        if key not in destination:
            destination[key] = _Accumulator.create(value.total.shape[0])
        target = destination[key]
        target.count += value.count
        target.total += value.total
        target.total_sq += value.total_sq
        target.minimum = np.minimum(target.minimum, value.minimum)
        target.maximum = np.maximum(target.maximum, value.maximum)
        target.zero_rows += value.zero_rows


def _manifest_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _serialize(summary: dict[str, _Accumulator], normalization: dict) -> dict:
    result = {}
    for key, value in sorted(summary.items()):
        if value.count == 0:
            result[key] = {"count": 0, "note": "no valid elements in sampled manifest"}
            continue
        mean = value.total / value.count
        variance = np.maximum(value.total_sq / value.count - np.square(mean), 0.0)
        std = np.sqrt(variance)
        entry = {
            "count": value.count,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "min": value.minimum.tolist(),
            "max": value.maximum.tolist(),
            "zero_row_fraction": value.zero_rows / value.count,
        }
        # A normalized comparison is useful for deciding whether a scale is merely
        # deliberate range control or is producing a large population of outliers.
        config_key = {
            "ego_agent_past_pose": "ego_agent_past",
            "ego_current_state": "ego_current_state",
            "neighbor_agents_past": "neighbor_agents_past",
            "static_objects": "static_objects",
            "lanes_geometry": "lanes",
            "route_lanes_geometry": "route_lanes",
            "polygons": "polygons",
            "line_strings": "line_strings",
            "goal_pose": "goal_pose",
            "lanes_speed_limit": "lanes_speed_limit",
            "route_lanes_speed_limit": "route_lanes_speed_limit",
            "ego_velocity": "ego_velocity",
        }.get(key)
        if config_key in normalization:
            configured_mean = np.asarray(normalization[config_key]["mean"], dtype=np.float64)
            configured_std = np.asarray(normalization[config_key]["std"], dtype=np.float64)
            if configured_mean.size >= mean.size:
                configured_mean = configured_mean[: mean.size]
                configured_std = configured_std[: mean.size]
                normalized_mean = (mean - configured_mean) / configured_std
                normalized_std = std / configured_std
                entry["configured_normalized_mean"] = normalized_mean.tolist()
                entry["configured_normalized_std"] = normalized_std.tolist()
        result[key] = entry
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-list", required=True, type=Path)
    parser.add_argument("--normalization", type=Path, default=DEFAULT_NORMALIZATION)
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1 or args.workers < 1:
        raise ValueError("samples and workers must be positive")

    with args.train_list.open(encoding="utf-8") as stream:
        paths = json.load(stream)
    if not isinstance(paths, list) or not paths:
        raise ValueError("train list must be a non-empty JSON array")
    samples = min(args.samples, len(paths))
    rng = random.Random(args.seed)
    selected = [paths[index] for index in rng.sample(range(len(paths)), samples)]
    workers = min(args.workers, samples)
    chunks = [selected[index::workers] for index in range(workers)]

    context = mp.get_context("fork")
    summary: dict[str, _Accumulator] = {}
    errors = 0
    versions: dict[str, int] = {}
    error_examples: list[dict[str, str]] = []
    with context.Pool(workers) as pool:
        for partial, partial_errors, partial_versions, partial_error_examples in pool.imap_unordered(
            _worker, chunks
        ):
            _merge(summary, partial)
            errors += partial_errors
            error_examples.extend(partial_error_examples)
            error_examples = error_examples[:5]
            for version, count in partial_versions.items():
                versions[version] = versions.get(version, 0) + count

    with args.normalization.open(encoding="utf-8") as stream:
        normalization = json.load(stream)
    output = {
        "manifest": str(args.train_list),
        "manifest_sha256": _manifest_sha256(args.train_list),
        "normalization_file": str(args.normalization),
        "sample_count_requested": args.samples,
        "sample_count_used": samples,
        "manifest_count": len(paths),
        "seed": args.seed,
        "workers": workers,
        "errors": errors,
        "error_examples": error_examples,
        "data_versions": versions,
        "notes": {
            "pose_origin_is_valid": True,
            "neighbor_and_map_masks": "zero geometry rows only; categorical channels are not used to decide padding",
            "continuous_candidate_stats_only": True,
            "do_not_replace_checked_in_contract_without_fresh_train": True,
        },
        "stats": _serialize(summary, normalization),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sample_count": samples, "errors": errors}))


if __name__ == "__main__":
    main()
