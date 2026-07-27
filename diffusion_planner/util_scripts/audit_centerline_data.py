#!/usr/bin/env python3
"""Audit logged ego trajectories against vector-map route centerlines.

This is a data-quality audit, not a training metric.  It uses point-to-segment
distance (rather than nearest sampled point), reports the logged expert and
the current-frame route alignment separately, and stratifies by the first
dataset directory component (vehicle/source).  The source manifests are never
modified.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path

import numpy as np


def _valid_route(route: np.ndarray) -> np.ndarray:
    # Geometry channels are x, y, dx, dy.  A point is valid only when all four
    # channels are non-padding; this avoids treating a zero tangent as a real
    # centerline sample.
    return np.any(np.abs(route[..., :4]) > 1e-6, axis=-1)


def _point_segment_distance(points: np.ndarray, route: np.ndarray) -> np.ndarray:
    """Return [num_points] nearest distance to valid route centerline segments."""
    center = route[..., :2].reshape(-1, 2)
    valid = _valid_route(route).reshape(-1)
    if center.shape[0] < 2:
        return np.full(points.shape[0], np.nan, dtype=np.float32)
    seg_valid = valid[:-1] & valid[1:]
    start = center[:-1][seg_valid]
    end = center[1:][seg_valid]
    if start.shape[0] == 0:
        return np.full(points.shape[0], np.nan, dtype=np.float32)
    delta = end - start
    denom = np.maximum(np.sum(delta * delta, axis=-1), 1e-10)
    out = np.empty(points.shape[0], dtype=np.float32)
    # Bound temporary memory for the 80 x 500 segment case.
    for lo in range(0, points.shape[0], 256):
        q = points[lo : lo + 256]
        rel = q[:, None, :] - start[None, :, :]
        alpha = np.clip(np.sum(rel * delta[None, :, :], axis=-1) / denom[None, :], 0.0, 1.0)
        nearest = start[None, :, :] + alpha[..., None] * delta[None, :, :]
        out[lo : lo + q.shape[0]] = np.sqrt(
            np.min(np.sum((q[:, None, :] - nearest) ** 2, axis=-1), axis=-1)
        )
    return out


def _current_route_distance(route: np.ndarray) -> float:
    d = _point_segment_distance(np.zeros((1, 2), dtype=np.float32), route)
    return float(d[0]) if np.isfinite(d[0]) else np.nan


def _one(path: str) -> tuple[str, dict[str, float]]:
    source = "unknown"
    parts = Path(path).parts
    try:
        # .../<dataset>/<source>/...; retain both source and vehicle where present.
        marker = parts.index("20260707_vehicle_params_with_mirror")
        source = "/".join(parts[marker + 1 : marker + 3]) or "unknown"
    except ValueError:
        if len(parts) >= 3:
            source = "/".join(parts[-3:-1])
    try:
        with np.load(path, allow_pickle=False) as z:
            route = np.asarray(z["route_lanes"], dtype=np.float32)
            future = np.asarray(z["ego_agent_future"], dtype=np.float32)
        if future.shape[-1] < 2 or route.ndim != 3:
            return source, {"invalid": 1.0}
        future_xy = future[..., :2]
        distance = _point_segment_distance(future_xy, route)
        finite = np.isfinite(distance)
        if not finite.any():
            return source, {"no_route": 1.0}
        distance = distance[finite]
        moved = float(np.linalg.norm(future_xy[-1])) >= 1.0
        return source, {
            "count": 1.0,
            "moving": float(moved),
            "current_distance": _current_route_distance(route),
            "mean": float(np.mean(distance)),
            "p50": float(np.percentile(distance, 50)),
            "p95": float(np.percentile(distance, 95)),
            "within_0_3": float(np.mean(distance <= 0.3)),
            "within_0_5": float(np.mean(distance <= 0.5)),
            "within_1": float(np.mean(distance <= 1.0)),
            "t0_5": float(distance[min(4, len(distance) - 1)]),
            "t2": float(distance[min(19, len(distance) - 1)]),
            "t4": float(distance[min(39, len(distance) - 1)]),
            "t8": float(distance[min(79, len(distance) - 1)]),
        }
    except Exception:
        return source, {"invalid": 1.0}


def _summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    out: dict[str, float] = {"samples": float(len(rows))}
    for key in keys:
        values = np.asarray([r[key] for r in rows if key in r and np.isfinite(r[key])], dtype=float)
        if values.size:
            out[f"{key}_mean"] = float(values.mean())
            if key in {"mean", "p50", "p95", "current_distance"}:
                out[f"{key}_p50"] = float(np.percentile(values, 50))
                out[f"{key}_p90"] = float(np.percentile(values, 90))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--sample", type=int, default=50000)
    parser.add_argument("--workers", type=int, default=max(mp.cpu_count() - 2, 1))
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    with open(args.manifest) as f:
        paths = json.load(f)
    if args.sample and len(paths) > args.sample:
        rng = np.random.default_rng(args.seed)
        paths = [paths[i] for i in rng.choice(len(paths), args.sample, replace=False)]
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    with mp.Pool(args.workers) as pool:
        for source, row in pool.imap_unordered(_one, paths, chunksize=64):
            grouped[source].append(row)
    result = {
        "manifest": args.manifest,
        "selected": len(paths),
        "by_source": {source: _summarize(rows) for source, rows in sorted(grouped.items())},
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
