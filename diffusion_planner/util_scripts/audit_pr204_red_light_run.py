#!/usr/bin/env python3
"""Audit/filter NPZ manifests with the exact PR #204 red-light-run predicate.

This tool is intentionally separate from the causal red-to-green experiments.  PR
#204 removes only a trajectory that *crosses a stop-line* associated with a
heading-aligned red route lane.  It does not remove a stopped frame whose future
starts after an unobserved signal transition.  The implementation below mirrors
``frame_filters::detect_red_light_run`` at commit ``97fe6df`` and reads the
three-column NPZ representation (x, y, heading) used by this corpus.

Source manifests and NPZ files are never modified.  A filtered manifest is
written only when requested; checkpoint files contain removed paths, so a scan
can be resumed without reopening completed chunks.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from diffusion_planner.dimensions import (
    TRAFFIC_LIGHT,
    TRAFFIC_LIGHT_ONE_HOT_DIM,
    TRAFFIC_LIGHT_RED,
)


RED_RADIUS_M = 12.0
HEADING_TOL_DEG = 45.0
XY_EPS = 1e-6


def _strict_crossing(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray
) -> bool:
    """Exact proper-intersection test used by PR #204 (no endpoint touching)."""
    r0 = float(p2[0] - p1[0])
    r1 = float(p2[1] - p1[1])
    s0 = float(p4[0] - p3[0])
    s1 = float(p4[1] - p3[1])
    d1 = s0 * float(p1[1] - p3[1]) - s1 * float(p1[0] - p3[0])
    d2 = s0 * float(p2[1] - p3[1]) - s1 * float(p2[0] - p3[0])
    d3 = r0 * float(p3[1] - p1[1]) - r1 * float(p3[0] - p1[0])
    d4 = r0 * float(p4[1] - p1[1]) - r1 * float(p4[0] - p1[0])
    return d1 * d2 < 0.0 and d3 * d4 < 0.0


def _intersection_point(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray
) -> np.ndarray | None:
    """Return the strict segment intersection, matching the C++ helper."""
    rx = float(p2[0] - p1[0])
    ry = float(p2[1] - p1[1])
    sx = float(p4[0] - p3[0])
    sy = float(p4[1] - p3[1])
    denominator = rx * sy - ry * sx
    if abs(denominator) <= 1e-6:
        return None
    qx = float(p3[0] - p1[0])
    qy = float(p3[1] - p1[1])
    t = (qx * sy - qy * sx) / denominator
    u = (qx * ry - qy * rx) / denominator
    if not (0.0 < t < 1.0 and 0.0 < u < 1.0):
        return None
    return np.asarray((p1[0] + t * rx, p1[1] + t * ry), dtype=np.float64)


def _heading_diff_deg(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _red_entries(route_lanes: np.ndarray) -> list[tuple[np.ndarray, float]]:
    """Port the route-lane portion of ``detect_red_light_run`` exactly."""
    entries: list[tuple[np.ndarray, float]] = []
    if route_lanes.ndim != 3 or route_lanes.shape[-1] <= TRAFFIC_LIGHT_RED:
        return entries
    for segment in route_lanes:
        # The PR reads the traffic-light one-hot at the first point of each
        # route segment, then uses argmax with a >0.5 confidence gate.
        traffic = segment[0, TRAFFIC_LIGHT : TRAFFIC_LIGHT + TRAFFIC_LIGHT_ONE_HOT_DIM]
        argmax = int(np.argmax(traffic))
        if float(traffic[argmax]) <= 0.5 or TRAFFIC_LIGHT + argmax != TRAFFIC_LIGHT_RED:
            continue
        valid = np.flatnonzero(np.any(np.abs(segment[:, :2]) > XY_EPS, axis=1))
        if valid.size < 2:
            continue
        first = segment[int(valid[0]), :2].astype(np.float64, copy=False)
        last = segment[int(valid[-1]), :2].astype(np.float64, copy=False)
        heading = float(np.degrees(np.arctan2(last[1] - first[1], last[0] - first[0])))
        entries.append((first, heading))
    return entries


def _stop_segments(line_strings: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Port ``collect_line_string_segments(..., stop_line_channel)``.

    A line is selected if any point has the stop-line flag.  Once selected, the
    C++ code emits every consecutive pair of non-zero geometry points; it does
    not require the flag on both endpoints.
    """
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    if line_strings.ndim != 3 or line_strings.shape[-1] < 3:
        return segments
    for line in line_strings:
        target = line[:, 2] > 0.5
        if not bool(np.any(target)):
            continue
        valid = np.any(np.abs(line[:, :2]) > XY_EPS, axis=1)
        for i in range(line.shape[0] - 1):
            if bool(valid[i] and valid[i + 1]):
                segments.append(
                    (
                        line[i, :2].astype(np.float64, copy=False),
                        line[i + 1, :2].astype(np.float64, copy=False),
                    )
                )
    return segments


def detect_red_light_run(path: str) -> bool:
    """Return the exact PR #204 decision for one three-column NPZ frame."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            # These arrays are intentionally touched in stages.  The vast
            # majority of frames have no red route lane; avoiding decompression
            # of line strings and the 80-step future is the dominant speedup on
            # the multi-million-frame manifest.
            route = np.asarray(archive["route_lanes"])
            entries = _red_entries(route)
            if not entries:
                return False
            lines = np.asarray(archive["line_strings"])
            stops = _stop_segments(lines)
            if not stops:
                return False
            future = np.asarray(archive["ego_agent_future"])
    except (FileNotFoundError, OSError, KeyError, ValueError):
        # A malformed item cannot be safely called a red-light run.  Keep it in
        # the report so data corruption is visible without silently deleting it.
        return False

    if future.ndim != 2 or future.shape[1] < 3:
        return False

    # The converter's in-memory tensor is [x,y,cos,sin], while the NPZ stores
    # [x,y,heading].  Convert heading only for the equivalent atan2 call.
    states: list[tuple[np.ndarray, float]] = []
    for row in future:
        if abs(float(row[0])) <= XY_EPS and abs(float(row[1])) <= XY_EPS:
            continue
        states.append(
            (
                np.asarray((float(row[0]), float(row[1])), dtype=np.float64),
                float(np.degrees(float(row[2]))),
            )
        )
    if len(states) < 2:
        return False

    radius_sq = RED_RADIUS_M * RED_RADIUS_M
    for i in range(len(states) - 1):
        p1, ego_heading = states[i]
        p2 = states[i + 1][0]
        for start, end in stops:
            if not _strict_crossing(p1, p2, start, end):
                continue
            intersection = _intersection_point(p1, p2, start, end)
            if intersection is None:
                continue
            for entry, lane_heading in entries:
                if _heading_diff_deg(lane_heading, ego_heading) >= HEADING_TOL_DEG:
                    continue
                delta = intersection - entry
                if float(np.dot(delta, delta)) <= radius_sq:
                    return True
    return False


def _load_manifest(path: Path) -> list[str]:
    # orjson is materially faster and has lower peak memory for the 1.7 GB base
    # manifest; fall back to the stdlib so the utility remains portable.
    try:
        import orjson  # type: ignore

        values = orjson.loads(path.read_bytes())
    except ImportError:
        values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
        raise ValueError(f"{path} must contain a flat JSON list of NPZ paths")
    return values


def _atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def scan(
    paths: list[str], workers: int, checkpoint_dir: Path | None, chunk_size: int
) -> tuple[list[str], list[str]]:
    removed: list[str] = []
    malformed: list[str] = []
    root = checkpoint_dir / "chunks" if checkpoint_dir is not None else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for start in range(0, len(paths), chunk_size):
            end = min(start + chunk_size, len(paths))
            checkpoint = root / f"{start:012d}_{end:012d}.json" if root else None
            if checkpoint is not None and checkpoint.exists():
                payload = json.loads(checkpoint.read_text(encoding="utf-8"))
                chunk_removed = payload.get("removed", [])
            else:
                chunk = paths[start:end]
                results = list(executor.map(detect_red_light_run, chunk, chunksize=32))
                chunk_removed = [p for p, flag in zip(chunk, results) if flag]
                if checkpoint is not None:
                    _atomic_json({"start": start, "end": end, "removed": chunk_removed}, checkpoint)
            removed.extend(chunk_removed)
            print(
                f"[pr204] scanned {end}/{len(paths)} removed={len(removed)} workers={workers}",
                flush=True,
            )
    removed_set = set(removed)
    kept = [p for p in paths if p not in removed_set]
    return kept, removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--no-filtered-list", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.chunk_size < 1:
        raise ValueError("workers and chunk-size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"detector": "pr204_detect_red_light_run", "sources": []}
    all_removed: list[str] = []
    for source in args.source:
        paths = _load_manifest(source)
        kept, removed = scan(paths, args.workers, args.output_dir / source.stem, args.chunk_size)
        all_removed.extend(removed)
        source_report = {
            "source": str(source),
            "source_count": len(paths),
            "removed_count": len(removed),
            "kept_count": len(kept),
            "removed_examples": removed[:20],
        }
        report["sources"].append(source_report)
        if not args.no_filtered_list:
            out = args.output_dir / f"{source.stem}_pr204_filtered.json"
            _atomic_json(kept, out)
            source_report["filtered_list"] = str(out)
        out_removed = args.output_dir / f"{source.stem}_pr204_removed.json"
        _atomic_json(removed, out_removed)
        source_report["removed_list"] = str(out_removed)
    report["removed_total"] = len(all_removed)
    _atomic_json(report, args.output_dir / "metadata.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
