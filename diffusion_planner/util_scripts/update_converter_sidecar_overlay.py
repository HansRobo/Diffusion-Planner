#!/usr/bin/env python3
"""Create a sparse local overlay for the latest converter skip predicates.

The June 2026 corpus was generated before converter PRs #204 and #207.  This
utility reads the existing NPZ tensors and old JSON sidecars, then writes only
newly detected skip decisions to a separate absolute-path mirror.  It never
modifies the shared NPZs, manifests, or sibling sidecars.

The overlay reproduces the *frame predicates* introduced by PR #204
(``DetectedRedLightRun``) and PR #207 (``GreenStop``).  It intentionally does
not claim to reproduce converter-only data changes that require re-reading the
rosbag, such as provenance metadata, goal-pose generation, or neighbor-future
serialization.  Existing ``is_skipped=true`` sidecars are left untouched; the
old decision already excludes those samples from training.

The output root is consumed with ``--skip_filter_sidecar_root``.  Files are
stored below ``<output-root>/mnt/...`` so duplicate frame stems from different
bags cannot collide.  A small metadata file records the exact remote revision,
thresholds, counts, and malformed inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Iterator

import numpy as np

# Make ``diffusion_planner.dimensions`` importable when this file is invoked
# directly from the repository root.
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

import contextlib

from diffusion_planner.dimensions import (  # noqa: E402
    INPUT_T,
    MAX_NUM_NEIGHBORS,
    OUTPUT_T,
    TRAFFIC_LIGHT,
    TRAFFIC_LIGHT_GREEN,
    TRAFFIC_LIGHT_ONE_HOT_DIM,
)

try:  # noqa: E402
    from audit_pr204_red_light_run import (
        _heading_diff_deg as _red_heading_diff_deg,
    )
    from audit_pr204_red_light_run import (
        _intersection_point,
        _red_entries,
        _stop_segments,
        _strict_crossing,
    )
except ImportError:  # pragma: no cover - package-style invocation
    from diffusion_planner.util_scripts.audit_pr204_red_light_run import (  # type: ignore # noqa: E501
        _heading_diff_deg as _red_heading_diff_deg,
    )
    from diffusion_planner.util_scripts.audit_pr204_red_light_run import (
        _intersection_point,
        _red_entries,
        _stop_segments,
        _strict_crossing,
    )


REMOTE_REF = "origin/tier4-main@8290101b018a536ccea3a85e473b310a8de44c51"
GREEN_HEADING_TOL_DEG = 45.0
GREEN_STAY_RADIUS_M = 2.0
GREEN_SPEED_MAX_MPS = 1.0
GREEN_AHEAD_M = 40.0
GREEN_LEAD_FWD_M = 30.0
GREEN_LEAD_LAT_M = 2.0
XY_EPS = 1e-6

RED_LABEL = 11
GREEN_LABEL = 12


def _iter_json_array(path: Path, read_size: int = 1 << 20) -> Iterator[object]:
    """Stream a JSON array without materializing a multi-million path list."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as stream:
        buffer = ""
        eof = False
        index = 0

        def fill() -> bool:
            nonlocal buffer, eof
            if eof:
                return False
            chunk = stream.read(read_size)
            if chunk:
                buffer += chunk
                return True
            eof = True
            return False

        while True:
            while index < len(buffer) and buffer[index].isspace():
                index += 1
            if index >= len(buffer) and not fill():
                raise ValueError(f"{path}: truncated JSON array")
            if buffer[index] == "[":
                index += 1
                break
            raise ValueError(f"{path}: expected '['")

        while True:
            while True:
                while index < len(buffer) and buffer[index].isspace():
                    index += 1
                if index < len(buffer):
                    break
                if not fill():
                    raise ValueError(f"{path}: truncated JSON array")
            if buffer[index] == "]":
                return
            if buffer[index] == ",":
                index += 1
                while index < len(buffer) and buffer[index].isspace():
                    index += 1
                if index >= len(buffer) and not fill():
                    raise ValueError(f"{path}: truncated JSON array")
            try:
                value, end = decoder.raw_decode(buffer, index)
            except json.JSONDecodeError:
                if not fill():
                    raise ValueError(f"{path}: invalid or truncated JSON value") from None
                continue
            index = end
            yield value
            # Retain only the unread tail so a long path list cannot grow memory.
            if index > read_size:
                buffer = buffer[index:]
                index = 0


def _read_sidecar(path: str) -> tuple[dict, bool]:
    sidecar = Path(path).with_suffix(".json")
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {}, False
        return payload, bool(payload.get("is_skipped", False))
    except (OSError, ValueError, UnicodeError):
        return {}, False


def _heading_diff_deg(a_deg: float, b_deg: float) -> float:
    d = abs(a_deg - b_deg) % 360.0
    return min(d, 360.0 - d)


def _red_light_run_arrays(future: np.ndarray, route: np.ndarray, line_strings: np.ndarray) -> bool:
    """Run PR #204 on already-open arrays (avoids a second NPZ decompression)."""
    entries = _red_entries(route)
    if not entries:
        return False
    stops = _stop_segments(line_strings)
    if not stops or future.ndim != 2 or future.shape[0] < OUTPUT_T or future.shape[1] < 3:
        return False
    states: list[tuple[np.ndarray, float]] = []
    for row in future[:OUTPUT_T]:
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
    radius_sq = 12.0 * 12.0
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
                if _red_heading_diff_deg(lane_heading, ego_heading) >= 45.0:
                    continue
                delta = intersection - entry
                if float(np.dot(delta, delta)) <= radius_sq:
                    return True
    return False


def _has_green_route(route: np.ndarray) -> bool:
    if (
        route.ndim != 3
        or route.shape[0] < 1
        or route.shape[2] < TRAFFIC_LIGHT + TRAFFIC_LIGHT_ONE_HOT_DIM
    ):
        return False
    for segment in route[:25]:
        traffic = segment[0, TRAFFIC_LIGHT : TRAFFIC_LIGHT + TRAFFIC_LIGHT_ONE_HOT_DIM]
        argmax = int(np.argmax(traffic))
        if float(traffic[argmax]) > 0.5 and TRAFFIC_LIGHT + argmax == TRAFFIC_LIGHT_GREEN:
            return True
    return False


def _green_stop_arrays(
    future: np.ndarray, current: np.ndarray, route: np.ndarray, neighbors: np.ndarray
) -> bool:
    """Port remote ``detect_green_stop`` for arrays loaded by the caller."""
    if future.ndim != 2 or future.shape[0] < OUTPUT_T or future.shape[1] < 2:
        return False
    if current.size < 6 or route.ndim != 3 or route.shape[0] < 1:
        return False

    exact_zero_rows = int(
        np.count_nonzero((future[:OUTPUT_T, 0] == 0.0) & (future[:OUTPUT_T, 1] == 0.0))
    )
    first_xy = future[0, :2].astype(np.float64, copy=False)
    deltas = future[:OUTPUT_T, :2].astype(np.float64, copy=False) - first_xy
    max_excursion = float(np.sqrt(np.sum(deltas * deltas, axis=1)).max(initial=0.0))
    if exact_zero_rows > 5 or max_excursion >= GREEN_STAY_RADIUS_M:
        return False

    heading = np.asarray(current[2:4], dtype=np.float64)
    heading_norm = float(np.linalg.norm(heading))
    if heading_norm <= XY_EPS:
        return False
    ego_ch, ego_sh = (heading / heading_norm).tolist()
    if float(np.linalg.norm(np.asarray(current[4:6], dtype=np.float64))) >= GREEN_SPEED_MAX_MPS:
        return False

    ego_heading_deg = float(np.degrees(np.arctan2(ego_sh, ego_ch)))
    traffic_end = TRAFFIC_LIGHT + TRAFFIC_LIGHT_ONE_HOT_DIM
    for segment in route[:25]:
        if segment.shape[0] < 2 or segment.shape[1] < traffic_end:
            continue
        traffic = segment[0, TRAFFIC_LIGHT:traffic_end]
        argmax = int(np.argmax(traffic))
        if float(traffic[argmax]) <= 0.5 or TRAFFIC_LIGHT + argmax != TRAFFIC_LIGHT_GREEN:
            continue
        valid = np.flatnonzero(np.any(np.abs(segment[:, :2]) > XY_EPS, axis=1))
        if valid.size < 2:
            continue
        entry = segment[int(valid[0]), :2].astype(np.float64, copy=False)
        end = segment[int(valid[-1]), :2].astype(np.float64, copy=False)
        forward = float(entry[0] * ego_ch + entry[1] * ego_sh)
        if forward <= 0.0 or forward >= GREEN_AHEAD_M:
            continue
        lane_heading = float(np.degrees(np.arctan2(end[1] - entry[1], end[0] - entry[0])))
        if _red_heading_diff_deg(lane_heading, ego_heading_deg) < GREEN_HEADING_TOL_DEG:
            break
    else:
        return False

    if (
        neighbors.ndim != 3
        or neighbors.shape[0] < MAX_NUM_NEIGHBORS
        or neighbors.shape[1] < INPUT_T + 1
    ):
        return False
    for state in neighbors[:MAX_NUM_NEIGHBORS, INPUT_T, :4]:
        if float(np.abs(state).sum()) <= XY_EPS:
            continue
        forward = float(state[0] * ego_ch + state[1] * ego_sh)
        lateral = float(-state[0] * ego_sh + state[1] * ego_ch)
        if 0.5 < forward < GREEN_LEAD_FWD_M and abs(lateral) < GREEN_LEAD_LAT_M:
            return False
    return True


def _process_one(
    path: str, *, only_green: bool = False
) -> tuple[str, int | None, str | None, bool]:
    payload, old_skipped = _read_sidecar(path)
    if old_skipped:
        return path, None, None, False

    try:
        with np.load(path, allow_pickle=False) as archive:
            route = np.asarray(archive["route_lanes"])
            red = False
            if not only_green and _red_entries(route):
                red = _red_light_run_arrays(
                    np.asarray(archive["ego_agent_future"]),
                    route,
                    np.asarray(archive["line_strings"]),
                )
            if red:
                green = False
            elif _has_green_route(route):
                current = np.asarray(archive["ego_current_state"])
                # Avoid decompressing the 80-step future and 320-agent history
                # unless the cheap current-state/route gate can pass.
                heading = np.asarray(current[2:4], dtype=np.float64)
                heading_norm = float(np.linalg.norm(heading))
                speed = float(np.linalg.norm(np.asarray(current[4:6], dtype=np.float64)))
                green = False
                if heading_norm > XY_EPS and speed < GREEN_SPEED_MAX_MPS:
                    green = _green_stop_arrays(
                        np.asarray(archive["ego_agent_future"]),
                        current,
                        route,
                        np.asarray(archive["neighbor_agents_past"]),
                    )
            else:
                green = False
    except (FileNotFoundError, OSError, KeyError, ValueError):
        # Malformed individual NPZs must not abort a multi-million-frame scan.
        red = green = False
    if red:
        return (
            path,
            RED_LABEL,
            (
                "Detected red light run: ego future crosses a stop line segment near the entry point "
                "of a heading-aligned red route lane"
            ),
            False,
        )
    if green:
        return path, GREEN_LABEL, "Stopped at green light with no lead neighbor", False
    return path, None, None, not payload


def _atomic_write(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, separators=(",", ":"), ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)


def _update_one(
    result: tuple[str, int | None, str | None, bool], output_root: Path
) -> tuple[int, int]:
    path, label, details, missing_old_sidecar = result
    if label is None:
        return 0, int(missing_old_sidecar)
    payload, _ = _read_sidecar(path)
    payload["is_skipped"] = True
    payload["skipping_info"] = {
        "details": details,
        "incomplete_data_types": [],
        "label": label,
    }
    payload["converter_overlay"] = {
        "remote_ref": REMOTE_REF,
        "detector": "DetectedRedLightRun" if label == RED_LABEL else "GreenStop",
    }
    _atomic_write(payload, _update_one.output_root / str(path).lstrip("/").replace(".npz", ".json"))
    return 1, int(missing_old_sidecar)


_update_one.output_root = Path(".")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument(
        "--executor",
        choices=("process", "thread"),
        default="process",
        help="parallel backend; process avoids the GIL in the geometry predicates",
    )
    parser.add_argument(
        "--only-green",
        action="store_true",
        help="scan only the PR#207 GreenStop predicate and preserve existing overlay files",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="optional local directory for atomic per-batch resume checkpoints",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.workers < 1 or args.chunk_size < 1:
        raise ValueError("workers and chunk-size must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    _update_one.output_root = args.output_root

    summary: dict[str, object] = {
        "remote_ref": REMOTE_REF,
        "only_green": bool(args.only_green),
        "sources": [],
        "thresholds": {
            "red_light_run_radius_m": 12.0,
            "red_light_run_heading_tol_deg": 45.0,
            "green_stop_heading_tol_deg": GREEN_HEADING_TOL_DEG,
            "green_stop_stay_radius_m": GREEN_STAY_RADIUS_M,
            "green_stop_speed_max_mps": GREEN_SPEED_MAX_MPS,
            "green_stop_ahead_m": GREEN_AHEAD_M,
            "green_stop_lead_fwd_m": GREEN_LEAD_FWD_M,
            "green_stop_lead_lat_m": GREEN_LEAD_LAT_M,
        },
    }
    total = {"scanned": 0, "overlays": 0, "missing_old_sidecar": 0}

    worker = partial(_process_one, only_green=bool(args.only_green))
    executor_type = ProcessPoolExecutor if args.executor == "process" else ThreadPoolExecutor
    with executor_type(max_workers=args.workers) as executor:
        for source in args.source:
            scanned = overlays = missing = 0
            batch: list[str] = []
            checkpoint_root = None
            if args.checkpoint_dir is not None:
                digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:12]
                checkpoint_root = args.checkpoint_dir / f"{source.stem}-{digest}"
                checkpoint_root.mkdir(parents=True, exist_ok=True)

            def flush() -> None:
                nonlocal scanned, overlays, missing, batch
                if not batch:
                    return
                chunk_start = scanned
                checkpoint = (
                    checkpoint_root / f"{chunk_start:012d}_{len(batch):06d}.json"
                    if checkpoint_root is not None
                    else None
                )
                if checkpoint is not None and checkpoint.is_file():
                    prior = json.loads(checkpoint.read_text(encoding="utf-8"))
                    overlays += int(prior.get("overlays", 0))
                    missing += int(prior.get("missing_old_sidecar", 0))
                else:
                    batch_overlays = batch_missing = 0
                    for result in executor.map(worker, batch, chunksize=32):
                        added, missing_one = _update_one(result, args.output_root)
                        batch_overlays += added
                        batch_missing += missing_one
                    overlays += batch_overlays
                    missing += batch_missing
                    if checkpoint is not None:
                        _atomic_write(
                            {
                                "source": str(source),
                                "start": chunk_start,
                                "count": len(batch),
                                "overlays": batch_overlays,
                                "missing_old_sidecar": batch_missing,
                            },
                            checkpoint,
                        )
                scanned += len(batch)
                print(
                    f"[sidecar-overlay] {source.name}: scanned={scanned} overlays={overlays}",
                    flush=True,
                )
                batch = []

            for value in _iter_json_array(source):
                if not isinstance(value, str):
                    raise ValueError(f"{source}: expected string NPZ paths")
                batch.append(value)
                if len(batch) >= args.chunk_size:
                    flush()
                if args.max_items is not None and scanned + len(batch) >= args.max_items:
                    break
            flush()
            source_result = {
                "source": str(source),
                "scanned": scanned,
                "overlay_count": overlays,
                "missing_old_sidecar": missing,
            }
            if checkpoint_root is not None:
                source_result["checkpoint_dir"] = str(checkpoint_root)
            cast_sources = summary["sources"]
            assert isinstance(cast_sources, list)
            cast_sources.append(source_result)
            total["scanned"] += scanned
            total["overlays"] += overlays
            total["missing_old_sidecar"] += missing

    # In green-only append mode, include files from the earlier red pass and
    # report the actual on-disk overlay count instead of only this pass's count.
    overlay_files = [p for p in args.output_root.rglob("*.json") if p.name != "metadata.json"]
    total["overlay_files_on_disk"] = len(overlay_files)
    summary["totals"] = total
    _atomic_write(summary, args.output_root / "metadata.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
