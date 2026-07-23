#!/usr/bin/env python3
"""Create immutable, causal-clean SFT manifests.

The converter's future can cross a traffic-light transition that is not present
in the current frame.  This pre-pass removes only samples where the nearest
forward signal controlling the ego route is red, the ego is stopped, and the
logged expert later starts moving.  A nearby green signal, a downstream red
signal, and full-horizon red-light holds remain in the manifest.

For the base list, all paths in the three full right-turn lists are removed first
so the oversampled right-turn data cannot also appear in the base distribution.
The input JSON files and NPZ files are never modified.  Outputs and metadata are
written atomically to the requested directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from diffusion_planner.dimensions import (
    TRAFFIC_LIGHT_GREEN,
    TRAFFIC_LIGHT_RED,
    TRAFFIC_LIGHT_YELLOW,
)


STOP_SPEED_MPS = 0.3
HISTORY_FRAMES = 5
HISTORY_SPEED_MPS = 0.5
LANE_BEHIND_M = 1.0
MAX_RED_DISTANCE_M = 100.0
MOTION_SPEED_MPS = 0.5
MOTION_DISTANCE_M = 0.25
MOTION_SUSTAIN_FRAMES = 3
FUTURE_DT_S = 0.1
SIGNAL_MATCH_M = 5.0
TRANSITION_WINDOW_FRAMES = 5
# A stop line whose nearest route point is behind the ego cannot control a
# forward stop.  Route points are ordered along the lanelet, so this remains
# valid through turns where the world-frame x coordinate may be negative.
FORWARD_ROUTE_POINT_INDEX_TOLERANCE = 1
SIGNAL_AMBIGUITY_M = 5.0
_FRAME_SUFFIX = re.compile(r"^(.*_)(\d{8})(\.npz)$")


def _sibling_frame_path(path: str, frame_offset: int) -> str | None:
    """Return a neighboring 10-Hz frame in the same recorded sequence."""
    match = _FRAME_SUFFIX.match(Path(path).name)
    if match is None:
        return None
    frame = int(match.group(2)) + frame_offset
    if frame < 0 or frame > 99_999_999:
        return None
    return str(Path(path).with_name(f"{match.group(1)}{frame:08d}{match.group(3)}"))


def _transform_points_to_future_frame(
    points: np.ndarray, future_pose: np.ndarray
) -> np.ndarray:
    """Transform current-frame points into the ego frame at a future pose."""
    dx, dy = future_pose[:2]
    if future_pose.shape[0] >= 4:
        heading = float(np.arctan2(future_pose[3], future_pose[2]))
    elif future_pose.shape[0] >= 3:
        heading = float(future_pose[2])
    else:
        heading = 0.0
    c, s = np.cos(heading), np.sin(heading)
    relative = points - np.asarray([dx, dy], dtype=points.dtype)
    return np.stack(
        (c * relative[:, 0] + s * relative[:, 1],
         -s * relative[:, 0] + c * relative[:, 1]),
        axis=-1,
    )


def _origin_to_segment_distance(start: np.ndarray, end: np.ndarray) -> float:
    """Return the distance from the ego origin to one stop-line segment."""
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 1e-12:
        return float(np.linalg.norm(start))
    t = float(np.dot(-start, segment) / length_sq)
    t = min(1.0, max(0.0, t))
    return float(np.linalg.norm(start + t * segment))


def _nearest_control_signal(
    route: np.ndarray, line_strings: np.ndarray
) -> tuple[str, float] | None:
    """Find the nearest unambiguous signal/stop-line pair on the ego route.

    ``route_lanes`` is a forward sequence and can contain several signalized
    lanelets.  Looking for any red point is incorrect: a downstream red can
    coexist with a nearer green signal and would delete valid green-start
    samples.  We associate stop-line segments with each signalized route
    lanelet, require the matching route point to be forward of the ego, then
    select the nearest pair.  Conflicting colors at the same distance are
    unresolved and deliberately retained by returning ``None``.
    """
    if route.ndim != 3 or route.shape[-1] <= TRAFFIC_LIGHT_RED:
        return None
    if line_strings.ndim != 3 or line_strings.shape[-1] < 3:
        return None

    route_geometry = route[..., :4]
    route_valid = np.any(np.abs(route_geometry) > 1e-6, axis=-1)
    has_signal = any(
        np.any((route[..., channel] > 0.5) & route_valid)
        for channel in (TRAFFIC_LIGHT_RED, TRAFFIC_LIGHT_GREEN, TRAFFIC_LIGHT_YELLOW)
    )
    if not has_signal:
        return None
    stop_segments: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for line in line_strings:
        line_valid = (line[:, 2] > 0.5) & np.any(np.abs(line[:, :2]) > 1e-6, axis=-1)
        for point_idx in range(len(line) - 1):
            if not (line_valid[point_idx] and line_valid[point_idx + 1]):
                continue
            start = np.asarray(line[point_idx, :2], dtype=np.float64)
            end = np.asarray(line[point_idx + 1, :2], dtype=np.float64)
            direction = end - start
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm <= 1e-6:
                continue
            stop_segments.append((start, end, direction / direction_norm))
    if not stop_segments:
        return None

    signal_states = (
        (TRAFFIC_LIGHT_RED, "red"),
        (TRAFFIC_LIGHT_GREEN, "green"),
        (TRAFFIC_LIGHT_YELLOW, "yellow"),
    )
    candidates: list[tuple[float, str, int]] = []
    for lane_idx in range(route.shape[0]):
        valid = route_valid[lane_idx]
        if not np.any(valid):
            continue
        states = [
            name
            for channel, name in signal_states
            if np.any((route[lane_idx, :, channel] > 0.5) & valid)
        ]
        # Mixed one-hot states within one lanelet are not safe to interpret.
        if len(states) != 1:
            continue
        state_name = states[0]
        points = np.asarray(route[lane_idx, valid, :2], dtype=np.float64)
        directions = np.asarray(route[lane_idx, valid, 2:4], dtype=np.float64)
        current_point = int(np.argmin(np.linalg.norm(points, axis=-1)))
        for start, end, stop_direction in stop_segments:
            midpoint = 0.5 * (start + end)
            point_distance = np.linalg.norm(points - midpoint[None, :], axis=-1)
            nearest_point = int(np.argmin(point_distance))
            if float(point_distance[nearest_point]) > SIGNAL_MATCH_M:
                continue
            # Route points are ordered along the lanelet. Reject stop lines
            # whose matching point is already behind the ego on that lanelet.
            if nearest_point + FORWARD_ROUTE_POINT_INDEX_TOLERANCE < current_point:
                continue
            route_direction = directions[nearest_point]
            route_norm = float(np.linalg.norm(route_direction))
            if route_norm > 1e-6:
                cosine = abs(float(np.dot(route_direction, stop_direction))) / route_norm
                # A stop line should be approximately perpendicular to travel.
                if cosine > 0.5:
                    continue
            candidates.append(
                (
                    _origin_to_segment_distance(start, end),
                    state_name,
                    lane_idx,
                )
            )
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    nearest_distance = candidates[0][0]
    nearby_colors = {
        state
        for distance, state, _ in candidates
        if distance <= nearest_distance + SIGNAL_AMBIGUITY_M
    }
    if len(nearby_colors) != 1:
        return None
    return candidates[0][1], nearest_distance


def _is_causal_red_light_release(path: str) -> bool:
    """Return whether one NPZ should be removed from an SFT manifest."""
    with np.load(path, allow_pickle=False) as archive:
        route = np.asarray(archive["route_lanes"])
        state = np.asarray(archive["ego_current_state"])
        if route.ndim != 3 or route.shape[-1] <= TRAFFIC_LIGHT_RED:
            return False
        route_valid = np.any(np.abs(route[..., :4]) > 1e-6, axis=-1)
        has_signal = any(
            np.any((route[..., channel] > 0.5) & route_valid)
            for channel in (TRAFFIC_LIGHT_RED, TRAFFIC_LIGHT_GREEN, TRAFFIC_LIGHT_YELLOW)
        )
        if not has_signal:
            return False
        line_strings = np.asarray(archive["line_strings"])
        control_signal = _nearest_control_signal(route, line_strings)
        if control_signal is None or control_signal[0] != "red":
            return False
        if state.ndim != 1 or state.shape[0] < 6:
            return False
        if float(np.linalg.norm(state[4:6])) > STOP_SPEED_MPS:
            return False

        # Past/future tensors are much larger than route/state.  They are only
        # decompressed after the current controlling signal has been confirmed.
        past = np.asarray(archive["ego_agent_past"])
        future = np.asarray(archive["ego_agent_future"])
    if future.ndim != 2 or future.shape[0] < 1 or future.shape[1] < 2:
        return False

    # A short stationary history guards against a transient low speed while the
    # car is actually moving. Missing/zero-padded history is neutral here; the
    # current velocity gate above remains mandatory.
    if past.ndim == 2 and past.shape[1] >= 2 and past.shape[0] >= 2:
        history_frames = min(HISTORY_FRAMES, past.shape[0] - 1)
        history_xy = past[-(history_frames + 1) :, :2]
        segment_distance = np.linalg.norm(np.diff(history_xy, axis=0), axis=-1)
        segment_valid = np.any(np.abs(history_xy[1:]) > 1e-6, axis=-1) | np.any(
            np.abs(history_xy[:-1]) > 1e-6, axis=-1
        )
        threshold = HISTORY_SPEED_MPS * FUTURE_DT_S
        if np.any(segment_distance[segment_valid] > threshold):
            return False

    future_xy = future[:, :2]
    previous_xy = np.concatenate((np.zeros((1, 2), dtype=future_xy.dtype), future_xy[:-1]))
    step_distance = np.linalg.norm(future_xy - previous_xy, axis=-1)
    moving = step_distance / FUTURE_DT_S >= MOTION_SPEED_MPS
    sustain = min(MOTION_SUSTAIN_FRAMES, future.shape[0])
    if sustain <= 0:
        return False
    moving_windows = np.lib.stride_tricks.sliding_window_view(moving, sustain).all(axis=-1)
    distance_windows = (
        np.lib.stride_tricks.sliding_window_view(step_distance, sustain).sum(axis=-1)
        >= MOTION_DISTANCE_M
    )
    release_windows = moving_windows & distance_windows
    if not np.any(release_windows):
        return False
    # No future signal transition is required. The target is defined by the
    # current controlling red signal plus a non-stationary logged future.
    return True


def _load_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as stream:
        values = json.load(stream)
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise ValueError(f"{path} must contain a flat JSON list of NPZ paths")
    return values


def _filter_paths(
    paths: list[str],
    workers: int,
    label: str,
    checkpoint_dir: Path | None = None,
    chunk_size: int = 100_000,
    prior_checkpoint_dir: Path | None = None,
) -> tuple[list[str], int]:
    """Filter paths with bounded concurrency and optional resumable chunks."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    removed = 0
    kept: list[str] = []
    checkpoint_root = None
    if checkpoint_dir is not None:
        checkpoint_root = checkpoint_dir / label
        checkpoint_root.mkdir(parents=True, exist_ok=True)
    prior_root = prior_checkpoint_dir / label if prior_checkpoint_dir is not None else None
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        # Do not submit millions of futures at once. Each source-order chunk is
        # atomically checkpointed, so an interrupted scan can resume without
        # reopening completed NPZ files.
        batch_size = max(256, workers * 32)
        for chunk_start in range(0, len(paths), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(paths))
            chunk = paths[chunk_start:chunk_end]
            checkpoint = (
                checkpoint_root / f"{chunk_start:012d}_{chunk_end:012d}.json"
                if checkpoint_root is not None
                else None
            )
            if checkpoint is not None and checkpoint.exists():
                with checkpoint.open("r", encoding="utf-8") as stream:
                    payload = json.load(stream)
                if (
                    not isinstance(payload, dict)
                    or payload.get("start") != chunk_start
                    or payload.get("end") != chunk_end
                    or not isinstance(payload.get("kept"), list)
                    or any(not isinstance(value, str) for value in payload["kept"])
                ):
                    raise ValueError(f"invalid checkpoint: {checkpoint}")
                chunk_kept = payload["kept"]
            else:
                prior_kept: list[str] | None = None
                prior_checkpoint = (
                    prior_root / checkpoint.name
                    if prior_root is not None and checkpoint is not None
                    else None
                )
                if prior_checkpoint is not None and prior_checkpoint.exists():
                    with prior_checkpoint.open("r", encoding="utf-8") as stream:
                        prior_payload = json.load(stream)
                    prior_kept = (
                        prior_payload.get("kept")
                        if isinstance(prior_payload, dict)
                        else None
                    )
                    if (
                        not isinstance(prior_payload, dict)
                        or prior_payload.get("start") != chunk_start
                        or prior_payload.get("end") != chunk_end
                        or not isinstance(prior_kept, list)
                        or any(not isinstance(value, str) for value in prior_kept)
                        or not set(prior_kept).issubset(set(chunk))
                    ):
                        raise ValueError(f"invalid prior checkpoint: {prior_checkpoint}")
                candidate_paths = chunk
                if prior_kept is not None:
                    prior_kept_set = set(prior_kept)
                    # The refined detector is a strict subset of the old one;
                    # only old removals can change classification.
                    candidate_paths = [path for path in chunk if path not in prior_kept_set]
                chunk_kept = []
                candidate_kept: set[str] = set()
                for batch_start in range(0, len(candidate_paths), batch_size):
                    batch = candidate_paths[batch_start : batch_start + batch_size]
                    for path, remove in zip(
                        batch,
                        executor.map(_is_causal_red_light_release, batch),
                    ):
                        if not remove:
                            candidate_kept.add(path)
                if prior_kept is not None:
                    chunk_kept = [
                        path
                        for path in chunk
                        if path in prior_kept_set or path in candidate_kept
                    ]
                else:
                    chunk_kept = [path for path in chunk if path in candidate_kept]
                if checkpoint is not None:
                    _atomic_json_dump(
                        {"start": chunk_start, "end": chunk_end, "kept": chunk_kept},
                        checkpoint,
                    )
            chunk_removed = len(chunk) - len(chunk_kept)
            removed += chunk_removed
            kept.extend(chunk_kept)
            print(
                f"{label}: scanned {chunk_end}/{len(paths)}; removed={removed}; "
                f"threads={max(1, workers)}",
                flush=True,
            )
    return kept, removed


def _atomic_json_dump(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-list", type=Path, required=True)
    parser.add_argument("--right-turn-list", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Optional directory for resumable source-order chunk checkpoints",
    )
    parser.add_argument(
        "--prior-checkpoint-dir",
        type=Path,
        default=None,
        help="Reuse completed chunks from an older, less selective detector",
    )
    parser.add_argument("--chunk-size", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    right_lists = [(path, _load_list(path)) for path in args.right_turn_list]
    right_union = set()
    for path, values in right_lists:
        right_union.update(values)

    base_values = _load_list(args.base_list)
    base_overlap = [value for value in base_values if value in right_union]
    base_non_overlap = [value for value in base_values if value not in right_union]
    print(
        f"base source={len(base_values)} right_union={len(right_union)} "
        f"overlap_removed_before_causal={len(base_overlap)}",
        flush=True,
    )
    base_kept, base_causal_removed = _filter_paths(
        base_non_overlap,
        args.workers,
        "base",
        args.checkpoint_dir,
        args.chunk_size,
        args.prior_checkpoint_dir,
    )

    metadata = {
        "detector": "causal_red_light_controlled_signal_v3_nearest_stop_line",
        "thresholds": {
            "stop_speed_mps": STOP_SPEED_MPS,
            "history_frames": HISTORY_FRAMES,
            "history_speed_mps": HISTORY_SPEED_MPS,
            "lane_behind_m": LANE_BEHIND_M,
            "max_red_distance_m": MAX_RED_DISTANCE_M,
            "motion_speed_mps": MOTION_SPEED_MPS,
            "motion_distance_m": MOTION_DISTANCE_M,
            "motion_sustain_frames": MOTION_SUSTAIN_FRAMES,
            "future_dt_s": FUTURE_DT_S,
            "signal_match_m": SIGNAL_MATCH_M,
            "forward_route_point_index_tolerance": FORWARD_ROUTE_POINT_INDEX_TOLERANCE,
            "signal_ambiguity_m": SIGNAL_AMBIGUITY_M,
        },
        "base": {
            "source_list": str(args.base_list),
            "source_count": len(base_values),
            "source_unique_count": len(set(base_values)),
            "right_turn_union_count": len(right_union),
            "overlap_removed_before_causal": len(base_overlap),
            "causal_removed": base_causal_removed,
            "output_count": len(base_kept),
        },
        "right_turn_lists": [],
    }
    base_output = args.output_dir / "path_list_train_base_causal_red_light_filtered.json"
    _atomic_json_dump(base_kept, base_output)

    for source_path, values in right_lists:
        kept, removed = _filter_paths(
            values,
            args.workers,
            source_path.stem,
            args.checkpoint_dir,
            args.chunk_size,
        )
        output = args.output_dir / f"{source_path.stem}_causal_red_light_filtered.json"
        _atomic_json_dump(kept, output)
        metadata["right_turn_lists"].append(
            {
                "source_list": str(source_path),
                "source_count": len(values),
                "causal_removed": removed,
                "output_count": len(kept),
                "output_list": str(output),
            }
        )

    metadata["base"]["output_list"] = str(base_output)
    metadata["generated_sha256"] = {
        "base": _sha256(base_output),
        "right_turn_lists": [
            _sha256(Path(entry["output_list"]))
            for entry in metadata["right_turn_lists"]
        ],
    }
    _atomic_json_dump(metadata, args.output_dir / "metadata.json")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
