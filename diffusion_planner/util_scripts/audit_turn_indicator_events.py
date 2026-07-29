#!/usr/bin/env python3
"""Audit turn-indicator states and reconstruct sequence-level signal events.

The tool is read-only.  It reads ``turn_indicators.npy`` directly from each NPZ
and opens the larger trajectory arrays only for samples close to a state change.
Results can therefore be produced from a multi-million-frame manifest without
copying or rewriting the source dataset.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import multiprocessing as mp
import re
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

VALID_REPORT_STATES = (1, 2, 3)
STATE_NAMES = {1: "disable", 2: "left", 3: "right"}
_SEGMENTED_FRAME_RE = re.compile(r"^(?P<prefix>.*)_(?P<segment>\d{8})_(?P<frame>\d{8})\.npz$")
_FLAT_FRAME_RE = re.compile(r"^(?P<prefix>.*)_(?P<segment>\d{8})(?P<frame>\d{8})\.npz$")
_PATHS: list[str] = []


def _read_npy(npz: zipfile.ZipFile, name: str) -> np.ndarray:
    with npz.open(f"{name}.npy") as stream:
        return np.lib.format.read_array(stream, allow_pickle=False)


def _suffix_run(values: np.ndarray) -> int:
    current = values[-1]
    changed = np.flatnonzero(values != current)
    return int(values.size if changed.size == 0 else values.size - 1 - changed[-1])


def _first_time(mask: np.ndarray, hz: float = 10.0) -> float | None:
    indices = np.flatnonzero(mask)
    return None if indices.size == 0 else float((indices[0] + 1) / hz)


def _trajectory_features(npz: zipfile.ZipFile, before: int, current: int) -> dict[str, Any]:
    future = np.asarray(_read_npy(npz, "ego_agent_future"), dtype=np.float64)
    ego_state = np.asarray(_read_npy(npz, "ego_current_state"), dtype=np.float64)
    if future.shape != (80, 3):
        raise ValueError(f"ego_agent_future must have shape (80, 3), got {future.shape}")

    yaw = np.unwrap(future[:, 2])
    yaw = yaw - yaw[0]
    xy = future[:, :2]
    direction_state = current if current in (2, 3) else before
    direction_sign = 1.0 if direction_state == 2 else -1.0 if direction_state == 3 else 0.0
    expected_yaw = direction_sign * yaw
    expected_lateral = direction_sign * xy[:, 1]

    result: dict[str, Any] = {
        "speed_mps": float(np.hypot(ego_state[4], ego_state[5])),
        "end_distance_m": float(np.linalg.norm(xy[-1])),
        "end_yaw_deg": float(np.rad2deg(yaw[-1])),
        "end_lateral_m": float(xy[-1, 1]),
        "max_expected_yaw_deg": float(np.rad2deg(np.max(expected_yaw))),
        "max_expected_lateral_m": float(np.max(expected_lateral)),
    }
    for degrees in (2, 5, 10, 20):
        result[f"time_to_expected_yaw_{degrees}deg_s"] = _first_time(
            expected_yaw >= np.deg2rad(degrees)
        )
    for distance in (0.3, 0.5, 1.0, 2.0):
        key = str(distance).replace(".", "p")
        result[f"time_to_expected_lateral_{key}m_s"] = _first_time(expected_lateral >= distance)
    return result


def _scan_transition_geometry(event: dict[str, Any]) -> dict[str, Any]:
    result = dict(event)
    try:
        with zipfile.ZipFile(result["path"]) as npz:
            result.update(_trajectory_features(npz, int(result["before"]), int(result["current"])))
    except Exception as error:
        result["geometry_error"] = repr(error)
    return result


def _scan_range(bounds: tuple[int, int]) -> dict[str, Any]:
    start, end = bounds
    size = end - start
    current = np.zeros(size, dtype=np.int8)
    before_run = np.zeros(size, dtype=np.int8)
    suffix = np.zeros(size, dtype=np.int8)
    shapes: Counter[int] = Counter()
    invalid: Counter[int] = Counter()
    errors: list[dict[str, str]] = []

    for offset, path in enumerate(_PATHS[start:end]):
        try:
            with zipfile.ZipFile(path) as npz:
                values = np.asarray(_read_npy(npz, "turn_indicators")).reshape(-1)
                shapes[int(values.size)] += 1
                if values.size == 0:
                    raise ValueError("turn_indicators is empty")
                values = values.astype(np.int64, copy=False)
                for value, count in zip(*np.unique(values, return_counts=True), strict=False):
                    if int(value) not in VALID_REPORT_STATES:
                        invalid[int(value)] += int(count)

                run = _suffix_run(values)
                state = int(values[-1])
                previous = state if run == values.size else int(values[-run - 1])
                current[offset] = state
                before_run[offset] = previous
                suffix[offset] = min(run, np.iinfo(np.int8).max)

        except Exception as error:  # keep the audit complete and report every bad source
            current[offset] = -1
            if len(errors) < 100:
                errors.append({"path": path, "error": repr(error)})

    return {
        "start": start,
        "current": current,
        "before_run": before_run,
        "suffix": suffix,
        "shapes": dict(shapes),
        "invalid": dict(invalid),
        "errors": errors,
    }


def _parse_frame(path: str) -> tuple[str, int] | None:
    source = Path(path)
    segmented = _SEGMENTED_FRAME_RE.match(source.name)
    if segmented is not None:
        key = str(source.parent / f"{segmented.group('prefix')}_{segmented.group('segment')}")
        return key, int(segmented.group("frame"))
    flat = _FLAT_FRAME_RE.match(source.name)
    if flat is not None:
        key = str(source.parent / f"{flat.group('prefix')}_{flat.group('segment')}")
        return key, int(flat.group("frame"))
    return None


def _quantiles(values: list[float | None]) -> dict[str, float] | None:
    total = len(values)
    finite = np.asarray([value for value in values if value is not None], dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"finite_count": 0, "total_count": total}
    points = np.quantile(finite, (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0))
    result = {"finite_count": int(finite.size), "total_count": total}
    result.update(
        {
            key: float(value)
            for key, value in zip(
                ("min", "p10", "p25", "p50", "p75", "p90", "max"), points, strict=False
            )
        }
    )
    return result


def _counter_quantiles(values: Counter[int]) -> dict[str, float] | None:
    total = sum(values.values())
    if total == 0:
        return None
    ordered = sorted(values.items())
    result: dict[str, float] = {}
    for name, quantile in zip(
        ("min", "p10", "p25", "p50", "p75", "p90", "max"),
        (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
        strict=False,
    ):
        target = quantile * (total - 1)
        cumulative = 0
        selected = ordered[-1][0]
        for value, count in ordered:
            cumulative += count
            if cumulative > target:
                selected = value
                break
        result[name] = float(selected)
    return result


def _summarize_geometry(events: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        before = int(event["before"])
        current = int(event["current"])
        name = f"{STATE_NAMES.get(before, str(before))}_to_{STATE_NAMES.get(current, str(current))}"
        grouped[name].append(event)

    output: dict[str, Any] = {}
    for name, rows in sorted(grouped.items()):
        numeric_keys = sorted(
            key for key in rows[0] if key.startswith(("speed_", "end_", "max_", "time_to_"))
        )
        output[name] = {
            "count": len(rows),
            "features": {key: _quantiles([row.get(key) for row in rows]) for key in numeric_keys},
        }
    return output


def _reconstruct_episodes(
    paths: list[str], states: np.ndarray, suffix: np.ndarray, before_run: np.ndarray
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    active: dict[str, dict[str, Any]] = {}
    last_frame: dict[str, int] = {}
    episodes: list[dict[str, Any]] = []
    transitions: Counter[tuple[int, int]] = Counter()
    transition_events: list[dict[str, Any]] = []
    frame_steps: Counter[int] = Counter()
    malformed = 0
    non_monotonic = 0
    uncertain_transitions = 0

    for index, (path, state_value) in enumerate(zip(paths, states, strict=False)):
        parsed = _parse_frame(path)
        state = int(state_value)
        if parsed is None:
            malformed += 1
            continue
        if state not in VALID_REPORT_STATES:
            continue
        key, frame = parsed
        previous_frame = last_frame.get(key)
        if previous_frame is not None:
            step = frame - previous_frame
            if step <= 0:
                non_monotonic += 1
                stale = active.pop(key, None)
                if stale is not None:
                    episodes.append(stale)
                previous_frame = None
            else:
                frame_steps[min(step, 1000)] += 1
        last_frame[key] = frame

        episode = active.get(key)
        if episode is None:
            run = int(suffix[index])
            start_frame = frame - run + 1 if run < 31 else None
            active[key] = {
                "sequence": key,
                "state": state,
                "start_frame": start_frame,
                "first_observed_frame": frame,
                "last_observed_frame": frame,
                "left_censored": start_frame is None,
                "right_censored": True,
                "gap_before_transition": False,
            }
            continue

        if state == int(episode["state"]):
            episode["last_observed_frame"] = frame
            continue

        previous_state = int(episode["state"])
        transitions[(previous_state, state)] += 1
        run = int(suffix[index])
        exact_transition = run < 31 and int(before_run[index]) == previous_state
        inferred_start = frame - run + 1 if exact_transition else None
        transition_min = None if previous_frame is None else previous_frame + 1
        transition_max = inferred_start if exact_transition else frame - 30 if run == 31 else None
        uncertain_transition = inferred_start is None
        manifest_gap_frames = None if previous_frame is None else max(frame - previous_frame - 1, 0)
        uncertain_transitions += int(uncertain_transition)
        if inferred_start is not None:
            episode["end_frame"] = inferred_start
            episode["right_censored"] = False
            if episode["start_frame"] is not None:
                episode["duration_s"] = (inferred_start - int(episode["start_frame"])) / 10.0
        else:
            episode["transition_frame_min"] = transition_min
            episode["transition_frame_max"] = transition_max
        episodes.append(episode)
        transition_events.append(
            {
                "manifest_index": index,
                "path": path,
                "sequence": key,
                "before": previous_state,
                "current": state,
                "observed_frame": frame,
                "history_suffix_frames": run,
                "transition_frame": inferred_start,
                "transition_frame_min": transition_min,
                "transition_frame_max": transition_max,
                "exact_transition_frame": exact_transition,
                "transition_time_uncertain": uncertain_transition,
                "manifest_gap_frames": manifest_gap_frames,
            }
        )
        active[key] = {
            "sequence": key,
            "state": state,
            "start_frame": inferred_start,
            "first_observed_frame": frame,
            "last_observed_frame": frame,
            "left_censored": inferred_start is None,
            "right_censored": True,
            "transition_time_uncertain": uncertain_transition,
            "previous_state_from_history": int(before_run[index]),
        }

    episodes.extend(active.values())
    duration_summary: dict[str, Any] = {}
    for state in VALID_REPORT_STATES:
        rows = [episode for episode in episodes if int(episode["state"]) == state]
        complete = [episode.get("duration_s") for episode in rows if "duration_s" in episode]
        duration_summary[STATE_NAMES[state]] = {
            "episodes": len(rows),
            "complete_episodes": len(complete),
            "duration_s": _quantiles(complete),
        }

    summary = {
        "sequence_count": len(last_frame),
        "episode_count": len(episodes),
        "malformed_filenames": malformed,
        "non_monotonic_frames": non_monotonic,
        "uncertain_transition_times": uncertain_transitions,
        "frame_step_count": sum(frame_steps.values()),
        "frame_step_frames": _counter_quantiles(frame_steps),
        "frame_step_most_common": {str(step): count for step, count in frame_steps.most_common(20)},
        "transitions": {
            f"{STATE_NAMES.get(a, str(a))}_to_{STATE_NAMES.get(b, str(b))}": count
            for (a, b), count in sorted(transitions.items())
        },
        "durations": duration_summary,
    }
    return summary, episodes, transition_events


def _json_open(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--events-output", type=Path)
    parser.add_argument(
        "--state-cache-output",
        type=Path,
        help="Optional compressed current/before/suffix arrays aligned to the manifest.",
    )
    parser.add_argument("--workers", type=int, default=min(64, mp.cpu_count()))
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    with args.manifest.open("r", encoding="utf-8") as stream:
        paths = json.load(stream)
    if args.max_samples > 0:
        paths = paths[: args.max_samples]
    if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
        raise TypeError("manifest must be a JSON list of NPZ paths")

    global _PATHS
    _PATHS = paths
    states = np.empty(len(paths), dtype=np.int8)
    before_run = np.empty(len(paths), dtype=np.int8)
    suffix = np.empty(len(paths), dtype=np.int8)
    shapes: Counter[int] = Counter()
    invalid: Counter[int] = Counter()
    errors: list[dict[str, str]] = []
    bounds = [
        (start, min(start + args.batch_size, len(paths)))
        for start in range(0, len(paths), args.batch_size)
    ]

    started = time.monotonic()
    context = mp.get_context("fork")
    with context.Pool(args.workers) as pool:
        for result in pool.imap_unordered(_scan_range, bounds, chunksize=1):
            start = int(result["start"])
            end = start + len(result["current"])
            states[start:end] = result["current"]
            before_run[start:end] = result["before_run"]
            suffix[start:end] = result["suffix"]
            shapes.update({int(key): int(value) for key, value in result["shapes"].items()})
            invalid.update({int(key): int(value) for key, value in result["invalid"].items()})
            errors.extend(result["errors"])

    if args.state_cache_output is not None:
        args.state_cache_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.state_cache_output,
            current=states,
            before_run=before_run,
            suffix=suffix,
            sample_count=np.asarray([len(paths)], dtype=np.int64),
        )

    sequence_summary, episodes, transition_events = _reconstruct_episodes(
        paths, states, suffix, before_run
    )
    with context.Pool(args.workers) as pool:
        geometry = list(
            pool.imap_unordered(_scan_transition_geometry, transition_events, chunksize=32)
        )
    class_counts = Counter(int(value) for value in states if int(value) >= 0)
    elapsed = time.monotonic() - started
    summary = {
        "manifest": str(args.manifest),
        "sample_count": len(paths),
        "elapsed_seconds": elapsed,
        "samples_per_second": len(paths) / elapsed if elapsed > 0 else math.inf,
        "valid_report_states": list(VALID_REPORT_STATES),
        "current_class_counts": {
            STATE_NAMES.get(state, str(state)): count
            for state, count in sorted(class_counts.items())
        },
        "history_shapes": {str(key): value for key, value in sorted(shapes.items())},
        "invalid_history_value_counts": {str(key): value for key, value in sorted(invalid.items())},
        "error_count": int(np.count_nonzero(states < 0)),
        "error_examples": errors[:100],
        "sequence": sequence_summary,
        "near_transition_geometry": _summarize_geometry(geometry),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")

    if args.events_output is not None:
        args.events_output.parent.mkdir(parents=True, exist_ok=True)
        geometry_by_index = {int(row["manifest_index"]): row for row in geometry}
        with _json_open(args.events_output, "w") as stream:
            for episode in episodes:
                stream.write(json.dumps({"type": "episode", **episode}, sort_keys=True) + "\n")
            for row in sorted(
                geometry_by_index.values(), key=lambda item: int(item["manifest_index"])
            ):
                stream.write(
                    json.dumps({"type": "transition_geometry", **row}, sort_keys=True) + "\n"
                )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
