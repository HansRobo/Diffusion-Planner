#!/usr/bin/env python3
"""Audit candidates removed by the original causal red-light detector.

This is deliberately an audit, not a filter.  It joins the current-frame
route signal, the route-associated stop line, the logged expert future, and
same-sequence neighboring frames.  A sample is never called a violation just
because a red route attribute remains; the report distinguishes a genuine
stop-line crossing from a stopped/queued vehicle and from missing evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from diffusion_planner.dimensions import TRAFFIC_LIGHT_GREEN, TRAFFIC_LIGHT_RED
# ``util_scripts`` is intentionally outside the installable model package;
# import the filter helpers by sibling path so this tool works with the same
# ``PYTHONPATH=diffusion_planner`` invocation as the filter itself.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from filter_causal_red_light_lists import (  # noqa: E402
    HISTORY_FRAMES,
    HISTORY_SPEED_MPS,
    LANE_BEHIND_M,
    MAX_RED_DISTANCE_M,
    MOTION_DISTANCE_M,
    MOTION_SPEED_MPS,
    MOTION_SUSTAIN_FRAMES,
    SIGNAL_MATCH_M,
    STOP_SPEED_MPS,
    FUTURE_DT_S,
    _sibling_frame_path,
    _transform_points_to_future_frame,
)


def _valid_xy(points: np.ndarray) -> np.ndarray:
    return np.any(np.abs(points) > 1e-6, axis=-1)


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _segments_cross(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    """Closed 2-D segment intersection with a small collinearity tolerance."""
    eps = 1e-5
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and (
        (o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)
    ):
        return True

    def on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
        return (
            min(p[0], r[0]) - eps <= q[0] <= max(p[0], r[0]) + eps
            and min(p[1], r[1]) - eps <= q[1] <= max(p[1], r[1]) + eps
            and abs(_orientation(p, q, r)) <= eps
        )

    return (abs(o1) <= eps and on_segment(a, c, b)) or (
        abs(o2) <= eps and on_segment(a, d, b)
    ) or (abs(o3) <= eps and on_segment(c, a, d)) or (
        abs(o4) <= eps and on_segment(c, b, d)
    )


def _stop_line_segments(
    line_strings: np.ndarray,
    red_cluster: np.ndarray,
    red_directions: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return only stop-line segments geometrically associated with the red route.

    The raw tensor contains stop lines for every nearby approach.  Counting a
    crossing of any one of them would manufacture red-light violations on a
    parallel or perpendicular approach.  We require proximity to the selected
    red route cluster and the same perpendicular-vs-route orientation check
    used by the RL red-light constraint.
    """
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for line in line_strings:
        if line.ndim != 2 or line.shape[1] < 3:
            continue
        points = line[:, :2]
        valid = (line[:, 2] > 0.5) & _valid_xy(points)
        for idx in range(len(points) - 1):
            if valid[idx] and valid[idx + 1]:
                start, end = points[idx], points[idx + 1]
                midpoint = 0.5 * (start + end)
                if np.linalg.norm(red_cluster - midpoint, axis=-1).min() > SIGNAL_MATCH_M:
                    continue
                nearest = int(np.linalg.norm(red_cluster - midpoint, axis=-1).argmin())
                route_direction = red_directions[nearest]
                route_norm = float(np.linalg.norm(route_direction))
                stop_direction = end - start
                stop_norm = float(np.linalg.norm(stop_direction))
                if route_norm > 1e-6 and stop_norm > 1e-6:
                    cosine = abs(float(np.dot(route_direction, stop_direction))) / (
                        route_norm * stop_norm
                    )
                    if cosine > 0.5:
                        continue
                segments.append((start, end))
    return segments


def _first_crossing(future_xy: np.ndarray, stop_segments: list[tuple[np.ndarray, np.ndarray]]) -> int | None:
    for frame in range(max(0, len(future_xy) - 1)):
        a, b = future_xy[frame], future_xy[frame + 1]
        if any(_segments_cross(a, b, c, d) for c, d in stop_segments):
            return frame + 1
    return None


def _red_cluster_info(route: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if route.ndim != 3 or route.shape[-1] <= TRAFFIC_LIGHT_RED:
        return None
    geometry = route[..., :4]
    valid = _valid_xy(geometry)
    red = (route[..., TRAFFIC_LIGHT_RED] > 0.5) & valid
    xy = route[..., :2]
    distance = np.linalg.norm(xy, axis=-1)
    relevant = red & (xy[..., 0] >= -LANE_BEHIND_M) & (distance <= MAX_RED_DISTANCE_M)
    if not np.any(relevant):
        return None
    red_xy = xy[relevant]
    red_directions = route[..., 2:4][relevant]
    d = np.linalg.norm(red_xy, axis=-1)
    cluster_mask = d <= d.min() + SIGNAL_MATCH_M
    return red_xy[cluster_mask], red_directions[cluster_mask]


def _signal_timeline(
    path: str,
    red_cluster: np.ndarray,
    future: np.ndarray,
    onset: int,
    required_end: int | None,
) -> tuple[list[bool | None], list[bool | None], bool]:
    """Track same-geometry red/green state for the whole expert horizon.

    Looking at the whole horizon is important for diagnosis: a light may turn
    green several seconds before a driver actually releases the brake.  That
    is different from a vehicle beginning to move while the matched signal is
    still red, and neither should be inferred from a narrow onset window.
    """
    red_states: list[bool | None] = [None] * len(future)
    green_states: list[bool | None] = [None] * len(future)
    loaded = False
    # Most candidates have a release before the logged motion onset.  Stop as
    # soon as that is proven; this avoids tens of thousands of unnecessary NAS
    # opens while retaining the full-horizon fallback for unresolved cases.
    end = len(future) if required_end is None else min(len(future), required_end + 1)
    for offset in range(1, end):
        sibling = _sibling_frame_path(path, offset)
        if sibling is None:
            continue
        try:
            with np.load(sibling, allow_pickle=False) as archive:
                route = np.asarray(archive["route_lanes"])
        except (FileNotFoundError, OSError, KeyError, ValueError):
            continue
        loaded = True
        if route.ndim != 3 or route.shape[-1] <= TRAFFIC_LIGHT_GREEN:
            continue
        valid = _valid_xy(route[..., :4])
        green_xy = route[..., :2][(route[..., TRAFFIC_LIGHT_GREEN] > 0.5) & valid]
        red_xy = route[..., :2][(route[..., TRAFFIC_LIGHT_RED] > 0.5) & valid]
        target = _transform_points_to_future_frame(red_cluster, future[offset])
        if len(green_xy):
            green_states[offset] = bool(
                np.linalg.norm(target[:, None, :] - green_xy[None, :, :], axis=-1).min()
                <= SIGNAL_MATCH_M
            )
        else:
            green_states[offset] = False
        if len(red_xy):
            red_states[offset] = bool(
                np.linalg.norm(target[:, None, :] - red_xy[None, :, :], axis=-1).min()
                <= SIGNAL_MATCH_M
            )
        else:
            red_states[offset] = False
        if offset <= onset and green_states[offset] is True:
            break
    return red_states, green_states, loaded


def _motion_onset(future: np.ndarray) -> int | None:
    if future.ndim != 2 or future.shape[0] == 0 or future.shape[1] < 2:
        return None
    xy = future[:, :2]
    previous = np.concatenate((np.zeros((1, 2), dtype=xy.dtype), xy[:-1]))
    step = np.linalg.norm(xy - previous, axis=-1)
    moving = step / FUTURE_DT_S >= MOTION_SPEED_MPS
    sustain = min(MOTION_SUSTAIN_FRAMES, len(future))
    windows = np.lib.stride_tricks.sliding_window_view(moving, sustain).all(axis=-1)
    distance = np.lib.stride_tricks.sliding_window_view(step, sustain).sum(axis=-1)
    hits = np.flatnonzero(windows & (distance >= MOTION_DISTANCE_M))
    return int(hits[0]) if len(hits) else None


def classify(path: str) -> dict[str, object]:
    result: dict[str, object] = {"path": path, "category": "invalid"}
    try:
        with np.load(path, allow_pickle=False) as archive:
            route = np.asarray(archive["route_lanes"])
            state = np.asarray(archive["ego_current_state"])
            past = np.asarray(archive["ego_agent_past"])
            future = np.asarray(archive["ego_agent_future"])
            line_strings = np.asarray(archive["line_strings"])
    except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
        result["category"] = "missing_or_unreadable"
        result["error"] = type(exc).__name__
        return result
    cluster_info = _red_cluster_info(route)
    if cluster_info is None or state.ndim != 1 or state.shape[0] < 6:
        result["category"] = "not_a_v1_candidate"
        return result
    cluster, cluster_directions = cluster_info
    if float(np.linalg.norm(state[4:6])) > STOP_SPEED_MPS:
        result["category"] = "not_stopped"
        return result
    if past.ndim == 2 and past.shape[1] >= 2 and past.shape[0] >= 2:
        n = min(HISTORY_FRAMES, past.shape[0] - 1)
        hist = past[-(n + 1) :, :2]
        distance = np.linalg.norm(np.diff(hist, axis=0), axis=-1)
        valid = _valid_xy(hist[1:]) | _valid_xy(hist[:-1])
        if np.any(distance[valid] > HISTORY_SPEED_MPS * FUTURE_DT_S):
            result["category"] = "moving_history"
            return result
    onset = _motion_onset(future)
    if onset is None:
        result["category"] = "no_future_motion"
        return result
    stop_segments = _stop_line_segments(line_strings, cluster, cluster_directions)
    crossing = _first_crossing(future[:, :2], stop_segments)
    red_states, green_states, neighbor_loaded = _signal_timeline(
        path, cluster, future, onset, crossing
    )
    green_before_onset_frames = [
        frame for frame in range(1, min(onset + 1, len(future))) if green_states[frame]
    ]
    green_after_onset_frames = [
        frame for frame in range(max(1, onset + 1), len(future)) if green_states[frame]
    ]
    green_before_onset = bool(green_before_onset_frames)
    green_after_onset = bool(green_after_onset_frames)
    red_at_onset = red_states[onset] if onset < len(red_states) else None
    red_at_crossing = None
    crossing_neighbor_loaded = False
    if crossing is not None:
        crossing_frame = min(crossing, len(future) - 1)
        red_at_crossing = red_states[crossing_frame]
        crossing_neighbor_loaded = red_at_crossing is not None
    if green_before_onset:
        category = "same_geometry_red_to_green_before_motion"
    elif crossing is None:
        if green_after_onset:
            category = "no_crossing_green_after_motion"
        elif red_at_onset is True:
            category = "no_crossing_red_at_motion_onset"
        else:
            category = "no_stop_line_crossing_signal_unobserved"
    elif red_at_crossing is True:
        category = "crossing_while_same_geometry_red"
    elif red_at_crossing is False and crossing_neighbor_loaded:
        category = "crossing_after_signal_release"
    else:
        category = "crossing_signal_unobserved"
    result.update(
        category=category,
        onset_frame=onset,
        crossing_frame=crossing,
        green_before_onset=green_before_onset,
        green_before_onset_first_frame=(green_before_onset_frames[0] if green_before_onset_frames else None),
        green_after_onset=green_after_onset,
        green_after_onset_first_frame=(green_after_onset_frames[0] if green_after_onset_frames else None),
        red_at_onset=red_at_onset,
        neighbor_loaded=neighbor_loaded,
        crossing_neighbor_loaded=crossing_neighbor_loaded,
        red_at_crossing=red_at_crossing,
        stop_line_segment_count=len(stop_segments),
        red_point_count=len(cluster),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--filtered", type=Path, action="append", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    removed: list[str] = []
    if len(args.source) != len(args.filtered):
        raise ValueError("--source and --filtered must have equal counts")
    for source, filtered in zip(args.source, args.filtered):
        source_values = json.loads(source.read_text(encoding="utf-8"))
        filtered_values = set(json.loads(filtered.read_text(encoding="utf-8")))
        removed.extend(path for path in source_values if path not in filtered_values)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        results = list(executor.map(classify, removed))
    counts: dict[str, int] = {}
    for row in results:
        category = str(row["category"])
        counts[category] = counts.get(category, 0) + 1
    report = {
        "candidate_count": len(removed),
        "counts": dict(sorted(counts.items())),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_count": len(removed), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
