"""Evaluate closed-loop replay trajectories against road borders.

Reads ``trajectory_log.json`` from a replay output directory and computes
per-step and aggregate metrics: road-border distance/crossing, speed
profile, path length, stopped fraction, and progress toward goal.

Usage:
    python -m scenario_generation.tools.eval_cl_trajectory \
        --run_dirs cl_baseline cl_ep5 cl_ep9 \
        --map_path /path/to/lanelet2_map.osm \
        --ego_length 4.5 --ego_width 1.9 --ego_wheelbase 2.925
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from planner_metrics.vehicle_collision import obb_corners


def _load_trajectory(run_dir: Path) -> list[dict]:
    log_path = run_dir / "trajectory_log.json"
    if not log_path.exists():
        raise FileNotFoundError(f"No trajectory_log.json in {run_dir}")
    with open(log_path) as f:
        return json.load(f)


_EDGE_SAMPLES_PER_SIDE = 8


def _compute_ego_corners(
    x: float,
    y: float,
    heading: float,
    half_length: float,
    half_width: float,
    wheelbase: float,
) -> list[tuple[float, float]]:
    """Compute 4 corners of the ego bounding box in world frame.

    Matches the repo-wide convention (e.g. ``gui.lanelet_scene_builder._obb_corners``
    and ``visualize.draw_agent_box``) where ``(x, y)`` is the rear-axle
    position. The longitudinal footprint spans from ``-rear_overhang`` behind
    the rear axle to ``wheelbase + rear_overhang`` in front of it.

    Corner construction is delegated to the canonical
    :func:`planner_metrics.vehicle_collision.obb_corners`; this helper takes
    HALF extents, so they are doubled before the call. The returned corners are
    used order-agnostically (min corner-to-border distance), so the ring order
    of the wrapper is fine.
    """
    corners = obb_corners(x, y, heading, 2.0 * half_length, 2.0 * half_width, wheelbase)
    return [(float(cx), float(cy)) for cx, cy in corners]


def _compute_ego_perimeter(
    x: float,
    y: float,
    heading: float,
    half_length: float,
    half_width: float,
    wheelbase: float,
    samples_per_side: int = _EDGE_SAMPLES_PER_SIDE,
) -> np.ndarray:
    """Sample points along the 4 OBB edges in world frame.

    Corner-only distance misses the case where a road-border segment
    intersects the footprint near the middle of an edge but stays far from
    every corner. We walk each of the 4 OBB edges with
    ``np.linspace(0, 1, samples_per_side, endpoint=False)`` — i.e. each
    edge contributes ``samples_per_side`` points, including its starting
    corner but excluding its end corner (which is the next edge's start).
    That means every corner appears exactly once globally and no point is
    duplicated.

    Returns (K, 2) with ``K == 4 * samples_per_side``.
    """
    length = 2 * half_length
    rear_overhang = (length - wheelbase) / 2
    front = wheelbase + rear_overhang
    rear = -rear_overhang
    # Local-frame corners in rectangle order (front-right, front-left, rear-left, rear-right).
    corners_local = np.array(
        [
            [front, -half_width],
            [front, half_width],
            [rear, half_width],
            [rear, -half_width],
        ],
        dtype=np.float64,
    )

    ts = np.linspace(0.0, 1.0, samples_per_side, endpoint=False)  # drop duplicate end
    edge_pts = []
    for i in range(4):
        a = corners_local[i]
        b = corners_local[(i + 1) % 4]
        edge_pts.append(a[None, :] + ts[:, None] * (b - a)[None, :])
    local = np.concatenate(edge_pts, axis=0)  # (4 * samples_per_side, 2)

    cos_h, sin_h = math.cos(heading), math.sin(heading)
    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    return (local @ rot.T) + np.array([x, y])


def _flatten_segments(border_segments: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Flatten list of polylines into (M, 2) segment start + (M, 2) segment end arrays."""
    starts = []
    ends = []
    for seg in border_segments:
        if len(seg) < 2:
            continue
        starts.append(seg[:-1])
        ends.append(seg[1:])
    if not starts:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.concatenate(starts, axis=0), np.concatenate(ends, axis=0)


_SEGMENT_BLOCK_SIZE = 4096


def _min_dist_vectorized(
    points: np.ndarray,  # (K, 2) any set of query points (corners or perimeter samples)
    seg_starts: np.ndarray,  # (M, 2)
    seg_ends: np.ndarray,  # (M, 2)
) -> float:
    """Vectorized min distance from any query point to any line segment.

    Processes segments in blocks of ``_SEGMENT_BLOCK_SIZE`` so the
    intermediate ``(K, M, …)`` arrays don't materialize over the entire
    map at once. Running min across blocks preserves the global minimum.
    """
    if len(points) == 0 or len(seg_starts) == 0:
        return float("inf")

    min_dist = float("inf")
    for start in range(0, len(seg_starts), _SEGMENT_BLOCK_SIZE):
        end = start + _SEGMENT_BLOCK_SIZE
        s_starts = seg_starts[start:end]
        s_ends = seg_ends[start:end]

        ab = s_ends - s_starts
        ab_len2 = (ab * ab).sum(axis=1)
        ab_len2_safe = np.where(ab_len2 < 1e-12, 1.0, ab_len2)

        ap = points[:, None, :] - s_starts[None, :, :]  # (K, B, 2)
        dot = (ap * ab[None, :, :]).sum(axis=2)  # (K, B)
        t = np.clip(dot / ab_len2_safe[None, :], 0.0, 1.0)
        proj = s_starts[None, :, :] + t[:, :, None] * ab[None, :, :]  # (K, B, 2)
        delta = points[:, None, :] - proj
        dist = np.sqrt((delta * delta).sum(axis=2))  # (K, B)
        # Degenerate (zero-length) segments fall back to point distance. Only pay for that
        # fallback when a block actually holds one: it is a full (K, B, 2) norm, and computing it
        # unconditionally was ~40% of this function's cost on real maps, which have none
        # (measured: 2.84 s -> 1.63 s on a 1700-tick case, bit-identical). The `ab_len2_safe`
        # divide already drives `proj` to `s_starts` for a degenerate segment, so the two agree
        # mathematically -- the branch is kept because `np.linalg.norm` and
        # `sqrt(sum(d*d))` need not round identically, and rb_dists feeds clearance statistics.
        degenerate = ab_len2 < 1e-12
        if degenerate.any():
            dist_deg = np.linalg.norm(points[:, None, :] - s_starts[None, :, :], axis=2)
            dist = np.where(degenerate[None, :], dist_deg, dist)

        block_min = float(dist.min())
        if block_min < min_dist:
            min_dist = block_min

    return min_dist


# A whole trajectory covers ~250 m, so narrowing the segment set once for all of it still leaves
# every tick paying for geometry hundreds of metres away. 50 ticks is ~5 s of driving -- a window
# small enough that the narrowing bites (measured 1073 -> 270 segments per tick on a real case) and
# large enough that the narrowing itself stays negligible. Below ~25 the returns flatten.
_TICK_BLOCK = 50
# Segments beyond this from the block's path cannot be the nearest, PROVIDED the distances the
# block reports stay inside it -- which is checked, not assumed (see the fallback below). Same
# bound and same argument as scenario_sim_rollout._prune_border_segments, one level finer.
_BLOCK_MARGIN_M = 50.0


def _block_segment_mask(
    seg_starts: np.ndarray,
    seg_ends: np.ndarray,
    xy: np.ndarray,
    reach: float,
    margin: float = _BLOCK_MARGIN_M,
) -> np.ndarray:
    """Which segments could be within ``margin`` of the ego anywhere in this block of ticks."""
    lo = xy.min(axis=0) - (reach + margin)
    hi = xy.max(axis=0) + (reach + margin)
    return (np.maximum(seg_starts, seg_ends) >= lo).all(axis=1) & (
        np.minimum(seg_starts, seg_ends) <= hi
    ).all(axis=1)


def evaluate_trajectory(
    traj: list[dict],
    border_segments: list[np.ndarray],
    ego_length: float,
    ego_width: float,
    ego_wheelbase: float,
    rb_cross_thresh: float = 0.20,
) -> dict:
    """Compute metrics for a single CL trajectory.

    The road-border distance is exact: the per-block narrowing below only ever drops segments
    that provably cannot be the nearest one, and falls back to the full set for any block whose
    result would depend on that being true.
    """
    half_l = ego_length / 2
    half_w = ego_width / 2

    seg_starts, seg_ends = _flatten_segments(border_segments)
    has_borders = seg_starts.shape[0] > 0
    # Furthest a sampled perimeter point can be from the reported pose (the OBB reaches
    # (length + wheelbase) / 2 forward of it -- see scenario_sim_rollout._ego_metric_box).
    reach = float(np.hypot(0.5 * (ego_length + ego_wheelbase), 0.5 * ego_width))

    rb_dists: list[float] = []
    speeds = []
    positions = []

    for entry in traj:
        positions.append((entry["x"], entry["y"]))
        speeds.append(entry["speed"])

    if has_borders:
        for b0 in range(0, len(traj), _TICK_BLOCK):
            block = traj[b0:b0 + _TICK_BLOCK]
            mask = _block_segment_mask(
                seg_starts, seg_ends, np.asarray(positions[b0:b0 + _TICK_BLOCK]), reach
            )
            b_starts, b_ends = seg_starts[mask], seg_ends[mask]

            block_dists = []
            for entry in block:
                # Sample the full OBB perimeter, not just corners: a border that
                # pierces the middle of a vehicle edge can leave every corner
                # outside rb_cross_thresh but still be a true crossing.
                perimeter = _compute_ego_perimeter(
                    entry["x"],
                    entry["y"],
                    entry["heading"],
                    half_l,
                    half_w,
                    ego_wheelbase,
                )
                block_dists.append(_min_dist_vectorized(perimeter, b_starts, b_ends))

            # The narrowing is equivalent to the full scan only while the distances it produces
            # stay inside the margin; a segment beyond it could otherwise have been nearer.
            # Checking the result rather than trusting the assumption is what keeps this exact --
            # and it costs the full scan only for the blocks that need it.
            #
            # "No finite distance" needs the fallback too, and is the easier case to get wrong:
            # a block that runs more than the margin from every border keeps NO segments, and
            # `_min_dist_vectorized` then returns inf for each of its ticks. Treating that as
            # "nothing to check" would report inf where the true distance is a large number --
            # not a rounding difference but a different value, and one that then propagates into
            # rb_dist_min/med/p5 as a silent inf.
            finite = [d for d in block_dists if math.isfinite(d)]
            narrowed = len(b_starts) < len(seg_starts)
            if narrowed and (not finite or max(finite) > _BLOCK_MARGIN_M):
                block_dists = [
                    _min_dist_vectorized(
                        _compute_ego_perimeter(
                            e["x"], e["y"], e["heading"], half_l, half_w, ego_wheelbase
                        ),
                        seg_starts,
                        seg_ends,
                    )
                    for e in block
                ]
            rb_dists.extend(block_dists)

    rb_dists = np.array(rb_dists)
    speeds = np.array(speeds)
    positions = np.array(positions)

    # Path length
    if len(positions) > 1:
        path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    else:
        path_length = 0.0

    # RB metrics (skipped if the map has no road-border polylines).
    if len(rb_dists) > 0:
        rb_crossings = int((rb_dists < rb_cross_thresh).sum())
        first_rb_cross = int(np.argmax(rb_dists < rb_cross_thresh)) if rb_crossings > 0 else -1
    else:
        rb_crossings = 0
        first_rb_cross = -1

    # Progress
    start_goal_d = traj[0]["goal_d"] if traj else 0
    end_goal_d = traj[-1]["goal_d"] if traj else 0
    progress = start_goal_d - end_goal_d

    # Duration
    duration_s = len(traj) * 0.1

    # When the map has no road-border polylines we return NaN for
    # distance-valued metrics so they are distinguishable from a real
    # zero-distance crossing in downstream summaries/plots.
    rb_has_data = len(rb_dists) > 0
    return {
        "n_steps": len(traj),
        "duration_s": duration_s,
        "path_length_m": path_length,
        "progress_m": progress,
        "start_goal_d": start_goal_d,
        "end_goal_d": end_goal_d,
        "mean_speed_mps": float(speeds.mean()) if len(speeds) > 0 else 0,
        "max_speed_mps": float(speeds.max()) if len(speeds) > 0 else 0,
        "rb_has_data": rb_has_data,
        "rb_dist_min": float(rb_dists.min()) if rb_has_data else float("nan"),
        "rb_dist_p5": float(np.percentile(rb_dists, 5)) if rb_has_data else float("nan"),
        "rb_dist_p25": float(np.percentile(rb_dists, 25)) if rb_has_data else float("nan"),
        "rb_dist_med": float(np.median(rb_dists)) if rb_has_data else float("nan"),
        "rb_cross_steps": rb_crossings,
        "rb_cross_frac": rb_crossings / max(len(traj), 1) if rb_has_data else float("nan"),
        "first_rb_cross_step": first_rb_cross,
        "stopped_steps": int((speeds < 0.1).sum()),
        "stopped_frac": float((speeds < 0.1).mean()) if len(speeds) > 0 else 0,
        # Per-step series, not just the summary quantiles above: the closed-loop segment-row
        # schema needs the raw distances to build the road_border block (event counts with
        # falling-edge debounce, and clearance stats shared with the reproducer path).
        # Empty array when the map ships no road-border polylines.
        "rb_dists": rb_dists,
    }


# map path -> its road_border polylines. Parsing a 46 MB lanelet2 map costs ~7 s and this is
# called once per SCENARIO from the rollout's finalize step, where 378 of the suite's 464 cases
# share one map -- the same per-map/per-scenario mismatch that `LaneletSceneBuilder` already
# caches around (scenario_sim_pool.py:120-124). It made `finalize` 9.19 s/case = 20% of a full
# run's cost (plan/11 9v/9x). Keyed by resolved path, and unbounded because the number of distinct
# maps a process sees is the number of maps in the suite (3), not the number of scenarios.
_BORDER_SEGMENT_CACHE: dict[str, list[np.ndarray]] = {}


def load_border_segments(map_path: str) -> list[np.ndarray]:
    """Load road border polylines from a lanelet2 map (cached per map).

    Returns a fresh list each call, so a caller may filter it in place (see
    ``scenario_sim_rollout._prune_border_segments``, which does not, but whose result feeds
    metrics -- a shared list that someone later trims would move ``rb_dists`` silently). The
    ARRAYS are shared, and must be treated as read-only; every current consumer only reads them.
    """
    key = str(Path(map_path).resolve())
    cached = _BORDER_SEGMENT_CACHE.get(key)
    if cached is not None:
        return list(cached)

    import lanelet2
    from autoware_lanelet2_extension_python.projection import MGRSProjector

    projector = MGRSProjector(lanelet2.io.Origin(0.0, 0.0))
    ll_map = lanelet2.io.load(map_path, projector)

    segments = []
    for ls in ll_map.lineStringLayer:
        attrs = ls.attributes
        ls_type = attrs["type"] if "type" in attrs else ""
        ls_subtype = attrs["subtype"] if "subtype" in attrs else ""
        if ls_type == "road_border" or ls_subtype == "road_border":
            pts = np.array([[p.x, p.y] for p in ls], dtype=np.float64)
            if len(pts) >= 2:
                segments.append(pts)
    print(f"Loaded {len(segments)} road border segments from map")
    _BORDER_SEGMENT_CACHE[key] = segments
    return list(segments)


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid float value: {value!r}") from e
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError(f"value must be > 0, got {parsed}")
    return parsed


def main():
    parser = argparse.ArgumentParser(description="Evaluate CL replay trajectories")
    parser.add_argument(
        "--run_dirs",
        nargs="+",
        required=True,
        help="Replay output directories (each must contain trajectory_log.json)",
    )
    parser.add_argument("--map_path", required=True, help="Lanelet2 map OSM file")
    parser.add_argument(
        "--ego_length",
        type=_positive_float,
        required=True,
        help="Ego length (m) — must match the vehicle used for replay",
    )
    parser.add_argument(
        "--ego_width",
        type=_positive_float,
        required=True,
        help="Ego width (m) — must match the vehicle used for replay",
    )
    parser.add_argument(
        "--ego_wheelbase",
        type=_positive_float,
        required=True,
        help="Ego wheelbase (m) — must match the vehicle used for replay",
    )
    parser.add_argument("--rb_cross_thresh", type=float, default=0.20)
    parser.add_argument("--output", type=str, default=None, help="Save results JSON")
    args = parser.parse_args()

    if args.ego_wheelbase > args.ego_length:
        parser.error(
            f"--ego_wheelbase ({args.ego_wheelbase}) must be <= --ego_length ({args.ego_length})"
        )

    border_segments = load_border_segments(args.map_path)

    results = {}
    for run_dir_str in args.run_dirs:
        run_dir = Path(run_dir_str)
        name = run_dir.name
        print(f"\n=== {name} ===")
        try:
            traj = _load_trajectory(run_dir)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        metrics = evaluate_trajectory(
            traj,
            border_segments,
            args.ego_length,
            args.ego_width,
            args.ego_wheelbase,
            args.rb_cross_thresh,
        )
        results[name] = metrics

        print(f"  Steps: {metrics['n_steps']}, Duration: {metrics['duration_s']:.1f}s")
        print(f"  Path: {metrics['path_length_m']:.1f}m, Progress: {metrics['progress_m']:.1f}m")
        print(
            f"  Speed: mean={metrics['mean_speed_mps']:.2f} max={metrics['max_speed_mps']:.2f} m/s"
        )
        print(
            f"  RB dist: min={metrics['rb_dist_min']:.3f} p5={metrics['rb_dist_p5']:.3f} "
            f"p25={metrics['rb_dist_p25']:.3f} med={metrics['rb_dist_med']:.3f}"
        )
        print(f"  RB crossings: {metrics['rb_cross_steps']} steps ({metrics['rb_cross_frac']:.1%})")
        if metrics["first_rb_cross_step"] >= 0:
            print(
                f"    First crossing at step {metrics['first_rb_cross_step']} "
                f"({metrics['first_rb_cross_step'] * 0.1:.1f}s)"
            )
        print(f"  Stopped: {metrics['stopped_steps']} steps ({metrics['stopped_frac']:.1%})")
        print(f"  Goal: {metrics['start_goal_d']:.0f}m -> {metrics['end_goal_d']:.0f}m")

    if len(results) > 1:
        print("\n=== COMPARISON ===")
        header = f"{'Metric':<20s}"
        for name in results:
            header += f" {name:>15s}"
        print(header)
        for key in [
            "path_length_m",
            "mean_speed_mps",
            "rb_dist_min",
            "rb_dist_med",
            "rb_cross_steps",
            "stopped_frac",
            "progress_m",
        ]:
            row = f"{key:<20s}"
            for name in results:
                v = results[name][key]
                row += f" {v:>15.3f}" if isinstance(v, float) else f" {v:>15d}"
            print(row)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
