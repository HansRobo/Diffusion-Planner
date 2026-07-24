"""Low-level closed-loop rollout helpers.

``run_closed_loop_eval`` is the tier4-main entry point used by ``train.py``.
New code should use :class:`scenario_generation.closed_loop_evaluation.FullRouteClosedLoopEvaluation`.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np

from scenario_generation.route_timeline import group_routes

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def route_label(npz_path: Path, key: str) -> str:
    """Human-readable route label ``<location>_<date>_<key>`` for video/PNG names.

    Dataset routes are laid out ``.../<location>/<split>/<date>/<time>/routes/<time>_<idx>_<frame>``
    -- the bag-prefix ``key`` (``<time>_<idx>``) alone drops the depot/site and date, which makes the
    per-segment MP4 names ambiguous. This prepends ``<location>`` (the dir two levels above the
    ``YYYY-MM-DD`` date component) and ``<date>``. Falls back to bare ``key`` for any path that does
    not match that layout (e.g. a flat single-dir npz tree).
    """
    parts = npz_path.parts
    date_idx = next((i for i, p in enumerate(parts) if _DATE_RE.match(p)), None)
    if date_idx is not None and date_idx >= 2:
        return f"{parts[date_idx - 2]}_{parts[date_idx]}_{key}"
    return key


def enumerate_routes(npz_root: Path) -> dict[str, list[Path]]:
    """Group all .npz under ``npz_root`` into routes (bag-prefix groups)."""
    paths = sorted(Path(npz_root).rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz under {npz_root}")
    return group_routes(paths)


def resolve_npz_roots(npz_root) -> list[Path]:
    """Resolve a closed-loop npz input into the list of root directories to enumerate.

    The input is either a single directory tree of NPZ frames (globbed recursively), or a
    ``.json`` file holding a list of such directory paths (one route dir per entry) -- the same
    "path list" form as ``--train_set_list`` / ``--valid_set_list``. A directory is returned as a
    one-element list; a JSON list is returned verbatim (each entry a ``Path``).
    """
    npz_root = Path(npz_root)
    if npz_root.suffix == ".json":
        entries = json.loads(npz_root.read_text())
        if not isinstance(entries, list) or not all(isinstance(e, str) for e in entries):
            raise ValueError(f"{npz_root} must be a JSON list of directory paths")
        if not entries:
            raise ValueError(f"{npz_root} is an empty path list")
        return [Path(e) for e in entries]
    return [npz_root]


def aggregate(rows: list[dict], near_miss_thresh: float) -> dict:
    """Aggregate per-segment metric rows into a single closed-loop summary."""
    n_seg = len(rows)
    total_steps = sum(r["n_steps_run"] for r in rows)
    total_collision_steps = sum(r["n_collision_steps"] for r in rows)
    total_near_miss_steps = sum(r["n_near_miss_steps"] for r in rows)
    total_snaps = sum(r["snap_count"] for r in rows)
    total_expand = sum(r["expand_count"] for r in rows)
    total_normal = sum(r["normal_steps"] for r in rows)
    total_repeat = sum(r["repeat_steps"] for r in rows)

    n_seg_collision = sum(1 for r in rows if r["n_collision_steps"] > 0)
    n_seg_near_miss = sum(1 for r in rows if r["n_near_miss_steps"] > 0)
    n_seg_diverged = sum(1 for r in rows if r["terminated"] == "diverged")

    finite_min_cl = [r["min_clearance"] for r in rows if np.isfinite(r["min_clearance"])]
    finite_mean_cl = [r["mean_clearance"] for r in rows if np.isfinite(r["mean_clearance"])]

    # Graded (non-saturating) headline metrics — see closed_loop_eval design notes: these
    # improve smoothly as the model trains, unlike the binary *_segment_rate / worst-moment
    # *_min_clearance keys (still computed below for continuity, but no longer displayed).
    total_collision_events = sum(r.get("n_collision_events", 0) for r in rows)
    total_curb_hits = sum(r.get("n_curb_hits", 0) for r in rows)
    completions = [r["route_completion"] for r in rows if "route_completion" in r]
    # gt-deviation pooled across all steps (weight each segment's mean by its step count, which
    # equals that segment's gt-deviation sample count), so long routes aren't under-weighted.
    dev_num = sum(
        r["mean_gt_deviation_m"] * r["n_steps_run"]
        for r in rows
        if np.isfinite(r.get("mean_gt_deviation_m", float("inf")))
    )
    dev_den = sum(
        r["n_steps_run"]
        for r in rows
        if np.isfinite(r.get("mean_gt_deviation_m", float("inf")))
    )
    mean_clearance = float(np.mean(finite_mean_cl)) if finite_mean_cl else float("inf")

    term_counts: dict[str, int] = {}
    for r in rows:
        term_counts[r["terminated"]] = term_counts.get(r["terminated"], 0) + 1

    return {
        "near_miss_thresh": near_miss_thresh,
        "n_segments": n_seg,
        "total_steps": total_steps,
        # --- displayed (graded) ---
        "mean_route_completion": float(np.mean(completions)) if completions else 0.0,
        "mean_clearance": mean_clearance,
        "mean_gt_deviation_m": float(dev_num / dev_den) if dev_den else float("inf"),
        "total_collision_events": total_collision_events,
        "total_curb_hits": total_curb_hits,
        "total_collision_steps": total_collision_steps,
        "collision_step_rate": total_collision_steps / total_steps if total_steps else 0.0,
        "total_near_miss_steps": total_near_miss_steps,
        "near_miss_step_rate": total_near_miss_steps / total_steps if total_steps else 0.0,
        "total_snaps": total_snaps,
        "n_segments_diverged": n_seg_diverged,
        # --- kept for continuity / episode table, NOT displayed as headline (saturating) ---
        "n_segments_with_collision": n_seg_collision,
        "collision_segment_rate": n_seg_collision / n_seg if n_seg else 0.0,
        "n_segments_with_near_miss": n_seg_near_miss,
        "near_miss_segment_rate": n_seg_near_miss / n_seg if n_seg else 0.0,
        "diverged_segment_rate": n_seg_diverged / n_seg if n_seg else 0.0,
        "global_min_clearance": float(min(finite_min_cl)) if finite_min_cl else float("inf"),
        "mean_segment_min_clearance": float(np.mean(finite_min_cl))
        if finite_min_cl
        else float("inf"),
        "mean_segment_mean_clearance": mean_clearance,
        "total_expand_count": total_expand,
        "total_normal_steps": total_normal,
        "total_repeat_steps": total_repeat,
        "repeat_step_rate": total_repeat / total_steps if total_steps else 0.0,
        "terminated_counts": term_counts,
    }


def build_mp4(png_dir: Path, mp4_path: Path, fps: float) -> None:
    """Encode the PNG sequence in ``png_dir`` to an MP4."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-pattern_type",
            "glob",
            "-i",
            str(png_dir / "*.png"),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            str(mp4_path),
        ],
        check=True,
    )


def run_closed_loop_eval(
    model,
    model_args,
    npz_root,
    out_dir,
    *,
    device: str,
    near_miss_thresh: float,
    search_radius: float,
    warmup_steps: int,
    unstick_after: int,
    unstick_advance_m: float,
    fps: float,
    draw_every: int,
    neighbor_history_mode: str,
    replan_interval: int = 10,
    unstick_radius_mult: float = 10.0,
    unstick_teleport_after: int = 300,
    tracker_mode: str = "mpc",
    verbose: bool = True,
    shard: tuple[int, int] | None = None,
    yaw_gate: bool = True,
    abort_deviation_m: float = 0.0,
    abort_after: int = 30,
    abort_max_snaps: int = 0,
    drop_objects: bool = False,
) -> dict:
    """Render closed-loop rollouts over every route under ``npz_root`` and aggregate metrics.

    ``shard=(rank, world_size)`` restricts this call to the ``rank``-th slice of the sorted route
    list (``route_keys[rank::world_size]``) and writes its rows to ``segments_{rank}.jsonl`` instead
    of the merged ``segments.jsonl``/``summary.json`` -- the route-level multi-GPU parallel driver in
    ``valid_predictor_closed_loop.py`` spawns one such call per worker (route keys are globally
    unique, so all shards share ``out_dir`` for the per-route PNG dirs and MP4s without collision).
    The returned summary is then per-shard; the parent merges the ``segments_*.jsonl`` rows.

    ``model`` must be an eval-mode Diffusion-Planner (callable ``model(data) -> (_, outputs)`` with
    ``outputs["prediction"]``); ``model_args`` provides ``observation_normalizer`` /
    ``predicted_neighbor_num`` / ``future_len`` (a ``Config`` or ``TrainConfig``). Each route is
    rolled out whole (no sub-segmenting) into one PNG dir + one MP4 (``<route>.mp4``).
    ``segments.jsonl`` (one row per route) and ``summary.json`` are written into ``out_dir``.

    Turn indicators are CLOSED-LOOP: the model's own predicted turn indicator is fed back into
    the input history each step, held across cached-plan steps when ``replan_interval`` > 1
    (see ``render_segment``).

    ``drop_objects``: empty-world ablation, forwarded to ``render_segment`` (see its docstring
    and ``objects_ablation_strategy.md``).

    Returns the summary dict with extra keys ``video_mp4s`` (list[Path] of every per-route MP4),
    ``segments`` (list[row]), and ``elapsed_sec``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # npz_root is either one directory tree or a JSON path list of route dirs; enumerate each and
    # merge, disambiguating any bag-prefix key that collides across roots and remembering the source
    # root of each route so its pose-sidecar fallback stays scoped to that tree.
    roots = resolve_npz_roots(npz_root)
    routes: dict[str, list[Path]] = {}
    route_sidecar_dir: dict[str, Path] = {}
    for root in roots:
        for key, paths in enumerate_routes(root).items():
            # Key each route by its <location>_<date>_<time>_<idx> label so the per-segment PNG dirs
            # and MP4s carry the site + date, not just the ambiguous time-of-day bag prefix.
            label = route_label(paths[0], key)
            uniq, n = label, 1
            while uniq in routes:
                uniq, n = f"{label}#{n}", n + 1
            routes[uniq] = paths
            route_sidecar_dir[uniq] = root
    route_keys = sorted(routes)
    if shard is not None:
        rank, world_size = shard
        route_keys = route_keys[rank::world_size]

    timers = Timers()
    rows: list[dict] = []
    video_mp4s: list[Path] = []
    t0 = time.perf_counter()

    segments_name = "segments.jsonl" if shard is None else f"segments_{shard[0]}.jsonl"
    fout = open(out_dir / segments_name, "w")
    try:
        for ri, key in enumerate(route_keys):
            tl = RouteTimeline(routes[key], sidecar_dir=route_sidecar_dir[key], timers=timers)
            # One route = one whole-route rollout = one <key>.mp4 (no sub-segmenting).
            png_dir = out_dir / key
            metrics = render_segment(
                model,
                model_args,
                tl,
                0,
                len(tl),
                png_dir,
                device=device,
                near_miss_thresh=near_miss_thresh,
                search_radius=search_radius,
                warmup_steps=warmup_steps,
                unstick_after=unstick_after,
                unstick_advance_m=unstick_advance_m,
                unstick_radius_mult=unstick_radius_mult,
                unstick_teleport_after=unstick_teleport_after,
                replan_interval=replan_interval,
                draw_every=draw_every,
                neighbor_history_mode=neighbor_history_mode,
                tracker_mode=tracker_mode,
                yaw_gate=yaw_gate,
                abort_deviation_m=abort_deviation_m,
                abort_after=abort_after,
                abort_max_snaps=abort_max_snaps,
                drop_objects=drop_objects,
            )
            row = {"route": key, **metrics}
            fout.write(json.dumps(row, default=float) + "\n")
            fout.flush()
            rows.append(row)

            # A route that terminates at step 0 (e.g. ego starts within goal_reach_m) draws no PNG;
            # skip the empty ffmpeg call (its glob would error on an empty dir).
            if not any(png_dir.glob("*.png")):
                if verbose:
                    print(f"[{ri + 1}/{len(route_keys)}] {key} -> 0 frames, no video")
                continue
            seg_mp4 = out_dir / f"{key}.mp4"
            # Raw fps: with only every draw_every-th frame drawn, the video plays
            # draw_every x faster than real time. For real time use fps = 10 / draw_every.
            build_mp4(png_dir, seg_mp4, fps)
            video_mp4s.append(seg_mp4)
            if verbose:
                print(
                    f"[{ri + 1}/{len(route_keys)}] {key} -> {seg_mp4.name}  "
                    f"coll={metrics['n_collision_steps']} near={metrics['n_near_miss_steps']} "
                    f"min_clr={metrics['min_clearance']:.3f}"
                )
    finally:
        fout.close()

    summary = aggregate(rows, near_miss_thresh)
    summary["npz_root"] = str(npz_root)
    summary["n_routes"] = len(route_keys)
    summary["elapsed_sec"] = time.perf_counter() - t0
    summary["video_mp4s"] = video_mp4s
    summary["segments"] = rows

    # A sharded worker leaves the merged summary.json to the parent driver (which aggregates every
    # shard's segments_*.jsonl); it only owns its own segments_{rank}.jsonl, written above.
    if shard is None:
        with open(out_dir / "summary.json", "w") as f:
            json.dump(
                {k: v for k, v in summary.items() if k not in ("video_mp4s", "segments")},
                f,
                indent=4,
            )

    return summary
