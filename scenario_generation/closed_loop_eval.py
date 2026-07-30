"""Reusable closed-loop rollout + render + metric aggregation.

Shared by the standalone CLI (``diffusion_planner/valid_predictor_closed_loop.py``) and the
per-epoch training validation (``diffusion_planner/diffusion_planner/train.py``): both drive the
ego in CLOSED LOOP through ``reproducer_rollout.render_segment`` over the route NPZ frames under
``npz_root``, write a per-step PNG, build one MP4 per segment, and aggregate the per-segment
metrics into a single summary.

``run_closed_loop_eval`` takes an already-loaded ``(model, model_args)`` (so training can pass its
live model + ``TrainConfig`` straight in, no checkpoint reload) and returns the summary dict plus
the per-segment MP4 paths (for wandb upload).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from scenario_generation.metrics.tdigest import TDIGEST_KEY, is_tdigest_key, merged_percentile
from scenario_generation.perf_timer import Timers
from scenario_generation.reproducer_rollout import render_segment
from scenario_generation.route_timeline import RouteTimeline, group_routes
from scenario_generation.scenario_sim_metrics import failed_segment_row

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Clearance t-digests live in a sidecar ``tdigests.jsonl`` / ``tdigests_{rank}.jsonl``
# (not in the human-readable ``segments*.jsonl``) so multi-GPU merge can still pool p5.


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

    The input is a single directory tree of NPZ frames (globbed recursively), a ``.json`` file
    holding a list of such directory paths (one route dir per entry) -- the same "path list"
    form as ``--train_set_list`` / ``--valid_set_list`` -- or an already-resolved list of paths
    (e.g. from ``site_discovery.discover_sites_from_json``, which does its own per-site
    grouping). A directory is returned as a one-element list; a JSON list or a pre-resolved
    list is returned verbatim (each entry a ``Path``).
    """
    if isinstance(npz_root, (list, tuple)):
        return [Path(p) for p in npz_root]
    npz_root = Path(npz_root)
    if npz_root.suffix == ".json":
        entries = json.loads(npz_root.read_text())
        if not isinstance(entries, list) or not all(isinstance(e, str) for e in entries):
            raise ValueError(f"{npz_root} must be a JSON list of directory paths")
        if not entries:
            raise ValueError(f"{npz_root} is an empty path list")
        return [Path(e) for e in entries]
    return [npz_root]


def enumerate_multi_root_routes(npz_root) -> tuple[dict[str, list[Path]], dict[str, Path]]:
    """Merge routes from every root in ``resolve_npz_roots(npz_root)`` into one route dict.

    Disambiguates any bag-prefix key that collides across roots (via ``route_label``, then a
    numeric suffix as a last resort) and remembers the source root of each route, so a
    downstream pose-sidecar fallback stays scoped to the correct tree. Returns
    ``(routes, route_sidecar_dir)``, both keyed by the final (disambiguated) route key.
    """
    roots = resolve_npz_roots(npz_root)
    routes: dict[str, list[Path]] = {}
    route_sidecar_dir: dict[str, Path] = {}
    for root in roots:
        for key, paths in enumerate_routes(root).items():
            # Key each route by its <location>_<date>_<time>_<idx> label so the per-segment PNG
            # dirs and MP4s carry the site + date, not just the ambiguous time-of-day bag prefix.
            label = route_label(paths[0], key)
            uniq, n = label, 1
            while uniq in routes:
                uniq, n = f"{label}#{n}", n + 1
            routes[uniq] = paths
            route_sidecar_dir[uniq] = root
    return routes, route_sidecar_dir


def _require_block(row: dict, category: str) -> dict:
    """Return nested metric block; fail fast if missing (no silent zeros)."""
    block = row.get(category)
    if not isinstance(block, dict):
        raise KeyError(
            f"segment metrics missing nested category {category!r} (got keys={sorted(row.keys())})"
        )
    return block


def _event_family_block(
    rows: list[dict],
    category: str,
    *,
    dual: bool,
    total_steps: int,
    n_seg: int,
    thresh_key: str | None = None,
    thresh_value: float | None = None,
) -> dict:
    """Roll up one nested metric category across segment rows."""
    out: dict = {}
    if thresh_key is not None and thresh_value is not None:
        out[thresh_key] = thresh_value

    if dual:
        prefixes = ("collision", "miss")
        for p in prefixes:
            steps = sum(int(_require_block(r, category)[f"{p}_steps"]) for r in rows)
            count = sum(int(_require_block(r, category)[f"{p}_count"]) for r in rows)
            segs = sum(1 for r in rows if int(_require_block(r, category)[f"{p}_steps"]) > 0)
            out[f"{p}_steps"] = steps
            out[f"{p}_count"] = count
            out[f"{p}_segments"] = segs
            out[f"{p}_step_rate"] = steps / total_steps if total_steps else 0.0
            out[f"{p}_segment_rate"] = segs / n_seg if n_seg else 0.0
    else:
        steps = sum(int(_require_block(r, category)["steps"]) for r in rows)
        count = sum(int(_require_block(r, category)["count"]) for r in rows)
        segs = sum(1 for r in rows if int(_require_block(r, category)["steps"]) > 0)
        out["steps"] = steps
        out["count"] = count
        out["segments"] = segs
        out["step_rate"] = steps / total_steps if total_steps else 0.0
        out["segment_rate"] = segs / n_seg if n_seg else 0.0

    return out


def _pool_clearance(rows: list[dict], category: str) -> dict[str, float]:
    """Global clearance min/mean/p5 from per-segment stats (+ optional t-digests)."""
    mins = []
    means = []
    weights = []
    p5s = []
    digests = []
    for r in rows:
        block = _require_block(r, category)
        mn = block["clearance_min_m"]
        mean = block["clearance_mean_m"]
        p5 = block["clearance_p5_m"]
        digest = block.get(TDIGEST_KEY)
        # Weight by finite clearance samples only (not n_steps_run, which includes inf steps).
        n = int(block.get("clearance_finite_steps", 0))
        if np.isfinite(mn):
            mins.append(float(mn))
        if np.isfinite(mean) and n > 0:
            means.append(float(mean))
            weights.append(n)
        if np.isfinite(p5):
            p5s.append(float(p5))
        if digest is not None:
            digests.append(digest)
    return {
        "clearance_min_m": float(min(mins)) if mins else float("inf"),
        "clearance_mean_m": float(np.average(means, weights=weights)) if means else float("inf"),
        # Prefer merged t-digest p5; fall back to min of segment p5s (conservative).
        "clearance_p5_m": (
            merged_percentile(digests, 5) if digests else (float(min(p5s)) if p5s else float("inf"))
        ),
    }


def metrics_for_json(metrics: dict) -> dict:
    """Numpy → JSON-friendly types; strip ``_tdigest`` (written to a sidecar instead)."""

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items() if not is_tdigest_key(k)}
        if isinstance(obj, list):
            return [_clean(x) for x in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    return _clean(metrics)


def segment_row_for_json(metrics: dict, **extra) -> dict:
    """Merge cleaned metrics with caller extras (no ``_tdigest``) for segments.jsonl.

    Shared by closed-loop eval and R2LPL mining writers so human-readable rows stay
    consistent. Digests still leave via ``tdigest_sidecar_row`` when the caller wants them.
    Caller ``extra`` keys override cleaned metrics on collision (e.g. ``route=...``).
    """
    return {**metrics_for_json(metrics), **extra}


def tdigest_sidecar_row(metrics: dict) -> dict | None:
    """Extract ``route`` + per-category ``_tdigest`` blobs for the digests sidecar file.

    Returns ``None`` when the row has no digests (nothing to persist for shard merge).
    """
    out: dict = {}
    if "route" in metrics:
        out["route"] = metrics["route"]
    for key, val in metrics.items():
        if not isinstance(val, dict):
            continue
        digest = val.get(TDIGEST_KEY)
        if digest is not None:
            out[key] = digest
    # Only ``route`` (and no digests) is not worth writing.
    return out if len(out) > (1 if "route" in out else 0) else None


def attach_tdigest_sidecars(rows: list[dict], sidecar_rows: list[dict]) -> None:
    """Mutate ``rows`` in place: attach sidecar digests under each category's ``_tdigest``."""
    by_route = {r["route"]: r for r in sidecar_rows if "route" in r}
    for row in rows:
        side = by_route.get(row.get("route"))
        if side is None:
            continue
        for cat, digest in side.items():
            if cat == "route":
                continue
            block = row.get(cat)
            if isinstance(block, dict):
                block[TDIGEST_KEY] = digest


def load_segment_rows_with_tdigests(
    out_dir: Path, *, shard_glob: str = "segments_*.jsonl"
) -> list[dict]:
    """Load human-readable segment rows and reattach digests from ``tdigests_*.jsonl`` sidecars."""
    rows: list[dict] = []
    for f in sorted(Path(out_dir).glob(shard_glob)):
        # Skip any accidental non-segment match (e.g. if naming ever collides).
        if "tdigest" in f.name:
            continue
        rows += [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]

    sidecar_rows: list[dict] = []
    for f in sorted(Path(out_dir).glob("tdigests_*.jsonl")):
        sidecar_rows += [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]
    # Sequential runs use tdigests.jsonl (no rank suffix).
    plain = Path(out_dir) / "tdigests.jsonl"
    if plain.is_file():
        sidecar_rows += [json.loads(ln) for ln in plain.read_text().splitlines() if ln.strip()]

    attach_tdigest_sidecars(rows, sidecar_rows)
    return rows


def format_summary_lines(summary: dict) -> list[str]:
    """Human-readable closed-loop summary lines (shared by CLI / train print)."""
    n_seg = int(summary["n_segments"])
    obj = summary["object"]
    rb = summary["road_border"]
    red = summary["red_light_violation"]
    brake = summary["strong_brake"]
    repro = summary["reproducer"]
    lines = [
        f"object collision: {obj['collision_segments']}/{n_seg} segments "
        f"(rate {obj['collision_segment_rate']:.4f}), "
        f"{obj['collision_steps']} steps (rate {obj['collision_step_rate']:.6f}), "
        f"{obj['collision_count']} events",
        f"object miss (<= {obj['miss_thresh_m']} m): "
        f"{obj['miss_segments']}/{n_seg} segments "
        f"(rate {obj['miss_segment_rate']:.4f}), {obj['miss_steps']} steps, "
        f"{obj['miss_count']} events",
        f"road_border collision: {rb['collision_segments']}/{n_seg} segments, "
        f"{rb['collision_steps']} steps, {rb['collision_count']} events",
        f"road_border miss (<= {rb['miss_thresh_m']} m): "
        f"{rb['miss_segments']}/{n_seg} segments, {rb['miss_steps']} steps, "
        f"{rb['miss_count']} events",
        f"red_light_violation: {red['segments']}/{n_seg} segments "
        f"(rate {red['segment_rate']:.4f}), {red['steps']} steps, {red['count']} events",
        f"strong-brake (<= {brake['thresh_mps2']} m/s^2): "
        f"{brake['segments']}/{n_seg} segments "
        f"(rate {brake['segment_rate']:.4f}), {brake['steps']} steps, "
        f"{brake['count']} events, strongest_mps2={brake['strongest_mps2']:.3f}",
        f"object clearance min/mean/p5="
        f"{obj['clearance_min_m']:.3f}/{obj['clearance_mean_m']:.3f}/{obj['clearance_p5_m']:.3f} m  "
        f"road_border clearance min/mean/p5="
        f"{rb['clearance_min_m']:.3f}/{rb['clearance_mean_m']:.3f}/{rb['clearance_p5_m']:.3f} m",
        f"reproducer snap_count={repro['snap_count']} expand_count={repro['expand_count']} "
        f"repeat_step_rate={repro['repeat_step_rate']:.4f}  "
        f"terminated={summary['terminated_counts']}",
    ]
    return lines


def aggregate(
    rows: list[dict], near_miss_thresh: float, *, strong_brake_mps2: float = -2.5
) -> dict:
    """Aggregate per-segment nested metric rows into a closed-loop summary."""
    n_seg = len(rows)
    total_steps = sum(int(r["n_steps_run"]) for r in rows)

    # Graded (non-saturating) headline metrics: these improve smoothly as the model trains,
    # unlike the binary *_segment_rate / worst-moment *_min_clearance keys nested below.
    completions = [r["route_completion"] for r in rows if "route_completion" in r]
    # gt-deviation pooled across all steps (weight each segment's mean by its step count, which
    # equals that segment's gt-deviation sample count), so long routes aren't under-weighted.
    dev_num = sum(
        r["mean_gt_deviation_m"] * r["n_steps_run"]
        for r in rows
        if np.isfinite(r.get("mean_gt_deviation_m", float("inf")))
    )
    dev_den = sum(
        r["n_steps_run"] for r in rows if np.isfinite(r.get("mean_gt_deviation_m", float("inf")))
    )

    term_counts: dict[str, int] = {}
    for r in rows:
        term = r["terminated"]
        term_counts[term] = term_counts.get(term, 0) + 1

    obj = _event_family_block(
        rows,
        "object",
        dual=True,
        total_steps=total_steps,
        n_seg=n_seg,
        thresh_key="miss_thresh_m",
        thresh_value=float(near_miss_thresh),
    )
    obj.update(_pool_clearance(rows, "object"))

    rb = _event_family_block(
        rows,
        "road_border",
        dual=True,
        total_steps=total_steps,
        n_seg=n_seg,
        thresh_key="miss_thresh_m",
        thresh_value=float(near_miss_thresh),
    )
    rb.update(_pool_clearance(rows, "road_border"))

    red = _event_family_block(
        rows, "red_light_violation", dual=False, total_steps=total_steps, n_seg=n_seg
    )
    brake = _event_family_block(
        rows,
        "strong_brake",
        dual=False,
        total_steps=total_steps,
        n_seg=n_seg,
        thresh_key="thresh_mps2",
        thresh_value=float(strong_brake_mps2),
    )
    # Strongest consecutive-pair accel across segments (mask-filtered; +inf if none).
    strongest = [float(_require_block(r, "strong_brake")["strongest_mps2"]) for r in rows]
    brake["strongest_mps2"] = min(strongest) if strongest else float("inf")

    expand = sum(int(_require_block(r, "reproducer")["expand_count"]) for r in rows)
    snap = sum(int(_require_block(r, "reproducer")["snap_count"]) for r in rows)
    normal = sum(int(_require_block(r, "reproducer")["normal_steps"]) for r in rows)
    repeat = sum(int(_require_block(r, "reproducer")["repeat_steps"]) for r in rows)

    n_seg_diverged = term_counts.get("diverged", 0)
    return {
        "n_segments": n_seg,
        "total_steps": total_steps,
        "mean_route_completion": float(np.mean(completions)) if completions else 0.0,
        "mean_gt_deviation_m": float(dev_num / dev_den) if dev_den else float("inf"),
        "n_segments_diverged": n_seg_diverged,
        "diverged_segment_rate": n_seg_diverged / n_seg if n_seg else 0.0,
        "object": obj,
        "road_border": rb,
        "red_light_violation": red,
        "strong_brake": brake,
        "terminated_counts": term_counts,
        "reproducer": {
            "expand_count": expand,
            "snap_count": snap,
            "normal_steps": normal,
            "repeat_steps": repeat,
            "repeat_step_rate": repeat / total_steps if total_steps else 0.0,
        },
    }


def build_mp4(png_dir: Path, mp4_path: Path, fps: float) -> None:
    """Encode the PNG sequence in ``png_dir`` to an MP4.

    PNGs are named by step ``k`` and may be sparse (``draw_every`` skips frames), so glob the
    directory (gap-tolerant, name-sorted) instead of a contiguous ``%05d`` counter. ``fps`` is the
    raw frame rate, so a sparse sequence plays faster than real time (shorter video).
    """
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
    replan_interval: int,
    draw_every: int,
    neighbor_history_mode: str,
    unstick_radius_mult: float = 10.0,
    unstick_teleport_after: int = 300,
    tracker_mode: str = "mpc",
    strong_brake_mps2: float = -2.5,
    yaw_gate: bool = True,
    verbose: bool = True,
    shard: tuple[int, int] | None = None,
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
    ``segments.jsonl`` (one row per route, human-readable), ``tdigests.jsonl`` (clearance
    sketches for shard merge), and ``summary.json`` are written into ``out_dir``.

    Turn indicators are CLOSED-LOOP: the model's own predicted turn indicator is fed back into
    the input history each step, held across cached-plan steps when ``replan_interval`` > 1
    (see ``render_segment``).

    ``drop_objects``: empty-world ablation, forwarded to ``render_segment`` (see its docstring).

    Returns the summary dict with extra keys ``video_mp4s`` (list[Path] of every per-route MP4),
    ``segments`` (list[row]), and ``elapsed_sec``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # npz_root is either one directory tree, a JSON path list of route dirs, or an
    # already-resolved list of roots -- enumerate_multi_root_routes merges them all,
    # disambiguating any bag-prefix key that collides across roots and remembering the source
    # root of each route so its pose-sidecar fallback stays scoped to that tree.
    routes, route_sidecar_dir = enumerate_multi_root_routes(npz_root)
    route_keys = sorted(routes)
    if shard is not None:
        rank, world_size = shard
        route_keys = route_keys[rank::world_size]

    timers = Timers()
    rows: list[dict] = []
    video_mp4s: list[Path] = []
    t0 = time.perf_counter()

    segments_name = "segments.jsonl" if shard is None else f"segments_{shard[0]}.jsonl"
    digests_name = "tdigests.jsonl" if shard is None else f"tdigests_{shard[0]}.jsonl"
    fout = open(out_dir / segments_name, "w")
    fdigest = open(out_dir / digests_name, "w")
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
                strong_brake_mps2=strong_brake_mps2,
                yaw_gate=yaw_gate,
                abort_deviation_m=abort_deviation_m,
                abort_after=abort_after,
                abort_max_snaps=abort_max_snaps,
                drop_objects=drop_objects,
            )
            row = {"route": key, **metrics}
            # Human-readable segments.jsonl (no _tdigest blobs). Digests go to a sidecar so
            # multi-GPU parents can still merge approximate global clearance p5.
            fout.write(json.dumps(segment_row_for_json(metrics, route=key), default=float) + "\n")
            fout.flush()
            side = tdigest_sidecar_row(row)
            if side is not None:
                fdigest.write(json.dumps(side, default=float) + "\n")
                fdigest.flush()
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
                obj = metrics["object"]
                brake = metrics["strong_brake"]
                print(
                    f"[{ri + 1}/{len(route_keys)}] {key} -> {seg_mp4.name}  "
                    f"obj_coll={obj['collision_count']} "
                    f"obj_miss={obj['miss_count']} "
                    f"brake={brake['count']} "
                    f"min_clr={obj['clearance_min_m']:.3f}"
                )
    finally:
        fout.close()
        fdigest.close()

    # In-memory ``rows`` still carry digests for this process's aggregate.
    summary = aggregate(rows, near_miss_thresh, strong_brake_mps2=strong_brake_mps2)
    summary["npz_root"] = str(npz_root)
    summary["n_routes"] = len(route_keys)
    summary["elapsed_sec"] = time.perf_counter() - t0
    summary["video_mp4s"] = video_mp4s

    # A sharded worker leaves the merged summary.json to the parent driver (which aggregates every
    # shard's segments_*.jsonl); it only owns its own segments_{rank}.jsonl, written above.
    if shard is None:
        with open(out_dir / "summary.json", "w") as f:
            json.dump(
                {k: v for k, v in summary.items() if k != "video_mp4s"},
                f,
                indent=2,
                default=float,
            )
    return summary


def enumerate_scenarios(scenario_root) -> list[Path]:
    """Resolve a scenario input into an ordered list of .xosc files.

    ``scenario_root`` is either a single ``.xosc`` file or a directory tree
    globbed recursively for ``*.xosc`` (sorted for determinism)."""
    root = Path(scenario_root)
    if root.suffix == ".xosc":
        return [root]
    paths = sorted(root.rglob("*.xosc"))
    if not paths:
        raise FileNotFoundError(f"No .xosc under {scenario_root}")
    return paths


# How long to wait after SIGTERM before SIGKILL. Short on purpose: a worker that ignores
# SIGTERM is already wedged, and the slot is what we are trying to get back.
_KILL_GRACE_SEC = 20.0


def _run_pooled(
    scenarios, out_dir: Path, scenario_root, *, device: str, model_path: str, jobs: int,
    gpu_ids: list[int], threads_per_worker: int, replan_interval: int, max_steps: int,
    warmup_steps: int, near_miss_thresh: float, fps: float, worker_timeout_sec: float,
    verbose: bool,
) -> dict[int, Path]:
    """Run every scenario through ``jobs`` PERSISTENT workers; return ``{index: row_out}``.

    The alternative to a process per scenario. A worker loads the model, compiles it and parses a
    map once and then takes scenarios until they run out, which is only possible because the
    interpreter -- the singleton -- now lives in a child of the worker (``sim_proc_client``). At
    job 1665 the per-scenario fixed cost this removes was 45.2 s of a 90.5 s case.

    Work is claimed through files rather than handed out over a socket: each worker creates
    ``claims/<index>`` with O_EXCL and runs what it wins. Dynamic, so the threefold spread in
    scenario duration does not leave a tail, and there is no queue to drain if a worker dies --
    its unclaimed work is simply still unclaimed.
    """
    import sys

    work_dir = out_dir / "_pool"
    claim_dir = work_dir / "claims"
    claim_dir.mkdir(parents=True, exist_ok=True)
    row_of: dict[int, Path] = {}
    work = []
    for si, scenario in enumerate(scenarios):
        d = out_dir / _scenario_key(scenario, Path(scenario_root))
        row_of[si] = d / "row.json"
        work.append([str(d), str(scenario)])
    work_list = work_dir / "work.json"
    work_list.write_text(json.dumps(work))

    procs = []
    for slot in range(max(1, jobs)):
        env = os.environ.copy()
        if gpu_ids:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[slot % len(gpu_ids)])
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env.setdefault(var, str(threads_per_worker))
        env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
        log = open(work_dir / f"worker{slot:02d}.log", "wb")
        cmd = [
            sys.executable, "-m", "scenario_generation.scenario_sim_pool",
            "--work_list", str(work_list), "--claim_dir", str(claim_dir),
            "--model_path", str(model_path), "--device", device,
            "--replan_interval", str(replan_interval), "--max_steps", str(max_steps),
            "--warmup_steps", str(warmup_steps), "--near_miss_thresh", str(near_miss_thresh),
            "--fps", str(fps), "--watchdog_sec", str(max(60.0, worker_timeout_sec - 120.0)),
        ]
        procs.append((slot, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log))

    if verbose:
        print(f"[scenario_sim] {len(procs)} persistent workers over {len(work)} scenarios")
    t0 = time.perf_counter()
    seen = 0
    while any(p.poll() is None for _, p, _ in procs):
        time.sleep(1.0)
        n = sum(1 for r in row_of.values() if r.exists())
        if verbose and n != seen:
            seen = n
            print(f"[scenario_sim] {n}/{len(work)} rows after {time.perf_counter() - t0:.0f}s",
                  flush=True)
    for slot, p, log in procs:
        log.close()
        if p.returncode != 0:
            print(f"[scenario_sim][WARN] pool worker {slot} exited rc={p.returncode} "
                  f"(see {work_dir}/worker{slot:02d}.log)")
    return row_of


def _scenario_key(scenario: Path, root: Path) -> str:
    """Unique, filesystem-safe id for a scenario within ``root``.

    Stems are NOT unique across a suite: the yaml->xosc converter emits flat
    ``scenario_<n>.xosc`` names inside each scenario's own directory, so the webauto suite has
    84 different ``scenario_0.xosc``. Keying the work directory (and the row's ``route``) on
    the stem alone would put many workers in one directory, overwriting each other's
    ``row.json`` and logs, and would collapse distinct scenarios into one route name.
    """
    root = Path(root)
    if root.is_file():
        return scenario.stem
    return "__".join(scenario.relative_to(root).with_suffix("").parts)


def run_scenario_sim_eval(
    scenario_root,
    map_path,
    out_dir,
    *,
    device: str,
    model_path: str,
    gpus: list[int] | None = None,
    max_cases: int | None = None,
    pooled: bool = False,
    worker_timeout_sec: float = 2400.0,
    near_miss_thresh: float = 1.0,
    replan_interval: int = 1,  # every tick = 10 Hz, matching the production node
    max_steps: int = 300,
    warmup_steps: int = 5,
    fps: float = 10.0,
    jobs: int = 16,
    verbose: bool = True,
) -> dict:
    """Closed-loop eval over OpenSCENARIO ``.xosc`` scenarios (scenario_sim path).

    Sibling of :func:`run_closed_loop_eval`: instead of replaying NPZ frames it
    drives each scenario through the in-process C++ interpreter with the model as
    ego. Reuses :func:`aggregate` and the same ``segments.jsonl`` / ``summary.json``
    output contract so downstream summaries (``scalar_keys`` / wandb) stay
    compatible. One scenario == one whole rollout == one row.

    PROCESS-PER-SCENARIO: the C++ ``SimulatorCore`` is a static singleton
    (1 process = 1 scenario -- reusing it across scenarios leaks nodes and aborts
    at teardown), so each scenario runs in a FRESH subprocess worker
    (``python -m scenario_generation.scenario_sim_rollout``). This also gives
    crash isolation and is the unit of plan/04's process parallelism. Unlike the
    NPZ path this takes a ``model_path`` (torch ``.pth``), not a live model
    object, because each worker builds its own model; the checkpoint the closed-loop
    cadence fires on is exactly what a real worker would load.

    ``jobs`` scenarios run concurrently. The default is the measured knee on an RTX 4090
    with CUDA MPS enabled (167 tick/s at 16 vs 67 at 1); without MPS concurrent contexts
    time-slice and the gain is far smaller, and the knee is device-specific -- re-tune it

    Workers run in the current interpreter's environment, so the ss_v2 overlay has to be on the
    source-chain already (on the production nodes that is the carried-in overlay prefix plus the
    node's own ``/opt/ros/humble`` -- see ``validator/env/setup_production.bash``).
    """
    import sys

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = enumerate_scenarios(scenario_root)
    n_available = len(scenarios)
    # Sampling for measurement runs. Per-stage ms/call needs a full worker pool held for a while,
    # not the whole suite: a few waves of `jobs` cases already give tens of thousands of timed
    # calls at the same contention, and a shared cluster is occupied for correspondingly less
    # time. Evenly spaced rather than the first N, because the sorted order groups by scenario id
    # -- the first N would be a handful of scenarios on a single map.
    sample_stride = 1.0
    if max_cases is not None and 0 < max_cases < n_available:
        sample_stride = n_available / max_cases
        scenarios = [scenarios[int(i * sample_stride)] for i in range(max_cases)]
        if verbose:
            print(
                f"[scenario_sim] SAMPLED {len(scenarios)}/{n_available} cases "
                f"(every ~{sample_stride:.1f}th) -- aggregate metrics are NOT the suite's"
            )
    # `jobs` workers are spread round-robin over these devices. Default: every GPU the job
    # was allocated, which is what makes a full-suite run use the whole node rather than
    # piling all workers onto cuda:0.
    t_start = time.perf_counter()
    # Cores this process may actually use (respects cgroup/affinity, unlike cpu_count()).
    threads_per_worker = max(1, len(os.sched_getaffinity(0)) // max(1, jobs))
    gpu_ids: list[int] = []
    if device.startswith("cuda"):
        if gpus is not None:
            gpu_ids = list(gpus)
        else:
            import torch

            gpu_ids = list(range(torch.cuda.device_count()))
    if verbose:
        print(
            f"[scenario_sim] {len(scenarios)} scenarios, jobs={jobs}, "
            f"gpus={gpu_ids or '(none: ' + device + ')'}, "
            f"threads/worker={threads_per_worker}, worker_timeout={worker_timeout_sec:.0f}s"
        )

    def launch(scenario: Path, slot: int):
        work = out_dir / _scenario_key(scenario, Path(scenario_root))
        work.mkdir(parents=True, exist_ok=True)
        row_out = work / "row.json"
        log = work / "worker.log"
        worker_args = [
            "--osc", str(scenario),
            "--out_dir", str(work),
            "--row_out", str(row_out),
            "--device", device,
            "--replan_interval", str(replan_interval),
            "--max_steps", str(max_steps),
            "--warmup_steps", str(warmup_steps),
            "--near_miss_thresh", str(near_miss_thresh),
            "--fps", str(fps),
            "--model_path", str(model_path),
            # Self-dump before the parent's hard deadline, so a hang leaves a stack even when
            # the process is past responding to signals.
            "--watchdog_sec", str(max(60.0, worker_timeout_sec - 120.0)),
        ]
        # No --map_path unless explicitly overridden: the worker reads the scenario's own
        # RoadNetwork/LogicFile, the same element the C++ interpreter resolves the map from,
        # so a suite spanning several maps cannot silently measure one map's geometry
        # against another's simulation.
        if map_path is not None:
            worker_args += ["--map_path", str(map_path)]
        cmd = [sys.executable, "-m", "scenario_generation.scenario_sim_rollout", *worker_args]
        # Fan out over the allocated GPUs. CUDA_VISIBLE_DEVICES rather than `--device cuda:N`
        # so each worker's "cuda:0" is unambiguous and a stray .cuda() cannot land on a
        # neighbour's device. Copy the parent env -- dropping it would lose
        # TORCHINDUCTOR_CACHE_DIR and give every worker a cold compile.
        env = os.environ.copy()
        if gpu_ids:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[slot % len(gpu_ids)])
        # Thread budget. Unset, torch takes one thread per core (112 on the production node)
        # and inductor compiles with 32 -- times `jobs` workers that is thousands of threads
        # over ~112 cores. The workers ARE the parallelism here; each one wants a slice, not
        # the whole machine. setdefault so an explicit export still wins.
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env.setdefault(var, str(threads_per_worker))
        env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
        # Per-scenario log rather than inherited stdout: with jobs>1 the workers would
        # interleave into an unreadable stream, and a crash needs its own trace.
        handle = open(log, "wb")
        proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env)
        return proc, row_out, log, handle, time.perf_counter()

    if pooled:
        # Persistent workers: the interpreter runs in a child of each worker, so model, compiled
        # graphs and parsed maps are paid per worker instead of per scenario.
        row_of = _run_pooled(
            scenarios, out_dir, scenario_root, device=device, model_path=model_path, jobs=jobs,
            gpu_ids=gpu_ids, threads_per_worker=threads_per_worker,
            replan_interval=replan_interval, max_steps=max_steps, warmup_steps=warmup_steps,
            near_miss_thresh=near_miss_thresh, fps=fps,
            worker_timeout_sec=worker_timeout_sec, verbose=verbose,
        )
        rows = []
        with open(out_dir / "segments.jsonl", "w") as fout:
            for si, scenario in enumerate(scenarios):
                key = _scenario_key(scenario, Path(scenario_root))
                rp = row_of[si]
                if rp.exists():
                    metrics = json.loads(rp.read_text())
                else:
                    err = (rp.parent / "error.txt")
                    metrics = failed_segment_row(
                        err.read_text().splitlines()[0] if err.exists()
                        else f"no row produced, see {rp.parent}",
                        near_miss_thresh,
                    )
                row = segment_row_for_json(metrics, route=key)
                rows.append(row)
                fout.write(json.dumps(row, default=float) + "\n")
        summary = aggregate(rows, near_miss_thresh)
        summary["scenario_root"] = str(scenario_root)
        summary["map_override"] = str(map_path) if map_path is not None else None
        summary["maps_used"] = sorted({m for r in rows if (m := r.get("map_path"))})
        summary["n_scenarios"] = len(scenarios)
        summary["n_scenarios_available"] = n_available
        summary["sampled"] = len(scenarios) < n_available
        summary["sample_stride"] = round(sample_stride, 3)
        summary["jobs"] = jobs
        summary["gpus"] = gpu_ids
        summary["pooled"] = True
        summary["elapsed_sec"] = time.perf_counter() - t_start
        summary["segments"] = rows
        with open(out_dir / "summary.json", "w") as f:
            json.dump(metrics_for_json({k: v for k, v in summary.items() if k != "segments"}),
                      f, indent=4)
        return summary

    rows: list[dict | None] = [None] * len(scenarios)
    pending = list(enumerate(scenarios))
    running: dict[int, tuple] = {}
    # A worker keeps its slot for its whole life and hands it back on exit, so slot -> GPU is
    # stable and the per-GPU load stays even however unevenly the scenarios finish.
    free_slots: list[int] = list(range(max(1, jobs)))
    slot_of: dict[int, int] = {}
    timed_out: dict[int, float] = {}   # scenario index -> when SIGTERM was sent
    done = 0
    t0 = time.perf_counter()
    with open(out_dir / "segments.jsonl", "w") as fout:
        while pending or running:
            while pending and len(running) < max(1, jobs):
                si, scenario = pending.pop(0)
                slot = free_slots.pop()
                slot_of[si] = slot
                running[si] = launch(scenario, slot)
            # Deadline enforcement. One hung worker held a slot for the entire 65-minute run
            # of the full suite -- 0.8% CPU, no GPU, 246 threads in futex_wait_queue_me -- and
            # with no timeout here the job would have sat until its Slurm --time expired. The
            # parent owns the slot, so the parent is what has to reclaim it. SIGTERM is tried
            # first for a clean teardown but is NOT trusted: that worker ignored it and needed
            # SIGKILL (measured), which is why the escalation is unconditional after the grace.
            now = time.perf_counter()
            for si, (proc, _row_out, _log, _handle, t_launch) in list(running.items()):
                if proc.poll() is not None:
                    continue
                over = now - t_launch - worker_timeout_sec
                if over < 0:
                    continue
                if si not in timed_out:
                    timed_out[si] = now
                    print(
                        f"[scenario_sim][WARN] {_scenario_key(scenarios[si], Path(scenario_root))} "
                        f"exceeded {worker_timeout_sec:.0f}s -> SIGTERM (see its worker.log for "
                        f"the watchdog's thread dump)"
                    )
                    proc.terminate()
                elif now - timed_out[si] > _KILL_GRACE_SEC:
                    print(f"[scenario_sim][WARN]   still alive after SIGTERM -> SIGKILL")
                    proc.kill()

            finished = [si for si, (p, *_) in running.items() if p.poll() is not None]
            if not finished:
                time.sleep(0.05)
                continue
            for si in finished:
                proc, row_out, log, handle, t_launch = running.pop(si)
                handle.close()
                slot = slot_of.pop(si)
                free_slots.append(slot)
                worker_wall = time.perf_counter() - t_launch
                # A worker we interrupted is NOT a measurement of its scenario, so its row is
                # discarded even when one exists. It usually does: SIGTERM reaches the C++
                # interpreter, which ends the scenario as `sim_terminated`, and the rollout then
                # finalises and writes a row for the ticks it happened to complete -- a
                # truncated run that is indistinguishable in the aggregate from a scenario that
                # genuinely ended early (measured: rows with steps=1 and steps=175 out of 300).
                if si in timed_out:
                    metrics = failed_segment_row(
                        f"worker exceeded {worker_timeout_sec:.0f}s and was killed "
                        f"(rc={proc.returncode}); any partial row was discarded, see {log}",
                        near_miss_thresh,
                    )
                    metrics["terminated"] = "worker_timeout"
                    status = f"TIMEOUT rc={proc.returncode}"
                # Otherwise the worker's row is authoritative; a nonzero rc at teardown (after
                # the row is written) must not lose a completed rollout, so key off row.json.
                elif row_out.exists():
                    metrics = json.loads(row_out.read_text())
                    status = "ok" if proc.returncode == 0 else f"rc={proc.returncode}(teardown)"
                else:
                    metrics = failed_segment_row(
                        f"worker rc={proc.returncode}, see {log}", near_miss_thresh
                    )
                    status = f"FAILED rc={proc.returncode}"
                row = segment_row_for_json(
                    metrics, route=_scenario_key(scenarios[si], Path(scenario_root))
                )
                # Parent-side timing: worker_wall includes process start, model load and
                # teardown, so `worker_wall_s` minus the worker's own `timing.rollout_total`
                # is the process overhead this design pays 1x per scenario. gpu/slot are
                # recorded so an uneven per-device load is visible after the fact.
                row["worker_wall_s"] = round(worker_wall, 3)
                row["slot"] = slot
                row["gpu"] = gpu_ids[slot % len(gpu_ids)] if gpu_ids else None
                rows[si] = row
                fout.write(json.dumps(row, default=float) + "\n")
                fout.flush()
                done += 1
                if verbose:
                    print(
                        f"[{done}/{len(scenarios)}] {row['route']} -> {status} "
                        f"{metrics.get('terminated')}/{metrics.get('result_kind', '')} "
                        f"steps={metrics.get('n_steps_run')} "
                        f"coll={metrics.get('n_collision_steps')} "
                        f"coord_ok={metrics.get('coord_check_ok')}"
                    )

    # segments.jsonl above is a progress log in COMPLETION order; the returned/aggregated
    # rows are in scenario order so summaries are comparable across jobs settings.
    rows = [r for r in rows if r is not None]
    summary = aggregate(rows, near_miss_thresh)
    summary["scenario_root"] = str(scenario_root)
    # Each worker resolves its own map, so record the set actually used rather than a single
    # path -- for a multi-map suite a scalar `map_path` would be a lie.
    summary["map_override"] = str(map_path) if map_path is not None else None
    summary["maps_used"] = sorted({m for r in rows if (m := r.get("map_path"))})
    summary["n_scenarios"] = len(scenarios)
    # Marked explicitly: a sampled summary must not be read as the suite's result.
    summary["n_scenarios_available"] = n_available
    summary["sampled"] = len(scenarios) < n_available
    summary["sample_stride"] = round(sample_stride, 3)
    summary["jobs"] = jobs
    summary["gpus"] = gpu_ids
    summary["elapsed_sec"] = time.perf_counter() - t0
    # Sum of per-scenario wall over elapsed = achieved concurrency. Below `jobs` means the
    # pool was starved (ramp-up/tail) rather than the devices being the limit.
    _wall = [r["worker_wall_s"] for r in rows if "worker_wall_s" in r]
    summary["worker_wall_sum_s"] = round(sum(_wall), 3)
    summary["mean_concurrency"] = (
        round(sum(_wall) / summary["elapsed_sec"], 2) if summary["elapsed_sec"] > 0 else None
    )
    summary["segments"] = rows
    with open(out_dir / "summary.json", "w") as f:
        json.dump(
            metrics_for_json({k: v for k, v in summary.items() if k != "segments"}),
            f,
            indent=4,
        )
    return summary


def main() -> None:
    """Standalone scenario_sim eval CLI (mirrors valid_predictor_closed_loop.py).

    Runs the scenario_sim closed loop over a scenario root with a torch ``.pth``
    checkpoint, spawning one worker process per scenario, writing segments.jsonl +
    summary.json. Training integration (train.py wiring) is a separate task."""
    import argparse

    p = argparse.ArgumentParser(description="scenario_sim closed-loop eval")
    p.add_argument("--scenario_root", required=True, help=".xosc file or dir tree")
    p.add_argument("--map_path", default=None,
                   help="override the map for EVERY scenario; omit so each worker uses its\nown RoadNetwork/LogicFile (required for a suite spanning several maps)")
    p.add_argument("--gpus", default=None,
                   help="comma-separated GPU indices to fan workers over; default all visible")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--model_path", required=True, help="best_model.pth (torch checkpoint)"
    )
    p.add_argument("--replan_interval", type=int, default=1,
                   help="re-plan every N ticks; 1 (default) = every tick = 10 Hz, matching production")
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--near_miss_thresh", type=float, default=1.0)
    p.add_argument("--jobs", type=int, default=16,
                   help="scenarios to run concurrently on this GPU (tuned for CUDA MPS)")
    p.add_argument("--max_cases", type=int, default=None,
                   help="run an evenly spaced subset of this many cases. For timing runs: "
                        "per-stage ms/call converges long before the suite does, and a subset "
                        "occupies the cluster for correspondingly less time. The summary is "
                        "marked sampled=true -- do not read its metrics as the suite's")
    p.add_argument("--pooled", action="store_true",
                   help="run persistent workers that keep the model, compiled graphs and parsed "
                        "maps across scenarios (requires the interpreter in a child process)")
    p.add_argument("--worker_timeout_sec", type=float, default=2400.0,
                   help="reclaim a worker's slot after this long. Sized against the measured "
                        "p99 of 820 s (max 834 s) at max_steps=1700/jobs=64 -- raise it if you "
                        "raise max_steps or run at higher contention")
    args = p.parse_args()

    summary = run_scenario_sim_eval(
        args.scenario_root,
        args.map_path,
        args.out_dir,
        device=args.device,
        model_path=args.model_path,
        gpus=[int(g) for g in args.gpus.split(",")] if args.gpus else None,
        near_miss_thresh=args.near_miss_thresh,
        replan_interval=args.replan_interval,
        max_steps=args.max_steps,
        jobs=args.jobs,
        max_cases=args.max_cases,
        pooled=args.pooled,
        worker_timeout_sec=args.worker_timeout_sec,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "segments"}, indent=2, default=float))


if __name__ == "__main__":
    main()
