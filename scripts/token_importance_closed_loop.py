"""Closed-loop feature-importance ablation via the Perception-Reproducer rollout.

``token_importance.py`` scores each ``drop:<group>`` input ablation by ONE-STEP
open-loop FDE/ADE against the recorded ground-truth future. That can't see
compounding error: a dropped input might barely move one-step error while still
causing the ego to drift into a collision several seconds later. This tool re-runs
the SAME ablation groups (reusing ``token_importance.py``'s own
``apply_ablation``/``CONFIGS`` unmodified) but scores each one by closed-loop
outcomes — collision/near-miss step counts and minimum clearance — over a short,
continuous, self-driven rollout instead.

Windowing mirrors ``rlvr.autoresearch.tools.mine_direct_reproducer_chunks``'s pattern
(fixed-length chunks at a stride, optional sharding for large corpora) but reuses this
repo's own ``RouteTimeline.iter_segments`` for the windowing/continuity guarantee
instead of reimplementing it — a route is one bag-prefix group of frame-contiguous
NPZs (``group_routes``), so any window drawn from it is contiguous by construction.

For each window, every ablation config is run as its own full closed-loop rollout via
``scenario_generation.reproducer_rollout_token_importance.run_segment`` — v1 favors
correctness/simplicity over throughput (sequential per (window, config) pair); switching
the inner loop to the batched ``run_segments_batched`` (same ``ablate_fn`` hook, added at
its own ``model(data)`` call site) is a natural v2 throughput upgrade once this is
validated.

Example::

  uv run python scripts/token_importance_closed_loop.py \
    --model_path /path/to/best_model.pth --npz_root /path/to/npz_root \
    --chunk_len 80 --start_stride 80 --device cuda --out_tsv closed_loop_ablation.tsv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from scenario_generation.reproducer_rollout_token_importance import run_segment
from scenario_generation.route_timeline import RouteTimeline, group_routes
from scenario_generation.simulate import load_model
from token_importance import CONFIGS, apply_ablation

# Only the class-level drop configs make sense for a closed-loop rollout (a window is
# driven once per config, not resampled per-batch like the open-loop Top-K/distance-cutoff
# sweeps) — start with these; Top-K/within sweeps are a documented follow-up, not v1.
DROP_CONFIGS = [name for name in CONFIGS if name == "baseline" or name.startswith("drop:")]

METRIC_KEYS = (
    "n_collision_steps",
    "n_near_miss_steps",
    "min_clearance",
    "mean_clearance",
    "progress_m",
    "n_snaps",
)
# Reported alongside METRIC_KEYS but not delta'd: a window can legitimately execute zero
# steps (the reproducer's per-window goal is the recorded ego's own end-of-window position,
# so a near-stationary recorded stretch reaches "goal" at k=0) — n_steps_run/terminated make
# that visible instead of silently producing a misleadingly "clean" all-zero/inf row.
DIAGNOSTIC_KEYS = ("n_steps_run", "terminated")


def _flat_metrics(result: dict) -> dict:
    """Flatten ``reproducer_rollout_token_importance.run_segment``'s (nested) metrics dict
    into the flat column names this script's TSV has always used.
    """
    obj = result["object"]
    return {
        "n_steps_run": result["n_steps_run"],
        "terminated": result["terminated"],
        "n_collision_steps": obj["collision_steps"],
        "n_near_miss_steps": obj["miss_steps"],
        "min_clearance": obj["clearance_min_m"],
        "mean_clearance": obj["clearance_mean_m"],
        "progress_m": result["progress_m"],
        "n_snaps": result["reproducer"]["snap_count"],
    }


def iter_windows(npz_root: str, sidecar_root: str | None, chunk_len: int, start_stride: int):
    """Yield (route_key, RouteTimeline, start, end) windows across every route under npz_root."""
    paths = sorted(Path(npz_root).rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no .npz under {npz_root}")
    routes = group_routes(paths)
    for route_key in sorted(routes):
        tl = RouteTimeline(routes[route_key], sidecar_dir=sidecar_root)
        for start, end in tl.iter_segments(chunk_len, start_stride):
            yield route_key, tl, start, end


def shard_windows(windows: list, num_shards: int, shard_index: int, sample_fraction: float) -> list:
    if sample_fraction < 1.0:
        step = max(1, round(1.0 / sample_fraction))
        windows = windows[::step]
    if num_shards > 1:
        windows = windows[shard_index::num_shards]
    return windows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", required=True)
    p.add_argument("--npz_root", required=True)
    p.add_argument("--sidecar_root", default=None)
    p.add_argument("--chunk_len", type=int, default=80, help="frames per window (~8s @10Hz)")
    p.add_argument("--start_stride", type=int, default=80, help="frames between window starts")
    p.add_argument("--min_chunk_len", type=int, default=20, help="drop windows shorter than this")
    p.add_argument(
        "--configs",
        default=None,
        help="comma-separated config names to run (default: baseline + all drop:* groups)",
    )
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--sample_fraction", type=float, default=1.0, help="randomly thin windows")
    p.add_argument("--max_windows", type=int, default=-1, help="limit windows (debug)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--near_miss_thresh", type=float, default=0.5)
    p.add_argument("--search_radius", type=float, default=1.5)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--unstick_after", type=int, default=300)
    p.add_argument("--out_tsv", default="token_importance_closed_loop.tsv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_index < max(args.num_shards, 1):
        raise ValueError("--shard_index must be in [0, num_shards)")

    configs = args.configs.split(",") if args.configs else DROP_CONFIGS
    if "baseline" not in configs:
        configs = ["baseline", *configs]  # baseline must run first for the delta columns

    windows = list(iter_windows(args.npz_root, args.sidecar_root, args.chunk_len, args.start_stride))
    windows = [(r, tl, s, e) for r, tl, s, e in windows if e - s >= args.min_chunk_len]
    windows = shard_windows(windows, args.num_shards, args.shard_index, args.sample_fraction)
    if args.max_windows > 0:
        windows = windows[: args.max_windows]
    print(f"{len(windows)} windows, {len(configs)} configs each", flush=True)

    model, model_args = load_model(Path(args.model_path), args.device)

    out_tsv = Path(args.out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "route",
        "window_start",
        "window_end",
        "config",
        *DIAGNOSTIC_KEYS,
        *METRIC_KEYS,
        *(f"d_{k}" for k in METRIC_KEYS),
    ]
    fout = out_tsv.open("w")
    fout.write("\t".join(cols) + "\n")

    n_rows = 0
    t0 = time.time()
    for wi, (route_key, tl, start, end) in enumerate(windows):
        baseline_metrics = None
        for name in configs:
            ablate_fn = None if name == "baseline" else (lambda data, _name=name: apply_ablation(data, _name))
            t_step = time.time()
            result = run_segment(
                model,
                model_args,
                tl,
                start,
                end,
                device=args.device,
                near_miss_thresh=args.near_miss_thresh,
                search_radius=args.search_radius,
                warmup_steps=args.warmup_steps,
                unstick_after=args.unstick_after,
                ablate_fn=ablate_fn,
            )
            m = _flat_metrics(result)
            if name == "baseline":
                baseline_metrics = m
            row = {
                "route": route_key,
                "window_start": start,
                "window_end": end,
                "config": name,
                **{k: m[k] for k in DIAGNOSTIC_KEYS},
                **{k: m[k] for k in METRIC_KEYS},
                **{f"d_{k}": m[k] - baseline_metrics[k] for k in METRIC_KEYS},
            }
            fout.write("\t".join(str(row[c]) for c in cols) + "\n")
            fout.flush()
            n_rows += 1
            print(
                f"[{wi + 1}/{len(windows)}] {route_key} [{start},{end}) {name:<22} "
                f"steps={m['n_steps_run']:3d}/{end - start} term={m['terminated']:<9} "
                f"collisions={m['n_collision_steps']:3d} near_miss={m['n_near_miss_steps']:3d} "
                f"min_clr={m['min_clearance']:.2f}  [{time.time() - t_step:.1f}s]",
                flush=True,
            )
    fout.close()
    print(f"\nwrote {n_rows} rows -> {out_tsv} ({time.time() - t0:.1f}s total)")


if __name__ == "__main__":
    main()
