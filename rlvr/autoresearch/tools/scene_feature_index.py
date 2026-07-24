"""Headless, shardable per-scene feature extractor over an NPZ ``path_list.json``.

Emits a **sharded parquet** feature table (one parquet file per shard) that
``build_small_dataset.py`` consumes for stratified sampling. Designed to scale to
tens of millions of NPZs:

* the path list is streamed in fixed-size shards (never fully materialized as rows),
* each shard is processed by a ``ProcessPoolExecutor`` and written to its own
  ``shard_NNNNN.parquet``,
* per-shard resume: a shard whose parquet already exists is skipped,
* optional ``--prefilter_frac`` seeded random pre-subsample of the path list.

Feature bodies are REUSED from the canonical implementations (never reimplemented):
``compute_gt_stats`` (curate_curve_scenes), ``lat_accel_curvature``
(eval_driving_metrics), ``read_sidecar`` (search_scenes), ``future_to_4col``
(planner_metrics.scene_format). The three features that live inline inside
``scene_search`` constraint ``.filter()`` methods (ego speed, GT travel distance,
neighbor radius count) have no importable callable, so the exact documented formula
is replicated with a ``file:line`` citation.

Usage:
    python -m rlvr.autoresearch.tools.scene_feature_index \
        --path_list <path_list.json> --out_dir <parquet_dir> \
        --shard_size 5000 --workers 8 [--prefilter_frac 0.1] [--seed 0]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# Repo root = <root>/rlvr/autoresearch/tools/scene_feature_index.py -> up 4.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _load_util_script(module_name: str, rel_path: str):
    """Load a ``diffusion_planner/util_scripts/*.py`` helper by file path.

    These files are NOT importable as a package: the editable install maps
    ``diffusion_planner`` to the nested ``diffusion_planner/diffusion_planner/``
    package, so ``diffusion_planner.util_scripts`` (a sibling of that package) has
    no module path. The scripts are self-contained (stdlib + numpy + tqdm), so we
    load the canonical file directly to reuse its function objects rather than
    copy them."""
    path = os.path.join(_REPO_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {rel_path} at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- feature-binning thresholds (declared argparse defaults, NOT silent fallbacks) ---
# Speed bins (m/s). is_stopped uses the <0.5 m/s test from eval_driving_metrics.py:163.
DEFAULT_SPEED_BINS = (0.5, 3.0, 8.0)  # stopped | low | mid | high
# Maneuver: signed total GT yaw (deg). |yaw| < STRAIGHT => straight; sign => turn L/R.
DEFAULT_STRAIGHT_YAW_DEG = 15.0
# Interaction: dense if neighbor count within radius exceeds this.
DEFAULT_DENSE_COUNT = 6
DEFAULT_NEIGHBOR_RADIUS_M = 30.0
# Front-cone lead vehicle: neighbor ahead within [0, range] and |lateral| <= half width.
DEFAULT_LEAD_RANGE_M = 40.0
DEFAULT_LEAD_HALF_WIDTH_M = 2.0

_FRAME_RE = re.compile(r"_(\d+)\.npz$")

# --- parquet column order (kept stable so shards concat cleanly) ---
FEATURE_COLUMNS = [
    "npz_path",
    "bag",
    "frame",
    "ego_speed_ms",
    "is_stopped",
    "travel_dist_m",
    "gt_yaw_deg",  # signed total GT yaw
    "gt_path_m",
    "peak_lat_accel",  # max |speed*omega| over ego future
    "peak_abs_yaw_deg",  # unsigned total yaw magnitude (from compute_gt_stats)
    "neighbor_count",  # active neighbors within radius
    "nearest_neighbor_m",
    "has_lead",
    "lead_dist_m",
    "has_pedestrian",
    "static_count",
    "goal_dist_m",
    "goal_heading_delta_deg",
    "world_x",  # nullable (sidecar)
    "world_y",
    "world_heading_deg",
    "speed_bin",  # stopped|low|mid|high
    "maneuver",  # straight|turn-L|turn-R
    "interaction",  # has-lead|dense|has-ped|has-static|none
    "label",  # "<speed_bin>|<maneuver>|<interaction>" (bucket triple)
]


def _bag_and_frame(npz_path: str) -> tuple[str, int]:
    """bag prefix + frame index from filename.

    Mirrors search_scenes ``_bag_prefix`` (``rsplit('_',1)[0]``) + frame regex
    ``_(\\d+)\\.npz$`` (search_scenes.py:317-ff).
    """
    base = os.path.basename(npz_path)
    m = _FRAME_RE.search(base)
    frame = int(m.group(1)) if m else -1
    bag = base.rsplit("_", 1)[0] if "_" in base else base[:-4]
    return bag, frame


def _signed_total_yaw_deg(yaw: np.ndarray) -> float:
    """Signed cumulative GT heading change (deg), wrapped per-step.

    Same per-step wrap as curate_curve_scenes.compute_gt_stats (lines 40-42) but
    keeps the sign so we can classify turn direction; the unsigned magnitude comes
    from compute_gt_stats directly.
    """
    dh = np.diff(yaw)
    dh = np.arctan2(np.sin(dh), np.cos(dh))
    return float(np.degrees(dh.sum()))


def extract_features(
    npz_path: str,
    *,
    speed_bins: tuple[float, float, float],
    straight_yaw_deg: float,
    dense_count: int,
    neighbor_radius_m: float,
    lead_range_m: float,
    lead_half_width_m: float,
) -> dict | None:
    """Compute the per-scene feature row. Returns None on a load/parse failure
    (the caller logs and tallies — one corrupt NPZ must not kill a shard)."""
    # Reuse the canonical bodies. Imported lazily so a worker only pays the cost
    # once, on first NPZ, and only in worker processes.
    from planner_metrics.scene_format import future_to_4col
    from rlvr.autoresearch.tools.curate_curve_scenes import compute_gt_stats
    from rlvr.autoresearch.tools.eval_driving_metrics import lat_accel_curvature

    read_sidecar = _load_util_script(
        "search_scenes", "diffusion_planner/util_scripts/search_scenes.py"
    ).read_sidecar

    d = np.load(npz_path, allow_pickle=True)
    bag, frame = _bag_and_frame(npz_path)
    row: dict = {c: None for c in FEATURE_COLUMNS}
    row["npz_path"] = npz_path
    row["bag"] = bag
    row["frame"] = frame

    # --- ego speed / is_stopped ---
    # speed_range.py:37-42: ego_current_state = [x,y,cos,sin,vx,vy,ax,ay,steer,yaw_rate]
    ego = np.asarray(d["ego_current_state"], dtype=np.float32).reshape(-1)
    vx, vy = float(ego[4]), float(ego[5])
    speed_ms = math.hypot(vx, vy)
    row["ego_speed_ms"] = speed_ms
    row["is_stopped"] = bool(speed_ms < 0.5)  # eval_driving_metrics.py:163

    fut = np.asarray(d["ego_agent_future"], dtype=np.float32)  # (80,3) [x,y,yaw] legacy

    # --- GT travel distance --- travel_distance.py:37-40
    dx = np.diff(fut[:, 0])
    dy = np.diff(fut[:, 1])
    row["travel_dist_m"] = float(np.sqrt(dx * dx + dy * dy).sum())

    # --- total GT yaw + path length (REUSE compute_gt_stats) ---
    stats = compute_gt_stats(npz_path)
    if stats is not None:
        row["peak_abs_yaw_deg"] = float(stats["yaw_deg"])
        row["gt_path_m"] = float(stats["path_m"])
    row["gt_yaw_deg"] = _signed_total_yaw_deg(fut[:, 2])

    # --- peak curvature / lateral accel (REUSE lat_accel_curvature) ---
    # Needs [T,4] [x,y,cos,sin]; ego future is a single trajectory at origin ->
    # zero_rows_are_padding=False so the t=0 origin row is not zeroed.
    ego_fut_4 = future_to_4col(fut, zero_rows_are_padding=False)
    la = lat_accel_curvature(ego_fut_4)
    row["peak_lat_accel"] = float(np.max(np.abs(la))) if la.size else 0.0

    # --- neighbors --- neighbor_count.py:45-54 (formula replicated: no callable)
    nap = np.asarray(d["neighbor_agents_past"], dtype=np.float32)  # (N,T,11)
    cur = nap[:, -1, :]  # last timestep (N,11)
    active = np.any(cur != 0, axis=-1)
    pos = cur[active, :2]
    if pos.shape[0]:
        dists = np.linalg.norm(pos, axis=-1)
        row["neighbor_count"] = int(np.sum(dists <= neighbor_radius_m))
        row["nearest_neighbor_m"] = float(dists.min())
        # front-cone lead vehicle (ego frame: +x forward, y lateral)
        fx = pos[:, 0]
        fy = pos[:, 1]
        lead_mask = (fx > 0) & (fx <= lead_range_m) & (np.abs(fy) <= lead_half_width_m)
        if np.any(lead_mask):
            row["has_lead"] = True
            row["lead_dist_m"] = float(fx[lead_mask].min())
        else:
            row["has_lead"] = False
        # pedestrian present: col 9 is_ped one-hot (npz_loader.py:208 [...,8,9,10])
        row["has_pedestrian"] = bool(np.any(cur[active, 9] == 1))
    else:
        row["neighbor_count"] = 0
        row["nearest_neighbor_m"] = None
        row["has_lead"] = False
        row["has_pedestrian"] = False

    # --- static objects ---
    static = (
        np.asarray(d.get("static_objects"), dtype=np.float32)
        if "static_objects" in d.files
        else None
    )
    row["static_count"] = (
        int(np.sum(np.any(static != 0, axis=-1))) if static is not None and static.size else 0
    )

    # --- goal ---
    goal = np.asarray(d["goal_pose"], dtype=np.float32).reshape(-1)  # [x,y,heading] ego frame
    row["goal_dist_m"] = float(math.hypot(goal[0], goal[1]))
    row["goal_heading_delta_deg"] = float(
        abs(math.degrees(math.atan2(math.sin(goal[2]), math.cos(goal[2]))))
    )

    # --- world pose from sidecar (nullable) ---
    side = read_sidecar(npz_path)
    if side is not None:
        row["world_x"] = float(side["x"])
        row["world_y"] = float(side["y"])
        row["world_heading_deg"] = float(side["heading_deg"])

    # --- derived labels (bucket triple) ---
    lo, mid, hi = speed_bins
    if speed_ms < lo:
        sbin = "stopped"
    elif speed_ms < mid:
        sbin = "low"
    elif speed_ms < hi:
        sbin = "mid"
    else:
        sbin = "high"
    row["speed_bin"] = sbin

    sy = row["gt_yaw_deg"]
    if abs(sy) < straight_yaw_deg:
        row["maneuver"] = "straight"
    elif sy > 0:
        row["maneuver"] = "turn-L"
    else:
        row["maneuver"] = "turn-R"

    # interaction priority: has-lead > dense > has-ped > has-static > none
    if row["has_lead"]:
        inter = "has-lead"
    elif row["neighbor_count"] >= dense_count:
        inter = "dense"
    elif row["has_pedestrian"]:
        inter = "has-ped"
    elif row["static_count"] > 0:
        inter = "has-static"
    else:
        inter = "none"
    row["interaction"] = inter

    row["label"] = f"{row['speed_bin']}|{row['maneuver']}|{row['interaction']}"
    return row


# Module-level worker wrapper so ProcessPoolExecutor can pickle it.
_WORKER_KW: dict = {}


def _worker(npz_path: str):
    try:
        return extract_features(npz_path, **_WORKER_KW)
    except Exception as e:  # per-NPZ error: log via return sentinel, don't crash shard
        return {"__error__": f"{npz_path}: {type(e).__name__}: {e}"}


def _init_worker(kw: dict) -> None:
    _WORKER_KW.clear()
    _WORKER_KW.update(kw)


def _write_parquet(rows: list[dict], path: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols = {c: [r.get(c) for r in rows] for c in FEATURE_COLUMNS}
    table = pa.table(cols)
    pq.write_table(table, path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--path_list", required=True, help="path_list.json (list of NPZ paths)")
    ap.add_argument("--out_dir", required=True, help="output dir for shard_NNNNN.parquet")
    ap.add_argument("--shard_size", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument(
        "--prefilter_frac",
        type=float,
        default=None,
        help="random pre-subsample fraction of the path list (0<frac<=1)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="cap total NPZs (testing); 0=all")
    ap.add_argument(
        "--overwrite", action="store_true", help="rewrite shards even if parquet exists"
    )
    ap.add_argument("--speed_bins", type=float, nargs=3, default=list(DEFAULT_SPEED_BINS))
    ap.add_argument("--straight_yaw_deg", type=float, default=DEFAULT_STRAIGHT_YAW_DEG)
    ap.add_argument("--dense_count", type=int, default=DEFAULT_DENSE_COUNT)
    ap.add_argument("--neighbor_radius_m", type=float, default=DEFAULT_NEIGHBOR_RADIUS_M)
    ap.add_argument("--lead_range_m", type=float, default=DEFAULT_LEAD_RANGE_M)
    ap.add_argument("--lead_half_width_m", type=float, default=DEFAULT_LEAD_HALF_WIDTH_M)
    args = ap.parse_args()

    if args.prefilter_frac is not None and not (0.0 < args.prefilter_frac <= 1.0):
        raise ValueError(f"--prefilter_frac must be in (0,1], got {args.prefilter_frac}")

    with open(args.path_list) as f:
        paths = json.load(f)
    if not isinstance(paths, list):
        raise ValueError(f"{args.path_list} must be a JSON list of NPZ paths")
    n_total = len(paths)

    if args.prefilter_frac is not None:
        rng = random.Random(args.seed)
        k = max(1, int(round(args.prefilter_frac * n_total)))
        paths = rng.sample(paths, k)
    if args.limit:
        paths = paths[: args.limit]
    print(f"[index] {n_total} paths in list -> {len(paths)} after prefilter/limit", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    worker_kw = dict(
        speed_bins=tuple(args.speed_bins),
        straight_yaw_deg=args.straight_yaw_deg,
        dense_count=args.dense_count,
        neighbor_radius_m=args.neighbor_radius_m,
        lead_range_m=args.lead_range_m,
        lead_half_width_m=args.lead_half_width_m,
    )

    n_shards = math.ceil(len(paths) / args.shard_size)
    err_log = os.path.join(args.out_dir, "errors.log")
    total_ok = 0
    total_err = 0
    for si in range(n_shards):
        shard_paths = paths[si * args.shard_size : (si + 1) * args.shard_size]
        out_pq = os.path.join(args.out_dir, f"shard_{si:05d}.parquet")
        if os.path.exists(out_pq) and not args.overwrite:
            print(f"[index] shard {si + 1}/{n_shards}: exists, skip (resume)", flush=True)
            continue
        t0 = time.time()
        rows: list[dict] = []
        errs = 0
        with ProcessPoolExecutor(
            max_workers=args.workers, initializer=_init_worker, initargs=(worker_kw,)
        ) as ex:
            futs = [ex.submit(_worker, p) for p in shard_paths]
            for fu in as_completed(futs):
                r = fu.result()
                if r is None:
                    errs += 1
                elif "__error__" in r:
                    errs += 1
                    with open(err_log, "a") as ef:
                        ef.write(r["__error__"] + "\n")
                else:
                    rows.append(r)
        # atomic write: temp then rename
        tmp = out_pq + ".tmp"
        _write_parquet(rows, tmp)
        os.replace(tmp, out_pq)
        total_ok += len(rows)
        total_err += errs
        dt = time.time() - t0
        rate = len(shard_paths) / dt if dt > 0 else 0.0
        print(
            f"[index] shard {si + 1}/{n_shards}: {len(rows)} rows, {errs} err, "
            f"{dt:.1f}s ({rate:.1f} npz/s) -> {os.path.basename(out_pq)}",
            flush=True,
        )

    print(
        f"[index] DONE: {total_ok} rows, {total_err} errors across {n_shards} shards -> {args.out_dir}",
        flush=True,
    )
    if total_err:
        print(f"[index] see {err_log}", flush=True)


if __name__ == "__main__":
    main()
