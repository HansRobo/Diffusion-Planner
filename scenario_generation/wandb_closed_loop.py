"""Build wandb log payloads for grouped closed-loop validation."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import wandb

from scenario_generation.metrics.group_report import WANDB_EXCLUDED_SCALAR_KEYS
from scenario_generation.trajectory_colormap import METRIC_CHOICES, render_trajectory_colormaps

# Full-route summary keys aggregated across sites by build_sites_aggregate_log — the
# "rate" pairs each have a numerator ("n_segments_with_collision" etc.) used for the
# micro (pooled) aggregate, and the rate itself used for the macro (per-site mean) one.
_SITE_RATE_FIELDS = (
    ("collision_segment_rate", "n_segments_with_collision"),
    ("near_miss_segment_rate", "n_segments_with_near_miss"),
    ("diverged_segment_rate", "n_segments_diverged"),
)

RESULTS_TABLE_COLUMNS = [
    "area_name",
    "bag",
    "span_index",
    "segment",
    "n_steps_run",
    "centerline_mean_m",
    "centerline_p95_m",
    "turn_match_rate",
    "neighbor_violation_steps",
    "rb_violation_steps",
    "collision_steps",
    "n_collision_steps",
    "n_near_miss_steps",
    "min_clearance_m",
    "mean_clearance",
    "video_path",
]


def build_grouped_closed_loop_wandb_log(summary: dict) -> dict:
    """Per-area scalars, validation totals, results table, and episode videos."""
    log: dict = {"closed_loop/mode": "grouped"}
    log["closed_loop/grouped/elapsed_sec"] = float(summary.get("elapsed_sec", 0.0))

    timing = summary.get("timing")
    if timing:
        stages = timing.get("stages") or {}
        for key, wandb_key in (
            ("model_forward", "model_forward_total_s"),
            ("draw", "draw_total_s"),
            ("timeline_load_npz", "timeline_load_npz_total_s"),
        ):
            stage = stages.get(key)
            if stage:
                log[f"closed_loop/profile/{wandb_key}"] = float(stage["total_s"])
        log["closed_loop/profile/model_forward_calls"] = int(timing.get("model_forward_calls", 0))
        log["closed_loop/profile/total_sim_steps"] = int(timing.get("total_sim_steps", 0))
        log["closed_loop/profile/model_forward_rate"] = float(timing.get("model_forward_rate", 0.0))

    agg = summary.get("grouped_summary") or {}
    _log_scalar_group(log, "closed_loop/grouped/total", agg.get("totals") or {})

    for area, stats in (agg.get("by_area_name") or {}).items():
        safe_area = area.replace("/", "_")
        _log_scalar_group(log, f"closed_loop/grouped/area/{safe_area}", stats)

    rows = summary.get("segments") or []
    if rows:
        df = pd.DataFrame(rows)
        cols = [c for c in RESULTS_TABLE_COLUMNS if c in df.columns]
        extra = [
            c for c in df.columns if c not in cols and c not in ("labeled_ranges", "metric_group")
        ]
        log["closed_loop/grouped/results_table"] = wandb.Table(dataframe=df[cols + extra])

    for mp4 in summary.get("video_mp4s") or []:
        mp4_path = Path(mp4)
        key = _video_wandb_key(mp4_path)
        log[key] = wandb.Video(str(mp4_path), format="mp4")

    return log


def _log_scalar_group(log: dict, prefix: str, stats: dict) -> None:
    for key, val in stats.items():
        if key in WANDB_EXCLUDED_SCALAR_KEYS:
            continue
        if not _wandb_scalar(val):
            continue
        log[f"{prefix}/{key}"] = val


def _segment_paths(out_dir: str | Path, row: dict) -> tuple[Path, Path]:
    """(png_dir, mp4_path) for one segments.jsonl row, matching FullRouteClosedLoopEvaluation's
    naming (``run_job``): ``{out_dir}/{route}_{start}_{end}`` (dir) and ``...mp4``."""
    start, end = row["segment"]
    stem = f"{row['route']}_{start}_{end}"
    out_dir = Path(out_dir)
    return out_dir / stem, out_dir / f"{stem}.mp4"


def pick_representative_row(rows: list[dict], mode: str = "worst") -> dict | None:
    """Pick one segment row to represent a site's whole run (for the 1 video/image W&B keeps).

    ``mode``: ``"worst"`` (default) = most collision steps, tie-broken by smallest
    min_clearance — the case most worth a human's attention. ``"first"`` = first
    discovered route/segment (stable, arbitrary). ``"longest"`` = most steps run.
    """
    if not rows:
        return None
    if mode == "first":
        return rows[0]
    if mode == "longest":
        return max(rows, key=lambda r: r.get("n_steps_run", 0))

    def _worst_key(r: dict) -> tuple[int, float]:
        coll = r.get("n_collision_steps", 0)
        cl = r.get("min_clearance", float("inf"))
        cl = cl if math.isfinite(cl) else 1e9
        return (coll, -cl)  # more collisions first, then smaller clearance first

    return max(rows, key=_worst_key)


EPISODE_TABLE_COLUMNS = [
    "site",
    "route",
    "segment",
    "n_steps_run",
    "terminated",
    "min_clearance",
    "mean_clearance",
    "n_collision_steps",
    "n_near_miss_steps",
    "progress_m",
    "n_snaps",
    "video_path",
]


def build_episode_table(
    rows: list[dict],
    *,
    out_dir: str | Path | None = None,
    site: str | None = None,
) -> wandb.Table:
    """Numeric per-episode table for W&B (sort/filter/group by column in the W&B UI).

    Deliberately carries no video/image data (see module docstring) — ``video_path`` is a
    plain string column pointing at where the full rendering lives (the run's out_dir on
    whichever machine produced it), for a human to open locally.
    """
    table = wandb.Table(columns=EPISODE_TABLE_COLUMNS)
    for r in rows:
        seg = r.get("segment")
        seg_str = f"[{seg[0]},{seg[1]}]" if seg else ""
        video_path = ""
        if out_dir is not None:
            video_path = str(_segment_paths(out_dir, r)[1])
        min_cl = r.get("min_clearance")
        mean_cl = r.get("mean_clearance")
        table.add_data(
            site or "",
            r.get("route", ""),
            seg_str,
            int(r.get("n_steps_run", 0)),
            r.get("terminated", ""),
            float(min_cl) if min_cl is not None and math.isfinite(min_cl) else None,
            float(mean_cl) if mean_cl is not None and math.isfinite(mean_cl) else None,
            int(r.get("n_collision_steps", 0)),
            int(r.get("n_near_miss_steps", 0)),
            float(r.get("progress_m", 0.0)),
            int(r.get("n_snaps", 0)),
            video_path,
        )
    return table


def resolve_report_link(out_dir: str | Path, report_base_url: str | None = None) -> str:
    """Where the rich local report (all videos + HTML gallery) for this run lives.

    Returns a clickable ``http(s)://...`` URL if ``report_base_url`` is set (the run's
    ``out_dir`` is being served over HTTP from that base, e.g. on a training server), else
    the plain local filesystem path (informational only — W&B's web UI cannot open
    ``file://`` links, so on a local dev machine this is for the human to copy/open by hand).
    """
    out_dir = Path(out_dir)
    if report_base_url:
        return report_base_url.rstrip("/") + "/" + out_dir.name
    return str(out_dir.resolve())


def build_sites_aggregate_log(summaries: dict[str, dict]) -> dict:
    """Cross-site aggregate scalars from each site's full-route summary dict.

    Two aggregates, both logged (they answer different questions):
    - ``closed_loop/_all_micro/*``: pooled across all segments (``Σ.../Σ...``) — weighted by
      how much data each site has; answers "what fraction of all exposure was bad".
    - ``closed_loop/_all_macro/*``: simple mean of each site's own rate — every site counts
      equally regardless of size; answers "how consistently safe is the model across sites".
    """
    log: dict = {}
    if not summaries:
        return log
    values = list(summaries.values())
    n_sites = len(values)
    total_segments = sum(int(s.get("n_segments", 0)) for s in values)
    total_steps = sum(int(s.get("total_steps", 0)) for s in values)

    log["closed_loop/_all_micro/n_sites"] = n_sites
    log["closed_loop/_all_micro/n_segments"] = total_segments
    log["closed_loop/_all_macro/n_sites"] = n_sites

    for rate_key, count_key in _SITE_RATE_FIELDS:
        total_count = sum(int(s.get(count_key, 0)) for s in values)
        log[f"closed_loop/_all_micro/{rate_key}"] = (
            total_count / total_segments if total_segments else 0.0
        )
        log[f"closed_loop/_all_macro/{rate_key}"] = (
            sum(float(s.get(rate_key, 0.0)) for s in values) / n_sites if n_sites else 0.0
        )

    total_collision_steps = sum(int(s.get("total_collision_steps", 0)) for s in values)
    log["closed_loop/_all_micro/collision_step_rate"] = (
        total_collision_steps / total_steps if total_steps else 0.0
    )
    finite_min_cl = [
        float(s["global_min_clearance"])
        for s in values
        if math.isfinite(s.get("global_min_clearance", float("inf")))
    ]
    log["closed_loop/_all_micro/global_min_clearance"] = (
        min(finite_min_cl) if finite_min_cl else float("inf")
    )
    return {k: v for k, v in log.items() if _wandb_scalar(v) or isinstance(v, int)}


def build_full_closed_loop_wandb_log(
    summary: dict,
    *,
    out_dir: str | Path | None = None,
    site: str | None = None,
    video_pick: str = "worst",
    colormap_metrics: tuple[str, ...] = METRIC_CHOICES,
    near_miss_thresh: float = 0.5,
    report_base_url: str | None = None,
) -> dict:
    """Full-route closed-loop wandb payload: scalars, one representative video + one
    trajectory-colormap image per metric in ``colormap_metrics`` (not all routes — see module
    docstring; still just ONE episode, so uploading every metric for it stays cheap), an
    episode table (numeric only, for W&B-side sort/filter), and a link to where the full
    report (all videos + HTML gallery) lives.
    """
    scalar_keys = [
        "collision_segment_rate",
        "collision_step_rate",
        "near_miss_segment_rate",
        "near_miss_step_rate",
        "diverged_segment_rate",
        "global_min_clearance",
        "mean_segment_min_clearance",
        "mean_segment_mean_clearance",
        "total_collision_steps",
        "total_near_miss_steps",
        "total_snaps",
        "total_steps",
        "n_segments",
        "n_segments_diverged",
    ]
    log = {"closed_loop/mode": "full"}
    for key in scalar_keys:
        val = summary.get(key)
        if _wandb_scalar(val):
            log[f"closed_loop/{key}"] = val

    rows = summary.get("segments") or []
    effective_out_dir = out_dir if out_dir is not None else summary.get("npz_root")
    rep = pick_representative_row(rows, mode=video_pick)
    if rep is not None and out_dir is not None:
        png_dir, mp4_path = _segment_paths(out_dir, rep)
        if mp4_path.is_file():
            log[f"closed_loop/video/{mp4_path.stem}"] = wandb.Video(str(mp4_path), format="mp4")
        try:
            rendered = render_trajectory_colormaps(
                png_dir,
                out_dir,
                mp4_path.stem,
                metrics=colormap_metrics,
                near_miss_thresh=near_miss_thresh,
                title=f"{site or ''} {mp4_path.stem}".strip(),
            )
        except Exception as e:  # pragma: no cover - rendering must never break training
            print(f"closed_loop: trajectory colormap failed for {mp4_path.stem}: {e}")
            rendered = {}
        for metric, png_path in rendered.items():
            log[f"closed_loop/traj_colormap/{metric}/{mp4_path.stem}"] = wandb.Image(
                str(png_path), caption=f"metric={metric}"
            )

    if rows:
        log["closed_loop/episodes_table"] = build_episode_table(rows, out_dir=out_dir, site=site)
    if out_dir is not None:
        log["closed_loop/report_path"] = resolve_report_link(out_dir, report_base_url)
    elif effective_out_dir:
        log["closed_loop/report_path"] = str(effective_out_dir)
    return log


def _wandb_scalar(val) -> bool:
    if val is None:
        return False
    if isinstance(val, (int, bool)):
        return True
    if isinstance(val, float):
        return math.isfinite(val)
    return False


def _video_wandb_key(mp4_path: Path) -> str:
    parts = mp4_path.parts
    if "videos" in parts:
        idx = parts.index("videos")
        rel_parts = list(parts[idx + 1 :])
        # videos/<metric_group>/<area>/file.mp4 -> area/file for stable area-level keys
        if len(rel_parts) >= 2:
            rel_parts = rel_parts[1:]
        rel = "/".join(rel_parts)
        return f"closed_loop/grouped/video/{rel.replace('/', '_')}"
    return f"closed_loop/grouped/video/{mp4_path.stem}"
