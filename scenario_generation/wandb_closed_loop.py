"""Build wandb log payloads for full-route (per-group) closed-loop validation."""

import math
import os
from pathlib import Path

import wandb

from scenario_generation.closed_loop_score_keys import (
    COMPARISON_OVERVIEW_SUM_KEYS,
    OBJECTS_ONLY_OVERVIEW_SUM_KEYS,
    SCORE_KEYS,
    extract_score,
)
from scenario_generation.trajectory_colormap import METRIC_CHOICES, render_trajectory_colormaps


def is_noobj_label(label: str) -> bool:
    """``__noobj/`` substring marks the empty-world-ablation mode; trailing ``/``
    avoids misdetecting group names that happen to contain ``__noobj``."""
    return "__noobj/" in label


def _json_name_from_label(label: str) -> str:
    """Split on ``"__noobj/"`` — same boundary as :func:`is_noobj_label`."""
    return label.split("__noobj/", 1)[0]


def episode_stem(out_dir: str | Path, row: dict) -> str:
    """Base filename stem for one segments.jsonl row's video/png-dir/colormap files.

    Prefers the segment-suffixed ``{route}_{start}_{end}`` stem (one row may span a
    sub-segment of a route); falls back to bare ``{route}`` when neither the suffixed
    video file nor png directory exists, which covers callers that write at whole-route
    granularity.
    """
    out_dir = Path(out_dir)
    start, end = row["segment"]
    segmented_stem = f"{row['route']}_{start}_{end}"
    if (out_dir / f"{segmented_stem}.mp4").is_file() or (out_dir / segmented_stem).is_dir():
        return segmented_stem
    return row["route"]


def _segment_paths(out_dir: str | Path, row: dict) -> tuple[Path, Path]:
    """(png_dir, mp4_path) for one segments.jsonl row -- see :func:`episode_stem`."""
    out_dir = Path(out_dir)
    stem = episode_stem(out_dir, row)
    return out_dir / stem, out_dir / f"{stem}.mp4"


def pick_representative_row(rows: list[dict], mode: str = "worst") -> dict | None:
    """Pick one segment row to represent a group's whole run (for the 1 video/image W&B keeps).

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
        obj = r.get("object", {})
        coll = obj.get("collision_steps", 0)
        cl = obj.get("clearance_min_m", float("inf"))
        cl = cl if math.isfinite(cl) else 1e9
        return (coll, -cl)  # more collisions first, then smaller clearance first

    return max(rows, key=_worst_key)


def scalar_key(group: str, metric: str) -> str:
    """``closed_loop/{metric}/{group}`` — ``metric`` leads so one regex overlay grabs all groups."""
    return f"{SCALAR_KEY_PREFIX}{metric}/{group}"


def media_key(group: str, slot: str) -> str:
    """``closed_loop/{group}/media/{slot}`` — ``group`` leads since media can't be overlaid."""
    return f"{SCALAR_KEY_PREFIX}{group}/media/{slot}"


SCALAR_KEY_PREFIX = "closed_loop/"
EPISODE_TABLE_NAME = "closed_loop_episodes/all"
OVERVIEW_KEY_PREFIX = "closed_loop_overview"


EPISODE_TABLE_COLUMNS = [
    "group",
    "route",
    "segment",
    "n_steps_run",
    "terminated",
    "route_completion",
    "n_collision_events",
    "n_curb_hits",
    "n_snaps",
    "n_red_light_violations",
    "n_strong_brakes",
    "progress_m",
    "video_path",
]


def _episode_row(table: wandb.Table, group: str, r: dict, out_dir: str | Path | None) -> None:
    seg = r.get("segment")
    seg_str = f"[{seg[0]},{seg[1]}]" if seg else ""
    video_path = str(_segment_paths(out_dir, r)[1]) if out_dir is not None else ""
    comp = r.get("route_completion")
    table.add_data(
        group,
        r.get("route", ""),
        seg_str,
        int(r.get("n_steps_run", 0)),
        r.get("terminated", ""),
        float(comp) if comp is not None and math.isfinite(comp) else None,
        int(extract_score(r, "total_collision_events") or 0),
        int(extract_score(r, "total_curb_hits") or 0),
        int(extract_score(r, "total_snaps") or 0),
        int(extract_score(r, "total_red_light_violations") or 0),
        int(extract_score(r, "total_strong_brakes") or 0),
        float(r.get("progress_m", 0.0)),
        video_path,
    )


def build_groups_wandb_log(
    group_summaries: dict[str, dict],
    *,
    out_root: str | Path,
    video_pick: str = "worst",
    colormap_metrics: tuple[str, ...] = METRIC_CHOICES,
    near_miss_thresh_default: float = 0.5,
    render_media: bool = True,
) -> dict:
    """Per-group closed-loop log payload for one W&B step. Caller owns the wandb session.

    See module constants for key schema: ``SCALAR_KEY_PREFIX`` / ``OVERVIEW_KEY_PREFIX`` /
    ``EPISODE_TABLE_NAME``.
    """
    out_root = Path(out_root)
    log: dict = {}
    episode_data: list = []

    for group_name, summary in group_summaries.items():
        rows = summary.get("segments") or []
        group_out_dir = out_root / group_name.replace("/", os.sep)
        log.update(
            build_full_closed_loop_wandb_log(
                summary,
                out_dir=str(group_out_dir),
                group=group_name,
                video_pick=video_pick,
                colormap_metrics=colormap_metrics,
                near_miss_thresh=summary.get("near_miss_thresh", near_miss_thresh_default),
                render_media=render_media,
            )
        )
        episode_data.append((group_name, rows, str(group_out_dir)))

    log[EPISODE_TABLE_NAME] = build_combined_episode_table(episode_data)

    log.update(build_groups_aggregate_log(group_summaries, prefix=OVERVIEW_KEY_PREFIX))

    # Per-json aggregates under ``closed_loop_overview/<json_label>/`` so the workspace's
    # per-json Overview panel has real data; objects and noobj modes are kept separate.
    per_mode_summaries: dict[str, dict[str, dict]] = {}
    for label, summary in group_summaries.items():
        mode_tag = (
            _json_name_from_label(label) + "__noobj"
            if is_noobj_label(label)
            else _json_name_from_label(label)
        )
        per_mode_summaries.setdefault(mode_tag, {})[label] = summary
    for json_label, sub_summaries in per_mode_summaries.items():
        log.update(
            build_groups_aggregate_log(sub_summaries, prefix=f"{OVERVIEW_KEY_PREFIX}/{json_label}")
        )

    return log


def build_combined_episode_table(
    group_episodes: list[tuple[str, list[dict], str | Path | None]],
) -> wandb.Table:
    """One episode table across every group (``group`` column filled per row) so the W&B UI can
    sort/filter across the whole run in a single panel.
    """
    table = wandb.Table(columns=EPISODE_TABLE_COLUMNS)
    for group, rows, out_dir in group_episodes:
        for r in rows:
            _episode_row(table, group, r, out_dir)
    return table


def build_groups_aggregate_log(
    summaries: dict[str, dict],
    *,
    prefix: str = "closed_loop_overview",
) -> dict:
    """Cross-group rollup under ``<prefix>/``: segment-weighted mean route-completion,
    plus plain sums of event counts. ``__noobj`` labels are excluded from collision-style
    sums (``OBJECTS_ONLY_OVERVIEW_SUM_KEYS``) since they're 0 by construction.
    """
    log: dict = {}
    if not summaries:
        return log
    values = list(summaries.values())
    objects_values = [s for label, s in summaries.items() if not is_noobj_label(label)]
    n_groups = len(values)
    total_segments = sum(int(s.get("n_segments", 0)) for s in values)

    log[f"{prefix}/n_groups"] = n_groups
    log[f"{prefix}/n_segments"] = total_segments

    route_completion_num = sum(
        float(s.get("mean_route_completion", 0.0)) * int(s.get("n_segments", 0)) for s in values
    )
    log[f"{prefix}/route_completion"] = route_completion_num / total_segments if total_segments else 0.0

    for key in COMPARISON_OVERVIEW_SUM_KEYS:
        log[f"{prefix}/{key}"] = sum(int(extract_score(s, key) or 0) for s in values)
    for key in OBJECTS_ONLY_OVERVIEW_SUM_KEYS:
        log[f"{prefix}/{key}"] = sum(int(extract_score(s, key) or 0) for s in objects_values)

    return {k: v for k, v in log.items() if _wandb_scalar(v) or isinstance(v, int)}


def build_full_closed_loop_wandb_log(
    summary: dict,
    *,
    out_dir: str | Path | None = None,
    group: str | None = None,
    video_pick: str = "worst",
    colormap_metrics: tuple[str, ...] = METRIC_CHOICES,
    near_miss_thresh: float = 0.5,
    render_media: bool = True,
) -> dict:
    """One per-group closed-loop wandb payload. Caller owns the wandb session."""
    log: dict = {}

    def _scalar_key(metric: str) -> str:
        return scalar_key(group, metric)

    for key in SCORE_KEYS:
        val = extract_score(summary, key)
        if _wandb_scalar(val):
            log[_scalar_key(key)] = val

    rows = summary.get("segments") or []
    rep = pick_representative_row(rows, mode=video_pick)
    if render_media and rep is not None and out_dir is not None:
        png_dir, mp4_path = _segment_paths(out_dir, rep)
        if mp4_path.is_file():
            log[_media_key("video")] = wandb.Video(str(mp4_path), format="mp4")
        try:
            rendered = render_trajectory_colormaps(
                png_dir,
                out_dir,
                mp4_path.stem,
                metrics=colormap_metrics,
                near_miss_thresh=near_miss_thresh,
                strong_brake_mps2=summary.get("strong_brake", {}).get("thresh_mps2", -2.5),
                title=f"{group or ''} {mp4_path.stem}".strip(),
            )
        except Exception as e:  # pragma: no cover - rendering must never break training
            print(f"closed_loop: trajectory colormap failed for {mp4_path.stem}: {e}")
            rendered = {}
        gallery = [
            wandb.Image(str(rendered[m]), caption=m) for m in colormap_metrics if m in rendered
        ]
        if gallery:
            log[_media_key("gallery")] = gallery

    return log


def _wandb_scalar(val) -> bool:
    if val is None:
        return False
    if isinstance(val, bool):
        return False
    try:
        return math.isfinite(float(val))
    except (TypeError, ValueError):
        return False
