"""Build wandb log payloads for full-route (per-site) closed-loop validation."""

from __future__ import annotations

import math
from pathlib import Path

import wandb

from scenario_generation.trajectory_colormap import METRIC_CHOICES, render_trajectory_colormaps

# Per-site summary count keys summed across ALL labels (objects + "__noobj" ablation) by
# build_sites_aggregate_log for the overview — meaningful in either mode.
_OVERVIEW_SUM_KEYS = ("total_curb_hits", "total_snaps", "n_segments_diverged")
# Summed across objects-labeled sites ONLY (see _is_noobj_label) — these are structurally 0 for
# the empty-world ablation (no traffic to collide with), so mixing "__noobj" labels in would
# just dilute the objects-mode signal with meaningless zeros. See objects_ablation_strategy.md.
_OVERVIEW_OBJECTS_ONLY_SUM_KEYS = ("total_collision_events",)


def _is_noobj_label(label: str) -> bool:
    """True for the empty-world-ablation site label convention (``{site}__noobj``) — see
    objects_ablation_strategy.md. Used to exclude ablation labels from collision-style
    aggregates that are 0 by construction in that mode."""
    return label.endswith("__noobj")


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
    "route_completion",
    "n_collision_events",
    "n_curb_hits",
    "n_snaps",
    "progress_m",
    "video_path",
]


def _episode_row(table: wandb.Table, site: str, r: dict, out_dir: str | Path | None) -> None:
    seg = r.get("segment")
    seg_str = f"[{seg[0]},{seg[1]}]" if seg else ""
    video_path = str(_segment_paths(out_dir, r)[1]) if out_dir is not None else ""
    comp = r.get("route_completion")
    table.add_data(
        site,
        r.get("route", ""),
        seg_str,
        int(r.get("n_steps_run", 0)),
        r.get("terminated", ""),
        float(comp) if comp is not None and math.isfinite(comp) else None,
        int(r.get("n_collision_events", 0)),
        int(r.get("n_curb_hits", 0)),
        int(r.get("n_snaps", 0)),
        float(r.get("progress_m", 0.0)),
        video_path,
    )


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
        _episode_row(table, site or "", r, out_dir)
    return table


def build_combined_episode_table(
    site_episodes: list[tuple[str, list[dict], str | Path | None]],
) -> wandb.Table:
    """ONE episode table across every site (``site`` column filled per row), so the W&B UI's
    native sort/filter/group-by works across the whole run — group by ``site``, sort by
    ``n_collision_steps`` desc, etc. — in a single interactive panel instead of one table
    per site. ``site_episodes`` is ``[(site_name, rows, out_dir), ...]``.
    """
    table = wandb.Table(columns=EPISODE_TABLE_COLUMNS)
    for site, rows, out_dir in site_episodes:
        for r in rows:
            _episode_row(table, site, r, out_dir)
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
    """Cross-site rollup under ``closed_loop_overview/`` (the at-a-glance block): the segment-
    weighted mean route-completion (so long routes aren't under-weighted), plus the plain
    cross-site SUM of each event count. Deliberately just the small non-saturating set — no
    segment-rates / min-clearances / means (those stay in each site's summary.json only).

    ``summaries`` is keyed by site LABEL — a ``{site}__noobj`` label (see
    ``objects_ablation_strategy.md``) is excluded from collision-style sums
    (``_OVERVIEW_OBJECTS_ONLY_SUM_KEYS``), since those are 0 by construction in the
    empty-world ablation and would just dilute the objects-mode number with zeros.
    """
    log: dict = {}
    if not summaries:
        return log
    values = list(summaries.values())
    objects_values = [s for label, s in summaries.items() if not _is_noobj_label(label)]
    n_sites = len(values)
    total_segments = sum(int(s.get("n_segments", 0)) for s in values)

    log["closed_loop_overview/n_sites"] = n_sites
    log["closed_loop_overview/n_segments"] = total_segments

    comp_num = sum(
        float(s.get("mean_route_completion", 0.0)) * int(s.get("n_segments", 0)) for s in values
    )
    log["closed_loop_overview/route_completion"] = (
        comp_num / total_segments if total_segments else 0.0
    )

    for key in _OVERVIEW_SUM_KEYS:
        log[f"closed_loop_overview/{key}"] = sum(int(s.get(key, 0)) for s in values)
    for key in _OVERVIEW_OBJECTS_ONLY_SUM_KEYS:
        log[f"closed_loop_overview/{key}"] = sum(int(s.get(key, 0)) for s in objects_values)

    return {k: v for k, v in log.items() if _wandb_scalar(v) or isinstance(v, int)}


# Per-site scalar trends worth a line chart over epochs — the deliberately small, non-saturating
# set: route_completion (↑, the headline "getting better" signal) plus event COUNTS (↓). The
# saturating segment-rates and worst-moment min-clearances, and the mean_* metrics, are NOT
# shown (still in summary.json / episode table for reference) — they either pin to 0/1 on a
# hard site or were judged not worth the panel by the user.
_SCORE_KEYS = (
    "mean_route_completion",
    "total_collision_events",
    "total_curb_hits",
    "total_snaps",
)


def _site_label(site: str | None) -> str:
    """W&B-key-safe site token; ``None`` (single-npz_root mode) -> ``"main"``."""
    return (site or "main").replace("/", "_")


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
    """Per-site full-route closed-loop wandb payload, keyed into role-based sections so the
    workspace stays navigable (one collapsible section each) instead of one flat ``closed_loop``
    blob of 100+ panels:

    - ``closed_loop_scores/{metric}/{site}`` — scalar trends (metric-first so the same metric's
      sites sort adjacently; the W&B panel-search box filters by either metric or site token).
    - ``closed_loop_media/{site}`` — ONE gallery panel holding every ``colormap_metrics`` image
      for the representative episode (captioned by metric), mirroring the HTML report's
      per-card metric dropdown; ``closed_loop_media/{site}__video`` — that episode's video.
    - ``closed_loop_links/{site}`` — where the full report (all videos + HTML) lives.

    The per-episode table is built once across ALL sites by :func:`build_combined_episode_table`
    at the caller (so it's a single filterable/groupable panel), not here.
    """
    label = _site_label(site)
    log: dict = {}
    for key in _SCORE_KEYS:
        val = summary.get(key)
        if _wandb_scalar(val):
            log[f"closed_loop_scores/{key}/{label}"] = val

    rows = summary.get("segments") or []
    rep = pick_representative_row(rows, mode=video_pick)
    if rep is not None and out_dir is not None:
        png_dir, mp4_path = _segment_paths(out_dir, rep)
        if mp4_path.is_file():
            log[f"closed_loop_media/{label}__video"] = wandb.Video(str(mp4_path), format="mp4")
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
        # One gallery panel per site: a list of images under a single key (captioned by metric)
        # -> the metric becomes an in-panel selector, not N separate panels. Ordered by the
        # requested colormap_metrics so the gallery is stable across epochs/sites.
        gallery = [
            wandb.Image(str(rendered[m]), caption=m) for m in colormap_metrics if m in rendered
        ]
        if gallery:
            log[f"closed_loop_media/{label}"] = gallery

    if out_dir is not None:
        log[f"closed_loop_links/{label}"] = resolve_report_link(out_dir, report_base_url)
    elif summary.get("npz_root"):
        log[f"closed_loop_links/{label}"] = str(summary["npz_root"])
    return log


def _wandb_scalar(val) -> bool:
    if val is None:
        return False
    if isinstance(val, (int, bool)):
        return True
    if isinstance(val, float):
        return math.isfinite(val)
    return False


