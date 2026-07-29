"""Build the self-contained local HTML gallery for a multi-site closed-loop run.

One page: a per-site summary table + a searchable/sortable/filterable grid of episode
cards (video + metrics), reading each site's ``summary.json``/``segments.jsonl`` from
``<out_root>/<site_name>/`` (the layout ``run_all_sites_closed_loop.py`` produces). No
external assets — safe to open directly from disk, and this is the "rich" artifact the
W&B side just links to (see ``scenario_generation.wandb_closed_loop.resolve_report_link``)
rather than duplicating (videos stay local/on the training server, not uploaded to W&B).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scenario_generation.closed_loop_score_keys import extract_score
from scenario_generation.trajectory_colormap import METRIC_CHOICES, render_trajectory_colormaps

# Shown first in each card's metric dropdown when available (the rest follow METRIC_CHOICES
# order) — clearance is the most broadly useful default view.
_DEFAULT_METRIC = "clearance"

# Cycled per site in discovery order — enough distinct hues for a handful of sites before
# repeating; exact color isn't semantically meaningful, just needs to be stable per-site
# within one report so the summary table and card tags visually match.
_PALETTE = ["#1a73e8", "#d68a1f", "#8a4ad6", "#2a8a6d", "#d63a3a", "#2aa5d6"]


def collect_site_data(
    out_root: str | Path,
    site_names: list[str],
    *,
    colormap_metrics: tuple[str, ...] = METRIC_CHOICES,
) -> tuple[list[dict], list[dict]]:
    """Read each site's ``summary.json`` + ``segments.jsonl`` into (items, summaries) for
    :func:`build_html_report`. ``items`` is one dict per episode (with a resolved, relative
    ``video_path`` when the mp4 exists, and a ``colormap_paths`` dict of ``{metric: relative
    path}`` — one rendered image per metric in ``colormap_metrics`` that actually produced
    something, so the report's per-card dropdown can switch between them); ``summaries`` is
    one aggregate dict per site.
    """
    out_root = Path(out_root)
    items: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for site_name in site_names:
        site_dir = out_root / site_name
        summary_path = site_dir / "summary.json"
        segments_path = site_dir / "segments.jsonl"
        if not summary_path.is_file():
            continue
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        near_miss_thresh = s.get("near_miss_thresh", 0.5)
        summaries.append(
            {
                "site": site_name,
                "n_segments": s.get("n_segments", 0),
                "total_steps": s.get("total_steps", 0),
                "route_completion": s.get("mean_route_completion", 0.0),
                "total_collision_events": extract_score(s, "total_collision_events") or 0,
                "total_curb_hits": extract_score(s, "total_curb_hits") or 0,
                "total_snaps": extract_score(s, "total_snaps") or 0,
                "total_red_light_violations": extract_score(s, "total_red_light_violations") or 0,
                "total_strong_brakes": extract_score(s, "total_strong_brakes") or 0,
                "n_segments_diverged": s.get("n_segments_diverged", 0),
            }
        )
        if not segments_path.is_file():
            continue
        with segments_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                start, end = r["segment"]
                stem = f"{r['route']}_{start}_{end}"
                video_name = f"{stem}.mp4"
                video_path = site_dir / video_name

                # Skip metrics already rendered on a prior call for this out_root (report
                # rebuilds are common — e.g. re-running --only_sites) rather than re-drawing
                # every episode's every metric from scratch each time.
                missing_metrics = tuple(
                    m
                    for m in colormap_metrics
                    if not (site_dir / f"{stem}_trajcolormap_{m}.png").is_file()
                )
                if missing_metrics:
                    try:
                        render_trajectory_colormaps(
                            site_dir / stem,
                            site_dir,
                            stem,
                            metrics=missing_metrics,
                            near_miss_thresh=near_miss_thresh,
                            title=f"{site_name} {stem}",
                        )
                    except (
                        Exception
                    ) as e:  # pragma: no cover - a bad episode must not break the report
                        print(f"closed_loop_html_report: colormap failed for {stem}: {e}")
                colormap_paths = {
                    m: f"{site_name}/{stem}_trajcolormap_{m}.png"
                    for m in colormap_metrics
                    if (site_dir / f"{stem}_trajcolormap_{m}.png").is_file()
                }

                items.append(
                    {
                        "site": site_name,
                        "route": r["route"],
                        "segment": f"[{start},{end}]",
                        "n_steps_run": r.get("n_steps_run", 0),
                        "terminated": r.get("terminated", ""),
                        "route_completion": round(r.get("route_completion", 0.0), 3),
                        "n_collision_events": extract_score(r, "total_collision_events") or 0,
                        "n_curb_hits": extract_score(r, "total_curb_hits") or 0,
                        "n_snaps": extract_score(r, "total_snaps") or 0,
                        "n_red_light_violations": extract_score(r, "total_red_light_violations")
                        or 0,
                        "n_strong_brakes": extract_score(r, "total_strong_brakes") or 0,
                        "progress_m": round(r.get("progress_m", 0.0), 1),
                        "video_path": f"{site_name}/{video_name}" if video_path.is_file() else None,
                        "colormap_paths": colormap_paths,
                    }
                )
    return items, summaries


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>__TITLE__</title>
<style>
  :root { --bg:#fff; --fg:#1a1a1a; --card:#f5f5f7; --border:#ddd; --danger:#d33; --muted:#888; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#1a1a1a; --fg:#eee; --card:#2a2a2c; --border:#444; --danger:#ff6b6b; --muted:#999; }
  }
  * { box-sizing: border-box; }
  body { background: var(--bg); color: var(--fg); font-family: -apple-system,"Segoe UI",Helvetica,Arial,sans-serif; margin:0; padding:16px 20px 60px; }
  h1 { font-size:1.3rem; margin:0 0 4px; }
  h2 { font-size:1.05rem; margin:28px 0 8px; }
  .sub { color:var(--muted); font-size:0.85rem; margin-bottom:14px; }
  .sub code { background:var(--card); padding:1px 5px; border-radius:4px; }
  table { border-collapse:collapse; width:100%; margin:8px 0 20px; font-size:0.85rem; }
  th,td { border:1px solid var(--border); padding:6px 10px; text-align:right; }
  th:first-child, td:first-child { text-align:left; }
  th { background: var(--card); }
  .controls { display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin-bottom:16px; position:sticky; top:0; background:var(--bg); padding:8px 0; z-index:5; }
  input[type=text] { flex:1; min-width:200px; padding:8px 10px; border:1px solid var(--border); border-radius:6px; background:var(--card); color:var(--fg); font-size:0.9rem; }
  select { padding:8px 10px; border:1px solid var(--border); border-radius:6px; background:var(--card); color:var(--fg); font-size:0.9rem; }
  .count { font-size:0.85rem; color:var(--muted); white-space:nowrap; }
  .grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap:14px; }
  .card { background: var(--card); border:1px solid var(--border); border-radius:10px; overflow:hidden; }
  .card video { width:100%; display:block; background:#000; }
  .card .colormap { width:100%; height:320px; object-fit:contain; display:block; background:#fff; cursor:zoom-in; }
  .card .colormap-controls { display:flex; justify-content:flex-end; align-items:center; gap:6px; padding:4px 10px 0; }
  .card .colormap-controls select { font-size:0.68rem; padding:1px 4px; border-radius:4px; border:1px solid var(--border); background:var(--card); color:var(--fg); }
  .meta { padding:8px 10px; font-size:0.78rem; line-height:1.5; }
  .title { font-weight:600; font-size:0.82rem; margin-bottom:2px; word-break:break-word; }
  .tag { display:inline-block; border-radius:4px; padding:1px 6px; font-size:0.7rem; margin-right:4px; color:#fff; }
  .metrics { display:flex; flex-wrap:wrap; gap:8px; margin-top:6px; color:var(--muted); }
  .metrics b { color: var(--fg); }
  .flag { color: var(--danger); font-weight:700; }
</style>
</head>
<body>

<h1>__TITLE__</h1>
<div class="sub">__SUBTITLE__</div>

<h2>Per-Site Summary</h2>
<table id="summaryTable">
<tr><th>Site</th><th>segments</th><th>Route Completion</th><th>Collisions</th><th>Curb Hits</th><th>Stuck (snaps)</th><th>Red Light</th><th>Strong Brake</th><th>Diverged</th></tr>
</table>

<h2>Episodes</h2>
<div class="controls">
  <input type="text" id="search" placeholder="Search by route">
  <select id="siteFilter"><option value="">Site: All</option></select>
  <select id="sortSel">
    <option value="default">Sort: Default</option>
    <option value="completion_asc">Route Completion (worst first)</option>
    <option value="collision_desc">Collisions (desc)</option>
    <option value="curb_desc">Curb Hits (desc)</option>
  </select>
  <span class="count" id="count"></span>
</div>

<div class="grid" id="grid"></div>

<script>
const ITEMS = __ITEMS_JSON__;
const SUMMARY = __SUMMARY_JSON__;
const PALETTE = __PALETTE_JSON__;
const DEFAULT_METRIC = __DEFAULT_METRIC_JSON__;
const siteColor = {};
[...new Set(ITEMS.map(i => i.site))].sort().forEach((s, i) => { siteColor[s] = PALETTE[i % PALETTE.length]; });

const summaryTable = document.getElementById('summaryTable');
for (const s of SUMMARY) {
  const tr = document.createElement('tr');
  const c = siteColor[s.site] || '#888';
  tr.innerHTML = `
    <td style="color:${c};font-weight:700">${s.site}</td>
    <td>${s.n_segments}</td>
    <td>${(s.route_completion*100).toFixed(0)}%</td>
    <td>${s.total_collision_events}</td>
    <td>${s.total_curb_hits}</td>
    <td>${s.total_snaps}</td>
    <td>${s.total_red_light_violations}</td>
    <td>${s.total_strong_brakes}</td>
    <td>${s.n_segments_diverged}</td>`;
  summaryTable.appendChild(tr);
}

const siteFilter = document.getElementById('siteFilter');
[...new Set(ITEMS.map(i => i.site))].sort().forEach(s => {
  const o = document.createElement('option'); o.value = s; o.textContent = s; siteFilter.appendChild(o);
});

const grid = document.getElementById('grid');
const searchEl = document.getElementById('search');
const sortEl = document.getElementById('sortSel');
const countEl = document.getElementById('count');

function num(v) { const n = parseFloat(v); return (v === null || isNaN(n)) ? null : n; }

function card(item) {
  const div = document.createElement('div');
  div.className = 'card';
  const coll = item.n_collision_events || 0, curb = item.n_curb_hits || 0, snaps = item.n_snaps || 0;
  const redLight = item.n_red_light_violations || 0, brakes = item.n_strong_brakes || 0;
  const completion = ((item.route_completion || 0)*100).toFixed(0)+'%';
  const c = siteColor[item.site] || '#888';
  const videoTag = item.video_path
    ? `<video controls muted preload="metadata" src="${item.video_path}"></video>`
    : `<div style="padding:40px;text-align:center;color:var(--muted);">video missing</div>`;

  const metrics = Object.keys(item.colormap_paths || {});
  const preferred = metrics.includes(DEFAULT_METRIC) ? DEFAULT_METRIC : metrics[0];
  let colormapBlock = '';
  if (metrics.length) {
    const opts = metrics.map(m => `<option value="${m}" ${m===preferred?'selected':''}>${m}</option>`).join('');
    colormapBlock = `
      <div class="colormap-controls">
        <select class="metricSel">${opts}</select>
      </div>
      <img class="colormap" src="${item.colormap_paths[preferred]}" loading="lazy" onclick="window.open(this.src,'_blank')" alt="trajectory overlay">`;
  }

  div.innerHTML = `
    ${videoTag}
    ${colormapBlock}
    <div class="meta">
      <div class="title">${item.route}</div>
      <span class="tag" style="background:${c}">${item.site}</span>
      <span class="tag" style="background:#888">${item.segment}</span>
      <div class="metrics">
        <span>Route completion: <b>${completion}</b></span>
        <span class="${coll>0?'flag':''}">Collisions: <b>${coll}</b></span>
        <span class="${curb>0?'flag':''}">Curb hits: <b>${curb}</b></span>
        <span class="${snaps>0?'flag':''}">Stuck: <b>${snaps}</b></span>
        <span class="${redLight>0?'flag':''}">Red light: <b>${redLight}</b></span>
        <span class="${brakes>0?'flag':''}">Strong brake: <b>${brakes}</b></span>
        <span>steps: <b>${item.n_steps_run}</b></span>
        <span>progress: <b>${item.progress_m}m</b></span>
        <span class="${item.terminated==='diverged'?'flag':''}">terminated: <b>${item.terminated}</b></span>
      </div>
    </div>`;

  if (metrics.length) {
    const sel = div.querySelector('.metricSel');
    const img = div.querySelector('.colormap');
    sel.addEventListener('change', () => { img.src = item.colormap_paths[sel.value]; });
  }
  return div;
}

function render() {
  const q = searchEl.value.trim().toLowerCase();
  const s = siteFilter.value;
  let filtered = ITEMS.filter(i => {
    const hay = `${i.route} ${i.site}`.toLowerCase();
    return (!q || hay.includes(q)) && (!s || i.site === s);
  });
  const mode = sortEl.value;
  if (mode === 'collision_desc') filtered.sort((a,b) => (b.n_collision_events||0)-(a.n_collision_events||0));
  else if (mode === 'curb_desc') filtered.sort((a,b) => (b.n_curb_hits||0)-(a.n_curb_hits||0));
  else if (mode === 'completion_asc') filtered.sort((a,b) => (a.route_completion??1)-(b.route_completion??1));
  grid.innerHTML = '';
  filtered.forEach(i => grid.appendChild(card(i)));
  countEl.textContent = `${filtered.length} / ${ITEMS.length} episodes`;
}

[searchEl,siteFilter,sortEl].forEach(el => el.addEventListener(el===searchEl?'input':'change', render));
render();
</script>
</body>
</html>
"""


def build_html_report(
    out_root: str | Path,
    site_names: list[str],
    *,
    title: str = "Per-Site Closed-Loop Evaluation",
    subtitle: str = "",
    report_filename: str = "report.html",
    colormap_metrics: tuple[str, ...] = METRIC_CHOICES,
) -> Path | None:
    """Write the self-contained gallery HTML to ``<out_root>/<report_filename>``.

    Returns the written path, or ``None`` if no site had a readable ``summary.json``
    (nothing to report yet). ``colormap_metrics`` picks which per-step metrics get a
    trajectory colormap image rendered (default: all of them) — each episode card shows
    one image at a time with a dropdown to switch between the metrics that rendered for
    it, see :mod:`scenario_generation.trajectory_colormap`.
    """
    out_root = Path(out_root)
    items, summaries = collect_site_data(out_root, site_names, colormap_metrics=colormap_metrics)
    if not summaries:
        return None

    html = _TEMPLATE
    html = html.replace("__TITLE__", title)
    html = html.replace("__SUBTITLE__", subtitle)
    html = html.replace("__DEFAULT_METRIC_JSON__", json.dumps(_DEFAULT_METRIC))
    html = html.replace("__ITEMS_JSON__", json.dumps(items, ensure_ascii=False))
    html = html.replace("__SUMMARY_JSON__", json.dumps(summaries, ensure_ascii=False))
    html = html.replace("__PALETTE_JSON__", json.dumps(_PALETTE))

    out_path = out_root / report_filename
    out_path.write_text(html, encoding="utf-8")
    return out_path
