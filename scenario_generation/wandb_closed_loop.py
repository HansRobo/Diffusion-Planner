"""Build ``wandb.Table``s for closed-loop evaluation results.

For each json_label (e.g. ``sites_sample``, ``sites_sample__noobj``,
``close_loop_devops_override_label``) this module produces:

- one **abs** ``wandb.Table`` (raw counts / fraction over the entire run),
  ready to be logged to W&B directly.
- one stacked-bar HTML panel per json_label (:func:`build_per_1000steps_stacked_panels`)
  showing the 5 count-metric columns normalized per 1000 steps.  The panel uses
  ECharts inlined via ``wandb.Html`` so it can be uploaded without any
  pre-registered Vega spec on the W&B backend.

The raw ``per_1000steps`` table is **not** exported — it exists only internally
to drive the stacked bar chart.
"""

from __future__ import annotations

import json

import wandb

from scenario_generation.closed_loop_score_keys import extract_score


# (display column name, source key in a per-group summary dict)
_ABS_COLUMNS = [
    ("Group", "group"),
    ("Segments", "n_segments"),
    ("Steps", "total_steps"),
    ("Route completion (%)", "mean_route_completion"),
    ("Curb hits", "total_curb_hits"),
    ("Snaps", "total_snaps"),
    ("Red light", "total_red_light_violations"),
    ("Strong brakes", "total_strong_brakes"),
    ("Segs diverged", "n_segments_diverged"),
    ("Collisions", "total_collision_events"),
]

_PER_1000STEPS_COLUMNS = [
    ("Group", "group"),
    ("Segments", "n_segments"),
    ("Steps", "total_steps"),
    ("Route completion (%)", "mean_route_completion"),
    ("Curb hits / 1k steps", "total_curb_hits"),
    ("Snaps / 1k steps", "total_snaps"),
    ("Red light / 1k steps", "total_red_light_violations"),
    ("Strong brakes / 1k steps", "total_strong_brakes"),
    ("Segs diverged / 1k steps", "n_segments_diverged"),
    ("Collisions / 1k steps", "total_collision_events"),
]


def _short_label(group_key: str) -> str:
    """``sites_sample/group_a`` → ``group_a``. ``group_a`` → ``group_a``."""
    return group_key.split("/", 1)[1] if "/" in group_key else group_key


def _abs_value(source_key: str, summary: dict):
    """Numeric value for an abs column.

    Looks up the raw key first so the cross-group aggregate row (which is a
    flat dict like ``{"total_curb_hits": 39, ...}``) is read correctly.
    Falls back to ``extract_score`` for raw per-group summaries whose
    headline numbers live in nested categories like ``road_border``.
    """
    if source_key == "mean_route_completion":
        if "mean_route_completion" in summary:
            return float(summary["mean_route_completion"] or 0.0)
        return float(extract_score(summary, source_key) or 0.0)
    if source_key in ("n_segments", "total_steps"):
        return int(summary.get(source_key, 0) or 0)
    if source_key in summary:
        return int(summary[source_key] or 0)
    return int(extract_score(summary, source_key) or 0)


def _per_1000steps_value(source_key: str, summary: dict) -> float:
    """Counts normalized per 1000 steps (or per 1000 segments for ``n_segments_diverged``)."""
    if source_key in ("n_segments", "total_steps", "mean_route_completion"):
        return _abs_value(source_key, summary)
    denom_key = "n_segments" if source_key == "n_segments_diverged" else "total_steps"
    denom = int(summary.get(denom_key, 0) or 0)
    if denom <= 0:
        return 0.0
    raw = summary.get(source_key)
    if raw is None:
        raw = extract_score(summary, source_key)
    return int(raw or 0) / denom * 1000.0


def _aggregate(group_summaries: dict[str, dict]) -> dict:
    """Cross-group aggregate, same shape as a per-group summary dict.

    ``__noobj`` groups are excluded from collision sums (they're 0 by
    construction in the no-object ablation). ``route_completion`` is a
    segment-weighted mean, not a plain average.
    """
    if not group_summaries:
        return {}

    values = list(group_summaries.values())
    objects_only_values = [
        s for k, s in group_summaries.items() if "__noobj/" not in k
    ]

    n_segments = sum(int(s.get("n_segments", 0) or 0) for s in values)
    route_num = sum(
        float(s.get("mean_route_completion", 0.0) or 0.0)
        * int(s.get("n_segments", 0) or 0)
        for s in values
    )

    agg = {
        "n_groups": len(values),
        "n_segments": n_segments,
        "total_steps": sum(int(s.get("total_steps", 0) or 0) for s in values),
        "mean_route_completion": (route_num / n_segments) if n_segments else 0.0,
    }
    for k in ("total_curb_hits", "total_snaps", "total_red_light_violations",
              "total_strong_brakes", "n_segments_diverged"):
        agg[k] = sum(int(extract_score(s, k) or 0) for s in values)
    agg["total_collision_events"] = sum(
        int(extract_score(s, "total_collision_events") or 0) for s in objects_only_values
    )
    return agg


def _build_table(
    json_label: str,
    kind: str,  # "abs" or "per_1000steps"
    group_summaries: dict[str, dict],
) -> wandb.Table:
    cols = _ABS_COLUMNS if kind == "abs" else _PER_1000STEPS_COLUMNS
    value_fn = _abs_value if kind == "abs" else _per_1000steps_value

    rows: list[list] = []
    for group_key in sorted(group_summaries.keys()):
        summary = group_summaries[group_key]
        rows.append([
            _short_label(group_key) if src == "group" else value_fn(src, summary)
            for _, src in cols
        ])

    all_agg = _aggregate(group_summaries)
    rows.append([
        "All" if src == "group" else value_fn(src, all_agg)
        for _, src in cols
    ])

    return wandb.Table(columns=[c[0] for c in cols], data=rows)


def build_closed_loop_tables(
    by_json: dict[str, dict[str, dict]],
) -> dict[str, wandb.Table]:
    """Build one **abs** ``wandb.Table`` per json_label.

    ``by_json`` maps ``json_label`` → ``group_key`` → per-group summary dict.
    Returns ``{"{json_label}/metrics": table, ...}``.
    """
    out: dict[str, wandb.Table] = {}
    for json_label, group_summaries in sorted(by_json.items()):
        out[f"{json_label}/metrics"] = _build_table(json_label, "abs", group_summaries)
    return out


# Metrics stacked into the per-1000-steps bar chart. Order is preserved so
# colors stay stable across runs in the rendered ECharts panel.
_STACKED_METRICS = (
    ("Curb hits / 1k steps", "#4C78A8"),
    ("Snaps / 1k steps", "#F58518"),
    ("Red light / 1k steps", "#E45756"),
    ("Strong brakes / 1k steps", "#72B7B2"),
    ("Collisions / 1k steps", "#EECA3B"),
)


def _build_stacked_bar_html(json_label: str, per_1000steps_table: wandb.Table) -> str:
    """Render an ECharts stacked-bar chart from a per_1000steps Table.

    Reads the table column-wise so we don't depend on row order.
    The ``All`` summary row is dropped — the chart shows the groups only.
    """
    cols = per_1000steps_table.columns
    stacked_cols = [metric for metric, _ in _STACKED_METRICS]
    table_data = per_1000steps_table.data
    col_idx = {name: i for i, name in enumerate(cols)}

    groups: list[str] = []
    series: dict[str, list[float]] = {m: [] for m in stacked_cols}
    for row in table_data:
        if row[col_idx["Group"]] == "All":
            continue
        groups.append(str(row[col_idx["Group"]]))
        for metric in stacked_cols:
            series[metric].append(float(row[col_idx[metric]] or 0.0))

    series_payload = [
        {
            "name": metric,
            "type": "bar",
            "stack": "events",
            "barWidth": 20,
            "emphasis": {"focus": "series"},
            "itemStyle": {"color": color},
            "data": series[metric],
        }
        for metric, color in _STACKED_METRICS
    ]

    max_label_len = max((len(g) for g in groups), default=0)
    grid_left = max(160, max_label_len * 6 + 5)
    chart_height = len(groups) * 50 + 30

    chart_id = f"echarts_closed_loop_{json_label}".replace("/", "_").replace(":", "_")
    option = {
        "title": {"show": False},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "legend": {"bottom": 0, "type": "scroll"},
        "grid": {"left": grid_left, "right": 20, "top": 10, "bottom": 50},
        "yAxis": {
            "type": "category",
            "data": groups,
            "axisLabel": {
                "width": grid_left - 20,
                "overflow": "truncate",
                "tooltip": {"show": False, "trigger": "item"},
            },
        },
        "xAxis": {"type": "value", "name": "Events / 1k steps"},
        "series": series_payload,
    }

    return (
        f'<div id="{chart_id}" style="width:100%;height:{chart_height}px;min-width:0;"></div>'
        '<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>'
        '<script>'
        f"var dom = document.getElementById({json.dumps(chart_id)});"
        f"var chart = echarts.init(dom);"
        f"chart.setOption({json.dumps(option)});"
        "window.addEventListener('resize', function () { chart.resize(); });"
        "</script>"
    )


def build_per_1000steps_stacked_panels(
    by_json: dict[str, dict[str, dict]],
) -> dict[str, wandb.Html]:
    """For each json_label, return ``{panel_key: wandb.Html}`` ready
    to be passed to ``run.log``. Builds the per_1000steps table internally
    (not exported); only the stacked chart is returned.
    """
    out: dict[str, wandb.Html] = {}
    for json_label, group_summaries in sorted(by_json.items()):
        out[f"{json_label}/count_per_1000steps"] = wandb.Html(
            _build_stacked_bar_html(json_label, _build_table(json_label, "per_1000steps", group_summaries))
        )
    return out
