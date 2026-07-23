"""Build (or refresh) a curated W&B workspace view for closed-loop eval.

The scalars/media/table this project logs under ``closed_loop_scores/``,
``closed_loop_overview/``, ``closed_loop_media/``, and ``closed_loop_episodes/`` (see
:mod:`scenario_generation.wandb_closed_loop`) are already namespaced into sections, but
W&B's DEFAULT auto-generated workspace still renders one panel per distinct key — for
N sites x ~8 score metrics that's still dozens of small single-line panels, one per
(metric, site) pair, with no overlay.

This module uses the ``wandb-workspaces`` SDK to define an explicit, curated workspace
instead: one LinePlot PER METRIC with every site overlaid as its own colored line (via
``metric_regex``, so it doesn't need to know site names), one MediaBrowser per site (its
colormap gallery + video), and one table panel for the combined episode table. Built with
``auto_generate_panels=False``, so none of the flat auto-panels appear alongside it.

This is a project-level view, not a per-run artifact — run it ONCE (or whenever the set of
score metrics changes) via the CLI below, not from inside the training loop. The resulting
``.url`` is a saved, shareable, LIVE view: it re-renders with each new ``wandb.log()`` call
under the same project, exactly like the default workspace does.

Example::

    python -m scenario_generation.wandb_closed_loop_workspace \\
        --entity advanced-technology-department \\
        --project Diffusion-Planner-smoke-test \\
        --site_names odaiba takanawa
"""

from __future__ import annotations

import argparse

# Kept in sync with scenario_generation.wandb_closed_loop._SCORE_KEYS by hand (importing it
# would pull in the heavier torch/matplotlib deps this module doesn't otherwise need). The
# deliberately small, non-saturating headline set: route_completion (↑) + event counts (↓).
_SCORE_KEYS = (
    "mean_route_completion",
    "total_collision_events",
    "total_curb_hits",
    "total_snaps",
)

# Cross-site overview keys (see build_sites_aggregate_log): the pooled completion + the count sums.
_OVERVIEW_KEYS = (
    "route_completion",
    "total_collision_events",
    "total_curb_hits",
    "total_snaps",
    "n_segments_diverged",
)


def build_closed_loop_workspace(
    entity: str,
    project: str,
    *,
    site_names: list[str],
    name: str = "Closed-Loop Dashboard",
) -> str:
    """Create/save a curated closed-loop workspace view and return its URL.

    ``site_names`` only needs to be exact for the per-site media galleries (W&B media
    panels take explicit keys, no regex) — the score line plots use ``metric_regex`` and
    pick up any site automatically, present or added later.
    """
    import wandb_workspaces.reports.v2 as wr
    import wandb_workspaces.workspaces as ws

    overview_panels = [
        wr.LinePlot(title=key, y=[f"closed_loop_overview/{key}"]) for key in _OVERVIEW_KEYS
    ] + [
        wr.LinePlot(title="n_sites / n_segments", y=["closed_loop_overview/n_sites", "closed_loop_overview/n_segments"]),
    ]

    # One panel PER METRIC, every site overlaid as its own line (metric_regex matches
    # "closed_loop_scores/{metric}/{any_site}" without needing to enumerate sites) — this is
    # the key fix over the default workspace's one-tiny-panel-per-(metric,site) layout.
    score_panels = [
        wr.LinePlot(title=metric, metric_regex=rf"^closed_loop_scores/{metric}/.*$")
        for metric in _SCORE_KEYS
    ]

    # Two SEPARATE sections (not just two panel types in one section): the colormap gallery
    # (an indexed LIST of 5 images) and the video (a single file) are different W&B media
    # shapes — combining both key types in one MediaBrowser with gallery_axis="index" silently
    # drops the video (the index axis only applies to the list-shaped key). Grouping all sites'
    # overlays together and all sites' videos together (rather than interleaving per site) also
    # keeps same-purpose panels next to each other.
    overlay_panels = [
        wr.MediaBrowser(
            title=site,
            media_keys=[f"closed_loop_media/{site}"],
            mode="gallery",
            gallery_axis="index",
        )
        for site in site_names
    ]
    video_panels = [
        wr.MediaBrowser(title=site, media_keys=[f"closed_loop_media/{site}__video"])
        for site in site_names
    ]

    episodes_panels = [
        wr.WeavePanelSummaryTable(table_name="closed_loop_episodes/all", layout=wr.Layout(w=24, h=16)),
    ]

    workspace = ws.Workspace(
        entity=entity,
        project=project,
        name=name,
        sections=[
            ws.Section(name="Overview", panels=overview_panels, is_open=True, pinned=True),
            ws.Section(name="Scores (all sites overlaid per metric)", panels=score_panels, is_open=True),
            ws.Section(name="Trajectory Overlay (per site)", panels=overlay_panels, is_open=False),
            ws.Section(name="Video (per site)", panels=video_panels, is_open=False),
            ws.Section(name="Episodes (all sites, one table)", panels=episodes_panels, is_open=False),
        ],
    )
    workspace.save()
    return workspace.url


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--site_names", nargs="+", required=True, help="e.g. odaiba takanawa")
    parser.add_argument("--name", default="Closed-Loop Dashboard")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    url = build_closed_loop_workspace(
        args.entity, args.project, site_names=args.site_names, name=args.name
    )
    print(f"Saved workspace view: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
