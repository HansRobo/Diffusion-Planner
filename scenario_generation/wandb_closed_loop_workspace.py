"""Build a curated W&B workspace view for closed-loop eval.

The closed-loop scalars/media/table are already namespaced
(see :data:`wandb_closed_loop.SCALAR_KEY_PREFIX` / :data:`EPISODE_TABLE_NAME` /
:data:`OVERVIEW_KEY_PREFIX`); W&B's default auto-generated workspace still renders
one tiny single-line panel per (metric, group) pair with no overlay, so we build
an explicit one instead.

``layer1`` slot in section names is either ``"ALL"`` (cross-json aggregate, overlays
new jsons via ``metric_regex``) or a specific ``json_name`` (per-json, explicit
``y=[...]`` enumeration — must re-run when the json list changes).

The intended caller is ``run_all_groups_closed_loop._log_to_wandb``. The standalone
CLI below is just an escape hatch for manual rebuilds.
"""

from __future__ import annotations

import argparse

from scenario_generation.closed_loop_score_keys import (
    COMPARISON_OVERVIEW_SUM_KEYS,
    COMPARISON_SCORE_KEYS,
    OBJECTS_ONLY_OVERVIEW_SUM_KEYS,
    OBJECTS_ONLY_SCORE_KEYS,
)
from scenario_generation.wandb_closed_loop import (
    EPISODE_TABLE_NAME,
    OVERVIEW_KEY_PREFIX,
    SCALAR_KEY_PREFIX,
    is_noobj_label,
    media_key,
    scalar_key,
)

_CLOSED_LOOP_PREFIX = "Closed-Loop / "
_ALL_LEVEL = "ALL"

# route_completion is its own segment-weighted-average key in build_groups_aggregate_log,
# not part of the generic sum loop driven by COMPARISON_OVERVIEW_SUM_KEYS.
_COMPARISON_OVERVIEW_KEYS = ("route_completion", *COMPARISON_OVERVIEW_SUM_KEYS)
_OBJECTS_ONLY_OVERVIEW_KEYS = OBJECTS_ONLY_OVERVIEW_SUM_KEYS


def _section_name(layer1: str, panel: str) -> str:
    return f"{_CLOSED_LOOP_PREFIX}{layer1} / {panel}"


def _split_labels_by_mode(group_names: list[str], json_name: str) -> dict[str, list[str]]:
    """Labels under ``json_name`` split into two mode buckets: ``<json_name>`` (objects
    labels) and ``<json_name>__noobj`` (noobj labels). Missing buckets are omitted.
    Each json renders as up to two separate sections in the dashboard."""
    objects = [g for g in group_names if g.startswith(f"{json_name}/") and not is_noobj_label(g)]
    noobj = [g for g in group_names if g.startswith(f"{json_name}__noobj/")]
    out: dict[str, list[str]] = {}
    if objects:
        out[json_name] = objects
    if noobj:
        out[f"{json_name}__noobj"] = noobj
    return out


def _infer_json_names(group_names: list[str]) -> list[str]:
    """Infer the unique ``json_name`` list from full W&B labels.

    Each label is ``"{json_name}/{group}"`` or ``"{json_name}__noobj/{group}"``.
    Objects-mode (``/``) and noobj-mode (``__noobj/``) prefix variants collapse to
    the same ``json_name``. First-seen order is preserved so section ordering stays
    stable across runs.

    Requires the project-wide convention enforced in :func:`_make_summary_key`
    (``/`` not in json_name or group_name).
    """
    seen: list[str] = []
    for label in group_names:
        for sep in ("__noobj/", "/"):
            idx = label.find(sep)
            if idx < 0:
                continue
            json_name = label[:idx]
            if json_name and json_name not in seen:
                seen.append(json_name)
            break
    return seen


def build_closed_loop_workspace(
    entity: str,
    project: str,
    *,
    group_names: list[str],
    json_names: list[str] | None = None,
    name: str = "Closed-Loop Dashboard",
) -> str:
    """Create/save a curated closed-loop workspace view and return its URL.

    ``group_names`` are full labels as logged (``{json_name}/{group}`` or
    ``{json_name}__noobj/{group}``); ``json_names`` is the per-json list, inferred
    from ``group_names`` when omitted. Sections: ``ALL`` (cross-json, regex overlay)
    plus per-json × per-mode (``objects`` / ``__noobj``). ``auto_generate_panels``
    is always off — the curated view IS the dashboard.
    """
    if json_names is None:
        json_names = _infer_json_names(group_names)
    if not json_names:
        raise ValueError(
            "build_closed_loop_workspace: no json_names could be inferred from group_names; "
            "pass json_names explicitly."
        )
    import wandb_workspaces.reports.v2 as wr
    import wandb_workspaces.workspaces as ws

    any_noobj = any(is_noobj_label(l) for l in group_names)
    objects_labels = [g for g in group_names if not is_noobj_label(g)]

    labels_per_json: dict[str, dict[str, list[str]]] = {
        json_name: _split_labels_by_mode(group_names, json_name) for json_name in json_names
    }

    overview_panels = [
        wr.LinePlot(title=key, y=[f"{OVERVIEW_KEY_PREFIX}/{key}"])
        for key in (*_COMPARISON_OVERVIEW_KEYS, *_OBJECTS_ONLY_OVERVIEW_KEYS)
    ] + [
        wr.LinePlot(
            title="n_groups / n_segments",
            y=[f"{OVERVIEW_KEY_PREFIX}/n_groups", f"{OVERVIEW_KEY_PREFIX}/n_segments"],
        ),
    ]

    # One panel PER METRIC; every group+mode overlaid via metric_regex so new jsons
    # auto-show up.
    all_comparison_score_panels = [
        wr.LinePlot(title=metric, metric_regex=rf"^{SCALAR_KEY_PREFIX}{metric}/.*$")
        for metric in COMPARISON_SCORE_KEYS
    ]
    # Objects-only collision: explicit y=[...] over objects_labels — a noobj line here
    # would be a flat 0 (no traffic to collide with).
    all_objects_only_score_panels = [
        wr.LinePlot(
            title=metric,
            y=sorted(scalar_key(label, metric) for label in objects_labels),
        )
        for metric in OBJECTS_ONLY_SCORE_KEYS
    ]
    # Gallery and video are different W&B media shapes (indexed list vs single file);
    # combining them in one MediaBrowser with gallery_axis="index" silently drops video.
    all_overlay_panels = [
        wr.MediaBrowser(
            title=label,
            media_keys=[media_key(label, "gallery")],
            mode="gallery",
            gallery_axis="index",
        )
        for label in group_names
    ]
    all_video_panels = [
        wr.MediaBrowser(title=label, media_keys=[media_key(label, "video")])
        for label in group_names
    ]
    all_episodes_panels = [wr.WeavePanelSummaryTable(table_name=EPISODE_TABLE_NAME)]

    # Per-json: explicit y=[...] enumeration so different jsons don't bleed together.
    def _per_json_score_panels(labels_for_this: list[str]) -> list:
        return [
            wr.LinePlot(
                title=metric,
                y=sorted(scalar_key(label, metric) for label in labels_for_this),
            )
            for metric in COMPARISON_SCORE_KEYS
        ]

    def _per_json_objects_only_score_panels(labels_for_this: list[str]) -> list:
        # Skip noobj bucket: every label is a noobj label, so this would just be empty
        # LinePlots.
        objects_labels = [l for l in labels_for_this if not is_noobj_label(l)]
        if not objects_labels:
            return []
        return [
            wr.LinePlot(
                title=metric,
                y=sorted(scalar_key(label, metric) for label in objects_labels),
            )
            for metric in OBJECTS_ONLY_SCORE_KEYS
        ]

    def _per_json_overview_panels(json_name: str) -> list:
        """Per-json rollup: ``closed_loop_overview/<json_name>/<key>``."""
        prefix = f"{OVERVIEW_KEY_PREFIX}/{json_name}/"
        return [
            wr.LinePlot(title=key, y=[f"{prefix}{key}"])
            for key in (*_COMPARISON_OVERVIEW_KEYS, *_OBJECTS_ONLY_OVERVIEW_KEYS)
        ] + [
            wr.LinePlot(
                title="n_groups / n_segments",
                y=[f"{prefix}n_groups", f"{prefix}n_segments"],
            ),
        ]

    def _per_json_overlay_panels(labels_for_this: list[str]) -> list:
        return [
            wr.MediaBrowser(
                title=label,
                media_keys=[media_key(label, "gallery")],
                mode="gallery",
                gallery_axis="index",
            )
            for label in labels_for_this
        ]

    def _per_json_video_panels(labels_for_this: list[str]) -> list:
        return [
            wr.MediaBrowser(title=label, media_keys=[media_key(label, "video")])
            for label in labels_for_this
        ]

    training_section = ws.Section(
        name="Training",
        panels=[
            wr.LinePlot(title="train_loss", metric_regex=r"^train_loss/.*$"),
            wr.LinePlot(title="valid_loss", metric_regex=r"^valid_loss/.*$"),
            wr.LinePlot(title="learning_rate", metric_regex=r"^lr/.*$"),
        ],
        is_open=True,
        pinned=True,
    )

    # ---- ALL sections ----
    all_sections: list[ws.Section] = [
        ws.Section(
            name=_section_name(_ALL_LEVEL, "Overview"),
            panels=overview_panels,
            is_open=True,
            pinned=True,
        ),
        ws.Section(
            name=_section_name(_ALL_LEVEL, "Scores (comparison)" if any_noobj else "Scores"),
            panels=all_comparison_score_panels,
            is_open=True,
            pinned=True,
        ),
        ws.Section(
            name=_section_name(_ALL_LEVEL, "Scores (objects-only)"),
            panels=all_objects_only_score_panels,
            is_open=False,
        ),
        ws.Section(
            name=_section_name(_ALL_LEVEL, "Trajectory Overlay"),
            panels=all_overlay_panels,
            is_open=False,
        ),
        ws.Section(
            name=_section_name(_ALL_LEVEL, "Video"),
            panels=all_video_panels,
            is_open=False,
        ),
        ws.Section(
            name=_section_name(_ALL_LEVEL, "Episodes"),
            panels=all_episodes_panels,
            is_open=False,
        ),
    ]

    per_json_sections: list[ws.Section] = []
    for json_name, mode_buckets in labels_per_json.items():
        for mode_label, labels_for_this in mode_buckets.items():
            # ``mode_label`` is ``<json_name>`` or ``<json_name>__noobj`` (no trailing
            # slash), so the group-level ``is_noobj_label`` rule doesn't apply.
            json_is_noobj = mode_label.endswith("__noobj")
            section_specs: list[tuple[str, list]] = [
                (
                    _section_name(mode_label, "Overview"),
                    _per_json_overview_panels(mode_label),
                ),
                (
                    _section_name(
                        mode_label,
                        "Scores (comparison)" if json_is_noobj else "Scores",
                    ),
                    _per_json_score_panels(labels_for_this),
                ),
                # Skip Scores (objects-only) in noobj bucket: every label is a noobj
                # label, so the helper would render N empty LinePlots.
                (
                    _section_name(mode_label, "Scores (objects-only)"),
                    _per_json_objects_only_score_panels(labels_for_this),
                ),
                (
                    _section_name(mode_label, "Trajectory Overlay"),
                    _per_json_overlay_panels(labels_for_this),
                ),
                (
                    _section_name(mode_label, "Video"),
                    _per_json_video_panels(labels_for_this),
                ),
            ]
            for section_name, panels in section_specs:
                if not panels:
                    continue
                per_json_sections.append(
                    ws.Section(name=section_name, panels=panels, is_open=False)
                )

    workspace = ws.Workspace(
        entity=entity,
        project=project,
        name=name,
        auto_generate_panels=False,
        sections=[training_section, *all_sections, *per_json_sections],
    )
    workspace.save()
    return workspace.url


def _resolve_entity(entity: str | None) -> str:
    """Resolve the W&B entity: explicit override, else the API default."""
    import wandb

    if entity:
        return entity
    resolved = wandb.Api().default_entity
    if not resolved:
        raise SystemExit("Could not infer --entity: pass it explicitly or configure `wandb init`.")
    return resolved


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entity",
        default=None,
        help="W&B entity (user or org). Defaults to `wandb.Api().default_entity`.",
    )
    parser.add_argument("--project", required=True)
    parser.add_argument(
        "--group_names",
        nargs="+",
        required=True,
        help="FULL per-group W&B labels actually logged (e.g. 'override_label/departure "
        "override_label__noobj/lane_change site/all site__noobj/all'). "
        "Caller-supplied list — does NOT auto-discover. Re-run the script when this list "
        "changes.",
    )
    parser.add_argument(
        "--json_names",
        nargs="+",
        default=None,
        help="Optional explicit per-JSON section list. When omitted, inferred from "
        "--group_names by stripping the '/<group>' or '__noobj/<group>' suffix. "
        "Pass explicitly only to pin section order or render a json with no groups.",
    )
    parser.add_argument(
        "--name",
        default="Closed-Loop Dashboard",
        help="Saved-view name. Pick something unique per run if you don't want earlier "
        "views overwritten (e.g. 'Closed-Loop / <run_name>').",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    url = build_closed_loop_workspace(
        _resolve_entity(args.entity),
        args.project,
        group_names=args.group_names,
        json_names=args.json_names,
        name=args.name,
    )
    print(f"Saved workspace view: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
