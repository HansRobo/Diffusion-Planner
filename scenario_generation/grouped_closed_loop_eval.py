"""Grouped closed-loop validation (scenario_classification_json episodes).

Shared by the standalone CLI and per-epoch training validation in ``train.py``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from scenario_generation.metrics.cl_route_by_area import (
    RouteRolloutResult,
    aggregate_area_metrics,
    build_area_video,
    run_full_route_rollout,
)
from scenario_generation.metrics.group_report import (
    aggregate_segment_rows,
    write_metrics_summary,
    write_results_table,
)
from scenario_generation.route_timeline import RouteTimeline, _frame_index
from scenario_generation.scenario_classification import (
    episodes_from_entry,
    load_area_metric_groups,
    load_classification_json,
    time_series_from_doc,
    validate_classification_npz_root,
)


@torch.no_grad()
def run_grouped_closed_loop_eval(
    model,
    model_args,
    npz_root: Path | str,
    classification_json: Path | str,
    out_dir: Path | str,
    *,
    device: str = "cuda",
    near_miss_thresh: float = 0.3,
    search_radius: float = 1.5,
    warmup_steps: int = 0,
    unstick_after: int = 300,
    unstick_advance_m: float = 2.5,
    unstick_radius_mult: float = 3.0,
    unstick_teleport_after: int = 300,
    draw_every: int = 8,
    fps: float = 10.0,
    areas: list[str] | None = None,
    verbose: bool = True,
) -> dict:
    """Run grouped-by-area closed-loop eval; return summary for disk + wandb.

    One full-route rollout per bag in the classification JSON, then per-episode
    metrics (non-``is_skipped`` labeled frames) and continuous-span videos.
    """
    npz_root = Path(npz_root).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = load_classification_json(Path(classification_json))
    time_series = time_series_from_doc(doc)
    if not time_series:
        raise ValueError(
            "classification JSON has no 'time_series' — re-run classify_scenario_corpus"
        )

    area_to_metric = load_area_metric_groups(doc)
    allowed_areas: set[str] | None = set(areas) if areas else None
    expected_episodes = sum(
        1
        for entry in time_series.values()
        for episode in episodes_from_entry(entry)
        if (allowed_areas is None or str(episode["area"]) in allowed_areas)
        and area_to_metric.get(str(episode["area"])) is not None
    )

    rollouts: dict[str, RouteRolloutResult] = {}
    timelines: dict[str, RouteTimeline] = {}
    episodes_by_bag: dict[str, list[dict]] = {}
    missing_bags: list[str] = []

    t0 = time.perf_counter()

    for warning in validate_classification_npz_root(doc, npz_root):
        if verbose:
            print(f"Warning: {warning}")

    for bag_name, entry in sorted(time_series.items()):
        seq_dir = npz_root / bag_name
        if not seq_dir.is_dir():
            missing_bags.append(bag_name)
            if verbose:
                print(f"Skip bag {bag_name}: not under {npz_root}")
            continue
        paths = sorted(seq_dir.glob("*.npz"), key=_frame_index)
        if len(paths) < 2:
            if verbose:
                print(f"Skip bag {bag_name}: fewer than 2 NPZ frames")
            continue

        route_key = str(entry["route_key"])
        episodes = episodes_from_entry(entry)
        episodes_by_bag[bag_name] = episodes
        tl = RouteTimeline(paths, sidecar_dir=npz_root)
        png_dir = out_dir / "rollouts" / bag_name
        if verbose:
            print(
                f"Full-route rollout: {bag_name} ({route_key}), "
                f"{len(paths)} frames, {len(episodes)} episodes..."
            )
        rollout = run_full_route_rollout(
            model,
            model_args,
            tl,
            route_key,
            bag_name,
            episodes,
            area_to_metric,
            png_dir,
            device=device,
            near_miss_thresh=near_miss_thresh,
            search_radius=search_radius,
            warmup_steps=warmup_steps,
            unstick_after=unstick_after,
            unstick_advance_m=unstick_advance_m,
            unstick_radius_mult=unstick_radius_mult,
            unstick_teleport_after=unstick_teleport_after,
            draw_every=draw_every,
        )
        rollouts[bag_name] = rollout
        timelines[bag_name] = tl
        if verbose:
            print(
                f"  done: steps={rollout.n_steps_run} terminated={rollout.terminated} "
                f"episodes={len(episodes)}"
            )

    all_rows: list[dict] = []
    span_counts: dict[str, int] = {}
    video_mp4s: list[Path] = []

    for bag_name, episodes in episodes_by_bag.items():
        rollout = rollouts.get(bag_name)
        if rollout is None:
            continue
        tl = timelines[bag_name]
        route_key = rollout.route_key

        for ep in episodes:
            area_name = str(ep["area"])
            if allowed_areas is not None and area_name not in allowed_areas:
                continue
            metric_group = area_to_metric.get(area_name)
            if metric_group is None:
                continue
            span_key = f"{bag_name}:{area_name}"
            span_index = span_counts.get(span_key, 0)
            span_counts[span_key] = span_index + 1

            row = aggregate_area_metrics(
                rollout.steps,
                area_name,
                metric_group,
                route_key,
                bag_name,
                tl,
                labeled_ranges=ep["labeled_ranges"],
                video_start_idx=int(ep["video_start_idx"]),
                video_end_idx=int(ep["video_end_idx"]),
                span_index=span_index,
                near_miss_thresh=near_miss_thresh,
            )
            if row is None:
                continue

            video_root = out_dir / "videos" / metric_group / area_name
            video_root.mkdir(parents=True, exist_ok=True)
            seg_tag = row["segment"].strip("[]").replace(",", "_").replace(" ", "")
            mp4 = video_root / f"{bag_name}_{span_index}_{seg_tag}.mp4"
            if build_area_video(
                rollout,
                mp4,
                fps,
                video_start_idx=int(ep["video_start_idx"]),
                video_end_idx=int(ep["video_end_idx"]),
            ):
                row["video_path"] = str(mp4.relative_to(out_dir))
                video_mp4s.append(mp4)
            all_rows.append(row)
            if verbose:
                print(
                    f"  [{area_name}] {bag_name} span#{span_index} {row['segment']} "
                    f"steps={row['n_steps_run']} coll={row.get('collision_steps', 0)}"
                )

    grouped_summary = aggregate_segment_rows(all_rows)
    area_names = sorted({r["area_name"] for r in all_rows})
    for area_name in area_names:
        area_rows = [r for r in all_rows if r["area_name"] == area_name]
        write_metrics_summary(
            aggregate_segment_rows(area_rows),
            out_dir / "by_area" / f"{area_name}_metrics_summary.json",
        )

    write_results_table(all_rows, out_dir / "results_table.csv")
    write_metrics_summary(grouped_summary, out_dir / "metrics_summary.json")

    elapsed_sec = time.perf_counter() - t0
    summary = {
        "mode": "grouped",
        "classification_json": str(Path(classification_json).resolve()),
        "npz_root": str(npz_root),
        "near_miss_thresh": near_miss_thresh,
        "n_episodes": len(all_rows),
        "n_episodes_expected": expected_episodes,
        "episode_report_fraction": len(all_rows) / max(expected_episodes, 1),
        "n_bags": len(rollouts),
        "n_bags_expected": len(time_series),
        "bag_report_fraction": len(rollouts) / max(len(time_series), 1),
        "missing_bags": missing_bags,
        "elapsed_sec": elapsed_sec,
        "grouped_summary": grouped_summary,
        "segments": all_rows,
        "video_mp4s": video_mp4s,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                k: v
                for k, v in summary.items()
                if k not in ("segments", "video_mp4s", "grouped_summary")
            }
            | {"grouped_summary": grouped_summary},
            f,
            indent=2,
            default=str,
        )
    return summary
