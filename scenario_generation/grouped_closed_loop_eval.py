"""Grouped-by-area closed-loop evaluation.

:class:`GroupedByAreaClosedLoopEvaluation` — one classification JSON, bags in
``discover_jobs``, instrumented rollout inlined in ``run_job``.

:class:`GroupedMultiDateClosedLoopEvaluation` — discovers classification dates
then bags (flattened jobs for DDP sharding).

:class:`ResolvedClosedLoopEvaluation` — training/CLI auto: grouped when JSON
resolves, else full-route (discovery only in ``discover_jobs``).
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from scenario_generation.closed_loop_evaluation import (
    ClosedLoopEvalConfig,
    ClosedLoopEvaluation,
    ClosedLoopJob,
    FullRouteClosedLoopEvaluation,
    JobRunResult,
    RolloutParams,
)
from scenario_generation.closed_loop_types import BagRolloutResult
from scenario_generation.metrics.group_report import (
    aggregate_segment_rows,
    write_metrics_summary,
    write_results_table,
)
from scenario_generation.metrics.grouped_by_area import (
    aggregate_episode_metrics,
    build_episode_video,
)
from scenario_generation.perf_timer import Timers
from scenario_generation.reproducer_rollout import SegmentResult, render_segment
from scenario_generation.route_timeline import RouteTimeline, _frame_index
from scenario_generation.scenario_classification import (
    ClassificationEvalJob,
    discover_classification_eval_jobs,
    episodes_from_entry,
    load_area_metric_groups,
    load_classification_json,
    merge_grouped_eval_summaries,
    time_series_from_doc,
    validate_classification_npz_root,
)


@dataclass
class GroupedBagJob(ClosedLoopJob):
    """One sequence (bag directory) listed in classification ``time_series``."""

    bag_name: str = ""
    entry: dict = field(default_factory=dict)
    npz_root: Path = field(default_factory=Path)
    classification_json: Path = field(default_factory=Path)
    date: str = ""
    dataset_key: str = ""
    job_out_dir: Path = field(default_factory=Path)


@dataclass
class GroupedClassificationParams:
    """Grouped discovery inputs (shared by multi-date and resolved evaluators)."""

    npz_root: Path
    classification_json_root: Path | None = None
    classification_json: Path | None = None
    dataset_name: str | None = None
    areas: list[str] | None = None

    def resolve_classification_jobs(self) -> list[ClassificationEvalJob]:
        cl_root = self.classification_json_root
        cl_explicit = self.classification_json
        if cl_explicit is not None and cl_explicit.is_dir():
            cl_root = cl_root or cl_explicit
            cl_explicit = None
        return discover_classification_eval_jobs(
            self.npz_root,
            classification_root=cl_root,
            explicit_path=cl_explicit,
            dataset_name=self.dataset_name,
        )


class GroupedByAreaClosedLoopEvaluation(ClosedLoopEvaluation):
    """Closed-loop eval over one classification JSON (bags = jobs)."""

    mode = "grouped"

    def __init__(
        self,
        model,
        model_args,
        config: ClosedLoopEvalConfig,
        npz_root: Path | str,
        classification_json: Path | str,
        *,
        areas: list[str] | None = None,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
    ) -> None:
        super().__init__(
            model,
            model_args,
            config,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
        )
        self.npz_root = Path(npz_root).expanduser().resolve()
        self.classification_json = Path(classification_json).expanduser().resolve()
        self.allowed_areas: set[str] | None = set(areas) if areas else None
        self._doc: dict | None = None
        self._area_to_metric_map: dict[str, str] | None = None

    @classmethod
    def from_checkpoint(
        cls,
        model_path: Path | str,
        npz_root: Path | str,
        classification_json: Path | str,
        config: ClosedLoopEvalConfig,
        *,
        areas: list[str] | None = None,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
    ) -> GroupedByAreaClosedLoopEvaluation:
        model, model_args = cls.load_model_pair(model_path, config.params.device)
        return cls(
            model,
            model_args,
            config,
            npz_root,
            classification_json,
            areas=areas,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
        )

    def _classification_doc(self) -> dict:
        if self._doc is None:
            self._doc = load_classification_json(self.classification_json)
        return self._doc

    def _area_to_metric(self) -> dict[str, str]:
        if self._area_to_metric_map is None:
            self._area_to_metric_map = load_area_metric_groups(self._classification_doc())
        return self._area_to_metric_map

    def discover_jobs(self) -> list[GroupedBagJob]:
        doc = self._classification_doc()
        time_series = time_series_from_doc(doc)
        if not time_series:
            raise ValueError(
                "classification JSON has no 'time_series' — re-run classify_scenario_corpus"
            )
        for warning in validate_classification_npz_root(doc, self.npz_root):
            if self.config.verbose:
                print(f"Warning: {warning}")
        return [
            GroupedBagJob(
                job_id=bag_name,
                bag_name=bag_name,
                entry=entry,
                npz_root=self.npz_root,
                classification_json=self.classification_json,
                job_out_dir=self.out_dir,
            )
            for bag_name, entry in sorted(time_series.items())
        ]

    def initial_job_extras(self) -> dict[str, Any]:
        return {"rollouts": {}, "per_bag_timing": [], "total_steps": 0}

    def _rollout_bag(
        self,
        tl: RouteTimeline,
        *,
        route_key: str,
        bag_name: str,
        episodes: list[dict],
        png_dir: Path,
    ) -> BagRolloutResult:
        """One instrumented full-route pass via :func:`render_segment` (no wrapper module)."""
        n = len(tl)
        if n < 2:
            empty = SegmentResult(
                metrics={"n_steps_run": 0, "terminated": "empty"},
                clearances=np.zeros(0),
                collisions=np.zeros(0, dtype=bool),
            )
            return BagRolloutResult(
                route_key=route_key, bag_name=bag_name, segment=empty, png_dir=png_dir
            )

        draw_dir = png_dir if png_dir is not None else Path("/tmp") / "_cl_rollout_no_png"
        if png_dir is not None:
            draw_dir.mkdir(parents=True, exist_ok=True)

        segment = render_segment(
            self.model,
            self.model_args,
            tl,
            0,
            n,
            draw_dir,
            goal_mode="route",
            instrument=True,
            area_episodes=episodes,
            area_to_metric_group=self._area_to_metric(),
            **self.config.params.render_kwargs(),
        )
        return BagRolloutResult(
            route_key=route_key,
            bag_name=bag_name,
            segment=segment,
            png_dir=png_dir,
            timers=segment.timers,
        )

    def run_job(self, job: ClosedLoopJob) -> JobRunResult:
        assert isinstance(job, GroupedBagJob)
        seq_dir = job.npz_root / job.bag_name
        if not seq_dir.is_dir():
            if self.config.verbose:
                print(f"Skip bag {job.bag_name}: not under {job.npz_root}")
            return JobRunResult()

        paths = sorted(seq_dir.glob("*.npz"), key=_frame_index)
        if len(paths) < 2:
            if self.config.verbose:
                print(f"Skip bag {job.bag_name}: fewer than 2 NPZ frames")
            return JobRunResult()

        route_key = str(job.entry["route_key"])
        episodes = episodes_from_entry(job.entry)
        bag_timers = Timers() if self.config.profile else None
        tl = RouteTimeline(paths, sidecar_dir=job.npz_root, timers=bag_timers)
        png_dir = job.job_out_dir / "rollouts" / job.bag_name
        if self.config.verbose:
            print(
                f"Full-route rollout: {job.bag_name} ({route_key}), "
                f"{len(paths)} frames, {len(episodes)} episodes..."
            )

        with bag_timers("rollout") if bag_timers else nullcontext():
            rollout = self._rollout_bag(
                tl,
                route_key=route_key,
                bag_name=job.bag_name,
                episodes=episodes,
                png_dir=png_dir,
            )
        if bag_timers is not None and rollout.timers is not None:
            bag_timers.merge(rollout.timers)

        rows, video_mp4s = self._postprocess_bag(
            job.bag_name,
            episodes,
            rollout,
            tl,
            out_dir=job.job_out_dir,
            timers=bag_timers,
        )
        extras: dict[str, Any] = {
            "rollouts": {job.bag_name: rollout},
            "total_steps": rollout.n_steps_run,
        }
        if bag_timers is not None:
            extras["per_bag_timing"] = [
                {
                    "bag": job.bag_name,
                    "n_steps_run": rollout.n_steps_run,
                    "stages": bag_timers.as_dict(),
                }
            ]
        return JobRunResult(rows=rows, video_mp4s=video_mp4s, extras=extras)

    def _postprocess_bag(
        self,
        bag_name: str,
        episodes: list[dict],
        rollout: BagRolloutResult,
        tl: RouteTimeline,
        *,
        out_dir: Path,
        timers: Timers | None,
    ) -> tuple[list[dict], list[Path]]:
        rows: list[dict] = []
        video_mp4s: list[Path] = []
        span_counts: dict[str, int] = {}
        area_to_metric = self._area_to_metric()
        near_miss = self.config.params.near_miss_thresh

        for ep in episodes:
            area_name = str(ep["area"])
            if self.allowed_areas is not None and area_name not in self.allowed_areas:
                continue
            metric_group = area_to_metric.get(area_name)
            if metric_group is None:
                continue
            span_key = f"{bag_name}:{area_name}"
            span_index = span_counts.get(span_key, 0)
            span_counts[span_key] = span_index + 1

            ctx = timers("aggregate_metrics") if timers else nullcontext()
            with ctx:
                row = aggregate_episode_metrics(
                    rollout.steps,
                    area_name,
                    metric_group,
                    rollout.route_key,
                    bag_name,
                    tl,
                    labeled_ranges=ep["labeled_ranges"],
                    video_start_idx=int(ep["video_start_idx"]),
                    video_end_idx=int(ep["video_end_idx"]),
                    span_index=span_index,
                    near_miss_thresh=near_miss,
                )
            if row is None:
                continue

            video_root = out_dir / "videos" / metric_group / area_name
            video_root.mkdir(parents=True, exist_ok=True)
            seg_tag = row["segment"].strip("[]").replace(",", "_").replace(" ", "")
            mp4 = video_root / f"{bag_name}_{span_index}_{seg_tag}.mp4"
            ctx = timers("build_videos") if timers else nullcontext()
            with ctx:
                built = build_episode_video(
                    rollout,
                    mp4,
                    self.config.fps,
                    video_start_idx=int(ep["video_start_idx"]),
                    video_end_idx=int(ep["video_end_idx"]),
                )
            if built:
                row["video_path"] = str(mp4.relative_to(out_dir))
                video_mp4s.append(mp4)
            rows.append(row)
            if self.config.verbose:
                print(
                    f"  [{area_name}] {bag_name} span#{span_index} {row['segment']} "
                    f"steps={row['n_steps_run']} coll={row.get('collision_steps', 0)}"
                )
        return rows, video_mp4s

    def on_job_complete(
        self,
        job: ClosedLoopJob,
        partial: JobRunResult,
        index: int,
        total: int,
    ) -> None:
        assert isinstance(job, GroupedBagJob)
        if not partial.rows and not partial.extras.get("rollouts"):
            return
        rollout = (partial.extras.get("rollouts") or {}).get(job.bag_name)
        if self.config.verbose and rollout is not None:
            print(
                f"[{index + 1}/{total}] {job.bag_name}: "
                f"{len(partial.rows)} episode row(s), "
                f"steps={rollout.n_steps_run} terminated={rollout.terminated}"
            )

    def ddp_shard_extras(self, result: JobRunResult) -> dict[str, Any]:
        return {
            "rollout_bags": sorted((result.extras.get("rollouts") or {}).keys()),
            "total_steps": int(result.extras.get("total_steps", 0)),
            "per_bag_timing": (result.extras.get("per_bag_timing") or [])
            if self.config.profile
            else [],
        }

    def ddp_partial_summary(self, result: JobRunResult, *, elapsed_sec: float) -> dict:
        return {
            "mode": "grouped",
            "ddp_shard": True,
            "ddp_rank": self.ddp_rank,
            "ddp_world_size": self.ddp_world_size,
            "n_bags": len(result.extras.get("rollouts") or {}),
            "n_episodes": len(result.rows),
            "elapsed_sec": elapsed_sec,
            "classification_json": str(self.classification_json),
            "npz_root": str(self.npz_root),
            "near_miss_thresh": self.config.params.near_miss_thresh,
        }

    def build_summary(self, result: JobRunResult, *, elapsed_sec: float) -> dict:
        rollouts = result.extras.get("rollouts") or {}
        rollout_bags = result.extras.get("rollout_bags") or sorted(rollouts.keys())
        grouped_summary = aggregate_segment_rows(result.rows)
        timing = None
        if self.config.profile:
            timing = build_grouped_timing_summary(
                result.extras.get("per_bag_timing") or [],
                int(result.extras.get("total_steps", 0)),
                elapsed_sec,
            )
        return {
            "mode": "grouped",
            "classification_json": str(self.classification_json),
            "npz_root": str(self.npz_root),
            "near_miss_thresh": self.config.params.near_miss_thresh,
            "n_episodes": len(result.rows),
            "n_bags": len(rollout_bags) if rollout_bags else len(rollouts),
            "elapsed_sec": elapsed_sec,
            "grouped_summary": grouped_summary,
            "segments": result.rows,
            "video_mp4s": result.video_mp4s,
            "timing": timing,
        }

    def write_artifacts(self, summary: dict, result: JobRunResult) -> None:
        all_rows = result.rows
        grouped_summary = summary["grouped_summary"]
        for area_name in sorted({r["area_name"] for r in all_rows}):
            area_rows = [r for r in all_rows if r["area_name"] == area_name]
            write_metrics_summary(
                aggregate_segment_rows(area_rows),
                self.out_dir / "by_area" / f"{area_name}_metrics_summary.json",
            )
        write_results_table(all_rows, self.out_dir / "results_table.csv")
        write_metrics_summary(grouped_summary, self.out_dir / "metrics_summary.json")

        timing = summary.get("timing")
        with open(self.out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    k: v
                    for k, v in summary.items()
                    if k not in ("segments", "video_mp4s", "grouped_summary", "timing")
                }
                | {"grouped_summary": grouped_summary}
                | ({"timing": timing} if timing else {}),
                f,
                indent=2,
                default=str,
            )
        if timing is not None:
            with open(self.out_dir / "timing.json", "w", encoding="utf-8") as f:
                json.dump(timing, f, indent=2)

    def print_summary(self, summary: dict) -> None:
        print(
            f"\n=== grouped closed-loop: {summary['n_episodes']} episodes, "
            f"{summary['n_bags']} bag(s) in {summary['elapsed_sec']:.1f}s ==="
        )
        totals = (summary.get("grouped_summary") or {}).get("totals") or {}
        print(
            f"steps={totals.get('n_steps_run', 0)} "
            f"coll={totals.get('collision_steps', 0)} "
            f"near_miss={totals.get('n_near_miss_steps', 0)}"
        )
        if summary.get("timing"):
            print("\n" + summary["timing"]["report"])
        print(f"results -> {self.out_dir / 'results_table.csv'}")


class GroupedMultiDateClosedLoopEvaluation(ClosedLoopEvaluation):
    """Grouped eval across all classification dates under ``npz_root``."""

    mode = "grouped"

    def __init__(
        self,
        model,
        model_args,
        config: ClosedLoopEvalConfig,
        grouped_params: GroupedClassificationParams,
        *,
        require_jobs: bool = False,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
    ) -> None:
        super().__init__(
            model,
            model_args,
            config,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
        )
        self.grouped_params = grouped_params
        self.require_jobs = require_jobs
        self._date_jobs: list[ClassificationEvalJob] = []
        self._cached_jobs: list[GroupedBagJob] | None = None

    @classmethod
    def from_checkpoint(
        cls,
        model_path: Path | str,
        out_dir: Path | str,
        grouped_params: GroupedClassificationParams,
        *,
        params: RolloutParams | None = None,
        fps: float = 10.0,
        verbose: bool = True,
        profile: bool = False,
        require_jobs: bool = True,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
    ) -> GroupedMultiDateClosedLoopEvaluation:
        params = params or RolloutParams()
        model, model_args = cls.load_model_pair(model_path, params.device)
        return cls(
            model,
            model_args,
            ClosedLoopEvalConfig(
                out_dir=Path(out_dir),
                params=params,
                fps=fps,
                verbose=verbose,
                profile=profile,
            ),
            grouped_params,
            require_jobs=require_jobs,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
        )

    def _single_json_evaluator(
        self, date_job: ClassificationEvalJob, job_out: Path
    ) -> GroupedByAreaClosedLoopEvaluation:
        return GroupedByAreaClosedLoopEvaluation(
            self.model,
            self.model_args,
            ClosedLoopEvalConfig(
                out_dir=job_out,
                params=self.config.params,
                fps=self.config.fps,
                verbose=self.config.verbose,
                profile=self.config.profile,
            ),
            date_job.npz_root,
            date_job.classification_json,
            areas=self.grouped_params.areas,
            ddp_rank=self.ddp_rank,
            ddp_world_size=self.ddp_world_size,
        )

    def discover_jobs(self) -> list[GroupedBagJob]:
        if self._cached_jobs is not None:
            return self._cached_jobs

        self._date_jobs = self.grouped_params.resolve_classification_jobs()
        if not self._date_jobs and self.require_jobs:
            raise SystemExit(
                "grouped mode: no classification JSON matched npz_root. Set "
                "--classification_json_root and ensure npz path follows valid/{date}/ layout."
            )

        all_bags: list[GroupedBagJob] = []
        for date_job in self._date_jobs:
            job_out = self.out_dir if len(self._date_jobs) == 1 else self.out_dir / date_job.date
            if self.config.verbose:
                print(
                    f"Grouped job: {date_job.dataset_key} {date_job.date} "
                    f"npz={date_job.npz_root} json={date_job.classification_json}"
                )
            single = self._single_json_evaluator(date_job, job_out)
            for bag in single.discover_jobs():
                bag.job_id = f"{date_job.date}:{bag.bag_name}"
                bag.classification_json = date_job.classification_json
                bag.date = date_job.date
                bag.dataset_key = date_job.dataset_key
                bag.job_out_dir = job_out
                all_bags.append(bag)
        self._cached_jobs = all_bags
        return all_bags

    def initial_job_extras(self) -> dict[str, Any]:
        return {"rollouts": {}, "per_bag_timing": [], "total_steps": 0}

    def run_job(self, job: ClosedLoopJob) -> JobRunResult:
        assert isinstance(job, GroupedBagJob)
        single = self._single_json_evaluator(
            ClassificationEvalJob(
                npz_root=job.npz_root,
                classification_json=job.classification_json,
                dataset_key=job.dataset_key,
                date=job.date,
            ),
            job.job_out_dir,
        )
        return single.run_job(job)

    def merge_job_result(self, merged: JobRunResult, partial: JobRunResult) -> None:
        super().merge_job_result(merged, partial)

    def ddp_shard_extras(self, result: JobRunResult) -> dict[str, Any]:
        return {
            "rollout_bags": sorted((result.extras.get("rollouts") or {}).keys()),
            "total_steps": int(result.extras.get("total_steps", 0)),
            "per_bag_timing": (result.extras.get("per_bag_timing") or [])
            if self.config.profile
            else [],
        }

    def build_summary(self, result: JobRunResult, *, elapsed_sec: float) -> dict:
        if len(self._date_jobs) <= 1:
            single = self._single_json_evaluator(
                self._date_jobs[0],
                self.out_dir,
            )
            return single.build_summary(result, elapsed_sec=elapsed_sec)

        bags_by_date: dict[str, set[str]] = {}
        for bag in self._cached_jobs or []:
            bags_by_date.setdefault(bag.date, set()).add(bag.bag_name)

        summaries: list[dict] = []
        for date_job in self._date_jobs:
            bag_names = bags_by_date.get(date_job.date, set())
            date_rows = [r for r in result.rows if r.get("bag_name") in bag_names]
            summaries.append(
                {
                    "mode": "grouped",
                    "classification_json": str(date_job.classification_json),
                    "npz_root": str(date_job.npz_root),
                    "near_miss_thresh": self.config.params.near_miss_thresh,
                    "n_episodes": len(date_rows),
                    "n_bags": len(bag_names),
                    "elapsed_sec": elapsed_sec,
                    "grouped_summary": aggregate_segment_rows(date_rows),
                    "segments": date_rows,
                    "video_mp4s": [p for p in result.video_mp4s if date_job.date in str(p)],
                    "date": date_job.date,
                    "dataset_key": date_job.dataset_key,
                }
            )
        return merge_grouped_eval_summaries(summaries)

    def write_artifacts(self, summary: dict, result: JobRunResult) -> None:
        if len(self._date_jobs) <= 1:
            self._single_json_evaluator(self._date_jobs[0], self.out_dir).write_artifacts(
                summary, result
            )
            return

        write_results_table(summary.get("segments") or [], self.out_dir / "results_table.csv")
        write_metrics_summary(
            summary.get("grouped_summary") or {},
            self.out_dir / "metrics_summary.json",
        )
        with open(self.out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    k: v
                    for k, v in summary.items()
                    if k not in ("segments", "video_mp4s", "grouped_summary")
                }
                | {"grouped_summary": summary.get("grouped_summary")},
                f,
                indent=2,
                default=str,
            )

    def print_summary(self, summary: dict) -> None:
        if len(self._date_jobs) <= 1:
            self._single_json_evaluator(self._date_jobs[0], self.out_dir).print_summary(summary)
            return
        print(
            f"\n=== grouped closed-loop: {summary['n_episodes']} episodes "
            f"({len(self._date_jobs)} date(s)) in {summary['elapsed_sec']:.1f}s ==="
        )
        print(f"results -> {self.out_dir / 'results_table.csv'}")


class ResolvedClosedLoopEvaluation(ClosedLoopEvaluation):
    """Training entry: ``discover_jobs`` picks grouped multi-date or full-route."""

    mode = "auto"

    def __init__(
        self,
        model,
        model_args,
        config: ClosedLoopEvalConfig,
        *,
        npz_root: Path | str,
        grouped_params: GroupedClassificationParams,
        seg_len: int = 100_000,
        fallback_full_route: bool = True,
        warn_on_fallback: bool = True,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
    ) -> None:
        super().__init__(
            model,
            model_args,
            config,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
        )
        self._npz_root = Path(npz_root)
        self._grouped_params = grouped_params
        self._seg_len = seg_len
        self._fallback_full_route = fallback_full_route
        self._warn_on_fallback = warn_on_fallback
        self._delegate: ClosedLoopEvaluation | None = None

    @classmethod
    def from_training(
        cls,
        model,
        model_args,
        out_dir: Path | str,
        *,
        npz_root: str,
        params: RolloutParams,
        fps: float,
        seg_len: int,
        classification_json_root: str | None,
        classification_json: str | None,
        dataset_name: str | None,
        verbose: bool,
        profile: bool,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
    ) -> ResolvedClosedLoopEvaluation:
        cl_root = Path(classification_json_root) if classification_json_root else None
        cl_explicit = Path(classification_json) if classification_json else None
        return cls(
            model,
            model_args,
            ClosedLoopEvalConfig(
                out_dir=Path(out_dir),
                params=params,
                fps=fps,
                verbose=verbose,
                profile=profile,
            ),
            npz_root=npz_root,
            grouped_params=GroupedClassificationParams(
                npz_root=Path(npz_root),
                classification_json_root=cl_root,
                classification_json=cl_explicit,
                dataset_name=dataset_name,
            ),
            seg_len=seg_len,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
        )

    def _ensure_delegate(self) -> ClosedLoopEvaluation:
        if self._delegate is not None:
            return self._delegate

        grouped_config = ClosedLoopEvalConfig(
            out_dir=self.config.out_dir / "grouped",
            params=self.config.params,
            fps=self.config.fps,
            verbose=self.config.verbose,
            profile=self.config.profile,
        )
        grouped = GroupedMultiDateClosedLoopEvaluation(
            self.model,
            self.model_args,
            grouped_config,
            self._grouped_params,
            require_jobs=False,
            ddp_rank=self.ddp_rank,
            ddp_world_size=self.ddp_world_size,
        )
        if grouped.discover_jobs():
            self._delegate = grouped
            return grouped

        if self._warn_on_fallback and self.config.verbose and self.ddp_rank == 0:
            print(
                "Warning: no classification JSON matched "
                f"npz_root={self._npz_root!r}; using full-route closed-loop"
            )

        if not self._fallback_full_route:
            self._delegate = grouped
            return grouped

        self._delegate = FullRouteClosedLoopEvaluation(
            self.model,
            self.model_args,
            self.config,
            self._npz_root,
            seg_len=self._seg_len,
            ddp_rank=self.ddp_rank,
            ddp_world_size=self.ddp_world_size,
        )
        return self._delegate

    def run(self) -> dict:
        """Run delegate workflow; preserve delegate ``mode`` (not ``auto``)."""
        delegate = self._ensure_delegate()
        summary = super().run()
        if self.mode == "auto":
            summary["mode"] = delegate.mode
        return summary

    def discover_jobs(self) -> list[ClosedLoopJob]:
        return self._ensure_delegate().discover_jobs()

    def run_job(self, job: ClosedLoopJob) -> JobRunResult:
        return self._ensure_delegate().run_job(job)

    def shard_jobs(self, jobs: list[ClosedLoopJob]) -> list[ClosedLoopJob]:
        return self._ensure_delegate().shard_jobs(jobs)

    def initial_job_extras(self) -> dict[str, Any]:
        return self._ensure_delegate().initial_job_extras()

    def merge_job_result(self, merged: JobRunResult, partial: JobRunResult) -> None:
        self._ensure_delegate().merge_job_result(merged, partial)

    def on_job_complete(
        self,
        job: ClosedLoopJob,
        partial: JobRunResult,
        index: int,
        total: int,
    ) -> None:
        self._ensure_delegate().on_job_complete(job, partial, index, total)

    def execute_jobs(self, jobs: list[ClosedLoopJob]) -> JobRunResult:
        return self._ensure_delegate().execute_jobs(jobs)

    def ddp_shard_extras(self, result: JobRunResult) -> dict[str, Any]:
        return self._ensure_delegate().ddp_shard_extras(result)

    def ddp_partial_summary(self, result: JobRunResult, *, elapsed_sec: float) -> dict:
        return self._ensure_delegate().ddp_partial_summary(result, elapsed_sec=elapsed_sec)

    def prepare_ddp_merge_artifacts(self, result: JobRunResult) -> None:
        self._ensure_delegate().prepare_ddp_merge_artifacts(result)

    def merge_ddp_shards(self, world_size: int) -> dict:
        return self._ensure_delegate().merge_ddp_shards(world_size)

    def build_summary(self, result: JobRunResult, *, elapsed_sec: float) -> dict:
        return self._ensure_delegate().build_summary(result, elapsed_sec=elapsed_sec)

    def write_artifacts(self, summary: dict, result: JobRunResult) -> None:
        self._ensure_delegate().write_artifacts(summary, result)

    def print_summary(self, summary: dict) -> None:
        self._ensure_delegate().print_summary(summary)


def build_grouped_timing_summary(
    per_bag_timing: list[dict],
    total_steps: int,
    elapsed_sec: float,
) -> dict[str, Any]:
    global_timers = Timers()
    for bag_info in per_bag_timing:
        for stage, info in bag_info.get("stages", {}).items():
            global_timers.add(stage, float(info["total_s"]), int(info["calls"]))
    stages = global_timers.as_dict()
    model_calls = stages.get("model_forward", {}).get("calls", 0)
    cached_calls = stages.get("model_forward_cached", {}).get("calls", 0)
    total_sim_steps = total_steps + cached_calls
    return {
        "elapsed_sec": elapsed_sec,
        "total_sim_steps": total_sim_steps,
        "model_forward_calls": model_calls,
        "model_forward_cached_calls": cached_calls,
        "model_forward_rate": model_calls / max(1, total_sim_steps),
        "stages": stages,
        "per_bag": per_bag_timing,
        "report": global_timers.report(total_sim_steps),
    }
