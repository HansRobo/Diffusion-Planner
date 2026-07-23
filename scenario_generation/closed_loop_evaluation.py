"""Class-based closed-loop evaluation framework.

``ClosedLoopEvaluation`` defines the shared workflow:

  1. ``discover_jobs`` — what to evaluate (routes, bags, classification dates, …)
  2. ``shard_jobs`` — DDP round-robin assignment (override for custom sharding)
  3. ``run_job`` — rollout for one job unit (segments via ``render_segment``, etc.)
  4. ``build_summary`` / ``write_artifacts`` / ``print_summary`` — serialize + terminal output

``FullRouteClosedLoopEvaluation`` is the legacy segment-wise evaluator (``segments.jsonl``).
Grouped-by-area eval subclasses this in a follow-up PR.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from scenario_generation.closed_loop_ddp import shard_items
from scenario_generation.closed_loop_eval import aggregate, build_mp4, enumerate_routes
from scenario_generation.perf_timer import Timers
from scenario_generation.reproducer_rollout import render_segment
from scenario_generation.route_timeline import RouteTimeline


@dataclass
class RolloutParams:
    """Knobs shared by full-route segment eval and grouped bag rollouts."""

    device: str = "cuda"
    near_miss_thresh: float = 0.5
    search_radius: float = 1.5
    warmup_steps: int = 0
    unstick_after: int = 300
    unstick_advance_m: float = 2.5
    unstick_radius_mult: float = 10.0
    unstick_teleport_after: int = 300
    draw_every: int = 8
    replan_interval: int = 10
    tracker_mode: str = "mpc"
    profile_sync_gpu: bool = False
    neighbor_history_mode: str = "recorded"
    # Early-abort a badly-diverged segment instead of burning the full step budget on a
    # rollout that will never recover (e.g. an undertrained model driving off-lane). Set well
    # above the unstick_* knobs above: unstick snaps the ego back onto GT to let a merely-stuck
    # rollout continue; abort gives up on a rollout unstick can't save. 0 = disabled for either
    # knob. See ``reproducer_rollout.render_segment`` for the exact trigger condition.
    abort_deviation_m: float = 8.0
    abort_after: int = 30
    abort_max_snaps: int = 0

    def render_kwargs(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "near_miss_thresh": self.near_miss_thresh,
            "search_radius": self.search_radius,
            "warmup_steps": self.warmup_steps,
            "unstick_after": self.unstick_after,
            "unstick_advance_m": self.unstick_advance_m,
            "unstick_radius_mult": self.unstick_radius_mult,
            "unstick_teleport_after": self.unstick_teleport_after,
            "draw_every": self.draw_every,
            "replan_interval": self.replan_interval,
            "tracker_mode": self.tracker_mode,
            "profile_sync_gpu": self.profile_sync_gpu,
            "neighbor_history_mode": self.neighbor_history_mode,
            "abort_deviation_m": self.abort_deviation_m,
            "abort_after": self.abort_after,
            "abort_max_snaps": self.abort_max_snaps,
        }


@dataclass
class ClosedLoopEvalConfig:
    """Shared configuration for any closed-loop evaluation mode."""

    out_dir: Path
    params: RolloutParams
    fps: float = 10.0
    verbose: bool = True
    profile: bool = False
    max_jobs: int | None = None


@dataclass
class ClosedLoopJob:
    """One schedulable unit (route, bag, classification date, …)."""

    job_id: str


@dataclass
class JobRunResult:
    """Accumulated rows + artifacts from one or more jobs on this rank."""

    rows: list[dict] = field(default_factory=list)
    video_mp4s: list[Path] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)
    elapsed_sec: float = 0.0


def resolve_default_out_dir(model_path: Path | str, out_dir: Path | str | None = None) -> Path:
    """Default output tree: ``<checkpoint_dir>/closed_loop/<timestamp>/``."""
    if out_dir is not None:
        return Path(out_dir)
    model_path = Path(model_path)
    return model_path.parent / "closed_loop" / datetime.now().strftime("%Y%m%d_%H%M%S")


class ClosedLoopEvaluation(ABC):
    """Template-method closed-loop evaluation workflow."""

    mode: str = "base"

    def __init__(
        self,
        model,
        model_args,
        config: ClosedLoopEvalConfig,
        *,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
    ) -> None:
        self.model = model
        self.model_args = model_args
        self.config = config
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.out_dir = Path(config.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def load_model_pair(model_path: Path | str, device: str):
        """Load a torch checkpoint or ONNX export (``args.json`` next to the file)."""
        from scenario_generation.simulate import load_model, load_onnx_model

        model_path = Path(model_path)
        if model_path.suffix == ".onnx":
            return load_onnx_model(model_path, device)
        return load_model(model_path, device)

    def run(self) -> dict:
        """Discover jobs, optionally shard for DDP, execute, summarize, and persist."""
        t0 = time.perf_counter()
        jobs = self.discover_jobs()
        if self.config.max_jobs is not None:
            jobs = jobs[: self.config.max_jobs]
        jobs = self.shard_jobs(jobs)
        if self.config.verbose and jobs and self.ddp_world_size > 1:
            print(
                f"DDP rank {self.ddp_rank}/{self.ddp_world_size}: "
                f"{len(jobs)} job(s) -> {[j.job_id for j in jobs]}"
            )

        result = self.execute_jobs(jobs)
        elapsed_sec = time.perf_counter() - t0

        if self.ddp_world_size > 1:
            return self.finalize_ddp(result, elapsed_sec=elapsed_sec)

        summary = self.build_summary(result, elapsed_sec=elapsed_sec)
        summary["mode"] = self.mode
        summary["ddp_rank"] = self.ddp_rank
        summary["ddp_world_size"] = self.ddp_world_size
        self.write_artifacts(summary, result)
        if self.config.verbose:
            self.print_summary(summary)
        return summary

    def run_distributed(self) -> dict:
        """Run evaluation; under DDP, barrier then rank-0 ``merge_ddp_shards``."""
        partial = self.run()
        if self.ddp_world_size <= 1:
            return partial
        try:
            import torch.distributed as dist
        except ImportError:
            return partial
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if self.ddp_rank != 0:
            return {}
        if partial.get("ddp_shard"):
            return self.merge_ddp_shards(self.ddp_world_size)
        return partial

    def shard_jobs(self, jobs: list[ClosedLoopJob]) -> list[ClosedLoopJob]:
        return shard_items(jobs, self.ddp_rank, self.ddp_world_size)

    def initial_job_extras(self) -> dict[str, Any]:
        return {}

    def merge_job_result(self, merged: JobRunResult, partial: JobRunResult) -> None:
        merged.rows.extend(partial.rows)
        merged.video_mp4s.extend(partial.video_mp4s)
        for key, value in partial.extras.items():
            if key in merged.extras and isinstance(merged.extras[key], list):
                merged.extras[key].extend(value)
            elif key in merged.extras and isinstance(merged.extras[key], dict):
                merged.extras[key].update(value)
            elif key in merged.extras and isinstance(merged.extras[key], int):
                merged.extras[key] = int(merged.extras[key]) + int(value)
            else:
                merged.extras[key] = value

    def on_job_complete(
        self,
        job: ClosedLoopJob,
        partial: JobRunResult,
        index: int,
        total: int,
    ) -> None:
        """Hook for per-job progress (subclasses print route/bag summaries here)."""

    def execute_jobs(self, jobs: list[ClosedLoopJob]) -> JobRunResult:
        merged = JobRunResult(extras=self.initial_job_extras())
        for ri, job in enumerate(jobs):
            partial = self.run_job(job)
            self.merge_job_result(merged, partial)
            self.on_job_complete(job, partial, ri, len(jobs))
        return merged

    def ddp_shard_extras(self, result: JobRunResult) -> dict[str, Any]:
        return dict(result.extras)

    def finalize_ddp(self, result: JobRunResult, *, elapsed_sec: float) -> dict:
        from scenario_generation.closed_loop_ddp import write_eval_shard

        write_eval_shard(
            self.out_dir,
            self.ddp_rank,
            mode=self.mode,
            rows=result.rows,
            video_mp4s=result.video_mp4s,
            elapsed_sec=elapsed_sec,
            extras=self.ddp_shard_extras(result),
            profile=self.config.profile,
        )
        return self.ddp_partial_summary(result, elapsed_sec=elapsed_sec)

    def ddp_partial_summary(self, result: JobRunResult, *, elapsed_sec: float) -> dict:
        return {
            "mode": self.mode,
            "ddp_shard": True,
            "ddp_rank": self.ddp_rank,
            "ddp_world_size": self.ddp_world_size,
            "n_rows": len(result.rows),
            "elapsed_sec": elapsed_sec,
        }

    def collect_ddp_shards(self, world_size: int) -> JobRunResult:
        """Load per-rank shard JSON written under ``out_dir/ddp_shards/``."""
        merged = JobRunResult()
        for rank in range(world_size):
            path = self.out_dir / "ddp_shards" / f"rank_{rank:03d}.json"
            if not path.is_file():
                continue
            shard = json.loads(path.read_text(encoding="utf-8"))
            merged.rows.extend(shard.get("rows") or [])
            merged.video_mp4s.extend(Path(p) for p in shard.get("video_mp4s") or [])
            merged.elapsed_sec = max(merged.elapsed_sec, float(shard.get("elapsed_sec", 0.0)))
            self.merge_job_result(merged, JobRunResult(extras=shard.get("extras") or {}))
        return merged

    def merge_ddp_shards(self, world_size: int) -> dict:
        """Rank-0: merge shard files, persist artifacts, and return the final summary."""
        result = self.collect_ddp_shards(world_size)
        self.prepare_ddp_merge_artifacts(result)
        summary = self.build_summary(result, elapsed_sec=result.elapsed_sec)
        summary["mode"] = self.mode
        summary["ddp_shard"] = True
        summary["ddp_world_size"] = world_size
        self.write_artifacts(summary, result)
        if self.config.verbose:
            self.print_summary(summary)
        return summary

    def prepare_ddp_merge_artifacts(self, result: JobRunResult) -> None:
        """Hook for mode-specific files written during rank-0 DDP merge."""

    @abstractmethod
    def discover_jobs(self) -> list[ClosedLoopJob]:
        raise NotImplementedError

    @abstractmethod
    def run_job(self, job: ClosedLoopJob) -> JobRunResult:
        raise NotImplementedError

    @abstractmethod
    def build_summary(self, result: JobRunResult, *, elapsed_sec: float) -> dict:
        raise NotImplementedError

    @abstractmethod
    def write_artifacts(self, summary: dict, result: JobRunResult) -> None:
        raise NotImplementedError

    @abstractmethod
    def print_summary(self, summary: dict) -> None:
        raise NotImplementedError


@dataclass
class FullRouteRouteJob(ClosedLoopJob):
    """One route (bag-prefix group) under an NPZ root."""

    npz_root: Path = field(default_factory=Path)
    route_key: str = ""
    route_paths: list[Path] = field(default_factory=list)
    seg_len: int = 100_000


class FullRouteClosedLoopEvaluation(ClosedLoopEvaluation):
    """Segment-wise full-route closed-loop over every route under ``npz_root``."""

    mode = "full"

    def __init__(
        self,
        model,
        model_args,
        config: ClosedLoopEvalConfig,
        npz_root: Path | str,
        *,
        seg_len: int = 100_000,
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
        self.npz_root = Path(npz_root)
        self.seg_len = seg_len

    @classmethod
    def from_checkpoint(
        cls,
        model_path: Path | str,
        npz_root: Path | str,
        config: ClosedLoopEvalConfig,
        *,
        seg_len: int = 100_000,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
    ) -> FullRouteClosedLoopEvaluation:
        model, model_args = cls.load_model_pair(model_path, config.params.device)
        return cls(
            model,
            model_args,
            config,
            npz_root,
            seg_len=seg_len,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
        )

    def discover_jobs(self) -> list[FullRouteRouteJob]:
        routes = enumerate_routes(self.npz_root)
        return [
            FullRouteRouteJob(
                job_id=key,
                npz_root=self.npz_root,
                route_key=key,
                route_paths=routes[key],
                seg_len=self.seg_len,
            )
            for key in sorted(routes)
        ]

    def initial_job_extras(self) -> dict[str, Any]:
        return {"route_keys": []}

    def execute_jobs(self, jobs: list[ClosedLoopJob]) -> JobRunResult:
        """Stream per-segment rows to ``segments.jsonl`` while jobs run."""
        merged = JobRunResult(extras={"route_keys": []})
        segments_path = self.out_dir / "segments.jsonl"
        with segments_path.open("w", encoding="utf-8") as fout:
            for ri, job in enumerate(jobs):
                assert isinstance(job, FullRouteRouteJob)
                partial = self.run_job(job, segments_file=fout)
                merged.rows.extend(partial.rows)
                merged.video_mp4s.extend(partial.video_mp4s)
                merged.extras["route_keys"].append(job.route_key)
                self.on_job_complete(job, partial, ri, len(jobs))
        return merged

    def run_job(
        self,
        job: ClosedLoopJob,
        *,
        segments_file=None,
    ) -> JobRunResult:
        assert isinstance(job, FullRouteRouteJob)
        params = self.config.params
        timers = Timers() if self.config.profile else None
        tl = RouteTimeline(job.route_paths, sidecar_dir=job.npz_root, timers=timers)
        rows: list[dict] = []
        video_mp4s: list[Path] = []

        for start, end in tl.iter_segments(job.seg_len):
            png_dir = self.out_dir / f"{job.route_key}_{start}_{end}"
            metrics = render_segment(
                self.model,
                self.model_args,
                tl,
                start,
                end,
                png_dir,
                **params.render_kwargs(),
            )
            row = {"route": job.route_key, **metrics}
            if segments_file is not None:
                segments_file.write(json.dumps(row, default=float) + "\n")
                segments_file.flush()
            rows.append(row)

            if not any(png_dir.glob("*.png")):
                if self.config.verbose:
                    print(
                        f"  [{job.route_key}] segment [{start},{end}] -> 0 frames, no video"
                    )
                continue
            seg_mp4 = self.out_dir / f"{job.route_key}_{start}_{end}.mp4"
            build_mp4(png_dir, seg_mp4, self.config.fps)
            video_mp4s.append(seg_mp4)
            if self.config.verbose:
                print(
                    f"  [{job.route_key}] segment [{start},{end}] -> {seg_mp4.name}  "
                    f"coll={metrics['n_collision_steps']} near={metrics['n_near_miss_steps']} "
                    f"min_clr={metrics['min_clearance']:.3f}"
                )

        extras: dict[str, Any] = {}
        if timers is not None:
            extras["timers"] = timers
        return JobRunResult(rows=rows, video_mp4s=video_mp4s, extras=extras)

    def on_job_complete(
        self,
        job: ClosedLoopJob,
        partial: JobRunResult,
        index: int,
        total: int,
    ) -> None:
        assert isinstance(job, FullRouteRouteJob)
        if self.config.verbose:
            print(
                f"[{index + 1}/{total}] {job.route_key}: "
                f"{len(partial.video_mp4s)} segment video(s)"
            )

    def build_summary(self, result: JobRunResult, *, elapsed_sec: float) -> dict:
        summary = aggregate(result.rows, self.config.params.near_miss_thresh)
        summary["npz_root"] = str(self.npz_root)
        summary["n_routes"] = len(set(result.extras.get("route_keys") or []))
        summary["elapsed_sec"] = elapsed_sec
        summary["video_mp4s"] = result.video_mp4s
        summary["segments"] = result.rows
        return summary

    def write_artifacts(self, summary: dict, result: JobRunResult) -> None:
        with open(self.out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {k: v for k, v in summary.items() if k not in ("video_mp4s", "segments")},
                f,
                indent=4,
            )

    def print_summary(self, summary: dict) -> None:
        near_miss_thresh = self.config.params.near_miss_thresh
        n_seg = summary["n_segments"]
        print(f"\n=== closed-loop validation: {n_seg} segments in {summary['elapsed_sec']:.1f}s ===")
        print(
            f"collision: {summary['n_segments_with_collision']}/{n_seg} segments "
            f"(rate {summary['collision_segment_rate']:.4f}), "
            f"{summary['total_collision_steps']} steps (rate {summary['collision_step_rate']:.6f})"
        )
        print(
            f"near-miss (<= {near_miss_thresh} m): "
            f"{summary['n_segments_with_near_miss']}/{n_seg} segments "
            f"(rate {summary['near_miss_segment_rate']:.4f}), {summary['total_near_miss_steps']} steps"
        )
        print(
            f"global_min_clearance={summary['global_min_clearance']:.3f} m  "
            f"mean_segment_min_clearance={summary['mean_segment_min_clearance']:.3f} m  "
            f"mean_segment_mean_clearance={summary['mean_segment_mean_clearance']:.3f} m"
        )
        print(f"total_snaps={summary['total_snaps']}  terminated={summary['terminated_counts']}")
        print(f"videos: per-segment <route>_<start>_<end>.mp4 in {self.out_dir}")

    def prepare_ddp_merge_artifacts(self, result: JobRunResult) -> None:
        with open(self.out_dir / "segments.jsonl", "w", encoding="utf-8") as fout:
            for row in result.rows:
                fout.write(json.dumps(row, default=float) + "\n")
