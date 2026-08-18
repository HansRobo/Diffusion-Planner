"""Closed-loop evaluation over OpenSCENARIO scenarios, on the shared evaluation framework.

The ``scenario_sim`` sibling of :class:`~scenario_generation.closed_loop_evaluation.
FullRouteClosedLoopEvaluation`: same job discovery -> shard -> run -> summarize workflow, same
``segments.jsonl`` / ``tdigests.jsonl`` / ``summary.json`` contract, same DDP merge discipline.
Only ``run_job`` differs, because only the rollout differs.

Parallelism comes from DDP sharding, one scenario at a time per rank, rather than from a parent
fanning out subprocess workers. That is what lets the model, its compiled graphs and the parsed
maps stay resident: ``SimulatorCore`` is a static singleton, but the rollout isolates it in a
child process (``RolloutConfig.sim_in_subprocess``), so the singleton no longer forces the
planner to be per-scenario too.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from scenario_generation.closed_loop_eval import (
    _scenario_key as scenario_key,
)
from scenario_generation.closed_loop_eval import (
    aggregate,
    enumerate_scenarios,
    format_summary_lines,
    metrics_for_json,
)
from scenario_generation.closed_loop_evaluation import (
    ClosedLoopEvalConfig,
    ClosedLoopEvaluation,
    ClosedLoopJob,
    JobRunResult,
)
from scenario_generation.gui.lanelet_scene_builder import LaneletSceneBuilder
from scenario_generation.perf_timer import Timers
from scenario_generation.scenario_sim_metrics import failed_segment_row
from scenario_generation.scenario_sim_rollout import (
    STRONG_BRAKE_MPS2,
    RolloutConfig,
    run_scenario_sim_rollout,
)
from scenario_generation.scenario_sim_route import map_from_osc


@dataclass
class ScenarioSimJob(ClosedLoopJob):
    """One OpenSCENARIO scenario."""

    osc_path: Path = field(default_factory=Path)
    # None means the scenario's own RoadNetwork/LogicFile, which is where the C++ interpreter
    # reads it from; an override is only for testing a substitute map.
    map_path: str | None = None


class ScenarioSimClosedLoopEvaluation(ClosedLoopEvaluation):
    """Closed-loop over every ``.xosc`` under ``scenario_root``, one row per scenario."""

    mode = "scenario_sim"

    def __init__(
        self,
        model,
        model_args,
        config: ClosedLoopEvalConfig,
        scenario_root: Path | str,
        *,
        rollout_config: RolloutConfig | None = None,
        map_path: Path | str | None = None,
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
        self.scenario_root = Path(scenario_root)
        self.map_path = str(map_path) if map_path is not None else None
        # The interpreter must be out of this process: it is the singleton, and this process is
        # about to run many scenarios. Forced here rather than asked of the caller, so a plain
        # RolloutConfig cannot quietly wedge the second scenario.
        self.rollout_config = replace(
            rollout_config or RolloutConfig(), sim_in_subprocess=True
        )
        # Parsed maps are reused across scenarios that declare the same one. Per rank, since a
        # rank is a process and a lanelet2 map is not cheap to hold twice.
        self._builders: dict[str, LaneletSceneBuilder] = {}

    @classmethod
    def from_checkpoint(
        cls,
        model_path: Path | str,
        scenario_root: Path | str,
        config: ClosedLoopEvalConfig,
        *,
        rollout_config: RolloutConfig | None = None,
        map_path: Path | str | None = None,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
    ) -> ScenarioSimClosedLoopEvaluation:
        model, model_args = cls.load_model_pair(model_path, config.params.device)
        return cls(
            model,
            model_args,
            config,
            scenario_root,
            rollout_config=rollout_config,
            map_path=map_path,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
        )

    @property
    def near_miss_thresh(self) -> float:
        """The rollout's threshold, not the reproducer params': it is baked into every row."""
        return self.rollout_config.near_miss_thresh

    def discover_jobs(self) -> list[ScenarioSimJob]:
        scenarios = enumerate_scenarios(self.scenario_root)
        return [
            ScenarioSimJob(
                job_id=scenario_key(scenario, self.scenario_root),
                osc_path=scenario,
                map_path=self.map_path,
            )
            for scenario in scenarios
        ]

    def _builder_for(self, osc_path: Path, map_override: str | None) -> tuple[str, Any]:
        map_path = map_override or map_from_osc(osc_path)
        if map_path not in self._builders:
            self._builders[map_path] = LaneletSceneBuilder(map_path)
        return map_path, self._builders[map_path]

    def run_job(
        self,
        job: ClosedLoopJob,
        *,
        segments_file=None,
        digest_file=None,
    ) -> JobRunResult:
        assert isinstance(job, ScenarioSimJob)
        work = self.out_dir / job.job_id
        work.mkdir(parents=True, exist_ok=True)
        timers = Timers() if self.config.profile else None
        t_case = time.perf_counter()
        try:
            map_path, builder = self._builder_for(job.osc_path, job.map_path)
            metrics = run_scenario_sim_rollout(
                self.model,
                self.model_args,
                job.osc_path,
                work,
                map_path=map_path,
                config=self.rollout_config,
                device=self.config.params.device,
                verbose=self.config.verbose,
                timers=timers,
                builder=builder,
            )
        except BaseException as e:  # noqa: BLE001 - one scenario must not end the rank's shard
            # A schema-complete failure row rather than a propagated exception: aggregate raises
            # on a missing block, so letting this escape would discard every scenario this rank
            # already finished. The reason is preserved on the row and on disk.
            (work / "error.txt").write_text(f"{type(e).__name__}: {e}\n")
            metrics = failed_segment_row(f"{type(e).__name__}: {e}", self.near_miss_thresh)
            map_path = job.map_path

        row: dict = {"route": job.job_id, **metrics}
        row["map_path"] = str(map_path) if map_path else None
        row["wall_s"] = round(time.perf_counter() - t_case, 3)
        if timers is not None:
            row["timing"] = timers.as_dict()

        # The rollout renders into its own directory, so the video is named after the row's
        # identity and lands beside every other route's -- which is what lets all ranks share
        # one out_dir and what the report globs for.
        seg_mp4 = self.maybe_build_mp4(work, self.out_dir / f"{job.job_id}.mp4")
        self.stream_row(row, segments_file, digest_file)
        return JobRunResult(rows=[row], video_mp4s=[seg_mp4] if seg_mp4 else [])

    def on_job_complete(
        self,
        job: ClosedLoopJob,
        partial: JobRunResult,
        index: int,
        total: int,
    ) -> None:
        if not self.config.verbose or not partial.rows:
            return
        row = partial.rows[0]
        print(
            f"[{index + 1}/{total}] {job.job_id} -> {row.get('terminated')}"
            f"/{row.get('result_kind', '')} steps={row.get('n_steps_run')} "
            f"coll={row.get('object', {}).get('collision_steps')} "
            f"coord_ok={row.get('coord_check_ok')} {row.get('wall_s')}s"
        )

    def build_summary(self, result: JobRunResult, *, elapsed_sec: float) -> dict:
        summary = aggregate(
            result.rows,
            self.near_miss_thresh,
            strong_brake_mps2=STRONG_BRAKE_MPS2,
        )
        summary["scenario_root"] = str(self.scenario_root)
        summary["map_override"] = self.map_path
        # Each scenario resolves its own map, so record the set actually used: for a multi-map
        # suite a scalar map path would be a lie.
        summary["maps_used"] = sorted({m for r in result.rows if (m := r.get("map_path"))})
        # Derived from the rows rather than from extras, which a DDP merge does not reconstruct.
        summary["n_scenarios"] = len({r["route"] for r in result.rows})
        summary["elapsed_sec"] = elapsed_sec
        summary["video_mp4s"] = result.video_mp4s
        summary["segments"] = result.rows
        return summary

    def write_artifacts(self, summary: dict, result: JobRunResult) -> None:
        with open(self.out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(
                metrics_for_json(
                    {k: v for k, v in summary.items() if k not in ("video_mp4s", "segments")}
                ),
                f,
                indent=4,
            )

    def print_summary(self, summary: dict) -> None:
        print(
            f"\n=== scenario_sim closed-loop: {summary['n_segments']} scenarios in "
            f"{summary['elapsed_sec']:.1f}s ==="
        )
        for line in format_summary_lines(summary):
            print(line)

