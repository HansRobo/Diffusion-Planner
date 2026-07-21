"""Shared types for instrumented closed-loop rollouts and grouped eval plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from scenario_generation.perf_timer import Timers
from scenario_generation.reproducer_rollout import SegmentResult


@dataclass
class StepRecord:
    """One closed-loop sim step with optional per-area metric fields.

    ``centerline_point_dist_m`` is nearest route_lanes centerline *point*
    distance in the ego frame (not a lateral / perpendicular offset).
    """

    k: int
    rec_idx: int
    area: str | None
    # Nearest route_lanes centerline *point* distance (m), not lateral offset.
    centerline_point_dist_m: float | None = None
    turn_match: bool | None = None
    clearance_m: float = float("inf")
    collision: bool = False
    neighbor_violation: bool = False
    rb_violation: bool = False
    rb_dist_m: float = float("inf")
    png_path: Path | None = None


@dataclass
class BagRolloutResult:
    """One full-route closed-loop pass over a bag (sequence directory)."""

    route_key: str
    bag_name: str
    segment: SegmentResult
    png_dir: Path | None = None
    timers: Timers | None = None

    @property
    def steps(self) -> list[StepRecord]:
        return self.segment.steps

    @property
    def n_steps_run(self) -> int:
        return int(self.segment.metrics.get("n_steps_run", 0))

    @property
    def terminated(self) -> str:
        return str(self.segment.metrics.get("terminated", ""))
