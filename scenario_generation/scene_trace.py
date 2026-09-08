"""Compact, browser-oriented scene trace artifacts for scenario_sim rollouts."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

SCENE_TRACE_VERSION = 1


def _points(points: np.ndarray) -> list[list[float]]:
    return [[round(float(x), 3), round(float(y), 3)] for x, y in points[:, :2]]


def asset_dir(run_dir: Path) -> Path:
    """Where the map assets for the cases under ``run_dir`` live.

    One map serves every case that ran on it, so the assets sit beside the case directories.
    Writer and export both call this; a disagreement is silent -- the export finds no map.
    """
    return Path(run_dir) / "scene_maps"


def write_map_asset(builder, asset_dir: Path) -> tuple[str, Path]:
    """Write one immutable map asset, shared by every case using this map.

    Keyed by a digest of the geometry, not a path, so it survives a copy to another host.
    """
    lanes = []
    for lane_id, lane in sorted(builder._cache.items()):
        lanes.append(
            {
                "id": int(lane_id),
                "center": _points(lane.raw_centerline),
                "left": _points(lane.raw_left),
                "right": _points(lane.raw_right),
            }
        )
    borders = [_points(points) for points in builder.road_border_polylines() if len(points) >= 2]
    payload = {"version": SCENE_TRACE_VERSION, "lanes": lanes, "road_borders": borders}
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
    ref = hashlib.sha256(encoded).hexdigest()[:16]
    asset_dir.mkdir(parents=True, exist_ok=True)
    destination = asset_dir / f"{ref}.json.gz"
    if destination.exists():
        return ref, destination
    fd, temporary = tempfile.mkstemp(prefix=f".{ref}.", suffix=".tmp", dir=asset_dir)
    try:
        with os.fdopen(fd, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
                zipped.write(encoded)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            pass
    finally:
        Path(temporary).unlink(missing_ok=True)
    return ref, destination


class SceneTraceWriter:
    """Stream per-step scene snapshots without importing matplotlib."""

    def __init__(
        self, output_path: Path, *, map_ref: str, route: list[list[float]], goal: list[float]
    ):
        self.output_path = output_path
        self._file = gzip.open(output_path, "wt", encoding="utf-8")
        self._agents: dict[str, dict[str, Any]] = {}
        self._header = {
            "event": "header",
            "version": SCENE_TRACE_VERSION,
            "map_ref": map_ref,
            "route": route,
            "goal": goal,
            "agents": self._agents,
        }

    def write_frame(
        self,
        step: int,
        scene,
        plan_world: np.ndarray,
        *,
        clearance: float | None,
        collision: bool | None,
    ) -> None:
        agents = []
        for agent in scene.agents:
            self._agents.setdefault(
                agent.id,
                {
                    "type": str(agent.agent_type),
                    "length": round(float(agent.length), 3),
                    "width": round(float(agent.width), 3),
                    "wheelbase": round(float(agent.wheelbase), 3),
                },
            )
            x, y = agent.current_position
            agents.append(
                [
                    agent.id,
                    round(float(x), 3),
                    round(float(y), 3),
                    round(float(agent.current_heading), 4),
                ]
            )
        frame: dict[str, Any] = {
            "k": step,
            "agents": agents,
            "plan": _points(plan_world),
            "clearance_m": None if clearance is None else round(clearance, 4),
            "collision": collision,
        }
        self._file.write(json.dumps(frame, separators=(",", ":"), ensure_ascii=False) + "\n")

    def close(self, terminated_reason: str) -> None:
        # Header is written last so the agent dictionary includes late spawns; readers accept
        # this trailer-style metadata and can begin frame parsing before it arrives.
        self._file.write(
            json.dumps(
                {**self._header, "terminated": terminated_reason},
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
        )
        self._file.close()
