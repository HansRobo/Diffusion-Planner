"""OpenSCENARIO-driven closed-loop rollout for the Diffusion-Planner validator.

The ``scenario_sim`` sibling of the NPZ-replay path (``reproducer_rollout`` /
``closed_loop_eval.run_closed_loop_eval``): it drives the C++ OpenSCENARIO
interpreter in-process (``openscenario_python.HeadlessRunner``, SSV2_HEADLESS_EGO)
and runs the live Diffusion-Planner as the ego each tick -- read sim truth, build a
``SceneContext`` snapshot, infer, inject the plan, ``step()``, repeat.

Critical design invariant: we NEVER advance the Python scene (no
``simulate.advance_scene``). The C++ ``step()`` is the sole integrator (it advances
both ego and NPCs); Python only reads truth and injects a plan -- advancing the
Python scene too would double-simulate and diverge from sim truth.

Frame contract: the sim reports map-frame poses stored into the SceneContext as-is
(the builder is also in the map's MGRS frame); ``to_model_tensors`` re-centers onto
the ego, and the ego-frame output is mapped back with the current ego pose via
``simulate._ego_to_world`` before injection.

Scope (task #5, Phase 3 bring-up): route resolved in pure Python via
``LaneletSceneBuilder`` (shortestPath / find_route -- INTERIM; mission_planner exact
match deferred, doc 03). Inference uses the torch ``.pth`` checkpoint (training-env
path) via ``simulate.load_model``.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from scenario_generation.gui.lanelet_scene_builder import LaneletSceneBuilder
from scenario_generation.reproducer_rollout import score_step
from scenario_generation.scene_context import Agent, AgentType, SceneContext
from scenario_generation.simulate import _ego_to_world, _predict_batch
from scenario_generation.tensor_converter import (
    MapTensorCache,
    _build_neighbor_agents_past,
)
from scenario_generation.tools.eval_cl_trajectory import (
    evaluate_trajectory,
    load_border_segments,
)
from scenario_generation.transforms import _rotation_matrix

DT = 0.1  # sim + model timestep (10 Hz). Must match local_frame_rate.
_INPUT_T_PLUS_1 = 31  # tensor_converter._INPUT_T + 1 (history length the model sees).

# Model turn-indicator classes (decode_turn_indicator): 0 NONE / 1 DISABLE /
# 2 LEFT / 3 RIGHT / 4 KEEP. Sim TurnIndicatorsCommand: 0 NO_COMMAND /
# 1 DISABLE / 2 ENABLE_LEFT / 3 ENABLE_RIGHT. KEEP -> retain previous command.
_TI_MODEL_TO_SIM = {0: 0, 1: 1, 2: 2, 3: 3}

EGO_NAME = "ego"  # the headless EgoEntity is always named "ego"


@dataclass
class RolloutConfig:
    """Tuning for a single scenario_sim rollout."""

    fps: float = 10.0
    replan_interval: int = 4
    max_steps: int = 300
    warmup_steps: int = 5  # skip scoring until the history buffer is warm
    near_miss_thresh: float = 1.0
    # Ego bounding box fallback when the sim does not report one. wheelbase is
    # derived as ``ego_wheelbase_ratio * length`` when not otherwise known.
    ego_wheelbase_ratio: float = 0.65
    # LaneletSceneBuilder map/route window params (mirror replay.py defaults).
    max_map_lanelets: int = 140
    map_mask_range_m: float = 100.0
    route_window_segments: int = 25
    find_route_min_len_m: float = 120.0
    # Coordinate-contract sanity check: after the first stepped tick, the ego's
    # realized pose must land within this tolerance (m) of the injected plan's
    # first future point. Larger = looser; a gross frame mismatch blows past it.
    coord_check_tol_m: float = 2.0


# --------------------------------------------------------------------------- #
# Scene construction from sim truth.
# --------------------------------------------------------------------------- #
def _sim_type_to_agent_type(sim_type: int) -> AgentType | None:
    """Map ``get_entity_states`` type (0=EGO 1=VEHICLE 2=PED 3=MISC) to AgentType.

    MISC objects are not dynamic agents -> None (skipped)."""
    if sim_type in (0, 1):
        return AgentType.VEHICLE
    if sim_type == 2:
        return AgentType.PEDESTRIAN
    return None


class _HistoryBuffers:
    """Per-entity rolling (x, y, heading) history keyed by stable sim name.

    Cold start: the first time an entity is seen its buffer is filled by
    repeating the current pose (pad-with-current, not zeros) so no teleport
    artifact enters the model input. ``age`` tracks how many real ticks each
    entity has accumulated so scoring can wait for ``warmup_steps``."""

    def __init__(self, length: int = _INPUT_T_PLUS_1):
        self.length = length
        self._buf: dict[str, deque] = {}
        self.age: dict[str, int] = {}

    def update(self, name: str, x: float, y: float, yaw: float) -> None:
        if name not in self._buf:
            self._buf[name] = deque(
                [(x, y, yaw)] * self.length, maxlen=self.length
            )
            self.age[name] = 1
        else:
            self._buf[name].append((x, y, yaw))
            self.age[name] += 1

    def trajectory(self, name: str) -> np.ndarray:
        return np.array(self._buf[name], dtype=np.float32)  # (length, 3)


def _entity_shape(state: dict, cfg: RolloutConfig) -> tuple[float, float, float]:
    dims = state["bounding_box"]["dimensions"]
    length, width = float(dims["x"]), float(dims["y"])
    return length, width, cfg.ego_wheelbase_ratio * length


def _build_scene(
    states: dict,
    buffers: _HistoryBuffers,
    builder: LaneletSceneBuilder,
    ego_route_ids: list[int],
    goal_pose: np.ndarray,
    cfg: RolloutConfig,
    ego_name: str,
) -> SceneContext:
    """Build a SceneContext SNAPSHOT (map/MGRS world frame) from sim truth."""
    ego_state = states[ego_name]
    ego_xy = np.array(
        [ego_state["pose"]["x"], ego_state["pose"]["y"]], dtype=np.float32
    )

    # Map lanelets: closest-N around ego + pinned ego route (route context must
    # never drop). Mirrors replay._compute_map_lanelet_ids (minus NPC pinning,
    # which is a fidelity nicety not needed for bring-up).
    closest = builder.closest_lanelets(
        ego_xy, cfg.max_map_lanelets, mask_range=cfg.map_mask_range_m
    )
    seen: set[int] = set()
    all_ids: list[int] = []
    for ll_id in list(ego_route_ids) + list(closest):
        if ll_id in seen or not builder.has_lanelet_id(ll_id):
            continue
        seen.add(ll_id)
        all_ids.append(ll_id)
        if len(all_ids) >= cfg.max_map_lanelets:
            break
    map_data = builder._build_map_data(all_ids, center_xy=ego_xy)

    # Ego route_lanes: forward sliding window (C++-style), refreshed each tick.
    window = (
        builder.select_route_segment_indices(
            ego_route_ids, ego_xy, max_segments=cfg.route_window_segments
        )
        or ego_route_ids[: cfg.route_window_segments]
    )
    route_lanes, route_sl, route_hsl = builder._route_to_33dim(window)

    agents: list[Agent] = []
    for name, st in states.items():
        atype = _sim_type_to_agent_type(int(st["type"]))
        if atype is None:
            continue
        length, width, wheelbase = _entity_shape(st, cfg)
        traj = buffers.trajectory(name)
        is_ego = name == ego_name
        agents.append(
            Agent(
                id=name,
                agent_type=atype,
                length=length,
                width=width,
                wheelbase=wheelbase,
                past_trajectory=traj,
                past_velocities=None,  # derived from position diffs in converter
                goal_pose=goal_pose.astype(np.float32) if is_ego else None,
                route_lanes=route_lanes if is_ego else None,
                route_speed_limit=route_sl if is_ego else None,
                route_has_speed_limit=route_hsl if is_ego else None,
                turn_indicators=(
                    np.zeros(traj.shape[0], dtype=np.int32) if is_ego else None
                ),
                route_lanelet_ids=list(ego_route_ids) if is_ego else None,
                age_steps=buffers.age[name],
            )
        )
    return SceneContext(agents=agents, map_data=map_data, ego_agent_id=ego_name, dt=DT)


# --------------------------------------------------------------------------- #
# Route resolution (interim: LaneletSceneBuilder shortestPath / find_route).
# --------------------------------------------------------------------------- #
def _parse_goal_lanelet(osc_path: str | Path) -> int | None:
    """Best-effort goal lanelet id from an OpenSCENARIO AcquirePositionAction
    LanePosition. Returns None when the scenario authors no routing goal (then
    the caller falls back to find_route). SSv2 lane ids == lanelet ids."""
    try:
        root = ET.parse(str(osc_path)).getroot()
    except ET.ParseError:
        return None
    # Scope to AcquirePositionAction subtrees (ET has no parent pointers).
    for routing in root.iter("AcquirePositionAction"):
        lp = routing.find(".//LanePosition")
        if lp is not None and "laneId" in lp.attrib:
            try:
                return int(lp.attrib["laneId"])
            except ValueError:
                return None
    return None


def _resolve_route(
    builder: LaneletSceneBuilder,
    ego_xy: np.ndarray,
    osc_path: str | Path,
    cfg: RolloutConfig,
) -> list[int]:
    """Resolve the ego route to an ordered lanelet-id list (interim, Python-side).

    Snap the sim's actual start pose to a lanelet, then shortestPath to the
    scenario goal when one is authored, else a forward find_route."""
    start_ll = builder.snap_to_nearest_ll(ego_xy)
    if start_ll is None:
        raise RuntimeError(f"Could not snap ego start pose {ego_xy} to any lanelet")
    goal_ll = _parse_goal_lanelet(osc_path)
    if goal_ll is not None and builder.has_lanelet_id(goal_ll):
        route = builder.route_between(start_ll, goal_ll)
        if route:
            return route
        print(f"  [scenario_sim][WARN] goal lanelet {goal_ll} unreachable from {start_ll}; find_route")
    return builder.find_route(start_ll, cfg.find_route_min_len_m)


# --------------------------------------------------------------------------- #
# Per-tick helpers (keep the main rollout loop readable).
# --------------------------------------------------------------------------- #
def _pose_xyh(state: dict) -> tuple[float, float, float]:
    p = state["pose"]
    return float(p["x"]), float(p["y"]), float(p["yaw"])


def _update_history(buffers: _HistoryBuffers, states: dict) -> None:
    """Append this tick's truth pose to each dynamic entity's rolling buffer."""
    for name, st in states.items():
        if _sim_type_to_agent_type(int(st["type"])) is not None:
            buffers.update(name, st["pose"]["x"], st["pose"]["y"], st["pose"]["yaw"])


def _map_turn_indicator(ti_model: int, prev_cmd: int) -> int:
    """Model turn-indicator class -> sim command. KEEP (4) retains ``prev_cmd``."""
    return _TI_MODEL_TO_SIM.get(ti_model, prev_cmd)


def _predict_ego_plan(model, model_args, scene, device, map_cache) -> tuple[np.ndarray, int]:
    """Run the model as ego -> (ego-frame plan ``(future_len, 4)``, turn-indicator class)."""
    preds, tis = _predict_batch(
        model, model_args, scene, [EGO_NAME], device,
        map_cache=map_cache, return_turn_indicators=True,
    )
    return preds[EGO_NAME], int(tis.get(EGO_NAME, 0))


def _ego_plan_to_map_trajectory(
    plan_ego: np.ndarray, ex: float, ey: float, eh: float
) -> np.ndarray:
    """Ego-frame plan -> map-frame trajectory ``[N, 4]`` = (x, y, yaw, longitudinal v)
    for ``set_ego_trajectory``, using the current ego pose as the frame origin."""
    world_xy, world_h = _ego_to_world(plan_ego[:, :2], plan_ego[:, 2:4], ex, ey, eh)
    seg = np.linalg.norm(np.diff(world_xy, axis=0), axis=1)  # plan is future_len>=2 points
    speeds = np.concatenate([seg[:1], seg]) / DT
    return np.column_stack([world_xy, world_h, speeds]).astype(float)


def _score_neighbors(
    scene, ego_state: dict, ex: float, ey: float, eh: float, cfg: RolloutConfig, device: str
) -> tuple[float, bool]:
    """Instantaneous (min_clearance, collision) from raw ego-frame neighbor OBBs."""
    R = _rotation_matrix(eh)
    neighbors_live = _build_neighbor_agents_past(
        scene, EGO_NAME, R, np.array([ex, ey], dtype=np.float64), eh
    )[0, :, -1, :]
    ego_shape = np.array(_entity_shape(ego_state, cfg), dtype=np.float32)[
        [2, 0, 1]
    ]  # (wheelbase, length, width)
    min_clr, coll, _ = score_step(neighbors_live, ego_shape, 0.0, device)
    return min_clr, coll


def _traj_entry(step: int, ego_state: dict, goal_xy: np.ndarray) -> dict:
    """One trajectory_log row (world pose + speed + goal distance) for post-hoc metrics."""
    x, y, h = _pose_xyh(ego_state)
    tw = ego_state["twist"]
    return {
        "step": step,
        "x": x,
        "y": y,
        "heading": h,
        "speed": float(math.hypot(tw["linear_x"], tw["linear_y"])),
        "goal_d": float(math.hypot(x - goal_xy[0], y - goal_xy[1])),
    }


def _start_and_resolve_route(
    runner, builder: LaneletSceneBuilder, osc_path: str | Path, cfg: RolloutConfig, verbose: bool
) -> tuple[list[int], np.ndarray]:
    """Configure+activate the sim, verify the ego spawned, and resolve its route.

    A scenario the interpreter rejects at parse time leaves configure() at
    "unconfigured" (the on_configure exception is swallowed by withExceptionHandler
    -> FAILURE); proceeding to get_entity_states() on an unconfigured SimulatorCore
    dereferences a null core and SEGFAULTS, so we bail cleanly on a bad transition.
    Returns ``(ego_route_lanelet_ids, goal_pose)``."""
    st_cfg = runner.configure()
    if st_cfg != "inactive":
        raise RuntimeError(
            f"configure() did not reach 'inactive' (got '{st_cfg}') -- scenario "
            f"rejected by the interpreter at parse/configure time: {osc_path}"
        )
    st_act = runner.activate()
    if st_act != "active":
        raise RuntimeError(f"activate() did not reach 'active' (got '{st_act}'): {osc_path}")
    if EGO_NAME not in runner.get_entity_states():
        raise RuntimeError(
            f"'{EGO_NAME}' entity not spawned; entities={list(runner.get_entity_states())}"
        )

    ego0 = runner.get_ego_state(ego_ref=EGO_NAME)
    ego0_xy = np.array([ego0["pose"]["x"], ego0["pose"]["y"]], dtype=np.float32)
    ego_route_ids = _resolve_route(builder, ego0_xy, osc_path, cfg)
    if not ego_route_ids:
        raise RuntimeError("Empty ego route -- cannot build SceneContext")
    goal_pose = builder._route_goal(ego_route_ids)
    if verbose:
        print(
            f"  [scenario_sim] route={len(ego_route_ids)} lanelets, "
            f"start_ll={ego_route_ids[0]}, goal_ll={ego_route_ids[-1]}"
        )
    return ego_route_ids, goal_pose


def _finalize_row(
    output_dir: Path,
    map_path: str | Path,
    trajectory_log: list[dict],
    ego_state: dict | None,
    cfg: RolloutConfig,
    clearances: list[float],
    collisions: list[bool],
    terminated_reason: str,
    result_kind: str,
    coord_ok: bool | None,
    coord_err: float,
) -> dict:
    """Dump the trajectory, compute post-hoc road-border metrics, and build the
    aggregate-ready metrics row."""
    (output_dir / "trajectory_log.json").write_text(json.dumps(trajectory_log))

    ego_len, ego_w, ego_wb = (
        _entity_shape(ego_state, cfg) if ego_state is not None else (4.0, 1.8, 2.6)
    )
    rb = evaluate_trajectory(
        trajectory_log, load_border_segments(str(map_path)), ego_len, ego_w, ego_wb
    )

    finite_clr = [c for c in clearances if np.isfinite(c)]
    return {
        "n_steps_run": len(trajectory_log),
        "terminated": terminated_reason,
        "result_kind": result_kind,
        "min_clearance": float(min(finite_clr)) if finite_clr else float("inf"),
        "mean_clearance": float(np.mean(finite_clr)) if finite_clr else float("inf"),
        "n_collision_steps": int(sum(collisions)),
        "n_near_miss_steps": int(sum(1 for c in clearances if c <= cfg.near_miss_thresh)),
        "worst_step": int(np.argmin(clearances)) if clearances else -1,
        "progress_m": rb["progress_m"],
        "n_snaps": 0,  # scenario_sim has no unstick/teleport mechanism
        # post-hoc road-border (doc 02) + coordinate-contract diagnostics.
        "rb_dist_min": rb["rb_dist_min"],
        "rb_cross_steps": rb["rb_cross_steps"],
        "rb_has_data": rb["rb_has_data"],
        "coord_check_ok": bool(coord_ok) if coord_ok is not None else False,
        "coord_check_err_m": coord_err,
    }


# --------------------------------------------------------------------------- #
# Main rollout.
# --------------------------------------------------------------------------- #
def run_scenario_sim_rollout(
    model,
    model_args,
    osc_path: str | Path,
    map_path: str | Path,
    output_dir: str | Path,
    *,
    config: RolloutConfig | None = None,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """Run one closed-loop OpenSCENARIO rollout and return an aggregate-ready row.

    ``model`` / ``model_args`` follow the ``run_closed_loop_eval`` contract
    (``model(data) -> (_, outputs)`` with ``outputs["prediction"]``; ``model_args``
    provides ``observation_normalizer`` / ``predicted_neighbor_num`` /
    ``future_len``). Returns a metrics dict matching the schema
    ``closed_loop_eval.aggregate`` consumes, plus post-hoc ``rb_*`` road-border
    metrics, ``result_kind``, and the coordinate-contract check outcome.
    """
    import openscenario_python as osp  # requires the SSV2_HEADLESS_EGO overlay

    cfg = config or RolloutConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    builder = LaneletSceneBuilder(str(map_path))
    buffers = _HistoryBuffers()
    trajectory_log: list[dict] = []
    clearances: list[float] = []
    collisions: list[bool] = []

    cached_plan_ego: np.ndarray | None = None  # (future_len, 4) ego frame
    ti_cmd = 0  # last sim turn-indicator command (KEEP retains this)
    coord_err: float = float("nan")
    coord_ok: bool | None = None
    ego_state: dict | None = None  # last tick's ego truth (for finalize ego shape)
    terminated_reason = "max_steps"

    with osp.HeadlessRunner(
        osc_path=str(osc_path),
        output_directory=str(output_dir / "osp_out"),
        local_frame_rate=cfg.fps,
    ) as runner:
        ego_route_ids, goal_pose = _start_and_resolve_route(
            runner, builder, osc_path, cfg, verbose
        )
        goal_xy = goal_pose[:2]

        for step in range(cfg.max_steps):
            states = runner.get_entity_states()
            if EGO_NAME not in states:
                raise RuntimeError(f"Sim reports no '{EGO_NAME}' entity")
            ego_state = states[EGO_NAME]
            ex, ey, eh = _pose_xyh(ego_state)

            _update_history(buffers, states)
            scene = _build_scene(
                states, buffers, builder, ego_route_ids, goal_pose, cfg, EGO_NAME
            )
            map_cache = MapTensorCache(scene.map_data)

            # Replan every ``replan_interval`` ticks; consume the cached plan open-loop between.
            if cached_plan_ego is None or step % cfg.replan_interval == 0:
                cached_plan_ego, ti_model = _predict_ego_plan(
                    model, model_args, scene, device, map_cache
                )
                ti_cmd = _map_turn_indicator(ti_model, ti_cmd)

            pts = _ego_plan_to_map_trajectory(cached_plan_ego, ex, ey, eh)
            runner.set_ego_trajectory(pts, ego_ref=EGO_NAME)
            runner.set_ego_turn_indicator(int(ti_cmd), ego_ref=EGO_NAME)

            if buffers.age[EGO_NAME] >= cfg.warmup_steps:
                clr, coll = _score_neighbors(scene, ego_state, ex, ey, eh, cfg, device)
                clearances.append(clr)
                collisions.append(coll)

            # step() is the sole integrator: it advances BOTH the ego and the NPCs.
            outcome = runner.step()

            ego_after = runner.get_ego_state(ego_ref=EGO_NAME)
            trajectory_log.append(_traj_entry(step, ego_after, goal_xy))
            if coord_ok is None:  # first stepped tick: verify the frame contract
                ax, ay, _ = _pose_xyh(ego_after)
                coord_err = float(math.hypot(ax - pts[0, 0], ay - pts[0, 1]))
                coord_ok = coord_err <= cfg.coord_check_tol_m
                if verbose:
                    print(
                        f"  [scenario_sim] coord check: err={coord_err:.3f} m "
                        f"(tol {cfg.coord_check_tol_m}) -> {'OK' if coord_ok else 'FAIL'}"
                    )

            if outcome == "terminated":
                terminated_reason = "sim_terminated"
                break

        result_kind = runner.result_kind()

    return _finalize_row(
        output_dir, map_path, trajectory_log, ego_state, cfg, clearances, collisions,
        terminated_reason, result_kind, coord_ok, coord_err,
    )


def main() -> int:
    """Single-scenario worker: run ONE rollout, write its row to ``--row_out``.

    Run as a subprocess (one per scenario) by ``closed_loop_eval.run_scenario_sim_eval``.
    The C++ ``SimulatorCore`` is a static singleton (1 process = 1 scenario), so a
    fresh process per scenario is required for isolation + a clean teardown.
    """
    import argparse

    p = argparse.ArgumentParser(description="scenario_sim single-scenario worker")
    p.add_argument("--osc", required=True)
    p.add_argument("--map_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--row_out", required=True, help="write the metrics row JSON here")
    p.add_argument("--device", default="cpu")
    p.add_argument("--model_path", required=True, help="torch .pth checkpoint")
    p.add_argument("--replan_interval", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--warmup_steps", type=int, default=5)
    p.add_argument("--near_miss_thresh", type=float, default=1.0)
    p.add_argument("--fps", type=float, default=10.0)
    a = p.parse_args()

    from scenario_generation.simulate import load_model

    model, model_args = load_model(a.model_path, a.device)
    cfg = RolloutConfig(
        fps=a.fps,
        replan_interval=a.replan_interval,
        max_steps=a.max_steps,
        warmup_steps=a.warmup_steps,
        near_miss_thresh=a.near_miss_thresh,
    )
    row = run_scenario_sim_rollout(
        model, model_args, a.osc, a.map_path, a.out_dir, config=cfg, device=a.device
    )
    Path(a.row_out).write_text(json.dumps(row, default=float))
    return 0


__all__ = [
    "RolloutConfig",
    "run_scenario_sim_rollout",
]


if __name__ == "__main__":
    import sys

    sys.exit(main())
