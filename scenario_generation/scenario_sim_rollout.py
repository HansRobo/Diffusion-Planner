"""OpenSCENARIO-driven closed-loop rollout for the Diffusion-Planner validator.

This is the ``scenario_sim`` sibling of the NPZ-replay closed-loop path
(``reproducer_rollout`` / ``closed_loop_eval.run_closed_loop_eval``). Instead of
replaying recorded frames, it drives the C++ OpenSCENARIO interpreter in-process
(``openscenario_python.HeadlessRunner``, the SSV2_HEADLESS_EGO pybind facade) and
runs the live Diffusion-Planner as the ego every tick:

    1. read ground-truth entity states from the sim (ego + NPCs)
    2. build a ``SceneContext`` SNAPSHOT (map/MGRS world frame) via LaneletSceneBuilder
    3. ``to_model_tensors`` -> model -> ego-frame prediction (80, 4)
    4. transform the plan back to map frame and inject via ``set_ego_trajectory``
    5. ``step()`` -- the C++ sim advances BOTH the ego (injected-trajectory
       sim_model integration) AND the NPCs (behavior_tree)
    6. read the new truth, score, repeat.

Critical design invariant: we NEVER advance the Python scene (no
``simulate.advance_scene``). The C++ ``step()`` is the sole integrator; Python
only reads truth and injects a plan. Advancing the Python scene too would
double-simulate and diverge from sim truth.

Frame contract: the sim reports map-frame poses; we store them into the
SceneContext as-is (the builder is also in the map's MGRS frame), and
``to_model_tensors`` re-centers onto the ego internally. The model's ego-frame
output is mapped back to the map frame with the CURRENT ego pose via
``simulate._ego_to_world`` before injection.

Scope (task #5, Phase 3 bring-up):
  * route is resolved in pure Python via ``LaneletSceneBuilder`` (lanelet2
    shortestPath / find_route). This is an INTERIM: the plan's route sidecar
    (mission_planner exact match, doc 03) is deferred -- the route only needs
    to be plausible for the model here, not bit-identical to mission_planner.
  * inference uses the torch ``.pth`` checkpoint (the training-env path), loaded
    via ``simulate.load_model`` (``model(data) -> (_, outputs)`` contract).
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
    # Distance (m) at which a synthetic neighbor is injected when the ego has no
    # real NPCs (works around the exported ONNX's empty-neighbor-gather crash).
    dummy_neighbor_dist_m: float = 500.0


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
    """(length, width, wheelbase) from a sim entity's bounding box."""
    dims = state["bounding_box"]["dimensions"]
    length = float(dims["x"]) or 4.0
    width = float(dims["y"]) or 1.8
    wheelbase = cfg.ego_wheelbase_ratio * length
    return length, width, wheelbase


def _build_scene(
    states: dict,
    buffers: _HistoryBuffers,
    builder: LaneletSceneBuilder,
    ego_route_ids: list[int],
    goal_pose: np.ndarray,
    cfg: RolloutConfig,
) -> SceneContext:
    """Build a SceneContext SNAPSHOT (map/MGRS world frame) from sim truth."""
    ego_state = states["ego"]
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
        is_ego = name == "ego"
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
    return SceneContext(agents=agents, map_data=map_data, ego_agent_id="ego", dt=DT)


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
    # No goal (or unreachable) -> forward route from start.
    return builder.find_route(start_ll, cfg.find_route_min_len_m)


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

    outcome = "running"
    result_kind = ""
    terminated_reason = "max_steps"

    with osp.HeadlessRunner(
        osc_path=str(osc_path),
        output_directory=str(output_dir / "osp_out"),
        local_frame_rate=cfg.fps,
    ) as runner:
        runner.configure()
        runner.activate()

        ego0 = runner.get_ego_state()
        ego0_xy = np.array([ego0["pose"]["x"], ego0["pose"]["y"]], dtype=np.float32)
        ego_route_ids = _resolve_route(builder, ego0_xy, osc_path, cfg)
        if not ego_route_ids:
            raise RuntimeError("Empty ego route -- cannot build SceneContext")
        goal_pose = builder._route_goal(ego_route_ids)
        goal_xy = goal_pose[:2]
        if verbose:
            print(
                f"  [scenario_sim] route={len(ego_route_ids)} lanelets, "
                f"start_ll={ego_route_ids[0]}, goal_ll={ego_route_ids[-1]}"
            )

        for step in range(cfg.max_steps):
            states = runner.get_entity_states()
            if "ego" not in states:
                raise RuntimeError("Sim reports no 'ego' entity")

            # 1. Update per-entity history buffers from truth.
            for name, st in states.items():
                if _sim_type_to_agent_type(int(st["type"])) is None:
                    continue
                buffers.update(
                    name, st["pose"]["x"], st["pose"]["y"], st["pose"]["yaw"]
                )

            # 2. Build the SceneContext snapshot (map/MGRS frame).
            scene = _build_scene(
                states, buffers, builder, ego_route_ids, goal_pose, cfg
            )
            map_cache = MapTensorCache(scene.map_data)

            ego_state = states["ego"]
            ex = float(ego_state["pose"]["x"])
            ey = float(ego_state["pose"]["y"])
            eh = float(ego_state["pose"]["yaw"])

            # 3. Replan gate: run the model, cache the ego-frame plan.
            if cached_plan_ego is None or step % cfg.replan_interval == 0:
                preds, tis = _predict_batch(
                    model,
                    model_args,
                    scene,
                    ["ego"],
                    device,
                    map_cache=map_cache,
                    return_turn_indicators=True,
                )
                cached_plan_ego = preds["ego"]  # (future_len, 4) ego frame
                ti_model = int(tis.get("ego", 0))
                if ti_model in _TI_MODEL_TO_SIM:  # KEEP (4) retains ti_cmd
                    ti_cmd = _TI_MODEL_TO_SIM[ti_model]

            # 4. Ego-frame plan -> map-frame trajectory, inject.
            world_xy, world_h = _ego_to_world(
                cached_plan_ego[:, :2], cached_plan_ego[:, 2:4], ex, ey, eh
            )
            seg = np.linalg.norm(np.diff(world_xy, axis=0), axis=1)
            speeds = np.concatenate([seg[:1], seg]) / DT if len(seg) else np.zeros(1)
            pts = np.column_stack([world_xy, world_h, speeds]).astype(float)  # (N, 4)
            runner.set_ego_trajectory(pts)
            runner.set_ego_turn_indicator(int(ti_cmd))
            planned_first_xy = world_xy[0].copy()

            # 5. Score at state k (raw ego-frame neighbors, un-normalized).
            if buffers.age["ego"] >= cfg.warmup_steps:
                R = _rotation_matrix(eh)
                neighbors_live = _build_neighbor_agents_past(
                    scene, "ego", R, np.array([ex, ey], dtype=np.float64), eh
                )[0, :, -1, :]
                ego_shape = np.array(
                    _entity_shape(ego_state, cfg), dtype=np.float32
                )[[2, 0, 1]]  # (wheelbase, length, width)
                min_clr, coll, _ = score_step(neighbors_live, ego_shape, 0.0, device)
                clearances.append(min_clr)
                collisions.append(coll)

            # 6. Advance the C++ sim by one tick (ego + NPCs).
            outcome = runner.step()

            # 7. Record realized ego pose + coordinate-contract check.
            ego_after = runner.get_ego_state()
            ax = float(ego_after["pose"]["x"])
            ay = float(ego_after["pose"]["y"])
            ah = float(ego_after["pose"]["yaw"])
            speed = float(
                math.hypot(
                    ego_after["twist"]["linear_x"], ego_after["twist"]["linear_y"]
                )
            )
            trajectory_log.append(
                {
                    "step": step,
                    "x": ax,
                    "y": ay,
                    "heading": ah,
                    "speed": speed,
                    "goal_d": float(math.hypot(ax - goal_xy[0], ay - goal_xy[1])),
                }
            )
            if coord_ok is None:
                # First stepped tick: realized pose should land near the plan's
                # first future point (one step ahead). Validates the full
                # ego-frame -> map-frame -> sim injection frame contract.
                coord_err = float(
                    math.hypot(ax - planned_first_xy[0], ay - planned_first_xy[1])
                )
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

    # --- Finalize: dump trajectory + post-hoc road-border metrics. ---
    (output_dir / "trajectory_log.json").write_text(json.dumps(trajectory_log))

    ego_len, ego_w, ego_wb = (
        _entity_shape(ego_state, cfg) if trajectory_log else (4.0, 1.8, 2.6)
    )
    borders = load_border_segments(str(map_path))
    rb = evaluate_trajectory(trajectory_log, borders, ego_len, ego_w, ego_wb)

    finite_clr = [c for c in clearances if np.isfinite(c)]
    n_near_miss = sum(1 for c in clearances if c <= cfg.near_miss_thresh)
    worst_step = int(np.argmin(clearances)) if clearances else -1

    row = {
        "n_steps_run": len(trajectory_log),
        "terminated": terminated_reason,
        "result_kind": result_kind,
        "min_clearance": float(min(finite_clr)) if finite_clr else float("inf"),
        "mean_clearance": float(np.mean(finite_clr)) if finite_clr else float("inf"),
        "n_collision_steps": int(sum(1 for c in collisions if c)),
        "n_near_miss_steps": int(n_near_miss),
        "worst_step": worst_step,
        "progress_m": rb["progress_m"],
        "n_snaps": 0,  # scenario_sim has no unstick/teleport mechanism
        # post-hoc road-border (doc 02) + coordinate-contract diagnostics.
        "rb_dist_min": rb["rb_dist_min"],
        "rb_cross_steps": rb["rb_cross_steps"],
        "rb_has_data": rb["rb_has_data"],
        "coord_check_ok": bool(coord_ok) if coord_ok is not None else False,
        "coord_check_err_m": coord_err,
    }
    return row


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
