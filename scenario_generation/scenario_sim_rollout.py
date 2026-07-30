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
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import torch

from scenario_generation.gui.lanelet_scene_builder import LaneletSceneBuilder
from scenario_generation.metrics.object import score_object_step
from scenario_generation.perf_timer import Timers
from scenario_generation.scenario_sim_metrics import build_segment_row
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
# Same default the reproducer path uses, so the strong_brake metric is comparable.
STRONG_BRAKE_MPS2 = -2.5
_INPUT_T_PLUS_1 = 31  # tensor_converter._INPUT_T + 1 (history length the model sees).

# Model turn-indicator classes (decode_turn_indicator): 0 NONE / 1 DISABLE /
# 2 LEFT / 3 RIGHT / 4 KEEP. Sim TurnIndicatorsCommand: 0 NO_COMMAND /
# 1 DISABLE / 2 ENABLE_LEFT / 3 ENABLE_RIGHT. KEEP -> retain previous command.
_TI_MODEL_TO_SIM = {0: 0, 1: 1, 2: 2, 3: 3}

_SIM_TYPE_EGO = 0  # get_entity_states()["type"]: 0=EGO 1=VEHICLE 2=PEDESTRIAN 3=MISC_OBJECT


class _CloneEncoderOutput(torch.nn.Module):
    """Clone the encoder's output: the encoding is held across all DPM steps, and a
    cudagraph-managed buffer would be overwritten by the DiT's next replay."""

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, *a, **kw):
        return self.inner(*a, **kw).clone()


@dataclass
class RolloutConfig:
    """Tuning for a single scenario_sim rollout."""

    fps: float = 10.0
    # 1 = every tick = 10 Hz, matching the production node's planning_frequency_hz.
    # Values > 1 consume a cached plan open-loop in between: cheaper, less reactive.
    replan_interval: int = 1
    max_steps: int = 300
    warmup_steps: int = 5  # skip scoring until the history buffer is warm
    near_miss_thresh: float = 1.0
    # Feeds ``Agent.wheelbase`` ONLY -- whose contract is the real axle spacing
    # (scene_context.Agent: "Distance between front and rear axles ... ~0.65 * length when
    # not available"), consumed as such by the bicycle model in simulate.py. The metric
    # geometry does NOT use this: it needs twice the bbox-centre offset, which the sim
    # reports exactly -- see :func:`_ego_metric_box`.
    ego_wheelbase_ratio: float = 0.65
    # LaneletSceneBuilder map/route window params (mirror replay.py defaults).
    max_map_lanelets: int = 140
    map_mask_range_m: float = 100.0
    route_window_segments: int = 25
    find_route_min_len_m: float = 120.0
    # Run the interpreter in its own process instead of this one. The SimulatorCore singleton is
    # the only reason a worker is per-scenario; moving it out is what would let the model, CUDA
    # context, compiled graphs and parsed map be reused across scenarios (plan/11 9s).
    sim_in_subprocess: bool = False
    # Coordinate-contract sanity check: after the first stepped tick, the ego's
    # realized pose must land within this tolerance (m) of the injected plan's
    # first future point. Larger = looser; a gross frame mismatch blows past it.
    coord_check_tol_m: float = 2.0


# --------------------------------------------------------------------------- #
# Scene construction from sim truth.
# --------------------------------------------------------------------------- #
def _resolve_ego_name(states: dict) -> str:
    """The ego's entity name as the simulator reports it.

    Identified by type, not by name: ``openscenario_python.cpp`` exposes ``type`` with 0 = EGO,
    which is authoritative whatever the scenario chose to call the entity ("ego", "Ego", ...).
    Matching on the name instead fails at spawn time with a message that points at the symptom
    rather than the cause.
    """
    egos = [name for name, st in states.items() if int(st["type"]) == _SIM_TYPE_EGO]
    if len(egos) == 1:
        return egos[0]
    if not egos:
        raise RuntimeError(
            f"no ego-typed (type={_SIM_TYPE_EGO}) entity spawned; entities="
            + str({n: st["type"] for n, st in states.items()})
        )
    raise RuntimeError(f"more than one ego-typed entity: {egos}")


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
    """``(length, width, axle_wheelbase)`` for ``Agent`` -- model input, not metrics.

    The third value follows ``Agent.wheelbase``'s contract (real axle spacing) and keeps the
    documented ``~0.65 * length`` estimate, because what the model input has to match is the
    convention its training data was built with. Metric geometry uses
    :func:`_ego_metric_box` instead -- the two want different numbers for the same vehicle.
    """
    dims = state["bounding_box"]["dimensions"]
    length, width = float(dims["x"]), float(dims["y"])
    return length, width, cfg.ego_wheelbase_ratio * length


def _ego_metric_box(state: dict) -> tuple[float, float, float]:
    """``(length, width, box_wheelbase)`` for the metric OBB builders.

    ``box_wheelbase`` is deliberately NOT the axle spacing. Both metric builders
    (``_build_ego_bbox_corners`` behind ``score_object_step`` for object clearance, and
    ``obb_corners`` behind ``evaluate_trajectory`` for the road border) treat the reported
    pose as the rear-axle midpoint and derive the box from ``length`` plus a ``wheelbase``
    argument **assuming symmetric overhangs**, which places the box centre exactly
    ``wheelbase/2`` ahead of the pose. So the value they need is twice the true box-centre
    offset, and the simulator reports that offset directly: the pybind always fills
    ``bounding_box.center`` (``openscenario_python.cpp::boundingBoxToDict``). Nothing is
    estimated here.

    Why this matters: the webauto suite's ego is bus-shaped with markedly asymmetric
    overhangs (length 7.2369, centre offset 2.0927, i.e. 1.53 m behind vs 3.04 m ahead of the
    axles), so the ``0.65 * length`` estimate that ``Agent.wheelbase`` uses would put the box
    0.259 m too far forward -- the same order as the near-miss margin itself. It biases
    clearance in opposite directions fore and aft, so it cannot be corrected after the fact.
    """
    bbox = state["bounding_box"]
    dims = bbox["dimensions"]
    return float(dims["x"]), float(dims["y"]), 2.0 * float(bbox["center"]["x"])


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
def map_from_osc(osc_path: str | Path) -> str:
    """The lanelet2 map this scenario declares, from its ``RoadNetwork/LogicFile``.

    Single source of truth for the map, on purpose. The C++ interpreter resolves the map from
    exactly this element (``Interpreter::makeCurrentConfiguration``), so deriving the
    Python-side map the same way makes it impossible for the two to disagree. Passing a map in
    from the caller cannot offer that guarantee: a suite spanning several maps (the webauto
    suite spans three) would then have the route, centreline and road-border geometry computed
    against a different map than the one the simulator actually ran, with no error -- just
    plausible, wrong numbers.
    """
    root = ET.parse(str(osc_path)).getroot()
    for node in root.iter("LogicFile"):
        path = node.attrib.get("filepath")
        if path:
            return path
    raise ValueError(f"No RoadNetwork/LogicFile filepath in {osc_path}")


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


@torch.no_grad()
def _predict_ego_plan(
    model, model_args, scene, device, map_cache, ego_name: str
) -> tuple[np.ndarray, int]:
    """Run the model as ego -> (ego-frame plan ``(future_len, 4)``, turn-indicator class).

    ``no_grad`` is load-bearing, not an optimisation: outputs carrying an autograd graph
    silently drop compile off the CUDA-graph fast path and give the whole 1.8x back."""
    torch.compiler.cudagraph_mark_step_begin()  # one inference == one cudagraph step
    preds, tis = _predict_batch(
        model, model_args, scene, [ego_name], device,
        map_cache=map_cache, return_turn_indicators=True,
    )
    return preds[ego_name], int(tis.get(ego_name, 0))


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
    scene, ego_state: dict, ex: float, ey: float, eh: float, device: str, ego_name: str
) -> tuple[float, bool]:
    """Instantaneous (min_clearance, collision) from raw ego-frame neighbor OBBs."""
    R = _rotation_matrix(eh)
    neighbors_live = _build_neighbor_agents_past(
        scene, ego_name, R, np.array([ex, ey], dtype=np.float64), eh
    )[0, :, -1, :]
    ego_shape = np.array(_ego_metric_box(ego_state), dtype=np.float32)[
        [2, 0, 1]
    ]  # (box_wheelbase, length, width) -- see _ego_metric_box, not _entity_shape
    min_clr, coll, _ = score_object_step(neighbors_live, ego_shape, device)
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
) -> tuple[str, list[int], np.ndarray]:
    """Configure+activate the sim, resolve the ego's name, and resolve its route.

    A scenario the interpreter rejects at parse time leaves configure() at
    "unconfigured" (the on_configure exception is swallowed by withExceptionHandler
    -> FAILURE); proceeding to get_entity_states() on an unconfigured SimulatorCore
    dereferences a null core and SEGFAULTS, so we bail cleanly on a bad transition.
    Returns ``(ego_name, ego_route_lanelet_ids, goal_pose)``."""
    st_cfg = runner.configure()
    if st_cfg != "inactive":
        raise RuntimeError(
            f"configure() did not reach 'inactive' (got '{st_cfg}') -- scenario "
            f"rejected by the interpreter at parse/configure time: {osc_path}"
        )
    st_act = runner.activate()
    if st_act != "active":
        raise RuntimeError(f"activate() did not reach 'active' (got '{st_act}'): {osc_path}")
    ego_name = _resolve_ego_name(runner.get_entity_states())

    ego0 = runner.get_ego_state(ego_ref=ego_name)
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
    return ego_name, ego_route_ids, goal_pose


# Border polylines further than this from the ego's path cannot be its nearest border. Only
# an upper bound is needed, so this is generous: at 50 m the pruning still removes ~98.6% of
# the Odaiba map's segments, and the margin is what makes the result exact (see below).
_RB_PRUNE_MARGIN_M = 50.0


def _prune_border_segments(
    segments: list[np.ndarray],
    trajectory_log: list[dict],
    ego_length: float,
    ego_wheelbase: float,
    ego_width: float,
    margin_m: float = _RB_PRUNE_MARGIN_M,
) -> list[np.ndarray]:
    """Drop border polylines that cannot possibly be the ego's nearest border.

    ``evaluate_trajectory`` compares the ego's sampled perimeter against EVERY border segment
    on the map at EVERY tick. On the Odaiba map that is 32 perimeter points x 38,206 segments
    x 1700 ticks = 2.1e9 point-segment distance evaluations, measured at 55 ms/tick -- the
    single largest cost of a full-suite run (46.6% of worker time; plan/11 9m). The ego covers
    ~250 m of a multi-kilometre map, so nearly all of that work is against geometry it never
    goes near.

    EXACT, not approximate -- which matters because ``rb_dists`` feeds mean/p5/tdigest
    clearance statistics, not just a threshold test, so an approximation would silently move
    the metrics. A polyline whose bounding box misses the trajectory's bounding box inflated
    by (ego reach + ``margin_m``) is more than ``margin_m`` from the ego at every tick, so it
    cannot beat any reported distance <= ``margin_m``. The caller verifies that condition and
    recomputes against the full set if it does not hold. Under it the returned distances are
    bit-identical to the unpruned computation -- verified on a real 300-tick trajectory:
    38,206 -> 541 segments, 65x faster, ``np.array_equal`` True.

    Returns the input unchanged when the map ships no borders at all, so ``rb_has_data`` keeps
    meaning "this map has no road_border geometry" and never degrades to "none was nearby".
    """
    if not segments or not trajectory_log:
        return segments
    xy = np.array([(e["x"], e["y"]) for e in trajectory_log], dtype=np.float64)
    # Furthest any sampled perimeter point can be from the reported pose: the OBB reaches
    # (length + wheelbase) / 2 forward of it (see _ego_metric_box for that convention).
    reach = float(np.hypot(0.5 * (ego_length + ego_wheelbase), 0.5 * ego_width))
    lo = xy.min(axis=0) - (reach + margin_m)
    hi = xy.max(axis=0) + (reach + margin_m)
    kept = [
        s for s in segments
        if np.all(s.max(axis=0) >= lo) and np.all(s.min(axis=0) <= hi)
    ]
    return kept


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
        _ego_metric_box(ego_state) if ego_state is not None else (4.0, 1.8, 2.6)
    )
    borders = load_border_segments(str(map_path))
    near = _prune_border_segments(borders, trajectory_log, ego_len, ego_wb, ego_w)
    rb = evaluate_trajectory(trajectory_log, near, ego_len, ego_w, ego_wb)
    # The pruning is only equivalent to the full scan while the reported distances stay
    # inside the margin (a border beyond it could otherwise have been nearer). Checking the
    # result rather than trusting the assumption keeps the metric exact even on a map where
    # the ego runs far from any border; the full recomputation then costs what it always did.
    _rb = np.asarray(rb["rb_dists"], dtype=np.float64)
    _finite = np.isfinite(_rb)
    if len(near) < len(borders) and _finite.any() and _rb[_finite].max() > _RB_PRUNE_MARGIN_M:
        print(
            f"  [scenario_sim][WARN] road-border distance {_rb[_finite].max():.1f} m exceeds "
            f"the {_RB_PRUNE_MARGIN_M} m prune margin; recomputing over all "
            f"{len(borders)} polylines"
        )
        rb = evaluate_trajectory(trajectory_log, borders, ego_len, ego_w, ego_wb)

    return build_segment_row(
        n_steps_run=len(trajectory_log),
        terminated=terminated_reason,
        result_kind=result_kind,
        clearances=clearances,
        collisions=collisions,
        rb_dists=rb["rb_dists"],
        speeds=[float(t["speed"]) for t in trajectory_log],
        dt=DT,
        near_miss_thresh=cfg.near_miss_thresh,
        strong_brake_mps2=STRONG_BRAKE_MPS2,
        progress_m=rb["progress_m"],
        # scenario_sim-only diagnostics. Kept flat (outside the shared category blocks) so
        # aggregate never sees them as a metric category.
        extra={
            "worst_step": int(np.argmin(clearances)) if clearances else -1,
            "rb_has_data": rb["rb_has_data"],
            "coord_check_ok": bool(coord_ok) if coord_ok is not None else False,
            "coord_check_err_m": coord_err,
        },
    )


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
    timers: Timers | None = None,
    builder: LaneletSceneBuilder | None = None,
) -> dict:
    """Run one closed-loop OpenSCENARIO rollout and return an aggregate-ready row.

    ``model`` / ``model_args`` follow the ``run_closed_loop_eval`` contract
    (``model(data) -> (_, outputs)`` with ``outputs["prediction"]``; ``model_args``
    provides ``observation_normalizer`` / ``predicted_neighbor_num`` /
    ``future_len``). Returns a metrics dict matching the schema
    ``closed_loop_eval.aggregate`` consumes, plus post-hoc ``rb_*`` road-border
    metrics, ``result_kind``, and the coordinate-contract check outcome.
    """
    cfg = config or RolloutConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Always-on stage timing, shared with the npz path (perf_timer.Timers) so both
    # closed-loop paths report the same shape. One process runs one scenario, so these
    # totals ARE this scenario's profile -- see the timing note in main().
    timers = timers or Timers()
    _t_rollout = time.perf_counter()
    _t = time.perf_counter()
    # A caller that outlives one scenario passes its cached builder: parsing a 46 MB lanelet2 map
    # costs 7.5 s per case (job 1665) and 378 of the suite's 464 cases share one map, so this is
    # per-map work that a per-scenario process was paying per scenario.
    if builder is None:
        builder = LaneletSceneBuilder(str(map_path))
    timers.add("map_build", time.perf_counter() - _t)
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

    # The interpreter writes its JUnit result here, and that file is the ONLY place the reason a
    # scenario was rejected survives: on_configure wraps everything in withExceptionHandler,
    # which records the exception via set<common::junit::Error>(type, what()) and returns
    # FAILURE, so the caller sees only the lifecycle state "unconfigured". write_to() silently
    # does nothing when the directory is missing, and nothing was creating it -- a full-suite run
    # produced zero result.junit.xml for 464 cases, which is why 11 rejected scenarios have been
    # unexplained since 2026-07-24 (plan/11 9m).
    osp_out = output_dir / "osp_out"
    osp_out.mkdir(parents=True, exist_ok=True)

    if cfg.sim_in_subprocess:
        from scenario_generation.sim_proc_client import RemoteRunner as _Runner
    else:
        import openscenario_python as osp  # requires the SSV2_HEADLESS_EGO overlay

        _Runner = osp.HeadlessRunner

    _t = time.perf_counter()
    # RemoteRunner is duck-type compatible for the calls below, so the tick loop is identical
    # either way -- deliberately: the loop is where fidelity lives.
    with _Runner(
        osc_path=str(osc_path),
        output_directory=str(osp_out),
        local_frame_rate=cfg.fps,
    ) as runner:
        timers.add("sim_open", time.perf_counter() - _t)
        _t = time.perf_counter()
        ego_name, ego_route_ids, goal_pose = _start_and_resolve_route(
            runner, builder, osc_path, cfg, verbose
        )
        timers.add("route_resolve", time.perf_counter() - _t)
        goal_xy = goal_pose[:2]

        for step in range(cfg.max_steps):
            with timers("sim_get_states"):
                states = runner.get_entity_states()
            if ego_name not in states:
                raise RuntimeError(f"Sim stopped reporting the ego entity '{ego_name}'")
            ego_state = states[ego_name]
            ex, ey, eh = _pose_xyh(ego_state)

            with timers("scene_build"):
                _update_history(buffers, states)
                scene = _build_scene(
                    states, buffers, builder, ego_route_ids, goal_pose, cfg, ego_name
                )
                map_cache = MapTensorCache(scene.map_data)

            # Replan every ``replan_interval`` ticks; consume the cached plan open-loop between.
            if cached_plan_ego is None or step % cfg.replan_interval == 0:
                # The first call also pays torch.compile; a separate stage keeps the cold
                # cost from being averaged into the steady-state ms/call (and vice versa).
                with timers("predict_cold" if cached_plan_ego is None else "predict"):
                    cached_plan_ego, ti_model = _predict_ego_plan(
                        model, model_args, scene, device, map_cache, ego_name
                    )
                ti_cmd = _map_turn_indicator(ti_model, ti_cmd)

            with timers("sim_set_traj"):
                pts = _ego_plan_to_map_trajectory(cached_plan_ego, ex, ey, eh)
                runner.set_ego_trajectory(pts, ego_ref=ego_name)
                runner.set_ego_turn_indicator(int(ti_cmd), ego_ref=ego_name)

            if buffers.age[ego_name] >= cfg.warmup_steps:
                with timers("score_objects"):
                    clr, coll = _score_neighbors(
                        scene, ego_state, ex, ey, eh, device, ego_name
                    )
                clearances.append(clr)
                collisions.append(coll)

            # step() is the sole integrator: it advances BOTH the ego and the NPCs.
            with timers("sim_step"):
                outcome = runner.step()

            ego_after = runner.get_ego_state(ego_ref=ego_name)
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

    _t = time.perf_counter()
    row = _finalize_row(
        output_dir, map_path, trajectory_log, ego_state, cfg, clearances, collisions,
        terminated_reason, result_kind, coord_ok, coord_err,
    )
    timers.add("finalize", time.perf_counter() - _t)
    # `sim_close` is not timed separately: teardown happens in HeadlessRunner.__exit__, so it
    # lands in the gap between the loop and here and shows up in rollout_s minus the parts.
    timers.add("rollout_total", time.perf_counter() - _t_rollout)
    row["timing"] = timers.as_dict()
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
    p.add_argument(
        "--map_path",
        default=None,
        help="lanelet2 .osm; defaults to the scenario's own RoadNetwork/LogicFile, which is "
        "also where the C++ interpreter reads it from -- override only to test a substitute map",
    )
    p.add_argument("--out_dir", required=True)
    p.add_argument("--row_out", required=True, help="write the metrics row JSON here")
    p.add_argument("--device", default="cpu")
    p.add_argument("--model_path", required=True, help="torch .pth checkpoint")
    p.add_argument("--replan_interval", type=int, default=1,
                   help="re-plan every N ticks; 1 (default) = every tick = 10 Hz, matching production")
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--warmup_steps", type=int, default=5)
    p.add_argument("--near_miss_thresh", type=float, default=1.0)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--sim_in_subprocess", action="store_true",
                   help="run the OpenSCENARIO interpreter in a child process instead of this one")
    p.add_argument(
        "--watchdog_sec", type=float, default=0.0,
        help="dump every thread's stack and exit after this many seconds (0 = off). The "
             "parent sets it just under its own kill deadline so a hang leaves a stack behind",
    )
    a = p.parse_args()

    # Armed before anything heavy so it also covers model load and torch.compile. A full-suite
    # run met a worker that sat 64 minutes in futex_wait_queue_me with 246 threads, no CPU and
    # no GPU, ignored SIGTERM and needed SIGKILL -- and left no stack, so the cause is still
    # unknown (plan/11 9m). faulthandler's watchdog runs on its own native thread and writes
    # straight to the fd, so it can dump even when the main thread is wedged in C without the
    # GIL, which is exactly that failure mode. `exit=True` makes it _exit() afterwards.
    if a.watchdog_sec > 0:
        import faulthandler

        faulthandler.enable()
        faulthandler.dump_traceback_later(a.watchdog_sec, exit=True)

    from scenario_generation.simulate import load_model

    timers = Timers()
    t_proc = time.perf_counter()
    with timers("model_load"):
        model, model_args = load_model(a.model_path, a.device)
    # T_fwd 26 -> 14.4 ms, bitwise identical to eager. Must come after load_state_dict
    # (compiling rewrites key names). Compiling the DPM loop as one graph is faster
    # single-stream (12.5 ms) but SLOWER under MPS, which is how eval actually runs --
    # measured numbers and the reason: plan/10 2b-11.
    model.decoder.dit = torch.compile(model.decoder.dit, mode="reduce-overhead")
    model.encoder = _CloneEncoderOutput(
        torch.compile(model.encoder, mode="reduce-overhead")
    )
    map_path = a.map_path or map_from_osc(a.osc)
    cfg = RolloutConfig(
        fps=a.fps,
        replan_interval=a.replan_interval,
        max_steps=a.max_steps,
        warmup_steps=a.warmup_steps,
        near_miss_thresh=a.near_miss_thresh,
        sim_in_subprocess=a.sim_in_subprocess,
    )
    row = run_scenario_sim_rollout(
        model, model_args, a.osc, map_path, a.out_dir,
        config=cfg, device=a.device, timers=timers,
    )
    # With one process per scenario, the pre-rollout costs are paid once per scenario -- a
    # full suite pays them hundreds of times -- so they belong in the same breakdown as the
    # per-tick sums. Otherwise a run dominated by startup is indistinguishable from one
    # dominated by inference. `torch.compile` is deliberately not a stage of its own: the
    # compile happens on the first forward, which is why `predict_cold` is separate.
    timers.add("worker_process", time.perf_counter() - t_proc)
    row["timing"] = timers.as_dict()
    row["map_path"] = str(map_path)
    Path(a.row_out).write_text(json.dumps(row, default=float))
    return 0


__all__ = [
    "RolloutConfig",
    "run_scenario_sim_rollout",
]


if __name__ == "__main__":
    import sys

    sys.exit(main())
