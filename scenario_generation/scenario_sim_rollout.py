"""OpenSCENARIO-driven closed-loop rollout for the Diffusion-Planner validator.

The ``scenario_sim`` sibling of the NPZ-replay path: it drives the C++ OpenSCENARIO
interpreter (``openscenario_python.HeadlessRunner``, built with SSV2_HEADLESS_EGO) and runs
the live Diffusion-Planner as the ego on every tick -- read sim truth, build a
:class:`~scenario_generation.scene_context.SceneContext` snapshot, infer, inject the plan,
``step()``, repeat.

Invariant: the Python scene is never advanced. The C++ ``step()`` is the sole integrator for
both the ego and the NPCs; Python only reads truth and injects a plan. Advancing the Python
scene as well would double-simulate and diverge from sim truth.

Frame contract: the sim reports map-frame poses, which the SceneContext stores as-is;
``to_model_tensors`` re-centers onto the ego, and the ego-frame plan is mapped back with the
current ego pose before injection.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from scenario_generation.gui.lanelet_scene_builder import LaneletSceneBuilder
from scenario_generation.metrics.object import score_object_step
from scenario_generation.perf_timer import Timers
from scenario_generation.scenario_sim_metrics import build_segment_row
from scenario_generation.scenario_sim_route import resolve_route
from scenario_generation.scenario_sim_scene import (
    DT,
    HistoryBuffers,
    SceneConfig,
    build_scene,
    ego_metric_box,
    pose_xyh,
    resolve_ego_name,
    update_history,
)
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

# Same default the reproducer path uses, so the strong_brake metric is comparable.
STRONG_BRAKE_MPS2 = -2.5

# Model turn-indicator classes (decode_turn_indicator): 0 NONE / 1 DISABLE / 2 LEFT /
# 3 RIGHT / 4 KEEP. Sim TurnIndicatorsCommand: 0 NO_COMMAND / 1 DISABLE / 2 ENABLE_LEFT /
# 3 ENABLE_RIGHT. KEEP has no sim counterpart and retains the previous command.
_TI_MODEL_TO_SIM = {0: 0, 1: 1, 2: 2, 3: 3}

# Used only when the sim never reported an ego state, i.e. the scenario ended before the
# first tick and the trajectory log is empty, so the value cannot affect any metric.
_FALLBACK_EGO_BOX = (4.0, 1.8, 2.6)


@dataclass
class RolloutConfig:
    """Tuning for a single scenario_sim rollout."""

    # Must satisfy fps == 1 / DT: the sim step and the model timestep are the same tick.
    fps: float = 10.0
    # 1 = every tick = 10 Hz, matching the production node's planning_frequency_hz. Values
    # > 1 consume a cached plan open-loop in between: cheaper, less reactive.
    replan_interval: int = 1
    max_steps: int = 300
    warmup_steps: int = 5  # skip scoring until the history buffer is warm
    near_miss_thresh: float = 1.0
    find_route_min_len_m: float = 120.0
    # Coordinate-contract check: after the first stepped tick the ego's realized pose must
    # land within this tolerance (m) of the injected plan's first future point. A gross frame
    # mismatch blows past it.
    coord_check_tol_m: float = 2.0
    scene: SceneConfig = field(default_factory=SceneConfig)


def _map_turn_indicator(ti_model: int, prev_cmd: int) -> int:
    """Model turn-indicator class -> sim command. KEEP retains ``prev_cmd``."""
    return _TI_MODEL_TO_SIM.get(ti_model, prev_cmd)


@torch.no_grad()
def _predict_ego_plan(
    model, model_args, scene, device, map_cache, ego_name: str
) -> tuple[np.ndarray, int]:
    """Run the model as ego -> (ego-frame plan ``(future_len, 4)``, turn-indicator class)."""
    preds, tis = _predict_batch(
        model,
        model_args,
        scene,
        [ego_name],
        device,
        map_cache=map_cache,
        return_turn_indicators=True,
    )
    return preds[ego_name], int(tis.get(ego_name, 0))


def _ego_plan_to_map_trajectory(
    plan_ego: np.ndarray, ex: float, ey: float, eh: float
) -> np.ndarray:
    """Ego-frame plan -> map-frame ``[N, 4]`` of (x, y, yaw, longitudinal v) for
    ``set_ego_trajectory``, using the current ego pose as the frame origin."""
    world_xy, world_h = _ego_to_world(plan_ego[:, :2], plan_ego[:, 2:4], ex, ey, eh)
    seg = np.linalg.norm(np.diff(world_xy, axis=0), axis=1)  # plan has >= 2 points
    speeds = np.concatenate([seg[:1], seg]) / DT
    return np.column_stack([world_xy, world_h, speeds]).astype(float)


def _score_neighbors(
    scene, ego_state: dict, ex: float, ey: float, eh: float, device: str, ego_name: str
) -> tuple[float, bool]:
    """Instantaneous (min_clearance, collision) from raw ego-frame neighbour OBBs."""
    R = _rotation_matrix(eh)
    neighbors_live = _build_neighbor_agents_past(
        scene, ego_name, R, np.array([ex, ey], dtype=np.float64), eh
    )[0, :, -1, :]
    # (box_wheelbase, length, width) -- metric geometry, so ego_metric_box and not entity_shape.
    ego_shape = np.array(ego_metric_box(ego_state), dtype=np.float32)[[2, 0, 1]]
    min_clr, coll, _ = score_object_step(neighbors_live, ego_shape, device)
    return min_clr, coll


def _traj_entry(step: int, ego_state: dict, goal_xy: np.ndarray) -> dict:
    """One trajectory_log row (world pose, speed, goal distance) for post-hoc metrics."""
    x, y, h = pose_xyh(ego_state)
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
    """Configure and activate the sim, then resolve the ego's name and route.

    A scenario the interpreter rejects at parse time leaves ``configure()`` at "unconfigured",
    and calling ``get_entity_states()`` on an unconfigured core dereferences a null pointer, so
    a bad lifecycle transition has to fail here rather than be carried into the loop.
    """
    st_cfg = runner.configure()
    if st_cfg != "inactive":
        raise RuntimeError(
            f"configure() did not reach 'inactive' (got '{st_cfg}') -- scenario "
            f"rejected by the interpreter at parse/configure time: {osc_path}"
        )
    st_act = runner.activate()
    if st_act != "active":
        raise RuntimeError(f"activate() did not reach 'active' (got '{st_act}'): {osc_path}")
    ego_name = resolve_ego_name(runner.get_entity_states())

    ego0 = runner.get_ego_state(ego_ref=ego_name)
    ego0_xy = np.array([ego0["pose"]["x"], ego0["pose"]["y"]], dtype=np.float32)
    ego_route_ids = resolve_route(builder, ego0_xy, osc_path, min_len_m=cfg.find_route_min_len_m)
    if not ego_route_ids:
        raise RuntimeError("Empty ego route -- cannot build SceneContext")
    goal_pose = builder._route_goal(ego_route_ids)
    if verbose:
        print(
            f"  [scenario_sim] route={len(ego_route_ids)} lanelets, "
            f"start_ll={ego_route_ids[0]}, goal_ll={ego_route_ids[-1]}"
        )
    return ego_name, ego_route_ids, goal_pose


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
    """Dump the trajectory, compute post-hoc road-border metrics and build the row."""
    (output_dir / "trajectory_log.json").write_text(json.dumps(trajectory_log))

    ego_len, ego_w, ego_wb = (
        ego_metric_box(ego_state) if ego_state is not None else _FALLBACK_EGO_BOX
    )
    borders = load_border_segments(str(map_path))
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
        # scenario_sim-only diagnostics, kept flat (outside the shared category blocks) so
        # aggregate never sees them as a metric category.
        extra={
            "worst_step": int(np.argmin(clearances)) if clearances else -1,
            "rb_has_data": rb["rb_has_data"],
            "coord_check_ok": bool(coord_ok) if coord_ok is not None else False,
            "coord_check_err_m": coord_err,
        },
    )


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
    (``model(data) -> (_, outputs)`` with ``outputs["prediction"]``; ``model_args`` provides
    ``observation_normalizer`` / ``predicted_neighbor_num`` / ``future_len``). ``builder`` lets
    a caller that outlives one scenario reuse a parsed map, which is per-map work a
    per-scenario process would otherwise pay per scenario.
    """
    import openscenario_python as osp  # requires the SSV2_HEADLESS_EGO overlay

    cfg = config or RolloutConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timers = timers or Timers()
    _t_rollout = time.perf_counter()
    _t = time.perf_counter()
    if builder is None:
        builder = LaneletSceneBuilder(str(map_path))
    timers.add("map_build", time.perf_counter() - _t)
    buffers = HistoryBuffers()
    trajectory_log: list[dict] = []
    clearances: list[float] = []
    collisions: list[bool] = []

    cached_plan_ego: np.ndarray | None = None  # (future_len, 4) ego frame
    ti_cmd = 0  # last sim turn-indicator command (KEEP retains this)
    coord_err: float = float("nan")
    coord_ok: bool | None = None
    ego_state: dict | None = None  # last tick's ego truth, for the finalize ego shape
    terminated_reason = "max_steps"

    # The interpreter's JUnit result is the only place the reason a scenario was rejected
    # survives -- on_configure reports failure as a lifecycle state and nothing else. write_to()
    # silently does nothing when its directory is missing, so it has to exist before the run.
    osp_out = output_dir / "osp_out"
    osp_out.mkdir(parents=True, exist_ok=True)

    _t = time.perf_counter()
    with osp.HeadlessRunner(
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
            ex, ey, eh = pose_xyh(ego_state)

            with timers("scene_build"):
                update_history(buffers, states)
                scene = build_scene(
                    states, buffers, builder, ego_route_ids, goal_pose, cfg.scene, ego_name
                )
                map_cache = MapTensorCache(scene.map_data)

            # Replan every ``replan_interval`` ticks; consume the cached plan in between.
            if cached_plan_ego is None or step % cfg.replan_interval == 0:
                # The first inference is cold (lazy allocation, kernel autotuning), so it is
                # timed as its own stage and kept out of the steady-state ms/call.
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
                    clr, coll = _score_neighbors(scene, ego_state, ex, ey, eh, device, ego_name)
                clearances.append(clr)
                collisions.append(coll)

            # step() is the sole integrator: it advances BOTH the ego and the NPCs.
            with timers("sim_step"):
                outcome = runner.step()

            ego_after = runner.get_ego_state(ego_ref=ego_name)
            trajectory_log.append(_traj_entry(step, ego_after, goal_xy))
            if coord_ok is None:  # first stepped tick: verify the frame contract
                ax, ay, _ = pose_xyh(ego_after)
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
        output_dir,
        map_path,
        trajectory_log,
        ego_state,
        cfg,
        clearances,
        collisions,
        terminated_reason,
        result_kind,
        coord_ok,
        coord_err,
    )
    timers.add("finalize", time.perf_counter() - _t)
    # Teardown happens in HeadlessRunner.__exit__, so it is not a stage of its own: it lands in
    # rollout_total minus the parts.
    timers.add("rollout_total", time.perf_counter() - _t_rollout)
    row["timing"] = timers.as_dict()
    return row


__all__ = [
    "RolloutConfig",
    "run_scenario_sim_rollout",
]
