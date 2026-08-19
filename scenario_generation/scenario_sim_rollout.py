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
from scenario_generation.inference_compile import mark_inference_step
from scenario_generation.metrics.object import score_object_step
from scenario_generation.perf_timer import Timers
from scenario_generation.scenario_sim_metrics import build_segment_row
from scenario_generation.scenario_sim_route import map_from_osc, resolve_route
from scenario_generation.scenario_sim_scene import (
    DT,
    HistoryBuffers,
    SceneConfig,
    baselink_xyh,
    build_scene,
    ego_metric_box,
    resolve_ego_name,
    update_history,
)
from scenario_generation.scene_trace import SceneTraceWriter, write_map_asset
from scenario_generation.simulate import (
    _ego_to_world,
    _predict_batch,
    resolve_keep_turn_indicator,
)
from scenario_generation.tensor_converter import _build_neighbor_agents_past
from scenario_generation.tools.eval_cl_trajectory import (
    border_segments_from_map,
    evaluate_trajectory,
)
from scenario_generation.transforms import _rotation_matrix

# Same default the reproducer path uses, so the strong_brake metric is comparable.
STRONG_BRAKE_MPS2 = -2.5

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
    # Run the interpreter in a child process instead of this one. SimulatorCore is a static
    # singleton, so a process that has driven one scenario cannot drive another; isolating the
    # simulator is what lets a caller outlive a scenario and keep the model, CUDA context and
    # parsed map resident across scenarios.
    sim_in_subprocess: bool = False
    # Save a PNG every N ticks; None renders nothing. Per-step matplotlib is the dominant cost
    # of a rollout that is otherwise all inference, so an evaluation that only wants metrics
    # leaves it off. Metrics do not depend on it.
    draw_every: int | None = None
    # Browser scene replay is the default reporting artifact.  It records vector geometry and
    # does not import matplotlib or invoke ffmpeg.
    write_scene_trace: bool = True
    scene: SceneConfig = field(default_factory=SceneConfig)

    def __post_init__(self) -> None:
        # DT drives the plan-point speeds, SceneContext.dt and the acceleration first
        # difference, while fps drives the simulator. A mismatch produces a complete row with
        # silently wrong speeds and brake counts, so it has to fail here.
        if abs(self.fps * DT - 1.0) > 1e-9:
            raise ValueError(f"fps={self.fps} does not match the model timestep DT={DT}")


@torch.no_grad()
def _predict_ego_plan(model, model_args, scene, device, ego_name: str) -> tuple[np.ndarray, int]:
    """Run the model as ego -> (ego-frame plan ``(future_len, 4)``, turn-indicator class).

    No ``map_cache`` is passed: the cache only pays off across steps that share one
    ``map_data``, and this loop rebuilds it around the ego every tick.
    """
    # No-op unless the model was compiled with cudagraphs; one inference is one step.
    mark_inference_step()
    preds, tis = _predict_batch(
        model,
        model_args,
        scene,
        [ego_name],
        device,
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


def _score_neighbors(scene, ego_state: dict, device: str, ego_name: str) -> tuple[float, bool]:
    """Instantaneous (min_clearance, collision) from raw ego-frame neighbour OBBs."""
    ex, ey, eh = baselink_xyh(ego_state)
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
    x, y, h = baselink_xyh(ego_state)
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

    x0, y0, _ = baselink_xyh(runner.get_ego_state(ego_ref=ego_name))
    ego0_xy = np.array([x0, y0], dtype=np.float32)
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


def _write_rollout_trace(
    output_dir: Path,
    *,
    trajectory_log: list[dict],
    clearances: list[float],
    collisions: list[bool],
    rb_dists,
    terminated_reason: str,
) -> None:
    """Write ``rollout.jsonl`` next to the PNGs: one line per sim step, plus a terminated line.

    The schema is the reproducer's, because the readers are shared: ``trajectory_colormap``
    takes ``ego`` (required) plus ``speed`` / ``clearance_m`` / ``collision`` / ``rb_dist_m``,
    and skips any line carrying an ``event`` key. Road-border distance only exists after the
    rollout -- it is computed over the whole trajectory at once -- so the trace is written here
    rather than streamed, which is the one place this path differs from the reproducer's.

    ``red_light_violation`` is deliberately absent: this path does not observe traffic lights,
    and a present-but-always-false key would render as a measured zero.
    """
    # Scoring starts once the history buffer is warm and never stops, so the metric series is
    # the trailing part of the run. Pad the head to line the samples up with the poses they
    # colour; an unscored tick has no clearance rather than a clearance of zero.
    pad = len(trajectory_log) - len(clearances)
    clearances = [float("nan")] * pad + list(clearances)
    collisions = [False] * pad + list(collisions)
    with (output_dir / "rollout.jsonl").open("w", encoding="utf-8") as f:
        for k, entry in enumerate(trajectory_log):
            clr = clearances[k]
            rb = float(rb_dists[k])
            f.write(
                json.dumps(
                    {
                        "k": k,
                        "ego": [round(float(entry["x"]), 3), round(float(entry["y"]), 3)],
                        "yaw": round(float(entry["heading"]), 4),
                        "dist_goal": round(float(entry["goal_d"]), 3),
                        "speed": round(float(entry["speed"]), 3),
                        "clearance_m": round(float(clr), 4) if np.isfinite(clr) else None,
                        "collision": bool(collisions[k]),
                        "rb_dist_m": round(rb, 4) if np.isfinite(rb) else None,
                    }
                )
                + "\n"
            )
        f.write(
            json.dumps(
                {"event": "terminated", "k": len(trajectory_log), "reason": terminated_reason}
            )
            + "\n"
        )


def _finalize_row(
    output_dir: Path,
    *,
    builder: LaneletSceneBuilder,
    trajectory_log: list[dict],
    ego_state: dict | None,
    cfg: RolloutConfig,
    clearances: list[float],
    collisions: list[bool],
    terminated_reason: str,
    result_kind: str,
    coord_err: float,
    borders,
) -> dict:
    """Dump the trajectory, compute post-hoc road-border metrics and build the row."""
    (output_dir / "trajectory_log.json").write_text(json.dumps(trajectory_log))

    ego_len, ego_w, ego_wb = (
        ego_metric_box(ego_state) if ego_state is not None else _FALLBACK_EGO_BOX
    )
    rb, series = evaluate_trajectory(trajectory_log, borders, ego_len, ego_w, ego_wb)
    _write_rollout_trace(
        output_dir,
        trajectory_log=trajectory_log,
        clearances=clearances,
        collisions=collisions,
        rb_dists=series["rb_dists"],
        terminated_reason=terminated_reason,
    )

    row = build_segment_row(
        n_steps_run=len(trajectory_log),
        terminated=terminated_reason,
        result_kind=result_kind,
        clearances=clearances,
        collisions=collisions,
        rb_dists=series["rb_dists"],
        speeds=series["speeds"],
        dt=DT,
        near_miss_thresh=cfg.near_miss_thresh,
        strong_brake_mps2=STRONG_BRAKE_MPS2,
        progress_m=rb["progress_m"],
    )
    # scenario_sim-only diagnostics, kept flat (outside the shared category blocks) so
    # aggregate never sees them as a metric category.
    return {
        **row,
        "worst_step": int(np.argmin(clearances)) if clearances else -1,
        "rb_has_data": rb["rb_has_data"],
        "coord_check_ok": bool(coord_err <= cfg.coord_check_tol_m),
        "coord_check_err_m": coord_err,
    }


def run_scenario_sim_rollout(
    model,
    model_args,
    osc_path: str | Path,
    output_dir: str | Path,
    map_path: str | Path | None = None,
    *,
    config: RolloutConfig | None = None,
    device: str = "cpu",
    verbose: bool = True,
    timers: Timers | None = None,
    builder: LaneletSceneBuilder | None = None,
    scene_asset_dir: str | Path | None = None,
) -> dict:
    """Run one closed-loop OpenSCENARIO rollout and return an aggregate-ready row.

    ``model`` / ``model_args`` follow the ``run_closed_loop_eval`` contract
    (``model(data) -> (_, outputs)`` with ``outputs["prediction"]``; ``model_args`` provides
    ``observation_normalizer`` / ``predicted_neighbor_num`` / ``future_len``). ``builder`` lets
    a caller that outlives one scenario reuse a parsed map, which is per-map work a
    per-scenario process would otherwise pay per scenario.
    """
    # Derived from the scenario by default so the Python-side geometry cannot be computed
    # against a different map than the interpreter loaded. Pass it only to test a substitute.
    if map_path is None:
        map_path = map_from_osc(osc_path)
    cfg = config or RolloutConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timers = timers or Timers()
    _t_rollout = time.perf_counter()
    with timers("map_build"):
        if builder is None:
            builder = LaneletSceneBuilder(str(map_path))
        elif builder.lanelet_path != str(map_path):
            # Route, centreline and road-border geometry all come off the builder's map. A
            # builder carried over from another scenario would compute every one of them
            # against a map the interpreter did not load.
            raise ValueError(
                f"builder holds {builder.lanelet_path}, but this scenario declares {map_path}"
            )
    buffers = HistoryBuffers()
    trajectory_log: list[dict] = []
    clearances: list[float] = []
    collisions: list[bool] = []
    trace_writer: SceneTraceWriter | None = None

    cached_plan_ego: np.ndarray | None = None  # (future_len, 4) ego frame
    ti_cmd = 0  # last sim turn-indicator command (KEEP retains this)
    # NaN until the first stepped tick; every comparison against the tolerance is then False,
    # so a rollout that never stepped reports the check as failed rather than as passed.
    coord_err: float = float("nan")
    ego_state: dict | None = None  # last tick's ego truth, for the finalize ego shape
    terminated_reason = "max_steps"

    # The interpreter's JUnit result is the only place the reason a scenario was rejected
    # survives -- on_configure reports failure as a lifecycle state and nothing else. write_to()
    # silently does nothing when its directory is missing, so it has to exist before the run.
    osp_out = output_dir / "osp_out"
    osp_out.mkdir(parents=True, exist_ok=True)

    # Off the map the builder already holds: re-loading the .osm would keep a second copy of the
    # whole map alive alongside it. Derived once here because both the drawing (every tick) and
    # the post-hoc road-border metrics (always) read the same polylines.
    borders = border_segments_from_map(builder._lanelet_map)

    # Imported only when it is going to be used: the renderer pulls in matplotlib and the whole
    # replay module, which an evaluation that only wants metrics has no reason to pay for.
    save_step_figure = None
    if cfg.draw_every:
        from scenario_generation.replay import save_step_figure

    # RemoteRunner is duck-type compatible for every call the loop below makes, so the tick loop
    # is identical either way -- deliberately: the loop is where fidelity lives. openscenario_python
    # is imported only on the in-process branch; the subprocess branch must not pull the
    # interpreter into a parent that will never drive it.
    if cfg.sim_in_subprocess:
        from scenario_generation.sim_proc_client import RemoteRunner as _Runner
    else:
        import openscenario_python as osp  # requires the SSV2_HEADLESS_EGO overlay

        _Runner = osp.HeadlessRunner

    # Measured across the ``with`` header: opening the sim is the constructor plus __enter__,
    # which no context manager of ours can wrap.
    _t = time.perf_counter()
    with _Runner(
        osc_path=str(osc_path),
        output_directory=str(osp_out),
        local_frame_rate=cfg.fps,
    ) as runner:
        timers.add("sim_open", time.perf_counter() - _t)
        with timers("route_resolve"):
            ego_name, ego_route_ids, goal_pose = _start_and_resolve_route(
                runner, builder, osc_path, cfg, verbose
            )
        goal_xy = goal_pose[:2]
        if cfg.write_scene_trace:
            # Evaluation supplies one run-level directory.  The fallback keeps the standalone
            # worker useful without duplicating map data within one case directory.
            assets = Path(scene_asset_dir) if scene_asset_dir is not None else output_dir / "scene_maps"
            with timers("scene_trace_map"):
                map_ref, _ = write_map_asset(builder, assets)
            route = [
                [round(float(x), 3), round(float(y), 3)]
                for lane_id in ego_route_ids
                for x, y in builder.raw_centerline(lane_id)[:, :2]
            ]
            trace_writer = SceneTraceWriter(
                output_dir / "scene_trace.jsonl.gz",
                map_ref=map_ref,
                route=route,
                goal=[round(float(goal_pose[0]), 3), round(float(goal_pose[1]), 3)],
            )

        for step in range(cfg.max_steps):
            with timers("sim_get_states"):
                states = runner.get_entity_states()
            if ego_name not in states:
                raise RuntimeError(f"Sim stopped reporting the ego entity '{ego_name}'")
            ego_state = states[ego_name]
            ex, ey, eh = baselink_xyh(ego_state)

            with timers("scene_build"):
                update_history(buffers, states, ego_name)
                scene = build_scene(
                    states, buffers, builder, ego_route_ids, goal_pose, cfg.scene, ego_name
                )

            # Replan every ``replan_interval`` ticks; consume the cached plan in between.
            if cached_plan_ego is None or step % cfg.replan_interval == 0:
                # The first inference is cold (lazy allocation, kernel autotuning), so it is
                # timed as its own stage and kept out of the steady-state ms/call.
                with timers("predict_cold" if cached_plan_ego is None else "predict"):
                    cached_plan_ego, ti_model = _predict_ego_plan(
                        model, model_args, scene, device, ego_name
                    )
                ti_cmd = resolve_keep_turn_indicator(ti_model, ti_cmd)

            with timers("sim_set_traj"):
                pts = _ego_plan_to_map_trajectory(cached_plan_ego, ex, ey, eh)
                runner.set_ego_trajectory(pts, ego_ref=ego_name)
                runner.set_ego_turn_indicator(int(ti_cmd), ego_ref=ego_name)

            if buffers.age[ego_name] >= cfg.warmup_steps:
                with timers("score_objects"):
                    clr, coll = _score_neighbors(scene, ego_state, device, ego_name)
                clearances.append(clr)
                collisions.append(coll)

            if trace_writer is not None:
                # `scene` and `pts` are the exact pre-step objects the legacy renderer used.
                # The first unscored warm-up ticks intentionally carry null metrics.
                trace_writer.write_frame(
                    step,
                    scene,
                    pts,
                    clearance=(clr if buffers.age[ego_name] >= cfg.warmup_steps else None),
                    collision=(coll if buffers.age[ego_name] >= cfg.warmup_steps else None),
                )

            if save_step_figure is not None and step % cfg.draw_every == 0:
                with timers("draw"):
                    # The scene is map-frame and the plan is ego-frame, which is the pair the
                    # renderer expects: it re-centres the plan on the ego pose it draws.
                    save_step_figure(
                        scene,
                        {ego_name: cached_plan_ego},
                        output_dir / f"{step:05d}.png",
                        step,
                        cfg.max_steps,
                        route_lanelet_ids=ego_route_ids,
                        road_border_polylines=borders,
                    )

            # step() is the sole integrator: it advances BOTH the ego and the NPCs.
            with timers("sim_step"):
                outcome = runner.step()

            ego_after = runner.get_ego_state(ego_ref=ego_name)
            trajectory_log.append(_traj_entry(step, ego_after, goal_xy))
            if step == 0:  # verify the frame contract on the first stepped tick
                ax, ay, _ = baselink_xyh(ego_after)
                coord_err = float(math.hypot(ax - pts[0, 0], ay - pts[0, 1]))
                if verbose:
                    verdict = "OK" if coord_err <= cfg.coord_check_tol_m else "FAIL"
                    print(
                        f"  [scenario_sim] coord check: err={coord_err:.3f} m "
                        f"(tol {cfg.coord_check_tol_m}) -> {verdict}"
                    )

            if outcome == "terminated":
                terminated_reason = "sim_terminated"
                break

        result_kind = runner.result_kind()

    if trace_writer is not None:
        trace_writer.close(terminated_reason)

    with timers("finalize"):
        row = _finalize_row(
            output_dir,
            builder=builder,
            trajectory_log=trajectory_log,
            ego_state=ego_state,
            cfg=cfg,
            clearances=clearances,
            collisions=collisions,
            terminated_reason=terminated_reason,
            result_kind=result_kind,
            coord_err=coord_err,
            borders=borders,
        )
    # Teardown happens in HeadlessRunner.__exit__, so it is not a stage of its own: it lands in
    # rollout_total minus the parts.
    timers.add("rollout_total", time.perf_counter() - _t_rollout)
    row["map_path"] = str(map_path)
    row["timing"] = timers.as_dict()
    return row
