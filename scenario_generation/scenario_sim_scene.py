"""Build a Diffusion-Planner ``SceneContext`` from OpenSCENARIO simulator truth.

The simulator reports poses in the map (MGRS) frame and ``LaneletSceneBuilder`` works in the
same frame, so poses are stored as-is; ``to_model_tensors`` re-centers them onto the ego.

Entities are identified by the simulator's type field rather than by name, and each one
carries a rolling pose history because the model consumes a fixed-length past.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from scenario_generation.gui.lanelet_scene_builder import LaneletSceneBuilder
from scenario_generation.scene_context import Agent, AgentType, SceneContext
from scenario_generation.tensor_converter import _INPUT_T

DT = 0.1  # sim + model timestep (10 Hz). Must match the interpreter's local_frame_rate.
_HISTORY_LEN = _INPUT_T + 1  # the past the model sees, plus the current pose.
_SIM_TYPE_EGO = 0  # get_entity_states()["type"]: 0=EGO 1=VEHICLE 2=PEDESTRIAN 3=MISC_OBJECT


@dataclass
class SceneConfig:
    """Window and shape parameters for the SceneContext snapshot."""

    max_map_lanelets: int = 140
    map_mask_range_m: float = 100.0
    route_window_segments: int = 25
    # Feeds ``Agent.wheelbase`` only, whose contract is the real axle spacing (~0.65 * length
    # when unavailable) -- what the model input has to match is the convention its training
    # data was built with. Metric geometry uses :func:`ego_metric_box` instead; the two want
    # different numbers for the same vehicle.
    wheelbase_ratio: float = 0.65


def resolve_ego_name(states: dict) -> str:
    """The ego's entity name as the simulator reports it.

    Identified by type, not by name: ``type == 0`` is authoritative whatever the scenario
    chose to call the entity. Matching on the name instead fails at spawn time with a message
    that points at the symptom rather than the cause.
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


def _agent_type(sim_type: int) -> AgentType | None:
    """Map the simulator's entity type to ``AgentType``; ``None`` for non-agents."""
    if sim_type in (0, 1):  # EGO, VEHICLE
        return AgentType.VEHICLE
    if sim_type == 2:  # PEDESTRIAN
        return AgentType.PEDESTRIAN
    return None  # MISC_OBJECT is not a dynamic agent


class HistoryBuffers:
    """Per-entity rolling ``(x, y, heading)`` history keyed by the stable sim name.

    A newly seen entity has its buffer filled by repeating the current pose rather than
    zeros, so no teleport artifact enters the model input. ``age`` counts the real ticks each
    entity has accumulated, so scoring can wait until the history is warm.
    """

    def __init__(self, length: int = _HISTORY_LEN):
        self.length = length
        self._buf: dict[str, deque] = {}
        self.age: dict[str, int] = {}

    def update(self, name: str, x: float, y: float, yaw: float) -> None:
        if name not in self._buf:
            self._buf[name] = deque([(x, y, yaw)] * self.length, maxlen=self.length)
            self.age[name] = 1
        else:
            self._buf[name].append((x, y, yaw))
            self.age[name] += 1

    def trajectory(self, name: str) -> np.ndarray:
        return np.array(self._buf[name], dtype=np.float32)  # (length, 3)


def pose_xyh(state: dict) -> tuple[float, float, float]:
    """The one place the simulator's pose dict shape is read."""
    p = state["pose"]
    return float(p["x"]), float(p["y"]), float(p["yaw"])


def update_history(buffers: HistoryBuffers, states: dict) -> None:
    """Append this tick's truth pose to each dynamic entity's rolling buffer."""
    for name, st in states.items():
        if _agent_type(int(st["type"])) is not None:
            buffers.update(name, *pose_xyh(st))


def entity_shape(state: dict, cfg: SceneConfig) -> tuple[float, float, float]:
    """``(length, width, axle_wheelbase)`` for ``Agent`` -- model input, not metrics."""
    dims = state["bounding_box"]["dimensions"]
    length, width = float(dims["x"]), float(dims["y"])
    return length, width, cfg.wheelbase_ratio * length


def ego_metric_box(state: dict) -> tuple[float, float, float]:
    """``(length, width, box_wheelbase)`` for the metric OBB builders.

    ``box_wheelbase`` is not the axle spacing. The OBB builders treat the reported pose as the
    rear-axle midpoint and assume symmetric overhangs, placing the box centre ``wheelbase / 2``
    ahead of it -- so what they need is twice the true centre offset, which the simulator
    reports directly. Estimating it from ``length`` biases clearance in opposite directions
    fore and aft on a vehicle with asymmetric overhangs, which cannot be corrected afterwards.
    """
    bbox = state["bounding_box"]
    dims = bbox["dimensions"]
    return float(dims["x"]), float(dims["y"]), 2.0 * float(bbox["center"]["x"])


def build_scene(
    states: dict,
    buffers: HistoryBuffers,
    builder: LaneletSceneBuilder,
    ego_route_ids: list[int],
    goal_pose: np.ndarray,
    cfg: SceneConfig,
    ego_name: str,
) -> SceneContext:
    """Build a SceneContext snapshot in the map frame from this tick's sim truth."""
    ex, ey, _ = pose_xyh(states[ego_name])
    ego_xy = np.array([ex, ey], dtype=np.float32)

    # Closest-N lanelets around the ego, with the ego route pinned first so route context can
    # never be dropped by the distance cut.
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

    # Ego route_lanes: a forward sliding window refreshed every tick.
    window = (
        builder.select_route_segment_indices(
            ego_route_ids, ego_xy, max_segments=cfg.route_window_segments
        )
        or ego_route_ids[: cfg.route_window_segments]
    )
    route_lanes, route_sl, route_hsl = builder._route_to_33dim(
        window, max_segments=cfg.route_window_segments
    )

    agents: list[Agent] = []
    for name, st in states.items():
        atype = _agent_type(int(st["type"]))
        if atype is None:
            continue
        length, width, wheelbase = entity_shape(st, cfg)
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
                past_velocities=None,  # derived from position diffs in the converter
                goal_pose=goal_pose.astype(np.float32) if is_ego else None,
                route_lanes=route_lanes if is_ego else None,
                route_speed_limit=route_sl if is_ego else None,
                route_has_speed_limit=route_hsl if is_ego else None,
                turn_indicators=(np.zeros(traj.shape[0], dtype=np.int32) if is_ego else None),
                route_lanelet_ids=list(ego_route_ids) if is_ego else None,
                age_steps=buffers.age[name],
            )
        )
    return SceneContext(agents=agents, map_data=map_data, ego_agent_id=ego_name, dt=DT)
