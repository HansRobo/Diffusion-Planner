"""The reference-point contract between simulator truth and ``SceneContext``.

The simulator reports an entity's reference point plus the offset to its bbox centre, and the
scene wants the ego at the reference point and every other agent at its centroid. Storing the
reported pose for a neighbour puts its box a whole centre-offset behind the truth -- in the model
input and in the metric OBBs alike -- so what these tests pin is which entity gets which frame.
"""

import math

import numpy as np
import pytest

from scenario_generation.scenario_sim_rollout import _score_neighbors
from scenario_generation.scenario_sim_scene import (
    DT,
    HistoryBuffers,
    SceneConfig,
    baselink_xyh,
    centroid_xyh,
    entity_shape,
    update_history,
)
from scenario_generation.scene_context import Agent, AgentType, MapData, SceneContext
from scenario_generation.tensor_converter import _build_neighbor_agents_past
from scenario_generation.transforms import _rotation_matrix

# sample_vehicle: rear axle at the origin, box centre 1.355 m ahead of it.
CENTER_X = 1.355


def _state(type_, x=0.0, y=0.0, yaw=0.0, center_x=CENTER_X, center_y=0.0, length=4.77):
    return {
        "type": type_,
        "pose": {"x": x, "y": y, "yaw": yaw},
        "bounding_box": {
            "center": {"x": center_x, "y": center_y, "z": 1.25},
            "dimensions": {"x": length, "y": 1.83, "z": 2.5},
        },
    }


@pytest.mark.parametrize(
    "yaw, expected",
    [
        (0.0, (10.0 + CENTER_X, 20.0)),
        (math.pi / 2, (10.0, 20.0 + CENTER_X)),
        (math.pi, (10.0 - CENTER_X, 20.0)),
    ],
)
def test_centroid_offset_follows_heading(yaw, expected):
    x, y, h = centroid_xyh(_state(1, x=10.0, y=20.0, yaw=yaw))
    assert (x, y) == pytest.approx(expected, abs=1e-6)
    assert h == pytest.approx(yaw)  # heading is unchanged by the shift


def test_lateral_center_offset_is_applied():
    """``center.y`` is rare but real, and dropping it biases lateral clearance."""
    x, y, _ = centroid_xyh(_state(1, x=0.0, y=0.0, yaw=math.pi / 2, center_x=0.0, center_y=0.5))
    assert (x, y) == pytest.approx((-0.5, 0.0), abs=1e-6)


def test_centered_reference_point_is_a_noop():
    """Pedestrians -- and vehicles catalogued that way -- report ``center.x == 0``."""
    st = _state(2, x=3.0, y=4.0, yaw=0.7, center_x=0.0, length=0.8)
    assert centroid_xyh(st) == pytest.approx(baselink_xyh(st))


def test_ego_keeps_reference_point_and_neighbours_are_centroids():
    states = {
        "ego": _state(0, x=0.0, y=0.0, yaw=0.0),
        "npc": _state(1, x=30.0, y=0.0, yaw=0.0),
        "walker": _state(2, x=5.0, y=8.0, yaw=0.0, center_x=0.0, length=0.8),
    }
    buffers = HistoryBuffers(length=3)
    update_history(buffers, states, "ego")

    assert buffers.trajectory("ego")[-1][:2] == pytest.approx([0.0, 0.0], abs=1e-6)
    assert buffers.trajectory("npc")[-1][:2] == pytest.approx([30.0 + CENTER_X, 0.0], abs=1e-6)
    assert buffers.trajectory("walker")[-1][:2] == pytest.approx([5.0, 8.0], abs=1e-6)


def test_seeded_history_is_centroid_referenced():
    """A newly seen entity's buffer is filled by repeating its pose, which must be the centroid
    too -- otherwise the whole past it is first scored on carries the offset."""
    buffers = HistoryBuffers(length=4)
    update_history(buffers, {"npc": _state(1, x=30.0, y=0.0, yaw=0.0)}, "ego")
    traj = buffers.trajectory("npc")
    assert traj.shape == (4, 3)
    assert np.allclose(traj[:, 0], 30.0 + CENTER_X, atol=1e-6)


def test_non_agents_are_not_buffered():
    buffers = HistoryBuffers(length=3)
    update_history(buffers, {"cone": _state(3, x=1.0, y=2.0)}, "ego")
    assert buffers.age == {}


def _scene(states: dict, buffers: HistoryBuffers) -> SceneContext:
    """The agent half of ``build_scene``, without the map the builder would supply."""
    agents = []
    for name, st in states.items():
        length, width, wheelbase = entity_shape(st, SceneConfig())
        agents.append(
            Agent(
                id=name,
                agent_type=AgentType.VEHICLE,
                length=length,
                width=width,
                wheelbase=wheelbase,
                past_trajectory=buffers.trajectory(name),
            )
        )
    map_data = MapData(
        lanes=np.zeros((0, 20, 33), dtype=np.float32),
        lanes_speed_limit=np.zeros((0, 1), dtype=np.float32),
        lanes_has_speed_limit=np.zeros((0, 1), dtype=bool),
        polygons=np.zeros((0, 40, 2), dtype=np.float32),
        line_strings=np.zeros((0, 20, 2), dtype=np.float32),
        static_objects=np.zeros((0, 10), dtype=np.float32),
    )
    return SceneContext(agents=agents, map_data=map_data, ego_agent_id="ego", dt=DT)


def test_neighbour_box_lands_on_the_truth_footprint():
    """The consumers, not the producer: a leading NPC reported 30 m ahead has its box centre a
    centre-offset further on, and the scored clearance is the gap between the two footprints."""
    npc_x, length = 30.0, 4.77
    states = {
        "ego": _state(0, x=0.0, y=0.0, yaw=0.0, length=length),
        "npc": _state(1, x=npc_x, y=0.0, yaw=0.0, length=length),
    }
    buffers = HistoryBuffers(length=3)
    update_history(buffers, states, "ego")
    scene = _scene(states, buffers)

    nb = _build_neighbor_agents_past(scene, "ego", _rotation_matrix(0.0), np.zeros(2), 0.0)
    assert nb[0, 0, -1, 0] == pytest.approx(npc_x + CENTER_X, abs=1e-3)

    # NPC rear to ego front; both boxes are centred a centre-offset ahead of their base_link.
    expected = (npc_x + CENTER_X - length / 2) - (CENTER_X + length / 2)
    clr, coll = _score_neighbors(scene, states["ego"], "cpu", "ego")
    assert clr == pytest.approx(expected, abs=1e-3)
    assert not coll
