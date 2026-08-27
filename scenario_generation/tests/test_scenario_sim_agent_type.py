"""scenario_sim entity type -> AgentType.

MISC_OBJECT used to be dropped, because before the Unknown class existed there was nothing
truthful to call it: forcing it to VEHICLE would have lied about its type, so the scene lost
an obstacle the planner has to avoid. Unknown is exactly that "not one of the three" label,
so it is now mapped rather than discarded.
"""

import numpy as np
import pytest

from scenario_generation.scenario_sim_scene import _agent_type
from scenario_generation.scene_context import Agent, AgentType, SceneContext
from scenario_generation.tensor_converter import _build_neighbor_agents_past


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"type": 0, "subtype": 0}, AgentType.VEHICLE),  # EGO
        ({"type": 1, "subtype": 0}, AgentType.VEHICLE),
        ({"type": 1, "subtype": 5}, AgentType.BICYCLE),  # motorcycle
        ({"type": 1, "subtype": 6}, AgentType.BICYCLE),
        ({"type": 2}, AgentType.PEDESTRIAN),
        ({"type": 3}, AgentType.UNKNOWN),  # MISC_OBJECT
    ],
)
def test_known_sim_types_map_to_an_agent_type(state, expected):
    assert _agent_type(state) == expected


def test_unrecognised_sim_type_is_still_dropped():
    """Only types the sim actually documents get a label; anything else has no honest one."""
    assert _agent_type({"type": 9}) is None


def test_misc_object_reaches_the_model_as_the_unknown_one_hot():
    """The point of mapping it at all: a MISC_OBJECT has to survive all the way into the
    model input as Unknown, not as a vehicle via an argmax tie-break on an empty one-hot."""
    agents = [
        Agent(
            id=name,
            agent_type=_agent_type(state),
            length=4.0,
            width=2.0,
            wheelbase=2.6,
            past_trajectory=np.tile(np.array([x, 0.0, 0.0], dtype=np.float32), (31, 1)),
        )
        for name, state, x in [
            ("ego", {"type": 0, "subtype": 0}, 0.0),
            ("misc", {"type": 3}, 10.0),
        ]
    ]
    scene = SceneContext(agents=agents, map_data=None, ego_agent_id="ego")

    nap = _build_neighbor_agents_past(
        scene,
        ego_id="ego",
        R=np.eye(2, dtype=np.float32),
        ego_xy=np.zeros(2, dtype=np.float32),
        ego_heading=0.0,
        num_neighbors=4,
    )

    assert nap.shape[-1] == 12
    assert nap[0, 0, -1, 8:12].tolist() == [0.0, 0.0, 0.0, 1.0]
