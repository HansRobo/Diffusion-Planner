"""Autoware classification -> Diffusion Planner class mapping.

Only Autoware's HAZARD classification should map to the model's Unknown class (index 3,
one-hot column 11); every other class keeps its existing mapping, and Autoware's own
UNKNOWN classification continues to be dropped, unchanged.

This module requires ``autoware_perception_msgs`` (a ROS/colcon-only dependency not
installed in a bare Python dev environment), so it's skipped outside a colcon test run --
see the module docstring in ``diffusion_planner_ros/utils.py`` for the runtime this
actually executes in.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

autoware_perception_msgs = pytest.importorskip("autoware_perception_msgs.msg")
ObjectClassification = autoware_perception_msgs.ObjectClassification

from diffusion_planner_ros.utils import (  # noqa: E402
    convert_tracked_objects_to_tensor,
    tracking_one_step,
)


def _classification(label: int, probability: float = 1.0):
    return SimpleNamespace(label=label, probability=probability)


def _pose(x=0.0, y=0.0, z=0.0):
    return SimpleNamespace(
        position=SimpleNamespace(x=x, y=y, z=z),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )


def _object(object_id_byte: int, label: int, x: float = 5.0, y: float = 0.0):
    return SimpleNamespace(
        object_id=SimpleNamespace(uuid=[object_id_byte] * 16),
        classification=[_classification(label)],
        kinematics=SimpleNamespace(
            pose_with_covariance=SimpleNamespace(pose=_pose(x, y)),
            twist_with_covariance=SimpleNamespace(
                twist=SimpleNamespace(linear=SimpleNamespace(x=0.0, y=0.0, z=0.0))
            ),
        ),
        shape=SimpleNamespace(dimensions=SimpleNamespace(x=4.5, y=1.8)),
    )


def _tracked_objects(*objects):
    return SimpleNamespace(objects=list(objects))


def test_hazard_maps_to_unknown_one_hot_column():
    msg = _tracked_objects(_object(1, ObjectClassification.HAZARD))
    tracked = tracking_one_step(msg, {})
    assert len(tracked) == 1
    tracking_obj = next(iter(tracked.values()))
    assert tracking_obj.class_label == 3

    neighbor = convert_tracked_objects_to_tensor(
        tracked, map2bl_matrix_4x4=__import__("numpy").eye(4), max_num_objects=1, max_timesteps=1
    )
    assert neighbor[0, 0, 0, 8:12].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_true_unknown_is_still_dropped():
    msg = _tracked_objects(_object(1, ObjectClassification.UNKNOWN))
    tracked = tracking_one_step(msg, {})
    assert tracked == {}


def test_other_classes_unaffected_by_hazard_change():
    msg = _tracked_objects(
        _object(1, ObjectClassification.CAR),
        _object(2, ObjectClassification.PEDESTRIAN),
        _object(3, ObjectClassification.BICYCLE),
    )
    tracked = tracking_one_step(msg, {})
    labels = sorted(t.class_label for t in tracked.values())
    assert labels == [0, 1, 2]


def test_unrecognized_label_is_dropped_not_a_keyerror():
    """ANIMAL (and any other class not in the map) must drop safely like UNKNOWN, not
    KeyError-crash the node."""
    msg = _tracked_objects(_object(1, ObjectClassification.ANIMAL))
    tracked = tracking_one_step(msg, {})
    assert tracked == {}
