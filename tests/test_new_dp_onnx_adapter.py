from pathlib import Path

import numpy as np

from scenario_generation.new_dp_onnx import legacy_to_new_dp, normalize_new_dp


def _legacy(batch=2):
    return {
        "ego_agent_past": np.zeros((batch, 31, 4), np.float32),
        "ego_current_state": np.zeros((batch, 10), np.float32),
        "neighbor_agents_past": np.zeros((batch, 320, 31, 11), np.float32),
        "lanes": np.zeros((batch, 140, 20, 33), np.float32),
        "lanes_speed_limit": np.zeros((batch, 140, 1), np.float32),
        "route_lanes": np.zeros((batch, 25, 20, 33), np.float32),
        "route_lanes_speed_limit": np.zeros((batch, 25, 1), np.float32),
        "polygons": np.zeros((batch, 10, 40, 3), np.float32),
        "line_strings": np.zeros((batch, 60, 20, 4), np.float32),
        "goal_pose": np.zeros((batch, 4), np.float32),
        "ego_shape": np.zeros((batch, 3), np.float32),
        "turn_indicators": np.zeros((batch, 31), np.float32),
    }


def test_adapter_matches_new_onnx_shapes():
    converted = legacy_to_new_dp(_legacy())
    expected = {
        "ego_agent_past": (2, 31, 6),
        "neighbor_agents_past": (2, 320, 31, 4),
        "agent_shape": (2, 320, 2),
        "agent_label": (2, 320, 3),
        "lanes": (2, 140, 20, 6),
        "lane_types": (2, 140, 20),
        "lane_traffic_light_past": (2, 140, 31, 6),
        "lane_traffic_light_future": (2, 140, 80, 6),
        "route_lanes": (2, 25, 20, 6),
        "route_lane_types": (2, 25, 20),
        "intersection_area": (2, 10, 40, 2),
        "stop_lines": (2, 30, 2, 2),
        "road_borders": (2, 30, 20, 2),
    }
    for key, shape in expected.items():
        assert converted[key].shape == shape


def test_new_normalization_scales_only_continuous_fields():
    data = legacy_to_new_dp(_legacy(batch=1))
    data["goal_pose"][0, :2] = [50, -100]
    data["ego_shape"][0] = [10, 20, 30]
    normalized = normalize_new_dp(data)
    np.testing.assert_allclose(normalized["goal_pose"][0, :2], [1, -2])
    np.testing.assert_allclose(normalized["ego_shape"][0], [1, 2, 3])
