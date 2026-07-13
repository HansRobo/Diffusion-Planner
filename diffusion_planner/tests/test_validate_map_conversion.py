"""Regression tests for explicit map-point validity in validation adapters."""

import torch
from diffusion_planner.validate_model import (
    _lane_tensor_to_centerlines,
    _lane_tensor_to_polygons,
    _line_strings_to_epdms_border_lines,
)


def test_epdms_border_keeps_valid_origin_point():
    line_strings = torch.zeros(1, 1, 2, 4)
    line_strings[0, 0, :, 3] = 1.0
    line_strings[0, 0, 1, 0] = 2.0

    border_lines = _line_strings_to_epdms_border_lines(line_strings)

    assert len(border_lines[0]) == 1
    assert border_lines[0][0].tolist() == [[0.0, 0.0], [2.0, 0.0]]


def test_lane_adapters_keep_valid_origin_point_from_geometry_channels():
    lanes = torch.zeros(1, 1, 2, 33)
    # The first center point is at the ego origin, but its direction and
    # left/right boundary offsets make it an explicitly valid lane point.
    lanes[0, 0, :, 2] = 1.0
    lanes[0, 0, :, 4:6] = torch.tensor([0.0, 2.0])
    lanes[0, 0, :, 6:8] = torch.tensor([0.0, -2.0])
    lanes[0, 0, 1, 0] = 2.0

    centerlines = _lane_tensor_to_centerlines(lanes)
    polygons = _lane_tensor_to_polygons(lanes)

    assert len(centerlines[0]) == 1
    assert centerlines[0][0].tolist() == [[0.0, 0.0], [2.0, 0.0]]
    assert len(polygons[0]) == 1
