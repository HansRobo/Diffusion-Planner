"""Tests for route-centerline distance and lateral error metrics."""

import torch

from planner_metrics.centerline import (
    compute_centerline_distance_batch,
    compute_centerline_error_components_batch,
    evaluate_centerline,
    evaluate_centerline_with_details,
)
from planner_metrics.evaluation import detail_value_to_json


def test_centerline_metric_projects_to_segments():
    """Project each predicted point onto the nearest route-centerline segment."""
    ego_trajs = torch.tensor([[[0.5, 1.0], [2.0, -2.0]]])
    route_lanes = torch.zeros((1, 2, 8))
    route_lanes[0, :, 0] = torch.tensor([0.0, 3.0])
    route_lanes[0, :, 2] = 1.0

    distances = compute_centerline_distance_batch(
        ego_trajs,
        {"route_lanes": route_lanes},
    )

    torch.testing.assert_close(distances, torch.tensor([[1.0, 2.0]]))


def test_centerline_metric_ignores_duplicate_point_segments():
    """A duplicate centerline point must not report a spurious zero error."""
    prediction = torch.tensor([[[10.0, 0.0]]])
    route_lanes = torch.zeros((1, 3, 8))
    route_lanes[0, :, 0] = torch.tensor([0.0, 0.0, 3.0])
    route_lanes[0, :, 2] = 1.0

    distances = compute_centerline_distance_batch(prediction, {"route_lanes": route_lanes})
    components = compute_centerline_error_components_batch(prediction, {"route_lanes": route_lanes})

    torch.testing.assert_close(distances, torch.tensor([[7.0]]))
    torch.testing.assert_close(components["lateral_error_m"], torch.tensor([[0.0]]))
    torch.testing.assert_close(components["longitudinal_error_m"], torch.tensor([[7.0]]))


def test_centerline_metric_returns_lateral_errors_at_requested_horizon():
    """Compute average/final lateral errors using the configured horizon."""
    prediction = torch.tensor([[[0.0, 1.0, 1.0, 0.0], [1.0, 2.0, 1.0, 0.0], [2.0, 3.0, 1.0, 0.0]]])
    # [batch, singleton context, lane, point, feature]
    route_lanes = torch.zeros((1, 1, 1, 4, 8))
    route_lanes[0, 0, 0, :, 0] = torch.arange(4)
    route_lanes[0, 0, 0, :, 2] = 1.0

    values = evaluate_centerline(
        prediction,
        {"route_lanes": route_lanes},
        {"horizon_seconds": 0.2},
    )

    torch.testing.assert_close(values["average_lateral_error_m"], torch.tensor([1.5]))
    torch.testing.assert_close(values["final_lateral_error_m"], torch.tensor([2.0]))
    assert set(values) == {
        "average_lateral_error_m",
        "final_lateral_error_m",
        "lateral_in_band_rate",
    }


def test_centerline_error_components_separate_endpoint_overshoot():
    """Endpoint overshoot is longitudinal, not lateral, error."""
    prediction = torch.tensor([[[4.0, 0.0], [5.0, 1.0]]])
    route_lanes = torch.zeros((1, 2, 8))
    route_lanes[0, :, 0] = torch.tensor([0.0, 3.0])
    route_lanes[0, :, 2] = 1.0

    values = compute_centerline_error_components_batch(
        prediction,
        {"route_lanes": route_lanes},
    )

    torch.testing.assert_close(values["lateral_error_m"], torch.tensor([[0.0, 1.0]]))
    torch.testing.assert_close(values["longitudinal_error_m"], torch.tensor([[1.0, 2.0]]))


def _x_axis_route_lanes() -> torch.Tensor:
    route_lanes = torch.zeros((1, 2, 8))
    route_lanes[0, :, 0] = torch.tensor([0.0, 3.0])
    route_lanes[0, :, 2] = 1.0
    return route_lanes


def test_centerline_heading_frame_offset_uses_left_normal_sign():
    """+x heading: y=+1 is n=+1 (left), y=-0.5 is n=-0.5 (right)."""
    prediction = torch.tensor(
        [[[0.5, 1.0, 1.0, 0.0], [1.5, -0.5, 1.0, 0.0]]],
    )
    evaluation = evaluate_centerline_with_details(
        prediction,
        {"route_lanes": _x_axis_route_lanes()},
        {"horizon_seconds": 0.2, "match_threshold_m": 1.0},
    )
    torch.testing.assert_close(
        evaluation.details["centerline"]["lateral_offset_m"],
        torch.tensor([[1.0, -0.5]]),
    )


def test_centerline_lateral_in_band_rate_uses_abs_offset_and_threshold():
    """lateral_in_band_rate is the fraction of steps with |n| <= threshold."""
    prediction = torch.tensor(
        [[[0.5, 0.5, 1.0, 0.0], [1.5, 1.0, 1.0, 0.0], [2.5, 1.5, 1.0, 0.0]]],
    )
    values = evaluate_centerline(
        prediction,
        {"route_lanes": _x_axis_route_lanes()},
        {"horizon_seconds": 0.3, "match_threshold_m": 1.0},
    )

    torch.testing.assert_close(values["lateral_in_band_rate"], torch.tensor([200.0 / 3.0]))


def test_centerline_details_include_polylines_and_offsets():
    """Details expose prediction, nearest feet, route polyline, and signed n."""
    prediction = torch.tensor([[[0.5, 1.0, 1.0, 0.0], [1.5, 1.0, 1.0, 0.0]]])
    route_lanes = torch.zeros((1, 1, 4, 8))
    route_lanes[0, 0, :, 0] = torch.tensor([0.0, 1.0, 2.0, 0.0])
    route_lanes[0, 0, :3, 2] = 1.0
    evaluation = evaluate_centerline_with_details(
        prediction,
        {"route_lanes": route_lanes},
        {"horizon_seconds": 0.2, "match_threshold_m": 1.0},
    )
    details = evaluation.details["centerline"]

    assert details["prediction_xy"].shape == (1, 2, 2)
    assert details["closest_centerline_xy"].shape == (1, 2, 2)
    assert details["lateral_offset_m"].shape == (1, 2)
    torch.testing.assert_close(details["prediction_xy"][0], prediction[0, :, :2])
    torch.testing.assert_close(
        details["centerline_xy"][0],
        torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
    )
    assert not torch.all(details["centerline_xy"][0][-1] == 0)
    torch.testing.assert_close(details["match_threshold_m"], torch.tensor([1.0]))


def test_detail_value_to_json_serializes_scalars_and_polylines():
    """0-d tensors become numbers/bools; polylines become nested lists."""
    assert detail_value_to_json(torch.tensor([1.5]), 0) == 1.5
    assert detail_value_to_json(torch.tensor([True]), 0) is True
    polyline = torch.tensor([[[0.0, 0.0], [1.0, 0.5]]])
    assert detail_value_to_json(polyline, 0) == [[0.0, 0.0], [1.0, 0.5]]
    already_list = [[[2.0, 3.0], [4.0, 5.0]]]
    assert detail_value_to_json(already_list, 0) == [[2.0, 3.0], [4.0, 5.0]]
    per_sample_polyline = [torch.tensor([[0.0, 0.0], [1.0, 0.0]])]
    assert detail_value_to_json(per_sample_polyline, 0) == [[0.0, 0.0], [1.0, 0.0]]
