"""Closed-loop validation metrics for scenario-group evaluation."""

from scenario_generation.metrics.centerline_point_distance import nearest_centerline_point_dist_m
from scenario_generation.metrics.group_report import aggregate_segment_rows, write_results_table
from scenario_generation.metrics.safety_clearance import score_safety_step
from scenario_generation.metrics.turn_indicator import (
    expected_turn_class,
    score_turn_indicator_step,
)

__all__ = [
    "nearest_centerline_point_dist_m",
    "score_safety_step",
    "score_turn_indicator_step",
    "expected_turn_class",
    "aggregate_segment_rows",
    "write_results_table",
]
