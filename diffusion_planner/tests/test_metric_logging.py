import torch
import pytest

from diffusion_planner.utils.metric_logging import (
    select_epdms_dashboard_metrics,
    select_valid_loss_dashboard_metrics,
)


def test_epdms_dashboard_keeps_one_canonical_name_per_subscore():
    metrics = {
        "history_comfort": torch.tensor(0.8),
        "comfort": torch.tensor(0.8),
        "drivable_area_compliance": 0.9,
        "dac": 0.9,
        "driving_direction_compliance": 0.95,
        "ddc": 0.95,
        "time_to_collision_within_bound": 0.7,
        "ttc": 0.7,
        "no_at_fault_collision": 1.0,
        "no_collision": 1.0,
        "ego_progress": 0.6,
        "lane_keeping": 0.75,
        "proxy_epdms": 0.5,
        "total": 0.5,
        "dac_available": 1.0,
        "dac_coverage": 1.0,
        "synthetic_epdms_raw": 0.0,
    }

    assert select_epdms_dashboard_metrics(metrics) == pytest.approx(
        {
            "ego_progress": 0.6,
            "comfort": 0.8,
            "lane_keeping": 0.75,
            "no_collision": 1.0,
            "ttc": 0.7,
            "dac": 0.9,
            "ddc": 0.95,
            "total": 0.5,
        }
    )


def test_epdms_dashboard_uses_long_name_when_alias_is_absent():
    metrics = {
        "history_comfort": 0.8,
        "drivable_area_compliance": 0.9,
        "driving_direction_compliance": 0.95,
        "time_to_collision_within_bound": 0.7,
        "no_at_fault_collision": 1.0,
        "ego_progress": 0.6,
        "lane_keeping": 0.75,
        "proxy_epdms": 0.5,
    }

    assert select_epdms_dashboard_metrics(metrics)["dac"] == 0.9
    assert select_epdms_dashboard_metrics(metrics)["total"] == 0.5


def test_valid_loss_dashboard_drops_derived_and_geometry_series():
    metrics = {
        "ego_simple_l2_loss": 1.0,
        "ego_position_l2_loss": 0.9,
        "ego_position_lat_loss": 0.2,
        "ego_position_lon_loss": 0.3,
        "ego_heading_l2_loss": 0.1,
        "ego_cosine_similarity_loss": 0.05,
        "ego_neighbor_margin_loss": 0.0,
        "ego_road_border_loss": 0.0,
    }

    assert select_valid_loss_dashboard_metrics(metrics) == {
        "ego_position_lat_loss": 0.2,
        "ego_position_lon_loss": 0.3,
        "ego_heading_l2_loss": 0.1,
    }
