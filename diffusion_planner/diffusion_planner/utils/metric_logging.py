"""Small, stable metric allowlists for training dashboards.

The validation implementation intentionally keeps raw aliases and availability
metadata because they are useful for compatibility and checkpoint guards.  They
are not all useful dashboard series, however: PDMS/EPDMS exposes both long and
short names for the same subscore and emits one availability/coverage pair for
each one.  This module selects one canonical view without changing the computed
metrics.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch


# Keep the names used by the RL source-policy guards and by existing dashboards.
# The values in the other aliases are mathematically identical raw subscores.
EPDMS_DASHBOARD_CANDIDATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ego_progress", ("ego_progress",)),
    ("comfort", ("comfort", "history_comfort")),
    ("lane_keeping", ("lane_keeping",)),
    ("no_collision", ("no_collision", "no_at_fault_collision")),
    ("ttc", ("ttc", "time_to_collision_within_bound")),
    ("dac", ("dac", "drivable_area_compliance")),
    ("ddc", ("ddc", "driving_direction_compliance")),
    ("total", ("total", "proxy_epdms")),
)

# These are the useful directional diagnostics.  simple_l2, position_l2, and
# cosine_similarity are derived from the same position/heading residuals and
# make the dashboard noisy without adding a new training signal.
VALID_LOSS_DASHBOARD_KEYS: tuple[str, ...] = (
    "ego_position_lat_loss",
    "ego_position_lon_loss",
    "ego_heading_l2_loss",
)


def _finite_scalar(value: object) -> float | None:
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def select_epdms_dashboard_metrics(metrics: Mapping[str, object]) -> dict[str, float]:
    """Return one finite value per canonical EPDMS dashboard subscore.

    Availability flags, coverage columns, strict synthetic intermediates, and
    long-name aliases are deliberately omitted.  The raw aggregate remains
    available to callers for validation and checkpoint decisions.
    """

    result: dict[str, float] = {}
    for canonical, candidates in EPDMS_DASHBOARD_CANDIDATES:
        for key in candidates:
            if key not in metrics:
                continue
            number = _finite_scalar(metrics[key])
            if number is not None:
                result[canonical] = number
                break
    return result


def select_valid_loss_dashboard_metrics(metrics: Mapping[str, object]) -> dict[str, float]:
    """Select non-derived ego validation loss components for the dashboard."""

    result: dict[str, float] = {}
    for key in VALID_LOSS_DASHBOARD_KEYS:
        if key not in metrics:
            continue
        number = _finite_scalar(metrics[key])
        if number is not None:
            result[key] = number
    return result
