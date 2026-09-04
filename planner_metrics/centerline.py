"""Route-centerline trajectory metrics usable across all evaluation scenarios."""

from __future__ import annotations

import torch

from planner_metrics.evaluation import MetricEvaluation
from planner_metrics.geometry import (
    _point_to_segments_error_components,
    _point_to_segments_min_dist,
)

_PREDICTION_TIMESTEP_SECONDS = 0.1
_CENTERLINE_SEGMENT_MIN_LENGTH = 1e-6
DEFAULT_MATCH_THRESHOLD_M = 0.5


def _centerline_segments(lanes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if lanes.ndim != 3 or lanes.shape[-1] < 4:
        raise ValueError(f"lanes must have shape (S, P, D>=4), got {tuple(lanes.shape)}")
    centerlines = lanes[..., :2]
    valid_points = lanes[..., :4].abs().sum(dim=-1) > 1e-6
    valid_segments = valid_points[:, :-1] & valid_points[:, 1:]
    segment_lengths = (centerlines[:, 1:] - centerlines[:, :-1]).norm(dim=-1)
    valid_segments &= segment_lengths > _CENTERLINE_SEGMENT_MIN_LENGTH
    if not valid_segments.any():
        raise ValueError("centerline metric found no valid route-centerline segments")
    return (
        centerlines[:, :-1][valid_segments],
        centerlines[:, 1:][valid_segments],
    )


def _centerline_polyline(lanes: torch.Tensor) -> torch.Tensor:
    """Return valid route-centerline vertices, dropping padded zeros."""
    if lanes.ndim != 3 or lanes.shape[-1] < 4:
        raise ValueError(f"lanes must have shape (S, P, D>=4), got {tuple(lanes.shape)}")
    centerlines = lanes[..., :2]
    valid_points = lanes[..., :4].abs().sum(dim=-1) > 1e-6
    return centerlines[valid_points]


def _batched_route_lanes(data: dict[str, torch.Tensor], batch_size: int) -> torch.Tensor:
    lanes = data.get("route_lanes", data.get("lanes"))
    if lanes is None:
        raise ValueError("centerline metric requires route_lanes or lanes in data")
    if lanes.ndim == 5:
        if lanes.shape[1] != 1:
            raise ValueError(
                f"expected singleton route_lanes context axis, got {tuple(lanes.shape)}"
            )
        lanes = lanes[:, 0]
    if lanes.ndim == 3:
        lanes = lanes.unsqueeze(0)
    if lanes.ndim != 4 or lanes.shape[0] not in (1, batch_size):
        raise ValueError(
            "lanes must have shape (S,P,D), (1,S,P,D), or (N,S,P,D); "
            f"got {tuple(lanes.shape)} for N={batch_size}"
        )
    return lanes


def _prediction_heading_units(ego_trajs: torch.Tensor, horizon_steps: int) -> torch.Tensor:
    """Return unit heading ``(cos, sin)`` at each selected prediction step.

    Uses stored ``(cos, sin)`` when present; otherwise finite differences of xy.
    Left-handed normal is ``(-sin, cos)``.
    """
    selected = ego_trajs[:, :horizon_steps]
    if selected.shape[-1] >= 4:
        heading = selected[..., 2:4]
        return heading / heading.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    xy = selected[..., :2]
    delta = torch.zeros_like(xy)
    if horizon_steps >= 2:
        forward = xy[:, 1:] - xy[:, :-1]
        delta[:, :-1] = forward
        delta[:, -1] = forward[:, -1]
    else:
        delta[..., 0] = 1.0
    return delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def _horizon_steps(ego_trajs: torch.Tensor, horizon_steps: int | None) -> int:
    if ego_trajs.ndim != 3 or ego_trajs.shape[-1] < 2:
        raise ValueError(f"ego_trajs must have shape (N, T, D>=2), got {tuple(ego_trajs.shape)}")
    if horizon_steps is None:
        horizon_steps = ego_trajs.shape[1]
    if not 1 <= horizon_steps <= ego_trajs.shape[1]:
        raise ValueError(f"horizon_steps must be in [1, {ego_trajs.shape[1]}]")
    return horizon_steps


@torch.no_grad()
def compute_centerline_distance_batch(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    horizon_steps: int | None = None,
) -> torch.Tensor:
    """Return nearest centerline distance with shape ``(N, T_selected)``.

    ``data`` may contain one shared lane tensor ``(S,P,D)``, a singleton scene
    dimension, or one lane tensor per trajectory. Distance evaluation delegates
    to the chunked geometry primitive to remain safe for large maps/batches.
    """
    horizon_steps = _horizon_steps(ego_trajs, horizon_steps)
    lanes = _batched_route_lanes(data, ego_trajs.shape[0])

    distances = []
    for index in range(ego_trajs.shape[0]):
        scene_lanes = lanes[0 if lanes.shape[0] == 1 else index]
        seg_p1, seg_p2 = _centerline_segments(scene_lanes)
        points = ego_trajs[index, :horizon_steps, :2]
        distances.append(_point_to_segments_min_dist(points, seg_p1.to(points), seg_p2.to(points)))
    return torch.stack(distances, dim=0)


@torch.no_grad()
def _compute_centerline_geometry_batch(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    horizon_steps: int | None = None,
) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    """Return centerline-axis errors and prediction-heading-frame offsets."""
    horizon_steps = _horizon_steps(ego_trajs, horizon_steps)
    lanes = _batched_route_lanes(data, ego_trajs.shape[0])
    prediction_xy = ego_trajs[:, :horizon_steps, :2]

    lateral_errors = []
    longitudinal_errors = []
    closest_points = []
    polylines: list[torch.Tensor] = []
    for index in range(ego_trajs.shape[0]):
        scene_lanes = lanes[0 if lanes.shape[0] == 1 else index]
        seg_p1, seg_p2 = _centerline_segments(scene_lanes)
        points = prediction_xy[index]
        lateral, longitudinal, closest = _point_to_segments_error_components(
            points, seg_p1.to(points), seg_p2.to(points)
        )
        lateral_errors.append(lateral)
        longitudinal_errors.append(longitudinal)
        closest_points.append(closest)
        polylines.append(
            _centerline_polyline(scene_lanes).to(device=points.device, dtype=points.dtype)
        )

    closest_xy = torch.stack(closest_points, dim=0)
    heading = _prediction_heading_units(ego_trajs, horizon_steps)
    left_normal = torch.stack([-heading[..., 1], heading[..., 0]], dim=-1)
    # n is the prediction's signed offset from the nearest centerline foot in
    # the prediction-heading Frenet frame (left positive, right negative).
    lateral_offset = ((prediction_xy - closest_xy) * left_normal).sum(dim=-1)
    return {
        "lateral_error_m": torch.stack(lateral_errors, dim=0),
        "longitudinal_error_m": torch.stack(longitudinal_errors, dim=0),
        "lateral_offset_m": lateral_offset,
        "closest_centerline_xy": closest_xy,
        "prediction_xy": prediction_xy,
        "centerline_xy": polylines,
    }


@torch.no_grad()
def compute_centerline_error_components_batch(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    horizon_steps: int | None = None,
) -> dict[str, torch.Tensor]:
    """Return lateral and beyond-segment longitudinal errors per timestep.

    Lateral error is measured against the selected centerline segment's
    supporting line.  Thus longitudinal overshoot beyond a segment endpoint is
    reported separately instead of being folded into lateral error.
    """
    geometry = _compute_centerline_geometry_batch(ego_trajs, data, horizon_steps)
    return {
        "lateral_error_m": geometry["lateral_error_m"],
        "longitudinal_error_m": geometry["longitudinal_error_m"],
    }


@torch.no_grad()
def compute_centerline_average_lateral_error_batch(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    horizon_steps: int | None = None,
) -> torch.Tensor:
    """Return average lateral error for each trajectory, shape ``(N,)``."""
    return compute_centerline_error_components_batch(ego_trajs, data, horizon_steps)[
        "lateral_error_m"
    ].mean(dim=1)


@torch.no_grad()
def compute_centerline_final_lateral_error_batch(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    horizon_steps: int | None = None,
) -> torch.Tensor:
    """Return final lateral error for each trajectory, shape ``(N,)``."""
    return compute_centerline_error_components_batch(ego_trajs, data, horizon_steps)[
        "lateral_error_m"
    ][:, -1]


def _centerline_horizon_steps(ego_trajs: torch.Tensor, parameters: dict) -> tuple[int, float, float]:
    horizon_seconds = float(parameters.get("horizon_seconds", 8.0))
    if horizon_seconds <= 0:
        raise ValueError("centerline horizon_seconds must be positive")
    match_threshold_m = float(parameters.get("match_threshold_m", DEFAULT_MATCH_THRESHOLD_M))
    if match_threshold_m < 0:
        raise ValueError("centerline match_threshold_m must be non-negative")
    steps = min(int(round(horizon_seconds / _PREDICTION_TIMESTEP_SECONDS)), ego_trajs.shape[1])
    if steps < 1:
        raise ValueError("centerline horizon selects zero prediction steps")
    return steps, horizon_seconds, match_threshold_m


def _centerline_scores(
    lateral_error: torch.Tensor,
    lateral_offset: torch.Tensor,
    match_threshold_m: float,
) -> dict[str, torch.Tensor]:
    matched = (lateral_offset.abs() <= match_threshold_m).to(dtype=lateral_offset.dtype)
    return {
        "average_lateral_error_m": lateral_error.mean(dim=1),
        "final_lateral_error_m": lateral_error[:, -1],
        "lateral_in_band_rate": matched.mean(dim=1) * 100.0,
    }


def evaluate_centerline(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    parameters: dict,
) -> dict[str, torch.Tensor]:
    """Return centerline-axis lateral error and heading-frame match scores."""
    return evaluate_centerline_with_details(ego_trajs, data, parameters).scores


def evaluate_centerline_with_details(
    ego_trajs: torch.Tensor,
    data: dict[str, torch.Tensor],
    parameters: dict,
) -> MetricEvaluation:
    """Evaluate centerline metrics using the common open-loop result format.

    ``lateral_error_m`` remains the unsigned distance to the selected
    centerline segment's supporting line.  ``lateral_offset_m`` is the signed
    n-coordinate of each prediction point in a Frenet frame whose s-axis is
    the prediction heading (left positive).  Score tensors are aggregated into
    the summary; polyline / timeseries fields stay in ``details``.
    """
    steps, horizon_seconds, match_threshold_m = _centerline_horizon_steps(ego_trajs, parameters)
    geometry = _compute_centerline_geometry_batch(ego_trajs, data, steps)
    lateral_error = geometry["lateral_error_m"]
    lateral_offset = geometry["lateral_offset_m"]
    assert isinstance(lateral_error, torch.Tensor)
    assert isinstance(lateral_offset, torch.Tensor)
    sample_count = lateral_offset.shape[0]
    filled = {
        "horizon_seconds": torch.full(
            (sample_count,),
            horizon_seconds,
            dtype=lateral_offset.dtype,
            device=lateral_offset.device,
        ),
        "match_threshold_m": torch.full(
            (sample_count,),
            match_threshold_m,
            dtype=lateral_offset.dtype,
            device=lateral_offset.device,
        ),
    }
    return MetricEvaluation(
        scores=_centerline_scores(lateral_error, lateral_offset, match_threshold_m),
        details={
            "centerline": {
                **filled,
                "prediction_xy": geometry["prediction_xy"],
                "closest_centerline_xy": geometry["closest_centerline_xy"],
                "centerline_xy": geometry["centerline_xy"],
                "lateral_offset_m": lateral_offset,
            }
        },
    )


__all__ = [
    "compute_centerline_average_lateral_error_batch",
    "compute_centerline_distance_batch",
    "compute_centerline_error_components_batch",
    "compute_centerline_final_lateral_error_batch",
    "evaluate_centerline",
    "evaluate_centerline_with_details",
    "DEFAULT_MATCH_THRESHOLD_M",
]
