"""Plotly trace factories for diffusion planner frame layers."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

from .frame import FrameData
from .schema import LaneIndex, NeighborIndex, TrafficLightIndex
from .style import FramePlotOptions, VisualizerStyle


def _joined_lines(
    lines: Iterable[NDArray[np.generic]],
) -> tuple[list[float | None], list[float | None]]:
    x: list[float | None] = []
    y: list[float | None] = []
    for line in lines:
        if len(line) == 0:
            continue
        x.extend(float(value) for value in line[:, 0])
        y.extend(float(value) for value in line[:, 1])
        x.append(None)
        y.append(None)
    return x, y


def _line_trace(
    lines: Iterable[NDArray[np.generic]],
    *,
    name: str,
    color: str,
    width: float,
    legendgroup: str,
    dash: str | None = None,
    showlegend: bool = True,
) -> go.Scattergl | None:
    x, y = _joined_lines(lines)
    if not x:
        return None
    return go.Scattergl(
        x=x,
        y=y,
        mode="lines",
        name=name,
        legendgroup=legendgroup,
        showlegend=showlegend,
        line={"color": color, "width": width, "dash": dash},
        hoverinfo="skip",
    )


def _lane_lines(lanes: NDArray[np.generic], kind: str) -> list[NDArray[np.generic]]:
    lines: list[NDArray[np.generic]] = []
    for lane in lanes[FrameData.valid_rows(lanes)]:
        center = lane[:, [LaneIndex.X, LaneIndex.Y]]
        if kind == "center":
            lines.append(center)
        elif kind == "left":
            lines.append(center + lane[:, [LaneIndex.LEFT_OFFSET_X, LaneIndex.LEFT_OFFSET_Y]])
        elif kind == "right":
            lines.append(center + lane[:, [LaneIndex.RIGHT_OFFSET_X, LaneIndex.RIGHT_OFFSET_Y]])
        else:
            raise ValueError(f"Unknown lane line kind: {kind}")
    return lines


def create_lane_traces(
    frame: FrameData,
    style: VisualizerStyle,
    options: FramePlotOptions,
) -> list[go.BaseTraceType]:
    """Create local-lane and route-lane traces."""
    traces: list[go.BaseTraceType] = []
    for key, label, color, boundary_color, width, group in (
        (
            "lanes",
            "Lanes",
            style.lane_color,
            style.lane_boundary_color,
            style.lane_width,
            "lanes",
        ),
        (
            "route_lanes",
            "Route",
            style.route_color,
            style.route_boundary_color,
            style.route_width,
            "route",
        ),
    ):
        lanes = frame[key]
        center = _line_trace(
            _lane_lines(lanes, "center"),
            name=label,
            color=color,
            width=width,
            legendgroup=group,
        )
        if center is not None:
            traces.append(center)
        if options.show_lane_boundaries:
            for kind in ("left", "right"):
                boundary = _line_trace(
                    _lane_lines(lanes, kind),
                    name=f"{label} boundaries",
                    color=boundary_color,
                    width=max(1.0, width * 0.6),
                    legendgroup=group,
                    showlegend=kind == "left",
                )
                if boundary is not None:
                    traces.append(boundary)

        if options.show_speed_limits:
            traces.extend(_create_speed_limit_trace(frame, key, lanes, group))
        if options.show_traffic_lights:
            traces.extend(_create_traffic_light_traces(frame, key, lanes, group))
    return traces


def _create_speed_limit_trace(
    frame: FrameData,
    lane_key: str,
    lanes: NDArray[np.generic],
    legendgroup: str,
) -> list[go.BaseTraceType]:
    speed_key = f"{lane_key}_speed_limit"
    speeds = frame.get(speed_key)
    if speeds is None:
        return []
    valid_indices = np.flatnonzero(FrameData.valid_rows(lanes))
    x: list[float] = []
    y: list[float] = []
    text: list[str] = []
    for index in valid_indices:
        speed = float(np.ravel(speeds[index])[0])
        if speed <= 0:
            continue
        midpoint = lanes[index, len(lanes[index]) // 2, :2]
        x.append(float(midpoint[0]))
        y.append(float(midpoint[1]))
        text.append(f"{speed:.1f} m/s")
    if not x:
        return []
    return [
        go.Scattergl(
            x=x,
            y=y,
            text=text,
            mode="markers+text",
            textposition="top center",
            marker={"size": 5, "color": "#374151"},
            name="Speed limits",
            legendgroup=legendgroup,
            showlegend=False,
            hovertemplate="%{text}<extra></extra>",
        )
    ]


def _create_traffic_light_traces(
    frame: FrameData,
    lane_key: str,
    lanes: NDArray[np.generic],
    legendgroup: str,
) -> list[go.BaseTraceType]:
    traffic_key = "lane_traffic_light_past" if lane_key == "lanes" else "route_traffic_light_past"
    traffic = frame.get(traffic_key)
    if traffic is None:
        return []

    colors = ("#22c55e", "#f59e0b", "#ef4444", "#6b7280")
    names = ("Green light", "Amber light", "Red light", "Unknown light")
    valid_lanes = FrameData.valid_rows(lanes)
    traces: list[go.BaseTraceType] = []
    for state, (color, name) in enumerate(zip(colors, names, strict=True)):
        x: list[float] = []
        y: list[float] = []
        symbols: list[str] = []
        for index in np.flatnonzero(valid_lanes):
            latest = traffic[index, -1]
            if latest.shape[-1] <= TrafficLightIndex.UNKNOWN:
                continue
            color_state = latest[: TrafficLightIndex.WHITE_OR_NONE + 1]
            if not np.any(color_state) or int(np.argmax(color_state)) != state:
                continue
            endpoint = lanes[index, -1, :2]
            x.append(float(endpoint[0]))
            y.append(float(endpoint[1]))
            is_arrow = (
                latest.shape[-1] > TrafficLightIndex.IS_ARROW
                and latest[TrafficLightIndex.IS_ARROW] > 0.5
            )
            symbols.append("triangle-up" if is_arrow else "circle")
        if x:
            traces.append(
                go.Scattergl(
                    x=x,
                    y=y,
                    mode="markers",
                    marker={"size": 8, "color": color, "symbol": symbols},
                    name=name,
                    legendgroup=f"{legendgroup}-traffic",
                    showlegend=lane_key == "route_lanes",
                    hovertemplate=f"{name}<extra></extra>",
                )
            )
    return traces


def create_map_element_traces(frame: FrameData, style: VisualizerStyle) -> list[go.BaseTraceType]:
    """Create polygon and line-string traces."""
    traces: list[go.BaseTraceType] = []
    polygons = frame["polygons"]
    polygon_lines = []
    for polygon in polygons[FrameData.valid_rows(polygons)]:
        points = polygon[:, :2]
        polygon_lines.append(np.concatenate((points, points[:1]), axis=0))
    x, y = _joined_lines(polygon_lines)
    if x:
        traces.append(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                fill="toself",
                fillcolor=style.polygon_color,
                line={"color": style.polygon_color, "width": 1},
                name="Polygons",
                legendgroup="map-elements",
                hoverinfo="skip",
            )
        )

    line_strings = frame["line_strings"]
    trace = _line_trace(
        (line[:, :2] for line in line_strings[FrameData.valid_rows(line_strings)]),
        name="Line strings",
        color=style.line_string_color,
        width=1.5,
        legendgroup="map-elements",
        dash="dash",
    )
    if trace is not None:
        traces.append(trace)
    return traces


def _pose_lines(array: NDArray[np.generic]) -> list[NDArray[np.generic]]:
    lines: list[NDArray[np.generic]] = []
    if array.ndim == 2:
        valid = FrameData.valid_steps(array)
        if np.any(valid):
            lines.append(array[valid, :2])
        return lines
    for poses in array:
        valid = FrameData.valid_steps(poses)
        if np.any(valid):
            lines.append(poses[valid, :2])
    return lines


def create_agent_traces(
    frame: FrameData,
    style: VisualizerStyle,
    options: FramePlotOptions,
) -> list[go.BaseTraceType]:
    """Create ego and neighboring-agent history/future traces."""
    traces: list[go.BaseTraceType] = []
    if options.show_agent_history:
        ego_past = _line_trace(
            _pose_lines(frame["ego_agent_past"]),
            name="Ego past",
            color=style.ego_color,
            width=style.history_width,
            legendgroup="ego",
        )
        if ego_past is not None:
            traces.append(ego_past)

        neighbors = frame["neighbor_agents_past"]
        neighbor_past = _line_trace(
            _pose_lines(neighbors[FrameData.valid_rows(neighbors)]),
            name="Neighbors past",
            color=style.neighbor_color,
            width=style.history_width,
            legendgroup="neighbors",
        )
        if neighbor_past is not None:
            traces.append(neighbor_past)
        traces.extend(_create_neighbor_markers(neighbors, style))

    if options.show_agent_future:
        ego_future = frame.get("ego_agent_future")
        if ego_future is not None:
            trace = _line_trace(
                _pose_lines(ego_future),
                name="Ego future",
                color=style.ego_future_color,
                width=style.future_width,
                legendgroup="ego",
                dash="dot",
            )
            if trace is not None:
                traces.append(trace)
        neighbor_future = frame.get("neighbor_agents_future")
        if neighbor_future is not None:
            trace = _line_trace(
                _pose_lines(neighbor_future),
                name="Neighbors future",
                color=style.neighbor_future_color,
                width=style.future_width,
                legendgroup="neighbors",
                dash="dot",
            )
            if trace is not None:
                traces.append(trace)
    return traces


def _create_neighbor_markers(
    neighbors: NDArray[np.generic], style: VisualizerStyle
) -> list[go.BaseTraceType]:
    valid_neighbors = neighbors[FrameData.valid_rows(neighbors)]
    if len(valid_neighbors) == 0:
        return []
    current = valid_neighbors[:, -1]
    labels = np.argmax(current[:, NeighborIndex.IS_VEHICLE : NeighborIndex.IS_BICYCLE + 1], axis=1)
    label_names = np.array(["vehicle", "pedestrian", "bicycle"])
    customdata = np.column_stack(
        (
            label_names[labels],
            current[:, NeighborIndex.LENGTH],
            current[:, NeighborIndex.WIDTH],
        )
    )
    return [
        go.Scattergl(
            x=current[:, NeighborIndex.X],
            y=current[:, NeighborIndex.Y],
            mode="markers",
            marker={"size": 7, "color": style.neighbor_color},
            customdata=customdata,
            name="Neighbors current",
            legendgroup="neighbors",
            showlegend=False,
            hovertemplate=(
                "%{customdata[0]}<br>x=%{x:.2f} m<br>y=%{y:.2f} m"
                "<br>length=%{customdata[1]:.2f} m<br>width=%{customdata[2]:.2f} m<extra></extra>"
            ),
        )
    ]


def create_annotation_traces(
    frame: FrameData,
    style: VisualizerStyle,
    options: FramePlotOptions,
) -> list[go.BaseTraceType]:
    """Create ego footprint and goal-pose traces."""
    traces: list[go.BaseTraceType] = []
    ego_past = frame["ego_agent_past"]
    current = ego_past[-1]
    if options.show_ego_shape:
        wheel_base, length, width = (float(value) for value in frame["ego_shape"][:3])
        corners = np.array(
            [
                [-length / 2, -width / 2],
                [length / 2, -width / 2],
                [length / 2, width / 2],
                [-length / 2, width / 2],
                [-length / 2, -width / 2],
            ]
        )
        cos_yaw, sin_yaw = float(current[2]), float(current[3])
        rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        corners = corners @ rotation.T + current[:2]
        traces.append(
            go.Scatter(
                x=corners[:, 0],
                y=corners[:, 1],
                mode="lines",
                fill="toself",
                fillcolor="rgba(220, 38, 38, 0.25)",
                line={"color": style.ego_color, "width": 2},
                name="Ego footprint",
                legendgroup="ego",
                customdata=np.full((len(corners), 1), wheel_base),
                hovertemplate="wheelbase=%{customdata[0]:.2f} m<extra></extra>",
            )
        )

    if options.show_goal:
        goal = frame["goal_pose"]
        heading_length = 3.0
        x = float(goal[0])
        y = float(goal[1])
        dx = heading_length * float(goal[2])
        dy = heading_length * float(goal[3])
        traces.append(
            go.Scattergl(
                x=[x, x + dx],
                y=[y, y + dy],
                mode="lines+markers",
                marker={"size": [10, 5], "color": style.goal_color},
                line={"color": style.goal_color, "width": 3},
                name="Goal pose",
                legendgroup="goal",
                hovertemplate="x=%{x:.2f} m<br>y=%{y:.2f} m<extra>Goal</extra>",
            )
        )
    return traces


def create_frame_traces(
    frame: FrameData,
    style: VisualizerStyle,
    options: FramePlotOptions,
) -> list[go.BaseTraceType]:
    """Create all enabled traces for one frame."""
    return [
        *create_map_element_traces(frame, style),
        *create_lane_traces(frame, style, options),
        *create_agent_traces(frame, style, options),
        *create_annotation_traces(frame, style, options),
    ]
