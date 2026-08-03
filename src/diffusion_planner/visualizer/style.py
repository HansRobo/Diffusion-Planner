"""Configuration objects for Plotly frame visualization."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VisualizerStyle:
    """Colors and sizes used by the default visualizer."""

    background_color: str = "#ffffff"
    lane_color: str = "#9ca3af"
    lane_boundary_color: str = "#d1d5db"
    route_color: str = "#2563eb"
    route_boundary_color: str = "#93c5fd"
    polygon_color: str = "rgba(168, 85, 247, 0.18)"
    line_string_color: str = "#6b7280"
    ego_color: str = "#dc2626"
    ego_future_color: str = "#16a34a"
    neighbor_color: str = "#d97706"
    neighbor_future_color: str = "#65a30d"
    goal_color: str = "#7c3aed"
    lane_width: float = 1.2
    route_width: float = 3.0
    history_width: float = 2.0
    future_width: float = 2.0


@dataclass(frozen=True)
class FramePlotOptions:
    """Layer switches and layout options for a frame plot."""

    show_lane_boundaries: bool = True
    show_agent_history: bool = True
    show_agent_future: bool = True
    show_ego_shape: bool = True
    show_goal: bool = True
    show_traffic_lights: bool = True
    show_speed_limits: bool = False
    equal_aspect: bool = True
    title: str | None = None
