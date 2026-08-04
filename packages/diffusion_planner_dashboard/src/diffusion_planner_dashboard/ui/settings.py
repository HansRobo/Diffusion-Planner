"""Sidebar controls for dashboard data and visualization settings."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from diffusion_planner.visualizer import FramePlotOptions
from diffusion_planner_dashboard.services import FrameIndex, FrameIndexRow, VehicleParameters


def render_parquet_settings() -> str | None:
    """Render the data-source form and return the applied Parquet path."""
    st.sidebar.subheader("Data source")
    with st.sidebar.form("parquet-settings"):
        candidate = st.text_input(
            "Frame index Parquet",
            value=st.session_state.get("configured_parquet_path", ""),
            placeholder="/path/to/frame_index.parquet",
        )
        applied = st.form_submit_button("Apply", use_container_width=True)
    if applied:
        assert candidate is not None
        st.session_state["configured_parquet_path"] = candidate.strip()
    return st.session_state.get("configured_parquet_path") or None


def render_frame_selector(index: FrameIndex) -> FrameIndexRow:
    """Render bag and frame selectors and return the selected index row."""
    st.sidebar.subheader("Frame")
    bag_options = ("All bags", *index.bags)
    selected_bag_label = st.sidebar.selectbox("Bag", bag_options)
    selected_bag = None if selected_bag_label == "All bags" else selected_bag_label
    indices = index.indices_for_bag(selected_bag)
    position = st.sidebar.slider(
        "Frame position",
        min_value=0,
        max_value=len(indices) - 1,
        value=0,
        step=1,
        key=f"frame-position::{index.path}::{selected_bag_label}",
    )
    row = index.row(int(indices[int(position)]))
    st.sidebar.caption(f"Parquet row: {row.index:,} · Time: {row.frame_time_ns} ns")
    return row


def render_vehicle_parameters() -> VehicleParameters:
    """Render editable vehicle dimensions."""
    with st.sidebar.expander("Vehicle parameters"):
        base_link_to_front = st.number_input(
            "Base link to front [m]", 0.1, 20.0, VehicleParameters.base_link_to_front, 0.01
        )
        vehicle_length = st.number_input(
            "Vehicle length [m]", 0.1, 30.0, VehicleParameters.vehicle_length, 0.01
        )
        vehicle_width = st.number_input(
            "Vehicle width [m]", 0.1, 10.0, VehicleParameters.vehicle_width, 0.01
        )
    return VehicleParameters(
        base_link_to_front=base_link_to_front,
        vehicle_length=vehicle_length,
        vehicle_width=vehicle_width,
    )


def render_plot_options() -> FramePlotOptions:
    """Render visualization layer switches."""
    with st.sidebar.expander("Layers", expanded=True):
        lane_boundaries = st.checkbox("Lane boundaries", value=True)
        agent_history = st.checkbox("Agent history", value=True)
        agent_future = st.checkbox("Agent future labels", value=True)
        ego_shape = st.checkbox("Ego footprint", value=True)
        goal = st.checkbox("Goal pose", value=True)
        traffic_lights = st.checkbox("Traffic lights", value=True)
        speed_limits = st.checkbox("Speed limits", value=False)
    return FramePlotOptions(
        show_lane_boundaries=lane_boundaries,
        show_agent_history=agent_history,
        show_agent_future=agent_future,
        show_ego_shape=ego_shape,
        show_goal=goal,
        show_traffic_lights=traffic_lights,
        show_speed_limits=speed_limits,
    )


def missing_frame_sources(row: FrameIndexRow) -> list[str]:
    """Return human-readable errors for missing bag or map sources."""
    errors = []
    if not (Path(row.bag_path) / "metadata.yaml").is_file():
        errors.append(f"Rosbag is unavailable: {row.bag_path}")
    if not Path(row.map_path).is_file():
        errors.append(f"Map is unavailable: {row.map_path}")
    return errors
