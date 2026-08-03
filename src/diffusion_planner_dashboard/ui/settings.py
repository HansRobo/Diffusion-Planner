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
        st.session_state["configured_parquet_path"] = candidate.strip()
    return st.session_state.get("configured_parquet_path") or None


def render_frame_selector(index: FrameIndex) -> FrameIndexRow:
    """Render bag and frame selectors and return the selected index row."""
    st.sidebar.subheader("Frame")
    bag_options = ("All bags", *index.bags)
    selected_bag_label = st.sidebar.selectbox("Bag", bag_options)
    selected_bag = None if selected_bag_label == "All bags" else selected_bag_label
    indices = index.indices_for_bag(selected_bag)
    position = st.sidebar.number_input(
        "Frame position",
        min_value=0,
        max_value=len(indices) - 1,
        value=0,
        step=1,
        key=f"frame-position::{index.path}::{selected_bag_label}",
    )
    return index.row(int(indices[int(position)]))


def render_vehicle_parameters() -> VehicleParameters:
    """Render editable vehicle dimensions."""
    with st.sidebar.expander("Vehicle parameters"):
        wheel_base = st.number_input("Wheel base [m]", 0.1, 20.0, 2.75, 0.01)
        wheel_tread = st.number_input("Wheel tread [m]", 0.1, 10.0, 1.59, 0.01)
        front_overhang = st.number_input("Front overhang [m]", 0.0, 10.0, 0.8, 0.01)
        rear_overhang = st.number_input("Rear overhang [m]", 0.0, 10.0, 1.1, 0.01)
        left_overhang = st.number_input("Left overhang [m]", 0.0, 5.0, 0.13, 0.01)
        right_overhang = st.number_input("Right overhang [m]", 0.0, 5.0, 0.13, 0.01)
    return VehicleParameters(
        wheel_base_m=wheel_base,
        wheel_tread_m=wheel_tread,
        front_overhang_m=front_overhang,
        rear_overhang_m=rear_overhang,
        left_overhang_m=left_overhang,
        right_overhang_m=right_overhang,
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
