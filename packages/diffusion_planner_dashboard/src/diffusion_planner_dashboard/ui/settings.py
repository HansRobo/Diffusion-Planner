"""Sidebar controls for dashboard data and visualization settings."""

from __future__ import annotations

import streamlit as st

from diffusion_planner.visualizer import FramePlotOptions
from diffusion_planner_dashboard.services import FrameIndex, FrameIndexRow


def render_data_source_settings() -> str | None:
    """Render H5/Parquet source settings and return the applied values."""
    st.sidebar.subheader("Data source")
    with st.sidebar.form("frame-source-settings"):
        candidate = st.text_input(
            "H5 or Parquet file",
            value=st.session_state.get("configured_frame_source_path", ""),
            placeholder="/path/to/frames.h5 or /path/to/train.parquet",
        )
        applied = st.form_submit_button("Apply", use_container_width=True)
    if applied:
        assert candidate is not None
        st.session_state["configured_frame_source_path"] = candidate.strip()
    source = st.session_state.get("configured_frame_source_path") or None
    return source


def render_frame_selector(index: FrameIndex) -> FrameIndexRow:
    """Render bag and frame selectors and return the selected index row."""
    st.sidebar.subheader("Frame")
    source_options = ("All H5 files", *index.sources)
    selected_source_label = st.sidebar.selectbox("H5 file", source_options)
    selected_source = (
        None if selected_source_label == "All H5 files" else selected_source_label
    )
    indices = index.indices_for_source(selected_source)
    position = st.sidebar.slider(
        "Frame position",
        min_value=0,
        max_value=len(indices) - 1,
        value=0,
        step=1,
        key=f"frame-position::{index.path}::{selected_source_label}",
    )
    row = index.row(int(indices[int(position)]))
    st.sidebar.caption(
        f"Source row: {row.index:,} · H5 frame: {row.frame_index:,} · Time: {row.frame_time_ns} ns"
    )
    return row


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
