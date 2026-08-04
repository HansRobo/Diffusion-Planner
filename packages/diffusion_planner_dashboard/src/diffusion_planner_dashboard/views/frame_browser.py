"""Frame-index browser view."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from diffusion_planner.visualizer import plot_frame
from diffusion_planner_dashboard.services import (
    FrameIndex,
    FrameIndexRow,
    FrameLoader,
    VehicleParameters,
    load_frame_index,
)
from diffusion_planner_dashboard.ui.metadata import render_index_summary, render_row_metadata
from diffusion_planner_dashboard.ui.settings import (
    missing_frame_sources,
    render_frame_selector,
    render_parquet_settings,
    render_plot_options,
    render_vehicle_parameters,
)
from diffusion_planner_dashboard.ui.tensor_inspector import render_tensor_inspector


@st.cache_data(show_spinner=False)
def _cached_index(path: str, modification_time_ns: int) -> FrameIndex:
    del modification_time_ns  # Included in the cache key to invalidate changed files.
    return load_frame_index(path)


@st.cache_resource
def _frame_loader() -> FrameLoader:
    return FrameLoader()


@st.cache_data(max_entries=64, show_spinner="Reading frame data from rosbag...")
def _cached_frame(
    row_index: int,
    bag_path: str,
    map_path: str,
    frame_time_ns: int,
    vehicle: VehicleParameters,
):
    row = FrameIndexRow(row_index, bag_path, map_path, frame_time_ns, {})
    return _frame_loader().load(row, vehicle)


def render_frame_browser() -> None:
    """Render the complete Parquet-backed frame browser."""
    st.title("Frame Browser")
    parquet_path_text = render_parquet_settings()
    if parquet_path_text is None:
        st.info("Configure a frame-index Parquet file from the sidebar.")
        return

    parquet_path = Path(parquet_path_text).expanduser()
    try:
        modification_time_ns = parquet_path.stat().st_mtime_ns
        index = _cached_index(str(parquet_path), modification_time_ns)
    except (OSError, ValueError) as error:
        st.error(str(error))
        return

    render_index_summary(index)
    row = render_frame_selector(index)
    render_row_metadata(row)

    source_errors = missing_frame_sources(row)
    if source_errors:
        st.error("\n\n".join(source_errors))
        return

    vehicle = render_vehicle_parameters()
    options = render_plot_options()
    try:
        frame_data = _cached_frame(
            row.index,
            row.bag_path,
            row.map_path,
            row.frame_time_ns,
            vehicle,
        )
    except (RuntimeError, ValueError) as error:
        st.error(str(error))
        return
    except Exception as error:  # Native rosbag/map errors vary by backend.
        st.exception(error)
        return

    if frame_data is None:
        st.warning("The selected index row could not be converted into frame data.")
        return

    figure = plot_frame(frame_data, options=options)
    chart_key = f"frame-plot::{index.path}::{row.index}::{row.frame_time_ns}"
    figure.update_layout(
        autosize=True,
        uirevision=chart_key,
    )
    st.plotly_chart(
        figure,
        width="stretch",
        height=900,
        key=chart_key,
        config={"responsive": True, "scrollZoom": True},
    )
    render_tensor_inspector(frame_data)
