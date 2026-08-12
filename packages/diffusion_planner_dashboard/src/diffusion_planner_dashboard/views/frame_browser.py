"""Frame-index browser view."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from diffusion_planner.visualizer import plot_frame
from diffusion_planner_dashboard.services import (
    FrameIndex,
    FrameIndexRow,
    FrameLoader,
    load_frame_index,
)
from diffusion_planner_dashboard.ui.metadata import (
    render_index_summary,
    render_row_metadata,
)
from diffusion_planner_dashboard.ui.settings import (
    render_data_source_settings,
    render_frame_selector,
    render_plot_options,
)
from diffusion_planner_dashboard.ui.tensor_inspector import render_tensor_inspector


@st.cache_data(show_spinner=False)
def _cached_index(path: str, modification_time_ns: int) -> FrameIndex:
    del modification_time_ns  # Included in the cache key to invalidate changed files.
    return load_frame_index(path)


@st.cache_resource
def _frame_loader() -> FrameLoader:
    return FrameLoader()


# Cached on primitives so that the key stays stable and cheap to hash.
@st.cache_data(max_entries=64, show_spinner="Reading frame data from H5...")
def _cached_frame(
    h5_path: str,
    frame_index: int,
    frame_time_ns: int,
    modification_time_ns: int,
):
    del modification_time_ns
    row = FrameIndexRow(0, h5_path, frame_index, frame_time_ns, {})
    return _frame_loader().load(row)


def render_frame_browser() -> None:
    """Render the complete H5/Parquet-backed frame browser."""
    st.title("Frame Browser")
    source_path_text = render_data_source_settings()
    if source_path_text is None:
        st.info("Configure an H5 file or H5 frame-index Parquet from the sidebar.")
        return

    source_path = Path(source_path_text).expanduser()
    try:
        modification_time_ns = source_path.stat().st_mtime_ns
        index = _cached_index(str(source_path), modification_time_ns)
    except (OSError, ValueError) as error:
        st.error(str(error))
        return

    render_index_summary(index)
    row = render_frame_selector(index)
    render_row_metadata(row)

    options = render_plot_options()
    try:
        h5_modification_time_ns = Path(row.h5_path).stat().st_mtime_ns
        frame_data = _cached_frame(
            row.h5_path,
            row.frame_index,
            row.frame_time_ns,
            h5_modification_time_ns,
        )
    except (RuntimeError, ValueError) as error:
        st.error(str(error))
        return
    except Exception as error:
        st.exception(error)
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
