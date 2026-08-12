"""Frame-index metadata presentation components."""

from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from diffusion_planner_dashboard.services import FrameIndex, FrameIndexRow


def render_index_summary(index: FrameIndex) -> None:
    """Display high-level counts for a loaded frame index."""
    columns = st.columns(2)
    columns[0].metric("Frames", f"{len(index):,}")
    columns[1].metric("H5 files", f"{len(index.sources):,}")


def render_row_metadata(row: FrameIndexRow) -> None:
    """Display source and curation statistics for a selected frame."""
    timestamp = datetime.fromtimestamp(row.frame_time_ns / 1e9, tz=timezone.utc)
    st.caption(
        f"Row {row.index:,} · H5 frame {row.frame_index:,} · `{row.h5_path}` · frame time "
        f"`{row.frame_time_ns}` ns ({timestamp.isoformat()})"
    )
    if not row.stats:
        return
    columns = st.columns(len(row.stats))
    labels = {
        "ego_speed_mps": "Ego speed [m/s]",
        "ego_yaw_rate_rps": "Yaw rate [rad/s]",
        "turn_indicator": "Turn indicator",
        "num_objects": "Objects",
    }
    for column, (name, value) in zip(columns, row.stats.items(), strict=True):
        column.metric(labels.get(name, name), str(value))
