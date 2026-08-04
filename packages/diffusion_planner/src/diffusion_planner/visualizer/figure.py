"""High-level Plotly figure construction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import plotly.graph_objects as go

from .frame import FrameData
from .style import FramePlotOptions, VisualizerStyle
from .traces import create_frame_traces


def plot_frame(
    frame_data: FrameData | Mapping[str, Any],
    *,
    batch_index: int = 0,
    options: FramePlotOptions | None = None,
    style: VisualizerStyle | None = None,
) -> go.Figure:
    """Build an interactive 2D figure for one model-ready frame.

    This function does not display or write the figure. Callers may use
    ``Figure.show`` or ``Figure.write_html`` as appropriate.
    """
    frame = (
        frame_data
        if isinstance(frame_data, FrameData)
        else FrameData.from_mapping(frame_data, batch_index=batch_index)
    )
    resolved_options = options or FramePlotOptions()
    resolved_style = style or VisualizerStyle()
    figure = go.Figure(create_frame_traces(frame, resolved_style, resolved_options))
    figure.update_layout(
        title=resolved_options.title,
        template="plotly_white",
        paper_bgcolor=resolved_style.background_color,
        plot_bgcolor=resolved_style.background_color,
        hovermode="closest",
        legend={"groupclick": "togglegroup"},
        xaxis={"title": "x [m]", "range": [-30, 30], "zeroline": True},
        yaxis={"title": "y [m]", "range": [-30, 30], "zeroline": True},
        margin={"l": 60, "r": 30, "t": 60, "b": 50},
    )
    if resolved_options.equal_aspect:
        figure.update_yaxes(scaleanchor="x", scaleratio=1)
    return figure
