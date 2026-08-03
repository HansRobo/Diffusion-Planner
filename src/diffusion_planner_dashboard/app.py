"""Application shell for the diffusion planner dashboard."""

from __future__ import annotations

import streamlit as st

from diffusion_planner_dashboard.views import render_frame_browser


def main() -> None:
    """Configure Streamlit and render the active dashboard view."""
    st.set_page_config(page_title="Diffusion Planner Dashboard", layout="wide")
    render_frame_browser()


main()
