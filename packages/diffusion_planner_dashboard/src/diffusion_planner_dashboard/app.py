"""Application shell for the diffusion planner dashboard."""

from __future__ import annotations

import streamlit as st

from diffusion_planner_dashboard.views import render_frame_browser, render_home


def main() -> None:
    """Configure Streamlit and run the selected dashboard feature."""
    st.set_page_config(page_title="Diffusion Planner Dashboard", layout="wide")
    navigation = st.navigation(
        {
            "Dashboard": [
                st.Page(render_home, title="Home", icon=":material/home:", default=True),
            ],
            "Features": [
                st.Page(
                    render_frame_browser,
                    title="Frame Browser",
                    icon=":material/animation:",
                ),
            ],
        }
    )
    navigation.run()


main()
