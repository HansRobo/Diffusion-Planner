"""Console entry point for the Streamlit dashboard."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """Run the packaged Streamlit application."""
    from streamlit.web import cli as streamlit_cli

    app_path = Path(__file__).with_name("app.py")
    sys.argv = ["streamlit", "run", str(app_path)]
    streamlit_cli.main()
