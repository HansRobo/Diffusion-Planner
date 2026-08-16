"""Visualize sampled trajectories from a training checkpoint."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import torch

from diffusion_planner.visualizer import plot_frame
from diffusion_planner_dashboard.services import (
    FrameIndex,
    FrameIndexRow,
    FrameLoader,
    LoadedPlanner,
    load_frame_index,
    load_planner_checkpoint,
    run_inference,
)
from diffusion_planner_dashboard.ui.metadata import (
    render_index_summary,
    render_row_metadata,
)
from diffusion_planner_dashboard.ui.settings import (
    render_frame_selector,
    render_plot_options,
)


@st.cache_data(show_spinner=False)
def _cached_index(path: str, modification_time_ns: int) -> FrameIndex:
    del modification_time_ns
    return load_frame_index(path)


@st.cache_resource
def _frame_loader() -> FrameLoader:
    return FrameLoader()


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


@st.cache_resource(show_spinner="Loading checkpoint...")
def _cached_planner(
    checkpoint_path: str,
    modification_time_ns: int,
    device: str,
) -> LoadedPlanner:
    del modification_time_ns
    return load_planner_checkpoint(checkpoint_path, device)


@st.cache_data(max_entries=32, show_spinner="Sampling trajectories...")
def _cached_prediction(
    checkpoint_path: str,
    checkpoint_modification_time_ns: int,
    h5_path: str,
    frame_index: int,
    frame_time_ns: int,
    h5_modification_time_ns: int,
    device: str,
    num_steps: int,
    noise_scale: float,
    seed: int,
):
    planner = _cached_planner(checkpoint_path, checkpoint_modification_time_ns, device)
    frame_data = _cached_frame(
        h5_path,
        frame_index,
        frame_time_ns,
        h5_modification_time_ns,
    )
    return run_inference(
        planner.model,
        frame_data,
        device=device,
        num_steps=num_steps,
        noise_scale=noise_scale,
        seed=seed,
    )


def _render_source_settings() -> tuple[str | None, str | None]:
    """Apply the frame source and checkpoint together."""
    st.sidebar.subheader("Sources")
    with st.sidebar.form("training-result-source-settings"):
        source_candidate = st.text_input(
            "H5 or Parquet file",
            value=st.session_state.get("configured_frame_source_path", ""),
            placeholder="/path/to/frames.h5 or /path/to/train.parquet",
        )
        checkpoint_candidate = st.text_input(
            "Checkpoint file",
            value=st.session_state.get("configured_checkpoint_path", ""),
            placeholder="/path/to/epoch_0001.pth",
        )
        applied = st.form_submit_button("Apply sources", use_container_width=True)
    if applied:
        st.session_state["configured_frame_source_path"] = source_candidate.strip()
        st.session_state["configured_checkpoint_path"] = checkpoint_candidate.strip()
    return (
        st.session_state.get("configured_frame_source_path") or None,
        st.session_state.get("configured_checkpoint_path") or None,
    )


def _render_inference_settings() -> tuple[str, int, float, int]:
    st.sidebar.subheader("Inference")
    devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    device = st.sidebar.selectbox("Device", devices)
    num_steps = int(
        st.sidebar.number_input("Sampling steps", min_value=1, value=20, step=1)
    )
    noise_scale = float(
        st.sidebar.number_input("Noise scale", min_value=0.0, value=1.0, step=0.1)
    )
    seed = int(st.sidebar.number_input("Random seed", min_value=0, value=42, step=1))
    return device, num_steps, noise_scale, seed


def render_training_results() -> None:
    """Render checkpoint inference alongside ground-truth trajectories."""
    st.title("Training Results")
    source_path_text, checkpoint_path_text = _render_source_settings()
    device, num_steps, noise_scale, seed = _render_inference_settings()
    if source_path_text is None or checkpoint_path_text is None:
        st.info("Configure both a frame source and a checkpoint from the sidebar.")
        return

    source_path = Path(source_path_text).expanduser()
    checkpoint_path = Path(checkpoint_path_text).expanduser()
    try:
        source_modification_time_ns = source_path.stat().st_mtime_ns
        checkpoint_modification_time_ns = checkpoint_path.stat().st_mtime_ns
        index = _cached_index(str(source_path), source_modification_time_ns)
        planner = _cached_planner(
            str(checkpoint_path), checkpoint_modification_time_ns, device
        )
    except (OSError, RuntimeError, ValueError) as error:
        st.error(str(error))
        return
    except Exception as error:
        st.exception(error)
        return

    render_index_summary(index)
    row = render_frame_selector(index)
    render_row_metadata(row)
    options = render_plot_options()

    metadata_columns = st.columns(5)
    metadata_columns[0].metric("Checkpoint epoch", planner.epoch)
    metadata_columns[1].metric("Global step", planner.global_step)
    metadata_columns[2].metric("Sampling steps", num_steps)
    metadata_columns[3].metric("Noise scale", noise_scale)
    metadata_columns[4].metric("Device", device)
    if planner.used_default_config:
        st.warning(
            "This checkpoint does not contain model_config; using the current "
            "DiffusionPlanner defaults."
        )

    try:
        h5_modification_time_ns = Path(row.h5_path).stat().st_mtime_ns
        frame_data = _cached_frame(
            row.h5_path,
            row.frame_index,
            row.frame_time_ns,
            h5_modification_time_ns,
        )
        prediction, inference_seconds = _cached_prediction(
            str(checkpoint_path),
            checkpoint_modification_time_ns,
            row.h5_path,
            row.frame_index,
            row.frame_time_ns,
            h5_modification_time_ns,
            device,
            num_steps,
            noise_scale,
            seed,
        )
    except Exception as error:
        st.exception(error)
        return

    st.caption(
        f"Inference: {inference_seconds:.3f} s · seed: {seed} · "
        f"checkpoint: `{checkpoint_path}`"
    )
    figure = plot_frame(
        frame_data,
        options=options,
        predicted_trajectory=prediction,
    )
    chart_key = (
        f"training-result::{checkpoint_path}::{checkpoint_modification_time_ns}::"
        f"{row.h5_path}::{row.frame_index}::{num_steps}::{noise_scale}::{seed}"
    )
    figure.update_layout(autosize=True, uirevision=chart_key)
    st.plotly_chart(
        figure,
        width="stretch",
        height=900,
        key=chart_key,
        config={"responsive": True, "scrollZoom": True},
    )
