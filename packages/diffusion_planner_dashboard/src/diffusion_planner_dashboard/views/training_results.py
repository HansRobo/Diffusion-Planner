"""Visualize sampled trajectories from a training checkpoint."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import torch

from diffusion_planner.data import PlannerDataAugmentation
from diffusion_planner.visualizer import plot_frame
from diffusion_planner_dashboard.services import (
    FrameIndex,
    FrameIndexRow,
    FrameLoader,
    LoadedPlanner,
    LoadedTurnIndicator,
    load_frame_index,
    load_planner_checkpoint,
    load_turn_indicator_checkpoint,
    run_inference,
    run_turn_indicator_inference,
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


@st.cache_resource(show_spinner="Loading turn indicator checkpoint...")
def _cached_turn_indicator(
    checkpoint_path: str,
    modification_time_ns: int,
    device: str,
) -> LoadedTurnIndicator:
    del modification_time_ns
    return load_turn_indicator_checkpoint(checkpoint_path, device)


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
    time_epsilon: float,
    noise_scale: float,
    seed: int,
    apply_augmentation: bool,
    lateral_offset: float,
    yaw_offset: float,
    ego_speed_scale: float,
):
    planner = _cached_planner(checkpoint_path, checkpoint_modification_time_ns, device)
    frame_data = _cached_frame(
        h5_path,
        frame_index,
        frame_time_ns,
        h5_modification_time_ns,
    )
    if apply_augmentation:
        frame_data = _augment_frame(
            frame_data, lateral_offset, yaw_offset, ego_speed_scale
        )
    return run_inference(
        planner.model,
        frame_data,
        device=device,
        num_steps=num_steps,
        time_epsilon=time_epsilon,
        noise_scale=noise_scale,
        seed=seed,
    )


@st.cache_data(max_entries=32, show_spinner="Predicting turn indicator...")
def _cached_turn_indicator_prediction(
    checkpoint_path: str,
    checkpoint_modification_time_ns: int,
    h5_path: str,
    frame_index: int,
    frame_time_ns: int,
    h5_modification_time_ns: int,
    device: str,
    apply_augmentation: bool,
    lateral_offset: float,
    yaw_offset: float,
    ego_speed_scale: float,
):
    loaded = _cached_turn_indicator(
        checkpoint_path, checkpoint_modification_time_ns, device
    )
    frame_data = _cached_frame(
        h5_path,
        frame_index,
        frame_time_ns,
        h5_modification_time_ns,
    )
    if apply_augmentation:
        frame_data = _augment_frame(
            frame_data, lateral_offset, yaw_offset, ego_speed_scale
        )
    return run_turn_indicator_inference(loaded.model, frame_data, device=device)


def _augment_frame(
    frame_data: dict[str, Any],
    lateral_offset: float,
    yaw_offset: float,
    ego_speed_scale: float,
) -> dict[str, Any]:
    """Apply a deterministic training augmentation to one frame."""
    augmentation = PlannerDataAugmentation(
        lateral_offset_range=(lateral_offset, lateral_offset),
        yaw_offset_range=(yaw_offset, yaw_offset),
        probability=1.0,
        ego_speed_scale_range=(ego_speed_scale, ego_speed_scale),
    )
    return augmentation(frame_data)


def _render_source_settings() -> tuple[str | None, str | None, str | None]:
    """Apply the frame source and checkpoint together."""
    st.sidebar.subheader("Sources")
    with st.sidebar.form("training-result-source-settings"):
        source_candidate = st.text_input(
            "H5 or Parquet file",
            value=st.session_state.get("configured_frame_source_path", ""),
            placeholder="/path/to/frames.h5 or /path/to/train.parquet",
        )
        checkpoint_candidate = st.text_input(
            "Planner checkpoint file",
            value=st.session_state.get("configured_checkpoint_path", ""),
            placeholder="/path/to/epoch_0001.pth",
        )
        turn_indicator_checkpoint_candidate = st.text_input(
            "Turn indicator checkpoint file (optional)",
            value=st.session_state.get("configured_turn_indicator_checkpoint_path", ""),
            placeholder="/path/to/turn_indicator/epoch_0001.pth",
        )
        applied = st.form_submit_button("Apply sources", use_container_width=True)
    if applied:
        st.session_state["configured_frame_source_path"] = source_candidate.strip()
        st.session_state["configured_checkpoint_path"] = checkpoint_candidate.strip()
        st.session_state["configured_turn_indicator_checkpoint_path"] = (
            turn_indicator_checkpoint_candidate.strip()
        )
    return (
        st.session_state.get("configured_frame_source_path") or None,
        st.session_state.get("configured_checkpoint_path") or None,
        st.session_state.get("configured_turn_indicator_checkpoint_path") or None,
    )


def _render_inference_settings() -> tuple[str, int, float, float, int]:
    st.sidebar.subheader("Inference")
    devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    device = st.sidebar.selectbox("Device", devices)
    num_steps = int(
        st.sidebar.number_input("Sampling steps", min_value=1, value=20, step=1)
    )
    time_epsilon = float(
        st.sidebar.number_input(
            "Time epsilon", min_value=1e-8, value=1e-5, format="%.1e"
        )
    )
    noise_scale = float(
        st.sidebar.number_input("Noise scale", min_value=0.0, value=1.0, step=0.1)
    )
    seed = int(st.sidebar.number_input("Random seed", min_value=0, value=42, step=1))
    return device, num_steps, time_epsilon, noise_scale, seed


def _render_augmentation_settings() -> tuple[bool, float, float, float]:
    """Render deterministic augmentation controls for checkpoint inference."""
    st.sidebar.subheader("Data augmentation")
    enabled = st.sidebar.checkbox("Apply augmentation", value=False)
    lateral_offset = float(
        st.sidebar.slider(
            "Lateral offset [m]",
            min_value=-5.0,
            max_value=5.0,
            value=1.0,
            step=0.1,
            disabled=not enabled,
        )
    )
    yaw_offset_degrees = float(
        st.sidebar.slider(
            "Yaw offset [deg]",
            min_value=-30.0,
            max_value=30.0,
            value=5.0,
            step=0.5,
            disabled=not enabled,
        )
    )
    ego_speed_scale = float(
        st.sidebar.slider(
            "Ego history speed scale",
            min_value=0.5,
            max_value=1.5,
            value=1.0,
            step=0.01,
            disabled=not enabled,
        )
    )
    return enabled, lateral_offset, math.radians(yaw_offset_degrees), ego_speed_scale


def render_training_results() -> None:
    """Render checkpoint inference alongside ground-truth trajectories."""
    st.title("Training Results")
    source_path_text, checkpoint_path_text, turn_indicator_checkpoint_path_text = (
        _render_source_settings()
    )
    device, num_steps, time_epsilon, noise_scale, seed = _render_inference_settings()
    apply_augmentation, lateral_offset, yaw_offset, ego_speed_scale = (
        _render_augmentation_settings()
    )
    if source_path_text is None or checkpoint_path_text is None:
        st.info("Configure both a frame source and a checkpoint from the sidebar.")
        return

    source_path = Path(source_path_text).expanduser()
    checkpoint_path = Path(checkpoint_path_text).expanduser()
    turn_indicator_checkpoint_path = (
        Path(turn_indicator_checkpoint_path_text).expanduser()
        if turn_indicator_checkpoint_path_text is not None
        else None
    )
    turn_indicator_checkpoint_modification_time_ns: int | None = None
    loaded_turn_indicator: LoadedTurnIndicator | None = None
    try:
        source_modification_time_ns = source_path.stat().st_mtime_ns
        checkpoint_modification_time_ns = checkpoint_path.stat().st_mtime_ns
        index = _cached_index(str(source_path), source_modification_time_ns)
        planner = _cached_planner(
            str(checkpoint_path), checkpoint_modification_time_ns, device
        )
        if turn_indicator_checkpoint_path is not None:
            turn_indicator_checkpoint_modification_time_ns = (
                turn_indicator_checkpoint_path.stat().st_mtime_ns
            )
            loaded_turn_indicator = _cached_turn_indicator(
                str(turn_indicator_checkpoint_path),
                turn_indicator_checkpoint_modification_time_ns,
                device,
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

    metadata_columns = st.columns(6)
    metadata_columns[0].metric("Checkpoint epoch", planner.epoch)
    metadata_columns[1].metric("Global step", planner.global_step)
    metadata_columns[2].metric("Sampling steps", num_steps)
    metadata_columns[3].metric("Time epsilon", f"{time_epsilon:.1e}")
    metadata_columns[4].metric("Noise scale", noise_scale)
    metadata_columns[5].metric("Device", device)
    try:
        h5_modification_time_ns = Path(row.h5_path).stat().st_mtime_ns
        frame_data = _cached_frame(
            row.h5_path,
            row.frame_index,
            row.frame_time_ns,
            h5_modification_time_ns,
        )
        visualized_frame = (
            _augment_frame(frame_data, lateral_offset, yaw_offset, ego_speed_scale)
            if apply_augmentation
            else frame_data
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
            time_epsilon,
            noise_scale,
            seed,
            apply_augmentation,
            lateral_offset,
            yaw_offset,
            ego_speed_scale,
        )
        turn_indicator_result = None
        if (
            turn_indicator_checkpoint_path is not None
            and turn_indicator_checkpoint_modification_time_ns is not None
        ):
            turn_indicator_result = _cached_turn_indicator_prediction(
                str(turn_indicator_checkpoint_path),
                turn_indicator_checkpoint_modification_time_ns,
                row.h5_path,
                row.frame_index,
                row.frame_time_ns,
                h5_modification_time_ns,
                device,
                apply_augmentation,
                lateral_offset,
                yaw_offset,
                ego_speed_scale,
            )
    except Exception as error:
        st.exception(error)
        return

    st.caption(
        f"Inference: {inference_seconds:.3f} s · seed: {seed} · "
        f"checkpoint: `{checkpoint_path}`"
    )
    if apply_augmentation:
        st.caption(
            f"Augmentation: lateral offset {lateral_offset:.2f} m · "
            f"yaw offset {math.degrees(yaw_offset):.2f} deg · "
            f"ego history speed scale {ego_speed_scale:.2f}"
        )
    if turn_indicator_result is not None and loaded_turn_indicator is not None:
        probabilities, predicted_report, turn_indicator_seconds = turn_indicator_result
        indicator_names = {0: "Missing", 1: "Disabled", 2: "Left", 3: "Right"}
        target_values = np.asarray(visualized_frame["turn_indicators_future"])
        history_values = np.asarray(visualized_frame["turn_indicators"])
        target_report = int(target_values.reshape(-1)[0])
        current_report = int(history_values.reshape(-1)[-1])
        st.subheader("Turn Indicator Prediction")
        indicator_columns = st.columns(6)
        indicator_columns[0].metric(
            "Prediction", indicator_names.get(predicted_report, str(predicted_report))
        )
        indicator_columns[1].metric(
            "Ground truth", indicator_names.get(target_report, str(target_report))
        )
        indicator_columns[2].metric(
            "Current", indicator_names.get(current_report, str(current_report))
        )
        indicator_columns[3].metric(
            "Confidence", f"{float(probabilities[predicted_report - 1]):.1%}"
        )
        indicator_columns[4].metric("Checkpoint epoch", loaded_turn_indicator.epoch)
        indicator_columns[5].metric("Inference", f"{turn_indicator_seconds:.3f} s")
        probability_columns = st.columns(3)
        for column, name, probability in zip(
            probability_columns,
            ("Disabled", "Left", "Right"),
            probabilities,
            strict=True,
        ):
            column.metric(f"P({name})", f"{float(probability):.1%}")
    figure = plot_frame(
        visualized_frame,
        options=options,
        predicted_trajectory=prediction,
    )
    chart_key = (
        f"training-result::{checkpoint_path}::{checkpoint_modification_time_ns}::"
        f"{row.h5_path}::{row.frame_index}::{num_steps}::{time_epsilon}::"
        f"{noise_scale}::{seed}::{apply_augmentation}::{lateral_offset}::"
        f"{yaw_offset}::{ego_speed_scale}"
    )
    figure.update_layout(autosize=True, uirevision=chart_key)
    st.plotly_chart(
        figure,
        width="stretch",
        height=900,
        key=chart_key,
        config={"responsive": True, "scrollZoom": True},
    )
