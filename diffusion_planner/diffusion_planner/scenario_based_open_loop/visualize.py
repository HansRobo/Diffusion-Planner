"""Visualization helpers for scenario-specific open-loop predictions."""

from pathlib import Path
from typing import Any

_PREDICTION_TIMESTEP_SECONDS = 0.1
_DEFAULT_MATCH_THRESHOLD_M = 0.5


def _as_xy_array(value: Any):
    import numpy as np
    import torch

    if value is None:
        return None
    if torch.is_tensor(value):
        array = value.detach().float().cpu().numpy()
    else:
        array = np.asarray(value, dtype=float)
    if array.size == 0:
        return None
    if array.ndim == 1 and array.shape[0] >= 2:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[-1] < 2:
        return None
    return array[:, :2]


def _as_float_array(value: Any):
    import numpy as np
    import torch

    if value is None:
        return None
    if torch.is_tensor(value):
        array = value.detach().float().cpu().numpy()
    else:
        array = np.asarray(value, dtype=float)
    if array.size == 0:
        return None
    return array.reshape(-1)


def _as_float(value: Any, default: float) -> float:
    import torch

    if value is None:
        return default
    if torch.is_tensor(value):
        return float(value.detach().cpu().reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        return float(value[0]) if value else default
    return float(value)


def visualize_scenario_prediction(
    inputs: dict,
    prediction,
    save_path: str | Path,
    title: str,
    show_neighbors: bool = False,
    view_range: float = 60.0,
    details: dict[str, Any] | None = None,
) -> None:
    """Render one input scene with the predicted ego trajectory overlaid.

    The input NPZ convention stores heading as ``(x, y, heading)`` for the ego
    history and goal pose. It is converted to the cosine/sine representation
    expected by the shared input visualizer before rendering.

    When ``details`` contains ``lateral_offset_m``, the PNG is split into a map
    on top and a heading-frame lateral-offset timeseries below. Offset values
    are not coordinates and are never drawn on the XY map. Metrics without
    that timeseries (for example departure) keep the original map-only figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from matplotlib.collections import LineCollection

    from diffusion_planner.train_epoch import heading_to_cos_sin
    from diffusion_planner.utils.visualize_input import visualize_inputs

    sample_inputs = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            sample_inputs[key] = value[:1]
        else:
            sample_inputs[key] = value

    # The scenario NPZs store headings as (x, y, heading), while the shared
    # visualization helpers expect (x, y, cos(heading), sin(heading)).
    for key in ("ego_agent_past", "goal_pose"):
        if key in sample_inputs:
            sample_inputs[key] = heading_to_cos_sin(sample_inputs[key])

    details = details or {}
    lateral_offset = _as_float_array(details.get("lateral_offset_m"))
    prediction_xy = _as_xy_array(details.get("prediction_xy"))
    closest_xy = _as_xy_array(details.get("closest_centerline_xy"))
    centerline_xy = _as_xy_array(details.get("centerline_xy"))
    has_offset = lateral_offset is not None
    if has_offset:
        fig, (ax_map, ax_offset) = plt.subplots(
            2,
            1,
            figsize=(8, 10),
            gridspec_kw={"height_ratios": [3.2, 1.0]},
        )
    else:
        fig, ax_map = plt.subplots(figsize=(8, 8))
        ax_offset = None

    visualize_inputs(
        sample_inputs,
        ax=ax_map,
        view_ranges=[view_range],
        show_neighbors=show_neighbors,
        show_ego_future=False,
        route_color="#00A6D6",
        route_label=None if centerline_xy is not None else "Route centerline",
    )

    if prediction_xy is None:
        prediction_xy = prediction.detach().float().cpu().numpy()[:, :2]

    if centerline_xy is not None:
        ax_map.plot(
            centerline_xy[:, 0],
            centerline_xy[:, 1],
            color="#00A6D6",
            linewidth=2.0,
            linestyle="--",
            label="route centerline",
            zorder=3,
        )
    ax_map.plot(
        prediction_xy[:, 0],
        prediction_xy[:, 1],
        color="orange",
        linewidth=2,
        label="scenario-based prediction",
        zorder=4,
    )
    ax_map.scatter(
        prediction_xy[-1, 0],
        prediction_xy[-1, 1],
        color="black",
        marker="x",
        label="final point",
        zorder=5,
    )
    if closest_xy is not None:
        ax_map.scatter(
            closest_xy[:, 0],
            closest_xy[:, 1],
            color="tab:red",
            s=14,
            zorder=5,
            label="nearest-centerline feet",
        )
        n_pairs = min(len(prediction_xy), len(closest_xy))
        if n_pairs > 0:
            correspondence = np.stack(
                [prediction_xy[:n_pairs], closest_xy[:n_pairs]],
                axis=1,
            )
            ax_map.add_collection(
                LineCollection(
                    correspondence,
                    colors="0.45",
                    linewidths=0.7,
                    alpha=0.75,
                    zorder=3,
                    label="prediction to centerline",
                )
            )

    ax_map.set_title(title)
    ax_map.legend(loc="best")

    if ax_offset is not None:
        times = np.arange(lateral_offset.shape[0], dtype=float) * _PREDICTION_TIMESTEP_SECONDS
        threshold = abs(_as_float(details.get("match_threshold_m"), _DEFAULT_MATCH_THRESHOLD_M))
        ax_offset.axhspan(
            -threshold,
            threshold,
            color="tab:green",
            alpha=0.18,
            label=f"|n| ≤ {threshold:g} m",
        )
        ax_offset.axhline(0.0, color="black", linewidth=0.9)
        ax_offset.plot(
            times,
            lateral_offset,
            color="tab:purple",
            linewidth=1.6,
            label="lateral_offset_m",
        )
        ax_offset.set_xlabel("time (s)")
        ax_offset.set_ylabel("lateral offset (m)")
        ax_offset.grid(True, alpha=0.3)
        ax_offset.legend(loc="best")

    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
