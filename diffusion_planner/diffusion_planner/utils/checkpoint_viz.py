from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.utils import ddp
from diffusion_planner.validate_model import _prepare_validation_inputs

# matplotlib and visualize_inputs are imported lazily inside
# render_checkpoint_trajectory_figure so that importing this module (which
# train.py always does) does not require matplotlib when checkpoint viz is
# disabled (enable_checkpoint_viz=False). select_and_load_validation_sample and
# the metric helpers below use only numpy/torch.


def _trajectory_metrics(prediction_xy: np.ndarray, gt_xy: np.ndarray) -> dict[str, float]:
    point_error = np.linalg.norm(prediction_xy - gt_xy, axis=-1)
    steps = np.diff(np.vstack(([0.0, 0.0], prediction_xy)), axis=0)
    second_difference = np.diff(prediction_xy, n=2, axis=0)
    return {
        "ADE": float(point_error.mean()),
        "FDE": float(point_error[-1]),
        "max_step": float(np.linalg.norm(steps, axis=-1).max()),
        "roughness": float(np.linalg.norm(second_difference, axis=-1).mean()),
    }


def _select_validation_sample(
    data_list: list[str], min_gt_endpoint_m: float = 10.0
) -> tuple[int, Path]:
    if not data_list:
        raise ValueError("Validation list is empty; cannot create checkpoint visualization.")

    fallback = Path(data_list[0])
    for idx, path_str in enumerate(data_list):
        path = Path(path_str)
        with np.load(path, allow_pickle=True) as data:
            gt_xy = np.asarray(data["ego_agent_future"], dtype=np.float32)[:80, :2]
        if np.linalg.norm(gt_xy[-1]) >= min_gt_endpoint_m:
            return idx, path
    return 0, fallback


def _load_raw_sample(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as data:
        sample = {key: np.asarray(value) for key, value in data.items() if key != "version"}
    return sample


def _batchify_raw_sample(raw_sample: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    batch = {}
    for key, value in raw_sample.items():
        tensor = torch.as_tensor(value)
        if tensor.ndim == 0:
            tensor = tensor.unsqueeze(0)
        else:
            tensor = tensor.unsqueeze(0)
        batch[key] = tensor
    batch["ego_agent_past"] = heading_to_cos_sin(batch["ego_agent_past"])
    batch["goal_pose"] = heading_to_cos_sin(batch["goal_pose"])
    return batch


def render_checkpoint_trajectory_figure(
    model,
    args,
    raw_sample: dict[str, np.ndarray],
    sample_path: Path,
    output_path: Path,
    *,
    title: str,
    seed: int,
) -> dict[str, float]:
    """Render a single validation scene with GT and the checkpoint prediction."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from diffusion_planner.utils.visualize_input import visualize_inputs

    device = torch.device(args.device)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_inputs = _batchify_raw_sample(raw_sample)
    prepared = _prepare_validation_inputs(plot_inputs, args, device)

    net = ddp.get_model(model, args.ddp)
    was_training = net.training
    net.eval()
    try:
        device_index = [torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.no_grad(), torch.random.fork_rng(devices=device_index):
            torch.manual_seed(seed)
            _, outputs = net(prepared.inputs)
    finally:
        net.train(was_training)

    if "prediction" not in outputs:
        raise KeyError("Model output does not contain `prediction`.")

    prediction_xy = outputs["prediction"][0, 0, :, :2].detach().cpu().numpy()
    gt_xy = prepared.ego_future[0, :, :2].detach().cpu().numpy()
    metrics = _trajectory_metrics(prediction_xy, gt_xy)

    fig, ax = plt.subplots(figsize=(11, 7))
    visualize_inputs(plot_inputs, save_path=None, ax=ax)

    origin = np.zeros((1, 2), dtype=np.float32)
    ax.plot(
        *np.vstack((origin, gt_xy)).T,
        color="black",
        linestyle="--",
        linewidth=2.2,
        label="Ground truth",
    )
    ax.plot(
        *np.vstack((origin, prediction_xy)).T,
        color="tab:red",
        linewidth=1.8,
        marker=".",
        markersize=3,
        label="Checkpoint prediction",
    )
    ax.scatter([0.0], [0.0], marker="*", s=110, color="gold", edgecolor="black", zorder=5)

    plotted_xy = np.vstack((origin, gt_xy, prediction_xy))
    xy_min = plotted_xy.min(axis=0)
    xy_max = plotted_xy.max(axis=0)
    span = np.maximum(xy_max - xy_min, 1.0)
    padding = np.maximum(0.12 * span, 2.0)
    ax.set_xlim(xy_min[0] - padding[0], xy_max[0] + padding[0])
    ax.set_ylim(xy_min[1] - padding[1], xy_max[1] + padding[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x in ego frame [m]")
    ax.set_ylabel("y in ego frame [m]")
    ax.set_title(f"{title}\n{sample_path.name}")
    ax.legend(loc="best")
    ax.text(
        0.01,
        0.01,
        (
            f"ADE={metrics['ADE']:.2f} m, FDE={metrics['FDE']:.2f} m\n"
            f"max step={metrics['max_step']:.2f} m, roughness={metrics['roughness']:.2f} m"
        ),
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return metrics


def select_and_load_validation_sample(
    data_list: list[str], min_gt_endpoint_m: float = 10.0
) -> tuple[int, Path, dict[str, np.ndarray]]:
    idx, path = _select_validation_sample(data_list, min_gt_endpoint_m=min_gt_endpoint_m)
    return idx, path, _load_raw_sample(path)
