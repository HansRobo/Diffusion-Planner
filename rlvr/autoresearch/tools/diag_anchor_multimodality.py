#!/usr/bin/env python3
"""T4 multimodality and per-trajectory reward diagnostics.

This is the current-branch adaptation of the ``tier4-main``
``diag_anchor_multimodality.py`` and ``grpo_viz.py`` tools.  The upstream tools
assume the anchor-guidance/GRPO stack (``guidance_gui``, ``rlvr.reward`` and a
different sampler).  This repository does not contain that stack on the active
HDP branch, so this tool keeps their useful visual contract but uses the local
T4 contract instead:

* sample K stochastic trajectories with the current diffusion planner;
* score every trajectory with the exact ``compute_hdp_reward`` evaluator used
  by the T4 RL code, with an explicit JSON reward config;
* show real map geometry, logged neighbours and true oriented bounding boxes;
* compare policy samples with structured behavioural anchors (hold, yield,
  turn, etc.) from the same real scene;
* write one detailed dashboard, one per-trajectory diagnostic photo, an
  endpoint/mode coverage report, and an optional time-resolved GIF per scene.

The behavioural anchors are diagnostic probes, not claimed model outputs.  The
orange/coloured trajectories are actual checkpoint samples.  Neighbours remain
logged-future replay, so the result is open-loop evidence rather than a
reactive closed-loop guarantee.

Example (from the repository root)::

    ./.venv/bin/python rlvr/autoresearch/tools/diag_anchor_multimodality.py \
        --model-path outputs/hdp_route_sft20_latest_rl_stable_current_g32_beta1_lr1e7_dp0_ema010_distance_red_ep6_node01/latest.pth \
        --scene-ids red_light_stop,moving_crossing \
        --output-dir docs/t4_rl_assets/t4_multimodality \
        --reward-config docs/t4_rl_assets/t4_hdp_reward_config.json \
        --K 16 --steps 6 --gif

The script intentionally fails if the reward config is missing.  A diagnostic
image with implicit defaults is not a reliable audit of a training run.

For the ordinary original-DP checkpoint used by this proposal, use the
explicit delegation path instead of the HDP anchor ABI::

    ./.venv/bin/python rlvr/autoresearch/tools/diag_anchor_multimodality.py \
        --original-dp-awr --model-path /path/to/original_dp_run/latest.pth \
        --scene-ids red_light_stop,moving_crossing,stopped_leader,tight_right_turn \
        --output-dir docs/t4_conference_assets/awr_colleague_hdp_aug \
        --hdp-trajectory-augmentation --gif

That path delegates all figures to the patched colleague ``grpo_viz.py``;
its default GIF is the complete 80-frame, 10 Hz replay.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docs import generate_t4_scene_reward_visualizations as scene_viz  # noqa: E402

DEFAULT_MODEL = ROOT / (
    "outputs/hdp_route_sft20_latest_rl_stable_current_g32_beta1_lr1e7_dp0_"
    "ema010_distance_red_ep6_node01/latest.pth"
)
DEFAULT_REWARD_CONFIG = ROOT / "docs/t4_rl_assets/t4_hdp_reward_config.json"
DEFAULT_OUTPUT = ROOT / "docs/t4_rl_assets/t4_multimodality"
DEFAULT_T4_ROOT = ROOT.parent / "Diffusion-Planner-t4-main"
DEFAULT_T4_GRPO_VIZ = DEFAULT_T4_ROOT / "rlvr/autoresearch/tools/grpo_viz.py"
DEFAULT_T4_SCENES = ROOT / "docs/t4_conference_assets/t4_scene_list.json"
DEFAULT_T4_REWARD_CONFIG = DEFAULT_T4_ROOT / "rlvr/configs/awr_original_dp_t4_hdp_faithful.json"

COMPONENT_KEYS = [key for key, _label in scene_viz.COMPONENTS]
COMPONENT_LABELS = [label for _key, label in scene_viz.COMPONENTS]


def _load_reward_args(config_path: Path):
    """Load the local HDP reward args and apply an explicit JSON override."""
    if not config_path.exists():
        raise FileNotFoundError(
            f"Reward config does not exist: {config_path}. "
            "Pass --reward-config with a versioned JSON contract."
        )
    with config_path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Reward config must be a JSON object: {config_path}")
    args = scene_viz._make_reward_args()
    for key, value in payload.items():
        if key.startswith("_"):
            continue
        setattr(args, key, value)
    return args, payload


def _args_path_for_checkpoint(checkpoint: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(explicit)
        return explicit
    candidates = [checkpoint.parent / "args.json", checkpoint.parent.parent / "args.json"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find args.json next to checkpoint {checkpoint}; pass --args-path."
    )


def _load_policy(
    checkpoint: Path,
    args_path: Path,
    *,
    device: torch.device,
    legacy_checkpoint_abi: bool,
):
    """Load an HDP/DP-compatible planner and return (model, config).

    The saved local RL checkpoint predates the current LineEncoder ABI.  The
    compatibility shim is enabled by default for that checkpoint, but can be
    disabled for a native current ordinary-DP checkpoint with
    ``--no-legacy-checkpoint-abi``.
    """
    if legacy_checkpoint_abi:
        scene_viz._install_legacy_line_encoder()
    from diffusion_planner.model.diffusion_planner import Diffusion_Planner
    from diffusion_planner.utils.config import Config

    cfg = Config(str(args_path))
    cfg.device = str(device)
    cfg.ddp = False
    cfg.amp_dtype = "off"
    if legacy_checkpoint_abi:
        # The checkpoint used the six-frame history ABI before this branch's
        # configurable longer history refactor.
        cfg.ego_history_frames = 6
    model = Diffusion_Planner(cfg).to(device).eval()
    package = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = package.get("ema_state_dict", package.get("model", package))
    state = {key.removeprefix("module."): value for key, value in state.items()}
    incompatible = model.load_state_dict(state, strict=False)
    missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith("decoder.turn_indicator_predictor.")
    ]
    unexpected = [
        key
        for key in incompatible.unexpected_keys
        if not key.startswith("decoder.turn_indicator_predictor.")
    ]
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model ABI mismatch. Use --legacy-checkpoint-abi for the "
            f"pre-refactor local RL checkpoint, or inspect the ordinary DP adapter. "
            f"missing={missing}, unexpected={unexpected}"
        )
    return model, cfg


def _sample_checkpoint(
    scenes: list[dict],
    checkpoint: Path,
    args_path: Path,
    *,
    generations: int,
    steps: int,
    noise_scale: float,
    seed: int,
    legacy_checkpoint_abi: bool,
) -> dict[str, list[np.ndarray]]:
    """Sample K trajectories per scene through the local RL sampler."""
    if generations < 2:
        raise ValueError("--K must be at least 2 for a multimodality diagnostic")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = _load_policy(
        checkpoint,
        args_path,
        device=device,
        legacy_checkpoint_abi=legacy_checkpoint_abi,
    )
    from diffusion_planner.hdp_rl_epoch import _grouped_policy_inputs
    from diffusion_planner.hdp_rl_utils import sample_group

    raw_scenes = [scene_viz._raw_tensors(scene)[0] for scene in scenes]
    raw = {key: torch.cat([item[key] for item in raw_scenes], dim=0) for key in raw_scenes[0]}
    normalized = cfg.observation_normalizer(raw)
    grouped = _grouped_policy_inputs(normalized, generations, True)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    samples = sample_group(
        model,
        grouped,
        float(noise_scale),
        device,
        scene_norm_inputs=normalized,
        group_size=generations,
        sample_steps=int(steps),
        generator=generator,
    )
    samples = samples.detach().cpu().numpy().reshape(len(scenes), generations, 80, 4)
    return {
        scene["id"]: [samples[scene_index, k].astype(np.float32) for k in range(generations)]
        for scene_index, scene in enumerate(scenes)
    }


def _arc_resampled(xy: np.ndarray, points: int = 20) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32)
    if len(xy) == 0:
        return np.zeros((points, 2), dtype=np.float32)
    segment = np.linalg.norm(np.diff(xy, axis=0), axis=-1)
    arc = np.concatenate(([0.0], np.cumsum(segment)))
    if arc[-1] <= 1e-6:
        return np.repeat(xy[:1], points, axis=0)
    samples = np.linspace(0.0, arc[-1], points)
    return np.column_stack(
        [np.interp(samples, arc, xy[:, axis]) for axis in range(xy.shape[1])]
    ).astype(np.float32)


def _shape_distance(a: np.ndarray, b: np.ndarray) -> float:
    aa = _arc_resampled(a[:, :2])
    bb = _arc_resampled(b[:, :2])
    return float(np.linalg.norm(aa - bb, axis=-1).mean())


def _connected_components(distance: np.ndarray, threshold: float) -> int:
    n = int(distance.shape[0])
    seen = np.zeros(n, dtype=bool)
    count = 0
    for start in range(n):
        if seen[start]:
            continue
        count += 1
        stack = [start]
        seen[start] = True
        while stack:
            node = stack.pop()
            neighbours = np.flatnonzero(distance[node] <= threshold)
            for neighbour in neighbours:
                if not seen[neighbour]:
                    seen[neighbour] = True
                    stack.append(int(neighbour))
    return count


def _nearest_anchor_names(
    trajectories: list[np.ndarray], anchors: list[dict]
) -> tuple[list[str], np.ndarray]:
    anchor_xy = [item["trajectory"][:, :2] for item in anchors]
    if not anchor_xy:
        return ["none"] * len(trajectories), np.full(len(trajectories), np.nan)
    names, distances = [], []
    for trajectory in trajectories:
        d = np.asarray([_shape_distance(trajectory, anchor) for anchor in anchor_xy])
        index = int(d.argmin())
        names.append(str(anchors[index]["name"]))
        distances.append(float(d[index]))
    return names, np.asarray(distances, dtype=np.float32)


def _score_model_group(scene: dict, model_trajectories: list[np.ndarray], reward_args):
    """Score expert, structured anchors and actual model samples separately."""
    probes = scene_viz._candidate_set(scene)
    expert = probes[0]
    anchors = probes[1:]
    actual = [
        {
            "name": f"RL sample {index + 1}",
            "kind": "model",
            "trajectory": trajectory,
        }
        for index, trajectory in enumerate(model_trajectories)
    ]
    actual_rows = scene_viz._score_candidates(scene, [expert] + actual, reward_args)[1:]
    anchor_rows = scene_viz._score_candidates(scene, [expert] + anchors, reward_args)[1:]
    raw, neighbours = scene_viz._raw_tensors(scene)
    clearance = scene_viz._signed_clearance(
        np.stack(model_trajectories).astype(np.float32), raw, neighbours
    )
    endpoint = np.asarray([trajectory[-1, :2] for trajectory in model_trajectories])
    endpoint_distance = np.linalg.norm(endpoint[:, None] - endpoint[None], axis=-1)
    shape_distance = np.asarray(
        [[_shape_distance(a, b) for b in model_trajectories] for a in model_trajectories],
        dtype=np.float32,
    )
    anchor_names, anchor_distance = _nearest_anchor_names(model_trajectories, anchors)
    for index, row in enumerate(actual_rows):
        row["sample_index"] = index + 1
        row["min_signed_clearance_m"] = float(np.nanmin(clearance[index]))
        row["anchor_mode"] = anchor_names[index]
        row["anchor_shape_distance_m"] = float(anchor_distance[index])
        row["trajectory"] = model_trajectories[index]
    model_order = np.argsort([-row["reward"] for row in actual_rows]).astype(int)
    rank = {int(index): int(position) for position, index in enumerate(model_order)}
    for index, row in enumerate(actual_rows):
        row["rank"] = rank[index] + 1
    metrics = {
        "K": len(model_trajectories),
        "reward_mean": float(np.mean([row["reward"] for row in actual_rows])),
        "reward_std": float(np.std([row["reward"] for row in actual_rows])),
        "reward_best": float(np.max([row["reward"] for row in actual_rows])),
        "reward_worst": float(np.min([row["reward"] for row in actual_rows])),
        "safe_candidate_count": int(
            sum(row["safety"] > 0.999 and row["reward"] > 0.0 for row in actual_rows)
        ),
        "endpoint_spread_m": float(endpoint_distance.max()),
        "endpoint_lateral_std_m": float(endpoint[:, 1].std()),
        "mean_pairwise_endpoint_m": float(
            endpoint_distance[np.triu_indices(len(endpoint), 1)].mean()
        ),
        "trajectory_shape_spread_m": float(shape_distance.max()),
        "trajectory_shape_mean_pairwise_m": float(
            shape_distance[np.triu_indices(len(shape_distance), 1)].mean()
        ),
        "endpoint_mode_count_2m": _connected_components(endpoint_distance, 2.0),
        "shape_mode_count_2m": _connected_components(shape_distance, 2.0),
        "anchor_mode_count": len(set(anchor_names)),
        "anchor_mode_histogram": {
            name: int(anchor_names.count(name)) for name in sorted(set(anchor_names))
        },
        "actual_model_device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    return {
        "expert": expert,
        "anchors": anchors,
        "anchor_rows": anchor_rows,
        "actual": actual,
        "actual_rows": actual_rows,
        "metrics": metrics,
        "clearance": clearance,
        "endpoint_distance": endpoint_distance,
        "shape_distance": shape_distance,
    }


def _rank_color(rank: int, count: int):
    # Higher reward rank = green; lower rank = red, matching grpo_viz.
    cmap = plt.get_cmap("RdYlGn")
    return cmap(1.0 - rank / max(count - 1, 1))


def _draw_scene_map(
    ax,
    scene: dict,
    group: dict,
    *,
    selected_index: int | None = None,
    title: str | None = None,
    extent=None,
    time_index: int = 40,
):
    all_candidates = [group["expert"]] + group["anchors"] + group["actual"]
    if extent is None:
        extent = scene_viz._scene_extent(scene, all_candidates)
    scene_viz._draw_map(ax, scene["data"], extent)
    scene_viz._draw_neighbors(ax, scene["data"], time_index, extent, selected=None, alpha_path=0.22)
    expert = group["expert"]["trajectory"]
    ax.plot(
        expert[:, 0],
        expert[:, 1],
        color="#16a34a",
        lw=2.8,
        linestyle=(0, (7, 3)),
        label="logged expert",
        zorder=8,
    )
    for anchor_index, anchor in enumerate(group["anchors"]):
        trajectory = anchor["trajectory"]
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color="#7c3aed",
            lw=0.9,
            linestyle=(0, (3, 3)),
            alpha=0.32,
            zorder=5,
            label="behavioural anchor" if anchor_index == 0 else None,
        )
    order = np.argsort([row["reward"] for row in group["actual_rows"]])[::-1]
    for position, index in enumerate(order):
        trajectory = group["actual"][index]["trajectory"]
        row = group["actual_rows"][index]
        color = _rank_color(int(row["rank"] - 1), len(order))
        alpha = 0.82 if position < 3 else 0.34
        linewidth = 2.4 if position < 3 else 1.15
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=color,
            lw=linewidth,
            alpha=alpha,
            zorder=7,
            label=f"RL samples (top {min(3, len(order))})" if position == 0 else None,
        )
        ax.scatter(
            trajectory[-1, 0],
            trajectory[-1, 1],
            color=color,
            s=24,
            edgecolor="white",
            linewidth=0.6,
            zorder=9,
        )
        if position < 3 or index == selected_index:
            pose = trajectory[min(time_index, len(trajectory) - 1)]
            heading = math.atan2(float(pose[3]), float(pose[2]))
            scene_viz._draw_oriented_box(
                ax,
                float(pose[0]),
                float(pose[1]),
                heading,
                float(scene["data"]["ego_shape"][1]),
                float(scene["data"]["ego_shape"][2]),
                color,
                alpha=0.33,
                lw=1.8,
                rear_axle=True,
                zorder=12,
            )
    scene_viz._draw_oriented_box(
        ax,
        0.0,
        0.0,
        0.0,
        float(scene["data"]["ego_shape"][1]),
        float(scene["data"]["ego_shape"][2]),
        "#2563eb",
        alpha=0.28,
        lw=1.8,
        rear_axle=True,
        zorder=13,
    )
    ax.scatter([0], [0], color="#2563eb", s=32, zorder=14, label="ego at t=0")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#cbd5e1", linewidth=0.5, alpha=0.35)
    ax.set_xlabel("ego-frame x (m)")
    ax.set_ylabel("ego-frame y (m)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.93)
    ax.text(
        0.01,
        0.01,
        f"oriented boxes at t={time_index / 10:.1f}s | red=border | yellow dash=stop line | blue=route",
        transform=ax.transAxes,
        fontsize=8,
        color="#475569",
        va="bottom",
    )
    if title:
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold")


def _draw_table(ax, group: dict):
    ax.axis("off")
    rows = sorted(group["actual_rows"], key=lambda row: row["rank"])
    headers = [
        "rank",
        "reward",
        "safety",
        "risk",
        "follow",
        "lane",
        "progress",
        "clear(m)",
        "nearest anchor",
    ]
    cell_text = []
    cell_colors = []
    for row in rows:
        cell_text.append(
            [
                f"#{row['rank']}",
                f"{row['reward']:.2f}",
                f"{row['safety']:.2f}",
                f"{row['risk']:.2f}",
                f"{row['follow']:.2f}",
                f"{row['lane']:.2f}",
                f"{row['progress']:.2f}",
                f"{row['min_signed_clearance_m']:+.2f}",
                row["anchor_mode"],
            ]
        )
        if row["safety"] > 0.999 and row["reward"] > 0.0:
            cell_colors.append(["#e8f5e9"] * len(headers))
        else:
            cell_colors.append(["#fff1f2"] * len(headers))
    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        cellColours=cell_colors,
        colColours=["#e8eef5"] * len(headers),
        cellLoc="center",
        colLoc="center",
        loc="upper center",
        bbox=[0.0, 0.05, 1.0, 0.88],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    for (row_index, _col_index), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        cell.set_linewidth(0.45)
        if row_index == 0:
            cell.set_text_props(weight="bold", color="#102a43")
    ax.set_title(
        "Every stochastic sample: reward and safety breakdown (green = feasible)",
        loc="left",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )
    metrics = group["metrics"]
    summary = (
        f"K={metrics['K']} | safe {metrics['safe_candidate_count']}/{metrics['K']} | "
        f"endpoint spread {metrics['endpoint_spread_m']:.2f} m | "
        f"shape spread {metrics['trajectory_shape_spread_m']:.2f} m | "
        f"behaviour modes {metrics['anchor_mode_count']}"
    )
    ax.text(0.0, 0.0, summary, transform=ax.transAxes, fontsize=9, color="#475569")


def _draw_heatmap(ax, group: dict):
    order = np.argsort([row["reward"] for row in group["actual_rows"]])[::-1]
    matrix = np.asarray(
        [[group["actual_rows"][index][key] for key in COMPONENT_KEYS] for index in order],
        dtype=np.float32,
    )
    labels = [
        f"#{group['actual_rows'][index]['rank']} · {group['actual_rows'][index]['anchor_mode']}"
        for index in order
    ]
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, aspect="auto", cmap="RdYlGn")
    ax.set_yticks(np.arange(len(order)), labels, fontsize=7)
    ax.set_xticks(
        np.arange(len(COMPONENT_KEYS)), COMPONENT_LABELS, rotation=28, ha="right", fontsize=8
    )
    ax.set_title(
        "Why trajectories rank differently: exact evaluator components",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            ax.text(
                col_index,
                row_index,
                f"{matrix[row_index, col_index]:.2f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="#102a43",
            )
    ax.set_xlabel("normalized component score")
    ax.grid(False)
    ax.figure.colorbar(image, ax=ax, fraction=0.025, pad=0.02)


def _draw_endpoint_modes(ax, scene: dict, group: dict):
    rows = group["actual_rows"]
    endpoints = np.asarray([item["trajectory"][-1, :2] for item in group["actual"]])
    anchor_names = sorted(set(row["anchor_mode"] for row in rows))
    colors = plt.get_cmap("tab10")
    color_map = {name: colors(index % 10) for index, name in enumerate(anchor_names)}
    for index, (point, row) in enumerate(zip(endpoints, rows, strict=False)):
        color = color_map[row["anchor_mode"]]
        ax.scatter(
            point[0], point[1], s=65, color=color, edgecolor="white", linewidth=1.0, zorder=5
        )
        ax.text(
            point[0],
            point[1],
            str(index + 1),
            fontsize=7,
            ha="center",
            va="center",
            color="#102a43",
            zorder=6,
        )
    expert = group["expert"]["trajectory"][-1, :2]
    ax.scatter(
        expert[0],
        expert[1],
        marker="*",
        s=170,
        color="#16a34a",
        edgecolor="white",
        linewidth=1.0,
        label="logged expert",
    )
    for name, color in color_map.items():
        ax.scatter([], [], color=color, s=45, label=name)
    ax.set_title(
        "Endpoint coverage and nearest behavioural mode", loc="left", fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("endpoint x (m)")
    ax.set_ylabel("endpoint y (m)")
    ax.grid(color="#cbd5e1", alpha=0.45)
    ax.legend(fontsize=7, loc="best", framealpha=0.94)
    ax.set_aspect("equal", adjustable="datalim")


def _plot_dashboard(scene: dict, group: dict, output: Path):
    all_candidates = [group["expert"]] + group["anchors"] + group["actual"]
    extent = scene_viz._scene_extent(scene, all_candidates)
    fig = plt.figure(figsize=(21, 14), facecolor="#f8fafc")
    grid = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1.25, 1.0],
        height_ratios=[1.08, 0.92],
        hspace=0.28,
        wspace=0.22,
    )
    ax_map = fig.add_subplot(grid[0, 0])
    _draw_scene_map(
        ax_map,
        scene,
        group,
        extent=extent,
        title=f"{scene['short']} · K stochastic outputs + behavioural anchors",
    )
    ax_table = fig.add_subplot(grid[0, 1])
    _draw_table(ax_table, group)
    ax_heat = fig.add_subplot(grid[1, 0])
    _draw_heatmap(ax_heat, group)
    ax_modes = fig.add_subplot(grid[1, 1])
    _draw_endpoint_modes(ax_modes, scene, group)
    metrics = group["metrics"]
    fig.suptitle(
        f"T4 multimodality diagnostic · {scene['title']} · "
        f"safe {metrics['safe_candidate_count']}/{metrics['K']} · "
        f"endpoint modes {metrics['endpoint_mode_count_2m']} · shape modes {metrics['shape_mode_count_2m']}",
        x=0.035,
        y=0.997,
        ha="left",
        fontsize=19,
        fontweight="bold",
        color="#0f172a",
    )
    fig.text(
        0.035,
        0.008,
        "Green dashed = logged expert; purple dashed = structured behavioural anchors; "
        "coloured lines = actual checkpoint samples ranked by exact T4 reward. "
        "Neighbour motion is recorded future replay; bboxes are oriented, not point markers.",
        fontsize=8.5,
        color="#475569",
        ha="left",
    )
    fig.savefig(output, dpi=185, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_single_trajectory(scene: dict, group: dict, row_index: int, output: Path):
    row = group["actual_rows"][row_index]
    trajectory = group["actual"][row_index]["trajectory"]
    all_candidates = [group["expert"]] + group["anchors"] + group["actual"]
    extent = scene_viz._scene_extent(scene, all_candidates)
    fig = plt.figure(figsize=(19, 11), facecolor="#f8fafc")
    grid = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1.25, 0.85],
        height_ratios=[1.1, 0.9],
        hspace=0.28,
        wspace=0.20,
    )
    ax_map = fig.add_subplot(grid[:, 0])
    _draw_scene_map(
        ax_map,
        scene,
        group,
        selected_index=row_index,
        extent=extent,
        title=f"Trajectory #{row['rank']} · RL sample {row['sample_index']} · {row['anchor_mode']}",
    )
    # Overlay the selected trajectory strongly on top of the group.
    ax_map.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        color="#0f172a",
        lw=3.0,
        alpha=0.90,
        zorder=15,
        label="selected trajectory",
    )
    pose = trajectory[40]
    scene_viz._draw_oriented_box(
        ax_map,
        float(pose[0]),
        float(pose[1]),
        math.atan2(float(pose[3]), float(pose[2])),
        float(scene["data"]["ego_shape"][1]),
        float(scene["data"]["ego_shape"][2]),
        "#0f172a",
        alpha=0.16,
        lw=2.2,
        rear_axle=True,
        zorder=16,
    )
    ax_map.legend(loc="upper left", fontsize=8, framealpha=0.93)

    ax_bar = fig.add_subplot(grid[0, 1])
    values = np.asarray([row[key] for key in COMPONENT_KEYS], dtype=float)
    bars = ax_bar.barh(
        np.arange(len(values)),
        values,
        color=plt.get_cmap("RdYlGn")(values),
        edgecolor="#334155",
        linewidth=0.4,
    )
    ax_bar.set_yticks(np.arange(len(values)), COMPONENT_LABELS)
    ax_bar.set_xlim(0.0, 1.05)
    ax_bar.set_xlabel("component score")
    ax_bar.set_title(
        f"Reward = {row['reward']:.3f} · exact T4 components",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )
    ax_bar.grid(axis="x", color="#cbd5e1", alpha=0.45)
    for bar, value in zip(bars, values, strict=False):
        ax_bar.text(
            min(value + 0.015, 1.01),
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:.2f}",
            va="center",
            fontsize=8,
        )

    ax_time = fig.add_subplot(grid[1, 1])
    clearance = group["clearance"][row_index]
    expert_clearance = scene_viz._signed_clearance(
        group["expert"]["trajectory"][None].astype(np.float32),
        scene_viz._raw_tensors(scene)[0],
        scene_viz._raw_tensors(scene)[1],
    )[0]
    times = np.arange(len(clearance)) / 10.0
    ax_time.plot(
        times,
        clearance,
        color="#dc2626" if np.nanmin(clearance) < 0 else "#0f766e",
        lw=2.2,
        label="selected OBB clearance",
    )
    ax_time.plot(
        times,
        expert_clearance,
        color="#16a34a",
        lw=1.4,
        linestyle=(0, (5, 3)),
        label="expert clearance",
    )
    ax_time.axhline(0, color="#dc2626", linestyle="--", lw=1.0)
    ax_time.fill_between(times, -3.0, 0.0, color="#fecaca", alpha=0.35)
    ax_time.set_xlabel("time (s)")
    ax_time.set_ylabel("minimum signed OBB clearance (m)")
    ax_time.set_title("Collision geometry over time", loc="left", fontsize=12, fontweight="bold")
    ax_time.grid(color="#cbd5e1", alpha=0.45)
    ax_time.legend(fontsize=7, loc="best")
    y_min = float(np.nanmin(np.r_[clearance, expert_clearance]))
    y_max = float(np.nanmax(np.r_[clearance, expert_clearance]))
    ax_time.set_ylim(min(-3.0, y_min - 0.4), max(3.0, y_max + 0.4))
    fig.suptitle(
        f"T4 per-trajectory audit · {scene['title']} · rank #{row['rank']} · "
        f"clearance {row['min_signed_clearance_m']:+.3f} m",
        x=0.035,
        y=0.997,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#0f172a",
    )
    fig.text(
        0.035,
        0.008,
        f"nearest behavioural anchor: {row['anchor_mode']} ({row['anchor_shape_distance_m']:.2f} m shape distance) | "
        f"endpoint=({row['endpoint_x_m']:.2f}, {row['endpoint_y_m']:.2f}) m | "
        "neighbours are logged future replay",
        fontsize=8.5,
        color="#475569",
        ha="left",
    )
    fig.savefig(output, dpi=190, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _render_gif(scene: dict, group: dict, output: Path):
    from PIL import Image

    all_candidates = [group["expert"]] + group["anchors"] + group["actual"]
    extent = scene_viz._scene_extent(scene, all_candidates)
    order = np.argsort([row["reward"] for row in group["actual_rows"]])[::-1]
    frames = []
    for time_index in range(0, 80, 2):
        fig, ax = plt.subplots(figsize=(10, 7), facecolor="#f8fafc")
        scene_viz._draw_map(ax, scene["data"], extent)
        scene_viz._draw_neighbors(scene["data"], time_index, extent, selected=None, alpha_path=0.18)
        expert = group["expert"]["trajectory"]
        ax.plot(
            expert[:, 0],
            expert[:, 1],
            color="#16a34a",
            lw=1.8,
            linestyle=(0, (5, 3)),
            label="logged expert",
        )
        for anchor_index, anchor in enumerate(group["anchors"]):
            ax.plot(
                anchor["trajectory"][:, 0],
                anchor["trajectory"][:, 1],
                color="#7c3aed",
                lw=0.7,
                linestyle=(0, (3, 3)),
                alpha=0.20,
                label="behavioural anchor" if anchor_index == 0 else None,
            )
        for position, index in enumerate(order):
            trajectory = group["actual"][index]["trajectory"]
            row = group["actual_rows"][index]
            color = _rank_color(int(row["rank"] - 1), len(order))
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color=color,
                lw=1.9 if position < 3 else 0.75,
                alpha=0.78 if position < 3 else 0.20,
                label="actual RL samples" if position == 0 else None,
            )
            pose = trajectory[time_index]
            if position < 3:
                scene_viz._draw_oriented_box(
                    ax,
                    float(pose[0]),
                    float(pose[1]),
                    math.atan2(float(pose[3]), float(pose[2])),
                    float(scene["data"]["ego_shape"][1]),
                    float(scene["data"]["ego_shape"][2]),
                    color,
                    alpha=0.34,
                    lw=1.5,
                    rear_axle=True,
                    zorder=12,
                )
        scene_viz._draw_oriented_box(
            ax,
            0.0,
            0.0,
            0.0,
            float(scene["data"]["ego_shape"][1]),
            float(scene["data"]["ego_shape"][2]),
            "#2563eb",
            alpha=0.25,
            lw=1.5,
            rear_axle=True,
            zorder=13,
        )
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="#cbd5e1", alpha=0.35)
        ax.set_xlabel("ego-frame x (m)")
        ax.set_ylabel("ego-frame y (m)")
        ax.set_title(
            f"{scene['short']} · replay time {time_index / 10:.1f}s · K={len(order)} stochastic outputs · oriented OBBs",
            loc="left",
            fontsize=13,
            fontweight="bold",
        )
        ax.legend(loc="upper left", fontsize=8, framealpha=0.93)
        ax.text(
            0.99,
            0.02,
            "neighbour motion = logged future\ncolours = reward rank",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#475569",
            bbox=dict(
                boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cbd5e1", alpha=0.90
            ),
        )
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=100,
        loop=0,
        disposal=2,
    )


def _scene_specs(scene_ids: list[str]) -> list[dict]:
    specs = {spec["id"]: spec for spec in scene_viz.SCENES}
    missing = [scene_id for scene_id in scene_ids if scene_id not in specs]
    if missing:
        raise ValueError(f"Unknown T4 scene ids: {missing}; available={sorted(specs)}")
    return [dict(specs[scene_id]) for scene_id in scene_ids]


def _serializable_row(row: dict) -> dict:
    result = {}
    for key, value in row.items():
        if key == "trajectory":
            continue
        if isinstance(value, np.generic):
            value = value.item()
        result[key] = value
    return result


def _delegate_original_dp_awr(args) -> None:
    """Run the colleague ``grpo_viz.py`` path for ordinary DP checkpoints.

    The original version of this anchor diagnostic loads the active HDP
    temporal-token ABI and therefore cannot load a plain x-start DP
    checkpoint.  Keeping a named delegation here prevents users from
    accidentally producing a misleading anchor plot: ordinary-DP evidence is
    generated by the patched colleague visualizer, with its own AWR sampler,
    reward-compatible view, +1 alignment and bbox/GIF renderer.
    """

    tool = Path(args.t4_grpo_viz).resolve()
    scenes = Path(args.t4_scenes_json).resolve()
    reward = Path(args.t4_reward_config).resolve()
    for required in (tool, scenes, reward):
        if not required.exists():
            raise FileNotFoundError(required)
    scene_map = {
        "red_light_stop": 0,
        "moving_crossing": 1,
        "stopped_leader": 2,
        "tight_right_turn": 3,
    }
    requested = [item.strip() for item in args.scene_ids.split(",") if item.strip()]
    unknown = [item for item in requested if item not in scene_map]
    if unknown:
        raise ValueError(f"unknown T4 scene ids for original-DP delegation: {unknown}")
    command = [
        sys.executable,
        str(tool),
        "--model_path",
        str(args.model_path.resolve()),
        "--scenes",
        str(scenes),
        "--output_dir",
        str(args.output_dir.resolve()),
        "--indices",
        *(str(scene_map[item]) for item in requested),
        "--K",
        str(args.K),
        "--tag",
        "original_dp_awr",
        "--awr_mode",
        "--reward_config",
        str(reward),
        "--sample_steps",
        str(args.steps),
        "--noise_scale",
        str(args.noise_scale),
        "--seed",
        str(args.seed),
        "--no-deterministic_first",
    ]
    if args.hdp_trajectory_augmentation:
        command.extend(
            [
                "--hdp_trajectory_augmentation",
                "--hdp_trajectory_augmentation_std",
                str(args.hdp_trajectory_augmentation_std),
            ]
        )
    if args.gif:
        command.extend(
            [
                "--gif",
                "--gif_frame_stride",
                str(args.gif_frame_stride),
                "--gif_frame_duration_ms",
                str(args.gif_frame_duration_ms),
            ]
        )
    if args.scene_meta:
        command.extend(["--scene_meta", str(args.scene_meta.resolve())])
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(DEFAULT_T4_ROOT / "diffusion_planner"),
            str(DEFAULT_T4_ROOT),
            env.get("PYTHONPATH", ""),
        ]
    )
    print("Delegating ordinary-DP AWR visualization to colleague grpo_viz.py:", flush=True)
    print(" ".join(command), flush=True)
    subprocess.run(command, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--args-path", type=Path, default=None)
    parser.add_argument("--scene-ids", default="red_light_stop,moving_crossing")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reward-config", type=Path, default=DEFAULT_REWARD_CONFIG)
    parser.add_argument(
        "--K", type=int, default=16, help="number of stochastic model trajectories per scene"
    )
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--noise-scale", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument(
        "--gif", action="store_true", help="write a 40-frame 10 Hz replay GIF per scene"
    )
    parser.add_argument(
        "--per-traj",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write a detailed PNG for every sampled trajectory",
    )
    parser.add_argument(
        "--legacy-checkpoint-abi",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use the pre-LineEncoder-refactor compatibility shim for the saved local RL checkpoint",
    )
    parser.add_argument(
        "--original-dp-awr",
        action="store_true",
        help="delegate ordinary-DP AWR visualization to the patched colleague grpo_viz.py",
    )
    parser.add_argument("--t4-grpo-viz", type=Path, default=DEFAULT_T4_GRPO_VIZ)
    parser.add_argument("--t4-scenes-json", type=Path, default=DEFAULT_T4_SCENES)
    parser.add_argument("--t4-reward-config", type=Path, default=DEFAULT_T4_REWARD_CONFIG)
    parser.add_argument(
        "--scene-meta", type=Path, default=ROOT / "docs/t4_conference_assets/t4_scene_metadata.json"
    )
    parser.add_argument("--hdp-trajectory-augmentation", action="store_true")
    parser.add_argument("--hdp-trajectory-augmentation-std", type=float, default=0.5)
    parser.add_argument("--gif-frame-stride", type=int, default=1)
    parser.add_argument("--gif-frame-duration-ms", type=int, default=100)
    args = parser.parse_args()

    if args.original_dp_awr:
        _delegate_original_dp_awr(args)
        return

    checkpoint = args.model_path.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    scene_ids = [item.strip() for item in args.scene_ids.split(",") if item.strip()]
    scenes = [scene_viz._load_scene(spec) for spec in _scene_specs(scene_ids)]
    reward_args, reward_payload = _load_reward_args(args.reward_config.resolve())
    args_path = _args_path_for_checkpoint(
        checkpoint, args.args_path.resolve() if args.args_path else None
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Sampling {args.K} trajectories for {len(scenes)} scene(s) with {args.steps} diffusion steps",
        flush=True,
    )
    print(f"checkpoint={checkpoint}", flush=True)
    print(f"reward_config={args.reward_config.resolve()}", flush=True)
    samples = _sample_checkpoint(
        scenes,
        checkpoint,
        args_path,
        generations=args.K,
        steps=args.steps,
        noise_scale=args.noise_scale,
        seed=args.seed,
        legacy_checkpoint_abi=args.legacy_checkpoint_abi,
    )
    manifest = {
        "tool": "rlvr/autoresearch/tools/diag_anchor_multimodality.py",
        "adapted_from": {
            "source_branch": "origin/tier4-main",
            "anchor_tool": "rlvr/autoresearch/tools/diag_anchor_multimodality.py",
            "breakdown_tool": "rlvr/autoresearch/tools/grpo_viz.py",
        },
        "checkpoint": str(checkpoint),
        "args_path": str(args_path),
        "reward_config_path": str(args.reward_config.resolve()),
        "reward_config": reward_payload,
        "K": int(args.K),
        "steps": int(args.steps),
        "noise_scale": float(args.noise_scale),
        "seed": int(args.seed),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "scenes": {},
    }
    for scene in scenes:
        group = _score_model_group(scene, samples[scene["id"]], reward_args)
        scene_dir = args.output_dir / scene["id"]
        scene_dir.mkdir(parents=True, exist_ok=True)
        dashboard = scene_dir / f"{scene['id']}_multimodality_dashboard.png"
        _plot_dashboard(scene, group, dashboard)
        per_traj = []
        if args.per_traj:
            for index, row in enumerate(group["actual_rows"]):
                path = (
                    scene_dir
                    / f"{scene['id']}_traj_{row['rank']:02d}_sample_{row['sample_index']:02d}.png"
                )
                _plot_single_trajectory(scene, group, index, path)
                per_traj.append(path.name)
        gif_path = None
        if args.gif:
            gif_path = scene_dir / f"{scene['id']}_multimodality.gif"
            _render_gif(scene, group, gif_path)
        manifest["scenes"][scene["id"]] = {
            "npz_path": scene["path"],
            "title": scene["title"],
            "kind": scene["kind"],
            "assets": {
                "dashboard": str(dashboard.relative_to(args.output_dir)),
                "per_trajectory": per_traj,
                "gif": str(gif_path.relative_to(args.output_dir)) if gif_path else None,
            },
            "metrics": group["metrics"],
            "anchor_rows": [_serializable_row(row) for row in group["anchor_rows"]],
            "model_rows": [_serializable_row(row) for row in group["actual_rows"]],
        }
        metrics = group["metrics"]
        print(
            f"{scene['id']}: safe={metrics['safe_candidate_count']}/{metrics['K']} "
            f"reward={metrics['reward_best']:.2f}/{metrics['reward_worst']:.2f} "
            f"endpoint_spread={metrics['endpoint_spread_m']:.2f}m "
            f"shape_spread={metrics['trajectory_shape_spread_m']:.2f}m "
            f"modes={metrics['shape_mode_count_2m']}",
            flush=True,
        )
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
