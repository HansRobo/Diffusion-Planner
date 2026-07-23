#!/usr/bin/env python3
"""Generate publication-quality, data-backed reward explainers for the T4 AWR report.

The renderer deliberately reuses the colleague-derived ``awr_viz`` map, actor,
and rear-axle OBB primitives.  Reward values come from the exact ``hdp_pdm``
configuration used by the full-sequence Original-DP AWR run.  Counterfactual
trajectories are evaluator unit tests, not policy samples; that provenance is
printed on every output.

The stopped-leader scene is mined from the complete paired validation corpus.
It must contain a same-direction, nearly stationary vehicle in the ego lane and
a logged ego future that is itself nearly stationary.  This avoids the old
visual's incorrect interpretation of a perpendicular stopped actor as a leader.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from dataclasses import fields
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle


DEFAULT_CODE_ROOT = Path("/mnt/nvme/wangbin/Diffusion-Planner-t4-main")
DEFAULT_PAIRED = Path(
    "docs/t4_conference_assets/final_validation/data/full_validation/"
    "full_validation_paired.json"
)
DEFAULT_CONFIG = Path(
    "/mnt/nvme/wangbin/Diffusion-Planner-t4-main/outputs/"
    "awr_t4_full_sequence_filtered/"
    "20260717-004516_full_sequence_filtered_hdp_plain_mse_replay_resume_single_checkpoint/"
    "effective_config.json"
)
DEFAULT_OUTPUT = Path("docs/t4_conference_assets/reward_explainers")


COLORS = {
    "hold": "#2563eb",
    "crawl": "#0f9f6e",
    "push": "#dc2626",
    "expert": "#15803d",
    "leader": "#f59e0b",
    "context": "#64748b",
    "ink": "#102a43",
    "muted": "#52677d",
}


def _bootstrap(code_root: Path) -> None:
    sys.path.insert(0, str(code_root))
    sys.path.insert(0, str(code_root / "diffusion_planner"))


def _path_length(xy: np.ndarray) -> float:
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xy[:, :2], axis=0), axis=-1).sum())


def _wrap_angle(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def _valid_pose(row: np.ndarray) -> bool:
    return bool(np.isfinite(row[:3]).all() and np.abs(row[:2]).sum() > 0.1)


def _leader_candidates(scene_path: Path) -> list[dict]:
    """Return strict same-lane stopped-leader candidates from one legacy NPZ."""

    with np.load(scene_path, allow_pickle=False) as z:
        gt = np.asarray(z["ego_agent_future"])
        if _path_length(gt[:, :2]) > 3.0:
            return []
        future = np.asarray(z["neighbor_agents_future"])
        past = np.asarray(z["neighbor_agents_past"])
        ego_shape = np.asarray(z.get("ego_shape", [2.75, 4.34, 1.84])).reshape(-1)
        ego_front = 0.5 * float(ego_shape[0]) + 0.5 * float(ego_shape[1])
        rows: list[dict] = []
        # Legacy contract: aligned[t] = raw[t+1].  The 4 s reward horizon ends
        # at aligned step 39, which corresponds to raw step 40.
        for slot in range(min(len(future), len(past))):
            p0 = future[slot, 1]
            p4 = future[slot, 40]
            current = past[slot, -1]
            if not (_valid_pose(p0) and _valid_pose(p4)):
                continue
            if current.shape[0] >= 11 and int(np.argmax(current[8:11])) != 0:
                continue
            heading = _wrap_angle(float(p0[2]))
            if math.cos(heading) < 0.92:
                continue
            width = float(current[6]) if current.shape[0] > 6 and current[6] > 0.5 else 1.8
            length = float(current[7]) if current.shape[0] > 7 and current[7] > 1.0 else 4.5
            if not (3.5 <= float(p0[0]) <= 25.0):
                continue
            if abs(float(p0[1])) > 0.45 * (float(ego_shape[2]) + width):
                continue
            motion_4s = float(np.linalg.norm(p4[:2] - p0[:2]))
            if motion_4s > 0.8:
                continue
            leader_rear_x = float(p0[0]) - 0.5 * length * math.cos(heading)
            bumper_gap = leader_rear_x - ego_front
            if not (1.5 <= bumper_gap <= 12.0):
                continue
            score = (
                abs(bumper_gap - 5.0)
                + 4.0 * abs(float(p0[1]))
                + 2.0 * abs(heading)
                + 3.0 * motion_4s
            )
            rows.append(
                {
                    "slot": slot,
                    "leader_x_m": float(p0[0]),
                    "leader_y_m": float(p0[1]),
                    "leader_heading_rad": heading,
                    "leader_width_m": width,
                    "leader_length_m": length,
                    "leader_motion_4s_m": motion_4s,
                    "initial_bumper_gap_m": bumper_gap,
                    "logged_path_length_m": _path_length(gt[:, :2]),
                    "selection_cost": score,
                }
            )
        return sorted(rows, key=lambda row: row["selection_cost"])


def _mine_scenes(paired_json: Path, max_scene_reads: int) -> list[tuple[Path, dict]]:
    payload = json.loads(paired_json.read_text())
    rows = payload.get("rows", payload)
    # Candidate ordering is deterministic.  Prefer scenes where both deployed
    # policies are safe and nearly stationary so the visual tests reward
    # semantics, not a checkpoint-specific failure.
    shortlist = [
        row
        for row in rows
        if float(row.get("baseline_path_len", 999.0)) <= 3.0
        and float(row.get("baseline_configured_safe", 0.0)) >= 0.5
        and float(row.get("awr_configured_safe", 0.0)) >= 0.5
    ]
    shortlist.sort(
        key=lambda row: (
            max(float(row.get("baseline_path_len", 999.0)), float(row.get("awr_path_len", 999.0))),
            str(row["scene_path"]),
        )
    )
    matches: list[tuple[float, Path, dict]] = []
    for row in shortlist[:max_scene_reads]:
        path = Path(row["scene_path"])
        if not path.is_file():
            continue
        leaders = _leader_candidates(path)
        if leaders:
            leader = leaders[0]
            # Prefer a readable 3--8 m bumper gap after all strict checks.
            final_cost = float(leader["selection_cost"]) + abs(
                float(row.get("baseline_path_len", 0.0)) - float(row.get("awr_path_len", 0.0))
            )
            matches.append((final_cost, path, leader | {"paired_metrics": row}))
            if len(matches) >= 24:
                break
    if not matches:
        raise RuntimeError(
            f"no strict stopped-leader scene found after {min(len(shortlist), max_scene_reads)} reads"
        )
    return [(path, metadata) for _, path, metadata in sorted(matches, key=lambda item: item[0])]


def _quintic(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def _longitudinal_trajectory(distance_4s: float, total_steps: int = 80) -> np.ndarray:
    t = (np.arange(total_steps, dtype=np.float32) + 1.0) * 0.1
    x = float(distance_4s) * _quintic(t / 4.0)
    trajectory = np.zeros((total_steps, 4), dtype=np.float32)
    trajectory[:, 0] = x
    trajectory[:, 2] = 1.0
    return trajectory


def _candidate_set(meta: dict) -> tuple[list[str], np.ndarray]:
    gap = float(meta["initial_bumper_gap_m"])
    crawl_distance = max(0.35, min(1.5, gap - 3.0))
    # Move through the leader's current OBB during the scored 4 s horizon.
    push_distance = max(6.0, float(meta["leader_x_m"]) + 1.0)
    names = ["Safe hold", "Safe crawl", "Progress push"]
    trajectories = np.stack(
        [
            _longitudinal_trajectory(0.05),
            _longitudinal_trajectory(crawl_distance),
            _longitudinal_trajectory(push_distance),
        ]
    )
    return names, trajectories


def _load_reward_config(path: Path):
    from planner_metrics.config import RewardConfig

    raw = json.loads(path.read_text())
    raw = dict(raw.get("reward", raw.get("reward_config", raw)))
    names = {field.name for field in fields(RewardConfig)}
    return RewardConfig(**{key: value for key, value in raw.items() if key in names})


def _score(scene_path: Path, config_path: Path, trajectories: np.ndarray) -> dict:
    from planner_metrics.aggregate import compute_subscores_batch
    from rlvr.awr import load_scene, reward_compatible_data
    from rlvr.reward import (
        _expert_route_progress_ratio,
        _hdp_multi_reward_components,
        _slice_reward_horizon,
        compute_reward_batch,
    )

    device = torch.device("cpu")
    data = load_scene(scene_path, device)
    reward_data = reward_compatible_data(data)
    config = _load_reward_config(config_path)
    if config.reward_profile != "hdp_pdm":
        raise RuntimeError(f"expected formal hdp_pdm reward, got {config.reward_profile!r}")
    tensor = torch.from_numpy(trajectories).to(device=device, dtype=torch.float32)
    rewards = compute_reward_batch(tensor, reward_data, config)
    horizon = int(config.reward_horizon_steps or tensor.shape[1])
    horizon = min(horizon, tensor.shape[1])
    scored_trajs = tensor[:, :horizon]
    scored_data = _slice_reward_horizon(reward_data, horizon)
    subs = compute_subscores_batch(scored_trajs, scored_data, config)
    ep = _expert_route_progress_ratio(scored_trajs, scored_data.get("ego_agent_future"))
    _, _, lane = _hdp_multi_reward_components(scored_trajs, scored_data, subs, config)
    comfort = torch.exp(
        subs["comfort"].clamp(-50.0, 0.0) / max(float(config.pdm_comfort_scale), 1e-3)
    ) * torch.exp(subs["feasibility"].clamp(-10.0, 0.0))
    result_rows = []
    for index, reward in enumerate(rewards):
        collision_ok = reward.collision_step is None
        dac_ok = not bool(reward.rb_crossing)
        kine_ok = not bool(reward.kinematic_violated)
        result_rows.append(
            {
                "total": float(reward.total),
                "collision_gate": int(collision_ok),
                "dac_gate": int(dac_ok),
                "kinematic_gate": int(kine_ok),
                "collision_step": None if reward.collision_step is None else int(reward.collision_step),
                "ttc": float(subs["ttc"][index]),
                "expert_progress": float(ep[index]),
                "lane": float(lane[index]),
                "comfort": float(comfort[index]),
                "min_ttc_lookahead_clearance_m": float(subs["ttc_min_clearance"][index].min()),
            }
        )
    return {
        "config": config,
        "rows": result_rows,
        "ttc_lookahead_clearance": subs["ttc_min_clearance"].detach().cpu().numpy(),
        "horizon": horizon,
    }


def _draw_box(ax, corners: np.ndarray, edge: str, face: str, alpha: float, lw: float, zorder: int) -> None:
    ax.add_patch(
        Polygon(
            corners,
            closed=True,
            facecolor=face,
            edgecolor=edge,
            linewidth=lw,
            alpha=alpha,
            joinstyle="round",
            zorder=zorder,
        )
    )


def _leader_corners(z: dict, meta: dict, frame: int) -> np.ndarray:
    from planner_metrics.vehicle_collision import obb_corners

    future = z["neighbor_agents_future"]
    pose = future[int(meta["slot"]), min(frame, future.shape[1] - 1)]
    return obb_corners(
        float(pose[0]),
        float(pose[1]),
        float(pose[2]),
        float(meta["leader_length_m"]),
        float(meta["leader_width_m"]),
    )


def _limits(z: dict, meta: dict) -> tuple[float, float, float, float]:
    x_max = max(16.0, float(meta["leader_x_m"]) + 8.0)
    return -5.0, x_max, -8.5, 8.5


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#f5f7fb",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": COLORS["muted"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "font.size": 10,
            "axes.titleweight": "bold",
        }
    )


def _draw_close_scene(ax, z: dict, meta: dict, trajectories: np.ndarray, frame: int = 39) -> None:
    from rlvr.autoresearch.tools.awr_viz import _draw_map, _draw_neighbor_boxes, _ego_corners

    _draw_map(ax, z)
    _draw_neighbor_boxes(ax, z, frame=frame, max_neighbors=16, trails=True, focus_xy=np.array([5.0, 0.0]))
    _draw_box(ax, _leader_corners(z, meta, frame), COLORS["leader"], "#fef3c7", 0.95, 2.6, 28)
    shape = np.asarray(z["ego_shape"]).reshape(-1)
    for index, (label, color) in enumerate(
        [("Safe hold", COLORS["hold"]), ("Safe crawl", COLORS["crawl"]), ("Progress push", COLORS["push"])]
    ):
        trajectory = trajectories[index]
        ax.plot(trajectory[: frame + 1, 0], trajectory[: frame + 1, 1], color=color, lw=3.0, label=label, zorder=15)
        for key in sorted(set([0, 19, frame])):
            _draw_box(
                ax,
                _ego_corners(trajectory, shape, key),
                color,
                color,
                0.08 if key != frame else 0.18,
                1.2 if key != frame else 2.0,
                19 + index,
            )
    current = np.zeros((80, 4), dtype=np.float32)
    current[:, 2] = 1.0
    _draw_box(ax, _ego_corners(current, shape, 0), "#111827", "#e2e8f0", 0.92, 2.0, 30)
    ax.annotate(
        "EGO now\nrear-axle reference",
        xy=(1.2, 0.0),
        xytext=(-3.8, 5.6),
        arrowprops={"arrowstyle": "->", "color": "#111827", "lw": 1.2},
        fontsize=9,
        fontweight="bold",
        color="#111827",
    )
    ax.annotate(
        f"STOPPED LEADER\nslot {int(meta['slot'])} · moved {meta['leader_motion_4s_m']:.2f} m / 4 s",
        xy=(float(meta["leader_x_m"]), float(meta["leader_y_m"])),
        xytext=(float(meta["leader_x_m"]) + 2.6, 5.8),
        arrowprops={"arrowstyle": "->", "color": COLORS["leader"], "lw": 1.5},
        fontsize=9,
        fontweight="bold",
        color="#92400e",
    )
    x0 = 0.5 * float(shape[0]) + 0.5 * float(shape[1])
    x1 = x0 + float(meta["initial_bumper_gap_m"])
    y_gap = -3.1
    ax.annotate("", xy=(x1, y_gap), xytext=(x0, y_gap), arrowprops={"arrowstyle": "<->", "color": "#7c3aed", "lw": 2.0})
    ax.text(
        0.5 * (x0 + x1),
        y_gap - 0.5,
        f"initial bumper gap = {meta['initial_bumper_gap_m']:.1f} m",
        ha="center",
        va="top",
        color="#6d28d9",
        fontsize=9,
        fontweight="bold",
    )
    limits = _limits(z, meta)
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.12)
    ax.set_xlabel("ego-frame x (m) · forward →")
    ax.set_ylabel("ego-frame y (m) · lateral")
    ax.set_title(
        f"1 · Decision frame at t={0.1 * frame:.1f}s: unsafe push overlaps the stopped leader",
        loc="left",
        fontsize=12,
    )
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.94)


def _draw_formula(ax) -> None:
    ax.axis("off")
    ax.set_title("2 · The exact formal reward used in training", loc="left", fontsize=12)
    box = FancyBboxPatch(
        (0.02, 0.46),
        0.96,
        0.43,
        boxstyle="round,pad=0.025,rounding_size=0.025",
        transform=ax.transAxes,
        facecolor="#eef6ff",
        edgecolor="#93c5fd",
        linewidth=1.3,
    )
    ax.add_patch(box)
    ax.text(
        0.50,
        0.75,
        "R = Collision gate × DAC road-border gate × Kinematic gate",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#1e3a8a",
    )
    ax.text(
        0.50,
        0.59,
        "×  (5·TTC  +  5·Expert-route Progress  +  2·Lane  +  2·Comfort) / 14",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#1e3a8a",
    )
    ax.text(
        0.04,
        0.35,
        "How to read this stopped scene",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.04,
        0.25,
        "• The logged expert travels < 5 m, so EP is intentionally neutral (=1); the reward does not demand motion.",
        transform=ax.transAxes,
        fontsize=9.2,
        color=COLORS["muted"],
    )
    ax.text(
        0.04,
        0.15,
        "• Hold and crawl are compared by TTC / lane / comfort. A collision makes the total exactly zero.",
        transform=ax.transAxes,
        fontsize=9.2,
        color=COLORS["muted"],
    )
    ax.text(
        0.04,
        0.05,
        "• Therefore the lesson is ‘safe progress when possible’, not ‘always move forward’.",
        transform=ax.transAxes,
        fontsize=9.2,
        color="#7c2d12",
        fontweight="bold",
    )


def _draw_score_table(ax, names: list[str], scores: dict) -> None:
    ax.axis("off")
    ax.set_title("3 · Exact evaluator output for the three actions", loc="left", fontsize=12)
    table_rows = []
    for name, row in zip(names, scores["rows"]):
        gate = "PASS" if row["collision_gate"] and row["dac_gate"] and row["kinematic_gate"] else "FAIL"
        table_rows.append(
            [
                name,
                gate,
                f"{row['ttc']:.2f}",
                f"{row['expert_progress']:.2f}",
                f"{row['lane']:.2f}",
                f"{row['comfort']:.2f}",
                f"{row['total']:.3f}",
            ]
        )
    table = ax.table(
        cellText=table_rows,
        colLabels=["Action", "hard gates", "TTC", "EP", "Lane", "Comfort", "Total"],
        cellLoc="center",
        colLoc="center",
        bbox=[0.01, 0.20, 0.98, 0.70],
        colWidths=[0.24, 0.14, 0.11, 0.10, 0.11, 0.13, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d6e0ea")
        if row == 0:
            cell.set_facecolor("#eaf1f8")
            cell.set_text_props(weight="bold", color=COLORS["ink"])
        elif col == 0:
            cell.set_text_props(weight="bold")
        text = cell.get_text().get_text()
        if text == "PASS":
            cell.set_facecolor("#dcfce7")
            cell.set_text_props(color="#166534", weight="bold")
        elif text == "FAIL":
            cell.set_facecolor("#fee2e2")
            cell.set_text_props(color="#991b1b", weight="bold")
    push = scores["rows"][2]
    event = "collision" if push["collision_step"] is not None else "terminal gate"
    event_time = (
        f" at t={0.1 * push['collision_step']:.1f}s" if push["collision_step"] is not None else ""
    )
    ax.text(
        0.02,
        0.07,
        f"Progress push: {event}{event_time} → hard gate = 0 → total reward = {push['total']:.3f}.",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#991b1b",
        fontweight="bold",
    )


def _draw_temporal(ax, names: list[str], scores: dict) -> None:
    # Keep the renderer/evaluator convention used throughout the colleague
    # tooling: array index k is displayed as k * dt.
    t = np.arange(scores["horizon"]) * 0.1
    for index, (name, color) in enumerate(
        zip(names, [COLORS["hold"], COLORS["crawl"], COLORS["push"]])
    ):
        values = scores["ttc_lookahead_clearance"][index]
        values = np.where(np.abs(values) > 1e4, np.nan, values)
        ax.plot(t, values, color=color, lw=2.3, label=name)
    ax.axhline(0.0, color="#991b1b", ls="--", lw=1.2, label="OBB overlap boundary")
    collision_step = scores["rows"][2]["collision_step"]
    if collision_step is not None:
        collision_time = 0.1 * int(collision_step)
        ax.axvline(collision_time, color="#991b1b", ls=":", lw=1.5)
        ax.text(
            collision_time + 0.04,
            0.04,
            f"actual overlap\n{collision_time:.1f}s",
            color="#991b1b",
            fontsize=8.2,
            fontweight="bold",
            va="bottom",
        )
    ax.fill_between(t, -3.0, 0.0, color="#fee2e2", alpha=0.42)
    ax.set_title("4 · TTC look-ahead: minimum signed OBB clearance", loc="left", fontsize=12)
    ax.set_xlabel("reward-horizon time (s)")
    ax.set_ylabel("look-ahead OBB clearance (m)\nnegative = overlap")
    ax.grid(alpha=0.18)
    ax.legend(loc="best", fontsize=8)


def _render_static(
    scene_path: Path,
    z: dict,
    meta: dict,
    names: list[str],
    trajectories: np.ndarray,
    scores: dict,
    output: Path,
) -> None:
    fig = plt.figure(figsize=(18, 11), facecolor="#f5f7fb")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.12, 1.0], height_ratios=[1.0, 1.0], hspace=0.22, wspace=0.15)
    ax_scene = fig.add_subplot(grid[:, 0])
    right_top = grid[0, 1].subgridspec(2, 1, height_ratios=[0.48, 0.52], hspace=0.12)
    ax_formula = fig.add_subplot(right_top[0, 0])
    ax_table = fig.add_subplot(right_top[1, 0])
    ax_temporal = fig.add_subplot(grid[1, 1])
    event_frame = scores["rows"][2]["collision_step"]
    if event_frame is None:
        event_frame = scores["horizon"] - 1
    _draw_close_scene(ax_scene, z, meta, trajectories, frame=int(event_frame))
    _draw_formula(ax_formula)
    _draw_score_table(ax_table, names, scores)
    _draw_temporal(ax_temporal, names, scores)
    fig.suptitle(
        "STOPPED-LEADER REWARD EXPLAINER · real T4 scene · formal HDP-PDM evaluator",
        x=0.04,
        y=0.992,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.04,
        0.018,
        f"COUNTERFACTUAL EVALUATOR UNIT TEST — NOT POLICY OUTPUT. Scene: {scene_path.name}. "
        "Map and recorded actors are real; three ego actions are deliberately constructed. "
        "Neighbor future uses the required raw[t+1] → aligned[t] correction; ego OBB uses rear axle + scene wheelbase.",
        fontsize=8.4,
        color=COLORS["muted"],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _render_gif(
    scene_path: Path,
    z: dict,
    meta: dict,
    names: list[str],
    trajectories: np.ndarray,
    scores: dict,
    output: Path,
    fps: int,
) -> None:
    from rlvr.autoresearch.tools.awr_viz import _draw_map, _draw_neighbor_boxes, _ego_corners

    shape = np.asarray(z["ego_shape"]).reshape(-1)
    horizon = scores["horizon"]
    limits = _limits(z, meta)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4), facecolor="#f5f7fb")
    timeline = Rectangle((0.08, 0.02), 0.0, 0.012, transform=fig.transFigure, facecolor="#2563eb", edgecolor="none")
    fig.add_artist(Rectangle((0.08, 0.02), 0.84, 0.012, transform=fig.transFigure, facecolor="#dbe5ef", edgecolor="#94a3b8", lw=0.6))
    fig.add_artist(timeline)

    def draw(frame: int):
        timeline.set_width(0.84 * (frame + 1) / horizon)
        for index, (ax, name, color) in enumerate(
            zip(axes, names, [COLORS["hold"], COLORS["crawl"], COLORS["push"]])
        ):
            ax.clear()
            _draw_map(ax, z)
            trajectory = trajectories[index]
            _draw_neighbor_boxes(
                ax,
                z,
                frame=frame,
                max_neighbors=12,
                trails=False,
                focus_xy=trajectory[frame, :2],
            )
            _draw_box(ax, _leader_corners(z, meta, frame), COLORS["leader"], "#fef3c7", 0.95, 2.5, 28)
            ax.plot(trajectory[: frame + 1, 0], trajectory[: frame + 1, 1], color=color, lw=3.0)
            _draw_box(ax, _ego_corners(trajectory, shape, frame), color, color, 0.24, 2.0, 30)
            row = scores["rows"][index]
            failed = row["collision_step"] is not None and frame >= int(row["collision_step"])
            if failed:
                ax.scatter(trajectory[frame, 0], trajectory[frame, 1], marker="X", s=180, color="#991b1b", edgecolor="white", lw=1.2, zorder=40)
            ax.set_xlim(limits[0], limits[1])
            ax.set_ylim(limits[2], limits[3])
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.12)
            if row["collision_step"] is not None and not failed:
                future_time = 0.1 * int(row["collision_step"])
                gate = f"future collision at {future_time:.1f}s → final total 0"
            else:
                gate = "COLLISION → final total 0" if failed else "gate PASS through horizon"
            ax.set_title(
                f"{name} · t={0.1 * frame:.1f}s\n{gate} · final reward={row['total']:.3f}",
                loc="left",
                fontsize=10.5,
                color="#991b1b" if failed else color,
            )
            ax.set_xlabel("x forward (m)")
            if index == 0:
                ax.set_ylabel("y lateral (m)")
        fig.suptitle(
            "REAL T4 ACTORS + THREE COUNTERFACTUAL EGO ACTIONS · formal 4 s reward horizon",
            fontsize=14,
            fontweight="bold",
            color=COLORS["ink"],
            y=0.99,
        )
        fig.text(
            0.5,
            0.045,
            "Orange = recorded stopped leader OBB · colored box = candidate ego OBB · unit test, not policy samples",
            ha="center",
            fontsize=8.5,
            color=COLORS["muted"],
        )
        return [*axes, timeline]

    movie = animation.FuncAnimation(fig, draw, frames=horizon, interval=1000 / max(fps, 1), blit=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    movie.save(output, writer=animation.PillowWriter(fps=fps), dpi=105)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, default=DEFAULT_CODE_ROOT)
    parser.add_argument("--paired-json", type=Path, default=DEFAULT_PAIRED)
    parser.add_argument("--reward-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scene", type=Path, help="optional audited scene; otherwise mine from the full validation corpus")
    parser.add_argument("--leader-slot", type=int, help="required only with --scene when auto-detection is ambiguous")
    parser.add_argument("--max-scene-reads", type=int, default=5000)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    _bootstrap(args.code_root.resolve())
    _style()
    scores = None
    trajectories = None
    names = None
    if args.scene is None:
        # Geometry first, exact reward second.  Prefer a scene where the unsafe
        # forward action fails specifically because of the leader collision,
        # while DAC and kinematics remain valid.  That isolates TTC/collision
        # from road-border semantics in the explainer.
        ranked = []
        for candidate_path, candidate_meta in _mine_scenes(
            args.paired_json.resolve(), int(args.max_scene_reads)
        ):
            candidate_names, candidate_trajectories = _candidate_set(candidate_meta)
            candidate_scores = _score(
                candidate_path, args.reward_config.resolve(), candidate_trajectories
            )
            rows = candidate_scores["rows"]
            isolated = (
                all(row["collision_gate"] and row["dac_gate"] and row["kinematic_gate"] for row in rows[:2])
                and not rows[2]["collision_gate"]
                and rows[2]["dac_gate"]
                and rows[2]["kinematic_gate"]
            )
            failure_count = (
                int(rows[2]["collision_gate"])
                + int(not rows[2]["dac_gate"])
                + int(not rows[2]["kinematic_gate"])
            )
            rank_key = (
                0 if isolated else 1,
                failure_count,
                abs(float(candidate_meta["initial_bumper_gap_m"]) - 5.0),
                float(candidate_meta["selection_cost"]),
            )
            ranked.append(
                (
                    rank_key,
                    candidate_path,
                    candidate_meta,
                    candidate_names,
                    candidate_trajectories,
                    candidate_scores,
                )
            )
        _, scene_path, meta, names, trajectories, scores = min(ranked, key=lambda item: item[0])
    else:
        scene_path = args.scene.resolve()
        matches = _leader_candidates(scene_path)
        if args.leader_slot is not None:
            matches = [row for row in matches if int(row["slot"]) == int(args.leader_slot)]
        if len(matches) != 1:
            raise RuntimeError(f"expected one strict leader for {scene_path}, got {len(matches)}")
        meta = matches[0]

    from rlvr.autoresearch.tools.awr_viz import _load_aligned_npz

    if names is None or trajectories is None or scores is None:
        names, trajectories = _candidate_set(meta)
        scores = _score(scene_path, args.reward_config.resolve(), trajectories)
    z = _load_aligned_npz(scene_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    png = args.output_dir / "stopped_leader_formal_reward_explainer.png"
    gif = args.output_dir / "stopped_leader_formal_reward_explainer.gif"
    _render_static(scene_path, z, meta, names, trajectories, scores, png)
    _render_gif(scene_path, z, meta, names, trajectories, scores, gif, int(args.fps))

    manifest = {
        "contract": {
            "evidence": "real T4 scene geometry plus constructed counterfactual ego actions",
            "policy_output": False,
            "purpose": "formal evaluator unit test",
            "reward_config": str(args.reward_config.resolve()),
            "reward_profile": "hdp_pdm",
            "neighbor_future_alignment": "raw[t+1] -> aligned[t]",
            "ego_obb": "rear axle + 0.5 * scene wheelbase",
        },
        "scene": str(scene_path),
        "leader": {key: value for key, value in meta.items() if key != "paired_metrics"},
        "paired_metrics": meta.get("paired_metrics"),
        "candidates": [name | row if isinstance(name, dict) else {"name": name, **row} for name, row in zip(names, scores["rows"])],
        "assets": {"png": png.name, "gif": gif.name},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"scene": str(scene_path), "leader": meta, "scores": scores["rows"], "png": str(png), "gif": str(gif)}, indent=2))


if __name__ == "__main__":
    main()
