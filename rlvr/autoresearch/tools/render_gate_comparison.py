#!/usr/bin/env python3
"""Render a paired source original-DP vs deterministic AWR checkpoint comparison.

Unlike the generic candidate gallery, this figure always draws the exact
deterministic source trajectory and the exact deterministic AWR checkpoint trajectory.
The event-time neighbor OBBs are repeated in the panel so a collision or
road-border interaction is visible rather than hidden in a metric table.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import fields
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Polygon, Rectangle

from planner_metrics.aggregate import compute_subscores_batch
from planner_metrics.config import RewardConfig
from planner_metrics.vehicle_collision import obb_corners
from rlvr.awr import load_scene, reward_compatible_data
from rlvr.reward import compute_reward_batch, reward_breakdown_to_json_dict
from rlvr.autoresearch.tools.awr_viz import (
    _all_points,
    _draw_map,
    _draw_neighbor_boxes,
    _ego_corners,
    _load_aligned_npz,
    _valid_xy,
)
from rlvr.train_awr import _trajectory_metrics


def _load_config(path: Path) -> RewardConfig:
    raw = json.loads(path.read_text())
    raw = dict(raw.get("reward", raw.get("reward_config", raw)))
    names = {field.name for field in fields(RewardConfig)}
    return RewardConfig(**{key: value for key, value in raw.items() if key in names})


def _load_selection_row(path: Path | None, scene: Path) -> dict | None:
    """Resolve the visual-audit role for one exact scene, failing closed."""

    if path is None:
        return None
    payload = json.loads(path.read_text())
    rows = payload.get("scenes", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"selection metadata has no scene list: {path}")
    scene_value = str(scene.resolve())
    matches = [row for row in rows if str(Path(row["scene_path"]).resolve()) == scene_value]
    if len(matches) != 1:
        raise RuntimeError(
            f"selection metadata must match exactly one row for {scene}: got {len(matches)}"
        )
    return matches[0]


def _promotion_label(status: str) -> str:
    return {
        "strict_promotion_passed": "PAPER-ALIGNED MEAN SCORE PASSED",
        "not_promoted_failure_audit": "NOT PROMOTED · MEAN SCORE AUDIT",
        "aggregate_promotion_passed": "PAPER-ALIGNED MEAN SCORE PASSED",
        "not_promoted_no_mean_gain": "NOT PROMOTED · MEAN GAIN UNPROVEN",
        "unspecified": "PROMOTION STATUS UNSPECIFIED",
    }[status]


def _draw_box(ax, corners: np.ndarray, color: str, alpha: float, lw: float = 1.0, zorder: int = 20):
    ax.add_patch(
        Polygon(
            corners,
            closed=True,
            facecolor=color,
            edgecolor=color,
            linewidth=lw,
            alpha=alpha,
            joinstyle="round",
            zorder=zorder,
        )
    )


def _first_rb_step(subscores: dict[str, object], index: int, horizon: int) -> int | None:
    values = subscores.get("rb_crossing_steps")
    if values is None:
        return None
    try:
        value = values[index]
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return int(value.item())
            value = value.detach().cpu().numpy()
        if value is None:
            return None
        array = np.asarray(value)
        if array.ndim == 0:
            return int(array.item())
        # Compatibility with older artifacts that stored a per-timestep mask
        # rather than the current list[int | None] first-step contract.
        positions = np.flatnonzero(array.astype(bool).reshape(-1)[:horizon])
        return int(positions[0]) if len(positions) else None
    except (IndexError, TypeError, ValueError):
        return None


def _policy_event_step(reward, subscores: dict[str, object], index: int, horizon: int) -> tuple[int, str] | None:
    if reward.collision_step is not None:
        return int(reward.collision_step), "collision event"
    rb = _first_rb_step(subscores, index, horizon)
    if rb is not None:
        return rb, "road-border event"
    if reward.kinematic_violated:
        return min(horizon - 1, 40), "kinematic reference"
    return None


def _paired_event_step(
    base_reward,
    awr_reward,
    base_subscores: dict[str, object],
    awr_subscores: dict[str, object],
    base: np.ndarray,
    awr: np.ndarray,
    horizon: int,
    awr_index: int = 0,
) -> tuple[int, str]:
    """Choose the frame that best explains either a recovery or a regression."""
    base_event = _policy_event_step(base_reward, base_subscores, 0, horizon)
    awr_event = _policy_event_step(awr_reward, awr_subscores, awr_index, horizon)
    if base_event is not None:
        return base_event[0], f"source {base_event[1]}"
    if awr_event is not None:
        return awr_event[0], f"AWR {awr_event[1]}"
    n = min(horizon, len(base), len(awr))
    if n > 0:
        separation = np.linalg.norm(base[:n, :2] - awr[:n, :2], axis=-1)
        step = int(np.nanargmax(np.nan_to_num(separation, nan=-1.0)))
        return step, "maximum policy divergence"
    return 0, "comparison reference"


def _draw_expert(ax, z: dict[str, np.ndarray]) -> None:
    gt = z.get("ego_agent_future")
    if gt is not None:
        valid = _valid_xy(gt[:, :2])
        if valid.sum() >= 2:
            ax.plot(
                gt[valid, 0],
                gt[valid, 1],
                "--",
                color="#15803d",
                lw=2.2,
                alpha=0.85,
                label="logged expert",
                zorder=7,
            )


def _draw_panel(
    ax,
    z: dict[str, np.ndarray],
    base: np.ndarray,
    repair: np.ndarray,
    base_reward,
    repair_reward,
    base_event_step: int,
    repair_index: int,
    primary_policy: str,
    title: str,
    event_label: str,
    limits: tuple[float, float, float, float],
    max_neighbors: int,
) -> None:
    _draw_map(ax, z)
    _draw_expert(ax, z)
    _draw_neighbor_boxes(ax, z, frame=0, max_neighbors=max_neighbors, trails=True)
    # Context samples remain faint.  The panel's named policy is thick while
    # the counterpart is a dashed reference, so the two sides are genuinely
    # different views rather than duplicated overlays.
    for trajectory in base[1:]:
        ax.plot(trajectory[:, 0], trajectory[:, 1], color="#94a3b8", lw=0.55, alpha=0.18, zorder=5)
    for trajectory in repair:
        ax.plot(trajectory[:, 0], trajectory[:, 1], color="#86efac", lw=0.55, alpha=0.12, zorder=4)
    selected = repair[repair_index]
    base_primary = primary_policy == "source"
    ax.plot(
        base[0, :, 0], base[0, :, 1],
        "-" if base_primary else "--",
        color="#b91c1c", lw=3.4 if base_primary else 1.7,
        alpha=0.98 if base_primary else 0.42,
        label="source original DP" if base_primary else "source reference",
        zorder=16 if base_primary else 12,
    )
    ax.plot(
        selected[:, 0], selected[:, 1],
        "-" if not base_primary else "--",
        color="#047857", lw=3.4 if not base_primary else 1.7,
        alpha=0.98 if not base_primary else 0.42,
        label="AWR checkpoint" if not base_primary else "AWR reference",
        zorder=16 if not base_primary else 12,
    )
    shape = z.get("ego_shape", np.asarray([2.75, 4.34, 1.70], dtype=np.float32))
    if shape.ndim > 1:
        shape = shape[0]
    for frame in sorted(set([0, 20, 40, 60, 79, base_event_step])):
        if frame >= len(selected):
            continue
        _draw_box(ax, _ego_corners(base[0], shape, frame), "#b91c1c", 0.12 if frame else 0.24, 1.2, 17)
        _draw_box(ax, _ego_corners(selected, shape, frame), "#047857", 0.12 if frame else 0.24, 1.2, 18)
    # Repeat the event-time neighbor boxes in stronger color.  This is the
    # frame at which the metric is decided, not only the t=0 observation.
    event_focus = 0.5 * (base[0, base_event_step, :2] + selected[base_event_step, :2])
    _draw_neighbor_boxes(
        ax, z, frame=base_event_step, max_neighbors=min(max_neighbors, 16),
        trails=False, focus_xy=event_focus,
    )
    if base_reward.collision_step is not None:
        hit_step = min(int(base_reward.collision_step), len(base[0]) - 1)
        hit = base[0, hit_step]
        ax.scatter(hit[0], hit[1], marker="X", s=170, color="#dc2626", edgecolors="white", linewidths=1.0, zorder=30, label="base collision")
    elif base_reward.rb_crossing and event_label.startswith("source"):
        hit = base[0, base_event_step]
        ax.scatter(hit[0], hit[1], marker="v", s=150, color="#dc2626", edgecolors="white", linewidths=1.0, zorder=30, label="base road-border crossing")
    elif base_reward.kinematic_violated:
        hit = base[0, base_event_step]
        ax.scatter(hit[0], hit[1], marker="P", s=150, color="#7c3aed", edgecolors="white", linewidths=1.0, zorder=30, label="base kinematic violation")
    if repair_reward.collision_step is not None:
        hit_step = min(int(repair_reward.collision_step), len(selected) - 1)
        hit = selected[hit_step]
        ax.scatter(hit[0], hit[1], marker="X", s=170, color="#be123c", edgecolors="white", linewidths=1.0, zorder=31, label="AWR collision")
    elif repair_reward.rb_crossing and event_label.startswith("AWR"):
        hit = selected[base_event_step]
        ax.scatter(hit[0], hit[1], marker="v", s=150, color="#be123c", edgecolors="white", linewidths=1.0, zorder=31, label="AWR road-border crossing")
    elif repair_reward.kinematic_violated and event_label.startswith("AWR"):
        hit = selected[base_event_step]
        ax.scatter(hit[0], hit[1], marker="P", s=150, color="#7c3aed", edgecolors="white", linewidths=1.0, zorder=31, label="AWR kinematic violation")
    ax.scatter(selected[-1, 0], selected[-1, 1], marker="o", s=42, color="#047857", edgecolors="white", linewidths=0.8, zorder=30)
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.16)
    ax.set_xlabel("ego-frame x (m)")
    ax.set_ylabel("ego-frame y (m)")
    ax.set_title(title, fontsize=11, loc="left")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.92, ncol=2)
    zoom = ax.inset_axes([0.63, 0.63, 0.34, 0.32], zorder=40)
    _draw_map(zoom, z)
    _draw_neighbor_boxes(
        zoom, z, frame=base_event_step, max_neighbors=min(max_neighbors, 12),
        trails=False, focus_xy=event_focus,
    )
    zoom.plot(base[0, :, 0], base[0, :, 1], color="#b91c1c", lw=1.8, alpha=0.9)
    zoom.plot(selected[:, 0], selected[:, 1], color="#047857", lw=1.8, alpha=0.95)
    _draw_box(zoom, _ego_corners(base[0], shape, base_event_step), "#b91c1c", 0.24, 1.0, 20)
    _draw_box(zoom, _ego_corners(selected, shape, base_event_step), "#047857", 0.24, 1.0, 21)
    ax_min = np.array([base[0, base_event_step, :2], selected[base_event_step, :2]])
    center = np.mean(ax_min, axis=0)
    half = 9.0
    zoom.set_xlim(center[0] - half, center[0] + half)
    zoom.set_ylim(center[1] - half, center[1] + half)
    zoom.set_aspect("equal", adjustable="box")
    zoom.set_xticks([])
    zoom.set_yticks([])
    zoom.set_title(f"{event_label} t={base_event_step * 0.1:.1f}s · OBB zoom", fontsize=7, pad=2)


def _metric_text(base_reward, repair_reward, base_metrics: dict[str, float], repair_metrics: dict[str, float]) -> str:
    base_collision = "yes" if base_reward.collision_step is not None else "no"
    repair_collision = "yes" if repair_reward.collision_step is not None else "no"
    base_rb = "yes" if base_reward.rb_crossing else "no"
    repair_rb = "yes" if repair_reward.rb_crossing else "no"
    return (
        f"BASE deterministic: reward={base_reward.total:+.3f}  collision={base_collision}  road-border={base_rb}  "
        f"ADE={base_metrics.get('ade', float('nan')):.2f}m  FDE={base_metrics.get('fde', float('nan')):.2f}m\n"
        f"AWR deterministic checkpoint: reward={repair_reward.total:+.3f}  collision={repair_collision}  road-border={repair_rb}  "
        f"ADE={repair_metrics.get('ade', float('nan')):.2f}m  FDE={repair_metrics.get('fde', float('nan')):.2f}m  "
        "(deployment K=1, zero noise)"
    )


def _evidence_label(base_reward, awr_reward, base_metrics: dict[str, float], awr_metrics: dict[str, float]) -> str:
    if base_reward.collision_step is not None and awr_reward.collision_step is None:
        return "COLLISION RECOVERED"
    if base_reward.rb_crossing and not awr_reward.rb_crossing:
        return "ROAD-BORDER RECOVERED"
    if base_reward.collision_step is None and awr_reward.collision_step is not None:
        return "NEW COLLISION REGRESSION"
    if not base_reward.rb_crossing and awr_reward.rb_crossing:
        return "NEW ROAD-BORDER REGRESSION"
    reward_delta = float(awr_reward.total) - float(base_reward.total)
    ade_delta = float(awr_metrics.get("ade", float("nan"))) - float(base_metrics.get("ade", float("nan")))
    if reward_delta > 0.0 and math.isfinite(ade_delta) and ade_delta <= 0.0:
        return "REWARD + IMITATION PARETO GAIN"
    if reward_delta > 0.0:
        return "REWARD GAIN / RETENTION AUDIT"
    if math.isfinite(ade_delta) and ade_delta < 0.0:
        return "IMITATION GAIN / REWARD TRADE-OFF"
    return "REGRESSION / FAILURE AUDIT"


def _draw_metric_table(ax, base_reward, awr_reward, base_metrics: dict[str, float], awr_metrics: dict[str, float]) -> None:
    ax.axis("off")

    def flag(value: bool) -> str:
        return "FAIL" if value else "PASS"

    def numeric(before: float, after: float, precision: int = 3) -> tuple[str, str, str]:
        if not (math.isfinite(float(before)) and math.isfinite(float(after))):
            return "n/a", "n/a", "n/a"
        fmt = f"{{:.{precision}f}}"
        return fmt.format(before), fmt.format(after), f"{after - before:+.{precision}f}"

    rows: list[tuple[str, str, str, str]] = []
    for label, before, after, precision in (
        ("Total reward ↑", base_reward.total, awr_reward.total, 3),
        ("ADE (m) ↓", base_metrics.get("ade", float("nan")), awr_metrics.get("ade", float("nan")), 2),
        ("FDE (m) ↓", base_metrics.get("fde", float("nan")), awr_metrics.get("fde", float("nan")), 2),
        ("Progress ↑", base_reward.progress, awr_reward.progress, 3),
        ("Lane robustness ↑", base_reward.centerline, awr_reward.centerline, 3),
        ("Comfort ↑", base_reward.smoothness, awr_reward.smoothness, 3),
        ("RB clearance (m) ↑", base_reward.rb_min_dist, awr_reward.rb_min_dist, 2),
    ):
        before_text, after_text, delta_text = numeric(float(before), float(after), precision)
        rows.append((label, before_text, after_text, delta_text))
    rows.extend(
        [
            ("Collision OBB", flag(base_reward.collision_step is not None), flag(awr_reward.collision_step is not None), "hard gate"),
            ("Road-border OBB", flag(bool(base_reward.rb_crossing)), flag(bool(awr_reward.rb_crossing)), "hard gate"),
            ("Kinematic", flag(bool(base_reward.kinematic_violated)), flag(bool(awr_reward.kinematic_violated)), "hard gate"),
        ]
    )
    table = ax.table(
        cellText=[row[1:] for row in rows],
        rowLabels=[row[0] for row in rows],
        colLabels=["Source", "AWR", "AWR − source"],
        cellLoc="center",
        rowLoc="left",
        bbox=[0.02, 0.02, 0.96, 0.90],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d6e0ea")
        if row == 0:
            cell.set_facecolor("#eaf1f8")
            cell.set_text_props(weight="bold", color="#19324a")
        elif col == -1:
            cell.set_facecolor("#f8fafc")
            cell.set_text_props(weight="bold", color="#334e68")
        elif "FAIL" in cell.get_text().get_text():
            cell.set_facecolor("#fee2e2")
            cell.set_text_props(color="#991b1b", weight="bold")
        elif "PASS" in cell.get_text().get_text():
            cell.set_facecolor("#dcfce7")
            cell.set_text_props(color="#166534", weight="bold")
    ax.set_title("Exact per-trajectory evaluator output", loc="left", fontsize=11, fontweight="bold")


def _draw_temporal_diagnostics(ax_distance, ax_speed, z: dict[str, np.ndarray], base: np.ndarray, awr: np.ndarray) -> None:
    n = min(len(base), len(awr))
    t = (np.arange(n) + 1) * 0.1
    separation = np.linalg.norm(base[:n, :2] - awr[:n, :2], axis=-1)
    ax_distance.plot(t, separation, color="#2563eb", lw=2.2, label="source ↔ AWR")
    expert = z.get("ego_agent_future")
    if expert is not None:
        expert = np.asarray(expert)[:n, :2]
        valid = _valid_xy(expert)
        base_error = np.full(n, np.nan)
        awr_error = np.full(n, np.nan)
        base_error[valid] = np.linalg.norm(base[:n, :2][valid] - expert[valid], axis=-1)
        awr_error[valid] = np.linalg.norm(awr[:n, :2][valid] - expert[valid], axis=-1)
        ax_distance.plot(t, base_error, color="#b91c1c", lw=1.5, alpha=0.8, label="source → expert")
        ax_distance.plot(t, awr_error, color="#047857", lw=1.5, alpha=0.85, label="AWR → expert")
    ax_distance.axvspan(0.1, min(4.0, float(t[-1])), color="#dbeafe", alpha=0.22, label="reward horizon")
    ax_distance.set_ylabel("distance (m)")
    ax_distance.set_title("Where the policies diverge", loc="left", fontsize=10, fontweight="bold")
    ax_distance.grid(alpha=0.2)
    ax_distance.legend(fontsize=7, ncol=2, loc="upper left")

    def speed(path: np.ndarray) -> np.ndarray:
        xy = path[:n, :2]
        origin = np.zeros((1, 2), dtype=xy.dtype)
        return np.linalg.norm(np.diff(np.concatenate([origin, xy], axis=0), axis=0), axis=-1) / 0.1

    ax_speed.plot(t, speed(base), color="#b91c1c", lw=1.8, label="source")
    ax_speed.plot(t, speed(awr), color="#047857", lw=1.8, label="AWR")
    if expert is not None:
        expert_speed = speed(np.concatenate([expert, np.zeros((0, 2))], axis=0))
        expert_speed[~valid] = np.nan
        ax_speed.plot(t, expert_speed, "--", color="#15803d", lw=1.3, alpha=0.7, label="logged expert")
    ax_speed.axvline(4.0, color="#64748b", lw=1.0, ls=":")
    ax_speed.set_xlabel("future time (s)")
    ax_speed.set_ylabel("speed (m/s)")
    ax_speed.set_title("Longitudinal response", loc="left", fontsize=10, fontweight="bold")
    ax_speed.grid(alpha=0.2)
    ax_speed.legend(fontsize=7, ncol=3, loc="upper right")


def _animate(
    scene: str,
    z: dict[str, np.ndarray],
    base: np.ndarray,
    repair: np.ndarray,
    repair_index: int,
    output: Path,
    base_event_step: int,
    event_label: str,
    max_neighbors: int,
    fps: int,
    max_frames: int,
    limits: tuple[float, float, float, float],
    selection_kind: str | None,
    promotion_status: str,
) -> None:
    shape = z.get("ego_shape", np.asarray([2.75, 4.34, 1.70], dtype=np.float32))
    if shape.ndim > 1:
        shape = shape[0]
    n = min(base.shape[1], repair.shape[1])
    frames = np.linspace(0, n - 1, min(max_frames, n)).round().astype(int)
    fig, axes = plt.subplots(
        1, 3, figsize=(20, 6.8), facecolor="white",
        gridspec_kw={"width_ratios": [1.0, 1.0, 0.82]},
    )
    # A real timeline, not a decorative changing timestamp, keeps stationary
    # stop/yield scenes temporally legible.  The event marker is fixed while
    # the blue bar advances through all 80 planner steps.
    timeline_left, timeline_bottom, timeline_width, timeline_height = 0.08, 0.018, 0.84, 0.012
    fig.add_artist(
        Rectangle(
            (timeline_left, timeline_bottom), timeline_width, timeline_height,
            transform=fig.transFigure, facecolor="#dbe5ef", edgecolor="#94a3b8",
            linewidth=0.6, zorder=100,
        )
    )
    timeline_progress = Rectangle(
        (timeline_left, timeline_bottom), 0.0, timeline_height,
        transform=fig.transFigure, facecolor="#2878ff", edgecolor="none", zorder=101,
    )
    fig.add_artist(timeline_progress)
    event_fraction = base_event_step / max(n - 1, 1)
    event_x = timeline_left + timeline_width * event_fraction
    fig.add_artist(
        Rectangle(
            (event_x - 0.001, timeline_bottom - 0.004), 0.002, timeline_height + 0.008,
            transform=fig.transFigure, facecolor="#dc2626", edgecolor="none", zorder=102,
        )
    )
    fig.text(timeline_left, 0.004, "0.0 s", ha="left", va="bottom", fontsize=7, color="#52677d")
    fig.text(
        event_x, 0.004, f"event {base_event_step * 0.1:.1f} s",
        ha="center", va="bottom", fontsize=7, color="#991b1b", fontweight="bold",
    )
    fig.text(
        timeline_left + timeline_width, 0.004, f"{(n - 1) * 0.1:.1f} s",
        ha="right", va="bottom", fontsize=7, color="#52677d",
    )

    def draw(frame_index: int):
        frame = int(frames[frame_index])
        timeline_progress.set_width(
            timeline_width * (frame + 1) / max(n, 1)
        )
        for ax, color, trajectory, label in [
            (axes[0], "#b91c1c", base[0], "original DP"),
            (axes[1], "#047857", repair[repair_index], "AWR deterministic checkpoint"),
        ]:
            ax.clear()
            _draw_map(ax, z)
            _draw_expert(ax, z)
            for context in base[1:]:
                ax.plot(context[:, 0], context[:, 1], color="#94a3b8", lw=0.45, alpha=0.11)
            upto = frame + 1
            ax.plot(trajectory[:upto, 0], trajectory[:upto, 1], color=color, lw=3.0, label=label)
            _draw_neighbor_boxes(
                ax, z, frame=frame, max_neighbors=max_neighbors, trails=False,
                focus_xy=trajectory[frame, :2],
            )
            _draw_box(ax, _ego_corners(trajectory, shape, frame), color, 0.28, 1.3, 22)
            _draw_box(ax, obb_corners(0.0, 0.0, 0.0, float(shape[1]), float(shape[2]), float(shape[0])), "#111827", 0.65, 1.0, 24)
            ax.set_xlim(limits[0], limits[1]); ax.set_ylim(limits[2], limits[3])
            ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=0.14)
            ax.set_title(f"t={frame * 0.1:.1f}s · {label} · real neighbor OBBs", loc="left", fontsize=10)
            ax.legend(fontsize=7, loc="upper left")
        # A synchronized metric-scale event camera makes actual OBB overlap or
        # border contact legible even when the global trajectory is long.
        event_ax = axes[2]
        event_ax.clear()
        _draw_map(event_ax, z)
        _draw_expert(event_ax, z)
        event_ax.plot(base[0, : frame + 1, 0], base[0, : frame + 1, 1], color="#b91c1c", lw=2.6, label="source")
        event_ax.plot(repair[repair_index, : frame + 1, 0], repair[repair_index, : frame + 1, 1], color="#047857", lw=2.6, label="AWR")
        center = 0.5 * (base[0, frame, :2] + repair[repair_index, frame, :2])
        _draw_neighbor_boxes(
            event_ax, z, frame=frame, max_neighbors=min(max_neighbors, 16),
            trails=False, focus_xy=center,
        )
        _draw_box(event_ax, _ego_corners(base[0], shape, frame), "#b91c1c", 0.30, 1.5, 24)
        _draw_box(event_ax, _ego_corners(repair[repair_index], shape, frame), "#047857", 0.30, 1.5, 25)
        half = 10.0
        event_ax.set_xlim(center[0] - half, center[0] + half)
        event_ax.set_ylim(center[1] - half, center[1] + half)
        event_ax.set_aspect("equal", adjustable="box")
        event_ax.grid(alpha=0.14)
        state = " ← DECISION FRAME" if frame == base_event_step else ""
        event_ax.set_title(
            f"OBB EVENT CAMERA · {event_label} @ {base_event_step * 0.1:.1f}s{state}",
            loc="left", fontsize=9.2, fontweight="bold",
            color="#991b1b" if frame == base_event_step else "#334e68",
        )
        event_ax.legend(fontsize=7, loc="upper left")
        fig.suptitle(
            f"{_promotion_label(promotion_status)} · {selection_kind or 'paired audit'}\n"
            f"{Path(scene).name} · frame {frame + 1}/{n}",
            fontsize=13, fontweight="bold", y=0.995,
        )
        return [*axes, timeline_progress]

    movie = animation.FuncAnimation(fig, draw, frames=len(frames), interval=1000 / max(fps, 1), blit=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    movie.save(output, writer=animation.PillowWriter(fps=fps), dpi=105)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--base_npz", type=Path, required=True)
    parser.add_argument("--repair_npz", type=Path, required=True)
    parser.add_argument("--reward_config", type=Path, required=True)
    parser.add_argument("--reward_profile", default=None)
    parser.add_argument("--reward_mode", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max_frames", type=int, default=80)
    parser.add_argument("--max_neighbors", type=int, default=24)
    parser.add_argument("--solver_steps", type=int, default=10)
    parser.add_argument(
        "--selection-json",
        type=Path,
        default=None,
        help="full_validation_scene_selection.json; injects this scene's audit role",
    )
    parser.add_argument(
        "--promotion-status",
        choices=[
            "aggregate_promotion_passed",
            "not_promoted_no_mean_gain",
            "strict_promotion_passed",
            "not_promoted_failure_audit",
            "unspecified",
        ],
        default="unspecified",
        help="global full-validation decision printed on every copied figure/frame",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    z = _load_aligned_npz(args.scene)
    data = load_scene(args.scene, device)
    config = _load_config(args.reward_config)
    if args.reward_profile is not None:
        config.reward_profile = args.reward_profile
    if args.reward_mode is not None:
        config.reward_mode = args.reward_mode
    base_npz = np.load(args.base_npz)
    repair_npz = np.load(args.repair_npz)
    base = base_npz["trajectories"]
    repair = repair_npz["trajectories"]
    repair_index = int(repair_npz["selected_index"].item())
    base_tensor = torch.from_numpy(base).to(device=device, dtype=torch.float32)
    repair_tensor = torch.from_numpy(repair).to(device=device, dtype=torch.float32)
    reward_data = reward_compatible_data(data)
    base_rewards = compute_reward_batch(base_tensor, reward_data, config)
    repair_rewards = compute_reward_batch(repair_tensor, reward_data, config)
    base_subscores = compute_subscores_batch(base_tensor, reward_data, config)
    repair_subscores = compute_subscores_batch(repair_tensor, reward_data, config)
    reward_horizon = min(
        int(getattr(config, "reward_horizon_steps", base.shape[1])),
        base.shape[1], repair.shape[1],
    )
    base_step, event_label = _paired_event_step(
        base_rewards[0], repair_rewards[repair_index], base_subscores, repair_subscores,
        base[0], repair[repair_index], reward_horizon, repair_index,
    )
    base_metrics = _trajectory_metrics(base_tensor[0], data)
    repair_metrics = _trajectory_metrics(repair_tensor[repair_index], data)
    evidence_label = _evidence_label(
        base_rewards[0], repair_rewards[repair_index], base_metrics, repair_metrics,
    )
    selection = _load_selection_row(args.selection_json, args.scene)
    selection_kind = (
        str(selection.get("kind", "paired_audit")).replace("_", " ")
        if selection is not None
        else None
    )
    all_trajs = np.concatenate([base[:1], repair], axis=0)
    points = _all_points(z, all_trajs, args.max_neighbors)
    center = np.median(points, axis=0)
    span = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])))
    half = max(12.0, 0.52 * span + 4.0)
    limits = (float(center[0] - half), float(center[0] + half), float(center[1] - half), float(center[1] + half))

    fig = plt.figure(figsize=(20, 13), facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=[4.25, 1.75], hspace=0.18, wspace=0.07)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    _draw_panel(
        axes[0], z, base, repair, base_rewards[0], repair_rewards[repair_index], base_step,
        repair_index, "source", "Source Original DP · primary", event_label, limits, args.max_neighbors,
    )
    _draw_panel(
        axes[1], z, base, repair, base_rewards[0], repair_rewards[repair_index], base_step,
        repair_index, "awr", "AWR checkpoint · primary · same scene / scale", event_label, limits, args.max_neighbors,
    )
    lower_left = grid[1, 0].subgridspec(2, 1, hspace=0.45)
    distance_ax = fig.add_subplot(lower_left[0, 0])
    speed_ax = fig.add_subplot(lower_left[1, 0])
    _draw_temporal_diagnostics(distance_ax, speed_ax, z, base[0], repair[repair_index])
    metric_ax = fig.add_subplot(grid[1, 1])
    _draw_metric_table(
        metric_ax, base_rewards[0], repair_rewards[repair_index], base_metrics, repair_metrics,
    )
    fig.suptitle(
        f"{_promotion_label(args.promotion_status)} · "
        f"{selection_kind or 'paired deployment scene'}\n"
        f"scene verdict: {evidence_label} · {Path(args.scene).name} · "
        f"{event_label} @ {base_step * 0.1:.1f}s · EMA K=1 / zero noise / DP-{args.solver_steps}",
        fontsize=14, fontweight="bold", y=0.997,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    if args.gif:
        _animate(
            str(args.scene), z, base, repair, repair_index, args.output.with_suffix(".gif"),
            base_step, event_label, args.max_neighbors, args.fps, args.max_frames, limits,
            selection_kind, args.promotion_status,
        )
    metadata = {
        "scene": str(args.scene),
        "base_npz": str(args.base_npz),
        "repair_npz": str(args.repair_npz),
        "source_npz": str(args.base_npz),
        "awr_npz": str(args.repair_npz),
        "output": str(args.output),
        "gif": str(args.output.with_suffix('.gif')) if args.gif else None,
        "event_step": base_step,
        "event_label": event_label,
        "evidence_label": evidence_label,
        "selected_index": repair_index,
        "selection": selection,
        "selection_kind": selection.get("kind") if selection is not None else None,
        "promotion_status": args.promotion_status,
        "evaluation_policy": "zero-noise deterministic K=1",
        "solver_steps": args.solver_steps,
        "geometry_contract": {
            "legacy_neighbor_future_alignment": "raw[t+1] -> aligned[t]",
            "ego_trajectory_reference": "rear axle",
            "ego_obb_center": "rear axle + 0.5 * wheelbase along heading",
            "neighbor_obb_reference": "box centroid",
            "timeline_hz": 10,
        },
        "base_reward": float(base_rewards[0].total),
        "repair_reward": float(repair_rewards[repair_index].total),
        "source_reward": float(base_rewards[0].total),
        "awr_reward": float(repair_rewards[repair_index].total),
        "source_reward_breakdown": reward_breakdown_to_json_dict(base_rewards[0]),
        "awr_reward_breakdown": reward_breakdown_to_json_dict(repair_rewards[repair_index]),
        "base_collision": bool(base_rewards[0].collision_step is not None),
        "repair_collision": bool(repair_rewards[repair_index].collision_step is not None),
        "base_rb_crossing": bool(base_rewards[0].rb_crossing),
        "repair_rb_crossing": bool(repair_rewards[repair_index].rb_crossing),
        "base_metrics": base_metrics,
        "repair_metrics": repair_metrics,
        "source_metrics": base_metrics,
        "awr_metrics": repair_metrics,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
