#!/usr/bin/env python3
"""Scene-level visualization for original-DP AWR.

The figure is deliberately a BEV diagnostic rather than a metric-only chart:
it draws the real T4 map, recorded neighbour motion and oriented bounding
boxes, the human future, every unguided DP candidate, and the candidate's
reward/AWR-weight breakdown.  The optional GIF animates the same real scene
at 10 Hz so the collision, border approach, yielding, or mode collapse is
visible as a temporal event.

Usage with a checkpoint emitted by ``train_awr.py``:

    python -m rlvr.autoresearch.tools.awr_viz \
      --model_path outputs/awr_original_dp/.../epoch_001.pth \
      --args_path outputs/awr_original_dp/.../model_args.json \
      --scene /path/to/scene.npz --output_dir /tmp/awr_viz \
      --device cuda --K 16 --gif
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import fields
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Polygon

from diffusion_planner.utils.neighbor_future_alignment import (
    align_neighbor_future_numpy,
)
from planner_metrics.config import RewardConfig
from planner_metrics.vehicle_collision import obb_corners
from rlvr.awr import (
    AWRRolloutConfig,
    load_original_dp_checkpoint,
    load_scene,
    rollout_and_score_scene,
    rollout_to_json,
)


_TYPE_COLORS = {0: "#2b6cb0", 1: "#dd6b20", 2: "#805ad5"}
_TYPE_NAMES = {0: "vehicle", 1: "pedestrian", 2: "bicycle"}


def _safe_float(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def _load_reward_config(path: Path | None) -> RewardConfig:
    if path is None:
        return RewardConfig(
            enable_lane_departure=True,
            lane_gate_enabled=True,
            enable_overprogress=True,
            reward_mode="survival",
        )
    raw = json.loads(path.read_text())
    raw = dict(raw.get("reward", raw.get("reward_config", raw)))
    names = {field.name for field in fields(RewardConfig)}
    return RewardConfig(**{key: value for key, value in raw.items() if key in names})


def _load_scene_list(scenes: Path | None, explicit: list[str] | None) -> list[str]:
    if explicit:
        return explicit
    if scenes is None:
        raise ValueError("provide --scene or --scenes")
    paths = json.loads(scenes.read_text())
    if not isinstance(paths, list):
        raise ValueError(f"scene list is not a JSON list: {scenes}")
    return [str(path) for path in paths]


def _load_aligned_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load the scene with the same legacy +1 neighbour timeline as AWR."""

    with np.load(str(path), allow_pickle=False) as archive:
        scene = {key: np.asarray(archive[key]) for key in archive.files}
    if "neighbor_agents_future" in scene:
        scene["neighbor_agents_future"] = align_neighbor_future_numpy(
            scene["neighbor_agents_future"]
        )
    return scene


def _yaw_from_state(state: np.ndarray) -> float:
    if state.shape[-1] >= 4:
        return float(np.arctan2(state[..., 3], state[..., 2]))
    return float(state[..., 2])


def _valid_xy(points: np.ndarray) -> np.ndarray:
    return np.isfinite(points).all(axis=-1) & (np.abs(points).sum(axis=-1) > 1e-4)


def _neighbor_type(state: np.ndarray) -> int:
    if state.shape[-1] >= 11:
        return int(np.argmax(state[8:11]))
    return 0


def _neighbor_dims(state: np.ndarray) -> tuple[float, float]:
    width = _safe_float(state[6]) if state.shape[-1] > 6 else float("nan")
    length = _safe_float(state[7]) if state.shape[-1] > 7 else float("nan")
    if not math.isfinite(width) or width < 0.2:
        width = 1.8
    if not math.isfinite(length) or length < 0.5:
        length = 4.5
    return length, width


def _lane_segments(z: dict[str, np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    lanes = z.get("lanes")
    if lanes is None:
        return segments
    for lane in lanes:
        pts = lane[:, :2]
        valid = _valid_xy(pts)
        if valid.sum() < 2 or lane.shape[-1] < 8:
            continue
        left = pts + lane[:, 4:6]
        right = pts + lane[:, 6:8]
        segments.append((left[valid], right[valid]))
    return segments


def _road_border_segments(z: dict[str, np.ndarray]) -> list[np.ndarray]:
    lines = z.get("line_strings")
    if lines is None:
        return []
    result = []
    for line in lines:
        valid = _valid_xy(line[:, :2])
        if line.shape[-1] >= 4:
            valid &= line[:, 3] > 0.5
        if valid.sum() >= 2:
            result.append(line[valid, :2])
    return result


def _raw_neighbors(z: dict[str, np.ndarray], frame: int = 0) -> list[dict[str, Any]]:
    past = z.get("neighbor_agents_past")
    future = z.get("neighbor_agents_future")
    if past is None:
        return []
    records: list[dict[str, Any]] = []
    n = past.shape[0]
    for slot in range(n):
        state = past[slot, -1]
        future_state = None
        if future is not None and slot < future.shape[0] and frame < future.shape[1]:
            candidate = future[slot, frame]
            if _valid_xy(candidate[None, :2])[0]:
                future_state = candidate
        if future_state is None and not _valid_xy(state[None, :2])[0]:
            continue
        if future_state is not None:
            x, y = float(future_state[0]), float(future_state[1])
            yaw = _yaw_from_state(future_state)
        else:
            x, y = float(state[0]), float(state[1])
            yaw = _yaw_from_state(state)
        length, width = _neighbor_dims(state)
        type_id = _neighbor_type(state)
        records.append(
            {
                "slot": slot,
                "x": x,
                "y": y,
                "yaw": yaw,
                "length": length,
                "width": width,
                "type": type_id,
                "state": state,
            }
        )
    return records


def _draw_polygon_box(ax, corners: np.ndarray, edge: str, face: str, alpha: float, lw: float = 1.0, zorder: int = 20):
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


def _draw_map(ax, z: dict[str, np.ndarray], show_route: bool = True) -> None:
    for left, right in _lane_segments(z):
        ax.plot(left[:, 0], left[:, 1], color="#94a3b8", lw=0.8, alpha=0.65, zorder=1)
        ax.plot(right[:, 0], right[:, 1], color="#94a3b8", lw=0.8, alpha=0.65, zorder=1)
    if show_route and "route_lanes" in z:
        for lane in z["route_lanes"]:
            pts = lane[:, :2]
            valid = _valid_xy(pts)
            if valid.sum() >= 2:
                ax.plot(pts[valid, 0], pts[valid, 1], color="#38bdf8", lw=2.2, alpha=0.55, zorder=2)
    for line in _road_border_segments(z):
        ax.plot(line[:, 0], line[:, 1], color="#dc2626", lw=2.5, alpha=0.85, zorder=3)


def _draw_neighbor_boxes(
    ax,
    z: dict[str, np.ndarray],
    frame: int,
    max_neighbors: int = 24,
    trails: bool = True,
    focus_xy: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Draw recorded neighbor OBBs, optionally ranked around a frame-local focus.

    The default preserves the historical t=0/ego-origin ranking used by the
    general galleries.  Paired event visualizations pass the current ego or
    event-camera center so an actor that approaches later cannot be omitted
    merely because it was far away at the beginning of the scene.
    """

    future = z.get("neighbor_agents_future")
    past = z.get("neighbor_agents_past")
    if past is None:
        return []
    focus = None if focus_xy is None else np.asarray(focus_xy, dtype=float).reshape(2)
    valid_slots = []
    for slot in range(past.shape[0]):
        state = past[slot, -1]
        if _valid_xy(state[None, :2])[0]:
            rank_xy = np.asarray(state[:2], dtype=float)
            if focus is not None and future is not None and slot < future.shape[0] and frame < future.shape[1]:
                future_state = future[slot, frame]
                if _valid_xy(future_state[None, :2])[0]:
                    rank_xy = np.asarray(future_state[:2], dtype=float)
            dist = float(np.linalg.norm(rank_xy if focus is None else rank_xy - focus))
            valid_slots.append((dist, slot))
    valid_slots = [slot for _, slot in sorted(valid_slots)[:max_neighbors]]
    points: list[np.ndarray] = []
    for slot in valid_slots:
        state = past[slot, -1]
        length, width = _neighbor_dims(state)
        type_id = _neighbor_type(state)
        color = _TYPE_COLORS.get(type_id, _TYPE_COLORS[0])
        if trails and future is not None and slot < future.shape[0]:
            tr = future[slot, :, :2]
            valid = _valid_xy(tr)
            if valid.sum() >= 2:
                ax.plot(tr[valid, 0], tr[valid, 1], color=color, lw=0.7, alpha=0.22, zorder=4)
                points.append(tr[valid])
        if future is not None and slot < future.shape[0] and frame < future.shape[1]:
            fs = future[slot, frame]
            if _valid_xy(fs[None, :2])[0]:
                x, y, yaw = float(fs[0]), float(fs[1]), _yaw_from_state(fs)
            else:
                x, y, yaw = float(state[0]), float(state[1]), _yaw_from_state(state)
        else:
            x, y, yaw = float(state[0]), float(state[1]), _yaw_from_state(state)
        corners = obb_corners(x, y, yaw, length, width, wheelbase=None)
        _draw_polygon_box(ax, corners, edge=color, face=color, alpha=0.28, lw=1.1, zorder=8)
        center = np.array([x, y])
        head = center + np.array([math.cos(yaw), math.sin(yaw)]) * (length * 0.6)
        ax.plot([center[0], head[0]], [center[1], head[1]], color=color, lw=1.2, alpha=0.9, zorder=9)
        points.append(corners)
    return points


def _ego_corners(traj: np.ndarray, ego_shape: np.ndarray, frame: int) -> np.ndarray:
    wheelbase, length, width = (float(x) for x in ego_shape[:3])
    frame = min(max(int(frame), 0), traj.shape[0] - 1)
    state = traj[frame]
    return obb_corners(
        float(state[0]),
        float(state[1]),
        _yaw_from_state(state),
        length,
        width,
        wheelbase=wheelbase,
    )


def _all_points(z: dict[str, np.ndarray], trajectories: np.ndarray, max_neighbors: int = 32) -> np.ndarray:
    chunks = [trajectories[:, :, :2].reshape(-1, 2)]
    if "ego_agent_future" in z:
        chunks.append(z["ego_agent_future"][:, :2])
    for left, right in _lane_segments(z):
        chunks.extend([left, right])
    for line in _road_border_segments(z):
        chunks.append(line)
    if "neighbor_agents_future" in z and "neighbor_agents_past" in z:
        past = z["neighbor_agents_past"]
        slots = []
        for slot in range(past.shape[0]):
            if _valid_xy(past[slot, -1:, :2])[0]:
                slots.append((float(np.linalg.norm(past[slot, -1, :2])), slot))
        for _, slot in sorted(slots)[:max_neighbors]:
            tr = z["neighbor_agents_future"][slot, :, :2]
            chunks.append(tr[_valid_xy(tr)])
    points = np.vstack([chunk for chunk in chunks if len(chunk)])
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    # The scene map contains distant lanes for the entire local map.  Keep
    # them as context, but do not let them set limits that make the ego and
    # neighbor OBBs unreadably small.
    motion = [trajectories[:, :, :2].reshape(-1, 2)]
    if "ego_agent_future" in z:
        motion.append(z["ego_agent_future"][:, :2])
    motion_points = np.vstack(motion)
    motion_points = motion_points[_valid_xy(motion_points)]
    if len(motion_points):
        center = np.median(motion_points, axis=0)
        motion_span = max(float(np.ptp(motion_points[:, 0])), float(np.ptp(motion_points[:, 1])))
        radius = max(70.0, 1.15 * motion_span + 12.0)
        close = np.linalg.norm(points - center, axis=1) < radius
        if close.any():
            points = points[close]
    return points


def _trajectory_results(rollout, data: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    trajectories = rollout.trajectories.detach().cpu().numpy()
    weights = rollout.weights.detach().cpu().numpy()
    order = np.argsort([-float(reward.total) for reward in rollout.rewards])
    results = []
    for rank, index in enumerate(order):
        reward = rollout.rewards[int(index)]
        traj = trajectories[int(index)]
        results.append(
            {
                "rank": rank + 1,
                "index": int(index),
                "trajectory": traj,
                "reward": reward,
                "weight": float(weights[int(index)]),
                "path_len": float(np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1).sum()),
            }
        )
    return results


def _reward_contributions(reward, cfg: RewardConfig, subscores: dict[str, Any], index: int) -> dict[str, float]:
    """Return an honest additive view of the shaped reward.

    The reward implementation normalizes progress, adds a TTC bonus, applies
    survival/hard gates, and then floors failed trajectories.  Those effects
    are not all stored in ``RewardBreakdown``.  We therefore show the known
    additive terms and put the exact remainder into ``shaping / gates`` rather
    than pretending that the raw breakdown fields sum to ``total``.
    """
    ttc = float(subscores["ttc"][index].detach().cpu().item())
    contributions = {
        "safety": float(cfg.w_safety * reward.safety),
        "progress": float(cfg.w_progress * reward.progress),
        "smoothness": float(cfg.w_smooth * reward.smoothness),
        "centerline": float(cfg.w_centerline * reward.centerline),
        "ttc_bonus": float(cfg.w_safety * (ttc - 0.5) * 2.0),
        "rb_penalty": float(-cfg.rb_near_scale * reward.rb_near_penalty - cfg.rb_wide_scale * reward.rb_wide_penalty),
        "lane_penalty": float(-cfg.lane_near_scale * reward.lane_near_frac - cfg.lane_wide_scale * reward.lane_wide_frac),
    }
    contributions["shaping_gates"] = float(reward.total - sum(contributions.values()))
    # Kept separately for the audit note below: the current quality_score
    # computes this subscore but does not add w_feasibility * feasibility.
    contributions["feasibility_audit"] = float(cfg.w_feasibility * reward.feasibility)
    return contributions


def _draw_reward_panel(
    ax,
    results: list[dict[str, Any]],
    cfg: RewardConfig,
    subscores: dict[str, Any],
    hdp_components: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> None:
    ax.set_title("Exact total decomposition (top / median / worst)", fontsize=10, loc="left")
    if not results:
        ax.axis("off")
        return
    picks = [results[0], results[len(results) // 2], results[-1]]
    pick_labels = [f"#{item['rank']} idx={item['index']}" for item in picks]
    if cfg.reward_profile == "hdp_multi" and hdp_components is not None:
        risk, follow, lane = hdp_components
        normalizer = max(
            float(cfg.hdp_risk_weight + cfg.hdp_follow_weight + cfg.hdp_lane_weight),
            1e-6,
        )
        keys = ["risk", "follow", "lane", "gate_remainder"]
        component_labels = {
            "risk": f"risk × {cfg.hdp_risk_weight:g}/{normalizer:g}",
            "follow": f"follow × {cfg.hdp_follow_weight:g}/{normalizer:g}",
            "lane": f"lane × {cfg.hdp_lane_weight:g}/{normalizer:g}",
            "gate_remainder": "kinematic gate remainder",
        }
        colors = ["#dc2626", "#2563eb", "#16a34a", "#7c3aed"]
        values = []
        for item in picks:
            index = int(item["index"])
            weighted = (
                float(cfg.hdp_risk_weight) * float(risk[index])
                + float(cfg.hdp_follow_weight) * float(follow[index])
                + float(cfg.hdp_lane_weight) * float(lane[index])
            ) / normalizer
            values.append(
                [
                    float(cfg.hdp_risk_weight) * float(risk[index]) / normalizer,
                    float(cfg.hdp_follow_weight) * float(follow[index]) / normalizer,
                    float(cfg.hdp_lane_weight) * float(lane[index]) / normalizer,
                    float(item["reward"].total) - weighted,
                ]
            )
        values = np.asarray(values, dtype=np.float64)
        y = np.arange(len(picks))
        left_pos = np.zeros(len(picks))
        left_neg = np.zeros(len(picks))
        for j, (key, color) in enumerate(zip(keys, colors)):
            value = values[:, j]
            pos = np.maximum(value, 0)
            neg = np.minimum(value, 0)
            ax.barh(
                y,
                pos,
                left=left_pos,
                color=color,
                alpha=0.84,
                height=0.42,
                label=component_labels[key],
            )
            ax.barh(y, neg, left=left_neg, color=color, alpha=0.42, height=0.42)
            left_pos += pos
            left_neg += neg
        ax.axvline(0, color="#111827", lw=0.8)
        ax.set_xlim(min(-0.05, float(left_neg.min()) - 0.03), 1.03)
        ax.set_ylim(-0.6, len(picks) - 0.4)
        ax.set_yticks(y, pick_labels, fontsize=8)
        ax.set_xlabel("contribution to exact HDP multi reward", fontsize=7)
        ax.grid(axis="x", alpha=0.2)
        ax.legend(ncol=2, fontsize=6, loc="lower right", framealpha=0.85)
        ax.set_title("HDP multi reward: bounded risk / follow / lane", fontsize=10, loc="left")
        ax.text(
            0.0,
            -0.22,
            "R = (1·risk + 3·follow + 2.5·lane) / 6.5; purple is only a hard kinematic-gate remainder",
            fontsize=6.2,
            color="#475569",
            transform=ax.transAxes,
        )
        return
    keys = ["safety", "progress", "smoothness", "centerline", "ttc_bonus", "rb_penalty", "lane_penalty", "shaping_gates"]
    component_labels = {
        "safety": "safety",
        "progress": "progress",
        "smoothness": "smoothness",
        "centerline": "centerline",
        "ttc_bonus": "TTC bonus",
        "rb_penalty": "RB penalty",
        "lane_penalty": "lane penalty",
        "shaping_gates": "shaping / gates",
    }
    colors = ["#2563eb", "#16a34a", "#64748b", "#0891b2", "#0f766e", "#dc2626", "#ea580c", "#9333ea"]
    values = np.asarray([
        [_reward_contributions(item["reward"], cfg, subscores, item["index"])[key] for key in keys]
        for item in picks
    ])
    y = np.arange(len(picks))
    left_pos = np.zeros(len(picks))
    left_neg = np.zeros(len(picks))
    for j, (key, color) in enumerate(zip(keys, colors)):
        value = values[:, j]
        pos = np.maximum(value, 0)
        neg = np.minimum(value, 0)
        ax.barh(y, pos, left=left_pos, color=color, alpha=0.82, height=0.42, label=component_labels[key])
        ax.barh(y, neg, left=left_neg, color=color, alpha=0.42, height=0.42)
        left_pos += pos
        left_neg += neg
    ax.axvline(0, color="#111827", lw=0.8)
    ax.set_yticks(y, pick_labels, fontsize=8)
    ax.grid(axis="x", alpha=0.2)
    ax.legend(ncol=2, fontsize=6, loc="lower right", framealpha=0.85)
    audit = [
        _reward_contributions(item["reward"], cfg, subscores, item["index"])["feasibility_audit"]
        for item in picks
    ]
    ax.text(
        0.0,
        -0.19,
        "* feasibility audit (configured but currently not added to quality_score): "
        + ", ".join(f"{value:+.2f}" for value in audit),
        fontsize=6.1,
        color="#92400e",
        transform=ax.transAxes,
    )


def _draw_table(
    ax,
    results: list[dict[str, Any]],
    hdp_components: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> None:
    ax.axis("off")
    is_hdp = hdp_components is not None
    if is_hdp:
        headers = ["rank", "idx", "R", "AWR w", "risk", "follow", "lane", "col", "RB", "lane*", "path"]
        widths = [0.06, 0.06, 0.09, 0.10, 0.09, 0.10, 0.09, 0.07, 0.08, 0.09, 0.12]
    else:
        headers = ["rank", "idx", "R", "AWR w", "safety", "prog", "smooth", "CL", "RB", "lane", "path"]
        widths = [0.07, 0.07, 0.10, 0.11, 0.10, 0.10, 0.10, 0.09, 0.07, 0.08, 0.11]
    widths = np.asarray(widths, dtype=np.float64)
    widths /= widths.sum()
    # ``[0.0] + np.ndarray`` performs elementwise NumPy addition rather than
    # list concatenation and silently drops the leading boundary.  Use an
    # explicit concatenate so every header has both left and right edges.
    x_positions = np.cumsum(np.concatenate(([0.0], widths)))
    title = "Per-trajectory OBB-aware diagnostics"
    if is_hdp:
        title += " · HDP components are exact"
    ax.text(0.0, 1.0, title, fontsize=10, fontweight="bold", va="top", transform=ax.transAxes)
    y = 0.94
    for index, header in enumerate(headers):
        ax.text((x_positions[index] + x_positions[index + 1]) / 2, y, header, fontsize=7, fontweight="bold", ha="center", va="top", transform=ax.transAxes)
    ax.plot([0, 1], [y - 0.025, y - 0.025], color="#111827", lw=0.7, transform=ax.transAxes)
    y -= 0.065
    for item in results:
        if y < 0.02:
            break
        reward = item["reward"]
        if is_hdp:
            risk, follow, lane = hdp_components
            index = int(item["index"])
            row = [
                str(item["rank"]),
                str(index),
                f"{reward.total:.3f}",
                f"{item['weight']:.2f}",
                f"{float(risk[index]):.3f}",
                f"{float(follow[index]):.3f}",
                f"{float(lane[index]):.3f}",
                "X" if reward.collision_step is not None else "ok",
                "X" if reward.rb_crossing else "ok",
                "X" if reward.lane_crossing else "ok",
                f"{item['path_len']:.1f}",
            ]
        else:
            row = [
                str(item["rank"]),
                str(item["index"]),
                f"{reward.total:+.1f}",
                f"{item['weight']:.2f}",
                f"{reward.safety:+.2f}",
                f"{reward.progress:+.1f}",
                f"{reward.smoothness:+.2f}",
                f"{reward.centerline:+.2f}",
                "X" if reward.rb_crossing else "ok",
                "X" if reward.lane_crossing else "ok",
                f"{item['path_len']:.1f}",
            ]
        bg = "#dcfce7" if item["rank"] == 1 else ("#fee2e2" if reward.rb_crossing or reward.collision_step is not None else "#f8fafc")
        ax.fill_between([0, 1], y - 0.024, y + 0.022, color=bg, alpha=0.9, transform=ax.transAxes)
        for index, value in enumerate(row):
            ax.text((x_positions[index] + x_positions[index + 1]) / 2, y, value, fontsize=6.6, ha="center", va="center", color="#111827", transform=ax.transAxes)
        y -= 0.045
    ax.text(
        0.0,
        max(0.01, y - 0.02),
        (
            "R: exact HDP total · risk/follow/lane: bounded components · lane*: diagnostic only · "
            "AWR w: group-normalized reward weight"
            if is_hdp
            else "R: total reward · prog: goal progress · CL: centerline · RB/lane: hard violation flags · AWR w: exp(beta·z), mean-normalized"
        ),
        fontsize=6.4,
        color="#475569",
        transform=ax.transAxes,
    )


def _draw_ego_zoom(
    ax,
    z: dict[str, np.ndarray],
    results: list[dict[str, Any]],
    ego_shape: np.ndarray,
    max_neighbors: int,
) -> None:
    """Add a readable local view so four-metre OBBs are not lost in map context."""

    _draw_map(ax, z)
    top = results[: min(3, len(results))]
    if not top:
        ax.axis("off")
        return
    trajectories = np.stack([item["trajectory"] for item in top])
    local = [trajectories[:, :, :2].reshape(-1, 2)]
    if "neighbor_agents_past" in z:
        past = z["neighbor_agents_past"]
        for slot in range(past.shape[0]):
            state = past[slot, -1]
            if _valid_xy(state[None, :2])[0] and float(np.linalg.norm(state[:2])) < 22.0:
                local.append(state[None, :2])
    points = np.vstack(local)
    center = np.median(points, axis=0)
    span = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])))
    half = max(8.0, 0.58 * span + 4.0)
    _draw_neighbor_boxes(ax, z, frame=0, max_neighbors=max_neighbors, trails=False)
    for rank, item in enumerate(top):
        traj = item["trajectory"]
        color = plt.cm.RdYlGn(1.0 - rank / max(len(top) - 1, 1))
        ax.plot(traj[:, 0], traj[:, 1], color=color, lw=2.0 if rank == 0 else 1.2, alpha=0.9, zorder=15)
        for frame in [0, 20, 40, 60, 79]:
            if frame < len(traj):
                _draw_polygon_box(
                    ax,
                    _ego_corners(traj, ego_shape, frame),
                    color,
                    color,
                    0.18 if frame == 0 else 0.08,
                    1.0,
                    18,
                )
    _draw_polygon_box(
        ax,
        obb_corners(0.0, 0.0, 0.0, float(ego_shape[1]), float(ego_shape[2]), wheelbase=float(ego_shape[0])),
        "#111827",
        "#111827",
        0.78,
        1.4,
        25,
    )
    ax.set_xlim(float(center[0] - half), float(center[0] + half))
    ax.set_ylim(float(center[1] - half), float(center[1] + half))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("ego-neighborhood zoom · OBB footprints", fontsize=7, loc="left", pad=2)
    for spine in ax.spines.values():
        spine.set_color("#111827")
        spine.set_linewidth(1.2)


def draw_scene_figure(
    scene_path: str,
    rollout,
    reward_config: RewardConfig,
    output_path: Path,
    tag: str,
    max_neighbors: int,
) -> None:
    z = _load_aligned_npz(scene_path)
    data = rollout.reward_data
    results = _trajectory_results(rollout, data)
    from planner_metrics.aggregate import compute_subscores_batch

    subscores = compute_subscores_batch(rollout.trajectories, data, reward_config)
    hdp_components = None
    if reward_config.reward_profile == "hdp_multi":
        from rlvr.reward import _hdp_multi_reward_components

        hdp_components = tuple(
            component.detach().cpu().numpy()
            for component in _hdp_multi_reward_components(
                rollout.trajectories, data, subscores, reward_config
            )
        )
    trajectories = np.stack([item["trajectory"] for item in results])
    points = _all_points(z, trajectories, max_neighbors)
    ego_shape = z.get("ego_shape", np.asarray([2.75, 4.34, 1.70], dtype=np.float32))
    if ego_shape.ndim > 1:
        ego_shape = ego_shape[0]

    fig = plt.figure(figsize=(21, 11), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.47, 1.0], wspace=0.08)
    ax = fig.add_subplot(grid[0, 0])
    side = grid[0, 1].subgridspec(2, 1, height_ratios=[0.9, 1.35], hspace=0.08)
    ax_reward = fig.add_subplot(side[0, 0])
    ax_table = fig.add_subplot(side[1, 0])

    _draw_map(ax, z)
    if "ego_agent_future" in z:
        gt = z["ego_agent_future"]
        valid = _valid_xy(gt[:, :2])
        if valid.sum() >= 2:
            ax.plot(gt[valid, 0], gt[valid, 1], "--", color="#15803d", lw=2.5, alpha=0.8, label="logged expert")

    _draw_neighbor_boxes(ax, z, frame=0, max_neighbors=max_neighbors, trails=True)
    # Current ego footprint, with the actual rear-axle convention used by the
    # canonical reward collision geometry.
    current_ego = np.asarray([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    _draw_polygon_box(ax, obb_corners(0.0, 0.0, 0.0, float(ego_shape[1]), float(ego_shape[2]), float(ego_shape[0])), "#111827", "#111827", 0.75, 1.8, 25)
    ax.text(0.0, -float(ego_shape[2]) * 0.9, "ego @ t=0", fontsize=7, ha="center", color="white", zorder=26)

    cmap = plt.cm.RdYlGn
    for item in results:
        traj = item["trajectory"]
        rank = item["rank"] - 1
        color = cmap(1.0 - rank / max(len(results) - 1, 1))
        alpha = 0.95 if rank < 3 else 0.25
        lw = 2.8 if rank < 3 else 0.9
        label = f"#{item['rank']} R={item['reward'].total:+.1f}" if rank < 3 else None
        ax.plot(traj[:, 0], traj[:, 1], color=color, lw=lw, alpha=alpha, label=label, zorder=12 - rank * 0.02)
        if rank < 3:
            for frame in [0, 20, 40, 60, 79]:
                if frame < len(traj):
                    corners = _ego_corners(traj, ego_shape, frame)
                    _draw_polygon_box(ax, corners, color, color, 0.10 if frame else 0.20, 0.8, 14)
            reward = item["reward"]
            if reward.collision_step is not None and reward.collision_step < len(traj):
                hit = traj[reward.collision_step]
                ax.plot(hit[0], hit[1], marker="X", ms=11, color="#b91c1c", mec="white", mew=0.8, zorder=30)
            if reward.rb_crossing:
                crossing = min(len(traj) - 1, 1 if reward.collision_step is None else reward.collision_step)
                # Matplotlib does not define "!" as a marker style.  A red
                # downward triangle is a stable, legible road-border flag.
                ax.plot(traj[crossing, 0], traj[crossing, 1], marker="v", ms=9, color="#dc2626", mec="white", mew=0.6, zorder=30)

    if len(points):
        center = np.median(points, axis=0)
        span = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])))
        half = max(12.0, 0.52 * span + 4.0)
        ax.set_xlim(float(center[0] - half), float(center[0] + half))
        ax.set_ylim(float(center[1] - half), float(center[1] + half))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.16)
    ax.set_xlabel("ego-frame x (m)")
    ax.set_ylabel("ego-frame y (m)")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9, ncol=2)
    zoom_ax = ax.inset_axes([0.61, 0.61, 0.36, 0.36], zorder=40)
    _draw_ego_zoom(zoom_ax, z, results, ego_shape, min(max_neighbors, 12))
    det = rollout.rewards[0]
    best = max(rollout.rewards, key=lambda r: r.total)
    ax.set_title(
        f"Original DP AWR scene — {Path(scene_path).name}\n"
        f"det R={det.total:+.1f} · best-of-{len(results)} R={best.total:+.1f} · "
        f"red=road border · colored boxes=recorded neighbors ({tag})",
        fontsize=11,
        loc="left",
    )
    _draw_reward_panel(ax_reward, results, reward_config, subscores, hdp_components)
    _draw_table(ax_table, results, hdp_components)
    fig.suptitle(
        "Scene-level AWR evidence: candidate mode diversity is visible as paths; safety is evaluated on oriented vehicle footprints",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _dynamic_frame(
    ax,
    z: dict[str, np.ndarray],
    trajectories: np.ndarray,
    ranked_indices: np.ndarray,
    ego_shape: np.ndarray,
    frame: int,
    reward_values: list[float],
    max_neighbors: int,
    limits: tuple[float, float, float, float],
):
    ax.clear()
    _draw_map(ax, z)
    if "ego_agent_future" in z:
        gt = z["ego_agent_future"]
        valid = _valid_xy(gt[:, :2])
        if valid.sum() >= 2:
            ax.plot(gt[valid, 0], gt[valid, 1], "--", color="#15803d", lw=1.8, alpha=0.55, label="expert")
    # All candidates remain faint context; only their prefixes move.
    for rank, index in enumerate(ranked_indices):
        traj = trajectories[int(index)]
        color = plt.cm.RdYlGn(1.0 - rank / max(len(ranked_indices) - 1, 1))
        ax.plot(traj[:, 0], traj[:, 1], color=color, lw=0.65, alpha=0.15, zorder=5)
        if rank < 3:
            upto = min(frame + 1, len(traj))
            ax.plot(traj[:upto, 0], traj[:upto, 1], color=color, lw=2.1 if rank == 0 else 1.4, alpha=0.9, zorder=10)
            corners = _ego_corners(traj, ego_shape, frame)
            _draw_polygon_box(ax, corners, color, color, 0.22, 1.4, 20)
    _draw_neighbor_boxes(ax, z, frame, max_neighbors=max_neighbors, trails=False)
    _draw_polygon_box(ax, obb_corners(0.0, 0.0, 0.0, float(ego_shape[1]), float(ego_shape[2]), float(ego_shape[0])), "#111827", "#111827", 0.65, 1.2, 25)
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.14)
    ax.set_xlabel("ego-frame x (m)")
    ax.set_ylabel("ego-frame y (m)")
    best_index = int(ranked_indices[0])
    ax.set_title(f"t={frame * 0.1:04.1f}s · best candidate R={reward_values[best_index]:+.1f} · top-3 prefixes + real neighbor OBBs", loc="left")


def save_gif(
    scene_path: str,
    rollout,
    output_path: Path,
    fps: int,
    max_neighbors: int,
    max_frames: int,
) -> None:
    z = _load_aligned_npz(scene_path)
    trajectories = rollout.trajectories.detach().cpu().numpy()
    rewards = [float(reward.total) for reward in rollout.rewards]
    ranked_indices = np.argsort([-value for value in rewards])
    ego_shape = z.get("ego_shape", np.asarray([2.75, 4.34, 1.70], dtype=np.float32))
    if ego_shape.ndim > 1:
        ego_shape = ego_shape[0]
    points = _all_points(z, trajectories, max_neighbors)
    center = np.median(points, axis=0)
    span = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])))
    half = max(12.0, 0.52 * span + 4.0)
    limits = (float(center[0] - half), float(center[0] + half), float(center[1] - half), float(center[1] + half))
    n_frames = min(trajectories.shape[1], max(1, max_frames))
    frames = np.linspace(0, trajectories.shape[1] - 1, n_frames).round().astype(int)
    fig, ax = plt.subplots(figsize=(8.5, 8.0), facecolor="white")

    def draw(frame_index: int):
        _dynamic_frame(ax, z, trajectories, ranked_indices, ego_shape, int(frames[frame_index]), rewards, max_neighbors, limits)
        return (ax,)

    movie = animation.FuncAnimation(fig, draw, frames=len(frames), interval=1000 / max(fps, 1), blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    movie.save(output_path, writer=animation.PillowWriter(fps=fps), dpi=110)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--args_path", type=Path, default=None)
    parser.add_argument("--scenes", type=Path, default=None)
    parser.add_argument("--scene", action="append", default=None)
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--reward_config", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=5,
        help="original-DP stochastic solver steps; HDP/PlannerRFT uses five",
    )
    parser.add_argument("--noise_min", type=float, default=0.5)
    parser.add_argument("--noise_max", type=float, default=2.0)
    parser.add_argument(
        "--deterministic_first",
        action="store_true",
        help="use the original-DP zero-noise behavior trajectory as candidate 0",
    )
    parser.add_argument("--tag", type=str, default="original-dp-awr")
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max_gif_frames", type=int, default=40)
    parser.add_argument("--max_neighbors", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; pass --device cpu for a CPU run")
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model, model_args = load_original_dp_checkpoint(args.model_path, device, args.args_path, use_ema=True)
    reward_config = _load_reward_config(args.reward_config)
    paths = _load_scene_list(args.scenes, args.scene)
    if args.indices:
        paths = [paths[index] for index in args.indices]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rollout_config = AWRRolloutConfig(
        n_trajectories=args.K,
        sample_steps=args.sample_steps,
        noise_scale_range=(args.noise_min, args.noise_max),
        deterministic_first=args.deterministic_first,
    )
    manifest = []
    for index, scene_path in enumerate(paths):
        print(f"Visualizing {index + 1}/{len(paths)}: {scene_path}")
        data = load_scene(scene_path, device)
        rollout = rollout_and_score_scene(model, model_args, data, rollout_config, reward_config, device)
        stem = f"scene_{index:03d}"
        json_path = args.output_dir / f"{stem}.json"
        json_path.write_text(json.dumps(rollout_to_json(rollout, scene_path), indent=2) + "\n")
        np.savez_compressed(
            args.output_dir / f"{stem}.npz",
            trajectories=rollout.trajectories.detach().cpu().numpy(),
            noise_scales=rollout.noise_scales.detach().cpu().numpy(),
            weights=rollout.weights.detach().cpu().numpy(),
        )
        draw_scene_figure(scene_path, rollout, reward_config, args.output_dir / f"{stem}.png", args.tag, args.max_neighbors)
        gif_path = None
        if args.gif:
            gif_path = args.output_dir / f"{stem}.gif"
            save_gif(scene_path, rollout, gif_path, args.fps, args.max_neighbors, args.max_gif_frames)
        manifest.append({"scene_path": scene_path, "png": str(args.output_dir / f"{stem}.png"), "gif": str(gif_path) if gif_path else None, "json": str(json_path)})
        print(
            f"  det={rollout.rewards[0].total:+.2f}, "
            f"best={max(reward.total for reward in rollout.rewards):+.2f}, "
            f"valid_group={rollout.diagnostics['valid_group']:.0f}"
        )
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved visualizations under {args.output_dir}")


if __name__ == "__main__":
    main()
