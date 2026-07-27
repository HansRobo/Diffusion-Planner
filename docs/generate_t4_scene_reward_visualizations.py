"""Scene-level, data-backed counterfactual reward probes for the T4 AWR report.

The default candidate set is a counterfactual probe set constructed from real
logged scenes.  It is scored by the historical local ``compute_hdp_reward``
evaluator.  These are explicitly *not* formal AWR replay samples or evidence
of checkpoint improvement: they make individual reward semantics visible.

The optional model path loads the pre-LineEncoder-refactor checkpoint through
a local compatibility adapter. The adapter is kept here, rather than changing
model/training code, because this is an audit artifact and the working tree
contains a newer encoder ABI than the saved checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
from timm.layers import Mlp

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "t4_rl_assets"
ASSET_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_ARGS = ROOT / (
    "outputs/hdp_ego_only_route_adaln_sft20_filteredbase_afterentry_tlsmask_allveh_x10_"
    "node01_8gpu_bf16_bs512_from_base20/best_epdms_model/args.json"
)
RL_ARGS = ROOT / (
    "outputs/hdp_route_sft20_latest_rl_stable_current_g32_beta1_lr1e7_dp0_"
    "ema010_distance_red_ep6_node01/best_model/args.json"
)

SCENES = [
    {
        "id": "red_light_stop",
        "title": "Red light: stop before the line",
        "short": "RED LIGHT",
        "kind": "red",
        "path": (
            "/mnt/storage_rdma/diffusion_planner/dataset/20260623_full_sequence/"
            "prd_jt/1423_shinagawa_odaiba/valid/2025-06-12/10-51-07/"
            "10-51-07_00000000_00005415.npz"
        ),
        "claim": (
            "The expert is essentially stationary while a red route lane and an "
            "associated stop line are visible. A forward counterfactual should "
            "lose the red-light constraint score even though it gains progress."
        ),
    },
    {
        "id": "moving_crossing",
        "title": "Moving neighbor crossing the ego corridor",
        "short": "MOVING CONFLICT",
        "kind": "dynamic",
        "path": (
            "/mnt/storage_rdma/diffusion_planner/dataset/20260623_full_sequence/"
            "prd_jt/1423_shinagawa_odaiba/valid/2025-06-12/11-02-50/"
            "11-02-50_00000000_00005556.npz"
        ),
        "claim": (
            "A recorded moving neighbor approaches the ego corridor around the "
            "first second. A forward candidate should be rejected by OBB safety "
            "or TTC/occupancy, while a braking or lateral-yield candidate should "
            "retain a feasible alternative if the geometry permits it."
        ),
    },
    {
        "id": "stopped_leader",
        "title": "Stopped leader: progress must not mean tailgating",
        "short": "STOPPED LEADER",
        "kind": "follow",
        "path": (
            "/mnt/storage_rdma/diffusion_planner/dataset/20260623_full_sequence/"
            "prd_jt/1423_shinagawa_odaiba/valid/2025-06-16/15-02-05/"
            "15-02-05_00000000_00003609.npz"
        ),
        "claim": (
            "The logged ego barely moves and a stopped vehicle is about five "
            "meters ahead. This exposes the tension between anti-stopping "
            "progress and follow/occupancy safety."
        ),
    },
    {
        "id": "tight_right_turn",
        "title": "Tight right turn: lane and border are geometric",
        "short": "TIGHT TURN",
        "kind": "turn",
        "path": (
            "/mnt/storage_rdma/diffusion_planner/dataset/20260623_full_sequence/"
            "prd_jt/1423_shinagawa_odaiba/valid/2025-06-12/10-51-07/"
            "10-51-07_00000000_00001331.npz"
        ),
        "claim": (
            "The logged route turns sharply right. A smooth lateral drift or "
            "straight-line shortcut can make the lane/border score fail even "
            "when it appears to make more progress in a bird's-eye plot."
        ),
    },
]

COMPONENTS = [
    ("safety", "Safety"),
    ("risk", "Risk"),
    ("follow", "Follow"),
    ("lane", "Lane"),
    ("progress", "Progress"),
    ("red_light", "Red light"),
    ("road_border", "Border"),
    ("comfort", "Comfort"),
]


def _display_candidate_name(row: dict) -> str:
    """Keep dashboard axes legible while preserving full names in the manifest."""
    if row.get("kind") == "model":
        suffix = str(row.get("name", "")).rsplit(" ", 1)[-1]
        return f"historical sample {suffix}"
    return str(row.get("name", "candidate"))


def _finite_xy(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points)
    if points.ndim == 1:
        points = points[None]
    points = points[..., :2]
    mask = np.isfinite(points).all(axis=-1) & (np.abs(points).sum(axis=-1) > 1e-5)
    return points[mask]


def _smoothstep(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _quintic(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def _pose_from_xy(xy: np.ndarray, fallback_heading: float = 0.0) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32)
    velocity = np.gradient(xy, axis=0)
    heading = np.arctan2(velocity[:, 1], velocity[:, 0])
    speed = np.linalg.norm(velocity, axis=-1)
    if np.all(speed < 1e-4):
        heading[:] = fallback_heading
    else:
        for i in range(len(heading)):
            if speed[i] < 1e-4:
                heading[i] = heading[i - 1] if i else fallback_heading
        heading = np.unwrap(heading)
    return np.column_stack((xy, np.cos(heading), np.sin(heading))).astype(np.float32)


def _time_resample(xy: np.ndarray, factor: float) -> np.ndarray:
    """Time-warp a logged path; factor > 1 reaches the future faster."""
    xy = np.asarray(xy, dtype=np.float32)
    source = np.linspace(0.0, len(xy) - 1, len(xy))
    query = np.clip(source * float(factor), 0.0, len(xy) - 1)
    return np.column_stack(
        [np.interp(query, source, xy[:, axis]) for axis in range(xy.shape[1])]
    ).astype(np.float32)


def _forward_path(
    scene: dict,
    target_distance: float,
    lateral_offset: float = 0.0,
) -> np.ndarray:
    data = scene["data"]
    state = np.asarray(data["ego_current_state"]).reshape(-1)
    heading = math.atan2(float(state[3]), float(state[2]))
    u = (np.arange(80, dtype=np.float32) + 1.0) / 80.0
    longitudinal = float(target_distance) * _quintic(u)
    lateral = float(lateral_offset) * _quintic(u)
    c, s = math.cos(heading), math.sin(heading)
    xy = np.column_stack((longitudinal * c - lateral * s, longitudinal * s + lateral * c))
    return _pose_from_xy(xy, heading)


def _offset_path(xy: np.ndarray, offset: float) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32)
    tangent = np.gradient(xy, axis=0)
    tangent = tangent / np.linalg.norm(tangent, axis=-1, keepdims=True).clip(min=1e-5)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    weight = _quintic((np.arange(len(xy), dtype=np.float32) + 1.0) / len(xy))
    return _pose_from_xy(xy + normal * (float(offset) * weight[:, None]))


def _candidate_set(scene: dict) -> list[dict]:
    data = scene["data"]
    expert = _pose_from_xy(data["ego_agent_future"][:, :2])
    xy = expert[:, :2]
    kind = scene["kind"]
    candidates = [{"name": "logged expert", "kind": "expert", "trajectory": expert}]

    def add(name, trajectory):
        candidates.append({"name": name, "kind": "counterfactual", "trajectory": trajectory})

    if kind == "red":
        add("hard hold", _forward_path(scene, 0.05))
        add("stop before line", _forward_path(scene, 0.45))
        add("roll toward line", _forward_path(scene, 2.4))
        add("go on red", _forward_path(scene, 12.0))
        add("fast go on red", _forward_path(scene, 22.0))
        add("lateral left", _forward_path(scene, 6.0, 2.7))
        add("lateral right", _forward_path(scene, 6.0, -2.7))
    elif kind == "dynamic":
        add("slow approach", _pose_from_xy(_time_resample(xy, 0.55)))
        add("fast approach", _pose_from_xy(_time_resample(xy, 1.45)))
        add("brake early", _forward_path(scene, 1.4))
        add("yield left", _offset_path(xy, -2.6))
        add("yield right", _offset_path(xy, 2.6))
        add("straight fast", _forward_path(scene, 48.0))
    elif kind == "follow":
        add("hard hold", _forward_path(scene, 0.05))
        add("safe crawl", _forward_path(scene, 1.2))
        add("slow expert path", _pose_from_xy(_time_resample(xy, 0.55)))
        add("progress push", _forward_path(scene, 8.0))
        add("tailgate", _forward_path(scene, 16.0))
        add("left bypass", _forward_path(scene, 8.0, 2.7))
        add("right bypass", _forward_path(scene, 8.0, -2.7))
    else:
        add("slow turn", _pose_from_xy(_time_resample(xy, 0.65)))
        add("fast turn", _pose_from_xy(_time_resample(xy, 1.35)))
        add("cut inside", _offset_path(xy, -2.5))
        add("wide outside", _offset_path(xy, 2.5))
        add("straight shortcut", _forward_path(scene, 45.0))
        add("far lateral drift", _offset_path(xy, 4.0))
    return candidates


def _load_scene(spec: dict) -> dict:
    path = Path(spec["path"])
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files if key != "version"}
    scene = dict(spec)
    scene["data"] = data
    scene["path"] = str(path)
    return scene


def _make_reward_args():
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "diffusion_planner"))
    from diffusion_planner.utils.config import Config

    args = Config(str(SOURCE_ARGS))
    args.device = "cpu"
    args.ddp = False
    args.amp_dtype = "off"
    args.rl_reward_w_safety = 0.0
    args.rl_reward_w_risk = 1.0
    args.rl_reward_w_follow = 3.0
    args.rl_reward_w_lane = 2.5
    args.rl_reward_w_progress = 3.0
    args.rl_reward_w_road_border = 0.0
    args.rl_behavior_gate = "safety"
    args.rl_occupancy_use_road_border = True
    args.rl_stationary_progress_mode = "distance"
    args.rl_red_light_constraint = True
    args.rl_reward_dt = 0.1
    args.rl_ttc_critical_s = 0.5
    args.rl_ttc_safe_s = 3.0
    args.rl_thw_critical_s = 0.5
    args.rl_thw_safe_s = 2.0
    args.rl_occupancy_critical_m = 0.25
    args.rl_occupancy_safe_m = 2.0
    args.rl_occupancy_speed_gain_s = 0.1
    args.rl_lane_half_width_m = 1.75
    args.rl_leader_lateral_margin_m = 0.75
    return args


def _raw_tensors(scene: dict) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "diffusion_planner"))
    from diffusion_planner.hdp_rl_epoch import _neighbor_future_world
    from diffusion_planner.train_epoch import heading_to_cos_sin

    data = scene["data"]
    raw = {}
    # Keep all numeric NPZ fields for observation_normalizer. The reward only
    # needs a subset, but the model's saved normalizer also contains speed-limit
    # and availability side channels.
    for key, value in data.items():
        if not isinstance(value, np.ndarray) or value.dtype.kind not in "biuf":
            continue
        tensor = torch.from_numpy(np.asarray(value).copy())
        tensor = tensor.bool() if key.endswith("has_speed_limit") else tensor.float()
        raw[key] = tensor.unsqueeze(0)
    # Ego-frame origin poses are valid stationary states, including in the
    # current/goal/future fields; only known neighbor padding uses zero masking.
    raw["ego_agent_past"] = heading_to_cos_sin(raw["ego_agent_past"])
    raw["goal_pose"] = heading_to_cos_sin(raw["goal_pose"])
    raw["ego_agent_future"] = heading_to_cos_sin(raw["ego_agent_future"])
    # Known contract for the legacy T4 full-sequence files used here: the raw
    # neighbor future is one frame early.  Map raw[t+1] -> aligned[t] and
    # zero-fill the final frame, exactly as in the formal original-DP AWR run.
    neighbor_future = raw["neighbor_agents_future"].clone()
    aligned_neighbor_future = torch.zeros_like(neighbor_future)
    aligned_neighbor_future[..., :-1, :] = neighbor_future[..., 1:, :]
    raw["neighbor_agents_future"] = aligned_neighbor_future
    return raw, _neighbor_future_world(raw["neighbor_agents_future"])


def _signed_clearance(
    trajectories: np.ndarray, raw: dict[str, torch.Tensor], neighbors: torch.Tensor
) -> np.ndarray:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "diffusion_planner"))
    from diffusion_planner.hdp_rl_utils import _scene_neighbors

    from planner_metrics.subscores import compute_ego_neighbor_signed_clearance

    future, shapes, valid, _, _ = _scene_neighbors(neighbors[0], raw["neighbor_agents_past"][0])
    if future.shape[0] == 0:
        return np.full((len(trajectories), trajectories.shape[1]), np.inf, dtype=np.float32)
    ego = torch.from_numpy(trajectories).float()
    clearance = compute_ego_neighbor_signed_clearance(
        ego,
        raw["ego_shape"][0, :3],
        future,
        shapes,
        valid,
    )
    return clearance.amin(dim=1).detach().cpu().numpy()


def _score_candidates(scene: dict, candidates: list[dict], args) -> list[dict]:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "diffusion_planner"))
    from diffusion_planner.hdp_rl_utils import compute_hdp_reward

    raw, neighbors = _raw_tensors(scene)
    trajectories = np.stack([item["trajectory"] for item in candidates]).astype(np.float32)
    signed_clearance = _signed_clearance(trajectories, raw, neighbors)
    rows = []
    for index, item in enumerate(candidates):
        reward, metrics = compute_hdp_reward(
            torch.from_numpy(item["trajectory"][None]).float(),
            raw,
            neighbors,
            num_scenes=1,
            n=1,
            args=args,
        )
        row = {
            "index": index,
            "name": item["name"],
            "kind": item["kind"],
            "reward": float(reward[0]),
            "min_signed_clearance_m": float(np.nanmin(signed_clearance[index])),
            "endpoint_x_m": float(item["trajectory"][-1, 0]),
            "endpoint_y_m": float(item["trajectory"][-1, 1]),
        }
        for key, _label in COMPONENTS:
            row[key] = float(metrics.get(f"reward_{key}_score", torch.tensor(1.0)))
        for key in (
            "collision_active",
            "collision_rear",
            "ttc_zero_fraction",
            "thw_zero_fraction",
            "occupancy_zero_fraction",
            "red_light_constraint_active_fraction",
            "red_light_violation_fraction",
            "road_border_violation_fraction",
            "progress_raw",
            "progress_ratio",
            "leader_fraction",
        ):
            metric_key = (
                f"reward_{key}"
                if key.endswith("fraction") or key.endswith("active") or key.endswith("rear")
                else f"reward_{key}_score"
            )
            if metric_key in metrics:
                row[key] = float(metrics[metric_key])
        rows.append(row)
    return rows


def _candidate_colors(rows: list[dict]) -> tuple[dict[int, tuple], dict[int, int]]:
    rewards = np.asarray([row["reward"] for row in rows], dtype=float)
    order = np.argsort(-rewards)
    ranks = {int(index): int(rank) for rank, index in enumerate(order)}
    norm = mcolors.Normalize(vmin=float(rewards.min()), vmax=float(rewards.max() + 1e-8))
    cmap = plt.get_cmap("plasma")
    colors = {int(i): cmap(norm(float(rows[i]["reward"]))) for i in range(len(rows))}
    return colors, ranks


def _draw_oriented_box(
    ax,
    x: float,
    y: float,
    heading: float,
    length: float,
    width: float,
    color,
    *,
    alpha: float = 0.75,
    lw: float = 1.2,
    rear_axle: bool = False,
    wheelbase: float | None = None,
    zorder: int = 10,
    fill: bool = True,
):
    if not np.isfinite([x, y, heading, length, width]).all():
        return
    center_x = 0.5 * abs(float(wheelbase or 0.0)) if rear_axle else 0.0
    corners = np.asarray(
        [
            [center_x - length / 2.0, -width / 2.0],
            [center_x + length / 2.0, -width / 2.0],
            [center_x + length / 2.0, width / 2.0],
            [center_x - length / 2.0, width / 2.0],
        ],
        dtype=float,
    )
    c, s = math.cos(heading), math.sin(heading)
    rot = np.asarray([[c, -s], [s, c]])
    corners = corners @ rot.T + np.asarray([x, y])
    ax.add_patch(
        Polygon(
            corners,
            closed=True,
            facecolor=color if fill else "none",
            edgecolor=color,
            linewidth=lw,
            alpha=alpha,
            zorder=zorder,
            joinstyle="round",
        )
    )


def _draw_map(ax, data: dict, extent: tuple[float, float, float, float]):
    for lane in np.asarray(data.get("lanes", np.zeros((0, 1, 2)))):
        points = lane[:, :2]
        valid = (np.abs(points).sum(-1) > 1e-5) & np.isfinite(points).all(-1)
        if valid.sum() < 2:
            continue
        ax.plot(points[valid, 0], points[valid, 1], color="#cbd5e1", lw=1.0, alpha=0.30, zorder=1)
        if lane.shape[-1] >= 8:
            for offset in (lane[:, 4:6], lane[:, 6:8]):
                boundary = points + offset
                ax.plot(
                    boundary[valid, 0],
                    boundary[valid, 1],
                    color="#94a3b8",
                    lw=0.45,
                    alpha=0.20,
                    zorder=1,
                )
    for lane in np.asarray(data.get("route_lanes", np.zeros((0, 1, 2)))):
        points = lane[:, :2]
        valid = (np.abs(points).sum(-1) > 1e-5) & np.isfinite(points).all(-1)
        if valid.sum() >= 2:
            ax.plot(
                points[valid, 0], points[valid, 1], color="#2563eb", lw=2.7, alpha=0.72, zorder=2
            )
    for polygon in np.asarray(data.get("polygons", np.zeros((0, 1, 2)))):
        points = polygon[:, :2]
        valid = (np.abs(points).sum(-1) > 1e-5) & np.isfinite(points).all(-1)
        if valid.sum() >= 3:
            ax.add_patch(
                Polygon(
                    points[valid],
                    closed=True,
                    facecolor="#e2e8f0",
                    edgecolor="#cbd5e1",
                    alpha=0.12,
                    linewidth=0.6,
                    zorder=0,
                )
            )
    for line in np.asarray(data.get("line_strings", np.zeros((0, 1, 4)))):
        points = line[:, :2]
        valid = (np.abs(points).sum(-1) > 1e-5) & np.isfinite(points).all(-1)
        if valid.sum() < 2:
            continue
        if line.shape[-1] >= 4 and np.any(line[:, 3] > 0.5):
            ax.plot(
                points[valid, 0], points[valid, 1], color="#dc2626", lw=1.3, alpha=0.68, zorder=3
            )
        elif line.shape[-1] >= 3 and np.any(line[:, 2] > 0.5):
            ax.plot(
                points[valid, 0],
                points[valid, 1],
                color="#f59e0b",
                lw=2.2,
                linestyle=(0, (5, 3)),
                alpha=0.85,
                zorder=3,
            )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])


def _scene_extent(scene: dict, candidates: list[dict]) -> tuple[float, float, float, float]:
    data = scene["data"]
    points = [_finite_xy(data["ego_agent_future"][:, :2])]
    points.extend(_finite_xy(item["trajectory"][:, :2]) for item in candidates)
    nf = np.asarray(data.get("neighbor_agents_future", np.zeros((0, 1, 3))))
    if nf.ndim == 3:
        points.append(_finite_xy(nf.reshape(-1, nf.shape[-1])[:, :2]))
    nb = np.asarray(data.get("neighbor_agents_past", np.zeros((0, 1, 11))))
    if nb.ndim == 3:
        points.append(_finite_xy(nb[:, -1, :2]))
    merged = np.concatenate([p for p in points if len(p)], axis=0)
    xmin, ymin = merged.min(axis=0) - 6.0
    xmax, ymax = merged.max(axis=0) + 6.0
    xmin, xmax = min(xmin, -8.0), max(xmax, 8.0)
    ymin, ymax = min(ymin, -8.0), max(ymax, 8.0)
    width, height = xmax - xmin, ymax - ymin
    if width / height > 2.2:
        extra = (width / 2.2 - height) / 2.0
        ymin, ymax = ymin - extra, ymax + extra
    elif height / width > 2.2:
        extra = (height / 2.2 - width) / 2.0
        xmin, xmax = xmin - extra, xmax + extra
    return float(xmin), float(xmax), float(ymin), float(ymax)


def _select_neighbors(data: dict, extent: tuple[float, float, float, float], limit: int = 12):
    nf = np.asarray(data.get("neighbor_agents_future", np.zeros((0, 1, 3))))
    nb = np.asarray(data.get("neighbor_agents_past", np.zeros((0, 1, 11))))
    if nf.ndim != 3 or nb.ndim != 3:
        return []
    valid = (np.abs(nf[:, :, :2]).sum(-1) > 1e-5).any(1)
    distance = np.linalg.norm(nb[:, -1, :2], axis=-1)
    return [int(i) for i in np.argsort(distance) if valid[i] and distance[i] < 70.0][:limit]


def _draw_neighbors(ax, data: dict, t: int, extent, selected=None, alpha_path=0.26):
    nf = np.asarray(data.get("neighbor_agents_future", np.zeros((0, 1, 3))))
    nb = np.asarray(data.get("neighbor_agents_past", np.zeros((0, 1, 11))))
    if nf.ndim != 3 or nb.ndim != 3:
        return
    if selected is None:
        selected = _select_neighbors(data, extent)
    palette = ["#64748b", "#0f766e", "#7c3aed", "#be123c", "#0369a1"]
    for rank, index in enumerate(selected):
        pose = nf[index]
        valid = (np.abs(pose[:, :2]).sum(-1) > 1e-5) & np.isfinite(pose).all(-1)
        if valid.sum() >= 2:
            ax.plot(
                pose[valid, 0],
                pose[valid, 1],
                color=palette[rank % len(palette)],
                lw=0.8,
                alpha=alpha_path,
                zorder=4,
            )
        ti = min(t, len(pose) - 1)
        if not valid[ti]:
            continue
        row = pose[ti]
        current = nb[index, -1]
        width = abs(float(current[6])) if abs(float(current[6])) > 1e-3 else 1.8
        length = abs(float(current[7])) if abs(float(current[7])) > 1e-3 else 4.5
        _draw_oriented_box(
            ax,
            float(row[0]),
            float(row[1]),
            float(row[2]),
            length,
            width,
            palette[rank % len(palette)],
            alpha=0.34,
            lw=0.9,
            zorder=7,
        )


def _draw_trajectory_bev(
    ax,
    scene: dict,
    candidates: list[dict],
    rows: list[dict],
    *,
    title: str | None = None,
    time_index: int = 40,
    extent=None,
    selected_neighbors=None,
):
    data = scene["data"]
    if extent is None:
        extent = _scene_extent(scene, candidates)
    _draw_map(ax, data, extent)
    _draw_neighbors(ax, data, time_index, extent, selected_neighbors)
    colors, ranks = _candidate_colors(rows)
    expert = candidates[0]["trajectory"]
    ax.plot(
        expert[:, 0],
        expert[:, 1],
        color="#16a34a",
        lw=2.6,
        linestyle=(0, (6, 3)),
        label="logged expert",
        zorder=8,
    )
    ticks = [0, 20, 40, 60, 79]
    ax.scatter(
        expert[ticks, 0],
        expert[ticks, 1],
        s=26,
        facecolor="#ffffff",
        edgecolor="#16a34a",
        linewidth=1.3,
        zorder=9,
    )
    for index in sorted(range(len(candidates)), key=lambda i: rows[i]["reward"]):
        if index == 0:
            continue
        trajectory = candidates[index]["trajectory"]
        rank = ranks[index]
        is_model = rows[index]["kind"] == "model"
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color="#ea580c" if is_model else colors[index],
            lw=1.65 if is_model else (1.25 if rank > 1 else 2.4),
            alpha=0.70 if is_model else (0.34 if rank > 1 else 0.92),
            linestyle=(0, (3, 2)) if is_model else "-",
            label="historical checkpoint sample"
            if is_model
            and index
            == next(
                (i for i, row in enumerate(rows) if row["kind"] == "model"),
                index,
            )
            else None,
            zorder=6,
        )
    best = int(np.argmax([row["reward"] for row in rows]))
    worst = int(np.argmin([row["reward"] for row in rows]))
    for index, color, label in (
        (best, "#0f766e", f"best: {rows[best]['name']}"),
        (worst, "#dc2626", f"worst: {rows[worst]['name']}"),
    ):
        trajectory = candidates[index]["trajectory"]
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=color,
            lw=2.6,
            alpha=0.94,
            label=label,
            zorder=10,
            path_effects=[
                patheffects.Stroke(linewidth=4.4, foreground="white"),
                patheffects.Normal(),
            ],
        )
        sample_index = min(time_index, len(trajectory) - 1)
        heading = math.atan2(float(trajectory[sample_index, 3]), float(trajectory[sample_index, 2]))
        _draw_oriented_box(
            ax,
            float(trajectory[sample_index, 0]),
            float(trajectory[sample_index, 1]),
            heading,
            float(data["ego_shape"][1]),
            float(data["ego_shape"][2]),
            color,
            alpha=0.24,
            lw=1.8,
            rear_axle=True,
            wheelbase=float(data["ego_shape"][0]),
            zorder=11,
        )
        ax.scatter(
            [trajectory[sample_index, 0]],
            [trajectory[sample_index, 1]],
            s=42,
            color=color,
            edgecolor="white",
            linewidth=1.1,
            zorder=12,
        )
    _draw_oriented_box(
        ax,
        0.0,
        0.0,
        0.0,
        float(data["ego_shape"][1]),
        float(data["ego_shape"][2]),
        "#2563eb",
        alpha=0.25,
        lw=1.5,
        rear_axle=True,
        wheelbase=float(data["ego_shape"][0]),
        zorder=12,
    )
    ax.scatter([0], [0], color="#2563eb", s=30, zorder=13)
    if title:
        ax.set_title(title, loc="left", fontsize=14, fontweight="bold", color="#0f172a")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#cbd5e1", linewidth=0.5, alpha=0.35)
    ax.set_xlabel("ego-frame x (m)")
    ax.set_ylabel("ego-frame y (m)")
    ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.94)
    ax.text(
        0.01,
        0.01,
        f"boxes at t={time_index / 10:.1f}s  |  red=border  yellow dash=stop line  blue=route",
        transform=ax.transAxes,
        fontsize=8,
        color="#475569",
        va="bottom",
    )
    return extent


def _plot_dashboard(scene: dict, candidates: list[dict], rows: list[dict], out_path: Path):
    extent = _scene_extent(scene, candidates)
    fig = plt.figure(figsize=(18, 12), facecolor="#f8fafc")
    grid = GridSpec(2, 2, figure=fig, width_ratios=[1.35, 1.0], height_ratios=[1.12, 0.88])
    ax_bev = fig.add_subplot(grid[0, 0])
    _draw_trajectory_bev(
        ax_bev,
        scene,
        candidates,
        rows,
        title=f"{scene['short']}  ·  real T4 geometry",
        time_index=40,
        extent=extent,
    )

    ax_bar = fig.add_subplot(grid[0, 1])
    order = np.argsort([row["reward"] for row in rows])
    values = [rows[index]["reward"] for index in order]
    labels = [_display_candidate_name(rows[index]) for index in order]
    bar_colors = [
        "#16a34a" if index == 0 else "#f59e0b" if rows[index]["kind"] == "model" else "#8b5cf6"
        for index in order
    ]
    ax_bar.barh(np.arange(len(order)), values, color=bar_colors, alpha=0.86)
    ax_bar.set_yticks(np.arange(len(order)), labels, fontsize=8)
    ax_bar.set_xlabel("diagnostic reward score (higher is preferred)")
    ax_bar.set_title(
        "Counterfactual ranking from the diagnostic evaluator",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )
    ax_bar.grid(axis="x", color="#cbd5e1", alpha=0.45)
    value_pad = max(float(np.max(np.abs(values))) * 0.012, 0.01)
    for y, value in enumerate(values):
        ax_bar.text(value + value_pad, y, f"{value:.2f}", va="center", fontsize=8, color="#334155")
    ax_bar.axvline(0, color="#64748b", lw=0.8)

    ax_heat = fig.add_subplot(grid[1, 0])
    matrix = np.asarray([[row[key] for key, _ in COMPONENTS] for row in rows])
    image = ax_heat.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
    ax_heat.set_yticks(
        np.arange(len(rows)),
        [_display_candidate_name(row) for row in rows],
        fontsize=8,
    )
    ax_heat.set_xticks(
        np.arange(len(COMPONENTS)),
        [label for _, label in COMPONENTS],
        rotation=28,
        ha="right",
        fontsize=8,
    )
    ax_heat.set_title(
        "Why the ranking changes: normalized reward components",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )
    for r, row_values in enumerate(matrix):
        for c, value in enumerate(row_values):
            ax_heat.text(
                c, r, f"{value:.2f}", ha="center", va="center", fontsize=7, color="#0f172a"
            )
    fig.colorbar(image, ax=ax_heat, fraction=0.025, pad=0.02, label="component score")

    ax_time = fig.add_subplot(grid[1, 1])
    best = int(np.argmax([row["reward"] for row in rows]))
    worst = int(np.argmin([row["reward"] for row in rows]))
    raw, neighbors = _raw_tensors(scene)
    clearance = _signed_clearance(
        np.stack([item["trajectory"] for item in candidates]).astype(np.float32),
        raw,
        neighbors,
    )
    # The geometry backend uses a large finite sentinel when no valid neighbor
    # exists at a time step (the +1-aligned final frame is deliberately
    # zero-filled).  Treat that as missing context, never as a 1e6 m clearance.
    clearance_plot = clearance.astype(float, copy=True)
    clearance_plot[(~np.isfinite(clearance_plot)) | (np.abs(clearance_plot) >= 1e5)] = np.nan
    for index, color, label in (
        (0, "#16a34a", "logged expert"),
        (best, "#0f766e", f"best: {rows[best]['name']}"),
        (worst, "#dc2626", f"worst: {rows[worst]['name']}"),
    ):
        ax_time.plot(np.arange(80) / 10.0, clearance_plot[index], color=color, lw=2.1, label=label)
    ax_time.axhline(0.0, color="#dc2626", linestyle="--", lw=1.0)
    ax_time.fill_between(np.arange(80) / 10.0, -3.0, 0.0, color="#fecaca", alpha=0.35)
    finite_clearance = clearance_plot[np.isfinite(clearance_plot)]
    y_min = float(np.min(finite_clearance)) if finite_clearance.size else -1.0
    y_max = float(np.max(finite_clearance)) if finite_clearance.size else 4.0
    ax_time.set_ylim(min(-3.0, y_min - 0.5), max(4.0, y_max + 0.5))
    ax_time.set_xlabel("time (s)")
    ax_time.set_ylabel("minimum signed OBB clearance (m)")
    ax_time.set_title(
        "Safety geometry over time (negative = overlap)",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )
    ax_time.grid(color="#cbd5e1", alpha=0.45)
    ax_time.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"STRUCTURED COUNTERFACTUAL REWARD PROBE  ·  {scene['title']}",
        x=0.04,
        y=0.995,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#0f172a",
    )
    fig.text(
        0.04,
        0.008,
        "NOT FORMAL AWR TRAINING EVIDENCE. Green dashed = logged expert. "
        "Purple/colored paths = structured counterfactual probes; "
        + (
            "orange dashed paths = historical checkpoint samples. "
            if any(row["kind"] == "model" for row in rows)
            else ""
        )
        + "all are scored by the exact local reward code. "
        + scene["claim"],
        fontsize=8.5,
        color="#475569",
        ha="left",
        va="bottom",
    )
    fig.savefig(out_path, dpi=190, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_bev(scene: dict, candidates: list[dict], rows: list[dict], out_path: Path):
    fig, ax = plt.subplots(figsize=(13, 9), facecolor="#f8fafc")
    _draw_trajectory_bev(
        ax,
        scene,
        candidates,
        rows,
        title=f"{scene['short']}  ·  {scene['title']}",
        time_index=40,
    )
    fig.text(
        0.02,
        0.012,
        "Evidence tier C / not formal AWR output. Data: local T4 NPZ with neighbor +1 alignment. "
        "Ego OBB uses the scene wheelbase; purple paths are structured reward probes.",
        fontsize=8.5,
        color="#475569",
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _install_legacy_line_encoder():
    """Make the current model class load checkpoints from before PR #228."""
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "diffusion_planner"))
    from diffusion_planner.model.module import encoder as encoder_module
    from diffusion_planner.model.module.mixer import MixerBlock

    class LegacyLineEncoder(nn.Module):
        def __init__(
            self,
            line_len,
            class_type,
            drop_path_rate,
            hidden_dim,
            depth,
            num_types=None,
            point_dim=None,
        ):
            super().__init__()
            self._class_type = class_type
            if point_dim is None:
                point_dim = 2 + int(num_types or 1)
            self.channel_pre_project = Mlp(
                in_features=point_dim + 2,
                hidden_features=128,
                out_features=128,
                act_layer=nn.GELU,
                drop=0.0,
            )
            self.token_pre_project = Mlp(
                in_features=line_len,
                hidden_features=64,
                out_features=64,
                act_layer=nn.GELU,
                drop=0.0,
            )
            self.blocks = nn.ModuleList([MixerBlock(64, 128, drop_path_rate) for _ in range(depth)])
            self.norm = nn.LayerNorm(128)
            self.emb_project = Mlp(
                in_features=128,
                hidden_features=hidden_dim,
                out_features=hidden_dim,
                drop=drop_path_rate,
            )

        def forward(self, x):
            batch, pieces, points, _dim = x.shape
            valid_pt = torch.sum(torch.ne(x, 0), dim=-1) != 0
            pair_valid = (valid_pt[:, :, 1:] & valid_pt[:, :, :-1]).to(x.dtype)
            diff_x = (x[:, :, 1:, 0] - x[:, :, :-1, 0]) * pair_valid
            diff_y = (x[:, :, 1:, 1] - x[:, :, :-1, 1]) * pair_valid
            diff_x = torch.cat([diff_x, torch.zeros_like(diff_x[:, :, :1])], dim=2)
            diff_y = torch.cat([diff_y, torch.zeros_like(diff_y[:, :, :1])], dim=2)
            diff_x = diff_x.view(batch, pieces, points, 1)
            diff_y = diff_y.view(batch, pieces, points, 1)
            x = torch.cat([x[..., :2], diff_x, diff_y, x[..., 2:]], dim=-1)

            valid = valid_pt.to(x.dtype).unsqueeze(-1)
            count = valid.sum(dim=2).clamp(min=1.0)
            center_xy = (x[..., :2] * valid).sum(dim=2) / count
            mean_diff = (x[..., 2:4] * valid).sum(dim=2) / count
            direction_norm = mean_diff.norm(dim=-1, keepdim=True)
            unit = mean_diff / direction_norm.clamp(min=1e-6)
            default_direction = torch.zeros_like(unit)
            default_direction[..., 0] = 1.0
            unit = torch.where(direction_norm > 1e-6, unit, default_direction)
            position = torch.cat([center_xy, unit], dim=-1)
            position = encoder_module.add_class_type(position, self._class_type)

            mask = ~valid_pt.any(dim=-1)
            valid_indices = ~mask.view(-1)
            x = x.view(batch * pieces, points, -1)
            x = torch.where(valid_indices.view(-1, 1, 1), x, torch.zeros_like(x))
            x = self.channel_pre_project(x).permute(0, 2, 1)
            x = self.token_pre_project(x).permute(0, 2, 1)
            for block in self.blocks:
                x = block(x)
            x = x.mean(dim=1)
            x = self.emb_project(self.norm(x))
            x = x * valid_indices.float().unsqueeze(-1)
            return (
                x.view(batch, pieces, -1),
                mask.reshape(batch, -1),
                position.view(batch, pieces, -1),
            )

    encoder_module.LineEncoder = LegacyLineEncoder


def _actual_model_candidates(
    scenes: list[dict], generations: int, steps: int
) -> dict[str, list[dict]]:
    """Sample the saved RL checkpoint, returning actual open-loop trajectories."""
    if not RL_ARGS.exists():
        raise FileNotFoundError(f"RL checkpoint args not found: {RL_ARGS}")
    _install_legacy_line_encoder()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "diffusion_planner"))
    from diffusion_planner.hdp_rl_epoch import _grouped_policy_inputs
    from diffusion_planner.hdp_rl_utils import sample_group
    from diffusion_planner.model.diffusion_planner import Diffusion_Planner
    from diffusion_planner.utils.config import Config

    cfg = Config(str(RL_ARGS))
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.ddp = False
    cfg.amp_dtype = "off"
    cfg.diffusion_sample_steps = int(steps)
    # These checkpoints predate the configurable 21-frame ego window.
    cfg.ego_history_frames = 6
    device = torch.device(cfg.device)
    checkpoint = RL_ARGS.parent / "best_model.pth"
    model = Diffusion_Planner(cfg).to(device).eval()
    state_package = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state_package.get("ema_state_dict", state_package.get("model", state_package))
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
            f"unexpected checkpoint ABI mismatch: missing={missing}, unexpected={unexpected}"
        )

    raw_scenes = [_raw_tensors(scene)[0] for scene in scenes]
    raw = {key: torch.cat([item[key] for item in raw_scenes], dim=0) for key in raw_scenes[0]}
    norm = cfg.observation_normalizer(raw)
    grouped = _grouped_policy_inputs(norm, generations, True)
    generator = torch.Generator(device=device).manual_seed(20260715)
    rollouts = sample_group(
        model,
        grouped,
        float(getattr(cfg, "rl_eval_noise_scale", 0.5)),
        device,
        scene_norm_inputs=norm,
        group_size=generations,
        sample_steps=int(steps),
        generator=generator,
    )
    rollouts = rollouts.detach().cpu().numpy().reshape(len(scenes), generations, 80, 4)
    return {
        scene["id"]: [
            {
                "name": f"historical checkpoint sample {index + 1}",
                "kind": "model",
                "trajectory": rollouts[scene_index, index].astype(np.float32),
            }
            for index in range(generations)
        ]
        for scene_index, scene in enumerate(scenes)
    }


def _render_gif(scene: dict, candidates: list[dict], rows: list[dict], out_path: Path):
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        imageio = None
        try:
            from PIL import Image
        except Exception:
            print(f"GIF skipped: imageio/Pillow unavailable ({exc})")
            return False
    extent = _scene_extent(scene, candidates)
    selected = _select_neighbors(scene["data"], extent)
    best = int(np.argmax([row["reward"] for row in rows]))
    worst = int(np.argmin([row["reward"] for row in rows]))
    frames = []
    for t in range(0, 80, 2):
        fig, ax = plt.subplots(figsize=(10, 7), facecolor="#f8fafc")
        _draw_map(ax, scene["data"], extent)
        _draw_neighbors(ax, scene["data"], t, extent, selected, alpha_path=0.18)
        expert = candidates[0]["trajectory"]
        ax.plot(expert[:, 0], expert[:, 1], color="#16a34a", lw=1.7, linestyle=(0, (5, 3)))
        for index, color, label in (
            (best, "#0f766e", f"best: {rows[best]['name']}"),
            (worst, "#dc2626", f"worst: {rows[worst]['name']}"),
        ):
            trajectory = candidates[index]["trajectory"]
            ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, lw=2.1, alpha=0.9, label=label)
            pose = trajectory[t]
            heading = math.atan2(float(pose[3]), float(pose[2]))
            _draw_oriented_box(
                ax,
                float(pose[0]),
                float(pose[1]),
                heading,
                float(scene["data"]["ego_shape"][1]),
                float(scene["data"]["ego_shape"][2]),
                color,
                alpha=0.35,
                lw=1.6,
                rear_axle=True,
                wheelbase=float(scene["data"]["ego_shape"][0]),
                zorder=12,
            )
        _draw_oriented_box(
            ax,
            0.0,
            0.0,
            0.0,
            float(scene["data"]["ego_shape"][1]),
            float(scene["data"]["ego_shape"][2]),
            "#2563eb",
            alpha=0.24,
            lw=1.5,
            rear_axle=True,
            wheelbase=float(scene["data"]["ego_shape"][0]),
            zorder=13,
        )
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="#cbd5e1", linewidth=0.5, alpha=0.30)
        ax.set_xlabel("ego-frame x (m)")
        ax.set_ylabel("ego-frame y (m)")
        ax.set_title(
            f"PROBE / NOT FORMAL AWR  ·  {scene['short']}  ·  t={t / 10.0:.1f}s",
            loc="left",
            fontsize=13,
            fontweight="bold",
        )
        ax.legend(loc="upper left", fontsize=8, framealpha=0.94)
        ax.text(
            0.99,
            0.02,
            "neighbor motion = recorded future (+1 aligned)\n"
            "candidate ego motion = open-loop probe",
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
    if imageio is not None:
        imageio.mimsave(out_path, frames, duration=0.10, loop=0)
    else:
        images = [Image.fromarray(frame) for frame in frames]
        images[0].save(
            out_path,
            format="GIF",
            save_all=True,
            append_images=images[1:],
            duration=100,
            loop=0,
            disposal=2,
        )
    return True


def _update_manifest(records: dict):
    manifest_path = ASSET_DIR / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        manifest = {}
    manifest["scene_reward_evidence"] = records
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gif",
        action="store_true",
        help="write animated replays for every selected scene; the dynamic scene is always animated",
    )
    parser.add_argument(
        "--run-model",
        action="store_true",
        help="also append actual samples from the saved RL checkpoint",
    )
    parser.add_argument("--model-generations", type=int, default=4)
    parser.add_argument("--model-steps", type=int, default=2)
    parser.add_argument(
        "--model-scenes",
        default="red_light_stop,moving_crossing",
        help="comma-separated scene ids for actual checkpoint sampling",
    )
    args_cli = parser.parse_args()
    reward_args = _make_reward_args()
    records = {}
    loaded = []
    for spec in SCENES:
        scene = _load_scene(spec)
        candidates = _candidate_set(scene)
        rows = _score_candidates(scene, candidates, reward_args)
        scene["candidates"] = candidates
        scene["rows"] = rows
        dashboard_path = ASSET_DIR / f"t4_scene_{scene['id']}_dashboard.png"
        bev_path = ASSET_DIR / f"t4_scene_{scene['id']}_bev.png"
        _plot_dashboard(scene, candidates, rows, dashboard_path)
        _plot_bev(scene, candidates, rows, bev_path)
        gif_path = ASSET_DIR / f"t4_scene_{scene['id']}.gif"
        gif_written = False
        if args_cli.gif or scene["kind"] == "dynamic":
            gif_written = _render_gif(scene, candidates, rows, gif_path)
        records[scene["id"]] = {
            "title": scene["title"],
            "kind": scene["kind"],
            "npz_path": scene["path"],
            "candidate_source": "structured_counterfactual_reward_probe",
            "claim": scene["claim"],
            "assets": {
                "dashboard": dashboard_path.name,
                "bev": bev_path.name,
                "gif": gif_path.name if gif_written else None,
            },
            "candidates": rows,
        }
        loaded.append(scene)
        best = rows[int(np.argmax([row["reward"] for row in rows]))]
        worst = rows[int(np.argmin([row["reward"] for row in rows]))]
        print(
            scene["id"],
            "best=",
            best["name"],
            "worst=",
            worst["name"],
            "red_active=",
            rows[0].get("red_light_constraint_active_fraction"),
            flush=True,
        )

    model_status = {"requested": bool(args_cli.run_model), "success": False}
    if args_cli.run_model:
        requested_ids = {item.strip() for item in args_cli.model_scenes.split(",") if item.strip()}
        model_scenes = [scene for scene in loaded if scene["id"] in requested_ids]
        try:
            model_samples = _actual_model_candidates(
                model_scenes,
                generations=int(args_cli.model_generations),
                steps=int(args_cli.model_steps),
            )
            for scene in model_scenes:
                scene["candidates"].extend(model_samples[scene["id"]])
                scene["rows"] = _score_candidates(
                    scene,
                    scene["candidates"],
                    reward_args,
                )
                dashboard_path = ASSET_DIR / f"t4_scene_{scene['id']}_dashboard.png"
                bev_path = ASSET_DIR / f"t4_scene_{scene['id']}_bev.png"
                _plot_dashboard(scene, scene["candidates"], scene["rows"], dashboard_path)
                _plot_bev(scene, scene["candidates"], scene["rows"], bev_path)
                gif_path = ASSET_DIR / f"t4_scene_{scene['id']}.gif"
                if args_cli.gif or scene["kind"] == "dynamic" or gif_path.exists():
                    _render_gif(scene, scene["candidates"], scene["rows"], gif_path)
                records[scene["id"]]["candidate_source"] = (
                    "structured_counterfactual_reward_probe_plus_actual_rl_checkpoint_samples"
                )
                records[scene["id"]]["model_checkpoint"] = str(RL_ARGS.parent / "best_model.pth")
                records[scene["id"]]["model_sample_count"] = int(args_cli.model_generations)
                records[scene["id"]]["candidates"] = scene["rows"]
            model_status.update(
                {
                    "success": True,
                    "checkpoint": str(RL_ARGS.parent / "best_model.pth"),
                    "scene_ids": [scene["id"] for scene in model_scenes],
                    "generations": int(args_cli.model_generations),
                    "steps": int(args_cli.model_steps),
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                }
            )
            print("actual RL checkpoint samples appended", model_status, flush=True)
        except Exception as exc:
            model_status.update(
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "checkpoint": str(RL_ARGS.parent / "best_model.pth"),
                }
            )
            print("actual RL checkpoint sampling failed:", repr(exc), flush=True)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14), facecolor="#f8fafc")
    for ax, scene in zip(axes.flat, loaded, strict=False):
        _draw_trajectory_bev(
            ax,
            scene,
            scene["candidates"],
            scene["rows"],
            title=scene["short"],
            time_index=40,
        )
    fig.suptitle(
        "COUNTERFACTUAL PROBES / NOT FORMAL AWR: reward semantics in real T4 geometry",
        x=0.04,
        y=0.995,
        ha="left",
        fontsize=19,
        fontweight="bold",
        color="#0f172a",
    )
    fig.savefig(
        ASSET_DIR / "t4_scene_reward_gallery.png",
        dpi=185,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    records["_model_sampling"] = model_status
    _update_manifest(records)
    print("wrote scene assets to", ASSET_DIR)


if __name__ == "__main__":
    main()
