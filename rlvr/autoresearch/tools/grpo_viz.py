#!/usr/bin/env python3
"""Visualize all K trajectories per scene with reward breakdown.

One figure per scene showing:
- Left: bird's eye view with all K trajectories colored by rank (green=best, red=worst)
  with route/road borders, logged-neighbor futures, true oriented boxes, GT, and ego footprints
- Right: reward breakdown table for each trajectory
- Bottom-right inset: endpoint spread and fixed-radius endpoint clusters
- Optional GIF: the same geometry primitives over the 10 Hz replay timeline

Usage:
    python -m rlvr.autoresearch.tools.grpo_viz \
        --model_path /path/to/best_model.pth \
        --scenes /path/to/scenes.json \
        --output_dir /path/to/output \
        --indices 0 1 2 3 4 \
        --K 16 --enable_lane --survival \
        --w_progress 3.0 --lane_near_scale 30.0 --lane_wide_scale 10.0

For the original-DP AWR diagnostic used by the conference material, add
``--awr_mode --reward_config rlvr/configs/awr_original_dp_t4_hdp_faithful.json``.
That path bypasses the GRPO guidance sampler: diversity then comes from the
same x-start diffusion group sampler used by AWR (plus optional HDP-style
trajectory augmentation), while this file remains the colleague's
all-trajectories/reward-breakdown/bbox visualization.  The manifest records
the fixed seed, legacy neighbour-future +1 alignment, endpoint metrics, and
whether a cluster is only a visualization heuristic rather than a semantic
maneuver label.  The conference figures use a source original-DP checkpoint;
they are an audit of the AWR sampling/evaluation path, not an AWR-trained
checkpoint result.

For a human-readable multimodality audit, add ``--multimodality_compare``.
This produces a paired raw-vs-augmentation trajectory fan, seed-matched endpoint
displacement plot, per-timestep branch plot, and (with ``--gif``) a side-by-side
80-frame movie.  It is deliberately a second figure: the reward table and the
multimodality explanation should not compete for the same pixels.
"""

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

# The source checkout keeps the importable ``diffusion_planner`` package one
# level below the repository project directory.  Make the colleague visualizer
# runnable directly from a clean checkout instead of relying on a caller's
# incidental PYTHONPATH (the training launcher already supplies this path).
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "diffusion_planner"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import torch
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from diffusion_planner.utils.neighbor_future_alignment import align_neighbor_future_numpy

try:  # the colleague helper lives at the T4 repo root, not in the package dir
    from visualize_grpo_samples import _bbox_corners
except ModuleNotFoundError:  # pragma: no cover - supports package-style installs
    from diffusion_planner.visualize_grpo_samples import _bbox_corners
from matplotlib.patches import Circle, Rectangle

from preference_optimization.lora_utils import load_lora_checkpoint
from preference_optimization.utils import effective_ego_shape_numpy, load_npz_data
from rlvr.awr import (
    AWRRolloutConfig,
    load_original_dp_checkpoint,
    reward_compatible_data,
    sample_unguided_dp_group,
)
from rlvr.grpo_sampler import SamplerConfig
from rlvr.grpo_sampler_batched import generate_diverse_group_batched
from rlvr.reward import (
    RewardConfig,
    compute_lane_departure_penalty,
    compute_reward_batch,
    compute_subscores_batch,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class _CommittedPrefixReplayReader:
    """Read only JSONL-committed rows while a formal writer is still active.

    This is visualization-only.  Training remains fail-closed on rank
    manifests; allowing report generation to inspect a committed prefix must
    not weaken the replay reader used by the optimizer.
    """

    def __init__(self, root: Path, rank: int):
        self.rank = int(rank)
        self.rank_dir = Path(root) / f"rank_{self.rank:04d}"
        with (self.rank_dir / "scene_paths.jsonl").open() as handle:
            self.scene_paths = [json.loads(line) for line in handle]
        self.scene_count = len(self.scene_paths)
        self.trajectories = np.load(self.rank_dir / "trajectories.npy", mmap_mode="r")
        self.weights = np.load(self.rank_dir / "weights.npy", mmap_mode="r")
        self.rewards = np.load(self.rank_dir / "rewards.npy", mmap_mode="r")
        self.group_size = int(self.trajectories.shape[1])
        self.future_len = int(self.trajectories.shape[2])
        if (
            self.trajectories.shape[0] < self.scene_count
            or self.weights.shape[0] < self.scene_count
            or self.rewards.shape[0] < self.scene_count
        ):
            raise RuntimeError("committed replay boundary exceeds preallocated arrays")

    def batch_from_indices(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or not len(indices):
            raise ValueError("replay indices must be a non-empty 1-D array")
        if int(indices.min()) < 0 or int(indices.max()) >= self.scene_count:
            raise IndexError(
                f"rank-local replay index outside committed prefix [0,{self.scene_count})"
            )
        trajectories = torch.from_numpy(
            np.array(self.trajectories[indices], dtype=np.float32, copy=True)
        ).reshape(len(indices) * self.group_size, self.future_len, 4)
        weights = torch.from_numpy(
            np.array(self.weights[indices], dtype=np.float32, copy=True)
        ).reshape(len(indices) * self.group_size)
        return [self.scene_paths[int(index)] for index in indices], SimpleNamespace(
            trajectories=trajectories,
            weights=weights,
        )


def _aligned_neighbor_future(npz):
    """Return the recorded neighbour future in the same +1 convention as AWR.

    The T4 archives contain the present neighbour state at raw index 0.  The
    training/reward loader shifts it away before scoring, so the visualization
    must do exactly the same thing.  Keeping this in the colleague visualizer
    avoids a visually plausible but temporally wrong collision picture.
    """

    future = np.asarray(npz["neighbor_agents_future"])
    return align_neighbor_future_numpy(future)


def _last_valid_index(path):
    xy = np.asarray(path)[..., :2]
    valid = np.flatnonzero(np.linalg.norm(xy, axis=-1) > 1e-4)
    return int(valid[-1]) if valid.size else 0


def _pairwise_distances(points):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return np.zeros(0, dtype=np.float64)
    i, j = np.triu_indices(len(points), k=1)
    return np.linalg.norm(points[i] - points[j], axis=-1)


def _connected_components(points, threshold):
    """Count endpoint clusters; this is an auditable visualization heuristic."""

    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n == 0:
        return 0, []
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    dist = _pairwise_distances(points)
    pairs = list(zip(*np.triu_indices(n, k=1)))
    for (a, b), d in zip(pairs, dist):
        if d <= threshold:
            union(a, b)
    roots = [find(i) for i in range(n)]
    sizes = {}
    for root in roots:
        sizes[root] = sizes.get(root, 0) + 1
    return len(sizes), sorted(sizes.values(), reverse=True)


def diversity_metrics(results):
    """Return group diversity diagnostics used in the conference figures.

    ``endpoint_modes_*`` are deliberately called heuristic clusters rather
    than semantic maneuvers.  They show whether the candidate set contains
    separated endpoint branches; a human/closed-loop audit is still required
    before calling a cluster a lane-change or yield mode.
    """

    if not results:
        return {
            "K": 0,
            "endpoint_pairwise_mean_m": 0.0,
            "endpoint_pairwise_max_m": 0.0,
            "endpoint_std_x_m": 0.0,
            "endpoint_std_y_m": 0.0,
            "endpoint_bbox_diagonal_m": 0.0,
            "path_pairwise_mean_m": 0.0,
            "path_pairwise_max_m": 0.0,
            "endpoint_modes_1m": 0,
            "endpoint_modes_2m": 0,
            "endpoint_modes_4m": 0,
            "endpoint_mode_sizes_2m": [],
            "nontrivial_endpoint_fraction": 0.0,
        }
    paths = np.stack([np.asarray(r["traj"]) for r in results], axis=0)
    endpoints = np.stack([p[_last_valid_index(p), :2] for p in paths], axis=0)
    ep_dist = _pairwise_distances(endpoints)
    # A fixed common reference point makes this comparable across scenes.
    medoid = endpoints[np.argmin(np.linalg.norm(endpoints[:, None] - endpoints[None, :], axis=-1).sum(axis=1))]
    path_dist = _pairwise_distances(paths[:, :, :2].reshape(len(paths), -1))
    # The flattened distance above is useful as a compact shape statistic but
    # has sqrt(T) scaling. Report the per-timestep mean pairwise distance too.
    if len(paths) >= 2:
        i, j = np.triu_indices(len(paths), k=1)
        path_pair = np.linalg.norm(paths[i, :, :2] - paths[j, :, :2], axis=-1).mean(axis=-1)
    else:
        path_pair = np.zeros(0, dtype=np.float64)
    modes = {}
    sizes = {}
    for threshold in (1.0, 2.0, 4.0):
        modes[threshold], sizes[threshold] = _connected_components(endpoints, threshold)
    return {
        "K": int(len(paths)),
        "endpoint_pairwise_mean_m": float(ep_dist.mean()) if ep_dist.size else 0.0,
        "endpoint_pairwise_max_m": float(ep_dist.max()) if ep_dist.size else 0.0,
        "endpoint_std_x_m": float(endpoints[:, 0].std()),
        "endpoint_std_y_m": float(endpoints[:, 1].std()),
        "endpoint_bbox_diagonal_m": float(np.linalg.norm(endpoints.max(axis=0) - endpoints.min(axis=0))),
        "path_pairwise_mean_m": float(path_pair.mean()) if path_pair.size else 0.0,
        "path_pairwise_max_m": float(path_pair.max()) if path_pair.size else 0.0,
        "endpoint_modes_1m": int(modes[1.0]),
        "endpoint_modes_2m": int(modes[2.0]),
        "endpoint_modes_4m": int(modes[4.0]),
        "endpoint_mode_sizes_2m": [int(v) for v in sizes[2.0]],
        "nontrivial_endpoint_fraction": float(np.mean(np.linalg.norm(endpoints - medoid, axis=-1) > 0.5)),
        "flattened_path_pairwise_mean_m": float(path_dist.mean()) if path_dist.size else 0.0,
    }


def _endpoint_component_labels(results, threshold=1.0):
    """Return endpoint labels for a visual cluster, not a maneuver label.

    The connected components are intentionally simple and deterministic.  They
    answer only: "which sampled endpoints are within *threshold* metres of one
    another?"  The web page must not rename these components as yield, go, or
    lane-change without a separate semantic/closed-loop audit.
    """

    if not results:
        return np.zeros(0, dtype=int), np.zeros((0, 2), dtype=float), []
    endpoints = np.asarray(
        [np.asarray(r["traj"])[_last_valid_index(r["traj"]), :2] for r in results],
        dtype=float,
    )
    n = len(endpoints)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(endpoints[i] - endpoints[j]) <= float(threshold):
                union(i, j)

    roots = [find(i) for i in range(n)]
    groups = {}
    for i, root in enumerate(roots):
        groups.setdefault(root, []).append(i)
    ordered = sorted(groups.values(), key=lambda g: (-len(g), min(g)))
    labels = np.zeros(n, dtype=int)
    for label, group in enumerate(ordered):
        labels[group] = label
    centroids = np.asarray([endpoints[group].mean(axis=0) for group in ordered], dtype=float)
    return labels, endpoints, ordered


def _multimodality_palette(n):
    """High-contrast colors for visible endpoint components."""

    base = [
        "#2563eb",  # blue
        "#e11d48",  # red
        "#f59e0b",  # amber
        "#7c3aed",  # violet
        "#059669",  # green
        "#0891b2",  # cyan
        "#c2410c",  # orange
        "#4f46e5",  # indigo
    ]
    return [base[i % len(base)] for i in range(max(int(n), 1))]


def _focus_limits(npz, result_sets):
    """Camera bounds that keep the actual trajectory fan readable."""

    points = []
    for results in result_sets:
        points.extend(np.asarray(r["traj"])[..., :2] for r in results)
    points.append(np.asarray(npz["ego_agent_future"])[..., :2])
    future = _aligned_neighbor_future(npz)
    tracks = _neighbor_tracks(npz, future)
    for _, p, _, valid in tracks[:6]:
        points.append(future[p, valid, :2])
    points = [p for p in points if len(p)]
    if not points:
        return (-10, 10), (-10, 10)
    points = np.concatenate(points, axis=0)
    lo, hi = points.min(axis=0), points.max(axis=0)
    center = (lo + hi) / 2.0
    span = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 8.0)
    margin = max(5.0, span * 0.10)
    half = span / 2.0 + margin
    return (center[0] - half, center[0] + half), (center[1] - half, center[1] + half)


def _draw_multimodality_panel(
    ax, npz, npz_path, results, title, *, endpoint_threshold=1.0
):
    """Draw a clean trajectory fan without the reward table.

    This is deliberately separate from ``draw_grpo_scene``.  The latter is a
    reward audit; this panel is an explanation of multimodality.  Every colored
    line is one sampled future, every endpoint marker is one sample, and the
    boxes at t=0/half/end expose the physical footprint of representative
    branches.
    """

    labels, endpoints, groups = _endpoint_component_labels(results, endpoint_threshold)
    colors = _multimodality_palette(len(groups))
    _draw_map_context(ax, npz)
    gt = np.asarray(npz["ego_agent_future"])
    ax.plot(gt[:, 0], gt[:, 1], "--", color="#15803d", lw=2.2, alpha=0.85, label="logged expert")

    future = _aligned_neighbor_future(npz)
    tracks = _neighbor_tracks(npz, future)
    neighbor_colors = ["#64748b", "#0f766e", "#b45309", "#7c3aed", "#be123c", "#0369a1"]
    for track_rank, (_, p, cur, valid) in enumerate(tracks[:5]):
        color = neighbor_colors[track_rank % len(neighbor_colors)]
        fut = future[p]
        valid = valid & np.isfinite(fut[:, :2]).all(axis=-1)
        if valid.sum() < 2:
            continue
        ax.plot(
            fut[valid, 0], fut[valid, 1], "--", color=color, lw=1.0, alpha=0.42,
            label="recorded neighbor future" if track_rank == 0 else None,
        )
        n_length = float(cur[7]) if cur.shape[0] > 7 and cur[7] > 0.1 else 4.5
        n_width = float(cur[6]) if cur.shape[0] > 6 and cur[6] > 0.1 else 1.8
        if np.linalg.norm(cur[:2]) > 1e-4:
            _add_obb(ax, cur[:4], n_length, n_width, color=color, alpha=0.82, lw=1.0, zorder=12)
        nt = int(np.flatnonzero(valid)[-1])
        pose = np.array([fut[nt, 0], fut[nt, 1], np.cos(fut[nt, 2]), np.sin(fut[nt, 2])])
        _add_obb(ax, pose, n_length, n_width, color=color, alpha=0.24, lw=0.8, zorder=8, ls=":")

    wb, length, width = map(
        float, effective_ego_shape_numpy(npz_path, npz.get("ego_shape"))[:3]
    )
    ego_current = np.asarray(npz.get("ego_current_state", np.zeros(10, dtype=np.float32)))
    if ego_current.ndim > 1:
        ego_current = ego_current[0]
    ego_pose = np.asarray(ego_current[:4], dtype=float)
    if not np.isfinite(ego_pose).all() or np.linalg.norm(ego_pose[2:4]) < 1e-4:
        ego_pose = np.array([0.0, 0.0, 1.0, 0.0])
    _add_obb(ax, ego_pose, length, width, color="#111827", alpha=0.95, lw=1.8, zorder=22, wheelbase=wb)
    ax.plot(ego_pose[0], ego_pose[1], "k*", ms=9, zorder=23, label="ego now")

    # Draw all sample paths with enough opacity that the reader can count them.
    # Representative paths receive physical ego boxes at three time points.
    representative = set()
    for group in groups:
        representative.add(group[0])
        if len(group) > 1:
            representative.add(group[-1])
    for i, result in enumerate(results):
        traj = np.asarray(result["traj"])
        color = colors[labels[i]]
        line_alpha = 0.78 if i in representative else 0.38
        line_width = 2.0 if i in representative else 1.15
        ax.plot(traj[:, 0], traj[:, 1], color=color, lw=line_width, alpha=line_alpha, zorder=14)
        ax.scatter(
            endpoints[i, 0], endpoints[i, 1], s=33 if i in representative else 21,
            color=color, edgecolor="white", linewidth=0.65, zorder=18,
        )
        if i in representative:
            for t in sorted(set([0, max(len(traj) // 2, 0), max(len(traj) - 1, 0)])):
                _add_obb(ax, traj[t], length, width, color=color, alpha=0.42, lw=0.9, zorder=16, wheelbase=wb)

    for label, group in enumerate(groups):
        centroid = endpoints[group].mean(axis=0)
        ax.scatter(centroid[0], centroid[1], marker="X", s=105, color=colors[label], edgecolor="#111827", linewidth=0.8, zorder=21)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.16)
    ax.set_xlabel("ego-frame x (m) · forward")
    ax.set_ylabel("ego-frame y (m) · lateral")
    ax.set_title(title, fontsize=13, weight="bold", pad=10)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.94)
    ax.text(
        0.02, 0.02,
        f"K={len(results)} · {len(groups)} endpoint components @ {endpoint_threshold:.0f}m\n"
        "each line = one sampled future · circle = endpoint · X = component centroid",
        transform=ax.transAxes, fontsize=8, color="#334155",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.88, "edgecolor": "#cbd5e1"},
    )
    # The full BEV keeps road context, but a long straight corridor can make
    # a 0.5--2 m lateral fan look visually flat.  Add a dedicated zoom inset
    # so the reader can count the actual sample branches without sacrificing
    # the bbox/map view above.
    zoom = ax.inset_axes([0.10, 0.56, 0.82, 0.30], facecolor="white")
    zoom.plot(gt[:, 0], gt[:, 1], "--", color="#15803d", lw=1.3, alpha=0.7)
    all_traj_xy = [gt[:, :2]]
    for i, result in enumerate(results):
        traj = np.asarray(result["traj"])
        color = colors[labels[i]]
        zoom.plot(traj[:, 0], traj[:, 1], color=color, lw=1.5 if i in representative else 0.8, alpha=0.74 if i in representative else 0.40)
        all_traj_xy.append(traj[:, :2])
    all_traj_xy = np.concatenate(all_traj_xy, axis=0)
    zlo, zhi = all_traj_xy.min(axis=0), all_traj_xy.max(axis=0)
    zmargin = np.maximum((zhi - zlo) * 0.16, np.array([1.0, 0.7]))
    zoom.set_xlim(zlo[0] - zmargin[0], zhi[0] + zmargin[0])
    zoom.set_ylim(zlo[1] - zmargin[1], zhi[1] + zmargin[1])
    zoom.grid(True, alpha=0.18)
    zoom.set_title("ZOOM · every colored line is one future", fontsize=8, pad=2, weight="bold")
    zoom.tick_params(labelsize=6, length=2)
    for spine in zoom.spines.values():
        spine.set_edgecolor("#334155")
        spine.set_linewidth(1.2)
    return {"labels": labels, "endpoints": endpoints, "groups": groups, "colors": colors}


def draw_multimodality_comparison(
    fig,
    npz,
    npz_path,
    raw_results,
    aug_results,
    *,
    scene_title,
    endpoint_threshold=1.0,
):
    """Create the primary human-readable raw-vs-augmentation evidence figure."""

    fig.patch.set_facecolor("#f8fafc")
    axes = fig.subplots(2, 2, gridspec_kw={"height_ratios": [1.36, 0.84]})
    raw_info = _draw_multimodality_panel(
        axes[0, 0], npz, npz_path, raw_results,
        f"RAW DP · same scene, K={len(raw_results)}\nno trajectory augmentation",
        endpoint_threshold=endpoint_threshold,
    )
    aug_info = _draw_multimodality_panel(
        axes[0, 1], npz, npz_path, aug_results,
        f"HDP-STYLE AUGMENTATION · same scene, K={len(aug_results)}\npaired seed · std=0.5m",
        endpoint_threshold=endpoint_threshold,
    )
    xlim, ylim = _focus_limits(npz, [raw_results, aug_results])
    for ax in axes[0]:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    ax_ep = axes[1, 0]
    raw_ep, aug_ep = raw_info["endpoints"], aug_info["endpoints"]
    raw_by_k = {int(r["k"]): raw_ep[i] for i, r in enumerate(raw_results)}
    aug_by_k = {int(r["k"]): aug_ep[i] for i, r in enumerate(aug_results)}
    for k in sorted(set(raw_by_k) & set(aug_by_k)):
        ax_ep.plot(
            [raw_by_k[k][0], aug_by_k[k][0]], [raw_by_k[k][1], aug_by_k[k][1]],
            color="#94a3b8", lw=0.8, alpha=0.5, zorder=1,
        )
    ax_ep.scatter(raw_ep[:, 0], raw_ep[:, 1], s=48, facecolors="none", edgecolors="#2563eb", linewidth=1.5, label="raw endpoint", zorder=3)
    ax_ep.scatter(aug_ep[:, 0], aug_ep[:, 1], s=60, marker="x", color="#e11d48", linewidth=1.6, label="augmented endpoint", zorder=4)
    for label, group in enumerate(raw_info["groups"]):
        c = raw_ep[group].mean(axis=0)
        ax_ep.add_patch(Circle(c, endpoint_threshold, fill=False, ls="--", lw=0.9, ec="#2563eb", alpha=0.5))
    for label, group in enumerate(aug_info["groups"]):
        c = aug_ep[group].mean(axis=0)
        ax_ep.add_patch(Circle(c, endpoint_threshold, fill=False, ls=":", lw=1.2, ec="#e11d48", alpha=0.65))
    ax_ep.set_aspect("equal", adjustable="box")
    ep_all = np.concatenate([raw_ep, aug_ep], axis=0)
    center = ep_all.mean(axis=0)
    span = max(float(np.ptp(ep_all[:, 0])), float(np.ptp(ep_all[:, 1])), 2.0)
    half = span / 2 + 1.4
    ax_ep.set_xlim(center[0] - half, center[0] + half)
    ax_ep.set_ylim(center[1] - half, center[1] + half)
    ax_ep.grid(True, alpha=0.18)
    ax_ep.set_title("Endpoint pairing: where does augmentation move the same samples?", fontsize=12, weight="bold")
    ax_ep.set_xlabel("endpoint x (m)")
    ax_ep.set_ylabel("endpoint y (m)")
    ax_ep.legend(loc="best", fontsize=8, framealpha=0.94)
    ax_ep.text(0.02, 0.03, "gray segments connect same-seed raw → augmented endpoints\nrings are geometric components, not maneuver labels", transform=ax_ep.transAxes, fontsize=8, color="#475569", bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cbd5e1"})

    # Pick the axis with the strongest visible spread for the time panel.  In
    # a crossing scene that is normally longitudinal progress; in a turn it
    # is usually lateral offset.
    raw_std = np.std(raw_ep, axis=0)
    aug_std = np.std(aug_ep, axis=0)
    dim = int(np.argmax(raw_std + aug_std))
    dim_name = "longitudinal x / forward progress" if dim == 0 else "lateral y / turn offset"
    ax_t = axes[1, 1]
    max_len = min([len(r["traj"]) for r in raw_results + aug_results] or [1])
    tt = np.arange(max_len) / 10.0
    for i, r in enumerate(raw_results):
        tr = np.asarray(r["traj"])[:max_len, dim]
        ax_t.plot(tt[:len(tr)], tr, color="#2563eb", lw=1.0, alpha=0.26, ls="--")
    for i, r in enumerate(aug_results):
        tr = np.asarray(r["traj"])[:max_len, dim]
        ax_t.plot(tt[:len(tr)], tr, color="#e11d48", lw=1.25, alpha=0.40)
    raw_stack = np.stack([np.asarray(r["traj"])[:max_len, dim] for r in raw_results])
    aug_stack = np.stack([np.asarray(r["traj"])[:max_len, dim] for r in aug_results])
    ax_t.plot(tt, np.median(raw_stack, axis=0), color="#1d4ed8", lw=2.5, ls="--", label="raw median")
    ax_t.plot(tt, np.median(aug_stack, axis=0), color="#be123c", lw=2.5, label="aug median")
    ax_t.fill_between(tt, np.percentile(raw_stack, 10, axis=0), np.percentile(raw_stack, 90, axis=0), color="#60a5fa", alpha=0.12)
    ax_t.fill_between(tt, np.percentile(aug_stack, 10, axis=0), np.percentile(aug_stack, 90, axis=0), color="#fb7185", alpha=0.12)
    for sec in (2.0, 4.0, 6.0, 8.0):
        if sec <= tt[-1]:
            ax_t.axvline(sec, color="#94a3b8", lw=0.7, alpha=0.35)
    ax_t.grid(True, alpha=0.18)
    ax_t.set_title(f"Time expansion: how do 16 futures separate over 8 seconds?\n{dim_name}", fontsize=12, weight="bold")
    ax_t.set_xlabel("time (s) · 10 Hz")
    ax_t.set_ylabel("ego-frame metres")
    ax_t.legend(loc="best", fontsize=8, framealpha=0.94)
    ax_t.text(0.02, 0.03, "thin lines=individual samples · thick=median · bands=10–90%", transform=ax_t.transAxes, fontsize=8, color="#475569", bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cbd5e1"})

    fig.suptitle(
        f"MULTIMODALITY AUDIT · {scene_title}\n"
        "same real scene · K complete futures · endpoint spread is not a semantic maneuver",
        fontsize=16, weight="bold", color="#102a43", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.955], pad=1.2)
    return {"raw": raw_info, "aug": aug_info}


def render_multimodality_gif(raw_results, aug_results, npz_path, output_path, *, scene_title, endpoint_threshold=1.0, frame_duration_ms=100):
    """Animate the same raw-vs-augmented candidate fan side by side."""

    from PIL import Image

    npz = np.load(npz_path)
    gt = np.asarray(npz["ego_agent_future"])
    future = _aligned_neighbor_future(npz)
    tracks = _neighbor_tracks(npz, future)
    all_results = list(raw_results) + list(aug_results)
    horizon = min([len(r["traj"]) for r in all_results] + [len(gt), future.shape[1]])
    xlim, ylim = _focus_limits(npz, [raw_results, aug_results])
    raw_info = _endpoint_component_labels(raw_results, endpoint_threshold)
    aug_info = _endpoint_component_labels(aug_results, endpoint_threshold)
    frame_images = []
    cmap_raw = _multimodality_palette(max(len(raw_info[2]), 1))
    cmap_aug = _multimodality_palette(max(len(aug_info[2]), 1))
    wb, length, width = map(
        float, effective_ego_shape_numpy(npz_path, npz.get("ego_shape"))[:3]
    )

    for t in range(horizon):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="#f8fafc", sharex=True, sharey=True)
        for ax, results, info, colors, label in [
            (axes[0], raw_results, raw_info, cmap_raw, "RAW DP"),
            (axes[1], aug_results, aug_info, cmap_aug, "HDP-STYLE AUGMENTATION"),
        ]:
            _draw_map_context(ax, npz)
            ax.plot(gt[:, 0], gt[:, 1], "--", color="#15803d", lw=1.7, alpha=0.7, label="logged expert")
            for i, result in enumerate(results):
                traj = np.asarray(result["traj"])
                upto = min(t + 1, len(traj))
                color = colors[info[0][i]]
                ax.plot(traj[:upto, 0], traj[:upto, 1], color=color, lw=1.5 if i in info[2][0] else 0.85, alpha=0.62 if i in info[2][0] else 0.28, zorder=14)
                if upto:
                    _add_obb(ax, traj[upto - 1], length, width, color=color, alpha=0.34 if i in info[2][0] else 0.10, lw=0.8, zorder=16, wheelbase=wb)
            for track_rank, (_, p, cur, valid) in enumerate(tracks[:5]):
                fut = future[p]
                seen = min(t + 1, len(fut))
                observed = valid[:seen]
                if not observed.any():
                    continue
                nt = int(np.flatnonzero(observed)[-1])
                color = ["#64748b", "#0f766e", "#b45309", "#7c3aed", "#be123c"][track_rank]
                n_length = float(cur[7]) if cur.shape[0] > 7 and cur[7] > 0.1 else 4.5
                n_width = float(cur[6]) if cur.shape[0] > 6 and cur[6] > 0.1 else 1.8
                pose = np.array([fut[nt, 0], fut[nt, 1], np.cos(fut[nt, 2]), np.sin(fut[nt, 2])])
                _add_obb(ax, pose, n_length, n_width, color=color, alpha=0.86, lw=1.0, zorder=18, label="recorded neighbor" if track_rank == 0 else None)
                ax.plot(fut[:seen, 0], fut[:seen, 1], "--", color=color, alpha=0.30, lw=0.7, zorder=5)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.16)
            ax.set_xlabel("ego-frame x (m) · forward")
            ax.set_ylabel("ego-frame y (m) · lateral")
            ax.set_title(f"{label} · t={t / 10.0:.1f}s\nall K={len(results)} sample footprints", fontsize=11, weight="bold")
            ax.legend(loc="upper left", fontsize=7, framealpha=0.92)
        fig.suptitle(f"MULTIMODALITY DYNAMICS · {scene_title}\nraw vs augmented · fixed camera · +1 aligned replay neighbors", fontsize=14, weight="bold", color="#102a43")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frame_images.append(Image.fromarray(rgba[..., :3].copy()))
        plt.close(fig)

    if frame_images:
        frame_images[0].save(output_path, save_all=True, append_images=frame_images[1:], duration=int(frame_duration_ms), loop=0, optimize=False)


def _scene_entry(entry):
    """Normalize a path-only or metadata-rich scene list entry."""

    if isinstance(entry, dict):
        path = entry.get("path") or entry.get("npz_path")
        if not path:
            raise ValueError(f"scene entry has no path/npz_path: {entry}")
        return str(path), str(entry.get("title") or entry.get("id") or Path(path).stem), entry
    path = str(entry)
    return path, Path(path).stem, {}


def _add_obb(
    ax,
    pose,
    length,
    width,
    *,
    color,
    alpha=0.7,
    lw=1.0,
    zorder=10,
    label=None,
    ls="-",
    fill_alpha=0.0,
    wheelbase=None,
):
    pose = np.asarray(pose, dtype=float)
    cx, cy = float(pose[0]), float(pose[1])
    if wheelbase is not None:
        heading = np.asarray(pose[2:4], dtype=float)
        norm = float(np.linalg.norm(heading))
        if norm > 1e-6:
            heading = heading / norm
            # DP trajectories are rear-axle referenced.  This is the same
            # centre shift as planner_metrics.geometry._build_ego_bbox_corners.
            cx += float(heading[0]) * 0.5 * float(wheelbase)
            cy += float(heading[1]) * 0.5 * float(wheelbase)
    corners = _bbox_corners(cx, cy, pose[2], pose[3], length, width)
    if fill_alpha > 0.0:
        ax.fill(
            corners[:, 0],
            corners[:, 1],
            color=color,
            alpha=fill_alpha,
            linewidth=0.0,
            zorder=zorder - 0.1,
        )
    ax.plot(corners[:, 0], corners[:, 1], ls, color=color, alpha=alpha, lw=lw, zorder=zorder, label=label)
    return corners


def _neighbor_tracks(npz, future, *, limit=12):
    """Select the visible recorded traffic tracks around the ego corridor."""

    past = np.asarray(npz.get("neighbor_agents_past", np.zeros((0, 1, 11))))
    if past.ndim != 3:
        return []
    records = []
    for p in range(min(past.shape[0], future.shape[0])):
        cur = past[p, -1]
        cur_valid = np.linalg.norm(cur[:2]) > 1e-4
        fut_xy = future[p, :, :2]
        fut_valid = np.linalg.norm(fut_xy, axis=-1) > 1e-4
        if not cur_valid and not fut_valid.any():
            continue
        center = cur[:2] if cur_valid else fut_xy[np.flatnonzero(fut_valid)[0]]
        distance = float(np.linalg.norm(center))
        records.append((distance, p, cur, fut_valid))
    # A full archive can contain 320 slots. Keep the nearest traffic in the
    # hero plot, while still making all candidate ego paths visible.
    records = sorted(records, key=lambda x: x[0])
    return records if limit is None else records[: int(limit)]


def _hero_hard_event(results):
    """Pick one event whose geometry should receive the explanatory zoom.

    Collision takes precedence over road-border crossing because it needs two
    moving OBBs to be understood.  Within a type, the earliest event is the
    clearest causal snapshot.  The returned rank is the reward-table rank,
    not the original sampler index.
    """

    choices = []
    for rank, result in enumerate(results):
        collision_step = result.get("collision_step")
        rb_step = result.get("rb_crossing_step")
        if collision_step is not None:
            choices.append((0, int(collision_step), rank, result, "COL"))
        elif rb_step is not None:
            choices.append((1, int(rb_step), rank, result, "RB"))
    if not choices:
        return None
    _, step, rank, result, label = min(choices, key=lambda item: (item[0], item[1], item[2]))
    return {"rank": rank, "step": step, "result": result, "label": label}


def _nearest_neighbor_footprint(traj_pose, step, neighbor_future, neighbor_tracks):
    """Return the recorded OBB nearest an ego candidate at one aligned step."""

    best = None
    ego_xy = np.asarray(traj_pose[:2], dtype=float)
    for track_rank, (_, p, cur, valid) in enumerate(neighbor_tracks):
        if step < 0 or step >= len(valid) or not bool(valid[step]):
            continue
        fut = neighbor_future[p]
        xy = np.asarray(fut[step, :2], dtype=float)
        if not np.isfinite(xy).all():
            continue
        distance = float(np.linalg.norm(xy - ego_xy))
        yaw = float(fut[step, 2])
        pose = np.array([xy[0], xy[1], np.cos(yaw), np.sin(yaw)], dtype=float)
        length = float(cur[7]) if cur.shape[0] > 7 and cur[7] > 0.1 else 4.5
        width = float(cur[6]) if cur.shape[0] > 6 and cur[6] > 0.1 else 1.8
        item = {
            "track_rank": track_rank,
            "slot": int(p),
            "pose": pose,
            "length": length,
            "width": width,
            "center_distance_m": distance,
        }
        if best is None or distance < best[0]:
            best = (distance, item)
    return None if best is None else best[1]


def _camera_points(gt, results, neighbor_future, neighbor_tracks, *, neighbor_radius_m=25.0):
    """Keep the camera on ego plans and traffic that can affect those plans."""

    paths = [np.asarray(result["traj"])[..., :2] for result in results]
    points = [np.asarray(gt)[..., :2], *paths]
    if not paths:
        return np.concatenate(points, axis=0)
    path_stack = np.stack(paths, axis=0)
    horizon = path_stack.shape[1]
    for _, p, _, valid in neighbor_tracks:
        fut = neighbor_future[p]
        n = min(horizon, len(fut), len(valid))
        if n <= 0:
            continue
        valid_n = np.asarray(valid[:n], dtype=bool) & np.isfinite(fut[:n, :2]).all(axis=-1)
        if not valid_n.any():
            continue
        times = np.flatnonzero(valid_n)
        dist = np.linalg.norm(
            path_stack[:, times, :2] - fut[times, :2][None, ...],
            axis=-1,
        )
        close = times[np.min(dist, axis=0) <= float(neighbor_radius_m)]
        if close.size:
            points.append(fut[close, :2])
    finite = [p[np.isfinite(p).all(axis=-1)] for p in points if len(p)]
    return np.concatenate(finite, axis=0)


def generate_and_score(
    model,
    model_args,
    npz_path,
    K,
    reward_config,
    sampler_config,
    *,
    awr_mode=False,
    sample_steps=5,
    noise_scale=0.5,
    deterministic_first=False,
    hdp_trajectory_augmentation=False,
    hdp_trajectory_augmentation_std=0.5,
):
    """Generate K trajectories and score each one with full breakdown."""
    device = torch.device(DEVICE)
    data = load_npz_data(npz_path, device)
    es = data.get("ego_shape")
    ego_shape = es[0] if es is not None and es.dim() > 1 else es

    if awr_mode:
        awr_config = AWRRolloutConfig(
            n_trajectories=int(K),
            sample_steps=int(sample_steps),
            noise_scale_range=(float(noise_scale), float(noise_scale)),
            deterministic_first=bool(deterministic_first),
            hdp_trajectory_augmentation=bool(hdp_trajectory_augmentation),
            hdp_trajectory_augmentation_std=float(hdp_trajectory_augmentation_std),
            inference_amp_dtype="off",
        )
        trajs, _, _ = sample_unguided_dp_group(model, model_args, data, awr_config, device)
    else:
        trajs = generate_diverse_group_batched(model, model_args, data, sampler_config, device)

    # Compute GT path length for underprogress context
    gt_path_len = 0.0
    if "ego_agent_future" in data:
        gt = data["ego_agent_future"]
        if gt.dim() == 3:
            gt = gt[0]
        gt_xy = gt[:, :2]
        gt_valid = gt_xy.abs().sum(dim=-1) > 0.1
        if gt_valid.sum() >= 10:
            gt_path_len = float(torch.diff(gt_xy[gt_valid], dim=0).norm(dim=-1).sum().item())

    results = score_trajectory_group(
        trajs,
        data,
        reward_config,
        ego_shape=ego_shape,
    )
    return results, data, gt_path_len


def score_trajectory_group(
    trajs,
    data,
    reward_config,
    *,
    ego_shape=None,
    replay_weights=None,
):
    """Score an existing K-group and retain the geometry behind each number.

    This is shared by live model sampling and frozen formal replay groups.  In
    addition to the public RewardBreakdown, it records the TTC score, minimum
    signed OBB clearance, first unsafe/collision time, and road-border margin.
    Those diagnostics make a scene figure explain *why* a candidate lost
    instead of presenting an opaque scalar reward.
    """

    if not isinstance(trajs, torch.Tensor):
        trajs = torch.as_tensor(trajs, dtype=torch.float32)
    if ego_shape is None:
        es = data.get("ego_shape")
        ego_shape = es[0] if es is not None and es.dim() > 1 else es

    # The original DP model consumes the historical 3-channel neighbour
    # future, while planner-metrics consumes x/y/cos/sin.
    reward_data = reward_compatible_data(data)
    horizon = int(getattr(reward_config, "reward_horizon_steps", 0))
    scored_trajs = trajs[:, :horizon] if 0 < horizon < trajs.shape[1] else trajs
    scored_data = dict(reward_data)
    if scored_trajs.shape[1] != trajs.shape[1]:
        for key in ("ego_agent_future", "neighbor_agents_future"):
            value = scored_data.get(key)
            if isinstance(value, torch.Tensor) and value.dim() >= 3:
                scored_data[key] = value[..., : scored_trajs.shape[1], :]

    rewards = compute_reward_batch(trajs, reward_data, reward_config)
    subs = compute_subscores_batch(scored_trajs, scored_data, reward_config)
    if replay_weights is not None:
        replay_weights = np.asarray(replay_weights, dtype=np.float64).reshape(-1)

    results = []
    for k_i, r in enumerate(rewards):
        traj_t = trajs[k_i : k_i + 1]
        compute_lane_departure_penalty(traj_t, ego_shape, reward_data)

        # Compute path length
        traj_np = trajs[k_i].detach().cpu().numpy()
        path_len = float(np.linalg.norm(np.diff(traj_np[:, :2], axis=0), axis=1).sum())

        clearance_series = subs["ttc_min_clearance"][k_i].detach().cpu().numpy()
        scored_clearance = clearance_series[1:] if len(clearance_series) > 1 else clearance_series
        min_clearance = float(np.min(scored_clearance)) if len(scored_clearance) else 99.0
        first_unsafe = subs["ttc_first_unsafe_steps"][k_i]
        collision_step = subs["collision_step"][k_i]
        rb_crossing_step = subs["rb_crossing_steps"][k_i]
        hard_reasons = []
        if collision_step is not None:
            hard_reasons.append("collision")
        if r.red_light < -0.5:
            hard_reasons.append("red-light")
        if r.rb_crossing:
            hard_reasons.append("road-border")
        if r.kinematic_violated:
            hard_reasons.append("kinematic")

        results.append(
            {
                "k": k_i,
                "traj": traj_np,
                "total": r.total,
                "progress": r.progress,
                "smoothness": r.smoothness,
                "centerline": r.centerline,
                "safety": r.safety,
                "lane_crossing": r.lane_crossing,
                "lane_near": r.lane_near_frac,
                "rb_crossing": r.rb_crossing,
                "rb_near": r.rb_near_penalty,
                "red_light": r.red_light,
                "collision": r.collision_step is not None,
                "collision_step": collision_step,
                "collision_time_s": None
                if collision_step is None
                else float(collision_step) / 10.0,
                "ttc_score": float(subs["ttc"][k_i].item()),
                "ttc_first_unsafe_step": first_unsafe,
                "ttc_first_unsafe_time_s": None
                if first_unsafe is None
                else float(first_unsafe) / 10.0,
                "min_obb_clearance_m": min_clearance,
                "_clearance_series_m": clearance_series,
                "kinematic": r.kinematic_violated,
                "hard_reasons": hard_reasons,
                "rb_min_dist_m": float(r.rb_min_dist),
                "rb_gate_threshold_m": float(reward_config.rb_cross_thresh),
                "rb_crossing_step": rb_crossing_step,
                "rb_crossing_time_s": None
                if rb_crossing_step is None
                else float(rb_crossing_step) / 10.0,
                "off_road_fraction": r.off_road_fraction,
                "path_len": path_len,
                "awr_weight": None
                if replay_weights is None
                else float(replay_weights[k_i]),
                "reward_horizon_s": float(scored_trajs.shape[1]) / 10.0,
            }
        )

    results.sort(key=lambda x: -x["total"])
    return results


def _draw_map_context(ax_map, npz):
    """Draw the colleague visualizer's map context with route emphasis."""

    lanes = npz["lanes"]
    for i in range(lanes.shape[0]):
        lane = lanes[i]
        if np.abs(lane[:, :2]).sum() < 1e-6:
            continue
        if lane.shape[1] > 7:
            pts = lane[:, :2]
            lb, rb = lane[:, 4:6], lane[:, 6:8]
            v = np.abs(pts).sum(axis=1) > 0.1
            if v.sum() > 1:
                ax_map.plot((pts + lb)[v, 0], (pts + lb)[v, 1], "-", color="#aeb7c2", alpha=0.45, lw=0.7, zorder=1)
                ax_map.plot((pts + rb)[v, 0], (pts + rb)[v, 1], "-", color="#aeb7c2", alpha=0.45, lw=0.7, zorder=1)
                ax_map.plot(pts[v, 0], pts[v, 1], "-", color="#d9dee5", alpha=0.22, lw=0.5, zorder=1)

    # Route lanes are the geometric prior used to interpret lateral spread.
    route = npz.get("route_lanes")
    if route is not None:
        for lane in route:
            if lane.shape[1] <= 7:
                continue
            pts = lane[:, :2]
            v = np.abs(pts).sum(axis=1) > 0.1
            if v.sum() > 1:
                ax_map.plot(pts[v, 0], pts[v, 1], "-", color="#2f80ed", alpha=0.72, lw=1.6, zorder=2)

    # Road borders remain a hard visual constraint, not a generic lane line.
    line_strings = npz["line_strings"]
    for line in line_strings:
        if np.abs(line[:, :2]).sum() < 1e-6:
            continue
        if line_strings.shape[-1] >= 4 and line[:, 3].max() > 0.5:
            valid = (line[:, 3] > 0.5) & (np.abs(line[:, :2]).sum(axis=1) > 0.01)
            if valid.sum() > 1:
                ax_map.plot(line[valid, 0], line[valid, 1], color="#e5484d", lw=2.2, alpha=0.78, zorder=3)


def draw_grpo_scene(
    fig,
    results,
    npz_path,
    scene_idx,
    gt_path_len,
    tag,
    *,
    scene_title=None,
    diversity=None,
    reference_trajectory=None,
):
    """Draw one scene with all K trajectories + reward breakdown."""
    npz = np.load(npz_path)
    K = len(results)

    # Layout: trajectory plot on left (60%), table on right (40%)
    ax_map = fig.add_axes([0.02, 0.05, 0.55, 0.88])
    ax_tbl = fig.add_axes([0.60, 0.05, 0.38, 0.88])
    ax_tbl.axis("off")

    wb, length, width = map(
        float, effective_ego_shape_numpy(npz_path, npz.get("ego_shape"))[:3]
    )
    ro = (length - wb) / 2

    _draw_map_context(ax_map, npz)

    # Draw GT (green dashed)
    gt = npz["ego_agent_future"]
    ax_map.plot(
        gt[:, 0], gt[:, 1], "g--", lw=2.5, alpha=0.6, zorder=5, label=f"GT ({gt_path_len:.1f}m)"
    )
    reference = None
    if reference_trajectory is not None:
        candidate_reference = np.asarray(reference_trajectory)
        if candidate_reference.ndim == 2 and candidate_reference.shape[1] >= 2:
            reference = candidate_reference
            ax_map.plot(
                reference[:, 0],
                reference[:, 1],
                color="#111827",
                ls=(0, (1.5, 2.0)),
                lw=2.0,
                alpha=0.82,
                zorder=6,
                label="frozen IL reference",
            )

    # Ego box at the current state (the state is usually the local origin,
    # but using the archive pose keeps the visualizer correct for replayed
    # reproducer scenes as well).
    ego_current = npz.get("ego_current_state", np.zeros(10, dtype=np.float32))
    if ego_current.ndim > 1:
        ego_current = ego_current[0]
    ego_pose = np.asarray(ego_current[:4], dtype=float)
    if not np.isfinite(ego_pose).all() or np.linalg.norm(ego_pose[2:4]) < 1e-4:
        ego_pose = np.array([0.0, 0.0, 1.0, 0.0])
    _add_obb(ax_map, ego_pose, length, width, color="#111827", alpha=0.95, lw=2.0, zorder=21, label="ego now", wheelbase=wb)
    ax_map.plot(ego_pose[0], ego_pose[1], "k*", ms=10, zorder=22)

    # Color map: rank 1=dark green, rank K=dark red
    cmap = plt.cm.RdYlGn_r  # reversed: green=low index=best rank
    all_pts = [gt[:, :2], np.array([[0, 0]])]

    # Recorded traffic context.  The future is aligned with the training
    # convention (+1 frame), and every visible box is an oriented box rather
    # than an axis-aligned marker.  This is the part that lets a reader see
    # what a safety score is protecting against.
    neighbor_future = _aligned_neighbor_future(npz)
    neighbor_tracks = _neighbor_tracks(npz, neighbor_future)
    event_neighbor_tracks = _neighbor_tracks(npz, neighbor_future, limit=None)
    hero_event = _hero_hard_event(results)
    neighbor_colors = ["#64748b", "#0f766e", "#b45309", "#7c3aed", "#be123c", "#0369a1"]
    for track_rank, (_, p, cur, fut_valid) in enumerate(neighbor_tracks):
        color = neighbor_colors[track_rank % len(neighbor_colors)]
        fut = neighbor_future[p]
        valid = fut_valid & np.isfinite(fut[:, :2]).all(axis=-1)
        if valid.sum() > 1:
            ax_map.plot(
                fut[valid, 0], fut[valid, 1], "--", color=color, lw=0.9,
                alpha=0.42, zorder=4, label="recorded neighbor future" if track_rank == 0 else None,
            )
            all_pts.append(fut[valid, :2])
        n_length = float(cur[7]) if cur.shape[0] > 7 and cur[7] > 0.1 else 4.5
        n_width = float(cur[6]) if cur.shape[0] > 6 and cur[6] > 0.1 else 1.8
        if np.linalg.norm(cur[:2]) > 1e-4:
            _add_obb(
                ax_map, cur[:4], n_length, n_width, color=color, alpha=0.85,
                lw=1.1, zorder=9, label="neighbor now" if track_rank == 0 else None,
            )
        # Show the recorded endpoint footprint for the nearest few tracks.
        if track_rank < 4 and valid.any():
            nt = int(np.flatnonzero(valid)[-1])
            pose = np.array([fut[nt, 0], fut[nt, 1], np.cos(fut[nt, 2]), np.sin(fut[nt, 2])])
            _add_obb(ax_map, pose, n_length, n_width, color=color, alpha=0.22, lw=0.8, zorder=6, ls=":")

    for rank, r in enumerate(results):
        traj = r["traj"]
        color = cmap(rank / max(K - 1, 1))
        alpha = 0.8 if rank < 3 else (0.5 if rank < 8 else 0.3)
        lw = 2.5 if rank < 3 else 1.2

        lc = "OUT" if r["lane_crossing"] else "IN"
        rb = "RB!" if r["rb_crossing"] else ""
        candidate_label = str(r.get("candidate_label", "")).strip()
        command_text = f" {candidate_label}" if candidate_label else ""
        label = (
            f"#{rank + 1}{command_text} R={r['total']:+.3f} {lc}{rb}"
            if rank < 5
            else None
        )

        ax_map.plot(
            traj[:, 0], traj[:, 1], "-", color=color, lw=lw, alpha=alpha, zorder=10 - rank * 0.1
        )
        event_step = r.get("collision_step")
        event_label = "COL"
        if event_step is None:
            event_step = r.get("rb_crossing_step")
            event_label = "RB"
        if event_step is not None and 0 <= int(event_step) < len(traj):
            event_step = int(event_step)
            ax_map.scatter(
                traj[event_step, 0],
                traj[event_step, 1],
                marker="X",
                s=72 if rank < 5 else 42,
                color="#dc2626",
                edgecolor="white",
                linewidth=0.8,
                zorder=24,
            )
            if rank < 5 or (hero_event is not None and rank == hero_event["rank"]):
                ax_map.annotate(
                    f"#{rank + 1} {event_label} @ {event_step / 10.0:.1f}s",
                    xy=traj[event_step, :2],
                    xytext=(6, 7),
                    textcoords="offset points",
                    fontsize=6.5,
                    color="#991b1b",
                    weight="bold",
                    zorder=25,
                )
            if hero_event is not None and rank == hero_event["rank"]:
                _add_obb(
                    ax_map,
                    traj[event_step],
                    length,
                    width,
                    color="#dc2626",
                    alpha=1.0,
                    lw=2.2,
                    fill_alpha=0.16,
                    zorder=23,
                    wheelbase=wb,
                    label=f"hard-zero #{rank + 1} {event_label}",
                )
                if event_label == "COL":
                    nearest = _nearest_neighbor_footprint(
                        traj[event_step], event_step, neighbor_future, event_neighbor_tracks
                    )
                    if nearest is not None:
                        _add_obb(
                            ax_map,
                            nearest["pose"],
                            nearest["length"],
                            nearest["width"],
                            color="#f97316",
                            alpha=1.0,
                            lw=2.2,
                            fill_alpha=0.18,
                            zorder=23,
                        )
        if rank < 5:
            ax_map.plot(
                traj[-1, 0], traj[-1, 1], "o", color=color, ms=6, alpha=0.9, zorder=15, label=label
            )

        # Endpoint footprint for top 3
        if rank < 3:
            t_end = _last_valid_index(traj)
            _add_obb(
                ax_map, traj[t_end], length, width, color=color, alpha=0.25,
                lw=1.0, zorder=8, ls="-", wheelbase=wb,
            )

        # Footprints along the best path make turning/braking dynamics
        # legible without covering the whole group with boxes.
        if rank == 0:
            for t in np.linspace(0, _last_valid_index(traj), 5, dtype=int):
                _add_obb(ax_map, traj[t], length, width, color=color, alpha=0.16, lw=0.8, zorder=7, ls="-", wheelbase=wb)

        all_pts.append(traj[:, :2])

    # Auto-zoom uses only ego plans, expert motion, and traffic close enough to
    # affect a plan.  A distant archive track must not shrink the event to a
    # few pixels.  The event inset below provides a metric-scale OBB close-up.
    all_pts = _camera_points(gt, results, neighbor_future, neighbor_tracks)
    if reference is not None:
        valid_reference = reference[:, :2]
        valid_reference = valid_reference[np.isfinite(valid_reference).all(axis=-1)]
        if len(valid_reference):
            all_pts = np.concatenate([all_pts, valid_reference], axis=0)
    cx, cy = np.mean(all_pts[:, 0]), np.mean(all_pts[:, 1])
    half = max(np.ptp(all_pts[:, 0]), np.ptp(all_pts[:, 1])) * 0.6 + 8
    ax_map.set_xlim(cx - half, cx + half)
    ax_map.set_ylim(cy - half, cy + half)
    ax_map.set_aspect("equal")
    ax_map.grid(True, alpha=0.15)
    ax_map.legend(loc="upper left", fontsize=7, framealpha=0.8)

    # Hard-event inset: for safety examples, show both physical footprints at
    # the exact aligned frame.  If there is no hard event, fall back to the
    # endpoint-spread inset used by the multimodality audit.
    if diversity is None:
        diversity = diversity_metrics(results)
    ax_end = ax_map.inset_axes([0.60, 0.04, 0.36, 0.27])
    ax_end.set_facecolor("white")
    if hero_event is not None:
        _draw_map_context(ax_end, npz)
        event_rank = hero_event["rank"]
        event_result = hero_event["result"]
        event_step = hero_event["step"]
        event_label = hero_event["label"]
        event_traj = np.asarray(event_result["traj"])
        safe_result = next((r for r in results if not r.get("hard_reasons")), results[0])
        safe_traj = np.asarray(safe_result["traj"])
        safe_step = min(event_step, len(safe_traj) - 1)
        ax_end.plot(safe_traj[: safe_step + 1, 0], safe_traj[: safe_step + 1, 1], color="#15803d", lw=1.4, label="safe candidate")
        ax_end.plot(event_traj[: event_step + 1, 0], event_traj[: event_step + 1, 1], color="#dc2626", lw=1.5, label=f"hard-zero #{event_rank + 1}")
        _add_obb(ax_end, safe_traj[safe_step], length, width, color="#15803d", alpha=1.0, lw=1.3, fill_alpha=0.12, zorder=10, wheelbase=wb)
        _add_obb(ax_end, event_traj[event_step], length, width, color="#dc2626", alpha=1.0, lw=1.8, fill_alpha=0.20, zorder=12, wheelbase=wb)
        focus = [event_traj[event_step, :2], safe_traj[safe_step, :2]]
        if event_label == "COL":
            nearest = _nearest_neighbor_footprint(event_traj[event_step], event_step, neighbor_future, event_neighbor_tracks)
            if nearest is not None:
                _add_obb(
                    ax_end,
                    nearest["pose"],
                    nearest["length"],
                    nearest["width"],
                    color="#f97316",
                    alpha=1.0,
                    lw=1.8,
                    fill_alpha=0.24,
                    zorder=13,
                    label="recorded neighbor",
                )
                focus.append(nearest["pose"][:2])
        focus = np.asarray(focus)
        center_event = focus.mean(axis=0)
        event_extent = max(float(np.ptp(focus[:, 0])), float(np.ptp(focus[:, 1])), 8.0) / 2.0 + 4.0
        ax_end.set_xlim(center_event[0] - event_extent, center_event[0] + event_extent)
        ax_end.set_ylim(center_event[1] - event_extent, center_event[1] + event_extent)
        if event_label == "COL":
            margin_text = f"OBB={event_result['min_obb_clearance_m']:.2f}m"
        else:
            margin_text = (
                f"failed={event_result['rb_min_dist_m']:.3f}m < "
                f"gate={event_result['rb_gate_threshold_m']:.3f}m ≤ "
                f"safe={safe_result['rb_min_dist_m']:.3f}m"
            )
        ax_end.set_title(
            f"EVENT ZOOM · {event_label} @ {event_step / 10.0:.1f}s · {margin_text}\n"
            "green=safe · red=failed ego · orange=neighbor",
            fontsize=6.5,
            pad=2,
            weight="bold",
        )
        ax_end.legend(loc="upper left", fontsize=5.2, framealpha=0.88)
    else:
        endpoints = np.stack([r["traj"][_last_valid_index(r["traj"]), :2] for r in results])
        for rank, endpoint in enumerate(endpoints):
            color = cmap(rank / max(K - 1, 1))
            ax_end.scatter(endpoint[0], endpoint[1], s=22 if rank < 3 else 12, color=color, edgecolor="white", linewidth=0.35, zorder=3)
            if rank < 3:
                ax_end.text(endpoint[0], endpoint[1], f"{rank + 1}", fontsize=5.5, ha="center", va="center", zorder=4)
        gt_valid = np.flatnonzero(np.linalg.norm(gt[:, :2], axis=-1) > 1e-4)
        if gt_valid.size:
            ax_end.scatter(gt[gt_valid[-1], 0], gt[gt_valid[-1], 1], marker="*", s=42, color="#111827", label="GT", zorder=5)
        ax_end.set_title(
            f"endpoints · modes@2m={diversity['endpoint_modes_2m']}\n"
            f"μpair={diversity['endpoint_pairwise_mean_m']:.2f}m · yσ={diversity['endpoint_std_y_m']:.2f}m",
            fontsize=6.5, pad=2,
        )
    ax_end.tick_params(labelsize=5, length=2)
    ax_end.grid(True, alpha=0.25, lw=0.4)
    ax_end.set_aspect("equal", adjustable="box")

    # Top-line summary uses the actual hard gate.  Lane departure remains a
    # diagnostic because a legitimate lane change must not be painted red.
    n_in = sum(1 for r in results if not r["lane_crossing"])
    n_stopped = sum(1 for r in results if r["path_len"] < 1.0)
    n_hard = sum(1 for r in results if r.get("hard_reasons"))
    n_safe = K - n_hard
    best_weight = results[0].get("awr_weight")
    best_weight_text = "n/a" if best_weight is None else f"{best_weight:.2f}"

    title = scene_title or f"Scene {scene_idx}"
    ax_map.set_title(
        f"{title} · {tag} — hard-safe={n_safe}/{K}, hard-zero={n_hard}/{K}, "
        f"best AWR weight={best_weight_text}\n"
        f"lane diagnostic={n_in}/{K} IN (not a hard gate) · stopped={n_stopped}/{K} · GT={gt_path_len:.1f}m · "
        f"endpoint spread={diversity['endpoint_pairwise_mean_m']:.2f}m · "
        f"modes@2m={diversity['endpoint_modes_2m']} · "
        f"reward horizon={results[0].get('reward_horizon_s', 8.0):.1f}s",
        fontsize=10,
    )

    # Reward breakdown table
    headers = [
        "Rk\ncmd",
        "Total",
        "AWR w",
        "Prog",
        "TTC",
        "TTC OBB\nmin",
        "Hard\nevent",
        f"RB min\n(gate {results[0].get('rb_gate_threshold_m', 0.2):.3f})",
        "Lane\ndiag",
        "CL",
        "Smth",
        "Path",
    ]
    col_widths = [0.04, 0.075, 0.07, 0.07, 0.065, 0.085, 0.13, 0.075, 0.065, 0.065, 0.07, 0.08]

    # Header.  The diagnostic figures use the configured gate profile, so
    # expose that context inside the image instead of relying on a web
    # caption. This prevents near-one feasibility scores from being read as
    # a finely calibrated semantic driving ranking.
    table_mode = "gate-profile feasibility audit" if "GATE" in str(tag).upper() else "configured reward profile"
    ax_tbl.text(
        0.0,
        0.995,
        f"REWARD TABLE · {table_mode} · geometry/time explain every hard zero",
        fontsize=6.8,
        fontweight="bold",
        color="#334155",
        transform=ax_tbl.transAxes,
        va="top",
    )
    y = 0.945
    for j, h in enumerate(headers):
        x = sum(col_widths[:j]) + col_widths[j] / 2
        ax_tbl.text(
            x,
            y,
            h,
            fontsize=7,
            fontweight="bold",
            ha="center",
            va="top",
            transform=ax_tbl.transAxes,
        )
    # Multi-line geometry headers need breathing room before the divider.
    y -= 0.045
    ax_tbl.plot([0, 1], [y, y], color="black", lw=0.5, transform=ax_tbl.transAxes, clip_on=False)

    for rank, r in enumerate(results):
        y -= 0.055
        if y < 0.0:
            break

        color = cmap(rank / max(K - 1, 1))
        lc = "OUT" if r["lane_crossing"] else "IN"
        stopped = r["path_len"] < 1.0

        if r["collision"]:
            event = f"COL@{r['collision_time_s']:.1f}s"
        elif r["red_light"] < -0.5:
            event = "RED"
        elif r["rb_crossing"]:
            rb_time = r.get("rb_crossing_time_s")
            event = "RB" if rb_time is None else f"RB@{rb_time:.1f}s"
        elif r["kinematic"]:
            event = "KIN"
        else:
            event = "—"

        awr_weight = r.get("awr_weight")
        weight_text = "—" if awr_weight is None else f"{awr_weight:.2f}"
        obb_clearance = float(r.get("min_obb_clearance_m", 99.0))
        obb_text = "n/a" if obb_clearance >= 90.0 else f"{obb_clearance:.2f}m"
        rb_margin = float(r.get("rb_min_dist_m", 99.0))
        rb_text = "n/a" if rb_margin >= 1.0e5 else f"{rb_margin:.3f}m"

        row = [
            f"{rank + 1}"
            + (f"\n{r['candidate_label']}" if r.get("candidate_label") else ""),
            f"{r['total']:+.3f}",
            weight_text,
            f"{r['progress']:.1f}",
            f"{r.get('ttc_score', 1.0):.2f}",
            obb_text,
            event,
            rb_text,
            f"{lc}",
            f"{r['centerline']:.2f}",
            f"{r['smoothness']:.2f}",
            f"{r['path_len']:.1f}m",
        ]

        # Keep the table's visual semantics mutually exclusive.  A stopped
        # candidate is not automatically unsafe: in a red-light or stopped-
        # leader scene, stopping may be exactly the desired behavior.  Reserve
        # red for an actual hard violation so readers do not confuse "wait"
        # with "bad" when scanning the figure.
        if r["collision"] or r["kinematic"] or r["rb_crossing"]:
            bg_color = "#ffe0e0"  # red: hard violation always wins
        elif stopped:
            bg_color = "#fff4cc"  # amber: stopped, not itself a violation
        else:
            bg_color = "#e0f7e6"  # light green: feasible; lane is diagnostic-only

        # Draw background row highlight
        ax_tbl.fill_between(
            [0, 1],
            y - 0.025,
            y + 0.03,
            color=bg_color,
            alpha=0.5,
            transform=ax_tbl.transAxes,
            clip_on=False,
        )

        for j, val in enumerate(row):
            x = sum(col_widths[:j]) + col_widths[j] / 2
            fw = "bold" if rank < 3 else "normal"
            ax_tbl.text(
                x,
                y,
                val,
                fontsize=6.5,
                ha="center",
                va="center",
                fontweight=fw,
                color=color if rank < 5 else "#555",
                transform=ax_tbl.transAxes,
            )

    # Legend at bottom of table
    y_leg = max(y - 0.06, 0.01)
    ax_tbl.text(
        0.0,
        y_leg,
        "TTC OBB min = minimum signed clearance in TTC lookahead · RB min = ego-footprint↔road-border margin · Lane diag is NOT a hard gate · X = first COL/RB frame",
        fontsize=6,
        color="#666",
        transform=ax_tbl.transAxes,
    )


def render_dynamics_gif(
    results,
    npz_path,
    output_path,
    scene_idx,
    tag,
    *,
    scene_title=None,
    frame_stride=1,
    frame_duration_ms=100,
    reference_trajectory=None,
):
    """Render a compact BEV movie from the same colleague-scene primitives.

    The movie is intentionally open-loop: the ego candidates are fixed plans,
    while recorded neighbours advance on their aligned (+1) replay timeline.
    It is therefore a visualization of the reward geometry and candidate
    coverage, not a closed-loop success claim.
    """

    from PIL import Image

    npz = np.load(npz_path)
    gt = np.asarray(npz["ego_agent_future"])
    neighbor_future = _aligned_neighbor_future(npz)
    tracks = _neighbor_tracks(npz, neighbor_future)
    event_tracks = _neighbor_tracks(npz, neighbor_future, limit=None)
    hero_event = _hero_hard_event(results)
    horizon = min([len(r["traj"]) for r in results] + [len(gt), neighbor_future.shape[1]])
    frames = list(range(0, max(horizon, 1), max(int(frame_stride), 1)))
    if not frames or frames[-1] != horizon - 1:
        frames.append(max(horizon - 1, 0))

    # Fixed camera across frames: motion remains interpretable instead of
    # zooming with the active candidate.
    reference = None
    if reference_trajectory is not None:
        candidate_reference = np.asarray(reference_trajectory)
        if candidate_reference.ndim == 2 and candidate_reference.shape[1] >= 2:
            reference = candidate_reference
    all_xy = _camera_points(gt, results, neighbor_future, tracks)
    if reference is not None:
        valid_reference = reference[:, :2]
        valid_reference = valid_reference[np.isfinite(valid_reference).all(axis=-1)]
        if len(valid_reference):
            all_xy = np.concatenate([all_xy, valid_reference], axis=0)
    center = all_xy.mean(axis=0)
    extent = max(np.ptp(all_xy[:, 0]), np.ptp(all_xy[:, 1])) * 0.55 + 10.0

    wb, length, width = map(
        float, effective_ego_shape_numpy(npz_path, npz.get("ego_shape"))[:3]
    )
    event_camera = None
    if hero_event is not None:
        event_result = hero_event["result"]
        event_step = hero_event["step"]
        event_traj = np.asarray(event_result["traj"])
        safe_result = next((r for r in results if not r.get("hard_reasons")), results[0])
        safe_traj = np.asarray(safe_result["traj"])
        safe_step = min(event_step, len(safe_traj) - 1)
        focus = [event_traj[event_step, :2], safe_traj[safe_step, :2]]
        nearest = None
        if hero_event["label"] == "COL":
            nearest = _nearest_neighbor_footprint(
                event_traj[event_step], event_step, neighbor_future, event_tracks
            )
            if nearest is not None:
                focus.append(nearest["pose"][:2])
        focus = np.asarray(focus, dtype=float)
        event_center = focus.mean(axis=0)
        event_extent = max(
            float(np.ptp(focus[:, 0])), float(np.ptp(focus[:, 1])), 8.0
        ) / 2.0 + 4.0
        event_camera = {
            "result": event_result,
            "traj": event_traj,
            "safe_traj": safe_traj,
            "safe_result": safe_result,
            "step": event_step,
            "label": hero_event["label"],
            "rank": hero_event["rank"],
            "nearest": nearest,
            "center": event_center,
            "extent": event_extent,
        }
    past = np.asarray(npz.get("neighbor_agents_past", np.zeros((0, 1, 11))))
    cmap = plt.cm.RdYlGn_r
    diversity = diversity_metrics(results)
    frames_out = []

    for t in frames:
        fig = plt.figure(figsize=(11, 7), facecolor="#f8fafc")
        ax = fig.add_axes([0.04, 0.08, 0.92, 0.84])
        _draw_map_context(ax, npz)
        ax.plot(gt[:, 0], gt[:, 1], "g--", lw=2.0, alpha=0.65, label="logged expert")
        if reference is not None:
            ref_upto = min(t + 1, len(reference))
            ax.plot(
                reference[:ref_upto, 0],
                reference[:ref_upto, 1],
                color="#111827",
                ls=(0, (1.5, 2.0)),
                lw=1.8,
                alpha=0.82,
                zorder=7,
                label="frozen IL reference",
            )
            if ref_upto:
                _add_obb(
                    ax,
                    reference[ref_upto - 1],
                    length,
                    width,
                    color="#111827",
                    alpha=0.42,
                    lw=0.9,
                    zorder=9,
                    wheelbase=wb,
                )

        # Candidate paths are colored by the static reward rank from the
        # corresponding PNG, so the movie and still tell the same story.
        violations_so_far = 0
        clearance_now = []
        for rank, result in enumerate(results):
            traj = np.asarray(result["traj"])
            color = cmap(rank / max(len(results) - 1, 1))
            upto = min(t + 1, len(traj))
            ax.plot(traj[:upto, 0], traj[:upto, 1], color=color, lw=2.0 if rank < 3 else 0.8, alpha=0.85 if rank < 4 else 0.22, zorder=8)
            if upto:
                _add_obb(ax, traj[upto - 1], length, width, color=color, alpha=0.28 if rank < 3 else 0.08, lw=1.0 if rank < 3 else 0.5, zorder=10, wheelbase=wb)
            event_step = result.get("collision_step")
            event_label = "COL"
            if event_step is None:
                event_step = result.get("rb_crossing_step")
                event_label = "RB"
            if event_step is not None and int(event_step) <= t:
                violations_so_far += 1
            if event_step is not None and int(event_step) <= t and int(event_step) < len(traj):
                event_xy = traj[int(event_step), :2]
                ax.scatter(
                    event_xy[0], event_xy[1], marker="X", s=95,
                    color="#dc2626", edgecolor="white", linewidth=1.0, zorder=25,
                )
                if hero_event is None or rank == hero_event["rank"]:
                    ax.annotate(
                        f"#{rank + 1} {event_label} @ {int(event_step) / 10.0:.1f}s",
                        xy=event_xy, xytext=(8, 9), textcoords="offset points",
                        fontsize=8, color="#991b1b", weight="bold", zorder=26,
                    )
            series = result.get("_clearance_series_m")
            if series is not None and t < len(series):
                value = float(series[t])
                if value < 90.0:
                    clearance_now.append(value)

        # Logged ego footprint and replayed neighbour footprints at the same
        # frame.  Neighbour shape is taken from the recorded current box.
        if t < len(gt):
            gt_yaw = float(gt[t, 2]) if len(gt[t]) > 2 else 0.0
            gt_pose = np.array([gt[t, 0], gt[t, 1], np.cos(gt_yaw), np.sin(gt_yaw)])
            _add_obb(ax, gt_pose, length, width, color="#15803d", alpha=0.72, lw=1.6, zorder=12, label="logged ego", wheelbase=wb)
        for track_rank, (_, p, cur, valid) in enumerate(tracks):
            fut = neighbor_future[p]
            seen = min(t + 1, len(fut))
            observed = valid[:seen]
            if not observed.any():
                continue
            nt = int(np.flatnonzero(observed)[-1])
            n_length = float(cur[7]) if cur.shape[0] > 7 and cur[7] > 0.1 else 4.5
            n_width = float(cur[6]) if cur.shape[0] > 6 and cur[6] > 0.1 else 1.8
            yaw = float(fut[nt, 2])
            pose = np.array([fut[nt, 0], fut[nt, 1], np.cos(yaw), np.sin(yaw)])
            color = ["#64748b", "#0f766e", "#b45309", "#7c3aed", "#be123c", "#0369a1"][track_rank % 6]
            _add_obb(ax, pose, n_length, n_width, color=color, alpha=0.88, lw=1.1, zorder=13, label="recorded neighbor" if track_rank == 0 else None)
            ax.plot(fut[:seen, 0], fut[:seen, 1], "--", color=color, alpha=0.35, lw=0.8, zorder=5)

        # Persist the exact hard-event geometry as a translucent snapshot.
        # Current boxes keep moving, while this red/orange pair shows where
        # the gate fired and makes the causal frame recoverable on pause.
        if hero_event is not None and t >= hero_event["step"]:
            event_result = hero_event["result"]
            event_step = hero_event["step"]
            event_pose = np.asarray(event_result["traj"])[event_step]
            _add_obb(
                ax,
                event_pose,
                length,
                width,
                color="#dc2626",
                alpha=0.95,
                lw=2.2,
                fill_alpha=0.14,
                zorder=23,
                label="failed ego @ gate frame",
                wheelbase=wb,
            )
            if hero_event["label"] == "COL":
                nearest = _nearest_neighbor_footprint(event_pose, event_step, neighbor_future, event_tracks)
                if nearest is not None:
                    _add_obb(
                        ax,
                        nearest["pose"],
                        nearest["length"],
                        nearest["width"],
                        color="#f97316",
                        alpha=0.95,
                        lw=2.2,
                        fill_alpha=0.18,
                        zorder=24,
                        label="neighbor @ collision frame",
                    )

        # A fixed metric-scale event camera keeps the physical overlap/border
        # legible even when the full 8 s path spans 70+ metres.  It is dynamic
        # before the gate and freezes a translucent event snapshot afterwards.
        if event_camera is not None:
            zoom = ax.inset_axes([0.61, 0.045, 0.36, 0.31], facecolor="white", zorder=30)
            _draw_map_context(zoom, npz)
            fail_traj = event_camera["traj"]
            safe_traj = event_camera["safe_traj"]
            fail_now = min(t, len(fail_traj) - 1)
            safe_now = min(t, len(safe_traj) - 1)
            zoom.plot(safe_traj[:, 0], safe_traj[:, 1], color="#15803d", lw=0.7, alpha=0.20)
            zoom.plot(fail_traj[:, 0], fail_traj[:, 1], color="#dc2626", lw=0.7, alpha=0.20)
            zoom.plot(safe_traj[: safe_now + 1, 0], safe_traj[: safe_now + 1, 1], color="#15803d", lw=1.5)
            zoom.plot(fail_traj[: fail_now + 1, 0], fail_traj[: fail_now + 1, 1], color="#dc2626", lw=1.6)
            _add_obb(
                zoom, safe_traj[safe_now], length, width,
                color="#15803d", alpha=0.95, lw=1.2, fill_alpha=0.10,
                zorder=12, wheelbase=wb,
            )
            _add_obb(
                zoom, fail_traj[fail_now], length, width,
                color="#dc2626", alpha=0.95, lw=1.4, fill_alpha=0.12,
                zorder=13, wheelbase=wb,
            )
            nearest = event_camera["nearest"]
            if nearest is not None:
                slot = int(nearest["slot"])
                if t < neighbor_future.shape[1]:
                    xy = neighbor_future[slot, t, :2]
                    if np.linalg.norm(xy) > 1e-4 and np.isfinite(xy).all():
                        yaw = float(neighbor_future[slot, t, 2])
                        pose_now = np.array([xy[0], xy[1], np.cos(yaw), np.sin(yaw)])
                        _add_obb(
                            zoom, pose_now, nearest["length"], nearest["width"],
                            color="#f97316", alpha=0.95, lw=1.4,
                            fill_alpha=0.14, zorder=14,
                        )
            if t >= event_camera["step"]:
                gate_step = event_camera["step"]
                gate_pose = fail_traj[gate_step]
                _add_obb(
                    zoom, gate_pose, length, width,
                    color="#991b1b", alpha=1.0, lw=1.8, fill_alpha=0.14,
                    zorder=20, wheelbase=wb,
                )
                if nearest is not None:
                    _add_obb(
                        zoom, nearest["pose"], nearest["length"], nearest["width"],
                        color="#c2410c", alpha=1.0, lw=1.8,
                        fill_alpha=0.16, zorder=21,
                    )
                zoom.scatter(
                    gate_pose[0], gate_pose[1], marker="X", s=48,
                    color="#dc2626", edgecolor="white", linewidth=0.6, zorder=22,
                )
            ec, ee = event_camera["center"], event_camera["extent"]
            zoom.set_xlim(ec[0] - ee, ec[0] + ee)
            zoom.set_ylim(ec[1] - ee, ec[1] + ee)
            zoom.set_aspect("equal", adjustable="box")
            zoom.grid(True, alpha=0.18, lw=0.5)
            gate_state = "GATE" if t >= event_camera["step"] else "approaching"
            if event_camera["label"] == "RB":
                failed = event_camera["result"]
                safe = event_camera["safe_result"]
                gate_detail = (
                    f"{failed['rb_min_dist_m']:.3f} < "
                    f"{failed['rb_gate_threshold_m']:.3f} ≤ "
                    f"{safe['rb_min_dist_m']:.3f}m"
                )
            else:
                gate_detail = gate_state
            zoom.set_title(
                f"EVENT CAMERA · {event_camera['label']} @ {event_camera['step']/10:.1f}s · "
                f"{gate_detail}",
                fontsize=7.2, weight="bold", pad=2,
            )
            zoom.tick_params(labelsize=5, length=2)
            for spine in zoom.spines.values():
                spine.set_edgecolor("#334155")
                spine.set_linewidth(1.1)

        ax.set_xlim(center[0] - extent, center[0] + extent)
        ax.set_ylim(center[1] - extent, center[1] + extent)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.18)
        ax.set_xlabel("ego-frame x (m)")
        ax.set_ylabel("ego-frame y (m)")
        scored_horizon_s = float(results[0].get("reward_horizon_s", horizon / 10.0))
        phase = (
            f"SCORED 0–{scored_horizon_s:.0f}s"
            if t / 10.0 < scored_horizon_s
            else f"CONTEXT after {scored_horizon_s:.0f}s"
        )
        best_weight = results[0].get("awr_weight")
        hard_weight = None if hero_event is None else hero_event["result"].get("awr_weight")
        weight_summary = (
            "AWR weights unavailable"
            if best_weight is None
            else f"AWR keeps top={best_weight:.2f}"
            + ("" if hard_weight is None else f" vs hard-event={hard_weight:.2f}")
        )
        ax.set_title(
            f"{scene_title or f'Scene {scene_idx}'} · {tag} · t={t / 10.0:.1f}s\n"
            f"{phase} · K={len(results)} complete paths · neighbor +1 aligned · "
            f"hard so far={violations_so_far}/{len(results)}\n"
            f"TTC-lookahead OBB min="
            f"{(f'{min(clearance_now):.2f}m' if clearance_now else 'n/a')} · "
            f"{weight_summary} · endpoint μpair={diversity['endpoint_pairwise_mean_m']:.2f}m",
            fontsize=10,
        )
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames_out.append(Image.fromarray(rgba[..., :3].copy()))
        plt.close(fig)

    if frames_out:
        frames_out[0].save(
            output_path,
            save_all=True,
            append_images=frames_out[1:],
            duration=int(frame_duration_ms),
            loop=0,
            optimize=False,
        )


def main():
    parser = argparse.ArgumentParser(
        description="GRPO trajectory visualization with reward breakdown"
    )
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--scenes", type=str, default=None)
    parser.add_argument(
        "--replay_cache_root",
        type=str,
        default=None,
        help="render exact frozen K-groups from a formal disk replay cache",
    )
    parser.add_argument("--replay_rank", type=int, default=0)
    parser.add_argument(
        "--replay_indices",
        type=int,
        nargs="*",
        default=None,
        help="rank-local scene-group indices for --replay_cache_root",
    )
    parser.add_argument(
        "--allow_partial_replay",
        action="store_true",
        help="visualization only: read JSONL-committed rows before strict rank close",
    )
    parser.add_argument(
        "--replay_positive_overlay",
        action="store_true",
        help=(
            "replace cached mining weights with the exact positive-advantage "
            "overlay weights used by formal replay"
        ),
    )
    parser.add_argument("--replay_overlay_beta", type=float, default=2.0)
    parser.add_argument("--replay_overlay_margin", type=float, default=0.01)
    parser.add_argument("--replay_behavior_anchor_weight", type=float, default=0.0)
    parser.add_argument(
        "--replay_unsafe_behavior_anchor_weight", type=float, default=0.0
    )
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--n_scenes", type=int, default=5)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--tag", type=str, default="model")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--awr_mode",
        action="store_true",
        help="use the original-DP AWR sampler instead of GRPO guidance",
    )
    parser.add_argument(
        "--reward_config",
        type=str,
        default=None,
        help="JSON config; accepts the nested reward object from an AWR config",
    )
    parser.add_argument(
        "--reward_horizon_steps",
        type=int,
        default=None,
        help=(
            "explicit visualization rescore horizon for the same frozen trajectories; "
            "use paired 40/80 renders to audit the Original-DP reward tail"
        ),
    )
    parser.add_argument("--sample_steps", type=int, default=5)
    parser.add_argument("--noise_scale", type=float, default=0.5)
    parser.add_argument(
        "--deterministic_first",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="keep the zero-noise DP behavior member as candidate 0",
    )
    parser.add_argument("--hdp_trajectory_augmentation", action="store_true")
    parser.add_argument("--hdp_trajectory_augmentation_std", type=float, default=0.5)
    parser.add_argument(
        "--gif",
        action="store_true",
        help="also render a BEV dynamics GIF from the same scene/bbox primitives",
    )
    parser.add_argument(
        "--multimodality_compare",
        action="store_true",
        help="also render paired raw-DP vs HDP-style-augmentation multimodality figures/GIFs",
    )
    parser.add_argument(
        "--multimodality_endpoint_threshold_m",
        type=float,
        default=1.0,
        help="endpoint-component radius used only for the visual audit (metres)",
    )
    parser.add_argument(
        "--gif_frame_stride",
        type=int,
        default=1,
        help="sample every N of the 80 recorded 10 Hz frames; default 1 keeps the full dynamics",
    )
    parser.add_argument("--gif_frame_duration_ms", type=int, default=100)
    parser.add_argument(
        "--scene_meta",
        type=str,
        default=None,
        help="optional metadata JSON keyed by scene index; labels are merged into the figure",
    )
    # Reward config
    parser.add_argument("--survival", action="store_true")
    parser.add_argument("--enable_lane", action="store_true")
    parser.add_argument("--lane_gate", action="store_true")
    parser.add_argument("--w_progress", type=float, default=3.0)
    parser.add_argument("--rb_near_scale", type=float, default=5.0)
    parser.add_argument("--rb_wide_scale", type=float, default=0.5)
    parser.add_argument("--lane_near_scale", type=float, default=30.0)
    parser.add_argument("--lane_wide_scale", type=float, default=10.0)
    parser.add_argument("--lane_cont_scale", type=float, default=5.0)
    parser.add_argument("--stopped_penalty", type=float, default=50.0)
    parser.add_argument("--underprogress_penalty", type=float, default=100.0)
    parser.add_argument("--underprogress_threshold", type=float, default=0.5)
    parser.add_argument("--progress_norm_scale", type=float, default=20.0)
    parser.add_argument("--overprogress_margin", type=float, default=1.0)
    parser.add_argument("--overprogress_penalty", type=float, default=3.0)
    args = parser.parse_args()

    # Keep the conference artifacts reproducible.  AWR's diversity is real
    # diffusion sampling, but a figure should not change merely because the
    # script was launched twice.
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    replay_mode = args.replay_cache_root is not None
    if replay_mode:
        if not args.replay_indices:
            parser.error("--replay_cache_root requires --replay_indices")
        if args.multimodality_compare:
            parser.error("multimodality comparison requires live model sampling")
        # Frozen groups need no model and deliberately stay on CPU so report
        # rendering cannot steal memory or cycles from the formal 8-GPU run.
        from rlvr.train_awr import _DiskReplayReader

        device = torch.device("cpu")
        reader_type = _CommittedPrefixReplayReader if args.allow_partial_replay else _DiskReplayReader
        replay_reader = reader_type(Path(args.replay_cache_root), int(args.replay_rank))
        args.K = int(replay_reader.group_size)
        model = model_args = None
    else:
        if args.replay_positive_overlay:
            parser.error("--replay_positive_overlay requires --replay_cache_root")
        if not args.model_path or not args.scenes:
            parser.error("live sampling requires --model_path and --scenes")
        replay_reader = None
        device = torch.device(DEVICE)
        model_dir = Path(args.model_path).parent
        args_path = model_dir / "args.json"
        if not args_path.exists():
            args_path = model_dir.parent / "args.json"
        if args.awr_mode:
            # Use the exact original-DP checkpoint ABI and EMA selection used by
            # the AWR trainer. The old GRPO path remains available for comparison,
            # but it is not used for the proposal's evidence.
            model, model_args = load_original_dp_checkpoint(
                args.model_path, device, args_path=args_path, use_ema=True
            )
        else:
            model_args = Config(str(args_path))
            model = Diffusion_Planner(model_args)
            ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
            state = ckpt.get("model", ckpt)
            state = {k.replace("module.", ""): v for k, v in state.items()}
            model.load_state_dict(state)
            model.to(device)
        if args.lora_path:
            model = load_lora_checkpoint(model, args.lora_path)
        model.eval()

    if args.reward_config:
        reward_payload = json.loads(Path(args.reward_config).read_text())
        reward_payload = reward_payload.get("reward", reward_payload)
        allowed = {field.name for field in fields(RewardConfig)}
        reward_kwargs = {
            key: value
            for key, value in reward_payload.items()
            if key in allowed and not key.startswith("_")
        }
        rcfg = RewardConfig(**reward_kwargs)
    else:
        rcfg = RewardConfig(
            enable_lane_departure=args.enable_lane,
            lane_gate_enabled=args.lane_gate,
            w_progress=args.w_progress,
            rb_near_scale=args.rb_near_scale,
            rb_wide_scale=args.rb_wide_scale,
            lane_near_scale=args.lane_near_scale,
            lane_wide_scale=args.lane_wide_scale,
            lane_cont_scale=args.lane_cont_scale,
            reward_mode="survival" if args.survival else "gate",
            enable_overprogress=True,
            overprogress_margin=args.overprogress_margin,
            overprogress_penalty=args.overprogress_penalty,
            stopped_penalty=args.stopped_penalty,
            underprogress_penalty=args.underprogress_penalty,
            underprogress_threshold=args.underprogress_threshold,
            progress_norm_scale=args.progress_norm_scale,
        )
    # ``--survival`` is an explicit CLI override.  It must also work when a
    # formal nested reward config is supplied; otherwise a command that says
    # survival would silently render the config's gate totals.
    if args.survival:
        rcfg.reward_mode = "survival"
    if args.reward_horizon_steps is not None:
        if int(args.reward_horizon_steps) < 1:
            parser.error("--reward_horizon_steps must be positive")
        rcfg.reward_horizon_steps = int(args.reward_horizon_steps)

    sampler_cfg = SamplerConfig(
        n_trajectories=args.K,
        enable_guidance=True,
        enable_centerline=True,
        enable_lane_keeping=True,
        enable_road_border=True,
        enable_speed=True,
        guidance_prob=0.7,
    )

    if replay_mode:
        scenes = None
    else:
        with open(args.scenes) as f:
            scenes = json.load(f)
    scene_meta = {}
    if args.scene_meta:
        scene_meta = json.loads(Path(args.scene_meta).read_text())

    if replay_mode:
        scene_indices = [int(index) for index in args.replay_indices]
    elif args.indices:
        scene_indices = args.indices
    else:
        scene_indices = list(range(min(args.n_scenes, len(scenes))))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "tool": "rlvr/autoresearch/tools/grpo_viz.py",
        "mode": "formal_replay_cache" if replay_mode else (
            "original_dp_awr" if args.awr_mode else "grpo_guidance"
        ),
        "model_path": str(Path(args.model_path).resolve()) if args.model_path else None,
        "replay_cache_root": str(Path(args.replay_cache_root).resolve())
        if replay_mode
        else None,
        "replay_rank": int(args.replay_rank) if replay_mode else None,
        "replay_source_complete": bool(
            replay_mode and not args.allow_partial_replay
        ),
        "partial_replay_boundary": (
            "scene_paths.jsonl committed rows; visualization only"
            if replay_mode and args.allow_partial_replay
            else None
        ),
        "replay_weight_source": (
            "recomputed positive-advantage overlay from immutable cached rewards"
            if replay_mode and args.replay_positive_overlay
            else "stored replay weights"
        ),
        "replay_positive_overlay": (
            {
                "beta": float(args.replay_overlay_beta),
                "margin": float(args.replay_overlay_margin),
                "behavior_anchor_weight": float(args.replay_behavior_anchor_weight),
                "unsafe_behavior_anchor_weight": float(
                    args.replay_unsafe_behavior_anchor_weight
                ),
                "drop_all_zero_groups": True,
            }
            if replay_mode and args.replay_positive_overlay
            else None
        ),
        "reward_config": str(Path(args.reward_config).resolve()) if args.reward_config else None,
        "reward_mode": str(rcfg.reward_mode),
        "reward_horizon_steps": int(rcfg.reward_horizon_steps),
        "K": int(args.K),
        "visualization_tag": str(args.tag),
        "seed": int(args.seed),
        "sample_steps": int(args.sample_steps),
        "noise_scale": float(args.noise_scale),
        "deterministic_first": bool(args.deterministic_first),
        "hdp_trajectory_augmentation": bool(args.hdp_trajectory_augmentation),
        "hdp_trajectory_augmentation_std": float(args.hdp_trajectory_augmentation_std),
        "neighbor_future_alignment": "raw[t+1] -> aligned[t] (offset=1)",
        "visualization_basis": "colleague grpo_viz.py, extended for original-DP AWR",
        "gif": bool(args.gif),
        "gif_frame_stride": int(args.gif_frame_stride),
        "gif_frame_duration_ms": int(args.gif_frame_duration_ms),
        "multimodality_compare": bool(args.multimodality_compare),
        "multimodality_endpoint_threshold_m": float(args.multimodality_endpoint_threshold_m),
        "render_version": "t4_original_dp_awr_scene_v3_reward_geometry",
        "visualization_role": (
            "exact frozen K=10 groups used by the formal full-sequence replay"
            if replay_mode
            else "K=16 source-checkpoint coverage diagnostic"
        ),
        "formal_run_reference": (
            "trajectories/rewards/weights are read directly from epoch-1 formal replay"
            if replay_mode
            else "formal full-sequence AWR uses K=10; live figures use K=16 for coverage audit"
        ),
        "table_color_semantics": {
            "green": "hard-feasible candidate; lane diagnostic may be IN or OUT",
            "amber": "stopped (<1 m path length); not itself a safety violation",
            "red": "collision, road-border, or kinematic hard violation",
        },
        "table_columns": "Rk, Total, AWR w, Prog, TTC, TTC-lookahead OBB min, Hard event, RB min, Lane diagnostic, CL, Smth, Path",
        "table_total_decimals": 3,
        "table_reward_context": (
            f"{rcfg.reward_mode}-profile feasibility audit; Total is not a maneuver label"
        ),
        "scenes": {},
    }

    def _reset_sampling_seed():
        # The paired raw/augmented audit should start from the same diffusion
        # noise.  The augmentation then becomes the only intended difference.
        torch.manual_seed(int(args.seed))
        np.random.seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))

    for output_index, si in enumerate(scene_indices):
        print(f"Processing scene {si}...")
        if replay_mode:
            replay_paths, replay_rollout = replay_reader.batch_from_indices([si])
            scene_path = replay_paths[0]
            scene_title = f"Formal replay group r{args.replay_rank}:{si}"
            entry_meta = {
                "source": "formal_epoch_1_replay_cache",
                "rank": int(args.replay_rank),
                "rank_local_index": int(si),
            }
            cached_reward_row = np.array(
                replay_reader.rewards[int(si)], dtype=np.float32, copy=True
            )
            if args.replay_positive_overlay:
                from rlvr.autoresearch.tools.build_positive_anchor_replay_overlay import (
                    compute_positive_anchor_weights,
                )

                overlay_weights = compute_positive_anchor_weights(
                    cached_reward_row[None, :],
                    beta=float(args.replay_overlay_beta),
                    margin=float(args.replay_overlay_margin),
                    behavior_anchor_weight=float(args.replay_behavior_anchor_weight),
                    unsafe_behavior_anchor_weight=float(
                        args.replay_unsafe_behavior_anchor_weight
                    ),
                    normalize=False,
                    drop_all_zero_groups=True,
                )[0]
                replay_rollout.weights = torch.from_numpy(overlay_weights)
                entry_meta["weight_source"] = (
                    "formal positive-advantage overlay recomputed from cached rewards"
                )
        else:
            scene_path, scene_title, entry_meta = _scene_entry(scenes[si])
        meta_override = scene_meta.get(str(si), scene_meta.get(si, {}))
        if isinstance(meta_override, dict):
            scene_title = str(meta_override.get("title") or scene_title)
            entry_meta = {**entry_meta, **meta_override}
        raw_results = aug_results = None
        if replay_mode:
            data = load_npz_data(scene_path, device)
            es = data.get("ego_shape")
            ego_shape = es[0] if es is not None and es.dim() > 1 else es
            results = score_trajectory_group(
                replay_rollout.trajectories,
                data,
                rcfg,
                ego_shape=ego_shape,
                replay_weights=replay_rollout.weights,
            )
            # The figure may explain a frozen training group only if its live
            # reward scorer reproduces the immutable cache.  This catches a
            # wrong reward config, neighbour offset, ego shape, or later scorer
            # drift before a visually plausible but false artifact is emitted.
            rescored_by_candidate = np.full(args.K, np.nan, dtype=np.float32)
            for result in results:
                rescored_by_candidate[int(result["k"])] = float(result["total"])
            if not np.allclose(
                rescored_by_candidate,
                cached_reward_row,
                rtol=0.0,
                atol=1e-5,
                equal_nan=False,
            ):
                max_delta = float(
                    np.max(np.abs(rescored_by_candidate - cached_reward_row))
                )
                raise RuntimeError(
                    "visualization reward does not reproduce frozen replay "
                    f"cache for rank={args.replay_rank}, index={si}: "
                    f"max_abs_delta={max_delta:.8f}"
                )
            gt_path_len = 0.0
            gt = data.get("ego_agent_future")
            if isinstance(gt, torch.Tensor):
                gt = gt[0] if gt.dim() == 3 else gt
                valid = gt[:, :2].abs().sum(dim=-1) > 0.1
                if int(valid.sum()) >= 2:
                    gt_path_len = float(
                        torch.diff(gt[valid, :2], dim=0).norm(dim=-1).sum().item()
                    )
        elif args.multimodality_compare:
            _reset_sampling_seed()
            raw_results, raw_data, raw_gt_path_len = generate_and_score(
                model,
                model_args,
                scene_path,
                args.K,
                rcfg,
                sampler_cfg,
                awr_mode=args.awr_mode,
                sample_steps=args.sample_steps,
                noise_scale=args.noise_scale,
                deterministic_first=args.deterministic_first,
                hdp_trajectory_augmentation=False,
                hdp_trajectory_augmentation_std=args.hdp_trajectory_augmentation_std,
            )
            _reset_sampling_seed()
            aug_results, aug_data, aug_gt_path_len = generate_and_score(
                model,
                model_args,
                scene_path,
                args.K,
                rcfg,
                sampler_cfg,
                awr_mode=args.awr_mode,
                sample_steps=args.sample_steps,
                noise_scale=args.noise_scale,
                deterministic_first=args.deterministic_first,
                hdp_trajectory_augmentation=True,
                hdp_trajectory_augmentation_std=args.hdp_trajectory_augmentation_std,
            )
            if args.hdp_trajectory_augmentation:
                results, data, gt_path_len = aug_results, aug_data, aug_gt_path_len
            else:
                results, data, gt_path_len = raw_results, raw_data, raw_gt_path_len
        else:
            results, data, gt_path_len = generate_and_score(
                model,
                model_args,
                scene_path,
                args.K,
                rcfg,
                sampler_cfg,
                awr_mode=args.awr_mode,
                sample_steps=args.sample_steps,
                noise_scale=args.noise_scale,
                deterministic_first=args.deterministic_first,
                hdp_trajectory_augmentation=args.hdp_trajectory_augmentation,
                hdp_trajectory_augmentation_std=args.hdp_trajectory_augmentation_std,
            )
        diversity = diversity_metrics(results)

        fig = plt.figure(figsize=(18, 10), facecolor="#f8fafc")
        draw_grpo_scene(
            fig,
            results,
            scene_path,
            si,
            gt_path_len,
            args.tag,
            scene_title=scene_title,
            diversity=diversity,
        )

        fname = out / f"scene_{output_index:03d}.png"
        fig.savefig(
            fname,
            dpi=150,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            transparent=False,
        )
        plt.close(fig)
        print(f"  Saved: {fname}")

        gif_name = None
        if args.gif:
            gif_path = out / f"scene_{output_index:03d}.gif"
            render_dynamics_gif(
                results,
                scene_path,
                gif_path,
                si,
                args.tag,
                scene_title=scene_title,
                frame_stride=args.gif_frame_stride,
                frame_duration_ms=args.gif_frame_duration_ms,
            )
            gif_name = gif_path.name
            print(f"  Saved: {gif_path}")

        multimodality_assets = {}
        multimodality_manifest = None
        if args.multimodality_compare:
            multi_path = out / f"scene_{output_index:03d}_multimodality.png"
            multi_fig = plt.figure(figsize=(18, 12), facecolor="#f8fafc")
            draw_multimodality_comparison(
                multi_fig,
                np.load(scene_path),
                scene_path,
                raw_results,
                aug_results,
                scene_title=scene_title,
                endpoint_threshold=args.multimodality_endpoint_threshold_m,
            )
            multi_fig.savefig(multi_path, dpi=160, bbox_inches="tight")
            plt.close(multi_fig)
            multimodality_assets["figure"] = multi_path.name
            print(f"  Saved: {multi_path}")

            if args.gif:
                multi_gif_path = out / f"scene_{output_index:03d}_multimodality.gif"
                render_multimodality_gif(
                    raw_results,
                    aug_results,
                    scene_path,
                    multi_gif_path,
                    scene_title=scene_title,
                    endpoint_threshold=args.multimodality_endpoint_threshold_m,
                    frame_duration_ms=args.gif_frame_duration_ms,
                )
                multimodality_assets["gif"] = multi_gif_path.name
                print(f"  Saved: {multi_gif_path}")
            multimodality_manifest = {
                "pairing": "same scene and reset diffusion seed; raw vs augmentation is the intended difference",
                "endpoint_component_threshold_m": float(args.multimodality_endpoint_threshold_m),
                "raw": diversity_metrics(raw_results),
                "augmented": diversity_metrics(aug_results),
                "semantic_warning": "endpoint components are geometric audit clusters, not maneuver labels",
            }

        assets = {"figure": fname.name}
        if gif_name:
            assets["gif"] = gif_name
        if multimodality_assets:
            assets["multimodality"] = multimodality_assets
        scene_manifest = {
            "scene_path": str(Path(scene_path).resolve()),
            "title": scene_title,
            "metadata": entry_meta,
            "gt_path_len_m": float(gt_path_len),
            "assets": assets,
            "diversity": diversity,
            "rows": [
                {
                    key: value
                    for key, value in row.items()
                    if key != "traj" and not key.startswith("_")
                }
                for row in results
            ],
        }
        if multimodality_manifest is not None:
            scene_manifest["multimodality"] = multimodality_manifest
        (out / f"scene_{output_index:03d}.json").write_text(
            json.dumps(scene_manifest, indent=2, ensure_ascii=False) + "\n"
        )
        manifest["scenes"][str(output_index)] = scene_manifest

        # Print quick summary
        n_in = sum(1 for r in results if not r["lane_crossing"])
        n_stopped = sum(1 for r in results if r["path_len"] < 1.0)
        print(
            f"  {n_in}/{args.K} in-lane, {n_stopped} stopped, "
            f"best={results[0]['total']:+.1f}, worst={results[-1]['total']:+.1f}, "
            f"endpoint_pair={diversity['endpoint_pairwise_mean_m']:.2f}m, "
            f"modes@2m={diversity['endpoint_modes_2m']}"
        )

    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"\nDone. {len(scene_indices)} figures saved to {out}")


if __name__ == "__main__":
    main()
