#!/usr/bin/env python3
"""Generate report assets from the local Tier-IV/T4 RL artifacts.

This script intentionally keeps the report's plots reproducible from files that
already live in this checkout: RL ``train_log.tsv`` files, the sampled NPZ
manifests, and the causal red-light audit.  It also renders one logged NPZ
scene and a short GIF of that scene's recorded future, so the report contains
data-derived dynamics rather than only paper illustrations.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "t4_rl_assets"
OUT.mkdir(parents=True, exist_ok=True)

G8_LOG = ROOT / "outputs/hdp_route_sft20_latest_rl_g8u4_base_v1_node01/train_log.tsv"
G32_LOG = ROOT / (
    "outputs/hdp_route_sft20_latest_rl_stable_current_g32_beta1_lr1e7_"
    "dp0_ema010_distance_red_ep6_node01/train_log.tsv"
)
BASELINE_JSON = ROOT / (
    "outputs/hdp_route_sft20_latest_rl_stable_current_g32_beta1_lr1e7_"
    "dp0_ema010_distance_red_ep6_node01/source_baseline_metrics.json"
)
BASE_MANIFEST = ROOT / "artifacts/rl_train_real/base_uniform_100000.json"
VALID_MANIFEST = ROOT / "artifacts/rl_train_real/valid_uniform_4096.json"
RED_AUDIT = ROOT / "docs/causal_red_light_candidate_audit_20260715.md"


def read_tsv(path: Path) -> list[dict[str, float | str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    out: list[dict[str, float | str]] = []
    for row in rows:
        parsed: dict[str, float | str] = {}
        for key, value in row.items():
            if value in (None, ""):
                parsed[key] = value or ""
                continue
            if value in ("True", "False"):
                parsed[key] = value == "True"
                continue
            try:
                parsed[key] = float(value)
            except ValueError:
                parsed[key] = value
        out.append(parsed)
    return out


def savefig(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#d9e2ec", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#9fb3c8")
    ax.spines["bottom"].set_color("#9fb3c8")


def add_value_labels(ax: plt.Axes, bars, fmt: str = "{:.3f}") -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#243b53",
        )


def plot_rl_diagnostics() -> None:
    g8 = read_tsv(G8_LOG)[0]
    g32_rows = read_tsv(G32_LOG)
    g32 = g32_rows[-1]
    baseline = json.loads(BASELINE_JSON.read_text())

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor="#f7fafc")
    fig.suptitle(
        "Local T4/Tier-IV RL evidence — signal, weighting, and validation",
        fontsize=16,
        fontweight="bold",
        color="#102a43",
    )

    ax = axes[0, 0]
    labels = ["risk", "safety", "follow", "lane", "progress", "road border"]
    fields = [
        "valid_reward_risk_correlation",
        "valid_reward_safety_correlation",
        "valid_reward_follow_correlation",
        "valid_reward_lane_correlation",
        "valid_reward_progress_correlation",
        "valid_reward_road_border_correlation",
    ]
    vals = [float(g32.get(field, 0.0)) for field in fields]
    colors = ["#ef8354", "#d64550", "#4c956c", "#2f80ed", "#2d9cdb", "#829ab1"]
    y = np.arange(len(labels))
    bars = ax.barh(y, vals, color=colors, edgecolor="white")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.62)
    ax.set_xlabel("correlation with total reward")
    ax.set_title("What actually drives the reward?", loc="left", fontweight="bold")
    style_axes(ax)
    for bar, value in zip(bars, vals):
        ax.text(value + 0.012, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=9)
    ax.text(
        0.02,
        0.03,
        "Final row of the stable G=32 run; progress/lane dominate, risk is weak.",
        transform=ax.transAxes,
        fontsize=8,
        color="#52606d",
    )

    ax = axes[0, 1]
    names = ["G=8, β=.5", "G=32, β=1"]
    max_w = [float(g8["train_reward_weight_max"]), float(g32["train_reward_weight_max"])]
    ess = [float(g8["train_reward_weight_ess_fraction"]), float(g32["train_reward_weight_ess_fraction"])]
    top1 = [float(g8["train_reward_weight_top1_share"]), float(g32["train_reward_weight_top1_share"])]
    xx = np.arange(2)
    bars = ax.bar(xx, max_w, color=["#2d9cdb", "#d64550"], width=0.62)
    ax.set_yscale("log")
    ax.set_xticks(xx, names)
    ax.set_ylabel("maximum reward weight (log scale)")
    ax.set_title("G=32 concentrates the update", loc="left", fontweight="bold")
    style_axes(ax)
    for bar, value in zip(bars, max_w):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.12, f"{value:.2f}×", ha="center", fontsize=9)
    ax2 = ax.twinx()
    ax2.plot(xx, ess, "o-", color="#4c956c", linewidth=2, label="ESS fraction")
    ax2.plot(xx, top1, "s--", color="#f2c94c", linewidth=2, label="top-1 share")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("fraction")
    ax2.spines["top"].set_visible(False)
    ax2.legend(frameon=False, fontsize=8, loc="lower right")
    ax.text(
        0.02,
        0.03,
        "G=32: max≈64, ESS≈0.526; G=8: max≈3.06, ESS≈0.828.",
        transform=ax.transAxes,
        fontsize=8,
        color="#52606d",
    )

    ax = axes[1, 0]
    epochs = [0] + [int(row["epoch"]) for row in g32_rows]
    selection = [float(baseline["selection_score"])] + [float(row["valid_selection_score"]) for row in g32_rows]
    ax.plot(epochs, selection, "o-", color="#2f80ed", linewidth=2.4, label="selection score")
    ax.axhline(float(baseline["selection_score"]), color="#829ab1", linestyle="--", linewidth=1.5, label="SFT baseline")
    ax.set_xticks(epochs)
    ax.set_xlabel("RL epoch (0 = SFT source)")
    ax.set_ylabel("proxy reward selection score")
    ax.set_title("No accepted checkpoint improvement yet", loc="left", fontweight="bold")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8)
    for x, yv in zip(epochs, selection):
        ax.annotate(f"{yv:.4f}", (x, yv), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

    ax = axes[1, 1]
    metrics = ["EPDMS total", "DAC", "no collision"]
    baseline_vals = [
        float(baseline["valid_epdms_total"]),
        float(baseline["valid_epdms_dac"]),
        float(baseline["valid_epdms_no_collision"]),
    ]
    final_vals = [
        float(g32["valid_epdms_total"]),
        float(g32["valid_epdms_dac"]),
        float(g32["valid_epdms_no_collision"]),
    ]
    xx = np.arange(len(metrics))
    width = 0.36
    b1 = ax.bar(xx - width / 2, baseline_vals, width, label="SFT source", color="#829ab1")
    b2 = ax.bar(xx + width / 2, final_vals, width, label="G=32 epoch 2", color="#d64550")
    ax.set_xticks(xx, metrics)
    ax.set_ylim(0.82, 1.0)
    ax.set_ylabel("metric")
    ax.set_title("Held-out guard metrics move only at noise level", loc="left", fontweight="bold")
    style_axes(ax)
    add_value_labels(ax, b1)
    add_value_labels(ax, b2)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig(fig, "t4_rl_diagnostics")


def source_name(path: str) -> str:
    parts = Path(path).parts
    for candidate in ("xx1_psim", "x2_dev", "prd_jt"):
        if candidate in parts:
            return candidate
    return "other"


def plot_data_composition() -> None:
    base = json.loads(BASE_MANIFEST.read_text())
    valid = json.loads(VALID_MANIFEST.read_text())
    base_counts = Counter(source_name(p) for p in base)
    valid_counts = Counter(source_name(p) for p in valid)
    labels = ["xx1_psim", "x2_dev", "prd_jt"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), facecolor="#f7fafc")
    fig.suptitle("Local Tier-IV/T4 manifest composition", fontsize=16, fontweight="bold", color="#102a43")

    ax = axes[0]
    vals = [base_counts.get(k, 0) for k in labels]
    bars = ax.bar(labels, vals, color=["#2d9cdb", "#4c956c", "#f2c94c"])
    ax.set_title("100k uniform training sample", loc="left", fontweight="bold")
    ax.set_ylabel("NPZ clips")
    style_axes(ax)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + max(vals) * 0.015, f"{val:,}", ha="center", fontsize=9)
    ax.text(0.02, 0.03, "Counts read from artifacts/rl_train_real/base_uniform_100000.json", transform=ax.transAxes, fontsize=8, color="#52606d")

    ax = axes[1]
    vals = [valid_counts.get(k, 0) for k in labels]
    bars = ax.bar(labels, vals, color=["#2d9cdb", "#4c956c", "#f2c94c"])
    ax.set_title("4,096 validation sample", loc="left", fontweight="bold")
    ax.set_ylabel("NPZ clips")
    style_axes(ax)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + max(vals) * 0.015, f"{val:,}", ha="center", fontsize=9)
    ax.text(0.02, 0.03, "The full valid manifest contains 53,382 unique clips; this is its uniform audit sample.", transform=ax.transAxes, fontsize=8, color="#52606d")

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    savefig(fig, "t4_data_composition")


def plot_metric_gallery() -> None:
    """Make a beginner-oriented map from metric names to their meanings.

    Every panel is computed from the final stable G=32 validation row (plus the
    source baseline and the G=8/G=32 optimizer diagnostics).  The report links
    each metric card back to this gallery so a reader can see the quantity rather
    than having to decode a table of numbers.
    """
    row = read_tsv(G32_LOG)[-1]
    baseline = json.loads(BASELINE_JSON.read_text())

    def value(name: str, default: float = 0.0) -> float:
        raw = row.get(name, default)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return float(default)

    fig, axes = plt.subplots(3, 3, figsize=(16, 14), facecolor="#f7fafc")
    fig.suptitle(
        "Historical diagnostic metric map — not the current formal AWR reward",
        fontsize=17,
        fontweight="bold",
        color="#102a43",
    )

    def panel_title(ax: plt.Axes, title: str, subtitle: str) -> None:
        ax.set_title(title, loc="left", fontweight="bold", color="#102a43", pad=10)
        ax.text(0.0, 1.01, subtitle, transform=ax.transAxes, fontsize=8, color="#52606d", va="bottom")
        style_axes(ax)

    def bar_panel(ax: plt.Axes, labels: list[str], vals: list[float], colors: list[str], *, xmax: float = 1.0) -> None:
        y = np.arange(len(labels))
        bars = ax.barh(y, vals, color=colors, edgecolor="white")
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlim(0.0, xmax)
        for bar, val in zip(bars, vals):
            ax.text(min(float(val) + 0.015, xmax - 0.02), bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=8, color="#243b53")

    ax = axes[0, 0]
    panel_title(ax, "A · Collision and safety gate", "score: higher is better; rates: lower is better")
    labels = ["collision safety", "behavior gate", "active collision rate", "rear collision rate"]
    vals = [value("valid_reward_collision_safety"), value("valid_reward_behavior_gate"), value("valid_reward_collision_active"), value("valid_reward_collision_rear")]
    bar_panel(ax, labels, vals, ["#4c956c", "#2d9cdb", "#d64550", "#ef8354"])
    ax.set_xlabel("normalized score / fraction")
    ax.text(0.02, 0.02, "A gate is not the same as a collision count: the gate summarizes whether safety dominates behavior.", transform=ax.transAxes, fontsize=7.5, color="#52606d")

    ax = axes[0, 1]
    panel_title(ax, "B · TTC, THW, and risk", "TTC/THW are time margins; risk is the aggregate risk score")
    labels = ["TTC score", "THW score", "risk score", "TTC zero fraction", "THW zero fraction", "risk zero fraction"]
    vals = [value("valid_reward_ttc"), value("valid_reward_thw"), value("valid_reward_risk"), value("valid_reward_ttc_zero_fraction"), value("valid_reward_thw_zero_fraction"), value("valid_reward_risk_zero_fraction")]
    bar_panel(ax, labels, vals, ["#2d9cdb", "#4c956c", "#ef8354", "#d64550", "#f2c94c", "#829ab1"])
    ax.set_xlabel("score / fraction")
    ax.text(0.02, 0.02, "TTC asks ‘when would we collide if closing continues?’; THW asks ‘how many seconds of gap?’", transform=ax.transAxes, fontsize=7.5, color="#52606d")

    ax = axes[0, 2]
    panel_title(ax, "C · Occupancy source breakdown", "source fractions in the same validation audit")
    labels = ["stopped neighbor", "road-border fallback", "static object", "occupancy zero"]
    vals = [value("valid_reward_occupancy_stopped_source"), value("valid_reward_occupancy_road_border_source"), value("valid_reward_occupancy_static_source"), value("valid_reward_occupancy_zero_fraction")]
    bar_panel(ax, labels, vals, ["#4c956c", "#ef8354", "#829ab1", "#d64550"])
    ax.set_xlabel("fraction")
    ax.text(0.02, 0.02, "This is why border clearance must be reported separately from object occupancy.", transform=ax.transAxes, fontsize=7.5, color="#52606d")

    ax = axes[1, 0]
    panel_title(ax, "D · Red light and road border", "hard geometry constraints should not be hidden in a scalar reward")
    labels = ["red-light score", "road-border score", "red constraint active", "red violation", "border violation"]
    vals = [value("valid_reward_red_light"), value("valid_reward_road_border"), value("valid_reward_red_light_constraint_active_fraction"), value("valid_reward_red_light_violation_fraction"), value("valid_reward_road_border_violation_fraction")]
    bar_panel(ax, labels, vals, ["#4c956c", "#2d9cdb", "#f2c94c", "#d64550", "#ef8354"])
    ax.set_xlabel("score / fraction")
    ax.text(0.02, 0.02, "A 0.989 red-light score still needs its 1.06% violation fraction shown.", transform=ax.transAxes, fontsize=7.5, color="#52606d")

    ax = axes[1, 1]
    panel_title(ax, "E · Behavior quality", "higher is usually better, but progress must be read with safety and stop-state slices")
    labels = ["progress", "comfort", "lane", "stationary progress", "follow"]
    vals = [value("valid_reward_progress"), value("valid_reward_comfort"), value("valid_reward_lane"), value("valid_reward_stationary_progress"), value("valid_reward_follow")]
    bar_panel(ax, labels, vals, ["#2d9cdb", "#4c956c", "#829ab1", "#f2c94c", "#ef8354"])
    ax.set_xlabel("normalized score")
    ax.text(0.02, 0.02, "follow is conditional on a leader; stationary progress is a separate mode for stopped scenes.", transform=ax.transAxes, fontsize=7.5, color="#52606d")

    ax = axes[1, 2]
    panel_title(ax, "F · EPDMS / held-out planner score", "external-style guard metrics; higher is better")
    epdms = [
        ("total", value("valid_epdms_total")),
        ("ego progress", value("valid_epdms_ego_progress")),
        ("DAC", value("valid_epdms_dac")),
        ("no-fault collision", value("valid_epdms_no_at_fault_collision")),
        ("TTC", value("valid_epdms_ttc")),
        ("comfort", value("valid_epdms_comfort")),
    ]
    bar_panel(ax, [x[0] for x in epdms], [x[1] for x in epdms], ["#2d9cdb", "#4c956c", "#f2c94c", "#d64550", "#829ab1", "#ef8354"], xmax=1.05)
    ax.set_xlabel("EPDMS normalized score")
    ax.text(0.02, 0.02, "EPDMS is a promotion guard, not the same object as the training reward.", transform=ax.transAxes, fontsize=7.5, color="#52606d")

    ax = axes[2, 0]
    panel_title(ax, "G · Reward correlation: what is learnable?", "correlation with total reward; not a quality score")
    labels = ["progress", "lane", "follow", "risk", "safety"]
    vals = [value("valid_reward_progress_correlation"), value("valid_reward_lane_correlation"), value("valid_reward_follow_correlation"), value("valid_reward_risk_correlation"), value("valid_reward_safety_correlation")]
    bar_panel(ax, labels, vals, ["#2d9cdb", "#829ab1", "#4c956c", "#ef8354", "#d64550"], xmax=0.62)
    ax.set_xlabel("within-validation correlation")
    ax.text(0.02, 0.02, "The local total reward is much more rankable by progress/lane than by safety/risk.", transform=ax.transAxes, fontsize=7.5, color="#52606d")

    ax = axes[2, 1]
    panel_title(ax, "H · weight / ESS / update health", "G=8 is the conservative local reference; G=32 is the concentrated arm")
    g8 = read_tsv(G8_LOG)[0]
    names = ["G=8 max weight", "G=32 max weight", "G=8 ESS", "G=32 ESS", "G=8 top-1", "G=32 top-1"]
    vals = [float(g8["train_reward_weight_max"]), float(row["train_reward_weight_max"]), float(g8["train_reward_weight_ess_fraction"]), float(row["train_reward_weight_ess_fraction"]), float(g8["train_reward_weight_top1_share"]), float(row["train_reward_weight_top1_share"])]
    y = np.arange(len(names))
    colors = ["#2d9cdb", "#d64550", "#4c956c", "#ef8354", "#829ab1", "#f2c94c"]
    bars = ax.barh(y, vals, color=colors, edgecolor="white")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel("value (symlog; max weight is ×)")
    for bar, val in zip(bars, vals):
        ax.text(float(val) + (0.15 if val < 2 else 1.5), bar.get_y() + bar.get_height() / 2, f"{val:.3f}" if val < 10 else f"{val:.1f}", va="center", fontsize=8, color="#243b53")
    ax.text(0.02, 0.02, "ESS = effective sample size; lower ESS means fewer candidates carry the update.", transform=ax.transAxes, fontsize=7.5, color="#52606d")

    ax = axes[2, 2]
    panel_title(ax, "I · source baseline → final row", "small changes are why we need source-relative promotion guards")
    pairs = [("EPDMS total", "valid_epdms_total"), ("EPDMS DAC", "valid_epdms_dac"), ("collision safety", "valid_reward_collision_safety"), ("progress", "valid_reward_progress")]
    baseline_names = {
        "EPDMS total": "valid_epdms_total",
        "EPDMS DAC": "valid_epdms_dac",
        "collision safety": "valid_reward_collision_safety",
        "progress": "valid_reward_progress",
    }
    bvals = [float(baseline.get(baseline_names[label], 0.0)) for label, _ in pairs]
    fvals = [value(field) for _, field in pairs]
    xx = np.arange(len(pairs))
    width = 0.36
    b1 = ax.bar(xx - width / 2, bvals, width, label="SFT source", color="#829ab1")
    b2 = ax.bar(xx + width / 2, fvals, width, label="G=32 final", color="#d64550")
    ax.set_xticks(xx, [x[0] for x in pairs], rotation=18, ha="right")
    ax.set_ylim(0.75, 1.02)
    ax.set_ylabel("score")
    add_value_labels(ax, b1)
    add_value_labels(ax, b2)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.text(0.02, 0.02, "No claim of improvement is made unless the pre-registered guards pass.", transform=ax.transAxes, fontsize=7.5, color="#52606d")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    savefig(fig, "t4_metric_gallery")


def plot_red_light_audit() -> None:
    text = RED_AUDIT.read_text()
    removed_match = re.search(r"([\d,]+) samples removed", text)
    removed = int(removed_match.group(1).replace(",", "")) if removed_match else 2051
    # These values are the audited categories documented in the local report.
    categories = ["red→green before motion", "red at onset; no crossing", "green after motion", "ambiguous", "confirmed crossing"]
    counts = [333, 157, 16, 18, 0]
    fig, ax = plt.subplots(figsize=(11, 5.2), facecolor="#f7fafc")
    y = np.arange(len(categories))
    bars = ax.barh(y, counts, color=["#2d9cdb", "#4c956c", "#f2c94c", "#829ab1", "#d64550"])
    ax.set_yticks(y, categories)
    ax.invert_yaxis()
    ax.set_xlabel("clips in the 524-case audit")
    ax.set_title("Causal red-light audit: the old filter removed 2,051 clips, but found zero confirmed crossings", fontsize=14, fontweight="bold", color="#102a43", loc="left")
    style_axes(ax)
    for bar, val in zip(bars, counts):
        ax.text(max(val + 5, 5), bar.get_y() + bar.get_height() / 2, str(val), va="center", fontsize=9)
    ax.text(
        0.02,
        0.02,
        f"Local audit: {removed:,} old-detector candidates removed; audit sample=524. Use the causal v3 filter only after regeneration/hash verification.",
        transform=ax.transAxes,
        fontsize=9,
        color="#52606d",
    )
    fig.tight_layout()
    savefig(fig, "t4_red_light_audit")


def valid_points(points: np.ndarray, threshold: float = 1e-3) -> np.ndarray:
    points = np.asarray(points)
    return points[np.any(np.abs(points[..., :2]) > threshold, axis=-1)]


def draw_polyline(ax: plt.Axes, arr: np.ndarray, color: str, lw: float, alpha: float, boundaries: bool = False) -> None:
    if arr.ndim != 2 or arr.shape[1] < 2:
        return
    xy = arr[:, :2]
    mask = np.any(np.abs(xy) > 1e-3, axis=1)
    if mask.sum() < 2:
        return
    ax.plot(xy[mask, 0], xy[mask, 1], color=color, linewidth=lw, alpha=alpha, solid_capstyle="round")
    if boundaries and arr.shape[1] >= 8:
        for offset in (arr[:, 4:6], arr[:, 6:8]):
            bound = xy + offset
            ax.plot(bound[mask, 0], bound[mask, 1], color="#9fb3c8", linewidth=0.65, alpha=0.55)


def pick_dynamic_scene() -> tuple[Path, dict[str, np.ndarray], int]:
    paths = json.loads(BASE_MANIFEST.read_text())
    best: tuple[int, int, Path, dict[str, np.ndarray]] | None = None
    for index, raw_path in enumerate(paths[:1000]):
        path = Path(raw_path)
        if not path.exists():
            continue
        try:
            with np.load(path, allow_pickle=True) as archive:
                data = {key: archive[key] for key in archive.files}
            n = data.get("neighbor_agents_future", np.zeros((0, 1, 3)))
            active = int(np.sum(np.any(np.abs(n[..., :2]) > 1e-4, axis=(1, 2))))
            static = int(np.sum(np.any(np.abs(data.get("static_objects", np.zeros((0, 2)))[:, :2]) > 1e-4, axis=1)))
            score = active * 1000 + static
            if best is None or score > best[0]:
                best = (score, index, path, data)
        except Exception:
            continue
    if best is None:
        raise RuntimeError("No readable NPZ found in the local T4 training manifest")
    return best[2], best[3], best[1]


def draw_scene(ax: plt.Axes, data: dict[str, np.ndarray], frame: int | None = None) -> None:
    lanes = data.get("lanes", np.zeros((0, 1, 8)))
    route = data.get("route_lanes", np.zeros((0, 1, 8)))
    for lane in lanes:
        draw_polyline(ax, lane, "#aab7c4", 0.55, 0.45, boundaries=True)
    for lane in route:
        draw_polyline(ax, lane, "#2f80ed", 1.5, 0.48, boundaries=False)

    line_strings = data.get("line_strings", np.zeros((0, 1, 4)))
    for line in line_strings:
        if line.shape[1] < 4:
            continue
        pts = line[:, :2]
        mask = np.any(np.abs(pts) > 1e-3, axis=1)
        if mask.sum() < 2:
            continue
        if np.any(line[mask, 3] > 0.5):
            ax.plot(pts[mask, 0], pts[mask, 1], color="#d64550", linewidth=1.1, alpha=0.6)
        if np.any(line[mask, 2] > 0.5):
            ax.plot(pts[mask, 0], pts[mask, 1], color="#f2c94c", linewidth=1.8, alpha=0.85)

    polygons = data.get("polygons", np.zeros((0, 1, 3)))
    for polygon in polygons:
        xy = valid_points(polygon)
        if len(xy) >= 3:
            ax.plot(xy[:, 0], xy[:, 1], color="#d64550", linewidth=1.0, alpha=0.5)

    ego_future = data.get("ego_agent_future", np.zeros((0, 3)))
    if len(ego_future):
        ax.plot(ego_future[:, 0], ego_future[:, 1], color="#2d9cdb", linewidth=2.2, alpha=0.65, label="logged ego future")
    ego_past = data.get("ego_agent_past", np.zeros((0, 3)))
    if len(ego_past):
        ax.plot(ego_past[:, 0], ego_past[:, 1], color="#829ab1", linewidth=1.6, alpha=0.75, label="logged ego history")

    nf = data.get("neighbor_agents_future", np.zeros((0, 0, 3)))
    if nf.ndim == 3 and nf.shape[1] > 0:
        active = np.where(np.any(np.abs(nf[..., :2]) > 1e-4, axis=(1, 2)))[0]
        # Keep the plot legible: closest 36 agents at the initial frame.
        if len(active):
            distances = np.linalg.norm(nf[active, 0, :2], axis=1)
            active = active[np.argsort(distances)[:36]]
        for idx in active:
            ax.plot(nf[idx, :, 0], nf[idx, :, 1], color="#f2994a", linewidth=0.55, alpha=0.20)
            t = min(frame if frame is not None else 0, nf.shape[1] - 1)
            if np.any(np.abs(nf[idx, t, :2]) > 1e-4):
                ax.scatter(nf[idx, t, 0], nf[idx, t, 1], s=13, color="#e67300", alpha=0.85, zorder=8)

    if frame is None:
        ex, ey, eyaw = 0.0, 0.0, 0.0
    else:
        frame = min(frame, len(ego_future) - 1)
        ex, ey, eyaw = ego_future[frame, :3]
    shape = data.get("ego_shape", np.array([4.34, 1.7, 2.75], dtype=np.float32))
    length, width = float(shape[1] if len(shape) > 1 else shape[0]), float(shape[2] if len(shape) > 2 else shape[1])
    rect = Rectangle((ex - length / 2, ey - width / 2), length, width, angle=np.degrees(eyaw), facecolor="#2d9cdb", edgecolor="#102a43", alpha=0.85, zorder=12)
    ax.add_patch(rect)
    ax.scatter([ex], [ey], color="#102a43", s=17, zorder=13)


def scene_limits(data: dict[str, np.ndarray]) -> tuple[float, float, float, float]:
    clouds: list[np.ndarray] = []
    for key in ("lanes", "route_lanes", "line_strings", "polygons", "ego_agent_future", "ego_agent_past"):
        arr = data.get(key)
        if arr is None:
            continue
        xy = np.asarray(arr)[..., :2].reshape(-1, 2)
        xy = xy[np.any(np.abs(xy) > 1e-3, axis=1)]
        if len(xy):
            clouds.append(xy)
    nf = data.get("neighbor_agents_future")
    if nf is not None:
        xy = np.asarray(nf)[..., :2].reshape(-1, 2)
        xy = xy[np.any(np.abs(xy) > 1e-3, axis=1)]
        if len(xy):
            clouds.append(xy)
    xy = np.concatenate(clouds, axis=0)
    lo = np.percentile(xy, 2, axis=0)
    hi = np.percentile(xy, 98, axis=0)
    # Preserve a useful aspect ratio around the ego and avoid a giant map view.
    span = np.maximum(hi - lo, 40.0)
    center = (lo + hi) / 2
    span = max(span[0], span[1]) * 1.08
    return center[0] - span / 2, center[0] + span / 2, center[1] - span / 2, center[1] + span / 2


def render_t4_scene() -> dict[str, str | int]:
    path, data, manifest_index = pick_dynamic_scene()
    limits = scene_limits(data)
    fig, ax = plt.subplots(figsize=(11, 8), facecolor="#f7fafc")
    draw_scene(ax, data, frame=None)
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Logged Tier-IV/T4 scene used for RL training", fontsize=15, fontweight="bold", color="#102a43", loc="left")
    ax.set_xlabel("ego-frame x (m)")
    ax.set_ylabel("ego-frame y (m)")
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.01, 0.01, f"manifest index {manifest_index}; {path.name}; lanes=gray/blue, border=red, stop line=yellow, neighbors=orange", transform=ax.transAxes, fontsize=8, color="#52606d")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    savefig(fig, "t4_logged_scene_bev")

    # A compact future-motion GIF from the same NPZ.  It is a log replay
    # visualization, not a generated policy rollout.
    nf = data.get("neighbor_agents_future", np.zeros((0, 0, 3)))
    ego_future = data.get("ego_agent_future", np.zeros((0, 3)))
    frames = min(len(ego_future), 40)
    active = np.where(np.any(np.abs(nf[..., :2]) > 1e-4, axis=(1, 2)))[0] if nf.ndim == 3 else np.array([], dtype=int)
    if len(active):
        distances = np.linalg.norm(nf[active, 0, :2], axis=1)
        active = active[np.argsort(distances)[:36]]

    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#f7fafc")

    def update(frame: int):
        ax.clear()
        draw_scene(ax, data, frame=frame)
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"T4 logged dynamics — t={frame * 0.1:.1f}s", fontsize=13, fontweight="bold", color="#102a43")
        ax.set_xlabel("ego-frame x (m)")
        ax.set_ylabel("ego-frame y (m)")
        ax.text(0.01, 0.01, "logged future replay; orange agents are recorded neighbors", transform=ax.transAxes, fontsize=8, color="#52606d")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        return []

    animation = FuncAnimation(fig, update, frames=frames, interval=110, blit=False)
    animation.save(OUT / "t4_logged_dynamics.gif", writer=PillowWriter(fps=10))
    plt.close(fig)
    return {"npz": str(path), "manifest_index": manifest_index, "frames": frames, "active_neighbors_plotted": int(len(active))}


def main() -> None:
    plot_rl_diagnostics()
    plot_data_composition()
    plot_metric_gallery()
    plot_red_light_audit()
    scene_info = render_t4_scene()
    manifest = {
        "generated_from": {
            "g8_log": str(G8_LOG),
            "g32_log": str(G32_LOG),
            "baseline": str(BASELINE_JSON),
            "base_manifest": str(BASE_MANIFEST),
            "valid_manifest": str(VALID_MANIFEST),
            "red_light_audit": str(RED_AUDIT),
        },
        "scene": scene_info,
        "assets": sorted(p.name for p in OUT.iterdir() if p.is_file()),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
