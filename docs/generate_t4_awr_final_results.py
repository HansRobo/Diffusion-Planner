#!/usr/bin/env python3
"""Render paper-aligned checkpoint-level paired AWR diagnostics.

The primary promotion metric is the dataset mean of the per-scene PDM-like reward.  Hard-event
transitions and imitation error remain visible diagnostics but are not independent promotion
vetoes.  A 2,048-scene checkpoint sweep is labelled diagnostic; only a 46,068-scene report is
labelled full held-out evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FONT = ROOT / "docs" / "assets" / "fonts" / "NotoSansJP-VF.ttf"

COLORS = {
    "ink": "#102a43",
    "muted": "#60758a",
    "grid": "#d8e2ec",
    "blue": "#2878ff",
    "cyan": "#20b8d8",
    "green": "#20a44b",
    "amber": "#df9410",
    "red": "#dc3f4d",
    "violet": "#7656e8",
    "paper": "#f7f9fc",
}


def configure_plotting(font_path: Path) -> None:
    if font_path.exists():
        mpl.font_manager.fontManager.addfont(str(font_path))
        family = mpl.font_manager.FontProperties(fname=str(font_path)).get_name()
        mpl.rcParams["font.family"] = family
    mpl.rcParams.update(
        {
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "axes.titlesize": 14,
            "axes.titleweight": 800,
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.65,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def metric(epoch: dict[str, Any], name: str) -> dict[str, Any] | None:
    value = epoch.get("metrics", {}).get(name)
    return value if isinstance(value, dict) else None


def available_epochs(report: dict[str, Any]) -> list[int]:
    epochs = []
    for value in report.get("epochs", {}):
        try:
            epochs.append(int(value))
        except (TypeError, ValueError):
            continue
    return sorted(epochs)


def get_epoch(report: dict[str, Any], epoch: int) -> dict[str, Any]:
    return report["epochs"][str(epoch)]


def improvement_series(
    report: dict[str, Any], epochs: list[int], name: str, scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center, low, high = [], [], []
    for epoch in epochs:
        row = metric(get_epoch(report, epoch), name)
        if row is None:
            center.append(np.nan); low.append(np.nan); high.append(np.nan)
            continue
        ci = row.get("improvement_ci95", [np.nan, np.nan])
        center.append(float(row["improvement"]) * scale)
        low.append(float(ci[0]) * scale)
        high.append(float(ci[1]) * scale)
    return np.asarray(center), np.asarray(low), np.asarray(high)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", linestyle="--", linewidth=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.axhline(0.0, color=COLORS["ink"], linewidth=0.9, alpha=0.75)


def plot_ci_line(
    ax: plt.Axes,
    epochs: list[int],
    center: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    label: str,
    color: str,
    marker: str = "o",
) -> None:
    x = np.asarray(epochs)
    ax.plot(x, center, color=color, marker=marker, linewidth=2.2, markersize=5, label=label)
    ax.fill_between(x, low, high, color=color, alpha=0.13, linewidth=0)


def add_evidence_banner(fig: plt.Figure, scene_count: int, bootstrap: int) -> str:
    full = scene_count >= 46068
    if full:
        label = f"FULL HELD-OUT PAIRED EVIDENCE · N={scene_count:,} · bootstrap={bootstrap:,}"
        color = COLORS["green"]
    else:
        label = (
            f"CHECKPOINT-SWEEP DIAGNOSTIC · N={scene_count:,} · bootstrap={bootstrap:,} · "
            "NOT THE 46,068-SCENE PROMOTION RESULT"
        )
        color = COLORS["amber"]
    fig.text(
        0.055,
        0.965,
        label,
        fontsize=9.5,
        color=color,
        weight=850,
        ha="left",
        va="center",
    )
    return "full_held_out" if full else "checkpoint_diagnostic"


def render_checkpoint_sweep(report: dict[str, Any], output: Path) -> dict[str, Any]:
    epochs = available_epochs(report)
    if not epochs:
        raise RuntimeError("paired report contains no candidate epochs")
    latest = get_epoch(report, epochs[-1])
    scene_count = int(latest.get("scene_count", 0))
    bootstrap = int(latest.get("bootstrap_samples", 0))

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.2), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.09, hspace=0.34, wspace=0.24)
    fig.suptitle(
        "Original DP × HDP-style AWR — checkpoint paired evaluation",
        x=0.055,
        y=0.925,
        ha="left",
        fontsize=23,
        weight=900,
        color=COLORS["ink"],
    )
    evidence_class = add_evidence_banner(fig, scene_count, bootstrap)

    ax = axes[0, 0]
    for name, label, color, marker in [
        ("det_reward", "Dataset Score", COLORS["blue"], "o"),
        ("best_group_reward", "Best-of-K score", COLORS["green"], "s"),
        ("mean_group_reward", "K-mean score", COLORS["violet"], "^"),
    ]:
        if not any(metric(get_epoch(report, epoch), name) is not None for epoch in epochs):
            continue
        c, lo, hi = improvement_series(report, epochs, name, scale=100.0)
        plot_ci_line(ax, epochs, c, lo, hi, label, color, marker)
    ax.set_title("A · Primary mean scene-score gain", loc="left")
    ax.set_xlabel("replay epoch")
    ax.set_ylabel("paired gain [score points] ↑")
    ax.set_xticks(epochs)
    style_axis(ax)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="best")

    ax = axes[0, 1]
    for name, label, color, marker in [
        ("det_collision", "collision", COLORS["red"], "o"),
        ("det_rb_crossing", "road-border", COLORS["amber"], "s"),
        ("det_kinematic_violation", "kinematic", COLORS["violet"], "^"),
    ]:
        if not any(metric(get_epoch(report, epoch), name) is not None for epoch in epochs):
            continue
        c, lo, hi = improvement_series(report, epochs, name, scale=100.0)
        plot_ci_line(ax, epochs, c, lo, hi, label, color, marker)
    ax.set_title("B · Event-rate reduction vs source", loc="left")
    ax.set_xlabel("replay epoch")
    ax.set_ylabel("reduction [percentage points] ↑")
    ax.set_xticks(epochs)
    style_axis(ax)
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="best")
    ax.text(
        0.01,
        0.02,
        "scene-score multipliers; lane remains a separate diagnostic slice",
        transform=ax.transAxes,
        fontsize=8,
        color=COLORS["muted"],
    )

    ax = axes[1, 0]
    for name, label, color, marker in [
        ("ade", "ADE", COLORS["blue"], "o"),
        ("fde", "FDE", COLORS["red"], "s"),
    ]:
        if not any(metric(get_epoch(report, epoch), name) is not None for epoch in epochs):
            continue
        c, lo, hi = improvement_series(report, epochs, name)
        plot_ci_line(ax, epochs, c, lo, hi, label, color, marker)
    ax.set_title("C · ADE / FDE behavior-drift diagnostic", loc="left")
    ax.set_xlabel("replay epoch")
    ax.set_ylabel("error reduction [m] ↑")
    ax.set_xticks(epochs)
    style_axis(ax)
    ax.legend(frameon=False, fontsize=9, loc="best")

    ax = axes[1, 1]
    ade, _, _ = improvement_series(report, epochs, "ade")
    reward, _, _ = improvement_series(report, epochs, "det_reward", scale=100.0)
    rb, _, _ = improvement_series(report, epochs, "det_rb_crossing", scale=100.0)
    finite_rb = np.nan_to_num(rb, nan=0.0)
    sizes = 90 + 260 * (np.abs(finite_rb) / max(1e-8, np.max(np.abs(finite_rb))))
    scatter = ax.scatter(
        ade,
        reward,
        c=finite_rb,
        s=sizes,
        cmap="RdYlGn",
        vmin=-max(0.01, float(np.max(np.abs(finite_rb)))),
        vmax=max(0.01, float(np.max(np.abs(finite_rb)))),
        edgecolor="white",
        linewidth=1.3,
        zorder=3,
    )
    for x, y, epoch in zip(ade, reward, epochs):
        ax.annotate(f"e{epoch}", (x, y), xytext=(7, 7), textcoords="offset points", fontsize=9, weight=800)
    ax.set_title("D · Mean score vs behavior drift", loc="left")
    ax.set_xlabel("ADE reduction [m] → closer to logged expert")
    ax.set_ylabel("mean scene-score gain [points] ↑")
    style_axis(ax)
    cb = fig.colorbar(scatter, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("road-border reduction [pp] ↑", fontsize=8)
    ax.text(
        0.02,
        0.96,
        "checkpoint ranking uses mean score CI; ADE is diagnostic",
        transform=ax.transAxes,
        va="top",
        fontsize=8.5,
        color=COLORS["muted"],
    )

    fig.text(
        0.07,
        0.025,
        "Primary selection: 100 × mean paired Rdet gain, ranked by the paired-bootstrap 95% CI lower bound. "
        "Hard events and ADE/FDE explain the change; they do not independently veto promotion.",
        color=COLORS["muted"],
        fontsize=9,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)
    return {"evidence_class": evidence_class, "scene_count": scene_count, "bootstrap": bootstrap}


def render_hard_event_transitions(
    report: dict[str, Any], output: Path, scene_count: int, bootstrap: int
) -> None:
    """Show recovered and newly introduced paired hard-event scenes separately."""
    epochs = available_epochs(report)
    specs = [
        ("det_collision", "Collision OBB"),
        ("det_rb_crossing", "Road-border OBB"),
        ("det_kinematic_violation", "Kinematic gate"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), sharey=False)
    fig.subplots_adjust(left=0.055, right=0.98, top=0.75, bottom=0.18, wspace=0.25)
    fig.suptitle(
        "Paired hard-event transitions — diagnostic decomposition",
        x=0.055, y=0.90, ha="left", fontsize=21, weight=900, color=COLORS["ink"],
    )
    add_evidence_banner(fig, scene_count, bootstrap)
    x = np.arange(len(epochs), dtype=float)
    width = 0.34
    for ax, (name, label) in zip(axes, specs):
        recovered: list[int] = []
        introduced: list[int] = []
        source_counts: list[int] = []
        candidate_counts: list[int] = []
        for epoch in epochs:
            row = metric(get_epoch(report, epoch), name)
            if row is None:
                recovered.append(0); introduced.append(0); source_counts.append(0); candidate_counts.append(0)
                continue
            paired = int(row.get("paired_scene_count", get_epoch(report, epoch).get("scene_count", 0)))
            recovered.append(int(round(float(row.get("improved_scene_fraction", 0.0)) * paired)))
            introduced.append(int(round(float(row.get("regressed_scene_fraction", 0.0)) * paired)))
            source_counts.append(int(round(float(row["baseline_mean"]) * paired)))
            candidate_counts.append(int(round(float(row["candidate_mean"]) * paired)))
        recovered_array = np.asarray(recovered)
        introduced_array = np.asarray(introduced)
        ax.bar(x - width / 2, recovered_array, width=width, color=COLORS["green"], label="recovered")
        ax.bar(x + width / 2, -introduced_array, width=width, color=COLORS["red"], label="new regression")
        for xpos, value in zip(x - width / 2, recovered_array):
            ax.text(xpos, value, str(int(value)), ha="center", va="bottom", fontsize=8, color=COLORS["green"], weight=800)
        for xpos, value in zip(x + width / 2, introduced_array):
            ax.text(xpos, -value, str(int(value)), ha="center", va="top", fontsize=8, color=COLORS["red"], weight=800)
        ax.axhline(0, color=COLORS["ink"], linewidth=1.0)
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.55)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks(x, [f"e{epoch}" for epoch in epochs])
        ax.set_title(label, loc="left", fontsize=13, weight=850)
        ax.set_ylabel("paired scene count\n↑ recovered / ↓ newly introduced", fontsize=9)
        net = [source - candidate for source, candidate in zip(source_counts, candidate_counts)]
        ax.text(
            0.02, 0.98,
            "net source−AWR: " + ", ".join(f"e{epoch} {value:+d}" for epoch, value in zip(epochs, net)),
            transform=ax.transAxes, va="top", fontsize=7.8, color=COLORS["muted"],
        )
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.text(
        0.055, 0.055,
        "Recovered and newly introduced events are both disclosed. The primary promotion decision remains the full-dataset mean scene score.",
        fontsize=9.5, color=COLORS["muted"],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def fmt_value(name: str, value: float) -> str:
    if name in {"det_collision", "det_rb_crossing", "det_kinematic_violation", "det_lane_crossing"}:
        return f"{100 * value:.3f}%"
    if name == "safe_candidate_count":
        return f"{value:.3f}/10"
    if name in {"ade", "fde"}:
        return f"{value:.3f} m"
    if name in {"det_reward", "best_group_reward", "mean_group_reward"}:
        return f"{100 * value:.3f}"
    return f"{value:.5f}"


def fmt_improvement(name: str, value: float) -> str:
    if name in {"det_collision", "det_rb_crossing", "det_kinematic_violation", "det_lane_crossing"}:
        return f"{100 * value:+.3f} pp"
    if name == "safe_candidate_count":
        return f"{value:+.3f}/10"
    if name in {"ade", "fde"}:
        return f"{value:+.3f} m"
    if name in {"det_reward", "best_group_reward", "mean_group_reward"}:
        return f"{100 * value:+.3f} pt"
    return f"{value:+.5f}"


def render_scorecard(
    report: dict[str, Any], epoch: int, output: Path, evidence_class: str
) -> None:
    data = get_epoch(report, epoch)
    rows = [
        ("Primary", "Dataset Score", "det_reward"),
        ("Sampling diagnostic", "Best-of-K score", "best_group_reward"),
        ("Coverage", "safe candidates", "safe_candidate_count"),
        ("Hard safety", "collision", "det_collision"),
        ("Hard safety", "road-border", "det_rb_crossing"),
        ("Hard safety", "kinematic", "det_kinematic_violation"),
        ("Diagnostic", "lane crossing", "det_lane_crossing"),
        ("Behavior diagnostic", "ADE", "ade"),
        ("Behavior diagnostic", "FDE", "fde"),
        ("Behavior diagnostic", "smoothness", "det_smoothness"),
    ]
    rows = [row for row in rows if metric(data, row[2]) is not None]

    fig, ax = plt.subplots(figsize=(15.5, 8.8))
    ax.set_xlim(0, 1); ax.set_ylim(0, len(rows) + 2.1); ax.axis("off")
    fig.subplots_adjust(left=0.04, right=0.98, top=0.91, bottom=0.08)
    fig.text(0.055, 0.94, f"Epoch {epoch} paired scorecard", fontsize=24, weight=900, color=COLORS["ink"])
    status = "FULL 46,068-SCENE EVIDENCE" if evidence_class == "full_held_out" else "2,048-SCENE DIAGNOSTIC — NOT PROMOTION EVIDENCE"
    fig.text(0.055, 0.905, status, fontsize=10, weight=850, color=COLORS["green"] if evidence_class == "full_held_out" else COLORS["amber"])

    columns = [0.04, 0.20, 0.42, 0.57, 0.72, 0.88]
    headers = ["class", "metric", "source", "checkpoint", "paired Δ", "95% CI"]
    for x, header in zip(columns, headers):
        ax.text(x, len(rows) + 1.25, header, fontsize=10, weight=850, color=COLORS["muted"], transform=ax.transData)
    ax.plot([0.035, 0.97], [len(rows) + 0.95] * 2, color=COLORS["grid"], linewidth=1.2)

    class_colors = {"Primary": COLORS["blue"], "Sampling diagnostic": COLORS["green"], "Coverage": COLORS["green"], "Hard safety": COLORS["red"], "Diagnostic": COLORS["cyan"], "Behavior diagnostic": COLORS["violet"]}
    for index, (group, label, name) in enumerate(rows):
        row = metric(data, name)
        assert row is not None
        y = len(rows) - index + 0.35
        if index % 2 == 0:
            ax.add_patch(FancyBboxPatch((0.03, y - 0.34), 0.94, 0.72, boxstyle="round,pad=0.005,rounding_size=0.015", linewidth=0, facecolor="#f6f9fc"))
        ci = row["improvement_ci95"]
        improved = ci[0] > 0
        regressed = ci[1] < 0
        delta_color = COLORS["green"] if improved else COLORS["red"] if regressed else COLORS["amber"]
        values = [
            group,
            label,
            fmt_value(name, float(row["baseline_mean"])),
            fmt_value(name, float(row["candidate_mean"])),
            fmt_improvement(name, float(row["improvement"])),
            f"[{fmt_improvement(name, float(ci[0]))}, {fmt_improvement(name, float(ci[1]))}]",
        ]
        for col, (x, value) in enumerate(zip(columns, values)):
            color = class_colors[group] if col == 0 else delta_color if col >= 4 else COLORS["ink"]
            weight = 850 if col in {0, 1, 4} else 500
            ax.text(x, y, value, fontsize=10.5, color=color, weight=weight, va="center")

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["green"], label="95% CI > 0"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["amber"], label="CI crosses 0"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["red"], label="95% CI < 0"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.03, -0.02), ncol=3, frameon=False, fontsize=9)
    fig.text(0.64, 0.055, "Δ is direction-normalized: positive always means better.", fontsize=9, color=COLORS["muted"])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paired_report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selected-epoch", type=int)
    parser.add_argument("--font", type=Path, default=DEFAULT_FONT)
    args = parser.parse_args()

    configure_plotting(args.font)
    report = load_json(args.paired_report)
    epochs = available_epochs(report)
    if not epochs:
        raise RuntimeError(f"no epochs in {args.paired_report}")
    selected = args.selected_epoch if args.selected_epoch is not None else epochs[-1]
    if selected not in epochs:
        raise RuntimeError(f"epoch {selected} is not present; available={epochs}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = args.output_dir / "checkpoint_paired_sweep.png"
    evidence = render_checkpoint_sweep(report, sweep_path)
    transition_path = args.output_dir / "hard_event_transition_audit.png"
    render_hard_event_transitions(
        report, transition_path, evidence["scene_count"], evidence["bootstrap"]
    )
    scorecard_path = args.output_dir / f"epoch_{selected:03d}_paired_scorecard.png"
    render_scorecard(report, selected, scorecard_path, evidence["evidence_class"])
    manifest = {
        "source_report": str(args.paired_report.resolve()),
        "available_epochs": epochs,
        "selected_epoch": selected,
        **evidence,
        "artifacts": [
            sweep_path.name,
            sweep_path.with_suffix(".svg").name,
            transition_path.name,
            transition_path.with_suffix(".svg").name,
            scorecard_path.name,
            scorecard_path.with_suffix(".svg").name,
        ],
        "contract": "promotion uses full-dataset mean per-scene reward gain with paired CI; component metrics are diagnostics",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.output_dir)


if __name__ == "__main__":
    main()
