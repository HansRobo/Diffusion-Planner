#!/usr/bin/env python3
"""Render the exact T4 AWR data path and measured ten-epoch-cycle cost."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
T4 = ROOT.parent / "Diffusion-Planner-t4-main"
RUN = T4 / (
    "outputs/awr_t4_full_sequence_filtered/"
    "20260717-004516_full_sequence_filtered_hdp_plain_mse_replay_resume_single_checkpoint"
)
CACHE = T4 / (
    "outputs/awr_t4_full_sequence_filtered/"
    "20260716-184121_full_sequence_filtered_hdp_plain_mse_loader2_single_checkpoint/replay_buffer"
)
FILTER = T4 / "outputs/data/full_sequence_is_skipped_filter_manifest.json"
OUT = ROOT / "docs/t4_conference_assets/data_runtime/full_data_replay_and_runtime.png"


def _box(axis, x, y, w, h, title, body, color):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=1.5, edgecolor=color, facecolor="white",
    )
    axis.add_patch(patch)
    axis.text(x + 0.025, y + h - 0.055, title, fontsize=11.5, fontweight="bold", color=color, va="top")
    axis.text(x + 0.025, y + h - 0.125, body, fontsize=9.4, color="#334e68", va="top", linespacing=1.35)


def main() -> None:
    filt = json.loads(FILTER.read_text())
    replay = json.loads((RUN / "replay_resume.json").read_text())
    summaries = [json.loads(path.read_text()) for path in sorted(RUN.glob("train_epoch_*/summary.json"))]
    train_hours = sum(float(row["elapsed_sec"]) for row in summaries) / 3600.0
    average_minutes = 60.0 * train_hours / len(summaries)
    mining_hours = 6.0914
    measured_hours = 13.3017

    fig = plt.figure(figsize=(16, 9), facecolor="#f5f7fb")
    axis = fig.add_axes([0.035, 0.08, 0.93, 0.80])
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    fig.suptitle(
        "HDP-STYLE AWR · rollout–replay decoupling on the T4 full dataset",
        x=0.04, y=0.965, ha="left", fontsize=21, fontweight="bold", color="#102a43",
    )
    fig.text(
        0.04, 0.91,
        "The RL-specific speedup is to run expensive K-sampling and reward geometry once, then reuse fixed pseudo-targets for several regression epochs.",
        fontsize=10.5, color="#52677d",
    )

    boxes = [
        (0.01, "1 · Raw train list", f"{filt['source_train_count']:,} scenes\nfull-sequence corpus", "#475569"),
        (0.21, "2 · is_skipped filter", f"drop {filt['dropped_train_count']:,} ({100*filt['dropped_train_fraction']:.1f}%)\nkeep {filt['filtered_train_count']:,}", "#b45309"),
        (0.41, "3 · Mine every clean scene", "K=10 stochastic DP samples\nscore + AWR weight + encoding", "#0369a1"),
        (0.61, "4 · Freeze RL pseudo-targets", f"{replay['total_scene_groups']:,} groups used\ntrajectory + reward + weight\n+ policy context", "#7c3aed"),
        (0.81, "5 · Replay epochs 2–10", "random with replacement\n9 gradient passes; no re-score", "#047857"),
    ]
    for x, title, body, color in boxes:
        _box(axis, x, 0.62, 0.17, 0.23, title, body, color)
    for left in (0.18, 0.38, 0.58, 0.78):
        axis.add_patch(FancyArrowPatch((left, 0.735), (left + 0.025, 0.735), arrowstyle="-|>", mutation_scale=15, color="#94a3b8", lw=1.5))

    _box(
        axis, 0.01, 0.25, 0.45, 0.25, "Data-selection meaning",
        "Easy/normal scenes are already present in the full replay.\n"
        "A tied finite reward group gets w=1 for all K candidates, acting as\n"
        "implicit old-policy self-distillation. bc_weight=0 and expert anchor=0,\n"
        "so this is not explicit human-GT rehearsal.", "#0f766e",
    )
    _box(
        axis, 0.50, 0.25, 0.49, 0.25, "Why replay is cheaper — measured budget",
        f"Mining / replay build: {mining_hours:.1f} h\n"
        f"9 replay-training passes: {train_hours:.1f} h ({average_minutes:.1f} min/pass)\n"
        f"Mining start → epoch-10 checkpoint: {measured_hours:.1f} h\n"
        "No K=10 rollout, reward geometry or encoder rerun in replay epochs.\n"
        "Approx. 10-epoch cycle: ~13 h; 20/50/100 epochs ≈ 27/67/133 h.\n"
        "Periodic refresh is required as the policy changes; epoch count is eval-selected.", "#1d4ed8",
    )

    axis.text(0.01, 0.16, "NEXT ABLATION — not part of the completed run", fontsize=11.5, fontweight="bold", color="#9a3412")
    axis.text(
        0.01, 0.105,
        "Oversample low-reward / high-spread / hard-event scenes for efficiency, while retaining a controlled normal-scene or explicit BC anchor. "
        "Compare against another faithful full-distribution refresh before adoption.",
        fontsize=9.8, color="#7c2d12",
        bbox={"boxstyle": "round,pad=0.55", "fc": "#fff7ed", "ec": "#fdba74"},
    )
    fig.text(
        0.04, 0.025,
        "Evidence boundary: counts come from the is_skipped and replay manifests; timing comes from epoch summaries and filesystem timestamps. "
        "The 20/50/100 estimates assume one full mining refresh per ten scheduled epochs.",
        fontsize=8.8, color="#52677d",
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
