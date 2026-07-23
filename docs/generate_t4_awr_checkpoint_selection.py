#!/usr/bin/env python3
"""Render the epoch-2--10 AWR pilot checkpoint-selection evidence.

The chart distinguishes optimization loss from deployment score and is built
only from finalized saved-EMA screen artifacts.  It deliberately avoids CJK
text so the exported PNG remains portable in Confluence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RUN = Path(
    "/mnt/nvme/wangbin/Diffusion-Planner-t4-main/outputs/"
    "awr_t4_full_sequence_filtered/"
    "20260717-004516_full_sequence_filtered_hdp_plain_mse_replay_resume_single_checkpoint"
)
DEFAULT_OUT = Path("docs/t4_conference_assets/checkpoint_selection")


def _load(path: Path):
    return json.loads(path.read_text())


def _train_losses(path: Path) -> dict[int, float]:
    values: dict[int, float] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("phase") == "train" and row.get("mean_loss") is not None:
            values[int(row["epoch"])] = float(row["mean_loss"])
    return values


def _render(run_dir: Path, output_dir: Path) -> dict:
    screen_path = run_dir / "deployment_screen_dp10/checkpoint_finalists.json"
    screen = _load(screen_path)
    rows = sorted(screen["all_epochs"], key=lambda row: int(row["epoch"]))
    selected_epoch = int(screen["promotion_recommendation"]["epoch"])
    losses = _train_losses(run_dir / "metrics.jsonl")

    epochs = np.asarray([int(row["epoch"]) for row in rows])
    scores = np.asarray([float(row["candidate_dataset_score"]) for row in rows])
    baseline = float(rows[0]["baseline_dataset_score"])
    ci_low = np.asarray([baseline + float(row["score_gain_ci95_points"][0]) for row in rows])
    ci_high = np.asarray([baseline + float(row["score_gain_ci95_points"][1]) for row in rows])
    yerr = np.vstack([scores - ci_low, ci_high - scores])

    colors = [
        "#047857" if epoch == selected_epoch else "#2563eb" if epoch == 10 else "#64748b"
        for epoch in epochs
    ]
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    grid = fig.add_gridspec(
        2, 2, height_ratios=[1.28, 1.0], width_ratios=[1.12, 1.0],
        hspace=0.38, wspace=0.18,
    )
    ax_score = fig.add_subplot(grid[0, :])
    ax_loss = fig.add_subplot(grid[1, 0])
    ax_table = fig.add_subplot(grid[1, 1])

    ax_score.axhline(
        baseline, color="#b91c1c", lw=1.5, ls="--",
        label=f"Original-DP baseline = {baseline:.3f}", zorder=1,
    )
    ax_score.errorbar(
        epochs, scores, yerr=yerr, fmt="none", ecolor="#94a3b8",
        elinewidth=1.5, capsize=5, capthick=1.5, zorder=2,
    )
    ax_score.plot(epochs, scores, color="#475569", lw=1.5, alpha=0.7, zorder=2)
    ax_score.scatter(epochs, scores, s=105, c=colors, edgecolors="white", linewidths=1.2, zorder=3)
    for epoch, score, color in zip(epochs, scores, colors):
        if epoch in {2, selected_epoch, 10}:
            tag = "selected" if epoch == selected_epoch else "10-epoch pilot end" if epoch == 10 else "CI crosses baseline"
            ax_score.annotate(
                f"epoch {epoch}: {score:.3f}\n{tag}",
                xy=(epoch, score), xytext=(0, 18 if epoch != 10 else -42),
                textcoords="offset points", ha="center", fontsize=9,
                fontweight="bold" if epoch in {selected_epoch, 10} else "normal",
                color=color,
                bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": color, "alpha": 0.96},
                arrowprops={"arrowstyle": "-", "color": color, "lw": 1.0},
            )
    ax_score.set_xticks(epochs)
    ax_score.set_ylabel("dataset score = 100 x mean formal reward")
    ax_score.set_title(
        "1 · Saved-EMA deployment screen: epoch 7 peaks; epoch 10 remains above baseline",
        loc="left", fontsize=13, fontweight="bold",
    )
    ax_score.grid(axis="y", alpha=0.18)
    ax_score.legend(loc="lower right", fontsize=9, framealpha=0.95)

    loss_epochs = np.asarray(sorted(losses))
    loss_values = np.asarray([losses[int(epoch)] for epoch in loss_epochs])
    ax_loss.plot(loss_epochs, loss_values, color="#7c3aed", marker="o", lw=2.2, ms=5)
    ax_loss.scatter([selected_epoch, 10], [losses[selected_epoch], losses[10]], s=85,
                    c=["#047857", "#2563eb"], edgecolors="white", linewidths=1.0, zorder=4)
    ax_loss.annotate(
        f"epoch 10\n{losses[10]:.6f}", xy=(10, losses[10]), xytext=(-6, 26),
        textcoords="offset points", ha="center", color="#2563eb", fontsize=9, fontweight="bold",
    )
    ax_loss.set_xticks(loss_epochs)
    ax_loss.set_xlabel("replay epoch")
    ax_loss.set_ylabel("mean weighted diffusion loss")
    ax_loss.set_title("2 · Replay loss keeps falling", loc="left", fontsize=12, fontweight="bold")
    ax_loss.grid(alpha=0.18)
    ax_loss.text(
        0.03, 0.08, "Optimization only: lower loss != better deployment score.",
        transform=ax_loss.transAxes, fontsize=8.8, color="#7c2d12", fontweight="bold",
        bbox={"boxstyle": "round,pad=0.3", "fc": "#fff7ed", "ec": "#fdba74"},
    )

    by_epoch = {int(row["epoch"]): row for row in rows}
    table_epochs = [selected_epoch, 10]
    table_rows = []
    for label, getter, fmt in [
        ("Screen score ↑", lambda row: row["candidate_dataset_score"], "{:.3f}"),
        ("Gain vs baseline", lambda row: row["score_gain_points"], "+{:.3f} pt"),
        ("95% CI lower", lambda row: row["score_gain_ci95_points"][0], "+{:.3f} pt"),
        ("Road-border rate ↓", lambda row: 100.0 * row["diagnostics"]["det_rb_crossing"]["candidate_mean"], "{:.3f}%"),
        ("ADE ↓", lambda row: row["diagnostics"]["ade"]["candidate_mean"], "{:.3f} m"),
        ("FDE ↓", lambda row: row["diagnostics"]["fde"]["candidate_mean"], "{:.3f} m"),
    ]:
        table_rows.append(
            [label, *[fmt.format(float(getter(by_epoch[epoch]))) for epoch in table_epochs]]
        )
    table_rows.append(["Decision", "SELECT", "DO NOT SELECT"])
    ax_table.axis("off")
    table = ax_table.table(
        cellText=table_rows,
        colLabels=["Metric", *[f"Epoch {epoch}" for epoch in table_epochs]],
        colWidths=[0.42, 0.29, 0.29],
        cellLoc="center", bbox=[0.0, 0.0, 1.0, 0.92],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.2)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d6e0ea")
        if row == 0:
            cell.set_facecolor("#eaf1f8")
            cell.set_text_props(weight="bold", color="#19324a")
        elif col == 0:
            cell.set_facecolor("#f8fafc")
            cell.set_text_props(weight="bold", color="#334e68")
        elif cell.get_text().get_text() == "SELECT":
            cell.set_facecolor("#dcfce7")
            cell.set_text_props(weight="bold", color="#166534")
        elif cell.get_text().get_text() == "DO NOT SELECT":
            cell.set_facecolor("#fee2e2")
            cell.set_text_props(weight="bold", color="#991b1b")
    ax_table.set_title("3 · Why the final checkpoint stays epoch 7", loc="left", fontsize=12, fontweight="bold")

    full = {}
    for epoch in (5, 6, 7):
        path = run_dir / f"full_validation_dp10/epoch_{epoch:03d}/summary.json"
        if path.is_file():
            full[epoch] = float(_load(path)["mean_det_reward"])

    fig.suptitle(
        "FULL-DATA AWR PILOT · 10 scheduled epochs = 1 mining pass + 9 replay-training passes",
        fontsize=17, fontweight="bold", y=0.985,
    )
    fig.text(
        0.5, 0.945,
        "8 x H100 · about 13.3 h from mining start to epoch-10 checkpoint · early evidence, not a known optimal RL length",
        ha="center", fontsize=10, color="#475569",
    )
    full_text = ", ".join(f"e{epoch}={value:.6f}" for epoch, value in sorted(full.items()))
    fig.text(
        0.5, 0.012,
        f"Finalists advanced to 46,068-scene validation: {full_text}. Epoch 7 remains highest; required epoch count must be measured. raw[t+1] -> aligned[t].",
        ha="center", fontsize=9.2, color="#475569",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "epoch_2_10_checkpoint_selection.png"
    svg = output_dir / "epoch_2_10_checkpoint_selection.svg"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "run_dir": str(run_dir),
        "screen": str(screen_path),
        "screen_scene_count": int(screen["expected_scene_count"]),
        "completed_through_epoch": int(max(epochs)),
        "selected_epoch": selected_epoch,
        "baseline_score": baseline,
        "epoch_10": by_epoch[10],
        "epoch_selected": by_epoch[selected_epoch],
        "train_loss_epoch_10": losses[10],
        "full_validation_mean_reward": full,
        "assets": {"png": str(png), "svg": str(svg)},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(_render(args.run_dir.resolve(), args.output_dir.resolve()), indent=2))


if __name__ == "__main__":
    main()
