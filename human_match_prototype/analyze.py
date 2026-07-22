"""Compare the three mismatch signals and render top-mismatch overlays.

Usage:
  .venv/bin/python -m human_match_prototype.analyze \
    --or_csv data/human_match/scores_or.csv \
    --normal_csv data/human_match/scores_normal.csv \
    --output_dir data/human_match/analysis [--top_k 20] [--plots 12]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from human_match_prototype.sampler import TrajectorySampler

SIGNALS = ["min_ade_4s", "maha_pct", "latent_knn_mean"]
MODEL_DIR = Path("/opt/autoware/mlmodels/diffusion_planner_for_x2")


def _md_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-markdown table (no tabulate dependency)."""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.reset_index()

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = [
        "| " + " | ".join(fmt(v) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep, *rows])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--or_csv", required=True)
    p.add_argument("--normal_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--plots", type=int, default=12)
    p.add_argument("--num_samples", type=int, default=64)
    return p.parse_args()


def bev_overlay(sampler, npz_path, out_png, num_samples):
    from src.visualization import PAST_FRAMES, precompute_static, render_frame

    data = dict(np.load(npz_path, allow_pickle=True))
    r = sampler.sample(str(npz_path), num_samples=num_samples, seed=0)

    fig, ax = plt.subplots(figsize=(10, 10))
    static = precompute_static(data)
    render_frame(fig, ax, data, static, t=PAST_FRAMES - 1, filename=Path(npz_path).name)

    for s in r.ego_samples:
        ax.plot(s[:, 0], s[:, 1], color="#2196F3", alpha=0.25, lw=1, zorder=35)
    ax.plot(
        r.human_future[:, 0],
        r.human_future[:, 1],
        color="tab:red",
        lw=2.5,
        zorder=36,
        label="human",
    )

    fig.savefig(out_png, dpi=120, facecolor="#1A1A1A", bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    orr = pd.read_csv(args.or_csv)
    nor = pd.read_csv(args.normal_csv)
    lines = ["# Human-match prototype analysis\n"]

    corr = orr[SIGNALS].corr(method="spearman")
    corr.to_csv(out / "signal_correlation.csv")
    lines += ["## Spearman correlation (risk scenes)\n", _md_table(corr), ""]

    lines.append("\n## Risk vs normal separation (median [p90])\n")
    for sig in SIGNALS + ["frac_close_4s", "mismatch_4s", "spread_4s"]:
        lines.append(
            f"- `{sig}`: risk {orr[sig].median():.3f} [{orr[sig].quantile(0.9):.3f}]"
            f" vs normal {nor[sig].median():.3f} [{nor[sig].quantile(0.9):.3f}]"
        )
    lines.append("\n## Mismatch rate by risk category (frac_close_4s == 0)\n")
    lines.append(_md_table(orr.groupby("category")["mismatch_4s"].mean()))

    tops = {}
    for sig in SIGNALS:
        top = orr.nlargest(args.top_k, sig)[["npz_path", sig]]
        top.to_csv(out / f"top_{sig}.csv", index=False)
        tops[sig] = set(top.npz_path)
    lines.append("\n## Top-K overlap between signals\n")
    names = list(tops)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n = len(tops[names[i]] & tops[names[j]])
            lines.append(f"- {names[i]} ∩ {names[j]}: {n}/{args.top_k}")

    sampler = TrajectorySampler(
        str(MODEL_DIR / "args.json"), str(MODEL_DIR / "diffusion_planner.onnx")
    )
    plot_dir = out / "overlays"
    plot_dir.mkdir(exist_ok=True)
    worst = orr.sort_values("min_ade_4s", ascending=False).head(args.plots)
    for _, row in worst.iterrows():
        name = Path(row.npz_path).stem
        bev_overlay(sampler, row.npz_path, plot_dir / f"mismatch_{name}.png", args.num_samples)
    for _, row in nor.sample(4, random_state=0).iterrows():
        name = Path(row.npz_path).stem
        bev_overlay(sampler, row.npz_path, plot_dir / f"normal_{name}.png", args.num_samples)
    lines.append(f"\nOverlay PNGs: {plot_dir}")

    (out / "summary.md").write_text("\n".join(lines))
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
