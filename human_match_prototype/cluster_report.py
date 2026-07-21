"""Per-cluster human-match report: aggregate scores, render local self-contained HTML.

Usage:
  .venv/bin/python -m human_match_prototype.cluster_report \
    --cluster_csv data/human_match/cluster_takanawa/scores_cluster.csv \
    --or_csv data/human_match/scores_or.csv \
    --normal_csv data/human_match/scores_normal.csv \
    --output_dir data/human_match/cluster_takanawa
"""

import argparse
import base64
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

MODEL_DIR = Path("/opt/autoware/mlmodels/diffusion_planner_for_x2")

# Colors from the validated dataviz reference palette (light surface #fcfcfb).
# Single data series -> one categorical hue (slot 1, blue); no legend box, the
# title names it. Baselines are reference lines carrying risk/normal semantics
# -> status palette, never reused as a series color.
_SURFACE = "#fcfcfb"
_BAR = "#2a78d6"  # categorical slot 1 (blue)
_INK_PRIMARY = "#0b0b0b"
_INK_MUTED = "#898781"  # axis / tick labels
_GRID = "#e1e0d9"  # hairline gridline
_BASELINE_COLORS = {  # status hues, distinct from the series blue
    "RISK baseline": "#d03b3b",  # critical
    "NORMAL baseline": "#0ca30c",  # good
}


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cat, g in df.groupby("category"):
        rows.append(
            {
                "cluster": cat,
                "n": len(g),
                "mismatch_rate_4s": g["mismatch_4s"].mean(),
                "median_min_ade_4s": g["min_ade_4s"].median(),
                "p90_min_ade_4s": g["min_ade_4s"].quantile(0.9),
                "median_frac_close_4s": g["frac_close_4s"].median(),
                "median_lon_err_4s": g["best_lon_err_4s"].median(),
                "median_lat_err_4s": g["best_lat_err_4s"].median(),
                "median_spread_4s": g["spread_4s"].median(),
                "median_latent_knn": g["latent_knn_mean"].median(),
            }
        )
    out = pd.DataFrame(rows).sort_values("mismatch_rate_4s", ascending=False)
    return out.reset_index(drop=True)


def baseline_rows(or_csv: str, normal_csv: str) -> pd.DataFrame:
    frames = []
    for csv, label in [(or_csv, "RISK baseline"), (normal_csv, "NORMAL baseline")]:
        df = pd.read_csv(csv)
        df = df.assign(category=label)
        frames.append(aggregate(df))
    return pd.concat(frames, ignore_index=True)


def _b64(png_path: Path) -> str:
    return base64.b64encode(png_path.read_bytes()).decode()


def bar_chart(agg: pd.DataFrame, base: pd.DataFrame, out_png: Path) -> None:
    # dataviz: magnitude-by-category -> bars anchored to baseline; single series
    # in one categorical hue; recessive muted axes + hairline y-grid; reference
    # lines in status colors, dashed and directly labeled (label = the icon+label
    # mitigation so meaning never rides on hue alone).
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)

    ax.bar(
        agg.cluster.str.replace("cluster_id", ""),
        agg.mismatch_rate_4s,
        color=_BAR,
        width=0.72,
        zorder=3,
    )
    for _, row in base.iterrows():
        color = _BASELINE_COLORS.get(row.cluster, _INK_MUTED)
        ax.axhline(row.mismatch_rate_4s, linestyle="--", linewidth=1.5, color=color, zorder=2)
        ax.annotate(
            f"{row.cluster} ({row.mismatch_rate_4s:.1%})",
            xy=(0.99, row.mismatch_rate_4s),
            xycoords=("axes fraction", "data"),
            ha="right",
            va="bottom",
            fontsize=8,
            color=color,
        )

    ax.set_xlabel("cluster id", color=_INK_MUTED, fontsize=10)
    ax.set_ylabel("mismatch rate @4s (frac_close_4s == 0)", color=_INK_MUTED, fontsize=10)
    ax.set_title(
        "Per-cluster human-match mismatch rate (Takanawa SFT k=20, 50 frames/cluster)",
        color=_INK_PRIMARY,
        fontsize=12,
    )
    ax.tick_params(colors=_INK_MUTED, labelsize=9)
    ax.grid(axis="y", color=_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(_GRID)

    fig.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor=_SURFACE)
    plt.close(fig)


def render_html(
    agg_all: pd.DataFrame, chart_png: Path, overlay_pngs: list[Path], out_html: Path
) -> None:
    cols = list(agg_all.columns)
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = ""
    for _, r in agg_all.iterrows():
        tds = "".join(
            f"<td>{r[c]:.3f}</td>" if isinstance(r[c], float) else f"<td>{r[c]}</td>" for c in cols
        )
        cls = ' class="baseline"' if "baseline" in str(r.cluster) else ""
        body += f"<tr{cls}>{tds}</tr>\n"
    overlays = "\n".join(
        f'<figure><img src="data:image/png;base64,{_b64(p)}" style="max-width:640px">'
        f"<figcaption>{p.stem}</figcaption></figure>"
        for p in overlay_pngs
    )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Per-cluster human-match report</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; max-width: 1100px; }}
table {{ border-collapse: collapse; font-size: 13px; }}
th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: right; }}
th {{ background: #f0f0f0; }}
tr.baseline {{ background: #fff6e0; font-weight: bold; }}
figure {{ display: inline-block; margin: 8px; }}
figcaption {{ font-size: 12px; text-align: center; }}
</style></head><body>
<h1>Per-cluster human-match report — Takanawa SFT k=20</h1>
<p>50 kmeans-sampled training frames per cluster, N=64 planner samples per frame,
deployed x2 ONNX. Baseline rows from the risk-scene / normal runs (same settings).</p>
<img src="data:image/png;base64,{_b64(chart_png)}" style="max-width:100%">
<h2>Cluster ranking (worst first)</h2>
<table><tr>{head}</tr>\n{body}</table>
<h2>Overlays — worst 2 clusters, top-3 mismatch frames each</h2>
{overlays}
</body></html>"""
    out_html.write_text(html)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cluster_csv", required=True)
    p.add_argument("--or_csv", required=True)
    p.add_argument("--normal_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_samples", type=int, default=64)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.cluster_csv)
    agg = aggregate(df)
    base = baseline_rows(args.or_csv, args.normal_csv)
    agg_all = pd.concat([agg, base], ignore_index=True)
    agg_all.to_csv(out / "cluster_summary.csv", index=False)

    chart_png = out / "mismatch_by_cluster.png"
    bar_chart(agg, base, chart_png)

    from human_match_prototype.analyze import overlay
    from human_match_prototype.sampler import TrajectorySampler

    sampler = TrajectorySampler(MODEL_DIR / "args.json", MODEL_DIR / "diffusion_planner.onnx")
    overlay_dir = out / "overlays"
    overlay_dir.mkdir(exist_ok=True)
    overlay_pngs = []
    for cluster in agg.cluster.head(2):
        worst = df[df.category == cluster].nlargest(3, "min_ade_4s")
        for _, row in worst.iterrows():
            png = overlay_dir / f"{cluster}_{Path(row.npz_path).stem}.png"
            overlay(sampler, row.npz_path, png, args.num_samples)
            overlay_pngs.append(png)

    render_html(agg_all, chart_png, overlay_pngs, out / "report.html")
    print(f"wrote {out / 'report.html'}")


if __name__ == "__main__":
    main()
