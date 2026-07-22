"""Per-location multi-human comparison report.

Usage:
  .venv/bin/python -m human_match_prototype.multi_human_report \
    --multi_csv data/human_match/scores_or_multi.csv \
    --output_dir data/human_match/multi_human_report \
    [--top_k 20]
"""

import argparse
import base64
from pathlib import Path

import pandas as pd

COVERAGE_HIGH_THRESHOLD = 0.5
INSUFFICIENT_DATA_THRESHOLD = 5


def diagnose_mismatch(row: dict) -> str:
    if row.get("mismatch_4s", 0) < 0.5:
        return "no_mismatch"
    if row.get("n_humans", 0) < INSUFFICIENT_DATA_THRESHOLD:
        return "insufficient_data"
    if row.get("dp_human_coverage_4s", 0) >= COVERAGE_HIGH_THRESHOLD:
        return "unusual_test_human"
    return "planner_deficiency"


def _b64(png_path: Path) -> str:
    return base64.b64encode(png_path.read_bytes()).decode()


def render_location_report(rows: list[dict], overlay_pngs: list[Path], out_html: Path) -> None:
    if not rows:
        out_html.write_text(
            '<!DOCTYPE html>\n<html><head><meta charset="utf-8">'
            "<title>Multi-human location report</title></head><body>"
            "<h1>Multi-human trajectory comparison report</h1>"
            "<p>No mismatches found.</p></body></html>"
        )
        return

    cols = [
        "npz_path",
        "n_humans",
        "dp_human_coverage_4s",
        "human_dp_coverage_4s",
        "human_spread_4s",
        "test_human_typicality_4s",
        "mismatch_4s",
        "min_ade_4s",
        "diagnosis",
    ]
    available_cols = [c for c in cols if c in rows[0]]

    head = "".join(f"<th>{c}</th>" for c in available_cols)
    body = ""
    for r in rows:
        tds = "".join(
            f"<td>{r[c]:.3f}</td>" if isinstance(r.get(c), float) else f"<td>{r.get(c, '')}</td>"
            for c in available_cols
        )
        cls = ' class="deficiency"' if r.get("diagnosis") == "planner_deficiency" else ""
        body += f"<tr{cls}>{tds}</tr>\n"

    overlays = "\n".join(
        f'<figure><img src="data:image/png;base64,{_b64(p)}" style="max-width:640px">'
        f"<figcaption>{p.stem}</figcaption></figure>"
        for p in overlay_pngs
        if p.exists()
    )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Multi-human location report</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; max-width: 1200px; }}
table {{ border-collapse: collapse; font-size: 13px; }}
th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: right; }}
th {{ background: #f0f0f0; }}
tr.deficiency {{ background: #ffe0e0; }}
figure {{ display: inline-block; margin: 8px; }}
figcaption {{ font-size: 12px; text-align: center; }}
</style></head><body>
<h1>Multi-human trajectory comparison report</h1>
<p>DP samples vs matched training humans at the same lanelet. Diagnosis:
planner_deficiency (red) = DP misses what humans do; unusual_test_human = single
test human atypical; insufficient_data = too few training examples.</p>
<h2>Top mismatch frames</h2>
<table><tr>{head}</tr>\n{body}</table>
<h2>BEV overlays (blue=DP, green=test human, orange=training humans)</h2>
{overlays}
</body></html>"""
    out_html.write_text(html)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--multi_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--top_k", type=int, default=20)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.multi_csv)

    df["diagnosis"] = df.apply(lambda r: diagnose_mismatch(r.to_dict()), axis=1)
    worst = df[df["mismatch_4s"] == 1].nlargest(args.top_k, "min_ade_4s")
    rows = worst.to_dict("records")

    overlay_dir = out / "overlays"
    overlay_dir.mkdir(exist_ok=True)
    overlay_pngs = [overlay_dir / f"{Path(r['npz_path']).stem}.png" for r in rows]

    render_location_report(rows, overlay_pngs, out / "report.html")
    print(f"Wrote {out / 'report.html'}")


if __name__ == "__main__":
    main()
