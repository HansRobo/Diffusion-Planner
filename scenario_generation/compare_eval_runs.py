"""Compare two closed-loop evaluation runs and report summary metrics and per-scenario deltas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_run_data(run_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    summary_path = run_dir / "summary.json"
    segments_path = run_dir / "segments.jsonl"

    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())

    segments = {}
    if segments_path.exists():
        for line in segments_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            scen_key = row.get("scenario") or row.get("name") or row.get("route_id")
            if not scen_key and "row_file" in row:
                scen_key = Path(row["row_file"]).stem
            if scen_key:
                segments[scen_key] = row
    return summary, segments


def is_passed(row: dict[str, Any]) -> bool:
    rk = str(row.get("result_kind", "")).lower()
    term = str(row.get("terminated", "")).lower()
    if rk in ("passed", "success", "exitsuccess"):
        return True
    if term in ("passed", "success", "exitsuccess"):
        return True
    return False


def compare_runs(base_dir: Path, treat_dir: Path, title_base: str = "Baseline", title_treat: str = "Treatment") -> str:
    base_sum, base_segs = load_run_data(base_dir)
    treat_sum, treat_segs = load_run_data(treat_dir)

    all_scenarios = sorted(set(base_segs.keys()) | set(treat_segs.keys()))
    common_scenarios = sorted(set(base_segs.keys()) & set(treat_segs.keys()))

    wins = []
    losses = []
    both_pass = []
    both_fail = []

    for s in common_scenarios:
        b_pass = is_passed(base_segs[s])
        t_pass = is_passed(treat_segs[s])
        if not b_pass and t_pass:
            wins.append(s)
        elif b_pass and not t_pass:
            losses.append(s)
        elif b_pass and t_pass:
            both_pass.append(s)
        else:
            both_fail.append(s)

    base_pass_cnt = sum(1 for s in base_segs.values() if is_passed(s))
    treat_pass_cnt = sum(1 for s in treat_segs.values() if is_passed(s))

    base_total = len(base_segs)
    treat_total = len(treat_segs)

    base_pr = (base_pass_cnt / base_total * 100) if base_total else 0.0
    treat_pr = (treat_pass_cnt / treat_total * 100) if treat_total else 0.0

    lines = []
    lines.append(f"# Closed-Loop Evaluation Comparison")
    lines.append(f"- **{title_base}**: `{base_dir}`")
    lines.append(f"- **{title_treat}**: `{treat_dir}`")
    lines.append("")
    lines.append("## 1. Summary Comparison")
    lines.append("| Metric | " + title_base + " | " + title_treat + " | Delta |")
    lines.append("|---|---|---|---|")
    lines.append(f"| **Total Cases** | {base_total} | {treat_total} | {treat_total - base_total:+d} |")
    lines.append(f"| **Passed Cases** | {base_pass_cnt} ({base_pr:.1f}%) | {treat_pass_cnt} ({treat_pr:.1f}%) | {treat_pass_cnt - base_pass_cnt:+d} ({treat_pr - base_pr:+.1f}%) |")
    lines.append(f"| **Wins (Improved)** | - | {len(wins)} | +{len(wins)} |")
    lines.append(f"| **Losses (Regressed)** | - | {len(losses)} | -{len(losses)} |")
    lines.append(f"| **Net Improvement** | - | - | **{len(wins) - len(losses):+d}** |")
    lines.append("")

    if losses:
        lines.append(f"## 2. Regressions ({len(losses)} cases: Pass -> Fail)")
        lines.append("| Scenario | " + title_base + " Reason | " + title_treat + " Reason |")
        lines.append("|---|---|---|")
        for s in losses:
            b_r = base_segs[s].get("result_kind") or base_segs[s].get("terminated")
            t_r = treat_segs[s].get("result_kind") or treat_segs[s].get("terminated")
            lines.append(f"| `{s}` | {b_r} | {t_r} |")
        lines.append("")

    if wins:
        lines.append(f"## 3. Improvements ({len(wins)} cases: Fail -> Pass)")
        lines.append("| Scenario | " + title_base + " Reason | " + title_treat + " Reason |")
        lines.append("|---|---|---|")
        for s in wins:
            b_r = base_segs[s].get("result_kind") or base_segs[s].get("terminated")
            t_r = treat_segs[s].get("result_kind") or treat_segs[s].get("terminated")
            lines.append(f"| `{s}` | {b_r} | {t_r} |")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two closed-loop eval runs.")
    parser.add_argument("base", type=Path, help="Baseline run directory")
    parser.add_argument("treat", type=Path, help="Treatment run directory")
    parser.add_argument("--base-name", default="Baseline", help="Baseline name")
    parser.add_argument("--treat-name", default="Treatment", help="Treatment name")
    parser.add_argument("--out", type=Path, default=None, help="Output markdown path")

    args = parser.parse_args()
    report = compare_runs(args.base, args.treat, args.base_name, args.treat_name)

    if args.out:
        args.out.write_text(report)
        print(f"Report written to: {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
