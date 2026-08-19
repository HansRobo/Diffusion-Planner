"""Compare two closed-loop evaluation runs and report summary metrics and per-scenario deltas."""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Any

from scenario_generation.scenario_sim_diagnostics import (
    classify_failure,
    classify_hazard_engagement,
    diagnose_case,
    is_passed,
    parse_junit_verdict,
)


def load_run_data(run_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    summary_path = run_dir / "summary.json"
    segments_path = run_dir / "segments.jsonl"

    summary = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    segments = {}
    if segments_path.exists():
        for line in segments_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            scen_key = row.get("route") or row.get("scenario") or row.get("name") or row.get("route_id")
            if not scen_key and "row_file" in row:
                scen_key = Path(row["row_file"]).stem
            if scen_key:
                # Use basename of route if it is a full path
                scen_key = Path(scen_key).name
                case_dir = run_dir / scen_key
                row["diagnostics"] = diagnose_case(row, case_dir if case_dir.is_dir() else None)
                segments[scen_key] = row
    else:
        # Fallback to scanning individual row.json files in subdirectories
        for rp in sorted(run_dir.glob("*/row.json")):
            try:
                row = json.loads(rp.read_text(encoding="utf-8"))
            except Exception:
                continue
            scen_key = rp.parent.name
            row["diagnostics"] = diagnose_case(row, rp.parent)
            segments[scen_key] = row

    return summary, segments


def _format_stat(vals: list[float], unit: str = "", fmt: str = ".2f") -> str:
    if not vals:
        return "N/A"
    clean = [v for v in vals if v is not None and not math.isinf(v) and not math.isnan(v)]
    if not clean:
        return "N/A"
    clean.sort()
    n = len(clean)
    med = clean[n // 2]
    mean = sum(clean) / n
    p5 = clean[int(0.05 * n)]
    p95 = clean[int(0.95 * n)]
    return f"mean {mean:{fmt}}{unit} | p5 {p5:{fmt}} | med {med:{fmt}} | p95 {p95:{fmt}}"


def compare_runs(base_dir: Path, treat_dir: Path, title_base: str = "Baseline", title_treat: str = "Treatment") -> str:
    base_sum, base_segs = load_run_data(base_dir)
    treat_sum, treat_segs = load_run_data(treat_dir)

    common_scenarios = sorted(set(base_segs.keys()) & set(treat_segs.keys()))

    wins = []
    losses = []
    both_pass = []
    both_fail = []

    base_cats = collections.Counter()
    treat_cats = collections.Counter()

    base_hazards = collections.Counter()
    treat_hazards = collections.Counter()

    for s in common_scenarios:
        b_row = base_segs[s]
        t_row = treat_segs[s]

        b_diag = b_row.get("diagnostics") or diagnose_case(b_row)
        t_diag = t_row.get("diagnostics") or diagnose_case(t_row)

        b_pass = b_diag["passed"]
        t_pass = t_diag["passed"]

        b_cat = b_diag["failure_category"]
        t_cat = t_diag["failure_category"]
        base_cats[b_cat] += 1
        treat_cats[t_cat] += 1

        base_hazards[b_diag["hazard_verdict"]] += 1
        treat_hazards[t_diag["hazard_verdict"]] += 1

        if not b_pass and t_pass:
            wins.append((s, b_cat, t_cat))
        elif b_pass and not t_pass:
            losses.append((s, b_cat, t_cat))
        elif b_pass and t_pass:
            both_pass.append(s)
        else:
            both_fail.append((s, b_cat, t_cat))

    base_total = len(base_segs)
    treat_total = len(treat_segs)

    base_pass_cnt = base_cats.get("PASS", 0)
    treat_pass_cnt = treat_cats.get("PASS", 0)

    base_pr = (base_pass_cnt / base_total * 100) if base_total else 0.0
    treat_pr = (treat_pass_cnt / treat_total * 100) if treat_total else 0.0

    lines = []
    lines.append("# Closed-Loop Scenario Evaluation Comparison")
    lines.append(f"- **{title_base}**: `{base_dir}`")
    lines.append(f"- **{title_treat}**: `{treat_dir}`")
    lines.append("")

    # Section 1: Summary Table
    lines.append("## 1. Summary Comparison")
    lines.append(f"| Metric | {title_base} | {title_treat} | Delta |")
    lines.append("|---|---|---|---|")
    lines.append(f"| **Total Cases** | {base_total} | {treat_total} | {treat_total - base_total:+d} |")
    lines.append(f"| **Common Evaluated Cases** | {len(common_scenarios)} | {len(common_scenarios)} | - |")
    lines.append(f"| **Passed Cases** | {base_pass_cnt} ({base_pr:.1f}%) | {treat_pass_cnt} ({treat_pr:.1f}%) | {treat_pass_cnt - base_pass_cnt:+d} ({treat_pr - base_pr:+.1f}%) |")
    lines.append(f"| **Wins (Improved Fail → Pass)** | - | {len(wins)} | +{len(wins)} |")
    lines.append(f"| **Losses (Regressed Pass → Fail)** | - | {len(losses)} | -{len(losses)} |")
    lines.append(f"| **Net Improvement** | - | - | **{len(wins) - len(losses):+d}** |")
    lines.append("")

    # Section 2: Failure Category Breakdown (Hierarchical)
    all_cats = [
        "PASS",
        "COLLISION",
        "ROAD_DEPARTURE",
        "FROZEN_STANDSTILL",
        "PROXIMITY_DEPARTURE",
        "GOAL_STOP_FAILURE",
        "SPEED_TIMEOUT",
        "UNMET_CONDITION",
        "SCENARIO_REFUSED",
        "ERROR",
    ]
    lines.append("## 2. Failure Root-Cause Breakdown")
    lines.append(f"| Category | {title_base} | {title_treat} | Delta | Description |")
    lines.append("|---|---|---|---|---|")
    for cat in all_cats:
        b_c = base_cats.get(cat, 0)
        t_c = treat_cats.get(cat, 0)
        b_pct = (b_c / len(common_scenarios) * 100) if common_scenarios else 0.0
        t_pct = (t_c / len(common_scenarios) * 100) if common_scenarios else 0.0
        delta = t_c - b_c
        sign = f"{delta:+d}" if delta != 0 else "0"
        lines.append(f"| **`{cat}`** | {b_c} ({b_pct:.1f}%) | {t_c} ({t_pct:.1f}%) | {sign} ({t_pct - b_pct:+.1f}%) | {_cat_description(cat)} |")
    lines.append("")

    # Section 3: Hazard Engagement Analysis
    lines.append("## 3. Hazard Engagement & Validity Analysis")
    lines.append(f"| Hazard Verdict | {title_base} | {title_treat} | Delta | Interpretation |")
    lines.append("|---|---|---|---|---|")
    hazard_keys = ["PASS_AVOIDED", "PASS_UNENGAGED", "FAIL_COLLISION", "FAIL_MOBILITY", "FAIL_OTHER"]
    for hz in hazard_keys:
        b_h = base_hazards.get(hz, 0)
        t_h = treat_hazards.get(hz, 0)
        d_h = t_h - b_h
        sign = f"{d_h:+d}" if d_h != 0 else "0"
        lines.append(f"| **`{hz}`** | {b_h} | {t_h} | {sign} | {_hazard_description(hz)} |")
    lines.append("")

    # Section 4: Driving Safety & Quality Metrics
    lines.append("## 4. Driving Quality & Safety Metrics")
    b_clearances = [r.get("object", {}).get("clearance_min_m") for r in base_segs.values()]
    t_clearances = [r.get("object", {}).get("clearance_min_m") for r in treat_segs.values()]
    b_accels = [r.get("strong_brake", {}).get("strongest_mps2") for r in base_segs.values()]
    t_accels = [r.get("strong_brake", {}).get("strongest_mps2") for r in treat_segs.values()]
    b_progress = [r.get("progress_m", 0.0) for r in base_segs.values()]
    t_progress = [r.get("progress_m", 0.0) for r in treat_segs.values()]
    b_steps = [r.get("n_steps_run", 0) for r in base_segs.values()]
    t_steps = [r.get("n_steps_run", 0) for r in treat_segs.values()]

    lines.append(f"| Metric | {title_base} | {title_treat} |")
    lines.append("|---|---|---|")
    lines.append(f"| **Min Obstacle Clearance** | {_format_stat(b_clearances, ' m')} | {_format_stat(t_clearances, ' m')} |")
    lines.append(f"| **Strongest Decel** | {_format_stat(b_accels, ' m/s²', fmt='.2f')} | {_format_stat(t_accels, ' m/s²', fmt='.2f')} |")
    lines.append(f"| **Progress** | {_format_stat(b_progress, ' m', fmt='.1f')} | {_format_stat(t_progress, ' m', fmt='.1f')} |")
    lines.append(f"| **Run Scored Steps** | {_format_stat(b_steps, ' steps', fmt='.0f')} | {_format_stat(t_steps, ' steps', fmt='.0f')} |")
    lines.append("")

    # Section 5: Regressions
    if losses:
        lines.append(f"## 5. Regressions ({len(losses)} cases: Pass → Fail)")
        lines.append(f"| Scenario | {title_base} Status | {title_treat} Cause | Details |")
        lines.append("|---|---|---|---|")
        for s, b_cat, t_cat in losses:
            t_row = treat_segs[s]
            t_diag = t_row.get("diagnostics") or {}
            trig = t_diag.get("verdict_trigger") or t_row.get("terminated") or ""
            lines.append(f"| `{s}` | `{b_cat}` | **`{t_cat}`** | {trig[:60]} |")
        lines.append("")

    # Section 6: Improvements
    if wins:
        lines.append(f"## 6. Improvements ({len(wins)} cases: Fail → Pass)")
        lines.append(f"| Scenario | {title_base} Cause | {title_treat} Status |")
        lines.append("|---|---|---|")
        for s, b_cat, t_cat in wins:
            lines.append(f"| `{s}` | `{b_cat}` | **`{t_cat}`** |")
        lines.append("")

    return "\n".join(lines)


def _cat_description(cat: str) -> str:
    descs = {
        "PASS": "Storyboard completed with pass verdict",
        "COLLISION": "Collided with dynamic/static obstacle",
        "ROAD_DEPARTURE": "Crossed road boundary (no obstacle collision)",
        "FROZEN_STANDSTILL": "Failed to pull away (v < 0.5 m/s) or StandStill cutoff",
        "PROXIMITY_DEPARTURE": "Triggered proximity condition (act_lateral_check etc.)",
        "GOAL_STOP_FAILURE": "Reached goal area but failed to stop / overshot",
        "SPEED_TIMEOUT": "Moved but failed to finish within simulation deadline",
        "UNMET_CONDITION": "Other OpenSCENARIO condition unmet",
        "SCENARIO_REFUSED": "Interpreter refused to parse/configure scenario",
        "ERROR": "Simulation crash or unhandled runtime error",
    }
    return descs.get(cat, "")


def _hazard_description(hz: str) -> str:
    descs = {
        "PASS_AVOIDED": "True Pass: Hazard actively presented and safely avoided",
        "PASS_UNENGAGED": "Trivial Pass: Vehicle did not engage/encounter hazard",
        "FAIL_COLLISION": "Safety Fail: Direct collision with obstacle",
        "FAIL_MOBILITY": "Mobility Fail: Ego froze or timed out before encountering/clearing hazard",
        "FAIL_OTHER": "Other Fail: Road departure, proximity trigger, or unmet conditions",
    }
    return descs.get(hz, "")


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
        args.out.write_text(report, encoding="utf-8")
        print(f"Report written to: {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
