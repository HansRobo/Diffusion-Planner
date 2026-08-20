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
    """Load evaluation run data strictly per FORMAT.md."""
    cases_path = run_dir / "cases.jsonl"
    if not cases_path.is_file():
        cases_path = run_dir / "segments.jsonl"
    run_json_path = run_dir / "run.json"
    if not run_json_path.is_file():
        run_json_path = run_dir / "summary.json"
    scenarios_json_path = run_dir / "scenarios.json"

    if not cases_path.is_file():
        raise FileNotFoundError(f"Neither cases.jsonl nor segments.jsonl found in {run_dir}")

    summary: dict[str, Any] = {}
    if run_json_path.is_file():
        try:
            summary = json.loads(run_json_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    if scenarios_json_path.is_file():
        try:
            summary["scenarios_meta"] = json.loads(scenarios_json_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    cases: dict[str, dict[str, Any]] = {}
    for line in cases_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        scen_key = row.get("case_key") or row.get("scene_id") or row.get("route") or row.get("id") or f"{row.get('scenario')}_{row.get('route')}"
        row["diagnostics"] = diagnose_case(row)
        cases[scen_key] = row

    return summary, cases


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
    lines.append("# クローズドループ シナリオ評価 比較レポート (Scenario Evaluation Comparison)")
    lines.append(f"- **{title_base} (基準)**: `{base_dir}`")
    lines.append(f"- **{title_treat} (比較対象)**: `{treat_dir}`")
    lines.append("")

    # Section 1: Summary Table
    lines.append("## 1. サマリー比較 (Summary Comparison)")
    lines.append(f"| 指標 (Metric) | {title_base} | {title_treat} | 増減差分 (Delta) |")
    lines.append("|---|---|---|---|")
    lines.append(f"| **総ケース数 (Total Cases)** | {base_total} | {treat_total} | {treat_total - base_total:+d} |")
    lines.append(f"| **共通評価ケース数 (Common Cases)** | {len(common_scenarios)} | {len(common_scenarios)} | - |")
    lines.append(f"| **合格ケース数 (Passed Cases)** | {base_pass_cnt} ({base_pr:.1f}%) | {treat_pass_cnt} ({treat_pr:.1f}%) | {treat_pass_cnt - base_pass_cnt:+d} ({treat_pr - base_pr:+.1f}%) |")
    lines.append(f"| **改善ケース (Wins: Fail → Pass)** | - | {len(wins)} | +{len(wins)} |")
    lines.append(f"| **悪化ケース (Losses: Pass → Fail)** | - | {len(losses)} | -{len(losses)} |")
    lines.append(f"| **純改善数 (Net Improvement)** | - | - | **{len(wins) - len(losses):+d}** |")
    lines.append("")

    # Section 2: Failure Category Breakdown (Hierarchical)
    all_cats = [
        ("PASS", "合格"),
        ("COLLISION", "障害物衝突"),
        ("ROAD_DEPARTURE", "路端逸脱"),
        ("FROZEN_STANDSTILL", "発進不能・静止"),
        ("PROXIMITY_DEPARTURE", "過接近打ち切り"),
        ("GOAL_STOP_FAILURE", "ゴール停止失敗"),
        ("SPEED_TIMEOUT", "速度不足時間切れ"),
        ("UNMET_CONDITION", "条件未達"),
        ("SCENARIO_REFUSED", "シナリオ拒否"),
        ("ERROR", "異常終了"),
    ]
    lines.append("## 2. 失敗要因の排他分類 (Failure Root-Cause Breakdown)")
    lines.append(f"| 失敗カテゴリ | {title_base} | {title_treat} | 増減差分 | 説明・影響 |")
    lines.append("|---|---|---|---|---|")
    for cat, cat_ja in all_cats:
        b_c = base_cats.get(cat, 0)
        t_c = treat_cats.get(cat, 0)
        b_pct = (b_c / len(common_scenarios) * 100) if common_scenarios else 0.0
        t_pct = (t_c / len(common_scenarios) * 100) if common_scenarios else 0.0
        delta = t_c - b_c
        sign = f"{delta:+d}" if delta != 0 else "0"
        lines.append(f"| **{cat_ja}** (`{cat}`) | {b_c} ({b_pct:.1f}%) | {t_c} ({t_pct:.1f}%) | {sign} ({t_pct - b_pct:+.1f}%) | {_cat_description(cat)} |")
    lines.append("")

    # Section 3: Hazard Engagement Analysis
    lines.append("## 3. ハザード提示検知と妥当性分析 (Hazard Engagement Analysis)")
    lines.append(f"| ハザード判定 | {title_base} | {title_treat} | 増減差分 | 解釈 |")
    lines.append("|---|---|---|---|---|")
    hazard_keys = [
        ("PASS_AVOIDED", "真の合格 (回避成功)"),
        ("PASS_UNENGAGED", "見かけの合格 (未遭遇)"),
        ("FAIL_COLLISION", "安全失敗 (衝突)"),
        ("FAIL_MOBILITY", "走行性失敗 (フリーズ/タイムアウト)"),
        ("FAIL_OTHER", "その他失敗"),
    ]
    for hz, hz_ja in hazard_keys:
        b_h = base_hazards.get(hz, 0)
        t_h = treat_hazards.get(hz, 0)
        d_h = t_h - b_h
        sign = f"{d_h:+d}" if d_h != 0 else "0"
        lines.append(f"| **{hz_ja}** (`{hz}`) | {b_h} | {t_h} | {sign} | {_hazard_description(hz)} |")
    lines.append("")

    # Section 4: Driving Safety & Quality Metrics
    lines.append("## 4. 走行品質・安全性メトリクス (Driving Quality & Safety Metrics)")
    b_clearances = [r.get("object", {}).get("clearance_min_m") for r in base_segs.values()]
    t_clearances = [r.get("object", {}).get("clearance_min_m") for r in treat_segs.values()]
    b_accels = [r.get("strong_brake", {}).get("strongest_mps2") for r in base_segs.values()]
    t_accels = [r.get("strong_brake", {}).get("strongest_mps2") for r in treat_segs.values()]
    b_progress = [r.get("progress_m", 0.0) for r in base_segs.values()]
    t_progress = [r.get("progress_m", 0.0) for r in treat_segs.values()]
    b_steps = [r.get("n_steps_run", 0) for r in base_segs.values()]
    t_steps = [r.get("n_steps_run", 0) for r in treat_segs.values()]

    lines.append(f"| 指標 (Metric) | {title_base} | {title_treat} |")
    lines.append("|---|---|---|")
    lines.append(f"| **最小障害物クリアランス** | {_format_stat(b_clearances, ' m')} | {_format_stat(t_clearances, ' m')} |")
    lines.append(f"| **最大減速度 (急ブレーキ)** | {_format_stat(b_accels, ' m/s²', fmt='.2f')} | {_format_stat(t_accels, ' m/s²', fmt='.2f')} |")
    lines.append(f"| **シナリオ走行距離** | {_format_stat(b_progress, ' m', fmt='.1f')} | {_format_stat(t_progress, ' m', fmt='.1f')} |")
    lines.append(f"| **採点走行ステップ数** | {_format_stat(b_steps, ' steps', fmt='.0f')} | {_format_stat(t_steps, ' steps', fmt='.0f')} |")
    lines.append("")

    from scenario_generation.scenario_comparison_html_report import CATEGORY_LABELS_JA

    # Section 5: Regressions
    if losses:
        lines.append(f"## 5. 悪化・リグレッション ({len(losses)} 件: Pass → Fail)")
        lines.append(f"| シナリオ | {title_base} 状態 | {title_treat} 原因 | 詳細 |")
        lines.append("|---|---|---|---|")
        for s, b_cat, t_cat in losses:
            t_row = treat_segs[s]
            t_diag = t_row.get("diagnostics") or {}
            trig = t_diag.get("verdict_trigger") or t_row.get("terminated") or ""
            b_cat_ja = CATEGORY_LABELS_JA.get(b_cat, b_cat)
            t_cat_ja = CATEGORY_LABELS_JA.get(t_cat, t_cat)
            lines.append(f"| `{s}` | `{b_cat_ja}` | **`{t_cat_ja}`** | {trig[:60]} |")
        lines.append("")

    # Section 6: Improvements
    if wins:
        lines.append(f"## 6. 改善 ({len(wins)} 件: Fail → Pass)")
        lines.append(f"| シナリオ | {title_base} 原因 | {title_treat} 状態 |")
        lines.append("|---|---|---|")
        for s, b_cat, t_cat in wins:
            b_cat_ja = CATEGORY_LABELS_JA.get(b_cat, b_cat)
            t_cat_ja = CATEGORY_LABELS_JA.get(t_cat, t_cat)
            lines.append(f"| `{s}` | `{b_cat_ja}` | **`{t_cat_ja}`** |")
        lines.append("")

    return "\n".join(lines)


def _cat_description(cat: str) -> str:
    descs = {
        "PASS": "正常完了（Pass 判定）",
        "COLLISION": "障害物衝突（歩行者・他車・二輪等）",
        "ROAD_DEPARTURE": "路端逸脱・境界接触（障害物衝突なし）",
        "FROZEN_STANDSTILL": "発進不能（最高速度 < 0.5 m/s）または静止検出打ち切り",
        "PROXIMITY_DEPARTURE": "過接近・近接条件打ち切り（act_lateral_check 等）",
        "GOAL_STOP_FAILURE": "ゴール到達後の停止失敗・行き過ぎ（goal_position 未達）",
        "SPEED_TIMEOUT": "低速・シミュレーション制限時間切れ",
        "UNMET_CONDITION": "その他 OpenSCENARIO 条件未達",
        "SCENARIO_REFUSED": "シナリオ解釈器による拒否（構文・設定エラー）",
        "ERROR": "シミュレータ/ワーカーの異常終了・クラッシュ",
    }
    return descs.get(cat, "")


def _hazard_description(hz: str) -> str:
    descs = {
        "PASS_AVOIDED": "真の合格 (True Pass): ハザードが提示され、安全に回避して走破",
        "PASS_UNENGAGED": "見かけの合格 (Trivial Pass): 低速等のためハザード未提示・非遭遇で通過",
        "FAIL_COLLISION": "安全失敗: 障害物との直接衝突",
        "FAIL_MOBILITY": "走行性失敗: 発進不能・フリーズまたは時間切れ",
        "FAIL_OTHER": "その他失敗: 路端逸脱・近接条件トリガー・その他条件未達",
    }
    return descs.get(hz, "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two closed-loop eval runs.")
    parser.add_argument("base", type=Path, help="Baseline run directory")
    parser.add_argument("treat", type=Path, help="Treatment run directory")
    parser.add_argument("--base-name", default="Baseline", help="Baseline name")
    parser.add_argument("--treat-name", default="Treatment", help="Treatment name")
    parser.add_argument("--out", type=Path, default=None, help="Output markdown path")
    parser.add_argument("--html", type=Path, default=None, help="Output interactive HTML report path")
    parser.add_argument("--title", default=None, help="Report title for HTML")
    parser.add_argument("--subtitle", default="", help="Report subtitle for HTML")

    args = parser.parse_args()
    report = compare_runs(args.base, args.treat, args.base_name, args.treat_name)

    if args.out:
        args.out.write_text(report, encoding="utf-8")
        print(f"Markdown report written to: {args.out}")
    else:
        print(report)

    if args.html:
        from scenario_generation.scenario_comparison_html_report import build_comparison_html_report

        title = args.title or f"Closed-Loop Scenario Model Comparison ({args.base_name} vs {args.treat_name})"
        html_path = build_comparison_html_report(
            base_dir=args.base,
            treat_dir=args.treat,
            out_path=args.html,
            title_base=args.base_name,
            title_treat=args.treat_name,
            title=title,
            subtitle=args.subtitle,
        )
        print(f"Interactive HTML report written to: {html_path}")


if __name__ == "__main__":
    main()
