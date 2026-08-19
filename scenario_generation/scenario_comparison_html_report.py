"""Build interactive, self-contained HTML scenario comparison reports.

Compares Baseline vs Treatment evaluation runs across the 10-tier failure taxonomy,
hazard engagement breakdown, driving safety/comfort metrics, and side-by-side video/colormap
inspection cards.
"""

from __future__ import annotations

import collections
import json
import math
import os
from pathlib import Path
from typing import Any

from scenario_generation.scenario_sim_diagnostics import (
    classify_failure,
    classify_hazard_engagement,
    diagnose_case,
    is_passed,
)

_TEMPLATE_PATH = Path(__file__).with_name("scenario_comparison_report_template.html")

CATEGORY_LABELS_JA = {
    "PASS": "合格",
    "COLLISION": "障害物衝突",
    "ROAD_DEPARTURE": "路端逸脱",
    "FROZEN_STANDSTILL": "発進不能・静止",
    "PROXIMITY_DEPARTURE": "過接近打ち切り",
    "GOAL_STOP_FAILURE": "ゴール停止失敗",
    "SPEED_TIMEOUT": "速度不足時間切れ",
    "UNMET_CONDITION": "条件未達",
    "SCENARIO_REFUSED": "シナリオ拒否",
    "ERROR": "異常終了",
}

HAZARD_LABELS_JA = {
    "PASS_AVOIDED": "真の合格 (回避成功)",
    "PASS_UNENGAGED": "見かけの合格 (未遭遇)",
    "FAIL_COLLISION": "安全失敗 (衝突)",
    "FAIL_MOBILITY": "走行性失敗 (フリーズ/タイムアウト)",
    "FAIL_OTHER": "その他失敗",
}

ALL_FAILURE_CATEGORIES = [
    ("PASS", "合格", "正常完了（Pass 判定）"),
    ("COLLISION", "障害物衝突", "障害物衝突（歩行者・他車・二輪等）"),
    ("ROAD_DEPARTURE", "路端逸脱", "路端逸脱・境界接触（障害物衝突なし）"),
    ("FROZEN_STANDSTILL", "発進不能・静止", "発進不能（最高速度 < 0.5 m/s）または静止検出打ち切り"),
    ("PROXIMITY_DEPARTURE", "過接近打ち切り", "過接近・近接条件打ち切り（act_lateral_check 等）"),
    ("GOAL_STOP_FAILURE", "ゴール停止失敗", "ゴール到達後の停止失敗・行き過ぎ（goal_position 未達）"),
    ("SPEED_TIMEOUT", "速度不足時間切れ", "低速・シミュレーション制限時間切れ"),
    ("UNMET_CONDITION", "条件未達", "その他 OpenSCENARIO 条件未達"),
    ("SCENARIO_REFUSED", "シナリオ拒否", "シナリオ解釈器による拒否（構文・設定エラー）"),
    ("ERROR", "異常終了", "シミュレータ/ワーカーの異常終了・クラッシュ"),
]

ALL_HAZARD_VERDICTS = [
    ("PASS_AVOIDED", "真の合格 (回避成功)", "真の合格: ハザードが提示され、安全に回避して走破"),
    ("PASS_UNENGAGED", "見かけの合格 (未遭遇)", "見かけの合格: 低速等のためハザード未提示・非遭遇で通過"),
    ("FAIL_COLLISION", "安全失敗 (衝突)", "安全失敗: 障害物との直接衝突"),
    ("FAIL_MOBILITY", "走行性失敗 (フリーズ/タイムアウト)", "走行性失敗: 発進不能・フリーズまたは時間切れ"),
    ("FAIL_OTHER", "その他失敗", "その他失敗: 路端逸脱・近接条件トリガー・その他条件未達"),
]


def _clean_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        if math.isinf(v) or math.isnan(v):
            return None
        return float(v)
    try:
        f = float(v)
        if math.isinf(f) or math.isnan(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def _sanitize_json_obj(obj: Any) -> Any:
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: _sanitize_json_obj(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_json_obj(v) for v in obj]
    return obj


def _format_stat_summary(vals: list[float], unit: str = "", fmt: str = ".2f") -> str:
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
    return f"平均 {mean:{fmt}}{unit} | p5 {p5:{fmt}} | 中央値 {med:{fmt}} | p95 {p95:{fmt}}"


def _resolve_scenario_media(
    scen_key: str,
    run_dir: Path,
    row: dict[str, Any],
    html_out_dir: Path | None = None,
    media_url_prefix: str | None = None,
    workspace_root: Path | None = None,
) -> tuple[str | None, dict[str, str]]:
    """Resolve video and colormap paths relative to the generated HTML file or as API media URLs."""
    video_rel_path: str | None = None
    colormap_paths: dict[str, str] = {}

    def to_rel(p: Path) -> str:
        if media_url_prefix:
            if workspace_root:
                try:
                    rel = p.resolve().relative_to(workspace_root.resolve())
                    return f"{media_url_prefix.rstrip('/')}/{rel}"
                except ValueError:
                    pass
            return f"{media_url_prefix.rstrip('/')}/{str(p).lstrip('/')}"
        if html_out_dir:
            try:
                return os.path.relpath(p, html_out_dir)
            except ValueError:
                return str(p)
        return str(p)

    scen_id = row.get("scenario") or row.get("scenario_id") or (scen_key.split("_")[0] if "_" in scen_key else scen_key)
    route_id = row.get("route") or (scen_key.split("_", 1)[1] if "_" in scen_key else "")

    media_dir = run_dir / "media" / str(scen_id)

    # 1. Check video
    vid_file = media_dir / f"{route_id}.mp4"
    if vid_file.is_file():
        video_rel_path = to_rel(vid_file)
    elif row.get("video_path") and Path(row["video_path"]).is_file():
        video_rel_path = to_rel(Path(row["video_path"]))

    # 2. Check colormaps
    if media_dir.is_dir() and route_id:
        for img in media_dir.glob(f"{route_id}.*.png"):
            # e.g. ego_speed2p7778.speed.png -> metric "speed"
            metric = img.name.split(f"{route_id}.")[1].rsplit(".png", 1)[0]
            colormap_paths[metric] = to_rel(img)
    elif isinstance(row.get("colormap_paths"), dict):
        for metric, cp in row["colormap_paths"].items():
            p = Path(cp)
            if p.is_file():
                colormap_paths[metric] = to_rel(p)

    # 3. Check rollout trace (JSONL)
    rollout_rel_path = None
    rollout_file = media_dir / f"{route_id}.rollout.jsonl"
    if rollout_file.is_file():
        rollout_rel_path = to_rel(rollout_file)
    elif row.get("rollout_path") and Path(row["rollout_path"]).is_file():
        rollout_rel_path = to_rel(Path(row["rollout_path"]))

    return video_rel_path, colormap_paths, rollout_rel_path


def build_comparison_payload(
    base_dir: Path,
    treat_dir: Path,
    base_segs: dict[str, dict[str, Any]],
    treat_segs: dict[str, dict[str, Any]],
    html_out_dir: Path | None = None,
    title_base: str = "Baseline",
    title_treat: str = "Treatment",
    media_url_prefix: str | None = None,
    workspace_root: Path | None = None,
) -> dict[str, Any]:
    """Construct structured JSON comparison payload."""
    common_scenarios = sorted(set(base_segs.keys()) & set(treat_segs.keys()))

    base_cats = collections.Counter()
    treat_cats = collections.Counter()
    base_hazards = collections.Counter()
    treat_hazards = collections.Counter()

    wins = []
    losses = []
    both_pass = []
    both_fail = []

    scenario_cards = []

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

        b_hz = b_diag["hazard_verdict"]
        t_hz = t_diag["hazard_verdict"]
        base_hazards[b_hz] += 1
        treat_hazards[t_hz] += 1

        if not b_pass and t_pass:
            delta_kind = "win"
            wins.append(s)
        elif b_pass and not t_pass:
            delta_kind = "loss"
            losses.append(s)
        elif b_pass and t_pass:
            delta_kind = "both_pass"
            both_pass.append(s)
        else:
            delta_kind = "both_fail"
            both_fail.append(s)

        b_vid, b_cm, b_roll = _resolve_scenario_media(
            s, base_dir, b_row, html_out_dir=html_out_dir, media_url_prefix=media_url_prefix, workspace_root=workspace_root
        )
        t_vid, t_cm, t_roll = _resolve_scenario_media(
            s, treat_dir, t_row, html_out_dir=html_out_dir, media_url_prefix=media_url_prefix, workspace_root=workspace_root
        )

        scenario_cards.append({
            "name": s,
            "delta_kind": delta_kind,
            "base": {
                "passed": b_pass,
                "failure_category": b_cat,
                "failure_category_ja": CATEGORY_LABELS_JA.get(b_cat, b_cat),
                "hazard_verdict": b_hz,
                "hazard_verdict_ja": HAZARD_LABELS_JA.get(b_hz, b_hz),
                "verdict_trigger": b_diag.get("verdict_trigger"),
                "verdict_unmet": b_diag.get("verdict_unmet", []),
                "terminated": b_row.get("terminated", ""),
                "min_clearance": _clean_float((b_row.get("object") or {}).get("clearance_min_m")),
                "max_speed": _clean_float(b_row.get("max_speed_mps")),
                "strongest_decel": _clean_float((b_row.get("strong_brake") or {}).get("strongest_mps2")),
                "progress_m": _clean_float(b_row.get("progress_m", 0.0)) or 0.0,
                "n_steps": b_row.get("n_steps_run", 0),
                "video_path": b_vid,
                "colormap_paths": b_cm,
                "rollout_path": b_roll,
            },
            "treat": {
                "passed": t_pass,
                "failure_category": t_cat,
                "failure_category_ja": CATEGORY_LABELS_JA.get(t_cat, t_cat),
                "hazard_verdict": t_hz,
                "hazard_verdict_ja": HAZARD_LABELS_JA.get(t_hz, t_hz),
                "verdict_trigger": t_diag.get("verdict_trigger"),
                "verdict_unmet": t_diag.get("verdict_unmet", []),
                "terminated": t_row.get("terminated", ""),
                "min_clearance": _clean_float((t_row.get("object") or {}).get("clearance_min_m")),
                "max_speed": _clean_float(t_row.get("max_speed_mps")),
                "strongest_decel": _clean_float((t_row.get("strong_brake") or {}).get("strongest_mps2")),
                "progress_m": _clean_float(t_row.get("progress_m", 0.0)) or 0.0,
                "n_steps": t_row.get("n_steps_run", 0),
                "video_path": t_vid,
                "colormap_paths": t_cm,
                "rollout_path": t_roll,
            },
        })

    base_total = len(base_segs)
    treat_total = len(treat_segs)
    base_pass_cnt = base_cats.get("PASS", 0)
    treat_pass_cnt = treat_cats.get("PASS", 0)
    base_pr = (base_pass_cnt / base_total * 100) if base_total else 0.0
    treat_pr = (treat_pass_cnt / treat_total * 100) if treat_total else 0.0

    # Failure breakdown rows
    failure_breakdown_rows = []
    n_common = len(common_scenarios)
    for cat, cat_ja, desc in ALL_FAILURE_CATEGORIES:
        b_c = base_cats.get(cat, 0)
        t_c = treat_cats.get(cat, 0)
        b_pct = (b_c / n_common * 100) if n_common else 0.0
        t_pct = (t_c / n_common * 100) if n_common else 0.0
        failure_breakdown_rows.append({
            "category": cat,
            "category_ja": cat_ja,
            "base_count": b_c,
            "base_pct": b_pct,
            "treat_count": t_c,
            "treat_pct": t_pct,
            "delta": t_c - b_c,
            "description": desc,
        })

    # Hazard breakdown rows
    hazard_breakdown_rows = []
    for hz, hz_ja, interp in ALL_HAZARD_VERDICTS:
        b_h = base_hazards.get(hz, 0)
        t_h = treat_hazards.get(hz, 0)
        hazard_breakdown_rows.append({
            "verdict": hz,
            "verdict_ja": hz_ja,
            "base_count": b_h,
            "treat_count": t_h,
            "delta": t_h - b_h,
            "interpretation": interp,
        })

    # Driving Safety & Comfort Metrics rows
    b_clearances = [(r.get("object") or {}).get("clearance_min_m") for r in base_segs.values()]
    t_clearances = [(r.get("object") or {}).get("clearance_min_m") for r in treat_segs.values()]
    b_accels = [(r.get("strong_brake") or {}).get("strongest_mps2") for r in base_segs.values()]
    t_accels = [(r.get("strong_brake") or {}).get("strongest_mps2") for r in treat_segs.values()]
    b_progress = [r.get("progress_m", 0.0) for r in base_segs.values()]
    t_progress = [r.get("progress_m", 0.0) for r in treat_segs.values()]
    b_steps = [r.get("n_steps_run", 0) for r in base_segs.values()]
    t_steps = [r.get("n_steps_run", 0) for r in treat_segs.values()]

    safety_metrics_rows = [
        {
            "metric": "最小障害物クリアランス",
            "base_stat": _format_stat_summary(b_clearances, " m"),
            "treat_stat": _format_stat_summary(t_clearances, " m"),
            "description": "障害物との最近接距離（大きいほど安全余裕あり）",
        },
        {
            "metric": "最大減速度",
            "base_stat": _format_stat_summary(b_accels, " m/s²", fmt=".2f"),
            "treat_stat": _format_stat_summary(t_accels, " m/s²", fmt=".2f"),
            "description": "急ブレーキの強さ（滑らかさ・乗心地指標）",
        },
        {
            "metric": "シナリオ走行距離 (Progress)",
            "base_stat": _format_stat_summary(b_progress, " m", fmt=".1f"),
            "treat_stat": _format_stat_summary(t_progress, " m", fmt=".1f"),
            "description": "ゴールに向けた総移動距離",
        },
        {
            "metric": "採点走行ステップ数",
            "base_stat": _format_stat_summary(b_steps, " steps", fmt=".0f"),
            "treat_stat": _format_stat_summary(t_steps, " steps", fmt=".0f"),
            "description": "ゴール到達または終了までの走行ステップ数",
        },
    ]

    payload = {
        "metadata": {
            "title_base": title_base,
            "title_treat": title_treat,
            "base_dir": str(base_dir),
            "treat_dir": str(treat_dir),
        },
        "summary": {
            "base_total": base_total,
            "treat_total": treat_total,
            "n_common": n_common,
            "base_pass_cnt": base_pass_cnt,
            "treat_pass_cnt": treat_pass_cnt,
            "base_pr": base_pr,
            "treat_pr": treat_pr,
            "pr_delta": treat_pr - base_pr,
            "n_wins": len(wins),
            "n_losses": len(losses),
            "n_both_pass": len(both_pass),
            "n_both_fail": len(both_fail),
        },
        "failure_breakdown": {
            "base": dict(base_cats),
            "treat": dict(treat_cats),
        },
        "hazard_breakdown": {
            "base": dict(base_hazards),
            "treat": dict(treat_hazards),
        },
        "failure_breakdown_rows": failure_breakdown_rows,
        "hazard_breakdown_rows": hazard_breakdown_rows,
        "safety_metrics_rows": safety_metrics_rows,
        "scenarios": scenario_cards,
    }
    return _sanitize_json_obj(payload)


def build_comparison_html_report(
    base_dir: str | Path,
    treat_dir: str | Path,
    out_path: str | Path,
    *,
    title_base: str = "Baseline",
    title_treat: str = "Treatment",
    title: str = "Closed-Loop Scenario Model Comparison",
    subtitle: str = "",
) -> Path:
    """Build and write a self-contained HTML comparison report."""
    base_dir = Path(base_dir)
    treat_dir = Path(treat_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from scenario_generation.compare_eval_runs import load_run_data
    _, base_segs = load_run_data(base_dir)
    _, treat_segs = load_run_data(treat_dir)

    payload = build_comparison_payload(
        base_dir=base_dir,
        treat_dir=treat_dir,
        base_segs=base_segs,
        treat_segs=treat_segs,
        html_out_dir=out_path.parent,
        title_base=title_base,
        title_treat=title_treat,
    )

    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    html = template.replace("__TITLE__", title)
    html = html.replace("__SUBTITLE__", subtitle)
    html = html.replace("__BASE_DIR__", str(base_dir))
    html = html.replace("__TREAT_DIR__", str(treat_dir))
    html = html.replace("__DATA_JSON__", json.dumps(payload, ensure_ascii=False))

    out_path.write_text(html, encoding="utf-8")
    return out_path
