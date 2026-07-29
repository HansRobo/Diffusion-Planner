#!/usr/bin/env python3
"""Audit the T4 HDP lane-mask proxy against logged lane-change labels.

The formal ``hdp_pdm`` adapter neutralizes its continuous lane score when the
logged expert triggers the repository's lane-departure geometry.  This tool
replays that exact mask on a stratified subset of a ``detect_lane_change``
audit and records where the two independent labels agree or disagree.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "diffusion_planner"))

from planner_metrics.config import RewardConfig
from planner_metrics.subscores import compute_lane_departure_penalty
from preference_optimization.utils import load_npz_data


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane-change-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--max-per-stratum", type=int, default=128)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def _stratum(row: dict) -> str:
    return f"actual_{int(bool(row.get('actual')))}__predicted_{int(bool(row.get('predicted')))}"


def _expert_traj(data: dict[str, torch.Tensor], horizon: int) -> torch.Tensor:
    gt = data["ego_agent_future"]
    if gt.dim() == 3:
        gt = gt[0]
    gt = gt[:horizon]
    if gt.shape[-1] == 3:
        gt = torch.cat(
            [gt[:, :2], torch.cos(gt[:, 2:3]), torch.sin(gt[:, 2:3])], dim=-1
        )
    return gt[:, :4].unsqueeze(0)


def main() -> None:
    args = _args()
    source = json.loads(args.lane_change_audit.read_text())
    eligible = [
        row
        for row in source["rows"]
        if "error" not in row and "actual" in row
    ]
    strata: dict[str, list[dict]] = {}
    for row in eligible:
        strata.setdefault(_stratum(row), []).append(row)

    rng = random.Random(args.seed)
    chosen: list[dict] = []
    for name in sorted(strata):
        rows = list(strata[name])
        rng.shuffle(rows)
        chosen.extend(rows[: args.max_per_stratum])

    config = RewardConfig(
        reward_horizon_steps=args.horizon,
        enable_lane_departure=True,
        lane_gate_enabled=False,
        lane_cross_thresh=0.2,
        lane_near_thresh=0.25,
        lane_wide_thresh=0.4,
        lane_cont_thresh=0.8,
    )
    audited: list[dict] = []
    for index, row in enumerate(chosen):
        record = {
            "scene": row["scene"],
            "actual_lane_change": bool(row["actual"]),
            "route_predicted_lane_change": bool(row["predicted"]),
            "source_stratum": _stratum(row),
        }
        try:
            data = load_npz_data(row["scene"], torch.device("cpu"))
            gt4 = _expert_traj(data, args.horizon)
            ego_shape = data["ego_shape"][0, :3]
            _, _, _, steps, _ = compute_lane_departure_penalty(
                gt4, ego_shape, data, config=config
            )
            record["expert_mask_active"] = steps[0] is not None
            record["expert_mask_first_step"] = steps[0]
        except Exception as exc:  # fail visible in the manifest
            record["error"] = repr(exc)
        audited.append(record)
        if (index + 1) % 25 == 0:
            print(f"{index + 1}/{len(chosen)}")

    valid = [row for row in audited if "error" not in row]
    tp = sum(row["expert_mask_active"] and row["actual_lane_change"] for row in valid)
    fp = sum(row["expert_mask_active"] and not row["actual_lane_change"] for row in valid)
    fn = sum(not row["expert_mask_active"] and row["actual_lane_change"] for row in valid)
    tn = sum(not row["expert_mask_active"] and not row["actual_lane_change"] for row in valid)
    payload = {
        "contract": {
            "purpose": "diagnostic comparison, not a ground-truth claim",
            "formal_reward_horizon_steps": args.horizon,
            "expert_mask": "lane score becomes neutral for the entire scene when logged expert lane-departure step is non-null",
            "lane_gate_enabled": False,
            "source_label": "independent logged-future versus surrounding-lanes detector",
        },
        "source": str(args.lane_change_audit),
        "sampling": {
            "seed": args.seed,
            "max_per_actual_predicted_stratum": args.max_per_stratum,
            "selected": len(chosen),
            "source_stratum_sizes": {key: len(value) for key, value in sorted(strata.items())},
        },
        "summary": {
            "audited": len(audited),
            "valid": len(valid),
            "errors": len(audited) - len(valid),
            "mask_active": tp + fp,
            "actual_lane_change": tp + fn,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
        },
        "rows": audited,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
