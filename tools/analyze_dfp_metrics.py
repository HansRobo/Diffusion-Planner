#!/usr/bin/env python3
"""Compare DFP and matched no-DFP training logs.

This is intentionally dependency-free so it can run on node02 or locally without
touching the training environment. It compares lower-is-better validation metrics
from the TSV logs written by Diffusion-Planner training.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


METRICS = (
    "valid_loss_ego",
    "valid_loss_neighbor",
    "valid_loss_ego_position_lat_loss",
    "valid_loss_ego_position_lon_loss",
)


@dataclass(frozen=True)
class MetricPoint:
    epoch: int
    value: float


def read_log(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def best(rows: list[dict[str, str]], metric: str) -> MetricPoint:
    candidates = [
        MetricPoint(epoch=int(row["epoch"]), value=float(row[metric]))
        for row in rows
        if row.get(metric, "") != ""
    ]
    if not candidates:
        raise ValueError(f"No values for {metric}")
    return min(candidates, key=lambda x: x.value)


def by_epoch(rows: list[dict[str, str]], metric: str, epoch: int) -> MetricPoint | None:
    for row in rows:
        if row.get("epoch") == str(epoch) and row.get(metric, "") != "":
            return MetricPoint(epoch=epoch, value=float(row[metric]))
    return None


def pct_improvement(dfp: float, baseline: float) -> float:
    return 100.0 * (baseline - dfp) / baseline


def verdict(
    best_delta: dict[str, float],
    epoch_delta: dict[str, float],
    neighbor_regression_limit_pct: float,
) -> str:
    ego_win = best_delta["valid_loss_ego"] > 0.0
    lon_win = best_delta["valid_loss_ego_position_lon_loss"] > 0.0
    lat_not_bad = best_delta["valid_loss_ego_position_lat_loss"] > -5.0
    neighbor_ok = best_delta["valid_loss_neighbor"] >= -neighbor_regression_limit_pct
    epoch20_ego_win = epoch_delta.get("valid_loss_ego", -999.0) > 0.0

    if ego_win and lon_win and lat_not_bad and neighbor_ok and epoch20_ego_win:
        return "confirmed_strong"
    if ego_win and lon_win and neighbor_ok:
        return "confirmed_by_best_epoch"
    if ego_win or lon_win:
        return "promising_but_not_confirmed"
    return "not_confirmed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dfp-log", type=Path, required=True)
    parser.add_argument("--baseline-log", type=Path, required=True)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--neighbor-regression-limit-pct", type=float, default=5.0)
    args = parser.parse_args()

    dfp_rows = read_log(args.dfp_log)
    baseline_rows = read_log(args.baseline_log)

    print("# DFP vs matched no-DFP metric comparison")
    print()
    print(f"- DFP log: `{args.dfp_log}`")
    print(f"- Baseline log: `{args.baseline_log}`")
    print(f"- Fixed epoch: `{args.epoch}`")
    print()

    print("## Best epoch comparison")
    print()
    print("| metric | dfp best | dfp epoch | baseline best | baseline epoch | improvement pct |")
    print("|---|---:|---:|---:|---:|---:|")
    best_delta: dict[str, float] = {}
    for metric in METRICS:
        d = best(dfp_rows, metric)
        b = best(baseline_rows, metric)
        improvement = pct_improvement(d.value, b.value)
        best_delta[metric] = improvement
        print(
            f"| {metric} | {d.value:.10f} | {d.epoch} | "
            f"{b.value:.10f} | {b.epoch} | {improvement:.3f}% |"
        )

    print()
    print(f"## Epoch {args.epoch} comparison")
    print()
    print("| metric | dfp | baseline | improvement pct |")
    print("|---|---:|---:|---:|")
    epoch_delta: dict[str, float] = {}
    for metric in METRICS:
        d = by_epoch(dfp_rows, metric, args.epoch)
        b = by_epoch(baseline_rows, metric, args.epoch)
        if d is None or b is None:
            print(f"| {metric} | missing | missing | missing |")
            continue
        improvement = pct_improvement(d.value, b.value)
        epoch_delta[metric] = improvement
        print(f"| {metric} | {d.value:.10f} | {b.value:.10f} | {improvement:.3f}% |")

    print()
    print("## Verdict")
    print()
    print(
        verdict(
            best_delta=best_delta,
            epoch_delta=epoch_delta,
            neighbor_regression_limit_pct=args.neighbor_regression_limit_pct,
        )
    )


if __name__ == "__main__":
    main()
