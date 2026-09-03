#!/usr/bin/env python3
"""Run the legacy scenario-based open-loop metrics with a new-DP sampler ONNX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from diffusion_planner.scenario_based_open_loop.open_loop import (
    run_scenario_based_open_loop_validation,
)
from scenario_generation.new_dp_onnx import IdentityNormalizer
from scenario_generation.simulate import load_onnx_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--model_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    model, _ = load_onnx_model(args.model_path, args.device)
    metric_args = SimpleNamespace(
        scenario_based_open_loop_list=str(args.matrix),
        scenario_centerline_horizon_seconds=8.0,
        scenario_simple_turn_horizon_seconds=8.0,
        scenario_departure_horizon_seconds=3.0,
        scenario_departure_minimum_displacement_m=2.0,
        scenario_traffic_light_go_horizon_seconds=3.0,
        scenario_traffic_light_go_minimum_displacement_m=2.0,
        scenario_pedestrian_yield_horizon_seconds=3.0,
        scenario_pedestrian_yield_maximum_forward_progress_m=0.5,
        scenario_vehicle_yield_horizon_seconds=3.0,
        scenario_vehicle_yield_maximum_forward_progress_m=0.5,
        scenario_temporal_stop_horizon_seconds=3.0,
        scenario_temporal_stop_maximum_forward_progress_m=0.5,
        scenario_obstacle_stop_tolerance_m=0.5,
        scenario_traffic_light_stop_tolerance_m=0.5,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_mem=False,
        device=args.device,
        predicted_neighbor_num=320,
        future_len=80,
        observation_normalizer=IdentityNormalizer(),
        ddp=False,
    )
    summary = run_scenario_based_open_loop_validation(
        model,
        metric_args,
        visualization_dir=args.output_dir / "visualization",
        details_dir=args.output_dir / "details",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"scenario-based-open-loop summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
