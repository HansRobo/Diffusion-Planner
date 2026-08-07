"""Single-scenario worker: run one scenario_sim rollout and write its metrics row.

The C++ ``SimulatorCore`` is a static singleton, so one process runs exactly one scenario;
isolation and a clean teardown both depend on that. The caller therefore spawns this as a
subprocess per scenario, which is also why the pre-rollout costs (model load, map parse) are
reported in the same timing breakdown as the per-tick sums -- a suite pays them once per
scenario, so a run dominated by startup must be distinguishable from one dominated by
inference.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from scenario_generation.perf_timer import Timers
from scenario_generation.scenario_sim_rollout import RolloutConfig, run_scenario_sim_rollout
from scenario_generation.scenario_sim_route import map_from_osc


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="scenario_sim single-scenario worker")
    p.add_argument("--osc", required=True)
    p.add_argument(
        "--map_path",
        default=None,
        help="lanelet2 .osm; defaults to the scenario's own RoadNetwork/LogicFile, which is "
        "also where the C++ interpreter reads it from -- override only to test a substitute map",
    )
    p.add_argument("--out_dir", required=True)
    p.add_argument("--row_out", required=True, help="write the metrics row JSON here")
    p.add_argument("--device", default="cpu")
    p.add_argument("--model_path", required=True, help="torch .pth checkpoint")
    p.add_argument(
        "--replan_interval",
        type=int,
        default=1,
        help="re-plan every N ticks; 1 (default) = every tick = 10 Hz, matching production",
    )
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--warmup_steps", type=int, default=5)
    p.add_argument("--near_miss_thresh", type=float, default=1.0)
    p.add_argument("--fps", type=float, default=10.0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from scenario_generation.simulate import load_model

    a = _parse_args(argv)
    timers = Timers()
    t_proc = time.perf_counter()
    with timers("model_load"):
        model, model_args = load_model(a.model_path, a.device)

    map_path = a.map_path or map_from_osc(a.osc)
    cfg = RolloutConfig(
        fps=a.fps,
        replan_interval=a.replan_interval,
        max_steps=a.max_steps,
        warmup_steps=a.warmup_steps,
        near_miss_thresh=a.near_miss_thresh,
    )
    row = run_scenario_sim_rollout(
        model,
        model_args,
        a.osc,
        map_path,
        a.out_dir,
        config=cfg,
        device=a.device,
        timers=timers,
    )
    timers.add("worker_process", time.perf_counter() - t_proc)
    row["timing"] = timers.as_dict()
    row["map_path"] = str(map_path)
    Path(a.row_out).write_text(json.dumps(row, default=float))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
