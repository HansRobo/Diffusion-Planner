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
import faulthandler
import json
import time
from pathlib import Path

from scenario_generation.closed_loop_eval import segment_row_for_json, tdigest_sidecar_row
from scenario_generation.perf_timer import Timers
from scenario_generation.scenario_sim_rollout import RolloutConfig, run_scenario_sim_rollout


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
    p.add_argument(
        "--route",
        default=None,
        help="identity carried by the row and its tdigest sidecar; defaults to the scenario "
        "stem. Stems are not unique across a suite (the yaml->xosc converter emits a flat "
        "scenario_<n>.xosc inside each scenario's own directory), and the sidecar is reattached "
        "by this value, so a caller spanning a suite must pass a unique key",
    )
    p.add_argument(
        "--watchdog_sec",
        type=float,
        default=0.0,
        help="dump this process's stacks and exit after N seconds (0 disables). A wedged "
        "interpreter is past responding to signals, so the deadline has to fire from inside",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from scenario_generation.simulate import load_model

    a = _parse_args(argv)
    if a.watchdog_sec > 0:
        faulthandler.enable()
        faulthandler.dump_traceback_later(a.watchdog_sec, exit=True)
    timers = Timers()
    t_proc = time.perf_counter()
    with timers("model_load"):
        model, model_args = load_model(a.model_path, a.device)

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
        a.out_dir,
        map_path=a.map_path,
        config=cfg,
        device=a.device,
        timers=timers,
    )
    timers.add("worker_process", time.perf_counter() - t_proc)

    # Same split the closed-loop eval writer uses: a human-readable row, with the clearance
    # digests in a sidecar so a parent can still pool an approximate global p5. ``route`` is
    # what carries a row's identity through that pair -- ``attach_tdigest_sidecars`` keys the
    # reattach on it, so a sidecar without one can be written but never read back.
    route = a.route or Path(a.osc).stem
    row_out = Path(a.row_out)
    row_out.write_text(
        json.dumps(segment_row_for_json(row, route=route, timing=timers.as_dict()), default=float)
    )
    side = tdigest_sidecar_row({"route": route, **row})
    if side is not None:
        row_out.with_suffix(".tdigests.json").write_text(json.dumps(side, default=float))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
