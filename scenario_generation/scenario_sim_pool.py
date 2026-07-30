"""A worker that outlives one scenario: load the model once, keep the maps, take work as it comes.

The per-scenario process exists only because ``SimulatorCore`` is a static singleton. Once the
interpreter runs in its own child process (``sim_proc_client``), that constraint no longer binds
the planner, and everything the planner pays per scenario becomes payable per WORKER instead --
45.2 s of fixed cost per case measured at job 1665, against a ``sim_step`` that is 1.7% of worker
time:

    predict_cold  9.0 s   torch.compile + CUDA graph capture, per process
    model_load    8.3 s   the checkpoint, per process
    map_build     7.5 s   a 46 MB lanelet2 map, per scenario -- but 378 of 464 cases share one map
    startup       5.4 s   interpreter + torch import, per process

Work distribution is by claim file rather than a parent->worker protocol. Each worker atomically
creates ``<claim_dir>/<index>`` with O_EXCL and runs that scenario if it won the race. That gives
dynamic balancing (scenario durations vary threefold, so static partitioning would leave a long
tail) without a second IPC channel to design, and it is crash-safe by construction: a worker that
dies holds no lock anyone waits on.
"""

from __future__ import annotations

import faulthandler
import json
import os
import sys
import traceback
from pathlib import Path

import torch

from scenario_generation.gui.lanelet_scene_builder import LaneletSceneBuilder
from scenario_generation.scenario_sim_rollout import (
    RolloutConfig,
    _CloneEncoderOutput,
    map_from_osc,
    run_scenario_sim_rollout,
)


def _claim(claim_dir: Path, index: int) -> bool:
    """True iff this process won the race for scenario ``index``."""
    try:
        # O_EXCL on a shared filesystem is the whole mechanism: exactly one worker can create it.
        fd = os.open(str(claim_dir / f"{index:06d}"), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    os.write(fd, f"{os.getpid()}\n".encode())
    os.close(fd)
    return True


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="persistent scenario_sim worker")
    p.add_argument("--work_list", required=True, help="JSON [[out_dir, osc_path], ...]")
    p.add_argument("--claim_dir", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--replan_interval", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--warmup_steps", type=int, default=5)
    p.add_argument("--near_miss_thresh", type=float, default=1.0)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--watchdog_sec", type=float, default=0.0,
                   help="per-SCENARIO deadline; the dump is re-armed for each one")
    a = p.parse_args(argv)

    faulthandler.enable()
    work = json.loads(Path(a.work_list).read_text())
    claim_dir = Path(a.claim_dir)

    from scenario_generation.simulate import load_model

    model, model_args = load_model(a.model_path, a.device)
    # Same compile setup as the single-scenario worker, so per-tick numbers stay comparable.
    model.decoder.dit = torch.compile(model.decoder.dit, mode="reduce-overhead")
    model.encoder = _CloneEncoderOutput(torch.compile(model.encoder, mode="reduce-overhead"))

    cfg = RolloutConfig(
        fps=a.fps,
        replan_interval=a.replan_interval,
        max_steps=a.max_steps,
        warmup_steps=a.warmup_steps,
        near_miss_thresh=a.near_miss_thresh,
        # The interpreter MUST be out of this process: it is the singleton, and this process is
        # about to run many scenarios.
        sim_in_subprocess=True,
    )

    builders: dict[str, LaneletSceneBuilder] = {}
    done = 0
    for index, (out_dir, osc_path) in enumerate(work):
        if not _claim(claim_dir, index):
            continue
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        # Armed per scenario, not per process: a single deadline for a worker's whole life would
        # fire in the middle of an unrelated scenario.
        if a.watchdog_sec > 0:
            faulthandler.dump_traceback_later(a.watchdog_sec, exit=True)
        try:
            map_path = map_from_osc(osc_path)
            if map_path not in builders:
                builders[map_path] = LaneletSceneBuilder(map_path)
            row = run_scenario_sim_rollout(
                model, model_args, osc_path, map_path, out,
                config=cfg, device=a.device, builder=builders[map_path], verbose=True,
            )
            (out / "row.json").write_text(json.dumps(row, default=float))
            done += 1
        except BaseException as e:  # noqa: BLE001
            # One scenario's failure must not take the worker down: the remaining claims are its
            # to run, and a crash here would strand them until another worker's pass. The parent
            # reads row.json, so its absence is already how a failure is reported.
            print(f"[pool] scenario failed: {osc_path}: {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc()
            (out / "error.txt").write_text(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        finally:
            if a.watchdog_sec > 0:
                faulthandler.cancel_dump_traceback_later()
    print(f"[pool] worker {os.getpid()} finished {done} scenarios")
    return 0


if __name__ == "__main__":
    sys.exit(main())
