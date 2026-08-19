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

import contextlib
import faulthandler
import json
import os
import sys
import time
import traceback
from pathlib import Path

from scenario_generation.closed_loop_ddp import claim
from scenario_generation.gui.lanelet_scene_builder import LaneletSceneBuilder
from scenario_generation.inference_compile import canonical_attention, compiled_for_inference
from scenario_generation.scenario_sim_rollout import RolloutConfig, run_scenario_sim_rollout
from scenario_generation.scenario_sim_route import map_from_osc


def _drain(
    a, work: list, claim_dir: Path, model, model_args, cfg: RolloutConfig, prof: dict
) -> int:
    """Run every scenario this worker wins a claim for; return how many it completed.

    Separate from ``main`` so the whole drain sits inside one compile scope: compiling is
    per-process work, and re-entering it per scenario would pay it back every time.
    """
    builders: dict[str, LaneletSceneBuilder] = {}
    done = 0
    for index, (out_dir, osc_path) in enumerate(work):
        if not claim(claim_dir, index):
            continue
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        # Armed per scenario, not per process: a single deadline for a worker's whole life would
        # fire in the middle of an unrelated scenario.
        if a.watchdog_sec > 0:
            faulthandler.dump_traceback_later(a.watchdog_sec, exit=True)
        _t_case = time.perf_counter()
        try:
            map_path = map_from_osc(osc_path)
            if map_path not in builders:
                _t = time.perf_counter()
                builders[map_path] = LaneletSceneBuilder(map_path)
                # Per MAP, not per scenario: scenarios sharing a map share this parse. The
                # road-border polylines the metrics path needs come off this same parsed map
                # (border_segments_from_map, called from the rollout's finalize), so there is no
                # second parse to warm here.
                prof["map_parse_s"][map_path] = time.perf_counter() - _t
            row = run_scenario_sim_rollout(
                model,
                model_args,
                osc_path,
                out,
                map_path=map_path,
                config=cfg,
                device=a.device,
                builder=builders[map_path],
                verbose=True,
                scene_asset_dir=out.parent / "scene_maps",
            )
            # The parent needs a per-scenario wall to compute achieved concurrency; in the
            # per-scenario-process design it measured that itself, and it cannot here.
            row["worker_wall_s"] = round(time.perf_counter() - _t_case, 3)
            # The same stamps the per-scenario-process path put on its rows. Without them the
            # summary cannot report which maps were used or whether the GPUs were loaded evenly --
            # and per-GPU balance is exactly what a run spread over 8 devices needs to show.
            row["map_path"] = str(map_path)
            row["slot"] = a.slot if a.slot >= 0 else None
            row["gpu"] = a.gpu if a.gpu >= 0 else None
            row["pool_pid"] = os.getpid()
            row["pool_seq"] = done  # 0 = this worker's first scenario, which absorbs the warmups
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
            prof["scenarios"].append(
                {"index": index, "osc": osc_path, "wall_s": round(time.perf_counter() - _t_case, 3)}
            )
    return done


def _run(a, work: list, claim_dir: Path, prof: dict) -> int:
    """Load the model once, hold it compiled for the worker's whole life, and drain the list."""
    from scenario_generation.simulate import load_model

    _t = time.perf_counter()
    model, model_args = load_model(a.model_path, a.device)
    prof["model_load_s"] = time.perf_counter() - _t

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

    _t = time.perf_counter()
    with contextlib.ExitStack() as stack:
        if a.compile_backend:
            # Both switches, in this order: a compiled model cannot take the fused MHA fastpath,
            # so the eager arithmetic has to be given up too or the two are not comparable. The
            # flag is not in dynamo's guard set, so it must hold before the first forward.
            stack.enter_context(canonical_attention())
            stack.enter_context(compiled_for_inference(model, backend=a.compile_backend))
        prof["compile_setup_s"] = time.perf_counter() - _t
        return _drain(a, work, claim_dir, model, model_args, cfg, prof)


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
    p.add_argument(
        "--compile_backend",
        default="cudagraphs",
        help="torch.compile backend for the model forward; empty disables compilation. "
        "'cudagraphs' is the backend measured bit-exact against an eager baseline run under "
        "the same attention setting",
    )
    p.add_argument("--watchdog_sec", type=float, default=0.0,
                   help="per-SCENARIO deadline; the dump is re-armed for each one")
    p.add_argument("--slot", type=int, default=-1, help="this worker's pool slot")
    p.add_argument("--gpu", type=int, default=-1, help="the GPU this slot was assigned")
    a = p.parse_args(argv)

    faulthandler.enable()
    work = json.loads(Path(a.work_list).read_text())
    claim_dir = Path(a.claim_dir)

    # Costs recorded at WORKER level, not folded into whichever scenario happened to be first.
    # The stage timers live inside the rollout, so with a pooled worker they attribute a
    # once-per-worker cost (the checkpoint, the compile, a map parse) to one scenario and report
    # ~0 for the rest -- true per call, but useless for telling a per-job cost from a per-scenario
    # one. That separation is the whole point of this file's profile output.
    prof: dict = {"pid": os.getpid(), "map_parse_s": {}, "scenarios": []}
    _t_proc = time.perf_counter()
    done = _run(a, work, claim_dir, prof)
    prof["worker_total_s"] = round(time.perf_counter() - _t_proc, 3)
    prof["n_scenarios"] = done
    (claim_dir.parent / f"profile-{os.getpid()}.json").write_text(json.dumps(prof, indent=1))
    print(f"[pool] worker {os.getpid()} finished {done} scenarios in {prof['worker_total_s']:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
