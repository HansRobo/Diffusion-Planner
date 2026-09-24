"""A worker that outlives one scenario: load the model once, keep the parsed maps.

``SimulatorCore`` is a static singleton, but it forbids two scenarios at once, not two in a
lifetime, so one process can drive scenario after scenario. What that saves is everything the
per-scenario worker pays per scenario and this one pays per worker: the CUDA context, the
checkpoint or the ONNX session, the parsed maps, and the renderer.

Work is taken, not handed out: each worker creates ``<claim_dir>/<index>`` with O_EXCL and runs
that scenario if it won. Scenario durations are heavily skewed, so static partitioning would
leave a long tail, and a worker that dies holds no claim anyone waits on.

Every file this writes is one the per-scenario arm writes too, with the same name and the same
contents, so a run driven either way aggregates and exports identically: ``row.json`` and its
digest sidecar per case, and the run-level lines the shell around a per-scenario process appends
on its behalf -- the case's wall in ``slot_log.tsv``, a refused scenario in ``rejected.txt``, a
failed one in ``failures.txt``.
"""

from __future__ import annotations

import argparse
import contextlib
import faulthandler
import json
import os
import sys
import time
import traceback
from concurrent.futures import BrokenExecutor
from pathlib import Path

from scenario_generation.closed_loop_eval import (
    build_mp4,
    segment_row_for_json,
    tdigest_sidecar_row,
)
from scenario_generation.gui.lanelet_scene_builder import LaneletSceneBuilder
from scenario_generation.perf_timer import Timers
from scenario_generation.render_pool import render_pool
from scenario_generation.scenario_sim_rollout import (
    RolloutConfig,
    ScenarioRejected,
    run_scenario_sim_rollout,
)

REJECTED_EXIT = 3
"""What a refused scenario reports, so a parent counts it apart from one that ran and failed."""

CONSECUTIVE_FAILURE_LIMIT = 3
"""Failures in a row after which a worker stops claiming. See the retirement note in ``_run``."""


def claim(claim_dir: Path, index: int) -> bool:
    """True iff this process won the race for case ``index``.

    ``O_EXCL`` is the whole mechanism: exactly one creator succeeds, so no second channel has to
    exist for a parent to hand work out, and it is the same mechanism -- keyed the same way on
    the case's position in the list -- that the per-scenario driver's ``noclobber`` redirect
    uses. It is crash-safe by construction: a worker that dies holds no lock anyone waits on,
    and the flip side is that its unfinished case stays claimed, so a crash costs that case
    rather than the run.
    """
    path = Path(claim_dir) / f"{index:06d}"
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    except FileNotFoundError:
        # Created only when it is actually missing: every worker is offered every case, so a
        # mkdir on the steady path would be one wasted metadata round trip per worker per case.
        Path(claim_dir).mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return False
    os.write(fd, f"{os.getpid()}\n".encode())
    os.close(fd)
    return True


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="persistent scenario_sim worker")
    p.add_argument("--work_list", required=True, help="JSON [[out_dir, osc_path], ...]")
    p.add_argument("--claim_dir", required=True)
    p.add_argument("--run_dir", required=True, help="where the run-level records are appended")
    p.add_argument(
        "--model_path",
        required=True,
        help="torch .pth checkpoint, or an exported .onnx graph -- the suffix picks the loader",
    )
    p.add_argument("--device", default="cuda")
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
    p.add_argument("--draw_every", type=int, default=None)
    p.add_argument(
        "--watchdog_sec",
        type=float,
        default=0.0,
        help="per-SCENARIO deadline; the dump is re-armed for each one. This is what the "
        "per-scenario arm gets from the timeout(1) around each process",
    )
    p.add_argument("--slot", type=int, default=-1, help="this worker's pool slot")
    p.add_argument("--gpu", type=int, default=-1, help="the GPU this slot was assigned")
    p.add_argument(
        "--row_pool_fields",
        action="store_true",
        help="also record pool_pid / slot / gpu on each row. Off by default: the per-scenario "
        "arm's rows have no such keys, and a comparison reads both",
    )
    return p.parse_args(argv)


def _append(path: Path, line: str) -> None:
    """One short line, appended. Every worker writes to the same file, as the shell driver does."""
    with open(path, "a") as f:
        f.write(line if line.endswith("\n") else line + "\n")


def _run(a: argparse.Namespace) -> int:
    from scenario_generation.simulate import load_model, load_onnx_model

    cfg = RolloutConfig(
        fps=a.fps,
        replan_interval=a.replan_interval,
        max_steps=a.max_steps,
        warmup_steps=a.warmup_steps,
        near_miss_thresh=a.near_miss_thresh,
        draw_every=a.draw_every,
    )
    # Timed and reported on the one case that paid it, not spread over the cases that did not:
    # what a pool changes is how often this is paid, so a row that claimed a share of it would
    # hide exactly the effect being measured.
    loader = load_onnx_model if str(a.model_path).endswith(".onnx") else load_model
    _t = time.perf_counter()
    model, model_args = loader(a.model_path, a.device)
    model_load_s: float | None = time.perf_counter() - _t

    work = json.loads(Path(a.work_list).read_text())
    claim_dir = Path(a.claim_dir)
    run_dir = Path(a.run_dir)
    # One parsed map serves every scenario that declares it. Which map a scenario runs is known
    # only after the interpreter activates it, so the rollout selects from this rather than
    # being handed one builder.
    builders: dict[str, LaneletSceneBuilder] = {}
    consecutive_failures = 0

    # One renderer for this worker's whole life; the rollout would otherwise spawn one per
    # scenario, and holds its resident memory for as long as the worker lives.
    with render_pool(1) if cfg.draw_every else contextlib.nullcontext() as draw:
        for index, (out_dir, osc_path) in enumerate(work):
            if not claim(claim_dir, index):
                continue
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            route = out.name
            timers = Timers()
            if model_load_s is not None:
                timers.add("model_load", model_load_s)
                model_load_s = None
            t_case = time.perf_counter()
            # Armed per scenario, not per process: one deadline for a worker's whole life would
            # fire in the middle of an unrelated scenario, and a worker that hangs would take
            # every scenario it has not claimed yet down with it.
            if a.watchdog_sec > 0:
                faulthandler.dump_traceback_later(a.watchdog_sec, exit=True)
            rc = 0
            rejected = False
            renderer_lost = False
            try:
                row = run_scenario_sim_rollout(
                    model,
                    model_args,
                    osc_path,
                    out,
                    config=cfg,
                    device=a.device,
                    timers=timers,
                    builders=builders,
                    draw_pool=draw,
                )
                timers.add("worker_process", time.perf_counter() - t_case)
                # Same two files the per-scenario worker writes, in the same order: ``route`` is
                # what reattaches a sidecar to its row, and the row is the case's receipt, so
                # nothing it vouches for may still fail after it lands.
                side_out = out / "row.tdigests.json"
                side = tdigest_sidecar_row({"route": route, **row})
                if side is not None:
                    side_out.write_text(json.dumps(side, default=float))
                else:
                    # Removed when there is nothing to write, so an earlier run's digests cannot
                    # outlive it.
                    side_out.unlink(missing_ok=True)
                extra = (
                    {"pool_pid": os.getpid(), "slot": a.slot, "gpu": a.gpu}
                    if a.row_pool_fields
                    else {}
                )
                (out / "row.json").write_text(
                    json.dumps(
                        segment_row_for_json(
                            row, route=route, timing=timers.as_dict(), **extra
                        ),
                        default=float,
                    )
                )
                # Encoded after the row and reported apart from it: an unhappy ffmpeg costs the
                # video, not the metrics, and must not read as a failed scenario. ffmpeg's glob
                # errors on a directory with no match.
                try:
                    if any(out.glob("*.png")):
                        build_mp4(out, out / f"{route}.mp4", a.fps, remove_pngs=True)
                except Exception as e:  # noqa: BLE001
                    (out / "mp4_error.txt").write_text(f"{type(e).__name__}: {e}\n")
                consecutive_failures = 0
            except Exception as e:  # noqa: BLE001
                # One scenario's failure must not end the worker: the scenarios it has not
                # claimed yet are still its to take. The parent reads row.json, so its absence
                # reports the failure, and the bucket matches what the per-scenario arm records
                # for the same scenario -- there, an exception is an exit code the shell files
                # under failures.txt whatever raised it.
                print(f"[pool] {osc_path}: {type(e).__name__}: {e}", file=sys.stderr)
                traceback.print_exc()
                (out / "error.txt").write_text(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
                rejected = isinstance(e, ScenarioRejected)
                if rejected:
                    _append(run_dir / "rejected.txt", route)
                    rc = REJECTED_EXIT
                else:
                    _append(run_dir / "failures.txt", f"{route} rc=1")
                    rc = 1
                # A worker whose own process is broken keeps claiming scenarios and failing them
                # fast, which destroys work a healthy worker would have finished -- so stop
                # claiming. A renderer that died takes the pool with it permanently, and that is
                # visible; a simulator core left instantiated by a throwing teardown is not,
                # since Python sees only "activate() did not reach 'active'" -- the same thing a
                # genuinely broken scenario reports. So the count is what distinguishes them.
                # Exiting costs parallelism only: whatever this worker has not claimed, another
                # takes.
                #
                # A scenario the interpreter refused at configure time is not evidence about
                # this process: configure runs before the core is built. Those cluster in the
                # work list -- a broken scenario's variants sit at consecutive indices, and one
                # worker claims the run of them -- so counting them would retire healthy workers.
                if not rejected:
                    consecutive_failures += 1
                    # A dead renderer never comes back, so there is nothing to count towards.
                    renderer_lost = isinstance(e, BrokenExecutor)
            finally:
                if a.watchdog_sec > 0:
                    faulthandler.cancel_dump_traceback_later()
                # The case's own wall, in the file the per-scenario driver writes from the
                # shell: a pool has no shell around a case, and the parent's reports read this.
                _append(
                    run_dir / "slot_log.tsv",
                    f"{route}\t{time.perf_counter() - t_case:.0f}\t{a.slot}\t{a.gpu}\t{rc}",
                )
            if rejected:
                continue
            if rc and (renderer_lost or consecutive_failures >= CONSECUTIVE_FAILURE_LIMIT):
                print(
                    f"[pool] giving up after {consecutive_failures} consecutive failures"
                    f"{' (renderer lost)' if renderer_lost else ''}",
                    file=sys.stderr,
                )
                break
    return 0


def main(argv: list[str] | None = None) -> int:
    return _run(_parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
