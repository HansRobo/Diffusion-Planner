"""A worker that outlives one scenario: load the model once, keep the parsed maps.

``SimulatorCore`` is a static singleton, but it forbids two scenarios at once, not two in a
lifetime, so one process can drive scenario after scenario.

Work is taken, not handed out: each worker creates ``<claim_dir>/<index>`` with O_EXCL and runs
that scenario if it won. Scenario durations are heavily skewed, so static partitioning would leave
a long tail, and a worker that dies holds no claim anyone waits on.
"""

from __future__ import annotations

import argparse
import contextlib
import faulthandler
import json
import os
import sys
import traceback
from concurrent.futures import BrokenExecutor
from pathlib import Path

from scenario_generation.closed_loop_eval import (
    build_mp4,
    segment_row_for_json,
    tdigest_sidecar_row,
)
from scenario_generation.gui.lanelet_scene_builder import LaneletSceneBuilder
from scenario_generation.render_pool import render_pool
from scenario_generation.scenario_sim_rollout import RolloutConfig, run_scenario_sim_rollout


def _claim(claim_dir: Path, index: int) -> bool:
    """True iff this process won the race for scenario ``index``."""
    path = claim_dir / f"{index:06d}"
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    except FileNotFoundError:
        claim_dir.mkdir(parents=True, exist_ok=True)
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
    p.add_argument("--model_path", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--replan_interval", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--warmup_steps", type=int, default=5)
    p.add_argument("--near_miss_thresh", type=float, default=1.0)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--draw_every", type=int, default=None)
    p.add_argument(
        "--watchdog_sec",
        type=float,
        default=0.0,
        help="per-SCENARIO deadline; the dump is re-armed for each one",
    )
    p.add_argument("--slot", type=int, default=-1, help="this worker's pool slot")
    p.add_argument("--gpu", type=int, default=-1, help="the GPU this slot was assigned")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from scenario_generation.simulate import load_model

    a = _parse_args(argv)
    cfg = RolloutConfig(
        fps=a.fps,
        replan_interval=a.replan_interval,
        max_steps=a.max_steps,
        warmup_steps=a.warmup_steps,
        near_miss_thresh=a.near_miss_thresh,
        draw_every=a.draw_every,
    )
    model, model_args = load_model(a.model_path, a.device)
    work = json.loads(Path(a.work_list).read_text())
    claim_dir = Path(a.claim_dir)
    builders: dict[str, LaneletSceneBuilder] = {}
    consecutive_failures = 0

    # One renderer for this worker's whole life; the rollout would otherwise spawn one per scenario.
    with render_pool(1) if cfg.draw_every else contextlib.nullcontext() as draw:
        for index, (out_dir, osc_path) in enumerate(work):
            if not _claim(claim_dir, index):
                continue
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            # Armed per scenario, not per process: one deadline for a worker's whole life would
            # fire in the middle of an unrelated scenario. A worker that hangs would otherwise
            # take every scenario it has not claimed yet down with it.
            if a.watchdog_sec > 0:
                faulthandler.dump_traceback_later(a.watchdog_sec, exit=True)
            try:
                row = run_scenario_sim_rollout(
                    model,
                    model_args,
                    osc_path,
                    out,
                    config=cfg,
                    device=a.device,
                    builders=builders,
                    draw_pool=draw,
                )
                # Same outputs the per-scenario worker writes: ``route`` is what reattaches a
                # sidecar to its row, and the frames exist only to become the video.
                route = out.name
                side_out = out / "row.tdigests.json"
                side = tdigest_sidecar_row({"route": route, **row})
                if side is not None:
                    side_out.write_text(json.dumps(side, default=float))
                else:
                    # Removed when there is nothing to write, so an earlier run's digests cannot
                    # outlive it.
                    side_out.unlink(missing_ok=True)
                # Written last of the two: the parent reads row.json as the case's receipt, so
                # nothing the row vouches for may still fail after it lands.
                # The per-scenario driver derives which slot and GPU ran a case from the process
                # it spawned; a pooled worker runs many, so it has to say so on each row.
                (out / "row.json").write_text(
                    json.dumps(
                        segment_row_for_json(
                            row,
                            route=route,
                            pool_pid=os.getpid(),
                            slot=a.slot if a.slot >= 0 else None,
                            gpu=a.gpu if a.gpu >= 0 else None,
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
                # reports the failure.
                print(f"[pool] {osc_path}: {type(e).__name__}: {e}", file=sys.stderr)
                traceback.print_exc()
                (out / "error.txt").write_text(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
                # A worker whose own process is broken keeps claiming scenarios and failing them
                # fast, which destroys work a healthy worker would have finished -- so stop
                # claiming. A renderer that died takes the pool with it permanently, and that is
                # visible; a simulator core left instantiated by a throwing teardown is not, since
                # the interpreter swallows it and Python sees only "activate() did not reach
                # 'active'" -- the same thing a genuinely broken scenario reports. So the count is
                # what distinguishes them. Exiting costs parallelism only: whatever this worker
                # has not claimed, another takes.
                consecutive_failures += 1
                if isinstance(e, BrokenExecutor) or consecutive_failures >= 3:
                    print(
                        f"[pool] giving up after {consecutive_failures} consecutive failures"
                        f" ({type(e).__name__})",
                        file=sys.stderr,
                    )
                    break
            finally:
                if a.watchdog_sec > 0:
                    faulthandler.cancel_dump_traceback_later()

    return 0


if __name__ == "__main__":
    sys.exit(main())
