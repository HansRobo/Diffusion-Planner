"""Run the OpenSCENARIO interpreter in its own process and forward method calls to it.

Why this exists: ``SimulatorCore`` is a static singleton, so one process can drive exactly one
scenario. Today that constraint is satisfied by making the WHOLE worker per-scenario -- model,
CUDA context, compiled graphs and parsed map included -- which costs 45 s per case (measured, job
1665: predict_cold 9.0 s, model_load 8.3 s, map_build 7.5 s, interpreter startup 5.4 s, ...)
against a ``sim_step`` that is 1.20 ms, i.e. 1.7% of worker time. The singleton is on the
simulator side only, so the split belongs there: the simulator is the short-lived process and
everything expensive stays put.

The cost of the split is one round trip per tick. Measured at 20.2 us with the real payload
(states 852 B, plan 2712 B) against ~29 ms of real work per tick -- 0.07%.

This module is deliberately dependency-free apart from ``openscenario_python`` and the standard
library, and is loaded by file path rather than as ``scenario_generation.sim_proc_server``:
importing the package would drag in torch (measured +0.8 s) into a process that will never touch
a GPU.
"""

from __future__ import annotations

import sys
import traceback
from multiprocessing.connection import Connection

# Methods the rollout is allowed to call on the runner. A whitelist rather than free-form getattr:
# this is a fork()/exec() boundary carrying pickles, and an RPC surface that can reach any
# attribute is a much larger thing to reason about than the eight calls the loop actually makes.
ALLOWED = frozenset(
    {
        "configure",
        "activate",
        "deactivate",
        "state",
        "step",
        "result_kind",
        "get_entity_states",
        "get_ego_state",
        "set_ego_trajectory",
        "set_ego_turn_indicator",
    }
)


def serve(fd: int, osc_path: str, output_directory: str, local_frame_rate: float) -> int:
    """Own a HeadlessRunner and answer ``(method, args, kwargs)`` requests on ``fd``.

    Replies are ``("ok", value)`` or ``("err", repr, traceback)``. Errors are sent back rather
    than raised locally so the parent -- which owns the metrics row -- decides what a failure
    means; a crash here would otherwise look identical to a hang.
    """
    import openscenario_python as osp  # noqa: PLC0415  (must not be imported by the parent)

    conn = Connection(fd)
    try:
        with osp.HeadlessRunner(
            osc_path=osc_path,
            output_directory=output_directory,
            local_frame_rate=local_frame_rate,
        ) as runner:
            conn.send(("ready", None))
            while True:
                msg = conn.recv()
                if msg is None:  # shutdown: let the context manager tear the runner down
                    break
                name, args, kwargs = msg
                if name not in ALLOWED:
                    conn.send(("err", f"method not allowed: {name!r}", ""))
                    continue
                try:
                    conn.send(("ok", getattr(runner, name)(*args, **kwargs)))
                except BaseException as e:  # noqa: BLE001 - report anything, including aborts
                    conn.send(("err", f"{type(e).__name__}: {e}", traceback.format_exc()))
    except BaseException as e:  # noqa: BLE001
        # Construction or teardown failed. The parent may already be waiting on a reply.
        try:
            conn.send(("err", f"{type(e).__name__}: {e}", traceback.format_exc()))
        except OSError:
            pass
        print(f"[sim_proc_server] fatal: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1
    finally:
        try:
            conn.close()
        except OSError:
            pass
    return 0


def main(argv: list[str]) -> int:
    fd, osc_path, output_directory, fps = int(argv[0]), argv[1], argv[2], float(argv[3])
    return serve(fd, osc_path, output_directory, fps)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
