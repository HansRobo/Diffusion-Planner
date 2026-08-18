"""A stand-in for ``openscenario_python.HeadlessRunner`` that lives in another process.

``RemoteRunner`` is duck-type compatible with the real runner for the calls the rollout makes, so
the tick loop does not change at all -- which is the point. The loop is where fidelity lives; a
refactor that rewrote it to talk to a socket would put every metric back in question, whereas
swapping the object it calls leaves the arithmetic untouched and makes bit-identity checkable.

See ``sim_proc_server`` for why the simulator is the process that gets split off.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from multiprocessing.connection import Connection
from pathlib import Path

_SERVER = Path(__file__).with_name("sim_proc_server.py")

# How long to wait for the child to finish constructing the interpreter. Generous: configure()
# parses the scenario and loads a 46 MB lanelet2 map, which measured 6.5 s per case under 32-way
# concurrency and can be slower on a loaded node.
_READY_TIMEOUT_S = 300.0
# How long to wait for a single call. The rollout's own per-worker deadline is the real backstop;
# this exists so a wedged child surfaces as an error here rather than as a parent blocked forever
# on recv(). Sized well above the slowest observed call (configure, above).
_CALL_TIMEOUT_S = 600.0
_EXIT_GRACE_S = 30.0


class RemoteSimError(RuntimeError):
    """A call failed inside the simulator process, or the process went away."""


class RemoteRunner:
    """Owns a child process holding the real runner; forwards the whitelisted calls to it."""

    def __init__(self, osc_path: str, output_directory: str, local_frame_rate: float):
        self._osc_path = str(osc_path)
        self._output_directory = str(output_directory)
        self._fps = float(local_frame_rate)
        self._proc: subprocess.Popen | None = None
        self._log = None
        self._log_path: Path | None = None
        self._conn: Connection | None = None
        self._sock: socket.socket | None = None

    # -- lifecycle ---------------------------------------------------------------------
    def __enter__(self) -> RemoteRunner:
        parent, child = socket.socketpair()
        # Loaded by file path, not as scenario_generation.sim_proc_server: importing the package
        # would pull torch into a process that never touches a GPU (+0.8 s measured).
        boot = (
            "import importlib.util as u,sys;"
            f"s=u.spec_from_file_location('sim_proc_server', {str(_SERVER)!r});"
            "m=u.module_from_spec(s);s.loader.exec_module(m);"
            "sys.exit(m.main(sys.argv[1:]))"
        )
        argv = [
            sys.executable, "-c", boot,
            str(child.fileno()), self._osc_path, self._output_directory, str(self._fps),
        ]
        # The child gets its own log. Inheriting the worker's stdout loses the distinction between
        # "the simulator said something" and "the worker did", and a child that dies silently is
        # then indistinguishable from one that never spoke -- which is exactly what happened on
        # the one transient death in job 1689: a BrokenPipe in the parent and no child output at all.
        self._log_path = Path(self._output_directory).parent / "sim_child.log"
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = open(self._log_path, "ab", buffering=0)  # unbuffered: a death must not eat it
        except OSError:
            self._log = None
        try:
            self._proc = subprocess.Popen(
                argv, pass_fds=(child.fileno(),), env=os.environ.copy(),
                stdout=self._log, stderr=subprocess.STDOUT,
            )
        finally:
            child.close()  # the child owns its end now; keeping it open would hide its exit
        self._sock = parent
        self._conn = Connection(parent.detach())

        if not self._conn.poll(_READY_TIMEOUT_S):
            self._shutdown()
            raise RemoteSimError(
                f"simulator process did not become ready within {_READY_TIMEOUT_S:.0f}s "
                f"(rc={self._proc.poll()}): {self._osc_path}"
            )
        kind, payload, *rest = self._conn.recv()
        if kind != "ready":
            self._shutdown()
            raise RemoteSimError(f"simulator process failed to start: {payload}\n{''.join(rest)}")
        return self

    def __exit__(self, *_exc) -> None:
        self._shutdown()

    def _shutdown(self) -> None:
        # Ask first: a clean exit lets the interpreter's own teardown run (deactivate, despawn,
        # pluginlib unload). Then escalate -- a worker deadlocked in exactly that teardown, with
        # SIGTERM ignored, is what the per-worker deadline exists for (plan/11 9m).
        if self._conn is not None:
            try:
                self._conn.send(None)
            except (OSError, BrokenPipeError):
                pass
        if self._proc is not None:
            try:
                self._proc.wait(timeout=_EXIT_GRACE_S)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                try:
                    self._proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
        for closer in (self._conn, self._sock, self._log):
            try:
                if closer is not None:
                    closer.close()
            except OSError:
                pass
        self._conn = self._sock = self._log = None

    # -- the forwarded surface ---------------------------------------------------------
    def _died(self, name: str, why: str) -> RemoteSimError:
        """Describe a dead child in terms someone can act on: signal or exit code, plus its log."""
        rc = None
        if self._proc is not None:
            try:
                rc = self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                rc = self._proc.poll()
        how = "still running" if rc is None else (
            f"killed by signal {-rc}" if rc < 0 else f"exited rc={rc}")
        tail = ""
        if self._log_path is not None and self._log_path.exists():
            lines = self._log_path.read_text(errors="replace").splitlines()[-6:]
            tail = ("\n  child log tail:\n    " + "\n    ".join(lines)) if lines else \
                   f"\n  child log is empty: {self._log_path}"
        return RemoteSimError(f"simulator process died during {name}() ({why}); {how}{tail}")

    def _call(self, name: str, *args, **kwargs):
        if self._conn is None:
            raise RemoteSimError(f"simulator process is gone; cannot call {name}()")
        try:
            self._conn.send((name, args, kwargs))
        except (OSError, BrokenPipeError) as e:
            raise self._died(name, repr(e)) from e
        if not self._conn.poll(_CALL_TIMEOUT_S):
            rc = self._proc.poll() if self._proc else None
            raise RemoteSimError(
                f"{name}() did not answer within {_CALL_TIMEOUT_S:.0f}s (child rc={rc})"
            )
        try:
            kind, payload, *rest = self._conn.recv()
        except EOFError as e:
            raise self._died(name, "EOF on the connection") from e
        if kind != "ok":
            raise RemoteSimError(f"{name}() failed in the simulator process: {payload}\n"
                                 f"{''.join(rest)}")
        return payload

    def __getattr__(self, name: str):
        # Only the runner's own methods; anything private is this class's business and a missing
        # attribute must still look missing rather than becoming a silent RPC.
        if name.startswith("_"):
            raise AttributeError(name)
        from scenario_generation.sim_proc_server import ALLOWED

        if name not in ALLOWED:
            raise AttributeError(f"{name} is not forwarded to the simulator process")

        def bound(*args, **kwargs):
            return self._call(name, *args, **kwargs)

        return bound
