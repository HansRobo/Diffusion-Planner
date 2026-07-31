"""Launch specs: one at a time, on GPUs that are actually free."""

from __future__ import annotations

import glob
import json
import os
import subprocess
import time
from dataclasses import replace
from pathlib import Path

from rlvr.posttrain.flags import _epoch_number, _last_epoch, command
from rlvr.posttrain.preflight import report
from rlvr.posttrain.spec import ROOT, Spec, absolute


def busy_gpu_processes() -> int:
    """Compute processes on the local GPUs; -1 when nvidia-smi cannot be read."""

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return -1
    if out.returncode != 0:
        return -1
    return len([line for line in out.stdout.splitlines() if line.strip()])


def wait_for_gpus(
    empty_streak: int = 3, poll_seconds: int = 60, max_wait_minutes: int = 480
) -> bool:
    """Block until the GPUs read empty ``empty_streak`` times in a row.

    A co-tenant that is between phases briefly shows zero processes, and
    starting there costs both runs; the streak is what makes the check honest.
    """

    deadline = time.monotonic() + max_wait_minutes * 60
    streak = 0
    while time.monotonic() < deadline:
        count = busy_gpu_processes()
        streak = streak + 1 if count == 0 else 0
        if streak >= empty_streak:
            return True
        time.sleep(poll_seconds)
    return False


def prepare(spec: Spec, log_dir: Path, env: dict[str, str]) -> int:
    """Run the arm's pre-steps, skipping any whose ``creates`` is already there."""

    for index, step in enumerate(spec.prepare):
        creates = absolute(step["creates"]) if step.get("creates") else None
        if creates and Path(creates).exists():
            print(f"{spec.name}: prepare {index} already built {creates}")
            continue
        log_path = log_dir / f"prepare_{index}.log"
        print(f"{spec.name}: prepare {index} -> {log_path}")
        with log_path.open("ab") as log:
            code = subprocess.run(
                step["argv"], stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT
            ).returncode
        if code:
            print(f"{spec.name}: prepare {index} exit {code}, see {log_path}")
            return code
        if creates and not Path(creates).exists():
            print(f"{spec.name}: prepare {index} did not produce {creates}")
            return 1
    return 0


def run(
    spec: Spec,
    *,
    log_root: Path = Path("outputs/posttrain"),
    dry_run: bool = False,
    wait: bool = True,
    detach: bool = False,
) -> int:
    try:
        argv = command(spec)
    except ValueError as error:
        # An invalid or not-yet-resolvable spec reads like any other preflight
        # failure; a traceback here would take the supervisor down with it.
        print(f"{spec.name}: {error}")
        return 78
    if dry_run:
        for step in spec.prepare:
            print("prepare: " + " ".join(step["argv"]))
        print(" ".join(argv))
        return 0

    log_dir = Path(log_root) / spec.name
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "spec.json").write_text(json.dumps(spec.to_dict(), indent=2) + "\n")
    (log_dir / "command.txt").write_text(" ".join(argv) + "\n")

    env = dict(os.environ)
    env.update(spec.env)
    # Pre-steps first: they build the artifacts preflight is about to check.
    if prepare(spec, log_dir, env):
        return 77
    if not report(spec):
        return 78
    if wait and not wait_for_gpus():
        print(f"{spec.name}: GPUs still busy, giving up")
        return 75

    log_path = log_dir / "train.log"
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            argv,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=ROOT,
            start_new_session=detach,
        )
        print(f"{spec.name}: pid {process.pid} -> {log_path}")
        if detach:
            return 0
        return process.wait()


def completed_epoch(output_dir: Path) -> int:
    """The highest epoch this arm has actually written, 0 if it has written none."""

    try:
        return _epoch_number(_last_epoch(str(output_dir))) or 0
    except (ValueError, OSError):
        return 0


def resume_spec(spec: Spec) -> Spec:
    """The same arm, pointed at its own newest checkpoint instead of its origin.

    An arm's first attempt starts from whatever produced it -- a supervised base,
    a mine -- and every attempt after that has to continue itself, which is a
    different checkpoint, a different start epoch, and the two resume switches
    that a mid-run checkpoint can satisfy and a fresh base cannot.
    """

    train = dict(spec.train)
    train["model_path"] = {"latest_epoch": train["output_dir"]}
    train["start_epoch"] = {"resume_after_latest_epoch": train["output_dir"]}
    train["use_policy_state"] = True
    train["resume_optimizer_state"] = True
    return replace(spec, train=train)


def run_until_complete(
    spec: Spec, *, max_stalls: int = 3, retry_seconds: int = 60, **kwargs
) -> int:
    """Relaunch an arm until it reaches its target epoch.

    Long post-training runs die for reasons that have nothing to do with the
    method -- a preempted node, an unattended upgrade restarting the scheduler, a
    transient CUDA fault -- and every one of those is a resume, not a decision.
    What is *not* a resume is a run that comes back and writes no new epoch: that
    is a real failure repeating itself, so identical stalls are counted and the
    supervisor gives up rather than burning the GPUs in a crash loop.

    An attempt that stalls before the trainer starts, though, is usually a race
    rather than a verdict -- an artifact from the arm ahead still landing, a node
    whose scheduler has not come back yet -- and retrying inside the same second
    spends the whole allowance before any of that could have resolved.  So a
    stalled attempt waits, and waits longer each time.
    """

    if kwargs.get("detach"):
        # Detach the supervisor itself; it cannot supervise what it does not wait for.
        print(f"{spec.name}: --supervise waits for each attempt, ignoring --detach")
    kwargs["detach"] = False
    target = int(spec.train.get("epochs") or 0)
    output_dir = Path(absolute(spec.train["output_dir"]))
    attempt = stalls = 0
    code = 0
    while True:
        before = completed_epoch(output_dir)
        if target and before >= target:
            print(f"{spec.name}: epoch {before}/{target} reached")
            return 0
        attempt += 1
        # Nothing written yet means there is nothing of its own to continue from.
        current = spec if before == 0 else resume_spec(spec)
        print(f"{spec.name}: attempt {attempt}, at epoch {before}/{target}")
        code = run(current, **kwargs)
        after = completed_epoch(output_dir)
        if after > before:
            stalls = 0
            continue
        stalls += 1
        print(
            f"{spec.name}: attempt {attempt} exited {code} without finishing an "
            f"epoch ({stalls}/{max_stalls})"
        )
        if stalls >= max_stalls:
            print(f"{spec.name}: giving up at epoch {after}/{target}")
            return code or 1
        if retry_seconds > 0:
            time.sleep(retry_seconds * stalls)


def wait_for_path(pattern: str | Path, poll_seconds: int = 60) -> None:
    """Block until something exists at ``pattern``, which may be a glob.

    How one arm waits for the artifact another arm produces, without either of
    them knowing the timestamped directory the other will pick -- which is also
    why a glob has to be allowed here and not just a literal path.
    """

    pattern = str(absolute(str(pattern)))
    announced = False
    while not glob.glob(pattern):
        if not announced:
            print(f"waiting for {pattern}")
            announced = True
        time.sleep(poll_seconds)


def run_queue(specs: list[Spec], *, supervise: bool = False, **kwargs) -> dict[str, int]:
    """Run specs back to back; a failure does not cancel the rest of the queue."""

    launch = run_until_complete if supervise else run
    results: dict[str, int] = {}
    for spec in specs:
        results[spec.name] = launch(spec, **kwargs)
        print(f"{spec.name}: exit {results[spec.name]}")
    return results
