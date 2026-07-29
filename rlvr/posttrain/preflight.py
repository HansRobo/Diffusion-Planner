"""Checks that fail in a second instead of after the epoch-0 eval.

Every rule is derived from the spec's own flags: a flag that needs an artifact
is what makes that artifact required.
"""

from __future__ import annotations

import json
from pathlib import Path

from rlvr.posttrain.flags import Unresolved, _dynamic
from rlvr.posttrain.spec import PATH_FLAGS as spec_path_flags
from rlvr.posttrain.spec import Spec

# output_dir is created by the run, so it is not one of the required artifacts.
PATH_FLAGS = tuple(f for f in spec_path_flags if f != "output_dir")

# A dynamic value names what an earlier arm produces, so before that arm has run
# it does not resolve.  Preflight exists to report exactly that, and a traceback
# out of it would take down whatever is supervising the queue.
UNRESOLVED = (Unresolved, OSError, KeyError)


def check(spec: Spec) -> list[str]:
    problems: list[str] = []
    resolved: dict[str, Path] = {}
    for flag in PATH_FLAGS:
        value = spec.train.get(flag)
        if value is None:
            continue
        try:
            path = Path(str(_dynamic(value)))
        except UNRESOLVED as error:
            problems.append(f"--{flag} unresolved: {error}")
            continue
        resolved[flag] = path
        if not path.exists():
            problems.append(f"--{flag} missing: {path}")

    config = resolved.get("config")
    if config is not None and config.is_file():
        problems.extend(_check_refresh_exploration(spec, config))

    overlay = resolved.get("resume_replay_root")
    if overlay is not None and overlay.is_dir():
        problems.extend(_check_overlay(spec, overlay))

    checkpoint = resolved.get("model_path")
    if checkpoint is not None and checkpoint.is_file():
        problems.extend(_check_checkpoint(spec, checkpoint))
    return problems


def _check_refresh_exploration(spec: Spec, config: Path) -> list[str]:
    """A refresh epoch drags an arm back onto the mining path, which aborts.

    Mining scores its candidates group-relative and so refuses candidate-0
    behaviour-anchor semantics, while on the replay path the same pair is what
    the arm is for -- which is why the mine spec drops those flags and the arm
    keeps them.  Nothing warns about the collision in advance: compile warmup
    trips the same guard and records it as data, and the abort itself waits
    until the first refresh epoch, which can be days in.
    """

    try:
        loaded = json.loads(config.read_text())
    except (OSError, ValueError):  # the entrypoint gives a better error for this
        return []
    # The config groups its settings into sections while the CLI is flat, and it
    # is the flat name that a spec overrides, so compare on the flat name.
    settings = {}
    for name, value in loaded.items():
        settings.update(value if isinstance(value, dict) else {name: value})

    def setting(name: str, *aliases: str):
        value = spec.train.get(name)
        if value is not None:
            return value
        for key in (name, *aliases):
            if key in settings:
                return settings[key]
        return None

    try:
        interval = int(setting("hdp_rollout_interval"))
        last = int(setting("epochs", "train_epochs"))
    except (TypeError, ValueError):
        return []
    if interval <= 0:  # disk replay off entirely, so no epoch reaches mining
        return []
    start = spec.train.get("start_epoch")
    try:
        start = 1 if start is None else int(_dynamic(start))
    except UNRESOLVED:  # reported against the checkpoint; the relaunch sees the real one
        return []
    refresh = [e for e in range(start, last + 1) if (e - 1) % interval == 0]
    if not refresh:
        return []
    if setting("positive_advantage_only") and setting("plannerrft_guided_exploration"):
        return [
            f"--hdp_rollout_interval {interval} makes epoch {refresh[0]} a refresh, "
            "and mining aborts every rank on --positive_advantage_only together with "
            f"--plannerrft_guided_exploration: raise the interval above --epochs {last}, "
            "or drop one of the pair the way the mine spec does"
        ]
    return []


def _check_overlay(spec: Spec, overlay: Path) -> list[str]:
    """A derived overlay crashes all ranks on a missing provenance contract."""

    problems: list[str] = []
    needs_gate_step = bool(spec.train.get("expert_anchor_safe_step_scoped"))
    for rank in range(spec.nproc):
        directory = overlay / f"rank_{rank:04d}"
        if not directory.is_dir():
            problems.append(f"overlay rank missing: {directory}")
            continue
        if not needs_gate_step:
            continue
        array = directory / "expert_hard_gate_step.npy"
        manifest = directory / "expert_anchor_manifest.json"
        if not array.is_file() or array.stat().st_size == 0:
            problems.append(f"--expert_anchor_safe_step_scoped needs {array}")
        elif not manifest.is_file():
            problems.append(f"gate step not registered, no manifest: {manifest}")
        elif "expert_hard_gate_step" not in manifest.read_text():
            problems.append(f"gate step not registered in {manifest}")
    return problems


def _check_checkpoint(spec: Spec, checkpoint: Path) -> list[str]:
    import torch

    problems: list[str] = []
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as error:  # a corrupt checkpoint reads as a preflight failure
        return [f"--model_path unreadable: {checkpoint} ({error})"]
    if not isinstance(state, dict):
        return [f"--model_path is not a checkpoint dict: {checkpoint}"]
    if spec.train.get("use_policy_state") and "model" not in state:
        problems.append(f"--use_policy_state needs 'model' in {checkpoint.name}")
    if spec.train.get("resume_optimizer_state") and "optimizer_state_dict" not in state:
        problems.append(
            f"--resume_optimizer_state needs 'optimizer_state_dict' in {checkpoint.name}"
        )
    # Only a continuation has to line up with the donor's epoch counter. A run
    # that starts from someone else's checkpoint numbers its own epochs from
    # scratch, and the epoch it happens to have been saved at means nothing here.
    if spec.train.get("use_policy_state"):
        start = spec.train.get("start_epoch")
        try:
            start = None if start is None else int(_dynamic(start))
        except UNRESOLVED as error:
            problems.append(f"--start_epoch unresolved: {error}")
            start = None
        saved = state.get("epoch")
        if start is not None and isinstance(saved, int) and saved + 1 != start:
            problems.append(
                f"--start_epoch {start} does not continue {checkpoint.name} (epoch {saved})"
            )
    return problems


def report(spec: Spec) -> bool:
    problems = check(spec)
    for problem in problems:
        print(f"{spec.name}: {problem}")
    return not problems
