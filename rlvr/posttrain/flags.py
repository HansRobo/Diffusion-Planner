"""Spec -> trainer argv, validated against that trainer's own parser.

Nothing here enumerates flags: the mapping is derived from the entrypoint's
``build_parser()``, so a spec key the trainer does not accept, or a value of the
wrong arity, fails before any GPU is taken.  That is also what makes one spec
language drive several trainers -- each brings its own flag surface.
"""

from __future__ import annotations

import argparse
import difflib
import glob
import importlib
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from rlvr.posttrain.spec import DEFAULT_ENTRYPOINT, Spec, absolute


class Unresolved(ValueError):
    """A dynamic value whose artifact is not there yet.

    Distinct from a malformed spec: the same spec resolves once the arm it
    depends on has run, so it is reported as a missing artifact rather than as a
    mistake in the file.
    """


@lru_cache(maxsize=8)
def actions(entrypoint: str = DEFAULT_ENTRYPOINT) -> dict[str, argparse.Action]:
    """The entrypoint's long options, by name without the dashes."""

    module = importlib.import_module(entrypoint)
    build = getattr(module, "build_parser", None)
    if build is None:
        raise ValueError(
            f"{entrypoint} has no build_parser(); a spec is validated against the "
            "trainer's own parser, so the trainer has to expose one"
        )
    result: dict[str, argparse.Action] = {}
    for action in build()._actions:
        for option in action.option_strings:
            if option.startswith("--") and not option.startswith("--no-"):
                result[option[2:]] = action
    return result


def _tokens(name: str, value: Any, action: argparse.Action) -> list[str]:
    flag = f"--{name}"
    if isinstance(action, argparse.BooleanOptionalAction):
        if not isinstance(value, bool):
            raise ValueError(f"{flag} takes true/false, got {value!r}")
        return [flag if value else f"--no-{name}"]
    if action.nargs == 0 or isinstance(
        action, (argparse._StoreTrueAction, argparse._StoreFalseAction)
    ):
        if not isinstance(value, bool):
            raise ValueError(f"{flag} is a switch, use true/false, not {value!r}")
        return [flag] if value else []
    if isinstance(value, (list, tuple)):
        expected = action.nargs
        if isinstance(expected, int) and len(value) != expected:
            raise ValueError(f"{flag} takes {expected} values, got {len(value)}")
        return [flag, *(_scalar(v) for v in value)]
    if action.nargs not in (None, "?"):
        raise ValueError(f"{flag} takes {action.nargs} values, got a scalar")
    return [flag, _scalar(value)]


def _scalar(value: Any) -> str:
    value = _dynamic(value)
    if isinstance(value, bool):
        raise ValueError(f"bool where a value was expected: {value!r}")
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e16:
        return str(int(value))
    return str(value)


def _reward_value(value: Any) -> str:
    """Unlike a flag, a reward field takes a bool as a value, not as a switch."""

    if isinstance(value, bool):
        return "true" if value else "false"
    return _scalar(value)


def _epoch_number(path: Path) -> int | None:
    tail = path.stem.split("_")[-1]
    return int(tail) if tail.isdigit() else None


def _last_epoch(directory: str) -> Path:
    """The arm's LAST epoch checkpoint, never a "best" by any proxy.

    Every trainer run owns a fresh timestamped directory under its ``output_dir``,
    so an arm's checkpoints sit one level below the path a spec names, and an arm
    that has been resumed has several such directories.  Both levels are searched
    and the highest epoch across all of them wins -- that is what lets a resumed
    arm continue from the epoch it actually reached rather than from wherever its
    newest attempt happened to start.
    """

    root = Path(directory)
    found = [
        (number, path)
        for path in (*root.glob("epoch_*.pth"), *root.glob("*/epoch_*.pth"))
        if (number := _epoch_number(path)) is not None
    ]
    if not found:
        raise Unresolved(f"no epoch_*.pth under {directory}")
    return max(found, key=lambda item: (item[0], item[1].stat().st_mtime))[1]


def _newest(pattern: str) -> Path:
    """The most recently written path matching ``pattern``.

    Every trainer run owns a timestamped directory, so an artifact one arm hands
    to the next is knowable by shape but not by name.  Globbing for it is what
    keeps the two arms from having to agree on a stamp, or on a symlink somebody
    has to remember to make between them.
    """

    matches = glob.glob(pattern)
    if not matches:
        raise Unresolved(f"nothing matches {pattern}")
    return Path(max(matches, key=lambda item: Path(item).stat().st_mtime))


def _dynamic(value: Any) -> Any:
    """Resolve the ``{directive: argument}`` forms a spec may use for a value.

    A directive that names a run directory is resolved against the repo root
    rather than the working directory, because these are read wherever the
    launching process happens to be sitting and only one of those two is a
    property of the spec.
    """

    if not isinstance(value, dict) or len(value) != 1:
        return value
    (directive, argument), = value.items()
    if directive == "latest_epoch":
        return _last_epoch(absolute(str(argument)))
    if directive == "resume_after_latest_epoch":
        stem = _last_epoch(absolute(str(argument))).stem
        return int(stem.split("_")[-1]) + 1
    if directive == "newest":
        return _newest(absolute(str(argument)))
    if directive == "env":
        import os

        return os.environ[str(argument)]
    raise ValueError(f"unknown value directive {directive!r}")


def validate(spec: Spec) -> list[str]:
    from rlvr.posttrain.objectives import available

    errors: list[str] = []
    try:
        known = actions(spec.entrypoint)
    except (ImportError, ValueError) as error:
        return [f"entrypoint {spec.entrypoint}: {error}"]
    # A trainer with its own optimization loop does not take a weighting objective,
    # and pricing its arm against the plugin registry would be a lie.
    if "awr_objective" in known and spec.objective not in available():
        errors.append(
            f"unknown objective {spec.objective!r}; available: {', '.join(available())}"
        )
    if spec.reward and "reward_override" not in known:
        errors.append(
            f"reward overrides need a trainer that takes --reward_override, "
            f"{spec.entrypoint} does not"
        )
    for name, value in spec.train.items():
        if name not in known:
            hint = difflib.get_close_matches(name, known, n=1)
            errors.append(
                f"unknown flag --{name}" + (f" (did you mean --{hint[0]}?)" if hint else "")
            )
            continue
        try:
            _tokens(name, value, known[name])
        except (Unresolved, OSError, KeyError) as error:
            errors.append(f"--{name}: unresolved value {value!r} ({error})")
        except ValueError as error:
            errors.append(str(error))
    from rlvr.reward import RewardConfig
    from dataclasses import fields as dataclass_fields

    known_reward = {f.name for f in dataclass_fields(RewardConfig)}
    for name in spec.reward:
        if name not in known_reward:
            hint = difflib.get_close_matches(name, known_reward, n=1)
            errors.append(
                f"unknown reward field {name}"
                + (f" (did you mean {hint[0]}?)" if hint else "")
            )
    for step in spec.prepare:
        if not isinstance(step.get("argv"), list) or not step["argv"]:
            errors.append(f"prepare entry needs a non-empty argv: {step!r}")
    if not ({"model_path", "weights_zip"} & set(spec.train)):
        errors.append("train needs model_path or weights_zip")
    if spec.nproc < 1:
        errors.append(f"launch.nproc must be >= 1, got {spec.nproc}")
    return errors


def to_argv(spec: Spec) -> list[str]:
    errors = validate(spec)
    if errors:
        raise ValueError(f"{spec.name}: " + "; ".join(errors))
    known = actions(spec.entrypoint)
    argv = ["--awr_objective", spec.objective] if "awr_objective" in known else []
    for name in sorted(spec.reward):
        argv.extend(["--reward_override", f"{name}={_reward_value(spec.reward[name])}"])
    for name in sorted(spec.train):
        argv.extend(_tokens(name, spec.train[name], known[name]))
    return argv


def command(spec: Spec) -> list[str]:
    # A trainer that never initialises a process group gains nothing from the
    # launcher, and a wrapper it does not expect only hides its exit code.
    if spec.launch.get("launcher") == "direct":
        return [sys.executable, "-m", spec.entrypoint, *to_argv(spec)]
    torchrun = list(spec.launch.get("torchrun", ["--standalone"]))
    # The console script is not on PATH in a detached session; the module is
    # the same launcher and comes from the interpreter that is already running.
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        *torchrun,
        f"--nproc_per_node={spec.nproc}",
        "-m",
        spec.entrypoint,
        *to_argv(spec),
    ]
