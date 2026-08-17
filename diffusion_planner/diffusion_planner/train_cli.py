"""The command line shared by train_predictor.py and train_run.py.

Both entrypoints take the same flags, built from ``TrainConfig`` fields marked
``cli(...)``. Names, types, defaults, choices and help text are read off the dataclass,
so adding a flag is one edit (mark the field with ``cli(...)`` in ``train_config.py``).
"""

import argparse
from dataclasses import MISSING, Field, fields
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin

from diffusion_planner.train_config import TrainConfig
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer


def boolean(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


def cli_fields() -> list[Field]:
    return [f for f in fields(TrainConfig) if f.metadata.get("cli")]


def _unwrap_optional(annotation: Any) -> Any:
    """``Optional[str]`` -> ``str``; anything else unchanged."""
    if get_origin(annotation) is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _add_argument(parser: argparse.ArgumentParser, f: Field) -> None:
    required = f.default is MISSING and f.default_factory is MISSING
    kwargs: dict[str, Any] = {"help": f.metadata["help"]}
    if required:
        kwargs["required"] = True
    else:
        if f.default is not MISSING:
            kwargs["default"] = f.default
        else:
            kwargs["default"] = f.default_factory()

    annotation = _unwrap_optional(f.type)
    if get_origin(annotation) is Literal:
        kwargs["type"] = str
        kwargs["choices"] = get_args(annotation)
    elif annotation is bool:
        # Bare form (``--use_amp``) is common; ``--use_amp False`` is needed to switch off
        # a default-on flag.
        kwargs["type"] = boolean
        kwargs["nargs"] = "?"
        kwargs["const"] = True
    elif get_origin(annotation) is list:
        # ``list[X]`` -> ``nargs="+"``. Forwarding the typing-form as ``type=`` would make
        # argparse call ``list[X](string)`` on every value, which always crashes.
        kwargs.pop("type", None)
        kwargs["nargs"] = "+"
    else:
        kwargs["type"] = annotation

    parser.add_argument(f"--{f.name}", **kwargs)


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    for f in cli_fields():
        _add_argument(parser, f)
    return parser


def resolve_paths(args: argparse.Namespace) -> None:
    """Make path-ish arguments absolute, in place.

    train_run.py runs the trainer with cwd set to the entrypoint directory, so a path
    the user typed relative to their own cwd has to be resolved before it is forwarded.
    """
    for f in cli_fields():
        if not f.metadata.get("path"):
            continue
        value = getattr(args, f.name)
        if not value:
            continue
        # Handle list types (nargs="+")
        if isinstance(value, list):
            resolved = [str(Path(v).resolve()) for v in value]
            setattr(args, f.name, resolved)
        else:
            setattr(args, f.name, str(Path(value).resolve()))


def build_train_config(args: argparse.Namespace, **overrides: Any) -> TrainConfig:
    """Build the config the trainer runs on, with normalizers attached."""
    values = {f.name: getattr(args, f.name) for f in cli_fields()}
    values.update(overrides)

    config = TrainConfig(**values)
    config.state_normalizer = StateNormalizer.from_json(config)
    config.observation_normalizer = ObservationNormalizer.from_json(config)
    return config


def to_command_line(args: argparse.Namespace, exclude: tuple[str, ...] = ()) -> list[str]:
    """Render the parsed arguments back into flags for the trainer subprocess.

    Generated from the same field list the parser was built from, so a new flag is
    forwarded automatically rather than being silently dropped by a launcher that was
    never updated.

    Values left at their default are omitted: both processes read those defaults from
    the same dataclass, so passing them adds nothing but noise to the logged command.
    """
    argv: list[str] = []
    for f in cli_fields():
        if f.name in exclude:
            continue
        value = getattr(args, f.name)
        if value is None:
            continue
        # A required field has no default, so it is always forwarded.
        if f.default is not MISSING and value == f.default:
            continue
        # Handle list types (nargs="+")
        if isinstance(value, list):
            if not value:  # empty list
                continue
            argv.append(f"--{f.name}")
            argv.extend(str(v) for v in value)
        else:
            argv += [f"--{f.name}", str(value)]
    return argv
