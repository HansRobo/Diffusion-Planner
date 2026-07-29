"""Experiment specs: one JSON file per arm, inheritance, and sweep expansion.

Schema (every field optional except ``train``):

    {
      "name":      "dual",                       # run label; sweeps suffix it
      "extends":   "base_fp32.json",             # relative to this spec, chainable
      "objective": "awr",                        # awr | grpo | dpo
      "entrypoint": "rlvr.train_awr",            # any module with build_parser()
      "reward":    {"<RewardConfig field>": value},
      "launch":    {"nproc": 8, "env": {...}, "torchrun": [...]},
      "train":     {"<train_awr flag>": value},  # verbatim flag surface
      "prepare":   [{"argv": [...], "creates": "<path>"}],
      "sweep":     {"<train_awr flag>": [v1, v2], "reward.<field>": [v1, v2]}
    }

``train`` keys are flag names without the dashes, taken from whichever trainer
``entrypoint`` names; ``reward`` keys are ``RewardConfig`` fields.  ``null`` deletes an
inherited key.  ``sweep`` expands to the cartesian product of its lists, and a
``reward.``-prefixed sweep key varies the reward instead of a flag.

``entrypoint`` is what lets one spec language drive more than one trainer: the
candidate-weighting objectives live inside ``rlvr.train_awr``, while a method that needs
its own optimization loop (importance ratios, a KL term, a reference pass) is its own
module.  Both are specs, planned and launched the same way.

``prepare`` runs before the arm and is how a sweep varies a knob that owns an
artifact: ``{flag}`` and ``{name}`` interpolate this arm's values, and ``creates``
makes the step idempotent, so a beta sweep gets one overlay per beta instead of
one overlay for all of them.
"""

from __future__ import annotations

import copy
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAX_EXTENDS_DEPTH = 8

DEFAULT_ENTRYPOINT = "rlvr.train_awr"

# Specs carry project-relative paths; the repo root is where they resolve from,
# so a spec is not tied to the host that wrote it.
ROOT = Path(__file__).resolve().parents[2]

PATH_FLAGS = (
    "args_path",
    "config",
    "model_path",
    "weights_zip",
    "train_npz_list",
    "valid_npz_list",
    "resume_replay_root",
    "compressed_replay_context_root",
    "shared_replay_encoding_root",
    "shared_replay_context_root",
    "shared_replay_expert_anchor_root",
    "output_dir",
)

_DIRECTIVE_PATHS = ("latest_epoch", "resume_after_latest_epoch", "newest")


def absolute(value: Any) -> Any:
    """Resolve a project-relative path, including inside a dynamic directive."""

    if isinstance(value, str):
        return value if Path(value).is_absolute() else str(ROOT / value)
    if isinstance(value, dict):
        return {
            key: absolute(item) if key in _DIRECTIVE_PATHS else item
            for key, item in value.items()
        }
    return value


@dataclass
class Spec:
    name: str
    objective: str = "awr"
    entrypoint: str = DEFAULT_ENTRYPOINT
    reward: dict[str, Any] = field(default_factory=dict)
    train: dict[str, Any] = field(default_factory=dict)
    launch: dict[str, Any] = field(default_factory=dict)
    prepare: list[dict[str, Any]] = field(default_factory=list)
    source: Path | None = None

    @property
    def nproc(self) -> int:
        return int(self.launch.get("nproc", 8))

    @property
    def env(self) -> dict[str, str]:
        return {str(k): str(v) for k, v in dict(self.launch.get("env", {})).items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "objective": self.objective,
            "entrypoint": self.entrypoint,
            "reward": self.reward,
            "launch": self.launch,
            "prepare": self.prepare,
            "train": self.train,
        }


def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge; a ``None`` value deletes the inherited key."""

    result = copy.deepcopy(base)
    for key, value in override.items():
        if value is None:
            result.pop(key, None)
        elif isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read(path: Path, depth: int = 0) -> dict[str, Any]:
    if depth > MAX_EXTENDS_DEPTH:
        raise ValueError(f"extends chain deeper than {MAX_EXTENDS_DEPTH} at {path}")
    raw = json.loads(path.read_text())
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    parent_path = (path.parent / str(parent)).resolve()
    if not parent_path.exists():
        raise FileNotFoundError(f"{path}: extends {parent} -> {parent_path} missing")
    return merge(_read(parent_path, depth + 1), raw)


def load(path: str | Path) -> list[Spec]:
    """Load a spec file, resolve inheritance, and expand its sweep."""

    path = Path(path).resolve()
    raw = _read(path)
    name = str(raw.get("name", path.stem))
    sweep = dict(raw.get("sweep", {}))
    for key, values in sweep.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"{path}: sweep.{key} must be a non-empty list")

    specs: list[Spec] = []
    keys = sorted(sweep)
    combinations = list(itertools.product(*(sweep[k] for k in keys))) or [()]
    for combination in combinations:
        chosen = dict(zip(keys, combination))
        reward_sweep = {
            k.split(".", 1)[1]: v for k, v in chosen.items() if k.startswith("reward.")
        }
        train = merge(
            dict(raw.get("train", {})),
            {k: v for k, v in chosen.items() if not k.startswith("reward.")},
        )
        suffix = "".join(
            f"_{k.split('.')[-1]}{_tag(v)}" for k, v in zip(keys, combination)
        )
        # Without this every arm of a sweep would inherit one output_dir and one
        # wandb run, and overwrite each other's checkpoints.
        for key in ("output_dir", "exp_name", "wandb_run_name"):
            if suffix and isinstance(train.get(key), str):
                train[key] = train[key] + suffix
        # A swept knob that owns a path has to reach that path: one definition
        # of the per-arm overlay, referenced by both the flag and its pre-step.
        scope = {**train, "name": name + suffix}
        train = {
            key: _fill(value, scope) if isinstance(value, str) else value
            for key, value in train.items()
        }
        for key in PATH_FLAGS:
            if key in train:
                train[key] = absolute(train[key])
        reward = merge(dict(raw.get("reward", {})), reward_sweep)
        specs.append(
            Spec(
                name=name + suffix,
                objective=str(raw.get("objective", "awr")),
                entrypoint=str(raw.get("entrypoint", DEFAULT_ENTRYPOINT)),
                reward=reward,
                train=train,
                launch=dict(raw.get("launch", {})),
                prepare=_prepare(
                    raw.get("prepare", []), {**train, **reward, "name": name + suffix}
                ),
                source=path,
            )
        )
    return specs


def _prepare(entries: Any, values: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve each pre-step's placeholders against this arm's own values."""

    resolved: list[dict[str, Any]] = []
    for entry in list(entries):
        if not isinstance(entry, dict) or not entry.get("argv"):
            raise ValueError(f"prepare entry needs a non-empty argv: {entry!r}")
        step: dict[str, Any] = {}
        if entry.get("creates"):
            step["creates"] = _fill(str(entry["creates"]), values)
        scope = {**values, "creates": step.get("creates", "")}
        step["argv"] = [_fill(str(token), scope) for token in entry["argv"]]
        resolved.append(step)
    return resolved


def _fill(text: str, values: dict[str, Any]) -> str:
    try:
        return text.format(**values)
    except (KeyError, IndexError) as error:
        raise ValueError(f"prepare: {text!r} has no value for {error}") from None


def _tag(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    text = str(value).replace(".", "p").replace("-", "m")
    return "".join(c for c in text if c.isalnum() or c == "p")
