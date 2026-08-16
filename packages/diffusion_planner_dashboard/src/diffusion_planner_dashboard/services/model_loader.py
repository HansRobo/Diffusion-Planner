"""Load a trained diffusion planner from a training checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from diffusion_planner.models.diffusion_planner import DiffusionPlanner


@dataclass(frozen=True)
class LoadedPlanner:
    """A restored planner and checkpoint metadata for dashboard inference."""

    model: DiffusionPlanner
    epoch: int
    global_step: int
    used_default_config: bool


def load_planner_checkpoint(path: str | Path, device: str) -> LoadedPlanner:
    """Restore a planner on ``device`` from a single-file training checkpoint."""
    checkpoint_path = Path(path).expanduser()
    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    raw_config = checkpoint.get("model_config")
    used_default_config = raw_config is None
    if raw_config is None:
        model = DiffusionPlanner()
    else:
        model_config = dict(raw_config)
        model_config.pop("_target_", None)
        model = DiffusionPlanner(**model_config)

    state = dict(checkpoint["model"])
    for prefix in ("_orig_mod.", "planner."):
        if state and all(key.startswith(prefix) for key in state):
            state = {key.removeprefix(prefix): value for key, value in state.items()}
    model.load_state_dict(state)
    model.to(torch.device(device))
    model.eval()
    return LoadedPlanner(
        model=model,
        epoch=int(checkpoint.get("epoch", 0)),
        global_step=int(checkpoint.get("global_step", 0)),
        used_default_config=used_default_config,
    )
