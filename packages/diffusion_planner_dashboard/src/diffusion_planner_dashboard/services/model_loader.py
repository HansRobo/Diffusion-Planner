"""Load diffusion planner checkpoints and sampler ONNX models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnxruntime as ort
import torch

from diffusion_planner.models.diffusion_planner import DiffusionPlanner


@dataclass(frozen=True)
class LoadedPlanner:
    """A restored planner and checkpoint metadata for dashboard inference."""

    model: DiffusionPlanner
    epoch: int
    global_step: int


@dataclass(frozen=True)
class LoadedOnnxPlanner:
    """An ONNX sampler session and its active execution provider."""

    session: ort.InferenceSession
    provider: str
    sampling_steps: int = 10


def load_planner_checkpoint(path: str | Path, device: str) -> LoadedPlanner:
    """Restore a planner on ``device`` from a single-file training checkpoint."""
    checkpoint_path = Path(path).expanduser()
    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model_config = dict(checkpoint["model_config"])
    model_config.pop("_target_", None)
    model = DiffusionPlanner(**model_config)
    model.load_state_dict(checkpoint["model"])
    model.to(torch.device(device))
    model.eval()
    return LoadedPlanner(
        model=model,
        epoch=int(checkpoint.get("epoch", 0)),
        global_step=int(checkpoint.get("global_step", 0)),
    )


def load_onnx_planner(path: str | Path, device: str) -> LoadedOnnxPlanner:
    """Load a fixed-step sampler ONNX with the best requested ORT provider."""
    model_path = Path(path).expanduser()
    available_providers = ort.get_available_providers()
    requested_provider = (
        "CUDAExecutionProvider"
        if device == "cuda" and "CUDAExecutionProvider" in available_providers
        else "CPUExecutionProvider"
    )
    session = ort.InferenceSession(str(model_path), providers=[requested_provider])
    return LoadedOnnxPlanner(
        session=session,
        provider=session.get_providers()[0],
    )


def load_planner(path: str | Path, device: str) -> LoadedPlanner | LoadedOnnxPlanner:
    """Load either a PyTorch checkpoint or a sampler ONNX model."""
    model_path = Path(path).expanduser()
    if model_path.suffix.lower() == ".onnx":
        return load_onnx_planner(model_path, device)
    return load_planner_checkpoint(model_path, device)
