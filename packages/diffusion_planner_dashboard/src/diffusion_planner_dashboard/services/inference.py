"""Run planner sampling for one dashboard frame."""

from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from diffusion_planner.models.diffusion_planner import DiffusionPlanner


def run_inference(
    model: DiffusionPlanner,
    frame_data: Mapping[str, Any],
    *,
    device: str,
    num_steps: int,
    noise_scale: float,
    seed: int,
) -> tuple[NDArray[np.float32], float]:
    """Sample one trajectory batch and return the prediction and elapsed seconds."""
    torch_device = torch.device(device)
    input_data = {
        key: torch.as_tensor(np.asarray(value), device=torch_device).unsqueeze(0)
        for key, value in frame_data.items()
    }
    generator = torch.Generator(device=torch_device).manual_seed(seed)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    start = perf_counter()
    with torch.inference_mode():
        prediction = model.sample(
            input_data,
            num_steps=num_steps,
            noise_scale=noise_scale,
            generator=generator,
        )
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    elapsed = perf_counter() - start
    return prediction[0].detach().float().cpu().numpy(), elapsed
