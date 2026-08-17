"""Run planner sampling for one dashboard frame."""

from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from diffusion_planner.data import PlannerDataNormalizer
from diffusion_planner.models.diffusion_planner import DiffusionPlanner
from diffusion_planner.models.turn_indicator import TurnIndicatorModel


def run_inference(
    model: DiffusionPlanner,
    frame_data: Mapping[str, Any],
    *,
    device: str,
    num_steps: int,
    time_epsilon: float,
    noise_scale: float,
    seed: int,
) -> tuple[NDArray[np.float32], float]:
    """Sample one trajectory batch and return the prediction and elapsed seconds."""
    torch_device = torch.device(device)
    normalizer = PlannerDataNormalizer()
    normalized_frame = normalizer(
        {key: np.asarray(value) for key, value in frame_data.items()}
    )
    input_data = {
        key: torch.as_tensor(value, device=torch_device).unsqueeze(0)
        for key, value in normalized_frame.items()
    }
    generator = torch.Generator(device=torch_device).manual_seed(seed)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    start = perf_counter()
    with torch.inference_mode():
        prediction = model.sample(
            input_data,
            num_steps=num_steps,
            time_epsilon=time_epsilon,
            noise_scale=noise_scale,
            generator=generator,
        )
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    elapsed = perf_counter() - start
    prediction_array = prediction[0].detach().float().cpu().numpy()
    prediction_array = normalizer.denormalize_trajectory(prediction_array)
    return prediction_array.astype(np.float32, copy=False), elapsed


def run_turn_indicator_inference(
    model: TurnIndicatorModel,
    frame_data: Mapping[str, Any],
    *,
    device: str,
) -> tuple[NDArray[np.float32], int, float]:
    """Predict the next turn indicator and return probabilities and elapsed seconds."""
    torch_device = torch.device(device)
    normalized_frame = PlannerDataNormalizer()(
        {key: np.asarray(value) for key, value in frame_data.items()}
    )
    input_data = {
        key: torch.as_tensor(value, device=torch_device).unsqueeze(0)
        for key, value in normalized_frame.items()
    }
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    start = perf_counter()
    with torch.inference_mode():
        logits = model(input_data)
        probabilities = torch.softmax(logits, dim=-1)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    elapsed = perf_counter() - start
    probability_array = probabilities[0].detach().float().cpu().numpy()
    predicted_report = int(probability_array.argmax()) + 1
    return probability_array.astype(np.float32, copy=False), predicted_report, elapsed
