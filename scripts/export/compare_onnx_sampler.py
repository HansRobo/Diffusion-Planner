#!/usr/bin/env python3
"""Compare the fixed 10-step ONNX sampler with its PyTorch checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from export_onnx import (
    SAMPLER_INPUT_NAMES,
    _load_frame,
    _load_model,
    _ort_outputs,
    _scene_inputs,
)

from diffusion_planner.data.dimensions import TRAJECTORY_DIM, TRAJECTORY_LENGTH
from diffusion_planner.models.onnx import DiffusionPlannerSamplerOnnxWrapper


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-source", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model = _load_model(args.checkpoint)
    frame = _load_frame(args.data_source, args.frame_index)
    scene_inputs = tuple(value[: args.batch_size] for value in _scene_inputs(frame))

    agents = scene_inputs[1].shape[1] + 1
    generator = torch.Generator().manual_seed(args.seed)
    initial_noise = torch.randn(
        args.batch_size,
        agents,
        TRAJECTORY_LENGTH,
        TRAJECTORY_DIM,
        generator=generator,
    )
    sampler_inputs = (initial_noise, *scene_inputs)

    wrapper = DiffusionPlannerSamplerOnnxWrapper(model).eval()
    with torch.inference_mode():
        expected = wrapper(*sampler_inputs).detach().cpu().numpy()
    actual = _ort_outputs(
        args.onnx,
        dict(zip(SAMPLER_INPUT_NAMES, sampler_inputs, strict=True)),
    )[0]

    absolute_error = np.abs(actual - expected)
    relative_error = absolute_error / np.maximum(np.abs(expected), 1e-12)
    close_mask = np.isclose(actual, expected, rtol=args.rtol, atol=args.atol)
    close = bool(close_mask.all())
    print(f"ONNX: {args.onnx}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"shape: {actual.shape}")
    print(f"max absolute error: {absolute_error.max():.9g}")
    print(f"mean absolute error: {absolute_error.mean():.9g}")
    print(f"p99 absolute error: {np.quantile(absolute_error, 0.99):.9g}")
    print(f"max relative error: {relative_error.max():.9g}")
    print(f"mismatched elements: {(~close_mask).sum()} / {close_mask.size}")
    print(f"allclose (rtol={args.rtol:g}, atol={args.atol:g}): {close}")
    if not close:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
