#!/usr/bin/env python3
"""Export a training checkpoint into scene-encoder and trajectory-decoder ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin
import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq
import torch
from numpy.typing import NDArray

from diffusion_planner.data import PlannerDataNormalizer
from diffusion_planner.data.dimensions import TRAJECTORY_DIM, TRAJECTORY_LENGTH
from diffusion_planner.models.diffusion_planner import DiffusionPlanner
from diffusion_planner.models.onnx import (
    SCENE_INPUT_NAMES,
    DiffusionPlannerSamplerOnnxWrapper,
    SceneEncoderOnnxWrapper,
    TrajectoryDecoderOnnxWrapper,
)

hdf5plugin.register(filters="zstd")

SCENE_OUTPUT_NAMES = ("scene", "scene_mask", "agent_pose", "agent_mask")
DECODER_INPUT_NAMES = (
    "x",
    "x_mask",
    "scene",
    "scene_mask",
    "agent_pose",
    "time",
)
SAMPLER_INPUT_NAMES = ("initial_noise", *SCENE_INPUT_NAMES)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--data-source",
        type=Path,
        required=True,
        help="H5 file or Parquet H5 index used to create representative inputs.",
    )
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--opset-version", type=int, default=18)
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def _load_model(checkpoint_path: Path) -> DiffusionPlanner:
    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path.expanduser(), map_location="cpu", weights_only=False
    )
    model_config = dict(checkpoint["model_config"])
    model_config.pop("_target_", None)
    model = DiffusionPlanner(**model_config)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def _resolve_frame(source: Path, frame_index: int) -> tuple[Path, int]:
    source = source.expanduser().resolve()
    if source.suffix.lower() in {".h5", ".hdf5"}:
        return source, frame_index
    if source.suffix.lower() == ".parquet":
        table = pq.read_table(source, columns=["h5_path", "frame_index"])
        if not 0 <= frame_index < table.num_rows:
            raise IndexError(
                f"frame-index {frame_index} is outside index with {table.num_rows} rows"
            )
        return (
            Path(table["h5_path"][frame_index].as_py()),
            int(table["frame_index"][frame_index].as_py()),
        )
    raise ValueError(f"data-source must be H5 or Parquet: {source}")


def _load_frame(source: Path, frame_index: int) -> dict[str, NDArray[Any]]:
    h5_path, h5_frame_index = _resolve_frame(source, frame_index)
    with h5py.File(h5_path, "r") as file:
        frames_object = file["frames"]
        if not isinstance(frames_object, h5py.Group):
            raise ValueError(f"H5 'frames' must be a group: {h5_path}")
        frame: dict[str, NDArray[Any]] = {}
        for name in SCENE_INPUT_NAMES:
            dataset = frames_object[name]
            if not isinstance(dataset, h5py.Dataset):
                raise ValueError(f"H5 'frames/{name}' must be a dataset: {h5_path}")
            frame[name] = np.asarray(dataset[h5_frame_index])
        return frame


def _scene_inputs(frame: dict[str, NDArray[Any]]) -> tuple[torch.Tensor, ...]:
    normalized = PlannerDataNormalizer()(frame)
    return tuple(
        torch.from_numpy(normalized[name])
        .unsqueeze(0)
        .repeat((2,) + (1,) * np.asarray(normalized[name]).ndim)
        for name in SCENE_INPUT_NAMES
    )


def _dynamic_shapes(inputs: tuple[torch.Tensor, ...]) -> tuple[dict[int, Any], ...]:
    batch = torch.export.Dim("batch", min=1, max=2)
    return tuple({0: batch} for _ in inputs)


def _export(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    path: Path,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    opset_version: int,
    optimize: bool = True,
) -> None:
    torch.onnx.export(
        model,
        inputs,
        path,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        dynamo=True,
        dynamic_shapes=_dynamic_shapes(inputs),
        external_data=False,
        optimize=optimize,
    )
    print(f"exported: {path}")


def _ort_outputs(path: Path, inputs: dict[str, torch.Tensor]) -> list[np.ndarray]:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    outputs = session.run(
        None,
        {name: value.detach().cpu().numpy() for name, value in inputs.items()},
    )
    return [np.asarray(output) for output in outputs]


def _validate(
    path: Path,
    input_names: tuple[str, ...],
    inputs: tuple[torch.Tensor, ...],
    expected: tuple[torch.Tensor, ...],
) -> None:
    for batch_size in (1, 2):
        batch_inputs = tuple(value[:batch_size] for value in inputs)
        actual = _ort_outputs(path, dict(zip(input_names, batch_inputs, strict=True)))
        for index, (torch_value, ort_value) in enumerate(
            zip(expected, actual, strict=True)
        ):
            np.testing.assert_allclose(
                ort_value,
                torch_value[:batch_size].detach().cpu().numpy(),
                rtol=1e-4,
                atol=1e-5,
                err_msg=(
                    f"ONNX output {index} differs from PyTorch at batch "
                    f"size {batch_size}"
                ),
            )
    print(f"validated: {path}")


def main() -> None:
    args = _parse_args()
    model = _load_model(args.checkpoint)
    frame = _load_frame(args.data_source, args.frame_index)
    scene_inputs = _scene_inputs(frame)
    scene_wrapper = SceneEncoderOnnxWrapper(model.scene_encoder).eval()
    with torch.inference_mode():
        scene_outputs = scene_wrapper(*scene_inputs)

    scene, scene_mask, agent_pose, agent_mask = scene_outputs
    batch, agents = agent_mask.shape
    decoder_inputs = (
        torch.randn(batch, agents, TRAJECTORY_LENGTH, TRAJECTORY_DIM),
        agent_mask,
        scene,
        scene_mask,
        agent_pose,
        torch.full((batch,), 0.5),
    )
    decoder_wrapper = TrajectoryDecoderOnnxWrapper(model.trajectory_decoder).eval()
    with torch.inference_mode():
        decoder_output = decoder_wrapper(*decoder_inputs)
    sampler_inputs = (decoder_inputs[0], *scene_inputs)
    sampler_wrapper = DiffusionPlannerSamplerOnnxWrapper(model).eval()
    with torch.inference_mode():
        sampler_output = sampler_wrapper(*sampler_inputs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene_path = args.output_dir / "scene_encoder.onnx"
    decoder_path = args.output_dir / "trajectory_decoder.onnx"
    sampler_path = args.output_dir / "diffusion_planner_sampler.onnx"
    _export(
        scene_wrapper,
        scene_inputs,
        scene_path,
        SCENE_INPUT_NAMES,
        SCENE_OUTPUT_NAMES,
        args.opset_version,
    )
    _export(
        decoder_wrapper,
        decoder_inputs,
        decoder_path,
        DECODER_INPUT_NAMES,
        ("x0_prediction",),
        args.opset_version,
    )
    _export(
        sampler_wrapper,
        sampler_inputs,
        sampler_path,
        SAMPLER_INPUT_NAMES,
        ("trajectory",),
        args.opset_version,
        optimize=False,
    )
    if not args.skip_validation:
        _validate(scene_path, SCENE_INPUT_NAMES, scene_inputs, scene_outputs)
        _validate(
            decoder_path,
            DECODER_INPUT_NAMES,
            decoder_inputs,
            (decoder_output,),
        )
        _validate(
            sampler_path,
            SAMPLER_INPUT_NAMES,
            sampler_inputs,
            (sampler_output,),
        )


if __name__ == "__main__":
    main()
