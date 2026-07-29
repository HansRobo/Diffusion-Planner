import argparse
import random
import subprocess
import sys
import tempfile
from pathlib import Path

# A standalone invocation from ``ros_scripts/`` does not automatically put the
# checkout's Python package (``diffusion_planner/``) ahead of an editable install
# from another worktree.  Prefer the model/exporter code from this checkout so
# checkpoints are always loaded with the matching architecture.
_CHECKOUT_SOURCE = Path(__file__).resolve().parents[1] / "diffusion_planner"
if _CHECKOUT_SOURCE.is_dir():
    sys.path.insert(0, str(_CHECKOUT_SOURCE))

import numpy as np  # noqa: E402  (import follows the sys.path bootstrap above)
import torch  # noqa: E402  (import follows the sys.path bootstrap above)
from diffusion_planner.dimensions import (  # noqa: E402  (import follows the sys.path bootstrap above)
    OUTPUT_T,
    POSE_DIM,
)
from diffusion_planner.train_epoch import heading_to_cos_sin  # noqa: E402
from diffusion_planner.utils.onnx_export import (  # noqa: E402  (import follows the sys.path bootstrap above)
    DECODER_INPUT_NAMES,
    ENCODER_INPUT_NAMES,
    FULL_INPUT_NAMES,
    ModelWrappers,
    NumpyDict,
    TensorDict,
    build_dummy_inputs,
    build_turn_indicator_final_x0,
    build_wrappers,
    export_model_to_onnx,
    load_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", type=Path)
    parser.add_argument("--eval_npz", type=Path, default=None)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="diffusion_planner",
        help="Prefix for output ONNX files (default: diffusion_planner)",
    )
    parser.add_argument(
        "--output-name",
        dest="output_prefix",
        type=str,
        default=argparse.SUPPRESS,
        help="Alias for --output-prefix, kept for torch2onnx.py compatibility",
    )
    parser.add_argument(
        "--use-simplify",
        action="store_true",
        help="Run onnxsim to produce simplified ONNX models",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=20,
        help="ONNX opset version used for export (default: 20)",
    )
    parser.add_argument(
        "--external-data",
        action="store_true",
        help="Export ONNX with external data (weights stored as separate files)",
    )
    return parser.parse_args()


def build_inputs_from_npz(npz_path: Path, action_agent_num: int = 1) -> TensorDict:
    required_keys = set(FULL_INPUT_NAMES) - {"sampled_trajectories"}
    with np.load(npz_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in required_keys if key in archive.files}
    inputs = {}
    inputs["sampled_trajectories"] = 0.5 * torch.randn(
        1, action_agent_num, OUTPUT_T, POSE_DIM, dtype=torch.float32
    )
    # A zero ego-frame pose is a valid stationary origin, not padding.
    inputs["ego_agent_past"] = heading_to_cos_sin(
        torch.tensor(data["ego_agent_past"], dtype=torch.float32).unsqueeze(0)
    )
    inputs["ego_current_state"] = torch.tensor(
        data["ego_current_state"], dtype=torch.float32
    ).unsqueeze(0)
    inputs["neighbor_agents_past"] = torch.tensor(
        data["neighbor_agents_past"], dtype=torch.float32
    ).unsqueeze(0)
    inputs["static_objects"] = torch.tensor(data["static_objects"], dtype=torch.float32).unsqueeze(
        0
    )
    inputs["lanes"] = torch.tensor(data["lanes"], dtype=torch.float32).unsqueeze(0)
    inputs["lanes_speed_limit"] = torch.tensor(
        data["lanes_speed_limit"], dtype=torch.float32
    ).unsqueeze(0)
    inputs["lanes_has_speed_limit"] = torch.tensor(
        data["lanes_has_speed_limit"], dtype=torch.bool
    ).unsqueeze(0)
    inputs["route_lanes"] = torch.tensor(data["route_lanes"], dtype=torch.float32).unsqueeze(0)
    inputs["route_lanes_speed_limit"] = torch.tensor(
        data["route_lanes_speed_limit"], dtype=torch.float32
    ).unsqueeze(0)
    inputs["route_lanes_has_speed_limit"] = torch.tensor(
        data["route_lanes_has_speed_limit"], dtype=torch.bool
    ).unsqueeze(0)
    inputs["polygons"] = torch.tensor(data["polygons"], dtype=torch.float32).unsqueeze(0)
    inputs["line_strings"] = torch.tensor(data["line_strings"], dtype=torch.float32).unsqueeze(0)
    goal_pose = torch.tensor(data["goal_pose"], dtype=torch.float32).unsqueeze(0)
    # The route goal may also coincide with the current origin.
    goal_pose = heading_to_cos_sin(goal_pose)
    inputs["goal_pose"] = goal_pose
    ego_shape = data["ego_shape"] if "ego_shape" in data else np.array([2.75, 4.34, 1.70])
    inputs["ego_shape"] = torch.tensor(ego_shape, dtype=torch.float32).reshape(1, 3)
    return inputs


def load_validation_inputs(eval_npz_path: Path | None, action_agent_num: int = 1) -> TensorDict:
    if eval_npz_path:
        return build_inputs_from_npz(eval_npz_path, action_agent_num)
    return build_dummy_inputs(action_agent_num)


def run_ort_in_subprocess(model_path: Path, np_inputs: NumpyDict) -> list[np.ndarray]:
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = f"{tmpdir}/inputs.npz"
        output_path = f"{tmpdir}/outputs.npz"
        np.savez(input_path, **np_inputs)

        script = f"""
import numpy as np
import onnxruntime as ort
with np.load("{input_path}", allow_pickle=False) as data:
    inputs = {{k: data[k] for k in data.files}}
sess_options = ort.SessionOptions()
sess_options.log_severity_level = 3
sess = ort.InferenceSession(
    "{model_path}", sess_options,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print("ORT providers:", sess.get_providers())
outputs = sess.run(None, inputs)
np.savez("{output_path}", **{{f"out_{{i}}": o for i, o in enumerate(outputs)}})
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ORT subprocess failed:\n{result.stderr[-1000:]}")
        print(result.stdout.strip())
        with np.load(output_path, allow_pickle=False) as data:
            return [data[f"out_{i}"] for i in range(len(data.files))]


def compare(name: str, torch_output: np.ndarray, onnx_output: np.ndarray) -> None:
    abs_diff = np.abs(torch_output - onnx_output)
    close = np.allclose(torch_output, onnx_output, rtol=1e-03, atol=2e-03)
    print(f"{name}: torch={torch_output.shape}, onnx={onnx_output.shape}")
    print(f"  Max diff: {abs_diff.max()}")
    print(f"  Mean diff: {abs_diff.mean()}")
    print(f"  Close? {close}")
    if not close:
        raise AssertionError(
            f"ONNX parity failed for {name}: max_abs={abs_diff.max():.6g}, "
            f"mean_abs={abs_diff.mean():.6g}"
        )


def validate_full_model(
    wrappers: ModelWrappers,
    inputs: TensorDict,
    full_onnx_path: Path,
) -> None:
    with torch.no_grad():
        torch_prediction, torch_turn_indicator = wrappers.full(
            *(inputs[name] for name in FULL_INPUT_NAMES)
        )

    full_onnx_inputs = {name: inputs[name].cpu().numpy() for name in FULL_INPUT_NAMES}
    onnx_prediction, onnx_turn_indicator = run_ort_in_subprocess(full_onnx_path, full_onnx_inputs)
    compare("prediction", torch_prediction.cpu().numpy(), onnx_prediction)
    compare(
        "full_turn_indicator_logit",
        torch_turn_indicator.cpu().numpy(),
        onnx_turn_indicator,
    )


def validate_split_models(
    wrappers: ModelWrappers,
    model,
    inputs: TensorDict,
    encoder_onnx_path: Path,
    decoder_onnx_path: Path,
    turn_indicator_onnx_path: Path,
) -> None:
    with torch.no_grad():
        torch_encoding, torch_global_route_condition = wrappers.encoder(
            *(inputs[name] for name in ENCODER_INPUT_NAMES)
        )
    encoder_onnx_inputs = {name: inputs[name].cpu().numpy() for name in ENCODER_INPUT_NAMES}
    onnx_encoder_outputs = run_ort_in_subprocess(encoder_onnx_path, encoder_onnx_inputs)
    if len(onnx_encoder_outputs) != 2:
        raise AssertionError(
            "Split encoder must output encoding and global_route_condition; "
            f"got {len(onnx_encoder_outputs)} outputs"
        )
    onnx_encoding, onnx_global_route_condition = onnx_encoder_outputs
    compare("encoding", torch_encoding.cpu().numpy(), onnx_encoding)
    compare(
        "global_route_condition",
        torch_global_route_condition.cpu().numpy(),
        onnx_global_route_condition,
    )

    decoder_inputs = {
        "encoding": torch_encoding,
        "sampled_trajectories": inputs["sampled_trajectories"],
        "diffusion_time": torch.ones(
            torch_encoding.shape[0], dtype=torch.float32, device=torch_encoding.device
        ),
        "ego_current_state": inputs["ego_current_state"],
        "global_route_condition": torch_global_route_condition,
    }
    with torch.no_grad():
        torch_model_output = wrappers.decoder(
            *(decoder_inputs[name] for name in DECODER_INPUT_NAMES)
        )
    decoder_onnx_inputs = {name: decoder_inputs[name].cpu().numpy() for name in DECODER_INPUT_NAMES}
    onnx_model_output = run_ort_in_subprocess(decoder_onnx_path, decoder_onnx_inputs)[0]
    compare("model_output", torch_model_output.cpu().numpy(), onnx_model_output)

    final_x0 = build_turn_indicator_final_x0(model, inputs)
    with torch.no_grad():
        torch_turn_indicator = wrappers.turn_indicator(
            torch_encoding,
            final_x0,
            torch_global_route_condition,
            inputs["ego_current_state"],
        )

    turn_indicator_onnx_inputs = {
        "encoding": onnx_encoding,
        "final_x0": final_x0.cpu().numpy(),
        "global_route_condition": onnx_global_route_condition,
        "ego_current_state": inputs["ego_current_state"].cpu().numpy(),
    }
    onnx_turn_indicator = run_ort_in_subprocess(
        turn_indicator_onnx_path, turn_indicator_onnx_inputs
    )[0]
    compare("turn_indicator_logit", torch_turn_indicator.cpu().numpy(), onnx_turn_indicator)


def convert_model(
    config_json_path: str,
    ckpt_path: str,
    full_onnx_path: Path,
    encoder_onnx_path: Path,
    decoder_onnx_path: Path,
    turn_indicator_onnx_path: Path,
    eval_npz_path: Path | None,
    use_ema: bool,
    use_simplify: bool,
    opset_version: int,
    external_data: bool,
) -> None:
    print(f"\n{'=' * 80}")
    print(f"Converting: {ckpt_path}")
    print(f"Config: {config_json_path}")
    print(f"Full output: {full_onnx_path}")
    print(f"Encoder output: {encoder_onnx_path}")
    print(f"Decoder output: {decoder_onnx_path}")
    print(f"Turn indicator output: {turn_indicator_onnx_path}")
    print(f"Using EMA: {use_ema}")
    print("ONNX exporter: legacy")
    print(f"ONNX opset version: {opset_version}")
    print(f"{'=' * 80}\n")

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = load_model(config_json_path, ckpt_path, use_ema)

    export_model_to_onnx(
        model,
        full_onnx_path,
        encoder_onnx_path,
        turn_indicator_onnx_path,
        use_simplify,
        opset_version,
        external_data,
    )

    print("\nORT validation")
    wrappers = build_wrappers(model)
    action_agent_num = 1 + model.decoder._predicted_neighbor_num
    validation_inputs = load_validation_inputs(eval_npz_path, action_agent_num)
    validation_inputs = model.decoder._observation_normalizer(validation_inputs)
    validate_full_model(wrappers, validation_inputs, full_onnx_path)
    validate_split_models(
        wrappers,
        model,
        validation_inputs,
        encoder_onnx_path,
        decoder_onnx_path,
        turn_indicator_onnx_path,
    )

    exported_paths = [
        full_onnx_path,
        encoder_onnx_path,
        decoder_onnx_path,
        turn_indicator_onnx_path,
    ]
    print("\nSuccessfully converted to ONNX:" + "".join(f"\n  {p}" for p in exported_paths) + "\n")


if __name__ == "__main__":
    args = parse_args()
    root_dir = Path(args.root_dir)

    if not root_dir.exists():
        print(f"Error: Directory '{root_dir}' does not exist")
        exit(1)
    if not root_dir.is_dir():
        print(f"Error: '{root_dir}' is not a directory")
        exit(1)

    pth_files = list(root_dir.rglob("*.pth"))
    print(f"Found {len(pth_files)} .pth file(s) in '{root_dir}'")

    skipped_count = 0
    for pth_file in pth_files:
        pth_dir = pth_file.parent
        config_file = pth_dir / "args.json"
        full_onnx_file = pth_dir / f"{args.output_prefix}.onnx"
        encoder_onnx_file = pth_dir / f"{args.output_prefix}_encoder.onnx"
        decoder_onnx_file = pth_dir / f"{args.output_prefix}_decoder.onnx"
        turn_indicator_onnx_file = pth_dir / f"{args.output_prefix}_turn_indicator.onnx"

        print(f"\n{'#' * 80}")
        print(f"Processing: {pth_file.relative_to(root_dir)}")

        if not config_file.exists():
            print(f"Skipping: args.json not found in {pth_dir}")
            skipped_count += 1
            continue

        convert_model(
            config_json_path=str(config_file),
            ckpt_path=str(pth_file),
            full_onnx_path=full_onnx_file,
            encoder_onnx_path=encoder_onnx_file,
            decoder_onnx_path=decoder_onnx_file,
            turn_indicator_onnx_path=turn_indicator_onnx_file,
            eval_npz_path=args.eval_npz,
            use_ema=args.use_ema,
            use_simplify=args.use_simplify,
            opset_version=args.opset_version,
            external_data=args.external_data,
        )

    print(f"\n{'=' * 80}")
    print("Conversion Summary:")
    print(f"  Total found: {len(pth_files)}")
    print(f"  Skipped (no args.json): {skipped_count}")
    print(f"{'=' * 80}")
