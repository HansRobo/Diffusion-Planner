"""Reusable ONNX export core for Diffusion-Planner.

This module holds the model wrappers and the export machinery shared between the standalone
``ros_scripts/torch2onnx.py`` CLI and the training entrypoints (``train_predictor.py`` /
``train_hdp_rl_predictor.py``), which export ONNX next to each saved checkpoint.

Unlike ``torch2onnx.py``, this module does NOT toggle the SDPA / MHA backends at import time.
The backends are only forced (and restored) inside :func:`onnx_export_backends`, scoped around
the export itself, so importing this module from a training process never slows training down.
"""

import contextlib
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from diffusion_planner.dimensions import *
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config

FULL_INPUT_NAMES = [
    "sampled_trajectories",
    "ego_agent_past",
    "ego_current_state",
    "neighbor_agents_past",
    "static_objects",
    "lanes",
    "lanes_speed_limit",
    "lanes_has_speed_limit",
    "route_lanes",
    "route_lanes_speed_limit",
    "route_lanes_has_speed_limit",
    "polygons",
    "line_strings",
    "goal_pose",
    "ego_shape",
]

ENCODER_INPUT_NAMES = [
    "ego_agent_past",
    "neighbor_agents_past",
    "static_objects",
    "lanes",
    "lanes_speed_limit",
    "lanes_has_speed_limit",
    "route_lanes",
    "route_lanes_speed_limit",
    "route_lanes_has_speed_limit",
    "polygons",
    "line_strings",
    "goal_pose",
    "ego_shape",
]

TURN_INDICATOR_INPUT_NAMES = [
    "encoding",
    "final_x0",
    "global_route_condition",
    "ego_current_state",
]

FULL_OUTPUT_NAMES = ["prediction", "turn_indicator_logit"]
# The split turn-indicator graph consumes the same route AdaLN condition as the
# Python decoder.  Export it from the encoder-side graph so the deployment node
# does not have to reimplement a learned route encoder.
ENCODER_OUTPUT_NAMES = ["encoding", "global_route_condition"]
TURN_INDICATOR_OUTPUT_NAMES = ["turn_indicator_logit"]

TensorDict = dict[str, torch.Tensor]
NumpyDict = dict[str, np.ndarray]


def validate_hdp_export_contract(config: Config) -> None:
    """Reject legacy signal-conditioned checkpoints before ONNX tracing.

    The velocity-based HDP deployed by the ROS node has no turn-indicator
    history input.  Older Python checkouts still expose ``turn_indicators``
    and can otherwise fail late during tracing (or, worse, export a graph
    with a silently different input protocol).
    """
    if "turn_indicators" in FULL_INPUT_NAMES or "turn_indicators" in ENCODER_INPUT_NAMES:
        raise RuntimeError(
            "Invalid HDP exporter contract: turn_indicators must not be an ONNX input"
        )
    if not bool(getattr(config, "use_velocity_representation", False)):
        raise RuntimeError(
            "This exporter only supports the velocity-based HDP deployment contract; "
            "refusing to export a waypoint-latent checkpoint."
        )
    if getattr(config, "decoder_tokenization", None) != "temporal":
        raise RuntimeError(
            "This exporter requires decoder_tokenization='temporal' for the HDP Node."
        )
    if int(getattr(config, "predicted_neighbor_num", 0)) != 0:
        raise RuntimeError("This exporter requires ego-only output (predicted_neighbor_num=0).")
    if int(getattr(config, "time_len", INPUT_T + 1)) != INPUT_T + 1:
        raise RuntimeError(
            f"This exporter requires time_len={INPUT_T + 1} for the Node input grid."
        )
    if int(getattr(config, "future_len", OUTPUT_T)) != OUTPUT_T:
        raise RuntimeError(
            f"This exporter requires future_len={OUTPUT_T} for the Node output horizon."
        )
    if getattr(config, "policy_uses_turn_indicator_history", False):
        raise RuntimeError(
            "This checkpoint declares policy_uses_turn_indicator_history=True; "
            "it is incompatible with the deployed ego-only HDP input contract."
        )
    output_dim = int(getattr(config, "turn_indicator_output_dim", TURN_INDICATOR_OUTPUT_DIM))
    if output_dim != TURN_INDICATOR_OUTPUT_DIM:
        raise RuntimeError(
            "HDP turn-indicator export requires exactly "
            f"{TURN_INDICATOR_OUTPUT_DIM} logits, got {output_dim}"
        )


@dataclass(frozen=True)
class ModelWrappers:
    full: nn.Module
    encoder: nn.Module
    turn_indicator: nn.Module


@dataclass(frozen=True)
class ExportSpec:
    wrapper: nn.Module
    inputs: TensorDict
    input_names: list[str]
    output_names: list[str]
    output_path: Path


@contextlib.contextmanager
def onnx_export_backends():
    """Force the math SDPA path and disable the MHA fastpath for the duration of an export.

    ``nn.MultiheadAttention`` (used by the encoder / DiT) takes a fused fastpath in eval mode
    that the legacy ONNX exporter cannot trace cleanly; forcing the plain math path makes the
    attention decompose into ONNX-representable ops. These are process-global switches, so they
    are restored on exit to avoid leaking into a surrounding training run.
    """
    prev_flash = torch.backends.cuda.flash_sdp_enabled()
    prev_mem_efficient = torch.backends.cuda.mem_efficient_sdp_enabled()
    prev_math = torch.backends.cuda.math_sdp_enabled()
    prev_fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        yield
    finally:
        torch.backends.cuda.enable_flash_sdp(prev_flash)
        torch.backends.cuda.enable_mem_efficient_sdp(prev_mem_efficient)
        torch.backends.cuda.enable_math_sdp(prev_math)
        torch.backends.mha.set_fastpath_enabled(prev_fastpath)


class EncoderONNXWrapper(nn.Module):
    def __init__(self, model: Diffusion_Planner):
        super().__init__()
        self.encoder = model.encoder
        self.global_route_encoder = model.decoder.global_route_encoder

    def forward(
        self,
        ego_agent_past: torch.Tensor,
        neighbor_agents_past: torch.Tensor,
        static_objects: torch.Tensor,
        lanes: torch.Tensor,
        lanes_speed_limit: torch.Tensor,
        lanes_has_speed_limit: torch.Tensor,
        route_lanes: torch.Tensor,
        route_lanes_speed_limit: torch.Tensor,
        route_lanes_has_speed_limit: torch.Tensor,
        polygons: torch.Tensor,
        line_strings: torch.Tensor,
        goal_pose: torch.Tensor,
        ego_shape: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = {
            "ego_agent_past": ego_agent_past,
            "neighbor_agents_past": neighbor_agents_past,
            "static_objects": static_objects,
            "lanes": lanes,
            "lanes_speed_limit": lanes_speed_limit,
            "lanes_has_speed_limit": lanes_has_speed_limit,
            "route_lanes": route_lanes,
            "route_lanes_speed_limit": route_lanes_speed_limit,
            "route_lanes_has_speed_limit": route_lanes_has_speed_limit,
            "polygons": polygons,
            "line_strings": line_strings,
            "goal_pose": goal_pose,
            "ego_shape": ego_shape,
        }
        return self.encoder(inputs), self.global_route_encoder(route_lanes)


class TurnIndicatorONNXWrapper(nn.Module):
    """Turn-indicator head evaluated once after denoising.

    ``final_x0`` is a waypoint latent for vanilla DP and an HDP velocity latent for the ego row
    when ``use_velocity_representation=True``.
    """

    def __init__(self, model: Diffusion_Planner):
        super().__init__()
        self.decoder = model.decoder

    def forward(
        self,
        encoding: torch.Tensor,
        final_x0: torch.Tensor,
        global_route_condition: torch.Tensor,
        ego_current_state: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = encoding.shape[0]
        agent_num = 1 + self.decoder._predicted_neighbor_num
        final_x0 = final_x0.reshape(batch_size, agent_num, self.decoder._future_len, 4)

        ego_trajectory = self.decoder._turn_indicator_trajectory_from_latent(final_x0)
        return self.decoder._compute_turn_indicator(
            ego_trajectory,
            encoding,
            global_route_condition,
            ego_current_state,
        )


class FullONNXWrapper(nn.Module):
    """Original all-in-one planner export."""

    def __init__(self, model: Diffusion_Planner):
        super().__init__()
        self.model = model

    def forward(
        self,
        sampled_trajectories: torch.Tensor,
        ego_agent_past: torch.Tensor,
        ego_current_state: torch.Tensor,
        neighbor_agents_past: torch.Tensor,
        static_objects: torch.Tensor,
        lanes: torch.Tensor,
        lanes_speed_limit: torch.Tensor,
        lanes_has_speed_limit: torch.Tensor,
        route_lanes: torch.Tensor,
        route_lanes_speed_limit: torch.Tensor,
        route_lanes_has_speed_limit: torch.Tensor,
        polygons: torch.Tensor,
        line_strings: torch.Tensor,
        goal_pose: torch.Tensor,
        ego_shape: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = {
            "sampled_trajectories": sampled_trajectories,
            "ego_agent_past": ego_agent_past,
            "ego_current_state": ego_current_state,
            "neighbor_agents_past": neighbor_agents_past,
            "static_objects": static_objects,
            "lanes": lanes,
            "lanes_speed_limit": lanes_speed_limit,
            "lanes_has_speed_limit": lanes_has_speed_limit,
            "route_lanes": route_lanes,
            "route_lanes_speed_limit": route_lanes_speed_limit,
            "route_lanes_has_speed_limit": route_lanes_has_speed_limit,
            "polygons": polygons,
            "line_strings": line_strings,
            "goal_pose": goal_pose,
            "ego_shape": ego_shape,
        }
        _, decoder_outputs = self.model(inputs)
        return decoder_outputs["prediction"], decoder_outputs["turn_indicator_logit"]


def build_dummy_inputs(action_agent_num: int = 1) -> TensorDict:
    inputs = {}
    inputs["sampled_trajectories"] = torch.ones(
        1, action_agent_num, OUTPUT_T, POSE_DIM, dtype=torch.float32
    )
    inputs["ego_agent_past"] = torch.randn(1, INPUT_T + 1, POSE_DIM, dtype=torch.float32)
    inputs["ego_current_state"] = torch.randn(1, 10, dtype=torch.float32)
    inputs["neighbor_agents_past"] = torch.randn(
        1, MAX_NUM_NEIGHBORS, INPUT_T + 1, 11, dtype=torch.float32
    )
    inputs["static_objects"] = torch.randn(1, NUM_STATIC_OBJECTS, 10, dtype=torch.float32)
    inputs["lanes"] = torch.randn(
        1, NUM_SEGMENTS_IN_LANE, POINTS_PER_LANELET, SEGMENT_POINT_DIM, dtype=torch.float32
    )
    inputs["lanes_speed_limit"] = torch.randn(1, NUM_SEGMENTS_IN_LANE, 1, dtype=torch.float32)
    inputs["lanes_has_speed_limit"] = torch.ones(1, NUM_SEGMENTS_IN_LANE, 1, dtype=torch.bool)
    inputs["route_lanes"] = torch.randn(
        1, NUM_SEGMENTS_IN_ROUTE, POINTS_PER_LANELET, SEGMENT_POINT_DIM, dtype=torch.float32
    )
    inputs["route_lanes_speed_limit"] = torch.randn(
        1, NUM_SEGMENTS_IN_ROUTE, 1, dtype=torch.float32
    )
    inputs["route_lanes_has_speed_limit"] = torch.ones(
        1, NUM_SEGMENTS_IN_ROUTE, 1, dtype=torch.bool
    )
    inputs["polygons"] = torch.randn(
        1, NUM_POLYGONS, POINTS_PER_POLYGON, 2 + POLYGON_TYPE_NUM, dtype=torch.float32
    )
    inputs["line_strings"] = torch.randn(
        1, NUM_LINE_STRINGS, POINTS_PER_LINE_STRING, 2 + LINE_STRING_TYPE_NUM, dtype=torch.float32
    )
    inputs["goal_pose"] = torch.randn(1, POSE_DIM, dtype=torch.float32)
    inputs["ego_shape"] = torch.tensor([[2.75, 4.34, 1.70]], dtype=torch.float32)
    return inputs


def build_turn_indicator_inputs(
    encoding: torch.Tensor,
    final_x0: torch.Tensor,
    global_route_condition: torch.Tensor,
    ego_current_state: torch.Tensor,
) -> TensorDict:
    return {
        "encoding": encoding,
        "final_x0": final_x0,
        "global_route_condition": global_route_condition,
        "ego_current_state": ego_current_state,
    }


def build_turn_indicator_final_x0(model: Diffusion_Planner, inputs: TensorDict) -> torch.Tensor:
    """Build a trace-only latent input for the turn-indicator graph.

    The standalone turn head only needs a correctly shaped velocity latent for export tracing.
    The deployable artifact is the full graph, which runs sampling and latent decoding itself.
    """
    decoder = model.decoder
    batch_size = inputs["sampled_trajectories"].shape[0]
    agent_num = 1 + decoder._predicted_neighbor_num
    return torch.zeros(
        batch_size,
        agent_num,
        decoder._future_len,
        4,
        dtype=inputs["sampled_trajectories"].dtype,
        device=inputs["sampled_trajectories"].device,
    )


def load_model(config_json_path: str, ckpt_path: str, use_ema: bool) -> Diffusion_Planner:
    config_obj = Config(config_json_path)
    validate_hdp_export_contract(config_obj)
    model = Diffusion_Planner(config_obj)
    model.eval()
    model.encoder.eval()
    model.decoder.eval()
    model.decoder.training = False

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if use_ema:
        if "ema_state_dict" not in ckpt or ckpt["ema_state_dict"] is None:
            raise ValueError(f"EMA state dict not found in checkpoint: {ckpt_path}")
        state_dict = ckpt["ema_state_dict"]
        print("Loading EMA model weights")
    else:
        state_dict = ckpt["model"]
        print("Loading regular model weights")
    model.load_state_dict({k.removeprefix("module."): v for k, v in state_dict.items()})
    return model


def build_wrappers(model: Diffusion_Planner) -> ModelWrappers:
    return ModelWrappers(
        full=FullONNXWrapper(model).eval(),
        encoder=EncoderONNXWrapper(model).eval(),
        turn_indicator=TurnIndicatorONNXWrapper(model).eval(),
    )


def build_export_specs(
    wrappers: ModelWrappers,
    inputs: TensorDict,
    turn_indicator_inputs: TensorDict,
    full_onnx_path: Path,
    encoder_onnx_path: Path,
    turn_indicator_onnx_path: Path,
) -> list[ExportSpec]:
    return [
        ExportSpec(
            wrapper=wrappers.full,
            inputs=inputs,
            input_names=FULL_INPUT_NAMES,
            output_names=FULL_OUTPUT_NAMES,
            output_path=full_onnx_path,
        ),
        ExportSpec(
            wrapper=wrappers.encoder,
            inputs=inputs,
            input_names=ENCODER_INPUT_NAMES,
            output_names=ENCODER_OUTPUT_NAMES,
            output_path=encoder_onnx_path,
        ),
        ExportSpec(
            wrapper=wrappers.turn_indicator,
            inputs=turn_indicator_inputs,
            input_names=TURN_INDICATOR_INPUT_NAMES,
            output_names=TURN_INDICATOR_OUTPUT_NAMES,
            output_path=turn_indicator_onnx_path,
        ),
    ]


def build_dynamic_axes(
    input_names: list[str], output_names: list[str]
) -> dict[str, dict[int, str]]:
    dynamic_axes = {name: {0: "batch"} for name in input_names}
    dynamic_axes.update({name: {0: "batch"} for name in output_names})
    return dynamic_axes


def _repair_onnx_output_shapes(
    output_path: Path,
    output_names: list[str],
    reference_outputs: tuple[torch.Tensor, ...],
) -> None:
    """Write concrete non-batch output dimensions after legacy ONNX tracing.

    The legacy exporter can emit ``0`` for a non-batch dimension when that dimension
    flows through a reshape using a symbolic batch size.  ORT executes such a graph
    leniently, but consumers that inspect the graph contract (notably C++ runtimes)
    can reject it.  The wrapper was evaluated with the same trace inputs, so its
    output tensors provide the authoritative dimensions; only axis 0 remains dynamic.
    """
    try:
        import onnx
    except ImportError:
        warnings.warn(
            "onnx is not installed; cannot repair exported output shape metadata",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    if len(output_names) != len(reference_outputs):
        raise RuntimeError(
            f"ONNX output contract has {len(output_names)} names but "
            f"{len(reference_outputs)} reference tensors"
        )
    model = onnx.load(str(output_path))
    graph_outputs = {value.name: value for value in model.graph.output}
    for name, reference in zip(output_names, reference_outputs):
        if name not in graph_outputs:
            raise RuntimeError(f"Exported ONNX output {name!r} is missing")
        if not torch.is_tensor(reference) or reference.ndim == 0:
            raise RuntimeError(f"ONNX output {name!r} must be a non-scalar tensor")
        shape = graph_outputs[name].type.tensor_type.shape
        shape.ClearField("dim")
        for axis, size in enumerate(reference.shape):
            dimension = shape.dim.add()
            if axis == 0:
                dimension.dim_param = "batch"
            else:
                dimension.dim_value = int(size)
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))


def export_onnx(
    wrapper: nn.Module,
    inputs: TensorDict,
    input_names: list[str],
    output_names: list[str],
    output_path: Path,
    use_simplify: bool,
    opset_version: int,
    external_data: bool,
) -> None:
    torch_input_tuple = tuple(inputs[name] for name in input_names)
    export_kwargs: dict[str, Any] = {
        "input_names": input_names,
        "output_names": output_names,
        "opset_version": opset_version,
        "dynamo": False,
        "dynamic_axes": build_dynamic_axes(input_names, output_names),
        "external_data": external_data,
    }

    print(f"Creating ONNX model with legacy exporter: {output_path}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        torch.onnx.export(
            wrapper,
            torch_input_tuple,
            str(output_path),
            **export_kwargs,
        )

    # Evaluate the wrapper once after tracing to repair output metadata that the
    # legacy exporter occasionally loses through symbolic reshapes. This does not
    # alter graph computation, only its declared non-batch dimensions.
    with torch.no_grad():
        traced_outputs = wrapper(*torch_input_tuple)
    if torch.is_tensor(traced_outputs):
        traced_outputs = (traced_outputs,)
    else:
        traced_outputs = tuple(traced_outputs)
    _repair_onnx_output_shapes(output_path, output_names, traced_outputs)

    if use_simplify:
        try:
            import onnx
            from onnxsim import simplify

            model_proto = onnx.load(str(output_path))
            model_simp, check = simplify(model_proto)
            if check:
                onnx.save(model_simp, str(output_path))
                print(f"Simplified ONNX saved: {output_path}")
            else:
                print("WARNING: onnxsim validation failed, keeping unsimplified model")
        except ImportError:
            print("WARNING: onnxsim not installed, skipping (pip install onnxsim)")
        except Exception as exc:
            print(f"WARNING: onnxsim failed ({exc}), keeping unsimplified model")


def export_spec(
    spec: ExportSpec,
    use_simplify: bool,
    opset_version: int,
    external_data: bool,
) -> None:
    export_onnx(
        spec.wrapper,
        spec.inputs,
        spec.input_names,
        spec.output_names,
        spec.output_path,
        use_simplify,
        opset_version,
        external_data,
    )


def export_model_to_onnx(
    model: Diffusion_Planner,
    full_onnx_path: Path,
    encoder_onnx_path: Path,
    turn_indicator_onnx_path: Path,
    use_simplify: bool,
    opset_version: int,
    external_data: bool,
) -> None:
    """Export ONNX graphs for ``model``.

    HDP exports full / encoder / turn_indicator graphs; the full graph is deployable.

    No ORT validation is performed; the caller is responsible for that (the standalone CLI does,
    the training loop skips it). The SDPA / MHA backends are forced only for the duration of the
    export via :func:`onnx_export_backends` and restored afterwards.
    """
    wrappers = build_wrappers(model)
    export_inputs = build_dummy_inputs(1 + model.decoder._predicted_neighbor_num)

    with onnx_export_backends():
        with torch.no_grad():
            encoding, global_route_condition = wrappers.encoder(
                *(export_inputs[name] for name in ENCODER_INPUT_NAMES)
            )

        final_x0 = build_turn_indicator_final_x0(model, export_inputs)
        turn_indicator_inputs = build_turn_indicator_inputs(
            encoding,
            final_x0,
            global_route_condition,
            export_inputs["ego_current_state"],
        )

        export_specs = build_export_specs(
            wrappers,
            export_inputs,
            turn_indicator_inputs,
            full_onnx_path,
            encoder_onnx_path,
            turn_indicator_onnx_path,
        )
        for spec in export_specs:
            export_spec(spec, use_simplify, opset_version, external_data)


def export_checkpoint_onnx(
    config_json_path: str,
    ckpt_path: str,
    output_dir: Path,
    output_prefix: str,
    use_ema: bool,
    use_simplify: bool,
    opset_version: int,
    external_data: bool,
) -> None:
    """Load a checkpoint from disk (on CPU) and export its ONNX graphs into ``output_dir``.

    Builds a fresh CPU model from the saved ``.pth`` + ``args.json`` so it never touches the
    in-memory (GPU / DDP-wrapped) training model. Intended to be called right after a checkpoint
    is written during training.
    """
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_dir = Path(output_dir)
    model = load_model(config_json_path, ckpt_path, use_ema)
    export_model_to_onnx(
        model,
        output_dir / f"{output_prefix}.onnx",
        output_dir / f"{output_prefix}_encoder.onnx",
        output_dir / f"{output_prefix}_turn_indicator.onnx",
        use_simplify,
        opset_version,
        external_data,
    )


def export_checkpoint_onnx_guarded(
    config_json_path: str,
    ckpt_path: str,
    output_dir: Path,
    output_prefix: str,
    use_ema: bool,
    use_simplify: bool,
    opset_version: int,
    external_data: bool,
) -> None:
    """Same as :func:`export_checkpoint_onnx`, but never raises.

    Intended for the training loops, where an ONNX export is a convenience artifact and must not
    bring the run down. Any failure is reported and swallowed.
    """
    try:
        export_checkpoint_onnx(
            config_json_path,
            ckpt_path,
            output_dir,
            output_prefix,
            use_ema,
            use_simplify,
            opset_version,
            external_data,
        )
    except Exception as exc:
        print(f"WARNING: ONNX export failed for {ckpt_path}: {exc}")
