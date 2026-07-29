#!/usr/bin/env python3
"""Real-scene PlannerRFT guidance response audit for Original DP.

This extends the colleague anchor-multimodality workflow with the actual
lateral/longitudinal guidance family proposed by PlannerRFT.  Every command is
run with the same diffusion noise, then scored and rendered with the existing
GRPO/AWR map, synchronized-neighbour and OBB primitives.

The tool is diagnostic only: it never changes a checkpoint or replay cache.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from diffusion_planner.utils.neighbor_future_alignment import (  # noqa: E402
    get_neighbor_future_offset,
)

from guidance_gui.generate_samples import generate_samples  # noqa: E402
from preference_optimization.utils import (  # noqa: E402
    X2_LEGACY_EGO_WIDTH_M,
    load_npz_data,
)
from rlvr.autoresearch.tools.grpo_viz import (  # noqa: E402
    diversity_metrics,
    draw_grpo_scene,
    render_dynamics_gif,
    score_trajectory_group,
)
from rlvr.awr import load_original_dp_checkpoint  # noqa: E402
from rlvr.closed_loop.batched_rollout import make_initial_latent  # noqa: E402
from rlvr.guidance_batched import build_head_composer  # noqa: E402
from rlvr.reward import RewardConfig  # noqa: E402
from rlvr.train_awr import (  # noqa: E402
    _compile_planner_modules,
    _configure_cuda_runtime,
)

DEFAULT_COMMANDS = [
    ("U", None, None),
    ("C", 0.0, 0.0),
    ("L+", 1.0, 0.0),
    ("L-", -1.0, 0.0),
    ("V+", 0.0, 1.0),
    ("V-", 0.0, -1.0),
    ("L+V+", 0.7, 0.7),
    ("L+V-", 0.7, -0.7),
    ("L-V+", -0.7, 0.7),
    ("L-V-", -0.7, -0.7),
]


def _load_reward_config(path: Path) -> RewardConfig:
    payload = json.loads(path.read_text())
    payload = payload.get("reward", payload)
    allowed = {field.name for field in fields(RewardConfig)}
    return RewardConfig(**{key: value for key, value in payload.items() if key in allowed})


def _same_noise_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _gt_path_length(data: dict[str, torch.Tensor]) -> float:
    gt = data["ego_agent_future"][0, :, :2].detach().float()
    valid = gt.abs().sum(dim=-1) > 0.1
    return float(torch.diff(gt[valid], dim=0).norm(dim=-1).sum().item()) if valid.sum() > 1 else 0.0


def _response_metrics(
    trajectory: np.ndarray,
    reference: np.ndarray,
    unguided: np.ndarray,
) -> dict[str, float]:
    """Measure the realized command response in the reference Frenet frame."""

    horizon = min(len(trajectory), len(reference))
    trajectory = np.asarray(trajectory[:horizon], dtype=np.float64)
    reference = np.asarray(reference[:horizon], dtype=np.float64)
    tangent = reference[:, 2:4]
    tangent /= np.maximum(np.linalg.norm(tangent, axis=-1, keepdims=True), 1.0e-8)
    normal = np.stack((-tangent[:, 1], tangent[:, 0]), axis=-1)
    delta = trajectory[:, :2] - reference[:, :2]
    lateral = np.sum(delta * normal, axis=-1)
    longitudinal = np.sum(delta * tangent, axis=-1)
    tail = slice(max(0, horizon - 10), horizon)
    return {
        "first_waypoint_from_reference_m": float(np.linalg.norm(delta[0])),
        "first_waypoint_from_unguided_m": float(
            np.linalg.norm(trajectory[0, :2] - np.asarray(unguided)[0, :2])
        ),
        "first_waypoint_position_norm_m": float(np.linalg.norm(trajectory[0, :2])),
        "tail_mean_lateral_from_reference_m": float(np.mean(lateral[tail])),
        "endpoint_lateral_from_reference_m": float(lateral[-1]),
        "tail_mean_longitudinal_from_reference_m": float(np.mean(longitudinal[tail])),
        "endpoint_longitudinal_from_reference_m": float(longitudinal[-1]),
        "endpoint_distance_from_reference_m": float(np.linalg.norm(delta[-1])),
        "endpoint_x_m": float(trajectory[-1, 0]),
        "endpoint_y_m": float(trajectory[-1, 1]),
    }


def _expand_scene_batch(data: dict[str, torch.Tensor], batch_size: int) -> dict[str, torch.Tensor]:
    """Expand one normalized scene without changing any tensor values."""

    return {
        key: (
            value.expand(batch_size, *value.shape[1:]).contiguous()
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == 1
            else value
        )
        for key, value in data.items()
    }


@torch.no_grad()
def _generate_from_fixed_latent(model, data, latent, composer) -> torch.Tensor:
    """Run one batched sampler call while restoring decoder guidance exactly."""

    from rlvr.guidance_batched import dit_memo

    original_fn = model.decoder._guidance_fn
    original_scale = model.decoder._guidance_scale
    model.decoder._guidance_fn = composer
    model.decoder._guidance_scale = (
        composer._set_config.global_scale if composer is not None else 0.5
    )
    data["sampled_trajectories"] = latent
    try:
        if composer is None:
            _, output = model(data)
        else:
            with dit_memo(model.decoder):
                _, output = model(data)
        return output["prediction"][:, 0].detach()
    finally:
        model.decoder._guidance_fn = original_fn
        model.decoder._guidance_scale = original_scale


@torch.no_grad()
def generate_response_group(
    model,
    model_args,
    reference_model,
    reference_model_args,
    data: dict[str, torch.Tensor],
    *,
    seed: int,
    noise_scale: float,
    lambda_lat: float,
    lambda_lon: float,
    guidance_scale: float,
    ramp_steps: int,
) -> tuple[torch.Tensor, list[dict], torch.Tensor]:
    norm = model_args.observation_normalizer(
        {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in data.items()
        }
    )
    reference_norm = reference_model_args.observation_normalizer(
        {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in data.items()
        }
    )
    _same_noise_seed(seed + 10_000_019)
    reference = generate_samples(
        reference_model,
        reference_model_args,
        reference_norm,
        0.0,
        1,
        None,
        next(reference_model.parameters()).device,
    )[0]
    norm["reference_trajectory"] = torch.as_tensor(
        reference[None], dtype=torch.float32, device=next(model.parameters()).device
    )

    commands = [
        {
            "label": label,
            "eta_lat": eta_lat,
            "eta_lon": eta_lon,
            "target_lateral_m": None if eta_lat is None else lambda_lat * eta_lat,
            "target_speed_scale": None if eta_lon is None else 1.0 + lambda_lon * eta_lon,
        }
        for label, eta_lat, eta_lon in DEFAULT_COMMANDS
    ]

    # One shared latent isolates the guidance command from diffusion-noise
    # variation. Keep the U control in a genuinely unguided composer=None call;
    # eta=0 still centers a candidate around x_ref and is not an identity.
    device = next(model.parameters()).device
    _same_noise_seed(seed)
    latent = make_initial_latent(
        1,
        1 + model_args.predicted_neighbor_num,
        model_args.future_len,
        device,
        noise_scale,
    )
    unguided = _generate_from_fixed_latent(model, norm, latent.clone(), None)

    guided_commands = DEFAULT_COMMANDS[1:]
    guided_batch = _expand_scene_batch(norm, len(guided_commands))
    eta_lat = torch.tensor(
        [command[1] for command in guided_commands], device=device, dtype=torch.float32
    )
    eta_lon = torch.tensor(
        [command[2] for command in guided_commands], device=device, dtype=torch.float32
    )
    composer = build_head_composer(
        {"lateral": eta_lat, "stretch": eta_lon},
        lambda_lat=lambda_lat,
        lat_scale=1.0,
        lambda_spd=lambda_lon,
        stretch_scale=1.0,
        guidance_scale=guidance_scale,
        envelope="v2",
        ramp_steps=ramp_steps,
        fast=True,
    )
    guided = _generate_from_fixed_latent(
        model,
        guided_batch,
        latent.expand(len(guided_commands), *latent.shape[1:]).contiguous(),
        composer,
    )
    return (
        torch.cat([unguided, guided], dim=0),
        commands,
        torch.as_tensor(reference, dtype=torch.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--args_path", type=Path, required=True)
    parser.add_argument(
        "--reference_model_path",
        type=Path,
        help="frozen IL reference checkpoint; defaults to --model_path",
    )
    parser.add_argument(
        "--reference_args_path",
        type=Path,
        help="frozen reference args; defaults to --args_path",
    )
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--reward_config", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="*")
    parser.add_argument("--num_scenes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--noise_scale", type=float, default=0.5)
    parser.add_argument("--lambda_lat", type=float, default=1.0)
    parser.add_argument("--lambda_lon", type=float, default=0.25)
    parser.add_argument("--guidance_scale", type=float, default=0.5)
    parser.add_argument("--ramp_steps", type=int, default=20)
    parser.add_argument("--disable_compile", action="store_true")
    parser.add_argument("--compile_mode", default="default")
    parser.add_argument("--gif", action="store_true")
    args = parser.parse_args()

    neighbor_future_offset = get_neighbor_future_offset()
    if neighbor_future_offset != 1:
        raise RuntimeError(
            "this pilot is for the known legacy data and requires "
            f"DP_NEIGHBOR_FUTURE_OFFSET=1, got {neighbor_future_offset}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _configure_cuda_runtime(device)
    model, model_args = load_original_dp_checkpoint(
        args.model_path, device, args_path=args.args_path, use_ema=True
    )
    reference_model_path = args.reference_model_path or args.model_path
    reference_args_path = args.reference_args_path or args.args_path
    if (
        reference_model_path.resolve() == args.model_path.resolve()
        and reference_args_path.resolve() == args.args_path.resolve()
    ):
        reference_model, reference_model_args = model, model_args
    else:
        reference_model, reference_model_args = load_original_dp_checkpoint(
            reference_model_path,
            device,
            args_path=reference_args_path,
            use_ema=True,
        )
    model.eval()
    reference_model.eval()
    if not args.disable_compile and device.type == "cuda":
        compile_config = {
            "enabled": True,
            "backend": "inductor",
            "mode": args.compile_mode,
            "fullgraph": False,
            "dynamic": False,
            "encoder": False,
            "decoder": True,
        }
        _compile_planner_modules(model, compile_config, "plannerrft_policy_probe")
        if reference_model is not model:
            _compile_planner_modules(
                reference_model,
                compile_config,
                "plannerrft_frozen_reference_probe",
            )
    reward_config = _load_reward_config(args.reward_config)
    scene_paths = json.loads(args.scenes.read_text())
    indices = args.indices or list(range(min(args.num_scenes, len(scene_paths))))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "model_path": str(args.model_path.resolve()),
        "args_path": str(args.args_path.resolve()),
        "reference_model_path": str(reference_model_path.resolve()),
        "reference_args_path": str(reference_args_path.resolve()),
        "reference_is_policy": reference_model is model,
        "reference_noise_scale": 0.0,
        "compile_enabled": not args.disable_compile and device.type == "cuda",
        "compile_mode": args.compile_mode,
        "scenes": str(args.scenes.resolve()),
        "same_noise_across_commands": True,
        "neighbor_future_offset": neighbor_future_offset,
        "x2_ego_width_m": 2.29156,
        "noise_scale": args.noise_scale,
        "lambda_lat": args.lambda_lat,
        "lambda_lon": args.lambda_lon,
        "guidance_scale": args.guidance_scale,
        "ramp_steps": args.ramp_steps,
        "lateral_onset": "quintic_smootherstep",
        "longitudinal_semantics": "speed_scale=1+lambda_lon*eta_lon",
        "scenes_results": [],
    }
    for order, scene_index in enumerate(indices):
        scene_path = scene_paths[scene_index]
        data = load_npz_data(scene_path, device)
        ego_width_m = float(data["ego_shape"][0, 2].item())
        if {"x2", "x2_dev"}.intersection(Path(scene_path).parts) and not np.isclose(
            ego_width_m, X2_LEGACY_EGO_WIDTH_M, rtol=0.0, atol=1e-6
        ):
            raise RuntimeError(
                f"scene {scene_path} uses X2 width {ego_width_m}, expected "
                f"legacy checkpoint width {X2_LEGACY_EGO_WIDTH_M}"
            )
        trajectories, commands, reference = generate_response_group(
            model,
            model_args,
            reference_model,
            reference_model_args,
            data,
            seed=args.seed + scene_index * 1009,
            noise_scale=args.noise_scale,
            lambda_lat=args.lambda_lat,
            lambda_lon=args.lambda_lon,
            guidance_scale=args.guidance_scale,
            ramp_steps=args.ramp_steps,
        )
        results = score_trajectory_group(trajectories, data, reward_config)
        unguided_trajectory = trajectories[0].detach().cpu().numpy()
        for result in results:
            result["candidate_label"] = commands[int(result["k"])]["label"]
            result.update(
                _response_metrics(
                    result["traj"],
                    reference.numpy(),
                    unguided_trajectory,
                )
            )
        diversity = diversity_metrics(results)
        title = f"PlannerRFT response · scene {scene_index} · same diffusion noise"
        figure = plt.figure(figsize=(20, 11), facecolor="#f8fafc")
        draw_grpo_scene(
            figure,
            results,
            scene_path,
            scene_index,
            _gt_path_length(data),
            "PLANNERRFT GUIDANCE PROBE",
            scene_title=title,
            diversity=diversity,
            reference_trajectory=reference.numpy(),
        )
        png = args.output_dir / f"scene_{order:03d}_{scene_index:06d}.png"
        figure.savefig(png, dpi=180, bbox_inches="tight")
        plt.close(figure)
        gif = None
        if args.gif:
            gif = args.output_dir / f"scene_{order:03d}_{scene_index:06d}.gif"
            render_dynamics_gif(
                results,
                scene_path,
                gif,
                scene_index,
                "PLANNERRFT GUIDANCE PROBE",
                scene_title=title,
                reference_trajectory=reference.numpy(),
            )
        reference_path = (
            args.output_dir / f"scene_{order:03d}_{scene_index:06d}_reference.npy"
        )
        np.save(reference_path, reference.numpy())
        trajectory_path = (
            args.output_dir / f"scene_{order:03d}_{scene_index:06d}_trajectories.npz"
        )
        np.savez_compressed(
            trajectory_path,
            trajectories=trajectories.detach().cpu().numpy(),
            reference=reference.numpy(),
            labels=np.asarray([command["label"] for command in commands]),
        )
        manifest["scenes_results"].append(
            {
                "scene_index": scene_index,
                "scene_path": scene_path,
                "effective_ego_width_m": ego_width_m,
                "png": png.name,
                "gif": None if gif is None else gif.name,
                "reference_trajectory": reference_path.name,
                "candidate_trajectories": trajectory_path.name,
                "commands": commands,
                "diversity": diversity,
                "ranked_candidates": [
                    {
                        key: value
                        for key, value in result.items()
                        if key
                        in {
                            "k",
                            "candidate_label",
                            "total",
                            "progress",
                            "ttc_score",
                            "min_obb_clearance_m",
                            "rb_min_dist_m",
                            "rb_crossing",
                            "collision",
                            "kinematic",
                            "red_light",
                            "hard_reasons",
                            "path_len",
                            "first_waypoint_from_reference_m",
                            "first_waypoint_from_unguided_m",
                            "first_waypoint_position_norm_m",
                            "tail_mean_lateral_from_reference_m",
                            "endpoint_lateral_from_reference_m",
                            "tail_mean_longitudinal_from_reference_m",
                            "endpoint_longitudinal_from_reference_m",
                            "endpoint_distance_from_reference_m",
                            "endpoint_x_m",
                            "endpoint_y_m",
                        }
                    }
                    for result in results
                ],
            }
        )
        print(f"rendered {scene_index}: {png}")

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
