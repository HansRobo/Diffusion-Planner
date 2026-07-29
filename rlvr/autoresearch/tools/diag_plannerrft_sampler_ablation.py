#!/usr/bin/env python3
"""Paired real-scene ablation of PlannerRFT-style exploration for Original DP.

For every scene this tool constructs one K-member bank of independent DP
latents and reuses it for every arm:

* native: all K members are unguided;
* all_guided: all K members use fixed symmetric-Beta lateral/speed guidance;
* mixed2: two native members and eight guided members (for K=10);
* mixed5: five native members and five guided members (for K=10).

The policy latent tensors, noise temperature, Beta draws, reward implementation
and scene are therefore paired.  The frozen IL reference can be generated
either from the zero latent used by the formal T4 adaptation or from one
Gaussian latent as specified by PlannerRFT supplementary Algorithm 1.  A fixed
reference seed makes separate zero/stochastic-reference invocations directly
comparable.  The mixed arms use a per-sample zero guidance-strength gate for
their native slots.  Those slots are checked numerically against a true
composer=None native forward and the run fails closed if they do not match.

This is an inference/candidate-discovery audit.  It never edits a checkpoint
or replay cache.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import fields
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from diffusion_planner.utils.neighbor_future_alignment import (  # noqa: E402
    NEIGHBOR_FUTURE_OFFSET_ENV,
    get_neighbor_future_offset,
)
from scipy.special import betaincinv  # noqa: E402

from guidance_gui.generate_samples import generate_samples  # noqa: E402
from preference_optimization.utils import (  # noqa: E402
    X2_LEGACY_EGO_WIDTH_M,
    load_npz_data,
)
from rlvr.autoresearch.tools.diag_plannerrft_guidance import (  # noqa: E402
    _expand_scene_batch,
    _generate_from_fixed_latent,
    _gt_path_length,
    _response_metrics,
    _same_noise_seed,
)
from rlvr.autoresearch.tools.grpo_viz import (  # noqa: E402
    diversity_metrics,
    draw_grpo_scene,
    render_dynamics_gif,
    score_trajectory_group,
)
from rlvr.awr import load_original_dp_checkpoint  # noqa: E402
from rlvr.guidance_batched import build_head_composer  # noqa: E402
from rlvr.reward import RewardConfig  # noqa: E402
from rlvr.train_awr import _compile_planner_modules, _configure_cuda_runtime  # noqa: E402

FIXED_BETA_CONCENTRATION = 1.0 + math.log(2.0)
DEFAULT_ARMS = ("native", "all_guided", "mixed2", "mixed5")


def _load_reward_config(path: Path) -> RewardConfig:
    payload = json.loads(path.read_text())
    payload = payload.get("reward", payload)
    allowed = {field.name for field in fields(RewardConfig)}
    return RewardConfig(**{key: value for key, value in payload.items() if key in allowed})


def _json_safe(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _result_record(result: dict) -> dict:
    """Keep every JSON-safe scorer field except the full trajectory array."""

    return {
        key: _json_safe(value)
        for key, value in result.items()
        if key != "traj"
    }


def _fixed_beta_eta(
    seed: int,
    group_size: int,
    scheme: str = "iid_beta",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample the local zero-init Beta prior, optionally with variance reduction.

    The paper specifies a fixed distribution at zero initialization but does
    not publish the exact concentration mapping.  ``iid_beta`` uses this
    repository's zero-logit ``softplus(0)+1`` parameterization.
    ``stratified_beta`` preserves that same one-dimensional distribution while
    putting one draw in every equal-probability stratum for each axis.
    Independent random permutations pair the lateral and longitudinal draws.
    Both the exact concentration and stratification are local adaptations, not
    attributed to PlannerRFT.
    """

    rng = np.random.default_rng(seed)
    if scheme == "iid_beta":
        eta_lat_01 = rng.beta(
            FIXED_BETA_CONCENTRATION,
            FIXED_BETA_CONCENTRATION,
            size=group_size,
        )
        eta_lon_01 = rng.beta(
            FIXED_BETA_CONCENTRATION,
            FIXED_BETA_CONCENTRATION,
            size=group_size,
        )
    elif scheme == "stratified_beta":
        # Random jitter inside every CDF stratum avoids fixed endpoints and
        # makes repeated scene seeds unbiased.  Shuffle each axis separately
        # so candidate k does not always pair slow with left, fast with right.
        q_lat = (np.arange(group_size) + rng.random(group_size)) / group_size
        q_lon = (np.arange(group_size) + rng.random(group_size)) / group_size
        eta_lat_01 = betaincinv(
            FIXED_BETA_CONCENTRATION, FIXED_BETA_CONCENTRATION, q_lat
        )
        eta_lon_01 = betaincinv(
            FIXED_BETA_CONCENTRATION, FIXED_BETA_CONCENTRATION, q_lon
        )
        rng.shuffle(eta_lat_01)
        rng.shuffle(eta_lon_01)
    else:
        raise ValueError(f"unknown eta sampling scheme {scheme!r}")
    eta_lat = 2.0 * eta_lat_01 - 1.0
    eta_lon = 2.0 * eta_lon_01 - 1.0
    return (
        torch.as_tensor(eta_lat, dtype=torch.float32),
        torch.as_tensor(eta_lon, dtype=torch.float32),
    )


def _native_slots(arm: str, group_size: int) -> int:
    if arm == "native":
        return group_size
    if arm == "all_guided":
        return 0
    if arm.startswith("mixed"):
        count = int(arm.removeprefix("mixed"))
        if not 0 <= count <= group_size:
            raise ValueError(f"{arm} requires 0 <= native slots <= K={group_size}")
        return count
    raise ValueError(f"unknown arm {arm!r}")


@torch.no_grad()
def generate_paired_arms(
    model,
    model_args,
    reference_model,
    reference_model_args,
    data: dict[str, torch.Tensor],
    *,
    seed: int,
    group_size: int,
    noise_scale: float,
    arms: tuple[str, ...],
    lambda_lat: float,
    lambda_lon: float,
    longitudinal_mode: str,
    guidance_scale: float,
    ramp_steps: int,
    equivalence_atol_m: float,
    eta_scheme: str,
    reference_mode: str,
    reference_noise_scale: float,
) -> tuple[dict[str, torch.Tensor], list[dict], torch.Tensor, dict]:
    """Generate all arms from exactly the same independent latent bank."""

    device = next(model.parameters()).device
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
    if reference_mode not in {"zero_noise", "stochastic"}:
        raise ValueError(f"unknown frozen-reference mode {reference_mode!r}")
    if reference_mode == "stochastic" and reference_noise_scale <= 0.0:
        raise ValueError(
            "stochastic frozen-reference mode requires reference_noise_scale > 0"
        )
    # Algorithm 1 draws one reference latent per scene.  Seed it independently
    # of both the current-policy candidate bank and the fixed-Beta commands so
    # changing reference mode cannot silently change either paired quantity.
    _same_noise_seed(seed + 10_000_019)
    effective_reference_noise = (
        0.0 if reference_mode == "zero_noise" else float(reference_noise_scale)
    )
    reference = generate_samples(
        reference_model,
        reference_model_args,
        reference_norm,
        effective_reference_noise,
        1,
        None,
        next(reference_model.parameters()).device,
    )[0]
    norm["reference_trajectory"] = torch.as_tensor(
        reference[None], dtype=torch.float32, device=device
    )

    # The released-HDP-compatible T4 AWR configuration uses K independent
    # Gaussian latents at one fixed temperature (0.5), no deterministic slot.
    _same_noise_seed(seed)
    players = 1 + int(model_args.predicted_neighbor_num)
    latent = noise_scale * torch.randn(
        group_size,
        players,
        int(model_args.future_len) + 1,
        4,
        device=device,
    )
    eta_lat_cpu, eta_lon_cpu = _fixed_beta_eta(
        seed + 20_000_033, group_size, eta_scheme
    )
    eta_lat = eta_lat_cpu.to(device)
    eta_lon = eta_lon_cpu.to(device)

    native_batch = _expand_scene_batch(norm, group_size)
    native = _generate_from_fixed_latent(model, native_batch, latent.clone(), None)
    trajectories: dict[str, torch.Tensor] = {"native": native}
    arm_diagnostics: dict[str, dict] = {}

    for arm in arms:
        if arm == "native":
            continue
        native_count = _native_slots(arm, group_size)
        strength = torch.ones(group_size, dtype=torch.float32, device=device)
        strength[:native_count] = 0.0
        longitudinal_key = {
            "candidate_stretch": "stretch",
            "reference_speed_push": "reference_speed",
        }.get(longitudinal_mode)
        if longitudinal_key is None:
            raise ValueError(f"unknown longitudinal mode {longitudinal_mode!r}")
        composer = build_head_composer(
            {"lateral": eta_lat, longitudinal_key: eta_lon},
            lambda_lat=lambda_lat,
            lat_scale=1.0,
            lambda_spd=lambda_lon,
            stretch_scale=1.0,
            guidance_scale=guidance_scale,
            envelope="v2",
            ramp_steps=ramp_steps,
            fast=True,
            guidance_strength=strength,
        )
        guided_batch = _expand_scene_batch(norm, group_size)
        candidate = _generate_from_fixed_latent(
            model,
            guided_batch,
            latent.clone(),
            composer,
        )
        trajectories[arm] = candidate
        if native_count:
            delta = (candidate[:native_count] - native[:native_count]).abs()
            max_abs = float(delta.max().item())
            max_xy = float(
                torch.linalg.vector_norm(
                    candidate[:native_count, :, :2] - native[:native_count, :, :2],
                    dim=-1,
                ).max().item()
            )
            if max_xy > equivalence_atol_m:
                raise RuntimeError(
                    f"{arm} zero-strength slots do not reproduce true unguided "
                    f"sampling: max_xy={max_xy:.9g} m > atol={equivalence_atol_m:.9g} m"
                )
        else:
            max_abs = 0.0
            max_xy = 0.0
        arm_diagnostics[arm] = {
            "native_slot_count": native_count,
            "guided_slot_count": group_size - native_count,
            "zero_strength_native_max_abs_all_channels": max_abs,
            "zero_strength_native_max_xy_m": max_xy,
        }

    descriptors = [
        {
            "k": index,
            "eta_lat": float(eta_lat_cpu[index]),
            "eta_lon": float(eta_lon_cpu[index]),
            "target_lateral_m": float(lambda_lat * eta_lat_cpu[index]),
            "target_speed_scale": float(1.0 + lambda_lon * eta_lon_cpu[index]),
        }
        for index in range(group_size)
    ]
    return trajectories, descriptors, torch.as_tensor(reference), arm_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--args_path", type=Path, required=True)
    parser.add_argument("--reference_model_path", type=Path, required=True)
    parser.add_argument("--reference_args_path", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--reward_config", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="*")
    parser.add_argument("--num_scenes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--group_size", type=int, default=10)
    parser.add_argument("--noise_scale", type=float, default=0.5)
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=10,
        help="explicit Original-DP denoising steps for policy and frozen reference",
    )
    parser.add_argument(
        "--reference_mode",
        choices=("zero_noise", "stochastic"),
        default="zero_noise",
        help=(
            "frozen-reference latent: formal T4 zero latent or the paper's "
            "one Gaussian sample per scene"
        ),
    )
    parser.add_argument(
        "--reference_noise_scale",
        type=float,
        default=0.5,
        help="Original-DP latent temperature for --reference_mode stochastic",
    )
    parser.add_argument(
        "--eta_scheme",
        choices=("iid_beta", "stratified_beta"),
        default="iid_beta",
    )
    parser.add_argument("--arms", nargs="+", choices=DEFAULT_ARMS, default=DEFAULT_ARMS)
    parser.add_argument("--lambda_lat", type=float, default=1.0)
    parser.add_argument("--lambda_lon", type=float, default=0.25)
    parser.add_argument(
        "--longitudinal_mode",
        choices=("candidate_stretch", "reference_speed_push"),
        default="candidate_stretch",
        help=(
            "formal candidate-derived stretch or opt-in frozen-reference "
            "speed push; paired diagnostic only"
        ),
    )
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--ramp_steps", type=int, default=20)
    parser.add_argument("--equivalence_atol_m", type=float, default=1.0e-5)
    parser.add_argument("--disable_compile", action="store_true")
    parser.add_argument("--compile_mode", default="default")
    parser.add_argument("--render_arms", nargs="*", choices=DEFAULT_ARMS, default=[])
    parser.add_argument("--gif_arms", nargs="*", choices=DEFAULT_ARMS, default=[])
    args = parser.parse_args()

    if args.group_size < 2:
        raise ValueError("group_size must be >= 2")
    if args.sample_steps < 2:
        raise ValueError("sample_steps must be >= 2")
    arms = tuple(dict.fromkeys(("native", *args.arms)))
    for arm in arms:
        _native_slots(arm, args.group_size)
    # Both alignment conventions exist in T4 data: legacy pre-20260707 NPZ
    # stores the current frame in neighbor_agents_future[0] (offset 1); the
    # regenerated 20260707 corpus is already aligned (offset 0).  The caller
    # must state which corpus it scores — refusing to guess is the guard;
    # hardcoding either value would silently corrupt the other generation.
    if NEIGHBOR_FUTURE_OFFSET_ENV not in os.environ:
        raise RuntimeError(
            f"set {NEIGHBOR_FUTURE_OFFSET_ENV} explicitly: 1 for legacy "
            "pre-20260707 NPZ, 0 for the regenerated corpus"
        )
    offset = get_neighbor_future_offset()
    if offset not in (0, 1):
        raise RuntimeError(
            f"unsupported {NEIGHBOR_FUTURE_OFFSET_ENV}={offset}; expected 0 or 1"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _configure_cuda_runtime(device)
    model, model_args = load_original_dp_checkpoint(
        args.model_path, device, args_path=args.args_path, use_ema=True
    )
    reference_model, reference_model_args = load_original_dp_checkpoint(
        args.reference_model_path,
        device,
        args_path=args.reference_args_path,
        use_ema=True,
    )
    model.eval()
    reference_model.eval()
    # Never inherit an implicit decoder default in a sampler comparison.  The
    # formal cache uses five steps while Original-DP deployment uses ten, and
    # the two produce measurably different trajectory distributions.
    model.decoder._sample_steps = int(args.sample_steps)
    reference_model.decoder._sample_steps = int(args.sample_steps)
    compile_enabled = not args.disable_compile and device.type == "cuda"
    if compile_enabled:
        compile_config = {
            "enabled": True,
            "backend": "inductor",
            "mode": args.compile_mode,
            "fullgraph": False,
            "dynamic": False,
            "encoder": False,
            "decoder": True,
        }
        _compile_planner_modules(model, compile_config, "plannerrft_sampler_policy")
        _compile_planner_modules(
            reference_model,
            compile_config,
            "plannerrft_sampler_frozen_reference",
        )

    reward_config = _load_reward_config(args.reward_config)
    scene_paths = json.loads(args.scenes.read_text())
    indices = args.indices or list(range(min(args.num_scenes, len(scene_paths))))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": {
            "role": "paired candidate-discovery ablation; not training",
            "model_path": str(args.model_path.resolve()),
            "args_path": str(args.args_path.resolve()),
            "frozen_reference_model_path": str(args.reference_model_path.resolve()),
            "frozen_reference_args_path": str(args.reference_args_path.resolve()),
            "scenes": str(args.scenes.resolve()),
            "neighbor_future_offset": offset,
            "x2_ego_width_m": X2_LEGACY_EGO_WIDTH_M,
            "group_size": args.group_size,
            "independent_latents": True,
            "same_latents_across_arms": True,
            "noise_scale": args.noise_scale,
            "sample_steps": int(args.sample_steps),
            "reference_mode": args.reference_mode,
            "reference_noise_scale": (
                0.0
                if args.reference_mode == "zero_noise"
                else args.reference_noise_scale
            ),
            "reference_seed_offset": 10_000_019,
            "deterministic_first": False,
            "fixed_beta_concentration": FIXED_BETA_CONCENTRATION,
            "fixed_beta_mapping": "eta=2*Beta(a,a)-1",
            "eta_sampling_scheme": args.eta_scheme,
            "arms": list(arms),
            "lambda_lat_m": args.lambda_lat,
            "lambda_lon_fraction": args.lambda_lon,
            "longitudinal_mode": args.longitudinal_mode,
            "guidance_scale": args.guidance_scale,
            "ramp_steps": args.ramp_steps,
            "lateral_onset": "quintic_smootherstep; Original-DP adaptation",
            "longitudinal_semantics": (
                "candidate displacement stretch=1+lambda_lon*eta_lon"
                if args.longitudinal_mode == "candidate_stretch"
                else "zero-centred frozen-reference forward-step push with quintic onset"
            ),
            "compile_enabled": compile_enabled,
            "equivalence_atol_m": args.equivalence_atol_m,
        },
        "scenes_results": [],
    }

    for order, scene_index in enumerate(indices):
        scene_path = scene_paths[scene_index]
        data = load_npz_data(scene_path, device)
        ego_width_m = float(data["ego_shape"][0, 2].item())
        if {"x2", "x2_dev"}.intersection(Path(scene_path).parts) and not np.isclose(
            ego_width_m, X2_LEGACY_EGO_WIDTH_M, rtol=0.0, atol=1.0e-6
        ):
            raise RuntimeError(
                f"scene {scene_path} uses X2 width {ego_width_m}, expected "
                f"{X2_LEGACY_EGO_WIDTH_M}"
            )
        trajectories, descriptors, reference, arm_diagnostics = generate_paired_arms(
            model,
            model_args,
            reference_model,
            reference_model_args,
            data,
            seed=args.seed + scene_index * 1009,
            group_size=args.group_size,
            noise_scale=args.noise_scale,
            arms=arms,
            lambda_lat=args.lambda_lat,
            lambda_lon=args.lambda_lon,
            longitudinal_mode=args.longitudinal_mode,
            guidance_scale=args.guidance_scale,
            ramp_steps=args.ramp_steps,
            equivalence_atol_m=args.equivalence_atol_m,
            eta_scheme=args.eta_scheme,
            reference_mode=args.reference_mode,
            reference_noise_scale=args.reference_noise_scale,
        )
        native_np = trajectories["native"].detach().cpu().numpy()
        scene_record = {
            "scene_index": scene_index,
            "scene_path": scene_path,
            "effective_ego_width_m": ego_width_m,
            "descriptors": descriptors,
            "arm_diagnostics": arm_diagnostics,
            "arms": {},
        }
        for arm in arms:
            results = score_trajectory_group(trajectories[arm], data, reward_config)
            native_count = _native_slots(arm, args.group_size)
            for result in results:
                index = int(result["k"])
                descriptor = descriptors[index]
                result["candidate_label"] = (
                    f"U{index}"
                    if index < native_count
                    else f"G{index} L{descriptor['eta_lat']:+.2f} V{descriptor['eta_lon']:+.2f}"
                )
                result["eta_lat"] = descriptor["eta_lat"]
                result["eta_lon"] = descriptor["eta_lon"]
                result["is_guided"] = index >= native_count
                result.update(
                    _response_metrics(
                        result["traj"],
                        reference.numpy(),
                        native_np[index],
                    )
                )
            diversity = diversity_metrics(results)
            png = None
            gif = None
            title = (
                f"PlannerRFT fixed-Beta sampler · {arm} · scene {scene_index} · "
                "paired independent latents"
            )
            if arm in args.render_arms:
                figure = plt.figure(figsize=(20, 11), facecolor="#f8fafc")
                draw_grpo_scene(
                    figure,
                    results,
                    scene_path,
                    scene_index,
                    _gt_path_length(data),
                    "PLANNERRFT SAMPLER ABLATION",
                    scene_title=title,
                    diversity=diversity,
                    reference_trajectory=reference.numpy(),
                )
                png = args.output_dir / f"scene_{order:03d}_{scene_index:06d}_{arm}.png"
                figure.savefig(png, dpi=180, bbox_inches="tight")
                plt.close(figure)
            if arm in args.gif_arms:
                gif = args.output_dir / f"scene_{order:03d}_{scene_index:06d}_{arm}.gif"
                render_dynamics_gif(
                    results,
                    scene_path,
                    gif,
                    scene_index,
                    "PLANNERRFT SAMPLER ABLATION",
                    scene_title=title,
                    reference_trajectory=reference.numpy(),
                )
            trajectory_path = (
                args.output_dir / f"scene_{order:03d}_{scene_index:06d}_{arm}.npz"
            )
            np.savez_compressed(
                trajectory_path,
                trajectories=trajectories[arm].detach().cpu().numpy(),
                reference=reference.numpy(),
                eta_lat=np.asarray([row["eta_lat"] for row in descriptors]),
                eta_lon=np.asarray([row["eta_lon"] for row in descriptors]),
                is_guided=np.asarray(
                    [index >= native_count for index in range(args.group_size)]
                ),
            )
            scene_record["arms"][arm] = {
                "native_slot_count": native_count,
                "guided_slot_count": args.group_size - native_count,
                "png": None if png is None else png.name,
                "gif": None if gif is None else gif.name,
                "candidate_trajectories": trajectory_path.name,
                "diversity": diversity,
                "ranked_candidates": [_result_record(result) for result in results],
            }
        manifest["scenes_results"].append(scene_record)
        print(f"scored scene {scene_index} ({order + 1}/{len(indices)})", flush=True)

    (args.output_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
