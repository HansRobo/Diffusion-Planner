#!/usr/bin/env python3
"""Dump deterministic baseline/AWR trajectories for paired scene rendering.

This deliberately does not train or select a sampled repair.  It runs the
same zero-noise, K=1 deployment policy used by the full validation evaluator
on both checkpoints, then writes small ``.npz`` files that
``render_gate_comparison.py`` can turn into high-resolution BEV/OBB figures
and GIFs.  Keeping model inference separate from rendering makes it possible
to render many selected scenes without holding two checkpoints in GPU memory.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from rlvr.awr import AWRRolloutConfig, load_original_dp_checkpoint, load_scene, rollout_and_score_scene
from rlvr.train_awr import _configure_cuda_runtime, _load_config


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_model", type=Path, required=True)
    p.add_argument("--baseline_args", type=Path, required=True)
    p.add_argument("--awr_model", type=Path, required=True)
    p.add_argument("--awr_args", type=Path, required=True)
    p.add_argument("--scenes", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--sample_steps",
        type=int,
        default=10,
        help="deployment solver steps; 10 matches the original DP checkpoint default",
    )
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def _dump_model(
    label: str,
    model_path: Path,
    args_path: Path,
    scenes: list[str],
    output_dir: Path,
    config,
    device: torch.device,
    sample_steps: int,
) -> None:
    model, model_args = load_original_dp_checkpoint(
        model_path, device, args_path, use_ema=True
    )
    model.eval()
    rollout_config = AWRRolloutConfig(
        n_trajectories=1,
        sample_steps=max(2, int(sample_steps)),
        noise_scale_range=(0.0, 0.0),
        beta=1.0,
        weight_clip=1e9,
        normalize_weights=False,
        min_group_std=1e-6,
        deterministic_first=True,
        hdp_trajectory_augmentation=False,
        # This is the executed-axis measurement tool, so it must run the precision the
        # vehicle runs.  Under bf16 the executed first waypoint snaps to a 39mm comb --
        # the tool would reproduce the exact artifact it exists to measure around.
        inference_amp_dtype="off",
    )
    label_dir = output_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, scene_path in enumerate(scenes):
        data = load_scene(scene_path, device)
        rollout = rollout_and_score_scene(
            model, model_args, data, rollout_config, config, device
        )
        stem = f"scene_{index:03d}"
        np.savez_compressed(
            label_dir / f"{stem}.npz",
            trajectories=rollout.trajectories.detach().cpu().numpy(),
            noise_scales=rollout.noise_scales.detach().cpu().numpy(),
            weights=rollout.weights.detach().cpu().numpy(),
            selected_index=np.asarray(0, dtype=np.int64),
        )
        manifest.append(
            {
                "scene_path": scene_path,
                "npz": str(label_dir / f"{stem}.npz"),
                "reward": float(rollout.rewards[0].total),
                "collision": bool(rollout.rewards[0].collision_step is not None),
                "road_border": bool(rollout.rewards[0].rb_crossing),
                "kinematic": bool(rollout.rewards[0].kinematic_violated),
            }
        )
        print(
            f"{label} {index + 1}/{len(scenes)} "
            f"R={rollout.rewards[0].total:+.4f} "
            f"collision={int(rollout.rewards[0].collision_step is not None)}",
            flush=True,
        )
    (label_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    del model, model_args
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


def main() -> None:
    args = _args()
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if device.index is None:
            device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
    _configure_cuda_runtime(device)
    _, _, reward_config = _load_config(args.config)
    scenes = json.loads(args.scenes.read_text())
    if not isinstance(scenes, list) or not scenes:
        raise ValueError(f"empty scene list: {args.scenes}")
    scenes = [str(path) for path in scenes]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _dump_model(
        "baseline", args.baseline_model, args.baseline_args, scenes,
        args.output_dir, reward_config, device, args.sample_steps,
    )
    _dump_model(
        "awr", args.awr_model, args.awr_args, scenes,
        args.output_dir, reward_config, device, args.sample_steps,
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "scenes": scenes,
                "baseline_model": str(args.baseline_model),
                "awr_model": str(args.awr_model),
                "config": str(args.config),
                "sample_steps": int(args.sample_steps),
                "candidate_count": 1,
                "noise_scale": 0.0,
                "checkpoint_payload": "ema_state_dict",
                "alignment_offset": 1,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
