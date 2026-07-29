#!/usr/bin/env python3
"""Render an exactly cached AWR gate group with OBBs and an optional GIF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rlvr.awr import AWRRollout, compute_awr_weights, load_scene, reward_compatible_data
from rlvr.autoresearch.tools.awr_viz import (
    _load_reward_config,
    draw_scene_figure,
    save_gif,
)
from rlvr.reward import compute_reward_batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--cached_npz", type=Path, required=True)
    parser.add_argument("--reward_config", type=Path, required=True)
    parser.add_argument("--reward_profile", default=None)
    parser.add_argument("--reward_mode", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", default="cached HDP safe repair")
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max_gif_frames", type=int, default=40)
    parser.add_argument("--max_neighbors", type=int, default=24)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    reward_config = _load_reward_config(args.reward_config)
    if args.reward_profile is not None:
        reward_config.reward_profile = args.reward_profile
    if args.reward_mode is not None:
        reward_config.reward_mode = args.reward_mode
    data = load_scene(args.scene, device)
    cached = np.load(args.cached_npz)
    trajectories = torch.from_numpy(cached["trajectories"]).to(device=device, dtype=torch.float32)
    reward_data = reward_compatible_data(data)
    rewards = compute_reward_batch(trajectories, reward_data, reward_config)
    weights, diagnostics = compute_awr_weights(
        rewards,
        beta=1.0,
        weight_clip=1e9,
        normalize=True,
        reward_config=reward_config,
    )
    rollout = AWRRollout(
        trajectories=trajectories,
        noise_scales=torch.zeros(trajectories.shape[0], device=device),
        rewards=rewards,
        weights=weights.to(device),
        reward_data=reward_data,
        scene_encoding=torch.empty(0, device=device),
        diagnostics=diagnostics,
    )
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    draw_scene_figure(
        str(args.scene),
        rollout,
        reward_config,
        output,
        args.tag,
        args.max_neighbors,
    )
    if args.gif:
        gif = output.with_suffix(".gif")
        save_gif(
            str(args.scene),
            rollout,
            gif,
            args.fps,
            args.max_neighbors,
            args.max_gif_frames,
        )
    metadata = {
        "scene": str(args.scene),
        "cached_npz": str(args.cached_npz),
        "output": str(output),
        "tag": args.tag,
        "diagnostics": diagnostics,
        "rewards": [float(reward.total) for reward in rewards],
        "selected_index": int(cached["selected_index"].item()) if "selected_index" in cached else None,
        "safe_indices": cached["safe_indices"].tolist() if "safe_indices" in cached else [],
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
