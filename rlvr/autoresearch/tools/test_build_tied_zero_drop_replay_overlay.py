from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from rlvr.autoresearch.tools.build_tied_zero_drop_replay_overlay import build_overlay
from rlvr.awr import AWRRolloutConfig
from rlvr.reward import RewardConfig


def test_builds_small_overlay_and_preserves_active_weights(tmp_path: Path) -> None:
    source_run = tmp_path / "source"
    source = source_run / "replay_buffer"
    rank = source / "rank_0000"
    rank.mkdir(parents=True)
    rewards = np.asarray([[0.0, 0.0], [0.2, 0.8]], dtype=np.float32)
    weights = np.asarray([[1.0, 1.0], [0.5, 2.0]], dtype=np.float32)
    np.save(rank / "rewards.npy", rewards)
    np.save(rank / "weights.npy", weights)
    np.save(rank / "noise_scales.npy", np.ones_like(rewards))
    np.save(rank / "trajectories.npy", np.zeros((2, 2, 3, 4), dtype=np.float32))
    np.save(rank / "scene_encoding.npy", np.zeros((2, 2, 4), dtype=np.float32))
    (rank / "scene_paths.jsonl").write_text("/a.npz\n/b.npz\n")
    (rank / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "rank": 0,
                "scene_count": 2,
                "group_size": 2,
                "paths": "scene_paths.jsonl",
                "arrays": {
                    "rewards": "rewards.npy",
                    "weights": "weights.npy",
                    "noise_scales": "noise_scales.npy",
                    "trajectories": "trajectories.npy",
                    "scene_encoding": "scene_encoding.npy",
                },
            }
        )
    )
    rollout = AWRRolloutConfig(drop_all_zero_groups=False)
    (source_run / "effective_config.json").write_text(
        json.dumps({"awr": asdict(rollout), "reward": asdict(RewardConfig())})
    )
    (source_run / "provenance.json").write_text(
        json.dumps({"neighbor_future_alignment_offset": 1})
    )

    output = tmp_path / "overlay" / "replay_buffer"
    result = build_overlay(source, output)

    corrected = np.load(output / "rank_0000" / "weights.npy")
    assert corrected.tolist() == [[0.0, 0.0], [0.5, 2.0]]
    assert (output / "rank_0000" / "rewards.npy").is_symlink()
    assert result["dropped_tied_zero_groups"] == 1
    assert result["dropped_fraction"] == 0.5
    effective = json.loads((output.parent / "effective_config.json").read_text())
    assert effective["awr"]["drop_all_zero_groups"] is True
