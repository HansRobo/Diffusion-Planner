from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from rlvr.autoresearch.tools.build_behavior_reward_filter_overlay import build_overlay
from rlvr.awr import AWRRolloutConfig
from rlvr.reward import RewardConfig


def test_preserves_source_weights_only_below_behavior_threshold(tmp_path: Path) -> None:
    source_run = tmp_path / "source"
    rank = source_run / "replay_buffer" / "rank_0000"
    rank.mkdir(parents=True)
    rewards = np.asarray([[0.89, 0.95], [0.90, 0.99], [0.0, 0.0]], dtype=np.float32)
    source_weights = np.asarray([[0.5, 2.0], [0.2, 3.0], [0.0, 0.0]], dtype=np.float32)
    np.save(rank / "rewards.npy", rewards)
    np.save(rank / "weights.npy", source_weights)
    np.save(rank / "noise_scales.npy", np.ones_like(rewards))
    np.save(rank / "trajectories.npy", np.zeros((3, 2, 4, 4), dtype=np.float32))
    np.save(rank / "scene_encoding.npy", np.zeros((3, 2, 4), dtype=np.float32))
    (rank / "scene_paths.jsonl").write_text('"/a.npz"\n"/b.npz"\n"/c.npz"\n')
    (rank / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "rank": 0,
                "scene_count": 3,
                "group_size": 2,
                "future_len": 4,
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
    (source_run / "effective_config.json").write_text(
        json.dumps(
            {
                "awr": asdict(AWRRolloutConfig(deterministic_first=True)),
                "reward": asdict(RewardConfig()),
            }
        )
    )
    (source_run / "provenance.json").write_text(
        json.dumps({"neighbor_future_alignment_offset": 1})
    )

    output = tmp_path / "filtered" / "replay_buffer"
    result = build_overlay(source_run / "replay_buffer", output, behavior_reward_lt=0.9)
    actual = np.load(output / "rank_0000" / "weights.npy")
    np.testing.assert_array_equal(actual[0], source_weights[0])
    np.testing.assert_array_equal(actual[1], np.zeros(2, dtype=np.float32))
    np.testing.assert_array_equal(actual[2], source_weights[2])
    assert result["eligible_groups"] == 2
    assert result["active_groups"] == 1
    assert result["active_targets"] == 2
    assert (output / "rank_0000" / "rewards.npy").is_symlink()
