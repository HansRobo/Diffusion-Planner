from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from rlvr.autoresearch.tools.build_positive_anchor_replay_overlay import (
    build_overlay,
    compute_positive_anchor_weights,
)
from rlvr.awr import AWRRolloutConfig, compute_awr_weights
from rlvr.reward import RewardBreakdown, RewardConfig
from rlvr.train_awr import _validate_replay_source_contract


def _reward(total: float) -> RewardBreakdown:
    return RewardBreakdown(
        safety=0.0,
        progress=0.0,
        smoothness=0.0,
        feasibility=0.0,
        centerline=0.0,
        red_light=0.0,
        total=total,
        collision_step=None,
        off_road_fraction=0.0,
    )


def test_vectorized_weights_match_runtime_positive_anchor() -> None:
    rewards = np.asarray(
        [[0.2, 0.8, 0.1, 0.5], [0.0, 0.0, 0.0, 0.0], [0.9, 0.2, 0.9, 0.1]],
        dtype=np.float32,
    )
    actual = compute_positive_anchor_weights(rewards)
    expected = []
    for row in rewards:
        weights, _ = compute_awr_weights(
            [_reward(float(value)) for value in row],
            beta=1.0,
            weight_clip=1e9,
            normalize=False,
            min_group_std=1e-6,
            positive_advantage_only=True,
            behavior_anchor_weight=1.0,
            unsafe_behavior_anchor_weight=1.0,
            drop_all_zero_groups=True,
        )
        expected.append(weights.numpy())
    np.testing.assert_allclose(actual, np.stack(expected), rtol=1e-6, atol=1e-7)


def _write_minimal_source_cache(
    tmp_path: Path, rewards: np.ndarray, source_weights: np.ndarray
) -> Path:
    source_run = tmp_path / "source"
    source = source_run / "replay_buffer"
    rank = source / "rank_0000"
    rank.mkdir(parents=True)
    count, group_size = rewards.shape
    np.save(rank / "rewards.npy", rewards)
    np.save(rank / "weights.npy", source_weights)
    np.save(rank / "noise_scales.npy", np.ones_like(rewards))
    np.save(
        rank / "trajectories.npy",
        np.zeros((count, group_size, 3, 4), dtype=np.float32),
    )
    np.save(
        rank / "scene_encoding.npy", np.zeros((count, 2, 4), dtype=np.float32)
    )
    np.save(
        rank / "expert_trajectories.npy", np.zeros((count, 3, 4), dtype=np.float32)
    )
    np.save(rank / "expert_rewards.npy", np.full(count, 0.9, dtype=np.float32))
    np.save(rank / "expert_safe.npy", np.ones(count, dtype=np.float32))
    (rank / "scene_paths.jsonl").write_text(
        "".join(f'"/scene_{index}.npz"\n' for index in range(count))
    )
    (rank / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "rank": 0,
                "scene_count": count,
                "group_size": group_size,
                "future_len": 3,
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
    (rank / "expert_anchor_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "schema": "logged_ego_future_hard_gate_v1",
                "scene_count": count,
                "future_len": 3,
                "paths": "scene_paths.jsonl",
                "arrays": {
                    "expert_trajectories": "expert_trajectories.npy",
                    "expert_rewards": "expert_rewards.npy",
                    "expert_safe": "expert_safe.npy",
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
    return source


def test_source_zero_weight_candidates_never_resurrect(tmp_path: Path) -> None:
    # Group 0: candidate 1 beats the anchor but was zero-weighted at mine time
    # (first-waypoint gate).  Group 1: the same reward shape with a positive
    # mining weight is the control and must stay an active target.
    rewards = np.asarray([[0.2, 0.8], [0.2, 0.8]], dtype=np.float32)
    source_weights = np.asarray([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    source = _write_minimal_source_cache(tmp_path, rewards, source_weights)

    output = tmp_path / "overlay" / "replay_buffer"
    result = build_overlay(source, output)

    weights = np.load(output / "rank_0000" / "weights.npy")
    assert weights[0, 1] == 0.0
    assert weights[1, 1] > 0.0
    assert result["source_zero_masked_candidates"] == 1
    assert result["parameters"]["respect_source_zero_weights"] is True
    assert result["ranks"][0]["source_zero_masked_candidates"] == 1


def test_builds_read_only_overlay_with_only_weights_materialized(tmp_path: Path) -> None:
    source_run = tmp_path / "source"
    source = source_run / "replay_buffer"
    rank = source / "rank_0000"
    rank.mkdir(parents=True)
    rewards = np.asarray([[0.2, 0.8], [0.0, 0.0]], dtype=np.float32)
    np.save(rank / "rewards.npy", rewards)
    np.save(rank / "weights.npy", np.ones_like(rewards))
    np.save(rank / "noise_scales.npy", np.ones_like(rewards))
    np.save(rank / "trajectories.npy", np.zeros((2, 2, 3, 4), dtype=np.float32))
    np.save(rank / "scene_encoding.npy", np.zeros((2, 2, 4), dtype=np.float32))
    np.save(rank / "expert_trajectories.npy", np.zeros((2, 3, 4), dtype=np.float32))
    np.save(rank / "expert_rewards.npy", np.asarray([0.9, 0.0], dtype=np.float32))
    np.save(rank / "expert_safe.npy", np.asarray([1.0, 0.0], dtype=np.float32))
    (rank / "scene_paths.jsonl").write_text("\"/a.npz\"\n\"/b.npz\"\n")
    (rank / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "rank": 0,
                "scene_count": 2,
                "group_size": 2,
                "future_len": 3,
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
    (rank / "expert_anchor_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "schema": "logged_ego_future_hard_gate_v1",
                "scene_count": 2,
                "future_len": 3,
                "paths": "scene_paths.jsonl",
                "arrays": {
                    "expert_trajectories": "expert_trajectories.npy",
                    "expert_rewards": "expert_rewards.npy",
                    "expert_safe": "expert_safe.npy",
                },
            }
        )
    )
    rollout = AWRRolloutConfig(deterministic_first=True)
    source_awr = asdict(rollout)
    # Emulate the already-running Cycle-1 writer, which started immediately
    # before these candidate-ablation knobs became explicit config fields.
    for key in (
        "plannerrft_reference_mode",
        "plannerrft_reference_noise_scale",
        "plannerrft_longitudinal_mode",
    ):
        source_awr.pop(key)
    (source_run / "effective_config.json").write_text(
        json.dumps({"awr": source_awr, "reward": asdict(RewardConfig())})
    )
    (source_run / "provenance.json").write_text(
        json.dumps({"neighbor_future_alignment_offset": 1})
    )

    output = tmp_path / "overlay" / "replay_buffer"
    result = build_overlay(source, output)

    weights = np.load(output / "rank_0000" / "weights.npy")
    assert weights[0, 0] == 1.0
    assert weights[0, 1] > 1.0
    assert weights[1].tolist() == [0.0, 0.0]
    assert (output / "rank_0000" / "rewards.npy").is_symlink()
    assert (output / "rank_0000" / "expert_anchor_manifest.json").is_symlink()
    assert (output / "rank_0000" / "expert_trajectories.npy").is_symlink()
    assert not (output / "rank_0000" / "weights.npy").is_symlink()
    assert result["expert_anchor_sidecar"] is True
    assert result["groups_with_better_candidate_fraction"] == 0.5
    assert result["dropped_all_zero_groups"] == 1
    overlay_manifest = json.loads(
        (output / "rank_0000" / "manifest.json").read_text()
    )
    expected_paths_sha256 = hashlib.sha256(
        (rank / "scene_paths.jsonl").read_bytes()
    ).hexdigest()
    assert overlay_manifest["decoder_context_attachment"] == {
        "candidate_arrays_unchanged": True,
        "expected_paths_sha256": expected_paths_sha256,
    }
    overlay_awr = json.loads(
        (output.parent / "effective_config.json").read_text()
    )["awr"]
    assert overlay_awr["plannerrft_reference_mode"] == "zero_noise"
    assert overlay_awr["plannerrft_reference_noise_scale"] == 0.5
    assert overlay_awr["plannerrft_longitudinal_mode"] == "candidate_stretch"

    # The overlay, rather than the immutable mine cache, is the formal resume
    # source.  Its rewritten effective config must describe the materialized
    # positive-only weights exactly or train_awr will (correctly) fail closed
    # before replay.  Keep this assertion beside the builder so later AWR
    # config changes cannot silently break the mine -> overlay -> replay handoff.
    replay_rollout = replace(
        rollout,
        beta=1.0,
        weight_clip=1e9,
        normalize_weights=False,
        positive_advantage_only=True,
        drop_all_zero_groups=True,
    )
    _validate_replay_source_contract(output, replay_rollout, RewardConfig())
