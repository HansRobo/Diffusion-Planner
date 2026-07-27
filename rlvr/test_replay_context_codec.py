from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rlvr.replay_context_codec import (
    CompressedReplayContextReader,
    build_compressed_replay_context_rank,
)
from rlvr.train_awr import (
    _DiskReplayReader,
    _can_materialize_only_nonzero_replay_groups,
    _replay_expert_safe_groups_are_trained,
)


def test_unrestricted_expert_anchor_widens_the_mask_instead_of_disabling_it() -> None:
    expert = {"expert_anchor_weight": 0.4, "expert_anchor_active_groups_only": False}
    # Behaviour cloning has no per-group flag to follow, so it still forces every
    # group's context.  An unrestricted expert anchor does, so the mask stays on
    # and grows by exactly the safe groups.
    assert not _can_materialize_only_nonzero_replay_groups({"bc_weight": 0.1})
    assert _can_materialize_only_nonzero_replay_groups(expert)
    assert _replay_expert_safe_groups_are_trained(expert)
    assert not _replay_expert_safe_groups_are_trained(
        {**expert, "expert_anchor_active_groups_only": True}
    )
    assert not _replay_expert_safe_groups_are_trained({**expert, "expert_anchor_weight": 0.0})
    assert not _replay_expert_safe_groups_are_trained({})


def _write_fake_replay(root: Path) -> tuple[Path, dict[str, np.ndarray], str]:
    rank = root / "rank_0000"
    rank.mkdir(parents=True)
    rng = np.random.default_rng(17)
    scenes, group, future, neighbors = 7, 3, 5, 4
    arrays = {
        "scene_encoding": rng.standard_normal((scenes, 6, 8), dtype=np.float32),
        "ego_current_state": rng.standard_normal((scenes, 10), dtype=np.float32),
        "neighbor_agents_past": rng.standard_normal(
            (scenes, neighbors, 1, 11), dtype=np.float32
        ),
        "neighbor_agents_future": rng.standard_normal(
            (scenes, neighbors, future, 3), dtype=np.float32
        ),
        "expert_trajectories": rng.standard_normal(
            (scenes, future, 4), dtype=np.float32
        ),
        "expert_rewards": rng.random(scenes, dtype=np.float32),
        "expert_safe": rng.integers(0, 2, scenes).astype(np.float32),
        "trajectories": rng.standard_normal(
            (scenes, group, future, 4), dtype=np.float32
        ),
        "noise_scales": rng.random((scenes, group), dtype=np.float32),
        "rewards": rng.random((scenes, group), dtype=np.float32),
        "weights": rng.random((scenes, group), dtype=np.float32),
    }
    filenames = {
        "scene_encoding": "scene_encoding.npy",
        "context_ego_current_state": "context_ego_current_state.npy",
        "context_neighbor_agents_past": "context_neighbor_agents_past.npy",
        "context_neighbor_agents_future": "context_neighbor_agents_future.npy",
        "trajectories": "trajectories.npy",
        "noise_scales": "noise_scales.npy",
        "rewards": "rewards.npy",
        "weights": "weights.npy",
    }
    source_keys = {
        "scene_encoding": "scene_encoding",
        "context_ego_current_state": "ego_current_state",
        "context_neighbor_agents_past": "neighbor_agents_past",
        "context_neighbor_agents_future": "neighbor_agents_future",
        "trajectories": "trajectories",
        "noise_scales": "noise_scales",
        "rewards": "rewards",
        "weights": "weights",
    }
    for manifest_name, filename in filenames.items():
        np.save(rank / filename, arrays[source_keys[manifest_name]], allow_pickle=False)
    for name in ("expert_trajectories", "expert_rewards", "expert_safe"):
        np.save(rank / f"{name}.npy", arrays[name], allow_pickle=False)
    paths = [f"/fake/scene_{index}.npz" for index in range(scenes)]
    path_payload = "".join(json.dumps(path) + "\n" for path in paths)
    (rank / "scene_paths.jsonl").write_text(path_payload)
    path_hash = hashlib.sha256(
        "".join(path + "\n" for path in paths).encode()
    ).hexdigest()
    replay_manifest = {
        "version": 3,
        "rank": 0,
        "scene_count": scenes,
        "group_size": group,
        "future_len": future,
        "paths": "scene_paths.jsonl",
        "arrays": filenames,
        "decoder_context_cached": True,
        "decoder_context_schema": "canonical_loader_aligned_v1",
        "decoder_context_shapes": {
            "ego_current_state": [10],
            "neighbor_agents_past": [neighbors, 1, 11],
            "neighbor_agents_future": [neighbors, future, 3],
        },
        "encoding_shape": [6, 8],
        "decoder_context_attachment": {
            "candidate_arrays_unchanged": True,
            "expected_paths_sha256": path_hash,
        },
    }
    (rank / "manifest.json").write_text(json.dumps(replay_manifest))
    expert_manifest = {
        "version": 1,
        "schema": "logged_ego_future_hard_gate_v1",
        "scene_count": scenes,
        "future_len": future,
        "paths": "scene_paths.jsonl",
        "arrays": {
            name: f"{name}.npy"
            for name in ("expert_trajectories", "expert_rewards", "expert_safe")
        },
    }
    (rank / "expert_anchor_manifest.json").write_text(json.dumps(expert_manifest))
    return rank, arrays, path_hash


def test_compressed_context_roundtrip_is_byte_exact(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _, arrays, path_hash = _write_fake_replay(source)
    output = tmp_path / "compressed"
    manifest = build_compressed_replay_context_rank(
        source, output, 0, progress_interval=0
    )
    assert manifest["compressed_size_bytes"] > 0
    assert manifest["raw_total_bytes"] > 0

    reader = CompressedReplayContextReader(
        output,
        0,
        expected_scene_count=7,
        expected_paths_sha256=path_hash,
    )
    try:
        indices = np.asarray([6, 2, 2, 0, 4])
        decoded = reader.materialize(indices)
    finally:
        reader.close()
    for name in (
        "scene_encoding",
        "ego_current_state",
        "neighbor_agents_past",
        "neighbor_agents_future",
        "expert_trajectories",
        "expert_rewards",
        "expert_safe",
    ):
        expected = np.asarray(arrays[name][indices], dtype=np.float32)
        assert decoded[name].tobytes() == expected.tobytes()


def test_parallel_compressed_context_roundtrip_is_byte_exact(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _, arrays, path_hash = _write_fake_replay(source)
    output = tmp_path / "compressed"
    build_compressed_replay_context_rank(source, output, 0, progress_interval=0)
    indices = np.asarray([6, 2, 2, 0, 4])
    reader = CompressedReplayContextReader(
        output,
        0,
        expected_scene_count=7,
        expected_paths_sha256=path_hash,
        workers=4,
    )
    try:
        decoded = reader.materialize(indices)
    finally:
        reader.close()
    for name, values in decoded.items():
        expected = np.asarray(arrays[name][indices], dtype=np.float32)
        assert values.tobytes() == expected.tobytes()


def test_compressed_context_fails_on_scene_order_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_fake_replay(source)
    output = tmp_path / "compressed"
    build_compressed_replay_context_rank(source, output, 0, progress_interval=0)
    with pytest.raises(RuntimeError, match="scene order/hash differs"):
        CompressedReplayContextReader(
            output,
            0,
            expected_scene_count=7,
            expected_paths_sha256="0" * 64,
        )


def test_disk_replay_compressed_and_memmap_batches_are_exact(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_fake_replay(source)
    compressed_root = tmp_path / "compressed"
    build_compressed_replay_context_rank(
        source, compressed_root, 0, progress_interval=0
    )
    indices = np.asarray([6, 1, 1, 3])
    plain = _DiskReplayReader(source, 0)
    compressed = _DiskReplayReader(
        source, 0, compressed_context_root=compressed_root
    )
    try:
        plain_paths, plain_rollout = plain.batch_from_indices(indices)
        compressed_paths, compressed_rollout = compressed.batch_from_indices(indices)
    finally:
        plain.close()
        compressed.close()
    assert plain_paths == compressed_paths
    assert torch.equal(plain_rollout.trajectories, compressed_rollout.trajectories)
    assert torch.equal(plain_rollout.noise_scales, compressed_rollout.noise_scales)
    assert torch.equal(plain_rollout.weights, compressed_rollout.weights)
    assert torch.equal(plain_rollout.scene_encoding, compressed_rollout.scene_encoding)
    assert plain_rollout.reward_data.keys() == compressed_rollout.reward_data.keys()
    for key in plain_rollout.reward_data:
        assert torch.equal(
            plain_rollout.reward_data[key], compressed_rollout.reward_data[key]
        )


def test_compressed_reader_preserves_random_batch_rng_and_order(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_fake_replay(source)
    compressed_root = tmp_path / "compressed"
    build_compressed_replay_context_rank(
        source, compressed_root, 0, progress_interval=0
    )
    plain = _DiskReplayReader(source, 0)
    compressed = _DiskReplayReader(
        source, 0, compressed_context_root=compressed_root
    )
    try:
        plain_batches = list(plain.iter_random_batches(5, 4, seed=3407))
        compressed_batches = list(
            compressed.iter_random_batches(5, 4, seed=3407)
        )
    finally:
        plain.close()
        compressed.close()
    assert len(plain_batches) == len(compressed_batches) == 4
    for (plain_paths, plain_rollout), (compressed_paths, compressed_rollout) in zip(
        plain_batches, compressed_batches, strict=True
    ):
        assert plain_paths == compressed_paths
        assert torch.equal(
            plain_rollout.trajectories, compressed_rollout.trajectories
        )
        assert torch.equal(plain_rollout.weights, compressed_rollout.weights)
        assert torch.equal(
            plain_rollout.scene_encoding, compressed_rollout.scene_encoding
        )
        for key in plain_rollout.reward_data:
            assert torch.equal(
                plain_rollout.reward_data[key],
                compressed_rollout.reward_data[key],
            )


def test_active_only_context_skips_only_zero_gradient_rows(tmp_path: Path) -> None:
    source = tmp_path / "source"
    rank, _, _ = _write_fake_replay(source)
    weights_path = rank / "weights.npy"
    weights = np.load(weights_path)
    weights[[1, 3]] = 0.0
    np.save(weights_path, weights, allow_pickle=False)
    compressed_root = tmp_path / "compressed"
    build_compressed_replay_context_rank(
        source, compressed_root, 0, progress_interval=0
    )
    indices = np.asarray([6, 1, 1, 3, 2])
    active = torch.from_numpy(weights[indices].sum(axis=1) > 0.0)
    plain = _DiskReplayReader(source, 0)
    compressed = _DiskReplayReader(
        source,
        0,
        compressed_context_root=compressed_root,
        compressed_context_workers=3,
        context_active_weight_only=True,
    )
    try:
        plain_paths, plain_rollout = plain.batch_from_indices(indices)
        compressed_paths, compressed_rollout = compressed.batch_from_indices(indices)
    finally:
        plain.close()
        compressed.close()

    assert plain_paths == compressed_paths
    assert torch.equal(plain_rollout.trajectories, compressed_rollout.trajectories)
    assert torch.equal(plain_rollout.weights, compressed_rollout.weights)
    assert torch.equal(
        plain_rollout.scene_encoding[active],
        compressed_rollout.scene_encoding[active],
    )
    assert torch.count_nonzero(compressed_rollout.scene_encoding[~active]) == 0
    for key in (
        "ego_current_state",
        "neighbor_agents_past",
        "neighbor_agents_future",
        "_replay_expert_trajectory",
    ):
        assert torch.equal(
            plain_rollout.reward_data[key][active],
            compressed_rollout.reward_data[key][active],
        )
        assert torch.count_nonzero(
            compressed_rollout.reward_data[key][~active]
        ) == 0
    for key in ("_replay_expert_reward", "_replay_expert_safe"):
        assert torch.equal(
            plain_rollout.reward_data[key],
            compressed_rollout.reward_data[key],
        )
    assert compressed_rollout.diagnostics[
        "replay_context_materialized_group_fraction"
    ] == pytest.approx(float(active.float().mean()))


def test_active_only_context_keeps_zero_weight_groups_the_expert_anchor_trains(
    tmp_path: Path,
) -> None:
    # An unrestricted expert anchor trains a group whose candidates are all
    # zero-weighted, so skipping its context would train it on a zeroed scene.
    # The alternative -- decoding every group -- costs 100% of the corpus where
    # following the safety flags costs ~52%.
    source = tmp_path / "source"
    rank, _, _ = _write_fake_replay(source)
    weights = np.load(rank / "weights.npy")
    weights[[1, 3]] = 0.0
    np.save(rank / "weights.npy", weights, allow_pickle=False)
    expert_safe = np.zeros(weights.shape[0], dtype=np.float32)
    expert_safe[1] = 1.0
    np.save(rank / "expert_safe.npy", expert_safe, allow_pickle=False)
    compressed_root = tmp_path / "compressed"
    build_compressed_replay_context_rank(
        source, compressed_root, 0, progress_interval=0
    )

    indices = np.asarray([6, 1, 1, 3, 2])
    weight_active = weights[indices].sum(axis=1) > 0.0
    safe = expert_safe[indices] > 0.5
    plain = _DiskReplayReader(source, 0)
    readers = {
        name: _DiskReplayReader(
            source,
            0,
            compressed_context_root=compressed_root,
            context_active_weight_only=True,
            context_expert_safe_groups=flag,
        )
        for name, flag in (("gated", True), ("candidates_only", False))
    }
    try:
        _, plain_rollout = plain.batch_from_indices(indices)
        rollouts = {
            name: reader.batch_from_indices(indices)[1]
            for name, reader in readers.items()
        }
    finally:
        plain.close()
        for reader in readers.values():
            reader.close()

    expected = {
        "gated": torch.from_numpy(weight_active | safe),
        "candidates_only": torch.from_numpy(weight_active),
    }
    for name, rollout in rollouts.items():
        mask = expected[name]
        assert torch.equal(
            plain_rollout.scene_encoding[mask], rollout.scene_encoding[mask]
        ), name
        assert torch.count_nonzero(rollout.scene_encoding[~mask]) == 0, name
        assert rollout.diagnostics[
            "replay_context_materialized_group_fraction"
        ] == pytest.approx(float(mask.float().mean())), name

    # Index 1 is the point: zero-weight, expert-safe, and only the gated reader
    # decodes it.  Index 3 is zero-weight and unsafe, so neither does.
    assert torch.count_nonzero(rollouts["gated"].scene_encoding[1]) > 0
    assert torch.count_nonzero(rollouts["candidates_only"].scene_encoding[1]) == 0
    for rollout in rollouts.values():
        assert torch.count_nonzero(rollout.scene_encoding[3]) == 0
