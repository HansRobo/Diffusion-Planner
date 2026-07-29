"""Regression tests for the legacy T4 neighbor-future +1 contract."""

from __future__ import annotations

import numpy as np
import pytest
import sys
import torch
from pathlib import Path

# The source checkout keeps the installable ``diffusion_planner`` package one
# directory below the repository-level ``diffusion_planner/`` project folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "diffusion_planner"))

from diffusion_planner.utils.neighbor_future_alignment import (
    NEIGHBOR_FUTURE_OFFSET_ENV,
    align_neighbor_future_numpy,
    align_neighbor_future_tensor,
    get_neighbor_future_offset,
)
from preference_optimization.utils import load_npz_data
from preference_optimization.utils import apply_npz_heading_transforms
from rlvr.train_awr import _compact_replay_decoder_context, _load_scene_batch


def test_default_legacy_alignment_maps_raw_t_plus_one_to_future_t(monkeypatch) -> None:
    monkeypatch.delenv(NEIGHBOR_FUTURE_OFFSET_ENV, raising=False)
    raw = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)

    aligned = align_neighbor_future_numpy(raw)

    np.testing.assert_array_equal(aligned[:, :-1], raw[:, 1:])
    np.testing.assert_array_equal(aligned[:, -1], 0.0)
    assert aligned.shape == raw.shape
    assert get_neighbor_future_offset() == 1


def test_tensor_alignment_uses_time_axis_for_batched_input() -> None:
    raw = torch.arange(2 * 3 * 5 * 4, dtype=torch.float32).reshape(2, 3, 5, 4)

    aligned = align_neighbor_future_tensor(raw, offset=1)

    torch.testing.assert_close(aligned[..., :-1, :], raw[..., 1:, :])
    torch.testing.assert_close(aligned[..., -1, :], torch.zeros_like(raw[..., -1, :]))


def test_new_data_can_disable_alignment_without_copying(monkeypatch) -> None:
    monkeypatch.setenv(NEIGHBOR_FUTURE_OFFSET_ENV, "0")
    raw_np = np.arange(12, dtype=np.float32).reshape(1, 4, 3)
    raw_t = torch.from_numpy(raw_np)

    assert align_neighbor_future_numpy(raw_np) is raw_np
    assert align_neighbor_future_tensor(raw_t) is raw_t
    assert get_neighbor_future_offset() == 0


def test_compact_replay_context_does_not_shift_canonical_loader_twice(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv(NEIGHBOR_FUTURE_OFFSET_ENV, "1")
    raw_future = np.arange(3 * 5 * 3, dtype=np.float32).reshape(3, 5, 3)
    npz = tmp_path / "scene.npz"
    np.savez(
        npz,
        ego_shape=np.asarray([2.8, 4.8, 2.0], dtype=np.float32),
        ego_current_state=np.zeros(10, dtype=np.float32),
        neighbor_agents_past=np.zeros((3, 2, 11), dtype=np.float32),
        neighbor_agents_future=raw_future,
    )
    loaded = load_npz_data(npz, torch.device("cpu"))
    compact = _compact_replay_decoder_context(loaded, predicted_neighbor_num=2)

    expected = torch.zeros(1, 2, 5, 3)
    expected[..., :-1, :] = torch.from_numpy(raw_future[:2, 1:, :]).unsqueeze(0)
    assert torch.equal(loaded["neighbor_agents_future"][:, :2], expected)
    assert torch.equal(compact["neighbor_agents_future"], expected)


@pytest.mark.parametrize("workers", [1, 4])
def test_legacy_replay_fast_loader_matches_canonical_decoder_context_exactly(
    tmp_path, monkeypatch, workers: int
) -> None:
    """The legacy NPZ replay shortcut must be byte-equivalent at the decoder.

    X2 ``ego_shape`` is consumed by the canonical encoder during mining and is
    therefore already represented in the cached scene encoding.  Replay only
    reloads the three policy-independent tensors consumed by the diffusion
    decoder.  This test covers that production boundary, including ordered
    threaded loading and the one-and-only-one legacy future shift.
    """

    monkeypatch.setenv(NEIGHBOR_FUTURE_OFFSET_ENV, "1")
    paths: list[str] = []
    for scene_index in range(5):
        scene = tmp_path / "x2" / "train" / f"scene_{scene_index}.npz"
        scene.parent.mkdir(parents=True, exist_ok=True)
        ego = (
            np.arange(10, dtype=np.float32) + np.float32(scene_index * 100)
        )
        neighbor_past = (
            np.arange(3 * 6 * 11, dtype=np.float32).reshape(3, 6, 11)
            + np.float32(scene_index * 1_000)
        )
        neighbor_future = (
            np.arange(3 * 5 * 3, dtype=np.float32).reshape(3, 5, 3)
            + np.float32(scene_index * 10_000)
        )
        np.savez(
            scene,
            ego_shape=np.asarray([3.0, 4.9, 2.42741], dtype=np.float32),
            ego_current_state=ego,
            neighbor_agents_past=neighbor_past,
            neighbor_agents_future=neighbor_future,
        )
        paths.append(str(scene))

    canonical_scenes = [
        load_npz_data(path, torch.device("cpu")) for path in paths
    ]
    # Prove that the model-side geometry contract was applied before the
    # behavior encoder was cached; it is intentionally not decoder context.
    assert all(
        scene["ego_shape"][0, 2].item()
        == pytest.approx(2.29156)
        for scene in canonical_scenes
    )
    expected = {
        "ego_current_state": torch.cat(
            [scene["ego_current_state"] for scene in canonical_scenes], dim=0
        ),
        "neighbor_agents_past": torch.cat(
            [scene["neighbor_agents_past"][:, :, -1:, :] for scene in canonical_scenes],
            dim=0,
        ),
        "neighbor_agents_future": torch.cat(
            [scene["neighbor_agents_future"] for scene in canonical_scenes], dim=0
        ),
    }

    actual = _load_scene_batch(
        paths,
        torch.device("cpu"),
        workers=workers,
        replay_context_only=True,
    )
    assert tuple(actual) == tuple(expected)
    for key in expected:
        assert torch.equal(actual[key], expected[key]), key


def test_full_scene_serial_and_threaded_batch_loaders_are_exactly_equal(
    tmp_path, monkeypatch
) -> None:
    """CPU merge + one device transfer must preserve canonical full inputs."""

    monkeypatch.setenv(NEIGHBOR_FUTURE_OFFSET_ENV, "1")
    paths: list[str] = []
    for scene_index in range(4):
        scene = tmp_path / "x2" / "valid" / f"scene_{scene_index}.npz"
        scene.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            scene,
            ego_shape=np.asarray([3.0, 4.9, 2.42741], dtype=np.float32),
            ego_current_state=(
                np.arange(10, dtype=np.float32) + scene_index * 100
            ),
            ego_agent_past=(
                np.arange(6 * 3, dtype=np.float32).reshape(6, 3)
                + scene_index * 200
            ),
            neighbor_agents_past=(
                np.arange(3 * 6 * 11, dtype=np.float32).reshape(3, 6, 11)
                + scene_index * 1_000
            ),
            neighbor_agents_future=(
                np.arange(3 * 5 * 3, dtype=np.float32).reshape(3, 5, 3)
                + scene_index * 10_000
            ),
        )
        paths.append(str(scene))

    serial = _load_scene_batch(
        paths, torch.device("cpu"), workers=1, replay_context_only=False
    )
    threaded = _load_scene_batch(
        paths, torch.device("cpu"), workers=4, replay_context_only=False
    )
    assert tuple(serial) == tuple(threaded)
    for key in serial:
        assert torch.equal(serial[key], threaded[key]), key
    assert torch.equal(
        serial["ego_shape"][:, 2],
        torch.full((len(paths),), 2.29156, dtype=torch.float32),
    )

    prefetched = _load_scene_batch(
        paths,
        torch.device("cpu"),
        workers=1,
        replay_context_only=False,
        defer_full_scene_heading_transforms=True,
    )
    assert prefetched["ego_agent_past"].shape[-1] == 3
    apply_npz_heading_transforms(prefetched)
    for key in serial:
        assert torch.equal(serial[key], prefetched[key]), key


@pytest.mark.parametrize("value", ["-1", "not-an-int"])
def test_invalid_environment_offset_fails_closed(monkeypatch, value: str) -> None:
    monkeypatch.setenv(NEIGHBOR_FUTURE_OFFSET_ENV, value)

    with pytest.raises(ValueError):
        get_neighbor_future_offset()
