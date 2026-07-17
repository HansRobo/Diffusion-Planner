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


@pytest.mark.parametrize("value", ["-1", "not-an-int"])
def test_invalid_environment_offset_fails_closed(monkeypatch, value: str) -> None:
    monkeypatch.setenv(NEIGHBOR_FUTURE_OFFSET_ENV, value)

    with pytest.raises(ValueError):
        get_neighbor_future_offset()
