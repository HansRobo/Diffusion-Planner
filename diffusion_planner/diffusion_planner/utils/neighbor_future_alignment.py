"""Temporal alignment for legacy neighbour futures.

The old T4 NPZ converter stored the present neighbour state as the first
element of ``neighbor_agents_future``. HDP removes that element when it
constructs the future target (``padded_agent_states[1:]``). AWR must use the
same convention for the model target, the reward/OBB replay, and visualisation.

The current data is legacy data, so the default offset is one frame. When a
future converter writes correctly aligned files, set
``DP_NEIGHBOR_FUTURE_OFFSET=0`` for the run. The operation is in-memory and
never mutates the NPZ files on disk.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch


DEFAULT_NEIGHBOR_FUTURE_OFFSET = 1
NEIGHBOR_FUTURE_OFFSET_ENV = "DP_NEIGHBOR_FUTURE_OFFSET"


def get_neighbor_future_offset() -> int:
    """Return the configured temporal offset for legacy neighbour futures."""

    raw = os.environ.get(
        NEIGHBOR_FUTURE_OFFSET_ENV, str(DEFAULT_NEIGHBOR_FUTURE_OFFSET)
    )
    try:
        offset = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{NEIGHBOR_FUTURE_OFFSET_ENV} must be a non-negative integer, got {raw!r}"
        ) from exc
    if offset < 0:
        raise ValueError(
            f"{NEIGHBOR_FUTURE_OFFSET_ENV} must be non-negative, got {offset}"
        )
    return offset


def align_neighbor_future_tensor(
    future: torch.Tensor,
    offset: int | None = None,
) -> torch.Tensor:
    """Shift the temporal axis while preserving its original length.

    ``future`` may be ``[P, T, C]`` or ``[B, P, T, C]``. A one-frame legacy
    shift maps raw ``t+1`` to aligned future ``t`` and zero-fills the final
    unavailable frame.
    """

    if offset is None:
        offset = get_neighbor_future_offset()
    if offset < 0:
        raise ValueError(f"neighbor future offset must be non-negative, got {offset}")
    if offset == 0 or future.ndim < 2 or future.shape[-2] == 0:
        return future

    aligned = torch.zeros_like(future)
    if offset < future.shape[-2]:
        aligned[..., :-offset, :] = future[..., offset:, :]
    return aligned


def align_neighbor_future_numpy(
    future: np.ndarray,
    offset: int | None = None,
) -> np.ndarray:
    """NumPy counterpart used by dataset and BEV visualisation loaders."""

    if offset is None:
        offset = get_neighbor_future_offset()
    if offset < 0:
        raise ValueError(f"neighbor future offset must be non-negative, got {offset}")
    if offset == 0 or future.ndim < 2 or future.shape[-2] == 0:
        return future

    aligned = np.zeros_like(future)
    if offset < future.shape[-2]:
        aligned[..., :-offset, :] = future[..., offset:, :]
    return aligned


def align_neighbor_future_data(
    data: dict[str, Any],
    offset: int | None = None,
) -> dict[str, Any]:
    """Return a shallow copy with the neighbour future aligned in memory."""

    future = data.get("neighbor_agents_future")
    if isinstance(future, torch.Tensor):
        out = dict(data)
        out["neighbor_agents_future"] = align_neighbor_future_tensor(future, offset)
        return out
    if isinstance(future, np.ndarray):
        out = dict(data)
        out["neighbor_agents_future"] = align_neighbor_future_numpy(future, offset)
        return out
    return data
