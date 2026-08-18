"""NeighborPatternDB must transparently widen a legacy 11-col DB (built before the
Unknown-class change) to 12 columns, and must treat Unknown-typed patterns as VRUs."""

import numpy as np
import pytest
import torch

from diffusion_planner.dimensions import INPUT_T, OUTPUT_T
from diffusion_planner.utils.neighbor_db import NeighborPatternDB, _match_past_width

PAST_T = INPUT_T + 1


def _write_db(tmp_path, past: np.ndarray, future: np.ndarray) -> str:
    path = str(tmp_path / "db.npz")
    np.savez(path, past=past, future=future)
    return path


def _kwargs():
    return dict(collision_margin=0.5, keep_clear_radius=1.0, min_collision_time=0.5,
                search_subsample=0)


def test_match_past_width_appends_zero_unknown_column():
    legacy = torch.ones(3, PAST_T, 11)
    widened = _match_past_width(legacy, 12)
    assert widened.shape == (3, PAST_T, 12)
    assert torch.equal(widened[..., :11], legacy)
    assert torch.all(widened[..., 11] == 0.0)


def test_match_past_width_is_noop_when_already_target():
    x = torch.zeros(2, PAST_T, 12)
    assert _match_past_width(x, 12) is x


def test_match_past_width_rejects_unknown_width():
    with pytest.raises(ValueError):
        _match_past_width(torch.zeros(2, PAST_T, 9), 12)


def test_legacy_11col_db_loads_and_widens(tmp_path):
    past = np.zeros((2, PAST_T, 11), dtype=np.float32)
    past[:, -1, 0] = 1.0  # nonzero position -> "valid"
    past[0, -1, 8] = 1.0  # vehicle
    past[1, -1, 9] = 1.0  # pedestrian
    future = np.zeros((2, OUTPUT_T, 4), dtype=np.float32)

    db_path = _write_db(tmp_path, past, future)
    db = NeighborPatternDB(db_path, **_kwargs())

    assert db.past.shape[-1] == 12
    assert torch.all(db.past[..., 11] == 0.0)
    # A legacy DB has no Unknown-labeled pattern, so is_vru must match the original
    # 3-way pedestrian/bicycle-only definition.
    assert db.is_vru.tolist() == [False, True]


def test_unknown_typed_pattern_is_treated_as_vru(tmp_path):
    past = np.zeros((3, PAST_T, 12), dtype=np.float32)
    past[:, -1, 0] = 1.0
    past[0, -1, 8] = 1.0  # vehicle
    past[1, -1, 10] = 1.0  # bicycle
    past[2, -1, 11] = 1.0  # unknown
    future = np.zeros((3, OUTPUT_T, 4), dtype=np.float32)

    db_path = _write_db(tmp_path, past, future)
    db = NeighborPatternDB(db_path, **_kwargs())

    assert db.past.shape[-1] == 12
    assert db.is_vru.tolist() == [False, True, True]
