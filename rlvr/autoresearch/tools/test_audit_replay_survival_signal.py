from __future__ import annotations

import numpy as np

from rlvr.autoresearch.tools.audit_replay_survival_signal import (
    _select_constrained_terminal_repair,
)


def _select(
    terminal_steps,
    progress,
    path_length,
    hard_reasons,
    **overrides,
):
    settings = {
        "min_delay_steps": 1,
        "progress_tolerance": 0.02,
        "path_length_tolerance_m": 0.2,
        "require_no_new_hard_reason": True,
    }
    settings.update(overrides)
    return _select_constrained_terminal_repair(
        terminal_steps=np.asarray(terminal_steps, dtype=np.int64),
        survival_total=np.asarray([0.0, 0.2, 0.1], dtype=np.float64),
        progress=np.asarray(progress, dtype=np.float64),
        path_length=np.asarray(path_length, dtype=np.float64),
        hard_reasons=[set(value) for value in hard_reasons],
        **settings,
    )


def test_prefers_largest_safe_delay() -> None:
    assert (
        _select(
            [10, 12, 15],
            [0.5, 0.5, 0.49],
            [8.0, 8.0, 7.9],
            [{"road_border"}, {"road_border"}, {"road_border"}],
        )
        == 2
    )


def test_rejects_stopping_and_new_hard_event() -> None:
    assert (
        _select(
            [10, 20, 18],
            [0.5, 0.3, 0.5],
            [8.0, 5.0, 8.0],
            [{"road_border"}, {"road_border"}, {"collision"}],
        )
        is None
    )


def test_requires_strict_terminal_improvement() -> None:
    assert (
        _select(
            [10, 10, 10],
            [0.5, 0.6, 0.6],
            [8.0, 9.0, 9.0],
            [{"collision"}, {"collision"}, {"collision"}],
        )
        is None
    )
