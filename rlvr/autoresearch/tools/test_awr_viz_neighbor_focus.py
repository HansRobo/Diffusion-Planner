"""Regression tests for frame-local neighbor selection in final OBB visuals."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rlvr.autoresearch.tools.awr_viz import _draw_neighbor_boxes


def _scene_with_late_approach() -> dict[str, np.ndarray]:
    past = np.zeros((2, 1, 11), dtype=np.float32)
    past[:, 0, 2] = 1.0  # heading x
    past[:, 0, 6] = 1.8  # width
    past[:, 0, 7] = 4.5  # length
    past[:, 0, 8] = 1.0  # vehicle type
    past[0, 0, 0] = 1.0
    past[1, 0, 0] = 50.0

    future = np.zeros((2, 2, 4), dtype=np.float32)
    future[:, :, 2] = 1.0
    future[0, :, 0] = np.array([80.0, 100.0])
    future[1, :, 0] = np.array([4.0, 2.0])
    return {"neighbor_agents_past": past, "neighbor_agents_future": future}


def test_focus_uses_frame_local_distance_while_default_preserves_t0_ranking() -> None:
    scene = _scene_with_late_approach()
    fig, (default_ax, focus_ax) = plt.subplots(1, 2)
    try:
        default_points = _draw_neighbor_boxes(
            default_ax, scene, frame=1, max_neighbors=1, trails=False
        )
        focused_points = _draw_neighbor_boxes(
            focus_ax,
            scene,
            frame=1,
            max_neighbors=1,
            trails=False,
            focus_xy=np.array([0.0, 0.0]),
        )
    finally:
        plt.close(fig)

    # Default chooses the actor that was nearest at t=0 even though its frame-1
    # box is now at x=100.  Event focus chooses the late-approaching actor at x=2.
    assert np.mean(default_points[0], axis=0)[0] == 100.0
    assert np.mean(focused_points[0], axis=0)[0] == 2.0

