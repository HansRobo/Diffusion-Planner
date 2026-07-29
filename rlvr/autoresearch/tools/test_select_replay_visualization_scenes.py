import json

import numpy as np
import pytest

from rlvr.autoresearch.tools.select_replay_visualization_scenes import (
    _endpoint_route_coordinates,
    select,
)


def test_route_coordinates_do_not_call_progress_on_a_curve_lateral():
    trajectories = np.zeros((1, 3, 4, 4), dtype=np.float32)
    reference = np.asarray([[1, 0], [2, 0], [2, 1], [2, 2]], dtype=np.float32)
    trajectories[0, 0, :, :2] = reference
    trajectories[0, 0, :, 2:4] = [[1, 0], [1, 0], [0, 1], [0, 1]]
    trajectories[0, 1, -1, :2] = [2, 1]
    trajectories[0, 2, -1, :2] = [3, 2]

    s, d = _endpoint_route_coordinates(trajectories)

    assert np.isclose(s[0, 0] - s[0, 1], 1.0)
    assert np.isclose(d[0, 1], 0.0)
    assert np.isclose(s[0, 2], s[0, 0])
    assert np.isclose(abs(d[0, 2]), 1.0)


def test_selects_complementary_committed_prefix_categories(tmp_path):
    root = tmp_path / "replay"
    rank = root / "rank_0000"
    rank.mkdir(parents=True)
    rewards = np.asarray(
        [[0.0, 0.95, 0.2], [0.5, 0.6, 0.9], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        dtype=np.float32,
    )
    trajectories = np.zeros((4, 3, 2, 3), dtype=np.float32)
    trajectories[0, :, -1, :2] = [[5, 0], [5, 2], [5, -1]]
    trajectories[1, :, -1, :2] = [[2, 0], [3, 0], [6, 0]]
    trajectories[2, :, -1, :2] = [[1, 0], [4, 0], [1, 3]]
    np.save(rank / "rewards.npy", rewards)
    np.save(rank / "trajectories.npy", trajectories)
    (rank / "scene_paths.jsonl").write_text(
        "".join(json.dumps(f"/scene/{index}.npz") + "\n" for index in range(4))
    )

    with pytest.raises(RuntimeError, match="not strictly closed"):
        select(root)

    report = select(root, per_category=2, chunk_size=2, allow_partial=True)
    assert report["prefix_groups"] == 4
    assert not report["source_complete"]
    assert report["group_size"] == 3
    assert report["future_len"] == 2
    categories = report["categories"]
    assert categories["deterministic_hard_recovery"][0]["rank_local_index"] == 0
    assert categories["lateral_multimodality"][0]["rank_local_index"] == 0
    assert categories["longitudinal_multimodality"][0]["rank_local_index"] == 1
    assert categories["largest_reward_gain"][0]["rank_local_index"] == 0
    assert categories["all_zero_diverse_failure"][0]["rank_local_index"] == 2
    assert categories["deterministic_hard_recovery"][0]["scene_path"] == "/scene/0.npz"
