import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import scenario_generation.replay as replay
from planner_metrics.config import RewardConfig
from scenario_generation.reproducer_rollout import _add_static_inputs


def test_reproducer_static_inputs_match_temporal_hdp_shape():
    data = {}
    args = SimpleNamespace(predicted_neighbor_num=0, future_len=80)

    _add_static_inputs(data, args, n=3, device="cpu")

    assert set(data) == {"sampled_trajectories"}
    assert data["sampled_trajectories"].shape == (3, 1, 80, 4)


def test_replay_reward_serializes_scalar_and_per_timestep_subscores(monkeypatch):
    monkeypatch.setattr(
        replay,
        "compute_subscores_batch",
        lambda *_args: {
            "safety": torch.tensor([0.8, 0.7]),
            "ttc_unsafe_at_t": torch.tensor([[False, True], [False, False]]),
            "ttc_min_clearance": torch.tensor([[2.0, 0.0], [3.0, 4.0]]),
            "lane_crossing_steps": [None, 4],
        },
    )

    result = replay.compute_reward_batch(torch.zeros(2, 2, 4), {}, RewardConfig())

    assert result[0]["safety"] == pytest.approx(0.8)
    assert result[0]["ttc_unsafe_at_t"] == [False, True]
    assert result[1]["ttc_min_clearance"] == [3.0, 4.0]
    assert not result[0]["lane_crossing"]
    assert result[1]["lane_crossing"]


def test_replay_reward_json_replaces_all_nonfinite_values_with_null():
    cleaned = replay.reward_breakdown_to_json_dict(
        {
            "scalar": float("inf"),
            "nested": [np.float32("nan"), {"value": torch.tensor(float("-inf"))}],
            "finite": torch.tensor([1.0, 2.0]),
            "array": np.array([1.0, np.inf]),
        }
    )

    assert cleaned == {
        "scalar": None,
        "nested": [None, {"value": None}],
        "finite": [1.0, 2.0],
        "array": [1.0, None],
    }
    json.dumps(cleaned, allow_nan=False)
