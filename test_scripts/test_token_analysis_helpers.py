"""Regression tests for the token-analysis helper functions."""

import importlib.util
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_neighbor_distance_ignores_history_discarded_by_encoder():
    token_importance = _load_script("token_importance")
    nbr = torch.zeros(1, 2, 10, 11)
    nbr[0, 0, 0, 0] = 3.0  # outside NeighborEncoder's retained window
    nbr[0, 1, -1, :2] = torch.tensor([3.0, 4.0])

    dist = token_importance._neighbor_dist(nbr)

    assert torch.isinf(dist[0, 0])
    assert dist[0, 1].item() == 5.0


def test_onnx_model_returns_prediction_on_requested_device():
    token_importance = _load_script("token_importance")
    model = object.__new__(token_importance.OnnxModel)
    model.device = "cpu"
    model.input_specs = []

    class Session:
        @staticmethod
        def run(_names, _feed):
            return [np.zeros((2, 1, 3, 4), dtype=np.float32)]

    model.sess = Session()
    _, outputs = model({"ego_current_state": torch.zeros(2, 10)})

    assert outputs["prediction"].device.type == "cpu"


def test_token_occupancy_statistics_include_capacity_and_utilization():
    attention_analysis = _load_script("attention_analysis")

    stats = attention_analysis.occupancy_stats([0, 1, 2, 4], slots=8)

    assert stats["mean"] == 1.75
    assert stats["max"] == 4
    assert stats["mean_utilization_pct"] == 21.875
    assert stats["p95_capacity"] == 4
    assert stats["p99_capacity"] == 4
