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


def test_metric_totals_are_converted_to_sample_weighted_means():
    token_importance = _load_script("token_importance")
    totals = {
        "fde_top": 9.0,
        "ade_top": 6.0,
        "min_fde": 3.0,
        "min_ade": 1.5,
    }

    metrics = token_importance.reduce_metric_totals(totals, count=3, device="cpu", world_size=1)

    assert metrics == {
        "fde_top": 3.0,
        "ade_top": 2.0,
        "min_fde": 1.0,
        "min_ade": 0.5,
    }


def test_attention_payload_merge_preserves_rank_local_alignment():
    attention_analysis = _load_script("attention_analysis")

    def payload(value, turning):
        return {
            "turning": [turning],
            "ego_share": [{name: [value] for name, _ in attention_analysis.CLASSES}],
            "all_share": [{name: [value + 1] for name, _ in attention_analysis.CLASSES}],
            "vw_share": [{name: [value + 2] for name, _ in attention_analysis.CLASSES}],
            "count_share": {name: [value + 3] for name, _ in attention_analysis.CLASSES},
            "valid_count": {name: [value + 4] for name, _ in attention_analysis.CLASSES},
            "total_valid_count": [value + 5],
            "lane_bin_share": [[value] for _ in attention_analysis.DIST_BINS],
            "nbr_bin_share": [[value + 1] for _ in attention_analysis.DIST_BINS],
            "route_share_per_sample": [value + 6],
        }

    merged = attention_analysis.merge_payloads([payload(10, False), payload(20, True)], n_layers=1)

    assert merged["turning"] == [False, True]
    assert merged["route_share_per_sample"] == [16, 26]
    assert merged["ego_share"][0]["neighbors"] == [10, 20]
    assert merged["valid_count"]["lanes"] == [14, 24]
