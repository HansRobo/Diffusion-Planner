"""ObservationNormalizer.from_json must transparently pad a real, on-disk 11-entry
neighbor_agents_past mean/std (the actual normalization.json format today -- these files
are not edited as part of the Unknown-class change) to 12 entries, so it lines up with the
model's 12-wide neighbor_agents_past without needing every normalization.json regenerated."""

import json

import torch

from diffusion_planner.utils.normalizer import ObservationNormalizer


def _write_normalization_json(path, neighbor_cols: int):
    neighbor_mean = [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][:neighbor_cols]
    neighbor_std = [20, 20, 1, 1, 20, 20, 20, 20, 1, 1, 1][:neighbor_cols]
    data = {
        "ego": {"mean": [10, 0, 0, 0], "std": [20, 20, 1, 1]},
        "neighbor": {"mean": [10, 0, 0, 0], "std": [20, 20, 1, 1]},
        "neighbor_agents_past": {"mean": neighbor_mean, "std": neighbor_std},
        "static_objects": {"mean": [0] * 10, "std": [1] * 10},
    }
    path.write_text(json.dumps(data))


def test_pads_legacy_11_entry_stats_to_12(tmp_path):
    p = tmp_path / "normalization.json"
    _write_normalization_json(p, neighbor_cols=11)

    norm = ObservationNormalizer.from_json(str(p))
    stats = norm._normalization_dict["neighbor_agents_past"]
    assert stats["mean"].shape == (12,)
    assert stats["std"].shape == (12,)
    # The padded column is a pass-through: mean=0, std=1.
    assert stats["mean"][11].item() == 0.0
    assert stats["std"][11].item() == 1.0
    # The original 11 entries are untouched.
    assert torch.equal(stats["mean"][:11], torch.tensor([10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0]))


def test_already_12_entry_stats_pass_through_unchanged(tmp_path):
    p = tmp_path / "normalization.json"
    _write_normalization_json(p, neighbor_cols=11)
    data = json.loads(p.read_text())
    data["neighbor_agents_past"]["mean"].append(0.0)
    data["neighbor_agents_past"]["std"].append(1.0)
    p.write_text(json.dumps(data))

    norm = ObservationNormalizer.from_json(str(p))
    stats = norm._normalization_dict["neighbor_agents_past"]
    assert stats["mean"].shape == (12,)
    assert stats["std"].shape == (12,)


def test_normalizer_applies_correctly_to_12wide_batch(tmp_path):
    """End-to-end: a real 11-entry normalization.json must correctly normalize a 12-wide
    neighbor_agents_past batch (the shape DiffusionPlannerData now produces)."""
    p = tmp_path / "normalization.json"
    _write_normalization_json(p, neighbor_cols=11)
    norm = ObservationNormalizer.from_json(str(p))

    batch = torch.zeros(1, 1, 1, 12)
    batch[..., 0] = 30.0  # x, mean=10 std=20 -> (30-10)/20 = 1.0
    batch[..., 11] = 1.0  # unknown one-hot, mean=0 std=1 -> unchanged

    out = norm({"neighbor_agents_past": batch})
    assert torch.allclose(out["neighbor_agents_past"][..., 0], torch.tensor(1.0))
    assert torch.allclose(out["neighbor_agents_past"][..., 11], torch.tensor(1.0))
