import json

import numpy as np
import pytest
from diffusion_planner.utils.dataset import DiffusionPlannerData


def _dataset_for_sample(tmp_path, vehicle_dir: str, ego_shape: np.ndarray):
    sample_dir = tmp_path / vehicle_dir / "train"
    sample_dir.mkdir(parents=True)
    sample_path = sample_dir / "sample.npz"
    np.savez(sample_path, ego_shape=ego_shape, marker=np.array([7], dtype=np.int32))
    manifest = tmp_path / f"{vehicle_dir}_manifest.json"
    manifest.write_text(json.dumps([str(sample_path)]), encoding="utf-8")
    return DiffusionPlannerData(str(manifest)), sample_path


@pytest.mark.parametrize("vehicle_dir", ["x2", "x2_dev"])
def test_x2_width_is_preserved_from_new_npz_data(tmp_path, vehicle_dir):
    original = np.array([4.76012, 7.23690, 2.42741], dtype=np.float32)
    dataset, sample_path = _dataset_for_sample(tmp_path, vehicle_dir, original)

    loaded = dataset[0]

    np.testing.assert_array_equal(loaded["ego_shape"], original)
    assert loaded["ego_shape"].dtype == np.float32
    assert loaded["marker"].item() == 7
    with np.load(sample_path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["ego_shape"], original)


def test_non_x2_width_is_not_changed(tmp_path):
    original = np.array([3.8, 6.2, 2.4], dtype=np.float32)
    dataset, _ = _dataset_for_sample(tmp_path, "erga", original)

    np.testing.assert_array_equal(dataset[0]["ego_shape"], original)


def test_malformed_x2_ego_shape_fails_loudly(tmp_path):
    dataset, _ = _dataset_for_sample(
        tmp_path, "x2_dev", np.array([4.76012, 7.23690], dtype=np.float32)
    )

    with pytest.raises(ValueError, match=r"ego_shape with shape \(2,\), expected \(3,\)"):
        dataset[0]
