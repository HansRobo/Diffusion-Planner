from pathlib import Path

import numpy as np
import pytest
import torch

from preference_optimization.utils import (
    X2_LEGACY_EGO_WIDTH_M,
    load_npz_data,
)


def _write_scene(path: Path, width: float) -> None:
    path.parent.mkdir(parents=True)
    np.savez(path, ego_shape=np.asarray([3.0, 4.9, width], dtype=np.float32))


@pytest.mark.parametrize("dataset_name", ["x2", "x2_dev"])
def test_x2_width_is_normalized_for_original_dp(tmp_path: Path, dataset_name: str) -> None:
    scene = tmp_path / dataset_name / "train" / "scene.npz"
    _write_scene(scene, width=2.42741)

    loaded = load_npz_data(scene, torch.device("cpu"))

    assert loaded["ego_shape"][0, 2].item() == pytest.approx(X2_LEGACY_EGO_WIDTH_M)


@pytest.mark.parametrize("dataset_name", ["xx1_real_train", "j6_real_train"])
def test_non_x2_width_is_unchanged(tmp_path: Path, dataset_name: str) -> None:
    scene = tmp_path / dataset_name / "train" / "scene.npz"
    _write_scene(scene, width=2.42741)

    loaded = load_npz_data(scene, torch.device("cpu"))

    assert loaded["ego_shape"][0, 2].item() == pytest.approx(2.42741)
