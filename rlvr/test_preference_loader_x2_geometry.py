"""Body-width normalization at the canonical RL loader boundary.

The released v5 Original-DP checkpoint was trained on the pre-rewrite archives.
20260707 rewrote *only* the width to mirror-inclusive values, so RL must revert
the width — and it must do so for **every** vehicle family the checkpoint saw,
not just the directories literally named ``x2``:

    x2 / x2_dev / j6_real_train         -> 2.29156  (all X2 geometry)
    xx1_real_train / xx1_psim / prd_jt  -> 1.70

``j6_real_train`` is the trap: the name says J6 but the archives carry X2
dimensions (rewrite_ego_shape.py maps it to the x2_dev parameter key, and its
pre-rewrite width was already 2.29156).  An earlier version of this module
normalized only ``x2``/``x2_dev``, which left 16.85% of the corpus feeding the
old model a 5.9% wider X2 and 74.3% an 8.2% wider xx1.  Width drives the OBB
collision and road-border terms, so the bias was systematic.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from preference_optimization.utils import (
    X2_LEGACY_EGO_WIDTH_M,
    XX1_LEGACY_EGO_WIDTH_M,
    effective_ego_shape_numpy,
    legacy_ego_width_m,
    load_npz_data,
)

X2_COMPONENTS = ["x2", "x2_dev", "j6_real_train"]
XX1_COMPONENTS = ["xx1_real_train", "xx1_psim", "prd_jt"]


def _write_scene(path: Path, width: float) -> None:
    path.parent.mkdir(parents=True)
    np.savez(path, ego_shape=np.asarray([3.0, 4.9, width], dtype=np.float32))


@pytest.mark.parametrize("dataset_name", X2_COMPONENTS)
def test_x2_family_width_is_reverted(tmp_path: Path, dataset_name: str) -> None:
    scene = tmp_path / dataset_name / "train" / "scene.npz"
    _write_scene(scene, width=2.42741)

    loaded = load_npz_data(scene, torch.device("cpu"))

    assert loaded["ego_shape"][0, 2].item() == pytest.approx(X2_LEGACY_EGO_WIDTH_M)


@pytest.mark.parametrize("dataset_name", XX1_COMPONENTS)
def test_xx1_family_width_is_reverted(tmp_path: Path, dataset_name: str) -> None:
    scene = tmp_path / dataset_name / "train" / "scene.npz"
    _write_scene(scene, width=1.84)

    loaded = load_npz_data(scene, torch.device("cpu"))

    assert loaded["ego_shape"][0, 2].item() == pytest.approx(XX1_LEGACY_EGO_WIDTH_M)


@pytest.mark.parametrize("dataset_name", X2_COMPONENTS + XX1_COMPONENTS)
def test_wheel_base_and_length_are_never_touched(
    tmp_path: Path, dataset_name: str
) -> None:
    """The rewrite only changed width; reverting anything else would be wrong."""
    scene = tmp_path / dataset_name / "train" / "scene.npz"
    _write_scene(scene, width=2.42741)

    loaded = load_npz_data(scene, torch.device("cpu"))

    assert loaded["ego_shape"][0, 0].item() == pytest.approx(3.0)
    assert loaded["ego_shape"][0, 1].item() == pytest.approx(4.9)


def test_unknown_component_is_left_alone(tmp_path: Path) -> None:
    scene = tmp_path / "some_future_vehicle" / "train" / "scene.npz"
    _write_scene(scene, width=2.0)

    assert legacy_ego_width_m(scene) is None
    loaded = load_npz_data(scene, torch.device("cpu"))
    assert loaded["ego_shape"][0, 2].item() == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("dataset_name", "expected"),
    [(c, X2_LEGACY_EGO_WIDTH_M) for c in X2_COMPONENTS]
    + [(c, XX1_LEGACY_EGO_WIDTH_M) for c in XX1_COMPONENTS],
)
def test_visualisers_see_the_same_width_as_the_reward(
    tmp_path: Path, dataset_name: str, expected: float
) -> None:
    scene = tmp_path / dataset_name / "valid" / "scene.npz"
    raw = np.asarray([3.0, 4.9, 2.42741], dtype=np.float32)

    effective = effective_ego_shape_numpy(scene, raw)

    assert effective.tolist() == pytest.approx([3.0, 4.9, expected])
    assert raw[2] == pytest.approx(2.42741), "helper must not mutate the NPZ array"
