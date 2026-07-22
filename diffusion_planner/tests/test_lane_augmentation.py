"""LaneAugmentation: route-copied lanes must survive every component
untouched (exact-match protection mask, with a nearest-lane fallback when the
route is empty), far-lane truncation and dropout must zero exactly the
expected non-route rows with the speed-limit tensors in sync, geometry noise
must be a constant lateral offset that leaves direction channels alone, and
width jitter must only scale the boundary offsets within its clip range."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from diffusion_planner.utils.lane_augmentation import LaneAugmentation

_NUM_LANES = 140
_NUM_ROUTE = 25
_POINTS = 20
_DIM = 33

_MINI_DATASET_LIST = Path(
    "/home/isamuyamashita/work/diffusion_plannner/mini_datasets/"
    "j6_2231_fullseq_mini_20260707/path_list_train.json"
)


def _make_inputs(num_lanes: int = 6, num_route: int = 2, batch: int = 1):
    """Straight lanes along +x at distinct y offsets; the first ``num_route``
    lanes are bit-identical copies in ``route_lanes`` (like the converter)."""
    lanes = torch.zeros(batch, _NUM_LANES, _POINTS, _DIM)
    step = 50.0 / (_POINTS - 1)
    for s in range(num_lanes):
        lanes[:, s, :, 0] = torch.arange(_POINTS) * step  # x from ego
        lanes[:, s, :, 1] = 3.5 * s + 1.0  # lateral offset per lane
        lanes[:, s, :, 2] = step  # dx
        lanes[:, s, :, 4:6] = torch.tensor([0.0, 1.75])  # left boundary offset
        lanes[:, s, :, 6:8] = torch.tensor([0.0, -1.75])  # right boundary offset
    route = torch.zeros(batch, _NUM_ROUTE, _POINTS, _DIM)
    route[:, :num_route] = lanes[:, :num_route]

    def limits(n_slots, n_valid):
        v = torch.zeros(batch, n_slots, 1)
        v[:, :n_valid] = 10.0
        h = torch.zeros(batch, n_slots, 1, dtype=torch.bool)
        h[:, :n_valid] = True
        return v, h

    lanes_sl, lanes_has = limits(_NUM_LANES, num_lanes)
    route_sl, route_has = limits(_NUM_ROUTE, num_route)
    return {
        "lanes": lanes,
        "lanes_speed_limit": lanes_sl,
        "lanes_has_speed_limit": lanes_has,
        "route_lanes": route,
        "route_lanes_speed_limit": route_sl,
        "route_lanes_has_speed_limit": route_has,
    }


def test_dropout_protects_route_copies():
    inputs = _make_inputs(num_lanes=6, num_route=2)
    expected_route_rows = inputs["lanes"][0, :2].clone()
    aug = LaneAugmentation(device="cpu", dropout_prob=1.0, dropout_ratio=1.0)
    out = aug(inputs)

    kept = out["lanes"].abs().sum(dim=(2, 3)) > 0
    # Route copies survive; every other valid lane is dropped (ratio=1).
    assert kept[0, :2].all()
    assert not kept[0, 2:].any()
    assert torch.equal(out["lanes"][0, :2], expected_route_rows)
    # Speed-limit tensors follow the dropped rows.
    assert out["lanes_has_speed_limit"][0, :2].all()
    assert not out["lanes_has_speed_limit"][0, 2:].any()


def test_dropout_fallback_protects_nearest_lane_without_route():
    inputs = _make_inputs(num_lanes=4, num_route=0)
    aug = LaneAugmentation(device="cpu", dropout_prob=1.0, dropout_ratio=1.0)
    out = aug(inputs)
    kept = out["lanes"].abs().sum(dim=(2, 3)) > 0
    # Lane 0 has the smallest lateral offset (y=1.0) -> nearest to ego.
    assert kept[0, 0]
    assert not kept[0, 1:].any()


def test_truncation_drops_only_far_lanes():
    inputs = _make_inputs(num_lanes=6, num_route=2)
    # Lane s sits at y = 3.5s + 1; radius 12 keeps s<=3 (y<=11.5) plus the
    # protected route copies, drops s=4 (15.0) and s=5 (18.5).
    aug = LaneAugmentation(
        device="cpu", truncation_prob=1.0, truncation_min_m=12.0, truncation_max_m=12.0
    )
    out = aug(inputs)
    kept = out["lanes"].abs().sum(dim=(2, 3)) > 0
    assert kept[0, :4].all()
    assert not kept[0, 4:].any()
    assert not out["lanes_has_speed_limit"][0, 4:].any()


def test_truncation_never_drops_protected_route_copy():
    inputs = _make_inputs(num_lanes=3, num_route=3)
    aug = LaneAugmentation(
        device="cpu", truncation_prob=1.0, truncation_min_m=0.1, truncation_max_m=0.1
    )
    out = aug(inputs)
    kept = out["lanes"].abs().sum(dim=(2, 3)) > 0
    assert kept[0, :3].all()  # all lanes are route copies -> nothing dropped


def test_geometry_noise_skips_route_copies_and_keeps_directions():
    inputs = _make_inputs(num_lanes=5, num_route=2)
    orig = inputs["lanes"].clone()
    torch.manual_seed(0)
    aug = LaneAugmentation(device="cpu", geometry_noise_prob=1.0, geometry_noise_std_m=1.0)
    out = aug(inputs)
    lanes = out["lanes"]

    # Route copies untouched.
    assert torch.equal(lanes[0, :2], orig[0, :2])
    moved = 0
    for seg in range(2, 5):
        delta = lanes[0, seg, :, :2] - orig[0, seg, :, :2]
        assert torch.allclose(delta, delta[0].expand_as(delta), atol=1e-6)
        # Lanes run along +x -> offset must be purely lateral (y only).
        assert torch.allclose(delta[:, 0], torch.zeros_like(delta[:, 0]), atol=1e-6)
        moved += int(delta[0].norm() > 1e-3)
    assert moved == 3
    assert torch.equal(lanes[..., 2:], orig[..., 2:])
    assert torch.all(lanes[0, 5:] == 0.0)


def test_width_jitter_scales_only_boundaries_within_clip():
    inputs = _make_inputs(num_lanes=5, num_route=2)
    orig = inputs["lanes"].clone()
    torch.manual_seed(0)
    aug = LaneAugmentation(device="cpu", width_jitter_prob=1.0, width_jitter_std=0.5)
    out = aug(inputs)
    lanes = out["lanes"]

    # Centerline, directions, and one-hots untouched; route copies untouched.
    assert torch.equal(lanes[..., :4], orig[..., :4])
    assert torch.equal(lanes[..., 8:], orig[..., 8:])
    assert torch.equal(lanes[0, :2], orig[0, :2])
    for seg in range(2, 5):
        for start in (4, 6):
            ratio = lanes[0, seg, :, start + 1] / orig[0, seg, :, start + 1]
            assert torch.allclose(ratio, ratio[0].expand_as(ratio), atol=1e-6)
            assert 0.85 - 1e-6 <= float(ratio[0]) <= 1.15 + 1e-6
    assert torch.all(lanes[0, 5:] == 0.0)


def test_all_probs_zero_is_identity():
    inputs = _make_inputs(num_lanes=5, num_route=2)
    expected = {k: v.clone() for k, v in inputs.items()}
    out = LaneAugmentation(device="cpu")(inputs)
    for k in expected:
        assert torch.equal(out[k], expected[k]), k


@pytest.mark.skipif(not _MINI_DATASET_LIST.exists(), reason="mini dataset not available")
def test_route_copies_match_lanes_exactly_on_real_data():
    """Regression guard for the exact-match protection mask: if the conversion
    pipeline ever stops writing route lanelets bit-identically into ``lanes``,
    the protection breaks and this test catches it."""
    paths = json.loads(_MINI_DATASET_LIST.read_text())[::1024][:4]
    for p in paths:
        d = np.load(p)
        lanes = torch.tensor(d["lanes"]).unsqueeze(0)
        route = torch.tensor(d["route_lanes"]).unsqueeze(0)
        route_valid = route[..., :8].abs().sum(dim=(2, 3)) > 0
        eq = (lanes[:, :, None, 0, :2] == route[:, None, :, 0, :2]).all(-1)
        matched = (eq.any(1) | ~route_valid).all()
        assert bool(matched), f"unmatched route segment in {p}"
