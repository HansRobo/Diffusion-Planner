"""RouteAugmentation: route tail truncation must drop exactly the segments
starting beyond the sampled arc-length budget (keeping the first segment and
the all-zero padding convention, with the speed-limit tensors kept in sync),
and speed-limit unknown dropout must clear lanes and route together."""

import torch
from diffusion_planner.utils.route_augmentation import RouteAugmentation

_NUM_SEGMENTS = 25
_POINTS = 20
_DIM = 33


def _make_route_inputs(num_valid: int, seg_len_m: float, batch: int = 1):
    """Straight route along +x: ``num_valid`` segments of ``seg_len_m`` each."""
    route = torch.zeros(batch, _NUM_SEGMENTS, _POINTS, _DIM)
    step = seg_len_m / (_POINTS - 1)
    for s in range(num_valid):
        x0 = s * seg_len_m
        xs = x0 + torch.arange(_POINTS) * step
        route[:, s, :, 0] = xs
        route[:, s, :, 1] = 1.0  # keep y non-zero so x=0 points stay valid
        route[:, s, :, 2] = step  # dx
    speed_limit = torch.zeros(batch, _NUM_SEGMENTS, 1)
    speed_limit[:, :num_valid] = 10.0
    has_speed_limit = torch.zeros(batch, _NUM_SEGMENTS, 1, dtype=torch.bool)
    has_speed_limit[:, :num_valid] = True
    lanes_speed_limit = torch.full((batch, 140, 1), 8.0)
    lanes_has_speed_limit = torch.ones(batch, 140, 1, dtype=torch.bool)
    return {
        "route_lanes": route,
        "route_lanes_speed_limit": speed_limit,
        "route_lanes_has_speed_limit": has_speed_limit,
        "lanes_speed_limit": lanes_speed_limit,
        "lanes_has_speed_limit": lanes_has_speed_limit,
    }


def _truncation_aug(min_m: float, max_m: float, prob: float = 1.0):
    return RouteAugmentation(
        device="cpu",
        truncation_prob=prob,
        truncation_min_m=min_m,
        truncation_max_m=max_m,
        speed_limit_unknown_prob=0.0,
    )


def test_truncation_drops_far_segments_and_keeps_near():
    # 5 segments x 50m; min == max == 120m makes the budget deterministic:
    # segments starting at 0/50/100m stay, 150/200m are dropped.
    inputs = _make_route_inputs(num_valid=5, seg_len_m=50.0)
    aug = _truncation_aug(120.0, 120.0)
    out = aug(inputs)

    kept = out["route_lanes"].abs().sum(dim=(2, 3)) > 0  # [B, P]
    assert kept[0, :3].all()
    assert not kept[0, 3:].any()
    # Speed-limit tensors follow the dropped rows.
    assert out["route_lanes_has_speed_limit"][0, :3].all()
    assert not out["route_lanes_has_speed_limit"][0, 3:].any()
    assert torch.all(out["route_lanes_speed_limit"][0, 3:] == 0.0)


def test_truncation_always_keeps_first_segment():
    # A single 400m segment starts at 0m and must survive any budget.
    inputs = _make_route_inputs(num_valid=3, seg_len_m=400.0)
    aug = _truncation_aug(60.0, 60.0)
    out = aug(inputs)
    kept = out["route_lanes"].abs().sum(dim=(2, 3)) > 0
    assert kept[0, 0]
    assert not kept[0, 1:].any()


def test_truncation_prob_zero_is_identity():
    inputs = _make_route_inputs(num_valid=5, seg_len_m=50.0)
    expected = {k: v.clone() for k, v in inputs.items()}
    aug = RouteAugmentation(
        device="cpu",
        truncation_prob=0.0,
        speed_limit_unknown_prob=0.0,
    )
    out = aug(inputs)
    for k in expected:
        assert torch.equal(out[k], expected[k]), k


def test_truncation_preserves_padding_rows():
    inputs = _make_route_inputs(num_valid=2, seg_len_m=30.0)
    aug = _truncation_aug(200.0, 200.0)
    out = aug(inputs)
    # Nothing exceeds the budget; valid rows unchanged, padding still zero.
    kept = out["route_lanes"].abs().sum(dim=(2, 3)) > 0
    assert kept[0, :2].all()
    assert torch.all(out["route_lanes"][0, 2:] == 0.0)


def test_speed_limit_unknown_clears_lanes_and_route_together():
    inputs = _make_route_inputs(num_valid=4, seg_len_m=50.0)
    aug = RouteAugmentation(
        device="cpu",
        truncation_prob=0.0,
        speed_limit_unknown_prob=1.0,
    )
    out = aug(inputs)
    for prefix in ("lanes", "route_lanes"):
        assert not out[f"{prefix}_has_speed_limit"].any(), prefix
        assert torch.all(out[f"{prefix}_speed_limit"] == 0.0), prefix
    # Route geometry itself is untouched by the speed-limit dropout.
    kept = out["route_lanes"].abs().sum(dim=(2, 3)) > 0
    assert kept[0, :4].all()


def test_speed_limit_unknown_prob_zero_is_identity():
    inputs = _make_route_inputs(num_valid=4, seg_len_m=50.0)
    expected = {k: v.clone() for k, v in inputs.items()}
    aug = RouteAugmentation(
        device="cpu",
        truncation_prob=0.0,
        speed_limit_unknown_prob=0.0,
    )
    out = aug(inputs)
    for k in expected:
        assert torch.equal(out[k], expected[k]), k
