# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for diffusion_planner/utils/neighbor_type_augment.py and the related
"unknown type" margin fix in diffusion_planner/loss.py.

Covers:
- apply_neighbor_unknown_augment: prob=0 no-op, prob=1 drops every eligible neighbor,
  padding slots never touched, non-type columns (0..7) never touched, a dropped
  neighbor's type is zeroed across every timestep (never partial),
  per-class probability matches the configured base rate, distance scaling raises
  the drop rate with range, and the probability cap is respected.
- compute_neighbor_collision_penalty: an all-zero-type ("unknown") neighbor uses
  margin_unknown rather than tie-breaking to margin_vehicle via argmax(all-zero).

Usage:
    python tests/test_neighbor_type_augment.py          # standalone
    pytest tests/test_neighbor_type_augment.py -v        # with pytest
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusion_planner.loss import compute_ego_edge_points, compute_neighbor_collision_penalty
from diffusion_planner.utils.neighbor_type_augment import apply_neighbor_unknown_augment

ATOL = 1e-5

# Column layout reminder (neighbor_agents_past last dim): 0:x 1:y 2:cos 3:sin 4:vx 5:vy
# 6:width 7:length 8:10 one-hot type [vehicle, pedestrian, bicycle].
_VEHICLE, _PEDESTRIAN, _BICYCLE = 0, 1, 2


def _nbr_batch(specs: list[tuple[float, float, int]], T: int = 3) -> torch.Tensor:
    """Build a neighbor_agents_past tensor, one real agent per spec.

    Each spec is (x, y, class_idx) with class_idx in {0: vehicle, 1: pedestrian,
    2: bicycle}. The type one-hot and size are constant across all T timesteps,
    matching how the real preprocessing constructs this tensor.
    """
    Pn = len(specs)
    out = torch.zeros(1, Pn, T, 11, dtype=torch.float32)
    for i, (x, y, cls) in enumerate(specs):
        out[0, i, :, 0] = x
        out[0, i, :, 1] = y
        out[0, i, :, 2] = 1.0  # cos(heading=0)
        out[0, i, :, 6] = 2.0  # width
        out[0, i, :, 7] = 4.0  # length
        out[0, i, :, 8 + cls] = 1.0
    return out


def _run(
    neighbor_agents_past: torch.Tensor,
    prob_vehicle: float = 0.0,
    prob_pedestrian: float = 0.0,
    prob_bicycle: float = 0.0,
    distance_scale_max: float = 1.0,
    distance_scale_range_m: float = 50.0,
    prob_cap: float = 1.0,
    seed: int = 0,
) -> torch.Tensor:
    torch.manual_seed(seed)
    return apply_neighbor_unknown_augment(
        neighbor_agents_past,
        prob_vehicle=prob_vehicle,
        prob_pedestrian=prob_pedestrian,
        prob_bicycle=prob_bicycle,
        distance_scale_max=distance_scale_max,
        distance_scale_range_m=distance_scale_range_m,
        prob_cap=prob_cap,
    )


# ──────────────────────── apply_neighbor_unknown_augment ────────────────────────


def test_prob_zero_is_noop():
    """With every base probability at 0, the output must equal the input exactly."""
    x = _nbr_batch([(5.0, 0.0, _VEHICLE), (10.0, 3.0, _PEDESTRIAN)])
    out = _run(x, prob_vehicle=0.0, prob_pedestrian=0.0, prob_bicycle=0.0)
    assert torch.equal(out, x), "prob=0 must not modify the tensor"


def test_prob_one_drops_every_eligible_neighbor():
    """With every base probability at 1 (and no cap), every real neighbor's type is zeroed."""
    x = _nbr_batch([(1.0, 0.0, _VEHICLE), (2.0, 0.0, _PEDESTRIAN), (3.0, 0.0, _BICYCLE)])
    out = _run(x, prob_vehicle=1.0, prob_pedestrian=1.0, prob_bicycle=1.0, prob_cap=1.0)
    dropped_type = out[..., 8:11]
    assert torch.all(dropped_type == 0.0), "prob=1 must zero the type one-hot for all neighbors"


def test_dropped_type_is_zeroed_across_every_timestep():
    """A dropped neighbor's type must be zero at *every* timestep, not just the last one."""
    x = _nbr_batch([(4.0, 0.0, _VEHICLE)], T=10)
    out = _run(x, prob_vehicle=1.0, prob_cap=1.0)
    assert torch.all(out[0, 0, :, 8:11] == 0.0), "type must be zeroed for the whole history"


def test_padding_slots_never_modified():
    """All-zero padding slots must stay all-zero, even with prob=1 (they are not eligible)."""
    x = _nbr_batch([(4.0, 0.0, _VEHICLE)])
    x = torch.cat([x, torch.zeros(1, 2, x.shape[2], 11)], dim=1)  # append 2 padding slots
    out = _run(x, prob_vehicle=1.0, prob_pedestrian=1.0, prob_bicycle=1.0, prob_cap=1.0)
    assert torch.all(out[0, 1:] == 0.0), "padding slots must remain exactly zero"


def test_non_type_columns_are_never_modified():
    """Position/heading/velocity/size (cols 0..7) must be untouched, whether dropped or not."""
    x = _nbr_batch([(4.0, 1.5, _PEDESTRIAN), (9.0, -2.0, _BICYCLE)])
    out = _run(x, prob_vehicle=1.0, prob_pedestrian=1.0, prob_bicycle=1.0, prob_cap=1.0)
    assert torch.equal(out[..., :8], x[..., :8]), "cols 0..7 must never be modified"


def test_per_class_probability_matches_configured_rate():
    """Empirical drop rate for a single class at distance 0 should match its base prob."""
    N = 20000
    specs = [(0.0, 0.0, _PEDESTRIAN) for _ in range(N)]  # dist=0 -> scale=1, no distance effect
    x = _nbr_batch(specs)
    out = _run(x, prob_pedestrian=0.3, distance_scale_max=1.0, seed=1)
    dropped = (out[0, :, -1, 8:11].sum(dim=-1) == 0).float().mean().item()
    assert abs(dropped - 0.3) < 0.02, f"expected ~0.3 drop rate, got {dropped}"


def test_vehicle_and_pedestrian_probabilities_are_independent():
    """Different classes must use their own configured base probability."""
    N = 20000
    specs = [(0.0, 0.0, _VEHICLE) for _ in range(N // 2)] + [
        (0.0, 0.0, _PEDESTRIAN) for _ in range(N // 2)
    ]
    x = _nbr_batch(specs)
    out = _run(x, prob_vehicle=0.05, prob_pedestrian=0.4, distance_scale_max=1.0, seed=2)
    dropped = out[0, :, -1, 8:11].sum(dim=-1) == 0
    vehicle_rate = dropped[: N // 2].float().mean().item()
    pedestrian_rate = dropped[N // 2 :].float().mean().item()
    assert abs(vehicle_rate - 0.05) < 0.02, f"vehicle rate {vehicle_rate}"
    assert abs(pedestrian_rate - 0.4) < 0.02, f"pedestrian rate {pedestrian_rate}"


def test_distance_scaling_raises_drop_rate():
    """A far neighbor should be dropped near distance_scale_max times as often as a near one."""
    N = 20000
    near = [(0.0, 0.0, _BICYCLE) for _ in range(N)]
    far = [(100.0, 0.0, _BICYCLE) for _ in range(N)]  # >> distance_scale_range_m

    out_near = _run(
        _nbr_batch(near),
        prob_bicycle=0.1,
        distance_scale_max=3.0,
        distance_scale_range_m=50.0,
        prob_cap=1.0,
        seed=3,
    )
    out_far = _run(
        _nbr_batch(far),
        prob_bicycle=0.1,
        distance_scale_max=3.0,
        distance_scale_range_m=50.0,
        prob_cap=1.0,
        seed=4,
    )
    near_rate = (out_near[0, :, -1, 8:11].sum(dim=-1) == 0).float().mean().item()
    far_rate = (out_far[0, :, -1, 8:11].sum(dim=-1) == 0).float().mean().item()
    assert abs(near_rate - 0.1) < 0.02, f"near rate {near_rate}"
    assert abs(far_rate - 0.3) < 0.02, f"far rate {far_rate} (expected ~0.1 * 3.0 scale)"


def test_probability_cap_is_respected():
    """Even with a base prob and distance scale that would exceed 1, the cap must hold."""
    N = 20000
    far = [(500.0, 0.0, _PEDESTRIAN) for _ in range(N)]
    out = _run(
        _nbr_batch(far),
        prob_pedestrian=0.9,
        distance_scale_max=5.0,
        distance_scale_range_m=10.0,
        prob_cap=0.5,
        seed=5,
    )
    rate = (out[0, :, -1, 8:11].sum(dim=-1) == 0).float().mean().item()
    assert rate < 0.52, f"drop rate {rate} exceeded the configured cap of 0.5"


ALL_AUGMENT_TESTS = [
    test_prob_zero_is_noop,
    test_prob_one_drops_every_eligible_neighbor,
    test_dropped_type_is_zeroed_across_every_timestep,
    test_padding_slots_never_modified,
    test_non_type_columns_are_never_modified,
    test_per_class_probability_matches_configured_rate,
    test_vehicle_and_pedestrian_probabilities_are_independent,
    test_distance_scaling_raises_drop_rate,
    test_probability_cap_is_respected,
]


# ──────────────── compute_neighbor_collision_penalty: unknown-type margin ────────────────


def _collision_penalty_for_type(type_onehot: list[float], margin_unknown: float) -> float:
    """Penalty for a single neighbor at x=8 (far from an ego box near the origin), with
    margin_vehicle=margin_pedestrian=margin_bicycle=0.0 and the given margin_unknown. The gap
    at margin 0 is large enough that no known type ever collides; margin_unknown=10 is large
    enough that the all-zero ("unknown") case always does -- so a nonzero penalty here can only
    come from the is_unknown branch selecting margin_unknown instead of tie-breaking to vehicle.
    """
    ego_traj = torch.tensor([[[0.0, 0.0, 1.0, 0.0]]])  # [B=1, T=1, 4]
    ego_shape = torch.tensor([[1.0, 4.0, 2.0]])  # [B=1, 3] (wheelbase, length, width)
    ego_edge_points = compute_ego_edge_points(ego_traj, ego_shape, n_interp=0)  # [1,1,4,2]

    neighbors_future = torch.tensor([[[[8.0, 0.0, 1.0, 0.0]]]])  # [B=1, Pn=1, T=1, 4]
    neighbors_future_valid = torch.tensor([[[True]]])  # [B=1, Pn=1, T=1]

    neighbor_agents_past = torch.zeros(1, 1, 1, 11)
    neighbor_agents_past[0, 0, 0, 6] = 1.0  # width
    neighbor_agents_past[0, 0, 0, 7] = 1.0  # length
    for i, v in enumerate(type_onehot):
        neighbor_agents_past[0, 0, 0, 8 + i] = v

    penalty = compute_neighbor_collision_penalty(
        ego_edge_points,
        neighbors_future,
        neighbors_future_valid,
        neighbor_agents_past,
        margin_vehicle=0.0,
        margin_pedestrian=0.0,
        margin_bicycle=0.0,
        margin_unknown=margin_unknown,
    )
    return penalty[0, 0].item()


def test_known_vehicle_type_unaffected_by_unknown_margin():
    """A real vehicle-typed neighbor must use margin_vehicle (0.0 here), not margin_unknown."""
    penalty = _collision_penalty_for_type([1.0, 0.0, 0.0], margin_unknown=10.0)
    assert penalty == 0.0, f"vehicle-typed neighbor should not collide at margin 0, got {penalty}"


def test_unknown_type_uses_margin_unknown_not_vehicle_tiebreak():
    """An all-zero-type neighbor must use margin_unknown, not silently tie-break to vehicle."""
    penalty = _collision_penalty_for_type([0.0, 0.0, 0.0], margin_unknown=10.0)
    assert penalty > 0.0, (
        "unknown-typed neighbor should collide once inflated by margin_unknown=10; "
        "a 0 penalty means the argmax tie-break is (still) picking margin_vehicle=0"
    )


ALL_LOSS_TESTS = [
    test_known_vehicle_type_unaffected_by_unknown_margin,
    test_unknown_type_uses_margin_unknown_not_vehicle_tiebreak,
]

ALL_TESTS = ALL_AUGMENT_TESTS + ALL_LOSS_TESTS


if __name__ == "__main__":
    print(f"Running {len(ALL_TESTS)} tests for neighbor_type_augment.py / loss.py\n")
    passed, failed, errors = 0, 0, []

    for fn in ALL_TESTS:
        try:
            fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((fn.__name__, e))
            print(f"  [FAIL] {fn.__name__}: {e}")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{len(ALL_TESTS)} passed, {failed} failed")
    if errors:
        print("\nFailed tests:")
        for name, err in errors:
            print(f"  {name}: {err}")
        sys.exit(1)
    else:
        print("All tests passed!")
