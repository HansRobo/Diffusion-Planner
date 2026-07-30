"""The road-border distance must stay EXACT while the segment set is narrowed for speed.

``evaluate_trajectory`` narrows the segment set per block of ``_TICK_BLOCK`` ticks, which made it
5.3x faster on real 1700-tick cases (`finalize` was 20% of a full suite run -- plan/11 9v/9x). The
speed is worthless if it moves the numbers: ``rb_dists`` feeds mean/p5/tdigest clearance
statistics, not a threshold test, so an approximation would silently shift the metrics rather than
flip a flag someone would notice.

Every test here compares against a brute-force scan of ALL segments at every tick -- the thing the
narrowing is supposed to be equivalent to -- and demands bit-equality, not closeness.
"""

from __future__ import annotations

import numpy as np
import pytest

from scenario_generation.tools.eval_cl_trajectory import (
    _compute_ego_perimeter,
    _flatten_segments,
    _min_dist_vectorized,
    _TICK_BLOCK,
    evaluate_trajectory,
)

EGO_LENGTH, EGO_WIDTH, EGO_WHEELBASE = 4.97, 1.93, 3.1


def _brute_force_rb_dists(traj, border_segments) -> np.ndarray:
    """Every tick against every segment -- the definition the narrowing must reproduce."""
    seg_starts, seg_ends = _flatten_segments(border_segments)
    if seg_starts.shape[0] == 0:
        return np.array([])
    half_l, half_w = EGO_LENGTH / 2, EGO_WIDTH / 2
    return np.array([
        _min_dist_vectorized(
            _compute_ego_perimeter(e["x"], e["y"], e["heading"], half_l, half_w, EGO_WHEELBASE),
            seg_starts,
            seg_ends,
        )
        for e in traj
    ])


def _traj(xs, ys=None, headings=None) -> list[dict]:
    ys = ys if ys is not None else np.zeros(len(xs))
    headings = headings if headings is not None else np.zeros(len(xs))
    return [
        {"step": i, "x": float(x), "y": float(y), "heading": float(h),
         "speed": 5.0, "goal_d": float(len(xs) - i)}
        for i, (x, y, h) in enumerate(zip(xs, ys, headings))
    ]


def _lane_borders(y_offsets, x0=-50.0, x1=400.0, n=40) -> list[np.ndarray]:
    xs = np.linspace(x0, x1, n)
    return [np.column_stack([xs, np.full_like(xs, y)]) for y in y_offsets]


def _assert_exact(traj, borders):
    got = np.asarray(evaluate_trajectory(
        traj, borders, EGO_LENGTH, EGO_WIDTH, EGO_WHEELBASE)["rb_dists"])
    want = _brute_force_rb_dists(traj, borders)
    assert got.shape == want.shape, (got.shape, want.shape)
    assert np.array_equal(got, want), (
        f"max|diff| = {np.abs(got - want).max():.3e} -- the narrowing is not exact"
    )
    return got


def test_straight_run_between_borders_is_exact():
    """The ordinary case: the ego drives along a corridor, many blocks, borders always near."""
    traj = _traj(np.linspace(0.0, 300.0, 7 * _TICK_BLOCK))
    _assert_exact(traj, _lane_borders([-3.5, 3.5]))


def test_curved_run_with_heading_changes_is_exact():
    """Heading enters through the OBB perimeter, so vary it: a block's bbox is built from poses,
    but the distance is measured from the perimeter, which `reach` has to cover."""
    n = 4 * _TICK_BLOCK + 7  # deliberately not a multiple of the block size
    t = np.linspace(0.0, 6.0, n)
    xs, ys = 25.0 * t, 8.0 * np.sin(t)
    headings = np.arctan2(np.gradient(ys), np.gradient(xs))
    _assert_exact(_traj(xs, ys, headings), _lane_borders([-6.0, 6.0]))


@pytest.mark.parametrize("n_ticks", [1, 2, _TICK_BLOCK - 1, _TICK_BLOCK, _TICK_BLOCK + 1])
def test_block_boundaries_are_exact(n_ticks):
    """Off-by-one in the block slicing would drop or duplicate ticks."""
    traj = _traj(np.linspace(0.0, 20.0, n_ticks))
    got = _assert_exact(traj, _lane_borders([-3.5, 3.5]))
    assert len(got) == n_ticks


def test_far_from_every_border_falls_back_and_stays_exact():
    """The narrowing's bound only holds while reported distances stay inside the margin. Drive a
    long way from the only border so every block breaches it and the full-scan fallback runs."""
    traj = _traj(np.linspace(0.0, 300.0, 4 * _TICK_BLOCK), ys=np.full(4 * _TICK_BLOCK, 900.0))
    got = _assert_exact(traj, _lane_borders([-3.5, 3.5]))
    assert got.min() > 50.0, "this test is only meaningful if the margin is actually breached"


def test_partially_far_trajectory_is_exact():
    """Some blocks inside the margin, some outside -- the per-block fallback must be independent."""
    n = 6 * _TICK_BLOCK
    ys = np.where(np.arange(n) < n // 2, 0.0, 800.0)
    _assert_exact(_traj(np.linspace(0.0, 300.0, n), ys=ys), _lane_borders([-3.5, 3.5]))


def test_block_between_margin_and_full_distance_is_exact():
    """The case the margin bound exists for: a border just outside a block's own window that is
    nonetheless the nearest one. Two parallel borders, one 30 m out (inside the margin, kept) and
    the geometry arranged so a block's bbox alone would not have reached it."""
    n = 4 * _TICK_BLOCK
    traj = _traj(np.linspace(0.0, 240.0, n), ys=np.zeros(n))
    borders = [
        np.array([[-100.0, 30.0], [500.0, 30.0]]),      # 30 m away: inside the 50 m margin
        np.array([[-100.0, 120.0], [500.0, 120.0]]),    # 120 m: outside, must not matter
    ]
    got = _assert_exact(traj, borders)
    assert 25.0 < got.min() < 35.0, "the nearest border should be the 30 m one"


def test_degenerate_segments_are_exact():
    """Zero-length polyline points exercise the branch that is now taken only when one exists."""
    borders = _lane_borders([-3.5, 3.5])
    borders.append(np.array([[10.0, 2.0], [10.0, 2.0]]))          # zero length
    borders.append(np.array([[120.0, -2.0], [120.0, -2.0], [120.0, -2.0]]))
    _assert_exact(_traj(np.linspace(0.0, 200.0, 3 * _TICK_BLOCK)), borders)


def test_no_borders_reports_no_data():
    """`rb_has_data` must keep meaning "this map ships no road_border geometry"."""
    out = evaluate_trajectory(
        _traj(np.linspace(0.0, 50.0, 2 * _TICK_BLOCK)), [], EGO_LENGTH, EGO_WIDTH, EGO_WHEELBASE)
    assert out["rb_has_data"] is False
    assert len(np.asarray(out["rb_dists"])) == 0


def test_single_point_polylines_are_ignored():
    """`_flatten_segments` drops polylines with < 2 points; the narrowing must not resurrect them."""
    borders = _lane_borders([-3.5, 3.5]) + [np.array([[5.0, 1.0]])]
    _assert_exact(_traj(np.linspace(0.0, 100.0, 2 * _TICK_BLOCK)), borders)


def test_matches_brute_force_through_the_rollouts_own_prune():
    """The production caller pre-prunes: `_finalize_row` narrows to within 50 m of the whole
    trajectory before calling this. Both narrowings then compose, and their margins interact --
    so check the composition against the unpruned brute force, which is what the metric means.
    """
    from scenario_generation.scenario_sim_rollout import _prune_border_segments

    n = 6 * _TICK_BLOCK
    t = np.linspace(0.0, 8.0, n)
    xs, ys = 30.0 * t, 10.0 * np.sin(t)
    headings = np.arctan2(np.gradient(ys), np.gradient(xs))
    traj = _traj(xs, ys, headings)
    # A map with borders both alongside the path and far from it, so pruning actually drops some.
    borders = _lane_borders([-7.0, 7.0], x0=-100.0, x1=500.0, n=120)
    borders += _lane_borders([600.0, -600.0], x0=-100.0, x1=500.0, n=120)

    near = _prune_border_segments(borders, traj, EGO_LENGTH, EGO_WHEELBASE, EGO_WIDTH)
    assert len(near) < len(borders), "this test needs the outer prune to actually drop something"
    got = np.asarray(evaluate_trajectory(
        traj, near, EGO_LENGTH, EGO_WIDTH, EGO_WHEELBASE)["rb_dists"])
    want = _brute_force_rb_dists(traj, borders)   # ALL borders, no pruning at all
    assert np.array_equal(got, want), f"max|diff| = {np.abs(got - want).max():.3e}"


def test_aggregate_keys_match_brute_force():
    """Not just rb_dists: crossings, progress and the speed profile come off the same pass."""
    traj = _traj(np.linspace(0.0, 300.0, 5 * _TICK_BLOCK))
    borders = _lane_borders([-2.2, 2.2])   # close enough that crossings actually fire
    out = evaluate_trajectory(traj, borders, EGO_LENGTH, EGO_WIDTH, EGO_WHEELBASE)
    want = _brute_force_rb_dists(traj, borders)
    assert out["rb_cross_steps"] == int((want < 0.20).sum())
    assert out["rb_dist_min"] == float(want.min())
    assert out["rb_dist_p5"] == float(np.percentile(want, 5))
    assert np.array_equal(np.asarray(out["rb_dists"]), want)
