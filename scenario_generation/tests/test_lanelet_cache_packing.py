"""Packed line-string / polygon caches produce byte-identical tensors.

``_build_line_or_polygon_tensor`` walked its cache with a Python loop doing a handful of tiny
numpy ops per entry. Packing the cache into one contiguous array turns that into a few
whole-array ops. The change is only worth anything if the output does not move, and the tests
below check that against the previous implementation, reproduced here as the reference.

The padding value is the part that needs pinning. Short entries are padded to ``num_points``
with ``+inf`` so they behave as if the missing points were not there: ``+inf`` fails the
``< x_max`` half of the AABB test and contributes ``+inf`` to a min-distance. Zeros would
instead put phantom points at the map origin, which changes both which entries are selected and
the order they are sorted into -- and would pass any assertion that only looks at shape/dtype.
No map is needed for any of this: the packing and the query are pure array code.
"""

import numpy as np
import pytest

from scenario_generation.gui.lanelet_scene_builder import (
    LaneletSceneBuilder,
    _pack_cache,
)

NUM_POINTS = 20
NUM_TYPES = 4
MAX_N = 8
MASK_RANGE = 100.0


def _reference(cache, center_xy, max_n, num_points, num_types, mask_range):
    """The per-entry Python loop this change replaces (verbatim, minus the packing)."""
    cx, cy = float(center_xy[0]), float(center_xy[1])
    x_min, x_max = cx - mask_range, cx + mask_range
    y_min, y_max = cy - mask_range, cy + mask_range

    scored = []
    for pts, type_idx in cache:
        inside = (
            (pts[:, 0] > x_min) & (pts[:, 0] < x_max) & (pts[:, 1] > y_min) & (pts[:, 1] < y_max)
        ).any()
        if not inside:
            continue
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        scored.append((float(np.sqrt(dx * dx + dy * dy).min()), pts, type_idx))
    scored.sort(key=lambda t: t[0])

    out = np.zeros((max_n, num_points, 2 + num_types), dtype=np.float32)
    for i, (_, pts, type_idx) in enumerate(scored[:max_n]):
        n = min(len(pts), num_points)
        out[i, :n, 0] = pts[:n, 0]
        out[i, :n, 1] = pts[:n, 1]
        out[i, :n, 2 + type_idx] = 1.0
    return out


def _packed(cache, center_xy, **kw):
    # The method reads nothing off ``self``; calling it unbound keeps the test free of a builder
    # (and therefore of lanelet2 and a 46 MB map).
    return LaneletSceneBuilder._build_line_or_polygon_tensor(
        None,
        _pack_cache(cache, kw.get("num_points", NUM_POINTS)),
        center_xy,
        kw.get("max_n", MAX_N),
        kw.get("num_points", NUM_POINTS),
        kw.get("num_types", NUM_TYPES),
        kw.get("mask_range", MASK_RANGE),
    )


def _cache(rng, n_entries: int, *, min_len: int = 2, max_len: int = NUM_POINTS):
    """Entries of DIFFERENT lengths, spread well beyond the mask range.

    Both properties matter: equal lengths would never exercise the padding, and entries that are
    all inside the AABB would never exercise the pre-filter.
    """
    cache = []
    for _ in range(n_entries):
        k = int(rng.integers(min_len, max_len + 1))
        pts = rng.uniform(-300.0, 300.0, size=(k, 2)).astype(np.float32)
        cache.append((pts, int(rng.integers(0, NUM_TYPES))))
    return cache


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_packed_matches_the_per_entry_loop(seed):
    rng = np.random.default_rng(seed)
    cache = _cache(rng, 40)
    for center in (np.array([0.0, 0.0]), np.array([120.0, -80.0]), np.array([-250.0, 250.0])):
        expected = _reference(cache, center, MAX_N, NUM_POINTS, NUM_TYPES, MASK_RANGE)
        assert np.array_equal(_packed(cache, center), expected), f"seed={seed} center={center}"


def test_padding_is_inf_not_zero():
    """The padded tail must be +inf. A zeros regression is caught by the assertions below."""
    pts = np.array([[5.0, 5.0], [6.0, 6.0]], dtype=np.float32)
    packed_pts, lens, types = _pack_cache([(pts, 2)], NUM_POINTS)

    assert packed_pts.shape == (1, NUM_POINTS, 2)
    assert np.array_equal(packed_pts[0, :2], pts)
    assert np.isinf(packed_pts[0, 2:]).all(), "short entries must be padded with +inf"
    assert lens.tolist() == [2] and types.tolist() == [2]


def test_zero_padding_would_change_the_answer():
    """Pins WHY the padding value matters, so a zeros regression cannot pass silently.

    One entry sits far from the query point but its zero-padding would land at the map origin,
    which is right next to it -- with zeros it wins the AABB test and the distance sort it
    should lose.
    """
    far = np.array([[400.0, 400.0], [401.0, 401.0]], dtype=np.float32)
    near = np.array([[3.0, 3.0], [4.0, 4.0]], dtype=np.float32)
    cache = [(far, 0), (near, 1)]
    center = np.array([0.0, 0.0])

    good = _packed(cache, center)
    assert np.array_equal(good, _reference(cache, center, MAX_N, NUM_POINTS, NUM_TYPES, MASK_RANGE))

    zero_pack = list(_pack_cache(cache, NUM_POINTS))
    zero_pack[0] = np.nan_to_num(zero_pack[0], posinf=0.0)
    bad = LaneletSceneBuilder._build_line_or_polygon_tensor(
        None, tuple(zero_pack), center, MAX_N, NUM_POINTS, NUM_TYPES, MASK_RANGE
    )
    assert not np.array_equal(bad, good), "zero padding must be observable, else this test is idle"


def test_empty_cache_returns_zeros():
    out = _packed([], np.array([0.0, 0.0]))
    assert out.shape == (MAX_N, NUM_POINTS, 2 + NUM_TYPES)
    assert not out.any()


def test_equal_distances_keep_cache_order():
    """``kind="stable"`` -- ties resolve to cache order, as ``list.sort`` gave before."""
    a = np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    b = np.array([[0.0, 1.0], [0.0, 2.0]], dtype=np.float32)  # same min distance as `a`
    cache = [(a, 0), (b, 1)]
    center = np.array([0.0, 0.0])

    out = _packed(cache, center)
    assert out[0, 0, 2 + 0] == 1.0, "first cache entry must stay first on a distance tie"
    assert out[1, 0, 2 + 1] == 1.0
    assert np.array_equal(out, _reference(cache, center, MAX_N, NUM_POINTS, NUM_TYPES, MASK_RANGE))
