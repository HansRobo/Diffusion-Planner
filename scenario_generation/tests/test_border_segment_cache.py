"""Tests for the per-map road_border cache in eval_cl_trajectory.

``load_border_segments`` re-parsed a 46 MB lanelet2 map once per SCENARIO, from the rollout's
finalize step, while 378 of the suite's 464 cases share one map -- 20% of a full run's cost
(plan/11 9v/9x). It is cached per map now. Two invariants make that cache safe, and both are the
kind that a later edit can break without any test failing anywhere else:

  1. a second call must not re-parse (otherwise the cache does nothing)
  2. each call must hand back a FRESH list (the arrays are shared, the list is not) -- a caller
     that trimmed a shared list in place would silently move ``rb_dists``, which feeds mean/p5
     clearance statistics rather than a threshold test

lanelet2 is faked here rather than imported: the real thing needs a ROS overlay and a 46 MB
fixture, and neither is needed to test the cache's contract.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

import scenario_generation.tools.eval_cl_trajectory as E


class _FakeAttrs(dict):
    """lanelet2 attributes support ``"type" in attrs`` and ``attrs["type"]``."""


class _FakePoint:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class _FakeLineString(list):
    def __init__(self, pts, attributes):
        super().__init__(pts)
        self.attributes = attributes


@pytest.fixture
def fake_lanelet2(monkeypatch):
    """Install a fake ``lanelet2`` + projection module and count the loads it serves."""
    calls = {"loads": 0}

    def _load(path, projector):  # noqa: ARG001
        calls["loads"] += 1
        border = _FakeLineString(
            [_FakePoint(0.0, 0.0), _FakePoint(1.0, 0.0), _FakePoint(2.0, 0.5)],
            _FakeAttrs(type="road_border"),
        )
        by_subtype = _FakeLineString(
            [_FakePoint(0.0, 3.0), _FakePoint(1.0, 3.0)],
            _FakeAttrs(type="line_thin", subtype="road_border"),
        )
        # Filtered out: wrong type, and a road_border with a single point (len < 2).
        other = _FakeLineString([_FakePoint(0.0, 9.0), _FakePoint(1.0, 9.0)],
                                _FakeAttrs(type="curbstone"))
        degenerate = _FakeLineString([_FakePoint(5.0, 5.0)], _FakeAttrs(type="road_border"))
        return types.SimpleNamespace(lineStringLayer=[border, by_subtype, other, degenerate])

    lanelet2 = types.ModuleType("lanelet2")
    lanelet2.io = types.SimpleNamespace(load=_load, Origin=lambda a, b: (a, b))
    proj = types.ModuleType("autoware_lanelet2_extension_python.projection")
    proj.MGRSProjector = lambda origin: origin
    pkg = types.ModuleType("autoware_lanelet2_extension_python")

    monkeypatch.setitem(sys.modules, "lanelet2", lanelet2)
    monkeypatch.setitem(sys.modules, "autoware_lanelet2_extension_python", pkg)
    monkeypatch.setitem(sys.modules, "autoware_lanelet2_extension_python.projection", proj)
    # A real run's cache is per process; a test's must not leak into the next test.
    monkeypatch.setattr(E, "_BORDER_SEGMENT_CACHE", {})
    return calls


def test_parses_once_per_map(fake_lanelet2, tmp_path):
    m = tmp_path / "lanelet2_map.osm"
    m.write_text("")
    for _ in range(3):
        E.load_border_segments(str(m))
    assert fake_lanelet2["loads"] == 1


def test_distinct_maps_are_parsed_separately(fake_lanelet2, tmp_path):
    a = tmp_path / "a" / "lanelet2_map.osm"
    b = tmp_path / "b" / "lanelet2_map.osm"
    for p in (a, b):
        p.parent.mkdir()
        p.write_text("")
    E.load_border_segments(str(a))
    E.load_border_segments(str(b))
    E.load_border_segments(str(a))
    assert fake_lanelet2["loads"] == 2


def test_same_path_spelled_differently_hits_the_cache(fake_lanelet2, tmp_path):
    """The rollout gets its map path from the scenario's RoadNetwork/LogicFile, so the same map
    can arrive spelled differently between cases; the key is the resolved path."""
    d = tmp_path / "m"
    d.mkdir()
    m = d / "lanelet2_map.osm"
    m.write_text("")
    E.load_border_segments(str(m))
    E.load_border_segments(str(d / ".." / "m" / "lanelet2_map.osm"))
    assert fake_lanelet2["loads"] == 1


def test_returns_equal_but_independent_lists(fake_lanelet2, tmp_path):
    m = tmp_path / "lanelet2_map.osm"
    m.write_text("")
    first = E.load_border_segments(str(m))
    second = E.load_border_segments(str(m))

    # Only the two road_border polylines with >= 2 points survive the filter.
    assert len(first) == 2
    assert all(np.array_equal(x, y) for x, y in zip(first, second))
    assert first is not second, "a shared list lets one caller's in-place trim move rb_dists"

    # Trimming what we were handed must not change what the next caller sees.
    first.clear()
    assert len(E.load_border_segments(str(m))) == 2
    assert fake_lanelet2["loads"] == 1


def test_dtype_and_shape_are_unchanged_by_caching(fake_lanelet2, tmp_path):
    m = tmp_path / "lanelet2_map.osm"
    m.write_text("")
    cold = E.load_border_segments(str(m))
    warm = E.load_border_segments(str(m))
    for c, w in zip(cold, warm):
        # evaluate_trajectory's _flatten_segments assumes (N, 2) float64.
        assert c.dtype == np.float64 and w.dtype == np.float64
        assert c.shape == w.shape and c.shape[1] == 2
