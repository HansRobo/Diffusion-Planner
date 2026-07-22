# human_match_prototype/tests/test_multi_human_match.py
import math

import numpy as np
import pytest

from human_match_prototype.multi_human_match import (
    MatchResult,
    build_lanelet_lookup,
    find_matches,
)


def _make_index_entries(n, lanelet_id, base_heading=0.0, base_x=100.0, base_y=200.0):
    """Create n index entries on the same lanelet with slight position variation."""
    return [
        {
            "npz_path": f"/data/bag_{i // 10}_frame_{i}.npz",
            "x": base_x + i * 0.5,
            "y": base_y,
            "heading_deg": base_heading + i * 0.1,
            "timestamp": 1000 + i,
            "lanelet_id": lanelet_id,
        }
        for i in range(n)
    ]


def test_build_lanelet_lookup():
    entries = _make_index_entries(3, lanelet_id=42) + _make_index_entries(2, lanelet_id=99)
    lookup = build_lanelet_lookup(entries)
    assert len(lookup[42]) == 3
    assert len(lookup[99]) == 2
    assert 0 not in lookup


def test_find_matches_filters_heading():
    entries = [
        {
            "npz_path": "/a.npz",
            "x": 100.0,
            "y": 200.0,
            "heading_deg": 0.0,
            "timestamp": 1,
            "lanelet_id": 42,
        },
        {
            "npz_path": "/b.npz",
            "x": 100.0,
            "y": 200.0,
            "heading_deg": 180.0,
            "timestamp": 2,
            "lanelet_id": 42,
        },
    ]
    lookup = build_lanelet_lookup(entries)
    test_sidecar = {
        "npz_path": "/test.npz",
        "x": 100.0,
        "y": 200.0,
        "heading_deg": 5.0,
        "timestamp": 0,
    }
    matches = find_matches(
        test_sidecar,
        lookup,
        test_lanelet_id=42,
        max_heading_diff_deg=45.0,
        max_distance_m=30.0,
        max_per_lanelet=200,
        seed_key="/test.npz",
    )
    assert len(matches) == 1
    assert matches[0]["npz_path"] == "/a.npz"


def test_find_matches_filters_distance():
    entries = [
        {
            "npz_path": "/near.npz",
            "x": 105.0,
            "y": 200.0,
            "heading_deg": 0.0,
            "timestamp": 1,
            "lanelet_id": 42,
        },
        {
            "npz_path": "/far.npz",
            "x": 200.0,
            "y": 200.0,
            "heading_deg": 0.0,
            "timestamp": 2,
            "lanelet_id": 42,
        },
    ]
    lookup = build_lanelet_lookup(entries)
    test_sidecar = {
        "npz_path": "/test.npz",
        "x": 100.0,
        "y": 200.0,
        "heading_deg": 0.0,
        "timestamp": 0,
    }
    matches = find_matches(
        test_sidecar,
        lookup,
        test_lanelet_id=42,
        max_heading_diff_deg=45.0,
        max_distance_m=30.0,
        max_per_lanelet=200,
        seed_key="/test.npz",
    )
    assert len(matches) == 1
    assert matches[0]["npz_path"] == "/near.npz"


def test_find_matches_caps_and_deduplicates():
    entries = _make_index_entries(50, lanelet_id=42, base_heading=0.0)
    lookup = build_lanelet_lookup(entries)
    test_sidecar = {
        "npz_path": "/test.npz",
        "x": 100.0,
        "y": 200.0,
        "heading_deg": 0.0,
        "timestamp": 0,
    }
    matches = find_matches(
        test_sidecar,
        lookup,
        test_lanelet_id=42,
        max_heading_diff_deg=45.0,
        max_distance_m=30.0,
        max_per_lanelet=10,
        seed_key="/test.npz",
    )
    assert len(matches) <= 10

    matches2 = find_matches(
        test_sidecar,
        lookup,
        test_lanelet_id=42,
        max_heading_diff_deg=45.0,
        max_distance_m=30.0,
        max_per_lanelet=10,
        seed_key="/test.npz",
    )
    assert [m["npz_path"] for m in matches] == [m["npz_path"] for m in matches2]


def test_find_matches_no_lanelet():
    lookup = build_lanelet_lookup([])
    test_sidecar = {
        "npz_path": "/test.npz",
        "x": 100.0,
        "y": 200.0,
        "heading_deg": 0.0,
        "timestamp": 0,
    }
    matches = find_matches(
        test_sidecar,
        lookup,
        test_lanelet_id=999,
        max_heading_diff_deg=45.0,
        max_distance_m=30.0,
        max_per_lanelet=200,
        seed_key="/test.npz",
    )
    assert matches == []
