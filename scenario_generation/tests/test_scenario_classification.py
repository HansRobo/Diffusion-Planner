"""Tests for portable scenario classification JSON resolution."""

from __future__ import annotations

from pathlib import Path

from scenario_generation.scenario_classification import (
    area_at_idx,
    episodes_from_entry,
    idx_in_labeled_ranges,
    iter_area_spans,
    iter_segments,
    load_area_metric_groups,
    load_classification_json,
    resolve_classification_json,
    time_series_from_doc,
    validate_classification_npz_root,
)


def test_iter_area_spans_legacy_dense() -> None:
    area_by_idx = ["a", "a", None, "b", "b", "a", "a"]
    spans = list(iter_area_spans(area_by_idx))
    assert spans == [("a", 0, 2), ("b", 3, 5), ("a", 5, 7)]


def test_episodes_bridge_unlabeled_gap() -> None:
    entry = {
        "segments": [
            {
                "area": "a",
                "metric_group": "g1",
                "video_start_idx": 0,
                "video_end_idx": 10,
                "labeled_ranges": [[0, 2], [5, 10]],
            }
        ]
    }
    eps = episodes_from_entry(entry)
    assert len(eps) == 1
    assert eps[0]["video_start_idx"] == 0
    assert eps[0]["video_end_idx"] == 10
    assert eps[0]["labeled_ranges"] == [[0, 2], [5, 10]]
    assert area_at_idx(eps, 1) == "a"
    assert area_at_idx(eps, 3) is None
    assert idx_in_labeled_ranges(3, eps[0]["labeled_ranges"]) is False


def test_episodes_from_v21_segments() -> None:
    entry = {
        "segments": [
            {"area": "a", "metric_group": "g1", "start_idx": 0, "end_idx": 2},
            {"area": "a", "metric_group": "g1", "start_idx": 5, "end_idx": 7},
            {"area": "b", "metric_group": "g2", "start_idx": 10, "end_idx": 12},
        ]
    }
    eps = episodes_from_entry(entry)
    assert len(eps) == 2
    assert eps[0]["area"] == "a"
    assert eps[0]["video_start_idx"] == 0
    assert eps[0]["video_end_idx"] == 7
    assert eps[0]["labeled_ranges"] == [[0, 2], [5, 7]]
    assert list(iter_segments(entry)) == [("a", 0, 2), ("a", 5, 7), ("b", 10, 12)]


def test_episodes_from_legacy_area_by_idx() -> None:
    entry = {
        "area_by_idx": ["a", "a", None, "b"],
        "metric_group_by_idx": ["g1", "g1", None, "g2"],
    }
    eps = episodes_from_entry(entry)
    assert len(eps) == 2
    assert eps[0]["labeled_ranges"] == [[0, 2]]


def test_load_area_metric_groups_from_catalog() -> None:
    doc = {
        "area_catalog": {"zeikan_curve": {"metric_group": "curve"}},
        "time_series": {},
    }
    assert load_area_metric_groups(doc) == {"zeikan_curve": "curve"}


def test_time_series_from_legacy_sequences() -> None:
    doc = {
        "sequences": {
            "13-42-45": {
                "route_key": "r",
                "segments": [
                    {
                        "area": "a",
                        "metric_group": "g",
                        "video_start_idx": 0,
                        "video_end_idx": 1,
                        "labeled_ranges": [[0, 1]],
                    }
                ],
            }
        }
    }
    ts = time_series_from_doc(doc)
    assert "13-42-45" in ts
    assert ts["13-42-45"]["name"] == "13-42-45"


def test_validate_classification_npz_root_date_mismatch(tmp_path: Path) -> None:
    doc = {"date": "2026-01-15", "time_series": {}}
    warnings = validate_classification_npz_root(doc, tmp_path / "2026-01-16")
    assert len(warnings) == 1


def test_resolve_classification_json_explicit(tmp_path: Path) -> None:
    json_path = tmp_path / "2026-01-15.json"
    json_path.write_text("{}", encoding="utf-8")
    resolved = resolve_classification_json(
        tmp_path / "2026-01-15",
        json_path,
    )
    assert resolved == json_path.resolve()


def test_resolve_classification_json_auto(tmp_path: Path) -> None:
    root = tmp_path / "scenario_classification_json"
    dataset = "x2_dev/foo"
    (root / dataset).mkdir(parents=True)
    json_path = root / dataset / "2026-01-15.json"
    json_path.write_text("{}", encoding="utf-8")
    resolved = resolve_classification_json(
        tmp_path / "2026-01-15",
        None,
        dataset_name=dataset,
        search_roots=[root],
    )
    assert resolved == json_path.resolve()


def test_resolve_classification_json_missing_returns_none(tmp_path: Path) -> None:
    assert resolve_classification_json(tmp_path, None) is None


def test_load_classification_json(tmp_path: Path) -> None:
    path = tmp_path / "2026-01-15.json"
    path.write_text(
        '{"date": "2026-01-15", "time_series": {}, "area_catalog": {}}',
        encoding="utf-8",
    )
    doc = load_classification_json(path)
    assert doc["date"] == "2026-01-15"
