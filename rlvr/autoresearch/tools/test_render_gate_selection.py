"""Tests for per-scene final-visual role provenance."""

from __future__ import annotations

import json

import pytest

from rlvr.autoresearch.tools.render_gate_comparison import (
    _load_selection_row,
    _promotion_label,
)


def test_selection_role_is_attached_by_exact_resolved_scene(tmp_path) -> None:
    scene = tmp_path / "scene.npz"
    scene.touch()
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "scenes": [
                    {
                        "kind": "collision_recovered",
                        "scene_path": str(scene),
                        "delta_det_reward": 0.8,
                    }
                ]
            }
        )
    )
    row = _load_selection_row(selection, scene)
    assert row is not None
    assert row["kind"] == "collision_recovered"
    assert row["delta_det_reward"] == pytest.approx(0.8)


def test_missing_or_duplicate_selection_fails_closed(tmp_path) -> None:
    scene = tmp_path / "scene.npz"
    scene.touch()
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"scenes": []}))
    with pytest.raises(RuntimeError, match="exactly one"):
        _load_selection_row(selection, scene)
    row = {"kind": "audit", "scene_path": str(scene)}
    selection.write_text(json.dumps({"scenes": [row, row]}))
    with pytest.raises(RuntimeError, match="got 2"):
        _load_selection_row(selection, scene)


def test_promotion_watermark_is_unambiguous() -> None:
    assert _promotion_label("strict_promotion_passed") == "PAPER-ALIGNED MEAN SCORE PASSED"
    assert _promotion_label("aggregate_promotion_passed") == "PAPER-ALIGNED MEAN SCORE PASSED"
    assert _promotion_label("not_promoted_failure_audit") == "NOT PROMOTED · MEAN SCORE AUDIT"
    assert _promotion_label("not_promoted_no_mean_gain") == "NOT PROMOTED · MEAN GAIN UNPROVEN"
