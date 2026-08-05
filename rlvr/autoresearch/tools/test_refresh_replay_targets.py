"""Unit tests for the monotone replay-memory refresh (max(frozen, fresh))."""

from __future__ import annotations

import json

import pytest

from rlvr.autoresearch.tools.refresh_replay_targets import build_rows, join


def _write(p, obj):
    p.write_text(json.dumps(obj))
    return p


def _rows(p, rows):
    with open(p, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return p


def test_build_rows_repairs_the_source_not_the_frozen_target(tmp_path):
    replay = _write(tmp_path / "replay.json", [str(tmp_path / "t1.npz")])
    prev = _rows(
        tmp_path / "prev.jsonl",
        [
            {
                "scene_path": str(tmp_path / "t1.npz"),
                "source_scene_path": str(tmp_path / "src1.npz"),
                "selected_total": -1.0,
                "repair_labels": ["expert_disagreement"],
            }
        ],
    )
    out = tmp_path / "rows.jsonl"
    stats = build_rows(replay, [prev], out)
    assert stats == {"replay": 1, "rows_written": 1, "missing": 0}
    row = json.loads(out.read_text().strip())
    # The fresh pass must re-repair the ORIGINAL window: the frozen target NPZ has
    # ego_agent_future overwritten, which would re-reference every deviation term.
    assert row["scene_path"] == str(tmp_path / "src1.npz")
    assert row["refresh_frozen_target"] == str(tmp_path / "t1.npz")
    assert row["refresh_frozen_total"] == -1.0
    assert row["repair_labels"] == ["expert_disagreement"]


def test_build_rows_fails_loudly_when_a_replay_scene_has_no_row(tmp_path):
    replay = _write(tmp_path / "replay.json", [str(tmp_path / "unknown.npz")])
    prev = _rows(tmp_path / "prev.jsonl", [])
    with pytest.raises(ValueError, match="no repaired row"):
        build_rows(replay, [prev], tmp_path / "rows.jsonl")
    stats = build_rows(replay, [prev], tmp_path / "rows.jsonl", allow_missing=True)
    assert stats["missing"] == 1 and stats["rows_written"] == 0


def _join_case(tmp_path, frozen_total, fresh_total, **kw):
    frozen, src, fresh_npz = (
        str(tmp_path / "frozen.npz"),
        str(tmp_path / "src.npz"),
        str(tmp_path / "fresh.npz"),
    )
    replay = _write(tmp_path / "replay.json", [frozen])
    prev = _rows(
        tmp_path / "prev.jsonl",
        [{"scene_path": frozen, "source_scene_path": src, "selected_total": frozen_total}],
    )
    fresh_rows = (
        []
        if fresh_total is None
        else [{"scene_path": fresh_npz, "source_scene_path": src, "selected_total": fresh_total}]
    )
    fr = _rows(tmp_path / "fresh.jsonl", fresh_rows)
    stats = join(replay, [prev], [fr], tmp_path / "out.json", tmp_path / "stats.json", **kw)
    return json.loads((tmp_path / "out.json").read_text()), stats, frozen, fresh_npz


def test_join_takes_the_fresh_target_when_it_scores_better(tmp_path):
    out, stats, _frozen, fresh = _join_case(tmp_path, -2.0, -1.0)
    assert out == [fresh]
    assert stats["improved_by_fresh"] == 1 and stats["kept_frozen"] == 0
    assert stats["mean_gain_on_improved"] == pytest.approx(1.0)


def test_join_keeps_the_frozen_target_when_the_policy_regressed(tmp_path):
    """The retention half: a drifted policy must not overwrite an old fix with a worse one."""
    out, stats, frozen, _fresh = _join_case(tmp_path, -1.0, -2.5)
    assert out == [frozen]
    assert stats["kept_frozen"] == 1 and stats["improved_by_fresh"] == 0


def test_join_keeps_frozen_when_the_policy_has_no_gate_passing_candidate(tmp_path):
    out, stats, frozen, _ = _join_case(tmp_path, -1.0, None)
    assert out == [frozen]
    assert stats["no_fresh_candidate"] == 1


def test_join_min_gain_adds_hysteresis_so_near_ties_stay_frozen(tmp_path):
    out, stats, frozen, _ = _join_case(tmp_path, -1.00, -0.99, min_gain=0.5)
    assert out == [frozen], "a 0.01 gain must not churn the memory when min_gain=0.5"
    assert stats["kept_frozen"] == 1


def test_join_is_monotone_target_score_never_decreases(tmp_path):
    """Whatever the fresh pass produces, the chosen target's score >= the frozen one's."""
    for frozen_total, fresh_total in ((-1.0, -3.0), (-1.0, 0.5), (-2.0, -2.0), (-2.0, None)):
        d = tmp_path / f"case_{frozen_total}_{fresh_total}"
        d.mkdir()
        out, _stats, frozen, fresh = _join_case(d, frozen_total, fresh_total)
        chosen = out[0]
        chosen_total = frozen_total if chosen == frozen else fresh_total
        assert chosen_total >= frozen_total


def test_rows_can_come_from_a_replay_memory_json(tmp_path):
    """A chain link only gets the previous round's MEMORY file, not its jsonl."""
    frozen, src = str(tmp_path / "t.npz"), str(tmp_path / "s.npz")
    replay = _write(tmp_path / "replay.json", [frozen])
    mem = _write(
        tmp_path / "memory.json",
        {
            "capacity": 10,
            "entries": [{"scene_path": frozen, "source_scene_path": src, "selected_total": -1.5}],
        },
    )
    out = tmp_path / "rows.jsonl"
    stats = build_rows(replay, [mem], out)
    assert stats["rows_written"] == 1
    row = json.loads(out.read_text().strip())
    assert row["scene_path"] == src and row["refresh_frozen_total"] == -1.5
