from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlvr.autoresearch.tools.select_global_train_checkpoint import (
    select_global_checkpoint,
)


def _candidate(tmp_path: Path, label: str, epoch: int, rewards: list[float]):
    root = tmp_path / label
    root.mkdir()
    checkpoint = root / f"epoch_{epoch:03d}.pth"
    checkpoint.write_bytes(f"checkpoint-{label}".encode())
    rows = [
        {"scene_path": f"/scene/{index}.npz", "det_reward": reward}
        for index, reward in enumerate(rewards)
    ]
    scenes = root / "scenes.json"
    scenes.write_text(json.dumps(rows))
    summary = root / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "scene_count": len(rows),
                "mean_det_reward": sum(rewards) / len(rewards),
            }
        )
    )
    return {
        "label": label,
        "epoch": epoch,
        "checkpoint": checkpoint,
        "summary": summary,
        "scenes": scenes,
    }


def test_selects_strictly_best_fixed_train_mean(tmp_path: Path) -> None:
    incumbent = _candidate(tmp_path, "incumbent", 11, [0.5, 0.7])
    improved = _candidate(tmp_path, "epoch12", 12, [0.6, 0.8])
    output = tmp_path / "best.pth"
    manifest = tmp_path / "selection.json"

    result = select_global_checkpoint(
        [incumbent, improved],
        expected_scene_count=2,
        output_checkpoint=output,
        output_manifest=manifest,
    )

    assert result["winner"]["label"] == "epoch12"
    assert result["winner_reward_delta_from_baseline"] == pytest.approx(0.1)
    assert output.read_bytes() == b"checkpoint-epoch12"
    assert json.loads(manifest.read_text())["selection_contract"] == (
        "fixed_train_selector_K1_mean_det_reward"
    )


def test_earliest_epoch_wins_exact_tie(tmp_path: Path) -> None:
    first = _candidate(tmp_path, "first", 11, [0.4, 0.8])
    second = _candidate(tmp_path, "second", 12, [0.4, 0.8])
    result = select_global_checkpoint(
        [first, second],
        expected_scene_count=2,
        output_checkpoint=tmp_path / "best.pth",
        output_manifest=tmp_path / "selection.json",
    )
    assert result["winner"]["label"] == "first"


def test_rejects_different_ordered_scene_contract(tmp_path: Path) -> None:
    first = _candidate(tmp_path, "first", 11, [0.4, 0.8])
    second = _candidate(tmp_path, "second", 12, [0.5, 0.9])
    rows = json.loads(second["scenes"].read_text())
    second["scenes"].write_text(json.dumps(list(reversed(rows))))

    with pytest.raises(ValueError, match="ordered selector scene contract differs"):
        select_global_checkpoint(
            [first, second],
            expected_scene_count=2,
            output_checkpoint=tmp_path / "best.pth",
            output_manifest=tmp_path / "selection.json",
        )
