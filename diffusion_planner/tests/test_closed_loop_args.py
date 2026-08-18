"""Verify closed_loop_* fields in TrainConfig are properly defined."""

from __future__ import annotations

from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_closed_loop_npz_root_is_list():
    """closed_loop_npz_root must be list (nargs='+') in CLI, not str."""
    from diffusion_planner.config import build_parser, GRPOConfig

    gp = build_parser(GRPOConfig, "test")
    for action in gp._actions:
        if action.dest == "closed_loop_npz_root":
            assert action.nargs == "+", (
                f"closed_loop_npz_root nargs={action.nargs!r}, should be '+' (list) — "
                "this is kosuke55 bug #2"
            )
            return
    pytest.fail("closed_loop_npz_root not found in GRPOConfig parser")


def test_closed_loop_config_fields_cli_marked():
    """Verify closed_loop_* fields are cli-marked in TrainConfig."""
    from diffusion_planner.config import TrainConfig

    cli_field_names = {
        f.name for f in TrainConfig.__dataclass_fields__.values() if f.metadata.get("cli")
    }

    expected_cli_fields = [
        "closed_loop_npz_root",
    ]

    for field_name in expected_cli_fields:
        assert field_name in cli_field_names, (
            f"{field_name} should be marked with cli() in TrainConfig"
        )


def test_train_predictor_uses_config_build_parser():
    """Verify train_predictor.py uses config.build_parser."""
    train_predictor = _REPO_ROOT / "diffusion_planner" / "train_predictor.py"
    source = train_predictor.read_text(encoding="utf-8")
    assert "from diffusion_planner.config import" in source
    assert "build_parser" in source
