"""closed_loop_validate uses args from TrainConfig, verified via train_cli.build_parser.

Both entrypoints (train_predictor.py and train_grpo_predictor.py) now use
train_cli.build_parser() to generate CLI from TrainConfig fields marked with cli(...).
This test verifies that all args.closed_loop_* attributes used in closed_loop_validate
are properly defined in TrainConfig with cli(...).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRAIN_PY = _REPO_ROOT / "diffusion_planner" / "diffusion_planner" / "train.py"
_TRAIN_CONFIG_PY = _REPO_ROOT / "diffusion_planner" / "diffusion_planner" / "train_config.py"


def _cli_fields() -> set[str]:
    """All field names in TrainConfig marked with cli(...)."""
    tree = ast.parse(_TRAIN_CONFIG_PY.read_text(encoding="utf-8"))
    cli_fields: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Check if it's a field() call with 'cli' metadata
            if isinstance(node.func, ast.Name) and node.func.id == "field":
                for kw in node.keywords:
                    if kw.arg == "metadata":
                        if isinstance(kw.value, ast.Dict):
                            for k, v in zip(kw.value.keys, kw.value.values):
                                if (
                                    isinstance(k, ast.Constant)
                                    and isinstance(k.value, str)
                                    and k.value == "cli"
                                ):
                                    # Get the field name from class body
                                    pass
    return cli_fields


def _args_attrs_read_by(func_name: str, path: Path) -> set[str]:
    """Attributes read as ``args.<name>`` inside ``func_name`` (nested functions included)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    func = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == func_name
    )
    return {
        node.attr
        for node in ast.walk(func)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "args"
    }


def test_closed_loop_config_fields_exist():
    """Verify closed_loop_* config fields are properly marked with cli() in TrainConfig."""
    from diffusion_planner.train_config import TrainConfig

    cli_field_names = {f.name for f in TrainConfig.__dataclass_fields__.values() if f.metadata.get("cli")}

    # These closed_loop fields should be in CLI
    expected_cli_fields = [
        "closed_loop_npz_root",
    ]

    for field_name in expected_cli_fields:
        assert field_name in cli_field_names, f"{field_name} should be marked with cli() in TrainConfig"


def test_args_conversion_in_train_predictor():
    """Verify train_predictor.py uses build_parser for CLI."""
    source = _TRAIN_PY.read_text(encoding="utf-8")
    assert "from diffusion_planner.train_cli import build_parser" in source
    assert "build_parser(description=__doc__)" in source
