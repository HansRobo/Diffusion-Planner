from pathlib import Path

import pytest
from diffusion_planner.train_cli import (
    build_parser,
    build_train_config,
    resolve_paths,
    to_command_line,
)
from diffusion_planner.train_config import TrainConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]
_NORMALIZATION_JSON = _REPO_ROOT / "diffusion_planner" / "normalization.json"


def test_build_parser_required_and_defaults(tmp_path: Path):
    parser = build_parser("test parser")
    train_list = str(tmp_path / "train.json")
    valid_list = str(tmp_path / "valid.json")

    args = parser.parse_args(
        [
            "--exp_name",
            "test_run",
            "--train_set_list",
            train_list,
            "--valid_set_list",
            valid_list,
        ]
    )
    assert args.exp_name == "test_run"
    assert args.train_set_list == train_list
    assert args.valid_set_list == valid_list
    assert args.use_wandb is True
    assert args.closed_loop_draw_workers == 4


def test_resolve_paths(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.json").write_text("[]")
    (tmp_path / "valid.json").write_text("[]")

    parser = build_parser("test parser")
    args = parser.parse_args(
        [
            "--exp_name",
            "test_run",
            "--train_set_list",
            "train.json",
            "--valid_set_list",
            "valid.json",
        ]
    )
    resolve_paths(args)
    assert Path(args.train_set_list).is_absolute()
    assert Path(args.valid_set_list).is_absolute()
    assert args.train_set_list == str((tmp_path / "train.json").resolve())


def test_to_command_line(tmp_path: Path):
    parser = build_parser("test parser")
    train_list = str(tmp_path / "train.json")
    valid_list = str(tmp_path / "valid.json")

    args = parser.parse_args(
        [
            "--exp_name",
            "test_run",
            "--train_set_list",
            train_list,
            "--valid_set_list",
            valid_list,
            "--closed_loop_draw_workers",
            "8",
        ]
    )
    cmd = to_command_line(args, exclude=("output_root",))
    assert "--exp_name" in cmd
    assert "test_run" in cmd
    assert "--train_set_list" in cmd
    assert "--closed_loop_draw_workers" in cmd
    assert "8" in cmd
    # Default values like use_wandb=True should be omitted
    assert "--use_wandb" not in cmd


def test_build_train_config(tmp_path: Path):
    parser = build_parser("test parser")
    train_list = str(tmp_path / "train.json")
    valid_list = str(tmp_path / "valid.json")

    args = parser.parse_args(
        [
            "--exp_name",
            "test_run",
            "--train_set_list",
            train_list,
            "--valid_set_list",
            valid_list,
        ]
    )
    config = build_train_config(
        args,
        num_workers=2,
        normalization_file_path=str(_NORMALIZATION_JSON),
    )
    assert isinstance(config, TrainConfig)
    assert config.exp_name == "test_run"
    assert config.train_set_list == train_list
    assert config.num_workers == 2
    assert config.save_dir != ""
    assert config.state_normalizer is not None
    assert config.observation_normalizer is not None
