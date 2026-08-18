"""Confirmation tooling for the Unknown-class rename augmentation: per-step scalar stats
(always on) and the optional before/after debug-image dump (off unless a debug_dir is set)."""

import torch

from diffusion_planner.utils.unknown_rename_debug import (
    apply_and_report_unknown_rename,
    rename_stats,
    save_unknown_rename_debug_image,
)

B, N, T, D = 1, 3, 4, 12


def _make_batch():
    x = torch.zeros(B, N, T, D)
    x[:, 0, :, 0] = 5.0
    x[:, 0, :, 8] = 1.0  # vehicle
    x[:, 1, :, 1] = 3.0
    x[:, 1, :, 9] = 1.0  # pedestrian
    return x


def test_rename_stats_reports_counts_and_rate():
    renamed = torch.tensor([[True, False, False]])
    valid = torch.tensor([[True, True, False]])
    stats = rename_stats(renamed, valid)
    assert stats == {
        "unknown_rename_count": 1,
        "unknown_rename_valid_count": 2,
        "unknown_rename_rate": 0.5,
    }


def test_rename_stats_rate_is_zero_when_no_valid_agents():
    renamed = torch.zeros(1, 2, dtype=torch.bool)
    valid = torch.zeros(1, 2, dtype=torch.bool)
    stats = rename_stats(renamed, valid)
    assert stats["unknown_rename_rate"] == 0.0


def test_apply_and_report_returns_numeric_only_stats_dict():
    """Regression: stats must stay pure numeric, since callers merge it straight into a loss
    dict that get_epoch_mean_loss averages key-by-key -- a string value would crash that."""
    x = _make_batch()
    out, stats = apply_and_report_unknown_rename(x, 0.0)
    assert out is x
    for v in stats.values():
        assert isinstance(v, (int, float))


def test_apply_and_report_disabled_by_default_no_debug_dir():
    torch.manual_seed(0)
    x = _make_batch()
    out, stats = apply_and_report_unknown_rename(x, 1.0, debug_dir="", step=0)
    assert stats["unknown_rename_count"] == stats["unknown_rename_valid_count"] == 2


def test_apply_and_report_dumps_image_on_cadence(tmp_path, capsys):
    torch.manual_seed(0)
    x = _make_batch()
    debug_dir = tmp_path / "debug"
    out, stats = apply_and_report_unknown_rename(
        x, 1.0, debug_dir=str(debug_dir), debug_every_n_steps=1, step=0, epoch=2
    )
    saved = list(debug_dir.glob("*.png"))
    assert len(saved) == 1
    assert saved[0].name == "epoch002_step00000.png"
    assert "saved" in capsys.readouterr().out


def test_apply_and_report_skips_dump_off_cadence(tmp_path):
    torch.manual_seed(0)
    x = _make_batch()
    debug_dir = tmp_path / "debug"
    apply_and_report_unknown_rename(
        x, 1.0, debug_dir=str(debug_dir), debug_every_n_steps=10, step=3, epoch=0
    )
    assert not debug_dir.exists() or not list(debug_dir.glob("*.png"))


def test_apply_and_report_skips_dump_when_nothing_renamed(tmp_path):
    x = _make_batch()
    debug_dir = tmp_path / "debug"
    apply_and_report_unknown_rename(
        x, 0.0, debug_dir=str(debug_dir), debug_every_n_steps=1, step=0
    )
    assert not debug_dir.exists() or not list(debug_dir.glob("*.png"))


def test_save_unknown_rename_debug_image_writes_a_file(tmp_path):
    before = _make_batch()
    after = before.clone()
    after[:, 0, :, 8:12] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    renamed = torch.tensor([[True, False, False]])

    out_path = tmp_path / "nested" / "debug.png"
    save_unknown_rename_debug_image(before, after, renamed, str(out_path))
    assert out_path.exists()
    assert out_path.stat().st_size > 0
