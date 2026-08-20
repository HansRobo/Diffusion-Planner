"""Confirmation tooling for the Unknown-class rename augmentation: per-step scalar stats
(always on) and the optional before/after debug-image dump (off unless a debug_dir is set;
one PNG per epoch, on that epoch's last training step)."""

import torch

from diffusion_planner.utils.unknown_rename_debug import (
    _build_unknown_rename_debug_figure,
    _context_from_inputs,
    apply_and_report_unknown_rename,
    rename_stats,
    save_unknown_rename_debug_image,
)

B, N, T, D = 1, 3, 4, 12
T_PAST = 5
T_FUTURE = 80  # draw_ego_vehicle indexes ego_agent_future[39]/[79] (the 4s/8s marks)


def _make_neighbors():
    x = torch.zeros(B, N, T, D)
    x[:, 0, :, 0] = 5.0
    x[:, 0, :, 8] = 1.0  # vehicle
    x[:, 1, :, 1] = 3.0
    x[:, 1, :, 9] = 1.0  # pedestrian
    return x


def _make_context_tensors():
    """Minimal-but-valid ego/lane/route tensors for draw_ego_vehicle/draw_lanes/draw_route."""
    ego_current_state = torch.zeros(B, 10)
    ego_current_state[:, 2] = 1.0  # cos(heading) = 1 -> heading 0

    ego_shape = torch.tensor([[2.5, 4.5, 1.8]])  # wheelbase, length, width

    ego_agent_past = torch.zeros(B, T_PAST, 4)
    ego_agent_past[:, :, 0] = torch.linspace(-4, 0, T_PAST)

    ego_agent_future = torch.zeros(B, T_FUTURE, 3)  # raw [x, y, heading]
    ego_agent_future[:, :, 0] = torch.linspace(1, 40, T_FUTURE)

    lanes = torch.zeros(B, 1, 3, 13)
    lanes[:, 0, :, 0] = torch.linspace(0, 10, 3)  # centerline x

    route_lanes = torch.zeros(B, 1, 3, 2)
    route_lanes[:, 0, :, 0] = torch.linspace(0, 10, 3)

    return {
        "ego_current_state": ego_current_state,
        "ego_shape": ego_shape,
        "ego_agent_past": ego_agent_past,
        "ego_agent_future": ego_agent_future,
        "lanes": lanes,
        "route_lanes": route_lanes,
    }


def _make_inputs():
    return {"neighbor_agents_past": _make_neighbors(), **_make_context_tensors()}


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
    inputs = _make_inputs()
    out, stats = apply_and_report_unknown_rename(inputs, 0.0)
    assert out is inputs["neighbor_agents_past"]
    for v in stats.values():
        assert isinstance(v, (int, float))


def test_apply_and_report_disabled_by_default_no_debug_dir():
    torch.manual_seed(0)
    inputs = _make_inputs()
    out, stats = apply_and_report_unknown_rename(inputs, 1.0, debug_dir="", is_last_step=True)
    assert stats["unknown_rename_count"] == stats["unknown_rename_valid_count"] == 2


def test_apply_and_report_dumps_image_on_last_step(tmp_path, capsys):
    torch.manual_seed(0)
    inputs = _make_inputs()
    debug_dir = tmp_path / "debug"
    apply_and_report_unknown_rename(
        inputs, 1.0, debug_dir=str(debug_dir), is_last_step=True, epoch=2
    )
    saved = list(debug_dir.glob("*.png"))
    assert len(saved) == 1
    assert saved[0].name == "epoch002.png"
    assert "saved" in capsys.readouterr().out


def test_apply_and_report_skips_dump_when_not_last_step(tmp_path):
    torch.manual_seed(0)
    inputs = _make_inputs()
    debug_dir = tmp_path / "debug"
    apply_and_report_unknown_rename(inputs, 1.0, debug_dir=str(debug_dir), is_last_step=False)
    assert not debug_dir.exists() or not list(debug_dir.glob("*.png"))


def test_apply_and_report_skips_dump_when_nothing_renamed(tmp_path):
    inputs = _make_inputs()
    debug_dir = tmp_path / "debug"
    apply_and_report_unknown_rename(inputs, 0.0, debug_dir=str(debug_dir), is_last_step=True)
    assert not debug_dir.exists() or not list(debug_dir.glob("*.png"))


def test_save_unknown_rename_debug_image_writes_a_file(tmp_path):
    before = _make_neighbors()
    after = before.clone()
    after[:, 0, :, 8:12] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    renamed = torch.tensor([[True, False, False]])
    context = _context_from_inputs(_make_context_tensors(), sample_idx=0)

    out_path = tmp_path / "nested" / "debug.png"
    save_unknown_rename_debug_image(before, after, renamed, context, str(out_path))
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_figure_has_explanatory_titles_and_legend():
    """The redesign's actual point: before/after must be explained in the image itself, the
    ego marker must be a labeled, legended element (not an unlabeled black square), and each
    panel's legend must cover ego, neighbor-type colors, and the renamed-agent marker."""
    before = _make_neighbors()
    after = before.clone()
    after[:, 0, :, 8:12] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    renamed = torch.tensor([[True, False, False]])
    context = _context_from_inputs(_make_context_tensors(), sample_idx=0)

    fig = _build_unknown_rename_debug_figure(before, after, renamed, context)
    try:
        assert len(fig.axes) == 2
        before_ax, after_ax = fig.axes

        assert "Before rename" in before_ax.get_title()
        assert "After rename" in after_ax.get_title()

        suptitle_text = fig._suptitle.get_text()
        assert "BEFORE" in suptitle_text
        assert "AFTER" in suptitle_text

        for ax in (before_ax, after_ax):
            legend = ax.get_legend()
            assert legend is not None
            labels = {t.get_text() for t in legend.get_texts()}
            assert "Ego (current pose)" in labels
            assert "Ego GT future (ground truth, not model output)" in labels
            assert "Vehicle" in labels
            assert "Pedestrian" in labels
            assert "Bicycle" in labels
            assert "Unknown" in labels
            assert "Renamed to Unknown this step" in labels
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)
