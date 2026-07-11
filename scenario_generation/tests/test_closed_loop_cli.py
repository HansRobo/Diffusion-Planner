from argparse import Namespace
from pathlib import Path

from scenario_generation import closed_loop_cli


def test_full_route_runner_forwards_replan_interval(monkeypatch, tmp_path):
    model = object()
    model_args = object()
    captured = {}

    monkeypatch.setattr(
        closed_loop_cli,
        "load_model",
        lambda model_path, device: (model, model_args),
    )

    def fake_run_closed_loop_eval(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"n_segments": 0}

    monkeypatch.setattr(closed_loop_cli, "run_closed_loop_eval", fake_run_closed_loop_eval)
    args = Namespace(
        model_path=tmp_path / "model.pth",
        npz_root=tmp_path / "npz",
        out_dir=tmp_path / "out",
        device="cpu",
        seg_len=123,
        near_miss_thresh=0.4,
        search_radius=1.7,
        warmup_steps=2,
        unstick_after=30,
        unstick_advance_m=2.0,
        unstick_radius_mult=4.0,
        unstick_teleport_after=50,
        fps=10,
        replan_interval=7,
        draw_every=3,
    )

    result = closed_loop_cli.run_full_route_eval(args)

    assert result["model_path"] == str(args.model_path)
    assert captured["args"][:2] == (model, model_args)
    assert captured["kwargs"]["replan_interval"] == 7
    assert captured["kwargs"]["neighbor_history_mode"] == "recorded"


def test_grouped_runner_requires_classification_and_output(tmp_path):
    args = Namespace(classification_json=None, out_dir=Path(tmp_path))

    try:
        closed_loop_cli.run_grouped_eval(args)
    except SystemExit as exc:
        assert "classification_json" in str(exc)
    else:
        raise AssertionError("missing classification JSON must be rejected")


def test_grouped_runner_forwards_unstick_stages(monkeypatch, tmp_path):
    model = object()
    model_args = object()
    captured = {}

    monkeypatch.setattr(
        closed_loop_cli,
        "load_model",
        lambda model_path, device: (model, model_args),
    )

    def fake_run_grouped_closed_loop_eval(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"n_episodes": 0, "n_episodes_expected": 0}

    monkeypatch.setattr(
        closed_loop_cli,
        "run_grouped_closed_loop_eval",
        fake_run_grouped_closed_loop_eval,
    )
    args = Namespace(
        model_path=tmp_path / "model.pth",
        npz_root=tmp_path / "npz",
        classification_json=tmp_path / "classification.json",
        out_dir=tmp_path / "out",
        device="cpu",
        near_miss_thresh=0.4,
        search_radius=1.7,
        warmup_steps=2,
        unstick_after=30,
        unstick_advance_m=2.0,
        unstick_radius_mult=4.0,
        unstick_teleport_after=50,
        draw_every=3,
        fps=10,
        areas=None,
    )

    assert closed_loop_cli.run_grouped_eval(args) == 0
    assert captured["args"][:2] == (model, model_args)
    assert captured["kwargs"]["unstick_radius_mult"] == 4.0
    assert captured["kwargs"]["unstick_teleport_after"] == 50
