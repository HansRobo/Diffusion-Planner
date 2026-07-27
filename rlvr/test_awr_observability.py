import copy
import hashlib
import json
import pathlib
import random
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import rlvr.train_awr as train_awr_module
import rlvr.autoresearch.tools.build_replay_decoder_context as build_context_module
from rlvr.autoresearch.tools.backfill_awr_replay_tail_distributed import (
    _open_resume_writer,
)
from rlvr.autoresearch.tools.attach_replay_decoder_context import (
    attach as attach_decoder_context,
)
from rlvr.awr import (
    AWRRollout,
    AWRRolloutConfig,
    _add_rollout_reward_diagnostics,
    _apply_hdp_trajectory_augmentation,
    _first_waypoint_gate_and_diagnostics,
    _first_waypoint_gate_and_diagnostics_batch,
    compute_awr_weights,
    sample_unguided_dp_group_batch,
    save_checkpoint_pair,
    update_ema,
)
from rlvr.grpo_loss import compute_batched_trajectory_losses
from rlvr.reward import RewardBreakdown, RewardConfig
from rlvr.train_awr import (
    _checkpoint_encoder_sha256,
    _commit_epoch_ema_policy_update,
    _compact_numeric_row_summary,
    _compact_replay_decoder_context,
    _DiskReplayReader,
    _DiskReplayWriter,
    _effective_eval_scene_load_workers,
    _file_sha256,
    _find_inherited_refresh_summaries,
    _is_mine_only_refresh_run,
    _iter_prefetched_replay_batches,
    _iter_prefetched_scene_batches,
    _load_inherited_refresh_baseline,
    _merge_compact_numeric_row_summaries,
    _ordered_paths_sha256,
    _preserve_training_rng,
    _restore_optimizer_state,
    _restore_best_policy_transaction,
    _salvage_partial_replay_manifest,
    _shared_decoder_context_contract,
    _shared_expert_anchor_contract,
    _validate_replay_source_contract,
    _validate_expected_paths_sha256,
    _wandb_log_epoch_summaries,
    _wandb_readable_aliases,
    _wandb_summary_payload,
    _warmup_compiled_models,
)


def test_ordered_scene_path_hash_is_delimited_and_order_sensitive() -> None:
    paths = ["/a/scene.npz", "/b/scene.npz"]
    expected = hashlib.sha256(b"/a/scene.npz\n/b/scene.npz\n").hexdigest()
    assert _ordered_paths_sha256(paths) == expected
    assert _ordered_paths_sha256(list(reversed(paths))) != expected
    assert _ordered_paths_sha256(["ab", "c"]) != _ordered_paths_sha256(
        ["a", "bc"]
    )


def test_expected_scene_path_hash_fails_closed() -> None:
    paths = ["a", "b"]
    expected = _ordered_paths_sha256(paths)
    assert _validate_expected_paths_sha256("selector", paths, expected) == expected
    with pytest.raises(RuntimeError, match="ordered scene SHA-256 differs"):
        _validate_expected_paths_sha256("selector", list(reversed(paths)), expected)
    with pytest.raises(ValueError, match="malformed"):
        _validate_expected_paths_sha256("selector", paths, "not-a-sha")


def test_eval_and_replay_loader_parallelism_can_be_tuned_independently() -> None:
    # Legacy configs retain their one shared value.
    assert _effective_eval_scene_load_workers({"scene_load_workers": 4}) == 4
    # Production config can use the measured replay optimum without slowing
    # full-scene validation on the validation NPZ store.
    assert (
        _effective_eval_scene_load_workers(
            {"scene_load_workers": 4, "eval_scene_load_workers": 1}
        )
        == 1
    )
    assert _effective_eval_scene_load_workers({"eval_scene_load_workers": 0}) == 1


def test_compile_warmup_uses_packed_loader_for_scene_lists(monkeypatch) -> None:
    bulk_data = {"source": "packed"}
    single_data = {"source": "single"}
    packed_calls = []
    scene_rollout_calls = []
    batch_rollout_calls = []

    def fake_packed(paths, device, workers):
        packed_calls.append((list(paths), device.type, workers))
        return bulk_data

    monkeypatch.setattr(train_awr_module, "_load_scene_batch", fake_packed)
    monkeypatch.setattr(
        train_awr_module,
        "load_scene",
        lambda path, device: single_data,
    )

    def fake_rollout(model, model_args, data, config, reward, device, **kwargs):
        del model, model_args, config, reward, device, kwargs
        scene_rollout_calls.append(data)
        return object()

    monkeypatch.setattr(
        train_awr_module, "rollout_and_score_scene", fake_rollout
    )
    def fake_batch_rollout(
        model, model_args, data, config, reward, device, **kwargs
    ):
        del model, model_args, config, reward, device, kwargs
        batch_rollout_calls.append(data)
        return [object(), object()]

    monkeypatch.setattr(
        train_awr_module,
        "rollout_and_score_scene_batch",
        fake_batch_rollout,
    )
    result = _warmup_compiled_models(
        object(),
        object(),
        SimpleNamespace(),
        ["a.npz", "b.npz"],
        AWRRolloutConfig(plannerrft_guided_exploration=False),
        RewardConfig(),
        torch.device("cpu"),
    )
    assert result["success"] is True
    assert result["native_scene_batch"] == 2
    assert packed_calls == [(["a.npz", "b.npz"], "cpu", 1)]
    assert scene_rollout_calls == []
    assert batch_rollout_calls == [bulk_data, bulk_data, bulk_data]


def test_checkpoint_can_preserve_optimizer_moments_without_overriding_new_lr(
    tmp_path,
) -> None:
    model = torch.nn.Linear(3, 2)
    ema = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.ones(4, 3)).sum().backward()
    optimizer.step()

    checkpoint = tmp_path / "restartable.pth"
    save_checkpoint_pair(
        model,
        ema,
        checkpoint,
        epoch=20,
        optimizer=optimizer,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["optimizer_class"] == "AdamW"
    assert payload["optimizer_state_dict"]["state"]
    assert all(
        not value.is_cuda
        for state in payload["optimizer_state_dict"]["state"].values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )

    resumed_model = torch.nn.Linear(3, 2)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=9e-4)
    info = _restore_optimizer_state(
        resumed_optimizer,
        checkpoint,
        learning_rate=2e-5,
    )
    assert info["source_epoch"] == 20
    assert info["state_parameter_count"] == 2
    assert all(group["lr"] == pytest.approx(2e-5) for group in resumed_optimizer.param_groups)


def test_ema_blends_only_trainable_parameters_and_keeps_frozen_state_exact() -> None:
    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.trainable = torch.nn.Parameter(torch.tensor([2.0]))
            self.frozen = torch.nn.Parameter(
                torch.tensor([1.234567]), requires_grad=False
            )
            self.register_buffer("running", torch.tensor([4.0]))

    policy = TinyPolicy()
    ema = copy.deepcopy(policy)
    with torch.no_grad():
        ema.trainable.fill_(0.0)
        ema.frozen.copy_(policy.frozen)
        policy.running.fill_(7.0)
    for _ in range(10_000):
        update_ema(ema, policy, decay=0.999)
    assert torch.equal(ema.frozen, policy.frozen)
    assert torch.equal(ema.running, policy.running)
    assert ema.trainable.item() == pytest.approx(
        2.0 * (1.0 - 0.999**10_000), rel=2e-4
    )


def test_conservative_epoch_policy_commit_uses_configured_rate_and_resets_proposal() -> None:
    class TinyPolicy(torch.nn.Module):
        def __init__(self, trainable: float, frozen: float):
            super().__init__()
            self.trainable = torch.nn.Parameter(torch.tensor([trainable]))
            self.frozen = torch.nn.Parameter(
                torch.tensor([frozen]), requires_grad=False
            )
            self.register_buffer("state", torch.tensor([3.0]))

    live = TinyPolicy(3.0, 7.0)
    old = TinyPolicy(1.0, -1.0)
    for parameter in old.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW([live.trainable], lr=1e-3)
    live.trainable.square().sum().backward()
    optimizer.step()
    # Make the proposal exact and leave populated Adam moments to verify the
    # optimizer reset performed by the explicit conservative transaction.
    with torch.no_grad():
        live.trainable.fill_(3.0)
    assert optimizer.state

    stats = _commit_epoch_ema_policy_update(
        live,
        old,
        optimizer,
        decay=0.95,
    )

    assert old.trainable.item() == pytest.approx(1.1)
    assert live.trainable.item() == pytest.approx(1.1)
    # Frozen state is copied exactly, never blended.
    assert torch.equal(old.frozen, live.frozen)
    assert torch.equal(old.state, live.state)
    assert len(optimizer.state) == 0
    assert stats["ema_policy_commit_rate"] == pytest.approx(0.05)
    assert stats["policy_accepted_relative_l2"] == pytest.approx(
        stats["policy_proposal_relative_l2"] * 0.05
    )
    assert stats["optimizer_state_reset_after_policy_commit"] == 1.0


def test_rejected_policy_transaction_restores_best_ema_with_compiled_keys(
    tmp_path,
) -> None:
    class TinyPolicy(torch.nn.Module):
        def __init__(self, value: float):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([value]))

    class CompiledLike(torch.nn.Module):
        def __init__(self, value: float):
            super().__init__()
            self._orig_mod = TinyPolicy(value)

    selected = TinyPolicy(2.5)
    checkpoint = tmp_path / "best_train.pth"
    save_checkpoint_pair(
        selected,
        selected,
        checkpoint,
        epoch=7,
    )

    live = CompiledLike(9.0)
    behavior = CompiledLike(-3.0)
    optimizer = torch.optim.AdamW(live.parameters(), lr=1e-3)
    live._orig_mod.weight.square().sum().backward()
    optimizer.step()
    assert optimizer.state

    info = _restore_best_policy_transaction(
        live,
        behavior,
        optimizer,
        checkpoint,
    )
    assert live._orig_mod.weight.item() == pytest.approx(2.5)
    assert behavior._orig_mod.weight.item() == pytest.approx(2.5)
    assert optimizer.state == {}
    assert info["restored_epoch"] == 7
    assert info["optimizer_state_reset"] is True


def test_compact_replay_summary_preserves_means_and_percentiles_exactly() -> None:
    rank0 = [
        {"scene_count": 192, "loss": 1.0, "grad_norm": 5.0, "other": 7.0},
        {"scene_count": 192, "loss": 2.0, "grad_norm": 4.0, "other": 9.0},
    ]
    rank1 = [
        {"scene_count": 192, "loss": 3.0, "grad_norm": 3.0, "other": 11.0},
        {"scene_count": 192, "loss": 4.0, "grad_norm": 2.0, "other": 13.0},
        {"scene_count": 64, "loss": 5.0, "grad_norm": 1.0, "other": 15.0},
    ]
    summaries = [
        _compact_numeric_row_summary(rank0, include_percentile_values=True),
        _compact_numeric_row_summary(rank1, include_percentile_values=True),
    ]
    merged, sums = _merge_compact_numeric_row_summaries(summaries)
    losses = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    gradients = np.asarray([5.0, 4.0, 3.0, 2.0, 1.0])

    assert merged["scene_count"] == 832
    assert merged["row_count"] == 5
    assert merged["scene_rows_materialized"] == 0
    assert merged["numeric_percentiles_omitted"] == 0
    assert merged["mean_loss"] == pytest.approx(float(losses.mean()))
    assert merged["mean_grad_norm"] == pytest.approx(float(gradients.mean()))
    assert merged["mean_other"] == pytest.approx(11.0)
    assert merged["p50_loss"] == pytest.approx(float(np.median(losses)))
    assert merged["p90_loss"] == pytest.approx(float(np.percentile(losses, 90)))
    assert merged["p50_grad_norm"] == pytest.approx(float(np.median(gradients)))
    assert merged["p90_grad_norm"] == pytest.approx(
        float(np.percentile(gradients, 90))
    )
    assert "p50_other" not in merged
    assert sums["loss"] == pytest.approx(float(losses.sum()))


def test_file_sha256_streams_checkpoint_provenance(tmp_path) -> None:
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"original-dp-awr\x00checkpoint")
    assert (
        _file_sha256(checkpoint, chunk_size=3)
        == "8513abc2dbe9a6ca83877038dbaaf570740d7fdca1d2f639cb544a8231db52a2"
    )


def _contract_drift(capsys, *args, _call=None, **kwargs) -> str:
    """Run a compatibility check and return what it reported about drift."""

    capsys.readouterr()
    (_call or _validate_replay_source_contract)(*args, **kwargs)
    return capsys.readouterr().out


def _write_inherited_baseline_source(tmp_path, monkeypatch):
    """Build a mine-only run that a replay may legitimately inherit from."""

    monkeypatch.setenv("DP_NEIGHBOR_FUTURE_OFFSET", "1")
    source = tmp_path / "mine"
    source.mkdir()
    checkpoint = source / "checkpoint.pth"
    args_path = source / "args.json"
    torch.save(
        {
            "model": {"weight": torch.tensor([1.0])},
            "ema_state_dict": {"weight": torch.tensor([2.0])},
        },
        checkpoint,
    )
    args_path.write_text('{"predicted_neighbor_num": 2}')
    rollout_config = AWRRolloutConfig()
    rollout_config.inference_amp_dtype = "bf16"
    reward_config = RewardConfig()
    source.joinpath("provenance.json").write_text(
        json.dumps(
            {
                "staged_model": str(checkpoint),
                "staged_args": str(args_path),
                "resume_replay_root": None,
                "use_policy_state": False,
                "start_epoch": 1,
                "neighbor_future_alignment_offset": 1,
                "x2_legacy_ego_width_m": 2.29156,
                "xx1_legacy_ego_width_m": 1.70,
            }
        )
    )
    reward_payload = asdict(reward_config)
    reward_payload["effective_quality_weights"] = {"ttc": 5.0}
    reward_payload["hdp_multi_weight_fields_active"] = False
    source.joinpath("effective_config.json").write_text(
        json.dumps(
            {
                "training": {
                    "train_epochs": 1,
                    "hdp_rollout_interval": 10,
                    "eval_k": rollout_config.n_trajectories,
                    "eval_sample_steps": rollout_config.sample_steps,
                },
                "awr": asdict(rollout_config),
                "reward": reward_payload,
            }
        )
    )
    source.joinpath("final_summary.json").write_text("{}")
    valid_paths = ["valid_a.npz", "valid_b.npz"]
    selector_paths = ["train_a.npz", "train_b.npz", "train_c.npz"]
    for directory, paths in (
        (source / "eval_epoch_000", valid_paths),
        (source / "train_selector_epoch_000", selector_paths),
    ):
        directory.mkdir()
        directory.joinpath("summary.json").write_text(
            json.dumps({"scene_count": float(len(paths)), "mean_det_reward": 0.9})
        )
        directory.joinpath("scenes.json").write_text(
            json.dumps([{"scene_path": path} for path in paths])
        )
    return (
        source,
        checkpoint,
        args_path,
        rollout_config,
        reward_config,
        valid_paths,
        selector_paths,
    )


def test_inherited_refresh_baseline_reports_checkpoint_config_and_order_drift(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    (
        source,
        checkpoint,
        args_path,
        rollout_config,
        reward_config,
        valid_paths,
        selector_paths,
    ) = _write_inherited_baseline_source(tmp_path, monkeypatch)

    inherited = _load_inherited_refresh_baseline(
        source,
        current_checkpoint=checkpoint,
        current_args=args_path,
        valid_paths=valid_paths,
        train_selector_paths=selector_paths,
        rollout_config=rollout_config,
        reward_config=reward_config,
        eval_k=rollout_config.n_trajectories,
        eval_sample_steps=rollout_config.sample_steps,
        verify_checkpoint_hash=True,
    )
    assert inherited[0]["scene_count"] == 2
    assert inherited[2]["scene_count"] == 3

    # de551822 converted these two aborts into warnings.  The baseline supplies
    # the "before" numbers the cycle-level commit veto tests against, so what
    # must hold is that reusing one measured on a different model or a different
    # scene order is never silent.
    different_checkpoint = tmp_path / "different.pth"
    torch.save({"model": {"weight": torch.tensor([9.0])}}, different_checkpoint)
    assert "checkpoint differs" in _contract_drift(
        capsys,
        source,
        current_checkpoint=different_checkpoint,
        current_args=args_path,
        valid_paths=valid_paths,
        train_selector_paths=selector_paths,
        rollout_config=rollout_config,
        reward_config=reward_config,
        eval_k=rollout_config.n_trajectories,
        eval_sample_steps=rollout_config.sample_steps,
        verify_checkpoint_hash=True,
        _call=_load_inherited_refresh_baseline,
    )
    assert "valid scene order differs" in _contract_drift(
        capsys,
        source,
        current_checkpoint=checkpoint,
        current_args=args_path,
        valid_paths=list(reversed(valid_paths)),
        train_selector_paths=selector_paths,
        rollout_config=rollout_config,
        reward_config=reward_config,
        eval_k=rollout_config.n_trajectories,
        eval_sample_steps=rollout_config.sample_steps,
        verify_checkpoint_hash=False,
        _call=_load_inherited_refresh_baseline,
    )

    # Sample steps still abort, and must: they change what the numbers mean.
    with pytest.raises(RuntimeError, match="evaluation sample steps differ"):
        _load_inherited_refresh_baseline(
            source,
            current_checkpoint=checkpoint,
            current_args=args_path,
            valid_paths=valid_paths,
            train_selector_paths=selector_paths,
            rollout_config=rollout_config,
            reward_config=reward_config,
            eval_k=rollout_config.n_trajectories,
            eval_sample_steps=rollout_config.sample_steps + 1,
            verify_checkpoint_hash=False,
        )


def test_checkpoint_encoder_hash_tracks_selected_payload_only(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    payload = {
        "model": {
            "module.encoder.weight": torch.tensor([[1.0, 2.0]]),
            "module.decoder.weight": torch.tensor([3.0]),
        },
        "ema_state_dict": {
            "encoder.weight": torch.tensor([[1.0, 2.0]]),
            "decoder.weight": torch.tensor([9.0]),
        },
    }
    torch.save(payload, checkpoint)
    live_hash = _checkpoint_encoder_sha256(checkpoint, use_policy_state=True)
    ema_hash = _checkpoint_encoder_sha256(checkpoint, use_policy_state=False)
    assert live_hash == ema_hash

    payload["ema_state_dict"]["decoder.weight"].fill_(17.0)
    torch.save(payload, checkpoint)
    assert _checkpoint_encoder_sha256(
        checkpoint, use_policy_state=False
    ) == ema_hash

    payload["ema_state_dict"]["encoder.weight"][0, 0] += 0.25
    torch.save(payload, checkpoint)
    assert _checkpoint_encoder_sha256(
        checkpoint, use_policy_state=False
    ) != ema_hash


def test_only_standalone_faithful_refresh_reuses_baseline_evaluation() -> None:
    common = {
        "rollout_interval": 10,
        "faithful_disk_replay": True,
        "resumed_replay": False,
    }
    assert _is_mine_only_refresh_run(start_epoch=21, epochs=21, **common)
    assert _is_mine_only_refresh_run(start_epoch=1, epochs=1, **common)
    assert not _is_mine_only_refresh_run(start_epoch=1, epochs=10, **common)
    assert not _is_mine_only_refresh_run(start_epoch=12, epochs=20, **common)
    assert not _is_mine_only_refresh_run(
        start_epoch=21,
        epochs=21,
        rollout_interval=10,
        faithful_disk_replay=True,
        resumed_replay=True,
    )
    assert not _is_mine_only_refresh_run(
        start_epoch=21,
        epochs=21,
        rollout_interval=10,
        faithful_disk_replay=False,
        resumed_replay=False,
    )


def test_inherited_refresh_prefers_global_epoch_and_supports_cycle2_legacy(
    tmp_path,
) -> None:
    legacy_train = tmp_path / "train_epoch_001" / "summary.json"
    legacy_eval = tmp_path / "eval_epoch_001" / "summary.json"
    legacy_train.parent.mkdir()
    legacy_eval.parent.mkdir()
    legacy_train.write_text("{}")
    legacy_eval.write_text("{}")
    assert _find_inherited_refresh_summaries(tmp_path, 11) == (
        legacy_train,
        legacy_eval,
    )

    global_train = tmp_path / "train_epoch_021" / "summary.json"
    global_eval = tmp_path / "eval_epoch_021" / "summary.json"
    global_train.parent.mkdir()
    global_eval.parent.mkdir()
    global_train.write_text("{}")
    global_eval.write_text("{}")
    assert _find_inherited_refresh_summaries(tmp_path, 21) == (
        global_train,
        global_eval,
    )


def test_rollout_prefetch_preserves_order_and_returns_loader_errors(
    monkeypatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_load(
        paths,
        device,
        workers=1,
        replay_context_only=False,
        defer_full_scene_heading_transforms=False,
    ):
        del device, workers, replay_context_only, defer_full_scene_heading_transforms
        calls.append(tuple(paths))
        if "scene_2" in paths:
            raise ValueError("bad packed scene")
        return {"marker": torch.tensor([int(paths[0].split("_")[-1])])}

    monkeypatch.setattr(train_awr_module, "_load_scene_batch", fake_load)
    prepared = list(
        _iter_prefetched_scene_batches(
            [f"scene_{index}" for index in range(5)],
            range(0, 5, 2),
            batch_size=2,
            scene_load_workers=1,
            prefetch_batches=2,
        )
    )

    assert calls == [
        ("scene_0", "scene_1"),
        ("scene_2", "scene_3"),
        ("scene_4",),
    ]
    assert [item[0] for item in prepared] == [0, 2, 4]
    assert [item[1] for item in prepared] == [
        ["scene_0", "scene_1"],
        ["scene_2", "scene_3"],
        ["scene_4"],
    ]
    assert prepared[0][2]["marker"].item() == 0
    assert prepared[0][3] is None
    assert prepared[1][2] is None
    assert isinstance(prepared[1][3], ValueError)
    assert prepared[2][2]["marker"].item() == 4


def test_rollout_prefetch_attaches_shared_encoding_in_exact_scene_order(
    monkeypatch,
) -> None:
    shared = np.arange(5 * 2 * 3, dtype=np.float32).reshape(5, 2, 3)

    def fake_load(
        paths,
        device,
        workers=1,
        replay_context_only=False,
        defer_full_scene_heading_transforms=False,
    ):
        del device, workers, replay_context_only, defer_full_scene_heading_transforms
        return {"marker": torch.tensor([int(path.split("_")[-1]) for path in paths])}

    monkeypatch.setattr(train_awr_module, "_load_scene_batch", fake_load)
    prepared = list(
        _iter_prefetched_scene_batches(
            [f"scene_{index}" for index in range(5)],
            range(0, 5, 2),
            batch_size=2,
            scene_load_workers=1,
            prefetch_batches=0,
            shared_scene_encoding=shared,
        )
    )
    assert [item[0] for item in prepared] == [0, 2, 4]
    assert torch.equal(prepared[0][2]["_cached_encoding"], torch.from_numpy(shared[:2]))
    assert torch.equal(prepared[1][2]["_cached_encoding"], torch.from_numpy(shared[2:4]))
    assert torch.equal(prepared[2][2]["_cached_encoding"], torch.from_numpy(shared[4:]))


def test_rollout_prefetch_attaches_shared_expert_anchor_in_exact_scene_order(
    monkeypatch,
) -> None:
    shared = {
        "expert_trajectories": np.arange(
            5 * 4 * 4, dtype=np.float32
        ).reshape(5, 4, 4),
        "expert_rewards": np.linspace(0.5, 0.9, 5, dtype=np.float32),
        "expert_safe": np.asarray([1, 0, 1, 1, 0], dtype=np.float32),
    }

    def fake_load(
        paths,
        device,
        workers=1,
        replay_context_only=False,
        defer_full_scene_heading_transforms=False,
    ):
        del device, workers, replay_context_only, defer_full_scene_heading_transforms
        return {"marker": torch.tensor([int(path.split("_")[-1]) for path in paths])}

    monkeypatch.setattr(train_awr_module, "_load_scene_batch", fake_load)
    prepared = list(
        _iter_prefetched_scene_batches(
            [f"scene_{index}" for index in range(5)],
            range(0, 5, 2),
            batch_size=2,
            scene_load_workers=1,
            prefetch_batches=0,
            shared_expert_anchor=shared,
        )
    )
    keys = {
        "expert_trajectories": train_awr_module._REPLAY_EXPERT_TRAJECTORY_KEY,
        "expert_rewards": train_awr_module._REPLAY_EXPERT_REWARD_KEY,
        "expert_safe": train_awr_module._REPLAY_EXPERT_SAFE_KEY,
    }
    for batch_index, (start, _, data, error) in enumerate(prepared):
        del batch_index
        assert error is None
        end = min(start + 2, 5)
        for name, key in keys.items():
            assert torch.equal(data[key], torch.from_numpy(shared[name][start:end]))


def test_batched_sampler_uses_shared_encoding_without_running_encoder() -> None:
    batch_size, group_size, future_len = 2, 3, 4

    class ForbiddenEncoder(torch.nn.Module):
        def forward(self, inputs):
            del inputs
            raise AssertionError("shared encoding must bypass the frozen encoder")

    class DecoderStub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._guidance_fn = None
            self._guidance_scale = 0.0
            self._sample_steps = 10

    class PlannerStub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = ForbiddenEncoder()
            self.decoder = DecoderStub()

        def forward(self, inputs):
            assert set(inputs) == {
                "ego_current_state",
                "neighbor_agents_past",
                "delay",
                "sampled_trajectories",
                "_cached_encoding",
                "_skip_turn_indicator",
            }
            assert inputs["ego_current_state"].shape == (
                batch_size * group_size,
                4,
            )
            assert inputs["neighbor_agents_past"].shape == (
                batch_size * group_size,
                1,
                1,
                4,
            )
            encoding = inputs["_cached_encoding"]
            assert encoding.shape[0] == batch_size * group_size
            prediction = torch.zeros(
                batch_size * group_size, 1, future_len, 4
            )
            return encoding, {"prediction": prediction}

    cached = torch.arange(batch_size * 2 * 3, dtype=torch.float32).reshape(
        batch_size, 2, 3
    )
    data = {
        "ego_current_state": torch.zeros(batch_size, 10),
        "neighbor_agents_past": torch.zeros(batch_size, 1, 3, 11),
        "delay": torch.zeros(batch_size, dtype=torch.long),
        "_cached_encoding": cached,
    }
    model_args = SimpleNamespace(
        predicted_neighbor_num=1,
        future_len=future_len,
        observation_normalizer=lambda value: value,
    )
    config = AWRRolloutConfig(
        n_trajectories=group_size,
        sample_steps=5,
        deterministic_first=False,
        inference_amp_dtype="off",
    )
    trajectories, noise_scales, observed_encoding = sample_unguided_dp_group_batch(
        PlannerStub(), model_args, data, config, torch.device("cpu")
    )
    assert trajectories.shape == (batch_size, group_size, future_len, 4)
    assert noise_scales.shape == (batch_size, group_size)
    assert torch.equal(observed_encoding, cached)


def test_evaluation_rng_transaction_preserves_training_streams() -> None:
    random.seed(101)
    np.random.seed(102)
    torch.manual_seed(103)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()

    with _preserve_training_rng(torch.device("cpu")):
        _ = random.random()
        _ = np.random.rand()
        _ = torch.rand(4)

    observed = (random.random(), float(np.random.rand()), torch.rand(4))
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    expected = (random.random(), float(np.random.rand()), torch.rand(4))
    assert observed[0] == expected[0]
    assert observed[1] == expected[1]
    assert torch.equal(observed[2], expected[2])


def test_inherited_baseline_drift_is_reported_to_the_caller(
    tmp_path, monkeypatch, capsys
) -> None:
    """The loader must hand its findings back, not just print them.

    inherited_refresh_baseline.json used to record verified=true unconditionally.
    Since de551822 the checks behind those flags only warn, so a baseline
    measured on a different checkpoint was still written down as verified -- and
    that baseline is the "before" side of the cycle-level commit veto.
    """

    (
        source,
        checkpoint,
        args_path,
        rollout_config,
        reward_config,
        valid_paths,
        selector_paths,
    ) = _write_inherited_baseline_source(tmp_path, monkeypatch)

    clean: list[str] = []
    _load_inherited_refresh_baseline(
        source,
        current_checkpoint=checkpoint,
        current_args=args_path,
        valid_paths=valid_paths,
        train_selector_paths=selector_paths,
        rollout_config=rollout_config,
        reward_config=reward_config,
        eval_k=rollout_config.n_trajectories,
        eval_sample_steps=rollout_config.sample_steps,
        verify_checkpoint_hash=True,
        drift=clean,
    )
    assert clean == [], f"a matching baseline reported drift: {clean}"

    different = tmp_path / "other.pth"
    torch.save({"model": {"weight": torch.tensor([9.0])}}, different)
    found: list[str] = []
    _load_inherited_refresh_baseline(
        source,
        current_checkpoint=different,
        current_args=args_path,
        valid_paths=valid_paths,
        train_selector_paths=selector_paths,
        rollout_config=rollout_config,
        reward_config=reward_config,
        eval_k=rollout_config.n_trajectories,
        eval_sample_steps=rollout_config.sample_steps,
        verify_checkpoint_hash=True,
        drift=found,
    )
    assert any("checkpoint differs" in item for item in found), found

    # The artifact is written from inside main(), which no test can drive, so
    # pin the one property that matters at source level: none of its verified
    # flags may be a literal again.
    source_text = pathlib.Path(train_awr_module.__file__).read_text()
    block = source_text.split('inherited_contract = {', 1)[1].split('}', 1)[0]
    assert '"verified": verified' in block, block
    for flag in (
        "valid_scene_order_verified",
        "train_selector_scene_order_verified",
        "awr_config_verified",
        "reward_config_verified",
    ):
        assert f'"{flag}": True' not in block, (
            f"{flag} is hardcoded true again; the check behind it only warns"
        )
    assert '"drift": list(inherited_drift)' in block


def test_replay_contract_reports_a_silent_weighting_change(
    tmp_path, monkeypatch, capsys
) -> None:
    # de551822 converted 17 of these aborts into warnings on purpose: the
    # pipeline is designed to retune between cycles, so probe-driven drift is
    # legitimate and aborting on it killed twelve consecutive replay restarts.
    # What still has to hold is that the drift is never SILENT -- that is the
    # property this guards, and asserting pytest.raises stopped guarding it.
    replay_root = tmp_path / "source" / "replay_buffer"
    replay_root.mkdir(parents=True)
    rollout = AWRRolloutConfig(
        deterministic_first=False,
        hdp_trajectory_augmentation=True,
        hdp_trajectory_augmentation_ramp_steps=20,
        hdp_trajectory_augmentation_heading_mode="preserve",
    )
    reward = RewardConfig(reward_profile="hdp_pdm")
    (replay_root.parent / "effective_config.json").write_text(
        json.dumps(
            {
                "awr": asdict(rollout),
                "reward": asdict(reward),
            }
        )
    )
    (replay_root.parent / "provenance.json").write_text(
        json.dumps({"neighbor_future_alignment_offset": 1})
    )
    monkeypatch.setenv("DP_NEIGHBOR_FUTURE_OFFSET", "1")
    # A matching cache must stay quiet, or "it warned" carries no information.
    assert _contract_drift(capsys, replay_root, rollout, reward) == ""

    source = json.loads((replay_root.parent / "effective_config.json").read_text())
    source["awr"].pop("plannerrft_reference_mode")
    (replay_root.parent / "effective_config.json").write_text(json.dumps(source))
    assert "plannerrft_reference_mode" in _contract_drift(
        capsys, replay_root, rollout, reward
    )
    # ...unless the caller declares the default the old cache predates.
    assert (
        _contract_drift(
            capsys,
            replay_root,
            rollout,
            reward,
            allow_missing_defaults={
                "awr": {"plannerrft_reference_mode": "zero_noise"}
            },
        )
        == ""
    )
    source["awr"]["plannerrft_reference_mode"] = "zero_noise"
    (replay_root.parent / "effective_config.json").write_text(json.dumps(source))

    # beta rescales every replay weight, so it is the change that must never
    # pass unmentioned: the arrays on disk were produced under the old value.
    changed = replace(rollout, beta=2.0)
    assert "awr.beta" in _contract_drift(capsys, replay_root, changed, reward)


def _reward(total: float, **kwargs) -> RewardBreakdown:
    return RewardBreakdown(
        safety=0.0,
        progress=0.0,
        smoothness=0.0,
        feasibility=0.0,
        centerline=0.0,
        red_light=0.0,
        total=total,
        collision_step=kwargs.get("collision_step"),
        off_road_fraction=0.0,
        rb_crossing=kwargs.get("rb_crossing", False),
        kinematic_violated=kwargs.get("kinematic_violated", False),
    )


def test_all_zero_reward_group_is_distinguished_from_equal_nonzero_group() -> None:
    zero = {}
    _add_rollout_reward_diagnostics(
        zero, [_reward(0.0, collision_step=3), _reward(0.0)]
    )
    assert zero["all_zero_reward_group"] == 1.0
    assert zero["all_equal_reward_group"] == 1.0
    assert zero["zero_reward_candidate_fraction"] == 1.0
    assert zero["det_zero_collision"] == 1.0

    tied = {}
    _add_rollout_reward_diagnostics(tied, [_reward(0.7), _reward(0.7)])
    assert tied["all_zero_reward_group"] == 0.0
    assert tied["all_equal_reward_group"] == 1.0
    assert tied["zero_reward_candidate_fraction"] == 0.0


def test_reward_group_reports_useful_rank_signal() -> None:
    diagnostics = {}
    _add_rollout_reward_diagnostics(
        diagnostics, [_reward(0.0), _reward(0.4), _reward(0.8)]
    )
    assert diagnostics["all_equal_reward_group"] == 0.0
    assert diagnostics["reward_unique_count"] == 3.0
    assert diagnostics["best_vs_det_reward_gain"] == pytest.approx(0.8)
    assert diagnostics["zero_reward_candidate_fraction"] == pytest.approx(1.0 / 3.0)


def test_positive_awr_keeps_behavior_and_only_real_improvements() -> None:
    weights, diagnostics = compute_awr_weights(
        [_reward(0.90), _reward(0.905), _reward(0.92), _reward(0.80)],
        normalize=False,
        positive_advantage_only=True,
        positive_advantage_margin=0.01,
        behavior_anchor_weight=1.0,
    )
    assert weights[0].item() == pytest.approx(1.0)
    assert weights[1].item() == 0.0
    assert weights[2].item() > 0.0
    assert weights[3].item() == 0.0
    assert diagnostics["positive_candidate_count"] == 1.0
    assert diagnostics["has_positive_candidate"] == 1.0
    assert diagnostics["active_weight_count"] == 2.0
    assert 0.5 <= diagnostics["effective_sample_size_fraction"] <= 1.0
    assert diagnostics["behavior_anchor_active"] == 1.0


def test_group_relative_metrics_do_not_claim_a_behavior_anchor() -> None:
    weights, diagnostics = compute_awr_weights(
        [_reward(0.90), _reward(0.92), _reward(0.80)],
        normalize=False,
        positive_advantage_only=False,
    )
    assert torch.all(weights > 0)
    assert "positive_candidate_count" not in diagnostics
    assert "has_positive_candidate" not in diagnostics
    assert "behavior_anchor_weight" not in diagnostics
    assert "behavior_anchor_active" not in diagnostics


def test_zero_reward_weight_share_reports_weight_not_candidate_count() -> None:
    weights, diagnostics = compute_awr_weights(
        [_reward(0.0), _reward(0.9), _reward(0.0)],
        normalize=False,
        positive_advantage_only=False,
    )
    expected = float((weights[0] + weights[2]) / weights.sum())
    assert diagnostics["zero_reward_weight_share"] == pytest.approx(expected)
    assert 0.0 < expected < 2.0 / 3.0

    _, tied = compute_awr_weights(
        [_reward(0.0), _reward(0.0)], normalize=False
    )
    assert tied["zero_reward_weight_share"] == 1.0


def test_train_wandb_labels_stochastic_candidate_zero_honestly() -> None:
    aliases = _wandb_readable_aliases("train", "mean_det_reward", 0.87)
    assert aliases == {
        "train_primary/awr_signal/candidate0_reward_pct": pytest.approx(87.0)
    }
    eval_aliases = _wandb_readable_aliases("eval", "mean_det_reward", 0.93)
    assert "eval_primary/reward/reward_pct" in eval_aliases
    zero_weight_alias = _wandb_readable_aliases(
        "train", "mean_zero_reward_weight_share", 0.021
    )
    assert zero_weight_alias == {
        "train_primary/awr_signal/zero_reward_weight_share_pct": pytest.approx(2.1)
    }


def test_disk_replay_hides_anchor_metrics_for_all_stochastic_profile(
    tmp_path,
) -> None:
    rank_dir = tmp_path / "rank_0000"
    rank_dir.mkdir()
    arrays = {
        "trajectories": np.zeros((2, 3, 4, 4), dtype=np.float32),
        "noise_scales": np.ones((2, 3), dtype=np.float32),
        "weights": np.asarray([[1.0, 2.0, 0.5], [1.0, 1.0, 1.0]], dtype=np.float32),
        "rewards": np.asarray([[0.8, 0.9, 0.7], [0.0, 0.0, 0.0]], dtype=np.float32),
        "scene_encoding": np.zeros((2, 2, 3), dtype=np.float32),
    }
    for name, values in arrays.items():
        np.save(rank_dir / f"{name}.npy", values)
    (rank_dir / "scene_paths.jsonl").write_text('"a.npz"\n"b.npz"\n')
    (rank_dir / "manifest.json").write_text(
        json.dumps(
            {
                "scene_count": 2,
                "group_size": 3,
                "future_len": 4,
                "arrays": {name: f"{name}.npy" for name in arrays},
                "paths": "scene_paths.jsonl",
            }
        )
    )

    reader = _DiskReplayReader(
        tmp_path,
        0,
        positive_advantage_only=False,
        deterministic_first=False,
    )
    _, rollout = reader.batch_from_indices([0, 1])
    reader.close()
    assert rollout.diagnostics["candidate0_deterministic"] == 0.0
    assert rollout.diagnostics["zero_reward_weight_share"] == pytest.approx(0.5)
    assert "has_positive_candidate" not in rollout.diagnostics
    assert "behavior_anchor_active" not in rollout.diagnostics


def test_disk_replay_sampling_diagnostic_distinguishes_shuffle(tmp_path) -> None:
    rank_dir = tmp_path / "rank_0000"
    rank_dir.mkdir()
    arrays = {
        "trajectories": np.zeros((2, 3, 4, 4), dtype=np.float32),
        "noise_scales": np.ones((2, 3), dtype=np.float32),
        "weights": np.ones((2, 3), dtype=np.float32),
        "rewards": np.asarray(
            [[0.8, 0.9, 0.7], [0.95, 0.96, 0.94]], dtype=np.float32
        ),
        "scene_encoding": np.zeros((2, 2, 3), dtype=np.float32),
    }
    for name, values in arrays.items():
        np.save(rank_dir / f"{name}.npy", values)
    (rank_dir / "scene_paths.jsonl").write_text('"a.npz"\n"b.npz"\n')
    (rank_dir / "manifest.json").write_text(
        json.dumps(
            {
                "scene_count": 2,
                "group_size": 3,
                "future_len": 4,
                "arrays": {name: f"{name}.npy" for name in arrays},
                "paths": "scene_paths.jsonl",
            }
        )
    )

    reader = _DiskReplayReader(tmp_path, 0)
    _, replay = reader.batch_from_indices(
        [0, 1], sampled_with_replacement=False
    )
    reader.close()
    assert replay.diagnostics["replay_sampled_with_replacement"] == 0.0
    assert replay.diagnostics["replay_priority_enabled"] == 0.0


def test_reward_stratified_replay_uses_exact_hard_fraction(tmp_path) -> None:
    rank_dir = tmp_path / "rank_0000"
    rank_dir.mkdir()
    rewards = np.asarray(
        [
            [0.70, 0.80, 0.75],
            [0.85, 0.90, 0.88],
            [0.93, 0.95, 0.94],
            [0.97, 0.99, 0.98],
            [0.00, 0.00, 0.00],
            [0.60, 0.70, 0.65],
        ],
        dtype=np.float32,
    )
    weights = np.ones_like(rewards)
    # Tied/invalid groups with no AWR target must not consume the hard quota.
    weights[4:] = 0.0
    arrays = {
        "trajectories": np.zeros((6, 3, 4, 4), dtype=np.float32),
        "noise_scales": np.ones((6, 3), dtype=np.float32),
        "weights": weights,
        "rewards": rewards,
        "scene_encoding": np.zeros((6, 2, 3), dtype=np.float32),
    }
    for name, values in arrays.items():
        np.save(rank_dir / f"{name}.npy", values)
    paths = ["hard0.npz", "hard1.npz", "easy0.npz", "easy1.npz", "zero.npz", "inactive.npz"]
    (rank_dir / "scene_paths.jsonl").write_text(
        "".join(json.dumps(path) + "\n" for path in paths)
    )
    (rank_dir / "manifest.json").write_text(
        json.dumps(
            {
                "scene_count": 6,
                "group_size": 3,
                "future_len": 4,
                "arrays": {name: f"{name}.npy" for name in arrays},
                "paths": "scene_paths.jsonl",
            }
        )
    )

    reader = _DiskReplayReader(tmp_path, 0)
    batches = list(
        reader.iter_reward_stratified_batches(
            batch_size=6,
            num_batches=5,
            seed=3407,
            priority_best_reward_lt=0.925,
            priority_fraction=0.5,
        )
    )
    reader.close()
    assert len(batches) == 5
    hard_paths = {"hard0.npz", "hard1.npz"}
    for scene_paths, replay in batches:
        assert sum(path in hard_paths for path in scene_paths) == 3
        assert replay.diagnostics["replay_sampled_with_replacement"] == 1.0
        assert replay.diagnostics["replay_priority_enabled"] == 1.0
        assert replay.diagnostics["replay_priority_fraction"] == pytest.approx(0.5)
        assert replay.diagnostics["replay_priority_best_reward_lt"] == pytest.approx(
            0.925
        )


def test_compact_replay_context_keeps_canonical_aligned_neighbor_slice() -> None:
    data = {
        "ego_current_state": torch.arange(20, dtype=torch.float32).reshape(2, 10),
        "neighbor_agents_past": torch.arange(
            2 * 5 * 3 * 11, dtype=torch.float32
        ).reshape(2, 5, 3, 11),
        "neighbor_agents_future": torch.arange(
            2 * 5 * 4 * 3, dtype=torch.float32
        ).reshape(2, 5, 4, 3),
    }
    context = _compact_replay_decoder_context(data, predicted_neighbor_num=2)
    assert torch.equal(context["ego_current_state"], data["ego_current_state"])
    assert context["neighbor_agents_past"].shape == (2, 2, 1, 11)
    assert torch.equal(
        context["neighbor_agents_past"],
        data["neighbor_agents_past"][:, :2, -1:],
    )
    assert context["neighbor_agents_future"].shape == (2, 2, 4, 3)
    assert torch.equal(
        context["neighbor_agents_future"],
        data["neighbor_agents_future"][:, :2],
    )


def test_compact_replay_context_is_loss_input_equivalent_to_legacy_npz_view(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DP_NEIGHBOR_FUTURE_OFFSET", "1")
    batch_size, neighbor_count, future_len = 2, 2, 4
    raw = {
        "ego_current_state": torch.arange(20, dtype=torch.float32).reshape(2, 10),
        "neighbor_agents_past": torch.arange(
            batch_size * 5 * 3 * 11, dtype=torch.float32
        ).reshape(batch_size, 5, 3, 11),
        "neighbor_agents_future": torch.arange(
            batch_size * 5 * future_len * 3, dtype=torch.float32
        ).reshape(batch_size, 5, future_len, 3),
    }
    compact = _compact_replay_decoder_context(raw, neighbor_count)
    legacy = {
        "ego_current_state": raw["ego_current_state"],
        "neighbor_agents_past": raw["neighbor_agents_past"][:, :, -1:],
        "neighbor_agents_future": torch.zeros_like(raw["neighbor_agents_future"]),
    }
    legacy["neighbor_agents_future"] = raw["neighbor_agents_future"]

    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inputs = []

        def forward(self, inputs):
            self.inputs.append(
                {
                    key: inputs[key].detach().clone()
                    for key in ("gt_trajectories", "sampled_trajectories")
                }
            )
            return None, {"model_output": torch.zeros_like(inputs["gt_trajectories"])}

    model_args = SimpleNamespace(
        predicted_neighbor_num=neighbor_count,
        future_len=future_len,
        state_normalizer=SimpleNamespace(
            mean=torch.zeros(1, 4), std=torch.ones(1, 4)
        ),
        observation_normalizer=lambda value: value,
        ego_prediction_horizon=future_len,
        coeff_velocity=1.0,
        coeff_timestep=[1.0],
    )
    targets = torch.arange(
        batch_size * future_len * 4, dtype=torch.float32
    ).reshape(batch_size, future_len, 4)
    noise = torch.linspace(
        -1.0,
        1.0,
        batch_size * (neighbor_count + 1) * future_len * 4,
    ).reshape(batch_size, neighbor_count + 1, future_len, 4)
    t = torch.tensor([0.2, 0.8])
    encoding = torch.zeros(batch_size, 2, 3)
    model = RecordingModel()
    legacy_loss = compute_batched_trajectory_losses(
        model,
        legacy,
        targets,
        model_args,
        noise,
        t,
        torch.device("cpu"),
        cached_encoding=encoding,
        awr_loss_type="plain_mse",
        use_prefix_mask=False,
    )
    compact_loss = compute_batched_trajectory_losses(
        model,
        compact,
        targets,
        model_args,
        noise,
        t,
        torch.device("cpu"),
        cached_encoding=encoding,
        awr_loss_type="plain_mse",
        use_prefix_mask=False,
    )
    assert torch.equal(legacy_loss, compact_loss)
    for key in model.inputs[0]:
        assert torch.equal(model.inputs[0][key], model.inputs[1][key])


def test_disk_replay_roundtrips_cached_decoder_context_without_npz_reload(
    tmp_path,
    monkeypatch,
) -> None:
    context = {
        "ego_current_state": torch.arange(20, dtype=torch.float32).reshape(2, 10),
        "neighbor_agents_past": torch.arange(
            2 * 2 * 1 * 11, dtype=torch.float32
        ).reshape(2, 2, 1, 11),
        "neighbor_agents_future": torch.arange(
            2 * 2 * 4 * 3, dtype=torch.float32
        ).reshape(2, 2, 4, 3),
    }
    source_encoding = torch.linspace(-1.0, 1.0, 12).reshape(2, 2, 3)
    rollout = AWRRollout(
        trajectories=torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(
            6, 4, 4
        ),
        noise_scales=torch.ones(6),
        rewards=[_reward(float(index) / 10.0) for index in range(6)],
        weights=torch.arange(1, 7, dtype=torch.float32),
        reward_data=context,
        scene_encoding=source_encoding,
        diagnostics={"valid_group": 1.0},
    )
    writer = _DiskReplayWriter(
        tmp_path,
        rank=0,
        expected_count=2,
        group_size=3,
        future_len=4,
        cache_decoder_context=True,
    )
    writer.append(["a.npz", "b.npz"], rollout, group_size=3)
    manifest = json.loads(writer.close().read_text())
    assert manifest["version"] == 3
    assert manifest["decoder_context_schema"] == "canonical_loader_aligned_v1"
    assert manifest["decoder_context_cached"] is True

    manifest_path = tmp_path / "rank_0000" / "manifest.json"
    unsafe_manifest = dict(manifest)
    unsafe_manifest.pop("decoder_context_schema")
    manifest_path.write_text(json.dumps(unsafe_manifest))
    with pytest.raises(RuntimeError, match="unsafe/unknown alignment schema"):
        _DiskReplayReader(tmp_path, 0)
    manifest_path.write_text(json.dumps(manifest))

    reader = _DiskReplayReader(tmp_path, 0)
    scenes, replay = reader.batch_from_indices([1, 0])
    assert scenes == ["b.npz", "a.npz"]
    assert replay.diagnostics["replay_decoder_context_cached"] == 1.0
    assert replay.scene_encoding.dtype == torch.float32
    assert torch.equal(replay.scene_encoding, source_encoding[[1, 0]])
    for key, value in context.items():
        assert torch.equal(replay.reward_data[key], value[[1, 0]])

    def forbidden_npz_reload(*args, **kwargs):
        del args, kwargs
        raise AssertionError("cached replay context must not reopen an NPZ")

    monkeypatch.setattr(train_awr_module, "_load_scene_batch", forbidden_npz_reload)
    prepared = list(
        _iter_prefetched_replay_batches(
            iter([(scenes, replay)]),
            scene_load_workers=1,
            prefetch_batches=0,
            replay_context_only=True,
        )
    )
    assert len(prepared) == 1
    prepared_scenes, prepared_rollout, prepared_data = prepared[0]
    assert prepared_scenes == scenes
    assert prepared_rollout.reward_data == {}
    for key, value in replay.reward_data.items():
        assert torch.equal(prepared_data[key], value)
    reader.close()


def test_disk_replay_roundtrips_expert_anchor_sidecar_without_npz_reload(
    tmp_path,
    monkeypatch,
) -> None:
    context = {
        "ego_current_state": torch.arange(20, dtype=torch.float32).reshape(2, 10),
        "neighbor_agents_past": torch.arange(
            2 * 2 * 1 * 11, dtype=torch.float32
        ).reshape(2, 2, 1, 11),
        "neighbor_agents_future": torch.arange(
            2 * 2 * 4 * 3, dtype=torch.float32
        ).reshape(2, 2, 4, 3),
    }
    experts = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4)
    expert_rewards = torch.tensor([0.95, 0.0], dtype=torch.float32)
    expert_safe = torch.tensor([1.0, 0.0], dtype=torch.float32)
    reward_data = {
        **context,
        train_awr_module._REPLAY_EXPERT_TRAJECTORY_KEY: experts,
        train_awr_module._REPLAY_EXPERT_REWARD_KEY: expert_rewards,
        train_awr_module._REPLAY_EXPERT_SAFE_KEY: expert_safe,
    }
    rollout = AWRRollout(
        trajectories=torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(
            6, 4, 4
        ),
        noise_scales=torch.ones(6),
        rewards=[_reward(float(index) / 10.0) for index in range(6)],
        weights=torch.arange(1, 7, dtype=torch.float32),
        reward_data=reward_data,
        scene_encoding=torch.linspace(-1.0, 1.0, 12).reshape(2, 2, 3),
        diagnostics={"valid_group": 1.0},
    )
    writer = _DiskReplayWriter(
        tmp_path,
        rank=0,
        expected_count=2,
        group_size=3,
        future_len=4,
        cache_decoder_context=True,
        cache_expert_anchor=True,
    )
    writer.append(["a.npz", "b.npz"], rollout, group_size=3)
    writer.close()

    sidecar = json.loads(
        (tmp_path / "rank_0000" / "expert_anchor_manifest.json").read_text()
    )
    assert sidecar["schema"] == "logged_ego_future_hard_gate_v1"
    reader = _DiskReplayReader(tmp_path, 0)
    scenes, replay = reader.batch_from_indices([1, 0])
    assert scenes == ["b.npz", "a.npz"]
    assert replay.diagnostics["replay_expert_anchor_cached"] == 1.0
    assert torch.equal(
        replay.reward_data[train_awr_module._REPLAY_EXPERT_TRAJECTORY_KEY],
        experts[[1, 0]],
    )
    assert torch.equal(
        replay.reward_data[train_awr_module._REPLAY_EXPERT_REWARD_KEY],
        expert_rewards[[1, 0]],
    )
    assert torch.equal(
        replay.reward_data[train_awr_module._REPLAY_EXPERT_SAFE_KEY],
        expert_safe[[1, 0]],
    )

    def forbidden_npz_reload(*args, **kwargs):
        del args, kwargs
        raise AssertionError("cached expert/context replay must not reopen an NPZ")

    monkeypatch.setattr(train_awr_module, "_load_scene_batch", forbidden_npz_reload)
    prepared = list(
        _iter_prefetched_replay_batches(
            iter([(scenes, replay)]),
            scene_load_workers=1,
            prefetch_batches=0,
            replay_context_only=True,
        )
    )
    assert len(prepared) == 1
    _, prepared_rollout, prepared_data = prepared[0]
    assert prepared_rollout.reward_data == {}
    assert torch.equal(
        prepared_data[train_awr_module._REPLAY_EXPERT_TRAJECTORY_KEY],
        experts[[1, 0]],
    )
    reader.close()


def test_decoder_context_attachment_preserves_candidates_and_enables_cached_reader(
    tmp_path,
) -> None:
    replay_root = tmp_path / "source_run" / "replay_buffer"
    context = {
        "ego_current_state": np.arange(20, dtype=np.float32).reshape(2, 10),
        "neighbor_agents_past": np.arange(
            2 * 2 * 1 * 11, dtype=np.float32
        ).reshape(2, 2, 1, 11),
        "neighbor_agents_future": np.arange(
            2 * 2 * 4 * 3, dtype=np.float32
        ).reshape(2, 2, 4, 3),
    }
    experts = torch.zeros(2, 4, 4)
    trajectories = torch.arange(
        2 * 3 * 4 * 4, dtype=torch.float32
    ).reshape(6, 4, 4)
    writer = _DiskReplayWriter(
        replay_root,
        rank=0,
        expected_count=2,
        group_size=3,
        future_len=4,
        cache_expert_anchor=True,
        expected_scene_paths=["a.npz", "b.npz"],
    )
    writer.append(
        ["a.npz", "b.npz"],
        AWRRollout(
            trajectories=trajectories,
            noise_scales=torch.ones(6),
            rewards=[_reward(0.5) for _ in range(6)],
            weights=torch.ones(6),
            reward_data={
                train_awr_module._REPLAY_EXPERT_TRAJECTORY_KEY: experts,
                train_awr_module._REPLAY_EXPERT_REWARD_KEY: torch.ones(2),
                train_awr_module._REPLAY_EXPERT_SAFE_KEY: torch.ones(2),
            },
            scene_encoding=torch.zeros(2, 2, 3),
            diagnostics={"valid_group": 1.0},
        ),
        group_size=3,
    )
    writer.close()
    rank_dir = replay_root / "rank_0000"
    expected_path = rank_dir / "expected_scene_paths.jsonl"
    expected_sha = hashlib.sha256(expected_path.read_bytes()).hexdigest()
    for key, value in context.items():
        np.save(rank_dir / f"context_{key}.npy", value)
    (rank_dir / "context_build_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "schema": "canonical_loader_aligned_v1",
                "scene_count": 2,
                "future_len": 4,
                "neighbor_future_alignment_offset": 1,
                "x2_legacy_ego_width_m": 2.29156,
                "xx1_legacy_ego_width_m": 1.70,
                "expected_paths_sha256": expected_sha,
                "arrays": {
                    key: f"context_{key}.npy" for key in context
                },
                "shapes": {
                    key: list(value.shape) for key, value in context.items()
                },
            }
        )
    )
    before_sha = _file_sha256(rank_dir / "trajectories.npy")
    payload = attach_decoder_context(replay_root, expected_world_size=1)
    assert payload["candidate_arrays_unchanged"] is True
    assert _file_sha256(rank_dir / "trajectories.npy") == before_sha
    manifest = json.loads((rank_dir / "manifest.json").read_text())
    assert manifest["decoder_context_cached"] is True
    assert manifest["decoder_context_schema"] == "canonical_loader_aligned_v1"

    reader = _DiskReplayReader(replay_root, 0)
    scenes, replay = reader.batch_from_indices([1, 0])
    reader.close()
    assert scenes == ["b.npz", "a.npz"]
    assert replay.diagnostics["replay_decoder_context_cached"] == 1.0
    for key, value in context.items():
        assert torch.equal(replay.reward_data[key], torch.from_numpy(value[[1, 0]]))


def test_resumable_decoder_context_builder_commits_canonical_arrays(
    tmp_path,
    monkeypatch,
) -> None:
    replay_root = tmp_path / "replay"
    rank_dir = replay_root / "rank_0000"
    rank_dir.mkdir(parents=True)
    paths = ["scene_0", "scene_1", "scene_2"]
    expected_path = rank_dir / "expected_scene_paths.jsonl"
    expected_path.write_text("".join(json.dumps(path) + "\n" for path in paths))
    (rank_dir / "writer_contract.json").write_text(
        json.dumps(
            {
                "expected_count": len(paths),
                "expected_paths": expected_path.name,
            }
        )
    )
    args_path = tmp_path / "args.json"
    args_path.write_text('{"future_len": 4, "predicted_neighbor_num": 2}')

    def fake_load(
        scene_paths,
        device,
        workers=1,
        replay_context_only=False,
        defer_full_scene_heading_transforms=False,
    ):
        del device, workers, defer_full_scene_heading_transforms
        assert replay_context_only is True
        index = torch.tensor(
            [int(path.rsplit("_", 1)[-1]) for path in scene_paths],
            dtype=torch.float32,
        )
        count = len(scene_paths)
        return {
            "ego_current_state": index[:, None].expand(count, 10).clone(),
            "neighbor_agents_past": index[:, None, None, None]
            .expand(count, 2, 1, 11)
            .clone(),
            "neighbor_agents_future": index[:, None, None, None]
            .expand(count, 2, 4, 3)
            .clone(),
        }

    monkeypatch.setattr(build_context_module, "_load_scene_batch", fake_load)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("DP_NEIGHBOR_FUTURE_OFFSET", "1")
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_replay_decoder_context.py",
            "--replay-root",
            str(replay_root),
            "--args-path",
            str(args_path),
            "--batch-size",
            "2",
            "--flush-interval-batches",
            "1",
        ],
    )
    build_context_module.main()
    manifest = json.loads((rank_dir / "context_build_manifest.json").read_text())
    assert manifest["scene_count"] == 3
    assert manifest["schema"] == "canonical_loader_aligned_v1"
    for key in train_awr_module._REPLAY_CONTEXT_KEYS:
        array = np.load(rank_dir / manifest["arrays"][key])
        assert array.shape[0] == 3
        assert array.dtype == np.float32
        assert array[2].mean() == pytest.approx(2.0)
    # A second invocation validates the committed generation and does no work.
    build_context_module.main()


def test_disk_replay_reuses_shared_encoding_by_symlink_without_copy(
    tmp_path,
) -> None:
    shared_path = tmp_path / "immutable_scene_encoding.npy"
    shared = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)
    np.save(shared_path, shared)
    rollout = AWRRollout(
        trajectories=torch.zeros(2 * 3, 4, 4),
        noise_scales=torch.ones(2 * 3),
        rewards=[_reward(0.5) for _ in range(2 * 3)],
        weights=torch.ones(2 * 3),
        reward_data={},
        scene_encoding=torch.from_numpy(shared.copy()),
        diagnostics={"valid_group": 1.0},
    )
    replay_root = tmp_path / "replay"
    writer = _DiskReplayWriter(
        replay_root,
        rank=0,
        expected_count=2,
        group_size=3,
        future_len=4,
        shared_scene_encoding_path=shared_path,
        expected_scene_paths=["a.npz", "b.npz"],
    )
    writer.append(["a.npz", "b.npz"], rollout, group_size=3)
    manifest = json.loads(writer.close().read_text())
    link = replay_root / "rank_0000" / "scene_encoding.npy"
    assert link.is_symlink()
    assert link.resolve() == shared_path.resolve()
    assert manifest["shared_scene_encoding_path"] == str(shared_path.resolve())

    reader = _DiskReplayReader(replay_root, 0)
    scenes, replay = reader.batch_from_indices([1, 0])
    reader.close()
    assert scenes == ["b.npz", "a.npz"]
    assert torch.equal(replay.scene_encoding, torch.from_numpy(shared[[1, 0]]))


def test_disk_replay_reuses_a_strict_prefix_of_longer_shared_encoding(
    tmp_path,
) -> None:
    shared_path = tmp_path / "longer_immutable_scene_encoding.npy"
    shared = np.arange(3 * 2 * 3, dtype=np.float32).reshape(3, 2, 3)
    np.save(shared_path, shared)
    source_bytes = shared_path.read_bytes()
    rollout = AWRRollout(
        trajectories=torch.zeros(2 * 3, 4, 4),
        noise_scales=torch.ones(2 * 3),
        rewards=[_reward(0.5) for _ in range(2 * 3)],
        weights=torch.ones(2 * 3),
        reward_data={},
        scene_encoding=torch.from_numpy(shared[:2].copy()),
        diagnostics={"valid_group": 1.0},
    )
    replay_root = tmp_path / "prefix_replay"
    writer = _DiskReplayWriter(
        replay_root,
        rank=0,
        expected_count=2,
        group_size=3,
        future_len=4,
        shared_scene_encoding_path=shared_path,
        expected_scene_paths=["a.npz", "b.npz"],
    )
    writer.append(["a.npz", "b.npz"], rollout, group_size=3)
    manifest = json.loads(writer.close().read_text())
    assert manifest["scene_count"] == 2
    assert manifest["shared_scene_encoding_source_count"] == 3
    assert manifest["shared_scene_encoding_prefix_count"] == 2
    assert shared_path.read_bytes() == source_bytes

    reader = _DiskReplayReader(replay_root, 0)
    scenes, replay = reader.batch_from_indices([1, 0])
    reader.close()
    assert scenes == ["b.npz", "a.npz"]
    assert torch.equal(replay.scene_encoding, torch.from_numpy(shared[[1, 0]]))


def test_disk_replay_rejects_shorter_shared_encoding(tmp_path) -> None:
    shared_path = tmp_path / "short_scene_encoding.npy"
    np.save(shared_path, np.zeros((1, 2, 3), dtype=np.float32))
    with pytest.raises(RuntimeError, match="shorter than replay order"):
        _DiskReplayWriter(
            tmp_path / "short_replay",
            rank=0,
            expected_count=2,
            group_size=3,
            future_len=4,
            shared_scene_encoding_path=shared_path,
            expected_scene_paths=["a.npz", "b.npz"],
        )


def test_disk_replay_reuses_shared_decoder_context_read_only_by_symlink(
    tmp_path,
) -> None:
    context_root = tmp_path / "immutable_context"
    context_root.mkdir()
    source_context = {
        "ego_current_state": np.arange(2 * 10, dtype=np.float32).reshape(2, 10),
        "neighbor_agents_past": np.arange(
            2 * 2 * 1 * 11, dtype=np.float32
        ).reshape(2, 2, 1, 11),
        "neighbor_agents_future": np.arange(
            2 * 2 * 4 * 3, dtype=np.float32
        ).reshape(2, 2, 4, 3),
    }
    source_paths = {}
    source_hashes = {}
    for key, value in source_context.items():
        path = context_root / f"{key}.npy"
        np.save(path, value)
        source_paths[key] = path
        source_hashes[key] = _file_sha256(path)

    rollout = AWRRollout(
        trajectories=torch.zeros(2 * 3, 4, 4),
        noise_scales=torch.ones(2 * 3),
        rewards=[_reward(0.5) for _ in range(2 * 3)],
        weights=torch.ones(2 * 3),
        reward_data={
            key: torch.from_numpy(value.copy()) + 10_000.0
            for key, value in source_context.items()
        },
        scene_encoding=torch.zeros(2, 2, 3),
        diagnostics={"valid_group": 1.0},
    )
    replay_root = tmp_path / "replay"
    writer = _DiskReplayWriter(
        replay_root,
        rank=0,
        expected_count=2,
        group_size=3,
        future_len=4,
        cache_decoder_context=True,
        shared_decoder_context_paths=source_paths,
        expected_scene_paths=["a.npz", "b.npz"],
    )
    writer.append(["a.npz", "b.npz"], rollout, group_size=3)
    manifest = json.loads(writer.close().read_text())
    assert manifest["shared_decoder_context_paths"] == {
        key: str(path.resolve()) for key, path in source_paths.items()
    }
    for key, path in source_paths.items():
        link = replay_root / "rank_0000" / f"context_{key}.npy"
        assert link.is_symlink()
        assert link.resolve() == path.resolve()
        assert _file_sha256(path) == source_hashes[key]

    reader = _DiskReplayReader(replay_root, 0)
    scenes, replay = reader.batch_from_indices([1, 0])
    reader.close()
    assert scenes == ["b.npz", "a.npz"]
    for key, value in source_context.items():
        assert torch.equal(replay.reward_data[key], torch.from_numpy(value[[1, 0]]))


def test_disk_replay_reuses_canonical_shared_context_prefix(tmp_path) -> None:
    context_root = tmp_path / "immutable_context_prefix"
    context_root.mkdir()
    source_context = {
        "ego_current_state": np.arange(3 * 10, dtype=np.float32).reshape(3, 10),
        "neighbor_agents_past": np.arange(
            3 * 2 * 1 * 11, dtype=np.float32
        ).reshape(3, 2, 1, 11),
        "neighbor_agents_future": np.arange(
            3 * 2 * 4 * 3, dtype=np.float32
        ).reshape(3, 2, 4, 3),
    }
    source_paths = {}
    for key, value in source_context.items():
        path = context_root / f"{key}.npy"
        np.save(path, value)
        source_paths[key] = path

    writer = _DiskReplayWriter(
        tmp_path / "prefix_context_replay",
        rank=0,
        expected_count=2,
        group_size=3,
        future_len=4,
        cache_decoder_context=True,
        shared_decoder_context_paths=source_paths,
        expected_scene_paths=["a.npz", "b.npz"],
    )
    writer.append(
        ["a.npz", "b.npz"],
        AWRRollout(
            trajectories=torch.zeros(6, 4, 4),
            noise_scales=torch.ones(6),
            rewards=[_reward(0.5) for _ in range(6)],
            weights=torch.ones(6),
            reward_data={
                key: torch.from_numpy(value[:2].copy())
                for key, value in source_context.items()
            },
            scene_encoding=torch.zeros(2, 2, 3),
            diagnostics={"valid_group": 1.0},
        ),
        group_size=3,
    )
    manifest = json.loads(writer.close().read_text())
    assert manifest["shared_decoder_context_source_counts"] == {
        key: 3 for key in source_context
    }
    assert manifest["shared_decoder_context_prefix_count"] == 2

    reader = _DiskReplayReader(tmp_path / "prefix_context_replay", 0)
    scenes, replay = reader.batch_from_indices([1, 0])
    reader.close()
    assert scenes == ["b.npz", "a.npz"]
    for key, value in source_context.items():
        assert torch.equal(replay.reward_data[key], torch.from_numpy(value[[1, 0]]))


def test_shared_decoder_context_contract_binds_loader_semantics_and_shapes(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("DP_NEIGHBOR_FUTURE_OFFSET", "1")
    source_run = tmp_path / "source_run"
    replay_root = source_run / "replay_buffer"
    args_path = source_run / "args.json"
    source_run.mkdir()
    args_path.write_text('{"predicted_neighbor_num": 2}')
    source_run.joinpath("provenance.json").write_text(
        json.dumps(
            {
                "staged_args": str(args_path),
                "neighbor_future_alignment_offset": 1,
                "x2_legacy_ego_width_m": 2.29156,
                "xx1_legacy_ego_width_m": 1.70,
            }
        )
    )
    context = {
        "ego_current_state": torch.zeros(2, 10),
        "neighbor_agents_past": torch.zeros(2, 2, 1, 11),
        "neighbor_agents_future": torch.zeros(2, 2, 4, 3),
    }
    writer = _DiskReplayWriter(
        replay_root,
        rank=0,
        expected_count=2,
        group_size=2,
        future_len=4,
        cache_decoder_context=True,
        expected_scene_paths=["a.npz", "b.npz"],
    )
    writer.append(
        ["a.npz", "b.npz"],
        AWRRollout(
            trajectories=torch.zeros(4, 4, 4),
            noise_scales=torch.ones(4),
            rewards=[_reward(0.5) for _ in range(4)],
            weights=torch.ones(4),
            reward_data=context,
            scene_encoding=torch.zeros(2, 2, 3),
            diagnostics={"valid_group": 1.0},
        ),
        group_size=2,
    )
    writer.close()
    contract = _shared_decoder_context_contract(
        replay_root,
        rank=0,
        current_args=args_path,
        order_epoch=21,
    )
    assert contract["scene_count"] == 2
    assert contract["decoder_context_schema"] == "canonical_loader_aligned_v1"
    assert set(contract["context_paths"]) == set(context)
    assert contract["order_epoch"] == 21

    provenance_path = source_run / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["neighbor_future_alignment_offset"] = 0
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(RuntimeError, match="neighbor-future alignment differs"):
        _shared_decoder_context_contract(
            replay_root,
            rank=0,
            current_args=args_path,
            order_epoch=21,
        )


def test_shared_expert_anchor_contract_binds_data_reward_and_geometry(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("DP_NEIGHBOR_FUTURE_OFFSET", "1")
    source_run = tmp_path / "source_run"
    replay_root = source_run / "replay_buffer"
    args_path = source_run / "args.json"
    train_list = tmp_path / "train.json"
    source_run.mkdir()
    args_path.write_text('{"future_len": 4}')
    train_list.write_text('["a.npz", "b.npz"]')
    reward_config = RewardConfig()
    source_run.joinpath("provenance.json").write_text(
        json.dumps(
            {
                "staged_args": str(args_path),
                "train_npz_list": str(train_list),
                "skip_filtered_scenes": False,
                "neighbor_future_alignment_offset": 1,
                "x2_legacy_ego_width_m": 2.29156,
                "xx1_legacy_ego_width_m": 1.70,
            }
        )
    )
    source_run.joinpath("effective_config.json").write_text(
        json.dumps({"reward": asdict(reward_config)})
    )
    experts = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4)
    writer = _DiskReplayWriter(
        replay_root,
        rank=0,
        expected_count=2,
        group_size=2,
        future_len=4,
        cache_expert_anchor=True,
        expected_scene_paths=["a.npz", "b.npz"],
    )
    writer.append(
        ["a.npz", "b.npz"],
        AWRRollout(
            trajectories=torch.zeros(4, 4, 4),
            noise_scales=torch.ones(4),
            rewards=[_reward(0.5) for _ in range(4)],
            weights=torch.ones(4),
            reward_data={
                train_awr_module._REPLAY_EXPERT_TRAJECTORY_KEY: experts,
                train_awr_module._REPLAY_EXPERT_REWARD_KEY: torch.tensor(
                    [0.9, 0.0], dtype=torch.float32
                ),
                train_awr_module._REPLAY_EXPERT_SAFE_KEY: torch.tensor(
                    [1.0, 0.0], dtype=torch.float32
                ),
            },
            scene_encoding=torch.zeros(2, 2, 3),
            diagnostics={"valid_group": 1.0},
        ),
        group_size=2,
    )
    writer.close()

    contract = _shared_expert_anchor_contract(
        replay_root,
        rank=0,
        current_args=args_path,
        current_train_npz_list=train_list,
        current_skip_filtered_scenes=False,
        current_reward_config=reward_config,
        order_epoch=1,
    )
    assert contract["scene_count"] == 2
    assert contract["future_len"] == 4
    assert contract["expert_anchor_schema"] == "logged_ego_future_hard_gate_v1"
    assert set(contract["array_paths"]) == {
        "expert_trajectories",
        "expert_rewards",
        "expert_safe",
    }

    changed_reward = copy.copy(reward_config)
    changed_reward.rb_cross_thresh += 0.01
    with pytest.raises(RuntimeError, match="reward contract differs"):
        _shared_expert_anchor_contract(
            replay_root,
            rank=0,
            current_args=args_path,
            current_train_npz_list=train_list,
            current_skip_filtered_scenes=False,
            current_reward_config=changed_reward,
            order_epoch=1,
        )


def test_shared_encoding_partial_cache_can_be_salvaged_and_backfilled(
    tmp_path,
) -> None:
    shared_path = tmp_path / "immutable_scene_encoding.npy"
    shared = np.arange(3 * 2 * 3, dtype=np.float32).reshape(3, 2, 3)
    np.save(shared_path, shared)

    def make_rollout(start: int, count: int) -> AWRRollout:
        return AWRRollout(
            trajectories=torch.full((count * 2, 4, 4), float(start + 1)),
            noise_scales=torch.ones(count * 2),
            rewards=[_reward(0.5) for _ in range(count * 2)],
            weights=torch.ones(count * 2),
            reward_data={},
            scene_encoding=torch.from_numpy(shared[start : start + count].copy()),
            diagnostics={"valid_group": 1.0},
        )

    replay_root = tmp_path / "replay"
    writer = _DiskReplayWriter(
        replay_root,
        rank=0,
        expected_count=3,
        group_size=2,
        future_len=4,
        shared_scene_encoding_path=shared_path,
        expected_scene_paths=["a.npz", "b.npz", "c.npz"],
    )
    with pytest.raises(RuntimeError, match="canonical path prefix"):
        writer.append(["a.npz", "c.npz"], make_rollout(0, 2), group_size=2)
    writer.append(["a.npz", "b.npz"], make_rollout(0, 2), group_size=2)
    with pytest.raises(RuntimeError, match="rollout was incomplete"):
        writer.close()

    manifest = _salvage_partial_replay_manifest(replay_root, rank=0)
    assert manifest["scene_count"] == 2
    assert manifest["shared_scene_encoding_path"] == str(shared_path.resolve())
    resumed = _open_resume_writer(
        replay_root,
        rank=0,
        expected_count=3,
        expected_prefix=2,
        writable=True,
    )
    assert resumed.shared_scene_encoding_path == shared_path.resolve()
    resumed.append(["c.npz"], make_rollout(2, 1), group_size=2)
    completed = json.loads(resumed.close().read_text())
    assert completed["scene_count"] == 3
    assert completed["shared_scene_encoding_path"] == str(shared_path.resolve())


def test_expert_sidecar_partial_cache_can_be_salvaged_and_backfilled(
    tmp_path,
) -> None:
    def make_rollout(start: int, count: int) -> AWRRollout:
        expert = torch.arange(
            start * 4 * 4,
            (start + count) * 4 * 4,
            dtype=torch.float32,
        ).reshape(count, 4, 4)
        return AWRRollout(
            trajectories=torch.full((count * 2, 4, 4), float(start + 1)),
            noise_scales=torch.ones(count * 2),
            rewards=[_reward(0.5) for _ in range(count * 2)],
            weights=torch.ones(count * 2),
            reward_data={
                train_awr_module._REPLAY_EXPERT_TRAJECTORY_KEY: expert,
                train_awr_module._REPLAY_EXPERT_REWARD_KEY: torch.full(
                    (count,), 0.9, dtype=torch.float32
                ),
                train_awr_module._REPLAY_EXPERT_SAFE_KEY: torch.ones(
                    count, dtype=torch.float32
                ),
            },
            scene_encoding=torch.full((count, 2, 3), float(start + 1)),
            diagnostics={"valid_group": 1.0},
        )

    replay_root = tmp_path / "replay"
    writer = _DiskReplayWriter(
        replay_root,
        rank=0,
        expected_count=3,
        group_size=2,
        future_len=4,
        cache_expert_anchor=True,
        expected_scene_paths=["a.npz", "b.npz", "c.npz"],
    )
    writer.append(["a.npz", "b.npz"], make_rollout(0, 2), group_size=2)
    with pytest.raises(RuntimeError, match="rollout was incomplete"):
        writer.close()
    assert not (
        replay_root / "rank_0000" / "expert_anchor_manifest.json"
    ).exists()

    _salvage_partial_replay_manifest(replay_root, rank=0)
    resumed = _open_resume_writer(
        replay_root,
        rank=0,
        expected_count=3,
        expected_prefix=2,
        writable=True,
    )
    assert resumed.cache_expert_anchor is True
    resumed.append(["c.npz"], make_rollout(2, 1), group_size=2)
    resumed.close()

    reader = _DiskReplayReader(replay_root, 0)
    _, replay = reader.batch_from_indices([2, 0, 1])
    reader.close()
    assert replay.diagnostics["replay_expert_anchor_cached"] == 1.0
    expected = torch.cat(
        [make_rollout(index, 1).reward_data[
            train_awr_module._REPLAY_EXPERT_TRAJECTORY_KEY
        ] for index in (2, 0, 1)],
        dim=0,
    )
    assert torch.equal(
        replay.reward_data[train_awr_module._REPLAY_EXPERT_TRAJECTORY_KEY],
        expected,
    )


def test_partial_cache_accepts_only_manifested_parallel_decoder_context(
    tmp_path,
) -> None:
    replay_root = tmp_path / "replay"
    writer = _DiskReplayWriter(
        replay_root,
        rank=0,
        expected_count=3,
        group_size=2,
        future_len=4,
        cache_decoder_context=False,
        expected_scene_paths=["a.npz", "b.npz", "c.npz"],
    )
    rollout = AWRRollout(
        trajectories=torch.ones(4, 4, 4),
        noise_scales=torch.ones(4),
        rewards=[_reward(0.5) for _ in range(4)],
        weights=torch.ones(4),
        reward_data={},
        scene_encoding=torch.ones(2, 2, 3),
        diagnostics={"valid_group": 1.0},
    )
    writer.append(["a.npz", "b.npz"], rollout, group_size=2)
    with pytest.raises(RuntimeError, match="rollout was incomplete"):
        writer.close()

    rank_dir = replay_root / "rank_0000"
    context = {
        "ego_current_state": np.ones((3, 10), dtype=np.float32),
        "neighbor_agents_past": np.ones((3, 2, 1, 11), dtype=np.float32),
        "neighbor_agents_future": np.ones((3, 2, 4, 3), dtype=np.float32),
    }
    for key, value in context.items():
        np.save(rank_dir / f"context_{key}.npy", value)
    expected_paths = rank_dir / "expected_scene_paths.jsonl"
    context_manifest = {
        "version": 1,
        "schema": "canonical_loader_aligned_v1",
        "rank": 0,
        "world_size": 1,
        "scene_count": 3,
        "future_len": 4,
        "predicted_neighbor_num": 2,
        "neighbor_future_alignment_offset": 1,
        "x2_legacy_ego_width_m": 2.29156,
        "xx1_legacy_ego_width_m": 1.70,
        "expected_paths": str(expected_paths),
        "expected_paths_sha256": _file_sha256(expected_paths),
        "arrays": {key: f"context_{key}.npy" for key in context},
        "shapes": {key: list(value.shape) for key, value in context.items()},
    }
    (rank_dir / "context_build_manifest.json").write_text(
        json.dumps(context_manifest)
    )
    salvaged = _salvage_partial_replay_manifest(replay_root, rank=0)
    assert salvaged["scene_count"] == 2
    assert salvaged["decoder_context_cached"] is True
    assert salvaged["decoder_context_schema"] == "canonical_loader_aligned_v1"


def test_shared_context_partial_cache_backfill_never_writes_source(
    tmp_path,
) -> None:
    source_root = tmp_path / "immutable_context"
    source_root.mkdir()
    source_context = {
        "ego_current_state": np.arange(3 * 10, dtype=np.float32).reshape(3, 10),
        "neighbor_agents_past": np.arange(
            3 * 2 * 1 * 11, dtype=np.float32
        ).reshape(3, 2, 1, 11),
        "neighbor_agents_future": np.arange(
            3 * 2 * 4 * 3, dtype=np.float32
        ).reshape(3, 2, 4, 3),
    }
    source_paths = {}
    source_hashes = {}
    for key, value in source_context.items():
        path = source_root / f"{key}.npy"
        np.save(path, value)
        source_paths[key] = path
        source_hashes[key] = _file_sha256(path)

    def make_rollout(start: int, count: int) -> AWRRollout:
        return AWRRollout(
            trajectories=torch.full((count * 2, 4, 4), float(start + 1)),
            noise_scales=torch.ones(count * 2),
            rewards=[_reward(0.5) for _ in range(count * 2)],
            weights=torch.ones(count * 2),
            reward_data={
                key: torch.from_numpy(value[start : start + count].copy())
                + 50_000.0
                for key, value in source_context.items()
            },
            scene_encoding=torch.full((count, 2, 3), float(start + 1)),
            diagnostics={"valid_group": 1.0},
        )

    replay_root = tmp_path / "replay"
    writer = _DiskReplayWriter(
        replay_root,
        rank=0,
        expected_count=3,
        group_size=2,
        future_len=4,
        cache_decoder_context=True,
        shared_decoder_context_paths=source_paths,
        expected_scene_paths=["a.npz", "b.npz", "c.npz"],
    )
    writer.append(["a.npz", "b.npz"], make_rollout(0, 2), group_size=2)
    with pytest.raises(RuntimeError, match="rollout was incomplete"):
        writer.close()

    salvaged = _salvage_partial_replay_manifest(replay_root, rank=0)
    assert salvaged["scene_count"] == 2
    assert salvaged["shared_decoder_context_paths"] == {
        key: str(path.resolve()) for key, path in source_paths.items()
    }
    resumed = _open_resume_writer(
        replay_root,
        rank=0,
        expected_count=3,
        expected_prefix=2,
        writable=True,
    )
    assert resumed.shared_decoder_context_paths == {
        key: path.resolve() for key, path in source_paths.items()
    }
    resumed.append(["c.npz"], make_rollout(2, 1), group_size=2)
    completed = json.loads(resumed.close().read_text())
    assert completed["scene_count"] == 3
    for key, path in source_paths.items():
        assert _file_sha256(path) == source_hashes[key]

    reader = _DiskReplayReader(replay_root, 0)
    _, replay = reader.batch_from_indices([2, 1, 0])
    reader.close()
    for key, value in source_context.items():
        assert torch.equal(
            replay.reward_data[key], torch.from_numpy(value[[2, 1, 0]])
        )


def test_positive_awr_drops_unsafe_anchor_when_repair_exists() -> None:
    weights, diagnostics = compute_awr_weights(
        [_reward(0.0, rb_crossing=True), _reward(0.9), _reward(0.0)],
        normalize=False,
        positive_advantage_only=True,
        positive_advantage_margin=0.01,
        behavior_anchor_weight=1.0,
        unsafe_behavior_anchor_weight=0.0,
    )
    assert weights[0].item() == 0.0
    assert weights[1].item() > 0.0
    assert weights[2].item() == 0.0
    assert diagnostics["behavior_anchor_active"] == 0.0
    assert diagnostics["active_weight_count"] == 1.0
    assert diagnostics["effective_sample_size_fraction"] == pytest.approx(1.0)


def test_first_waypoint_rejected_candidates_receive_no_awr_weight() -> None:
    weights, diagnostics = compute_awr_weights(
        [_reward(0.8), _reward(0.99), _reward(0.7)],
        normalize=False,
        candidate_valid_mask=torch.tensor([True, False, True]),
    )
    assert weights[1].item() == 0.0
    assert weights[0].item() > 0.0
    assert weights[2].item() > 0.0
    assert diagnostics["first_waypoint_invalid_candidate_count"] == 1.0
    assert diagnostics["first_waypoint_invalid_candidate_fraction"] == pytest.approx(
        1.0 / 3.0
    )


def test_all_zero_reward_group_can_be_an_intentional_no_gradient_row() -> None:
    weights, diagnostics = compute_awr_weights(
        [_reward(0.0, rb_crossing=True), _reward(0.0, rb_crossing=True)],
        positive_advantage_only=True,
        drop_all_zero_groups=True,
    )
    assert weights.tolist() == [0.0, 0.0]
    assert diagnostics["valid_group"] == 1.0
    assert diagnostics["all_zero_reward_group"] == 1.0
    assert diagnostics["dropped_all_zero_group"] == 1.0


def test_dp_safe_augmentation_keeps_anchor_and_ramps_from_current_state() -> None:
    trajectories = torch.zeros(2, 40, 4)
    trajectories[..., 0] = torch.arange(1, 41) * 0.1
    trajectories[..., 2] = 1.0
    # Deliberately disagree with the finite-difference path tangent so this
    # catches accidental heading recomputation of the behavior anchor.
    trajectories[0, :, 2] = 0.0
    trajectories[0, :, 3] = 1.0
    torch.manual_seed(17)
    augmented = _apply_hdp_trajectory_augmentation(
        trajectories,
        std=0.5,
        deterministic_first=True,
        ramp_steps=20,
        heading_mode="preserve",
    )
    assert torch.equal(augmented[0], trajectories[0])
    offsets = (augmented[1, :, :2] - trajectories[1, :, :2]).norm(dim=-1)
    assert offsets[0] < offsets[19] * 0.01
    assert offsets[19] == pytest.approx(offsets[-1], rel=1e-5)
    assert torch.allclose(
        augmented[1, :, 2:4].norm(dim=-1), torch.ones(40), atol=1e-5
    )
    assert torch.equal(augmented[1, :, 2:4], trajectories[1, :, 2:4])


def test_velocity_offset_heading_is_smooth_without_position_tangent_noise() -> None:
    trajectories = torch.zeros(2, 40, 4)
    trajectories[..., 0] = torch.arange(1, 41) * 0.5
    trajectories[..., 2] = 1.0
    # Inject x/y jitter whose one-frame tangent would oscillate strongly.
    trajectories[1, 1::2, 1] = 0.05
    torch.manual_seed(17)
    augmented = _apply_hdp_trajectory_augmentation(
        trajectories,
        std=0.5,
        deterministic_first=True,
        ramp_steps=20,
        heading_mode="velocity_offset",
    )
    assert torch.equal(augmented[0], trajectories[0])
    assert torch.allclose(
        augmented[1, :, 2:4].norm(dim=-1), torch.ones(40), atol=1e-5
    )
    # Once the route-frame offset has reached its plateau on this straight
    # path, its derivative is zero and heading returns exactly forward.
    assert augmented[1, 25:, 2].min() > 0.999
    assert augmented[1, 25:, 3].abs().max() < 1e-5


def test_zero_ramp_reproduces_constant_hdp_offset() -> None:
    trajectories = torch.zeros(2, 5, 4)
    trajectories[..., 0] = torch.arange(1, 6) * 0.2
    trajectories[..., 2] = 1.0
    torch.manual_seed(29)
    a = torch.randn(2, 1) * 0.5
    b = torch.randn(2, 1) * 0.5
    a[0] = 0.0
    b[0] = 0.0
    expected = trajectories.clone()
    expected[..., 0] += a
    expected[..., 1] += b
    torch.manual_seed(29)
    augmented = _apply_hdp_trajectory_augmentation(
        trajectories,
        std=0.5,
        deterministic_first=True,
        ramp_steps=0,
    )
    assert torch.equal(augmented, expected)


def test_low_speed_hdp_offset_is_skipped_without_touching_candidates() -> None:
    trajectories = torch.zeros(2, 5, 4)
    trajectories[..., 0] = torch.arange(1, 6) * 0.2
    trajectories[..., 2] = 1.0
    torch.manual_seed(29)
    augmented = _apply_hdp_trajectory_augmentation(
        trajectories,
        std=0.5,
        deterministic_first=False,
        ramp_steps=0,
        augmentation_enabled=torch.tensor([False]),
    )
    assert torch.equal(augmented, trajectories)


def test_first_waypoint_gate_rejects_only_low_speed_non_anchor_targets() -> None:
    trajectories = torch.zeros(3, 5, 4)
    trajectories[0, 0, 0] = 0.1
    trajectories[1, 0, :2] = torch.tensor([0.6, 0.3])
    trajectories[2, 0, :2] = torch.tensor([-0.1, 0.0])
    data = {
        "ego_current_state": torch.zeros(1, 10),
        "turn_indicators": torch.tensor([[4] * 30 + [2]], dtype=torch.float32),
    }
    valid, diagnostics = _first_waypoint_gate_and_diagnostics(
        trajectories,
        data,
        AWRRolloutConfig(original_dp_first_waypoint_gate_enabled=True),
    )
    assert valid.tolist() == [True, False, False]
    assert diagnostics["stop_turn_scene"] == 1.0
    assert diagnostics["first_waypoint_gate_rejected_count"] == 2.0
    assert diagnostics["stop_turn_first_waypoint_gate_rejected_fraction"] == pytest.approx(
        2.0 / 3.0
    )


def test_first_waypoint_tangent_gate_ignores_sub_floor_sampler_noise() -> None:
    # At standstill the model emits x=0 exactly, so sub-millimeter lateral noise
    # makes the raw path tangent 90 degrees for every candidate.  Direction is
    # meaningless below the displacement floor and must not zero the group.
    #
    # The floor is 5 mm — deliberately *below* the measured stop-turn p95 first
    # step (3.9 cm).  An earlier 5 cm floor sat above it and silenced the gate
    # entirely at standstill, so candidates 2 and 3 below (6 mm and 15 cm
    # lateral, i.e. ~90 degrees of implied front-wheel angle) sailed through as
    # training targets and showed up as steering jitter on the vehicle.
    trajectories = torch.zeros(4, 5, 4)
    trajectories[1, 0, :2] = torch.tensor([0.0, 0.0003])
    trajectories[2, 0, :2] = torch.tensor([0.001, -0.006])
    trajectories[3, 0, :2] = torch.tensor([0.03, 0.15])
    data = {
        "ego_current_state": torch.zeros(1, 10),
        "turn_indicators": torch.tensor([[4] * 30 + [2]], dtype=torch.float32),
    }
    valid, diagnostics = _first_waypoint_gate_and_diagnostics(
        trajectories,
        data,
        AWRRolloutConfig(original_dp_first_waypoint_gate_enabled=True),
    )
    # Candidate 0 is the behaviour anchor (never gated); 1 is sub-floor noise
    # and stays exempt; 2 and 3 are real steering-lock commands.
    assert valid.tolist() == [True, True, False, False]
    assert diagnostics["first_waypoint_gate_rejected_count"] == 2.0


def test_first_waypoint_tangent_floor_matches_between_scene_and_batch_paths() -> None:
    generator = torch.Generator().manual_seed(11)
    trajectories = torch.zeros(6, 8, 5, 4)
    # Mix of standstill noise, forward starts, and above-floor jumps.
    trajectories[..., 0, :2] = torch.randn(6, 8, 2, generator=generator) * 0.08
    trajectories[0, :, 0, 0] = 0.0
    trajectories[0, :, 0, 1] = 0.004
    data = {
        "ego_current_state": torch.zeros(6, 10),
        "turn_indicators": torch.tensor(
            [[4] * 30 + [2]] * 6, dtype=torch.float32
        ),
    }
    config = AWRRolloutConfig(original_dp_first_waypoint_gate_enabled=True)
    batch_valid, batch_diagnostics = _first_waypoint_gate_and_diagnostics_batch(
        trajectories, data, config
    )
    for index in range(6):
        scene_data = {
            "ego_current_state": data["ego_current_state"][index : index + 1],
            "turn_indicators": data["turn_indicators"][index : index + 1],
        }
        scene_valid, scene_diagnostics = _first_waypoint_gate_and_diagnostics(
            trajectories[index], scene_data, config
        )
        assert scene_valid.tolist() == batch_valid[index].tolist()
        assert scene_diagnostics["first_waypoint_gate_rejected_count"] == (
            batch_diagnostics[index]["first_waypoint_gate_rejected_count"]
        )
    assert batch_valid[0].all()


def test_first_waypoint_gate_is_diagnostic_only_at_speed() -> None:
    trajectories = torch.zeros(2, 5, 4)
    trajectories[0, 0, :2] = torch.tensor([0.1, 0.0])
    trajectories[1, 0, :2] = torch.tensor([0.6, 0.3])
    data = {
        "ego_current_state": torch.tensor([[0.0, 0.0, 1.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        "turn_indicators": torch.tensor([[4] * 30 + [2]], dtype=torch.float32),
    }
    valid, diagnostics = _first_waypoint_gate_and_diagnostics(
        trajectories,
        data,
        AWRRolloutConfig(original_dp_first_waypoint_gate_enabled=True),
    )
    assert valid.tolist() == [True, True]
    assert diagnostics["first_waypoint_gate_low_speed"] == 0.0


def test_batched_first_waypoint_gate_matches_scene_path() -> None:
    trajectories = torch.zeros(2, 3, 5, 4)
    trajectories[0, 0, 0, 0] = 0.1
    trajectories[0, 1, 0, :2] = torch.tensor([0.6, 0.3])
    trajectories[0, 2, 0, :2] = torch.tensor([-0.1, 0.0])
    trajectories[1, :, 0, 0] = 0.1
    data = {
        "ego_current_state": torch.tensor(
            [[0.0] * 10, [0.0, 0.0, 1.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        ),
        "turn_indicators": torch.tensor(
            [[4] * 30 + [2], [4] * 30 + [2]], dtype=torch.float32
        ),
    }
    config = AWRRolloutConfig(original_dp_first_waypoint_gate_enabled=True)
    valid, diagnostics = _first_waypoint_gate_and_diagnostics_batch(
        trajectories, data, config
    )
    assert valid.tolist() == [[True, False, False], [True, True, True]]
    assert diagnostics[0]["stop_turn_scene"] == 1.0
    assert diagnostics[1]["first_waypoint_gate_low_speed"] == 0.0


def test_wandb_readable_aliases_use_percent_and_positive_penalty_magnitude() -> None:
    assert _wandb_readable_aliases("eval", "mean_det_collision", 0.0015) == {
        "eval_primary/safety/collision_rate_pct": pytest.approx(0.15),
    }
    assert _wandb_readable_aliases("eval", "mean_det_smoothness", -0.8) == {
        "eval_primary/driving_quality/smoothness_penalty_magnitude": pytest.approx(0.8),
    }
    assert _wandb_readable_aliases("eval_delta", "mean_det_collision", -0.001) == {}


def test_wandb_primary_payload_does_not_emit_legacy_raw_namespace() -> None:
    payload = _wandb_summary_payload(
        "eval",
        {"mean_det_reward": 0.9, "mean_det_collision": 0.001},
    )
    assert payload["eval_primary/reward/reward_pct"] == pytest.approx(90.0)
    assert payload["eval_primary/safety/collision_rate_pct"] == pytest.approx(0.1)
    assert not any(key.startswith("eval_01_") for key in payload)


def test_wandb_multimodality_is_grouped_under_train_candidate_view() -> None:
    payload = _wandb_summary_payload(
        "train",
        {
            "mean_candidate_count": 10.0,
            "mean_candidate_endpoint_spread_mean": 1.25,
            "mean_candidate_pairwise_ade": 0.42,
        },
    )
    assert payload[
        "train_primary/multimodality/endpoint_spread_mean_m"
    ] == pytest.approx(1.25)
    assert payload[
        "train_primary/multimodality/pairwise_trajectory_ade_m"
    ] == pytest.approx(0.42)


def test_wandb_k1_eval_suppresses_definitionally_zero_multimodality() -> None:
    payload = _wandb_summary_payload(
        "eval",
        {
            "mean_candidate_count": 1.0,
            "mean_candidate_endpoint_spread_mean": 0.0,
            "mean_candidate_pairwise_ade": 0.0,
            "mean_candidate_reward_std": 0.0,
            "mean_det_reward": 0.91,
        },
    )
    assert payload["eval_primary/reward/reward_pct"] == pytest.approx(91.0)
    assert not any(key.startswith("eval_primary/multimodality/") for key in payload)
    assert not any("endpoint_spread" in key for key in payload)
    assert not any("pairwise_trajectory" in key for key in payload)
    assert not any("candidate_reward_std" in key for key in payload)


def test_wandb_k_greater_than_one_eval_retains_multimodality() -> None:
    payload = _wandb_summary_payload(
        "eval",
        {
            "mean_candidate_count": 10.0,
            "mean_candidate_endpoint_spread_mean": 1.25,
        },
    )
    assert payload[
        "eval_primary/multimodality/endpoint_spread_mean_m"
    ] == pytest.approx(1.25)


def test_wandb_train_selector_exposes_checkpoint_reward_and_delta() -> None:
    payload = _wandb_summary_payload(
        "train_selector_ema",
        {"mean_det_reward": 0.91},
        baseline={"mean_det_reward": 0.90},
    )
    assert payload[
        "train_selector_ema_primary/reward/reward_pct"
    ] == pytest.approx(91.0)
    assert payload[
        "train_selector_ema_delta_primary/reward/improvement_reward_pp"
    ] == pytest.approx(1.0)


def test_wandb_train_exposes_policy_transaction_without_raw_legacy_keys() -> None:
    payload = _wandb_summary_payload(
        "train",
        {
            "policy_transaction_candidate": 1.0,
            "policy_transaction_accepted": 0.0,
            "policy_transaction_rolled_back": 0.0,
            "policy_training_continued_from_proposal": 1.0,
            "optimizer_state_reset_after_policy_commit": 1.0,
            "policy_proposal_relative_l2": 4.0e-5,
            "policy_accepted_relative_l2": 2.0e-6,
            "ema_policy_commit_rate": 0.05,
            "train_selection_reward_before_epoch": 0.932,
            "train_selection_reward_delta_vs_best": -0.00012,
        },
    )
    prefix = "train_primary/policy_transaction/"
    assert payload[prefix + "proposal_evaluated_0_or_1"] == 1.0
    assert payload[prefix + "proposal_selected_as_best_0_or_1"] == 0.0
    assert payload[prefix + "proposal_rolled_back_0_or_1"] == 0.0
    assert payload[prefix + "proposal_used_as_next_start_0_or_1"] == 1.0
    assert payload[prefix + "optimizer_state_reset_0_or_1"] == 1.0
    assert payload[prefix + "proposal_relative_l2"] == pytest.approx(4.0e-5)
    assert payload[prefix + "committed_update_relative_l2"] == pytest.approx(2.0e-6)
    assert payload[prefix + "ema_commit_rate_pct"] == pytest.approx(5.0)
    assert payload[prefix + "best_reward_before_epoch_pct"] == pytest.approx(93.2)
    assert payload[prefix + "reward_delta_vs_best_pp"] == pytest.approx(-0.012)
    assert not any(key.startswith("train_01_") for key in payload)


def test_wandb_improvement_aliases_use_positive_is_better_convention() -> None:
    payload = _wandb_summary_payload(
        "eval",
        {
            "mean_det_reward": 0.91,
            "mean_ade": 2.0,
            "mean_det_collision": 0.001,
        },
        baseline={
            "mean_det_reward": 0.90,
            "mean_ade": 2.2,
            "mean_det_collision": 0.002,
        },
    )
    assert payload["eval_delta_primary/reward/improvement_reward_pp"] == pytest.approx(1.0)
    assert payload["eval_delta_primary/imitation/improvement_ade_m"] == pytest.approx(0.2)
    assert payload["eval_delta_primary/safety/improvement_collision_rate_pp"] == pytest.approx(0.1)


def test_wandb_epoch_logger_commits_train_and_eval_together() -> None:
    class FakeRun:
        def __init__(self):
            self.payloads = []

        def log(self, payload):
            self.payloads.append(payload)

    run = FakeRun()
    _wandb_log_epoch_summaries(
        run,
        epoch=9,
        train_summary={"mean_loss": 0.1},
        eval_summary={"mean_det_reward": 0.9, "mean_det_collision": 0.001},
    )
    assert len(run.payloads) == 1
    payload = run.payloads[0]
    assert payload["epoch"] == 9
    assert payload["train_primary/optimization/loss"] == pytest.approx(0.1)
    assert payload["eval_primary/reward/reward_pct"] == pytest.approx(90.0)
    assert payload["eval_primary/safety/collision_rate_pct"] == pytest.approx(0.1)


def test_resumed_refresh_is_mapped_to_preceding_global_epoch(tmp_path) -> None:
    source_run = tmp_path / "mine"
    replay_root = source_run / "replay_buffer"
    (source_run / "train_epoch_001").mkdir(parents=True)
    (source_run / "eval_epoch_001").mkdir(parents=True)
    replay_root.mkdir()
    (source_run / "train_epoch_001" / "summary.json").write_text(
        json.dumps({"epoch": 1, "mean_loss": 0.0, "rollout_refresh": 1.0})
    )
    (source_run / "eval_epoch_001" / "summary.json").write_text(
        json.dumps({"epoch": 1, "mean_det_reward": 0.9})
    )

    class FakeRun:
        def __init__(self):
            self.payloads = []
            self.summary = {}

        def log(self, payload):
            self.payloads.append(payload)

    run = FakeRun()
    resumed_run = tmp_path / "replay"
    resumed_run.mkdir()
    train_awr_module._wandb_log_inherited_refresh_epoch(
        run,
        SimpleNamespace(resume_replay_root=replay_root, start_epoch=12),
        resumed_run,
        baseline={"mean_det_reward": 0.9},
    )

    assert run.payloads[0]["epoch"] == 11
    assert run.summary["resume/inherited_refresh_epoch"] == 11
    inherited = json.loads(
        (resumed_run / "inherited_epoch_011_train_summary.json").read_text()
    )
    assert inherited["epoch"] == 11
    assert inherited["source_local_epoch"] == 1
    rows = [json.loads(line) for line in (resumed_run / "metrics.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in rows] == [11, 11]
    assert rows[1]["validation_scope"] == (
        "full_baseline_reused_policy_unchanged_during_refresh"
    )
    assert rows[1]["policy_unchanged_during_refresh"] == 1.0


def test_resumed_refresh_follows_overlay_to_source_mine(tmp_path) -> None:
    mine = tmp_path / "mine"
    source_replay = mine / "replay_buffer"
    overlay = tmp_path / "overlay"
    overlay_replay = overlay / "replay_buffer"
    replay_run = tmp_path / "replay_run"
    (mine / "train_epoch_001").mkdir(parents=True)
    (mine / "eval_epoch_001").mkdir(parents=True)
    source_replay.mkdir()
    overlay_replay.mkdir(parents=True)
    replay_run.mkdir()
    (mine / "train_epoch_001" / "summary.json").write_text(
        json.dumps(
            {
                "epoch": 1,
                "scene_count": 5_446_656,
                "mean_loss": 0.0,
                "rollout_refresh": 1.0,
            }
        )
    )
    (mine / "eval_epoch_001" / "summary.json").write_text(
        json.dumps({"epoch": 1, "scene_count": 8, "mean_det_reward": 0.1})
    )
    (overlay / "overlay_manifest.json").write_text(
        json.dumps({"source_replay": str(source_replay.resolve())})
    )

    class FakeRun:
        def __init__(self):
            self.payloads = []
            self.summary = {}

        def log(self, payload):
            self.payloads.append(payload)

    run = FakeRun()
    full_baseline = {"scene_count": 46_262, "mean_det_reward": 0.9}
    train_awr_module._wandb_log_inherited_refresh_epoch(
        run,
        SimpleNamespace(resume_replay_root=overlay_replay, start_epoch=2),
        replay_run,
        baseline=full_baseline,
    )

    assert run.payloads[0]["epoch"] == 1
    assert run.payloads[0]["eval_primary/reward/reward_pct"] == pytest.approx(
        90.0
    )
    inherited_eval = json.loads(
        (replay_run / "inherited_epoch_001_eval_summary.json").read_text()
    )
    assert inherited_eval["scene_count"] == 46_262
    assert inherited_eval["source_refresh_eval_scene_count"] == 8
    assert inherited_eval["policy_unchanged_during_refresh"] == 1.0
    assert run.summary["resume/source_run_dir"] == str(mine.resolve())


def test_compact_mine_row_summary_matches_finite_scalar_means():
    """Mine-only compaction keeps scalar reporting without gathering rows."""

    from rlvr.train_awr import (
        _compact_numeric_row_summary,
        _merge_compact_numeric_row_summaries,
    )

    rank0 = [
        {"scene_count": 2, "det_reward": 0.8, "optimizer_step": 0.0},
        {"scene_count": 3, "det_reward": 1.0, "loss": float("nan")},
    ]
    rank1 = [
        {"scene_count": 5, "det_reward": 0.6, "optimizer_step": 0.0},
        {"scene_count": 7, "det_reward": float("inf"), "note": "ignored"},
    ]
    merged, sums = _merge_compact_numeric_row_summaries(
        [
            _compact_numeric_row_summary(rank0),
            _compact_numeric_row_summary(rank1),
        ]
    )

    assert merged["scene_count"] == 17
    assert merged["row_count"] == 4
    assert merged["scene_rows_materialized"] == 0
    assert merged["compact_distributed_summary"] == 1
    assert merged["numeric_percentiles_omitted"] == 1
    assert merged["mean_det_reward"] == pytest.approx((0.8 + 1.0 + 0.6) / 3.0)
    assert merged["mean_optimizer_step"] == 0.0
    assert "mean_loss" not in merged
    assert sums["optimizer_step"] == 0.0
