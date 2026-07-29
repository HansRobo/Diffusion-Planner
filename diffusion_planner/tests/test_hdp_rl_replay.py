"""Tests for the 100-epoch mine/replay relay state machine."""

import json
from types import SimpleNamespace

import pytest
import torch
from diffusion_planner.hdp_rl_replay import (
    _META,
    CycleReplayReader,
    CycleReplayWriter,
    cleanup_previous_cycle,
    cycle_index,
    is_mine_epoch,
    relay_epoch,
    reward_fingerprint,
)

from diffusion_planner import hdp_rl_epoch


def _args(**overrides):
    defaults = dict(
        num_generations=4,
        rl_reward_source="native",
        rl_reward_aggregation="weighted_sum",
        rl_reward_horizon_steps=0,
        rl_reward_beta=1.0,
        rl_reward_w_safety=0.0,
        rl_reward_w_risk=1.0,
        rl_reward_w_follow=3.0,
        rl_reward_w_lane=2.5,
        rl_reward_w_progress=3.0,
        rl_reward_w_road_border=0.0,
        rl_behavior_gate="safety",
        rl_pdm_red_light_gate=True,
        rl_candidate_aug_epochs=0,
        rl_candidate_aug_std=0.5,
        rl_noise_scale=1.5,
        rl_rollout_steps=6,
        ddp=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _shard(num_scenes=2, n=4, T=8):
    reward = torch.arange(float(num_scenes * n))
    return {
        "ego_world": torch.randn(num_scenes * n, T, 4),
        "reward": reward,
        "reward_weights": torch.rand(num_scenes * n),
        "valid_sample": torch.ones(num_scenes * n, dtype=torch.bool),
        "ego_current_state": torch.randn(num_scenes, 10),
        "route_lanes": torch.randn(num_scenes, 2, 3, 33),
        "encoding": torch.randn(num_scenes, 5, 16),
    }


def test_relay_schedule_is_ten_cycles_of_ten():
    mine_epochs = [e for e in range(1, 101) if is_mine_epoch(e, 10)]
    assert mine_epochs == [1, 11, 21, 31, 41, 51, 61, 71, 81, 91]
    assert [cycle_index(e, 10) for e in mine_epochs] == list(range(10))
    assert cycle_index(100, 10) == 9  # the last replay epoch trains cycle 9's cache


def test_trainer_epoch_zero_mines_cycle_zero():
    """The training loop counts from zero; the relay is numbered from one.

    Without the shift, a fresh run's first epoch resolves to cycle -1 and reads as
    a replay epoch, so it aborts on a cache that no mining pass ever wrote.
    """
    assert is_mine_epoch(relay_epoch(0), 10)
    assert cycle_index(relay_epoch(0), 10) == 0
    assert not is_mine_epoch(relay_epoch(1), 10)
    assert cycle_index(relay_epoch(9), 10) == 0
    assert is_mine_epoch(relay_epoch(10), 10)
    assert cycle_index(relay_epoch(10), 10) == 1
    trainer_mine_epochs = [e for e in range(100) if is_mine_epoch(relay_epoch(e), 10)]
    assert trainer_mine_epochs == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]


def test_writer_reader_roundtrip_preserves_frozen_weights(tmp_path):
    args = _args()
    writer = CycleReplayWriter(tmp_path, cycle=0, rank=0, args=args)
    shard = _shard()
    writer.append(shard)
    writer.finalize(use_ddp=False)
    reader = CycleReplayReader(tmp_path, cycle=0, rank=0, args=args)
    shards = reader.epoch_shards(seed=1)
    assert len(shards) == 1
    loaded = CycleReplayReader.load(shards[0], torch.device("cpu"))
    torch.testing.assert_close(loaded["reward_weights"], shard["reward_weights"])
    torch.testing.assert_close(loaded["ego_world"], shard["ego_world"])
    assert loaded["valid_sample"].dtype in (torch.bool, torch.float32)


def test_fingerprint_keeps_the_retired_first_waypoint_gate_field(tmp_path):
    """A cache mined before `rl_first_waypoint_gate` was deleted must still load.

    Every cache ever mined carries the field (it was False in all 32 runs), so
    dropping it from the fingerprint would turn 'False' into 'None' and abort the
    resume of an otherwise-valid multi-terabyte cycle instead of reading it.
    """
    args = _args()
    assert not hasattr(args, "rl_first_waypoint_gate")
    writer = CycleReplayWriter(tmp_path, cycle=0, rank=0, args=args)
    writer.append(_shard())
    writer.finalize(use_ddp=False)
    meta = json.loads((tmp_path / "cycle_000" / _META).read_text())
    assert meta["fingerprint"]["rl_first_waypoint_gate"] == repr(False)
    CycleReplayReader(tmp_path, cycle=0, rank=0, args=args)  # must not raise


def test_fingerprint_is_unchanged_by_the_default_candidate_augmentation():
    """The off default must be invisible; enabling it must invalidate the cache.

    `rl_candidate_aug_epochs` was added after multi-terabyte caches existed, so
    recording it unconditionally would abort every historical resume. It is recorded
    only when it deviates from off -- exactly when the mined groups really differ.
    """
    off = reward_fingerprint(_args())
    assert "rl_candidate_aug_epochs" not in off
    assert reward_fingerprint(_args(rl_candidate_aug_epochs=0)) == off
    on = reward_fingerprint(_args(rl_candidate_aug_epochs=5))
    assert on["rl_candidate_aug_epochs"] == repr(5)
    assert on != off


def test_reader_fails_closed_on_contract_mismatch(tmp_path):
    writer = CycleReplayWriter(tmp_path, cycle=0, rank=0, args=_args())
    writer.append(_shard())
    writer.finalize(use_ddp=False)
    with pytest.raises(RuntimeError, match="contract mismatch"):
        CycleReplayReader(tmp_path, cycle=0, rank=0, args=_args(rl_reward_beta=2.0))
    with pytest.raises(RuntimeError, match="incomplete"):
        CycleReplayReader(tmp_path, cycle=1, rank=0, args=_args())


def test_writer_refuses_completed_cycle_and_cleanup_is_scoped(tmp_path):
    args = _args()
    writer = CycleReplayWriter(tmp_path, cycle=0, rank=0, args=args)
    writer.append(_shard())
    writer.finalize(use_ddp=False)
    with pytest.raises(RuntimeError, match="completed replay cycle"):
        CycleReplayWriter(tmp_path, cycle=0, rank=0, args=args)
    cleanup_previous_cycle(tmp_path, cycle=1, rank=0)
    assert not (tmp_path / "cycle_000").exists()


def test_replay_update_uses_frozen_weights_and_skips_invalid(monkeypatch):
    captured = {}

    def fake_update(model, norm_exp, ego_world, reward, weights, valid, *rest, **kwargs):
        captured["weights"] = weights.clone()
        captured["norm_exp_batch"] = norm_exp["ego_current_state"].shape[0]
        return {"loss": torch.zeros(())}

    monkeypatch.setattr(hdp_rl_epoch, "_backward_reward_weighted_update", fake_update)

    class _Opt:
        def zero_grad(self, set_to_none=False):
            pass

        def step(self):
            captured["stepped"] = True

    shard = _shard(num_scenes=2, n=4)
    args = _args()
    result = hdp_rl_epoch._replay_update_from_shard(
        shard, torch.nn.Linear(1, 1), _Opt(), [torch.nn.Parameter(torch.zeros(1))], args
    )
    assert result is not None and captured["stepped"]
    torch.testing.assert_close(captured["weights"], shard["reward_weights"])
    assert captured["norm_exp_batch"] == 8  # scenes x group size

    shard["valid_sample"] = torch.zeros(8, dtype=torch.bool)
    captured.clear()
    assert (
        hdp_rl_epoch._replay_update_from_shard(shard, torch.nn.Linear(1, 1), _Opt(), [], args)
        is None
    )
    assert "stepped" not in captured  # discarded batch must not touch the optimizer
