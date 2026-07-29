import json

import torch

from rlvr.posttrain.preflight import check
from rlvr.posttrain.spec import Spec


def overlay(root, ranks=2, gate_step=True, registered=True):
    for rank in range(ranks):
        directory = root / f"rank_{rank:04d}"
        directory.mkdir(parents=True)
        if gate_step:
            (directory / "expert_hard_gate_step.npy").write_bytes(b"\x00" * 8)
        (directory / "expert_anchor_manifest.json").write_text(
            json.dumps({"arrays": ["expert_hard_gate_step"] if registered else []})
        )
    return root


def checkpoint(path, *, epoch=3, optimizer=True):
    state = {"model": {}, "epoch": epoch}
    if optimizer:
        state["optimizer_state_dict"] = {}
    torch.save(state, path)
    return path


def spec(tmp_path, **train):
    base = {"model_path": str(checkpoint(tmp_path / "epoch_003.pth"))}
    base.update(train)
    return Spec(name="t", train=base, launch={"nproc": 2})


def test_a_missing_config_is_reported(tmp_path):
    problems = check(spec(tmp_path, config=str(tmp_path / "nope.json")))
    assert any("--config missing" in p for p in problems)


def test_every_overlay_rank_is_checked_up_front(tmp_path):
    root = overlay(tmp_path / "buffer", ranks=1)
    problems = check(
        spec(tmp_path, resume_replay_root=str(root), expert_anchor_safe_step_scoped=True)
    )
    assert any("rank_0001" in p for p in problems)


def test_step_scoping_requires_the_gate_step_array(tmp_path):
    root = overlay(tmp_path / "buffer", gate_step=False)
    problems = check(
        spec(tmp_path, resume_replay_root=str(root), expert_anchor_safe_step_scoped=True)
    )
    assert len(problems) == 2
    assert all("expert_hard_gate_step.npy" in p for p in problems)
    # Without the flag the array is not needed and the same overlay is fine.
    assert check(spec(tmp_path, resume_replay_root=str(root))) == []


def test_an_unregistered_gate_step_is_a_provenance_failure(tmp_path):
    root = overlay(tmp_path / "buffer", registered=False)
    problems = check(
        spec(tmp_path, resume_replay_root=str(root), expert_anchor_safe_step_scoped=True)
    )
    assert all("not registered" in p for p in problems)


def test_optimizer_resume_needs_an_optimizer_in_the_checkpoint(tmp_path):
    path = checkpoint(tmp_path / "epoch_003.pth", optimizer=False)
    problems = check(
        Spec(name="t", train={"model_path": str(path), "resume_optimizer_state": True})
    )
    assert any("optimizer_state_dict" in p for p in problems)


def test_start_epoch_must_continue_the_checkpoint(tmp_path):
    path = checkpoint(tmp_path / "epoch_003.pth", epoch=3)
    resume = {"model_path": str(path), "use_policy_state": True}
    assert check(Spec(name="t", train={**resume, "start_epoch": 4})) == []
    problems = check(Spec(name="t", train={**resume, "start_epoch": 6}))
    assert any("does not continue" in p for p in problems)


def test_a_fresh_base_numbers_its_own_epochs(tmp_path):
    """Starting from someone else's checkpoint is not a continuation.

    A supervised base carries the epoch *its* training reached, which says
    nothing about this run -- and it has neither a live policy nor AdamW
    moments to continue from, so the gate is the same switch either way.
    """

    path = checkpoint(tmp_path / "epoch_080.pth", epoch=80)
    assert check(Spec(name="t", train={"model_path": str(path), "start_epoch": 1})) == []


def test_a_corrupt_checkpoint_reads_as_a_preflight_failure(tmp_path):
    path = tmp_path / "epoch_003.pth"
    path.write_bytes(b"not a checkpoint")
    assert any("unreadable" in p for p in check(Spec(name="t", train={"model_path": str(path)})))


def test_an_artifact_the_earlier_arm_has_not_produced_yet_is_reported(tmp_path):
    """Preflight is where "not mined yet" gets said, not where it crashes.

    The whole chain is supervised, so a traceback out of a check would take down
    the process that was going to relaunch the arm once the artifact appeared.
    """

    problems = check(
        Spec(
            name="t",
            train={
                "model_path": {"latest_epoch": str(tmp_path)},
                "start_epoch": {"resume_after_latest_epoch": str(tmp_path)},
                "resume_replay_root": {"newest": f"{tmp_path}/*/replay_buffer"},
            },
        )
    )
    assert any("--model_path unresolved" in p for p in problems)
    assert any("--resume_replay_root unresolved" in p for p in problems)


def config(path, *, guided=True, interval=10):
    """The entrypoint config, in sections, the way the real one is written."""

    path.write_text(
        json.dumps(
            {
                "awr": {
                    "positive_advantage_only": False,
                    "plannerrft_guided_exploration": guided,
                },
                "training": {"hdp_rollout_interval": interval, "train_epochs": 10},
            }
        )
    )
    return path


def refresh_spec(tmp_path, **train):
    """An arm that applies candidate weighting on top of a guided-mining config."""

    base = {
        "config": str(config(tmp_path / "config.json")),
        "positive_advantage_only": True,
        "start_epoch": 4,
        "epochs": 60,
    }
    base.update(train)
    return spec(tmp_path, **base)


def test_a_refresh_epoch_that_would_abort_mining_is_reported(tmp_path):
    """The abort waits for the refresh epoch, so only preflight can be early.

    Compile warmup trips the same guard and records it as data rather than
    failing, so nothing before the first refresh epoch warns about this.
    """

    problems = check(refresh_spec(tmp_path, hdp_rollout_interval=30))
    assert any("epoch 31 a refresh" in p for p in problems)


def test_an_interval_above_the_epoch_count_never_reaches_mining(tmp_path):
    assert check(refresh_spec(tmp_path, hdp_rollout_interval=1000)) == []


def test_dropping_either_flag_makes_the_refresh_legal_again(tmp_path):
    """What `mine` does: it drops the candidate-weighting flags rather than set them."""

    assert check(refresh_spec(tmp_path, hdp_rollout_interval=30, positive_advantage_only=None)) == []
    assert (
        check(
            spec(
                tmp_path,
                config=str(config(tmp_path / "unguided.json", guided=False)),
                positive_advantage_only=True,
                start_epoch=4,
                epochs=60,
                hdp_rollout_interval=30,
            )
        )
        == []
    )


def test_an_unresolved_start_epoch_defers_the_refresh_check(tmp_path):
    """Guessing the range would fail an arm whose predecessor is still running."""

    problems = check(
        refresh_spec(
            tmp_path,
            hdp_rollout_interval=30,
            start_epoch={"resume_after_latest_epoch": str(tmp_path / "nothing")},
        )
    )
    assert not any("refresh" in p for p in problems)
