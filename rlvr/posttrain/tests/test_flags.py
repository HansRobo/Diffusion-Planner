import os
import shutil
from dataclasses import replace
from pathlib import Path

from rlvr.posttrain.flags import command, to_argv, validate
from rlvr.posttrain.spec import Spec, load

PRESETS = Path(__file__).resolve().parents[1] / "presets"


def spec(**train) -> Spec:
    return Spec(name="t", train={"model_path": "/x/epoch_001.pth", **train})


def values(argv, flag):
    start = argv.index(flag) + 1
    end = next(
        (i for i in range(start, len(argv)) if argv[i].startswith("--")), len(argv)
    )
    return argv[start:end]


def test_unknown_flags_are_caught_before_a_gpu_is_taken():
    errors = validate(spec(expert_anchor_weigt=0.4))
    assert any("unknown flag --expert_anchor_weigt" in e for e in errors)
    assert any("expert_anchor_weight" in e for e in errors)


def test_a_run_needs_a_checkpoint():
    assert "train needs model_path or weights_zip" in validate(Spec(name="t"))


def test_switches_are_written_as_switches():
    argv = to_argv(spec(use_policy_state=True, resume_optimizer_state=False))
    assert "--use_policy_state" in argv
    assert "--resume_optimizer_state" not in argv


def test_boolean_optional_flags_emit_their_negation():
    argv = to_argv(spec(expert_anchor_active_groups_only=False))
    assert "--no-expert_anchor_active_groups_only" in argv
    argv = to_argv(spec(expert_anchor_active_groups_only=True))
    assert "--expert_anchor_active_groups_only" in argv


def test_a_switch_given_a_value_is_an_error():
    assert any("switch" in e for e in validate(spec(use_policy_state=1)))


def test_multi_value_flags_keep_their_arity():
    argv = to_argv(spec(diffusion_t_range=[0.001, 0.2]))
    assert values(argv, "--diffusion_t_range") == ["0.001", "0.2"]
    assert any("takes 2 values" in e for e in validate(spec(diffusion_t_range=[0.001])))


def test_the_objective_is_always_passed_explicitly():
    assert values(to_argv(spec()), "--awr_objective") == ["awr"]
    assert any(
        "unknown objective" in e
        for e in validate(Spec(name="t", objective="ppo", train={"model_path": "/x"}))
    )


def test_latest_epoch_picks_the_last_checkpoint_not_the_best(tmp_path):
    for name in ("epoch_002.pth", "epoch_010.pth", "best_model.pth"):
        (tmp_path / name).write_bytes(b"x")
    argv = to_argv(
        Spec(
            name="t",
            train={
                "model_path": {"latest_epoch": str(tmp_path)},
                "start_epoch": {"resume_after_latest_epoch": str(tmp_path)},
            },
        )
    )
    assert values(argv, "--model_path") == [str(tmp_path / "epoch_010.pth")]
    assert values(argv, "--start_epoch") == ["11"]


def test_an_unresolvable_value_is_an_error_not_a_crash(tmp_path):
    errors = validate(Spec(name="t", train={"model_path": {"latest_epoch": str(tmp_path)}}))
    assert any("no epoch_" in e for e in errors)


def test_command_puts_the_launcher_in_front():
    import sys

    argv = command(spec())
    # A PATH lookup for the torchrun script fails in a detached session.
    assert argv[:4] == [sys.executable, "-m", "torch.distributed.run", "--standalone"]
    assert argv[len(argv) - 1 - argv[::-1].index("-m") + 1] == "rlvr.train_awr"


def test_another_trainer_is_validated_against_its_own_flags():
    """One spec language, several trainers: the flag surface follows the entrypoint."""

    grpo = Spec(
        name="t",
        entrypoint="rlvr.train_grpo",
        train={"model_path": "/x", "config": "/c.json", "exp_name": "e"},
    )
    assert validate(grpo) == []
    # A train_awr flag is not a train_grpo flag, and saying so is the point.
    assert any(
        "unknown flag --expert_anchor_weight" in e
        for e in validate(
            Spec(
                name="t",
                entrypoint="rlvr.train_grpo",
                train={"model_path": "/x", "expert_anchor_weight": 0.4},
            )
        )
    )


def test_a_trainer_without_the_seam_gets_no_objective_flag():
    grpo = Spec(name="t", entrypoint="rlvr.train_grpo", train={"model_path": "/x"})
    assert "--awr_objective" not in to_argv(grpo)
    # ...and its own loop owns the method, so the registry does not police it.
    assert validate(Spec(name="t", entrypoint="rlvr.train_grpo", objective="ppo",
                         train={"model_path": "/x"})) == []
    assert any(
        "--reward_override" in e
        for e in validate(
            Spec(
                name="t",
                entrypoint="rlvr.train_grpo",
                train={"model_path": "/x"},
                reward={"hdp_lane_weight": 1},
            )
        )
    )


def test_an_entrypoint_that_cannot_be_introspected_is_an_error():
    errors = validate(Spec(name="t", entrypoint="rlvr.reward", train={"model_path": "/x"}))
    assert any("build_parser" in e for e in errors)
    assert any("no.module" in e or "No module" in e
               for e in validate(Spec(name="t", entrypoint="no.module", train={"model_path": "/x"})))


def test_direct_launcher_skips_torchrun():
    import sys

    argv = command(
        Spec(
            name="t",
            entrypoint="rlvr.train_grpo",
            launch={"launcher": "direct"},
            train={"model_path": "/x"},
        )
    )
    assert argv[:3] == [sys.executable, "-m", "rlvr.train_grpo"]
    assert "torch.distributed.run" not in argv


#: Scaffolding, not runnable arms: site.json is the paths you edit, base.json the
#: shared method. Named rather than inferred so a real arm cannot skip itself.
INHERITANCE_TARGETS = {"site.json", "base.json"}

#: Four kinds of flag are deliberately absent from the frozen list below, because
#: they are bookkeeping rather than method:
#:
#:   --use_policy_state / --resume_optimizer_state continue a long AWR run, and the
#:   preset starts from the site's base checkpoint, which has neither a live policy
#:   state nor AdamW moments to restore.
#:   --amp_dtype pins a precision; a preset takes the trainer's own default.
#:   the --wandb_* family names one machine's entity, project, run and tags.
#:
#: Every remaining flag is method and must survive.


def test_bundled_presets_all_validate():
    # A dynamic value names an artifact an earlier arm produces, so on a machine
    # that has not run one it is missing rather than wrong; every other error is
    # a mistake in the preset itself and must fail here.
    for path in sorted(PRESETS.glob("*.json")):
        if path.name in INHERITANCE_TARGETS:
            continue
        for item in load(path):
            errors = [e for e in validate(item) if "unresolved value" not in e]
            assert not errors, f"{path.name}: {errors}"


#: Every method flag the retired bash launcher passed, frozen here so retiring the
#: script did not retire what it guaranteed.  Read it as the floor of the dual arm:
#: the preset may add flags, but dropping one of these silently changes the method.
#: NOT_METHOD is already excluded -- those were the launcher's own bookkeeping.
LAUNCHER_METHOD_FLAGS = (
    "--args_path",
    "--awr_beta",
    "--awr_candidate_loss_horizon",
    "--awr_loss_type",
    "--behavior_anchor_weight",
    "--config",
    "--device",
    "--diffusion_t_range",
    "--drop_all_zero_groups",
    "--ema_decay",
    "--eval_k",
    "--exp_name",
    "--expert_anchor_loss_horizon",
    "--expert_anchor_safe_step_scoped",
    "--expert_anchor_weight",
    "--full_valid_interval",
    "--learning_rate",
    "--max_train_scenes",
    "--model_path",
    "--no-skip_filtered_scenes",
    "--output_dir",
    "--positive_advantage_only",
    "--replay_sampling",
    "--resume_replay_root",
    "--save_rollouts",
    "--scene_batch_size",
    "--scene_load_workers",
    "--train_npz_list",
    "--train_selector_scenes",
    "--trainable_scope",
    "--valid_npz_list",
)


def test_the_dual_preset_keeps_every_method_flag_the_launcher_had():
    """The framework replaced a bash launcher; same method or it did not."""

    # The two dynamic values resolve against a mine that only exists on a machine
    # that has run one.  Flag *names* do not depend on what they resolve to, so
    # they are stubbed and the guarantee holds everywhere instead of skipping.
    spec = load(PRESETS / "dual.json")[0]
    train = {**spec.train, "model_path": "/x/epoch_001.pth", "start_epoch": 2}
    argv = set(to_argv(replace(spec, train=train)))
    missing = {
        flag
        for flag in LAUNCHER_METHOD_FLAGS
        if flag not in argv and flag.replace("--no-", "--") not in argv
    }
    assert not missing, f"preset drops launcher flags: {sorted(missing)}"


def test_reward_fields_become_overrides():
    argv = to_argv(Spec(name="t", train={"model_path": "/x"}, reward={"hdp_lane_weight": 0.25}))
    assert "--reward_override" in argv
    assert values(argv, "--reward_override") == ["hdp_lane_weight=0.25"]


def test_an_unknown_reward_field_is_caught_with_a_suggestion():
    errors = validate(Spec(name="t", train={"model_path": "/x"}, reward={"hdp_lane_wieght": 1}))
    assert any("unknown reward field hdp_lane_wieght" in e for e in errors)
    assert any("hdp_lane_weight" in e for e in errors)


def test_reward_overrides_reach_the_trainer_as_it_parses_them():
    from rlvr.reward import RewardConfig
    from rlvr.train_awr import apply_reward_overrides

    argv = to_argv(
        Spec(
            name="t",
            train={"model_path": "/x"},
            reward={"rb_gate_enabled": False, "reward_horizon_steps": 80},
        )
    )
    items = [argv[i + 1] for i, token in enumerate(argv) if token == "--reward_override"]
    config = RewardConfig()
    applied = apply_reward_overrides(config, items)
    assert applied == {"rb_gate_enabled": False, "reward_horizon_steps": 80}
    assert config.rb_gate_enabled is False and config.reward_horizon_steps == 80


def test_latest_epoch_finds_the_run_directory_below_the_output_dir(tmp_path):
    """A spec names an output_dir; the trainer writes into a stamp below it."""

    run_dir = tmp_path / "20260101-000000_dual"
    run_dir.mkdir()
    for name in ("epoch_002.pth", "epoch_010.pth", "best_model.pth"):
        (run_dir / name).write_bytes(b"x")
    argv = to_argv(
        Spec(
            name="t",
            train={
                "model_path": {"latest_epoch": str(tmp_path)},
                "start_epoch": {"resume_after_latest_epoch": str(tmp_path)},
            },
        )
    )
    assert values(argv, "--model_path") == [str(run_dir / "epoch_010.pth")]
    assert values(argv, "--start_epoch") == ["11"]


def test_newest_resolves_a_stamped_directory_nobody_can_name_in_advance(tmp_path):
    """The mine picks its own stamp; the arm that reads its buffer globs for it."""

    for stamp in ("20260101-000000_mine", "20260102-000000_mine"):
        (tmp_path / stamp / "replay_buffer").mkdir(parents=True)
    newest = tmp_path / "20260102-000000_mine" / "replay_buffer"
    os.utime(newest, (2_000_000_000, 2_000_000_000))
    argv = to_argv(
        Spec(
            name="t",
            train={
                "model_path": "/x",
                "resume_replay_root": {"newest": f"{tmp_path}/*/replay_buffer"},
            },
        )
    )
    assert values(argv, "--resume_replay_root") == [str(newest)]


def test_a_glob_that_matches_nothing_reads_as_a_missing_artifact(tmp_path):
    """Not yet mined is a different thing from a broken spec, and says so."""

    errors = validate(
        Spec(
            name="t",
            train={"model_path": "/x", "resume_replay_root": {"newest": f"{tmp_path}/*/buf"}},
        )
    )
    assert any("unresolved value" in e and "nothing matches" in e for e in errors)


def test_a_resumed_arm_continues_the_highest_epoch_of_any_attempt(tmp_path):
    """Each relaunch gets its own directory, so the newest is not the furthest."""

    first, second = tmp_path / "20260101-000000_a", tmp_path / "20260102-000000_a"
    for directory, epochs in ((first, (1, 2, 3)), (second, (4, 5))):
        directory.mkdir()
        for epoch in epochs:
            (directory / f"epoch_{epoch:03d}.pth").write_bytes(b"x")
    # A third attempt that died before its first checkpoint must not reset it.
    (tmp_path / "20260103-000000_a").mkdir()
    argv = to_argv(Spec(name="t", train={"model_path": {"latest_epoch": str(tmp_path)}}))
    assert values(argv, "--model_path") == [str(second / "epoch_005.pth")]


def test_a_relative_directive_resolves_against_the_repo_root(monkeypatch, tmp_path):
    """A supervisor is launched from wherever; the spec still means one thing."""

    from rlvr.posttrain.spec import ROOT

    run_dir = ROOT / "runs" / "_test_directive_root"
    (run_dir / "20260101-000000_t").mkdir(parents=True)
    (run_dir / "20260101-000000_t" / "epoch_007.pth").write_bytes(b"x")
    try:
        monkeypatch.chdir(tmp_path)
        argv = to_argv(
            Spec(
                name="t",
                train={
                    "model_path": "/x",
                    "start_epoch": {
                        "resume_after_latest_epoch": "runs/_test_directive_root"
                    },
                },
            )
        )
        assert values(argv, "--start_epoch") == ["8"]
    finally:
        shutil.rmtree(run_dir)
