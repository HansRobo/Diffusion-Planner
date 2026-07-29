import json

import pytest

from rlvr.posttrain.spec import load, merge


def write(path, **body):
    path.write_text(json.dumps(body))
    return path


def test_extends_merges_and_null_deletes(tmp_path):
    write(tmp_path / "base.json", train={"a": 1, "b": 2, "seed": 3407}, launch={"nproc": 8})
    child = write(
        tmp_path / "child.json",
        extends="base.json",
        train={"b": 20, "c": 30, "seed": None},
    )
    spec = load(child)[0]
    assert spec.train == {"a": 1, "b": 20, "c": 30}
    assert spec.nproc == 8


def test_extends_is_chainable_and_bounded(tmp_path):
    write(tmp_path / "a.json", train={"x": 1})
    write(tmp_path / "b.json", extends="a.json", train={"y": 2})
    spec = load(write(tmp_path / "c.json", extends="b.json", train={"z": 3}))[0]
    assert spec.train == {"x": 1, "y": 2, "z": 3}

    loop = write(tmp_path / "loop.json", extends="loop.json", train={})
    with pytest.raises(ValueError, match="extends chain"):
        load(loop)


def test_missing_parent_is_reported_with_both_paths(tmp_path):
    child = write(tmp_path / "child.json", extends="nope.json", train={})
    with pytest.raises(FileNotFoundError, match="nope.json"):
        load(child)


def test_env_and_nproc_come_from_launch(tmp_path):
    spec = load(
        write(
            tmp_path / "s.json",
            launch={"nproc": 4, "env": {"NVIDIA_TF32_OVERRIDE": 0}},
            train={},
        )
    )[0]
    assert spec.nproc == 4
    assert spec.env == {"NVIDIA_TF32_OVERRIDE": "0"}


def test_sweep_expands_to_the_cartesian_product(tmp_path):
    spec = write(
        tmp_path / "s.json",
        name="arm",
        train={"expert_anchor_weight": 0.4},
        sweep={"expert_anchor_weight": [0.2, 0.8], "awr_beta": [1, 2]},
    )
    specs = load(spec)
    assert len(specs) == 4
    assert {(s.train["expert_anchor_weight"], s.train["awr_beta"]) for s in specs} == {
        (0.2, 1), (0.2, 2), (0.8, 1), (0.8, 2)
    }
    assert len({s.name for s in specs}) == 4


def test_sweep_arms_do_not_share_an_output_dir(tmp_path):
    specs = load(
        write(
            tmp_path / "s.json",
            name="arm",
            train={"output_dir": "/out/arm", "exp_name": "arm", "wandb_run_name": "arm"},
            sweep={"awr_beta": [1, 2]},
        )
    )
    for key in ("output_dir", "exp_name", "wandb_run_name"):
        assert len({s.train[key] for s in specs}) == 2


def test_no_sweep_leaves_the_name_alone(tmp_path):
    spec = load(write(tmp_path / "s.json", name="arm", train={"output_dir": "/out"}))[0]
    assert spec.name == "arm" and spec.train["output_dir"] == "/out"


def test_sweep_values_must_be_non_empty_lists(tmp_path):
    with pytest.raises(ValueError, match="non-empty list"):
        load(write(tmp_path / "s.json", train={}, sweep={"awr_beta": 1}))


def test_merge_does_not_mutate_its_inputs():
    base = {"train": {"a": 1}}
    merge(base, {"train": {"b": 2}})
    assert base == {"train": {"a": 1}}


def test_reward_fields_live_in_their_own_section(tmp_path):
    spec = load(
        write(tmp_path / "s.json", reward={"low_speed_steer_penalty": 2.0}, train={"a": 1})
    )[0]
    assert spec.reward == {"low_speed_steer_penalty": 2.0}
    assert "low_speed_steer_penalty" not in spec.train


def test_a_reward_prefixed_sweep_varies_the_reward(tmp_path):
    specs = load(
        write(
            tmp_path / "s.json",
            name="arm",
            reward={"low_speed_steer_penalty": 1.0},
            train={"awr_beta": 1},
            sweep={"reward.low_speed_steer_penalty": [1.0, 3.0]},
        )
    )
    assert [s.reward["low_speed_steer_penalty"] for s in specs] == [1.0, 3.0]
    assert all("low_speed_steer_penalty" not in s.train for s in specs)
    assert specs[1].name == "arm_low_speed_steer_penalty3p0"


def test_project_relative_paths_resolve_against_the_repo(tmp_path):
    from rlvr.posttrain.spec import ROOT

    spec = load(
        write(
            tmp_path / "s.json",
            train={
                "config": "rlvr/configs/x.json",
                "model_path": {"latest_epoch": "outputs/run"},
                "exp_name": "not/a/path",
            },
        )
    )[0]
    assert spec.train["config"] == str(ROOT / "rlvr/configs/x.json")
    assert spec.train["model_path"] == {"latest_epoch": str(ROOT / "outputs/run")}
    assert spec.train["exp_name"] == "not/a/path"
    assert (ROOT / "rlvr/posttrain").is_dir()


def test_an_absolute_path_is_left_alone(tmp_path):
    spec = load(write(tmp_path / "s.json", train={"config": "/data/x.json"}))[0]
    assert spec.train["config"] == "/data/x.json"


def test_prepare_takes_its_values_from_the_arm(tmp_path):
    from rlvr.posttrain.spec import ROOT

    specs = load(
        write(
            tmp_path / "s.json",
            name="arm",
            train={"awr_beta": 1, "resume_replay_root": "outputs/overlay_b{awr_beta}"},
            sweep={"awr_beta": [0.5, 2.0]},
            prepare=[{"creates": "{resume_replay_root}", "argv": ["build", "{creates}", "{name}"]}],
        )
    )
    # One definition of the per-arm overlay: the flag and its pre-step agree.
    for spec in specs:
        assert spec.prepare[0]["argv"][1] == spec.train["resume_replay_root"]
    assert specs[0].train["resume_replay_root"] == str(ROOT / "outputs/overlay_b0.5")
    assert specs[0].prepare[0]["argv"][2] == "arm_awr_beta0p5"


def test_a_prepare_placeholder_with_no_value_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="no value for"):
        load(write(tmp_path / "s.json", train={}, prepare=[{"argv": ["build", "{nope}"]}]))


def test_a_prepare_entry_needs_an_argv(tmp_path):
    with pytest.raises(ValueError, match="non-empty argv"):
        load(write(tmp_path / "s.json", train={}, prepare=[{"creates": "x"}]))
