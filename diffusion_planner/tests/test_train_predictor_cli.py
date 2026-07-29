import pytest
from train_predictor import get_args


def _required_args():
    return [
        "--exp_name",
        "test",
        "--save_dir",
        "out",
        "--train_set_list",
        "train.json",
        "--valid_set_list",
        "valid.json",
    ]


@pytest.mark.parametrize(
    "option, value, message",
    [
        ("--learning_rate", "nan", "must be finite"),
        ("--augment_prob", "1.1", r"must be in \[0, 1\]"),
        ("--ego_past_noise_std", "-1", "must be >= 0"),
        ("--multisample_eval_num_samples", "-1", "must be >= 0"),
        ("--closed_loop_search_radius", "0", "must be > 0"),
    ],
)
def test_training_cli_rejects_invalid_numeric_configuration(option, value, message):
    with pytest.raises(ValueError, match=message):
        get_args(_required_args() + [option, value])


def test_training_cli_accepts_hdp_defaults():
    args = get_args(_required_args())
    assert args.predicted_neighbor_num == 0
    assert args.future_len == 80
    assert args.ego_history_frames == 21
    assert args.use_velocity_representation is True
    assert args.diffusion_model_type == "x_start"
    assert args.align_legacy_neighbor_futures is False
    assert args.policy_uses_turn_indicator_history is False
    assert args.turn_indicator_output_dim == 3
    assert args.supervised_training_stage == "policy"
    assert args.turn_indicator_head_training_mode == "deployment"


def test_training_cli_selects_sequential_supervised_stages():
    for stage in ("policy", "turn_indicator"):
        args = get_args(_required_args() + ["--supervised_training_stage", stage])
        assert args.supervised_training_stage == stage


@pytest.mark.parametrize("mode", ("expert", "deployment"))
def test_training_cli_selects_turn_indicator_head_mode(mode):
    args = get_args(
        _required_args()
        + [
            "--supervised_training_stage",
            "turn_indicator",
            "--turn_indicator_head_training_mode",
            mode,
        ]
    )
    assert args.turn_indicator_head_training_mode == mode
