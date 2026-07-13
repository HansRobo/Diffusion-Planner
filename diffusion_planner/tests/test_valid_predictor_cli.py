import pytest
from valid_predictor import get_args


def _required_args():
    return [
        "--resume_model_path",
        "model.pth",
        "--args_json_path",
        "args.json",
    ]


def test_reward_eval_is_disabled_by_default():
    args = get_args(_required_args())

    assert args.reward_eval_num_generations == 0
    assert args.metrics_output is None


def test_reward_eval_cli_overrides_protocol_and_output():
    args = get_args(
        _required_args()
        + [
            "--reward_eval_num_generations",
            "32",
            "--reward_eval_noise_scale",
            "1.5",
            "--reward_eval_sample_steps",
            "6",
            "--reward_eval_w_safety",
            "3.0",
            "--metrics_output",
            "metrics.json",
        ]
    )

    assert args.reward_eval_num_generations == 32
    assert args.reward_eval_noise_scale == 1.5
    assert args.reward_eval_sample_steps == 6
    assert args.reward_eval_w_safety == 3.0
    assert args.metrics_output == "metrics.json"


def test_reward_eval_cli_exposes_road_border_weight_and_thresholds():
    args = get_args(
        _required_args()
        + [
            "--reward_eval_w_road_border",
            "1.5",
            "--reward_eval_road_border_critical_m",
            "0.15",
            "--reward_eval_road_border_safe_m",
            "0.55",
        ]
    )
    assert args.reward_eval_w_road_border == 1.5
    assert args.reward_eval_road_border_critical_m == 0.15
    assert args.reward_eval_road_border_safe_m == 0.55


@pytest.mark.parametrize(
    "options, message",
    [
        (["--reward_eval_num_generations", "-1"], "must be >= 0"),
        (
            ["--reward_eval_num_generations", "1", "--reward_eval_sample_steps", "1"],
            "must be >= 2",
        ),
        (["--reward_eval_noise_scale", "-0.1"], "must be >= 0"),
    ],
)
def test_reward_eval_cli_rejects_invalid_protocol(options, message):
    with pytest.raises(ValueError, match=message):
        get_args(_required_args() + options)
