"""Train the trajectory predictor.

Launched under torch.distributed.run, normally via train_run.py:

    python train_run.py --exp_name my_exp \
        --train_set_list /path/to/train_list.json \
        --valid_set_list /path/to/valid_list.json

All flags are declared on :class:`diffusion_planner.train_config.TrainConfig` with
``cli(...)`` and mirrored on train_run.py.
"""

from diffusion_planner.train import model_training
from diffusion_planner.train_cli import build_parser, build_train_config


def main() -> None:
    args = build_parser(description=__doc__).parse_args()
    model_training(build_train_config(args))


if __name__ == "__main__":
    main()
