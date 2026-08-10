#!/usr/bin/env python3
"""Training entrypoint (single process; see train_run.py for the multi-GPU launcher).

Only the settings that have to vary per run are flags -- run with --help to see them.
Everything else is a field of TrainConfig in diffusion_planner/train_config.py and is
changed by editing its default there.
"""

from diffusion_planner.train import model_training
from diffusion_planner.train_cli import build_parser, build_train_config


def main() -> None:
    args = build_parser(description=__doc__).parse_args()
    model_training(build_train_config(args))


if __name__ == "__main__":
    main()
