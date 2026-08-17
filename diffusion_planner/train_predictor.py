"""Train the trajectory predictor.

Launched under torch.distributed.run, normally via train_run.py:

    python train_run.py --exp_name my_exp \
        --train_set_list /path/to/train_list.json \
        --valid_set_list /path/to/valid_list.json

All flags are declared on :class:`diffusion_planner.train_config.TrainConfig` with
``cli(...)`` and mirrored on train_run.py.

Hydra Config Mode (optional):

    python train_predictor.py --config-name default \
        train_set_list=/path/to/train.json valid_set_list=/path/to/valid.json
"""

import os
import sys
from pathlib import Path

from diffusion_planner.train import model_training
from diffusion_planner.train_cli import build_parser, build_train_config

# Hydra imports (optional - graceful fallback if not installed)
try:
    import hydra
    from hydra import compose, initialize_config_dir
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    DictConfig = None


def main() -> None:
    args = build_parser(description=__doc__).parse_args()
    model_training(build_train_config(args))


def hydra_main() -> None:
    """Hydra config entrypoint - run with: python train_predictor.py --config-name default"""
    if not HYDRA_AVAILABLE:
        print("ERROR: Hydra is not installed. Install with: pip install hydra-core omegaconf")
        sys.exit(1)

    # Get config path relative to diffusion_planner package (parent of this file's directory)
    config_dir = Path(__file__).parent.parent / "configs"

    # Parse known args to get --config-name and --config-path
    parser = build_parser(description=__doc__)
    known, unknown = parser.parse_known_args()

    config_name = known.config_name or "default"
    config_path = known.config_path or str(config_dir)

    # Initialize Hydra
    with initialize_config_dir(config_dir=os.path.abspath(config_path), version_base=None):
        cfg: DictConfig = compose(config_name=config_name, overrides=unknown)

    # Build TrainConfig from Hydra config
    train_config = build_train_config(cfg)

    model_training(train_config)


if __name__ == "__main__":
    # Check if Hydra mode is requested
    if "--config-name" in sys.argv or "--config-path" in sys.argv:
        hydra_main()
    else:
        main()
