"""Training entry point driven by Hydra."""

from __future__ import annotations

import hydra
import torch
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../../configs", config_name="train/train")
def main(config: DictConfig) -> None:
    """Build the dataset and stream one batch as a smoke test."""
    print(OmegaConf.to_yaml(config))
    torch.manual_seed(config.seed)

    # The dataloader config nests the dataset, which Hydra instantiates along with it.
    loader = hydra.utils.instantiate(config.dataloader)
    print(f"{len(loader.dataset)} frames")

    for batch in loader:
        if batch is None:  # every frame in this batch failed to build
            continue
        for key, value in sorted(batch.items()):
            print(f"{key:28s} {tuple(value.shape)}")
        break


if __name__ == "__main__":
    main()
