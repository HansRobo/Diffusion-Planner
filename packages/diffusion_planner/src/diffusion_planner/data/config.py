"""Hydra glue for the data configuration."""

from __future__ import annotations

from hydra.utils import instantiate
from omegaconf import DictConfig

from .planner_dataset import VehicleParameters


def build_vehicles(config: DictConfig) -> dict[str, VehicleParameters]:
    """Instantiate the project id to vehicle mapping.

    Config keys are not necessarily strings, so they are converted to the project ids that
    ``PlannerDataset`` matches against the index.
    """
    return {str(project): instantiate(node) for project, node in config.items()}
