from dataclasses import dataclass

from .config_cli import cli


@dataclass
class ScenarioOpenLoopConfig:
    """Scenario-based open-loop validation configuration."""

    # ---------------------------------------------------------
    # Scenario-based open-loop validation
    # ---------------------------------------------------------
    scenario_based_open_loop_list: str = cli(
        "JSON matrix of scenario-based open-loop settings. Empty = disabled.",
        default="",
        path=True,
    )
    scenario_based_open_loop_only: bool = cli(
        "run the scenario-based open-loop validation and nothing else",
        default=False,
    )

    # ---------------------------------------------------------
    # Scenario-based Open-loop
    # ---------------------------------------------------------
    scenario_centerline_horizon_seconds: float = 8.0
    scenario_departure_horizon_seconds: float = 3.0
    scenario_departure_minimum_displacement_m: float = 2.0
