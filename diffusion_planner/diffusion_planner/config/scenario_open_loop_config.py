from dataclasses import dataclass


@dataclass
class ScenarioOpenLoopConfig:
    """Scenario-based open-loop validation configuration."""

    # ---------------------------------------------------------
    # Scenario-based open-loop validation
    # ---------------------------------------------------------
    scenario_based_open_loop_list: str = ""
    scenario_based_open_loop_only: bool = False

    # ---------------------------------------------------------
    # Scenario-based Open-loop
    # ---------------------------------------------------------
    scenario_centerline_horizon_seconds: float = 8.0
    scenario_departure_horizon_seconds: float = 3.0
    scenario_departure_minimum_displacement_m: float = 2.0
