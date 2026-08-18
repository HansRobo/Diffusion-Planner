from dataclasses import dataclass


@dataclass
class BaseConfig:
    """Shared execution config used by both TrainConfig and ValidConfig."""

    seed: int = 3407
    ddp: bool = True
    port: str = "22323"
    device: str = "cuda"

    enable_temporal_stability_eval: bool = True
    enable_replan_consistency_eval: bool = True
    replan_consistency_expected_gap: int = 1
    enable_epdms_eval: bool = True
    enable_pdms_eval: bool = False
    epdms_eval_use_agent_boxes: bool = True
    epdms_eval_use_road_border: bool = True

    scenario_based_open_loop_list: str = ""
    scenario_based_open_loop_only: bool = False
