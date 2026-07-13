from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class ValidConfig:
    # --- 必須パラメータ ---
    resume_model_path: str
    args_json_path: str

    # --- 上書き・推論用パラメータ ---
    valid_set_list: Optional[str] = None
    save_predictions_dir: Optional[str] = None
    metrics_output: Optional[str] = None

    # --- 実行環境パラメータ ---
    batch_size: int = 32
    num_workers: int = 4
    pin_mem: bool = True
    device: str = "cuda"
    seed: int = 3407
    future_len: int = 80
    agent_num: int = 32
    predicted_neighbor_num: int = 0
    ddp: bool = True
    compile_model: bool = True
    port: str = "22323"

    enable_epdms_eval: bool = True
    epdms_eval_use_agent_boxes: bool = True
    epdms_eval_use_road_border: bool = True
    multisample_eval_num_samples: int = 6
    multisample_eval_noise_scale: float = 0.1
    multisample_eval_sample_steps: int = 6
    multisample_eval_seed: int = 3407
    reward_eval_num_generations: int = 0
    reward_eval_noise_scale: float = 0.5
    reward_eval_sample_steps: int = 6
    reward_eval_w_safety: float = 0.0
    reward_eval_w_risk: float = 1.0
    reward_eval_w_follow: float = 3.0
    reward_eval_w_lane: float = 2.5
    reward_eval_w_progress: float = 3.0
    reward_eval_behavior_gate: Literal["none", "safety", "risk"] = "safety"
    # None inherits the exact data-alignment setting saved by training.
    align_legacy_neighbor_futures: Optional[bool] = None
