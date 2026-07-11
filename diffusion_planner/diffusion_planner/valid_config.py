from dataclasses import dataclass
from typing import Optional


@dataclass
class ValidConfig:
    # --- 必須パラメータ ---
    resume_model_path: str
    args_json_path: str

    # --- 上書き・推論用パラメータ ---
    valid_set_list: Optional[str] = None
    save_predictions_dir: Optional[str] = None

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
    # None inherits the exact data-alignment setting saved by training.
    align_legacy_neighbor_futures: Optional[bool] = None
