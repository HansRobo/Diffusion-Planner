from diffusion_planner.utils import ddp
from dataclasses import fields
from typing import Any
import json
from pathlib import Path


def save_config(cfg: Any, out_root: str | Path, filename: str) -> None:
    if not filename.endswith(".json"):
        filename += ".json"
    out_root = Path(out_root)
    if ddp.get_rank() == 0:
        config_dict = {f.name: getattr(cfg, f.name) for f in fields(cfg) if f.repr}
        with open(out_root / filename, "w") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
