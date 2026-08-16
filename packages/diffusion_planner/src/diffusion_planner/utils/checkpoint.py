"""Single-file training checkpoints for Accelerate-based training."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from timm.scheduler.scheduler import Scheduler
from torch import nn
from torch.optim import Optimizer


def _remove_state_prefix(state: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    """Remove a wrapper prefix when every model-state key contains it."""
    if state and all(key.startswith(prefix) for key in state):
        return {key.removeprefix(prefix): value for key, value in state.items()}
    return dict(state)


def save_checkpoint(
    accelerator: Accelerator,
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Scheduler,
    *,
    model_config: Mapping[str, Any],
    epoch: int,
    step_in_epoch: int,
    global_step: int,
    steps_per_epoch: int,
) -> None:
    """Atomically save model and training state from the main process."""
    checkpoint_path = Path(path)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    model_state = _remove_state_prefix(accelerator.get_state_dict(model), "_orig_mod.")
    state = {
        "model": model_state,
        "model_config": dict(model_config),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "global_step": global_step,
        "steps_per_epoch": steps_per_epoch,
        "world_size": accelerator.num_processes,
    }

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, temporary_path)
    temporary_path.replace(checkpoint_path)
    print(f"saved checkpoint: {checkpoint_path}")


def load_checkpoint(
    accelerator: Accelerator,
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Scheduler,
    *,
    steps_per_epoch: int,
) -> tuple[int, int]:
    """Restore training state and return the restart epoch and global step."""
    checkpoint_path = Path(path).expanduser()
    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )

    saved_steps_per_epoch = int(checkpoint["steps_per_epoch"])
    if saved_steps_per_epoch != steps_per_epoch:
        raise RuntimeError(
            "steps_per_epoch changed since the checkpoint was created: "
            f"saved={saved_steps_per_epoch}, current={steps_per_epoch}"
        )
    saved_world_size = int(checkpoint["world_size"])
    if saved_world_size != accelerator.num_processes:
        raise RuntimeError(
            "world size changed since the checkpoint was created: "
            f"saved={saved_world_size}, current={accelerator.num_processes}"
        )

    unwrapped_model = accelerator.unwrap_model(
        model, keep_fp32_wrapper=False, keep_torch_compile=False
    )
    model_state = _remove_state_prefix(checkpoint["model"], "_orig_mod.")
    unwrapped_model.load_state_dict(model_state)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    accelerator.wait_for_everyone()

    epoch = int(checkpoint["epoch"])
    global_step = int(checkpoint["global_step"])
    if accelerator.is_main_process:
        print(
            f"loaded checkpoint: {checkpoint_path} "
            f"(restart_epoch={epoch + 1}, global_step={global_step})"
        )
    return epoch, global_step
