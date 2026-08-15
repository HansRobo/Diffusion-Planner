"""Single-GPU training entry point driven by Hydra."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from diffusion_planner.utils.lr_scheduler import (
    build_lr_scheduler,
    describe_lr_scheduler,
)


def _move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True) for key, value in batch.items()
    }


def _autocast_context(device: torch.device, amp_dtype: str) -> Any:
    if device.type != "cuda" or amp_dtype == "none":
        return nullcontext()
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[amp_dtype]
    return torch.autocast(device_type="cuda", dtype=dtype)


@hydra.main(version_base=None, config_path="../../configs", config_name="train/train")
def main(config: DictConfig) -> None:
    """Train a diffusion planner for the configured number of optimizer steps."""
    print(OmegaConf.to_yaml(config))
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    configured_device = str(config.training.device)
    device = torch.device(
        "cuda" if configured_device == "auto" and torch.cuda.is_available() else
        "cpu" if configured_device == "auto" else configured_device
    )
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    loader = hydra.utils.instantiate(config.dataloader)
    if len(loader) == 0:
        raise RuntimeError(
            "dataloader has no batches; reduce batch_size or disable drop_last"
        )
    model = hydra.utils.instantiate(config.model).to(device)
    optimizer = hydra.utils.instantiate(
        config.optimizer,
        model=model,
        output_layers=(model.trajectory_decoder.output_projection,),
    )
    total_steps = int(config.training.total_steps)
    scheduler = build_lr_scheduler(optimizer, config.scheduler, total_steps)
    scheduler.step_update(0)

    print(f"training {len(loader.dataset)} frames on {device}")
    print(describe_lr_scheduler(config.scheduler, total_steps))
    model.train()
    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(total=total_steps, desc="training", unit="step")
    global_step = 0
    epoch = 0
    while global_step < total_steps:
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
        for batch in loader:
            batch = _move_batch(batch, device)
            with _autocast_context(device, str(config.training.amp_dtype)):
                loss = model.compute_loss(batch)

            loss.backward()
            max_grad_norm = float(config.training.max_grad_norm)
            if max_grad_norm > 0:
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )
            else:
                gradient_norm = torch.zeros((), device=device)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            scheduler.step_update(global_step)
            progress.update()
            if global_step % int(config.training.log_interval) == 0:
                progress.set_postfix(
                    loss=f"{loss.detach().item():.5f}",
                    grad=f"{gradient_norm.detach().item():.3f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    epoch=epoch,
                )
            if global_step >= total_steps:
                break
        epoch += 1
    progress.close()


if __name__ == "__main__":
    main()
