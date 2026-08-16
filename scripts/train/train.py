"""Accelerate-based diffusion planner training entry point."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import hydra
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm.auto import tqdm

import wandb
from diffusion_planner.models.diffusion_planner import DiffusionPlanner
from diffusion_planner.utils.checkpoint import load_checkpoint, save_checkpoint
from diffusion_planner.utils.lr_scheduler import (
    build_lr_scheduler,
    describe_lr_scheduler,
)


class TrainingModel(nn.Module):
    """Expose planner loss through forward for distributed gradient synchronization."""

    def __init__(self, planner: DiffusionPlanner) -> None:
        super().__init__()
        self.planner = planner

    def forward(self, input_data: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute scalar flow-matching loss for one batch."""
        return self.planner.compute_loss(input_data)


@hydra.main(version_base=None, config_path="../../configs", config_name="train/train")
def main(config: DictConfig) -> None:
    """Train on one or more devices managed by Accelerate."""
    dynamo_backend = (
        str(config.training.compile_backend) if bool(config.training.compile) else "no"
    )
    accelerator = Accelerator(
        cpu=bool(config.training.cpu),
        mixed_precision=str(config.training.mixed_precision),
        split_batches=False,
        dynamo_backend=dynamo_backend,
    )
    set_seed(int(config.seed), device_specific=True)

    run_name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{config.experiment_name}"
    checkpoint_dir = Path(str(config.checkpoint.output_dir)) / run_name

    if accelerator.is_main_process:
        wandb_config = OmegaConf.to_container(
            config, resolve=True, throw_on_missing=True
        )
        wandb.init(name=run_name, config=wandb_config)  # type: ignore

    loader = hydra.utils.instantiate(config.dataloader)
    if len(loader) == 0:
        raise RuntimeError(
            "dataloader has no batches; reduce batch_size or disable drop_last"
        )

    planner: DiffusionPlanner = hydra.utils.instantiate(config.model)
    training_model = TrainingModel(planner)
    if accelerator.is_main_process:
        parameter_count = sum(
            parameter.numel() for parameter in training_model.parameters()
        )
        trainable_parameter_count = sum(
            parameter.numel()
            for parameter in training_model.parameters()
            if parameter.requires_grad
        )
        print(
            f"parameters={parameter_count / 1e6:.2f}M "
            f"trainable={trainable_parameter_count / 1e6:.2f}M"
        )
    optimizer = hydra.utils.instantiate(
        config.optimizer,
        model=training_model,
        output_layers=(planner.trajectory_decoder.output_projection,),
        verbose=accelerator.is_main_process,
    )
    training_model, optimizer, loader = accelerator.prepare(
        training_model, optimizer, loader
    )
    total_epochs = int(config.training.total_epochs)
    steps_per_epoch = len(loader)
    total_steps = total_epochs * steps_per_epoch
    scheduler = build_lr_scheduler(optimizer, config.scheduler, total_steps)
    scheduler.step_update(0)

    start_epoch = 0
    global_step = 0
    if config.checkpoint.resume_from is not None:
        start_epoch, global_step = load_checkpoint(
            accelerator,
            str(config.checkpoint.resume_from),
            training_model,
            optimizer,
            scheduler,
            steps_per_epoch=steps_per_epoch,
        )

    if accelerator.is_main_process:
        print(
            f"training {len(loader.dataset)} frames on "
            f"{accelerator.num_processes} process(es), configured batch size "
            f"{config.dataloader.batch_size}"
        )
        print(
            f"epochs={total_epochs} steps_per_epoch={steps_per_epoch} "
            f"total_steps={total_steps}"
        )
        print(f"torch_compile={bool(config.training.compile)} backend={dynamo_backend}")
        print(describe_lr_scheduler(config.scheduler, total_steps))
        print(f"run_name={run_name} checkpoint_dir={checkpoint_dir}")

    training_model.train()
    optimizer.zero_grad(set_to_none=True)
    log_interval = int(config.training.log_interval)
    latest_interval = int(config.checkpoint.latest_interval_steps)
    for epoch in range(start_epoch, total_epochs):
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)
        progress = tqdm(
            loader,
            desc=f"epoch {epoch + 1}/{total_epochs}",
            disable=not accelerator.is_main_process,
            dynamic_ncols=True,
        )
        for step_in_epoch, batch in enumerate(progress, start=1):
            loss = training_model(batch)
            accelerator.backward(loss)
            gradient_norm = accelerator.clip_grad_norm_(
                training_model.parameters(), float(config.training.max_grad_norm)
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if accelerator.optimizer_step_was_skipped:
                continue

            global_step += 1
            scheduler.step_update(global_step)
            if global_step % log_interval == 0:
                gradient_norm_value = (
                    gradient_norm.detach().float()
                    if gradient_norm is not None
                    else torch.full((), torch.nan, device=loss.device)
                )
                metrics = accelerator.reduce(
                    torch.stack((loss.detach().float(), gradient_norm_value)),
                    reduction="mean",
                )
                metric_values = {
                    "train/loss": metrics[0].item(),
                    "train/grad_norm": metrics[1].item(),
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                }
                if accelerator.is_main_process:
                    progress.set_postfix(
                        loss=f"{metric_values['train/loss']:.5f}",
                        lr=f"{metric_values['train/learning_rate']:.2e}",
                    )
                    wandb.log(metric_values, step=global_step)
                    print(
                        f"step={global_step}/{total_steps} "
                        f"loss={metric_values['train/loss']:.5f} "
                        f"grad_norm={metric_values['train/grad_norm']:.3f} "
                        f"lr={metric_values['train/learning_rate']:.2e}"
                    )
            if accelerator.is_main_process and global_step % latest_interval == 0:
                save_checkpoint(
                    accelerator,
                    checkpoint_dir / "latest.pth",
                    training_model,
                    optimizer,
                    scheduler,
                    epoch=epoch,
                    step_in_epoch=step_in_epoch,
                    global_step=global_step,
                    steps_per_epoch=steps_per_epoch,
                )
        if accelerator.is_main_process:
            save_checkpoint(
                accelerator,
                checkpoint_dir / f"epoch_{epoch + 1:04d}.pth",
                training_model,
                optimizer,
                scheduler,
                epoch=epoch + 1,
                step_in_epoch=0,
                global_step=global_step,
                steps_per_epoch=steps_per_epoch,
            )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
