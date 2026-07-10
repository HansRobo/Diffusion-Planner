import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def atomic_torch_save(obj, path) -> None:
    """Write a checkpoint atomically so an interrupted save cannot corrupt the target."""
    target = Path(path)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    try:
        torch.save(obj, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def openjson(path):
    with open(path, "r", encoding="utf-8") as f:
        dict = json.load(f)
    return dict


def set_seed(CUR_SEED):
    random.seed(CUR_SEED)
    np.random.seed(CUR_SEED)
    torch.manual_seed(CUR_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_grad_stats(parameters, prefix="grad"):
    """
    Compute gradient statistics over all parameters to monitor
    vanishing/exploding gradients during training.

    The statistics are computed on the concatenation of every parameter's
    gradient (i.e. the global gradient vector):
        - L1 norm
        - L2 norm
        - Linf norm (max absolute value)
        - mean
        - standard deviation

    Args:
        parameters: iterable of model parameters (e.g. ``model.parameters()``).
        prefix: key prefix for the returned dictionary.

    Returns:
        dict mapping ``f"{prefix}/<stat>"`` to a python float. Empty dict if
        no parameter has a gradient.
    """
    grads = [p.grad.detach().flatten() for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return {}

    grads = torch.cat(grads)
    return {
        f"{prefix}/l1_norm": grads.abs().sum().item(),
        f"{prefix}/l2_norm": grads.norm(2).item(),
        f"{prefix}/linf_norm": grads.abs().max().item(),
        f"{prefix}/mean": grads.mean().item(),
        f"{prefix}/std": grads.std().item(),
    }


def get_epoch_mean_loss(epoch_loss):
    epoch_mean_loss = {}
    for current_loss in epoch_loss:
        for key, value in current_loss.items():
            if key in epoch_mean_loss:
                epoch_mean_loss[key].append(
                    value if isinstance(value, (int, float)) else value.item()
                )
            else:
                epoch_mean_loss[key] = [value if isinstance(value, (int, float)) else value.item()]

    for key, values in epoch_mean_loss.items():
        epoch_mean_loss[key] = np.mean(np.array(values))

    return epoch_mean_loss


def resume_model(
    path: str,
    model,
    optimizer,
    scheduler,
    ema,
    device,
    *,
    strict_training_state: bool = False,
):
    """Restore a checkpoint, optionally requiring every training-state component."""
    ckpt = torch.load(path, map_location=device)

    if strict_training_state and (not isinstance(ckpt, dict) or "model" not in ckpt):
        raise RuntimeError(f"Strict resume checkpoint has no model state: {path}")
    try:
        model.load_state_dict(ckpt["model"])
    except (KeyError, TypeError):
        model.load_state_dict(ckpt)
    print("Model load done")

    optimizer_state = ckpt.get("optimizer") if isinstance(ckpt, dict) else None
    if optimizer_state is None:
        if strict_training_state:
            raise RuntimeError(f"Strict resume checkpoint has no optimizer state: {path}")
        print("no pretrained optimizer found")
    else:
        try:
            optimizer.load_state_dict(optimizer_state)
            print("Optimizer load done")
        except Exception as exc:
            if strict_training_state:
                raise RuntimeError(f"Failed to restore optimizer state from {path}") from exc
            print(f"no compatible pretrained optimizer found: {exc}")

    if scheduler is not None:
        schedule_state = ckpt.get("schedule") if isinstance(ckpt, dict) else None
        if schedule_state is None:
            if strict_training_state:
                raise RuntimeError(f"Strict resume checkpoint has no scheduler state: {path}")
            print("no schedule found")
        else:
            try:
                scheduler.load_state_dict(schedule_state)
                print("Schedule load done")
            except Exception as exc:
                if strict_training_state:
                    raise RuntimeError(f"Failed to restore scheduler state from {path}") from exc
                print(f"no compatible schedule found: {exc}")

    if isinstance(ckpt, dict) and "epoch" in ckpt:
        init_epoch = int(ckpt["epoch"])
        print("Step load done")
    elif strict_training_state:
        raise RuntimeError(f"Strict resume checkpoint has no epoch: {path}")
    else:
        init_epoch = 0
    if isinstance(ckpt, dict) and "global_step" in ckpt:
        model._resume_global_step = int(ckpt["global_step"])

    wandb_id = ckpt.get("wandb_id") if isinstance(ckpt, dict) else None
    if wandb_id is not None:
        print("wandb id load done")

    if ema is not None:
        ema_state = ckpt.get("ema_state_dict") if isinstance(ckpt, dict) else None
        if ema_state is not None:
            ema.ema.load_state_dict(ema_state)
            ema.ema.eval()
            for parameter in ema.ema.parameters():
                parameter.requires_grad_(False)
            ema.loaded_from_checkpoint = True
            print("ema load done")
        elif strict_training_state:
            raise RuntimeError(f"Strict resume checkpoint has no EMA state: {path}")
        else:
            ema.ema.load_state_dict(model.state_dict())
            ema.ema.eval()
            for parameter in ema.ema.parameters():
                parameter.requires_grad_(False)
            ema.loaded_from_checkpoint = False
            print("no ema shadow found; initialized EMA from loaded model")

    return model, optimizer, scheduler, init_epoch, wandb_id, ema
