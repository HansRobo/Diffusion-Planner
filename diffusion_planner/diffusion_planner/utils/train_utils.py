import json
import random
from typing import Any

import numpy as np
import torch


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


def strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Remove DDP ``module.`` prefix from checkpoint keys."""
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state_dict.items()}


def extract_model_state_dict(ckpt: dict[str, Any] | Any) -> dict[str, Any]:
    """Return model weights from a training checkpoint or raw state dict."""
    if isinstance(ckpt, dict) and "model" in ckpt:
        return strip_module_prefix(ckpt["model"])
    if isinstance(ckpt, dict):
        return strip_module_prefix(ckpt)
    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")


def train_config_defaults_from_args_json(path: str) -> dict[str, Any]:
    """Load saved training hyperparameters for argparse defaults (architecture + dims)."""
    from diffusion_planner.train_config import TrainConfig

    data = openjson(path)
    skip = frozenset({"state_normalizer", "observation_normalizer"})
    defaults: dict[str, Any] = {}
    for key in TrainConfig.__dataclass_fields__:
        if key in skip:
            continue
        if key in data:
            defaults[key] = data[key]
    return defaults


def resume_model(path: str, model, optimizer, scheduler, ema, device):
    """
    load ckpt from path
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(extract_model_state_dict(ckpt))
    print("Model load done")

    # load optimizer
    try:
        optimizer.load_state_dict(ckpt["optimizer"])
        print("Optimizer load done")
    except:
        print("no pretrained optimizer found")

    # load schedule
    try:
        scheduler.load_state_dict(ckpt["schedule"])
        print("Schedule load done")
    except:
        print("no schedule found,")

    # load step
    try:
        init_epoch = ckpt["epoch"]
        print("Step load done")
    except:
        init_epoch = 0

    # Load wandb id
    try:
        wandb_id = ckpt["wandb_id"]
        print("wandb id load done")
    except:
        wandb_id = None

    try:
        ema.ema.load_state_dict(strip_module_prefix(ckpt["ema_state_dict"]))
        ema.ema.eval()
        for p in ema.ema.parameters():
            p.requires_grad_(False)

        print("ema load done")
    except:
        print("no ema shadow found")

    return model, optimizer, scheduler, init_epoch, wandb_id, ema
