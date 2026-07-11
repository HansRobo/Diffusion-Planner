import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from timm.utils import ModelEmaV3


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


class ModelEma(ModelEmaV3):
    """Foreach EMA with the legacy ``.ema`` attribute used by our checkpoints."""

    def __init__(self, model, decay=0.9999, device=None):
        model_device = next(model.parameters()).device
        requested_device = torch.device(device) if device not in (None, "") else None
        if requested_device is not None and requested_device.type == model_device.type:
            if requested_device.index is None or requested_device.index == model_device.index:
                requested_device = None
        super().__init__(
            model,
            decay=decay,
            device=requested_device,
            foreach=requested_device is None,
        )

    @property
    def ema(self):
        return self.module


def capture_rng_state() -> dict:
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        # Pure primitives work with both distributed.gather_object and
        # torch.load(weights_only=True); raw NumPy arrays and tensors do not
        # survive every PyTorch object-gather serialization path.
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "pos": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state().tolist(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state().tolist()
    return state


def gather_rng_states() -> list[dict] | None:
    """Collect one RNG snapshot per DDP rank; only rank 0 receives the full list."""
    local_state = capture_rng_state()
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return [local_state]
    rank = torch.distributed.get_rank()
    gathered = [None] * torch.distributed.get_world_size() if rank == 0 else None
    torch.distributed.gather_object(local_state, gathered, dst=0)
    return gathered


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            np.asarray(numpy_state["state"], dtype=np.uint32),
            numpy_state["pos"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8, device="cpu"))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(torch.as_tensor(state["cuda"], dtype=torch.uint8, device="cpu"))


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
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return {}

    device = grads[0].device
    total = torch.zeros((), dtype=torch.float32, device=device)
    square_total = torch.zeros_like(total)
    l1 = torch.zeros_like(total)
    linf = torch.zeros_like(total)
    count = 0
    for grad in grads:
        values = grad.float()
        total += values.sum()
        square_total += values.square().sum()
        l1 += values.abs().sum()
        linf = torch.maximum(linf, values.abs().amax())
        count += values.numel()

    mean = total / count
    variance = (square_total - total.square() / count) / max(count - 1, 1)
    stats = torch.stack([l1, square_total.sqrt(), linf, mean, variance.clamp_min(0).sqrt()])
    l1_value, l2_value, linf_value, mean_value, std_value = stats.cpu().tolist()
    return {
        f"{prefix}/l1_norm": l1_value,
        f"{prefix}/l2_norm": l2_value,
        f"{prefix}/linf_norm": linf_value,
        f"{prefix}/mean": mean_value,
        f"{prefix}/std": std_value,
    }


def update_epoch_loss_sums(
    sums: dict[str, torch.Tensor | float],
    counts: dict[str, int],
    losses: dict[str, torch.Tensor | float],
) -> None:
    """Accumulate detached scalar metrics without retaining one CUDA tensor per step."""
    for key, value in losses.items():
        if torch.is_tensor(value):
            detached = value.detach()
            if detached.numel() != 1:
                raise ValueError(f"Epoch metric {key!r} must be scalar, got {detached.shape}")
            if key not in sums:
                sums[key] = torch.zeros_like(detached)
            sums[key].add_(detached)
        else:
            sums[key] = float(sums.get(key, 0.0)) + float(value)
        counts[key] = counts.get(key, 0) + 1


def finalize_epoch_loss_sums(
    sums: dict[str, torch.Tensor | float], counts: dict[str, int]
) -> dict[str, float]:
    if not counts:
        raise ValueError("Cannot average zero training steps")
    return {
        key: float((value / counts[key]).item()) if torch.is_tensor(value) else value / counts[key]
        for key, value in sums.items()
    }


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
    ckpt = torch.load(path, map_location=device, weights_only=True)

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

    rng_states = ckpt.get("rng_states") if isinstance(ckpt, dict) else None
    if rng_states is None and strict_training_state:
        raise RuntimeError(
            f"Strict resume checkpoint has no per-rank RNG state: {path}. "
            "Use it only as a weights initialization or resume from a checkpoint written "
            "by the current trainer."
        )
    if rng_states is not None:
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if distributed else 0
        world_size = torch.distributed.get_world_size() if distributed else 1
        if len(rng_states) != world_size:
            raise RuntimeError(
                f"Checkpoint has RNG states for {len(rng_states)} ranks, current world size is "
                f"{world_size}; exact resume requires the same DDP world size"
            )
        restore_rng_state(rng_states[rank])
        print("RNG state load done")

    return model, optimizer, scheduler, init_epoch, wandb_id, ema
