import json
import random
from typing import Any

import numpy as np
import torch


def openjson(path):
    """Load a json file; transparently handles zstd-compressed ``*.zst`` files."""
    if str(path).endswith(".zst"):
        import io

        import zstandard

        with open(path, "rb") as f:
            reader = zstandard.ZstdDecompressor().stream_reader(f)
            return json.load(io.TextIOWrapper(reader, encoding="utf-8"))
    with open(path, "r", encoding="utf-8") as f:
        dict = json.load(f)
    return dict


def set_seed(CUR_SEED):
    random.seed(CUR_SEED)
    np.random.seed(CUR_SEED)
    torch.manual_seed(CUR_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_grad_linf_norm(parameters) -> torch.Tensor | None:
    """Largest absolute gradient element across all parameters, as a **device tensor**.

    Returns a 0-dim tensor rather than a Python float on purpose: train_epoch stashes it in
    the per-step loss dict and ``get_epoch_mean_loss`` does the single ``.item()`` at epoch
    end, alongside every other logged scalar. The version deleted in #10 returned floats,
    which forced one device->host sync per step -- that sync, not the arithmetic, was the
    cost being removed.

    Only linf is computed here. The matching L2 norm comes for free from
    ``clip_grad_norm_``'s return value (it is the *pre-clip* norm), and l1/mean/std were
    never actionable. linf is the one that reveals a single exploding element that an
    averaged norm hides.

    Returns None when no parameter has a gradient.
    """
    per_param = [p.grad.detach().abs().max() for p in parameters if p.grad is not None]
    if not per_param:
        return None
    return torch.stack(per_param).amax()


def _nonfinite_sample_mask(tensor: torch.Tensor) -> torch.Tensor | None:
    """Per-sample "contains a non-finite element" mask for a ``[B, ...]`` float tensor.

    Returns None for tensors the check does not apply to (scalars, integer tensors).
    """
    if tensor.dim() == 0 or not tensor.is_floating_point():
        return None
    bad = ~torch.isfinite(tensor)
    return bad if tensor.dim() == 1 else bad.flatten(1).any(dim=1)


def describe_nonfinite_step(
    *,
    grad_norm: torch.Tensor,
    losses: dict[str, Any],
    tensors: dict[str, torch.Tensor],
    parameters,
    sample_paths: list[str] | None = None,
    max_listed: int = 8,
) -> dict[str, Any]:
    """Diagnose a training step whose gradient norm went non-finite.

    Returns a dict with a human-readable ``"text"`` rendering plus the same findings as
    structured fields, so the caller can both print it and put it in a wandb table.

    Only called on the rare skip path, so it may sync freely.

    ``tensors`` is scanned per batch element so the offending NPZ files can be named --
    this is the whole point of the report. Note the scan sees the batch *after*
    augmentation and observation normalization: a NaN already present in the file survives
    both, but a NaN manufactured by augmentation is also reported here.

    Parameter gradients are inspected *after* ``clip_grad_norm_`` has run, which matters
    for reading the output: when the pre-clip norm was ``inf`` the clip scale is
    ``5/inf == 0``, so ``inf * 0 -> nan`` marks exactly the parameters that carried the
    inf while every other gradient is cleanly zeroed -- the listed names are the real
    source. When the pre-clip norm was already ``nan`` every gradient is nan and the list
    says nothing.
    """
    lines = [f"non-finite gradient norm ({grad_norm.item()}); step skipped"]

    bad_losses = [
        k
        for k, v in losses.items()
        if torch.is_tensor(v) and v.numel() == 1 and not torch.isfinite(v)
    ]
    lines.append(f"  non-finite loss terms: {bad_losses or 'none (loss was finite)'}")

    bad_samples: set[int] = set()
    bad_keys = []
    for key, value in tensors.items():
        if not torch.is_tensor(value):
            continue
        mask = _nonfinite_sample_mask(value)
        if mask is None or not bool(mask.any()):
            continue
        bad_keys.append(f"{key}(x{int((~torch.isfinite(value)).sum())})")
        bad_samples.update(mask.nonzero(as_tuple=True)[0].tolist())
    lines.append(f"  non-finite input tensors: {bad_keys or 'none'}")

    bad_paths: list[str] = []
    if bad_samples:
        listed = sorted(bad_samples)[:max_listed]
        if sample_paths is not None:
            bad_paths = [sample_paths[i] if i < len(sample_paths) else "?" for i in listed]
            lines.append(f"  offending NPZ files ({len(bad_samples)} in batch):")
            lines.extend(f"    {p}" for p in bad_paths)
        else:
            lines.append(f"  offending batch elements: {listed}")

    bad_params = [
        name
        for name, p in parameters
        if p.grad is not None and not bool(torch.isfinite(p.grad).all())
    ]
    lines.append(
        f"  parameters with non-finite grad (post-clip, see docstring): "
        f"{bad_params[:max_listed] or 'none'}"
        + (f" (+{len(bad_params) - max_listed} more)" if len(bad_params) > max_listed else "")
    )

    return {
        "text": "\n".join(lines),
        "grad_norm": float(grad_norm.item()),
        "bad_losses": bad_losses,
        "bad_inputs": bad_keys,
        "bad_sample_count": len(bad_samples),
        "bad_paths": bad_paths,
        "bad_params": bad_params[:max_listed],
    }


def assert_parameters_finite(model) -> None:
    """Raise if any parameter has gone non-finite.

    Once weights are NaN every subsequent step is skipped by the non-finite gradient guard
    and the run burns GPU hours producing nothing, so this fails loudly instead. One sync,
    called once per epoch.
    """
    bad = [name for name, p in model.named_parameters() if not bool(torch.isfinite(p).all())]
    if bad:
        raise RuntimeError(
            f"{len(bad)} parameter tensor(s) are non-finite; training cannot recover. "
            f"First few: {bad[:8]}"
        )


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
    return {
        k.replace("module.", "", 1) if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }


def resume_model(path: str, model, optimizer, scheduler, ema, device, use_ddp: bool = False):
    """
    load ckpt from path
    """
    ckpt = torch.load(path, map_location=device)

    # load model
    if use_ddp:
        try:
            model.load_state_dict(ckpt["model"])
        except:
            model.load_state_dict(ckpt)
    else:
        try:
            model.load_state_dict(strip_module_prefix(ckpt["model"]))
        except:
            model.load_state_dict(strip_module_prefix(ckpt))
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
        ema_state = ckpt["ema_state_dict"]
        if not use_ddp:
            ema_state = strip_module_prefix(ema_state)
        ema.ema.load_state_dict(ema_state)
        ema.ema.eval()
        for p in ema.ema.parameters():
            p.requires_grad_(False)

        print("ema load done")
    except:
        print("no ema shadow found")

    return model, optimizer, scheduler, init_epoch, wandb_id, ema
