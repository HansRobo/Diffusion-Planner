import json
import random
from pathlib import Path

import numpy as np
import torch


def openjson(path):
    with open(path, "r", encoding="utf-8") as f:
        dict = json.load(f)
    return dict


def check_resume_args_compat(resume_model_path: str, current_args) -> list[str]:
    """Diff a resumed checkpoint's saved ``args.json`` (expected next to the .pth, same
    convention ``valid_predictor_closed_loop.py`` relies on) against the current run's args.

    Returns one ``"field: checkpoint=X current=Y"`` line per field present in both that
    differs — or a single explanatory line if no ``args.json`` was found. This can't tell
    which fields are architecture-critical (would break ``model.load_state_dict``) vs.
    intentionally different (e.g. ``learning_rate``, ``train_set_list``) — it just surfaces
    every difference so a real mismatch isn't a total surprise, and the caller decides
    whether/how loudly to warn.
    """
    args_json_path = Path(resume_model_path).parent / "args.json"
    if not args_json_path.is_file():
        return [f"no args.json found next to {resume_model_path} — cannot check config compatibility"]
    try:
        ckpt_args = openjson(args_json_path)
    except Exception as e:  # pragma: no cover - defensive, this is a diagnostic helper
        return [f"failed to read {args_json_path}: {e}"]

    current = vars(current_args) if not isinstance(current_args, dict) else current_args
    diffs: list[str] = []
    for key, ckpt_val in ckpt_args.items():
        if key not in current:
            continue
        cur_val = current[key]
        if not isinstance(cur_val, (bool, int, float, str, list, tuple, dict, type(None))):
            continue  # skip non-JSON-plain objects (normalizers etc.) — not comparable as-is
        if cur_val != ckpt_val:
            diffs.append(f"{key}: checkpoint={ckpt_val!r} current={cur_val!r}")
    return diffs


def set_seed(CUR_SEED):
    random.seed(CUR_SEED)
    np.random.seed(CUR_SEED)
    torch.manual_seed(CUR_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_grad_stats(parameters, prefix="grad"):
    """
    Compute global-gradient statistics (l1/l2/linf/mean/std) WITHOUT
    materializing a single concatenated copy of every gradient.

    torch.cat([...all grads...]) allocated one huge contiguous block at peak
    memory (right after backward), which was a primary driver of caching-allocator
    fragmentation. Here we accumulate the reductions per-parameter instead, and
    sync to host only once (a single 5-element .tolist()).
    """
    l1 = sq = smax = ssum = None
    n = 0
    for p in parameters:
        if p.grad is None:
            continue
        g = p.grad.detach()
        a = g.abs()
        g_l1 = a.sum()
        g_sq = (g * g).sum()
        g_max = a.max()
        g_sum = g.sum()
        l1 = g_l1 if l1 is None else l1 + g_l1
        sq = g_sq if sq is None else sq + g_sq
        smax = g_max if smax is None else torch.maximum(smax, g_max)
        ssum = g_sum if ssum is None else ssum + g_sum
        n += g.numel()

    if n == 0:
        return {}

    mean = ssum / n
    var = (sq / n - mean * mean).clamp_min(0)
    # Single device->host sync for all five scalars.
    l1_v, l2_v, linf_v, mean_v, std_v = torch.stack(
        [l1, sq.sqrt(), smax, mean, var.sqrt()]
    ).tolist()
    return {
        f"{prefix}/l1_norm": l1_v,
        f"{prefix}/l2_norm": l2_v,
        f"{prefix}/linf_norm": linf_v,
        f"{prefix}/mean": mean_v,
        f"{prefix}/std": std_v,
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


def resume_model(path: str, model, optimizer, scheduler, ema, device):
    """
    load ckpt from path
    """
    ckpt = torch.load(path, map_location=device)

    # load model
    try:
        model.load_state_dict(ckpt["model"])
    except:
        model.load_state_dict(ckpt)
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
        ema.ema.load_state_dict(ckpt["ema_state_dict"])
        ema.ema.eval()
        for p in ema.ema.parameters():
            p.requires_grad_(False)

        print("ema load done")
    except:
        print("no ema shadow found")

    return model, optimizer, scheduler, init_epoch, wandb_id, ema
