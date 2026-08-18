"""Confirmation tooling for the training-time Unknown-class rename augmentation
(``rename_agents_to_unknown`` in ``data_augmentation.py``).

Two complementary ways to check what the augmentation is actually doing during a run:

1. Cheap per-step scalars (agent counts / rename rate) -- always computed, merged into the
   same loss dict already logged to stdout/wandb every step (see ``rename_stats``).
2. Occasional before/after PNGs -- an actual picture of which agents got relabeled, so you
   can eyeball that the augmentation is hitting sensible agents and not, say, only ever
   picking the same slot. Off by default; enabled via ``unknown_rename_debug_dir``.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from diffusion_planner.utils.data_augmentation import rename_agents_to_unknown
from diffusion_planner.utils.visualize_input import draw_neighbor_agents


def rename_stats(renamed_mask: torch.Tensor, valid_mask: torch.Tensor) -> dict:
    """Per-batch scalar summary of one rename_agents_to_unknown call, meant to be merged
    straight into a training-step loss dict (get_epoch_mean_loss averages it like any other
    metric, and train.py's wandb.log already logs every loss-dict key)."""
    renamed = int(renamed_mask.sum().item())
    valid = int(valid_mask.sum().item())
    return {
        "unknown_rename_count": renamed,
        "unknown_rename_valid_count": valid,
        "unknown_rename_rate": renamed / valid if valid > 0 else 0.0,
    }


def save_unknown_rename_debug_image(
    before: torch.Tensor,
    after: torch.Tensor,
    renamed_mask: torch.Tensor,
    save_path: str,
    sample_idx: int = 0,
) -> None:
    """Save a before/after PNG of one batch sample's neighbor agents so a human can confirm
    which agents rename_agents_to_unknown relabeled -- the color flip to gray (Unknown) is
    also circled in red since two side-by-side panels can be easy to eyeball past.

    before/after: [B, N, T, D] raw (pre-normalization) neighbor_agents_past (after is the
        tensor rename_agents_to_unknown returned; before must be a clone taken beforehand,
        since that function mutates in place).
    renamed_mask: [B, N] bool, as returned by rename_agents_to_unknown.
    sample_idx: which batch row to render (defaults to the first).
    """
    before_np = before[sample_idx : sample_idx + 1].detach().cpu().numpy()
    after_np = after[sample_idx : sample_idx + 1].detach().cpu().numpy()
    renamed = renamed_mask[sample_idx].detach().cpu().numpy()
    last_t = before_np.shape[2] - 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, arr, title in ((axes[0], before_np, "before"), (axes[1], after_np, "after")):
        draw_neighbor_agents(ax, {"neighbor_agents_past": arr})
        ax.plot(0, 0, marker="s", color="black", markersize=8, zorder=11)  # ego at origin
        for i in np.nonzero(renamed)[0]:
            x, y = arr[0, i, last_t, 0], arr[0, i, last_t, 1]
            if abs(x) + abs(y) < 1e-6:  # padding slot, nothing to circle
                continue
            ax.add_patch(plt.Circle((x, y), 3.0, fill=False, ec="red", lw=2, zorder=10))
        ax.set_title(f"{title} ({int(renamed.sum())} renamed)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)

    fig.suptitle(f"unknown_class_rename_prob debug -- {Path(save_path).stem}")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.close(fig)


def apply_and_report_unknown_rename(
    neighbor_agents_past: torch.Tensor,
    prob: float,
    *,
    debug_dir: str = "",
    debug_every_n_steps: int = 200,
    step: int = 0,
    epoch: int = 0,
) -> tuple[torch.Tensor, dict]:
    """Apply rename_agents_to_unknown and return (tensor, stats) -- stats always includes the
    per-step rename count/rate; if debug_dir is set and this step lands on the dump cadence,
    also writes a before/after PNG there. This is the call site helper train_epoch.py /
    grpo_epoch.py use so the confirmation logic lives in one place, not three.
    """
    dump_image = bool(debug_dir) and step % max(debug_every_n_steps, 1) == 0
    before = neighbor_agents_past.clone() if dump_image else None

    neighbor_agents_past, renamed_mask, valid_mask = rename_agents_to_unknown(
        neighbor_agents_past, prob
    )
    # Numeric only: this dict gets merged straight into the per-step loss dict, which
    # get_epoch_mean_loss averages key-by-key -- a string value here would break that.
    stats = rename_stats(renamed_mask, valid_mask)

    if dump_image and bool(renamed_mask.any()):
        save_path = str(Path(debug_dir) / f"epoch{epoch:03d}_step{step:05d}.png")
        save_unknown_rename_debug_image(before, neighbor_agents_past, renamed_mask, save_path)
        print(f"[unknown_rename_debug] saved {save_path} ({stats['unknown_rename_count']} renamed)")

    return neighbor_agents_past, stats
