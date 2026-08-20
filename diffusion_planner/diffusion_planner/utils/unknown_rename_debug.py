"""Confirmation tooling for the training-time Unknown-class rename augmentation
(``rename_agents_to_unknown`` in ``data_augmentation.py``).

Two complementary ways to check what the augmentation is actually doing during a run:

1. Cheap per-step scalars (agent counts / rename rate) -- always computed, merged into the
   same loss dict already logged to stdout/wandb every step (see ``rename_stats``).
2. One before/after PNG per epoch, dumped on that epoch's LAST training step -- an actual
   picture of which agents got relabeled, with the ego vehicle (current pose + past +
   ground-truth future) and lane/route context so the scene reads on its own without
   needing to know the augmentation's internals. Off by default; enabled via
   ``unknown_rename_debug_dir``.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from diffusion_planner.utils.data_augmentation import rename_agents_to_unknown
from diffusion_planner.utils.visualize_input import (
    draw_ego_vehicle,
    draw_lanes,
    draw_neighbor_agents,
    draw_route,
)

_VIEW_RANGE_M = 60.0

# Matches draw_neighbor_agents' own vehicle_type = argmax(neighbor[8:12]) color mapping in
# visualize_input.py -- kept in sync manually since that function draws each neighbor via an
# unlabeled LineCollection/bounding box, so this legend can't be derived automatically.
_NEIGHBOR_TYPE_LEGEND = [
    Line2D([0], [0], color="blue", lw=2, label="Vehicle"),
    Line2D([0], [0], color="green", lw=2, label="Pedestrian"),
    Line2D([0], [0], color="purple", lw=2, label="Bicycle"),
    Line2D([0], [0], color="gray", lw=2, label="Unknown"),
]

_CONTEXT_KEYS = (
    "ego_current_state",
    "ego_shape",
    "ego_agent_past",
    "ego_agent_future",
    "lanes",
    "route_lanes",
)


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


def _context_from_inputs(inputs: dict, sample_idx: int) -> dict:
    """Slice+detach the scene-context keys draw_ego_vehicle/draw_lanes/draw_route need,
    matching the same to_numpy + batch-slice convention visualize_inputs uses. Identical
    across the before/after panels (only neighbor_agents_past changes), so this is built
    once per dump, not once per panel."""

    def one(key):
        v = inputs[key]
        v = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
        return v[sample_idx : sample_idx + 1]

    return {key: one(key) for key in _CONTEXT_KEYS}


def _draw_rename_panel(
    ax, neighbor_agents_past: np.ndarray, renamed: np.ndarray, context: dict, title: str
) -> None:
    """Draw one before/after panel: lane/route background, a labeled ego (current pose, past,
    ground-truth future), and neighbor agents colored by type with renamed agents circled."""
    draw_lanes(ax, context)
    draw_route(ax, context, color="olive", label="Route")
    ego_x, ego_y, _ego_state = draw_ego_vehicle(ax, context, show_future=True)
    draw_neighbor_agents(ax, {"neighbor_agents_past": neighbor_agents_past})

    last_t = neighbor_agents_past.shape[2] - 1
    any_circled = False
    for i in np.nonzero(renamed)[0]:
        x, y = neighbor_agents_past[0, i, last_t, 0], neighbor_agents_past[0, i, last_t, 1]
        if abs(x) + abs(y) < 1e-6:  # padding slot, nothing to circle
            continue
        ax.add_patch(plt.Circle((x, y), 3.0, fill=False, ec="red", lw=2, zorder=10))
        any_circled = True

    ax.set_title(title)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlim(ego_x - _VIEW_RANGE_M, ego_x + _VIEW_RANGE_M)
    ax.set_ylim(ego_y - _VIEW_RANGE_M, ego_y + _VIEW_RANGE_M)

    # draw_ego_vehicle/draw_route already set label= on their own past-trajectory/route lines;
    # everything else (current pose, GT future, neighbor-type colors, renamed marker) has no
    # automatic legend entry, so add proxy handles for those explicitly.
    handles, _labels = ax.get_legend_handles_labels()
    handles += [
        Line2D([0], [0], color="red", lw=2, label="Ego (current pose)"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="purple",
            markersize=6,
            label="Ego GT future (ground truth, not model output)",
        ),
        *_NEIGHBOR_TYPE_LEGEND,
    ]
    if any_circled:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markeredgecolor="red",
                markersize=10,
                markeredgewidth=2,
                label="Renamed to Unknown this step",
            )
        )
    ax.legend(handles=handles, fontsize=7, loc="upper left")


def _build_unknown_rename_debug_figure(
    before: torch.Tensor,
    after: torch.Tensor,
    renamed_mask: torch.Tensor,
    context: dict,
    sample_idx: int = 0,
) -> Figure:
    """Build (but don't save) the before/after debug figure. Split out from
    save_unknown_rename_debug_image so callers (tests) can inspect the figure -- legend
    contents, titles -- before it gets closed.

    before/after: [B, N, T, D] raw (pre-normalization) neighbor_agents_past (after is the
        tensor rename_agents_to_unknown returned; before must be a clone taken beforehand,
        since that function mutates in place).
    renamed_mask: [B, N] bool, as returned by rename_agents_to_unknown.
    context: dict with the raw (pre-normalization), un-sliced-by-sample scene tensors
        ego_current_state/ego_shape/ego_agent_past/ego_agent_future/lanes/route_lanes --
        identical across the before/after panels, see _context_from_inputs.
    sample_idx: which batch row to render (defaults to the first).
    """
    before_np = before[sample_idx : sample_idx + 1].detach().cpu().numpy()
    after_np = after[sample_idx : sample_idx + 1].detach().cpu().numpy()
    renamed = renamed_mask[sample_idx].detach().cpu().numpy()
    n_renamed = int(renamed.sum())

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    _draw_rename_panel(
        axes[0],
        before_np,
        renamed,
        context,
        f"Before rename ({n_renamed} agent(s) will be relabeled Unknown)",
    )
    _draw_rename_panel(
        axes[1],
        after_np,
        renamed,
        context,
        f"After rename ({n_renamed} agent(s) relabeled Unknown)",
    )
    fig.suptitle(
        "unknown_class_rename_prob augmentation check\n"
        "Left: neighbor_agents_past BEFORE rename_agents_to_unknown ran. Right: the SAME "
        "tensor AFTER. Red circles mark agents relabeled to the Unknown class this step."
    )
    return fig


def save_unknown_rename_debug_image(
    before: torch.Tensor,
    after: torch.Tensor,
    renamed_mask: torch.Tensor,
    context: dict,
    save_path: str,
    sample_idx: int = 0,
) -> None:
    """Save a before/after PNG of one batch sample's scene (see
    _build_unknown_rename_debug_figure for what's drawn and why)."""
    fig = _build_unknown_rename_debug_figure(before, after, renamed_mask, context, sample_idx)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.close(fig)


def apply_and_report_unknown_rename(
    inputs: dict,
    prob: float,
    *,
    debug_dir: str = "",
    is_last_step: bool = False,
    epoch: int = 0,
) -> tuple[torch.Tensor, dict]:
    """Apply rename_agents_to_unknown to inputs["neighbor_agents_past"] and return
    (tensor, stats) -- stats always includes the per-step rename count/rate; if debug_dir is
    set and this is the last training step of the epoch, also writes a before/after PNG
    there (one per epoch, not one per dump-cadence step). This is the call site helper
    train_epoch.py / grpo_epoch.py use so the confirmation logic lives in one place, not
    three.

    inputs: the raw (pre-normalization) training-step inputs dict -- read for
        neighbor_agents_past (mutated by rename_agents_to_unknown) and, only when actually
        dumping an image, ego_current_state/ego_shape/ego_agent_past/ego_agent_future/
        lanes/route_lanes (untouched, just read for context).
    """
    dump_image = bool(debug_dir) and is_last_step
    before = inputs["neighbor_agents_past"].clone() if dump_image else None

    neighbor_agents_past, renamed_mask, valid_mask = rename_agents_to_unknown(
        inputs["neighbor_agents_past"], prob
    )
    # Numeric only: this dict gets merged straight into the per-step loss dict, which
    # get_epoch_mean_loss averages key-by-key -- a string value here would break that.
    stats = rename_stats(renamed_mask, valid_mask)

    if dump_image and bool(renamed_mask.any()):
        context = _context_from_inputs(inputs, sample_idx=0)
        save_path = str(Path(debug_dir) / f"epoch{epoch:03d}.png")
        save_unknown_rename_debug_image(before, neighbor_agents_past, renamed_mask, context, save_path)
        print(f"[unknown_rename_debug] saved {save_path} ({stats['unknown_rename_count']} renamed)")

    return neighbor_agents_past, stats
