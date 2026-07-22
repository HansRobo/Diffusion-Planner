"""Verify RouteAugmentation visually and numerically.

Loads npz samples, applies route tail truncation (truncation_prob=1.0, with a
deterministic min==max budget) and the scene-wide speed-limit unknown dropout
(prob=1.0), renders original vs augmented side by side with visualize_inputs,
and runs numeric invariant checks (kept segments untouched, dropped segments
and their speed-limit rows zeroed, dropped set is a suffix starting beyond the
budget, first segment always kept, non-route tensors untouched, prob=0
identity). Exits non-zero if any check fails.

Usage (from the diffusion_planner directory):
    uv run python util_scripts/visualize_route_augmentation.py <npz> [<npz> ...] \
        --out_dir <dir> [--truncate_m 100]
"""

import argparse
import copy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.utils.route_augmentation import RouteAugmentation
from diffusion_planner.utils.visualize_input import visualize_inputs

_GEOM_DIM = 8


def load_inputs(npz_path: Path) -> dict:
    """Build the batch dict exactly like train_epoch right before route_aug is applied."""
    loaded = np.load(npz_path)
    data = {}
    for key, value in loaded.items():
        if key in ("map_name", "token", "version"):
            continue
        t = torch.tensor(np.expand_dims(value, axis=0))
        if t.dtype == torch.float64:
            t = t.float()
        data[key] = t
    data["ego_agent_past"] = heading_to_cos_sin(data["ego_agent_past"])
    data["goal_pose"] = heading_to_cos_sin(data["goal_pose"])
    return data


def segment_start_distances(route: torch.Tensor) -> torch.Tensor:
    """Arc length from the route start to each segment's start, as in the aug."""
    valid_pt = route[..., :_GEOM_DIM].abs().sum(-1) > 0
    xy = route[..., :2]
    step = (xy[:, :, 1:] - xy[:, :, :-1]).norm(dim=-1)
    pair_valid = (valid_pt[:, :, 1:] & valid_pt[:, :, :-1]).to(step.dtype)
    seg_len = (step * pair_valid).sum(-1)
    return torch.cumsum(seg_len, dim=1) - seg_len


def run_checks(orig: dict, aug: dict, sl_aug: dict, truncate_m: float) -> list[str]:
    failures = []

    def check(name, cond):
        if not cond:
            failures.append(name)

    route_o = orig["route_lanes"]
    route_a = aug["route_lanes"]
    valid_o = route_o.abs().sum(dim=(2, 3)) > 0  # [B, P]
    kept = route_a.abs().sum(dim=(2, 3)) > 0  # [B, P]

    # 1. Expected drop set: valid segments starting at/beyond the budget (never seg 0).
    cum_before = segment_start_distances(route_o)
    expected_drop = (cum_before >= truncate_m) & valid_o
    expected_drop[:, 0] = False
    check("drop set matches budget", torch.equal(kept, valid_o & ~expected_drop))
    check("first segment kept", bool(kept[0, 0] == valid_o[0, 0]))

    # 2. Kept segments are bit-identical; dropped rows fully zeroed.
    check(
        "kept segments untouched",
        torch.equal(route_a[valid_o & ~expected_drop], route_o[valid_o & ~expected_drop]),
    )
    check("dropped segments zeroed", bool(route_a[expected_drop].abs().sum() == 0))

    # 3. Speed-limit tensors follow the dropped rows.
    check(
        "dropped speed_limit zeroed",
        bool(aug["route_lanes_speed_limit"][expected_drop].abs().sum() == 0),
    )
    check(
        "dropped has_speed_limit false",
        bool(~aug["route_lanes_has_speed_limit"][expected_drop].any()),
    )

    # 4. Truncation must not touch anything but the three route tensors.
    route_keys = {"route_lanes", "route_lanes_speed_limit", "route_lanes_has_speed_limit"}
    for key in orig:
        if key in route_keys:
            continue
        check(f"non-route key untouched ({key})", torch.equal(aug[key], orig[key]))

    # 5. Speed-limit unknown dropout (prob=1): lanes and route cleared together,
    #    geometry untouched.
    for prefix in ("lanes", "route_lanes"):
        check(
            f"sl-dropout {prefix} values zero",
            bool(sl_aug[f"{prefix}_speed_limit"].abs().sum() == 0),
        )
        check(f"sl-dropout {prefix} flags false", bool(~sl_aug[f"{prefix}_has_speed_limit"].any()))
    check("sl-dropout geometry untouched", torch.equal(sl_aug["route_lanes"], orig["route_lanes"]))
    check("sl-dropout lanes untouched", torch.equal(sl_aug["lanes"], orig["lanes"]))

    # 6. prob=0 is the identity.
    ident = RouteAugmentation(device="cpu", truncation_prob=0.0, speed_limit_unknown_prob=0.0)(
        copy.deepcopy(orig)
    )
    for key in orig:
        check(f"prob=0 identity ({key})", torch.equal(ident[key], orig[key]))

    return failures


def _valid_segment_points(route: torch.Tensor, seg: int) -> torch.Tensor:
    """Valid (non-padding) xy points of one route segment. route: [B, P, V, D]."""
    pts = route[0, seg]
    valid = pts[..., :_GEOM_DIM].abs().sum(-1) > 0
    return pts[valid][:, :2]


def render_truncation(orig, truncated, budget_m, status, npz_name, out_path):
    """Original vs truncated on the scene render.

    On the left panel the segments that the truncation removes are overdrawn in
    red (with their start arc-length), the kept ones in green, so the exact
    difference is visible even when the cut lies outside the view window.
    """
    route_o = orig["route_lanes"]
    valid_o = (route_o.abs().sum(dim=(2, 3)) > 0)[0]
    kept = (truncated["route_lanes"].abs().sum(dim=(2, 3)) > 0)[0]
    cum_before = segment_start_distances(route_o)[0]
    n_orig, n_kept = int(valid_o.sum()), int(kept.sum())

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    visualize_inputs(copy.deepcopy(orig), ax=axes[0])
    visualize_inputs(copy.deepcopy(truncated), ax=axes[1])
    for seg in torch.where(valid_o)[0].tolist():
        xy = _valid_segment_points(route_o, seg)
        is_kept = bool(kept[seg])
        color = "tab:green" if is_kept else "tab:red"
        axes[0].plot(xy[:, 0], xy[:, 1], color=color, lw=3.5, alpha=0.6, zorder=20)
        axes[0].annotate(
            f"{cum_before[seg]:.0f}m",
            xy=(float(xy[0, 0]), float(xy[0, 1])),
            fontsize=9,
            color=color,
            fontweight="bold",
            zorder=21,
        )
        if is_kept:
            axes[1].plot(xy[:, 0], xy[:, 1], color="tab:green", lw=3.5, alpha=0.6, zorder=20)
    axes[0].set_title(
        f"original ({n_orig} route segments)  "
        f"green=kept / red=dropped by {budget_m:.0f}m budget (label: start arc length)"
    )
    axes[1].set_title(f"after truncation @ {budget_m:.0f}m ({n_kept} route segments)")
    fig.suptitle(
        f"route tail truncation   {npz_name}   [{status}]",
        color="red" if "NG" in status else "green",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def _render_speed_limit_panel(ax, inputs, title):
    """Draw lane / route centerlines colored by their speed limit.

    Blue->red colormap = limit value; gray dashed = unknown (has_speed_limit
    False). Route segments are thicker and annotated with the value in m/s.
    """
    cmap = plt.get_cmap("coolwarm")
    vmax = max(
        float(inputs["lanes_speed_limit"].max()),
        float(inputs["route_lanes_speed_limit"].max()),
        1.0,
    )
    for key, lw in (("lanes", 1.5), ("route_lanes", 4.0)):
        tensor = inputs[key]
        limit = inputs[f"{key}_speed_limit"][0, :, 0]
        has = inputs[f"{key}_has_speed_limit"][0, :, 0]
        valid = (tensor.abs().sum(dim=(2, 3)) > 0)[0]
        for seg in torch.where(valid)[0].tolist():
            xy = _valid_segment_points(tensor, seg)
            if bool(has[seg]):
                color = cmap(float(limit[seg]) / vmax)
                ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw, zorder=10)
            else:
                ax.plot(xy[:, 0], xy[:, 1], color="gray", lw=lw, ls="--", alpha=0.7, zorder=10)
            if key == "route_lanes":
                mid = xy[len(xy) // 2]
                label = f"{float(limit[seg]):.1f}m/s" if bool(has[seg]) else "unknown"
                ax.annotate(
                    label,
                    xy=(float(mid[0]), float(mid[1])),
                    fontsize=9,
                    fontweight="bold",
                    color="black" if bool(has[seg]) else "dimgray",
                    zorder=11,
                )
    ax.plot(0, 0, marker="*", color="black", markersize=12, zorder=12)  # ego
    n_known = int(
        inputs["lanes_has_speed_limit"].sum() + inputs["route_lanes_has_speed_limit"].sum()
    )
    ax.set_title(f"{title} (known speed limits: {n_known})")
    ax.set_aspect("equal")
    ax.set_xlim(-80, 80)
    ax.set_ylim(-80, 80)
    ax.grid(alpha=0.3)


def render_speed_limit(orig, sl_dropped, status, npz_name, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    _render_speed_limit_panel(axes[0], orig, "original: color = speed limit, dashed gray = unknown")
    _render_speed_limit_panel(axes[1], sl_dropped, "after unknown dropout (lanes + route together)")
    fig.suptitle(
        f"speed-limit unknown dropout   {npz_name}   [{status}]",
        color="red" if "NG" in status else "green",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_paths", type=Path, nargs="+")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--truncate_m",
        type=float,
        default=100.0,
        help="deterministic truncation budget (min == max) in meters",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    trunc_aug = RouteAugmentation(
        device="cpu",
        truncation_prob=1.0,
        truncation_min_m=args.truncate_m,
        truncation_max_m=args.truncate_m,
        speed_limit_unknown_prob=0.0,
    )
    sl_only_aug = RouteAugmentation(device="cpu", truncation_prob=0.0, speed_limit_unknown_prob=1.0)

    any_fail = False
    for npz_path in args.npz_paths:
        orig = load_inputs(npz_path)
        truncated = trunc_aug(copy.deepcopy(orig))
        sl_dropped = sl_only_aug(copy.deepcopy(orig))

        failures = run_checks(orig, truncated, sl_dropped, args.truncate_m)
        n_orig = int((orig["route_lanes"].abs().sum(dim=(2, 3)) > 0).sum())
        n_kept = int((truncated["route_lanes"].abs().sum(dim=(2, 3)) > 0).sum())
        status = "OK" if not failures else "NG: " + ", ".join(failures)
        print(f"{npz_path.name}: {status} (route segments {n_orig} -> {n_kept})")
        any_fail |= bool(failures)

        # One before/after figure per augmentation type.
        out_trunc = args.out_dir / f"{npz_path.stem}_truncation_check.png"
        render_truncation(orig, truncated, args.truncate_m, status, npz_path.name, out_trunc)
        print(f"  -> {out_trunc}")
        out_sl = args.out_dir / f"{npz_path.stem}_speed_limit_check.png"
        render_speed_limit(orig, sl_dropped, status, npz_path.name, out_sl)
        print(f"  -> {out_sl}")

    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
