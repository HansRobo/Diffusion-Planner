"""Verify LaneAugmentation visually and numerically.

Loads npz samples and, per augmentation type, applies the component with
prob=1 and a deterministic setting, renders a before/after figure, and runs
numeric invariant checks (route-copied lanes untouched by every component,
truncation radius correctness, dropout confined to non-protected lanes with
speed-limit sync, constant lateral noise with untouched direction channels,
width scaling within the clip range, prob=0 identity). Exits non-zero if any
check fails.

Usage (from the diffusion_planner directory):
    uv run python util_scripts/visualize_lane_augmentation.py <npz> [<npz> ...] \
        --out_dir <dir> [--radius_m 60] [--dropout_ratio 0.3] [--geom_noise_std 0.5]
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
from diffusion_planner.utils.lane_augmentation import LaneAugmentation
from diffusion_planner.utils.route_augmentation import RouteAugmentation

_GEOM_DIM = 8


def load_inputs(npz_path: Path) -> dict:
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


def _valid_rows(tensor: torch.Tensor) -> torch.Tensor:
    return tensor[..., :_GEOM_DIM].abs().sum(dim=(2, 3)) > 0


def _seg_points(tensor: torch.Tensor, seg: int) -> torch.Tensor:
    pts = tensor[0, seg]
    valid = pts[..., :_GEOM_DIM].abs().sum(-1) > 0
    return pts[valid][:, :2]


def _protection_mask(inputs: dict) -> torch.Tensor:
    lanes = inputs["lanes"]
    valid_pt = lanes[..., :_GEOM_DIM].abs().sum(-1) > 0
    dist = torch.where(
        valid_pt, lanes[..., :2].norm(dim=-1), torch.full(valid_pt.shape, float("inf"))
    )
    aug = LaneAugmentation(device="cpu")
    return aug._route_protection_mask(inputs, valid_pt, dist.amin(dim=2))[0]


def run_checks(orig, dropped, truncated, jittered, widened, radius_m) -> list[str]:
    failures = []

    def check(name, cond):
        if not cond:
            failures.append(name)

    lanes_o = orig["lanes"]
    valid_o = _valid_rows(lanes_o)[0]
    protected = _protection_mask(orig)

    # Protected (route-copied / nearest-fallback) lanes survive every component.
    for name, out in (
        ("dropout", dropped),
        ("truncation", truncated),
        ("noise", jittered),
        ("width", widened),
    ):
        check(
            f"{name} keeps protected lanes",
            torch.equal(out["lanes"][0, protected], lanes_o[0, protected]),
        )

    # Dropout: dropped rows are a subset of non-protected valid lanes; the
    # speed-limit rows follow; survivors are bit-identical.
    kept_d = _valid_rows(dropped["lanes"])[0]
    removed = valid_o & ~kept_d
    check("dropout only non-protected", bool((removed & protected).sum() == 0))
    check(
        "dropout speed-limit sync",
        bool(~dropped["lanes_has_speed_limit"][0, removed].any())
        and bool(dropped["lanes_speed_limit"][0, removed].abs().sum() == 0),
    )
    check("dropout keeps survivors", torch.equal(dropped["lanes"][0, kept_d], lanes_o[0, kept_d]))

    # Truncation: exactly the non-protected lanes whose nearest point is
    # beyond the radius are gone.
    valid_pt = lanes_o[..., :_GEOM_DIM].abs().sum(-1) > 0
    dist = lanes_o[..., :2].norm(dim=-1)
    dist = torch.where(valid_pt, dist, torch.full_like(dist, float("inf")))
    min_dist = dist.amin(dim=2)[0]
    expected_drop = (min_dist > radius_m) & valid_o & ~protected
    kept_t = _valid_rows(truncated["lanes"])[0]
    check("truncation drop set", torch.equal(kept_t, valid_o & ~expected_drop))

    # Geometry noise: constant lateral per-segment offset on non-protected
    # lanes only; direction channels and one-hots bit-identical.
    lanes_j = jittered["lanes"]
    for seg in torch.where(valid_o & ~protected)[0].tolist():
        pts_valid = valid_pt[0, seg]
        delta = (lanes_j[0, seg, :, :2] - lanes_o[0, seg, :, :2])[pts_valid]
        check(
            f"noise constant (lane {seg})",
            torch.allclose(delta, delta[0].expand_as(delta), atol=1e-2),
        )
        mean_dir = lanes_o[0, seg, pts_valid, 2:4].sum(0)
        unit = mean_dir / mean_dir.norm().clamp_min(1e-6)
        check(f"noise lateral (lane {seg})", bool(abs(float(delta[0] @ unit)) < 1e-2))
    check("noise keeps dx/dy+attrs", torch.equal(lanes_j[..., 2:], lanes_o[..., 2:]))

    # Width jitter: only boundary channels scaled, within the clip range.
    lanes_w = widened["lanes"]
    check("width keeps centerline", torch.equal(lanes_w[..., :4], lanes_o[..., :4]))
    check("width keeps attrs", torch.equal(lanes_w[..., 8:], lanes_o[..., 8:]))
    for start in (4, 6):
        o = lanes_o[0, :, :, start : start + 2]
        w = lanes_w[0, :, :, start : start + 2]
        nz = o.abs() > 1e-6
        ratio = (w[nz] / o[nz]).abs()
        check(
            f"width scale in clip (ch {start})",
            bool(((ratio > 0.85 - 1e-4) & (ratio < 1.15 + 1e-4)).all()) if ratio.numel() else True,
        )

    # prob=0 identity.
    ident = LaneAugmentation(device="cpu")(copy.deepcopy(orig))
    for key in orig:
        check(f"prob=0 identity ({key})", torch.equal(ident[key], orig[key]))

    return failures


def _draw_lane_map(ax, inputs, protected, colors, title):
    """Draw lane centerlines; ``colors`` maps lane index -> matplotlib color."""
    lanes = inputs["lanes"]
    valid = _valid_rows(lanes)[0]
    for seg in torch.where(valid)[0].tolist():
        xy = _seg_points(lanes, seg)
        color = colors.get(seg, "darkgray")
        lw = 2.5 if bool(protected[seg]) or color != "darkgray" else 1.2
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw)
    ax.plot(0, 0, marker="*", color="black", markersize=12)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)


def _before_after_figure(
    orig, after, protected, removed, suptitle, right_title, out_path, circle_r=None
):
    fig, axes = plt.subplots(1, 2, figsize=(20, 9), sharex=True, sharey=True)
    colors_before = {int(s): "tab:blue" for s in torch.where(protected)[0]}
    colors_before.update({int(s): "tab:red" for s in torch.where(removed)[0]})
    _draw_lane_map(
        axes[0], orig, protected, colors_before, "original (blue=protected route copy, red=removed)"
    )
    colors_after = {int(s): "tab:blue" for s in torch.where(protected)[0]}
    _draw_lane_map(axes[1], after, protected, colors_after, right_title)
    if circle_r is not None:
        for ax in axes:
            ax.add_patch(plt.Circle((0, 0), circle_r, fill=False, color="tab:red", ls=":", lw=1.5))
    fig.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def render_noise(orig, jittered, protected, std, status, name, out_path):
    lanes_o, lanes_j = orig["lanes"], jittered["lanes"]
    valid = _valid_rows(lanes_o)[0]
    fig, ax = plt.subplots(figsize=(12, 11))
    for seg in torch.where(valid)[0].tolist():
        xy_o = _seg_points(lanes_o, seg)
        xy_j = _seg_points(lanes_j, seg)
        if bool(protected[seg]):
            ax.plot(xy_o[:, 0], xy_o[:, 1], color="tab:blue", lw=2.5)
        else:
            ax.plot(xy_o[:, 0], xy_o[:, 1], "k--", lw=1.2)
            ax.plot(xy_j[:, 0], xy_j[:, 1], color="tab:orange", lw=1.5)
    ax.plot(0, 0, marker="*", color="black", markersize=12)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlim(-80, 80)
    ax.set_ylim(-80, 80)
    ax.set_title(
        f"lane geometry noise (std={std}m): black dashed=original, orange=jittered, "
        "blue=protected route copy (untouched)"
    )
    fig.suptitle(f"{name}   [{status}]", color="red" if "NG" in status else "green")
    plt.tight_layout()
    plt.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def render_width(orig, widened, protected, status, name, out_path):
    """Zoom on the nearest non-protected lane: center + absolute boundaries."""
    lanes_o, lanes_w = orig["lanes"], widened["lanes"]
    valid = _valid_rows(lanes_o)[0]
    candidates = torch.where(valid & ~protected)[0]
    fig, axes = plt.subplots(1, 2, figsize=(20, 9), sharex=True, sharey=True)
    for ax, tensor, title in ((axes[0], lanes_o, "original"), (axes[1], lanes_w, "width jittered")):
        for seg in candidates[:6].tolist():
            pts = tensor[0, seg]
            v = pts[..., :_GEOM_DIM].abs().sum(-1) > 0
            c = pts[v][:, :2]
            lb = c + pts[v][:, 4:6]
            rb = c + pts[v][:, 6:8]
            ax.plot(c[:, 0], c[:, 1], color="darkgray", lw=1.0)
            ax.plot(lb[:, 0], lb[:, 1], color="tab:green", lw=1.5)
            ax.plot(rb[:, 0], rb[:, 1], color="tab:purple", lw=1.5)
        ax.plot(0, 0, marker="*", color="black", markersize=12)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_xlim(-60, 60)
        ax.set_ylim(-60, 60)
        ax.set_title(f"{title} (green=left boundary, purple=right boundary)")
    fig.suptitle(
        f"lane width jitter   {name}   [{status}]", color="red" if "NG" in status else "green"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_paths", type=Path, nargs="+")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--radius_m", type=float, default=60.0)
    parser.add_argument("--dropout_ratio", type=float, default=0.3)
    parser.add_argument(
        "--geom_noise_std",
        type=float,
        default=0.5,
        help="visualization noise std [m] (exaggerated vs. the training default)",
    )
    parser.add_argument(
        "--width_jitter_std",
        type=float,
        default=0.5,
        help="visualization width jitter std (clip keeps it in [0.85, 1.15])",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    drop_aug = LaneAugmentation(device="cpu", dropout_prob=1.0, dropout_ratio=args.dropout_ratio)
    trunc_aug = LaneAugmentation(
        device="cpu",
        truncation_prob=1.0,
        truncation_min_m=args.radius_m,
        truncation_max_m=args.radius_m,
    )
    noise_aug = LaneAugmentation(
        device="cpu", geometry_noise_prob=1.0, geometry_noise_std_m=args.geom_noise_std
    )
    width_aug = LaneAugmentation(
        device="cpu", width_jitter_prob=1.0, width_jitter_std=args.width_jitter_std
    )

    any_fail = False
    for npz_path in args.npz_paths:
        orig = load_inputs(npz_path)
        dropped = drop_aug(copy.deepcopy(orig))
        truncated = trunc_aug(copy.deepcopy(orig))
        jittered = noise_aug(copy.deepcopy(orig))
        widened = width_aug(copy.deepcopy(orig))
        protected = _protection_mask(orig)

        failures = run_checks(orig, dropped, truncated, jittered, widened, args.radius_m)
        valid_o = _valid_rows(orig["lanes"])[0]
        status = "OK" if not failures else "NG: " + ", ".join(failures)
        print(
            f"{npz_path.name}: {status} (lanes {int(valid_o.sum())}, "
            f"protected {int(protected.sum())}, "
            f"dropout -> {int(_valid_rows(dropped['lanes'])[0].sum())}, "
            f"truncation -> {int(_valid_rows(truncated['lanes'])[0].sum())})"
        )
        any_fail |= bool(failures)

        removed_d = valid_o & ~_valid_rows(dropped["lanes"])[0]
        _before_after_figure(
            orig,
            dropped,
            protected,
            removed_d,
            f"lane dropout (ratio={args.dropout_ratio})   {npz_path.name}   [{status}]",
            f"after dropout ({int((valid_o & ~removed_d).sum())} lanes)",
            args.out_dir / f"{npz_path.stem}_lane_dropout_check.png",
        )
        removed_t = valid_o & ~_valid_rows(truncated["lanes"])[0]
        _before_after_figure(
            orig,
            truncated,
            protected,
            removed_t,
            f"lane truncation @ {args.radius_m:.0f}m   {npz_path.name}   [{status}]",
            f"after truncation ({int((valid_o & ~removed_t).sum())} lanes)",
            args.out_dir / f"{npz_path.stem}_lane_truncation_check.png",
            circle_r=args.radius_m,
        )
        render_noise(
            orig,
            jittered,
            protected,
            args.geom_noise_std,
            status,
            npz_path.name,
            args.out_dir / f"{npz_path.stem}_lane_noise_check.png",
        )
        render_width(
            orig,
            widened,
            protected,
            status,
            npz_path.name,
            args.out_dir / f"{npz_path.stem}_lane_width_check.png",
        )
        for f in (
            args.out_dir / f"{npz_path.stem}_lane_{n}_check.png"
            for n in ("dropout", "truncation", "noise", "width")
        ):
            print(f"  -> {f}")

    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
