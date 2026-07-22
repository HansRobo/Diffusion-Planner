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

        # Side-by-side render. visualize_inputs mutates its dict -> pass copies.
        fig, axes = plt.subplots(1, 2, figsize=(20, 9))
        visualize_inputs(copy.deepcopy(orig), ax=axes[0])
        axes[0].set_title(f"original ({n_orig} route segments)")
        visualize_inputs(copy.deepcopy(truncated), ax=axes[1])
        axes[1].set_title(f"truncated @ {args.truncate_m:.0f}m ({n_kept} route segments)")
        fig.suptitle(f"{npz_path.name}   [{status}]", color="red" if failures else "green")
        out = args.out_dir / f"{npz_path.stem}_route_aug_check.png"
        plt.tight_layout()
        plt.savefig(out, dpi=90, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {out}")

    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
