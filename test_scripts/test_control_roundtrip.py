"""Roundtrip tests for the ego trajectory <-> control pipeline.

Runnable with:
    pytest test_scripts/test_control_roundtrip.py                    # synthetic data
    python3 test_scripts/test_control_roundtrip.py <path_list.json>  # recorded data
    python3 test_scripts/test_control_roundtrip.py <path_list.json> --out_dir <dir>  # + plots
"""

import argparse
import json
import math
import os

import numpy as np
import torch
from diffusion_planner.dimensions import INPUT_T, OUTPUT_T
from diffusion_planner.loss import ACTION_SPACE
from diffusion_planner.utils.normalizer import ControlNormalizer
from diffusion_planner.utils.unicycle_accel_curvature import action_to_traj4d, traj4d_to_action

T_HIST = INPUT_T + 1  # 31
T_FUTURE = OUTPUT_T  # 80
DT = 0.1

# Ego control statistics over 3000 frames of the training set, measured with
# ros_scripts/analyze_control_stats.py. Plots use mean +- 4 std as a fixed axis
# range so that control magnitudes stay comparable across frames.
ACCEL_MEAN, ACCEL_STD = -0.012965, 0.505454
CURVATURE_MEAN, CURVATURE_STD = -0.001479, 0.033967
PLOT_SIGMA = 4.0

# Minimum extent of the trajectory panel, so that near-stationary frames are not
# blown up to millimeter scale.
MIN_PLOT_SPAN_M = 1.0


# ---------------------------------------------------------------------------
# Synthetic trajectory generators
# ---------------------------------------------------------------------------


def make_straight_line(
    v: float = 5.0,
    heading: float = 0.0,
    T: int = T_FUTURE,
    dt: float = DT,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Generate a straight-line trajectory.

    Returns:
        past:  [1, T_HIST, 4]  (x, y, cos, sin)
        future: [1, T, 4]
        v0: initial speed (m/s)
    """
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    total_steps = T_HIST + T

    xs, ys = [], []
    for i in range(total_steps):
        t = (i - T_HIST + 1) * dt  # t=0 at last history step
        xs.append(v * t * cos_h)
        ys.append(v * t * sin_h)

    traj = torch.zeros(1, total_steps, 4)
    traj[0, :, 0] = torch.tensor(xs)
    traj[0, :, 1] = torch.tensor(ys)
    traj[0, :, 2] = cos_h
    traj[0, :, 3] = sin_h

    past = traj[:, :T_HIST]
    future = traj[:, T_HIST:]
    return past, future, v


def make_circular_arc(
    v: float = 5.0,
    radius: float = 50.0,
    T: int = T_FUTURE,
    dt: float = DT,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Generate a circular-arc trajectory (constant speed, constant curvature).

    Returns:
        past:  [1, T_HIST, 4]
        future: [1, T, 4]
        v0: initial speed
    """
    omega = v / radius  # angular velocity
    total_steps = T_HIST + T

    traj = torch.zeros(1, total_steps, 4)
    for i in range(total_steps):
        t = (i - T_HIST + 1) * dt
        theta = omega * t
        traj[0, i, 0] = radius * math.sin(theta)
        traj[0, i, 1] = radius * (1.0 - math.cos(theta))
        traj[0, i, 2] = math.cos(theta)
        traj[0, i, 3] = math.sin(theta)

    past = traj[:, :T_HIST]
    future = traj[:, T_HIST:]
    return past, future, v


# ---------------------------------------------------------------------------
# Pytest tests (synthetic data)
# ---------------------------------------------------------------------------


class TestEgoRoundtrip:
    """trajectory -> control -> trajectory for ego."""

    def _run(self, past, future, v0, atol=0.01):
        ctrl = traj4d_to_action(ACTION_SPACE, past, future, t0_states={"v": torch.tensor([v0])})
        assert not ctrl.isnan().any(), "NaN in ego control"
        assert not ctrl.isinf().any(), "Inf in ego control"

        recon = action_to_traj4d(ACTION_SPACE, past, ctrl, t0_states={"v": torch.tensor([v0])})
        assert torch.allclose(future[..., :2], recon[..., :2], atol=atol), (
            f"Ego roundtrip pos error too large: max={(future[..., :2] - recon[..., :2]).abs().max().item():.6f}"
        )

    def test_straight_line(self):
        # The ridge term of the velocity/acceleration solver biases the reconstruction by
        # ~1 cm over the 40 m the trajectory covers, so the tolerance sits above that.
        past, future, v0 = make_straight_line(v=5.0, heading=0.0)
        self._run(past, future, v0, atol=0.02)

    def test_circular_arc(self):
        # Larger atol for arcs: unicycle discrete integration accumulates error
        past, future, v0 = make_circular_arc(v=5.0, radius=50.0)
        self._run(past, future, v0, atol=0.25)

    def test_circular_arc_tight(self):
        past, future, v0 = make_circular_arc(v=3.0, radius=20.0)
        self._run(past, future, v0, atol=0.20)


class TestControlNormalizerRoundtrip:
    """control -> normalize -> inverse -> compare."""

    def test_roundtrip(self):
        mean = [0.1, 0.002]
        std = [1.5, 0.05]
        norm = ControlNormalizer(mean, std)

        ctrl = torch.randn(2, T_FUTURE, 2) * 2.0
        normed = norm(ctrl)
        recovered = norm.inverse(normed)

        assert torch.allclose(ctrl, recovered, atol=1e-6), (
            f"ControlNormalizer roundtrip error: max={(ctrl - recovered).abs().max().item():.2e}"
        )

    def test_zero_preserving(self):
        norm = ControlNormalizer([0.0, 0.0], [1.0, 1.0])
        ctrl = torch.zeros(1, T_FUTURE, 2)
        normed = norm(ctrl)
        recovered = norm.inverse(normed)
        assert torch.allclose(ctrl, recovered, atol=1e-6)


# ---------------------------------------------------------------------------
# Standalone mode: roundtrip on recorded npz frames
# ---------------------------------------------------------------------------


def _load_sample(path: str) -> dict:
    """Load a single npz file and prepare tensors for roundtrip testing."""
    from diffusion_planner.train_epoch import heading_to_cos_sin

    d = np.load(path)

    ego_past = heading_to_cos_sin(
        torch.from_numpy(d["ego_agent_past"]).unsqueeze(0).float()
    )  # [1, T_HIST, 4]
    ego_v0 = float(d["ego_current_state"][4])
    ego_future = heading_to_cos_sin(
        torch.from_numpy(d["ego_agent_future"]).unsqueeze(0).float()
    )  # [1, T, 4]

    return {
        "path": path,
        "ego_past": ego_past,
        "ego_v0": ego_v0,
        "ego_future": ego_future,
    }


def _set_equal_limits(ax, xs, ys, min_span: float):
    """Give the axes a 1:1 aspect ratio spanning at least ``min_span`` metres."""
    x_lo, x_hi = float(min(xs)), float(max(xs))
    y_lo, y_hi = float(min(ys)), float(max(ys))
    margin = 1.05  # a little air around the trajectory
    span = max((x_hi - x_lo) * margin, (y_hi - y_lo) * margin, min_span)
    x_center = 0.5 * (x_lo + x_hi)
    y_center = 0.5 * (y_lo + y_hi)
    ax.set_xlim(x_center - 0.5 * span, x_center + 0.5 * span)
    ax.set_ylim(y_center - 0.5 * span, y_center + 0.5 * span)
    ax.set_aspect("equal", adjustable="box")


def _plot_sample(sample, ctrl, recon, out_path: str):
    """Save a figure of GT vs roundtrip trajectory, position error and control."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    past = sample["ego_past"][0]
    future = sample["ego_future"][0]
    recon = recon[0]
    err = (future[:, :2] - recon[:, :2]).norm(dim=-1)

    fig, (ax_xy, ax_err, ax_ctrl) = plt.subplots(1, 3, figsize=(21, 7))

    ax_xy.plot(past[:, 0], past[:, 1], "k.-", lw=1, ms=3, label="past")
    ax_xy.plot(future[:, 0], future[:, 1], "b-", lw=2, label="GT future")
    ax_xy.plot(recon[:, 0], recon[:, 1], "r--", lw=2, label="roundtrip")
    ax_xy.set_title(os.path.basename(sample["path"]))
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    _set_equal_limits(
        ax_xy,
        torch.cat([past[:, 0], future[:, 0], recon[:, 0]]),
        torch.cat([past[:, 1], future[:, 1], recon[:, 1]]),
        MIN_PLOT_SPAN_M,
    )
    ax_xy.grid(alpha=0.3)
    ax_xy.legend(fontsize=9)

    ax_err.plot(err, "b-")
    ax_err.set_title(f"position error (mean={err.mean():.5f}m, max={err.max():.5f}m)")
    ax_err.set_xlabel("future step")
    ax_err.set_ylabel("error [m]")
    ax_err.set_yscale("log")
    ax_err.grid(alpha=0.3)

    ax_ctrl.plot(ctrl[0, :, 0], "b-", label="accel")
    ax_ctrl.set_xlabel("future step")
    ax_ctrl.set_ylabel("accel [m/s^2]", color="b")
    ax_ctrl.set_title(f"control (v0={sample['ego_v0']:.2f} m/s, axes = mean +- {PLOT_SIGMA:g} std)")
    ax_ctrl.set_ylim(ACCEL_MEAN - PLOT_SIGMA * ACCEL_STD, ACCEL_MEAN + PLOT_SIGMA * ACCEL_STD)
    ax_ctrl.grid(alpha=0.3)
    ax_curv = ax_ctrl.twinx()
    ax_curv.plot(ctrl[0, :, 1], "g-", label="curvature")
    ax_curv.set_ylabel("curvature [1/m]", color="g")
    ax_curv.set_ylim(
        CURVATURE_MEAN - PLOT_SIGMA * CURVATURE_STD,
        CURVATURE_MEAN + PLOT_SIGMA * CURVATURE_STD,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _run_standalone(path_list_json: str, num_samples: int, stride: int | None, out_dir: str | None):
    """Run roundtrip tests on recorded data and print detailed results.

    Frames are picked deterministically by striding through the path list. When
    ``stride`` is None it is derived so that ``num_samples`` frames spread evenly
    over the whole list.
    """
    with open(path_list_json) as f:
        paths = json.load(f)

    if stride is None:
        stride = max(1, len(paths) // max(1, num_samples))
    indices = list(range(0, len(paths), stride))[:num_samples]
    n_sample = len(indices)

    print(f"Testing {n_sample} samples from {path_list_json} (stride={stride}, total={len(paths)})")
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
    print(f"{'=' * 70}")

    ego_errors = []

    for i, idx in enumerate(indices):
        sample = _load_sample(paths[idx])
        print(f"\nSample {idx}: {sample['path']}")

        ego_ctrl = traj4d_to_action(
            ACTION_SPACE,
            sample["ego_past"],
            sample["ego_future"],
            t0_states={"v": torch.tensor([sample["ego_v0"]])},
        )
        ego_recon = action_to_traj4d(
            ACTION_SPACE,
            sample["ego_past"],
            ego_ctrl,
            t0_states={"v": torch.tensor([sample["ego_v0"]])},
        )
        ego_err = (sample["ego_future"][0, :, :2] - ego_recon[0, :, :2]).abs()
        ego_max = ego_err.max().item()
        ego_mean = ego_err.mean().item()
        ego_errors.append(ego_max)
        status = "PASS" if ego_max < 0.01 else "FAIL"
        print(f"  EGO roundtrip: [{status}] mean={ego_mean:.6f}m  max={ego_max:.6f}m")

        if ego_ctrl.isnan().any() or ego_ctrl.isinf().any():
            print("  EGO ctrl has NaN/Inf!")

        if out_dir is not None:
            out_path = os.path.join(out_dir, f"roundtrip_{i:02d}_idx{idx}.png")
            _plot_sample(sample, ego_ctrl, ego_recon, out_path)
            print(f"  saved: {out_path}")

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(
        f"  Ego   : {len(ego_errors)} samples, "
        f"max_err range [{min(ego_errors):.6f}, {max(ego_errors):.6f}]m, "
        f"all<0.01: {all(e < 0.01 for e in ego_errors)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path_list",
        type=str,
        help="JSON list of npz frame paths (the same form as --train_set_list)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="how many frames to sample from the path list",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="take every Nth frame; default spreads --num_samples evenly over the list",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="if given, save a visualization PNG per sampled frame into this directory",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _run_standalone(args.path_list, args.num_samples, args.stride, args.out_dir)
