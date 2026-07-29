#!/usr/bin/env python3
"""Render real-scene evidence for the T4 expert lane-mask audit.

The map, aligned-neighbor and rear-axle ego OBB primitives are imported from
the colleague-based ``grpo_viz.py`` visualizer.  This renderer adds a four-role
logged-expert audit: covered lane change, missed lane change, over-exemption,
and ordinary control.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from rlvr.autoresearch.tools.grpo_viz import (
    _add_obb,
    _aligned_neighbor_future,
    _draw_map_context,
    _neighbor_tracks,
)


ROLE_SPECS = (
    ("covered", True, True, "A · CORRECT EXEMPTION · lane change + mask ON", "#16a34a"),
    ("missed", True, False, "B · MISSED EXEMPTION · lane change + mask OFF", "#dc2626"),
    ("over_exempt", False, True, "C · OVER-EXEMPTION · no change + mask ON", "#d97706"),
    ("control", False, False, "D · CORRECT CONTROL · no change + mask OFF", "#2563eb"),
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-audit", type=Path, required=True)
    parser.add_argument("--lane-change-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--font", type=Path)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--duration-ms", type=int, default=140)
    return parser.parse_args()


def _font(path: Path | None) -> None:
    if path and path.exists():
        fm.fontManager.addfont(path)
        family = fm.FontProperties(fname=path).get_name()
        plt.rcParams["font.family"] = family
    plt.rcParams.update(
        {
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "font.size": 9,
            "figure.facecolor": "#f5f7fb",
            "axes.facecolor": "#ffffff",
        }
    )


def _turn_value(npz) -> int | None:
    value = npz.get("turn_indicators")
    if value is None:
        return None
    flat = np.asarray(value).reshape(-1)
    return int(flat[-1]) if len(flat) else None


def _choose_rows(mask_payload: dict, lane_payload: dict) -> list[dict]:
    lane_by_scene = {row["scene"]: row for row in lane_payload["rows"]}
    out = []
    for role, actual, mask, title, color in ROLE_SPECS:
        candidates = [
            row
            for row in mask_payload["rows"]
            if "error" not in row
            and bool(row["actual_lane_change"]) is actual
            and bool(row["expert_mask_active"]) is mask
        ]
        if not candidates:
            continue

        def priority(row: dict) -> tuple:
            source = lane_by_scene.get(row["scene"], {})
            meta = source.get("actual_meta", {})
            drift = float(meta.get("max_ego_lateral_drift_m", 0.0))
            motion = float(meta.get("total_motion_m", 0.0))
            # Prefer an adjacent-lane-scale lateral move over a huge turn.  For
            # change roles, agreement with the independent route prediction
            # makes the scene easier to interpret.  Controls prefer low drift.
            route_agrees = int(bool(row.get("route_predicted_lane_change")) is actual)
            if actual:
                return (route_agrees, -abs(drift - 3.5), min(motion, 60.0), row["scene"])
            return (-drift, min(motion, 60.0), route_agrees, row["scene"])

        chosen = sorted(candidates, key=priority, reverse=True)[0]
        source = lane_by_scene.get(chosen["scene"], {})
        chosen = dict(chosen)
        chosen.update(
            {
                "role": role,
                "title": title,
                "color": color,
                "actual_meta": source.get("actual_meta", {}),
                "predict_meta": source.get("predict_meta", {}),
            }
        )
        out.append(chosen)
    return out


def _scene(npz_path: str) -> dict:
    npz = np.load(npz_path)
    gt = np.asarray(npz["ego_agent_future"])
    future = _aligned_neighbor_future(npz)
    tracks = _neighbor_tracks(npz, future, limit=10)
    ego_shape = np.asarray(npz.get("ego_shape", [2.75, 4.34, 1.70])).reshape(-1)
    wb, length, width = map(float, ego_shape[:3])
    focus_gt = gt[:40]
    pts = [focus_gt[:, :2]]
    for _, slot, _, valid in tracks[:6]:
        if valid.any():
            valid40 = valid.copy()
            valid40[40:] = False
            if valid40.any():
                pts.append(future[slot, valid40, :2])
    all_pts = np.concatenate(pts, axis=0)
    lo, hi = all_pts.min(axis=0), all_pts.max(axis=0)
    center = (lo + hi) / 2.0
    span = max(float(np.ptp(all_pts[:, 0])), float(np.ptp(all_pts[:, 1])), 15.0)
    half = span / 2.0 + max(4.0, span * 0.08)
    return {
        "npz": npz,
        "gt": gt,
        "future": future,
        "tracks": tracks,
        "wb": wb,
        "length": length,
        "width": width,
        "xlim": (center[0] - half, center[0] + half),
        "ylim": (center[1] - half, center[1] + half),
        "turn": _turn_value(npz),
    }


def _ego_pose(gt: np.ndarray, step: int) -> np.ndarray:
    step = min(max(int(step), 0), len(gt) - 1)
    return np.array(
        [gt[step, 0], gt[step, 1], np.cos(gt[step, 2]), np.sin(gt[step, 2])],
        dtype=float,
    )


def _draw_neighbors(ax, scene: dict, step: int) -> None:
    palette = ["#b45309", "#7c3aed", "#0369a1", "#64748b", "#be123c"]
    for rank, (_, slot, current, valid) in enumerate(scene["tracks"][:5]):
        if step >= len(valid) or not valid[step]:
            continue
        value = scene["future"][slot, step]
        if np.linalg.norm(value[:2]) < 1e-3:
            continue
        length = float(current[7]) if len(current) > 7 and current[7] > 0.1 else 4.5
        width = float(current[6]) if len(current) > 6 and current[6] > 0.1 else 1.8
        pose = np.array([value[0], value[1], np.cos(value[2]), np.sin(value[2])])
        _add_obb(
            ax,
            pose,
            length,
            width,
            color=palette[rank % len(palette)],
            alpha=0.68,
            fill_alpha=0.08,
            lw=0.9,
            zorder=10,
        )


def _event_step(row: dict, gt: np.ndarray, fallback: int) -> int:
    mask_step = row.get("expert_mask_first_step")
    if mask_step is not None:
        return min(max(int(mask_step), 0), len(gt) - 1)
    for transition in row.get("actual_meta", {}).get("transitions", []):
        if transition.get("qualifies_as_lane_change"):
            return min(max(int(transition.get("t", fallback)), 0), len(gt) - 1)
    return min(max(int(fallback), 0), len(gt) - 1)


def _draw_event_camera(parent, row: dict, scene: dict, step: int) -> None:
    inset = parent.inset_axes([0.625, 0.56, 0.34, 0.35])
    _draw_map_context(inset, scene["npz"])
    gt = scene["gt"]
    end = min(step + 1, len(gt))
    inset.plot(gt[:end, 0], gt[:end, 1], color="#15803d", lw=1.7, zorder=12)
    _draw_neighbors(inset, scene, step)
    pose = _ego_pose(gt, step)
    _add_obb(
        inset,
        pose,
        scene["length"],
        scene["width"],
        color=row["color"],
        alpha=1.0,
        fill_alpha=0.18,
        lw=2.0,
        zorder=18,
        wheelbase=scene["wb"],
    )
    radius = 8.0
    inset.set_xlim(pose[0] - radius, pose[0] + radius)
    inset.set_ylim(pose[1] - radius, pose[1] + radius)
    inset.set_aspect("equal", adjustable="box")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title(f"OBB zoom · {step / 10:.1f}s", fontsize=6.8, loc="left", pad=2)
    for spine in inset.spines.values():
        spine.set_edgecolor(row["color"])
        spine.set_linewidth(1.3)


def _draw_panel(ax, row: dict, scene: dict, step: int, *, static: bool) -> None:
    npz, gt = scene["npz"], scene["gt"]
    _draw_map_context(ax, npz)
    end = min(step + 1, len(gt))
    ax.plot(gt[:, 0], gt[:, 1], "--", color="#86a893", lw=1.1, alpha=0.55, zorder=5)
    ax.plot(gt[:end, 0], gt[:end, 1], color="#15803d", lw=2.6, zorder=12)

    if static:
        key_steps = sorted(set([0, min(13, len(gt) - 1), min(26, len(gt) - 1), min(39, len(gt) - 1)]))
        for key in key_steps:
            color = row["color"] if key == min(39, len(gt) - 1) else "#475569"
            _add_obb(
                ax,
                _ego_pose(gt, key),
                scene["length"],
                scene["width"],
                color=color,
                alpha=0.62,
                fill_alpha=0.04,
                lw=1.0,
                zorder=13,
                wheelbase=scene["wb"],
            )
        mask_step = row.get("expert_mask_first_step")
        if mask_step is not None and int(mask_step) < len(gt):
            _add_obb(
                ax,
                _ego_pose(gt, int(mask_step)),
                scene["length"],
                scene["width"],
                color="#dc2626",
                alpha=1.0,
                fill_alpha=0.16,
                lw=2.2,
                zorder=16,
                wheelbase=scene["wb"],
            )
            ax.scatter(gt[int(mask_step), 0], gt[int(mask_step), 1], marker="X", s=58, color="#dc2626", zorder=17)
        _draw_event_camera(ax, row, scene, _event_step(row, gt, min(step, 39)))
    else:
        _draw_neighbors(ax, scene, step)
        _add_obb(
            ax,
            _ego_pose(gt, step),
            scene["length"],
            scene["width"],
            color=row["color"],
            alpha=1.0,
            fill_alpha=0.14,
            lw=2.0,
            zorder=16,
            wheelbase=scene["wb"],
        )
        _draw_event_camera(ax, row, scene, step)

    drift = float(row.get("actual_meta", {}).get("max_ego_lateral_drift_m", 0.0))
    mask_step = row.get("expert_mask_first_step")
    mask_text = "ON" if row["expert_mask_active"] else "OFF"
    label_text = "LANE CHANGE" if row["actual_lane_change"] else "NO CHANGE"
    route_text = "YES" if row["route_predicted_lane_change"] else "NO"
    ax.set_title(row["title"], loc="left", color=row["color"], pad=7)
    ax.text(
        0.015,
        0.02,
        f"independent label: {label_text}   |   formal mask: {mask_text}"
        f"{'' if mask_step is None else f' @ {mask_step / 10:.1f}s'}\n"
        f"route cue: {route_text}   |   indicator: {scene['turn']}   |   lateral drift: {drift:.1f} m",
        transform=ax.transAxes,
        fontsize=7.2,
        color="#0f172a",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#cbd5e1", "alpha": 0.92},
        zorder=30,
    )
    ax.set_xlim(*scene["xlim"])
    ax.set_ylim(*scene["ylim"])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.10)
    ax.set_xlabel("ego-frame x [m]")
    ax.set_ylabel("y [m]")
    for spine in ax.spines.values():
        spine.set_edgecolor(row["color"])
        spine.set_linewidth(2.2)


def _figure(rows: list[dict], scenes: list[dict], step: int, *, static: bool):
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=False)
    fig.subplots_adjust(left=0.045, right=0.985, top=0.89, bottom=0.075, wspace=0.14, hspace=0.22)
    fig.suptitle(
        "Does the lane mask exempt only intended lane changes? · four real validation scenes",
        x=0.045,
        y=0.982,
        ha="left",
        fontsize=19,
        fontweight="bold",
        color="#102a43",
    )
    fig.text(
        0.045,
        0.932,
        "Read A/D as desired behavior; B/C are the two error modes to audit. Green path = logged expert · blue = route · red = road border.",
        fontsize=9.5,
        color="#52677d",
    )
    for ax, row, scene in zip(axes.flat, rows, scenes):
        _draw_panel(ax, row, scene, step, static=static)
    fig.text(
        0.985,
        0.025,
        "Diagnostic only: the independent label is not perfect ground truth. Formal lane hard gate is OFF; this mask changes only the 2/14 lane quality term.",
        ha="right",
        fontsize=8.5,
        color="#64748b",
    )
    return fig


def _image(fig) -> Image.Image:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def main() -> None:
    args = _args()
    _font(args.font)
    mask_payload = json.loads(args.mask_audit.read_text())
    lane_payload = json.loads(args.lane_change_audit.read_text())
    rows = _choose_rows(mask_payload, lane_payload)
    if len(rows) != 4:
        raise SystemExit(f"expected all four audit roles, got {[row['role'] for row in rows]}")
    scenes = [_scene(row["scene"]) for row in rows]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    static_path = args.output_dir / "lane_mask_audit.png"
    _figure(rows, scenes, step=min(args.frames - 1, 39), static=True).savefig(
        static_path, dpi=150, facecolor="#f5f7fb"
    )
    plt.close("all")

    frames = []
    for step in range(args.frames):
        frames.append(_image(_figure(rows, scenes, step=step, static=False)))
    gif_path = args.output_dir / "lane_mask_audit.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )

    manifest = {
        "contract": {
            "evidence": "real logged T4 validation scenes; evaluator audit, not checkpoint improvement",
            "frames": args.frames,
            "frame_duration_ms": args.duration_ms,
            "neighbor_future_alignment": "raw[t+1] -> aligned[t]",
            "ego_obb": "rear-axle pose; center shifted by 0.5 * wheelbase",
            "formal_lane_gate_enabled": False,
        },
        "source_mask_audit": str(args.mask_audit),
        "source_lane_change_audit": str(args.lane_change_audit),
        "assets": {"png": static_path.name, "gif": gif_path.name},
        "scenes": [
            {
                key: row.get(key)
                for key in (
                    "role",
                    "title",
                    "scene",
                    "actual_lane_change",
                    "route_predicted_lane_change",
                    "expert_mask_active",
                    "expert_mask_first_step",
                    "actual_meta",
                    "predict_meta",
                )
            }
            | {"turn_indicator_current": scene["turn"]}
            for row, scene in zip(rows, scenes)
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for scene in scenes:
        scene["npz"].close()
    print(json.dumps({"png": str(static_path), "gif": str(gif_path), "roles": [r["role"] for r in rows]}))


if __name__ == "__main__":
    main()
