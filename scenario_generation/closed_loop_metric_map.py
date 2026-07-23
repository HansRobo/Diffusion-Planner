"""Whole-route overview (Plotly): local map + live-ego arrows + GT dashed path + metric onsets.

Map geometry is stitched sparsely from route NPZ frames (ego-frame lanes /
line_strings / route_lanes -> world), cropped to the live∪GT bbox.

Live ego is downsampled (default every 5 ticks / 0.5 s) and drawn as heading
arrows. Recorded GT is a gray dashed polyline. Metric event onsets stay at
their exact step positions.

Color = metric category; marker = severity (X hard failure, o near-miss).
Snap/teleport = yellow star.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from scenario_generation.metrics.strong_brake import strong_brake_mask
from scenario_generation.reproducer_rollout import RB_COLLISION_THRESH_M, _event_onsets

# Match replay.save_step_figure map colors (thinner / more transparent than PNG frames).
_LANE_COLOR = "rgba(187,187,187,0.25)"
_LANE_BORDER_COLOR = "rgba(136,136,136,0.30)"
_ROAD_BORDER_DRAW = "rgba(221,34,34,0.30)"
_ROUTE_COLOR = "rgba(51,102,204,0.28)"

_COLOR_OBJECT = "#cc2222"
_COLOR_ROAD_BORDER = "#ee8800"
_COLOR_RED_LIGHT = "#cc22aa"
_COLOR_STRONG_BRAKE = "#6622cc"
_COLOR_SNAP = "#e6c200"
_COLOR_LIVE = "#222222"
_COLOR_GT = "#888888"

_MARKER_BAD = "x"
_MARKER_DANGER = "circle"
_MARKER_SNAP = "star"

_DEFAULT_MARGIN_M = 10.0
# Sparse map stitch (~8 samples along the route — less stacked NPZ overlap).
_TARGET_MAP_SAMPLES = 8
# Closed-loop ticks are 0.1 s; 5 ticks = 0.5 s between live-ego arrows.
_TRAJ_ARROW_STRIDE = 5
_PNG_WIDTH = 2200
_PNG_SCALE = 3


def _ego_xy_to_world(xy: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """``(N,2)`` ego-frame points -> world using recorded ego ``pose`` ``(x,y,yaw)``."""
    ex, ey, eyaw = float(pose[0]), float(pose[1]), float(pose[2])
    c, s = math.cos(eyaw), math.sin(eyaw)
    x = xy[..., 0]
    y = xy[..., 1]
    wx = ex + x * c - y * s
    wy = ey + x * s + y * c
    return np.stack([wx, wy], axis=-1).astype(np.float64)


def _sample_indices(start: int, end: int, stride: int | None = None) -> list[int]:
    """Frame indices in ``[start, end)`` with adaptive stride; always includes ends."""
    lo = int(start)
    hi = max(int(end), lo + 1)
    n = hi - lo
    if stride is None:
        stride = max(1, n // _TARGET_MAP_SAMPLES)
    idxs = list(range(lo, hi, int(stride)))
    if not idxs or idxs[0] != lo:
        idxs.insert(0, lo)
    last = hi - 1
    if idxs[-1] != last:
        idxs.append(last)
    return idxs


def _lane_polylines_ego(lanes: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Split a lanes tensor into centerline + left/right border polylines (ego frame)."""
    centers: list[np.ndarray] = []
    borders: list[np.ndarray] = []
    lanes = np.asarray(lanes)
    if lanes.ndim == 4:
        lanes = lanes[0]
    for i in range(lanes.shape[0]):
        lane = lanes[i]
        if np.abs(lane[:, :2]).sum() < 1e-6:
            continue
        pts = lane[:, :2].astype(np.float64)
        valid = np.abs(pts).sum(axis=1) > 0.1
        if valid.sum() < 2:
            continue
        centers.append(pts[valid])
        if lane.shape[1] > 7:
            borders.append((pts + lane[:, 4:6])[valid])
            borders.append((pts + lane[:, 6:8])[valid])
    return centers, borders


def _line_string_borders_ego(line_strings: np.ndarray) -> list[np.ndarray]:
    """Road-border polylines from line_strings (channel 3 flag), ego frame."""
    out: list[np.ndarray] = []
    ls = np.asarray(line_strings)
    if ls.ndim == 4:
        ls = ls[0]
    for seg in ls:
        if seg.shape[0] < 2:
            continue
        if seg.shape[1] > 3 and float(seg[:, 3].max()) <= 0.5:
            continue
        v = np.abs(seg[:, :2]).sum(1) > 0.1
        if v.sum() < 2:
            continue
        out.append(seg[v, :2].astype(np.float64))
    return out


def _route_polylines_ego(route_lanes: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    rl = np.asarray(route_lanes)
    if rl.ndim == 4:
        rl = rl[0]
    for seg in rl:
        v = np.abs(seg[:, :2]).sum(1) > 0.1
        if v.sum() < 2:
            continue
        out.append(seg[v, :2].astype(np.float64))
    return out


def collect_route_map_polylines(
    tl,
    start: int,
    end: int,
    stride: int | None = None,
) -> dict[str, list[np.ndarray]]:
    """Stitch world-frame map polylines from sparsely sampled route NPZ frames."""
    lane_centerlines: list[np.ndarray] = []
    lane_borders: list[np.ndarray] = []
    road_borders: list[np.ndarray] = []
    route_lanes: list[np.ndarray] = []

    for idx in _sample_indices(start, end, stride=stride):
        pose = np.asarray(tl.poses[idx], dtype=np.float64)
        npz = tl.npz(idx)

        if "lanes" in npz:
            cents, bords = _lane_polylines_ego(npz["lanes"])
            lane_centerlines.extend(_ego_xy_to_world(p, pose) for p in cents)
            lane_borders.extend(_ego_xy_to_world(p, pose) for p in bords)
        if "line_strings" in npz:
            road_borders.extend(
                _ego_xy_to_world(p, pose) for p in _line_string_borders_ego(npz["line_strings"])
            )
        if "route_lanes" in npz:
            route_lanes.extend(
                _ego_xy_to_world(p, pose) for p in _route_polylines_ego(npz["route_lanes"])
            )

    return {
        "lane_centerlines": lane_centerlines,
        "lane_borders": lane_borders,
        "road_borders": road_borders,
        "route_lanes": route_lanes,
    }


def _polys_to_nan_separated(polylines: list[np.ndarray], origin: np.ndarray) -> tuple[list, list]:
    """Flatten polylines into Plotly line traces with None breaks between segments."""
    xs: list = []
    ys: list = []
    for pl in polylines:
        if pl is None or len(pl) < 2:
            continue
        p = np.asarray(pl, dtype=np.float64).reshape(-1, 2) - origin
        xs.extend(p[:, 0].tolist())
        ys.extend(p[:, 1].tolist())
        xs.append(None)
        ys.append(None)
    return xs, ys


def _headings_deg(xy: np.ndarray) -> np.ndarray:
    """Per-point heading in degrees (math: 0 = +X). Last point copies previous delta."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    n = len(xy)
    h = np.zeros(n, dtype=np.float64)
    if n < 2:
        return h
    d = np.diff(xy, axis=0)
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    h[:-1] = ang
    h[-1] = ang[-1]
    # Zero-length segments: fill forward from last nonzero.
    for i in range(n - 1):
        if abs(d[i, 0]) + abs(d[i, 1]) < 1e-6 and i > 0:
            h[i] = h[i - 1]
    return h


def _plotly_arrow_angles(heading_deg: np.ndarray) -> np.ndarray:
    """Plotly arrow symbol angle: 0 = up (+Y), clockwise. Math heading 0 = +X."""
    return 90.0 - np.asarray(heading_deg, dtype=np.float64)


def _subsample_arrows(
    xy: np.ndarray, stride: int = _TRAJ_ARROW_STRIDE
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(xy_sub, plotly_angles)`` at ~1 Hz plus the final pose."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if len(xy) == 0:
        return xy, np.zeros(0)
    idxs = list(range(0, len(xy), max(1, int(stride))))
    if idxs[-1] != len(xy) - 1:
        idxs.append(len(xy) - 1)
    sub = xy[np.asarray(idxs)]
    # Headings from the full path at those indices (use neighboring full-path deltas).
    full_h = _headings_deg(xy)
    return sub, _plotly_arrow_angles(full_h[np.asarray(idxs)])


def _xy_at(
    live_xy: np.ndarray, idxs: list[int], origin: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if not idxs:
        return np.zeros(0), np.zeros(0)
    pts = live_xy[np.asarray(idxs, dtype=np.int64)] - origin
    return pts[:, 0], pts[:, 1]


def _viewport_bbox(
    live: np.ndarray,
    gt: np.ndarray | None,
    margin_m: float,
) -> tuple[float, float, float, float]:
    pts = [live]
    if gt is not None and len(gt) >= 1:
        pts.append(np.asarray(gt, dtype=np.float64).reshape(-1, 2))
    all_xy = np.concatenate(pts, axis=0)
    xmin = float(all_xy[:, 0].min()) - margin_m
    xmax = float(all_xy[:, 0].max()) + margin_m
    ymin = float(all_xy[:, 1].min()) - margin_m
    ymax = float(all_xy[:, 1].max()) + margin_m
    if xmax - xmin < 1.0:
        xmin, xmax = xmin - 0.5, xmax + 0.5
    if ymax - ymin < 1.0:
        ymin, ymax = ymin - 0.5, ymax + 0.5
    return xmin, xmax, ymin, ymax


def _add_map_lines(
    fig: go.Figure, polylines: list[np.ndarray], origin: np.ndarray, *, color: str, width: float
):
    xs, ys = _polys_to_nan_separated(polylines, origin)
    if not xs:
        return
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color=color, width=width),
            hoverinfo="skip",
            showlegend=False,
        )
    )


def write_route_metric_map(
    out_path: Path | str,
    live_xy: np.ndarray,
    gt_xy: np.ndarray | None,
    clearances: np.ndarray,
    collisions: np.ndarray,
    rb_dists: np.ndarray,
    red_light: np.ndarray,
    accels: np.ndarray,
    near_miss_thresh: float,
    strong_brake_mps2: float,
    snap_xy: list | None = None,
    title: str = "",
    metrics: dict | None = None,
    *,
    tl=None,
    start: int | None = None,
    end: int | None = None,
    map_polylines: dict[str, list[np.ndarray]] | None = None,
    margin_m: float = _DEFAULT_MARGIN_M,
    map_stride: int | None = None,
    traj_arrow_stride: int = _TRAJ_ARROW_STRIDE,
) -> Path:
    """Write ``{route}_metrics_map.png`` (+ ``.html``) via Plotly.

    Provide either ``map_polylines`` or ``(tl, start, end)`` to stitch map geometry.
    Near-miss markers exclude collision steps so X and o do not stack.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html_path = out_path.with_suffix(".html")

    live = np.asarray(live_xy, dtype=np.float64).reshape(-1, 2)
    n = live.shape[0]
    if n == 0:
        fig = go.Figure()
        fig.update_layout(title=f"{title} (empty)")
        fig.write_html(str(html_path))
        fig.write_image(str(out_path), width=800, height=800, scale=_PNG_SCALE)
        return out_path

    origin = live[0].copy()
    rel = live - origin
    gt = None
    if gt_xy is not None and len(gt_xy) >= 1:
        gt = np.asarray(gt_xy, dtype=np.float64).reshape(-1, 2)

    if map_polylines is None and tl is not None:
        s0 = int(start if start is not None else 0)
        s1 = int(end if end is not None else len(tl))
        map_polylines = collect_route_map_polylines(tl, s0, s1, stride=map_stride)
    map_polylines = map_polylines or {
        "lane_centerlines": [],
        "lane_borders": [],
        "road_borders": [],
        "route_lanes": [],
    }

    cl = np.asarray(clearances, dtype=np.float32)[:n]
    obj_coll = np.asarray(collisions, dtype=bool)[:n]
    finite = np.isfinite(cl)
    obj_miss = finite & (cl <= float(near_miss_thresh)) & ~obj_coll

    rb = np.asarray(rb_dists, dtype=np.float32)[:n]
    rb_finite = np.isfinite(rb)
    rb_coll = rb_finite & (rb < RB_COLLISION_THRESH_M)
    rb_miss = rb_finite & (rb <= float(near_miss_thresh)) & ~rb_coll

    red_mask = np.asarray(red_light, dtype=bool)[:n]
    brake_mask = strong_brake_mask(
        np.asarray(accels, dtype=np.float32)[:n], thresh_mps2=float(strong_brake_mps2)
    )

    layers = [
        ("object collision", _COLOR_OBJECT, _MARKER_BAD, _event_onsets(obj_coll)),
        ("object near-miss", _COLOR_OBJECT, _MARKER_DANGER, _event_onsets(obj_miss)),
        ("road_border collision", _COLOR_ROAD_BORDER, _MARKER_BAD, _event_onsets(rb_coll)),
        ("road_border near-miss", _COLOR_ROAD_BORDER, _MARKER_DANGER, _event_onsets(rb_miss)),
        ("red light", _COLOR_RED_LIGHT, _MARKER_BAD, _event_onsets(red_mask)),
        ("strong brake", _COLOR_STRONG_BRAKE, _MARKER_DANGER, _event_onsets(brake_mask)),
    ]

    fig = go.Figure()

    # Map underneath trajectories / events (thin + translucent so arrows/events dominate).
    _add_map_lines(
        fig, map_polylines.get("lane_centerlines", []), origin, color=_LANE_COLOR, width=0.2
    )
    _add_map_lines(
        fig, map_polylines.get("lane_borders", []), origin, color=_LANE_BORDER_COLOR, width=0.3
    )
    _add_map_lines(
        fig, map_polylines.get("road_borders", []), origin, color=_ROAD_BORDER_DRAW, width=0.45
    )
    _add_map_lines(fig, map_polylines.get("route_lanes", []), origin, color=_ROUTE_COLOR, width=0.6)

    # GT as a dashed path; live ego as thicker heading arrows (every traj_arrow_stride ticks).
    if gt is not None and len(gt) >= 2:
        gt_rel = gt - origin
        fig.add_trace(
            go.Scatter(
                x=gt_rel[:, 0],
                y=gt_rel[:, 1],
                mode="lines",
                name="recorded GT",
                line=dict(color=_COLOR_GT, width=2.0, dash="dash"),
                hovertemplate="GT x=%{x:.1f} y=%{y:.1f}<extra></extra>",
            )
        )

    live_sub, live_ang = _subsample_arrows(rel, stride=traj_arrow_stride)
    fig.add_trace(
        go.Scatter(
            x=live_sub[:, 0],
            y=live_sub[:, 1],
            mode="markers",
            name="live ego",
            marker=dict(
                symbol="arrow",
                size=14,
                color=_COLOR_LIVE,
                angle=live_ang.tolist(),
                line=dict(width=0),
            ),
            hovertemplate="live x=%{x:.1f} y=%{y:.1f}<extra></extra>",
        )
    )

    for label, color, marker, onsets in layers:
        xs, ys = _xy_at(live, onsets, origin)
        if xs.size == 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name=f"{label} ({len(onsets)})",
                marker=dict(
                    symbol=marker,
                    size=12 if marker == _MARKER_BAD else 10,
                    color=color,
                    line=dict(width=1.5 if marker == _MARKER_BAD else 1.0, color=color),
                ),
                hovertemplate=f"{label}<br>x=%{{x:.1f}} y=%{{y:.1f}}<extra></extra>",
            )
        )

    if snap_xy:
        snaps = np.asarray(snap_xy, dtype=np.float64).reshape(-1, 2)
        if snaps.size:
            fig.add_trace(
                go.Scatter(
                    x=snaps[:, 0] - origin[0],
                    y=snaps[:, 1] - origin[1],
                    mode="markers",
                    name=f"snap/teleport ({len(snaps)})",
                    marker=dict(
                        symbol=_MARKER_SNAP,
                        size=14,
                        color=_COLOR_SNAP,
                        line=dict(width=1, color="black"),
                    ),
                )
            )

    xmin, xmax, ymin, ymax = _viewport_bbox(rel, None if gt is None else gt - origin, margin_m)
    width_m = max(xmax - xmin, 1e-3)
    height_m = max(ymax - ymin, 1e-3)
    # Keep equal aspect via matching axis ranges + figure size.
    png_h = max(900, int(_PNG_WIDTH * height_m / width_m))

    summary_bits = []
    if metrics:
        obj = metrics.get("object") or {}
        rb_m = metrics.get("road_border") or {}
        red = metrics.get("red_light_violation") or {}
        brake = metrics.get("strong_brake") or {}
        repro = metrics.get("reproducer") or {}
        summary_bits = [
            f"obj coll={obj.get('collision_count', '?')} miss={obj.get('miss_count', '?')}",
            f"rb coll={rb_m.get('collision_count', '?')} miss={rb_m.get('miss_count', '?')}",
            f"red={red.get('count', '?')} brake={brake.get('count', '?')}",
            f"snap={repro.get('snap_count', '?')}",
        ]
    title_line = title or out_path.stem
    if summary_bits:
        title_line = (
            f"{title_line}<br><span style='font-size:12px'>" + " | ".join(summary_bits) + "</span>"
        )

    fig.update_layout(
        title=dict(text=title_line, x=0.5, xanchor="center"),
        width=_PNG_WIDTH,
        height=png_h,
        plot_bgcolor="#f8f8f8",
        paper_bgcolor="#f8f8f8",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(255,255,255,0.85)"
        ),
        margin=dict(l=60, r=40, t=100, b=60),
        xaxis=dict(
            title="X [m] (relative to start)",
            range=[xmin, xmax],
            scaleanchor="y",
            scaleratio=1,
            gridcolor="rgba(0,0,0,0.08)",
            zeroline=False,
        ),
        yaxis=dict(
            title="Y [m] (relative to start)",
            range=[ymin, ymax],
            gridcolor="rgba(0,0,0,0.08)",
            zeroline=False,
        ),
    )

    fig.write_html(str(html_path), include_plotlyjs="cdn")
    fig.write_image(str(out_path), width=_PNG_WIDTH, height=png_h, scale=_PNG_SCALE)
    return out_path
