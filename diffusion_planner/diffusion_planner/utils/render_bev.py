"""Render the vector inputs of one NPZ sample into multi-channel BEV images.

One image is produced per spatial scale (a near view and a far view), both centred on the
current ego pose with the ego heading pointing up.  An image is a stack of semantic binary
channels rather than an RGB picture, so overlapping elements never hide each other.

Motion is carried by the history channels: the past track of every agent is drawn as a
polyline ending at that agent's current bounding box.  Speed is recoverable from the length
of the trace and heading from the box, so no separate per-timestep raster is needed.

All coordinates in the NPZ are already expressed in the current ego frame; rendering only
re-centres them on the ego position and rotates the ego heading to the image's up axis.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from diffusion_planner.dimensions import (
    LB_X,
    LB_Y,
    RB_X,
    RB_Y,
    TRAFFIC_LIGHT_GREEN,
    TRAFFIC_LIGHT_RED,
    TRAFFIC_LIGHT_WHITE,
    TRAFFIC_LIGHT_YELLOW,
    X,
    Y,
)

# ---------------------------------------------------------------------------
# Raster layout
# ---------------------------------------------------------------------------

IMAGE_SIZE = 224
NEAR_EXTENT_M = 50.0
FAR_EXTENT_M = 200.0
VIEW_EXTENTS_M = (NEAR_EXTENT_M, FAR_EXTENT_M)
NUM_SCALES = len(VIEW_EXTENTS_M)

CH_LANE_BOUNDARY = 0
CH_LANE_CENTERLINE = 1
CH_ROUTE = 2
CH_TRAFFIC_LIGHT_GO = 3
CH_TRAFFIC_LIGHT_CAUTION = 4
CH_TRAFFIC_LIGHT_STOP = 5
CH_TRAFFIC_LIGHT_UNKNOWN = 6
CH_POLYGON = 7
CH_LINE_STRING = 8
CH_STATIC_OBJECT = 9
CH_GOAL_POSE = 10
CH_VEHICLE = 11
CH_PEDESTRIAN = 12
CH_BICYCLE = 13
CH_EGO = 14
CH_NEIGHBOR_HISTORY = 15
CH_EGO_HISTORY = 16
NUM_CHANNELS = 17

# Plane each traffic light state is drawn onto.  Every state gets its own plane instead of being
# folded into one "stop" plane, so the encoder can tell a yellow light from a red one, and a light
# whose state never arrived from a lane that has no light at all.  ``TRAFFIC_LIGHT_WHITE`` is the
# slot the observation writes when the lane has a light but its colour is unknown or was never
# received; it is by far the most common non-empty state in the recorded data.
#
# ``TRAFFIC_LIGHT_NO_TRAFFIC_LIGHT`` deliberately has no plane: such a lane is already drawn on the
# centerline channel, and its silence across all four light planes is what identifies it.
TRAFFIC_LIGHT_CHANNEL_OF_STATE = {
    TRAFFIC_LIGHT_GREEN: CH_TRAFFIC_LIGHT_GO,
    TRAFFIC_LIGHT_YELLOW: CH_TRAFFIC_LIGHT_CAUTION,
    TRAFFIC_LIGHT_RED: CH_TRAFFIC_LIGHT_STOP,
    TRAFFIC_LIGHT_WHITE: CH_TRAFFIC_LIGHT_UNKNOWN,
}

FILL = -1  # cv2 thickness value meaning "filled"
LINE_TYPE = cv2.LINE_8  # binary masks: keep hard edges, no anti-aliased partial values

LANE_BOUNDARY_THICKNESS = 1
LANE_CENTERLINE_THICKNESS = 1
TRAFFIC_LIGHT_THICKNESS = 2
ROUTE_THICKNESS = 3
LINE_STRING_THICKNESS = 1
HISTORY_THICKNESS = 2
GOAL_MARKER_LENGTH_M = 5.0
GOAL_MARKER_RADIUS_PX = 3

# Neighbor state layout inside ``neighbor_agents_past``
NEIGHBOR_IDX_COS = 2
NEIGHBOR_IDX_SIN = 3
NEIGHBOR_IDX_WIDTH = 6
NEIGHBOR_IDX_LENGTH = 7
NEIGHBOR_IDX_TYPE_START = 8
NEIGHBOR_TYPE_NUM = 3

# Static object state layout inside ``static_objects``
STATIC_IDX_COS = 2
STATIC_IDX_SIN = 3
STATIC_IDX_WIDTH = 4
STATIC_IDX_LENGTH = 5

VALID_EPS = 1e-6
MIN_BOX_SIZE_M = 0.1


@dataclass(frozen=True)
class BevView:
    """Pixel mapping of one scale: ego at the image centre, ego heading pointing up."""

    image_size: int
    extent_m: float

    @property
    def pixels_per_meter(self) -> float:
        return self.image_size / self.extent_m

    def to_pixel(self, points_rel: np.ndarray) -> np.ndarray:
        """Map (..., 2) ego-relative metres to (..., 2) integer (col, row) pixels."""
        scale = self.pixels_per_meter
        half = self.image_size / 2.0
        col = half - points_rel[..., 1] * scale
        row = half - points_rel[..., 0] * scale
        return np.stack([col, row], axis=-1).round().astype(np.int32)


def _yaw_from_cos_sin(cos_value: float, sin_value: float) -> float:
    return float(np.arctan2(sin_value, cos_value))


def _make_transform(ego_x: float, ego_y: float, ego_yaw: float):
    """Return a callable mapping (..., 2) points into the ego-centred, ego-aligned frame."""
    cos_yaw = np.cos(-ego_yaw)
    sin_yaw = np.sin(-ego_yaw)
    rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float64)
    origin = np.array([ego_x, ego_y], dtype=np.float64)

    def transform(points: np.ndarray) -> np.ndarray:
        return (np.asarray(points, dtype=np.float64) - origin) @ rotation.T

    return transform


def _valid_prefix_length(points: np.ndarray) -> int:
    """Number of leading points before the zero padding of a polyline."""
    magnitude = np.abs(points[:, :2]).sum(axis=1)
    invalid = np.flatnonzero(magnitude < VALID_EPS)
    if invalid.size == 0:
        return points.shape[0]
    return int(invalid[0])


def _draw_polyline(canvas: np.ndarray, view: BevView, points_rel: np.ndarray, thickness: int):
    if points_rel.shape[0] < 2:
        return
    cv2.polylines(
        canvas, [view.to_pixel(points_rel).reshape(-1, 1, 2)], False, 255, thickness, LINE_TYPE
    )


def _draw_box(
    canvas: np.ndarray,
    view: BevView,
    center_rel: np.ndarray,
    length: float,
    width: float,
    yaw_rel: float,
):
    """Fill an oriented bounding box; ``length`` runs along ``yaw_rel``."""
    half_length = max(length, MIN_BOX_SIZE_M) / 2.0
    half_width = max(width, MIN_BOX_SIZE_M) / 2.0
    local = np.array(
        [
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width],
        ],
        dtype=np.float64,
    )
    cos_yaw = np.cos(yaw_rel)
    sin_yaw = np.sin(yaw_rel)
    rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float64)
    corners = local @ rotation.T + center_rel
    cv2.fillPoly(canvas, [view.to_pixel(corners).reshape(-1, 1, 2)], 255, LINE_TYPE)


def _draw_pose_marker(canvas: np.ndarray, view: BevView, center_rel: np.ndarray, yaw_rel: float):
    """Mark a pose as a dot with a short segment pointing along the yaw."""
    tip = center_rel + np.array([np.cos(yaw_rel), np.sin(yaw_rel)]) * GOAL_MARKER_LENGTH_M
    _draw_polyline(canvas, view, np.stack([center_rel, tip]), 2)
    cv2.circle(
        canvas,
        tuple(view.to_pixel(center_rel).tolist()),
        GOAL_MARKER_RADIUS_PX,
        255,
        FILL,
        LINE_TYPE,
    )


def _traffic_light_channel(state: np.ndarray) -> int | None:
    """Plane the lane's traffic light belongs on, or None when the lane has no light at all."""
    for index, channel in TRAFFIC_LIGHT_CHANNEL_OF_STATE.items():
        if state[index] == 1:
            return channel
    return None


def _neighbor_channel(state: np.ndarray) -> int:
    type_one_hot = state[NEIGHBOR_IDX_TYPE_START : NEIGHBOR_IDX_TYPE_START + NEIGHBOR_TYPE_NUM]
    return (CH_VEHICLE, CH_PEDESTRIAN, CH_BICYCLE)[int(np.argmax(type_one_hot))]


# ---------------------------------------------------------------------------
# Per-element drawing
# ---------------------------------------------------------------------------


def _draw_lanes(canvas: np.ndarray, view: BevView, transform, lanes: np.ndarray):
    for i in range(lanes.shape[0]):
        num_points = _valid_prefix_length(lanes[i])
        if num_points < 2:
            continue
        segment = lanes[i, :num_points]
        center = transform(segment[:, [X, Y]])
        left = transform(segment[:, [X, Y]] + segment[:, [LB_X, LB_Y]])
        right = transform(segment[:, [X, Y]] + segment[:, [RB_X, RB_Y]])

        _draw_polyline(canvas[CH_LANE_CENTERLINE], view, center, LANE_CENTERLINE_THICKNESS)
        _draw_polyline(canvas[CH_LANE_BOUNDARY], view, left, LANE_BOUNDARY_THICKNESS)
        _draw_polyline(canvas[CH_LANE_BOUNDARY], view, right, LANE_BOUNDARY_THICKNESS)

        # The state one-hot is replicated across the segment, so the first point carries it.
        channel = _traffic_light_channel(segment[0])
        if channel is not None:
            _draw_polyline(canvas[channel], view, center, TRAFFIC_LIGHT_THICKNESS)


def _draw_route(canvas: np.ndarray, view: BevView, transform, route_lanes: np.ndarray):
    for i in range(route_lanes.shape[0]):
        num_points = _valid_prefix_length(route_lanes[i])
        if num_points < 2:
            continue
        center = transform(route_lanes[i, :num_points][:, [X, Y]])
        _draw_polyline(canvas[CH_ROUTE], view, center, ROUTE_THICKNESS)


def _draw_polygons(canvas: np.ndarray, view: BevView, transform, polygons: np.ndarray):
    for i in range(polygons.shape[0]):
        num_points = _valid_prefix_length(polygons[i])
        if num_points < 3:
            continue
        contour = transform(polygons[i, :num_points, :2])
        cv2.fillPoly(canvas[CH_POLYGON], [view.to_pixel(contour).reshape(-1, 1, 2)], 255, LINE_TYPE)


def _draw_line_strings(canvas: np.ndarray, view: BevView, transform, line_strings: np.ndarray):
    for i in range(line_strings.shape[0]):
        num_points = _valid_prefix_length(line_strings[i])
        if num_points < 2:
            continue
        points = transform(line_strings[i, :num_points, :2])
        _draw_polyline(canvas[CH_LINE_STRING], view, points, LINE_STRING_THICKNESS)


def _draw_static_objects(
    canvas: np.ndarray, view: BevView, transform, static_objects: np.ndarray, ego_yaw: float
):
    for i in range(static_objects.shape[0]):
        obj = static_objects[i]
        if np.abs(obj[:4]).sum() < VALID_EPS:
            continue
        _draw_box(
            canvas[CH_STATIC_OBJECT],
            view,
            transform(obj[:2]),
            float(obj[STATIC_IDX_LENGTH]),
            float(obj[STATIC_IDX_WIDTH]),
            _yaw_from_cos_sin(obj[STATIC_IDX_COS], obj[STATIC_IDX_SIN]) - ego_yaw,
        )


def _draw_neighbors(
    canvas: np.ndarray, view: BevView, transform, neighbors: np.ndarray, ego_yaw: float
):
    """Draw each neighbor's past track plus its current box."""
    current = neighbors.shape[1] - 1
    for i in range(neighbors.shape[0]):
        state = neighbors[i, current]
        if np.abs(state[:4]).sum() < VALID_EPS:
            continue

        track = neighbors[i, :, :2]
        observed = np.abs(track).sum(axis=1) > VALID_EPS
        if observed.sum() > 1:
            _draw_polyline(
                canvas[CH_NEIGHBOR_HISTORY], view, transform(track[observed]), HISTORY_THICKNESS
            )

        _draw_box(
            canvas[_neighbor_channel(state)],
            view,
            transform(state[:2]),
            float(state[NEIGHBOR_IDX_LENGTH]),
            float(state[NEIGHBOR_IDX_WIDTH]),
            _yaw_from_cos_sin(state[NEIGHBOR_IDX_COS], state[NEIGHBOR_IDX_SIN]) - ego_yaw,
        )


def _draw_ego(
    canvas: np.ndarray,
    view: BevView,
    transform,
    ego_past: np.ndarray,
    ego_current: np.ndarray,
    ego_shape: np.ndarray,
    ego_yaw: float,
):
    _draw_polyline(canvas[CH_EGO_HISTORY], view, transform(ego_past[:, :2]), HISTORY_THICKNESS)
    _draw_box(
        canvas[CH_EGO],
        view,
        transform(ego_current[:2]),
        float(ego_shape[1]),
        float(ego_shape[2]),
        _yaw_from_cos_sin(ego_current[2], ego_current[3]) - ego_yaw,
    )


def _draw_goal_pose(
    canvas: np.ndarray, view: BevView, transform, goal_pose: np.ndarray, ego_yaw: float
):
    if np.abs(goal_pose[:2]).sum() < VALID_EPS:
        return
    if goal_pose.shape[0] >= 4:
        goal_yaw = _yaw_from_cos_sin(goal_pose[2], goal_pose[3])
    else:
        goal_yaw = float(goal_pose[2])
    _draw_pose_marker(canvas[CH_GOAL_POSE], view, transform(goal_pose[:2]), goal_yaw - ego_yaw)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def render_view(data: dict, view: BevView, transform, ego_yaw: float) -> np.ndarray:
    """Rasterise one scale into a (NUM_CHANNELS, H, W) uint8 stack."""
    canvas = np.zeros((NUM_CHANNELS, view.image_size, view.image_size), dtype=np.uint8)

    _draw_lanes(canvas, view, transform, data["lanes"])
    _draw_route(canvas, view, transform, data["route_lanes"])
    _draw_polygons(canvas, view, transform, data["polygons"])
    _draw_line_strings(canvas, view, transform, data["line_strings"])
    _draw_static_objects(canvas, view, transform, data["static_objects"], ego_yaw)
    _draw_goal_pose(canvas, view, transform, np.asarray(data["goal_pose"]).reshape(-1), ego_yaw)
    _draw_neighbors(canvas, view, transform, data["neighbor_agents_past"], ego_yaw)
    _draw_ego(
        canvas,
        view,
        transform,
        data["ego_agent_past"],
        np.asarray(data["ego_current_state"]).reshape(-1),
        np.asarray(data["ego_shape"]).reshape(-1),
        ego_yaw,
    )
    return canvas


def render_sample(data: dict, image_size: int, view_extents_m: tuple) -> np.ndarray:
    """Rasterise one unbatched sample into (S, NUM_CHANNELS, H, W) uint8 in {0, 255}.

    S indexes ``view_extents_m``; every scale shares the current ego pose as its origin.
    """
    ego_current = np.asarray(data["ego_current_state"]).reshape(-1)
    ego_yaw = _yaw_from_cos_sin(ego_current[2], ego_current[3])
    transform = _make_transform(float(ego_current[0]), float(ego_current[1]), ego_yaw)

    views = [BevView(image_size=image_size, extent_m=extent) for extent in view_extents_m]
    return np.stack([render_view(data, view, transform, ego_yaw) for view in views], axis=0)


# ---------------------------------------------------------------------------
# Debug preview
# ---------------------------------------------------------------------------

CHANNEL_COLORS_BGR = {
    CH_LANE_BOUNDARY: (160, 160, 160),
    CH_LANE_CENTERLINE: (90, 90, 90),
    CH_ROUTE: (255, 0, 255),
    CH_TRAFFIC_LIGHT_GO: (0, 200, 0),
    CH_TRAFFIC_LIGHT_CAUTION: (0, 255, 255),
    CH_TRAFFIC_LIGHT_STOP: (0, 0, 255),
    CH_TRAFFIC_LIGHT_UNKNOWN: (128, 0, 128),
    CH_POLYGON: (60, 110, 60),
    CH_LINE_STRING: (0, 140, 255),
    CH_STATIC_OBJECT: (0, 200, 200),
    CH_GOAL_POSE: (255, 128, 0),
    CH_NEIGHBOR_HISTORY: (128, 64, 0),
    CH_EGO_HISTORY: (0, 165, 255),
    CH_VEHICLE: (255, 60, 60),
    CH_PEDESTRIAN: (60, 255, 60),
    CH_BICYCLE: (255, 0, 128),
    CH_EGO: (255, 255, 255),
}

# Later entries win where channels overlap, so agents stay visible on top of the map.  Lanes cross
# at an intersection, so the light planes are ordered by how much they demand attention: an
# unknown light loses to a green one, which loses to yellow, which loses to red.
PREVIEW_DRAW_ORDER = (
    CH_POLYGON,
    CH_LANE_CENTERLINE,
    CH_LANE_BOUNDARY,
    CH_LINE_STRING,
    CH_ROUTE,
    CH_TRAFFIC_LIGHT_UNKNOWN,
    CH_TRAFFIC_LIGHT_GO,
    CH_TRAFFIC_LIGHT_CAUTION,
    CH_TRAFFIC_LIGHT_STOP,
    CH_GOAL_POSE,
    CH_NEIGHBOR_HISTORY,
    CH_EGO_HISTORY,
    CH_STATIC_OBJECT,
    CH_VEHICLE,
    CH_PEDESTRIAN,
    CH_BICYCLE,
    CH_EGO,
)

# A channel missing from either table would silently vanish from every preview, which is exactly
# how a folded-together traffic light state went unnoticed before.
assert len(PREVIEW_DRAW_ORDER) == NUM_CHANNELS
assert set(PREVIEW_DRAW_ORDER) == set(range(NUM_CHANNELS))
assert set(CHANNEL_COLORS_BGR) == set(PREVIEW_DRAW_ORDER)


def colorize(image: np.ndarray) -> np.ndarray:
    """Composite one (NUM_CHANNELS, H, W) raster into a BGR preview image."""
    height, width = image.shape[-2:]
    preview = np.zeros((height, width, 3), dtype=np.uint8)
    for channel in PREVIEW_DRAW_ORDER:
        preview[image[channel] > 0] = CHANNEL_COLORS_BGR[channel]
    return preview


def _main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, help="an NPZ file, or a directory of NPZ files")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--image_size", type=int, required=True)
    parser.add_argument("--skip", type=int, required=True, help="render every Nth file")
    args = parser.parse_args()

    if args.target.is_dir():
        npz_paths = sorted(args.target.glob("*.npz"))[:: args.skip]
    else:
        npz_paths = [args.target]

    scale_dirs = [args.out_dir / f"extent{int(extent)}m" for extent in VIEW_EXTENTS_M]
    for scale_dir in scale_dirs:
        scale_dir.mkdir(parents=True, exist_ok=True)

    for npz_path in npz_paths:
        data = dict(np.load(npz_path, allow_pickle=True))
        images = render_sample(data, args.image_size, VIEW_EXTENTS_M)
        for scale, scale_dir in enumerate(scale_dirs):
            cv2.imwrite(str(scale_dir / f"{npz_path.stem}.png"), colorize(images[scale]))
    print(f"rendered {len(npz_paths)} samples x {NUM_SCALES} scales into {args.out_dir}")


if __name__ == "__main__":
    _main()
