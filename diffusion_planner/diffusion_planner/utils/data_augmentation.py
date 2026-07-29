import math

import numpy as np
import torch

from diffusion_planner.utils.masks import (
    lane_point_padding_mask,
    neighbor_future_padding_mask,
    neighbor_past_padding_mask,
    pose_padding_mask,
    static_object_padding_mask,
    zero_padding_mask,
)
from diffusion_planner.utils.unicycle_accel_curvature import (
    UnicycleAccelCurvatureActionSpace,
    smoothing_future_trajectory,
)

TIME_INTERVAL = 0.1


def vector_transform(vector, transform_mat, bias=None):
    """
    vector: (B, ..., 2)
    transform_mat: (B, 2, 2)
    bias: (B, ..., 2)
    """
    shape = vector.shape
    B = vector.shape[0]
    nexpand = vector.ndim - 2
    if bias is not None:
        vector = vector - bias.reshape(B, *([1] * nexpand), -1)
    vector = vector.reshape(B, -1, 2).permute(0, 2, 1)  # (B, 2, N1 * N2 ...)
    return torch.bmm(transform_mat, vector).permute(0, 2, 1).reshape(*shape)  # (B, ..., 2)


def heading_transform(heading, transform_mat):
    """
    heading: (B, ...)
    transform_mat: (B, 2, 2)
    """
    B = heading.shape[0]
    shape = heading.shape
    heading = heading.reshape(B, -1)
    transform_mat = transform_mat.reshape(B, 1, 2, 2)
    return torch.atan2(
        torch.cos(heading) * transform_mat[..., 1, 0]
        + torch.sin(heading) * transform_mat[..., 1, 1],
        torch.cos(heading) * transform_mat[..., 0, 0]
        + torch.sin(heading) * transform_mat[..., 0, 1],
    ).reshape(*shape)


def _rect_corners(rect: torch.Tensor) -> torch.Tensor:
    """
    rect: [B, 6] — (x, y, cos_h, sin_h, length, width)
    Returns [B, 4, 2] corner points.
    """
    B = rect.shape[0]
    xy, cos_h, sin_h, lw = rect[:, :2], rect[:, 2], rect[:, 3], rect[:, 4:]
    rot = torch.stack([cos_h, -sin_h, sin_h, cos_h], dim=1).reshape(B, 2, 2)
    signs = lw.new_tensor([[1.0, 1], [-1, 1], [-1, -1], [1, -1]])
    local = torch.einsum("bj,ij->bij", lw / 2, signs)  # [B, 4, 2]
    local = torch.einsum("bij,bkj->bik", local, rot)  # [B, 4, 2]
    return xy[:, None, :] + local


def _sat_signed_distance(c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
    """
    SAT signed distance between two rectangles.
    c1, c2: [B, 4, 2] corner points
    Returns [B] — negative means overlap.
    """
    nv = torch.stack(
        [c1[:, 0] - c1[:, 1], c1[:, 1] - c1[:, 2], c2[:, 0] - c2[:, 1], c2[:, 1] - c2[:, 2]],
        dim=1,
    )  # [B, 4, 2]
    nv = nv / torch.norm(nv, dim=2, keepdim=True).clamp(min=1e-6)
    p1 = torch.einsum("bij,bkj->bik", nv, c1)  # [B, 4, 4]
    p2 = torch.einsum("bij,bkj->bik", nv, c2)
    overlap = torch.cat(
        [p1.min(2).values - p2.max(2).values, p2.min(2).values - p1.max(2).values],
        dim=1,
    )  # [B, 8]
    is_overlap = (overlap < 0).all(dim=1)
    pos = torch.where(overlap < 0, torch.full_like(overlap, 1e5), overlap)
    return torch.where(is_overlap, overlap.max(1).values, pos.min(1).values)


class StatePerturbation:
    """
    Data augmentation that perturbs the current ego position and generates a feasible trajectory that
    satisfies polynomial constraints.
    """

    def __init__(
        self,
        augment_prob: float = 0.5,
        num_refine: int = 20,
        device: torch.device | str = "cpu",
        ego_past_noise_std: float = 0.1,
        use_smoothing_future_trajectory: bool = True,
        wheel_base: float = 2.75,
    ) -> None:
        """
        Initialize the augmentor,
        :param augment_prob: probability between 0 and 1 of applying the data augmentation
        :param num_refine: number of refinement steps for quintic interpolation
        :param device: torch device
        :param ego_past_noise_std: std of noise applied to ego past trajectory
        :param use_smoothing_future_trajectory: whether to apply smoothing to future trajectory
        """
        if not math.isfinite(augment_prob) or not 0.0 <= augment_prob <= 1.0:
            raise ValueError("augment_prob must be finite and in [0, 1]")
        if not math.isfinite(ego_past_noise_std) or ego_past_noise_std < 0.0:
            raise ValueError("ego_past_noise_std must be finite and >= 0")
        if not math.isfinite(wheel_base) or wheel_base <= 0.0:
            raise ValueError("wheel_base must be finite and > 0")
        if num_refine < 3:
            raise ValueError("num_refine must be >= 3 for the terminal finite differences")
        self._augment_prob = augment_prob
        self._device = torch.device(device)
        self._ego_past_noise_std = ego_past_noise_std
        self._use_smoothing_future_trajectory = use_smoothing_future_trajectory
        self._smoothing_action_space = (
            UnicycleAccelCurvatureActionSpace(n_waypoints=80).to(self._device)
            if use_smoothing_future_trajectory
            else None
        )
        # Production samples carry per-vehicle wheelbase in ego_shape. Keep the historical
        # constructor value as a fallback for lightweight callers and unit tests.
        self._wheel_base = wheel_base
        low = [0.0, -0.75, -0.2, -1, -0.5, -0.2, -0.1, 0.0, 0.0]
        high = [0.0, +0.75, +0.2, +1, +0.5, +0.2, +0.1, 0.0, 0.0]
        self._low = torch.tensor(low, dtype=torch.float32, device=self._device)
        self._high = torch.tensor(high, dtype=torch.float32, device=self._device)

        self.num_refine = num_refine
        self.time_interval = TIME_INTERVAL

        REFINE_HORIZON = num_refine * TIME_INTERVAL

        T = REFINE_HORIZON + TIME_INTERVAL
        self.coeff_matrix = torch.linalg.inv(
            torch.tensor(
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 2, 0, 0, 0],
                    [1, T, T**2, T**3, T**4, T**5],
                    [0, 1, 2 * T, 3 * T**2, 4 * T**3, 5 * T**4],
                    [0, 0, 2, 6 * T, 12 * T**2, 20 * T**3],
                ],
                device=device,
                dtype=torch.float32,
            )
        )
        self.t_matrix = torch.pow(
            torch.linspace(TIME_INTERVAL, REFINE_HORIZON, num_refine).unsqueeze(1),
            torch.arange(6).unsqueeze(0),
        ).to(device=device)  # shape (B, N+1)

    def __call__(self, inputs, ego_future, neighbors_future):
        aug_flag, aug_ego_current_state = self.augment(inputs)

        # Apply the history/state scale before constructing the quintic target so the
        # observed current velocity and the target's initial boundary condition agree.
        W = self._ego_past_noise_std
        scale = torch.normal(
            mean=1.0,
            std=W,
            size=(aug_flag.shape[0], 1, 1),
            device=inputs["ego_agent_past"].device,
        ).clamp(1.0 - 2 * W, 1.0 + 2 * W)
        ego_past_aug = inputs["ego_agent_past"].clone()
        ego_past_aug[..., :2] *= scale
        # Keep the observed (blue) history in its current-ego frame.  The sampled
        # pose is a hypothetical current-state offset relative to the scene, not a
        # request to synthesize/interpolate a new history.  This preserves the
        # online contract that the last history pose is the current origin; the
        # optional history transform in centric_transform remains available to
        # callers that explicitly need a rigid scene transform.
        inputs["ego_agent_past"] = torch.where(
            aug_flag[:, None, None], ego_past_aug, inputs["ego_agent_past"]
        )

        scale_1d = scale.squeeze(-1)
        aug_ego_current_state[:, 4:6] *= scale_1d
        aug_ego_current_state[:, 6:8] *= scale_1d
        if "ego_shape" in inputs:
            ego_shape = inputs["ego_shape"]
            if ego_shape.ndim != 2 or ego_shape.shape != (aug_ego_current_state.shape[0], 3):
                raise ValueError(
                    f"ego_shape must have shape {(aug_ego_current_state.shape[0], 3)} "
                    f"for augmentation, got {tuple(ego_shape.shape)}"
                )
            wheel_base = ego_shape[:, 0]
        else:
            wheel_base = aug_ego_current_state.new_full(
                (aug_ego_current_state.shape[0],), self._wheel_base
            )
        speed = aug_ego_current_state[:, 4].abs()
        steering = torch.atan(
            aug_ego_current_state[:, 9] * wheel_base / speed.clamp_min(0.2)
        ).clamp(-2 / 3 * np.pi, 2 / 3 * np.pi)
        steering = torch.where(speed >= 0.2, steering, torch.zeros_like(steering))
        aug_ego_current_state[:, 8] = steering

        interpolated_ego_future = self.interpolation_future_trajectory(
            aug_ego_current_state, ego_future
        )
        inputs["ego_current_state"] = torch.where(
            aug_flag[:, None], aug_ego_current_state, inputs["ego_current_state"]
        )
        ego_future = torch.where(aug_flag[:, None, None], interpolated_ego_future, ego_future)

        return self.centric_transform(
            inputs,
            ego_future,
            neighbors_future,
            transform_ego_history=False,
        )

    def augment(self, inputs):
        # Only aug current state
        ego_current_state = inputs["ego_current_state"].clone()
        B = ego_current_state.shape[0]
        if "ego_shape" in inputs:
            wheel_base = inputs["ego_shape"][:, 0]  # (B,)
        else:
            wheel_base = ego_current_state.new_full((B,), self._wheel_base)
        aug_flag = (torch.rand(B, device=self._device) < self._augment_prob) & ~(
            abs(ego_current_state[:, 4]) < 2.0
        )

        random_tensor = torch.rand(B, len(self._low), device=self._device)
        scaled_random_tensor = self._low + (self._high - self._low) * random_tensor

        new_state = torch.zeros((B, 9), dtype=torch.float32, device=self._device)
        new_state[:, 3:] = ego_current_state[
            :, 4:10
        ]  # x, y, h is 0 because of ego-centric, update vx, vy, ax, ay, steering angle, yaw rate
        new_state = new_state + scaled_random_tensor
        new_state[:, 3].clamp_min_(0.0)
        new_state[:, -1] = torch.clip(new_state[:, -1], -0.85, 0.85)

        ego_current_state[:, :2] = new_state[:, :2]
        ego_current_state[:, 2] = torch.cos(new_state[:, 2])
        ego_current_state[:, 3] = torch.sin(new_state[:, 2])
        ego_current_state[:, 4:8] = new_state[:, 3:7]
        ego_current_state[:, 8:10] = new_state[:, -2:]  # steering angle, yaw rate

        # update steering angle and yaw rate
        cur_velocity = ego_current_state[:, 4]
        yaw_rate = ego_current_state[:, 9]

        mask = torch.abs(cur_velocity) < 0.2
        not_mask = ~mask
        steering_angle = torch.atan(yaw_rate * wheel_base / torch.abs(cur_velocity).clamp_min(0.2))
        steering_angle = torch.where(
            not_mask,
            torch.clamp(steering_angle, -2 / 3 * np.pi, 2 / 3 * np.pi),
            torch.zeros_like(steering_angle),
        )
        new_yaw_rate = torch.where(not_mask, yaw_rate, torch.zeros_like(yaw_rate))

        ego_current_state[:, 8] = steering_angle
        ego_current_state[:, 9] = new_yaw_rate

        # Discard augmentations that cause collisions
        collision = self._check_aug_validity(ego_current_state, inputs)
        aug_flag = aug_flag & ~collision

        return aug_flag, ego_current_state

    def _check_aug_validity(self, aug_ego_state: torch.Tensor, inputs: dict) -> torch.Tensor:
        """
        Returns [B] bool — True where the augmented ego position is invalid.

        An augmented pose is invalid when its ego polygon overlaps a neighbour polygon.

        Lane boundaries are deliberately not treated as solid obstacles. ``lanes`` contains
        every nearby lane, including internal markings and overlapping intersection geometry;
        rejecting against all of them suppresses even centimetre-scale recovery perturbations.
        """
        B = aug_ego_state.shape[0]
        device = aug_ego_state.device
        dtype = aug_ego_state.dtype

        collision = torch.zeros(B, dtype=torch.bool, device=device)

        # ── 1. Neighbour agent polygon collision ──────────────────────────────
        if "neighbor_agents_past" in inputs:
            nbr = inputs["neighbor_agents_past"][:, :, -1, :]  # [B, N, 11]
            N = nbr.shape[1]
            valid = torch.any(nbr[:, :, :4] != 0, dim=-1)  # [B, N]
            if valid.any():
                if "ego_shape" not in inputs:
                    raise KeyError(
                        "ego_shape is required for augmentation collision checks when "
                        "neighbor_agents_past contains a valid agent"
                    )
                ego_shape = inputs["ego_shape"].to(device=device, dtype=dtype)
                if ego_shape.ndim != 2 or ego_shape.shape != (B, 3):
                    raise ValueError(
                        f"ego_shape must have shape {(B, 3)} for augmentation, "
                        f"got {tuple(ego_shape.shape)}"
                    )
                ego_length = ego_shape[:, 1:2]  # [B, 1]
                ego_width = ego_shape[:, 2:3]  # [B, 1]

                heading = aug_ego_state[:, 2:4]
                heading = heading / torch.linalg.norm(heading, dim=-1, keepdim=True).clamp_min(1e-6)
                # Ego poses are rear-axle referenced while collision boxes are center
                # referenced. Match the convention used by training penalties and
                # planner metrics.
                ego_center = aug_ego_state[:, :2] + heading * (ego_shape[:, 0:1] * 0.5)
                ego_rect = torch.cat([ego_center, heading, ego_length, ego_width], dim=-1)
                # neighbor_agents_past layout: x,y,cos,sin (0:4), width (6), length (7)
                nbr_rect = torch.cat(
                    [nbr[:, :, :4], nbr[:, :, 7:8], nbr[:, :, 6:7]], dim=-1
                )  # [B, N, 6]  — (x,y,cos,sin,length,width)
                dists = _sat_signed_distance(
                    _rect_corners(ego_rect.unsqueeze(1).expand(-1, N, -1).reshape(B * N, 6)),
                    _rect_corners(nbr_rect.reshape(B * N, 6)),
                ).reshape(B, N)
                collision = collision | ((dists < 0) & valid).any(dim=1)

        return collision

    def normalize_angle(self, angle: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def get_transform_matrix_batch(self, cur_state):
        processed_input = torch.column_stack(
            (
                cur_state[:, 2],  # cos
                cur_state[:, 3],  # sin
            )
        )

        reshaping_tensor = processed_input.new_tensor(
            [
                [1, 0, 0, 1],
                [0, 1, -1, 0],
            ]
        )
        return (processed_input @ reshaping_tensor).reshape(-1, 2, 2)

    def centric_transform(
        self,
        inputs: torch.Tensor,
        ego_future: torch.Tensor,
        neighbors_future: torch.Tensor,
        *,
        transform_ego_history: bool = True,
    ):
        cur_state = inputs["ego_current_state"].clone()
        center_xy = cur_state[:, :2]
        transform_matrix = self.get_transform_matrix_batch(cur_state)

        # ego xy
        inputs["ego_current_state"][..., :2] = vector_transform(
            inputs["ego_current_state"][..., :2], transform_matrix, center_xy
        )
        # ego cos sin
        inputs["ego_current_state"][..., 2:4] = vector_transform(
            inputs["ego_current_state"][..., 2:4], transform_matrix
        )
        # ego vx, vy
        inputs["ego_current_state"][..., 4:6] = vector_transform(
            inputs["ego_current_state"][..., 4:6], transform_matrix
        )
        # ego ax, ay
        inputs["ego_current_state"][..., 6:8] = vector_transform(
            inputs["ego_current_state"][..., 6:8], transform_matrix
        )

        # ego future
        ego_future_is_heading = ego_future.shape[-1] == 3
        ego_future[..., :2] = vector_transform(ego_future[..., :2], transform_matrix, center_xy)
        if ego_future_is_heading:
            ego_future[..., 2] = heading_transform(ego_future[..., 2], transform_matrix)
            ego_future4d = torch.cat(
                [
                    ego_future[..., :2],
                    torch.cos(ego_future[..., 2:3]),
                    torch.sin(ego_future[..., 2:3]),
                ],
                dim=-1,
            )
        else:
            ego_future[..., 2:4] = vector_transform(ego_future[..., 2:4], transform_matrix)
            ego_future4d = ego_future

        ego_past_is_heading = inputs["ego_agent_past"].shape[-1] == 3
        # Raw 3-column ego poses use (0, 0, 0) for a valid origin. Only a
        # converted 4-column tensor can use the all-zero padding sentinel.
        ego_past_mask = None if ego_past_is_heading else pose_padding_mask(inputs["ego_agent_past"])
        if transform_ego_history:
            inputs["ego_agent_past"][..., :2] = vector_transform(
                inputs["ego_agent_past"][..., :2], transform_matrix, center_xy
            )
            if ego_past_is_heading:
                inputs["ego_agent_past"][..., 2] = heading_transform(
                    inputs["ego_agent_past"][..., 2], transform_matrix
                )
            else:
                inputs["ego_agent_past"][..., 2:4] = vector_transform(
                    inputs["ego_agent_past"][..., 2:4], transform_matrix
                )
            if ego_past_mask is not None:
                inputs["ego_agent_past"].masked_fill_(ego_past_mask.unsqueeze(-1), 0.0)

        if self._use_smoothing_future_trajectory:
            smoothing_ego_past = inputs["ego_agent_past"]
            if ego_past_is_heading:
                smoothing_ego_past = torch.cat(
                    [
                        smoothing_ego_past[..., :2],
                        torch.cos(smoothing_ego_past[..., 2:3]),
                        torch.sin(smoothing_ego_past[..., 2:3]),
                    ],
                    dim=-1,
                )
                if ego_past_mask is not None:
                    smoothing_ego_past.masked_fill_(ego_past_mask.unsqueeze(-1), 0.0)
            ego_future4d = smoothing_future_trajectory(
                smoothing_ego_past,
                inputs["ego_current_state"],
                ego_future4d,
                action_space=self._smoothing_action_space,
            )

        if ego_future_is_heading:
            ego_future = torch.cat(
                [
                    ego_future4d[..., :2],
                    torch.atan2(ego_future4d[..., 3], ego_future4d[..., 2]).unsqueeze(-1),
                ],
                dim=-1,
            )
        else:
            ego_future = ego_future4d
        inputs["ego_agent_future"] = ego_future

        # neighbor past xy
        mask = neighbor_past_padding_mask(inputs["neighbor_agents_past"])
        inputs["neighbor_agents_past"][..., :2] = vector_transform(
            inputs["neighbor_agents_past"][..., :2], transform_matrix, center_xy
        )
        # neighbor past cos sin
        inputs["neighbor_agents_past"][..., 2:4] = vector_transform(
            inputs["neighbor_agents_past"][..., 2:4], transform_matrix
        )
        # neighbor past vx, vy
        inputs["neighbor_agents_past"][..., 4:6] = vector_transform(
            inputs["neighbor_agents_past"][..., 4:6], transform_matrix
        )
        inputs["neighbor_agents_past"].masked_fill_(mask.unsqueeze(-1), 0.0)

        # neighbor future
        mask = neighbor_future_padding_mask(neighbors_future)
        neighbors_future[..., :2] = vector_transform(
            neighbors_future[..., :2], transform_matrix, center_xy
        )
        if neighbors_future.shape[-1] == 3:
            neighbors_future[..., 2] = heading_transform(neighbors_future[..., 2], transform_matrix)
        else:
            neighbors_future[..., 2:4] = vector_transform(
                neighbors_future[..., 2:4], transform_matrix
            )
        neighbors_future.masked_fill_(mask.unsqueeze(-1), 0.0)

        # lanes
        mask = lane_point_padding_mask(inputs["lanes"])
        inputs["lanes"][..., :2] = vector_transform(
            inputs["lanes"][..., :2], transform_matrix, center_xy
        )
        inputs["lanes"][..., 2:4] = vector_transform(inputs["lanes"][..., 2:4], transform_matrix)
        inputs["lanes"][..., 4:6] = vector_transform(inputs["lanes"][..., 4:6], transform_matrix)
        inputs["lanes"][..., 6:8] = vector_transform(inputs["lanes"][..., 6:8], transform_matrix)
        inputs["lanes"].masked_fill_(mask.unsqueeze(-1), 0.0)

        # route_lanes
        mask = lane_point_padding_mask(inputs["route_lanes"])
        inputs["route_lanes"][..., :2] = vector_transform(
            inputs["route_lanes"][..., :2], transform_matrix, center_xy
        )
        inputs["route_lanes"][..., 2:4] = vector_transform(
            inputs["route_lanes"][..., 2:4], transform_matrix
        )
        inputs["route_lanes"][..., 4:6] = vector_transform(
            inputs["route_lanes"][..., 4:6], transform_matrix
        )
        inputs["route_lanes"][..., 6:8] = vector_transform(
            inputs["route_lanes"][..., 6:8], transform_matrix
        )
        inputs["route_lanes"].masked_fill_(mask.unsqueeze(-1), 0.0)

        # polygons
        mask = zero_padding_mask(inputs["polygons"])
        inputs["polygons"][..., :2] = vector_transform(
            inputs["polygons"][..., :2], transform_matrix, center_xy
        )
        inputs["polygons"].masked_fill_(mask.unsqueeze(-1), 0.0)

        # line_strings
        mask = zero_padding_mask(inputs["line_strings"])
        inputs["line_strings"][..., :2] = vector_transform(
            inputs["line_strings"][..., :2], transform_matrix, center_xy
        )
        inputs["line_strings"].masked_fill_(mask.unsqueeze(-1), 0.0)

        # static objects xy
        mask = static_object_padding_mask(inputs["static_objects"])
        inputs["static_objects"][..., :2] = vector_transform(
            inputs["static_objects"][..., :2], transform_matrix, center_xy
        )
        # static objects cos sin
        inputs["static_objects"][..., 2:4] = vector_transform(
            inputs["static_objects"][..., 2:4], transform_matrix
        )
        inputs["static_objects"].masked_fill_(mask.unsqueeze(-1), 0.0)

        if "goal_pose" in inputs:
            goal_is_heading = inputs["goal_pose"].shape[-1] == 3
            goal_mask = None if goal_is_heading else torch.all(inputs["goal_pose"] == 0, dim=-1)
            inputs["goal_pose"][..., :2] = vector_transform(
                inputs["goal_pose"][..., :2], transform_matrix, center_xy
            )
            if inputs["goal_pose"].shape[-1] == 3:
                inputs["goal_pose"][..., 2] = heading_transform(
                    inputs["goal_pose"][..., 2], transform_matrix
                )
            else:
                inputs["goal_pose"][..., 2:4] = vector_transform(
                    inputs["goal_pose"][..., 2:4], transform_matrix
                )
            if goal_mask is not None:
                inputs["goal_pose"].masked_fill_(goal_mask.unsqueeze(-1), 0.0)

        return inputs, ego_future, neighbors_future

    def interpolation_future_trajectory(self, aug_current_state, ego_future, keep_remaining=True):
        """
        refine future trajectory with quintic Hermite interpolation

        Args:
            aug_current_state: (B, 16) current state of the ego vehicle after augmentation
            ego_future:        (B, T, 3) future trajectory of the ego vehicle
            keep_remaining:    If True, keep the remaining trajectory after P frames (default: True)

        Returns:
            ego_future: refined future trajectory of the ego vehicle
        """

        future_is_cos_sin = ego_future.shape[-1] == 4
        if future_is_cos_sin:
            ego_future_heading = torch.cat(
                [
                    ego_future[..., :2],
                    torch.atan2(ego_future[..., 3], ego_future[..., 2]).unsqueeze(-1),
                ],
                dim=-1,
            )
        else:
            ego_future_heading = ego_future

        P = self.num_refine
        if ego_future.shape[1] <= P:
            raise ValueError(
                f"num_refine ({P}) must be smaller than the future horizon ({ego_future.shape[1]})"
            )
        dt = self.time_interval
        B = aug_current_state.shape[0]
        # The training path is float32, but this utility is also used directly by
        # diagnostics and exporters. Adapt the cached constants rather than letting
        # a valid double/other-dtype caller fail at matmul with a mixed-dtype error.
        M_t = self.t_matrix.to(device=aug_current_state.device, dtype=aug_current_state.dtype)
        M_t = M_t.unsqueeze(0).expand(B, -1, -1)
        A = self.coeff_matrix.to(device=aug_current_state.device, dtype=aug_current_state.dtype)
        A = A.unsqueeze(0).expand(B, -1, -1)

        # state: [x, y, heading, velocity, acceleration, yaw_rate]

        theta0 = torch.atan2(
            ego_future_heading[:, int(P / 2), 1] - aug_current_state[:, 1],
            ego_future_heading[:, int(P / 2), 0] - aug_current_state[:, 0],
        )
        a0 = aug_current_state[:, 6] * torch.cos(theta0) + aug_current_state[:, 7] * torch.sin(
            theta0
        )
        x0, y0, v0, omega0 = (
            aug_current_state[:, 0],
            aug_current_state[:, 1],
            torch.norm(aug_current_state[:, 4:6], dim=-1),
            aug_current_state[:, 9],
        )

        xT = ego_future_heading[:, P, 0]
        yT = ego_future_heading[:, P, 1]
        thetaT = ego_future_heading[:, P, 2]
        terminal_velocity = (
            3 * ego_future_heading[:, P, :2]
            - 4 * ego_future_heading[:, P - 1, :2]
            + ego_future_heading[:, P - 2, :2]
        ) / (2 * dt)
        terminal_acceleration = (
            2 * ego_future_heading[:, P, :2]
            - 5 * ego_future_heading[:, P - 1, :2]
            + 4 * ego_future_heading[:, P - 2, :2]
            - ego_future_heading[:, P - 3, :2]
        ) / dt**2
        vT = torch.norm(terminal_velocity, dim=-1)
        aT = terminal_acceleration[:, 0] * torch.cos(thetaT) + terminal_acceleration[
            :, 1
        ] * torch.sin(thetaT)
        last_heading_delta = self.normalize_angle(
            ego_future_heading[:, P, 2] - ego_future_heading[:, P - 1, 2]
        )
        previous_heading_delta = self.normalize_angle(
            ego_future_heading[:, P - 1, 2] - ego_future_heading[:, P - 2, 2]
        )
        omegaT = (3 * last_heading_delta - previous_heading_delta) / (2 * dt)

        # Boundary conditions
        sx = torch.stack(
            [
                x0,
                v0 * torch.cos(theta0),
                a0 * torch.cos(theta0) - v0 * torch.sin(theta0) * omega0,
                xT,
                vT * torch.cos(thetaT),
                aT * torch.cos(thetaT) - vT * torch.sin(thetaT) * omegaT,
            ],
            dim=-1,
        )

        sy = torch.stack(
            [
                y0,
                v0 * torch.sin(theta0),
                a0 * torch.sin(theta0) + v0 * torch.cos(theta0) * omega0,
                yT,
                vT * torch.sin(thetaT),
                aT * torch.sin(thetaT) + vT * torch.cos(thetaT) * omegaT,
            ],
            dim=-1,
        )

        ax = A @ sx[:, :, None]  # B, 6, 1
        ay = A @ sy[:, :, None]  # B, 6, 1

        traj_x = M_t @ ax
        traj_y = M_t @ ay
        traj_heading = torch.cat(
            [
                torch.atan2(
                    traj_y[:, :1, 0] - y0.unsqueeze(-1), traj_x[:, :1, 0] - x0.unsqueeze(-1)
                ),
                torch.atan2(
                    traj_y[:, 1:, 0] - traj_y[:, :-1, 0], traj_x[:, 1:, 0] - traj_x[:, :-1, 0]
                ),
            ],
            dim=1,
        )

        interpolated = torch.cat([traj_x, traj_y, traj_heading[..., None]], axis=-1)

        if keep_remaining and ego_future_heading.shape[1] > P:
            interpolated = torch.cat([interpolated, ego_future_heading[:, P:, :]], dim=1)

        if future_is_cos_sin:
            return torch.cat(
                [
                    interpolated[..., :2],
                    torch.cos(interpolated[..., 2:3]),
                    torch.sin(interpolated[..., 2:3]),
                ],
                dim=-1,
            )
        return interpolated
