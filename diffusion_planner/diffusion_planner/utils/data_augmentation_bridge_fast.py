"""Fast, fully-batched variant of the bridge state-perturbation augmentation.

The original bridge augmentation (``data_augmentation_bridge.py``) builds, per sample,
a laterally offset path that reconnects to the GT trajectory and searches over
(past, future) bridge-duration candidates until the lateral-acceleration / speed-gap /
longitudinal-jerk limits are satisfied. The search evaluates hundreds of candidates
per sample in Python, which makes it too slow to enable in training.

This variant produces the same kind of sample (lateral + heading offset at t=0,
C2-smooth reconnection to the GT past and future, kinematically consistent current
state) but with no per-sample search:

1. The lateral offset profile d(t) is a quintic in time with closed-form
   coefficients (d(0)=offset, d'(0)=v*tan(heading_offset), d''(0)=0, and
   d=d'=d''=0 at the bridge end).
2. The bridge durations are computed in closed form from the quintic's known
   extremal constants so that the offset-induced lateral acceleration and jerk
   stay within budget, then clamped to the available horizon.
3. The built paths are verified against the same limits in one batched pass;
   because d is linear in (offset, heading offset), violations are removed by a
   single linear rescale of the offsets. Samples that still violate the limits
   (e.g. GT itself exceeds them) are left un-augmented.

Everything operates on (B, T) tensors; there are no Python loops over the batch.
"""

from __future__ import annotations

import torch

from diffusion_planner.utils.data_augmentation_bridge import (
    StatePerturbation as BridgeStatePerturbation,
)

# Extremal constants of the quintic profile q(u) on u in [0, 1] with
# q(0)=1, q'(0)=0, q''(0)=0, q(1)=q'(1)=q''(1)=0 (pure offset term):
#   max |q''|  = 10 / sqrt(3) / T^2
#   max |q'''| = 60 / T^3
_QUINTIC_ACCEL_CONST = 10.0 / (3.0**0.5)
_QUINTIC_JERK_CONST = 60.0
_MIN_BRIDGE_SEC = 0.5
_MIN_SPEED_FOR_HEADING = 0.5


def _quintic_coeffs(offset, slope, duration):
    """Closed-form quintic coefficients for d(0)=offset, d'(0)=slope, d''(0)=0
    and d(T)=d'(T)=d''(T)=0. All args (B,), returns tuple of (B,) tensors."""
    T = duration
    a0 = offset
    a1 = slope
    a3 = -(10.0 * offset / T**3 + 6.0 * slope / T**2)
    a4 = 15.0 * offset / T**4 + 8.0 * slope / T**3
    a5 = -(6.0 * offset / T**5 + 3.0 * slope / T**4)
    return a0, a1, a3, a4, a5


def _quintic_eval(t, offset, slope, duration):
    """Evaluate d(t) and d'(t) on a (B, N) grid; zero outside [0, duration]."""
    a0, a1, a3, a4, a5 = (c.unsqueeze(-1) for c in _quintic_coeffs(offset, slope, duration))
    T = duration.unsqueeze(-1)
    tc = torch.clamp(t, min=torch.zeros_like(T), max=T)
    d = a0 + a1 * tc + a3 * tc**3 + a4 * tc**4 + a5 * tc**5
    dd = a1 + 3.0 * a3 * tc**2 + 4.0 * a4 * tc**3 + 5.0 * a5 * tc**4
    inside = t <= T
    return d * inside, dd * inside


def _arc_speed(xy, dt):
    """(B, N, 2) -> (B, N) arc speed via central differences."""
    grad = torch.gradient(xy, dim=1)[0] / dt
    return torch.linalg.norm(grad, dim=-1)


def _path_heading(xy, fallback_heading, dt):
    """Heading from position differences; falls back where the path is (nearly) stopped."""
    grad = torch.gradient(xy, dim=1)[0] / dt
    speed = torch.linalg.norm(grad, dim=-1)
    heading = torch.atan2(grad[..., 1], grad[..., 0])
    return torch.where(speed > _MIN_SPEED_FOR_HEADING, heading, fallback_heading)


def _normalize_angle(angle):
    return (angle + torch.pi) % (2.0 * torch.pi) - torch.pi


def _lateral_accel(xy, heading, dt):
    """|curvature| * arc_speed^2 on the path, batched."""
    speed = _arc_speed(xy, dt)
    dheading = torch.gradient(heading, dim=1)[0]
    dheading = _normalize_angle(dheading)
    ds = torch.clamp(speed * dt, min=1.0e-3)
    curvature = dheading / ds
    return torch.abs(curvature) * speed**2


def _longitudinal_jerk(xy, dt):
    speed = _arc_speed(xy, dt)
    accel = torch.gradient(speed, dim=1)[0] / dt
    return torch.gradient(accel, dim=1)[0] / dt


class StatePerturbation(BridgeStatePerturbation):
    """Batched, search-free drop-in replacement for the bridge augmentation."""

    def augment(self, inputs, ego_future):
        ego_current_state = inputs["ego_current_state"]
        ego_past = inputs["ego_agent_past"]  # (B, P+1, 4) x, y, cos, sin; last = current
        device = ego_current_state.device
        dtype = ego_current_state.dtype
        B, F = ego_future.shape[0], ego_future.shape[1]
        P = ego_past.shape[1] - 1
        dt = self.time_interval

        lateral_offsets = self._sample_lateral_offsets(B).to(device=device, dtype=dtype)
        heading_offsets = self._sample_heading_offsets(B).to(device=device, dtype=dtype)
        valid_speed = torch.abs(ego_current_state[:, 4]) >= 2.0
        valid_offset = (torch.abs(lateral_offsets) > 1.0e-3) | (torch.abs(heading_offsets) > 1.0e-3)
        aug_flag = (torch.rand(B, device=device) < self._augment_prob) & valid_speed & valid_offset
        if not torch.any(aug_flag):
            return aug_flag, ego_current_state.clone(), ego_past.clone(), ego_future.clone()

        speed = torch.clamp(torch.abs(ego_current_state[:, 4]), min=_MIN_SPEED_FOR_HEADING)
        slope = speed * torch.tan(heading_offsets)  # d'(0): lateral velocity at t=0

        # GT timeline: past (t = -P*dt .. 0) + future (t = dt .. F*dt), all in one tensor.
        gt_xy = torch.cat([ego_past[:, :, :2], ego_future[:, :, :2]], dim=1)  # (B, P+1+F, 2)
        gt_heading = torch.cat(
            [torch.atan2(ego_past[:, :, 3], ego_past[:, :, 2]), ego_future[:, :, 2]], dim=1
        )
        t_grid = (torch.arange(P + 1 + F, device=device, dtype=dtype) - P) * dt  # (N,)
        t_grid = t_grid.unsqueeze(0).expand(B, -1)

        # Budget for the offset-induced lateral acceleration: leave room for the GT's own.
        gt_lat_accel = _lateral_accel(gt_xy, _path_heading(gt_xy, gt_heading, dt), dt)
        gt_lat_max = gt_lat_accel.max(dim=1).values
        accel_budget = torch.clamp(self._max_lateral_accel_mps2 - gt_lat_max, min=0.5)

        offsets, slopes = lateral_offsets, slope
        for _ in range(2):  # second pass only rescales; d is linear in (offsets, slopes)
            abs_off = torch.abs(offsets) + 1.0e-6
            # Closed-form bridge durations from the quintic extremal constants.
            t_accel = torch.sqrt(_QUINTIC_ACCEL_CONST * abs_off / accel_budget)
            t_jerk = (_QUINTIC_JERK_CONST * abs_off / self._max_bridge_jerk_mps3) ** (1.0 / 3.0)
            t_slope = 3.94 * torch.abs(slopes) / accel_budget
            duration = torch.clamp(
                torch.maximum(torch.maximum(t_accel, t_jerk), t_slope), min=_MIN_BRIDGE_SEC
            )
            t_future = torch.clamp(
                torch.maximum(duration, torch.full_like(duration, self._future_bridge_sec)),
                max=F * dt,
            )
            t_past = torch.clamp(
                torch.maximum(duration, torch.full_like(duration, self._past_bridge_sec)),
                max=P * dt,
            )

            # Lateral offset profile over the full timeline (future side / mirrored past side).
            d_f, dd_f = _quintic_eval(t_grid, offsets, slopes, t_future)
            d_p, dd_p = _quintic_eval(-t_grid, offsets, -slopes, t_past)
            is_future = t_grid > 0.0
            d = torch.where(is_future, d_f, d_p)
            dd = torch.where(is_future, dd_f, -dd_p)

            normal = torch.stack([-torch.sin(gt_heading), torch.cos(gt_heading)], dim=-1)
            aug_xy = gt_xy + d.unsqueeze(-1) * normal

            # Batched verification with the same limits as the original implementation.
            aug_heading = _path_heading(aug_xy, gt_heading, dt)
            bridge_mask = (t_grid >= -t_past.unsqueeze(-1)) & (t_grid <= t_future.unsqueeze(-1))
            lat_accel = (
                torch.where(
                    bridge_mask,
                    _lateral_accel(aug_xy, aug_heading, dt),
                    torch.zeros_like(t_grid),
                )
                .max(dim=1)
                .values
            )
            speed_gap = (
                torch.where(
                    bridge_mask,
                    torch.abs(_arc_speed(aug_xy, dt) - _arc_speed(gt_xy, dt)),
                    torch.zeros_like(t_grid),
                )
                .max(dim=1)
                .values
            )
            jerk = (
                torch.where(
                    bridge_mask,
                    torch.abs(_longitudinal_jerk(aug_xy, dt)),
                    torch.zeros_like(t_grid),
                )
                .max(dim=1)
                .values
            )

            ratio = torch.maximum(
                lat_accel / self._max_lateral_accel_mps2,
                torch.maximum(
                    speed_gap / self._max_bridge_speed_gap_mps, jerk / self._max_bridge_jerk_mps3
                ),
            )
            feasible = ratio <= 1.0 + 1.0e-6
            if bool(torch.all(feasible | ~aug_flag)):
                break
            scale = torch.where(feasible, torch.ones_like(ratio), 0.9 / torch.clamp(ratio, min=1.0))
            offsets = offsets * scale
            slopes = slopes * scale
        aug_flag = aug_flag & feasible

        # Assemble outputs (all batched).
        aug_past = ego_past.clone()
        aug_future = ego_future.clone()
        past_xy = aug_xy[:, : P + 1]
        future_xy = aug_xy[:, P + 1 :]
        heading_corr = torch.atan2(
            dd, torch.clamp(_arc_speed(gt_xy, dt), min=_MIN_SPEED_FOR_HEADING)
        )
        new_heading = _normalize_angle(gt_heading + heading_corr)
        aug_past[:, :, 0:2] = past_xy
        aug_past[:, :, 2] = torch.cos(new_heading[:, : P + 1])
        aug_past[:, :, 3] = torch.sin(new_heading[:, : P + 1])
        aug_future[:, :, 0:2] = future_xy
        aug_future[:, :, 2] = new_heading[:, P + 1 :]

        # Current state from central differences across t=0 (indices P-1, P, P+1).
        aug_state = ego_current_state.clone()
        vel_map = (aug_xy[:, P + 1] - aug_xy[:, P - 1]) / (2.0 * dt)
        acc_map = (aug_xy[:, P + 1] - 2.0 * aug_xy[:, P] + aug_xy[:, P - 1]) / dt**2
        yaw = new_heading[:, P]
        cos_h, sin_h = torch.cos(yaw), torch.sin(yaw)
        # Rotate map-frame velocity/acceleration into the (new) body frame.
        vel_body = torch.stack(
            [
                cos_h * vel_map[:, 0] + sin_h * vel_map[:, 1],
                -sin_h * vel_map[:, 0] + cos_h * vel_map[:, 1],
            ],
            dim=-1,
        )
        acc_body = torch.stack(
            [
                cos_h * acc_map[:, 0] + sin_h * acc_map[:, 1],
                -sin_h * acc_map[:, 0] + cos_h * acc_map[:, 1],
            ],
            dim=-1,
        )
        yaw_rate = _normalize_angle(new_heading[:, P + 1] - new_heading[:, P - 1]) / (2.0 * dt)
        body_speed = torch.linalg.norm(vel_body, dim=-1)
        steering = torch.where(
            body_speed >= 0.2,
            torch.clamp(
                torch.atan(yaw_rate * self._wheel_base / torch.clamp(body_speed, min=0.2)),
                -2.0 / 3.0 * torch.pi,
                2.0 / 3.0 * torch.pi,
            ),
            torch.zeros_like(yaw_rate),
        )
        yaw_rate = torch.where(body_speed >= 0.2, yaw_rate, torch.zeros_like(yaw_rate))

        aug_state[:, 0:2] = aug_xy[:, P]
        aug_state[:, 2] = cos_h
        aug_state[:, 3] = sin_h
        aug_state[:, 4:6] = vel_body
        aug_state[:, 6:8] = acc_body
        aug_state[:, 8] = steering
        aug_state[:, 9] = yaw_rate

        flag = aug_flag.unsqueeze(-1)
        out_state = torch.where(flag, aug_state, ego_current_state)
        out_past = torch.where(flag.unsqueeze(-1), aug_past, ego_past)
        out_future = torch.where(flag.unsqueeze(-1), aug_future, ego_future)
        return aug_flag, out_state, out_past, out_future
