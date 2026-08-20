"""Batched GPU PDM oracle: DrivoR scorer labels for Diffusion-Planner tensors.

DrivoR trains its six scorer heads against PDM sub-scores computed *online* for
the model's own proposals.  In the NAVSIM code base those labels come from
``PDMScorer`` walking shapely polygons on the CPU, one proposal at a time; the
e2e-devkit's GPU scorer (``evaluation/score_navsim_batch``) needs ``T4Scene`` /
``Trajectory`` objects that a Diffusion-Planner NPZ simply does not carry.

So this module is the adapter: it consumes the Diffusion-Planner observation
tensors (ego-centric, zero-padded, fixed-shape) together with ``[B, N, T, 4]``
proposals and returns the six sub-scores plus their PDMS aggregate as one
``[B, N, 7]`` tensor, fully batched over (scene x proposal x timestep) so the
labels can be produced inside the training step on the GPU.

Every threshold, weight and decision rule is taken from
``planner_metrics.pdms_navsim`` -- the repository's NAVSIM PDM port, which is the
same reference the validation panel reports against.  ``test_drivor_oracle.py``
checks this implementation against that (scipy/shapely) reference numerically, so
"the oracle agrees with the reported metric" is a test, not a claim.

Comfort is the one sub-score that cannot be read off the poses: NAVSIM scores the
states an LQR tracker + kinematic bicycle model produce while *following* the
proposal, never the waypoints themselves.  That rollout is ported exactly in
:mod:`planner_metrics.pdm_simulator_torch` and :meth:`DrivoROracle._comfort` runs
it, so the label measures ride quality rather than trajectory-head noise.

Deviations from the CPU reference, all deliberate:

* The horizon is sub-sampled with a per-metric stride (``nc_stride`` and friends)
  instead of evaluating all 80 steps.  A collision missed by a 0.2 s grid needs a
  relative speed above ~10 m/s *and* a sub-0.2 s overlap window.
* Candidate neighbours / border segments are reduced to a fixed top-K by a
  conservative bounding-circle prefilter, keeping every tensor shape static
  (no ``.item()``, no host sync, ``torch.compile``-friendly).
* Static objects are ignored: the ``static_objects`` tensor is all-zero
  throughout this dataset (verified over a random sample of the valid split), so
  NC's 0.5 "static object" branch can never fire and building the boxes would be
  pure overhead.
* ``no_at_fault_collision``'s cross-frame track de-duplication and the
  map-dependent lateral-at-fault branch are not reproduced -- the same two
  deviations the CPU port documents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from planner_metrics.pdms_navsim import (
    COLLISION_STOPPED_SPEED_THRESHOLD,
    DRIVING_DIRECTION_COMPLIANCE_THRESHOLD,
    DRIVING_DIRECTION_HORIZON,
    DRIVING_DIRECTION_VIOLATION_THRESHOLD,
    FUTURE_COLLISION_HORIZON_WINDOW,
    MAX_ABS_LAT_ACCEL,
    MAX_ABS_LON_JERK,
    MAX_ABS_MAG_JERK,
    MAX_ABS_YAW_ACCEL,
    MAX_ABS_YAW_RATE,
    MAX_LON_ACCEL,
    MIN_LON_ACCEL,
    PROGRESS_DISTANCE_THRESHOLD,
    STOPPED_SPEED_THRESHOLD,
    STATE_ACC_X,
    STATE_ACC_Y,
    STATE_HEADING,
    STATE_SIZE,
    STATE_VEL_X,
    STATE_VEL_Y,
    STATE_X,
    STATE_Y,
)
from planner_metrics.pdm_simulator_torch import initial_states_from_ego, simulate_proposals

# The oracle's output channels, in order.  The first six are exactly DrivoR's
# scorer heads (``DRIVOR_HEAD_METRICS``); the seventh is their PDMS aggregate,
# DrivoR's ``score`` pseudo-metric.
ORACLE_METRIC_NAMES: tuple[str, ...] = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "history_comfort",
    "score",
)

#: TTC label used where the metric is undefined (ego never moves over the whole
#: horizon, so no step is evaluable).  DrivoR's loss masks exactly this value.
TTC_UNDEFINED = 2.0

_COS_AHEAD = float(np.cos(np.deg2rad(30.0)))
_COS_BEHIND = float(np.cos(np.deg2rad(150.0)))
_EPS = 1e-9


# ---------------------------------------------------------------------------
# Savitzky-Golay as a dense linear operator
# ---------------------------------------------------------------------------
# ``scipy.signal.savgol_filter(..., mode="interp")`` is linear in its input, so
# for a fixed length it *is* a ``[T, T]`` matrix: ``filtered = signal @ M``.
# Building M once by pushing the identity through scipy makes the GPU path
# bit-comparable with the reference (including scipy's polynomial edge fit, which
# a plain convolution gets wrong) and turns the whole comfort computation into a
# handful of matmuls.
_SAVGOL_CACHE: dict[tuple, np.ndarray] = {}


def _savgol_operator(length: int, window: int, poly: int, deriv: int, delta: float) -> np.ndarray:
    key = (length, window, poly, deriv, float(delta))
    cached = _SAVGOL_CACHE.get(key)
    if cached is None:
        from scipy.signal import savgol_filter

        eye = np.eye(length, dtype=np.float64)
        cached = savgol_filter(
            eye,
            window_length=min(window, length),
            polyorder=poly,
            deriv=deriv,
            delta=delta,
            axis=-1,
        )
        _SAVGOL_CACHE[key] = cached
    return cached


def _gradient(values: torch.Tensor, delta: float) -> torch.Tensor:
    """``numpy.gradient`` along the last dim: central inside, one-sided at the ends."""
    out = torch.empty_like(values)
    out[..., 1:-1] = (values[..., 2:] - values[..., :-2]) / (2.0 * delta)
    out[..., 0] = (values[..., 1] - values[..., 0]) / delta
    out[..., -1] = (values[..., -1] - values[..., -2]) / delta
    return out


def _states_from_poses(poses: torch.Tensor, dt: float) -> torch.Tensor:
    """``pdms_navsim.states_from_poses``: kinematics by finite differences.

    Only the comfort *prefix* uses this -- the recorded past, which no tracker
    can be run on.  The future goes through the rollout instead
    (:mod:`planner_metrics.pdm_simulator_torch`).  ``poses`` is ``[B, H, 4]`` =
    (x, y, cos, sin); returns ``[B, H, STATE_SIZE]``.
    """
    x, y = poses[..., 0], poses[..., 1]
    heading = torch.atan2(poses[..., 3], poses[..., 2])
    vx = _gradient(x, dt)
    vy = _gradient(y, dt)
    ax = _gradient(vx, dt)
    ay = _gradient(vy, dt)
    cos_h, sin_h = heading.cos(), heading.sin()

    states = poses.new_zeros(poses.shape[:-1] + (STATE_SIZE,))
    states[..., STATE_X] = x
    states[..., STATE_Y] = y
    states[..., STATE_HEADING] = heading
    states[..., STATE_VEL_X] = vx * cos_h + vy * sin_h
    states[..., STATE_VEL_Y] = -vx * sin_h + vy * cos_h
    states[..., STATE_ACC_X] = ax * cos_h + ay * sin_h
    states[..., STATE_ACC_Y] = -ax * sin_h + ay * cos_h
    return states


def _phase_unwrap(headings: torch.Tensor) -> torch.Tensor:
    two_pi = 2.0 * float(np.pi)
    adjustments = torch.zeros_like(headings)
    adjustments[..., 1:] = torch.cumsum(
        torch.round(torch.diff(headings, dim=-1) / two_pi), dim=-1
    )
    return headings - two_pi * adjustments


def _heading_unit(poses: torch.Tensor) -> torch.Tensor:
    """(..., 4) poses -> unit heading vector, robust to un-normalized cos/sin."""
    heading = poses[..., 2:4]
    return heading / heading.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def _perp(vec: torch.Tensor) -> torch.Tensor:
    """Rotate a 2-vector by +90 deg (forward -> left)."""
    return torch.stack((-vec[..., 1], vec[..., 0]), dim=-1)


def _obb_overlap(
    delta: torch.Tensor,
    ego_axis: torch.Tensor,
    ego_half: torch.Tensor,
    other_axis: torch.Tensor,
    other_half: torch.Tensor,
) -> torch.Tensor:
    """Exact rectangle-rectangle overlap by the 2D separating-axis theorem.

    Four candidate axes (two per box) are necessary and sufficient for convex
    quads in the plane, so this matches the reference's SAT + shapely
    ``intersects`` pair, touching included.

    Args:
        delta: ``(..., 2)`` other centre minus ego centre.
        ego_axis / other_axis: ``(..., 2)`` unit forward vectors.
        ego_half / other_half: ``(..., 2)`` half (length, width).
    """
    ego_x, ego_y = ego_axis, _perp(ego_axis)
    other_x, other_y = other_axis, _perp(other_axis)

    a11 = (other_x * ego_x).sum(-1).abs()
    a12 = (other_y * ego_x).sum(-1).abs()
    a21 = (other_x * ego_y).sum(-1).abs()
    a22 = (other_y * ego_y).sum(-1).abs()

    ehl, ehw = ego_half[..., 0], ego_half[..., 1]
    ohl, ohw = other_half[..., 0], other_half[..., 1]

    sep = (delta * ego_x).sum(-1).abs() > ehl + ohl * a11 + ohw * a12
    sep |= (delta * ego_y).sum(-1).abs() > ehw + ohl * a21 + ohw * a22
    sep |= (delta * other_x).sum(-1).abs() > ohl + ehl * a11 + ehw * a21
    sep |= (delta * other_y).sum(-1).abs() > ohw + ehl * a12 + ehw * a22
    return ~sep


def _segment_hits_box(
    p0: torch.Tensor,
    p1: torch.Tensor,
    centre: torch.Tensor,
    axis: torch.Tensor,
    half: torch.Tensor,
) -> torch.Tensor:
    """Liang-Barsky: does segment ``p0->p1`` touch the oriented box?

    The segment is taken into the box's own frame, where the box is the
    axis-aligned ``[-hl, hl] x [-hw, hw]``, so the clip is two 1D interval
    intersections.  Broadcasting is the caller's job.
    """
    axis_y = _perp(axis)
    rel0 = p0 - centre
    rel1 = p1 - centre
    local0 = torch.stack(((rel0 * axis).sum(-1), (rel0 * axis_y).sum(-1)), dim=-1)
    local1 = torch.stack(((rel1 * axis).sum(-1), (rel1 * axis_y).sum(-1)), dim=-1)

    direction = local1 - local0
    t_enter = torch.zeros_like(local0[..., 0])
    t_exit = torch.ones_like(local0[..., 0])
    inside = torch.ones_like(t_enter, dtype=torch.bool)

    for dim in range(2):
        d = direction[..., dim]
        lo = -half[..., dim] - local0[..., dim]
        hi = half[..., dim] - local0[..., dim]
        parallel = d.abs() < _EPS
        # Parallel to this slab: either wholly inside it or no intersection.
        inside &= ~parallel | ((lo <= 0.0) & (hi >= 0.0))
        safe_d = torch.where(parallel, torch.ones_like(d), d)
        t_lo = lo / safe_d
        t_hi = hi / safe_d
        near = torch.minimum(t_lo, t_hi)
        far = torch.maximum(t_lo, t_hi)
        t_enter = torch.where(parallel, t_enter, torch.maximum(t_enter, near))
        t_exit = torch.where(parallel, t_exit, torch.minimum(t_exit, far))

    return inside & (t_enter <= t_exit)


def _point_segment_distance(
    point: torch.Tensor, seg0: torch.Tensor, seg1: torch.Tensor
) -> torch.Tensor:
    """Euclidean point-to-segment distance; broadcasting is the caller's job."""
    seg = seg1 - seg0
    length_sq = (seg * seg).sum(-1).clamp_min(_EPS)
    t = (((point - seg0) * seg).sum(-1) / length_sq).clamp(0.0, 1.0)
    closest = seg0 + t.unsqueeze(-1) * seg
    return (point - closest).norm(dim=-1)


def _project_arclength(
    point: torch.Tensor, seg0: torch.Tensor, seg1: torch.Tensor, seg_start_arc: torch.Tensor
) -> torch.Tensor:
    """``shapely.LineString.project``: arclength of the nearest point on a polyline.

    Args:
        point: ``[..., 2]``.
        seg0 / seg1: ``[..., S, 2]`` segment endpoints (broadcast against ``point``).
        seg_start_arc: ``[..., S]`` cumulative arclength at ``seg0``.
    """
    seg = seg1 - seg0
    length = seg.norm(dim=-1)
    length_sq = (length * length).clamp_min(_EPS)
    rel = point.unsqueeze(-2) - seg0
    t = ((rel * seg).sum(-1) / length_sq).clamp(0.0, 1.0)
    closest = seg0 + t.unsqueeze(-1) * seg
    distance = (point.unsqueeze(-2) - closest).norm(dim=-1)
    best = distance.argmin(dim=-1, keepdim=True)
    arc = seg_start_arc + t * length
    return arc.gather(-1, best).squeeze(-1)


# ---------------------------------------------------------------------------
# Scene-side pre-processing (proposal independent, done once per batch)
# ---------------------------------------------------------------------------
@dataclass
class OracleScene:
    """Everything the oracle needs from a batch that does not depend on proposals."""

    ego_half: torch.Tensor  # [B, 2] half (length, width)
    centre_offset: torch.Tensor  # [B] rear-axle -> footprint-centre shift
    neighbour_xy: torch.Tensor  # [B, K, T, 2]
    neighbour_axis: torch.Tensor  # [B, K, T, 2]
    neighbour_half: torch.Tensor  # [B, K, 2]
    neighbour_valid: torch.Tensor  # [B, K, T] bool
    neighbour_stopped: torch.Tensor  # [B, K, T] bool
    border_p0: torch.Tensor  # [B, M, 2]
    border_p1: torch.Tensor  # [B, M, 2]
    border_valid: torch.Tensor  # [B, M] bool
    route_centre0: torch.Tensor  # [B, R, 2]
    route_centre1: torch.Tensor  # [B, R, 2]
    route_left: torch.Tensor  # [B, R] signed left offset
    route_right: torch.Tensor  # [B, R] signed right offset
    route_valid: torch.Tensor  # [B, R] bool
    reference_p0: torch.Tensor  # [B, S, 2] expert polyline segments
    reference_p1: torch.Tensor  # [B, S, 2]
    reference_arc: torch.Tensor  # [B, S] cumulative arclength at p0
    reference_progress: torch.Tensor  # [B] expert's own arclength extent
    ego_initial: torch.Tensor  # [B, STATE_SIZE] fp64 rollout start state
    wheel_base: torch.Tensor  # [B] fp64, the motion model's wheel base
    history_states: torch.Tensor  # [B, H - 1, STATE_SIZE] fp64 comfort prefix


class DrivoROracle:
    """Turns ``[B, N, T, 4]`` proposals into DrivoR's ``[B, N, 7]`` scorer labels."""

    def __init__(
        self,
        *,
        dt: float = 0.1,
        collision_stride: int = 2,
        ttc_stride: int = 2,
        border_stride: int = 2,
        route_stride: int = 4,
        max_neighbours: int = 32,
        max_border_segments: int = 96,
        max_route_segments: int = 128,
        proposal_chunk: int = 0,
        score_weights: tuple[float, ...] = (1.0, 1.0, 0.0, 5.0, 5.0, 2.0),
    ) -> None:
        self.dt = float(dt)
        self.collision_stride = int(collision_stride)
        self.ttc_stride = int(ttc_stride)
        self.border_stride = int(border_stride)
        self.route_stride = int(route_stride)
        self.max_neighbours = int(max_neighbours)
        self.max_border_segments = int(max_border_segments)
        self.max_route_segments = int(max_route_segments)
        self.proposal_chunk = int(proposal_chunk)
        self.score_weights = tuple(float(w) for w in score_weights)
        self._savgol: dict[tuple, torch.Tensor] = {}

    # -- public API --------------------------------------------------------
    @torch.no_grad()
    def __call__(
        self,
        proposals: torch.Tensor,
        inputs: dict,
        ego_future: torch.Tensor,
        scene: Optional[OracleScene] = None,
    ) -> torch.Tensor:
        """Score proposals.

        Args:
            proposals: ``[B, N, T, 4]`` metric (x, y, cos, sin), ego frame.
            inputs: the **un-normalized** observation dict (the oracle needs
                metres; ``ObservationNormalizer`` returns a shallow copy, so the
                caller's original dict is the right thing to pass).
            ego_future: ``[B, T, 4]`` expert future, the EP reference path.
            scene: optional pre-built :class:`OracleScene` (see :meth:`prepare`).

        Returns:
            ``[B, N, 7]`` float32 labels in :data:`ORACLE_METRIC_NAMES` order.
        """
        proposals = proposals.detach().float()
        if scene is None:
            scene = self.prepare(inputs, ego_future, proposals)

        batch, num_proposals = proposals.shape[:2]
        chunk = self.proposal_chunk if self.proposal_chunk > 0 else num_proposals
        parts = [
            self._score_chunk(proposals[:, start : start + chunk], scene)
            for start in range(0, num_proposals, chunk)
        ]
        components = torch.cat(parts, dim=1)  # [B, N, 6], EP still a placeholder

        # EP needs the multiplicative terms as its gate, so it is the one metric
        # left for after the chunks are re-joined.
        nc, dac = components[..., 0], components[..., 1]
        components[..., 4] = self._progress(proposals, scene, nc * dac)

        score = self._aggregate(components)
        out = torch.cat((components, score[..., None]), dim=-1)
        assert out.shape == (batch, num_proposals, len(ORACLE_METRIC_NAMES))
        return out

    def _aggregate(self, components: torch.Tensor) -> torch.Tensor:
        """PDMS over the oracle labels: NC * DAC * weighted mean of the rest."""
        weights = torch.tensor(
            self.score_weights, device=components.device, dtype=components.dtype
        )
        behaviour = components[..., 2:].clone()
        # TTC's undefined sentinel must not leak into the aggregate: treat it as
        # "no infraction" for the score while the loss masks it for the head.
        ttc = components[..., 3]
        behaviour[..., 1] = torch.where(ttc == TTC_UNDEFINED, torch.ones_like(ttc), ttc)
        denominator = weights[2:].sum().clamp_min(_EPS)
        behaviour_score = (behaviour * weights[2:]).sum(-1) / denominator
        return components[..., 0] * components[..., 1] * behaviour_score

    @torch.no_grad()
    def prepare(
        self,
        inputs: dict,
        ego_future: torch.Tensor,
        proposals: Optional[torch.Tensor] = None,
    ) -> OracleScene:
        """Pre-process the proposal-independent part of a batch.

        ``proposals`` is only read to size the candidate prefilters: the guard
        ball around the proposal set at each step is what makes the fixed top-K
        provably conservative, so passing them is strongly preferred.
        """
        ego_shape = inputs["ego_shape"].float()
        device = ego_shape.device
        batch = ego_shape.shape[0]

        ego_half = torch.stack((ego_shape[:, 1] * 0.5, ego_shape[:, 2] * 0.5), dim=-1)
        centre_offset = ego_shape[:, 0] * 0.5

        # --- neighbours ---------------------------------------------------
        past = inputs["neighbor_agents_past"].float()
        future = inputs["neighbor_agents_future"].float()
        horizon = future.shape[2]

        track_valid = past[:, :, -1, :].abs().sum(-1) > 0  # [B, A]
        step_valid = future.abs().sum(-1) > 0  # [B, A, T]
        valid = step_valid & track_valid[:, :, None]

        nb_xy = future[..., :2]
        nb_heading = future[..., 2]
        nb_axis = torch.stack((torch.cos(nb_heading), torch.sin(nb_heading)), dim=-1)
        # (width, length) in the shard -> (half length, half width) here.
        nb_half = torch.stack((past[:, :, -1, 7] * 0.5, past[:, :, -1, 6] * 0.5), dim=-1)

        nb_speed = self._neighbour_speed(nb_xy, valid, past)
        nb_stopped = nb_speed <= COLLISION_STOPPED_SPEED_THRESHOLD

        # Conservative prefilter.  Every scored ego reference point at step ``t``
        # lies inside the ball ``(guard_centre[t], guard_radius[t])`` that covers
        # the whole proposal set (and the expert path), and an ego footprint
        # reaches at most ``ego_radius`` beyond its reference point -- so a
        # neighbour whose distance to that ball stays above both radii cannot
        # touch any scored trajectory.  Rank by that distance and keep a fixed
        # top-K, which keeps every downstream shape static and the whole oracle
        # free of host syncs.
        ego_radius = ego_half.norm(dim=-1)  # [B]
        nb_radius = nb_half.norm(dim=-1)  # [B, A]
        ref_xy = ego_future[..., :2].float()  # [B, T, 2]
        guard_centre, guard_radius = self._guard_ball(ref_xy, proposals)
        gap = (nb_xy - guard_centre[:, None]).norm(dim=-1)  # [B, A, T]
        gap = gap - guard_radius[:, None, :] - ego_radius[:, None, None] - nb_radius[:, :, None]
        gap = torch.where(valid, gap, torch.full_like(gap, 1e6))
        rank = gap.min(dim=-1).values  # [B, A]
        keep = min(self.max_neighbours, rank.shape[1])
        order = rank.topk(keep, dim=1, largest=False).indices  # [B, K]

        def take(tensor: torch.Tensor) -> torch.Tensor:
            index = order.reshape(batch, keep, *([1] * (tensor.dim() - 2)))
            return tensor.gather(1, index.expand(batch, keep, *tensor.shape[2:]))

        # --- road borders -------------------------------------------------
        line_strings = inputs["line_strings"].float()
        border_p0, border_p1, border_valid = self._border_segments(line_strings)
        border_p0, border_p1, border_valid = self._reduce_segments(
            border_p0,
            border_p1,
            border_valid,
            guard_centre,
            guard_radius,
            self.max_border_segments,
        )

        # --- route ribbons ------------------------------------------------
        route = inputs["route_lanes"].float()
        (
            route_centre0,
            route_centre1,
            route_left,
            route_right,
            route_valid,
        ) = self._route_segments(route)
        keep_route = min(self.max_route_segments, route_centre0.shape[1])
        route_rank = (
            _point_segment_distance(
                guard_centre[:, :, None, :], route_centre0[:, None], route_centre1[:, None]
            )
            - guard_radius[:, :, None]
        ).min(dim=1).values  # [B, R]
        route_rank = torch.where(route_valid, route_rank, torch.full_like(route_rank, 1e6))
        route_order = route_rank.topk(keep_route, dim=1, largest=False).indices

        def take_route(tensor: torch.Tensor) -> torch.Tensor:
            index = route_order.reshape(batch, keep_route, *([1] * (tensor.dim() - 2)))
            return tensor.gather(1, index.expand(batch, keep_route, *tensor.shape[2:]))

        # --- expert reference polyline (EP) -------------------------------
        origin = torch.zeros(batch, 1, 2, device=device, dtype=torch.float32)
        ref_path = torch.cat((origin, ref_xy), dim=1)  # [B, T+1, 2]
        ref_p0 = ref_path[:, :-1]
        ref_p1 = ref_path[:, 1:]
        seg_len = (ref_p1 - ref_p0).norm(dim=-1)
        ref_arc = torch.cumsum(seg_len, dim=-1) - seg_len
        ref_progress = seg_len.sum(dim=-1)

        return OracleScene(
            ego_half=ego_half,
            centre_offset=centre_offset,
            neighbour_xy=take(nb_xy),
            neighbour_axis=take(nb_axis),
            neighbour_half=take(nb_half),
            neighbour_valid=take(valid),
            neighbour_stopped=take(nb_stopped),
            border_p0=border_p0,
            border_p1=border_p1,
            border_valid=border_valid,
            route_centre0=take_route(route_centre0),
            route_centre1=take_route(route_centre1),
            route_left=take_route(route_left),
            route_right=take_route(route_right),
            route_valid=take_route(route_valid),
            reference_p0=ref_p0,
            reference_p1=ref_p1,
            reference_arc=ref_arc,
            reference_progress=ref_progress,
            ego_initial=initial_states_from_ego(inputs["ego_current_state"].to(torch.float64)),
            # ``ego_shape`` is (wheel_base, length, width); this dataset's ego is a
            # 10.7 m vehicle with a 4.99 m wheel base, so feeding the real value to
            # the motion model rather than pacifica's 3.089 m is not cosmetic.
            wheel_base=ego_shape[:, 0].to(torch.float64),
            history_states=self._history_states(inputs),
        )

    def _history_states(self, inputs: dict) -> torch.Tensor:
        """The comfort prefix: finite-difference states for the recorded past.

        ``history_comfort`` prepends the ego's own past to the simulated future
        (``navsim_score.py::_history_comfort``), dropping the last recorded pose
        because the rollout's row 0 already is the current state.  The past is
        never simulated -- it already happened, so there is no tracker to run.

        The past is taken in the frame ``StatePerturbation`` leaves it in (the
        recorded one -- its transform block is commented out), which is also the
        right choice: the six comfort bounds read only the body-frame accel
        channels, which a rigid transform leaves invariant, and the raw heading,
        where re-framing into the perturbed frame would inject the perturbation
        as a step at the junction.  Measured over 256 augmented samples: accel
        channels agree to 4e-4 either way, but the junction heading step grows
        from 0.031 to 0.196 rad and yaw_accel drops from 1.000 to 0.973.
        """
        past = inputs.get("ego_agent_past")
        if past is None:
            raise KeyError("history_comfort needs 'ego_agent_past' in the oracle inputs")
        past = past.to(torch.float64)
        if past.shape[-1] == 3:  # (x, y, heading) -> (x, y, cos, sin)
            heading = past[..., 2]
            past = torch.stack((past[..., 0], past[..., 1], heading.cos(), heading.sin()), dim=-1)
        if past.shape[-2] < 2:
            return past.new_zeros(past.shape[0], 0, STATE_SIZE)
        return _states_from_poses(past[:, :-1], self.dt)

    # -- scene helpers -----------------------------------------------------
    def _neighbour_speed(
        self, nb_xy: torch.Tensor, valid: torch.Tensor, past: torch.Tensor
    ) -> torch.Tensor:
        """Per-step neighbour speed from the GT future, falling back to the past.

        The GT future carries no velocity channel, so speed comes from
        consecutive *valid* positions; where no valid neighbour step exists the
        last observed past velocity stands in.  ``is_track_stopped`` only cares
        about a 0.05 m/s threshold, so this is plenty.
        """
        dt = self.dt
        forward = torch.zeros_like(nb_xy[..., 0])
        forward[..., :-1] = (nb_xy[..., 1:, :] - nb_xy[..., :-1, :]).norm(dim=-1) / dt
        forward_ok = torch.zeros_like(valid)
        forward_ok[..., :-1] = valid[..., 1:] & valid[..., :-1]

        backward = torch.zeros_like(forward)
        backward[..., 1:] = forward[..., :-1]
        backward_ok = torch.zeros_like(valid)
        backward_ok[..., 1:] = forward_ok[..., :-1]

        fallback = past[:, :, -1, 4:6].norm(dim=-1)[:, :, None].expand_as(forward)
        speed = torch.where(forward_ok, forward, torch.where(backward_ok, backward, fallback))
        return speed

    def _border_segments(self, line_strings: torch.Tensor):
        """Road-border polylines -> flat segment list.

        ``line_strings`` is ``[B, L, P, 4]`` = (x, y, type one-hot[2]); channel 3
        is the road-border type (``loss.py`` and ``validate_model.py`` use the
        same ``> 0.5`` test) and a padded point has an all-zero one-hot.
        """
        batch, num_lines, num_points, _ = line_strings.shape
        point_valid = line_strings[..., 2:4].sum(-1) > 0.5
        is_border = (line_strings[..., 3] > 0.5).any(dim=-1)  # [B, L]

        xy = line_strings[..., :2]
        p0 = xy[:, :, :-1]
        p1 = xy[:, :, 1:]
        valid = point_valid[:, :, :-1] & point_valid[:, :, 1:] & is_border[:, :, None]
        flat = batch, num_lines * (num_points - 1)
        return (
            p0.reshape(*flat, 2),
            p1.reshape(*flat, 2),
            valid.reshape(*flat),
        )

    def _route_segments(self, route: torch.Tensor):
        """Route lanes -> centreline segments with signed left/right offsets.

        ``route`` is ``[B, R, P, 33]``; channels 0:2 are the centreline point and
        4:6 / 6:8 the left / right boundary *offsets* from it.  Rather than
        rasterising the 40-gon ribbon, each segment keeps its lateral extent as a
        pair of signed offsets, which makes the point-in-ribbon test a projection
        plus two comparisons.
        """
        batch, num_route, num_points, _ = route.shape
        centre = route[..., :2]
        left = route[..., 4:6]
        right = route[..., 6:8]
        point_valid = route[..., :8].abs().sum(-1) > 0

        c0 = centre[:, :, :-1]
        c1 = centre[:, :, 1:]
        valid = point_valid[:, :, :-1] & point_valid[:, :, 1:]

        # Signed lateral extent of the ribbon, measured in the segment's own
        # left-normal direction and averaged over the segment's two endpoints.
        tangent = c1 - c0
        tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(_EPS)
        normal = _perp(tangent)
        left_signed = 0.5 * (
            (left[:, :, :-1] * normal).sum(-1) + (left[:, :, 1:] * normal).sum(-1)
        )
        right_signed = 0.5 * (
            (right[:, :, :-1] * normal).sum(-1) + (right[:, :, 1:] * normal).sum(-1)
        )
        lower = torch.minimum(left_signed, right_signed)
        upper = torch.maximum(left_signed, right_signed)

        flat = batch, num_route * (num_points - 1)
        return (
            c0.reshape(*flat, 2),
            c1.reshape(*flat, 2),
            upper.reshape(*flat),
            lower.reshape(*flat),
            valid.reshape(*flat),
        )

    @staticmethod
    def _guard_ball(
        ref_xy: torch.Tensor, proposals: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-step ball covering the expert path and every proposal.

        Returns ``(centre [B, T, 2], radius [B, T])``.  With no proposals the
        ball degenerates to the expert path itself, which is only conservative
        for trajectories that stay near the demonstration -- callers that score
        real proposals should always pass them.
        """
        if proposals is None:
            return ref_xy, torch.zeros_like(ref_xy[..., 0])
        cloud = torch.cat((proposals[..., :2].float(), ref_xy[:, None]), dim=1)  # [B, N+1, T, 2]
        centre = cloud.mean(dim=1)
        radius = (cloud - centre[:, None]).norm(dim=-1).amax(dim=1)
        return centre, radius

    def _reduce_segments(
        self,
        p0: torch.Tensor,
        p1: torch.Tensor,
        valid: torch.Tensor,
        guard_centre: torch.Tensor,
        guard_radius: torch.Tensor,
        keep: int,
    ):
        """Keep the ``keep`` segments closest to the guard ball (static shapes)."""
        batch = p0.shape[0]
        keep = min(keep, p0.shape[1])
        distance = _point_segment_distance(guard_centre[:, :, None, :], p0[:, None], p1[:, None])
        rank = (distance - guard_radius[:, :, None]).min(dim=1).values
        rank = torch.where(valid, rank, torch.full_like(rank, 1e6))
        order = rank.topk(keep, dim=1, largest=False).indices
        gather2 = order[:, :, None].expand(batch, keep, 2)
        return p0.gather(1, gather2), p1.gather(1, gather2), valid.gather(1, order)

    # -- per-chunk scoring -------------------------------------------------
    def _score_chunk(self, proposals: torch.Tensor, scene: OracleScene) -> torch.Tensor:
        """The five proposal-local metrics; EP is filled in by the caller."""
        axis = _heading_unit(proposals)  # [B, N, T, 2]
        centre = proposals[..., :2] + axis * scene.centre_offset[:, None, None, None]
        speed = self._ego_speed(proposals)  # [B, N, T]

        nc, ttc = self._collision_scores(proposals, centre, axis, speed, scene)
        dac = self._drivable_area(centre, axis, scene)
        ddc = self._driving_direction(proposals, scene)
        comfort = self._comfort(proposals, scene)
        placeholder = torch.zeros_like(nc)
        return torch.stack((nc, dac, ddc, ttc, placeholder, comfort), dim=-1)

    def _ego_speed(self, proposals: torch.Tensor) -> torch.Tensor:
        xy = proposals[..., :2]
        velocity = torch.stack(
            (_gradient(xy[..., 0], self.dt), _gradient(xy[..., 1], self.dt)), dim=-1
        )
        return velocity.norm(dim=-1)

    def _collision_scores(
        self,
        proposals: torch.Tensor,
        centre: torch.Tensor,
        axis: torch.Tensor,
        speed: torch.Tensor,
        scene: OracleScene,
    ):
        """NC and TTC in one pass over the (proposal, step, neighbour) grid."""
        batch, num, horizon, _ = proposals.shape
        device = proposals.device

        steps = torch.arange(
            self.collision_stride - 1, horizon, self.collision_stride, device=device
        )
        ego_half = scene.ego_half[:, None, None, None, :]
        nb_half = scene.neighbour_half[:, None, None, :, :]

        # [B, N, Ts, K]
        nb_xy = scene.neighbour_xy[:, None, :, :, :].permute(0, 1, 3, 2, 4)[:, :, steps]
        nb_axis = scene.neighbour_axis[:, None, :, :, :].permute(0, 1, 3, 2, 4)[:, :, steps]
        nb_valid = scene.neighbour_valid[:, None, :, :].permute(0, 1, 3, 2)[:, :, steps]
        nb_stopped = scene.neighbour_stopped[:, None, :, :].permute(0, 1, 3, 2)[:, :, steps]

        ego_centre = centre[:, :, steps, None, :]
        ego_axis = axis[:, :, steps, None, :]
        ego_raw = proposals[:, :, steps, None, :2]
        ego_speed = speed[:, :, steps, None]

        delta = nb_xy - ego_centre
        overlap = _obb_overlap(delta, ego_axis, ego_half, nb_axis, nb_half) & nb_valid

        # Ahead / behind is measured from the RAW (rear-axle) pose, as in nuplan.
        to_agent = nb_xy - ego_raw
        cos_angle = self._cos_relative(to_agent, ego_axis)
        behind = cos_angle < _COS_BEHIND

        front_hit = self._front_edge_hits(
            ego_centre, ego_axis, ego_half, nb_xy, nb_axis, nb_half
        )
        ego_stopped = ego_speed <= COLLISION_STOPPED_SPEED_THRESHOLD
        at_fault = overlap & ~ego_stopped & (nb_stopped | (~behind & front_hit))
        nc = 1.0 - at_fault.any(dim=-1).any(dim=-1).float()

        ttc = self._time_to_collision(proposals, centre, axis, speed, scene)
        return nc, ttc

    @staticmethod
    def _cos_relative(to_agent: torch.Tensor, ego_axis: torch.Tensor) -> torch.Tensor:
        """cos of nuplan's ego-heading-to-agent angle; full overlap counts as ahead."""
        norm = to_agent.norm(dim=-1)
        unit = to_agent / norm.clamp_min(_EPS).unsqueeze(-1)
        cos_angle = (unit * ego_axis).sum(-1).clamp(-1.0, 1.0)
        return torch.where(norm < 1e-9, torch.ones_like(cos_angle), cos_angle)

    @staticmethod
    def _front_edge_hits(
        ego_centre: torch.Tensor,
        ego_axis: torch.Tensor,
        ego_half: torch.Tensor,
        nb_xy: torch.Tensor,
        nb_axis: torch.Tensor,
        nb_half: torch.Tensor,
    ) -> torch.Tensor:
        """Does the ego's front bumper edge (FL->FR) touch the neighbour box?"""
        left = _perp(ego_axis)
        front = ego_centre + ego_axis * ego_half[..., 0:1]
        offset = left * ego_half[..., 1:2]
        return _segment_hits_box(front + offset, front - offset, nb_xy, nb_axis, nb_half)

    def _time_to_collision(
        self,
        proposals: torch.Tensor,
        centre: torch.Tensor,
        axis: torch.Tensor,
        speed: torch.Tensor,
        scene: OracleScene,
    ) -> torch.Tensor:
        """1 s constant-velocity forward projection against the GT futures."""
        batch, num, horizon, _ = proposals.shape
        device = proposals.device

        future_idcs = list(range(0, int(FUTURE_COLLISION_HORIZON_WINDOW * 10), 3))
        max_future = max(future_idcs)
        last = horizon - max_future
        steps = torch.arange(self.ttc_stride - 1, max(last, 1), self.ttc_stride, device=device)
        if steps.numel() == 0:
            steps = torch.zeros(1, dtype=torch.long, device=device)

        ego_half = scene.ego_half[:, None, None, None, :]
        nb_half = scene.neighbour_half[:, None, None, :, :]
        nb_xy_all = scene.neighbour_xy.permute(0, 2, 1, 3)[:, None]  # [B, 1, T, K, 2]
        nb_axis_all = scene.neighbour_axis.permute(0, 2, 1, 3)[:, None]
        nb_valid_all = scene.neighbour_valid.permute(0, 2, 1)[:, None]  # [B, 1, T, K]

        ego_centre = centre[:, :, steps, None, :]
        ego_axis = axis[:, :, steps, None, :]
        ego_raw = proposals[:, :, steps, None, :2]
        ego_speed = speed[:, :, steps, None]
        moving = ego_speed >= STOPPED_SPEED_THRESHOLD

        infraction = torch.zeros(
            moving.shape[:-1] + (nb_valid_all.shape[-1],), dtype=torch.bool, device=device
        )
        for future in future_idcs:
            target = (steps + future).clamp(max=horizon - 1)
            nb_xy = nb_xy_all[:, :, target]
            nb_axis = nb_axis_all[:, :, target]
            nb_valid = nb_valid_all[:, :, target]

            shifted = ego_centre + ego_axis * (ego_speed * (future * self.dt)).unsqueeze(-1)
            overlap = _obb_overlap(nb_xy - shifted, ego_axis, ego_half, nb_axis, nb_half)
            ahead = self._cos_relative(nb_xy - ego_raw, ego_axis) > _COS_AHEAD
            infraction |= overlap & nb_valid & ahead & moving

        any_evaluable = moving.any(dim=-1).any(dim=-1)
        hit = infraction.any(dim=-1).any(dim=-1)
        ttc = 1.0 - hit.float()
        # Nothing evaluable (ego stationary for the whole horizon) -> undefined,
        # which DrivoR's loss masks out instead of supervising a made-up label.
        return torch.where(any_evaluable, ttc, torch.full_like(ttc, TTC_UNDEFINED))

    def _drivable_area(
        self, centre: torch.Tensor, axis: torch.Tensor, scene: OracleScene
    ) -> torch.Tensor:
        """DAC: 1 unless the footprint ever crosses a road-border polyline."""
        batch, num, horizon, _ = centre.shape
        device = centre.device
        steps = torch.arange(self.border_stride - 1, horizon, self.border_stride, device=device)

        ego_centre = centre[:, :, steps, None, :]
        ego_axis = axis[:, :, steps, None, :]
        ego_half = scene.ego_half[:, None, None, None, :]

        p0 = scene.border_p0[:, None, None]
        p1 = scene.border_p1[:, None, None]
        hit = _segment_hits_box(p0, p1, ego_centre, ego_axis, ego_half)
        hit &= scene.border_valid[:, None, None]
        return 1.0 - hit.any(dim=-1).any(dim=-1).float()

    def _driving_direction(self, proposals: torch.Tensor, scene: OracleScene) -> torch.Tensor:
        """DDC: displacement accumulated while off-route, over a 1 s window."""
        batch, num, horizon, _ = proposals.shape
        device = proposals.device
        steps = torch.arange(0, horizon, self.route_stride, device=device)

        origin = torch.zeros(batch, num, 1, 2, device=device, dtype=proposals.dtype)
        path = torch.cat((origin, proposals[:, :, steps, :2]), dim=2)  # [B, N, Ts+1, 2]

        point = path[:, :, :, None, :]
        seg0 = scene.route_centre0[:, None, None]
        seg1 = scene.route_centre1[:, None, None]
        seg = seg1 - seg0
        length_sq = (seg * seg).sum(-1).clamp_min(_EPS)
        rel = point - seg0
        t = ((rel * seg).sum(-1) / length_sq)
        lateral = (rel * _perp(seg / seg.norm(dim=-1, keepdim=True).clamp_min(_EPS))).sum(-1)
        on_segment = (t >= 0.0) & (t <= 1.0)
        within = (lateral >= scene.route_right[:, None, None]) & (
            lateral <= scene.route_left[:, None, None]
        )
        inside = (on_segment & within & scene.route_valid[:, None, None]).any(dim=-1)

        # No route coverage at all -> no oncoming evidence (the CPU port's
        # semantics; a windowed route tensor must not punish coverage gaps).
        has_route = scene.route_valid.any(dim=-1)[:, None]

        displacement = torch.zeros_like(path[..., 0])
        displacement[:, :, 1:] = (path[:, :, 1:] - path[:, :, :-1]).norm(dim=-1)
        displacement = torch.where(inside, torch.zeros_like(displacement), displacement)

        # Sliding 1 s window: sum(disp[t-window : t+1]) == cum[t] - cum[t-window-1].
        window = max(int(round(DRIVING_DIRECTION_HORIZON / (self.dt * self.route_stride))), 1)
        cumulative = torch.cumsum(displacement, dim=-1)
        padded = torch.cat(
            (torch.zeros_like(cumulative[:, :, :1]).expand(-1, -1, window + 1), cumulative), dim=-1
        )
        worst = (cumulative - padded[:, :, : cumulative.shape[-1]]).amax(dim=-1)

        ddc = torch.where(
            worst < DRIVING_DIRECTION_COMPLIANCE_THRESHOLD,
            torch.ones_like(worst),
            torch.where(
                worst < DRIVING_DIRECTION_VIOLATION_THRESHOLD,
                torch.full_like(worst, 0.5),
                torch.zeros_like(worst),
            ),
        )
        return torch.where(has_route, ddc, torch.ones_like(ddc))

    def _comfort(self, proposals: torch.Tensor, scene: OracleScene) -> torch.Tensor:
        """The six NAVSIM comfort bounds, all-or-nothing, over the whole horizon.

        NAVSIM never applies these bounds to raw waypoints.  ``PDMScorer`` scores
        the states ``PDMSimulator.simulate_proposals`` produces -- a
        ``BatchLQRTracker`` computing (acceleration, steering-rate) commands that
        a ``BatchKinematicBicycleModel`` integrates -- so the metric measures how
        a tracked vehicle would ride, not the trajectory head's output noise.  The
        model's first-order lags (``accel_time_constant = 0.2 s``,
        ``steering_angle_time_constant = 0.05 s``) are what remove the
        high-frequency component that finite differences amplify by ``1/dt**2``.
        Without the rollout every proposal fails ``lon_accel`` and the label
        collapses to a constant 0, killing both the comfort head's gradient and
        the metric's 2/12 share of the selection aggregate.

        The rollout runs in fp64 (see :mod:`planner_metrics.pdm_simulator_torch`)
        because the LQR fits are ill-conditioned enough that fp32 changes the
        commands; on an H100 fp64 is ~1:2 of fp32 so this costs nothing.  Row 0
        of the simulated array is the ego's own state and NAVSIM scores it, so the
        comfort horizon is ``T + 1`` samples at ``time_point_s = arange(0, T+1) *
        dt`` -- ``_calculate_is_comfortable`` verbatim.

        The metric is the devkit's ``history_comfort``, not navsim's bare
        ``COMFORTABLE``: the ego's recorded past is prepended to the simulated
        future and every bound must hold over the concatenation
        (``navsim_score.py::_history_comfort``).  The prefix is shared by all
        proposals of a scene, so it gates that scene to 0 when the recorded
        driving itself was uncomfortable -- 3.3 % of scenes, measured.
        """
        batch, num, horizon_in, _ = proposals.shape
        poses = torch.stack(
            (
                proposals[..., 0],
                proposals[..., 1],
                torch.atan2(proposals[..., 3], proposals[..., 2]),
            ),
            dim=-1,
        ).reshape(-1, horizon_in, 3)
        initial = (
            scene.ego_initial[:, None].expand(batch, num, STATE_SIZE).reshape(-1, STATE_SIZE)
        )
        wheel_base = scene.wheel_base[:, None].expand(batch, num).reshape(-1)
        states = simulate_proposals(poses.to(torch.float64), initial, self.dt, wheel_base)

        prefix = scene.history_states
        if prefix.shape[1] > 0:
            prefix = (
                prefix[:, None]
                .expand(batch, num, prefix.shape[1], STATE_SIZE)
                .reshape(-1, prefix.shape[1], STATE_SIZE)
            )
            states = torch.cat((prefix, states), dim=1)

        horizon = states.shape[1]  # H - 1 + T + 1
        dt = self.dt
        acc_lon = states[..., STATE_ACC_X]
        # Identically zero for every row past the first: ``_update_commands``
        # writes 0.0 into ACCELERATION_Y and ``propagate_state`` copies it
        # through, which is what makes NAVSIM's lateral bound near-vacuous.
        acc_lat = states[..., STATE_ACC_Y]
        heading = states[..., STATE_HEADING]

        smooth = self._operator(horizon, 8, 2, 0, 1.0, states)
        jerk_op = self._operator(horizon, 15, 2, 1, dt, states)
        yaw_rate_op = self._operator(horizon, 5, 2, 1, dt, states)
        yaw_accel_op = self._operator(horizon, 5, 3, 2, dt, states)

        acc_lon_s = acc_lon @ smooth
        acc_lat_s = acc_lat @ smooth
        acc_mag_s = torch.hypot(acc_lon, acc_lat) @ smooth
        unwrapped = _phase_unwrap(heading)

        checks = (
            _within(acc_lon_s, MIN_LON_ACCEL, MAX_LON_ACCEL),
            _within(acc_lat_s, -MAX_ABS_LAT_ACCEL, MAX_ABS_LAT_ACCEL),
            _within(acc_mag_s @ jerk_op, -MAX_ABS_MAG_JERK, MAX_ABS_MAG_JERK),
            _within(acc_lon_s @ jerk_op, -MAX_ABS_LON_JERK, MAX_ABS_LON_JERK),
            _within(unwrapped @ yaw_accel_op, -MAX_ABS_YAW_ACCEL, MAX_ABS_YAW_ACCEL),
            _within(unwrapped @ yaw_rate_op, -MAX_ABS_YAW_RATE, MAX_ABS_YAW_RATE),
        )
        comfortable = checks[0]
        for check in checks[1:]:
            comfortable = comfortable & check
        return comfortable.float().reshape(batch, num)

    def _progress(
        self, proposals: torch.Tensor, scene: OracleScene, multiplicative: torch.Tensor
    ) -> torch.Tensor:
        """EP: arclength gained along the expert path, normalized proposal-relative."""
        seg0 = scene.reference_p0[:, None]
        seg1 = scene.reference_p1[:, None]
        arc = scene.reference_arc[:, None]

        start = _project_arclength(proposals[:, :, 0, :2], seg0, seg1, arc)
        end = _project_arclength(proposals[:, :, -1, :2], seg0, seg1, arc)
        raw = (end - start).clamp_min(0.0) * multiplicative

        # DrivoR's train_pdm_scorer.py:169-179 normalizes each proposal by
        # ``np.maximum(raw_progress, self.pdm_progress)`` -- per proposal, not by
        # the best of the set.  Here the reference polyline *is* the expert's own
        # path and ``_project_arclength`` clamps onto it, so ``raw <= reference``
        # always and this maximum is just ``reference``; keeping DrivoR's exact
        # form costs nothing and stays correct if the reference is ever swapped
        # for the route centerline, where a proposal can out-progress it.
        reference = scene.reference_progress[:, None]
        max_raw = torch.maximum(raw, reference)
        gated = max_raw <= PROGRESS_DISTANCE_THRESHOLD
        score = (raw / max_raw.clamp_min(_EPS)).clamp(0.0, 1.0)
        # Gate: navsim returns 1.0 (or 0.0 when the multiplicative terms zeroed
        # the proposal) whenever no proposal can make meaningful progress.
        return torch.where(gated, (multiplicative != 0).float(), score)

    def _operator(
        self, length: int, window: int, poly: int, deriv: int, delta: float, like: torch.Tensor
    ) -> torch.Tensor:
        key = (length, window, poly, deriv, float(delta), like.device, like.dtype)
        cached = self._savgol.get(key)
        if cached is None:
            matrix = _savgol_operator(length, window, poly, deriv, delta)
            cached = torch.as_tensor(matrix, device=like.device, dtype=like.dtype)
            self._savgol[key] = cached
        return cached


def _within(values: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """``pdms_navsim._within_bound``: strict bounds, all timesteps."""
    return ((values > low) & (values < high)).all(dim=-1)


__all__ = [
    "DrivoROracle",
    "OracleScene",
    "ORACLE_METRIC_NAMES",
    "TTC_UNDEFINED",
]
