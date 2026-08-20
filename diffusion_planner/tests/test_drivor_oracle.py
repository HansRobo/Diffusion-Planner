"""Parity tests: the batched GPU PDM oracle vs the scipy/shapely CPU reference.

``diffusion_planner.utils.drivor_oracle`` produces DrivoR's scorer labels for
every (scene, proposal) pair on the GPU.  Its whole reason to exist is that the
labels agree with ``planner_metrics.pdms_navsim`` -- the port the validation panel
reports against -- so that agreement is tested here rather than asserted in a
docstring.

Run with the per-metric strides set to 1 so both sides see the same horizon grid;
production strides are a documented speed/fidelity trade-off, not a semantic one.
"""

import numpy as np
import pytest
import torch

from diffusion_planner.utils.drivor_oracle import (
    ORACLE_METRIC_NAMES,
    TTC_UNDEFINED,
    DrivoROracle,
    _obb_overlap,
)
from planner_metrics import pdms_navsim as ref
from planner_metrics.pdm_simulator import STATE_ANGULAR_VELOCITY
from planner_metrics.pdm_simulator_torch import initial_states_from_ego, simulate_proposals

DT = 0.1
HORIZON = 40
HISTORY = 31  # the NPZ's ego_agent_past length: 3.0 s of past plus the current pose
NUM_NEIGHBOURS = 6
NUM_LINE_STRINGS = 4
POINTS_PER_LINE_STRING = 20
NUM_ROUTE = 3
POINTS_PER_ROUTE = 20

EGO_SHAPE = (2.8, 4.9, 2.0)  # (wheel_base, length, width)


def _exact_oracle() -> DrivoROracle:
    """Strides of 1 and no candidate pruning: the reference's exact grid."""
    return DrivoROracle(
        dt=DT,
        collision_stride=1,
        ttc_stride=1,
        border_stride=1,
        route_stride=1,
        max_neighbours=NUM_NEIGHBOURS,
        max_border_segments=NUM_LINE_STRINGS * (POINTS_PER_LINE_STRING - 1),
        max_route_segments=NUM_ROUTE * (POINTS_PER_ROUTE - 1),
        proposal_chunk=0,
    )


# ---------------------------------------------------------------------------
# synthetic scene construction
# ---------------------------------------------------------------------------
def _poses_from_xy(xy: np.ndarray) -> np.ndarray:
    """(T, 2) -> (T, 4) with headings from the path tangent."""
    delta = np.gradient(xy, axis=0)
    heading = np.arctan2(delta[:, 1], delta[:, 0])
    return np.stack([xy[:, 0], xy[:, 1], np.cos(heading), np.sin(heading)], axis=-1)


def _straight_proposal(speed: float, lateral: float = 0.0, curvature: float = 0.0) -> np.ndarray:
    t = np.arange(1, HORIZON + 1) * DT
    x = speed * t
    y = lateral + 0.5 * curvature * x**2
    return _poses_from_xy(np.stack([x, y], axis=-1))


def _ego_current_state(batch: int, speed: float = 0.0) -> torch.Tensor:
    """DP's ``[x, y, cos, sin, vx, vy, ax, ay, steering_angle, yaw_rate]`` row.

    This is where the comfort rollout starts, so it is not filler: a rollout
    started from rest spends its first second catching up to a moving reference
    and blows the longitudinal-accel bound.
    """
    ego = torch.zeros(batch, 10, dtype=torch.float32)
    ego[:, 2] = 1.0  # cos(heading)
    ego[:, 4] = speed
    return ego


def _straight_past(batch: int, speed: float) -> torch.Tensor:
    """A constant-speed straight past ending exactly at the ego, as in the NPZ.

    ``history_comfort`` finite-differences this and prepends it to the rollout, so
    it is part of every comfort label -- a past disagreeing with the ego's speed
    would put an acceleration step at the junction.  Layout matches the dataset
    after ``prepare_batch``: ``[B, 31, 4]`` = (x, y, cos, sin), oldest first, last
    row exactly (0, 0, 1, 0).
    """
    t = (np.arange(HISTORY) - (HISTORY - 1)) * DT  # ..., -0.2, -0.1, 0.0
    past = np.zeros((HISTORY, 4), dtype=np.float32)
    past[:, 0] = speed * t
    past[:, 2] = 1.0
    return torch.as_tensor(np.tile(past, (batch, 1, 1)))


def _empty_inputs(batch: int, speed: float = 0.0) -> dict:
    return {
        "ego_shape": torch.tensor([EGO_SHAPE] * batch, dtype=torch.float32),
        "ego_current_state": _ego_current_state(batch, speed),
        "ego_agent_past": _straight_past(batch, speed),
        "neighbor_agents_past": torch.zeros(batch, NUM_NEIGHBOURS, 31, 11),
        "neighbor_agents_future": torch.zeros(batch, NUM_NEIGHBOURS, HORIZON, 3),
        "line_strings": torch.zeros(
            batch, NUM_LINE_STRINGS, POINTS_PER_LINE_STRING, 4
        ),
        "route_lanes": torch.zeros(batch, NUM_ROUTE, POINTS_PER_ROUTE, 33),
    }


def _add_neighbour(inputs: dict, batch_index: int, slot: int, xy: np.ndarray, heading, size):
    """size = (width, length); ``xy`` is (T, 2), ``heading`` scalar or (T,)."""
    heading = np.broadcast_to(np.asarray(heading, dtype=np.float32), (xy.shape[0],))
    future = inputs["neighbor_agents_future"]
    future[batch_index, slot, :, 0] = torch.as_tensor(xy[:, 0], dtype=torch.float32)
    future[batch_index, slot, :, 1] = torch.as_tensor(xy[:, 1], dtype=torch.float32)
    future[batch_index, slot, :, 2] = torch.as_tensor(heading.copy(), dtype=torch.float32)
    past = inputs["neighbor_agents_past"]
    past[batch_index, slot, -1, 0] = float(xy[0, 0])
    past[batch_index, slot, -1, 1] = float(xy[0, 1])
    past[batch_index, slot, -1, 2] = float(np.cos(heading[0]))
    past[batch_index, slot, -1, 3] = float(np.sin(heading[0]))
    speed = np.linalg.norm(np.gradient(xy, DT, axis=0), axis=-1)
    past[batch_index, slot, -1, 4] = float(speed[0] * np.cos(heading[0]))
    past[batch_index, slot, -1, 5] = float(speed[0] * np.sin(heading[0]))
    past[batch_index, slot, -1, 6] = float(size[0])
    past[batch_index, slot, -1, 7] = float(size[1])
    past[batch_index, slot, -1, 8] = 1.0


def _reference_boxes(inputs: dict, batch_index: int) -> list:
    """Per-timestep ``[n, 9]`` agent boxes in the reference's (x,y,z,w,l,h,yaw,vx,vy) layout.

    Velocities use the same consecutive-position estimate the oracle derives, so a
    mismatch can only come from the metric logic, not from a different speed.
    """
    future = inputs["neighbor_agents_future"][batch_index].numpy()
    past = inputs["neighbor_agents_past"][batch_index].numpy()
    valid_track = np.abs(past[:, -1, :]).sum(-1) > 0
    valid_step = np.abs(future).sum(-1) > 0
    boxes_per_t = []
    for t in range(future.shape[1]):
        rows = []
        for a in range(future.shape[0]):
            if not (valid_track[a] and valid_step[a, t]):
                continue
            xy = future[a, :, :2]
            if t + 1 < future.shape[1] and valid_step[a, t + 1]:
                speed = np.linalg.norm(xy[t + 1] - xy[t]) / DT
            elif t > 0 and valid_step[a, t - 1]:
                speed = np.linalg.norm(xy[t] - xy[t - 1]) / DT
            else:
                speed = float(np.linalg.norm(past[a, -1, 4:6]))
            yaw = float(future[a, t, 2])
            rows.append(
                [
                    float(xy[t, 0]),
                    float(xy[t, 1]),
                    0.0,
                    float(past[a, -1, 6]),
                    float(past[a, -1, 7]),
                    1.5,
                    yaw,
                    speed * np.cos(yaw),
                    speed * np.sin(yaw),
                ]
            )
        boxes_per_t.append(np.asarray(rows, dtype=np.float64).reshape(-1, 9))
    return boxes_per_t


def _add_line_string(inputs: dict, batch_index: int, slot: int, xy: np.ndarray, is_border: bool):
    tensor = inputs["line_strings"]
    count = xy.shape[0]
    tensor[batch_index, slot, :count, 0] = torch.as_tensor(xy[:, 0], dtype=torch.float32)
    tensor[batch_index, slot, :count, 1] = torch.as_tensor(xy[:, 1], dtype=torch.float32)
    tensor[batch_index, slot, :count, 3 if is_border else 2] = 1.0


def _add_route(inputs: dict, batch_index: int, slot: int, centre: np.ndarray, half_width: float):
    tensor = inputs["route_lanes"]
    count = centre.shape[0]
    tangent = np.gradient(centre, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=-1, keepdims=True), 1e-9)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=-1)
    tensor[batch_index, slot, :count, 0:2] = torch.as_tensor(centre, dtype=torch.float32)
    tensor[batch_index, slot, :count, 2:4] = torch.as_tensor(tangent, dtype=torch.float32)
    tensor[batch_index, slot, :count, 4:6] = torch.as_tensor(
        normal * half_width, dtype=torch.float32
    )
    tensor[batch_index, slot, :count, 6:8] = torch.as_tensor(
        -normal * half_width, dtype=torch.float32
    )


def _ego_future(paths: list) -> torch.Tensor:
    return torch.as_tensor(np.stack(paths), dtype=torch.float32)


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
def _sat_reference(ego, other) -> bool:
    ego_corners = ref.ego_corners(ego[0], ego[1], ego[2], ego[3], ego[4])
    # reference box layout: (x, y, z, width, length, height, yaw, vx, vy)
    box = np.asarray(
        [[other[0], other[1], 0.0, other[3], other[4], 1.5, other[2], 0.0, 0.0]],
        dtype=np.float64,
    )
    other_corners = ref.box_corners_batch(box)
    return bool(ref._sat_intersects_one_to_many(ego_corners, other_corners)[0])


def test_obb_overlap_matches_sat_reference():
    rng = np.random.default_rng(0)
    agree = 0
    for _ in range(500):
        ego = (
            rng.uniform(-6, 6),
            rng.uniform(-6, 6),
            rng.uniform(-np.pi, np.pi),
            rng.uniform(2.0, 5.0),  # length
            rng.uniform(1.0, 2.5),  # width
        )
        other = (
            rng.uniform(-6, 6),
            rng.uniform(-6, 6),
            rng.uniform(-np.pi, np.pi),
            rng.uniform(1.0, 2.5),  # width
            rng.uniform(2.0, 5.0),  # length
        )
        expected = _sat_reference(ego, other)
        got = bool(
            _obb_overlap(
                torch.tensor([other[0] - ego[0], other[1] - ego[1]]),
                torch.tensor([np.cos(ego[2]), np.sin(ego[2])], dtype=torch.float32),
                torch.tensor([ego[3] / 2, ego[4] / 2], dtype=torch.float32),
                torch.tensor([np.cos(other[2]), np.sin(other[2])], dtype=torch.float32),
                torch.tensor([other[4] / 2, other[3] / 2], dtype=torch.float32),
            )
        )
        assert got == expected, (ego, other, got, expected)
        agree += 1
    assert agree == 500


# ---------------------------------------------------------------------------
# comfort
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "speed,curvature",
    [(0.0, 0.0), (2.0, 0.0), (8.0, 0.0), (8.0, 0.01), (12.0, 0.05), (5.0, 0.2)],
)
def test_comfort_matches_reference(speed, curvature):
    poses = _straight_proposal(speed, curvature=curvature)
    inputs = _empty_inputs(1, speed=speed)
    expected = float(
        ref.comfort_score(
            poses,
            DT,
            inputs["ego_current_state"].numpy(),
            EGO_SHAPE[0],
            history_poses=inputs["ego_agent_past"].numpy(),
        )
    )
    oracle = _exact_oracle()
    got = oracle(
        torch.as_tensor(poses, dtype=torch.float32)[None, None],
        inputs,
        _ego_future([poses]),
    )
    assert got[0, 0, ORACLE_METRIC_NAMES.index("history_comfort")].item() == expected


def test_comfort_matches_reference_on_random_paths():
    rng = np.random.default_rng(7)
    poses = []
    for _ in range(24):
        xy = np.cumsum(rng.normal(scale=0.35, size=(HORIZON, 2)) + np.array([0.8, 0.0]), axis=0)
        poses.append(_poses_from_xy(xy))
    stacked = np.stack(poses)
    inputs = _empty_inputs(1, speed=8.0)
    ego = inputs["ego_current_state"].numpy()
    past = inputs["ego_agent_past"].numpy()
    expected = np.asarray(
        [ref.comfort_score(p, DT, ego, EGO_SHAPE[0], history_poses=past) for p in poses],
        dtype=np.float32,
    )

    oracle = _exact_oracle()
    got = oracle(
        torch.as_tensor(stacked, dtype=torch.float32)[None],
        inputs,
        _ego_future([poses[0]]),
    )
    index = ORACLE_METRIC_NAMES.index("history_comfort")
    np.testing.assert_array_equal(got[0, :, index].numpy(), expected)


def test_comfort_rescues_jittery_proposals():
    """A smooth 8 m/s path with sub-decimetre per-step jitter must stay comfortable.

    This is the regression that made the comfort label a constant 0 for all 64
    proposals, zeroing the head's gradient and capping PDMS at 10/12.  Finite
    differences turn 5 cm of pose noise into ~10 m/s^2 of apparent longitudinal
    acceleration (5 cm / 0.01 s^2), well past the +2.40 bound; the LQR tracker's
    first-order accel lag does not, so the simulated rollout passes.
    """
    rng = np.random.default_rng(3)
    base = np.stack([np.arange(1, HORIZON + 1) * 0.8, np.zeros(HORIZON)], axis=-1)
    jittered = base + rng.normal(scale=0.05, size=base.shape)
    poses = _poses_from_xy(jittered)[None]
    inputs = _empty_inputs(1, speed=8.0)

    oracle = _exact_oracle()
    got = oracle(
        torch.as_tensor(poses, dtype=torch.float32)[None], inputs, _ego_future([poses[0]])
    )
    index = ORACLE_METRIC_NAMES.index("history_comfort")
    assert got[0, 0, index].item() == 1.0

    # ... and the finite-difference path this replaced does not.
    naive = ref.comfort_score_from_states(ref.states_from_poses(poses, DT), DT)
    assert float(naive[0]) == 0.0


def test_comfort_still_rejects_a_genuinely_harsh_trajectory():
    """The bounds must stay live: emergency braking from 15 m/s fails lon-accel.

    Guards against "the simulator makes everything comfortable", which would be
    just as useless a label as a constant 0.
    """
    speed = 15.0
    t = np.arange(1, HORIZON + 1) * DT
    # Stop in 1.5 s: ~-10 m/s^2, far outside the -4.05 lower bound.
    x = np.clip(speed * t - 0.5 * 10.0 * t**2, 0.0, None)
    x = np.maximum.accumulate(x)
    poses = _poses_from_xy(np.stack([x, np.zeros_like(x)], axis=-1))[None]

    oracle = _exact_oracle()
    got = oracle(
        torch.as_tensor(poses, dtype=torch.float32)[None],
        _empty_inputs(1, speed=speed),
        _ego_future([poses[0]]),
    )
    assert got[0, 0, ORACLE_METRIC_NAMES.index("history_comfort")].item() == 0.0


def test_history_comfort_prefixes_the_recorded_past():
    """``history_comfort`` scores the recorded past, not just the rollout.

    navsim's ``COMFORTABLE`` scores the proposal alone; the devkit's
    ``history_comfort`` (``navsim_score.py::_history_comfort``) finite-differences
    the ego's past, drops its last pose -- the rollout's row 0 already is the
    current state -- and requires all six bounds over the concatenation.  The
    prefix is shared by every proposal in a scene, so it is a scene-level gate:
    the same proposal must flip to 0 when only the past changes.
    """
    speed = 8.0
    poses = _straight_proposal(speed)
    proposals = torch.as_tensor(poses, dtype=torch.float32)[None, None]
    inputs = _empty_inputs(1, speed=speed)
    oracle = _exact_oracle()
    index = ORACLE_METRIC_NAMES.index("history_comfort")

    assert oracle(proposals, inputs, _ego_future([poses]))[0, 0, index].item() == 1.0

    # 0.5 m of lateral zig-zag in the past; the proposal is untouched.  The last
    # recorded pose is left alone -- it is dropped -- so this is purely the prefix.
    harsh = inputs["ego_agent_past"].clone()
    harsh[0, :-1:2, 1] += 0.5
    inputs["ego_agent_past"] = harsh
    assert oracle(proposals, inputs, _ego_future([poses]))[0, 0, index].item() == 0.0


def test_history_comfort_matches_reference_on_random_pasts():
    """The GPU prefix is the numpy reference's, bound for bound.

    The jitter grows draw by draw so the prefix sweeps *through* the bounds
    instead of sitting on one side of them -- localization noise crosses the
    lateral-accel bound somewhere around 1-2 cm at this dt, and an all-pass or
    all-fail sweep would make the comparison vacuous, so that is asserted too.
    """
    rng = np.random.default_rng(11)
    speed = 6.0
    poses = _straight_proposal(speed)
    proposals = torch.as_tensor(poses, dtype=torch.float32)[None, None]
    oracle = _exact_oracle()
    index = ORACLE_METRIC_NAMES.index("history_comfort")

    seen = []
    for draw in range(12):
        inputs = _empty_inputs(1, speed=speed)
        past = inputs["ego_agent_past"].numpy().copy()
        scale = 0.005 + 0.003 * draw
        past[..., :2] += rng.normal(scale=scale, size=past[..., :2].shape).astype(np.float32)
        inputs["ego_agent_past"] = torch.as_tensor(past)
        expected = float(
            ref.comfort_score(
                poses,
                DT,
                inputs["ego_current_state"].numpy(),
                EGO_SHAPE[0],
                history_poses=past,
            )
        )
        got = oracle(proposals, inputs, _ego_future([poses]))[0, 0, index].item()
        assert got == expected
        seen.append(expected)

    assert 0.0 in seen and 1.0 in seen, seen


def test_comfort_uses_the_dataset_wheel_base_for_the_motion_model():
    """``ego_shape[0]`` is the wheel base and it reaches the bicycle model.

    The tracker keeps pacifica's 3.089 m either way -- navsim never overrides
    ``_tracker._wheel_base``, only ``_motion_model._vehicle`` -- but the motion
    model must see the real vehicle: this dataset's ego has a 4.99 m wheel base,
    which changes the yaw response the yaw-rate and yaw-accel bounds read.
    """
    t = np.arange(1, HORIZON + 1) * DT
    x = 10.0 * t
    y = 1.75 * np.clip((t - 0.4) / 1.6, 0.0, 1.0) ** 2
    poses = _poses_from_xy(np.stack([x, y], axis=-1))[None]
    proposals = torch.as_tensor(poses, dtype=torch.float32)[None]

    # (1) the plumbing: ego_shape[:, 0] lands in the scene as the wheel base.
    for wheel_base in (2.0, 4.99, 12.0):
        inputs = _empty_inputs(1, speed=10.0)
        inputs["ego_shape"] = torch.tensor([(wheel_base,) + EGO_SHAPE[1:]], dtype=torch.float32)
        scene = _exact_oracle().prepare(inputs, _ego_future([poses[0]]), proposals)
        assert float(scene.wheel_base[0]) == pytest.approx(wheel_base, rel=1e-6)

    # (2) the mechanism: the motion model's yaw response scales with 1 / wheel_base.
    xyh = torch.as_tensor(
        np.stack([x, y, np.arctan2(np.gradient(y), np.gradient(x))], axis=-1)[None],
        dtype=torch.float64,
    )
    initial = initial_states_from_ego(_ego_current_state(1, speed=10.0).to(torch.float64))
    yaw_rates = [
        simulate_proposals(xyh, initial, DT, wheel_base)[0, 1:, STATE_ANGULAR_VELOCITY]
        .abs()
        .max()
        .item()
        for wheel_base in (2.0, 12.0)
    ]
    assert yaw_rates[0] > 2.0 * yaw_rates[1], yaw_rates


# ---------------------------------------------------------------------------
# NC / TTC
# ---------------------------------------------------------------------------
def _nc_ttc_reference(poses: np.ndarray, inputs: dict, batch_index: int):
    states = ref.states_from_poses(poses, DT)
    boxes = _reference_boxes(inputs, batch_index)
    offset = EGO_SHAPE[0] / 2.0
    nc = ref.no_at_fault_collision(
        states, boxes, EGO_SHAPE[1], EGO_SHAPE[2], center_offset=offset
    )
    ttc = ref.time_to_collision(
        states, boxes, EGO_SHAPE[1], EGO_SHAPE[2], DT, center_offset=offset
    )
    return nc, ttc


def _collision_scenarios():
    """(name, proposal, neighbour spec list) covering every at-fault branch."""
    scenarios = []

    # 1. empty scene
    scenarios.append(("empty", _straight_proposal(8.0), []))

    # 2. stationary car straight ahead -> front / stopped-track collision
    stopped = np.repeat(np.array([[12.0, 0.0]]), HORIZON, axis=0)
    scenarios.append(("stopped_ahead", _straight_proposal(8.0), [(stopped, 0.0, (1.8, 4.2))]))

    # 3. same obstacle but the ego does not move -> stopped-ego, not at fault
    scenarios.append(("stopped_ego", _straight_proposal(0.0), [(stopped, 0.0, (1.8, 4.2))]))

    # 4. car overtaking from behind -> rear collision, not at fault
    t = np.arange(1, HORIZON + 1) * DT
    from_behind = np.stack([-14.0 + 16.0 * t, np.zeros_like(t)], axis=-1)
    scenarios.append(
        ("from_behind", _straight_proposal(6.0), [(from_behind, 0.0, (1.8, 4.2))])
    )

    # 5. crossing car from the left -> lateral (no map flags => not at fault)
    crossing = np.stack([np.full_like(t, 15.0), 14.0 - 16.0 * t], axis=-1)
    scenarios.append(
        ("crossing", _straight_proposal(8.0), [(crossing, -np.pi / 2, (1.8, 4.2))])
    )

    # 6. slow leading car -> ego catches up, front collision
    leading = np.stack([9.0 + 1.0 * t, np.zeros_like(t)], axis=-1)
    scenarios.append(("leading_slow", _straight_proposal(9.0), [(leading, 0.0, (1.8, 4.5))]))

    # 7. near miss in the neighbouring lane
    parallel = np.stack([2.0 + 8.0 * t, np.full_like(t, 3.6)], axis=-1)
    scenarios.append(("near_miss", _straight_proposal(8.0), [(parallel, 0.0, (1.8, 4.5))]))

    # 8. multiple agents, one at fault
    scenarios.append(
        (
            "mixed",
            _straight_proposal(7.0),
            [
                (parallel, 0.0, (1.8, 4.5)),
                (from_behind, 0.0, (1.8, 4.2)),
                (np.repeat(np.array([[16.0, 0.3]]), HORIZON, axis=0), 0.1, (1.9, 4.6)),
            ],
        )
    )
    return scenarios


@pytest.mark.parametrize(
    "name,poses,neighbours",
    _collision_scenarios(),
    ids=[scenario[0] for scenario in _collision_scenarios()],
)
def test_nc_and_ttc_match_reference(name, poses, neighbours):
    inputs = _empty_inputs(1)
    for slot, (xy, heading, size) in enumerate(neighbours):
        _add_neighbour(inputs, 0, slot, xy, heading, size)

    expected_nc, expected_ttc = _nc_ttc_reference(poses, inputs, 0)
    got = _exact_oracle()(
        torch.as_tensor(poses, dtype=torch.float32)[None, None], inputs, _ego_future([poses])
    )[0, 0]
    nc = got[ORACLE_METRIC_NAMES.index("no_at_fault_collisions")].item()
    ttc = got[ORACLE_METRIC_NAMES.index("time_to_collision_within_bound")].item()
    assert nc == expected_nc, f"{name}: NC {nc} != {expected_nc}"
    if ttc != TTC_UNDEFINED:
        assert ttc == expected_ttc, f"{name}: TTC {ttc} != {expected_ttc}"
    else:
        # sentinel only where nothing was evaluable (ego never moves)
        assert name == "stopped_ego"


def test_ttc_sentinel_for_stationary_ego():
    poses = _straight_proposal(0.0)
    inputs = _empty_inputs(1)
    got = _exact_oracle()(
        torch.as_tensor(poses, dtype=torch.float32)[None, None], inputs, _ego_future([poses])
    )
    ttc = got[0, 0, ORACLE_METRIC_NAMES.index("time_to_collision_within_bound")].item()
    assert ttc == TTC_UNDEFINED


# ---------------------------------------------------------------------------
# DAC
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("lateral,expected", [(0.0, 1.0), (4.5, 0.0), (-4.5, 0.0), (2.0, 1.0)])
def test_dac_matches_reference(lateral, expected):
    inputs = _empty_inputs(1)
    span = np.linspace(-5.0, 40.0, POINTS_PER_LINE_STRING)
    left = np.stack([span, np.full_like(span, 4.0)], axis=-1)
    right = np.stack([span, np.full_like(span, -4.0)], axis=-1)
    _add_line_string(inputs, 0, 0, left, is_border=True)
    _add_line_string(inputs, 0, 1, right, is_border=True)
    # a non-border line string (lane marking) must be ignored
    _add_line_string(
        inputs, 0, 2, np.stack([span, np.zeros_like(span)], axis=-1), is_border=False
    )

    poses = _straight_proposal(6.0, lateral=lateral)
    borders = [left, right]
    reference = ref.dac_from_road_borders(
        poses, borders, EGO_SHAPE[1], EGO_SHAPE[2], center_offset=EGO_SHAPE[0] / 2.0
    )
    got = _exact_oracle()(
        torch.as_tensor(poses, dtype=torch.float32)[None, None], inputs, _ego_future([poses])
    )
    dac = got[0, 0, ORACLE_METRIC_NAMES.index("drivable_area_compliance")].item()
    assert dac == reference
    assert dac == expected


def test_dac_is_one_without_borders():
    inputs = _empty_inputs(1)
    poses = _straight_proposal(6.0)
    got = _exact_oracle()(
        torch.as_tensor(poses, dtype=torch.float32)[None, None], inputs, _ego_future([poses])
    )
    assert got[0, 0, ORACLE_METRIC_NAMES.index("drivable_area_compliance")].item() == 1.0


# ---------------------------------------------------------------------------
# EP
# ---------------------------------------------------------------------------
def _reference_raw_progress(poses: np.ndarray, reference_path: np.ndarray) -> float:
    """``shapely``-projected arclength gain, the numerator of the reference's EP."""
    from shapely.geometry import LineString, Point

    line = LineString([tuple(p) for p in reference_path[:, :2]])
    start = line.project(Point(poses[0, 0], poses[0, 1]))
    end = line.project(Point(poses[-1, 0], poses[-1, 1]))
    return max(end - start, 0.0)


def test_progress_matches_reference():
    expert = _straight_proposal(8.0)
    candidates = [
        _straight_proposal(8.0),
        _straight_proposal(4.0),
        _straight_proposal(0.0),
        _straight_proposal(12.0),
    ]
    inputs = _empty_inputs(1)
    proposals = torch.as_tensor(np.stack(candidates), dtype=torch.float32)[None]
    got = _exact_oracle()(proposals, inputs, _ego_future([expert]))
    index = ORACLE_METRIC_NAMES.index("ego_progress")

    # DrivoR normalizes each proposal by ``max(its own raw, reference)``.
    # Reproduce that with the reference's own shapely projection so both sides
    # answer the same question.
    reference_path = np.concatenate([np.zeros((1, 4)), expert])
    raw = np.asarray(
        [_reference_raw_progress(candidate, reference_path) for candidate in candidates]
    )
    expert_extent = float(
        np.linalg.norm(np.diff(reference_path[:, :2], axis=0), axis=-1).sum()
    )
    # The invariant that makes the per-proposal and per-set denominators agree
    # here: projecting onto the expert's own path cannot exceed its length, so
    # ``max(raw_i, extent) == extent`` for every proposal.
    assert raw.max() <= expert_extent + 1e-6
    denominator = np.maximum(raw, expert_extent)
    expected = np.clip(raw / denominator, 0.0, 1.0)
    np.testing.assert_allclose(got[0, :, index].numpy(), expected, atol=2e-3)

    # Standing still scores zero and progress is monotone in speed -- but only up
    # to the expert's own extent: a candidate faster than the expert projects
    # past the end of the reference polyline and saturates there, exactly as the
    # shapely reference does (candidates 0 and 3 both sit just under 1.0).
    values = got[0, :, index].tolist()
    assert values[2] == pytest.approx(0.0, abs=1e-3)
    assert values[2] < values[1] < values[0]
    assert values[3] == pytest.approx(values[0], abs=0.05)


def test_progress_gate_for_stationary_expert():
    expert = _straight_proposal(0.0)
    inputs = _empty_inputs(1)
    proposals = torch.as_tensor(np.stack([_straight_proposal(0.0)]), dtype=torch.float32)[None]
    got = _exact_oracle()(proposals, inputs, _ego_future([expert]))
    # navsim's <=5 m branch: the score is 1.0 regardless of the prediction
    assert got[0, 0, ORACLE_METRIC_NAMES.index("ego_progress")].item() == 1.0


# ---------------------------------------------------------------------------
# DDC
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("lateral,expected", [(0.0, 1.0), (10.0, 0.0)])
def test_ddc_on_straight_route(lateral, expected):
    inputs = _empty_inputs(1)
    centre = np.stack(
        [np.linspace(-5.0, 40.0, POINTS_PER_ROUTE), np.zeros(POINTS_PER_ROUTE)], axis=-1
    )
    _add_route(inputs, 0, 0, centre, half_width=2.0)

    poses = _straight_proposal(8.0, lateral=lateral)
    got = _exact_oracle()(
        torch.as_tensor(poses, dtype=torch.float32)[None, None], inputs, _ego_future([poses])
    )
    ddc = got[0, 0, ORACLE_METRIC_NAMES.index("driving_direction_compliance")].item()
    assert ddc == expected

    polygons = ref.route_polygons_from_tensor(inputs["route_lanes"][0].numpy())
    assert ref.ddc_from_route_lanes(poses, polygons, DT) == expected


def test_ddc_is_one_without_route():
    poses = _straight_proposal(8.0, lateral=20.0)
    inputs = _empty_inputs(1)
    got = _exact_oracle()(
        torch.as_tensor(poses, dtype=torch.float32)[None, None], inputs, _ego_future([poses])
    )
    assert got[0, 0, ORACLE_METRIC_NAMES.index("driving_direction_compliance")].item() == 1.0


# ---------------------------------------------------------------------------
# batching / plumbing
# ---------------------------------------------------------------------------
def test_batching_and_chunking_are_equivalent():
    rng = np.random.default_rng(3)
    batch, num = 3, 8
    inputs = _empty_inputs(batch)
    proposals = []
    experts = []
    for b in range(batch):
        span = np.linspace(-5.0, 40.0, POINTS_PER_LINE_STRING)
        _add_line_string(
            inputs, b, 0, np.stack([span, np.full_like(span, 4.0 + b)], axis=-1), True
        )
        _add_route(
            inputs,
            b,
            0,
            np.stack(
                [np.linspace(-5.0, 40.0, POINTS_PER_ROUTE), np.zeros(POINTS_PER_ROUTE)], axis=-1
            ),
            half_width=2.0,
        )
        t = np.arange(1, HORIZON + 1) * DT
        _add_neighbour(
            inputs,
            b,
            0,
            np.stack([10.0 + 2.0 * t, np.full_like(t, 0.2 * b)], axis=-1),
            0.0,
            (1.8, 4.4),
        )
        experts.append(_straight_proposal(7.0))
        proposals.append(
            np.stack(
                [
                    _straight_proposal(
                        float(rng.uniform(0.0, 12.0)),
                        lateral=float(rng.uniform(-3.0, 3.0)),
                        curvature=float(rng.uniform(-0.02, 0.02)),
                    )
                    for _ in range(num)
                ]
            )
        )
    tensor = torch.as_tensor(np.stack(proposals), dtype=torch.float32)
    future = _ego_future(experts)

    full = _exact_oracle()
    chunked = _exact_oracle()
    chunked.proposal_chunk = 3
    torch.testing.assert_close(full(tensor, inputs, future), chunked(tensor, inputs, future))


def test_output_shape_and_ranges():
    batch, num = 2, 5
    inputs = _empty_inputs(batch)
    proposals = torch.as_tensor(
        np.stack(
            [np.stack([_straight_proposal(4.0 + i) for i in range(num)]) for _ in range(batch)]
        ),
        dtype=torch.float32,
    )
    future = _ego_future([_straight_proposal(6.0)] * batch)
    out = _exact_oracle()(proposals, inputs, future)
    assert out.shape == (batch, num, len(ORACLE_METRIC_NAMES))
    assert out.dtype == torch.float32
    for index, name in enumerate(ORACLE_METRIC_NAMES):
        values = out[..., index]
        if name == "time_to_collision_within_bound":
            values = values[values != TTC_UNDEFINED]
        assert values.min() >= 0.0, name
        assert values.max() <= 1.0, name


def test_production_strides_track_the_exact_grid():
    """The shipped strides must not change the labels on ordinary scenes."""
    rng = np.random.default_rng(11)
    batch, num = 2, 12
    inputs = _empty_inputs(batch)
    experts, proposals = [], []
    for b in range(batch):
        span = np.linspace(-5.0, 60.0, POINTS_PER_LINE_STRING)
        _add_line_string(inputs, b, 0, np.stack([span, np.full_like(span, 5.0)], axis=-1), True)
        _add_line_string(inputs, b, 1, np.stack([span, np.full_like(span, -5.0)], axis=-1), True)
        _add_route(
            inputs,
            b,
            0,
            np.stack(
                [np.linspace(-5.0, 60.0, POINTS_PER_ROUTE), np.zeros(POINTS_PER_ROUTE)], axis=-1
            ),
            half_width=2.5,
        )
        t = np.arange(1, HORIZON + 1) * DT
        _add_neighbour(
            inputs, b, 0, np.stack([18.0 + 3.0 * t, np.zeros_like(t)], axis=-1), 0.0, (1.8, 4.4)
        )
        experts.append(_straight_proposal(8.0))
        proposals.append(
            np.stack(
                [
                    _straight_proposal(
                        float(rng.uniform(2.0, 11.0)), lateral=float(rng.uniform(-2.0, 2.0))
                    )
                    for _ in range(num)
                ]
            )
        )
    tensor = torch.as_tensor(np.stack(proposals), dtype=torch.float32)
    future = _ego_future(experts)

    shipped = DrivoROracle(
        dt=DT,
        max_neighbours=NUM_NEIGHBOURS,
        max_border_segments=NUM_LINE_STRINGS * (POINTS_PER_LINE_STRING - 1),
        max_route_segments=NUM_ROUTE * (POINTS_PER_ROUTE - 1),
    )
    torch.testing.assert_close(
        shipped(tensor, inputs, future), _exact_oracle()(tensor, inputs, future)
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
