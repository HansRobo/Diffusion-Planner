# Per-Scene Multi-Human Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the existing multi-human scoring pipeline with a per-scene evaluation that scores 64 DP samples against each validation scene's own human trajectory, ranks scenes by Energy Score percentile, and produces BEV-overlay review reports for the worst disagreements.

**Architecture:** Three-stage pipeline: (1) `score_scenes.py` samples and scores each scene to CSV (GPU), (2) `rank_and_select.py` computes percentile ranks and selects Top5(overall) ∪ Top5(lateral) review candidates (CPU), (3) `render_report.py` produces self-contained HTML with BEV overlays (GPU). Two library modules supply the math: `energy_score.py` (per-scene Energy Score with observation/diversity split) and `route_projection.py` (route stitching, s/d projection, QA).

**Tech Stack:** Python 3.11+, numpy, scipy (cdist), pandas, matplotlib, onnxruntime (existing sampler.py), clip-review-tool (existing BEV rendering)

## Global Constraints

- **sakurab is read-only**: rsync with `--bwlimit=10000` for NPZ fetch only. No GPU, no heavy I/O. $1M training running.
- **Local PC**: GPU free for ONNX. Storage limited — `df -h /home/chenglin/` before any large fetch. System unrecoverable until next week.
- **Core invariant**: Each scene scored against its own human. Never pool trajectories across scenes.
- **Temperature**: 1.0 for all evaluation runs (training-matched distribution, not deployed 0.5).
- **Pre-commit**: ruff lint+format, uv-lock. Run before every commit.
- **Test runner**: `uv run pytest human_match_prototype/tests/ -v`
- **Existing code to reuse**: `sampler.py` (TrajectorySampler), `sidecar.py`, `coord_transform.py`, `build_lanelet_index.py`
- **Existing code to delete** (after new pipeline works): `metrics.py`, `run_all.py`, `analyze.py`, `multi_human_match.py`, `multi_human_report.py`, `cluster_report.py`, `typicality.py` and their tests

---

### Task 1: Energy Score Module

Implement `energy_score.py` — the per-scene Energy Score computation using full-trajectory path norms. This is the foundation metric; everything else builds on it.

**Files:**
- Create: `human_match_prototype/energy_score.py`
- Test: `human_match_prototype/tests/test_energy_score.py`

**Interfaces:**
- Consumes: numpy arrays — `human_xy: (80, 2)`, `samples_xy: (N, 80, 2)`
- Produces: `per_scene_energy_score(human_xy, samples_xy, horizons) -> dict[str, float]` with keys `es_obs_{h}`, `es_div_{h}`, `es_{h}` for h ∈ {2s, 4s, 8s}

- [ ] **Step 1: Write the failing tests**

Create `human_match_prototype/tests/test_energy_score.py`:

```python
import numpy as np
import pytest

from human_match_prototype.energy_score import per_scene_energy_score

HORIZONS = {"2s": 20, "4s": 40, "8s": 80}


def _straight_line(v: float = 5.0, T: int = 80, dt: float = 0.1) -> np.ndarray:
    """(T, 2) straight trajectory at constant speed v m/s along +x."""
    t = np.arange(T) * dt
    return np.stack([v * t, np.zeros(T)], axis=-1)


class TestPerSceneEnergyScore:
    def test_identical_samples_zero_divergence(self):
        """When all samples == human, obs ≈ 0, div ≈ 0, es ≈ 0."""
        human = _straight_line()
        samples = np.tile(human, (64, 1, 1))
        result = per_scene_energy_score(human, samples, HORIZONS)
        for h in HORIZONS:
            assert result[f"es_obs_{h}"] == pytest.approx(0.0, abs=1e-6)
            assert result[f"es_div_{h}"] == pytest.approx(0.0, abs=1e-6)
            assert result[f"es_{h}"] == pytest.approx(0.0, abs=1e-6)

    def test_shifted_samples_positive_obs(self):
        """Samples offset from human should have positive obs term."""
        human = _straight_line()
        samples = np.tile(human, (64, 1, 1))
        samples[:, :, 1] += 2.0  # shift 2m laterally
        result = per_scene_energy_score(human, samples, HORIZONS)
        for h in HORIZONS:
            assert result[f"es_obs_{h}"] > 0.0
            # All samples identical -> div ≈ 0, es ≈ obs
            assert result[f"es_div_{h}"] == pytest.approx(0.0, abs=1e-6)
            assert result[f"es_{h}"] == pytest.approx(result[f"es_obs_{h}"], rel=1e-4)

    def test_diverse_samples_positive_diversity(self):
        """Spread-out samples should have positive div term."""
        rng = np.random.default_rng(42)
        human = _straight_line()
        samples = np.tile(human, (64, 1, 1)) + rng.normal(0, 1.0, (64, 80, 2))
        result = per_scene_energy_score(human, samples, HORIZONS)
        for h in HORIZONS:
            assert result[f"es_div_{h}"] > 0.0

    def test_output_keys_complete(self):
        """All 9 expected keys are present."""
        human = _straight_line()
        samples = np.tile(human, (64, 1, 1))
        result = per_scene_energy_score(human, samples, HORIZONS)
        for h in HORIZONS:
            assert f"es_obs_{h}" in result
            assert f"es_div_{h}" in result
            assert f"es_{h}" in result
        assert len(result) == 9

    def test_longer_horizon_geq_shorter(self):
        """For offset samples, obs at 4s >= obs at 2s (more points, larger norm)."""
        human = _straight_line()
        samples = np.tile(human, (64, 1, 1))
        samples[:, :, 1] += 1.0
        result = per_scene_energy_score(human, samples, HORIZONS)
        assert result["es_obs_4s"] >= result["es_obs_2s"]
        assert result["es_obs_8s"] >= result["es_obs_4s"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest human_match_prototype/tests/test_energy_score.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'human_match_prototype.energy_score'`

- [ ] **Step 3: Implement energy_score.py**

Create `human_match_prototype/energy_score.py`:

```python
"""Per-scene Energy Score: 64 planner samples scored against one human trajectory."""

import numpy as np
from scipy.spatial.distance import cdist

HORIZONS = {"2s": 20, "4s": 40, "8s": 80}


def per_scene_energy_score(
    human_xy: np.ndarray,
    samples_xy: np.ndarray,
    horizons: dict[str, int] = HORIZONS,
) -> dict[str, float]:
    """Compute per-scene Energy Score at multiple horizons.

    Args:
        human_xy: (T, 2) human trajectory [x, y] in ego frame.
        samples_xy: (N, T, 2) planner sample trajectories.
        horizons: mapping of name -> number of timesteps.

    Returns:
        dict with keys es_obs_{h}, es_div_{h}, es_{h} for each horizon h.
        ES_h = obs_h - 0.5 * div_h where:
          obs_h = mean_m ||X_m[:h] - y[:h]||_2  (flattened trajectory norm)
          div_h = mean_{m!=n} ||X_m[:h] - X_n[:h]||_2  (distinct-pair mean)
    """
    N = len(samples_xy)
    out: dict[str, float] = {}

    for name, h in horizons.items():
        y_flat = human_xy[:h].reshape(1, -1)          # (1, h*2)
        x_flat = samples_xy[:, :h].reshape(N, -1)     # (N, h*2)

        obs = float(cdist(x_flat, y_flat).mean())

        pw = cdist(x_flat, x_flat)
        np.fill_diagonal(pw, 0.0)
        div = float(pw.sum() / (N * (N - 1)))

        out[f"es_obs_{name}"] = obs
        out[f"es_div_{name}"] = div
        out[f"es_{name}"] = obs - 0.5 * div

    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest human_match_prototype/tests/test_energy_score.py -v`
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add human_match_prototype/energy_score.py human_match_prototype/tests/test_energy_score.py
git commit -m "feat: add per-scene Energy Score module with observation/diversity split"
```

---

### Task 2: Route Projection Module

Implement `route_projection.py` — route stitching from the NPZ `route_lanes` field, Frenet (s, d) projection, and route QA fields. This is the most complex module.

**Files:**
- Create: `human_match_prototype/route_projection.py`
- Test: `human_match_prototype/tests/test_route_projection.py`

**Interfaces:**
- Consumes: `route_lanes: np.ndarray` shape (25, 20, 33) or (1, 25, 20, 33)
- Produces:
  - `StitchedRoute` dataclass with `centerline: (L, 2)`, `arc_length: (L,)`, `qa: RouteQA`
  - `RouteQA` dataclass with 9 fields (see spec)
  - `stitch_route_lanes(route_lanes) -> StitchedRoute`
  - `project_to_route(route, points) -> (s, d, proj_dist)` where points is `(T, 2)` or `(N, T, 2)`
  - `frenet_energy_scores(human_sd, samples_sd, horizons) -> dict[str, float]` with keys `es_lon_{h}`, `es_lat_{h}`

- [ ] **Step 1: Write the failing tests**

Create `human_match_prototype/tests/test_route_projection.py`:

```python
import numpy as np
import pytest

from human_match_prototype.route_projection import (
    RouteQA,
    StitchedRoute,
    frenet_energy_scores,
    project_to_route,
    stitch_route_lanes,
)


def _make_route_lanes(segments: list[np.ndarray], total_slots: int = 25, pts_per_seg: int = 20, dim: int = 33) -> np.ndarray:
    """Build a (total_slots, pts_per_seg, dim) route_lanes array from a list of (K, 2) xy segments."""
    out = np.zeros((total_slots, pts_per_seg, dim), dtype=np.float32)
    for i, seg_xy in enumerate(segments):
        n = min(len(seg_xy), pts_per_seg)
        out[i, :n, :2] = seg_xy[:n]
    return out


def _straight_segments(n_segs: int = 3, pts: int = 20, spacing: float = 1.0) -> list[np.ndarray]:
    """Create n_segs straight segments along +x with exact endpoint matching."""
    segs = []
    for i in range(n_segs):
        x0 = i * (pts - 1) * spacing
        xs = x0 + np.arange(pts) * spacing
        ys = np.zeros(pts)
        segs.append(np.stack([xs, ys], axis=-1))
    return segs


class TestStitchRouteLanes:
    def test_straight_continuous(self):
        """Three straight segments with exact joins produce a valid route."""
        segs = _straight_segments(3)
        rl = _make_route_lanes(segs)
        route = stitch_route_lanes(rl)
        assert route.qa.route_valid is True
        assert route.qa.n_valid_segments == 3
        assert route.qa.max_segment_gap < 0.01
        assert route.centerline.shape[1] == 2
        assert len(route.centerline) > 0
        # Arc length should be monotonically increasing
        assert np.all(np.diff(route.arc_length) >= 0)

    def test_empty_segments_skipped(self):
        """Segments that are all-zero are ignored."""
        segs = _straight_segments(2)
        rl = _make_route_lanes(segs)
        # Slots 2-24 are already zero
        route = stitch_route_lanes(rl)
        assert route.qa.n_valid_segments == 2
        assert route.qa.route_valid is True

    def test_small_gap_deduplicated(self):
        """Gap <= 0.5m between segments is deduplicated."""
        segs = _straight_segments(2)
        # Shift segment 1 start by 0.3m (below dedup threshold)
        segs[1][:, 0] += 0.3
        rl = _make_route_lanes(segs)
        route = stitch_route_lanes(rl)
        assert route.qa.route_valid is True
        assert route.qa.max_segment_gap < 0.5

    def test_moderate_gap_interpolated(self):
        """Gap between 0.5m and 3.0m is linearly interpolated."""
        segs = _straight_segments(2)
        segs[1][:, 0] += 2.0  # 2m gap
        rl = _make_route_lanes(segs)
        route = stitch_route_lanes(rl)
        assert route.qa.route_valid is True
        assert route.qa.total_interpolated_gap > 1.5

    def test_large_gap_invalid(self):
        """Gap > 3.0m marks route as invalid."""
        segs = _straight_segments(2)
        segs[1][:, 0] += 5.0  # 5m gap
        rl = _make_route_lanes(segs)
        route = stitch_route_lanes(rl)
        assert route.qa.route_valid is False
        assert route.qa.max_segment_gap > 3.0

    def test_batch_dim_squeezed(self):
        """(1, 25, 20, 33) shape is handled by squeezing batch dim."""
        segs = _straight_segments(2)
        rl = _make_route_lanes(segs)
        rl_batched = rl[np.newaxis, ...]  # (1, 25, 20, 33)
        route = stitch_route_lanes(rl_batched)
        assert route.qa.route_valid is True
        assert route.qa.n_valid_segments == 2


class TestProjectToRoute:
    def test_on_route_zero_lateral(self):
        """Points exactly on a straight route have d ≈ 0."""
        segs = _straight_segments(3)
        rl = _make_route_lanes(segs)
        route = stitch_route_lanes(rl)
        # Query points along the route
        points = np.array([[5.0, 0.0], [15.0, 0.0], [30.0, 0.0]])
        s, d, proj_dist = project_to_route(route, points)
        np.testing.assert_allclose(d, 0.0, atol=1e-4)
        np.testing.assert_allclose(proj_dist, 0.0, atol=1e-4)
        # Arc lengths should be increasing
        assert s[0] < s[1] < s[2]

    def test_lateral_offset_sign(self):
        """Point left of route (positive y for +x route) has d > 0."""
        segs = _straight_segments(3)
        rl = _make_route_lanes(segs)
        route = stitch_route_lanes(rl)
        left_point = np.array([[10.0, 2.0]])
        right_point = np.array([[10.0, -2.0]])
        _, d_left, _ = project_to_route(route, left_point)
        _, d_right, _ = project_to_route(route, right_point)
        assert d_left[0] > 0  # left of route
        assert d_right[0] < 0  # right of route

    def test_batch_projection(self):
        """(N, T, 2) input returns (N, T) shaped outputs."""
        segs = _straight_segments(3)
        rl = _make_route_lanes(segs)
        route = stitch_route_lanes(rl)
        points = np.zeros((4, 20, 2))
        points[:, :, 0] = np.linspace(0, 30, 20)
        s, d, proj_dist = project_to_route(route, points)
        assert s.shape == (4, 20)
        assert d.shape == (4, 20)


class TestFrenetEnergyScores:
    def test_identical_projections_zero(self):
        """When all samples have same s,d as human, ES ≈ 0."""
        T = 80
        human_sd = np.stack([np.linspace(0, 50, T), np.zeros(T)], axis=-1)
        samples_sd = np.tile(human_sd, (64, 1, 1))
        result = frenet_energy_scores(human_sd, samples_sd, {"4s": 40})
        assert result["es_lon_4s"] == pytest.approx(0.0, abs=1e-6)
        assert result["es_lat_4s"] == pytest.approx(0.0, abs=1e-6)

    def test_lateral_offset_detected(self):
        """Samples with lateral offset produce positive es_lat."""
        T = 80
        human_sd = np.stack([np.linspace(0, 50, T), np.zeros(T)], axis=-1)
        samples_sd = np.tile(human_sd, (64, 1, 1))
        samples_sd[:, :, 1] += 1.5  # lateral offset
        result = frenet_energy_scores(human_sd, samples_sd, {"4s": 40})
        assert result["es_lat_4s"] > 0
        assert result["es_lon_4s"] == pytest.approx(0.0, abs=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest human_match_prototype/tests/test_route_projection.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement route_projection.py**

Create `human_match_prototype/route_projection.py`:

```python
"""Route stitching, Frenet (s, d) projection, and route QA."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist

GAP_DEDUP_THRESHOLD = 0.5
GAP_INTERPOLATION_THRESHOLD = 3.0
PROJECTION_FAIL_THRESHOLD = 5.0
ROUTE_COVERAGE_THRESHOLD = 0.9
MONOTONICITY_TOLERANCE = 0.1

HORIZONS = {"2s": 20, "4s": 40, "8s": 80}


@dataclass
class RouteQA:
    route_valid: bool
    max_segment_gap: float
    total_interpolated_gap: float
    n_valid_segments: int
    route_arc_length: float
    route_coverage_insufficient: bool
    human_max_proj_dist: float
    n_monotonic_violations: int
    frac_planner_proj_fail: float

    def to_dict(self) -> dict[str, float]:
        return {
            "route_valid": int(self.route_valid),
            "max_segment_gap": self.max_segment_gap,
            "total_interpolated_gap": self.total_interpolated_gap,
            "n_valid_segments": self.n_valid_segments,
            "route_arc_length": self.route_arc_length,
            "route_coverage_insufficient": int(self.route_coverage_insufficient),
            "human_max_proj_dist": self.human_max_proj_dist,
            "n_monotonic_violations": self.n_monotonic_violations,
            "frac_planner_proj_fail": self.frac_planner_proj_fail,
        }


@dataclass
class StitchedRoute:
    centerline: np.ndarray  # (L, 2)
    arc_length: np.ndarray  # (L,)
    qa: RouteQA


def stitch_route_lanes(route_lanes: np.ndarray) -> StitchedRoute:
    """Stitch ordered route_lanes segments into one continuous centerline.

    Args:
        route_lanes: (25, 20, 33) or (1, 25, 20, 33). Features[0:2] are (x, y).
    """
    if route_lanes.ndim == 4:
        route_lanes = route_lanes.squeeze(0)

    segments: list[np.ndarray] = []
    for seg in route_lanes:
        xy = seg[:, :2]
        if np.allclose(xy, 0.0):
            continue
        nonzero_mask = ~np.all(xy == 0.0, axis=-1)
        if not nonzero_mask.any():
            continue
        last_valid = np.where(nonzero_mask)[0][-1]
        segments.append(xy[: last_valid + 1].copy())

    n_valid = len(segments)
    if n_valid == 0:
        empty_qa = RouteQA(
            route_valid=False, max_segment_gap=float("inf"),
            total_interpolated_gap=0.0, n_valid_segments=0,
            route_arc_length=0.0, route_coverage_insufficient=True,
            human_max_proj_dist=float("inf"), n_monotonic_violations=0,
            frac_planner_proj_fail=1.0,
        )
        return StitchedRoute(
            centerline=np.empty((0, 2)),
            arc_length=np.empty((0,)),
            qa=empty_qa,
        )

    max_gap = 0.0
    total_interp = 0.0
    route_valid = True
    points = [segments[0]]

    for i in range(1, n_valid):
        prev_end = segments[i - 1][-1]
        curr_start = segments[i][0]
        gap = float(np.linalg.norm(curr_start - prev_end))
        max_gap = max(max_gap, gap)

        if gap <= GAP_DEDUP_THRESHOLD:
            points.append(segments[i][1:])
        elif gap <= GAP_INTERPOLATION_THRESHOLD:
            midpoint = (prev_end + curr_start) / 2.0
            points.append(midpoint[np.newaxis])
            points.append(segments[i][1:])
            total_interp += gap
        else:
            route_valid = False
            points.append(segments[i])

    centerline = np.concatenate(points, axis=0)
    diffs = np.linalg.norm(np.diff(centerline, axis=0), axis=-1)
    arc_length = np.zeros(len(centerline))
    arc_length[1:] = np.cumsum(diffs)

    qa = RouteQA(
        route_valid=route_valid,
        max_segment_gap=max_gap,
        total_interpolated_gap=total_interp,
        n_valid_segments=n_valid,
        route_arc_length=float(arc_length[-1]) if len(arc_length) > 0 else 0.0,
        route_coverage_insufficient=False,  # set after human projection
        human_max_proj_dist=0.0,
        n_monotonic_violations=0,
        frac_planner_proj_fail=0.0,
    )
    return StitchedRoute(centerline=centerline, arc_length=arc_length, qa=qa)


def _project_points_to_polyline(
    polyline: np.ndarray,
    arc_length: np.ndarray,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project (M, 2) points onto a polyline. Returns s, d, proj_dist each (M,)."""
    n_seg = len(polyline) - 1
    if n_seg < 1:
        M = len(points)
        return np.full(M, np.nan), np.full(M, np.nan), np.full(M, np.inf)

    seg_starts = polyline[:-1]   # (n_seg, 2)
    seg_ends = polyline[1:]      # (n_seg, 2)
    seg_vecs = seg_ends - seg_starts  # (n_seg, 2)
    seg_lens = np.linalg.norm(seg_vecs, axis=-1)  # (n_seg,)
    seg_lens_safe = np.maximum(seg_lens, 1e-10)

    M = len(points)
    s_out = np.empty(M)
    d_out = np.empty(M)
    dist_out = np.empty(M)

    for i in range(M):
        p = points[i]
        dp = p - seg_starts  # (n_seg, 2)
        t = np.sum(dp * seg_vecs, axis=-1) / (seg_lens_safe**2)
        t = np.clip(t, 0.0, 1.0)
        proj = seg_starts + t[:, np.newaxis] * seg_vecs  # (n_seg, 2)
        dists = np.linalg.norm(p - proj, axis=-1)  # (n_seg,)
        best = int(np.argmin(dists))

        s_out[i] = arc_length[best] + t[best] * seg_lens[best]
        dist_out[i] = dists[best]

        # Signed lateral offset: cross product gives sign
        tangent = seg_vecs[best]
        to_point = p - proj[best]
        cross = tangent[0] * to_point[1] - tangent[1] * to_point[0]
        d_out[i] = float(np.sign(cross)) * dists[best]

    return s_out, d_out, dist_out


def project_to_route(
    route: StitchedRoute,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project points onto the stitched route centerline.

    Args:
        route: StitchedRoute from stitch_route_lanes.
        points: (T, 2) single trajectory or (N, T, 2) batch of trajectories.

    Returns:
        (s, d, proj_dist) with shape matching input (T,) or (N, T).
    """
    if points.ndim == 2:
        return _project_points_to_polyline(route.centerline, route.arc_length, points)

    N, T, _ = points.shape
    s_all = np.empty((N, T))
    d_all = np.empty((N, T))
    dist_all = np.empty((N, T))
    for n in range(N):
        s_all[n], d_all[n], dist_all[n] = _project_points_to_polyline(
            route.centerline, route.arc_length, points[n]
        )
    return s_all, d_all, dist_all


def update_qa_after_projection(
    route: StitchedRoute,
    human_s: np.ndarray,
    human_proj_dist: np.ndarray,
    samples_proj_dist: np.ndarray,
) -> None:
    """Update route QA fields after projecting human and samples."""
    qa = route.qa
    qa.human_max_proj_dist = float(np.max(human_proj_dist)) if len(human_proj_dist) > 0 else float("inf")

    mono_violations = np.sum(np.diff(human_s) < -MONOTONICITY_TOLERANCE)
    qa.n_monotonic_violations = int(mono_violations)

    if samples_proj_dist.size > 0:
        qa.frac_planner_proj_fail = float(
            (samples_proj_dist > PROJECTION_FAIL_THRESHOLD).mean()
        )
    else:
        qa.frac_planner_proj_fail = 1.0

    if route.qa.route_arc_length > 0:
        qa.route_coverage_insufficient = (
            float(np.max(human_s)) > route.qa.route_arc_length * ROUTE_COVERAGE_THRESHOLD
        )
    else:
        qa.route_coverage_insufficient = True


def frenet_energy_scores(
    human_sd: np.ndarray,
    samples_sd: np.ndarray,
    horizons: dict[str, int] = HORIZONS,
) -> dict[str, float]:
    """Compute longitudinal and lateral Energy Scores separately.

    Args:
        human_sd: (T, 2) [s, d] of the human trajectory.
        samples_sd: (N, T, 2) [s, d] of planner samples.
        horizons: mapping of name -> number of timesteps.
    """
    N = len(samples_sd)
    out: dict[str, float] = {}

    for name, h in horizons.items():
        for comp_idx, comp_name in [(0, "lon"), (1, "lat")]:
            y = human_sd[:h, comp_idx].reshape(1, -1)        # (1, h)
            x = samples_sd[:, :h, comp_idx].reshape(N, -1)   # (N, h)

            obs = float(cdist(x, y).mean())

            pw = cdist(x, x)
            np.fill_diagonal(pw, 0.0)
            div = float(pw.sum() / (N * (N - 1)))

            out[f"es_{comp_name}_{name}"] = obs - 0.5 * div

    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest human_match_prototype/tests/test_route_projection.py -v`
Expected: all 11 tests PASS

- [ ] **Step 5: Commit**

```bash
git add human_match_prototype/route_projection.py human_match_prototype/tests/test_route_projection.py
git commit -m "feat: add route stitching, Frenet projection, and route QA module"
```

---

### Task 3: Dataset Fetch & Format Verification

Fetch a 500-scene sample of validation NPZs from sakurab, create the smoke-test path list, and verify the route_lanes format matches expectations.

**Files:**
- Create: `human_match_prototype/fetch_validation.py`
- Test: `human_match_prototype/tests/test_fetch_validation.py`

**Interfaces:**
- Consumes: `path_list_valid.json` on sakurab
- Produces:
  - `create_subsample(full_list_path, n, seed, output_path)` — write a JSON path list of n random paths
  - `fetch_npzs_local(path_list, dest, host, bwlimit)` — rsync NPZs to local storage
  - `verify_route_format(npz_path) -> dict` — check route_lanes shape and content
  - Local files: `data/per_scene_eval/path_list_valid_500.json`, NPZs in `data/per_scene_eval/mirror/`

- [ ] **Step 1: Write the failing tests**

Create `human_match_prototype/tests/test_fetch_validation.py`:

```python
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from human_match_prototype.fetch_validation import create_subsample, verify_route_format


class TestCreateSubsample:
    def test_deterministic(self, tmp_path):
        """Same seed produces same subsample."""
        full = [f"/path/to/frame_{i:05d}.npz" for i in range(1000)]
        full_json = tmp_path / "full.json"
        full_json.write_text(json.dumps(full))
        out1 = tmp_path / "sub1.json"
        out2 = tmp_path / "sub2.json"
        create_subsample(str(full_json), 50, seed=42, output_path=str(out1))
        create_subsample(str(full_json), 50, seed=42, output_path=str(out2))
        assert json.loads(out1.read_text()) == json.loads(out2.read_text())

    def test_correct_count(self, tmp_path):
        full = [f"/path/frame_{i}.npz" for i in range(200)]
        full_json = tmp_path / "full.json"
        full_json.write_text(json.dumps(full))
        out = tmp_path / "sub.json"
        create_subsample(str(full_json), 50, seed=0, output_path=str(out))
        result = json.loads(out.read_text())
        assert len(result) == 50


class TestVerifyRouteFormat:
    def test_standard_shape(self, tmp_path):
        npz_path = tmp_path / "test.npz"
        route_lanes = np.zeros((25, 20, 33), dtype=np.float32)
        route_lanes[0, :, 0] = np.linspace(0, 19, 20)  # x values
        np.savez(npz_path, route_lanes=route_lanes, ego_agent_future=np.zeros((80, 3)))
        info = verify_route_format(str(npz_path))
        assert info["shape"] == (25, 20, 33)
        assert info["n_nonempty_segments"] == 1

    def test_batched_shape(self, tmp_path):
        npz_path = tmp_path / "test.npz"
        route_lanes = np.zeros((1, 25, 20, 33), dtype=np.float32)
        route_lanes[0, 0, :, 0] = np.linspace(0, 19, 20)
        np.savez(npz_path, route_lanes=route_lanes, ego_agent_future=np.zeros((80, 3)))
        info = verify_route_format(str(npz_path))
        assert info["shape_raw"] == (1, 25, 20, 33)
        assert info["shape"] == (25, 20, 33)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest human_match_prototype/tests/test_fetch_validation.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement fetch_validation.py**

Create `human_match_prototype/fetch_validation.py`:

```python
"""Fetch validation NPZs from sakurab and verify route format."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


def create_subsample(
    full_list_path: str,
    n: int,
    seed: int,
    output_path: str,
) -> list[str]:
    """Write a JSON path list of n randomly sampled paths."""
    with open(full_list_path) as f:
        full_list = json.load(f)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(full_list), size=min(n, len(full_list)), replace=False)
    subsample = [full_list[i] for i in sorted(indices)]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(subsample, f, indent=2)
    return subsample


def fetch_npzs_local(
    path_list: list[str],
    dest: str,
    host: str = "sakurab",
    bwlimit: int = 10000,
) -> None:
    """rsync NPZ files from host to local dest."""
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    to_fetch = []
    for p in path_list:
        local = dest_path / p.lstrip("/")
        if not local.exists():
            to_fetch.append(p)
    if not to_fetch:
        print(f"All {len(path_list)} NPZs already present locally.")
        return

    listfile = dest_path / ".rsync_fetch_list.txt"
    listfile.write_text("\n".join(p.lstrip("/") for p in to_fetch) + "\n")

    cmd = [
        "rsync", "-a", "--info=progress2",
        f"--bwlimit={bwlimit}",
        f"--files-from={listfile}",
        f"{host}:/",
        str(dest_path),
    ]
    print(f"Fetching {len(to_fetch)} NPZs from {host}...")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"rsync exited with code {result.returncode}")


def verify_route_format(npz_path: str) -> dict:
    """Check route_lanes shape and content from one NPZ."""
    data = np.load(npz_path)
    rl = data["route_lanes"]
    shape_raw = rl.shape
    if rl.ndim == 4:
        rl = rl.squeeze(0)

    n_nonempty = 0
    gaps = []
    prev_end = None
    for seg in rl:
        xy = seg[:, :2]
        if np.allclose(xy, 0.0):
            continue
        nonzero = ~np.all(xy == 0.0, axis=-1)
        if not nonzero.any():
            continue
        last_valid = np.where(nonzero)[0][-1]
        seg_xy = xy[: last_valid + 1]
        if prev_end is not None:
            gap = float(np.linalg.norm(seg_xy[0] - prev_end))
            gaps.append(gap)
        prev_end = seg_xy[-1]
        n_nonempty += 1

    return {
        "shape_raw": shape_raw,
        "shape": rl.shape,
        "n_nonempty_segments": n_nonempty,
        "segment_gaps": gaps,
        "max_gap": max(gaps) if gaps else 0.0,
    }


def main():
    p = argparse.ArgumentParser(description="Fetch and verify validation NPZs.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub_sample = sub.add_parser("subsample", help="Create a random subsample path list.")
    sub_sample.add_argument("--full_list", required=True)
    sub_sample.add_argument("--n", type=int, default=500)
    sub_sample.add_argument("--seed", type=int, default=42)
    sub_sample.add_argument("--output", required=True)

    sub_fetch = sub.add_parser("fetch", help="rsync NPZs from sakurab.")
    sub_fetch.add_argument("--path_list", required=True)
    sub_fetch.add_argument("--dest", default="data/per_scene_eval/mirror")
    sub_fetch.add_argument("--host", default="sakurab")
    sub_fetch.add_argument("--bwlimit", type=int, default=10000)

    sub_verify = sub.add_parser("verify", help="Check route_lanes format on sample NPZs.")
    sub_verify.add_argument("--path_list", required=True)
    sub_verify.add_argument("--mirror", default="data/per_scene_eval/mirror")
    sub_verify.add_argument("--n", type=int, default=10)

    args = p.parse_args()
    if args.cmd == "subsample":
        paths = create_subsample(args.full_list, args.n, args.seed, args.output)
        print(f"Wrote {len(paths)} paths to {args.output}")
    elif args.cmd == "fetch":
        with open(args.path_list) as f:
            paths = json.load(f)
        fetch_npzs_local(paths, args.dest, args.host, args.bwlimit)
    elif args.cmd == "verify":
        with open(args.path_list) as f:
            paths = json.load(f)
        mirror = Path(args.mirror)
        rng = np.random.default_rng(0)
        sample = rng.choice(paths, size=min(args.n, len(paths)), replace=False)
        all_gaps = []
        for p_str in sample:
            local = mirror / p_str.lstrip("/")
            if not local.exists():
                print(f"MISSING: {local}")
                continue
            info = verify_route_format(str(local))
            print(f"{Path(p_str).name}: {info['n_nonempty_segments']} segs, "
                  f"max_gap={info['max_gap']:.3f}m, shape={info['shape']}")
            all_gaps.extend(info["segment_gaps"])
        if all_gaps:
            print(f"\nGap histogram: min={min(all_gaps):.3f} median={np.median(all_gaps):.3f} "
                  f"p95={np.percentile(all_gaps, 95):.3f} max={max(all_gaps):.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest human_match_prototype/tests/test_fetch_validation.py -v`
Expected: all 4 tests PASS

- [ ] **Step 5: Fetch the 500-scene smoke sample**

First copy the full validation path list from sakurab (small JSON, not the NPZs):

```bash
scp sakurab:/mnt/storage_rdma/diffusion_planner/dataset/20260715_basic_dataset/x2_dev/2355_Takanawa_gateway_copied_from_Aisantec/path_list_valid.json \
  data/per_scene_eval/path_list_valid_full.json
```

Then subsample:

```bash
uv run python -m human_match_prototype.fetch_validation subsample \
  --full_list data/per_scene_eval/path_list_valid_full.json \
  --n 500 --seed 42 \
  --output data/per_scene_eval/path_list_valid_500.json
```

Expected: `Wrote 500 paths to data/per_scene_eval/path_list_valid_500.json`

- [ ] **Step 6: Check local disk space and fetch NPZs**

```bash
df -h /home/chenglin/
```

If space is sufficient (need ~50MB for 500 NPZs):

```bash
uv run python -m human_match_prototype.fetch_validation fetch \
  --path_list data/per_scene_eval/path_list_valid_500.json \
  --dest data/per_scene_eval/mirror \
  --host sakurab --bwlimit 10000
```

- [ ] **Step 7: Verify route_lanes format on fetched data**

```bash
uv run python -m human_match_prototype.fetch_validation verify \
  --path_list data/per_scene_eval/path_list_valid_500.json \
  --mirror data/per_scene_eval/mirror --n 10
```

Expected output shows segment counts, gaps, and a histogram. **If the gap histogram p95 > 3.0m, adjust `GAP_INTERPOLATION_THRESHOLD` in `route_projection.py` before continuing.**

- [ ] **Step 8: Commit**

```bash
git add human_match_prototype/fetch_validation.py human_match_prototype/tests/test_fetch_validation.py
git commit -m "feat: add validation data fetch and route format verification"
```

Note: Do NOT commit the path list JSONs or mirror data — they belong in `.gitignore` / `data/`.

---

### Task 4: Score Scenes Pipeline (Stage 1)

Implement `score_scenes.py` — the GPU-intensive Stage 1 that loops over validation scenes, samples DP trajectories, computes Energy Scores and route-relative scores, and writes incremental CSV.

**Files:**
- Create: `human_match_prototype/score_scenes.py`
- Test: `human_match_prototype/tests/test_score_scenes.py`

**Interfaces:**
- Consumes:
  - `TrajectorySampler.sample(npz_path, num_samples, seed, temperature) -> SampleResult` from `sampler.py`
  - `per_scene_energy_score(human_xy, samples_xy, horizons)` from `energy_score.py` (Task 1)
  - `stitch_route_lanes(route_lanes)` from `route_projection.py` (Task 2)
  - `project_to_route(route, points)` from `route_projection.py` (Task 2)
  - `update_qa_after_projection(route, human_s, human_pd, samples_pd)` from `route_projection.py` (Task 2)
  - `frenet_energy_scores(human_sd, samples_sd, horizons)` from `route_projection.py` (Task 2)
- Produces:
  - `score_one_scene(npz_path, sampler, num_samples, seed, temperature) -> dict[str, float]`
  - `scores.csv` with all columns from the output contract

- [ ] **Step 1: Write the failing tests**

Create `human_match_prototype/tests/test_score_scenes.py`:

```python
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from human_match_prototype.score_scenes import score_one_scene

EXPECTED_ES_KEYS = [
    "es_obs_2s", "es_div_2s", "es_2s",
    "es_obs_4s", "es_div_4s", "es_4s",
    "es_obs_8s", "es_div_8s", "es_8s",
]
EXPECTED_FRENET_KEYS = [
    "es_lon_2s", "es_lon_4s", "es_lon_8s",
    "es_lat_2s", "es_lat_4s", "es_lat_8s",
]
EXPECTED_QA_KEYS = [
    "route_valid", "max_segment_gap", "total_interpolated_gap",
    "n_valid_segments", "route_arc_length", "route_coverage_insufficient",
    "human_max_proj_dist", "n_monotonic_violations", "frac_planner_proj_fail",
]


def _make_npz(tmp_path: Path, with_route: bool = True) -> str:
    """Create a minimal NPZ with human trajectory and route_lanes."""
    npz_path = tmp_path / "test_frame.npz"
    T = 80
    human = np.zeros((T, 3), dtype=np.float32)
    human[:, 0] = np.linspace(0, 40, T)  # straight +x

    route_lanes = np.zeros((25, 20, 33), dtype=np.float32)
    if with_route:
        for seg_i in range(3):
            x0 = seg_i * 19.0
            route_lanes[seg_i, :, 0] = np.linspace(x0, x0 + 19, 20)

    np.savez(npz_path, ego_agent_future=human, route_lanes=route_lanes)
    return str(npz_path)


def _mock_sampler(npz_path: str, num_samples: int = 64, seed: int = 0, temperature: float = 1.0):
    """Return a SampleResult-like object with samples near the human."""
    data = np.load(npz_path)
    human = data["ego_agent_future"][:, :3].astype(np.float32)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.5, (num_samples, 80, 3)).astype(np.float32)
    samples = np.tile(human, (num_samples, 1, 1)) + noise
    result = MagicMock()
    result.ego_samples = samples
    result.human_future = human
    return result


class TestScoreOneScene:
    def test_all_keys_present(self, tmp_path):
        npz = _make_npz(tmp_path, with_route=True)
        sampler = MagicMock()
        sampler.sample.side_effect = lambda *a, **kw: _mock_sampler(npz, **kw)
        row = score_one_scene(npz, sampler, num_samples=64, seed=0, temperature=1.0)
        assert "npz_path" in row
        for k in EXPECTED_ES_KEYS:
            assert k in row, f"missing {k}"
        for k in EXPECTED_QA_KEYS:
            assert k in row, f"missing {k}"

    def test_valid_route_has_frenet(self, tmp_path):
        npz = _make_npz(tmp_path, with_route=True)
        sampler = MagicMock()
        sampler.sample.side_effect = lambda *a, **kw: _mock_sampler(npz, **kw)
        row = score_one_scene(npz, sampler, num_samples=64, seed=0, temperature=1.0)
        if row["route_valid"]:
            for k in EXPECTED_FRENET_KEYS:
                assert not np.isnan(row[k]), f"{k} should not be NaN with valid route"

    def test_no_route_frenet_nan(self, tmp_path):
        npz = _make_npz(tmp_path, with_route=False)
        sampler = MagicMock()
        sampler.sample.side_effect = lambda *a, **kw: _mock_sampler(npz, **kw)
        row = score_one_scene(npz, sampler, num_samples=64, seed=0, temperature=1.0)
        for k in EXPECTED_FRENET_KEYS:
            assert np.isnan(row[k]), f"{k} should be NaN with no route"

    def test_es_values_reasonable(self, tmp_path):
        npz = _make_npz(tmp_path, with_route=True)
        sampler = MagicMock()
        sampler.sample.side_effect = lambda *a, **kw: _mock_sampler(npz, **kw)
        row = score_one_scene(npz, sampler, num_samples=64, seed=0, temperature=1.0)
        # With small noise, obs should be positive and finite
        for h in ["2s", "4s", "8s"]:
            assert 0 < row[f"es_obs_{h}"] < 1000
            assert row[f"es_div_{h}"] >= 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest human_match_prototype/tests/test_score_scenes.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement score_scenes.py**

Create `human_match_prototype/score_scenes.py`:

```python
"""Stage 1: Sample DP trajectories and score each scene. Writes incremental CSV."""

import argparse
import csv
import json
import traceback
from pathlib import Path

import numpy as np
from tqdm import tqdm

from human_match_prototype.energy_score import HORIZONS, per_scene_energy_score
from human_match_prototype.route_projection import (
    frenet_energy_scores,
    project_to_route,
    stitch_route_lanes,
    update_qa_after_projection,
)
from human_match_prototype.sampler import TrajectorySampler

DEFAULT_MODEL_DIR = Path("/opt/autoware/mlmodels/diffusion_planner_for_x2")

NAN_FRENET = {f"es_{c}_{h}": float("nan") for h in HORIZONS for c in ("lon", "lat")}


def score_one_scene(
    npz_path: str,
    sampler: TrajectorySampler,
    num_samples: int = 64,
    seed: int = 0,
    temperature: float = 1.0,
) -> dict[str, float]:
    """Score a single scene: Energy Score + route Frenet + QA."""
    result = sampler.sample(npz_path, num_samples=num_samples, seed=seed, temperature=temperature)
    human_xy = result.human_future[:, :2]    # (80, 2)
    samples_xy = result.ego_samples[:, :, :2]  # (N, 80, 2)

    row: dict[str, float] = {"npz_path": npz_path}

    # x-y Energy Score
    row.update(per_scene_energy_score(human_xy, samples_xy))

    # Route projection
    data = np.load(npz_path)
    route_lanes = data["route_lanes"]
    route = stitch_route_lanes(route_lanes)

    if not route.qa.route_valid or len(route.centerline) < 2:
        row.update(NAN_FRENET)
        row.update(route.qa.to_dict())
        return row

    human_s, human_d, human_pd = project_to_route(route, human_xy)
    samples_s, samples_d, samples_pd = project_to_route(route, samples_xy)
    update_qa_after_projection(route, human_s, human_pd, samples_pd)

    if route.qa.route_coverage_insufficient:
        row.update(NAN_FRENET)
        row.update(route.qa.to_dict())
        return row

    human_sd = np.stack([human_s, human_d], axis=-1)      # (80, 2)
    samples_sd = np.stack([samples_s, samples_d], axis=-1)  # (N, 80, 2)
    row.update(frenet_energy_scores(human_sd, samples_sd))
    row.update(route.qa.to_dict())

    return row


def main():
    p = argparse.ArgumentParser(description="Stage 1: Score validation scenes.")
    p.add_argument("--npz_list", required=True, help="JSON path list of NPZs")
    p.add_argument("--output", required=True, help="Output CSV path")
    p.add_argument("--num_samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model_dir", type=Path, default=None)
    p.add_argument("--mirror", default=None, help="Local mirror root; paths are resolved relative to this.")
    args = p.parse_args()

    model_dir = args.model_dir or DEFAULT_MODEL_DIR
    sampler = TrajectorySampler(
        str(model_dir / "args.json"),
        str(model_dir / "diffusion_planner.onnx"),
        args.device,
    )

    with open(args.npz_list) as f:
        paths = json.load(f)
    if args.limit:
        paths = paths[: args.limit]

    if args.mirror:
        mirror = Path(args.mirror)
        paths = [str(mirror / p.lstrip("/")) for p in paths]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = None
    skipped = 0
    with open(out_path, "w", newline="") as csvfile:
        writer = None
        for path in tqdm(paths, desc="Scoring"):
            try:
                row = score_one_scene(path, sampler, args.num_samples, args.seed, args.temperature)
                if writer is None:
                    fieldnames = list(row.keys())
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                writer.writerow(row)
                csvfile.flush()
            except Exception:
                skipped += 1
                print(f"skip {path}")
                traceback.print_exc()

    print(f"Wrote {len(paths) - skipped} rows to {out_path} ({skipped} skipped)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest human_match_prototype/tests/test_score_scenes.py -v`
Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add human_match_prototype/score_scenes.py human_match_prototype/tests/test_score_scenes.py
git commit -m "feat: add Stage 1 score_scenes pipeline with incremental CSV output"
```

---

### Task 5: Rank and Select Pipeline (Stage 2)

Implement `rank_and_select.py` — CPU-only Stage 2 that computes percentile ranks, combined rankings, and selects the review set.

**Files:**
- Create: `human_match_prototype/rank_and_select.py`
- Test: `human_match_prototype/tests/test_rank_and_select.py`

**Interfaces:**
- Consumes: `scores.csv` from Stage 1
- Produces:
  - `compute_ranks(df) -> pd.DataFrame` with percentile and combined rank columns
  - `select_review_set(ranked_df, top_k) -> pd.DataFrame` with selection reason
  - CLI writes `ranked.csv`, `review_set.csv`, `distributions.png`

- [ ] **Step 1: Write the failing tests**

Create `human_match_prototype/tests/test_rank_and_select.py`:

```python
import numpy as np
import pandas as pd
import pytest

from human_match_prototype.rank_and_select import compute_ranks, select_review_set


def _fake_scores(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate a plausible scores DataFrame for testing."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        row = {"npz_path": f"/path/frame_{i:05d}.npz"}
        for h in ["2s", "4s", "8s"]:
            row[f"es_obs_{h}"] = rng.exponential(5.0)
            row[f"es_div_{h}"] = rng.exponential(2.0)
            row[f"es_{h}"] = row[f"es_obs_{h}"] - 0.5 * row[f"es_div_{h}"]
        has_frenet = rng.random() > 0.1  # 90% have valid route
        for h in ["2s", "4s", "8s"]:
            row[f"es_lon_{h}"] = rng.exponential(3.0) if has_frenet else float("nan")
            row[f"es_lat_{h}"] = rng.exponential(2.0) if has_frenet else float("nan")
        row["route_valid"] = int(has_frenet)
        rows.append(row)
    return pd.DataFrame(rows)


class TestComputeRanks:
    def test_percentile_range(self):
        df = _fake_scores(100)
        ranked = compute_ranks(df)
        assert "pct_es_2s" in ranked.columns
        assert ranked["pct_es_2s"].min() >= 0
        assert ranked["pct_es_2s"].max() <= 100

    def test_combined_ranks_present(self):
        df = _fake_scores(100)
        ranked = compute_ranks(df)
        assert "R_overall" in ranked.columns
        assert "R_lateral" in ranked.columns
        assert "R_longitudinal" in ranked.columns

    def test_nan_frenet_handled(self):
        """Scenes with NaN Frenet should have NaN R_lateral."""
        df = _fake_scores(100)
        ranked = compute_ranks(df)
        nan_route = ranked[ranked["route_valid"] == 0]
        if len(nan_route) > 0:
            assert nan_route["R_lateral"].isna().all()


class TestSelectReviewSet:
    def test_returns_5_to_10(self):
        df = _fake_scores(100)
        ranked = compute_ranks(df)
        review = select_review_set(ranked, top_k=5)
        assert 5 <= len(review) <= 10

    def test_includes_top_overall(self):
        df = _fake_scores(100)
        ranked = compute_ranks(df)
        review = select_review_set(ranked, top_k=5)
        top5_overall = ranked.nlargest(5, "R_overall")["npz_path"].tolist()
        for p in top5_overall:
            assert p in review["npz_path"].values

    def test_selection_reason_populated(self):
        df = _fake_scores(100)
        ranked = compute_ranks(df)
        review = select_review_set(ranked, top_k=5)
        assert "selection_reason" in review.columns
        assert review["selection_reason"].notna().all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest human_match_prototype/tests/test_rank_and_select.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement rank_and_select.py**

Create `human_match_prototype/rank_and_select.py`:

```python
"""Stage 2: Compute percentile ranks and select review candidates."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from human_match_prototype.energy_score import HORIZONS

ES_SCORE_COLS = [f"es_{h}" for h in HORIZONS]
LAT_SCORE_COLS = [f"es_lat_{h}" for h in HORIZONS]
LON_SCORE_COLS = [f"es_lon_{h}" for h in HORIZONS]


def compute_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Add percentile rank columns and combined rankings."""
    ranked = df.copy()

    for col in ES_SCORE_COLS + LAT_SCORE_COLS + LON_SCORE_COLS:
        pct_col = f"pct_{col}"
        ranked[pct_col] = ranked[col].rank(pct=True, na_option="keep") * 100

    ranked["R_overall"] = ranked[[f"pct_es_{h}" for h in HORIZONS]].mean(axis=1)

    lat_pcts = ranked[[f"pct_es_lat_{h}" for h in HORIZONS]]
    ranked["R_lateral"] = lat_pcts.mean(axis=1)  # NaN if any horizon is NaN

    lon_pcts = ranked[[f"pct_es_lon_{h}" for h in HORIZONS]]
    ranked["R_longitudinal"] = lon_pcts.mean(axis=1)

    # Label strongest contributing horizon for overall
    horizon_pcts = ranked[[f"pct_es_{h}" for h in HORIZONS]]
    ranked["strongest_overall_horizon"] = horizon_pcts.idxmax(axis=1).str.replace("pct_es_", "")

    return ranked


def select_review_set(ranked: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """Select Top-K(R_overall) ∪ Top-K(R_lateral), deduplicated."""
    top_overall = ranked.nlargest(top_k, "R_overall")
    top_overall = top_overall.assign(selection_reason="top_overall")

    lat_valid = ranked.dropna(subset=["R_lateral"])
    if len(lat_valid) >= top_k:
        top_lateral = lat_valid.nlargest(top_k, "R_lateral")
    else:
        top_lateral = lat_valid
    top_lateral = top_lateral.assign(selection_reason="top_lateral")

    combined = pd.concat([top_overall, top_lateral])
    # For duplicates, combine reasons
    deduped = combined.groupby("npz_path", sort=False).agg(
        {**{c: "first" for c in combined.columns if c not in ("npz_path", "selection_reason")},
         "selection_reason": lambda x: "+".join(sorted(set(x)))}
    ).reset_index()

    return deduped.sort_values("R_overall", ascending=False).reset_index(drop=True)


def plot_distributions(ranked: pd.DataFrame, output_path: str) -> None:
    """Plot score distributions for overall, longitudinal, and lateral ES."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.patch.set_facecolor("#fcfcfb")

    for i, h in enumerate(HORIZONS):
        ax = axes[0, i]
        ax.set_facecolor("#fcfcfb")
        col = f"es_{h}"
        vals = ranked[col].dropna()
        ax.hist(vals, bins=50, color="#2a78d6", edgecolor="none", alpha=0.85)
        ax.set_title(f"Overall ES ({h})", fontsize=11)
        ax.set_xlabel("Energy Score")

        ax = axes[1, i]
        ax.set_facecolor("#fcfcfb")
        col = f"es_lat_{h}"
        vals = ranked[col].dropna()
        ax.hist(vals, bins=50, color="#d03b3b", edgecolor="none", alpha=0.85)
        ax.set_title(f"Lateral ES ({h})", fontsize=11)
        ax.set_xlabel("Energy Score")

    for ax in axes.flat:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=130, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Stage 2: Rank scenes and select review set.")
    p.add_argument("--scores", required=True, help="scores.csv from Stage 1")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--top_k", type=int, default=5)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.scores)
    ranked = compute_ranks(df)
    ranked.to_csv(out / "ranked.csv", index=False)

    review = select_review_set(ranked, args.top_k)
    review.to_csv(out / "review_set.csv", index=False)

    plot_distributions(ranked, str(out / "distributions.png"))

    print(f"Ranked {len(ranked)} scenes -> {out / 'ranked.csv'}")
    print(f"Selected {len(review)} review candidates -> {out / 'review_set.csv'}")
    print(f"  Overall top-{args.top_k}: {ranked.nlargest(args.top_k, 'R_overall')['npz_path'].tolist()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest human_match_prototype/tests/test_rank_and_select.py -v`
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add human_match_prototype/rank_and_select.py human_match_prototype/tests/test_rank_and_select.py
git commit -m "feat: add Stage 2 percentile ranking and review-set selection"
```

---

### Task 6: Report Rendering (Stage 3)

Implement `render_report.py` — renders BEV overlays with route centerline for the review set and produces a self-contained HTML report.

**Files:**
- Create: `human_match_prototype/render_report.py`
- Test: `human_match_prototype/tests/test_render_report.py`

**Interfaces:**
- Consumes:
  - `review_set.csv` from Stage 2
  - `ranked.csv` from Stage 2
  - `TrajectorySampler.sample()` from `sampler.py`
  - `stitch_route_lanes()` from `route_projection.py`
  - clip-review-tool's `render_frame`, `precompute_static`, `PAST_FRAMES`
- Produces:
  - `render_scene_overlay(sampler, npz_path, route, out_png, ...)` — BEV overlay with route
  - `render_html_report(review_df, ranked_df, overlay_pngs, dist_png, out_html, metadata)` — self-contained HTML
  - CLI writes `report.html`

- [ ] **Step 1: Write the failing tests**

Create `human_match_prototype/tests/test_render_report.py`:

```python
import base64
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from human_match_prototype.render_report import render_html_report


def _fake_review_set(n: int = 6) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "npz_path": f"/path/frame_{i:05d}.npz",
            "R_overall": 95.0 - i,
            "R_lateral": 90.0 - i * 2,
            "selection_reason": "top_overall" if i < 3 else "top_lateral",
            "es_2s": 10.0 + i,
            "es_4s": 15.0 + i,
            "es_8s": 20.0 + i,
        })
    return pd.DataFrame(rows)


class TestRenderHtmlReport:
    def test_produces_html(self, tmp_path):
        review = _fake_review_set()
        ranked = _fake_review_set(20)
        # Create dummy overlay PNGs
        overlay_pngs = []
        for i in range(len(review)):
            png = tmp_path / f"overlay_{i}.png"
            png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
            overlay_pngs.append(png)
        dist_png = tmp_path / "distributions.png"
        dist_png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        out = tmp_path / "report.html"
        render_html_report(review, ranked, overlay_pngs, dist_png, out, {
            "temperature": 1.0, "seed": 0, "num_samples": 64, "n_scenes": 500,
        })
        assert out.exists()
        html = out.read_text()
        assert "review candidate" in html.lower() or "Review" in html
        assert "base64" in html  # images embedded
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest human_match_prototype/tests/test_render_report.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement render_report.py**

Create `human_match_prototype/render_report.py`:

```python
"""Stage 3: Render BEV overlays and self-contained HTML report."""

import argparse
import base64
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from human_match_prototype.route_projection import stitch_route_lanes
from human_match_prototype.sampler import TrajectorySampler

DEFAULT_MODEL_DIR = Path("/opt/autoware/mlmodels/diffusion_planner_for_x2")


def _b64(png_path: Path) -> str:
    return base64.b64encode(png_path.read_bytes()).decode()


def render_scene_overlay(
    sampler: TrajectorySampler,
    npz_path: str,
    out_png: Path,
    num_samples: int = 64,
    seed: int = 0,
    temperature: float = 1.0,
) -> None:
    """Render BEV overlay with route centerline and planner samples."""
    try:
        from src.visualization import PAST_FRAMES, precompute_static, render_frame
    except ImportError:
        raise ImportError(
            "clip-review-tool is required for BEV overlays. "
            "Install with: uv pip install -e ../clip-review-tool"
        )

    data = dict(np.load(npz_path, allow_pickle=True))
    r = sampler.sample(str(npz_path), num_samples=num_samples, seed=seed, temperature=temperature)

    fig, ax = plt.subplots(figsize=(14, 14))
    static = precompute_static(data)
    static["view_half"] = static["view_half"] * 0.7
    render_frame(fig, ax, data, static, t=PAST_FRAMES - 1, filename=Path(npz_path).name)

    # Planner samples
    for s in r.ego_samples:
        ax.plot(s[:, 0], s[:, 1], color="#E040FB", alpha=0.3, lw=0.5, zorder=40)

    # Human trajectory
    human = r.human_future[:, :2]
    ax.plot(human[:, 0], human[:, 1], color="#00E676", lw=2.5, zorder=45, label="Human")

    # Route centerline
    route_lanes = data["route_lanes"]
    route = stitch_route_lanes(np.asarray(route_lanes))
    if route.qa.route_valid and len(route.centerline) > 1:
        ax.plot(
            route.centerline[:, 0], route.centerline[:, 1],
            color="#00BCD4", lw=1.5, ls="--", alpha=0.7, zorder=38, label="Route",
        )

    ax.legend(loc="upper right", fontsize=9, framealpha=0.7)
    fig.savefig(out_png, dpi=120, facecolor="#1A1A1A", bbox_inches="tight")
    plt.close(fig)


def render_html_report(
    review_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
    overlay_pngs: list[Path],
    dist_png: Path,
    out_html: Path,
    metadata: dict,
) -> None:
    """Generate self-contained HTML report with embedded images."""
    n_total = len(ranked_df)
    n_route_valid = int(ranked_df["route_valid"].sum()) if "route_valid" in ranked_df.columns else "N/A"

    # Build review table
    review_cols = [c for c in ["npz_path", "R_overall", "R_lateral", "selection_reason",
                                "es_2s", "es_4s", "es_8s", "es_lat_2s", "es_lat_4s", "es_lat_8s",
                                "route_valid"] if c in review_df.columns]
    head = "".join(f"<th>{c}</th>" for c in review_cols)
    body = ""
    for _, row in review_df.iterrows():
        tds = "".join(
            f"<td>{row[c]:.3f}</td>" if isinstance(row.get(c), float) and not np.isnan(row.get(c, float("nan")))
            else f"<td>{row.get(c, '')}</td>"
            for c in review_cols
        )
        body += f"<tr>{tds}</tr>\n"

    overlays_html = "\n".join(
        f'<figure><img src="data:image/png;base64,{_b64(p)}" style="max-width:700px">'
        f"<figcaption>{p.stem}</figcaption></figure>"
        for p in overlay_pngs if p.exists()
    )

    dist_img = f'<img src="data:image/png;base64,{_b64(dist_png)}" style="max-width:100%">' if dist_png.exists() else ""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Per-Scene Evaluation Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 2rem; max-width: 1200px; color: #1a1a1a; }}
h1 {{ border-bottom: 2px solid #2a78d6; padding-bottom: 0.5em; }}
table {{ border-collapse: collapse; font-size: 13px; margin: 1em 0; }}
th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: right; }}
th {{ background: #f5f5f5; text-align: center; }}
.meta {{ background: #f0f4ff; padding: 1em; border-radius: 6px; margin: 1em 0; font-size: 13px; }}
figure {{ display: inline-block; margin: 8px; }}
figcaption {{ font-size: 12px; text-align: center; color: #666; }}
.warning {{ color: #d03b3b; font-weight: 600; }}
</style></head><body>
<h1>Per-Scene Evaluation Report</h1>
<div class="meta">
<strong>Temperature:</strong> {metadata.get("temperature", "?")}&emsp;
<strong>Seed:</strong> {metadata.get("seed", "?")}&emsp;
<strong>Samples:</strong> {metadata.get("num_samples", "?")}&emsp;
<strong>Scenes scored:</strong> {n_total}&emsp;
<strong>Valid routes:</strong> {n_route_valid}&emsp;
<p class="warning">Interpretation: Top-ranked scenes are highest-disagreement review candidates, not automatic planner failures.
T=1.0 evaluates the training-matched distribution, not deployed T=0.5.</p>
</div>

<h2>Score Distributions</h2>
{dist_img}

<h2>Review Candidates ({len(review_df)} scenes)</h2>
<p>Selection: Top5(R_overall) &cup; Top5(R_lateral), deduplicated.</p>
<table><tr>{head}</tr>{body}</table>

<h2>BEV Overlays</h2>
<p>Magenta: planner samples. Green: human trajectory. Cyan dashed: route centerline.</p>
{overlays_html}

</body></html>"""
    out_html.write_text(html)


def main():
    p = argparse.ArgumentParser(description="Stage 3: Render report with BEV overlays.")
    p.add_argument("--review_set", required=True)
    p.add_argument("--scores", required=True, help="ranked.csv from Stage 2")
    p.add_argument("--output", required=True, help="Output HTML path")
    p.add_argument("--num_samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model_dir", type=Path, default=None)
    args = p.parse_args()

    model_dir = args.model_dir or DEFAULT_MODEL_DIR
    sampler = TrajectorySampler(
        str(model_dir / "args.json"),
        str(model_dir / "diffusion_planner.onnx"),
        args.device,
    )

    review_df = pd.read_csv(args.review_set)
    ranked_df = pd.read_csv(args.scores)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_path.parent / "overlays"
    overlay_dir.mkdir(exist_ok=True)

    overlay_pngs = []
    for _, row in review_df.iterrows():
        npz = row["npz_path"]
        png = overlay_dir / f"{Path(npz).stem}.png"
        print(f"Rendering {Path(npz).name}...")
        render_scene_overlay(sampler, npz, png, args.num_samples, args.seed, args.temperature)
        overlay_pngs.append(png)

    dist_png = out_path.parent / "distributions.png"

    render_html_report(
        review_df, ranked_df, overlay_pngs, dist_png, out_path,
        {"temperature": args.temperature, "seed": args.seed,
         "num_samples": args.num_samples, "n_scenes": len(ranked_df)},
    )
    print(f"Report: {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest human_match_prototype/tests/test_render_report.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add human_match_prototype/render_report.py human_match_prototype/tests/test_render_report.py
git commit -m "feat: add Stage 3 BEV overlay rendering and HTML report"
```

---

### Task 7: Remove Superseded Code & Run Full Test Suite

Delete the old pipeline files that the new per-scene evaluation replaces, update any imports, and verify the full test suite passes.

**Files:**
- Delete: `human_match_prototype/metrics.py`, `human_match_prototype/run_all.py`, `human_match_prototype/analyze.py`, `human_match_prototype/multi_human_match.py`, `human_match_prototype/multi_human_report.py`, `human_match_prototype/cluster_report.py`, `human_match_prototype/typicality.py`, `human_match_prototype/fetch_cluster_samples.py`
- Delete: `human_match_prototype/tests/test_metrics.py`, `human_match_prototype/tests/test_multi_human_match.py`, `human_match_prototype/tests/test_multi_human_report.py`, `human_match_prototype/tests/test_cluster_report.py`, `human_match_prototype/tests/test_typicality.py`, `human_match_prototype/tests/test_fetch_cluster_samples.py`
- Keep: `human_match_prototype/tests/test_bev_overlay.py`, `human_match_prototype/tests/test_coord_transform.py`, `human_match_prototype/tests/test_build_lanelet_index.py`

- [ ] **Step 1: Delete superseded files**

```bash
git rm human_match_prototype/metrics.py \
       human_match_prototype/run_all.py \
       human_match_prototype/analyze.py \
       human_match_prototype/multi_human_match.py \
       human_match_prototype/multi_human_report.py \
       human_match_prototype/cluster_report.py \
       human_match_prototype/typicality.py \
       human_match_prototype/fetch_cluster_samples.py

git rm human_match_prototype/tests/test_metrics.py \
       human_match_prototype/tests/test_multi_human_match.py \
       human_match_prototype/tests/test_multi_human_report.py \
       human_match_prototype/tests/test_cluster_report.py \
       human_match_prototype/tests/test_typicality.py \
       human_match_prototype/tests/test_fetch_cluster_samples.py
```

- [ ] **Step 2: Check for broken imports in kept files**

```bash
uv run python -c "from human_match_prototype.sampler import TrajectorySampler; print('sampler OK')"
uv run python -c "from human_match_prototype.sidecar import read_sidecar; print('sidecar OK')"
uv run python -c "from human_match_prototype.coord_transform import WorldPose; print('coord OK')"
uv run python -c "from human_match_prototype.energy_score import per_scene_energy_score; print('es OK')"
uv run python -c "from human_match_prototype.route_projection import stitch_route_lanes; print('route OK')"
uv run python -c "from human_match_prototype.score_scenes import score_one_scene; print('score OK')"
uv run python -c "from human_match_prototype.rank_and_select import compute_ranks; print('rank OK')"
```

Expected: all print "OK"

- [ ] **Step 3: Run full test suite**

```bash
uv run pytest human_match_prototype/tests/ -v
```

Expected: all tests PASS. If `test_bev_overlay.py` has import issues from the deleted `analyze.py`, update its import to use `render_report.render_scene_overlay` instead.

- [ ] **Step 4: Commit**

```bash
git add -A human_match_prototype/
git commit -m "refactor: remove superseded multi-human pipeline, keep per-scene evaluation"
```

---

### Task 8: Smoke Test on Real Validation Data

Run the full three-stage pipeline on the 500-scene smoke sample to validate end-to-end correctness on real data. This requires the NPZs to have been fetched in Task 3.

**Files:**
- No new code files. This task validates the pipeline on real data.

**Interfaces:**
- Consumes: all three stages from Tasks 4-6, NPZs from Task 3
- Produces: `data/per_scene_eval/results/report.html` — the final artifact

- [ ] **Step 1: Run Stage 1 — score 500 scenes**

```bash
uv run python -m human_match_prototype.score_scenes \
  --npz_list data/per_scene_eval/path_list_valid_500.json \
  --output data/per_scene_eval/scores.csv \
  --num_samples 64 --seed 0 --temperature 1.0 \
  --mirror data/per_scene_eval/mirror \
  --device cuda
```

Expected: `Wrote ~500 rows to data/per_scene_eval/scores.csv` (some may skip). Runtime: ~30-60 min on GPU depending on batch speed.

Spot-check: open `scores.csv`, verify columns are present, es values are finite, some route_valid=1 rows have non-NaN Frenet scores.

- [ ] **Step 2: Run Stage 2 — rank and select**

```bash
uv run python -m human_match_prototype.rank_and_select \
  --scores data/per_scene_eval/scores.csv \
  --output_dir data/per_scene_eval/results/ \
  --top_k 5
```

Expected: prints ranked count and 5-10 review candidates. Check `results/review_set.csv` has `selection_reason` values. Check `results/distributions.png` is non-empty.

- [ ] **Step 3: Run Stage 3 — render report**

```bash
uv run python -m human_match_prototype.render_report \
  --review_set data/per_scene_eval/results/review_set.csv \
  --scores data/per_scene_eval/results/ranked.csv \
  --output data/per_scene_eval/results/report.html \
  --seed 0 --temperature 1.0
```

Expected: produces `report.html` with BEV overlays. Open in browser to verify:
- Score distributions look reasonable (not all zeros, not all identical)
- BEV overlays show planner samples (magenta), human (green), route (cyan dashed)
- Route QA fields are populated
- Review table has the selected scenes with scores

- [ ] **Step 4: Inspect route QA statistics**

```bash
uv run python -c "
import pandas as pd
df = pd.read_csv('data/per_scene_eval/scores.csv')
print('Route valid:', df['route_valid'].mean())
print('Route coverage insufficient:', df['route_coverage_insufficient'].mean())
print('Max segment gap distribution:')
print(df['max_segment_gap'].describe())
print('Frac planner proj fail:')
print(df['frac_planner_proj_fail'].describe())
"
```

If `route_valid` rate is very low (< 50%), revisit `GAP_INTERPOLATION_THRESHOLD` in `route_projection.py`.

- [ ] **Step 5: Commit results summary (not data)**

```bash
git add -A human_match_prototype/  # only code changes if any fixes were needed
git commit -m "fix: adjustments from smoke test on real validation data"
```

Do NOT commit the CSV files, NPZs, or HTML report to git.
