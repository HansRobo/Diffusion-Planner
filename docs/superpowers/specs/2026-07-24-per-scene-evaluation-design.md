# Per-Scene Multi-Human Evaluation — Design Spec

**Date:** 2026-07-24
**Brainstorm source:** `_bmad-output/brainstorming/brainstorm-per-scene-multi-human-evaluation-2026-07-24/.memlog.md`
**Status:** Draft

## Goal

Find the locations and scenario types where Diffusion Planner disagrees most with actual human driving, using held-out validation data scored per-scene (never pooling human trajectories across scenes).

The evaluation unit is: **one validation scene → 64 DP samples at T=1.0 → scored against that scene's own human trajectory → aggregated across scenes by percentile rank**.

## Core Invariant

Each human outcome is scored only against DP samples generated from that exact scene's conditions. Other scenes contribute only by aggregating their independently computed scores. Humans from different drives at the same map location are never pooled into a shared target.

## Approach: Three-Stage Pipeline

Replace the existing scoring/analysis pipeline (`metrics.py`, `run_all.py`, `analyze.py`, `multi_human_match.py`, `multi_human_report.py`, `cluster_report.py`) with a three-stage pipeline. Shared utilities (`sampler.py`, `sidecar.py`, `coord_transform.py`, `build_lanelet_index.py`) remain.

### Stage 1: `score_scenes.py` — Sample & Score (GPU)

```bash
python -m human_match_prototype.score_scenes \
  --npz_list data/per_scene_eval/path_list_valid_500.json \
  --output data/per_scene_eval/scores.csv \
  --num_samples 64 --seed 0 --temperature 1.0 \
  [--limit 500]
```

Per scene:
1. Load NPZ locally (never from sakurab at runtime)
2. Sample 64 DP trajectories at temperature 1.0 via `TrajectorySampler`
3. Extract human trajectory: `ego_agent_future[:, :3]` → (80, 3) [x, y, yaw]
4. Compute x-y Energy Score at three horizons
5. Stitch route and compute s/d projection + QA
6. If route valid and coverage sufficient: compute longitudinal and lateral Energy Scores
7. If route invalid or `route_coverage_insufficient` (human extends beyond 90% of route arc length): set Frenet scores to NaN, populate QA fields
8. Write row to CSV incrementally (partial results preserved on interrupt)

### Stage 2: `rank_and_select.py` — Percentile Ranking & Selection (CPU)

```bash
python -m human_match_prototype.rank_and_select \
  --scores data/per_scene_eval/scores.csv \
  --output_dir data/per_scene_eval/results/
```

1. Compute within-validation-set percentile for each score
2. Combined ranks:
   - `R_overall = mean(P(es_2s), P(es_4s), P(es_8s))`
   - `R_lateral = mean(P(es_lat_2s), P(es_lat_4s), P(es_lat_8s))`
   - `R_longitudinal = mean(P(es_lon_2s), P(es_lon_4s), P(es_lon_8s))` — diagnostic only, not used for selection
3. Select `Top5(R_overall) ∪ Top5(R_lateral)` → 5-10 review candidates
4. Label each by strongest contributing dimension and horizon

Output: `ranked.csv` (all scenes with R_overall, R_lateral, R_longitudinal), `review_set.csv` (selected scenes), `distributions.png`.

### Stage 3: `render_report.py` — BEV Overlays & HTML Report (GPU)

```bash
python -m human_match_prototype.render_report \
  --review_set data/per_scene_eval/results/review_set.csv \
  --scores data/per_scene_eval/results/ranked.csv \
  --output data/per_scene_eval/results/report.html
```

For each review scene: re-sample (same seed), render BEV overlay with route centerline, projection connectors, and planner sample cloud. Produce self-contained HTML with distributions, tables, and embedded images.

## Module Design

### `energy_score.py` — Per-Scene Energy Score

```python
def per_scene_energy_score(
    human_xy: np.ndarray,    # (80, 2)
    samples_xy: np.ndarray,  # (N, 80, 2)
    horizons: dict[str, int] = {"2s": 20, "4s": 40, "8s": 80},
) -> dict[str, float]:
    """ES_h = mean_m D(X_m, y) - 0.5 * mean_{m≠n} D(X_m, X_n)
    where D is L2 norm of flattened trajectory vector at horizon h."""
```

For each horizon h:
- Flatten human and samples to vectors: human `(h*2,)`, samples `(N, h*2)`
- `obs_h = mean_m ||X_m - y||₂` (sample-to-human distances, then mean)
- `div_h = sum_{m≠n} ||X_m - X_n||₂ / (N*(N-1))` (distinct-pair pairwise mean)
- `es_h = obs_h - 0.5 * div_h`

Output keys: `es_obs_{h}`, `es_div_{h}`, `es_{h}` for each horizon.

### `route_projection.py` — Route Stitching & Frenet Projection

```python
@dataclass
class StitchedRoute:
    centerline: np.ndarray   # (L, 2) stitched route points
    arc_length: np.ndarray   # (L,) cumulative arc length
    qa: RouteQA              # validation fields

@dataclass
class RouteQA:
    route_valid: bool
    max_segment_gap: float
    total_interpolated_gap: float   # sum of linearly interpolated gap distances
    n_valid_segments: int
    route_arc_length: float
    route_coverage_insufficient: bool  # True when max(s_human) > route_arc_length * 0.9
    human_max_proj_dist: float
    n_monotonic_violations: int
    frac_planner_proj_fail: float

def stitch_route_lanes(route_lanes: np.ndarray) -> StitchedRoute:
    """route_lanes: (25, 20, 33). Extract x,y from first 2 features.
    Remove empty (all-zero) segments, check connectivity, deduplicate
    junction points, accumulate arc length."""

def project_to_route(
    route: StitchedRoute,
    points: np.ndarray,  # (T, 2) or (N, T, 2)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (s, d, proj_dist) — arc-length, signed lateral offset,
    and distance to nearest route point."""
```

Route stitching:
1. For each of 25 segments: check if all points are zero → skip
2. Extract `(x, y)` from `route_lanes[seg, :, :2]`
3. Check gap between segment `i` last point and segment `i+1` first point
4. If gap ≤ 0.5m: deduplicate the junction point, concatenate
5. If 0.5m < gap ≤ 3.0m: linear-interpolate 1-2 points across the gap, concatenate, record in `total_interpolated_gap` QA field
6. If gap > 3.0m: flag route as invalid, record `max_segment_gap`
7. Compute cumulative arc length along stitched points

**Note:** The gap tolerance (3.0m) is preliminary. Before production runs, calibrate by running a gap histogram across 500 validation scenes from the 20260715 dataset. The inspected 20260715 scene had exact endpoint matching, but older datasets show gaps up to ~2m.

Projection:
1. For each query point, find the nearest segment of the centerline
2. Project orthogonally onto that segment to get exact `s` and signed `d`
3. `d` is positive when the point is to the left of the route centerline (looking in the direction of increasing arc length), consistent with the standard right-hand Frenet convention
4. Record `proj_dist` (distance from query point to its projection on route)
5. For time-series (trajectories): enforce temporal continuity by limiting search to the previous timestep's segment index ± 5 segments (or ± 30m arc length, whichever is larger). Fall back to global search for the first timestep.

### Longitudinal/Lateral Energy Scores

After projecting human and all 64 samples to `(s, d)`:

```python
def frenet_energy_scores(
    human_sd: np.ndarray,    # (T, 2) [s, d]
    samples_sd: np.ndarray,  # (N, T, 2) [s, d]
    horizons: dict[str, int],
) -> dict[str, float]:
```

Compute Energy Score on the `s` (longitudinal) and `d` (lateral) projections separately:
- `D_lon(X_m, y) = ||s_m[:h] - s_human[:h]||₂` (L2 norm of arc-length timeseries difference)
- `D_lat(X_m, y) = ||d_m[:h] - d_human[:h]||₂` (L2 norm of lateral offset timeseries difference)

Then apply the standard ES formula to each: `ES_lon_h = mean_m D_lon(X_m, y) - 0.5 * mean_{m≠n} D_lon(X_m, X_n)`, and similarly for lateral.

Output keys: `es_lon_{h}`, `es_lat_{h}` for each horizon.

## Data Strategy

### Validation data
- Source: `path_list_valid.json` on sakurab (156,204 entries)
- Location: `/mnt/storage_rdma/diffusion_planner/dataset/20260715_basic_dataset/x2_dev/2355_Takanawa_gateway_copied_from_Aisantec/path_list_valid.json`
- Fetch: rsync to local storage with bandwidth limits (`--bwlimit=10000`)
- Smoke test: random 500 scenes
- Full run: all 156K (requires ~10-15GB local storage)

### Resource constraints
- **sakurab:** Read-only rsync for NPZ fetch. No GPU use, no heavy I/O. $1M training running.
- **Local PC:** GPU free for ONNX inference. Storage limited — check `df -h` before large fetches. System unavailable for recovery until next week.

### NPZ field reference
- `ego_agent_future`: (80, 3) [x, y, yaw] — human trajectory in ego frame
- `route_lanes`: (25, 20, 33) or (1, 25, 20, 33) — ordered route lane segments, features[0:2] are (x, y) in ego frame. Squeeze batch dim if present.
- All other fields consumed by `TrajectorySampler` for ONNX inference

### Constants
- Segment gap deduplication threshold: 0.5m (gaps ≤ this are deduplicated)
- Segment gap interpolation threshold: 3.0m (gaps 0.5-3.0m are linearly interpolated; gaps > 3.0m mark route as invalid). Calibrate against 20260715 dataset before production runs.
- Projection distance threshold: 5.0m (points farther than this from route count as projection failures)
- Route coverage threshold: 0.9 (if max(s_human) > route_arc_length × 0.9, Frenet scores are NaN)
- Monotonicity tolerance: allowed small backward arc-length steps ≤ 0.1m (stationary/reversing vehicle)

### Step 0: Dataset Format Verification
Before implementing route stitching, verify on 5-10 NPZs from the target 20260715 dataset:
1. Confirm `route_lanes` shape is (25, 20, 33) or (1, 25, 20, 33)
2. Confirm features[0:2] are (x, y) in ego frame
3. Run gap histogram across 500 sampled scenes to calibrate the gap tolerance
4. Verify that non-zero segments are ordered and contiguous (no interleaved empty segments)

## Output Contract

### Level 1: Per-scene score table (`scores.csv`)
One row per validation scene with:
- x-y Energy Score: `es_obs_{h}`, `es_div_{h}`, `es_{h}` for h ∈ {2s, 4s, 8s}
- Route Frenet scores: `es_lon_{h}`, `es_lat_{h}` (NaN when route invalid)
- Route QA: `route_valid`, `route_coverage_insufficient`, `max_segment_gap`, `total_interpolated_gap`, `n_valid_segments`, `route_arc_length`, `human_max_proj_dist`, `n_monotonic_violations`, `frac_planner_proj_fail`
- Metadata: `npz_path`

### Level 2: Distributions & ranked table (`ranked.csv`, `distributions.png`)
- Percentile ranks for all scores
- Combined ranks: `R_overall`, `R_lateral`
- Histograms of score distributions across the validation set

### Level 3: Inspectable evidence (`report.html`)
- The 5-10 selected review scenes (Top5 overall ∪ Top5 lateral)
- BEV overlay for each with route centerline, planner cloud, and human trajectory
- Route QA exception views where projection failed
- Metadata: temperature, seed, sample count, validation set size

## Interpretation Guardrails

1. Top-ranked scenes are "highest-disagreement review candidates," never automatic planner failures
2. Temperature 1.0 evaluates the training-matched distribution, not deployed T=0.5 — state in metadata
3. One human future cannot determine whether non-human planner modes are valid alternatives or confusion
4. Route QA failures retain x-y scores; Frenet scores are marked unavailable, not silently dropped
5. Groups (SHOULD-level) aggregate already-computed per-scene scores; they never pool human trajectories

## MoSCoW Scope

### MUST (this prototype)
- Score each validation scene independently: 64 DP samples at T=1.0 vs. own human
- Full-trajectory x-y Energy Score at 2s/4s/8s with observation/diversity split
- Route stitching, s/d projection, longitudinal/lateral Energy Scores with QA
- Three-level output: per-scene CSV, distributions, Top5-overall ∪ Top5-lateral review clips

### SHOULD
- Post-hoc route-location grouping with drive counts (aggregate scores, not trajectories)
- Shortlist stability: rerun top scenes with extra seeds
- Route-QA exception clip rendering

### COULD
- minADE/minFDE/close-sample-fraction as secondary diagnostics
- Temperature 0.5 sensitivity audit

### WON'T (v1)
- Semantic stop/go/turn classification
- Automatic mode clustering or uncertainty labeling
- Pooled multi-human target distribution
- Temporal event deduplication (no evidence of adjacent duplicates in validation list)

## Files Changed

### New files
- `human_match_prototype/energy_score.py` — per-scene ES computation
- `human_match_prototype/route_projection.py` — route stitching, s/d projection, QA
- `human_match_prototype/score_scenes.py` — Stage 1 CLI
- `human_match_prototype/rank_and_select.py` — Stage 2 CLI
- `human_match_prototype/render_report.py` — Stage 3 CLI

### Replaced (superseded, can be deleted or archived)
- `human_match_prototype/metrics.py` — old ADE/distributional metrics
- `human_match_prototype/run_all.py` — old scoring pipeline
- `human_match_prototype/analyze.py` — old analysis (BEV overlay utility extracted first)
- `human_match_prototype/multi_human_match.py` — pooled multi-human comparison
- `human_match_prototype/multi_human_report.py` — multi-human reporting
- `human_match_prototype/cluster_report.py` — cluster reporting
- `human_match_prototype/typicality.py` — Mahalanobis typicality (not used in new design)

### Kept unchanged
- `human_match_prototype/sampler.py` — ONNX trajectory sampler
- `human_match_prototype/sidecar.py` — NPZ sidecar reading
- `human_match_prototype/coord_transform.py` — SE(2) transforms
- `human_match_prototype/build_lanelet_index.py` — lanelet index (for SHOULD-level grouping)
