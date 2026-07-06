# Map-less Driving Data Utilization — Design & Implementation Plan

**Date:** 2026-07-07
**Status:** DESIGN ONLY — implementation deferred (per team decision). This document is the
hand-off artifact: everything needed to implement later without re-deriving.
**Context:** We hold a large corpus of driving logs WITHOUT HD-map annotations (no
`lanes` / `route_lanes` / `line_strings` / `goal_pose`); ego history, ego states, neighbor
perception, and CAN turn indicators ARE available. Perception-based auto-labeled maps exist
but at precision well below HD-map quality. Overall data is scarce, so leaving this corpus
unused is wasteful. Goal: exploit it to improve robustness / general performance of the
Diffusion-Planner (incl. HDP velocity mode).

---

## 1. Core judgment

**The architecture already supports map-less samples structurally.** All map inputs are
independent token groups with zero-fill + mask conventions, verified safe in the 2026-07-06
review (fully-padded groups keep parameters in-graph; masked softmax is guarded;
FusionEncoder pins token 0 valid). The C++/Python converters emit fixed-size arrays —
a map-less sample is just zeros in the map arrays. **No pipeline schema change is needed
to make the data flow; the design problem is what to supervise and what to guard.**

Relevant inputs (authoritative shapes from `diffusion_planner/dimensions.py`):

| npz / input key | shape | map-less availability |
|---|---|---|
| `ego_agent_past`, `ego_current_state`, `ego_shape` | — | ✔ available |
| `neighbor_agents_past` (+futures) | (32/320, T, 11/…) | ✔ available (perception) |
| `turn_indicators` | (INPUT_T+1,) | ✔ available (CAN) — key intent signal |
| `static_objects` | (5, 10) | ✔/partial (perception) |
| `lanes`(+speed_limit ×2) | (NUM_SEGMENTS_IN_LANE, POINTS_PER_LANELET, SEGMENT_POINT_DIM) | ✘ (or noisy auto-label) |
| `route_lanes`(+speed_limit ×2) | (NUM_SEGMENTS_IN_ROUTE, POINTS_PER_LANELET, SEGMENT_POINT_DIM) | ✘ → replace with hindsight pseudo-route |
| `line_strings`, `polygons` | (10, 20/40, 2) | ✘ — loss/reward-side consumer (road border) |
| `goal_pose` | (4,) | ✘ → replace with hindsight pseudo-goal |

### Two traps that make naive zero-filling WRONG

1. **Zero `goal_pose` means "stop here".** After `heading_to_cos_sin` a zero goal becomes
   `(0, 0, 1, 0)` = "goal at ego position" — training on that teaches braking whenever the
   map is absent. Map-less samples MUST get a hindsight pseudo-goal (§3.2) or an explicit
   goal-invalid signal; never plain zeros.
2. **Route-free intersections are multimodal.** BC supervision without route teaches
   indecision at junctions — this directly attacks our #1 KPI (temporal stability /
   closed-loop flicker). **Never train map-less ego supervision without a substitute
   intent signal** (hindsight route/goal + CAN turn indicators).

---

## 2. Phased plan (P0 → P4, increasing effort)

### P0 — Map-degraded evaluation baseline (zero training risk; do first)
Purpose: a ruler for every later phase. Two eval variants:
- (a) **map-masked eval on existing HD data**: at inference, zero out
  `lanes*`, `route_lanes*`, `polygons`, `line_strings` and substitute the hindsight goal
  (§3.2) for `goal_pose`; run the standard open-loop probes AND closed-loop probes
  (`closed_loop_stability.py`, flicker metric — per team practice, do NOT judge on
  valid_loss).
- (b) **true map-less eval set**: convert a slice of the map-less corpus via the existing
  C++ pipeline with map arrays zeroed + pseudo-goal/route filled (§3), reserve as eval-only.
Deliverable: table of {collision, flicker, jerk, lateral error} for {full map, masked map,
true map-less} × {current best checkpoint}. Acceptance: numbers exist and are reproducible;
this quantifies the robustness gap before any training change.

### P1 — Map-dropout co-training + hindsight relabeling (the core phase)
Train on mixed data: HD-map samples unchanged; map-less samples with
- map token groups zeroed/masked,
- `goal_pose` := hindsight pseudo-goal, `route_lanes` := hindsight pseudo-route (§3.2),
- `turn_indicators` kept (CAN),
- map-dependent loss terms disabled (§3.3).
Expected wins: (i) direct BC signal from the new corpus for ego dynamics (HDP velocity
prior — map-free by construction) and neighbor prediction; (ii) robustness to map
degradation (graceful degradation story for deployment); (iii) enables **map-CFG** at
inference: `w·ε(map) + (1−w)·ε(no-map)` as a map-adherence dial — same inference-side
leverage family as the validated multi-mode re-rank.
Start at 20–30% map-less fraction; sweep {0.1, 0.2, 0.3, 0.5} against P0 metrics.

### P2 — Lane-noise augmentation + source flag (deploy-readiness for auto-labeled maps)
On HD samples, randomly perturb lane tokens to mimic auto-label noise: lateral jitter
(σ ≈ 0.2–0.5 m per point, correlated along the polyline), random segment dropout
(10–30%), endpoint truncation, duplicate/merge artifacts (mimic the auto-labeler's actual
failure modes — measure them first on a small paired set). Add a **source flag** so the
model can calibrate trust (§3.4). This phase needs no map-less data at all and makes the
model robust to the auto-labeled maps we already can produce.

### P3 — Auto-labeled maps as NOISY INPUT (never as supervision)
Upgrade map-less samples from "no map" to "rough map": feed auto-labeled lanes as
conditioning tokens with source flag = auto. HARD RULE: auto-labeled geometry never enters
any loss or reward (no road-border penalty against noisy borders — label noise poisons
training; input noise is learnable). Combine with P1: pseudo-route + noisy lanes.

### P4 — Map-less GRPO/RL
The reward stack is mostly map-free already: collision (perception boxes ✔), gt_l2 realism
(✔), kinematic consistency (✔). Replace the road-border term with a **hindsight-corridor
penalty**: distance of the rollout beyond X meters (start X=1.5–2.0 m) from the actually
driven path, same relu(margin−·) shape as the existing border penalty. Map-less scenes are
interaction-rich; collision-focused GRPO on them is pure additional signal.

Recommended order: P0 → P1 (highest confidence, smallest diff, synergistic with HDP/DFP
goals) → P2 → decide P3/P4 after P0-metric deltas are in.

---

## 3. Concrete implementation specification

### 3.1 Data schema & converter
- New per-sample metadata field in the npz (and sidecar): `map_available: uint8`
  (0 = map-less, 1 = HD, 2 = auto-labeled — reserve 2 for P3).
- Converter for map-less logs: identical C++/Python path, writing zeros for `lanes*`,
  `route_lanes*` (superseded by pseudo-route below), `polygons`, `line_strings`,
  `static_objects` if unavailable.
- **Do not rely on zero-detection alone** for map absence at train time — carry the
  explicit flag (zero-detection is how padding masks work internally, but sample-level
  semantics should be explicit).

### 3.2 Hindsight pseudo-goal / pseudo-route generation (converter-side)
- `goal_pose` := ego pose at t + H_goal (start H_goal = 8 s = the full future horizon; also
  try 15–20 s look-ahead from the log if frames exist — matches mission-goal semantics
  better). Ego-centric frame like the real goal; (x, y, cos, sin).
- `route_lanes` := the driven path from t to t + H_route (H_route = 20 s or until log end),
  resampled to POINTS_PER_LANELET points per segment, split into NUM_SEGMENTS_IN_ROUTE-max
  segments in the SAME feature layout as real route tokens (SEGMENT_POINT_DIM=13:
  centerline xy + left/right boundary offsets — synthesize boundaries at ±half-lane-width
   1.75 m; traffic-light/speed-limit channels zeroed, `route_lanes_has_speed_limit`=0).
- Mark pseudo tokens via the source-flag channel (§3.4), NOT by leaving them
  indistinguishable — otherwise the model can learn the causal leak "route == the exact
  future I must reproduce" and over-trust routes at deployment.
- Edge cases: log ends before t+H (truncate + flag last-segment); ego stationary the whole
  window (pseudo-goal = current pose is then TRUE semantics "hold still" — keep, this is
  the one case where a stop-goal is correct).

### 3.3 Trainer changes (P1)
- **Batch-homogeneous mixing** (strongly preferred over per-sample loss masks): two
  datasets/samplers, each batch entirely HD or entirely map-less, alternating by a
  configured probability — exactly the `grpo_epoch.py` sft/grpo mixing pattern
  (rank-synced RNG; reuse that code shape). This keeps loss code almost untouched:
  map-less batches simply run with `coeff_road_border_loss=0`,
  `coeff_neighbor_collision_loss` unchanged (collision penalty needs only neighbor boxes),
  ego + neighbor + turn-indicator losses unchanged.
- New TrainConfig fields (opt-in, defaults preserve current behavior):
  `mapless_train_set_list: Optional[str] = None`, `mapless_batch_prob: float = 0.0`,
  plus argparse via `_train_config_default` in both trainers.
- DDP note: batch-type choice must be rank-synchronized (same seeded RNG as grpo mixing)
  so all ranks run the same loss graph per step (`find_unused_parameters=False` stays safe:
  road-border penalty is a loss term, not a parameterized branch).
- Loader: assert `map_available` consistency within a batch.

### 3.4 Source-flag channel (P2/P3, small model change)
- Add +1 feature channel to lane & route point features (SEGMENT_POINT_DIM 13 → 14):
  0.0 = HD, 1.0 = auto-labeled/pseudo. Backward compat: warm starts from 13-ch checkpoints
  need the first Linear expanded — load old weights, zero-init the new input column
  (behavior-identical at flag=0), note in `assert_checkpoint_compatible` /
  `load_weights_only` allowlist. Alternative zero-diff variant: a learned embedding added
  to the token AFTER the first projection, keyed by the flag — no checkpoint surgery.
  (Prefer the embedding variant for warm-start friendliness.)

### 3.5 Map-CFG inference (P1 deliverable, optional)
- Train-time: P1 already provides the two conditions. Inference: run the DiT twice per
  step (map / map-masked conditioning) and combine predictions with weight w; expose as
  `map_cfg_weight` in Config (default None = off). Evaluate on the P0 rulers, and only
  ship through the closed-loop re-rank harness if it wins there (inference-cost note:
  2× denoiser passes, same cost class as the multi-mode re-rank we already validated).

### 3.6 Map-less GRPO (P4)
- `compute_hindsight_corridor_penalty(ego_world, ego_future_gt, margin)`: point-to-polyline
  distance of rollout waypoints to the driven path, relu(d − margin), cummax like the
  border penalty; plug where `w_road_border` currently gates. New arg
  `w_hindsight_corridor`, default 0 (off).
- Reuse `_RL_COLLISION_EVAL_STEPS` dense sampling as-is.

### 3.7 Files expected to change (when implemented)
converter (cpp_tools frame_filters/… + `ros_scripts/parse_rosbag.py`): pseudo-goal/route +
`map_available`; `dimensions.py` (flag channel option); `utils/dataset` loader + samplers;
`train_config.py` / both trainer argparses; `train_epoch.py` (batch-type dispatch);
`encoder.py` (flag embedding, P2+); eval scripts + closed-loop probes (P0);
`grpo_utils.py` (P4). No decoder/DiT change required for P0–P1.

---

## 4. Metrics & acceptance criteria

- Primary: closed-loop flicker / jerk / collision-segment-rate on (i) HD eval, (ii)
  map-masked eval, (iii) true map-less eval — P1 must not regress (i) beyond noise while
  improving (ii)/(iii). Judge ONLY on these probes (valid_loss is known uncorrelated).
- Secondary: turn-indicator accuracy (CAN-supervised, availability unchanged), neighbor
  prediction ADE on map-less eval.
- Ablations worth one run each: {P1 w/o hindsight relabel} (expected to hurt flicker —
  documents why the relabel is mandatory), {mapless fraction sweep}, {P2 noise on/off at
  deployment-noise level}.

## 5. Red lines (repeated from the review-style analysis)

1. No map-less ego supervision without an intent substitute (hindsight goal/route + CAN
   turn indicators) — flicker risk.
2. Auto-labeled map geometry: input-only until its precision is quantified; never in
   losses/rewards.
3. Pseudo-route always carries the source flag — prevent the "route == my exact future"
   causal leak.
4. Zero `goal_pose` is a stop command, not a null — never feed it accidentally.
5. Every phase gets judged on the P0 rulers before the next phase starts.

## 6. Open questions (decide at implementation time)

- H_goal / H_route horizons (8 s vs 15–20 s) — depends on how far logs extend past each
  sample window.
- Whether `static_objects` from perception is reliable enough on the map-less corpus or
  should be zeroed there too.
- Auto-labeler failure-mode measurement set for P2 noise calibration (needs a small paired
  HD/auto sample).
- Interaction with the (still-open) eval init-distribution alignment decision — P0 should
  use whatever init the team standardizes on.
