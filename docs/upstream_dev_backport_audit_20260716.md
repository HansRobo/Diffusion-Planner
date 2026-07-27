# Upstream dev backport audit (2026-07-16)

## Scope and baseline

This audit compares the local HDP branch with the freshly fetched original
Diffusion Planner branch origin/dev at 3838792. The merge base with this branch is
eb2e7f7. The target is the original Tier IV Diffusion Planner, not the HDP
architecture. RL, HDP velocity/hybrid loss, temporal decoder changes, traffic-light
masking experiments, custom manifests, and deployment-only contracts are therefore
not upstream candidates.

The list is split by what can be sent upstream. A local commit being useful for HDP
does not make it safe for original DP: original DP still predicts neighbors, uses the
81-step action shape, and has a different loss and decoder contract.

The audit did not push, commit, or submit any change.

## Local correction applied after the audit

The worktree previously passed `preserve_zero_padding=True` for raw
`ego_agent_past` and `goal_pose`. That is incorrect for this dataset: a stopped
vehicle at the ego origin is legitimately `(0,0,0)`, and the route goal can
coincide with that origin. Those calls now use the ordinary conversion in SFT
training, validation, HDP-RL preprocessing, ONNX input preparation, and
visualization utilities. The low-level augmentation paths now apply the same
field-aware rule. The padding-preserving behavior remains available only for
callers that have an explicit field-specific padding mask; no blanket all-zero
pose rule was added. This changes the effective input semantics for
affected samples, so a checkpoint trained with the old masking behavior should
not be treated as equivalent to a corrected run.

## Executive decision

Highest-value upstream work:

1. Fix the original DDP rendezvous and resume sampler ordering.
2. Fix the stale ego-history slice.
3. Fix pose-conversion aliasing (without masking valid origin poses) and the Python
   neighbor-future off-by-one.
4. Fix short/padded map token geometry and validation-map validity.
5. Remove per-step CUDA stalls and make optional AMP/optimizer/DDP speedups
   feature-detected rather than defaulted.
6. Add focused tests before opening separate PRs.

## P0/P1 correctness candidates

Items 1-3, 6 and 8-12 are suitable for upstream PRs after the tests described
below. History and label changes alter the training data seen by the model, so
they must be explicit retraining changes. Items 4-5 are retained for traceability
but are intentional augmentation choices that should not be upstreamed.

### 1. DDP rendezvous and single-process fallback

Local source: af34cb4 and DDP hardening in 3fc5010/f021a46.

Remote code: diffusion_planner/diffusion_planner/utils/ddp.py,
ddp_setup_universal still uses file:///tmp/tmp_dist_init, unconditionally overwrites
MASTER_ADDR/MASTER_PORT for torchrun, and returns without disabling args.ddp when no
launcher variables exist.

Why this matters: a shared fixed file rendezvous can collide with another job and is
not a valid multi-node rendezvous when /tmp is node-local. A direct invocation with
the default --ddp=True returns a rank but later wraps the model in DDP without an
initialized process group. Both failures happen before useful training and can look
like intermittent Slurm failures.

Upstream patch: use env://; use setdefault for launcher-provided MASTER_ADDR and
MASTER_PORT; set args.ddp=False for the explicit single-process fallback; keep the
SLURM address/port path; pass device_id and CUDA barrier only when supported by the
project PyTorch version.

Tests: two-process torchrun/Gloo smoke test with a temporary port, a no-launcher
single-process test, and a two-node/SLURM environment test with pre-set master
variables. Do not change the NCCL timeout or silently invent a new port allocator.

### 2. Set the distributed sampler epoch before iteration

Local source: 666098d.

Remote code: train.py calls train_sampler.set_epoch(epoch + 1) at the end of the
loop and never calls it at the top.

Bug: after resuming at epoch k, the sampler still has its constructor epoch (0) for
the first resumed iteration. The first resumed epoch can replay the epoch-0 shuffle,
then the next epoch uses a delayed value. This is a reproducibility and data-order
bug, not just logging.

Upstream patch: call train_sampler.set_epoch(epoch) immediately after the DDP barrier
at the top of the epoch and remove the end-of-loop call.

Test: construct a small DistributedSampler, resume from a non-zero epoch, and assert
that the first iterator equals a fresh sampler configured with that epoch and does
not equal epoch zero.

### 3. Use the recent six ego-history frames, not the oldest six

Local source: 06a4e4e/a04e163, Encoder.forward.

Remote code: model/module/encoder.py:169-176 keeps ego[:, :6] and zeros ego[:, 6:],
while the neighboring-agent path keeps neighbors[:, :, -6:].

Why it is a bug: the converter and parser store history oldest-to-current (the last
slot is the reference/current frame). The original ego encoder therefore sees the
oldest 0.5 s and zeros the current state, while the neighbor encoder sees the recent
0.5 s. This is an asymmetric stale-input error.

Upstream patch: keep the fixed input shape and mixer cost, but left-pad the latest
six ego frames with zeros, exactly as the neighbor path does. Do not expand the
contract to 31 temporal tokens in this PR.

Impact: retraining is required because the six input slots change meaning; there is
no ONNX tensor-shape change.

Test: feed a time-ramped history and assert that only the final six values survive
in their original order; test shorter-than-six histories and
use_ego_history=False.

### 4. Keep the intentional ego-history augmentation behavior

The remote implementation leaves the ego-history transform disabled, and this
branch has treated that as an intentional design choice rather than a correctness
bug. It must not be presented as an upstream fix merely because current state,
future, and map inputs are transformed.

Decision: do not transform ego history in the original-DP upstream PR set and do
not change the existing HDP/SFT behavior for this reason. Goal-pose augmentation
is likewise excluded in item 5.

### 5. Keep the intentional goal-pose augmentation behavior

The current augmentation contract intentionally leaves this goal input unchanged,
along with the ego-history behavior in item 4. This is not part of the upstream
correctness scope for the branch.

Decision: do not open a goal-pose augmentation PR and do not change the existing
HDP/SFT augmentation policy for this reason.

### 6. Avoid in-place aliasing in heading conversion (do not blanket-mask zero poses)

Local source: 3fc5010, heading_to_cos_sin.

Remote code: train_epoch.py:11-28 returns a 4-column tensor by alias. The conversion
of a 3-column (0,0,0) pose to (0,0,1,0) is normally correct for this dataset.

Failure mode: later in-place masking can mutate a batch-owned tensor when input was
already 4-column. That aliasing is a real generic footgun. The earlier claim that
every raw all-zero pose is padding is not valid for official DP data: the current ego
frame is represented in the ego frame and can legitimately be (0,0,0), and a route
goal can legitimately be at the current origin.

Upstream patch: validate last dimension 3 or 4 and return a clone for 4-column input.
Keep the existing conversion for ego history, ego future, and goal_pose. The official
neighbor-future path already computes its field-specific padding mask before
conversion and restores it afterwards. If a future dataset introduces missing ego or
goal poses, add an explicit validity field or metadata; do not infer it from xy/yaw
zeros.

Tests: 3-column conversion including a valid origin pose, 4-column
clone/non-aliasing, and the existing neighbor-future mask-before-conversion path.

### 7. Do not mask an all-zero goal token (rejected finding)

Local source: f021a46 proposed a GoalPoseEncoder mask.

Remote code: model/module/encoder.py:727-743 always returns mask=False and embeds
the goal.

Why the proposed fix is unsafe: the official converter always derives goal_pose from
the route goal. A valid goal at the current origin is represented as raw
(0,0,0) or converted (0,0,1,0), not as a missing-token sentinel. Masking based only
on all-zero values would erase a valid goal and change the original DP contract.

Decision: do not send f021a46's GoalPoseEncoder mask upstream. Only add a mask if the
converter first supplies an explicit goal-validity flag and the semantics are agreed.

Test/guard: add a valid-origin goal regression test if touching this code; do not
assert that an all-zero raw goal is necessarily absent.

### 8. Correct the Python converter neighbor-future start time

Local source: 55eff4f, Python part only.

Remote code: ros_scripts/parse_rosbag.py:361-400 appends the current-frame object
to the future deque and then appends frames i+1... Therefore neighbor_future[0]
duplicates the current/past state instead of the first future frame.

Cross-check: the current remote C++ converter intentionally seeds current state in
AgentHistory(OUTPUT_T+1) and writes arr[(t+1)*AGENT_STATE_DIM], which is correct.
The local C++ no-seed rewrite is equivalent, not a necessary upstream fix.

Upstream patch: remove the current-frame seed from the Python helper and write future
i+1...i+OUTPUT_T; retain leading zero padding when an object disappears.
Add a version/fixture test comparing Python output to the C++ convention. Existing
NPZs have the old alignment and must be regenerated or explicitly versioned; this is
not a DataLoader-time silent migration.

### 9. Use valid map geometry for short/padded lanes and lines

Split this into two small PRs so upstream #228 remains easy to review.

#### 9a. Lane token position/heading

Local source: current local LaneEncoder implementation from 3fc5010.

Remote code: model/module/encoder.py:562-567 always takes the fixed midpoint point.
For a short lanelet whose valid points occupy fewer than half of the fixed 20 slots,
that point is padding and yields a fake (0,0)/zero-direction positional token.

Patch: compute a validity-masked centroid and a stable direction from valid points;
keep the all-empty mask. Add short-lane, valid-origin, and empty-lane tests.

#### 9b. Line token differences/position

Remote code: after upstream #228, LineEncoder computes differences through the zero
suffix and takes the fixed midpoint (:666-679). A valid last point followed by
padding creates a fake segment to the origin, and short lines get a padding position.

Patch: compute pair differences only when both endpoints are valid. Preserve the
existing fixed midpoint/local-tangent representation whenever all `V=20` points are
present; for a short prefix, use the middle valid segment (or the single point with
a deterministic heading fallback). This fixes the padding boundary without moving
normal map tokens. Do not use an arithmetic centroid as the general replacement:
our real-data scan shows it can sit metres away from curved road-border geometry.

Impact: complete converter outputs are bit-identical and need no retraining. Only
malformed/short rows change. The patch adds a small mask/gather cost, so benchmark
the full encoder before sending it upstream; it is a robustness fix, not a claimed
SFT quality gain.

Tests: one-point/short line, zero suffix, valid point at (0,0) with a type bit, empty
line, and ONNX export shape parity.

### 10. Preserve explicit map validity in validation

Local source: 8f56d0b.

Remote code: validate_model.py:93-110 filters line points with _valid_xy, so a valid
geometry point at the ego origin is discarded. Similar fallback logic should be used
consistently for lane/route centerlines while retaining explicit lane geometry
validity channels.

Patch: use the converter's explicit type/validity channel where it exists and fall
back to xy-nonzero only for legacy arrays. Do not globally treat every (0,0) as valid;
the distinction must follow the NPZ layout.

Impact: validation numbers become less biased; model training is unchanged.

Tests: line/border point at origin, explicit-valid padded/nonvalid point, lane
polygon, and empty map.

### 11. Replace production asserts with explicit exceptions

Local source: generic portions of 30a11f9.

Remote code: DPM wrapper and sampler use asserts for user/config validation
(dpm_solver_pytorch.py:243-244 and sampler checks); Python -O removes them.

Patch: replace with ValueError/RuntimeError for unsupported model/guidance types,
missing classifier guidance, invalid solver step/time ranges, and malformed encoder
feature dimensions. Do not carry over HDP-only fixed [B,P,T,D], delay, or 80/81-token
checks.

Tests: run invalid inputs under normal Python and python -O; verify exception types
and messages.

### 12. Make DDP metric reduction safe for empty/divergent ranks

Local source: 29b9e30.

Remote code: ddp.reduce_and_average_losses inserts a barrier and then performs one
all-reduce per key, assuming every rank has the same nonempty dictionary and key order.

Failure mode: a rank with no local valid samples can leave early while another rank
enters an all-reduce (deadlock). Same-length but differently ordered dictionaries can
silently pair the wrong metrics.

Patch: collectively validate key count/order, pack scalar values into one tensor,
perform one all-reduce, and handle all-empty dictionaries collectively. Preserve the
existing return type and use this only for epoch metrics, not gradients.

Tests: all-empty, one-empty/one-nonempty (must raise on all ranks without hanging),
different key order, and normal reduction. Key validation must use a collective-safe
timeout/error path.

## P1 performance candidates

The local ddb0bb9 profile is the most useful upstream performance work. Port it in
small commits to original DP, leaving the original neighbor loss and 81-step tensors
intact.

### 13. Remove per-step CUDA pipeline stalls

Remote train_epoch synchronizes at the start and end of every step. The end sync has
no training-correctness role: CUDA stream ordering and DDP all-reduce provide the
required dependencies. Remove only per-step syncs; retain explicit syncs around
benchmarks/diagnostics where wall-clock timing is required.

### 14. Use non-blocking H2D and persistent/prefetched workers

Use value.to(device, non_blocking=True) with the existing pinned loader, and add
persistent_workers=True and configurable prefetch_factor only when num_workers>0.
Keep the num_workers=0 fallback; do not pass an invalid prefetch_factor in that case.

### 15. Cache normalizer statistics per device

ObservationNormalizer and StateNormalizer repeatedly call .to(device) on the same
mean/std tensors in the hot loop. Cache per (key, device) and invalidate if
statistics are ever mutated. Add numerical identity and multi-device/unpickle tests.

### 16. Do not retain every step's autograd graph for epoch logging

Remote train_epoch appends loss tensors to a Python list until epoch end. Although
get_epoch_mean_loss eventually calls .item(), the list keeps each graph alive for the
whole epoch. Accumulate detached scalar sums/counts or detach at append time. This is
a memory/throughput fix with no metric change.

### 17. Avoid materializing a full ones_like SDE tensor

The original path calls VPSDE_linear.marginal_prob(ones_like(...), t) only to obtain
schedule coefficients. Add schedule-only marginal_alpha/std paths and keep noising,
SDE, and loss math in fp32. Verify equivalent values on CPU and CUDA for the original
81-step shape before merging.

### 18. Avoid unnecessary encoder input clones

The original encoder clones ego and neighbor history before replacing them with fixed
zero-padded slices. Once forward is proven not to mutate input, remove those clones or
use views. This is a small bandwidth optimization; test that input dictionaries remain
unchanged.

### 19. Avoid host synchronization in log-SNR timestep construction

Remote DPM get_time_steps calls .cpu().item() for two scalar endpoints. Construct the
linspace on the target device and interpolate there. This is inference/validation
latency rather than a model change; add an exact-value tolerance test.

Local measurement: the corresponding HDP run measured roughly 1193 vs 794 samples/s
on an 8-GPU setup, but that number must not be promised for original DP until an
original-DP A/B is run. Gain depends on neighbor-loss geometry, loader, GPU, PyTorch,
and batch size.

## P2 opt-in speed/optimization candidates

These can help upstream, but change numerics or require a feature/driver matrix. Do
not enable them by default in a compatibility PR.

### 20. bf16 model-forward autocast

Local 5d926ff scopes bf16 autocast to model forward and casts output back to fp32;
noising, SDE, and losses remain fp32. This is the right scope for diffusion-sensitive
math, but original DP has neighbor prediction and penalty paths that need A/B tests.
Add --amp_dtype {off,bf16}, log the mode, gate on CUDA/PyTorch support, and compare
loss/PDMS/turn metrics and NaN rate.

### 21. Fused AdamW

Use an explicit startup capability probe and fallback before the first optimizer step.
Do not catch only one exception type and do not make checkpoint resume depend on a
different optimizer implementation. Keep the flag opt-in and test CUDA without fused
support.

### 22. DDP bucket views and static graph

gradient_as_bucket_view=True is generally low-risk; static_graph=True is not. The
original branch uses find_unused_parameters=True and has optional turn, neighbor, and
penalty paths. Static graph must remain opt-in until a full epoch proves identical
parameter usage and no graph breaks. Never upstream the HDP default-on combination.

### 23. TorchInductor component compilation

Local compile_model_components compiles encoder/decoder with dynamic=False. This can
help fixed-shape training, but original DP has different decoder/prefix behavior and
may trigger recompiles or incorrect assumptions. Send only as an experimental flag
after shape guards, warm-up timing, backward checks, and complete validation A/B. Do
not port HDP fixed-shape or 80-token restrictions.

### 24. TF32

Local 8d94137 is a clean opt-in CUDA throughput flag. Gate on CUDA capability, log the
setting, default it off, and report metric deltas. It is useful for matrix-heavy
original DP but is not a correctness fix.

### 25. AdamW no-decay parameter groups

Local cced09d excludes norm gains, biases, and position/embedding tables from weight
decay. This is a plausible optimization, not a bug: it changes regularization and
the checkpoint trajectory. Submit as a separate opt-in experiment with a strict
parameter-group test and original-DP metric comparison.

### 26. Foreach EMA updates

Local d5773ea replaces the old timm EMA wrapper with ModelEmaV3/foreach while
preserving .ema. It can reduce EMA overhead, but timm/PyTorch compatibility and
checkpoint key behavior must be tested. Keep a legacy fallback and do not mix this
with a model-architecture PR.

### 27. Validation sampler without duplicate padding

Local DistributedEvalSampler shards validation indices by rank without
DistributedSampler duplicate padding. This is a generic metric-correctness and
occasionally throughput improvement; upstream separately with a count-weighted
aggregation test. Do not replace the training sampler with it.

### 28. Batch-aligned training sampler (opt-in only)

Local BatchAlignedDistributedSampler avoids dropping the tail and keeps every local
batch shape fixed by repeating a minimal shuffled prefix. It helps compile/static-graph
experiments, but repeats slightly alter epoch weighting. Keep original drop_last=True
default and expose this only behind a flag.

## P3 maintenance and portability

- 5b9ec10: replace training-log-only pandas import with csv.DictWriter. Keep pandas in
  package dependencies until a repository-wide search proves no other DP script needs
  it; test TSV field order and restart behavior.
- 06a4e4e/4956206: open numeric NPZ archives with allow_pickle=False in a context
  manager and normalize non-uint8 unsigned metadata for torch collation. First verify
  official converter outputs have no object arrays; retain version handling where the
  upstream dataset API needs it.
- 4d3fc51: lazy PDMS re-exports in planner_metrics. This helps minimal imports when
  optional metrics dependencies are absent, but current normal imports include
  scipy/shapely, so call this packaging robustness, not a training bug.
- Share schedule constants between VPSDE_linear and NoiseScheduleVP and assert
  equality. They match today; this prevents future train/inference drift but is not an
  observed current failure.
- Fix mutable default dictionaries in model_wrapper; this belongs with explicit
  exception validation, not a large refactor.

## Already in remote dev — do not duplicate

Latest remote dev already contains the relevant upstream work from #228, #236, #237,
#238, #239, and #243:

- line/polygon type embedding and the #228 line-encoder refactor;
- q/k/v LayerNorm correction in self-attention;
- turn token type and masked scene pooling;
- independent turn-indicator network;
- agent-history update and capacity fixes;
- per-parameter gradient-stat reduction and the CUDA allocator setting;
- converter is_skipped/red-light/goal-pose/CMake changes from the upstream PRs.

Local HDP tests or reimplementations of those commits should not be sent again.

## Explicitly reject for original DP

The following local changes are valuable only for HDP or a separate experiment:

- velocity latent, hybrid waypoint loss, temporal ego decoder, route AdaLN, and any
  80-token/80-vs-81 shape changes;
- ego-only removal of neighbor future supervision or removal of original neighbor
  velocity masking;
- road-border/collision loss policy changes and RL rewards/gates/EMA policy-update
  behavior;
- traffic-light masking augmentation and causal red-light manifests;
- HDP normalizer/checkpoint/resume fields, RL microbatch/static reward kernels,
  closed-loop rollout tooling, and W&B project/run conventions;
- stale Autoware benchmark shape guards, ONNX delay/prefix contract changes, and ROS
  node changes;
- blanket zero-pose preservation for ego history/goal conversion and the proposed
  all-zero GoalPoseEncoder mask. Official origin-frame poses can be valid;
- transforming ego history during augmentation; this branch treats the current
  untransformed-history behavior as intentional and it is not an upstream bug;
- transforming goal_pose during augmentation; this is also an intentional
  augmentation policy and is not an upstream bug for this branch;
- local C++ neighbor-history rewrite (unknown-object filtering, history deletion,
  leading padding) without a C++/Python golden-output study. Upstream C++ current-state
  seed plus arr[t+1] is already semantically correct;
- local manifest-sidecar scanning or DataLoader-time is_skipped filtering. Upstream
  list-generation utility is the appropriate place for that policy.

## Recommended PR split and acceptance matrix

Do not submit one large branch. Recommended sequence:

The former input/data PR is intentionally split into separate PRs:

- PR2-A: recent-six ego-history slice only.
- PR2-B: heading-conversion clone and valid-origin semantics only.
- PR2-C: Python neighbor-future one-step alignment only.

There is no goal-pose-augmentation PR and no ego-history-augmentation PR.

1. DP correctness/data contract: DDP fallback/rendezvous, sampler set_epoch, recent
   ego slice, pose padding/clone, Python neighbor-future
   alignment. Include original-DP tests and state that affected datasets need
   regeneration/retraining.
2. Map/validation correctness: lane/line valid geometry, explicit validation map
   validity, production exception checks. No model shape changes.
3. Metric/DDP robustness: packed safe metric reduction and duplicate-free eval sampler.
4. Measured hot-loop performance: nonblocking transfers, persistent workers,
   normalizer cache, detached epoch accumulation, no per-step sync, schedule-only SDE
   coefficients. Publish original-DP A/B throughput and metric parity.
5. Opt-in acceleration: bf16, fused AdamW, bucket views/static graph, TF32,
   compilation, no-decay, and foreach EMA as independently switchable features.
6. Low-risk maintenance: csv logging, safe NPZ loading, lazy metrics imports.

Every PR should run original DP unit tests, a two-GPU smoke train, one validation
epoch with a deliberately non-divisible dataset, resume-from-epoch-k, ONNX export for
shape-preserving PRs, and numerical/metric A/B against unmodified origin/dev.
Architecture/label changes additionally require a fresh checkpoint and must never be
silently applied to an old checkpoint.
