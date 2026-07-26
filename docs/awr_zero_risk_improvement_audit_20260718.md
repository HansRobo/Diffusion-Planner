# AWR suggestion audit: zero-risk changes versus new experiments

Last updated: 2026-07-19

## Scope

This note reviews proposed HDP/PlannerRFT/R2LPL-inspired changes against the AWR implementation that is actually running for the original T4 Diffusion Planner. Its purpose is to prevent an observability improvement from being confused with an algorithm change, and to prevent stale configuration assumptions from driving a full-corpus run.

## 中文结论

可以立即吸收、且不改变 reward/target/gradient 的改进包括：评估 RNG 隔离、ESS/coverage 指标语义修正、冻结 replay 配置契约校验、正式启动的 checkpoint/data/disk fail-closed 检查，以及把 AdamW moments 写进 continuation checkpoint。这些已经实现。是否在下一 cycle **恢复** optimizer 是另一项显式实验语义，默认关闭，不能因为状态已经保存就声称当前 run 使用了连续 optimizer。

DP-safe quintic ramp 是必要的表示修复，但它会改变 sampled trajectory 和排序，所以不是“零风险性能优化”。现在已经完成从原始 best 出发的完整 5,446,154-scene 对照：未经保守缩放的 raw policy 会过冲；从 epoch 4 更新方向构造的 `alpha=0.05` 单一 EMA checkpoint 则在固定 train selector 和完整 validation 上都取得小幅正收益。Survival reward、改 beta/G、signed border、短闭环和 R2LPL 都会改变学习问题，只能后续独立消融，不能偷偷混入当前主线。

前一个分析里的 `G=8, beta=0.5` 已过期；当前是公开代码量级的 `K=10, beta=1.0`。论文、公开代码和当前数据适配并不完全一致，后文逐项列出，不能把其中任意一个版本简称为“原版”。

## Current implementation facts

There are two distinct local AWR profiles:

| Profile | Group construction | Weighting | Local evidence |
|---|---|---|---|
| Released-HDP-style weighting with DP-safe ramp | all `K=10` plans stochastic | `exp(beta * group-z-score)`, `beta=1`, no weight-mean normalization | first audited full-data cycle selected a retained EMA with full deployment reward `0.933058 → 0.933557` |
| Conservative original-DP adaptation | deterministic candidate 0 plus stochastic plans | only `R > Rdet + 0.01`, with behavior retention anchor | smooth-ramp 512-update point remained negative |

The stale description `G=8, beta=0.5` does not describe either current formal profile. It must not be used to choose a new beta or group size.

The current reward path already provides:

- full ego oriented bounding-box collision and road-border checks;
- full-ego-perimeter minimum road-border clearance (`rb_min_dist`) and a configurable crossing margin;
- TTC, expert-relative progress, lane robustness with logged-lane-change exemption, and comfort;
- yaw-rate, steering/curvature and lateral-acceleration feasibility diagnostics;
- all-zero/equal reward-group rates, candidate safety rates and multimodality statistics;
- fixed train-set K=1 EMA checkpoint selection and source epoch-0 fallback.

These are not missing features.

Two details in the earlier suggestion need correction:

1. The paper says to discard finite tied-reward groups because they contain no within-scene preference signal. The clean public NAVSIM release instead gives those candidates unit weight after a zero z-score. Cycle 2 preserves the cache semantics with which it was mined; the audited tied-zero overlay makes the paper-described behavior available for later experiments. The paper, public release and local implementation must not be conflated.
2. `rb_min_dist` is an unsigned distance from 80 sampled ego-perimeter points to road-border polyline segments. The OBB/footprint semantics are present, but a genuinely signed clearance would require a trustworthy border orientation and drivable-side convention that the current NPZ line strings do not provide. It must not be claimed as already implemented or added by assigning an arbitrary sign.

### Paper prose, public code and current reconstruction are not identical

| Item | HDP paper | Public NAVSIM code/config | Current controlled profile |
|---|---|---|---|
| Group size | 32 | 10 | 10 |
| Temperature | 1.0 | implicit `exp(z-score)`, therefore 1.0 | 1.0 |
| Rollout latent | standard diffusion prior, `N(0,I)` | `model.generate(...)` uses its native prior | original-DP adaptation uses `0.5 * N(0,I)` before the separate 0.5 m trajectory augmentation |
| Reward | safety baseline or weighted risk/follow/lane in the appendix | official NAVSIM PDM scorer | local-data `hdp_pdm` proxy: Col×DAC terminal product with TTC/EP/LK/comfort quality |
| Tied finite group | discard | released NAVSIM path assigns unit weights after zero z-score | Cycle 2 retained its frozen cache contract; the audited overlay can discard tied groups in a separately identified arm |
| Augmentation schedule | not fully specified in the method table | only while `current_epoch < 5`; with refresh every 10 epochs this means the first rollout cache only | `every_refresh`, deliberately re-opening smooth DP-safe exploration at epochs 1/11/21/...; acceptance and low-LR replay, rather than disabling later exploration, control drift |
| EMA | says EMA is used for policy updates and lists `0.05`, but does not define its decay/update convention or boundary | clean released NAVSIM config sets `use_ema: false`; `ModelEma` initialization is commented out | Cycle 1 and the low-LR Cycle-2 arm used decay 0.999 per minibatch; the matched epoch-12 arm tests a local T4 interpretation: one 5% epoch-boundary accepted-policy interpolation with optimizer reset |

Consequently, “faithful” must name the source being followed. K=10 and beta=1 follow the clean public NAVSIM release; tied-group dropping follows the paper prose; the 5% epoch-boundary policy commit follows a later local Tier IV implementation and Cycle-1 empirical trust-region evidence. The paper's beta is the AWR reward-weight temperature in `exp(beta * normalized_reward)`, not a diffusion sampler-noise multiplier. The original-DP `0.5` latent scale and the augmentation's `0.5 m` standard deviation are two separate quantities. Cycle 2 is never changed in place: the conservative policy-interpolation implementation is evaluated as a separate, same-cache arm.

The optional first-failure-time `survival` mode is PlannerRFT-inspired. It is not HDP's published safety/risk reward and remains off in the controlled profile.

### Exact epoch and checkpoint semantics

The public HDP implementation makes every `epoch % replay_buffer_update_epoch == 0` epoch a **rollout-only** pass: it clears and fills replay, returns metrics without a loss, and performs no optimizer step. All other epochs sample scene-groups from that frozen replay buffer and optimize the reward-weighted diffusion loss. The local faithful disk implementation preserves this alternation.

Therefore the formal `100`-epoch schedule means:

- rollout refresh at epochs `1, 11, 21, ..., 91` (10 full-corpus mining passes);
- replay optimization during the remaining 90 epochs;
- approximately 3,547 synchronized optimizer updates in each replay epoch for the 5,446,154-scene train corpus and the configured 192 scenes/rank batch;
- a fresh DP-safe smooth-ramp augmentation group at every refresh. This is an explicit T4 adaptation, not a claim of line-for-line public-code equivalence: the released condition would augment only the first cache.

This is the same epoch *state machine* as the public code, not 100 independent fresh rollouts and not 100 replay updates over each scene. Public `total_epochs: 10000` would analogously contain about 1,000 rollout-only refresh epochs and 9,000 replay-training epochs; its number is not interchangeable with the paper's diffusion timestep count or the IL schedule.

Model states also have distinct roles. The live synchronized policy receives gradients. The slow EMA is the deployable/behavior policy used for the next rollout refresh and is the policy evaluated both on the fixed 65,536-scene train selector and on validation each epoch; `best_train.pth` is selected only by that train-selector EMA reward. Each checkpoint saves both states. Validation does not select the replacement model, and the ultimately selected EMA must receive a separate full-validation/deployment evaluation before a final performance claim.

### Optimizer lifecycle across refresh cycles

The clean public Lightning implementation keeps AdamW alive across refreshes and disables EMA in its released configuration. A later local Tier IV branch implements a different, explicit policy-iteration transaction: it trains a live proposal for one replay epoch, accepts exactly 5% of that proposal into the old behavior policy, copies the accepted policy back into the live model, and clears Adam state. This reset belongs to that local interpretation; it is not executable behavior established by the clean public release.

The T4 full-corpus implementation splits a costly rollout-only refresh and its replay epochs into separate processes so a strict multi-terabyte cache can be closed, audited and reused. Earlier local checkpoints saved only live model and EMA tensors, so every process boundary necessarily reset AdamW. That is a real lifecycle difference from public HDP, not an intrinsic property of AWR.

Continuation checkpoints now also save a portable CPU `optimizer_state_dict`. Restoration is opt-in and fail-closed:

- `--use_policy_state` loads the live policy associated with those moments;
- `--resume_optimizer_state` restores moments and step counts, while retaining the new cycle's explicitly configured learning rate;
- requesting optimizer restoration from an EMA load, or from a checkpoint without optimizer state, is rejected.

This distinction matters because two strategies answer different questions. **Persistent-Adam continuation** matches the released Lightning lifecycle. **Epoch-boundary 5% accepted-policy interpolation** tests the local T4 interpretation and deliberately resets the proposal optimizer after every accepted commit. The first Cycle-2 low-LR arm is retained as an empirical trust-region ablation; it is not relabeled as faithful. The same immutable Cycle-2 replay cache is used for the one-epoch conservative-commit arm, so reward, samples, candidate order and replay count remain matched. Future-cycle optimizer semantics will be chosen from that train-set result and recorded explicitly.

## Accepted zero-objective-risk changes

### 1. Evaluation RNG transaction

K-sample validation is stochastic. Previously, changing validation size or cadence advanced Python, NumPy and Torch/CUDA RNG streams, silently changing the diffusion noise used by the next replay epoch.

The evaluator now saves and restores all local RNG states around every distributed evaluation. Validation output remains stochastic and unchanged; only its accidental influence on later optimization is removed. This makes comparisons across `2,048`, `8,192` and full `46,262` validation scenes reproducible.

Implementation: `rlvr/train_awr.py::_preserve_training_rng`.

### 2. Exact split-process baseline inheritance

The full-corpus implementation separates one rollout-only refresh from its nine replay epochs. Both processes start from the same behavior/EMA checkpoint, fixed validation set, and fixed train selector. Re-evaluating that unchanged policy at replay startup was therefore duplicate work; because evaluation is wrapped in the RNG transaction above, omitting the duplicate call cannot alter subsequent rollout noise, replay indices, diffusion times, or gradients.

Replay may now inherit the mine-only process's epoch-0 artifacts, but only after a fail-closed audit verifies the complete checkpoint SHA-256, model-argument SHA-256, AWR and reward configs, legacy neighbor `+1`, X2 width `2.29156`, evaluation K, and the exact ordered path of all 46,262 validation plus 65,536 selector scenes. The current Cycle-2 production artifacts passed that audit and reproduced validation/selector rewards `0.9325865093 / 0.9318990432` exactly. The replay run copies the verified JSON into its own run directory and records `inherited_refresh_baseline.json`; any mismatch falls back to failure, not silent recomputation or acceptance.

Implementation: `rlvr/train_awr.py::_load_inherited_refresh_baseline` and `--inherit_refresh_baseline_root`.

### 3. Correct ESS semantics and coverage metrics

The previous metric guide incorrectly described ESS as an effective number of scenes. AWR computes it over active trajectory weights **inside one scene-group**.

The code now reports:

- `active_weight_count`: weighted trajectories per group;
- `effective_sample_size`: trajectory-equivalents per group;
- `effective_sample_size_fraction = ESS / active_weight_count`;
- `top1_weight_share`;
- `repair_target_group_rate`: groups with at least one positive repair target;
- `behavior_anchor_active_rate`.

These are derived from saved weights during replay, so they do not change reward, cache targets, gradients or runtime sampling.

Repair/behavior-anchor metrics are now emitted only for the deterministic-anchor positive-only profile. In the formal all-stochastic profile, candidate 0 is just another stochastic sample: its training chart is labeled `candidate0_reward`, and the misleading 0% repair / 100% behavior-anchor charts are omitted. Evaluation and train-selector reward remain true zero-noise K=1 deployment metrics.

Implementation: `rlvr/awr.py::compute_awr_weights` and `rlvr/train_awr.py::_DiskReplayReader.batch_from_indices`.

### 4. Frozen replay contract validation

A replay cache already contains final trajectories, rewards and AWR weights. Passing a different beta, group mode, augmentation/heading mode or reward config at resume time cannot change those stored values; previously it could silently make the command line disagree with the data actually being optimized.

Resume now compares the source run's complete AWR/reward effective config and neighbor-future alignment (`+1` for the current legacy data) against the requested run. Any mismatch fails closed and requires re-mining. This changes no valid replay update; it only prevents a cache from being mislabeled or reused under another method.

Implementation: `rlvr/train_awr.py::_validate_replay_source_contract`.

### 5. Formal-launch integrity guards

The full-data launcher verifies the untouched v5 source checkpoint SHA-256, makes the legacy neighbor-future `+1` alignment explicit, requires all input manifests to be readable, checks the audited manifest sizes (5,446,154 train and 46,262 valid scenes), and checks conservative NVMe headroom before spawning DDP. These checks cannot improve reward, but they prevent a long run from answering a different question or failing after filling the cache filesystem.

Implementation: `rlvr/autoresearch/run_full_sequence_awr_group_relative_ramp.sh`.

### 6. Frozen decoder context for replay-only epochs

The full original-DP NPZ stores 320 neighbors, 31 history frames and all map/route tensors. After rollout mining, replay already has the frozen behavior encoder and the formal objective has no expert-anchor or neighbor auxiliary loss. Nevertheless this v5 checkpoint's joint denoiser has `predicted_neighbor_num=320`, so it still needs three tensors to reconstruct the noisy trajectory input: ego current state, the final current-state frame for all 320 predicted neighbors, and all 320 aligned neighbor futures. The legacy replay path reopened and decompressed every source NPZ in every replay epoch only to recover these slices.

The formal PlannerRFT chain stores that exact decoder context once in its
Cycle-1 canonical replay generation:

- `ego_current_state` unchanged;
- all `predicted_neighbor_num=320` neighbors, but only their final history frame;
- all 320 neighbor futures, preserving the canonical loader's already-applied legacy `+1` temporal alignment without a second shift.

For the current shapes this is 80,330 float32 values, about 321.3 KB/scene or
1.75 TB (1.59 TiB) over 5,446,656 padded groups. This is a substantial
one-time static-data cache, not a rounding error. It eliminates up to 49
million compressed-NPZ opens across nine replay epochs, but is retained as one
immutable shared generation and linked by Cycles 2--10 rather than copied every
ten epochs. The independent Cycle-1 builder reads the writer's precommitted
rank-local expected path order, is resumable from a flushed scene position,
and publishes its own manifest only after completing a rank. After both the
candidate and context products close, a transactional attachment verifies all
eight path hashes, counts, tensor shapes, legacy `+1` alignment and X2 width
before changing any replay manifest. That shared path also validates the
source model-argument hash and explicit context schema before later cycles
create read-only links. Old manifests remain readable and fall back to the NPZ
path. New manifests fail closed if only part of the three-tensor context exists;
partial-prefix salvage validates the canonical path prefix and publishes the
context arrays together with trajectories/weights/encoding. Tail backfill
opens externally shared context read-only and regression tests verify the
source-file hashes remain unchanged.

The compact and legacy contexts produce exactly equal `gt_trajectories`, `sampled_trajectories` and per-trajectory diffusion losses in the regression test. A production-boundary regression also loads a raw NPZ through `load_npz_data`, verifies the required `+1` shift there, and verifies that compact publication preserves rather than repeats it. Context manifests now carry the explicit `canonical_loader_aligned_v1` schema; earlier experimental version-2 context arrays are rejected. This is a storage/I/O transformation only; it does not alter sampled plans, rewards, AWR weights, replay indices, diffusion RNG or optimizer update order. Runtime gain must still be measured on the first Cycle-3 replay rather than inferred from file-count reduction alone.

A measured uncached full-replay microbenchmark processed 6,144 global groups
at about `109 groups/s`, projecting to `13.9 h` for one 5.45-million-group
pass. A previous exact cached-context pass completed 5,448,192 groups in
3,697 s (`1.03 h`), a measured `13.5x` wall-time difference. These runs did
not share the current candidate cache, so the ratio is an engineering runtime
estimate rather than model-performance evidence; the first formal replay will
record the actual realized speed. A real-cache loader benchmark over disjoint
192-scene batches measured `266.6`, `400.8`, `423.0`, and `426.5 scenes/s` for
1, 2, 4, and 8 loader threads respectively. Four threads/rank therefore
captures almost all available I/O gain (`+58.7%` over one) without the doubled
thread count of the marginally faster eight-thread case. Before enabling it,
the ordered threaded path was compared with the canonical loader on both
synthetic scenes and a real X2 scene: ego current state, the final
neighbor-history frame, and the already-shifted neighbor future were bitwise
equal. The real scene also confirmed canonical X2 width `2.29156 m`. Width is
represented in the cached behavior encoder, not in the three decoder-context
tensors; omitting `ego_shape` from the replay NPZ shortcut therefore avoids
redundant I/O without dropping the geometry correction.

Full validation has the opposite NPZ parsing profile because it needs every model and reward field. A disjoint real-validation CPU probe measured `134.3`, `127.5`, `120.6`, and `99.4 scenes/s` at 1, 2, 4, and 8 readers. Replay and evaluation therefore now have independent settings: four readers/rank for the legacy three-array replay shortcut and one reader/rank for full-scene evaluation. Parser concurrency is also decoupled from CUDA transfer batching: every packed multi-scene batch is assembled on CPU and copied to the GPU once per field, even with one reader. The former one-reader branch copied each field of every scene separately before concatenation, so it could not realize the measured serial-parser advantage at GPU runtime.

The first CPU-merge prototype exposed an important exactness trap: computing `ego_agent_past` cosine/sine on CPU instead of CUDA changed 14 of 496 tested values by one float32 ULP (`5.96e-8`). The final loader therefore keeps raw heading angles during CPU parsing/prefetch, performs the batched host-to-device copy, and invokes the canonical transform on CUDA. On 24 real validation scenes, all 19 loader fields then matched the legacy per-scene CUDA path bit-for-bit while loader time fell from `0.3650 s` to `0.1929 s` (`1.89x` in this isolated probe). The rollout-prefetch boundary was separately checked with the same real scenes and was also bitwise exact after its CUDA-side finalization. A four-scene real X2 CUDA audit likewise had no mismatched field, carried width `2.29156 m` for every scene, and left the aligned neighbor-future tail at zero as required by the legacy `+1` contract. A second disjoint GPU probe after this fix measured `152.1`, `127.6`, `131.8`, and `95.5 scenes/s` at 1, 2, 4, and 8 readers respectively, confirming one reader as the full-validation setting after transfer batching—not merely in the CPU parser probe. Rollout loading remains at one reader. This avoids trading replay speed for slower per-epoch validation and changes neither mining order nor candidate generation.

BF16 storage was considered and rejected using the real Cycle-2 cache. Although the encoder runs inside an autocast scope, its final cached output is genuinely float32: over 2,310,144 sampled values only `0.00035%` survived a BF16 round trip exactly, with maximum absolute error `0.0120`. Halving the 3.15-TB encoding array would therefore be lossy and could change decoder outputs/gradients. The formal cache keeps float32. This is an example of why an acceleration inferred from the autocast setting must be validated on the actual boundary tensor before adoption.

Implementation: `rlvr/train_awr.py::_compact_replay_decoder_context`, `_DiskReplayWriter`, `_DiskReplayReader`, and `_iter_prefetched_replay_batches`.

### 7. Exact frozen state under EMA

Only the original-DP DiT is trainable in the formal AWR run. The encoder is frozen, but the previous generic EMA update still evaluated `decay * x + (1-decay) * x` on every float32 encoder parameter. That expression is not bitwise identical to `x`; repeated updates changed 634/639 frozen encoder tensors by up to `8.34e-7`. On real scenes, the small parameter drift amplified to a maximum cached-encoding difference of `0.022` between refresh cycles. This was unintended state mutation, not learning.

EMA now blends only parameters whose live `requires_grad` is true. Frozen parameters and all buffers are copied exactly from the live model. A 10,000-update regression test verifies bitwise equality of frozen state while trainable state still follows the requested EMA equation. This prevents an encoder that is outside the optimizer from silently changing across refreshes; it does not change the AWR reward, target, loss or DiT update.

Implementation: `rlvr/awr.py::update_ema`.

### 8. Immutable frozen-encoder reuse across refreshes

The public HDP data path consumes an offline frozen encoder cache. The T4 disk adaptation initially recomputed and rewrote approximately 3.15 TB of scene encodings at every ten-epoch refresh even though only the DiT is trained. A real duplicate-scene audit established that the same checkpoint produces bitwise-identical cached encodings; with the exact frozen-state EMA fix above, the encoder payload can remain identical across later selected checkpoints.

From Cycle 2 onward, each refresh reuses the completed formal Cycle-1
rank-local encoding, decoder context, expert-anchor sidecar and scene order.
Only policy-independent products are shared; trajectories, rewards and AWR
weights are freshly mined from the latest selected checkpoint. This path is
deliberately fail-closed:

- SHA-256 is computed over the exact encoder payload selected by checkpoint loading (live or EMA); a mismatch aborts before model execution;
- neighbor-future alignment and all eight strict replay manifests must match;
- the canonical padded DDP scene order must match every path index exactly;
- a new deterministic seed is assigned per refresh, so candidate diffusion noise remains fresh even though the scene-to-encoding index is fixed;
- the new replay cache links to the immutable float32 encoding/context rather
  than writing another copy; every later cycle freshly mines only
  policy-dependent trajectories, rewards and AWR weights;
- partial-prefix salvage and tail backfill preserve the external encoding contract, including read-only access to the shared array.

The sampler regression test uses an encoder that deliberately raises if called and verifies that cached encoding produces the expected complete candidate batch. Separate tests cover exact prefetch ordering, selected-payload hashing, inherited-baseline identity, replay round-trip, production-boundary neighbor alignment, read-only shared context, and interrupted-cache salvage/backfill. The full regression set currently passes (`55` focused AWR/alignment/geometry/visualization tests; `38` AWR observability tests). Runtime gain remains an empirical Cycle-3 measurement, not a claimed reward improvement.

Implementation: `rlvr/train_awr.py::_checkpoint_encoder_sha256`, `_iter_prefetched_scene_batches`, `_DiskReplayWriter`; `rlvr/awr.py::sample_unguided_dp_group_batch`.

### 9. Compact mine-only distributed reporting

Cycle 2 exposed a pure close-time bottleneck after all eight strict manifests
had already been published: every rank attempted to serialize roughly 680,960
Python metric dictionaries through `gather_object`, approximately 5.45 million
rows in total, only so rank zero could compute scalar means and write a second
JSON representation of facts already stored in the replay memmaps. This kept
the GPUs allocated and CPUs busy for minutes after mining reached 100%.

Future mine-only refreshes summarize finite numeric values locally on each rank
and gather eight compact summaries. The exact per-scene trajectories, rewards,
weights, noise scales and scene paths remain in the strict replay cache and are
read by `analyze_full_replay_signal.py`; `train_epoch_*/scenes.json` is an empty
compatibility artifact whose summary points to that cache. Replay-training
epochs keep their existing row collection because they produce only one row per
optimizer update, not one row per scene.

This branch runs only when the epoch is provably a standalone, update-free
refresh. It executes after cache close and cannot change sampling, reward,
weights, RNG, optimizer, EMA or checkpoint selection. The compact merge has a
focused regression test and the full AWR observability suite passes (`40`
tests).

Implementation: `rlvr/train_awr.py::_compact_numeric_row_summary` and
`_merge_compact_numeric_row_summaries`.

## Required representation correction, but not a zero-risk algorithm change

For absolute x-start DP only, HDP's constant route-frame offset is replaced by a state-continuous quintic onset. When a deterministic behavior candidate exists, all four channels remain bitwise unchanged. Invalid or incomplete caches remain fail-closed.

This does not redesign the reward, but it does change sampled trajectories, their ranking and therefore the regression targets. It is a necessary representation correction with strong geometry evidence, **not** a guaranteed performance-neutral change. That is why it is being tested from the untouched source checkpoint before full-corpus use. Details: `docs/hdp_augmentation_dp_representation_pitfall.md`.

## Useful ideas that are not zero-risk

| Proposal | Why it is not zero-risk | Decision now |
|---|---|---|
| Survival/partial-credit reward | changes candidate ordering and can prefer late failure or conservative stopping | already implemented as an optional mode; keep formal baseline on HDP-PDM `gate` until a matched reward ablation |
| Accept/rollback every EMA step | changes the policy that mines the next replay cache; finite selector noise can reject a useful intermediate policy | keep source and `best_train.pth`; study transactional policy iteration separately |
| Beta sweep | changes concentration among weighted trajectories; beta values are not comparable across reward/group definitions | use new ESS fraction first, then run a fixed-cache ablation if concentration is unhealthy |
| `K=16/32` | changes exploration coverage, cache size, compute and group statistics | retain `K=10` for the matched reconstruction; test later only if diversity/repair coverage is insufficient |
| New signed-border/curvature reward | footprint clearance and kinematic/curvature diagnostics already exist, but border sign does not; defining a sign or changing weights/gates changes semantics and candidate ordering | keep the audited unsigned footprint clearance now; do not invent border orientation |
| Short closed-loop rollout | changes state distribution and interaction semantics | later-stage experiment after open-loop AWR is positive |
| R2LPL repair mining | changes data selection and target construction | reserve for unrecoverable/all-zero or override scenes after the base AWR cycle |
| Critic/value network | introduces a learned baseline and additional optimization failure modes | not justified while group-relative signal is healthy |

## Current lowest-variable experiment

The strongest local evidence comes from the earlier all-stochastic group-relative full run, which first dipped at epoch 2 and improved from epoch 3. The current controlled run therefore keeps that proven AWR construction and changes only the original-DP-incompatible constant onset to the smooth ramp.

Configuration: `rlvr/configs/awr_original_dp_t4_hdp_group_relative_ramp.json`.

Before replay, the final preserve-heading cache was audited over 133,120 padded scene-groups (131,072 selected source scenes). It has enough ranking signal without pathological weight concentration:

| Cache diagnostic | Measured value | Interpretation |
|---|---:|---|
| Active trajectories per group | 10.00 | faithful all-stochastic group-relative AWR |
| All-zero / tied groups | 1.84% | only a small fraction lacks ranking signal |
| Best sampled reward | 0.93889 | feasible alternatives exist in the sampled group |
| Candidate collision / kinematic / road-border rates | 0.383% / 0.054% / 6.230% | no tangent-heading kinematic explosion |
| ESS | 7.02 trajectories | weights are not collapsing to one candidate |
| ESS fraction | 70.21% | directly comparable across different active counts |
| Top-1 weight share | 21.72% | stronger than uniform 10%, but not winner-take-all |
| First-waypoint spread P50 / P90 | 0.008 / 0.041 m | current-state continuity is preserved |
| Endpoint spread P50 / P90 | 2.22 / 3.46 m | future multimodality remains available |

Cache: `outputs/awr_t4_stable_ablation/20260718-183426_source_best_group_relative_smooth_ramp20_preserve_cache131072/`.

For context, the earlier positive constant-offset full-corpus cache had ESS 6.99, top-1 share 21.93%, all-zero groups 1.53%, candidate kinematic failures 0.040%, road-border crossings 6.70% and collisions 0.458%. The new preserve-heading ramp cache is in the same signal regime rather than a collapsed or winner-take-all regime. Because the old cache covers the full corpus and the new cache is a 131k subset, this comparison is a health check, not a causal ramp-vs-constant result.

The cache occupies 78.63 GB for 133,120 padded groups. Linear scaling predicts approximately 3.22 TB (2.93 TiB) for all 5,446,154 train scenes. The formal launcher refuses to start with less than 6 TiB free on the cache filesystem; this is a capacity guard, not a training optimization.

The optimizer-budget-matched stress run is `outputs/awr_t4_stable_ablation/20260718-184638_group_relative_smooth_ramp20_preserve_2fullreplay/`. It starts from the untouched original v5 best DP and performs 3,547 optimizer updates per replay epoch. This matches the number of updates in one full 5.446-million-scene replay pass, but **not** its data coverage: the 131,072-scene support is sampled with replacement about 40.9 times per epoch on average. A negative result can therefore include subset overuse; a positive result still requires fresh full-corpus mining and training. Its result must be reported separately from the cache audit; good cache statistics are evidence that training input is healthy, not evidence that the resulting checkpoint improves.

### Stress-run epoch 2 result

After one 3,547-update stress epoch, the first checkpoint was negative on aggregate reward but showed a specific safety/progress trade rather than reward collapse:

| 8,192-scene metric | Source DP | Epoch 2 | Delta |
|---|---:|---:|---:|
| Validation K=1 reward | 0.932404 | 0.931135 | -0.001269 |
| Fixed train-selector EMA reward | 0.932504 | 0.928499 | -0.004005 |
| Validation path length | 42.125 m | 40.633 m | -1.491 m |
| Validation progress | 19.736 | 19.576 | -0.160 |
| Validation ADE | 2.602 m | 2.826 m | +0.224 m |
| Collision rate | 0.171% | 0.134% | -0.037 pp |
| Road-border crossing rate | 1.733% | 1.294% | -0.439 pp |
| Kinematic failure rate | 0.037% | 0.110% | +0.073 pp |

Scene-paired analysis found reward improvement in 3,516/8,192 scenes (42.92%), centerline improvement in 67.97%, and road-border recovery in 42 scenes. Median path-length delta was -0.624 m and only 44.81% of scenes improved progress. Thus the reward is producing real safety/centering pressure, while the first deploy-time projection remains systematically shorter. Epoch 2 is not selectable over source. Epoch 3 is retained because the earlier full-data run also dipped on epoch 2 before turning positive, but this subset stress result is not full-data evidence.

The contraction is not uniform across traffic states. A paired stratification by the logged expert's valid path length shows that genuinely stopped/low-speed scenes improve, while moving scenes absorb the regression:

| Expert valid path | Scenes | Reward delta | Predicted path delta | ADE delta | Progress delta | RB-rate delta |
|---|---:|---:|---:|---:|---:|---:|
| <= 1 m | 236 | +0.01265 | -0.11 m | -0.008 m | -0.050 | -1.271 pp |
| 1--10 m | 844 | +0.01015 | -0.72 m | -0.080 m | +0.057 | -1.066 pp |
| 10--30 m | 1,665 | -0.00529 | -1.89 m | +0.189 m | -0.140 | -0.781 pp |
| 30--60 m | 2,538 | -0.00415 | -1.71 m | +0.242 m | -0.182 | -0.276 pp |
| > 60 m | 2,719 | -0.00093 | -1.50 m | +0.343 m | -0.240 | -0.147 pp |

This matters for diagnosis: AWR is learning useful stopping/border behavior in short-path scenes, but the same fixed-cache stress update is too contractive on moving scenes. It motivates a future matched progress-aware ablation if the fresh full-corpus run reproduces the pattern; it does **not** justify changing the formal reward after seeing one subset checkpoint.

Do not judge it after 512 updates: one full-corpus replay epoch is approximately 3,547 optimizer updates. Report both the early trajectory and the full replay-scale result.

### Stress-run epoch 3 result

The second 3,547-update pass partially recovered the live policy, but it still did not produce a selectable EMA checkpoint:

| 8,192-scene metric | Source DP | Epoch 2 | Epoch 3 | Epoch 3 vs source |
|---|---:|---:|---:|---:|
| Live validation reward | 0.932404 | 0.931135 | 0.934060 | +0.001656 |
| Fixed train-selector EMA reward | 0.932504 | 0.928499 | 0.929754 | -0.002751 |
| Validation path length | 42.125 m | 40.633 m | 41.214 m | -0.911 m |
| Validation progress | 19.736 | 19.576 | 19.676 | -0.060 |
| Validation ADE / FDE | 2.602 / 5.728 m | 2.826 / 6.434 m | 2.705 / 6.094 m | +0.103 / +0.366 m |
| Collision rate | 0.171% | 0.134% | 0.146% | -0.024 pp |
| Road-border crossing rate | 1.733% | 1.294% | 1.147% | -0.586 pp |
| Kinematic failure rate | 0.037% | 0.110% | 0.098% | +0.061 pp |

Paired epoch-3 analysis gives mean reward delta `+0.001656` but median delta `-0.000965`; only 3,324/8,192 scenes (40.58%) improve reward. It recovers 53 individual road-border failures and 3 collision failures, so a small set of terminal-event transitions lifts the mean. At the same time, the typical scene is still slightly worse and moving-scene contraction remains, though materially smaller than at epoch 2. In the 30--60 m expert-path stratum, mean reward remains `-0.00172` with predicted path `-1.11 m`; in the >60 m stratum reward becomes `+0.00070` while path remains `-0.75 m`.

The precommitted train-set selector therefore keeps source epoch 0 as `best_train.pth`. The stress run is evidence that the update can recover hard safety failures and that the second pass reverses part of the first-pass contraction; it is not replacement evidence. Because its 131k support is sampled about 40.9 times per replay epoch, the next experiment must start from untouched source and mine the full 5,446,154-scene corpus rather than continue epoch 3.

### Is the reward directly selecting shorter trajectories?

No, not at corpus level. A complete replay-cache audit over all 133,120 padded groups compared each candidate's full path length with its frozen AWR weight and reward:

| Within-group diagnostic | Mean | Median | Positive fraction |
|---|---:|---:|---:|
| AWR-weighted path minus uniform candidate mean | +0.0566 m | +0.0231 m | 54.98% |
| Best-reward path minus uniform candidate mean | +0.1378 m | +0.0703 m | 54.77% |
| Reward/path-length correlation | +0.0409 | +0.0654 | 54.46% |

The candidate bundle has a mean within-group path-length standard deviation of `0.679 m`, and 98.16% of groups have non-zero reward spread. Thus the epoch-2/3 deployment contraction is not explained by a reward that globally ranks shorter candidates higher. The stronger hypothesis is distributional projection: the 131k support is replayed about 40.9 times per epoch, while the x-start model regresses several perturbed stochastic modes and is evaluated at a zero-noise deployment latent. Repeated finite-support regression can move that zero-noise projection toward a shorter conditional mean even when the weighted sampled targets are slightly longer. Fresh full-corpus support is therefore a causal test, not merely a scale-up exercise.

If full-corpus training reproduces the contraction, the next matched ablation should target this mechanism rather than blindly increasing progress reward: include the zero-noise behavior trajectory in an otherwise all-candidate group, compare lower replay reuse/learning rate, or confidence-modulate nearly tied groups. Each changes the learned distribution and must be tested separately; none is silently mixed into the faithful formal baseline.

## Formal full-corpus run

The first formal cycle started from the untouched v5 checkpoint on 2026-07-18 at 20:12 JST:

- run: `outputs/awr_t4_full_sequence_filtered/20260718-201206_full_sequence_20260707_group_relative_ramp20_preserve_e100/`;
- W&B: `advanced-technology-department/original-dp-awr/dusb7jjy`;
- train: all 5,446,154 audited scenes; valid: all 46,262 audited scenes; train selector: fixed 65,536-scene sample;
- source full-valid reward `0.93215814`, ADE/FDE `2.6023/5.7201 m`, collision `0.1405%`, road-border crossing `1.7444%`, kinematic failure `0.1059%`;
- source train-selector EMA reward `0.93181720`, which is the epoch-0 `best_train.pth` threshold;
- epoch 1 was a rollout-only full-corpus refresh from source, not a continuation of either stress checkpoint or its 131k replay cache.

All nine replay epochs completed. The raw epoch checkpoints over-shot the fixed train selector, so none was promoted directly. A retained update was then selected strictly on the fixed 65,536-scene train selector:

- checkpoint: `best_model_awr_retained_e004_a0p05.pth`;
- construction: `source + 0.05 × (epoch_004_AWR - source)`, with model and EMA payloads interpolated independently;
- train selector K=1 reward: `0.93181720 → 0.93189782` (`+0.00008062`), with 54.21% of scenes improving;
- independent full 46,262-scene, zero-noise K=1, 10-step deployment reward: `0.93305821 → 0.93355743` (`+0.00049921`);
- paired bootstrap 95% interval for the full-data mean reward delta: `[+0.00021309, +0.00080759]`;
- collision / road-border / lane diagnostic / kinematic event-rate deltas: `-0.0022 / -0.0562 / -0.1146 / -0.0086` percentage points;
- ADE/FDE deltas: `+0.0080 / +0.0176 m`, reported as trade-offs rather than used as a hard rejection gate.

This proves a small positive large-scale deterministic open-loop post-training result for the stated reward and data contract. It does not by itself prove reactive closed-loop or real-vehicle improvement.

Cycle 2 starts from that selected EMA, re-mines all training scenes on-policy with the same reward contract and the same smooth-ramp augmentation, and uses `2e-8` replay LR to scan a much finer trust region instead of repeating the first cycle's over-shoot. The separate mining process writes a local epoch-1 cache, but the replay run maps it to global epoch 11 and performs optimizer updates at global epochs 12–20. The inherited checkpoint is recorded as the epoch-11 fallback if no cycle-2 train-selector EMA exceeds it. From cycle 3 onward the mining process itself uses global epoch 21/31/...; this changes scene/noise assignment as it would in a continuous HDP run, instead of resetting every refresh to the same local epoch-1 stochastic stream.

The low-LR arm was stopped after its first complete replay epoch because the
fixed train selector was decisively negative, despite a small road-border
improvement:

| Cycle-2 low-LR epoch 12 | Incumbent | Candidate | Delta / paired result |
|---|---:|---:|---:|
| Train-selector reward | 0.931899 | 0.928364 | -0.003535; 95% CI `[-0.003746, -0.003324]` |
| Reward-improved scene fraction | — | 24.66% | 73.23% regressed |
| Road-border crossing | 2.0676% | 2.0432% | -0.0244 percentage points |
| ADE / FDE | 0.9319 / 1.9770 m | 1.1944 / 2.6224 m | +0.2626 / +0.6453 m |

Thus `2e-8` does not solve the policy-step problem under minibatch EMA: the
branch learned a slightly more centered/border-safe but broadly contracted
trajectory distribution. `epoch_012.pth`, the complete per-scene rows and the
10,000-sample paired bootstrap report are retained as negative evidence; no
epoch 13–20 update is consumed.

The next matched arm reuses the exact same immutable Cycle-2 cache and starts
from the same incumbent, but applies the local T4 conservative transaction:
`1e-6` proposal optimization followed by a single 5% epoch-boundary policy
commit and optimizer reset. This isolates the policy-commit semantics without
changing reward, candidates, weights, scene order or selector.

Every cycle-transition supervisor now verifies the selection rule string, finite starting/best rewards, exact reported delta, and `best_train_reward >= starting_train_selection_reward` before propagating a checkpoint. It also verifies that an incumbent keeps the starting epoch, while a new winner belongs to that cycle's replay range and its checkpoint hash matches the final summary. This makes the fixed train-selector score monotone across the automated epoch-100 chain. Full validation remains report-only, as requested; monotonicity on that fixed train monitor is not a claim that every validation submetric is monotone.

Cycle 2 retains the cache semantics with which it was started and is never
modified mid-replay.  Its exact full-cache audit found that every reward-tied
group is an all-zero terminal group (`1.8730%` of scene-groups).  The HDP paper
and the local reference implementation both discard identical-reward groups,
whereas the Cycle-2 Original-DP cache had assigned them uniform unit weights.
Starting with Cycle 3, `drop_all_zero_groups=true` restores the reference
behavior for this empirically identical tied stratum: these rows remain in the
strict cache for complete data/order provenance but carry zero AWR weight and
therefore no preference-free self-distillation gradient.  Reward definitions,
non-tied group weights, sampling, replay count, EMA and checkpoint selection do
not change.  The effective config and W&B tag `drop-tied-groups` make the cycle
boundary explicit rather than presenting it as an unchanged experiment.

After every strict refresh close and before any replay update, the supervisor also audits the complete reward/weight memmaps and emits `full_replay_signal.json`, PNG, and SVG. It checks the exact 5,447,680 padded-group/54,476,800-candidate count, `K=10`, positive mean weighted-vs-uniform reward direction, ESS within `[1,10]`, and a non-degenerate all-zero fraction. This is a training-signal health gate and visual provenance artifact, not checkpoint-performance evidence.

## Tests

`rlvr/test_awr_observability.py` covers the candidate-0 invariant, exact constant-transform compatibility at ramp zero, smooth onset geometry, heading modes, ESS diagnostics, RNG-state preservation, inherited-baseline identity, replay-contract rejection, ordered/error-safe rollout prefetch, immutable encoder/context reuse, and strict partial-cache recovery. Together with the neighbor-alignment, X2-geometry, and visualization suites, the current focused audit run is 59 passing tests. The loader/alignment/observability subset was also rerun after separating replay and evaluation I/O (`52 passed`). An additional 26 preference/scenario-generation compatibility tests pass with the default canonical-loader API, whose behavior remains unchanged unless the new internal defer flag is explicitly requested by the packed AWR loader.
