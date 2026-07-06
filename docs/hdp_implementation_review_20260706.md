# HDP Implementation Review — `feature/hyper-diffusion-planner`

**Date:** 2026-07-06
**Scope:** HDP implementation (base `4956206` → working tree, i.e. commits `dbee266` + `6b20ca3` **plus all uncommitted changes**) **and** the pre-existing (original Diffusion-Planner) code in this repo, per request.
**Method:** 14 independent finder passes (10 review angles on the HDP diff + 4 full-file audits of the base code) → deduplication → 9 adversarial verification passes (every candidate CONFIRMED / PLAUSIBLE / REFUTED against the code, the vendored official HDP reference, and numeric checks) → final gap sweep. Raw process log: `docs/hdp_review_handoff_log.md`.
**Reference used:** HDP paper = arXiv 2602.22801 (Hyper-Diffusion-Planner); official code vendored at `reference/external/Hyper-Diffusion-Planner/HDP-nuplan`.

---

## Executive summary

**The HDP core is mathematically sound and paper-faithful.** All 16 `sde.transform` parameterization identities, the v/score/noise targets, DiT score scaling, VPSDE↔NoiseScheduleVP schedule consistency, and the model_type wiring into the DPM solver were verified numerically correct. The velocity representation (per-step displacement via `torch.diff`, heading copied), the hybrid loss `L_v + ω·L_wpt` with ω = 0.01, the gradient-detach-window integral, and the `exp(β·normalized reward)` objective all match the official HDP release — and this port even **fixes** an official-code bug (the official code double-transforms the score prediction before the hybrid integral). `normalization.json`'s `ego_velocity` stats are copied verbatim from the official release.

**The problems are at the integration seams, the defaults, and in pre-existing base code.** The dangerous theme: the "ego row of the latent is now a different representation" invariant is enforced by hand at each touch point, and several touch points were missed (ONNX turn-indicator wrapper, guidance stack, rlvr/DPO stacks, delay-prefix contract). Separately, flipping `use_velocity_representation` and `rl_objective` defaults in shared `TrainConfig` silently changes the behavior of every existing launch command and warm start. Finally, two **empirically confirmed** data bugs exist in the original DP pipeline (neighbor-future misalignment verified in 747/747 sampled cases in the live training corpus; goal_pose never transformed under augmentation).

Finding counts (verified): **1 critical, 12 high, 15 medium, ~21 low, plus 16 cleanup/efficiency items.** 14 suspicions were investigated and refuted (listed in the appendix so nobody re-chases them).

---

## 1. Critical

### C-1. Silent default flips corrupt every waypoint checkpoint that is resumed, warm-started, or RL-fine-tuned
`diffusion_planner/diffusion_planner/train_config.py:113` (`use_velocity_representation: bool = True`, was `False`) and `:122` (`rl_objective = "official_reward_weighted"`, an option that did not exist on tier4-main).
Both trainers pull argparse defaults from `TrainConfig`, and both build the model **from CLI args, never from the resumed checkpoint's `args.json`** (`train_grpo_predictor.py:557` builds → `:581` resumes; `train.py:301` builds → `:311` warm-starts). The flag creates no layers, so `load_state_dict` (strict) succeeds with identical keys/shapes — an old waypoint-space checkpoint loads cleanly into a velocity-mode model with `ego_velocity` stats (std 0.5 vs 20), and `sample_group`/`_latent_to_prediction` then cumsum-integrate waypoint-normalized outputs as displacements. **Garbage trajectories/rewards from step 0, no error.** Likewise, every pre-existing GRPO launch command silently trains the reward-weighted objective instead of GRPO.
**Fix direction:** make HDP opt-in (default False), or validate CLI flags against the resumed checkpoint's `args.json` and hard-fail on representation mismatch.

---

## 2. High

### H-1. GRPO's SFT anchor steps drop the HDP waypoint loss
`grpo_epoch.py:70-76` composes the total loss without the `args.planning_hybrid_loss * loss["ego_planning_hybrid_loss"]` term that `train_epoch.py:82-83` adds. In velocity mode `ego_planning_loss` contains **only** the velocity-diffusion term (decoder.py:180/252); the waypoint term lives solely in the separate key (decoder.py:254). RL steps *do* include it (`grpo_utils.py:382`). So supervised anchor steps inside GRPO training never backprop integrated-position accuracy while RL steps and the supervised trainer do — inconsistent objective, silent. (Found independently by 7 of 14 finder passes.)

### H-2. All pre-HDP checkpoints crash at model construction (`diffusion_sample_steps`)
`decoder.py:358` reads `config.diffusion_sample_steps` unconditionally; `utils/config.py` (29 lines) only `setattr`s keys present in the saved `args.json` and backfills nothing except `guidance_scale`. Any pre-HDP `args.json` → `AttributeError`. Entry points affected: `valid_predictor.py:85/128`, `diffusion_planner_ros/diffusion_planner_node.py:77-78`, `onnx_export.py:326-328` (via `torch2onnx.py`), `preference_optimization/model_utils.py:65/69`, ~20 rlvr tools. Loud failure, trivially fixed with a Config default table.

### H-3. Exported turn-indicator ONNX is doubly wrong
`utils/onnx_export.py:215-218` (`TurnIndicatorONNXWrapper`):
(a) pools with **plain `torch.mean` over all ~564 tokens** while the decoder uses a masked mean over valid tokens only (decoder.py:681-684; masked mean added in commit `795a76d`, which updated only decoder.py) — several-to-10× attenuation, affects **all** checkpoints, waypoint or velocity;
(b) slices `final_x0[:, 0, 1::10, :2]` raw from the latent — in velocity mode the head was trained on integrate→re-normalize waypoints, so it receives per-step displacements (~0.1–1.5) instead of waypoint-scale features.
The ONNX is auto-exported next to **every** checkpoint (`train.py:529/556`), and `torch2onnx.py:197` "validates" the split graph against the same broken wrapper, so the mismatch is self-consistent and never caught. `FullONNXWrapper` is unaffected; split-graph (external denoising loop) consumers are.

### H-4. Guidance stack is semantically broken for velocity mode and non-x_start model types
`model/guidance/composer.py:45-49`, `guidance_wrapper.py:59-64`: the "x_start correction" `model(x,t) − x` assumes DiT outputs x_start (false for noise/score/v), and the inverse-normalized ego row is treated as **positions** when it is per-step displacements — collision/road-border/lane energies evaluate geometry on ~1.5 m-from-origin phantom trajectories, and the cumsum chain rule is entirely missing from the gradient (composer.py:75-76). Nuance: the normalizer inverse *is* row-aware (no 40× scale error; the break is semantic). Guidance is off by default at inference (`Config.guidance_fn=None` → uncond) but actively used by rlvr `grpo_sampler(_batched)`/`batched_rollout` and `preference_optimization` — broken there for any velocity or non-x_start checkpoint.

### H-5. rlvr / preference_optimization stacks silently corrupt velocity checkpoints
`rlvr/grpo_loss.py:76-78,119-120`, `rlvr/grpo_sft_trainer.py` (`_compute_sft_diffusion_loss`), `preference_optimization/dpo_loss.py:47-49,119`: all build ego GT as normalized **waypoints** using the checkpoint normalizer's row 0 — which for HDP checkpoints holds **velocity** stats (exactly 40× off on x/y) — and MSE it against a model that outputs normalized velocity. Zero occurrences of `use_velocity_representation` under `rlvr/` or `preference_optimization/` (no guard anywhere). These stacks are live (commits through 2026-07-01); with the new default, every new checkpoint triggers it. Fine-tuning through them silently destroys a velocity checkpoint.

### H-6. Prefix/delay columns get mathematically wrong targets in the noise/score/v supervision arms
`decoder.py:136` pins prefix columns to clean GT (`xT = where(prefix_mask, all_gt, xT)`), but supervision still targets the never-applied `z` (noise, `:163-166`), `v(z)`, or score (`:159/193/210`). No mask excludes the prefix columns from the ego loss (`:252` averages over the full horizon incl. columns 1..delay). Worst case (`model_type=x_start`, `supervision=noise`) the transform divides by `σ(ε)=0.0105` → ~95× amplified residuals containing pure unpredictable noise, on the control-critical first ≤5 steps, every batch. Identical duplicate in `grpo_utils.py:340/365-368/383-411`. **The default x_start arm is unaffected** — this corrupts exactly the `diffusion_supervision_type` experiment arms the branch was built to compare.

### H-7. 1/α amplification landmine for noise/score/v model types
`decoder.py:151` reconstructs `pred_x_start = (x_t − σ·ε̂)/(α+1e-6)`; at t≈1, 1/α ≈ 152. This feeds the always-on hybrid waypoint loss (cumsum over 80 steps, then squared — at init the loss is ~1e5–1e6 and dominates everything even after ω=0.01) and the penalty geometry, with no SNR weighting or clamping. Grad-clip bounds magnitude, not direction. Note: this design is inherited verbatim from the official HDP reference (also no SNR weighting); default x_start is unaffected (`pred_x_start = model_output`). Any `diffusion_model_type=noise/score/v` + velocity run is at high risk of noise-dominated gradients.

### H-8. Hardcoded "final 10 epochs" LR override crushes short runs
`train.py:430-441`: at the top of **every** epoch, if `epoch >= train_epochs − 10` the LR is forced to `0.1×`/`0.01×` base — for `train_epochs ≤ 10` this applies from epoch 0, so a short warm-start fine-tune (exactly the use case `--init_weights_path` adds) silently trains at 1/10–1/100 of the configured LR for its entire run, clobbering the warmup scheduler. Current launchers use 60 epochs (override hits 50–59, plausibly intended), so this bites the next person who runs a short job. *(Pre-existing base bug.)*

### H-9. `--use_ema false` crashes supervised training
`train.py:314-315` assigns `model_ema` only under `if args.use_ema:`, but it is referenced unconditionally at `:342` (resume), `:446-448` (train_epoch call), `:511` (checkpoint save) → `NameError`. (`train_grpo_predictor.py` has the mirror wart: EMA is unconditional and its `--use_ema` flag is silently ignored.) *(Pre-existing.)*

### H-10. All closed-loop wandb logs are silently dropped in supervised runs
`train.py:193` (`closed_loop_validate`) logs with explicit `step=epoch+1`, but the run's auto-step has already advanced past that (stepless epoch-0 log `:409-422`, stepless per-epoch log `:478-491` — and thousands of steps ahead if `wandb_step_log_interval>0`). wandb silently drops explicit-step logs below the current step → closed-loop scalars and rollout videos never appear. Also: `args._wandb_global_step` is never checkpointed; on `--wandb_run_id` resume the charts collide.

### H-11. Augmentation never transforms `goal_pose` *(pre-existing base bug, live in every current run)*
`utils/data_augmentation.py` (and the bridge variant): zero references to `goal`. `centric_transform` rotates/translates ego, neighbors, lanes, route, polygons, line_strings, static_objects — but `goal_pose`, which the encoder consumes as a spatial token (`encoder.py:205/247/262`), stays in the pre-perturbation frame. With defaults (`use_data_augment=True, augment_prob=0.5`, confirmed in the live run's args.json), ~half of moving-ego samples train with a goal token inconsistent with the map by up to ~0.2 rad / 0.75 m (≈20 m lateral at 100 m ahead). The model is being trained to distrust the goal input.

### H-12. Neighbor GT futures are one step late for every short-tracked neighbor *(pre-existing base bug, empirically confirmed)*
`ros_scripts/parse_rosbag.py:365-388` and `cpp_tools/.../neighbor_processor.cpp:160-165` both seed the future deque/history with the **current-frame** state; the seed is only evicted if the agent persists all 80 future frames. Ego futures start at t+0.1 s (`frame_processor.cpp:144-148`). **Verified on the live training corpus: in 50 sampled npz, 747 of 747 short-tracked neighbor futures have `future[0] == past[-1]`** — the entire neighbor GT trajectory lags reality by 0.1 s (≈1 m at 10 m/s) and duplicates an input frame. Affects neighbor prediction loss, collision penalties, and eval metrics.

---

## 3. Medium

| # | Where | Finding |
|---|-------|---------|
| M-1 | `decoder.py:117, 429-431` + `normalization.json` | **Turn-indicator input scale**: in velocity mode, ego waypoints are normalized by *velocity* std 0.5 → features reach ~±240–480 (vs ~±4–11 in waypoint mode) into a bare `nn.Linear`. Train/inference are consistent (so it degrades, not breaks), but the head is ill-conditioned, and a warm-started waypoint-mode head receives inputs 40× outside its training distribution. Damage bounded: head is auxiliary. |
| M-2 | `train_utils.py:115-123`, `train.py:314→342`, `train_grpo_predictor.py:563→583` | **EMA seeded from random weights on resume**: `ModelEma` is constructed before `resume_model`; a bare `except` swallows a missing/failed `ema_state_dict` load, so resuming from a plain weights file leaves the EMA shadow contaminated (~5% random after 3k updates at decay 0.999) and the poisoned EMA is saved into every subsequent checkpoint. (The new `init_weights_path` path is ordered correctly — verified.) |
| M-3 | `train_predictor.py:100-105`, `train_grpo_predictor.py:287` | `--coeff_timestep` uses argparse `type=list` → splits the string into characters; the flag is unusable in every form (assert or `Tensor *= str` TypeError). *(Pre-existing.)* |
| M-4 | `utils/ddp.py:15-20` | torchrun branch unconditionally overwrites `MASTER_ADDR='localhost'` / `MASTER_PORT=args.port` before `env://` init → multi-node rendezvous impossible; concurrent single-node jobs collide on default port 22323 unless the launcher threads a unique `--port`. *(Pre-existing.)* |
| M-5 | `encoder.py:171-176` | Ego-history slice flip `[:6]`→`[-6:]` is **correct** (verified against converter time ordering: t=0 oldest; neighbor slice was already `[-6:]`; tier4-main's `[:6]` kept the *oldest* steps — a real bug this branch fixes). But nothing records the slice convention: pre-flip checkpoints warm-start cleanly and silently see their learned input slots moved. Note in release notes / bump a config marker. |
| M-6 | `utils/onnx_export.py:193-200` | `DecoderONNXWrapper` exports raw DiT output named `final_x0` with no model_type conversion and no velocity integration in the split-graph contract. Fine for default x_start; silently mislabeled for noise/score/v; velocity ego row lacks an integration op in the exported artifacts (FullONNXWrapper is correct). External C++/TensorRT consumers implementing the pre-velocity contract emit wrong ego trajectories. |
| M-7 | `model/guidance/collision.py:118, 156-157` | `CollisionGuidance.reward()` runs `torch.autograd.grad` under `@torch.no_grad()` → guaranteed `RuntimeError` on first call. Currently unreachable in production (only a test calls `compute_rewards`, without `collision`), but the composer docstring invites exactly this use. *(Pre-existing.)* |
| M-8 | `decoder.py:545-558` vs `:119-121,136` | **Delay/latency-compensation contract broken in velocity mode**: training pins the ego prefix in *velocity-latent* space; inference pins the caller's tensor verbatim and then cumsum-integrates it. The in-repo spec (`ros_scripts/test_delay.py:121`, `test_delay_onnx.py:138`) is waypoint-space → `output[:delay] == prefix` fails for delay>1 on velocity checkpoints. All in-repo runtime producers use delay=0 (safe); the external C++ node's delay feature is the exposure. |
| M-9 | `grpo_utils.py:244-262` | `compute_official_reward_weights`: (a) `'batch'` normalize computes mean/std on raw rewards *before* finiteness filtering — one NaN reward degrades the step to unweighted behavior cloning, silently; (b) the per-group `std>eps` gate applies in **all** normalize modes and **deviates from the official HDP**, which keeps zero-variance groups at weight `exp(0)=1` — with `grpo_noise_scale=0` ("deterministic") every group is zero-variance → **silent RL no-op**; (c) `n=1` → std NaN → silent no-op. |
| M-10 | `grpo_utils.py:85-88` vs `validate_model.py:252` | **Three coexisting sampling-init distributions**: RL rollouts `randn·U[0,3]`, validation/deployment `zeros`, VP prior `N(0,1)`. The official HDP is off-prior but *consistent* (`randn·0.5` everywhere). The reward-weighted objective is estimated on rollouts drawn from an init never used at inference. |
| M-11 | `grpo_utils.py:186-187` + `loss.py:7,569-571` | RL reward holes: stationary-ego GT (exactly-zero waypoints) is treated as padding → ADE forced to 0 → the `w_gt_l2` realism anchor vanishes precisely in stop scenes; and the collision penalty is evaluated at only 5 of 80 timesteps (`[0,20,40,60,79]`, cummax carries only *detected* hits) — a collision entered and exited between eval steps contributes zero penalty. *(Pre-existing.)* |
| M-12 | `decoder.py:88` + `loss.py:230-232` | `longitudinal_velocity` is read **after** ObservationNormalizer (channel std 20), so `clamp_min(|v/20·1.0|, 1.0)` = 1.0 below 20 m/s raw — the documented high-speed longitudinal-loss attenuation is a silent no-op on urban data (at 30 m/s it divides by 1.5, not ~30). *(Pre-existing — identical in base and in `/mnt/nvme/Diffusion-Planner`.)* Fixing changes long-standing behavior; decide deliberately. |
| M-13 | `utils/data_augmentation.py:195, 555-613` | `interpolation_future_trajectory` is 3-col-only and runs unconditionally (even for non-augmented samples) → **any 4-col future corpus crashes on the first batch**, which also makes the diff's new 4-col `centric_transform` branches unreachable dead code. Live corpus is 3-col (verified), so no current crash — but the moment the 4-col npz (`rlvr/autoresearch/tools/cpp_bin_to_npz.py`, scene-gen) meet augmentation, training dies. |
| M-14 | `utils/data_augmentation_bridge.py:1379-1445` | Bridge augmentation: same goal_pose omission; 4-col futures would be corrupted (cos overwritten with an angle-transformed value, sin untouched) or crash on the (T,3)→(T,4) assignment. Selectable via `--augment_type bridge`, non-default. |
| M-15 | `train_config.py:118` + `decoder.py:592` | `diffusion_sample_steps=25` vs 10 in both the official HDP reference and the pre-HDP code (hardcoded) → NFE 26 vs 11 ≈ **2.4× inference cost** for deployment ticks, closed-loop eval, and every GRPO rollout — introduced by this branch with no recorded justification. |

---

## 4. Low / minor (all verified)

1. `grpo_epoch.py:87` checks key `ego_hdp_velocity_loss` but the SFT path emits `ego_hdp_diffusion_loss` (decoder.py:255) → the SFT-side metric silently never logs; wandb series averages GRPO steps only.
2. `train_epoch.py:81-83` vs `grpo_utils.py:382`: SFT weights ego terms `α_planning : ω` while RL uses `1 : ω` → effective hybrid ω diverges when `alpha_planning_loss ≠ 1` (default is 1).
3. `train_epoch.py:108`: always-true comprehension guard (`if key != "loss" or torch.is_tensor(value)`); epoch/global_step/lr each logged under two names.
4. `grpo_epoch.py:169-212`: constant-zero placeholder metric series (`abs_advantage` in official mode; `official_reward_weight_*` in grpo mode) logged to wandb as fake signal.
5. `train_grpo_predictor.py:650`: `agg.get("epdms_total", 0.0)` — `aggregate_valid_metrics` never returns that key (EPDMS lives in `epdms_means`) → `valid/epdms_total` is a constant-0.0 series forever. *(Gap-sweep finding, verified.)*
6. `train_grpo_predictor.py:598` + `:373`: GRPO wandb project silently changed from hardcoded `Diffusion-Planner-GRPO` to default `Diffusion-Planner` → RL runs land in the supervised project, interleaving metric namespaces. *(Gap-sweep finding, verified.)*
7. `utils/lr_schedule.py:8-13`: `warm_up_epoch=1` would pin LR at 0.1× forever (`LinearLR(total_iters=0)`); and `CosineAnnealingWarmUpRestarts` contains **no cosine phase at all** (post-warmup = constant). No current launcher uses 1; the name is a lie. *(Pre-existing.)*
8. `utils/normalizer.py:16-21`: `KeyError('ego_velocity')` for legacy normalization files — loud, informative guard (fine), but note the vendored reference file lacks the key; the silent-waypoint-stats scenario was refuted (no real call site lacks the flag).
9. `utils/normalizer.py:52-53`: `ego_velocity` leaks into `ObservationNormalizer` and every new `args.json` — runtime-benign (unknown keys skipped; no consumer iterates them), cosmetic pollution.
10. `guidance_wrapper.py:74-76`: all-NaN-energy fallback builds `torch.zeros` without a graph → `cond_grad_fn` raises instead of degrading to unguided sampling (composer's equivalent fallback is safe). Narrow path. *(Pre-existing.)*
11. `dpm_solver_pytorch.py:509-514`: `timesteps_masked` computed with byte-identical args as `timesteps` → prefix time-pinning at inference is a **no-op** (dead code; training's `mask_coeff~U[0,1]` covers the boundary so inference sits at the supported edge). Intent from commit `0b66691` was evidently a scaled schedule. *(Pre-existing.)*
12. `decoder.py:182-184`: velocity-branch neighbor supervision uses unmasked `waypoint_gt` while other branches use masked `all_gt` — verified **harmless** (`:243` filters per-(b,p,t) before the mean); consistency smell only.
13. `normalization.json` `ego_velocity` isotropic std 0.5 under-weights lateral per-step error (dy ~0.01–0.1 m/step) by 100–10000× in the squared velocity loss; official-faithful, heading channels + hybrid term partially compensate — watch lateral metrics (tuning risk, not defect).
14. `grpo_utils.py`: `num_generations=1` silently degenerates GRPO/official to a no-op instead of raising (validation gap).
15. `grpo_utils.py:75,114,441`: "world frame" docstrings — everything is actually consistently ego-centric (verified end-to-end); fix the docs.
16. `ros_scripts/parse_rosbag.py:165,877-899`: ego past/future saved as float64 (schema says float32) → loud dtype crash if training directly on parse_rosbag npz. Live corpus (C++ pipeline) is float32.
17. `utils/data_augmentation.py:415` + `train_epoch.py:57`: `pose_padding_mask` runs after `heading_to_cos_sin` (padded `(0,0,0)`→`(0,0,1,0)`) → ego-past padding never detected; 7.5% of live npz contain such rows; sub-meter corruption on augmented samples (plus the pre-existing `(0,0,1,0)` pseudo-pose quirk itself).
18. `utils/normalizer.py:70-77`: `inverse` computes its padding mask on normalized data — asymmetric with `__call__`, but a wrong zero needs raw==mean element-wise across all channels in float32 (measure-zero).
19. `decoder.py:597-599`: inference `prediction` rows for invalid neighbors contain inverse-normalized garbage (≈10±20 m); the loss side is fixed in this fork, the output side is not. Verified blast radius: `preference_optimization/visualization.py:88-95` plots phantom neighbors; all other consumers re-mask. *(Known upstream bug, still present on the output path.)*
20. `dit.py:123-134`: hardcoded `T=81` for preproj/t_embedder vs config-driven `output_dim` → any `future_len ≠ 80` breaks; effectively pinned today. *(Pre-existing.)*
21. `decoder.py:504-511` + `ode_solver.py:9-14`: flow-matching inference passes 3-D x into DiT's `assert x.dim()==4` → the branch has **never** been runnable (base failed on both x and t asserts; this branch fixed only t). Dormant (no config uses flow_matching).
22. Hygiene: `preference_optimization/model_utils.py:43` passes `weights_only=False` to `torch.load` (real pickle surface); `train.py:31` omits the arg (safe on the pinned torch 2.11 default, add explicitly for clarity).
23. `grpo_epoch.py:162` getattr fallback `'grpo'` contradicts the config default `'official_reward_weighted'`; `official_reward_beta` is getattr-guarded while `official_reward_normalize` is a direct attribute — latent inconsistency (no current caller triggers it).

---

## 5. Cleanup / efficiency / maintainability (all fact-checked)

**Top refactor — the SFT/RL loss duplication.** `grpo_utils._compute_policy_ego_loss_per_sample` mirrors ~90 lines of `decoder.compute_training_loss` (prefix setup, latent build, noising, `sde.transform`, and the 4-way supervision ladder — which itself appears 4× across the two files). The copies have **already diverged** (H-1, Low-2 are exactly such divergences). Extract one shared latent-encode/supervised-loss helper parameterized by the ego target; this eliminates the standing "SFT and RL silently optimize different objectives" failure mode.

Dead code / duplication:
- `loss.py:61-96` `diffusion_prediction_target` / `diffusion_prediction_to_x_start`: **zero callers**, duplicate `sde.transform` — delete.
- `loss.py:128` `hybrid_loss`: zero callers after the rewrite — delete.
- `hybrid_loss_components` computes `l_v` that both callers discard — and under default x_start it is element-for-element identical to `ego_diffusion_loss` computed three lines earlier (pure double compute).
- `decoder.py:269-271` re-implements `inverse_normalize_ego_state` inline, three lines below a call to the real helper.
- `decoder.py:434` `_latent_to_prediction(…, current_states)` — parameter unused (leftover of removed `add_current_xy`); remove from both call sites.
- `_train_config_default` byte-identical in both trainers → move next to `TrainConfig`.
- Three coexisting dim-expansion helpers (`sde.expand_dim`, `sde.reshape_time`, `dpm_solver.expand_dims`).
- `diffusion_time_sample_method`: 5 plumbing sites for exactly one implemented option (YAGNI).

Efficiency (hot paths):
- `sde.transform` lacks a `src == tgt` short-circuit → the **default** config runs an identity transform through a noise round-trip twice per step (~10 extra kernels over [B,33,80,4] in the autograd graph, plus a benign ~1.5e-4 bias with worst case on prefix columns). Add `if src == tgt: return input` and reuse `pred_x_start` when types match.
- `normalize_ego_state` / `inverse_normalize_ego_state` do `.to(device)` on CPU tensors per call (≥6 blocking H2D copies per train step; 6-8 per inference tick) → cache/register buffers once.
- `marginal_prob(torch.ones_like(all_gt…))` materializes a full [B,321,80,4] tensor just to read α (decoder.py:130, grpo_utils.py:336) — `marginal_alpha(t)` exists.
- `waypoint_gt.clone()` copies ~22 MB/step when only the ego row is rewritten; `dpm_loss = zeros → overwrite both slices` adds a wasted fill + two index_put backward nodes (use `torch.cat`).
- `_turn_indicator_trajectory_from_latent` and `_latent_to_prediction` each redo the ego inverse-normalize+cumsum per inference call, and the latter inverses the full latent including the row it overwrites.
- `load_weights_only`: `map_location='cpu'` avoids a transient per-rank GPU copy of the whole checkpoint.

Altitude / design:
- The "row 0 is a different representation" invariant is hand-enforced at every touch point; the missed ones are exactly the High findings (ONNX, guidance, rlvr, delay). Encapsulate a **latent codec** (encode GT→latent, decode latent→prediction, turn-indicator features) owned by the Decoder/StateNormalizer, and make `StateNormalizer` carry an explicit representation tag instead of `from_json` silently swapping row-0 semantics on a getattr'd flag.
- `turn_indicator_trajectories` input-dict key silently falls back to `gt_trajectories` — only correct in waypoint mode; derive it internally or assert the key when `_use_velocity`.
- `_wandb_global_step`/`_current_epoch`/`_train_epochs` smuggled through the `args` namespace → explicit train-state object, and checkpoint the step counter.
- DiT's score-mode `/(std+1e-6)` inside `forward` makes the module's output semantics model_type-dependent at a hidden layer (the parameterization matrix already lives in `sde.transform`); document or centralize.

---

## 6. What is verified sound (do not re-investigate)

- All 16 `sde.transform` conversions, v/score/noise target formulas, `marginal_prob`/`marginal_alpha`/`marginal_prob_std` consistency (α²+σ²=1 to 1e-10), DiT score scaling ≡ standard ε-loss, NoiseScheduleVP ≡ VPSDE_linear, correct `model_type` passed to `model_wrapper` (numeric verification).
- Paper fidelity: velocity definition, ego-only scope, hybrid loss & ω placement, detach-window Algorithm 1, group-normalized exp weighting, `ego_velocity` stats — all match the official release; the port fixes one official bug (score double-transform).
- x_start→x_start round trip is benign (~1.5e-4), **not** catastrophic; the in-place tensor ops in the new loss code are autograd-safe; `Diffusion_Planner.sde` delegates correctly; `find_unused_parameters=False` is safe for the supervised default path (full parameter walk); base→HDP checkpoints load cleanly through `load_weights_only` (no new state-dict keys; prefix reconciliation correct for all four combinations); `init_weights_path` EMA ordering is correct; SFT and RL construct the ego latent identically (incl. time-0 slot); frames are consistently ego-centric; neighbor-loss masking makes the unmasked `waypoint_gt` harmless; `compute_group_advantages` zero-variance behavior is standard GRPO; encoder pooling/masking guards are present and correct; epoch-0 wandb block is properly guarded.

## 7. Test gaps & recommended verification

No unit tests exist for any HDP core mechanism. Recommended, in order of value:
1. **Round-trip codec test**: `velocity_to_waypoints(waypoints_to_velocity(w)) == w` and normalized-latent encode/decode for both representations.
2. **`sde.transform` identity matrix test**: all src→tgt pairs vs closed-form on random tensors, incl. t≈ε and t≈1.
3. **ONNX ↔ torch parity test** for the split turn-indicator/decoder graphs on a velocity checkpoint (would have caught H-3; today's torch2onnx validation is self-referential).
4. **Delay-contract test on a velocity checkpoint** (`test_delay.py` currently encodes the waypoint-only contract).
5. **Trainer-loss parity test**: assert SFT (`train_epoch`), GRPO-SFT (`_sft_step`), and RL per-sample losses agree on the ego term composition for a fixed batch (would have caught H-1).
6. **CLI smoke test**: parse all argparse flags with non-default values (`--coeff_timestep`, `--use_ema false` crash today).
7. **Data audit assertion** in the converters: `neighbor future[0] != past[-1]` for tracked agents (H-12) and goal_pose covariance under augmentation (H-11).

## 8. Notes

- The working tree contains substantial **uncommitted** changes that are part of this implementation — commit them; this review covers them.
- `docs/hdp_review_handoff_log.md` contains the full per-verifier evidence (file:line quotes) behind every verdict in this document.
