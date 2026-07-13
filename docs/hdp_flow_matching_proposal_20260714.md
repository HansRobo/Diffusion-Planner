# Proposal: Flow Matching for the Hyper Diffusion Planner (HDP → "HFP")

**Date:** 2026-07-14
**Scope:** Replace the VP-SDE diffusion formulation of the current HDP branch (temporal ego-only DiT,
velocity latent, hybrid loss) with conditional flow matching (rectified-flow / linear-interpolation
path), while keeping the scene encoder, latent representation, hybrid objective, turn-indicator head,
and the RL-hybrid stage structurally unchanged.
**Status:** Phase 0 implemented on `feature/hdp-flow-matching` (`--diffusion_path linear_fm`); Phase 1 A/B training not yet run.

---

## 1. Executive summary

- **Feasible: yes, and cheaply.** Because the current model is x_start-parameterized, flow matching
  can be introduced as a *schedule + sampler swap* rather than a model redesign. The DiT keeps the
  exact same I/O contract (noisy latent + time + scene → predicted clean velocity latent), the HDP
  hybrid loss and the turn head consume `x̂0` exactly as today, and the RL reward-weighted objective
  ports with a two-line change. Estimated core diff: **~300–500 lines + tests**.
- **Expected outcome:** open-loop quality ≈ parity with the current diffusion model; the concrete,
  near-certain win is **inference cost — 7 decoder forwards → 2–4 (potentially 1 after a
  distillation phase)** — plus simpler solver math and a cleaner ONNX graph. External evidence on
  nuPlan/NAVSIM suggests closed-loop quality is at worst neutral and possibly slightly better.
- **Recommendation:** run it as a controlled A/B (Phase 1 below) using the x̂0-parameterized variant,
  which minimizes deviation from the current recipe. Treat quality parity as the gate, and treat
  NFE reduction (latency) as the prize.

---

## 2. External evidence (deep-research digest)

| Work | Setting | Relevance to us |
|---|---|---|
| **FlashPlanner** (OpenReview 2026) | Flow-matching planner, nuPlan closed loop, "globally consistent velocity field", **one-step** generation | Beats the upstream **Diffusion Planner** (our fork base) by **+5.56% Test14-hard (reactive)** and **+2.04% Val14-hard**, at **166 FPS vs 13 FPS**. The single most comparable data point: same benchmark, same baseline family. |
| **GoalFlow** (CVPR 2025) | Goal-conditioned flow matching for multimodal trajectories, NAVSIM | PDMS **90.3** (SOTA at publication) with **a single denoising step**; explicitly motivated by the trajectory-divergence problem of diffusion at low NFE. |
| **π0** (Physical Intelligence, 2024) | Flow matching for robot **action chunks** at 50 Hz | Production-grade precedent that FM works well on exactly our kind of latent: short sequences of continuous actions, strongly conditioned, few-step sampling. |
| **FlowDrive** (arXiv 2509.21961) | Moderated flow matching + data balancing for trajectory planning | FM applied to driving trajectories with attention to data imbalance; corroborates FM as the current default for new planners. |
| **Stable Diffusion 3 report** (Esser et al. 2024) | Rectified flow vs diffusion at scale (images) | Systematic comparison: rectified flow with **logit-normal time sampling** ≥ diffusion schedules. Motivates our time-sampling ablation. |
| **MeanFlow / Shortcut models / Consistency-FM / Rectified MeanFlow** (2024–2026) | One/few-step generative models built on FM | The upgrade path to **1 NFE** is much better developed for FM than for VP-SDE diffusion. This is the strategic reason to switch even at quality parity. |

Two honest caveats from the same literature:

1. Quality gains of FM over well-tuned diffusion at **moderate NFE (≥6)** are small; most reported
   wins come from the low-NFE regime (1–4 steps) where diffusion degrades faster.
2. Papers that beat Diffusion Planner (FlashPlanner) change more than the probability path (training
   objective, redundancy reduction), so their margins are an upper bound, not an expectation, for a
   pure path swap.

---

## 3. What we have today (precise baseline)

Per-scene latent `a ∈ R^{80×4}` = normalized ego velocity sequence `(Δx, Δy, cosθ, sinθ)`.

- **Forward corruption:** `x_t = α_t·x0 + σ_t·ε`, VP-SDE linear (β: 0.1→20), `t ~ U[1e-3, 1]`
  (`decoder.compute_training_loss`, `sde.VPSDE_linear`).
- **Model:** temporal DiT predicts `x̂0(x_t, t, scene tokens, ego velocity, route AdaLN)`.
- **Loss:** `MSE(x̂0, x0)` (per-step, summed over 4 channels) `+ ω·L_waypoint(x̂0)` with ω = 0.01 and
  detach window W = 10; turn head trained on detached `x̂0`; penalties off by default.
- **Sampler:** DPM-Solver++(2M), 6 steps (logSNR spacing) + denoise-to-zero = **7 NFE**,
  deterministic given the initial latent; temperature = scaling of the initial noise
  (0.1 multisample eval / 0.5 eval policy / 1.5 RL rollouts).
- **RL-hybrid:** groups of N rollouts from the EMA policy (7 NFE each), HDP reward,
  `exp(β·normalized reward)` weights on the same x0-reconstruction loss at a fresh random `t`
  (`hdp_rl_utils._compute_policy_ego_loss_per_sample`).

Note: the base branch (`fix/all-data-model-quality-fixes`) had a v-prediction rectified-flow mode
(`flow_matching_utils/ode_solver.py`, removed from the live path on this branch). It was built for
the old agent-token decoder with waypoint latents and no hybrid loss — design precedent, not
reusable code.

---

## 4. Proposed formulation

### 4.1 Probability path

Keep the branch's time convention (t=0 data, t=1 noise) to minimize code churn:

```
x_t = (1 − t)·x0 + t·ε,     ε ~ N(0, I),   t ∈ [ε_t, 1]
conditional velocity  u(x_t, t | x0) = ε − x0 = (x_t − x0) / t
```

This is the conditional-OT / rectified-flow path: straight lines from data to noise.

### 4.2 Parameterization — keep x̂0-prediction (key decision)

**Option A (recommended): the model keeps predicting `x̂0`.** FM training then differs from today
*only* in the corruption coefficients:

```
today:  x_t = α_t·x0 + σ_t·ε     (VP-SDE)      loss: MSE(x̂0, x0) + ω·L_wp(x̂0)
FM:     x_t = (1−t)·x0 + t·ε     (linear)      loss: MSE(x̂0, x0) + ω·L_wp(x̂0)   ← unchanged
```

This is mathematically equivalent to velocity-field regression up to a t-dependent loss weighting
(`u_θ = (x_t − x̂0)/t` ⇒ `‖u_θ − u‖² = ‖x̂0 − x0‖²/t²`), i.e. x̂0-MSE = FM loss with λ(t) = t².
Everything downstream of `x̂0` — hybrid waypoint loss, turn-indicator latent decode, RL pseudo-GT
reconstruction, all validation metrics — is untouched.

**Option B: v-prediction** (predict `u`), recovering `x̂0 = x_t − t·û` where needed. More standard in
the literature and the natural form for MeanFlow/consistency-FM objectives, but it changes the
output semantics, loss scales, and the checkpoint bootstrap story for zero benefit in Phase 1.
Defer to Phase 3 (distillation) if pursued at all.

### 4.3 Sampler

Euler on the ODE `dx/dt = (x_t − x̂0)/t`, which in x̂0 form is a numerically trivial interpolation:

```
x_{t'} = (t'/t)·x_t + (1 − t'/t)·x̂0(x_t, t)        (t' < t)
```

- Coefficients are in [0, 1] — no `expm1`/logSNR machinery, no blow-up near t→0 (the 1/t in the
  velocity form cancels).
- Uniform time spacing on `[1, ε_t]`; final step = take `x̂0` directly (the exact analogue of the
  current denoise-to-zero).
- Optional midpoint (2nd order) variant for the 4–6-step configs; at 1–2 steps Euler is standard.
- Sampler NFE becomes a pure knob: `k` steps = `k` decoder forwards (vs `steps + 1` today).

### 4.4 Temperature, multimodality, warm start

- **Temperature:** identical mechanism — scale the initial noise `x_1 = s·ε`. Semantics shift
  slightly (a different path geometry), so `multisample_eval_noise_scale`, `rl_noise_scale`,
  `rl_eval_noise_scale` need a re-sweep, not a copy.
- **Warm start (deployment, later):** FM makes partial re-noising clean:
  `x_{t_w} = (1−t_w)·a_prev + t_w·ε`, integrate `t_w → 0` in 1–2 steps. Relevant to the known
  closed-loop warm-start investigation on the deployment branch; not part of this proposal's scope
  but a follow-up FM enables nicely.

### 4.5 Time sampling

Default `t ~ U[ε_t, 1]` for the controlled A/B. Add `logit_normal(0,1)` as a config option
(`diffusion_time_sample_method`) — the SD3 report found it consistently better for rectified flow;
cheap ablation.

---

## 5. Implementation plan (file by file)

The trick that keeps the diff small: the training code only consumes the schedule through
`sde.marginal_alpha(t)` / `sde.marginal_prob_std(t)`. A linear-path object satisfies the same
interface.

| File | Change |
|---|---|
| `model/diffusion_utils/sde.py` | Add `LinearFlowPath` with `marginal_alpha(t)=1−t`, `marginal_prob_std(t)=t` (plus `T=1`). Training-side noising in `compute_training_loss` and `_compute_policy_ego_loss_per_sample` then works **unchanged** via duck typing. |
| `model/diffusion_utils/fm_solver.py` (new, ~60 lines) | k-step Euler/midpoint sampler in x̂0 form (§4.3), same `model_kwargs` plumbing as `dpm.model_wrapper`. |
| `model/module/decoder.py` | `diffusion_type: Literal["vpsde_x_start", "fm_x_start"]`; select path object in `__init__`; `_inference_x_start` dispatches to the FM solver in FM mode. Everything else (turn head, route AdaLN, `_sample_steps`) unchanged. |
| `train_config.py`, `train_predictor.py`, `train_hdp_rl_predictor.py` | New flag + argparse; **add it to the strict resume-compat `training_fields` list in `train.py`** (do not repeat the `rl_reward_w_road_border` omission), and to `assert_checkpoint_compatible` architecture checks so a VP-SDE checkpoint can never be silently resumed as FM. |
| `utils/hdp_compat.py`, `utils/config.py` | Guard: deployment/validation `Config` must read the flag from `args.json`; refuse mismatched checkpoints (same pattern as the velocity-representation guard). |
| `hdp_rl_utils.py` | `_compute_policy_ego_loss_per_sample`: use the model's path object (already fetched via `getattr(model_ref, "sde", ...)`) — effectively zero-change if the decoder exposes the FM path under the same property. `sample_group` unchanged (solver is behind the decoder interface). |
| `utils/onnx_export.py` | No structural change — the full graph re-traces the new (simpler) k-step loop. Turn-indicator and encoder graphs identical. |
| Tests | Path-coefficient unit tests; Euler-step identity test (one step from t with exact x̂0=x0 must land on x0); overfit smoke test; NFE-monotonicity test; checkpoint-guard tests. |

**Weight warm start:** the DiT I/O contract is identical (`x_t, t → x̂0`); only the marginal
distribution of `x_t` changes. Initializing FM training from the current Base EMA checkpoint
(`init_weights_path` + a new `allow_vpsde_to_fm_init` escape hatch, mirroring
`allow_waypoint_to_velocity_init`) is likely to substantially cut convergence time — worth an arm in
Phase 1.

---

## 6. Will it improve performance? (honest assessment)

| Axis | Expectation | Basis |
|---|---|---|
| Open-loop (minADE/FDE, valid_loss_ego) | **≈ parity** (±small) | Strongly conditioned, low-dim latent; at 6–7 NFE the current sampler is not the bottleneck. SD3-scale evidence favors FM slightly; driving papers show parity-to-better. |
| Closed-loop (EPDMS, collisions, grouped eval) | **parity to small gain** | FlashPlanner (+2.0–5.6% over Diffusion Planner on nuPlan closed loop) and GoalFlow (SOTA NAVSIM PDMS with 1 step) — but both bundle extra tricks; a pure path swap should expect the lower end. |
| Inference latency | **~2–3.5× fewer decoder forwards now (7 → 2–4), up to 7× after distillation (→1)** | FM degrades far more gracefully at low NFE (straight paths); the 1-step upgrade path (MeanFlow/shortcut/reflow) is FM-native. This also proportionally cuts RL rollout cost (7 NFE × N candidates today) and multisample validation cost. |
| Training | Simpler and equally stable; hyperparameters (lr, ω=0.01, W=10) expected to transfer under Option A | Loss is the same x̂0-MSE, just a different corruption distribution. |
| Risks | (1) deviation from the official HDP recipe → paper-comparability lost; (2) implied t-weighting changes what the model prioritizes (mitigate: λ(t) / logit-normal ablation); (3) temperature knobs need re-sweeping, incl. `rl_noise_scale` group diversity; (4) turn head sees slightly different `x̂0`-at-random-t statistics (it retrains with the model anyway); (5) one more config axis to guard in strict-resume/ONNX/deployment. | — |

**Bottom line:** do not expect flow matching to make trajectories *better* at the current 7-NFE
budget — expect it to make the same quality **much cheaper**, unlock 1–2-step deployment, and
slightly improve the low-NFE robustness that matters for on-vehicle latency headroom. If the Phase 1
A/B shows parity on EPDMS/closed-loop, the switch pays for itself through latency and the
distillation runway alone.

---

## 7. Experiment plan

- **Phase 0 — correctness (≈0.5 day):** implement; unit tests; single-batch overfit (loss→~0 and the
  k-step sampler reproduces the overfit trajectory); NFE sanity sweep on an untrained model.
- **Phase 1 — controlled A/B (main gate):** train FM-Base with identical data/seed/schedule as the
  current Base (same 8-GPU bs512 recipe); two arms: (a) from scratch, (b) warm-started from Base EMA.
  Compare: `valid_loss_ego` (lat/lon), multisample minADE/minFDE@6, EPDMS total + subscores, turn
  accuracy, grouped closed-loop on the standard routes. **Gate: FM ≥ diffusion − noise on EPDMS and
  closed-loop.**
- **Phase 2 — NFE Pareto:** sweep FM at {1, 2, 4, 6, 10} steps vs diffusion at {2, 4, 6}; pick the
  deployment NFE; re-sweep eval temperatures at that NFE.
- **Phase 3 (optional) — one-step:** logit-normal time-sampling ablation; then reflow or
  MeanFlow/shortcut fine-tune targeting 1 NFE for the vehicle.
- **Phase 4 — RL port (hand off to the RL owner):** the reward-weighted objective is unchanged;
  swap the noising path, re-sweep `rl_noise_scale` ∈ {0.5, 1.0, 1.5} for group diversity
  (`reward_*_group_std` diagnostics already exist), keep the held-out eval on a fixed protocol so
  selection scores stay comparable.

## 8. References

- FlashPlanner — https://openreview.net/forum?id=NAXQWSwDUD
- GoalFlow (CVPR 2025) — https://arxiv.org/abs/2503.05689
- FlowDrive — https://arxiv.org/pdf/2509.21961
- π0: A Vision-Language-Action Flow Model — https://www.pi.website/download/pi0.pdf
- One-Step Diffusion via Shortcut Models — https://arxiv.org/pdf/2410.12557
- Rectified MeanFlow — https://arxiv.org/html/2511.23342v1
- Consistency Flow Matching — https://arxiv.org/abs/2407.02398
- Flow Matching (Lipman et al., 2023); Rectified Flow (Liu et al., 2022); SD3 report (Esser et al., 2024)
