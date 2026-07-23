# ConsistencyPlanner vs. HDP: Performance Audit

**Date:** 2026-07-18
**Scope:** SFT/model quality first. Inference latency is deliberately not treated as a
constraint. RL and ROS are out of scope for this comparison.

## 1. Source and verdict

The paper source and PDF are stored at:

```text
reference/papers/arxiv_2606.11569_consistencyplanner/
```

The primary source is `conference_101719.tex`; the supplementary schedule and training
table are in `Supplementary.tex`. The paper is [arXiv:2606.11569](https://arxiv.org/abs/2606.11569).

**Main verdict:** Do not replace the current HDP DPM-Solver policy with the paper's
single-step consistency policy in the production SFT branch. The paper's own ablation
reports:

| Decoder | Off-road rate | Collision rate | Progress |
|---|---:|---:|---:|
| ConsistencyPlanner | 2.09 | 2.77 | 93.72 |
| DiffusionPlanner, 10 DPM steps | 2.07 | 2.76 | 93.63 |

Thus the consistency model is faster, but the iterative diffusion model is slightly safer
on the paper's Waymax protocol. Since speed is not our limiting factor, the paper does not
justify a production consistency conversion for our branch.

The paper's other central contribution, temporal action tokens plus heterogeneous
conditioning, is already present in our current HDP decoder. It should be retained and
strengthened through controlled ablations rather than replaced.

## 2. What the paper actually implements

### 2.1 Scene representation

The paper converts observations into ego-frame vector objects:

- 40 road rectangles;
- 20 route rectangles;
- 127 neighboring vehicles plus ego;
- current object features rather than the richer 21-frame ego and six-frame neighbor
  history used by our Tier IV input contract;
- a type/class token for each object;
- a BERT-style encoder that produces a compact scene embedding.

Road and route vectors are compressed with Ramer-Douglas-Peucker into
`[x, y, w, h, psi, id]` rectangles. Neighbor objects use velocity in place of the id
field. This is a compact Waymax representation, not evidence that map detail should be
discarded in our Tier IV corpus.

### 2.2 Decoder and conditioning

The paper uses a temporal action sequence and a Transformer decoder:

1. Noisy action features are encoded as temporal action tokens.
2. Self-attention mixes future action timesteps.
3. The action tokens cross-attend to the scene embedding.
4. Route geometry is separately processed by an MLP-Mixer.
5. The route embedding is added to the diffusion-time embedding.
6. The resulting condition modulates the decoder with AdaLN-Zero.

Our current `DiT` already has the important quality-relevant structure:

- `[B, 1, 80, 4]` ego-only temporal action tokens;
- temporal self-attention over all 80 future steps;
- scene-token cross-attention in each block;
- a global ordered-route encoder added to the timestep condition;
- AdaLN modulation with zero-initialized residual gates;
- current ego velocity as an action-token condition.

Relevant implementation locations are:

- `diffusion_planner/diffusion_planner/model/module/dit.py`
- `diffusion_planner/diffusion_planner/model/module/decoder.py`
- `GlobalRouteEncoder` in `decoder.py`

The paper's compact BERT scene embedding should **not** replace our detailed scene-token
cross-attention without an ablation. Our lanes, line strings/road borders, goal, ego shape,
typed objects, ego history, and neighbor history carry information that the paper's
rectangle representation does not preserve.

### 2.3 Consistency objective

The paper writes the policy as:

```text
f_theta(a^t, t | s) = c_skip(t) a^t + c_out(t) F_theta(a^t, t | s)
```

and trains with an L1 target:

```text
E[d(f_theta(a + t_{m+1} z, t_{m+1} | s), a)]
```

using a Karras boundary schedule (`t_T=80`, `epsilon=0.002`, `T=40`, `rho=7`). The
published equation is a direct noisy-to-expert denoising target; it does not show an EMA
teacher, a pairwise student/teacher consistency target, or a separate closed-loop safety
loss. It is therefore not a drop-in replacement for our VPSDE/x-start/DPM objective.

Our current SFT objective is:

- normalized ego delta/velocity x-start MSE;
- `omega=0.01` integrated waypoint loss with detached integration window `W=10`;
- optional safety penalties, disabled by default in the unbiased Base/SFT contract;
- isolated three-state turn-intent head with no signal-history policy input;
- six-step DPM-Solver++ sampling.

The velocity/hybrid objective is better aligned with our fixed-rate vehicle action and the
HDP implementation than the paper's plain L1 waypoint-style action target.

## 3. Direct comparison

| Area | ConsistencyPlanner | Current HDP branch | Performance judgment |
|---|---|---|---|
| Future representation | Temporal action tokens | Temporal 80-token ego action | Already adopted; keep |
| Scene encoder | Compact BERT/vector rectangles | Rich Tier IV typed token encoder | Ours has more usable map/history information; do not compress blindly |
| Route condition | Separate route MLP-Mixer into AdaLN | Full route tokens plus global route MLP-Mixer into AdaLN | Ours is a strict superset of the useful path |
| Decoder fusion | Action self-attention + scene cross-attention + AdaLN | Same, with six larger DiT blocks and detailed scene keys | Already adopted; no structural replacement justified |
| Action target | Direct noisy action to expert with L1 | x-start velocity/delta MSE + hybrid waypoint loss | Ours is better suited to smooth vehicle motion; paper L1 is an ablation only |
| Sampler | One-step consistency | Six-step DPM-Solver++ | Paper data favors diffusion slightly on safety; test more steps before reducing them |
| Training schedule | OneCycleLR, 20 epochs | Five-epoch warmup then current HDP schedule | Worth an LR ablation, not an automatic transplant |
| Safety objective | Closed-loop evaluation, no trajectory penalty in the described loss | Map input plus independent metrics; optional border loss | Keep unbiased SFT; use safety terms only with measured ablations |
| Closed-loop protocol | 8-second Waymax rollouts, IDM non-ego agents | Tier IV data and HDP validation/RL stack | Borrow the sequence-level diagnostic idea, not the simulator assumptions |

## 4. What is worth borrowing

### A. Increase DPM sampling steps for quality (highest-confidence, no retraining)

The paper's own comparison says 10-step diffusion is marginally better than its
consistency model on safety. Our model is a continuous-time x-start predictor, so the
first experiment should compare the same final checkpoint at 6, 10, and 12 DPM steps with
identical seeds and validation scenes.

This changes inference/evaluation only, not the learned weights. Because latency is not a
constraint for this decision, 10 or 12 steps should be the default candidate if closed-loop
road-border, collision, DAC, and progress metrics improve. The comparison must use:

- the same checkpoint and EMA policy;
- the same noise scale and seed;
- the same valid list;
- both deterministic and six/multi-sample metrics;
- sequential closed-loop evaluation, not only per-frame loss.

Do not infer the answer from the paper's 10-vs-1 comparison alone; it is a different data
distribution and architecture.

### B. Karras-shaped time coverage as an isolated SFT arm

The paper concentrates its consistency training on a Karras boundary schedule. Our current
`sample_diffusion_time` only supports uniform normalized VPSDE time. A useful experiment is
to add a `karras`/importance-sampled time mode **in the current VPSDE parameterization**, not
to copy the paper's raw `[0.002, 80]` time values. The mapping must be through the current
SDE's noise or log-SNR coordinate, with the same DPM schedule used at evaluation.

The safe experiment is:

1. keep x-start velocity and the hybrid loss unchanged;
2. sample a Karras-shaped distribution over current VPSDE noise levels;
3. log loss by time/noise bin;
4. select by closed-loop safety and trajectory quality;
5. discard the arm if it improves only denoising loss but worsens red-light, border, DAC,
   or progress behavior.

This is a retraining experiment, not a patch to the active run. There is no evidence in the
paper that Karras sampling alone improves our HDP objective.

### C. Consistency as a teacher-student auxiliary regularizer (separate branch only)

If the 10/12-step DPM baseline is strong, a consistency loss can be tested without changing
the production sampler:

- keep the current x-start MSE and hybrid waypoint loss as the dominant objective;
- evaluate the same scene/action at two adjacent noise levels;
- match the lower-noise prediction to an EMA stop-gradient target from the higher-noise
  prediction, or match both to the expert with a small robust coefficient;
- use the EMA weights only as a teacher, never as an additional policy reward;
- validate whether the regularizer improves multimodal candidate quality and closed-loop
  safety.

This avoids replacing the calibrated diffusion score with a potentially biased auxiliary
objective. It must not be enabled in the main Base run without an ablation.

### D. Attention-fusion ablation, not a rewrite

The paper's MLP-vs-Transformer ablation supports attention for safety (OR 2.09 vs 2.39 and
CR 2.77 vs 2.82). Our temporal DiT already has both self- and cross-attention. The only
reasonable quality ablation is a small global scene readout added to AdaLN alongside the
route condition, while retaining full scene-token cross-attention. This should be tested
against the current route-only AdaLN condition; it is not justified as a mandatory change.

## 5. What should not be copied

1. **Single-step consistency as the main SFT model.** The paper's speed advantage is
   irrelevant to our objective, and its own diffusion comparison is slightly safer.
2. **Compact 40/20 rectangle scene input.** It would throw away Tier IV map and history
   detail and could worsen the exact road-border problem we are trying to solve.
3. **Plain L1 action loss replacing velocity/hybrid supervision.** There is no evidence this
   preserves our displacement scale, heading representation, or border quality.
4. **Paper's raw time constants.** Their `t_T=80` is a consistency-model noise scale, not
   our normalized VPSDE time `t in [1e-3, 1]`.
5. **Paper's Waymax/IDM conclusions as direct real-vehicle evidence.** The metrics and agent
   controller differ from our Tier IV data and deployment stack.
6. **Removing ego/neighbor history.** Their compact current-state representation is a data
   choice, not proof that history is harmful for our noisy perception corpus.

## 6. Recommended experiment order

1. **No retraining:** evaluate current Base/SFT checkpoints at DPM steps 6 vs 10 vs 12.
2. **No architecture change:** select the best step count using closed-loop border,
   collision, DAC, progress, comfort, and red-light metrics.
3. **One retraining arm:** current HDP objective plus Karras-shaped VPSDE time sampling.
4. **One retraining arm:** current HDP objective plus a small EMA consistency regularizer.
5. **One architecture arm:** pooled scene readout into AdaLN, with full scene tokens kept.
6. Only promote an arm if it wins on held-out sequential evaluation; open-loop loss alone is
   insufficient.

## 7. Bottom line

The paper validates choices we already made: temporal future tokens, route-conditioned AdaLN,
and attention-based heterogeneous fusion. Its novel consistency sampler is primarily a speed
innovation and is not the best default for our performance-first objective. The highest-value
action is to spend the available compute on better DPM sampling/evaluation and controlled
noise-training/consistency ablations, while keeping the current ego-only velocity/hybrid HDP
model as the reference policy.
