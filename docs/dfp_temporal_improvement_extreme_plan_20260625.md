# Extreme temporal improvement plan for DFP integration

Date: 2026-06-25

## Current evidence

Current matched status at 2026-06-25 03:05 JST:

- DFP unified ego run: epoch 20 complete
- matched no-DFP baseline: epoch 7 running
- final matched report: pending

Observed so far:

- DFP best ego: 1.961324 at epoch 17
- DFP best lat: 0.193309 at epoch 18
- DFP best lon: 1.472875 at epoch 17
- DFP best neighbor: 3.916277 at epoch 20
- baseline best ego through epoch 7: 3.236562 at epoch 7
- baseline best lat through epoch 7: 0.241339 at epoch 5
- baseline best lon through epoch 7: 1.873129 at epoch 7
- baseline best neighbor through epoch 7: 4.358158 at epoch 2

This is not final proof because baseline has not reached epoch 20, but it strongly
suggests that history-conditioned DFP ego generation is useful.

## Strong conclusion

The current two-decoder architecture is not the best final architecture.

It is a good proof-of-effect architecture because it maximizes checkpoint reuse and
isolates DFP ego behavior. But it does not maximize temporal modeling because:

- ego and neighbor futures are produced by different decoder stacks
- DFP temporal chunks interact with scene encoding, but not directly with neighbor future tokens
- temporal consistency is enforced inside one sample, not across consecutive samples
- chunk-level MLP projection hides intra-chunk per-step temporal structure
- final-epoch behavior is less stable than best-epoch behavior, which indicates loss/schedule issues

## Ranking of next architectures

### Level 0: Current two-decoder unified ego

Status: implemented and running.

Purpose:

- prove that DFP-style history/current/future chunk forcing improves ego trajectory
- preserve original DP neighbor path

Weakness:

- duplicated decoder stack
- weak ego-neighbor coupling
- not the final maintainable architecture

### Level 1: Shared-stack DFP ego

Status: prepared in `/mnt/nvme/Diffusion-Planner-dfp-shared-stack`.

Purpose:

- remove separate DFP block stack
- reuse original `DiT.blocks`
- keep original neighbor head and DFP ego head
- preserve checkpoint reuse

This is the best immediate next experiment because it tests whether the DFP gain can
survive a simpler single-stack architecture without rerunning baseline.

### Level 2: Joint agent-temporal token decoder

This is the best long-term architecture if we want the strongest temporal model.

Token sequence:

```text
[ego agent future token]
[neighbor future tokens]
[ego history chunk token]
[ego current chunk token]
[ego future chunk tokens]
```

One transformer stack processes all tokens jointly.

Output heads:

- original all-agent future head for neighbor
- DFP chunk head for ego
- optional original ego head as distillation/protection only

Stability mechanism:

- initialize original agent path from the pretrained checkpoint
- initialize DFP token adapters randomly
- add a zero-initialized gate from DFP tokens into agent tokens
- optionally freeze or low-LR original blocks during warmup

Why this is better:

- ego temporal chunks can attend to neighbor future tokens
- neighbor prediction can condition on ego temporal intent after the gate opens
- still preserves original DP model as the initial behavior

Risk:

- more complex than Level 1
- can damage neighbor metrics if gates/loss ramp are not conservative

### Level 3: Route-sequential latent memory

This is the strongest temporal direction, but it changes the data/training protocol.

Instead of independent NPZ samples, train on consecutive route windows and carry a
latent memory:

```text
memory_t = update(memory_{t-1}, encoder_t, predicted_ego_t, observed_history_t)
prediction_t = decoder(encoder_t, memory_t)
```

This directly addresses temporal flicker and closed-loop consistency.

Risk:

- requires sequence dataloader
- baseline fairness is harder
- W&B comparison must define steps/epochs carefully

This should be a second-stage project after Level 1/2 confirms DFP benefit.

## Highest-value training objective changes

### 1. DFP loss ramp

Current DFP reaches best ego at epoch 17 and worsens by epoch 20. This suggests DFP
loss and original planner loss are not perfectly balanced.

Use:

```text
effective_dfp_lambda = target_lambda * min(1, epoch / ramp_epochs)
```

Recommended first setting:

- target lambda: 0.3 for two-decoder
- target lambda: 0.1 for shared-stack
- ramp epochs: 5

### 2. Boundary continuity loss

DFP chunks should not only match per-step positions. Chunk boundaries must be smooth:

```text
loss_boundary = L2(chunk_i[-1] - chunk_{i+1}[0])
```

For ego future:

- compare future chunk boundary position
- compare heading continuity
- optionally compare velocity continuity

This targets temporal discontinuity directly.

### 3. Velocity / acceleration / curvature consistency

Current validation mostly sees position losses. Add auxiliary temporal losses:

- velocity finite difference
- acceleration finite difference
- yaw-rate or curvature finite difference

These should have small weights to avoid oversmoothing:

```text
lambda_velocity = 0.05
lambda_accel = 0.02
lambda_curvature = 0.01
```

### 4. Cross-sample consistency

If consecutive NPZs are available, enforce:

```text
prediction_at_t shifted by k frames ~= prediction_at_t+k
```

This is the most direct way to reduce temporal flicker without requiring a recurrent
model. It requires careful coordinate transform between consecutive ego frames.

This can reuse existing step3 data if corresponding consecutive entries exist, but
it needs exact route/frame metadata.

## Immediate next experiments

Run only after current matched baseline/exact queue finishes.

### Experiment A: shared-stack DFP lambda 0.1

Purpose:

- test single-stack simplification
- protect neighbor path

Expected outcome:

- ego improvement smaller than two-decoder but still positive
- neighbor not worse

### Experiment B: shared-stack DFP lambda 0.3

Run only if A preserves neighbor.

Purpose:

- test whether full DFP strength survives shared stack

### Experiment C: two-decoder DFP with loss ramp

Purpose:

- test whether epoch20 degradation versus epoch17 is schedule-related

### Experiment D: shared-stack with temporal smoothness losses

Purpose:

- directly target trajectory temporal quality, not just endpoint position metrics

## Current best practical answer

The best immediate path is not to discard the current model. Keep it as the first
validated DFP win candidate, then test shared-stack as the maintainable version.

If shared-stack matches the ego gain, use shared-stack.

If shared-stack loses too much ego gain, move to joint agent-temporal tokens with
zero-gated DFP-to-agent interaction.

If final metrics show DFP helps ego but hurts neighbor, do not reject DFP; add
neighbor protection and loss ramp first.
