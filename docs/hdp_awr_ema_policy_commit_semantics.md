# HDP EMA 0.05: what is published and what is our adaptation

Updated: 2026-07-19

## Correction and source boundary

Three artifacts must not be conflated:

1. The HDP paper says that EMA is used for policy updates and lists
   `EMA = 0.05` in its RL hyperparameter table. It does **not** specify whether
   `0.05` is the new-policy update rate, the old-policy decay, a per-minibatch
   decay, or an epoch-boundary acceptance transaction.
2. The clean public repository
   `ZhengYinan-AIR/Hyper-Diffusion-Planner` at commit `1ec9bd4` sets
   `use_ema: false` in the NAVSIM training configuration. Its `ModelEma`
   initialization is commented out, so the released NAVSIM RL path does not
   provide executable semantics for the paper's `0.05` value.
3. The local Tier IV hyper-diffusion-planner branch added
   `commit_ema_policy_update` in commit `8b20022` on 2026-07-12. That function
   interprets `0.05` as an epoch-boundary new-policy rate:
   `accepted = 0.95 * old + 0.05 * proposal`, then copies the accepted policy
   back to the live model and clears Adam state.

The third item is a useful and testable team implementation, but it is not an
exact behavior demonstrated by the clean public code. Future reports must call
it the **conservative accepted-policy interpolation** or the **local T4
interpretation**, not “official/exact HDP policy EMA.”

## Why we are evaluating the local interpretation for Original DP

The first full-data Original-DP AWR cycle used conventional per-minibatch EMA
with decay `0.999`. A replay epoch contains 3,547 optimizer steps, leaving

`0.999^3547 = 0.0288`

of the initial behavior policy in the final shadow. Thus roughly 97.1% of the
sequence of intermediate proposals enters the shadow. This is much less
conservative than a single 5% interpolation of the final proposal and is not
equivalent to one, even if a per-step decay is tuned to match a final scalar
coefficient.

The empirical reason to test a 5% boundary interpolation is stronger than the
attribution to HDP: Cycle 1's raw `1e-6` replay checkpoints overshot the fixed
train selector, while the post-hoc checkpoint

`source + 0.05 * (epoch_4_proposal - source)`

improved both the fixed 65,536-scene train selector and the independent full
46,262-scene deployment evaluation. Epoch 12 is therefore a matched-cache
test of this conservative Original-DP update, not a relabelled public-HDP
reproduction.

## Original-DP implementation contract

`rlvr.train_awr::_commit_epoch_ema_policy_update` performs one explicit
transaction:

- keep the behavior/reference policy fixed while optimizing the live proposal;
- at the epoch boundary, blend trainable parameters with the configured rate;
- copy frozen parameters and buffers exactly;
- load the accepted state into both behavior and live policies;
- clear optimizer state and gradients;
- record proposal-relative and accepted-relative parameter L2.

It is enabled only by:

```text
ema_per_epoch=true
ema_commit_live_policy=true
```

With `ema_decay=0.95`, the new proposal contributes 5%. This value is an
Original-DP experiment setting supported by Cycle-1 evidence; it must not be
described as uniquely implied by the paper.

## Current experiment boundary

The Cycle-2 `2e-8` branch used minibatch EMA and was decisively negative on the
fixed train selector (`0.931899 -> 0.928364`, paired 95% CI
`[-0.003746, -0.003324]`), so it was stopped after epoch 12.

The matched epoch-boundary branch reuses the same immutable replay cache and
the same incumbent. Checkpoint selection remains fixed train-set deterministic
mean reward; validation is report-only. Continuation beyond epoch 12 is
conditional on the paired train-selector result, not on a lower training loss
or on the claimed authority of a reference implementation.
