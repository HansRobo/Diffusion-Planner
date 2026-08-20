"""DrivoR's optimizer schedule, ported from
``navsim/agents/drivoR/drivor_agent.py::get_optimizers`` (lines 408-483).

Two things there differ from this repo's diffusion-path schedule, and both
matter on the full train list:

* DrivoR returns ``{"scheduler": ..., "interval": "step"}`` (drivor_agent.py:483),
  i.e. Lightning advances it once per OPTIMIZER STEP over
  ``T_max = batches_per_epoch * num_epochs``.  ``utils/lr_schedule.py`` is
  advanced once per EPOCH by ``drivor_train_loop``.  With the 2.5 % subsample
  (1,063 steps/epoch) that difference was cosmetic; on the full list (21,274
  steps/epoch) the 5-epoch warm-up became ~106k steps / ~10 h at a tenth of the
  peak LR, which is what stalled v6.
* DrivoR's peak LR is ``base_lr * sqrt(global_batch / base_batch_size)``
  (drivor_agent.py:415) with ``base_lr=1e-4`` at ``base_batch_size=64``
  (config/training/t4_training.yaml:279-280).  The ramp starts at
  ``start_factor=1e-6`` -- effectively from zero, not from 0.1 of peak -- over
  ``warmup_ratio=0.1`` of the whole run, then cosine-anneals to ``eta_min=0``.

The T4 config also records the empirical band on DrivoR's own architecture
(t4_training.yaml:269-275): ``base_lr=2e-4`` "diverged during warm-up at
1.71e-4", and 8.3e-5 was the largest rate previously known to be stable.  Note
that sqrt-scaling 1e-4 from batch 64 up to a global batch of 256 lands on 2e-4,
i.e. *above* that recorded divergence point.  Which is why ``probe`` exists:
measure the band on this encoder and this data rather than inherit DrivoR's.
"""

from __future__ import annotations

import math

from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    LinearLR,
    SequentialLR,
)

# DrivoR's released reference point, config/training/t4_training.yaml:279-280.
DRIVOR_BASE_LR = 1e-4
DRIVOR_BASE_BATCH_SIZE = 64
DRIVOR_WARMUP_RATIO = 0.1
# drivor_agent.py:461 -- the ramp starts from essentially zero, not from a
# fraction of peak the way ``CosineAnnealingWarmUpRestarts`` does.
DRIVOR_RAMP_START_FACTOR = 1e-6


def scaled_peak_lr(base_lr: float, global_batch: int, base_batch_size: int) -> float:
    """DrivoR's sqrt batch scaling (drivor_agent.py:409/415).

    Square-root, not linear: DrivoR tuned at batch 64 and the released config
    keeps this form for the 8-GPU launch.
    """
    if base_batch_size <= 0:
        return float(base_lr)
    return float(base_lr) * math.sqrt(float(global_batch) / float(base_batch_size))


def build_drivor_scheduler(optimizer, *, total_steps: int, warmup_ratio: float):
    """``LinearLR`` ramp then ``CosineAnnealingLR``, both advanced per step.

    Mirrors drivor_agent.py:452-481 including the ``milestones=[T_max_ramp]``
    hand-off and the ``T_max = total - ramp`` cosine length, so the LR reaches
    exactly 0 on the final step instead of stopping part-way down the cosine.
    """
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError(f"warmup_ratio must be in [0, 1), got {warmup_ratio}")
    total_steps = max(1, int(total_steps))
    ramp = int(total_steps * warmup_ratio)
    if ramp <= 0:
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0.0, last_epoch=-1)
    return SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=DRIVOR_RAMP_START_FACTOR, total_iters=ramp),
            CosineAnnealingLR(
                optimizer, T_max=max(1, total_steps - ramp), eta_min=0.0, last_epoch=-1
            ),
        ],
        milestones=[ramp],
    )


def build_lr_probe(optimizer, *, steps: int, lr_min: float, lr_max: float):
    """LR range test: sweep the LR geometrically from ``lr_min`` to ``lr_max``.

    Not part of DrivoR -- it is how we find the peak LR to hand to
    :func:`build_drivor_scheduler`, since DrivoR's recorded divergence point
    belongs to a different encoder.  ``LambdaLR`` multiplies the optimizer's own
    ``base_lrs``, so the divergence guard's LR cuts still survive a step (see
    ``DivergenceGuard.attach_scheduler``); the caller must have set the
    optimizer's lr to ``lr_min``.
    """
    steps = max(1, int(steps))
    ratio = float(lr_max) / float(lr_min)
    return LambdaLR(optimizer, lr_lambda=lambda step: ratio ** (min(step, steps) / steps))


def scheduler_base_lrs(scheduler) -> list:
    """Every ``base_lrs`` list behind ``scheduler``, including ``SequentialLR``'s.

    ``SequentialLR`` keeps no ``base_lrs`` of its own -- each wrapped scheduler
    has one, and a per-step scheduler recomputes the LR from them on every
    ``step()``.  So anything that wants to permanently scale the LR (the
    divergence guard) has to reach these, not ``param_groups``.
    """
    children = getattr(scheduler, "_schedulers", None) or [scheduler]
    return [child.base_lrs for child in children if hasattr(child, "base_lrs")]


__all__ = [
    "DRIVOR_BASE_LR",
    "DRIVOR_BASE_BATCH_SIZE",
    "DRIVOR_WARMUP_RATIO",
    "build_drivor_scheduler",
    "build_lr_probe",
    "scaled_peak_lr",
    "scheduler_base_lrs",
]
