from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    MultiplicativeLR,
    SequentialLR,
)


def WarmupConstantLR(optimizer, epoch, warm_up_epoch, start_factor=0.1):
    """Linear warmup for ``warm_up_epoch`` epochs, then HOLD the lr constant.

    This is the original schedule, formerly (mis)named
    ``CosineAnnealingWarmUpRestarts``: the post-warmup phase is
    ``MultiplicativeLR(lambda=1.0)``, so the learning rate never decays. Kept
    under an accurate name for backward-compatible reproduction of pre-existing
    runs and resumed checkpoints. New training should prefer
    :func:`WarmupCosineAnnealingLR`.

    Called with ``scheduler.step()`` once per epoch, so the sub-schedulers count
    in epochs.
    """
    assert epoch >= warm_up_epoch
    warmup = LinearLR(optimizer, start_factor=start_factor, total_iters=warm_up_epoch - 1)
    fixed = MultiplicativeLR(optimizer, lr_lambda=lambda e: 1.0)
    return SequentialLR(optimizer, schedulers=[warmup, fixed], milestones=[warm_up_epoch])


def WarmupCosineAnnealingLR(optimizer, epoch, warm_up_epoch, start_factor=0.1, eta_min=1e-6):
    """Linear warmup for ``warm_up_epoch`` epochs, then cosine-decay to
    ``eta_min`` over the remaining epochs.

    Replaces the constant post-warmup phase of :func:`WarmupConstantLR`, which
    let training collapse right after warmup: the learning rate stayed at its
    peak for the whole run, so every model peaked in the warmup epochs and then
    degraded. Decaying with a proper cosine schedule (matching original planTF)
    keeps the lr low enough after the initial phase to actually converge.

    Called with ``scheduler.step()`` once per epoch, so the sub-schedulers count
    in epochs. ``max(..., 1)`` guards the ``warm_up_epoch == 1`` /
    ``epoch == warm_up_epoch`` edge cases so the sub-schedulers always get
    positive spans.
    """
    assert epoch >= warm_up_epoch
    warmup = LinearLR(optimizer, start_factor=start_factor, total_iters=max(warm_up_epoch - 1, 1))
    cosine = CosineAnnealingLR(optimizer, T_max=max(epoch - warm_up_epoch, 1), eta_min=eta_min)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warm_up_epoch])


# Backward-compatible alias. The original public name advertised cosine annealing
# but the implementation held the lr constant after warmup; keep the name bound
# to that exact (constant) behavior so existing imports and resumed checkpoints
# reproduce unchanged. New code should call WarmupCosineAnnealingLR directly (or
# select it via TrainConfig.lr_schedule_type="cosine").
CosineAnnealingWarmUpRestarts = WarmupConstantLR


def build_lr_scheduler(optimizer, epoch, warm_up_epoch, schedule_type="cosine", **kwargs):
    """Dispatch to the configured lr schedule.

    ``schedule_type="cosine"`` (default) -> :func:`WarmupCosineAnnealingLR`;
    ``"constant"`` -> :func:`WarmupConstantLR` (the legacy behavior).
    """
    if schedule_type == "cosine":
        return WarmupCosineAnnealingLR(optimizer, epoch, warm_up_epoch, **kwargs)
    if schedule_type == "constant":
        return WarmupConstantLR(optimizer, epoch, warm_up_epoch)
    raise ValueError(
        f"Unknown lr schedule_type: {schedule_type!r} (expected 'cosine' or 'constant')"
    )
