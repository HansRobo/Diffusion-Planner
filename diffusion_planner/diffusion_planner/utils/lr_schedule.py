from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def CosineAnnealingWarmUpRestarts(optimizer, epoch, warm_up_epoch, start_factor=0.1, eta_min=1e-6):
    """Linear warmup for ``warm_up_epoch`` epochs, then cosine decay to
    ``eta_min`` over the remaining epochs.

    Previously the post-warmup phase used ``MultiplicativeLR(lambda=1.0)``, i.e.
    it held the learning rate CONSTANT after warmup (despite the name). That let
    training collapse right after warmup — every model, diffusion included,
    peaked in the warmup epochs and then degraded. Decaying the lr with a proper
    cosine schedule (matching original planTF) keeps the lr low enough after the
    initial phase to actually converge. See
    docs/plantf_original_comparison_and_roadmap.md.

    Called with ``scheduler.step()`` once per epoch, so the sub-schedulers count
    in epochs.
    """
    assert epoch >= warm_up_epoch
    warmup = LinearLR(optimizer, start_factor=start_factor, total_iters=max(warm_up_epoch - 1, 1))
    cosine = CosineAnnealingLR(optimizer, T_max=max(epoch - warm_up_epoch, 1), eta_min=eta_min)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warm_up_epoch])
