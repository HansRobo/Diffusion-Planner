from torch.optim.lr_scheduler import LinearLR, MultiplicativeLR, SequentialLR


def CosineAnnealingWarmUpRestarts(optimizer, train_steps, warm_up_steps, start_factor=0.1):
    """Step-based schedule: linear warm-up over ``warm_up_steps`` optimizer steps, then a
    constant learning rate. ``scheduler.step()`` is expected to be called once per optimizer
    step (not once per epoch)."""
    assert train_steps >= warm_up_steps

    # total_iters = warm_up_steps - 1 (with the milestone at warm_up_steps) so LinearLR saturates
    # to the full base LR exactly as the switch to the constant-LR scheduler happens.
    warmup_scheduler = LinearLR(optimizer, start_factor=start_factor, total_iters=warm_up_steps - 1)
    fixed_scheduler = MultiplicativeLR(optimizer, lr_lambda=lambda step: 1.0)

    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, fixed_scheduler], milestones=[warm_up_steps]
    )

    return scheduler
