from torch.optim.lr_scheduler import LinearLR, MultiplicativeLR, SequentialLR


def LinearWarmupConstantLR(optimizer, total_epochs, warm_up_epochs, start_factor=0.1):
    """Official HDP schedule: linear warmup followed by a constant learning rate."""
    if total_epochs < 1:
        raise ValueError("total_epochs must be >= 1")
    if not 0 <= warm_up_epochs <= total_epochs:
        raise ValueError("warm_up_epochs must be between 0 and total_epochs")

    fixed_scheduler = MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 1.0)
    if warm_up_epochs <= 1:
        return fixed_scheduler

    warmup_scheduler = LinearLR(
        optimizer, start_factor=start_factor, total_iters=warm_up_epochs - 1
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, fixed_scheduler],
        milestones=[warm_up_epochs],
    )

    return scheduler
