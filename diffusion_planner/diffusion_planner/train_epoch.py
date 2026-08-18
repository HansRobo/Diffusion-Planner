import torch
from torch import nn
from tqdm import tqdm

from diffusion_planner.model.module.decoder import compute_training_loss
from diffusion_planner.utils import ddp
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.train_utils import (
    assert_parameters_finite,
    compute_grad_linf_norm,
    describe_nonfinite_step,
    get_epoch_mean_loss,
)


def heading_to_cos_sin(x):
    """
    Convert heading angle to cosine and sine.
    Args:
        x: [B, T, 3] where last dimension is (x, y, heading)
    Output:
        x: [B, T, 4] where last dimension is (x, y, cos(heading), sin(heading))

    Idempotent: a [..., 4] input that is already (x, y, cos, sin) is returned
    unchanged. This guards against double-conversion (cos(cos)) now that scene-gen
    emits 4-col futures — callers can hand it either layout safely.
    """
    if x.shape[-1] == 4:
        return x
    return torch.cat(
        [
            x[..., :2],
            x[..., 2:3].cos(),
            x[..., 2:3].sin(),
        ],
        dim=-1,
    )


def train_epoch(data_loader, model, optimizer, args, ema, aug: StatePerturbation = None):
    if len(data_loader) == 0:
        empty = {"loss": 0.0, "turn_indicator_accuracy": 0.0}
        return empty, 0.0, []

    epoch_loss = []

    model.train()

    if args.ddp:
        torch.cuda.synchronize()

    # Captured before the tqdm wrap below: tqdm does not forward attribute access, so on
    # rank 0 -- the only rank that prints -- data_loader.dataset would be gone by then.
    dataset = getattr(data_loader, "dataset", None)

    if ddp.get_rank() == 0:
        data_loader = tqdm(data_loader, desc="Training", unit="batch")

    # Epoch maxima of the gradient norms, kept on device and synced once at epoch end.
    # The epoch *mean* dilutes a single exploding step into invisibility, which is exactly
    # the failure this instrumentation exists to catch, so the max is logged alongside it.
    grad_max: dict[str, torch.Tensor] = {}
    skipped_steps = 0
    # Reports for the skipped steps, gathered across ranks at epoch end so rank 0 can put
    # them in wandb. Capped because a run that skips thousands of steps would otherwise
    # pickle thousands of strings through the collective; skipped_steps carries the count.
    nonfinite_reports: list[dict] = []
    max_reports_per_rank = 10

    for step, inputs in enumerate(data_loader):
        # Optional (see DiffusionPlannerData(with_index=True)): lets the skip path below
        # name the NPZ files a bad batch came from. Popped before the .to(device) sweep so
        # it never reaches the model.
        sample_index = inputs.pop("sample_index", None)

        inputs = {key: value.to(args.device) for key, value in inputs.items()}
        inputs["ego_agent_past"] = heading_to_cos_sin(inputs["ego_agent_past"])
        inputs["goal_pose"] = heading_to_cos_sin(inputs["goal_pose"])

        ego_future = inputs["ego_agent_future"]
        neighbors_future = inputs["neighbor_agents_future"]
        # Normalize to ego-centric
        if aug is not None:
            inputs, ego_future, neighbors_future = aug(inputs, ego_future, neighbors_future)

        # heading to cos sin
        ego_future = heading_to_cos_sin(ego_future)

        mask = torch.sum(torch.ne(neighbors_future[..., :3], 0), dim=-1) == 0
        neighbors_future = heading_to_cos_sin(neighbors_future)
        neighbors_future[mask] = 0.0
        inputs = args.observation_normalizer(inputs)

        # call the model
        optimizer.zero_grad()

        loss = compute_training_loss(model, inputs, (ego_future, neighbors_future, mask), args)

        loss["loss"] = (
            args.alpha_neighbor_loss * loss["neighbor_prediction_loss"]
            + args.alpha_planning_loss * loss["ego_planning_loss"]
            + loss["turn_indicator_loss"]
            + args.coeff_road_border_loss * loss["road_border_loss"]
            + args.coeff_neighbor_collision_loss * loss["neighbor_collision_loss"]
        )

        # loss backward
        loss["loss"].backward()

        # Gradient statistics, computed BEFORE clipping so that exploding gradients are not
        # masked by clip_grad_norm_. Values stay as device tensors; get_epoch_mean_loss does
        # the one .item() per key at epoch end, so this costs no extra host sync per step.
        interval = getattr(args, "grad_stats_interval", 1)
        if interval and step % interval == 0:
            linf = compute_grad_linf_norm(model.parameters())
            if linf is not None:
                loss["grad/linf_norm"] = linf

        # clip_grad_norm_ returns the *pre-clip* total norm, so this is the real gradient
        # magnitude, free of charge.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 5)
        loss["grad/l2_norm"] = grad_norm.detach()

        for key in ("grad/l2_norm", "grad/linf_norm"):
            if key in loss:
                prev = grad_max.get(key)
                grad_max[key] = (
                    loss[key] if prev is None else torch.maximum(prev, loss[key].detach())
                )

        # A single Inf gradient element is fatal without this guard: clip_grad_norm_ scales
        # by max_norm/total_norm, which is 5/inf == 0, and inf * 0 -> nan. One non-finite
        # element therefore turns whole parameter tensors into NaN on the next
        # optimizer.step(), and every subsequent forward is NaN. Dropping the step instead
        # costs one device->host sync and leaves the model intact.
        #
        # DDP-safe: backward has already all-reduced the gradients, so grad_norm is computed
        # from identical values on every rank and every rank takes the same branch. Were that
        # not true, one rank skipping optimizer.step() while another took it would silently
        # desynchronize the replicas.
        if not torch.isfinite(grad_norm):
            named = list(ddp.get_model(model, args.ddp).named_parameters())
            paths = None
            if sample_index is not None and dataset is not None:
                paths = [str(dataset.data_list[i]) for i in sample_index.tolist()]
            report = describe_nonfinite_step(
                grad_norm=grad_norm,
                losses=loss,
                tensors={
                    **inputs,
                    "ego_future": ego_future,
                    "neighbors_future": neighbors_future,
                },
                parameters=named,
                sample_paths=paths,
            )
            rank = ddp.get_rank()
            # Not `print`: setup_for_distributed suppresses non-rank-0 output, and the bad
            # batch is usually on some other rank.
            ddp.print_from_every_rank(f"[rank {rank}] step {step}: {report['text']}")
            if len(nonfinite_reports) < max_reports_per_rank:
                nonfinite_reports.append({"rank": rank, "step": step, **report})

            optimizer.zero_grad(set_to_none=True)
            skipped_steps += 1
            # Not appended to epoch_loss: a NaN in the mean would hide every other metric
            # for the epoch. The skipped-step count below is what makes this visible.
            continue

        optimizer.step()

        ema.update(model)

        if args.ddp:
            torch.cuda.synchronize()
        epoch_loss.append(loss)

    # get_epoch_mean_loss returns {} if every step was skipped. That is not raised on here:
    # doing so on one rank while the others enter the collectives below would hang the job
    # until the 10000s NCCL timeout instead of failing. reduce_and_average_losses tolerates
    # the missing keys, so the check moves after it, by which point every rank agrees.
    epoch_mean_loss = get_epoch_mean_loss(epoch_loss)
    epoch_mean_loss["grad/skipped_steps"] = float(skipped_steps)
    epoch_mean_loss["grad/completed_steps"] = float(len(epoch_loss))
    for key, value in grad_max.items():
        epoch_mean_loss[f"{key}_max"] = value.item()

    # Catches a model already poisoned before this guard existed (e.g. resumed from a NaN
    # checkpoint), which would otherwise skip every step forever.
    assert_parameters_finite(ddp.get_model(model, args.ddp))

    if args.ddp:
        epoch_mean_loss = ddp.reduce_and_average_losses(epoch_mean_loss, torch.device(args.device))

    # Collective on every rank; rank 0 is the only one with a wandb run to log them to.
    nonfinite_reports = [
        row for rank_rows in ddp.gather_objects(nonfinite_reports) for row in rank_rows
    ]

    if epoch_mean_loss["grad/completed_steps"] == 0:
        raise RuntimeError(
            f"every one of the {int(epoch_mean_loss['grad/skipped_steps'])} training steps "
            "this epoch produced a non-finite gradient; nothing was learned. See the reports "
            "above for the offending NPZ files."
        )

    if ddp.get_rank() == 0:
        print(f"{epoch_mean_loss['loss']=:.4f}")
        print(f"{epoch_mean_loss['turn_indicator_accuracy']=:.4f}")
        if epoch_mean_loss["grad/skipped_steps"]:
            print(f"{epoch_mean_loss['grad/skipped_steps']=:.0f} (non-finite gradient)")

    return epoch_mean_loss, epoch_mean_loss["loss"], nonfinite_reports
