import torch
from torch import nn

from diffusion_planner.model.module.decoder import compute_training_loss
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.train_utils import compute_grad_stats


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


def train_one_step(inputs, model, optimizer, args, ema, aug: StatePerturbation = None):
    """Run a single optimizer step on one batch and return the loss dictionary.

    The caller owns the training loop (step counting, LR scheduling, validation cadence);
    this function only performs forward/backward/step for one batch. ``model.train()`` is
    assumed to have been set by the caller.
    """
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

    # Gradient statistics (computed before clipping so that exploding
    # gradients are not masked by clip_grad_norm_).
    loss.update(compute_grad_stats(model.parameters()))

    nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()

    ema.update(model)

    if args.ddp:
        torch.cuda.synchronize()

    # Detach to python floats so the caller can accumulate an interval's worth of losses
    # without retaining autograd graphs / GPU tensors.
    return {
        key: (value if isinstance(value, (int, float)) else value.item())
        for key, value in loss.items()
    }
