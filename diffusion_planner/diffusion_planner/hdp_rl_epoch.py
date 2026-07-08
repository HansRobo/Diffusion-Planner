"""HDP reward-weighted RL-Hybrid epoch loop.

The only supported RL path is the official HDP-style reward-weighted hybrid loss:

  1. sample a group of trajectories per scene;
  2. score them with the Tier IV NPZ EPDMS-style multi-reward;
  3. group-normalize the reward and weight the HDP hybrid diffusion loss with
     exp(beta * normalized_reward).
"""

import torch
from torch import nn
from tqdm import tqdm

from diffusion_planner.hdp_rl_utils import (
    compute_epdms_style_reward,
    compute_official_reward_weighted_loss,
    expand_batch,
    sample_group,
)
from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.utils import ddp
from diffusion_planner.utils.masks import pose_padding_mask
from diffusion_planner.utils.train_utils import get_epoch_mean_loss


def _set_official_rl_train_mode(model, args):
    if getattr(args, "rl_train_scope", "decoder") != "decoder":
        return
    net = ddp.get_model(model, args.ddp)
    encoder = getattr(net, "encoder", None)
    if encoder is not None:
        encoder.eval()


def _neighbor_future_world(neighbor_future_raw: torch.Tensor):
    mask = pose_padding_mask(neighbor_future_raw)
    neighbors_future = heading_to_cos_sin(neighbor_future_raw)
    neighbors_future[mask] = 0.0
    return neighbors_future, mask


def _hdp_rl_step(raw_inputs, model, optimizer, args, ema, aug):
    n = args.num_generations

    raw_inputs = dict(raw_inputs)
    if aug is not None:
        raw_inputs["ego_agent_past"] = heading_to_cos_sin(raw_inputs["ego_agent_past"])
        raw_inputs["goal_pose"] = heading_to_cos_sin(raw_inputs["goal_pose"])
        raw_inputs, ego_future_aug, neighbors_future_aug = aug(
            raw_inputs, raw_inputs["ego_agent_future"], raw_inputs["neighbor_agents_future"]
        )
        raw_inputs["ego_agent_future"] = ego_future_aug
        raw_inputs["neighbor_agents_future"] = neighbors_future_aug
    else:
        raw_inputs["ego_agent_past"] = heading_to_cos_sin(raw_inputs["ego_agent_past"])
        raw_inputs["goal_pose"] = heading_to_cos_sin(raw_inputs["goal_pose"])

    exp = expand_batch(raw_inputs, n)
    batch_size = exp["ego_current_state"].shape[0]
    num_scenes = batch_size // n

    neighbors_future, neighbor_future_mask = _neighbor_future_world(exp["neighbor_agents_future"])
    norm_exp = args.observation_normalizer(exp)

    ego_world = sample_group(model, norm_exp, args.rl_noise_scale, args.device)
    _set_official_rl_train_mode(model, args)

    reward, reward_metrics = compute_epdms_style_reward(
        ego_world, raw_inputs, neighbors_future, num_scenes, n, args
    )

    optimizer.zero_grad()
    loss_dict = compute_official_reward_weighted_loss(
        model,
        norm_exp,
        ego_world,
        neighbors_future,
        neighbor_future_mask,
        reward,
        num_scenes,
        n,
        args,
    )
    loss_dict["loss"].backward()
    nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()
    if ema is not None:
        ema.update(model)

    result = {
        "loss": loss_dict["loss"].detach(),
        "rl_loss": loss_dict["loss"].detach(),
        "official_reward_weighted_loss": loss_dict["official_reward_weighted_loss"],
        "reward_mean": reward.mean().detach(),
        "reward_max": reward.reshape(num_scenes, n).max(dim=1).values.mean().detach(),
        "ego_hdp_diffusion_loss": loss_dict["ego_hdp_diffusion_loss"],
        "ego_hdp_waypoint_loss": loss_dict["ego_hdp_waypoint_loss"],
        "official_reward_weight_mean": loss_dict["official_reward_weight_mean"].detach(),
        "official_reward_weight_max": loss_dict["official_reward_weight_max"].detach(),
        "official_reward_weight_min": loss_dict["official_reward_weight_min"].detach(),
        "official_valid_group_fraction": loss_dict["official_valid_group_fraction"].detach(),
    }
    result.update({key: value.detach() for key, value in reward_metrics.items()})
    return result


def train_hdp_rl_epoch(data_loader, model, optimizer, args, ema, aug):
    epoch_loss = []

    model.train()
    _set_official_rl_train_mode(model, args)

    if args.ddp:
        torch.cuda.synchronize()

    if ddp.get_rank() == 0:
        data_loader = tqdm(data_loader, desc="HDP-RL", unit="batch")

    for raw_inputs in data_loader:
        raw_inputs = {
            key: value.to(args.device, non_blocking=True) for key, value in raw_inputs.items()
        }
        step_loss = _hdp_rl_step(raw_inputs, model, optimizer, args, ema, aug)

        if args.ddp:
            torch.cuda.synchronize()
        epoch_loss.append(step_loss)

    epoch_mean_loss = get_epoch_mean_loss(epoch_loss)

    if args.ddp:
        epoch_mean_loss = ddp.reduce_and_average_losses(epoch_mean_loss, torch.device(args.device))

    if ddp.get_rank() == 0:
        print(f"{epoch_mean_loss['loss']=:.4f}")
        print(f"{epoch_mean_loss['reward_mean']=:.4f}")
        print(f"{epoch_mean_loss['reward_max']=:.4f}")

    return epoch_mean_loss, epoch_mean_loss["loss"]
