import random
from argparse import Namespace
from functools import partial

import torch
import torch.nn as nn

import diffusion_planner.model.diffusion_utils.dpm_solver_pytorch as dpm
from diffusion_planner.dimensions import TURN_INDICATOR_OUTPUT_DIM
from diffusion_planner.loss import (
    compute_ego_edge_points,
    compute_neighbor_collision_penalty,
    compute_road_border_penalty,
    hybrid_loss_components,
    inverse_normalize_ego_state,
    inverse_normalize_ego_velocity,
    make_turn_indicator_gt,
    normalize_ego_state,
    normalize_ego_velocity,
    sample_diffusion_time,
    vp_supervision_elementwise_loss,
    weighted_waypoint_dpm_loss,
    velocity_to_waypoints,
    waypoints_to_velocity,
)
from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear
from diffusion_planner.model.flow_matching_utils.ode_solver import (
    euler_integration,
    heun_integration,
    rk4_integration,
)
from diffusion_planner.model.module.dit import DiT
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer


def generate_prefix_mask(delay: torch.Tensor, num_agents: int, max_len: int) -> torch.Tensor:
    """Generates a prefix mask based on a delay tensor.

    Args:
        delay: A 1D tensor of shape (B,) with delay values.
        num_agents: The number of agents (P).
        max_len: The maximum length of the sequence (T+1 or T_plus_1).

    Returns:
        A 4D boolean tensor of shape (B, num_agents, max_len, 1) where mask[i, :, j, 0] is True if j <= delay[i].
    """
    # Create steps tensor (1, 1, max_len, 1)
    steps = torch.arange(max_len, device=delay.device).view(1, 1, -1, 1)
    # Reshape delay to (B, 1, 1, 1) for broadcasting
    reshaped_delay = delay.reshape(delay.shape[0], 1, 1, 1)
    # Perform the comparison, result is (B, 1, max_len, 1)
    mask = steps <= reshaped_delay
    ego_mask = mask.expand(-1, 1, -1, -1)
    neighbor_mask = torch.zeros(
        (delay.shape[0], num_agents - 1, max_len, 1), dtype=torch.bool, device=delay.device
    )
    return torch.cat([ego_mask, neighbor_mask], dim=1)


def replace_current_state(x: torch.Tensor, current_states: torch.Tensor) -> torch.Tensor:
    """Return a trajectory tensor with the first timestep replaced."""
    return torch.cat([current_states[:, :, None, :], x[:, :, 1:, :]], dim=2)


def compute_training_loss(
    model: nn.Module,
    inputs: dict[str, torch.Tensor],
    futures: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    args: Namespace,
):
    norm = args.state_normalizer
    model_type = args.diffusion_model_type
    use_velocity = args.use_velocity_representation
    vp_model_types = {"x_start", "noise", "score", "v"}
    if use_velocity and model_type not in vp_model_types:
        raise NotImplementedError("Velocity representation is only defined for VP diffusion.")
    supervision_type = getattr(args, "diffusion_supervision_type", model_type)
    if model_type in vp_model_types and supervision_type not in vp_model_types:
        raise ValueError(f"Unsupported diffusion_supervision_type={supervision_type!r}")
    if use_velocity and (model_type != "x_start" or supervision_type != "x_start"):
        raise NotImplementedError(
            "HDP velocity representation is enabled only for x_start prediction with x_start supervision."
        )
    hybrid_window = args.hybrid_loss_window

    ego_future, neighbors_future, neighbor_future_mask = futures
    neighbors_future_valid = ~neighbor_future_mask  # [B, Pn, V]

    B, Pn, T, _ = neighbors_future.shape
    P = 1 + Pn
    ego_current, neighbors_current = (
        inputs["ego_current_state"][:, :4],
        inputs["neighbor_agents_past"][:, :Pn, -1, :4],
    )
    # inputs are observation-normalized here; convert the longitudinal-velocity channel back
    # to m/s so coeff_velocity weights physical speed (with the default coeff_velocity=0.05
    # this exactly reproduces the legacy normalized-units behavior).
    _lv_mean, _lv_std = args.observation_normalizer.stats("ego_current_state")
    longitudinal_velocity = inputs["ego_current_state"][:, 4:5] * float(_lv_std[4]) + float(
        _lv_mean[4]
    )
    neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
    neighbor_mask = torch.concat(
        (neighbor_current_mask.unsqueeze(-1), neighbor_future_mask), dim=-1
    )

    gt_future = torch.cat(
        [ego_future[:, None, :, :], neighbors_future[..., :]], dim=1
    )  # [B, P, T, 4]
    current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1)  # [B, P, 4]

    eps = 1e-3
    t = sample_diffusion_time(
        B,
        gt_future.device,
        eps,
        getattr(args, "diffusion_time_sample_method", "uniform"),
    )  # [B,]
    t = t.view(B, 1, 1, 1)
    t = t.expand(B, P, T + 1, 1)
    z = torch.randn_like(gt_future, device=gt_future.device)  # [B, P, T, 4]

    max_delay = 5
    delay = torch.randint(0, max_delay + 1, (B,), device=gt_future.device)  # [B,]
    prefix_mask = generate_prefix_mask(delay, 1 + Pn, T + 1)  # (B, P, T+1, 1)
    mask_coeff = random.uniform(0.0, 1.0)
    curr_mask_time = torch.maximum(t * mask_coeff, torch.tensor(eps, device=gt_future.device))
    t = torch.where(prefix_mask, curr_mask_time, t)

    waypoint_gt = torch.cat([current_states[:, :, None, :], norm(gt_future)], dim=2)
    all_gt = waypoint_gt.clone()
    if use_velocity:
        ego_velocity_gt = waypoints_to_velocity(ego_future)  # [B, T, 4]
        all_gt[:, 0, 1:, :] = normalize_ego_velocity(ego_velocity_gt, norm)
    all_gt[:, 1:] = all_gt[:, 1:].masked_fill(neighbor_mask.unsqueeze(-1), 0.0)

    if model_type in vp_model_types:
        model_ref = getattr(model, "module", model)
        sde = getattr(model_ref, "sde", None)
        if sde is None:
            sde = VPSDE_linear()
        t_future = t[..., 1:, :]
        alpha, std = sde.marginal_prob(torch.ones_like(all_gt[..., 1:, :]), t_future)
        mean = alpha * all_gt[..., 1:, :]
        # mean([B, P, T, D]), std([B, 1, T, 1]), z([B, P, T, D])
        xT = mean + std * z

        xT = torch.cat([all_gt[:, :, :1, :], xT], dim=2)
        xT = torch.where(prefix_mask, all_gt, xT)  # [B, P, 1 + T, 4]
        xT_future = xT[:, :, 1:, :]

        merged_inputs = {
            **inputs,
            "gt_trajectories": all_gt,
            "turn_indicator_trajectories": waypoint_gt,
            "sampled_trajectories": xT,
            "diffusion_time": t,
            "prefix_mask": prefix_mask,
        }
        _, decoder_output = model(merged_inputs)  # [B, P, 1 + T, 4]
        model_output = decoder_output["model_output"][:, :, 1:, :]  # [B, P, T, 4]

        gt_target = all_gt[:, :, 1:, :]  # [B, P, T, 4]
        pred_x_start = sde.transform(f"{model_type}->x_start", model_output, t_future, xT_future)
        supervised_prediction = sde.transform(
            f"{model_type}->{supervision_type}", model_output, t_future, xT_future
        )

        if use_velocity:
            # Guarded at function entry: velocity mode implies model_type == supervision_type
            # == "x_start", so supervised_prediction IS the x_start prediction and the ego
            # target is the normalized velocity GT.
            ego_diffusion_loss = torch.sum(
                (supervised_prediction[:, 0] - gt_target[:, 0]) ** 2, dim=-1
            )
            ego_pred_velocity_raw = inverse_normalize_ego_velocity(pred_x_start[:, 0], norm)
            _, ego_waypoint_loss = hybrid_loss_components(
                pred_x_start[:, 0],
                gt_target[:, 0],
                ego_pred_velocity_raw,
                ego_future,
                W=hybrid_window,
            )
            neighbor_dpm_loss = weighted_waypoint_dpm_loss(
                pred_x_start[:, 1:],
                waypoint_gt[:, 1:, 1:, :],
                longitudinal_velocity,
                args.coeff_position_lat_loss,
                args.coeff_position_lon_loss,
                args.coeff_heading_l2_loss,
                args.coeff_velocity,
                args.coeff_timestep,
            )
            dpm_loss = torch.cat([ego_diffusion_loss[:, None, :], neighbor_dpm_loss], dim=1)
        elif supervision_type == "x_start":
            dpm_loss = weighted_waypoint_dpm_loss(
                pred_x_start,
                gt_target,
                longitudinal_velocity,
                args.coeff_position_lat_loss,
                args.coeff_position_lon_loss,
                args.coeff_heading_l2_loss,
                args.coeff_velocity,
                args.coeff_timestep,
            )  # [B, P, T]
        else:
            dpm_loss = vp_supervision_elementwise_loss(
                supervised_prediction, z, std, supervision_type, sde, t_future, xT_future
            )

    elif model_type == "flow_matching":
        # t=0 is noise, t=1 is data
        t = t.reshape(-1, *([1] * (len(all_gt.shape) - 1)))  # [B, 1, 1, 1]
        xT = (1 - t) * z + t * all_gt[:, :, 1:, :]  # [B, P, T, 4]
        t = t.reshape(-1)  # [B,]

        xT = torch.cat([all_gt[:, :, :1, :], xT], dim=2)
        merged_inputs = {
            **inputs,
            "gt_trajectories": all_gt,
            "turn_indicator_trajectories": waypoint_gt,
            "sampled_trajectories": xT,
            "diffusion_time": t,
            "prefix_mask": prefix_mask,
        }
        _, decoder_output = model(merged_inputs)  # [B, P, 1 + T, 4]
        model_output = decoder_output["model_output"][:, :, 1:, :]  # [B, P, T, 4]

        target_v = all_gt[:, :, 1:, :] - z
        dpm_loss = torch.sum((model_output - target_v) ** 2, dim=-1)
    else:
        raise NotImplementedError(f"Unknown diffusion model type: {model_type}")

    masked_prediction_loss = dpm_loss[:, 1:, :][neighbors_future_valid]

    loss = {}

    if masked_prediction_loss.numel() > 0:
        loss["neighbor_prediction_loss"] = masked_prediction_loss.mean()
    else:
        loss["neighbor_prediction_loss"] = torch.tensor(0.0, device=masked_prediction_loss.device)

    ego_loss_horizon = dpm_loss[:, 0, : args.ego_prediction_horizon]
    ego_loss_valid = ~prefix_mask[:, 0, 1 : 1 + args.ego_prediction_horizon, 0]
    loss["ego_planning_loss"] = (
        ego_loss_horizon.masked_fill(~ego_loss_valid, 0.0).sum()
        / ego_loss_valid.sum().clamp_min(1)
    )
    if use_velocity:
        ego_waypoint_horizon = ego_waypoint_loss[:, : args.ego_prediction_horizon]
        loss["ego_planning_hybrid_loss"] = (
            ego_waypoint_horizon.masked_fill(~ego_loss_valid, 0.0).sum()
            / ego_loss_valid.sum().clamp_min(1)
        )
        loss["ego_hdp_diffusion_loss"] = loss["ego_planning_loss"].detach()
        loss["ego_hdp_waypoint_loss"] = loss["ego_planning_hybrid_loss"].detach()

    # Compute ego edge points for penalty losses
    need_ego_edge = model_type in vp_model_types and (
        args.coeff_road_border_loss > 0 or args.coeff_neighbor_collision_loss > 0
    )
    if need_ego_edge:
        ego_pred = model_output[:, 0]  # [B, T, 4]
        if use_velocity:
            ego_pred_world = velocity_to_waypoints(
                inverse_normalize_ego_velocity(pred_x_start[:, 0], norm)
            )
        else:
            ego_pred_world = pred_x_start[:, 0] * norm.std[0].to(model_output.device) + norm.mean[0].to(
                model_output.device
            )  # [B, T, 4]
        ego_edge_points = compute_ego_edge_points(
            ego_pred_world, inputs["ego_shape"], n_interp=args.road_border_n_interp
        )
        denorm_inputs = args.observation_normalizer.inverse(inputs)

    # Road border collision loss (ego only, x_start mode)
    if args.coeff_road_border_loss > 0 and model_type in vp_model_types:
        rb_loss = compute_road_border_penalty(
            ego_edge_points,
            denorm_inputs["line_strings"],
            margin=args.road_border_margin,
        )  # [B, T]
        loss["road_border_loss"] = rb_loss.mean()
    else:
        loss["road_border_loss"] = torch.tensor(0.0, device=dpm_loss.device)

    # Neighbor collision loss (ego only, x_start mode)
    if args.coeff_neighbor_collision_loss > 0 and model_type in vp_model_types:
        nc_loss = compute_neighbor_collision_penalty(
            ego_edge_points,
            neighbors_future,
            neighbors_future_valid,
            denorm_inputs["neighbor_agents_past"],
            margin_vehicle=args.neighbor_collision_margin_vehicle,
            margin_pedestrian=args.neighbor_collision_margin_pedestrian,
            margin_bicycle=args.neighbor_collision_margin_bicycle,
        )  # [B, T]
        loss["neighbor_collision_loss"] = nc_loss.mean()
    else:
        loss["neighbor_collision_loss"] = torch.tensor(0.0, device=dpm_loss.device)

    assert not torch.isnan(dpm_loss).sum(), f"loss cannot be nan, z={z}"

    turn_indicator_logit = decoder_output["turn_indicator_logit"]  # [B, TURN_INDICATOR_OUTPUT_KEEP]
    turn_indicator_gt = make_turn_indicator_gt(inputs["turn_indicators"])  # [B,]
    turn_indicator_loss = nn.functional.cross_entropy(
        turn_indicator_logit, turn_indicator_gt, reduction="none"
    )
    turn_indicator_change = inputs["turn_indicators"][:, -2] != inputs["turn_indicators"][:, -1]
    turn_indicator_coeff = torch.where(turn_indicator_change, 1.0, 0.05)
    turn_indicator_loss = (turn_indicator_loss * turn_indicator_coeff).mean()
    loss["turn_indicator_loss"] = turn_indicator_loss

    with torch.no_grad():
        turn_indicator_accuracy = (
            (turn_indicator_logit.argmax(dim=-1) == turn_indicator_gt).float().mean()
        )
        loss["turn_indicator_accuracy"] = turn_indicator_accuracy

    return loss


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        dpr = config.decoder_drop_path_rate
        self._predicted_neighbor_num = config.predicted_neighbor_num
        self._future_len = config.future_len
        self._sde = VPSDE_linear()

        self.dit = DiT(
            depth=config.decoder_depth,
            output_dim=(config.future_len + 1) * 4,  # x, y, cos, sin
            hidden_dim=config.hidden_dim,
            heads=config.num_heads,
            dropout=dpr,
            model_type=config.diffusion_model_type,
            sde=self._sde,
        )
        self.turn_indicator_predictor = nn.Linear(
            2 * (self._future_len // 10) + config.hidden_dim, TURN_INDICATOR_OUTPUT_DIM
        )

        self._state_normalizer: StateNormalizer = config.state_normalizer
        self._observation_normalizer: ObservationNormalizer = config.observation_normalizer

        # self._guidance_fn = config.guidance_fn
        self._guidance_fn = (
            config.guidance_fn if config.__dict__.get("guidance_fn") is not None else None
        )
        self._guidance_scale = config.guidance_scale
        self._model_type = config.diffusion_model_type
        self._use_velocity = config.use_velocity_representation
        if self._use_velocity and self._model_type != "x_start":
            raise NotImplementedError("HDP velocity representation is enabled only for x_start diffusion.")
        self._sample_steps = config.diffusion_sample_steps

        # Initialize transformer layers:
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        self.apply(_basic_init)

        # Zero-out output layers:
        nn.init.constant_(self.dit.final_layer.proj[-1].weight, 0)
        nn.init.constant_(self.dit.final_layer.proj[-1].bias, 0)

    @property
    def sde(self):
        return self._sde

    def _prepare_current_states(self, inputs):
        """Extract and prepare current states for ego and neighbors.

        Args:
            inputs: Dict containing ego_current_state and neighbor_agents_past

        Returns:
            Tuple of (current_states, neighbor_current_mask, ego_current, neighbors_current)
                - current_states: [B, P, 4] concatenated ego and neighbor current states
                - neighbor_current_mask: [B, Pn] mask for invalid neighbors
                - ego_current: [B, 1, 4] ego current state
                - neighbors_current: [B, Pn, 4] neighbor current states
        """
        ego_current = inputs["ego_current_state"][:, None, :4]
        neighbors_current = inputs["neighbor_agents_past"][
            :, : self._predicted_neighbor_num, -1, :4
        ]
        neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
        inputs["neighbor_current_mask"] = neighbor_current_mask

        current_states = torch.cat([ego_current, neighbors_current], dim=1)  # [B, P, 4]

        return current_states, neighbor_current_mask, ego_current, neighbors_current

    def _compute_turn_indicator(self, ego_trajectory, encoding_pooled):
        """Compute turn indicator logit from ego trajectory and encoding.

        Args:
            ego_trajectory: [B, 2 * (T // 10)] flattened ego trajectory positions
            encoding_pooled: [B, D] pooled encoding

        Returns:
            turn_indicator_logit: [B, TURN_INDICATOR_OUTPUT_DIM]
        """
        turn_indicator_input = torch.cat([ego_trajectory, encoding_pooled], dim=-1)
        return self.turn_indicator_predictor(turn_indicator_input)

    def _pool_encoding(self, encoding):
        encoding_valid = torch.any(encoding != 0, dim=-1)
        encoding_count = encoding_valid.sum(dim=1).clamp_min(1).unsqueeze(-1)
        return (encoding * encoding_valid.unsqueeze(-1)).sum(dim=1) / encoding_count

    def _ego_velocity_to_waypoints(self, ego_velocity):
        ego_velocity = inverse_normalize_ego_velocity(ego_velocity, self._state_normalizer)
        return velocity_to_waypoints(ego_velocity)

    def _normalize_ego_future(self, ego_future):
        return normalize_ego_state(ego_future, self._state_normalizer)

    def _turn_indicator_trajectory_from_latent(self, latent):
        B = latent.shape[0]
        if self._use_velocity:
            ego_future = self._ego_velocity_to_waypoints(latent[:, :1, 1:, :])
            ego_future = self._normalize_ego_future(ego_future)
            return ego_future[:, 0, ::10, :2].reshape(B, 2 * (self._future_len // 10))
        return latent[:, 0, 1::10, :2].reshape(B, 2 * (self._future_len // 10))

    def _latent_to_prediction(self, latent, current_states):
        prediction = self._state_normalizer.inverse(latent)[:, :, 1:]
        if self._use_velocity:
            prediction[:, :1] = self._ego_velocity_to_waypoints(latent[:, :1, 1:, :])
        return prediction

    def _forward_training(self, encoding, inputs, neighbor_current_mask, encoding_pooled):
        """Forward pass for training mode.

        Args:
            encoding: [B, N, D] encoded features
            inputs: Dict containing sampled_trajectories, gt_trajectories, diffusion_time, etc.
            neighbor_current_mask: [B, Pn] mask for invalid neighbors
            encoding_pooled: [B, D] pooled encoding

        Returns:
            Dict containing model_output and turn_indicator_logit
        """
        B = encoding.shape[0]
        P = 1 + self._predicted_neighbor_num

        sampled_trajectories = inputs["sampled_trajectories"].reshape(
            B, P, (1 + self._future_len), 4
        )
        diffusion_time = inputs["diffusion_time"]

        gt_trajectories = inputs["gt_trajectories"].reshape(B, P, (1 + self._future_len), 4)
        turn_indicator_trajectories = inputs.get("turn_indicator_trajectories", gt_trajectories)
        turn_indicator_trajectories = turn_indicator_trajectories.reshape(
            B, P, (1 + self._future_len), 4
        )
        ego_trajectory = turn_indicator_trajectories[:, 0, 1::10, :2].reshape(
            B, 2 * (self._future_len // 10)
        )
        turn_indicator_logit = self._compute_turn_indicator(ego_trajectory, encoding_pooled)

        return {
            "model_output": self.dit(
                sampled_trajectories,
                diffusion_time,
                encoding,
                neighbor_current_mask,
            ).reshape(B, P, -1, 4),
            "turn_indicator_logit": turn_indicator_logit,
        }

    def _inference_flow_matching(
        self,
        encoding,
        inputs,
        current_states,
        neighbor_current_mask,
        encoding_pooled,
        sampled_trajectories,
    ):
        """Inference using Flow Matching approach.

        Args:
            encoding: [B, N, D] encoded features
            inputs: Dict containing input data
            neighbor_current_mask: [B, Pn] mask for invalid neighbors
            encoding_pooled: [B, D] pooled encoding
            sampled_trajectories: [B, P, (1 + T) * 4] sampled trajectories

        Returns:
            Dict containing prediction and turn_indicator_logit
        """
        B = encoding.shape[0]
        P = 1 + self._predicted_neighbor_num

        x = sampled_trajectories
        NUM_STEP = 10
        func = partial(
            self.dit,
            cross_c=encoding,
            neighbor_current_mask=neighbor_current_mask,
        )
        x = euler_integration(func, x, NUM_STEP)
        # x = heun_integration(func, x, NUM_STEP)
        # x = rk4_integration(func, x, NUM_STEP)
        x = x.reshape(B, P, (1 + self._future_len), 4)
        ego_trajectory = self._turn_indicator_trajectory_from_latent(x)
        turn_indicator_logit = self._compute_turn_indicator(ego_trajectory, encoding_pooled)
        x = self._latent_to_prediction(x, current_states)
        return {"prediction": x, "turn_indicator_logit": turn_indicator_logit}

    def _inference_x_start(
        self,
        encoding,
        inputs,
        current_states,
        neighbor_current_mask,
        encoding_pooled,
        sampled_trajectories,
    ):
        """Inference using X-Start (DPM Solver) approach.

        Args:
            encoding: [B, N, D] encoded features
            inputs: Dict containing input data
            current_states: [B, P, 4] current states
            neighbor_current_mask: [B, Pn] mask for invalid neighbors
            encoding_pooled: [B, D] pooled encoding
            sampled_trajectories: [B, P, (1 + T) * 4] sampled trajectories

        Returns:
            Dict containing prediction and turn_indicator_logit
        """
        B = encoding.shape[0]
        P = 1 + self._predicted_neighbor_num

        action_prefix = sampled_trajectories.reshape(B, P, 1 + self._future_len, 4)
        action_prefix = replace_current_state(action_prefix, current_states)
        prefix_latent = action_prefix.clone()
        xT = prefix_latent.reshape(B, P, (1 + self._future_len) * 4)

        B, P, T_plus_1, D = action_prefix.shape

        delay = inputs["delay"].to(device=action_prefix.device)
        mask = generate_prefix_mask(delay, P, T_plus_1)  # (B, P, T_plus_1, 1)

        def prefix_constraint(xt, t, step):
            xt = xt.reshape(B, P, 1 + self._future_len, 4)
            xt = replace_current_state(xt, current_states)
            xt = torch.where(mask, prefix_latent, xt)
            return xt

        model_wrapper_params = {
            "classifier_fn": self._guidance_fn,
            "classifier_kwargs": {
                "model": self.dit,
                "model_condition": {
                    "cross_c": encoding,
                    "neighbor_current_mask": neighbor_current_mask,
                },
                "inputs": inputs,
                "observation_normalizer": self._observation_normalizer,
                "state_normalizer": self._state_normalizer,
            },
            "guidance_scale": self._guidance_scale,
            "guidance_type": "classifier" if self._guidance_fn is not None else "uncond",
        }
        if self._guidance_fn is not None and (self._use_velocity or self._model_type != "x_start"):
            raise RuntimeError(
                "Classifier guidance is currently only supported for waypoint x_start checkpoints."
            )

        noise_schedule = dpm.NoiseScheduleVP()

        model_fn = dpm.model_wrapper(
            self.dit,
            noise_schedule,
            model_type=self._model_type,
            model_kwargs={
                "cross_c": encoding,
                "neighbor_current_mask": neighbor_current_mask,
            },
            **model_wrapper_params,
        )

        dpm_solver = dpm.DPM_Solver(model_fn, noise_schedule, correcting_xt_fn=prefix_constraint)

        x0 = dpm_solver.sample(xT, steps=self._sample_steps, prefix_mask=mask, skip_type="logSNR")

        x0 = x0.reshape(B, P, (1 + self._future_len), 4)
        ego_trajectory = self._turn_indicator_trajectory_from_latent(x0)
        turn_indicator_logit = self._compute_turn_indicator(ego_trajectory, encoding_pooled)
        x0 = self._latent_to_prediction(x0, current_states)

        return {"prediction": x0, "turn_indicator_logit": turn_indicator_logit}

    def _forward_inference(
        self, encoding, inputs, current_states, neighbor_current_mask, encoding_pooled
    ):
        """Forward pass for inference mode.

        Args:
            encoding: [B, N, D] encoded features
            inputs: Dict containing input data
            current_states: [B, P, 4] current states
            neighbor_current_mask: [B, Pn] mask for invalid neighbors
            encoding_pooled: [B, D] pooled encoding

        Returns:
            Dict containing prediction and turn_indicator_logit
        """
        B = encoding.shape[0]
        P = 1 + self._predicted_neighbor_num

        sampled_trajectories = inputs["sampled_trajectories"].reshape(
            B, P, (1 + self._future_len) * 4
        )

        if self._model_type == "flow_matching":
            return self._inference_flow_matching(
                encoding,
                inputs,
                current_states,
                neighbor_current_mask,
                encoding_pooled,
                sampled_trajectories,
            )
        elif self._model_type in {"x_start", "noise", "score", "v"}:
            return self._inference_x_start(
                encoding,
                inputs,
                current_states,
                neighbor_current_mask,
                encoding_pooled,
                sampled_trajectories,
            )
        else:
            raise NotImplementedError(f"Unknown model type {self._model_type}")

    def forward(self, encoding, inputs):
        """
        Diffusion decoder process.

        Args:
            encoding: [B, N, D] encoded features
            inputs: Dict
                {
                    ...
                    "ego_current_state": current ego states,
                    "neighbor_agent_past": past and current neighbor states,

                    "sampled_trajectories": sampled current-future ego & neighbor states,        [B, P, 1 + self._future_len, 4]
                    "delay": number of initial steps to keep fixed (>=0),
                    [training-only] "diffusion_time": timestep of diffusion process $t \in [0, 1]$,              [B]
                    ...
                }

        Returns:
            decoder_outputs: Dict
                {
                    ...
                    [training-only] "model_output": Predicted future states, [B, P, 1 + self._future_len, 4]
                    [inference-only] "prediction": Predicted future states, [B, P, self._future_len, 4]
                    "turn_indicator_logit": Turn indicator prediction, [B, TURN_INDICATOR_OUTPUT_DIM]
                    ...
                }

        """
        # Common preprocessing
        current_states, neighbor_current_mask, ego_current, neighbors_current = (
            self._prepare_current_states(inputs)
        )

        B, P, _ = current_states.shape
        assert P == (1 + self._predicted_neighbor_num)

        encoding_pooled = self._pool_encoding(encoding)

        # Dispatch to training or inference
        if self.training:
            return self._forward_training(encoding, inputs, neighbor_current_mask, encoding_pooled)
        else:
            return self._forward_inference(
                encoding, inputs, current_states, neighbor_current_mask, encoding_pooled
            )
