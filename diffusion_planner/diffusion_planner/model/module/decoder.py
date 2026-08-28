import random
from argparse import Namespace

import torch
import torch.nn as nn

import diffusion_planner.model.diffusion_utils.dpm_solver_pytorch as dpm
from diffusion_planner.dimensions import TURN_INDICATOR_OUTPUT_DIM
from diffusion_planner.loss import (
    compute_ego_edge_points,
    compute_neighbor_collision_penalty,
    compute_road_border_penalty,
    hybrid_loss,
    loss_func,
    make_turn_indicator_gt,
    velocity_to_waypoints,
    waypoints_to_velocity,
)
from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear
from diffusion_planner.model.flow_matching_utils.ode_solver import euler_integration
from diffusion_planner.model.module.dit import DiT
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer

# The decoder generates the ego trajectory only, but its output keeps an agent axis so that the
# ONNX graph, the ROS runtime and the offline tooling all stay on the same tensor layout. The
# neighbor rows of that axis are zero-filled placeholders and carry no prediction.


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


def generate_ego_prefix_mask(delay: torch.Tensor, max_len: int) -> torch.Tensor:
    """Prefix mask for the ego trajectory.

    Args:
        delay: A 1D tensor of shape (B,) with delay values.
        max_len: The length of the sequence (T + 1).

    Returns:
        A boolean tensor of shape (B, max_len, 1), True where the step is held fixed.
    """
    steps = torch.arange(max_len, device=delay.device).view(1, -1, 1)
    return steps <= delay.reshape(delay.shape[0], 1, 1)


def extract_ego_trajectory(trajectories: torch.Tensor, seq_len: int, state_dim: int):
    """Take the ego trajectory out of a tensor that may still carry an agent axis.

    Callers are allowed to hand over the padded layout (B, P, seq_len, state_dim) - or any
    flattened variant of it - because the ego row is at index 0 and everything else is a
    placeholder. The ego-only layout (B, seq_len, state_dim) is passed through unchanged.
    """
    batch_size = trajectories.shape[0]
    return trajectories.reshape(batch_size, -1, seq_len, state_dim)[:, 0]


def replace_current_state(x: torch.Tensor, current_state: torch.Tensor) -> torch.Tensor:
    """Return a trajectory tensor (B, T+1, D) with the first timestep replaced."""
    return torch.cat([current_state[:, None, :], x[:, 1:, :]], dim=1)


def add_current_xy(future: torch.Tensor, current_state: torch.Tensor) -> torch.Tensor:
    """Add current xy position to future xy channels without mutating the input."""
    xy = future[..., :2] + current_state[:, None, :2]
    return torch.cat([xy, future[..., 2:]], dim=-1)


def compute_training_loss(
    model: nn.Module,
    inputs: dict[str, torch.Tensor],
    futures: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    args: Namespace,
):
    norm = args.state_normalizer
    model_type = args.diffusion_model_type
    use_velocity = args.use_velocity_representation
    hybrid_omega = args.hybrid_loss_omega
    hybrid_window = args.hybrid_loss_window

    # Only the ego future is denoised. The neighbor ground truth never enters the denoising
    # input; it is used solely by the collision penalty further down.
    ego_future, neighbors_future, neighbor_future_mask = futures
    neighbors_future_valid = ~neighbor_future_mask  # [B, Pn, T]

    B, T, _ = ego_future.shape
    ego_current = inputs["ego_current_state"][:, :4]  # [B, 4]
    longitudinal_velocity = inputs["ego_current_state"][:, 4:5]

    eps = 1e-3
    t = torch.rand(B, device=ego_future.device) * (1 - eps) + eps  # [B,]
    t = t.view(B, 1, 1).expand(B, T + 1, 1)
    z = torch.randn_like(ego_future, device=ego_future.device)  # [B, T, 4]

    max_delay = 5
    delay = torch.randint(0, max_delay + 1, (B,), device=ego_future.device)  # [B,]
    prefix_mask = generate_ego_prefix_mask(delay, T + 1)  # [B, T+1, 1]
    mask_coeff = random.uniform(0.0, 1.0)
    curr_mask_time = torch.maximum(t * mask_coeff, torch.tensor(eps, device=ego_future.device))
    t = torch.where(prefix_mask, curr_mask_time, t)

    if use_velocity:
        full_traj = torch.cat([ego_current[:, None, :], ego_future], dim=1)  # [B, T+1, 4]
        gt_velocity = waypoints_to_velocity(full_traj)  # [B, T, 4]
        all_gt = torch.cat([ego_current[:, None, :], gt_velocity], dim=1)
    else:
        all_gt = torch.cat([ego_current[:, None, :], norm(ego_future)], dim=1)  # [B, T+1, 4]

    if model_type == "x_start":
        mean, std = VPSDE_linear().marginal_prob(all_gt[:, 1:, :], t[:, 1:, :])
        # mean([B, T, D]), std([B, T, 1]), z([B, T, D])
        xT = mean + std * z

        xT = torch.cat([all_gt[:, :1, :], xT], dim=1)
        xT = torch.where(prefix_mask, all_gt, xT)  # [B, 1 + T, 4]

        merged_inputs = {
            **inputs,
            "gt_trajectories": all_gt,
            "sampled_trajectories": xT,
            "diffusion_time": t,
            "prefix_mask": prefix_mask,
        }
        _, decoder_output = model(merged_inputs)  # [B, 1 + T, 4]
        model_output = decoder_output["model_output"][:, 1:, :]  # [B, T, 4]

        gt_target = all_gt[:, 1:, :]  # [B, T, 4]

        if use_velocity:
            # Hybrid loss: velocity L2 + omega * waypoint L2 (with detach window)
            dpm_loss = hybrid_loss(
                model_output,
                gt_target,
                omega=hybrid_omega,
                W=hybrid_window,
            )  # [B, T]
        else:
            loss_dict = loss_func(model_output, gt_target)
            heading_l2_loss = loss_dict["heading_l2_loss"]  # [B, T]
            position_lat_loss = loss_dict["position_lat_loss"]  # [B, T]
            position_lon_loss = loss_dict["position_lon_loss"]  # [B, T]

            # velocity weight
            velocity_weight = longitudinal_velocity * args.coeff_velocity
            velocity_weight = torch.abs(velocity_weight)
            velocity_weight = torch.clamp_min(velocity_weight, 1.0)  # [B, 1]
            position_lon_loss = position_lon_loss / velocity_weight

            # timestep weight
            timestep_weight = args.coeff_timestep
            assert T % len(timestep_weight) == 0, (
                f"Timestep {T} is not divisible by the number of timestep weights {len(timestep_weight)}"
            )
            unit = T // len(timestep_weight)
            for i in range(len(timestep_weight)):
                position_lat_loss[:, (i + 0) * unit : (i + 1) * unit] *= timestep_weight[i]
                position_lon_loss[:, (i + 0) * unit : (i + 1) * unit] *= timestep_weight[i]
                heading_l2_loss[:, (i + 0) * unit : (i + 1) * unit] *= timestep_weight[i]

            dpm_loss = (
                args.coeff_position_lat_loss * position_lat_loss
                + args.coeff_position_lon_loss * position_lon_loss
                + args.coeff_heading_l2_loss * heading_l2_loss
            )  # [B, T]

    elif model_type == "flow_matching":
        # t=0 is noise, t=1 is data
        xT = (1 - t[:, 1:, :]) * z + t[:, 1:, :] * all_gt[:, 1:, :]  # [B, T, 4]

        xT = torch.cat([all_gt[:, :1, :], xT], dim=1)
        merged_inputs = {
            **inputs,
            "gt_trajectories": all_gt,
            "sampled_trajectories": xT,
            "diffusion_time": t,
            "prefix_mask": prefix_mask,
        }
        _, decoder_output = model(merged_inputs)  # [B, 1 + T, 4]
        model_output = decoder_output["model_output"][:, 1:, :]  # [B, T, 4]

        target_v = all_gt[:, 1:, :] - z
        dpm_loss = torch.sum((model_output - target_v) ** 2, dim=-1)
    else:
        raise NotImplementedError(f"Unknown diffusion model type: {model_type}")

    loss = {}

    loss["ego_planning_loss"] = dpm_loss[:, : args.ego_prediction_horizon].mean()

    # Compute ego edge points for penalty losses
    need_ego_edge = model_type == "x_start" and (
        args.coeff_road_border_loss > 0 or args.coeff_neighbor_collision_loss > 0
    )
    if need_ego_edge:
        ego_pred = model_output  # [B, T, 4]
        if use_velocity:
            ego_pred_world = velocity_to_waypoints(ego_pred)
            ego_pred_world[..., :2] = ego_pred_world[..., :2] + ego_current[:, None, :2]
        else:
            ego_pred_world = ego_pred * norm.std.to(model_output.device) + norm.mean.to(
                model_output.device
            )  # [B, T, 4]
        ego_edge_points = compute_ego_edge_points(
            ego_pred_world, inputs["ego_shape"], n_interp=args.road_border_n_interp
        )
        denorm_inputs = args.observation_normalizer.inverse(inputs)

    # Road border collision loss (ego only, x_start mode)
    if args.coeff_road_border_loss > 0 and model_type == "x_start":
        rb_loss = compute_road_border_penalty(
            ego_edge_points,
            denorm_inputs["line_strings"],
            margin=args.road_border_margin,
        )  # [B, T]
        loss["road_border_loss"] = rb_loss.mean()
    else:
        loss["road_border_loss"] = torch.tensor(0.0, device=dpm_loss.device)

    # Neighbor collision loss (ego only, x_start mode). The neighbors come from the ground
    # truth futures, not from a prediction.
    if args.coeff_neighbor_collision_loss > 0 and model_type == "x_start":
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
    """Ego-only trajectory decoder.

    The denoising network sees one trajectory - the ego one - and conditions it on the scene
    memory, which already carries the observed histories of the neighbors. No neighbor future
    is generated, and none is read from the sampled trajectories either.
    """

    def __init__(self, config):
        super().__init__()

        dpr = config.decoder_drop_path_rate
        # Zero-filled neighbor rows appended to the prediction so downstream consumers keep
        # their tensor layout. They are placeholders, not predictions.
        self._dummy_neighbor_num = config.predicted_neighbor_num
        self._future_len = config.future_len

        self.dit = DiT(
            depth=config.decoder_depth,
            output_dim=(config.future_len + 1) * 4,  # x, y, cos, sin
            hidden_dim=config.hidden_dim,
            heads=config.num_heads,
            dropout=dpr,
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

    def _prepare_current_state(self, inputs):
        """Extract the ego current state and publish the neighbor validity mask.

        Args:
            inputs: Dict containing ego_current_state and neighbor_agents_past

        Returns:
            ego_current: [B, 4] ego current state
        """
        # Guidance functions read this mask to know which neighbor observations are real.
        neighbors_current = inputs["neighbor_agents_past"][:, :, -1, :4]
        inputs["neighbor_current_mask"] = torch.sum(torch.ne(neighbors_current, 0), dim=-1) == 0

        return inputs["ego_current_state"][:, :4]

    def _pad_dummy_neighbors(self, ego_trajectory: torch.Tensor) -> torch.Tensor:
        """Turn an ego trajectory [B, T, 4] into the padded layout [B, 1 + Pn, T, 4].

        The neighbor rows are zeros: they exist only so that consumers of the prediction keep
        indexing agent 0 for ego, exactly as before.
        """
        prediction = ego_trajectory[:, None]
        if self._dummy_neighbor_num == 0:
            return prediction
        dummy_shape = (
            prediction.shape[0],
            self._dummy_neighbor_num,
            *prediction.shape[2:],
        )
        return torch.cat([prediction, prediction.new_zeros(dummy_shape)], dim=1)

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

    def _denoise_with_agent_axis(self, x, t, cross_c):
        """Adapter for the solvers, which carry a formal agent axis of size one.

        Args:
            x: [B, 1, T+1, 4] or [B, 1, (T+1) * 4] noised ego trajectory.
            t: [B, 1, T+1, 1] or [B] diffusion time.
            cross_c: [B, N, D] scene memory.

        Returns:
            [B, 1, T+1, 4] denoised ego trajectory.
        """
        seq_len = 1 + self._future_len
        ego_x = extract_ego_trajectory(x, seq_len, 4)
        if t.dim() == 1:
            ego_t = t.view(-1, 1, 1).expand(-1, seq_len, 1)
        else:
            ego_t = extract_ego_trajectory(t, seq_len, 1)
        return self.dit(ego_x, ego_t, cross_c)[:, None]

    def _forward_training(self, encoding, inputs, encoding_pooled):
        """Forward pass for training mode.

        Args:
            encoding: [B, N, D] encoded features
            inputs: Dict containing sampled_trajectories, gt_trajectories, diffusion_time, etc.
            encoding_pooled: [B, D] pooled encoding

        Returns:
            Dict containing model_output [B, T+1, 4] and turn_indicator_logit
        """
        B = encoding.shape[0]
        seq_len = 1 + self._future_len

        sampled_trajectories = extract_ego_trajectory(inputs["sampled_trajectories"], seq_len, 4)
        diffusion_time = extract_ego_trajectory(inputs["diffusion_time"], seq_len, 1)

        gt_trajectories = extract_ego_trajectory(inputs["gt_trajectories"], seq_len, 4)
        ego_trajectory = gt_trajectories[:, 1::10, :2].reshape(B, 2 * (self._future_len // 10))
        turn_indicator_logit = self._compute_turn_indicator(ego_trajectory, encoding_pooled)

        return {
            "model_output": self.dit(sampled_trajectories, diffusion_time, encoding),
            "turn_indicator_logit": turn_indicator_logit,
        }

    def _inference_flow_matching(
        self,
        encoding,
        inputs,
        ego_current,
        encoding_pooled,
        sampled_trajectories,
    ):
        """Inference using Flow Matching approach.

        Args:
            encoding: [B, N, D] encoded features
            inputs: Dict containing input data
            ego_current: [B, 4] ego current state
            encoding_pooled: [B, D] pooled encoding
            sampled_trajectories: [B, 1, (1 + T) * 4] sampled ego trajectory

        Returns:
            Dict containing prediction and turn_indicator_logit
        """
        B = encoding.shape[0]

        x = sampled_trajectories
        NUM_STEP = 10

        def func(state, time):
            return self._denoise_with_agent_axis(state, time, encoding)

        x = euler_integration(func, x, NUM_STEP)
        x = extract_ego_trajectory(x, 1 + self._future_len, 4)  # [B, T+1, 4]
        ego_trajectory = x[:, 1::10, :2].reshape(B, 2 * (self._future_len // 10))
        turn_indicator_logit = self._compute_turn_indicator(ego_trajectory, encoding_pooled)
        if self._use_velocity:
            future = velocity_to_waypoints(x[:, 1:, :])
            future = add_current_xy(future, ego_current)
        else:
            future = self._state_normalizer.inverse(x)[:, 1:]
        return {
            "prediction": self._pad_dummy_neighbors(future),
            "turn_indicator_logit": turn_indicator_logit,
        }

    def _inference_x_start(
        self,
        encoding,
        inputs,
        ego_current,
        encoding_pooled,
        sampled_trajectories,
    ):
        """Inference using X-Start (DPM Solver) approach.

        Args:
            encoding: [B, N, D] encoded features
            inputs: Dict containing input data
            ego_current: [B, 4] ego current state
            encoding_pooled: [B, D] pooled encoding
            sampled_trajectories: [B, 1, (1 + T) * 4] sampled ego trajectory

        Returns:
            Dict containing prediction and turn_indicator_logit
        """
        B = encoding.shape[0]
        T_plus_1 = 1 + self._future_len

        action_prefix = extract_ego_trajectory(sampled_trajectories, T_plus_1, 4)
        action_prefix = replace_current_state(action_prefix, ego_current)
        xT = action_prefix.reshape(B, 1, T_plus_1 * 4)

        delay = inputs["delay"].to(device=action_prefix.device)
        # The solver keeps a formal agent axis of size one; the network itself has none.
        mask = generate_prefix_mask(delay, 1, T_plus_1)  # (B, 1, T_plus_1, 1)

        def prefix_constraint(xt, t, step):
            xt = extract_ego_trajectory(xt, T_plus_1, 4)
            xt = replace_current_state(xt, ego_current)
            return xt[:, None]

        model_wrapper_params = {
            "classifier_fn": self._guidance_fn,
            "classifier_kwargs": {
                "model": self._denoise_with_agent_axis,
                "model_condition": {"cross_c": encoding},
                "inputs": inputs,
                "observation_normalizer": self._observation_normalizer,
                "state_normalizer": self._state_normalizer,
            },
            "guidance_scale": self._guidance_scale,
            "guidance_type": "classifier" if self._guidance_fn is not None else "uncond",
        }

        noise_schedule = dpm.NoiseScheduleVP()

        model_fn = dpm.model_wrapper(
            self._denoise_with_agent_axis,
            noise_schedule,
            model_type=self._model_type,
            model_kwargs={"cross_c": encoding},
            **model_wrapper_params,
        )

        dpm_solver = dpm.DPM_Solver(model_fn, noise_schedule, correcting_xt_fn=prefix_constraint)

        x0 = dpm_solver.sample(xT, steps=10, prefix_mask=mask, skip_type="logSNR")

        x0 = extract_ego_trajectory(x0, T_plus_1, 4)  # [B, T+1, 4]
        ego_trajectory = x0[:, 1::10, :2].reshape(B, 2 * (self._future_len // 10))
        turn_indicator_logit = self._compute_turn_indicator(ego_trajectory, encoding_pooled)
        if self._use_velocity:
            future = velocity_to_waypoints(x0[:, 1:, :])
            future = add_current_xy(future, ego_current)
        else:
            future = self._state_normalizer.inverse(x0)[:, 1:]

        return {
            "prediction": self._pad_dummy_neighbors(future),
            "turn_indicator_logit": turn_indicator_logit,
        }

    def _forward_inference(self, encoding, inputs, ego_current, encoding_pooled):
        """Forward pass for inference mode.

        Args:
            encoding: [B, N, D] encoded features
            inputs: Dict containing input data
            ego_current: [B, 4] ego current state
            encoding_pooled: [B, D] pooled encoding

        Returns:
            Dict containing prediction and turn_indicator_logit
        """
        B = encoding.shape[0]

        sampled_trajectories = extract_ego_trajectory(
            inputs["sampled_trajectories"], 1 + self._future_len, 4
        ).reshape(B, 1, (1 + self._future_len) * 4)

        if self._model_type == "flow_matching":
            return self._inference_flow_matching(
                encoding,
                inputs,
                ego_current,
                encoding_pooled,
                sampled_trajectories,
            )
        elif self._model_type == "x_start":
            return self._inference_x_start(
                encoding,
                inputs,
                ego_current,
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

                    "sampled_trajectories": sampled current-future ego states,        [B, 1 + self._future_len, 4]
                        A padded [B, P, 1 + self._future_len, 4] tensor is also accepted; only the
                        ego row is read.
                    "delay": number of initial steps to keep fixed (>=0),
                    [training-only] "diffusion_time": timestep of diffusion process $t \in [0, 1]$,              [B, 1 + self._future_len, 1]
                    ...
                }

        Returns:
            decoder_outputs: Dict
                {
                    ...
                    [training-only] "model_output": Predicted ego states, [B, 1 + self._future_len, 4]
                    [inference-only] "prediction": Predicted states, [B, 1 + Pn, self._future_len, 4],
                        where row 0 is ego and the remaining rows are zero placeholders
                    "turn_indicator_logit": Turn indicator prediction, [B, TURN_INDICATOR_OUTPUT_DIM]
                    ...
                }

        """
        ego_current = self._prepare_current_state(inputs)

        # Pool encoding to get a fixed-size representation
        encoding_pooled = torch.mean(encoding, dim=1)  # [B, D]

        # Dispatch to training or inference
        if self.training:
            return self._forward_training(encoding, inputs, encoding_pooled)
        else:
            return self._forward_inference(encoding, inputs, ego_current, encoding_pooled)
