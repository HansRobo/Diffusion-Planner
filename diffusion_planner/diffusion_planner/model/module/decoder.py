import random
from argparse import Namespace
from functools import partial

import torch
import torch.nn as nn

import diffusion_planner.model.diffusion_utils.dpm_solver_pytorch as dpm
from diffusion_planner.dimensions import (
    CONTROL_DIM,
    POSE_DIM,
    TURN_INDICATOR_OUTPUT_DIM,
)
from diffusion_planner.loss import (
    ACTION_SPACE,
    compute_control_traj_loss,
    compute_ego_edge_points,
    compute_neighbor_collision_penalty,
    compute_road_border_penalty,
    make_turn_indicator_gt,
)
from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear
from diffusion_planner.model.flow_matching_utils.ode_solver import (
    euler_integration,
    heun_integration,
    rk4_integration,
)
from diffusion_planner.model.module.dit import DiT
from diffusion_planner.utils.normalizer import (
    ControlNormalizer,
    ObservationNormalizer,
    StateNormalizer,
)
from diffusion_planner.utils.unicycle_accel_curvature import action_to_traj4d, traj4d_to_action


def generate_future_prefix_mask(delay: torch.Tensor, num_agents: int, horizon: int) -> torch.Tensor:
    """Mask of the future steps a delayed ego has already committed to.

    Real-time chunking holds the first ``delay`` future steps at the ground truth. The
    trajectory version (``generate_prefix_mask``) counts a leading current-state slot, so
    it marks ``step <= delay``; the control tensor has no such slot, so the committed
    steps are ``step < delay`` and ``delay = 0`` pins nothing.

    Returns:
        [B, num_agents, horizon, 1] bool, True only on the ego row.
    """
    steps = torch.arange(horizon, device=delay.device).view(1, 1, -1, 1)
    ego_mask = steps < delay.reshape(-1, 1, 1, 1)  # (B, 1, horizon, 1)
    neighbor_mask = torch.zeros(
        (delay.shape[0], num_agents - 1, horizon, 1), dtype=torch.bool, device=delay.device
    )
    return torch.cat([ego_mask, neighbor_mask], dim=1)


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


def build_gt_representation(
    gt_future: torch.Tensor,
    current_states: torch.Tensor,
    inputs: dict[str, torch.Tensor],
    norm: StateNormalizer,
    control_norm: ControlNormalizer,
    obs_norm: ObservationNormalizer,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the control GT the diffusion runs on.

    The control tensor carries the T future steps only. A pose-space trajectory needs its
    current pose to be anchored somewhere, but T controls integrate into T poses on their
    own -- the initial velocity they start from is ``ego_current_state``, which the encoder
    already sees.

    Returns:
        all_gt: [B, P, T, CONTROL_DIM] normalized (accel, curvature).
        all_gt_pose: [B, P, T+1, 4] the same future in pose space (current pose first), for
            the turn indicator, which works in trajectory space.
    """
    # Control is only meaningful for ego. Neighbor control in the ego-centric frame is
    # ill-defined (the unicycle model wants the agent at the origin with zero heading),
    # so the neighbor channels are zero -- train with alpha_neighbor_loss = 0.
    raw_inputs = obs_norm.inverse(inputs)
    B, P, T = gt_future.shape[:3]

    ego_history = raw_inputs["ego_agent_past"]  # [B, T_hist, 4] raw
    ego_v0 = raw_inputs["ego_current_state"][:, 4:5]  # [B, 1] raw velocity
    ego_ctrl = traj4d_to_action(
        ACTION_SPACE, ego_history, gt_future[:, 0], t0_states={"v": ego_v0.squeeze(-1)}
    )  # [B, T, 2]

    all_gt = torch.zeros(B, P, T, CONTROL_DIM, device=gt_future.device)
    all_gt[:, 0] = control_norm(ego_ctrl)

    all_gt_pose = torch.cat([current_states[:, :, None, :], norm(gt_future)], dim=2)

    return all_gt, all_gt_pose


def compute_training_loss(
    model: nn.Module,
    inputs: dict[str, torch.Tensor],
    futures: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    args: Namespace,
):
    norm = args.state_normalizer
    control_norm = args.control_normalizer
    obs_norm = args.observation_normalizer
    model_type = args.diffusion_model_type
    D = CONTROL_DIM

    ego_future, neighbors_future, neighbor_future_mask = futures
    neighbors_future_valid = ~neighbor_future_mask  # [B, Pn, V]

    B, Pn, T, _ = neighbors_future.shape
    P = 1 + Pn
    ego_current, neighbors_current = (
        inputs["ego_current_state"][:, :4],
        inputs["neighbor_agents_past"][:, :Pn, -1, :4],
    )
    longitudinal_velocity = inputs["ego_current_state"][:, 4:5]
    neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
    neighbor_mask = torch.concat(
        (neighbor_current_mask.unsqueeze(-1), neighbor_future_mask), dim=-1
    )

    gt_future = torch.cat(
        [ego_future[:, None, :, :], neighbors_future[..., :]], dim=1
    )  # [B, P, T, 4]
    current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1)  # [B, P, 4]

    all_gt, all_gt_pose = build_gt_representation(
        gt_future,
        current_states,
        inputs,
        norm,
        control_norm,
        obs_norm,
    )
    # all_gt covers the future only, all_gt_pose additionally carries the current pose.
    all_gt[:, 1:][neighbor_future_mask] = 0.0
    all_gt_pose[:, 1:][neighbor_mask] = 0.0

    eps = 1e-3
    t = torch.rand(B, device=gt_future.device) * (1 - eps) + eps  # [B,]
    t = t.view(B, 1, 1, 1)
    t = t.expand(B, P, T, 1)
    z = torch.randn(B, P, T, D, device=gt_future.device)  # [B, P, T, D]

    max_delay = 0
    delay = torch.randint(0, max_delay + 1, (B,), device=gt_future.device)  # [B,]
    prefix_mask = generate_future_prefix_mask(delay, 1 + Pn, T)  # (B, P, T, 1)
    mask_coeff = random.uniform(0.0, 1.0)
    curr_mask_time = torch.maximum(t * mask_coeff, torch.tensor(eps, device=gt_future.device))
    t = torch.where(prefix_mask, curr_mask_time, t)

    if model_type == "x_start":
        mean, std = VPSDE_linear().marginal_prob(all_gt, t)
        # mean([B, P, T, D]), std([B, 1, T, 1]), z([B, P, T, D])
        xT = mean + std * z
        xT = torch.where(prefix_mask, all_gt, xT)  # [B, P, T, D]

        merged_inputs = {
            **inputs,
            "gt_trajectories": all_gt,
            "sampled_trajectories": xT,
            "diffusion_time": t,
            "prefix_mask": prefix_mask,
            "gt_trajectories_pose": all_gt_pose,
        }
        _, decoder_output = model(merged_inputs)
        model_output = decoder_output["model_output"]  # [B, P, T, D]

        dpm_loss = torch.sum((model_output - all_gt) ** 2, dim=-1)  # [B, P, T]

    elif model_type == "flow_matching":
        # t=0 is noise, t=1 is data
        t = t.reshape(-1, *([1] * (len(all_gt.shape) - 1)))  # [B, 1, 1, 1]
        xT = (1 - t) * z + t * all_gt  # [B, P, T, D]
        t = t.reshape(-1)  # [B,]

        merged_inputs = {
            **inputs,
            "gt_trajectories": all_gt,
            "sampled_trajectories": xT,
            "diffusion_time": t,
            "prefix_mask": prefix_mask,
            "gt_trajectories_pose": all_gt_pose,
        }
        _, decoder_output = model(merged_inputs)
        model_output = decoder_output["model_output"]  # [B, P, T, D]

        target_v = all_gt - z
        dpm_loss = torch.sum((model_output - target_v) ** 2, dim=-1)
    else:
        raise NotImplementedError(f"Unknown diffusion model type: {model_type}")

    masked_prediction_loss = dpm_loss[:, 1:, :][neighbors_future_valid]

    loss = {}

    if masked_prediction_loss.numel() > 0:
        loss["neighbor_prediction_loss"] = masked_prediction_loss.mean()
    else:
        loss["neighbor_prediction_loss"] = torch.tensor(0.0, device=masked_prediction_loss.device)

    loss["ego_planning_loss"] = dpm_loss[:, 0, : args.ego_prediction_horizon].mean()

    # Compute ego edge points for penalty losses
    need_ego_edge = model_type == "x_start" and (
        args.coeff_road_border_loss > 0 or args.coeff_neighbor_collision_loss > 0
    )
    if need_ego_edge:
        # Edge losses always operate in trajectory space, so the predicted control is
        # integrated back into waypoints first.
        ego_pred_world = action_to_traj4d(
            ACTION_SPACE,
            inputs["ego_agent_past"],
            model_output[:, 0],
            t0_states={"v": longitudinal_velocity.squeeze(-1)},
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

    # Neighbor collision loss (ego only, x_start mode)
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

    # Control-to-trajectory loss (sliding window)
    ctrl_traj_horizon = args.control_traj_loss_horizon
    if ctrl_traj_horizon > 0 and model_type == "x_start":
        ego_ctrl_pred = model_output[:, 0]  # [B, T, 2]

        raw_inputs = obs_norm.inverse(inputs)
        ego_current_raw = raw_inputs["ego_current_state"][:, :4]  # [B, 4]
        ego_v0_raw = raw_inputs["ego_current_state"][:, 4]  # [B]

        loss["control_traj_loss"] = compute_control_traj_loss(
            ego_ctrl_pred,
            ego_future,
            ego_current_raw,
            ego_v0_raw,
            control_norm,
            ctrl_traj_horizon,
        )
    else:
        loss["control_traj_loss"] = torch.tensor(0.0, device=dpm_loss.device)

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

        self.dit = DiT(
            depth=config.decoder_depth,
            output_dim=config.future_len * CONTROL_DIM,
            hidden_dim=config.hidden_dim,
            heads=config.num_heads,
            dropout=dpr,
            T=config.future_len,
            D=CONTROL_DIM,
        )
        self.turn_indicator_predictor = nn.Linear(
            2 * (self._future_len // 10) + config.hidden_dim, TURN_INDICATOR_OUTPUT_DIM
        )

        self._state_normalizer: StateNormalizer = config.state_normalizer
        self._observation_normalizer: ObservationNormalizer = config.observation_normalizer
        self._control_normalizer: ControlNormalizer = config.control_normalizer

        # self._guidance_fn = config.guidance_fn
        self._guidance_fn = (
            config.guidance_fn if config.__dict__.get("guidance_fn") is not None else None
        )
        self._guidance_scale = config.guidance_scale
        self._model_type = config.diffusion_model_type

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
        D = CONTROL_DIM

        sampled_trajectories = inputs["sampled_trajectories"].reshape(B, P, self._future_len, D)
        diffusion_time = inputs["diffusion_time"]

        # The denoised channels are (accel, curvature), so the turn indicator reads the
        # pose-space GT the trainer passes alongside them.
        gt_traj_pose = inputs["gt_trajectories_pose"].reshape(
            B, P, (1 + self._future_len), POSE_DIM
        )
        ego_trajectory = gt_traj_pose[:, 0, 1::10, :2].reshape(B, 2 * (self._future_len // 10))
        turn_indicator_logit = self._compute_turn_indicator(ego_trajectory, encoding_pooled)

        return {
            "model_output": self.dit(
                sampled_trajectories,
                diffusion_time,
                encoding,
                neighbor_current_mask,
            ).reshape(B, P, -1, D),
            "turn_indicator_logit": turn_indicator_logit,
        }

    def _denormalize(self, inputs, key):
        """Undo the observation normalization of one input tensor, ONNX-exportably."""
        norm = self._observation_normalizer._normalization_dict[key]
        x = inputs[key]
        return x * norm["std"].to(x.device) + norm["mean"].to(x.device)

    def denoised_to_trajectory(self, x, inputs):
        """Convert the denoised ego control [B, P, T, 2] into a trajectory [B, P, T, 4].

        Only the ego row carries a trained signal: neighbor control GT is zero (control in
        the ego-centric frame is ill-defined for them), so this branch trains with
        alpha_neighbor_loss = 0 and the neighbor rows are returned as zeros rather than
        integrated -- a unicycle rollout of 320 untrained rows costs real time and would
        read as a prediction that nothing behind it supports.
        """
        B, P = x.shape[:2]

        ego_ctrl = self._control_normalizer.inverse(x[:, 0])  # [B, T, 2]

        # Denormalize the two ego tensors elementwise instead of going through
        # ObservationNormalizer.inverse: that zeroes padded entries with `tensor[mask] = 0`,
        # whose advanced indexing exports as a Where whose condition is one rank short of
        # its operands -- PyTorch accepts it, TensorRT rejects the graph. Ego history is
        # never padded, so there is nothing to zero here.
        ego_agent_past_raw = self._denormalize(inputs, "ego_agent_past")
        ego_v0 = self._denormalize(inputs, "ego_current_state")[:, 4:5]  # [B, 1] raw velocity
        ego_traj = action_to_traj4d(
            ACTION_SPACE,
            ego_agent_past_raw,
            ego_ctrl,
            t0_states={"v": ego_v0.squeeze(-1)},
        )  # [B, T, 4]

        neighbor_traj = torch.zeros(
            B, P - 1, self._future_len, POSE_DIM, device=x.device, dtype=ego_traj.dtype
        )
        return torch.cat([ego_traj[:, None], neighbor_traj], dim=1)

    def _compute_turn_indicator_from_denoised(self, x, encoding_pooled):
        """Turn indicator logit for a denoised control tensor.

        The first two channels are (accel, curvature), not (x, y), so the trajectory slot
        is fed zeros and the prediction rests on the pooled encoding.
        """
        B = x.shape[0]
        ego_xy = torch.zeros(B, 2 * (self._future_len // 10), device=x.device, dtype=x.dtype)
        return self._compute_turn_indicator(ego_xy, encoding_pooled)

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
            sampled_trajectories: [B, P, (1 + T) * CONTROL_DIM] sampled controls

        Returns:
            Dict containing prediction and turn_indicator_logit
        """
        B = encoding.shape[0]
        P = 1 + self._predicted_neighbor_num
        D = CONTROL_DIM

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
        x = x.reshape(B, P, self._future_len, D)

        turn_indicator_logit = self._compute_turn_indicator_from_denoised(x, encoding_pooled)
        prediction = self.denoised_to_trajectory(x, inputs)

        return {
            "prediction": prediction,
            "turn_indicator_logit": turn_indicator_logit,
            "denoised": x,
        }

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
            sampled_trajectories: [B, P, (1 + T) * CONTROL_DIM] sampled controls

        Returns:
            Dict containing prediction and turn_indicator_logit
        """
        B = encoding.shape[0]
        P = 1 + self._predicted_neighbor_num
        D = CONTROL_DIM

        xT = sampled_trajectories

        delay = inputs["delay"].to(device=xT.device)
        # The already-committed steps keep their own (earlier) diffusion time; nothing else
        # is pinned, so the solver needs no xt correction.
        mask = generate_future_prefix_mask(delay, P, self._future_len)  # (B, P, T, 1)

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

        noise_schedule = dpm.NoiseScheduleVP()

        model_fn = dpm.model_wrapper(
            self.dit,
            noise_schedule,
            model_type=self._model_type,
            model_kwargs={
                "cross_c": encoding,
                "neighbor_current_mask": neighbor_current_mask,
            },
            D=D,
            **model_wrapper_params,
        )

        dpm_solver = dpm.DPM_Solver(model_fn, noise_schedule, D=D)

        x0 = dpm_solver.sample(xT, steps=10, prefix_mask=mask, skip_type="logSNR")

        x0 = x0.reshape(B, P, self._future_len, D)

        turn_indicator_logit = self._compute_turn_indicator_from_denoised(x0, encoding_pooled)
        prediction = self.denoised_to_trajectory(x0, inputs)

        return {
            "prediction": prediction,
            "turn_indicator_logit": turn_indicator_logit,
            "denoised": x0,
        }

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
        D = CONTROL_DIM

        sampled_trajectories = inputs["sampled_trajectories"].reshape(B, P, self._future_len * D)

        if self._model_type == "flow_matching":
            return self._inference_flow_matching(
                encoding,
                inputs,
                current_states,
                neighbor_current_mask,
                encoding_pooled,
                sampled_trajectories,
            )
        elif self._model_type == "x_start":
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

        encoding_pooled = torch.mean(encoding, dim=1)  # [B, D]

        # Dispatch to training or inference
        if self.training:
            return self._forward_training(encoding, inputs, neighbor_current_mask, encoding_pooled)
        else:
            return self._forward_inference(
                encoding, inputs, current_states, neighbor_current_mask, encoding_pooled
            )
