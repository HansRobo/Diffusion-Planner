from argparse import Namespace

import torch
import torch.nn as nn

import diffusion_planner.model.diffusion_utils.dpm_solver_pytorch as dpm
from diffusion_planner.dimensions import OUTPUT_T, TURN_INDICATOR_OUTPUT_DIM
from diffusion_planner.loss import (
    compute_ego_edge_points,
    compute_neighbor_collision_penalty,
    compute_road_border_penalty,
    hybrid_loss_components,
    inverse_normalize_ego_velocity,
    make_turn_indicator_gt,
    normalize_ego_state,
    normalize_ego_velocity,
    sample_diffusion_time,
    velocity_to_waypoints,
    waypoints_to_velocity,
)
from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear
from diffusion_planner.model.module.dit import DiT
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer


def compute_training_loss(
    model: nn.Module,
    inputs: dict[str, torch.Tensor],
    futures: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    args: Namespace,
    collision_futures: tuple[torch.Tensor, torch.Tensor] | None = None,
):
    norm = args.state_normalizer
    if not args.use_velocity_representation:
        raise ValueError("HDP training requires velocity representation")
    if args.diffusion_model_type != "x_start" or args.diffusion_supervision_type != "x_start":
        raise ValueError("HDP training requires x_start prediction and supervision")
    hybrid_window = args.hybrid_loss_window
    # bf16 autocast is scoped to the model forward ONLY: noising, SDE schedule math and
    # every loss below stay fp32 (the diffusion-sensitive parts). Off by default.
    use_bf16 = getattr(args, "amp_dtype", "off") == "bf16"

    ego_future, neighbors_future, neighbor_future_mask = futures
    if neighbors_future.shape[1] != 0:
        raise ValueError("HDP training is ego-only; neighbor future supervision is unsupported")
    neighbors_future_valid = ~neighbor_future_mask  # [B, Pn, V]
    if collision_futures is None:
        collision_neighbors_future = neighbors_future
        collision_neighbors_valid = neighbors_future_valid
    else:
        collision_neighbors_future, collision_neighbor_mask = collision_futures
        collision_neighbors_valid = ~collision_neighbor_mask

    B, Pn, T, _ = neighbors_future.shape
    gt_future = ego_future[:, None]  # [B, 1, T, 4]

    eps = 1e-3
    t = sample_diffusion_time(
        B,
        gt_future.device,
        eps,
        getattr(args, "diffusion_time_sample_method", "uniform"),
    )  # [B,]
    t_broadcast = t.view(B, 1, 1, 1)
    z = torch.randn_like(gt_future, device=gt_future.device)  # [B, P, T, 4]

    waypoint_gt = normalize_ego_state(gt_future, norm)
    ego_velocity_gt = waypoints_to_velocity(ego_future)  # [B, T, 4]
    all_gt = normalize_ego_velocity(ego_velocity_gt, norm)[:, None]

    model_ref = getattr(model, "module", model)
    sde = getattr(model_ref, "sde", None) or VPSDE_linear()
    alpha = sde.marginal_alpha(t_broadcast)
    std = sde.marginal_prob_std(t_broadcast)
    xT = alpha * all_gt + std * z
    merged_inputs = {
        **inputs,
        "gt_trajectories": all_gt,
        "turn_indicator_trajectories": waypoint_gt,
        "sampled_trajectories": xT,
        "diffusion_time": t,
    }
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
        _, decoder_output = model(merged_inputs)
    pred_x_start = decoder_output["model_output"].float()
    ego_diffusion_loss = torch.sum((pred_x_start[:, 0] - all_gt[:, 0]) ** 2, dim=-1)
    ego_pred_velocity = pred_x_start[:, 0]
    ego_pred_velocity_raw = inverse_normalize_ego_velocity(ego_pred_velocity, norm)
    _, ego_waypoint_loss = hybrid_loss_components(
        ego_pred_velocity,
        all_gt[:, 0],
        ego_pred_velocity_raw,
        ego_future,
        W=hybrid_window,
    )
    dpm_loss = ego_diffusion_loss[:, None, :]

    loss = {}
    loss["neighbor_prediction_loss"] = dpm_loss.new_zeros(())

    ego_loss_horizon = dpm_loss[:, 0, : args.ego_prediction_horizon]
    loss["ego_planning_loss"] = ego_loss_horizon.mean()
    ego_waypoint_horizon = ego_waypoint_loss[:, : args.ego_prediction_horizon]
    loss["ego_planning_hybrid_loss"] = ego_waypoint_horizon.mean()
    loss["ego_hdp_diffusion_loss"] = loss["ego_planning_loss"].detach()
    loss["ego_hdp_waypoint_loss"] = loss["ego_planning_hybrid_loss"].detach()

    # Compute ego edge points for penalty losses
    need_ego_edge = args.coeff_road_border_loss > 0 or args.coeff_neighbor_collision_loss > 0
    if need_ego_edge:
        ego_pred_world = velocity_to_waypoints(
            inverse_normalize_ego_velocity(pred_x_start[:, 0], norm)
        )
        ego_edge_points = compute_ego_edge_points(
            ego_pred_world, inputs["ego_shape"], n_interp=args.road_border_n_interp
        )
        # Only the penalty terms consume denormalized inputs — inverse just those keys
        # instead of every observation tensor each step.
        penalty_input_keys = []
        if args.coeff_road_border_loss > 0:
            penalty_input_keys.append("line_strings")
        if args.coeff_neighbor_collision_loss > 0:
            penalty_input_keys.append("neighbor_agents_past")
        denorm_inputs = args.observation_normalizer.inverse(
            {key: inputs[key] for key in penalty_input_keys}
        )

    # Road border collision loss (ego only, x_start mode)
    if args.coeff_road_border_loss > 0:
        rb_loss = compute_road_border_penalty(
            ego_edge_points,
            denorm_inputs["line_strings"],
            margin=args.road_border_margin,
        )  # [B, T]
        loss["road_border_loss"] = rb_loss.mean()
    else:
        loss["road_border_loss"] = torch.tensor(0.0, device=dpm_loss.device)

    # Neighbor collision loss (ego only, x_start mode)
    if args.coeff_neighbor_collision_loss > 0:
        nc_loss = compute_neighbor_collision_penalty(
            ego_edge_points,
            collision_neighbors_future,
            collision_neighbors_valid,
            denorm_inputs["neighbor_agents_past"],
            margin_vehicle=args.neighbor_collision_margin_vehicle,
            margin_pedestrian=args.neighbor_collision_margin_pedestrian,
            margin_bicycle=args.neighbor_collision_margin_bicycle,
        )  # [B, T]
        loss["neighbor_collision_loss"] = nc_loss.mean()
    else:
        loss["neighbor_collision_loss"] = torch.tensor(0.0, device=dpm_loss.device)

    assert not torch.isnan(dpm_loss).sum(), f"loss cannot be nan, z={z}"

    turn_indicator_logit = decoder_output["turn_indicator_logit"].float()
    turn_indicator_expert_logit = decoder_output.get(
        "turn_indicator_expert_logit", turn_indicator_logit
    ).float()
    turn_indicator_gt = make_turn_indicator_gt(inputs["turn_indicators"])  # [B,]
    generated_turn_indicator_loss = nn.functional.cross_entropy(
        turn_indicator_logit, turn_indicator_gt, reduction="none"
    )
    expert_turn_indicator_loss = nn.functional.cross_entropy(
        turn_indicator_expert_logit, turn_indicator_gt, reduction="none"
    )
    turn_indicator_change = inputs["turn_indicators"][:, -2] != inputs["turn_indicators"][:, -1]
    turn_indicator_coeff = torch.where(turn_indicator_change, 1.0, 0.05)
    generated_turn_indicator_loss = (generated_turn_indicator_loss * turn_indicator_coeff).mean()
    expert_turn_indicator_loss = (expert_turn_indicator_loss * turn_indicator_coeff).mean()
    generated_weight = float(getattr(args, "turn_indicator_generated_loss_weight", 1.0))
    expert_weight = float(getattr(args, "turn_indicator_expert_loss_weight", 1.0))
    turn_indicator_loss = (
        generated_weight * generated_turn_indicator_loss
        + expert_weight * expert_turn_indicator_loss
    ) / max(generated_weight + expert_weight, 1e-12)
    loss["turn_indicator_loss"] = turn_indicator_loss
    loss["turn_indicator_generated_loss"] = generated_turn_indicator_loss.detach()
    loss["turn_indicator_expert_loss"] = expert_turn_indicator_loss.detach()

    with torch.no_grad():
        generated_accuracy = (
            (turn_indicator_logit.argmax(dim=-1) == turn_indicator_gt).float().mean()
        )
        expert_accuracy = (
            (turn_indicator_expert_logit.argmax(dim=-1) == turn_indicator_gt).float().mean()
        )
        # Keep the historical key deployment-aligned: validation/inference also consumes the
        # model-generated trajectory rather than an expert trajectory.
        loss["turn_indicator_accuracy"] = generated_accuracy
        loss["turn_indicator_generated_accuracy"] = generated_accuracy
        loss["turn_indicator_expert_accuracy"] = expert_accuracy

    return loss


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        if getattr(config, "decoder_tokenization", "temporal") != "temporal":
            raise ValueError("HDP requires decoder_tokenization='temporal'")
        if config.future_len != OUTPUT_T:
            raise ValueError(f"HDP requires future_len={OUTPUT_T}, got {config.future_len}")
        dpr = config.decoder_drop_path_rate
        self._predicted_neighbor_num = config.predicted_neighbor_num
        if self._predicted_neighbor_num != 0:
            raise ValueError(
                "Hyper Diffusion Planner is ego-only; predicted_neighbor_num must be 0"
            )
        if not config.use_velocity_representation:
            raise ValueError("HDP requires use_velocity_representation=True")
        if (
            config.diffusion_model_type != "x_start"
            or config.diffusion_supervision_type != "x_start"
        ):
            raise ValueError("HDP requires x_start prediction and x_start supervision")
        self._future_len = config.future_len
        self._sde = VPSDE_linear()

        self.dit = DiT(
            depth=config.decoder_depth,
            output_dim=4,  # dx, dy, cos, sin per future time token
            hidden_dim=config.hidden_dim,
            heads=config.num_heads,
            dropout=dpr,
            future_len=config.future_len,
        )
        self.turn_indicator_predictor = nn.Linear(
            2 * (self._future_len // 10) + config.hidden_dim, TURN_INDICATOR_OUTPUT_DIM
        )

        self._state_normalizer: StateNormalizer = config.state_normalizer
        self._observation_normalizer: ObservationNormalizer = config.observation_normalizer
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

        # Official HDP-style DiT initialization: start every conditioned residual
        # branch and the output projection at zero, then learn them progressively.
        nn.init.normal_(self.dit.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.dit.t_embedder.mlp[2].weight, std=0.02)
        position = torch.arange(self._future_len, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(
            -torch.log(torch.tensor(100.0))
            * torch.arange(0, config.hidden_dim, 2, dtype=torch.float32)
            / config.hidden_dim
        )
        with torch.no_grad():
            self.dit.action_pos_emb[0, :, 0::2] = torch.sin(position * frequency)
            self.dit.action_pos_emb[0, :, 1::2] = torch.cos(position * frequency)
        for block in self.dit.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.dit.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.dit.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.dit.final_layer.proj[-1].weight, 0)
        nn.init.constant_(self.dit.final_layer.proj[-1].bias, 0)

    @property
    def sde(self):
        return self._sde

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
        ego_future = self._ego_velocity_to_waypoints(latent[:, :1])
        ego_future = self._normalize_ego_future(ego_future)
        return ego_future[:, 0, ::10, :2].reshape(B, 2 * (self._future_len // 10))

    def _latent_to_prediction(self, latent):
        return self._ego_velocity_to_waypoints(latent[:, :1])

    @staticmethod
    def _training_x_start_latent(model_output):
        """Build a detached inference-like trajectory for turn-head supervision."""
        return model_output.detach()

    def _forward_training(self, encoding, inputs, encoding_pooled):
        """Forward pass for training mode.

        Args:
            encoding: [B, N, D] encoded features
            inputs: Dict containing sampled_trajectories, gt_trajectories, diffusion_time, etc.
            encoding_pooled: [B, D] pooled encoding

        Returns:
            Dict containing model_output and turn_indicator_logit
        """
        B = encoding.shape[0]
        P = 1 + self._predicted_neighbor_num

        sampled_trajectories = inputs["sampled_trajectories"].reshape(B, P, self._future_len, 4)
        diffusion_time = inputs["diffusion_time"]

        model_output = self.dit(
            sampled_trajectories,
            diffusion_time,
            encoding,
            inputs["ego_current_state"][:, 4:6],
        ).reshape(B, P, -1, 4)
        output = {"model_output": model_output}
        if inputs.get("_skip_turn_indicator_training", False):
            return output

        gt_trajectories = inputs["gt_trajectories"].reshape(B, P, self._future_len, 4)
        expert_trajectories = inputs.get("turn_indicator_trajectories", gt_trajectories).reshape(
            B, P, self._future_len, 4
        )
        expert_ego_trajectory = expert_trajectories[:, 0, ::10, :2].reshape(
            B, 2 * (self._future_len // 10)
        )
        # Keep the deployment head without letting its auxiliary classification loss
        # reshape the HDP scene condition or diffusion policy.
        turn_encoding = encoding_pooled.detach()
        expert_logit = self._compute_turn_indicator(expert_ego_trajectory, turn_encoding)

        generated_latent = self._training_x_start_latent(model_output)
        generated_ego_trajectory = self._turn_indicator_trajectory_from_latent(generated_latent)
        generated_logit = self._compute_turn_indicator(generated_ego_trajectory, turn_encoding)

        output["turn_indicator_logit"] = generated_logit
        output["turn_indicator_expert_logit"] = expert_logit
        return output

    def _inference_x_start(
        self,
        encoding,
        inputs,
        encoding_pooled,
        sampled_trajectories,
    ):
        """Inference using X-Start (DPM Solver) approach.

        Args:
            encoding: [B, N, D] encoded features
            inputs: Dict containing input data
            encoding_pooled: [B, D] pooled encoding
            sampled_trajectories: [B, P, (1 + T) * 4] sampled trajectories

        Returns:
            Dict containing prediction and turn_indicator_logit
        """
        B = encoding.shape[0]
        P = 1 + self._predicted_neighbor_num

        xT = sampled_trajectories.reshape(B, P, self._future_len, 4)

        noise_schedule = dpm.NoiseScheduleVP()

        model_fn = dpm.model_wrapper(
            self.dit,
            noise_schedule,
            model_type="x_start",
            model_kwargs={
                "cross_c": encoding,
                "ego_current_velocity": inputs["ego_current_state"][:, 4:6],
            },
            guidance_type="uncond",
        )

        dpm_solver = dpm.DPM_Solver(model_fn, noise_schedule)

        x0 = dpm_solver.sample(xT, steps=self._sample_steps, skip_type="logSNR")

        x0 = x0.reshape(B, P, self._future_len, 4)
        ego_trajectory = self._turn_indicator_trajectory_from_latent(x0)
        turn_indicator_logit = self._compute_turn_indicator(ego_trajectory, encoding_pooled)
        x0 = self._latent_to_prediction(x0)

        return {"prediction": x0, "turn_indicator_logit": turn_indicator_logit}

    def _forward_inference(self, encoding, inputs, encoding_pooled):
        """Forward pass for inference mode.

        Args:
            encoding: [B, N, D] encoded features
            inputs: Dict containing input data
            encoding_pooled: [B, D] pooled encoding

        Returns:
            Dict containing prediction and turn_indicator_logit
        """
        B = encoding.shape[0]
        P = 1 + self._predicted_neighbor_num

        sampled_trajectories = inputs["sampled_trajectories"].reshape(B, P, self._future_len * 4)

        return self._inference_x_start(
            encoding,
            inputs,
            encoding_pooled,
            sampled_trajectories,
        )

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

                    "sampled_trajectories": noised future ego actions, [B, 1, self._future_len, 4]
                    [training-only] "diffusion_time": diffusion timestep in [0, 1], [B]
                    ...
                }

        Returns:
            decoder_outputs: Dict
                {
                    ...
                    [training-only] "model_output": Predicted future actions, [B, 1, self._future_len, 4]
                    [inference-only] "prediction": Predicted ego waypoints, [B, 1, self._future_len, 4]
                    "turn_indicator_logit": Turn indicator prediction, [B, TURN_INDICATOR_OUTPUT_DIM]
                    ...
                }

        """
        encoding_pooled = self._pool_encoding(encoding)

        # Dispatch to training or inference
        if self.training:
            return self._forward_training(encoding, inputs, encoding_pooled)
        return self._forward_inference(encoding, inputs, encoding_pooled)
