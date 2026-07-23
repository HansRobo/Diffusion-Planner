from argparse import Namespace

import torch
import torch.nn as nn
from timm.layers import Mlp

import diffusion_planner.model.diffusion_utils.dpm_solver_pytorch as dpm
from diffusion_planner.dimensions import OUTPUT_T, TURN_INDICATOR_OUTPUT_DIM
from diffusion_planner.loss import (
    compute_ego_edge_points,
    compute_neighbor_collision_penalty,
    compute_road_border_penalty,
    hybrid_waypoint_loss,
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
from diffusion_planner.model.module.mixer import MixerBlock
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer


class TurnIndicatorHead(nn.Module):
    """Stateless turn-intent classifier, gradient-isolated from the policy.

    Reads (1) planned-trajectory features, (2) a learned single-query attention
    probe over the scene tokens, (3) the global route condition, and (4) current
    ego dynamics. All upstream model features are detached inside the head, so no
    caller can leak the auxiliary gradient into the encoder or diffusion policy.

    Deliberately stateless: it never sees the vehicle's own signal history, so
    there is no copy shortcut in training and no self-feedback loop at deployment.
    "Keep" semantics belong to the deployment node (hysteresis/debounce), not to
    model memory.
    """

    def __init__(
        self,
        hidden_dim: int,
        trajectory_dim: int,
        proprio_dim: int = 6,
        num_heads: int = 4,
    ):
        super().__init__()
        self.trajectory_encoder = Mlp(
            in_features=trajectory_dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.proprio_encoder = Mlp(
            in_features=proprio_dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.scene_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.classifier = Mlp(
            in_features=4 * hidden_dim,
            hidden_features=2 * hidden_dim,
            out_features=TURN_INDICATOR_OUTPUT_DIM,
            act_layer=nn.GELU,
            drop=0.0,
        )

    def forward(
        self,
        trajectory_features: torch.Tensor,
        scene_tokens: torch.Tensor,
        route_condition: torch.Tensor,
        proprioception: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            trajectory_features: ``[B, trajectory_dim]`` subsampled planned trajectory.
            scene_tokens: ``[B, N, H]`` encoder tokens (padding rows are exact zeros).
            route_condition: ``[B, H]`` global route AdaLN condition.
            proprioception: ``[B, 6]`` normalized vx/vy/ax/ay/steering/yaw-rate.
        """
        trajectory_features = trajectory_features.detach()
        scene_tokens = scene_tokens.detach()
        route_condition = route_condition.detach()
        proprioception = proprioception.detach()
        trajectory_embedding = self.trajectory_encoder(trajectory_features)
        proprio_embedding = self.proprio_encoder(proprioception)
        query = (trajectory_embedding + proprio_embedding + route_condition)[:, None]
        padding_mask = torch.all(scene_tokens == 0, dim=-1)
        scene_readout = self.scene_attention(
            query,
            scene_tokens,
            scene_tokens,
            key_padding_mask=padding_mask,
            need_weights=False,
        )[0][:, 0]
        head_input = torch.cat(
            [scene_readout, route_condition, trajectory_embedding, proprio_embedding], dim=-1
        )
        return self.classifier(head_input)


class GlobalRouteEncoder(nn.Module):
    """Compress ordered route geometry into the HDP AdaLN condition."""

    def __init__(
        self,
        route_num: int,
        route_len: int,
        hidden_dim: int,
        drop_path_rate: float,
        tokens_mlp_dim: int = 32,
        channels_mlp_dim: int = 64,
    ):
        super().__init__()
        route_points = route_num * route_len
        self._route_shape = (route_num, route_len)
        self.channel_pre_project = Mlp(
            in_features=4,
            hidden_features=channels_mlp_dim,
            out_features=channels_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.token_pre_project = Mlp(
            in_features=route_points,
            hidden_features=tokens_mlp_dim,
            out_features=tokens_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.mixer = MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate)
        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(
            in_features=channels_mlp_dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )

    def forward(self, route_lanes: torch.Tensor) -> torch.Tensor:
        if route_lanes.dim() != 4 or tuple(route_lanes.shape[1:3]) != self._route_shape:
            raise ValueError(
                f"Expected route lanes [B,{self._route_shape[0]},{self._route_shape[1]},D], "
                f"got {tuple(route_lanes.shape)}"
            )
        if route_lanes.shape[-1] < 4:
            raise ValueError(
                f"Route lanes need at least four geometry channels, got {route_lanes.shape[-1]}"
            )

        # Match the official NuPlan HDP route conditioner: ordered x/y geometry and
        # direction vectors are mixed across the complete route, then pooled once.
        route_geometry = route_lanes[..., :4]
        batch_size = route_geometry.shape[0]
        valid_route = torch.any(route_geometry != 0, dim=-1).flatten(1).any(dim=1)
        x = route_geometry.reshape(batch_size, -1, 4)
        x = self.channel_pre_project(x)
        x = self.token_pre_project(x.transpose(1, 2)).transpose(1, 2)
        x = self.mixer(x).mean(dim=1)
        x = self.emb_project(self.norm(x))
        return x * valid_route.to(dtype=x.dtype).unsqueeze(-1)


def compute_training_loss(
    model: nn.Module,
    inputs: dict[str, torch.Tensor],
    futures: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    args: Namespace,
    collision_futures: tuple[torch.Tensor, torch.Tensor] | None = None,
):
    # Keep the low-level helper safe when called with a legacy Namespace: policy
    # training must not silently re-enable the auxiliary turn-indicator branch.
    training_stage = getattr(args, "supervised_training_stage", "policy")
    if training_stage not in {"joint", "policy"}:
        raise ValueError(
            f"compute_training_loss supports joint/policy stages, got {training_stage!r}"
        )
    norm = args.state_normalizer
    if not args.use_velocity_representation:
        raise ValueError("HDP training requires velocity representation")
    if args.diffusion_model_type != "x_start" or args.diffusion_supervision_type != "x_start":
        raise ValueError("HDP training requires x_start prediction and supervision")
    hybrid_window = args.hybrid_loss_window
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
    # bf16 autocast is scoped to the model forward ONLY: noising, SDE schedule math and
    # every loss below stay fp32 (the diffusion-sensitive parts). Off on CPU.
    use_bf16 = getattr(args, "amp_dtype", "off") == "bf16" and gt_future.device.type == "cuda"

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
    if training_stage == "policy":
        merged_inputs["_skip_turn_indicator"] = True
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
        _, decoder_output = model(merged_inputs)
    pred_x_start = decoder_output["model_output"].float()
    ego_diffusion_loss = torch.sum((pred_x_start[:, 0] - all_gt[:, 0]) ** 2, dim=-1)
    ego_pred_velocity = pred_x_start[:, 0]
    ego_pred_velocity_raw = inverse_normalize_ego_velocity(ego_pred_velocity, norm)
    ego_waypoint_loss = hybrid_waypoint_loss(
        ego_pred_velocity_raw,
        ego_future,
        W=hybrid_window,
    )
    dpm_loss = ego_diffusion_loss[:, None, :]

    horizon = min(int(args.ego_prediction_horizon), dpm_loss.shape[-1])

    loss = {}
    ego_loss_horizon = dpm_loss[:, 0, :horizon]
    ego_waypoint_horizon = ego_waypoint_loss[:, :horizon]
    loss["ego_planning_loss"] = ego_loss_horizon.mean()
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
        loss["road_border_loss"] = rb_loss[:, :horizon].mean()
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
        loss["neighbor_collision_loss"] = nc_loss[:, :horizon].mean()
    else:
        loss["neighbor_collision_loss"] = torch.tensor(0.0, device=dpm_loss.device)

    if not torch.isfinite(dpm_loss).all():
        raise FloatingPointError(f"diffusion loss is non-finite, z={z}")

    if training_stage == "policy":
        return loss

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
    # The generated branch consumes x_start predicted at the sampled diffusion time;
    # near t=1 that trajectory is close to the conditional mean and teaches the head
    # an input distribution inference never produces. Use a normalized quality
    # weighting so low-noise examples dominate without shrinking this branch's loss.
    generated_quality = (1.0 - t).clamp_min(0.0)
    generated_turn_indicator_loss = (
        generated_turn_indicator_loss * generated_quality
    ).sum() / generated_quality.sum().clamp_min(1e-6)
    expert_turn_indicator_loss = expert_turn_indicator_loss.mean()
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

    non_finite_losses = [
        key
        for key, value in loss.items()
        if torch.is_tensor(value) and not torch.isfinite(value).all()
    ]
    if non_finite_losses:
        raise FloatingPointError(f"non-finite training losses: {non_finite_losses}")

    return loss


def compute_turn_indicator_head_training_loss(
    model: nn.Module,
    inputs: dict[str, torch.Tensor],
    ego_future: torch.Tensor,
    args: Namespace,
) -> dict[str, torch.Tensor]:
    """Train only the intent head on expert or deployment trajectories.

    The caller freezes and evaluates the trajectory policy. Expert mode skips diffusion
    sampling while the new head learns clean intent features. Deployment mode uses the
    exact final DPM trajectory and retains gradients only for the detached head calls.
    """
    if getattr(args, "supervised_training_stage", "policy") != "turn_indicator":
        raise ValueError("head-only loss requires supervised_training_stage='turn_indicator'")

    batch_size = ego_future.shape[0]
    model_ref = getattr(model, "module", model)
    decoder = model_ref.decoder
    if model_ref.training or decoder.training:
        raise RuntimeError("The frozen trajectory policy must be in eval mode for head training")

    training_mode = getattr(args, "turn_indicator_head_training_mode", "deployment")
    if training_mode not in {"expert", "deployment"}:
        raise ValueError(f"Unsupported turn-indicator head training mode: {training_mode!r}")
    normalized_expert = normalize_ego_state(ego_future[:, None], args.state_normalizer)
    head_inputs = {
        **inputs,
        "turn_indicator_trajectories": normalized_expert,
    }
    if training_mode == "expert":
        head_inputs["_turn_indicator_expert_only"] = True
    else:
        head_inputs["sampled_trajectories"] = torch.zeros(
            batch_size,
            1 + decoder._predicted_neighbor_num,
            decoder._future_len,
            4,
            dtype=torch.float32,
            device=ego_future.device,
        )
    use_bf16 = getattr(args, "amp_dtype", "off") == "bf16" and ego_future.device.type == "cuda"
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
        _, decoder_output = model(head_inputs)

    expert_logit = decoder_output["turn_indicator_expert_logit"].float()
    target = make_turn_indicator_gt(inputs["turn_indicators"])
    expert_loss = nn.functional.cross_entropy(expert_logit, target)
    if training_mode == "expert":
        total = expert_loss
        generated_logit = None
        generated_loss = None
    else:
        generated_logit = decoder_output["turn_indicator_logit"].float()
        generated_loss = nn.functional.cross_entropy(generated_logit, target)
        generated_weight = float(getattr(args, "turn_indicator_generated_loss_weight", 1.0))
        expert_weight = float(getattr(args, "turn_indicator_expert_loss_weight", 1.0))
        total = (generated_weight * generated_loss + expert_weight * expert_loss) / max(
            generated_weight + expert_weight, 1e-12
        )
    if not torch.isfinite(total):
        raise FloatingPointError("turn-indicator head loss is non-finite")

    with torch.no_grad():
        expert_accuracy = (expert_logit.argmax(dim=-1) == target).float().mean()
        accuracy = (
            (generated_logit.argmax(dim=-1) == target).float().mean()
            if generated_logit is not None
            else expert_accuracy
        )
    result = {
        "turn_indicator_loss": total,
        "turn_indicator_expert_loss": expert_loss.detach(),
        "turn_indicator_accuracy": accuracy,
        "turn_indicator_expert_accuracy": expert_accuracy,
    }
    if generated_loss is not None:
        result["turn_indicator_generated_loss"] = generated_loss.detach()
        result["turn_indicator_generated_accuracy"] = accuracy
    return result


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        configured_indicator_dim = int(
            getattr(config, "turn_indicator_output_dim", TURN_INDICATOR_OUTPUT_DIM)
        )
        if configured_indicator_dim != TURN_INDICATOR_OUTPUT_DIM:
            raise ValueError(
                "HDP turn intent requires exactly "
                f"{TURN_INDICATOR_OUTPUT_DIM} classes, got {configured_indicator_dim}"
            )
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
        if not 1 <= config.ego_prediction_horizon <= config.future_len:
            raise ValueError("ego_prediction_horizon must be in [1, future_len]")
        if not 1 <= config.hybrid_loss_window <= config.future_len:
            raise ValueError("hybrid_loss_window must be in [1, future_len]")
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
        # 16 points at 0.5 s spacing including the endpoint, with headings: turn intent
        # lives in heading change and the trajectory tail, which the historical
        # xy-only [::10] slice (0.1-7.1 s) could not see.
        self._turn_trajectory_dim = 4 * (self._future_len // 5)
        self.turn_indicator_predictor = TurnIndicatorHead(
            hidden_dim=config.hidden_dim,
            trajectory_dim=self._turn_trajectory_dim,
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

        # Construct new conditioning modules only after the historical model is fully
        # initialized. This keeps every pre-route parameter bitwise identical for a fixed
        # seed, so route AdaLN is a controlled architecture change rather than a seed change.
        self.global_route_encoder = GlobalRouteEncoder(
            route_num=config.route_num,
            route_len=config.route_len,
            hidden_dim=config.hidden_dim,
            drop_path_rate=config.encoder_drop_path_rate,
        )
        self.global_route_encoder.apply(_basic_init)

    @property
    def sde(self):
        return self._sde

    def _compute_turn_indicator(
        self,
        ego_trajectory,
        encoding,
        global_route_condition,
        ego_current_state,
    ):
        """Compute turn-indicator state logits (head detaches scene/route internally).

        Args:
            ego_trajectory: [B, 4 * (T // 5)] flattened trajectory features
            encoding: [B, N, D] scene tokens
            global_route_condition: [B, D] route AdaLN condition
            ego_current_state: [B, 10] normalized current ego state

        Returns:
            turn_indicator_logit: [B, TURN_INDICATOR_OUTPUT_DIM]
        """
        if ego_current_state.ndim != 2 or ego_current_state.shape[1] < 10:
            raise ValueError(
                f"ego_current_state must be [B, >=10], got {tuple(ego_current_state.shape)}"
            )
        return self.turn_indicator_predictor(
            ego_trajectory,
            encoding,
            global_route_condition,
            ego_current_state[:, 4:10],
        )

    def _turn_indicator_features(self, normalized_ego_future):
        """Subsample [B, 1, T, 4] normalized waypoints to the head's feature vector."""
        B = normalized_ego_future.shape[0]
        return normalized_ego_future[:, 0, 4::5, :].reshape(B, self._turn_trajectory_dim)

    def _ego_velocity_to_waypoints(self, ego_velocity):
        ego_velocity = inverse_normalize_ego_velocity(ego_velocity, self._state_normalizer)
        return velocity_to_waypoints(ego_velocity)

    def _normalize_ego_future(self, ego_future):
        return normalize_ego_state(ego_future, self._state_normalizer)

    def _turn_indicator_trajectory_from_latent(self, latent):
        ego_future = self._ego_velocity_to_waypoints(latent[:, :1])
        return self._turn_indicator_features(self._normalize_ego_future(ego_future))

    def _latent_to_prediction(self, latent):
        return self._ego_velocity_to_waypoints(latent[:, :1])

    @staticmethod
    def _training_x_start_latent(model_output):
        """Build a detached inference-like trajectory for turn-head supervision."""
        return model_output.detach()

    def _forward_training(self, encoding, inputs, global_route_condition):
        """Forward pass for training mode.

        Args:
            encoding: [B, N, D] encoded features
            inputs: Dict containing sampled_trajectories, gt_trajectories, diffusion_time, etc.

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
            global_route_condition,
        ).reshape(B, P, -1, 4)
        output = {"model_output": model_output}
        if inputs.get("_skip_turn_indicator", False):
            return output

        gt_trajectories = inputs["gt_trajectories"].reshape(B, P, self._future_len, 4)
        expert_trajectories = inputs.get("turn_indicator_trajectories", gt_trajectories).reshape(
            B, P, self._future_len, 4
        )
        expert_ego_trajectory = self._turn_indicator_features(expert_trajectories[:, :1])
        # The head detaches scene tokens and route condition internally, keeping the
        # deployment head without letting its auxiliary classification loss reshape
        # the HDP scene condition or diffusion policy.
        expert_logit = self._compute_turn_indicator(
            expert_ego_trajectory,
            encoding,
            global_route_condition,
            inputs["ego_current_state"],
        )

        generated_latent = self._training_x_start_latent(model_output)
        generated_ego_trajectory = self._turn_indicator_trajectory_from_latent(generated_latent)
        generated_logit = self._compute_turn_indicator(
            generated_ego_trajectory,
            encoding,
            global_route_condition,
            inputs["ego_current_state"],
        )

        output["turn_indicator_logit"] = generated_logit
        output["turn_indicator_expert_logit"] = expert_logit
        return output

    def _inference_x_start(
        self,
        encoding,
        inputs,
        global_route_condition,
        sampled_trajectories,
    ):
        """Inference using X-Start (DPM Solver) approach.

        Args:
            encoding: [B, N, D] encoded features
            inputs: Dict containing input data
            sampled_trajectories: [B, P, T * 4] sampled trajectories

        Returns:
            Dict containing prediction and turn_indicator_logit
        """
        B = encoding.shape[0]
        P = 1 + self._predicted_neighbor_num

        xT = sampled_trajectories.reshape(B, P, self._future_len, 4)

        noise_schedule = dpm.NoiseScheduleVP(
            beta_0=self._sde.beta_min,
            beta_1=self._sde.beta_max,
        )

        model_kwargs = {
            "cross_c": encoding,
            "ego_current_velocity": inputs["ego_current_state"][:, 4:6],
            "global_condition": global_route_condition,
        }

        def x_start_model_fn(x, diffusion_time):
            x = x.reshape(B, P, self._future_len, 4)
            return self.dit(x, diffusion_time, **model_kwargs)

        dpm_solver = dpm.DPM_Solver(
            x_start_model_fn,
            noise_schedule,
            model_type="x_start",
        )

        x0 = dpm_solver.sample(xT, steps=self._sample_steps, skip_type="logSNR")

        x0 = x0.reshape(B, P, self._future_len, 4)
        output = {"prediction": self._latent_to_prediction(x0)}
        if not inputs.get("_skip_turn_indicator", False):
            ego_trajectory = self._turn_indicator_trajectory_from_latent(x0)
            output["turn_indicator_logit"] = self._compute_turn_indicator(
                ego_trajectory,
                encoding,
                global_route_condition,
                inputs["ego_current_state"],
            )
        return output

    def _forward_inference(self, encoding, inputs, global_route_condition):
        """Forward pass for inference mode.

        Args:
            encoding: [B, N, D] encoded features
            inputs: Dict containing input data

        Returns:
            Dict containing prediction and turn_indicator_logit
        """
        B = encoding.shape[0]
        P = 1 + self._predicted_neighbor_num

        sampled_trajectories = inputs["sampled_trajectories"].reshape(B, P, self._future_len * 4)

        output = self._inference_x_start(
            encoding,
            inputs,
            global_route_condition,
            sampled_trajectories,
        )
        expert_trajectories = inputs.get("turn_indicator_trajectories")
        if expert_trajectories is not None:
            expert_trajectories = expert_trajectories.reshape(B, P, self._future_len, 4)
            expert_features = self._turn_indicator_features(expert_trajectories[:, :1])
            output["turn_indicator_expert_logit"] = self._compute_turn_indicator(
                expert_features,
                encoding,
                global_route_condition,
                inputs["ego_current_state"],
            )
        return output

    def _forward_turn_indicator_expert(self, encoding, inputs, global_route_condition):
        """Run the detached intent head without evaluating the diffusion policy."""
        expert_trajectories = inputs.get("turn_indicator_trajectories")
        if expert_trajectories is None:
            raise KeyError("turn_indicator_trajectories is required for expert head training")
        B = encoding.shape[0]
        expert_trajectories = expert_trajectories.reshape(B, -1, self._future_len, 4)
        expert_features = self._turn_indicator_features(expert_trajectories[:, :1])
        return {
            "turn_indicator_expert_logit": self._compute_turn_indicator(
                expert_features,
                encoding,
                global_route_condition,
                inputs["ego_current_state"],
            )
        }

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
        global_route_condition = inputs.get("_cached_global_route_condition")
        if global_route_condition is None:
            global_route_condition = self.global_route_encoder(inputs["route_lanes"])
            repeat_interleave = int(inputs.get("_global_route_repeat_interleave", 1))
            if repeat_interleave < 1:
                raise ValueError("_global_route_repeat_interleave must be >= 1")
            if repeat_interleave > 1:
                global_route_condition = global_route_condition.repeat_interleave(
                    repeat_interleave, dim=0
                )
        if global_route_condition.shape != (encoding.shape[0], encoding.shape[-1]):
            raise ValueError(
                "Global route condition must match [batch, hidden_dim], got "
                f"{tuple(global_route_condition.shape)} for encoding {tuple(encoding.shape)}"
            )

        if inputs.get("_turn_indicator_expert_only", False):
            return self._forward_turn_indicator_expert(
                encoding,
                inputs,
                global_route_condition,
            )

        # Dispatch to training or inference
        if self.training:
            return self._forward_training(encoding, inputs, global_route_condition)
        return self._forward_inference(encoding, inputs, global_route_condition)
