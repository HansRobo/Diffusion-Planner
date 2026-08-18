"""PlanTF-style regression decoder head on top of the Diffusion-Planner encoder.

Ported from planTF (https://github.com/jchengai/planTF, Cheng et al., ICRA 2024)
and adapted to the Diffusion-Planner interface:

- consumes the shared encoder output ``[B, token_num, hidden_dim]`` and the raw
  ``inputs`` dict, like :class:`diffusion_planner.model.module.decoder.Decoder`
- outputs (x, y, cos, sin) trajectories in the ``StateNormalizer`` space during
  training and denormalized ``prediction`` ``[B, P, T, 4]`` at inference, so
  validation / simulation / ROS consumers work unchanged
- diffusion-only inputs (``sampled_trajectories``, ``diffusion_time``,
  ``delay``) are accepted but ignored

``compute_plantf_training_loss`` is the planTF counterpart of
``decoder.compute_training_loss`` (same convention as ``grpo_utils`` mirroring
it): winner-takes-all regression over ``num_modes`` ego candidates + mode
classification, with DP's lat/lon/heading decomposition, velocity/timestep
weighting and penalty losses reused as-is.
"""

import math
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_planner.dimensions import TURN_INDICATOR_OUTPUT_DIM
from diffusion_planner.loss import (
    compute_ego_edge_points,
    compute_neighbor_collision_penalty,
    compute_road_border_penalty,
    make_turn_indicator_gt,
    velocity_to_waypoints,
)
from diffusion_planner.utils.normalizer import StateNormalizer


def bezier_basis(num_control_points: int, num_steps: int) -> torch.Tensor:
    """Bernstein/Bezier basis matrix ``[num_control_points, num_steps]``.

    Column ``t`` holds the Bernstein weights ``B_{i,n}(t)`` (n = num_control_points-1)
    at ``t = τ/(num_steps-1)`` for control point ``i``. A trajectory is the
    basis-weighted control points: ``traj[t] = Σ_i ctrl_i · basis[i, t]``.

    Why: the default head is a flat ``Linear(hidden, T*C)`` — the ``T`` output
    steps are independent linear read-outs with no temporal inductive bias, so
    temporal coherence is only bolted on afterwards (velocity-rep cumsum +
    smoothness penalty). A Bezier expansion makes time an explicit axis: the
    partition-of-unity (``Σ_i basis[i,t]=1``) and convex-hull property bound the
    curve by its control points, so it is *structurally* smooth (C∞) instead of
    smooth-by-penalty. See docs/plantf_head_development_notes.md §9.
    """
    n = num_control_points - 1
    t = torch.linspace(0.0, 1.0, num_steps)  # [T]
    i = torch.arange(num_control_points)  # [n+1]
    coeff = torch.tensor([math.comb(n, int(k)) for k in i], dtype=torch.float32)  # [n+1]
    # B_i(t) = C(n,i) t^i (1-t)^(n-i); torch defines 0**0 = 1 so the endpoints
    # (t=0 for i=0, t=1 for i=n) evaluate to 1 as required.
    t_pow = t[None, :] ** i[:, None].float()  # [n+1, T]
    tm_pow = (1.0 - t)[None, :] ** (n - i)[:, None].float()  # [n+1, T]
    return coeff[:, None] * t_pow * tm_pow  # [n+1, T]


class PlanTFTrajectoryHead(nn.Module):
    """Multi-modal ego trajectory head (planTF ``TrajectoryDecoder``).

    When ``num_control_points > 0`` the head regresses that many Bezier control
    points per mode/channel instead of ``future_steps`` free waypoints, then
    expands them through a fixed :func:`bezier_basis` matrix. This injects a
    temporal inductive bias (structural smoothness) the flat head lacks while
    keeping the exact same ``[B, K, future_steps, out_channels]`` output contract
    (velocity integration, WTA, zero-init, ONNX unchanged). The basis matrix is a
    constant buffer, so ONNX export is just an extra matmul — no recurrence.
    """

    def __init__(
        self,
        embed_dim,
        num_modes,
        future_steps,
        out_channels=4,
        ego_state_dim=0,
        predict_scale=False,
        num_control_points=0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_modes = num_modes
        self.future_steps = future_steps
        self.out_channels = out_channels
        self.ego_state_dim = ego_state_dim
        self.predict_scale = predict_scale
        # Temporal basis: 0 disables (flat per-step head); >0 regresses that many
        # Bezier control points and expands them to future_steps via the buffer.
        self.num_control_points = num_control_points
        if num_control_points > 0:
            self.register_buffer(
                "basis", bezier_basis(num_control_points, future_steps), persistent=False
            )
            self._out_steps = num_control_points
        else:
            self._out_steps = future_steps

        # Inject the current ego motion state (vx, vy, ax, ay, steering, yaw_rate)
        # into the ego token before branching into modes. Unlike the diffusion
        # decoder, the planTF head never sees the current state otherwise, so its
        # absolute-waypoint regression is not anchored to "where/how fast am I
        # now" — the cause of the start-point scatter / stop-jitter / divergence
        # that the diffusion head (which pins the current state) does not show.
        # See docs/plantf_dead_mode_improvement.md.
        if ego_state_dim > 0:
            self.ego_state_proj = nn.Sequential(
                nn.Linear(ego_state_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, embed_dim),
            )

        self.multimodal_proj = nn.Linear(embed_dim, num_modes * embed_dim)

        hidden = 2 * embed_dim
        self.loc = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self._out_steps * out_channels),
        )
        self.pi = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        # Optional per-point log-scale head for the Laplace NLL loss (planTF's
        # probabilistic regression). Only used when predict_scale is True.
        if predict_scale:
            self.scale = nn.Sequential(
                nn.Linear(embed_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, self._out_steps * out_channels),
            )

    def _expand(self, flat):
        """[B, K, _out_steps*C] -> [B, K, future_steps, C], applying the Bezier
        basis when enabled. Zero-init of the final Linear makes control points
        zero -> trajectory zero, identical to the flat head's zero-init prior."""
        if self.num_control_points > 0:
            ctrl = flat.view(-1, self.num_modes, self.num_control_points, self.out_channels)
            # traj[t] = Σ_c ctrl[c] · basis[c, t]
            return torch.einsum("bkcd,ct->bktd", ctrl, self.basis)
        return flat.view(-1, self.num_modes, self.future_steps, self.out_channels)

    def forward(self, x, ego_state=None):
        """
        Args:
            x: [B, embed_dim] ego token from the encoder.
            ego_state: [B, ego_state_dim] normalized current ego motion state,
                or None when ego_state_dim == 0.

        Returns:
            loc: [B, num_modes, future_steps, out_channels] mode trajectories.
            pi: [B, num_modes] mode logits.
            (predict_scale only) log_scale: [B, num_modes, future_steps, out_channels]
        """
        if self.ego_state_dim > 0 and ego_state is not None:
            x = x + self.ego_state_proj(ego_state)
        x = self.multimodal_proj(x).view(-1, self.num_modes, self.embed_dim)
        loc = self._expand(self.loc(x))
        pi = self.pi(x).squeeze(-1)
        if self.predict_scale:
            log_scale = self._expand(self.scale(x))
            return loc, pi, log_scale
        return loc, pi


class PlanTFCrossAttnHead(nn.Module):
    """K learnable mode queries that cross-attend to the full encoder memory.

    Unlike :class:`PlanTFTrajectoryHead` (which reshapes the single pooled +
    dropout ego token into K modes), each mode query attends over ALL encoder
    tokens (ego / neighbors / map / route), so the 80-point regression gets the
    scene context that the single-token bottleneck loses — a candidate fix for
    the tail divergence. See docs/plantf_head_development_notes.md §9. Same
    (loc, pi) output contract as PlanTFTrajectoryHead, so the rest of the decoder
    (velocity integration, WTA, zero-init, ONNX) is unchanged.
    """

    def __init__(
        self,
        embed_dim,
        num_modes,
        future_steps,
        out_channels=4,
        ego_state_dim=0,
        num_heads=8,
        depth=2,
        predict_scale=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modes = num_modes
        self.future_steps = future_steps
        self.out_channels = out_channels
        self.ego_state_dim = ego_state_dim
        self.predict_scale = predict_scale

        if ego_state_dim > 0:
            self.ego_state_proj = nn.Sequential(
                nn.Linear(ego_state_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, embed_dim),
            )

        self.mode_queries = nn.Parameter(torch.randn(num_modes, embed_dim) * 0.02)
        self.attn_layers = nn.ModuleList(
            [nn.MultiheadAttention(embed_dim, num_heads, batch_first=True) for _ in range(depth)]
        )
        self.attn_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(depth)])
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, 2 * embed_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(2 * embed_dim, embed_dim),
                )
                for _ in range(depth)
            ]
        )
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(depth)])

        hidden = 2 * embed_dim
        self.loc = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, future_steps * out_channels),
        )
        self.pi = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        if predict_scale:
            self.scale = nn.Sequential(
                nn.Linear(embed_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, future_steps * out_channels),
            )

    def forward(self, memory, memory_valid, ego_state=None):
        """
        Args:
            memory: [B, N, embed_dim] all encoder tokens.
            memory_valid: [B, N] bool, True where the token is valid.
            ego_state: [B, ego_state_dim] normalized current ego motion state.

        Returns:
            loc: [B, num_modes, future_steps, out_channels]
            pi:  [B, num_modes]
        """
        B = memory.shape[0]
        q = self.mode_queries.unsqueeze(0).expand(B, -1, -1)  # [B, K, D]
        if self.ego_state_dim > 0 and ego_state is not None:
            q = q + self.ego_state_proj(ego_state).unsqueeze(1)
        key_padding_mask = ~memory_valid  # True => ignore this key
        for attn, an, ffn, fn in zip(self.attn_layers, self.attn_norms, self.ffns, self.ffn_norms):
            a, _ = attn(q, memory, memory, key_padding_mask=key_padding_mask, need_weights=False)
            q = an(q + a)
            q = fn(q + ffn(q))
        loc = self.loc(q).view(B, self.num_modes, self.future_steps, self.out_channels)
        pi = self.pi(q).squeeze(-1)
        if self.predict_scale:
            log_scale = self.scale(q).view(B, self.num_modes, self.future_steps, self.out_channels)
            return loc, pi, log_scale
        return loc, pi


class PlanTFGRUHead(nn.Module):
    """Recurrent trajectory head: a GRU unrolls the ``future_steps`` waypoints.

    Same mlp-style mode formation as :class:`PlanTFTrajectoryHead` (reshape the
    single ego token into K modes), but instead of a flat ``Linear(hidden, T*C)``
    the per-step waypoints come from unrolling a GRU. The recurrence couples
    adjacent steps in the *architecture* (each step's hidden state carries the
    previous), giving the output an explicit temporal inductive bias the flat
    head lacks. This is NON-autoregressive — the GRU is fed a per-step input
    (learned temporal embedding + the mode context) rather than its own previous
    xy — so it exports as a single ONNX GRU op (no output-feedback loop). Same
    (loc, pi[, log_scale]) contract, so velocity integration / WTA / zero-init
    are unchanged. Note: RNN heads are more deploy-fragile than mlp/basis; this
    is an experimental toggle, not the deploy default.
    """

    def __init__(
        self,
        embed_dim,
        num_modes,
        future_steps,
        out_channels=4,
        ego_state_dim=0,
        predict_scale=False,
        gru_hidden=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modes = num_modes
        self.future_steps = future_steps
        self.out_channels = out_channels
        self.ego_state_dim = ego_state_dim
        self.predict_scale = predict_scale
        self.gru_hidden = gru_hidden or embed_dim

        if ego_state_dim > 0:
            self.ego_state_proj = nn.Sequential(
                nn.Linear(ego_state_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, embed_dim),
            )

        self.multimodal_proj = nn.Linear(embed_dim, num_modes * embed_dim)
        # Per-step temporal embedding: breaks the symmetry of feeding the same
        # context every step and gives the GRU an explicit time signal.
        self.step_emb = nn.Parameter(torch.randn(future_steps, embed_dim) * 0.02)
        self.h0_proj = nn.Linear(embed_dim, self.gru_hidden)
        self.gru = nn.GRU(embed_dim, self.gru_hidden, batch_first=True)

        hidden = 2 * embed_dim
        # loc/scale kept as Sequential ending in Linear so the shared zero-init
        # (PlanTFDecoder indexes head[-1]) works uniformly across head types.
        self.loc = nn.Sequential(nn.Linear(self.gru_hidden, out_channels))
        self.pi = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        if predict_scale:
            self.scale = nn.Sequential(nn.Linear(self.gru_hidden, out_channels))

    def _unroll(self, modes):
        """[B, K, D] mode embeddings -> [B, K, T, gru_hidden] GRU outputs."""
        B = modes.shape[0]
        m = modes.reshape(B * self.num_modes, self.embed_dim)  # [BK, D]
        h0 = torch.tanh(self.h0_proj(m)).unsqueeze(0)  # [1, BK, H]
        inp = self.step_emb.unsqueeze(0) + m.unsqueeze(1)  # [BK, T, D]
        out, _ = self.gru(inp, h0)  # [BK, T, H]
        return out.view(B, self.num_modes, self.future_steps, self.gru_hidden)

    def forward(self, x, ego_state=None):
        """
        Args:
            x: [B, embed_dim] ego token from the encoder.
            ego_state: [B, ego_state_dim] normalized current ego motion state.

        Returns:
            loc: [B, num_modes, future_steps, out_channels]
            pi:  [B, num_modes]
            (predict_scale only) log_scale: [B, num_modes, future_steps, out_channels]
        """
        if self.ego_state_dim > 0 and ego_state is not None:
            x = x + self.ego_state_proj(ego_state)
        modes = self.multimodal_proj(x).view(-1, self.num_modes, self.embed_dim)
        out = self._unroll(modes)
        loc = self.loc(out)  # [B, K, T, C]
        pi = self.pi(modes).squeeze(-1)
        if self.predict_scale:
            log_scale = self.scale(out)
            return loc, pi, log_scale
        return loc, pi


class PlanTFDecoder(nn.Module):
    """One-shot regression decoder with the same forward contract as ``Decoder``."""

    def __init__(self, config):
        super().__init__()

        self._predicted_neighbor_num = config.predicted_neighbor_num
        self._future_len = config.future_len
        self._num_modes = getattr(config, "num_modes", 6)
        # When True the heads regress per-step displacement instead of absolute
        # waypoints; _decode integrates it (cumsum) into absolute waypoints, which
        # gives the temporal continuity a per-timestep absolute regression lacks
        # (fixes the "comb" jitter and stalled forward progress of the planTF head).
        self._use_velocity = getattr(config, "use_velocity_representation", False)
        # Feed the current ego motion state (ego_current_state[:, 4:10] =
        # vx, vy, ax, ay, steering, yaw_rate) into the trajectory head so the
        # absolute-waypoint regression is anchored to the current motion, the
        # way the diffusion decoder is via its pinned current state.
        self._use_ego_state = getattr(config, "plantf_use_ego_state_in_head", True)
        self._ego_state_slice = slice(4, 10)
        ego_state_dim = self._ego_state_slice.stop - self._ego_state_slice.start

        # Optional toggles (default = original behavior) for combined ablation.
        # head_type "cross_attn": mode queries cross-attend to all encoder tokens
        #   instead of reshaping the single ego token (docs §9). route_rerank:
        #   at inference, pick the mode by route adherence among the top-k pi
        #   modes instead of argmax(pi) (only in `forward`, not the ONNX deploy
        #   graph, which lacks route_lanes).
        self._head_type = getattr(config, "plantf_head_type", "mlp")
        # head_type "basis": the mlp head, but the trajectory is a Bezier curve of
        # `plantf_basis_control_points` control points expanded over time (a
        # temporal inductive bias the flat per-step head lacks). Uses the same
        # single-ego-token mode formation as "mlp" (keeps deploy robustness).
        self._basis_control_points = (
            getattr(config, "plantf_basis_control_points", 8) if self._head_type == "basis" else 0
        )
        self._route_rerank = getattr(config, "plantf_route_rerank", False)
        self._route_rerank_topk = getattr(config, "plantf_route_rerank_topk", 3)
        self._observation_normalizer = getattr(config, "observation_normalizer", None)
        # Laplace NLL loss: the head additionally regresses a per-point log-scale.
        self._predict_scale = getattr(config, "plantf_use_laplace_nll", False)

        hidden_dim = config.hidden_dim
        head_ego_state_dim = ego_state_dim if self._use_ego_state else 0
        if self._head_type == "cross_attn":
            self.trajectory_head = PlanTFCrossAttnHead(
                embed_dim=hidden_dim,
                num_modes=self._num_modes,
                future_steps=self._future_len,
                out_channels=4,  # x, y, cos, sin
                ego_state_dim=head_ego_state_dim,
                num_heads=config.num_heads,
                predict_scale=self._predict_scale,
            )
        elif self._head_type == "gru":
            # Recurrent head: GRU unrolls the waypoints (temporal recurrence).
            # Experimental — more deploy-fragile than mlp/basis (RNN ONNX op).
            self.trajectory_head = PlanTFGRUHead(
                embed_dim=hidden_dim,
                num_modes=self._num_modes,
                future_steps=self._future_len,
                out_channels=4,  # x, y, cos, sin
                ego_state_dim=head_ego_state_dim,
                predict_scale=self._predict_scale,
            )
        else:
            self.trajectory_head = PlanTFTrajectoryHead(
                embed_dim=hidden_dim,
                num_modes=self._num_modes,
                future_steps=self._future_len,
                out_channels=4,  # x, y, cos, sin
                ego_state_dim=head_ego_state_dim,
                predict_scale=self._predict_scale,
                num_control_points=self._basis_control_points,
            )
        # planTF's agent_predictor, widened from (x, y) to DP's (x, y, cos, sin)
        self.neighbor_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.LayerNorm(2 * hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2 * hidden_dim, self._future_len * 4),
        )
        self.turn_indicator_predictor = nn.Linear(
            2 * (self._future_len // 10) + hidden_dim, TURN_INDICATOR_OUTPUT_DIM
        )
        self._state_normalizer: StateNormalizer = config.state_normalizer

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

        # Zero-out the output layers (same convention as Decoder.dit.final_layer).
        # Under winner-takes-all training, rarely-winning modes receive almost no
        # regression gradient; with Xavier output weights they keep emitting
        # white-noise trajectories, which the argmax(pi) mode selection can pick
        # at inference (randomly jagged outputs). Zero-init makes every mode
        # start at the normalized-space mean — a smooth prior trajectory — so an
        # undertrained mode degrades gracefully instead of into noise.
        zero_init_heads = [
            self.trajectory_head.loc,
            self.trajectory_head.pi,
            self.neighbor_predictor,
        ]
        # Zero-init the log-scale head too -> scale starts at exp(0)=1, a neutral
        # Laplace width, so early training is not dominated by scale noise.
        if self._predict_scale:
            zero_init_heads.append(self.trajectory_head.scale)
        for head in zero_init_heads:
            nn.init.constant_(head[-1].weight, 0)
            nn.init.constant_(head[-1].bias, 0)

    def _compute_turn_indicator(self, ego_trajectory, encoding_pooled):
        turn_indicator_input = torch.cat([ego_trajectory, encoding_pooled], dim=-1)
        return self.turn_indicator_predictor(turn_indicator_input)

    def _integrate_velocity(self, velocity: torch.Tensor, ego: bool) -> torch.Tensor:
        """Integrate per-step displacement into normalized absolute waypoints.

        ``velocity_to_waypoints`` cumsums the xy displacement (ego-centric metres,
        current pose = origin) and passes the heading channel through. The result
        is then mapped into the StateNormalizer space so every downstream consumer
        (WTA selection, smooth-L1 loss, mode metrics, prediction, ONNX) is unchanged.
        """
        wp = velocity_to_waypoints(velocity)  # [..., T, 4] absolute metres, heading passthrough
        idx = 0 if ego else 1  # ego uses row 0; neighbors share the same neighbor stats
        mean = self._state_normalizer.mean[idx].to(wp.device)  # [1, 4]
        std = self._state_normalizer.std[idx].to(wp.device)  # [1, 4]
        return (wp - mean) / std

    def _ego_state_feat(self, inputs):
        """Normalized current ego motion state fed to the trajectory head, or
        None when the feature is disabled. inputs are already observation-
        normalized when this runs (train_epoch / node apply the normalizer)."""
        if not self._use_ego_state:
            return None
        return inputs["ego_current_state"][:, self._ego_state_slice]

    def _decode(self, encoding, ego_state=None):
        """Run both heads on the encoder tokens.

        Returns (trajectory [B, K, T, 4], probability [B, K],
        neighbor_prediction [B, Pn, T, 4]), all in normalized space.
        """
        B = encoding.shape[0]
        Pn = self._predicted_neighbor_num
        if self._head_type == "cross_attn":
            memory_valid = torch.any(encoding != 0, dim=-1)  # [B, N]
            head_out = self.trajectory_head(encoding, memory_valid, ego_state)
        else:
            head_out = self.trajectory_head(encoding[:, 0], ego_state)
        if self._predict_scale:
            trajectory, probability, log_scale = head_out
        else:
            trajectory, probability = head_out
            log_scale = None
        neighbor_prediction = self.neighbor_predictor(encoding[:, 1 : 1 + Pn]).view(
            B, Pn, self._future_len, 4
        )
        if self._use_velocity:
            trajectory = self._integrate_velocity(trajectory, ego=True)
            neighbor_prediction = self._integrate_velocity(neighbor_prediction, ego=False)
        if self._predict_scale:
            return trajectory, probability, neighbor_prediction, log_scale
        return trajectory, probability, neighbor_prediction

    def _best_mode_trajectory(self, trajectory, probability, inputs=None):
        """Select one ego mode. By default argmax(pi); with route_rerank enabled
        (and route_lanes available), pick the mode that best follows the route
        among the top-k pi modes. gather keeps the batch axis dynamic for ONNX."""
        if (
            self._route_rerank
            and inputs is not None
            and "route_lanes" in inputs
            and self._observation_normalizer is not None
        ):
            index = self._route_gated_index(trajectory, probability, inputs)
        else:
            index = probability.argmax(dim=-1)
        index = index.view(-1, 1, 1, 1).expand(-1, 1, self._future_len, 4)
        return trajectory.gather(1, index).squeeze(1)

    def _route_gated_index(self, trajectory, probability, inputs):
        """Pick, among the top-k pi modes, the one whose waypoints stay closest
        to the ego-frame route centreline (mean over time of min distance to any
        route point). Both trajectory and route are de-normalized to metres.
        Returns [B] mode indices. See docs/plantf_head_development_notes.md §9."""
        B, K, _, _ = trajectory.shape
        dev = trajectory.device
        std0 = self._state_normalizer.std[0].to(dev)  # [1, 4]
        mean0 = self._state_normalizer.mean[0].to(dev)
        xy = (trajectory * std0 + mean0)[..., :2]  # [B, K, T, 2] metres

        route = self._observation_normalizer.inverse({"route_lanes": inputs["route_lanes"]})[
            "route_lanes"
        ].to(dev)  # [B, S, P, C] metres (zeroed where invalid)
        C = route.shape[-1]
        route_flat = route.reshape(B, -1, C)  # [B, M, C]
        route_xy = route_flat[..., :2]  # [B, M, 2]
        valid = torch.any(route_flat != 0, dim=-1)  # [B, M]

        d = torch.cdist(xy.reshape(B, K * xy.shape[2], 2), route_xy)  # [B, K*T, M]
        d = d.reshape(B, K, xy.shape[2], -1)
        d = d + torch.where(valid[:, None, None, :], 0.0, 1e6)  # ignore invalid points
        route_cost = d.min(dim=-1).values.mean(dim=-1)  # [B, K]

        k = min(self._route_rerank_topk, K)
        topk = probability.topk(k, dim=-1).indices  # [B, k]
        sel = route_cost.gather(1, topk).argmin(dim=-1)  # [B]
        return topk.gather(1, sel[:, None]).squeeze(1)  # [B]

    @staticmethod
    def _pool_encoding(encoding):
        # Pool only valid encoder tokens. The encoder zero-fills masked tokens.
        encoding_valid = torch.any(encoding != 0, dim=-1)  # [B, N]
        encoding_count = encoding_valid.sum(dim=1).clamp_min(1).unsqueeze(-1)
        return (encoding * encoding_valid.unsqueeze(-1)).sum(dim=1) / encoding_count

    def _subsampled_ego_xy(self, ego_trajectory):
        """Every-10th-step xy, matching gt_trajectories[:, 0, 1::10, :2] used in
        training (index 1 on the current-state-prepended axis = future step 0)."""
        return ego_trajectory[:, ::10, :2].reshape(-1, 2 * (self._future_len // 10))

    def forward_deploy(self, encoding, ego_current_state=None):
        """One-shot deployment path for the split ONNX export.

        Unlike the diffusion decoder there is no external denoising loop, so a
        single call maps the encoder output to the final prediction. The
        independent turn-indicator head is excluded (it re-encodes raw map
        inputs itself and is a training-time auxiliary, not a deploy output).

        ``ego_current_state`` (normalized) is required when the head consumes the
        current motion state; the ONNX decoder graph then takes it as a second input.

        Returns:
            prediction: [B, 1 + Pn, T, 4] best-mode ego + neighbors, denormalized.
            probability: [B, K] mode logits.
            turn_indicator_logit: [B, TURN_INDICATOR_OUTPUT_DIM]
        """
        ego_state = (
            ego_current_state[:, self._ego_state_slice]
            if (self._use_ego_state and ego_current_state is not None)
            else None
        )
        decode_out = self._decode(encoding, ego_state)
        # Deploy path ignores the log-scale (used only in the training NLL loss).
        trajectory, probability, neighbor_prediction = decode_out[:3]
        best_trajectory = self._best_mode_trajectory(trajectory, probability)
        turn_indicator_logit = self._compute_turn_indicator(
            self._subsampled_ego_xy(best_trajectory), self._pool_encoding(encoding)
        )
        prediction = torch.cat([best_trajectory[:, None], neighbor_prediction], dim=1)
        prediction = self._state_normalizer.inverse(prediction)
        return prediction, probability, turn_indicator_logit

    def forward(self, encoding, inputs):
        """
        Args:
            encoding: [B, token_num, D] encoder output (token 0 = ego,
                tokens 1 .. 1+predicted_neighbor_num = neighbors).
            inputs: Dict. Training additionally requires "gt_trajectories"
                [B, P, 1 + future_len, 4] (normalized, current state prepended),
                injected by ``compute_plantf_training_loss``.

        Returns:
            Dict:
                [both] "trajectory": [B, K, T, 4] ego mode trajectories (normalized)
                [both] "probability": [B, K] mode logits
                [both] "neighbor_prediction": [B, Pn, T, 4] (normalized)
                [both] "turn_indicator_logit": [B, TURN_INDICATOR_OUTPUT_DIM]
                [inference-only] "prediction": [B, 1 + Pn, T, 4] best-mode ego +
                    neighbors, denormalized — same contract as ``Decoder``.
        """
        B = encoding.shape[0]
        Pn = self._predicted_neighbor_num

        decode_out = self._decode(encoding, self._ego_state_feat(inputs))
        if self._predict_scale:
            trajectory, probability, neighbor_prediction, log_scale = decode_out
        else:
            trajectory, probability, neighbor_prediction = decode_out
            log_scale = None
        encoding_pooled = self._pool_encoding(encoding)

        outputs = {
            "trajectory": trajectory,
            "probability": probability,
            "neighbor_prediction": neighbor_prediction,
        }
        if log_scale is not None:
            outputs["scale"] = log_scale  # [B, K, T, 4] per-point log-scale for Laplace NLL

        if self.training:
            gt_trajectories = inputs["gt_trajectories"].reshape(B, 1 + Pn, 1 + self._future_len, 4)
            ego_trajectory = gt_trajectories[:, 0, 1::10, :2].reshape(
                B, 2 * (self._future_len // 10)
            )
            outputs["turn_indicator_logit"] = self._compute_turn_indicator(
                ego_trajectory, encoding_pooled
            )
            return outputs

        best_trajectory = self._best_mode_trajectory(trajectory, probability, inputs)

        outputs["turn_indicator_logit"] = self._compute_turn_indicator(
            self._subsampled_ego_xy(best_trajectory), encoding_pooled
        )

        prediction = torch.cat([best_trajectory[:, None], neighbor_prediction], dim=1)
        outputs["prediction"] = self._state_normalizer.inverse(prediction)
        return outputs


def compute_plantf_training_loss(
    model: nn.Module,
    inputs: dict[str, torch.Tensor],
    futures: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    args: Namespace,
):
    """PlanTF-head counterpart of ``decoder.compute_training_loss``.

    Returns the same loss keys plus ``mode_cls_loss`` so ``train_epoch`` can
    combine them with the shared coefficients.
    """
    norm = args.state_normalizer

    ego_future, neighbors_future, neighbor_future_mask = futures
    neighbors_future_valid = ~neighbor_future_mask  # [B, Pn, T]

    B, Pn, T, _ = neighbors_future.shape
    ego_current, neighbors_current = (
        inputs["ego_current_state"][:, :4],
        inputs["neighbor_agents_past"][:, :Pn, -1, :4],
    )
    neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
    neighbor_mask = torch.concat(
        (neighbor_current_mask.unsqueeze(-1), neighbor_future_mask), dim=-1
    )

    gt_future = torch.cat(
        [ego_future[:, None, :, :], neighbors_future[..., :]], dim=1
    )  # [B, P, T, 4]
    current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1)  # [B, P, 4]

    all_gt = torch.cat([current_states[:, :, None, :], norm(gt_future)], dim=2)
    all_gt[:, 1:][neighbor_mask] = 0.0

    merged_inputs = {**inputs, "gt_trajectories": all_gt}
    _, decoder_output = model(merged_inputs)

    trajectory = decoder_output["trajectory"]  # [B, K, T, 4], normalized
    probability = decoder_output["probability"]  # [B, K]
    neighbor_prediction = decoder_output["neighbor_prediction"]  # [B, Pn, T, 4]

    gt_norm = all_gt[:, :, 1:, :]  # [B, P, T, 4]
    ego_gt = gt_norm[:, 0]  # [B, T, 4]

    # Winner-takes-all mode selection on xy ADE in meters (the normalized x/y
    # scales differ, so rescale by std; the mean shift cancels in the diff).
    with torch.no_grad():
        ego_std_xy = norm.std[0][..., :2].to(trajectory.device)  # [1, 2]
        xy_diff = (trajectory[..., :2] - ego_gt[:, None, :, :2]) * ego_std_xy
        ade = torch.norm(xy_diff, dim=-1).mean(-1)  # [B, K]
        best_mode = ade.argmin(dim=-1)  # [B,]
    best_trajectory = trajectory[torch.arange(B, device=trajectory.device), best_mode]

    prediction = torch.cat([best_trajectory[:, None], neighbor_prediction], dim=1)  # [B, P, T, 4]

    # Original planTF ego/neighbor loss (jchengai/planTF): smooth L1 on the
    # winner-takes-all best mode over all channels (x, y, cos, sin), uniform
    # across timesteps, plus a cross-entropy on the DETACHED best mode. The DP
    # diffusion-decoder tuning previously grafted on here — lat/lon/heading L2
    # decomposition, longitudinal velocity down-weighting, timestep weighting and
    # the endpoint term — is intentionally dropped: it destabilized this one-shot
    # regression head (团子 / 直進 / 反対方向 / 櫛状; see
    # docs/plantf_dead_mode_improvement.md). turn-indicator and penalty losses
    # below are kept for the DP/Autoware interface.
    loss = {}

    ego_pred = prediction[:, 0, : args.ego_prediction_horizon]  # [B, T', 4]
    ego_tgt = gt_norm[:, 0, : args.ego_prediction_horizon]
    # Ego regression loss (opt-in variants, docs/plantf_head_development_notes.md §9):
    #   - plantf_use_laplace_nll (#5): Laplace NLL |y-mu|*exp(-s)+s with the head's
    #     per-point log-scale s (planTF's probabilistic regression; calibrates the
    #     tail uncertainty and the mode confidence).
    #   - plantf_tail_weight (#6 variant): weight the per-timestep loss toward the
    #     tail (w_t = 1 + tail_weight * t/(T-1)).
    #   - both 0/off -> original uniform smooth-L1.
    tail_weight = getattr(args, "plantf_tail_weight", 0.0)
    tp = ego_pred.shape[1]
    time_w = (
        1.0 + tail_weight * torch.arange(tp, device=ego_pred.device) / max(tp - 1, 1)
        if tail_weight > 0
        else None
    )
    if getattr(args, "plantf_use_laplace_nll", False) and "scale" in decoder_output:
        log_scale_best = decoder_output["scale"][
            torch.arange(B, device=trajectory.device), best_mode
        ][:, : args.ego_prediction_horizon]  # [B, T', 4]
        # Clamp the log-scale: a near-perfect mode drives log_scale -> -inf and
        # exp(-log_scale) -> inf, blowing up the NLL/gradient (standard Laplace
        # NLL instability). [-6, 6] bounds exp(-s) to ~[0.0025, 400].
        log_scale_best = log_scale_best.clamp(-6.0, 6.0)
        nll_t = ((ego_pred - ego_tgt).abs() * torch.exp(-log_scale_best) + log_scale_best).mean(-1)
        loss["ego_planning_loss"] = (
            (nll_t * time_w).sum() / (time_w.sum() * B) if time_w is not None else nll_t.mean()
        )
    elif time_w is not None:
        per_t = F.smooth_l1_loss(ego_pred, ego_tgt, reduction="none").mean(dim=-1)  # [B, T']
        loss["ego_planning_loss"] = (per_t * time_w).sum() / (time_w.sum() * B)
    else:
        loss["ego_planning_loss"] = F.smooth_l1_loss(ego_pred, ego_tgt)

    neighbor_pred = prediction[:, 1:]  # [B, Pn, T, 4]
    neighbor_tgt = gt_norm[:, 1:]
    if bool(neighbors_future_valid.any()):
        loss["neighbor_prediction_loss"] = F.smooth_l1_loss(
            neighbor_pred[neighbors_future_valid], neighbor_tgt[neighbors_future_valid]
        )
    else:
        loss["neighbor_prediction_loss"] = torch.tensor(0.0, device=prediction.device)

    # With one mode, cross entropy is identically zero and its logits cannot
    # affect the trajectory.  Do not compute or log this PlantF-only no-op;
    # train_epoch already treats mode_cls_loss as optional.  Keep the original
    # winner-mode classification objective unchanged for multimodal runs.
    if probability.shape[-1] > 1:
        loss["mode_cls_loss"] = F.cross_entropy(probability, best_mode.detach())

    # Smoothness penalty: the planTF head regresses 80 absolute waypoints
    # independently per timestep, which produces "comb" jitter (large second
    # difference) even when ADE/FDE look fine. Penalizing the xy second
    # difference of the best mode directly suppresses that jitter. Computed in
    # the normalized space (same scale as ego_planning_loss). Off by default
    # (coeff_smoothness_loss). See docs/plantf_original_comparison_and_roadmap.md.
    best_xy = best_trajectory[:, :, :2]
    second_diff = best_xy[:, 2:] - 2.0 * best_xy[:, 1:-1] + best_xy[:, :-2]
    sq = (second_diff**2).sum(dim=-1)  # [B, T-2]
    # Loss improvement (#6): weight the curvature penalty toward the tail, where
    # the divergence is worst. plantf_smoothness_tail_weight=0 -> uniform mean.
    sm_tail = getattr(args, "plantf_smoothness_tail_weight", 0.0)
    if sm_tail > 0:
        ns = sq.shape[1]
        ws = 1.0 + sm_tail * torch.arange(ns, device=sq.device) / max(ns - 1, 1)
        loss["smoothness_loss"] = (sq * ws).sum() / (ws.sum() * sq.shape[0])
    else:
        loss["smoothness_loss"] = sq.mean()

    # Compute ego edge points for penalty losses (best mode only)
    need_ego_edge = args.coeff_road_border_loss > 0 or args.coeff_neighbor_collision_loss > 0
    if need_ego_edge:
        ego_pred_world = best_trajectory * norm.std[0].to(trajectory.device) + norm.mean[0].to(
            trajectory.device
        )  # [B, T, 4]
        ego_edge_points = compute_ego_edge_points(
            ego_pred_world, inputs["ego_shape"], n_interp=args.road_border_n_interp
        )
        denorm_inputs = args.observation_normalizer.inverse(inputs)

    if args.coeff_road_border_loss > 0:
        rb_loss = compute_road_border_penalty(
            ego_edge_points,
            denorm_inputs["line_strings"],
            margin=args.road_border_margin,
        )  # [B, T]
        loss["road_border_loss"] = rb_loss.mean()
    else:
        loss["road_border_loss"] = torch.tensor(0.0, device=prediction.device)

    if args.coeff_neighbor_collision_loss > 0:
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
        loss["neighbor_collision_loss"] = torch.tensor(0.0, device=prediction.device)

    assert not torch.isnan(loss["ego_planning_loss"]), "loss cannot be nan"

    turn_indicator_logit = decoder_output["turn_indicator_logit"]
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
