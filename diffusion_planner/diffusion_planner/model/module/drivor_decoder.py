"""DrivoR predictor head for Diffusion-Planner.

This replaces the diffusion (DiT / DPM-Solver) decoder with the DrivoR head:

    learned proposal embeddings (+ ego token)
      -> per-stage trajectory MLP                    (initial proposals)
      -> ``ref_num`` refinement blocks over the scene tokens
      -> per-stage trajectory MLP                    (refined proposals)
      -> detached proposals re-embedded (``pos_embed``)
      -> ``scorer_ref_num`` scorer blocks + ego token
      -> six independent BCE PDM logit heads
      -> PDMS aggregate -> argmax -> ONE ego trajectory

The scene tokens are the Diffusion-Planner encoder's fused token stream, so the
whole vector/mixer encoder is reused verbatim; only the head changes.  No
neighbour prediction is produced.  With ``drivor_turn_indicator`` the diffusion
Decoder's ``TurnIndicatorNetwork`` rides along and emits the 5-class
``turn_indicator_logit`` (teacher-forced with the demonstration future in
training, fed the selected proposal at inference).
"""

from typing import Optional

import torch
import torch.nn as nn

from diffusion_planner.model.module.drivor_layers import MLP, TransformerDecoder
from diffusion_planner.model.module.drivor_scorer import (
    SCORE_WEIGHT_ORDER,
    Scorer,
    aggregate_pdm_score,
)
from diffusion_planner.model.module.turn_indicator import TurnIndicatorNetwork

POSE_STATE_SIZE = 4  # x, y, cos(yaw), sin(yaw) -- Diffusion-Planner's pose layout


class DrivoRDecoder(nn.Module):
    """Proposal-generate-then-score ego planner head."""

    def __init__(self, config):
        super().__init__()

        d_model = int(config.hidden_dim)
        d_ffn = int(config.drivor_tf_d_ffn)

        self.proposal_num = int(config.drivor_proposal_num)
        self.ref_num = int(config.drivor_ref_num)
        # The head's own trajectory length, NOT Diffusion-Planner's ``future_len``
        # (80 poses / 8 s): this head's horizon is navsim's 4 s, by default 40
        # poses at 0.1 s.  One linear layer's width, so the pose count is free --
        # see ``TrainConfig.drivor_num_poses`` for the measurement.
        self.poses_num = int(config.drivor_num_poses)
        self.state_size = POSE_STATE_SIZE

        # Learned proposal queries.  DrivoR uses one token per trajectory.
        self.init_feature = nn.Embedding(self.proposal_num, d_model)

        self.trajectory_decoder = TransformerDecoder(
            num_layers=self.ref_num,
            d_model=d_model,
            num_heads=int(config.drivor_refiner_num_heads),
            ls_values=float(config.drivor_refiner_ls_values),
            proj_drop=float(config.drivor_trajectory_proj_drop),
            drop_path=float(config.drivor_trajectory_drop_path),
            return_intermediate=True,
        )
        self.scorer_attention = TransformerDecoder(
            num_layers=int(config.drivor_scorer_ref_num),
            d_model=d_model,
            num_heads=int(config.drivor_refiner_num_heads),
            ls_values=float(config.drivor_refiner_ls_values),
            proj_drop=float(config.drivor_scorer_proj_drop),
            drop_path=float(config.drivor_scorer_drop_path),
            return_intermediate=False,
        )

        self.traj_head = nn.ModuleList(
            [MLP(d_model, d_ffn, self.poses_num * self.state_size) for _ in range(self.ref_num + 1)]
        )

        self.pos_embed = nn.Sequential(
            nn.Linear(self.poses_num * self.state_size, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model),
        )

        self.human_weight = float(config.drivor_human_teacher_weight or 0.0)
        self.scorer = Scorer(
            d_model=d_model,
            d_ffn=d_ffn,
            human_teacher_weight=self.human_weight,
            logit_bound=float(config.drivor_logit_bound or 0.0),
        )

        # Model-selection profile (noc, dac, ddc, ttc, ep, comfort).
        weights = [float(getattr(config, f"drivor_weight_{name}")) for name in _WEIGHT_ATTRS]
        self.register_buffer(
            "score_weights", torch.tensor(weights, dtype=torch.float32), persistent=False
        )

        # Ego-only slice of the shared state normalizer: the trajectory heads
        # regress normalized states (well-conditioned at init) and the exported
        # proposals are metric, which is what both the WTA loss and the PDM
        # oracle consume.
        normalizer = getattr(config, "state_normalizer", None)
        if normalizer is None:
            mean = torch.zeros(1, POSE_STATE_SIZE)
            std = torch.ones(1, POSE_STATE_SIZE)
        else:
            mean = torch.as_tensor(normalizer.mean, dtype=torch.float32).reshape(-1, 1, 4)[0]
            std = torch.as_tensor(normalizer.std, dtype=torch.float32).reshape(-1, 1, 4)[0]
        self.register_buffer("state_mean", mean.reshape(1, 1, 1, 4), persistent=False)
        self.register_buffer("state_std", std.reshape(1, 1, 1, 4), persistent=False)

        self.apply(_basic_init)
        nn.init.normal_(self.init_feature.weight, mean=0.0, std=0.02)

        # Optional turn-indicator head: the same parameter- and feature-independent
        # network the diffusion Decoder uses (it reads the ego trajectory plus the
        # raw turn-history / lane / route inputs, never these proposal tokens), at
        # the diffusion path's half-size hyperparameters, on this head's horizon.
        # Built AFTER ``_basic_init`` so its own initialisation survives.
        # ``getattr(..., False)``: an args.json written before this field existed
        # describes a checkpoint without the head.
        self.turn_indicator_predictor = None
        if bool(getattr(config, "drivor_turn_indicator", False)):
            self.turn_indicator_predictor = TurnIndicatorNetwork(
                hidden_dim=d_model // 2,
                num_heads=max(1, int(getattr(config, "num_heads", 8)) // 2),
                mixer_depth=max(1, int(getattr(config, "encoder_mixer_depth", 6)) // 2),
                fusion_depth=max(1, int(getattr(config, "encoder_fusion_depth", 6)) // 2),
                drop_path_rate=float(getattr(config, "encoder_drop_path_rate", 0.0)),
                trajectory_len=self.poses_num,
            )

    def forward(
        self,
        encoding: torch.Tensor,
        encoding_mask: Optional[torch.Tensor] = None,
        inputs: Optional[dict] = None,
        target_trajectory: Optional[torch.Tensor] = None,
    ) -> dict:
        """Args:
        encoding: [B, S, D] fused scene tokens from the DP encoder.
        encoding_mask: [B, S] bool, True where the token is padding.
        inputs: the (normalized) model input dict; required when the
            turn-indicator head is built (it reads ``lanes``, ``route_lanes``,
            their speed limits and ``turn_indicators``).
        target_trajectory: [B, poses_num, 4] METRIC demonstration future.  In
            training mode the turn-indicator head is teacher-forced with it;
            otherwise (or when None) it conditions on the selected proposal.
        """

        batch_size = encoding.shape[0]
        # Token 0 of the DP encoder is the fused ego token; it plays the role of
        # DrivoR's ``hist_encoding(ego_status)``.
        ego_token = encoding[:, :1]

        traj_tokens = ego_token + self.init_feature.weight[None]

        proposal_list = [self._decode(self.traj_head[0], traj_tokens)]
        token_list = self.trajectory_decoder(
            traj_tokens, encoding, x_cross_padding_mask=encoding_mask
        )
        for index in range(self.ref_num):
            proposal_list.append(self._decode(self.traj_head[index + 1], token_list[index]))
        proposals = proposal_list[-1]

        # The scorer sees the proposals only through a detached embedding, so it
        # never back-propagates into the trajectory generator.
        num_proposals = proposals.shape[1]
        embedded_traj = self.pos_embed(
            proposals.reshape(batch_size, num_proposals, -1).detach().to(encoding.dtype)
        )
        scorer_tokens = self.scorer_attention(
            embedded_traj, encoding, x_cross_padding_mask=encoding_mask
        )
        scorer_tokens = scorer_tokens + ego_token

        pred_logit = self.scorer(scorer_tokens)
        score_weights = self.score_weights.to(proposals.dtype).expand(batch_size, -1)
        pdm_score, score_components = aggregate_pdm_score(
            pred_logit, score_weights, human_weight=self.human_weight
        )

        chosen = torch.argmax(pdm_score, dim=1)
        trajectory = proposals[torch.arange(batch_size, device=proposals.device), chosen]

        out = {
            # Ego-only output, kept in the repository's [B, P, T, 4] layout so
            # ONNX export / visualisation / closed-loop consume it unchanged.
            "prediction": trajectory[:, None],
            "trajectory": trajectory,
            "proposals": proposals,
            "proposal_list": proposal_list,
            "pred_logit": pred_logit,
            "pdm_score": pdm_score,
            "score_components": score_components,
            "score_weights": score_weights,
            "chosen_index": chosen,
        }

        if self.turn_indicator_predictor is not None:
            if inputs is None:
                raise ValueError(
                    "DrivoRDecoder: the turn-indicator head needs the model inputs "
                    "(lanes, route_lanes, turn_indicators); pass inputs="
                )
            # Same convention as the diffusion Decoder: the GT future while
            # training, the model's own trajectory at inference.
            ego = (
                target_trajectory
                if (self.training and target_trajectory is not None)
                else trajectory
            )
            out["turn_indicator_logit"] = self._compute_turn_indicator(ego, inputs)
        return out

    def _compute_turn_indicator(self, ego_trajectory: torch.Tensor, inputs: dict) -> torch.Tensor:
        """Turn-indicator logits ``[B, 5]`` from a METRIC ego trajectory.

        The network was designed on the diffusion path, where it sees the
        state-normalized trajectory, so the metric input is normalized here with
        the same ego slice of the state normalizer the proposal heads use.
        Method name shared with ``Decoder`` so ``TurnIndicatorONNXWrapper`` can
        drive either head.
        """
        normalized = (ego_trajectory.float() - self.state_mean[0, 0]) / self.state_std[0, 0]
        return self.turn_indicator_predictor(normalized, inputs)

    def _decode(self, head: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
        normalized = head(tokens).reshape(tokens.shape[0], -1, self.poses_num, self.state_size)
        return normalized.float() * self.state_std + self.state_mean


_WEIGHT_ATTRS = SCORE_WEIGHT_ORDER


def _basic_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)


__all__ = ["DrivoRDecoder"]
