# HDP and Diffusion Planner: SFT Model Design Comparison

**Last updated:** 2026-07-13

**Scope:** Base training and SFT only. This document covers model architecture, conditioning inputs, trajectory representation, supervised objectives, and training trade-offs.

## 1. Executive Summary

The current `feature/hyper-diffusion-planner` model is not simply Tier IV Diffusion Planner with an added velocity loss. It makes two fundamental changes required by the HDP design:

1. The Tier IV DP decoder represents each agent with one token and flattens the complete 81-frame trajectory into that token's channels. The current HDP decoder instead represents each of the 80 ego future steps with its own token, so decoder self-attention operates over future time.
2. The output is strictly ego-only. Neighbor histories remain important conditioning inputs, but the model no longer predicts neighbor futures or spends action-decoder capacity on a neighbor future prediction objective.

The current model does not copy the smaller public NuPlan HDP encoder. It retains the richer Tier IV scene encoder and adds an official-style global route condition to AdaLN. The model can therefore be summarized as:

> **Tier IV high-capacity scene encoder + official-style HDP temporal ego decoder + HDP velocity/hybrid SFT objective + an isolated turn-indicator head required by the Tier IV task.**

The trajectory-policy objective for the current Base/SFT model is:

\[
\mathcal{L}_{policy}
= \mathcal{L}_{velocity\text{-}x_0}
+ 0.01\,\mathcal{L}_{waypoint}^{W=10}.
\]

The road-border and neighbor-collision penalties remain available for ablation, but both coefficients are zero by default in current Base/SFT training. Production Base/SFT uses a policy-only stage: the turn-indicator head is frozen and skipped, then trained in a separate head-only stage. The explicit `joint` mode is retained only for controlled ablations; its detached head loss still cannot backpropagate into the scene encoder or diffusion decoder.

## 2. Overview of the Four Models

| Dimension | Tier IV DP on `main` | Our Initial HDP Port | Authors' Official HDP | Current HDP Branch |
|---|---|---|---|---|
| Primary design | Joint diffusion planner | HDP representation/loss added to the DP joint architecture | Native ego-only HDP | High-capacity HDP for Tier IV data |
| Action token | One token per predicted agent | One token per predicted agent | One token per future step | One token per future step |
| Temporal representation | 81 frames flattened into `81x4` channels | 81 frames flattened into `81x4` channels | 80 temporal tokens | 80 temporal tokens |
| Decoder self-attention | Across agents | Across agents; sequence length is one in ego-only mode | Across future time | Across future time |
| Decoder size | dim 256 / 3 blocks / 8 heads | dim 256 / 3 blocks / 8 heads | NuPlan: dim 192 / 3 blocks / 6 heads | dim 256 / 6 blocks / 8 heads |
| Predicted entities | Ego and neighbor futures | Ego and neighbor futures; later configurable as ego-only | Ego future only | Strictly ego future only |
| Neighbor information | History input and future output/supervision | History input and future output/supervision | History is conditioning only | History is conditioning only |
| Action latent | Absolute waypoints | Ego per-step deltas; neighbor futures still supported | Per-step delta/velocity | Per-step delta/velocity |
| Diffusion target | `x_start` waypoints by default; flow matching also supported | HDP path uses `x_start` deltas | `x_start` deltas by default | Enforced `x_start` deltas |
| Policy SFT loss | Ego waypoint + 0.1 neighbor + turn + road-border by default | Velocity + 0.01 waypoint + neighbor + turn + road-border by default | Velocity + 0.01 waypoint | Velocity + 0.01 waypoint; the turn head is trained later in a separate stage |
| Route conditioning | Route tokens in the scene encoder/cross-attention | Same as Tier IV DP | NuPlan: global route embedding in AdaLN | Route-token cross-attention plus global route AdaLN condition |
| Additional inputs | Ego history, goal pose, map, shape, turn history, and others | Same as Tier IV DP | More compact NuPlan input set | Full Tier IV scene inputs; turn history is label-only and not a policy input |
| Turn-indicator output | Yes; teacher-forced trajectory head | Yes; later gained generated/expert branches | No | Yes; isolated head with no policy gradient |
| Road border | Map input present; loss enabled by default | Map input present; loss enabled by default | No auxiliary road-border loss | Map input retained; loss disabled by default |

## 3. Tier IV Diffusion Planner on `main`

### 3.1 Scene Encoder

Tier IV DP was designed for Tier IV data and downstream interfaces. Its scene inputs are richer than those of the released NuPlan HDP model:

- 21 frames of ego history;
- neighbor-agent histories;
- static objects;
- lanes, route lanes, polygons, and line strings/road borders;
- goal pose and ego shape;
- historical turn-indicator category codes, provided as one scalar per timestep.

The feature-specific MLP-Mixer or MLP encoders produce scene tokens. A fusion transformer combines these tokens, and the resulting scene representation is consumed by decoder cross-attention.

### 3.2 Joint Action Decoder

The original decoder action tensor has shape `[B, P, 81, 4]`, where `P=1+predicted_neighbor_num`. The 81 frames consist of the current state and 80 future states. Before entering DiT, the tensor is reshaped to `[B, P, 324]`:

- one action token represents one agent;
- all 81 temporal states are flattened into the token's channel dimension;
- self-attention models interactions among the ego and predicted-neighbor action tokens, not interactions among future timesteps.

This design has a clear purpose for joint prediction. However, when `P=1`, the self-attention sequence length is also one. It therefore cannot use self-attention to model dependencies among the 80 future steps.

The original architecture also includes the current state as an action prefix and supports delay/prefix constraints that keep the first part of a diffusion trajectory fixed.

### 3.3 SFT Objective

Tier IV DP uses absolute-waypoint `x_start` prediction by default:

- ego planning loss with weight 1.0;
- neighbor future prediction loss with weight 0.1;
- turn-indicator classification loss;
- road-border penalty with default weight 1.0;
- neighbor-collision penalty with default weight 0.

During training, the turn-indicator head reads the ground-truth ego trajectory and pooled scene encoding. It is therefore teacher-forced, and its gradient updates the scene encoder and turn head, although it does not directly update the action decoder through the ground-truth trajectory.

## 4. Our Initial HDP Port Before the Temporal Decoder

This section refers to the implementation lineage beginning at `330f400` and ending before `253fb17` introduced the temporal decoder. It implemented the HDP trajectory representation and objective without immediately changing Tier IV DP action tokenization.

### 4.1 HDP Components That Were Already Correct

- Ego absolute waypoints were converted into differences between adjacent frames. The value called `velocity` in the code is more precisely a **per-step displacement** at a fixed sampling interval.
- Heading remained `(cos, sin)` at every future step and was not cumulatively integrated.
- Ego-delta normalization was separated from absolute-waypoint normalization.
- The HDP path required `x_start` prediction and `x_start` supervision.
- The official hybrid objective was added: delta L2 plus `0.01 x` integrated-waypoint L2.
- Detached integration used `W=10`. Its forward value remained the full integral, while each waypoint's backward path was limited to the most recent ten steps.

### 4.2 Inherited DP Components and Their Limitations

- The decoder still represented one agent with one token and flattened all 81 frames.
- Joint neighbor future prediction remained available with the inherited 0.1 neighbor loss.
- The current-state prefix, delay mask, and joint action mask remained in the architecture.
- Road-border loss remained enabled by default with weight 1.0.
- The turn-indicator head and other Tier IV auxiliary paths remained present.

The initial port was therefore a **DP joint decoder with an HDP ego representation and loss**, not the native temporal decoder described by HDP. Later setting `predicted_neighbor_num=0` did not resolve the architectural issue: it only left one large ego action token.

### 4.3 Effect of Neighbor Future Prediction

Neighbor future prediction was inherited from Tier IV DP. It is not a core component of the paper or the released HDP action head.

Its possible benefit is additional dynamic supervision for the shared encoder. Its costs are that the action decoder must model the uncertain, multimodal futures of multiple actors, consumes additional capacity and memory, and introduces gradients from neighbor ground truth into the model used to produce the ego action. For this branch's objective of maximizing on-vehicle ego-planning performance, there is no current evidence that this auxiliary task provides a net gain. The current model therefore removes neighbor future outputs while retaining the full neighbor-history condition.

## 5. Authors' Official HDP

### 5.1 Paper Design

The paper represents the future ego action as a sequence of length `L`. The noised action is split into `L` temporal tokens and augmented with temporal position embeddings:

- self-attention models relationships among future timesteps;
- cross-attention reads scene conditions;
- diffusion time modulates the layers through AdaLN-Zero;
- the output is an ego trajectory and contains no neighbor future action tokens.

The paper's experiments select `x_start` prediction with `x_start` supervision. Compared with noise or diffusion-velocity prediction, direct data prediction converges faster and generates smoother, more stable trajectories.

### 5.2 Velocity Representation and Hybrid Loss

The paper replaces absolute waypoints with a physical velocity/delta representation and recovers waypoints by integration:

\[
\mathcal{L}_{hybrid}
= \lVert v_\theta-v_0\rVert_2^2
+\omega\lVert Mv_\theta\Delta t-x_0\rVert_2^2.
\]

The released NuPlan code directly applies `torch.diff` to adjacent waypoints without explicitly dividing by `dt`. Its implemented `velocity` is therefore more precisely a per-step displacement, with the fixed interval absorbed into the scale. Our implementation follows the released code.

The paper proves that the hybrid loss is a positive-definite quadratic Bregman divergence and preserves the correct conditional mean. It contrasts this with prediction-dependent, non-Bregman auxiliary planning losses, which produce biased score estimators. The paper discusses auxiliary planning losses generally and mentions collision loss in a comment. Applying the same argument to a prediction-dependent road-border penalty is our theoretical inference, not a verbatim paper claim about road-border loss.

### 5.3 Released NuPlan Implementation

The main public configuration uses `future_len=80`, hidden dimension 192, three decoder blocks, six heads, and `omega=0.01`:

- 80 future temporal tokens;
- current ego `vx, vy` added to the action-token embedding;
- global route geometry compressed by a Mixer and added to the diffusion-timestep AdaLN condition;
- scene cross-attention over neighbors, static objects, and lanes;
- an SFT objective containing only the ego velocity/delta loss and waypoint hybrid loss;
- no turn-indicator head, road-border loss, or neighbor future loss.

The public CLI still contains `predicted_neighbor_num=10` and labels neighbor prediction as deprecated in HDP. This value remains in a dataset argument, but the actual decoder, objective, and output are ego-only. It is not evidence that official HDP is a joint predictor.

The released NAVSIM model uses a different vision/language frontend and adds ego proprioception to AdaLN. NuPlan and NAVSIM use different global conditions, but they share the important structural properties of temporal action tokens and an AdaLN condition that contains more than diffusion time alone.

## 6. Current HDP Branch

### 6.1 Strict Ego-Only Temporal Decoder

The current action latent is fixed to `[B, 1, 80, 4]`:

- each of the 80 future steps is an action token;
- every token outputs `(dx, dy, cos, sin)`;
- self-attention explicitly models temporal consistency in speed, curvature, and behavior across the full future horizon;
- the current state is no longer included as an 81st action frame;
- delay/prefix action constraints are no longer supported;
- any nonzero `predicted_neighbor_num` is rejected by both configuration validation and model construction.

Ego-only output does not mean that the model ignores neighbors. Neighbor histories still affect the ego future through the high-capacity scene encoder and cross-attention in every decoder block. Only neighbor future outputs and supervision have been removed.

### 6.2 High-Capacity Tier IV Scene Condition

The current model does not revert to the compact public NuPlan encoder. It retains Tier IV ego history, neighbor history, static objects, lanes, routes, polygons, line strings, goal pose, and ego shape. The fusion encoder processes these fine-grained tokens before decoder cross-attention. Turn history was removed from the policy in the 2026-07-21 SFT design because deployment would otherwise feed the model's own command back into its trajectory condition.

Route information follows two complementary paths:

1. Full route tokens preserve local geometry for cross-attention, helping the model determine where to turn and which curvature to follow.
2. A lightweight `GlobalRouteEncoder` compresses the ordered route geometry and adds it to the timestep condition used to modulate every decoder block through AdaLN.

The second path aligns with the released NuPlan HDP conditioner. The first path is additional capacity retained for the Tier IV task. The global route embedding is computed once per scene and reused rather than recomputed at every DPM step.

### 6.3 Current Policy SFT Objective

Training enforces velocity representation, `x_start` prediction, and `x_start` supervision:

- normalized delta `x_start` L2;
- waypoint L2 after denormalized position integration;
- hybrid coefficient `omega=0.01`;
- detached integration window `W=10`;
- supervision over all 80 future steps.

bf16 autocast is limited to the model forward pass. Noising, the SDE schedule, inverse normalization, and all loss calculations remain in fp32 for diffusion-training numerical stability.

### 6.4 Turn-Indicator Head

Tier IV requires a turn-indicator output that is absent from official HDP. The current implementation retains it as an isolated auxiliary head:

- historical turn indicators are labels only and never enter the scene encoder;
- the head is trained from both the detached generated trajectory and the expert trajectory;
- the target has exactly three dense states: disable, left, and right;
- no nonexistent raw class 0, transition class, onset multiplier, or stale class weighting is used;
- generated trajectory, expert trajectory, scene encoding, route condition, and current proprioception are detached inside the head, so classification loss updates only the head and cannot modify the diffusion policy;
- output probabilities are stabilized by a deployment-side hysteresis state machine, rather than teaching the neural head an exact human switch frame.

See [HDP turn-indicator SFT design](hdp_turn_indicator_sft.md) for the full data audit and checkpoint migration.

### 6.5 Road-Border and Collision Losses

Road borders and line strings remain scene inputs, so the model can use HD-map boundaries as planning conditions. What is disabled is the auxiliary loss that computes a penalty from the predicted trajectory and backpropagates it. The map input itself has not been removed.

Current Base/SFT defaults are:

- `coeff_road_border_loss=0.0`;
- `coeff_neighbor_collision_loss=0.0`.
- AdamW uses `weight_decay=0.01` for ordinary weights and `0.0` for normalization scales,
  biases, embedding tables, and the two explicit temporal/route position tables
  (`adamw_no_decay=True`). This optimizer policy is strict-resume compatible.

The trajectory-policy objective therefore remains the paper-supported quadratic hybrid loss. Validation may still report road-border violations, collision diagnostics, or EPDMS-style metrics. A W&B key such as `valid_loss/ego_road_border_loss` indicates that a diagnostic was computed; it does not mean that the value contributed to the training objective.

## 7. Design Evolution and Final Trade-Offs

The important evolution from Tier IV DP to the current HDP model is not the addition of one loss. It consists of the following changes:

1. **Waypoints to deltas:** improves local temporal smoothness and on-vehicle comfort.
2. **One full-trajectory token per agent to 80 temporal tokens:** allows DiT self-attention to model future-time relationships directly.
3. **Joint output to ego-only output:** focuses decoder capacity on the executed ego policy while retaining neighbor histories as conditions.
4. **Cross-attention-only route to a local and global dual route path:** preserves both local geometry and global maneuver intent.
5. **Prediction-dependent auxiliary penalties disabled by default:** avoids changing the diffusion conditional mean; map and safety quality remain represented through inputs, data quality, and independent validation diagnostics.
6. **Turn head decoupled from the policy:** satisfies the Tier IV output requirement without allowing classification loss to reshape the trajectory policy.

The current model is therefore not a line-by-line reproduction of the public implementation. It preserves the core HDP SFT principles while using richer Tier IV inputs and a larger model. For this project's performance-first objective, it is the most appropriate primary training architecture among the four variants.

## 8. Current Base/SFT Data Policy

The current training data organization follows these rules:

- Base training uses the base list that excludes unprotected-right-turn after-entry samples.
- Separate unprotected-right-turn extra lists from all three vehicle data sources are included at 10x frequency.
- Traffic-light features are unchanged; no traffic-light masking augmentation is applied.
- SFT uses the same target right-turn data composition and initializes from the latest checkpoint of the temporal ego-only Base run.
- Ego deltas use dedicated normalization statistics; an HDP checkpoint must not be interpreted with absolute-waypoint normalization.

The shared base list remains immutable. Filtering and frequency changes are implemented in independent manifests or extra-list handling so that they do not alter data consumed by other teams.

## 9. Local Sources of Ground Truth

This comparison was verified against the following local sources and code snapshots:

- Paper LaTeX: `reference/hyper_diffusion_planner_paper/src/neurips_2026.tex`
- Paper hybrid-loss pseudocode: `reference/hyper_diffusion_planner_paper/src/code.tex`
- Official NuPlan implementation: `reference/external/Hyper-Diffusion-Planner/HDP-nuplan`
- Official NAVSIM implementation: `reference/external/Hyper-Diffusion-Planner/HDP-navsim`
- Tier IV DP `main`: `dba2de07d435e11ae316612aff6d80089c98ea0f`
- Initial HDP velocity port: `330f4004ecf9bef8ca5bfb5c4ba4e9f338dbb2b1`
- Temporal ego decoder introduction: `253fb17c396f1e3782862bb7834791421c99e6a7`
- Current HDP branch: `feature/hyper-diffusion-planner`
