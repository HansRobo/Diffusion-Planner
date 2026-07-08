# Joint Temporal One-Decoder 100-Point Architecture Review - 2026-06-27

Objective: design the long-term, most principled architecture for integrating DFP-style temporal modeling into Diffusion-Planner without two trajectory decoders, without ego-output gating, and without ad-hoc post-hoc trajectory blending.

Final target name:

```text
Joint Temporal Agent DFP Decoder
```

Core target formula:

```text
all_agent_future = JointTemporalDFPDecoder(
    all_agent_history,
    all_agent_current,
    noisy_all_agent_future_chunks,
    scene_encoding
)

ego_future = all_agent_future[:, ego]
neighbor_future = all_agent_future[:, neighbors]
```

This means there is one trajectory decoder stack. It jointly predicts ego and neighbor future. Ego-neighbor interaction happens inside the same temporal decoder, not through a second decoder, not through a gate, and not through a staged neighbor-prediction handoff.

## 100 design-review records

### Thought 001/100: What exactly is wrong with hard ego replacement?
Hard replacement says `ego_final = ego_dfp` while neighbor future still comes from the old DiT decoder. This is simple and empirically useful, but structurally it creates two trajectory generation mechanisms. Long-term architecture should not have one temporal model for ego and another for other agents, because ego planning and neighbor motion are coupled.

### Thought 002/100: What exactly is wrong with gated ego blending?
Gating says `ego_final = w * ego_dfp + (1-w) * ego_original`. This can diagnose whether the original ego head has residual value, but it is not a principled generative model. Two plausible trajectories can average into an implausible trajectory. Therefore gate is not the final architecture.

### Thought 003/100: What exactly is wrong with staged interaction DFP?
Staged interaction says old decoder predicts neighbor future, then DFP ego consumes that prediction. This is more principled than gate because ego sees neighbor future, but still has two trajectory decoders and an artificial ordering. Neighbor future becomes an intermediate product rather than a jointly optimized latent interaction.

### Thought 004/100: What is the cleanest long-term abstraction?
The clean abstraction is one decoder that models the future of all dynamic agents jointly. Ego and neighbors are not two separate prediction systems; they are different agents in one temporal interaction system. This matches the physical problem better.

### Thought 005/100: What does “one decoder” allow and disallow?
It allows one shared trajectory decoder stack with task-specific final projections if needed. It disallows a separate original DiT trajectory decoder plus a separate DFP ego trajectory decoder. A turn-indicator classifier can remain a small task head, because it is not a separate trajectory decoder.

### Thought 006/100: What must be preserved from original Diffusion-Planner?
The scene encoder, map/lane/route representation, neighbor masks, turn indicator supervision, road-border metrics, neighbor collision metrics, and validation behavior should be preserved unless they conflict with the one-decoder principle. The old decoder stack should not remain as a parallel trajectory generator.

### Thought 007/100: What must be preserved from DFP?
The essential DFP idea is temporal chunk modeling with history-current-future structure and diffusion forcing. The model should learn consistency across past, current, and future chunks, not just denoise a flat future trajectory from the current frame.

### Thought 008/100: Should DFP be ego-only or all-agent?
For the final architecture, DFP should be all-agent. Ego-only DFP treats ego as temporally special and leaves neighbors in a legacy predictor. Since ego behavior depends on neighbor behavior, all-agent temporal chunks are more coherent.

### Thought 009/100: What should the decoder tokens represent?
Tokens should represent `(agent, temporal_chunk)` pairs. For example, one ego agent plus `N` neighbor agents, each with history chunk, current chunk, and future chunks. This makes interaction and temporal reasoning explicit.

### Thought 010/100: What chunk layout is most natural for our 10 Hz data?
With `future_len=80` and `chunk_len=20`, future is four 2-second chunks at 10 Hz. A practical layout is one history chunk of 20 frames, one current chunk, and four future chunks: six chunks per agent. This matches existing DFP configuration and avoids over-fragmenting.

### Thought 011/100: Should history chunk be one chunk or many smaller chunks?
One 20-frame history chunk is a stable first implementation because it matches current DFP. Smaller history chunks could model finer temporal changes but increase token count and training complexity. The first one-decoder architecture should not expand chunk count before proving the unified decoder works.

### Thought 012/100: How should neighbor history be represented?
Use `neighbor_agents_past` to build neighbor history chunks in the same `[x, y, cos, sin]` format as ego. Invalid neighbors remain masked. This gives all agents temporal context, not just ego.

### Thought 013/100: Is one current chunk necessary?
Yes. The current chunk anchors the diffusion process. It gives the decoder an exact observed state per agent and prevents future chunks from drifting away from the current physical state.

### Thought 014/100: Should current chunk be noised?
For initial implementation, current chunk should stay clean or near-clean. It is a condition, not a prediction target. Noising current aggressively can degrade state anchoring and hurt short-horizon planning.

### Thought 015/100: Should history chunk be noised?
Yes, lightly or according to DFP guidance. Noised history forces temporal reconstruction and prevents the decoder from relying only on current state. However history loss should be lower weight than future planning loss.

### Thought 016/100: Should future chunks be independently noised?
Yes. Diffusion forcing requires per-chunk or per-token noising. Future chunks should have their own noise times so the decoder learns to condition across partially clean and partially noisy temporal segments.

### Thought 017/100: Should all agents share the same noise time?
Not necessarily. Agent-chunk-specific noise is more expressive, but too much randomness may destabilize training. A good first design samples per-sample/per-chunk noise and shares across agents within a chunk, or samples per-agent chunk noise with masks. The implementation can start with per-sample chunk noise for stability.

### Thought 018/100: Should ego and neighbors use identical diffusion targets?
Yes for the unified decoder. Both ego and neighbors are trajectories in the same coordinate system. Loss weights can differ, but the decoder should not use a separate modeling rule for ego and neighbors.

### Thought 019/100: How should invalid neighbor futures be handled?
Invalid future frames must be masked in reconstruction loss and attention. A neighbor that is absent should not contribute zero-trajectory supervision as if it were a real parked object.

### Thought 020/100: Should the decoder output current/history or only future?
The decoder may reconstruct all chunks internally, but validation and downstream outputs need future only. Training can include history/current reconstruction losses as auxiliary temporal regularizers, but final prediction should be future trajectories.

### Thought 021/100: What is the main loss?
The main loss remains ego planning loss plus neighbor prediction loss. The unified decoder should not make DFP reconstruction the only objective, because planning metrics depend on the final physical future trajectory.

### Thought 022/100: Should ego loss be stronger than neighbor loss?
Yes. Our goal is planning quality. Neighbor prediction is essential for interaction and collision reasoning, but ego trajectory is the primary output. Keep `alpha_planning_loss` and `alpha_neighbor_loss` style weighting.

### Thought 023/100: Should neighbor future loss still exist?
Yes. Without neighbor supervision, the model could invent neighbor futures that help ego loss but are physically wrong. Neighbor loss anchors interaction modeling to reality.

### Thought 024/100: Should road-border loss apply to ego only?
Yes. Road-border violation is a planning constraint for ego output. Neighbor agents may leave lane or follow non-route behavior, so applying ego road-border constraints to all agents would be wrong.

### Thought 025/100: Should collision loss use predicted neighbor or ground-truth neighbor?
For training penalty, using predicted ego against ground-truth neighbor is stable. Using predicted ego against predicted neighbor can become self-consistent but physically wrong. For validation PDMS, both proxy versions can be considered, but the initial metric should remain comparable to prior runs.

### Thought 026/100: Should turn indicator use predicted ego from the unified decoder?
Yes. Turn indicator should be derived from the final ego trajectory, not from a separate original ego head. This keeps behavior aligned with actual output.

### Thought 027/100: Should there be any original ego head after unification?
No. A separate original ego head recreates the gate problem. The one decoder itself produces ego. Any compatibility loss to an old head should be removed from the final architecture.

### Thought 028/100: Should there be any original neighbor head after unification?
No. The unified decoder itself produces neighbor future. Keeping old neighbor head would mean the model still has two trajectory decoders.

### Thought 029/100: Can the scene encoder remain unchanged?
Yes. The user asked for one decoder, not one whole model. The encoder is not the problem; the duplicated trajectory decoders are. Reusing the encoder preserves data compatibility and reduces unnecessary risk.

### Thought 030/100: Should the unified decoder reuse existing DiT blocks?
Preferably yes for the first implementation, but the block structure must be adapted to agent-chunk tokens. Reusing proven normalization, cross-attention, and timestep conditioning reduces risk.

### Thought 031/100: Is full self-attention over all agent-chunk tokens feasible?
Naively, tokens are roughly `(1+320)*6 = 1926`. Full attention over 1926 tokens per block is expensive. It may run, but it is not elegant and will likely be slower than needed.

### Thought 032/100: What attention factorization is more elegant?
Use factorized attention inside one decoder block: temporal attention per agent, interaction attention across agents, and cross-attention to scene. This remains one decoder stack while avoiding full quadratic cost over all agent-chunk pairs.

### Thought 033/100: Does factorized attention violate “one decoder”?
No. A decoder block can contain multiple attention sublayers. The requirement is one trajectory decoder stack and one joint output, not one attention operation.

### Thought 034/100: What is the minimal factorized block?
A good block has temporal self-attention over chunks for each agent, interaction self-attention over agents for each future chunk or agent summary, and scene cross-attention. This lets time and interaction both be modeled explicitly.

### Thought 035/100: Should interaction attention include all 320 neighbors?
Long-term yes as a fallback, but efficient implementation should support top-K relevant neighbors. Full 320-agent interaction can be expensive and may dilute attention with irrelevant agents.

### Thought 036/100: What top-K rule is principled?
Rank neighbors by current distance, route proximity, future/observed validity, and relative speed. For the first implementation, distance plus validity is acceptable. Later, route-aware relevance can improve it.

### Thought 037/100: Should top-K be hard-coded or configurable?
Configurable. Use `joint_agent_topk` with default perhaps 64 or 128. For exact comparability, a full setting `joint_agent_topk=320` should remain available.

### Thought 038/100: Could top-K hurt rare interactions?
Yes. A far but fast approaching vehicle might matter. Therefore the ranker should eventually include time-to-interaction or velocity. First implementation can use distance but should not lock the architecture to distance-only.

### Thought 039/100: How should static objects be handled?
Static objects already enter scene encoding. They do not need trajectory tokens. Dynamic agents are the only trajectory tokens in the unified decoder.

### Thought 040/100: Should map tokens enter decoder through cross-attention?
Yes. Map/lane/route information belongs in scene encoding and is consumed via cross-attention. It should not be duplicated as fake agent tokens.

### Thought 041/100: How to distinguish ego from neighbors?
Use agent-type embeddings: ego, valid neighbor, possibly padded neighbor. The same decoder parameters are shared, but embeddings tell the model which token is ego.

### Thought 042/100: How to distinguish history/current/future chunks?
Use chunk-type embeddings and chunk-position embeddings. History, current, and each future chunk have different semantic roles.

### Thought 043/100: How to preserve temporal order?
Use temporal chunk position embeddings and optionally frame-level position inside chunk through the preprojection. Without temporal position, chunk tokens are ambiguous.

### Thought 044/100: Is chunk token enough or do we need frame tokens?
Chunk token is enough for the first unified design because each chunk final layer outputs 20 frames. Frame tokens would be more expressive but much more expensive.

### Thought 045/100: How should each chunk be embedded?
Flatten `chunk_len * pose_dim` then project to hidden dimension. This matches current shared-stack DFP and is simple. Later, a small temporal convolution could improve within-chunk representation.

### Thought 046/100: What pose dimension should be used?
Use 4D `[x, y, cos, sin]` consistently. This avoids angle discontinuity and matches existing DP losses.

### Thought 047/100: Should velocity representation be used now?
Not in the first unified decoder. Existing stable setup uses absolute normalized waypoints. Velocity representation can be a later ablation, not mixed into the architectural migration.

### Thought 048/100: How should normalization work?
Reuse `StateNormalizer` and `ObservationNormalizer`. All future chunk targets should be normalized consistently with original DP. Mixing physical and normalized units inside the decoder would be error-prone.

### Thought 049/100: How should ego history be normalized?
Use the same state normalizer as future trajectories. History chunks should live in the same normalized coordinate frame as future chunks.

### Thought 050/100: How should neighbor history be normalized?
Use the same trajectory normalization for neighbor `[x,y,cos,sin]` if coordinates are ego-centric. Masks must zero invalid entries after normalization or before with consistent behavior.

### Thought 051/100: Should the unified decoder sample all agents at inference?
Yes, if it is truly one decoder. It should generate ego and neighbor future jointly. Validation still consumes final `[B, P, T, 4]` prediction.

### Thought 052/100: Does joint sampling increase compute?
Yes, but one decoder removes the old separate DiT + DFP combination. The cost moves into a single joint temporal stack. Efficient factorized attention is essential.

### Thought 053/100: How many diffusion sampling steps should be used?
Start with the same `dfp_sampler_steps=10` to compare fairly. Later tune sampler steps after architecture performance is known.

### Thought 054/100: Should neighbor future be sampled or one-shot decoded?
For a unified diffusion decoder, neighbor future should be sampled/denoised along with ego. One-shot neighbor prediction would reintroduce asymmetric modeling.

### Thought 055/100: How to condition on clean history/current during inference?
History and current chunks should be clamped or repeatedly reset to observed clean values during sampling. Future chunks are sampled; observed chunks are conditions.

### Thought 056/100: How to train with teacher-forced history/current?
Build clean chunks from data, noise future chunks, optionally noise history chunks, and keep current chunk clean. The decoder predicts clean chunks. Loss applies mainly to future plus auxiliary history/current.

### Thought 057/100: Should the decoder predict padded neighbors?
It may output values for padded neighbors, but loss and metrics must ignore them. Attention masks should prevent padded neighbors from influencing real agents.

### Thought 058/100: How to prevent padded agents from attending to real agents?
Mask padded tokens in key/value and optionally query. Query outputs for padded agents do not matter, but they should not consume compute if possible.

### Thought 059/100: Should invalid future frames within a valid neighbor be masked?
Yes. Some neighbors may disappear. Loss must be per-frame valid. Chunk loss should account for partial validity rather than dropping whole agent whenever one frame is missing.

### Thought 060/100: How to aggregate chunk loss with partial validity?
Compute per-frame loss and mask by valid future frame. For chunk-level DFP loss, average only valid frames. This avoids bias toward long-lived tracks.

### Thought 061/100: Should history reconstruction loss apply to invalid past frames?
No. Only valid past frames should contribute. For neighbors with sparse past, mask invalid frames.

### Thought 062/100: Should ego history always be valid?
Usually yes, but code should not assume blindly. If missing, fallback or mask is safer.

### Thought 063/100: What is the most important output contract?
The model must still output `prediction` with shape `[B, 1+predicted_neighbor_num, future_len, 4]`. This preserves validation, metrics, visualization, and downstream tooling.

### Thought 064/100: What about `model_output_orig` compatibility?
Remove it for the final one-decoder mode. Keeping it invites old-head regularization and confusion. The unified decoder is the only trajectory source.

### Thought 065/100: What about `dfp_x0` output?
The unified decoder can expose `joint_dfp_x0` or `dfp_x0` for loss, but it refers to all-agent chunks, not ego-only chunks. Naming should avoid implying ego-only DFP.

### Thought 066/100: What should the new mode be named?
Use `joint_temporal_agent`. It is short, explicit, and does not imply a shared-stack hack. Avoid names like `final`, `best`, or `ultimate` in code because they age badly.

### Thought 067/100: What should the run name be?
Use `dfp_joint_temporal_agent_step1_base_pdms_tier4main_node01_8gpu_tf32`. It states method, data regime, train type, node, GPU count, and precision.

### Thought 068/100: Should code keep old experimental modes?
For now yes, to preserve existing runs and compare. But the new launch should use only `joint_temporal_agent`. Documentation should state old modes are diagnostic or legacy.

### Thought 069/100: What is the biggest engineering risk?
Shape and mask bugs. Agent-chunk tensors have multiple axes and partial validity. The implementation must be explicit with names and dimensions.

### Thought 070/100: How to reduce shape bugs?
Use helper functions with clear names: `build_joint_temporal_chunks`, `flatten_agent_chunk_tokens`, `unflatten_future_chunks`, and `make_joint_agent_masks`. Avoid inline reshape chains in forward.

### Thought 071/100: What is the biggest modeling risk?
The neighbor prediction task may dominate or conflict with ego planning. Loss weighting must keep ego planning primary while preserving enough neighbor accuracy for interaction.

### Thought 072/100: How to detect neighbor domination?
Watch valid ego loss, PDMS, trajectory consistency, and neighbor loss together. If neighbor loss improves while ego planning worsens, weights are wrong or capacity is misallocated.

### Thought 073/100: Should ego token receive special capacity?
Yes through ego type embedding and ego loss weight, but not through a separate decoder. Specialization should be inside one shared decoder.

### Thought 074/100: Should future neighbor predictions influence ego inside same block?
Yes. Interaction attention should allow ego future chunks to attend neighbor future chunks at each denoising step. This is the key benefit of joint modeling.

### Thought 075/100: Should neighbor future chunks attend ego future chunks?
Yes. Neighbor prediction also depends on ego motion. Joint decoding should model mutual interaction, not one-way conditioning.

### Thought 076/100: Does mutual interaction risk circular dependency?
Diffusion decoding naturally handles coupled variables by denoising all variables jointly. This is more principled than staged dependency.

### Thought 077/100: Should scene cross-attention happen before or after interaction?
A good block order is temporal attention, interaction attention, then scene cross-attention, followed by MLP. This lets tokens first organize their temporal state, then interact, then ground in map/route context. Exact order is ablatable.

### Thought 078/100: Should timestep conditioning be per agent-chunk?
Yes. Different chunks can have different noise levels. The timestep embedder should accept flattened `(agent, chunk)` noise times.

### Thought 079/100: Should current/history condition tokens have timestep zero?
Current should use zero or near-zero noise time. History can use sampled low-to-medium noise. This mirrors DFP guidance and preserves current anchoring.

### Thought 080/100: How to initialize final layer?
Zero final projection is consistent with current DiT/DFP practice and stabilizes diffusion training. Keep zero initialization for final prediction layer.

### Thought 081/100: How to initialize new embeddings?
Small normal initialization, e.g. std 0.02. Avoid large embeddings that distort existing normalized trajectory scale.

### Thought 082/100: Should the unified decoder be trained from scratch?
Yes for the cleanest base train. Partial loading old decoder weights is possible but conceptually muddies the baseline. User explicitly prioritizes long-term architecture and base training.

### Thought 083/100: Does from-scratch training waste previous pretrained models?
It costs time, but avoids structural mismatch. Since the decoder changes fundamentally, old decoder weights may bias the model toward the previous flat all-agent denoising behavior.

### Thought 084/100: Can encoder weights be pretrained later?
Possibly, but for the cleanest baseline use the same base training policy as requested. Later experiments can test encoder-only initialization separately.

### Thought 085/100: What data regime should be used?
Use the filtered full-sequence step1 list derived from the user-specified datalist, not the global mixed full-sequence list. This preserves dataset comparability.

### Thought 086/100: Is 1-step/full-sequence necessary for one decoder?
It is strongly beneficial. The unified temporal decoder is designed to exploit continuous temporal data and history. Old skip=3 data can train it, but full-sequence is more aligned.

### Thought 087/100: Should old NPZ fallback zero history be allowed?
For real training, no. Zero history undermines the temporal architecture. It can remain as compatibility fallback, but experiments should use NPZs with real `ego_agent_past` and neighbor past.

### Thought 088/100: What validation metrics matter most?
Primary: ego planning loss, PDMS total, DAC, no-collision, TTC, trajectory consistency. Neighbor loss is secondary but still important for interaction quality.

### Thought 089/100: Should gate metrics remain?
No for the final one-decoder run. Gate metrics are irrelevant. Logging should focus on joint temporal losses, maybe per-agent or per-chunk losses.

### Thought 090/100: What new metrics should be added for one decoder?
Add `valid_joint/ego_future_chunk_loss`, `valid_joint/neighbor_future_chunk_loss`, and optionally near/mid/far horizon losses. This helps identify whether temporal chunks improve long-horizon behavior.

### Thought 091/100: Should per-chunk losses be in W&B?
Yes. DFP's purpose is temporal structure. We need to know if improvements come from short horizon only or from future chunks.

### Thought 092/100: What is the expected performance failure mode?
Initial epochs may be worse than hard DFP because the decoder must learn neighbor and ego jointly from scratch. Early comparison should be at matched epoch, but final judgment should also consider convergence trends.

### Thought 093/100: What is the expected performance advantage?
If successful, ego should become more temporally consistent and interaction-aware, while neighbor prediction remains supervised in the same decoder. This can improve PDMS, collision/TTC, and trajectory consistency without ugly post-hoc blending.

### Thought 094/100: What is the compute failure mode?
Full agent-chunk attention can be too slow or memory-heavy. The architecture must support top-K or factorized interaction from the start, otherwise base training may be impractical.

### Thought 095/100: What is the cleanest first implementation path?
Implement one new decoder mode `joint_temporal_agent` with factorized agent-chunk tokens. Do not modify existing modes except adding the option. Add helper functions and launch script. Keep outputs compatible.

### Thought 096/100: What is the most dangerous shortcut to avoid?
Do not implement “one decoder” as old decoder plus DFP inside a wrapper. That would be a naming trick, not a real one-decoder architecture. There must be only one trajectory decoder stack producing both ego and neighbor futures.

### Thought 097/100: What should happen to staged interaction code?
It can remain as an experimental mode for comparison, but the active training should stop and not be treated as final. Documentation should clearly mark it as superseded by `joint_temporal_agent`.

### Thought 098/100: What exact launch should be used?
Use node01, 8 GPUs, TF32, same filtered full-sequence train/valid lists, same LR `1e-4`, same total epochs `80`, same batch `512`, same PDMS eval enabled. Only architecture changes.

### Thought 099/100: What exact success criterion should decide whether this is better?
At matched epoch, compare planning-quality metrics first: PDMS total, DAC, no-collision, TTC, trajectory consistency, ego lat/lon. Then check neighbor loss to ensure interaction quality did not collapse. Final decision should not rely on one scalar.

### Thought 100/100: What is the final architectural conclusion?
The final architecture should be a single joint temporal trajectory decoder. It should jointly denoise/predict ego and neighbor future chunks from all-agent history/current/future chunk tokens and scene encoding. It should not use hard ego replacement, ego-output gate blending, or staged neighbor-to-ego decoder chaining. The principled path is unified joint temporal diffusion forcing over all dynamic agents.

## Final design specification

### Decoder mode

```text
joint_temporal_agent
```

### Inputs

```text
ego_agent_past
neighbor_agents_past
ego_current_state
noisy all-agent future chunks
scene_encoding
agent validity masks
future validity masks
chunk diffusion times
```

### Internal token layout

```text
[B, A, C, D]
A = 1 + predicted_neighbor_num
C = 1 history chunk + 1 current chunk + future_len / chunk_len future chunks
D = hidden_dim
```

### Output contract

```text
prediction: [B, A, future_len, 4]
turn_indicator_logit: [B, TURN_INDICATOR_OUTPUT_DIM]
joint_dfp_x0: optional all-agent chunk reconstruction output for loss/debug
```

### Loss

```text
total_loss =
    alpha_planning_loss * ego_planning_loss
  + alpha_neighbor_loss * neighbor_prediction_loss
  + turn_indicator_loss
  + coeff_road_border_loss * road_border_loss
  + coeff_neighbor_collision_loss * neighbor_collision_loss
  + dfp_lambda_future * joint_future_chunk_loss
  + dfp_lambda_hist * joint_history_chunk_loss
  + dfp_lambda_current * joint_current_chunk_loss
```

### Inference

```text
sample all-agent future chunks jointly
clamp history/current chunks to observed clean conditions
return future chunks as ego + neighbor predictions
compute turn indicator from final ego prediction
```

### Why this is the clean final direction

```text
One trajectory decoder.
One interaction space.
One temporal diffusion-forcing formulation.
No ego hard replacement.
No ego trajectory averaging gate.
No staged neighbor decoder feeding ego decoder.
Ego and neighbors are modeled as coupled dynamic agents.
```

## Training launch record: 2026-06-27 02:24 JST

Final selected long-term architecture is running as the first base-train experiment.

- Run name: dfp_joint_temporal_agent_step1_base_pdms_tier4main_node01_8gpu_tf32
- W&B project: advanced-technology-department/Diffusion-Planner-Temporal
- W&B run: https://wandb.ai/advanced-technology-department/Diffusion-Planner-Temporal/runs/yq79g7jk
- Node: node01
- GPUs: 8 x CUDA, DDP nproc_per_node=8
- Decoder mode: joint_temporal_agent
- Data: filtered full-sequence SFT-derived step3 lists, not global mixed full-sequence data
- Train list: /mnt/nvme/Diffusion-Planner-dfp-shared-stack/artifacts/full_sequence_sft_from_20260622_step3/path_list_train_sft_fullseq_from_20260622_step3.json
- Valid list: /mnt/nvme/Diffusion-Planner-dfp-shared-stack/artifacts/full_sequence_sft_from_20260622_step3/path_list_valid_sft_fullseq_from_20260622_step3.json
- Launcher log: /mnt/nvme/Diffusion-Planner-dfp-shared-stack/launch_step1_base_pdms_joint_temporal_node01.log
- Output dir: /mnt/nvme/Diffusion-Planner-dfp-shared-stack/outputs/dfp_joint_temporal_agent_step1_base_pdms_tier4main_node01_8gpu_tf32
- Main matched settings: batch_size=512, gradient_accumulation_steps=2, learning_rate=1e-4, warm_up_epoch=5, train_epochs=80, TF32=True, PDMS eval=True, skip_filter=False
- Initial validation completed without traceback or OOM, then entered Training epoch 0.
- Initial valid snapshot before training: valid_loss_ego=136.798, valid_loss_neighbor=1235.769, valid_loss_ego_position_lat_loss=0.977, valid_loss_ego_position_lon_loss=16.523, valid_loss_ego_trajectory_consistency=249.095.

Architectural meaning of this run: no hard ego replacement, no learnable ego gate, no staged two-decoder interaction branch. One temporal agent decoder predicts ego and neighbors jointly using history/current/future chunks under DFP-style temporal forcing.
