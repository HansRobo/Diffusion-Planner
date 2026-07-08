# DFP joint temporal implementation audit - 2026-06-27

## Paper facts re-read from arXiv 2606.11019 source

DFP is a chunk-wise diffusion planner for ego trajectory generation. The trajectory state is `(x, y, cos(theta), sin(theta))`. The full trajectory is split into history, current, and future chunks. In the paper's implementation details, each clip uses 2 s history and 8 s future sampled at 10 Hz, chunk length `L=20`, and `N=6` chunks: one history chunk, one repeated-current chunk, and four future chunks. The current state is repeated to one chunk and its diffusion time is fixed to zero as a hard anchor.

Training predicts `x0` for history and future chunks under independently sampled chunk noise. The main method states per-block uniform timesteps; the implementation details specify Beta-distributed history timesteps and uniform future timesteps. The loss is the weighted sum of history reconstruction and future reconstruction.

Inference uses two branches. The unguided branch replaces history with pure noise and sets history timestep to one. The guided branch noises the clean history with `t_hist = t_s ** beta`. The two `x0` predictions are fused as `x0_unguided + w * (x0_guided - x0_unguided)`. The best reported hyperparameters in the paper are `w=0.2` and `beta=2.0`.

The paper is ego-only DFP on top of Diffusion Planner's scene encoder. Our `joint_temporal_agent` is an intentional extension: one temporal-agent decoder predicts ego and neighbors together. It is not exactly the paper architecture, but it preserves DFP's core mechanisms: history/current/future chunks, current hard anchor, independent per-chunk timestep, x0 prediction, history noising, and annealed history CFG.

## Invalid run

The previous run `dfp_joint_temporal_agent_step1_base_pdms_tier4main_node01_8gpu_tf32` / W&B `yq79g7jk` is invalid for model selection. Its validation path originally evaluated the original DiT inference path while training used the joint temporal decoder. Therefore constant validation/PDM-DS metrics from that run were not evidence about the trained joint model.

## Confirmed bugs fixed

1. Joint temporal validation/inference dispatch now goes through `_forward_joint_temporal_inference`, not the original DiT inference path.
2. DFP history/current construction now inverse-normalizes observations before applying DFP state normalization, avoiding double normalization.
3. Joint DFP uses agent-wise state normalization, so neighbors use neighbor statistics instead of ego statistics.
4. Joint inference inverse-normalizes agent-wise predictions and masks invalid neighbor futures back to zero.
5. Validation creates `sampled_trajectories` on the target device and no longer double-counts `total_samples_ego`.
6. Validation saves/restores RNG and uses a fixed validation seed, so stochastic DFP validation is deterministic and does not perturb training RNG.
7. Joint DFP training turn-indicator logits use GT trajectory when available, matching original DP training behavior and avoiding turn loss backprop through predicted ego future.
8. Joint DFP auxiliary loss now separates ego and neighbor terms and scales neighbor terms by `alpha_neighbor_loss`; neighbor count no longer dominates DFP loss magnitude.
9. Invalid joint tokens are zeroed before and after temporal/agent/cross attention blocks.
10. Resume train-log fallback now has top-level `import csv`; `csv.DictReader` will not crash when pandas is unavailable.
11. `joint_temporal_agent` no longer instantiates the unused original `dit` decoder or unused ego-only `dfp_dit`; the active architecture is one temporal-agent decoder plus the turn-indicator head.
12. The node01 launcher default run name no longer says node02.

## Design decisions after paper alignment

`use_ego_history=False` is the correct default for the joint DFP launcher. The paper explicitly criticizes static ego-history conditioning because it can cause copying/causal confusion. History should enter through DFP chunks and annealed CFG, while the scene encoder supplies map, route, neighbor, goal, shape, and turn-indicator context.

The current implementation keeps neighbor history in the encoder because the paper's critique is specifically about ego history copying. Neighbor history is still important scene context for interaction. The joint decoder also receives neighbor history/current/future chunks and predicts neighbors jointly, which is a stronger interaction-aware extension than the paper's ego-only DFP.

The sampler is DFP-style x0 iterative sampling, not exact paper DPM-Solver. This is a remaining implementation difference. It should be treated as a design approximation unless replaced with a true chunk-wise DPM-Solver wrapper later.

The paper mentions linear feathering when stitching overlapping blocks. Our chunks do not overlap, so no feathering is applied. This is consistent with non-overlap chunking but not a literal implementation of that sentence.

## Clean run to use next

Use a fresh run only. Do not resume `yq79g7jk`.

Recommended run name:
`dfp_joint_temporal_agent_v3_clean_step1_base_pdms_tier4main_node01_8gpu_tf32`

Recommended key launcher overrides:

- `DFP_DECODER_MODE=joint_temporal_agent`
- `USE_EGO_HISTORY=False`
- `SKIP_FILTER=False` for `artifacts/full_sequence_sft_from_20260622_step3/*fullseq_from_20260622_step3.json` because `summary.json` records that these lists were already filtered by adjacent JSON `is_skipped == false`; use `SKIP_FILTER=True` only when consuming an unfiltered full-sequence list directly
- no `RESUME_MODEL_PATH`
- no `WANDB_RUN_ID`
