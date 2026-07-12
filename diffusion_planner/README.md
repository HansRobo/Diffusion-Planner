# diffusion_planner entrypoints for the HDP branch

This directory contains the training, validation, and RL entrypoints used by the HDP branch.

## Main entrypoints

| Entrypoint | Purpose |
| --- | --- |
| `train_predictor.py` | Supervised base / SFT training. |
| `train_hdp_rl_predictor.py` | HDP reward-weighted RL-Hybrid fine-tuning. |
| `valid_run.sh` | Validation wrapper for a saved checkpoint. |
| `train_run.sh` | Legacy convenience wrapper. Prefer explicit commands for HDP experiments. |

## HDP base training

Base training should start from scratch and use the same model/data settings as the comparison runs.

The only encoder implementation in this branch integrates Tier IV PRs
[#212](https://github.com/tier4/Diffusion-Planner/pull/212) and
[#210](https://github.com/tier4/Diffusion-Planner/pull/210). Line-string/polygon position
geometry now comes from valid points, and turn-indicator history is categorical one-hot.
The changed turn encoder shape prevents old checkpoints from loading. Start the new ego-only
Base from scratch; there is no legacy encoder mode in this branch.

```bash
cd /mnt/nvme/Diffusion-Planner-hyper-diffusion-planner/diffusion_planner
source ../.venv/bin/activate

python3 -m torch.distributed.run --nproc_per_node=8 --master_port=<PORT> train_predictor.py \
  --exp_name <RUN_NAME> \
  --save_dir <OUTPUT_DIR> \
  --train_set_list <FULL_SEQUENCE_BASE_TRAIN_LIST> \
  --valid_set_list <FULL_SEQUENCE_BASE_VALID_LIST> \
  --train_subsample_step 1 \
  --normalization_file_path ./normalization.json \
  --batch_size 512 \
  --learning_rate 2e-4 \
  --warm_up_epoch 5 \
  --train_epochs 20 \
  --save_utd 10 \
  --future_len 80 \
  --time_len 31 \
  --agent_num 320 \
  --predicted_neighbor_num 0 \
  --diffusion_model_type x_start \
  --diffusion_supervision_type x_start \
  --diffusion_time_sample_method uniform \
  --use_velocity_representation True \
  --planning_hybrid_loss 0.01 \
  --hybrid_loss_window 10 \
  --turn_indicator_generated_loss_weight 1.0 \
  --turn_indicator_expert_loss_weight 1.0 \
  --diffusion_sample_steps 6 \
  --enable_epdms_eval True \
  --use_wandb True \
  --wandb_project_name Diffusion-Planner-Temporal \
  --tf32 True \
  --amp_dtype bf16 \
  --fused_optimizer True \
  --ddp_static_graph True \
  --compile_model True
```

## HDP SFT

SFT uses the same HDP architecture and starts from a base checkpoint with `--init_weights_path`.

```bash
python3 -m torch.distributed.run --nproc_per_node=8 --master_port=<PORT> train_predictor.py \
  --exp_name <SFT_RUN_NAME> \
  --save_dir <SFT_OUTPUT_DIR> \
  --train_set_list <FULL_SEQUENCE_SFT_TRAIN_LIST> \
  --valid_set_list <FULL_SEQUENCE_SFT_VALID_LIST> \
  --init_weights_path <HDP_BASE_CHECKPOINT> \
  --train_subsample_step 1 \
  --normalization_file_path ./normalization.json \
  --train_epochs 20 \
  --save_utd 10 \
  --agent_num 320 \
  --predicted_neighbor_num 0 \
  --use_velocity_representation True \
  --planning_hybrid_loss 0.01 \
  --hybrid_loss_window 10 \
  --diffusion_sample_steps 6 \
  --enable_epdms_eval True \
  --use_wandb True \
  --wandb_project_name Diffusion-Planner-Temporal \
  --tf32 True \
  --amp_dtype bf16 \
  --compile_model True
```

Use `--resume_model_path` only for continuing the same interrupted run. Do not use it for base-to-SFT or SFT-to-RL transfer.

For oversampling without creating a giant combined manifest, pass
`--extra_train_set_list <LIST>` once per source and `--extra_train_set_repeat <N>`. The
Dataset concatenates the extra paths and appends their references in memory; it does not
write either list or any NPZ. `--extra_train_set_mask_traffic_lights True` masks traffic
signals only in those extra samples, in worker memory. Large W&B dataset-list artifacts
are disabled by default and can be explicitly enabled with `WANDB_LOG_DATASET_ARTIFACT=1`.

## HDP-RL for real-vehicle evaluation

RL must start from an HDP SFT checkpoint. The only RL objective is reward-weighted HDP hybrid loss.

```bash
python3 -m torch.distributed.run --nproc_per_node=8 --master_port=<PORT> train_hdp_rl_predictor.py \
  --exp_name <RL_RUN_NAME> \
  --save_dir <RL_OUTPUT_DIR> \
  --train_set_list <FULL_SEQUENCE_SFT_TRAIN_LIST> \
  --valid_set_list <FULL_SEQUENCE_SFT_VALID_LIST> \
  --init_weights_path <HDP_SFT_CHECKPOINT> \
  --rl_init_use_ema True \
  --normalization_file_path ./normalization.json \
  --learning_rate 1e-7 \
  --weight_decay 0 \
  --rl_reward_normalize group \
  --rl_reward_beta 0.5 \
  --rl_reward_w_risk 1.0 \
  --rl_reward_w_follow 3.0 \
  --rl_reward_w_lane 2.5 \
  --rl_reward_w_progress 3.0 \
  --rl_bc_weight 0.0 \
  --num_generations 8 \
  --rl_noise_scale 1.5 \
  --rl_eval_noise_scale 0.5 \
  --rl_eval_num_generations 32 \
  --rl_rollout_steps 6 \
  --rl_updates_per_rollout 4 \
  --rl_ema_update_rate 0.05 \
  --rl_train_scope decoder \
  --predicted_neighbor_num 0 \
  --use_velocity_representation True \
  --planning_hybrid_loss 0.01 \
  --hybrid_loss_window 10 \
  --diffusion_sample_steps 6 \
  --multisample_eval_num_samples 0 \
  --multisample_eval_sample_steps 6 \
  --rl_full_eval_utd 5 \
  --enable_epdms_eval True \
  --use_wandb True \
  --wandb_project_name hdp-rl \
  --tf32 True \
  --amp_dtype bf16 \
  --compile_model True
```

Important semantics:

- `--rl_noise_scale` and `--num_generations` control training exploration. Their `--rl_eval_*`
  counterparts stay fixed at the public-policy temperature and paper group size so held-out
  reward remains comparable across sweeps. The training defaults are the strongest measured
  Tier IV setting (`G=8`, noise `1.5`), not an assertion that the paper used those values.

- `--compile_model True` compiles the encoder and decoder in place. Checkpoint keys and ONNX
  export remain unchanged. Set `TORCHINDUCTOR_CACHE_DIR` to persistent local storage so restarts
  reuse compiled artifacts.
- `--init_weights_path`: fresh RL run initialized from SFT weights only; by default it selects the SFT EMA shadow.
- `--resume_model_path`: strict resume of an interrupted RL run, including optimizer/scheduler state.
- Checkpoints are written through a same-directory temporary file and atomically replaced; strict resume rejects missing or incompatible optimizer, scheduler, epoch, EMA, architecture, and normalization state.
- Strict resume also requires identical data, augmentation, loss, EMA, and RL reward settings. `init_weights_path` remains the weights-only stage-transfer mode, but checkpoints from before the two encoder fixes cannot initialize the current model.
- `--rl_train_scope decoder`: DiT trajectory-policy fine-tuning with one cached scene encoding per candidate group. The separate SFT turn-indicator classifier stays frozen because the RL reward does not supervise it.
- `--train_subsample_step` defaults to `1`; this branch optimizes for final model quality unless explicitly overridden.
- The EMA shadow generates candidate actions as the previous policy; the live decoder receives the reward-weighted update, then refreshes EMA.
- The non-risk behavior reward (following, lane keeping, and DP-native progress) is gated by
  collision safety, so an active collision cannot be traded for another behavior score and rear
  collisions retain continuous attenuation. The independent paper risk term still scores TTC,
  THW, and occupancy near misses.
- Reward uses SAT collision, continuous TTC, THW, static/stopped-agent/road-border occupancy clearance, leader-conditioned following, lane-center scoring, lane-change/off-lane masking, and rear-end attenuation. Logged expert motion determines leader identity while each candidate still determines its gap, speed match, and comfort; this prevents a lateral candidate from earning a neutral following score merely by leaving the leader-association corridor. Road-border occupancy uses exact rectangle-to-segment clearance rather than perimeter-point sampling.
- Candidate groups with identical or non-finite rewards are discarded before the hybrid loss.
- `--predicted_neighbor_num` must be `0`. The temporal HDP action head predicts only ego while the encoder and safety reward still consume all logged neighbors.
- `--align_legacy_neighbor_futures true` fixes the pre-`55eff4f` short-track `t=0` duplication inside each DataLoader worker. It never rewrites or migrates shared NPZ files. Disable it for regenerated data.
- `--export_onnx_on_save` defaults to `false` for SFT and RL so synchronous CPU export does not leave the other DDP ranks idle. Use the standalone converter for a strict release export.
- Six-sample minADE/minFDE is an open-loop diagnostic. EPDMS and pseudo-closed-loop metrics drive safety-oriented model selection; final acceptance requires real-vehicle A/B evaluation.
- Legacy RL alternatives have been removed from this branch; HDP-RL has a single supported reward-weighted hybrid path.
- SFT trains the turn-indicator head on both the detached model-generated x-start trajectory and the expert trajectory. Their normalized weighted mean preserves loss scale; validation reports generated-path overall, change-only, and per-class metrics.

## HDP ONNX export

For HDP velocity checkpoints, deploy the full ONNX graph:

```text
diffusion_planner.onnx
```

This HDP-only branch does not export the legacy split decoder because its ego row is a velocity
latent, not a waypoint latent. The full graph runs sampling and decodes ego velocity back to
waypoint-space `prediction`.

Smoke-test conversion pattern:

```bash
CUDA_VISIBLE_DEVICES="" ../.venv/bin/python ../ros_scripts/torch2onnx.py \
  <CHECKPOINT_DIR_WITH_ARGS_JSON_AND_PTH> \
  --output-prefix diffusion_planner_hdp \
  --opset-version 20
```

Expected HDP output shapes:

```text
ego-only prediction: [B, 1, 80, 4]
turn_indicator_logit: [B, 5]
```

The temporal full graph passed ONNX checker and ONNX Runtime CPU execution with dynamic
batch 1 and 2. Its sampled action input and prediction are both 80-step ego-only tensors;
there is no delay-prefix input.
