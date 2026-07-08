# DFP shared-stack decoder experiment, 2026-06-25

## Purpose

This worktree is for the next DFP integration step after the validated
`unified_ego` two-decoder experiment.

The current effective implementation has:

- original DP `DiT` decoder for all-agent future denoising, especially neighbor prediction
- separate `DFPDiT` decoder for ego history/current/future chunk diffusion forcing
- DFP ego replacing original ego output in planner loss, validation, and inference

That structure is useful as proof of effect, but it is not the cleanest long-term
architecture because it carries two decoder block stacks.

## New design

Mode name:

```text
dfp_decoder_mode=shared_stack_unified_ego
```

Core change:

- remove the separate DFP transformer block stack for this mode
- reuse the original DP `DiT.blocks` as the only decoder block stack
- keep a small DFP-specific chunk input projection, chunk position embedding, timestep embedder, and output head
- keep original DP all-agent future head for neighbor prediction
- use DFP chunk future as the primary ego trajectory, same as `unified_ego`

This is intentionally a shared-stack design rather than full one-pass joint-token
attention. Full joint attention would allow DFP tokens to perturb agent tokens from
step zero, which weakens checkpoint reuse and neighbor fairness. This first version
therefore shares block weights but decodes DFP chunks through the same block stack as
a DFP pass.

## Weight reuse

Expected checkpoint behavior:

- original encoder weights load unchanged
- original `DiT.blocks` load unchanged and are reused by both original all-agent future decoding and DFP chunk decoding
- original neighbor future head loads unchanged
- original turn indicator head loads unchanged
- new DFP chunk projection, timestep embedder, position embedding, and final head are randomly initialized
- DFP final head is zero-initialized, matching the previous DFP branch convention

This means pretraining does not need to be rerun. The experiment should continue to
initialize from:

```text
/mnt/nvme/Diffusion-Planner-dfp-tier4-additive/checkpoints/base_sft/best_model.pth
```

## Baseline policy

No new baseline is needed if these stay unchanged:

- train/valid lists
- init checkpoint
- batch size
- optimizer and LR schedule
- augmentation
- validation metrics
- original non-DFP losses

The current matched no-DFP baseline can be reused for this shared-stack experiment
because this branch only changes the DFP-enabled model path.

## First run script

```text
diffusion_planner/run_dfp_shared_stack_unified_ego_lam01_sft20.sh
```

Important choices:

- `train_epochs=20`, matching the intended SFT length
- `tf32=True`
- `dfp_lambda_hist=0.1`
- `dfp_lambda_future=0.1`
- `dfp_lambda_original_ego=0.2`
- same train/valid lists as the current matched experiments
- same W&B project: `Diffusion-Planner-Temporal`

The DFP loss weight is lower than the two-decoder run because DFP gradients now flow
through the original shared decoder blocks. This is a neighbor-protection choice.
If neighbor performance remains stable, run a follow-up with `DFP_LAMBDA=0.3` for a
closer comparison to the current two-decoder DFP setting.

## Implementation files

- `diffusion_planner/diffusion_planner/model/module/decoder.py`
- `diffusion_planner/diffusion_planner/train_config.py`
- `diffusion_planner/run_dfp_shared_stack_unified_ego_lam01_sft20.sh`

## Current status

Prepared but not launched yet. The current node02 GPUs are occupied by the matched
no-DFP baseline and queued exact SFT20 experiments. Launch after those reports are
available to preserve a clean comparison timeline.

## Decision tree after matched baseline finishes

Use the current matched stop20 report first:

```text
/tmp/dfp_vs_baseline20_report.tsv
```

If DFP improves ego/lon and neighbor is not materially worse:

- keep the current two-decoder DFP result as the first confirmed win
- launch this shared-stack run with `DFP_LAMBDA=0.1`
- compare shared-stack against the same no-DFP baseline
- if shared-stack keeps the ego win, prefer shared-stack for long-term maintenance

If DFP improves ego/lon but neighbor is materially worse:

- do not discard DFP, because this means the ego method is useful but interaction
  with original neighbor prediction needs protection
- launch shared-stack with `DFP_LAMBDA=0.1`, not `0.3`
- keep `dfp_lambda_original_ego=0.2`
- if neighbor is still worse, add a protected variant:
  - freeze or low-LR original `DiT.blocks` for the first warmup epochs
  - keep new DFP head at normal LR
  - optionally raise `alpha_neighbor_loss`

If DFP does not improve ego/lon by epoch20:

- inspect whether DFP had an earlier best epoch that beats baseline best epoch
- if early best wins but epoch20 does not, the method is useful but needs early
  stopping or lower DFP loss
- if no epoch wins, do not launch higher-lambda DFP; try lower DFP lambda and
  original-ego distillation first

The strongest acceptance condition is:

```text
DFP best ego/lon/lat beats matched no-DFP best on the same valid set,
and neighbor degradation is small enough that downstream closed-loop risk is acceptable.
```

The conservative acceptance condition is:

```text
DFP improves ego/lon significantly, while neighbor and turn are no worse than baseline
within normal validation noise.
```

## Manual launch command

Do not launch while the matched baseline or exact SFT20 queue is still running.
When node02 is idle, run:

```bash
ssh node02 'cd /mnt/nvme/Diffusion-Planner-dfp-shared-stack/diffusion_planner && RUN_NAME=dfp_shared_stack_unified_ego_lam01_sft20_tier4main_node02_8gpu_tf32 PORT=22391 DFP_LAMBDA=0.1 bash ./run_dfp_shared_stack_unified_ego_lam01_sft20.sh'
```

If the lambda 0.1 run preserves neighbor and still improves ego, follow with:

```bash
ssh node02 'cd /mnt/nvme/Diffusion-Planner-dfp-shared-stack/diffusion_planner && RUN_NAME=dfp_shared_stack_unified_ego_lam03_sft20_tier4main_node02_8gpu_tf32 PORT=22392 DFP_LAMBDA=0.3 bash ./run_dfp_shared_stack_unified_ego_lam01_sft20.sh'
```

## Full-sequence SFT variant

The 20260623 full-sequence dataset has been constrained to the current SFT route
whitelist and JSON-filtered:

```text
/mnt/nvme/Diffusion-Planner-dfp-shared-stack/artifacts/full_sequence_sft_from_20260622_step3/path_list_train_sft_fullseq_from_20260622_step3.json
/mnt/nvme/Diffusion-Planner-dfp-shared-stack/artifacts/full_sequence_sft_from_20260622_step3/path_list_valid_sft_fullseq_from_20260622_step3.json
```

Run script:

```text
diffusion_planner/run_dfp_shared_stack_fullseq_lam01_sft20.sh
```

Important caveat:

This full-sequence variant changes the training data density and the available valid
set coverage. It is the right input for temporal-consistency experiments, but it is
not a strict apples-to-apples comparison against the current matched no-DFP baseline
unless a no-DFP baseline is also run on the same full-sequence lists.
