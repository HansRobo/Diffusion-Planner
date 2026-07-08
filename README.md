# Diffusion Planner - Hyper Diffusion Planner branch

This branch is the Tier IV integration branch for Hyper Diffusion Planner (HDP). It keeps the original Diffusion Planner training/evaluation pipeline usable, but the branch is documented and maintained as an HDP-focused branch.

## Branch status

Primary mode: HDP.

Compatibility mode: original Diffusion Planner supervised training can still be run by disabling the HDP flags, but this branch should not be treated as a clean vanilla-DP branch. Use upstream `tier4-main` for a pure baseline.

The HDP path adds:

- Ego velocity representation for the HDP ego trajectory target.
- Hybrid velocity-to-waypoint planning loss.
- Official HDP-style reward-weighted RL-Hybrid objective.
- EPDMS / temporal-stability validation support inherited from this development line.
- Full-sequence data compatibility for temporal metrics and history-aware training.

Local references used for implementation:

- Paper TeX: `reference/hyper_diffusion_planner_paper/src/`
- Official code: `reference/external/Hyper-Diffusion-Planner/`

These reference files are for local research and do not need to be included in production PRs unless explicitly requested.

## Setup

This workspace uses `uv`.

```bash
uv sync
source .venv/bin/activate
uv run pre-commit install
python3 -c "import torch; print(torch.cuda.is_available())"
```

Run hooks manually before preparing PRs:

```bash
uv run pre-commit run --all-files
```

## Data

Use the fixed full-sequence train/valid lists for HDP experiments. Do not mix unrelated project/area lists into these runs.

Current HDP experiments use branch-local artifacts such as:

```text
artifacts/full_sequence_base_from_20260622_step3/path_list_train_fullseq_from_20260622_step3.json
artifacts/full_sequence_base_from_20260622_step3/path_list_valid_fullseq_from_20260622_step3.json
```

SFT should use the corresponding SFT full-sequence list. Temporal consistency metrics require consecutive frames; single-frame lists can still train the model but cannot evaluate inter-frame consistency correctly.

## Recommended training order

The intended HDP workflow is:

1. Base train from scratch with HDP velocity representation and hybrid loss.
2. SFT from the base checkpoint using `--init_weights_path`.
3. Official HDP-RL from the SFT checkpoint using `train_grpo_predictor.py` and `--init_weights_path`.

Do not start RL directly from base unless the experiment is explicitly labeled as an ablation.

Detailed commands and flag policy are in:

```text
docs/hyper_diffusion_planner.md
diffusion_planner/README.md
```

## Checkpoint compatibility

HDP velocity checkpoints and vanilla waypoint checkpoints are not semantically interchangeable.

Safe patterns:

- Vanilla waypoint checkpoint to HDP base/SFT bootstrap: use `--init_weights_path` only when this is intentional.
- HDP base to HDP SFT: use `--init_weights_path`.
- HDP SFT to HDP-RL: use `--init_weights_path`.
- Exact interrupted-run continuation: use `--resume_model_path`.

Unsafe pattern:

- Using `--resume_model_path` to reinterpret waypoint latents as velocity latents.

The training code performs representation checks to prevent silent checkpoint misuse.

## W&B

Use the same comparison project for HDP / DFP / quality-fix experiments unless a run is intentionally isolated:

```text
Diffusion-Planner-Temporal
```

Temporary failed runs should be removed after investigation so the project remains readable.
