# Diffusion Planner - Hyper Diffusion Planner branch

This branch is the Tier IV integration branch for Hyper Diffusion Planner (HDP). It keeps the original Diffusion Planner training/evaluation pipeline usable, but the branch is documented and maintained as an HDP-focused branch.

## Branch status

Primary mode: HDP.

This branch is HDP-only: the temporal ego-only decoder and velocity action contract are
required. Vanilla Diffusion Planner training is not a supported compatibility mode here;
use upstream `tier4-main` for a pure waypoint/joint baseline.

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

## Quality gates

Both gates must be green before a PR:

```bash
uv run --group dev python -m pytest -m "not benchmark"
uv run ruff check .
uv run ruff format --check .
```

`uv run pytest` resolves to a system binary on the GPU nodes and fails with
`ModuleNotFoundError`; always go through `python -m pytest` as above.

Ruff enforces `B, E, F, I, RET, SIM, W`. Reviewed exceptions are listed per file in
`[tool.ruff.lint.per-file-ignores]` with the reason for each, so a fresh violation is always
a real regression rather than accumulated noise. `F` in particular is what catches a name
that no longer resolves — do not reintroduce `import *`, which silently disables it.

## Data

Use the fixed train/valid lists for HDP experiments. Do not mix unrelated project or area
lists into a run.

**The launchers in `diffusion_planner/slurm/` are the source of truth for dataset paths.**
Each one pins its corpus in a `HDP_*_LIST` default at the top of the file; read that rather
than copying paths from prose, which goes stale. The corpora currently in use are:

| Corpus | Used by |
| --- | --- |
| `20260707_vehicle_params_with_mirror/path_list_train_concatenated.json` | Base80, HDP-RL |
| `20260623_full_sequence/path_list_train_sft_is_skipped_filtered.json` | SFT |
| `20260702_basic_dataset/path_list_valid_sft_balanced.json` | Base80 validation |

Source lists are treated as immutable inputs; a job never rewrites them. Temporal consistency
metrics require consecutive frames, so a single-frame list can still train the model but
cannot evaluate inter-frame consistency correctly.

## Recommended training order

The intended HDP workflow is:

1. Base train from scratch with HDP velocity representation and hybrid loss.
2. SFT from the Base final `latest.pth` using `--init_weights_path`; weights-only init uses
   the checkpoint EMA policy that Base validation evaluated.
3. HDP-RL from the SFT final `latest.pth` using `train_hdp_rl_predictor.py` and
   `--init_weights_path`; RL likewise initializes from the SFT EMA policy.

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

## Cluster and Slurm

Launchers live in `diffusion_planner/slurm/`. Submit with the staged commit pinned:

```bash
sbatch --export=ALL,HDP_EXPECTED_COMMIT=<sha> \
  diffusion_planner/slurm/run_hdp_ego_only_base80.sbatch
```

- **Commit guard.** `HDP_EXPECTED_COMMIT` is mandatory and must equal the staged checkout's
  HEAD. Use `HDP_ALLOW_DIRTY=1` only for a deliberate dirty-worktree snapshot.
- **Auto-resume.** When `latest.pth` exists in the run's save directory the launcher performs
  a strict resume — model, optimizer, scheduler, EMA, RNG state and epoch — instead of
  starting over.
- **W&B re-attach.** A resume continues the original run automatically: the checkpoint stores
  its `wandb_id` and `train.py` falls back to it when `--wandb_run_id` is not given.
- **Requeue on SIGTERM.** The batch script traps SIGTERM and requeues itself, so losing the
  node costs at most the epoch in flight. To make a deliberate `scancel` stick, either
  `touch <save_dir>/NO_REQUEUE` first or submit with `HDP_REQUEUE_ON_SIGTERM=0`.

The requeue path exists because a node-level `slurmd` restart cancels every job on that node.
Keep automatic apt upgrades disabled on the GPU nodes; `node01` and `node02` currently have
`unattended-upgrades` disabled and `apt-daily-upgrade.timer` masked for that reason.

## W&B

Use the same comparison project for HDP / DFP / quality-fix experiments unless a run is intentionally isolated:

```text
Diffusion-Planner-Temporal
```

Temporary failed runs should be removed after investigation so the project remains readable.
