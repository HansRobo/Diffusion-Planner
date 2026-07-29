# Checkpoint selection: always `latest.pth`

The checkpoint for **every** downstream use is `<run_dir>/latest.pth`. This covers
SFT / RL / staged-head initialization, validation and closed-loop evaluation, ONNX
export, and any checkpoint a report or recommendation points at.

Never select `best_model/` or `best_epdms_model/`. Those directories are written by a
validation-proxy selector (`valid_epdms_total`, `valid_loss_ego`) measured on a small
split that does not predict closed-loop or real-vehicle behaviour. The whole pipeline
is built on the last weights instead:

- `run_hdp_ego_only_base_node01.sbatch` and `run_hdp_ego_only_base80_node02.sbatch`
  strict-resume from `${SAVE_DIR}/latest.pth` and verify the final epoch through it;
- `run_hdp_ego_only_sft_node01.sbatch` initializes from `${BASE_RUN}/latest.pth` and
  asserts that checkpoint is at the expected epoch with an EMA policy present;
- `run_hdp_staged_sft_node02.sbatch` hands each stage's `latest.pth` EMA to the next
  stage, and `run_hdp_policy_lr_probe_node02.sbatch` initializes from
  `${BASE_RUN}/latest.pth`;
- `valid_run.sh` evaluates `${MODEL_DIR}/latest.pth`.

Substituting an earlier "best" checkpoint silently desyncs a run from those epoch
assertions and from the EMA state the next stage expects, and reports numbers for
weights nothing downstream ever consumes.

If a validation metric peaked early, report it — that is a real fact about
overfitting, and it may be a reason to change the *epoch budget* of the next run. It
is not a reason to change which checkpoint is used.

Two things that look like exceptions but are not:

- `epochNNNN/best_model.pth` is a filename quirk. Those are plain periodic epoch
  snapshots, not a selection. The rule is about never picking the `best_model/`
  directory.
- In `train_hdp_rl_predictor.py`, `best_model/` holds the last *accepted* RL policy
  and the `best_valid_score` bookkeeping that strict resume restores. That is internal
  trainer state, not a checkpoint-selection knob.

Any new tool or script that takes a checkpoint must default to `latest.pth`.
