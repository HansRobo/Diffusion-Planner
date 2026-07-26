# RLVR campaign code structure: assessment and merge plan

Last updated: 2026-07-26

## 中文摘要（先读这段）

当前 PlannerRFT/AWR 管线**能跑，但不具备合并到主分支的条件**，三个阻塞项按严重程度排序：

1. **整套生产管线不在版本控制里。** 正在跑的 6 个脚本和它依赖的 8 个工具全部 untracked，
   `rlvr/autoresearch/` 下已跟踪的 125 个 `tools/*.py` 全是上一代 GRPO 的。这块盘丢了管线就没了。
2. **实验配置被编码进 shell 脚本。** 每做一个新实验就复制一份 300-400 行脚本改常量，
   `gatefix` 与 `jitterfix` 两个入口有 273 行完全相同。2026-07-26 的三次 campaign 重启
   全部源于此：`beta` 在三处声明了两个不同的值；alpha 阶梯缺了唯一被审计验证过的 `0.05`；
   语料规模常量写死在两个脚本里。
3. **决策逻辑写在 bash 里，无法测试。** supervisor 是 1192 行 bash 内嵌 9 段共约 172 行 Python。
   当天两个 veto 缺陷（读一个不存在的 summary key、用一个饱和的统计量）恰好都在这个边界上。

已经做了的第一步：把数值契约收敛到 `rlvr/campaign_contract.py`（Python 与 shell 共用一个来源，
12 个测试主动检测漂移）。剩下的需要在 campaign 停机窗口做，原因见下面的操作纪律。

## Inventory

`rlvr/autoresearch/` holds 140 tracked + 108 untracked files. The untracked set
classifies as:

| Class | Count | Disposition |
|---|---|---|
| Canonical campaign chain (`.sh`) | 7 | **track** |
| Pipeline tools the chain imports | 8 | **track** |
| Tests | 19 | **track** — they are the only safety net |
| Design/audit docs | 5 | **track** |
| Configs + sensitivity probes the chain invokes | 8 | **track** |
| Historical one-shot scripts | 87 | triage: archive or delete |

The canonical chain, for the record — nothing else should be treated as an entry
point:

```
run_plannerrft_autonomous_campaign.sh      keeper: restarts the entrypoint, bounded backoff
└── run_plannerrft_jitterfix_to_epoch100.sh  entrypoint: cycle-1 mine → overlay → replay
    └── run_plannerrft_full_to_epoch100.sh   supervisor: cycles 2-10
        └── run_conditional_train_selector_line_search.sh   per-cycle commit decision
monitor_plannerrft_epoch100.sh             health monitor (started at handoff)
launch_wandb_heartbeat.sh                  W&B status publisher
check_gatefix_campaign.sh                  one-shot local verdict
```

Tools: `analyze_replay_target_geometry`, `attach_replay_decoder_context`,
`build_compressed_replay_context`, `build_positive_anchor_replay_overlay`,
`campaign_wandb_heartbeat`, `enforce_cycle_non_regression`,
`full_paired_eval_report`, `interpolate_awr_checkpoint`.

## Blocker 1 — the pipeline is not in git

Every script above is untracked. `rlvr/autoresearch/README.md` (453 lines)
documents an earlier GRPO/LoRA generation and never mentions these entry points,
so a reader cannot find them.

**Action:** track the 47 essential files; rewrite the README's entry-point
section to the chain above; move the 87 one-shots under
`rlvr/autoresearch/oneshot/` with a note that they are historical and unmaintained.

## Blocker 2 — configuration lives in forked scripts

New experiment = copy the entrypoint, edit constants. Consequences observed on
2026-07-26:

| Value | Where it was declared | Damage |
|---|---|---|
| `beta` | config `1`, supervisor `2`, entrypoint hardcoded `2` | cycle 1 trained with the weighting the probe ranked worst |
| commit alpha ladder | a third script, `(0.10 0.25 0.50 0.75)` | `0.05` — the only audited-good step — unreachable; cycles committed at 0.5-1.0 |
| corpus size | literals in two scripts | changing the oversampling lists failed a cache-shape check hours into a mine |

**Done:** `rlvr/campaign_contract.py` is now the single source. Python imports
it; shell gets the same values via
`eval "$(python -m rlvr.campaign_contract --shell)"`. `rlvr/test_campaign_contract.py`
fails if the AWR config, the line-search ladder, or the supervisor default drifts
from it, and checks the corpus padding arithmetic against the three cache shapes
actually observed.

**Remaining:** wire the shell entrypoints to `--shell` and delete their local
re-declarations. Requires a stop window (see below).

## Blocker 3 — decision logic is untestable bash

The supervisor is 1192 lines of bash with 9 embedded Python heredocs (~172
lines). Both non-regression-veto defects found on 2026-07-26 lived at that
boundary and neither could have been caught by a test.

**Target:** the per-cycle decision path becomes Python modules with tests, and
bash keeps only process orchestration (launch torchrun, wait, restart):

```
rlvr/campaign/            new package
  contract.py             ← campaign_contract.py moves here
  selection.py            ← the alpha ladder + step preference
  non_regression.py       ← enforce_cycle_non_regression.py moves here
  corpus.py               ← list assembly, oversampling, padding arithmetic
  spec.py                 ← a campaign spec (JSON) replaces the forked entrypoint
```

`enforce_cycle_non_regression.py` (9 tests) is the precedent: it was extracted,
tested, and immediately caught a defect that would have vetoed every cycle for
100 epochs.

## Blocker 4 — hardcoded personal paths

All 11 canonical scripts hardcode `ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main`.
The pipeline only runs from one person's home directory. `ROOT` should derive from
the script location (`git rev-parse --show-toplevel` or `${BASH_SOURCE[0]}`), with
dataset roots and artifact roots as env overrides.

## Operational discipline — do not edit a running script

`bash` reads a script incrementally as it executes. Editing a script that is
mid-run shifts byte offsets and can make it execute garbage. The campaign
entrypoint runs for ~10 h during a cycle-1 mine, and the supervisor runs for
days.

Therefore: **all shell refactoring waits for a stop window** — between a cycle's
commit and the next mine, or at a deliberate restart. Python modules and tests
are safe to change at any time because each stage re-imports them at launch.

The same rule produced a confusing incident on 2026-07-26: a surviving
entrypoint rebuilt an overlay while its script was being edited, and the
resulting state took several minutes to diagnose.

## Sequencing

1. **Now, safe:** track the 47 essential files; contract module + tests (done);
   this document.
2. **Next stop window:** wire the entrypoints to the contract; derive `ROOT`;
   collapse `gatefix`/`jitterfix` into one entrypoint + a campaign spec.
3. **Before the merge request:** move the decision path into `rlvr/campaign/`
   with tests; archive the 87 one-shots; rewrite
   `rlvr/autoresearch/README.md` around the canonical chain.

Step 2 changes no numeric behaviour, so it does not invalidate a running
campaign's results. Step 3 must not change behaviour either — the extracted
Python has to reproduce the bash decisions bit-for-bit, which is exactly what the
contract tests are there to pin.
