# DFP result update template

Use this after `/tmp/dfp_vs_baseline20_report.tsv` or both exact SFT20 logs are
available.

## Inputs

```text
DFP log:
Baseline log:
Report:
W&B DFP run:
W&B baseline run:
```

## Required comparison

Run the offline analyzer:

```bash
python /mnt/nvme/Diffusion-Planner-dfp-shared-stack/tools/analyze_dfp_metrics.py \
  --dfp-log <dfp train_log.tsv> \
  --baseline-log <baseline train_log.tsv> \
  --epoch 20
```

## Decision

```text
confirmed_strong | confirmed_by_best_epoch | promising_but_not_confirmed | not_confirmed
```

## Notes

Record:

- best ego delta
- best lat delta
- best lon delta
- best neighbor delta
- epoch20 ego delta
- epoch20 neighbor delta
- whether shared-stack should launch next
