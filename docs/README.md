# T4 post-training documents

This directory contains the current Original-DP AWR proposal, its evidence pages, and the
engineering references needed to reproduce or review the work. The full-data training run is
still in progress; none of the pages should be read as a final claim that post-training has
already improved the complete validation set.

## Current reader-facing pages

| Page | Use |
| --- | --- |
| [`t4_dp_awr_conference_presentation.html`](t4_dp_awr_conference_presentation.html) | Chinese conference presentation. This is the preserved conference-edited version. |
| [`t4_dp_awr_material_ja.html`](t4_dp_awr_material_ja.html) | Full Japanese know-how: background, AWR, original HDP-RL, T4 adaptation, metrics, real scenes, multimodality, R2LPL, and proposal. |
| [`t4_dp_awr_presentation_ja.html`](t4_dp_awr_presentation_ja.html) | Japanese presentation version for a short conference-style explanation. |
| [`t4_dp_awr_confluence_ja.html`](t4_dp_awr_confluence_ja.html) | Compact Japanese page prepared for the Confluence handoff. |
| [`t4_dp_awr_visual_assets_ja.html`](t4_dp_awr_visual_assets_ja.html) | Visual evidence library with provenance and the interpretation contract for every figure/GIF. |

Open the full know-how page first; use the presentation pages for meetings and the visual library
when checking whether a scene-level claim is actually supported by the image.

## Evidence and source material

- [`t4_conference_assets/`](t4_conference_assets/) — formal replay, candidate coverage,
  multimodality, training-monitor, safety, lane-mask, OBB, and dynamic GIF assets. The local
  [`README.md`](t4_conference_assets/README.md) defines the evidence tiers.
- [`t4_rl_assets/`](t4_rl_assets/) — real-scene reward probes and historical/context diagnostics.
  These are explicitly labelled as probes or context; they are not full-run checkpoint results.
- [`assets/`](assets/) — shared fonts used by the Japanese pages.

## Engineering references

- [`hdp_rl.md`](hdp_rl.md) — current HDP-RL experiment contract.
- [`hyper_diffusion_planner.md`](hyper_diffusion_planner.md) — current HDP model contract.
- [`hdp_audit_verification_20260714.md`](hdp_audit_verification_20260714.md) and
  [`hdp_code_review_20260712.md`](hdp_code_review_20260712.md) — consolidated implementation
  review and verification findings.
- [`code_review_findings_20260711.md`](code_review_findings_20260711.md) — earlier finalised
  findings referenced by the fourth-round review.
- [`hdp_sft_model_comparison_20260713.md`](hdp_sft_model_comparison_20260713.md) — model/SFT
  comparison needed when explaining why this AWR run targets Original DP.
- [`causal_red_light_candidate_audit_20260715.md`](causal_red_light_candidate_audit_20260715.md)
  and [`upstream_dev_backport_audit_20260716.md`](upstream_dev_backport_audit_20260716.md) —
  current data-quality and upstream-scope audits.

## Generators and publication helpers

The `generate_t4_*`, `audit_t4_visual_assets.py`, `package_t4_visual_assets.py`, and
`publish_t4_awr_confluence.py` scripts are kept because the pages and their provenance can be
regenerated from them. One-off generated HTML, replay outputs, checkpoints, and Python bytecode
are not part of the document source of truth.

## Status vocabulary

- **Formal replay evidence:** exact rows from the frozen training replay cache; proves that the
  reward/replay pipeline contains usable signal, not that a trained checkpoint improved.
- **Probe/diagnostic:** a real scene or fixed validation slice used to test evaluator semantics.
- **Final checkpoint result:** only after the full run, fixed validation, paired comparison, and
  closed-loop checks are complete.
