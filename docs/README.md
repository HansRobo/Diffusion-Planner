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

- [`hdp_final_state.md`](hdp_final_state.md) — **start here for HDP.** The single branch, the
  pipeline and its launchers, the checkpoint rule, how to read a result without repeating a
  known measurement error, and the archive tags that hold every unmerged experiment.
- [`hdp_rl.md`](hdp_rl.md) — current HDP-RL experiment contract.
- [`hyper_diffusion_planner.md`](hyper_diffusion_planner.md) — current HDP model contract.
- [`hdp_rl_paper_fidelity.md`](hdp_rl_paper_fidelity.md) — clause-by-clause correspondence
  between the RL implementation and the paper, and what `--rl_paper_exact` pins.
- [`hdp_turn_indicator_sft.md`](hdp_turn_indicator_sft.md) — turn-intent head training and
  its deployment contract.
- [`checkpoint_selection.md`](checkpoint_selection.md) — why every downstream consumer takes
  `latest.pth`.
- [`causal_red_light_candidate_audit_20260715.md`](causal_red_light_candidate_audit_20260715.md)
  — the audit note linked from the visual-evidence library above; kept because that page
  cites it as the provenance of the red-light candidate figure.

Point-in-time review and audit records are deliberately not kept here. They pin line numbers
and findings to a HEAD that has since moved, so a reader cannot tell a live constraint from a
closed one. What survived those reviews is stated in the contract pages above.

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
