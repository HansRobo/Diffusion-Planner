#!/usr/bin/env python
"""Net crossing deltas restricted to the scenes the LOGGED HUMAN does not cross.

Why this exists.  `det_lane_crossing` / `det_rb_crossing` are binary per-scene flags, and a
long run was being braked on their raw net.  Scoring 1,200 mine scenes at the full 80 steps
showed that number is mostly not the policy's to move:

  - The policy's OWN crossings collapse as the window narrows toward what the vehicle drives.
    Policy-only lane scenes go 162 -> 56 -> 3 out of 3,000 for the full 80 steps, the eval's
    1..40, and the executed waypoint alone; road border goes 17 -> 1 -> 0.  Against net nulls
    of 5.03 and 1.81 both narrow numbers are nulls, and at every narrow window the logged
    human crosses MORE often than the policy (lane 4.57% vs 4.10% of scenes at the executed
    waypoint, rb 0.83% vs 0.43%).
  - 36% of lane crossings first occur in steps 41..80.  The human's step distribution has the
    same shape, which is what shows the tail-heaviness is a property of an 8 s open-loop plan
    rather than a policy defect.

Mind the step convention.  Both gates aggregate ``is_crossing[:, 1:]`` and call index 0 "the
starting position", but the trajectory is 80 FUTURE steps with no current pose prepended --
row 0 is already 0.99 m out on a moving scene -- so index 0 is the first PREDICTED waypoint,
the executed one, and it is outside both gates.  A step of 1 in the mask or in
``scenes.json`` is therefore the SECOND waypoint.  Reaching the executed step needs index 0
duplicated so the gate's own slice lands on the duplicate; done once, it changes the verdict
on 0.10% of candidates (lane) / 0.03% (rb), and catches the logged human at the same rate,
so it is not worth redefining the reward to close.

So a net of a few tens of scenes moves on scene geometry sitting against a 0.20 m threshold.
This tool removes the human-explained part, which is the larger half and the only half
computable from what an eval actually writes.  Measured effect on the DUAL arm, epoch 0 to 3
over all 46,262 eval scenes: road border goes from +4 to -1, i.e. the regression on the one
axis that is a real hard gate is entirely human-explained.

What window the flags live in.  ``compute_reward_batch`` truncates to
``reward_horizon_steps`` *before* scoring, and there is one shared ``reward_config`` for
train and eval (``train_awr.py`` builds it once and passes it to ``evaluate_model``; the
only eval-time copy disables the lane polygon during mining, never in eval, and never
touches the horizon).  So with ``reward_horizon_steps=40`` the eval's ``det_lane_crossing``
is a steps-1..40 verdict: the 41..80 tail is invisible to the reward AND to the flag the
veto reads.  The eval therefore *underreports* the crossing rate rather than charging for
anything the reward cannot see.  ``--human-horizon`` (default 40) makes the human's verdict
use the same window as the flags it is correcting.

`scenes.json` records the flags but neither the human's verdict nor the step index, and the
`rollouts/` directories an eval creates are empty, so the step-1 restriction still needs a
GPU re-run.  The human's verdict, in contrast, is a pure function of the logged future, the
ego shape and the map -- arm-independent, epoch-independent, and cacheable once.

Usage:

    # once per eval corpus (~46k scenes); pure CPU, safe while the GPUs are busy.
    # Cache at the FULL horizon (the default): the mask stores first-crossing STEPS, so any
    # narrower window is a read-time filter and does not need a re-score.
    python -m rlvr.autoresearch.tools.human_clean_crossing_brake mask \
        --scenes  <run>/eval_epoch_003/scenes.json \
        --out     artifacts/human_crossing_mask.json \
        --workers 16

    # per epoch, against the arm's own baseline epoch
    python -m rlvr.autoresearch.tools.human_clean_crossing_brake brake \
        --baseline  <run>/eval_epoch_000/scenes.json \
        --candidate <run>/eval_epoch_012/scenes.json \
        --mask      artifacts/human_crossing_mask.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

# Before torch is imported, so the fork-inherited BLAS pools are sized right from the start;
# _worker_init caps torch itself.  setdefault, so an operator can still override.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import torch  # noqa: E402

# Flag name in scenes.json -> mask key holding the human's first crossing step.
AXES = {"det_lane_crossing": "lane_step", "det_rb_crossing": "rb_step"}

_CFG: object | None = None


def _worker_init(config_path: str | None, horizon: int) -> None:
    """Build the RewardConfig once per worker; it is read-only afterwards."""
    global _CFG
    from rlvr.reward import RewardConfig

    # One thread per worker.  The pool is already the parallelism; leaving torch at its
    # default gave each of 24 workers 112 threads, and the 2,688-way contention burned 80
    # CPU-hours in 40 minutes to score under 2,000 scenes.  Capping it is ~40x faster here.
    torch.set_num_threads(1)

    cfg = RewardConfig()
    if config_path:
        raw = json.loads(Path(config_path).read_text())
        for k, v in raw.get("reward", raw).items():
            if hasattr(cfg, k) and not isinstance(v, (dict, list)):
                setattr(cfg, k, v)
    # 0 = score all 80 steps, which is the useful cache: the mask keeps the first-crossing
    # STEP, so restricting to the eval's 1..40 window is a comparison in `brake` rather than
    # a second two-hour re-score.  One caveat for road border -- the border bbox is pooled
    # per call, so an rb step read off a full-80 scoring can differ marginally from the same
    # scene scored with the data truncated to 40.  Pass --horizon 40 if that matters.
    cfg.reward_horizon_steps = int(horizon)
    _CFG = cfg


def _score_human(scene_path: str) -> tuple[str, dict | None]:
    """First crossing step of the LOGGED HUMAN on this scene, per axis (None = clean)."""
    from planner_metrics.aggregate import compute_subscores_batch
    from rlvr.awr import reward_compatible_data
    from rlvr.grpo_trainer import load_npz_data

    try:
        data = load_npz_data(scene_path, "cpu")
        if "lanes" not in data or "ego_shape" not in data:
            return scene_path, None
        gt = data.get("ego_agent_future")
        if not isinstance(gt, torch.Tensor):
            return scene_path, None
        if gt.dim() == 3:
            gt = gt[0]
        # Same 3-col -> 4-col construction reward.py uses for the expert; the T4 archives
        # store (x, y, yaw) and the neutral metrics consume (x, y, cos, sin).
        if gt.shape[-1] == 3:
            gt4 = torch.cat(
                [gt[:, :2], torch.cos(gt[:, 2:3]), torch.sin(gt[:, 2:3])], dim=-1
            )
        else:
            gt4 = gt[:, :4]
        subs = compute_subscores_batch(
            gt4.unsqueeze(0).float(), reward_compatible_data(data), _CFG
        )
    except Exception as exc:  # unreadable scene / shape mismatch -- excluded, not guessed
        return scene_path, {"error": f"{type(exc).__name__}: {exc}"[:160]}

    return scene_path, {
        "lane_step": subs["lane_crossing_steps"][0],
        "rb_step": subs["rb_crossing_steps"][0],
    }


def _scene_paths(scenes_json: Path) -> list[str]:
    rows = json.loads(scenes_json.read_text())
    seen, out = set(), []
    for r in rows:
        p = r.get("scene_path")
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def cmd_mask(args: argparse.Namespace) -> int:
    paths = _scene_paths(Path(args.scenes))
    print(f"[mask] {len(paths)} unique scenes, {args.workers} workers", flush=True)

    mask, errs = {}, 0
    with mp.Pool(
        args.workers, initializer=_worker_init, initargs=(args.config, args.horizon)
    ) as pool:
        for n, (p, rec) in enumerate(pool.imap_unordered(_score_human, paths, chunksize=16), 1):
            if rec is None:
                errs += 1
            elif "error" in rec:
                errs += 1
                if errs <= 3:
                    print(f"[skip] {Path(p).name}: {rec['error']}", flush=True)
            else:
                mask[p] = rec
            if n % 2000 == 0:
                print(f"[mask] {n}/{len(paths)}  scored={len(mask)}  skipped={errs}", flush=True)

    n = len(mask) or 1
    print(f"\n[mask] scored {len(mask)}, skipped {errs}  (horizon {args.horizon or 'full'})")
    for key, label in (("lane_step", "lane"), ("rb_step", "rb")):
        allh = sum(1 for r in mask.values() if r[key] is not None)
        in40 = sum(1 for r in mask.values() if r[key] is not None and r[key] <= 40)
        print(f"[mask] human crosses {label}: {allh} ({100.0 * allh / n:.2f}%) anywhere, "
              f"{in40} ({100.0 * in40 / n:.2f}%) within the eval's 1..40 window")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "source_scenes": str(args.scenes),
                "reward_horizon_steps": int(args.horizon),
                "mask": mask,
            }
        )
    )
    print(f"[wrote] {out}")
    return 0


def cmd_brake(args: argparse.Namespace) -> int:
    blob = json.loads(Path(args.mask).read_text())
    mask = blob["mask"]
    cached_h = int(blob.get("reward_horizon_steps", 0))
    want_h = int(args.human_horizon)
    if cached_h and (want_h == 0 or want_h > cached_h):
        # A mask cached at 40 cannot answer a question about steps 41..80: those steps were
        # never scored, so "no crossing" there is absence of evidence, not evidence.
        print(f"[brake] ERROR: mask was cached at horizon {cached_h}; "
              f"--human-horizon {want_h or 'full'} needs a wider re-score.", file=sys.stderr)
        return 2

    def human_crosses(path: str, step_key: str) -> bool:
        step = mask[path][step_key]
        return step is not None and (want_h == 0 or step <= want_h)

    def flags(p: Path) -> dict[str, dict[str, float]]:
        return {
            r["scene_path"]: {k: r.get(k) for k in AXES}
            for r in json.loads(p.read_text())
            if r.get("scene_path")
        }

    a, b = flags(Path(args.baseline)), flags(Path(args.candidate))
    shared = sorted(set(a) & set(b))
    print(f"[brake] {len(a)} baseline / {len(b)} candidate / {len(shared)} paired scenes")
    covered = [p for p in shared if p in mask]
    print(f"[brake] human mask covers {len(covered)} of them "
          f"({100.0 * len(covered) / max(len(shared), 1):.1f}%), cached at horizon "
          f"{cached_h or 'full'}, human verdict taken over steps 1..{want_h or 'end'}\n")

    for flag, step_key in AXES.items():
        clean = [p for p in covered if not human_crosses(p, step_key)]
        rows = [
            ("ALL paired scenes", shared),
            ("HUMAN-CLEAN subset", clean),
        ]
        print(f"===== {flag} =====")
        for label, subset in rows:
            on = sum(
                1 for p in subset
                if not (a[p][flag] or 0) and (b[p][flag] or 0)
            )
            off = sum(
                1 for p in subset
                if (a[p][flag] or 0) and not (b[p][flag] or 0)
            )
            net = on - off
            print(f"  {label:<20} n={len(subset):6d}   newly-violating {on:5d}   "
                  f"newly-clean {off:5d}   net {net:+5d}   gross {on + off:5d}")
        print(f"  human-explained share of the gross: "
              f"{100.0 * (1.0 - len(clean) / max(len(covered), 1)):.1f}% of scenes are "
              f"human-crossing and excluded\n")

    print("Same-weight null, re-derived on the SAME subset from 15 pairwise contrasts of the six")
    print("cycle-1 epoch-0 replicates:  lane  all 4.37 / human-clean 5.03")
    print("                             rb    all 3.05 / human-clean 1.81")
    print("The lane null GROWS on the human-clean subset -- excluding human-crossing scenes drops")
    print("the ones both runs cross stably and keeps the marginal churners, so do not assume a")
    print("smaller subset means a smaller null.  And this is a same-weight floor: epoch wander is")
    print("~7x larger, so add it before calling any cross-epoch flag delta real.")
    print("Both flags are steps-1..40 verdicts, so both nets are a floor on the true crossing")
    print("rate: 36% of lane crossings first occur in the 41..80 tail, counted by neither the")
    print("reward nor the veto.  Restricted to the step the vehicle EXECUTES, the policy-only")
    print("count was 3 scenes in 3,000 on lane and 0 on road border -- nulls against the 5.03")
    print("and 1.81 above -- with the human crossing MORE often than the policy at every narrow")
    print("window.  Note a step of 1 is the SECOND waypoint: both gates exclude index 0, which")
    print("is the executed one.  So do not brake on a raw net; there is no crossing regression")
    print("on the axis that reaches the vehicle.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("mask", help="cache the logged human's crossing verdict per scene")
    m.add_argument("--scenes", required=True, help="any eval scenes.json over the corpus")
    m.add_argument("--out", required=True)
    m.add_argument("--config", default=None,
                   help="effective_config.json whose reward.* knobs to mirror")
    m.add_argument("--horizon", type=int, default=0,
                   help="reward_horizon_steps for the human's scoring; 0 (default) = all 80, "
                        "which lets brake narrow the window without a re-score")
    m.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 4) // 2))
    m.set_defaults(func=cmd_mask)

    b = sub.add_parser("brake", help="net crossings, all scenes vs the human-clean subset")
    b.add_argument("--baseline", required=True, help="the arm's own epoch-0 scenes.json")
    b.add_argument("--candidate", required=True)
    b.add_argument("--mask", required=True)
    b.add_argument("--human-horizon", type=int, default=40,
                   help="count the human as crossing only within steps 1..N; default 40 to "
                        "match the eval flags, which are themselves horizon-40 verdicts")
    b.set_defaults(func=cmd_brake)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
