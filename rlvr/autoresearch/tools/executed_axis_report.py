"""Read the executed waypoint off two eval epochs and say whether it moved.

One frame is one plan step: the vehicle drives index 0 of the plan and the next replan
throws the other 79 away.  Every aggregate the eval already prints is about the other 79 --
``ade`` averages the executed step with them, ``det_progress`` is anti-correlated with it
per scene (r = -0.13), and the crossing gates exclude it outright.  So an anchor arm's whole
justification used to need a separate offline re-score, which is why a 10-epoch cycle judged
it at ~17% of its effect.

``scenes.json`` now carries the primitives (``det_exec_*`` / ``gt_exec_*``, added alongside
the ade alignment fix), so this is a read.  Everything here is PAIRED per scene against a
baseline epoch of the same arm -- the corpus is 46,262 fixed scenes in a fixed order, and
the between-scene spread is two orders above the effect.

The headline is ``lateral excess closed``: the model's mean |lateral| at the executed step
minus the human's, which is the quantity the step-1 anchor exists to drive to zero.  The
human's own floor is not zero (it is ~0.94 mm), so a percentage of the *excess* is the only
honest framing; the raw mean flatters any arm that simply drives straighter.

The second axis is ``chain jitter``.  Reach and steadiness are the two things the arms
actually differ on, and steadiness is not a per-scene quantity: it is the difference of the
executed lateral between *consecutive replans*, i.e. the lateral command the vehicle would
jerk on.  Part of the corpus is sampled densely enough to recover those chains (see
``chain_runs``), so it costs nothing but was previously unreadable here for a second reason --
only ``|lateral|`` was recorded, and a difference of absolute values is not a jitter.
``det_exec_lat_m`` is signed, so it is a jitter now.

Usage:
  python -m rlvr.autoresearch.tools.executed_axis_report \
    --baseline  OUT/eval_epoch_000/scenes.json \
    --candidate OUT/eval_epoch_012/scenes.json
  # or a whole arm at once, every epoch against the first:
  python -m rlvr.autoresearch.tools.executed_axis_report --arm OUT
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Any

import numpy as np

# The executed-step primitives.  A scenes.json written before they existed has none of
# them, which is reported rather than silently treated as zeros.
REQUIRED = (
    "det_exec_lat_m",
    "gt_exec_lat_m",
    "det_exec_lon_m",
    "gt_exec_lon_m",
)
# Instantaneous speed at the executed step, from the logged human: one step is 0.1 s.
HZ = 10.0
# 1.0 m/s is a boundary, not a round number: it is `reward.low_speed_steer_speed_mps`, above
# which the only reward term that reads the executed step goes blind, and the sign of AWR's
# effect on that step flips across it.  A band that straddles it averages two opposite effects.
BANDS = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 1e9))


def load(path: str) -> dict[str, np.ndarray]:
    with open(path) as handle:
        blob = json.load(handle)
    rows: list[dict[str, Any]] = blob["scenes"] if isinstance(blob, dict) else blob
    missing = [key for key in REQUIRED if key not in rows[0]]
    if missing:
        raise SystemExit(
            f"{path} predates the executed-step fields (missing {missing}).\n"
            "Only epochs evaluated after they were added can be read; re-evaluate the "
            "checkpoint if an older epoch is needed as the baseline."
        )
    keys = [
        key
        for key in rows[0]
        if key.startswith(("det_exec_", "gt_exec_")) or key == "scene_path"
    ]
    out: dict[str, np.ndarray] = {}
    for key in keys:
        if key == "scene_path":
            out[key] = np.asarray([str(row[key]) for row in rows])
        else:
            out[key] = np.asarray(
                [float(row.get(key, float("nan"))) for row in rows], dtype=np.float64
            )
    return out


def align(base: dict[str, np.ndarray], cand: dict[str, np.ndarray]) -> np.ndarray:
    """Index pairs by FULL scene path -- basenames repeat across parent dirs."""

    index = {path: i for i, path in enumerate(base["scene_path"])}
    pairs = [
        (index[path], j) for j, path in enumerate(cand["scene_path"]) if path in index
    ]
    if not pairs:
        raise SystemExit("no scene paths in common between the two epochs")
    return np.asarray(pairs, dtype=np.int64)


def paired(delta: np.ndarray) -> tuple[float, float, float]:
    """Mean, t on the per-scene delta, and a sign-test z. Both, because the mean is
    dominated by a heavy tail while the sign test is not."""

    good = delta[np.isfinite(delta)]
    if good.size < 2:
        return float("nan"), float("nan"), float("nan")
    mean = float(good.mean())
    sd = float(good.std(ddof=1))
    t = mean / (sd / math.sqrt(good.size)) if sd > 0 else float("nan")
    moved = good[good != 0.0]
    if moved.size:
        # Signed the same way as the mean and the t: negative means the majority of scenes
        # moved DOWN, i.e. the error fell.  Counting downward moves instead would print a
        # positive z beside a negative delta for the same improvement.
        up = float((moved > 0).sum())
        z = (up - moved.size / 2.0) / math.sqrt(moved.size / 4.0)
    else:
        z = float("nan")
    return mean, t, z


def chain_runs(paths: np.ndarray, min_len: int = 4) -> list[np.ndarray]:
    """Runs of consecutive replans, as index arrays into ``paths``.

    A scene path is ``.../<log>/<log>_<segment>_<frame>.npz``, and the corpus mixes two
    samplings: most logs are strided (frame gaps of 12, 13, 20, 21) and some are dense
    (gap 1).  Only the dense runs are replan chains, and only they are usable here -- one
    frame is one plan step, so between two consecutive frames the difference of the executed
    lateral IS the lateral command change the vehicle would jerk on.  Strided rows are two
    seconds apart and are dropped rather than differenced, which is why this is not simply
    ``gap != 1`` over a sorted list.

    The key includes the segment field, not just the parent dir: the same frame number recurs
    across segments of one log (1,739 times in the 46,262-scene corpus) and a dirname-only key
    splices those into one chain with a zero-length step in the middle.
    """

    groups: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for i, path in enumerate(paths):
        stem = os.path.basename(str(path)).rsplit(".", 1)[0]
        head, _, frame = stem.rpartition("_")
        if not frame.isdigit():
            continue
        groups.setdefault((os.path.dirname(str(path)), head), []).append((int(frame), i))

    runs: list[np.ndarray] = []
    for items in groups.values():
        items.sort()
        run = [items[0][1]]
        previous = items[0][0]
        for frame, i in items[1:]:
            if frame - previous == 1:
                run.append(i)
            else:
                if len(run) >= min_len:
                    runs.append(np.asarray(run, dtype=np.int64))
                run = [i]
            previous = frame
        if len(run) >= min_len:
            runs.append(np.asarray(run, dtype=np.int64))
    return runs


def _chain_diffs(values: np.ndarray, runs: list[np.ndarray], order: int) -> np.ndarray:
    parts = [np.diff(values[run], n=order) for run in runs if run.size > order]
    return np.concatenate(parts) if parts else np.zeros(0)


def chain_census(paths: np.ndarray, finite: np.ndarray) -> list[np.ndarray]:
    """The chains, printed once, with what fraction of the corpus they cover."""

    runs = chain_runs(paths)
    dropped = sum(1 for run in runs if not finite[run].all())
    runs = [run for run in runs if finite[run].all()]
    if not runs:
        print("  chain jitter: no dense replan chains in this corpus")
        return runs
    lengths = np.asarray([run.size for run in runs])
    print(
        f"  chain jitter over {len(runs)} replan chains, {int(lengths.sum())} frames "
        f"({100.0 * lengths.sum() / max(finite.size, 1):.1f}% of the corpus), "
        f"len min/med/max {lengths.min()}/{int(np.median(lengths))}/{lengths.max()}"
        + (f", {dropped} chains dropped for non-finite rows" if dropped else "")
    )
    return runs


def report_chain_jitter(
    runs: list[np.ndarray],
    base: np.ndarray,
    cand: np.ndarray,
    human: np.ndarray,
    axis: str,
    orders: tuple[int, ...] = (1, 3),
) -> None:
    """Steadiness of an executed axis along replan chains, against the human's own.

    This is what the aggregated metrics cannot see at all: a property of the *sequence* of
    executed waypoints, not of any scene.  Reported as rms of the first and third differences
    -- d1 is the command step and d3 its jerk -- with the human's own chain jitter as the
    floor, since a human driver does not hold a constant offset either.

    Lateral is the axis the anchor arms separate on.  Longitudinal is a guardrail: it is the
    comfort axis nothing else here watches, and it is free from the same fields, so a silent
    acceleration-jerk regression cannot hide behind a lateral win.
    """

    for order in orders:
        name = f"chain d{order} ({axis})"
        b = _chain_diffs(base, runs, order)
        c = _chain_diffs(cand, runs, order)
        h = _chain_diffs(human, runs, order)
        if b.size == 0:
            continue
        b_rms = float(np.sqrt((b**2).mean())) * 1000.0
        c_rms = float(np.sqrt((c**2).mean())) * 1000.0
        h_rms = float(np.sqrt((h**2).mean())) * 1000.0
        # The test is on |d| per sample, not on the rms: the chains are few (hundreds) but the
        # differences are many (tens of thousands), and both epochs difference the same frames.
        mean, t, z = paired(np.abs(c) - np.abs(b))
        base_excess, cand_excess = b_rms - h_rms, c_rms - h_rms
        closed = (
            100.0 * (base_excess - cand_excess) / base_excess
            if abs(base_excess) > 1e-9
            else float("nan")
        )
        print(
            f"  {name:<22s} {b_rms:9.4f} -> {c_rms:9.4f} mm rms"
            f"   |d| delta {mean*1000.0:+9.4f}  t {t:+7.2f}  sign z {z:+6.2f}\n"
            f"  {'':<22s} human floor {h_rms:.4f} mm rms; excess closed {closed:+6.2f}%"
        )


def report_axis(
    name: str,
    unit: float,
    suffix: str,
    b_err: np.ndarray,
    c_err: np.ndarray,
    human: np.ndarray | None,
) -> None:
    delta = c_err - b_err
    mean, t, z = paired(delta)
    line = (
        f"  {name:<22s} {b_err.mean()*unit:9.4f} -> {c_err.mean()*unit:9.4f} {suffix}"
        f"   delta {mean*unit:+9.4f}  t {t:+7.2f}  sign z {z:+6.2f}"
    )
    if human is not None:
        floor = float(human.mean())
        base_excess = float(b_err.mean()) - floor
        cand_excess = float(c_err.mean()) - floor
        closed = (
            100.0 * (base_excess - cand_excess) / base_excess
            if abs(base_excess) > 1e-12
            else float("nan")
        )
        line += f"\n  {'':<22s} human floor {floor*unit:.4f} {suffix}; excess closed {closed:+6.2f}%"
    print(line)


def compare(base_path: str, cand_path: str, label: str = "") -> None:
    base, cand = load(base_path), load(cand_path)
    pairs = align(base, cand)
    bi, ci = pairs[:, 0], pairs[:, 1]
    print(f"\n=== {label or os.path.basename(os.path.dirname(cand_path))} "
          f"vs {os.path.basename(os.path.dirname(base_path))}  (n={len(pairs)}) ===")

    # The human is the same trajectory in both epochs, so its columns are the target, not
    # a measurement -- taken from the baseline and asserted identical in the candidate.
    human_lat = np.abs(base["gt_exec_lat_m"][bi])
    if "gt_exec_lat_m" in cand:
        drift = float(np.nanmax(np.abs(cand["gt_exec_lat_m"][ci] - base["gt_exec_lat_m"][bi])))
        if drift > 1e-6:
            print(f"  WARNING: the logged human differs between the two epochs by up to "
                  f"{drift:.6f} m -- the corpora are not the same scenes")

    report_axis(
        "|lateral|",
        1000.0,
        "mm",
        np.abs(base["det_exec_lat_m"][bi]),
        np.abs(cand["det_exec_lat_m"][ci]),
        human_lat,
    )
    # Signed, because the step-0 lateral defect is a bias (+4.972 mm against 1.739 mm of
    # dispersion) and an absolute value cannot show a bias closing.
    report_axis(
        "signed lateral error",
        1000.0,
        "mm",
        base["det_exec_lat_m"][bi] - base["gt_exec_lat_m"][bi],
        cand["det_exec_lat_m"][ci] - cand["gt_exec_lat_m"][ci],
        None,
    )
    if "det_exec_lat_err_m" in base:
        report_axis(
            "paired lateral error",
            1000.0,
            "mm",
            base["det_exec_lat_err_m"][bi],
            cand["det_exec_lat_err_m"][ci],
            None,
        )
    if "det_exec_lon_err_m" in base:
        report_axis(
            "paired longitudinal err",
            1000.0,
            "mm",
            base["det_exec_lon_err_m"][bi],
            cand["det_exec_lon_err_m"][ci],
            None,
        )
    if "det_exec_heading_err_rad" in base:
        report_axis(
            "heading error",
            1000.0,
            "mrad",
            base["det_exec_heading_err_rad"][bi],
            cand["det_exec_heading_err_rad"][ci],
            None,
        )

    speed = np.abs(base["gt_exec_lon_m"][bi]) * HZ
    print("  by speed at the executed step (logged human):")
    for lo, hi in BANDS:
        mask = (speed >= lo) & (speed < hi)
        if not mask.any():
            continue
        b_lat = np.abs(base["det_exec_lat_m"][bi][mask])
        c_lat = np.abs(cand["det_exec_lat_m"][ci][mask])
        h_lat = human_lat[mask]
        mean, t, z = paired(c_lat - b_lat)
        excess = float(b_lat.mean() - h_lat.mean())
        closed = (
            100.0 * -mean / excess if abs(excess) > 1e-12 else float("nan")
        )
        hi_label = "inf" if hi > 1e8 else f"{hi:g}"
        print(
            f"    {lo:5.1f}-{hi_label:<5s} m/s  n={int(mask.sum()):6d}"
            f"  |lat| {b_lat.mean()*1000:8.4f} -> {c_lat.mean()*1000:8.4f} mm"
            f"  human {h_lat.mean()*1000:7.4f}  excess closed {closed:+7.2f}%  t {t:+6.2f}"
        )

    # Signed on all series here, unlike the reach axis: a difference of absolute values is not
    # a jitter, and it was the absence of a signed record that made this unreadable.
    finite = np.ones(bi.size, dtype=bool)
    for key, source, index in (
        ("det_exec_lat_m", base, bi),
        ("det_exec_lat_m", cand, ci),
        ("gt_exec_lat_m", base, bi),
        ("det_exec_lon_m", base, bi),
        ("det_exec_lon_m", cand, ci),
        ("gt_exec_lon_m", base, bi),
    ):
        finite &= np.isfinite(source[key][index])
    runs = chain_census(base["scene_path"][bi], finite)
    if runs:
        report_chain_jitter(
            runs,
            base["det_exec_lat_m"][bi],
            cand["det_exec_lat_m"][ci],
            base["gt_exec_lat_m"][bi],
            "lateral",
        )
        report_chain_jitter(
            runs,
            base["det_exec_lon_m"][bi],
            cand["det_exec_lon_m"][ci],
            base["gt_exec_lon_m"][bi],
            "longitudinal",
            orders=(1,),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", help="scenes.json of the reference epoch")
    parser.add_argument("--candidate", help="scenes.json of the epoch under test")
    parser.add_argument(
        "--arm",
        help="an output dir: every eval_epoch_*/scenes.json is compared to the earliest, "
        "which is how a long run is read while it is still running",
    )
    args = parser.parse_args()

    if args.arm:
        found = sorted(glob.glob(os.path.join(args.arm, "eval_epoch_*", "scenes.json")))
        if len(found) < 2:
            raise SystemExit(f"need at least two eval_epoch_*/scenes.json under {args.arm}")
        for candidate in found[1:]:
            compare(found[0], candidate)
        return
    if not (args.baseline and args.candidate):
        parser.error("give --arm, or both --baseline and --candidate")
    compare(args.baseline, args.candidate)


if __name__ == "__main__":
    main()
