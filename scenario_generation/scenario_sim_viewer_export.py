"""Export a finished scenario_sim suite run into the tree the result viewer reads.

Post-hoc only: it reads a run directory the driver already wrote and writes a second tree
beside it, never running inside a rollout.

Anything a reader groups or filters by is a field rather than a directory, so a new grouping
costs no re-export. :class:`ViewerTree` names every path that gets written.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scenario_generation.closed_loop_eval import aggregate

# ``aggregate`` reduces these from row keys this path never writes, and a mean over no samples
# is 0.0 -- which reads as a measured total failure. Dropped, and named instead.
_UNMEASURED_SUMMARY_KEYS = ("mean_route_completion", "mean_gt_deviation_m")
_UNMEASURED_MARKER_KEY = "unmeasured_keys"

# Two producers disagree on the separator, so a case directory is matched against both.
_KEY_SEPARATORS = ("_", "__")

_SCENARIO_ID_RE = re.compile(
    r"(?<![0-9a-fA-F-])([0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})(?![0-9a-fA-F-])"
)

# The interpreter states the trigger and the unmet conditions in the failure message, so they
# are read back out of it rather than stored a second time.
_TRIGGER_RE = re.compile(r"\):\s*(.*?)(?:\nUnmet success conditions:|\Z)", re.S)
_UNMET_RE = re.compile(r'^\s*-\s*"(.+?)"\s*$', re.M)

# What a case is grouped under when its path carries no scenario id. A run made entirely of
# these is a suite the export cannot read, not an empty run.
_UNKNOWN_SCENARIO = "unknown_scenario"


@dataclass(frozen=True)
class ViewerTree:
    """Every path the export writes. Nothing else in this module builds one."""

    root: Path

    @property
    def run(self) -> Path:
        return self.root / "run.json"

    @property
    def scenarios(self) -> Path:
        return self.root / "scenarios.json"

    @property
    def cases(self) -> Path:
        return self.root / "cases.jsonl"


def sanitize(obj: Any) -> Any:
    """Recursively replace non-finite floats with ``None``.

    ``inf`` is in band here (no finite sample), but it serialises as ``Infinity``, which is
    outside the JSON spec and which ``JSON.parse`` rejects.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    return obj


def _dump_json(path: Path, obj: Any) -> None:
    """Write sanitized JSON that a browser can parse."""
    path.write_text(
        json.dumps(sanitize(obj), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_run_context(run_dir: Path) -> dict[str, str]:
    """The driver's ``key=value`` stamps. First wins; a missing file is not an error, since a
    run assembled by hand still exports."""
    ctx: dict[str, str] = {}
    path = run_dir / "run_context.txt"
    if not path.is_file():
        return ctx
    for match in re.finditer(r"(\w+)=(\S+)", path.read_text(encoding="utf-8", errors="replace")):
        ctx.setdefault(match.group(1), match.group(2))
    return ctx


def key_aliases(rel: str) -> tuple[str, ...]:
    """Every case-directory name ``rel`` could have been written as. One rel is one submitted
    case, so aliases recognise a directory and must never be counted as separate cases."""
    stem = rel[: -len(".xosc")] if rel.endswith(".xosc") else rel
    return tuple(dict.fromkeys(stem.replace("/", sep) for sep in _KEY_SEPARATORS))


def load_submitted(run_dir: Path) -> list[tuple[str, str]]:
    """``[(case key, scenario path)]`` for every case the run *submitted*.

    The only honest denominator: a case that died before writing ``row.json`` leaves no
    directory to count.
    """
    work_json = run_dir / "work.json"
    if work_json.is_file():
        try:
            pairs = json.loads(work_json.read_text(encoding="utf-8"))
        except ValueError:
            pairs = []
        # Resolved against the manifest's own directory, so either form of path works.
        return [(Path(out).name, str(work_json.parent / osc)) for out, osc in pairs]

    work_tsv = run_dir / "work.tsv"
    if not work_tsv.is_file():
        return []
    out = []
    for line in work_tsv.read_text(encoding="utf-8").splitlines():
        _, _, rel = line.partition("\t")
        rel = rel.strip()
        if rel:
            out.append((key_aliases(rel)[0], rel))
    return out


def load_case_rels(run_dir: Path) -> dict[str, str]:
    """``{case directory name: scenario path}`` for every alias a case could be under."""
    return {
        alias: rel
        for key, rel in load_submitted(run_dir)
        for alias in dict.fromkeys((key, *key_aliases(rel)))
    }


def scenario_id_of(rel: str | None, case_key: str) -> str:
    """The id the scenario is known by elsewhere, which is what lines runs up against it."""
    found = _SCENARIO_ID_RE.search(rel or case_key)
    return found.group(1) if found else _UNKNOWN_SCENARIO


def read_verdict(case_dir: Path) -> dict[str, Any]:
    """The scenario's own verdict on a case, or a statement that it never reached one.

    The interpreter writes ``result.junit.xml`` only when the storyboard resolves, so an absent
    file means the rollout hit the step limit first. The row's ``result_kind`` cannot say that:
    it is preset to a Timeout failure at configure time, so an undecided case still reads
    ``Failure`` there. Undecided is its own state.
    """
    path = case_dir / "osp_out" / "result.junit.xml"
    if not path.is_file():
        return {"decided": False}
    try:
        case = ET.parse(path).getroot().find(".//testcase")
    except (OSError, ET.ParseError) as exc:
        print(f"viewer_export: unreadable verdict for {case_dir.name}: {exc}", file=sys.stderr)
        return {"decided": False}
    if case is None:
        return {"decided": False}
    node, kind = case.find("failure"), "Failure"
    if node is None:
        node, kind = case.find("error"), "Error"
    if node is None:
        return {"decided": True, "kind": "Pass"}
    message = node.get("message") or ""
    trigger = _TRIGGER_RE.search(message)
    return {
        "decided": True,
        "kind": kind,
        "type": node.get("type"),
        # A configure-time error carries no triggering condition, only a message, so that
        # message stands in as the trigger.
        "trigger": trigger.group(1).strip() if trigger else (message.strip() or None),
        "unmet": _UNMET_RE.findall(message),
    }


def verdict_reason(case_dir: Path) -> str | None:
    """One line naming why a case reached no row, from the verdict it managed to write.

    Nothing else survives such a case: it has no row to carry a metric, and the raw run it
    failed in is not part of the export.
    """
    verdict = read_verdict(case_dir)
    if not verdict.get("decided"):
        return None
    parts = [verdict.get("type") or verdict.get("kind"), verdict.get("trigger")]
    return ": ".join(p for p in parts if p) or None


def _tally_verdicts(verdicts: list[dict[str, Any]]) -> dict[str, int]:
    """Three decided counts and the undecided one, so no consumer has to subtract."""
    out = {"pass": 0, "failure": 0, "error": 0, "undecided": 0}
    for verdict in verdicts:
        out[verdict["kind"].lower() if verdict.get("decided") else "undecided"] += 1
    return out


def collect_cases(run_dir: Path) -> tuple[dict[str, list[dict]], list[str]]:
    """Read every case that produced a row, keyed by scenario.

    Returns ``({scenario: [row, ...]}, [case_key of every submitted case with no row])``.
    """
    rels = load_case_rels(run_dir)
    submitted = load_submitted(run_dir)
    by_scenario: dict[str, list[dict]] = {}
    found: set[str] = set()
    for row_path in sorted(run_dir.glob("*/row.json")):
        case_dir = row_path.parent
        case_key = case_dir.name
        try:
            row = json.loads(row_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"viewer_export: unreadable row for {case_key}: {exc}", file=sys.stderr)
            continue
        row["case_key"] = case_key
        row.setdefault("route", case_key)
        row["_case_dir"] = str(case_dir)
        row["_rel"] = rels.get(case_key)
        found.add(case_key)
        scenario = scenario_id_of(rels.get(case_key), case_key)
        by_scenario.setdefault(scenario, []).append(row)
    # One entry per submitted case, so two spellings of one directory never count twice.
    missing = [
        key
        for key, rel in submitted
        if not any(a in found for a in dict.fromkeys((key, *key_aliases(rel))))
    ]
    return by_scenario, missing


def write_viewer_tree(
    out_root: Path,
    by_scenario: dict[str, list[dict]],
    *,
    meta: dict,
    scenario_errors: dict[str, str],
) -> dict[str, dict[str, int]]:
    """Write the export. Returns ``{scenario: {"rows": n}}``.

    Three files at the run root, so opening the listing costs three reads rather than two per
    scenario. See :class:`ViewerTree`.
    """
    # Laid over an earlier export the index would be replaced while its other contents stayed:
    # one run described, two contained.
    if out_root.exists() and any(out_root.iterdir()):
        raise SystemExit(f"viewer_export: {out_root} is not empty -- export into a new tree")
    out_root.mkdir(parents=True, exist_ok=True)
    tree = ViewerTree(out_root)
    counts: dict[str, dict[str, int]] = {}
    scenarios: dict[str, Any] = {}
    case_lines: list[str] = []

    for scenario, rows in sorted(by_scenario.items()):
        tally = {"rows": 0}

        near_miss = float(rows[0].get("object", {}).get("miss_thresh_m") or 1.0)
        strong_brake = float(rows[0].get("strong_brake", {}).get("thresh_mps2") or -2.5)

        clean_rows = []
        case_verdicts = []
        for row in rows:
            case_dir = Path(row.pop("_case_dir"))
            row.pop("_rel", None)
            row["scenario"] = scenario

            clean_rows.append(row)
            verdict = read_verdict(case_dir)
            case_verdicts.append(verdict)
            # Not in the aggregated rows: ``aggregate`` rolls a row up by key prefix and has
            # no business seeing a block it cannot reduce.
            case_lines.append(
                json.dumps(
                    sanitize({**row, "verdict": verdict}), ensure_ascii=False, allow_nan=False
                )
            )
            tally["rows"] += 1

        summary = aggregate(clean_rows, near_miss, strong_brake_mps2=strong_brake)
        unmeasured = [k for k in _UNMEASURED_SUMMARY_KEYS if summary.pop(k, None) is not None]
        # The rollup drops the row's ``measured`` flag, so carry it up: without it an
        # unobserved family aggregates to a zero that reads as "checked, found nothing".
        block = summary.get("red_light_violation")
        if isinstance(block, dict) and not any(
            r.get("red_light_violation", {}).get("measured", True) for r in clean_rows
        ):
            block["measured"] = False
            unmeasured.append("red_light_violation")

        scenarios[scenario] = {
            "n_cases": len(clean_rows),
            "verdicts": _tally_verdicts(case_verdicts),
            "error": scenario_errors.get(scenario),
            _UNMEASURED_MARKER_KEY: unmeasured + ["reproducer"],
            "summary": sanitize(summary),
        }
        counts[scenario] = tally

    # A scenario whose every case failed writes no row, so without an entry of its own a run
    # that lost cases would read as complete.
    for scenario, message in scenario_errors.items():
        if scenario in scenarios:
            continue
        scenarios[scenario] = {
            "n_cases": 0,
            "verdicts": {"pass": 0, "failure": 0, "error": 0, "undecided": 0},
            "error": message,
            _UNMEASURED_MARKER_KEY: [],
            "summary": None,
        }

    meta["verdicts"] = {
        key: sum(e["verdicts"][key] for e in scenarios.values())
        for key in ("pass", "failure", "error", "undecided")
    }

    tree.cases.write_text("\n".join(case_lines) + "\n", encoding="utf-8")
    _dump_json(tree.scenarios, scenarios)
    _dump_json(tree.run, meta)
    return counts


def export(run_dir: Path, out_root: Path) -> dict[str, dict[str, int]]:
    """Export one scenario_sim run directory into ``out_root``."""
    ctx = parse_run_context(run_dir)
    by_scenario, missing = collect_cases(run_dir)
    if not by_scenario:
        raise SystemExit(f"viewer_export: no */row.json under {run_dir}")
    if set(by_scenario) == {_UNKNOWN_SCENARIO}:
        sample = next(iter(next(iter(by_scenario.values()))), {}).get("route")
        raise SystemExit(f"viewer_export: resolved nothing for any case (sample case: {sample})")

    rels = load_case_rels(run_dir)
    submitted = load_submitted(run_dir)

    # A case with no row is still attributable, because the submitted list carries its path.
    scenario_errors: dict[str, str] = {}
    missing_rows: list[dict[str, Any]] = []
    if missing:
        per_scenario: dict[str, list[str]] = {}
        for key in missing:
            per_scenario.setdefault(scenario_id_of(rels.get(key), key), []).append(key)
        scenario_errors = {
            s: f"{len(k)} of this scenario's case(s) produced no row: {', '.join(k[:5])}"
            for s, k in per_scenario.items()
        }
        # A key alone says a case vanished; the verdict it managed to write says why.
        missing_rows = [
            {"case_key": key, "reason": verdict_reason(run_dir / key)} for key in missing
        ]

    meta = {
        "run_dir": str(run_dir),
        "draw_every": ctx.get("draw_every"),
        "scenario_root": ctx.get("scenario_root"),
        "ckpt": ctx.get("ckpt"),
        "dp_commit": ctx.get("dp_commit"),
        "branch": ctx.get("branch"),
        "max_steps": ctx.get("max_steps"),
        "submitted_cases": len(submitted),
        "missing_rows": missing_rows,
    }

    counts = write_viewer_tree(
        out_root,
        by_scenario,
        meta=meta,
        scenario_errors=scenario_errors,
    )
    # Submitted is the only honest denominator, so it is what the run's log states.
    exported = sum(t["rows"] for t in counts.values())
    print(
        f"viewer_export: {exported} row(s) from {len(submitted) or 'an unknown number of'} "
        f"submitted case(s), {len(missing)} missing"
    )
    if missing:
        print("  missing: " + ", ".join(missing[:20]) + ("  ..." if len(missing) > 20 else ""))
    return counts


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run_dir", required=True, type=Path, help="a finished suite run directory")
    p.add_argument("--out_root", required=True, type=Path, help="viewer tree to write")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = _parse_args(argv)
    export(a.run_dir, a.out_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
