"""CLI: python -m rlvr.posttrain <command> [args]

    objectives                 registered candidate-weighting objectives
    presets                    bundled spec files
    show    SPEC               resolved spec(s) after inheritance and sweep
    plan    SPEC               the exact command line(s), validated, nothing launched
    run     SPEC [--now]       launch, waiting for free GPUs unless --now
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PRESETS = Path(__file__).parent / "presets"


def _specs(path: str):
    from rlvr.posttrain.spec import load

    candidate = Path(path)
    if not candidate.exists():
        for guess in (PRESETS / path, PRESETS / f"{path}.json"):
            if guess.exists():
                candidate = guess
                break
    return load(candidate)


def main(argv: list[str] | None = None) -> int:
    # A supervised queue is read while it is still running, and a block-buffered
    # log only arrives once there is nothing left to watch.
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(prog="rlvr.posttrain", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("objectives")
    sub.add_parser("presets")
    for name in ("show", "plan"):
        item = sub.add_parser(name)
        item.add_argument("spec")
    launch = sub.add_parser("run")
    launch.add_argument("spec")
    launch.add_argument("--now", action="store_true", help="skip the free-GPU wait")
    launch.add_argument("--detach", action="store_true")
    launch.add_argument(
        "--supervise",
        action="store_true",
        help="relaunch, resuming itself, until the arm reaches its target epoch",
    )
    launch.add_argument(
        "--after",
        help="wait for this path, which may be a glob, before starting "
        "(an earlier arm's artifact, under a stamp it picks itself)",
    )
    launch.add_argument("--log_root", type=Path, default=Path("outputs/posttrain"))
    args = parser.parse_args(argv)

    if args.command == "objectives":
        from rlvr.posttrain.objectives import available

        print("\n".join(available()))
        return 0
    if args.command == "presets":
        print("\n".join(sorted(p.name for p in PRESETS.glob("*.json"))))
        return 0

    specs = _specs(args.spec)
    if args.command == "show":
        print(json.dumps([s.to_dict() for s in specs], indent=2))
        return 0
    if args.command == "plan":
        from rlvr.posttrain.flags import command
        from rlvr.posttrain.preflight import report

        ok = True
        for spec in specs:
            ok &= report(spec)
        for spec in specs:
            print(f"# {spec.name}")
            env = " ".join(f"{k}={v}" for k, v in spec.env.items())
            try:
                argv = command(spec)
            except ValueError as error:  # an invalid spec is a report, not a crash
                print(f"{spec.name}: {error}")
                ok = False
                continue
            print((env + " " if env else "") + " ".join(argv))
        return 0 if ok else 78

    from rlvr.posttrain.runner import run_queue, wait_for_path

    if args.after is not None:
        wait_for_path(args.after)

    results = run_queue(
        specs,
        log_root=args.log_root,
        wait=not args.now,
        detach=args.detach,
        supervise=args.supervise,
    )
    return 0 if all(code == 0 for code in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
