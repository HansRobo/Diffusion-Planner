#!/usr/bin/env python3
"""Check that ONNX export is identical between the current branch and tier4-main.

The v5.0 checkpoint (``diffusion_planner.pth`` / ``diffusion_planner.param.json``)
is downloaded into ``test_scripts/assets`` if it is not there yet, exported to ONNX
on the current branch, exported again on ``tier4-main``, and the resulting ONNX
files are compared byte by byte.
The original branch is checked out again before the script exits, even on failure.

Example:
    python3 test_scripts/test_onnx_export_consistency.py
"""

import filecmp
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

TEST_SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_SCRIPTS_DIR.parent
ASSETS_DIR = TEST_SCRIPTS_DIR / "assets"
TORCH2ONNX = REPO_ROOT / "ros_scripts" / "torch2onnx.py"

BASE_REF = "tier4-main"

MODEL_BASE_URL = "https://awf.ml.dev.web.auto/planning/models/diffusion_planner/v5.0"
CKPT_NAME = "diffusion_planner.pth"
PARAM_NAME = "diffusion_planner.param.json"

OUTPUT_PREFIX = "diffusion_planner"
ONNX_NAMES = [
    f"{OUTPUT_PREFIX}.onnx",
    f"{OUTPUT_PREFIX}_encoder.onnx",
    f"{OUTPUT_PREFIX}_decoder.onnx",
    f"{OUTPUT_PREFIX}_turn_indicator.onnx",
]

HEAD_EXPORT_DIR = ASSETS_DIR / "export_head"
BASE_EXPORT_DIR = ASSETS_DIR / "export_base"


def run_git(git_args: list[str]) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *git_args],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def resolve_head_ref() -> str:
    """Return the current branch name, or the commit hash when HEAD is detached."""
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "symbolic-ref", "--quiet", "--short", "HEAD"],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return run_git(["rev-parse", "HEAD"])


def assert_no_tracked_changes() -> None:
    """Abort unless every tracked file matches HEAD.

    Uncommitted edits would survive the checkout and silently leak into the
    tier4-main export, which would make the comparison meaningless.
    """
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "diff", "--quiet", "HEAD"],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        changed = run_git(["diff", "--name-only", "HEAD"])
        raise RuntimeError(
            "The working tree has uncommitted changes to tracked files:\n"
            f"{changed}\n"
            "Commit or stash them before running this test."
        )


def ensure_assets_dir() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    gitignore_path = ASSETS_DIR / ".gitignore"
    if not gitignore_path.exists():
        gitignore_path.write_text("*\n")


def download_if_absent(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"Already downloaded: {dest} ({dest.stat().st_size} bytes)")
        return
    print(f"Downloading {url}\n         -> {dest}")
    tmp_path = dest.with_name(dest.name + ".partial")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as f:
        shutil.copyfileobj(response, f)
    tmp_path.rename(dest)
    print(f"Downloaded: {dest} ({dest.stat().st_size} bytes)")


def prepare_export_dir(export_dir: Path) -> None:
    """Set up a fresh directory that torch2onnx.py accepts (``*.pth`` + ``args.json``)."""
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)
    # Symlink the 200MB+ checkpoint instead of copying it for every export.
    (export_dir / CKPT_NAME).symlink_to(ASSETS_DIR / CKPT_NAME)
    shutil.copyfile(ASSETS_DIR / PARAM_NAME, export_dir / "args.json")


def export_onnx(export_dir: Path) -> None:
    """Run torch2onnx.py with the code of the currently checked out ref."""
    # Keep this script's own logs interleaved correctly with the subprocess output.
    sys.stdout.flush()
    env = dict(os.environ)
    # `uv run` re-resolves the environment for the checked out ref, so an inherited
    # VIRTUAL_ENV from the caller must not override it.
    env.pop("VIRTUAL_ENV", None)
    subprocess.run(
        ["uv", "run", "python", str(TORCH2ONNX), str(export_dir)],
        check=True,
        cwd=REPO_ROOT,
        env=env,
    )


def collect_mismatches() -> list[str]:
    mismatches = []
    for name in ONNX_NAMES:
        head_onnx = HEAD_EXPORT_DIR / name
        base_onnx = BASE_EXPORT_DIR / name
        if not head_onnx.exists():
            raise RuntimeError(f"Export did not produce {head_onnx}")
        if not base_onnx.exists():
            raise RuntimeError(f"Export did not produce {base_onnx}")
        if filecmp.cmp(head_onnx, base_onnx, shallow=False):
            print(f"OK   {name} ({head_onnx.stat().st_size} bytes)")
        else:
            print(f"DIFF {name} ({head_onnx.stat().st_size} vs {base_onnx.stat().st_size} bytes)")
            mismatches.append(name)
    return mismatches


def main() -> int:
    ensure_assets_dir()
    download_if_absent(f"{MODEL_BASE_URL}/{PARAM_NAME}", ASSETS_DIR / PARAM_NAME)
    download_if_absent(f"{MODEL_BASE_URL}/{CKPT_NAME}", ASSETS_DIR / CKPT_NAME)

    assert_no_tracked_changes()
    head_ref = resolve_head_ref()
    head_commit = run_git(["rev-parse", "HEAD"])
    base_commit = run_git(["rev-parse", "--verify", f"{BASE_REF}^{{commit}}"])
    print(f"\nhead: {head_ref} ({head_commit})")
    print(f"base: {BASE_REF} ({base_commit})")
    if head_commit == base_commit:
        print("Warning: head and base point at the same commit, the comparison is trivial")

    print(f"\n{'#' * 80}\nExporting ONNX on {head_ref}\n{'#' * 80}")
    prepare_export_dir(HEAD_EXPORT_DIR)
    export_onnx(HEAD_EXPORT_DIR)

    print(f"\n{'#' * 80}\nExporting ONNX on {BASE_REF}\n{'#' * 80}")
    prepare_export_dir(BASE_EXPORT_DIR)
    run_git(["checkout", BASE_REF])
    try:
        export_onnx(BASE_EXPORT_DIR)
    finally:
        run_git(["checkout", head_ref])
        restored = run_git(["rev-parse", "HEAD"])
        print(f"\nRestored {head_ref} ({restored})")
        if restored != head_commit:
            raise RuntimeError(f"Failed to restore {head_ref}: HEAD is {restored}")

    print(f"\n{'#' * 80}\nComparing ONNX files\n{'#' * 80}")
    mismatches = collect_mismatches()
    if mismatches:
        print(f"\nFAILED: {len(mismatches)} file(s) differ between {head_ref} and {BASE_REF}")
        print(f"  head: {HEAD_EXPORT_DIR}")
        print(f"  base: {BASE_EXPORT_DIR}")
        return 1
    print(f"\nPASSED: all {len(ONNX_NAMES)} ONNX files are identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
