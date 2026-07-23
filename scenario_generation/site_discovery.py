"""Discover per-site npz roots under a shared ``{project}/`` directory.

A "site" is a direct subdirectory of ``sites_root`` following the existing
``{project}/{area_map_id}_{area_map_name}/{split}/...`` layout convention (e.g.
``x2_dev/2231_odaiba_shinagawa_copied_from_xx1``). Each site's split subtree is
evaluated independently — callers must never merge multiple sites' npz under
one ``--npz_root``/``rglob``, since route grouping is filename-only and
ignores directory structure (see route_timeline.route_prefix); mixing sites
risks silently merging unrelated routes that happen to share a bag-name
prefix.

``split`` is ``valid`` for the label-based downloader's original layout, or
``manual``/``auto`` for the PR253 ros_scripts reorg (train/valid -> manual,
auto stays auto) — both conventions coexist across existing datasets, so all
three are checked.
"""

from __future__ import annotations

from pathlib import Path

_SPLIT_DIR_NAMES = ("valid", "manual", "auto")


def discover_sites(sites_root: str | Path) -> dict[str, Path]:
    """Map site_name -> that site's split npz root (``valid/`` or ``manual/``/``auto/``).

    Skips subdirectories without a recognized split dir or with no npz under it.
    """
    root = Path(sites_root)
    sites: dict[str, Path] = {}
    if not root.is_dir():
        return sites
    for site_dir in sorted(root.iterdir()):
        if not site_dir.is_dir():
            continue
        for split_name in _SPLIT_DIR_NAMES:
            split_dir = site_dir / split_name
            if not split_dir.is_dir():
                continue
            if not any(split_dir.rglob("*.npz")):
                continue
            sites[site_dir.name] = split_dir
            break
    return sites
