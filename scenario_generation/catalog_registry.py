"""Registry and path resolver for WebAuto Vehicle Catalogs and Scenario Suites.

Provides helpers to resolve catalog/suite names or IDs to on-disk scenario roots,
read manifests, and enumerate test suites for evaluation runs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

DEFAULT_SCENARIO_BASE = Path(
    os.getenv("OPENSCENARIOS_BASE", "/mnt/storage_rdma/diffusion_planner/openscenarios")
)


def load_manifest(manifest_path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON manifest from a suite or catalog directory."""
    p = Path(manifest_path)
    if p.is_dir():
        p = p / "manifest.json"
    if not p.is_file():
        raise FileNotFoundError(f"Manifest not found at {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def find_suite_dir(
    suite_query: str,
    base_root: Optional[Union[str, Path]] = None,
) -> Path:
    """Find suite directory by exact ID, display name, or direct path."""
    base = Path(base_root or DEFAULT_SCENARIO_BASE)
    
    # 1. Direct path check
    direct = Path(suite_query)
    if direct.is_dir() and (direct / "scenarios").is_dir():
        return direct
    if direct.is_dir() and (direct / "manifest.json").is_file():
        return direct

    # 2. Check under base/suites/<suite_query>
    suites_dir = base / "suites"
    if suites_dir.is_dir():
        cand = suites_dir / suite_query
        if cand.is_dir():
            return cand

        # Search by display name in manifest.json
        for sdir in suites_dir.iterdir():
            if sdir.is_dir():
                mf = sdir / "manifest.json"
                if mf.is_file():
                    try:
                        with open(mf, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if data.get("suite_name") == suite_query or data.get("suite_id") == suite_query:
                                return sdir
                    except Exception:
                        pass

    raise FileNotFoundError(f"Could not find suite '{suite_query}' in {base}")


def find_catalog_dir(
    catalog_query: str,
    base_root: Optional[Union[str, Path]] = None,
) -> Path:
    """Find catalog directory by exact ID, display name, or direct path."""
    base = Path(base_root or DEFAULT_SCENARIO_BASE)

    # 1. Direct path check
    direct = Path(catalog_query)
    if direct.is_dir() and (direct / "manifest.json").is_file():
        return direct

    # 2. Check under base/catalogs/<catalog_query>
    catalogs_dir = base / "catalogs"
    if catalogs_dir.is_dir():
        cand = catalogs_dir / catalog_query
        if cand.is_dir():
            return cand

        # Search by display name in manifest.json
        for cdir in catalogs_dir.iterdir():
            if cdir.is_dir():
                mf = cdir / "manifest.json"
                if mf.is_file():
                    try:
                        with open(mf, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if data.get("catalog_name") == catalog_query or data.get("catalog_id") == catalog_query:
                                return cdir
                    except Exception:
                        pass

    raise FileNotFoundError(f"Could not find catalog '{catalog_query}' in {base}")


def resolve_scenario_roots(
    suite: Optional[str] = None,
    catalog: Optional[str] = None,
    scenario_root: Optional[Union[str, Path]] = None,
    base_root: Optional[Union[str, Path]] = None,
) -> List[Tuple[str, str, Path]]:
    """Resolve targets into list of (catalog_id_or_none, suite_name_or_id, scenario_root_path)."""
    base = Path(base_root or DEFAULT_SCENARIO_BASE)

    # If suite explicitly given
    if suite:
        sdir = find_suite_dir(suite, base)
        sname = sdir.name
        mf_path = sdir / "manifest.json"
        if mf_path.is_file():
            try:
                data = load_manifest(mf_path)
                sname = data.get("suite_name", sdir.name)
            except Exception:
                pass
        scenarios_path = sdir / "scenarios" if (sdir / "scenarios").is_dir() else sdir
        return [(None, sname, scenarios_path)]

    # If catalog given
    if catalog:
        cdir = find_catalog_dir(catalog, base)
        c_manifest = load_manifest(cdir / "manifest.json")
        cid = c_manifest.get("catalog_id", cdir.name)
        suites_dict = c_manifest.get("suites", {})
        results = []
        for sid, sinfo in suites_dict.items():
            if not sinfo.get("supported", True):
                continue
            sname = sinfo.get("suite_name", sid)
            try:
                sdir = find_suite_dir(sid, base)
                scenarios_path = sdir / "scenarios" if (sdir / "scenarios").is_dir() else sdir
                results.append((cid, sname, scenarios_path))
            except FileNotFoundError:
                continue
        if results:
            return results

    # If direct scenario_root given
    if scenario_root:
        p = Path(scenario_root)
        return [(None, p.parent.name if p.name == "scenarios" else p.name, p)]

    # Default legacy fallback
    legacy_path = base / "scenarios"
    return [(None, "default_suite", legacy_path if legacy_path.is_dir() else base)]
