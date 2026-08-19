import json
import tempfile
from pathlib import Path
import pytest

from scenario_generation.catalog_registry import (
    find_catalog_dir,
    find_suite_dir,
    load_manifest,
    resolve_scenario_roots,
)


def test_catalog_registry_resolution():
    with tempfile.TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        suites_dir = base / "suites"
        catalogs_dir = base / "catalogs"

        # Create mock suite 1
        s1 = suites_dir / "suite-1-uuid"
        (s1 / "scenarios" / "sc1" / "1").mkdir(parents=True)
        (s1 / "scenarios" / "sc1" / "1" / "scenario_0.xosc").touch()
        with open(s1 / "manifest.json", "w") as f:
            json.dump({
                "suite_id": "suite-1-uuid",
                "suite_name": "SuiteOne",
                "type": "planning_sim_v2",
                "total_cases": 1,
            }, f)

        # Create mock suite 2
        s2 = suites_dir / "suite-2-uuid"
        (s2 / "scenarios" / "sc2" / "1").mkdir(parents=True)
        (s2 / "scenarios" / "sc2" / "1" / "scenario_0.xosc").touch()
        with open(s2 / "manifest.json", "w") as f:
            json.dump({
                "suite_id": "suite-2-uuid",
                "suite_name": "SuiteTwo",
                "type": "planning_sim_v2",
                "total_cases": 1,
            }, f)

        # Create mock catalog
        cat = catalogs_dir / "catalog-1-uuid"
        cat.mkdir(parents=True)
        with open(cat / "manifest.json", "w") as f:
            json.dump({
                "catalog_id": "catalog-1-uuid",
                "catalog_name": "MyCatalog",
                "suites": {
                    "suite-1-uuid": {"suite_name": "SuiteOne", "supported": True},
                    "suite-2-uuid": {"suite_name": "SuiteTwo", "supported": True},
                }
            }, f)

        # 1. Test finding suite by ID and by Name
        assert find_suite_dir("suite-1-uuid", base) == s1
        assert find_suite_dir("SuiteOne", base) == s1
        assert find_suite_dir("SuiteTwo", base) == s2

        # 2. Test finding catalog by ID and by Name
        assert find_catalog_dir("catalog-1-uuid", base) == cat
        assert find_catalog_dir("MyCatalog", base) == cat

        # 3. Test resolving scenario roots for a specific suite
        res_s1 = resolve_scenario_roots(suite="SuiteOne", base_root=base)
        assert len(res_s1) == 1
        assert res_s1[0][1] == "SuiteOne"
        assert res_s1[0][2] == s1 / "scenarios"

        # 4. Test resolving scenario roots for a catalog
        res_cat = resolve_scenario_roots(catalog="MyCatalog", base_root=base)
        assert len(res_cat) == 2
        names = [r[1] for r in res_cat]
        assert "SuiteOne" in names
        assert "SuiteTwo" in names

        # 5. Test direct scenario root
        direct_res = resolve_scenario_roots(scenario_root=s1 / "scenarios", base_root=base)
        assert len(direct_res) == 1
        assert direct_res[0][2] == s1 / "scenarios"
