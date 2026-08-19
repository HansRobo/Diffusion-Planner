"""Diagnostics, root-cause categorization, and hazard engagement for scenario_sim rollouts.

Standardizes the failure taxonomy and model quality analysis established in plan/13:
- Exclusive failure classification:
    1. PASS: Storyboard completed with exitSuccess / pass verdict.
    2. COLLISION: Ego collided with an obstacle (pedestrian, NPC, bicycle, etc.).
    3. ROAD_DEPARTURE: Ego crossed road border / boundary without colliding with an obstacle.
    4. FROZEN_STANDSTILL: Ego failed to pull away (max speed < 0.5 m/s) or was cut off by StandStill.
    5. PROXIMITY_DEPARTURE: Early termination from proximity limit conditions (e.g. act_lateral_check).
    6. GOAL_STOP_FAILURE: Ego reached goal proximity (<5m) but failed to stop / overshot.
    7. SPEED_TIMEOUT: Ego moved but failed to reach goal within the simulation deadline (low speed).
    8. UNMET_CONDITION: Other OpenSCENARIO story conditions were not met.
    9. SCENARIO_REFUSED: OpenSCENARIO interpreter refused to parse/configure scenario.
    10. ERROR: Worker/simulator crashed or runtime error.

- Hazard Engagement classification:
    - Detects whether the critical test interaction (hazard) actually presented to the ego,
      distinguishing a 'True Pass (Hazard Avoided)' from a 'Trivial Pass (Hazard Unengaged)'.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

_TRIGGER_RE = re.compile(r'triggered by the Condition named "([^"]+)"')
_UNMET_RE = re.compile(r'- "([^"]+)"')
_ANONYMOUS_RE = re.compile(r'\):\s*(.{1,120})', re.DOTALL)


def parse_junit_verdict(junit_xml_path_or_content: str | Path) -> dict[str, Any]:
    """Parse an OpenSCENARIO interpreter result.junit.xml file or string.

    Returns dict with keys:
        - decided (bool): Whether a verdict was recorded.
        - kind (str): 'Pass' | 'Failure' | 'Error' | 'Undecided'
        - type (str | None): Failure/Error type attribute.
        - trigger (str | None): Condition name or trigger message.
        - unmet (list[str]): List of unmet success condition names.
        - message (str): Raw message string.
    """
    if isinstance(junit_xml_path_or_content, Path) or (
        isinstance(junit_xml_path_or_content, str) and not junit_xml_path_or_content.startswith("<")
    ):
        path = Path(junit_xml_path_or_content)
        if not path.is_file():
            return {
                "decided": False,
                "kind": "Undecided",
                "type": None,
                "trigger": None,
                "unmet": [],
                "message": "",
            }
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return {
                "decided": False,
                "kind": "Undecided",
                "type": None,
                "trigger": None,
                "unmet": [],
                "message": "",
            }
    else:
        content = str(junit_xml_path_or_content)

    content_clean = content.replace("&#10;", "\n").replace("&quot;", '"')
    try:
        root = ET.fromstring(content)
        testcase = root.find(".//testcase")
    except ET.ParseError:
        # Fallback text parsing if XML is malformed
        is_fail = "<failure" in content_clean
        is_err = "<error" in content_clean
        is_pass = "<testcase" in content_clean and not (is_fail or is_err)
        kind = "Failure" if is_fail else ("Error" if is_err else ("Pass" if is_pass else "Undecided"))
        trig_match = _TRIGGER_RE.search(content_clean)
        unmet = _UNMET_RE.findall(content_clean)
        return {
            "decided": kind in ("Pass", "Failure", "Error"),
            "kind": kind,
            "type": None,
            "trigger": trig_match.group(1).strip() if trig_match else None,
            "unmet": unmet,
            "message": content_clean,
        }

    if testcase is None:
        return {
            "decided": False,
            "kind": "Undecided",
            "type": None,
            "trigger": None,
            "unmet": [],
            "message": "",
        }

    node = testcase.find("failure")
    kind = "Failure"
    if node is None:
        node = testcase.find("error")
        kind = "Error"
    if node is None:
        return {
            "decided": True,
            "kind": "Pass",
            "type": None,
            "trigger": None,
            "unmet": [],
            "message": "",
        }

    message = (node.get("message") or "").replace("&#10;", "\n").replace("&quot;", '"')
    trigger_match = _TRIGGER_RE.search(message)
    if trigger_match:
        trigger = trigger_match.group(1).strip()
    else:
        anon_match = _ANONYMOUS_RE.search(message)
        if anon_match:
            trigger = f"anonymous: {anon_match.group(1).strip().replace(chr(10), ' ')}"
        else:
            trigger = message.strip() or None

    return {
        "decided": True,
        "kind": kind,
        "type": node.get("type"),
        "trigger": trigger,
        "unmet": _UNMET_RE.findall(message),
        "message": message,
    }


def is_passed(row: dict[str, Any], verdict: dict[str, Any] | None = None) -> bool:
    """Determine whether a scenario rollout passed."""
    if verdict and verdict.get("decided") and verdict.get("kind") == "Pass":
        return True
    rk = str(row.get("result_kind", "")).lower()
    term = str(row.get("terminated", "")).lower()
    vk = str(row.get("verdict_kind", "")).lower()
    if vk in ("pass", "passed", "success", "exitsuccess"):
        return True
    if rk in ("pass", "passed", "success", "exitsuccess"):
        return True
    if term in ("pass", "passed", "success", "exitsuccess"):
        return True
    return False


def classify_failure(row: dict[str, Any], verdict: dict[str, Any] | None = None) -> str:
    """Classify the root cause of a scenario result in hierarchical order.

    Hierarchy:
    1. PASS
    2. SCENARIO_REFUSED
    3. ERROR
    4. COLLISION
    5. ROAD_DEPARTURE
    6. FROZEN_STANDSTILL
    7. PROXIMITY_DEPARTURE
    8. GOAL_STOP_FAILURE
    9. SPEED_TIMEOUT
    10. UNMET_CONDITION
    """
    if is_passed(row, verdict):
        return "PASS"

    term = str(row.get("terminated", "")).lower()
    if term in ("worker_failed", "scenario_rejected") or row.get("error"):
        if "rejected" in str(row.get("error", "")).lower():
            return "SCENARIO_REFUSED"
        return "ERROR"

    # Check object collision
    obj_block = row.get("object") or {}
    coll_count = obj_block.get("collision_count", 0)
    coll_steps = obj_block.get("collision_steps", 0)
    if coll_count > 0 or coll_steps > 0:
        return "COLLISION"

    # Check road border collision / departure (when no object collision)
    rb_block = row.get("road_border") or {}
    rb_coll_count = rb_block.get("collision_count", 0)
    rb_coll_steps = rb_block.get("collision_steps", 0)
    if rb_coll_count > 0 or rb_coll_steps > 0:
        return "ROAD_DEPARTURE"

    vmax = row.get("max_speed_mps")
    trigger = (verdict.get("trigger") or "") if verdict else ""
    trigger_lower = trigger.lower()

    # Check standstill / freezing
    if (vmax is not None and vmax < 0.5) or "standstill" in trigger_lower or "stand_still" in trigger_lower:
        return "FROZEN_STANDSTILL"

    # Check proximity cutoff conditions (act_lateral_check, act_longitudinal_check, RelativeDistance)
    if any(k in trigger_lower for k in ("lateral_check", "longitudinal_check", "relativedistance", "obstacle_distance")):
        return "PROXIMITY_DEPARTURE"

    # Check goal stop failure: ego reached near goal, but failed to stop within tolerance
    at_goal = row.get("scored_stopped_at_goal") or False
    unmet = verdict.get("unmet", []) if verdict else []
    if at_goal or "goal_position" in unmet:
        # Reached near goal or goal_position is the blocker
        if "goal_position" in unmet:
            return "GOAL_STOP_FAILURE"

    # Check speed / time limit timeout
    if term in ("max_steps", "sim_terminated") or "simulationtime" in trigger_lower or "timeout" in trigger_lower:
        return "SPEED_TIMEOUT"

    if unmet or trigger:
        return "UNMET_CONDITION"

    return "UNKNOWN_FAILURE"


def classify_hazard_engagement(row: dict[str, Any], ego_speed_param: float | None = None) -> dict[str, Any]:
    """Detect whether the scenario's critical hazard was actively engaged.

    Returns dict with:
        - hazard_engaged (bool)
        - hazard_reason (str)
        - hazard_verdict (str): 'PASS_AVOIDED' | 'PASS_UNENGAGED' | 'FAIL_COLLISION' | 'FAIL_MOBILITY' | 'FAIL_OTHER'
    """
    passed = is_passed(row)
    category = classify_failure(row)

    obj_block = row.get("object") or {}
    cl_min = obj_block.get("clearance_min_m")
    if cl_min is None:
        cl_min = float("inf")
    vmax = row.get("max_speed_mps")
    if vmax is None:
        vmax = 0.0

    # An obstacle came within 20m, collision occurred, or vehicle attained specified speed
    engaged = False
    reason = "no_interaction"

    if obj_block.get("collision_count", 0) > 0 or obj_block.get("collision_steps", 0) > 0:
        engaged = True
        reason = "collision"
    elif cl_min < 20.0:
        engaged = True
        reason = f"proximity_clearance_{cl_min:.1f}m"
    elif ego_speed_param is not None and ego_speed_param > 0:
        if vmax >= 0.90 * ego_speed_param:
            engaged = True
            reason = f"speed_reached_{vmax:.1f}mps"
        else:
            engaged = False
            reason = f"low_speed_gate_missed_{vmax:.1f}mps_vs_{ego_speed_param:.1f}mps"
    elif vmax >= 3.0:
        # Generic heuristic: if vehicle drove at moderate speed, hazard likely triggered
        engaged = True
        reason = f"driving_speed_{vmax:.1f}mps"

    if passed:
        hazard_verdict = "PASS_AVOIDED" if engaged else "PASS_UNENGAGED"
    elif category == "COLLISION":
        hazard_verdict = "FAIL_COLLISION"
    elif category in ("FROZEN_STANDSTILL", "SPEED_TIMEOUT"):
        hazard_verdict = "FAIL_MOBILITY"
    else:
        hazard_verdict = "FAIL_OTHER"

    return {
        "hazard_engaged": engaged,
        "hazard_reason": reason,
        "hazard_verdict": hazard_verdict,
    }


def diagnose_case(row: dict[str, Any], case_dir: Path | None = None) -> dict[str, Any]:
    """Produce comprehensive diagnostics for a scenario row."""
    verdict = None
    if case_dir:
        junit_path = case_dir / "osp_out" / "result.junit.xml"
        if junit_path.is_file():
            verdict = parse_junit_verdict(junit_path)

    if verdict is None and isinstance(row.get("verdict"), dict):
        verdict = row["verdict"]

    category = classify_failure(row, verdict)
    hazard = classify_hazard_engagement(row)
    passed = is_passed(row, verdict)

    diag = {
        "passed": passed,
        "failure_category": category,
        **hazard,
    }
    if verdict and verdict.get("decided"):
        diag["verdict_kind"] = verdict.get("kind")
        diag["verdict_trigger"] = verdict.get("trigger")
        diag["verdict_unmet"] = verdict.get("unmet", [])
    return diag
