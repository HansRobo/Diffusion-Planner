import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from scenario_generation.scenario_sim_diagnostics import (
    classify_failure,
    classify_hazard_engagement,
    diagnose_case,
    is_passed,
    parse_junit_verdict,
)

_JUNIT_PASS = """<?xml version="1.0"?>
<testsuites failures="0" errors="0" tests="1">
  <testsuite name="6" failures="0" errors="0" tests="1">
    <testcase name="scenario_0" />
  </testsuite>
</testsuites>
"""

_JUNIT_FAIL_GOAL = """<?xml version="1.0"?>
<testsuites failures="1" errors="0" tests="1">
  <testsuite name="12" failures="1" errors="0" tests="1">
    <testcase name="scenario_1">
      <failure type="SimulationFailure" message="CustomCommandAction typed &quot;exitFailure&quot; was triggered by the anonymous Condition (OpenSCENARIO.Storyboard.Story[0].Act[&quot;_EndCondition&quot;].ManeuverGroup[0].Maneuver[0].Event[1].StartTrigger.ConditionGroup[0].Condition[0]): Is the simulation time (= 180.1) is greaterThan 180.0?&#10;Unmet success conditions:&#10;  - &lt;anonymous>&#10;  - &quot;goal_position&quot;" />
    </testcase>
  </testsuite>
</testsuites>
"""

_JUNIT_FAIL_TRIGGER = """<?xml version="1.0"?>
<testsuites failures="1" errors="0" tests="1">
  <testsuite name="5" failures="1" errors="0" tests="1">
    <testcase name="scenario_0">
      <failure type="SimulationFailure" message="CustomCommandAction typed &quot;exitFailure&quot; was triggered by the Condition named &quot;act_lateral_check&quot;: lateral distance exceeded limit." />
    </testcase>
  </testsuite>
</testsuites>
"""


def test_parse_junit_verdict_pass():
    res = parse_junit_verdict(_JUNIT_PASS)
    assert res["decided"] is True
    assert res["kind"] == "Pass"
    assert res["trigger"] is None
    assert res["unmet"] == []


def test_parse_junit_verdict_fail_unmet():
    res = parse_junit_verdict(_JUNIT_FAIL_GOAL)
    assert res["decided"] is True
    assert res["kind"] == "Failure"
    assert "goal_position" in res["unmet"]
    assert "anonymous" in res["trigger"]


def test_parse_junit_verdict_fail_trigger():
    res = parse_junit_verdict(_JUNIT_FAIL_TRIGGER)
    assert res["decided"] is True
    assert res["kind"] == "Failure"
    assert res["trigger"] == "act_lateral_check"


def test_classify_failure_hierarchy():
    # Pass
    assert classify_failure({"result_kind": "Pass"}) == "PASS"

    # Collision takes precedence
    row_coll = {
        "result_kind": "Failure",
        "object": {"collision_count": 1},
        "road_border": {"collision_count": 1},
        "max_speed_mps": 0.1,
    }
    assert classify_failure(row_coll) == "COLLISION"

    # Road departure
    row_rb = {
        "result_kind": "Failure",
        "object": {"collision_count": 0},
        "road_border": {"collision_count": 1},
        "max_speed_mps": 5.0,
    }
    assert classify_failure(row_rb) == "ROAD_DEPARTURE"

    # Frozen
    row_frozen = {
        "result_kind": "Failure",
        "object": {"collision_count": 0},
        "road_border": {"collision_count": 0},
        "max_speed_mps": 0.2,
    }
    assert classify_failure(row_frozen) == "FROZEN_STANDSTILL"

    # Proximity departure
    verdict_prox = {"decided": True, "kind": "Failure", "trigger": "act_lateral_check", "unmet": []}
    row_norm = {
        "result_kind": "Failure",
        "object": {"collision_count": 0},
        "road_border": {"collision_count": 0},
        "max_speed_mps": 8.0,
    }
    assert classify_failure(row_norm, verdict_prox) == "PROXIMITY_DEPARTURE"

    # Goal stop failure
    verdict_goal = {"decided": True, "kind": "Failure", "trigger": "timeout", "unmet": ["goal_position"]}
    row_goal = {
        "result_kind": "Failure",
        "object": {"collision_count": 0},
        "road_border": {"collision_count": 0},
        "max_speed_mps": 8.0,
        "scored_stopped_at_goal": True,
    }
    assert classify_failure(row_goal, verdict_goal) == "GOAL_STOP_FAILURE"


def test_classify_hazard_engagement():
    row_avoid = {
        "result_kind": "Pass",
        "object": {"clearance_min_m": 4.2, "collision_count": 0},
        "max_speed_mps": 8.0,
    }
    h = classify_hazard_engagement(row_avoid)
    assert h["hazard_engaged"] is True
    assert h["hazard_verdict"] == "PASS_AVOIDED"

    row_unengaged_pass = {
        "result_kind": "Pass",
        "object": {"clearance_min_m": 50.0, "collision_count": 0},
        "max_speed_mps": 1.0,
    }
    h2 = classify_hazard_engagement(row_unengaged_pass, ego_speed_param=10.0)
    assert h2["hazard_engaged"] is False
    assert h2["hazard_verdict"] == "PASS_UNENGAGED"
