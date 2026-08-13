"""Resolve the map and the ego route a scenario declares.

Interim: the route is computed in Python via ``LaneletSceneBuilder`` (shortestPath /
find_route) rather than by the mission planner, so it is not guaranteed to match the route
the production stack would resolve for the same scenario.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from scenario_generation.gui.lanelet_scene_builder import LaneletSceneBuilder


def map_from_osc(osc_path: str | Path) -> str:
    """The lanelet2 map this scenario declares, from its ``RoadNetwork/LogicFile``.

    The C++ interpreter resolves the map from exactly this element, so deriving the
    Python-side map the same way makes it impossible for the two to disagree. A map passed in
    by the caller cannot offer that guarantee: across a suite spanning several maps, route,
    centreline and road-border geometry would silently be computed against a different map
    than the one the simulator ran.
    """
    root = ET.parse(str(osc_path)).getroot()
    for node in root.iter("LogicFile"):
        path = node.attrib.get("filepath")
        if path:
            return path
    raise ValueError(f"No RoadNetwork/LogicFile filepath in {osc_path}")


def _ego_action_scopes(root: ET.Element, ego_name: str):
    """The elements whose private actions belong to the ego: its ``Init`` block and every
    ``ManeuverGroup`` listing it as an actor."""
    for private in root.iter("Private"):
        if private.get("entityRef") == ego_name:
            yield private
    for group in root.iter("ManeuverGroup"):
        if any(ref.get("entityRef") == ego_name for ref in group.iter("EntityRef")):
            yield group


def _goal_lanelet(osc_path: str | Path, ego_name: str) -> int | None:
    """Goal lanelet id from the ego's own ``AcquirePositionAction`` LanePosition, if the
    scenario authors one. SSv2 lane ids are lanelet ids."""
    root = ET.parse(str(osc_path)).getroot()
    for scope in _ego_action_scopes(root, ego_name):
        for routing in scope.iter("AcquirePositionAction"):
            lp = routing.find(".//LanePosition")
            if lp is not None and "laneId" in lp.attrib:
                try:
                    return int(lp.attrib["laneId"])
                except ValueError:
                    return None
    return None


def resolve_route(
    builder: LaneletSceneBuilder,
    ego_xy: np.ndarray,
    ego_heading: float,
    osc_path: str | Path,
    ego_name: str,
    *,
    min_len_m: float = 120.0,
) -> list[int]:
    """Ordered lanelet ids for the ego route.

    Snaps the sim's actual start pose to a lanelet, then takes the shortest path to the
    scenario's goal when one is authored and reachable, else a forward route of at least
    ``min_len_m``. That fallback is resolved deterministically: a branch picked at random
    would make route_lanes, progress and the road-border geometry differ between runs of the
    same scenario, which is not something an evaluation may leave to chance.
    """
    start_ll = builder.snap_to_nearest_ll(ego_xy, heading_rad=ego_heading)
    if start_ll is None:
        raise RuntimeError(f"Could not snap ego start pose {ego_xy} to any lanelet")
    goal_ll = _goal_lanelet(osc_path, ego_name)
    if goal_ll is not None and builder.has_lanelet_id(goal_ll):
        route = builder.route_between(start_ll, goal_ll)
        if route:
            return route
        print(
            f"  [scenario_sim][WARN] goal lanelet {goal_ll} unreachable from {start_ll}; "
            "falling back to find_route"
        )
    return builder.find_route(start_ll, min_len_m, deterministic=True)
