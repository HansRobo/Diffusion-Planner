#!/usr/bin/env python3
"""Extract lanelet2_map.osm from a rosbag's /map/vector_map topic.

The `/map/vector_map` topic carries an `autoware_map_msgs/msg/LaneletMapBin`
message. Its `.data` field is *not* raw OSM XML text -- it is a Boost
serialization archive of an in-memory lanelet2 map (the format produced by
`autoware_lanelet2_extension`'s `toBinMsg`/message_conversion, similar to
`lanelet::io_handlers::BinHandler::write()`). Conveniently, lanelet2's own
".bin" IO handler can deserialize this directly: writing the raw bytes to a
file and calling `lanelet2.io.load()` on it (extension-based dispatch) works
without needing any C++ helper.

Once loaded, the map's point coordinates are already in the bag's native
local "map" frame -- the same frame used by `/localization/kinematic_state`
and other topics. These ORScene override bags use a private/local coordinate
convention (not real-world MGRS georeference): local x/y can be negative and
span only a few km.

`MGRSProjector` (used by this repo's `diffusion_planner_ros.lanelet2_utils
.lanelet_converter.convert_lanelet`, which `parse_rosbag.py` calls
unmodified) can only represent non-negative offsets in [0, 100000) from a
100km MGRS grid corner: `reverse()` with a bare (precision-0) MGRS code adds
the raw point value to an exact multiple-of-100000 UTM corner, and
`forward()` always reports `fmod(utm, 100000)` which is non-negative by
construction. A local frame straddling zero will "wrap" into an adjacent
grid square on reload, corrupting geometry.

To keep the OSM file loadable via the existing (unmodified) `convert_lanelet
()` pipeline while preserving the map's internal geometry exactly, this
script shifts all point coordinates by a constant vector so every point
lands safely inside a single grid square (with margin), then uses
`MGRSProjector.reverse()` with a bare MGRS code to compute lat/lon for the
OSM writer.

IMPORTANT CAVEAT: because of this shift, the extracted map's *absolute*
coordinates will not line up with this bag's actual ego pose
(`/localization/kinematic_state`) once reloaded via `MGRSProjector`. Lane /
route lookups keyed on absolute ego position (as `parse_rosbag.py` does) will
not find nearby lanelets. The map's *shape* (topology, relative distances
between points) is preserved exactly -- only the absolute placement is
shifted. See .superpowers/sdd/task-2-report.md for details and the tracked
follow-up.

SEPARATELY: the bin-deserialized lanelet objects carry a hidden/cached
internal "centerline" relation member left over from whatever submap
extraction produced this bin snapshot; for a subset of lanelets that member
points at an ID that doesn't exist anywhere in the map. `lanelet2.io.write()`
drops any primitive with such a dangling member entirely (not just the bad
member), which silently deletes ~2-3% of lanelets -- including ones
referenced by `/planning/mission_planning/route`, which crashes
`parse_rosbag.py` with a `KeyError`. To avoid this, every lanelet is
reconstructed from scratch (id, left/right bound, attributes, regulatory
elements only) before writing, which drops the stale cached member while
preserving geometry and regulatory-element associations (e.g. traffic
lights) exactly.
"""
import argparse
from pathlib import Path

import lanelet2
import rosbag2_py
from autoware_lanelet2_extension_python.projection import MGRSProjector
from lanelet2.core import Lanelet, LaneletMap
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# Any bare (precision-0) MGRS Grid Zone Designator + 100km-square ID works --
# it only serves as an arbitrary anchor point far from a real 100km grid
# boundary once our shift is applied. The actual real-world location is
# irrelevant since these maps aren't really MGRS-georeferenced.
DEFAULT_MGRS_CODE = "54SUE"
# Margin (m) kept between the shifted map's extents and the [0, 100000)
# grid-square bounds, so no point can wrap into a neighboring square.
MARGIN_M = 1000.0


def _read_lanelet_map_bin_bytes(bag_path: Path) -> bytes:
    """Read the raw `.data` bytes of the first /map/vector_map message in a bag."""
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr"
    )
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    storage_filter = rosbag2_py.StorageFilter(topics=["/map/vector_map"])
    reader.set_filter(storage_filter)

    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == "/map/vector_map":
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)
            return bytes(msg.data)

    raise RuntimeError("/map/vector_map topic not found in bag")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_path", type=Path)
    parser.add_argument("--output", type=Path, default=Path("lanelet2_map.osm"))
    parser.add_argument("--mgrs_code", type=str, default=DEFAULT_MGRS_CODE)
    args = parser.parse_args()

    raw_bytes = _read_lanelet_map_bin_bytes(args.bag_path)
    print(f"Read {len(raw_bytes)} bytes from /map/vector_map")

    # lanelet2 dispatches its IO handler by file extension, so the temp file
    # must end in ".bin" for the Boost-serialization loader to be selected.
    bin_path = args.output.parent / (args.output.stem + "_raw_tmp.bin")
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    bin_path.write_bytes(raw_bytes)

    # No projector work needed to deserialize the bin archive -- point x/y/z
    # are stored directly. Origin is unused by the bin loader.
    load_projector = MGRSProjector(lanelet2.io.Origin(0.0, 0.0))
    raw_map = lanelet2.io.load(str(bin_path), load_projector)
    bin_path.unlink()

    # Rebuild every lanelet from scratch to drop the stale/dangling cached
    # member (see module docstring) while keeping geometry, attributes, and
    # regulatory elements (traffic lights, right-of-way, crosswalks, ...)
    # intact. Polygons and standalone line strings are copied over as-is.
    lanelet_map = LaneletMap()
    for lanelet in raw_map.laneletLayer:
        fresh = Lanelet(
            lanelet.id,
            lanelet.leftBound,
            lanelet.rightBound,
            lanelet.attributes,
            list(lanelet.regulatoryElements),
        )
        lanelet_map.add(fresh)
    for polygon in raw_map.polygonLayer:
        lanelet_map.add(polygon)
    for line_string in raw_map.lineStringLayer:
        lanelet_map.add(line_string)

    xs = [p.x for p in lanelet_map.pointLayer]
    ys = [p.y for p in lanelet_map.pointLayer]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x, span_y = max_x - min_x, max_y - min_y
    print(
        f"Loaded {len(lanelet_map.laneletLayer)} lanelets, {len(xs)} points. "
        f"Local x range=[{min_x:.1f}, {max_x:.1f}] (span {span_x:.1f} m), "
        f"y range=[{min_y:.1f}, {max_y:.1f}] (span {span_y:.1f} m)"
    )
    if max(span_x, span_y) + 2 * MARGIN_M >= 100_000:
        raise RuntimeError(
            f"Map span ({span_x:.0f} x {span_y:.0f} m) is too large to fit inside a "
            "single 100km MGRS grid square with margin -- the shift trick in this "
            "script cannot avoid grid-boundary wrapping. Aborting."
        )

    shift_x = -min_x + MARGIN_M
    shift_y = -min_y + MARGIN_M
    print(f"Applying shift ({shift_x:.1f}, {shift_y:.1f}) to keep all points >= 0 "
          f"and well inside [0, 100000) for a single MGRS grid square.")

    for point in lanelet_map.pointLayer:
        point.x += shift_x
        point.y += shift_y

    write_projector = MGRSProjector(lanelet2.io.Origin(0.0, 0.0))
    write_projector.setMGRSCode(args.mgrs_code)
    try:
        lanelet2.io.write(str(args.output), lanelet_map, write_projector)
    except RuntimeError as e:
        # lanelet2 raises after writing if some relations reference
        # primitives missing from the (bin-deserialized) map -- the file is
        # still written; log and continue.
        print(f"WARNING: lanelet2 reported errors while writing (file still "
              f"written): {e}")

    print(f"Saved OSM map to {args.output}")
    print(
        "NOTE: absolute coordinates in this OSM file are shifted by "
        f"({shift_x:.1f}, {shift_y:.1f}) m relative to the bag's native map "
        "frame (needed to round-trip through MGRSProjector without grid "
        "wraparound). Lane/route lookups keyed on this bag's raw ego pose "
        "will not find nearby lanelets -- see script docstring."
    )


if __name__ == "__main__":
    main()
