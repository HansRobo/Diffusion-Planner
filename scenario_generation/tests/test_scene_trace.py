import gzip
import json

import numpy as np

from scenario_generation.scene_trace import SceneTraceWriter, write_map_asset


class _Lane:
    raw_centerline = np.array([[0.0, 0.0], [10.0, 0.0]])
    raw_left = np.array([[0.0, 2.0], [10.0, 2.0]])
    raw_right = np.array([[0.0, -2.0], [10.0, -2.0]])


class _Builder:
    _cache = {7: _Lane()}

    def road_border_polylines(self):
        return [np.array([[0.0, -4.0], [10.0, -4.0]])]


def test_map_asset_is_deduplicated_and_trace_is_valid_jsonl(tmp_path):
    ref, asset = write_map_asset(_Builder(), tmp_path / "maps")
    assert asset.is_file()
    assert write_map_asset(_Builder(), tmp_path / "maps")[0] == ref
    with gzip.open(asset, "rt") as f:
        assert json.load(f)["lanes"][0]["id"] == 7

    trace = SceneTraceWriter(
        tmp_path / "case.scene.jsonl.gz", map_ref=ref, route=[[0, 0]], goal=[10, 0]
    )
    agent = type(
        "Agent",
        (),
        {
            "id": "ego",
            "agent_type": "VEHICLE",
            "length": 4.0,
            "width": 1.8,
            "wheelbase": 2.6,
            "current_position": np.array([1.0, 2.0]),
            "current_heading": 0.1,
        },
    )()
    scene = type("Scene", (), {"agents": [agent]})()
    trace.write_frame(0, scene, np.array([[2.0, 2.0], [3.0, 2.0]]), clearance=None, collision=None)
    trace.close("goal")
    with gzip.open(tmp_path / "case.scene.jsonl.gz", "rt") as f:
        lines = [json.loads(line) for line in f]
    assert lines[0]["agents"] == [["ego", 1.0, 2.0, 0.1]]
    assert lines[-1]["map_ref"] == ref
    assert lines[-1]["agents"]["ego"]["length"] == 4.0
