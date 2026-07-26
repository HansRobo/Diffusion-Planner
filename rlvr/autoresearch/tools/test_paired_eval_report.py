import numpy as np

from rlvr.autoresearch.tools.paired_eval_report import compare


def test_progress_is_directional_and_path_length_is_neutral() -> None:
    baseline = {
        "a": {"det_progress": 1.0, "path_len": 2.0},
        "b": {"det_progress": 3.0, "path_len": 4.0},
    }
    candidate = {
        "a": {"det_progress": 2.0, "path_len": 1.0},
        "b": {"det_progress": 5.0, "path_len": 7.0},
    }
    result = compare(baseline, candidate, bootstrap=1000, seed=1)
    progress = result["metrics"]["det_progress"]
    assert progress["direction"] == "higher"
    assert np.isclose(progress["raw_delta"], 1.5)
    assert np.isclose(progress["improvement"], 1.5)

    path = result["metrics"]["path_len"]
    assert path["direction"] == "neutral"
    assert np.isclose(path["raw_delta"], 1.0)
    assert "improvement" not in path
    assert path["raw_delta_ci95"][0] <= 1.0 <= path["raw_delta_ci95"][1]
