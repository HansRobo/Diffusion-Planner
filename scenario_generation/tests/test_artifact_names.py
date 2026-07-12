from scenario_generation.artifact_names import artifact_component


def test_artifact_component_is_single_stable_unique_path_part() -> None:
    unsafe = artifact_component("Tokyo/../right turn")

    assert "/" not in unsafe
    assert unsafe not in {"", ".", ".."}
    assert unsafe == artifact_component("Tokyo/../right turn")
    assert unsafe != artifact_component("Tokyo_.._right_turn")
    assert len(artifact_component("x" * 200)) <= 96
