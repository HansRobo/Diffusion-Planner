"""scene_skip: drop only the sidecar-flagged frames, keep everything else, and be a
no-op (backward compatible) when sidecars/flags are absent."""

import json

from diffusion_planner.utils.scene_skip import (
    filter_manifest_file,
    filter_scene_list,
    is_skipped,
    prepare_filtered_manifest,
    prepare_filtered_manifests,
)


def _frame(tmp, stem, skipped=None):
    """Create <stem>.npz and (optionally) a sibling <stem>.json with is_skipped."""
    npz = tmp / f"{stem}.npz"
    npz.write_bytes(b"")  # content irrelevant; we only resolve the sidecar
    if skipped is not None:
        (tmp / f"{stem}.json").write_text(json.dumps({"x": 0.0, "is_skipped": skipped}))
    return str(npz)


def test_is_skipped_reads_sibling_sidecar(tmp_path):
    flagged = _frame(tmp_path, "a", skipped=True)
    keep = _frame(tmp_path, "b", skipped=False)
    nosc = _frame(tmp_path, "c", skipped=None)  # no sidecar
    assert is_skipped(flagged) is True
    assert is_skipped(keep) is False
    assert is_skipped(nosc) is False  # missing sidecar => not skipped


def test_filter_drops_only_flagged(tmp_path):
    scenes = [
        _frame(tmp_path, "a", skipped=True),
        _frame(tmp_path, "b", skipped=False),
        _frame(tmp_path, "c", skipped=True),
        _frame(tmp_path, "d", skipped=False),
    ]
    kept = filter_scene_list(scenes, label="unit")
    assert kept == [scenes[1], scenes[3]]


def test_missing_sidecar_passthrough(tmp_path):
    scenes = [_frame(tmp_path, f"n{i}", skipped=None) for i in range(3)]
    kept = filter_scene_list(scenes)  # no sidecars => all kept (backward compatible)
    assert kept == scenes


def test_disabled_is_noop(tmp_path):
    scenes = [_frame(tmp_path, "a", skipped=True), _frame(tmp_path, "b", skipped=False)]
    assert filter_scene_list(scenes, enabled=False) == scenes


def test_dict_entries_preserved(tmp_path):
    a, b = _frame(tmp_path, "a", skipped=True), _frame(tmp_path, "b", skipped=False)
    scenes = [{"path": a, "weight": 1}, {"npz": b, "weight": 2}]
    kept = filter_scene_list(scenes)
    assert kept == [{"npz": b, "weight": 2}]  # the flagged dict dropped, entry type kept


def test_sidecar_root_index(tmp_path):
    # NPZs in one dir, sidecars nested under a separate root (padded-corpus style).
    npz_dir, sc_root = tmp_path / "npz", tmp_path / "sc" / "date" / "bag"
    npz_dir.mkdir(parents=True)
    sc_root.mkdir(parents=True)
    (npz_dir / "x.npz").write_bytes(b"")
    (sc_root / "x.json").write_text(json.dumps({"is_skipped": True}))
    # sidecar resolved by stem under sidecar_root => the flagged frame is dropped
    assert is_skipped(npz_dir / "x.npz", sidecar_root=tmp_path / "sc") is True
    assert filter_scene_list([str(npz_dir / "x.npz")], sidecar_root=tmp_path / "sc") == []


def test_sidecar_root_overlay_overrides_sibling(tmp_path):
    # A sparse local overlay must win over the old shared sibling sidecar. Store
    # it under the absolute-path mirror so duplicate frame stems cannot collide.
    npz = tmp_path / "dataset" / "bag" / "x.npz"
    npz.parent.mkdir(parents=True)
    npz.write_bytes(b"")
    npz.with_suffix(".json").write_text(json.dumps({"is_skipped": False}))
    overlay = tmp_path / "overlay"
    mirrored = (overlay / str(npz).lstrip("/")).with_suffix(".json")
    mirrored.parent.mkdir(parents=True)
    mirrored.write_text(json.dumps({"is_skipped": True}))
    assert is_skipped(npz, sidecar_root=overlay) is True


def test_filter_manifest_file_preserves_order_and_duplicates(tmp_path):
    a = _frame(tmp_path, "a", skipped=True)
    b = _frame(tmp_path, "b", skipped=False)
    source = tmp_path / "source.json"
    destination = tmp_path / "cache" / "filtered.json"
    source.write_text(json.dumps([a, b, a, b]))
    stats = filter_manifest_file(source, destination, workers=2, label="test")
    assert json.loads(destination.read_text()) == [b, b]
    assert stats["source_count"] == 4
    assert stats["kept_count"] == 2
    assert stats["dropped_count"] == 2


def test_prepare_filtered_manifests_reuses_run_cache(tmp_path):
    train = _frame(tmp_path, "train", skipped=True)
    valid = _frame(tmp_path, "valid", skipped=False)
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    train_list = manifest_dir / "train.json"
    valid_list = manifest_dir / "valid.json"
    train_list.write_text(json.dumps([train]))
    valid_list.write_text(json.dumps([valid]))
    first = prepare_filtered_manifests(
        train_set_list=str(train_list),
        valid_set_list=str(valid_list),
        extra_train_set_list=None,
        enabled=True,
        cache_dir=tmp_path / "run",
        is_master=True,
        workers=2,
    )
    with open(first[0], encoding="utf-8") as handle:
        assert json.load(handle) == []
    second = prepare_filtered_manifests(
        train_set_list=str(train_list),
        valid_set_list=str(valid_list),
        extra_train_set_list=None,
        enabled=True,
        cache_dir=tmp_path / "run",
        is_master=True,
        workers=2,
    )
    assert second[0] == first[0]
    assert second[3][0].get("reused") == "1"


def test_prepare_filtered_manifest_reuses_training_effective_path(tmp_path):
    flagged = _frame(tmp_path, "flagged", skipped=True)
    source = tmp_path / "valid.json"
    source.write_text(json.dumps([flagged]))
    cache_dir = tmp_path / "run"
    filtered, stats = prepare_filtered_manifest(
        str(source),
        role="valid",
        cache_dir=cache_dir,
        enabled=True,
        is_master=True,
        workers=2,
    )
    assert stats["dropped_count"] == 1
    with open(filtered, encoding="utf-8") as handle:
        assert json.load(handle) == []
    # A standalone evaluator receives the effective path from args.json. It must
    # not create valid_valid.json and rescan the corpus a second time.
    same, second_stats = prepare_filtered_manifest(
        filtered,
        role="valid",
        cache_dir=cache_dir,
        enabled=True,
        is_master=True,
        workers=2,
    )
    assert same == filtered
    assert second_stats is None
