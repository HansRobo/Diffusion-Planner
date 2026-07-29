"""Contract tests for ``--extra_train_set_list`` oversampling.

The oversampled lists are appended *after* the fixed train selector is drawn so
every cycle keeps measuring its commit decision against the same yardstick.
These tests pin that invariant plus the repeat arithmetic.
"""

import json

import pytest

from rlvr.train_awr import _choose_paths, _ordered_paths_sha256


def _write_list(tmp_path, name, count):
    paths = [f"/data/{name}/scene_{i:05d}.npz" for i in range(count)]
    target = tmp_path / name
    target.write_text(json.dumps(paths))
    return target, paths


def test_extra_lists_are_repeated_verbatim(tmp_path):
    target, paths = _write_list(tmp_path, "extra_a.json", 7)
    loaded = _choose_paths(target, 0, 3407, "extra_train:a")
    assert loaded == paths

    repeat = 10
    appended = loaded * repeat
    assert len(appended) == len(paths) * repeat
    # Every scene appears exactly ``repeat`` times — a plain repeat, not a
    # reshuffle, so the oversampling weight is exactly what was requested.
    for path in paths:
        assert appended.count(path) == repeat


def test_appending_extras_keeps_the_selector_prefix_byte_identical(tmp_path):
    _, base = _write_list(tmp_path, "base.json", 5)
    extra_target, _ = _write_list(tmp_path, "extra_b.json", 3)

    train_paths = list(base)
    selector = [train_paths[i] for i in (0, 2, 4)]
    selector_sha = _ordered_paths_sha256(selector)

    extras = _choose_paths(extra_target, 0, 3407, "extra_train:b")
    train_paths.extend(extras * 10)

    # The selector was drawn before the extend, so it is untouched and the
    # original scenes still occupy the same indices.
    assert train_paths[: len(base)] == base
    assert [train_paths[i] for i in (0, 2, 4)] == selector
    assert _ordered_paths_sha256(selector) == selector_sha
    assert len(train_paths) == len(base) + len(extras) * 10


def test_zero_repeat_appends_nothing(tmp_path):
    _, base = _write_list(tmp_path, "base2.json", 4)
    extra_target, _ = _write_list(tmp_path, "extra_c.json", 6)

    train_paths = list(base)
    extras = _choose_paths(extra_target, 0, 3407, "extra_train:c")
    train_paths.extend(extras * 0)
    assert train_paths == base


def test_empty_extra_list_is_rejected(tmp_path):
    target = tmp_path / "empty.json"
    target.write_text("[]")
    with pytest.raises(ValueError, match="empty or not a JSON list"):
        _choose_paths(target, 0, 3407, "extra_train:empty")
