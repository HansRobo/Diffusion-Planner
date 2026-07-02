"""Central loader/index for dataset path lists (``path_list_*.json``).

Two on-disk formats are supported so the same JSON key can point at either the legacy
per-frame dataset (``basic_dataset``) or the new *packed* dataset (one ``.npz`` per
sequence, every feature array carrying a leading frame dimension):

  * **legacy (flat)** -- a JSON list of ``.npz`` path strings, one *sample* per file::

        ["/abs/a.npz", "/abs/b.npz", ...]

  * **manifest (packed)** -- a JSON list of per-file objects, one *sequence* per file::

        [{"path": "/abs/seq.npz", "num_frames": 581, "valid": [0, 1, 4, ...]}, ...]

    ``valid`` is optional; when omitted every frame ``0..num_frames-1`` is used. It is the
    place to drop bad frames (padding, avoidance ``origin`` frames, SFT selection, ...) once,
    offline, so the training loop never re-checks validity.

Both formats normalize to the same in-memory entry ``{"path", "num_frames", "valid"}`` and
expand to a flat *frame index* ``[(entry_idx, frame_idx), ...]`` whose length is the number of
trainable samples. A packed ``.npz`` is recognised at read time by the presence of a
``frame_indices`` key; a single frame is then sliced out so downstream code receives exactly
the same per-sample dict shape it always did.
"""

from collections import OrderedDict

import numpy as np

from diffusion_planner.utils.train_utils import openjson

# Keys never sliced by frame when materialising a packed sample: ``version`` is scene-format
# metadata (shape ``(1,)``) and ``frame_indices`` is the packing bookkeeping itself.
_NON_FRAME_KEYS = ("version",)
_PACKED_MARKER = "frame_indices"


def load_path_list(json_path):
    """Load a path-list JSON and normalise it to a list of entries.

    Each entry is ``{"path": str, "num_frames": int, "valid": list[int] | None}``.
    ``valid=None`` means "all frames" (kept lazy to avoid materialising huge index lists).
    """
    raw = openjson(json_path)
    entries = []
    for item in raw:
        if isinstance(item, str):
            # Legacy flat format: one sample per file.
            entries.append({"path": item, "num_frames": 1, "valid": None})
        elif isinstance(item, dict):
            path = item["path"]
            num_frames = int(item.get("num_frames", 1))
            valid = item.get("valid")
            entries.append(
                {
                    "path": path,
                    "num_frames": num_frames,
                    "valid": list(valid) if valid is not None else None,
                }
            )
        else:
            raise TypeError(f"Unsupported path-list entry type: {type(item)!r}")
    return entries


def build_frame_index(entries):
    """Expand entries into a flat ``[(entry_idx, frame_idx), ...]`` list of trainable frames."""
    index = []
    for e_idx, entry in enumerate(entries):
        valid = entry["valid"]
        frames = valid if valid is not None else range(entry["num_frames"])
        for f_idx in frames:
            index.append((e_idx, f_idx))
    return index


def scene_paths(json_path):
    """Return just the file paths from a path list (for file-level consumers that do not
    care about individual frames, e.g. neighbor-DB build or clustering)."""
    return [e["path"] for e in load_path_list(json_path)]


class _FileCache:
    """Per-process LRU cache of *decompressed* npz files.

    A packed ``.npz`` is ~67 MB on disk but ~700 MB decompressed, and indexing any key
    decompresses that whole array. Caching the materialised dict means sequential frames from
    the same file (as produced by a shard-local sampler) decompress the file only once.
    Random global shuffling defeats the cache -- pair packed training with a shard-grouped
    sampler / shuffle buffer, not a naive global shuffle.
    """

    def __init__(self, maxsize=1):
        self.maxsize = maxsize
        self._cache = OrderedDict()

    def get(self, path):
        cached = self._cache.get(path)
        if cached is not None:
            self._cache.move_to_end(path)
            return cached
        with np.load(path, allow_pickle=True) as npz:
            data = {k: npz[k] for k in npz.files}
        self._cache[path] = data
        while len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)
        return data


_DEFAULT_CACHE = _FileCache(maxsize=1)


def load_frame(path, frame_idx, cache=None):
    """Load a single sample dict from ``path``.

    For a packed file (has ``frame_indices``) the ``frame_idx``-th frame is sliced out of every
    per-frame array; for a legacy single-sample file the dict is returned as-is. In both cases
    the returned dict matches the per-sample shapes the model has always consumed.
    """
    cache = cache if cache is not None else _DEFAULT_CACHE
    data = cache.get(path)
    if _PACKED_MARKER not in data:
        return {k: v for k, v in data.items()}
    num_frames = len(data[_PACKED_MARKER])
    out = {}
    for k, v in data.items():
        if k == _PACKED_MARKER:
            continue
        if (
            k not in _NON_FRAME_KEYS
            and hasattr(v, "shape")
            and v.ndim >= 1
            and v.shape[0] == num_frames
        ):
            out[k] = v[frame_idx]
        else:
            out[k] = v
    return out
