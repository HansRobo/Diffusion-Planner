"""Skip-for-training filter for the unified NPZ corpus.

The converter can emit *every* 10 Hz frame (``--write_skipped_npz=1``), including the
ones the production filter would normally drop (red/yellow-light stop, no-future-
progress, collision, off-lane, stale data). Each frame's JSON **sidecar** carries
``is_skipped`` (the ``.npz`` itself has no such field). Those frames exist only so the
closed-loop perception reproducer has a gap-free timeline — they are NOT valid
supervision, so training / eval / data-gen must drop them.

This module is the one shared place that resolves a frame's sidecar and reads the
flag, so every consumer filters identically. It is intentionally dependency-light
(stdlib + tqdm) and lives at the lowest package layer so importers above it have no
circular dependency.

Backward-compatible: a frame with no resolvable sidecar (older corpora, or padded
NPZs whose sidecars live elsewhere and no ``sidecar_root`` was given) is treated as
NOT skipped.
"""

from __future__ import annotations

import functools
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path
from threading import Lock

from tqdm import tqdm

try:
    import orjson as _orjson  # type: ignore
except ImportError:  # pragma: no cover - exercised only in minimal installations
    _orjson = None

# stem -> sidecar path index per sidecar_root, built once (one rglob) and reused for every
# scene, so a large scene list doesn't pay an O(files) rglob per entry. Cached for the
# process lifetime (never invalidated) — correct for the batch train/eval CLIs here; a
# long-lived process that writes new sidecars after the first lookup won't see them.
_SIDECAR_INDEX_CACHE: dict[Path, dict[str, Path]] = {}
_SIDECAR_INDEX_LOCK = Lock()
_FILTER_VERSION = 1


def _scene_path(entry: object) -> str:
    if isinstance(entry, dict):
        return str(entry.get("path") or entry.get("npz") or entry.get("scene_path") or "")
    return entry if isinstance(entry, str) else ""


def _keep_entry(entry: object, sidecar_root: str | Path | None = None) -> bool:
    """Pickle-friendly keep predicate used by the multiprocessing pre-pass."""
    return not is_skipped(_scene_path(entry), sidecar_root=sidecar_root)


def is_skipped(npz_path: str | Path, sidecar_root: str | Path | None = None) -> bool:
    """True iff the frame's JSON sidecar marks it skip_for_training.

    Resolves the sidecar and reads the flag in one self-contained pass. When
    ``sidecar_root`` is supplied, an exact absolute-path mirror or flat overlay
    entry is preferred, then the sibling ``<stem>.json`` is used, and finally a
    cached stem index handles legacy separate-sidecar roots. A missing/empty path,
    an unresolved or unreadable sidecar, or a sidecar without the field all mean
    *not skipped* (False), which keeps older corpora backward compatible.
    """
    if not npz_path or str(npz_path) in ("", "."):
        return False
    npz_path = Path(npz_path)

    # An explicit root is allowed to be a sparse local overlay. Prefer an exact
    # overlay match before reading the shared sibling sidecar; otherwise an old
    # sibling with ``is_skipped=false`` would mask a newly generated local result.
    # The first candidate mirrors the absolute NPZ path below the overlay root,
    # which avoids basename collisions across bags. The flat/stem candidates keep
    # compatibility with the original separate-sidecar layout.
    raw: bytes | None = None
    sidecar: Path | None = None
    if sidecar_root is not None:
        sidecar_root = Path(sidecar_root)
        absolute_relative = Path(str(npz_path).lstrip("/"))
        exact = sidecar_root / absolute_relative.with_suffix(".json")
        flat = sidecar_root / f"{npz_path.stem}.json"
        for candidate in (exact, flat):
            if candidate.is_file():
                sidecar = candidate
                break

    # The common corpus layout has a sibling sidecar. Reading directly avoids a
    # separate stat syscall for every frame on network storage.
    if sidecar is None:
        sibling = npz_path.with_suffix(".json")
        if sibling.is_file():
            sidecar = sibling

    if sidecar is None and sidecar_root is not None:
        # Fall back to a cached stem index for legacy roots that do not mirror the
        # absolute NPZ path. This is inherently weaker than the exact overlay path,
        # but preserves the existing API for old sidecar collections.
        key = sidecar_root.resolve()
        index = _SIDECAR_INDEX_CACHE.get(key)
        if index is None:
            # Several worker threads can resolve the first separate-root sidecar
            # concurrently. Serialize the one expensive rglob so a 32/64-worker
            # scan does not launch the same walk repeatedly.
            with _SIDECAR_INDEX_LOCK:
                index = _SIDECAR_INDEX_CACHE.get(key)
                if index is None:
                    index = {p.stem: p for p in sidecar_root.rglob("*.json")}
                    _SIDECAR_INDEX_CACHE[key] = index
        sidecar = index.get(npz_path.stem)
    if sidecar is None:
        return False
    try:
        if raw is None:
            raw = sidecar.read_bytes()
        payload = _orjson.loads(raw) if _orjson is not None else json.loads(raw)
        return bool(payload.get("is_skipped", False)) if isinstance(payload, dict) else False
    except (OSError, ValueError, UnicodeError):
        return False


def filter_scene_list(
    scenes: list,
    sidecar_root: str | Path | None = None,
    enabled: bool = True,
    label: str = "",
) -> list:
    """Drop scenes flagged ``is_skipped`` in their sidecar. ``enabled=False`` returns the
    input unchanged. Entry types are preserved: an entry is a bare npz-path string, or a
    dict keyed ``path`` / ``npz`` / ``scene_path``. Frames with no resolvable sidecar are
    kept (treated as not-skipped)."""
    if not enabled or not scenes:
        return scenes
    kept, dropped = [], 0
    for entry in tqdm(scenes, desc="[skip-filter]", unit="scene"):
        if isinstance(entry, dict):
            path = entry.get("path") or entry.get("npz") or entry.get("scene_path") or ""
        else:
            path = entry if isinstance(entry, str) else ""
        if is_skipped(path, sidecar_root):
            dropped += 1
        else:
            kept.append(entry)
    tag = f"{label}: " if label else ""
    print(
        f"[skip-filter] {tag}kept {len(kept)}/{len(scenes)} (dropped {dropped} skip_for_training)"
    )
    return kept


def _load_manifest(path: Path) -> list:
    """Load a flat manifest with the fastest available JSON decoder."""
    values = (
        _orjson.loads(path.read_bytes())
        if _orjson is not None
        else json.loads(path.read_text(encoding="utf-8"))
    )
    if not isinstance(values, list):
        raise ValueError(f"{path}: scene list must be a JSON array")
    return values


def _atomic_json_dump(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        try:
            if _orjson is None:
                raise ImportError
            encoded = _orjson.dumps(value, option=_orjson.OPT_APPEND_NEWLINE)
            with os.fdopen(fd, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            fd = -1
        except ImportError:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(value, stream, separators=(",", ":"))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            fd = -1
        os.replace(temporary_name, path)
    finally:
        if fd >= 0:
            # ``fdopen`` closes the descriptor in either branch; this is only a
            # guard for an exception raised before the context manager is entered.
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def filter_manifest_file(
    source: str | Path,
    destination: str | Path,
    *,
    sidecar_root: str | Path | None = None,
    workers: int = 32,
    processes: int = 0,
    label: str = "",
) -> dict[str, int | str]:
    """Write an ``is_skipped=False`` copy of one manifest.

    The source manifest and all NPZ/JSON sidecars remain untouched.  Filtering is
    performed once by the training launcher (rank zero), rather than in every
    ``__getitem__`` call or independently on every DDP rank.  The bounded thread
    pool overlaps the many small sidecar reads on network storage while preserving
    source order and duplicates.
    """
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if processes < 0:
        raise ValueError("processes must be >= 0")
    source_path = Path(source)
    destination_path = Path(destination)
    scenes = _load_manifest(source_path)

    def keep(entry: object) -> bool:
        return _keep_entry(entry, sidecar_root=sidecar_root)

    kept: list = []
    dropped = 0
    # Chunk submissions prevent millions of pending Future objects and keep the
    # process's memory bounded by the manifest itself.
    # Sidecar reads are latency-bound on the RDMA corpus.  A bounded process
    # pool avoids the GIL and, unlike hundreds of threads, does not create an
    # I/O queue large enough to collapse throughput.  Keep the thread-pool
    # implementation as the default for callers that need a lightweight
    # in-process filter or run on local storage.
    if processes > 1:
        executor = Pool(processes=processes)
        map_flags = lambda chunk: executor.imap(  # noqa: E731
            functools.partial(_keep_entry, sidecar_root=sidecar_root),
            chunk,
            chunksize=256,
        )
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        map_flags = lambda chunk: executor.map(keep, chunk)  # noqa: E731
    try:
        chunk_size = max(4096, (processes if processes > 1 else workers) * 256)
        log_interval = max(chunk_size, (len(scenes) + 99) // 100)
        next_log = log_interval
        for start in range(0, len(scenes), chunk_size):
            chunk = scenes[start : start + chunk_size]
            flags = map_flags(chunk)
            for entry, should_keep in zip(chunk, flags, strict=True):
                if should_keep:
                    kept.append(entry)
                else:
                    dropped += 1
            scanned = min(start + len(chunk), len(scenes))
            if label and (scanned >= next_log or scanned == len(scenes)):
                print(
                    f"[skip-filter] {label}: scanned {scanned}/{len(scenes)}; dropped={dropped}",
                    flush=True,
                )
                while next_log <= scanned:
                    next_log += log_interval
    finally:
        if processes > 1:
            executor.close()
            executor.join()
        else:
            executor.shutdown(wait=True)

    _atomic_json_dump(kept, destination_path)
    result: dict[str, int | str] = {
        "source": str(source_path),
        "destination": str(destination_path),
        "source_count": len(scenes),
        "kept_count": len(kept),
        "dropped_count": dropped,
    }
    return result


def prepare_filtered_manifests(
    *,
    train_set_list: str,
    valid_set_list: str,
    extra_train_set_list: str | list[str] | None,
    enabled: bool,
    cache_dir: str | Path,
    is_master: bool,
    sidecar_root: str | Path | None = None,
    workers: int = 32,
    processes: int = 0,
) -> tuple[str, str, str | list[str] | None, list[dict[str, int | str]]]:
    """Prepare cached filtered manifests and return effective list paths.

    Only ``is_master`` writes the cache.  Other DDP ranks receive deterministic
    destination paths and wait at the caller's collective barrier.  Cache entries
    are reused when the source path's size/mtime is unchanged; sidecars are part of
    the immutable corpus contract and therefore do not require a multi-million-file
    rehash on every launch.
    """
    if not enabled:
        extras = extra_train_set_list
        return train_set_list, valid_set_list, extras, []

    cache_root = Path(cache_dir) / "filtered_manifests"
    metadata_path = cache_root / "metadata.json"
    resolved_sidecar_root = str(Path(sidecar_root).resolve()) if sidecar_root is not None else None
    metadata: dict = {}
    if is_master and metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
    old_entries = metadata.get("entries", {}) if isinstance(metadata, dict) else {}
    new_entries: dict[str, dict] = {}
    stats: list[dict[str, int | str]] = []

    extras = (
        [extra_train_set_list]
        if isinstance(extra_train_set_list, str)
        else list(extra_train_set_list or [])
    )
    sources = [("train", train_set_list), ("valid", valid_set_list)] + [
        (f"extra_{index}", path) for index, path in enumerate(extras)
    ]
    effective: dict[str, str] = {}
    for role, source in sources:
        source_path = Path(source)
        destination = cache_root / f"{role}_{source_path.name}"
        try:
            source_stat = source_path.stat()
        except OSError as exc:
            raise FileNotFoundError(f"cannot stat manifest {source_path}: {exc}") from exc
        cache_key = f"{role}:{source_path.resolve()}"
        prior = old_entries.get(cache_key, {}) if isinstance(old_entries, dict) else {}
        reusable = (
            is_master
            and destination.is_file()
            and isinstance(prior, dict)
            and prior.get("filter_version") == _FILTER_VERSION
            and prior.get("source_size") == source_stat.st_size
            and prior.get("source_mtime_ns") == source_stat.st_mtime_ns
            and prior.get("sidecar_root") == resolved_sidecar_root
        )
        if is_master and not reusable:
            stat = filter_manifest_file(
                source_path,
                destination,
                sidecar_root=sidecar_root,
                workers=workers,
                processes=processes,
                label=role,
            )
            stats.append(stat)
        elif is_master:
            stats.append(
                {
                    "source": str(source_path),
                    "destination": str(destination),
                    "source_count": int(prior.get("source_count", -1)),
                    "kept_count": int(prior.get("kept_count", -1)),
                    "dropped_count": int(prior.get("dropped_count", -1)),
                    "reused": "1",
                }
            )
        new_entries[cache_key] = {
            "filter_version": _FILTER_VERSION,
            "source": str(source_path),
            "destination": str(destination),
            "source_size": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
            "sidecar_root": resolved_sidecar_root,
            **(stats[-1] if stats and stats[-1].get("source") == str(source_path) else {}),
        }
        effective[role] = str(destination)

    if is_master:
        _atomic_json_dump(
            {"filter_version": _FILTER_VERSION, "entries": new_entries}, metadata_path
        )
    effective_extras = [effective[f"extra_{index}"] for index in range(len(extras))]
    return (
        effective["train"],
        effective["valid"],
        effective_extras if extra_train_set_list is not None else None,
        stats,
    )


def prepare_filtered_manifest(
    source: str,
    *,
    role: str,
    cache_dir: str | Path,
    enabled: bool,
    is_master: bool,
    sidecar_root: str | Path | None = None,
    workers: int = 32,
    processes: int = 0,
) -> tuple[str, dict[str, int | str] | None]:
    """Prepare one cached manifest for standalone evaluation tools.

    ``prepare_filtered_manifests`` handles the train/valid/extra set together so a
    training run writes one metadata file.  Validation is also commonly launched
    later as a separate process, so it needs the same cache semantics without
    inventing a dummy training list.  The destination is deterministic and only
    rank zero writes it; callers must place a DDP barrier after this function.
    """
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if not enabled:
        return source, None

    source_path = Path(source)
    cache_root = Path(cache_dir) / "filtered_manifests"
    # Training serializes the effective (already filtered) validation path into
    # args.json.  Reusing that path directly avoids rescanning millions of sidecars
    # when a later standalone validation is launched on the same checkpoint.
    if source_path.parent.resolve() == cache_root.resolve() and source_path.is_file():
        return str(source_path), None
    destination = cache_root / f"{role}_{source_path.name}"
    metadata_path = cache_root / "metadata.json"
    resolved_sidecar_root = str(Path(sidecar_root).resolve()) if sidecar_root is not None else None
    metadata: dict = {}
    if is_master and metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
    old_entries = metadata.get("entries", {}) if isinstance(metadata, dict) else {}
    if not isinstance(old_entries, dict):
        old_entries = {}
    try:
        source_stat = source_path.stat()
    except OSError as exc:
        raise FileNotFoundError(f"cannot stat manifest {source_path}: {exc}") from exc
    cache_key = f"{role}:{source_path.resolve()}"
    prior = old_entries.get(cache_key, {})
    reusable = (
        is_master
        and destination.is_file()
        and isinstance(prior, dict)
        and prior.get("filter_version") == _FILTER_VERSION
        and prior.get("source_size") == source_stat.st_size
        and prior.get("source_mtime_ns") == source_stat.st_mtime_ns
        and prior.get("sidecar_root") == resolved_sidecar_root
    )
    if not is_master:
        return str(destination), None

    if reusable:
        stats: dict[str, int | str] = {
            "source": str(source_path),
            "destination": str(destination),
            "source_count": int(prior.get("source_count", -1)),
            "kept_count": int(prior.get("kept_count", -1)),
            "dropped_count": int(prior.get("dropped_count", -1)),
            "reused": "1",
        }
    else:
        stats = filter_manifest_file(
            source_path,
            destination,
            sidecar_root=sidecar_root,
            workers=workers,
            processes=processes,
            label=role,
        )

    entry = {
        "filter_version": _FILTER_VERSION,
        "source": str(source_path),
        "destination": str(destination),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "sidecar_root": resolved_sidecar_root,
        **stats,
    }
    _atomic_json_dump(
        {"filter_version": _FILTER_VERSION, "entries": {**old_entries, cache_key: entry}},
        metadata_path,
    )
    return str(destination), stats
