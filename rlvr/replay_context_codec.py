"""Lossless row-addressable compression for immutable AWR replay context.

The full T4 replay stores policy-independent encoder/context tensors as dense
float32 memmaps.  Random group replay then reads roughly 580 GiB per rank even
though the tensors are highly compressible.  This module stores one independent
Zstandard frame per scene so random replay can fetch exactly the requested
rows.  Decompression reconstructs the original float32 bytes; it is a storage
and scheduling change, not a numerical approximation.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import zstandard as zstd
except ImportError as error:  # pragma: no cover - exercised by deployment guard
    zstd = None
    _ZSTD_IMPORT_ERROR = error
else:
    _ZSTD_IMPORT_ERROR = None


SCHEMA = "awr_replay_context_zstd_rows_v1"
ROWS_FILE = "context_rows.zst"
OFFSETS_FILE = "context_row_offsets.npy"
MANIFEST_FILE = "manifest.json"


@dataclass(frozen=True)
class _SourceField:
    name: str
    array: np.ndarray
    source_path: Path


def _require_zstd() -> None:
    if zstd is None:
        raise RuntimeError(
            "compressed replay context requires the 'zstandard' package"
        ) from _ZSTD_IMPORT_ERROR


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json_fsync(path: Path, value: dict[str, Any]) -> None:
    with path.open("w") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_path(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_source_fields(source_rank_dir: Path) -> tuple[int, str, list[_SourceField]]:
    replay_manifest_path = source_rank_dir / "manifest.json"
    expert_manifest_path = source_rank_dir / "expert_anchor_manifest.json"
    if not replay_manifest_path.is_file():
        raise FileNotFoundError(replay_manifest_path)
    if not expert_manifest_path.is_file():
        raise FileNotFoundError(expert_manifest_path)
    replay_manifest = json.loads(replay_manifest_path.read_text())
    expert_manifest = json.loads(expert_manifest_path.read_text())
    scene_count = int(replay_manifest.get("scene_count", -1))
    if scene_count <= 0 or int(expert_manifest.get("scene_count", -2)) != scene_count:
        raise RuntimeError("replay/expert scene counts differ or are invalid")
    if not bool(replay_manifest.get("decoder_context_cached", False)):
        raise RuntimeError("source replay has no strict decoder-context cache")

    path_hash = str(
        replay_manifest.get("decoder_context_attachment", {}).get(
            "expected_paths_sha256", ""
        )
    )
    if len(path_hash) != 64:
        raise RuntimeError("source replay has no expected scene-path hash")

    replay_names = replay_manifest.get("arrays", {})
    expert_names = expert_manifest.get("arrays", {})
    specs = (
        ("scene_encoding", replay_names.get("scene_encoding")),
        ("ego_current_state", replay_names.get("context_ego_current_state")),
        (
            "neighbor_agents_past",
            replay_names.get("context_neighbor_agents_past"),
        ),
        (
            "neighbor_agents_future",
            replay_names.get("context_neighbor_agents_future"),
        ),
        ("expert_trajectories", expert_names.get("expert_trajectories")),
        ("expert_rewards", expert_names.get("expert_rewards")),
        ("expert_safe", expert_names.get("expert_safe")),
    )
    fields: list[_SourceField] = []
    for name, filename in specs:
        if not filename:
            raise RuntimeError(f"source replay is missing {name}")
        path = (source_rank_dir / str(filename)).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        array = np.load(path, mmap_mode="r")
        if array.dtype != np.float32 or int(array.shape[0]) != scene_count:
            raise RuntimeError(
                f"invalid source field {name}: dtype={array.dtype}, "
                f"shape={tuple(array.shape)}, scenes={scene_count}"
            )
        fields.append(_SourceField(name=name, array=array, source_path=path))
    return scene_count, path_hash, fields


def _validate_existing_rank(
    rank_dir: Path,
    *,
    expected_scene_count: int,
    expected_paths_sha256: str,
) -> dict[str, Any]:
    reader = CompressedReplayContextReader(
        rank_dir.parent,
        int(rank_dir.name.split("_")[-1]),
        expected_scene_count=expected_scene_count,
        expected_paths_sha256=expected_paths_sha256,
    )
    try:
        return dict(reader.manifest)
    finally:
        reader.close()


def build_compressed_replay_context_rank(
    source_replay_root: Path,
    output_root: Path,
    rank: int,
    *,
    compression_level: int = 1,
    progress_interval: int = 10_000,
) -> dict[str, Any]:
    """Build and atomically publish one rank-local compressed context sidecar."""

    _require_zstd()
    source_replay_root = Path(source_replay_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    rank = int(rank)
    source_rank_dir = source_replay_root / f"rank_{rank:04d}"
    scene_count, expected_paths_sha256, fields = _load_source_fields(source_rank_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    rank_dir = output_root / f"rank_{rank:04d}"
    if rank_dir.exists():
        return _validate_existing_rank(
            rank_dir,
            expected_scene_count=scene_count,
            expected_paths_sha256=expected_paths_sha256,
        )

    temporary = output_root / f".rank_{rank:04d}.building.{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    rows_path = temporary / ROWS_FILE
    offsets_path = temporary / OFFSETS_FILE
    manifest_path = temporary / MANIFEST_FILE
    offsets = np.empty(scene_count + 1, dtype=np.uint64)
    offsets[0] = 0
    raw_hashes = {field.name: hashlib.sha256() for field in fields}
    compressed_hash = hashlib.sha256()
    compressor = zstd.ZstdCompressor(
        level=int(compression_level),
        write_checksum=True,
        write_content_size=True,
    )
    field_rows: list[dict[str, Any]] = []
    raw_offset = 0
    for field in fields:
        row_shape = [int(value) for value in field.array.shape[1:]]
        row_bytes = int(np.prod(row_shape, dtype=np.int64)) * 4
        if not row_shape:
            row_bytes = 4
        field_rows.append(
            {
                "name": field.name,
                "dtype": "float32",
                "row_shape": row_shape,
                "row_bytes": row_bytes,
                "raw_offset": raw_offset,
                "source_path": str(field.source_path),
                "source_size_bytes": int(field.source_path.stat().st_size),
            }
        )
        raw_offset += row_bytes
    raw_row_bytes = raw_offset

    try:
        with rows_path.open("wb", buffering=16 * 1024 * 1024) as output:
            for scene_index in range(scene_count):
                parts: list[bytes] = []
                for field in fields:
                    row = np.asarray(field.array[scene_index], dtype=np.float32)
                    payload = row.tobytes(order="C")
                    raw_hashes[field.name].update(payload)
                    parts.append(payload)
                raw = b"".join(parts)
                if len(raw) != raw_row_bytes:
                    raise RuntimeError(
                        f"rank {rank} row {scene_index} has {len(raw)} bytes; "
                        f"expected {raw_row_bytes}"
                    )
                frame = compressor.compress(raw)
                output.write(frame)
                compressed_hash.update(frame)
                offsets[scene_index + 1] = int(offsets[scene_index]) + len(frame)
                if progress_interval > 0 and (scene_index + 1) % progress_interval == 0:
                    ratio = int(offsets[scene_index + 1]) / (
                        (scene_index + 1) * raw_row_bytes
                    )
                    print(
                        f"rank {rank}: compressed {scene_index + 1:,}/{scene_count:,} "
                        f"scenes ratio={ratio:.4f}",
                        flush=True,
                    )
            output.flush()
            os.fsync(output.fileno())
        np.save(offsets_path, offsets, allow_pickle=False)
        _fsync_path(offsets_path)
        for row in field_rows:
            row["source_raw_sha256"] = raw_hashes[row["name"]].hexdigest()
        manifest = {
            "version": 1,
            "schema": SCHEMA,
            "rank": rank,
            "scene_count": scene_count,
            "expected_paths_sha256": expected_paths_sha256,
            "codec": "zstd",
            "compression_level": int(compression_level),
            "frame_checksum": True,
            "one_frame_per_scene": True,
            "raw_row_bytes": raw_row_bytes,
            "raw_total_bytes": raw_row_bytes * scene_count,
            "compressed_rows": ROWS_FILE,
            "compressed_size_bytes": int(rows_path.stat().st_size),
            "compressed_sha256": compressed_hash.hexdigest(),
            "offsets": OFFSETS_FILE,
            "offsets_sha256": _sha256_file(offsets_path),
            "source_replay_manifest_sha256": _json_sha256(
                source_rank_dir / "manifest.json"
            ),
            "source_expert_manifest_sha256": _json_sha256(
                source_rank_dir / "expert_anchor_manifest.json"
            ),
            "publication_probe_seed": 910_247 + rank,
            "publication_probe_scene_count": min(scene_count, 259),
            "fields": field_rows,
        }
        _write_json_fsync(manifest_path, manifest)

        # Fail closed before publication: decode boundary rows and compare the
        # exact float32 bytes against every source field.
        probe_reader = CompressedReplayContextReader(
            temporary.parent,
            rank,
            expected_scene_count=scene_count,
            expected_paths_sha256=expected_paths_sha256,
            rank_dir_override=temporary,
        )
        try:
            probe_rng = np.random.default_rng(int(manifest["publication_probe_seed"]))
            random_probe = probe_rng.choice(
                scene_count,
                size=min(scene_count, 256),
                replace=False,
            )
            probe = np.asarray(
                sorted(
                    {
                        0,
                        scene_count // 2,
                        scene_count - 1,
                        *(int(value) for value in random_probe),
                    }
                )
            )
            manifest["publication_probe_scene_count"] = int(probe.size)
            _write_json_fsync(manifest_path, manifest)
            decoded = probe_reader.materialize(probe)
            for field in fields:
                expected = np.asarray(field.array[probe], dtype=np.float32)
                actual = decoded[field.name]
                if expected.tobytes(order="C") != actual.tobytes(order="C"):
                    raise RuntimeError(
                        f"compressed context probe differs for {field.name}"
                    )
        finally:
            probe_reader.close()
        _fsync_path(temporary)
        os.replace(temporary, rank_dir)
        _fsync_path(output_root)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


class CompressedReplayContextReader:
    """Random row reader for a strict compressed replay-context sidecar."""

    def __init__(
        self,
        root: Path,
        rank: int,
        *,
        expected_scene_count: int,
        expected_paths_sha256: str,
        rank_dir_override: Path | None = None,
        workers: int = 1,
    ) -> None:
        _require_zstd()
        self.rank = int(rank)
        self.rank_dir = (
            Path(rank_dir_override)
            if rank_dir_override is not None
            else Path(root).expanduser().resolve() / f"rank_{self.rank:04d}"
        )
        manifest_path = self.rank_dir / MANIFEST_FILE
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("schema") != SCHEMA:
            raise RuntimeError(
                f"unsupported compressed replay schema {self.manifest.get('schema')!r}"
            )
        self.scene_count = int(self.manifest.get("scene_count", -1))
        if self.scene_count != int(expected_scene_count):
            raise RuntimeError(
                "compressed replay scene count differs: "
                f"{self.scene_count} != {expected_scene_count}"
            )
        if self.manifest.get("expected_paths_sha256") != expected_paths_sha256:
            raise RuntimeError("compressed replay scene order/hash differs")
        self.raw_row_bytes = int(self.manifest.get("raw_row_bytes", -1))
        if self.raw_row_bytes <= 0:
            raise RuntimeError("compressed replay has an invalid raw row size")
        rows_path = self.rank_dir / str(self.manifest.get("compressed_rows", ""))
        offsets_path = self.rank_dir / str(self.manifest.get("offsets", ""))
        if not rows_path.is_file() or not offsets_path.is_file():
            raise RuntimeError("compressed replay sidecar is incomplete")
        if int(rows_path.stat().st_size) != int(
            self.manifest.get("compressed_size_bytes", -1)
        ):
            raise RuntimeError("compressed replay row-file size differs from manifest")
        if _sha256_file(offsets_path) != self.manifest.get("offsets_sha256"):
            raise RuntimeError("compressed replay offsets checksum differs")
        self.offsets = np.load(offsets_path, mmap_mode="r")
        if self.offsets.dtype != np.uint64 or self.offsets.shape != (
            self.scene_count + 1,
        ):
            raise RuntimeError(
                f"invalid compressed replay offsets {self.offsets.dtype} "
                f"{self.offsets.shape}"
            )
        if int(self.offsets[0]) != 0 or int(self.offsets[-1]) != rows_path.stat().st_size:
            raise RuntimeError("compressed replay offsets do not span the row file")
        if not bool(np.all(self.offsets[1:] > self.offsets[:-1])):
            raise RuntimeError("compressed replay offsets are not strictly increasing")
        self.fields = list(self.manifest.get("fields", []))
        expected_names = {
            "scene_encoding",
            "ego_current_state",
            "neighbor_agents_past",
            "neighbor_agents_future",
            "expert_trajectories",
            "expert_rewards",
            "expert_safe",
        }
        if {str(field.get("name")) for field in self.fields} != expected_names:
            raise RuntimeError("compressed replay fields are incomplete")
        final_end = 0
        for field in self.fields:
            if field.get("dtype") != "float32":
                raise RuntimeError("compressed replay field is not float32")
            start = int(field.get("raw_offset", -1))
            size = int(field.get("row_bytes", -1))
            if start != final_end or size <= 0:
                raise RuntimeError("compressed replay field layout is not contiguous")
            final_end = start + size
        if final_end != self.raw_row_bytes:
            raise RuntimeError("compressed replay field layout has a wrong row size")
        self._fd = os.open(rows_path, os.O_RDONLY)
        self._decompressor = zstd.ZstdDecompressor()
        self.workers = max(1, int(workers))
        self._thread_local = threading.local()
        self._executor = (
            ThreadPoolExecutor(
                max_workers=self.workers,
                thread_name_prefix=f"replay-zstd-r{self.rank}",
            )
            if self.workers > 1
            else None
        )

    def close(self) -> None:
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        fd = getattr(self, "_fd", None)
        if fd is not None:
            os.close(fd)
            self._fd = None
        mmap = getattr(self.offsets, "_mmap", None)
        if mmap is not None:
            mmap.close()

    def _read_row(self, scene_index: int) -> bytes:
        start = int(self.offsets[scene_index])
        end = int(self.offsets[scene_index + 1])
        frame = os.pread(self._fd, end - start, start)
        if len(frame) != end - start:
            raise RuntimeError(
                f"short compressed replay read for scene {scene_index}"
            )
        if self._executor is None:
            decompressor = self._decompressor
        else:
            decompressor = getattr(self._thread_local, "decompressor", None)
            if decompressor is None:
                decompressor = zstd.ZstdDecompressor()
                self._thread_local.decompressor = decompressor
        raw = decompressor.decompress(frame, max_output_size=self.raw_row_bytes)
        if len(raw) != self.raw_row_bytes:
            raise RuntimeError(
                f"scene {scene_index} decoded to {len(raw)} bytes; "
                f"expected {self.raw_row_bytes}"
            )
        return raw

    def materialize(self, indices: np.ndarray | list[int]) -> dict[str, np.ndarray]:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or indices.size <= 0:
            raise ValueError("compressed replay indices must be non-empty 1-D")
        if int(indices.min()) < 0 or int(indices.max()) >= self.scene_count:
            raise IndexError("compressed replay index is out of range")
        outputs = {
            str(field["name"]): np.empty(
                (indices.size, *[int(value) for value in field["row_shape"]]),
                dtype=np.float32,
            )
            for field in self.fields
        }
        unique_indices, inverse = np.unique(indices, return_inverse=True)
        if self._executor is None:
            unique_rows = [self._read_row(int(value)) for value in unique_indices]
        else:
            unique_rows = list(
                self._executor.map(
                    self._read_row,
                    (int(value) for value in unique_indices),
                )
            )
        for output_index, unique_position in enumerate(inverse):
            raw = unique_rows[int(unique_position)]
            view = memoryview(raw)
            for field in self.fields:
                name = str(field["name"])
                start = int(field["raw_offset"])
                end = start + int(field["row_bytes"])
                values = np.frombuffer(view[start:end], dtype=np.float32).reshape(
                    tuple(int(value) for value in field["row_shape"])
                )
                outputs[name][output_index] = values
        return outputs
