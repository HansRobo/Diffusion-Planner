"""Build one or more lossless row-addressable replay-context sidecars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlvr.replay_context_codec import build_compressed_replay_context_rank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-replay", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rank", type=int, action="append", required=True)
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--progress-interval", type=int, default=10_000)
    args = parser.parse_args()

    summaries = []
    for rank in args.rank:
        manifest = build_compressed_replay_context_rank(
            args.source_replay,
            args.output_root,
            rank,
            compression_level=args.compression_level,
            progress_interval=args.progress_interval,
        )
        summaries.append(
            {
                "rank": int(manifest["rank"]),
                "scene_count": int(manifest["scene_count"]),
                "raw_total_bytes": int(manifest["raw_total_bytes"]),
                "compressed_size_bytes": int(manifest["compressed_size_bytes"]),
                "compression_ratio": float(manifest["compressed_size_bytes"])
                / float(manifest["raw_total_bytes"]),
            }
        )
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
