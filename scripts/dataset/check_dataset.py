"""Check that every frame of a Parquet index can be loaded through the DataLoader."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import argcomplete
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from diffusion_planner.data import PlannerDataset


@dataclass(frozen=True)
class Verdict:
    """Outcome of loading one frame, small enough to send back from a worker."""

    index: int
    reason: str | None
    signature: tuple[tuple[str, tuple[int, ...]], ...] = ()

    @property
    def ok(self) -> bool:
        """Whether the frame was built."""
        return self.reason is None


class FrameCheck(Dataset):
    """Load each frame in a worker and return only its verdict.

    Passing the tensors themselves back to the main process would move gigabytes for no
    reason, so the frames are inspected where they are built.
    """

    def __init__(self, dataset: PlannerDataset) -> None:
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Verdict:
        """Build one frame and describe what came out of it."""
        try:
            frame = self._dataset[index]
        except Exception as error:  # a broken bag must not abort the whole check
            return Verdict(index, f"{type(error).__name__}: {error}")
        if frame is None:
            return Verdict(index, "create_frame_data returned None")
        signature = tuple(sorted((key, tuple(value.shape)) for key, value in frame.items()))
        return Verdict(index, None, signature)


def main() -> None:
    """Walk the whole index and stop at the first frame that fails to load."""
    parser = argparse.ArgumentParser(
        description="Load every frame of a frame-index parquet and fail on the first error"
    )
    parser.add_argument("parquet", type=Path, help="frame index to check")
    parser.add_argument(
        "--jobs", type=int, default=8, help="worker processes (default: %(default)s)"
    )
    parser.add_argument("--limit", type=int, default=None, help="check only the first N frames")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if args.jobs < 0:
        parser.error("--jobs must not be negative")
    dataset = PlannerDataset(args.parquet, map_capacity=10000, reader_capacity=40000)

    checked = FrameCheck(dataset)
    total = len(checked) if args.limit is None else min(args.limit, len(checked))
    print(f"checking {total} of {len(dataset)} frames from {args.parquet}")

    loader = DataLoader(
        checked,
        batch_size=32,  # only controls how often progress is reported
        shuffle=False,
        num_workers=args.jobs,
        collate_fn=list,
        sampler=range(total),
    )

    signatures: Counter[tuple[tuple[str, tuple[int, ...]], ...]] = Counter()
    with tqdm(total=total, desc="checking", unit="frame", smoothing=0.0) as progress:
        for batch in loader:
            for verdict in batch:
                if verdict.ok:
                    signatures[verdict.signature] += 1
                    continue
                bag_path, frame_time_ns = dataset.source(verdict.index)
                print(f"\n{bag_path} @ {frame_time_ns}: {verdict.reason}", file=sys.stderr)
                sys.exit(1)
            progress.update(len(batch))

    print(f"\n{total}/{total} frames loaded")
    if len(signatures) > 1:
        print(f"WARNING: {len(signatures)} different tensor layouts in one index")
        for signature, count in signatures.most_common():
            print(f"  {count} frames: {dict(signature)}")


if __name__ == "__main__":
    main()
