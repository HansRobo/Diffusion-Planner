"""Benchmark on-the-fly rosbag dataset loading through a training-style DataLoader."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import argcomplete
from torch.utils.data import DataLoader, RandomSampler, Subset
from tqdm import tqdm

from diffusion_planner.data import PlannerDataset
from diffusion_planner.data.planner_dataset import collate_frames


def benchmark_loading(
    dataset: PlannerDataset,
    *,
    jobs: int,
    batch_size: int,
    warmup_batches: int,
    limit: int | None,
    shuffle: bool,
) -> None:
    """Measure preprocessing, collation, and worker-transfer throughput."""
    total = len(dataset) if limit is None else min(len(dataset), limit)
    source = dataset if shuffle else Subset(dataset, range(total))
    sampler = (
        RandomSampler(dataset, replacement=False, num_samples=total)
        if shuffle
        else None
    )
    loader = DataLoader(
        source,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=jobs,
        collate_fn=collate_frames,
        persistent_workers=jobs > 0,
    )

    print(
        f"benchmarking {total} frames with batch_size={batch_size}, jobs={jobs}, "
        f"shuffle={shuffle}"
    )
    started = time.perf_counter()
    measured_started: float | None = started if warmup_batches == 0 else None
    measured_input_frames = 0
    measured_output_frames = 0
    measured_batches = 0
    output_frames = 0
    first_batch_seconds: float | None = None

    with tqdm(total=total, desc="benchmark", unit="frame", smoothing=0.0) as progress:
        for batch_index, batch in enumerate(loader):
            input_frames = min(batch_size, total - progress.n)
            batch_frames = 0 if batch is None else len(next(iter(batch.values())))
            output_frames += batch_frames
            now = time.perf_counter()
            if first_batch_seconds is None:
                first_batch_seconds = now - started
            if batch_index >= warmup_batches:
                measured_input_frames += input_frames
                measured_output_frames += batch_frames
                measured_batches += 1
            elif batch_index + 1 == warmup_batches:
                measured_started = now
            progress.update(input_frames)

    if measured_started is None or measured_batches == 0:
        raise ValueError(
            "benchmark did not reach a measured batch; increase --limit or reduce "
            "--warmup-batches"
        )

    finished = time.perf_counter()
    measured_elapsed = finished - measured_started
    total_elapsed = finished - started
    input_rate = measured_input_frames / measured_elapsed
    output_rate = measured_output_frames / measured_elapsed
    seconds_per_batch = measured_elapsed / measured_batches
    failed_frames = total - output_frames

    print(f"first batch: {first_batch_seconds:.3f} s")
    print(
        f"total: {total} input frames, {output_frames} output frames in "
        f"{total_elapsed:.3f} s"
    )
    print(
        f"measured: {measured_input_frames} input frames in {measured_elapsed:.3f} s "
        f"({input_rate:.1f} input frames/s, {output_rate:.1f} output frames/s, "
        f"{seconds_per_batch:.3f} s/batch)"
    )
    print(f"failed frames: {failed_frames}")
    if total == len(dataset):
        print(f"full epoch wall time: {total_elapsed / 60:.1f} min")
    else:
        epoch_seconds = len(dataset) / input_rate
        print(f"estimated full epoch: {epoch_seconds / 60:.1f} min")


def main() -> None:
    """Parse benchmark settings and measure dataset loading."""
    parser = argparse.ArgumentParser(
        description="Benchmark training-style preprocessing from a rosbag frame index"
    )
    parser.add_argument("parquet", type=Path, help="frame index to benchmark")
    parser.add_argument(
        "--jobs", type=int, default=8, help="worker processes (default: %(default)s)"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="benchmark only N frames"
    )
    parser.add_argument(
        "--batch-size", type=int, default=512, help="benchmark batch size"
    )
    parser.add_argument(
        "--warmup-batches", type=int, default=2, help="untimed benchmark batches"
    )
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="shuffle frames as training does (default: true)",
    )
    parser.add_argument(
        "--reader-capacity", type=int, default=4, help="open rosbags per worker"
    )
    parser.add_argument(
        "--map-capacity", type=int, default=2, help="parsed maps per worker"
    )
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if args.jobs < 0:
        parser.error("--jobs must not be negative")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.warmup_batches < 0:
        parser.error("--warmup-batches must not be negative")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.reader_capacity < 1:
        parser.error("--reader-capacity must be positive")
    if args.map_capacity < 1:
        parser.error("--map-capacity must be positive")

    dataset = PlannerDataset(
        args.parquet,
        reader_capacity=args.reader_capacity,
        map_capacity=args.map_capacity,
    )
    benchmark_loading(
        dataset,
        jobs=args.jobs,
        batch_size=args.batch_size,
        warmup_batches=args.warmup_batches,
        limit=args.limit,
        shuffle=args.shuffle,
    )


if __name__ == "__main__":
    main()
