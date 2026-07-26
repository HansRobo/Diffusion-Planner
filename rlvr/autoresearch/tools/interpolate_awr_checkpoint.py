#!/usr/bin/env python3
"""Interpolate an AWR checkpoint back toward its original deployment EMA.

The Original-DP AWR runner initializes both the trainable policy and behavior
EMA from the source checkpoint's ``ema_state_dict``.  This utility therefore
uses that exact state as alpha=0 and interpolates the candidate's live and EMA
states independently:

    output = source_ema + alpha * (candidate - source_ema)

It rejects architecture mismatches and differing non-floating buffers, and
records file hashes plus the interpolation contract in the output checkpoint.
The tool only constructs candidates; checkpoint selection must still be done
on an independent fixed scene set.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for raw_key, value in state.items():
        key = raw_key.replace("module.", "").replace("._orig_mod.", ".")
        key = key.replace("_orig_mod.", "")
        if key in result:
            raise ValueError(f"normalised checkpoint key collision: {key}")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state value is not a tensor: {raw_key}={type(value)}")
        result[key] = value.detach().cpu()
    return result


def _source_ema(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("source checkpoint must be a dictionary")
    state = checkpoint.get("ema_state_dict")
    if not isinstance(state, dict):
        raise KeyError("source checkpoint has no ema_state_dict")
    return _normalise_state(state)


def _candidate_state(
    checkpoint: Any, key: str
) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("candidate checkpoint must be a dictionary")
    state = checkpoint.get(key)
    if not isinstance(state, dict):
        raise KeyError(f"candidate checkpoint has no {key}")
    return _normalise_state(state)


def interpolate_state(
    source: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    if source.keys() != candidate.keys():
        missing = sorted(source.keys() - candidate.keys())
        extra = sorted(candidate.keys() - source.keys())
        raise ValueError(
            "checkpoint architecture mismatch: "
            f"missing={missing[:10]} extra={extra[:10]}"
        )

    result: dict[str, torch.Tensor] = {}
    for key, source_value in source.items():
        candidate_value = candidate[key]
        if source_value.shape != candidate_value.shape:
            raise ValueError(
                f"shape mismatch for {key}: "
                f"source={tuple(source_value.shape)} "
                f"candidate={tuple(candidate_value.shape)}"
            )
        if source_value.dtype != candidate_value.dtype:
            raise ValueError(
                f"dtype mismatch for {key}: "
                f"source={source_value.dtype} candidate={candidate_value.dtype}"
            )
        if torch.is_floating_point(source_value):
            mixed = torch.lerp(
                source_value.to(torch.float32),
                candidate_value.to(torch.float32),
                float(alpha),
            )
            result[key] = mixed.to(dtype=source_value.dtype)
        else:
            if not torch.equal(source_value, candidate_value):
                raise ValueError(f"non-floating buffer differs for {key}")
            result[key] = source_value.clone()
    return result


def main() -> None:
    args = _parse_args()
    alpha = float(args.alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    if not args.candidate.is_file():
        raise FileNotFoundError(args.candidate)

    source_checkpoint = torch.load(
        args.source, map_location="cpu", weights_only=False
    )
    candidate_checkpoint = torch.load(
        args.candidate, map_location="cpu", weights_only=False
    )
    source = _source_ema(source_checkpoint)
    live = _candidate_state(candidate_checkpoint, "model")
    ema = _candidate_state(candidate_checkpoint, "ema_state_dict")

    output = {
        "epoch": int(candidate_checkpoint.get("epoch", 0)),
        "model": interpolate_state(source, live, alpha),
        "ema_state_dict": interpolate_state(source, ema, alpha),
        "metrics": candidate_checkpoint.get("metrics", {}),
        "interpolation": {
            "alpha": alpha,
            "formula": "source_ema + alpha * (candidate - source_ema)",
            "source": str(args.source.resolve()),
            "source_sha256": _sha256(args.source),
            "candidate": str(args.candidate.resolve()),
            "candidate_sha256": _sha256(args.candidate),
            "candidate_epoch": int(candidate_checkpoint.get("epoch", 0)),
            "state_tensor_count": len(source),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(
        f"wrote {args.output} alpha={alpha:g} "
        f"source={args.source.name} candidate={args.candidate.name}"
    )


if __name__ == "__main__":
    main()
