"""Batch post-training toolchain: declarative specs in, training runs out.

    rlvr/posttrain/objectives.py  candidate-weighting objectives (awr | grpo | dpo)
    rlvr/posttrain/spec.py        spec loading, inheritance, sweep expansion
    rlvr/posttrain/flags.py       spec -> train_awr argv, validated against its parser
    rlvr/posttrain/runner.py      launch one spec, or a queue of them, on N GPUs
    rlvr/posttrain/presets/       the arms worth reproducing

    python -m rlvr.posttrain list|show|plan|run presets/dual.json
"""

from rlvr.posttrain.objectives import available, register, resolve

__all__ = ["available", "register", "resolve"]
