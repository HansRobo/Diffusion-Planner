"""Shared command-line parsing helpers."""

import argparse


def boolean(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in ("yes", "true", "t", "y", "1"):
        return True
    if normalized in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def dataclass_default(config_type, name):
    return config_type.__dataclass_fields__[name].default
