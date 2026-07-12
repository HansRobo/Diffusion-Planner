"""Stable filesystem and logging names for classification-provided labels."""

from __future__ import annotations

import hashlib
import re


def artifact_component(value: str, *, max_length: int = 96) -> str:
    """Return one ASCII path component, preserving uniqueness after sanitization."""
    original = str(value)
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", original).strip("._")
    clean = clean or "unnamed"
    changed = clean != original or len(clean) > max_length
    if changed:
        digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:10]
        prefix_length = max(max_length - len(digest) - 1, 1)
        clean = f"{clean[:prefix_length]}_{digest}"
    return clean
