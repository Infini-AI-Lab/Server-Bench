"""Canonical serialization and content hashing.

A digest is only useful if the same *meaning* always produces the same bytes.
Two rules make that true here:

1. **Canonical form.** JSON with sorted keys, no insignificant whitespace, UTF-8,
   and no ``NaN``/``Infinity`` (which are not JSON and do not round-trip).
2. **Semantic content only.** Timestamps, absolute paths, host names, run
   identifiers and formatting are *excluded* from a contract's digest by the
   contract itself. Two runs of the same frozen plan on different days must hash
   identically, otherwise the digest measures the clock instead of the plan.

Digests are SHA-256, printed lowercase hex, and never truncated in stored
evidence. Short prefixes appear in human-facing output only.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

#: Keys stripped from any mapping before it is hashed. These describe *when* or
#: *where* something happened, never *what* it says.
VOLATILE_KEYS = frozenset({
    "generated_at", "timestamp", "anchored_at", "created_utc", "started_at",
    "finished_at", "researched_utc", "duration_s", "wall_seconds", "hostname",
    "pid", "artifact_root", "repo_root", "workspace",
})


def canonical_bytes(obj: Any) -> bytes:
    """Serialize to the one canonical byte string for this value."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")


def strip_volatile(obj: Any) -> Any:
    """Recursively drop keys that record when/where rather than what."""
    if isinstance(obj, Mapping):
        return {k: strip_volatile(v) for k, v in obj.items() if k not in VOLATILE_KEYS}
    if isinstance(obj, list | tuple):
        return [strip_volatile(v) for v in obj]
    return obj


def digest(obj: Any, *, semantic: bool = True) -> str:
    """SHA-256 of a value's canonical form.

    With ``semantic=True`` (the default) volatile keys are removed first, so the
    digest binds meaning rather than the moment of writing. Pass ``False`` to
    hash a document exactly as given.
    """
    return hashlib.sha256(canonical_bytes(strip_volatile(obj) if semantic else obj)).hexdigest()


def digest_file(path: str | Path, *, chunk: int = 1 << 20) -> str:
    """Stream a file so a large artifact never has to fit in memory."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def digest_tree(root: str | Path, paths: Iterable[str | Path]) -> str:
    """Digest of a set of files, bound to their relative paths as well as content.

    Path is part of the identity: moving a file changes the tree digest even if
    every byte is unchanged, which is what makes this usable as a source digest.
    """
    root = Path(root)
    h = hashlib.sha256()
    for rel in sorted(str(Path(p)) for p in paths):
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(digest_file(root / rel).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def short(d: str, n: int = 12) -> str:
    """Human-facing prefix. Never store this; store the full digest."""
    return d[:n]
