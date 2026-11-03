"""Helpers with no phase of their own.

Things that every phase does and none of them owns. They live here rather
than in `record.py` because a module that decides nothing should not be reached
through one that decides everything, and because one `os.replace` helper keeps
a single idea of what "written" means.

Nothing here reads a task, moves a state, or takes a lock. If a helper needs to
do any of those it is not a helper; it belongs to the phase that does it.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

#: The harness package root, so a module can reach `serving_evolve` and
#: `final_validate` without copies of the same `parents[1]` arithmetic.
HARNESS = Path(__file__).resolve().parents[1]


def ensure_harness_on_path() -> None:
    """Put `harness/` on `sys.path`, once.

    The lifecycle imports `serving_evolve` and is imported by files that are
    run from several directories, so it cannot rely on the caller having set
    this up. Idempotent: a second call is a no-op, not a second entry.
    """
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))


def write_atomically(path: Path, text: str) -> None:
    """Write so a reader sees the whole document or the previous one.

    A run's evidence is read by a person mid-run and by the next process after
    a crash, and a truncated file is indistinguishable from a defect in
    whatever wrote it. `os.replace` is atomic within a filesystem; the fsync is
    what makes it survive the machine going away rather than just the process.

    The temporary name carries the real suffix so a half-written head is not
    picked up by a glob for `*.json` and read as the head.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def write_json_atomically(path: Path, body: object) -> None:
    """`write_atomically` for the one JSON shape this package writes."""
    write_atomically(path, json.dumps(body, indent=1, sort_keys=True) + "\n")


def describe(exc: BaseException) -> str:
    """One line that says WHICH failure this was, and where.

    `repr(exc)` gives `PermissionError(13, 'Permission denied')`: the class
    and the errno, but not the file that could not be written. So this adds
    the errno and filename when the OS supplied them, and the innermost frame,
    which names the line that raised rather than the handler that caught it.
    """
    import traceback

    parts = [type(exc).__name__]
    errno = getattr(exc, "errno", None)
    if errno is not None:
        parts.append(f"errno {errno}")
    for attr in ("filename", "filename2"):
        name = getattr(exc, attr, None)
        if name:
            parts.append(str(name))
    text = str(exc)
    if text and text not in parts:
        parts.append(text)
    stack = traceback.extract_tb(exc.__traceback__)
    if stack:
        last = stack[-1]
        parts.append(f"at {Path(last.filename).name}:{last.lineno} "
                     f"in {last.name}")
    return " | ".join(parts)[:600]
