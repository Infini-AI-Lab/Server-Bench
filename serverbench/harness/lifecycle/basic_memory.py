"""One plain `memory.md` per lineage, taken back from the agent's workspace.

The agent may keep factual notes in `memory.md` in its workspace. At session
end the harness persists that file into the lineage directory, for the
operator. It is never delivered to a later agent: no previous evolve is visible
inside a workspace. There is no memory agent, ranking, selector or attribution
model -- the agent owns the content, and the harness owns only that it survives.

ATOMIC, AND WITH ORDINARY HISTORY
----------------------------------
Writes go through a temp file and `os.replace`, so a crash mid-write leaves the
previous memory intact rather than half of two. Every write also copies the
previous content to `memory.history/`, which is ordinary file history and not a
version store: it exists so that an agent that deletes everything has not
destroyed the lineage's record.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
import time
from pathlib import Path

#: The documented bound. 32 KiB is roughly 8,000 tokens: large enough for a
#: lineage's real findings. A persisted file over it is reported as
#: `over_bound`, not cut.
MAX_BYTES = 32 * 1024

FILENAME = "memory.md"
HISTORY_DIR = "memory.history"


def path_for(lineage_dir: Path) -> Path:
    return Path(lineage_dir) / FILENAME


def write(lineage_dir: Path, text: str) -> Path:
    """Replace the memory atomically, keeping the previous version in history."""
    path = path_for(lineage_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        history = path.parent / HISTORY_DIR
        history.mkdir(exist_ok=True)
        # Millisecond resolution AND a collision guard: two writes inside one
        # second must leave two history entries, not one.
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        stamp = f"{stamp}.{int(time.time() * 1000) % 1000:03d}Z"
        keep = history / f"{stamp}-{FILENAME}"
        suffix = 1
        while keep.exists():
            keep = history / f"{stamp}-{suffix}-{FILENAME}"
            suffix += 1
        shutil.copy2(path, keep)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-memory-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return path


def collect(workspace: Path, lineage_dir: Path) -> dict:
    """Take back what the agent wrote, at the end of a session.

    The agent owns the content. The harness owns only that it is persisted
    atomically and that the previous version is kept.
    """
    source = Path(workspace) / FILENAME
    if not source.is_file():
        return {"persisted": False,
                "why": f"the agent did not write {FILENAME}"}
    text = source.read_text("utf-8", errors="replace")
    write(lineage_dir, text)
    return {"persisted": True, "bytes": len(text.encode("utf-8")),
            "over_bound": len(text.encode("utf-8")) > MAX_BYTES}
