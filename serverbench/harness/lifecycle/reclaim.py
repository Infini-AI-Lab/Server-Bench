"""Give back a finished run's disk without giving back its evidence.

## Why this exists

Finished runs accumulate on disk until a measurement dies of
`OSError: [Errno 28] No space left on device`. Almost all of it is one
directory per run:

    workspace/.venv     gigabytes  torch, transformers, the wheelhouse install
    workspace/agentx    hundreds of MB of raw aiperf exports

Every `.venv` is a copy of the same packages, rebuildable in minutes from
`/opt/env-input/wheelhouse`, which is the only place the agent was allowed to
install from in the first place.

## What is disposable, and what is never

THE LINE IS REPRODUCIBILITY, NOT SIZE. A run's evidence is the record of what
happened and cannot be regenerated at any price; its environment is a build
artefact of a frozen input list.

    DISPOSABLE          .venv, __pycache__, .pytest_cache, pip/uv/HF caches
    EVIDENCE            journal.jsonl, SUMMARY.json, TRAJECTORY.json,
                        trajectory.atif.json, identity.json, agent.*.log,
                        agent.stream.jsonl, container.json,
                        harbor_environment.json, egress*, firewall.log,
                        evidence/, profiles/, and the whole of the agent's
                        source tree

The agent's source is also already published to `server_prototypes/` as one
version per accepted boundary, so the workspace copy is a convenience, not the
record -- but it is small, and it is kept anyway.

## The secret this also clears

`lifecycle/credentials.py` stages the operator's OAuth token at
`<run>/secrets/.credentials.json` so a host refresh can reach the container.
A FINISHED run has no use for it and a live token should not outlive the thing
that needed it. It is overwritten before it is unlinked -- a token recovered
from a deleted inode is still a token.

## What this never does

Touch a RUNNING run. `reclaimable()` refuses any run whose driver is alive or
whose state is not terminal, because an agent mid-round needs its `.venv` more
than the disk needs the gigabytes.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

#: Regenerable from `/opt/env-input/wheelhouse` and a frozen requirements list.
DISPOSABLE_DIRS = (".venv", "__pycache__", ".pytest_cache", ".mypy_cache",
                   ".ruff_cache", ".cache")

#: Run states after which nothing else will be written.
TERMINAL = ("COMPLETED", "TIMEBOX_EXPIRED", "FAILED", "STOPPED_EARLY",
            "CANCELLED")


def _run_state(run_root: Path) -> str:
    """The last state this run reached, from its own journal."""
    summary = run_root / "SUMMARY.json"
    if summary.is_file():
        try:
            return str(json.loads(summary.read_text("utf-8")).get("state", ""))
        except (OSError, ValueError):
            pass
    journal = run_root / "journal.jsonl"
    if not journal.is_file():
        return ""
    state = ""
    try:
        for line in journal.read_text("utf-8").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("event") == "run_state" and event.get("to"):
                state = str(event["to"])
    except OSError:
        return ""
    return state


def _driver_alive(run_root: Path) -> bool:
    """Whether the process that owns this run is still running.

    Read from `identity.json`'s recorded pid where there is one. A pid that has
    been reused by an unrelated process would make this over-cautious, which is
    the right direction to be wrong in.
    """
    identity = run_root / "identity.json"
    if not identity.is_file():
        return False
    try:
        pid = int((json.loads(identity.read_text("utf-8")) or {}).get("pid") or 0)
    except (OSError, ValueError, TypeError):
        return False
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def reclaimable(run_root: Path) -> tuple[bool, str]:
    """Whether this run's disposable bulk may be removed, and why not."""
    run_root = Path(run_root)
    if not run_root.is_dir():
        return False, "not a directory"
    if _driver_alive(run_root):
        return False, "its driver is still running"
    state = _run_state(run_root)
    if state and state not in TERMINAL:
        return False, f"state is {state}, which is not terminal"
    if not state and not (run_root / "SUMMARY.json").is_file():
        # No journal and no summary: this may be a run being set up right now.
        return False, "no terminal state recorded; it may still be starting"
    return True, ""


def shred_secrets(run_root: Path) -> int:
    """Overwrite and remove a finished run's staged credentials. Bytes freed.

    Overwritten first: a live token recovered from a deleted inode is still a
    live token, and this file holds one.
    """
    secrets = Path(run_root) / "secrets"
    if not secrets.is_dir():
        return 0
    freed = 0
    for path in secrets.rglob("*"):
        if not path.is_file():
            continue
        try:
            size = path.stat().st_size
            with open(path, "r+b") as handle:
                handle.write(b"\0" * size)
                handle.flush()
                os.fsync(handle.fileno())
            path.unlink()
            freed += size
        except OSError:
            continue
    with suppress_oserror():
        secrets.rmdir()
    return freed


class suppress_oserror:                                        # noqa: N801
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return exc_type is not None and issubclass(exc_type, OSError)


def _size_of(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def reclaim(run_root: Path, *, dry_run: bool = False) -> dict:
    """Remove a finished run's disposable bulk. Returns what was freed."""
    run_root = Path(run_root)
    ok, why = reclaimable(run_root)
    if not ok:
        return {"run": run_root.name, "skipped": why, "bytes": 0, "paths": []}

    freed, removed = 0, []
    for name in DISPOSABLE_DIRS:
        for path in run_root.rglob(name):
            if not path.is_dir() or path.is_symlink():
                continue
            size = _size_of(path)
            removed.append({"path": str(path.relative_to(run_root)),
                            "bytes": size})
            freed += size
            if not dry_run:
                shutil.rmtree(path, ignore_errors=True)
    if not dry_run:
        secret_bytes = shred_secrets(run_root)
        if secret_bytes:
            removed.append({"path": "secrets/ (overwritten, then removed)",
                            "bytes": secret_bytes})
            freed += secret_bytes
    return {"run": run_root.name, "skipped": "", "bytes": freed,
            "paths": removed}

