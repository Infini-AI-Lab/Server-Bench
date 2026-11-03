"""Where final-verdict evidence may be written, and where it is refused.

The repository is not bind-mounted into any agent container, so evidence kept
under it is private by construction. The one agent-reachable place on the host
is a run WORKSPACE, which is mounted read-write into a container the agent
controls; `refuse_if_agent_reachable` refuses (rather than relocating) a write
under one, because a final result an agent can read is a channel to tune
against.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Files that mark a directory as an agent-controlled run workspace.
WORKSPACE_MARKERS = ("serve.sh", "start_server.sh", "TASK.md", "AGENT.md",
                     ".workspace", "wait")

#: Directory names that are agent-reachable roots by construction.
AGENT_REACHABLE_PARTS = ("workspace", "evolve-scratch")


class NotPrivate(RuntimeError):
    """Final evidence was about to be written where the coding agent can read it."""


def default_root(repo: Path) -> Path:
    """Where final-verdict evidence lives: `SERVERBENCH_FINAL_RESULTS`, else
    `serverbench/artifacts/final_validation` in the (unmounted) repository."""
    override = os.environ.get("SERVERBENCH_FINAL_RESULTS", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (Path(repo) / "serverbench" / "artifacts" / "final_validation").resolve()


def refuse_if_agent_reachable(path: Path) -> None:
    """Raise unless `path` and every ancestor are outside any run workspace."""
    resolved = Path(path).expanduser().resolve()
    for parent in (resolved, *resolved.parents):
        if parent.name in AGENT_REACHABLE_PARTS:
            raise NotPrivate(
                f"{resolved} lies under {parent}, which is bind-mounted into "
                f"the agent's container read-write")
        if not parent.is_dir():
            continue
        present = [m for m in WORKSPACE_MARKERS if (parent / m).exists()]
        if len(present) >= 2:
            raise NotPrivate(
                f"{resolved} lies inside what looks like a run workspace "
                f"({parent}, carrying {present})")
