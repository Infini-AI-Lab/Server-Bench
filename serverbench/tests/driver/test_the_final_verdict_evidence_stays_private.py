"""The private final-verdict evidence must stay out of the agent's reach.

`final_validate.store` refuses an agent-reachable path, the default root is
outside every workspace, and the agent container's own mount builder is read
rather than trusted to have been kept in step. Those hold whatever selects the
candidate, and they are why a final outcome cannot become optimisation feedback.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "serverbench" / "harness"))

from final_validate.store import (  # noqa: E402
    NotPrivate,
    default_root,
    refuse_if_agent_reachable,
)


def test_the_default_result_root_is_outside_every_workspace() -> None:
    root = default_root(REPO)
    assert "workspace" not in root.parts
    refuse_if_agent_reachable(root)     # raises if it is reachable


def test_a_result_under_a_workspace_is_refused_not_relocated(
        tmp_path: Path) -> None:
    """A workspace is bind-mounted read-write into a container the agent
    controls. Silently relocating the write would leave the caller believing it
    landed where it asked."""
    workspace = tmp_path / "run" / "workspace"
    (workspace / "tests").mkdir(parents=True)
    (workspace / "AGENT_TASK.md").write_text("x")
    with pytest.raises(NotPrivate):
        refuse_if_agent_reachable(workspace / "final.json")


def test_a_private_root_is_accepted(tmp_path: Path) -> None:
    """The refusal must not be doing its job by refusing everything."""
    refuse_if_agent_reachable(tmp_path / "private" / "some_task" / "published")


def test_the_agent_container_never_mounts_the_results_root() -> None:
    """Read from the real mount builder, not from a list kept in step by hand."""
    import sys
    sys.path.insert(0, str(REPO / "serverbench" / "scripts" / "agent"))
    import harbor_env

    mounts = harbor_env.mounts_for(
        workspace=REPO / "nonexistent-workspace",
        model=REPO / "serverbench/artifacts/models/qwen3-4b",
        env_input=REPO / "serverbench/artifacts/env-input",
        uv_cache=REPO / "nonexistent-uv-cache")
    results = str(default_root(REPO))
    for mount in mounts:
        source = str(Path(mount["source"]).resolve())
        assert not results.startswith(source + "/") and source != results, (
            f"the agent container mounts {source}, which contains the private "
            f"final-validation results at {results}")
