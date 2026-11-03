"""No file in a task is outside every check.

`taskctl check` covers a task three ways: GENERATED trees (`tests/final_verdict/`,
`tests/replay-verify`) are rebuilt from the harness; AUTHORED text files (by
suffix, see `sites.authored_files`) are read back or regenerated; and the few
files with neither -- listed in `UNMANAGED_BUT_CHECKED` -- each have a named
check. This is the ratchet: a new file of an unexpected kind fails here instead
of going unchecked.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
MANAGE = REPO / "serving_tasks" / "_manage"
TASKS = REPO / "serving_tasks"

#: Files that are neither generated nor prose, and the check that covers each.
#: An entry here is a promise that `taskctl check` reads the file; the tests
#: below prove the check actually fires.
UNMANAGED_BUT_CHECKED = {
    "environment/Dockerfile": ("image_drift",
                               "the AgentX commit the task image grades with"),
    "environment/agent-environment": ("agent_environment_drift",
                                      "the task's own agent image and wheelhouse agree"),
}

#: Build artefacts. Never in git, and excluded from every workspace a task
#: builds.
NOISE = ("__pycache__", ".pytest_cache")


@pytest.fixture(scope="module")
def sites():
    assert (MANAGE / "sites.py").is_file(), MANAGE
    for path in (str(MANAGE), str(REPO / "serverbench" / "harness")):
        if path not in sys.path:
            sys.path.insert(0, path)
    import sites as module
    return module


def _tasks() -> list[str]:
    if not TASKS.is_dir():
        return []
    return sorted(p.name for p in TASKS.glob("*") if (p / "contract.yaml").is_file())


@pytest.mark.parametrize("task", _tasks())
def test_every_file_falls_into_a_managed_class(sites, task) -> None:
    root = sites.task_dir(task)
    authored = {p.resolve() for p in sites.authored_files(task)}

    unclassified = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(n in path.parts for n in NOISE):
            continue
        rel = path.relative_to(root).as_posix()
        if sites.is_generated(path, root) or path.resolve() in authored:
            continue
        if rel in UNMANAGED_BUT_CHECKED:
            continue
        unclassified.append(rel)

    assert not unclassified, (
        f"{task} carries file(s) that are neither generated, nor read back by "
        f"`taskctl check`, nor listed as checked-in-place: {unclassified}. A "
        f"file in a task is part of its contract; one that nothing compares "
        f"with anything is one that can say whatever it likes.\n"
        f"Either give it a prose suffix so `authored_files` picks it up, put "
        f"it under a generated tree, or add it to UNMANAGED_BUT_CHECKED with "
        f"a named check in `taskctl`.")


def test_build_caches_never_reach_the_agents_workspace() -> None:
    """Excluded at the COPY, not forbidden in the tree.

    This suite imports a task's packaged verdict, so caches appear in the tree
    whenever it runs; what matters is that they never travel into the
    workspace the agent reads. A `.pytest_cache` would name the host's test
    ids to the agent.
    """
    ws = pytest.importorskip("lifecycle.workspace")
    assert set(getattr(ws, "BUILD_NOISE", ())) >= {"__pycache__",
                                                   ".pytest_cache"}, (
        "the workspace builder does not exclude both build-cache kinds, so "
        "they reach the agent")
    for noisy in ("t/tests/final_verdict/__pycache__/scenario.pyc",
                  "t/.pytest_cache/v/cache/nodeids"):
        assert ws._is_build_noise(Path(noisy)), f"{noisy} would be delivered"
    assert not ws._is_build_noise(Path("t/tests/test_public.py")), (
        "the exclusion is too broad and would drop a real task file")


@pytest.mark.parametrize("name", sorted(UNMANAGED_BUT_CHECKED))
def test_the_check_for_each_actually_fires(name) -> None:
    """The named check exists, is run by `check`, and names the file."""
    import ast

    ctl = MANAGE / "taskctl.py"
    assert ctl.is_file(), ctl
    check, _ = UNMANAGED_BUT_CHECKED[name]
    tree = ast.parse(ctl.read_text("utf-8"))
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert check in functions, f"taskctl has no {check}()"
    assert name.rsplit("/", 1)[-1] in ast.unparse(functions[check])
    assert f"{check}(" in ast.unparse(functions["problems_of"]), (
        f"`taskctl check` does not run {check}()")


def test_the_pin_constant_it_compares_against_exists() -> None:
    """Name the constant, so a rename cannot make the check inert."""
    pin = pytest.importorskip("final_validate.agentx_pin")
    assert getattr(pin, "PINNED_COMMIT", ""), (
        "agentx_pin declares no PINNED_COMMIT. taskctl compares the task "
        "image's AgentX build against it; without it the comparison is inert "
        "and a task could grade with a different benchmark than the one that "
        "measured its baselines.")
