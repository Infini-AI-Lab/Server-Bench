"""Where the repository is, resolved once instead of counted per file.

Counting directories up from a test's own `__file__` (`parents[1]`,
`parents[2]`) works only while every test file sits at the same depth; when one
moves, the count still resolves, just to the wrong directory, so a test looking
for `serving_tasks/` reports a missing task rather than a broken path.

`tests/` is on `sys.path` (see conftest.py), so these names import from anywhere
under it, at any depth, and moving a test file is free.
"""

from __future__ import annotations

from pathlib import Path

#: `serverbench/` -- the harness, its scripts and its tests.
SERVERBENCH = Path(__file__).resolve().parent.parent

#: The repository root: `serving_tasks/`, `server_output/`, `evaluation_progress/`.
REPO = SERVERBENCH.parent

#: This directory, for fixtures and helper modules that live beside the tests.
TESTS = Path(__file__).resolve().parent

#: The two files a task's data lives in. Tests read them through the helpers
#: below rather than parsing yaml each, so a change of shape is one edit here.
CONTRACT_FILE = "contract.yaml"
REFERENCE_FILE = "reference.yaml"


def contract(task) -> dict:
    """One task's frozen bars: gates, prices, devices, window."""
    import yaml

    where = task if isinstance(task, Path) else REPO / "serving_tasks" / task
    return yaml.safe_load((where / CONTRACT_FILE).read_text("utf-8")) or {}


def reference(task) -> dict:
    """One task's measured evidence, or {} when it has none yet."""
    import yaml

    where = task if isinstance(task, Path) else REPO / "serving_tasks" / task
    path = where / REFERENCE_FILE
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text("utf-8")) or {}


def task_module(task_name: str):
    """A task's own module -- identity, scenario, corpus, seed, durations."""
    import sys

    if str(SERVERBENCH / "harness") not in sys.path:
        sys.path.insert(0, str(SERVERBENCH / "harness"))
    from final_validate import tasks as task_modules
    return task_modules.python_module(task_name)


def released_tasks() -> list:
    """The live `serving_tasks/` directories: those that carry a contract.

    A directory with a `contract.yaml` is a task and one without is not; a
    separate `state:` field would be a second answer to the same question.
    Every caller asserts the set is non-empty, so a typo cannot make a whole
    module vacuously green.
    """
    return [task for task in sorted(
        (REPO / "serving_tasks").glob("*"))
        if task.is_dir() and (task / CONTRACT_FILE).is_file()]
