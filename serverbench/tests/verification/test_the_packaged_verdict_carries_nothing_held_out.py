"""The evaluator copy a task ships carries nothing held out.

Each live task ships `tests/final_verdict/`, a byte-identical copy of the
harness's `final_validate`, so the agent can run the verdict's own replay and
protocol checks against its server. The output-correctness stage is private:
its code and data stay on the host. This checks the package did not acquire
held-out-shaped files anyway, which would be a leak with no purpose.

Harbor's verifier does not run the grader -- each task's `tests/test.sh`
refuses, and the verdict runs on the host.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from paths import released_tasks

TASKS = released_tasks()


def test_there_are_tasks_to_check() -> None:
    assert TASKS


@pytest.mark.parametrize("task_dir", TASKS, ids=lambda p: p.name)
def test_the_export_carries_no_held_out_evidence(task_dir: Path) -> None:
    grader = task_dir / "tests" / "final_verdict"
    assert grader.is_dir(), f"{task_dir.name} ships no tests/final_verdict"
    leaked = [p.relative_to(grader).as_posix()
              for p in grader.rglob("*")
              if p.is_file() and (
                  p.name.startswith("references-")
                  or p.name.startswith("consensus-")
                  or p.name.startswith("trace-")
                  or p.name.startswith("gsm8k-")
                  or p.name == "private_correctness.py"
                  or "qualification" in p.name
                  or p.name.endswith("-final.json"))]
    assert not leaked, (
        f"{task_dir.name}'s exported grader carries held-out-shaped files: "
        f"{leaked}")
