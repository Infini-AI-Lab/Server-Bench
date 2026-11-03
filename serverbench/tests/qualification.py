"""Whether a task has the measured evidence its checks are about.

A task is qualified once SGLang has been measured on its own hardware and
`reference.yaml` carries that `concurrency_search`. An unmeasured task is a
legitimate state between `taskctl new` and the measurement: it is printed by
`contracts/test_a_task_is_managed_not_hand_edited.py`, and a check that needs
the measured evidence skips it by name.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "serving_tasks"

def missing_evidence(task: str) -> list[str]:
    """What a task still needs before a verdict on it means anything.

    Only the hardware measurement: the SGLang operating point on the task's
    own hardware, which the instruction states as a reference.
    """
    import yaml

    reference = yaml.safe_load(
        (TASKS / task / "reference.yaml").read_text("utf-8")) or {}
    if reference.get("concurrency_search"):
        return []
    return ["reference.yaml has no concurrency_search: SGLang has not been "
            "measured on this task's hardware, so no reference operating "
            "point is stated"]
