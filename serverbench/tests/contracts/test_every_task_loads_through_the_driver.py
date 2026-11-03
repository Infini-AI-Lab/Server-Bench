"""Every live task loads through the contract the driver actually uses.

A suite can be large, green and still miss the FIRST step of the production
path, because unit tests build their subjects from fixtures and fixtures do not
go stale the way contracts do. A task the driver cannot load fails at launch,
after a workspace has been built and cards have been claimed.

So this loads every live task the way `build_workspace` does, from disk,
through the real loader, with no fixtures anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "serverbench" / "harness"))

TASKS_ROOT = REPO / "serving_tasks"


def _live_tasks() -> list[Path]:
    return sorted(p for p in TASKS_ROOT.glob("*")
                  if p.is_dir() and (p / "contract.yaml").is_file())


@pytest.mark.parametrize("task_dir", _live_tasks(), ids=lambda p: p.name)
def test_the_driver_can_load_this_task(task_dir: Path) -> None:
    """The first thing the production path reads, on the real files.

    A task's data is `contract.yaml` plus `reference.yaml`, and what reads the
    frozen parameters is `final_validate.task_spec`. This loads every live task
    through it.
    """
    from final_validate import task_spec

    spec = task_spec.read(task_dir.name)
    assert spec.task_id == task_dir.name
    assert spec.service_levels, f"{task_dir.name} loaded with no gates"
    assert spec.devices >= 1 and spec.usd_per_gpu_hour > 0
    assert spec.verdict_duration_s > 0


@pytest.mark.parametrize("task_dir", _live_tasks(), ids=lambda p: p.name)
def test_the_objective_is_a_member_of_the_closed_enum(task_dir: Path) -> None:
    """The declared objective is one the driver's enum can coerce.

    `Objective` is a closed enum and `_coerce` calls it on whatever the yaml
    says. A task naming an objective the enum does not carry cannot be loaded
    at all -- and the failure surfaces as a HARNESS FAILURE at launch, after a
    workspace has been built and cards have been claimed.
    """
    import yaml
    from serving_evolve.contracts.workload import Objective

    declared = (yaml.safe_load((task_dir / "contract.yaml").read_text("utf-8"))
                or {}).get("objective")
    assert declared, f"{task_dir.name} declares no objective"
    assert declared in {member.value for member in Objective}, (
        f"{task_dir.name} declares objective {declared!r}, which is not a "
        f"member of Objective. The driver coerces this value through the enum "
        f"as the first thing it does, so the run dies at launch with "
        f"'{declared} is not a valid Objective' -- after claiming its cards")


@pytest.mark.parametrize("task_dir", _live_tasks(), ids=lambda p: p.name)
def test_the_verdict_can_build_this_tasks_contract(task_dir: Path) -> None:
    """The second thing the production path reads: the verdict's own contract.

    `spine.load_contract` is what every arm is measured through, and it maps
    the two files onto the shape its readers expect. A task whose files it
    cannot map is a task the verdict cannot judge.
    """
    from final_validate import spine

    contract = spine.load_contract(task_dir.name)
    assert contract.task_id == task_dir.name
    assert contract.devices >= 1
    levels = contract.slo.get("service_levels") or []
    # Every live task scores requests against the p90 row alone.
    assert contract.objective == "dollars_per_dollar"
    assert contract.slo["request_scoring"]["kind"] == "request_p90_billable_tokens_v4"
    want = [90]
    assert sorted(int(r["percentile"]) for r in levels) == want, (
        f"{task_dir.name} has {len(levels)} gate row(s)")
    # The private correctness settings are read by the host-side stage
    # alone; the shared contract (shipped to the agent) does not carry them.
    assert "correctness" not in contract.slo
    from final_validate import private_correctness
    assert private_correctness.answer_settings(contract) == (20, 0.2)


def test_this_covers_every_task_on_disk() -> None:
    """A parametrised test over an empty list is a test that passes vacuously."""
    tasks = _live_tasks()
    assert tasks, (
        "no live tasks were found, so every test in this file passed without "
        "checking anything")
    assert len(tasks) >= 1
