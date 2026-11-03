"""Every live task is in the state `taskctl` would leave it in.

`serving_tasks/_manage/taskctl.py` is the one writer of task parameters; its
`check` regenerates every generated file in memory and compares, reads the
task back as a spec, and checks the candidate-declared load and the packaged
evaluator. This runs the same checks in process, per task.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qualification  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
MANAGE = REPO / "serving_tasks" / "_manage"


@pytest.fixture(scope="module")
def taskctl():
    assert (MANAGE / "taskctl.py").is_file(), MANAGE
    for path in (str(MANAGE), str(REPO / "serverbench" / "harness")):
        if path not in sys.path:
            sys.path.insert(0, path)
    import taskctl as module
    return module


def _tasks() -> list[str]:
    root = REPO / "serving_tasks"
    return sorted(p.name for p in root.glob("*")
                  if (p / "contract.yaml").is_file()) if root.is_dir() else []


@pytest.mark.parametrize("task", _tasks())
def test_every_live_task_is_clean(taskctl, task: str) -> None:
    problems = taskctl.problems_of(task)
    assert not problems, (
        f"{task} is not in the state `taskctl` leaves it in:\n  "
        + "\n  ".join(problems)
        + f"\n\nRegenerate with `taskctl render {task}`, or change a "
          f"parameter with `taskctl set {task} ...`; never by hand.")


def test_the_live_tasks_share_one_execution_contract(taskctl) -> None:
    assert taskctl.consistency() == []


def test_authored_files_cover_the_task_data_and_the_starter(taskctl) -> None:
    import sites
    tasks = _tasks()
    assert tasks, "no tasks on disk"
    covered = {p.name for p in sites.authored_files(tasks[0])}
    for name in ("server.py", "contract.yaml", "reference.yaml"):
        assert name in covered
    # The starter declares no offered load: the agent chooses it.
    assert "concurrency.json" not in covered


def test_scalar_writer_replaces_a_multiline_yaml_value_whole(taskctl) -> None:
    """Price provenance may contain paragraphs without orphaning their tail."""
    import blocks
    import yaml

    source = """token_prices:
  source: 'old provenance first paragraph


    old provenance second paragraph'
  cache_rate_is: old
"""
    rendered = taskctl._render(
        "Operator-supplied tariff: $0.02/M input", "yaml")
    fresh, changed = blocks.set_scalar(
        source, "source", rendered, indent=2, under="token_prices")

    assert changed
    body = yaml.safe_load(fresh)
    assert body["token_prices"]["source"] == \
        "Operator-supplied tariff: $0.02/M input"
    assert body["token_prices"]["cache_rate_is"] == "old"
    assert "old provenance" not in fresh


def test_every_set_parameter_has_a_registered_site(taskctl) -> None:
    """A `set` flag whose name no site registers writes nothing."""
    import sites
    for _, name in taskctl.SCALAR_FLAGS:
        assert sites.SCALARS.get(name), f"no site registered for {name!r}"


def test_the_cli_carries_only_the_live_verbs(taskctl) -> None:
    parser = taskctl.build_parser()
    verbs = set(parser._subparsers._group_actions[0].choices)
    assert verbs == {"show", "check", "render", "set", "rename", "new"}
    with pytest.raises(SystemExit):
        parser.parse_args(["set", _tasks()[0], "--p95-ttft", "1"])


def test_rename_refuses_a_hardware_change(taskctl) -> None:
    task = _tasks()[0]
    other = task.rsplit("_", 1)[0] + "_9xH999"
    parsed = taskctl.build_parser().parse_args(["rename", task, "--to", other])
    with pytest.raises(SystemExit, match="different hardware"):
        parsed.fn(parsed)
    assert not (REPO / "serving_tasks" / other).exists()


def test_the_allocation_is_fixed_and_the_topology_is_the_candidates() -> None:
    import yaml

    for task in _tasks():
        root = REPO / "serving_tasks" / task
        body = yaml.safe_load((root / "contract.yaml").read_text("utf-8"))
        assert body["devices"] >= 1 and body["usd_per_gpu_hour"] > 0
        for forbidden in ("tensor_parallel_degree", "data_parallel_degree",
                          "required_collectives", "disaggregation", "topology"):
            assert forbidden not in body
        for name in ("README.md", "instruction.md"):
            assert ("internal organisation is the candidate's choice"
                    in (root / name).read_text("utf-8"))


def test_every_unqualified_task_is_visible_rather_than_forgotten() -> None:
    """An unqualified task is a legitimate state; it is printed, not hidden."""
    unqualified = {task: qualification.missing_evidence(task)
                   for task in _tasks()}
    lines = [f"{task}:\n" + "".join(f"    - {m}\n" for m in missing)
             for task, missing in sorted(unqualified.items()) if missing]
    if lines:
        print("\nNOT QUALIFIED:\n" + "".join(lines))
