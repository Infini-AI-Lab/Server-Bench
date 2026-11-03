"""Every measurement draws its own workload seed; the verdict never reuses one.

There is no public development seed. A development measurement -- the in-Harbor
evaluator copy, `--measure-only`, a development full test, a concurrency search
point -- draws a fresh EVEN seed; the final verdict draws a fresh ODD one. So a
verdict can never replay a request mix the agent measured, and the seed that
was used is always on the recorded command.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "serverbench" / "harness"))

from final_validate.tasks import _agentx_mvp as shared  # noqa: E402

LIVE = tuple(p.name for p in sorted((REPO / "serving_tasks").glob("*"))
             if (p / "contract.yaml").is_file())


def _seed(cmd: list[str]) -> int:
    assert cmd.count("--random-seed") == 1, cmd
    return int(cmd[cmd.index("--random-seed") + 1])


@pytest.mark.parametrize("task", LIVE)
def test_no_task_pins_a_development_seed(task: str) -> None:
    from final_validate import tasks as task_modules
    assert getattr(task_modules.python_module(task), "RANDOM_SEED", None) is None


@pytest.mark.parametrize("task", LIVE)
def test_every_development_command_draws_a_fresh_even_seed(task: str) -> None:
    from final_validate import tasks as task_modules
    module = task_modules.python_module(task)
    seeds = {_seed(module.invocation("http://x", "/tok", "/art", concurrency=2))
             for _ in range(8)}
    assert len(seeds) > 1 and all(seed % 2 == 0 for seed in seeds)


def test_final_and_development_seed_spaces_are_disjoint() -> None:
    dev = {shared.development_seed() for _ in range(2000)}
    final = {shared.final_seed() for _ in range(2000)}
    assert all(s % 2 == 0 for s in dev) and all(s % 2 == 1 for s in final)


def test_the_concurrency_search_can_fix_a_seed_without_a_second_flag() -> None:
    from final_validate.concurrency.point import aiperf_command
    task = LIVE[0]
    drawn = aiperf_command(task, url="u", tokenizer=Path("/t"),
                           artifact_dir=Path("/a"), duration_s=1800,
                           concurrency=2)
    assert _seed(drawn) % 2 == 0
    fixed = aiperf_command(task, url="u", tokenizer=Path("/t"),
                           artifact_dir=Path("/a"), duration_s=1800,
                           concurrency=2, random_seed=999)
    assert _seed(fixed) == 999


@pytest.mark.parametrize("task", LIVE)
def test_the_brief_says_every_run_draws_its_own_seed(task: str) -> None:
    body = (REPO / "serving_tasks" / task / "instruction.md").read_text("utf-8")
    assert "<fresh random seed>" in body
    assert "disjoint from every development seed" in body
