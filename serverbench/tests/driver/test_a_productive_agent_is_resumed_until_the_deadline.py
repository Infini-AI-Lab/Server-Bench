"""A launch that worked is resumed. A launch that did nothing is told why.

An agent can work for hours and then fall into a loop of turns that end in
seconds ("Waiting."), often while a long benchmark it started is still running.
A flat launch count would spend productive and empty launches from the same
purse and end the run with budget unspent.

Two properties are pinned here, and the second is why the first is safe:

1. A productive launch is resumed for as long as the deadline allows. If it is
   not, `--hours` is not the budget; a launch count is.
2. CONSECUTIVE idle launches are bounded, with a backoff, so an agent that
   will not start cannot spin on a held card until the deadline.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for _p in (str(REPO / "serverbench" / "harness"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lifecycle import agent as agent_mod  # noqa: E402
from lifecycle import run as run_mod  # noqa: E402


def _run(tmp_path: Path, *, seconds: float, calls: int,
         scheduled: int = 0) -> agent_mod.AgentRun:
    """One finished launch, with a stdout log of the given shape."""
    log = tmp_path / f"agent.{seconds}.{calls}.{scheduled}.log"
    body = "".join('{"type":"assistant","message":{"content":[{"type":'
                   '"tool_use","name":"Bash"}]}}\n' for _ in range(calls))
    body += "".join('{"type":"assistant","message":{"content":[{"type":'
                    '"tool_use","name":"ScheduleWakeup"}]}}\n'
                    for _ in range(scheduled))
    log.write_text(body, "utf-8")
    return agent_mod.AgentRun(
        session_id="s", continued=True, exit_code=0, seconds=seconds,
        budget_seconds=28800.0, stdout_path=str(log), stopped_early=True)


class TestTheLaunchShapes:
    """`was_idle` against long working launches and short waiting ones."""

    def test_the_two_long_launches_are_work(self, tmp_path):
        assert not agent_mod.was_idle(_run(tmp_path, seconds=9692.6, calls=556))
        assert not agent_mod.was_idle(_run(tmp_path, seconds=5950.9, calls=191))

    def test_the_ten_second_waiting_launches_are_idle(self, tmp_path):
        assert agent_mod.was_idle(_run(tmp_path, seconds=10.0, calls=1))
        assert agent_mod.was_idle(_run(tmp_path, seconds=5.0, calls=1))

    def test_the_boundary_launch_is_idle_too(self, tmp_path):
        # 120 seconds and two calls, both waiting. Launch length is quantised
        # by the driver's own five-second poll, which is why the threshold
        # cannot sit at 120.
        assert agent_mod.was_idle(_run(tmp_path, seconds=120.0, calls=2))

    def test_a_short_but_busy_launch_is_work(self, tmp_path):
        # An agent interrupted early has still done something: 275 seconds
        # and nine calls, re-checking its server.
        assert not agent_mod.was_idle(_run(tmp_path, seconds=275.3, calls=9))

    def test_a_no_callback_tool_does_not_count_as_work(self, tmp_path):
        # A launch whose only calls are `ScheduleWakeup` is spinning, not
        # working; counting them would read the spin as work.
        assert agent_mod.was_idle(
            _run(tmp_path, seconds=10.0, calls=0, scheduled=2))


class TestThePolicy:
    def test_the_ceiling_is_a_runaway_guard_not_a_budget(self):
        # Whatever this number is, it must not be reachable by an agent
        # working in hour-long turns through an eight-hour run.
        assert run_mod.MAX_LAUNCHES >= 100

    def test_the_idle_cap_is_small_and_counts_consecutively(self):
        assert 2 <= run_mod.MAX_IDLE_LAUNCHES <= 8

    def test_an_idle_relaunch_waits(self):
        assert run_mod.IDLE_BACKOFF_S > 0
        assert run_mod.IDLE_BACKOFF_CAP_S >= run_mod.IDLE_BACKOFF_S
