"""A measurement measures one server, over a timebox, with clean inputs.

Repeat runs of one SGLang configuration at one offered load can land on
different sides of a gate, because each run draws its own workload seed and
the scenario enters each session at a random depth. So every run draws a fresh
seed (even in development, odd in the verdict,
`measurement/test_one_seed_reaches_every_measurement.py`), and systems are
compared over repeated runs rather than one draw.

What these tests pin: the development window is one the agent chooses inside
the allowed range; the stop rule is a timebox -- a count
rule would fix the numerator of an objective whose denominator is GPU-hours;
and a concurrency-search point measures only its own server, from its own
output, on a card nothing else is computing on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "serverbench" / "harness"))

LIVE_TASKS = ("qwen3_5_9b_standard_2xL40", "gemma4_e4b_it_standard_1xL40")


def _cmd(task_id: str, **kw) -> list[str]:
    from final_validate import tasks as task_modules
    module = task_modules.python_module(task_id)
    return list(module.invocation("http://x", "/tok", "/art",
                                  concurrency=kw.pop("concurrency", 4), **kw))


@pytest.mark.parametrize("duration", [1, 299, 1801, 3600])
def test_no_window_outside_the_development_range_reaches_aiperf(duration: int) -> None:
    with pytest.raises(ValueError, match="300-1800 s"):
        _cmd("qwen3_5_9b_standard_2xL40", duration_s=duration)


@pytest.mark.parametrize("duration", [300, 450, 900, 1799, 1800])
def test_the_agent_chooses_its_development_window(duration: int) -> None:
    """The agent decides how long to measure; the verdict's own 1800 s is
    enforced in `scenario.main`."""
    cmd = _cmd("qwen3_5_9b_standard_2xL40", duration_s=duration)
    assert cmd[cmd.index("--benchmark-duration") + 1] == str(duration)


def test_the_stop_rule_is_a_timebox_and_never_a_count() -> None:
    """GPU-hours are the objective's denominator; the box holds it constant."""
    for task in LIVE_TASKS:
        cmd = _cmd(task)
        assert "--benchmark-duration" in cmd, f"{task}: no timebox"
        for forbidden in ("--request-count", "--num-requests",
                          "--conversation-num", "--num-conversations",
                          "--num-sessions"):
            assert forbidden not in cmd, (
                f"{task}: {forbidden} makes the stop rule a count, which fixes "
                f"the numerator of an objective whose denominator is time")


def test_a_point_never_reports_an_aggregate_it_did_not_write() -> None:
    """A stale export would be read as this run's result.

    `aggregate()` globs for the last profile_export_aiperf.json it finds, so
    if a point did not clear its artifact directory, a server that failed to
    start would report the previous configuration's numbers as its own.
    """
    body = (REPO / "serverbench" / "harness" / "final_validate" / "concurrency"
            / "point.py").read_text("utf-8")
    start = body.index("def measure(")
    block = body[start:body.index("    try:", start)]
    assert "shutil.rmtree(artifact_dir" in block, (
        "a point does not clear its artifact directory, so a previous run's "
        "export can be reported as this one's result")
    assert block.index("artifact_dir = run_dir") < block.index("rmtree"), (
        "the directory is cleared before it is named, which cannot work")


def test_a_card_is_not_quiet_while_something_is_computing_on_it(
        monkeypatch) -> None:
    """A live process makes a card busy, however little memory it holds.

    A CUDA context a torn-down container never released can be a few hundred
    MiB -- below a memory threshold, but a live process on a card the next
    point believes it owns.
    """
    from final_validate import lease

    monkeypatch.setattr(lease, "_foreign_processes",
                        lambda d: ["4242, 496 MiB"] if d == 5 else [])
    monkeypatch.setattr(lease, "_busy_mib", lambda d: 0)
    monkeypatch.setattr(lease.time, "sleep", lambda _s: None)
    why = lease.wait_for_quiet([4, 5], timeout_s=0)
    assert "card 5" in why, "a card with a live process was reported quiet"
    monkeypatch.setattr(lease, "_foreign_processes", lambda d: [])
    monkeypatch.setattr(lease, "_busy_mib", lambda d: 300)
    assert "300 MiB" in lease.wait_for_quiet([4], timeout_s=0, quiet_mib=256)
    monkeypatch.setattr(lease, "_busy_mib", lambda d: 0)
    assert lease.wait_for_quiet([4, 5], timeout_s=0) == ""


def test_teardown_waits_for_the_container_to_actually_go() -> None:
    """`docker rm -f` returns before the CUDA contexts are released."""
    import inspect

    from final_validate import deploy
    assert "settle_s" in inspect.signature(deploy._release).parameters, (
        "release returns as soon as docker does, so the next point starts on "
        "a card that has not finished emptying")
    body = inspect.getsource(deploy._release)
    assert "docker" in body and "ps" in body, (
        "nothing confirms the container record is gone before returning")
