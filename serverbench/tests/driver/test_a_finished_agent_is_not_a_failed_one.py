"""A launch that finished its work cannot be "could not reach its model".

`terminal_failure` reads a launch log for markers such as "Invalid API key".
A launch log can carry one from a single subagent's failed authentication, in a
session that then works for hours, exits 0 and writes a closing report. Reading
that as "could not reach its model" would end the run with budget unspent.

Two rules keep it honest, and neither weakens the real case: a marker from a
subagent is not the session's, and a completed turn settles that the session
reached its model. A launch that never reaches its model produces no completed
turn, and its failure is not a subagent's.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for _p in (str(REPO / "serverbench" / "harness"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lifecycle import agent as agent_mod  # noqa: E402

RESULT = json.dumps({
    "type": "result", "subtype": "success", "is_error": False,
    "duration_ms": 27057653, "result": "the tree is published as-is",
})

SUBAGENT_AUTH_FAILURE = json.dumps({
    "type": "user",
    "message": {"content": [{"type": "text",
                             "text": "Invalid API key - Fix external API key"}]},
    "error": "authentication_failed", "subagent_type": "Explore",
    "is_api_error_message": True, "timestamp": "2026-01-01T00:13:00.000Z",
})

REAL_FAILURE = json.dumps({
    "type": "system", "subtype": "init", "apiKeySource": "none",
    "result": "Invalid API key",
})


def _log(tmp_path: Path, *lines: str) -> Path:
    path = tmp_path / "agent.start.0.log"
    path.write_text("\n".join(lines) + "\n", "utf-8")
    return path


def test_a_subagents_auth_failure_is_not_the_sessions(tmp_path):
    log = _log(tmp_path, SUBAGENT_AUTH_FAILURE, RESULT)
    assert agent_mod.terminal_failure(log) == ""


def test_a_completed_turn_settles_it(tmp_path):
    """Even with the marker on a line of its own, after the result."""
    log = _log(tmp_path, RESULT, '{"text":"Invalid API key"}')
    assert agent_mod.terminal_failure(log) == ""


def test_a_launch_that_never_started_is_still_terminal(tmp_path):
    """The case the function exists for: no turn, no subagent, a dead key."""
    log = _log(tmp_path, REAL_FAILURE)
    why = agent_mod.terminal_failure(log)
    assert "could not reach its model" in why
    assert "Invalid API key" in why


def test_an_errored_result_does_not_count_as_a_turn(tmp_path):
    """`is_error: true` is the CLI reporting the failure, not a finished turn."""
    errored = json.dumps({"type": "result", "is_error": True,
                          "result": "Invalid API key"})
    assert agent_mod.terminal_failure(_log(tmp_path, errored)) != ""


def test_a_finished_early_launch_is_resumed_not_declined(tmp_path):
    """The three predicates the continuation loop turns on, together.

    The loop asks `terminal_because` first, and a non-empty answer returns
    from `_run_agent` before the idle and cold-restart branches are ever
    reached. So all three answers must point the same way -- not terminal, not
    idle, and early enough to be worth resuming.
    """
    log = _log(tmp_path, SUBAGENT_AUTH_FAILURE, RESULT)
    run = agent_mod.AgentRun(
        session_id="s", continued=False, exit_code=0, seconds=27060.2,
        budget_seconds=28800.0, stdout_path=str(log), stopped_early=True,
        terminal_because=agent_mod.terminal_failure(log))
    assert run.terminal_because == "", "the loop would return before resuming"
    assert not agent_mod.was_idle(run), "the loop would count it toward the idle cap"
    assert run.stopped_early, "time was left on the deadline"


def test_the_order_of_the_loops_questions_is_unchanged():
    """`terminal_because` is asked first, which is why it must be right."""
    body = (REPO / "serverbench/harness/lifecycle/run.py").read_text("utf-8")
    terminal = body.index("if run.terminal_because:")
    idle = body.index("idle = agent_mod.was_idle(run)")
    assert terminal < idle, (
        "the idle check precedes the terminal one; re-read which branch "
        "ends the run and update this test deliberately")
