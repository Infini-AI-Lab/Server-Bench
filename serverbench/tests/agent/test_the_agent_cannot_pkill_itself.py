"""The agent's brief must not be a `pkill -f` pattern for its own process.

The agent has a shell and is expected to run and stop its own benchmarks, for
instance with `pkill -f "aiperf profile"`. `pkill -f` matches command lines,
and the brief names `aiperf profile`; were the brief on the `claude` process's
command line, that cleanup would kill the agent itself (exit 137, no `result`
event, a session that cannot be resumed). So the launcher stages the prompt
into a file and feeds it on stdin: flags and paths on the argv, no task text.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "serverbench" / "harness"))

AGENT = REPO / "serverbench" / "harness" / "lifecycle" / "agent.py"


def _launch_body() -> str:
    body = AGENT.read_text("utf-8")
    start = body.index("def launch(*, identity: RunIdentity")
    return body[start:body.index("\ndef ", start + 10)]


def test_the_prompt_is_not_interpolated_into_the_argv() -> None:
    block = _launch_body()
    # `argv = ` rather than `argv = [`: the launch asks the backend for the
    # argv, because a Modal sandbox has no `docker exec`.
    argv = block[block.index("argv = "):block.index("journal.append(")]
    assert "shlex.quote(prompt)" not in argv, (
        "the brief is on the command line of the process whose death "
        "ends the run; any `pkill -f` the agent aims at a word in its own "
        "task will kill it")
    assert "{prompt}" not in argv
    assert "--print " in argv and "< " in argv, (
        "the prompt should arrive on stdin from the staged file")


def test_the_prompt_is_staged_into_the_container() -> None:
    block = _launch_body()
    assert "stage_prompt(" in block, (
        "nothing stages the prompt, so either it is on the argv or the "
        "agent is launched with no prompt at all")


def test_staging_writes_outside_the_workspace() -> None:
    """A file the agent did not create, inside its workspace, reads as task."""
    from lifecycle import agent

    path = agent.PROMPT_IN_CONTAINER
    assert not path.startswith("/workspace"), path
    assert path.startswith("/tmp/"), path


def test_staging_passes_the_text_over_stdin_not_a_heredoc(monkeypatch) -> None:
    """Quoting inside a brief must not be able to terminate the write early.

    A brief can contain shell, JSON and a heredoc of its own.
    """
    from lifecycle import agent

    seen = {}

    class Done:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(argv, **kw):
        seen["argv"] = argv
        seen["input"] = kw.get("input")
        return Done()

    monkeypatch.setattr(agent.subprocess, "run", fake_run)
    brief = "run `aiperf profile` <<'EOF'\nnot the end\nEOF\n'quoted'\n"
    path = agent.stage_prompt(container="c", text=brief, index=0)

    assert seen["input"] == brief, "the brief did not travel over stdin intact"
    assert "-i" in seen["argv"], "docker exec cannot read stdin without -i"
    joined = " ".join(seen["argv"])
    assert "aiperf profile" not in joined, (
        "the brief reached the argv of the staging command instead")
    assert path == agent.PROMPT_IN_CONTAINER.format(n=0)


def test_the_launch_still_pins_what_was_sent() -> None:
    """Dropping the prompt from the argv must not drop it from the evidence."""
    block = _launch_body()
    assert "prompt_sha256" in block, (
        "nothing records a digest of the prompt, so a run cannot say "
        "afterwards what its agent was asked")
