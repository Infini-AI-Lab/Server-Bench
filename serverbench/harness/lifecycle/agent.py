"""The processes the harness starts inside the agent's container.

The configured coding agent is launched once and continued in place whenever
its provider can resume the recorded session. Deterministic checks on each
version run in `lifecycle/boundary_checks.py`, not beside the agent.

## The coding agent
The primary coding agent: launched once and continued in place.

There is no writer role and no prescribed subagent topology. The agent owns
development end to end and decides for itself how to divide the work.

**Continuation is real or it is not claimed.** Claude receives a session id
chosen by the harness; Codex reports the thread id it chose, which the harness
records. Both are resumed through the provider's native continuation command,
in the same workspace, against the deadline the run was created with.

If the platform cannot continue the session, the driver starts a cold agent in
the same workspace against the same deadline and records it as a cold restart.
It never calls that a continuation, and it does not pretend the unused hours
were handed back.
"""
from __future__ import annotations

import contextlib
import dataclasses
import datetime
import hashlib
import json
import re
import shlex
import subprocess
import threading
import time
import uuid
from pathlib import Path

from . import backend, billing, instructions
from .record import Journal, RunIdentity, utc_now

#: Budget below this is not worth resuming into.
EARLY_STOP_MARGIN_S = 300.0

#: The full context window of every model the Codex CLI drives here, from
#: Codex's model catalog (`max_context_window`). Left to itself Codex runs
#: these models at a smaller default window (272k or 500k tokens) and compacts
#: the agent's history below it. A model missing from this table is refused
#: rather than run at a reduced window. Models on the Claude CLI get theirs in
#: `serverbench/scripts/agent/harbor_env.py` (`AGENT_CONTEXT_WINDOWS`).
CODEX_CONTEXT_WINDOWS = {
    "gpt-6.1-sol": 872_000,
    "gpt-6-astra": 872_000,
    "gpt-6-sol": 872_000,
    "gpt-5.6-sol": 872_000,
}


def codex_config(model: str, effort: str = "") -> list[str]:
    """The `--model` and `-c` arguments of every Codex invocation: the model,
    its full context window and the reasoning effort."""
    window = CODEX_CONTEXT_WINDOWS.get(model)
    if not window:
        raise SystemExit(
            f"REFUSED: the full context window of Codex model {model!r} is unknown. "
            f"Add its `max_context_window` from Codex's model catalog to "
            f"CODEX_CONTEXT_WINDOWS (serverbench/harness/lifecycle/agent.py).")
    config = ["--model", model, "-c", f"model_context_window={window}"]
    if effort:
        config += ["-c", f'model_reasoning_effort="{effort}"']
    return config


class ContinuationUnavailable(RuntimeError):
    """The platform could not continue the recorded session."""


@dataclasses.dataclass(frozen=True)
class AgentRun:
    session_id: str
    continued: bool
    exit_code: int
    seconds: float
    budget_seconds: float
    stdout_path: str
    stopped_early: bool
    #: Non-empty when the agent could not reach its model at all, so trying
    #: again cannot help. See `terminal_failure()`.
    terminal_because: str = ""

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["kind"] = "agent_run"
        d["schema_version"] = 1
        return d


#: What the CLI says when the run can never work, whatever we do next.
#:
#: These are CONFIGURATION failures wearing a runtime failure's clothes. The
#: resume loop exists for an agent that stopped with budget left -- a crash, a
#: context exhaustion, a dropped stream -- and for all of those, asking again
#: is the right move. For these it is not: the model does not exist, or no
#: credential reached the container (`404 model_not_found ... It may not exist
#: or you may not have access to it`), and every attempt fails exactly like the
#: first. Retrying would also end with a summary that blames the agent for a
#: defect in the command line.
TERMINAL_MARKERS = (
    '"error":"model_not_found"',
    '"api_error_status":404',
    "It may not exist or you may not have access to it",
    "Invalid API key",
    '"apiKeySource":"none"',
    # THE MONEY RAN OUT, which is terminal in exactly the same way:
    #
    #   403 Failed to authenticate. API Error: 403 Key limit exceeded
    #   (total limit).
    #
    # No number of retries fixes a spend limit.
    '"api_error_status":403',
    "Key limit exceeded",
    "Failed to authenticate",
    "insufficient_quota",
    "exceeded your current quota",
)

#: Present in healthy runs too, so never terminal on its own -- only reported
#: alongside a hard marker, because they help explain it.
#:
#: `[claude-code:unrecognized_model]` is emitted for ANY model the CLI does not
#: ship a card for, which includes every OpenRouter-hosted one, and working
#: runs carry it. Treating it as terminal would refuse to start every
#: OpenRouter-hosted configuration.
#:
#: `apiKeySource: none` is here for the same reason: a host whose credentials
#: arrive by mounted file can report it and work.
TERMINAL_NEEDS_CORROBORATION = ('"apiKeySource":"none"',
                                "[claude-code:unrecognized_model]")

#: How the reason is worded when the cause is money rather than configuration,
#: because the two need different actions from whoever reads the journal.
BILLING_MARKERS = ("Key limit exceeded", "insufficient_quota",
                   "exceeded your current quota", '"api_error_status":403')

#: A SESSION LIMIT, which is retryable but NOT IMMEDIATELY -- and that
#: distinction is the whole point of separating it from everything else:
#:
#:     You've hit your session limit · resets 6:50am (UTC)
#:
#: Relaunching at once would die in seconds, again and again, and exhaust the
#: launch ceiling long before the limit resets.
#:
#: It is the same SHAPE as a billing pause -- a wall that clears on a clock,
#: not a configuration error and not a crash -- so it gets the same treatment:
#: park, then resume, with the pause added back to the deadline. What it is
#: not is terminal: retrying DOES help, once the clock has run.
SESSION_LIMIT_MARKERS = ("hit your session limit", "session limit reached",
                         "usage limit reached")

#: "resets 6:50am (UTC)", "resets at 11pm", "resets 6:50 am (UTC)".
_RESET_AT = re.compile(
    r"resets?\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*([ap])m?\b",
    re.IGNORECASE)


def session_limit_reset(text: str, *, now=None) -> float | None:
    """Seconds until the session limit resets, from the CLI's own message.

    The message states a wall-clock time, not a duration, so the answer
    depends on `now` -- and a time already past today means tomorrow. Returns
    None when no reset time is stated, so a caller falls back to a fixed wait
    rather than guessing zero and hammering the wall again.
    """
    import datetime as _dt

    found = _RESET_AT.search(text or "")
    if not found:
        return None
    hour = int(found.group(1)) % 12
    minute = int(found.group(2) or 0)
    if found.group(3).lower() == "p":
        hour += 12
    moment = now or _dt.datetime.now(_dt.timezone.utc)
    target = moment.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= moment:
        # A TIME ALREADY PAST MEANS IT HAS CLEARED, NOT THAT IT IS TOMORROW.
        #
        # The CLI writes this message AT the moment it refuses, and the driver
        # reads the log seconds later -- so a stated time earlier than now is
        # a wall that has already lifted, usually because the log carries an
        # older message too. Rolling it forward a day would park a run for
        # most of a day on a limit that has already cleared.
        return 0.0
    return max(0.0, (target - moment).total_seconds())


def tool_calls(stdout_path: Path | str) -> int:
    """How many tools one launch called, from its own stdout.

    The driver needs to tell two endings apart, because they deserve opposite
    responses:

    * a launch that WORKED and then ended its turn -- hundreds of tool calls
      -- which should be resumed as often as the deadline allows;
    * a launch that did NOTHING -- a few seconds, a couple of tool calls, both
      `ScheduleWakeup`, ending `"result": "Waiting."` -- which will do the
      same thing again if asked the same way.

    Counted from the stream rather than inferred from duration: a legitimate
    turn can be short, and a spin can be slow.

    CALLS TO A NO-CALLBACK TOOL DO NOT COUNT. That is what a spin is made of,
    and counting them would read it as work. Those tools are disallowed at
    launch, so normally this subtraction finds nothing -- it is here because
    the policy must not depend on the disallow list being complete.
    """
    try:
        body = Path(stdout_path).read_text("utf-8", errors="ignore")
    except OSError:
        return 0
    calls = body.count('"type":"tool_use"')
    for name in NO_CALLBACK_TOOLS:
        calls -= body.count(f'"name":"{name}"')
    return max(calls, 0)


#: Under this many seconds, a launch has not done a unit of this task's work.
#:
#: A round here is: edit a server, boot it, offer it load for minutes, read the
#: result. A spin can last exactly 120.0 s -- quantised by this module's own
#: five-second poll -- which is why the line cannot sit at two minutes.
#: Nothing that moves this task finishes in three, and the response to
#: crossing the line is not to stop the run: it is to state the environment
#: fact once and resume. A false positive costs one sentence and one backoff.
IDLE_LAUNCH_S = 180.0

#: ...unless it called more tools than this while it was up. BOTH conditions
#: have to hold, because either alone misreads a real ending: a productive
#: turn can be interrupted early by the deadline, and a spin can be slow. A
#: spin makes a status poll or two before saying "Waiting."; a productive
#: launch makes hundreds.
IDLE_LAUNCH_CALLS = 4


def was_idle(run: AgentRun) -> bool:
    """Whether this launch ended without doing anything.

    The distinction the continuation policy turns on. An agent that WORKED and
    then ended its turn is resumed for as long as the deadline allows -- that
    is the point of the loop. An agent that ended having done nothing will do
    the same again if asked the same way, so it is told what the environment
    does and does not do, after a backoff, a bounded number of times.
    """
    return (run.seconds < IDLE_LAUNCH_S
            and tool_calls(run.stdout_path) <= IDLE_LAUNCH_CALLS)


def session_limited(stdout_path: Path | str, *, now=None) -> float | None:
    """Seconds to wait if this agent hit a session limit, else None.

    A separate reader from `terminal_failure` on purpose: that function answers
    "can this ever start", and the answer here is "yes, later".
    """
    try:
        body = Path(stdout_path).read_text("utf-8", errors="ignore")
    except OSError:
        return None
    if not any(marker in body for marker in SESSION_LIMIT_MARKERS):
        return None
    # The LAST stated reset wins: a log can carry an earlier wall too, and
    # only the most recent one is about the refusal that just happened.
    latest = None
    for match in _RESET_AT.finditer(body):
        latest = match
    if latest is None:
        return float(SESSION_LIMIT_FALLBACK_S)
    seconds = session_limit_reset(latest.group(0), now=now)
    if seconds is None:
        return float(SESSION_LIMIT_FALLBACK_S)
    # Zero means the wall has already lifted. Return a SHORT wait rather than
    # zero so the next launch is not instantaneous.
    return max(seconds, 30.0)


#: When the message states no reset time. Long enough to clear a rolling
#: window, short enough not to spend a run's remaining hours on one guess.
SESSION_LIMIT_FALLBACK_S = 30 * 60

#: What docker says when the container is not there any more, as distinct from
#: a container that is up and refusing something.
_CONTAINER_GONE = ("is not running", "No such container",
                   "is not paused", "removal of container")


def _reached_its_model(body: str) -> bool:
    """Whether this launch ever completed a turn.

    The CLI writes one `{"type":"result", ...}` record per finished turn, with
    the assistant's final text in it. An agent that produced one reached its
    model, whatever else the log says, and "could never start" is then a false
    statement about it.
    """
    # MATCHED WITH WHITESPACE ALLOWED. The CLI's own stream is compact, but a
    # tee, a re-serialisation or another writer's line is not, and a rule that
    # only reads one spelling silently stops protecting the case it was
    # written for.
    result = re.compile(r'"type"\s*:\s*"result"')
    errored = re.compile(r'"is_error"\s*:\s*true')
    return any(result.search(line) and not errored.search(line)
               for line in body.splitlines())


def terminal_failure(stdout_path: Path | str) -> str:
    """Why this agent can never start, or "" if it merely stopped.

    Read from the launch log rather than the exit code: exit 1 is what a
    crashed agent and an unreachable model have in common, and the difference
    decides whether the next launches are worth anything.

    ## Markers that are not about the launch

    A subagent can fail to authenticate once while the session carries on for
    hours:

        {"content":[{"type":"text","text":"Invalid API key - Fix external API
         key"}], "error":"authentication_failed", "subagent_type":"Explore",
         "is_api_error_message":true, ...}

    A whole-file grep would turn that into a verdict about the session. So:

    * a launch that produced a completed turn is never terminal -- it reached
      its model by definition;
    * a marker inside a subagent's own error record is that subagent's, not
      the session's.

    Neither rule weakens the real case. A launch that never reaches its model
    produces no result record, and its error is not a subagent's.
    """
    try:
        body = Path(stdout_path).read_text("utf-8", errors="ignore")
    except OSError:
        return ""
    if _reached_its_model(body):
        return ""
    body = "\n".join(line for line in body.splitlines()
                     if '"subagent_type"' not in line)
    hits = [marker for marker in TERMINAL_MARKERS if marker in body]
    hard = [marker for marker in hits
            if marker not in TERMINAL_NEEDS_CORROBORATION]
    if not hard:
        return ""
    detail = ""
    for line in body.splitlines():
        if '"result":' in line and "model" in line:
            start = line.find('"result":"')
            if start != -1:
                detail = line[start + 10:start + 260].split('","')[0]
                break
    billing = [m for m in hits if m in BILLING_MARKERS]
    kind = ("a BILLING limit, not a configuration error -- the account or key "
            "is out of budget and no number of retries changes that"
            if billing else
            "a configuration failure, not an agent that stopped")
    return (f"the agent could not reach its model at all "
            f"({', '.join(hits)}). Retrying cannot help: this is {kind}."
            + (f" The CLI said: {detail}" if detail else ""))


def remaining_seconds(identity: RunIdentity, *, now: float | None = None,
                      paused_s: float = 0.0) -> float:
    """Budget left against the ORIGINAL deadline. Never re-based on a restart:
    that is what makes a continuation a continuation.

    `paused_s` is the ONE thing that moves the deadline, and it is not a
    re-basing: it is time the run was parked with no agent running, waiting for
    an operator to raise a spend limit or drop in a new key (see
    `lifecycle/billing.py`). `--hours` bounds the AGENT's work, so an 8-hour run
    parked for 2 hours has had 6 hours of agent and is owed the other 2.

    It is passed in rather than read here because this module does not own the
    journal, and the total is a fact recorded there by the driver.
    """
    deadline = datetime.datetime.strptime(
        identity.deadline_utc, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=datetime.timezone.utc)
    if paused_s:
        deadline += datetime.timedelta(seconds=max(0.0, float(paused_s)))
    current = (datetime.datetime.fromtimestamp(now, datetime.timezone.utc)
               if now is not None
               else datetime.datetime.now(datetime.timezone.utc))
    return max(0.0, (deadline - current).total_seconds())


def session_of(journal: Journal) -> str:
    """The session this run is having. Minted once, then read back forever."""
    started = journal.last("agent_session")
    return started["session_id"] if started else ""


def ensure_session(journal: Journal) -> str:
    existing = session_of(journal)
    if existing:
        return existing
    session_id = str(uuid.uuid4())
    journal.append("agent_session", session_id=session_id, minted_utc=utc_now())
    return session_id


def new_session(journal: Journal) -> str:
    """Mint a FRESH session id and record that the old one was abandoned.

    Used only for a cold restart -- see `_cold_start_prompt` -- and journalled
    so that a reader can see the conversation was broken, rather than
    discovering it from a session id that silently changed.
    """
    session_id = str(uuid.uuid4())
    journal.append("agent_session", session_id=session_id,
                   minted_utc=utc_now(), cold_start=True,
                   replaces=session_of(journal))
    return session_id


def _canonical_event(event: dict, agent_cli: str) -> dict:
    """Translate Codex JSONL into the event shape the ATIF reader consumes."""
    if agent_cli != "codex":
        return event
    kind = event.get("type")
    item = event.get("item") or {}
    item_kind = item.get("type")
    item_id = str(item.get("id") or "codex-item")
    if kind == "item.started" and item_kind == "command_execution":
        return {
            "type": "assistant",
            "message": {"content": [{
                "type": "tool_use", "id": item_id, "name": "Bash",
                "input": {"command": str(item.get("command") or "")},
            }]},
            "provider_event": event,
        }
    if kind == "item.completed" and item_kind == "command_execution":
        return {
            "type": "user",
            "message": {"content": [{
                "type": "tool_result", "tool_use_id": item_id,
                "content": str(item.get("aggregated_output") or ""),
                "is_error": item.get("exit_code") not in (0, None),
            }]},
            "provider_event": event,
        }
    if kind == "item.completed" and item_kind == "agent_message":
        return {
            "type": "assistant",
            "message": {"content": [{
                "type": "text", "text": str(item.get("text") or ""),
            }]},
            "provider_event": event,
        }
    if kind == "item.completed" and item_kind == "file_change":
        return {
            "type": "assistant",
            "message": {"content": [{
                "type": "tool_use", "id": item_id, "name": "Write",
                "input": {"file_path": str(item.get("path") or "(file change)"),
                          "changes": item.get("changes") or []},
            }]},
            "provider_event": event,
        }
    if kind == "turn.completed":
        return {
            "type": "result", "subtype": "success",
            "usage": event.get("usage") or {}, "provider_event": event,
        }
    return event


#: Each tool result in the per-launch raw log is bounded to this; the stream
#: (`agent.stream.jsonl`) keeps `trajectory_capture.TOOL_RESULT_LIMIT`.
RAW_LOG_RESULT_LIMIT = 2_048


def tee_stream(pipe, raw_path: Path, stream_path: Path, *,
               agent_cli: str = "claude", on_session=None,
               workspace: Path | None = None,
               changes_path: Path | None = None,
               cli_path: Path | None = None, cli_header: str = "") -> int:
    """Write meaningful agent events, and return how many events landed.

    `stream_path` is opened for APPEND. That is the whole mechanism behind a
    continued session reading as one trajectory: launch two, three or ten times
    into the same run and the ATIF collector still sees a single ordered
    stream. Truncating here would give every continuation its own trajectory
    and make a resumed run look like several unrelated ones.

    Per-token system/thinking_tokens estimates carry no reasoning or billing
    evidence. Drop them before either write/flush. Actual thinking content,
    tool calls, diagnostics and final usage (including thinking-token totals)
    remain intact. This boundary is shared by every task/backend/continuation.

    Both files are LEAN (`trajectory_capture.LeanFilter`): heartbeats, repeated
    rate-limit statuses, thinking signatures and thinking blocks the CLI shows
    no text for, the duplicate `tool_use_result`, per-event envelope and the
    middle of oversized tool outputs are dropped; every behavioural event is
    kept. The raw log bounds each tool result more tightly than the stream
    (`RAW_LOG_RESULT_LIMIT`) -- the stream is the record, the raw log is for the
    detectors and for skimming. JSON lines in the raw log are re-serialised
    compactly, which is the CLI's own spelling, so the substring detectors that
    read it (`tool_calls`, `terminal_failure`, `session_limited`) see what they
    did.

    With `workspace` and `changes_path`, every tool result is followed by a
    workspace diff into `agent.changes.jsonl`, attributed to that tool call.

    With `cli_path`, the same events are rendered as the CLI's terminal output
    (`trajectory_capture.CliRenderer`) -- the main agent's and every
    subagent's -- and APPENDED there under `cli_header`, as the stream is.
    """
    from . import trajectory_capture as capture

    lean_raw = capture.LeanFilter(result_limit=RAW_LOG_RESULT_LIMIT)
    lean_stream = capture.LeanFilter()
    render = capture.CliRenderer()
    cli = None
    watcher = None
    if workspace is not None and changes_path is not None:
        try:
            watcher = capture.WorkspaceWatcher(workspace, changes_path)
        except OSError:
            watcher = None
    names: dict[str, str] = {}
    written = 0
    with contextlib.ExitStack() as files:
        raw = files.enter_context(open(raw_path, "w"))
        stream = files.enter_context(open(stream_path, "a"))
        if cli_path is not None:
            cli = files.enter_context(open(cli_path, "a", encoding="utf-8"))
            if cli_header:
                cli.write(f"\n═══ {cli_header} ═══\n")
                cli.flush()
        for line in pipe:
            text = line.strip()
            event = None
            if text.startswith("{"):
                # Malformed output is retained below as diagnostic text.
                with contextlib.suppress(json.JSONDecodeError):
                    event = json.loads(text)
            if (isinstance(event, dict) and event.get("type") == "system"
                    and event.get("subtype") == "thinking_tokens"):
                continue
            if not isinstance(event, dict):
                raw.write(line)
                raw.flush()
                if cli is not None and text:
                    cli.write(text + "\n")
                    cli.flush()
                continue
            kept = lean_raw(event)
            if kept is not None:
                raw.write(json.dumps(kept, separators=(",", ":"),
                                     ensure_ascii=False) + "\n")
                raw.flush()
            if (agent_cli == "codex" and event.get("type") == "thread.started"
                    and event.get("thread_id") and on_session is not None):
                on_session(str(event["thread_id"]))
            canonical = lean_stream(_canonical_event(event, agent_cli))
            if canonical is None:
                continue
            now = time.time()
            stream.write(json.dumps({"t": now, "event": canonical},
                                    ensure_ascii=False) + "\n")
            stream.flush()
            written += 1
            if cli is not None:
                with contextlib.suppress(Exception):
                    shown = render(canonical, now)
                    if shown:
                        cli.write(shown)
                        cli.flush()
            if watcher is not None:
                names.update(capture.tool_names(canonical))
                for tool_use_id in capture.tool_results(canonical):
                    with contextlib.suppress(OSError, ValueError):
                        watcher.scan(after_tool_use_id=tool_use_id,
                                     tool_name=names.get(tool_use_id, ""),
                                     when=now)
    if watcher is not None:
        # Whatever the last turn wrote without a tool result after it.
        with contextlib.suppress(OSError, ValueError):
            watcher.scan(after_tool_use_id="", tool_name="(launch ended)")
    return written


#: Where a launch's prompt is staged INSIDE the container.
#:
#: Not `/workspace`: the agent lists that directory, and a file it did not
#: create reads as part of the task. `/tmp` inside the container is the
#: agent's own and is not mounted from anywhere.
PROMPT_IN_CONTAINER = "/tmp/.serverbench-prompt.{n}.txt"


def stage_prompt(*, container: str, text: str, index: int) -> str:
    """Put the prompt in a FILE in the container and return its path.

    ## Why the prompt may not travel on the argv

    `claude ... --print -- '<the whole brief>'` would put every word of the
    task into the command line of the one process whose death ends the run,
    inside a container where the agent has a shell and is expected to manage
    its own background benchmarks. An agent cleaning up with

        pkill -f "aiperf profile"

    would match the `claude` process carrying a brief that mentions
    `aiperf profile`, and kill itself. With the prompt on stdin the argv holds
    flags and paths only, and no pattern drawn from the task can match the
    process running it.

    Staged with `docker exec -i` over stdin rather than through a shell
    heredoc, so no quoting in the brief can terminate it early.
    """
    path = PROMPT_IN_CONTAINER.format(n=index)
    try:
        backend.put_file(container, text, path)
    except RuntimeError as exc:
        detail = str(exc)[:400]
        # SAY WHICH KIND OF UNAVAILABLE THIS IS.
        #
        # Two very different things reach here. A session the platform will not
        # resume is recoverable -- a cold restart in the same workspace, which
        # the driver already does. A container that no longer EXISTS is not: no
        # prompt can be staged into it and no `docker exec` will ever succeed
        # ("Error response from daemon: container ... is not running"), even
        # though the workspace is intact on the host.
        gone = any(marker in detail for marker in _CONTAINER_GONE)
        raise ContinuationUnavailable(
            f"could not stage the agent prompt into {container}: {detail}"
            + ("\n\nTHE CONTAINER NO LONGER EXISTS. This is not a session "
               "that cannot be resumed -- it is a missing container, and no "
               "number of execs will fix it. The workspace survives on the "
               "host, so the remaining budget is recoverable by standing a "
               "new container on it; the harness does not do that "
               "automatically."
               if gone else "")) from exc
    return path


#: Tools that promise the harness will invoke the agent AGAIN later. Nothing
#: here does: an evolve agent runs under `claude --print`, one turn per
#: launch, and there is no interactive session to deliver a wakeup to.
#:
#:   Next wakeup scheduled for 09:25:00 (in 907s). Nothing more to do this
#:   turn -- the harness re-invokes you when the wakeup fires or a
#:   task-notification arrives.
#:   Waiting.
#:
#: The wakeup never fires. The agent ends its turn, the driver relaunches it,
#: and it schedules another and ends again, spinning through launches while a
#: benchmark it correctly started is still running.
#:
#: What the workspace gives instead is `wait/`: blocking waiters that return
#: the moment their condition holds, inside one turn.
NO_CALLBACK_TOOLS = ("ScheduleWakeup", "CronCreate", "CronDelete", "CronList")


def launch(*, identity: RunIdentity, journal: Journal, container: str,
           brief: str, model_flag: str = "", effort_flag: str = "",
           agent_cli: str = "claude", agent_model: str = "",
           agent_effort: str = "",
           workdir: str = "/workspace", poll: float = 5.0,
           on_tick=None, cold_start: bool = False,
           resume_session_id: str = "", idle_restart: bool = False,
           task_dir: Path | None = None) -> AgentRun:
    """Start, or continue, the one coding agent for this run.

    Blocks until the agent exits or the run's deadline passes. `on_tick` is
    called while it runs, which is where round boundaries are taken: the
    lifecycle drives the boundary, the agent is not asked to stop for it.
    """
    continued = bool(resume_session_id or journal.last("agent_launched")) \
        and not cold_start
    if agent_cli == "claude":
        session_id = (new_session(journal) if cold_start
                      else ensure_session(journal))
    else:
        # Codex chooses its thread id and reports it in `thread.started`.
        session_id = "" if cold_start else (
            resume_session_id or session_of(journal))
        if continued and not session_id:
            raise ContinuationUnavailable(
                "Codex emitted no thread id for its first launch, so this run "
                "cannot truthfully resume the same session")
    budget = remaining_seconds(
        identity, paused_s=billing.paused_seconds(journal))
    if budget <= 0:
        raise ContinuationUnavailable(
            f"the run's deadline {identity.deadline_utc} has passed; there is "
            f"no budget to continue into")

    if task_dir is None and (cold_start or resume_session_id or continued):
        raise RuntimeError(
            "a continued, restarted or make-up launch needs task_dir: the "
            "prompt is the task's `instruction.md`, not this module's")

    # THE WORDS ARE THE TASK'S, not this module's.
    #
    # Every prompt lives in `serving_tasks/<TASK>/instruction.md`. This module
    # selects a section by name and substitutes two numbers.
    if cold_start:
        prompt = instructions.session_prompt(
            task_dir, kind="cold-restart", budget_seconds=budget,
            deadline_utc=identity.deadline_utc)
    elif idle_restart:
        # The same session, resumed, but told the one environment fact an
        # idle agent is missing. Selected by name so it can never fall back to
        # the plain continuation's words.
        prompt = instructions.session_prompt(
            task_dir, kind="idle-restart", budget_seconds=budget,
            deadline_utc=identity.deadline_utc)
    elif resume_session_id:
        prompt = instructions.session_prompt(
            task_dir, kind="makeup", budget_seconds=budget,
            deadline_utc=identity.deadline_utc)
    elif continued:
        prompt = instructions.session_prompt(
            task_dir, kind="continuation", budget_seconds=budget,
            deadline_utc=identity.deadline_utc)
    else:
        prompt = brief
    stdout_path = identity.root / (
        f"agent.{'continue' if continued else 'start'}."
        f"{len(journal.events('agent_launched'))}.log")

    # The prompt goes in over stdin from a file in the container. See
    # `stage_prompt`: on the argv it is a `pkill -f` pattern the agent can
    # match against itself.
    prompt_path = stage_prompt(container=container, text=prompt,
                               index=len(journal.events("agent_launched")))
    if agent_cli == "codex":
        common = [
            "--ignore-user-config", "--strict-config", "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
        ]
        common += codex_config(agent_model, agent_effort)
        if continued:
            command = ["codex", "exec", "resume", *common, "--json",
                       session_id, "-"]
        else:
            command = ["codex", "exec", *common, "-C", workdir, "--json", "-"]
        argv = backend.exec_argv(
            container, workdir,
            f"{shlex.join(command)} < {shlex.quote(prompt_path)}")
    else:
        session_flag = (f"--resume {shlex.quote(session_id)}" if continued
                        else f"--session-id {shlex.quote(session_id)}")
        argv = backend.exec_argv(
            container, workdir,
            f"claude --dangerously-skip-permissions "
            f"{model_flag} {effort_flag} {session_flag} "
            f"--disallowed-tools {' '.join(NO_CALLBACK_TOOLS)} "
            # Subagents' own text and thinking, not only their tool calls.
            f"--forward-subagent-text "
            f"--output-format stream-json --verbose --print "
            f"< {shlex.quote(prompt_path)}")
    # The argv does not carry the prompt, so the evidence carries its
    # digest instead: what was sent is still pinned, and still checkable
    # against `workspace/AGENT_TASK.md`.
    journal.append("agent_launched", session_id=session_id, continued=continued,
                   agent_cli=agent_cli, agent_model=agent_model,
                   agent_effort=agent_effort, cold_start=cold_start,
                   imported_session=bool(resume_session_id),
                   budget_seconds=round(budget, 1), container=container,
                   argv=argv, prompt_path=prompt_path,
                   prompt_chars=len(prompt),
                   prompt_sha256=hashlib.sha256(
                       prompt.encode("utf-8")).hexdigest())

    started = time.monotonic()
    # The stream is teed as it arrives: the raw log for reading, and a
    # timestamped `agent.stream.jsonl` that the ATIF collector consumes. It is
    # APPENDED across launches, so a continuation extends one trajectory rather
    # than starting a second one -- which is what makes the continued session
    # legible afterwards as a single run.
    stream_path = identity.root / "agent.stream.jsonl"

    process = subprocess.Popen(argv, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    resolved = {"session_id": session_id}

    def bind_session(found: str) -> None:
        if resolved["session_id"]:
            return
        resolved["session_id"] = found
        journal.append("agent_session", session_id=found,
                       discovered_utc=utc_now(), provider="codex")

    reader = threading.Thread(target=tee_stream,
                              args=(process.stdout, stdout_path, stream_path),
                              kwargs={"agent_cli": agent_cli,
                                      "on_session": bind_session,
                                      "workspace": identity.root / "workspace",
                                      "changes_path": (identity.root
                                                       / "agent.changes.jsonl"),
                                      "cli_path": identity.root / "agent.cli.log",
                                      "cli_header": (
                                          f"{stdout_path.stem} · {utc_now()} · "
                                          f"session {session_id}")},
                              daemon=True,
                              name="agent-stream")
    reader.start()
    journal.append("owned_process", pid=process.pid, what="coding agent",
                   argv=argv)
    while True:
        code = process.poll()
        if code is not None:
            break
        if time.monotonic() - started >= budget:
            process.terminate()
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
            code = process.returncode if process.returncode is not None else -15
            journal.append("agent_budget_expired",
                           seconds=round(time.monotonic() - started, 1))
            break
        if on_tick is not None:
            on_tick(round(time.monotonic() - started, 1))
        time.sleep(poll)

    reader.join(timeout=30)
    seconds = round(time.monotonic() - started, 1)
    # The CLI's own transcripts -- the main session, every subagent sidechain,
    # background-task outputs -- before anything can remove the container.
    # Never fatal: a failed copy is recorded, and the stream is still complete.
    try:
        from . import trajectory_capture
        harvest = trajectory_capture.harvest_native(
            exec_argv=backend.exec_argv, container=container,
            dest=identity.root / "agent-native")
    except Exception as exc:                               # noqa: BLE001
        harvest = {"harvested": False, "why": repr(exc)[:300]}
    journal.append("agent_transcripts_harvested", **harvest)
    left = remaining_seconds(
        identity, paused_s=billing.paused_seconds(journal))
    # "Early" means the agent stopped while the run still had time. That is a
    # continuation opportunity, not a failure, and not a completed run.
    #
    # The exit code is deliberately NOT part of this: requiring `code == 0`
    # would exclude the very case that most needs continuing, an agent that
    # crashed with budget left. A crash with budget remaining is an early
    # stop; what differs is only whether resuming is likely to help.
    stopped_early = left > EARLY_STOP_MARGIN_S
    session_id = resolved["session_id"]
    run = AgentRun(session_id=session_id, continued=continued,
                   exit_code=int(code), seconds=seconds,
                   budget_seconds=round(budget, 1),
                   stdout_path=str(stdout_path), stopped_early=stopped_early,
                   terminal_because=terminal_failure(stdout_path))
    journal.append("agent_finished", **run.to_dict(),
                   remaining_seconds=round(left, 1))
    return run


def can_continue(*, container: str, session_id: str,
                 agent_cli: str = "claude") -> bool:
    """Ask the platform whether the recorded session can actually be resumed.

    Checked rather than assumed: a continuation that silently becomes a cold
    start is the failure this whole module exists to prevent.
    """
    if agent_cli == "codex":
        if not session_id:
            return False
        got = subprocess.run(
            backend.exec_argv(
                container, "",
                "find -L /home/agent/.codex/sessions -type f "
                f"-name '*{shlex.quote(session_id)}*.jsonl' -print -quit"),
            capture_output=True, text=True, timeout=60, check=False)
        return got.returncode == 0 and bool(got.stdout.strip())
    try:
        got = subprocess.run(
            backend.exec_argv(
                container, "",
                f"claude --resume {shlex.quote(session_id)} --print "
                f"-- 'Reply with the single word OK and nothing else.'"),
            capture_output=True, text=True, timeout=240, check=False)
    except subprocess.TimeoutExpired:
        # A resume re-sends the whole session, and a 1M-token session can take
        # longer than the probe's timeout to answer. A probe that cannot
        # answer is a session that cannot be resumed.
        return False
    return got.returncode == 0 and "OK" in got.stdout
