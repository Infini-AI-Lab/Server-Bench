"""The agent's design decisions, read out of its event stream.

`agent.stream.jsonl` holds one `{"t": <epoch seconds>, "event": {...}}` line
per stream-json event. The reasoning an agent states between a tool result and
its next change is where a design choice and its justification live, so each
entry here is the agent's own words plus the files it changed immediately
afterwards. `run.py` renders them into each version's `DECISIONS.md`.

Nothing is estimated or summarised: an entry is text the agent wrote.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

#: How a tool call is classified. Matched against the tool name and the
#: flattened text of its input, so a Bash command is read for what it does.
TOOL_KINDS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("benchmark", re.compile(
        r"\bbench\b|bench\.py|bench_|--concurrency|\bwrk\b|locust|aiperf", re.I)),
    ("serve_launch", re.compile(
        r"start_server|serve\.sh|server\.py\b|launch_server|uvicorn|"
        r"SERVER_PIDFILE|/health\b", re.I)),
    ("tests", re.compile(r"\bpytest\b|unittest|-m pytest", re.I)),
    ("edit", re.compile(r"^(Edit|Write|NotebookEdit|MultiEdit)$", re.I)),
    ("read", re.compile(r"^(Read|Grep|Glob|LS|WebFetch|WebSearch)$", re.I)),
    ("shell", re.compile(r"^Bash$", re.I)),
)

#: Tools whose NAME already says what the call was; their payload is a file
#: path or a query, so it must not be searched.
BY_NAME: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (kind, pattern) for kind, pattern in TOOL_KINDS if kind in ("edit", "read"))

#: What a shell command was doing, inferred from the command itself.
BY_PAYLOAD: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (kind, pattern) for kind, pattern in TOOL_KINDS
    if kind in ("benchmark", "serve_launch", "tests"))

#: Tool calls that CHANGE the system, as opposed to reading or measuring it.
CHANGES = ("edit",)

#: Prose that usually marks a decision.
DECISION_CUE = re.compile(
    r"\b(hypothes|decid|chose|choosing|instead of|rather than|because|"
    r"trade-?off|reverted|rejected|the reason|why\b|risk|I will|I am going to)",
    re.I)


def classify(tool: str, payload: Any) -> str:
    """What a tool call was doing. The tool name wins over its payload.

    An `Edit` of `server.py` is an edit, not a server launch: matching the
    payload of a named tool would record edits as launches and drop the
    reasoning that `decisions()` pairs with changes.
    """
    for kind, pattern in BY_NAME:
        if pattern.search(tool):
            return kind
    text = json.dumps(payload, default=str)[:4000] if payload is not None else ""
    if re.match(r"^Bash$", tool, re.I):
        for kind, pattern in BY_PAYLOAD:
            if pattern.search(text):
                return kind
        return "shell"
    for kind, pattern in TOOL_KINDS:
        if pattern.search(tool) or pattern.search(text):
            return kind
    return "other"


def _events(path: str | Path) -> list[tuple[float, dict]]:
    events: list[tuple[float, dict]] = []
    for raw in Path(path).read_text(errors="ignore").splitlines():
        raw = raw.strip()
        if not raw.startswith("{"):
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if "t" in row and "event" in row:
            events.append((float(row["t"]), row["event"]))
    return events


def decisions(path: str | Path, min_chars: int = 120) -> list[dict]:
    """The agent's stated reasoning, paired with what it changed next."""
    events = _events(path)
    if not events:
        return []
    start = events[0][0]
    out: list[dict] = []
    pending_text, pending_at = "", start
    for stamp, event in events:
        # System events use `message` for a plain sentence, not an object.
        message = event.get("message")
        if not isinstance(message, dict):
            message = {}
        blocks = message.get("content") or []
        if not isinstance(blocks, list):
            blocks = []
        if event.get("type") != "assistant":
            continue
        said = "".join(b.get("text", "") for b in blocks
                       if isinstance(b, dict) and b.get("type") == "text").strip()
        # Only substantive text displaces substantive text: short narration
        # between the reasoning and the edit it justified must not erase it.
        if said and len(said) >= min_chars:
            pending_text, pending_at = said, stamp
        touched = [
            str((b.get("input") or {}).get("file_path")
                or (b.get("input") or {}).get("command") or "")[:160]
            for b in blocks
            if isinstance(b, dict) and b.get("type") == "tool_use"
            and classify(str(b.get("name") or ""), b.get("input")) in CHANGES
        ]
        if touched and pending_text:
            out.append({
                "at_seconds": round(pending_at - start, 1),
                "said": pending_text,
                "changed": [t for t in touched if t],
                "looks_like_a_decision": bool(DECISION_CUE.search(pending_text)),
            })
            pending_text = ""
    return out


def render_decisions(entries: list[dict], limit: int = 40) -> str:
    """The decision log as Markdown."""
    if not entries:
        return ("# Design decisions\n\nNo decision text was captured for this "
                "run.\n")
    lines = [
        "# Design decisions",
        "",
        "What the agent said as it worked, paired with what it changed "
        "immediately afterwards, extracted from the run's event stream. These "
        "are the agent's own words at the moment of the change.",
        "",
        f"{len(entries)} change(s) with stated reasoning; "
        f"{sum(1 for e in entries if e['looks_like_a_decision'])} read as an "
        f"explicit decision.",
        "",
    ]
    for entry in entries[:limit]:
        mark = " — decision" if entry["looks_like_a_decision"] else ""
        lines += [f"## t+{entry['at_seconds'] / 60:.0f} min{mark}", "",
                  entry["said"].strip(), "",
                  "Changed: " + ", ".join(f"`{c}`" for c in entry["changed"]),
                  ""]
    if len(entries) > limit:
        lines.append(f"_{len(entries) - limit} further change(s) not shown; the "
                     f"full stream is in `agent.stream.jsonl`._")
    return "\n".join(lines) + "\n"
