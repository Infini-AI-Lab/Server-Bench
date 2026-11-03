"""What the agent did, recorded precisely and without the bulk that says nothing.

Four records sit beside `agent.stream.jsonl` and are written by the same
launch that writes it:

* **the stream itself, made lean** (`LeanFilter`). A raw stream is mostly
  `tool_progress` and `task_progress` heartbeats that only say something is
  still running, `rate_limit_event`s repeating the same status, opaque
  thinking-block `signature`s (most blocks carry a signature and NO text), a
  second copy of every tool result under `tool_use_result`, per-event envelope
  (session id, uuid, null message fields) and tool outputs up to hundreds of
  KB. None of that is behaviour. Everything that is -- every
  text block, the thinking SUMMARY exactly as the CLI prints it, every tool call
  with its full input, every tool result (bounded), every subagent message
  (`parent_tool_use_id`), the start / backgrounding / end of every subagent and
  background task, every usage record -- is kept, with the harness's receive
  time.

  Thinking is recorded only as the CLI displays it. A block with no displayed
  text is dropped outright, and nothing about a signature or a redacted block
  -- not even its length -- is kept.

* **`agent.cli.log`** (`CliRenderer`). The same events as a person reads them in
  a terminal: the agent's text, its displayed thinking summary, each tool call
  and the head of its result, and every subagent's own lines indented under the
  call that opened it. Appended across launches, readable while the run is
  live, and a fraction of the JSON's size.

* **`agent.changes.jsonl`** (`WorkspaceWatcher`). An agent changes code with
  far more Bash edits (heredocs, `sed -i`, python `str.replace`) than `Write`
  calls, so the stream shows the command but never the resulting diff. After
  every tool result the watcher diffs the workspace and writes one record per
  changed file: time, the tool call that preceded it, and a unified
  diff.

* **`agent-native/`** (`harvest_native`). The CLI's own session transcript,
  including every subagent's sidechain transcript and the output files of
  background tasks, copied out of the container when each launch ends. The
  container is not kept forever, and these would be lost with it.

Stdlib only: this module is imported by the harness core.
"""

from __future__ import annotations

import difflib
import gzip
import hashlib
import io
import json
import os
import shlex
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Any

#: A tool result longer than this is kept as head + tail with its length and
#: digest. 16 KB holds a compiler error or test summary; what it cuts is log
#: spew and file dumps.
TOOL_RESULT_LIMIT = 16_384
TOOL_RESULT_HEAD = 10_240
TOOL_RESULT_TAIL = 4_096

#: Event kinds that carry no behaviour. `tool_progress` is a 30-second
#: heartbeat for a running tool; its start and its result are both kept.
DROPPED_EVENT_TYPES = frozenset({"tool_progress", "stream_event"})

#: `system` subtypes that only restate something recorded elsewhere:
#: `task_progress` is a subagent's or background task's heartbeat (the
#: subagent's own tool calls are in the stream, and its final usage is on the
#: `task_notification`), `background_tasks_changed` is a snapshot of the set
#: that `task_started` / `task_notification` already open and close, and
#: `thinking_tokens` is a per-token estimate.
DROPPED_SYSTEM_SUBTYPES = frozenset({"task_progress", "background_tasks_changed",
                                     "thinking_tokens"})

#: Envelope fields repeated on every event. Kept on the events that identify a
#: launch (`system/init`, `result`), where they are said once.
_ENVELOPE_NOISE = ("session_id", "uuid")

#: A background shell task's `description` / `summary` is the command again,
#: which the `Bash` call that started it already carries in full.
_SHELL_TASK_TEXT = 300


def _clip(text: str, limit: int = TOOL_RESULT_LIMIT) -> tuple[str, dict[str, Any] | None]:
    if len(text) <= limit:
        return text, None
    head = min(TOOL_RESULT_HEAD, limit * 5 // 8)
    tail = min(TOOL_RESULT_TAIL, limit // 4)
    digest = hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()
    omitted = len(text) - head - tail
    return (text[:head]
            + f"\n[... {omitted} characters omitted by the trajectory recorder ...]\n"
            + text[-tail:],
            {"original_chars": len(text), "sha256": digest})


def _lean_block(block: Any, limit: int = TOOL_RESULT_LIMIT) -> Any:
    """One content block, or None when it carries nothing a reader can see.

    Thinking is kept exactly as the CLI displays it -- the summarised text --
    and never more: the signature is dropped, a block the CLI shows no text
    for is dropped, and a `redacted_thinking` block is dropped. No length,
    count or digest of what was not displayed is recorded.
    """
    if not isinstance(block, dict):
        return block
    kind = block.get("type")
    if kind == "redacted_thinking":
        return None
    if kind == "thinking":
        text = block.get("thinking")
        return {"type": "thinking", "thinking": text} if text else None
    if kind == "image":
        return {"type": "image", "omitted": True}
    if kind == "tool_result":
        out = dict(block)
        content = block.get("content")
        if isinstance(content, str):
            out["content"], clipped = _clip(content, limit)
            if clipped:
                out["clipped"] = clipped
        elif isinstance(content, list):
            parts = []
            for part in content:
                if (isinstance(part, dict) and part.get("type") == "text"
                        and isinstance(part.get("text"), str)):
                    text, clipped = _clip(part["text"], limit)
                    part = {**part, "text": text}
                    if clipped:
                        part["clipped"] = clipped
                elif isinstance(part, dict) and part.get("type") == "image":
                    part = {"type": "image", "omitted": True}
                parts.append(part)
            out["content"] = parts
        return out
    return block


#: Token counts are kept; the rest of `usage` (tier, geography, the 5m/1h
#: split of cache writes) is account plumbing.
_USAGE_KEYS = ("input_tokens", "output_tokens", "cache_creation_input_tokens",
               "cache_read_input_tokens")


def _lean_message(message: dict, limit: int) -> dict:
    out = {}
    for key, value in message.items():
        if value is None or value == [] or value == {}:
            continue
        if key == "content" and isinstance(value, list):
            value = [b for b in (_lean_block(b, limit) for b in value) if b is not None]
        elif key == "usage" and isinstance(value, dict):
            value = {k: value[k] for k in _USAGE_KEYS if value.get(k)}
        elif key in ("type", "stop_sequence", "container"):
            continue
        out[key] = value
    return out


class LeanFilter:
    """Stateful: a rate-limit event is kept only when its status changes.

    `result_limit` bounds each tool result. The stream keeps 16 KB of each; the
    per-launch raw log, which exists for the driver's own detectors and a
    person skimming it, can afford much less.
    """

    def __init__(self, *, result_limit: int = TOOL_RESULT_LIMIT) -> None:
        self._rate_status: tuple | None = None
        self._limit = result_limit
        self._shell_tasks: set[str] = set()

    def __call__(self, event: dict) -> dict | None:
        kind = event.get("type")
        if kind in DROPPED_EVENT_TYPES:
            return None
        if kind == "system" and event.get("subtype") in DROPPED_SYSTEM_SUBTYPES:
            return None
        if kind == "rate_limit_event":
            info = event.get("rate_limit_info") or {}
            status = (info.get("status"), info.get("rateLimitType"),
                      info.get("isUsingOverage"))
            if status == self._rate_status:
                return None
            self._rate_status = status
            return self._envelope(event)
        if kind in ("assistant", "user"):
            event = dict(event)
            # The CLI repeats every tool result here in a second shape.
            event.pop("tool_use_result", None)
            message = event.get("message")
            if isinstance(message, dict):
                event["message"] = _lean_message(message, self._limit)
                if (kind == "assistant"
                        and isinstance(message.get("content"), list)
                        and not event["message"].get("content")):
                    # Only a thinking block the CLI showed nothing of.
                    return None
            return self._envelope(event)
        if kind == "system" and event.get("subtype", "").startswith("task_"):
            event = dict(event)
            if event.get("task_type") == "local_bash":
                self._shell_tasks.add(str(event.get("task_id")))
            if str(event.get("task_id")) in self._shell_tasks:
                for key in ("description", "summary"):
                    if isinstance(event.get(key), str):
                        event[key] = event[key][:_SHELL_TASK_TEXT]
        return self._envelope(event)

    @staticmethod
    def _envelope(event: dict) -> dict:
        if event.get("type") == "result" or event.get("subtype") == "init":
            return event
        return {k: v for k, v in event.items()
                if k not in _ENVELOPE_NOISE and v is not None}


# --- the CLI's output, as a person reads it -------------------------------------

#: How much of a tool result the rendered log shows, as the CLI does.
_CLI_RESULT_LINES = 6
_CLI_LINE_CHARS = 240


def _one_line(text: Any, limit: int = _CLI_LINE_CHARS) -> str:
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[:limit - 1] + "…"


def _call_summary(name: str, args: Any) -> str:
    if not isinstance(args, dict):
        return _one_line(args)
    if name in ("Agent", "Task"):
        return (f"{args.get('subagent_type') or 'agent'}: "
                f"{_one_line(args.get('description'), 160)}")
    for key in ("command", "file_path", "pattern", "url", "query", "to",
                "description", "skill", "task_id"):
        if args.get(key):
            return _one_line(args[key])
    return _one_line(json.dumps(args, ensure_ascii=False))


def _result_text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content
                       if isinstance(p, dict) and p.get("type") == "text")
    return str(content or "")


class CliRenderer:
    """One lean event -> the lines a terminal would have shown for it.

    Subagents are told apart by `parent_tool_use_id`: the `Agent` call that
    opened one names it, and every line it produces is indented under that
    name, as the CLI nests them.
    """

    def __init__(self) -> None:
        self._agents: dict[str, str] = {}      # Agent call id -> label
        self._tasks: dict[str, str] = {}       # task id -> label
        self._foreground: set[str] = set()     # shell tasks not backgrounded

    def _who(self, event: dict) -> tuple[str, str]:
        parent = event.get("parent_tool_use_id")
        if not parent:
            return "", ""
        label = self._agents.get(parent, f"subagent {str(parent)[-8:]}")
        return "    │ ", f"[{label}] "

    def __call__(self, event: dict, when: float | None = None) -> str:
        stamp = time.strftime("%H:%M:%S", time.gmtime(when or time.time()))
        pad, who = self._who(event)
        lead = f"{stamp} {pad}{who}"
        cont = " " * len(stamp) + " " + pad + " " * len(who)
        kind = event.get("type")
        lines: list[str] = []

        if kind == "assistant":
            for block in (event.get("message") or {}).get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and block.get("text", "").strip():
                    body = block["text"].strip().splitlines()
                    lines.append(f"{lead}● {body[0]}")
                    lines += [f"{cont}  {line}" for line in body[1:]]
                elif block.get("type") == "thinking" and block.get("thinking"):
                    body = block["thinking"].strip().splitlines()
                    lines.append(f"{lead}∴ Thinking: {body[0]}")
                    lines += [f"{cont}  {line}" for line in body[1:]]
                elif block.get("type") == "tool_use":
                    name = block.get("name", "tool")
                    args = block.get("input")
                    if name in ("Agent", "Task") and isinstance(args, dict):
                        self._agents[block.get("id", "")] = (
                            args.get("subagent_type") or "agent")
                    lines.append(f"{lead}● {name}({_call_summary(name, args)})")
        elif kind == "user":
            for block in (event.get("message") or {}).get("content") or []:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                body = _result_text(block.get("content")).rstrip().splitlines()
                mark = "⎿  " + ("Error: " if block.get("is_error") else "")
                if not body:
                    lines.append(f"{cont}  {mark}(no output)")
                    continue
                shown = body[:_CLI_RESULT_LINES]
                lines.append(f"{cont}  {mark}{_one_line(shown[0])}")
                lines += [f"{cont}     {_one_line(line)}" for line in shown[1:]]
                if len(body) > len(shown):
                    lines.append(f"{cont}     … +{len(body) - len(shown)} lines")
        elif kind == "system":
            sub = event.get("subtype")
            if sub == "init":
                lines.append(f"{lead}═ session {event.get('session_id', '')} · "
                             f"{event.get('model', '')} · Claude Code "
                             f"{event.get('claude_code_version', '')}")
            elif sub in ("task_started", "task_updated", "task_notification"):
                lines.append(self._task_line(event, lead))
            elif sub == "compact_boundary":
                lines.append(f"{lead}═ context compacted")
            elif sub:
                lines.append(f"{lead}· {sub}: "
                             f"{_one_line(event.get('message') or '', 200)}")
        elif kind == "result":
            cost = event.get("total_cost_usd")
            lines.append(
                f"{lead}═ {event.get('subtype', 'result')} · "
                f"{event.get('num_turns', '?')} turns · "
                f"{(event.get('duration_ms') or 0) / 1000:.0f}s"
                + (f" · ${cost:.2f}" if isinstance(cost, (int, float)) else ""))
            if event.get("result"):
                lines += [f"{cont}  {line}" for line in
                          str(event["result"]).strip().splitlines()]
        elif kind == "rate_limit_event":
            info = event.get("rate_limit_info") or {}
            lines.append(f"{lead}· rate limit {info.get('status')} "
                         f"({info.get('rateLimitType')})")
        return "".join(line.rstrip() + "\n" for line in lines if line)

    def _task_line(self, event: dict, lead: str) -> str:
        """A subagent or background task opening, moving or closing.

        A foreground shell command is also a "task" to the CLI; its call and
        result are already on screen, so it gets a line only once it has been
        moved to the background.
        """
        sub = event.get("subtype")
        task = str(event.get("task_id", ""))
        kind = event.get("task_type") or ""
        if sub == "task_started":
            label = event.get("subagent_type") or (
                "background shell" if kind == "local_bash" else kind or "task")
            self._tasks[task] = label
            if kind == "local_agent" and event.get("tool_use_id"):
                self._agents[event["tool_use_id"]] = label
            if kind == "local_bash" and not event.get("is_backgrounded"):
                self._foreground.add(task)
                return ""
            how = " in the background" if event.get("is_backgrounded") else ""
            return (f"{lead}⧉ {label} {task} started{how}: "
                    f"{_one_line(event.get('description'), 200)}")
        label = self._tasks.get(task, "task")
        if sub == "task_updated":
            if (event.get("patch") or {}).get("is_backgrounded"):
                self._foreground.discard(task)
                return f"{lead}⧉ {label} {task} moved to the background"
            return ""
        if task in self._foreground:
            return ""
        return (f"{lead}⧉ {label} {task} {event.get('status', 'finished')}: "
                f"{_one_line(event.get('summary'), 400)}")


# --- file changes ---------------------------------------------------------------

#: Written by the harness, or produced by measurement, not authored by the
#: agent. Recording them would attribute the harness's writes to a tool call.
_SKIP_DIRS = frozenset({"agentx", ".review", ".versions", "task_context",
                        "__pycache__", ".git", ".claude", "node_modules",
                        ".pytest_cache", ".triton", "measurements_out"})
_SKIP_FILES = frozenset({"AGENT_TASK.md", "WORKSPACE.json", "MAP.md",
                         "DELIVERY_POLICY.json", "DELIVERY_REPORT.json",
                         "BOUNDARY_CHECKS.md"})
_MAX_FILE_BYTES = 1_000_000
_MAX_DIFF_CHARS = 200_000


def _is_text(data: bytes) -> bool:
    if b"\0" in data[:8192]:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


class WorkspaceWatcher:
    """Diff the agent's workspace after each tool result.

    Keeps the last seen text of each tracked file in memory (the tracked tree
    is source code: tens of files, well under a few MB) and compares by
    (mtime_ns, size) first, so an unchanged tree costs one stat per file.
    """

    def __init__(self, root: Path, out: Path) -> None:
        self.root = Path(root)
        self.out = Path(out)
        self._seen: dict[str, tuple[int, int, str | None]] = {}
        self.enabled = self.root.is_dir()
        if self.enabled:
            # The baseline: what was there when this launch started. Changes
            # made before it (previous launches) are already on record.
            for rel, stat in self._walk():
                self._seen[rel] = (stat.st_mtime_ns, stat.st_size,
                                   self._read(rel, stat))

    def _walk(self):
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for name in filenames:
                path = Path(dirpath) / name
                rel = str(path.relative_to(self.root))
                if rel in _SKIP_FILES or name.endswith((".pyc", ".log")):
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                if not path.is_file():
                    continue
                yield rel, stat

    def _read(self, rel: str, stat: os.stat_result) -> str | None:
        if stat.st_size > _MAX_FILE_BYTES:
            return None
        try:
            data = (self.root / rel).read_bytes()
        except OSError:
            return None
        return data.decode("utf-8") if _is_text(data) else None

    def scan(self, *, after_tool_use_id: str = "", tool_name: str = "",
             when: float | None = None) -> int:
        if not self.enabled:
            return 0
        when = time.time() if when is None else when
        records = []
        present = set()
        for rel, stat in self._walk():
            present.add(rel)
            old = self._seen.get(rel)
            if old is not None and old[:2] == (stat.st_mtime_ns, stat.st_size):
                continue
            new_text = self._read(rel, stat)
            old_text = old[2] if old else None
            self._seen[rel] = (stat.st_mtime_ns, stat.st_size, new_text)
            if old is not None and new_text == old_text and new_text is not None:
                continue            # touched, not changed
            records.append(self._record(rel, "modified" if old else "created",
                                        old_text, new_text, stat.st_size))
        for rel in [r for r in self._seen if r not in present]:
            old_text = self._seen.pop(rel)[2]
            records.append(self._record(rel, "deleted", old_text, None, 0))
        if records:
            with open(self.out, "a", encoding="utf-8") as handle:
                for record in records:
                    record.update(t=when, after_tool_use_id=after_tool_use_id,
                                  after_tool=tool_name)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return len(records)

    @staticmethod
    def _record(rel: str, op: str, old: str | None, new: str | None,
                size: int) -> dict[str, Any]:
        record: dict[str, Any] = {"path": rel, "op": op, "bytes": size}
        if new is not None:
            record["sha256"] = hashlib.sha256(new.encode("utf-8")).hexdigest()
        if old is None and new is None:
            record["binary_or_large"] = True
            return record
        diff = "".join(difflib.unified_diff(
            (old or "").splitlines(keepends=True),
            (new or "").splitlines(keepends=True),
            fromfile=f"a/{rel}", tofile=f"b/{rel}", n=2))
        added = sum(1 for line in diff.splitlines()
                    if line.startswith("+") and not line.startswith("+++"))
        removed = sum(1 for line in diff.splitlines()
                      if line.startswith("-") and not line.startswith("---"))
        record.update(lines_added=added, lines_removed=removed)
        if len(diff) > _MAX_DIFF_CHARS:
            record["diff"] = diff[:_MAX_DIFF_CHARS]
            record["diff_truncated_from"] = len(diff)
        else:
            record["diff"] = diff
        return record


def tool_results(event: dict) -> list[str]:
    """The tool_use ids whose results this event carries."""
    if event.get("type") != "user":
        return []
    message = event.get("message") or {}
    content = message.get("content") if isinstance(message, dict) else None
    return [b.get("tool_use_id", "") for b in content or []
            if isinstance(b, dict) and b.get("type") == "tool_result"]


def tool_names(event: dict) -> dict[str, str]:
    if event.get("type") != "assistant":
        return {}
    message = event.get("message") or {}
    content = message.get("content") if isinstance(message, dict) else None
    return {b.get("id", ""): b.get("name", "") for b in content or []
            if isinstance(b, dict) and b.get("type") == "tool_use"}


# --- the CLI's own transcripts ------------------------------------------------------

#: A background task's output file is kept up to this many trailing bytes.
_TASK_OUTPUT_TAIL = 262_144

_HARVEST_SCRIPT = r"""
set -e
stage=$(mktemp -d /tmp/.sb-harvest.XXXXXX)
for src in /home/agent/.claude/projects "$HOME/.claude/projects" /home/agent/.codex/sessions; do
  [ -d "$src" ] || continue
  kind=projects; case "$src" in *codex*) kind=codex-sessions;; esac
  mkdir -p "$stage/$kind"
  (cd "$src" && find . -type f \( -name '*.jsonl' -o -name '*.json' -o -name '*.md' \) -print0 \
     | tar --null -T - -cf - ) | tar -C "$stage/$kind" -xf -
done
mkdir -p "$stage/tasks"
find /tmp -maxdepth 5 -type f -path '*/tasks/*.output' 2>/dev/null | while read -r f; do
  d="$stage/tasks/$(echo "$f" | sed 's#^/tmp/##; s#/#__#g')"
  tail -c {tail} "$f" > "$d" 2>/dev/null || true
done
tar -C "$stage" -czf - .
rm -rf "$stage"
"""


def harvest_native(*, exec_argv, container: str, dest: Path,
                   timeout_s: float = 300.0) -> dict[str, Any]:
    """Copy the CLI's session transcripts and background-task outputs out.

    `exec_argv(container, workdir, command)` is the backend's own exec. The
    session file is rewritten on every launch (a resumed session appends to
    it), so `dest` holds the latest complete copy, not one per launch.
    Transcripts are stored gzip-compressed and passed through the same lean
    filter's block rules (no thinking signatures, bounded tool results).
    """
    command = _HARVEST_SCRIPT.replace("{tail}", str(_TASK_OUTPUT_TAIL))
    try:
        got = subprocess.run(
            exec_argv(container, "", f"bash -c {shlex.quote(command)}"),
            capture_output=True, timeout=timeout_s, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"harvested": False, "why": f"{type(exc).__name__}: {exc}"}
    if got.returncode != 0 or not got.stdout:
        return {"harvested": False,
                "why": (got.stderr or b"").decode("utf-8", "replace")[-400:]}
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    files = 0
    raw_bytes = 0
    kept_bytes = 0
    with tarfile.open(fileobj=io.BytesIO(got.stdout), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            name = os.path.normpath(member.name).lstrip("./")
            if name.startswith("..") or os.path.isabs(name):
                continue
            data = archive.extractfile(member).read()
            raw_bytes += len(data)
            if name.endswith(".jsonl") and name.startswith(("projects/", "codex-sessions/")):
                data = _lean_transcript(data)
            target = dest / (name + ".gz")
            target.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(target, "wb") as handle:
                handle.write(data)
            kept_bytes += target.stat().st_size
            files += 1
    return {"harvested": True, "files": files, "raw_bytes": raw_bytes,
            "stored_bytes": kept_bytes, "dest": str(dest)}


def _lean_transcript(data: bytes) -> bytes:
    """Apply the stream's block rules to the CLI's own transcript format."""
    out = io.StringIO()
    for line in data.decode("utf-8", "replace").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            out.write(line + "\n")
            continue
        if isinstance(entry, dict):
            entry.pop("toolUseResult", None)
            message = entry.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), list):
                entry["message"] = {**message, "content": [
                    b for b in (_lean_block(b) for b in message["content"])
                    if b is not None]}
        out.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return out.getvalue().encode("utf-8")
