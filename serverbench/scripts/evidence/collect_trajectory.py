#!/usr/bin/env python3
"""The agent layer of a run's trajectory, in the Agent Trajectory Interchange Format.

`harbor/rfcs/0001-trajectory-format.md` specifies ATIF v1.7, and the vendored
Harbor carries its Pydantic models and a validator. ATIF describes what an agent
did: messages, tool calls, observations, tokens. Emitting it makes a run's
trajectory readable by the same tools that read SWE-agent, OpenHands and Letta
trajectories, for debugging, visualisation, SFT and RL.

What a run did with what the agent built (candidates,
the accepted head) is not a message, so it goes in ATIF's own `extra` field as
`extra.evolution`. That half is read from the run's journal by
`lifecycle/report.py`, which imports `build_steps`, `read_jsonl` and
`_final_metrics` from here; the document still validates against the standard.

Usage:
  collect_trajectory.py --run <run dir>              # rewrite <run>/trajectory.atif.json
  collect_trajectory.py --run <run dir> --validate   # and check it against Harbor's models
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
ATIF_VERSION = "ATIF-v1.7"


def iso(epoch: float) -> str:
    """Millisecond resolution: a tool call and its result are often <1 s apart."""
    stamp = dt.datetime.fromtimestamp(epoch, dt.timezone.utc)
    return stamp.strftime("%Y-%m-%dT%H:%M:%S.") + f"{stamp.microsecond // 1000:03d}Z"


#: CLI lifecycle events for subagents and background tasks, kept as `system`
#: steps so the creation, backgrounding and closing of each one is on the
#: timeline of the agent that opened it. `task_progress` and
#: `background_tasks_changed` are not here: they are heartbeats and snapshots of
#: what these already say.
_LIFECYCLE = {"task_started": "started", "task_updated": "updated",
              "task_notification": "finished"}
_LIFECYCLE_FIELDS = ("task_id", "tool_use_id", "task_type", "subagent_type",
                     "description", "prompt", "is_backgrounded", "spawn_depth",
                     "status", "summary", "output_file", "usage",
                     "last_tool_name", "patch")

#: The tools that open a subagent.
SUBAGENT_TOOLS = frozenset({"Agent", "Task"})


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text("utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# --------------------------------------------------------------------------
# The agent layer: Claude Code's stream-json -> ATIF steps
# --------------------------------------------------------------------------

def build_steps(events: list[dict],
                changes: list[dict] | None = None) -> tuple[list[dict], dict]:
    """One ATIF step per LLM inference, with observations attached to their call.

    The stream gives an `assistant` event per inference (whose content holds
    `thinking`, `text` and `tool_use` blocks) and a `user` event per batch of
    `tool_result`s. ATIF wants the result attached to the step that CAUSED it,
    correlated by `tool_call_id` <- `source_call_id`, so the results are buffered
    and matched rather than emitted as their own turns.

    ## Subagents are their own trajectories

    Every event a subagent produced carries `parent_tool_use_id`, naming the
    `Agent` call it runs inside. Those events are built into a separate,
    independently valid ATIF trajectory per call and returned in
    `meta["subagent_trajectories"]`, for the caller to embed as the root's
    `subagent_trajectories` (ATIF v1.7). The `Agent` call's own observation
    result carries the `subagent_trajectory_ref` that resolves to it. The
    opening, backgrounding and closing of each subagent stay on the timeline of
    the agent that opened it, as `system` steps. A subagent that opens its own
    subagent nests the same way, one level down.

    ## Thinking

    `reasoning_content` is the thinking text the CLI displayed -- the summary --
    and nothing else. Blocks it showed no text for contribute nothing.
    """
    #: Whose timeline each step goes on: "" is the main agent, otherwise the
    #: `Agent` call id the subagent runs inside.
    lanes: dict[str, list[dict]] = {"": []}
    pending: dict[str, dict] = {}        # tool_call_id -> the step that made it
    call_lane: dict[str, str] = {}       # tool_call_id -> the lane it was made in
    agent_calls: dict[str, dict] = {}    # Agent call id -> its tool_call
    totals = {"prompt": 0, "completion": 0, "cached": 0}
    model_name = ""
    version = ""
    session = ""
    lane_models: dict[str, str] = {}
    #: Every `Agent` call a subagent ran inside, in the order they were opened.
    #: Reported even when empty, so "this agent worked alone" and "subagent
    #: activity was not collected" stay different answers.
    subagents: list[str] = []
    #: Shell tasks that stayed in the foreground: their call and result are the
    #: `Bash` step itself, so their lifecycle events add nothing.
    shell_tasks: dict[str, bool] = {}    # task id -> ever backgrounded
    tasks: dict[str, dict] = {}          # local_agent task id -> what is known
    #: `agent.changes.jsonl`, by the tool call each change followed.
    changed: dict[str, list[dict]] = {}
    for change in changes or []:
        changed.setdefault(str(change.get("after_tool_use_id") or ""), []).append({
            k: change.get(k) for k in ("path", "op", "lines_added",
                                       "lines_removed", "sha256")
            if change.get(k) is not None})

    for record in events:
        event = record.get("event") or {}
        if (event.get("type") == "system" and event.get("subtype") == "task_started"
                and event.get("task_type") == "local_bash"):
            shell_tasks[str(event.get("task_id"))] = bool(event.get("is_backgrounded"))
        if event.get("type") == "system" and event.get("subtype") == "task_updated":
            task = str(event.get("task_id"))
            if task in shell_tasks and (event.get("patch") or {}).get("is_backgrounded"):
                shell_tasks[task] = True

    def lane_of(parent: str | None) -> str:
        key = str(parent or "")
        if key and key not in lanes:
            lanes[key] = []
            subagents.append(key)
        return key

    for record in events:
        event = record.get("event") or {}
        kind = event.get("type")
        when = record.get("t")

        if kind == "system" and event.get("subtype") == "init":
            version = version or str(event.get("claude_code_version") or "")
            session = session or str(event.get("session_id") or "")

        if kind == "assistant":
            message = event.get("message") or {}
            lane = lane_of(event.get("parent_tool_use_id"))
            steps = lanes[lane]
            if message.get("model"):
                if lane:
                    lane_models.setdefault(lane, message["model"])
                else:
                    model_name = message["model"]
            blocks = message.get("content") or []
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            thinking = "".join(b.get("thinking", "") for b in blocks
                               if b.get("type") == "thinking")
            calls = [b for b in blocks if b.get("type") == "tool_use"]
            if not (text or thinking or calls):
                continue                         # nothing the CLI showed

            usage = message.get("usage") or {}
            prompt = int(usage.get("input_tokens") or 0) + \
                int(usage.get("cache_creation_input_tokens") or 0)
            completion = int(usage.get("output_tokens") or 0)
            cached = int(usage.get("cache_read_input_tokens") or 0)
            totals["prompt"] += prompt
            totals["completion"] += completion
            totals["cached"] += cached

            step: dict[str, Any] = {
                "step_id": len(steps) + 1,
                "source": "agent",
                "message": text,
                "llm_call_count": 1,
            }
            if when:
                step["timestamp"] = iso(when)
            if message.get("model"):
                step["model_name"] = message["model"]
            if thinking:
                step["reasoning_content"] = thinking
            metrics = {k: v for k, v in (("prompt_tokens", prompt),
                                         ("completion_tokens", completion),
                                         ("cached_tokens", cached)) if v}
            if metrics:
                step["metrics"] = metrics
            if calls:
                step["tool_calls"] = [{
                    "tool_call_id": c.get("id", f"call-{lane}-{len(steps)}-{i}"),
                    "function_name": c.get("name", "unknown"),
                    "arguments": c.get("input") or {},
                } for i, c in enumerate(calls)]
                for c in step["tool_calls"]:
                    pending[c["tool_call_id"]] = step
                    call_lane[c["tool_call_id"]] = lane
                    if c["function_name"] in SUBAGENT_TOOLS:
                        agent_calls[c["tool_call_id"]] = c
            steps.append(step)

        elif kind == "user":
            message = event.get("message") or {}
            lane = lane_of(event.get("parent_tool_use_id"))
            for block in (message.get("content") or []):
                if block.get("type") != "tool_result":
                    continue
                call_id = block.get("tool_use_id")
                content = block.get("content")
                if isinstance(content, list):
                    content = "".join(p.get("text", "") for p in content
                                      if isinstance(p, dict))
                result = {"source_call_id": call_id,
                          "content": str(content or "")}
                if block.get("is_error"):
                    result.setdefault("extra", {})["is_error"] = True
                if when:
                    result.setdefault("extra", {})["timestamp"] = iso(when)
                if block.get("clipped"):
                    result.setdefault("extra", {})["clipped"] = block["clipped"]
                if call_id in changed:
                    result.setdefault("extra", {})["file_changes"] = changed[call_id]
                owner = pending.get(call_id)
                if owner is None:                      # orphan: keep it, say so
                    steps = lanes[lane]
                    owner = {"step_id": len(steps) + 1, "source": "user",
                             "message": "", "extra": {"orphan_tool_result": True}}
                    steps.append(owner)
                owner.setdefault("observation", {"results": []})
                owner["observation"]["results"].append(result)

        elif kind == "system" and event.get("subtype") in _LIFECYCLE:
            task = str(event.get("task_id") or "")
            if task in shell_tasks and not shell_tasks[task]:
                continue                         # a foreground command: the call says it
            if event.get("subtype") == "task_updated" and not any(
                    k in (event.get("patch") or {}) for k in ("is_backgrounded", "status")):
                continue
            if event.get("task_type") == "local_agent" and event.get("tool_use_id"):
                tasks.setdefault(task, {})["tool_use_id"] = event["tool_use_id"]
            known = tasks.get(task)
            if known is not None:
                known.update({k: event[k] for k in ("subagent_type", "description",
                                                    "status", "summary", "usage",
                                                    "output_file", "is_backgrounded")
                              if event.get(k) is not None})
            # On the timeline of whoever made the call: the main agent for a
            # subagent it opened, the subagent for a shell it backgrounded.
            origin = event.get("tool_use_id") or (known or {}).get("tool_use_id")
            steps = lanes[call_lane.get(str(origin), "")]
            detail = {k: event[k] for k in _LIFECYCLE_FIELDS if k in event}
            what = _LIFECYCLE[event["subtype"]]
            label = (event.get("subagent_type") or (known or {}).get("subagent_type")
                     or event.get("task_type") or "task")
            step = {"step_id": len(steps) + 1, "source": "system",
                    "message": f"{label} {what}: "
                               f"{event.get('description') or event.get('summary') or ''}"[:500],
                    "extra": {"lifecycle": event["subtype"], **detail}}
            if origin and str(origin) in agent_calls:
                step["extra"]["subagent_trajectory_id"] = f"subagent-{origin}"
            if when:
                step["timestamp"] = iso(when)
            steps.append(step)

    by_call = {v.get("tool_use_id"): {"task_id": k, **v} for k, v in tasks.items()}
    embedded: list[dict] = []
    for call_id in subagents:
        call = agent_calls.get(call_id) or {}
        args = call.get("arguments") or {}
        task = by_call.get(call_id, {})
        trajectory_id = f"subagent-{call_id}"
        steps = lanes[call_id]
        if args.get("prompt"):
            steps.insert(0, {"step_id": 0, "source": "user",
                             "message": str(args["prompt"]),
                             "extra": {"role": "subagent_prompt"}})
        for index, step in enumerate(steps):
            step["step_id"] = index + 1
        described = {k: v for k, v in (
            ("subagent_type", args.get("subagent_type") or task.get("subagent_type")),
            ("description", args.get("description") or task.get("description")),
            ("task_id", task.get("task_id")), ("status", task.get("status")),
            ("is_backgrounded", task.get("is_backgrounded")),
            ("parent_tool_call_id", call_id)) if v is not None}
        agent: dict[str, Any] = {"name": "claude-code-subagent",
                                 "version": version or "unknown",
                                 "extra": {k: described[k] for k in
                                           ("subagent_type", "description")
                                           if k in described}}
        if lane_models.get(call_id):
            agent["model_name"] = lane_models[call_id]
        document: dict[str, Any] = {
            "schema_version": ATIF_VERSION,
            "trajectory_id": trajectory_id,
            "agent": agent,
            "steps": steps,
            "final_metrics": {"total_steps": len(steps),
                              "extra": {k: task[k] for k in ("usage",)
                                        if task.get(k)}},
            "extra": {**described,
                      **{k: task[k] for k in ("summary", "output_file")
                         if task.get(k)}},
        }
        if session:
            document["session_id"] = session
        if not steps:
            continue
        embedded.append(document)
        ref = {"trajectory_id": trajectory_id,
               "extra": {k: described[k] for k in ("subagent_type", "description",
                                                   "status") if k in described}}
        if session:
            ref["session_id"] = session
        owner = pending.get(call_id)
        if owner is None:
            continue
        owner.setdefault("observation", {"results": []})
        mine = [r for r in owner["observation"]["results"]
                if r.get("source_call_id") == call_id]
        if mine:
            mine[0]["subagent_trajectory_ref"] = [ref]
        else:
            owner["observation"]["results"].append(
                {"source_call_id": call_id, "subagent_trajectory_ref": [ref]})

    return lanes[""], {"totals": totals, "model_name": model_name,
                       "subagents": subagents,
                       "subagent_trajectories": embedded}


# --------------------------------------------------------------------------

def _final_metrics(result_event: dict, per_step: dict, step_count: int) -> dict:
    """Totals from the run's own accounting, not from summing the stream.

    The per-event `usage` on an `assistant` message is a streaming snapshot, and
    summing it undercounts completion tokens by well over an order of magnitude
    against the run's `result` event, so any cost or throughput figure derived
    from it would be fiction.

    `modelUsage` is also per model, the only place a run records that its
    subagents ran on a different model from its main loop.
    """
    usage = result_event.get("modelUsage") or {}
    prompt = sum(int(v.get("inputTokens") or 0) for v in usage.values())
    completion = sum(int(v.get("outputTokens") or 0) for v in usage.values())
    cached = sum(int(v.get("cacheReadInputTokens") or 0) for v in usage.values())
    creation = sum(int(v.get("cacheCreationInputTokens") or 0) for v in usage.values())
    cost = sum(float(v.get("costUSD") or 0.0) for v in usage.values())

    if not usage:                       # no result event: fall back, and say so
        prompt, completion, cached = (per_step["prompt"], per_step["completion"],
                                      per_step["cached"])

    return {
        "total_prompt_tokens": prompt + creation,
        "total_completion_tokens": completion,
        "total_cached_tokens": cached,
        "total_cost_usd": round(cost, 6) or None,
        "total_steps": step_count,
        "extra": {
            "source": ("result.modelUsage" if usage else
                       "summed from per-step usage (no result event)"),
            "cache_creation_input_tokens": creation,
            "per_model": {name: {"output_tokens": v.get("outputTokens"),
                                 "cost_usd": v.get("costUSD")}
                          for name, v in usage.items()},
            "stream_sum_completion_tokens": per_step["completion"],
            "stream_sum_note": ("per-event usage is a streaming snapshot and "
                                "undercounts; kept only so the discrepancy is "
                                "visible rather than silently resolved"),
            "duration_ms": result_event.get("duration_ms"),
            "duration_api_ms": result_event.get("duration_api_ms"),
            "num_turns": result_event.get("num_turns"),
            "stop_reason": result_event.get("stop_reason"),
            "subagent_stats": result_event.get("subagent_stats"),
        },
    }


def validate(document: dict) -> str:
    """Check against Harbor's own Pydantic models, or say why we could not."""
    try:
        import importlib.metadata as md
        original = md.version
        md.version = lambda n: "0.0.0+vendored" if n == "harbor" else original(n)
        sys.path.insert(0, str(REPO / "harbor" / "src"))
        from harbor.models.trajectories import Trajectory
    except Exception as error:                          # noqa: BLE001
        return f"not validated ({type(error).__name__}: {error})"
    try:
        Trajectory.model_validate(document)
        return "valid ATIF-v1.7 (Harbor's models)"
    except Exception as error:                          # noqa: BLE001
        return f"INVALID: {str(error)[:400]}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True,
                    help="a run directory (it holds identity.json and journal.jsonl)")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(REPO / "serverbench" / "harness"))
    from lifecycle import record, report

    run = args.run.resolve()
    identity = record.load(run)
    journal = record.Journal(run)
    created = journal.events("run_created")
    out = report.collect(identity=identity, journal=journal,
                         agent_note=(created[0].get("agent_model", "")
                                     if created else ""))
    document = json.loads(out.read_text("utf-8"))
    print(f"  ok  {run.name}: {len(document['steps'])} step(s), "
          f"{len(document['extra']['evolution']['rounds'])} round(s) -> {out}")
    if args.validate:
        print(f"      {validate(document)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
