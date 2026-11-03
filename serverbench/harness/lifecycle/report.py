"""The run's ATIF trajectory, assembled from the stream and the journal.

The agent layer is ATIF v1.7, built by
`scripts/evidence/collect_trajectory.build_steps`, so a trajectory this harness
emits has the same shape everything else reads. The *evolution* half comes from
the journal alone, because the journal is the only place the lifecycle records
anything.

A continued session extends one trajectory. `agent.stream.jsonl` is appended
across launches, so a run that stopped early and resumed reads as the single
session it was, not as two runs that happened to share a directory.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from .record import Journal, RunIdentity, candidates, run_state, utc_now
from .utils import write_atomically

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _collector():
    """Import the ATIF builder by path; `scripts/` is not a package."""
    path = _SCRIPTS / "evidence" / "collect_trajectory.py"
    spec = importlib.util.spec_from_file_location("_atif_collector", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_atif_collector", module)
    spec.loader.exec_module(module)
    return module


def write_trajectory(out: Path, document: dict) -> None:
    """Write the ATIF document so a reader never sees a prefix of it.

    `_deliver_trajectory_to_versions` `copy2`s the result into every published
    version directory, so a truncating write interrupted mid-serialisation
    could put a truncated trajectory in every version of the lineage.

    Named rather than inlined so a test can drive it: the failure being guarded
    is a crash INSIDE the serialisation, which cannot be provoked through
    `collect()` without a whole run.
    """
    write_atomically(out, json.dumps(document, indent=1, sort_keys=True) + "\n")


def evolution(journal: Journal, identity: RunIdentity) -> dict:
    """What the run learned, read from the journal and nowhere else."""
    events = list(journal)
    by: dict[str, list[dict]] = {}
    for e in events:
        by.setdefault(e["event"], []).append(e)

    versions = {v["version_id"]: v for v in by.get("version_registered", [])}
    for outcome in by.get("version_outcome", []):
        versions.setdefault(outcome["version_id"], {}).update(
            {"status": outcome["status"],
             "qualification": outcome.get("qualification", {})})

    rounds = []
    for cap in by.get("candidate_captured", []):
        cid = cap["candidate_id"]
        registered = next((v for v in by.get("version_registered", [])
                           if v.get("candidate_id") == cid), {})
        # The agent's OWN sentence about what this round was testing, as
        # journalled by `./version "why"`: a result should be attributable to
        # one hypothesis, and this is the hypothesis.
        noted = next((n for n in by.get("candidate_noted", [])
                      if n.get("candidate_id") == cid), {})
        rounds.append({
            "round_index": cap["round_index"],
            "candidate_id": cid,
            "hypothesis": str(noted.get("note") or ""),
            "tree_sha": cap["tree_sha"],
            "captured_utc": cap["captured_utc"],
            "capture_seconds": cap["capture_seconds"],
            "files": cap["files"],
            "accepted_head_at_capture": cap["accepted_head_at_capture"],
            "version_id": registered.get("version_id", ""),
        })

    heads = by.get("head_advanced", [])
    return {
        "kind": "serving_evolution",
        "run_id": identity.run_id,
        "task_id": identity.task_id,
        "lineage_id": identity.lineage_id,
        "control_epoch": identity.control_epoch,
        "workspace_id": identity.workspace_id,
        "namespace": identity.namespace,
        # What the run was asked for, read from `run_created`. The stop rule is
        # half the acceptance rule and the agent identity is a third of what
        # makes a lineage.
        "stop": (by.get("run_created") or [{}])[0].get("stop", ""),
        "agent": {
            "model": (by.get("run_created") or [{}])[0].get("agent_model", ""),
            "effort": (by.get("run_created") or [{}])[0].get("agent_effort", ""),
        },
        "final_run_state": run_state(journal).value,
        "agent_launches": [
            {"continued": e["continued"], "budget_seconds": e["budget_seconds"],
             "session_id": e["session_id"], "at_utc": e["at_utc"]}
            for e in by.get("agent_launched", [])],
        "rounds": rounds,
        "versions": [{"version_id": k, **{kk: vv for kk, vv in v.items()
                                          if kk not in ("event", "at_utc")}}
                     for k, v in versions.items()],
        "accepted_head": {
            "final": heads[-1]["version_id"] if heads else "",
            "advances": [{"from": h["predecessor"], "to": h["version_id"],
                          "authorized_by": h["authorized_by"],
                          "at_utc": h["at_utc"]} for h in heads],
            "conflicts": [{"expected": c["expected"], "found": c["found"]}
                          for c in by.get("head_cas_conflict", [])],
        },
        "candidate_states": {k: v.value for k, v in candidates(journal).items()},
        # Lineage memory is never delivered to an agent; what is recorded is
        # what the agent's own notes came back as, and any fault doing so.
        "memory_faults": [
            {"why": e.get("why", ""), "at_utc": e.get("at_utc", "")}
            for e in by.get("memory_persist_failed", [])],
        "memory_persisted": [
            {"lineage": e.get("lineage", ""), "bytes": e.get("bytes", 0),
             "over_bound": e.get("over_bound", False),
             "at_utc": e.get("at_utc", "")}
            for e in by.get("memory_persisted", [])],
        # That no lineage memory was delivered, and where it would have been.
        "memory_not_recalled": [
            {"why": e.get("why", ""), "lineage_id": e.get("lineage_id", ""),
             "looked_in": e.get("looked_in", ""),
             "at_utc": e.get("at_utc", "")}
            for e in by.get("memory_not_delivered", [])],
        "harness_faults": [
            {"event": e["event"], "why": e.get("why", "")}
            for e in events
            if e["event"] in ("boundary_failed", "after_capture_failed",
                              "harness_setup_failed")],
        "note": ("Assembled from the run's append-only journal. Every field "
                 "here is an event that was recorded when it happened; nothing "
                 "is inferred from a directory, a filename or a process."),
    }


#: How much of one tool result to keep in the readable rendering. The JSON
#: carries all of it; a person scrolling wants the shape of the run.
_EXCERPT = 1200


def as_markdown(doc: dict) -> str:
    """The same trajectory, for a person.

    `trajectory.atif.json` is the machine record, large and nested; nobody
    opens it to find out what a run did. This is the same content, in the
    order it happened.
    """
    agent = doc.get("agent") or {}
    final = doc.get("final_metrics") or {}
    steps = doc.get("steps") or []
    out = [f"# Trajectory — {doc.get('trajectory_id', '')}", "",
           f"- schema `{doc.get('schema_version', '')}`",
           f"- session `{doc.get('session_id', '')}`",
           f"- agent `{agent.get('name') or agent.get('model') or ''}`",
           f"- {len(steps)} steps", ""]
    if final:
        out += ["| final metric | value |", "|---|---:|"]
        out += [f"| {k} | {v} |" for k, v in sorted(final.items())]
        out.append("")
    if doc.get("notes"):
        out += ["> " + str(doc["notes"]).strip().replace("\n", "\n> "), ""]

    out += ["---", ""]
    for step in steps:
        who = step.get("source", "?")
        when = str(step.get("timestamp") or "")[11:19]
        out.append(f"## {step.get('step_id')} · {who}"
                   + (f" · {when}" if when else ""))
        message = str(step.get("message") or "").strip()
        if message:
            out += ["", message[:6000], ""]
        for call in step.get("tool_calls") or []:
            name = (call.get("function") or {}).get("name") or call.get("name") or "tool"
            args = call.get("arguments") or (call.get("function") or {}).get("arguments")
            shown = ((args.get("command") or args.get("file_path") or "")
                     if isinstance(args, dict) else str(args or ""))
            out += ["", f"**{name}**" + (f" — `{str(shown)[:200]}`" if shown else "")]
        for result in (step.get("observation") or {}).get("results") or []:
            body = str(result.get("content") or "").strip()
            if body:
                out += ["", "```", body[:_EXCERPT]
                        + ("\n… truncated; the JSON has all of it"
                           if len(body) > _EXCERPT else ""), "```"]
            for ref in result.get("subagent_trajectory_ref") or []:
                out += ["", f"→ subagent trajectory `{ref.get('trajectory_id')}`"
                        " (below)"]
        out.append("")
    for sub in doc.get("subagent_trajectories") or []:
        out += ["---", "", as_markdown(sub)]
    return "\n".join(out)


def collect(*, identity: RunIdentity, journal: Journal,
            agent_note: str = "") -> Path:
    """Write `trajectory.atif.json` for this run and return its path."""
    collector = _collector()
    run = identity.root
    events = collector.read_jsonl(run / "agent.stream.jsonl")
    steps, meta = collector.build_steps(
        events, collector.read_jsonl(run / "agent.changes.jsonl"))

    brief = run / "workspace" / "AGENT_TASK.md"
    if brief.is_file():
        steps.insert(0, {"step_id": 0, "source": "user",
                         "message": brief.read_text("utf-8", errors="ignore"),
                         "extra": {"role": "task_brief"}})
        for index, step in enumerate(steps):
            step["step_id"] = index + 1

    result_event = next(
        (e.get("event") or {} for e in reversed(events)
         if (e.get("event") or {}).get("type") == "result"), {})
    init = next((e.get("event") or {} for e in events
                 if (e.get("event") or {}).get("subtype") == "init"), {})
    launches = journal.events("agent_launched")
    session = launches[0]["session_id"] if launches else identity.run_id

    document = {
        "schema_version": collector.ATIF_VERSION,
        "session_id": init.get("session_id") or session,
        "trajectory_id": f"{identity.task_id}-{identity.run_id}",
        "agent": {
            "name": "claude-code",
            "version": str(init.get("claude_code_version") or "unknown"),
            "model_name": meta["model_name"],
            "extra": {"note": agent_note,
                      "continuations": sum(1 for e in launches
                                           if e.get("continued"))},
        },
        "steps": steps,
        "final_metrics": collector._final_metrics(
            result_event, meta["totals"], len(steps)),
        # ATIF v1.7's embedded form: each subagent's own trajectory, resolved
        # from the `subagent_trajectory_ref` on the `Agent` call that opened it.
        **({"subagent_trajectories": meta["subagent_trajectories"]}
           if meta.get("subagent_trajectories") else {}),
        "notes": ("Serving-system evolution run. The agent layer is ATIF v1.7. "
                  "`extra.evolution` carries what the run did with what the "
                  "agent built, read from the run's journal."),
        "extra": {"evolution": evolution(journal, identity)},
    }
    out = run / "trajectory.atif.json"
    write_trajectory(out, document)
    journal.append("trajectory_collected", path=str(out), steps=len(steps),
                   rounds=len(document["extra"]["evolution"]["rounds"]),
                   at=utc_now())
    return out
