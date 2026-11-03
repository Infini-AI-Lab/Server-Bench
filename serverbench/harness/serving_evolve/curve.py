"""What a run actually did, per version, in one table.

This module is the one reader for a run's own evidence (`journal.jsonl`,
`TRAJECTORY.json`, `agent.stream.jsonl`), so the shape of that evidence is
encoded once. A throwaway script that reads the stream's `{"t": ..., "event":
{...}}` rows at the top level, or asks for `metrics["ttft_ms_p90"]` instead of
`metrics["ttft_ms"]["p90"]`, reports missing data that reads exactly like a
broken harness path; with one reader, a surprise in the table is a surprise
about the RUN.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any


def _events(run_dir: Path) -> list[dict[str, Any]]:
    body = run_dir / "journal.jsonl"
    if not body.is_file():
        return []
    out = []
    for line in body.read_text("utf-8", errors="ignore").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def _unix(stamp: str) -> float:
    try:
        return (datetime.datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")
                .replace(tzinfo=datetime.timezone.utc).timestamp())
    except (TypeError, ValueError):
        return 0.0


def _stat(metrics: dict[str, Any], name: str, which: str) -> float | None:
    """One percentile of one metric, from the NESTED shape aiperf produces.

    `metrics["ttft_ms"]["p90"]`, never `metrics["ttft_ms_p90"]`. Written down
    here because getting it wrong silently returns None for every version and
    looks precisely like a measurement path that never ran.
    """
    entry = (metrics or {}).get(name)
    if not isinstance(entry, dict):
        return None
    value = entry.get(which)
    return float(value) if isinstance(value, (int, float)) else None


def read(run_dir: Path) -> dict[str, Any]:
    """Everything a reader needs about one run, gathered from its own evidence."""
    run_dir = Path(run_dir)
    events = _events(run_dir)

    created = next((e for e in events if e.get("event") == "run_created"), {})
    final_state = ""
    for e in events:
        if e.get("event") == "run_state":
            final_state = e.get("to") or final_state

    # Boundary checks, per candidate, so a check that fails on EVERY version
    # (which is what a false positive looks like) is visible as such.
    check_outcomes: dict[str, int] = {}
    for e in events:
        if e.get("event") == "boundary_checks":
            for name in e.get("failed") or []:
                check_outcomes[name] = check_outcomes.get(name, 0) + 1

    verdicts = [e for e in events
                if e.get("event") in ("final_validation",
                                      "final_validation_failed")]

    trajectory: dict[str, Any] = {}
    body = run_dir / "TRAJECTORY.json"
    if body.is_file():
        try:
            trajectory = json.loads(body.read_text("utf-8"))
        except (OSError, ValueError):
            trajectory = {}

    points = []
    seen_signature: dict[tuple, str] = {}
    for point in trajectory.get("points") or []:
        metrics = point.get("metrics") or {}
        # A REPEAT IS THE DEFAULT, NOT THE EXCEPTION.
        #
        # An agent publishes more often than it benchmarks, so most versions
        # carry the previous version's numbers. Undeclared, that draws a flat
        # line through real improvement and reads as a stalled run.
        signature = (_stat(metrics, "ttft_ms", "p90"),
                     _stat(metrics, "output_tokens_per_s", "avg"),
                     point.get("tokens_per_usd"))
        repeat_of = None
        if any(v is not None for v in signature):
            repeat_of = seen_signature.get(signature)
            if repeat_of is None:
                seen_signature[signature] = point.get("version_id")
        correctness = point.get("correctness") or {}
        points.append({
            "version_id": point.get("version_id"),
            # The agent's OWN correctness, read from its aiperf export. Not a
            # verdict -- `final_validate` decides -- but the difference
            # between a fast server and a server that did less work, visible
            # while the run is still going.
            "correct": correctness.get("holds"),
            "correct_why": correctness.get("why", ""),
            "evolve_hours": round((point.get("evolve_seconds") or 0) / 3600, 2),
            "tokens_per_usd": point.get("tokens_per_usd"),
            "ttft_ms_p90": _stat(metrics, "ttft_ms", "p90"),
            "ttft_ms_p95": _stat(metrics, "ttft_ms", "p95"),
            "output_tokens_per_s": _stat(metrics, "output_tokens_per_s", "avg"),
            "api_cost_usd": point.get("api_cost_usd"),
            "measured": bool(metrics),
            "repeat_of": repeat_of,
            "no_measurement_because": point.get("no_measurement_because"),
        })

    return {
        "run_dir": str(run_dir),
        "run_id": created.get("run_id", run_dir.name),
        "task_id": created.get("task_id", ""),
        "agent_model": created.get("agent_model", ""),
        "agent_effort": created.get("agent_effort", ""),
        "gpus": created.get("gpus") or [],
        "hours": created.get("hours"),
        "deadline_utc": created.get("deadline_utc", ""),
        "final_state": final_state,
        "versions": len(points),
        "measured_versions": sum(1 for p in points if p["measured"]),
        "points": points,
        "boundary_failures": check_outcomes,
        "verdict_events": [
            {"event": e.get("event"),
             "text": (e.get("because") or e.get("why") or "")[:4000]}
            for e in verdicts],
        "api_cost": _cost(run_dir, events),
    }


def _cost(run_dir: Path, events: list[dict[str, Any]]) -> dict[str, Any]:
    """The run's API cost, recomputed from the stream by the one reader.

    Recomputed rather than read off the journal so that a fix to the
    attribution shows up here without re-running anything.
    """
    stream = run_dir / "agent.stream.jsonl"
    if not stream.is_file():
        return {"available": False,
                "why": f"{stream} does not exist"}
    captures = [(e["version_id"], _unix(e.get("at_utc", "")))
                for e in events
                if e.get("event") == "head_advanced" and e.get("version_id")]
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from lifecycle import api_cost
        out = api_cost.attribute(stream, captures)
    except Exception as exc:                                   # noqa: BLE001
        return {"available": False, "why": repr(exc)}
    return {
        "available": bool(out.get("available")),
        "model": out.get("model"),
        "run_total_cost_usd": out.get("run_total_cost_usd"),
        "run_total_from": out.get("run_total_from"),
        "stalled_share": out.get("stalled_share"),
        "cost_unavailable_because": out.get("cost_unavailable_because"),
        "versions": {k: v.get("cost_usd")
                     for k, v in (out.get("versions") or {}).items()},
    }


def lines(report: dict[str, Any]) -> list[str]:
    """The human rendering. One row per version, and nothing implied."""
    out = [
        f"run       {report['run_id']}  ({report['task_id']})",
        f"agent     {report['agent_model']} {report['agent_effort']}"
        f"   gpus {report['gpus']}   budget {report['hours']} h",
        f"state     {report['final_state'] or 'in progress'}"
        f"   {report['measured_versions']}/{report['versions']} version(s) "
        f"carry a measurement",
    ]
    cost = report.get("api_cost") or {}
    if cost.get("available"):
        total = cost.get("run_total_cost_usd")
        out.append("api cost  "
                   + (f"${total:.4f}" if isinstance(total, (int, float))
                      else "UNAVAILABLE")
                   + f"   ({str(cost.get('run_total_from'))[:90]})")
        if cost.get("stalled_share"):
            out.append(f"          token counters stalled with "
                       f"{cost['stalled_share']:.0%} of the run still to come")
        if cost.get("cost_unavailable_because"):
            out.append(f"          {cost['cost_unavailable_because'][:150]}")
    out.append("")
    out.append(f"  {'ver':>4} {'hours':>6} {'ok?':>4} {'tokens/$':>11} "
               f"{'ttft p90':>10} {'ttft p95':>10} {'tok/s':>8} {'cost':>9}  note")
    for p in report["points"]:
        def fmt(value: float | None, spec: str) -> str:
            return format(value, spec) if isinstance(value, (int, float)) else "-"
        note = ""
        if p["repeat_of"]:
            note = f"same measurement as {p['repeat_of']}"
        elif not p["measured"]:
            note = (p.get("no_measurement_because") or "never benchmarked")[:60]
        ok = {True: "yes", False: "NO", None: "-"}[p.get("correct")]
        if p.get("correct") is False and not note:
            note = (p.get("correct_why") or "correctness failed")[:64]
        out.append(
            f"  {str(p['version_id']):>4} {p['evolve_hours']:>6.2f} "
            f"{ok:>4} "
            f"{fmt(p['tokens_per_usd'], '>11,.0f')} "
            f"{fmt(p['ttft_ms_p90'], '>10,.0f')} "
            f"{fmt(p['ttft_ms_p95'], '>10,.0f')} "
            f"{fmt(p['output_tokens_per_s'], '>8,.1f')} "
            f"{fmt(p['api_cost_usd'], '>9,.2f')}  {note}")
    if report["boundary_failures"]:
        out.append("")
        out.append("  boundary checks that FAILED, and on how many candidates:")
        for name, count in sorted(report["boundary_failures"].items()):
            flag = ("  <- every candidate; suspect the check, not the agent"
                    if count >= max(1, report["versions"]) else "")
            out.append(f"    {name:<22} {count}{flag}")
    for verdict in report["verdict_events"]:
        out.append("")
        out.append(f"  {verdict['event']}:")
        for line in verdict["text"].splitlines()[:14]:
            out.append(f"    {line}")
    return out
