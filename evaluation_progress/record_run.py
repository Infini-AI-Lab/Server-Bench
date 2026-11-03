#!/usr/bin/env python3
"""Write one finished run to its result CSV, read from final evidence only.

    $SERVERBENCH_PY evaluation_progress/record_run.py --run <run-root>
    $SERVERBENCH_PY evaluation_progress/record_run.py --run <run-root> --output <csv>

ONE EVOLVE, ONE FINAL ROW. Re-running the recorder is idempotent and replaces
the file atomically. Lifecycle events and development measurements belong in
``journal.jsonl`` and the run workspace, never in this CSV.

NOTHING HERE MEASURES ANYTHING. Every field is read from evidence the verdict
already wrote, so this file cannot disagree with the verdict; if a number is
absent here it is absent there.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
LOCK_ROOT = REPO / "serverbench/artifacts/operations/evaluation-progress-locks"

#: The header. `concurrency` is recorded even though it is fixed per task --
#: it is the offered load every latency number below is only true at, and a
#: column that is constant today is the column you need the day it is not.
FIELDS = ("rank", "task", "system", "arm", "model", "agent", "effort",
          "concurrency", "requests", "correctness", "eos_gate", "replay",
          "ttft_p90_ms", "tpot_p90_ms",
          "output_tokens_per_s", "requests_per_s",
          "gross_dollars_per_dollar", "dollars_per_dollar",
          "gates_missed", "verdict",
          "evolve_hours", "started_utc", "measured_utc",
          "run_id", "version", "versions_published", "backend", "source",
          "evidence")

#: Where a row's numbers came from:
#:
#:   verdict     read from a FINAL.json + scenario-*.json this harness wrote
#:   run_summary a terminal run with no candidate/final measurement
SOURCES = ("verdict", "run_summary")
TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "STOPPED_EARLY"}
NON_RESULTS = {
    "WAITING_GPU", "LAUNCHING", "RUNNING", "RUN_ALLOCATED", "RUN_CREATED",
    "AGENT_STARTED", "VERSION_REGISTERED", "CHECKPOINT_PROMOTED",
    "HEAD_ADVANCED", "FINAL_VALIDATION_STARTED", "MEASUREMENT",
}


def _system_label(context: dict) -> str:
    """A readable name for the thing measured, from what the run recorded.

    `accounts/fireworks/models/minimax-m3` at effort `xhigh` becomes
    `minimax-m3 xhigh`. The full model id stays in its own column; this is the
    name a person scanning the file reads.
    """
    model = str(context.get("agent_model") or "").rsplit("/", 1)[-1]
    effort = str(context.get("agent_effort") or "")
    return " ".join(part for part in (model, effort) if part)


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return {}


def csv_for(task_id: str) -> Path:
    """One results file per task, named by the task id."""
    return HERE / f"{task_id}.csv"


def result_path(output: Path | None, task_id: str) -> Path:
    """One CSV per task in this directory, whatever name ``--output`` gives.

    Any output path inside this directory resolves to the task's own file, so
    the ledger cannot fork into several files for one task. Explicit outputs
    elsewhere (tests, exports) stay exact.
    """
    if output is None:
        return csv_for(task_id)
    output = Path(output)
    if output.parent.resolve() == HERE.resolve():
        return csv_for(task_id)
    return output


def _display_path(path: Path) -> Path:
    return path.relative_to(REPO) if path.is_relative_to(REPO) else path


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


#: The one outcome that scores. Every other verdict -- a failed hard gate,
#: NOT_RUN, a terminal run with no measurement -- scores nothing and is never
#: ranked, whatever number its evidence carries.
SCORED_OUTCOME = "HOLDS"

#: Columns that hold the score, blank on every row that is not HOLDS. The
#: gross figure and the request multiplier stay as diagnostics.
SCORE_FIELDS = ("dollars_per_dollar",)


def _scores(row: dict) -> bool:
    return str(row.get("verdict") or "").upper() == SCORED_OUTCOME


def _unscored_unless_holds(row: dict) -> dict:
    """A verdict that failed a hard gate scores nothing (CLAUDE.md)."""
    if not _scores(row):
        for field in SCORE_FIELDS:
            row[field] = ""
        row["rank"] = ""
    return row


def _budget(row: dict) -> float:
    """The agent budget this row was produced under, or 0.0 if unrecorded."""
    try:
        return float(row.get("evolve_hours") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _rank(row: dict) -> tuple:
    """Longest agent budget first, then best scored dollars_per_dollar.

    The budget is part of the sort because how long an agent was allowed to
    work dominates what it builds. Rows are grouped by `evolve_hours` and
    numbered within a group, so a rank only ever compares runs given the same
    time. Every row stays, and `evolve_hours` is a column, so the cohort a rank
    belongs to is visible beside it.
    """
    if not _scores(row):
        return (1, 0.0, 0.0, row.get("system", ""))
    try:
        value = float(row.get("dollars_per_dollar") or "")
    except ValueError:
        return (1, 0.0, 0.0, row.get("system", ""))
    return (0, -_budget(row), -value, row.get("system", ""))


def write_rows(path: Path, rows: list[dict]) -> None:
    """Sorted by the objective, and numbered, on every write.

    `rank` is recomputed rather than stored, so it can never disagree with the
    column it ranks. Ties share a number: two systems at the same $/$ are not
    ahead of each other, and numbering them 5 and 6 would say they are.
    """
    rows = sorted((_unscored_unless_holds(dict(row)) for row in rows),
                  key=_rank)
    # Numbered WITHIN a budget cohort. Restarting at each new `evolve_hours`
    # is what keeps a rank from claiming a comparison across agent budgets;
    # `index` would carry the previous cohort's count into the next one.
    previous, place, cohort, first = None, 0, None, 1
    for index, row in enumerate(rows, start=1):
        value = row.get("dollars_per_dollar") or ""
        budget = _budget(row) if _scores(row) else None
        if budget != cohort:
            cohort, first, previous = budget, index, None
        if value != previous:
            place = index - first + 1
            previous = value
        row["rank"] = str(place) if value and _scores(row) else ""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_id = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
    LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = LOCK_ROOT / f"{lock_id}.lock"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            with temporary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS,
                                        extrasaction="ignore",
                                        lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def write_final_result(path: Path, row: dict) -> None:
    """Upsert exactly one final row for this evolve into its task CSV."""
    verdict = str(row.get("verdict") or "").upper()
    if verdict in NON_RESULTS:
        raise ValueError(f"{verdict} is lifecycle telemetry, not a final result")
    kept = [existing for existing in read_rows(path)
            if _key(existing) != _key(row)]
    write_rows(path, [*kept, row])


def _key(row: dict) -> tuple:
    """What makes a row the same row.

    REPEATS OF ONE CONFIGURATION ARE DIFFERENT ROWS. The same model at the
    same effort on the same task, run twice, is two measurements and the
    second does not replace the first -- so the run id (which carries its own
    timestamp) and, for hand-entered rows, `started_utc` are part of the key.
    Without them a re-run would silently overwrite the result it should be
    compared against.
    """
    # The version is not part of the key: a run recorded as REFUSED with no
    # version and later given one is the same evolve, and replaces its row.
    return (row.get("run_id") or "", row.get("arm") or "",
            row.get("system") or "",
            row.get("started_utc") or "")


def upsert(row: dict) -> Path:
    """Add the row, replacing only a row that is the same measurement."""
    path = csv_for(row["task"])
    write_final_result(path, row)
    return path


def gates_for(task_id: str) -> list[dict]:
    """The task's frozen service levels, read from the task itself."""
    try:
        spec = yaml.safe_load(
            (REPO / "serving_tasks" / task_id / "contract.yaml")
            .read_text("utf-8")) or {}
        return list(spec.get("service_levels") or [])
    except Exception:                                          # noqa: BLE001
        return []


def gates_missed(task_id: str, row: dict,
                 levels: list[dict] | None = None) -> str:
    """Which frozen gates these numbers miss. DERIVED, never typed.

    Computed from the task's own `slo.service_levels` against the row's
    measured numbers, so it cannot drift from the contract and cannot be
    entered wrongly by hand. A number that is absent is not a miss -- it is
    unknown, and saying "PASS" about an unmeasured gate would make a task look
    easier than it is. `levels`, when the verdict recorded them, are the bars
    it actually judged against, and win over the task's current contract.
    """
    missed = []
    for level in (levels if levels else gates_for(task_id)):
        percentile = level.get("percentile")
        for metric, ceiling in (("ttft", level.get("ttft_ms")),
                                ("tpot", level.get("tpot_ms"))):
            if ceiling is None:
                continue
            raw = row.get(f"{metric}_p{percentile}_ms")
            try:
                measured = float(raw)
            except (TypeError, ValueError):
                continue
            if measured > float(ceiling):
                missed.append(f"p{percentile} {metric} "
                              f"{measured:g}>{float(ceiling):g}")
    return "; ".join(missed)


def _concurrency(row: dict, task_id: str) -> str:
    """The offered load these latency numbers are true at.

    Read from the aiperf argv the verdict recorded: the argv is what was
    actually offered. Empty when an arm never ran far enough to have a command.
    """
    command = row.get("command") or []
    for index, token in enumerate(command):
        if token == "--concurrency" and index + 1 < len(command):
            return str(command[index + 1])
    return ""


def _correctness(row: dict) -> tuple[str, str, str]:
    """(output correctness, EOS gate, replay fidelity), reported separately.

    Output correctness is the private stage: the held-out GSM8K answer
    accuracy against the plain checkpoint on the same questions, and the EOS
    gate. This file is the operator's record and is never delivered to an
    agent.
    """
    correctness = row.get("correctness") or {}
    answers = (correctness.get("output_correctness")
               or row.get("output_correctness") or {})
    parts = []
    if answers.get("candidate_correct") is not None:
        n = answers.get("questions") or 0
        got, ref = answers["candidate_correct"], answers.get("reference_correct")
        # The verdict's own decision is copied, never recomputed here.
        if answers.get("evaluable") is False or not ref:
            verdict = "unevaluable"
        else:
            verdict = "pass" if answers.get("holds") else "FAIL"
        parts.append(f"gsm8k {got}/{n} vs ref {ref}/{n} {verdict}")
    elif answers.get("evaluable") is False or answers.get("available") is False:
        parts.append("output check unevaluable")
    eos = answers.get("eos_gate")
    eos_gate = ("" if not isinstance(eos, dict)
                else "pass" if eos.get("holds") else "FAIL")
    replay = correctness.get("replay_fidelity") or {}
    if not replay:
        fidelity = ""
    elif replay.get("holds"):
        fidelity = "ok"
    else:
        failed = replay.get("error_request_count")
        fidelity = (f"FAILS ({int(failed)} errored)" if failed
                    else "FAILS")
    return "; ".join(parts), eos_gate, fidelity


def _requests(row: dict) -> str:
    """How many profiling requests the verdict's replay offered (errored and
    cancelled ones included -- they count against the score)."""
    attainment = (row.get("request_attainment")
                  or (row.get("decision") or {}).get("request_attainment") or {})
    if attainment.get("request_count") is not None:
        return str(int(attainment["request_count"]))
    aggregate = row.get("aggregate") or {}
    done, failed = aggregate.get("request_count"), aggregate.get("error_request_count")
    if done is None:
        return ""
    return str(int(float(done)) + int(float(failed or 0)))


def _stream_timing(artifact_dir: Path) -> dict:
    """`final_validate.stream_timing.assess`, the one definition, or {}."""
    sys.path.insert(0, str(REPO / "serverbench" / "harness"))
    try:
        from final_validate import stream_timing
    except ImportError:
        return {}
    return stream_timing.assess(artifact_dir)


def _number(value, digits: int = 1) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def _economics(decision: dict, final: dict, outcome: str = "") -> dict:
    """The scored objective, its gross diagnostic, and the multiplier between.

    Only a HOLDS outcome carries a score: for any other the scored columns
    are blank and the gross figure is reported as a diagnostic only.

    `dollars_per_dollar` is the scored objective and `gross_dollars_per_dollar`
    counts every request.
    """
    gross = decision.get("gross_dollars_per_dollar",
                         final.get("gross_dollars_per_dollar"))
    scored = decision.get("dollars_per_dollar",
                          final.get("dollars_per_dollar"))
    multiplier = decision.get("request_multiplier")
    if str(outcome or "").upper() != SCORED_OUTCOME:
        scored = None
    return {"gross_dollars_per_dollar": _number(gross, 2),
            "dollars_per_dollar": _number(scored, 2),
            "request_multiplier": _number(multiplier, 4)}


def row_from_evidence(scenario: Path, *, final: dict,
                      context: dict) -> dict | None:
    """One CSV row, entirely from what the verdict wrote."""
    body = _load(scenario)
    row = body.get("row") or {}
    verdict = body.get("verdict") or {}
    task_id = verdict.get("task_id") or context.get("task", "")
    arms = {a.get("arm"): a for a in (verdict.get("arms") or [])}
    arm_name = row.get("arm") or "candidate"
    decision = arms.get(arm_name) or {}
    outcome = decision.get("outcome") or final.get("status")
    aggregate = row.get("aggregate") or {}
    accuracy, eos_gate, replay = _correctness(row)
    row_out = {
        "task": task_id,
        "system": context.get("system") or _system_label(context),
        "arm": arm_name,
        "model": context.get("agent_model", ""),
        "agent": context.get("agent_cli", ""),
        "effort": context.get("agent_effort", ""),
        "concurrency": _concurrency(row, task_id),
        "requests": _requests({**row, "decision": decision}),
        "correctness": accuracy,
        "eos_gate": eos_gate,
        "replay": replay,
        "ttft_p90_ms": _number(aggregate.get("ttft_p90_ms"), 2),
        "tpot_p90_ms": _number(aggregate.get("tpot_p90_ms"), 2),
        "output_tokens_per_s": _number(aggregate.get("output_tokens_per_s"), 2),
        "requests_per_s": _number(aggregate.get("requests_per_s"), 3),
        **_economics(decision, final, outcome or ""),
        "verdict": outcome or "",
        "run_id": context.get("run_id", ""),
        "version": final.get("head_version_id", ""),
        "versions_published": context.get("versions_published", ""),
        "evolve_hours": context.get("hours", ""),
        "started_utc": context.get("started_utc", ""),
        "measured_utc": final.get("measured_utc", ""),
        "backend": context.get("backend", ""),
        "source": "verdict",
        "evidence": str(scenario.parent),
    }
    # WHEN THE OUTPUT ARRIVED, by the verdict's own rule over the verdict's
    # own per-request export. A server that holds every token until the
    # request ends shows a near-zero TPOT; printed beside a gate, that number
    # would read as a pass.
    timing = _stream_timing(scenario.parent / "agentx" / arm_name)
    if timing.get("buffered"):
        row_out["tpot_p90_ms"] = "not measured"
    # The verdict records one failed gate in several places; say it once.
    row_out["gates_missed"] = "; ".join(dict.fromkeys(filter(None, [
        gates_missed(task_id, row_out, verdict.get("service_levels")),
        ("p90 tpot not measured: output was not streamed "
         f"({timing['burst_delivered']} of {timing['eligible']} requests "
         f"burst-delivered)") if timing.get("buffered") else "",
        ((row.get("output_correctness") or {}).get("why")
         if (row.get("output_correctness") or {}).get("holds") is not True
         else ""),
        ((row.get("correctness") or {}).get("why")
         if (row.get("correctness") or {}).get("holds") is not True
         else ""),
        decision.get("why_misses") or decision.get("why") or "",
    ])))
    return row_out


def context_of(run_root: Path) -> dict:
    """What the RUN says about itself: which model, which effort, how long."""
    journal = run_root / "journal.jsonl"
    context = {"run_id": run_root.name}
    if journal.is_file():
        for line in journal.read_text("utf-8").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("event") == "run_created":
                context.update({
                    "task": event.get("task_id", ""),
                    "agent_model": event.get("agent_model", ""),
                    "agent_cli": event.get("agent_cli", ""),
                    "agent_effort": event.get("agent_effort", ""),
                    "hours": str(event.get("hours", "")),
                    "started_utc": event.get("at_utc", ""),
                })
                break
    container = _load(run_root / "container.json")
    name = str(container.get("container") or "")
    # Where the agent ran: a local container or a Modal sandbox. (Harbor is
    # the tool that composes the container, not a backend of its own.)
    context["backend"] = "modal" if name.startswith("modal:") else "container"
    published = 0
    if journal.is_file():
        for line in journal.read_text("utf-8").splitlines():
            if '"version_registered"' in line:
                published += 1
    context["versions_published"] = str(published)
    return context


def lineage_of(run_root: Path) -> Path | None:
    """The prototype lineage this run published into."""
    journal = run_root / "journal.jsonl"
    if not journal.is_file():
        return None
    for line in reversed(journal.read_text("utf-8").splitlines()):
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("event") == "version_registered" and event.get("directory"):
            return Path(event["directory"]).parent
    return None


def _terminal_row(run_root: Path, context: dict) -> dict:
    """Represent a run that ended before it produced a final measurement."""
    summary = _load(run_root / "SUMMARY.json")
    task = summary.get("task_id") or context.get("task", "")
    measured = ""
    terminal_outcome = ""
    terminal_reason = ""
    journal = run_root / "journal.jsonl"
    if journal.is_file():
        for line in reversed(journal.read_text("utf-8").splitlines()):
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("event") in {"final_validation", "final_validation_failed"}:
                measured = event.get("at_utc", "")
                terminal_outcome = str(
                    event.get("outcome") or event.get("status") or "")
                terminal_reason = str(event.get("why") or event.get("because") or "")
                if event.get("event") == "final_validation_failed":
                    terminal_reason = "HARNESS final validation: " + terminal_reason
                break
    state = str(summary.get("state") or "")
    row = {
        "task": task,
        "system": _system_label(context),
        "arm": "candidate",
        "model": context.get("agent_model", ""),
        "agent": context.get("agent_cli", ""),
        "effort": context.get("agent_effort", ""),
        "concurrency": _concurrency({}, task),
        # A FAILED/CANCELLED evolve stays a failed/cancelled evolve even when
        # its closing validator also records REFUSED because there was no
        # candidate. For a normally ended run, the final-validation outcome is
        # the useful terminal result (for example STOPPED_EARLY + REFUSED).
        "verdict": ((state if state in {"FAILED", "CANCELLED"} else "")
                    or terminal_outcome
                    or (state if state == "STOPPED_EARLY" else "NO_RESULT")),
        "evolve_hours": context.get("hours", ""),
        "started_utc": context.get("started_utc", ""),
        "measured_utc": measured,
        "gates_missed": terminal_reason,
        "run_id": context.get("run_id", run_root.name),
        "version": summary.get("accepted_head", ""),
        "versions_published": context.get("versions_published", ""),
        "backend": context.get("backend", ""),
        "source": "run_summary",
        "evidence": str(run_root / "SUMMARY.json"),
    }
    return row


def _verdict_running(run_root: Path) -> bool:
    """Final validation started and has not ended (per the run's journal)."""
    started = ended = False
    try:
        lines = (run_root / "journal.jsonl").read_text("utf-8").splitlines()
    except OSError:
        return False
    for line in lines:
        try:
            event = json.loads(line).get("event")
        except ValueError:
            continue
        if event == "final_validation_started":
            started, ended = True, False
        elif event in ("final_validation", "final_validation_failed",
                       "final_validation_skipped"):
            ended = True
    return started and not ended


def record(run_root: Path, *, output: Path | None = None) -> int:
    context = context_of(run_root)
    summary = _load(run_root / "SUMMARY.json")
    if summary.get("state") not in TERMINAL_STATES:
        print(f"{run_root.name}: refusing to write a result before the run "
              f"has a terminal SUMMARY.json", file=sys.stderr)
        return 1
    lineage = lineage_of(run_root)
    if lineage is None:
        row = _terminal_row(run_root, context)
        path = result_path(output, row["task"])
        write_final_result(path, row)
        print(f"recorded final {row['task']} {row.get('verdict')} -> "
              f"{_display_path(path)}")
        return 0
    final = _load(lineage / "FINAL.json")
    if not final and _verdict_running(run_root):
        # SUMMARY.json turns terminal when final validation STARTS; the row
        # waits for the verdict rather than recording a placeholder.
        print(f"{run_root.name}: the final verdict is still running; "
              f"record after the lineage's FINAL.json exists", file=sys.stderr)
        return 1
    if not final:
        row = _terminal_row(run_root, context)
        path = result_path(output, row["task"])
        write_final_result(path, row)
        print(f"recorded final {row['task']} {row.get('verdict')} -> "
              f"{_display_path(path)}")
        return 0
    evidence = lineage / "FINAL_VALIDATION" / str(
        final.get("head_version_dir") or "")
    scenario = evidence / "scenario-candidate.json"
    if not scenario.is_file():
        # The verdict ran and the arm produced no aggregate. That is still a
        # result and belongs in the file -- an arm nobody can find is how a
        # task comes to look easier than it is.
        row = {"task": final.get("task_id", context.get("task", "")),
               "arm": "candidate", "verdict": final.get("status", "NO_RESULT"),
               **_economics({}, final, final.get("status", "")),
               "version": final.get("head_version_id", ""),
               "measured_utc": final.get("measured_utc", ""),
               "evidence": str(evidence), "source": "verdict",
               "system": _system_label(context), **{
                   k: context.get(v, "") for k, v in
                   (("model", "agent_model"), ("agent", "agent_cli"),
                    ("effort", "agent_effort"), ("run_id", "run_id"),
                    ("evolve_hours", "hours"), ("started_utc", "started_utc"),
                    ("backend", "backend"),
                    ("versions_published", "versions_published"))}}
        row["concurrency"] = _concurrency({}, row["task"])
    else:
        row = row_from_evidence(scenario, final=final, context=context)
    path = result_path(output, row["task"])
    write_final_result(path, row)
    print(f"recorded {row['task']} {row.get('version')} "
          f"{row.get('verdict')} $/$ {row.get('dollars_per_dollar') or '--'} "
          f"-> {_display_path(path)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, action="append", default=[],
                    help="a finished run's root; repeatable")
    ap.add_argument("--all-runs", action="store_true",
                    help="every run under evolve_space")
    ap.add_argument("--output", type=Path,
                    help="explicit result CSV (valid with exactly one --run)")
    args = ap.parse_args(argv)

    runs = [r.resolve() for r in args.run]
    if args.output and (args.all_runs or len(runs) != 1):
        ap.error("--output requires exactly one --run")
    if args.all_runs:
        space = REPO / "serverbench" / "evolve_space" / "evolve"
        runs += sorted(p for p in space.glob("*/*")
                       if (p / "journal.jsonl").is_file())
    if not runs:
        print("give --run <root> or --all-runs", file=sys.stderr)
        return 2
    worst = 0
    for run_root in runs:
        worst = max(worst, record(run_root, output=args.output))
    return 0 if args.all_runs else worst


if __name__ == "__main__":
    raise SystemExit(main())
