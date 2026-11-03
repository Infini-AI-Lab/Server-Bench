"""What each published version was worth, WHEN it was published.

Each version is measured in parallel with its publication, from the coding
agent's own AgentX runs, so the run can be drawn as performance against evolve
time and API cost. The final validation never uses the coding agent's data.

So there are TWO streams of numbers about one run, and the whole point is that
they never mix:

| | the trajectory (this file) | the final verdict |
|---|---|---|
| measured by | the AGENT, inside Harbor | the harness, outside |
| instrument | the agent's own `aiperf` runs | `final_validate` (the final verdict) |
| inputs | agent-visible development inputs | the task's independent final run and
|        |                                  | its private samples, where declared |
| when | continuously, as the agent works | once, after the run ends |
| answers | "did this improve, and at what cost?" | "what is its final result under the task contract?" |
| authoritative for a verdict | **no** | **yes** |

## Why the agent's own numbers are the right ones for the curve

A trajectory needs a point per version, and a run publishes many. Re-measuring
each one properly costs a full replay on a free card, which would take longer
than the run itself.

The agent, meanwhile, is already measuring after nearly every edit, with the
SAME pinned AgentX build the verdict uses. Those numbers are noisier (short
`--benchmark-duration`, a card shared with whatever else the agent is doing) and
they are measured on prompts the agent can read. For "did this get faster over
the run" that is exactly good enough, and it is free.

## Why this can never feed back to the agent

Reading the agent's own artifacts is safe in the direction this goes: the
harness looks at what the agent produced. Nothing here is written back into the
workspace, and nothing here reaches `final_validate`. The moment a trajectory
number influenced an acceptance the hidden stage would become development with
extra steps -- which is why `harvest` is called from `_publish` and its output
lands in the version record and the run's `TRAJECTORY.json`, never in the
workspace.

`final_validate` does not import this module, and a test asserts that.

## What "in parallel with the version being published" means here

Not a second process. The agent's measurement has ALREADY happened by the time
it publishes -- that is why it published -- so the harvest is a read of
artifacts already on disk at the moment the version is registered. It costs no
GPU time and adds no wall clock to the run, which is what makes a point per
version affordable at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA = "serverbench.trajectory/1"

#: Where the brief tells the agent to write its aiperf artifacts.
#:
#: `--output-artifact-dir /workspace/agentx/run-N`. The agent is free to use
#: another path and some do, so the search falls back to any directory holding
#: an aiperf export rather than trusting the convention.
CONVENTION = "agentx"

#: The aiperf export files that mark a directory as one of its runs.
MARKERS = ("profile_export_aiperf.json", "profile_export.jsonl",
           "profile_export_raw.jsonl")

#: Top-level workspace directories the harness DELIVERS. An aiperf export
#: under one of these is somebody else's measurement, not the candidate's.
#:
#: `baselines/` carries this task's measured reference arms, and a reference
#: export is a complete, valid aiperf run -- so the scanner would prefer it
#: over an in-progress candidate run and report SGLang's numbers and replay
#: fidelity as the agent's own.
#:
#: A hidden-directory rule does not cover this: the delivered trees are
#: visible on purpose, because the agent is meant to read them.
DELIVERED = ("baselines",)


def _runs_under(workspace: Path) -> list[Path]:
    """Every aiperf artifact directory in the workspace, newest last.

    Sorted by MTIME rather than by name: `run-10` sorts before `run-2`
    lexically, and a trajectory built on that ordering would report the run's
    tenth measurement as its second.
    """
    workspace = Path(workspace)
    if not workspace.is_dir():
        return []
    found: dict[Path, float] = {}
    for marker in MARKERS:
        for path in workspace.rglob(marker):
            if _is_a_snapshot(path, workspace):
                continue
            if _is_delivered(path, workspace):
                continue
            # The public replay verifier uses AgentX and therefore writes the
            # same aggregate filename, but its fixed eight-request fixture is
            # explicitly not a performance measurement or curve point.
            try:
                relative_parts = path.relative_to(workspace).parts
            except ValueError:
                relative_parts = path.parts
            if "lightweight-agentx" in relative_parts:
                continue
            directory = path.parent
            try:
                when = path.stat().st_mtime
            except OSError:
                continue
            # Harvested live, the workspace only holds what existed by then,
            # so no time cutoff is needed.
            found[directory] = max(found.get(directory, 0.0), when)
    return [d for d, _ in sorted(found.items(), key=lambda kv: kv[1])]


def _is_delivered(path: Path, workspace: Path) -> bool:
    """Whether this export was DELIVERED to the workspace rather than measured.

    The workspace map divides itself into the files the agent owns and the
    trees the harness delivers; `baselines/` is delivered. An export there
    describes a reference arm, on reference hardware, at a point the candidate
    had nothing to do with.
    """
    try:
        parts = path.relative_to(workspace).parts
    except ValueError:
        return False
    return bool(parts) and parts[0] in DELIVERED


def _is_a_snapshot(path: Path, workspace: Path) -> bool:
    """Whether this export is the harness's COPY of a run rather than a run.

    Every candidate capture snapshots the tree into `.review/<candidate-id>/`,
    which duplicates whatever aiperf artifacts the agent had produced by then.
    Counting those would break the trajectory twice over:

    * **the count would inflate**, so `agentx_runs_found` would overstate how
      often the agent measured;
    * **the WRONG run would be chosen.** `_runs_under` orders by mtime and
      `harvest` takes the last. A snapshot's mtime is when the COPY was made,
      so a snapshot of an older benchmark would sort after a newer real one.

    A trajectory whose later points silently repeat an earlier measurement is
    worse than one with gaps: it draws a flat line through real improvement.

    Hidden directories are the rule rather than a `.review` literal -- the
    agent's own runs live in visible paths, and everything the harness stages
    beside them is hidden.
    """
    try:
        parts = path.relative_to(workspace).parts
    except ValueError:
        return False
    return any(part.startswith(".") for part in parts[:-1])


def _summary_of(run_dir: Path) -> dict[str, Any]:
    """The headline metrics aiperf itself computed for one run.

    Read out of `profile_export_aiperf.json`, which is aiperf's own summary --
    not recomputed here: one definition of each percentile, aiperf's.
    """
    body: dict[str, Any] = {}
    summary = Path(run_dir) / "profile_export_aiperf.json"
    if not summary.is_file():
        return body
    try:
        raw = json.loads(summary.read_text("utf-8"))
    except (OSError, ValueError):
        return body

    #: aiperf's own metric names, and what this repository calls them.
    wanted = {
        "time_to_first_token": "ttft_ms",
        "inter_token_latency": "tpot_ms",
        "request_latency": "e2e_ms",
        "output_token_throughput": "output_tokens_per_s",
        "request_throughput": "requests_per_s",
        "output_sequence_length": "output_tokens",
    }
    records = raw if isinstance(raw, dict) else {}
    for key, name in wanted.items():
        entry = records.get(key)
        if not isinstance(entry, dict):
            continue
        body[name] = {
            statistic: entry.get(statistic)
            for statistic in ("avg", "p50", "p90", "p95", "p99", "min", "max")
            if entry.get(statistic) is not None
        }
        if entry.get("unit"):
            body[name]["unit"] = entry["unit"]
    # THE CORRECTNESS FIELDS TRAVEL WITH THE METRICS, or the correctness check
    # reads a dict that cannot contain the answer.
    #
    # `wanted` above is a LATENCY map. Without `osl_mismatch_count` here,
    # `correctness_of` would default `mismatches` to 0 and every version would
    # come back `holds: true` for free. A correctness column that cannot say
    # no is worse than none, because it reads as evidence.
    #
    # `input_config` comes along for `scenario_of`: a plain load test's
    # `osl_mismatch_count: 0` is true and meaningless.
    # `total_isl`/`total_osl` are the OBJECTIVE's numerator: without them the
    # curve would compute from output throughput alone while the verdict uses
    # total tokens. The cache-hit sufficient statistics are pricing evidence
    # too: dropping them prices every input token as an uncached token even
    # when the trace says nearly all prefixes were reused. See
    # `final_validate/objective.py`.
    for key in ("request_count", "error_request_count", "benchmark_duration",
                "osl_mismatch_count", "osl_mismatch_diff_pct",
                "error_summary", "input_config", "scenario",
                "total_isl", "total_osl", "theoretical_prefix_cache_hit"):
        if records.get(key) is not None:
            body[key] = records[key]
    return body


#: The correctness fields aiperf writes, and what `final_validate`'s
#: `replay_fidelity` reads from them. Named here so the lightweight in-evolve
#: check and the authoritative verdict ask the SAME question of the same
#: numbers -- a development check that measures something else is worse than
#: none, because it passes for the wrong reason.
CORRECTNESS_FIELDS = ("osl_mismatch_count", "error_request_count",
                      "request_count", "osl_mismatch_diff_pct")


def _objective():
    """`final_validate.objective`, imported lazily.

    The objective belongs to the verdict; the trajectory REPORTS it. The
    integrity test forbids traffic the other way -- `final_validate` importing
    `lifecycle` -- and nothing here reverses that.
    """
    from .utils import ensure_harness_on_path
    ensure_harness_on_path()
    from final_validate import objective

    return objective


def scenario_of(summary: dict[str, Any]) -> str:
    """The AgentX scenario this export replayed, or "" for a plain load test.

    `aiperf` will happily benchmark anything. An agent can point it at its own
    server with synthetic prompts and get a complete-looking export --
    percentiles, throughput, `osl_mismatch_count: 0` -- that says nothing about
    the task, because the task's correctness rule is about replaying THIS
    corpus: trace-derived output lengths, `ignore_eos`, the prefix reuse, the
    quarter-million-token turns. A microbenchmark's `osl_mismatch_count: 0`
    is trivially true of a short prompt, and the tokens/$ that comes with it
    is not a number about this task at all.
    """
    config = summary.get("input_config")
    if isinstance(config, dict):
        named = config.get("scenario")
        if isinstance(named, str) and named:
            return named
        if isinstance(named, dict) and named.get("name"):
            return str(named["name"])
    named = summary.get("scenario")
    return named if isinstance(named, str) and named else ""


#: Evidence older than this describes code that may no longer be the version's.
#: Ten minutes is generous for an agent editing a kernel every few minutes.
CORRECTNESS_STALE_S = 600.0

#: A sample this small cannot see an intermittent fault: a truncated-stream
#: defect can appear in a quarter of the requests of one run and none of the
#: next, and a short microbenchmark would call it clean.
CORRECTNESS_THIN_REQUESTS = 100
CORRECTNESS_THIN_SECONDS = 300.0


def correctness_of(summary: dict[str, Any],
                   *, age_seconds: float | None = None) -> dict[str, Any]:
    """Did the agent's own last benchmark serve the work it was offered?

    A LIGHTWEIGHT CHECK, AND DELIBERATELY NOT A NEW MEASUREMENT.
    ------------------------------------------------------------
    The boundary checks are structural -- does the entry point exist, does the
    Python parse, does it reach for held-out evidence, does it hardcode a
    development path -- and the brief TELLS the agent to check its own output
    with the packaged verdict. This records whether its own export says the
    work was served.

    This costs nothing: the agent runs aiperf itself, and its export already
    carries the correctness numbers beside the latency ones. So the harness
    READS what was measured rather than measuring again -- no GPU time, no
    second measurement path, and no risk of disagreeing with the verdict,
    because it reads the same fields the verdict's rule reads.

    `holds` is None, never False, when there is nothing to judge. An absent
    measurement is not a failed one, and a run that published without
    benchmarking should say so rather than look broken.
    """
    if not summary:
        return {"holds": None,
                "why": "no aiperf export, so the agent's own correctness is "
                       "unknown for this version -- not failed, unknown"}

    # THE WRONG WORKLOAD IS NOT WEAK EVIDENCE, IT IS NONE.
    #
    # `replay_fidelity` asks whether the offered work was done: the trace's
    # output lengths, every turn served, the input not truncated. A plain
    # aiperf load test offers none of that, so its `osl_mismatch_count: 0` is
    # a statement about 62-token synthetic prompts and not about this task.
    # Reported as UNKNOWN, never as a pass.
    scenario = scenario_of(summary)
    if not scenario:
        return {
            "holds": None,
            "rule": "replay_fidelity",
            "scenario": "",
            "why": ("this export is a plain aiperf load test, not the task's "
                    "scenario, so correctness is UNKNOWN for this version. "
                    "`replay_fidelity` is about replaying the frozen corpus -- "
                    "trace-derived output lengths, ignore_eos, prefix reuse, "
                    "quarter-million-token turns -- and none of that was "
                    "offered. Use `./tests/replay-verify` as a cheap public "
                    "work-shape preflight, then run the actual scenario with "
                    "`final_verdict.scenario ... --measure-only --duration "
                    "1800` for evidence about the frozen workload"),
            "authoritative": False,
        }

    def value(name: str):
        entry = summary.get(name)
        if isinstance(entry, dict):
            return entry.get("avg")
        return entry

    mismatches = value("osl_mismatch_count") or 0
    errors = value("error_request_count") or 0
    requests = value("request_count") or 0
    summary_errors = summary.get("error_summary") or []
    holds = not mismatches and not errors and not summary_errors
    # HOW MUCH THIS "YES" IS WORTH, carried with it.
    #
    # `holds: true` from a short microbenchmark taken an hour before the
    # version was captured is a much weaker claim than the same word from a
    # full replay taken seconds before. A reader who cannot tell them apart
    # will over-trust the first, and the failure it would miss is precisely
    # the intermittent kind.
    duration = summary.get("benchmark_duration")
    seconds = (duration.get("avg") if isinstance(duration, dict) else duration) or 0
    caveats = []
    if age_seconds is not None and age_seconds > CORRECTNESS_STALE_S:
        caveats.append(
            f"the benchmark behind it finished {age_seconds / 60:.0f} min "
            f"before this version was captured, so it may describe earlier code")
    if requests and requests < CORRECTNESS_THIN_REQUESTS:
        caveats.append(f"only {requests:.0f} requests")
    if seconds and seconds < CORRECTNESS_THIN_SECONDS:
        caveats.append(f"a {seconds:.0f}s window")
    return {
        "holds": bool(holds),
        "rule": "replay_fidelity (read from the agent's own aiperf export)",
        "scenario": scenario,
        "sample_requests": requests,
        "sample_seconds": round(seconds, 1),
        "evidence_age_seconds": (round(age_seconds, 1)
                                 if age_seconds is not None else None),
        "well_evidenced": bool(holds) and not caveats,
        "caveat": ("; ".join(caveats) + ". A short or stale sample cannot see "
                   "an intermittent fault" if caveats else ""),
        "osl_mismatch_count": mismatches,
        "error_request_count": errors,
        "request_count": requests,
        "error_kinds": [e.get("error_details", {}).get("type")
                        for e in summary_errors[:3] if isinstance(e, dict)],
        "why": ("" if holds else
                f"{mismatches:.0f} of {requests:.0f} request(s) returned an "
                f"output length the trace did not ask for and {errors:.0f} "
                f"failed. The verdict checks this FIRST and a failure there "
                f"makes dollars_per_dollar unscorable, however high it is"),
        "authoritative": False,
        "authoritative_note": (
            "the agent's own development measurement, on inputs it may read. "
            "`final_validate` decides from an independent run outside the "
            "workspace, using private samples only where the task declares "
            "them"),
    }


def harvest(*, workspace: Path, version_id: str, captured_unix: float,
            started_unix: float, api_cost: dict[str, Any] | None = None,
            round_index: int | None = None,
            usd_per_hour: float = 0.0,
            objective_detail: dict[str, Any] | None = None) -> dict[str, Any]:
    """One trajectory point, from the agent's own most recent measurement.

    `captured_unix` is when this version's tree was frozen and `started_unix`
    when the run began, so `evolve_seconds` is the x axis of the figure.

    A version with no measurement behind it is recorded WITHOUT one rather than
    skipped: an agent that published without measuring is a real and
    interesting thing for a trajectory to show, and dropping the point would
    silently redraw the curve through the ones that did.
    """
    runs = _runs_under(Path(workspace))
    latest = runs[-1] if runs else None
    point: dict[str, Any] = {
        "version_id": version_id,
        "round_index": round_index,
        "captured_unix": captured_unix,
        "evolve_seconds": (round(captured_unix - started_unix, 3)
                           if captured_unix and started_unix else None),
        "measured_by": "the coding agent, inside Harbor, with the pinned aiperf",
        "authoritative": False,
        "authoritative_note": (
            "NOT a verdict and not comparable with a baseline. These are the "
            "agent's own development measurements: a short benchmark duration, "
            "a card it is also serving on, and prompts drawn from the half of "
            "the benchmark it is allowed to read. They answer 'did this get "
            "faster over the run', which is what the trajectory figure "
            "asks. The verdict is `final_validate`, run outside, on the "
            "independent final run and any task-declared private sample"),
        "agentx_runs_found": len(runs),
    }
    # THE NEWEST RUN THAT ACTUALLY FINISHED, not simply the newest.
    #
    # An aiperf directory appears as soon as the benchmark STARTS -- it holds
    # `profile_export.jsonl` while requests are still in flight -- and its
    # summary, `profile_export_aiperf.json`, is written at the end. Taking the
    # newest directory therefore picks an IN-PROGRESS run whenever a version is
    # published while one is going, and `_summary_of` returns `{}` for it.
    # The version would then look like one the agent never measured, which is
    # a different and more damning thing than one measured a minute earlier.
    chosen, summary = None, {}
    for candidate in reversed(runs):
        got = _summary_of(candidate)
        if got:
            chosen, summary = candidate, got
            break
    if chosen is not None:
        point["source"] = str(chosen)
        point["metrics"] = summary
        if chosen is not latest:
            point["note"] = (
                f"the newest export in the workspace ({Path(latest).name}) had "
                f"no `profile_export_aiperf.json` and was still running when "
                f"this version was published; this point reports the most "
                f"recent COMPLETED benchmark instead")
    elif latest is not None:
        point["source"] = str(latest)
        point["metrics"] = {}
        point["no_measurement_because"] = (
            f"{len(runs)} aiperf director(ies) are in the workspace and none "
            f"carries a `profile_export_aiperf.json`, so no benchmark had "
            f"finished when this version was published. Recorded rather than "
            f"dropped: a version published mid-benchmark is something the "
            f"curve should show")
    else:
        point["source"] = ""
        point["metrics"] = {}
        point["no_measurement_because"] = (
            "no aiperf export was found in the workspace at publication time. "
            "The version is still recorded: an agent that published without "
            "measuring is something the curve should show, not something to "
            "drop so the remaining points look tidier")
    # CORRECTNESS, ON EVERY POINT, from the same numbers the verdict reads.
    #
    # It costs nothing to read: the agent's own export carries the
    # correctness fields beside the latency ones. See `correctness_of`.
    # The AGE of the evidence matters as much as its verdict: an agent
    # publishes more often than it benchmarks, so a point can carry a
    # measurement taken an hour earlier against different code.
    age = None
    if chosen is not None:
        export = Path(chosen) / "profile_export_aiperf.json"
        try:
            age = max(0.0, captured_unix - export.stat().st_mtime)
        except OSError:
            age = None
    point["correctness"] = correctness_of(point.get("metrics") or {},
                                          age_seconds=age)

    # THE Y AXIS, computed here rather than by whoever draws the figure.
    #
    # `usd_per_hour` is the TASK's frozen `objective_detail.usd_per_hour_total`
    # -- the same denominator the verdict uses -- so an agent-side point and a
    # verdict point are at least on the same scale. They are still not the same
    # measurement and the point says so: `authoritative` is False either way.
    #
    # Note the duration cancels: tokens/(hours * rate) is
    # (tok/s * 3600) / rate, so a 30 s benchmark and a 1800 s one both give a
    # RATE per dollar. That is what makes the agent's short runs usable for a
    # curve at all -- what they cannot tell you is the absolute number under a
    # workload that has had time to start evicting.
    # THE SAME FORMULA THE VERDICT USES, from the same module, so the curve
    # and the verdict cannot disagree about the objective's definition.
    if usd_per_hour:
        scored = _objective().score(point.get("metrics") or {},
                                    usd_per_hour=float(usd_per_hour),
                                    prices=objective_detail)
        # EVERY FIELD THE OBJECTIVE PRODUCES, so the curve plots what the
        # agent is judged on.
        for field in ("gross_dollars_per_dollar", "gross_dollars_per_dollar_basis",
                      "value_usd", "cached_input_tokens",
                      "uncached_input_tokens",
                      "token_prices_usd_per_million", "token_price_source",
                      "tokens_per_usd",
                      "total_tokens", "input_tokens",
                      "output_tokens", "hours", "usd"):
            if scored.get(field) is not None:
                point[field] = scored[field]
        if scored.get("basis"):
            point["tokens_per_usd_basis"] = (
                scored["basis"] + ". The agent's own benchmark, NOT the "
                "verdict's")
        if scored.get("totals_note"):
            point["tokens_per_usd_note"] = scored["totals_note"]
    if isinstance(api_cost, dict):
        # TWO SHAPES, because two callers produce them.
        #
        # `api_cost.attribute()` returns every version under a `versions` key;
        # `run.py::_api_cost_for` picks ONE version out of that and returns it
        # FLAT, which is what actually reaches this function from `_publish`.
        #
        # Reading only the nested shape would silently produce
        # `api_cost_usd: None` on every point while the version record beside
        # it carried the cost, because both shapes are dicts and neither
        # raises.
        versions = api_cost.get("versions")
        mine = ((versions or {}).get(version_id) or {}) if versions else api_cost
        point["api_cost_usd"] = mine.get("cost_usd")
        point["api_usage"] = mine.get("usage")
        point["api_cost_model"] = api_cost.get("model") or mine.get("model")
        point["run_total_cost_usd"] = api_cost.get("run_total_cost_usd")
        point["api_cost_is_approximate"] = bool(
            api_cost.get("split_is_approximate"))
    return point


def mark_repeats(points: list[dict[str, Any]]) -> int:
    """Flag every point that carries the previous version's benchmark forward.

    Returns how many DISTINCT measurements the run actually produced, which is
    the number a reader needs before trusting the shape of the curve.

    Called from `append` on every write, so the marking always covers the
    whole curve.
    """
    previous_source = ""
    for entry in points:
        source = entry.get("source") or ""
        entry["repeats_previous_measurement"] = bool(
            source and source == previous_source)
        if entry["repeats_previous_measurement"]:
            entry["repeats_note"] = (
                "the agent did not run a new benchmark between the previous "
                "version and this one, so this point carries the earlier "
                "measurement forward. It is not evidence that nothing changed")
        if source:
            previous_source = source
    return len({p.get("source") for p in points if p.get("source")})


def append(run_root: Path, point: dict[str, Any]) -> Path:
    """Add one point to the run's `TRAJECTORY.json`, and return its path.

    Rewritten whole rather than appended to as JSONL, because the file is meant
    to be read by whatever draws the figure and a single JSON document with a
    schema and a note is easier to hand to a reader than a stream of lines.
    A run's point count is not a size where that costs anything.
    """
    path = Path(run_root) / "TRAJECTORY.json"
    body: dict[str, Any] = {"schema": SCHEMA, "points": []}
    if path.is_file():
        try:
            existing = json.loads(path.read_text("utf-8"))
            if isinstance(existing, dict) and existing.get("schema") == SCHEMA:
                body = existing
        except (OSError, ValueError):
            pass
    points = [p for p in body.get("points") or []
              if p.get("version_id") != point.get("version_id")]
    points.append(point)
    points.sort(key=lambda p: p.get("captured_unix") or 0.0)

    # A POINT THAT REPEATS THE PREVIOUS VERSION'S BENCHMARK SAYS SO.
    #
    # The agent measures when it chooses to, not once per version, so several
    # versions can carry one benchmark's numbers. Plotted naively that is a
    # flat line, which reads as "the agent stopped improving" when what
    # happened is "the agent stopped measuring".
    #
    # Not dropped, because a version that published without a fresh
    # measurement is a real fact about the run. Marked, so the figure can draw
    # it as a carried-forward value rather than as evidence.
    body["points"] = points
    body["distinct_measurements"] = mark_repeats(points)
    body["note"] = (
        "The evolve TRAJECTORY: one point per published version, measured by "
        "the AGENT with the pinned aiperf at the moment it published. x is "
        "`evolve_seconds`, y is whatever metric the task is scored on, and "
        "`api_cost_usd` is the third axis. None of it is a verdict -- see "
        "`authoritative_note` on each point -- and none of it is read by "
        "`final_validate`, which measures the published version independently "
        "in an independent run outside the agent workspace.")
    path.write_text(json.dumps(body, indent=1) + "\n", "utf-8")
    return path
