"""What a calibration writes down, and what it says it is about.

A record that does not say what it is about will eventually be read as being
about something else. Every field here answers one of: which host, which task,
which arm, which model, how many cards, what gates, what sample.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

from final_validate.concurrency import REPO, SCHEMA, select
from final_validate.concurrency.arm import gates_for
from final_validate.scenario import ADMISSION_LIMIT


def save(plan, rows: list[dict], last_hold: int | None,
         first_miss: int | None) -> dict:
    """Write `calibration.json` and return it."""
    task_dir = REPO / "serving_tasks" / plan.task
    from final_validate import model_identity
    model_id, revision = model_identity.declared_for(task_dir)
    model = {"model_id": model_id, "revision": revision}

    seeds = {(r.get("measured") or {}).get("random_seed")
             for r in rows if r.get("measured")}
    seeds.discard(None)

    report = {
        "schema": SCHEMA,
        # WHAT THIS RECORD IS ABOUT. All of it, because the answer transfers
        # between none of them.
        "host": socket.gethostname(),
        "task": plan.task,
        "arm": plan.arm,
        # THE MODEL ACTUALLY SERVED, not the one the task declares. They differ
        # whenever a calibration asks "what would a DIFFERENT model sustain
        # under this task's gates on this hardware" -- which is the question
        # you ask before authoring a task for that model. Recording the task's
        # declared model here would label a Gemma measurement as Qwen.
        "model_dir": str(Path(plan.model_dir).resolve()),
        "model_served": Path(plan.model_dir).resolve().name,
        "task_declares_model_id": model.get("model_id"),
        "task_declares_revision": model.get("revision"),
        "model_matches_the_task": (
            str(model.get("revision") or "") in str(Path(plan.model_dir))),
        "devices": plan.gpu,
        "scored_window_seconds": plan.duration,
        "stop_rule": "timebox",
        # THE SAMPLE. Each point is its own draw unless --random-seed fixed
        # one; either way two offered loads do not draw the same work, because
        # `--concurrency N` opens N lanes over the stream. `sample_warning`
        # below says so where a reader ranking the points will see it.
        "random_seeds_observed": sorted(seeds),
        "seed_fixed": plan.random_seed is not None,
        "gates": [{k: lv[k] for k in lv} for lv in gates_for(task_dir)],
        "serve_extra_overridden": bool(plan.serve_extra),
        "server_admission_limit": ADMISSION_LIMIT,
        "max_concurrency_holding_all_gates": last_hold,
        "lowest_concurrency_that_missed": first_miss,
        "invalid_points": [r["concurrency"] for r in rows if r.get("invalid")],
        "answer_is_absent_because": (
            "" if last_hold is not None else
            "no point held every gate. If `invalid_points` is non-empty this "
            "is an infrastructure failure and NOT a statement about the "
            "server: nothing was measured at those loads. If it is EMPTY, the "
            "search reached the lowest load asked for and the gates were "
            "missed there too -- which is a measured finding about the bar, "
            "not a failed search: this arm does not hold this task's gates at "
            "any offered load on this hardware"),
        "sample_warning": (
            "Inside a fixed time box different offered loads do NOT complete "
            "the same work, and one seed does not change that: --concurrency N "
            "opens N lanes that consume the seeded stream in parallel, so "
            "adjacent points diverge immediately and can differ several-fold "
            "in request count and mean output length. Each point records its own "
            "`total_output_tokens` and `mean_output_tokens_per_request` -- "
            "read them before comparing throughput. Rank on what is stable "
            "across repeats (where throughput peaks, where eviction starts, "
            "whether the client ever reaches the offered number), never on a "
            "fine difference between adjacent points."),
        "how_to_read_this": (
            "`max_concurrency_holding_all_gates` is the highest OFFERED "
            "top-level trajectory count at which this arm held every frozen "
            "performance gate on this host. Replay fidelity is recorded "
            "separately for protocol diagnosis. "
            "`effective_concurrency_max` is peak HTTP request concurrency; "
            "AgentX branch fan-out may make it larger than the offered "
            "trajectory count. Record the exports under "
            "`serving_tasks/_concurrency_search_archive/<task>/` and adopt the "
            "measured setting with `taskctl set <task> --reference-concurrency N`."),
        # WHICH LOAD TO ADOPT, by the standard rules rather than by eye.
        # SLO-constrained max throughput where a point holds; the non-dominated
        # set and its knee where none does. See `select.py`.
        "selection": select.recommend(rows, gates=gates_for(task_dir)),
        "points": rows,
    }
    out = Path(plan.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(
        json.dumps(report, indent=1) + "\n", "utf-8")
    return report


def _wrap(text: str, width: int) -> list[str]:
    import textwrap
    return textwrap.wrap(text, width) if text else []


def summarise(report: dict) -> str:
    """The human line a search prints when it finishes."""
    lines = [
        "",
        "=== calibration ===",
        f"  task    {report['task']}  arm {report['arm']}  "
        f"host {report['host']}  devices {report['devices']}",
        f"  HOLDS every gate up to offered concurrency: "
        f"{report['max_concurrency_holding_all_gates']}",
        f"  first miss at: {report['lowest_concurrency_that_missed']}",
        f"  seeds: {report['random_seeds_observed']}"
        f"{'  (fixed by --random-seed)' if report['seed_fixed'] else ''}",
    ]
    pick = report.get("selection") or {}
    if pick.get("rule"):
        lines.append("")
        lines.append(f"  RECOMMENDED: offered concurrency {pick.get('recommended')}"
                     f"   [{pick['rule']}]")
        for chunk in _wrap(pick.get("why", ""), 72):
            lines.append(f"    {chunk}")
        lines.append(f"    pareto front {pick.get('pareto_front')}   "
                     f"knee {(pick.get('knee') or {}).get('concurrency')} "
                     f"(distance {(pick.get('knee') or {}).get('chord_distance')})")
        if pick.get("sample_moved"):
            lines.append("    ! the points did NOT do comparably sized work; "
                         "their throughput is not comparable")
    rows = [r for r in report.get("points") or [] if r.get("measured")]
    if rows:
        lines.append("")
        lines.append(f"  {'offered':>8}{'holds':>7}{'TTFT p90':>10}{'p95':>9}"
                     f"{'TPOT p90':>10}{'p95':>7}{'out tok/s':>11}"
                     f"{'requests':>10}{'mean OSL':>10}")
        for row in sorted(rows, key=lambda r: r.get("concurrency", 0)):
            m = row["measured"]

            def f(key, nd=0, m=m):
                v = m.get(key)
                return "-" if v is None else f"{v:,.{nd}f}"

            lines.append(
                f"  {row['concurrency']:>8}"
                f"{'yes' if row.get('holds') else 'no':>7}"
                f"{f('ttft_p90_ms'):>10}{f('ttft_p95_ms'):>9}"
                f"{f('tpot_p90_ms', 1):>10}{f('tpot_p95_ms', 1):>7}"
                f"{f('output_tokens_per_s', 1):>11}"
                f"{f('request_count'):>10}"
                f"{f('mean_output_tokens_per_request'):>10}")
        lines.append("")
        lines.append("  The last two columns are the SAMPLE, not the result. "
                     "If they move between")
        lines.append("  points, the throughput column is partly about the "
                     "draw -- see sample_warning.")
    return "\n".join(lines)
