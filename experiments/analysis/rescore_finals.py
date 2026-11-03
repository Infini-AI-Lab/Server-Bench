"""Every verdict's per-request export, re-scored with its task's rule (no GPU).

A verdict is a directory holding `scenario-candidate.json` and the candidate's
AgentX export (`agentx/candidate/profile_export.jsonl`). Two kinds are found:

    final       a lineage's own verdict, under server_prototypes/<lineage>/FINAL_VALIDATION/
    remeasured  a version measured again outside its run, e.g. by
                experiments/ablation_budget_retrospective.sh

Each row joins the verdict to the run that published the version (model, CLI,
effort, budget), reports the verdict's own decision and operating point, and
re-scores the export with the task's current request rule.

    $SERVERBENCH_PY experiments/analysis/rescore_finals.py [--out DIR]

Writes <out>/final_rescored.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import common
import rescore


def verdict_dirs(roots: list[Path]) -> list[Path]:
    found = set()
    for root in roots:
        for scenario in Path(root).rglob("scenario-candidate.json"):
            if (scenario.parent / "agentx" / "candidate" / "profile_export.jsonl").is_file():
                found.add(scenario.parent)
    return sorted(found)


def window_s(verdict: Path, records: list[dict]) -> float | None:
    """The replay's benchmark duration as AgentX reported it, else the profiling span."""
    agg = common.load_json(verdict / "agentx" / "candidate" / "profile_export_aiperf.json")
    duration = agg.get("benchmark_duration") or (agg.get("records") or {}).get("benchmark_duration")
    if isinstance(duration, dict) and duration.get("avg"):
        return float(duration["avg"])
    return rescore.span_s(records)


def decision(scenario: dict) -> dict:
    """The candidate arm's decision and which of its gates failed."""
    verdict = scenario.get("verdict") or {}
    arm = next((a for a in verdict.get("arms") or [] if a.get("arm") == "candidate"), {})
    correctness = arm.get("correctness") or {}
    failed = [k for k, v in correctness.items() if isinstance(v, dict) and v.get("holds") is False]
    answers = correctness.get("output_correctness") or {}
    undecided = ""
    if arm.get("outcome") in (None, "NOT_RUN"):
        undecided = "verdict not run"
    elif answers and not answers.get("evaluable", answers.get("available", True)):
        undecided = "output correctness unevaluable"
    return {"outcome": arm.get("outcome", ""),
            "eligible": arm.get("outcome") == "HOLDS",
            "failed_checks": "; ".join(f"{k}: {(correctness[k].get('why') or '')[:110]}" for k in failed),
            "undecided": undecided,
            "verdict_dpd": arm.get("dollars_per_dollar"),
            "verdict_gross_dpd": arm.get("gross_dollars_per_dollar")}


def identify(verdict: Path) -> tuple[str, str]:
    """(lineage, version directory name) the verdict judged."""
    measured = common.load_json(verdict / "MEASURED.json")
    if measured.get("version"):
        version = Path(measured["version"])
        return version.parent.name, version.name
    # A lineage's own copy: <lineage>/FINAL_VALIDATION/<version>/
    return verdict.parent.parent.name, verdict.name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--remeasured", type=Path, action="append", default=None,
                    help=f"root of re-measured verdicts; repeatable "
                         f"(default: {common.display(common.REMEASURED)})")
    common.add_paths(ap, "evolve", "prototypes", "tasks", "out")
    args = ap.parse_args()
    remeasured = args.remeasured or [common.REMEASURED]
    runs = common.runs_by_lineage(args.evolve)

    rows = []
    for verdict in verdict_dirs([args.prototypes, *remeasured]):
        body = common.load_json(verdict / "scenario-candidate.json")
        task = (body.get("verdict") or {}).get("task_id", "")
        if not task or not (args.tasks / task / "contract.yaml").is_file():
            print(f"skip {common.display(verdict)}: no live task {task!r}")
            continue
        export = verdict / "agentx" / "candidate" / "profile_export.jsonl"
        window = window_s(verdict, rescore.profiling(export))
        if not window:
            print(f"skip {common.display(verdict)}: no profiling window")
            continue
        lineage, version = identify(verdict)
        run = runs.get(lineage, {})
        scored = rescore.rescore(export, common.task_terms(task, args.tasks), window)
        row = body.get("row") or {}
        aggregate = row.get("aggregate") or {}
        is_final = verdict.parent.name == "FINAL_VALIDATION"
        rows.append({
            "task": task, "kind": "final" if is_final else "remeasured",
            "run_id": run.get("run_id", ""), "agent_cli": run.get("agent_cli", ""),
            "agent_model": run.get("agent_model", ""), "effort": run.get("effort", ""),
            "evolve_hours": run.get("evolve_hours", ""), "started_utc": run.get("started_utc", ""),
            "lineage": lineage, "version": version.rsplit("_", 1)[-1], "version_dir": version,
            **decision(body),
            "concurrency": row.get("offered_concurrency"),
            "ttft_p90_ms": aggregate.get("ttft_p90_ms"), "tpot_p90_ms": aggregate.get("tpot_p90_ms"),
            "output_tok_s": aggregate.get("output_tokens_per_s"),
            "window_s": round(window, 1), "requests": scored["requests"], "qualified": scored["qualified"],
            "gross_dpd": round(scored["gross_dpd"], 4), "dpd": round(scored["dpd"], 4),
            "export": str(export), "evidence": str(verdict),
        })
    if not rows:
        print("no verdicts found")
        return 1
    common.write_csv(args.out / "final_rescored.csv", rows)
    for r in rows:
        print(f"{r['task'][:26]:26} {r['kind']:10} {common.model_label(r)[:24]:24} {r['version']:4} "
              f"gross={r['gross_dpd']:6.2f} dpd={r['dpd']:6.2f} {r['outcome']:7} {r['failed_checks'][:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
