"""Development trajectories, scored on one rule (no GPU).

Every AgentX export an agent left in its workspace is re-scored with its task's
request rule over its own profiling span. Fidelity uses AgentX's output-length
check (|actual - requested| > min(5%, 50 tokens)) plus request errors.
Cumulative Claude Code API cost is anchored at the `total_cost_usd` the CLI
reports at each session end and shaped within a session by token usage; runs
whose CLI reports no cost carry none.

    $SERVERBENCH_PY experiments/analysis/dev_trajectories.py [--run <run root> ...] [--task <task>]

Default: every run under --evolve. Writes <out>/dev_trajectories.json.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import common
import rescore

#: Relative token weights used only to split a session's reported cost over its turns.
USAGE_WEIGHTS = {"input_tokens": 1.0, "cache_creation_input_tokens": 1.25,
                 "cache_read_input_tokens": 0.1, "output_tokens": 5.0}
MIN_WINDOW_S = 60


def cost_curve(root: Path) -> list[tuple[float, float]]:
    """[(unix time, cumulative USD)] anchored at the CLI's reported session totals."""
    points, segment, anchor = [], [], 0.0
    for rec in common.read_jsonl(root / "agent.stream.jsonl"):
        t, event = rec.get("t"), rec.get("event") or {}
        if event.get("type") == "assistant":
            usage = (event.get("message") or {}).get("usage") or {}
            segment.append((t, sum(w * (usage.get(k) or 0) for k, w in USAGE_WEIGHTS.items())))
        elif event.get("type") == "result" and event.get("total_cost_usd") is not None:
            total = float(event["total_cost_usd"])
            delta, weight_sum, acc = max(total - anchor, 0.0), sum(w for _, w in segment) or 1.0, 0.0
            for at, w in segment:
                acc += w
                points.append((at, anchor + delta * acc / weight_sum))
            points.append((t, total))
            anchor, segment = total, []
    return points


def cost_at(curve: list[tuple[float, float]], t: float) -> float:
    cost = 0.0
    for at, value in curve:
        if at > t:
            break
        cost = value
    return cost


def fidelity(records: list[dict]) -> tuple[int, int]:
    """(requests with the wrong output length, requests that errored)."""
    mismatched = errored = 0
    for r in records:
        if r.get("error"):
            errored += 1
            continue
        y = rescore.metric(r, "output_sequence_length")
        pct = rescore.metric(r, "osl_mismatch_diff_pct")
        if y is not None and pct is not None and pct > -100:
            requested = y / (1 + pct / 100.0)
            if abs(y - requested) > min(requested * 0.05, 50):
                mismatched += 1
    return mismatched, errored


def run(root: Path, tasks: Path = common.TASKS) -> dict:
    info = common.run_info(root)
    t0 = info["t0"]
    curve = cost_curve(root)
    terms = common.task_terms(info["task"], tasks)
    points = []
    for export in sorted(root.glob("workspace/**/profile_export.jsonl")):
        records = rescore.profiling(export)
        window = rescore.span_s(records)
        if not window or window < MIN_WINDOW_S:
            continue
        mtime = os.path.getmtime(export)
        scored = rescore.rescore(export, terms, window)
        mismatched, errored = fidelity(records)
        points.append({"path": str(export.relative_to(root / "workspace")),
                       "hours": round((mtime - t0) / 3600, 3),
                       "api_cost_usd": round(cost_at(curve, mtime), 2), "window_s": round(window),
                       "requests": len(records), "osl_mismatch": mismatched, "errors": errored,
                       "fidelity_ok": mismatched == 0 and errored == 0,
                       "dpd": round(scored["dpd"], 4), "gross_dpd": round(scored["gross_dpd"], 4)})
    points.sort(key=lambda p: p["hours"])
    versions = [{"version": v["version"], "hours": v["hours"],
                 "api_cost_usd": round(cost_at(curve, v["at_unix"]), 2)} for v in info["versions"]]
    sampled = curve[::25] + ([curve[-1]] if curve else [])
    return {"run_id": info["run_id"], "task": info["task"], "agent_model": info["agent_model"],
            "agent_cli": info["agent_cli"], "effort": info["effort"],
            "evolve_hours": info["evolve_hours"], "points": points, "versions": versions,
            "api_cost_total_usd": round(curve[-1][1], 2) if curve else None,
            "cost_curve": [(round((t - t0) / 3600, 3), round(v, 2)) for t, v in sampled]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, action="append", default=[], help="a run root; repeatable")
    ap.add_argument("--task", action="append", default=[], help="only runs of this task; repeatable")
    common.add_paths(ap, "evolve", "tasks", "out")
    args = ap.parse_args()
    roots = args.run or common.run_roots(args.evolve)
    out = {}
    for root in roots:
        info = common.run_info(root)
        if args.task and info["task"] not in args.task:
            continue
        if info["t0"] is None or not (args.tasks / info["task"] / "contract.yaml").is_file():
            print(f"skip {root.name}: no start time or no live task {info['task']!r}")
            continue
        r = out[info["run_id"]] = run(root, args.tasks)
        print(f"{info['run_id']:40} {common.model_label(r)[:24]:24} pts={len(r['points']):3} "
              f"versions={len(r['versions']):2} cost=${r['api_cost_total_usd']}")
    path = args.out / "dev_trajectories.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n", "utf-8")
    print(f"{len(out)} runs -> {common.display(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
