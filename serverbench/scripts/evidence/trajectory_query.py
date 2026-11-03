#!/usr/bin/env python3
"""Search and summarise a trajectory without reading all of it.

A full ATIF document runs to hundreds of kilobytes. An agent that reads one to
answer "did a previous version already try this?" spends its context on another
transcript, and a person grepping raw JSON gets base64 and tool envelopes.

So this is the retrieval layer: ask a question, get the few steps that answer it.
It is a CLI over the on-disk document rather than a service, because
trajectories live beside the version they describe and an agent already has a
shell.

    trajectory_query.py <traj> --summary          what happened, in ~40 lines
    trajectory_query.py <traj> --rounds           hypotheses and their verdicts
    trajectory_query.py <traj> --verdicts         what the held-out check said
    trajectory_query.py <traj> --search "prefix cache"    steps mentioning it
    trajectory_query.py <traj> --tools Bash --grep nvidia-smi
    trajectory_query.py <traj> --step 42          one step in full
    trajectory_query.py <traj> --files            files the run touched
    trajectory_query.py <traj> --timeline         where the wall clock went

`<traj>` may be a trajectory file, a published version directory, or a run
directory; the file is found either way.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

MAX_SNIPPET = 240


def locate(target: Path) -> Path:
    """The trajectory file, given a file, a version directory or a run."""
    target = Path(target)
    if target.is_file():
        return target
    if (target / "trajectory.atif.json").is_file():
        return target / "trajectory.atif.json"
    matches = sorted(target.rglob("trajectory.atif.json"))
    if matches:
        return matches[0]
    raise SystemExit(f"no trajectory.atif.json under {target}")


def load(target: Path) -> dict:
    return json.loads(locate(target).read_text("utf-8"))


def _text_of(step: dict) -> str:
    """Everything searchable in a step: message, reasoning, calls, results."""
    parts = [str(step.get("message") or ""), str(step.get("reasoning_content") or "")]
    for call in step.get("tool_calls") or []:
        parts.append(call.get("function_name", ""))
        parts.append(json.dumps(call.get("arguments") or {}))
    for result in (step.get("observation") or {}).get("results") or []:
        parts.append(str(result.get("content") or ""))
    return "\n".join(parts)


def _snippet(text: str, needle: str, width: int = MAX_SNIPPET) -> str:
    match = re.search(re.escape(needle), text, re.IGNORECASE)
    if not match:
        return text[:width].replace("\n", " ")
    start = max(0, match.start() - width // 3)
    return ("…" if start else "") + text[start:start + width].replace("\n", " ") + "…"


def summary(doc: dict) -> str:
    evolution = (doc.get("extra") or {}).get("evolution") or {}
    final = doc.get("final_metrics") or {}
    task = evolution.get("task") or {}
    out = [f"{task.get('id')} · {task.get('stop_condition')} · "
           f"visibility {(evolution.get('visibility') or {}).get('profile', '?')}",
           f"{final.get('total_steps', 0)} steps · "
           f"{final.get('total_completion_tokens', 0):,} completion tokens · "
           f"${final.get('total_cost_usd') or 0:.2f}", ""]
    rounds = evolution.get("rounds") or []
    if rounds:
        out.append("ROUNDS")
        for entry in rounds:
            out.append(f"  R{entry.get('round'):<4} {entry.get('outcome', ''):<13} "
                       f"{entry.get('title', '')[:60]}")
        out.append("")
    checks = evolution.get("stop_checks") or []
    if checks:
        out.append("HELD-OUT VERDICTS")
        for check in checks:
            out.append(f"  {check.get('check')}: gates={check.get('gate_outcome')} "
                       f"met={check.get('met')}")
            out.append(f"    {str(check.get('reason'))[:100]}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trajectory", type=Path)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--rounds", action="store_true")
    ap.add_argument("--verdicts", action="store_true")
    ap.add_argument("--search", default="")
    ap.add_argument("--tools", default="", help="only steps calling this tool")
    ap.add_argument("--grep", default="", help="regex over tool arguments and results")
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--files", action="store_true")
    ap.add_argument("--timeline", action="store_true")
    ap.add_argument("--limit", type=int, default=12)
    args = ap.parse_args(argv)

    doc = load(args.trajectory)
    steps = doc.get("steps") or []
    evolution = (doc.get("extra") or {}).get("evolution") or {}

    if args.step is not None:
        step = next((s for s in steps if s.get("step_id") == args.step), None)
        if step is None:
            raise SystemExit(f"no step {args.step}; the trajectory has {len(steps)}")
        print(json.dumps(step, indent=1)[:20000])
        return 0

    if args.rounds:
        for entry in evolution.get("rounds") or []:
            print(f"── Round {entry.get('round')} [{entry.get('outcome')}] "
                  f"{entry.get('title', '')}")
            body = (entry.get("text") or "").strip()
            if body:
                print("\n".join("   " + ln for ln in body.splitlines()[:14]))
            print()
        return 0

    if args.verdicts:
        for check in evolution.get("stop_checks") or []:
            print(f"── {check.get('check')}  gates={check.get('gate_outcome')}  "
                  f"met={check.get('met')}")
            print(f"   {check.get('reason')}")
            for gap in (check.get("gaps") or [])[:4]:
                print(f"   gap: {str(gap)[:150]}")
            for gate, status in (check.get("gates") or {}).items():
                print(f"     {gate:<14} {status}")
            measured = [m for m in (check.get("measurements") or [])
                        if m.get("completed")]
            for m in measured[:6]:
                print(f"     C={m['concurrency']:<5} completed {m['completed']}/"
                      f"{m.get('offered')}  p99 ttft {m.get('p99_ttft_ms')}")
            print()
        return 0

    if args.files:
        touched: dict[str, int] = {}
        for step in steps:
            for call in step.get("tool_calls") or []:
                arguments = call.get("arguments") or {}
                for key in ("file_path", "path", "notebook_path"):
                    if arguments.get(key):
                        touched[str(arguments[key])] = touched.get(str(arguments[key]), 0) + 1
                command = str(arguments.get("command") or "")
                for hit in re.findall(r"[\w./-]+\.(?:py|sh|json|md|yaml|toml)", command):
                    touched[hit] = touched.get(hit, 0) + 1
        for name, count in sorted(touched.items(), key=lambda kv: -kv[1])[:40]:
            print(f"  {count:>4}  {name}")
        return 0

    if args.timeline:
        stamped = [(s.get("timestamp"), s) for s in steps if s.get("timestamp")]
        if len(stamped) < 2:
            print("no timestamps recorded")
            return 0
        import datetime as dt
        parse = lambda t: dt.datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
        rows = []
        for (t0, s0), (t1, _) in zip(stamped, stamped[1:], strict=False):
            gap = (parse(t1) - parse(t0)).total_seconds()
            names = ",".join(c.get("function_name", "?")
                             for c in (s0.get("tool_calls") or [])) or "(no tool)"
            rows.append((gap, s0.get("step_id"), names))
        rows.sort(reverse=True)
        print(f"  total {sum(r[0] for r in rows):.0f}s across {len(rows)} gap(s); "
              f"the slowest steps:")
        for gap, step_id, names in rows[:args.limit]:
            print(f"  {gap:>7.1f}s  step {step_id:<5} {names[:60]}")
        return 0

    if args.search or args.tools or args.grep:
        pattern = re.compile(args.grep, re.IGNORECASE) if args.grep else None
        shown = 0
        for step in steps:
            calls = step.get("tool_calls") or []
            if args.tools and not any(c.get("function_name") == args.tools for c in calls):
                continue
            text = _text_of(step)
            if args.search and args.search.lower() not in text.lower():
                continue
            if pattern and not pattern.search(text):
                continue
            shown += 1
            if shown > args.limit:
                print(f"  … more matches; raise --limit (currently {args.limit})")
                break
            names = ",".join(c.get("function_name", "?") for c in calls) or step.get("source")
            print(f"── step {step.get('step_id')} [{names}] {step.get('timestamp', '')}")
            print("   " + _snippet(text, args.search or args.grep or ""))
            print()
        if not shown:
            print("  no step matches")
        return 0

    print(summary(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
