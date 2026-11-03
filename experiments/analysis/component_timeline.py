"""When each serving component first appears in a run's published-version notes (no GPU).

Source: the `candidate_noted` journal events, the agent's own sentence about
each version it asked for. A component is tagged by keywords; the hour is
relative to run start. Concurrency is reported as the hour of the LAST note
that sets it (the final choice).

    $SERVERBENCH_PY experiments/analysis/component_timeline.py [--run <run root> ...] [--out DIR]

Default: every run with a final verdict in <out>/final_rescored.csv, else every
run under --evolve. Writes <out>/component_timeline.csv.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import common

COMPONENTS = {
    "own engine": r"engine|paged|tensor-parallel|two-gpu|tp=2",
    "prefix/state reuse": r"prefix|snapshot|radix|state reuse|checkpoint|kv reuse",
    "host-memory KV tier": r"host",
    "CUDA graphs": r"graph",
    "speculative decoding": (r"spec|mtp|draft|n-gram|block (verification|decoding)"
                             r"|same-model .*verification|proposal"),
    "prefill/decode scheduling": (r"prefill|schedul|starvation|srpt|admission|gating|deadline"
                                  r"|decode priority|fair decode"),
}
CONCURRENCY = r"concurrency|lanes|\bc=\d|\bn=\d"


def default_runs(args) -> list[Path]:
    rescored = args.out / "final_rescored.csv"
    if rescored.is_file():
        rows = [r for r in common.read_csv(rescored) if r["kind"] == "final" and r["run_id"]]
        return [args.evolve / r["task"].lower() / r["run_id"] for r in rows]
    return common.run_roots(args.evolve)


def timeline(root: Path) -> dict:
    info = common.run_info(root)
    notes = [(n["hours"], n["note"]) for n in info["notes"]]
    row = {"run_id": info["run_id"], "task": info["task"], "agent_model": info["agent_model"],
           "effort": info["effort"], "versions": len(notes),
           "first_version_h": notes[0][0] if notes else ""}
    for name, pattern in COMPONENTS.items():
        hits = [h for h, n in notes if re.search(pattern, n, re.IGNORECASE)]
        row[name] = hits[0] if hits else ""
    concurrency = [h for h, n in notes if re.search(CONCURRENCY, n, re.IGNORECASE)]
    row["final concurrency set"] = concurrency[-1] if concurrency else ""
    row["notes"] = " | ".join(f"{h}h: {n[:110]}" for h, n in notes)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, action="append", default=[], help="a run root; repeatable")
    common.add_paths(ap, "evolve", "out")
    args = ap.parse_args()
    roots = [r for r in (args.run or default_runs(args)) if (r / "journal.jsonl").is_file()]
    out = [timeline(root) for root in roots]
    if not out:
        print("no runs")
        return 1
    common.write_csv(args.out / "component_timeline.csv", out)
    cols = ["first_version_h", *COMPONENTS, "final concurrency set"]
    print(f"{'run':40}" + "".join(f"{c[:13]:>14}" for c in cols))
    for r in out:
        print(f"{r['run_id'][:40]:40}" + "".join(f"{r[c]!s:>14}" for c in cols))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
