"""A reference arm's seed runs at its operating point, and the p90 bars they imply (no GPU).

Reads the `calibration.json` that `final_validate.concurrency` wrote for each
seed run (`<dir>/<task>/<arm>/slo-c<N>-s<K>/`, the layout
experiments/reference_bars.sh writes) and reports, per (task, arm, offered
load), every run's p90s, throughput and value, and the worst (highest) p90 TTFT
and p90 TPOT over the valid runs: the rule a task's SGLang-derived bars follow.

    $SERVERBENCH_PY experiments/analysis/reference_bars.py [--dir DIR] [--out DIR]

Writes <out>/reference_seed_runs.csv and <out>/reference_bars.csv. Adopting a
bar is a task change and goes through `taskctl set <task> --p90-ttft ... --p90-tpot
... --reason "..."`; this script only prints that command.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import common
import rescore

DEFAULT_DIR = common.REPO / "serverbench" / "artifacts" / "concurrency" / "reference"
SEED_RUN = re.compile(r"^slo-c(?P<c>\d+)-s(?P<k>\d+)$")


def value(run_dir: Path, arm: str, terms: dict) -> dict:
    """The point's gross and scored $/$ under the task's request rule, from its export."""
    exports = sorted((run_dir / "agentx" / arm).rglob("profile_export.jsonl"))
    if not exports:
        return {"gross_dollars_per_dollar": "", "dollars_per_dollar": ""}
    window = rescore.span_s(rescore.profiling(exports[0]))
    if not window:
        return {"gross_dollars_per_dollar": "", "dollars_per_dollar": ""}
    got = rescore.rescore(exports[0], terms, window)
    return {"gross_dollars_per_dollar": round(got["gross_dpd"], 4),
            "dollars_per_dollar": round(got["dpd"], 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR,
                    help=f"root of the seed runs (default: {common.display(DEFAULT_DIR)})")
    common.add_paths(ap, "tasks", "out")
    args = ap.parse_args()

    runs, groups = [], defaultdict(list)
    for calibration in sorted(args.dir.glob("*/*/slo-c*-s*/calibration.json")):
        match = SEED_RUN.match(calibration.parent.name)
        if not match:
            continue
        task, arm = calibration.parent.parent.parent.name, calibration.parent.parent.name
        report = common.load_json(calibration)
        terms = common.task_terms(task, args.tasks)
        for point in report.get("points") or []:
            measured = point.get("measured") or {}
            row = {"task": task, "arm": arm, "offered_concurrency": int(match["c"]),
                   "seed_run": f"s{match['k']}",
                   "valid": bool(measured) and not point.get("invalid")
                   and point.get("correctness_holds") is not False,
                   "random_seed": measured.get("random_seed"),
                   "requests": measured.get("request_count"),
                   "errors": measured.get("error_request_count"),
                   "osl_mismatches": measured.get("osl_mismatch_count"),
                   "ttft_p90_ms": measured.get("ttft_p90_ms"), "tpot_p90_ms": measured.get("tpot_p90_ms"),
                   "output_tokens_per_s": measured.get("output_tokens_per_s"),
                   **value(calibration.parent / f"c{point.get('concurrency')}", arm, terms),
                   "holds": point.get("holds"), "evidence": str(calibration.parent)}
            runs.append(row)
            if row["valid"] and row["ttft_p90_ms"] is not None and row["tpot_p90_ms"] is not None:
                groups[(task, arm, row["offered_concurrency"])].append(row)
    if not runs:
        print(f"no seed runs under {common.display(args.dir)}")
        return 1
    common.write_csv(args.out / "reference_seed_runs.csv", runs)

    bars = []
    for (task, arm, concurrency), rows in sorted(groups.items()):
        ttft = max(float(r["ttft_p90_ms"]) for r in rows)
        tpot = max(float(r["tpot_p90_ms"]) for r in rows)
        bars.append({"task": task, "arm": arm, "offered_concurrency": concurrency, "valid_runs": len(rows),
                     "worst_ttft_p90_ms": ttft, "worst_tpot_p90_ms": tpot})
        print(f"{task} {arm} c={concurrency}: {len(rows)} valid runs, worst p90 TTFT {ttft:.1f} ms, "
              f"worst p90 TPOT {tpot:.2f} ms")
        print(f"  to adopt: taskctl set {task} --p90-ttft {ttft:.1f} --p90-tpot {tpot:.2f} "
              f"--reason \"worst p90 of {len(rows)} {arm} seed runs at c={concurrency}\"")
    if bars:
        common.write_csv(args.out / "reference_bars.csv", bars)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
