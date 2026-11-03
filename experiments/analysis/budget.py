"""Which version a run would have shipped had its budget ended earlier (no GPU).

    versions   the lineage's accepted head at each cutoff, for one run
               (experiments/ablation_budget_retrospective.sh measures these)
    curve      $/$ against budget per run: the re-measured verdict of the head
               at each cutoff (the final verdict where that head is the final
               head), and the run's own final verdict at its budget

    $SERVERBENCH_PY experiments/analysis/budget.py versions --run <run root> --hours 2,4,6
    $SERVERBENCH_PY experiments/analysis/budget.py curve --hours 2,4,6 [--out DIR]

`versions` prints one `<hours> <version directory>` line per cutoff, `-` where
no version had been published yet. `curve` reads <out>/final_rescored.csv and
writes <out>/budget_curve.csv; a cutoff with no version scores 0.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import common


def heads_at(info: dict, cutoffs: list[float]) -> dict[float, Path | None]:
    """The accepted head at each cutoff (hours from the run's start)."""
    if not info["lineage"] or info["t0"] is None:
        return dict.fromkeys(cutoffs)
    lineage = Path(info["versions"][-1]["directory"]).parent
    history = [(common.ts(h["at_utc"]), h["version_id"])
               for h in common.read_jsonl(lineage / "head_history.jsonl")
               if h.get("kind") == "accepted_head" and h.get("version_id")]
    out = {}
    for cutoff in cutoffs:
        limit = info["t0"] + cutoff * 3600
        current = [v for at, v in history if at <= limit]
        out[cutoff] = lineage / f"{lineage.name}_{current[-1]}" if current else None
    return out


def cutoff_list(text: str) -> list[float]:
    return [float(h) for h in text.split(",") if h.strip()]


def versions(args) -> int:
    info = common.run_info(args.run)
    for cutoff, version in heads_at(info, cutoff_list(args.hours)).items():
        print(f"{cutoff:g}\t{version if version else '-'}")
    return 0


def curve(args) -> int:
    rows = common.read_csv(args.out / "final_rescored.csv")
    remeasured = {(r["lineage"], r["version_dir"]): r for r in rows if r["kind"] == "remeasured"}
    finals = {r["lineage"]: r for r in rows if r["kind"] == "final"}
    runs = common.runs_by_lineage(args.evolve)
    out = []
    for lineage in sorted({lin for lin, _ in remeasured}):
        info, final = runs.get(lineage), finals.get(lineage)
        if not info:
            print(f"skip {lineage}: no run root under {common.display(args.evolve)}")
            continue
        base = {"task": info["task"], "run_id": info["run_id"], "agent_model": info["agent_model"],
                "effort": info["effort"]}
        for cutoff, version in heads_at(info, cutoff_list(args.hours)).items():
            row = remeasured.get((lineage, version.name)) if version else None
            if not row and final and version and version.name == final["version_dir"]:
                row = final                # the final head's own verdict
            out.append({**base, "budget_h": cutoff, "kind": "retrospective",
                        "version": version.name.rsplit("_", 1)[-1] if version else "",
                        "measured": bool(row) or version is None,
                        "eligible": row["eligible"] if row else ("" if version else "False"),
                        "dpd": row["dpd"] if row else ("" if version else 0.0),
                        "gross_dpd": row["gross_dpd"] if row else ("" if version else 0.0)})
        if final:
            out.append({**base, "budget_h": info["evolve_hours"], "kind": "final",
                        "version": final["version"], "measured": True, "eligible": final["eligible"],
                        "dpd": final["dpd"], "gross_dpd": final["gross_dpd"]})
    if not out:
        print("no re-measured verdicts in final_rescored.csv")
        return 1
    common.write_csv(args.out / "budget_curve.csv", out)
    for o in out:
        print(f"{o['run_id'][:40]:40} {o['budget_h']!s:>5} h {o['kind']:13} {o['version']:4} "
              f"dpd={o['dpd']!s:7} eligible={o['eligible']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    one = sub.add_parser("versions", help="the accepted head at each cutoff, for one run")
    one.add_argument("--run", type=Path, required=True)
    one.add_argument("--hours", default="2,4,6")
    many = sub.add_parser("curve", help="$/$ against budget per re-measured run")
    many.add_argument("--hours", default="2,4,6")
    common.add_paths(many, "evolve", "out")
    args = ap.parse_args()
    return versions(args) if args.verb == "versions" else curve(args)


if __name__ == "__main__":
    raise SystemExit(main())
