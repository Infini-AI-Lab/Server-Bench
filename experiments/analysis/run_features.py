"""Development-behaviour features of every run that reached a final verdict (no GPU).

Joins <out>/final_rescored.csv (written by rescore_finals.py) to each run's own
development trajectory: when it first measured and first published, how many
versions and measurements it made, how many full-length rehearsals, how many of
its own replays failed fidelity, and its API cost.

    $SERVERBENCH_PY experiments/analysis/run_features.py [--out DIR]

Writes <out>/run_features.csv.
"""

from __future__ import annotations

import argparse

import common
import dev_trajectories

#: A development replay at least this long counts as a full-length rehearsal.
FULL_LENGTH_S = 1500


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_paths(ap, "evolve", "tasks", "out")
    args = ap.parse_args()
    finals = [r for r in common.read_csv(args.out / "final_rescored.csv")
              if r["kind"] == "final" and r["run_id"]]
    out = []
    for r in finals:
        root = args.evolve / r["task"].lower() / r["run_id"]
        if not (root / "journal.jsonl").is_file():
            print(f"skip {r['run_id']}: no run root under {common.display(args.evolve)}")
            continue
        t = dev_trajectories.run(root, args.tasks)
        points = [p for p in t["points"] if common.is_measurement(p["path"])]
        full_tests = [p for p in t["points"] if "/full-tests/" in p["path"]]
        out.append({
            "task": r["task"], "agent_cli": r["agent_cli"], "agent_model": r["agent_model"],
            "effort": r["effort"], "evolve_hours": r["evolve_hours"], "run_id": r["run_id"],
            "final_version": r["version"], "outcome": r["outcome"], "eligible": r["eligible"],
            "failed_checks": r["failed_checks"][:160],
            "final_gross_dpd": r["gross_dpd"], "final_dpd": r["dpd"],
            "first_measure_h": min((p["hours"] for p in points), default=""),
            "first_version_h": min((v["hours"] for v in t["versions"]), default=""),
            "versions": len(t["versions"]), "dev_measurements": len(points),
            "harness_full_tests": len(full_tests),
            "full_length_rehearsals": sum(p["window_s"] >= FULL_LENGTH_S for p in points),
            "dev_fidelity_failures": sum(not p["fidelity_ok"] for p in points),
            "last_version_h": max((v["hours"] for v in t["versions"]), default=""),
            "best_dev_gross": max((p["gross_dpd"] for p in points if p["fidelity_ok"]), default=""),
            "api_cost_usd": t["api_cost_total_usd"],
        })
    if not out:
        print("no final verdict joins a run root")
        return 1
    common.write_csv(args.out / "run_features.csv", out)
    for o in out:
        print(f"{o['task'][:24]:24} {common.model_label(o)[:22]:22} {o['outcome']:7} "
              f"gross={float(o['final_gross_dpd']):5.2f} v1={o['first_version_h']!s:6} "
              f"m1={o['first_measure_h']!s:6} nv={o['versions']:2} nm={o['dev_measurements']:2} "
              f"full={o['full_length_rehearsals']} fidfail={o['dev_fidelity_failures']} "
              f"cost={o['api_cost_usd']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
