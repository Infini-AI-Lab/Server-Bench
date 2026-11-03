"""Why development measurements read differently from the verdict (no GPU).

1. Window head effect: for every eligible final verdict, the scored value rate
   of the first t seconds of its own 1,800 s window, as a multiple of the full
   window. Development replays are usually shorter than the verdict's, so they
   measure the head of a window.
2. Development dips: every drop of at least --dip (default 15%) between
   consecutive development replays of a run, decomposed as
   $/$ = gross $/$ x qualified share. A falling qualified share is a latency
   regression (usually a probe of a higher concurrency); a falling gross with
   the share held is the workload draw.

    $SERVERBENCH_PY experiments/analysis/dev_vs_verdict.py [--out DIR]

Reads <out>/final_rescored.csv and <out>/dev_trajectories.json; writes
<out>/window_head_effect.csv and <out>/dev_dips.csv.
"""

from __future__ import annotations

import argparse
import json
import statistics

import common
import rescore

HEADS_S = [120, 300, 600, 900, 1200, 1500, 1800]
#: Development replays shorter than this are not compared.
MIN_WINDOW_S = 120


def head_curve(export, terms: dict) -> tuple[float, dict]:
    """(full-window $/$, {t: $/$ of the first t seconds / full-window $/$})."""
    records = rescore.profiling(export)
    t0 = min(r["metadata"]["request_start_ns"] for r in records)
    window = (max(r["metadata"]["request_end_ns"] for r in records) - t0) / 1e9
    done = []
    for r in records:
        value, qualifies = rescore.request_value(r, terms)
        done.append(((r["metadata"]["request_end_ns"] - t0) / 1e9, value if qualifies else 0.0))
    rate = terms["usd_per_hour"] / 3600
    full = sum(v for _, v in done) / (rate * window)
    out = {}
    for t in HEADS_S:
        w = min(t, window)
        out[t] = sum(v for end, v in done if end <= w) / (rate * w) / full if full else None
    return full, out


def dips(points: list[dict], threshold: float) -> list[dict]:
    found = []
    for a, b in zip(points, points[1:], strict=False):
        if b["dpd"] >= a["dpd"] * (1 - threshold):
            continue
        share_a, share_b = a["dpd"] / a["gross_dpd"], b["dpd"] / b["gross_dpd"]
        gross_change, share_change = b["gross_dpd"] / a["gross_dpd"] - 1, share_b - share_a
        cause = ("latency" if share_change <= -0.10 and gross_change > -0.15
                 else "draw" if share_change > -0.10 else "both")
        found.append({"from_h": a["hours"], "to_h": b["hours"], "dpd_from": a["dpd"], "dpd_to": b["dpd"],
                      "gross_change": round(gross_change, 3), "qualified_share_from": round(share_a, 3),
                      "qualified_share_to": round(share_b, 3), "share_change": round(share_change, 3),
                      "cause": cause, "measurement": b["path"], "window_s": b["window_s"]})
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-dpd", type=float, default=0.2,
                    help="head effect only for verdicts scoring at least this $/$")
    ap.add_argument("--dip", type=float, default=0.15, help="relative drop that counts as a dip")
    ap.add_argument("--run", action="append", default=[],
                    help="only these run ids for the dips; repeatable (default: every run)")
    common.add_paths(ap, "tasks", "out")
    args = ap.parse_args()

    finals = [r for r in common.read_csv(args.out / "final_rescored.csv")
              if r["kind"] == "final" and r["eligible"] == "True" and float(r["dpd"]) > args.min_dpd]
    curves = []
    for r in finals:
        full, curve = head_curve(r["export"], common.task_terms(r["task"], args.tasks))
        curves.append({"task": r["task"], "run_id": r["run_id"], "agent_model": r["agent_model"],
                       "effort": r["effort"], "concurrency": r["concurrency"], "full_dpd": round(full, 3),
                       **{f"x_first_{t}s": round(curve[t], 3) for t in HEADS_S}})
    if curves:
        common.write_csv(args.out / "window_head_effect.csv", curves)
        for t in HEADS_S:
            v = sorted(c[f"x_first_{t}s"] for c in curves)
            print(f"first {t:4}s: median {statistics.median(v):.2f}x  "
                  f"IQR {v[len(v) // 4]:.2f}-{v[3 * len(v) // 4]:.2f}  "
                  f"higher in {sum(x > 1 for x in v)}/{len(v)}")
    else:
        print("no eligible final verdict above --min-dpd")

    trajectories = json.loads((args.out / "dev_trajectories.json").read_text("utf-8"))
    found = []
    for run_id, t in trajectories.items():
        if args.run and run_id not in args.run:
            continue
        points = [p for p in t["points"] if common.is_measurement(p["path"])
                  and p["window_s"] >= MIN_WINDOW_S and p["gross_dpd"] > 0]
        found += [{"run_id": run_id, "task": t["task"], "agent_model": t["agent_model"],
                   "effort": t["effort"], **d} for d in dips(points, args.dip)]
    if found:
        common.write_csv(args.out / "dev_dips.csv", found)
        causes = {k: sum(d["cause"] == k for d in found) for k in ("latency", "draw", "both")}
        print("dips:", causes, "of", len(found))
    else:
        print("no development dips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
