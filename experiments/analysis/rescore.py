"""Score an AgentX per-request export with a task's request rule (no GPU).

    D/D = sum_i q_i * (p_in * x_i + p_out * y_i) / 1e6  /  (usd_per_hour * W / 3600)

q_i = 1 iff the request completed with TTFT <= tau_first and TPOT <= tau_token.
Only profiling records count (warmup is excluded); errors and cancellations
count and score 0. The gross figure counts every completed request.

    $SERVERBENCH_PY experiments/analysis/rescore.py --task <task> --export <profile_export.jsonl>
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import common


def metric(record: dict, key: str):
    value = (record.get("metrics") or {}).get(key)
    return value.get("value") if isinstance(value, dict) else value


def profiling(export: Path) -> list[dict]:
    return [r for r in common.read_jsonl(export)
            if (r.get("metadata") or {}).get("benchmark_phase") == "profiling"]


def span_s(records: list[dict]) -> float | None:
    """The profiling window: first request start to last request end."""
    starts = [(r.get("metadata") or {}).get("request_start_ns") for r in records]
    ends = [(r.get("metadata") or {}).get("request_end_ns") for r in records]
    starts, ends = [s for s in starts if s], [e for e in ends if e]
    return (max(ends) - min(starts)) / 1e9 if starts and ends else None


def request_value(record: dict, terms: dict) -> tuple[float, bool]:
    """(token value in USD, whether the request qualifies under the p90 bars)."""
    md = record.get("metadata") or {}
    x = metric(record, "input_sequence_length") or 0
    y = metric(record, "output_sequence_length") or 0
    value = (terms["p_in"] * x + terms["p_out"] * y) / 1e6
    completed = not md.get("was_cancelled") and not record.get("error")
    ttft, tpot = metric(record, "time_to_first_token"), metric(record, "inter_token_latency")
    qualifies = (completed and ttft is not None and tpot is not None and y > 1
                 and math.isfinite(ttft) and math.isfinite(tpot)
                 and ttft <= terms["tau_first"] and tpot <= terms["tau_token"])
    return (value if completed else 0.0), qualifies


def rescore(export: Path, terms: dict, window_s: float) -> dict:
    records = profiling(export)
    qualified_value = gross_value = 0.0
    n_qualified = 0
    for record in records:
        value, qualifies = request_value(record, terms)
        gross_value += value
        if qualifies:
            n_qualified += 1
            qualified_value += value
    cost = terms["usd_per_hour"] * window_s / 3600.0
    return {"requests": len(records), "qualified": n_qualified,
            "dpd": qualified_value / cost if cost else 0.0,
            "gross_dpd": gross_value / cost if cost else 0.0,
            "qualified_value_usd": qualified_value, "gpu_cost_usd": cost}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--window-s", type=float, default=None,
                    help="scored window; default: the export's own profiling span")
    common.add_paths(ap, "tasks")
    args = ap.parse_args()
    window = args.window_s or span_s(profiling(args.export))
    if not window:
        raise SystemExit(f"{args.export} has no profiling records")
    got = rescore(args.export, common.task_terms(args.task, args.tasks), window)
    print(f"window {window:.0f} s  requests {got['requests']}  qualified {got['qualified']}  "
          f"$/$ {got['dpd']:.4f}  gross $/$ {got['gross_dpd']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
