"""Which offered loads to measure. Two strategies, two different questions.

`frontier()` asks **where does this arm stop holding the gates**.
`sweep()` asks **which of these loads is best**.

They are not interchangeable, and picking the wrong one costs hours. The task
build workflow uses a complete sweep so the selector sees every candidate load
before it freezes the operating point.
"""

from __future__ import annotations

import time

from final_validate.concurrency.arm import gates_for
from final_validate.concurrency.point import measure
from final_validate.concurrency.report import save


def _stamp() -> str:
    return time.strftime("%H:%M:%SZ", time.gmtime())


def _measure_with_one_retry(point: int, *, plan, levels) -> dict:
    """A card that was still being handed back settles in seconds.

    A configuration the server refuses does not settle at all, so this retries
    once and no more: grinding through a whole ladder against a server that
    cannot start burns hours to learn nothing.
    """
    row = measure(point, plan=plan, levels=levels)
    if row.get("invalid"):
        print("    -> INVALID (nothing measured); settling 120s and "
              "retrying once", flush=True)
        time.sleep(120)
        row = measure(point, plan=plan, levels=levels)
    return row


def _report(point: int, row: dict) -> None:
    got = (row.get("measured") or {}).get("effective_concurrency_max")
    replay = row.get("correctness_holds")
    replay_text = ("not measured" if replay is None else
                   ("holds" if replay else "misses"))
    print(f"    -> performance {'HOLDS' if row.get('holds') else 'MISSES'}; "
          f"replay fidelity {replay_text}; "
          f"{row.get('why') or row.get('error', '')} "
          f"(peak HTTP request concurrency {got}; "
          f"offered AgentX lanes {point})", flush=True)


def sweep(plan, points: list[int] | None = None) -> dict:
    """Measure exactly these loads and RANK them. No search, no early exit.

    Every point is measured whether or not an earlier one missed, because a
    miss is not a reason to stop asking which load performs best. Pair it with
    a deliberate `plan.random_seed` only when you want a different draw than
    the verdict's; otherwise the harness seed applies.

    ## What a ranking from this can and cannot claim

    Inside a fixed time box different loads do not complete the same work, and
    the seed does not fix that -- `--concurrency N` opens N lanes that consume
    the seeded stream in parallel, so adjacent loads diverge at once and can
    differ several-fold in request count and mean output length.

    So rank on what is stable across repeats -- where throughput peaks, where
    eviction starts, whether the client ever reaches the offered number -- and
    not on a fine difference between adjacent points, which is the draw.
    """
    points = list(points if points is not None else (plan.points or []))
    if not points:
        raise SystemExit("sweep() needs at least one offered load")
    if any(p < 1 for p in points):
        raise SystemExit("offered loads must be >= 1")

    levels = gates_for(plan.task_dir)
    rows: list[dict] = []
    last_hold = first_miss = None
    for point in points:
        print(f"[{_stamp()}] sweeping: {plan.arm} @ offered concurrency "
              f"{point} ...", flush=True)
        row = _measure_with_one_retry(point, plan=plan, levels=levels)
        rows.append(row)
        if row.get("holds"):
            last_hold = point if last_hold is None else max(last_hold, point)
        elif not row.get("invalid"):
            first_miss = point if first_miss is None else min(first_miss, point)
        save(plan, rows, last_hold, first_miss)
        _report(point, row)
    return save(plan, rows, last_hold, first_miss)


def frontier(plan) -> dict:
    """Find the highest load that holds every gate: double, walk down, bisect.

        start, 2x, 4x, ...   until a gate misses
        then bisect between the last holding point and the first missing one,
        stopping when the bracket is within plan.resolution

    Each point is a full scored window, so budget ~35 minutes per point. The
    doubling phase bounds the answer in log(N) points.
    """
    levels = gates_for(plan.task_dir)
    rows: list[dict] = []
    last_hold = first_miss = None

    point = plan.start
    while point <= plan.ceiling:
        print(f"[{_stamp()}] doubling: {plan.arm} @ offered concurrency "
              f"{point} ...", flush=True)
        row = _measure_with_one_retry(point, plan=plan, levels=levels)
        rows.append(row)
        save(plan, rows, last_hold, first_miss)
        if row.get("invalid"):
            print(f"    -> ABORTING at {point}: {row.get('why')}. This is an "
                  f"infrastructure failure, not a gate miss, and no "
                  f"concurrency answer is reported from it", flush=True)
            break
        _report(point, row)
        if row.get("holds"):
            last_hold = point
            point *= 2
        else:
            first_miss = point
            break

    # THE START POINT CAN ITSELF MISS. Doubling upward from there answers
    # nothing, and the report would say `max_concurrency: None`, which reads
    # as "the search failed" rather than "the answer is below where you
    # started".
    #
    # So walk DOWN instead, halving to 1. If even 1 misses, that is a real and
    # reportable finding about the bar rather than a hole in the search: this
    # arm does not hold these gates at ANY load, and an operator needs to be
    # told that in those words.
    if last_hold is None and first_miss is not None and first_miss > 1:
        point = first_miss // 2
        while point >= 1:
            print(f"[{_stamp()}] walking DOWN: {plan.arm} @ offered "
                  f"concurrency {point} (the start point missed) ...",
                  flush=True)
            row = _measure_with_one_retry(point, plan=plan, levels=levels)
            rows.append(row)
            if row.get("invalid"):
                print(f"    -> ABORTING at {point}: {row.get('why')}",
                      flush=True)
                break
            _report(point, row)
            save(plan, rows, last_hold, first_miss)
            if row.get("holds"):
                last_hold = point
                break
            first_miss = point
            if point == 1:
                break
            point //= 2

    if last_hold is not None and first_miss is not None:
        while first_miss - last_hold > plan.resolution:
            mid = (last_hold + first_miss) // 2
            print(f"[{_stamp()}] bisect: {mid} (bracket "
                  f"{last_hold}..{first_miss}) ...", flush=True)
            row = measure(mid, plan=plan, levels=levels)
            rows.append(row)
            if row.get("holds"):
                last_hold = mid
            else:
                first_miss = mid
            save(plan, rows, last_hold, first_miss)
            _report(mid, row)

    return save(plan, rows, last_hold, first_miss)
