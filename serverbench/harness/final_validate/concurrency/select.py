"""Which offered load is best? Three standard answers, and they disagree.

A sweep produces points; choosing one is a separate question with a literature.
This implements the three selections a serving benchmark normally uses, reports
all three, and says which one the task should adopt and why.

## 1. SLO-constrained maximum throughput — the primary rule

The standard used by MLPerf Inference's Server scenario and by vLLM's and
SGLang's own benchmark suites: **among the points that meet every latency
target, take the one with the highest throughput.** Load that violates an SLO
is not a candidate at any throughput, so the constraint comes first and
throughput breaks the tie.

This is the rule to prefer whenever a point holds. It needs no weights, no
normalisation, and nobody has to agree on how many milliseconds a token is
worth.

## 2. The Pareto frontier — when nothing holds

When SGLang misses at least one gate at every load measured, `frontier()`
returns nothing. "No answer" is the honest output of rule 1 there, and it is
also useless for choosing an operating point, which the task still needs.

So the fallback is the **non-dominated set** over (throughput up, TTFT p90 down,
TPOT p90 down). A point is dominated when another is at least as good on all
three and strictly better on one; dominated points are never the right choice
under any weighting, so discarding them costs nothing and usually leaves two or
three real candidates.

## 3. The knee — choosing among the non-dominated

Among non-dominated points somebody must still pick, and the standard
tie-breaker is the **knee of the throughput-versus-latency curve**: the point of
maximum curvature, past which buying more throughput costs disproportionate
latency. Computed here by the usual chord method -- normalise both axes to
[0, 1], draw the line from the first point to the last, and take the point
furthest from it (Satopää et al., "Finding a Kneedle in a Haystack").

The knee is a heuristic, not a truth. It is reported with its distance so a
reader can see whether the curve actually HAS a knee: a distance near zero means
the trade-off is linear and the knee is an artefact of where the sweep stopped.

## What none of these can fix

Throughput across offered loads is only comparable if the points did the same
work, and inside a fixed time box they do not: `--concurrency N` opens N lanes
that consume the seeded stream in parallel, so adjacent points can differ
several-fold in request count and mean output length. Every selection below
therefore carries `sample_moved`, which is true when the points' mean output
lengths differ by more than 25%. When it is true, prefer the latency axes --
they are per-request and far less sensitive to which turns were drawn -- and
treat a throughput ranking as a tie-break at best. A flat throughput curve
next to a TTFT cliff is not a throughput win for whichever point happened to
read highest.
"""

from __future__ import annotations

from typing import Any

#: Mean output length may differ this much between points before their
#: throughput figures stop being comparable with each other.
SAMPLE_DRIFT_TOLERANCE = 0.25


def _f(row: dict, key: str) -> float | None:
    value = (row.get("measured") or {}).get(key)
    return float(value) if isinstance(value, (int, float)) else None


def holds_all(row: dict) -> bool:
    return bool(row.get("holds"))


def usable(points: list[dict]) -> list[dict]:
    """Points that measured something. An `invalid` point is not a data point."""
    return [p for p in points
            if not p.get("invalid") and (p.get("measured") or {})
            and _f(p, "output_tokens_per_s") is not None]


def sample_moved(points: list[dict]) -> bool:
    """Did these points do comparably sized work?

    False does not make throughput trustworthy; True makes it untrustworthy.
    """
    lengths = [_f(p, "mean_output_tokens_per_request") for p in points]
    lengths = [x for x in lengths if x]
    if len(lengths) < 2:
        return False
    return (max(lengths) - min(lengths)) / max(lengths) > SAMPLE_DRIFT_TOLERANCE


def slo_constrained_best(points: list[dict]) -> dict | None:
    """Highest throughput among the points that hold every gate. Rule 1."""
    holding = [p for p in usable(points) if holds_all(p)]
    if not holding:
        return None
    return max(holding, key=lambda p: _f(p, "output_tokens_per_s") or 0.0)


def pareto(points: list[dict]) -> list[dict]:
    """The non-dominated set over throughput up, TTFT p90 down, TPOT p90 down."""
    rows = [p for p in usable(points)
            if _f(p, "ttft_p90_ms") is not None
            and _f(p, "tpot_p90_ms") is not None]

    def dominates(a: dict, b: dict) -> bool:
        at, bt = _f(a, "output_tokens_per_s"), _f(b, "output_tokens_per_s")
        af, bf = _f(a, "ttft_p90_ms"), _f(b, "ttft_p90_ms")
        ap, bp = _f(a, "tpot_p90_ms"), _f(b, "tpot_p90_ms")
        no_worse = at >= bt and af <= bf and ap <= bp
        better = at > bt or af < bf or ap < bp
        return no_worse and better

    return [a for a in rows if not any(dominates(b, a) for b in rows if b is not a)]


def knee(points: list[dict]) -> tuple[dict | None, float]:
    """Maximum-curvature point of throughput versus TTFT p90, and its distance.

    The chord method: normalise both axes, draw a line between the extreme
    points, take the point furthest from it. The distance is returned so a
    caller can tell a real knee from a straight line -- near zero means the
    trade-off is linear and this is an artefact of where the sweep stopped.
    """
    rows = sorted(usable(points), key=lambda p: _f(p, "output_tokens_per_s") or 0.0)
    rows = [p for p in rows if _f(p, "ttft_p90_ms") is not None]
    if len(rows) < 3:
        return (rows[-1] if rows else None), 0.0
    xs = [_f(p, "output_tokens_per_s") for p in rows]
    ys = [_f(p, "ttft_p90_ms") for p in rows]
    dx, dy = (max(xs) - min(xs)) or 1.0, (max(ys) - min(ys)) or 1.0
    nx = [(x - min(xs)) / dx for x in xs]
    ny = [(y - min(ys)) / dy for y in ys]
    x0, y0, x1, y1 = nx[0], ny[0], nx[-1], ny[-1]
    span = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 or 1.0
    best, best_d = None, -1.0
    # taskctl is an operator script and must also run with the repository's
    # system Python 3.9. The three lists are built from the same `rows` above,
    # so plain zip has the same invariant without requiring Python 3.10.
    for row, x, y in zip(rows, nx, ny):  # noqa: B905 - see the note above
        d = abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0) / span
        if d > best_d:
            best, best_d = row, d
    return best, round(best_d, 4)


def recommend(points: list[dict], *, gates: list | None = None) -> dict[str, Any]:
    """All three selections, the one to adopt, and why — in one record."""
    rows = usable(points)
    if not rows:
        return {"recommended": None, "rule": "none",
                "why": "no point measured anything, so there is nothing to choose "
                       "between. This is an infrastructure result, not a finding "
                       "about the server."}

    held = slo_constrained_best(rows)
    front = pareto(rows)
    bend, distance = knee(rows)
    drifted = sample_moved(rows)

    def at(row: dict | None) -> int | None:
        return row.get("concurrency") if row else None

    if held is not None:
        chosen, rule = held, "slo_constrained_max_throughput"
        why = (f"{at(held)} has the highest measured throughput among the "
               f"offered loads where every gate holds. The SLO constraint "
               f"comes first and throughput selects within that feasible set.")
    elif len(front) == 1:
        chosen, rule = front[0], "sole_non_dominated_point"
        why = (f"NO point holds every gate, so there is no frontier. "
               f"{at(front[0])} is the only load not beaten on all of "
               f"throughput, TTFT p90 and TPOT p90 at once.")
    else:
        chosen, rule = bend, "knee_of_the_non_dominated_set"
        why = (f"NO point holds every gate, so there is no frontier and the "
               f"choice is a trade-off. {at(bend)} is the knee of the "
               f"throughput-versus-TTFT curve (chord distance {distance}), past "
               f"which throughput costs disproportionate latency."
               + ("" if distance >= 0.05 else
                  " WARNING: that distance is near zero, so the curve is close "
                  "to linear and this knee is largely an artefact of where the "
                  "sweep stopped. Choose on other grounds."))
    return {
        "recommended": at(chosen),
        "rule": rule,
        "why": why,
        "holds_all_gates_at": at(held),
        "pareto_front": sorted(at(p) for p in front),
        "knee": {"concurrency": at(bend), "chord_distance": distance,
                 "is_a_real_knee": distance >= 0.05},
        "sample_moved": drifted,
        "sample_warning": (
            "" if not drifted else
            "The points' mean output length per request differs by more than "
            f"{SAMPLE_DRIFT_TOLERANCE:.0%}, so they did not do comparably sized "
            "work and their THROUGHPUT figures are not comparable with each "
            "other. Prefer the latency axes, which are per-request. This is a "
            "property of a fixed time box over a heavy-tailed corpus and a "
            "fixed seed does not remove it: --concurrency N opens N lanes that "
            "consume the seeded stream in parallel."),
        "candidates": [
            {"concurrency": p.get("concurrency"),
             "holds": holds_all(p),
             "output_tokens_per_s": _f(p, "output_tokens_per_s"),
             "ttft_p90_ms": _f(p, "ttft_p90_ms"),
             "ttft_p95_ms": _f(p, "ttft_p95_ms"),
             "tpot_p90_ms": _f(p, "tpot_p90_ms"),
             "tpot_p95_ms": _f(p, "tpot_p95_ms"),
             "mean_output_tokens_per_request":
                 _f(p, "mean_output_tokens_per_request"),
             "on_pareto_front": any(p is q for q in front)}
            for p in sorted(rows, key=lambda r: r.get("concurrency", 0))],
        "how_to_read_this": (
            "`recommended` is the measured choice. Adopt it with "
            "`taskctl set <task> --reference-concurrency N`, which regenerates "
            "the complete task."),
    }
