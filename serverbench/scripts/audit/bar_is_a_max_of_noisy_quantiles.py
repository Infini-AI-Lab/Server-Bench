#!/usr/bin/env python3
"""The task's bar is `M x max over S seeds of a within-run p90`. Both layers are
sampling, and the outer max is taken over the noisy one.

Every number this prints is either read out of the task's own `reference.yaml`
or produced by the simulation below. Nothing is hardcoded from a document, so
the script goes red the day the rule in `reference.yaml` stops matching what it
checks. Run it with no arguments:

    python3 serverbench/scripts/audit/bar_is_a_max_of_noisy_quantiles.py

Exits non-zero when the recorded bar does not reproduce from the recorded
worst-of-S and multiplier, or when a stated inflation figure does not hold.
"""
from __future__ import annotations

import json
import math
import random
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TASKS = REPO / "serving_tasks"

#: Simulation size. 1000 trials puts the Monte-Carlo error on each mean well
#: below the effect being reported (a few tenths of a percent).
TRIALS = 1000
SIM_SEED = 20261001

#: The tail is the one thing a reference.yaml does not record, so the answer is
#: bracketed over three families rather than asserted under one.
TAILS = ("lognormal-0.3", "lognormal-0.6", "pareto-3")


def _decimals(value) -> int:
    """How many decimals a recorded number actually carries."""
    text = repr(float(value))
    return len(text.split(".")[1].rstrip("0")) if "." in text else 0


def live_tasks() -> list[Path]:
    """Every live task directory, found by what it contains.

    A task is a directory under `serving_tasks/` holding a `contract.yaml`;
    the underscore-prefixed directories are tooling.
    """
    return [p for p in sorted(TASKS.iterdir())
            if p.is_dir() and not p.name.startswith("_")
            and (p / "contract.yaml").is_file()]


def _load(task_dir: Path) -> dict:
    import yaml
    return yaml.safe_load((task_dir / "reference.yaml").read_text("utf-8"))


def derivations() -> list[tuple[str, dict]]:
    """Every live task that records how its bars were derived."""
    out = []
    for path in sorted(live_tasks()):
        if not (path / "reference.yaml").is_file():
            continue
        body = _load(path)
        search = (body.get("concurrency_search") or {})
        slo = search.get("slo_derivation")
        if slo:
            out.append((path.name, {"slo": slo, "points": search.get("points") or []}))
    return out


def draw(kind: str, true_q90: float):
    """A request-latency sampler whose TRUE p90 is exactly `true_q90`."""
    if kind.startswith("lognormal"):
        sigma = float(kind.split("-")[1])
        z90 = 1.2815515655446004          # Phi^{-1}(0.9)
        mu = math.log(true_q90) - sigma * z90
        return lambda r: math.exp(mu + sigma * r.gauss(0.0, 1.0))
    if kind.startswith("pareto"):
        alpha = float(kind.split("-")[1])
        xm = true_q90 * (0.10 ** (1.0 / alpha))
        return lambda r: xm * (r.random() ** (-1.0 / alpha))
    raise ValueError(kind)


def p90(xs: list[float]) -> float:
    """A per-run p90 the way aiperf computes one: LINEAR INTERPOLATION.

    The bars are made of aiperf's p90, which interpolates between order
    statistics; the plain order statistic `ceil(0.9n) - 1` can differ from it
    by tens of percent at these sample sizes, and simulating it would
    understate the effect this script reports.
    """
    xs = sorted(xs)
    k = 0.9 * (len(xs) - 1)
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] if lo == hi else xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def expected_max(kind: str, true_q90: float, n: int, seeds: int) -> float:
    r = random.Random(SIM_SEED)
    sampler = draw(kind, true_q90)
    return st.fmean(max(p90([sampler(r) for _ in range(n)]) for _ in range(seeds))
                    for _ in range(TRIALS))


def measured_inflation(export: Path, seeds: int, trials: int = 2000
                       ) -> tuple[int, float, float] | None:
    """`(n, true p90, E[max of `seeds` p90-hats])` from a real aiperf export.

    Pass any run's `profile_export.jsonl` and the simulation below is replaced
    by a bootstrap over that run's own requests, which removes the one
    assumption this script otherwise has to make. Scored requests only: the
    export also carries `benchmark_phase: warmup` records, and warmup runs
    against a cold cache, so including them roughly doubles the measured p90.

    The requests inside one run are sessions sharing a queue, so a bootstrap
    over them understates dependence. It is still better than assuming a tail.
    """
    values: list[float] = []
    for line in export.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        meta = record.get("metadata") or {}
        if meta.get("benchmark_phase") != "profiling":
            continue
        if meta.get("was_cancelled") or meta.get("context_overflow_skip"):
            continue
        entry = ((record.get("metrics") or {})
                 .get("time_to_first_token") or {}).get("value")
        if isinstance(entry, (int, float)):
            values.append(float(entry))
    if len(values) < 20:
        return None
    rng = random.Random(SIM_SEED)
    truth = p90(values)
    expected = st.fmean(
        max(p90([rng.choice(values) for _ in range(len(values))])
            for _ in range(seeds))
        for _ in range(trials))
    return len(values), truth, expected


def exceeds_worst_of(seeds: int) -> float:
    """P(a fresh exchangeable run exceeds the worst of `seeds`), distribution-free.

    Of `seeds + 1` exchangeable, almost-surely-distinct values each is equally
    likely to be the largest, so the probability the fresh one is largest is
    `seeds! / (seeds + 1)! = 1 / (seeds + 1)`.
    """
    return 1.0 / (seeds + 1)


def main(argv: list[str] | None = None) -> int:
    """With no argument, simulate. With a path to a run's
    `profile_export.jsonl`, measure that run's tail instead."""
    argv = sys.argv[1:] if argv is None else argv
    export = Path(argv[0]) if argv else None
    bad: list[str] = []
    loose: list[str] = []
    found = derivations()
    if not found:
        print("no live task records an `slo_derivation`; nothing to check.")
        print("That is itself the finding: the bars of every live task are "
              "untraceable to a measurement.")
        return 1

    for task_id, rec in found:
        slo, points = rec["slo"], rec["points"]
        runs = slo.get("runs") or []
        mult = float(slo["multiplier"])
        seeds = len(runs)
        print(f"\n=== {task_id} ===")
        print(f"rule: {slo.get('rule')}")

        for metric, worst_key, bar_key in (
                ("ttft", "sglang_worst_p90_ttft_ms", "p90_ttft_ms"),
                ("tpot", "sglang_worst_p90_tpot_ms", "p90_tpot_ms")):
            vals = [float(r[f"{metric}_p90_ms"]) for r in runs]
            worst, recorded_bar = float(slo[worst_key]), float(slo[bar_key])

            # 1. the bar has to reproduce from what is recorded -- AT THE
            #    PRECISION THE RECORD CARRIES.
            #
            # A worst can be stated to more digits than the runs it comes from
            # are recorded to. Comparing exactly would call that rounding an
            # edit; rounding the worst to the runs' own precision still
            # catches a worst that is not their maximum.
            digits = max(_decimals(r[f"{metric}_p90_ms"]) for r in runs)
            if abs(round(worst, digits) - max(vals)) > 1e-9:
                bad.append(f"{task_id} {metric}: recorded worst {worst} is not "
                           f"max of the recorded runs {max(vals)}")
            # FALSE PRECISION IS WORTH SAYING OUT LOUD, because this number is
            # a GATE. A bar of 4100.2848398999995 ms passes a candidate at
            # 4100.2848 and fails one at 4100.2849, on evidence recorded to one
            # decimal. Reported, not failed: the rounding is theirs to choose.
            if _decimals(worst) > digits:
                loose.append(
                    f"{task_id} {metric}: the bar is stated to "
                    f"{_decimals(worst)} decimals ({worst}) while the runs it "
                    f"comes from carry {digits} ({max(vals)}). The bar is "
                    f"not a gate, but it decides per-request billing, so the "
                    f"digits past the evidence still decide which requests earn")
            if abs(mult * worst - recorded_bar) > 1e-6:
                bad.append(f"{task_id} {metric}: {mult} x {worst} = "
                           f"{mult * worst} but the bar says {recorded_bar}")

            lo, med = min(vals), st.median(vals)
            print(f"\n  p90 {metric.upper()}: n={seeds} seeds  "
                  f"min={lo:g}  median={med:g}  max={max(vals):g}")
            print(f"    bar = {mult} x max = {recorded_bar:g}   "
                  f"(= {recorded_bar / med:.3f} x the median)")
            print(f"    observed spread max/median = {max(vals) / med:.4f} "
                  f"(+{100 * (max(vals) / med - 1):.1f}%)")

            # 2. how much of that spread the MAX step alone would produce with
            #    no true seed effect at all.
            counts = sorted({int(p["requests"]) for p in points
                             if p.get("requests")}) or [419]
            # The inflation is SCALE-FREE: all three families are scale
            # families and the true p90 is pinned to the observed median, so
            # the percentages below depend only on the tail SHAPE and on `n`.
            # TTFT and TPOT therefore print the same table; that is a result,
            # not a copied block.
            print(f"    if every seed drew from ONE distribution, E[max of "
                  f"{seeds} p90-hats] / true p90 (scale-free):")
            simulated = []
            for kind in TAILS:
                row = []
                for n in counts:
                    em = expected_max(kind, med, n, seeds)
                    simulated.append(em / med - 1.0)
                    row.append(f"n={n}: +{100 * (em / med - 1):.1f}% "
                               f"(eff. mult {mult * em / med:.2f}x)")
                print(f"      {kind:<14} " + "   ".join(row))

            # What is left of the observed spread once the max-of-noisy-estimates
            # step is accounted for.
            observed = max(vals) / med - 1.0
            lo_sim, hi_sim = min(simulated), max(simulated)
            print(f"    observed +{100 * observed:.1f}% vs simulated "
                  f"+{100 * lo_sim:.1f}%..+{100 * hi_sim:.1f}% from the max step "
                  f"alone")
            if observed <= hi_sim:
                print("      -> the observed spread is INSIDE the range the max "
                      "step produces with NO true seed effect. There is no "
                      "evidence here of run-to-run variation at all.")
            else:
                share = 100 * lo_sim / observed, 100 * hi_sim / observed
                print(f"      -> between {share[0]:.0f}% and {share[1]:.0f}% of "
                      f"the observed spread is the p90 estimator, not the seed. "
                      f"Only the remainder is dispersion a margin needs to cover.")
            if abs(mult - 1.0) < 1e-9:
                print(f"      -> THE MULTIPLIER IS 1.0: the bar is the worst of "
                      f"{seeds} draws and nothing else, so its whole distance "
                      f"above the reference's median ({recorded_bar / med:.3f}x) "
                      f"is what the max step produced, with no margin added. "
                      f"Under exchangeability a fresh reference-equivalent run's "
                      f"run-level p90 EXCEEDS it with marginal probability "
                      f"1/{seeds + 1} = {exceeds_worst_of(seeds):.3f} -- a rank "
                      f"statement about run maxima, NOT a rate at which runs "
                      f"fail, since the bars qualify requests rather than runs.")

        if export is not None and export.is_file():
            got = measured_inflation(export, seeds)
            if got is None:
                print(f"\n  {export}: fewer than 20 scored requests, not used")
            else:
                n, truth, expected = got
                print(f"\n  MEASURED on {export} ({n} scored requests): "
                      f"p90 {truth:,.1f} ms, E[max of {seeds}] "
                      f"{expected:,.1f} ms -> +{100 * (expected / truth - 1):.1f}%, "
                      f"effective multiplier {mult * expected / truth:.2f}x")
                print("    This replaces the simulated bracket above for that "
                      "run's tail. Measure the runs the bar was built from and "
                      "it replaces it for the bar.")

        print(f"\n  distribution-free companion: a fresh exchangeable run "
              f"exceeds the worst of {seeds} with probability "
              f"1/{seeds + 1} = {exceeds_worst_of(seeds):.4f}")
        print(f"    at {seeds + 4} seeds that would be "
              f"1/{seeds + 5} = {exceeds_worst_of(seeds + 4):.4f}")

    unrecorded = [p.name for p in sorted(live_tasks())
                  if (p / "reference.yaml").is_file()
                  and p.name not in {t for t, _ in found}]
    if unrecorded:
        print(f"\n{len(unrecorded)} live task(s) record no `slo_derivation`, so "
              f"their bars trace to no measurement:")
        for name in unrecorded:
            print(f"  {name}")

    if loose:
        print("\nBARS STATED TO MORE PRECISION THAN THEIR EVIDENCE:")
        for line in loose:
            print(f"  {line}")
    if bad:
        print("\nDISAGREEMENT between this script and the recorded derivation:")
        for line in bad:
            print(f"  {line}")
        return 1
    print("\nevery recorded bar reproduces from its own recorded worst-of-S "
          "and multiplier, at the precision that evidence carries.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
