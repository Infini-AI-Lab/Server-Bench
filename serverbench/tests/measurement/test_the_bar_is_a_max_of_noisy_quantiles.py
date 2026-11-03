"""The bar's derivation, checked where it is cheap to check.

`scripts/audit/bar_is_a_max_of_noisy_quantiles.py` carries the simulation; this
module checks the parts that need no sampling, so the suite stays fast:

* the recorded bar reproduces from the recorded worst-of-S and multiplier,
  for every task that records a derivation -- this is the assertion that goes
  red if someone moves a bar and updates only one of the three places;
* the distribution-free `1/(S+1)` companion is exact arithmetic;
* `E[max of unbiased estimates] >= max of the true values` holds on the
  smallest sample that could falsify a sign error.

The simulation itself is not run here. It is 1,000 trials per cell and belongs
in the audit script, which an operator runs when a bar moves.
"""

from __future__ import annotations

import math
import re
import sys
from fractions import Fraction
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
TASKS = REPO / "serving_tasks"
sys.path.insert(0, str(REPO / "serverbench" / "scripts" / "audit"))
from bar_is_a_max_of_noisy_quantiles import live_tasks  # noqa: E402


def _reference(task_dir: Path) -> dict:
    return yaml.safe_load((task_dir / "reference.yaml").read_text("utf-8"))


def _with_derivation() -> list[Path]:
    return [p for p in sorted(live_tasks())
            if (p / "reference.yaml").is_file()
            and ((_reference(p).get("concurrency_search") or {})
                 .get("slo_derivation"))]


def _ids(paths: list[Path]) -> list[str]:
    return [p.name for p in paths]


def _decimals(value) -> int:
    """How many decimals a recorded number actually carries."""
    text = repr(float(value))
    return len(text.split(".")[1].rstrip("0")) if "." in text else 0


DERIVED = _with_derivation()


@pytest.mark.skipif(not DERIVED, reason="no live task records an slo_derivation")
@pytest.mark.parametrize("task_dir", DERIVED, ids=_ids(DERIVED))
@pytest.mark.parametrize("metric", ("ttft", "tpot"))
def test_the_recorded_bar_reproduces_from_its_own_recorded_runs(
        task_dir: Path, metric: str) -> None:
    slo = _reference(task_dir)["concurrency_search"]["slo_derivation"]
    runs = slo["runs"]
    observed = [float(r[f"{metric}_p90_ms"]) for r in runs]
    worst = float(slo[f"sglang_worst_p90_{metric}_ms"])
    bar = float(slo[f"p90_{metric}_ms"])
    multiplier = float(slo["multiplier"])

    # COMPARED AT THE PRECISION THE EVIDENCE CARRIES.
    #
    # A run may record `tpot_p90_ms: 23.5` while the worst is stated as
    # `23.497353185237426`, so an exact comparison fails on digits the run
    # never recorded. Rounding the worst to the runs' own precision still
    # catches the thing this guards -- a worst that is not the maximum of the
    # runs beneath it -- while not calling a publication rounding an edit.
    digits = max(_decimals(r[f"{metric}_p90_ms"]) for r in runs)
    assert round(worst, digits) == pytest.approx(max(observed)), (
        f"{task_dir.name} {metric}: the derivation records a worst-of-"
        f"{len(runs)} of {worst}, which to the {digits} decimal(s) its runs "
        f"carry is {round(worst, digits)}, but the maximum of those runs is "
        f"{max(observed)}. One of the two was edited without the other")
    derived = multiplier * worst
    if slo.get("published_rounding") == "up to 6 significant figures":
        # `taskctl` writes `:g`. A bar published that way is the derived value
        # rounded UP to 6 significant figures: never tighter, and looser by
        # less than one unit in the sixth figure.
        step = 10.0 ** (math.floor(math.log10(derived)) - 5)
        expected = math.ceil(round(derived / step, 9)) * step
        assert bar == pytest.approx(expected, rel=0, abs=step * 1e-6), (
            f"{task_dir.name} {metric}: {multiplier} x {worst} = {derived}, "
            f"which rounds up to {expected} at 6 significant figures, but the "
            f"derivation states a bar of {bar}")
    else:
        assert derived == pytest.approx(bar, rel=0, abs=1e-6), (
            f"{task_dir.name} {metric}: {multiplier} x {worst} = "
            f"{derived}, but the derivation states a bar of {bar}")

    # And the bar the task is actually scored against is that same number.
    rows = yaml.safe_load(
        (task_dir / "contract.yaml").read_text("utf-8"))["service_levels"]
    scored = float(rows[0][f"{metric}_ms"])
    assert scored == pytest.approx(bar), (
        f"{task_dir.name} {metric}: reference.yaml derives {bar} but "
        f"contract.yaml scores against {scored}")


@pytest.mark.skipif(not DERIVED, reason="no live task records an slo_derivation")
@pytest.mark.parametrize("task_dir", DERIVED, ids=_ids(DERIVED))
def test_the_rule_names_as_many_seeds_as_it_records(task_dir: Path) -> None:
    """The prose and the data must agree about S.

    Dropping a seed below the worst one leaves the bar reproducing perfectly
    while silently making the stated rule false, so the arithmetic checks above
    cannot see it. S is also what sets the `1/(S+1)` guarantee, so a rule that
    says five while recording four overstates what the bar buys.
    """
    slo = _reference(task_dir)["concurrency_search"]["slo_derivation"]
    rule = str(slo.get("rule", ""))
    stated = re.findall(r"\bover (\d+) random seeds\b", rule)
    assert stated, (
        f"{task_dir.name}: the derivation rule does not state how many seeds "
        f"it is over, so nothing can check it: {rule!r}")
    assert int(stated[0]) == len(slo["runs"]), (
        f"{task_dir.name}: the rule says 'over {stated[0]} random seeds' but "
        f"{len(slo['runs'])} run(s) are recorded. Whichever is right, the "
        f"other one is a claim with no evidence behind it")

    seeds = {r["random_seed"] for r in slo["runs"]}
    assert len(seeds) == len(slo["runs"]), (
        f"{task_dir.name}: two recorded runs share a random seed, so they are "
        f"one draw counted twice: {sorted(slo['runs'], key=str)}")


@pytest.mark.parametrize("seeds", (1, 2, 5, 9, 19))
def test_a_fresh_run_exceeds_the_worst_of_s_seeds_with_probability_one_over_s_plus_one(
        seeds: int) -> None:
    """Distribution-free, by exchangeability. Exact arithmetic, no sampling.

    Of `seeds + 1` exchangeable almost-surely-distinct values, each is equally
    likely to be the largest, so the fresh one is largest with probability
    `seeds! / (seeds + 1)!`.
    """
    from bar_is_a_max_of_noisy_quantiles import exceeds_worst_of

    exact = Fraction(math.factorial(seeds), math.factorial(seeds + 1))
    assert exact == Fraction(1, seeds + 1)
    assert exceeds_worst_of(seeds) == pytest.approx(float(exact))


def test_the_max_of_unbiased_estimates_sits_above_the_largest_truth() -> None:
    """Lemma 1 on the smallest sample that catches a sign error.

    `max_i Yhat_i >= Yhat_j` pointwise for each `j`, so taking expectations
    `E[max_i Yhat_i] >= E[Yhat_j] = y_j` for every `j`. Here: two arms, two-point
    symmetric noise, all four sign combinations enumerated with equal weight.
    """
    for y1, y2, a, b in ((1.0, 1.0, 0.5, 0.5),      # a tie, symmetric noise
                         (1.0, 0.9, 0.3, 0.0),      # only the leader is noisy
                         (1.0, 0.9, 0.0, 0.3),      # only the laggard is noisy
                         (2.0, 1.0, 0.1, 5.0)):     # noise dwarfs the gap
        expected = sum(max(y1 + sa * a, y2 + sb * b)
                       for sa in (+1, -1) for sb in (+1, -1)) / 4.0
        assert expected >= max(y1, y2) - 1e-12, (
            f"E[max] {expected} fell below max(truth) {max(y1, y2)} for "
            f"y=({y1},{y2}) noise=({a},{b}); Lemma 1 says it cannot")


def test_the_audit_script_reports_every_task_whose_bar_traces_to_nothing() -> None:
    """A task with no `slo_derivation` has a bar that no measurement supports.

    This is an inventory assertion, not a pass/fail on the design: it fails only
    if the audit script would stay silent about such a task.
    """
    from bar_is_a_max_of_noisy_quantiles import derivations

    live = {p.name for p in sorted(live_tasks())
            if (p / "reference.yaml").is_file()}
    recorded = {name for name, _ in derivations()}
    assert recorded <= live
    # Every live task is either traced or listed as untraced; none is invisible.
    assert live == recorded | (live - recorded)


def _searches() -> list[Path]:
    return [p for p in sorted(live_tasks())
            if (p / "reference.yaml").is_file()
            and ((_reference(p).get("concurrency_search") or {}).get("points"))
            and ((_reference(p).get("concurrency_search") or {}).get("adopted")
                 is not None)]


SEARCHED = _searches()


@pytest.mark.skipif(not SEARCHED, reason="no live task records a search")
@pytest.mark.parametrize("task_dir", SEARCHED, ids=_ids(SEARCHED))
def test_the_adopted_point_is_what_the_recorded_rule_picks(task_dir: Path) -> None:
    """The operating point every bar is measured at has to follow from the rule.

    `adopted` is a number in a file and the rule is prose beside it. Unless
    the first is re-derived from the second, an edit to either --- a
    re-measured point, a hand-set `adopted` --- could leave the task declaring
    a selection it did not make.
    """
    search = _reference(task_dir)["concurrency_search"]
    rule = str(search.get("selection_rule", ""))
    points = search["points"]

    # TWO OBJECTIVES ARE IN USE, and they are not interchangeable.
    #
    # The qwen tasks maximise dollar efficiency per unit latency. The gemma
    # tasks maximise throughput subject to the SLO, and record no `e2e_p90_s`
    # at all, so they could not use the qwen rule even if we wanted them to.
    # Both reproduce on their own terms; what they do NOT support is comparing
    # an operating point chosen under one against a point chosen under the
    # other.
    #
    # The two qwen names, `..._over_p90_request_latency` and
    # `..._over_p90_e2e_seconds`, are the SAME computation under two recorded
    # names, not a second method.
    efficiency = ("max_gross_dollars_per_dollar_over_p90_request_latency",
                  "max_gross_dollars_per_dollar_over_p90_e2e_seconds")
    throughput = ("slo_constrained_max_throughput",)

    if rule in efficiency:
        ranked = sorted(
            ((float(p["gross_dollars_per_dollar"]) / float(p["e2e_p90_s"]),
              int(p["offered"])) for p in points),
            reverse=True)
    elif rule in throughput:
        ranked = sorted(
            ((float(p["output_tokens_per_s"]), int(p["offered"]))
             for p in points),
            reverse=True)
    else:
        raise AssertionError(
            f"{task_dir.name}: selection rule {rule!r} is not one this suite "
            f"knows how to reproduce. A new rule is a new objective, and a "
            f"task whose point cannot be re-derived from its own record states "
            f"a selection nobody can check. Known: {efficiency + throughput}")

    assert ranked[0][1] == int(search["adopted"]), (
        f"{task_dir.name}: the rule {rule} applied to the recorded points "
        f"picks c={ranked[0][1]}, but the task adopted c={search['adopted']}. "
        f"Ranking: {[(c, round(r, 5)) for r, c in ranked]}")


@pytest.mark.skipif(not SEARCHED, reason="no live task records a search")
@pytest.mark.parametrize("task_dir", SEARCHED, ids=_ids(SEARCHED))
def test_each_search_point_is_a_single_draw(task_dir: Path) -> None:
    """Which is why the margin between the top two is worth stating.

    Every point carries its own `random_seed` and there is one point per
    concurrency, so the ranking above compares single draws. When the top two
    are closer than the seed-to-seed spread at one point, the ordering between
    them is not resolvable by this evidence. The test guards that premise: if
    a point gains repeats, this fails and the claim has to be restated.
    """
    search = _reference(task_dir)["concurrency_search"]
    offered = [int(p["offered"]) for p in search["points"]]
    assert len(offered) == len(set(offered)), (
        f"{task_dir.name}: a concurrency appears twice in the search, so the "
        f"points are not one draw each: {sorted(offered)}")
    for point in search["points"]:
        assert "random_seed" in point, (
            f"{task_dir.name}: a search point records no random_seed, so it "
            f"cannot be said whether the search compared draws or averages")
