"""Statistical comparison: a paired cluster bootstrap and its golden tests.

``serverbench evolve golden`` runs it against populations whose right answer is
known by construction. No serving verdict uses it; the final verdict scores
requests against the task's p90 bars.

Three rules are built into the estimator rather than left to the caller:

**The block is the unit of independence, not the request.** Requests inside one
run share a machine, a cache, a queue and a thermal state. Treating them as
independent samples inflates precision by roughly the square root of the requests
per block and manufactures significance out of nothing.

**Equivalence is the only route to a PASS.** "No significant difference" is not
a demonstration of no difference — it is compatible with an enormous difference
and a small sample. An A/A passes only when the whole confidence interval sits
inside the predeclared margin.

**Validity is checked before statistics.** An environmentally invalid block, an
SLO violation, or a denominator that shrank are all decided first, in a fixed
order, so no amount of resampling can rescue a run that should not be analysed.

Deterministic: same raw data, same seed, same verdict, byte for byte.
"""

from __future__ import annotations

import dataclasses
import math
import random
from collections.abc import Sequence
from typing import Any, ClassVar

from ..contracts.base import Contract


@dataclasses.dataclass(frozen=True)
class Block(Contract):
    """One paired observation: the same trace run under both arms."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "paired_block"

    block_id: int = 0
    a_objective: float = 0.0
    b_objective: float = 0.0
    a_valid: bool = True
    b_valid: bool = True
    a_slo_ok: bool = True
    b_slo_ok: bool = True
    offered: int = 0
    completed_a: int = 0
    completed_b: int = 0
    #: "AB" or "BA". Counterbalancing is what exposes an order effect, and an
    #: order effect that is not looked for is an order effect attributed to the
    #: system under test.
    order: str = "AB"

    def validate(self) -> None:
        self.require(self.order in ("AB", "BA"), "order is AB or BA")
        if self.a_valid and self.b_valid:
            self.require(self.a_objective > 0 and self.b_objective > 0,
                         "a log-ratio needs strictly positive objectives on both arms")


@dataclasses.dataclass(frozen=True)
class AnalysisPlan(Contract):
    """Frozen before any data is seen. Changing it afterwards invalidates the run."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "analysis_plan"

    plan_id: str = ""
    resamples: int = 100_000
    seed: int = 0
    confidence: float = 0.95
    #: The smallest relative effect worth calling real. The equivalence margin is
    #: ln(1 + this) either side of zero on the log-ratio scale.
    min_relative_effect: float = 0.03
    min_blocks: int = 10
    min_achieved_over_offered: float = 0.995
    optional_stopping: bool = False
    frozen_before_observing: bool = True

    def validate(self) -> None:
        self.require(bool(self.plan_id), "an analysis plan needs an id")
        self.require(0.5 < self.confidence < 1.0, "confidence is a probability above one half")
        self.require(self.min_relative_effect > 0, "the margin must be positive")
        self.require(self.min_blocks >= 2, "a paired bootstrap needs at least two blocks")
        self.require(not self.optional_stopping,
                     "optional stopping is forbidden: a run that may stop when the "
                     "answer looks good has no error rate worth stating")

    @property
    def margin(self) -> float:
        return math.log(1.0 + self.min_relative_effect)


class PairedBootstrap:
    """Percentile bootstrap over blocks, on the paired log-ratio.

    The log-ratio is the right scale for a strictly positive objective: it makes
    a 10% gain and a 10% loss symmetric, which the raw difference does not.
    """

    name = "paired_cluster_bootstrap"

    def verdict(self, blocks: Sequence[Block], plan: AnalysisPlan) -> dict[str, Any]:
        n_total = len(blocks)

        # 1. validity. An invalid block is reported, never silently dropped.
        invalid = [b.block_id for b in blocks if not (b.a_valid and b.b_valid)]
        usable = [b for b in blocks if b.a_valid and b.b_valid]

        zero_denom = [b.block_id for b in usable if b.offered == 0]
        if zero_denom:
            return _v("INVALID", "a block has a zero denominator",
                      n_total=n_total, blocks=zero_denom)
        if invalid:
            return _v("INVALID", "environmentally invalid block(s)",
                      n_total=n_total, n_usable=len(usable), blocks=invalid)

        # 2. accounting and SLO, before any statistics.
        slo_fail = [b.block_id for b in usable if not (b.a_slo_ok and b.b_slo_ok)]
        if slo_fail:
            return _v("FAIL", "the SLO was violated", n_total=n_total, blocks=slo_fail)
        incomplete = [b.block_id for b in usable
                      if min(b.completed_a, b.completed_b) / b.offered
                      < plan.min_achieved_over_offered]
        if incomplete:
            return _v("FAIL",
                      f"achieved/offered fell below {plan.min_achieved_over_offered}",
                      n_total=n_total, blocks=incomplete)
        if len(usable) < plan.min_blocks:
            return _v("INVALID",
                      f"only {len(usable)} usable paired blocks, {plan.min_blocks} required",
                      n_total=n_total, n_usable=len(usable))

        # 3. paired log-ratios, one per block.
        ratios = [math.log(b.b_objective / b.a_objective) for b in usable]

        # 4. order/carryover, on the same scale, before the interval is read.
        ab = [r for r, b in zip(ratios, usable, strict=False) if b.order == "AB"]
        ba = [r for r, b in zip(ratios, usable, strict=False) if b.order == "BA"]
        carry = abs((sum(ab) / len(ab) if ab else 0.0) - (sum(ba) / len(ba) if ba else 0.0))

        # 5. cluster bootstrap over blocks. Fixed seed, so the verdict is a
        #    function of the data and the plan -- never of when it was run.
        rng = random.Random(plan.seed)
        n = len(ratios)
        means: list[float] = []
        for _ in range(plan.resamples):
            total = 0.0
            for _ in range(n):
                total += ratios[rng.randrange(n)]
            means.append(total / n)
        means.sort()
        alpha = 1.0 - plan.confidence
        lo = means[int(alpha / 2 * plan.resamples)]
        hi = means[int((1 - alpha / 2) * plan.resamples) - 1]

        margin = plan.margin
        if carry >= margin:
            return _v("FAIL",
                      "an order effect at least as large as the equivalence margin means "
                      "the arms are not exchangeable; the comparison is confounded",
                      n_total=n_total, n_usable=n, carryover=round(carry, 6),
                      ci_lower=round(lo, 6), ci_upper=round(hi, 6), margin=round(margin, 6))

        inside = lo > -margin and hi < margin
        return _v(
            "PASS" if inside else "FAIL",
            "the interval lies wholly inside the equivalence margin" if inside
            else "the interval extends outside the equivalence margin",
            n_total=n_total, n_usable=n,
            point_estimate_log_ratio=round(sum(ratios) / n, 6),
            ci_lower=round(lo, 6), ci_upper=round(hi, 6),
            margin=round(margin, 6), carryover=round(carry, 6),
            note=("equivalence is the ONLY route to PASS; failing to find a difference "
                  "is not a demonstration that there is none"))


def _v(verdict: str, reason: str, **detail: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"verdict": verdict, "reason": reason}
    out.update(detail)
    return out


# ---------------------------------------------------------------------------
# Golden populations of known outcome
# ---------------------------------------------------------------------------

def golden_populations(seed: int = 11) -> list[dict[str, Any]]:
    """Populations whose right answer is known by construction."""
    rng = random.Random(seed)

    def make(fn, n: int = 12) -> list[Block]:
        return [fn(i, rng) for i in range(n)]

    def equal(i: int, r: random.Random) -> Block:
        v = 100 + r.gauss(0, 0.4)
        return Block(block_id=i, a_objective=v, b_objective=v + r.gauss(0, 0.4),
                     offered=4096, completed_a=4096, completed_b=4096,
                     order="AB" if i % 2 else "BA")

    def scaled(factor: float):
        def fn(i: int, r: random.Random) -> Block:
            a = 100 + r.gauss(0, 0.3)
            return Block(block_id=i, a_objective=a, b_objective=a * factor,
                         offered=4096, completed_a=4096, completed_b=4096,
                         order="AB" if i % 2 else "BA")
        return fn

    def with_change(base, **changes):
        def fn(i: int, r: random.Random) -> Block:
            b = base(i, r)
            return b.evolve(**{k: v(i) for k, v in changes.items()})
        return fn

    def carryover(i: int, r: random.Random) -> Block:
        a = 100 + r.gauss(0, 0.2)
        # A large order effect that still averages to roughly nothing overall --
        # exactly the shape a mean would miss.
        factor = 1.20 if i % 2 == 0 else 1 / 1.20
        return Block(block_id=i, a_objective=a, b_objective=a * factor,
                     offered=4096, completed_a=4096, completed_b=4096,
                     order="AB" if i % 2 == 0 else "BA")

    return [
        {"name": "known_equivalent", "blocks": make(equal), "expect": "PASS"},
        {"name": "inside_margin", "blocks": make(scaled(1.005)), "expect": "PASS"},
        {"name": "outside_margin_positive", "blocks": make(scaled(1.25)), "expect": "FAIL"},
        {"name": "outside_margin_negative", "blocks": make(scaled(0.75)), "expect": "FAIL"},
        {"name": "slo_fail",
         "blocks": make(with_change(equal, b_slo_ok=lambda i: i != 3)), "expect": "FAIL"},
        {"name": "dropped_incomplete",
         "blocks": make(with_change(equal, completed_b=lambda i: 4000 if i == 2 else 4096)),
         "expect": "FAIL"},
        {"name": "environmental_invalid",
         "blocks": make(with_change(equal, a_valid=lambda i: i != 5)), "expect": "INVALID"},
        {"name": "too_few_blocks", "blocks": make(equal, n=7), "expect": "INVALID"},
        {"name": "zero_denominator",
         "blocks": make(with_change(equal, offered=lambda i: 0 if i == 4 else 4096)),
         "expect": "INVALID"},
        {"name": "order_carryover", "blocks": make(carryover), "expect": "FAIL"},
    ]


def run_golden(plan: AnalysisPlan | None = None) -> dict[str, Any]:
    """Run every golden population, and check determinism on identical data."""
    plan = plan or AnalysisPlan(plan_id="golden", seed=20_260_820, resamples=20_000)
    est = PairedBootstrap()
    results: list[dict[str, Any]] = []
    for pop in golden_populations():
        got = est.verdict(pop["blocks"], plan)
        results.append({"population": pop["name"], "expected": pop["expect"],
                        "observed": got["verdict"],
                        "pass": got["verdict"] == pop["expect"],
                        "reason": got["reason"]})
    same = golden_populations()[0]["blocks"]
    v1, v2 = est.verdict(same, plan), est.verdict(same, plan)
    deterministic = v1 == v2
    return {"estimator": est.name, "plan": plan.to_dict(), "populations": results,
            "deterministic": deterministic,
            "ok": all(r["pass"] for r in results) and deterministic}
