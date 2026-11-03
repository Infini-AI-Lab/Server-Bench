"""Golden cases: prove the gates reject what they are supposed to reject.

A tolerance wide enough to admit legitimate numerical noise is only useful if it
is still narrow enough to catch a real defect. So each case here perturbs a
correct system in a way that mimics one specific serving bug, hands it to the
gate suite, and asserts the verdict. **A corruption that PASSES is a failure of
the policy, not of the case.**

A mutation must be *mathematically* capable of tripping its gate before it can
measure anything: applying ``log_softmax`` to logits leaves softmax exactly
unchanged (KL is 0.0, not "small"), and scaling logits by 1.001 gives a KL around
1e-6 against a 1e-3 bound. Under a noisy backend both look like detections while
measuring only noise, so the corruptions below are sized so that the gate has
to actually work.

Everything here is CPU-only, deterministic, and free of any engine.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

from ..contracts.gates import GateStatus
from ..contracts.workload import ModelIdentity, SamplingMode, ShapeEnvelope, WorkloadSpec
from .gates import GateSuite, build_default_suite


def reference_workload() -> WorkloadSpec:
    return WorkloadSpec(
        workload_id="golden",
        model=ModelIdentity(model_id="fake-model", revision="0" * 40, dtype="n/a"),
        envelope=ShapeEnvelope(min_prompt_tokens=1, max_prompt_tokens=64,
                               min_output_tokens=1, max_output_tokens=16,
                               max_batch_rows=8, max_concurrent_requests=8),
        sampling=SamplingMode.GREEDY,
    )


def clean_context(n: int = 12) -> dict[str, Any]:
    """A correct system's evidence. Every mutation starts from this."""
    workload = reference_workload()
    tokens = {f"p{i}": [f"t{i}{j}" for j in range(8)] for i in range(n)}
    outcomes = [{"request_id": f"p{i}", "outcome": "ok", "output_tokens": 8,
                 "fallback": [], "took_hidden_fallback": False} for i in range(n)]
    arm = {"trace_digest": "d" * 64, "offered": n, "total_prompt_tokens": 40 * n,
           "total_max_tokens": 8 * n, "workload_digest": workload.digest()}
    return {
        "workload": workload,
        "oracle_identity": {"model_id": "fake-model", "revision": "0" * 40,
                            "sampling": "greedy"},
        "token_cases": [{"prompt_id": k, "expected": v, "actual": list(v)}
                        for k, v in tokens.items()],
        "length_cases": [{"request_id": k, "expected_tokens": 8, "actual_tokens": 8}
                         for k in tokens],
        "alone": {k: " ".join(v) for k, v in tokens.items()},
        "interleaved": {k: " ".join(v) for k, v in tokens.items()},
        "offered": n,
        "outcomes": outcomes,
        "declared_fallbacks": [],
        "envelope_probes": [{"prompt_tokens": p, "output_tokens": o, "ok": True}
                            for p, o in ((1, 1), (32, 8), (64, 16))],
        "arm_a": dict(arm),
        "arm_b": dict(arm),
    }


# ---------------------------------------------------------------------------
# Mutations. Each is a named, deliberate defect.
# ---------------------------------------------------------------------------

def _wrong_model(ctx: dict[str, Any]) -> dict[str, Any]:
    ctx["oracle_identity"] = dict(ctx["oracle_identity"], model_id="some-other-model")
    return ctx


def _wrong_revision(ctx: dict[str, Any]) -> dict[str, Any]:
    ctx["oracle_identity"] = dict(ctx["oracle_identity"], revision="f" * 40)
    return ctx


def _sampling_changed(ctx: dict[str, Any]) -> dict[str, Any]:
    ctx["oracle_identity"] = dict(ctx["oracle_identity"], sampling="seeded_sampling")
    return ctx


def _one_token_differs(ctx: dict[str, Any]) -> dict[str, Any]:
    # One token in one prompt. The smallest defect the exact gate must still see:
    # anything weaker would be a tolerance, and this gate has none.
    case = ctx["token_cases"][3]
    actual = list(case["actual"])
    actual[5] = "DIFFERENT"
    ctx["token_cases"][3] = dict(case, actual=actual)
    return ctx


def _state_leaks(ctx: dict[str, Any]) -> dict[str, Any]:
    # One prompt answers differently when it shares a batch: its first token is
    # its neighbour's. That is what a reused cache slot or a stale position
    # offset actually looks like.
    key = sorted(ctx["interleaved"])[2]
    words = ctx["interleaved"][key].split()
    words[0] = "LEAKED_FROM_NEIGHBOUR"
    ctx["interleaved"] = dict(ctx["interleaved"], **{key: " ".join(words)})
    return ctx


def _requests_dropped(ctx: dict[str, Any]) -> dict[str, Any]:
    ctx["outcomes"] = list(ctx["outcomes"])
    ctx["outcomes"][1] = dict(ctx["outcomes"][1], outcome="dropped", output_tokens=0)
    return ctx


def _outcomes_missing(ctx: dict[str, Any]) -> dict[str, Any]:
    # The subtler form: the slow requests are not marked dropped, they simply
    # never appear. offered stays honest, so the gate catches the gap.
    ctx["outcomes"] = ctx["outcomes"][:-3]
    return ctx


def _shorter_output(ctx: dict[str, Any]) -> dict[str, Any]:
    ctx["length_cases"] = list(ctx["length_cases"])
    ctx["length_cases"][4] = dict(ctx["length_cases"][4], actual_tokens=5)
    return ctx


def _hidden_fallback(ctx: dict[str, Any]) -> dict[str, Any]:
    ctx["outcomes"] = list(ctx["outcomes"])
    ctx["outcomes"][2] = dict(ctx["outcomes"][2], took_hidden_fallback=True)
    return ctx


def _undeclared_fallback(ctx: dict[str, Any]) -> dict[str, Any]:
    ctx["outcomes"] = list(ctx["outcomes"])
    ctx["outcomes"][6] = dict(ctx["outcomes"][6], fallback=["reference_engine"])
    return ctx


def _envelope_narrowed(ctx: dict[str, Any]) -> dict[str, Any]:
    ctx["envelope_probes"] = list(ctx["envelope_probes"])
    ctx["envelope_probes"][2] = dict(ctx["envelope_probes"][2], ok=False)
    return ctx


def _unequal_work(ctx: dict[str, Any]) -> dict[str, Any]:
    ctx["arm_b"] = dict(ctx["arm_b"], total_max_tokens=ctx["arm_b"]["total_max_tokens"] // 2)
    return ctx


def _stale_fixture(ctx: dict[str, Any]) -> dict[str, Any]:
    # The reference itself is wrong: expectations from an older workload. Every
    # case diverges at once, which is the signature of a stale fixture rather
    # than of a defect in the system.
    ctx["token_cases"] = [dict(c, expected=[f"OLD{i}" for i in range(8)])
                          for c in ctx["token_cases"]]
    return ctx


def _no_samples(ctx: dict[str, Any]) -> dict[str, Any]:
    ctx["token_cases"] = []
    return ctx


def _zero_denominator(ctx: dict[str, Any]) -> dict[str, Any]:
    ctx["offered"] = 0
    ctx["outcomes"] = []
    return ctx


def _evaluator_tampered(ctx: dict[str, Any]) -> dict[str, Any]:
    # The system reports its own success. The gate never reads a self-report, so
    # this must still pass on the real evidence -- the case exists to prove the
    # claim is ignored rather than believed.
    ctx["self_reported_pass"] = True
    return ctx


@dataclasses.dataclass(frozen=True)
class GoldenCase:
    """One population of known outcome."""

    name: str
    mutate: Callable[[dict[str, Any]], dict[str, Any]]
    expect_status: GateStatus
    expect_gate: str
    why: str


CASES: Sequence[GoldenCase] = (
    GoldenCase("clean", lambda c: c, GateStatus.PASS, "",
               "a correct system must pass every gate"),
    GoldenCase("wrong_model", _wrong_model, GateStatus.FAIL, "identity",
               "serving different weights is not a faster system"),
    GoldenCase("wrong_revision", _wrong_revision, GateStatus.FAIL, "identity",
               "a moved revision is a different model"),
    GoldenCase("sampling_changed", _sampling_changed, GateStatus.FAIL, "identity",
               "non-greedy sampling makes an exact-token reference meaningless"),
    GoldenCase("one_token_differs", _one_token_differs, GateStatus.FAIL, "exact_tokens",
               "the smallest real divergence must still be caught"),
    GoldenCase("state_leaks", _state_leaks, GateStatus.FAIL, "isolation",
               "an answer that depends on its neighbours is wrong"),
    GoldenCase("requests_dropped", _requests_dropped, GateStatus.FAIL, "accounting",
               "dropped requests stay in the denominator"),
    GoldenCase("outcomes_missing", _outcomes_missing, GateStatus.FAIL, "accounting",
               "silently absent outcomes are dropped requests"),
    GoldenCase("shorter_output", _shorter_output, GateStatus.FAIL, "output_length",
               "returning fewer tokens is the cheapest fake speedup"),
    GoldenCase("hidden_fallback", _hidden_fallback, GateStatus.FAIL, "hidden_fallback",
               "an undeclared path means the measurement is of something else"),
    GoldenCase("undeclared_fallback", _undeclared_fallback, GateStatus.FAIL,
               "hidden_fallback", "a fallback not on the declared list is hidden"),
    GoldenCase("envelope_narrowed", _envelope_narrowed, GateStatus.FAIL, "envelope",
               "refusing a supported shape is a workload change"),
    GoldenCase("unequal_work", _unequal_work, GateStatus.FAIL, "equal_work",
               "arms given different work cannot be compared"),
    GoldenCase("stale_fixture", _stale_fixture, GateStatus.FAIL, "exact_tokens",
               "an out-of-date reference must fail loudly, not quietly pass"),
    GoldenCase("no_samples", _no_samples, GateStatus.INCONCLUSIVE, "exact_tokens",
               "nothing checked is not the same as nothing wrong"),
    GoldenCase("zero_denominator", _zero_denominator, GateStatus.INVALID, "accounting",
               "an empty denominator is an evaluator failure, not a candidate failure"),
    GoldenCase("evaluator_tampered", _evaluator_tampered, GateStatus.PASS, "",
               "a system's own claim of success is ignored, not believed"),
)


def run_case(case: GoldenCase, suite: GateSuite | None = None) -> dict[str, Any]:
    """Run one golden case and report what the suite concluded."""
    suite = suite or build_default_suite()
    ctx = case.mutate(clean_context())
    results = suite.run(ctx, stop_on_required_failure=False)
    by_id = {r.gate_id: r for r in results}
    if case.expect_status is GateStatus.PASS:
        offenders = [r.gate_id for r in results if GateStatus(r.status) is not GateStatus.PASS]
        ok = not offenders
        observed = "PASS" if ok else f"{offenders[0]} -> {by_id[offenders[0]].status}"
    else:
        target = by_id.get(case.expect_gate)
        ok = target is not None and GateStatus(target.status) is case.expect_status
        observed = "gate absent" if target is None else str(GateStatus(target.status).value)
    return {
        "case": case.name,
        "expected": f"{case.expect_gate or 'all'} -> {case.expect_status.value}",
        "observed": observed,
        "pass": ok,
        "why": case.why,
        "message": "" if case.expect_status is GateStatus.PASS or case.expect_gate not in by_id
        else by_id[case.expect_gate].message,
    }


def run_all(suite: GateSuite | None = None) -> dict[str, Any]:
    """The whole golden suite. This is what proves the gates are worth trusting."""
    suite = suite or build_default_suite()
    results: list[dict[str, Any]] = [run_case(c, suite) for c in CASES]
    return {
        "cases": results,
        "total": len(results),
        "passed": sum(1 for r in results if r["pass"]),
        "ok": all(r["pass"] for r in results),
        "note": ("a corruption that passes its gate is a failure of the policy, not of "
                 "the case; every mutation here is sized so that the gate must "
                 "actually work to catch it"),
    }
