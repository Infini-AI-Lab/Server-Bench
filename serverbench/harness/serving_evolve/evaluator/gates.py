"""The gate suite: registration, cheapest-first execution, and the hard checks.

Gates run in ``cost_rank`` order and stop at the first required failure, so a
change that a two-second check can kill never reaches an hour of device time.
Advisory gates (``can_accept=False``) run too, but their PASS is never a reason
to accept anything — a microbenchmark that agrees with you is not evidence the
system got faster.

The checks in this module are the ones that keep a result honest. Each exists
because the corresponding shortcut is *available* to a system under evaluation:

  identity            serve a different model, tokenizer or template
  exact tokens        change what the model produces and call it faster
  isolation           let one request see another's cache
  accounting          drop the hard requests out of the denominator
  output length       return fewer tokens and finish sooner
  hidden fallback     answer some other way and report it as the fast path
  envelope            narrow the supported shapes until only easy ones remain
  equal work          give the two arms different amounts to do

None of them is clever. They are all just refusals to assume.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from ..contracts.gates import GateResult, GateSpec, GateStatus, ReasonCode
from ..contracts.workload import WorkloadSpec

#: A gate is a callable taking a context mapping and returning its result.
GateFn = Callable[[dict[str, Any]], GateResult]


class GateSuite:
    """An ordered collection of gates and the rules for running them."""

    def __init__(self) -> None:
        self._gates: dict[str, tuple[GateSpec, GateFn]] = {}

    def register(self, spec: GateSpec, fn: GateFn) -> None:
        if spec.gate_id in self._gates:
            raise ValueError(f"gate {spec.gate_id!r} is already registered")
        self._gates[spec.gate_id] = (spec, fn)

    def specs(self) -> list[GateSpec]:
        return [s for s, _ in sorted(self._gates.values(), key=lambda p: (p[0].cost_rank,
                                                                         p[0].gate_id))]

    def run(self, context: dict[str, Any], *, only: Iterable[str] | None = None,
            stop_on_required_failure: bool = True) -> list[GateResult]:
        """Run gates cheapest-first.

        A gate that raises becomes ``INVALID``, not ``FAIL``: an evaluator that
        crashed has told us nothing about the candidate, and charging it to the
        candidate would reject good work for infrastructure reasons.
        """
        wanted = set(only) if only is not None else None
        results: list[GateResult] = []
        for spec in self.specs():
            if wanted is not None and spec.gate_id not in wanted:
                continue
            _, fn = self._gates[spec.gate_id]
            started = time.monotonic()
            try:
                result = fn(context)
            except Exception as exc:  # noqa: BLE001 - converted to a typed verdict
                result = GateResult.invalid(
                    spec.gate_id, ReasonCode.INFRASTRUCTURE_ERROR,
                    f"the gate itself raised {type(exc).__name__}: {exc}")
            result = result.evolve(wall_seconds=round(time.monotonic() - started, 6))
            if not spec.can_accept and GateStatus(result.status) is GateStatus.PASS:
                # An advisory gate may reject but never accept. Its PASS is
                # downgraded so it cannot satisfy a required-gate check.
                result = result.evolve(
                    message=(result.message + " (advisory: this gate may reject but never "
                                              "accept; its PASS is not evidence of "
                                              "improvement)").strip())
            results.append(result)
            if (stop_on_required_failure and spec.required
                    and GateStatus(result.status) is not GateStatus.PASS):
                break
        return results


# ---------------------------------------------------------------------------
# The hard checks
# ---------------------------------------------------------------------------

def identity_gate(context: dict[str, Any]) -> GateResult:
    """Model, tokenizer, template and sampling must be what the workload froze."""
    gid = "identity"
    workload: WorkloadSpec = context["workload"]
    actual: dict[str, str] = context["oracle_identity"]
    expected = {
        "model_id": workload.identity.model_id,
        "revision": workload.identity.revision,
        "sampling": workload.sampling.value if hasattr(workload.sampling, "value")
        else str(workload.sampling),
    }
    if workload.identity.tokenizer_id:
        expected["tokenizer_id"] = workload.identity.tokenizer_id
    diffs = {k: (v, actual.get(k)) for k, v in expected.items() if actual.get(k) != v}
    if diffs:
        return GateResult.failed(
            gid, ReasonCode.PROTOCOL_VIOLATION,
            "the system is not serving the frozen identity: "
            + "; ".join(f"{k} expected {want!r}, got {got!r}" for k, (want, got) in diffs.items()),
            observed=len(expected), failures=len(diffs), detail={"differences": list(diffs)})
    return GateResult.passed(gid, observed=len(expected),
                             message="model, revision and sampling match the frozen workload")


def exact_token_gate(context: dict[str, Any]) -> GateResult:
    """Every prompt must reproduce the reference token sequence exactly.

    Legitimate only against a *single* reference under greedy decoding. Between
    two independent engines, kernel choice moves logits and one different argmax
    diverges everything after it — so cross-engine agreement is a diagnostic and
    never a gate.
    """
    gid = "exact_tokens"
    cases: Sequence[dict[str, Any]] = context["token_cases"]
    mismatches = [c for c in cases if c["actual"] != c["expected"]]
    if not cases:
        return GateResult.inconclusive_(gid, ReasonCode.INSUFFICIENT_SAMPLES,
                                        "no cases were offered, so nothing was checked")
    if mismatches:
        first = mismatches[0]
        idx = next((i for i, (a, b) in enumerate(zip(first["expected"], first["actual"], strict=False))
                    if a != b), min(len(first["expected"]), len(first["actual"])))
        return GateResult.failed(
            gid, ReasonCode.TOKEN_MISMATCH,
            f"{len(mismatches)}/{len(cases)} prompts diverge from the reference; "
            f"first divergence in {first.get('prompt_id', '?')} at token {idx}",
            observed=len(cases), conclusive=len(cases), failures=len(mismatches),
            detail={"first_divergence_index": idx,
                    "prompt_ids": [m.get("prompt_id") for m in mismatches[:20]]})
    return GateResult.passed(gid, observed=len(cases),
                             message=f"{len(cases)}/{len(cases)} exact")


def isolation_gate(context: dict[str, Any]) -> GateResult:
    """The same prompt must produce the same answer regardless of its neighbours.

    Run once alone and once interleaved with others. A difference means state
    crossed between requests — a cache slot, a position offset, a shared buffer.
    """
    gid = "isolation"
    alone: dict[str, Any] = context["alone"]
    interleaved: dict[str, Any] = context["interleaved"]
    keys = sorted(set(alone) | set(interleaved))
    bad = [k for k in keys if alone.get(k) != interleaved.get(k)]
    if not keys:
        return GateResult.inconclusive_(gid, ReasonCode.INSUFFICIENT_SAMPLES,
                                        "no prompts were run in both conditions")
    if bad:
        return GateResult.failed(
            gid, ReasonCode.STATE_LEAKED_BETWEEN_REQUESTS,
            f"{len(bad)}/{len(keys)} prompts answer differently when batched with "
            f"others; state is crossing between requests",
            observed=len(keys), conclusive=len(keys), failures=len(bad),
            detail={"prompts": bad[:20]})
    return GateResult.passed(gid, observed=len(keys),
                             message="batching changes no answer")


def accounting_gate(context: dict[str, Any]) -> GateResult:
    """Every offered request must appear in the outcome set.

    The most effective way to improve a percentile is to stop counting the slow
    requests. This makes that impossible to do quietly.
    """
    gid = "accounting"
    offered: int = context["offered"]
    outcomes: Sequence[dict[str, Any]] = context["outcomes"]
    if offered <= 0:
        return GateResult.invalid(gid, ReasonCode.INFRASTRUCTURE_ERROR,
                                  "zero offered requests: the denominator is empty")
    missing = offered - len(outcomes)
    dropped = [o for o in outcomes if o.get("outcome") == "dropped"]
    if missing > 0:
        return GateResult.failed(
            gid, ReasonCode.REQUEST_DROPPED,
            f"{missing} of {offered} offered requests have no recorded outcome; "
            f"an unaccounted request is a dropped one",
            observed=offered, conclusive=len(outcomes), failures=missing)
    if dropped:
        return GateResult.failed(
            gid, ReasonCode.REQUEST_DROPPED,
            f"{len(dropped)} of {offered} requests were dropped; dropped requests "
            f"stay in the denominator and a run that needs them removed has failed",
            observed=offered, conclusive=len(outcomes), failures=len(dropped),
            detail={"request_ids": [d.get("request_id") for d in dropped[:20]]})
    return GateResult.passed(gid, observed=offered,
                             message=f"all {offered} offered requests accounted for")


def output_length_gate(context: dict[str, Any]) -> GateResult:
    """Output length must match the reference, per request.

    Returning fewer tokens is the cheapest possible speedup and the least
    honest, so length is compared request by request rather than on average —
    an average hides a system that truncates the long ones and pads the short.
    """
    gid = "output_length"
    cases: Sequence[dict[str, Any]] = context["length_cases"]
    if not cases:
        return GateResult.inconclusive_(gid, ReasonCode.INSUFFICIENT_SAMPLES,
                                        "no length cases were offered")
    short = [c for c in cases if c["actual_tokens"] < c["expected_tokens"]]
    long_ = [c for c in cases if c["actual_tokens"] > c["expected_tokens"]]
    if short:
        return GateResult.failed(
            gid, ReasonCode.OUTPUT_TOO_SHORT,
            f"{len(short)}/{len(cases)} responses are shorter than the reference "
            f"(worst: {min(c['actual_tokens'] - c['expected_tokens'] for c in short)} tokens)",
            observed=len(cases), conclusive=len(cases), failures=len(short),
            detail={"request_ids": [c.get("request_id") for c in short[:20]]})
    if long_:
        return GateResult.failed(
            gid, ReasonCode.PROTOCOL_VIOLATION,
            f"{len(long_)}/{len(cases)} responses are longer than the reference; "
            f"stop semantics differ",
            observed=len(cases), conclusive=len(cases), failures=len(long_))
    return GateResult.passed(gid, observed=len(cases), message="every length matches")


def hidden_fallback_gate(context: dict[str, Any]) -> GateResult:
    """No request may take an undeclared path.

    A declared fallback is fine and is reported. An undeclared one means the
    measurement describes a path nobody intended to measure.
    """
    gid = "hidden_fallback"
    outcomes: Sequence[dict[str, Any]] = context["outcomes"]
    declared = set(context.get("declared_fallbacks", ()))
    hidden = [o for o in outcomes if o.get("took_hidden_fallback")]
    undeclared = [o for o in outcomes
                  for f in o.get("fallback", ()) if f not in declared]
    if hidden or undeclared:
        return GateResult.failed(
            gid, ReasonCode.HIDDEN_FALLBACK,
            f"{len(hidden) + len(undeclared)} request(s) took a path that was not "
            f"declared; the result describes something other than the system under test",
            observed=len(outcomes), failures=len(hidden) + len(undeclared))
    return GateResult.passed(gid, observed=len(outcomes),
                             message="no undeclared path was taken")


def envelope_gate(context: dict[str, Any]) -> GateResult:
    """Every shape the workload declares supported must actually work."""
    gid = "envelope"
    workload: WorkloadSpec = context["workload"]
    probes: Sequence[dict[str, Any]] = context["envelope_probes"]
    if not probes:
        return GateResult.inconclusive_(gid, ReasonCode.INSUFFICIENT_SAMPLES,
                                        "no in-envelope shapes were probed")
    refused = [p for p in probes
               if workload.envelope.contains(p["prompt_tokens"], p["output_tokens"])
               and not p["ok"]]
    if refused:
        return GateResult.failed(
            gid, ReasonCode.ENVELOPE_NARROWED,
            f"{len(refused)}/{len(probes)} in-envelope shapes were refused or failed; "
            f"narrowing the envelope is a workload change, not an optimisation",
            observed=len(probes), conclusive=len(probes), failures=len(refused),
            detail={"shapes": [(p["prompt_tokens"], p["output_tokens"])
                               for p in refused[:20]]})
    return GateResult.passed(gid, observed=len(probes),
                             message="every declared shape is served")


def equal_work_gate(context: dict[str, Any]) -> GateResult:
    """Two comparison arms must have been given the same job.

    Compares the *work descriptors* — prompts, token counts, trace digest — not
    the results. Arms that were asked different questions cannot be compared no
    matter how carefully the numbers were collected.
    """
    gid = "equal_work"
    a: dict[str, Any] = context["arm_a"]
    b: dict[str, Any] = context["arm_b"]
    fields = ("trace_digest", "offered", "total_prompt_tokens", "total_max_tokens",
              "workload_digest")
    diffs = [f for f in fields if a.get(f) != b.get(f)]
    if diffs:
        return GateResult.failed(
            gid, ReasonCode.UNEQUAL_WORK,
            f"the two arms did not do the same work; they differ in {diffs}",
            observed=len(fields), failures=len(diffs),
            detail={f: [a.get(f), b.get(f)] for f in diffs})
    return GateResult.passed(gid, observed=len(fields),
                             message="both arms were given identical work")


def build_default_suite() -> GateSuite:
    """The standard ordering: cheap structural checks, then semantics, then load."""
    suite = GateSuite()
    for spec, fn in (
        (GateSpec(gate_id="identity", cost_rank=10, estimated_seconds=1,
                  description="model, revision, tokenizer and sampling match the frozen workload"),
         identity_gate),
        (GateSpec(gate_id="envelope", cost_rank=20, estimated_seconds=10,
                  description="every shape declared supported is actually served"),
         envelope_gate),
        (GateSpec(gate_id="exact_tokens", cost_rank=30, estimated_seconds=30,
                  description="outputs reproduce the reference token sequence exactly"),
         exact_token_gate),
        (GateSpec(gate_id="output_length", cost_rank=35, estimated_seconds=5,
                  description="per-request output length matches the reference"),
         output_length_gate),
        (GateSpec(gate_id="isolation", cost_rank=40, estimated_seconds=60,
                  description="batching a request with others does not change its answer"),
         isolation_gate),
        (GateSpec(gate_id="accounting", cost_rank=50, estimated_seconds=5,
                  description="every offered request has a recorded outcome"),
         accounting_gate),
        (GateSpec(gate_id="hidden_fallback", cost_rank=55, estimated_seconds=5,
                  description="no request took an undeclared path"),
         hidden_fallback_gate),
        (GateSpec(gate_id="equal_work", cost_rank=60, estimated_seconds=1,
                  description="comparison arms were given identical work"),
         equal_work_gate),
    ):
        suite.register(spec, fn)
    return suite
