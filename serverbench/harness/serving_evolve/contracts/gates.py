"""Gates: what is checked, what came back, and what that is allowed to mean.

Four verdicts, and the distinction between the last two is the important one:

``PASS`` / ``FAIL``   the gate ran and judged the candidate.
``INCONCLUSIVE``      the gate ran and could not decide — too few samples, a
                      near-tie, references that disagree with each other. The
                      candidate is not condemned and is not credited.
``INVALID``           the *gate* failed, not the candidate: a broken fixture, a
                      contaminated machine, an evaluator crash. Charging this to
                      the candidate would reject good work for infrastructure
                      reasons, so it is tracked in its own denominator.

Two rules the harness enforces rather than documents:

* A gate marked ``can_accept=False`` — every microbenchmark, every screen — may
  reject a candidate but may never be the reason one is accepted. A fast
  microbenchmark that agrees with you is not evidence that the system got faster.
* Gates run cheapest-first, so an idea that a two-second check could kill never
  consumes an hour of device time.
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Sequence
from typing import Any, ClassVar

from .base import Contract


class GateStatus(str, enum.Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"
    INVALID = "INVALID"
    SKIPPED = "SKIPPED"

    @property
    def blocks_acceptance(self) -> bool:
        """Anything that is not an outright PASS stops the candidate advancing."""
        return self is not GateStatus.PASS


class ReasonCode(str, enum.Enum):
    """Stable codes. Automation matches on these; humans read the message."""

    OK = "ok"
    PROTECTED_PATH_TOUCHED = "protected_path_touched"
    BUILD_FAILED = "build_failed"
    IMPORT_FORBIDDEN = "import_forbidden"
    SERVER_DID_NOT_START = "server_did_not_start"
    PROTOCOL_VIOLATION = "protocol_violation"
    TOKEN_MISMATCH = "token_mismatch"
    NUMERICAL_DIVERGENCE = "numerical_divergence"
    STATE_LEAKED_BETWEEN_REQUESTS = "state_leaked_between_requests"
    CACHE_INCONSISTENT = "cache_inconsistent"
    HIDDEN_FALLBACK = "hidden_fallback"
    REQUEST_DROPPED = "request_dropped"
    OUTPUT_TOO_SHORT = "output_too_short"
    UNEQUAL_WORK = "unequal_work"
    ENVELOPE_NARROWED = "envelope_narrowed"
    TIMEOUT = "timeout"
    CANCELLATION_MISHANDLED = "cancellation_mishandled"
    INSUFFICIENT_SAMPLES = "insufficient_samples"
    NEAR_TIE = "near_tie"
    REFERENCE_DISAGREEMENT = "reference_disagreement"
    ENVIRONMENT_CONTAMINATED = "environment_contaminated"
    FIXTURE_STALE = "fixture_stale"
    EVALUATOR_TAMPERED = "evaluator_tampered"
    CI_CROSSES_ZERO = "ci_crosses_zero"
    BUDGET_EXCEEDED = "budget_exceeded"
    ADAPTER_UNSUPPORTED = "adapter_unsupported"
    INFRASTRUCTURE_ERROR = "infrastructure_error"


@dataclasses.dataclass(frozen=True)
class GateSpec(Contract):
    """What a gate is and what it is allowed to conclude."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "gate_spec"

    gate_id: str = ""
    description: str = ""
    #: Ordering key. Lower runs first; the units are only meaningful relative to
    #: one another, and the point is that a cheap gate never waits behind a costly one.
    cost_rank: int = 0
    required: bool = True
    #: False for screens and microbenchmarks: they may reject, never accept.
    can_accept: bool = True
    estimated_seconds: float = 0.0
    estimated_device_seconds: float = 0.0

    def validate(self) -> None:
        self.require(bool(self.gate_id), "a gate needs an id")
        self.require(bool(self.description),
                     "a gate must say what it checks, or its verdict cannot be read")


@dataclasses.dataclass(frozen=True)
class GateResult(Contract):
    """What one gate concluded, with the evidence that justifies it."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "gate_result"

    gate_id: str = ""
    status: GateStatus = GateStatus.INVALID
    reason_code: ReasonCode = ReasonCode.INFRASTRUCTURE_ERROR
    message: str = ""
    #: Denominators are part of the result, not a footnote: a pass over three
    #: samples and a pass over three thousand are different findings.
    observed: int = 0
    conclusive: int = 0
    inconclusive: int = 0
    failures: int = 0
    evidence: dict[str, str] = dataclasses.field(default_factory=dict)
    detail: dict[str, Any] = dataclasses.field(default_factory=dict)
    wall_seconds: float = 0.0
    device_seconds: float = 0.0

    def validate(self) -> None:
        self.require(bool(self.gate_id), "a result needs its gate id")
        status = GateStatus(self.status)
        if status is GateStatus.PASS:
            self.require(ReasonCode(self.reason_code) is ReasonCode.OK,
                         "a PASS carries reason_code ok")
            self.require(self.failures == 0,
                         "a PASS with recorded failures is a contradiction",
                         failures=self.failures)
        else:
            self.require(bool(self.message),
                         f"a {status.value} must explain itself")
        self.require(self.conclusive + self.inconclusive <= self.observed
                     if self.observed else True,
                     "conclusive plus inconclusive cannot exceed what was observed")

    @classmethod
    def passed(cls, gate_id: str, *, observed: int = 0, conclusive: int = 0,
               message: str = "", **kw: Any) -> GateResult:
        return cls(gate_id=gate_id, status=GateStatus.PASS, reason_code=ReasonCode.OK,
                   message=message, observed=observed,
                   conclusive=conclusive or observed, **kw)

    @classmethod
    def failed(cls, gate_id: str, reason: ReasonCode, message: str, **kw: Any) -> GateResult:
        return cls(gate_id=gate_id, status=GateStatus.FAIL, reason_code=reason,
                   message=message, **kw)

    @classmethod
    def inconclusive_(cls, gate_id: str, reason: ReasonCode, message: str,
                      **kw: Any) -> GateResult:
        return cls(gate_id=gate_id, status=GateStatus.INCONCLUSIVE, reason_code=reason,
                   message=message, **kw)

    @classmethod
    def skipped(cls, gate_id: str, message: str, **kw: Any) -> GateResult:
        """A gate that did not run, and says why.

        Used on fail-closed paths, so a gate that cannot run refuses with a
        recorded reason rather than crashing and reporting nothing.
        """
        return cls(gate_id=gate_id, status=GateStatus.SKIPPED,
                   reason_code=ReasonCode.OK, message=message, **kw)

    @classmethod
    def invalid(cls, gate_id: str, reason: ReasonCode, message: str, **kw: Any) -> GateResult:
        return cls(gate_id=gate_id, status=GateStatus.INVALID, reason_code=reason,
                   message=message, **kw)


@dataclasses.dataclass(frozen=True)
class AcceptancePolicy(Contract):
    """The rule that turns gate results into a decision, fixed before they run."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "acceptance_policy"

    policy_id: str = ""
    required_gates: list[str] = dataclasses.field(default_factory=list)
    #: Gates that may reject but never accept — screens, microbenchmarks.
    advisory_gates: list[str] = dataclasses.field(default_factory=list)
    #: INCONCLUSIVE is not a pass. Kept explicit so nobody has to infer it.
    inconclusive_is_acceptance: bool = False
    #: An INVALID gate means the infrastructure failed; the candidate is retried,
    #: not condemned. Retries are bounded so a broken fixture cannot spin forever.
    max_invalid_retries: int = 2

    def validate(self) -> None:
        self.require(bool(self.policy_id), "an acceptance policy needs an id")
        self.require(bool(self.required_gates),
                     "a policy with no required gate accepts everything")
        self.require(not self.inconclusive_is_acceptance,
                     "INCONCLUSIVE may never count as acceptance; failing to find a "
                     "difference is not a demonstration that there is none")
        overlap = set(self.required_gates) & set(self.advisory_gates)
        self.require(not overlap, f"gate(s) both required and advisory: {sorted(overlap)}")


@dataclasses.dataclass(frozen=True)
class AcceptanceDecision(Contract):
    """The verdict, and everything needed to re-derive it."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "acceptance_decision"

    candidate_id: str = ""
    policy_id: str = ""
    accepted: bool = False
    outcome: str = ""
    reason: str = ""
    results: list[GateResult] = dataclasses.field(default_factory=list)
    missing_gates: list[str] = dataclasses.field(default_factory=list)
    #: What the policy required, copied onto the decision so a later reader can
    #: re-derive the verdict from the document alone rather than having to find
    #: the policy it names. When it is empty every result is treated as required,
    #: which is the strict reading.
    required_gates: list[str] = dataclasses.field(default_factory=list)
    decided_utc: str = ""

    def validate(self) -> None:
        self.require(bool(self.candidate_id), "a decision needs its candidate id")
        self.require(bool(self.outcome), "a decision needs an outcome")
        if self.accepted:
            self.require(not self.missing_gates,
                         f"accepted with required gate(s) never run: {self.missing_gates}")
            # Only a *required* gate can contradict an acceptance. An advisory
            # microbenchmark that could not decide, or a rung explicitly skipped
            # with a reason and never required, is not a contradiction.
            considered = (set(self.required_gates) if self.required_gates
                          else {r.gate_id for r in self.results})
            bad = [r.gate_id for r in self.results
                   if r.gate_id in considered and GateStatus(r.status).blocks_acceptance]
            self.require(not bad,
                         f"accepted despite non-PASS required gate(s): {sorted(bad)}")


def decide(candidate_id: str, policy: AcceptancePolicy, results: Sequence[GateResult],
           decided_utc: str = "") -> AcceptanceDecision:
    """Apply a policy to results deterministically.

    The order of checks is fixed and does not depend on which gates happen to be
    present, so the same inputs always produce the same verdict.
    """
    # Sorted so the decision document -- and therefore its digest -- depends on
    # what the gates concluded, never on which of them happened to finish first.
    results = sorted(results, key=lambda r: r.gate_id)
    by_id: dict[str, GateResult] = {r.gate_id: r for r in results}
    missing = [g for g in policy.required_gates if g not in by_id]

    invalid = [r for r in results if GateStatus(r.status) is GateStatus.INVALID]
    if invalid:
        return AcceptanceDecision(
            candidate_id=candidate_id, policy_id=policy.policy_id, accepted=False,
            required_gates=list(policy.required_gates),
            outcome="INVALID",
            reason=("the evaluation failed, not the candidate: "
                    + ", ".join(f"{r.gate_id} ({ReasonCode(r.reason_code).value})"
                                for r in invalid)),
            results=results, missing_gates=missing, decided_utc=decided_utc)

    failed = [r for r in results if GateStatus(r.status) is GateStatus.FAIL]
    if failed:
        return AcceptanceDecision(
            candidate_id=candidate_id, policy_id=policy.policy_id, accepted=False,
            required_gates=list(policy.required_gates),
            outcome="REJECTED",
            reason=", ".join(f"{r.gate_id}: {r.message}" for r in failed),
            results=results, decided_utc=decided_utc)

    unclear = [r for r in results
               if GateStatus(r.status) is GateStatus.INCONCLUSIVE
               and r.gate_id in policy.required_gates]
    if unclear:
        return AcceptanceDecision(
            candidate_id=candidate_id, policy_id=policy.policy_id, accepted=False,
            required_gates=list(policy.required_gates),
            outcome="INCONCLUSIVE",
            reason=("required gate(s) could not decide: "
                    + ", ".join(f"{r.gate_id} ({ReasonCode(r.reason_code).value})"
                                for r in unclear)
                    + ". Not a pass: failing to find a difference is not evidence of none."),
            results=results, decided_utc=decided_utc)

    # A required gate that was *skipped* is neither PASS, FAIL, INCONCLUSIVE nor
    # missing. A skip is a deliberate act with a recorded reason -- and it still
    # means the gate did not judge the candidate, so it cannot support acceptance.
    skipped = [r for r in results
               if GateStatus(r.status) is GateStatus.SKIPPED
               and r.gate_id in policy.required_gates]
    if skipped:
        return AcceptanceDecision(
            candidate_id=candidate_id, policy_id=policy.policy_id, accepted=False,
            required_gates=list(policy.required_gates),
            outcome="INVALID",
            reason=("required gate(s) were skipped and therefore judged nothing: "
                    + ", ".join(f"{r.gate_id} ({r.message})" for r in skipped)),
            results=results, missing_gates=missing, decided_utc=decided_utc)

    # Only now does an unrun gate matter. Fail-fast ordering means a decisive
    # FAIL or INCONCLUSIVE always leaves later gates unrun; treating that as
    # "the evaluation did not complete" would turn every rejection into an
    # infrastructure problem and would trigger retries of a settled verdict.
    # What remains here is the case that really is incomplete: everything that
    # ran passed, and something required never ran at all.
    if missing:
        return AcceptanceDecision(
            candidate_id=candidate_id, policy_id=policy.policy_id, accepted=False,
            required_gates=list(policy.required_gates),
            outcome="INVALID",
            reason=f"required gate(s) never ran: {missing}",
            results=results, missing_gates=missing, decided_utc=decided_utc)

    accepting = [r.gate_id for r in results
                 if GateStatus(r.status) is GateStatus.PASS
                 and r.gate_id in policy.required_gates]
    if not accepting:
        return AcceptanceDecision(
            candidate_id=candidate_id, policy_id=policy.policy_id, accepted=False,
            required_gates=list(policy.required_gates),
            outcome="INCONCLUSIVE",
            reason="no required gate returned a PASS that could support acceptance",
            results=results, decided_utc=decided_utc)

    return AcceptanceDecision(
        candidate_id=candidate_id, policy_id=policy.policy_id, accepted=True,
        required_gates=list(policy.required_gates),
        outcome="ACCEPTED",
        reason=f"every required gate passed: {sorted(accepting)}",
        results=results, decided_utc=decided_utc)
