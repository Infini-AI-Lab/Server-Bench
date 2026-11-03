"""A deterministic kernel campaign that needs no GPU and no compiler.

The harness has to be trustworthy before it is pointed at real hardware, and
"trustworthy" here means something specific: every outcome the machinery can
produce — accepted, rejected, inconclusive, invalid, interrupted, resumed — has
been produced on purpose and looks the way it should.

That is what this module does. Kernels are simulated by a table of scripted rung
outcomes, so a test can ask for an INVALID at rung 6 and get exactly that, every
time, on any machine. Nothing here imports torch, allocates device memory, or
shells out to a compiler.

It is a fixture for the harness, not a way to evaluate a kernel: a scripted rung
result is not evidence about any kernel, and :class:`ScriptedKernel` is not
reachable from the real ladder implementations.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any, ClassVar

from serving_evolve.contracts.base import Contract
from serving_evolve.contracts.gates import (
    AcceptanceDecision,
    GateResult,
    GateStatus,
    ReasonCode,
    decide,
)

from .contracts.candidate import CandidateState
from .verification import RUNGS_BY_ID, LadderPlan, RungFn, run_ladder


@dataclasses.dataclass(frozen=True)
class ScriptedRung(Contract):
    """What one rung is told to conclude, and what it costs while doing it."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "scripted_rung"

    rung: str = ""
    status: GateStatus = GateStatus.PASS
    reason_code: ReasonCode = ReasonCode.OK
    message: str = ""
    device_seconds: float = 0.0
    #: Makes the rung raise instead of returning, so the ladder's own
    #: crash-to-INVALID path is exercised rather than assumed.
    raises: str = ""

    def validate(self) -> None:
        self.require(self.rung in RUNGS_BY_ID, f"{self.rung!r} is not a rung")
        if GateStatus(self.status) is not GateStatus.PASS and not self.raises:
            self.require(bool(self.message), "a non-PASS rung must explain itself")

    def to_result(self) -> GateResult:
        status = GateStatus(self.status)
        common: dict[str, Any] = {"device_seconds": self.device_seconds}
        if status is GateStatus.PASS:
            return GateResult.passed(self.rung, observed=1, message=self.message
                                     or "scripted pass", **common)
        maker = {GateStatus.FAIL: GateResult.failed,
                 GateStatus.INCONCLUSIVE: GateResult.inconclusive_,
                 GateStatus.INVALID: GateResult.invalid}[status]
        return maker(self.rung, ReasonCode(self.reason_code), self.message, **common)


@dataclasses.dataclass(frozen=True)
class ScriptedKernel(Contract):
    """One simulated candidate: an id, and what each rung will say about it."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "scripted_kernel"

    candidate_id: str = ""
    rungs: list[ScriptedRung] = dataclasses.field(default_factory=list)
    #: Speedup reported by the end-to-end rung when it passes.
    speedup: float = 1.0
    speedup_ci: list[float] = dataclasses.field(default_factory=list)

    def validate(self) -> None:
        self.require(bool(self.candidate_id), "a scripted kernel needs an id")
        self.require(bool(self.rungs), "a scripted kernel with no rung scripts nothing")
        ids = [r.rung for r in self.rungs]
        self.require(len(ids) == len(set(ids)), f"duplicate rung script(s) in {ids}")

    def implementations(self) -> dict[str, RungFn]:
        table = {r.rung: r for r in self.rungs}

        def make(rung: str) -> RungFn:
            def fn(_ctx: dict[str, Any]) -> GateResult:
                scripted = table[rung]
                if scripted.raises:
                    raise RuntimeError(scripted.raises)
                return scripted.to_result()
            return fn

        return {rung: make(rung) for rung in table}


@dataclasses.dataclass(frozen=True)
class CampaignOutcome(Contract):
    """What a scripted campaign produced, in the shape a report reads."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "kernel_campaign_outcome"

    campaign_id: str = ""
    decisions: list[AcceptanceDecision] = dataclasses.field(default_factory=list)
    states: dict[str, str] = dataclasses.field(default_factory=dict)
    device_seconds: float = 0.0
    #: Candidates never opened because the budget could not close them. Recorded,
    #: because a silently truncated campaign reads as a complete one.
    not_opened: list[str] = dataclasses.field(default_factory=list)

    def validate(self) -> None:
        self.require(bool(self.campaign_id), "a campaign outcome needs an id")

    @property
    def accepted(self) -> list[str]:
        return [d.candidate_id for d in self.decisions if d.accepted]

    def state_of(self, candidate_id: str) -> CandidateState:
        return CandidateState(self.states[candidate_id])


#: Outcome name -> the candidate state it lands in. One place, so the mapping
#: cannot drift between the controller and the report.
OUTCOME_STATE: dict[str, CandidateState] = {
    "ACCEPTED": CandidateState.ACCEPTED,
    "REJECTED": CandidateState.REJECTED,
    "INCONCLUSIVE": CandidateState.INCONCLUSIVE,
    "INVALID": CandidateState.INVALID,
}


def run_campaign(campaign_id: str, plan: LadderPlan, kernels: Sequence[ScriptedKernel],
                 *, device_budget_seconds: float = float("inf"),
                 per_candidate_seconds: float = 0.0,
                 stop_after: int | None = None) -> CampaignOutcome:
    """Run scripted candidates through the real ladder and the real policy.

    Only the rung *implementations* are simulated. Ordering, fail-fast, the
    accounting of device time, the acceptance policy and the decision are the
    production code paths — which is the point: a fixture that also fakes the
    decision proves nothing about the decision.

    ``stop_after`` simulates an interruption. The campaign returns what it had,
    and a resume continues from the candidates that have no decision yet.
    """
    policy = plan.policy()
    decisions: list[AcceptanceDecision] = []
    states: dict[str, str] = {}
    spent = 0.0
    not_opened: list[str] = []

    for index, kernel in enumerate(kernels):
        if stop_after is not None and index >= stop_after:
            not_opened.extend(k.candidate_id for k in kernels[index:])
            break
        need = per_candidate_seconds or sum(r.device_seconds for r in kernel.rungs)
        if spent + need > device_budget_seconds:
            not_opened.append(kernel.candidate_id)
            continue
        results = run_ladder(plan, kernel.implementations(), {"candidate": kernel})
        spent += sum(r.device_seconds for r in results)
        decision = decide(kernel.candidate_id, policy, results)
        decisions.append(decision)
        states[kernel.candidate_id] = OUTCOME_STATE[decision.outcome].value

    return CampaignOutcome(campaign_id=campaign_id, decisions=decisions, states=states,
                           device_seconds=round(spent, 6), not_opened=not_opened)


def resume_campaign(previous: CampaignOutcome, plan: LadderPlan,
                    kernels: Sequence[ScriptedKernel],
                    **kw: Any) -> CampaignOutcome:
    """Continue a campaign from what it already decided.

    Reads the previous outcome rather than recomputing it. A resume that re-runs
    a decided candidate can reach a different verdict on the same evidence, which
    is the one thing a resume must never do.
    """
    done = {d.candidate_id for d in previous.decisions}
    remaining = [k for k in kernels if k.candidate_id not in done]
    fresh = run_campaign(previous.campaign_id, plan, remaining, **kw)
    return CampaignOutcome(
        campaign_id=previous.campaign_id,
        decisions=[*previous.decisions, *fresh.decisions],
        states={**previous.states, **fresh.states},
        device_seconds=round(previous.device_seconds + fresh.device_seconds, 6),
        not_opened=fresh.not_opened)


# -- ready-made scripts -------------------------------------------------------

def _pass_through(*rungs: str) -> list[ScriptedRung]:
    return [ScriptedRung(rung=r, status=GateStatus.PASS, device_seconds=1.0)
            for r in rungs]


def accepted_kernel(candidate_id: str = "k-accepted", speedup: float = 1.18,
                    plan: LadderPlan | None = None) -> ScriptedKernel:
    plan = plan or LadderPlan(plan_id="scripted")
    return ScriptedKernel(candidate_id=candidate_id, rungs=_pass_through(*plan.rungs),
                          speedup=speedup, speedup_ci=[1.06, 1.31])


def rejected_kernel(candidate_id: str = "k-rejected",
                    at: str = "kernel_parity") -> ScriptedKernel:
    """Fails at one rung, which must stop the ladder before any device-heavy one."""
    plan = LadderPlan(plan_id="scripted")
    rungs: list[ScriptedRung] = []
    for rung in plan.ordered():
        if rung.gate_id == at:
            rungs.append(ScriptedRung(
                rung=at, status=GateStatus.FAIL,
                reason_code=ReasonCode.NUMERICAL_DIVERGENCE,
                message="output diverges from the reference beyond tolerance",
                device_seconds=2.0))
            break
        rungs.append(ScriptedRung(rung=rung.gate_id, device_seconds=1.0))
    return ScriptedKernel(candidate_id=candidate_id, rungs=rungs)


def invalid_kernel(candidate_id: str = "k-invalid",
                   at: str = "kernel_build") -> ScriptedKernel:
    """The toolchain fails. The candidate is not condemned for it."""
    plan = LadderPlan(plan_id="scripted")
    rungs: list[ScriptedRung] = []
    for rung in plan.ordered():
        if rung.gate_id == at:
            rungs.append(ScriptedRung(
                rung=at, status=GateStatus.INVALID,
                reason_code=ReasonCode.INFRASTRUCTURE_ERROR,
                message="nvcc is not installed in the declared environment",
                device_seconds=0.0))
            break
        rungs.append(ScriptedRung(rung=rung.gate_id, device_seconds=1.0))
    return ScriptedKernel(candidate_id=candidate_id, rungs=rungs)


def inconclusive_kernel(candidate_id: str = "k-inconclusive") -> ScriptedKernel:
    """Everything is correct; the end-to-end interval does not exclude no effect."""
    plan = LadderPlan(plan_id="scripted")
    rungs: list[ScriptedRung] = []
    for rung in plan.ordered():
        if rung.gate_id == "kernel_serving_endtoend":
            rungs.append(ScriptedRung(
                rung=rung.gate_id, status=GateStatus.INCONCLUSIVE,
                reason_code=ReasonCode.CI_CROSSES_ZERO,
                message="the paired interval spans no effect; failing to find a "
                        "difference is not evidence that there is none",
                device_seconds=30.0))
            continue
        rungs.append(ScriptedRung(rung=rung.gate_id, device_seconds=1.0))
    return ScriptedKernel(candidate_id=candidate_id, rungs=rungs)


def crashing_kernel(candidate_id: str = "k-crash",
                    at: str = "kernel_numerics") -> ScriptedKernel:
    """The rung itself raises. The ladder must call that INVALID, not FAIL."""
    plan = LadderPlan(plan_id="scripted")
    rungs: list[ScriptedRung] = []
    for rung in plan.ordered():
        if rung.gate_id == at:
            rungs.append(ScriptedRung(rung=at, raises="the evaluator segfaulted",
                                      status=GateStatus.INVALID,
                                      reason_code=ReasonCode.INFRASTRUCTURE_ERROR,
                                      message="unused: this rung raises"))
            break
        rungs.append(ScriptedRung(rung=rung.gate_id, device_seconds=1.0))
    return ScriptedKernel(candidate_id=candidate_id, rungs=rungs)


def microbench_cannot_accept(candidate_id: str = "k-micro-only") -> ScriptedKernel:
    """Fast in isolation, wrong in the model. Must not be accepted."""
    plan = LadderPlan(plan_id="scripted")
    rungs: list[ScriptedRung] = []
    for rung in plan.ordered():
        if rung.gate_id == "kernel_model_integration":
            rungs.append(ScriptedRung(
                rung=rung.gate_id, status=GateStatus.FAIL,
                reason_code=ReasonCode.TOKEN_MISMATCH,
                message="the model's output changes once the kernel is wired in",
                device_seconds=20.0))
            break
        rungs.append(ScriptedRung(rung=rung.gate_id, device_seconds=1.0))
    return ScriptedKernel(candidate_id=candidate_id, rungs=rungs, speedup=3.0)
