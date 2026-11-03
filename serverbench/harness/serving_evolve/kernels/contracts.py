"""What a model needs computed, and which kernel is offered for each.

The division of labour is the point of this module. Finding out what the
state of the art is — reading papers, release notes and benchmarks, judging what
is credible — happens **outside** the sandbox, in the harness, by whoever is
driving it. The agent inside the sandbox receives the result as a frozen,
hashed catalog and gets on with building a server.

That split exists because the two jobs need different things. Research needs the
open web, judgement about sources, and no time limit. Construction needs a closed
environment, a fixed input, and a reproducible starting point. Asking a sandboxed
agent to do the first is asking it to guess.

**A catalog entry is a pointer with provenance, never a performance claim.**
Every candidate records where the claim came from and how strong that is, and the
strongest level available in this release is a vendor or third-party claim —
nothing here has been measured. An agent that treats a catalog entry as a
measured result has misread it, and the provisioned material says so in as many
words.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import ClassVar

from ..contracts.base import Contract
from ..contracts.workload import ModelIdentity


class OperatorRole(str, enum.Enum):
    """What a kernel is for. Roles, not implementations."""

    ATTENTION_PREFILL = "attention_prefill"
    ATTENTION_DECODE = "attention_decode"
    PAGED_ATTENTION = "paged_attention"
    NORMALISATION = "normalisation"
    POSITION_EMBEDDING = "position_embedding"
    ACTIVATION_MLP = "activation_mlp"
    LINEAR = "linear"
    QUANTISED_LINEAR = "quantised_linear"
    KV_CACHE_APPEND = "kv_cache_append"
    SAMPLING = "sampling"
    MOE_ROUTING = "moe_routing"
    COLLECTIVE = "collective"


class EvidenceLevel(str, enum.Enum):
    """How much weight a claim about a kernel may carry."""

    #: Measured in this repository, under a frozen plan. A catalog may not
    #: assert it (see ``KernelCandidate.validate``).
    MEASURED_HERE = "measured_here"
    #: The project that ships it says so.
    VENDOR_CLAIM = "vendor_claim"
    #: An independent benchmark or paper says so.
    THIRD_PARTY_CLAIM = "third_party_claim"
    #: Believed, and not sourced. Present so this can be said rather than implied.
    UNVERIFIED = "unverified"

    @property
    def is_measurement(self) -> bool:
        return self is EvidenceLevel.MEASURED_HERE


@dataclasses.dataclass(frozen=True)
class OperatorSpec(Contract):
    """One thing a model needs computed, and what it costs to compute it."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "operator_spec"

    name: str = ""
    role: OperatorRole = OperatorRole.LINEAR
    #: Why this model has it. A serving system that does not know why an operator
    #: is there will replace it with something that does not do the same thing.
    rationale: str = ""
    #: Roughly how much of a decode step it accounts for, if known. A hint for
    #: ordering work, never a measurement of this deployment.
    typical_step_share: float | None = None
    required: bool = True
    notes: str = ""

    def validate(self) -> None:
        self.require(bool(self.name), "an operator needs a name")
        self.require(bool(self.rationale),
                     "an operator must say why the model has it")
        if self.typical_step_share is not None:
            self.require(0.0 <= self.typical_step_share <= 1.0,
                         "a step share is a fraction of one")


@dataclasses.dataclass(frozen=True)
class KernelCandidate(Contract):
    """One implementation offered for one or more roles.

    ``obtain`` says how to get it — a package specifier, or a repository and a
    path. Nothing is vendored into this repository, so the catalog is a set of
    pointers and the environment is what resolves them.
    """

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "kernel_candidate"

    name: str = ""
    project: str = ""
    roles: list[OperatorRole] = dataclasses.field(default_factory=list)
    #: e.g. "pip: flashinfer-python==0.5.2" or "git: <url>@<commit> path/to/dir"
    obtain: str = ""
    license: str = ""
    #: What the kernel needs of the machine. A kernel that needs Hopper is not a
    #: candidate on Ada, and saying so here stops that being discovered at runtime.
    requires: str = ""
    claim: str = ""
    evidence_level: EvidenceLevel = EvidenceLevel.UNVERIFIED
    source: str = ""
    retrieved: str = ""
    notes: str = ""

    def validate(self) -> None:
        self.require(bool(self.name), "a candidate needs a name")
        self.require(bool(self.roles), "a candidate must say what it is for")
        self.require(bool(self.obtain),
                     "a candidate must say how to obtain it, or it cannot be used")
        self.require(bool(self.license),
                     "a candidate must record its licence before anyone depends on it")
        if self.claim:
            self.require(bool(self.source),
                         "a claim needs a source; an unsourced claim is a rumour")
        if EvidenceLevel(self.evidence_level).is_measurement:
            self.require(
                False,
                "MEASURED_HERE may not be asserted in a catalog: it means a "
                "measurement was taken in this repository under a frozen plan, and "
                "the catalog is research rather than measurement")


@dataclasses.dataclass(frozen=True)
class KernelChoice(Contract):
    """What is offered for one operator, and what to do when it does not apply."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "kernel_choice"

    operator: str = ""
    preferred: str = ""
    alternatives: list[str] = dataclasses.field(default_factory=list)
    rationale: str = ""
    #: What to do when nothing above is available. Never "silently do something
    #: else": a fallback is declared, and a system that takes it says so.
    fallback: str = ""

    def validate(self) -> None:
        self.require(bool(self.operator), "a choice needs its operator")
        self.require(bool(self.preferred) or bool(self.fallback),
                     "a choice must offer something, or say what to do instead")
        self.require(bool(self.fallback),
                     "every choice needs a declared fallback; an undeclared one is "
                     "the hidden-fallback defect the gates exist to catch")


@dataclasses.dataclass(frozen=True)
class KernelCatalog(Contract):
    """The frozen result of researching one model on one class of hardware."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "kernel_catalog"

    catalog_id: str = ""
    #: Optional in the type, required once frozen. See WorkloadSpec.model.
    model: ModelIdentity | None = None
    architecture: str = ""
    hardware_target: str = ""
    operators: list[OperatorSpec] = dataclasses.field(default_factory=list)
    candidates: list[KernelCandidate] = dataclasses.field(default_factory=list)
    choices: list[KernelChoice] = dataclasses.field(default_factory=list)
    researched_utc: str = ""
    researched_by: str = ""
    frozen: bool = False
    #: Printed wherever the catalog is delivered. Not a footnote.
    standing_caveat: str = (
        "Every entry here is a pointer with provenance, not a measurement. No "
        "kernel in this catalog has been benchmarked in this repository. Verify "
        "on the target hardware, in the target envelope, before believing any "
        "claim -- and verify correctness before verifying speed."
    )

    def validate(self) -> None:
        self.require(bool(self.catalog_id), "a catalog needs an id")
        if self.frozen:
            self.require(self.model is not None,
                         "a frozen catalog must name the model it was researched for")
        names = {c.name for c in self.candidates}
        for choice in self.choices:
            unknown = [n for n in [choice.preferred, *choice.alternatives]
                       if n and n not in names]
            self.require(not unknown,
                         f"choice for {choice.operator!r} names candidates that are "
                         f"not in the catalog: {unknown}")
        if self.frozen:
            covered = {c.operator for c in self.choices}
            missing = [o.name for o in self.operators if o.required
                       and o.name not in covered]
            self.require(not missing,
                         f"a frozen catalog must cover every required operator; "
                         f"missing: {missing}")

    @property
    def identity(self) -> ModelIdentity:
        """The model, non-optional once frozen. See WorkloadSpec.identity."""
        if self.model is None:
            raise ValueError(
                f"catalog {self.catalog_id!r} names no model; freeze it first")
        return self.model

    def choice_for(self, operator: str) -> KernelChoice | None:
        return next((c for c in self.choices if c.operator == operator), None)

    def candidate(self, name: str) -> KernelCandidate | None:
        return next((c for c in self.candidates if c.name == name), None)

    def coverage(self) -> dict[str, object]:
        required = [o.name for o in self.operators if o.required]
        covered = {c.operator for c in self.choices}
        return {
            "operators": len(self.operators),
            "required": len(required),
            "covered": len([n for n in required if n in covered]),
            "uncovered": sorted(n for n in required if n not in covered),
            "candidates": len(self.candidates),
            "evidence_levels": {
                level.value: sum(1 for c in self.candidates
                                 if EvidenceLevel(c.evidence_level) is level)
                for level in EvidenceLevel},
        }

    def freeze(self) -> KernelCatalog:
        return self.evolve(frozen=True)
