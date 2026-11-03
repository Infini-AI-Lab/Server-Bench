"""The kernel task: what is being evolved, against what reference, under what rule.

A kernel task is the kernel-level analogue of a serving task. It names one
operator at one capability, fixes the reference implementation that defines
correct, and fixes the strategy — may the agent write kernel source at all, or
only dispatch to a frozen catalog.

The reference is the contract's centre of gravity. "Correct" has no meaning
without one, and a reference that is a thin wrapper around the thing being
measured is not a reference: it moves whenever the candidate does. So a reference
declares its own provenance and is hashed, and the ladder refuses a reference it
cannot resolve rather than passing a candidate against nothing.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import ClassVar

from serving_evolve.contracts.base import Contract

from .capability import KernelCapability


class KernelStrategy(str, enum.Enum):
    """What the agent may do with kernels.

    ``EVOLVE`` -- kernel modification is permitted, not required.
    ``PROVIDED_SOTA`` -- the agent may dispatch only to the frozen supplied
    catalog and may not mutate kernel source.
    """

    EVOLVE = "evolve"
    PROVIDED_SOTA = "provided_sota"

    @property
    def may_mutate_kernel_source(self) -> bool:
        return self is KernelStrategy.EVOLVE

    @property
    def may_read_reference_kernels(self) -> bool:
        return self is KernelStrategy.EVOLVE


class ReferenceKind(str, enum.Enum):
    #: A framework-independent implementation in the reference framework's own
    #: operators. Slow, obvious, and the only thing "correct" can mean.
    NAIVE = "naive"
    #: A second independent implementation, used to catch a reference that is
    #: itself wrong. Two references that disagree is INCONCLUSIVE, not FAIL.
    CROSS_CHECK = "cross_check"
    #: Recorded input/output pairs. Cheap, exact, and blind to anything not recorded.
    FIXTURE = "fixture"


@dataclasses.dataclass(frozen=True)
class ReferenceImplementation(Contract):
    """The definition of correct for one kernel task."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "kernel_reference"

    reference_id: str = ""
    kind: ReferenceKind = ReferenceKind.NAIVE
    #: Import path of a module exposing ``make_inputs`` and ``reference_run``.
    module: str = ""
    #: Digest of the reference source, so a moved reference is detectable.
    source_digest: str = ""
    #: Where the semantics came from — a paper, a model card, a spec. Not source
    #: code, and never an upstream serving framework's implementation.
    provenance: str = ""
    #: A reference that calls the same library the candidate dispatches to proves
    #: nothing. Declared so a reviewer can check the claim rather than assume it.
    independent_of_candidate: bool = True

    def validate(self) -> None:
        self.require(bool(self.reference_id), "a reference needs an id")
        self.require(bool(self.module) or self.kind is ReferenceKind.FIXTURE,
                     "a non-fixture reference needs a module to run")
        self.require(bool(self.provenance),
                     "a reference must say where its semantics came from")
        self.require(self.independent_of_candidate,
                     "a reference that shares an implementation with the candidate "
                     "cannot falsify it")


@dataclasses.dataclass(frozen=True)
class KernelTaskSpec(Contract):
    """One kernel, one capability, one reference, one strategy.

    Frozen before any candidate result is visible. ``digest()`` is what a later
    reader compares against to prove the target did not move once the numbers
    started arriving.
    """

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "kernel_task_spec"

    task_id: str = ""
    title: str = ""
    capability: KernelCapability | None = None
    reference: ReferenceImplementation | None = None
    strategy: KernelStrategy = KernelStrategy.EVOLVE
    #: The model whose serving path this kernel sits in. A kernel task with no
    #: model is a microbenchmark pretending to be an optimisation.
    model_id: str = ""
    #: The exact architecture the run is qualified on, e.g. ``sm90``.
    architecture: str = ""
    #: What the kernel must beat to be worth integrating, as a fraction. 0.05
    #: means a 5% local improvement; below it the integration risk is not repaid.
    practical_margin: float = 0.05
    notes: str = ""

    def validate(self) -> None:
        self.require(bool(self.task_id), "a kernel task needs an id")
        self.require(self.capability is not None, "a kernel task needs a capability")
        self.require(self.reference is not None,
                     "a kernel task needs a reference; without one 'correct' is undefined")
        self.require(bool(self.model_id),
                     "a kernel task must name the model whose serving path it is in")
        self.require(bool(self.architecture),
                     "a kernel task must name the architecture it is qualified on")
        self.require(0.0 < self.practical_margin < 1.0,
                     f"practical margin {self.practical_margin} is not a fraction in (0,1)")
        cap = self.capability
        if cap is not None:
            self.require(cap.supports_architecture(self.architecture),
                         f"the capability is not qualified on {self.architecture}; it "
                         f"names {cap.architectures}")
