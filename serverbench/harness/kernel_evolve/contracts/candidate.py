"""A kernel candidate: its source, how it is built, what state it is in, what it got.

Two things make this different from an ordinary code change.

The first is that a kernel candidate has a *build*, and a build can fail in ways
that are not the candidate's fault — a missing toolchain, a full disk, a compiler
that segfaults. Those are ``INVALID``, not ``FAIL``, and the distinction is
carried in the receipt rather than reconstructed from a log.

The second is that a kernel is easy to fake. A four-line Python wrapper around
the library the baseline already uses will pass every numerical check and report
the library's speed as the candidate's. So the manifest records what source
exists, the ladder proves a custom kernel actually executed, and neither is
optional.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any, ClassVar

from serving_evolve.contracts.base import Contract


class CandidateState(str, enum.Enum):
    """Where a kernel candidate is. Transitions are one-way except INVALID retry."""

    PROPOSED = "proposed"
    FROZEN = "frozen"
    BUILDING = "building"
    BUILT = "built"
    VERIFYING = "verifying"
    MEASURING = "measuring"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"
    INVALID = "invalid"

    @property
    def terminal(self) -> bool:
        return self in (CandidateState.ACCEPTED, CandidateState.REJECTED,
                        CandidateState.INCONCLUSIVE, CandidateState.INVALID)


#: The only transitions the machine allows. Anything else is a bug in the caller,
#: not a state to be tolerated: a candidate that goes from REJECTED back to
#: BUILDING has lost the record of why it was rejected.
LEGAL_TRANSITIONS: dict[CandidateState, frozenset[CandidateState]] = {
    CandidateState.PROPOSED: frozenset({CandidateState.FROZEN,
                                        CandidateState.REJECTED,
                                        CandidateState.INVALID}),
    CandidateState.FROZEN: frozenset({CandidateState.BUILDING,
                                      CandidateState.INVALID}),
    CandidateState.BUILDING: frozenset({CandidateState.BUILT,
                                        CandidateState.REJECTED,
                                        CandidateState.INVALID}),
    CandidateState.BUILT: frozenset({CandidateState.VERIFYING,
                                     CandidateState.INVALID}),
    CandidateState.VERIFYING: frozenset({CandidateState.MEASURING,
                                         CandidateState.REJECTED,
                                         CandidateState.INCONCLUSIVE,
                                         CandidateState.INVALID}),
    CandidateState.MEASURING: frozenset({CandidateState.ACCEPTED,
                                         CandidateState.REJECTED,
                                         CandidateState.INCONCLUSIVE,
                                         CandidateState.INVALID}),
    CandidateState.ACCEPTED: frozenset(),
    CandidateState.REJECTED: frozenset(),
    CandidateState.INCONCLUSIVE: frozenset(),
    #: An INVALID candidate may be retried from the top, because the evaluation
    #: failed rather than the candidate. Bounded by the budget, not by this table.
    CandidateState.INVALID: frozenset({CandidateState.FROZEN}),
}


@dataclasses.dataclass(frozen=True)
class SourceFile(Contract):
    """One file of candidate source, with the digest that pins it."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "kernel_source_file"

    path: str = ""
    digest: str = ""
    language: str = ""
    lines: int = 0

    def validate(self) -> None:
        self.require(bool(self.path), "a source file needs a path")
        self.require(bool(self.digest), "a source file needs a digest; unhashed source "
                                        "cannot be shown to be what was measured")
        self.require(not self.path.startswith("/"),
                     f"source path {self.path!r} is absolute; candidate manifests are "
                     f"relative to the candidate root so they survive being moved")


@dataclasses.dataclass(frozen=True)
class BuildManifest(Contract):
    """How to turn candidate source into something loadable, reproducibly."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "kernel_build_manifest"

    #: The build system in use. Recorded rather than inferred, because inferring
    #: it from file extensions is how a candidate ends up built two ways.
    builder: str = "torch_cpp_extension"
    sources: list[SourceFile] = dataclasses.field(default_factory=list)
    #: The module to import once built, and the callable inside it.
    entry_module: str = ""
    entry_callable: str = "run"
    #: Compiler flags. ``-arch=native`` and its equivalents belong here, not in a
    #: script: the flags are part of what was measured.
    flags: list[str] = dataclasses.field(default_factory=list)
    #: Extra environment for the build. Values are recorded in the receipt.
    env: dict[str, str] = dataclasses.field(default_factory=dict)
    timeout_seconds: float = 900.0

    def validate(self) -> None:
        self.require(bool(self.sources), "a build with no source builds nothing")
        self.require(bool(self.entry_module),
                     "a build manifest must name the module to import")
        self.require(self.timeout_seconds > 0, "a build timeout must be positive")
        paths = [s.path for s in self.sources]
        self.require(len(paths) == len(set(paths)), f"duplicate source path(s) in {paths}")
        bad = [f for f in self.flags if "fast_math" in f.replace("-", "_")]
        self.require(not bad,
                     f"fast-math flag(s) {bad} change the numerics being compared; a "
                     f"candidate that needs them is not computing the same function")

    @property
    def has_device_source(self) -> bool:
        """Whether any source is a device-language file rather than a Python wrapper."""
        return any(s.path.endswith((".cu", ".cuh", ".cpp", ".cc", ".hip"))
                   or s.language in ("cuda", "triton", "cpp")
                   for s in self.sources)


@dataclasses.dataclass(frozen=True)
class KernelHypothesis(Contract):
    """The one falsifiable claim a candidate is allowed to make.

    One claim, not a list. A candidate that changes three things and gets faster
    has taught nobody which of the three did it, and cannot be rolled back to the
    part that worked.
    """

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "kernel_hypothesis"

    statement: str = ""
    #: What in the profile says this is worth doing. Not "attention is slow".
    profile_evidence: str = ""
    expected_local_speedup: float = 0.0
    #: What the local win is worth end to end once Amdahl is applied. Usually much
    #: smaller than the local number, and the reason many candidates are not worth
    #: running.
    expected_end_to_end_ceiling: float = 0.0
    #: The cheapest observation that would kill the idea, named before the work.
    cheap_falsifier: str = ""
    correctness_risks: list[str] = dataclasses.field(default_factory=list)
    rollback: str = ""

    def validate(self) -> None:
        self.require(bool(self.statement), "a hypothesis needs a statement")
        self.require(bool(self.profile_evidence),
                     "a hypothesis needs profile evidence; without it this is a guess "
                     "with a number attached")
        self.require(bool(self.cheap_falsifier),
                     "a hypothesis must name what would kill it cheaply")
        self.require(self.expected_local_speedup >= 0, "expected speedup cannot be negative")
        self.require(
            self.expected_end_to_end_ceiling <= max(self.expected_local_speedup, 1e-9)
            or self.expected_local_speedup == 0.0,
            f"an end-to-end ceiling ({self.expected_end_to_end_ceiling}) above the local "
            f"speedup ({self.expected_local_speedup}) is arithmetically impossible")
        self.require(bool(self.rollback), "a candidate must say how it is undone")


@dataclasses.dataclass(frozen=True)
class KernelCandidate(Contract):
    """A frozen kernel candidate: identity, source, build, and its single claim."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "kernel_evolve_candidate"

    candidate_id: str = ""
    task_id: str = ""
    parent_candidate_id: str = ""
    hypothesis: KernelHypothesis | None = None
    build: BuildManifest | None = None
    #: Set once the candidate is frozen; every later stage is checked against it.
    source_tree_digest: str = ""
    created_utc: str = ""
    author: str = ""

    def validate(self) -> None:
        self.require(bool(self.candidate_id), "a candidate needs an id")
        self.require(bool(self.task_id), "a candidate needs the task it answers")
        self.require(self.build is not None, "a candidate needs a build manifest")
        self.require(self.hypothesis is not None,
                     "a candidate needs exactly one hypothesis; a change with no claim "
                     "cannot be accepted or refuted")

    @property
    def frozen(self) -> bool:
        return bool(self.source_tree_digest)


@dataclasses.dataclass(frozen=True)
class KernelCandidateResult(Contract):
    """What happened to a candidate, in a form a later reader can re-derive."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "kernel_evolve_candidate_result"

    candidate_id: str = ""
    task_id: str = ""
    state: CandidateState = CandidateState.PROPOSED
    #: Gate results, in the order the ladder ran them.
    rungs: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    #: The rung that stopped the ladder, empty if it ran to the end.
    stopped_at: str = ""
    measured_speedup: float | None = None
    speedup_ci: list[float] = dataclasses.field(default_factory=list)
    wall_seconds: float = 0.0
    device_seconds: float = 0.0
    evidence: dict[str, str] = dataclasses.field(default_factory=dict)
    message: str = ""

    def validate(self) -> None:
        self.require(bool(self.candidate_id), "a result needs its candidate id")
        state = CandidateState(self.state)
        if state is CandidateState.ACCEPTED:
            self.require(not self.stopped_at,
                         f"accepted but the ladder stopped at {self.stopped_at!r}")
            self.require(self.measured_speedup is not None,
                         "accepted with no measured speedup")
            self.require(len(self.speedup_ci) == 2,
                         "an accepted candidate carries a two-sided interval")
            self.require(self.speedup_ci[0] > 0.0,
                         f"accepted with a confidence interval whose lower bound is "
                         f"{self.speedup_ci[0]}; that does not exclude no effect")
        if state in (CandidateState.REJECTED, CandidateState.INVALID,
                     CandidateState.INCONCLUSIVE):
            self.require(bool(self.message), f"a {state.value} result must explain itself")


def check_transition(current: CandidateState, target: CandidateState) -> None:
    """Raise if the move is not one the machine allows."""
    from serving_evolve.errors import IllegalTransition

    allowed = LEGAL_TRANSITIONS[CandidateState(current)]
    if CandidateState(target) not in allowed:
        raise IllegalTransition(str(current.value), str(target.value),
                                sorted(s.value for s in allowed))
