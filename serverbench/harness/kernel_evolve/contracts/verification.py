"""The verification plan: tolerance, boundaries, negative controls.

Three ideas do most of the work here.

**Tolerance is dtype-aware, and stated before the numbers arrive.** A bf16 kernel
that matches an fp32 reference to 1e-6 is not more correct, it is misconfigured;
a bf16 kernel checked at 1e-1 is not checked. The tolerance is part of the frozen
plan so it cannot be widened once a candidate misses it.

**Boundaries are where kernels are wrong.** A tiled kernel is right in the middle
of its domain by construction and wrong at the edge by omission — the last
partial tile, the unaligned head dimension, the sequence length one past a block.
The plan enumerates those points rather than sampling the interior.

**A test that cannot fail proves nothing.** Negative controls are deliberately
broken variants — a dropped normalisation, a stale KV offset, a truncated
vocabulary — and the plan is only qualified if the checks *kill* them. This is
the single most valuable rung, because it is the one that detects a check which
has quietly stopped checking.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import ClassVar

from serving_evolve.contracts.base import Contract

from .capability import DType

#: What each dtype can actually represent, as a relative tolerance. These are the
#: floors: a plan may be stricter, never looser, and the contract enforces it.
DTYPE_FLOOR: dict[DType, float] = {
    DType.FP32: 1e-5,
    DType.FP16: 1e-3,
    DType.BF16: 1e-2,
    DType.FP8_E4M3: 6e-2,
    DType.FP8_E5M2: 1.2e-1,
    DType.INT8: 1e-2,
    DType.INT4: 1e-1,
}


class MutantClass(str, enum.Enum):
    """Semantic corruptions a correctness check must be able to detect.

    Each names a real defect seen in real kernels, not an abstract fault model.
    """

    WRONG_MASK = "wrong_mask"
    MISSING_NORMALIZATION = "missing_normalization"
    STALE_KV_OFFSET = "stale_kv_offset"
    TRUNCATED_VOCABULARY = "truncated_vocabulary"
    WRONG_REDUCTION = "wrong_reduction"
    PARTIAL_WRITE = "partial_write"
    SILENT_FALLBACK = "silent_fallback"
    INACCURATE_TIMING = "inaccurate_timing"
    WRONG_POSITION = "wrong_position"
    DROPPED_GQA_GROUP = "dropped_gqa_group"


#: The classes every kernel plan must kill. A plan that omits one of these has a
#: hole exactly the shape of a defect that has actually shipped.
REQUIRED_MUTANTS: frozenset[MutantClass] = frozenset({
    MutantClass.WRONG_MASK,
    MutantClass.MISSING_NORMALIZATION,
    MutantClass.STALE_KV_OFFSET,
    MutantClass.WRONG_REDUCTION,
    MutantClass.PARTIAL_WRITE,
    MutantClass.SILENT_FALLBACK,
    MutantClass.INACCURATE_TIMING,
})


@dataclasses.dataclass(frozen=True)
class Tolerance(Contract):
    """How close counts as equal, and how many elements may miss."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "kernel_tolerance"

    dtype: DType = DType.BF16
    rtol: float = 1e-2
    atol: float = 1e-2
    #: Fraction of elements that must land inside (rtol, atol). Below 1.0 because
    #: a different-but-valid reduction order genuinely flips a rare element; a
    #: value that lets a systematically wrong kernel through is not a tolerance.
    matched_ratio: float = 0.99
    require_finite: bool = True
    require_deterministic: bool = True

    def validate(self) -> None:
        floor = DTYPE_FLOOR.get(DType(self.dtype))
        self.require(self.rtol > 0 and self.atol > 0, "tolerances are positive")
        if floor is not None:
            self.require(
                self.rtol <= floor * 10,
                f"rtol {self.rtol} is more than 10x the representable floor {floor} for "
                f"{DType(self.dtype).value}; a tolerance this wide accepts a kernel that "
                f"is computing something else")
        self.require(0.5 < self.matched_ratio <= 1.0,
                     f"matched ratio {self.matched_ratio} is not a meaningful fraction")
        self.require(self.require_finite,
                     "a plan that tolerates NaN or Inf output is not checking correctness")

    @classmethod
    def for_dtype(cls, dtype: DType) -> Tolerance:
        """The default tolerance for a dtype: its floor, on both scales."""
        floor = DTYPE_FLOOR.get(DType(dtype), 1e-2)
        return cls(dtype=dtype, rtol=floor, atol=floor)


@dataclasses.dataclass(frozen=True)
class NegativeControl(Contract):
    """A deliberately broken variant, and the rung expected to catch it."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "kernel_negative_control"

    control_id: str = ""
    mutant: MutantClass = MutantClass.WRONG_MASK
    description: str = ""
    #: The rung that must return FAIL for this control. Naming it means a control
    #: that is caught by the wrong check is reported rather than counted as a win.
    expected_killer: str = ""
    #: Set when the control is bound to a frozen source tree. A control generated
    #: against one revision and applied to another is INVALID, not a result.
    bound_source_digest: str = ""

    def validate(self) -> None:
        self.require(bool(self.control_id), "a control needs an id")
        self.require(bool(self.description), "a control must say what it breaks")
        self.require(bool(self.expected_killer),
                     "a control must name the rung expected to kill it")


@dataclasses.dataclass(frozen=True)
class BoundaryMatrix(Contract):
    """The shape/dtype/layout points a candidate is required to be right at."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "kernel_boundary_matrix"

    points: list[dict[str, int]] = dataclasses.field(default_factory=list)
    dtypes: list[DType] = dataclasses.field(default_factory=list)
    layouts: list[str] = dataclasses.field(default_factory=list)
    #: Inputs the kernel must *refuse*: wrong dtype, wrong device, wrong rank.
    #: A kernel that quietly computes something for an input it does not support
    #: is worse than one that raises.
    rejection_cases: list[str] = dataclasses.field(default_factory=list)

    def validate(self) -> None:
        self.require(bool(self.points), "a boundary matrix with no point tests nothing")
        self.require(bool(self.dtypes), "a boundary matrix must name its dtypes")
        self.require(bool(self.rejection_cases),
                     "a boundary matrix must include inputs the kernel has to refuse; "
                     "silent computation on an unsupported input is a correctness bug")

    @property
    def size(self) -> int:
        return len(self.points) * len(self.dtypes) * max(len(self.layouts), 1)


@dataclasses.dataclass(frozen=True)
class VerificationPlan(Contract):
    """Everything the ladder needs, frozen before any candidate result is seen."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "kernel_verification_plan"

    plan_id: str = ""
    tolerance: Tolerance | None = None
    boundaries: BoundaryMatrix | None = None
    controls: list[NegativeControl] = dataclasses.field(default_factory=list)
    #: The corruption classes this plan is required to kill. Defaults to every
    #: class that has shipped as a real defect. A plan may narrow it — an RMS norm
    #: has no mask to get wrong and no KV offset to go stale — but narrowing is an
    #: explicit act with a stated reason, never a silent omission, because a plan
    #: that quietly drops a class reports a clean sheet over a hole.
    required_mutants: list[MutantClass] = dataclasses.field(
        default_factory=lambda: sorted(REQUIRED_MUTANTS, key=lambda m: m.value))
    narrowing_rationale: str = ""
    #: Whether the kernel promises run-to-run identical output. Most do; a kernel
    #: using atomics over floats does not, and must say so rather than fail a
    #: determinism check it never claimed to satisfy.
    determinism_promised: bool = True
    #: Seeds for input generation. Fixed, so a failure is reproducible.
    seeds: list[int] = dataclasses.field(default_factory=lambda: [0, 1, 2])
    #: Whether an integration run against the real serving path is required before
    #: acceptance. For anything in the hot path the answer is yes: an operator
    #: that is correct in isolation and wrong in the model is the common case.
    require_serving_integration: bool = True

    def validate(self) -> None:
        self.require(bool(self.plan_id), "a verification plan needs an id")
        self.require(self.tolerance is not None, "a plan needs a tolerance")
        self.require(self.boundaries is not None, "a plan needs a boundary matrix")
        self.require(bool(self.seeds), "a plan needs at least one seed")
        required = {MutantClass(m) for m in self.required_mutants}
        dropped = REQUIRED_MUTANTS - required
        self.require(
            not dropped or bool(self.narrowing_rationale),
            f"the plan drops corruption class(es) "
            f"{sorted(m.value for m in dropped)} without saying why; a class that "
            f"cannot occur for this operator is a fine reason and an unstated one is "
            f"a hole")
        missing = required - {MutantClass(c.mutant) for c in self.controls}
        self.require(
            not missing,
            f"the plan does not test for {sorted(m.value for m in missing)}; a check "
            f"battery that cannot kill these has a hole shaped like a defect that has "
            f"already shipped")

    def controls_for(self, rung: str) -> list[NegativeControl]:
        return [c for c in self.controls if c.expected_killer == rung]
