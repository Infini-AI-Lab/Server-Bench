"""Typed, versioned contracts for kernel evolution.

They extend the serving harness's :class:`serving_evolve.contracts.base.Contract`
rather than restating it, so a kernel contract hashes, serialises and validates
exactly the way every other contract does.
"""

from .budget import KernelBudget
from .candidate import (
    LEGAL_TRANSITIONS,
    BuildManifest,
    CandidateState,
    KernelCandidate,
    KernelCandidateResult,
    KernelHypothesis,
    SourceFile,
    check_transition,
)
from .capability import (
    DType,
    KernelCapability,
    Layout,
    MaskConvention,
    Operator,
    PositionConvention,
    ServingPhase,
    ShapeAxis,
    ShapeEnvelope,
)
from .measure import MicrobenchmarkPlan, MicrobenchmarkResult, TimingSample
from .task import (
    KernelStrategy,
    KernelTaskSpec,
    ReferenceImplementation,
    ReferenceKind,
)
from .verification import (
    DTYPE_FLOOR,
    REQUIRED_MUTANTS,
    BoundaryMatrix,
    MutantClass,
    NegativeControl,
    Tolerance,
    VerificationPlan,
)

__all__ = [
    "DTYPE_FLOOR",
    "LEGAL_TRANSITIONS",
    "REQUIRED_MUTANTS",
    "BoundaryMatrix",
    "BuildManifest",
    "CandidateState",
    "DType",
    "KernelBudget",
    "KernelCandidate",
    "KernelCandidateResult",
    "KernelCapability",
    "KernelHypothesis",
    "KernelStrategy",
    "KernelTaskSpec",
    "Layout",
    "MaskConvention",
    "MicrobenchmarkPlan",
    "MicrobenchmarkResult",
    "MutantClass",
    "NegativeControl",
    "Operator",
    "PositionConvention",
    "ReferenceImplementation",
    "ReferenceKind",
    "ServingPhase",
    "ShapeAxis",
    "ShapeEnvelope",
    "SourceFile",
    "TimingSample",
    "Tolerance",
    "VerificationPlan",
    "check_transition",
]
