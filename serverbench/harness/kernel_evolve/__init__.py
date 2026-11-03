"""``kernel_evolve`` — evolving one composable GPU kernel, safely.

A sibling of :mod:`serving_evolve`, not a fork of it: it reuses that package's
contracts, gate verdicts and acceptance policy, and adds only what is genuinely
kernel-shaped:

* **what a kernel claims** — capability, shape envelope, and the model semantics
  (QK normalisation, GQA grouping, masks, positions, cache layout) that decide
  whether a substitution is safe at all;
* **the verification ladder** — eleven rungs, cheapest first, in
  :mod:`kernel_evolve.verification`;
* **the check on the checks** — known-broken variants and the kill matrix, in
  :mod:`kernel_evolve.verification`;
* **a build that cannot see the harness** — :mod:`kernel_evolve.sandbox`;
* **the catalog** a candidate may dispatch to — :mod:`kernel_evolve.shapes`;
* **the proof the megakernel is not here** — :mod:`kernel_evolve.verification`.

Two boundaries hold the design together.

**The serving runtime must never import this package.** Evolution control code in
a serving hot path is a defect: it makes the thing being measured depend on the
thing measuring it. The dependency runs one way, and
:func:`kernel_evolve.verification.check_imports` is how that is checked rather than
asserted.

**A microbenchmark may reject and may never accept.** It is stated in the ladder,
enforced by ``can_accept=False`` on the rung, and worth repeating here because it
is the rule most often argued with: a kernel three times faster on an operator
holding four per cent of the critical path cannot move the served system by more
than about three per cent, and if the operator was not on the critical path it
cannot move it at all.

Scope: **ordinary composable kernels only**. A single persistent kernel owning
the model iteration is a different research question and is deliberately
excluded; see :mod:`kernel_evolve.verification` for the gate that keeps it out.

megakernel-exclusion-scope: this file documents the exclusion
"""

from __future__ import annotations

from . import campaign, sandbox, shapes, verification
from .contracts import (
    BoundaryMatrix,
    BuildManifest,
    CandidateState,
    DType,
    KernelBudget,
    KernelCandidate,
    KernelCandidateResult,
    KernelCapability,
    KernelHypothesis,
    KernelStrategy,
    KernelTaskSpec,
    Layout,
    MaskConvention,
    MicrobenchmarkPlan,
    MicrobenchmarkResult,
    MutantClass,
    NegativeControl,
    Operator,
    PositionConvention,
    ReferenceImplementation,
    ReferenceKind,
    ServingPhase,
    ShapeAxis,
    ShapeEnvelope,
    SourceFile,
    TimingSample,
    Tolerance,
    VerificationPlan,
)
from .verification import RUNGS, LadderPlan, default_policy, run_ladder

__version__ = "1.0.0"

__all__ = [
    "RUNGS",
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
    "LadderPlan",
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
    "__version__",
    "campaign",
    "default_policy",
    "run_ladder",
    "sandbox",
    "shapes",
    "verification",
]
