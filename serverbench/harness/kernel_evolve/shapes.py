"""shapes -- what a candidate may dispatch, over which input shapes.

Two sections, each opening with its own banner and description: the kernel
catalog, and the stratified shape matrix.
"""

from __future__ import annotations

import dataclasses
import enum
import statistics
from collections.abc import Sequence
from typing import Any, ClassVar

from serving_evolve.contracts.base import Contract

from .contracts.capability import DType, KernelCapability, Layout, ShapeEnvelope
from .contracts.task import KernelStrategy
from .contracts.verification import Tolerance

# ==========================================================================
# The kernel catalog: what a candidate is allowed to dispatch to.
# ==========================================================================
"""The kernel catalog: what a candidate is allowed to dispatch to.

The catalog answers one question for each strategy, and they are different
questions.

Under ``provided_sota`` it is the **whole permitted surface**. The agent may call
these entries and may not write kernel source, so the catalog has to carry
everything needed to use an entry correctly and nothing that would let it be
copied: capability, ABI, tolerance, fallback, and the conditions under which the
entry must refuse. The reference implementations behind the entries are withheld,
because handing them over turns the arm into a slower version of the other one.

Under ``evolve`` it is the **frozen starting point**: what the hot path uses
today, what a candidate is being compared against, and which vendor primitives
are approved to build on.

This is not the same object as :class:`serving_evolve.kernels.contracts.KernelCatalog`,
and the two are deliberately separate. That one records **what exists in the
world** — a researched pointer with provenance, a licence and a claim, written
outside the sandbox and frozen before the run. This one records **what this run
may dispatch to**: the subset that has been qualified on this hardware for this
capability, with the ABI, the tolerance and the fallback a caller needs. A
researched pointer is a lead; a catalog entry here is a permission.

One rule is enforced rather than documented: a catalog entry records a *claim
with a source*, never a measurement. A number
measured in this repository belongs to a run, under a frozen plan, with a device
receipt. Letting a catalog carry one lets a vendor's marketing figure be quoted
later as this project's result.
"""
class EvidenceLevel(str, enum.Enum):
    """How good the evidence behind a claim is, from a catalog's point of view."""

    VENDOR_CLAIM = "vendor_claim"
    THIRD_PARTY_CLAIM = "third_party_claim"
    PUBLISHED_BENCHMARK = "published_benchmark"
    #: Reserved and refused inside a catalog: a measurement taken here belongs to a
    #: run under a frozen plan, not to a pointer.
    MEASURED_HERE = "measured_here"


class QualificationState(str, enum.Enum):
    """How far an entry has got through this project's own checks."""

    DECLARED = "declared"
    BUILDS = "builds"
    PARITY_CHECKED = "parity_checked"
    #: Passed the full ladder on this hardware for this capability.
    QUALIFIED = "qualified"
    REJECTED = "rejected"


@dataclasses.dataclass(frozen=True)
class CatalogEntry(Contract):
    """One kernel a candidate may dispatch to, with everything needed to use it."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "kernel_evolve_catalog_entry"

    entry_id: str = ""
    capability: KernelCapability | None = None
    #: The package and symbol, as a caller would write it.
    package: str = ""
    symbol: str = ""
    package_version: str = ""
    package_digest: str = ""
    licence: str = ""
    #: Where the claim comes from, and when it was read. An unsourced claim is a
    #: rumour, and a claim with no date cannot be re-checked.
    source_url: str = ""
    retrieved: str = ""
    evidence_level: EvidenceLevel = EvidenceLevel.VENDOR_CLAIM
    claim: str = ""
    qualification: QualificationState = QualificationState.DECLARED
    tolerance: Tolerance | None = None
    #: Where the parity fixtures for this entry live, by digest.
    parity_fixture_digest: str = ""
    #: What to fall back to, and how the fallback announces itself.
    fallback_entry_id: str = ""
    fallback_telemetry: str = ""

    def validate(self) -> None:
        self.require(bool(self.entry_id), "a catalog entry needs an id")
        self.require(self.capability is not None,
                     "a catalog entry needs a capability; without one nothing decides "
                     "whether it may be substituted here")
        self.require(bool(self.package) and bool(self.symbol),
                     "a catalog entry must name the package and symbol to call")
        self.require(bool(self.licence),
                     "a catalog entry needs a licence, recorded before anything depends "
                     "on it")
        self.require(bool(self.source_url) and bool(self.retrieved),
                     "a claim needs a source and the date it was read")
        self.require(
            EvidenceLevel(self.evidence_level) is not EvidenceLevel.MEASURED_HERE,
            "MEASURED_HERE is not an evidence level a catalog may use: a measurement "
            "taken here belongs to a run under a frozen plan with a device receipt, "
            "and putting one in a pointer is how a vendor's number becomes ours")
        self.require(bool(self.fallback_entry_id) or bool(self.fallback_telemetry),
                     "an entry must declare its fallback and how the fallback announces "
                     "itself; an undeclared fallback is a hidden one")

    @property
    def qualified(self) -> bool:
        return QualificationState(self.qualification) is QualificationState.QUALIFIED


@dataclasses.dataclass(frozen=True)
class KernelCatalog(Contract):
    """The catalog for one model, hardware and strategy."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "kernel_evolve_catalog"

    catalog_id: str = ""
    model_id: str = ""
    architecture: str = ""
    strategy: KernelStrategy = KernelStrategy.EVOLVE
    entries: list[CatalogEntry] = dataclasses.field(default_factory=list)
    #: Vendor primitives a candidate may build on under ``evolve`` — cuBLAS, a
    #: reduction library. Distinct from ``entries``, which are whole operators.
    approved_primitives: list[str] = dataclasses.field(default_factory=list)
    notes: str = ""

    def validate(self) -> None:
        self.require(bool(self.catalog_id), "a catalog needs an id")
        self.require(bool(self.model_id) and bool(self.architecture),
                     "a catalog is for one model on one architecture; both are required "
                     "because 'the fastest attention kernel' is not a question with a "
                     "model-independent answer")
        ids = [e.entry_id for e in self.entries]
        self.require(len(ids) == len(set(ids)), f"duplicate entry id(s) in {ids}")
        known = set(ids)
        dangling = sorted({e.fallback_entry_id for e in self.entries
                           if e.fallback_entry_id and e.fallback_entry_id not in known})
        self.require(not dangling,
                     f"entry fallback(s) name absent entries {dangling}; a fallback that "
                     f"is not in the catalog cannot be dispatched to")
        if KernelStrategy(self.strategy) is KernelStrategy.PROVIDED_SOTA:
            unqualified = [e.entry_id for e in self.entries if not e.qualified]
            self.require(
                not unqualified,
                f"under provided_sota every entry is the whole permitted surface, so an "
                f"unqualified one has no fallback position: {unqualified}")
        for entry in self.entries:
            cap = entry.capability
            if cap is not None:
                self.require(
                    cap.supports_architecture(self.architecture),
                    f"entry {entry.entry_id} is not qualified on {self.architecture}; it "
                    f"names {cap.architectures}")

    def for_operator(self, operator: str) -> list[CatalogEntry]:
        return [e for e in self.entries
                if e.capability is not None
                and str(e.capability.operator.value) == str(operator)]

    def qualified_entries(self) -> list[CatalogEntry]:
        return [e for e in self.entries if e.qualified]

    def brief(self) -> str:
        """What the agent is told the catalog contains. No source, no fixtures."""
        lines = [f"# Kernel catalog {self.catalog_id}",
                 f"model {self.model_id} on {self.architecture}, "
                 f"strategy {KernelStrategy(self.strategy).value}", ""]
        if KernelStrategy(self.strategy) is KernelStrategy.PROVIDED_SOTA:
            lines += ["You may dispatch to these entries. You may not write or modify "
                      "kernel source, and the implementations behind these entries are "
                      "not available to you.", ""]
        else:
            lines += ["This is the frozen starting point your candidates are compared "
                      "against. You may write kernel source; you may build on the "
                      "approved primitives listed at the end.", ""]
        for entry in sorted(self.entries, key=lambda e: e.entry_id):
            cap = entry.capability
            op = cap.operator.value if cap is not None else "?"
            lines.append(f"- **{entry.entry_id}** ({op}) — `{entry.package}.{entry.symbol}` "
                         f"v{entry.package_version}, {entry.licence}")
            if cap is not None and cap.unsupported:
                lines.append(f"    refuses: {'; '.join(cap.unsupported)}")
            if entry.fallback_entry_id:
                lines.append(f"    falls back to `{entry.fallback_entry_id}`, announced by "
                             f"{entry.fallback_telemetry or 'the entry itself'}")
        if self.approved_primitives:
            lines += ["", "Approved primitives you may build on: "
                      + ", ".join(sorted(self.approved_primitives))]
        return "\n".join(lines) + "\n"


# ==========================================================================
# The shape matrix: stratified from the workload, not enumerated from the axes.
# ==========================================================================
"""The shape matrix: stratified from the workload, not enumerated from the axes.

A Cartesian product over every axis is the obvious way to build a verification
matrix and the wrong one. It is exponential, most of its points are interior
duplicates that a tiled kernel is correct at by construction, and the cost pushes
people to run it rarely — so the matrix that was meant to be thorough ends up
being skipped.

What actually catches defects is *boundaries* and *what the workload produces*.
This module builds a matrix from four strata:

**Frequency.** The minimum, median, p95 and maximum shapes the task's own trace
generates. A kernel that is fast at the median and wrong at p95 is wrong where it
matters and passes an unweighted matrix.

**Boundaries.** Alignment edges, the value one past a tile, the last partial tile.
A tiled kernel is right in the middle of its domain by construction and wrong at
the edge by omission.

**Semantics.** Mask, position, cache-offset, GQA grouping, dtype and layout
combinations that change what the operator *means* rather than only its size.

**Regression.** Shapes that have failed before. They stay in the matrix forever;
a defect that recurs after being fixed is the cheapest one to catch.

The result is frozen prospectively and hashed. A matrix chosen after seeing which
shapes a candidate happens to pass is not a matrix, it is a description.
"""
@dataclasses.dataclass(frozen=True)
class ShapePoint(Contract):
    """One point in the matrix, and why it is there."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "kernel_shape_point"

    axes: dict[str, int] = dataclasses.field(default_factory=dict)
    dtype: DType = DType.BF16
    layout: Layout = Layout.CONTIGUOUS
    #: frequency | boundary | semantic | regression
    stratum: str = "frequency"
    reason: str = ""

    def validate(self) -> None:
        self.require(bool(self.axes), "a shape point needs at least one axis")
        self.require(self.stratum in ("frequency", "boundary", "semantic",
                                      "regression"),
                     f"unknown stratum {self.stratum!r}")
        self.require(bool(self.reason),
                     "a shape point must say why it is in the matrix; a point "
                     "nobody can justify is one nobody will defend when it is slow")

    @property
    def key(self) -> tuple:
        return (tuple(sorted(self.axes.items())), str(self.dtype.value),
                str(self.layout.value))


@dataclasses.dataclass(frozen=True)
class ShapeMatrix(Contract):
    """A frozen, stratified verification matrix."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "kernel_shape_matrix"

    matrix_id: str = ""
    points: list[ShapePoint] = dataclasses.field(default_factory=list)
    #: Inputs the kernel must refuse rather than quietly compute.
    rejection_cases: list[str] = dataclasses.field(default_factory=list)
    #: Where the frequency stratum came from, so it can be re-derived.
    workload_source: str = ""

    def validate(self) -> None:
        self.require(bool(self.matrix_id), "a matrix needs an id")
        self.require(bool(self.points), "an empty matrix tests nothing")
        strata = {p.stratum for p in self.points}
        self.require("boundary" in strata,
                     "a matrix with no boundary point tests the interior, where a "
                     "tiled kernel is correct by construction")
        self.require(bool(self.rejection_cases),
                     "a matrix must name inputs the kernel has to refuse; silent "
                     "computation on an unsupported input is a correctness bug")

    def by_stratum(self, stratum: str) -> list[ShapePoint]:
        return [p for p in self.points if p.stratum == stratum]

    @property
    def size(self) -> int:
        return len(self.points)

    def summary(self) -> str:
        counts = {s: len(self.by_stratum(s))
                  for s in ("frequency", "boundary", "semantic", "regression")}
        return (f"{self.size} point(s): "
                + ", ".join(f"{n} {s}" for s, n in counts.items() if n)
                + f"; {len(self.rejection_cases)} rejection case(s)")


def _percentile(values: Sequence[int], q: float) -> int:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * len(ordered) + 0.5)) - 1))
    return ordered[index]


def from_workload(matrix_id: str, envelope: ShapeEnvelope, *,
                  observed: dict[str, Sequence[int]] | None = None,
                  semantics: Sequence[dict[str, Any]] = (),
                  regressions: Sequence[dict[str, Any]] = (),
                  rejection_cases: Sequence[str] = (),
                  workload_source: str = "") -> ShapeMatrix:
    """Build a stratified matrix from an envelope and, where available, real shapes.

    ``observed`` maps an axis name to the values a task's trace actually produces.
    Without it the frequency stratum falls back to the envelope's declared range,
    which is honest but weaker — the envelope says what is *allowed*, and only the
    trace says what is *served*.
    """
    points: list[ShapePoint] = []
    seen: set[tuple] = set()

    def add(point: ShapePoint) -> None:
        if point.key not in seen:
            seen.add(point.key)
            points.append(point)

    dtypes = list(envelope.dtypes) or [DType.BF16]
    layouts = list(envelope.layouts) or [Layout.CONTIGUOUS]
    primary_dtype, primary_layout = dtypes[0], layouts[0]
    base = {a.name: a.minimum for a in envelope.axes}

    # -- frequency: what the workload actually produces -----------------------
    for axis in envelope.axes:
        values = list((observed or {}).get(axis.name, ()))
        if values:
            picks = [
                (min(values), "the smallest shape the workload produces"),
                (int(statistics.median(values)), "the median shape"),
                (_percentile(values, 0.95), "the p95 shape, where a kernel that is "
                                            "fast at the median often is not"),
                (max(values), "the largest shape the workload produces"),
            ]
        else:
            picks = [
                (axis.minimum, "the envelope's declared minimum (no trace supplied)"),
                ((axis.minimum + axis.maximum) // 2,
                 "the envelope's midpoint (no trace supplied)"),
                (axis.maximum, "the envelope's declared maximum (no trace supplied)"),
            ]
        for value, reason in picks:
            add(ShapePoint(axes={**base, axis.name: value}, dtype=primary_dtype,
                           layout=primary_layout, stratum="frequency",
                           reason=f"{axis.name}: {reason}"))

    # -- boundary: where tiling and predication go wrong ----------------------
    for axis in envelope.axes:
        for value in axis.boundary_values():
            add(ShapePoint(axes={**base, axis.name: value}, dtype=primary_dtype,
                           layout=primary_layout, stratum="boundary",
                           reason=f"{axis.name}={value}: an edge of the declared "
                                  f"envelope or its alignment"))
        if axis.alignment > 1:
            for candidate in (axis.alignment + 1, axis.alignment - 1):
                if axis.minimum <= candidate <= axis.maximum:
                    add(ShapePoint(
                        axes={**base, axis.name: candidate}, dtype=primary_dtype,
                        layout=primary_layout, stratum="boundary",
                        reason=f"{axis.name}={candidate}: one either side of the "
                               f"{axis.alignment}-element tile, which is where a "
                               f"partial-tile bug lives"))

    # -- semantic: dtype and layout change meaning, not only size -------------
    for dtype in dtypes[1:]:
        add(ShapePoint(axes=dict(base), dtype=dtype, layout=primary_layout,
                       stratum="semantic",
                       reason=f"dtype {dtype.value}: a different tolerance and a "
                              f"different accumulation"))
    for layout in layouts[1:]:
        add(ShapePoint(axes=dict(base), dtype=primary_dtype, layout=layout,
                       stratum="semantic",
                       reason=f"layout {layout.value}: a kernel assuming contiguity "
                              f"reads the wrong elements and stays plausible"))
    for extra in semantics:
        add(ShapePoint(axes={**base, **extra.get("axes", {})},
                       dtype=extra.get("dtype", primary_dtype),
                       layout=extra.get("layout", primary_layout),
                       stratum="semantic",
                       reason=extra.get("reason", "declared semantic boundary")))

    # -- regression: it failed once and stays forever -------------------------
    for failure in regressions:
        add(ShapePoint(axes={**base, **failure.get("axes", {})},
                       dtype=failure.get("dtype", primary_dtype),
                       layout=failure.get("layout", primary_layout),
                       stratum="regression",
                       reason=failure.get("reason",
                                          "this shape failed before and is kept "
                                          "permanently")))

    return ShapeMatrix(matrix_id=matrix_id, points=points,
                       rejection_cases=list(rejection_cases) or [
                           "wrong dtype", "wrong device", "wrong rank"],
                       workload_source=workload_source or
                       ("the envelope alone; no trace shapes were supplied, so the "
                        "frequency stratum describes what is allowed rather than "
                        "what is served"))


def release_matrix(matrix_id: str, envelope: ShapeEnvelope, *,
                   rejection_cases: Sequence[str] = ()) -> ShapeMatrix:
    """The exhaustive matrix, for a release only.

    The full cross-product exists and has a place: a kernel being promoted to an
    accepted deployment is checked everywhere its envelope claims. What it must
    not be is the per-draft matrix — that is the cost that makes people skip it.
    """
    points: list[ShapePoint] = []
    for point in envelope.boundary_points():
        for dtype in envelope.dtypes:
            for layout in envelope.layouts:
                points.append(ShapePoint(
                    axes=point, dtype=dtype, layout=layout, stratum="boundary",
                    reason="release: the full declared supported envelope"))
    return ShapeMatrix(matrix_id=matrix_id, points=points,
                       rejection_cases=list(rejection_cases) or [
                           "wrong dtype", "wrong device", "wrong rank"],
                       workload_source="the full declared envelope, for release only")
