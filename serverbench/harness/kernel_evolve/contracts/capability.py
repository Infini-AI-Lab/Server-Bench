"""What a kernel is asked to compute, and where it is allowed to be used.

A *capability* is the honest statement of a kernel's competence: which operator,
in which serving phase, over which shapes, dtypes and layouts, on which hardware,
and — the part that is usually left implicit and then bites — what it explicitly
does **not** support.

The reason this is a contract rather than a docstring is a specific failure. A
fused attention kernel written for a model that does not normalise Q and K will
run on Qwen3, produce fluent output, and silently omit an operation the model
needs. Nothing in a latency measurement can see that. So the semantics that make
a kernel substitutable — QK normalisation, GQA grouping, mask convention,
position convention, KV cache layout — are declared here, matched against the
model, and refused when they disagree.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import ClassVar

from serving_evolve.contracts.base import Contract


class Operator(str, enum.Enum):
    """Operators an ordinary decoder serving path is built from.

    Deliberately composable units. A single persistent kernel that owns the whole
    model iteration is **not** on this list, and :mod:`kernel_evolve.verification`
    exists to keep it off.
    """

    EMBEDDING = "embedding"
    RMS_NORM = "rms_norm"
    QK_NORM = "qk_norm"
    ROPE = "rope"
    QKV_PROJECTION = "qkv_projection"
    ATTENTION_PREFILL = "attention_prefill"
    ATTENTION_DECODE = "attention_decode"
    KV_CACHE_APPEND = "kv_cache_append"
    OUTPUT_PROJECTION = "output_projection"
    GATED_MLP = "gated_mlp"
    SILU_MUL = "silu_mul"
    LM_HEAD = "lm_head"
    SAMPLING = "sampling"
    TOPK = "topk"
    MOE_ROUTER = "moe_router"
    MOE_GROUPED_GEMM = "moe_grouped_gemm"
    QUANTIZE = "quantize"
    DEQUANTIZE = "dequantize"
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"


class ServingPhase(str, enum.Enum):
    """Where in a serving step the operator runs.

    Prefill and decode give the same operator utterly different shapes: a decode
    GEMM is a skinny GEMV whose grid does not fill the device, a prefill GEMM is
    compute-bound. A kernel that wins at one routinely loses at the other, so the
    phase is part of the capability and never averaged away.
    """

    PREFILL = "prefill"
    DECODE = "decode"
    BOTH = "both"
    OFFLINE_BATCH = "offline_batch"


class DType(str, enum.Enum):
    BF16 = "bfloat16"
    FP16 = "float16"
    FP32 = "float32"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    INT8 = "int8"
    INT4 = "int4"


class Layout(str, enum.Enum):
    """Memory layout the kernel accepts for its principal tensors."""

    CONTIGUOUS = "contiguous"
    #: A view with a non-unit innermost stride, e.g. a slice of a fused QKV
    #: projection. Kernels that assume contiguity read the wrong elements here and
    #: are usually still numerically plausible, which is what makes it worth a
    #: separate declaration.
    STRIDED = "strided"
    PAGED = "paged"
    TRANSPOSED = "transposed"


class MaskConvention(str, enum.Enum):
    NONE = "none"
    CAUSAL = "causal"
    SLIDING_WINDOW = "sliding_window"
    CUSTOM = "custom"


class PositionConvention(str, enum.Enum):
    NONE = "none"
    ROPE_NEOX = "rope_neox"
    ROPE_GPTJ = "rope_gptj"
    ALIBI = "alibi"


@dataclasses.dataclass(frozen=True)
class ShapeAxis(Contract):
    """One named dimension of the workload, and the range the kernel claims."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "kernel_shape_axis"

    name: str = ""
    minimum: int = 0
    maximum: int = 0
    #: Values the kernel must handle exactly — tile boundaries, and the values one
    #: off them, which is where a tiling bug lives.
    required_values: list[int] = dataclasses.field(default_factory=list)
    alignment: int = 1

    def validate(self) -> None:
        self.require(bool(self.name), "an axis needs a name")
        self.require(self.minimum >= 0, "an axis extent cannot be negative")
        self.require(self.maximum >= self.minimum,
                     f"axis {self.name}: maximum {self.maximum} is below minimum "
                     f"{self.minimum}")
        self.require(self.alignment >= 1, "alignment is at least 1")
        outside = [v for v in self.required_values
                   if not (self.minimum <= v <= self.maximum)]
        self.require(not outside,
                     f"axis {self.name}: required value(s) {outside} lie outside "
                     f"[{self.minimum}, {self.maximum}]")

    def boundary_values(self) -> list[int]:
        """The values worth testing: the ends, the alignment edges, the declared ones.

        A kernel is wrong at boundaries or nowhere. This returns the ends, the
        first value that is not a whole number of tiles, and anything the author
        declared, deduplicated and sorted.
        """
        candidates = {self.minimum, self.maximum, *self.required_values}
        if self.alignment > 1:
            for base in (self.minimum, self.maximum):
                for delta in (-1, 1):
                    v = base + delta
                    if self.minimum <= v <= self.maximum:
                        candidates.add(v)
            unaligned = self.minimum + 1
            if self.minimum < unaligned <= self.maximum:
                candidates.add(unaligned)
        return sorted(v for v in candidates if v >= 0)


@dataclasses.dataclass(frozen=True)
class ShapeEnvelope(Contract):
    """The full space of inputs a kernel claims to handle, and its edges."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "kernel_shape_envelope"

    axes: list[ShapeAxis] = dataclasses.field(default_factory=list)
    dtypes: list[DType] = dataclasses.field(default_factory=list)
    layouts: list[Layout] = dataclasses.field(default_factory=lambda: [Layout.CONTIGUOUS])

    def validate(self) -> None:
        self.require(bool(self.axes), "an envelope with no axis describes nothing")
        self.require(bool(self.dtypes), "an envelope must name at least one dtype")
        self.require(bool(self.layouts), "an envelope must name at least one layout")
        names = [a.name for a in self.axes]
        self.require(len(names) == len(set(names)),
                     f"duplicate axis name(s) in {names}")

    def axis(self, name: str) -> ShapeAxis | None:
        return next((a for a in self.axes if a.name == name), None)

    def boundary_points(self) -> list[dict[str, int]]:
        """One point per axis boundary, holding the other axes at their minimum.

        The full cross-product is exponential and mostly redundant; what catches
        tiling and predication bugs is varying one axis to its edge at a time.
        """
        base = {a.name: a.minimum for a in self.axes}
        points: list[dict[str, int]] = []
        for a in self.axes:
            for v in a.boundary_values():
                point = dict(base)
                point[a.name] = v
                if point not in points:
                    points.append(point)
        return points


@dataclasses.dataclass(frozen=True)
class KernelCapability(Contract):
    """Everything that decides whether a kernel may be substituted here.

    ``unsupported`` is required to be non-empty. A kernel that claims no
    limitation has not been characterised; the field exists to make the author
    say where the edge is, because the alternative is discovering it in
    production as fluent, wrong output.
    """

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "kernel_capability"

    capability_id: str = ""
    operator: Operator = Operator.RMS_NORM
    phase: ServingPhase = ServingPhase.BOTH
    envelope: ShapeEnvelope | None = None
    #: Compute capabilities the kernel is qualified on, as ``sm90``, ``sm100``.
    #: A capability is never assumed to carry to an arch nobody measured.
    architectures: list[str] = dataclasses.field(default_factory=list)
    #: Model-semantic switches. Wrong values here are the silent-corruption class.
    qk_norm: bool = False
    gqa_group_size: int = 1
    mask: MaskConvention = MaskConvention.NONE
    positions: PositionConvention = PositionConvention.NONE
    kv_cache_layout: Layout = Layout.CONTIGUOUS
    tensor_parallel_safe: bool = False
    disaggregation_safe: bool = False
    #: Conditions under which the kernel must refuse rather than compute.
    unsupported: list[str] = dataclasses.field(default_factory=list)
    #: The backend to use when a condition above is hit. An undeclared fallback is
    #: the hidden-fallback defect; there is no "it just works" answer.
    fallback: str = ""

    def validate(self) -> None:
        self.require(bool(self.capability_id), "a capability needs an id")
        self.require(self.envelope is not None,
                     "a capability without a shape envelope claims everything")
        self.require(bool(self.architectures),
                     "a capability must name the architecture(s) it is qualified on")
        self.require(self.gqa_group_size >= 1, "GQA group size is at least 1")
        self.require(bool(self.unsupported),
                     "a capability must state what it does not support; a kernel with "
                     "no declared limit has not been characterised")
        self.require(bool(self.fallback),
                     "a capability must name its fallback backend; an undeclared "
                     "fallback is a hidden fallback")

    def supports_architecture(self, arch: str) -> bool:
        return arch in self.architectures

    def disagreements_with(self, other: KernelCapability) -> list[str]:
        """Semantic mismatches that make substitution unsafe, in plain words."""
        problems: list[str] = []
        if self.operator is not other.operator:
            problems.append(f"operator {self.operator.value} != {other.operator.value}")
        if self.qk_norm != other.qk_norm:
            problems.append(
                f"QK normalisation {self.qk_norm} != {other.qk_norm}: substituting "
                f"here silently drops or adds a normalisation the model needs")
        if self.gqa_group_size != other.gqa_group_size:
            problems.append(
                f"GQA group size {self.gqa_group_size} != {other.gqa_group_size}")
        if self.mask is not other.mask:
            problems.append(f"mask {self.mask.value} != {other.mask.value}")
        if self.positions is not other.positions:
            problems.append(f"positions {self.positions.value} != {other.positions.value}")
        if self.kv_cache_layout is not other.kv_cache_layout:
            problems.append(
                f"KV cache layout {self.kv_cache_layout.value} != "
                f"{other.kv_cache_layout.value}")
        return problems
