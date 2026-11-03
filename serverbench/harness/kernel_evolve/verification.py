"""verification -- what correct means for a kernel, and how it is proved.

Six sections, each opening with its own banner and description: the ladder,
structured probes, reference implementations, mutants and the kill matrix, the
dispatch anti-cheat, and the megakernel exclusion.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import importlib
import re
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from serving_evolve.contracts.base import Contract
from serving_evolve.contracts.gates import (
    AcceptancePolicy,
    GateResult,
    GateSpec,
    GateStatus,
    ReasonCode,
)

from .contracts.capability import Operator
from .contracts.verification import REQUIRED_MUTANTS, MutantClass, NegativeControl

# ==========================================================================
# The ladder: what is checked, in what order, and why that order.
# ==========================================================================
"""The verification ladder: what is checked, in what order, and why that order.

Eleven rungs, cheapest first. The ordering is not tidiness — it is the difference
between spending two seconds to learn a candidate does not compile and spending
forty minutes of device time to learn the same thing at the end of an end-to-end
benchmark.

::

    1  manifest      the source is what it says it is
    2  build         it compiles and imports, in isolation
    3  dispatch      a custom kernel actually runs; no library shortcut
    4  parity        it agrees with the reference on the nominal shape
    5  boundaries    ... and at every edge of its declared envelope
    6  numerics      finite values, guard regions intact, no memory damage
    7  determinism   it repeats, if it promised to
    8  controls      the checks above can still kill a known-broken variant
    9  microbench    it is faster in isolation                (may reject, never accept)
    10 integration   it is still correct inside the model
    11 serving       the server is correct and faster end to end

Two rules the ladder enforces rather than documents.

**A microbenchmark can reject and can never accept.** Rung 9 is marked
``can_accept=False``. A kernel that is 3x faster on an operator holding 4% of the
critical path cannot move the end-to-end number more than 2.7%; if the operator
was not on the critical path, it cannot move it at all. Only rung 11 accepts.

**Rung 8 is the rung that matters most.** It is the one that notices when the
rungs above it have quietly stopped checking anything. A parity check with a
tolerance wide enough to admit a dropped normalisation passes every candidate,
including the broken ones, and looks exactly like a healthy check until a mutant
walks through it.
"""
# -- the rungs ----------------------------------------------------------------

MANIFEST = "kernel_manifest"
BUILD = "kernel_build"
DISPATCH = "kernel_dispatch"
PARITY = "kernel_parity"
BOUNDARIES = "kernel_boundaries"
NUMERICS = "kernel_numerics"
DETERMINISM = "kernel_determinism"
CONTROLS = "kernel_negative_controls"
MICROBENCH = "kernel_microbenchmark"
INTEGRATION = "kernel_model_integration"
SERVING = "kernel_serving_endtoend"

#: The ladder, in the order it runs. ``cost_rank`` is what orders it; the numbers
#: are meaningful only relative to one another.
RUNGS: tuple[GateSpec, ...] = (
    GateSpec(gate_id=MANIFEST, cost_rank=10,
             description="the candidate's declared source exists, hashes to what the "
                         "manifest says, and stays inside the permitted scope",
             estimated_seconds=1.0),
    GateSpec(gate_id=BUILD, cost_rank=20,
             description="the candidate compiles and imports in an isolated process, "
                         "and exposes its declared entry point",
             estimated_seconds=120.0),
    GateSpec(gate_id=DISPATCH, cost_rank=30,
             description="a custom kernel actually executes on the target workload, "
                         "rather than a wrapper dispatching to the library the "
                         "baseline already uses",
             estimated_seconds=15.0, estimated_device_seconds=10.0),
    GateSpec(gate_id=PARITY, cost_rank=40,
             description="output matches the reference on the nominal shape, within "
                         "the frozen dtype-aware tolerance",
             estimated_seconds=30.0, estimated_device_seconds=25.0),
    GateSpec(gate_id=BOUNDARIES, cost_rank=50,
             description="output matches at every declared shape, dtype and layout "
                         "boundary, and unsupported inputs are refused rather than "
                         "quietly computed",
             estimated_seconds=180.0, estimated_device_seconds=150.0),
    GateSpec(gate_id=NUMERICS, cost_rank=60,
             description="no NaN or Inf, guard regions intact, no out-of-bounds write, "
                         "and resources released after a failure",
             estimated_seconds=90.0, estimated_device_seconds=80.0),
    GateSpec(gate_id=DETERMINISM, cost_rank=70,
             description="repeated runs agree, where the candidate promised they would",
             estimated_seconds=60.0, estimated_device_seconds=50.0),
    GateSpec(gate_id=CONTROLS, cost_rank=80,
             description="the checks above still kill every required known-broken "
                         "variant; this is the check on the checks",
             estimated_seconds=240.0, estimated_device_seconds=200.0),
    GateSpec(gate_id=MICROBENCH, cost_rank=90, can_accept=False,
             description="faster in isolation under the frozen timing protocol. May "
                         "reject a candidate and may never accept one",
             estimated_seconds=300.0, estimated_device_seconds=280.0),
    GateSpec(gate_id=INTEGRATION, cost_rank=100,
             description="still correct once wired into the model's forward path",
             estimated_seconds=600.0, estimated_device_seconds=500.0),
    GateSpec(gate_id=SERVING, cost_rank=110,
             description="the served system is correct and measurably faster end to end",
             estimated_seconds=1800.0, estimated_device_seconds=1700.0),
)

RUNGS_BY_ID: dict[str, GateSpec] = {g.gate_id: g for g in RUNGS}

#: Rungs that must pass before a kernel candidate may be accepted. The
#: microbenchmark is deliberately absent: it is advisory by construction.
REQUIRED_RUNGS: tuple[str, ...] = (
    MANIFEST, BUILD, DISPATCH, PARITY, BOUNDARIES, NUMERICS, CONTROLS,
    INTEGRATION, SERVING,
)

#: Rungs that may reject and never accept.
ADVISORY_RUNGS: tuple[str, ...] = (MICROBENCH,)


def default_policy(policy_id: str = "kernel-ladder-v1") -> AcceptancePolicy:
    """The acceptance rule for a kernel candidate.

    ``DETERMINISM`` is not required, because a kernel that never promised
    determinism should not be failed for not having it; when a plan does promise
    it, :func:`run_ladder` adds the rung to the required set for that plan.
    """
    return AcceptancePolicy(policy_id=policy_id,
                            required_gates=list(REQUIRED_RUNGS),
                            advisory_gates=list(ADVISORY_RUNGS))


@dataclasses.dataclass(frozen=True)
class LadderPlan(Contract):
    """Which rungs run for this candidate, and what each is allowed to conclude."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "kernel_ladder_plan"

    plan_id: str = ""
    rungs: list[str] = dataclasses.field(default_factory=lambda: [g.gate_id for g in RUNGS])
    #: Rungs skipped for a stated reason. A skip is recorded, never silent: a
    #: ladder that quietly drops its integration rung reports "all gates passed".
    skipped: dict[str, str] = dataclasses.field(default_factory=dict)
    require_determinism: bool = True

    def validate(self) -> None:
        self.require(bool(self.plan_id), "a ladder plan needs an id")
        unknown = [r for r in self.rungs if r not in RUNGS_BY_ID]
        self.require(not unknown, f"unknown rung(s) {unknown}")
        required_missing = [r for r in REQUIRED_RUNGS
                            if r not in self.rungs and r not in self.skipped]
        self.require(not required_missing,
                     f"required rung(s) {required_missing} are neither planned nor "
                     f"explicitly skipped with a reason")
        for rung, reason in self.skipped.items():
            self.require(bool(reason), f"rung {rung} is skipped with no reason given")

    def ordered(self) -> list[GateSpec]:
        """Every rung this plan will produce a result for, cheapest first.

        Skipped rungs are included. A rung that is dropped from the plan and
        silently omitted from the results turns "every gate passed" into a claim
        about a shorter ladder than the reader has in mind.
        """
        ids = {*self.rungs, *self.skipped}
        return sorted((RUNGS_BY_ID[r] for r in ids), key=lambda g: g.cost_rank)

    def policy(self) -> AcceptancePolicy:
        """The acceptance rule for this plan.

        A required rung that was skipped stays required. Dropping it here would
        make skipping a rung a way to be accepted without it, which is the
        opposite of what a recorded skip is for: the decision comes back INVALID,
        naming the rung, rather than ACCEPTED over a ladder that never ran.
        """
        required = [r for r in REQUIRED_RUNGS if r in self.rungs or r in self.skipped]
        if self.require_determinism and DETERMINISM in self.rungs:
            required.append(DETERMINISM)
        return AcceptancePolicy(policy_id=f"{self.plan_id}-policy",
                                required_gates=required,
                                advisory_gates=[r for r in ADVISORY_RUNGS
                                                if r in self.rungs])


# -- running it ---------------------------------------------------------------

#: A rung implementation: given the run context, return what it concluded.
RungFn = Callable[[dict[str, Any]], GateResult]


def skipped_result(gate_id: str, reason: str) -> GateResult:
    return GateResult(gate_id=gate_id, status=GateStatus.SKIPPED,
                      reason_code=ReasonCode.ADAPTER_UNSUPPORTED,
                      message=reason)


def not_run_result(gate_id: str, because: str) -> GateResult:
    """A rung that never ran because an earlier one stopped the ladder.

    Recorded rather than omitted. "Nine gates passed" reads very differently when
    the tenth is missing than when it is present and says why.
    """
    return GateResult(gate_id=gate_id, status=GateStatus.SKIPPED,
                      reason_code=ReasonCode.OK,
                      message=f"not run: {because}")


def run_ladder(plan: LadderPlan, implementations: Mapping[str, RungFn],
               context: dict[str, Any] | None = None) -> list[GateResult]:
    """Run the rungs in cost order, stopping at the first decisive non-pass.

    Stopping early is the point: a candidate that fails to build should not spend
    device time on a benchmark. Every rung that consequently did not run is still
    present in the results, marked and explained, so a reader never has to guess
    whether a missing rung passed or was never reached.

    A rung that raises is ``INVALID`` — the evaluator failed, not the candidate —
    and the exception's text is preserved rather than turned into a verdict.
    """
    ctx = dict(context or {})
    results: list[GateResult] = []
    stopped_by = ""

    for spec in plan.ordered():
        if spec.gate_id in plan.skipped:
            results.append(skipped_result(spec.gate_id, plan.skipped[spec.gate_id]))
            continue
        if stopped_by:
            results.append(not_run_result(spec.gate_id, f"{stopped_by} did not pass"))
            continue
        fn = implementations.get(spec.gate_id)
        if fn is None:
            results.append(GateResult.invalid(
                spec.gate_id, ReasonCode.ADAPTER_UNSUPPORTED,
                f"no implementation is registered for rung {spec.gate_id}; the ladder "
                f"cannot conclude anything about it"))
            stopped_by = spec.gate_id
            continue
        t0 = time.monotonic()
        try:
            result = fn(ctx)
        except Exception as exc:  # noqa: BLE001 - an evaluator crash is INVALID, not FAIL
            result = GateResult.invalid(
                spec.gate_id, ReasonCode.INFRASTRUCTURE_ERROR,
                f"the rung itself failed ({type(exc).__name__}: {exc}); this is an "
                f"evaluator failure and is not charged to the candidate")
        if result.wall_seconds == 0.0:
            result = result.evolve(wall_seconds=round(time.monotonic() - t0, 6))
        results.append(result)
        if GateStatus(result.status).blocks_acceptance and \
                GateStatus(result.status) is not GateStatus.SKIPPED:
            stopped_by = spec.gate_id

    return results


def summarize(results: Sequence[GateResult]) -> str:
    counts: dict[str, int] = {}
    for r in results:
        counts[GateStatus(r.status).value] = counts.get(GateStatus(r.status).value, 0) + 1
    parts = [f"{n} {status.lower()}" for status, n in sorted(counts.items())]
    return ", ".join(parts)


# ==========================================================================
# Probes: structured inputs that catch what random inputs and tolerances do not.
# ==========================================================================
"""Structured inputs that catch what random inputs and tolerances do not.

`allclose(rtol, atol)` on random data is the standard correctness check and it is
comfortably foolable. Three ways, all seen in real kernels:

* return zeros for the small tail of values that drive the relative tolerance;
* accumulate in a lower precision that is "close enough" on random inputs and
  biased on structured ones;
* skip work for masked-out positions and happen to land inside tolerance anyway.

Each probe below builds an input whose *exact* answer is known by construction,
so the check is an equality rather than a tolerance. That removes the room the
three failures above live in.

Every probe returns ``(ok, detail)`` and never raises for a wrong answer — a
wrong answer is a finding, not an error.
"""
#: Largest integer each format represents exactly. Above it, sums round, and an
#: exactness check would start failing correct kernels.
#:   bfloat16: 8-bit significand -> integers below 256 are exact
#:   float16: 11-bit significand -> integers below 2048 are exact
EXACT_INTEGER_LIMIT = {"bfloat16": 256, "float16": 2048, "float32": 1 << 24}


def binary_matmul_exact(run: Callable[..., Any], m: int, n: int, k: int, *,
                        density: float = 0.25, dtype: str = "bfloat16",
                        seed: int = 1234) -> tuple[bool, str]:
    """A matmul over 0/1 matrices, where every output is an exact small integer.

    With both operands in {0, 1}, each output element is a count — a
    non-negative integer no larger than ``k``. Choose the density so the largest
    count stays under the format's exact-integer limit and the reference is not
    approximately right, it is *exactly* right. Any deviation at all is a defect,
    and a kernel accumulating in the wrong precision fails here while sailing
    through a tolerance check on random data.
    """
    import torch

    device = torch.device("cuda")
    torch_dtype = getattr(torch, dtype)
    limit = EXACT_INTEGER_LIMIT[dtype]
    generator = torch.Generator(device=device).manual_seed(seed)
    a = (torch.rand(m, k, generator=generator, device=device) < density).to(torch_dtype)
    b = (torch.rand(n, k, generator=generator, device=device) < density).to(torch_dtype)

    expected = torch.matmul(a.float(), b.float().t())
    largest = expected.max().item()
    if largest >= limit:
        return False, (f"probe is mis-configured: the largest count is {largest:.0f} "
                       f"and {dtype} is only exact below {limit}. Lower the density "
                       f"rather than relaxing the check")

    got = run(a, b).float()
    if got.shape != expected.shape:
        return False, f"shape {tuple(got.shape)} != {tuple(expected.shape)}"
    wrong = int((got != expected).sum().item())
    if wrong:
        worst = (got - expected).abs().max().item()
        return False, (f"{wrong} of {expected.numel()} outputs are not exact "
                       f"(worst |delta| {worst:.1f}). Every value here is a small "
                       f"integer the format represents exactly, so any difference "
                       f"is arithmetic that did not happen or happened in the "
                       f"wrong precision")
    return True, (f"all {expected.numel()} outputs exact over 0/1 inputs "
                  f"(max count {largest:.0f} < {limit})")


def position_encoded_attention(run: Callable[..., Any], batch: int, seq: int,
                               q_heads: int, kv_heads: int, head_dim: int, *,
                               dtype: str = "bfloat16") -> tuple[bool, str]:
    """Attention whose answer encodes *which position* was attended to.

    Keys are one-hot along the head dimension and values carry their own index in
    channel zero. With a uniform query, the output's channel zero must be the mean
    of the positions attended — a number that moves the moment indexing is off by
    one, a mask is applied at the wrong offset, or a KV entry is read stale.

    Random inputs hide all three: they produce a plausible weighted average
    whatever the indexing, and the error disappears into the tolerance.
    """
    import torch

    device = torch.device("cuda")
    torch_dtype = getattr(torch, dtype)

    keys = torch.zeros(batch, seq, kv_heads, head_dim, device=device, dtype=torch_dtype)
    index = torch.arange(seq, device=device) % head_dim
    for head in range(kv_heads):
        flat = keys[:, :, head, :].reshape(batch * seq, head_dim)
        flat[torch.arange(batch * seq, device=device), index.repeat(batch)] = 5.0

    values = torch.zeros(batch, seq, kv_heads, head_dim, device=device,
                         dtype=torch_dtype)
    values[:, :, :, 0] = torch.arange(seq, device=device,
                                      dtype=torch_dtype).view(1, seq, 1)

    queries = torch.zeros(batch, seq, q_heads, head_dim, device=device,
                          dtype=torch_dtype)

    got = run(queries, keys, values, causal=False).float()
    expected = (seq - 1) / 2.0
    observed = got[..., 0].mean().item()
    error = abs(observed - expected) / max(expected, 1e-9)
    if error > 0.05:
        return False, (f"channel-0 mean {observed:.2f}, expected {expected:.2f} "
                       f"(relative error {error:.3f}). The values encode their own "
                       f"position, so this says the kernel attended somewhere else")
    return True, (f"attended the right positions: channel-0 mean {observed:.2f} "
                  f"against {expected:.2f}")


def causal_mask_leak(run: Callable[..., Any], batch: int, seq: int, q_heads: int,
                     kv_heads: int, head_dim: int, *,
                     dtype: str = "bfloat16") -> tuple[bool, str]:
    """A causal kernel that reads the future, caught by making the future loud.

    Everything after position zero is given an enormous value. Under a correct
    causal mask the first query attends only to position zero and its output is
    small. A kernel that leaks even one future position produces something huge,
    and the size of the failure is the point: it cannot hide inside a tolerance.
    """
    import torch

    device = torch.device("cuda")
    torch_dtype = getattr(torch, dtype)
    keys = torch.zeros(batch, seq, kv_heads, head_dim, device=device, dtype=torch_dtype)
    keys[:, :, :, 0] = 1.0
    values = torch.zeros(batch, seq, kv_heads, head_dim, device=device,
                         dtype=torch_dtype)
    values[:, 1:, :, :] = 100.0
    queries = torch.zeros(batch, seq, q_heads, head_dim, device=device,
                          dtype=torch_dtype)
    queries[:, :, :, 0] = 1.0

    got = run(queries, keys, values, causal=True).float()
    first = got[:, 0].abs().max().item()
    if first > 1.0:
        return False, (f"the first query's output reaches {first:.1f} where every "
                       f"legal source is zero: the causal mask is leaking future "
                       f"positions")
    return True, f"first query stayed at {first:.3f}; no future leaked into it"


def poisoned_output(run: Callable[..., Any], make_inputs: Callable[[], Any], *,
                    dtype: str = "bfloat16") -> tuple[bool, str]:
    """A partial write, caught by making unwritten memory obvious.

    Fill the allocator's next block with NaN and free it, so the kernel's output
    tensor is very likely to land on it. Anything the kernel fails to write comes
    back NaN. Comparing against a fresh allocation would not catch this:
    uninitialised memory is frequently plausible and occasionally zero.
    """
    import torch

    args = make_inputs()
    reference = run(*args)
    poison = torch.full_like(reference, float("nan"))
    del poison, reference
    got = run(*args)
    bad = int((~torch.isfinite(got.float())).sum().item())
    if bad:
        return False, (f"{bad} output element(s) came back non-finite over poisoned "
                       f"memory: the kernel did not write them")
    return True, "every output element was written over poisoned memory"


#: Which probes apply to which operator. An operator with no entry gets none, and
#: that gap is visible in the kill matrix rather than silently absent.
BY_OPERATOR: dict[str, tuple[str, ...]] = {
    "qkv_projection": ("binary_matmul_exact",),
    "output_projection": ("binary_matmul_exact",),
    "gated_mlp": ("binary_matmul_exact",),
    "lm_head": ("binary_matmul_exact",),
    "attention_prefill": ("position_encoded_attention", "causal_mask_leak"),
    "attention_decode": ("position_encoded_attention",),
    "rms_norm": ("poisoned_output",),
    "qk_norm": ("poisoned_output",),
    "rope": ("poisoned_output",),
    "silu_mul": ("poisoned_output",),
}

PROBES: dict[str, Callable[..., tuple[bool, str]]] = {
    "binary_matmul_exact": binary_matmul_exact,
    "position_encoded_attention": position_encoded_attention,
    "causal_mask_leak": causal_mask_leak,
    "poisoned_output": poisoned_output,
}


def for_operator(operator: str) -> list[str]:
    """The structured probes that apply to an operator, by name."""
    return list(BY_OPERATOR.get(str(operator), ()))


# ==========================================================================
# References: what "correct" is measured against.
# ==========================================================================
"""Reference implementations: what "correct" is measured against.

A reference is not a slower version of the candidate. It is an *independent*
statement of what the operator computes, and the independence is the whole
value: a reference that calls the same library the candidate dispatches to
cannot falsify anything, and every rung above it becomes decoration.

So each reference here is written from the operator's definition in framework
primitives, with no attempt to be fast, and each declares where its semantics
came from. `KernelTaskSpec` refuses a reference that does not.

The registry maps an operator to its reference and its input generator, so the
ladder can build a fixture for a shape without the candidate supplying one —
a candidate that generates its own test inputs is grading its own homework.
"""
@dataclasses.dataclass(frozen=True)
class ReferenceEntry:
    """One operator's reference, its inputs, and where its semantics came from."""

    operator: Operator
    run: Callable[..., Any]
    make_inputs: Callable[..., tuple[Any, ...]]
    provenance: str
    #: Rough arithmetic per call, for a roofline sanity check. Not a measurement.
    flops: Callable[..., int] | None = None


def _torch():
    import torch

    return torch


# -- normalisation -----------------------------------------------------------

def rms_norm(x: Any, weight: Any, eps: float = 1e-6) -> Any:
    """RMS normalisation, from the definition.

    FP32 accumulation because a bf16 sum of squares over a few thousand elements
    loses enough precision to shift the scale systematically — which would make
    the *reference* the wrong one, and every candidate wrong by comparison.
    """
    torch = _torch()
    x32 = x.to(torch.float32)
    scale = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (x32 * scale * weight.to(torch.float32)).to(x.dtype)


def make_rms_norm_inputs(rows: int, cols: int, dtype: str = "bfloat16",
                         device: str = "cuda", seed: int = 0) -> tuple[Any, ...]:
    torch = _torch()
    generator = torch.Generator(device=device).manual_seed(seed)
    torch_dtype = getattr(torch, dtype)
    x = torch.randn(rows, cols, generator=generator, device=device, dtype=torch_dtype)
    weight = torch.randn(cols, generator=generator, device=device, dtype=torch_dtype)
    return (x, weight)


# -- position encoding -------------------------------------------------------

def rope_neox(q: Any, k: Any, cos: Any, sin: Any) -> tuple[Any, Any]:
    """Rotary embedding, NeoX convention: pairs ``(i, i + d/2)``.

    The convention is the point. GPT-J pairs ``(2i, 2i+1)`` instead, and a kernel
    written for one and applied to the other produces well-formed, wrong output —
    so the reference names which one it implements rather than leaving it to be
    inferred from a shape.
    """
    torch = _torch()

    def rotate(t: Any) -> Any:
        half = t.shape[-1] // 2
        left, right = t[..., :half], t[..., half:]
        return torch.cat((-right, left), dim=-1)

    q32, k32 = q.to(torch.float32), k.to(torch.float32)
    cos32, sin32 = cos.to(torch.float32), sin.to(torch.float32)
    return ((q32 * cos32 + rotate(q32) * sin32).to(q.dtype),
            (k32 * cos32 + rotate(k32) * sin32).to(k.dtype))


def make_rope_inputs(batch: int, seq: int, heads: int, head_dim: int,
                     dtype: str = "bfloat16", device: str = "cuda",
                     seed: int = 0) -> tuple[Any, ...]:
    torch = _torch()
    generator = torch.Generator(device=device).manual_seed(seed)
    torch_dtype = getattr(torch, dtype)
    shape = (batch, seq, heads, head_dim)
    q = torch.randn(*shape, generator=generator, device=device, dtype=torch_dtype)
    k = torch.randn(*shape, generator=generator, device=device, dtype=torch_dtype)
    positions = torch.arange(seq, device=device, dtype=torch.float32)
    inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device,
                                        dtype=torch.float32) / head_dim))
    angles = positions[:, None] * inv[None, :]
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1).view(1, seq, 1, head_dim)
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1).view(1, seq, 1, head_dim)
    return (q, k, cos.to(torch_dtype), sin.to(torch_dtype))


# -- activation --------------------------------------------------------------

def silu_mul(gate: Any, up: Any) -> Any:
    """``silu(gate) * up``, computed in FP32.

    bf16 sigmoid near zero is not accurate enough to leave alone, and a reference
    that is slightly wrong there makes every candidate slightly wrong with it.
    """
    torch = _torch()
    g = gate.to(torch.float32)
    return ((g * torch.sigmoid(g)) * up.to(torch.float32)).to(gate.dtype)


def make_silu_mul_inputs(rows: int, cols: int, dtype: str = "bfloat16",
                         device: str = "cuda", seed: int = 0) -> tuple[Any, ...]:
    torch = _torch()
    generator = torch.Generator(device=device).manual_seed(seed)
    torch_dtype = getattr(torch, dtype)
    gate = torch.randn(rows, cols, generator=generator, device=device,
                       dtype=torch_dtype)
    up = torch.randn(rows, cols, generator=generator, device=device, dtype=torch_dtype)
    return (gate, up)


# -- linear ------------------------------------------------------------------

def linear(x: Any, weight: Any) -> Any:
    """``y = x @ W.T``, accumulated in FP32."""
    torch = _torch()
    return torch.matmul(x.to(torch.float32), weight.to(torch.float32).t()).to(x.dtype)


def make_linear_inputs(m: int, n: int, k: int, dtype: str = "bfloat16",
                       device: str = "cuda", seed: int = 0) -> tuple[Any, ...]:
    torch = _torch()
    generator = torch.Generator(device=device).manual_seed(seed)
    torch_dtype = getattr(torch, dtype)
    x = torch.randn(m, k, generator=generator, device=device, dtype=torch_dtype)
    weight = torch.randn(n, k, generator=generator, device=device, dtype=torch_dtype)
    return (x, weight)


# -- attention ---------------------------------------------------------------

def attention(q: Any, k: Any, v: Any, *, causal: bool = True) -> Any:
    """Grouped-query attention, written out rather than dispatched.

    Materialises the full score matrix, which is exactly what a real kernel must
    not do — and exactly why it is the right reference: it is the definition,
    with no streaming-softmax rescale to get subtly wrong.

    Shapes are ``(batch, seq, heads, head_dim)``. When ``q`` has more heads than
    ``k``, the KV heads are repeated to match, which is what GQA means.
    """
    torch = _torch()
    batch, seq, q_heads, head_dim = q.shape
    kv_heads = k.shape[2]
    group = q_heads // kv_heads

    q32 = q.to(torch.float32).transpose(1, 2)
    k32 = k.to(torch.float32).transpose(1, 2)
    v32 = v.to(torch.float32).transpose(1, 2)
    if group > 1:
        k32 = k32.repeat_interleave(group, dim=1)
        v32 = v32.repeat_interleave(group, dim=1)

    scores = torch.matmul(q32, k32.transpose(-1, -2)) / (head_dim ** 0.5)
    if causal:
        mask = torch.triu(torch.ones(seq, seq, device=q.device, dtype=torch.bool),
                          diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v32).transpose(1, 2).to(q.dtype)


def make_attention_inputs(batch: int, seq: int, q_heads: int, kv_heads: int,
                          head_dim: int, dtype: str = "bfloat16",
                          device: str = "cuda", seed: int = 0) -> tuple[Any, ...]:
    torch = _torch()
    generator = torch.Generator(device=device).manual_seed(seed)
    torch_dtype = getattr(torch, dtype)
    q = torch.randn(batch, seq, q_heads, head_dim, generator=generator,
                    device=device, dtype=torch_dtype)
    k = torch.randn(batch, seq, kv_heads, head_dim, generator=generator,
                    device=device, dtype=torch_dtype)
    v = torch.randn(batch, seq, kv_heads, head_dim, generator=generator,
                    device=device, dtype=torch_dtype)
    return (q, k, v)


REGISTRY: dict[Operator, ReferenceEntry] = {
    Operator.RMS_NORM: ReferenceEntry(
        operator=Operator.RMS_NORM, run=rms_norm, make_inputs=make_rms_norm_inputs,
        provenance="RMS normalisation as defined in Zhang & Sennrich (2019) and "
                   "as implemented by the model's own configuration",
        flops=lambda rows, cols, **_: rows * cols * 4),
    Operator.QK_NORM: ReferenceEntry(
        operator=Operator.QK_NORM, run=rms_norm, make_inputs=make_rms_norm_inputs,
        provenance="per-head RMS normalisation applied to Q and K before RoPE, as "
                   "the Qwen3 model card describes"),
    Operator.ROPE: ReferenceEntry(
        operator=Operator.ROPE, run=rope_neox, make_inputs=make_rope_inputs,
        provenance="rotary position embedding, NeoX pairing, from Su et al. (2021) "
                   "and the checkpoint's rope configuration"),
    Operator.SILU_MUL: ReferenceEntry(
        operator=Operator.SILU_MUL, run=silu_mul, make_inputs=make_silu_mul_inputs,
        provenance="SwiGLU as defined in Shazeer (2020)"),
    Operator.QKV_PROJECTION: ReferenceEntry(
        operator=Operator.QKV_PROJECTION, run=linear, make_inputs=make_linear_inputs,
        provenance="a dense linear layer, y = x @ W.T",
        flops=lambda m, n, k, **_: 2 * m * n * k),
    Operator.OUTPUT_PROJECTION: ReferenceEntry(
        operator=Operator.OUTPUT_PROJECTION, run=linear,
        make_inputs=make_linear_inputs, provenance="a dense linear layer",
        flops=lambda m, n, k, **_: 2 * m * n * k),
    Operator.LM_HEAD: ReferenceEntry(
        operator=Operator.LM_HEAD, run=linear, make_inputs=make_linear_inputs,
        provenance="a dense linear layer over the vocabulary",
        flops=lambda m, n, k, **_: 2 * m * n * k),
    Operator.ATTENTION_PREFILL: ReferenceEntry(
        operator=Operator.ATTENTION_PREFILL, run=attention,
        make_inputs=make_attention_inputs,
        provenance="scaled dot-product attention from Vaswani et al. (2017) with "
                   "grouped-query heads as in Ainslie et al. (2023)"),
    Operator.ATTENTION_DECODE: ReferenceEntry(
        operator=Operator.ATTENTION_DECODE, run=attention,
        make_inputs=make_attention_inputs,
        provenance="the same definition, at a single query position"),
}


def entry(operator: Operator | str) -> ReferenceEntry | None:
    """The reference for an operator, or None when nobody has written one.

    None is a real answer and the ladder reports it as `INVALID` rather than
    passing the candidate: an operator with no reference has nothing to be
    correct against, and a gate that quietly skips is worse than one that fails.
    """
    key = operator if isinstance(operator, Operator) else Operator(operator)
    return REGISTRY.get(key)


# ==========================================================================
# Mutants: known-broken kernels, and the kill matrix that proves the checks still work.
# ==========================================================================
"""Known-broken kernels, and the kill matrix that proves the checks still work.

Every entry here is a defect that has actually shipped in a real kernel, not an
abstract fault model. That matters, because a mutant nobody would write tells you
nothing about a check battery that has to catch the ones people do write.

The kill matrix is the deliverable. For each required corruption class it records
which rung caught it, and a class caught by *no* rung is a hole — the checks
above it will pass a kernel with that defect and report a clean sheet. A class
caught by the *wrong* rung is also reported, because it usually means the rung
that should have caught it is not checking what its name says.

There is a symmetric failure the matrix also has to catch: a legitimate control —
an unmodified reference, or a mathematically equivalent rewrite — being rejected.
A battery that kills everything is not sensitive, it is broken, and it will
reject every real candidate for the rest of the run.
"""
#: What each corruption class does, and which rung is supposed to notice.
#:
#: The pairing is the interesting part. A dropped normalisation is a *numerical*
#: defect and belongs to parity; a stale KV offset only shows up once the cache
#: has history, so it belongs to the boundary sweep; a silent fallback is not a
#: numerical defect at all — the answer is right — and only the dispatch rung can
#: see it.
CATALOG: dict[MutantClass, dict[str, str]] = {
    MutantClass.WRONG_MASK: {
        "description": "attend to positions after the current one",
        "expected_killer": PARITY,
        "why": "a causal kernel that leaks future tokens still produces fluent text "
               "and normal-looking logits; only comparison against the reference sees it",
    },
    MutantClass.MISSING_NORMALIZATION: {
        "description": "skip the RMS normalisation of Q and K",
        "expected_killer": PARITY,
        "why": "the Qwen3 case exactly: a kernel written for a model without QK "
               "normalisation runs, is faster, and omits an operation the model needs",
    },
    MutantClass.STALE_KV_OFFSET: {
        "description": "read the KV cache at the previous step's write offset",
        "expected_killer": BOUNDARIES,
        "why": "invisible on a single-token test and on the first decode step; it needs "
               "a sequence with history, which is what the boundary sweep supplies",
    },
    MutantClass.TRUNCATED_VOCABULARY: {
        "description": "reduce over the first 2^k logits and ignore the tail",
        "expected_killer": BOUNDARIES,
        "why": "a 151936-entry vocabulary is not a power of two; a kernel that assumes "
               "it is loses the last partial tile, and loses it only for rare tokens",
    },
    MutantClass.WRONG_REDUCTION: {
        "description": "sum where the reference takes a maximum, or reduce the wrong axis",
        "expected_killer": PARITY,
        "why": "the classic softmax defect; catastrophic numerically and therefore easy "
               "to catch, which makes it a good canary for a tolerance set too wide",
    },
    MutantClass.PARTIAL_WRITE: {
        "description": "leave the last elements of the output tensor unwritten",
        "expected_killer": NUMERICS,
        "why": "uninitialised memory is often plausible and occasionally zero, so the "
               "check has to be a poisoned buffer rather than a value comparison",
    },
    MutantClass.SILENT_FALLBACK: {
        "description": "dispatch to the library implementation without recording it",
        "expected_killer": DISPATCH,
        "why": "the answer is correct, so no numerical check can see it. It reports the "
               "library's speed as the candidate's, which is the whole failure",
    },
    MutantClass.INACCURATE_TIMING: {
        "description": "measure without synchronising, timing kernel launch only",
        "expected_killer": MICROBENCH,
        "why": "produces enormous, entirely fictitious speedups; the timing protocol "
               "has to be able to detect its own misuse",
    },
    MutantClass.WRONG_POSITION: {
        "description": "apply rotary embeddings at position zero for every token",
        "expected_killer": BOUNDARIES,
        "why": "correct for the first token of every sequence, wrong for all the rest",
    },
    MutantClass.DROPPED_GQA_GROUP: {
        "description": "broadcast one KV head to every query head in the group",
        "expected_killer": PARITY,
        "why": "shapes stay valid and the output is well-formed; only the values differ",
    },
}


def default_controls(bound_source_digest: str = "") -> list[NegativeControl]:
    """One control per required corruption class, bound to a frozen source tree."""
    return [
        NegativeControl(
            control_id=f"control-{mutant.value}",
            mutant=mutant,
            description=CATALOG[mutant]["description"],
            expected_killer=CATALOG[mutant]["expected_killer"],
            bound_source_digest=bound_source_digest,
        )
        for mutant in sorted(REQUIRED_MUTANTS, key=lambda m: m.value)
    ]


@dataclasses.dataclass(frozen=True)
class KillMatrix(Contract):
    """What each control did, and whether the battery is qualified.

    ``qualified`` is deliberately strict. Any hole, any wrong killer, and any
    legitimate control rejected all fail it — because each of those means a
    different thing is wrong, and none of them means "close enough".
    """

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "kernel_kill_matrix"

    matrix_id: str = ""
    #: The classes this battery was required to cover. Carried on the matrix rather
    #: than assumed globally, because an operator with no mask cannot have a
    #: wrong-mask defect and failing it for that would be theatre.
    required: list[str] = dataclasses.field(
        default_factory=lambda: sorted(m.value for m in REQUIRED_MUTANTS))
    #: mutant class -> the rung that actually killed it, or "" for survived.
    killed_by: dict[str, str] = dataclasses.field(default_factory=dict)
    #: Controls that were meant to be caught and were not. Each is a hole.
    survived: list[str] = dataclasses.field(default_factory=list)
    #: Controls caught by a rung other than the one that should have.
    wrong_killer: dict[str, str] = dataclasses.field(default_factory=dict)
    #: Legitimate variants that were rejected. A battery that kills these is not
    #: sensitive; it is broken, and will reject real work for the rest of the run.
    false_positives: list[str] = dataclasses.field(default_factory=list)
    #: Controls that could not be applied at all — a moved target, a patch that no
    #: longer fits. Not a pass and not a failure: a broken tool.
    invalid_controls: list[str] = dataclasses.field(default_factory=list)

    def validate(self) -> None:
        self.require(bool(self.matrix_id), "a kill matrix needs an id")

    @property
    def qualified(self) -> bool:
        """Every required class tested, killed, by the right rung, with no false positive.

        Untested counts against it. A matrix that reports "nothing survived" over
        three of seven classes is describing a battery with four holes, and it
        reads exactly like a healthy one — which is the failure this property
        exists to prevent.
        """
        return not (self.survived or self.wrong_killer or self.false_positives
                    or self.invalid_controls or self.missing_classes())

    def missing_classes(self) -> list[str]:
        """Required classes with no verdict at all — untested, not merely unkilled."""
        tested = set(self.killed_by) | set(self.survived) | set(self.invalid_controls)
        return sorted(m for m in self.required if m not in tested)

    def explain(self) -> str:
        if self.qualified:
            return (f"every required corruption class was killed by the rung meant to "
                    f"catch it ({len(self.killed_by)} of {len(self.required)})")
        parts = []
        if missing := self.missing_classes():
            parts.append(f"never tested: {missing}")
        if self.survived:
            parts.append(f"survived (the checks cannot see these): {sorted(self.survived)}")
        if self.wrong_killer:
            parts.append(
                "caught by the wrong rung, so the rung that should have is not "
                f"checking what its name says: {dict(sorted(self.wrong_killer.items()))}")
        if self.false_positives:
            parts.append(
                f"legitimate variants rejected, so the battery is over-sensitive and "
                f"will reject real candidates: {sorted(self.false_positives)}")
        if self.invalid_controls:
            parts.append(
                f"could not be applied, so they qualified nothing: "
                f"{sorted(self.invalid_controls)}")
        return "; ".join(parts)


#: Which corruption classes can actually occur for an operator. A class that
#: cannot occur is not a hole; failing a plan for not testing it would be theatre,
#: and — worse — would train people to add empty controls to make a check go green.
APPLICABLE: dict[str, frozenset[MutantClass]] = {
    "rms_norm": frozenset({MutantClass.MISSING_NORMALIZATION,
                           MutantClass.WRONG_REDUCTION, MutantClass.PARTIAL_WRITE,
                           MutantClass.SILENT_FALLBACK,
                           MutantClass.INACCURATE_TIMING}),
    "qk_norm": frozenset({MutantClass.MISSING_NORMALIZATION,
                          MutantClass.WRONG_REDUCTION, MutantClass.PARTIAL_WRITE,
                          MutantClass.SILENT_FALLBACK,
                          MutantClass.INACCURATE_TIMING}),
    "rope": frozenset({MutantClass.WRONG_POSITION, MutantClass.PARTIAL_WRITE,
                       MutantClass.SILENT_FALLBACK, MutantClass.INACCURATE_TIMING}),
    "silu_mul": frozenset({MutantClass.PARTIAL_WRITE, MutantClass.SILENT_FALLBACK,
                           MutantClass.INACCURATE_TIMING}),
    "kv_cache_append": frozenset({MutantClass.STALE_KV_OFFSET,
                                  MutantClass.PARTIAL_WRITE,
                                  MutantClass.SILENT_FALLBACK,
                                  MutantClass.INACCURATE_TIMING}),
    "sampling": frozenset({MutantClass.TRUNCATED_VOCABULARY,
                           MutantClass.WRONG_REDUCTION, MutantClass.PARTIAL_WRITE,
                           MutantClass.SILENT_FALLBACK,
                           MutantClass.INACCURATE_TIMING}),
    "topk": frozenset({MutantClass.TRUNCATED_VOCABULARY,
                       MutantClass.WRONG_REDUCTION, MutantClass.PARTIAL_WRITE,
                       MutantClass.SILENT_FALLBACK, MutantClass.INACCURATE_TIMING}),
}


def applicable_to(operator: str) -> frozenset[MutantClass]:
    """The corruption classes worth testing for this operator.

    Anything not listed gets the full required set: an unknown operator is not a
    reason to test less, and defaulting to *more* is the direction that fails
    safely.
    """
    return APPLICABLE.get(str(operator), REQUIRED_MUTANTS)


def build_matrix(matrix_id: str, controls: list[NegativeControl],
                 outcomes: dict[str, list[GateResult]],
                 legitimate: dict[str, list[GateResult]] | None = None,
                 invalid: list[str] | None = None,
                 required: list[str] | None = None) -> KillMatrix:
    """Turn per-control ladder results into a kill matrix.

    ``outcomes`` maps a control id to the results of running the ladder against
    that mutant. ``legitimate`` maps the id of an *unbroken* variant to its
    results; anything rejected there is a false positive.
    """
    killed_by: dict[str, str] = {}
    survived: list[str] = []
    wrong: dict[str, str] = {}

    for control in controls:
        results = outcomes.get(control.control_id, [])
        killer = next((r.gate_id for r in results
                       if GateStatus(r.status) is GateStatus.FAIL), "")
        mutant = str(control.mutant.value if hasattr(control.mutant, "value")
                     else control.mutant)
        if not killer:
            survived.append(mutant)
            continue
        killed_by[mutant] = killer
        if killer != control.expected_killer:
            wrong[mutant] = (f"expected {control.expected_killer}, caught by {killer}")

    false_positives = [
        name for name, results in (legitimate or {}).items()
        if any(GateStatus(r.status) is GateStatus.FAIL for r in results)
    ]
    return KillMatrix(matrix_id=matrix_id, killed_by=killed_by, survived=survived,
                      wrong_killer=wrong, false_positives=false_positives,
                      invalid_controls=sorted(invalid or []),
                      required=(sorted(required) if required is not None
                                else sorted(m.value for m in REQUIRED_MUTANTS)))


# ==========================================================================
# Anti-cheat: catching a candidate that dispatches to the library it was meant to replace.
# ==========================================================================
"""Catching a candidate that dispatches to the library it was meant to replace.

This is the one defect no numerical check can see. A four-line wrapper around the
framework's own matmul produces bit-identical output, passes every tolerance,
every boundary case and every determinism check — and reports the library's speed
as the candidate's.

Two independent probes, because each misses what the other catches:

**Static.** Scan the candidate's source for imports of upstream kernel packages.
Cheap, runs without a GPU, catches `import flash_attn` at the top of a file, and
is blind to a call made through an alias or a `getattr`.

**Runtime.** Monkey-patch the forbidden symbols to record and still execute, run
the candidate's entry point once on a small input, and read the log. Catches
anything that actually calls them, however it got there, and is blind to a path
this particular input does not take — which is why the probe input matters and
why the static scan stays.

What is *not* forbidden is plumbing: allocation (``empty``, ``zeros``, ``full``
and their ``_like`` forms), views, reshapes, transposes, permutes, concatenation
and stacking, splits, ``contiguous``, ``to``, ``clone``, and scalar arithmetic on
small auxiliary tensors. The line is the
primary compute path, and a declared fallback for unsupported shapes is fine
provided it is declared and announces itself.
"""
#: ``(module path, attribute)`` pairs a candidate's primary compute path may not
#: call, grouped by what each would be standing in for.
FORBIDDEN_SYMBOLS: tuple[tuple[str, str], ...] = (
    ("torch", "matmul"), ("torch", "mm"), ("torch", "bmm"), ("torch", "einsum"),
    ("torch", "_scaled_mm"), ("torch", "addmm"), ("torch", "baddbmm"),
    ("torch.nn.functional", "linear"),
    ("torch.nn.functional", "scaled_dot_product_attention"),
    ("torch", "topk"), ("torch", "sort"), ("torch", "multinomial"),
    ("torch", "searchsorted"), ("torch", "argsort"),
    ("torch.nn.functional", "rms_norm"), ("torch.nn.functional", "layer_norm"),
    ("torch.nn.functional", "normalize"), ("torch.nn.functional", "silu"),
    ("torch.nn.functional", "softmax"), ("torch.nn.functional", "log_softmax"),
    ("torch.nn.functional", "gelu"), ("torch.nn.functional", "embedding"),
    # bound methods alias the same implementations
    ("torch.Tensor", "matmul"), ("torch.Tensor", "mm"), ("torch.Tensor", "bmm"),
    ("torch.Tensor", "topk"), ("torch.Tensor", "sort"),
    ("torch.Tensor", "softmax"),
)

#: Packages that *are* the kernel a candidate was asked to write. Importing one
#: anywhere in the candidate's graph is disqualifying on its own.
FORBIDDEN_IMPORTS: tuple[str, ...] = (
    "flash_attn", "vllm", "sglang", "flashinfer", "xformers",
    "transformer_engine", "tensorrt_llm", "vllm_flash_attn", "sgl_kernel",
)

@dataclasses.dataclass(frozen=True)
class DispatchReport(Contract):
    """What the candidate actually called, and whether that is allowed."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "kernel_dispatch_report"

    forbidden_calls: list[str] = dataclasses.field(default_factory=list)
    forbidden_imports: list[str] = dataclasses.field(default_factory=list)
    #: Custom kernel launches the candidate reported. Zero is the whole finding.
    custom_launches: int = 0
    has_device_source: bool = False
    probe_input: str = ""
    #: Set when the probe could not run — an evaluator failure, not a verdict.
    error: str = ""

    def validate(self) -> None:
        self.require(self.custom_launches >= 0, "launch count cannot be negative")

    @property
    def dispatched_only(self) -> bool:
        return bool(self.forbidden_calls or self.forbidden_imports)

    def explain(self) -> str:
        if self.error:
            return f"the probe could not run: {self.error}"
        parts: list[str] = []
        if self.forbidden_imports:
            parts.append(f"imports {sorted(set(self.forbidden_imports))}, which are "
                         f"the kernel this candidate was asked to write")
        if self.forbidden_calls:
            counted = {c: self.forbidden_calls.count(c)
                       for c in sorted(set(self.forbidden_calls))}
            parts.append(f"calls {counted} on the primary compute path. The output "
                         f"will be correct and the speed will be the library's")
        if not parts and self.custom_launches == 0:
            parts.append("launched no kernel of its own; whatever this computes, it "
                         "is not a kernel this candidate wrote")
        if not parts:
            return (f"{self.custom_launches} custom launch(es), no forbidden call or "
                    f"import on the probe input {self.probe_input!r}")
        return "; ".join(parts)


# -- static ------------------------------------------------------------------

def scan_imports(root: Path) -> list[str]:
    """Forbidden packages imported anywhere under ``root``.

    Parses rather than greps. A comment mentioning `vllm`, or a docstring
    explaining that it is forbidden, is not an import — and a scanner that flags
    those is a scanner people switch off.
    """
    found: list[str] = []
    for path in sorted(Path(root).rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text("utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.split(".")[0] in FORBIDDEN_IMPORTS:
                    found.append(f"{path.name}: {name}")
    return found


def scan_device_source(root: Path) -> bool:
    """Whether any device-language source exists to have written a kernel in."""
    suffixes = {".cu", ".cuh", ".cpp", ".cc", ".hip"}
    if any(p.suffix in suffixes for p in Path(root).rglob("*")):
        return True
    # Triton lives in .py, so look for the decorator rather than the extension.
    pattern = re.compile(r"@\s*triton\.jit|tl\.program_id")
    return any(pattern.search(p.read_text("utf-8", errors="replace"))
               for p in Path(root).rglob("*.py") if "__pycache__" not in p.parts)


# -- runtime -----------------------------------------------------------------

def _resolve(path: str) -> Any:
    module = importlib.import_module(path.split(".")[0])
    for part in path.split(".")[1:]:
        module = getattr(module, part)
    return module


@contextlib.contextmanager
def recording() -> Iterator[list[str]]:
    """Patch the forbidden symbols to record and still execute.

    Recording rather than blocking, deliberately. A probe that raises tells you
    the candidate *touched* something; one that records tells you what it called
    and how often, and lets the run finish so a later call is seen too. It also
    means a candidate using a forbidden symbol in genuine plumbing — outside the
    compute path — produces a report a person can read rather than a crash.
    """
    log: list[str] = []
    restores: list[tuple[Any, str, Any]] = []
    for path, attr in FORBIDDEN_SYMBOLS:
        try:
            owner = _resolve(path)
            original = getattr(owner, attr)
        except (ImportError, AttributeError):
            continue

        def wrapper(*args: Any, _original: Any = original,
                    _name: str = f"{path}.{attr}", **kwargs: Any) -> Any:
            log.append(_name)
            return _original(*args, **kwargs)

        try:
            setattr(owner, attr, wrapper)
            restores.append((owner, attr, original))
        except (AttributeError, TypeError):
            # Some attributes on extension types are not settable. The static
            # scan and the launch counter still cover those.
            continue
    try:
        yield log
    finally:
        for owner, attr, original in reversed(restores):
            with contextlib.suppress(AttributeError, TypeError):
                setattr(owner, attr, original)


def probe(entry: Callable[..., Any], args: Sequence[Any], *,
          kwargs: dict[str, Any] | None = None, root: Path | None = None,
          launch_counter: Callable[[], int] | None = None,
          label: str = "") -> DispatchReport:
    """Run the candidate once with the forbidden symbols under observation."""
    imports = scan_imports(root) if root is not None else []
    device_source = scan_device_source(root) if root is not None else False
    before = launch_counter() if launch_counter else 0
    try:
        with recording() as log:
            entry(*args, **(kwargs or {}))
    except Exception as exc:  # noqa: BLE001 - a probe failure is not a verdict
        return DispatchReport(forbidden_imports=imports, probe_input=label,
                              has_device_source=device_source,
                              error=f"{type(exc).__name__}: {exc}")
    after = launch_counter() if launch_counter else 0
    return DispatchReport(forbidden_calls=list(log), forbidden_imports=imports,
                          custom_launches=after - before,
                          has_device_source=device_source, probe_input=label)


# ==========================================================================
# Exclusion: proving the megakernel is not here.
# ==========================================================================
"""Proving the megakernel is not here.

A *megakernel* is a single persistent GPU kernel that owns the whole model
iteration — one launch, an internal instruction schedule, no per-operator kernel
boundaries. It is a legitimate and interesting technique, and it is deliberately
out of scope: it is a different research question from "can an agent evolve a
serving system", it needs a schedule compiler and a virtual machine before it
computes anything, and it does not compose with the ordinary-kernel candidate
loop this harness is built around.

Out of scope is easy to say and hard to hold. A quarantine that is not
*checked* decays: a reference is left in a skill, an import survives in a
package, a prompt still routes an agent there, and the exclusion is fiction.

So this module scans, and the scan is a gate. Scope is set by ``roots`` — the
things that are actually shipped or importable — because the whole point of a
quarantine is that the excluded material may go on existing somewhere. What may
never happen is for it to be reachable.

The provenance manifests are deliberately exempt: they *name* excluded material,
which is how a reader learns it was excluded on purpose rather than lost. So is
any file carrying :data:`EXEMPT_MARKER`, including this one.

megakernel-exclusion-scope: this file documents the exclusion
"""
#: Terms that name the excluded technique. Matched case-insensitively on word
#: boundaries so ``megakernel``, ``mega_kernel`` and ``MegaKernel`` all hit, and
#: a word that merely contains them does not.
TERMS: tuple[str, ...] = (
    r"mega[\s_-]?kernel",
    r"megair",
    r"three[\s_-]?layer[\s_-]?mk",
    r"persistent[\s_-]?single[\s_-]?kernel",
    r"schedule[\s_-]?level[\s_-]?megakernel",
)

_PATTERN = re.compile("|".join(TERMS), re.IGNORECASE)

#: Files whose *job* is to record what was excluded. Naming the excluded thing is
#: their content, not a leak.
PROVENANCE_NAMES: frozenset[str] = frozenset({
    "provenance.yaml", "adaptation_manifest.json", "quarantine.md",
    "QUARANTINE.md", "exclusion_report.json", "ADAPTATION.md",
})

#: A file may also declare itself exempt by carrying this marker. The scanner
#: cannot exempt itself by name without exempting anything that happens to be
#: called ``exclusion.py``, and a module whose entire subject is the quarantine
#: has to be able to name what it quarantines. The marker is long and specific
#: precisely so it cannot end up in a file by accident.
EXEMPT_MARKER = "megakernel-exclusion-scope: this file documents the exclusion"

#: Extensions worth scanning. A weights file is not a routing decision.
SCANNED_SUFFIXES: frozenset[str] = frozenset({
    ".py", ".md", ".json", ".yaml", ".yml", ".toml", ".txt", ".cu", ".cuh",
    ".cpp", ".h", ".sh", ".cfg", ".ini",
})


@dataclasses.dataclass(frozen=True)
class ExclusionHit(Contract):
    """One place the excluded technique is reachable from."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "megakernel_exclusion_hit"

    path: str = ""
    line: int = 0
    text: str = ""
    surface: str = ""

    def validate(self) -> None:
        self.require(bool(self.path), "a hit needs a path")


@dataclasses.dataclass(frozen=True)
class ExclusionReport(Contract):
    """What was scanned, what was found, and what was deliberately exempt."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "megakernel_exclusion_report"

    report_id: str = ""
    roots: list[str] = dataclasses.field(default_factory=list)
    files_scanned: int = 0
    hits: list[ExclusionHit] = dataclasses.field(default_factory=list)
    exempt_files: list[str] = dataclasses.field(default_factory=list)
    #: Surfaces checked, named so a reader can see which were *not*.
    surfaces: list[str] = dataclasses.field(default_factory=list)

    def validate(self) -> None:
        self.require(bool(self.report_id), "an exclusion report needs an id")
        self.require(bool(self.surfaces),
                     "a report that does not say which surfaces it checked cannot be "
                     "read as evidence of anything")

    @property
    def clean(self) -> bool:
        return not self.hits

    def summary(self) -> str:
        if self.clean:
            return (f"{self.files_scanned} file(s) across {len(self.surfaces)} surface(s): "
                    f"no reachable megakernel path, import, route or package content "
                    f"({len(self.exempt_files)} provenance file(s) exempt, since naming "
                    f"the excluded material is their job)")
        by_surface: dict[str, int] = {}
        for hit in self.hits:
            by_surface[hit.surface] = by_surface.get(hit.surface, 0) + 1
        return (f"{len(self.hits)} reachable reference(s) in "
                f"{dict(sorted(by_surface.items()))}")


def _nested_repositories(root: Path) -> set[Path]:
    """Directories under ``root`` that are separate checkouts, not our source.

    A directory carrying its own ``.git`` is another project that happens to sit
    inside this tree: it is not tracked here, it is not in any wheel built from
    here, and it is not delivered to an agent workspace. Scanning it turns this
    gate into a report on whatever anyone has left in the working directory --
    which is how a gate starts failing for reasons unrelated to what it guards,
    and a gate that cries wolf is one people learn to override.

    A vendored copy that someone *committed* is a different thing and still fires:
    only the presence of a nested `.git` exempts a subtree, and adding one is not
    something that happens by accident.
    """
    return {marker.parent for marker in Path(root).rglob(".git")
            if marker.parent != Path(root)}


def _iter_files(root: Path) -> Iterable[Path]:
    nested = _nested_repositories(root)
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or ".git" in path.parts:
            continue
        if any(repository in path.parents for repository in nested):
            continue
        if path.suffix.lower() in SCANNED_SUFFIXES:
            yield path


def scan(roots: Sequence[tuple[str, Path]], report_id: str = "megakernel-exclusion",
         exempt: Iterable[str] = ()) -> ExclusionReport:
    """Scan each named surface for a reachable reference to the excluded technique.

    ``roots`` is a list of ``(surface_name, path)``: the surface is what a reader
    needs — "the package", "the skills", "the knowledge bundle" — because "three
    hits" means very different things depending on where they are.
    """
    exempt_names = PROVENANCE_NAMES | set(exempt)
    hits: list[ExclusionHit] = []
    exempted: list[str] = []
    scanned = 0

    for surface, root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for path in _iter_files(root):
            if path.name in exempt_names:
                exempted.append(str(path))
                continue
            try:
                text = path.read_text("utf-8", errors="replace")
            except OSError:  # pragma: no cover - unreadable file
                continue
            if EXEMPT_MARKER in text:
                exempted.append(str(path))
                continue
            scanned += 1
            if not _PATTERN.search(text):
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if _PATTERN.search(line):
                    hits.append(ExclusionHit(path=str(path), line=number,
                                             text=line.strip()[:200], surface=surface))

    return ExclusionReport(report_id=report_id, roots=[str(p) for _, p in roots],
                           files_scanned=scanned, hits=hits,
                           exempt_files=sorted(exempted),
                           surfaces=[s for s, _ in roots])


def gate(report: ExclusionReport, gate_id: str = "megakernel_exclusion") -> GateResult:
    """Turn a scan into a verdict.

    A hit is a ``FAIL`` and not a warning. The exclusion either holds or it does
    not, and a warning that nobody acts on is how it stops holding.
    """
    if report.clean:
        return GateResult.passed(gate_id, observed=report.files_scanned,
                                 message=report.summary())
    return GateResult.failed(
        gate_id, ReasonCode.ENVELOPE_NARROWED,
        f"megakernel material is reachable: {report.summary()}",
        observed=report.files_scanned, failures=len(report.hits),
        detail={"hits": [h.to_dict() for h in report.hits[:40]]})


def check_imports(module_names: Sequence[str]) -> GateResult:
    """Import each named module and confirm nothing megakernel-ish came with it.

    A static text scan cannot see a module that arrives through a registry lookup
    or a dynamic import. Actually importing and then reading ``sys.modules`` can.
    """
    import sys

    before = set(sys.modules)
    for name in module_names:
        __import__(name)
    arrived = sorted(set(sys.modules) - before)
    bad = [m for m in arrived if _PATTERN.search(m)]
    if bad:
        return GateResult.failed(
            "megakernel_import", ReasonCode.IMPORT_FORBIDDEN,
            f"importing {list(module_names)} pulled in {bad}",
            observed=len(arrived), failures=len(bad))
    return GateResult.passed(
        "megakernel_import", observed=len(arrived),
        message=f"importing {list(module_names)} pulled in {len(arrived)} module(s), "
                f"none of them megakernel")
