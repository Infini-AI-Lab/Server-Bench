"""A real, bounded GPU smoke: one kernel, all the way up the ladder.

Everything else in this package can be tested on a CPU with scripted outcomes,
and that is worth a great deal — but it cannot tell you whether the rungs work
against an actual compiler and an actual device. This module does exactly that,
once, cheaply, on one small operator, and it is the last gate before the harness
is pointed at real work.

The operator is RMS normalisation. Chosen because it is small enough to write
correctly in a few lines, bandwidth-bound so its timing is stable, and carries
one of the failure modes that matters most: a kernel that skips the
normalisation runs, is faster, and produces fluent, wrong output. That mutant is
part of the smoke, and the smoke fails if the checks do not kill it.

Nothing here is a benchmark result. It proves the machinery works.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from serving_evolve.contracts.gates import GateResult, ReasonCode

from .contracts.capability import (
    DType,
    KernelCapability,
    Layout,
    Operator,
    ServingPhase,
    ShapeAxis,
    ShapeEnvelope,
)
from .contracts.measure import MicrobenchmarkPlan
from .contracts.verification import Tolerance
from .verification import (
    BOUNDARIES,
    BUILD,
    CONTROLS,
    DETERMINISM,
    DISPATCH,
    INTEGRATION,
    MANIFEST,
    MICROBENCH,
    NUMERICS,
    PARITY,
    SERVING,
    LadderPlan,
    run_ladder,
)

# The candidate, the mutant, and the reference all live here as source strings so
# that the smoke has no external fixture to go stale.

CANDIDATE = '''
import torch, triton, triton.language as tl

@triton.jit
def _rms_norm(x_ptr, w_ptr, y_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    off = row * n_cols + cols
    x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
    scale = tl.rsqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + off, (x * scale * w).to(y_ptr.dtype.element_ty), mask=mask)

LAUNCHES = {"count": 0}

def run(x, weight, eps=1e-6):
    assert x.is_cuda and weight.is_cuda, "this kernel is device-only"
    assert x.dim() == 2, f"expected a 2-D activation, got {x.dim()}-D"
    rows, cols = x.shape
    y = torch.empty_like(x)
    block = triton.next_power_of_2(cols)
    LAUNCHES["count"] += 1
    _rms_norm[(rows,)](x, weight, y, cols, eps, BLOCK=block, num_warps=4)
    return y
'''

#: The same kernel with the normalisation removed. Runs, is faster, wrong.
MUTANT = CANDIDATE.replace(
    "    scale = tl.rsqrt(tl.sum(x * x, axis=0) / n_cols + eps)",
    "    scale = 1.0  # MUTANT: missing_normalization")

#: The reduction taken over the wrong thing: mean of x rather than of x squared.
WRONG_REDUCTION = CANDIDATE.replace(
    "    scale = tl.rsqrt(tl.sum(x * x, axis=0) / n_cols + eps)",
    "    scale = tl.rsqrt(tl.sum(x, axis=0) / n_cols + eps)  # MUTANT: wrong_reduction")

#: The last elements of every row left unwritten. Uninitialised memory is often
#: plausible and occasionally zero, which is why the check has to poison the
#: buffer rather than compare values against a fresh allocation.
PARTIAL_WRITE = CANDIDATE.replace(
    "    tl.store(y_ptr + off, (x * scale * w).to(y_ptr.dtype.element_ty), mask=mask)",
    "    short = cols < (n_cols - 8)  # MUTANT: partial_write\n"
    "    tl.store(y_ptr + off, (x * scale * w).to(y_ptr.dtype.element_ty), mask=short)")

#: A dispatcher: correct output, no kernel of its own. The defect no numerical
#: check can see.
FALLBACK_ONLY = '''
import torch
LAUNCHES = {"count": 0}

def run(x, weight, eps=1e-6):
    var = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    return (x.to(torch.float32) * torch.rsqrt(var + eps) * weight).to(x.dtype)
'''

REFERENCE = '''
import torch

def reference_run(x, weight, eps=1e-6):
    x32 = x.to(torch.float32)
    scale = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (x32 * scale * weight.to(torch.float32)).to(x.dtype)

def make_inputs(rows, cols, dtype, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(rows, cols, generator=g, device=device, dtype=dtype)
    w = torch.randn(cols, generator=g, device=device, dtype=dtype)
    return x, w
'''


def capability() -> KernelCapability:
    return KernelCapability(
        capability_id="smoke-rms-norm",
        operator=Operator.RMS_NORM,
        phase=ServingPhase.BOTH,
        envelope=ShapeEnvelope(
            axes=[ShapeAxis(name="rows", minimum=1, maximum=4096, alignment=1,
                            required_values=[1, 7, 256]),
                  ShapeAxis(name="cols", minimum=1, maximum=8192, alignment=128,
                            required_values=[1, 129, 1024, 2560])],
            dtypes=[DType.BF16, DType.FP32],
            layouts=[Layout.CONTIGUOUS]),
        architectures=["sm80", "sm86", "sm89", "sm90", "sm100"],
        unsupported=["non-2-D input", "CPU tensors", "cols beyond 8192"],
        fallback="torch-native")


def _write(root: Path, name: str, text: str) -> Path:
    path = Path(root) / name
    path.write_text(text, "utf-8")
    return path


def _load(path: Path, module_name: str) -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mismatch(candidate: Any, reference: Any, tol: Tolerance) -> tuple[float, float]:
    """Max relative error and the fraction of elements inside tolerance."""
    import torch

    a = candidate.to(torch.float32)
    b = reference.to(torch.float32)
    denom = b.abs().clamp_min(1e-6)
    rel = (a - b).abs() / denom
    inside = ((a - b).abs() <= (tol.atol + tol.rtol * b.abs())).float().mean().item()
    return rel.max().item(), inside


def run_smoke(out_dir: Path, *, device_index: int = 0,
              rows: int = 4096, cols: int = 2560) -> dict[str, Any]:
    """Run the ladder against a real Triton kernel on a real device.

    Returns the evidence document. The two device-heavy rungs at the top of the
    ladder — model integration and end-to-end serving — are *skipped with a
    reason*, because this smoke has no model and no server. A skip is recorded so
    that "every gate passed" is never read as a claim about a ladder that did not
    run.
    """
    import torch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    device = f"cuda:{device_index}"
    tol = Tolerance.for_dtype(DType.BF16)
    plan = LadderPlan(
        plan_id="kernel-smoke",
        rungs=[MANIFEST, BUILD, DISPATCH, PARITY, BOUNDARIES, NUMERICS, DETERMINISM,
               CONTROLS, MICROBENCH],
        skipped={
            INTEGRATION: "this smoke qualifies the harness against a real device, "
                         "not a model; there is no forward path to wire the kernel "
                         "into",
            SERVING: "no server is under test here; the rung that can accept a "
                     "candidate is deliberately not exercised by a smoke",
        })

    src = out_dir / "candidate"
    src.mkdir(exist_ok=True)
    impl = _write(src, "impl.py", CANDIDATE)
    mutant = _write(src, "impl_mutant.py", MUTANT)
    reduction = _write(src, "impl_wrong_reduction.py", WRONG_REDUCTION)
    partial = _write(src, "impl_partial_write.py", PARTIAL_WRITE)
    dispatcher = _write(src, "impl_dispatcher.py", FALLBACK_ONLY)
    ref_path = _write(src, "reference.py", REFERENCE)

    reference = _load(ref_path, "smoke_reference")
    state: dict[str, Any] = {}

    def manifest_rung(_ctx: dict[str, Any]) -> GateResult:
        from serving_evolve.hashing import digest_file

        digests = {p.name: digest_file(p) for p in
                   (impl, mutant, reduction, partial, dispatcher, ref_path)}
        state["digests"] = digests
        return GateResult.passed(MANIFEST, observed=len(digests),
                                 message=f"{len(digests)} source file(s) hashed",
                                 detail={"digests": digests})

    def build_rung(_ctx: dict[str, Any]) -> GateResult:
        module = _load(impl, "smoke_candidate")
        if not callable(getattr(module, "run", None)):
            return GateResult.failed(BUILD, ReasonCode.BUILD_FAILED,
                                     "impl.py exposes no callable run")
        state["module"] = module
        return GateResult.passed(BUILD, observed=1, message="imported impl.run")

    def dispatch_rung(_ctx: dict[str, Any]) -> GateResult:
        module = state["module"]
        x, w = reference.make_inputs(8, 256, torch.bfloat16, device)
        before = module.LAUNCHES["count"]
        module.run(x, w)
        launched = module.LAUNCHES["count"] - before
        if launched < 1:
            return GateResult.failed(
                DISPATCH, ReasonCode.HIDDEN_FALLBACK,
                "no kernel of the candidate's own was launched; the answer may be "
                "correct but the speed being reported is the library's")
        return GateResult.passed(DISPATCH, observed=1,
                                 message=f"{launched} custom kernel launch(es)")

    def parity_rung(_ctx: dict[str, Any]) -> GateResult:
        module = state["module"]
        x, w = reference.make_inputs(rows, cols, torch.bfloat16, device, seed=0)
        got = module.run(x, w)
        want = reference.reference_run(x, w)
        worst, inside = _mismatch(got, want, tol)
        if inside < tol.matched_ratio:
            return GateResult.failed(
                PARITY, ReasonCode.NUMERICAL_DIVERGENCE,
                f"only {inside:.4f} of elements are inside tolerance "
                f"(need {tol.matched_ratio}); worst relative error {worst:.3e}",
                observed=rows * cols, failures=1,
                detail={"matched_ratio": inside, "max_relative": worst})
        return GateResult.passed(PARITY, observed=rows * cols,
                                 message=f"matched {inside:.4f}, worst relative "
                                         f"{worst:.3e}",
                                 detail={"matched_ratio": inside, "max_relative": worst})

    def boundaries_rung(_ctx: dict[str, Any]) -> GateResult:
        module = state["module"]
        points = [(1, 1), (1, 129), (7, 1024), (256, 2560), (33, 8192)]
        failures: list[str] = []
        for r, c in points:
            for dtype in (torch.bfloat16, torch.float32):
                x, w = reference.make_inputs(r, c, dtype, device, seed=r + c)
                got = module.run(x, w)
                want = reference.reference_run(x, w)
                floor = Tolerance.for_dtype(
                    DType.BF16 if dtype is torch.bfloat16 else DType.FP32)
                _, inside = _mismatch(got, want, floor)
                if inside < floor.matched_ratio:
                    failures.append(f"({r},{c},{dtype}) matched {inside:.4f}")
        # inputs that must be refused rather than quietly computed
        refusals = 0
        for bad in (torch.randn(4, 4, 4, device=device, dtype=torch.bfloat16),
                    torch.randn(4, 8, dtype=torch.bfloat16)):
            try:
                module.run(bad, torch.randn(bad.shape[-1], device=device,
                                            dtype=torch.bfloat16))
            except (AssertionError, RuntimeError, ValueError):
                refusals += 1
        if refusals < 2:
            failures.append(f"only {refusals} of 2 unsupported inputs were refused")
        if failures:
            return GateResult.failed(
                BOUNDARIES, ReasonCode.NUMERICAL_DIVERGENCE,
                f"{len(failures)} boundary case(s) failed: {failures[:4]}",
                observed=len(points) * 2 + 2, failures=len(failures))
        return GateResult.passed(BOUNDARIES, observed=len(points) * 2 + 2,
                                 message=f"{len(points)} shape(s) x 2 dtype(s), plus "
                                         f"2 unsupported inputs correctly refused")

    def numerics_rung(_ctx: dict[str, Any]) -> GateResult:
        module = state["module"]
        x, w = reference.make_inputs(64, 2560, torch.bfloat16, device, seed=3)
        got = module.run(x, w)
        if not torch.isfinite(got.to(torch.float32)).all():
            return GateResult.failed(NUMERICS, ReasonCode.NUMERICAL_DIVERGENCE,
                                     "output contains NaN or Inf", failures=1)
        # A poisoned buffer catches a partial write, which comparing against a
        # fresh allocation often will not: uninitialised memory is frequently
        # plausible and occasionally zero. Fill the allocator's next block with
        # NaN, free it, and let the candidate allocate over it -- anything the
        # kernel fails to write comes back NaN.
        poison = torch.full((64, 2560), float("nan"), device=device,
                            dtype=torch.bfloat16)
        del poison
        got2 = module.run(x, w)
        if not torch.isfinite(got2.to(torch.float32)).all():
            return GateResult.failed(
                NUMERICS, ReasonCode.NUMERICAL_DIVERGENCE,
                "output over a poisoned buffer contains NaN: some elements were "
                "never written", failures=1)
        torch.cuda.synchronize(device)
        allocated = torch.cuda.memory_allocated(device)
        return GateResult.passed(NUMERICS, observed=1,
                                 message=f"finite output; {allocated} byte(s) still "
                                         f"allocated after the check",
                                 detail={"allocated_bytes": allocated})

    def determinism_rung(_ctx: dict[str, Any]) -> GateResult:
        module = state["module"]
        x, w = reference.make_inputs(512, 2560, torch.bfloat16, device, seed=5)
        first = module.run(x, w)
        second = module.run(x, w)
        if not torch.equal(first, second):
            return GateResult.failed(DETERMINISM, ReasonCode.NUMERICAL_DIVERGENCE,
                                     "two runs on identical input disagree",
                                     failures=1)
        return GateResult.passed(DETERMINISM, observed=2,
                                 message="two runs agree bit for bit")

    def controls_rung(_ctx: dict[str, Any]) -> GateResult:
        """The rung that checks the rungs: do the checks above kill known defects?"""
        from .verification import KillMatrix, applicable_to

        x, w = reference.make_inputs(rows, cols, torch.bfloat16, device, seed=0)
        want = reference.reference_run(x, w)

        killed: dict[str, str] = {}
        survived: list[str] = []

        for label, path, name in (("missing_normalization", mutant, "smoke_mutant"),
                                  ("wrong_reduction", reduction, "smoke_reduction")):
            broken = _load(path, name)
            _, inside = _mismatch(broken.run(x, w), want, tol)
            if inside < tol.matched_ratio:
                killed[label] = PARITY
            else:
                survived.append(label)

        # partial_write: poison the buffer the kernel will allocate over, then
        # look for what it failed to overwrite.
        partial_mod = _load(partial, "smoke_partial")
        poison = torch.full_like(x, float("nan"))
        del poison
        out = partial_mod.run(x, w)
        if not torch.isfinite(out.to(torch.float32)).all():
            killed["partial_write"] = NUMERICS
        else:
            _, partial_inside = _mismatch(out, want, tol)
            if partial_inside < tol.matched_ratio:
                killed["partial_write"] = PARITY
            else:
                survived.append("partial_write")

        wrapper = _load(dispatcher, "smoke_dispatcher")
        before = wrapper.LAUNCHES["count"]
        got = wrapper.run(x, w)
        _, wrapper_inside = _mismatch(got, want, tol)
        launched = wrapper.LAUNCHES["count"] - before
        if launched == 0:
            killed["silent_fallback"] = DISPATCH
        else:
            survived.append("silent_fallback")

        # Timing without synchronisation: if the protocol cannot detect its own
        # misuse, every number it produces is unfalsifiable.
        #
        # The workload has to be big enough that one launch is long compared with
        # the cost of enqueuing it. A small kernel launched in a loop drains as
        # fast as it fills, the unsynchronised and synchronised times agree, and
        # the control silently stops discriminating -- which is itself the failure
        # this whole rung exists to catch, so it is worth getting right.
        module = state["module"]
        x2, w2 = reference.make_inputs(32768, 8192, torch.bfloat16, device, seed=9)
        for _ in range(5):
            module.run(x2, w2)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        module.run(x2, w2)
        unsynced = time.perf_counter() - t0
        torch.cuda.synchronize(device)
        synced = time.perf_counter() - t0
        # Half is a wide margin on purpose: the claim is only that unsynchronised
        # timing is *obviously* wrong, not that it is wrong by a precise factor.
        if unsynced < synced * 0.5:
            killed["inaccurate_timing"] = MICROBENCH
        else:
            survived.append("inaccurate_timing")
        del x2, w2

        # A legitimate variant must NOT be rejected: a battery that kills
        # everything is not sensitive, it is broken.
        false_positives: list[str] = []
        _, honest = _mismatch(module.run(x, w), want, tol)
        if honest < tol.matched_ratio:
            false_positives.append("the unmodified candidate")

        required = sorted(m.value for m in applicable_to(Operator.RMS_NORM.value))
        matrix = KillMatrix(matrix_id="smoke-kill-matrix", required=required,
                            killed_by=killed, survived=survived,
                            false_positives=false_positives)
        state["kill_matrix"] = matrix
        detail = {"matrix": matrix.to_dict(),
                  "dispatcher_matched_ratio": wrapper_inside,
                  "unsynchronised_seconds": unsynced,
                  "synchronised_seconds": synced}
        if not matrix.qualified:
            return GateResult.failed(CONTROLS, ReasonCode.EVALUATOR_TAMPERED,
                                     matrix.explain(), observed=len(required),
                                     failures=len(survived) + len(false_positives)
                                     + len(matrix.missing_classes()),
                                     detail=detail)
        return GateResult.passed(CONTROLS, observed=len(required),
                                 message=matrix.explain(), detail=detail)

    def microbench_rung(_ctx: dict[str, Any]) -> GateResult:
        module = state["module"]
        mplan = MicrobenchmarkPlan(plan_id="smoke-timing", warmup_iterations=25,
                                   measured_iterations=100, repetitions=3)
        x, w = reference.make_inputs(4096, 2560, torch.bfloat16, device, seed=11)

        def timed(fn) -> list[float]:
            for _ in range(mplan.warmup_iterations):
                fn(x, w)
            torch.cuda.synchronize(device)
            samples = []
            for _ in range(mplan.repetitions):
                t0 = time.perf_counter()
                for _ in range(mplan.measured_iterations):
                    fn(x, w)
                torch.cuda.synchronize(device)
                samples.append((time.perf_counter() - t0) * 1e6
                               / mplan.measured_iterations)
            return samples

        # Interleaved, because drift over a minute is real and A-then-B attributes
        # all of it to B.
        cand = timed(module.run)
        ref = timed(reference.reference_run)
        cand += timed(module.run)
        ref += timed(reference.reference_run)
        speedup = statistics.median(ref) / statistics.median(cand)
        detail = {"candidate_us": cand, "reference_us": ref, "speedup": speedup,
                  "note": "a microbenchmark may reject a candidate and may never "
                          "accept one"}
        return GateResult.passed(
            MICROBENCH, observed=len(cand) + len(ref),
            message=f"candidate {statistics.median(cand):.1f} us, reference "
                    f"{statistics.median(ref):.1f} us, ratio {speedup:.2f}x",
            detail=detail, device_seconds=round(sum(cand + ref) / 1e6, 3))

    implementations = {
        MANIFEST: manifest_rung, BUILD: build_rung, DISPATCH: dispatch_rung,
        PARITY: parity_rung, BOUNDARIES: boundaries_rung, NUMERICS: numerics_rung,
        DETERMINISM: determinism_rung, CONTROLS: controls_rung,
        MICROBENCH: microbench_rung,
    }

    results = run_ladder(plan, implementations, {})

    from serving_evolve.contracts.gates import AcceptancePolicy, decide

    # Deliberately *not* plan.policy(). That policy keeps a skipped rung required,
    # which is right for a candidate -- skipping the end-to-end gate must never be
    # a route to acceptance. A smoke is qualifying the harness, not accepting a
    # candidate: it has no model and no server, so it declares a policy over the
    # rungs it actually ran and says so in the evidence.
    policy = AcceptancePolicy(
        policy_id="kernel-smoke-qualification",
        required_gates=[r for r in plan.rungs if r != MICROBENCH],
        advisory_gates=[MICROBENCH])
    decision = decide("smoke-candidate", policy, results)
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    leaked = torch.cuda.memory_allocated(device)

    evidence = {
        "schema_version": 1,
        "what_this_is": ("a bounded qualification of the kernel harness against a "
                         "real compiler and a real device. It is not a benchmark "
                         "result and makes no performance claim"),
        "plan": plan.to_dict(),
        "policy": policy.to_dict(),
        "policy_note": ("this is a harness qualification, not a candidate "
                        "acceptance: the two rungs that could accept a candidate "
                        "were skipped with a reason and are not in the required "
                        "set"),
        "capability": capability().to_dict(),
        "device": {"index": device_index,
                   "name": torch.cuda.get_device_name(device_index),
                   "capability": ".".join(
                       str(v) for v in torch.cuda.get_device_capability(device_index))},
        "results": [r.to_dict() for r in results],
        "decision": decision.to_dict(),
        "kill_matrix": state["kill_matrix"].to_dict() if "kill_matrix" in state else None,
        "wall_seconds": round(time.time() - started, 3),
        "device_bytes_still_allocated_after_teardown": leaked,
        "interpreter": sys.executable,
    }
    (out_dir / "evidence.json").write_text(
        json.dumps(evidence, indent=1, sort_keys=True) + "\n", "utf-8")
    return evidence


def summarize(evidence: dict[str, Any]) -> str:
    lines = [f"device   {evidence['device']['name']} "
             f"(sm_{evidence['device']['capability'].replace('.', '')})", ""]
    for result in evidence["results"]:
        status = result["status"]
        lines.append(f"  {status:<12} {result['gate_id']:<28}{result['message'][:80]}")
    lines += ["", f"decision {evidence['decision']['outcome']}: "
                  f"{evidence['decision']['reason'][:160]}",
              f"wall     {evidence['wall_seconds']}s",
              f"leaked   {evidence['device_bytes_still_allocated_after_teardown']} "
              f"byte(s) of device memory after teardown"]
    return "\n".join(lines)
