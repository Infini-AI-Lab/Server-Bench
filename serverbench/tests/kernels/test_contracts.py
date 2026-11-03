"""What the kernel contracts refuse, and why each refusal is worth having."""

from __future__ import annotations

import pytest
from kernel_evolve.contracts import (
    BoundaryMatrix,
    BuildManifest,
    CandidateState,
    DType,
    KernelBudget,
    KernelCapability,
    KernelHypothesis,
    KernelStrategy,
    KernelTaskSpec,
    Layout,
    MaskConvention,
    MicrobenchmarkPlan,
    MicrobenchmarkResult,
    Operator,
    PositionConvention,
    ReferenceImplementation,
    ServingPhase,
    ShapeAxis,
    ShapeEnvelope,
    SourceFile,
    TimingSample,
    Tolerance,
    VerificationPlan,
    check_transition,
)
from kernel_evolve.verification import default_controls
from serving_evolve.errors import ContractError, IllegalTransition


def envelope() -> ShapeEnvelope:
    return ShapeEnvelope(
        axes=[ShapeAxis(name="batch", minimum=1, maximum=512, alignment=1),
              ShapeAxis(name="seq", minimum=1, maximum=8192, alignment=128,
                        required_values=[1, 2048])],
        dtypes=[DType.BF16], layouts=[Layout.CONTIGUOUS, Layout.STRIDED])


def capability(**kw) -> KernelCapability:
    base: dict = {
        "capability_id": "cap-attn", "operator": Operator.ATTENTION_DECODE,
        "phase": ServingPhase.DECODE, "envelope": envelope(),
        "architectures": ["sm90"], "qk_norm": True, "gqa_group_size": 4,
        "mask": MaskConvention.CAUSAL, "positions": PositionConvention.ROPE_NEOX,
        "unsupported": ["head_dim not in {64,128}", "fp8 KV cache"],
        "fallback": "torch-native",
    }
    base.update(kw)
    return KernelCapability(**base)


# -- capability ---------------------------------------------------------------

def test_a_capability_must_say_what_it_does_not_support():
    with pytest.raises(ContractError, match="what it does not support"):
        capability(unsupported=[])


def test_a_capability_must_name_its_fallback():
    with pytest.raises(ContractError, match="undeclared fallback is a hidden fallback"):
        capability(fallback="")


def test_qk_normalisation_mismatch_is_reported_in_words():
    a = capability()
    b = capability(capability_id="cap-other", qk_norm=False)
    problems = a.disagreements_with(b)
    assert any("QK normalisation" in p for p in problems)
    assert any("silently drops" in p for p in problems)


def test_boundary_points_visit_each_axis_edge():
    points = envelope().boundary_points()
    seqs = {p["seq"] for p in points}
    assert 1 in seqs and 8192 in seqs and 2048 in seqs
    # every point holds the other axes at their minimum, so the count is linear
    assert len(points) <= sum(len(a.boundary_values()) for a in envelope().axes)


def test_an_axis_may_not_require_a_value_outside_its_range():
    with pytest.raises(ContractError, match="lie outside"):
        ShapeAxis(name="seq", minimum=1, maximum=16, required_values=[64])


# -- task ---------------------------------------------------------------------

def reference() -> ReferenceImplementation:
    return ReferenceImplementation(reference_id="ref-attn", module="refs.attention",
                                   provenance="the model card's attention definition")


def test_a_task_without_a_reference_is_refused():
    with pytest.raises(ContractError, match="'correct' is undefined"):
        KernelTaskSpec(task_id="t", capability=capability(), model_id="m",
                       architecture="sm90")


def test_a_task_on_an_unqualified_architecture_is_refused():
    with pytest.raises(ContractError, match="not qualified on sm100"):
        KernelTaskSpec(task_id="t", capability=capability(), reference=reference(),
                       model_id="m", architecture="sm100")


def test_provided_sota_withholds_the_reference_kernels():
    assert not KernelStrategy.PROVIDED_SOTA.may_mutate_kernel_source
    assert not KernelStrategy.PROVIDED_SOTA.may_read_reference_kernels
    assert KernelStrategy.EVOLVE.may_mutate_kernel_source


# -- candidate ----------------------------------------------------------------

def manifest(**kw) -> BuildManifest:
    base: dict = {
        "sources": [SourceFile(path="impl.cu", digest="a" * 64, language="cuda",
                               lines=200)],
        "entry_module": "impl",
    }
    base.update(kw)
    return BuildManifest(**base)


def test_a_fast_math_flag_is_refused():
    with pytest.raises(ContractError, match="not computing the same function"):
        manifest(flags=["-use_fast_math"])


def test_an_absolute_source_path_is_refused():
    with pytest.raises(ContractError, match="absolute"):
        SourceFile(path="/etc/passwd", digest="a" * 64)


def test_a_python_only_manifest_has_no_device_source():
    assert not manifest(sources=[SourceFile(path="impl.py", digest="b" * 64,
                                            language="python")]).has_device_source
    assert manifest().has_device_source


def test_a_hypothesis_needs_profile_evidence_and_a_falsifier():
    with pytest.raises(ContractError, match="profile evidence"):
        KernelHypothesis(statement="s", cheap_falsifier="f", rollback="r")
    with pytest.raises(ContractError, match="kill it cheaply"):
        KernelHypothesis(statement="s", profile_evidence="p", rollback="r")


def test_an_end_to_end_ceiling_above_the_local_speedup_is_refused():
    with pytest.raises(ContractError, match="arithmetically impossible"):
        KernelHypothesis(statement="s", profile_evidence="p", cheap_falsifier="f",
                         rollback="r", expected_local_speedup=0.10,
                         expected_end_to_end_ceiling=0.30)


def test_illegal_state_transitions_raise():
    check_transition(CandidateState.BUILT, CandidateState.VERIFYING)
    with pytest.raises(IllegalTransition):
        check_transition(CandidateState.REJECTED, CandidateState.BUILDING)


def test_an_invalid_candidate_may_be_retried():
    check_transition(CandidateState.INVALID, CandidateState.FROZEN)


# -- verification -------------------------------------------------------------

def test_a_tolerance_far_wider_than_the_dtype_floor_is_refused():
    with pytest.raises(ContractError, match="computing something else"):
        Tolerance(dtype=DType.BF16, rtol=0.5, atol=0.5)


def test_the_default_tolerance_tracks_the_dtype():
    assert Tolerance.for_dtype(DType.FP32).rtol < Tolerance.for_dtype(DType.BF16).rtol


def test_a_plan_missing_a_required_mutant_class_is_refused():
    boundaries = BoundaryMatrix(points=[{"batch": 1}], dtypes=[DType.BF16],
                                rejection_cases=["fp32 input"])
    controls = [c for c in default_controls() if c.mutant.value != "silent_fallback"]
    with pytest.raises(ContractError, match="silent_fallback"):
        VerificationPlan(plan_id="p", tolerance=Tolerance.for_dtype(DType.BF16),
                         boundaries=boundaries, controls=controls)


def test_a_boundary_matrix_must_include_inputs_to_refuse():
    with pytest.raises(ContractError, match="refuse"):
        BoundaryMatrix(points=[{"batch": 1}], dtypes=[DType.BF16])


# -- measurement --------------------------------------------------------------

def test_a_microbenchmark_plan_refuses_to_skip_synchronisation():
    with pytest.raises(ContractError, match="times kernel launch"):
        MicrobenchmarkPlan(plan_id="p", synchronize=False)


def test_a_single_repetition_is_refused():
    with pytest.raises(ContractError, match="one thermal state"):
        MicrobenchmarkPlan(plan_id="p", repetitions=1)


def test_a_cross_device_speedup_is_refused():
    result = MicrobenchmarkResult(
        plan_id="p", same_device=False,
        candidate_samples=[TimingSample(label="c", microseconds=[10.0, 11.0])],
        reference_samples=[TimingSample(label="r", microseconds=[20.0, 21.0])])
    with pytest.raises(ValueError, match="not a speedup"):
        result.speedup()


def test_a_same_device_speedup_is_the_ratio_of_medians():
    result = MicrobenchmarkResult(
        plan_id="p", same_device=True,
        candidate_samples=[TimingSample(label="c", microseconds=[10.0, 10.0, 10.0])],
        reference_samples=[TimingSample(label="r", microseconds=[20.0, 20.0, 20.0])])
    assert result.speedup() == pytest.approx(2.0)


# -- budget -------------------------------------------------------------------

def test_a_candidate_is_not_opened_without_room_to_close_it():
    budget = KernelBudget(budget_id="b", device_seconds=600,
                          cleanup_reserve_device_seconds=120)
    assert budget.may_open(800)
    assert not budget.may_open(700)
    assert "produce no verdict" in budget.refusal(700)
