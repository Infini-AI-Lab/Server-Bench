"""The microbenchmark plan and its results, and the one rule about what it means.

A kernel microbenchmark can **reject** a candidate and can never **accept** one.
That is not conservatism, it is arithmetic: a kernel that is 3x faster on an
operator holding 4% of the critical path cannot move the end-to-end number by
more than 2.7%, and if the operator was not on the critical path at all it cannot
move it at all. The measurement that decides acceptance is the end-to-end one.

What a microbenchmark is genuinely good for is killing ideas cheaply — before an
hour of device time goes into integrating something that was never going to pay.

The timing protocol is the boring part that decides whether any of it is real:
warm up until the clocks settle, synchronise around the measured region and not
inside it, keep raw samples, and report a robust statistic rather than a mean
that one outlier owns.
"""

from __future__ import annotations

import dataclasses
import math
import statistics
from typing import ClassVar

from serving_evolve.contracts.base import Contract


@dataclasses.dataclass(frozen=True)
class MicrobenchmarkPlan(Contract):
    """How a kernel is timed, fixed before it is."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "kernel_microbenchmark_plan"

    plan_id: str = ""
    warmup_iterations: int = 25
    measured_iterations: int = 100
    #: Independent repetitions of the whole warm-up-and-measure cycle. One long
    #: run measures one thermal state; several short ones measure the device.
    repetitions: int = 3
    #: Explicit synchronisation around the measured region. Without it the number
    #: is the time to *enqueue* the kernel, which is smaller and meaningless.
    synchronize: bool = True
    #: Interleave candidate and reference within a repetition. Drift over minutes
    #: is real, and A-then-B attributes it entirely to B.
    interleave_with_reference: bool = True
    #: Below this fraction, the difference is not worth acting on regardless of
    #: how tight the interval is.
    practical_margin: float = 0.05

    def validate(self) -> None:
        self.require(bool(self.plan_id), "a microbenchmark plan needs an id")
        self.require(self.warmup_iterations >= 5,
                     f"{self.warmup_iterations} warm-up iterations is not enough for "
                     f"clocks and caches to settle; the first samples measure the ramp")
        self.require(self.measured_iterations >= 20,
                     f"{self.measured_iterations} measured iterations cannot support a "
                     f"robust statistic")
        self.require(self.repetitions >= 2,
                     "a single repetition measures one thermal state, not the device")
        self.require(self.synchronize,
                     "without synchronisation the measurement times kernel launch, not "
                     "kernel execution")
        self.require(0.0 < self.practical_margin < 1.0,
                     "the practical margin is a fraction in (0, 1)")

    @property
    def total_iterations(self) -> int:
        return (self.warmup_iterations + self.measured_iterations) * self.repetitions


@dataclasses.dataclass(frozen=True)
class TimingSample(Contract):
    """One repetition's raw samples. Raw, because a summary cannot be re-analysed."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "kernel_timing_sample"

    label: str = ""
    repetition: int = 0
    microseconds: list[float] = dataclasses.field(default_factory=list)

    def validate(self) -> None:
        self.require(bool(self.label), "a timing sample needs a label")
        self.require(bool(self.microseconds), "a timing sample with no measurement")
        bad = [t for t in self.microseconds if t <= 0 or not math.isfinite(t)]
        self.require(not bad, f"non-positive or non-finite timing(s): {bad[:5]}")

    @property
    def median(self) -> float:
        return statistics.median(self.microseconds)


@dataclasses.dataclass(frozen=True)
class MicrobenchmarkResult(Contract):
    """What the timing said, with enough detail to disbelieve it."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "kernel_microbenchmark_result"

    plan_id: str = ""
    candidate_samples: list[TimingSample] = dataclasses.field(default_factory=list)
    reference_samples: list[TimingSample] = dataclasses.field(default_factory=list)
    #: Peak workspace the candidate allocated. A kernel that is faster because it
    #: allocates 8 GB of scratch has not made the server faster.
    candidate_workspace_bytes: int = 0
    reference_workspace_bytes: int = 0
    device_name: str = ""
    device_uuid: str = ""
    #: True when candidate and reference were timed on the same device in the same
    #: session. A cross-device ratio is not a ratio.
    same_device: bool = False

    def validate(self) -> None:
        self.require(bool(self.plan_id), "a result needs its plan id")
        self.require(bool(self.candidate_samples) and bool(self.reference_samples),
                     "a ratio needs both sides measured")

    @property
    def candidate_median(self) -> float:
        return statistics.median([s.median for s in self.candidate_samples])

    @property
    def reference_median(self) -> float:
        return statistics.median([s.median for s in self.reference_samples])

    def speedup(self) -> float:
        """Reference time over candidate time. Greater than 1 means faster.

        Refuses to produce a number across devices, because that ratio measures
        the two machines rather than the two kernels.
        """
        if not self.same_device:
            raise ValueError(
                "a speedup across two devices is not a speedup; measure both sides on "
                "the same device in the same session")
        return self.reference_median / self.candidate_median

    def spread(self) -> float:
        """Interquartile spread of the candidate, relative to its median.

        A wide spread means the number is not stable enough to act on, whatever
        the median says.
        """
        values = sorted(t for s in self.candidate_samples for t in s.microseconds)
        if len(values) < 4:
            return float("inf")
        q1 = values[len(values) // 4]
        q3 = values[(3 * len(values)) // 4]
        return (q3 - q1) / statistics.median(values)
