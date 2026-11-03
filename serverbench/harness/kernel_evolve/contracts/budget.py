"""Budget and timeout for one kernel candidate.

The reservation rule: a candidate is not opened unless there is enough device
time left to *finish* it -- to reject it or qualify it -- with the cleanup
reserve still intact. Opening a candidate that the budget cannot close produces
the worst outcome available: device time spent, no verdict, and no reserve left
to tidy up.
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar

from serving_evolve.contracts.base import Contract


@dataclasses.dataclass(frozen=True)
class KernelBudget(Contract):
    """What one kernel candidate is allowed to cost, and what must survive it."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "kernel_budget"

    budget_id: str = ""
    build_timeout_seconds: float = 900.0
    verify_timeout_seconds: float = 1800.0
    measure_timeout_seconds: float = 1800.0
    #: Device seconds this candidate may consume across every rung.
    device_seconds: float = 900.0
    #: Device seconds that must remain unspent after it, for evidence and teardown.
    cleanup_reserve_device_seconds: float = 120.0
    max_invalid_retries: int = 2

    def validate(self) -> None:
        self.require(bool(self.budget_id), "a budget needs an id")
        for name in ("build_timeout_seconds", "verify_timeout_seconds",
                     "measure_timeout_seconds", "device_seconds"):
            self.require(getattr(self, name) > 0, f"{name} must be positive")
        self.require(self.cleanup_reserve_device_seconds >= 0,
                     "the cleanup reserve cannot be negative")
        self.require(self.max_invalid_retries >= 0, "retries cannot be negative")

    def may_open(self, remaining_device_seconds: float) -> bool:
        """Whether there is room to run this candidate *to a verdict*."""
        return remaining_device_seconds >= (self.device_seconds
                                            + self.cleanup_reserve_device_seconds)

    def refusal(self, remaining_device_seconds: float) -> str:
        return (f"opening this candidate needs {self.device_seconds:.0f}s of device time "
                f"plus a {self.cleanup_reserve_device_seconds:.0f}s cleanup reserve, and "
                f"{remaining_device_seconds:.0f}s remain; a candidate opened here would "
                f"spend device time and produce no verdict")
