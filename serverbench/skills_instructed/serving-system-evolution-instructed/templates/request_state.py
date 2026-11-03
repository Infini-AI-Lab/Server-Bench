"""A starting point for request state. Rewrite it; do not import it.

Deliberately boring: a frozen dataclass, an explicit phase, and terminal states
that are terminal. The two things worth copying are the *offered/admitted* split
and the fact that a terminal phase must record why.
"""

from __future__ import annotations

import dataclasses
import enum


class Phase(str, enum.Enum):
    QUEUED = "queued"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    PREEMPTED = "preempted"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"
    REJECTED = "rejected"

    @property
    def terminal(self) -> bool:
        return self in (Phase.FINISHED, Phase.CANCELLED, Phase.FAILED, Phase.REJECTED)


class EndReason(str, enum.Enum):
    STOP = "stop"
    LENGTH = "length"
    CANCELLED = "cancelled"
    ERROR = "error"
    REJECTED = "rejected"


@dataclasses.dataclass(frozen=True)
class RequestState:
    request_id: str
    prompt_tokens: int
    max_tokens: int
    phase: Phase = Phase.QUEUED
    generated_tokens: int = 0

    # Offered is when the client asked. Admitted is when we let it in. Measuring
    # time-to-first-token from `admitted` hides queueing entirely, which is
    # exactly the delay a scheduling change is supposed to affect.
    offered_at: float = 0.0
    admitted_at: float | None = None
    first_token_at: float | None = None
    finished_at: float | None = None

    end_reason: EndReason | None = None
    cache_handle: str | None = None

    def __post_init__(self) -> None:
        if self.phase.terminal and self.end_reason is None:
            raise ValueError(f"{self.request_id} is {self.phase.value} without a reason")
        if not self.phase.terminal and self.end_reason is not None:
            raise ValueError(f"{self.request_id} is not terminal but carries a reason")

    @property
    def ttft(self) -> float | None:
        """From offered, never from admitted."""
        return None if self.first_token_at is None else self.first_token_at - self.offered_at

    @property
    def queue_delay(self) -> float | None:
        return None if self.admitted_at is None else self.admitted_at - self.offered_at

    @property
    def tokens_remaining(self) -> int:
        return max(0, self.max_tokens - self.generated_tokens)

    def advance(self, token_at: float) -> RequestState:
        if self.phase.terminal:
            raise ValueError(f"{self.request_id} already ended as {self.phase.value}")
        return dataclasses.replace(
            self,
            phase=Phase.DECODING,
            generated_tokens=self.generated_tokens + 1,
            first_token_at=self.first_token_at if self.first_token_at is not None else token_at,
        )

    def end(self, reason: EndReason, at: float) -> RequestState:
        phase = {
            EndReason.CANCELLED: Phase.CANCELLED,
            EndReason.ERROR: Phase.FAILED,
            EndReason.REJECTED: Phase.REJECTED,
        }.get(reason, Phase.FINISHED)
        if self.phase.terminal:
            # Idempotent: a cancel that races a natural finish is normal under
            # load, not an error.
            return self
        return dataclasses.replace(self, phase=phase, end_reason=reason, finished_at=at)
