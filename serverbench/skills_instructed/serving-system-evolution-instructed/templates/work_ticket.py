"""A starting point for the asynchronous work ticket.

The ticket exists so overlap is *checkable*: there is an interval where a ticket
is device-pending and real host work is happening. A future wrapped around a
synchronous call never shows that interval, and that is the test.
"""

from __future__ import annotations

import dataclasses
import enum
import threading


class TicketState(str, enum.Enum):
    QUEUED = "queued"
    DISPATCHING = "dispatching"
    DEVICE_PENDING = "device_pending"
    COMPLETED = "completed"
    RETIRED = "retired"
    CANCELLED = "cancelled"
    FAILED = "failed"
    QUARANTINED = "quarantined"

    @property
    def terminal(self) -> bool:
        return self in (TicketState.RETIRED, TicketState.CANCELLED,
                        TicketState.FAILED, TicketState.QUARANTINED)


# The only legal moves. Failing closed on anything else is how a "completed"
# ticket that never ran gets caught instead of believed.
LEGAL: dict[TicketState, frozenset[TicketState]] = {
    TicketState.QUEUED: frozenset({TicketState.DISPATCHING, TicketState.CANCELLED,
                                   TicketState.FAILED}),
    TicketState.DISPATCHING: frozenset({TicketState.DEVICE_PENDING, TicketState.FAILED,
                                        TicketState.QUARANTINED}),
    TicketState.DEVICE_PENDING: frozenset({TicketState.COMPLETED, TicketState.FAILED,
                                           TicketState.QUARANTINED}),
    TicketState.COMPLETED: frozenset({TicketState.RETIRED, TicketState.QUARANTINED}),
    TicketState.RETIRED: frozenset(),
    TicketState.CANCELLED: frozenset(),
    TicketState.FAILED: frozenset(),
    TicketState.QUARANTINED: frozenset(),
}


@dataclasses.dataclass
class WorkTicket:
    ticket_id: str
    step: int
    rows: int
    state: TicketState = TicketState.QUEUED
    error: str = ""
    result: object | None = None
    _done: threading.Event = dataclasses.field(default_factory=threading.Event, repr=False)

    def move_to(self, state: TicketState, *, error: str = "") -> None:
        if state not in LEGAL[self.state]:
            raise RuntimeError(f"{self.ticket_id}: {self.state.value} -> {state.value} "
                               f"is not a legal move")
        if state is TicketState.FAILED and not error:
            raise ValueError("a failed ticket must record why")
        self.state = state
        self.error = error
        if state.terminal or state is TicketState.COMPLETED:
            self._done.set()

    def cancel(self) -> bool:
        """Idempotent by contract. A request that goes away mid-flight must not
        leave its row or its cache half-freed."""
        if self.state.terminal:
            return False
        if TicketState.CANCELLED in LEGAL[self.state]:
            self.move_to(TicketState.CANCELLED)
            return True
        # Already on the device: it will finish, and the result is discarded at
        # retirement rather than pretended away here.
        return False

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)

    @property
    def pending(self) -> bool:
        """True exactly while the device holds it -- the window overlap needs."""
        return self.state is TicketState.DEVICE_PENDING
