"""A starting point for the component that owns KV memory.

Static slots, because that is the layout to start from. What is worth copying is
not the layout: it is the reserve/commit/rollback shape, the deferred free, and
the refusal to hand a freed slot back before every reader has retired.

Swap the backend for paged blocks when a measurement says fragmentation is your
limit -- and not before, or you will have changed two things at once.
"""

from __future__ import annotations

import dataclasses


class CapacityError(RuntimeError):
    """No room. A clean refusal, never a truncation and never a crash."""


@dataclasses.dataclass
class Allocation:
    request_id: str
    slot: int
    capacity: int
    used: int = 0
    generation: int = 0


class CacheOwner:
    """The single writer of cache memory. Everyone else asks."""

    def __init__(self, slots: int, slot_capacity: int) -> None:
        self.slot_capacity = slot_capacity
        self._free: list[int] = list(range(slots))
        self._live: dict[str, Allocation] = {}
        self._generation: dict[int, int] = dict.fromkeys(range(slots), 0)
        # Slots whose request is gone but which an in-flight step may still be
        # reading. They become free only when that step retires.
        self._draining: dict[int, int] = {}
        self._reserved: dict[str, int] = {}

    # -- capacity questions, asked by the planner ----------------------------

    def fits(self, request_id: str, tokens: int) -> bool:
        live = self._live.get(request_id)
        if live is not None:
            return live.used + tokens <= live.capacity
        return bool(self._free) and tokens <= self.slot_capacity

    # -- reserve / commit / roll back ----------------------------------------

    def reserve(self, request_id: str) -> int:
        """Claim a slot for a plan that is still being built."""
        if request_id in self._live or request_id in self._reserved:
            raise ValueError(f"{request_id} already holds a slot")
        if not self._free:
            raise CapacityError("no free slot")
        slot = self._free.pop(0)
        self._reserved[request_id] = slot
        return slot

    def commit(self, request_id: str) -> Allocation:
        slot = self._reserved.pop(request_id)
        allocation = Allocation(request_id=request_id, slot=slot,
                                capacity=self.slot_capacity,
                                generation=self._generation[slot])
        self._live[request_id] = allocation
        return allocation

    def rollback(self, request_id: str) -> None:
        """A plan can fail after several requests were reserved. Without this you
        leak a slot on every failed schedule -- a leak that appears only under
        load."""
        slot = self._reserved.pop(request_id, None)
        if slot is not None:
            self._free.append(slot)
            self._free.sort()

    # -- growth ---------------------------------------------------------------

    def extend(self, request_id: str, tokens: int) -> Allocation:
        allocation = self._live[request_id]
        if allocation.used + tokens > allocation.capacity:
            raise CapacityError(f"{request_id} exceeds its slot")
        allocation.used += tokens
        return allocation

    # -- release --------------------------------------------------------------

    def release(self, request_id: str, *, in_flight_step: int | None = None) -> None:
        """Idempotent. A cancel that arrives twice is normal under load."""
        allocation = self._live.pop(request_id, None)
        if allocation is None:
            self.rollback(request_id)
            return
        self._generation[allocation.slot] += 1   # invalidate any stale reference
        if in_flight_step is None:
            self._free.append(allocation.slot)
            self._free.sort()
        else:
            # Deferred: a step may still be reading this slot. Handing it out now
            # gives the next occupant the previous one's history, which produces
            # fluent, plausible, wrong text under load.
            self._draining[allocation.slot] = in_flight_step

    def retire_step(self, step: int) -> list[int]:
        """Call when a step has fully retired. Returns the slots it releases."""
        released = [s for s, owner in self._draining.items() if owner <= step]
        for slot in released:
            del self._draining[slot]
            self._free.append(slot)
        self._free.sort()
        return released

    # -- introspection --------------------------------------------------------

    @property
    def utilisation(self) -> float:
        total = len(self._free) + len(self._live) + len(self._draining) + len(self._reserved)
        return 0.0 if total == 0 else 1.0 - len(self._free) / total
