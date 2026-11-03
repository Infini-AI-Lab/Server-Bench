"""A starting point for the step plan and the planner that produces it.

The plan carries decisions and no memory: no tensors, no device pointers, no
request objects. That is what lets the planner be unit-tested with no device
present, which is where policy bugs are cheap to find.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence


@dataclasses.dataclass(frozen=True)
class StepPlan:
    step: int
    # request id -> tokens to process this step. A large count is a prefill, a
    # count of one is a decode, and a prefill that did not fit is a chunk. The
    # planner does not need to know which is which.
    tokens: dict[str, int] = dataclasses.field(default_factory=dict)
    preempted: tuple[str, ...] = ()
    budget: int = 0
    reason: str = ""

    def __post_init__(self) -> None:
        used = sum(self.tokens.values())
        if self.budget and used > self.budget:
            raise ValueError(f"plan uses {used} tokens against a budget of {self.budget}")
        overlap = set(self.tokens) & set(self.preempted)
        if overlap:
            raise ValueError(f"request(s) both scheduled and preempted: {sorted(overlap)}")

    @property
    def empty(self) -> bool:
        return not self.tokens

    @property
    def rows(self) -> int:
        return len(self.tokens)


@dataclasses.dataclass
class Planner:
    """One step's worth of policy. No memory, no device, no protocol."""

    budget: int
    # Reserve part of the budget for rows that are already generating, or a
    # burst of long prompts starves every in-flight request.
    decode_reserve: int = 0

    def plan(self, step: int, ready: Sequence[str], needed: dict[str, int],
             is_decode: dict[str, bool],
             fits: Callable[[str, int], bool]) -> StepPlan:
        """
        ``needed[r]``  how many more tokens request r must have processed
        ``fits(r, n)`` may r be given n tokens? Asked of the cache owner, which
                       is the only component that knows where anything lives.
        """
        remaining = self.budget
        decode_left = self.decode_reserve
        tokens: dict[str, int] = {}
        preempted: list[str] = []

        # Generating rows first: they are already holding memory, and finishing
        # them releases it.
        for request in sorted(ready, key=lambda r: (not is_decode.get(r, False), r)):
            if remaining <= 0:
                break
            want = min(needed.get(request, 0), remaining)
            if want <= 0:
                continue
            if not is_decode.get(request, False) and remaining - want < decode_left:
                want = max(0, remaining - decode_left)   # chunk it rather than starve decode
            if want <= 0:
                continue
            if not fits(request, want):
                preempted.append(request)
                continue
            tokens[request] = want
            remaining -= want
            if is_decode.get(request, False):
                decode_left = max(0, decode_left - want)

        return StepPlan(step=step, tokens=tokens, preempted=tuple(preempted),
                        budget=self.budget,
                        reason=f"{len(tokens)} rows, {self.budget - remaining} tokens")
