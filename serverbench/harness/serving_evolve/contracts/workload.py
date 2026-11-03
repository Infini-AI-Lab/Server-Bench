"""What the system under evaluation is asked to do, and what must stay true.

:class:`WorkloadSpec` is the *operating envelope*: the frozen identity of the
model and its inputs, the shapes the system claims to support, the distribution
of traffic, the objective, and the semantic invariants a change may never break.
Freezing it is what makes later results comparable — and what makes it
impossible to manufacture a win by quietly serving an easier problem.

Two rules hold for every workload.

**Token counts are exact, never approximate.** "About 512 tokens" makes a
shared-prefix invariant unverifiable and lets prefix-cache behaviour differ
between systems for reasons that have nothing to do with the engine.

**Every suffix is distinct.** Identical suffixes let a system memoise whole
responses, and make prompt-hash special-casing undetectable.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import ClassVar

from .base import Contract


class Objective(str, enum.Enum):
    """What "better" means for this workload. One per workload, chosen up front."""

    CORRECTNESS_ONLY = "correctness_only"
    OUTPUT_TOKEN_THROUGHPUT = "output_token_throughput"
    SLO_QUALIFIED_GOODPUT = "slo_qualified_goodput"
    P99_LATENCY = "p99_latency"

    #: Output tokens delivered per dollar of GPU-hour, at a rate the task
    #: freezes. Maximised by keeping the card busy producing NEW tokens, which
    #: makes wasted work -- a re-prefill after an eviction -- visible in a way no
    #: latency metric does. The latency thresholds become gates rather than the
    #: score, so a system cannot buy throughput by missing them.
    TOKENS_PER_USD = "tokens_per_usd"

    #: TOTAL tokens -- input plus output -- over the scored window. On an
    #: agentic-replay corpus output is a small fraction of what the GPU moves,
    #: so an objective ignoring the input measures the cheapest part of the work.
    TOTAL_TOKENS = "total_tokens"

    #: REVENUE OVER COST: what the served work would sell for at the model's
    #: published API price, divided by what the GPU-hours cost. It can be below
    #: one, and that is the point: counting tokens says how much work a system
    #: moved, pricing them says whether moving it was worth doing. The prices
    #: are frozen with the task, never chosen by the measurer.
    DOLLARS_PER_DOLLAR = "dollars_per_dollar"


class SamplingMode(str, enum.Enum):
    """Only greedy makes an exact-token reference legitimate."""

    GREEDY = "greedy"
    SEEDED_SAMPLING = "seeded_sampling"


@dataclasses.dataclass(frozen=True)
class ModelIdentity(Contract):
    """Exactly which weights, tokenizer and template. A name alone is not identity.

    ``revision`` is required: model repositories move, and a comparison against
    "the current main" is a comparison against a moving target.
    """

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "model_identity"

    model_id: str = ""
    revision: str = ""
    dtype: str = "bfloat16"
    tokenizer_id: str = ""
    chat_template_sha256: str = ""

    def validate(self) -> None:
        self.require(bool(self.model_id), "a model identity needs a model id")
        self.require(bool(self.revision),
                     "a model identity needs a pinned revision; 'latest' is a moving target")


@dataclasses.dataclass(frozen=True)
class ShapeEnvelope(Contract):
    """The prompt, output and batch shapes the system claims to support.

    A shape inside the envelope must work. Narrowing the envelope after seeing
    results is a workload change, not a fix, and the acceptance policy treats it
    as one.
    """

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "shape_envelope"

    min_prompt_tokens: int = 1
    max_prompt_tokens: int = 2048
    min_output_tokens: int = 1
    max_output_tokens: int = 256
    max_batch_rows: int | None = 1
    #: `None` means THERE IS NO CEILING, and that is a real state rather than a
    #: missing value: when the arrival process replays recorded sessions, their
    #: own inter-turn delays decide how many requests are in flight. Writing `1`
    #: would declare a ceiling nothing enforces; a large number would invent one.
    max_concurrent_requests: int | None = 1
    shared_prefix_tokens: int | None = None
    supported_features: list[str] = dataclasses.field(default_factory=list)
    unsupported_features: list[str] = dataclasses.field(default_factory=list)

    def validate(self) -> None:
        self.require(self.min_prompt_tokens >= 1, "prompts have at least one token")
        self.require(self.max_prompt_tokens >= self.min_prompt_tokens, "prompt range inverted")
        self.require(self.max_output_tokens >= self.min_output_tokens, "output range inverted")
        if self.max_batch_rows is not None:
            self.require(self.max_batch_rows >= 1, "batch rows must be positive")
        if self.max_concurrent_requests is not None:
            self.require(self.max_concurrent_requests >= 1,
                         "a declared concurrency ceiling is at least one; use "
                         "None for a workload whose arrival process sets it")
        overlap = set(self.supported_features) & set(self.unsupported_features)
        self.require(not overlap,
                     f"feature(s) declared both supported and unsupported: {sorted(overlap)}")

    def contains(self, prompt_tokens: int, output_tokens: int) -> bool:
        return (self.min_prompt_tokens <= prompt_tokens <= self.max_prompt_tokens
                and self.min_output_tokens <= output_tokens <= self.max_output_tokens)


@dataclasses.dataclass(frozen=True)
class SemanticInvariant(Contract):
    """One thing that must remain true no matter how the system is rebuilt.

    ``hard`` invariants fail the candidate outright. A soft invariant is a
    diagnostic and may never, on its own, accept anything.
    """

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "semantic_invariant"

    name: str = ""
    statement: str = ""
    hard: bool = True
    #: A capability the candidate must expose for this invariant to be checkable
    #: at all. Teacher-forced agreement is the motivating case: it is the right
    #: way to separate arithmetic from cascade, and it needs a scoring endpoint
    #: that an OpenAI-shaped server does not have. An invariant naming a
    #: capability is reported NOT_RUN when the candidate lacks it, rather than
    #: silently passing.
    #:
    #: Such an invariant may not be ``hard``. A hard gate that cannot run against
    #: an ordinary candidate either blocks every candidate or, worse, gets quietly
    #: skipped -- and a hard invariant nothing checks is more dangerous than no
    #: invariant, because the task card claims a guarantee the gates do not give.
    requires_capability: str = ""

    def validate(self) -> None:
        self.require(bool(self.name) and bool(self.statement),
                     "an invariant needs a name and a statement")
        self.require(not (self.hard and self.requires_capability),
                     f"{self.name!r} is hard and requires the capability "
                     f"{self.requires_capability!r}. An invariant that cannot be "
                     f"checked against a candidate lacking that capability must "
                     f"not be hard: it would either block every candidate or be "
                     f"skipped, and a hard invariant nothing checks claims a "
                     f"guarantee the gates do not give")


@dataclasses.dataclass(frozen=True)
class WorkloadSpec(Contract):
    """The frozen operating envelope.

    Its digest is the workload's identity. If the digest changes, results from
    before and after are not comparable, and the harness refuses to treat them
    as if they were.
    """

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "workload_spec"

    workload_id: str = ""

    #: Optional in the type, required by validate(). A default factory here would
    #: build an invalid ModelIdentity and raise on any caller who omitted it, from
    #: inside a constructor they did not write.
    model: ModelIdentity | None = None
    #: `None` FOR A SCENARIO WORKLOAD, and that is the point rather than an
    #: omission. A `ShapeEnvelope` says what shapes a generated trace will offer.
    #: When the workload is a published corpus replayed by a locked AgentX
    #: scenario, the shapes are facts about the corpus -- read off it, not
    #: chosen -- and restating them here would be a second declaration that can
    #: drift from the data and that nothing enforces.
    envelope: ShapeEnvelope | None = None
    protocol: str = "openai_completions"
    sampling: SamplingMode = SamplingMode.GREEDY
    objective: Objective = Objective.CORRECTNESS_ONLY
    #: Objective thresholds. Empty for a correctness-only workload -- and empty is
    #: honest, where a placeholder would look like a target nobody agreed.
    slo: dict[str, float] = dataclasses.field(default_factory=dict)
    arrival_process: str = "closed_loop"
    prompt_length_distribution: str = ""
    output_length_distribution: str = ""
    stop_semantics: str = "max_tokens or explicit stop strings; no other truncation"
    invariants: list[SemanticInvariant] = dataclasses.field(default_factory=list)
    denominator_rule: str = ("every offered request stays in the denominator, including "
                             "failures, timeouts, cancellations and incomplete responses")
    frozen: bool = False

    def validate(self) -> None:
        self.require(bool(self.workload_id), "a workload needs an id")
        self.require(self.model is not None,
                     "a workload must name the model it is about")
        if self.objective is not Objective.CORRECTNESS_ONLY and self.frozen:
            self.require(bool(self.slo),
                         "a frozen performance workload must state its SLO; an unstated "
                         "threshold is one that can be chosen after seeing results")
        if self.sampling is SamplingMode.SEEDED_SAMPLING:
            self.require(
                any("seed" in i.statement.lower() for i in self.invariants),
                "seeded sampling needs an invariant pinning the seed, or the reference "
                "is not reproducible")

    @property
    def identity(self) -> ModelIdentity:
        """The model, non-optional.

        ``validate()`` guarantees it is present; this is where that guarantee is
        expressed to a reader and to a type checker, instead of at every one of a
        dozen call sites.
        """
        assert self.model is not None      # guaranteed by validate()
        return self.model

    @property
    def hard_invariants(self) -> tuple[SemanticInvariant, ...]:
        return tuple(i for i in self.invariants if i.hard)

    def freeze(self) -> WorkloadSpec:
        """Return the frozen form. Freezing is one-way within a control epoch."""
        return self.evolve(frozen=True)
