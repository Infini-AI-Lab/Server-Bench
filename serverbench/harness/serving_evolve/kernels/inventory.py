"""What operators a model architecture actually needs.

"Search for the state of the art kernel for every kernel in the model" needs a
list of what those are, and getting it from the architecture rather than from
memory is what stops the list quietly omitting the one that matters.

The tables below are deliberately small and explicit. An architecture that is not
listed returns the generic decoder set with a flag saying it was generic, so a
caller can tell "this is the Qwen3 operator set" from "this is what any decoder
has, and nobody checked".

The step shares are *typical* figures for a decoder of this shape, useful for
ordering work. They are not measurements of any deployment, and
:mod:`..evaluator` will not let them be used as one.
"""

from __future__ import annotations

import dataclasses

from .contracts import OperatorRole, OperatorSpec


@dataclasses.dataclass(frozen=True)
class Inventory:
    """What an architecture needs, and whether anyone actually checked.

    The distinction between a recognised architecture and the generic fallback
    is load-bearing enough to deserve a typed field rather than a dict key.
    """

    architecture: str
    resolved: str | None
    operators: list[OperatorSpec]
    note: str = ""

    @property
    def recognised(self) -> bool:
        return self.resolved is not None

    def to_dict(self) -> dict:
        return {"architecture": self.architecture, "resolved": self.resolved,
                "recognised": self.recognised, "note": self.note,
                "operators": [o.to_dict() for o in self.operators]}


#: Present in essentially every autoregressive decoder.
_GENERIC: tuple[OperatorSpec, ...] = (
    OperatorSpec(name="attention_prefill", role=OperatorRole.ATTENTION_PREFILL,
                 rationale="a prompt's whole history is attended at once; this is "
                           "compute-bound and dominates time-to-first-token",
                 typical_step_share=None),
    OperatorSpec(name="attention_decode", role=OperatorRole.ATTENTION_DECODE,
                 rationale="one query row attends the whole cached history; this is "
                           "memory-bound and dominates steady-state decode",
                 typical_step_share=0.30),
    OperatorSpec(name="kv_cache_append", role=OperatorRole.KV_CACHE_APPEND,
                 rationale="each step writes one key and value per layer per row; "
                           "if this scales with history, something repacks",
                 typical_step_share=0.03),
    OperatorSpec(name="qkv_projection", role=OperatorRole.LINEAR,
                 rationale="the largest dense GEMM per layer at small batch",
                 typical_step_share=0.18),
    OperatorSpec(name="output_projection", role=OperatorRole.LINEAR,
                 rationale="attention output back to model width",
                 typical_step_share=0.08),
    OperatorSpec(name="mlp", role=OperatorRole.ACTIVATION_MLP,
                 rationale="the widest GEMMs in the model, and the usual first "
                           "target once attention is handled",
                 typical_step_share=0.25),
    OperatorSpec(name="normalisation", role=OperatorRole.NORMALISATION,
                 rationale="cheap per call and called constantly; worth fusing "
                           "with its neighbours rather than optimising alone",
                 typical_step_share=0.05),
    OperatorSpec(name="position_embedding", role=OperatorRole.POSITION_EMBEDDING,
                 rationale="applied to queries and keys every step; a per-row "
                           "offset, and the classic off-by-one in a first slice",
                 typical_step_share=0.03),
    OperatorSpec(name="sampling", role=OperatorRole.SAMPLING,
                 rationale="logits to a token; small, but a synchronisation point "
                           "if the host reads the result",
                 typical_step_share=0.04),
    OperatorSpec(name="lm_head", role=OperatorRole.LINEAR,
                 rationale="model width to vocabulary; large for a big vocabulary "
                           "and often the single widest GEMM at batch one",
                 typical_step_share=0.04),
)

#: Roles that only some architectures have.
_QK_NORM = OperatorSpec(
    name="qk_normalisation", role=OperatorRole.NORMALISATION,
    rationale="Qwen3 normalises queries and keys inside attention, which Qwen2 did "
              "not. A kernel written for a Qwen2-shaped attention block will be "
              "numerically wrong here, not merely slower",
    typical_step_share=0.02)

_PAGED = OperatorSpec(
    name="paged_attention", role=OperatorRole.PAGED_ATTENTION,
    rationale="attention over a block table rather than a contiguous slot; needed "
              "only if the cache is paged, and a different kernel from contiguous "
              "decode attention",
    typical_step_share=None, required=False)

_MOE = OperatorSpec(
    name="moe_routing", role=OperatorRole.MOE_ROUTING,
    rationale="router plus grouped expert GEMMs; the dominant cost in a sparse "
              "model and absent from a dense one",
    typical_step_share=None, required=False)

_QUANT = OperatorSpec(
    name="quantised_linear", role=OperatorRole.QUANTISED_LINEAR,
    rationale="mixed-precision GEMM for quantised weights; a different kernel "
              "family from the dense one, with its own numerics to verify",
    typical_step_share=None, required=False)

_COLLECTIVE = OperatorSpec(
    name="collective", role=OperatorRole.COLLECTIVE,
    rationale="all-reduce or all-gather across shards; only present under tensor "
              "or expert parallelism, and often the critical path when it is",
    typical_step_share=None, required=False)

#: Architecture -> the operators it adds beyond the generic set.
_EXTRA: dict[str, tuple[OperatorSpec, ...]] = {
    "qwen3": (_QK_NORM, _PAGED, _QUANT, _COLLECTIVE),
    "qwen3_moe": (_QK_NORM, _MOE, _PAGED, _QUANT, _COLLECTIVE),
    "llama": (_PAGED, _QUANT, _COLLECTIVE),
    "mixtral": (_MOE, _PAGED, _QUANT, _COLLECTIVE),
}

#: Aliases so a caller can pass a model id rather than an architecture.
_ALIASES: dict[str, str] = {
    "qwen/qwen3": "qwen3",
    "qwen3": "qwen3",
    "qwen3-moe": "qwen3_moe",
    "llama": "llama",
    "mistral": "llama",
    "mixtral": "mixtral",
}


def normalise(architecture: str) -> str | None:
    """Resolve a model id or architecture name to a known architecture."""
    key = architecture.strip().lower()
    if key in _EXTRA:
        return key
    for prefix, arch in sorted(_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if key.startswith(prefix) or prefix in key:
            return arch
    return None


def operators_for(architecture: str) -> Inventory:
    """The operator set for an architecture.

    Returns the operators plus whether the architecture was recognised. The
    caller needs to know the difference: an unrecognised architecture gets the
    generic decoder set, which is a reasonable starting point and is *not* the
    same as having checked.
    """
    resolved = normalise(architecture)
    extra = _EXTRA.get(resolved or "", ())
    return Inventory(
        architecture=architecture,
        resolved=resolved,
        operators=list(_GENERIC) + list(extra),
        note=("" if resolved else
              f"{architecture!r} is not a known architecture, so this is the "
              f"generic decoder set. Check what it adds -- a fused attention "
              f"kernel written for a different block shape is numerically "
              f"wrong, not merely slower."))


def known_architectures() -> list[str]:
    return sorted(_EXTRA)
