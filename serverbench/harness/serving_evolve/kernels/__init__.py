"""Kernel research, catalogs, and delivering them to a sandboxed agent.

The division of labour is the whole point. Finding out what the state of the art
is — reading papers and release notes, judging what is credible — happens in the
harness, outside the sandbox, where there is a web and a reader. The agent inside
receives a frozen, hashed catalog and builds a server with it.

    serverbench evolve kernels inventory  --architecture qwen3
    serverbench evolve kernels brief      --model Qwen/Qwen3-0.6B --architecture qwen3
    serverbench evolve kernels validate   <catalog.json>
    serverbench evolve kernels provision  <catalog> --workspace <dir>

A catalog entry is a pointer with provenance, never a measurement. Nothing in any
catalog here has been benchmarked in this repository, and the contracts refuse to
let a catalog claim otherwise.
"""

from __future__ import annotations

from .contracts import (
    EvidenceLevel,
    KernelCandidate,
    KernelCatalog,
    KernelChoice,
    OperatorRole,
    OperatorSpec,
)
from .inventory import known_architectures, operators_for
from .provision import briefing_note, install, render
from .research import brief, freeze, read, validate, write

__all__ = [
    "EvidenceLevel", "KernelCandidate", "KernelCatalog", "KernelChoice",
    "OperatorRole", "OperatorSpec", "brief", "briefing_note", "freeze",
    "install", "known_architectures", "operators_for", "read", "render",
    "validate", "write",
]
