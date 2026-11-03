"""The outer research step: what to look for, and how to freeze what was found.

This module runs **outside the sandbox**, in the harness. It does not search the
web itself — it produces the brief that tells whoever is driving the harness
exactly what to search for, operator by operator, and it validates and freezes
what comes back.

Keeping the search out of the code is deliberate rather than a limitation. A
function that scraped a benchmark table would produce a number with no judgement
attached, at a moment nobody chose, from a source nobody read. Research needs a
reader. What the code can do is make sure the reader is asked the right questions
and that the answers arrive with their provenance intact — which is what
:func:`brief` and :func:`freeze` are for.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from ..contracts.workload import ModelIdentity
from ..errors import ContractError
from ..hashing import digest
from .contracts import EvidenceLevel, KernelCatalog
from .inventory import operators_for

#: The questions a candidate has to survive before it belongs in a catalog. They
#: are in this order because a later one is wasted effort if an earlier one fails.
QUESTIONS: tuple[str, ...] = (
    "Does it compute the same thing this model needs? A fused attention kernel "
    "written for a differently-shaped block is numerically wrong, not merely "
    "slower — Qwen3's query/key normalisation is exactly this trap.",
    "Does it run on the target hardware? Many of the fastest kernels need Hopper "
    "or Blackwell and simply do not build elsewhere.",
    "What is its licence, and is depending on it acceptable here?",
    "How is it obtained — a released package, or a repository at a commit? A "
    "kernel that has to be built from an unpinned branch is not a dependency.",
    "What exactly is claimed, by whom, measured how, against what baseline? "
    "'2x faster' with no envelope is not a claim.",
    "What is the fallback when it is unavailable, and does taking that fallback "
    "get reported rather than hidden?",
)


def brief(model: ModelIdentity, *, architecture: str = "",
          hardware_target: str = "") -> dict[str, Any]:
    """What to research, operator by operator.

    Hand this to whoever is driving the harness. It is the outer half of the
    work: the sandboxed agent never sees it, and never needs to.
    """
    inventory = operators_for(architecture or model.model_id)
    operators = inventory.operators
    return {
        "model": model.to_dict(),
        "architecture": inventory.architecture,
        "recognised_architecture": inventory.recognised,
        "hardware_target": hardware_target,
        "note": inventory.note,
        "questions_per_candidate": list(QUESTIONS),
        "operators": [
            {
                "name": op.name,
                "role": op.role.value if hasattr(op.role, "value") else str(op.role),
                "required": op.required,
                "why_the_model_has_it": op.rationale,
                "typical_step_share": op.typical_step_share,
                "search_for": (
                    f"the fastest correct implementation of {op.name} for "
                    f"{model.model_id} on {hardware_target or 'the target hardware'}, "
                    f"with a licence, a pinned way to obtain it, and a claim that "
                    f"names its envelope"),
            }
            for op in operators
        ],
        "rules": [
            "Record where every claim came from. An unsourced claim is a rumour.",
            "MEASURED_HERE is not available to a catalog. Nothing in it has been "
            "benchmarked in this repository, and saying otherwise would turn "
            "research into a result.",
            "Do not vendor a kernel's source into this repository. Record how to "
            "obtain it; the environment resolves it.",
            "Every operator needs a declared fallback. An undeclared one is the "
            "hidden-fallback defect the gates exist to catch.",
            "Correctness first: a kernel that is faster and different is not a "
            "candidate at all.",
        ],
    }


def validate(catalog: KernelCatalog) -> dict[str, Any]:
    """Check a catalog is fit to hand to an agent, and say what is wrong if not."""
    problems: list[str] = []

    coverage = catalog.coverage()
    if coverage["uncovered"]:
        problems.append(f"required operators with no choice: {coverage['uncovered']}")

    for candidate in catalog.candidates:
        if candidate.claim and not candidate.source:
            problems.append(f"{candidate.name}: a claim with no source")
        if not candidate.license:
            problems.append(f"{candidate.name}: no licence recorded")
        if EvidenceLevel(candidate.evidence_level) is EvidenceLevel.UNVERIFIED \
                and candidate.claim:
            problems.append(
                f"{candidate.name}: a claim at UNVERIFIED; either source it or drop it")

    for choice in catalog.choices:
        if not choice.fallback:
            problems.append(f"{choice.operator}: no declared fallback")

    orphans = [c.name for c in catalog.candidates
               if not any(c.name in [ch.preferred, *ch.alternatives]
                          for ch in catalog.choices)]
    if orphans:
        problems.append(f"candidates nothing selects: {orphans}")

    return {"ok": not problems, "problems": problems, "coverage": coverage}


def freeze(catalog: KernelCatalog, *, researched_by: str) -> KernelCatalog:
    """Freeze a validated catalog. Refuses an invalid one."""
    result = validate(catalog)
    if not result["ok"]:
        raise ContractError(
            "refusing to freeze an incomplete catalog: " + "; ".join(result["problems"]),
            problems=result["problems"])
    return catalog.evolve(
        frozen=True, researched_by=researched_by,
        researched_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def write(path: os.PathLike | str, catalog: KernelCatalog) -> dict[str, str]:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = catalog.to_dict()
    p.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n", "utf-8")
    return {"path": str(p), "catalog_digest": digest(body, semantic=True)}


def read(path: os.PathLike | str) -> KernelCatalog:
    return KernelCatalog.from_dict(json.loads(Path(path).read_text("utf-8")))
