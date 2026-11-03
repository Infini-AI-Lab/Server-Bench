"""Delivering a frozen catalog into a candidate workspace.

This is the seam between the two halves. Research happened outside, with the open
web and someone's judgement; construction happens inside, with a fixed input. The
catalog crosses that line as data — a machine-readable `kernels.json` and a
`KERNELS.md` the agent actually reads — and its digests go into the input
manifest, so what the agent was handed is auditable rather than asserted.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .contracts import EvidenceLevel, KernelCatalog
from .research import validate

DEST = "kernels"


def render(catalog: KernelCatalog) -> str:
    """The document the agent reads.

    Written for someone about to use these, so it leads with the caveat and the
    fallback rule rather than burying them under a table of names.
    """
    out: list[str] = [
        f"# Kernels available for {catalog.identity.model_id}", "",
        catalog.standing_caveat, "",
        "## Before you use any of these", "",
        "1. **Check it computes what this model needs.** A fused kernel written "
        "for a differently-shaped attention block is numerically *wrong*, not "
        "merely slower. Compare against the reference before you compare speed.",
        "2. **Check it runs here.** Several of these need a specific GPU "
        "generation. `requires` says which.",
        "3. **If it is unavailable, take the declared fallback and say so.** A "
        "request that silently took a different path makes every later "
        "measurement describe something other than the system under test.",
        "4. **Nothing here is measured.** These are pointers with provenance. "
        "Treating a vendor's claim as your result is how a benchmark becomes a "
        "press release.",
        "",
        f"Catalog `{catalog.catalog_id}` · architecture `{catalog.architecture}` · "
        f"target `{catalog.hardware_target}` · researched {catalog.researched_utc} "
        f"by {catalog.researched_by}", "",
        "## What this model needs computed", "",
        "| operator | why the model has it | typical share of a decode step |",
        "|---|---|---:|",
    ]
    for op in catalog.operators:
        share = ("—" if op.typical_step_share is None
                 else f"{op.typical_step_share:.0%}")
        flag = "" if op.required else " *(only in some configurations)*"
        out.append(f"| `{op.name}`{flag} | {op.rationale} | {share} |")

    out += ["", "Shares are typical for a decoder of this shape and are **not** "
            "measurements of this deployment. Profile before you believe them.",
            "", "## What is offered", ""]

    for choice in catalog.choices:
        out.append(f"### `{choice.operator}`")
        out.append("")
        if choice.rationale:
            out.append(choice.rationale)
            out.append("")
        for label, name in [("preferred", choice.preferred),
                            *[("alternative", a) for a in choice.alternatives]]:
            if not name:
                continue
            candidate = catalog.candidate(name)
            if candidate is None:
                continue
            level = EvidenceLevel(candidate.evidence_level).value
            out += [
                f"**{label}: {candidate.name}** — {candidate.project}, "
                f"{candidate.license}",
                "",
                f"- obtain: `{candidate.obtain}`",
                f"- requires: {candidate.requires or 'no stated requirement'}",
                f"- claim: {candidate.claim or 'none stated'}  \n"
                f"  *(evidence: {level}"
                + (f", {candidate.source}" if candidate.source else "")
                + (f", retrieved {candidate.retrieved}" if candidate.retrieved else "")
                + ")*",
            ]
            if candidate.notes:
                out.append(f"- note: {candidate.notes}")
            out.append("")
        out += [f"**Fallback:** {choice.fallback}", ""]

    out += ["## If none of this applies", "",
            "Write the operator yourself and gate it against the reference. A "
            "correct kernel you understand beats a fast one you cannot verify, and "
            "the harness will not accept the second one anyway.", ""]
    return "\n".join(out)


def install(catalog: KernelCatalog, workspace: os.PathLike | str, *,
            require_frozen: bool = True) -> dict[str, Any]:
    """Write the catalog into a workspace and record exactly what was written."""
    workspace = Path(workspace)
    if require_frozen and not catalog.frozen:
        raise ValueError(
            f"refusing to install unfrozen catalog {catalog.catalog_id!r}: an agent "
            f"that receives a catalog still being edited has an input nobody can "
            f"reproduce")
    result = validate(catalog)
    if not result["ok"]:
        raise ValueError(f"refusing to install an invalid catalog: {result['problems']}")

    dest = workspace / DEST
    dest.mkdir(parents=True, exist_ok=True)
    body = json.dumps(catalog.to_dict(), indent=1, sort_keys=True) + "\n"
    text = render(catalog)
    (dest / "kernels.json").write_text(body, "utf-8")
    (dest / "KERNELS.md").write_text(text, "utf-8")

    written = [(f"{DEST}/kernels.json", body.encode()),
               (f"{DEST}/KERNELS.md", text.encode())]
    files: list[dict[str, Any]] = []
    combined = hashlib.sha256()
    for rel, raw in written:
        sha = hashlib.sha256(raw).hexdigest()
        files.append({"path": rel, "bytes": len(raw), "sha256": sha})
        combined.update(rel.encode())
        combined.update(sha.encode())

    return {
        "catalog": catalog.catalog_id,
        "model": catalog.identity.model_id,
        "installed_at": DEST,
        "files": files,
        "install_sha256": combined.hexdigest(),
        "coverage": result["coverage"],
    }


def briefing_note(manifest: dict[str, Any]) -> str:
    """The sentence appended to the agent's brief. A pointer, not a summary."""
    coverage = manifest["coverage"]
    return (
        f"A researched kernel catalog for {manifest['model']} is installed at "
        f"`{manifest['installed_at']}/KERNELS.md`, covering "
        f"{coverage['covered']}/{coverage['required']} required operators. It was "
        f"assembled outside this environment and is a set of pointers with "
        f"provenance — nothing in it has been benchmarked. Read its caveats before "
        f"depending on anything it names, verify correctness against the reference "
        f"before speed, and if a kernel is unavailable take the declared fallback "
        f"and report that you did."
    )
