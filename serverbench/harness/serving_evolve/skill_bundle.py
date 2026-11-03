"""Validate a skill bundle before it is delivered to an agent.

A bundle is a router (`SKILL.md`) plus references and a `provenance.yaml`. The
checks: required files present, the entry point small enough to stay a router,
every reference resolvable and reachable from it, no copied upstream serving
source, no local or retired paths, and complete provenance. Source exclusion is
conservative: it can flag prose that quotes an identifier, which a reviewer
resolves by looking; it cannot miss a copied source file.

    serverbench evolve skill serverbench/skills/serving-system-evolution
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

#: A router, not a manual. Past this the bundle is loading things nobody asked for.
MAX_SKILL_MD_BYTES = 24_000

#: Markers of copied implementation rather than distilled prose. Prose about a
#: concept does not contain a decorator, a class definition, or an import of a
#: framework's private module.
SOURCE_MARKERS: Sequence[str] = (
    "from vllm", "import vllm", "from sglang", "import sglang",
    "import tensorrt_llm", "from tensorrt_llm", "import sgl_kernel",
    "@torch.compile", "class LLMEngine", "class Scheduler(", "class ModelRunner(",
    "class BlockManager", "class KVCacheManager", "SamplingParams(",
    "AsyncLLMEngine", "vllm.core", "sglang.srt", "tensorrt_llm.runtime",
)

#: Paths that must never appear: a local checkout is not a stable reference, and
#: a retired directory is a dangling one.
FORBIDDEN_PATHS: Sequence[str] = ("third_party/",)

#: An absolute path into somebody's account. Written as a pattern rather than a
#: literal so that this module does not itself contain the string it forbids --
#: a scan of a built distribution would otherwise flag its own detector. It also
#: covers macOS and root, which a `/home/` literal would miss.
ABSOLUTE_HOME = re.compile(r"/(home|Users|root)/[A-Za-z0-9._-]+/")

REQUIRED_FILES = ("SKILL.md", "provenance.yaml")

_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
_BARE = re.compile(r"`(references/[^`]+|templates/[^`]+)`")


def _iter_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts)


def _referenced_paths(text: str) -> list[str]:
    out = list(_LINK.findall(text))
    out += _BARE.findall(text)
    return [p for p in out
            if not p.startswith(("http://", "https://", "#", "mailto:"))]


def validate_bundle(path: os.PathLike | str) -> dict[str, Any]:
    """Run every bundle check and report each one separately."""
    root = Path(path)
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    if not root.is_dir():
        return {"ok": False, "path": str(root),
                "checks": [{"check": "bundle exists", "ok": False,
                            "detail": f"{root} is not a directory"}]}

    files = _iter_files(root)
    names = {str(p.relative_to(root)).replace(os.sep, "/") for p in files}

    missing = [f for f in REQUIRED_FILES if f not in names]
    check("required files", not missing,
          "SKILL.md and provenance.yaml present" if not missing else f"missing {missing}")

    skill_md = root / "SKILL.md"
    if skill_md.exists():
        size = skill_md.stat().st_size
        check("entry point is a router", size <= MAX_SKILL_MD_BYTES,
              f"SKILL.md is {size} bytes (limit {MAX_SKILL_MD_BYTES}; past this it "
              f"stops being progressive disclosure)")

        text = skill_md.read_text("utf-8")
        refs = _referenced_paths(text)
        unresolved = [r for r in refs if not (root / r).exists()]
        check("references resolve", not unresolved,
              f"{len(refs)} reference(s) checked"
              + ("" if not unresolved else f"; unresolved: {unresolved}"))
    else:
        refs = []

    # Every reference file should be reachable from the entry point, or it is
    # material the agent will never be routed to.
    reference_files = sorted(n for n in names
                             if n.startswith(("references/", "templates/")))
    if skill_md.exists():
        text = skill_md.read_text("utf-8")
        unreachable = [n for n in reference_files if n not in text]
        check("every reference is reachable", not unreachable,
              f"{len(reference_files)} file(s) under references/ and templates/"
              + ("" if not unreachable
                 else f"; never routed to: {unreachable}"))

    # Source exclusion.
    offenders: list[dict[str, str]] = []
    for p in files:
        if p.suffix in (".png", ".jpg", ".pdf", ".zip"):
            continue
        try:
            body = p.read_text("utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rel = str(p.relative_to(root)).replace(os.sep, "/")
        if rel == "provenance.yaml":
            continue        # provenance names upstream on purpose
        for marker in SOURCE_MARKERS:
            if marker in body:
                offenders.append({"file": rel, "marker": marker})
    check("no upstream source", not offenders,
          "no implementation-level marker found" if not offenders
          else f"{len(offenders)} marker(s): {offenders[:5]}")

    # Forbidden paths.
    path_hits: list[dict[str, str]] = []
    for p in files:
        try:
            body = p.read_text("utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rel = str(p.relative_to(root)).replace(os.sep, "/")
        if rel == "provenance.yaml":
            continue
        for bad in FORBIDDEN_PATHS:
            if bad in body:
                path_hits.append({"file": rel, "path": bad})
        found = ABSOLUTE_HOME.search(body)
        if found:
            path_hits.append({"file": rel, "path": "an absolute account path"})
    check("no local or retired paths", not path_hits,
          "no checkout, phase directory or absolute path referenced"
          if not path_hits else f"{path_hits[:5]}")

    # Provenance completeness.
    prov = root / "provenance.yaml"
    if prov.exists():
        body = prov.read_text("utf-8")
        needed = ("repository", "commit", "license", "retrieved", "files_inspected")
        absent = [k for k in needed if k not in body]
        check("provenance is complete", not absent,
              "repository, commit, licence, retrieval date and inspected files recorded"
              if not absent else f"missing key(s): {absent}")

    ok = all(c["ok"] for c in checks)
    return {"ok": ok, "path": str(root), "files": len(files),
            "checks": checks,
            "note": ("source exclusion is conservative: it can flag prose that quotes an "
                     "identifier, which a reviewer resolves by looking. It cannot miss a "
                     "copied source file, which is the direction that matters.")}


