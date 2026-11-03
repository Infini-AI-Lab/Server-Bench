# Instructed task skills

This tree is the complete reference-skill input for the explicit
`_instructed_` task variants.  It is selected by the lifecycle from the full
task ID and is never the default for `_standard_` tasks.

- `serving-system-evolution-instructed/` contains the detailed serving
  architecture, correctness, profiling, evolution loop, vLLM/SGLang design
  notes, exact Qwen3.5/Gemma 4 notes, debugging, and template material.
- `kernel-evolution/` carries the complete kernel decision, construction,
  verification, measurement, and substitution material, with an L40/Ada
  CUDA/PTX build chapter.
- `ncu-report-skill/` retains the full upstream report-analysis bundle and
  adds an L40-specific top-level workflow for Nsight Systems and Nsight
  Compute.
- `KernelWiki/` is a clean git submodule pinned by the parent repository's
  gitlink. `_overlays/KernelWiki/` contains the parent-owned skill-metadata fix,
  recorded upstream revision, and L40 scope note. The lifecycle applies that
  overlay only to the materialized agent copy, leaving the submodule clean.

The lifecycle mounts the four skill directories individually under
`.claude/skills/`. Coding agents discover them from each `SKILL.md` and may
choose any, several, or none. This README is repository documentation and is
not an additional agent skill.
