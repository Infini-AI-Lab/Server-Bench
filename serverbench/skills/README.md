# Standard task skills

This tree is the reference-skill input for the `_standard_` task variants; the
`_instructed_` variants take theirs from `../skills_instructed/`. The lifecycle
(`harness/lifecycle/run.py::_guidance_defaults`) mounts each skill directory
individually under the workspace's `.claude/skills/`, and coding agents discover
them from each `SKILL.md`. This README is repository documentation and is not
delivered.

- `serving-system-evolution/` is reference material for building and debugging
  an LLM serving system; it prescribes no architecture.
- `kernel-evolution/` covers deciding whether a kernel is the right change, then
  writing, verifying and measuring one. Its `knowledge/` tree is the kernel
  knowledge corpus (hardware, profiling, per-operator notes, verification),
  indexed by `knowledge/catalog.yaml` and checked by
  `serverbench evolve kernels knowledge`.
- `ncu-report-skill/` is a git submodule (upstream Nsight Compute
  report-analysis skill), delivered as it is pinned.

`serverbench evolve skill <bundle>` validates a bundle before delivery.
