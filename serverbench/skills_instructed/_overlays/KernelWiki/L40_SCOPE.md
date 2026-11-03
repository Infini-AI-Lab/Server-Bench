# Scope note for the instructed serving tasks

This is the complete KernelWiki checkout pinned in `UPSTREAM_REVISION`.  Its
skill front matter correctly describes its primary scope as Hopper and
Blackwell. The instructed serving tasks run on NVIDIA L40 or L40S GPUs
(`sm_89`, Ada), so architecture-specific instructions for `sm_90` or `sm_100`
must not be copied onto this hardware.

Use the corpus as evidence for profiling methods, optimization reasoning,
source provenance, CUDA/PTX language patterns that also exist on `sm_89`, and
general kernel design.  Before adopting any instruction or intrinsic, confirm
its architecture support in the CUDA compiler and PTX ISA documentation and
compile for the task's actual `sm_89` target.  The task-specific
`kernel-evolution` and `ncu-report-skill` bundles are authoritative where their
L40 guidance conflicts with a Hopper/Blackwell page.
