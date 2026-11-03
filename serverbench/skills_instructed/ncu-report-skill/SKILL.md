---
name: ncu-report-skill
description: Profile an LLM serving system on NVIDIA L40 with Nsight Systems, then diagnose a profile-selected CUDA kernel with Nsight Compute. Use for CPU/GPU timelines, launch gaps, synchronization, kernel filtering, NCU reports, or evidence-backed optimization plans.
---

# Nsight profiling for the instructed L40 tasks

This is optional profiling help, not the serving iteration/publication loop.
Use cheap supplied correctness/replay or focused microbenchmark feedback and
publish asynchronous checkpoints frequently; do not wait for a profiler chain
or an elaborate evidence dossier. Keep all scripts/results under /workspace/agentx.
The public protocol command checks protocol, native token IDs and declared
context handling; it does not check output correctness, which is verified
privately after the run. Build your own correctness checks.
When doing full 1800-second AgentX, skip redundant lightweight replay; actual
replay fidelity is included. Both gates remain mandatory for final acceptance.

This instructed bundle includes the full former upstream skill tree for
reference, but its original top-level workflow targeted B200 / `sm_100`.
These tasks run on Ada L40 / `sm_89`. This file and
[`references/00-l40-serving-workflow.md`](references/00-l40-serving-workflow.md)
override every B200-specific target, metric name, hardware budget, and
programming recommendation elsewhere in the copied tree.

The task brief and delivered evaluator are authoritative. Profilers explain
performance; they do not measure an unperturbed verdict. Only the complete
1,800-second AgentX `--measure-only` replay measures TTFT/TPOT and the task's
objective, `dollars_per_dollar`. `--check-protocol-context` is separate.

## Workflow

1. Start with Nsight Systems on a bounded, representative serving interval.
   Trace CUDA, NVTX, and OS runtime activity with CPU sampling disabled. Mark
   admission, scheduling, batch construction, prefill, decode, sampling, output
   retirement, and cache operations with NVTX.
2. Read the timeline and CUDA/NVTX summaries. Name the actual critical path:
   host scheduling, synchronization, launch gaps, copies, prefill, decode,
   communication, or a cumulative kernel family.
3. Move to Nsight Compute only for a kernel that the system trace showed to
   matter. Use the real dtype/layout/shape/cache state in a deterministic
   harness, compile with line information, and filter to one or a few warmed
   launches.
4. Diagnose from evidence: launch geometry and occupancy; compute and memory
   throughput; register/shared-memory limits; memory transactions; scheduler
   and warp-stall data; source/SASS correlation. Query the L40's available
   metric names rather than copying the bundled B200 metric table.
5. State one falsifiable optimization hypothesis and its Amdahl ceiling. Verify
   independent numerical parity and boundaries, then use an interleaved
   microbenchmark only to reject weak candidates.
6. Re-profile after integration. Run the public correctness gate separately.
   Choose a full AgentX window whenever workload feedback is useful; cheap
   tests are not prerequisites, and none of these steps blocks checkpointing.

## Collection shapes

Nsight Systems:

    nsys profile --trace=cuda,nvtx,osrt --sample=none \
      --force-overwrite=true --output=<unique-prefix> <command>

    nsys stats --report cuda_api_sum --report cuda_gpu_kern_sum \
      --report cuda_kern_exec_sum --report nvtx_sum <report>.nsys-rep

Nsight Compute, overview followed by a focused full collection:

    ncu --set basic --target-processes all \
      --kernel-name 'regex:<kernel-pattern>' --launch-skip <warmups> \
      --launch-count 1 --export <unique-basic-report> <command>

    ncu --set full --target-processes all \
      --kernel-name 'regex:<kernel-pattern>' --launch-skip <warmups> \
      --launch-count 1 --export <unique-full-report> <command>

    ncu --import <report>.ncu-rep --page details

Use unique output paths and verify process, device, kernel, input shape, and
candidate identity before reading a report. Nsight Compute may replay a kernel
many times and substantially perturb execution. Never use its duration as a
service latency.

## Full copied tree

The former `reference/`, `helpers/`, and Blackwell companion files remain in
this instructed bundle so none of the previous skill was discarded. Their
report-parsing and organizational ideas may still be useful. Any file that
states B200, Blackwell, `sm_100`, or a B200-only metric is non-authoritative for
these tasks. The L40 workflow above and the current tool's enumerated metrics
take precedence.

## Primary sources

- NVIDIA Nsight Systems User Guide:
  https://docs.nvidia.com/nsight-systems/UserGuide/
- NVIDIA Nsight Compute documentation and CLI manual:
  https://docs.nvidia.com/nsight-compute/
- NVIDIA CUDA Programming and Best Practices guides:
  https://docs.nvidia.com/cuda/cuda-programming-guide/
  and https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
